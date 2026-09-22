import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { DISPLAY_HEADROOM_GB, HEADLESS_HEADROOM_GB, singlePlacement, tensorPlacement, type GpuFacts } from "../src/hardware/budget.ts";
import { PROFILE_VERSION, type MachineProfile } from "../src/hardware/profile.ts";
import type { SizingConfig } from "../src/sizing/search.ts";
import type { UnslothClient } from "../src/api/client.ts";
import {
  calibrationEntry,
  foldCalibration,
  measureLoaded,
  readCalibration,
  touchesDisplay,
  usageAboveIdle,
  type CalibrationStamp,
} from "../src/sizing/verify.ts";

const STAMP: CalibrationStamp = { unslothVersion: "2026.9.7", backend: "rocm" };

const TWO_IDENTICAL: GpuFacts[] = [
  { index: 0, totalGb: 23.98, idleUsedGb: 1.21, display: true, headroomGb: DISPLAY_HEADROOM_GB },
  { index: 1, totalGb: 23.98, idleUsedGb: 0.03, display: false, headroomGb: HEADLESS_HEADROOM_GB },
];

function profile(calibration: Record<string, unknown> | undefined): MachineProfile {
  return { version: PROFILE_VERSION, gpus: [], ...(calibration ? { calibration } : {}) };
}

describe("readCalibration", () => {
  it("reads a delta measured against this server", () => {
    const held = readCalibration(profile({ estimateDeltaGiB: 0.7, samples: 2, unslothVersion: "2026.9.7", backend: "rocm" }), STAMP);
    assert.equal(held?.deltaGb, 0.7);
    assert.equal(held?.samples, 2);
  });

  it("drops it when the server's version has moved — compute_bytes moves with llama.cpp", () => {
    const stale = profile({ estimateDeltaGiB: 0.7, samples: 2, unslothVersion: "2026.9.6", backend: "rocm" });
    assert.equal(readCalibration(stale, STAMP), undefined);
    // …and when the server stopped reporting a version at all.
    assert.equal(readCalibration(stale, { unslothVersion: undefined, backend: "rocm" }), undefined);
  });

  it("drops it when it was measured on another backend", () => {
    const other = profile({ estimateDeltaGiB: 0.7, unslothVersion: "2026.9.7", backend: "cuda" });
    assert.equal(readCalibration(other, STAMP), undefined);
  });

  it("drops an entry with no stamp rather than assuming it still holds", () => {
    assert.equal(readCalibration(profile({ estimateDeltaGiB: 0.7, samples: 1 }), STAMP), undefined);
  });

  it("has no opinion when nothing has been measured", () => {
    assert.equal(readCalibration(profile(undefined), STAMP), undefined);
    assert.equal(readCalibration(undefined, STAMP), undefined);
    // A hand-edited figure of the wrong type is not a measurement.
    assert.equal(readCalibration(profile({ estimateDeltaGiB: "lots" }), STAMP), undefined);
  });
});

describe("foldCalibration", () => {
  it("is the measurement itself the first time", () => {
    const first = foldCalibration(undefined, 0.8, STAMP);
    assert.equal(first.deltaGb, 0.8);
    assert.equal(first.samples, 1);
  });

  it("averages rather than believing the most recent accident", () => {
    const second = foldCalibration({ deltaGb: 1.0, samples: 1, stamp: STAMP }, 0.0, STAMP);
    assert.equal(second.deltaGb, 0.5);
    assert.equal(second.samples, 2);
  });

  it("keeps moving when the truth does, however old the machine is", () => {
    let held = foldCalibration(undefined, 0, STAMP);
    for (let sample = 0; sample < 30; sample++) held = foldCalibration(held, 4, STAMP);
    assert.ok(held.samples <= 8);
    assert.ok(held.deltaGb > 3, `settled at ${held.deltaGb}`);
  });

  it("carries the stamp of the server it was measured against", () => {
    const folded = foldCalibration({ deltaGb: 1, samples: 1, stamp: STAMP }, 1, { unslothVersion: "2026.10.0", backend: "rocm" });
    assert.equal(folded.stamp.unslothVersion, "2026.10.0");
  });
});

describe("calibrationEntry", () => {
  it("writes a figure someone could read, and what it is evidence about", () => {
    const entry = calibrationEntry({ deltaGb: 0.666666, samples: 3, stamp: STAMP }, undefined);
    assert.equal(entry["estimateDeltaGiB"], 0.67);
    assert.equal(entry["samples"], 3);
    assert.equal(entry["unslothVersion"], "2026.9.7");
    assert.equal(entry["backend"], "rocm");
  });

  it("keeps a hand-added key in the calibration block", () => {
    const entry = calibrationEntry({ deltaGb: 1, samples: 1, stamp: STAMP }, { note: "measured by hand" });
    assert.equal(entry["note"], "measured by hand");
  });
});

describe("usageAboveIdle", () => {
  const idle = [
    { index: 0, usedGb: 1.6, totalGb: 24 },
    { index: 1, usedGb: 0.03, totalGb: 24 },
  ];

  it("counts what the model holds, not what the desktop does", () => {
    const live = [
      { index: 0, usedGb: 21.1, totalGb: 24 },
      { index: 1, usedGb: 18.4, totalGb: 24 },
    ];
    assert.equal(usageAboveIdle(live, idle).toFixed(2), "37.87");
  });

  it("never credits a model with negative memory when the baseline was too high", () => {
    assert.equal(usageAboveIdle([{ index: 0, usedGb: 0.5, totalGb: 24 }], idle), 0);
  });

  it("counts a card it has no baseline for in full, rather than skipping it", () => {
    assert.equal(usageAboveIdle([{ index: 7, usedGb: 3, totalGb: 24 }], idle), 3);
  });

  it("bills the desktop to the model when there is no baseline at all", () => {
    // Which is why the caller refuses to measure without one: this is the
    // reference box's 5.9 GiB model reading as 7.8 GiB.
    const live = [
      { index: 0, usedGb: 1.88, totalGb: 24 },
      { index: 1, usedGb: 5.92, totalGb: 24 },
    ];
    assert.equal(usageAboveIdle(live, []).toFixed(1), "7.8");
    assert.equal(usageAboveIdle(live, idle).toFixed(1), "6.2");
  });
});

describe("measureLoaded", () => {
  const estimate = {
    available: true, reason: undefined,
    weightsBytes: 0, kvBytes: 0, computeBytes: 0,
    drafterRuntimeBytes: 0, drafterRuntimeGpuBytes: 0, projectorRuntimeBytes: 0,
    totalBytes: 6 * 1024 ** 3, gpuBytes: 6 * 1024 ** 3,
    kvEstimable: true, kvOnGpu: true, drafterKvUnsized: false, adaptersUnsized: false,
    moeOffloadUnmodelled: false, nCtx: 40960, cacheTypeKv: "q8_0", nParallel: 1, layerCount: 36,
  };
  const idle = [
    { index: 0, usedGb: 1.89, totalGb: 24 },
    { index: 1, usedGb: 0.03, totalGb: 24 },
  ];

  /** A client whose `/api/system` answers from a script of readings. */
  function scripted(readings: number[]): { client: UnslothClient; asked: () => number } {
    let at = 0;
    const client = {
      async get() {
        const used = readings[Math.min(at++, readings.length - 1)] ?? 0;
        return {
          gpu: {
            available: true,
            backend: "rocm",
            devices: [
              { index: 0, memory_total_gb: 24, vram_used_gb: 1.89, shared_memory: false, unified_memory: false },
              { index: 1, memory_total_gb: 24, vram_used_gb: used, shared_memory: false, unified_memory: false },
            ],
          },
        };
      },
    } as unknown as UnslothClient;
    return { client, asked: () => at };
  }

  const fast = { pollMs: 1, windowMs: 400 };

  /**
   * Hold the event loop open for the duration.
   *
   * Every timer in `src/` is `unref`'d — the extension must never keep Pi
   * alive on its own — so a bare `await` on a polling loop lets node exit out
   * from under the test. In the product the TUI is what holds the loop.
   */
  async function whileRunning<T>(work: Promise<T>): Promise<T> {
    const keepAlive = setInterval(() => {}, 5);
    try {
      return await work;
    } finally {
      clearInterval(keepAlive);
    }
  }

  it("waits for the memory to appear — a ready load has not allocated yet", async () => {
    // Observed on the reference box: POST /load answered at 3.6 s with
    // `phase: ready`, and both cards still at idle; the 5.92 GiB arrived four
    // seconds later. Two quick samples measured a model that cost nothing.
    const { client } = scripted([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 5.92]);
    const result = await whileRunning(measureLoaded(client, estimate, idle, fast));
    assert.equal(result?.measuredGb.toFixed(2), "5.89");
    assert.ok((result?.deltaGb ?? 0) < 0, "the estimator charged a little more than the cards held");
  });

  it("keeps the highest figure once it has stopped climbing", async () => {
    const { client } = scripted([0.03, 2.0, 4.5, 5.92, 5.9, 5.91, 5.92, 5.92, 5.92, 5.92]);
    const result = await whileRunning(measureLoaded(client, estimate, idle, fast));
    assert.equal(result?.measuredGb.toFixed(2), "5.89");
  });

  it("reports nothing rather than a model that costs nothing", async () => {
    const { client } = scripted([0.03]);
    assert.equal(await whileRunning(measureLoaded(client, estimate, idle, fast)), undefined);
  });

  it("stops asking once the figure has settled, rather than polling to the deadline", async () => {
    const { client, asked } = scripted([5.92]);
    await whileRunning(measureLoaded(client, estimate, idle, { pollMs: 1, windowMs: 5_000 }));
    assert.ok(asked() < 20, `asked ${asked()} times`);
  });
});

describe("touchesDisplay", () => {
  const config = (gpuIds: number[], tensor = false): SizingConfig => ({
    contextTokens: 4096,
    kvDtype: "q8_0",
    placement: tensor ? tensorPlacement(TWO_IDENTICAL) : singlePlacement(gpuIds[0] ?? 0),
    speculativeType: undefined,
    specDraftNMax: undefined,
    nParallel: 1,
    nBatch: undefined,
    nUbatch: undefined,
    disableVision: undefined,
  });

  it("is the question the second confirmation asks", () => {
    assert.equal(touchesDisplay(config([1]), TWO_IDENTICAL), false);
    assert.equal(touchesDisplay(config([0]), TWO_IDENTICAL), true);
    // Tensor-parallel across everything includes the card with the monitor.
    assert.equal(touchesDisplay(config([], true), TWO_IDENTICAL), true);
  });

  it("says no on a machine whose display GPU is not known", () => {
    const unknown = TWO_IDENTICAL.map((gpu) => ({ ...gpu, display: false }));
    assert.equal(touchesDisplay(config([0]), unknown), false);
  });
});
