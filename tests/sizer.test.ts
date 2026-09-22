import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { visibleWidth } from "@earendil-works/pi-tui";

import type { MemoryEstimate } from "../src/api/estimate.ts";
import {
  DISPLAY_HEADROOM_GB,
  HEADLESS_HEADROOM_GB,
  fitsPlacement,
  singlePlacement,
  tensorPlacement,
  type GpuFacts,
} from "../src/hardware/budget.ts";
import { placementLabel, type PlacementOption, type SizingConfig } from "../src/sizing/search.ts";
import {
  KV_DTYPES,
  SIZER_FIELDS,
  SPECULATIVE_MODES,
  applySizerKey,
  breakdownLines,
  fieldHint,
  fieldLines,
  fieldValue,
  fitLines,
  sizerLines,
  sizerSummary,
  sizerTitle,
  type SizerState,
  type SizerTheme,
} from "../src/ui/sizer.ts";

/** A theme that tags rather than colours, so widths are the real thing. */
function probe(): SizerTheme & { tokens: string[] } {
  const seen: string[] = [];
  return {
    tokens: seen,
    fg(color: string, text: string) {
      if (text !== "") seen.push(color);
      return text;
    },
    bold(text: string) {
      return text;
    },
  } as SizerTheme & { tokens: string[] };
}

const theme = probe();

const GPUS: GpuFacts[] = [
  { index: 0, totalGb: 23.98, idleUsedGb: 1.21, display: true, headroomGb: DISPLAY_HEADROOM_GB },
  { index: 1, totalGb: 23.98, idleUsedGb: 0.03, display: false, headroomGb: HEADLESS_HEADROOM_GB },
];

/**
 * The reference 192 K breakdown at **one** decode slot, as the live server
 * returns it — which is the configuration the reference box actually runs
 * (`--parallel 1`). The documented sample is the same model at the server's
 * default of four slots, and is used below to show what that costs.
 */
const ESTIMATE: MemoryEstimate = {
  available: true,
  reason: undefined,
  weightsBytes: 29_974_693_536,
  kvBytes: 10_139_860_992,
  computeBytes: 2_336_376_012,
  drafterRuntimeBytes: 1_483_245_977,
  drafterRuntimeGpuBytes: 1_483_245_977,
  projectorRuntimeBytes: 371_042_995,
  totalBytes: 44_305_219_512,
  gpuBytes: 41_167_355_832,
  kvEstimable: true,
  kvOnGpu: true,
  drafterKvUnsized: false,
  adaptersUnsized: false,
  moeOffloadUnmodelled: false,
  nCtx: 196_608,
  cacheTypeKv: "q8_0",
  nParallel: 1,
  layerCount: 65,
};

const CONFIG: SizingConfig = {
  contextTokens: 196_608,
  kvDtype: "q8_0",
  placement: tensorPlacement(GPUS),
  speculativeType: "mtp",
  specDraftNMax: 2,
  nParallel: 1,
  nBatch: undefined,
  nUbatch: undefined,
  disableVision: undefined,
};

const OPTIONS: PlacementOption[] = [
  { placement: singlePlacement(1), label: placementLabel(singlePlacement(1)), maxContext: undefined, estimate: undefined, verdict: undefined },
  { placement: singlePlacement(0), label: placementLabel(singlePlacement(0)), maxContext: undefined, estimate: undefined, verdict: undefined },
  { placement: tensorPlacement(GPUS), label: placementLabel(tensorPlacement(GPUS)), maxContext: 196_608, estimate: ESTIMATE, verdict: undefined },
];

function state(overrides: Partial<SizerState> = {}): SizerState {
  const config = overrides.config ?? CONFIG;
  return {
    model: { id: "local-model", name: "local-model", quant: "Q8_0", modelPath: "/models/local-model", ceiling: 262_144 },
    gpus: GPUS,
    config,
    step: 4096,
    options: OPTIONS,
    kvChoices: KV_DTYPES,
    specChoices: SPECULATIVE_MODES,
    field: 0,
    estimate: ESTIMATE,
    verdict: fitsPlacement(ESTIMATE.gpuBytes, GPUS, config.placement),
    maxContext: 196_608,
    busy: false,
    problem: undefined,
    caveat: undefined,
    calibration: undefined,
    confirming: false,
    note: undefined,
    ...overrides,
  };
}

describe("applySizerKey — moving between fields", () => {
  it("wraps in both directions", () => {
    const down = applySizerKey(state(), "\u001b[B");
    assert.equal(down.state.field, 1);
    const up = applySizerKey(state(), "\u001b[A");
    assert.equal(up.state.field, SIZER_FIELDS.length - 1);
  });

  it("changes nothing else, and asks for no new estimate", () => {
    const moved = applySizerKey(state(), "\u001b[B");
    assert.equal(moved.reprice, false);
    assert.deepEqual(moved.state.config, CONFIG);
  });
});

describe("applySizerKey — adjusting", () => {
  it("moves the context by one step, snapped and inside the model's ceiling", () => {
    const up = applySizerKey(state(), "\u001b[C");
    assert.equal(up.state.config.contextTokens, 196_608 + 4096);
    assert.equal(up.reprice, true);
    // …and never past the ceiling, however long the key is held.
    let held = state({ config: { ...CONFIG, contextTokens: 262_144 } });
    for (let press = 0; press < 5; press++) held = applySizerKey(held, "\u001b[C").state;
    assert.equal(held.config.contextTokens, 262_144);
  });

  it("never goes below one step", () => {
    let held = state({ config: { ...CONFIG, contextTokens: 4096 } });
    for (let press = 0; press < 3; press++) held = applySizerKey(held, "\u001b[D").state;
    assert.equal(held.config.contextTokens, 4096);
  });

  it("cycles the KV dtype, and re-searches because the ceiling moves with it", () => {
    const next = applySizerKey(state({ field: 1 }), "\u001b[C");
    assert.notEqual(next.state.config.kvDtype, "q8_0");
    assert.equal(next.reprice, true);
    assert.equal(next.research, true);
  });

  it("cycles the placement through the ladder this machine offers", () => {
    const next = applySizerKey(state({ field: 2 }), "\u001b[C");
    assert.equal(placementLabel(next.state.config.placement), "single: GPU 1");
    assert.equal(next.research, true);
  });

  it("cycles the speculative mode, including back to the server's default", () => {
    const seen = new Set<string>();
    let held = state({ field: 3 });
    for (let press = 0; press < SPECULATIVE_MODES.length; press++) {
      held = applySizerKey(held, "\u001b[C").state;
      seen.add(held.config.speculativeType ?? "default");
    }
    assert.ok(seen.has("default"));
    assert.ok(seen.has("mtp"));
  });

  it("moves the decode slots one at a time and stops at one", () => {
    const more = applySizerKey(state({ field: 4 }), "\u001b[C");
    assert.equal(more.state.config.nParallel, 2);
    const fewer = applySizerKey(state({ field: 4 }), "\u001b[D");
    assert.equal(fewer.state.config.nParallel, 1);
    assert.equal(fewer.reprice, false);
  });
});

describe("applySizerKey — deciding", () => {
  it("applies only a configuration that fits", () => {
    assert.equal(applySizerKey(state(), "\r").outcome, "apply");
    const over = state({ verdict: { fits: false, shortfallGb: 2.3, tightestIndex: 0 } });
    assert.equal(applySizerKey(over, "\r").outcome, "open");
  });

  it("asks twice before verifying on the card driving the monitor", () => {
    const asked = applySizerKey(state(), "v");
    assert.equal(asked.outcome, "open");
    assert.equal(asked.state.confirming, true);

    const confirmed = applySizerKey(asked.state, "v");
    assert.equal(confirmed.outcome, "verify");
    assert.equal(confirmed.state.confirming, false);
  });

  it("asks once when no display GPU is involved", () => {
    // A model small enough for the headless card on its own.
    const small: MemoryEstimate = { ...ESTIMATE, gpuBytes: 6_003_967_872, weightsBytes: 2_497_280_640 };
    const headless = state({
      config: { ...CONFIG, placement: singlePlacement(1) },
      estimate: small,
      verdict: fitsPlacement(small.gpuBytes, GPUS, singlePlacement(1)),
    });
    assert.equal(headless.verdict?.fits, true);
    assert.equal(applySizerKey(headless, "v").outcome, "verify");
  });

  it("lets escape back out of the confirmation without verifying", () => {
    const asked = applySizerKey(state(), "v");
    const backedOut = applySizerKey(asked.state, "\u001b");
    assert.equal(backedOut.outcome, "open");
    assert.equal(backedOut.state.confirming, false);
  });

  it("ignores every other key while the confirmation is up", () => {
    const asked = applySizerKey(state(), "v").state;
    for (const key of ["\u001b[C", "\u001b[B", "r", "u"]) {
      const result = applySizerKey(asked, key);
      assert.equal(result.outcome, "open", key);
      assert.equal(result.state.confirming, true, key);
      assert.deepEqual(result.state.config, asked.config, key);
    }
  });

  it("does not offer to verify a configuration that does not fit", () => {
    const over = state({ verdict: { fits: false, shortfallGb: 2.3, tightestIndex: 0 } });
    const result = applySizerKey(over, "v");
    assert.equal(result.outcome, "open");
    assert.equal(result.state.confirming, false);
  });

  it("escapes out of the screen itself", () => {
    assert.equal(applySizerKey(state(), "\u001b").outcome, "cancel");
  });
});

describe("the breakdown", () => {
  it("names every term, including the two the old estimator ignored", () => {
    const text = breakdownLines(state(), 60, theme).join("\n");
    assert.match(text, /weights\s+27\.9 GiB/);
    assert.match(text, /kv @ 196608\s+9\.4 GiB/);
    assert.match(text, /drafter\s+1\.4 GiB/);
    assert.match(text, /projector\s+0\.3 GiB/);
  });

  it("stacks into one column when there is no room for two", () => {
    const narrow = breakdownLines(state(), 34, theme);
    assert.equal(narrow.length, 5);
    for (const line of narrow) assert.ok(visibleWidth(line) <= 34, line);
  });

  it("says it is pricing rather than showing a stale total", () => {
    const text = breakdownLines(state({ estimate: undefined, busy: true }), 60, theme).join("");
    assert.match(text, /pricing…/);
  });
});

describe("the fit verdict", () => {
  it("shows each card's own share against its own budget", () => {
    const text = fitLines(state(), 60, theme).join("\n");
    assert.match(text, /GPU 0\s+19\.2 GiB of 19\.8 GiB budget/);
    assert.match(text, /GPU 1\s+19\.2 GiB of 23\.4 GiB budget/);
    assert.match(text, /fits ✓/);
  });

  it("shows the display card running out first — which four decode slots would do", () => {
    // The same model, same context, at the server's own default of four slots:
    // 39.7 GiB instead of 38.3, and the card with the monitor on it is 0.1 GiB
    // short. This is why `slots` is a field on this screen.
    const fourSlots: MemoryEstimate = { ...ESTIMATE, gpuBytes: 42_625_401_988, kvBytes: 10_610_540_544, nParallel: 4 };
    const crowded = state({
      estimate: fourSlots,
      config: { ...CONFIG, nParallel: 4 },
      verdict: fitsPlacement(fourSlots.gpuBytes, GPUS, CONFIG.placement),
    });
    const text = fitLines(crowded, 60, theme).join("\n");
    assert.match(text, /over by 0\.1 GiB on GPU 0/);
  });

  it("names the tightest card and the shortfall when it does not fit", () => {
    const over = state({ verdict: { fits: false, shortfallGb: 2.3, tightestIndex: 0 } });
    const text = fitLines(over, 60, theme).join("\n");
    assert.match(text, /over by 2\.3 GiB on GPU 0/);
  });

  it("says when a measurement is in the figure", () => {
    const measured = state({
      calibration: { deltaGb: 0.62, samples: 1, stamp: { unslothVersion: "2026.9.7", backend: "rocm" } },
    });
    assert.match(fitLines(measured, 60, theme).join("\n"), /includes \+0\.62 GiB measured/);
  });
});

describe("the fields", () => {
  it("draws each value in the `‹ … ›` shape, with the server's default named", () => {
    assert.equal(fieldValue(state(), "context"), "196608");
    assert.equal(fieldValue(state(), "placement"), "tensor-parallel 0+1");
    assert.equal(fieldValue(state({ config: { ...CONFIG, kvDtype: undefined } }), "kv"), "default");
    assert.equal(fieldValue(state({ config: { ...CONFIG, speculativeType: undefined } }), "speculative"), "default");
  });

  it("puts `max safe` beside the context, and says so while it is unknown", () => {
    assert.equal(fieldHint(state(), "context"), "max safe 196608");
    assert.equal(fieldHint(state({ busy: true }), "context"), "max safe …");
    assert.match(fieldHint(state({ maxContext: undefined }), "context") ?? "", /nothing fits/);
  });

  it("fits the width it is given", () => {
    for (const width of [80, 54, 40]) {
      for (const line of fieldLines(state(), width, theme)) {
        assert.ok(visibleWidth(line) <= width, `${width}: ${line}`);
      }
    }
  });
});

describe("sizerLines", () => {
  it("draws the model, the breakdown, the verdict, the fields and the keys", () => {
    const text = sizerLines(state(), 60, theme).join("\n");
    // The model's name heads the frame (`sizerTitle`), not the body.
    assert.equal(sizerTitle(state()), "Size  local-model  Q8_0");
    assert.match(text, /weights/);
    assert.match(text, /fits ✓/);
    assert.match(text, /context/);
    assert.match(text, /⏎ apply/);
    assert.match(text, /v verify/);
  });

  it("replaces everything with the confirmation when one is up", () => {
    const text = sizerLines(state({ confirming: true }), 60, theme).join("\n");
    assert.equal(sizerTitle(state({ confirming: true })), "Verify on the display GPU?");
    assert.match(text, /GPU 0 drives your monitor/);
    assert.doesNotMatch(text, /⏎ apply/);
  });

  it("says why there is nothing to show instead of an empty box", () => {
    const text = sizerLines(state({ problem: "not downloaded — nothing on this disk to measure" }), 60, theme).join("\n");
    assert.match(text, /⚠ not downloaded/);
    assert.match(text, /esc close/);
  });

  it("carries a caveat about a total that is only a floor", () => {
    const text = sizerLines(state({ caveat: "the KV cache could not be sized from this model's header" }), 60, theme).join("\n");
    assert.match(text, /KV cache could not be sized/);
  });

  it("never draws wider than it is allowed", () => {
    for (const width of [80, 60, 54, 50]) {
      for (const line of sizerLines(state(), width, theme)) {
        assert.ok(visibleWidth(line) <= width, `${width}: ${line}`);
      }
    }
  });
});

describe("sizerSummary", () => {
  it("is one line for a terminal too narrow to draw in", () => {
    assert.equal(sizerSummary(state()), "⬢ local-model — 192K on tensor-parallel 0+1 · fits");
  });

  it("leads with the shortfall when nothing fits", () => {
    const over = state({ verdict: { fits: false, shortfallGb: 2.3, tightestIndex: 0 } });
    assert.match(sizerSummary(over), /^⚠ local-model — over by 2\.3 GiB/);
  });

  it("carries the reason when the model could not be priced", () => {
    assert.match(sizerSummary(state({ problem: "not a GGUF model" })), /not a GGUF model/);
  });
});
