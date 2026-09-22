import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { EstimateRequest, MemoryEstimate } from "../src/api/estimate.ts";
import {
  DISPLAY_HEADROOM_GB,
  HEADLESS_HEADROOM_GB,
  bytesToGb,
  placementLadder,
  singlePlacement,
  tensorPlacement,
  type GpuFacts,
} from "../src/hardware/budget.ts";
import {
  DEFAULT_CTX_STEP,
  buildEstimateRequest,
  buildOverrideEntry,
  cachedEstimator,
  configFromOverride,
  correctedGpuBytes,
  maxContextFor,
  perGpuGb,
  placementLabel,
  searchMaxContext,
  sizeModel,
  snapContext,
  type SizingConfig,
} from "../src/sizing/search.ts";

const GIB = 1024 ** 3;

/** Two 24 GiB cards, one driving the monitor — the reference shape. */
const TWO_IDENTICAL: GpuFacts[] = [
  { index: 0, totalGb: 23.98, idleUsedGb: 1.21, display: true, headroomGb: DISPLAY_HEADROOM_GB },
  { index: 1, totalGb: 23.98, idleUsedGb: 0.03, display: false, headroomGb: HEADLESS_HEADROOM_GB },
];

/** One card, and the monitor is on it. */
const SINGLE_DESKTOP: GpuFacts[] = [
  { index: 0, totalGb: 23.98, idleUsedGb: 1.21, display: true, headroomGb: DISPLAY_HEADROOM_GB },
];

function estimate(gpuBytes: number, overrides: Partial<MemoryEstimate> = {}): MemoryEstimate {
  return {
    available: true,
    reason: undefined,
    weightsBytes: 0,
    kvBytes: 0,
    computeBytes: 0,
    drafterRuntimeBytes: 0,
    drafterRuntimeGpuBytes: 0,
    projectorRuntimeBytes: 0,
    totalBytes: gpuBytes,
    gpuBytes,
    kvEstimable: true,
    kvOnGpu: true,
    drafterKvUnsized: false,
    adaptersUnsized: false,
    moeOffloadUnmodelled: false,
    nCtx: undefined,
    cacheTypeKv: "q8_0",
    nParallel: 1,
    layerCount: 65,
    ...overrides,
  };
}

/**
 * A stand-in estimator **fitted to the live server**.
 *
 * Two real answers from the reference machine for the same 27B — one at
 * 196 608 tokens and one at 262 144 — determine a line exactly:
 *
 *     gpu_bytes(n_ctx) = 41 167 355 832 + 50 176 × (n_ctx − 196 608)
 *
 * and that line reproduces both measured `gpu_bytes` to the byte. So the search
 * below is exercised against the real estimator's arithmetic rather than
 * against a shape invented to make it pass.
 */
const REFERENCE_AT_192K = 41_167_355_832;
const REFERENCE_BYTES_PER_TOKEN = 50_176;

function referenceEstimator(): (request: EstimateRequest) => Promise<MemoryEstimate> {
  return async (request) => {
    const contextTokens = request.n_ctx ?? 262_144;
    const gpuBytes = REFERENCE_AT_192K + REFERENCE_BYTES_PER_TOKEN * (contextTokens - 196_608);
    return estimate(gpuBytes, { nCtx: contextTokens, weightsBytes: 29_974_693_536 });
  };
}

const BASE: SizingConfig = {
  contextTokens: 4096,
  kvDtype: "q8_0",
  placement: singlePlacement(0),
  speculativeType: "mtp",
  specDraftNMax: 2,
  nParallel: 1,
  nBatch: undefined,
  nUbatch: undefined,
  disableVision: undefined,
};

describe("snapContext", () => {
  it("snaps down, so the answer stays inside the budget it was checked against", () => {
    assert.equal(snapContext(70000), 69632);
    assert.equal(snapContext(65536), 65536);
  });

  it("never returns less than one step, and survives nonsense", () => {
    assert.equal(snapContext(1), DEFAULT_CTX_STEP);
    assert.equal(snapContext(Number.NaN), 0);
    assert.equal(snapContext(8192, 0), 0);
  });
});

describe("searchMaxContext", () => {
  it("finds the largest step that fits", async () => {
    const fits = async (tokens: number) => tokens <= 70_000;
    assert.equal(await searchMaxContext(fits, { max: 262_144 }), 69_632);
  });

  it("answers the ceiling in one question when everything fits", async () => {
    let asked = 0;
    const fits = async () => {
      asked += 1;
      return true;
    };
    assert.equal(await searchMaxContext(fits, { max: 262_144 }), 262_144);
    assert.equal(asked, 1);
  });

  it("gives up after two questions when not even the floor fits", async () => {
    let asked = 0;
    const fits = async () => {
      asked += 1;
      return false;
    };
    assert.equal(await searchMaxContext(fits, { max: 262_144 }), undefined);
    assert.equal(asked, 2);
  });

  it("costs a handful of estimates, not a scan", async () => {
    let asked = 0;
    const fits = async (tokens: number) => {
      asked += 1;
      return tokens <= 123_456;
    };
    const found = await searchMaxContext(fits, { max: 1_048_576 });
    assert.equal(found, 122_880);
    assert.ok(asked <= 10, `asked ${asked} times`);
  });

  it("never proposes more than the model's own ceiling", async () => {
    const found = await searchMaxContext(async () => true, { max: 40_960 });
    assert.equal(found, 40_960);
  });
});

describe("sizeModel — the fit decision", () => {
  const target = { modelPath: "/models/local-model", quant: "Q8_0" };

  it("reports `won't fit` for the documented 32 K single-GPU case, despite available: true", async () => {
    // 's gotcha, as a regression test: the estimator called a 32.4 GB estimate
    // `available` against a 23.98 GB card, because it permits CPU offload. The
    // per-GPU rule is the extension's own.
    const estimator = async () => estimate(32.4e9, { available: true, nCtx: 32_768 });
    const proposal = await sizeModel(estimator, target, SINGLE_DESKTOP, BASE, { ceiling: 32_768 });

    assert.equal(proposal.config, undefined);
    assert.equal(proposal.verdict?.fits, false);
    assert.equal(proposal.verdict?.tightestIndex, 0);
    assert.ok((proposal.verdict?.shortfallGb ?? 0) > 10);
    assert.equal(proposal.unsizable, undefined);
  });

  it("puts a small model on the headless card, not the one with the monitor", async () => {
    // A 4B-shaped model: small enough for either card.
    const estimator = async () => estimate(6.0e9, { nCtx: 40_960 });
    const proposal = await sizeModel(estimator, target, TWO_IDENTICAL, BASE, { ceiling: 40_960 });

    assert.deepEqual(proposal.config?.placement.gpuIds, [1]);
    assert.equal(proposal.config?.placement.kind, "single");
    assert.equal(proposal.config?.contextTokens, 40_960);
    assert.equal(proposal.verdict?.fits, true);
  });

  it("goes tensor-parallel for a model no single card can hold", async () => {
    const proposal = await sizeModel(referenceEstimator(), target, TWO_IDENTICAL, BASE, { ceiling: 262_144 });

    assert.equal(proposal.config?.placement.kind, "tensor-parallel");
    assert.deepEqual(proposal.config?.placement.gpuIds, [0, 1]);
    assert.equal(proposal.verdict?.fits, true);
  });

  it("finds the largest context the pair can hold when the ceiling does not fit", async () => {
    const proposal = await sizeModel(referenceEstimator(), target, TWO_IDENTICAL, BASE, { ceiling: 262_144 });
    const found = proposal.config?.contextTokens ?? 0;

    // Not the ceiling — 262 144 needs 41.4 GiB, and the display card's budget
    // caps the pair at about 39.5.
    assert.ok(found < 262_144, `found ${found}`);
    assert.ok(found >= 196_608, `found ${found}`);
    assert.equal(found % DEFAULT_CTX_STEP, 0);

    // …and one step more really does not fit, which is what "largest" means.
    const oneMore = await referenceEstimator()({ model_path: target.modelPath, n_ctx: found + DEFAULT_CTX_STEP });
    assert.ok(bytesToGb(oneMore.gpuBytes) / 2 > 19.77);
  });

  it("reports the real maximum, not just that the target fits", async () => {
    // A 4B-shaped model with a stored 8 K override on a machine that could hold
    // its whole 40 960: the screen must say so, or `max safe` is the user's own
    // setting read back to them.
    const estimator = async (request: EstimateRequest) => estimate(2e9 + 40_000 * (request.n_ctx ?? 0), { nCtx: request.n_ctx });
    const proposal = await sizeModel(estimator, target, TWO_IDENTICAL, BASE, { ceiling: 40_960, target: 8192 });

    assert.equal(proposal.config?.contextTokens, 8192, "the user's own setting is the proposal");
    assert.equal(proposal.options.find((option) => option.label === "single: GPU 1")?.maxContext, 40_960);
  });

  it("offers every placement, in preference order, for the user to cycle", async () => {
    const proposal = await sizeModel(referenceEstimator(), target, TWO_IDENTICAL, BASE, { ceiling: 262_144 });
    assert.deepEqual(
      proposal.options.map((option) => option.label),
      ["single: GPU 1", "single: GPU 0", "tensor-parallel 0+1"],
    );
    assert.deepEqual(
      placementLadder(TWO_IDENTICAL).map(placementLabel),
      proposal.options.map((option) => option.label),
    );
  });

  it("keeps a model off the display GPU even when the display GPU would fit it", async () => {
    const estimator = async () => estimate(18e9, { nCtx: 8192 });
    const proposal = await sizeModel(estimator, target, TWO_IDENTICAL, BASE, { ceiling: 8192 });
    assert.deepEqual(proposal.config?.placement.gpuIds, [1]);

    const relaxed = await sizeModel(estimator, target, TWO_IDENTICAL, BASE, { ceiling: 8192, preferHeadless: false });
    // Tightest-first among all cards once the preference is off.
    assert.deepEqual(relaxed.config?.placement.gpuIds, [0]);
  });

  it("says why rather than how much when the model cannot be priced at all", async () => {
    const estimator = async () => estimate(0, { available: false, reason: "not_downloaded" });
    const proposal = await sizeModel(estimator, target, TWO_IDENTICAL, BASE, { ceiling: 40_960 });
    assert.equal(proposal.unsizable, "not_downloaded");
    assert.equal(proposal.config, undefined);
  });

  it("has nothing to propose on a machine with no compute GPUs", async () => {
    const proposal = await sizeModel(referenceEstimator(), target, [], BASE, { ceiling: 40_960 });
    assert.deepEqual(proposal.options, []);
    assert.equal(proposal.config, undefined);
  });
});

describe("calibration in the fit rule", () => {
  it("charges a learned delta against the budget", () => {
    const priced = estimate(20 * GIB);
    assert.equal(bytesToGb(correctedGpuBytes(priced, 0)), 20);
    assert.equal(bytesToGb(correctedGpuBytes(priced, 1.5)), 21.5);
    // A correction can never make a model cost negative memory.
    assert.equal(correctedGpuBytes(estimate(0), -5), 0);
  });

  it("shrinks the context a machine is offered once it has measured itself", async () => {
    const target = { modelPath: "/models/local-model", quant: "Q8_0" };
    // The 27B only goes across the pair at all, so that is the placement to ask
    // about — a single card cannot hold its weights at any context.
    const config = { ...BASE, placement: tensorPlacement(TWO_IDENTICAL) };
    const plain = await maxContextFor(referenceEstimator(), target, TWO_IDENTICAL, config, { ceiling: 262_144 });
    const corrected = await maxContextFor(referenceEstimator(), target, TWO_IDENTICAL, config, {
      ceiling: 262_144,
      calibrationGb: 2.0,
    });
    assert.ok((corrected ?? 0) < (plain ?? 0), `${corrected} vs ${plain}`);
  });
});

describe("perGpuGb", () => {
  it("splits a tensor-parallel total the way llama.cpp does", () => {
    const shares = perGpuGb(estimate(40 * GIB), tensorPlacement(TWO_IDENTICAL));
    assert.deepEqual(
      shares.map((value) => value.toFixed(1)),
      ["20.0", "20.0"],
    );
  });

  it("puts all of it on one card for a single placement", () => {
    assert.deepEqual(perGpuGb(estimate(12 * GIB), singlePlacement(1)), [12]);
  });
});

describe("the configuration round trip", () => {
  const config: SizingConfig = {
    ...BASE,
    contextTokens: 65_536,
    placement: tensorPlacement(TWO_IDENTICAL),
  };

  it("prices what it will write, and writes what it priced", () => {
    const request = buildEstimateRequest({ modelPath: "/models/local-model", quant: "Q8_0" }, config);
    assert.equal(request.n_ctx, 65_536);
    assert.equal(request.cache_type_kv, "q8_0");
    assert.equal(request.tensor_parallel, true);
    assert.deepEqual(request.selected_gpu_ids, [0, 1]);
    assert.equal(request.gguf_variant, "Q8_0");

    const entry = buildOverrideEntry(config);
    // The override spells two of these differently from the estimate.
    assert.equal(entry["custom_context_length"], 65_536);
    assert.equal(entry["kv_cache_dtype"], "q8_0");
    assert.equal(entry["tensor_parallel"], true);
    assert.deepEqual(entry["gpu_ids"], [0, 1]);
    assert.equal(entry["speculative_type"], "mtp");
  });

  it("omits what the user left to the server, rather than inventing a default", () => {
    const bare = buildOverrideEntry({
      ...config,
      placement: singlePlacement(1),
      kvDtype: undefined,
      speculativeType: undefined,
      specDraftNMax: undefined,
      nParallel: undefined,
    });
    assert.deepEqual(Object.keys(bare).sort(), ["custom_context_length", "gpu_ids"]);
    // …including `tensor_parallel: false`, which the server drops anyway.
    assert.equal("tensor_parallel" in bare, false);
  });

  it("reads a stored override back without repairing what is not there", () => {
    const read = configFromOverride({ kv_cache_dtype: "q8_0", n_parallel: 1 }, singlePlacement(1), 40_960);
    assert.equal(read.kvDtype, "q8_0");
    assert.equal(read.nParallel, 1);
    assert.equal(read.speculativeType, undefined);
    assert.equal(read.contextTokens, 40_960);
  });
});

describe("cachedEstimator", () => {
  it("asks once per configuration, however often the search revisits it", async () => {
    let asked = 0;
    const estimator = cachedEstimator(async () => {
      asked += 1;
      return estimate(1);
    });
    await estimator({ model_path: "a", n_ctx: 4096 });
    await estimator({ model_path: "a", n_ctx: 4096 });
    await estimator({ model_path: "a", n_ctx: 8192 });
    assert.equal(asked, 2);
  });

  it("does not remember a failure as an answer", async () => {
    let asked = 0;
    const estimator = cachedEstimator(async () => {
      asked += 1;
      if (asked === 1) throw new Error("down");
      return estimate(1);
    });
    await assert.rejects(() => estimator({ model_path: "a" }));
    await estimator({ model_path: "a" });
    assert.equal(asked, 2);
  });
});
