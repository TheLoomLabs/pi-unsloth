import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { buildLoadRequest, type ModelOverride } from "../src/api/lifecycle.ts";
import { computeDevices, type GpuDevice } from "../src/api/system.ts";
import { isInferenceCommand } from "../src/process.ts";

/**
 * A fully tuned override, shaped exactly as the server stores one but with
 * neutral values — no machine's GPU ids or context lengths belong in this
 * repository.
 */
const TUNED: ModelOverride = {
  custom_context_length: 65536,
  kv_cache_dtype: "q8_0",
  speculative_type: "mtp",
  spec_draft_n_max: 2,
  n_parallel: 1,
  n_batch: 2048,
  n_ubatch: 512,
  tensor_parallel: true,
  gpu_ids: [0, 1],
};

describe("buildLoadRequest", () => {
  it("renames the two fields the load endpoint spells differently", () => {
    const request = buildLoadRequest("/models/a-model", "Q8_0", TUNED);
    assert.equal(request["max_seq_length"], 65536);
    assert.equal(request["cache_type_kv"], "q8_0");
    // The override's own spellings must not leak through as well.
    assert.equal(request["custom_context_length"], undefined);
    assert.equal(request["kv_cache_dtype"], undefined);
  });

  it("carries speculative decoding, which is the whole point of loading explicitly", () => {
    const request = buildLoadRequest("/models/a-model", "Q8_0", TUNED);
    assert.equal(request["speculative_type"], "mtp");
    assert.equal(request["spec_draft_n_max"], 2);
  });

  it("carries placement verbatim", () => {
    const request = buildLoadRequest("/models/a-model", "Q8_0", TUNED);
    assert.equal(request["tensor_parallel"], true);
    assert.deepEqual(request["gpu_ids"], [0, 1]);
  });

  it("overrides whatever auto-switch may have loaded a moment earlier", () => {
    const request = buildLoadRequest("/models/a-model", "Q8_0", TUNED);
    assert.equal(request.force_reload, true);
    assert.equal(request.force_cancel_active, true);
  });

  it("omits every field the override does not set, rather than guessing one", () => {
    const request = buildLoadRequest("/models/a-model", "Q4_K_M", { custom_context_length: 40960 });
    assert.deepEqual(Object.keys(request).sort(), [
      "force_cancel_active",
      "force_reload",
      "gguf_variant",
      "max_seq_length",
      "model_path",
    ]);
  });

  it("survives a model with no override and no quant at all", () => {
    const request = buildLoadRequest("org/hub-model", undefined, undefined);
    assert.deepEqual(request, {
      model_path: "org/hub-model",
      force_reload: true,
      force_cancel_active: true,
    });
  });
});

describe("computeDevices", () => {
  const device = (index: number, flags: Partial<GpuDevice> = {}): GpuDevice => ({
    index,
    name: undefined,
    memoryTotalGb: 24,
    vramUsedGb: 0,
    vramFreeGb: 24,
    sharedMemory: false,
    unifiedMemory: false,
    ...flags,
  });

  it("drops integrated graphics by the flags the server already sets", () => {
    const kept = computeDevices([
      device(0),
      device(1),
      device(2, { unifiedMemory: true, sharedMemory: true, memoryTotalGb: 30.34 }),
    ]);
    assert.deepEqual(
      kept.map((entry) => entry.index),
      [0, 1],
    );
  });

  it("drops a device flagged shared even when it is not flagged unified", () => {
    assert.deepEqual(computeDevices([device(0, { sharedMemory: true })]), []);
  });

  it("keeps a small discrete GPU — size is not the signal", () => {
    assert.equal(computeDevices([device(0, { memoryTotalGb: 4 })]).length, 1);
  });
});

describe("isInferenceCommand", () => {
  it("matches a server that has a model resident", () => {
    assert.equal(isInferenceCommand("/opt/llama.cpp/llama-server -m /models/a.gguf --port 41677"), true);
  });

  it("ignores a server started without a model", () => {
    assert.equal(isInferenceCommand("/opt/llama.cpp/llama-server --port 41677"), false);
  });

  it("ignores unrelated processes that merely mention a model file", () => {
    assert.equal(isInferenceCommand("cp -m /models/a.gguf /backup"), false);
  });
});
