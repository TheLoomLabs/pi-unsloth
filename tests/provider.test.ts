import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { findOverride, overrideKey } from "../src/api/lifecycle.ts";
import type { CachedGguf, CatalogueEntry, LoadedStatus, LocalModel } from "../src/api/models.ts";
import {
  DEFAULT_CONTEXT_WINDOW,
  buildModels,
  buildThinkingLevelMap,
  findLocal,
  formatContext,
  resolveContextWindow,
} from "../src/provider.ts";

/**
 * The two reference models, shaped exactly as a live Unsloth Studio reports
 * them, but with neutral ids — no machine or model name belongs in this
 * repository.
 */
const ENTRIES: CatalogueEntry[] = [
  {
    id: "local-model",
    displayName: undefined,
    quant: "Q8_0",
    loaded: true,
    contextLength: 196608,
    maxContextLength: 262144,
    nativeContextLength: 262144,
  },
  {
    id: "org/hub-model",
    displayName: "hub-model",
    quant: "Q4_K_M",
    loaded: false,
    contextLength: undefined,
    maxContextLength: undefined,
    nativeContextLength: undefined,
  },
];

const LOCAL: LocalModel[] = [
  {
    id: "/models/local-model",
    displayName: "local-model",
    path: "/models/local-model",
    source: "models_dir",
    repoId: undefined,
    modelFormat: "gguf",
    task: "text-generation",
    partial: false,
  },
  {
    id: "org/hub-model",
    displayName: "hub-model",
    path: "/cache/models--org--hub-model",
    source: "hf_cache",
    repoId: "org/hub-model",
    modelFormat: "gguf",
    task: "text-generation",
    partial: false,
  },
];

const CACHED: CachedGguf[] = [
  {
    repoId: "org/hub-model",
    cachePath: "/cache/models--org--hub-model",
    sizeBytes: 2_497_280_640,
    hasVision: false,
    task: "text-generation",
  },
];

const STATUS: LoadedStatus = {
  isVision: true,
  supportsReasoning: true,
  reasoningEffortLevels: ["low", "medium", "high", "xhigh"],
  contextLength: 196608,
  maxContextLength: 262144,
  nativeContextLength: 262144,
  modelPath: "/models/local-model",
};

const OVERRIDES = {
  "/models/local-model:Q8_0": { custom_context_length: 196608, kv_cache_dtype: "q8_0" },
  "org/hub-model:Q4_K_M": { custom_context_length: 40960, kv_cache_dtype: "q8_0" },
};

const BASE_URL = "http://127.0.0.1:8888";

function build(overrides: Partial<Parameters<typeof buildModels>[0]> = {}) {
  return buildModels({
    baseUrl: BASE_URL,
    entries: ENTRIES,
    local: LOCAL,
    cached: CACHED,
    overrides: OVERRIDES,
    status: STATUS,
    remembered: [],
    ...overrides,
  });
}

describe("formatContext", () => {
  it("renders token counts the way the panel shows them", () => {
    assert.equal(formatContext(196608), "192K");
    assert.equal(formatContext(40960), "40K");
    assert.equal(formatContext(8192), "8K");
    assert.equal(formatContext(1024 * 1024), "1M");
    assert.equal(formatContext(512), "512");
  });
});

describe("buildThinkingLevelMap", () => {
  it("maps advertised levels and nulls the rest", () => {
    assert.deepEqual(buildThinkingLevelMap(["low", "medium", "high", "xhigh"]), {
      minimal: null,
      low: "low",
      medium: "medium",
      high: "high",
      xhigh: "xhigh",
      max: null,
    });
  });

  it("leaves `off` unset so pi keeps it selectable and sends no effort", () => {
    assert.equal("off" in (buildThinkingLevelMap(["low"]) ?? {}), false);
  });

  it("is undefined when the server said nothing about effort levels", () => {
    assert.equal(buildThinkingLevelMap(undefined), undefined);
  });

  it("ignores server levels that are not pi levels", () => {
    assert.deepEqual(buildThinkingLevelMap(["low", "none", "auto"]), {
      minimal: null,
      low: "low",
      medium: null,
      high: null,
      xhigh: null,
      max: null,
    });
  });
});

describe("override keys", () => {
  it("keys a models_dir model by path and a hub model by repo id", () => {
    assert.equal(overrideKey("/models/local-model", "Q8_0"), "/models/local-model:Q8_0");
    assert.equal(overrideKey("org/hub-model", "Q4_K_M"), "org/hub-model:Q4_K_M");
  });

  it("tries every plausible key rather than one clever one", () => {
    assert.equal(
      findOverride(OVERRIDES, ["org/hub-model", "/cache/models--org--hub-model"], "Q4_K_M")
        ?.custom_context_length,
      40960,
    );
    assert.equal(
      findOverride(OVERRIDES, [undefined, "/models/local-model", "local-model"], "Q8_0")
        ?.custom_context_length,
      196608,
    );
    assert.equal(findOverride(OVERRIDES, ["nothing-like-this"], "Q8_0"), undefined);
  });
});

describe("findLocal", () => {
  it("resolves a catalogue id to its filesystem entry", () => {
    assert.equal(findLocal(ENTRIES[0]!, LOCAL)?.path, "/models/local-model");
    assert.equal(findLocal(ENTRIES[1]!, LOCAL)?.repoId, "org/hub-model");
  });

  it("returns undefined rather than guessing", () => {
    assert.equal(findLocal({ ...ENTRIES[0]!, id: "unknown", displayName: undefined }, LOCAL), undefined);
  });
});

describe("resolveContextWindow", () => {
  const bare: CatalogueEntry = {
    id: "x",
    displayName: undefined,
    quant: "Q8_0",
    loaded: false,
    contextLength: undefined,
    maxContextLength: undefined,
    nativeContextLength: undefined,
  };

  it("prefers the tuned override", () => {
    assert.equal(
      resolveContextWindow({ ...bare, contextLength: 1024, nativeContextLength: 262144 },
        { "x:Q8_0": { custom_context_length: 196608 } }, undefined),
      196608,
    );
  });

  it("falls back to the configured context, then the native one", () => {
    assert.equal(resolveContextWindow({ ...bare, contextLength: 65536, nativeContextLength: 262144 }, {}, undefined), 65536);
    assert.equal(resolveContextWindow({ ...bare, nativeContextLength: 32768 }, {}, undefined), 32768);
  });

  it("ends at a conservative default rather than a guess", () => {
    assert.equal(resolveContextWindow(bare, {}, undefined), DEFAULT_CONTEXT_WINDOW);
  });

  it("clamps an override above the server's own ceiling", () => {
    assert.equal(
      resolveContextWindow({ ...bare, maxContextLength: 65536, nativeContextLength: 65536 },
        { "x:Q8_0": { custom_context_length: 999999 } }, undefined),
      65536,
    );
  });
});

describe("buildModels", () => {
  it("maps the loaded model from its live status", () => {
    const loaded = build()[0]!;
    assert.equal(loaded.id, "local-model");
    assert.equal(loaded.provider, "unsloth");
    assert.equal(loaded.api, "openai-completions");
    assert.equal(loaded.baseUrl, `${BASE_URL}/v1`);
    assert.equal(loaded.reasoning, true);
    assert.deepEqual(loaded.input, ["text", "image"]);
    assert.equal(loaded.contextWindow, 196608);
    assert.equal(loaded.maxTokens, 8192);
    assert.deepEqual(loaded.cost, { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
    assert.deepEqual(loaded.compat, { supportsDeveloperRole: false });
    assert.equal(loaded.thinkingLevelMap?.high, "high");
    assert.equal(loaded.thinkingLevelMap?.max, null);
  });

  it("gives a cold model its tuned context, not a guess", () => {
    const cold = build()[1]!;
    assert.equal(cold.contextWindow, 40960);
  });

  it("does not apply the loaded model's capabilities to a cold one", () => {
    const cold = build()[1]!;
    assert.equal(cold.reasoning, false);
    assert.deepEqual(cold.input, ["text"]);
    assert.equal(cold.thinkingLevelMap, undefined);
  });

  it("caps maxTokens at the context window for a tiny model", () => {
    const [tiny] = build({
      entries: [{ ...ENTRIES[0]!, nativeContextLength: 4096, maxContextLength: 4096 }],
      overrides: {},
      status: undefined,
    });
    assert.equal(tiny!.contextWindow, 4096);
    assert.equal(tiny!.maxTokens, 4096);
  });

  it("remembers capabilities observed while a model was loaded", () => {
    const observed = build()[0]!;
    const [cold] = build({
      entries: [{ ...ENTRIES[0]!, loaded: false, contextLength: undefined }],
      status: undefined,
      remembered: [observed],
    });
    assert.equal(cold!.reasoning, true);
    assert.deepEqual(cold!.input, ["text", "image"]);
    assert.equal(cold!.thinkingLevelMap?.high, "high");
  });

  it("uses cached-gguf for vision when no status is available", () => {
    const [cold] = build({
      entries: [ENTRIES[1]!],
      cached: [{ ...CACHED[0]!, hasVision: true }],
      status: undefined,
    });
    assert.deepEqual(cold!.input, ["text", "image"]);
  });

  it("names a model from what the server calls it, plus quant and context", () => {
    assert.equal(build()[0]!.name, "local-model · Q8_0 · 192K");
    assert.equal(build()[1]!.name, "hub-model · Q4_K_M · 40K");
  });

  it("produces nothing from an empty catalogue instead of failing", () => {
    assert.deepEqual(build({ entries: [] }), []);
  });
});
