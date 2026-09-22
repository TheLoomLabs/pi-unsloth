import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, afterEach, before, beforeEach, describe, it } from "node:test";

import { DEFAULT_HUB_ENDPOINT, baseModelOf, fetchGenerationConfig, isRepoId } from "../src/api/hub.ts";
import { PROFILE_FILENAME, readSamplingStore, rememberSampling } from "../src/hardware/profile.ts";
import { buildModels, type CatalogueInputs } from "../src/provider.ts";
import {
  SAMPLING_FIELDS,
  SAMPLING_KEYS,
  adjustParam,
  clearParam,
  formatParam,
  hasParams,
  normaliseEntry,
  normaliseParams,
  normaliseStore,
  parseGenerationConfig,
  type SamplingParams,
} from "../src/sampling.ts";
import {
  applySamplingKey,
  fieldLines,
  initialState,
  provenanceLine,
  samplingLines,
  samplingSummary,
  samplingTitle,
  type SamplingTheme,
  type SamplingUiState,
} from "../src/ui/sampling.ts";

/** A theme that tags rather than colours, so widths are the real thing. */
function probe(): SamplingTheme & { tokens: string[] } {
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
  } as SamplingTheme & { tokens: string[] };
}

const theme = probe();

function state(params: SamplingParams = {}, overrides: Partial<SamplingUiState> = {}): SamplingUiState {
  return {
    ...initialState({ id: "Qwen3.8-27B", name: "Qwen3.8-27B", repoId: "Qwen/Qwen3-27B" }, params, "user", undefined),
    ...overrides,
  };
}

/** The real thing, from `Qwen/Qwen3-4B` on 2026-09-22. */
const QWEN_GENERATION_CONFIG = {
  bos_token_id: 151643,
  do_sample: true,
  eos_token_id: [151645, 151643],
  pad_token_id: 151643,
  temperature: 0.6,
  top_k: 20,
  top_p: 0.95,
  transformers_version: "4.51.0",
};

describe("the stored shape", () => {
  it("keeps the four samplers the server models, and drops everything else", () => {
    const params = normaliseParams({ temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0.05, repeat_penalty: 1.1, junk: "x" });
    assert.deepEqual(params, { temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0.05 });
  });

  it("drops a value outside its range rather than clamping it", () => {
    // A `top_p` of 4 is a mistake in a hand-edited profile. Turning it into 1
    // would hide the mistake instead of the value.
    assert.deepEqual(normaliseParams({ top_p: 4, temperature: 0.7 }), { temperature: 0.7 });
    assert.deepEqual(normaliseParams({ temperature: "hot" }), {});
  });

  it("reads a bare hand edit as well as the documented shape", () => {
    assert.deepEqual(normaliseEntry({ temperature: 0.7 })?.params, { temperature: 0.7 });
    assert.deepEqual(normaliseEntry({ params: { temperature: 0.7 }, source: "hub" })?.source, "hub");
  });

  it("is nothing at all when nothing survives", () => {
    assert.equal(normaliseEntry({ params: { temperature: 99 } }), undefined);
    assert.deepEqual(normaliseStore({ "a-model": { temperature: 99 } }), {});
    assert.deepEqual(normaliseStore("not an object"), {});
  });

  it("keeps an unknown key a hand edit added", () => {
    const entry = normaliseEntry({ params: { temperature: 0.7 }, note: "mine" });
    assert.equal(entry?.["note"], "mine");
  });
});

describe("adjusting a sampler", () => {
  it("materialises llama.cpp's own default on the first press, either way", () => {
    assert.equal(adjustParam({}, "temperature", 1).temperature, SAMPLING_FIELDS.temperature.start);
    assert.equal(adjustParam({}, "temperature", -1).temperature, SAMPLING_FIELDS.temperature.start);
  });

  it("steps without float drift", () => {
    let params: SamplingParams = { temperature: 0.8 };
    for (let at = 0; at < 3; at += 1) params = adjustParam(params, "temperature", 1);
    assert.equal(params.temperature, 0.95);
    assert.equal(formatParam(params, "temperature"), "0.95");
  });

  it("clamps at both ends", () => {
    assert.equal(adjustParam({ temperature: 0 }, "temperature", -1).temperature, 0);
    assert.equal(adjustParam({ top_p: 1 }, "top_p", 1).top_p, 1);
  });

  it("shows an integer sampler as an integer", () => {
    assert.equal(formatParam({ top_k: 20 }, "top_k"), "20");
    assert.equal(formatParam({}, "top_k"), "default");
  });

  it("clears to unset, which is not zero", () => {
    const cleared = clearParam({ temperature: 0.6, top_k: 20 }, "temperature");
    assert.equal(cleared.temperature, undefined);
    assert.equal(hasParams(cleared), true);
    assert.equal(hasParams(clearParam(cleared, "top_k")), false);
  });
});

describe("a model card", () => {
  it("yields the three samplers Qwen publishes", () => {
    assert.deepEqual(parseGenerationConfig(QWEN_GENERATION_CONFIG), { temperature: 0.6, top_p: 0.95, top_k: 20 });
  });

  it("yields nothing when the card says the model is decoded greedily", () => {
    assert.deepEqual(parseGenerationConfig({ ...QWEN_GENERATION_CONFIG, do_sample: false }), {});
  });

  it("yields nothing for a file that is not one", () => {
    assert.deepEqual(parseGenerationConfig("<!doctype html>"), {});
  });
});

describe("finding a model card", () => {
  it("knows a repo id from a path on disk", () => {
    assert.equal(isRepoId("Qwen/Qwen3-4B"), true);
    assert.equal(isRepoId("/srv/models/qwen.gguf"), false);
    assert.equal(isRepoId("Qwen3.8-27B"), false);
    assert.equal(isRepoId(undefined), false);
  });

  it("reads `base_model` out of a card, as a string or a list", () => {
    assert.equal(baseModelOf({ cardData: { base_model: "Qwen/Qwen3-4B" } }), "Qwen/Qwen3-4B");
    assert.equal(baseModelOf({ cardData: { base_model: ["Qwen/Qwen3-4B"] } }), "Qwen/Qwen3-4B");
    assert.equal(baseModelOf({ cardData: {} }), undefined);
  });

  const ok = (body: unknown): Response => new Response(JSON.stringify(body), { status: 200 });
  const missing = (): Response => new Response("Entry not found", { status: 404 });

  it("takes the repo's own config when it has one", async () => {
    const calls: string[] = [];
    const result = await fetchGenerationConfig(DEFAULT_HUB_ENDPOINT, "Qwen/Qwen3-4B", {
      fetchImpl: async (url) => {
        calls.push(String(url));
        return ok(QWEN_GENERATION_CONFIG);
      },
    });
    assert.deepEqual(result.params, { temperature: 0.6, top_p: 0.95, top_k: 20 });
    assert.equal(result.from, "Qwen/Qwen3-4B");
    assert.equal(calls.length, 1);
  });

  it("follows `base_model` when the GGUF repo has none — the usual case", async () => {
    // Checked live: ggml-org/Qwen3-4B-GGUF and unsloth/Qwen3-4B-GGUF both 404
    // for generation_config.json, and both declare their base model.
    const calls: string[] = [];
    const result = await fetchGenerationConfig(DEFAULT_HUB_ENDPOINT, "ggml-org/Qwen3-4B-GGUF", {
      fetchImpl: async (url) => {
        const href = String(url);
        calls.push(href);
        if (href.endsWith("/ggml-org/Qwen3-4B-GGUF/resolve/main/generation_config.json")) return missing();
        if (href.endsWith("/api/models/ggml-org/Qwen3-4B-GGUF")) return ok({ cardData: { base_model: "Qwen/Qwen3-4B" } });
        return ok(QWEN_GENERATION_CONFIG);
      },
    });
    assert.deepEqual(result.params, { temperature: 0.6, top_p: 0.95, top_k: 20 });
    assert.equal(result.from, "Qwen/Qwen3-4B");
    assert.equal(calls.length, 3);
  });

  it("says so, rather than throwing, when nothing publishes anything", async () => {
    const result = await fetchGenerationConfig(DEFAULT_HUB_ENDPOINT, "someone/model", {
      fetchImpl: async (url) => (String(url).includes("/api/models/") ? ok({ cardData: {} }) : missing()),
    });
    assert.deepEqual(result.params, {});
    assert.match(result.detail ?? "", /does not publish/);
  });

  it("names gating as gating, which has a different fix from absence", async () => {
    // `google/gemma-3-27b-it` answers 401 with nobody signed in — checked live
    // on 2026-09-22 — and "does not publish" would send the user looking for
    // the wrong thing.
    const result = await fetchGenerationConfig(DEFAULT_HUB_ENDPOINT, "unsloth/gemma-3-27b-it-GGUF", {
      fetchImpl: async (url) => {
        const href = String(url);
        if (href.includes("/api/models/")) return ok({ cardData: { base_model: "google/gemma-3-27b-it" } });
        if (href.includes("/google/")) return new Response("Access to model ... is restricted.", { status: 401 });
        return missing();
      },
    });
    assert.match(result.detail ?? "", /gated on Hugging Face/);
    assert.match(result.detail ?? "", /google\/gemma-3-27b-it/);
  });

  it("sends a Hub token when there is one, and never puts it in the message", async () => {
    const saved = process.env["HF_TOKEN"];
    process.env["HF_TOKEN"] = "hf_secret_value";
    try {
      let sent: string | undefined;
      const result = await fetchGenerationConfig(DEFAULT_HUB_ENDPOINT, "meta/gated", {
        fetchImpl: async (_url, init) => {
          sent = new Headers(init?.headers).get("authorization") ?? undefined;
          return new Response("nope", { status: 403 });
        },
      });
      assert.equal(sent, "Bearer hf_secret_value");
      assert.doesNotMatch(result.detail ?? "", /hf_secret_value/);
      assert.match(result.detail ?? "", /does not grant access/);
    } finally {
      if (saved === undefined) delete process.env["HF_TOKEN"];
      else process.env["HF_TOKEN"] = saved;
    }
  });

  it("does not ask the Hub about a local file", async () => {
    let called = false;
    const result = await fetchGenerationConfig(DEFAULT_HUB_ENDPOINT, "/srv/models/qwen.gguf", {
      fetchImpl: async () => {
        called = true;
        return missing();
      },
    });
    assert.equal(called, false);
    assert.match(result.detail ?? "", /local file/);
  });

  it("turns a network failure into a sentence", async () => {
    const result = await fetchGenerationConfig(DEFAULT_HUB_ENDPOINT, "Qwen/Qwen3-4B", {
      fetchImpl: async () => {
        throw new Error("getaddrinfo ENOTFOUND huggingface.co");
      },
    });
    assert.match(result.detail ?? "", /ENOTFOUND/);
  });

  it("honours PI_OFFLINE without a request", async () => {
    const saved = process.env["PI_OFFLINE"];
    process.env["PI_OFFLINE"] = "1";
    try {
      let called = false;
      const result = await fetchGenerationConfig(DEFAULT_HUB_ENDPOINT, "Qwen/Qwen3-4B", {
        fetchImpl: async () => {
          called = true;
          return missing();
        },
      });
      assert.equal(called, false);
      assert.match(result.detail ?? "", /PI_OFFLINE/);
    } finally {
      if (saved === undefined) delete process.env["PI_OFFLINE"];
      else process.env["PI_OFFLINE"] = saved;
    }
  });
});

describe("the keys", () => {
  it("moves between the four fields and wraps", () => {
    let current = state();
    current = applySamplingKey(current, "\x1b[A").state;
    assert.equal(current.field, SAMPLING_KEYS.length - 1);
    current = applySamplingKey(current, "\x1b[B").state;
    assert.equal(current.field, 0);
  });

  it("adjusts the selected field and makes the values the user's", () => {
    const opened = state({ temperature: 0.6 }, { source: "hub", from: "Qwen/Qwen3-4B" });
    const after = applySamplingKey(opened, "\x1b[C").state;
    assert.equal(after.params.temperature, 0.65);
    assert.equal(after.source, "user");
    assert.equal(after.from, undefined);
  });

  it("clears the selected field with x, and forgets the source with the last one", () => {
    const after = applySamplingKey(state({ temperature: 0.6 }), "x").state;
    assert.equal(after.params.temperature, undefined);
    assert.equal(after.source, undefined);
  });

  it("reports save, fetch and cancel rather than doing them", () => {
    assert.equal(applySamplingKey(state(), "\r").outcome, "save");
    assert.equal(applySamplingKey(state(), "f").outcome, "fetch");
    assert.equal(applySamplingKey(state(), "\x1b").outcome, "cancel");
  });

  it("ignores adjustments while a fetch is in flight, but not escape", () => {
    const busy = state({ temperature: 0.6 }, { busy: true });
    assert.equal(applySamplingKey(busy, "\x1b[C").state.params.temperature, 0.6);
    assert.equal(applySamplingKey(busy, "\x1b").outcome, "cancel");
  });
});

describe("the screen", () => {
  it("draws a value per sampler, and says which are the server's", () => {
    const lines = fieldLines(state({ temperature: 0.6 }), 74, theme);
    assert.equal(lines.length, SAMPLING_KEYS.length);
    assert.match(lines[0] ?? "", /temperature\s+‹ 0\.60 ›/);
    assert.match(lines[1] ?? "", /top_p\s+‹ default ›\s+the server decides/);
  });

  it("names where the numbers came from", () => {
    assert.match(provenanceLine(state()), /nothing pinned/);
    assert.match(provenanceLine(state({ temperature: 0.6 })), /yours/);
    assert.match(
      provenanceLine(state({ temperature: 0.6 }, { source: "hub", from: "Qwen/Qwen3-4B" })),
      /from Qwen\/Qwen3-4B/,
    );
    assert.match(provenanceLine(state({ temperature: 0.6 }, { busy: true })), /reading the model card/);
  });

  it("fits its frame at every width it is drawn at", () => {
    for (const width of [50, 60, 74]) {
      for (const line of samplingLines(state({ temperature: 0.6, top_k: 20 }), width, theme)) {
        assert.ok(line.length <= width, `${line.length} > ${width}: ${line}`);
      }
    }
  });

  it("carries the model's name in the title, and its settings in one line", () => {
    assert.equal(samplingTitle(state()), "Sampling  Qwen3.8-27B");
    assert.match(samplingSummary(state({ temperature: 0.6, top_k: 20 })), /temperature 0\.60 · top_k 20/);
    assert.match(samplingSummary(state()), /the server decides/);
  });

  it("shows a fetch that found nothing, without losing what is on screen", () => {
    const shown = samplingLines(
      state({ temperature: 0.6 }, { note: "⚠ ggml-org/Qwen3-4B-GGUF does not publish a generation_config.json" }),
      74,
      theme,
    ).join("\n");
    assert.match(shown, /does not publish/);
    assert.match(shown, /0\.60/);
  });
});

describe("what Pi is published", () => {
  const entry = {
    id: "Qwen3.8-27B",
    displayName: "Qwen3.8-27B",
    quant: "Q8_0",
    loaded: false,
    contextLength: 196608,
    nativeContextLength: 262144,
    maxContextLength: undefined,
    isVision: false,
    supportsReasoning: false,
    reasoningEffortLevels: undefined,
  } as unknown as CatalogueInputs["entries"][number];

  const inputs = (sampling: CatalogueInputs["sampling"]): CatalogueInputs => ({
    baseUrl: "http://127.0.0.1:8888",
    entries: [entry],
    local: [],
    cached: [],
    overrides: {},
    status: undefined,
    remembered: [],
    ...(sampling ? { sampling } : {}),
  });

  it("carries the stored samplers onto the model", () => {
    const models = buildModels(inputs({ "Qwen3.8-27B": { params: { temperature: 0.6, top_k: 20 }, source: "user" } }));
    assert.deepEqual(models[0]?.samplingParams, { temperature: 0.6, top_k: 20 });
  });

  it("leaves the field off entirely when nothing is pinned", () => {
    assert.equal(buildModels(inputs(undefined))[0]?.samplingParams, undefined);
    assert.equal(buildModels(inputs({ "Qwen3.8-27B": { params: {}, source: "user" } }))[0]?.samplingParams, undefined);
  });

  it("ignores an entry for a model this server does not have", () => {
    const models = buildModels(inputs({ "some-other-model": { params: { temperature: 0.6 }, source: "user" } }));
    assert.equal(models[0]?.samplingParams, undefined);
  });
});

describe("the store on disk", () => {
  const AGENT_DIR_ENV = "PI_CODING_AGENT_DIR";
  let dir: string;
  let saved: string | undefined;

  before(() => {
    saved = process.env[AGENT_DIR_ENV];
    dir = mkdtempSync(join(tmpdir(), "pi-unsloth-sampling-"));
  });

  beforeEach(() => {
    process.env[AGENT_DIR_ENV] = dir;
  });

  afterEach(() => {
    rmSync(join(dir, PROFILE_FILENAME), { force: true });
  });

  after(() => {
    if (saved === undefined) delete process.env[AGENT_DIR_ENV];
    else process.env[AGENT_DIR_ENV] = saved;
    rmSync(dir, { recursive: true, force: true });
  });

  it("round-trips one model's settings", () => {
    const written = rememberSampling("Qwen3.8-27B", { params: { temperature: 0.6 }, source: "user" });
    assert.equal(written.ok, true);
    assert.deepEqual(readSamplingStore()["Qwen3.8-27B"]?.params, { temperature: 0.6 });
  });

  it("leaves other models, the policy and hand-added keys alone", () => {
    writeFileSync(
      join(dir, PROFILE_FILENAME),
      JSON.stringify({
        version: 1,
        gpus: [{ index: 0, display: true, headroomGiB: 3, displayEvidence: [] }],
        policy: { footer: false },
        mine: "keep me",
        sampling: { "other-model": { params: { top_k: 40 }, source: "user" } },
      }),
    );

    rememberSampling("Qwen3.8-27B", { params: { temperature: 0.6 }, source: "hub", from: "Qwen/Qwen3-27B" });

    const store = readSamplingStore();
    assert.deepEqual(store["other-model"]?.params, { top_k: 40 });
    assert.equal(store["Qwen3.8-27B"]?.from, "Qwen/Qwen3-27B");

    const raw = JSON.parse(readFileSync(join(dir, PROFILE_FILENAME), "utf8"));
    assert.equal(raw.mine, "keep me");
    assert.equal(raw.policy.footer, false);
    assert.equal(raw.gpus.length, 1);
  });

  it("removes an entry when the last sampler is cleared", () => {
    rememberSampling("Qwen3.8-27B", { params: { temperature: 0.6 }, source: "user" });
    rememberSampling("Qwen3.8-27B", undefined);
    assert.deepEqual(readSamplingStore(), {});
  });

  it("is empty, not an error, when there is no profile at all", () => {
    assert.deepEqual(readSamplingStore(), {});
  });
});
