import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import { describe, it } from "node:test";

import { UnslothClient } from "../src/api/client.ts";
import { describeUnavailable, estimateCaveat, estimateMemory, type MemoryEstimate } from "../src/api/estimate.ts";
import {
  buildLoadRequest,
  buildValidateRequest,
  mergeOverride,
  preflightLoad,
  putOverride,
  type ModelOverride,
} from "../src/api/lifecycle.ts";
import { bytesToGb } from "../src/hardware/budget.ts";

/**
 * A stand-in Unsloth, recording what it was asked.
 *
 * These tests are about the two things a wrapper can get wrong — the body it
 * sends and the *shape* of the answer it believes — so they go through the real
 * client and a real socket rather than a hand-made object.
 */
interface Stub {
  url: string;
  close: () => Promise<void>;
  bodies: unknown[];
  paths: string[];
}

async function startStub(reply: (path: string, body: unknown) => { status: number; body: string }): Promise<Stub> {
  const stub: Stub = { url: "", close: async () => {}, bodies: [], paths: [] };
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf8");
      let parsed: unknown;
      try {
        parsed = raw === "" ? undefined : JSON.parse(raw);
      } catch {
        parsed = raw;
      }
      stub.bodies.push(parsed);
      stub.paths.push(req.url ?? "/");
      const answer = reply(req.url ?? "/", parsed);
      res.writeHead(answer.status, { "Content-Type": "application/json" });
      res.end(answer.body);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (typeof address === "string" || address === null) throw new Error("no port");
  stub.url = `http://127.0.0.1:${address.port}`;
  stub.close = () => new Promise<void>((resolve) => server.close(() => resolve()));
  return stub;
}

async function withStub(
  reply: (path: string, body: unknown) => { status: number; body: string },
  run: (client: UnslothClient, stub: Stub) => Promise<void>,
): Promise<void> {
  const stub = await startStub(reply);
  try {
    await run(new UnslothClient({ baseUrl: stub.url, apiKey: "test-key" }), stub);
  } finally {
    await stub.close();
  }
}

/**
 * The documented 192 K reference estimate, verbatim.
 */
const REFERENCE_192K = {
  available: true,
  reason: null,
  weights_bytes: 29974693536,
  kv_bytes: 10610540544,
  compute_bytes: 2364504472,
  drafter_runtime_bytes: 2442484121,
  drafter_runtime_gpu_bytes: 2442484121,
  projector_runtime_bytes: 371042995,
  total_bytes: 45763265668,
  gpu_bytes: 42625401988,
  kv_estimable: true,
  kv_on_gpu: true,
  drafter_kv_unsized: false,
  adapters_unsized: false,
  moe_offload_unmodelled: false,
  n_ctx: 196608,
  cache_type_kv: "q8_0",
  n_parallel: 4,
  layer_count: 65,
  gpu_layers: null,
};

describe("estimateMemory", () => {
  it("reproduces the documented 192 K reference breakdown", async () => {
    await withStub(
      () => ({ status: 200, body: JSON.stringify(REFERENCE_192K) }),
      async (client, stub) => {
        const estimate = await estimateMemory(client, {
          model_path: "/models/local-model",
          gguf_variant: "Q8_0",
          n_ctx: 196608,
          cache_type_kv: "q8_0",
          speculative_type: "mtp",
          spec_draft_n_max: 2,
          tensor_parallel: true,
          selected_gpu_ids: [0, 1],
        });

        assert.equal(estimate.available, true);
        assert.equal(estimate.weightsBytes, 29974693536);
        assert.equal(estimate.kvBytes, 10610540544);
        assert.equal(estimate.computeBytes, 2364504472);
        // The two the hand-rolled estimator never counted.
        assert.equal(estimate.drafterRuntimeGpuBytes, 2442484121);
        assert.equal(estimate.projectorRuntimeBytes, 371042995);
        assert.equal(estimate.totalBytes, 45763265668);
        assert.equal(estimate.gpuBytes, 42625401988);
        assert.equal(bytesToGb(estimate.gpuBytes).toFixed(1), "39.7");
        assert.equal(estimate.nCtx, 196608);
        assert.equal(estimate.layerCount, 65);
        assert.equal(stub.paths[0], "/api/inference/estimate-memory");
      },
    );
  });

  it("sends the configuration it was asked about, and nothing it was not", async () => {
    await withStub(
      () => ({ status: 200, body: JSON.stringify(REFERENCE_192K) }),
      async (client, stub) => {
        await estimateMemory(client, { model_path: "/models/local-model", n_ctx: 8192 });
        assert.deepEqual(stub.bodies[0], { model_path: "/models/local-model", n_ctx: 8192 });
      },
    );
  });

  it("survives a payload with fields missing rather than emptying the screen", async () => {
    await withStub(
      () => ({ status: 200, body: JSON.stringify({ available: false, reason: "not_downloaded" }) }),
      async (client) => {
        const estimate = await estimateMemory(client, { model_path: "org/not-here" });
        assert.equal(estimate.available, false);
        assert.equal(estimate.gpuBytes, 0);
        assert.equal(estimate.nCtx, undefined);
        assert.match(describeUnavailable(estimate.reason), /not downloaded/);
      },
    );
  });
});

describe("describeUnavailable", () => {
  it("turns each documented code into a sentence", () => {
    for (const code of ["not_gguf", "not_downloaded", "unsupported_source", "unsizable"]) {
      assert.ok(describeUnavailable(code).length > 10, code);
      assert.doesNotMatch(describeUnavailable(code), /_/);
    }
  });

  it("passes an unknown code through rather than swallowing it", () => {
    assert.equal(describeUnavailable("something_new"), "something_new");
  });
});

describe("estimateCaveat", () => {
  const estimate = (overrides: Partial<MemoryEstimate>): MemoryEstimate =>
    ({
      available: true,
      reason: undefined,
      weightsBytes: 1,
      kvBytes: 1,
      computeBytes: 1,
      drafterRuntimeBytes: 0,
      drafterRuntimeGpuBytes: 0,
      projectorRuntimeBytes: 0,
      totalBytes: 3,
      gpuBytes: 3,
      kvEstimable: true,
      kvOnGpu: true,
      drafterKvUnsized: false,
      adaptersUnsized: false,
      moeOffloadUnmodelled: false,
      nCtx: 4096,
      cacheTypeKv: "q8_0",
      nParallel: 1,
      layerCount: 1,
      ...overrides,
    }) satisfies MemoryEstimate;

  it("says nothing when the breakdown is whole", () => {
    assert.equal(estimateCaveat(estimate({})), undefined);
  });

  it("names an unsized KV cache, the one absence that matters most", () => {
    assert.match(estimateCaveat(estimate({ kvEstimable: false })) ?? "", /KV cache/);
  });

  it("calls a total with an unsized drafter a floor", () => {
    assert.match(estimateCaveat(estimate({ drafterKvUnsized: true })) ?? "", /floor/);
  });
});

describe("buildValidateRequest", () => {
  const load = buildLoadRequest("/models/local-model", "Q8_0", {
    custom_context_length: 65536,
    kv_cache_dtype: "q8_0",
    speculative_type: "mtp",
    tensor_parallel: true,
    gpu_ids: [0, 1],
    n_parallel: 1,
  });

  it("asks about the configuration that will run", () => {
    const request = buildValidateRequest(load);
    assert.equal(request["max_seq_length"], 65536);
    assert.equal(request["cache_type_kv"], "q8_0");
    assert.equal(request["speculative_type"], "mtp");
    assert.deepEqual(request["gpu_ids"], [0, 1]);
  });

  it("leaves out the instructions that are about this load rather than the config", () => {
    const request = buildValidateRequest(load);
    assert.equal(request["force_reload"], undefined);
    assert.equal(request["force_cancel_active"], undefined);
  });

  it("asks for the header's context length only when told to", () => {
    assert.equal(buildValidateRequest(load)["include_context_length"], undefined);
    assert.equal(buildValidateRequest(load, { includeContextLength: true })["include_context_length"], true);
  });
});

describe("preflightLoad", () => {
  const request = buildLoadRequest("/models/local-model", "Q8_0", { gpu_ids: [7] });

  it("accepts a configuration the server calls valid, and carries what it learned", async () => {
    await withStub(
      () => ({
        status: 200,
        body: JSON.stringify({ valid: true, message: "Model identifier is valid.", context_length: 40960, layer_count: 36 }),
      }),
      async (client) => {
        const verdict = await preflightLoad(client, request, {}, { includeContextLength: true });
        assert.equal(verdict.checked, true);
        assert.equal(verdict.valid, true);
        assert.equal(verdict.contextLength, 40960);
        assert.equal(verdict.layerCount, 36);
      },
    );
  });

  it("refuses a configuration the server rejects, in the server's own words", async () => {
    // Verified live: a bad GPU id answers 400 with a *string* detail.
    await withStub(
      () => ({
        status: 400,
        body: JSON.stringify({ detail: "Invalid gpu_ids [7]: IDs must be physical GPU IDs between 0 and 2." }),
      }),
      async (client) => {
        const verdict = await preflightLoad(client, request);
        assert.equal(verdict.checked, true);
        assert.equal(verdict.valid, false);
        assert.match(verdict.message ?? "", /Invalid gpu_ids \[7\]/);
      },
    );
  });

  it("refuses a model the server has hard-blocked for security review", async () => {
    await withStub(
      () => ({ status: 200, body: JSON.stringify({ valid: true, message: "ok", requires_security_review: true }) }),
      async (client) => {
        const verdict = await preflightLoad(client, request);
        assert.equal(verdict.valid, false);
        assert.match(verdict.message ?? "", /security scan/);
      },
    );
  });

  it("does not block a load because *our* payload was wrong", async () => {
    // FastAPI's schema error: a list detail, not a string one. Refusing here
    // would break every load the day this payload's shape drifts.
    await withStub(
      () => ({
        status: 422,
        body: JSON.stringify({ detail: [{ type: "int_parsing", loc: ["body", "max_seq_length"] }] }),
      }),
      async (client) => {
        const verdict = await preflightLoad(client, request);
        assert.equal(verdict.checked, false);
        assert.equal(verdict.valid, true);
      },
    );
  });

  it("does not block a load on a server that cannot answer at all", async () => {
    for (const status of [404, 500, 503]) {
      await withStub(
        () => ({ status, body: JSON.stringify({ detail: "boom" }) }),
        async (client) => {
          const verdict = await preflightLoad(client, request);
          assert.equal(verdict.checked, false, `status ${status}`);
          assert.equal(verdict.valid, true, `status ${status}`);
        },
      );
    }
  });
});

describe("mergeOverride", () => {
  const stored: ModelOverride = {
    custom_context_length: 40960,
    kv_cache_dtype: "q8_0",
    n_batch: 2048,
    reasoning_budget: 1234,
  };

  it("keeps the fields sizing does not own — a PUT replaces the whole entry", () => {
    const merged = mergeOverride(stored, { custom_context_length: 65536, gpu_ids: [1] });
    assert.equal(merged["n_batch"], 2048);
    assert.equal(merged["reasoning_budget"], 1234);
    assert.equal(merged["custom_context_length"], 65536);
    assert.deepEqual(merged["gpu_ids"], [1]);
  });

  it("drops a field set back to the server's default", () => {
    const merged = mergeOverride(stored, { kv_cache_dtype: undefined });
    assert.equal("kv_cache_dtype" in merged, false);
  });

  it("never carries a control field across, or re-saving would delete the model", () => {
    const merged = mergeOverride({ ...stored, remove: true, model_id: "somewhere" }, {});
    assert.equal("remove" in merged, false);
    assert.equal("model_id" in merged, false);
  });
});

describe("putOverride", () => {
  it("writes one model, keyed the way the server keys it, and reads back the map", async () => {
    const overrides: Record<string, unknown> = { "org/other-model:Q4_K_M": { custom_context_length: 8192 } };
    await withStub(
      (_path, body) => {
        const payload = body as Record<string, unknown>;
        const key = payload["model_id"] as string;
        const entry: Record<string, unknown> = { ...payload };
        delete entry["model_id"];
        overrides[key] = entry;
        return { status: 200, body: JSON.stringify({ overrides }) };
      },
      async (client, stub) => {
        const result = await putOverride(client, "/models/local-model:Q8_0", {
          custom_context_length: 65536,
          gpu_ids: [1],
        });

        assert.deepEqual(stub.bodies[0], {
          model_id: "/models/local-model:Q8_0",
          custom_context_length: 65536,
          gpu_ids: [1],
        });
        // The endpoint takes one model, so the neighbour survives by construction.
        assert.ok(result["org/other-model:Q4_K_M"]);
        assert.equal(result["/models/local-model:Q8_0"]?.["custom_context_length"], 65536);
      },
    );
  });

  it("tolerates a reply that carries no map rather than claiming everything vanished", async () => {
    await withStub(
      () => ({ status: 200, body: JSON.stringify({ ok: true }) }),
      async (client) => {
        assert.deepEqual(await putOverride(client, "a:Q8_0", { custom_context_length: 1 }), {});
      },
    );
  });
});
