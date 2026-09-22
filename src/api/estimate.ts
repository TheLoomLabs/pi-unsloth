/**
 * `POST /api/inference/estimate-memory` — Unsloth's own memory estimator.
 *
 * This is the endpoint that deletes `register-model.py`: GGUF header parsing,
 * the KV-bytes-per-element table and the `overhead = 3000` constant all become
 * one request to a server that already knows the backend, the quantisation and
 * the drafter. Reference:.
 *
 * Two properties make the search in src/sizing/search.ts possible at all:
 *
 *   - it is a **pure calculation** — nothing is loaded, no device is touched,
 *     nothing is downloaded — so it is safe to call in a loop;
 *   - it prices *this* configuration, so the answer moves with the context, the
 *     KV dtype and the placement, which is what a binary search needs.
 *
 * ⚠ And one property makes it dangerous to use naively: `available: true` is a
 * system-wide feasibility verdict that permits CPU offload, **not** a per-GPU
 * fit check. The fit rule is the extension's own (src/hardware/budget.ts).
 */

import type { RequestOptions, UnslothClient } from "./client.ts";

type Call = Pick<RequestOptions, "signal" | "timeoutMs">;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function num(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function str(value: unknown): string | undefined {
  return typeof value === "string" && value !== "" ? value : undefined;
}

/**
 * The request body.
 *
 * Field names are the server's own so they read against the OpenAPI spec
 * without a translation table, and the index signature lets a field the server
 * gains later be passed without a change here.
 */
export interface EstimateRequest {
  model_path: string;
  gguf_variant?: string;
  /** Omitted or `0` prices the model's **native** context, which is how the
   * ceiling is discovered for a model that has never been loaded. */
  n_ctx?: number;
  cache_type_kv?: string;
  n_parallel?: number;
  n_batch?: number;
  n_ubatch?: number;
  speculative_type?: string;
  spec_draft_n_max?: number;
  spec_draft_cache_type?: string;
  tensor_parallel?: boolean;
  /** Tensor mode replicates compute buffers per device, so the count matters. */
  selected_gpu_ids?: number[];
  disable_vision?: boolean;
  [key: string]: unknown;
}

/**
 * The breakdown, in the terms shows it.
 *
 * `drafterRuntimeGpuBytes` and `projectorRuntimeBytes` are the two the
 * hand-rolled estimator never counted — 2.44 GiB of speculative decoding state
 * and 0.37 GiB of vision tower on the reference box — which is exactly how it
 * came to over-report the context that would fit.
 */
export interface MemoryEstimate {
  /** A breakdown could be produced at all. **Not** a per-GPU fit check. */
  available: boolean;
  /** `not_gguf` | `not_downloaded` | `unsupported_source` | `unsizable`. */
  reason: string | undefined;
  weightsBytes: number;
  kvBytes: number;
  computeBytes: number;
  drafterRuntimeBytes: number;
  drafterRuntimeGpuBytes: number;
  projectorRuntimeBytes: number;
  totalBytes: number;
  /** The share of `totalBytes` that lands on the GPU — what the fit rule uses. */
  gpuBytes: number;
  /** False when the header lacks the dims: `kvBytes` is then 0 meaning UNKNOWN. */
  kvEstimable: boolean;
  kvOnGpu: boolean;
  /** A drafter is charged whose cache could not be sized — the total is a floor. */
  drafterKvUnsized: boolean;
  adaptersUnsized: boolean;
  moeOffloadUnmodelled: boolean;
  /** The context actually priced. With `n_ctx` omitted, the model's native one. */
  nCtx: number | undefined;
  /** The dtype actually priced, after flags and fallbacks resolve. */
  cacheTypeKv: string | undefined;
  nParallel: number | undefined;
  layerCount: number | undefined;
}

function parseEstimate(body: unknown): MemoryEstimate {
  const raw = isRecord(body) ? body : {};
  const bytes = (key: string): number => num(raw[key]) ?? 0;
  return {
    available: raw["available"] === true,
    reason: str(raw["reason"]),
    weightsBytes: bytes("weights_bytes"),
    kvBytes: bytes("kv_bytes"),
    computeBytes: bytes("compute_bytes"),
    drafterRuntimeBytes: bytes("drafter_runtime_bytes"),
    drafterRuntimeGpuBytes: bytes("drafter_runtime_gpu_bytes"),
    projectorRuntimeBytes: bytes("projector_runtime_bytes"),
    totalBytes: bytes("total_bytes"),
    gpuBytes: bytes("gpu_bytes"),
    kvEstimable: raw["kv_estimable"] !== false,
    kvOnGpu: raw["kv_on_gpu"] !== false,
    drafterKvUnsized: raw["drafter_kv_unsized"] === true,
    adaptersUnsized: raw["adapters_unsized"] === true,
    moeOffloadUnmodelled: raw["moe_offload_unmodelled"] === true,
    nCtx: num(raw["n_ctx"]),
    cacheTypeKv: str(raw["cache_type_kv"]),
    nParallel: num(raw["n_parallel"]),
    layerCount: num(raw["layer_count"]),
  };
}

/** `POST /api/inference/estimate-memory`. Allocates nothing; safe in a loop. */
export async function estimateMemory(
  client: UnslothClient,
  request: EstimateRequest,
  call: Call = {},
): Promise<MemoryEstimate> {
  return parseEstimate(await client.post<unknown>("/api/inference/estimate-memory", request, call));
}

/**
 * The estimator's refusal codes, as sentences.
 *
 * They are codes, not prose (`not_downloaded`, `unsizable`), so something has
 * to turn them into a line a user can act on — and an unknown code is passed
 * through rather than swallowed, because a word from the server beats a word
 * from us.
 */
export function describeUnavailable(reason: string | undefined): string {
  switch (reason) {
    case "not_gguf":
      return "not a GGUF model — Unsloth can only size llama.cpp models";
    case "not_downloaded":
      return "not downloaded — nothing on this disk to measure";
    case "unsupported_source":
      return "this model's source cannot be sized";
    case "unsizable":
      return "the GGUF header does not carry the dimensions needed to size it";
    default:
      return reason ?? "the server could not produce a breakdown";
  }
}

/**
 * Why an estimate cannot be trusted as a whole number, when it cannot.
 *
 * Three flags say "this total is a lower bound", and a lower bound is the one
 * thing a sizing decision must not silently round off: `kv_estimable: false`
 * means the cache — the term that outgrows the weights at long context — is
 * reported as zero *meaning unknown*. Sizing against that would hand back a
 * context the machine cannot hold.
 */
export function estimateCaveat(estimate: MemoryEstimate): string | undefined {
  if (!estimate.kvEstimable) return "the KV cache could not be sized from this model's header";
  if (estimate.drafterKvUnsized) return "the drafter's cache could not be sized — the total is a floor";
  if (estimate.adaptersUnsized) return "an adapter could not be sized — the total is a floor";
  if (estimate.moeOffloadUnmodelled) return "expert offload is not modelled in this split";
  return undefined;
}
