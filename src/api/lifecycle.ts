/**
 * Load / unload / validate, and the per-model tuned settings that drive them.
 *
 * Reference: and "Auto-switch overrides". M2 added the load lifecycle on top of
 * M1's override *read*; M5 adds the two halves that only make sense once the
 * extension can size a model — the pre-flight that every load now passes
 * through, and the override **write** that is sizing's output.
 */

import { UnslothApiError, type RequestOptions, type UnslothClient } from "./client.ts";

type Call = Pick<RequestOptions, "signal" | "timeoutMs">;

/**
 * A tuned per-model entry of `/api/settings/openai-auto-switch/overrides`.
 * Field names mirror the server's payload exactly so they round-trip on write
 * without a translation layer to keep in sync.
 */
export interface ModelOverride {
  custom_context_length?: number;
  kv_cache_dtype?: string;
  speculative_type?: string;
  spec_draft_n_max?: number;
  n_parallel?: number;
  n_batch?: number;
  n_ubatch?: number;
  tensor_parallel?: boolean;
  gpu_ids?: number[];
  /** Anything the server adds later is preserved verbatim on write-back. */
  [key: string]: unknown;
}

/** Keyed by `"<model_path|repo_id>:<QUANT>"`. */
export type Overrides = Record<string, ModelOverride>;

/**
 * `GET /api/settings/openai-auto-switch/overrides`.
 *
 * Never throws for an absent or malformed body — an empty map just means the
 * context ladder falls through to the next rung.
 */
export async function getOverrides(client: UnslothClient, call: Call = {}): Promise<Overrides> {
  const body = await client.get<unknown>("/api/settings/openai-auto-switch/overrides", call);
  const raw = typeof body === "object" && body !== null ? (body as Record<string, unknown>)["overrides"] : undefined;
  if (typeof raw !== "object" || raw === null) return {};
  const result: Overrides = {};
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof value === "object" && value !== null && !Array.isArray(value)) {
      result[key] = value as ModelOverride;
    }
  }
  return result;
}

/**
 * The override key for a model.
 *
 * Verified against a live server: a `models_dir` model is keyed by its
 * absolute path, a hub model by its repo id — i.e. `repoId ?? path`.
 */
export function overrideKey(pathOrRepoId: string, quant: string | undefined): string {
  return quant ? `${pathOrRepoId}:${quant}` : pathOrRepoId;
}

/**
 * Find a model's override by trying every plausible key.
 *
 * Guessing wrong is free — the caller just falls to the next rung of the
 * ladder — so several candidates are tried rather than one clever one.
 */
export function findOverride(
  overrides: Overrides,
  candidates: readonly (string | undefined)[],
  quant: string | undefined,
): ModelOverride | undefined {
  for (const candidate of candidates) {
    if (!candidate) continue;
    const exact = overrides[overrideKey(candidate, quant)];
    if (exact) return exact;
  }
  // Quant-less fallback: an entry written before the server keyed by quant.
  for (const candidate of candidates) {
    if (!candidate) continue;
    const bare = overrides[candidate];
    if (bare) return bare;
  }
  return undefined;
}

/**
 * A `POST /api/inference/load` body.
 *
 * Only `model_path` is required; every other field comes from the model's
 * tuned override. Index signature: the server owns this payload's vocabulary,
 * and a field it gains later must be passable without a code change here.
 */
export interface LoadRequest {
  model_path: string;
  gguf_variant?: string;
  force_reload?: boolean;
  force_cancel_active?: boolean;
  [key: string]: unknown;
}

/**
 * Override field → load field. They are not all the same name, and the two
 * that differ are exactly the two that matter most.
 *
 * Anything absent from the override is absent from the request: the server's
 * own default is a better guess than ours.
 */
const LOAD_FIELDS: ReadonlyArray<readonly [keyof ModelOverride & string, string]> = [
  ["custom_context_length", "max_seq_length"],
  ["kv_cache_dtype", "cache_type_kv"],
  ["speculative_type", "speculative_type"],
  ["spec_draft_n_max", "spec_draft_n_max"],
  ["tensor_parallel", "tensor_parallel"],
  ["n_parallel", "n_parallel"],
  ["n_batch", "n_batch"],
  ["n_ubatch", "n_ubatch"],
  ["gpu_ids", "gpu_ids"],
  ["disable_vision", "disable_vision"],
];

/**
 * Build an **explicit** load from a tuned override.
 *
 * This exists because auto-switch — Unsloth loading a model because a request
 * named it — silently drops `--spec-type draft-mtp` even when the override asks
 * for it, costing the whole speculative-decoding win. An explicit load applies
 * it. `force_reload` and `force_cancel_active` make the call authoritative over
 * whatever auto-switch may have loaded a moment earlier.
 */
export function buildLoadRequest(
  modelPath: string,
  quant: string | undefined,
  override: ModelOverride | undefined,
): LoadRequest {
  const request: LoadRequest = { model_path: modelPath, force_reload: true, force_cancel_active: true };
  if (quant) request.gguf_variant = quant;
  for (const [from, to] of LOAD_FIELDS) {
    const value = override?.[from];
    if (value !== undefined) request[to] = value;
  }
  return request;
}

/**
 * `POST /api/inference/load`.
 *
 * Deliberately has no default timeout of its own: a cold 27B load is minutes of
 * disk and VRAM traffic, so the caller states the budget it is prepared to wait
 * and supplies the signal that cancels it.
 */
export async function loadModel(client: UnslothClient, request: LoadRequest, call: Call): Promise<void> {
  await client.post<unknown>("/api/inference/load", request, call);
}

/**
 * `POST /api/inference/unload`.
 *
 * ⚠ This clears Unsloth's "active model" state but **does not free VRAM** —
 * `llama-server` keeps running with the weights resident. It is half of the
 * unload; the other half is terminating that process (src/process.ts). Calling
 * this first is what stops the supervisor respawning it.
 */
export async function unloadModel(client: UnslothClient, modelPath: string, call: Call = {}): Promise<void> {
  await client.post<unknown>("/api/inference/unload", { model_path: modelPath }, call);
}

/** `GET /api/inference/load-progress`. Drives the footer and the M3 overlay. */
export interface LoadProgress {
  /** `resolving` → `loading` → `warming up` → `ready`, per the server. */
  phase: string | undefined;
  bytesLoaded: number | undefined;
  bytesTotal: number | undefined;
  /** 0–1. Absent until the server knows the total. */
  fraction: number | undefined;
}

export async function getLoadProgress(client: UnslothClient, call: Call = {}): Promise<LoadProgress | undefined> {
  const body = await client.get<unknown>("/api/inference/load-progress", call);
  if (typeof body !== "object" || body === null) return undefined;
  const raw = body as Record<string, unknown>;
  const numeric = (value: unknown): number | undefined =>
    typeof value === "number" && Number.isFinite(value) ? value : undefined;
  return {
    phase: typeof raw["phase"] === "string" && raw["phase"] !== "" ? (raw["phase"] as string) : undefined,
    bytesLoaded: numeric(raw["bytes_loaded"]),
    bytesTotal: numeric(raw["bytes_total"]),
    fraction: numeric(raw["fraction"]),
  };
}

/* --------------------------------------------------------------------------
 * Pre-flight
 * ------------------------------------------------------------------------ */

/**
 * The fields `POST /api/inference/validate` shares with a load.
 *
 * It takes the load's own vocabulary, which is the point: the config that is
 * checked has to be the config that runs, or the check is theatre.
 */
const VALIDATE_FIELDS: readonly string[] = [
  "model_path",
  "gguf_variant",
  "max_seq_length",
  "cache_type_kv",
  "tensor_parallel",
  "gpu_ids",
  "n_parallel",
  "n_batch",
  "n_ubatch",
  "speculative_type",
  "spec_draft_n_max",
  "spec_draft_cache_type",
  "ctx_checkpoints",
  "disable_vision",
  "gpu_memory_mode",
  "gpu_layers",
  "llama_extra_args",
];

/**
 * A validate body from a load body.
 *
 * `force_reload` and `force_cancel_active` are deliberately not carried: they
 * are instructions about *this* load, not properties of the configuration, and
 * this endpoint is being asked about the configuration.
 */
export function buildValidateRequest(
  load: LoadRequest,
  options: { includeContextLength?: boolean } = {},
): Record<string, unknown> {
  const request: Record<string, unknown> = {};
  for (const field of VALIDATE_FIELDS) {
    const value = load[field];
    if (value !== undefined) request[field] = value;
  }
  request["model_path"] = load.model_path;
  // The native context, read from the GGUF header. Opt-in because it costs a
  // header read, and it is the only free way to learn the ceiling for a model
  // that has never been loaded.
  if (options.includeContextLength) request["include_context_length"] = true;
  return request;
}

/** What the pre-flight learned. `checked: false` means it could not be asked. */
export interface ValidationVerdict {
  /** The server gave a verdict. When false, everything below is a non-answer. */
  checked: boolean;
  valid: boolean;
  /** The server's own sentence, which is the only one worth showing. */
  message: string | undefined;
  /** Native context from the GGUF header, with `includeContextLength`. */
  contextLength: number | undefined;
  layerCount: number | undefined;
  /** These weights are the ones already loaded. */
  resident: boolean | undefined;
  isVision: boolean | undefined;
}

/**
 * `POST /api/inference/validate` — the pre-flight before every load.
 *
 * Never throws, and — the rule this function turns on — **never refuses a load
 * because we could not ask.** A rejection and a misunderstanding arrive on the
 * same wire, and telling them apart is the difference between catching a bad
 * config and blocking every load the day the payload shape drifts:
 *
 *   - `200` with `valid: false` → the server refused the configuration;
 *   - a `4xx` whose `detail` is a **string** → an application refusal, in the
 *     server's own words (`Invalid gpu_ids [7]: …`, observed);
 *   - a `4xx` whose `detail` is a **list** → FastAPI rejecting our *payload*,
 *     not the user's config, so the load proceeds unchecked;
 *   - `5xx`, a timeout, or a server without the endpoint → unchecked.
 *
 * A model flagged for security review is a refusal of its own: the server
 * hard-blocks the load, so saying so here beats failing later with less
 * context.
 */
export async function preflightLoad(
  client: UnslothClient,
  request: LoadRequest,
  call: Call = {},
  options: { includeContextLength?: boolean } = {},
): Promise<ValidationVerdict> {
  const unchecked: ValidationVerdict = {
    checked: false,
    valid: true,
    message: undefined,
    contextLength: undefined,
    layerCount: undefined,
    resident: undefined,
    isVision: undefined,
  };

  let body: unknown;
  try {
    body = await client.post<unknown>("/api/inference/validate", buildValidateRequest(request, options), call);
  } catch (error) {
    if (!(error instanceof UnslothApiError) || error.status >= 500 || error.status === 404) return unchecked;
    // A list `detail` is FastAPI describing *our* payload; refusing the load on
    // that would punish the user for our own drift.
    const detail = error.body && typeof error.body === "object" ? (error.body as { detail?: unknown }).detail : undefined;
    if (Array.isArray(detail)) return unchecked;
    if (error.isAuthError) return unchecked;
    return { ...unchecked, checked: true, valid: false, message: error.reason ?? error.message };
  }

  if (typeof body !== "object" || body === null) return unchecked;
  const raw = body as Record<string, unknown>;
  const message = typeof raw["message"] === "string" && raw["message"] !== "" ? (raw["message"] as string) : undefined;
  const blocked = raw["requires_security_review"] === true;
  const number = (key: string): number | undefined => {
    const value = raw[key];
    return typeof value === "number" && Number.isFinite(value) ? value : undefined;
  };
  const flag = (key: string): boolean | undefined => (typeof raw[key] === "boolean" ? (raw[key] as boolean) : undefined);

  return {
    checked: true,
    valid: raw["valid"] !== false && !blocked,
    message: blocked ? "Hugging Face's security scan flagged this model's files" : message,
    contextLength: number("context_length"),
    layerCount: number("layer_count"),
    resident: flag("resident"),
    isVision: flag("is_vision"),
  };
}

/* --------------------------------------------------------------------------
 * Writing an override (the output of sizing)
 * ------------------------------------------------------------------------ */

/**
 * Fields that steer the *write* rather than describing the model.
 *
 * They must never be carried across from a stored entry, or re-saving a model
 * would delete it.
 */
const OVERRIDE_CONTROL_FIELDS: readonly string[] = ["model_id", "remove", "fill_absent_fields"];

/**
 * The entry to store, from the entry already stored and the fields being set.
 *
 * ⚠ **A PUT replaces the model's whole entry**, verified against a live server:
 * sending only `custom_context_length` left an entry with nothing else in it,
 * and sending nothing effective deleted it. So "preserve what we did not
 * touch" is this function's job, not the endpoint's — the endpoint's own
 * guarantee is only that *other models* are left alone.
 *
 * A field set to `undefined` in `changes` is dropped rather than written, which
 * is how the sizer says "leave this to the server's default".
 */
export function mergeOverride(
  existing: ModelOverride | undefined,
  // A plain record, not a `ModelOverride`: the point of this function is that a
  // field may arrive as `undefined` meaning "unset it", which is exactly what
  // an optional property may not hold.
  changes: Readonly<Record<string, unknown>>,
): ModelOverride {
  const merged: ModelOverride = { ...existing };
  for (const field of OVERRIDE_CONTROL_FIELDS) delete merged[field];
  for (const [key, value] of Object.entries(changes)) {
    if (value === undefined) delete merged[key];
    else merged[key] = value;
  }
  return merged;
}

/**
 * `PUT /api/settings/openai-auto-switch/overrides` for one model.
 *
 * The payload is one model's settings plus the key it is stored under — not the
 * whole map — so every other model's entry survives by construction rather than
 * by us reading, merging and writing back a file we do not own.
 *
 * `key` is `"<model_path|repo_id>:<QUANT>"`, exactly what `overrideKey` builds
 * and what the server stores: it is used verbatim as the map key (verified).
 * Returns the map the server now holds, so a caller can check its neighbours.
 */
export async function putOverride(
  client: UnslothClient,
  key: string,
  entry: ModelOverride,
  call: Call = {},
): Promise<Overrides> {
  const payload: Record<string, unknown> = { model_id: key };
  for (const [field, value] of Object.entries(entry)) {
    if (OVERRIDE_CONTROL_FIELDS.includes(field) || value === undefined) continue;
    payload[field] = value;
  }
  const body = await client.put<unknown>("/api/settings/openai-auto-switch/overrides", payload, call);
  const raw = typeof body === "object" && body !== null ? (body as Record<string, unknown>)["overrides"] : undefined;
  if (typeof raw !== "object" || raw === null) return {};
  const result: Overrides = {};
  for (const [name, value] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof value === "object" && value !== null && !Array.isArray(value)) result[name] = value as ModelOverride;
  }
  return result;
}
