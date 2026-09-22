/**
 * Model discovery and capability endpoints.
 *
 * Reference: and "Capabilities of the loaded model". Every parser here is
 * defensive: a field the server stops sending must degrade to `undefined`,
 * never throw, so one malformed entry cannot empty the whole catalogue.
 */

import type { RequestOptions, UnslothClient } from "./client.ts";

type Call = Pick<RequestOptions, "signal" | "timeoutMs">;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function str(value: unknown): string | undefined {
  return typeof value === "string" && value !== "" ? value : undefined;
}

function num(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function bool(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

/** One entry of `GET /api/inference/models` — a downloaded, usable model. */
export interface CatalogueEntry {
  id: string;
  displayName: string | undefined;
  quant: string | undefined;
  loaded: boolean;
  /** Configured context. Present for the loaded model, often absent otherwise. */
  contextLength: number | undefined;
  maxContextLength: number | undefined;
  nativeContextLength: number | undefined;
}

function parseCatalogueEntry(value: unknown): CatalogueEntry | undefined {
  if (!isRecord(value)) return undefined;
  const id = str(value["id"]);
  if (!id) return undefined;
  return {
    id,
    displayName: str(value["display_name"]),
    quant: str(value["quant"]),
    loaded: bool(value["loaded"]) ?? false,
    contextLength: num(value["context_length"]),
    maxContextLength: num(value["max_context_length"]),
    nativeContextLength: num(value["native_context_length"]),
  };
}

/** `GET /api/inference/models` — the catalogue feed for `refreshModels`. */
export async function listModels(client: UnslothClient, call: Call = {}): Promise<CatalogueEntry[]> {
  const body = await client.get<unknown>("/api/inference/models", call);
  const data = isRecord(body) ? body["data"] : undefined;
  if (!Array.isArray(data)) return [];
  return data.map(parseCatalogueEntry).filter((entry): entry is CatalogueEntry => entry !== undefined);
}

/** One entry of `GET /api/models/local` — the filesystem view. */
export interface LocalModel {
  id: string;
  displayName: string | undefined;
  /** Absolute path; what `estimate-memory`, `validate` and `load` call `model_path`. */
  path: string | undefined;
  /** `models_dir` | `hf_cache` | … */
  source: string | undefined;
  /** Hub repo id for cached models; null for models living in `models_dir`. */
  repoId: string | undefined;
  modelFormat: string | undefined;
  task: string | undefined;
  partial: boolean;
}

export interface LocalModels {
  modelsDir: string | undefined;
  hfCacheDir: string | undefined;
  models: LocalModel[];
}

function parseLocalModel(value: unknown): LocalModel | undefined {
  if (!isRecord(value)) return undefined;
  const id = str(value["id"]);
  if (!id) return undefined;
  return {
    id,
    displayName: str(value["display_name"]),
    path: str(value["path"]),
    source: str(value["source"]),
    repoId: str(value["model_id"]),
    modelFormat: str(value["model_format"]),
    task: str(value["task"]),
    partial: bool(value["partial"]) ?? false,
  };
}

/** `GET /api/models/local` — resolves a catalogue id to its `model_path`. */
export async function listLocalModels(client: UnslothClient, call: Call = {}): Promise<LocalModels> {
  const body = await client.get<unknown>("/api/models/local", call);
  const raw = isRecord(body) && Array.isArray(body["models"]) ? body["models"] : [];
  return {
    modelsDir: isRecord(body) ? str(body["models_dir"]) : undefined,
    hfCacheDir: isRecord(body) ? str(body["hf_cache_dir"]) : undefined,
    models: raw.map(parseLocalModel).filter((entry): entry is LocalModel => entry !== undefined),
  };
}

/** One entry of `GET /api/models/cached-gguf` — best source for `has_vision`. */
export interface CachedGguf {
  repoId: string;
  cachePath: string | undefined;
  sizeBytes: number | undefined;
  hasVision: boolean | undefined;
  task: string | undefined;
}

function parseCachedGguf(value: unknown): CachedGguf | undefined {
  if (!isRecord(value)) return undefined;
  const repoId = str(value["repo_id"]);
  if (!repoId) return undefined;
  return {
    repoId,
    cachePath: str(value["cache_path"]),
    sizeBytes: num(value["size_bytes"]),
    hasVision: bool(value["has_vision"]),
    task: str(value["task"]),
  };
}

/** `GET /api/models/cached-gguf` — HF-cache GGUF repos. */
export async function listCachedGguf(client: UnslothClient, call: Call = {}): Promise<CachedGguf[]> {
  const body = await client.get<unknown>("/api/models/cached-gguf", call);
  const cached = isRecord(body) ? body["cached"] : undefined;
  if (!Array.isArray(cached)) return [];
  return cached.map(parseCachedGguf).filter((entry): entry is CachedGguf => entry !== undefined);
}

/**
 * `GET /api/inference/status` — capabilities of the **currently loaded** model
 * only. The rest of the catalogue has no equivalent endpoint, which is why
 * observed capabilities are remembered (see src/provider.ts).
 */
export interface LoadedStatus {
  isVision: boolean | undefined;
  supportsReasoning: boolean | undefined;
  /** e.g. `["low","medium","high","xhigh"]`. Absent levels are unsupported. */
  reasoningEffortLevels: string[] | undefined;
  contextLength: number | undefined;
  maxContextLength: number | undefined;
  nativeContextLength: number | undefined;
  modelPath: string | undefined;
}

export async function getInferenceStatus(client: UnslothClient, call: Call = {}): Promise<LoadedStatus | undefined> {
  const body = await client.get<unknown>("/api/inference/status", call);
  if (!isRecord(body)) return undefined;
  const levels = body["reasoning_effort_levels"];
  return {
    isVision: bool(body["is_vision"]),
    supportsReasoning: bool(body["supports_reasoning"]),
    reasoningEffortLevels: Array.isArray(levels)
      ? levels.filter((level): level is string => typeof level === "string" && level !== "")
      : undefined,
    contextLength: num(body["context_length"]),
    maxContextLength: num(body["max_context_length"]),
    nativeContextLength: num(body["native_context_length"]),
    modelPath: str(body["model_path"]),
  };
}

/**
 * `GET /api/inference/loaded-models` — what is resident right now.
 *
 * Same entry shape as the catalogue. ⚠ Trust it only together with a process
 * check: Unsloth keeps reporting a model as active after `llama-server` has
 * died, so "is it loaded?" is two questions, not one
 * (src/process.ts, and SETTINGS.md in the Unsloth-Api repo).
 */
export async function listLoadedModels(client: UnslothClient, call: Call = {}): Promise<CatalogueEntry[]> {
  const body = await client.get<unknown>("/api/inference/loaded-models", call);
  const data = isRecord(body) ? (body["data"] ?? body["models"]) : undefined;
  if (!Array.isArray(data)) return [];
  return data.map(parseCatalogueEntry).filter((entry): entry is CatalogueEntry => entry !== undefined);
}
