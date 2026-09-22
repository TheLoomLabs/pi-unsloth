/**
 * Per-model sampling defaults — the one thing the server cannot tell us.
 *
 * Everything else this extension publishes about a model is *measured*: the
 * context comes from the override, the capabilities from
 * `/api/inference/models`, the memory from the estimator. Sampling is
 * different. The server's OpenAPI carries `temperature`, `top_p`, `top_k` and
 * `min_p` in exactly one role — fields of `ChatCompletionRequest`, things a
 * client **sends** — and no endpoint anywhere reports what a given model would
 * like them to be. The auto-switch override has no slot for them either: its
 * payload is a fixed field list.
 *
 * So they are ours to keep. Two ways in, and the file records which:
 *
 *   - **`user`** — typed on the sampling screen. The user's taste, and the
 *     final word.
 *   - **`hub`** — read from the model's own `generation_config.json` on Hugging
 *     Face (`src/api/hub.ts`). A starting point, not an authority: it is
 *     overwritten the moment the user adjusts anything.
 *
 * This file is pure. The store lives in `src/hardware/profile.ts`, which owns
 * `~/.pi/agent/unsloth.json`, for the same reason everything else about that
 * file goes through one place.
 */

/**
 * The samplers offered.
 *
 * Not a taste, and not a fact about anyone's hardware: these are the four
 * fields the **server's own** `ChatCompletionRequest` schema accepts. Pi merges
 * `samplingParams` into the request body verbatim, so a key the server does not
 * model would be sent and rejected — which is why this list is the server's
 * list rather than a wishlist.
 */
export type SamplingKey = "temperature" | "top_p" | "top_k" | "min_p";

export const SAMPLING_KEYS: readonly SamplingKey[] = ["temperature", "top_p", "top_k", "min_p"];

/** A sampler's range, its step, and where `←→` starts it from when it is unset. */
export interface SamplingField {
  min: number;
  max: number;
  step: number;
  /** Decimal places shown and stored. `0` for an integer sampler. */
  places: number;
  /**
   * The value the first keypress materialises.
   *
   * llama.cpp's own defaults, so pressing `→` on an unset field writes down
   * what the server was already doing rather than introducing a change the
   * user did not ask for.
   */
  start: number;
}

export const SAMPLING_FIELDS: Readonly<Record<SamplingKey, SamplingField>> = {
  temperature: { min: 0, max: 2, step: 0.05, places: 2, start: 0.8 },
  top_p: { min: 0, max: 1, step: 0.05, places: 2, start: 0.95 },
  top_k: { min: 0, max: 200, step: 1, places: 0, start: 40 },
  min_p: { min: 0, max: 1, step: 0.01, places: 2, start: 0.05 },
};

/** What Pi is handed as `Model.samplingParams`. Absent keys are not sent. */
export type SamplingParams = Partial<Record<SamplingKey, number>>;

export type SamplingSource = "user" | "hub";

export interface SamplingEntry {
  params: SamplingParams;
  source: SamplingSource;
  /** Where a `hub` entry came from — the repo, so the screen can name it. */
  from?: string;
  fetchedAt?: string;
  /** A hand edit is owed its keys back, like everywhere else in the profile. */
  [key: string]: unknown;
}

/** Round to a field's own precision. Float steps otherwise drift visibly. */
export function roundTo(value: number, places: number): number {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

/**
 * One sampler, cleaned.
 *
 * Out of range is dropped rather than clamped: a `top_p` of 4 in a hand-edited
 * profile is a mistake, and silently turning it into 1 hides the mistake
 * instead of the value.
 */
export function normaliseValue(key: SamplingKey, value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) return undefined;
  const field = SAMPLING_FIELDS[key];
  if (value < field.min || value > field.max) return undefined;
  return roundTo(value, field.places);
}

/** A whole set, cleaned. Unknown keys are dropped — the server would reject them. */
export function normaliseParams(raw: unknown): SamplingParams {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return {};
  const source = raw as Record<string, unknown>;
  const params: SamplingParams = {};
  for (const key of SAMPLING_KEYS) {
    const value = normaliseValue(key, source[key]);
    if (value !== undefined) params[key] = value;
  }
  return params;
}

/** A stored entry, cleaned. `undefined` when nothing usable survives. */
export function normaliseEntry(raw: unknown): SamplingEntry | undefined {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return undefined;
  const source = raw as Record<string, unknown>;
  // A bare `{ "temperature": 0.7 }` is what a hand edit looks like before
  // anyone has read the schema, and it means exactly what it says.
  const params = normaliseParams(source["params"] ?? source);
  if (Object.keys(params).length === 0) return undefined;
  const entry: SamplingEntry = { ...source, params, source: source["source"] === "hub" ? "hub" : "user" };
  const from = typeof source["from"] === "string" && source["from"] !== "" ? source["from"] : undefined;
  if (from === undefined) delete entry.from;
  else entry.from = from;
  return entry;
}

export type SamplingStore = Record<string, SamplingEntry>;

/** Every entry in a stored `sampling` block, cleaned. */
export function normaliseStore(raw: unknown): SamplingStore {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return {};
  const store: SamplingStore = {};
  for (const [modelId, value] of Object.entries(raw as Record<string, unknown>)) {
    const entry = normaliseEntry(value);
    if (entry) store[modelId] = entry;
  }
  return store;
}

/**
 * Move one sampler by one step.
 *
 * From unset, either direction materialises the field's `start` rather than
 * jumping to an end of its range: the first press is "give me a value to work
 * from", and only the second is an opinion about which way.
 */
export function adjustParam(params: SamplingParams, key: SamplingKey, direction: 1 | -1): SamplingParams {
  const field = SAMPLING_FIELDS[key];
  const current = params[key];
  const wanted = current === undefined ? field.start : current + direction * field.step;
  const clamped = Math.min(field.max, Math.max(field.min, wanted));
  return { ...params, [key]: roundTo(clamped, field.places) };
}

/** Drop one sampler, which is different from setting it to zero. */
export function clearParam(params: SamplingParams, key: SamplingKey): SamplingParams {
  const next = { ...params };
  delete next[key];
  return next;
}

/** `0.95`, `20`, or `default` for a sampler this model does not pin. */
export function formatParam(params: SamplingParams, key: SamplingKey): string {
  const value = params[key];
  if (value === undefined) return "default";
  return value.toFixed(SAMPLING_FIELDS[key].places);
}

/** Has the user pinned anything at all? An empty set is stored as nothing. */
export function hasParams(params: SamplingParams): boolean {
  return SAMPLING_KEYS.some((key) => params[key] !== undefined);
}

/**
 * A Hugging Face `generation_config.json`, reduced to the four samplers.
 *
 * `do_sample: false` is the card saying the model is meant to be decoded
 * greedily; the temperature beside it is then not a recommendation but a
 * leftover, so nothing is taken from it. Everything else in the file —
 * token ids, `transformers_version` — is not ours.
 */
export function parseGenerationConfig(raw: unknown): SamplingParams {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return {};
  const config = raw as Record<string, unknown>;
  if (config["do_sample"] === false) return {};
  return normaliseParams(config);
}
