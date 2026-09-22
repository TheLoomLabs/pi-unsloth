/**
 * Hugging Face, read-only: where a model's *own* sampling recommendation lives.
 *
 * This is the one file that talks to something other than the Unsloth server,
 * and it exists because the server has nothing to say on the subject. A model's
 * publisher does: `generation_config.json` in the model repo is where
 * `temperature: 0.6, top_k: 20, top_p: 0.95` comes from for Qwen3, and it is a
 * file, not a table someone typed into this extension.
 *
 * Three rules, all of them the same rule in different clothes — **this is a
 * convenience, never a dependency**:
 *
 *   - Nothing here is ever called on a session path. It runs on `f`, from a
 *     screen the user opened, and nowhere else.
 *   - `PI_OFFLINE` means no. Pi's own switch, honoured without a request.
 *   - Every failure is a *sentence*, not an exception. "Not published" is the
 *     common case, not an error: GGUF repos almost never carry the file.
 *
 * The endpoint is the **server's** (`hf_endpoint` from `/api/health`), not a
 * constant here: a machine pointed at a mirror is pointed at it for a reason.
 */

import { type SamplingParams, parseGenerationConfig } from "../sampling.ts";
import type { UnslothClient } from "./client.ts";

/**
 * Environment variables a Hub token may arrive in, in the order the Hub's own
 * tooling reads them.
 *
 * Read only, never stored, never printed: a gated repo (Gemma, Llama) answers
 * `401` without one, and the alternative to sending it is telling the user to
 * type four numbers the publisher already wrote down.
 */
const TOKEN_VARS = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"] as const;

/** Hugging Face itself, used only when the server does not name an endpoint. */
export const DEFAULT_HUB_ENDPOINT = "https://huggingface.co";

/** One Hub read may not hang a screen the user is looking at. */
const HUB_TIMEOUT_MS = 8_000;

export interface HubOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
  /** Injected in tests, and by nothing else. */
  fetchImpl?: typeof fetch;
}

/**
 * The Hub the *server* is configured against.
 *
 * `/api/health` is anonymous and always cheap, and it carries `hf_endpoint`
 * because Unsloth downloads through it. A server behind a mirror should be
 * asked about its mirror.
 */
export async function hubEndpoint(client: UnslothClient, options: HubOptions = {}): Promise<string> {
  try {
    const body = await client.request<unknown>("/api/health", {
      anonymous: true,
      timeoutMs: options.timeoutMs ?? HUB_TIMEOUT_MS,
      ...(options.signal ? { signal: options.signal } : {}),
    });
    const endpoint =
      typeof body === "object" && body !== null ? (body as Record<string, unknown>)["hf_endpoint"] : undefined;
    if (typeof endpoint === "string" && endpoint.trim() !== "") return endpoint.trim().replace(/\/+$/, "");
  } catch {
    // A server that will not say is a server that gets the default.
  }
  return DEFAULT_HUB_ENDPOINT;
}

/**
 * Does this look like a Hub repo id rather than a path on disk?
 *
 * `owner/name`, one slash, no drive letters, no leading slash. A model loaded
 * from `/srv/models/qwen.gguf` has no model card to read, and saying so is
 * better than asking the Hub about a filename.
 */
export function isRepoId(value: string | undefined): value is string {
  if (!value) return false;
  if (value.startsWith("/") || value.startsWith(".") || value.includes("\\")) return false;
  const parts = value.split("/");
  return parts.length === 2 && parts.every((part) => part.trim() !== "");
}

export interface HubFetch {
  status: number;
  body: unknown;
}

/** The Hub token from the environment, if the user has one set. */
function hubToken(): string | undefined {
  for (const name of TOKEN_VARS) {
    const value = process.env[name]?.trim();
    if (value) return value;
  }
  return undefined;
}

/** One GET, with a timeout and an abort, returning the status rather than throwing. */
async function get(url: string, options: HubOptions): Promise<HubFetch> {
  const timeoutMs = options.timeoutMs ?? HUB_TIMEOUT_MS;
  const controller = new AbortController();
  const signals: AbortSignal[] = [controller.signal];
  if (options.signal) signals.push(options.signal);
  const signal = signals.length === 1 ? signals[0]! : AbortSignal.any(signals);

  const timer = setTimeout(() => controller.abort(), timeoutMs);
  timer.unref?.();

  const headers: Record<string, string> = { Accept: "application/json" };
  const token = hubToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const call = options.fetchImpl ?? fetch;
  try {
    const response = await call(url, { headers, signal });
    if (!response.ok) return { status: response.status, body: undefined };
    const text = await response.text();
    try {
      return { status: response.status, body: JSON.parse(text) };
    } catch {
      return { status: response.status, body: undefined };
    }
  } finally {
    clearTimeout(timer);
  }
}

/**
 * The repo a quantisation was made from, per the Hub's own card metadata.
 *
 * GGUF repos do not carry `generation_config.json` — checked against
 * `ggml-org/Qwen3-4B-GGUF` and `unsloth/Qwen3-4B-GGUF`, both `404` — but they
 * do declare `base_model`, and the base model publishes one. That hop is the
 * whole reason this lookup is two requests rather than one.
 */
export function baseModelOf(card: unknown): string | undefined {
  if (typeof card !== "object" || card === null) return undefined;
  const data = (card as Record<string, unknown>)["cardData"];
  const base = typeof data === "object" && data !== null ? (data as Record<string, unknown>)["base_model"] : undefined;
  const candidate = Array.isArray(base) ? base[0] : base;
  return isRepoId(typeof candidate === "string" ? candidate : undefined) ? (candidate as string) : undefined;
}

export interface GenerationConfigResult {
  /** What the card recommends. Empty when there is nothing to take. */
  params: SamplingParams;
  /** The repo the numbers came from — the base model, when we followed one. */
  from: string | undefined;
  /** Why there is nothing, in the words the screen will show. */
  detail: string | undefined;
}

const NOTHING = (detail: string): GenerationConfigResult => ({ params: {}, from: undefined, detail });

/**
 * A model's published sampling defaults, following `base_model` once.
 *
 * Never throws. Every way this can fail to produce numbers — offline, not a
 * repo, no such file, a file with nothing in it — ends as a sentence, because
 * the screen behind it stays open either way and the user's own values are
 * still there.
 */
export async function fetchGenerationConfig(
  endpoint: string,
  repoId: string | undefined,
  options: HubOptions = {},
): Promise<GenerationConfigResult> {
  if (process.env["PI_OFFLINE"]) return NOTHING("PI_OFFLINE is set — nothing was fetched");
  if (!isRepoId(repoId)) {
    return NOTHING("this model is a local file, so there is no model card to read");
  }

  const base = endpoint.replace(/\/+$/, "");
  /** The repos that answered `401`/`403`, so the message can name the real cause. */
  const gated: string[] = [];
  const attempt = async (repo: string): Promise<SamplingParams | number> => {
    const result = await get(`${base}/${repo}/resolve/main/generation_config.json`, options);
    if (result.status === 401 || result.status === 403) gated.push(repo);
    if (result.status !== 200) return result.status;
    return parseGenerationConfig(result.body);
  };

  let params: SamplingParams | number;
  try {
    params = await attempt(repoId);
  } catch (error) {
    return NOTHING(`could not reach ${base} — ${error instanceof Error ? error.message : String(error)}`);
  }

  if (typeof params !== "number" && Object.keys(params).length > 0) {
    return { params, from: repoId, detail: undefined };
  }

  // Either the file is missing (the usual GGUF case) or it held nothing we
  // can use. The base model is the one place worth looking next.
  let card: HubFetch;
  try {
    card = await get(`${base}/api/models/${repoId}`, options);
  } catch (error) {
    return NOTHING(`could not reach ${base} — ${error instanceof Error ? error.message : String(error)}`);
  }

  const source = baseModelOf(card.body);
  if (!source || source === repoId) {
    if (gated.length > 0) return NOTHING(gatedDetail(gated));
    return NOTHING(`${repoId} does not publish a generation_config.json`);
  }

  let fromBase: SamplingParams | number;
  try {
    fromBase = await attempt(source);
  } catch (error) {
    return NOTHING(`could not reach ${base} — ${error instanceof Error ? error.message : String(error)}`);
  }
  if (typeof fromBase === "number" || Object.keys(fromBase).length === 0) {
    // Gating is a different problem from absence and has a different fix, so
    // it never hides behind "does not publish" — checked against
    // `google/gemma-3-27b-it`, which answers `401` with nobody signed in.
    if (gated.length > 0) return NOTHING(gatedDetail(gated));
    return NOTHING(`neither ${repoId} nor ${source} publishes sampling defaults`);
  }
  return { params: fromBase, from: source, detail: undefined };
}

/**
 * A gated repo, in words that name the fix rather than the status code.
 *
 * The token itself is never part of this — or of any other string this file
 * produces (cross-cutting gate: no secret on a status line).
 */
function gatedDetail(gated: readonly string[]): string {
  const who = gated[gated.length - 1] ?? "the model";
  const how = hubToken() ? "your HF_TOKEN does not grant access" : "set HF_TOKEN, or type the values yourself";
  return `${who} is gated on Hugging Face — ${how}`;
}
