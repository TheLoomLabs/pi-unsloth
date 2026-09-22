/**
 * Typed fetch wrapper for the Unsloth Studio HTTP API.
 *
 * Contract:
 *   - every call carries a timeout *and* an AbortSignal
 *   - a cancelled call rejects promptly and leaves no timer behind
 *   - errors carry the server's own `reason` string when it sends one
 *
 * Endpoint reference
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { getAgentDir } from "@earendil-works/pi-coding-agent";

/** Documented default for a local Unsloth Studio install. */
export const DEFAULT_BASE_URL = "http://127.0.0.1:8888";

/** Default per-call budget. Load/unload pass their own, longer, value. */
export const DEFAULT_TIMEOUT_MS = 10_000;

/** `/api/health` must answer fast or it is not up as far as we are concerned. */
export const HEALTH_TIMEOUT_MS = 1_500;

export interface UnslothEndpoint {
  baseUrl: string;
  apiKey: string | undefined;
}

export interface UnslothClientOptions {
  baseUrl?: string;
  /** Static key, or a resolver so Pi's credential store stays the source of truth. */
  apiKey?: string | (() => string | undefined | Promise<string | undefined>);
  timeoutMs?: number;
}

export interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "DELETE";
  /** JSON body. Serialised here so callers never hand-roll headers. */
  body?: unknown;
  query?: Record<string, string | number | boolean | undefined>;
  timeoutMs?: number;
  /** Caller's cancellation, merged with the per-call timeout. */
  signal?: AbortSignal;
  /** Skip the Authorization header (`/api/health`, `/v1/models`). */
  anonymous?: boolean;
}

/** A non-2xx response. `reason` is the server's own explanation when it gave one. */
export class UnslothApiError extends Error {
  override readonly name = "UnslothApiError";
  readonly status: number;
  readonly path: string;
  readonly reason: string | undefined;
  /**
   * The parsed error body, when it was JSON.
   *
   * Kept because the *shape* of a rejection carries meaning this server's
   * callers need: FastAPI answers an application refusal with a string
   * `detail` and a schema mismatch with a list of them, and a pre-flight that
   * cannot tell those apart would block loads whenever our own payload drifted
   * (src/api/lifecycle.ts → `preflightLoad`).
   */
  readonly body: unknown;

  constructor(path: string, status: number, reason: string | undefined, message?: string, body?: unknown) {
    super(message ?? `${path} → ${status}${reason ? `: ${reason}` : ""}`);
    this.status = status;
    this.path = path;
    this.reason = reason;
    this.body = body;
  }

  /** 401/403 — the key is missing or wrong, which the UI reports differently. */
  get isAuthError(): boolean {
    return this.status === 401 || this.status === 403;
  }
}

/** The call exceeded its own budget. Distinct from a caller-initiated abort. */
export class UnslothTimeoutError extends Error {
  override readonly name = "UnslothTimeoutError";
  readonly path: string;
  readonly timeoutMs: number;

  constructor(path: string, timeoutMs: number) {
    super(`${path} timed out after ${timeoutMs} ms`);
    this.path = path;
    this.timeoutMs = timeoutMs;
  }
}

/**
 * The server could not be reached at all (down, refused, DNS).
 *
 * `fetch` reports every one of these as a bare "fetch failed", so the real
 * cause is unwrapped here — "connect ECONNREFUSED 127.0.0.1:8888" tells the
 * user their server is not running; "fetch failed" tells them nothing.
 */
export class UnslothUnreachableError extends Error {
  override readonly name = "UnslothUnreachableError";
  readonly path: string;

  constructor(path: string, cause: unknown) {
    super(`${path}: ${describeCause(cause)}`);
    this.path = path;
    this.cause = cause;
  }
}

/** Walk the `cause` chain for the most specific message it carries. */
function describeCause(error: unknown): string {
  let current = error;
  let message = typeof error === "string" ? error : "unknown error";
  for (let depth = 0; depth < 5 && current instanceof Error; depth++) {
    if (current.message && current.message !== "fetch failed") message = current.message;
    const code = (current as NodeJS.ErrnoException).code;
    if (typeof code === "string" && code !== "") {
      return message.includes(code) ? message : `${code} (${message})`;
    }
    current = current.cause;
  }
  return message;
}

function normaliseBaseUrl(value: string): string {
  return value.replace(/\/+$/, "");
}

/**
 * Best-effort read of `baseUrl` from the machine profile.
 * src/hardware/profile.ts (M4) owns the full read/write/migrate cycle; this
 * only needs the one field and must never throw on a corrupt or absent file.
 */
function baseUrlFromProfile(): string | undefined {
  try {
    const raw = readFileSync(join(getAgentDir(), "unsloth.json"), "utf8");
    const parsed: unknown = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      const candidate = (parsed as { baseUrl?: unknown }).baseUrl;
      if (typeof candidate === "string" && candidate.trim() !== "") return candidate.trim();
    }
  } catch {
    // No profile yet, unreadable, or not JSON — the default is correct.
  }
  return undefined;
}

/** Best-effort read of the API key Pi stores for the `unsloth` provider. */
function apiKeyFromAuthFile(): string | undefined {
  try {
    const raw = readFileSync(join(getAgentDir(), "auth.json"), "utf8");
    const parsed: unknown = JSON.parse(raw);
    const entry = (parsed as { unsloth?: { key?: unknown } } | null)?.unsloth;
    if (entry && typeof entry.key === "string" && entry.key !== "") return entry.key;
  } catch {
    // Not logged in yet — callers surface that as an auth error, not a crash.
  }
  return undefined;
}

/** Resolution order: explicit option → environment → machine profile → default. */
export function resolveEndpoint(options: UnslothClientOptions = {}): UnslothEndpoint {
  const fromOption = typeof options.baseUrl === "string" ? options.baseUrl.trim() : "";
  const fromEnv = process.env.UNSLOTH_BASE_URL?.trim();
  const baseUrl = normaliseBaseUrl(
    fromOption || fromEnv || baseUrlFromProfile() || DEFAULT_BASE_URL,
  );
  const apiKey =
    typeof options.apiKey === "string"
      ? options.apiKey
      : (process.env.UNSLOTH_API_KEY?.trim() ?? apiKeyFromAuthFile());
  return { baseUrl, apiKey };
}

export class UnslothClient {
  readonly baseUrl: string;
  private readonly resolveKey: () => string | undefined | Promise<string | undefined>;
  private readonly defaultTimeoutMs: number;

  constructor(options: UnslothClientOptions = {}) {
    const endpoint = resolveEndpoint(options);
    this.baseUrl = endpoint.baseUrl;
    this.defaultTimeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.resolveKey =
      typeof options.apiKey === "function"
        ? options.apiKey
        : () => (typeof options.apiKey === "string" ? options.apiKey : endpoint.apiKey);
  }

  private url(path: string, query: RequestOptions["query"]): string {
    const url = new URL(path.startsWith("/") ? path : `/${path}`, `${this.baseUrl}/`);
    for (const [key, value] of Object.entries(query ?? {})) {
      if (value !== undefined) url.searchParams.set(key, String(value));
    }
    return url.toString();
  }

  /**
   * Issue one request. Always aborts on the earlier of the caller's signal and
   * the per-call timeout, and always clears its own timer.
   */
  async request<T>(path: string, options: RequestOptions = {}): Promise<T> {
    const timeoutMs = options.timeoutMs ?? this.defaultTimeoutMs;
    const timeoutController = new AbortController();
    const signals: AbortSignal[] = [timeoutController.signal];
    if (options.signal) signals.push(options.signal);
    const signal = signals.length === 1 ? signals[0]! : AbortSignal.any(signals);

    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      timeoutController.abort();
    }, timeoutMs);
    // Never let a pending request hold the process open by itself.
    timer.unref?.();

    const headers: Record<string, string> = { Accept: "application/json" };
    if (!options.anonymous) {
      const key = await this.resolveKey();
      if (key) headers["Authorization"] = `Bearer ${key}`;
    }

    const init: RequestInit = { method: options.method ?? "GET", headers, signal };
    if (options.body !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(options.body);
    }

    let response: Response;
    try {
      response = await fetch(this.url(path, options.query), init);
    } catch (error) {
      if (timedOut) throw new UnslothTimeoutError(path, timeoutMs);
      // A caller-initiated abort propagates unchanged so callers can ignore it.
      if (options.signal?.aborted) throw error;
      throw new UnslothUnreachableError(path, error);
    } finally {
      clearTimeout(timer);
    }

    if (!response.ok) {
      const failure = await readErrorBody(response);
      throw new UnslothApiError(path, response.status, failure.reason, undefined, failure.body);
    }
    if (response.status === 204) return undefined as T;

    const text = await response.text();
    if (text.trim() === "") return undefined as T;
    try {
      return JSON.parse(text) as T;
    } catch {
      throw new UnslothApiError(path, response.status, undefined, `${path}: response was not JSON`);
    }
  }

  get<T>(path: string, options: Omit<RequestOptions, "method" | "body"> = {}): Promise<T> {
    return this.request<T>(path, { ...options, method: "GET" });
  }

  post<T>(path: string, body?: unknown, options: Omit<RequestOptions, "method" | "body"> = {}): Promise<T> {
    return this.request<T>(path, { ...options, method: "POST", body });
  }

  put<T>(path: string, body?: unknown, options: Omit<RequestOptions, "method" | "body"> = {}): Promise<T> {
    return this.request<T>(path, { ...options, method: "PUT", body });
  }
}

/**
 * Pull the server's own explanation, and the parsed body, out of a failure.
 *
 * Both halves are kept: the string is what the UI shows, and the object is what
 * lets a caller tell an application refusal from a schema mismatch.
 */
async function readErrorBody(response: Response): Promise<{ reason: string | undefined; body: unknown }> {
  let text: string;
  try {
    text = await response.text();
  } catch {
    return { reason: undefined, body: undefined };
  }
  if (text.trim() === "") return { reason: undefined, body: undefined };

  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    // Not JSON — a short plain-text body is still useful.
  }
  if (body !== null && typeof body === "object") {
    const parsed = body as Record<string, unknown>;
    for (const field of ["reason", "detail", "message", "error"]) {
      const value = parsed[field];
      if (typeof value === "string" && value !== "") return { reason: value, body };
    }
  }
  const trimmed = text.trim();
  return { reason: trimmed.length > 200 ? `${trimmed.slice(0, 200)}…` : trimmed, body };
}

export type HealthState = "up" | "down" | "unauthorized";

export interface HealthResult {
  state: HealthState;
  /** Round-trip time in ms, for the diagnostics line. */
  latencyMs: number;
  /** Why it is not up — the server's `reason` when there is one. */
  detail: string | undefined;
}

/**
 * Is the server answering? `/api/health` needs no key, so a failure here is
 * always reachability, never credentials.
 *
 * Never throws: a down server is an expected state, not an error.
 */
export async function checkHealth(
  client: UnslothClient,
  options: { timeoutMs?: number; signal?: AbortSignal } = {},
): Promise<HealthResult> {
  const timeoutMs = options.timeoutMs ?? HEALTH_TIMEOUT_MS;
  const started = Date.now();
  try {
    await client.request<unknown>("/api/health", {
      anonymous: true,
      timeoutMs,
      ...(options.signal ? { signal: options.signal } : {}),
    });
    return { state: "up", latencyMs: Date.now() - started, detail: undefined };
  } catch (error) {
    const latencyMs = Date.now() - started;
    if (error instanceof UnslothApiError && error.isAuthError) {
      return { state: "unauthorized", latencyMs, detail: error.reason };
    }
    const detail =
      error instanceof UnslothApiError
        ? (error.reason ?? `HTTP ${error.status}`)
        : error instanceof Error
          ? error.message
          : String(error);
    return { state: "down", latencyMs, detail };
  }
}

/** Convenience for callers that only need the boolean. */
export async function isHealthy(
  client: UnslothClient,
  options: { timeoutMs?: number; signal?: AbortSignal } = {},
): Promise<boolean> {
  return (await checkHealth(client, options)).state === "up";
}
