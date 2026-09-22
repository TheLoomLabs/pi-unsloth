/**
 * Keeping the server, the catalogue and the loaded model in step — everything
 * `pil` and `piloff` used to do, from inside the editor.
 *
 * Three jobs:
 *
 *   - **autostart** — bring Unsloth Studio up when a session needs it, without
 *     the editor ever waiting for it;
 *   - **ensure-tuned-load** — re-apply a model's tuned settings on *every*
 *     switch, because auto-switch silently drops speculative decoding;
 *   - **unload** — the two-step that actually frees VRAM.
 *
 * The rule throughout: nothing here blocks the TUI. Slow work runs in the
 * background with a status line, and the one place that may wait —
 * `before_agent_start` — waits only when the answer would otherwise be wrong,
 * and can be escaped.
 */

import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { Api, Model } from "@earendil-works/pi-ai";

import { checkHealth, type UnslothClient } from "./api/client.ts";
import {
  buildLoadRequest,
  findOverride,
  getLoadProgress,
  getOverrides,
  loadModel,
  preflightLoad,
  unloadModel,
  type ModelOverride,
} from "./api/lifecycle.ts";
import { listLoadedModels, listLocalModels, listModels, type CatalogueEntry, type LocalModel } from "./api/models.ts";
import { readComputeUsage } from "./api/system.ts";
import { mergeIdleVram } from "./hardware/detect.ts";
import {
  isInferenceRunning,
  listInferenceProcesses,
  processListingSupported,
  type ProcessInfo,
  type TerminateResult,
  startServerProcess,
  terminateInferenceProcesses,
} from "./process.ts";
import { PROVIDER_ID, createSessionClient } from "./provider.ts";
import { endpointLabel, isLocalEndpoint } from "./endpoint.ts";
import { autostartEnabled, launchCommand } from "./settings.ts";
import { ctxLive, hasUI } from "./session.ts";
import { state, type GpuUsage } from "./state.ts";
import { paint } from "./ui/footer.ts";

/** Budget for one catalogue refresh. Generous: it is off the critical path. */
const REFRESH_TIMEOUT_MS = 15_000;

/** How long to wait for a server we just started to answer `/api/health`. */
export const STARTUP_TIMEOUT_MS = 120_000;
/** Gap between health probes while waiting for startup. */
const STARTUP_POLL_MS = 1_000;

/**
 * Budget for one explicit load. A cold large model is minutes of disk and VRAM
 * traffic; the escape key, not the clock, is the intended way out.
 */
const LOAD_TIMEOUT_MS = 900_000;
/** Gap between `load-progress` polls while a load runs. */
const PROGRESS_POLL_MS = 1_000;

/** Short budget for the questions asked *about* a load, not the load itself. */
const QUERY_TIMEOUT_MS = 10_000;

/** How long to let a VRAM figure settle after an unload before believing it. */
const SETTLE_TIMEOUT_MS = 10_000;
const SETTLE_POLL_MS = 500;
/** A drop smaller than this is noise, not memory coming back. */
const SETTLE_EPSILON_GB = 0.05;
/** Polls without improvement that mean the figure has stopped falling. */
const SETTLE_STABLE_POLLS = 6;

/**
 * The in-flight ensure, if any.
 *
 * Module scope rather than `state`, because it is a handle on work rather than
 * an observation, and because `resetState()` must not silently orphan it.
 */
let pending: { modelId: string; promise: Promise<void>; controller: AbortController } | undefined;

/** True once we have told the user we could not start the server. */
let startupReported = false;

/**
 * Models we have already complained about this session.
 *
 * `before_agent_start` ensures on *every* turn, so a warning that is a property
 * of the model rather than of this turn — no override, not on disk — would
 * otherwise repeat on every single prompt.
 */
const warned = new Set<string>();

function warnOnce(ctx: ExtensionContext, key: string, text: string): void {
  if (warned.has(key)) return;
  warned.add(key);
  report(ctx, text, "warning");
}

/** Pull the catalogue from the server into Pi's registry. */
export async function refreshCatalogue(
  ctx: ExtensionContext,
  options: { signal?: AbortSignal } = {},
): Promise<void> {
  const timeout = new AbortController();
  const timer = setTimeout(() => timeout.abort(), REFRESH_TIMEOUT_MS);
  timer.unref?.();
  const signal = options.signal ? AbortSignal.any([options.signal, timeout.signal]) : timeout.signal;

  try {
    await ctx.modelRegistry.refresh({
      providers: [PROVIDER_ID],
      // We are talking to a server on this machine; PI_OFFLINE is about the
      // internet, and should not stop a local catalogue from loading.
      allowNetwork: true,
      signal,
    });
  } catch {
    // fetchModels already recorded why in the session state.
  } finally {
    clearTimeout(timer);
  }
}

/** One line, wherever this run can show it. */
export function report(ctx: ExtensionContext, text: string, level: "info" | "warning" | "error"): void {
  // Background work can still be holding the ctx of a session that has since
  // been replaced. Dropping the line is the whole answer: there is no window
  // left to notify, and stderr in a live TUI would print through the render.
  if (!ctxLive(ctx)) return;
  if (hasUI(ctx)) {
    ctx.ui.notify(text, level);
  } else {
    // print/json modes: stderr keeps stdout clean for pipelines.
    process.stderr.write(`${text}\n`);
  }
}

function setServer(ctx: ExtensionContext, next: typeof state.server, detail?: string): void {
  state.server = next;
  state.serverDetail = detail;
  paint(ctx);
}

/**
 * Make sure the server is answering, starting it if it is not.
 *
 * Returns `true` when the server is up. Never throws, and never waits on a
 * server it did not start: if autostart is off and nothing is listening, the
 * answer is simply "no", which the footer shows as one dim line.
 */
export async function ensureServer(
  ctx: ExtensionContext,
  options: { signal?: AbortSignal } = {},
): Promise<boolean> {
  const client = createSessionClient(ctx);
  const first = await checkHealth(client, options.signal ? { signal: options.signal } : {});
  if (first.state === "up") {
    setServer(ctx, "up");
    return true;
  }
  if (first.state === "unauthorized") {
    setServer(ctx, "unauthorized", first.detail);
    return false;
  }

  if (!autostartEnabled()) {
    setServer(ctx, "offline", first.detail);
    return false;
  }

  // Autostart starts a server *here*. When the endpoint is another machine that
  // would bind a second, local Studio that nobody is talking to, and the user
  // would be left wondering which of the two they are configured against.
  if (!isLocalEndpoint(client.baseUrl)) {
    const where = endpointLabel(client.baseUrl);
    setServer(ctx, "offline", `${where} is not answering — it can only be started on that machine`);
    if (!startupReported) {
      startupReported = true;
      report(ctx, `○ Unsloth at ${where} is not answering — start it on that machine`, "warning");
    }
    return false;
  }

  const command = launchCommand();
  const spawned = startServerProcess(command);
  if (!spawned.started) {
    setServer(ctx, "unreachable", spawned.error);
    if (!startupReported) {
      startupReported = true;
      // The sentence names the other way out, because this is exactly what a
      // fresh client machine sees: no local install, and no reason to guess
      // that the server is allowed to be somewhere else.
      report(
        ctx,
        `✗ Could not start Unsloth Studio (${command[0]}): ${spawned.error} — or /unsloth setup to point at another machine`,
        "error",
      );
    }
    return false;
  }

  setServer(ctx, "starting");
  const deadline = Date.now() + STARTUP_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (options.signal?.aborted) break;
    await sleep(STARTUP_POLL_MS, options.signal);
    if (options.signal?.aborted) break;
    const health = await checkHealth(client, options.signal ? { signal: options.signal } : {});
    if (health.state === "up") {
      setServer(ctx, "up");
      return true;
    }
    if (health.state === "unauthorized") {
      setServer(ctx, "unauthorized", health.detail);
      return false;
    }
  }

  // A spawn that produced no listener is a different failure from "not
  // running": the user asked for it, it was attempted, and it did not work.
  setServer(ctx, options.signal?.aborted ? "offline" : "unreachable", `no response within ${STARTUP_TIMEOUT_MS / 1000}s`);
  if (!startupReported && !options.signal?.aborted) {
    startupReported = true;
    report(ctx, `✗ Unsloth Studio did not come up at ${client.baseUrl}`, "error");
  }
  return false;
}

/** A cancellable, unref'd sleep. */
function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal?.aborted) return resolve();
    const timer = setTimeout(finish, ms);
    timer.unref?.();
    signal?.addEventListener("abort", finish, { once: true });
    function finish(): void {
      clearTimeout(timer);
      signal?.removeEventListener("abort", finish);
      resolve();
    }
  });
}

/** A model the extension is responsible for. */
export function isOurModel(model: Model<Api> | undefined): boolean {
  return model?.provider === PROVIDER_ID;
}

/**
 * Quant and context tokens, which the footer shows in their own columns.
 *
 * `Q8_0`, `Q4_K_M`, `IQ4_XS`, `F16`, `192K`, `40960` — and `local`, which is
 * how a hand-written `models.json` marks an on-disk model. None of them is part
 * of a model's *name*, and all of them turn up glued to one.
 */
const NOISE = /^(?:[QqIi][Qq]?\d[\w.]*|[Ff]16|[Bb][Ff]16|[Ff]32|MXFP\d|\d+[KkMm]|\d{3,}|local|gguf)$/;

/**
 * The short name for the footer — never the whole `name · quant · context`.
 *
 * Two sources of noise, and only the first has a separator to cut on. Our own
 * catalogue builds `Qwen3.8-27B · Q8_0 · 192K`, so the first segment is the
 * name. A `models.json` written by hand does not: it names the same model
 * `Qwen3.8-27B Q8_0 local 192K`, and taking that whole string gave a footer
 * reading `⬢ Qwen3.8-27B Q8_0 local 192K · 192K · 21.5+18.4/48.0 GiB` — the
 * context twice and the quant in the wrong place.
 *
 * So: cut at the separator if there is one, then drop trailing words that are
 * facts the footer already shows in their own right. Trailing only — a model
 * really called `Q8` stays `Q8`, and nothing is ever dropped from the middle.
 */
export function shortLabel(model: Model<Api>): string {
  const first = model.name.split(" · ")[0]?.trim() ?? "";
  const words = (first !== "" ? first : model.id).split(/\s+/).filter((word) => word !== "");
  while (words.length > 1 && NOISE.test(words[words.length - 1] as string)) words.pop();
  const label = words.join(" ");
  return label !== "" ? label : model.id;
}

export interface Resolution {
  entry: CatalogueEntry | undefined;
  local: LocalModel | undefined;
  override: ModelOverride | undefined;
  /** What `load` wants: the absolute path, or the repo id for a hub model. */
  modelPath: string | undefined;
  quant: string | undefined;
}

/**
 * Work out how to load a Pi model id.
 *
 * ⚠ The same model has two identities — a directory and the GGUF file inside
 * it — and only the directory one carries the override, so the path comes from
 * `/api/models/local` rather than from the catalogue id.
 */
export async function resolveModel(client: UnslothClient, modelId: string, signal: AbortSignal): Promise<Resolution> {
  const call = { signal, timeoutMs: QUERY_TIMEOUT_MS };
  const [entries, local, overrides] = await Promise.all([
    listModels(client, call).catch(() => [] as CatalogueEntry[]),
    listLocalModels(client, call).then((result) => result.models, () => [] as LocalModel[]),
    getOverrides(client, call).catch(() => ({})),
  ]);

  const entry = entries.find((candidate) => candidate.id === modelId);
  const match =
    local.find((candidate) => candidate.id === modelId) ??
    local.find((candidate) => candidate.repoId === modelId) ??
    local.find((candidate) => candidate.displayName === modelId);

  const quant = entry?.quant;
  const override = findOverride(overrides, [match?.repoId, match?.path, modelId], quant);
  return {
    entry,
    local: match,
    override,
    modelPath: match?.repoId ?? match?.path ?? undefined,
    quant,
  };
}

/**
 * Find the catalogue entry a user meant.
 *
 * Shared by `/unsloth add` and `/unsloth sampling` so the two cannot come to
 * disagree about what a name means. Deliberately forgiving — case, and any
 * unique substring — because the ids are paths and repo names, and nobody
 * types `ggml-org/Qwen3-4B-GGUF` twice. An exact match always wins over a
 * substring, so a model whose name is contained in another's is still
 * reachable; anything still ambiguous is reported with its candidates rather
 * than resolved by picking the first.
 */
export type CatalogueMatch =
  | { kind: "one"; entry: CatalogueEntry }
  | { kind: "none" }
  | { kind: "ambiguous"; candidates: string[] };

export function matchCatalogue(entries: readonly CatalogueEntry[], wanted: string): CatalogueMatch {
  const needle = wanted.trim().toLowerCase();
  if (needle === "") return { kind: "none" };

  const names = (entry: CatalogueEntry): string[] => [entry.id, entry.displayName ?? ""].filter(Boolean);
  const exact = entries.filter((entry) => names(entry).some((value) => value.toLowerCase() === needle));
  const matches =
    exact.length > 0 ? exact : entries.filter((entry) => names(entry).some((value) => value.toLowerCase().includes(needle)));

  if (matches.length === 0) return { kind: "none" };
  if (matches.length > 1) {
    return { kind: "ambiguous", candidates: matches.map((entry) => entry.displayName ?? entry.id) };
  }
  return { kind: "one", entry: matches[0] as CatalogueEntry };
}

/**
 * Is this model already loaded *for real*?
 *
 * Two questions, because Unsloth keeps reporting a model as active after
 * `llama-server` has died. Where processes cannot be listed at all we trust the
 * server and accept the occasional redundant load — which is correct, just
 * slow — rather than refusing to work on that platform.
 */
export async function isModelResident(client: UnslothClient, modelId: string, signal: AbortSignal): Promise<boolean> {
  const loaded = await listLoadedModels(client, { signal, timeoutMs: QUERY_TIMEOUT_MS }).catch(
    () => [] as CatalogueEntry[],
  );
  const claimed = loaded.some((entry) => entry.id === modelId || entry.displayName === modelId);
  if (!claimed) return false;
  // The process check exists because Unsloth reports a model as active after
  // `llama-server` has died. Against a remote server there is no such process
  // here to look at, and a *local* one would be about something else entirely,
  // so the server is believed. The cost of being wrong is a redundant load.
  if (!isLocalEndpoint(client.baseUrl)) return true;
  if (!processListingSupported()) return true;
  return isInferenceRunning();
}

/**
 * Load `model` with its tuned settings, unless it is already resident.
 *
 * A model with no override is left to Unsloth's auto-switch: this extension
 * cannot yet produce a tuned config for it (that is `/unsloth add`, M5), and
 * inventing one here is exactly the guesswork the project exists to delete.
 */
export async function ensureTunedLoad(
  ctx: ExtensionContext,
  model: Model<Api>,
  options: { signal?: AbortSignal } = {},
): Promise<void> {
  const controller = new AbortController();
  const signal = options.signal
    ? AbortSignal.any([options.signal, controller.signal])
    : controller.signal;

  adoptModel(ctx, model);

  if (!(await ensureServer(ctx, { signal }))) return;
  if (signal.aborted) return;

  const client = createSessionClient(ctx);
  if (await isModelResident(client, model.id, signal)) return;
  if (signal.aborted) return;

  const resolution = await resolveModel(client, model.id, signal);
  if (signal.aborted) return;

  if (!resolution.modelPath) {
    warnOnce(ctx, `missing:${model.id}`, `⚠ ${shortLabel(model)} is not on disk — Unsloth cannot load it`);
    return;
  }
  if (!resolution.override) {
    // M5 gave this warning something to offer. Until the sizer existed it could
    // only name what would happen instead.
    warnOnce(
      ctx,
      `untuned:${model.id}`,
      `⚠ ${shortLabel(model)} has no tuned settings — press ctrl+alt+u then a to size it`,
    );
    return;
  }

  const outcome = await runTunedLoad(ctx, client, shortLabel(model), resolution, { signal });
  if (outcome.ok) report(ctx, `✓ Unsloth ready — ${shortLabel(model)} loaded`, "info");
  else if (outcome.error) report(ctx, `✗ Load failed: ${outcome.error}`, "error");
}

/** What a load did. `error` is absent when the caller walked away from it. */
export interface LoadOutcome {
  ok: boolean;
  error: string | undefined;
}

/** The parts of a `Resolution` a load actually needs. */
export type LoadTarget = Pick<Resolution, "modelPath" | "quant" | "override">;

/**
 * Run one explicit load, reporting progress into `state.loading`.
 *
 * Split out of `ensureTunedLoad` because the panel loads a model the *session*
 * has not selected: everything above this point is about keeping Pi's chosen
 * model ready, and everything below it is about putting weights on a GPU. Only
 * the second half is shared.
 *
 * Says nothing to the user itself — the two callers phrase it differently, and
 * one of them draws it in an overlay rather than a notification.
 */
export async function runTunedLoad(
  ctx: ExtensionContext,
  client: UnslothClient,
  label: string,
  target: LoadTarget,
  options: { signal?: AbortSignal } = {},
): Promise<LoadOutcome> {
  if (!target.modelPath) return { ok: false, error: "not on disk" };

  const request = buildLoadRequest(target.modelPath, target.quant, target.override);

  // Tier A: every load is pre-flighted, because a configuration the server will
  // refuse costs nothing to catch here and a failed load halfway through 29 GiB
  // of disk traffic costs minutes. The pre-flight never *blocks* a load it
  // could not check — see `preflightLoad`.
  const call = options.signal
    ? { signal: options.signal, timeoutMs: LOAD_TIMEOUT_MS }
    : { timeoutMs: LOAD_TIMEOUT_MS };
  const verdict = await preflightLoad(client, request, {
    ...(options.signal ? { signal: options.signal } : {}),
    timeoutMs: QUERY_TIMEOUT_MS,
  });
  if (verdict.checked && !verdict.valid) {
    return { ok: false, error: verdict.message ?? "the server refused this configuration" };
  }

  state.loading = { label, fraction: undefined };
  paint(ctx);

  // Ends the progress poll the moment the load settles, however it settled.
  const watching = new AbortController();
  const progress = trackProgress(ctx, client, watching.signal);

  try {
    await loadModel(client, request, call);
    return { ok: true, error: undefined };
  } catch (error) {
    // A cancelled load is not a failed one: the server carries on regardless,
    // and the caller already knows it walked away.
    return { ok: false, error: options.signal?.aborted ? undefined : describe(error) };
  } finally {
    watching.abort();
    await progress;
    state.loading = undefined;
    paint(ctx);
  }
}

/** Poll `load-progress` into the footer until the load settles. */
async function trackProgress(ctx: ExtensionContext, client: UnslothClient, signal: AbortSignal): Promise<void> {
  while (!signal.aborted) {
    await sleep(PROGRESS_POLL_MS, signal);
    if (signal.aborted) return;
    const progress = await getLoadProgress(client, { signal, timeoutMs: QUERY_TIMEOUT_MS }).catch(() => undefined);
    if (!state.loading) return;
    // ⚠ `load-progress` keeps describing the *previous* load until this one
    // produces a figure, so the first poll of a cold load reports the last
    // one's `ready` at 100 %. Our own load is by definition not ready while we
    // are still awaiting it, so such a sample is someone else's and is
    // dropped — observed on the reference box as a bar that showed
    // "Ready … 100%" and then fell back to 0%.
    if (progress?.phase?.trim().toLowerCase() === "ready") continue;
    const fraction =
      progress?.fraction ??
      (progress?.bytesTotal && progress.bytesLoaded !== undefined && progress.bytesTotal > 0
        ? progress.bytesLoaded / progress.bytesTotal
        : undefined);
    state.loading = {
      label: state.loading.label,
      fraction,
      phase: progress?.phase,
      loadedBytes: progress?.bytesLoaded,
      totalBytes: progress?.bytesTotal,
    };
    paint(ctx);
  }
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Say that this is the model the session is talking to.
 *
 * Separate from loading it, and synchronous, because the footer must name the
 * model the moment Pi selects one — a session that starts with a model already
 * chosen (`--model`, or a restored session) and already resident would
 * otherwise sit on `○ unsloth idle` while happily answering prompts.
 */
export function adoptModel(ctx: ExtensionContext, model: Model<Api>): void {
  state.activeModelId = model.id;
  state.activeModelLabel = shortLabel(model);
  state.activeContextWindow = model.contextWindow;
  paint(ctx);
}

/**
 * Start an ensure in the background and remember it, so `before_agent_start`
 * can wait for the same work instead of starting a second copy of it.
 */
export function startEnsure(ctx: ExtensionContext, model: Model<Api>): void {
  adoptModel(ctx, model);
  if (pending?.modelId === model.id) return;
  cancelEnsure();
  const controller = new AbortController();
  const promise = ensureTunedLoad(ctx, model, { signal: controller.signal }).finally(() => {
    if (pending?.controller === controller) pending = undefined;
  });
  pending = { modelId: model.id, promise, controller };
  // Nothing awaits this: the editor must stay usable for the whole load.
  void promise.catch(() => {});
}

/** The ensure currently in flight for `modelId`, if that is what is running. */
export function pendingEnsure(modelId: string): Promise<void> | undefined {
  return pending?.modelId === modelId ? pending.promise : undefined;
}

/** Abandon any in-flight ensure. The server keeps loading; we stop watching. */
export function cancelEnsure(): void {
  pending?.controller.abort();
  pending = undefined;
}

/**
 * The local process layer, as one injectable object.
 *
 * It exists so the rule that matters most in this file — *never signal a
 * process for a server on another machine* — is a unit test rather than a
 * promise. Asserting that nothing was called is the only way to test an
 * absence, and the real implementation's absence of effect cannot be observed
 * without killing something.
 */
export interface ProcessLayer {
  supported(): boolean;
  list(): Promise<ProcessInfo[]>;
  terminate(options: { signal?: AbortSignal }): Promise<TerminateResult>;
}

const REAL_PROCESSES: ProcessLayer = {
  supported: processListingSupported,
  list: listInferenceProcesses,
  terminate: (options) => terminateInferenceProcesses(options),
};

export interface UnloadResult {
  /** The server accepted an unload for these model paths. */
  unloaded: string[];
  /** Inference processes we ended. */
  stopped: number[];
  /** Inference processes still running. Non-empty means VRAM is still held. */
  survivors: number[];
  /** VRAM given back, measured across compute GPUs rather than inferred. */
  freedGb: number | undefined;
  error: string | undefined;
  /**
   * Set when the endpoint is another machine: the API unload was sent and the
   * process half was **not attempted**, because the process that holds the
   * VRAM is over there. The caller says so rather than claiming a figure.
   */
  heldAt: string | undefined;
}

/**
 * The VRAM reading once it has stopped falling.
 *
 * Ending `llama-server` does not free its VRAM synchronously: the driver
 * reclaims it over the next few seconds, and `/api/system` keeps reporting the
 * old figure meanwhile. Worse, it does not fall smoothly — it **plateaus**, so
 * "wait for two consecutive readings to agree" settles on a number that is
 * still tens of gigabytes too high. Measured on the reference box: a reading
 * taken immediately reported 0.0 GiB freed, and one taken on the first plateau
 * reported 4.0 GiB, for an unload that actually returned ~38 GiB.
 *
 * So take the **minimum** over a short window instead. During an unload nothing
 * is allocating, so the smallest figure seen is the truest one, and a plateau
 * on the way down cannot be mistaken for the floor.
 *
 * The window ends early once the minimum has held for `SETTLE_STABLE_POLLS`
 * polls — but only after a drop has actually been seen. A driver that has not
 * begun releasing yet is *also* perfectly stable, and taking that for the floor
 * is how a 38 GiB unload came to report "0.0 GiB freed" a second time. When no
 * drop is seen at all, `dropped` is false and the caller says nothing about how
 * much was freed rather than claiming zero.
 */
async function settledVram(
  client: UnslothClient,
  call: { signal?: AbortSignal; timeoutMs: number },
  options: { signal?: AbortSignal } = {},
): Promise<{ gpus: GpuUsage[]; usedGb: number; dropped: boolean } | undefined> {
  const deadline = Date.now() + SETTLE_TIMEOUT_MS;
  const first = await usedVram(client, call);
  if (!first) return undefined;

  let lowest = first;
  let stable = 0;

  while (Date.now() < deadline && !options.signal?.aborted) {
    // ⚠ The plateau rule only applies *after* the release has begun. A driver
    // that has not started giving memory back yet also looks perfectly stable,
    // and exiting there settles on the pre-unload figure — which is how an
    // unload of 38 GiB came to report "0.0 GiB freed" on the reference box.
    if (stable >= SETTLE_STABLE_POLLS && lowest.usedGb < first.usedGb - SETTLE_EPSILON_GB) break;

    await sleep(SETTLE_POLL_MS, options.signal);
    const current = await usedVram(client, call);
    if (!current) break;
    if (current.usedGb < lowest.usedGb - SETTLE_EPSILON_GB) {
      lowest = current;
      stable = 0;
    } else {
      stable++;
    }
  }

  return { gpus: lowest.gpus, usedGb: lowest.usedGb, dropped: lowest.usedGb < first.usedGb - SETTLE_EPSILON_GB };
}

/** VRAM in use across compute GPUs, or `undefined` if we cannot look. */
async function usedVram(
  client: UnslothClient,
  call: { signal?: AbortSignal; timeoutMs: number },
): Promise<{ gpus: GpuUsage[]; usedGb: number } | undefined> {
  try {
    const gpus = await readComputeUsage(client, call);
    if (gpus.length === 0) return undefined;
    state.gpus = gpus;
    return { gpus, usedGb: gpus.reduce((sum, gpu) => sum + gpu.usedGb, 0) };
  } catch {
    return undefined;
  }
}

/**
 * Free the GPUs — the old `piloff`.
 *
 * Both halves are required and the order matters: the API call clears Unsloth's
 * "active model" state so its supervisor does not immediately respawn what we
 * are about to stop, and ending `llama-server` is what actually releases the
 * VRAM. Either one alone leaves the GPUs busy.
 */
export async function unloadAll(
  ctx: ExtensionContext,
  options: { signal?: AbortSignal; processes?: ProcessLayer } = {},
): Promise<UnloadResult> {
  const processes = options.processes ?? REAL_PROCESSES;
  const result: UnloadResult = {
    unloaded: [],
    stopped: [],
    survivors: [],
    freedGb: undefined,
    error: undefined,
    heldAt: undefined,
  };
  const call = options.signal ? { signal: options.signal, timeoutMs: QUERY_TIMEOUT_MS } : { timeoutMs: QUERY_TIMEOUT_MS };
  const client = createSessionClient(ctx);
  const local = isLocalEndpoint(client.baseUrl);

  cancelEnsure();

  const before = await usedVram(client, call);
  const health = await checkHealth(client, options.signal ? { signal: options.signal } : {});
  if (health.state === "up") {
    const loaded = await listLoadedModels(client, call).catch(() => [] as CatalogueEntry[]);
    const local = await listLocalModels(client, call).then((r) => r.models, () => [] as LocalModel[]);
    for (const entry of loaded) {
      const match =
        local.find((candidate) => candidate.id === entry.id) ??
        local.find((candidate) => candidate.repoId === entry.id) ??
        local.find((candidate) => candidate.displayName === entry.id);
      const path = match?.path ?? match?.repoId;
      if (!path) continue;
      try {
        await unloadModel(client, path, call);
        result.unloaded.push(path);
      } catch (error) {
        result.error ??= describe(error);
      }
    }
  }

  if (!local) {
    // The half of the unload that frees VRAM is a signal to a process, and the
    // process is on the other machine. Nothing here is enumerated, nothing is
    // signalled: a local `llama-server` belongs to whoever started it, and
    // killing it because a *remote* model was unloaded is the failure this
    // whole rule exists to prevent.
    result.heldAt = endpointLabel(client.baseUrl);
  } else if (processes.supported()) {
    const running = await processes.list();
    const terminated = await processes.terminate(options.signal ? { signal: options.signal } : {});
    result.stopped = terminated.found.length > 0 ? terminated.found : running.map((entry) => entry.pid);
    result.survivors = terminated.survivors;
  } else if (result.unloaded.length > 0) {
    result.error ??= `cannot stop llama-server on ${process.platform} — VRAM may stay held`;
  }

  state.activeModelId = undefined;
  state.activeModelLabel = undefined;
  state.activeContextWindow = undefined;
  state.loading = undefined;

  // Waiting for a figure to settle only makes sense when something was ended:
  // a remote unload leaves the weights resident by design, so there is nothing
  // to wait for and the wait would be seconds of nothing happening.
  const after = local ? await settledVram(client, call, options.signal ? { signal: options.signal } : {}) : undefined;
  if (before && after?.dropped) {
    result.freedGb = Math.max(0, before.usedGb - after.usedGb);
    // Nothing is allocating now, so this is the truest idle reading the session
    // will get — and the one the display-GPU signal wants.
    state.idleGpus = mergeIdleVram(state.idleGpus, after.gpus);
  }
  // A figure is reported only when the memory was seen to come back. The driver
  // reclaims asynchronously and can take longer than this window, and
  // "0.0 GiB freed" for an unload that worked is worse than saying nothing.
  paint(ctx);
  return result;
}

/** Reset everything this module owns. Idempotent; `session_shutdown` calls it. */
export function resetSupervisor(): void {
  cancelEnsure();
  startupReported = false;
  warned.clear();
}

/**
 * Unload, and say what happened — the reporting half of `piloff`.
 *
 * Lives here rather than in the command handler because three entry points want
 * exactly the same sentence: `/unsloth off`, `ctrl+alt+o`, and the panel's `u`.
 * `quiet` is for the one caller that must not speak — the auto-unload on exit,
 * where the TUI is already tearing down.
 */
export async function unloadAndReport(
  ctx: ExtensionContext,
  options: { quiet?: boolean; signal?: AbortSignal; processes?: ProcessLayer } = {},
): Promise<UnloadResult> {
  const result = await unloadAll(ctx, {
    ...(options.signal ? { signal: options.signal } : {}),
    ...(options.processes ? { processes: options.processes } : {}),
  });
  if (options.quiet) return result;
  const line = unloadMessage(result);
  report(ctx, line.text, line.level);
  return result;
}

/**
 * What an unload has to say for itself.
 *
 * Pure, and separate from sending it, because the four sentences are the whole
 * user-visible contract of `/unsloth off` and one of them can only be produced
 * by a server on another machine — which is not something a test can arrange.
 */
export function unloadMessage(result: UnloadResult): { text: string; level: "info" | "warning" } {
  if (result.survivors.length > 0) {
    return {
      text: `⚠ Unload incomplete — llama-server still running (${result.survivors.join(", ")})`,
      level: "warning",
    };
  }
  if (result.unloaded.length === 0 && result.stopped.length === 0) {
    return { text: `○ Nothing loaded${result.error ? ` — ${result.error}` : ""}`, level: "info" };
  }
  // A remote unload did half the job, and says which half: Unsloth has no
  // endpoint that stops `llama-server`, so the weights stay on that machine's
  // cards until someone acts there. Naming the host matters — "this machine"
  // would be a lie told to the one person who could fix it.
  if (result.heldAt !== undefined) {
    return {
      text: `✓ Unloaded on ${result.heldAt} — VRAM stays held there until llama-server is stopped on that machine`,
      level: "info",
    };
  }
  const freed = result.freedGb !== undefined ? ` ${result.freedGb.toFixed(1)} GiB freed.` : "";
  return { text: `✓ Unloaded.${freed}`, level: "info" };
}
