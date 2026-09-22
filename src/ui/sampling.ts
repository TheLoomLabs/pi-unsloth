/**
 * Sampling defaults — `s` in the panel, and `/unsloth sampling <model>`.
 *
 * The screen that holds the one thing the rest of this extension cannot
 * measure. Context, placement and speculative mode are all answers the server
 * or the estimator gives; `temperature` and its three neighbours are a
 * judgement about how a model should behave, and nothing on the machine knows
 * it. So there are exactly two sources, and the screen always says which one it
 * is showing:
 *
 *   - the user, typing;
 *   - `f`, which reads the model's own `generation_config.json` on the Hub
 *     (`src/api/hub.ts`) and fills the fields in as a starting point.
 *
 * Deliberately **not** part of the sizer, although both are per-model screens.
 * The sizer writes to the server and is gated on a server new enough to have
 * the estimator and on a topology it may not have; this writes to our own
 * profile and works against any server, offline, with no GPUs detected. Sharing
 * `⏎` between two destinations with different preconditions is how a save comes
 * to half-work.
 *
 * As everywhere else, the keys and the drawing are pure functions over one
 * state object, so each line is a unit test rather than something to go and
 * look at.
 */

import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Container, Key, matchesKey, truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";

import { fetchGenerationConfig, hubEndpoint } from "../api/hub.ts";
import { listModels, type CatalogueEntry } from "../api/models.ts";
import { readSamplingStore, rememberSampling } from "../hardware/profile.ts";
import { createSessionClient } from "../provider.ts";
import {
  SAMPLING_FIELDS,
  SAMPLING_KEYS,
  adjustParam,
  clearParam,
  formatParam,
  hasParams,
  type SamplingKey,
  type SamplingParams,
  type SamplingSource,
} from "../sampling.ts";
import { matchCatalogue, refreshCatalogue, report, resolveModel } from "../supervisor.ts";
import { Framed, Lines, MIN_OVERLAY_COLUMNS, overlayGeometry, padTo, terminalColumns } from "./draw.ts";

/** One open of this screen asks the server at most one question. */
const SAMPLING_TIMEOUT_MS = 10_000;

/** Margins and the label column, matching the sizer so the overlays line up. */
const MARGIN = 2;
const GAP = 2;
const LABEL_WIDTH = 12;

export type SamplingToken = "accent" | "success" | "warning" | "error" | "muted" | "dim" | "text";

export interface SamplingTheme {
  fg(color: SamplingToken, text: string): string;
  bold(text: string): string;
}

export interface SamplingModel {
  /** Pi's catalogue id — the key the store and the published model share. */
  id: string;
  name: string;
  /** `owner/name` when this model came from the Hub; a path otherwise. */
  repoId: string | undefined;
}

export interface SamplingUiState {
  model: SamplingModel;
  params: SamplingParams;
  /** What is on disk, so `⏎` can tell a change from a no-op. */
  stored: SamplingParams;
  /** Where the values on screen came from. */
  source: SamplingSource | undefined;
  /** The repo a `hub` fetch took them from. */
  from: string | undefined;
  /** Index into `SAMPLING_KEYS`. */
  field: number;
  /** A Hub read is in flight; the screen says so rather than looking idle. */
  busy: boolean;
  /** The last thing that happened, in the words the screen shows. */
  note: string | undefined;
}

/* --------------------------------------------------------------------------
 * Keys
 * ------------------------------------------------------------------------ */

export type SamplingOutcome = "open" | "save" | "fetch" | "cancel";

export interface SamplingKeyResult {
  state: SamplingUiState;
  outcome: SamplingOutcome;
}

/**
 * Apply one keypress.
 *
 * Any adjustment makes the values the user's, even when the number came from a
 * model card a moment ago: the provenance line has to describe what is on the
 * screen, and "from the model card" stops being true the first time someone
 * moves a field.
 */
export function applySamplingKey(state: SamplingUiState, input: string): SamplingKeyResult {
  const next: SamplingUiState = { ...state, params: { ...state.params } };
  const still = (outcome: SamplingOutcome = "open"): SamplingKeyResult => ({ state: next, outcome });

  // A fetch owns nothing, but it does own the fields it is about to overwrite:
  // adjusting during one would be silently undone when it lands.
  if (state.busy) {
    if (matchesKey(input, Key.escape)) return still("cancel");
    return still();
  }

  if (matchesKey(input, Key.escape)) return still("cancel");
  if (matchesKey(input, Key.enter)) return still("save");
  if (input === "f" || input === "F") return still("fetch");

  if (matchesKey(input, Key.up)) {
    next.field = (next.field + SAMPLING_KEYS.length - 1) % SAMPLING_KEYS.length;
    return still();
  }
  if (matchesKey(input, Key.down)) {
    next.field = (next.field + 1) % SAMPLING_KEYS.length;
    return still();
  }

  const key = SAMPLING_KEYS[next.field] as SamplingKey;

  if (input === "x" || input === "X") {
    next.params = clearParam(next.params, key);
    next.source = hasParams(next.params) ? "user" : undefined;
    next.from = undefined;
    next.note = undefined;
    return still();
  }

  const direction: 1 | -1 | undefined = matchesKey(input, Key.right) ? 1 : matchesKey(input, Key.left) ? -1 : undefined;
  if (direction === undefined) return still();

  next.params = adjustParam(next.params, key, direction);
  next.source = "user";
  next.from = undefined;
  next.note = undefined;
  return still();
}

/* --------------------------------------------------------------------------
 * Drawing
 * ------------------------------------------------------------------------ */

/** The four `‹ … ›` lines. */
export function fieldLines(state: SamplingUiState, width: number, theme: SamplingTheme): string[] {
  return SAMPLING_KEYS.map((key, position) => {
    const selected = position === state.field;
    const label = theme.fg("muted", padTo(key, LABEL_WIDTH, key.length));
    const value = `‹ ${formatParam(state.params, key)} ›`;
    const pinned = state.params[key] !== undefined;
    const hint = selected ? range(key) : pinned ? undefined : "the server decides";
    const line =
      " ".repeat(MARGIN) +
      label +
      theme.fg(selected ? "accent" : pinned ? "text" : "dim", value) +
      (hint ? " ".repeat(GAP) + theme.fg("dim", hint) : "");
    return truncateToWidth(line, width, "");
  });
}

function range(key: SamplingKey): string {
  const field = SAMPLING_FIELDS[key];
  return `${field.min}–${field.max}  step ${field.step}`;
}

/**
 * Where the numbers on screen come from.
 *
 * The whole reason this screen exists rather than a line in the sizer: values
 * that look identical mean different things depending on whether a person
 * chose them, a model's publisher did, or nobody has.
 */
export function provenanceLine(state: SamplingUiState): string {
  if (state.busy) return "reading the model card…";
  if (!hasParams(state.params)) return "nothing pinned — every request uses the server's own defaults";
  if (state.source === "hub") return `from ${state.from ?? "the model card"} — adjust to make them yours`;
  return "yours, saved with this model";
}

export function samplingTitle(state: SamplingUiState): string {
  return `Sampling  ${state.model.name}`;
}

const KEYS_FULL = "↑↓ field  ←→ adjust  f model card  x clear  ⏎ save  esc";
const KEYS_SHORT = "↑↓ ←→  f fetch  x clear  ⏎ save  esc";

/** Wrap on spaces — the same rule the sizer and the wizard use. */
export function wrap(text: string, width: number): string[] {
  if (width <= 1) return [text];
  const lines: string[] = [];
  let current = "";
  for (const word of text.split(" ")) {
    const candidate = current === "" ? word : `${current} ${word}`;
    if (visibleWidth(candidate) <= width) {
      current = candidate;
      continue;
    }
    if (current !== "") lines.push(current);
    current = word;
  }
  if (current !== "") lines.push(current);
  return lines.length > 0 ? lines : [""];
}

function block(text: string, token: SamplingToken, width: number, theme: SamplingTheme): string[] {
  return wrap(text, width - MARGIN).map((line) => " ".repeat(MARGIN) + theme.fg(token, line));
}

/** Everything the overlay draws, in order. */
export function samplingLines(state: SamplingUiState, width: number, theme: SamplingTheme): string[] {
  const pad = " ".repeat(MARGIN);
  const lines: string[] = [];

  lines.push(...block("Sent with every request Pi makes to this model.", "dim", width, theme), "");
  lines.push(...fieldLines(state, width, theme), "");
  lines.push(...block(provenanceLine(state), state.source === "hub" ? "muted" : "dim", width, theme));
  if (state.note) lines.push(...block(state.note, "warning", width, theme));
  lines.push("");

  const keys = visibleWidth(KEYS_FULL) + 2 * MARGIN <= width ? KEYS_FULL : KEYS_SHORT;
  lines.push(pad + theme.fg("dim", keys));
  return lines.map((line) => truncateToWidth(line, width, ""));
}

/** One line, for a terminal too narrow to draw on and for non-TUI modes. */
export function samplingSummary(state: SamplingUiState): string {
  if (!hasParams(state.params)) {
    return `○ ${state.model.name} — no sampling defaults; the server decides`;
  }
  const pinned = SAMPLING_KEYS.filter((key) => state.params[key] !== undefined)
    .map((key) => `${key} ${formatParam(state.params, key)}`)
    .join(" · ");
  return `⬢ ${state.model.name} — ${pinned}`;
}

/* --------------------------------------------------------------------------
 * Running it
 * ------------------------------------------------------------------------ */

/** Draw the screen and resolve once the user has decided something. */
function showSampling(
  ctx: ExtensionContext,
  initial: SamplingUiState,
  onFetch: (state: SamplingUiState) => Promise<SamplingUiState>,
): Promise<{ state: SamplingUiState; outcome: SamplingOutcome }> {
  return ctx.ui.custom<{ state: SamplingUiState; outcome: SamplingOutcome }>(
    (tui, theme, _keybindings, done) => {
      let state = initial;
      let finished = false;

      const settle = (outcome: SamplingOutcome): void => {
        if (finished) return;
        finished = true;
        done({ state, outcome });
      };

      const container = new Container();
      container.addChild(new Lines((width) => samplingLines(state, width, theme)));
      const framed = new Framed(() => samplingTitle(state), container, theme);

      // The fetch happens without closing the screen: the values it replaces
      // are on it, and a user who does not like what arrives presses `x`.
      const fetch = (): void => {
        state = { ...state, busy: true, note: undefined };
        tui.requestRender();
        void onFetch(state).then((next) => {
          if (finished) return;
          state = { ...next, busy: false };
          tui.requestRender();
        });
      };

      // A hidden overlay owns input nowhere and can no longer be escaped.
      const onResize = (): void => {
        if (terminalColumns() < MIN_OVERLAY_COLUMNS) settle("cancel");
      };
      process.stdout.on("resize", onResize);

      return {
        render: (width: number) => framed.render(width),
        invalidate: () => framed.invalidate(),
        handleInput: (input: string) => {
          const next = applySamplingKey(state, input);
          state = next.state;
          if (next.outcome === "open") {
            tui.requestRender();
            return;
          }
          if (next.outcome === "fetch") {
            fetch();
            return;
          }
          settle(next.outcome);
        },
        dispose: () => {
          process.stdout.off("resize", onResize);
        },
      };
    },
    { overlay: true, overlayOptions: overlayGeometry },
  );
}

/** `⏎` — store this model's defaults, and republish the catalogue. */
async function save(ctx: ExtensionContext, state: SamplingUiState): Promise<void> {
  const pinned = hasParams(state.params);
  const entry = pinned
    ? {
        params: { ...state.params },
        source: state.source ?? "user",
        ...(state.from ? { from: state.from } : {}),
        ...(state.source === "hub" ? { fetchedAt: new Date().toISOString() } : {}),
      }
    : undefined;

  const written = rememberSampling(state.model.id, entry);
  if (!written.ok) {
    report(ctx, `✗ Could not save: ${written.error ?? "unknown error"}`, "error");
    return;
  }

  // Pi's catalogue carries `samplingParams`, so the model's entry is stale
  // until it is re-read — the same reason the sizer refreshes after a write.
  await refreshCatalogue(ctx);
  report(
    ctx,
    pinned
      ? `✓ ${samplingSummary(state).replace(/^⬢ /, "")}`
      : `○ ${state.model.name} — sampling defaults cleared; the server decides again`,
    "info",
  );
}

/** `f` — the model's own published recommendation, when it has one. */
async function fetchFromHub(ctx: ExtensionContext, state: SamplingUiState): Promise<SamplingUiState> {
  const client = createSessionClient(ctx);
  const endpoint = await hubEndpoint(client, { timeoutMs: SAMPLING_TIMEOUT_MS });
  const result = await fetchGenerationConfig(endpoint, state.model.repoId, { timeoutMs: SAMPLING_TIMEOUT_MS });

  if (result.detail !== undefined) {
    return { ...state, note: `⚠ ${result.detail}` };
  }
  return { ...state, params: result.params, source: "hub", from: result.from, note: undefined };
}

/** What the screen opens on: the stored entry, or nothing pinned. */
export function initialState(model: SamplingModel, stored: SamplingParams, source: SamplingSource | undefined, from: string | undefined): SamplingUiState {
  return {
    model,
    params: { ...stored },
    stored: { ...stored },
    source: hasParams(stored) ? (source ?? "user") : undefined,
    from,
    field: 0,
    busy: false,
    note: undefined,
  };
}

/**
 * Open the screen for one model.
 *
 * The model's repo id is resolved first because `f` needs it, and because a
 * model that is a file on disk has no model card — which the screen should be
 * able to say before the user presses anything.
 */
export async function openSampling(ctx: ExtensionContext, modelId: string, name?: string): Promise<void> {
  const client = createSessionClient(ctx);
  const controller = new AbortController();
  const resolved = await resolveModel(client, modelId, controller.signal).catch(() => undefined);

  const model: SamplingModel = {
    id: modelId,
    name: name ?? resolved?.entry?.displayName ?? modelId,
    repoId: resolved?.local?.repoId ?? resolved?.modelPath ?? modelId,
  };

  const entry = readSamplingStore()[modelId];
  const state = initialState(model, entry?.params ?? {}, entry?.source, entry?.from);

  if (ctx.mode !== "tui" || terminalColumns() < MIN_OVERLAY_COLUMNS) {
    // Nowhere to accept anything, so nothing is written — the same rule the
    // wizard and the sizer follow.
    report(ctx, samplingSummary(state), "info");
    return;
  }

  let current = state;
  for (;;) {
    const result = await showSampling(ctx, current, (next) => fetchFromHub(ctx, next));
    current = result.state;
    if (result.outcome === "cancel") return;
    if (result.outcome === "save") {
      await save(ctx, current);
      return;
    }
  }
}

/**
 * `/unsloth sampling <model>` — find the model the user named, then open it.
 *
 * Matching is the catalogue's, shared with `/unsloth add`: nobody types
 * `ggml-org/Qwen3-4B-GGUF` twice, and an ambiguous name is reported with its
 * candidates rather than resolved by picking the first.
 */
export async function samplingByName(ctx: ExtensionContext, name: string): Promise<void> {
  const wanted = name.trim();
  if (wanted === "") {
    report(ctx, "Usage: /unsloth sampling <model>", "warning");
    return;
  }

  const client = createSessionClient(ctx);
  const entries = await listModels(client, { timeoutMs: SAMPLING_TIMEOUT_MS }).catch(() => [] as CatalogueEntry[]);
  if (entries.length === 0) {
    report(ctx, "⚠ No models — the server did not answer", "warning");
    return;
  }

  const match = matchCatalogue(entries, wanted);
  if (match.kind === "none") {
    report(ctx, `⚠ No model matching "${wanted}"`, "warning");
    return;
  }
  if (match.kind === "ambiguous") {
    report(ctx, `⚠ "${wanted}" matches ${match.candidates.join(", ")}`, "warning");
    return;
  }
  await openSampling(ctx, match.entry.id, match.entry.displayName ?? match.entry.id);
}
