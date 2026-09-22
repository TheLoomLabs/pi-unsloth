/**
 * The load progress overlay
 *
 * Opened by the panel's `⏎`, and driven entirely by `state.loading`, which the
 * supervisor's one `load-progress` poll already fills. Nothing here polls, and
 * nothing here owns a timer: it redraws when the footer does (`onPaint`).
 *
 * **Escape abandons the watching, not the load** — the same rule as the wait in
 * `waiting.ts`, for the same reason. There is no endpoint that cancels a load,
 * and ending `llama-server` halfway would leave the GPUs holding weights nobody
 * can use. So the overlay closes, the load carries on, and the footer's `◐
 * loading …` is where it is watched from instead.
 */

import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Container, Key, matchesKey, truncateToWidth } from "@earendil-works/pi-tui";

import type { ModelOverride } from "../api/lifecycle.ts";
import { formatContext } from "../provider.ts";
import { state, type LoadingState } from "../state.ts";
import { Framed, Lines, meterBar, overlayGeometry, type MeterTheme } from "./draw.ts";
import { onPaint } from "./footer.ts";

/** The subset of Pi's theme this file uses (no raw ANSI). */
export type ProgressToken = "accent" | "success" | "warning" | "error" | "muted" | "dim";

export interface ProgressTheme extends MeterTheme {
  fg(color: ProgressToken, text: string): string;
  bold(text: string): string;
}

const GIB = 1024 ** 3;

/**
 * The server's phase, in a word that can head a panel.
 *
 * Unknown phases fall back to "Loading" rather than being printed raw: a title
 * is not the place to surface a string the server invented after we shipped.
 */
const PHASE_WORDS: Readonly<Record<string, string>> = {
  resolving: "Resolving",
  loading: "Loading",
  "warming up": "Warming up",
  warming_up: "Warming up",
  warmup: "Warming up",
  ready: "Ready",
};

export function phaseTitle(phase: string | undefined, label: string): string {
  const word = (phase && PHASE_WORDS[phase.trim().toLowerCase()]) ?? "Loading";
  return `${word} ${label}`;
}

/** `19.8 / 29.0 GiB`, or just the part the server has told us. */
export function bytesLine(loaded: number | undefined, total: number | undefined): string | undefined {
  if (total !== undefined && total > 0) {
    return `${((loaded ?? 0) / GIB).toFixed(1)} / ${(total / GIB).toFixed(1)} GiB`;
  }
  if (loaded !== undefined && loaded > 0) return `${(loaded / GIB).toFixed(1)} GiB`;
  return undefined;
}

function numberList(value: unknown): number[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const ids = value.filter((entry): entry is number => typeof entry === "number" && Number.isFinite(entry));
  return ids.length > 0 ? ids : undefined;
}

/**
 * The config being loaded, in the user's terms:
 * `tensor-parallel 0+1 · ctx 192K · kv q8_0 · mtp`.
 *
 * Every part is omitted when the override does not set it, because an absent
 * field means "the server's own default", and naming a default we did not
 * choose would be inventing a fact.
 */
export function describeConfig(override: ModelOverride | undefined): string | undefined {
  if (!override) return undefined;
  const parts: string[] = [];

  const ids = numberList(override.gpu_ids);
  if (override.tensor_parallel === true) {
    parts.push(ids ? `tensor-parallel ${ids.join("+")}` : "tensor-parallel");
  } else if (ids) {
    parts.push(ids.length > 1 ? `gpu ${ids.join("+")}` : `gpu ${ids[0]}`);
  }

  const context = override.custom_context_length;
  if (typeof context === "number" && context > 0) parts.push(`ctx ${formatContext(context)}`);

  const kv = override.kv_cache_dtype;
  if (typeof kv === "string" && kv !== "") parts.push(`kv ${kv}`);

  const speculative = override.speculative_type;
  if (typeof speculative === "string" && speculative !== "" && speculative !== "none") parts.push(speculative);

  return parts.length > 0 ? parts.join(" · ") : undefined;
}

/** Everything the overlay draws from. A snapshot, never live state. */
export interface ProgressSnapshot {
  label: string;
  loading: LoadingState | undefined;
  config: string | undefined;
  /** Set once the load has failed; the overlay then waits to be dismissed. */
  error: string | undefined;
}

/** The width the bar gets, once the percentage beside it is accounted for. */
const PERCENT_WIDTH = 5;
/** The same left margin and right slack the panel uses, so `⏎` replaces it in place. */
const MARGIN = 2;

/**
 * Draw the whole overlay body. Pure, so every line is a unit test rather than
 * something you have to load a 27B model to see.
 */
export function renderProgress(snap: ProgressSnapshot, width: number, theme: ProgressTheme): string[] {
  const inner = Math.max(1, width - 2 * MARGIN);
  const pad = " ".repeat(MARGIN);
  const lines: string[] = [];
  // The bar sizes itself; the sentences do not, and a config line is easily
  // longer than a narrow panel. Clipping beats wrapping here: a second line
  // would push the escape hint off the bottom of the overlay.
  const fit = (line: string): string => truncateToWidth(line, width, "");

  // The title is the frame's — `phaseTitle`, or the failure — so the body
  // starts with what the title cannot say.
  if (snap.error) {
    lines.push(fit(pad + theme.fg("error", `✗ ${snap.error}`)));
    lines.push("");
    lines.push(fit(pad + theme.fg("dim", "esc  dismiss")));
    return lines;
  }

  const fraction = snap.loading?.fraction;
  if (fraction === undefined) {
    // The server has not sized the job yet. Saying "0%" would be a guess.
    lines.push(fit(pad + theme.fg("muted", "working…")));
  } else {
    const barWidth = Math.max(1, inner - PERCENT_WIDTH - 2);
    const percent = `${Math.max(0, Math.min(100, Math.round(fraction * 100)))}%`;
    lines.push(`${pad}${meterBar(fraction, barWidth, theme)}  ${theme.fg("muted", percent)}`);
  }

  const bytes = bytesLine(snap.loading?.loadedBytes, snap.loading?.totalBytes);
  if (bytes) lines.push(fit(pad + theme.fg("muted", bytes)));

  if (snap.config) {
    lines.push("");
    lines.push(fit(pad + theme.fg("dim", snap.config)));
  }

  lines.push("");
  lines.push(fit(pad + theme.fg("dim", "esc  continue in background")));
  return lines;
}

export type ProgressOutcome = "settled" | "backgrounded";

/**
 * Watch a load until it settles, the user leaves, or it fails.
 *
 * `work` is the load itself: this function never starts one, so a caller that
 * is not in the TUI simply awaits it and reports in its own words.
 */
export async function showLoadProgress(
  ctx: ExtensionContext,
  label: string,
  override: ModelOverride | undefined,
  work: Promise<{ ok: boolean; error: string | undefined }>,
): Promise<ProgressOutcome> {
  if (ctx.mode !== "tui") {
    await work.catch(() => undefined);
    return "settled";
  }

  const config = describeConfig(override);

  return ctx.ui.custom<ProgressOutcome>((tui, theme, _keybindings, done) => {
    let error: string | undefined;
    let finished = false;

    const settle = (outcome: ProgressOutcome): void => {
      if (finished) return;
      finished = true;
      done(outcome);
    };

    const container = new Container();
    container.addChild(
      new Lines((width) =>
        renderProgress({ label, loading: state.loading, config, error }, width, theme),
      ),
    );
    // The phase heads the frame, so the title changes as the load moves on.
    const framed = new Framed(
      () => (error ? `Load failed — ${label}` : phaseTitle(state.loading?.phase, label)),
      container,
      theme,
    );

    // Redraw whenever the supervisor's poll moves the figures. `ctx` starts the
    // tick if the status line is switched off: a load is the one time the
    // figures matter most.
    const unsubscribe = onPaint(() => tui.requestRender(), ctx);

    work.then(
      (outcome) => {
        // A failure keeps the overlay and its reason until dismissed; anything
        // else closes it.
        if (outcome.ok || !outcome.error) return settle("settled");
        error = outcome.error;
        tui.requestRender();
        return undefined;
      },
      (reason: unknown) => {
        error = reason instanceof Error ? reason.message : String(reason);
        tui.requestRender();
      },
    );

    return {
      render: (width: number) => framed.render(width),
      invalidate: () => framed.invalidate(),
      handleInput: (data: string) => {
        if (matchesKey(data, Key.escape) || matchesKey(data, Key.enter)) {
          settle(error ? "settled" : "backgrounded");
        }
      },
      dispose: () => unsubscribe(),
    };
  }, { overlay: true, overlayOptions: overlayGeometry });
}
