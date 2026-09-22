/**
 * The footer status line — one glance: which model, is it busy, how much VRAM
 * is left. On by default, and switchable off, in which case it costs neither
 * the line nor the 4 s poll.
 *
 * Split in two on purpose: `renderFooter` is pure, so every state is a unit
 * test rather than something you have to load a 27B model to see; the tick
 * below is the only part that touches a timer or the network.
 */

import { getSystemGpu, computeDevices } from "../api/system.ts";
import { hasUI } from "../session.ts";
import { createSessionClient, formatContext } from "../provider.ts";
import { state, type GpuUsage, type SessionState } from "../state.ts";
import { meterBar, terminalColumns } from "./draw.ts";

/** The status key we own in Pi's footer. */
export const STATUS_KEY = "unsloth";

/** VRAM poll interval. Slow enough to keep the render quiet. */
export const TICK_MS = 4_000;

/** Below this the footer keeps the glyph and the model name, nothing else. */
export const NARROW_COLUMNS = 60;

/** A poll must not outlive its own tick. */
const POLL_TIMEOUT_MS = 2_000;

/** Below this the VRAM meter is dropped: the figure beside it says more. */
export const METER_COLUMNS = 96;
/** Cells the meter gets. Small on purpose — it is a glance, not a gauge. */
const METER_WIDTH = 8;

/** The subset of Pi's theme this file uses (no raw ANSI). */
export type FooterToken = "accent" | "success" | "warning" | "muted" | "dim" | "error" | "text";

export interface FooterTheme {
  fg(color: FooterToken, text: string): string;
}

/** 1 decimal, always — `0.0`, `21.4`. Ragged widths make the line jump. */
function gib(value: number): string {
  return value.toFixed(1);
}

/**
 * `39.9/48.0 GiB` — the machine, as one number over another.
 *
 * The per-GPU breakdown used to be here (`21.4+18.2/48.0`). It moved to the
 * panel, where there are bars to hang it on: the footer's job is the glance,
 * and "how much of this machine is left" is one subtraction, while "where did
 * it go" is a question you open the panel to answer.
 */
export function formatVram(gpus: readonly GpuUsage[]): string | undefined {
  if (gpus.length === 0) return undefined;
  const total = gpus.reduce((sum, gpu) => sum + gpu.totalGb, 0);
  const used = gpus.reduce((sum, gpu) => sum + gpu.usedGb, 0);
  return `${gib(used)}/${gib(total)} GiB`;
}

/** How full the machine is, 0–1, for the meter and its colour. */
export function vramFraction(gpus: readonly GpuUsage[]): number | undefined {
  if (gpus.length === 0) return undefined;
  const total = gpus.reduce((sum, gpu) => sum + gpu.totalGb, 0);
  if (total <= 0) return undefined;
  return gpus.reduce((sum, gpu) => sum + gpu.usedGb, 0) / total;
}

/** Everything the line can be built from. A snapshot, never live state. */
export type FooterSnapshot = Pick<
  SessionState,
  "server" | "activeModelLabel" | "activeContextWindow" | "loading" | "streaming" | "gpus"
>;

export function snapshot(): FooterSnapshot {
  return {
    server: state.server,
    activeModelLabel: state.activeModelLabel,
    activeContextWindow: state.activeContextWindow,
    loading: state.loading,
    streaming: state.streaming,
    gpus: state.gpus,
  };
}

/**
 * One themed piece of the line.
 *
 * The footer used to be a single string in a single colour. Colouring the
 * pieces separately is what lets the model's name lead and the numbers recede,
 * which is the difference between a status line and a sentence.
 */
interface Chunk {
  text: string;
  token: FooterToken;
  /** 0–1: draw a meter before this chunk, when the terminal is wide enough. */
  meter?: number;
}

interface Line {
  /** The first chunk is the state, and survives every width. */
  chunks: Chunk[];
  /** A state marker, appended without a separator and never dropped. */
  marker?: string;
}

function compose(snap: FooterSnapshot): Line | undefined {
  // Starting and loading share a shape: something is happening, wait for it.
  if (snap.server === "starting") {
    return { chunks: [{ text: "◐ starting unsloth", token: "warning" }] };
  }
  if (snap.loading) {
    const percent =
      snap.loading.fraction === undefined
        ? undefined
        : `${Math.max(0, Math.min(100, Math.round(snap.loading.fraction * 100)))}%`;
    return {
      chunks: [
        { text: `◐ loading ${snap.loading.label}`, token: "warning" },
        ...(percent ? [{ text: percent, token: "muted" as FooterToken }] : []),
      ],
    };
  }

  switch (snap.server) {
    case "unknown":
      // Nothing observed yet: say nothing rather than guess at a state.
      return undefined;
    case "offline":
      return { chunks: [{ text: "◌ unsloth offline", token: "dim" }] };
    case "unreachable":
      return { chunks: [{ text: "⚠ unsloth unreachable", token: "error" }] };
    case "unauthorized":
      return { chunks: [{ text: "⚠ unsloth unauthorized", token: "error" }] };
    case "up":
      break;
  }

  const vram = formatVram(snap.gpus);
  const fraction = vramFraction(snap.gpus);
  const usage: Chunk[] = vram
    ? [{ text: vram, token: "text", ...(fraction !== undefined ? { meter: fraction } : {}) }]
    : [];

  if (!snap.activeModelLabel) {
    return { chunks: [{ text: "○ unsloth idle", token: "muted" }, ...usage] };
  }

  const context =
    snap.activeContextWindow !== undefined ? formatContext(snap.activeContextWindow) : undefined;
  return {
    chunks: [
      { text: `⬢ ${snap.activeModelLabel}`, token: snap.streaming ? "success" : "accent" },
      // The label has already had a trailing context stripped from it
      // (`shortLabel`), so this cannot be the same fact twice — but a model
      // genuinely *called* `…-128K` would be, and is not worth repeating.
      ...(context && !snap.activeModelLabel.includes(context)
        ? [{ text: context, token: "muted" as FooterToken }]
        : []),
      ...usage,
    ],
    ...(snap.streaming ? { marker: "▸" } : {}),
  };
}

/**
 * Build the status line, or `undefined` when there is nothing worth saying.
 *
 * `columns` is the terminal width. Two responsive rules, and both shed detail
 * rather than state: below `NARROW_COLUMNS` only the first chunk survives, and
 * below `METER_COLUMNS` the meter goes but its figure stays — the figure is the
 * fact, the bar is the glance.
 */
export function renderFooter(snap: FooterSnapshot, columns: number, theme: FooterTheme): string | undefined {
  const line = compose(snap);
  if (!line) return undefined;
  const shown = columns < NARROW_COLUMNS ? line.chunks.slice(0, 1) : line.chunks;

  const painted = shown.map((chunk) => {
    const body = theme.fg(chunk.token, chunk.text);
    if (chunk.meter === undefined || columns < METER_COLUMNS) return body;
    return `${meterBar(chunk.meter, METER_WIDTH, theme)} ${body}`;
  });

  // The marker says *what the model is doing*, so it survives a narrow
  // terminal; the numbers before it are detail, and do not.
  return painted.join(theme.fg("dim", " · ")) + (line.marker ? ` ${theme.fg("success", line.marker)}` : "");
}

/**
 * The minimum of Pi's context that the footer needs. Keeping it structural
 * means the tick can be driven by a fake in tests.
 */
export interface FooterContext {
  hasUI: boolean;
  ui: { setStatus(key: string, text: string | undefined): void; theme: FooterTheme };
  modelRegistry: { getApiKeyForProvider(provider: string): Promise<string | undefined> };
}

let tick: ReturnType<typeof setInterval> | undefined;
let polling = false;

/**
 * Anything that draws from the same state as the footer.
 *
 * The overlays show live VRAM and live load progress, but a TUI component only
 * redraws when something asks it to. Rather than give each overlay its own
 * timer and its own poll — two more things to clear on shutdown, and two more
 * requests per tick — they subscribe here and re-render when the footer does.
 * One timer, one poll, several views.
 */
const painters = new Set<() => void>();

/** Redraw with the footer. Returns the unsubscribe; call it from `dispose`. */
export function onPaint(listener: () => void, ctx?: FooterContext): () => void {
  painters.add(listener);
  // An overlay opening with the footer hidden is the case that needs the tick
  // started; one closing is the case that may let it stop.
  if (ctx) ensureTick(ctx);
  return () => {
    painters.delete(listener);
    idleTick();
  };
}

/** Redraw from current state. Cheap and synchronous — call it after any change. */
export function paint(ctx: FooterContext): void {
  for (const painter of painters) {
    try {
      painter();
    } catch {
      // A view that cannot draw must not stop the status line from updating.
    }
  }
  // Safe read, not `ctx.hasUI`: paint is reached from the 4 s tick and from
  // background work that outlives its session, and on a stale ctx the raw
  // property throws where a skipped repaint is the right answer.
  if (!hasUI(ctx)) return;
  // Hidden means the line is *cleared*, not merely left stale — and the
  // painters above still ran, because an overlay the user has open is a
  // different question from whether the status line is drawn.
  ctx.ui.setStatus(
    STATUS_KEY,
    state.footerVisible ? renderFooter(snapshot(), terminalColumns(), ctx.ui.theme) : undefined,
  );
}

/**
 * What `/unsloth footer [on|off]` asks for: `undefined` when it asks for
 * something that is not a word this subcommand knows.
 *
 * Pure, because "no argument toggles, a wrong argument is a usage error, and
 * `on` twice is not a toggle" is three rules and each of them is one line of
 * test rather than a session to try it in.
 */
export function footerTarget(word: string | undefined, current: boolean): boolean | undefined {
  if (word === undefined || word === "") return !current;
  const normalised = word.trim().toLowerCase();
  if (normalised === "on") return true;
  if (normalised === "off") return false;
  return undefined;
}

/**
 * Turn the status line on or off.
 *
 * The tick follows the *views*, not the line: an open panel draws live VRAM
 * from the same poll, so hiding the footer while the panel is up must not
 * freeze its bars. `wanted` is the whole rule.
 */
export function setFooterVisible(ctx: FooterContext, visible: boolean): void {
  state.footerVisible = visible;
  if (visible) ensureTick(ctx);
  else idleTick();
  paint(ctx);
}

/**
 * Read live VRAM into state. Never throws: the footer showing a stale figure
 * for four seconds is not worth an error path.
 */
export async function pollVram(ctx: FooterContext): Promise<void> {
  try {
    const gpu = await getSystemGpu(createSessionClient(ctx), { timeoutMs: POLL_TIMEOUT_MS });
    state.gpus = computeDevices(gpu.devices).map((device) => ({
      index: device.index,
      usedGb: device.vramUsedGb ?? 0,
      totalGb: device.memoryTotalGb ?? 0,
    }));
  } catch {
    // Down, or busy loading. `state.server` already carries the explanation.
  }
}

/**
 * Start the 4 s tick. Idempotent — a second call is a no-op, not a second
 * timer.
 *
 * Polling pauses while a turn is streaming: the render loop is busy, and the
 * VRAM figure cannot move much once the weights are resident anyway. The line
 * is still repainted each tick so the `▸` marker appears and clears promptly.
 */
export function startFooter(ctx: FooterContext): void {
  if (!hasUI(ctx)) return;
  ensureTick(ctx);
  paint(ctx);
}

/** Does anything still need the poll: the line itself, or an open overlay? */
function wanted(): boolean {
  return state.footerVisible || painters.size > 0;
}

/** Start the tick if something wants it and there is not one already. */
function ensureTick(ctx: FooterContext): void {
  if (!hasUI(ctx) || tick || !wanted()) return;
  tick = setInterval(() => {
    if (state.streaming || polling) {
      paint(ctx);
      return;
    }
    polling = true;
    void pollVram(ctx).finally(() => {
      polling = false;
      paint(ctx);
    });
  }, TICK_MS);
  // The footer must never be the reason Pi cannot exit.
  tick.unref?.();
}

/** Stop the tick once nothing is left to draw. Turning the footer off is the
 * point: it has to cost the poll as well as the line, or it only hides it. */
function idleTick(): void {
  if (!tick || wanted()) return;
  clearInterval(tick);
  tick = undefined;
  polling = false;
}

/**
 * Stop the tick and clear the line. Idempotent, and safe to call in a session
 * that never started one — `session_shutdown` fires in both cases.
 */
export function stopFooter(ctx: FooterContext): void {
  if (tick) {
    clearInterval(tick);
    tick = undefined;
  }
  polling = false;
  painters.clear();
  // A shutdown that arrives after the session is already gone still has to
  // clear the timers above; it simply has no line left to clear.
  if (hasUI(ctx)) ctx.ui.setStatus(STATUS_KEY, undefined);
}

/** Test seam: is a tick currently registered? */
export function footerRunning(): boolean {
  return tick !== undefined;
}
