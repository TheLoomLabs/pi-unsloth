/**
 * The drawing primitives the overlays share: the meter bar, a width-aware line
 * block, and the terminal width.
 *
 * Both the panel's GPU bars and the load progress bar are the same shape —
 * filled cells, empty cells, and a colour that reflects how full it is. Keeping
 * them here means there is one bar in the product rather than two that drift
 * apart, and it keeps `panel.ts` and `progress.ts` free of a circular import.
 */

import { visibleWidth, type Component, type OverlayOptions } from "@earendil-works/pi-tui";

/** The token a bar is drawn in. Thresholds are */
export type MeterToken = "accent" | "warning" | "error";

/** Above this a bar is a warning: the GPU is nearly out of room. */
export const WARNING_FRACTION = 0.85;
/** Above this it is an error: a load of any size will fail. */
export const ERROR_FRACTION = 0.95;

const FILLED = "▓";
const EMPTY = "░";

export interface MeterTheme {
  fg(color: MeterToken | "dim", text: string): string;
}

export function meterToken(fraction: number): MeterToken {
  if (fraction >= ERROR_FRACTION) return "error";
  if (fraction >= WARNING_FRACTION) return "warning";
  return "accent";
}

/**
 * `▓▓▓▓░░░░` in `width` cells.
 *
 * Any non-zero fraction keeps at least one filled cell: a bar that reads as
 * completely empty next to a figure of "1.6 GiB in use" looks like a bug. The
 * figure beside it is the truth; the bar is the glance.
 */
export function meterBar(fraction: number, width: number, theme: MeterTheme): string {
  if (width <= 0) return "";
  const safe = Number.isFinite(fraction) ? fraction : 0;
  const clamped = Math.max(0, Math.min(1, safe));
  const scaled = Math.round(clamped * width);
  const filled = clamped > 0 ? Math.max(1, Math.min(width, scaled)) : 0;
  return theme.fg(meterToken(clamped), FILLED.repeat(filled)) + theme.fg("dim", EMPTY.repeat(width - filled));
}

/**
 * A block of lines that needs to know how wide it may be.
 *
 * `Text` takes a fixed string, so anything that has to fill the available
 * width — a bar, a padded column, a right-aligned hint — cannot use it. This is
 * the smallest component that closes that gap.
 */
export class Lines implements Component {
  // A plain field, not a constructor parameter property: `node --test` strips
  // types rather than compiling them, and parameter properties are the one
  // TypeScript-only construct that cannot be stripped.
  private readonly draw: (width: number) => string[];

  constructor(draw: (width: number) => string[]) {
    this.draw = draw;
  }

  render(width: number): string[] {
    return this.draw(width);
  }

  invalidate(): void {
    // Nothing cached: `draw` runs fresh on every render.
  }
}

/** Terminal width, with a sane assumption when the stream cannot say. */
export function terminalColumns(): number {
  const value = process.stdout.columns;
  return typeof value === "number" && value > 0 ? value : 80;
}

/** Terminal height, same caveat. */
export function terminalRows(): number {
  const value = process.stdout.rows;
  return typeof value === "number" && value > 0 ? value : 24;
}

/** The share of the terminal an overlay may fill, matching `maxHeight`. */
export const MAX_OVERLAY_HEIGHT_SHARE = 0.8;

/**
 * The widest an overlay ever gets, however wide the terminal is.
 *
 * A panel is a *card*, not a column of the screen. Sized as a share of the
 * terminal it degenerates on an ultrawide monitor: a 250-column terminal gave
 * a 130-column panel whose GPU bars stretched across half the desk and whose
 * model rows had their name at one end and their quant at the other, with
 * thirty blank columns in between. Reading is what a line length is for, and
 * this is the usual one.
 */
export const MAX_OVERLAY_COLUMNS = 74;

/** Breathing room kept either side of an overlay on a narrow terminal. */
const OVERLAY_MARGIN = 2;

/** How wide the overlay will actually be, for layout that must know. */
export function overlayWidth(columns: number = terminalColumns()): number {
  return Math.max(MIN_OVERLAY_COLUMNS, Math.min(MAX_OVERLAY_COLUMNS, columns - 2 * OVERLAY_MARGIN));
}

/** Pad `text` to `width` visible columns. Never truncates — the caller does. */
export function padTo(text: string, width: number, visible: number): string {
  return visible >= width ? text : text + " ".repeat(width - visible);
}

/**
 * Below this the overlays do not draw at all.
 *
 * A panel squeezed into forty columns is unreadable, and an unreadable modal is
 * worse than none: the commands still work and report through `notify`.
 */
export const MIN_OVERLAY_COLUMNS = 50;

/** Below this an overlay stops being a side panel and takes the whole width. */
export const NARROW_COLUMNS = 60;

/**
 * Where every overlay sits.
 *
 * **Centred, and never wider than it is readable.** Shared by all five overlays
 * so that `⏎` and `a` really do replace the panel *in place* rather than moving
 * it, and evaluated per render, so a resize re-centres it.
 */
export function overlayGeometry(): OverlayOptions {
  const visible = (width: number): boolean => width >= MIN_OVERLAY_COLUMNS;
  return { anchor: "center", width: overlayWidth(), maxHeight: "80%", visible };
}

/* --------------------------------------------------------------------------
 * The frame
 * ------------------------------------------------------------------------ */

export type FrameToken = "accent" | "dim" | "muted" | "text";

export interface FrameTheme {
  fg(color: FrameToken, text: string): string;
  bold(text: string): string;
}

/** Columns the frame itself costs: a border and a gutter on each side. */
export const FRAME_CHROME = 4;
/** Rows it costs: the titled top edge and the bottom edge. */
export const FRAME_ROWS = 2;

const TOP_LEFT = "╭";
const TOP_RIGHT = "╮";
const BOTTOM_LEFT = "╰";
const BOTTOM_RIGHT = "╯";
const HORIZONTAL = "─";
const VERTICAL = "│";

/**
 * A titled, rounded box around a block of lines.
 *
 * Replaces the pair of full-width rules the overlays used to draw. Two reasons,
 * and the second is the one that matters: a box *contains* its content, so an
 * overlay narrower than the terminal reads as one object rather than as text
 * floating beside an empty screen; and the title belongs on the edge, where
 * every other modern terminal UI puts it, rather than costing a line inside.
 *
 * The caller's lines are already themed, so their width is measured rather
 * than assumed, and each is padded to the inner width before the right border
 * goes on — otherwise the box's right edge would follow the ragged text.
 */
export function frameLines(title: string, body: readonly string[], width: number, theme: FrameTheme): string[] {
  const inner = Math.max(1, width - FRAME_CHROME);
  const edge = (text: string): string => theme.fg("dim", text);

  const label = title === "" ? "" : ` ${title} `;
  const labelWidth = visibleWidth(label);
  const top =
    labelWidth + 4 <= width
      ? edge(`${TOP_LEFT}${HORIZONTAL}`) +
        theme.fg("accent", theme.bold(label)) +
        edge(HORIZONTAL.repeat(Math.max(0, width - labelWidth - 3)) + TOP_RIGHT)
      : edge(TOP_LEFT + HORIZONTAL.repeat(Math.max(0, width - 2)) + TOP_RIGHT);

  const lines = body.map((line) => {
    const shown = visibleWidth(line);
    const clipped = shown > inner ? truncateVisible(line, inner) : line;
    return `${edge(VERTICAL)} ${padTo(clipped, inner, Math.min(shown, inner))} ${edge(VERTICAL)}`;
  });

  return [top, ...lines, edge(BOTTOM_LEFT + HORIZONTAL.repeat(Math.max(0, width - 2)) + BOTTOM_RIGHT)];
}

/**
 * Clip a themed line to `width` visible columns.
 *
 * Only ever a backstop — every caller lays out to the width it was given — so
 * it keeps the escape sequences it passes and counts only what is printed.
 */
function truncateVisible(line: string, width: number): string {
  let out = "";
  let shown = 0;
  let at = 0;
  while (at < line.length && shown < width) {
    if (line[at] === "\u001b") {
      const end = line.indexOf("m", at);
      if (end === -1) break;
      out += line.slice(at, end + 1);
      at = end + 1;
      continue;
    }
    const character = line[at] as string;
    const step = visibleWidth(character);
    if (shown + step > width) break;
    out += character;
    shown += step;
    at += 1;
  }
  return out + (line.includes("\u001b") ? "\u001b[0m" : "");
}

/**
 * A component that frames whatever it wraps.
 *
 * The child is rendered at the inner width, so every layout inside stays
 * width-aware and none of them has to know a border exists.
 */
export class Framed implements Component {
  private readonly title: () => string;
  private readonly child: Component;
  private readonly theme: FrameTheme;

  constructor(title: string | (() => string), child: Component, theme: FrameTheme) {
    this.title = typeof title === "function" ? title : () => title;
    this.child = child;
    this.theme = theme;
  }

  render(width: number): string[] {
    return frameLines(this.title(), this.child.render(Math.max(1, width - FRAME_CHROME)), width, this.theme);
  }

  invalidate(): void {
    this.child.invalidate?.();
  }
}
