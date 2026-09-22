/**
 * The panel — `ctrl+alt+u` and `/unsloth`.
 *
 * One overlay that answers the three questions the footer has no room for:
 * where is the VRAM going, what is on disk, and what would it take to load it.
 * `⏎` loads, `u` frees the GPUs, `r` re-reads the catalogue, escape closes.
 *
 * The rendering is pure and the fetching is not, deliberately: every line is a
 * unit test rather than something you have to own two 24 GB cards to look at.
 *
 * ⚠ The panel drives the **server**, not Pi's model selection — there is no
 * extension API for choosing a model, and `ctrl+p` already does it properly.
 * Loading a model the session is not pointed at is still useful (it is what
 * `pil` did), so it is allowed and then said out loud.
 */

import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import {
  Container,
  Key,
  SelectList,
  matchesKey,
  truncateToWidth,
  visibleWidth,
  type SelectItem,
  type SelectListTheme,
} from "@earendil-works/pi-tui";

import { findOverride, getOverrides, type ModelOverride, type Overrides } from "../api/lifecycle.ts";
import { listLocalModels, listModels, type CatalogueEntry, type LocalModel } from "../api/models.ts";
import { computeDevices, getSystemGpu } from "../api/system.ts";
import { displayGpus, mergeIdleVram } from "../hardware/detect.ts";
import { createSessionClient, findLocal, formatContext } from "../provider.ts";
import { state, type GpuUsage } from "../state.ts";
import { cancelEnsure, refreshCatalogue, report, runTunedLoad, unloadAndReport } from "../supervisor.ts";
import {
  FRAME_ROWS,
  Framed,
  MAX_OVERLAY_HEIGHT_SHARE,
  MIN_OVERLAY_COLUMNS,
  Lines,
  meterBar,
  overlayGeometry,
  padTo,
  terminalColumns,
  terminalRows,
  type MeterTheme,
} from "./draw.ts";
import { onPaint } from "./footer.ts";
import { showLoadProgress } from "./progress.ts";
import { openSampling } from "./sampling.ts";
import { openSizer } from "./sizer.ts";

/** One panel open is four local requests; none of them may hang the overlay. */
const PANEL_TIMEOUT_MS = 5_000;

/** Rows on screen at once before the list scrolls, however tall the terminal. */
const MAX_VISIBLE_ROWS = 12;
/** …and the fewest worth showing before scrolling is all the panel does. */
const MIN_VISIBLE_ROWS = 3;
/**
 * Lines the panel spends on everything that is not a model row: the frame's two
 * edges, two bars and the blank under them, the blank before the hints and the
 * hints, plus `SelectList`'s own scroll counter.
 */
const CHROME_LINES = 8 + FRAME_ROWS;

/**
 * Left margin, and the slack kept on the right.
 *
 * Two columns, because that is exactly what `SelectList` indents its rows by —
 * so the bars, the title and the hints line up with the models instead of
 * sitting two columns to their left.
 */
const MARGIN = 2;
const GAP = 2;

/** Column widths. Sized to the longest thing each column can hold. */
const GLYPH_WIDTH = 2;
const QUANT_WIDTH = 6;
const CONTEXT_WIDTH = 5;
const STATUS_WIDTH = 11;
const MIN_NAME_WIDTH = 12;

/** Below this a bar says less than the figure beside it, so it is dropped. */
const MIN_BAR_WIDTH = 6;
/**
 * …and above this it says no more than it did at half the length.
 *
 * A meter is read as a proportion, and a proportion does not get easier to see
 * by being a foot wide. Capping it also keeps the figures beside the bars in a
 * column the eye can run down.
 */
const MAX_BAR_WIDTH = 26;

/**
 * The widest a model's name is allowed to get.
 *
 * Without a cap the name column absorbs every spare column and pushes quant,
 * context and placement to the far right of the panel, which is how a table
 * turns into two unrelated lists. The slack goes between the two groups
 * instead, where it reads as a gutter.
 */
const MAX_NAME_WIDTH = 26;

/** The subset of Pi's theme this file uses (no raw ANSI). */
export type PanelToken = "accent" | "success" | "warning" | "error" | "muted" | "dim" | "text";

export interface PanelTheme extends MeterTheme {
  fg(color: PanelToken, text: string): string;
  bold(text: string): string;
}

/**
 * Why a row can or cannot be loaded.
 *
 * `wont-fit` is narrower here than 's eventual meaning: it is a **placement**
 * verdict — the tuned config names GPUs this machine does not have — not an
 * estimator verdict. The estimator-backed "no context fits on these cards"
 * answer needs M4's budgets and M5's search; this is the part that can be known
 * for free, and it is the part that is certain.
 */
export type RowFit = "sized" | "unsized" | "wont-fit" | "incomplete" | "missing";

export interface PanelRow {
  id: string;
  /** What the user calls it — never the whole `name · quant · context` label. */
  name: string;
  quant: string | undefined;
  /** Tuned context from the override. Absent means the model is not sized. */
  contextTokens: number | undefined;
  /** `tp 0+1` / `gpu 1`, when the override says where the model goes. */
  placement: string | undefined;
  loaded: boolean;
  fit: RowFit;
  /** What `⏎` needs. Absent when the model cannot be resolved on disk. */
  modelPath: string | undefined;
  override: ModelOverride | undefined;
}

function numberList(value: unknown): number[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const ids = value.filter((entry): entry is number => typeof entry === "number" && Number.isFinite(entry));
  return ids.length > 0 ? ids : undefined;
}

/**
 * Where a tuned override puts the model, and whether that is possible here.
 *
 * `gpuIndices` empty means the topology is unknown — the server is down, or
 * reports no devices. That is not evidence of a bad fit, so nothing is claimed.
 */
export function placementOf(
  override: ModelOverride,
  gpuIndices: readonly number[],
): { text: string | undefined; fits: boolean } {
  const ids = numberList(override.gpu_ids);
  const tensor = override.tensor_parallel === true;
  if (!ids) return { text: tensor ? "tp all" : undefined, fits: true };
  const fits = gpuIndices.length === 0 || ids.every((id) => gpuIndices.includes(id));
  const prefix = tensor || ids.length > 1 ? "tp" : "gpu";
  return { text: `${prefix} ${ids.join("+")}`, fits };
}

export interface RowInputs {
  entries: readonly CatalogueEntry[];
  local: readonly LocalModel[];
  overrides: Overrides;
  /** Compute GPU indices `/api/system` reports. Empty when it could not say. */
  gpuIndices: readonly number[];
}

/**
 * Turn the catalogue into rows.
 *
 * Loaded models first — they are the ones holding the VRAM the user is looking
 * at — then alphabetically, so the order does not shuffle between refreshes.
 */
export function buildRows(inputs: RowInputs): PanelRow[] {
  const rows = inputs.entries.map((entry): PanelRow => {
    const local = findLocal(entry, inputs.local);
    const override = findOverride(inputs.overrides, [local?.repoId, local?.path, entry.id], entry.quant);
    const context = typeof override?.custom_context_length === "number" ? override.custom_context_length : undefined;
    const placement = override ? placementOf(override, inputs.gpuIndices) : undefined;

    let fit: RowFit;
    if (!local) fit = "missing";
    else if (local.partial) fit = "incomplete";
    else if (!override) fit = "unsized";
    else if (placement && !placement.fits) fit = "wont-fit";
    else fit = "sized";

    return {
      id: entry.id,
      name: entry.displayName ?? local?.displayName ?? entry.id,
      quant: entry.quant,
      contextTokens: context,
      placement: placement?.text,
      loaded: entry.loaded,
      fit,
      modelPath: local?.repoId ?? local?.path,
      override,
    };
  });

  return rows.sort((a, b) => {
    if (a.loaded !== b.loaded) return a.loaded ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
}

interface Cell {
  text: string;
  token: PanelToken;
}

/**
 * The fourth column: where the model goes, or why it cannot go anywhere.
 *
 * A status word beats a placement whenever both could be shown — "not sized"
 * is something to act on, "tp 0+1" is something to know.
 */
export function statusCell(row: PanelRow): Cell | undefined {
  switch (row.fit) {
    case "missing":
      return { text: "not on disk", token: "warning" };
    case "incomplete":
      return { text: "incomplete", token: "warning" };
    case "unsized":
      return { text: "not sized", token: "dim" };
    case "wont-fit":
      return { text: "won't fit", token: "error" };
    case "sized":
      return row.placement ? { text: row.placement, token: "muted" } : undefined;
  }
}

/** How much of a row fits. Detail drops; state never does. */
export type RowTier = "full" | "compact" | "minimal";

const FULL_ROW_WIDTH =
  GLYPH_WIDTH + MIN_NAME_WIDTH + GAP + QUANT_WIDTH + GAP + CONTEXT_WIDTH + GAP + STATUS_WIDTH;
const COMPACT_ROW_WIDTH = GLYPH_WIDTH + MIN_NAME_WIDTH + GAP + STATUS_WIDTH;

export function rowTier(width: number): RowTier {
  if (width >= FULL_ROW_WIDTH) return "full";
  if (width >= COMPACT_ROW_WIDTH) return "compact";
  return "minimal";
}

/** Draw one column: themed, clipped, and padded to exactly `width`. */
function column(
  cell: Cell | undefined,
  width: number,
  theme: PanelTheme,
  align: "left" | "right" = "left",
): string {
  const clipped = truncateToWidth(cell?.text ?? "", width, "");
  const shown = visibleWidth(clipped);
  const painted = cell ? theme.fg(cell.token, clipped) : clipped;
  return align === "right" ? " ".repeat(Math.max(0, width - shown)) + painted : padTo(painted, width, shown);
}

/**
 * One list row, themed and padded to `width`.
 *
 * `width` is what `SelectList` offers after its own prefix, so the responsive
 * rules are decided here rather than from the terminal size — the panel is an
 * overlay, and its width is not the terminal's.
 */
/**
 * How wide the name column should be for a whole list.
 *
 * Sized to the longest name there actually is, not to the space there happens
 * to be. A column stretched to the panel's width puts each row's name at one
 * end and its quant at the other, which is two lists rather than a table.
 */
export function nameColumnWidth(rows: readonly PanelRow[]): number {
  const longest = rows.reduce((widest, row) => Math.max(widest, visibleWidth(row.name)), 0);
  return Math.max(MIN_NAME_WIDTH, Math.min(MAX_NAME_WIDTH, longest));
}

export function renderRow(
  row: PanelRow,
  width: number,
  theme: PanelTheme,
  selected: boolean,
  /** The list's shared name column. Defaults to whatever this row needs. */
  nameColumn?: number,
): string {
  const tier = rowTier(width);
  const glyph = row.loaded ? theme.fg("success", "●") : theme.fg("muted", "○");
  const status = statusCell(row);
  const contextCell: Cell =
    row.contextTokens !== undefined
      ? { text: formatContext(row.contextTokens), token: "accent" }
      : { text: "—", token: "dim" };

  if (tier === "minimal") {
    const name = column({ text: row.name, token: selected ? "accent" : "text" }, width - GLYPH_WIDTH, theme);
    return `${glyph} ${name}`;
  }

  if (tier === "compact") {
    // Quant and the exact placement are detail; the state word is not. A sized
    // model has no state word, so its context takes the column instead.
    const tail = row.fit === "sized" ? contextCell : status;
    const nameWidth = width - GLYPH_WIDTH - GAP - STATUS_WIDTH;
    const name = column({ text: row.name, token: selected ? "accent" : "text" }, nameWidth, theme);
    return `${glyph} ${name}${" ".repeat(GAP)}${column(tail, STATUS_WIDTH, theme)}`;
  }

  // Columns sized to their content and packed from the left, so the row reads
  // as one line rather than as two groups with a hole between them. Whatever
  // is left over is the card's padding, which is where slack belongs.
  const nameWidth = Math.max(
    MIN_NAME_WIDTH,
    Math.min(nameColumn ?? MAX_NAME_WIDTH, width - (FULL_ROW_WIDTH - MIN_NAME_WIDTH)),
  );
  const name = column({ text: row.name, token: selected ? "accent" : "text" }, nameWidth, theme);
  const quant = column(row.quant ? { text: row.quant, token: "muted" } : undefined, QUANT_WIDTH, theme);
  const context = column(contextCell, CONTEXT_WIDTH, theme, "right");
  return [`${glyph} ${name}`, quant, context, column(status, STATUS_WIDTH, theme)].join(" ".repeat(GAP));
}

/**
 * The GPU bars above the list.
 *
 * Every number comes from `/api/system`; the `display` tag comes from the idle
 * measurement in `src/hardware/detect.ts`. Nothing here knows how many GPUs
 * there are or how big they are.
 */
export function renderGpuBars(
  gpus: readonly GpuUsage[],
  displayIndices: readonly number[] | undefined,
  width: number,
  theme: PanelTheme,
): string[] {
  if (gpus.length === 0) return [];

  const labels = gpus.map((gpu) => `GPU ${gpu.index}`);
  const figures = gpus.map((gpu) => `${gpu.usedGb.toFixed(1)}/${gpu.totalGb.toFixed(1)}`);
  const labelWidth = Math.max(...labels.map((label) => label.length));
  const figureWidth = Math.max(...figures.map((figure) => figure.length));

  // Shed in order of how much each part says: the bar is a glance, the tag is a
  // property of the machine, the figure is the fact.
  const available = width - 2 * MARGIN;
  const fixed = (withTag: boolean): number =>
    labelWidth + GAP + figureWidth + (withTag ? GAP + "display".length : 0);
  const wanted = gpus.some((gpu) => displayIndices?.includes(gpu.index));
  const tagged = wanted && fixed(true) <= available;
  const tagWidth = tagged ? "display".length : 0;
  const barWidth = Math.min(MAX_BAR_WIDTH, available - fixed(tagged) - GAP);
  const withBar = barWidth >= MIN_BAR_WIDTH;

  return gpus.map((gpu, position) => {
    const fraction = gpu.totalGb > 0 ? gpu.usedGb / gpu.totalGb : 0;
    const name = labels[position] ?? "";
    const label = theme.fg("muted", padTo(name, labelWidth, name.length));
    const figure = column({ text: figures[position] ?? "", token: "text" }, figureWidth, theme, "right");
    const tag = tagged
      ? " ".repeat(GAP) +
        column(displayIndices?.includes(gpu.index) ? { text: "display", token: "warning" } : undefined, tagWidth, theme)
      : "";
    const bar = withBar ? meterBar(fraction, barWidth, theme) + " ".repeat(GAP) : "";
    const line = `${" ".repeat(MARGIN)}${label}${" ".repeat(GAP)}${bar}${figure}${tag}`;
    // Last resort: a terminal below the floor should still not corrupt a line.
    return truncateToWidth(line, width, "");
  });
}

/** One line for a terminal too narrow to draw in, and for `pi -p`. */
export function panelSummary(data: PanelData): string {
  if (data.problem) return `◌ unsloth — ${data.problem}`;
  const loaded = data.rows.filter((row) => row.loaded).map((row) => row.name);
  const vram =
    data.gpus.length > 0
      ? `${data.gpus.map((gpu) => gpu.usedGb.toFixed(1)).join("+")}/${data.gpus
          .reduce((sum, gpu) => sum + gpu.totalGb, 0)
          .toFixed(1)} GiB`
      : undefined;
  const head = loaded.length > 0 ? `⬢ ${loaded.join(", ")}` : "○ unsloth idle";
  const tail = [`${data.rows.length} model${data.rows.length === 1 ? "" : "s"}`, vram].filter(Boolean);
  return [head, ...tail].join(" · ");
}

/**
 * The frame's title: the name, and the one figure the list cannot show.
 *
 * "How much of this machine is in use" is what the panel is opened to find
 * out, and a total belongs where the eye lands first rather than being left to
 * be added up from the bars.
 */
export function panelTitle(data: PanelData): string {
  if (data.gpus.length === 0) return "Unsloth";
  const used = data.gpus.reduce((sum, gpu) => sum + gpu.usedGb, 0);
  const total = data.gpus.reduce((sum, gpu) => sum + gpu.totalGb, 0);
  return `Unsloth  ${used.toFixed(1)}/${total.toFixed(1)} GiB`;
}

export interface PanelData {
  rows: PanelRow[];
  gpus: GpuUsage[];
  /** `undefined` means nothing has been measured at idle yet, so nothing is tagged. */
  displayGpus: number[] | undefined;
  /** Why there is nothing to show, when there is nothing to show. */
  problem: string | undefined;
}

/**
 * Everything the panel draws, in one round trip.
 *
 * The catalogue is required; the rest are enrichments, so a GPU query that
 * fails costs the bars rather than the panel.
 */
export async function fetchPanelData(ctx: ExtensionContext, options: { signal?: AbortSignal } = {}): Promise<PanelData> {
  const client = createSessionClient(ctx);
  const call = options.signal
    ? { signal: options.signal, timeoutMs: PANEL_TIMEOUT_MS }
    : { timeoutMs: PANEL_TIMEOUT_MS };

  const [gpu, entries, local, overrides] = await Promise.all([
    getSystemGpu(client, call).then(
      (result) => result,
      () => undefined,
    ),
    listModels(client, call).then(
      (result) => result,
      (error: unknown) => (error instanceof Error ? error : new Error(String(error))),
    ),
    listLocalModels(client, call).then(
      (result) => result.models,
      () => [] as LocalModel[],
    ),
    getOverrides(client, call).then(
      (result) => result,
      () => ({}) as Overrides,
    ),
  ]);

  const gpus: GpuUsage[] = gpu
    ? computeDevices(gpu.devices).map((device) => ({
        index: device.index,
        usedGb: device.vramUsedGb ?? 0,
        totalGb: device.memoryTotalGb ?? 0,
      }))
    : [];
  if (gpu) state.gpus = gpus;

  if (entries instanceof Error) {
    return { rows: [], gpus, displayGpus: displayGpus(state.idleGpus), problem: entries.message };
  }

  // re-measure idle VRAM whenever the panel opens and nothing is resident. It
  // is the only moment the figure means anything — and even then it is merged
  // rather than replaced, because "no model loaded" and "the driver has
  // finished handing the memory back" are not the same instant.
  if (gpus.length > 0 && !entries.some((entry) => entry.loaded)) {
    state.idleGpus = mergeIdleVram(state.idleGpus, gpus);
  }

  const rows = buildRows({ entries, local, overrides, gpuIndices: gpus.map((entry) => entry.index) });
  return { rows, gpus, displayGpus: displayGpus(state.idleGpus), problem: undefined };
}

export type PanelAction =
  | { kind: "close" }
  | { kind: "refresh" }
  | { kind: "unload" }
  | { kind: "load"; row: PanelRow }
  | { kind: "size"; row: PanelRow }
  | { kind: "sampling"; row: PanelRow };

function selectTheme(theme: PanelTheme): SelectListTheme {
  return {
    selectedPrefix: (text: string) => theme.fg("accent", text),
    // The row is already fully themed by `truncatePrimary`; wrapping it again
    // would flatten every colour inside it after the first reset. Only the
    // two-column "→ " prefix that SelectList adds itself is ours to paint.
    selectedText: (line: string) => theme.fg("accent", line.slice(0, 2)) + line.slice(2),
    description: (text: string) => theme.fg("muted", text),
    scrollInfo: (text: string) => theme.fg("dim", text),
    noMatch: (text: string) => theme.fg("warning", text),
  };
}

/**
 * What sits above the list: the GPU bars, and nothing else.
 *
 * The panel's name used to be the first line inside it; it is now the frame's
 * title, which is both where a title belongs and one line of list back.
 */
export function headerLines(data: PanelData, width: number, theme: PanelTheme): string[] {
  const bars = renderGpuBars(data.gpus, data.displayGpus, width, theme);
  return bars.length > 0 ? [...bars, ""] : [];
}

export function emptyLines(data: PanelData, _width: number, theme: PanelTheme): string[] {
  const pad = " ".repeat(MARGIN);
  if (data.problem) return [pad + theme.fg("dim", `◌ unsloth unavailable — ${data.problem}`), ""];
  return [pad + theme.fg("muted", "○ no models downloaded"), ""];
}

/**
 * The key hints, in the longest form that fits.
 *
 * Keys are shed in reverse order of how much they offer: `s sampling` first,
 * then `r refresh`, then the arrows, then `u unload`. `⏎ load`, `a size` and
 * `esc` survive every width — the first two are the answers to the two states a
 * row can be in, and the third is how the overlay is closed at all.
 */
export function hintLines(hasRows: boolean, width: number, theme: PanelTheme): string[] {
  const pad = " ".repeat(MARGIN);
  const tiers = hasRows
    ? [
        "↑↓ select  ⏎ load  a size  s sampling  u unload  r refresh  esc",
        "↑↓ select  ⏎ load  a size  u unload  r refresh  esc",
        "⏎ load  a size  u unload  esc",
        "⏎ load  a size  esc",
      ]
    : ["u unload   r refresh   esc", "u unload  esc"];
  const hint = tiers.find((tier) => visibleWidth(tier) + 2 * MARGIN <= width) ?? tiers[tiers.length - 1] ?? "esc";
  return [pad + theme.fg("dim", hint)];
}

/**
 * How many rows the list may show.
 *
 * The overlay is capped at `maxHeight`, and anything past that cap is clipped
 * from the bottom — which is where the key hints live. Better to scroll the
 * list than to lose the line that says escape closes the panel.
 */
export function visibleRows(count: number, rows: number): number {
  const budget = Math.floor(rows * MAX_OVERLAY_HEIGHT_SHARE) - CHROME_LINES;
  return Math.max(MIN_VISIBLE_ROWS, Math.min(count, MAX_VISIBLE_ROWS, budget));
}

/** Draw the panel once and resolve with what the user asked for. */
function showPanel(ctx: ExtensionContext, data: PanelData): Promise<PanelAction> {
  return ctx.ui.custom<PanelAction>(
    (tui, theme, _keybindings, done) => {
      let finished = false;
      const settle = (action: PanelAction): void => {
        if (finished) return;
        finished = true;
        done(action);
      };

      const container = new Container();
      container.addChild(new Lines((width) => headerLines(data, width, theme)));

      const byId = new Map(data.rows.map((row) => [row.id, row]));
      const nameColumn = nameColumnWidth(data.rows);
      let list: SelectList | undefined;
      if (data.rows.length > 0) {
        const items: SelectItem[] = data.rows.map((row) => ({ value: row.id, label: row.name }));
        list = new SelectList(items, visibleRows(items.length, terminalRows()), selectTheme(theme), {
          truncatePrimary: ({ item, isSelected, maxWidth }) => {
            const row = byId.get(item.value);
            return row ? renderRow(row, maxWidth, theme, isSelected, nameColumn) : item.label;
          },
        });
        list.onSelect = (item) => {
          const row = byId.get(item.value);
          if (row) settle({ kind: "load", row });
        };
        list.onCancel = () => settle({ kind: "close" });
        container.addChild(list);
      } else {
        container.addChild(new Lines((width) => emptyLines(data, width, theme)));
      }

      container.addChild(new Lines(() => [""]));
      container.addChild(new Lines((width) => hintLines(data.rows.length > 0, width, theme)));

      // The title carries the one fact the list cannot: how much of the
      // machine is in use, which is what the panel is opened to find out.
      const framed = new Framed(() => panelTitle(data), container, theme);

      // The bars are live: redraw with the footer's tick rather than running a
      // second timer and a second poll of our own. `ctx` is passed so the tick
      // starts even when the status line is switched off — the poll follows the
      // views, not the line.
      const unsubscribe = onPaint(() => tui.requestRender(), ctx);

      // An overlay hidden by `visible` keeps input ownership nowhere and can no
      // longer be escaped, so a terminal that shrinks past the floor closes it
      // instead of stranding it.
      const onResize = (): void => {
        if (terminalColumns() < MIN_OVERLAY_COLUMNS) settle({ kind: "close" });
      };
      process.stdout.on("resize", onResize);

      return {
        render: (width: number) => framed.render(width),
        invalidate: () => framed.invalidate(),
        handleInput: (input: string) => {
          if (input === "u") {
            settle({ kind: "unload" });
            return;
          }
          if (input === "r") {
            settle({ kind: "refresh" });
            return;
          }
          if (input === "a" || input === "s") {
            // Both screens need a model, so neither is a key on an empty list.
            const selected = list?.getSelectedItem();
            const row = selected ? byId.get(selected.value) : undefined;
            if (row) settle({ kind: input === "a" ? "size" : "sampling", row });
            return;
          }
          if (!list) {
            if (matchesKey(input, Key.escape)) settle({ kind: "close" });
            return;
          }
          list.handleInput(input);
          tui.requestRender();
        },
        dispose: () => {
          unsubscribe();
          process.stdout.off("resize", onResize);
        },
      };
    },
    { overlay: true, overlayOptions: overlayGeometry },
  );
}

/**
 * Open the panel and keep it open until the user closes it.
 *
 * A loop rather than recursion: `⏎` and `u` both end with the panel back on
 * screen showing what just changed, and a session of loading models should not
 * build a call stack. Every pass re-fetches, so the bars and the `●` are the
 * state after the action rather than before it.
 */
export async function openPanel(ctx: ExtensionContext): Promise<void> {
  // Nowhere to draw — another mode, or a terminal below the floor. The command
  // still answers, in one line.
  if (ctx.mode !== "tui" || terminalColumns() < MIN_OVERLAY_COLUMNS) {
    report(ctx, panelSummary(await fetchPanelData(ctx)), "info");
    return;
  }

  for (;;) {
    const data = await fetchPanelData(ctx);
    const action = await showPanel(ctx, data);

    if (action.kind === "close") return;
    if (action.kind === "refresh") {
      await refreshCatalogue(ctx);
      continue;
    }
    if (action.kind === "unload") {
      await unloadAndReport(ctx);
      continue;
    }
    if (action.kind === "sampling") {
      // Replaces the panel in place, like the sizer, and the loop brings the
      // panel back with whatever was saved already published.
      await openSampling(ctx, action.row.id, action.row.name);
      if (terminalColumns() < MIN_OVERLAY_COLUMNS) return;
      continue;
    }
    if (action.kind === "size") {
      // The sizer replaces the panel in place, and the loop brings the panel
      // back afterwards with whatever the override write changed already in it.
      await openSizer(ctx, action.row.id);
      if (terminalColumns() < MIN_OVERLAY_COLUMNS) return;
      continue;
    }
    await loadFromPanel(ctx, action.row);
    if (terminalColumns() < MIN_OVERLAY_COLUMNS) return;
  }
}

/** Why a row cannot be loaded, in the words the notification will use. */
function refusal(row: PanelRow): string | undefined {
  switch (row.fit) {
    case "missing":
      return `${row.name} is not on disk — Unsloth cannot load it`;
    case "incomplete":
      return `${row.name} is only partly downloaded`;
    case "wont-fit":
      return `${row.name} is tuned for GPUs this machine does not have (${row.placement ?? "unknown"})`;
    default:
      return undefined;
  }
}

/**
 * `⏎` — load the selected model with its tuned settings.
 *
 * The in-flight session ensure is cancelled first: two explicit loads racing
 * for the same GPUs is the one thing `force_cancel_active` cannot sort out, and
 * the user pressing `⏎` is a clearer statement of intent than a background job.
 */
async function loadFromPanel(ctx: ExtensionContext, row: PanelRow): Promise<void> {
  const refused = refusal(row);
  if (refused) {
    report(ctx, `⚠ ${refused}`, "warning");
    return;
  }

  cancelEnsure();
  const client = createSessionClient(ctx);
  const work = runTunedLoad(ctx, client, row.name, row);
  const watched = await showLoadProgress(ctx, row.name, row.override, work);

  const announce = (outcome: { ok: boolean; error: string | undefined }): void => {
    if (!outcome.ok) {
      // A failure the overlay already showed does not need saying twice.
      if (watched === "backgrounded" && outcome.error) report(ctx, `✗ Load failed: ${outcome.error}`, "error");
      return;
    }
    const untuned = row.override ? "" : " — with Unsloth's defaults, not tuned settings";
    const elsewhere = ctx.model && ctx.model.id !== row.id ? " · ctrl+p to point Pi at it" : "";
    report(ctx, `✓ ${row.name} loaded${untuned}${elsewhere}`, "info");
  };

  if (watched === "backgrounded") {
    void work.then(announce, () => undefined);
    return;
  }
  announce(await work);
}
