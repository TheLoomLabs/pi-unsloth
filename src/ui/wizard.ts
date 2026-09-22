/**
 * The setup wizard — `/unsloth setup`, and once on a machine with no profile.
 *
 * It exists for one reason: the extension has to decide which GPU drives the
 * monitor, and that decision is a measurement, not a certainty. So the wizard
 * shows **the conclusion and the evidence behind it**, lets `d` overturn it,
 * and writes what the user accepted. A wrong guess then costs a keystroke
 * rather than a frozen desktop.
 *
 * Two rules, both load-bearing:
 *
 *   - **Nothing is written until `⏎`.** Escape leaves no file behind, so
 *     running the wizard to look at it is free.
 *   - **Nothing here measures anything.** Every figure comes from
 *     `src/hardware/detect.ts`, which asks the server; the wizard only draws
 *     what detection found and records what the user did about it.
 *
 * The drawing and the keys are pure functions over one state object, so every
 * line — including the ones this machine cannot produce, like a headless box or
 * a corrected flag — is a unit test.
 */

import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Container, Key, matchesKey, truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";

import { checkHealth } from "../api/client.ts";
import { endpointLabel, isLocalEndpoint, normaliseEndpoint } from "../endpoint.ts";
import { budgetGb, defaultHeadroomGb, type GpuFacts } from "../hardware/budget.ts";
import { detectTopology, type DisplaySource, type Topology, type TopologyGpu } from "../hardware/detect.ts";
import {
  PROFILE_VERSION,
  profileForEndpoint,
  readMachineProfile,
  setupHasRun,
  writeMachineProfile,
  type MachineProfile,
  type ProfileGpu,
} from "../hardware/profile.ts";
import type { SizingSupport } from "../hardware/version.ts";
import { createSessionClient } from "../provider.ts";
import { report } from "../supervisor.ts";
import {
  FRAME_ROWS,
  Framed,
  Lines,
  MAX_OVERLAY_HEIGHT_SHARE,
  MIN_OVERLAY_COLUMNS,
  overlayGeometry,
  padTo,
  terminalColumns,
  terminalRows,
} from "./draw.ts";
import { logoLines } from "./logo.ts";

/** One open of the wizard is three local requests; none may hang the overlay. */
const WIZARD_TIMEOUT_MS = 5_000;

/** Left margin, matching the panel so the two overlays line up. */
const MARGIN = 2;
/** Indent for the lines belonging to a GPU, under its heading. */
const DETAIL = 6;

/** How much `←` / `→` move the headroom, in GiB. */
export const HEADROOM_STEP_GB = 0.5;

/** The subset of Pi's theme this file uses (no raw ANSI). */
export type WizardToken = "accent" | "success" | "warning" | "error" | "muted" | "dim" | "text";

export interface WizardTheme {
  fg(color: WizardToken, text: string): string;
  bold(text: string): string;
}

export interface WizardGpu {
  index: number;
  name: string | undefined;
  totalGb: number;
  /** Live VRAM, whatever is resident right now. */
  usedGb: number;
  /** VRAM with nothing loaded. `undefined` when it has never been observed. */
  idleUsedGb: number | undefined;
  display: boolean;
  headroomGb: number;
  evidence: string[];
  /** Outputs sysfs sees plugged in — the cross-check, never the verdict. */
  connected: string[];
  /** True once the user has moved this GPU's headroom or flag themselves. */
  touched: boolean;
}

export interface WizardServer {
  baseUrl: string;
  up: boolean;
  detail: string | undefined;
  version: string | undefined;
  backend: string | undefined;
  /** The endpoint is not this machine (src/endpoint.ts). */
  remote: boolean;
}

export interface WizardState {
  server: WizardServer;
  gpus: WizardGpu[];
  ignoredCount: number;
  displaySource: DisplaySource;
  sizing: SizingSupport;
  /** Why there is no topology to show, when there is none. */
  degradedReason: string | undefined;
  /** Why the DRM cross-check could not run here, when it could not. */
  connectorNote: string | undefined;
  selected: number;
  /** True once anything has been changed from what detection concluded. */
  changed: boolean;
  /**
   * The address being typed, or `undefined` when the field is closed.
   *
   * While it is open it owns every key: a `d` typed into a hostname must not
   * toggle a display flag on the GPU behind the field.
   */
  editing: string | undefined;
  /** Why the last address was refused, when it was. */
  editError: string | undefined;
  /** An address is being tested right now. */
  probing: boolean;
}

export function wizardState(topology: Topology, server: WizardServer): WizardState {
  return {
    server,
    gpus: topology.gpus.map((gpu: TopologyGpu) => ({
      index: gpu.index,
      name: gpu.name,
      totalGb: gpu.totalGb,
      usedGb: gpu.usedGb,
      idleUsedGb: gpu.idleUsedGb,
      display: gpu.display,
      headroomGb: gpu.headroomGb,
      evidence: [...gpu.evidence],
      connected: [...gpu.connected],
      touched: false,
    })),
    ignoredCount: topology.ignoredCount,
    displaySource: topology.displaySource,
    sizing: topology.sizing,
    degradedReason: topology.degraded ? (topology.reason ?? "no GPU topology available") : undefined,
    connectorNote: topology.connectorNote,
    selected: 0,
    changed: false,
    editing: undefined,
    editError: undefined,
    probing: false,
  };
}

function facts(gpu: WizardGpu): GpuFacts {
  return {
    index: gpu.index,
    totalGb: gpu.totalGb,
    idleUsedGb: gpu.idleUsedGb ?? 0,
    display: gpu.display,
    headroomGb: gpu.headroomGb,
  };
}

/** What a model may use here, after the desktop's own VRAM and the headroom. */
export function usable(gpu: WizardGpu): number {
  return budgetGb(facts(gpu));
}

export type WizardOutcome = "save" | "cancel" | "open" | "endpoint";

/** The longest address worth typing; a stray paste cannot grow the field. */
const MAX_ENDPOINT_LENGTH = 120;

/**
 * One key, while the address field is open.
 *
 * Split out because the field owns input completely: every key that is not
 * text, backspace, `⏎` or `esc` is *swallowed* rather than falling through to
 * the GPU list behind it.
 */
function editKey(next: WizardState, input: string): { state: WizardState; outcome: WizardOutcome } {
  const buffer = next.editing ?? "";
  if (matchesKey(input, Key.escape)) {
    // Escape abandons the edit, not the wizard: the address on screen is the
    // one that was there before, and nothing has been probed or written.
    next.editing = undefined;
    next.editError = undefined;
    return { state: next, outcome: "open" };
  }
  if (matchesKey(input, Key.enter)) return { state: next, outcome: "endpoint" };
  if (matchesKey(input, Key.backspace)) {
    next.editing = buffer.slice(0, -1);
    next.editError = undefined;
    return { state: next, outcome: "open" };
  }
  // Single printable characters only. A cursor key arrives as an escape
  // sequence, and appending it would put `\u001b[D` into a hostname.
  if (input.length === 1 && input >= " " && input !== "\u007f" && buffer.length < MAX_ENDPOINT_LENGTH) {
    next.editing = buffer + input;
    next.editError = undefined;
  }
  return { state: next, outcome: "open" };
}

/**
 * Apply one keypress. Returns the state to draw next and what to do about it.
 *
 * Mutating a copy rather than the caller's object keeps the reducer testable
 * one key at a time, which is the only way the `d`-then-`←` interaction — where
 * a toggle moves a headroom the user has not touched, and then stops doing so —
 * can be pinned down.
 */
export function applyKey(state: WizardState, input: string): { state: WizardState; outcome: WizardOutcome } {
  const next: WizardState = { ...state, gpus: state.gpus.map((gpu) => ({ ...gpu })) };
  const gpu = next.gpus[next.selected];

  if (next.editing !== undefined) return editKey(next, input);

  if (matchesKey(input, Key.escape)) return { state: next, outcome: "cancel" };
  if (matchesKey(input, Key.enter)) {
    // Nothing to save on a machine with no topology: the profile would record
    // an empty GPU list as though it were a finding.
    return { state: next, outcome: next.gpus.length > 0 ? "save" : "cancel" };
  }

  if (input === "e" || input === "E") {
    // Opens on the current address, so correcting a port is two keystrokes
    // rather than retyping a hostname.
    next.editing = endpointLabel(next.server.baseUrl);
    next.editError = undefined;
    return { state: next, outcome: "open" };
  }

  if (matchesKey(input, Key.up) && next.selected > 0) next.selected -= 1;
  else if (matchesKey(input, Key.down) && next.selected < next.gpus.length - 1) next.selected += 1;
  else if (gpu && (input === "d" || input === "D")) {
    gpu.display = !gpu.display;
    // The flag and the headroom default move together — until the user has set
    // a headroom of their own, which a toggle must never overwrite.
    if (!gpu.touched) gpu.headroomGb = defaultHeadroomGb(gpu.display);
    // The record of *why*, as the user saw it: their own decision first, and
    // any connector evidence that was on screen beside it. `displayEvidence`
    // exists so that a wrong flag is debuggable a month later.
    gpu.evidence = ["user", ...gpu.connected.map((output) => `drm-connector:${output}`)];
    next.changed = true;
  } else if (gpu && matchesKey(input, Key.left)) {
    gpu.headroomGb = clampHeadroom(gpu.headroomGb - HEADROOM_STEP_GB, gpu.totalGb);
    gpu.touched = true;
    next.changed = true;
  } else if (gpu && matchesKey(input, Key.right)) {
    gpu.headroomGb = clampHeadroom(gpu.headroomGb + HEADROOM_STEP_GB, gpu.totalGb);
    gpu.touched = true;
    next.changed = true;
  }

  return { state: next, outcome: "open" };
}

/** Headroom stays inside the card. Rounded, so 0.1 + 0.2 never reaches the UI. */
function clampHeadroom(value: number, totalGb: number): number {
  const ceiling = totalGb > 0 ? totalGb : value;
  return Math.round(Math.max(0, Math.min(ceiling, value)) * 10) / 10;
}

/**
 * The profile the wizard would write.
 *
 * Separate from writing it so the shape is a test rather than a file someone
 * has to look at, and so `⏎` is the only thing that ever touches the disk.
 * Everything the wizard did not collect — `policy`, `calibration`, and anything
 * hand-added — is carried over from what is already there.
 */
export function profileFrom(state: WizardState, existing: MachineProfile | undefined, now: Date = new Date()): MachineProfile {
  const gpus: ProfileGpu[] = state.gpus.map((gpu) => {
    const saved = existing?.gpus.find((entry) => entry.index === gpu.index);
    const entry: ProfileGpu = {
      ...saved,
      index: gpu.index,
      display: gpu.display,
      headroomGiB: gpu.headroomGb,
      displayEvidence: gpu.display ? [...gpu.evidence] : [],
    };
    if (gpu.name !== undefined) entry.name = gpu.name;
    if (gpu.totalGb > 0) entry.totalGiB = gpu.totalGb;
    // An idle figure already on file survives a session that could not take one
    // — it was measured with nothing loaded, which is the only time it means
    // anything, and a session spent with a model resident is not new evidence.
    if (gpu.idleUsedGb !== undefined) entry.idleUsedGiB = gpu.idleUsedGb;
    return entry;
  });

  const profile: MachineProfile = {
    ...existing,
    version: PROFILE_VERSION,
    baseUrl: state.server.baseUrl,
    detectedAt: now.toISOString(),
    gpus,
  };
  if (state.server.version !== undefined) profile.unslothVersion = state.server.version;
  if (state.server.backend !== undefined) profile.backend = state.server.backend;
  return profile;
}

/* --------------------------------------------------------------------------
 * Drawing
 * ------------------------------------------------------------------------ */

const GAP = 2;

/**
 * Wrap on spaces to `width`.
 *
 * The overlay is 54 columns at its narrowest, and the two longest lines here —
 * the sizing gate and the reason a cross-check was skipped — both carry the
 * fact that makes them useful at the *end*: a version, a count. Clipping them
 * would leave a warning that says only that something is wrong, so they wrap.
 */
export function wrapText(text: string, width: number): string[] {
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

/** A wrapped, themed, indented block. Continuation lines line up with the first. */
function block(text: string, token: WizardToken, indent: number, width: number, theme: WizardTheme): string[] {
  return wrapText(text, width - indent).map((line) => " ".repeat(indent) + theme.fg(token, line));
}

function gb(value: number, places = 1): string {
  return `${value.toFixed(places)} GiB`;
}

/** The host, without a scheme nobody needs to read. */
function host(baseUrl: string): string {
  return baseUrl.replace(/^https?:\/\//, "");
}

export function serverLine(state: WizardState, theme: WizardTheme): string {
  const label = theme.fg("muted", padTo("Server", 8, 6));

  // The field replaces the status while it is open: what the server *was*
  // saying is about an address that is being replaced.
  if (state.editing !== undefined) {
    const field = theme.fg("accent", `[ ${state.editing}_ ]`);
    return " ".repeat(MARGIN) + [label, field].join(" ".repeat(GAP));
  }

  const parts: string[] = [label, theme.fg("text", host(state.server.baseUrl))];
  // Before the status, not after it: the line is clipped to the frame, and a
  // health detail like "/api/health: getaddrinfo ENOTFOUND gpubox.invalid" is
  // long enough to push the badge off the end — in the one case where it
  // explains the most. Observed in the pty run of 2026-09-22.
  if (state.server.remote) parts.push(theme.fg("warning", "⌁ remote"));
  if (state.probing) {
    parts.push(theme.fg("dim", "… testing"));
    return " ".repeat(MARGIN) + parts.join(" ".repeat(GAP));
  }
  if (state.server.up) {
    parts.push(theme.fg("success", "✓ up"));
    if (state.server.version) parts.push(theme.fg("muted", `v${state.server.version}`));
    if (state.server.backend) parts.push(theme.fg("muted", state.server.backend));
  } else {
    parts.push(theme.fg("error", "✗ not answering"));
    if (state.server.detail) parts.push(theme.fg("dim", state.server.detail));
  }
  return " ".repeat(MARGIN) + parts.join(" ".repeat(GAP));
}

/**
 * The lines under the server line: what the field is for, why an address did
 * not answer, and what a remote endpoint turns off.
 *
 * All three are about the *endpoint*, so they live together — and the one that
 * matters most is the refusal, because the fix for it is on another machine and
 * nobody guesses `-H 0.0.0.0` unaided.
 */
export function endpointLines(state: WizardState, width: number, theme: WizardTheme): string[] {
  if (state.editing !== undefined) {
    const lines = [
      ...block(`⏎ test and use   esc keep ${host(state.server.baseUrl)}`, "dim", DETAIL, width, theme),
    ];
    if (state.editError) lines.push(...block(`✗ ${state.editError}`, "error", DETAIL, width, theme));
    return lines;
  }
  if (state.probing) return [];

  const lines: string[] = [];
  if (!state.server.up && state.server.remote) {
    lines.push(
      ...block(
        "The server is not listening there, or is bound to loopback on its own machine (-H 0.0.0.0).",
        "dim",
        DETAIL,
        width,
        theme,
      ),
    );
    // No key hint here: a server that is not answering has no topology either,
    // so the degraded footer below already offers `e server   esc close`. Two
    // hints on one screen was what the pty run of 2026-09-22 actually drew.
    return lines;
  }
  if (state.server.remote) {
    lines.push(
      "",
      ...block(
        "⌁ Another machine's GPUs. Autostart, freeing VRAM and the display cross-check are this machine's only — all off.",
        "warning",
        MARGIN,
        width,
        theme,
      ),
    );
  }
  return lines;
}

/** `Detected 2 compute GPUs  (1 integrated, ignored)`. */
export function countLine(state: WizardState, theme: WizardTheme): string {
  const count = state.gpus.length;
  const head = `Detected ${count} compute GPU${count === 1 ? "" : "s"}`;
  const ignored =
    state.ignoredCount > 0 ? `  (${state.ignoredCount} integrated, ignored)` : "";
  return " ".repeat(MARGIN) + theme.fg("text", head) + theme.fg("dim", ignored);
}

/** What this GPU is holding, and whether that figure means anything yet. */
export function usageLine(gpu: WizardGpu): string {
  if (gpu.idleUsedGb !== undefined) return `${gb(gpu.idleUsedGb, 2)} in use at idle`;
  return `${gb(gpu.usedGb, 2)} in use now — nothing has been measured at idle`;
}

export interface Conclusion {
  text: string;
  token: WizardToken;
}

/**
 * The `▸` line: the verdict **and** what it rests on.
 *
 * Five cases, and the two that matter most are the ones where the evidence does
 * not agree with itself: a card flagged from idle VRAM with no connected
 * output, and a card with a monitor plugged into it that idle VRAM did not
 * flag. Both are drawn as warnings naming the disagreement, because `d` is one
 * keystroke away and a silent wrong answer is not.
 */
export function conclusion(gpu: WizardGpu, source: DisplaySource): Conclusion {
  const outputs = gpu.connected.join(", ");
  if (gpu.display) {
    if (gpu.connected.length > 0) return { text: `display attached — ${outputs} connected`, token: "warning" };
    if (source === "profile" || gpu.evidence.includes("user")) {
      return { text: "display attached — set by you", token: "warning" };
    }
    return { text: `display attached — ${gb(gpu.idleUsedGb ?? gpu.usedGb, 2)} held at idle`, token: "warning" };
  }
  const measured = gpu.idleUsedGb !== undefined || source !== "none";
  if (gpu.connected.length > 0) {
    // Two different sentences, because they need two different fixes: a
    // measurement contradicted by a cable is a finding, while a cable with no
    // measurement beside it is simply the only evidence there is.
    return measured
      ? { text: `headless by VRAM — but ${outputs} is connected`, token: "error" }
      : { text: `not measured — but ${outputs} is connected; d to set`, token: "warning" };
  }
  if (!measured) {
    return { text: "not measured — unload to measure, or press d", token: "dim" };
  }
  return { text: "headless — preferred for models", token: "success" };
}

/** `headroom  ‹ 3.0 GiB ›   20.0 GiB usable`. */
export function headroomLine(gpu: WizardGpu, selected: boolean, theme: WizardTheme): string {
  const value = `‹ ${gb(gpu.headroomGb)} ›`;
  return [
    theme.fg("muted", "headroom"),
    theme.fg(selected ? "accent" : "text", value),
    theme.fg("dim", `${gb(usable(gpu))} usable`),
  ].join(" ".repeat(GAP));
}

/** The four lines of one GPU, themed and clipped to `width`. */
export function gpuLines(gpu: WizardGpu, selected: boolean, source: DisplaySource, width: number, theme: WizardTheme): string[] {
  const marker = selected ? theme.fg("accent", "→ ") : "  ";
  const name = gpu.name ?? "GPU";
  const label = theme.fg(selected ? "accent" : "text", theme.bold(`GPU ${gpu.index}`));
  const total = gpu.totalGb > 0 ? theme.fg("muted", gb(gpu.totalGb)) : "";
  const verdict = conclusion(gpu, source);

  return [
    `${marker}${label}${" ".repeat(GAP)}${theme.fg("muted", name)}${" ".repeat(GAP)}${total}`,
    ...block(usageLine(gpu), "dim", DETAIL, width, theme),
    ...block(`▸ ${verdict.text}`, verdict.token, DETAIL, width, theme),
    `${" ".repeat(DETAIL)}${headroomLine(gpu, selected, theme)}`,
  ].map((line) => truncateToWidth(line, width, ""));
}

const KEYS_FULL = "↑↓ move  ←→ adjust  d toggle  e server  ⏎ save  esc cancel";
const KEYS_SHORT = "↑↓  ←→  d  e  ⏎ save  esc";
/** While the field is open there are only three keys, and two of them differ. */
const KEYS_EDITING = "type an address  ⏎ test and use  esc cancel";

/**
 * Everything the overlay draws, in order.
 *
 * `rows` is how tall the terminal is, and it is a parameter rather than a
 * lookup so that every height this screen can be drawn at is a unit test. It
 * buys one thing: the mascot (`logo.ts`) is laid on top only once the rest of
 * the screen has been measured and found to leave room for it. The wizard is
 * the tallest overlay in the extension — a server line, a GPU block each, two
 * conditional warnings and the keys — and on a short terminal those are all
 * load-bearing, so the decoration is what gives way.
 */
export function wizardLines(
  state: WizardState,
  width: number,
  theme: WizardTheme,
  rows: number = terminalRows(),
): string[] {
  const body = wizardBody(state, width, theme);
  // What the overlay may occupy (`overlayGeometry`), less its own two edges.
  const available = Math.floor(rows * MAX_OVERLAY_HEIGHT_SHARE) - FRAME_ROWS;
  const logo = logoLines(width, available - body.length - 1, theme);
  const lines = logo.length === 0 ? body : [...logo, "", ...body];
  return lines.map((line) => truncateToWidth(line, width, ""));
}

function wizardBody(state: WizardState, width: number, theme: WizardTheme): string[] {
  const pad = " ".repeat(MARGIN);
  const lines: string[] = [serverLine(state, theme), ...endpointLines(state, width, theme), ""];

  if (state.editing !== undefined) {
    // Nothing below the field while it is open: the GPUs on screen belong to
    // the address being replaced, and drawing them under a half-typed hostname
    // invites `⏎` to be read as "save those".
    lines.push(`${pad}${theme.fg("dim", KEYS_EDITING)}`);
    return lines;
  }

  if (state.degradedReason !== undefined) {
    lines.push(
      ...block(`⚠ No GPU topology — ${state.degradedReason}`, "warning", MARGIN, width, theme),
      ...block("Models still load and unload; sizing is disabled.", "dim", MARGIN, width, theme),
      "",
      // `e` belongs here above all: a wrong address is exactly why there is no
      // topology, and it is the only key that can fix it from this screen.
      `${pad}${theme.fg("dim", "e server   esc close")}`,
    );
    return lines;
  }

  lines.push(countLine(state, theme), "");
  state.gpus.forEach((gpu, position) => {
    lines.push(...gpuLines(gpu, position === state.selected, state.displaySource, width, theme), "");
  });

  if (!state.sizing.enabled && state.sizing.reason) {
    lines.push(...block(`⚠ ${state.sizing.reason}`, "warning", MARGIN, width, theme), "");
  }
  if (state.connectorNote) {
    lines.push(...block(`No display cross-check — ${state.connectorNote}`, "dim", MARGIN, width, theme), "");
  }

  lines.push(...block("Wrong?  d  toggles the display flag on the selected GPU", "dim", MARGIN, width, theme), "");
  const keys = visibleWidth(KEYS_FULL) + 2 * MARGIN <= width ? KEYS_FULL : KEYS_SHORT;
  lines.push(`${pad}${theme.fg("dim", keys)}`);
  return lines;
}

/** One line for a terminal too narrow to draw in, and for `pi -p`. */
export function wizardSummary(state: WizardState): string {
  const where = state.server.remote ? ` at ${host(state.server.baseUrl)}` : "";
  if (state.degradedReason !== undefined) return `◌ unsloth${where} — ${state.degradedReason}`;
  const display = state.gpus.filter((gpu) => gpu.display).map((gpu) => `GPU ${gpu.index}`);
  const parts = [
    `${state.gpus.length} compute GPU${state.gpus.length === 1 ? "" : "s"}${where}`,
    display.length > 0 ? `display: ${display.join(", ")}` : "no display GPU detected",
  ];
  if (!state.sizing.enabled && state.sizing.reason) parts.push(state.sizing.reason);
  return `⬢ unsloth — ${parts.join(" · ")}`;
}

/* --------------------------------------------------------------------------
 * Running it
 * ------------------------------------------------------------------------ */

interface WizardResult {
  state: WizardState;
  outcome: "save" | "cancel";
}

/** Draw the wizard and resolve once the user has accepted or abandoned it. */
function showWizard(ctx: ExtensionContext, initial: WizardState): Promise<WizardResult> {
  return ctx.ui.custom<WizardResult>(
    (tui, theme, _keybindings, done) => {
      let state = initial;
      let finished = false;
      const settle = (outcome: "save" | "cancel"): void => {
        if (finished) return;
        finished = true;
        done({ state, outcome });
      };

      const container = new Container();
      container.addChild(new Lines((width) => wizardLines(state, width, theme)));
      const framed = new Framed("Unsloth setup", container, theme);

      /**
       * `⏎` on the address field: read it, test it, and redraw on what came
       * back. Nothing is written here — the profile is still only touched by
       * `⏎` on the wizard proper.
       *
       * Detection is re-run rather than patched, because every figure on the
       * screen belongs to the old machine: its GPUs, its idle VRAM, its
       * version, and any `d` or `←→` the user had applied to them.
       */
      const adopt = async (): Promise<void> => {
        const typed = state.editing ?? "";
        const baseUrl = normaliseEndpoint(typed);
        if (baseUrl === undefined) {
          state = { ...state, editError: `${typed.trim() === "" ? "an address" : typed} is not an address` };
          tui.requestRender();
          return;
        }

        state = {
          ...state,
          editing: undefined,
          editError: undefined,
          probing: true,
          server: { ...state.server, baseUrl, remote: !isLocalEndpoint(baseUrl) },
        };
        tui.requestRender();

        const fresh = await collect(ctx, baseUrl);
        if (finished) return;
        state = fresh;
        tui.requestRender();
      };

      // A hidden overlay owns input nowhere and can no longer be escaped, so a
      // terminal that shrinks past the floor closes it.
      const onResize = (): void => {
        if (terminalColumns() < MIN_OVERLAY_COLUMNS) settle("cancel");
      };
      process.stdout.on("resize", onResize);

      return {
        render: (width: number) => framed.render(width),
        invalidate: () => framed.invalidate(),
        handleInput: (input: string) => {
          const next = applyKey(state, input);
          state = next.state;
          if (next.outcome === "save") settle("save");
          else if (next.outcome === "cancel") settle("cancel");
          else if (next.outcome === "endpoint") void adopt();
          else tui.requestRender();
        },
        dispose: () => {
          process.stdout.off("resize", onResize);
        },
      };
    },
    { overlay: true, overlayOptions: overlayGeometry },
  );
}

/**
 * Ask the machine what it is, and the server how it is.
 *
 * `baseUrl` overrides the configured endpoint, which is what `⏎` on the address
 * field uses: an address is *tested* before it is adopted, so a typo costs a
 * second rather than a session.
 */
async function collect(ctx: ExtensionContext, baseUrl?: string): Promise<WizardState> {
  const client = createSessionClient(ctx, baseUrl);
  const call = { timeoutMs: WIZARD_TIMEOUT_MS };
  const [health, topology] = await Promise.all([checkHealth(client), detectTopology(client, call)]);
  return wizardState(topology, {
    baseUrl: client.baseUrl,
    up: health.state === "up",
    detail: health.detail,
    version: topology.unslothVersion,
    backend: topology.backend,
    remote: !isLocalEndpoint(client.baseUrl),
  });
}

/** `⏎` — the only thing in this file that touches the disk. */
function save(ctx: ExtensionContext, state: WizardState): void {
  // Merged against the profile *as it applies to this endpoint*: entries
  // recorded for another server would otherwise carry that machine's name and
  // card size into this one's.
  const existing = profileForEndpoint(readMachineProfile().profile, state.server.baseUrl);
  const result = writeMachineProfile(profileFrom(state, existing));
  if (!result.ok) {
    report(ctx, `✗ Could not write ${result.path}: ${result.error ?? "unknown error"}`, "error");
    return;
  }
  const display = state.gpus.filter((gpu) => gpu.display).map((gpu) => `GPU ${gpu.index}`);
  const where = display.length > 0 ? `display: ${display.join(", ")}` : "no display GPU";
  report(ctx, `✓ Saved ${result.path} — ${state.gpus.length} GPUs, ${where}`, "info");
}

/**
 * Run the wizard.
 *
 * Outside the TUI, and in a terminal too narrow to draw a modal in, it reports
 * what detection found in one line and writes nothing: "nothing is written
 * until the user accepts" has to hold where there is no way to accept.
 */
export async function runWizard(ctx: ExtensionContext): Promise<void> {
  const state = await collect(ctx);

  if (ctx.mode !== "tui" || terminalColumns() < MIN_OVERLAY_COLUMNS) {
    report(ctx, wizardSummary(state), state.degradedReason !== undefined ? "warning" : "info");
    return;
  }

  const result = await showWizard(ctx, state);
  if (result.outcome === "save") save(ctx, result.state);
}

/**
 * The first run on a machine with no profile.
 *
 * Deliberately silent in every case where there is nothing to confirm: another
 * mode, a terminal too narrow for a modal, a profile that already exists, a
 * server that is not up, or a machine with no GPUs. A modal that appears on
 * every session until the server comes up would be nagging rather than setting
 * up, and the wizard is always one `/unsloth setup` away.
 */
export async function offerSetup(ctx: ExtensionContext): Promise<void> {
  if (ctx.mode !== "tui" || terminalColumns() < MIN_OVERLAY_COLUMNS) return;
  // Not "is there a file": the footer toggle writes one policy key, and a
  // profile holding only a preference is not a machine that has been set up
  // (src/hardware/profile.ts → `setupHasRun`).
  if (setupHasRun()) return;

  const state = await collect(ctx);
  if (!state.server.up || state.gpus.length === 0) return;

  const result = await showWizard(ctx, state);
  if (result.outcome === "save") save(ctx, result.state);
}
