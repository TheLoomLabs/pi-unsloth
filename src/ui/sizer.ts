/**
 * The sizer — `a` in the panel, and `/unsloth add <model>`.
 *
 * The portable replacement for `register-model.py`: every figure on this screen
 * comes from `POST /api/inference/estimate-memory`, and every decision comes
 * from the per-GPU fit rule in src/hardware/budget.ts. Nothing is computed from
 * a table of constants, because there is no table any more.
 *
 * Three things this screen is careful about:
 *
 *   - **It shows the breakdown, not just the verdict.** `weights / kv / compute
 *     / drafter / projector` are what makes a context fit or not, and the two
 *     the old script ignored — the drafter's runtime and the vision projector —
 *     are the two that made it over-promise.
 *   - **Every adjustment re-prices.** The estimate allocates nothing, so there
 *     is no reason to show a number that belongs to a setting the user has
 *     already changed. While a price is in flight the field says so rather than
 *     leaving the old one on screen looking current.
 *   - **`⏎` writes one model's override and nothing else.** `v` writes nothing
 *     at all until it has measured something.
 *
 * As in the panel and the wizard, the drawing and the keys are pure functions
 * over one state object, so every line — including the ones this machine cannot
 * produce, like a model that fits nowhere — is a unit test.
 */

import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Container, Key, matchesKey, truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";

import type { UnslothClient } from "../api/client.ts";
import { estimateCaveat, describeUnavailable, estimateMemory, type MemoryEstimate } from "../api/estimate.ts";
import { getOverrides, mergeOverride, overrideKey, putOverride, type ModelOverride } from "../api/lifecycle.ts";
import { listModels, type CatalogueEntry } from "../api/models.ts";
import { budgetGb, type FitVerdict, type GpuFacts } from "../hardware/budget.ts";
import { detectTopology, toFacts } from "../hardware/detect.ts";
import { readMachineProfile } from "../hardware/profile.ts";
import { createSessionClient, formatContext } from "../provider.ts";
import {
  DEFAULT_CTX_STEP,
  buildEstimateRequest,
  buildOverrideEntry,
  cachedEstimator,
  configFromOverride,
  fitOf,
  maxContextFor,
  perGpuGb,
  placementLabel,
  sizeModel,
  snapContext,
  type Estimator,
  type PlacementOption,
  type SizingConfig,
  type SizingTarget,
} from "../sizing/search.ts";
import {
  readCalibration,
  runVerification,
  touchesDisplay,
  type Calibration,
  type CalibrationStamp,
} from "../sizing/verify.ts";
import { matchCatalogue, refreshCatalogue, report, resolveModel } from "../supervisor.ts";
import { Framed, Lines, MIN_OVERLAY_COLUMNS, overlayGeometry, padTo, terminalColumns } from "./draw.ts";

/** One open of the sizer asks the server several questions; none may hang it. */
const SIZER_TIMEOUT_MS = 20_000;

/** Left margin and inter-column gap, matching the panel so the overlays line up. */
const MARGIN = 2;
const GAP = 2;
/** Width of the field-name column: the longest label plus a space. */
const LABEL_WIDTH = 12;

/** The subset of Pi's theme this file uses (no raw ANSI). */
export type SizerToken = "accent" | "success" | "warning" | "error" | "muted" | "dim" | "text";

export interface SizerTheme {
  fg(color: SizerToken, text: string): string;
  bold(text: string): string;
}

/**
 * KV cache dtypes offered, largest first.
 *
 * These are llama.cpp's cache types, not a fact about anyone's hardware, and
 * the list is a *menu*, not a constraint: whatever the model's existing
 * override names is added to it, and the server prices whatever is chosen. The
 * first entry is the server's own default — omitting `cache_type_kv` entirely,
 * which is a real choice and the honest one when the user has no opinion.
 */
export const KV_DTYPES: readonly (string | undefined)[] = [undefined, "f16", "q8_0", "q5_1", "q4_0"];

/**
 * Speculative decoding modes offered.
 *
 * `mtp` is the one that matters on the reference box — it is what auto-switch
 * silently drops — but the estimator prices each mode differently, so the
 * choice belongs on this screen rather than in a constant.
 */
export const SPECULATIVE_MODES: readonly (string | undefined)[] = [undefined, "auto", "none", "mtp"];

/** Decode slots the `‹ slots ›` field will offer. The server's own ceiling is 64. */
const MIN_SLOTS = 1;
const MAX_SLOTS = 64;

export type SizerFieldKey = "context" | "kv" | "placement" | "speculative" | "slots";

/**
 * The adjustable fields, in the order they are drawn.
 *
 * `slots` is here — and not 's first draft of this screen — because
 * `n_parallel` multiplies the KV cache, and the sizer writes it into the
 * override. A setting that moves the biggest term in the breakdown cannot be
 * set invisibly.
 */
export const SIZER_FIELDS: readonly SizerFieldKey[] = ["context", "kv", "placement", "speculative", "slots"];

const FIELD_LABELS: Readonly<Record<SizerFieldKey, string>> = {
  context: "context",
  kv: "kv dtype",
  placement: "placement",
  speculative: "speculative",
  slots: "slots",
};

/** The model being sized, resolved to everything the endpoints need. */
export interface SizerModel {
  /** Pi's catalogue id, for the confirmation line. */
  id: string;
  name: string;
  quant: string | undefined;
  /** Absolute path or repo id — `model_path`, and the override's key stem. */
  modelPath: string;
  /** `native_context_length`, or whatever the GGUF header reported. */
  ceiling: number;
}

export interface SizerState {
  model: SizerModel;
  gpus: GpuFacts[];
  config: SizingConfig;
  step: number;
  /** Every placement this machine offers, in preference order. */
  options: PlacementOption[];
  kvChoices: readonly (string | undefined)[];
  specChoices: readonly (string | undefined)[];
  /** Index into `SIZER_FIELDS`. */
  field: number;
  estimate: MemoryEstimate | undefined;
  verdict: FitVerdict | undefined;
  /** Largest context that fits at this placement, or `undefined` if none does. */
  maxContext: number | undefined;
  /** An estimate is in flight: the figures on screen are one keystroke old. */
  busy: boolean;
  /** Why there is nothing to show — the estimator could not price this model. */
  problem: string | undefined;
  /** Why this estimate is a lower bound, when it is. */
  caveat: string | undefined;
  calibration: Calibration | undefined;
  /** The display-GPU confirmation is up. */
  confirming: boolean;
  /** What the last verification measured, in the words the screen shows. */
  note: string | undefined;
}

/* --------------------------------------------------------------------------
 * Keys
 * ------------------------------------------------------------------------ */

export type SizerOutcome = "open" | "apply" | "verify" | "cancel";

export interface SizerKeyResult {
  state: SizerState;
  outcome: SizerOutcome;
  /** The configuration changed, so the figures must be re-priced. */
  reprice: boolean;
  /** …and the change was one the `max safe` figure depends on. */
  research: boolean;
}

function cycle<T>(values: readonly T[], current: T, direction: 1 | -1): T {
  if (values.length === 0) return current;
  const at = values.findIndex((value) => value === current);
  const next = ((at === -1 ? 0 : at) + direction + values.length) % values.length;
  return values[next] as T;
}

/**
 * Apply one keypress.
 *
 * A copy is mutated rather than the caller's state, so the interesting
 * sequences — adjust, then confirm, then escape out of the confirmation — are
 * testable one key at a time.
 */
export function applySizerKey(state: SizerState, input: string): SizerKeyResult {
  const next: SizerState = { ...state, config: { ...state.config } };
  const still = (outcome: SizerOutcome = "open"): SizerKeyResult => ({ state: next, outcome, reprice: false, research: false });

  // The confirmation owns every key while it is up: it is the second
  // confirmation a load onto the display GPU requires, and a stray arrow key
  // must not dismiss it into a verification.
  if (state.confirming) {
    if (matchesKey(input, Key.escape)) {
      next.confirming = false;
      return still();
    }
    if (matchesKey(input, Key.enter) || input === "v" || input === "y") {
      next.confirming = false;
      return { state: next, outcome: "verify", reprice: false, research: false };
    }
    return still();
  }

  if (matchesKey(input, Key.escape)) return still("cancel");
  if (matchesKey(input, Key.enter)) {
    // Nothing to apply while nothing fits: writing an override that names a
    // context this machine cannot hold is the failure this screen prevents.
    return still(next.verdict?.fits ? "apply" : "open");
  }
  if (input === "v" || input === "V") {
    if (!next.verdict?.fits) return still();
    if (touchesDisplay(next.config, next.gpus)) {
      next.confirming = true;
      return still();
    }
    return { state: next, outcome: "verify", reprice: false, research: false };
  }

  if (matchesKey(input, Key.up)) {
    next.field = (next.field + SIZER_FIELDS.length - 1) % SIZER_FIELDS.length;
    return still();
  }
  if (matchesKey(input, Key.down)) {
    next.field = (next.field + 1) % SIZER_FIELDS.length;
    return still();
  }

  const direction: 1 | -1 | undefined = matchesKey(input, Key.right) ? 1 : matchesKey(input, Key.left) ? -1 : undefined;
  if (direction === undefined) return still();

  const field = SIZER_FIELDS[next.field];
  switch (field) {
    case "context": {
      const wanted = next.config.contextTokens + direction * next.step;
      next.config.contextTokens = Math.max(next.step, Math.min(snapContext(next.model.ceiling, next.step), snapContext(wanted, next.step)));
      return { state: next, outcome: "open", reprice: next.config.contextTokens !== state.config.contextTokens, research: false };
    }
    case "kv":
      next.config.kvDtype = cycle(next.kvChoices, next.config.kvDtype, direction);
      return { state: next, outcome: "open", reprice: true, research: true };
    case "placement": {
      if (next.options.length === 0) return still();
      const at = next.options.findIndex((option) => option.label === placementLabel(next.config.placement));
      const chosen = next.options[((at === -1 ? 0 : at) + direction + next.options.length) % next.options.length];
      if (!chosen) return still();
      next.config.placement = chosen.placement;
      return { state: next, outcome: "open", reprice: true, research: true };
    }
    case "speculative":
      next.config.speculativeType = cycle(next.specChoices, next.config.speculativeType, direction);
      return { state: next, outcome: "open", reprice: true, research: true };
    case "slots": {
      const current = next.config.nParallel ?? MIN_SLOTS;
      next.config.nParallel = Math.max(MIN_SLOTS, Math.min(MAX_SLOTS, current + direction));
      return { state: next, outcome: "open", reprice: next.config.nParallel !== current, research: true };
    }
    default:
      return still();
  }
}

/* --------------------------------------------------------------------------
 * Drawing
 * ------------------------------------------------------------------------ */

function gb(value: number, places = 1): string {
  return `${value.toFixed(places)} GiB`;
}

/** `‹ 65536 ›`, the shape every adjustable value on this screen wears. */
export function fieldValue(state: SizerState, field: SizerFieldKey): string {
  switch (field) {
    case "context":
      return String(state.config.contextTokens);
    case "kv":
      return state.config.kvDtype ?? "default";
    case "placement":
      return placementLabel(state.config.placement);
    case "speculative":
      return state.config.speculativeType ?? "default";
    case "slots":
      return String(state.config.nParallel ?? MIN_SLOTS);
  }
}

/** The note beside a field: what this choice costs, or what it allows. */
export function fieldHint(state: SizerState, field: SizerFieldKey): string | undefined {
  if (field === "context") {
    if (state.busy) return "max safe …";
    return state.maxContext !== undefined ? `max safe ${state.maxContext}` : "nothing fits here";
  }
  if (field === "slots" && (state.config.nParallel ?? MIN_SLOTS) > 1) return "×KV cache";
  return undefined;
}

/** The five `‹ … ›` lines. */
export function fieldLines(state: SizerState, width: number, theme: SizerTheme): string[] {
  return SIZER_FIELDS.map((field, position) => {
    const selected = position === state.field;
    const label = theme.fg("muted", padTo(FIELD_LABELS[field], LABEL_WIDTH, FIELD_LABELS[field].length));
    const value = `‹ ${fieldValue(state, field)} ›`;
    const hint = fieldHint(state, field);
    const line =
      " ".repeat(MARGIN) +
      label +
      theme.fg(selected ? "accent" : "text", value) +
      (hint ? " ".repeat(GAP) + theme.fg("dim", hint) : "");
    return truncateToWidth(line, width, "");
  });
}

/**
 * The breakdown.
 *
 * Two columns when there is room for two, one when there is not — and the KV
 * line carries the context it was priced at, because that is the number the
 * user is moving and the term it moves.
 */
export function breakdownLines(state: SizerState, width: number, theme: SizerTheme): string[] {
  const estimate = state.estimate;
  if (!estimate) return [" ".repeat(MARGIN) + theme.fg("muted", state.busy ? "pricing…" : "no estimate")];

  const GIB = 1024 ** 3;
  const items: Array<[string, number]> = [
    ["weights", estimate.weightsBytes / GIB],
    ["compute", estimate.computeBytes / GIB],
    [`kv @ ${estimate.nCtx ?? state.config.contextTokens}`, estimate.kvBytes / GIB],
    ["drafter", estimate.drafterRuntimeGpuBytes / GIB],
    ["projector", estimate.projectorRuntimeBytes / GIB],
  ];

  // Both columns are fixed width and the figures are right-aligned inside
  // theirs, so the decimal points line up down the block. Padding by the
  // *measured* width matters: `27.9 GiB` and `9.4 GiB` differ by a column, and
  // padding both as though they were the same length is what makes a table
  // look like a ransom note.
  const nameWidth = Math.max(...items.map(([name]) => visibleWidth(name)));
  const valueWidth = Math.max(...items.map(([, value]) => visibleWidth(gb(value))));
  const cell = ([name, value]: [string, number]): string => {
    const figure = gb(value);
    return (
      theme.fg("muted", padTo(name, nameWidth, visibleWidth(name))) +
      " ".repeat(GAP) +
      " ".repeat(Math.max(0, valueWidth - visibleWidth(figure))) +
      theme.fg("text", figure)
    );
  };
  const cellWidth = nameWidth + GAP + valueWidth;

  const twoUp = width - 2 * MARGIN >= 2 * cellWidth + 2 * GAP;
  const lines: string[] = [];
  if (twoUp) {
    for (let at = 0; at < items.length; at += 2) {
      const left = cell(items[at] as [string, number]);
      const right = items[at + 1] ? cell(items[at + 1] as [string, number]) : "";
      lines.push(" ".repeat(MARGIN) + padTo(left, cellWidth + 2 * GAP, cellWidth) + right);
    }
  } else {
    for (const item of items) lines.push(" ".repeat(MARGIN) + cell(item));
  }
  return lines.map((line) => truncateToWidth(line, width, ""));
}

/**
 * The total, where it goes, and whether it fits.
 *
 * One line per GPU of the chosen placement, each showing that card's expected
 * share against that card's own budget — because a tensor-parallel split fails
 * on one card, not on an average, and the whole point of the fit rule is that
 * it is per GPU.
 */
export function fitLines(state: SizerState, width: number, theme: SizerTheme): string[] {
  const estimate = state.estimate;
  if (!estimate) return [];
  const lines: string[] = [];
  const shares = perGpuGb(estimate, state.config.placement, state.calibration?.deltaGb ?? 0);
  const total = shares.reduce((sum, value) => sum + value, 0);

  // One figure column for the whole block, right-aligned, so the total and the
  // shares read down rather than across.
  const figures = [gb(total), ...shares.map((value) => gb(value))];
  const figureWidth = Math.max(...figures.map(visibleWidth));
  const right = (text: string): string => " ".repeat(Math.max(0, figureWidth - visibleWidth(text))) + text;

  lines.push(
    " ".repeat(MARGIN) +
      theme.fg("muted", padTo("total", LABEL_WIDTH, "total".length)) +
      right(theme.fg("text", theme.bold(gb(total)))),
  );

  state.config.placement.gpuIds.forEach((id, position) => {
    const gpu = state.gpus.find((candidate) => candidate.index === id);
    const budget = gpu ? budgetGb(gpu) : 0;
    const wanted = shares[position] ?? 0;
    const over = wanted > budget;
    const label = `GPU ${id}`;
    lines.push(
      " ".repeat(MARGIN) +
        theme.fg("muted", padTo(label, LABEL_WIDTH, label.length)) +
        " ".repeat(Math.max(0, figureWidth - visibleWidth(gb(wanted)))) +
        theme.fg(over ? "error" : "text", gb(wanted)) +
        theme.fg("dim", ` of ${gb(budget)} budget`),
    );
  });

  const verdict = state.verdict;
  if (verdict) {
    const text = verdict.fits
      ? "fits ✓"
      : `over by ${gb(verdict.shortfallGb)}${verdict.tightestIndex !== undefined ? ` on GPU ${verdict.tightestIndex}` : ""} ✗`;
    lines.push(" ".repeat(MARGIN) + theme.fg(verdict.fits ? "success" : "error", text));
  }

  if (state.calibration) {
    const delta = state.calibration.deltaGb;
    lines.push(
      " ".repeat(MARGIN) +
        theme.fg("dim", `includes ${delta >= 0 ? "+" : ""}${gb(delta, 2)} measured on this machine`),
    );
  }
  return lines.map((line) => truncateToWidth(line, width, ""));
}

/** Wrap on spaces — the same rule the wizard uses, for the same reason. */
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

function block(text: string, token: SizerToken, width: number, theme: SizerTheme): string[] {
  return wrap(text, width - MARGIN).map((line) => " ".repeat(MARGIN) + theme.fg(token, line));
}

const KEYS_FULL = "↑↓ field  ←→ adjust  ⏎ apply  v verify  esc";
const KEYS_SHORT = "↑↓ ←→  ⏎ apply  v verify  esc";

/**
 * The confirmation before a verification load touches the display GPU.
 *
 * Drawn in place rather than as a second overlay: a modal on top of a modal
 * cannot be escaped in a predictable order, and the thing being confirmed is
 * already on the screen behind it.
 */
export function confirmLines(state: SizerState, width: number, theme: SizerTheme): string[] {
  const display = state.config.placement.gpuIds.filter((id) =>
    state.gpus.some((gpu) => gpu.index === id && gpu.display),
  );
  const spare = Math.max(0, spareGb(state));
  return [
    ...block(
      `GPU ${display.join(", ")} drives your monitor. A failed load can freeze the desktop session.`,
      "text",
      width,
      theme,
    ),
    ...block(`The estimate says it fits with ${gb(spare)} to spare.`, "dim", width, theme),
    "",
    " ".repeat(MARGIN) + theme.fg("dim", "v  verify    esc  cancel"),
  ].map((line) => truncateToWidth(line, width, ""));
}

/** Room left on the tightest GPU of the placement, in GiB. */
function spareGb(state: SizerState): number {
  if (!state.estimate) return 0;
  const shares = perGpuGb(state.estimate, state.config.placement, state.calibration?.deltaGb ?? 0);
  const spare = state.config.placement.gpuIds.map((id, position) => {
    const gpu = state.gpus.find((candidate) => candidate.index === id);
    return (gpu ? budgetGb(gpu) : 0) - (shares[position] ?? 0);
  });
  return Math.min(...spare);
}

/** The frame's title: what is being sized, or what is being confirmed. */
export function sizerTitle(state: SizerState): string {
  if (state.confirming) return "Verify on the display GPU?";
  return `Size  ${[state.model.name, state.model.quant].filter(Boolean).join("  ")}`;
}

/** Everything the overlay draws, in order. */
export function sizerLines(state: SizerState, width: number, theme: SizerTheme): string[] {
  const pad = " ".repeat(MARGIN);
  const lines: string[] = [];

  if (state.confirming) return confirmLines(state, width, theme);

  if (state.problem) {
    lines.push(...block(`⚠ ${state.problem}`, "warning", width, theme), "", `${pad}${theme.fg("dim", "esc close")}`);
    return lines.map((line) => truncateToWidth(line, width, ""));
  }

  lines.push(...breakdownLines(state, width, theme), "");
  lines.push(...fitLines(state, width, theme), "");
  lines.push(...fieldLines(state, width, theme), "");

  if (state.caveat) lines.push(...block(`⚠ ${state.caveat}`, "warning", width, theme), "");
  if (state.note) lines.push(...block(state.note, "muted", width, theme), "");

  const keys = visibleWidth(KEYS_FULL) + 2 * MARGIN <= width ? KEYS_FULL : KEYS_SHORT;
  lines.push(`${pad}${theme.fg("dim", keys)}`);
  return lines.map((line) => truncateToWidth(line, width, ""));
}

/** One line for a terminal too narrow to draw in, and for `pi -p`. */
export function sizerSummary(state: SizerState): string {
  if (state.problem) return `◌ ${state.model.name} — ${state.problem}`;
  const verdict = state.verdict;
  const where = placementLabel(state.config.placement);
  if (!verdict) return `⬢ ${state.model.name} — no estimate`;
  return verdict.fits
    ? `⬢ ${state.model.name} — ${formatContext(state.config.contextTokens)} on ${where} · fits`
    : `⚠ ${state.model.name} — over by ${gb(verdict.shortfallGb)} on ${where}`;
}

/* --------------------------------------------------------------------------
 * Running it
 * ------------------------------------------------------------------------ */

/** Everything the sizer needs, gathered before the overlay opens. */
interface Collected {
  state: SizerState;
  target: SizingTarget;
  estimator: Estimator;
  client: UnslothClient;
  stamp: CalibrationStamp;
  /** The entry already stored for this model, so a write preserves the rest. */
  existing: ModelOverride | undefined;
  overrideKey: string;
}

/** The choices a cycle offers, with whatever the model already uses folded in. */
function choicesWith<T>(known: readonly T[], current: T): readonly T[] {
  return known.includes(current) ? known : [...known, current];
}

/**
 * Ask the machine, the server and the model everything this screen needs.
 *
 * Returns a string when there is nothing to open — a server too old, a machine
 * with no topology, a model that is not on disk — because each of those is a
 * sentence the user needs rather than an empty panel.
 */
async function collect(ctx: ExtensionContext, modelId: string): Promise<Collected | string> {
  const client = createSessionClient(ctx);
  const call = { timeoutMs: SIZER_TIMEOUT_MS };

  const topology = await detectTopology(client, call);
  if (topology.degraded) return `no GPU topology — ${topology.reason ?? "the server did not answer"}`;
  if (!topology.sizing.enabled) return topology.sizing.reason ?? "sizing is not available on this server";

  const controller = new AbortController();
  const resolution = await resolveModel(client, modelId, controller.signal);
  if (!resolution.modelPath) return `${modelId} is not on disk — Unsloth cannot size it`;

  const gpus = topology.gpus.map(toFacts);
  const target: SizingTarget = { modelPath: resolution.modelPath, quant: resolution.quant };
  const estimator = cachedEstimator((request) => estimateMemory(client, request, call));

  // The ceiling, in the order the facts are trustworthy: what the catalogue
  // says about this model, then what the estimator priced when asked for the
  // model's *native* context — which is what an omitted `n_ctx` means.
  const entry: CatalogueEntry | undefined = resolution.entry;
  let ceiling = entry?.maxContextLength ?? entry?.nativeContextLength;
  let native: MemoryEstimate | undefined;
  if (ceiling === undefined) {
    native = await estimator({
      model_path: target.modelPath,
      ...(target.quant ? { gguf_variant: target.quant } : {}),
    }).catch(() => undefined);
    if (native && !native.available) return `${modelId} cannot be sized — ${describeUnavailable(native.reason)}`;
    ceiling = native?.nCtx;
  }
  if (ceiling === undefined || ceiling <= 0) return `${modelId}: the server does not report a context length for it`;

  const profile = readMachineProfile().profile;
  const step = typeof profile?.policy?.ctxStepTokens === "number" ? profile.policy.ctxStepTokens : DEFAULT_CTX_STEP;
  const stamp: CalibrationStamp = { unslothVersion: topology.unslothVersion, backend: topology.backend };
  const calibration = readCalibration(profile, stamp);

  // What the model is tuned to today is the starting point: the sizer opens on
  // the user's own configuration, not on a proposal that discards it.
  const stored = resolution.override;
  const storedContext = typeof stored?.custom_context_length === "number" ? stored.custom_context_length : undefined;
  const base = configFromOverride(stored, { kind: "single", gpuIds: [], shares: [] }, storedContext ?? ceiling);
  if (base.nParallel === undefined) base.nParallel = MIN_SLOTS;
  if (base.kvDtype === undefined && typeof profile?.policy?.kvDtype === "string") base.kvDtype = profile.policy.kvDtype;

  const preferHeadless = profile?.policy?.preferHeadless;
  const proposal = await sizeModel(estimator, target, gpus, base, {
    ceiling,
    target: storedContext ?? ceiling,
    step,
    calibrationGb: calibration?.deltaGb ?? 0,
    ...(typeof preferHeadless === "boolean" ? { preferHeadless } : {}),
  });

  if (proposal.unsizable) return `${modelId} cannot be sized — ${describeUnavailable(proposal.unsizable)}`;

  const config = proposal.config ?? {
    ...base,
    placement: proposal.options[0]?.placement ?? { kind: "single", gpuIds: [], shares: [] },
    contextTokens: snapContext(storedContext ?? ceiling, step),
  };

  const model: SizerModel = {
    id: modelId,
    name: resolution.entry?.displayName ?? resolution.local?.displayName ?? modelId,
    quant: resolution.quant,
    modelPath: target.modelPath,
    ceiling,
  };

  const state: SizerState = {
    model,
    gpus,
    config,
    step,
    options: proposal.options,
    kvChoices: choicesWith(KV_DTYPES, config.kvDtype),
    specChoices: choicesWith(SPECULATIVE_MODES, config.speculativeType),
    field: 0,
    estimate: proposal.estimate,
    verdict: proposal.verdict,
    maxContext: proposal.options.find((option) => option.label === placementLabel(config.placement))?.maxContext,
    busy: false,
    problem: undefined,
    caveat: proposal.estimate ? estimateCaveat(proposal.estimate) : undefined,
    calibration,
    confirming: false,
    note: undefined,
  };

  return {
    state,
    target,
    estimator,
    client,
    stamp,
    existing: stored,
    overrideKey: overrideKey(target.modelPath, target.quant),
  };
}

/** Re-price the current configuration, and re-search when the ceiling moved. */
async function reprice(collected: Collected, state: SizerState, research: boolean): Promise<SizerState> {
  const next: SizerState = { ...state, busy: false };
  const estimate = await collected.estimator(buildEstimateRequest(collected.target, state.config)).catch(() => undefined);
  if (!estimate) {
    next.problem = "the server stopped answering the estimator";
    return next;
  }
  if (!estimate.available) {
    next.problem = describeUnavailable(estimate.reason);
    return next;
  }
  next.estimate = estimate;
  next.verdict = fitOf(estimate, state.gpus, state.config.placement, state.calibration?.deltaGb ?? 0);
  next.caveat = estimateCaveat(estimate);
  next.problem = undefined;

  if (research) {
    next.maxContext = await maxContextFor(collected.estimator, collected.target, state.gpus, state.config, {
      ceiling: state.model.ceiling,
      step: state.step,
      calibrationGb: state.calibration?.deltaGb ?? 0,
    });
  }
  return next;
}

/** Draw the sizer and resolve once the user has decided something. */
function showSizer(ctx: ExtensionContext, collected: Collected, initial: SizerState): Promise<{ state: SizerState; outcome: SizerOutcome }> {
  return ctx.ui.custom<{ state: SizerState; outcome: SizerOutcome }>(
    (tui, theme, _keybindings, done) => {
      let state = initial;
      let finished = false;
      let generation = 0;

      const settle = (outcome: SizerOutcome): void => {
        if (finished) return;
        finished = true;
        done({ state, outcome });
      };

      const container = new Container();
      container.addChild(new Lines((width) => sizerLines(state, width, theme)));
      const framed = new Framed(() => sizerTitle(state), container, theme);

      const price = (research: boolean): void => {
        const mine = ++generation;
        state = { ...state, busy: true };
        tui.requestRender();
        void reprice(collected, state, research).then((next) => {
          // A price that arrives after the user has moved on describes a
          // configuration nobody is looking at any more.
          if (finished || mine !== generation) return;
          state = next;
          tui.requestRender();
        });
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
          const next = applySizerKey(state, input);
          state = next.state;
          if (next.outcome === "open") {
            if (next.reprice) price(next.research);
            else tui.requestRender();
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

/** `⏎` — write this model's override, and nothing else's. */
async function apply(ctx: ExtensionContext, collected: Collected, state: SizerState): Promise<void> {
  const client = collected.client;
  const call = { timeoutMs: SIZER_TIMEOUT_MS };
  const before = await getOverrides(client, call).catch(() => undefined);
  const entry = mergeOverride(collected.existing, buildOverrideEntry(state.config));

  let after;
  try {
    after = await putOverride(client, collected.overrideKey, entry, call);
  } catch (error) {
    report(ctx, `✗ Could not save the override: ${error instanceof Error ? error.message : String(error)}`, "error");
    return;
  }

  // The endpoint writes one model's entry, so the others survive by
  // construction — but "by construction" is a claim, and this is the one place
  // that can check it for free.
  const lost = Object.keys(before ?? {}).filter((key) => key !== collected.overrideKey && after[key] === undefined);
  if (lost.length > 0) {
    report(ctx, `⚠ Saved, but ${lost.length} other override${lost.length === 1 ? "" : "s"} disappeared: ${lost.join(", ")}`, "warning");
  }

  // Pi's catalogue carries the context window, so the model's entry in `/model`
  // is stale until the catalogue is re-read.
  await refreshCatalogue(ctx);
  report(
    ctx,
    `✓ ${state.model.name} sized — ${formatContext(state.config.contextTokens)} on ${placementLabel(state.config.placement)}`,
    "info",
  );
}

/** `v` — load once, measure, unload, and say what was learned. */
async function verify(ctx: ExtensionContext, collected: Collected, state: SizerState): Promise<SizerState> {
  if (!state.estimate) return state;
  const result = await runVerification(ctx, collected.client, {
    target: collected.target,
    label: state.model.name,
    config: state.config,
    estimate: state.estimate,
    stamp: collected.stamp,
  });

  switch (result.outcome) {
    case "measured": {
      const measurement = result.measurement;
      const delta = measurement ? measurement.deltaGb : 0;
      const sign = delta >= 0 ? "+" : "";
      const kept = result.saved ? "" : " (not saved — no machine profile yet; run /unsloth setup)";
      report(
        ctx,
        `✓ Verified ${state.model.name}: ${gb(measurement?.measuredGb ?? 0)} held, estimate said ${gb(
          measurement?.estimatedGb ?? 0,
        )} (${sign}${gb(delta, 2)})${kept}`,
        "info",
      );
      // A measurement moves every figure on the screen, `max safe` included —
      // which is the whole point of having measured. Re-deciding here rather
      // than waiting for the next keystroke means the screen the user comes
      // back to is the corrected one, not the one they left.
      const corrected: SizerState = {
        ...state,
        calibration: result.calibration,
        note: `measured ${sign}${gb(delta, 2)} against the estimate${result.saved ? "" : " — not saved"}`,
      };
      const calibrationGb = result.calibration?.deltaGb ?? 0;
      if (corrected.estimate) {
        corrected.verdict = fitOf(corrected.estimate, corrected.gpus, corrected.config.placement, calibrationGb);
      }
      corrected.maxContext = await maxContextFor(
        collected.estimator,
        collected.target,
        corrected.gpus,
        corrected.config,
        { ceiling: corrected.model.ceiling, step: corrected.step, calibrationGb },
      );
      return corrected;
    }
    case "load-failed":
      report(ctx, `✗ Verification load failed: ${result.error ?? "unknown error"}`, "error");
      return { ...state, note: `verification load failed${result.error ? `: ${result.error}` : ""}` };
    case "no-measurement":
      // The load reported ready and the cards never showed the weights. Saying
      // "0 GiB held" here would be a calibration that the model is free.
      report(ctx, "⚠ Loaded, but the GPUs never reported the model's memory — nothing was learned", "warning");
      return { ...state, note: "loaded, but the cards never showed the weights" };
    case "cancelled":
      return { ...state, note: "verification cancelled — nothing left loaded" };
  }
}

/**
 * Size a model.
 *
 * A loop, like the panel: verifying returns to the screen with what it learned
 * applied, because the number the user came for has just changed.
 */
export async function openSizer(ctx: ExtensionContext, modelId: string): Promise<void> {
  const collected = await collect(ctx, modelId);
  if (typeof collected === "string") {
    report(ctx, `⚠ ${collected}`, "warning");
    return;
  }

  if (ctx.mode !== "tui" || terminalColumns() < MIN_OVERLAY_COLUMNS) {
    // Nowhere to accept anything, so nothing is written.
    report(ctx, sizerSummary(collected.state), collected.state.verdict?.fits ? "info" : "warning");
    return;
  }

  let state = collected.state;
  for (;;) {
    const result = await showSizer(ctx, collected, state);
    state = result.state;
    if (result.outcome === "cancel") return;
    if (result.outcome === "apply") {
      await apply(ctx, collected, state);
      return;
    }
    state = await verify(ctx, collected, state);
    if (terminalColumns() < MIN_OVERLAY_COLUMNS) return;
  }
}

/**
 * `/unsloth add <model>` — find the model the user named, then size it.
 *
 * Matching is deliberately forgiving (case, and any unique substring) because
 * the catalogue's ids are paths and repo names, and nobody types
 * `ggml-org/Qwen3-4B-GGUF` twice. An ambiguous name is reported with the
 * candidates rather than resolved by picking the first.
 */
export async function sizeByName(ctx: ExtensionContext, name: string): Promise<void> {
  const wanted = name.trim();
  if (wanted === "") {
    report(ctx, "Usage: /unsloth add <model>", "warning");
    return;
  }

  const client = createSessionClient(ctx);
  const entries = await listModels(client, { timeoutMs: SIZER_TIMEOUT_MS }).catch(() => [] as CatalogueEntry[]);
  if (entries.length === 0) {
    report(ctx, "⚠ No models to size — the server did not answer", "warning");
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
  await openSizer(ctx, match.entry.id);
}
