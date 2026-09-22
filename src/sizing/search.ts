/**
 * Sizing: how much context fits, and where the model goes.
 *
 * This is all of `register-model.py` that survives the move to the API, and it
 * is smaller than its constant table was: **pick GPUs → binary-search `n_ctx`
 * against the estimator → check the per-GPU fit rule.** No GGUF header, no
 * KV-bytes-per-element, no `overhead = 3000`.
 *
 * Three rules govern everything here:
 *
 *   - **Never OOM-probe.** The search runs entirely against `estimate-memory`,
 *     which allocates nothing. A real load only ever happens at a value already
 *     believed to fit.
 *   - **`available` is not a fit check.** The estimator answers a system-wide
 *     question that permits CPU offload; the per-GPU rule in
 *     src/hardware/budget.ts is the extension's own and is the one that
 *     decides.
 *   - **Every number is measured or asked for.** The only literal in this file
 *     is the context step, which is a *policy* the profile can change, shown
 *     and edited in the sizer.
 *
 * The estimator is passed in rather than fetched, so every case that matters —
 * a model too big for any card, a machine with one GPU, a header that cannot
 * size its own KV cache — is a unit test instead of hardware someone has to
 * own.
 */

import type { EstimateRequest, MemoryEstimate } from "../api/estimate.ts";
import type { ModelOverride } from "../api/lifecycle.ts";
import {
  bytesToGb,
  fitsPlacement,
  placementLadder,
  type FitVerdict,
  type GpuFacts,
  type Placement,
  type PlacementOptions,
} from "../hardware/budget.ts";

const GIB = 1024 ** 3;

/**
 * Context lengths are snapped to this many tokens.
 *
 * `ctxStepTokens` in the machine profile, 4096 by default: llama.cpp allocates
 * the cache in blocks, and a context of 63 744 buys nothing over 61 440 but
 * makes every number on screen unreadable. Snapping is always **down**, so the
 * answer stays inside the budget it was checked against.
 */
export const DEFAULT_CTX_STEP = 4096;

/** The model being sized, in the two terms every endpoint here needs. */
export interface SizingTarget {
  /** Absolute path, or repo id for a hub model — what `model_path` takes. */
  modelPath: string;
  quant: string | undefined;
}

/**
 * One complete configuration.
 *
 * Every field maps to a field of the estimate, the load and the override, so
 * what is priced, what is written and what runs cannot drift apart. `undefined`
 * means "the server's own default" everywhere — an absent field is never filled
 * in with a guess of ours.
 */
export interface SizingConfig {
  contextTokens: number;
  kvDtype: string | undefined;
  placement: Placement;
  speculativeType: string | undefined;
  specDraftNMax: number | undefined;
  /**
   * Decode slots. One by default: Pi holds one conversation, and the server's
   * own default of four quadruples the KV cache to serve callers that do not
   * exist. Shown in the breakdown, because it is the second-biggest lever.
   */
  nParallel: number | undefined;
  nBatch: number | undefined;
  nUbatch: number | undefined;
  disableVision: boolean | undefined;
}

/** `single: GPU 1` / `tensor-parallel 0+1` — the words uses. */
export function placementLabel(placement: Placement): string {
  if (placement.kind === "tensor-parallel") return `tensor-parallel ${placement.gpuIds.join("+")}`;
  return `single: GPU ${placement.gpuIds.join("+")}`;
}

/** Snap down to a whole number of steps, never below one step. */
export function snapContext(tokens: number, step: number = DEFAULT_CTX_STEP): number {
  if (!Number.isFinite(tokens) || !Number.isFinite(step) || step <= 0) return 0;
  return Math.max(step, Math.floor(tokens / step) * step);
}

/** The estimate request for one configuration. */
export function buildEstimateRequest(target: SizingTarget, config: SizingConfig): EstimateRequest {
  const request: EstimateRequest = {
    model_path: target.modelPath,
    n_ctx: config.contextTokens,
    tensor_parallel: config.placement.kind === "tensor-parallel",
    selected_gpu_ids: [...config.placement.gpuIds],
  };
  if (target.quant) request.gguf_variant = target.quant;
  if (config.kvDtype) request.cache_type_kv = config.kvDtype;
  if (config.nParallel !== undefined) request.n_parallel = config.nParallel;
  if (config.nBatch !== undefined) request.n_batch = config.nBatch;
  if (config.nUbatch !== undefined) request.n_ubatch = config.nUbatch;
  if (config.speculativeType !== undefined) request.speculative_type = config.speculativeType;
  if (config.specDraftNMax !== undefined) request.spec_draft_n_max = config.specDraftNMax;
  if (config.disableVision !== undefined) request.disable_vision = config.disableVision;
  return request;
}

/**
 * The override entry this configuration becomes.
 *
 * The names change on the way — `custom_context_length`, `kv_cache_dtype` —
 * because the override spells two fields differently from the load, which is
 * the same rename `buildLoadRequest` undoes in the other direction.
 */
export function buildOverrideEntry(config: SizingConfig): ModelOverride {
  const entry: ModelOverride = {
    custom_context_length: config.contextTokens,
    gpu_ids: [...config.placement.gpuIds],
  };
  // Written only when true: a stored `tensor_parallel: false` is dropped by the
  // server anyway, so writing it would make a round-trip look like a change.
  if (config.placement.kind === "tensor-parallel") entry.tensor_parallel = true;
  if (config.kvDtype) entry.kv_cache_dtype = config.kvDtype;
  if (config.speculativeType !== undefined) entry.speculative_type = config.speculativeType;
  if (config.specDraftNMax !== undefined) entry.spec_draft_n_max = config.specDraftNMax;
  if (config.nParallel !== undefined) entry.n_parallel = config.nParallel;
  if (config.nBatch !== undefined) entry.n_batch = config.nBatch;
  if (config.nUbatch !== undefined) entry.n_ubatch = config.nUbatch;
  if (config.disableVision !== undefined) entry.disable_vision = config.disableVision;
  return entry;
}

/** The reverse: what a stored override says this model was tuned to. */
export function configFromOverride(
  override: ModelOverride | undefined,
  placement: Placement,
  contextTokens: number,
): SizingConfig {
  const number = (value: unknown): number | undefined =>
    typeof value === "number" && Number.isFinite(value) ? value : undefined;
  const text = (value: unknown): string | undefined =>
    typeof value === "string" && value !== "" ? value : undefined;
  return {
    contextTokens,
    kvDtype: text(override?.kv_cache_dtype),
    placement,
    speculativeType: text(override?.speculative_type),
    specDraftNMax: number(override?.spec_draft_n_max),
    nParallel: number(override?.n_parallel),
    nBatch: number(override?.n_batch),
    nUbatch: number(override?.n_ubatch),
    disableVision: typeof override?.disable_vision === "boolean" ? override.disable_vision : undefined,
  };
}

/**
 * A learned correction to the estimator, in GiB.
 *
 * Added to `gpu_bytes` before the fit rule sees it, so a machine that has been
 * verified once sizes against what it actually measured rather than against the
 * model of it. Zero when nothing has been measured, which is the same
 * arithmetic as not having the feature.
 */
export function correctedGpuBytes(estimate: MemoryEstimate, calibrationGb = 0): number {
  return Math.max(0, estimate.gpuBytes + calibrationGb * GIB);
}

/** The per-GPU verdict for one estimate, corrected and placed. */
export function fitOf(
  estimate: MemoryEstimate,
  gpus: readonly GpuFacts[],
  placement: Placement,
  calibrationGb = 0,
): FitVerdict {
  return fitsPlacement(correctedGpuBytes(estimate, calibrationGb), gpus, placement);
}

/** How much each GPU of a placement is expected to hold, in GiB. */
export function perGpuGb(estimate: MemoryEstimate, placement: Placement, calibrationGb = 0): number[] {
  const total = bytesToGb(correctedGpuBytes(estimate, calibrationGb));
  return placement.gpuIds.map((_, position) => total * (placement.shares[position] ?? 1));
}

/** Anything that turns a request into an estimate: the client, or a test. */
export type Estimator = (request: EstimateRequest) => Promise<MemoryEstimate>;

/**
 * Memoise an estimator by its request.
 *
 * The search revisits the same configuration constantly — every placement
 * candidate re-asks about the ceiling, and every keystroke in the sizer
 * re-asks about the context it already showed — and the answers cannot change
 * within one sizing session, because nothing here allocates anything.
 */
export function cachedEstimator(estimator: Estimator): Estimator {
  const cache = new Map<string, Promise<MemoryEstimate>>();
  return (request) => {
    const key = JSON.stringify(request);
    const held = cache.get(key);
    if (held) return held;
    const fresh = estimator(request);
    cache.set(key, fresh);
    // A failed estimate must not be remembered as an answer.
    void fresh.catch(() => cache.delete(key));
    return fresh;
  };
}

export interface SearchBounds {
  /** Never search below this; one step by default. */
  min?: number;
  /** The model's own ceiling — `native_context_length`, never guessed past. */
  max: number;
  step?: number;
}

/**
 * The largest snapped context for which `fits` holds, or `undefined`.
 *
 * A descending binary search **against the estimator**, which is free: the
 * ceiling is tried first because it is the common answer for a small model on
 * a big card, and the floor is tried before searching at all so that "nothing
 * fits" costs two calls rather than seven.
 *
 * Monotonicity is the assumption — more context never needs less memory — and
 * it is the estimator's own arithmetic, not ours.
 */
export async function searchMaxContext(
  fits: (contextTokens: number) => Promise<boolean>,
  bounds: SearchBounds,
): Promise<number | undefined> {
  const step = bounds.step && bounds.step > 0 ? bounds.step : DEFAULT_CTX_STEP;
  const highest = Math.floor(bounds.max / step);
  const lowest = Math.max(1, Math.ceil((bounds.min ?? step) / step));
  if (highest < lowest) return undefined;

  if (await fits(highest * step)) return highest * step;
  if (highest === lowest) return undefined;
  if (!(await fits(lowest * step))) return undefined;

  // Invariant: `low` fits, `high` does not.
  let low = lowest;
  let high = highest;
  while (high - low > 1) {
    const middle = low + Math.floor((high - low) / 2);
    if (await fits(middle * step)) low = middle;
    else high = middle;
  }
  return low * step;
}

/** One rung of the placement ladder, with what it could hold. */
export interface PlacementOption {
  placement: Placement;
  label: string;
  /** Largest context that fits here, or `undefined` when nothing does. */
  maxContext: number | undefined;
  /** The estimate at `maxContext`, or at the ceiling when nothing fits. */
  estimate: MemoryEstimate | undefined;
  /** The verdict at `maxContext`, or the near-miss at the ceiling. */
  verdict: FitVerdict | undefined;
}

export interface SizingOptions extends PlacementOptions {
  /** The model's ceiling: `native_context_length`, or what the header says. */
  ceiling: number;
  /** Context the user is asking for. The ceiling when they have not said. */
  target?: number;
  step?: number;
  calibrationGb?: number;
  signal?: AbortSignal;
}

export interface SizingProposal {
  /** The recommended configuration, or `undefined` when nothing fits. */
  config: SizingConfig | undefined;
  estimate: MemoryEstimate | undefined;
  verdict: FitVerdict | undefined;
  /** Every rung, in preference order — what the sizer's `‹ placement ›` cycles. */
  options: PlacementOption[];
  /** Why no breakdown could be produced at all, when none could. */
  unsizable: string | undefined;
}

/**
 * Where this model should go and how much context it can have.
 *
 * The ladder is walked twice, deliberately. First every rung is priced at the
 * **target** context, and the first rung that fits wins — that is the
 * preference order doing its job, and it costs one estimate per rung. Only if
 * none of them reaches the target does each rung get searched for its own
 * maximum, and the roomiest answer wins, ties going to the earlier rung.
 *
 * So a model that fits on a headless card at full context never gets sized onto
 * the display GPU, and a model that fits nowhere at full context is offered the
 * largest context this machine can actually hold.
 */
export async function sizeModel(
  estimator: Estimator,
  target: SizingTarget,
  gpus: readonly GpuFacts[],
  base: SizingConfig,
  options: SizingOptions,
): Promise<SizingProposal> {
  const step = options.step && options.step > 0 ? options.step : DEFAULT_CTX_STEP;
  const ceiling = snapContext(options.ceiling, step);
  const wanted = snapContext(Math.min(options.target ?? options.ceiling, options.ceiling), step);
  const calibrationGb = options.calibrationGb ?? 0;
  const estimate = cachedEstimator(estimator);

  const ladder = placementLadder(gpus, options.preferHeadless === undefined ? {} : { preferHeadless: options.preferHeadless });
  if (ladder.length === 0) {
    return { config: undefined, estimate: undefined, verdict: undefined, options: [], unsizable: undefined };
  }

  const at = async (placement: Placement, contextTokens: number): Promise<MemoryEstimate> =>
    estimate(buildEstimateRequest(target, { ...base, placement, contextTokens }));

  // Pass one: the target, on every rung, in preference order.
  const results: PlacementOption[] = [];
  for (const placement of ladder) {
    const priced = await at(placement, wanted);
    if (!priced.available) {
      return {
        config: undefined,
        estimate: priced,
        verdict: undefined,
        options: [],
        unsizable: priced.reason ?? "unsizable",
      };
    }
    const verdict = fitOf(priced, gpus, placement, calibrationGb);
    results.push({ placement, label: placementLabel(placement), maxContext: undefined, estimate: priced, verdict });
    if (verdict.fits) {
      // `max safe` is a maximum, not "the target fits" — so the rung that won
      // is searched above the target even though the answer is already known
      // to be good enough. Without this, a model whose stored context is well
      // under what the machine can hold reports its own setting as the
      // ceiling, which is the one number on this screen nobody could check.
      const chosen = results[results.length - 1];
      if (chosen) {
        chosen.maxContext =
          wanted >= ceiling
            ? wanted
            : ((await searchMaxContext(
                async (contextTokens) => fitOf(await at(placement, contextTokens), gpus, placement, calibrationGb).fits,
                { min: wanted, max: ceiling, step },
              )) ?? wanted);
      }
      // Every later rung is still offered to the user, but unsearched: the
      // answer is already the preferred one, and a search costs requests.
      for (const rest of ladder.slice(results.length)) {
        results.push({ placement: rest, label: placementLabel(rest), maxContext: undefined, estimate: undefined, verdict: undefined });
      }
      const config: SizingConfig = { ...base, placement, contextTokens: wanted };
      return { config, estimate: priced, verdict, options: results, unsizable: undefined };
    }
  }

  // Pass two: nothing reached the target, so ask each rung for its own maximum.
  for (const option of results) {
    if (options.signal?.aborted) break;
    option.maxContext = await searchMaxContext(
      async (contextTokens) => fitOf(await at(option.placement, contextTokens), gpus, option.placement, calibrationGb).fits,
      { max: ceiling, step },
    );
    if (option.maxContext !== undefined) {
      option.estimate = await at(option.placement, option.maxContext);
      option.verdict = fitOf(option.estimate, gpus, option.placement, calibrationGb);
    }
  }

  const best = results.reduce<PlacementOption | undefined>(
    (winner, option) =>
      option.maxContext !== undefined && (winner?.maxContext === undefined || option.maxContext > winner.maxContext)
        ? option
        : winner,
    undefined,
  );

  if (!best || best.maxContext === undefined || !best.estimate) {
    // Nothing fits anywhere. The verdict shown is the best near-miss, which is
    // the number the user can act on.
    const nearest = results.reduce<FitVerdict | undefined>(
      (lowest, option) =>
        option.verdict && (!lowest || option.verdict.shortfallGb < lowest.shortfallGb) ? option.verdict : lowest,
      undefined,
    );
    return {
      config: undefined,
      estimate: results[0]?.estimate,
      verdict: nearest,
      options: results,
      unsizable: undefined,
    };
  }

  return {
    config: { ...base, placement: best.placement, contextTokens: best.maxContext },
    estimate: best.estimate,
    verdict: best.verdict,
    options: results,
    unsizable: undefined,
  };
}

/**
 * The `max safe` figure for one placement.
 *
 * Used when the user changes something the search's answer depends on — the KV
 * dtype, the placement, the drafter — so the line above the context field keeps
 * telling the truth rather than the truth as it was two keystrokes ago.
 */
export async function maxContextFor(
  estimator: Estimator,
  target: SizingTarget,
  gpus: readonly GpuFacts[],
  config: SizingConfig,
  options: SizingOptions,
): Promise<number | undefined> {
  const step = options.step && options.step > 0 ? options.step : DEFAULT_CTX_STEP;
  return searchMaxContext(
    async (contextTokens) => {
      const priced = await estimator(buildEstimateRequest(target, { ...config, contextTokens }));
      if (!priced.available) return false;
      return fitOf(priced, gpus, config.placement, options.calibrationGb ?? 0).fits;
    },
    { max: snapContext(options.ceiling, step), step },
  );
}
