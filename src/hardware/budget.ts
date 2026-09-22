/**
 * Budgets, the fit rule, and where a model goes.
 *
 * This is the file that replaces `register-model.py`'s constants — `CARD`,
 * `HEADLESS_USABLE`, `DISPLAY_USABLE`, `DISPLAY_SHARE`, `gpu_ids` — with the
 * same three decisions expressed in terms of numbers the server measured:
 *
 *   budget[g] = total[g] − idle[g] − headroom[g]
 *
 * and then, for a candidate placement, whether the estimator's `gpu_bytes` fits
 * inside those budgets.
 *
 * ⚠ The estimator's own `available: true` is **not** a fit check — it permits
 * CPU offload, and returned `true` for a 32 K estimate wanting 32.4 GB on a
 * 23.98 GB card. Everything here exists because that answer cannot be trusted
 * for placement.
 *
 * A note on units: the server reports `memory_total_gb: 23.98` for a 24 GiB
 * card, so its "gb" is what everyone else calls GiB and what the UI prints.
 * Bytes from the estimator are converted the same way, by 1024³.
 */

/**
 * Default headroom, in the same units the server reports.
 *
 * A display GPU gets the larger one because a compositor and a browser grow
 * while a model is resident; a headless one only needs slack for allocator
 * rounding. Both are shown, explained and editable in the wizard — a default
 * that can be seen and changed is not a hardware constant.
 */
export const DISPLAY_HEADROOM_GB = 3.0;
export const HEADLESS_HEADROOM_GB = 0.5;

export function defaultHeadroomGb(display: boolean): number {
  return display ? DISPLAY_HEADROOM_GB : HEADLESS_HEADROOM_GB;
}

const GIB = 1024 ** 3;

export function bytesToGb(bytes: number): number {
  return bytes / GIB;
}

/** One GPU, as the fit rule needs it. Every figure is measured or user-set. */
export interface GpuFacts {
  /** The id `/api/system` gave it, and the id `gpu_ids` will be given. */
  index: number;
  totalGb: number;
  /** VRAM held with nothing loaded. Zero when nothing has been measured. */
  idleUsedGb: number;
  display: boolean;
  headroomGb: number;
}

/**
 * What a model may use on this GPU.
 *
 * Never negative: a card whose desktop already exceeds its own size is a card
 * with no room, not a card with a negative budget that would then "fit" a
 * negative share of a tensor-parallel split.
 */
export function budgetGb(gpu: GpuFacts): number {
  return Math.max(0, gpu.totalGb - gpu.idleUsedGb - gpu.headroomGb);
}

export type PlacementKind = "single" | "tensor-parallel";

export interface Placement {
  kind: PlacementKind;
  /** Exactly what goes into `gpu_ids`. */
  gpuIds: number[];
  /** Fraction of `gpu_bytes` each of `gpuIds` is expected to hold. */
  shares: number[];
}

/**
 * Tensor-parallel shares, proportional to each card's **total** VRAM.
 *
 * This models what llama.cpp actually does, which is the only thing a fit rule
 * may model. Nothing in the load payload can express a split — there is no
 * `tensor_split` field, and the server's own argv carries `--split-mode tensor`
 * with no `--tensor-split` — so the weights land in the proportions llama.cpp
 * picks for itself, which are by device size, not by how much room each card
 * happens to have left.
 *
 * Measured on the reference box, two identical cards holding a 38.3 GiB
 * estimate: 19.5 GiB and 18.4 GiB of model, i.e. an even split. That is also
 * what `register-model.py`'s hardcoded `DISPLAY_SHARE = 0.51` was: an
 * observation of llama.cpp's own halving, not a budget calculation.
 *
 * ⚠ Sharing by *budget* instead — which is what this function did before the
 * argv was read — charges the display GPU the smaller share precisely because
 * its desktop leaves it less room. It is the wrong direction on the one card
 * where being wrong costs a desktop session.
 *
 * With no sizes reported at all the split is even: the shares are then
 * multiplied by a `gpu_bytes` nobody can place anyway, and an even split at
 * least reports a shortfall against every card instead of dividing by zero.
 */
export function tensorShares(gpus: readonly GpuFacts[]): number[] {
  if (gpus.length === 0) return [];
  const totals = gpus.map((gpu) => (gpu.totalGb > 0 ? gpu.totalGb : 0));
  const total = totals.reduce((sum, value) => sum + value, 0);
  if (total <= 0) return gpus.map(() => 1 / gpus.length);
  return totals.map((value) => value / total);
}

export interface FitVerdict {
  fits: boolean;
  /** How much the tightest GPU is over, in GiB. Zero when it fits. */
  shortfallGb: number;
  /** Which GPU is tightest — the one a message should name. */
  tightestIndex: number | undefined;
}

/**
 * The per-GPU fit rule: does `gpuBytes` fit in this placement's budgets?
 *
 *   single           → gpu_bytes ≤ budget[g]
 *   tensor-parallel  → for each g: gpu_bytes × share[g] ≤ budget[g]
 */
export function fitsPlacement(gpuBytes: number, gpus: readonly GpuFacts[], placement: Placement): FitVerdict {
  const byIndex = new Map(gpus.map((gpu) => [gpu.index, gpu]));
  const wanted = bytesToGb(gpuBytes);

  let worst = 0;
  let tightestIndex: number | undefined;
  placement.gpuIds.forEach((id, position) => {
    const gpu = byIndex.get(id);
    // A placement naming a GPU this machine does not have never fits — the
    // panel's `won't fit`, reached here from the other direction.
    const over = gpu ? wanted * (placement.shares[position] ?? 1) - budgetGb(gpu) : Number.POSITIVE_INFINITY;
    if (tightestIndex === undefined || over > worst) {
      worst = over;
      tightestIndex = id;
    }
  });

  return { fits: worst <= 0, shortfallGb: worst > 0 ? worst : 0, tightestIndex };
}

export function singlePlacement(index: number): Placement {
  return { kind: "single", gpuIds: [index], shares: [1] };
}

export function tensorPlacement(gpus: readonly GpuFacts[]): Placement {
  return { kind: "tensor-parallel", gpuIds: gpus.map((gpu) => gpu.index), shares: tensorShares(gpus) };
}

export interface PlacementChoice {
  /** `undefined` when nothing on this machine can hold it. */
  placement: Placement | undefined;
  verdict: FitVerdict;
}

export interface PlacementOptions {
  /**
   * Keep models off the display GPU while any headless card exists. Default on
   * — a failed load on the card driving the monitor can take the desktop with
   * it, which is the one failure this project treats as unacceptable.
   */
  preferHeadless?: boolean;
}

/**
 * Where a model of `gpuBytes` should go.
 *
 * The ladder, in order: one headless GPU, then one display GPU, then
 * tensor-parallel across every compute GPU, then nowhere. Among single GPUs
 * that fit, the **tightest** is chosen rather than the roomiest: it leaves the
 * larger card free for whatever is loaded next, and every candidate at this
 * point already satisfies the rule.
 *
 * When nothing fits, the verdict is the *best* near-miss — the smallest
 * shortfall of the options considered — because that is the number a user can
 * act on ("over by 2.3 GiB"), not the worst one.
 */
export function choosePlacement(
  gpuBytes: number,
  gpus: readonly GpuFacts[],
  options: PlacementOptions = {},
): PlacementChoice {
  const ladder = placementLadder(gpus, options);
  if (ladder.length === 0) {
    return { placement: undefined, verdict: { fits: false, shortfallGb: bytesToGb(gpuBytes), tightestIndex: undefined } };
  }

  const misses: FitVerdict[] = [];
  for (const placement of ladder) {
    const verdict = fitsPlacement(gpuBytes, gpus, placement);
    if (verdict.fits) return { placement, verdict };
    misses.push(verdict);
  }

  const best = misses.reduce(
    (lowest, verdict) => (verdict.shortfallGb < lowest.shortfallGb ? verdict : lowest),
    misses[0] ?? { fits: false, shortfallGb: bytesToGb(gpuBytes), tightestIndex: undefined },
  );
  return { placement: undefined, verdict: best };
}

/**
 * Every placement worth considering, in preference order.
 *
 * One headless GPU, then one display GPU, then tensor-parallel across every
 * compute GPU — with the singles inside each rung ordered tightest-budget
 * first, so a model that fits on several cards takes the one it fills, leaving
 * the roomier card for whatever is loaded next.
 *
 * `choosePlacement` walks this list and takes the first that fits; the sizer
 * walks the same list asking each rung how much context it could hold
 * (src/sizing/search.ts). One ladder, so the panel's answer and the sizer's
 * cannot disagree about where a model belongs.
 */
export function placementLadder(gpus: readonly GpuFacts[], options: PlacementOptions = {}): Placement[] {
  if (gpus.length === 0) return [];
  const preferHeadless = options.preferHeadless ?? true;
  const headless = gpus.filter((gpu) => !gpu.display);
  const displays = gpus.filter((gpu) => gpu.display);
  // With `preferHeadless` off, a display GPU is simply another candidate
  // rather than a last resort.
  const rungs: GpuFacts[][] = preferHeadless ? [headless, displays] : [[...gpus]];

  const ladder: Placement[] = [];
  for (const rung of rungs) {
    for (const gpu of [...rung].sort((a, b) => budgetGb(a) - budgetGb(b))) {
      ladder.push(singlePlacement(gpu.index));
    }
  }
  if (gpus.length > 1) ladder.push(tensorPlacement(gpus));
  return ladder;
}
