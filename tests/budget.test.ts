import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  DISPLAY_HEADROOM_GB,
  HEADLESS_HEADROOM_GB,
  budgetGb,
  bytesToGb,
  choosePlacement,
  defaultHeadroomGb,
  fitsPlacement,
  singlePlacement,
  tensorPlacement,
  tensorShares,
  type GpuFacts,
} from "../src/hardware/budget.ts";

const GIB = 1024 ** 3;

/** The reference box: two 24 GiB cards, one of them driving the monitor. */
const TWO_IDENTICAL: GpuFacts[] = [
  { index: 0, totalGb: 23.98, idleUsedGb: 1.21, display: true, headroomGb: DISPLAY_HEADROOM_GB },
  { index: 1, totalGb: 23.98, idleUsedGb: 0.03, display: false, headroomGb: HEADLESS_HEADROOM_GB },
];

/** Mismatched cards — the case `DISPLAY_SHARE = 0.51` could never express. */
const MISMATCHED: GpuFacts[] = [
  { index: 0, totalGb: 23.98, idleUsedGb: 1.21, display: true, headroomGb: DISPLAY_HEADROOM_GB },
  { index: 1, totalGb: 11.99, idleUsedGb: 0.03, display: false, headroomGb: HEADLESS_HEADROOM_GB },
];

/** One card, and it is the one with the monitor on it. */
const SINGLE_DESKTOP: GpuFacts[] = [
  { index: 0, totalGb: 23.98, idleUsedGb: 1.21, display: true, headroomGb: DISPLAY_HEADROOM_GB },
];

/** A rented box with no desktop at all. Two GPUs, neither of them display. */
const HEADLESS: GpuFacts[] = [
  { index: 0, totalGb: 79.15, idleUsedGb: 0.0, display: false, headroomGb: HEADLESS_HEADROOM_GB },
  { index: 1, totalGb: 79.15, idleUsedGb: 0.0, display: false, headroomGb: HEADLESS_HEADROOM_GB },
];

describe("budgetGb", () => {
  it("is total minus what the desktop holds minus the headroom", () => {
    assert.equal(budgetGb(TWO_IDENTICAL[0]!).toFixed(2), "19.77");
    assert.equal(budgetGb(TWO_IDENTICAL[1]!).toFixed(2), "23.45");
  });

  it("never goes negative, however small the card", () => {
    assert.equal(budgetGb({ index: 0, totalGb: 2, idleUsedGb: 1.5, display: true, headroomGb: 3 }), 0);
  });

  it("gives the display GPU the larger default headroom", () => {
    assert.equal(defaultHeadroomGb(true), DISPLAY_HEADROOM_GB);
    assert.equal(defaultHeadroomGb(false), HEADLESS_HEADROOM_GB);
    assert.ok(DISPLAY_HEADROOM_GB > HEADLESS_HEADROOM_GB);
  });
});

describe("tensorShares", () => {
  it("halves across two identical cards — what llama.cpp does, and what 0.51 was", () => {
    // Measured on the reference box: a 38.3 GiB estimate landed as 19.5 + 18.4
    // GiB of model with `--split-mode tensor` and no `--tensor-split`.
    const shares = tensorShares(TWO_IDENTICAL);
    assert.deepEqual(shares, [0.5, 0.5]);
  });

  it("does not shrink the display card's share because its desktop left less room", () => {
    // The dangerous direction: sharing by *budget* charges the card driving the
    // monitor less precisely because it has less to give.
    const [display, headless] = tensorShares(TWO_IDENTICAL);
    assert.equal(display, headless);
    assert.ok(budgetGb(TWO_IDENTICAL[0]!) < budgetGb(TWO_IDENTICAL[1]!));
  });

  it("gives the smaller card the smaller share when they are mismatched", () => {
    const [big, small] = tensorShares(MISMATCHED);
    assert.ok((big ?? 0) > (small ?? 0));
    assert.ok(Math.abs((big ?? 0) + (small ?? 0) - 1) < 1e-9);
    // Proportional to the cards, so a 24 + 12 pair splits two to one.
    assert.equal((big ?? 0).toFixed(2), "0.67");
  });

  it("splits evenly when the server reports no sizes, rather than dividing by zero", () => {
    const sizeless = TWO_IDENTICAL.map((gpu) => ({ ...gpu, totalGb: 0 }));
    assert.deepEqual(tensorShares(sizeless), [0.5, 0.5]);
  });
});

describe("fitsPlacement", () => {
  it("is a per-GPU rule, not the estimator's system-wide `available`", () => {
    // The documented gotcha: 32.4 GB reported `available: true` against a
    // 23.98 GB card, because the estimator permits CPU offload.
    const verdict = fitsPlacement(32.4e9, SINGLE_DESKTOP, singlePlacement(0));
    assert.equal(verdict.fits, false);
    assert.equal(verdict.tightestIndex, 0);
    assert.ok(verdict.shortfallGb > 10);
  });

  it("checks each GPU's own share in a tensor-parallel split", () => {
    const placement = tensorPlacement(MISMATCHED);
    // Comfortably inside the 31 GiB the pair can spare between them.
    assert.equal(fitsPlacement(24 * GIB, MISMATCHED, placement).fits, true);
    const over = fitsPlacement(40 * GIB, MISMATCHED, placement);
    assert.equal(over.fits, false);
    assert.ok(over.shortfallGb > 0);
  });

  it("never fits a placement naming a GPU this machine does not have", () => {
    const verdict = fitsPlacement(1 * GIB, TWO_IDENTICAL, { kind: "single", gpuIds: [7], shares: [1] });
    assert.equal(verdict.fits, false);
    assert.equal(verdict.tightestIndex, 7);
  });
});

describe("choosePlacement", () => {
  it("keeps a model that fits off the display GPU", () => {
    const choice = choosePlacement(18 * GIB, TWO_IDENTICAL);
    assert.deepEqual(choice.placement?.gpuIds, [1]);
    assert.equal(choice.placement?.kind, "single");
    assert.equal(choice.verdict.fits, true);
  });

  it("goes tensor-parallel when no single card can hold it", () => {
    const choice = choosePlacement(30 * GIB, TWO_IDENTICAL);
    assert.equal(choice.placement?.kind, "tensor-parallel");
    assert.deepEqual(choice.placement?.gpuIds, [0, 1]);
  });

  it("uses the display GPU when it is the only GPU", () => {
    const choice = choosePlacement(15 * GIB, SINGLE_DESKTOP);
    assert.deepEqual(choice.placement?.gpuIds, [0]);
    // …and it is sized against a budget that already gave the desktop 3 GiB.
    assert.ok(budgetGb(SINGLE_DESKTOP[0]!) < 20);
  });

  it("reports the smallest shortfall when nothing fits, not the worst", () => {
    const choice = choosePlacement(60 * GIB, MISMATCHED);
    assert.equal(choice.placement, undefined);
    assert.equal(choice.verdict.fits, false);
    // Over the pair's ~31 GiB by ~29, not over the 12 GiB card by ~50.
    assert.ok(choice.verdict.shortfallGb < 35, `shortfall ${choice.verdict.shortfallGb}`);
  });

  it("treats every card the same on a headless box, and packs the tightest first", () => {
    const uneven: GpuFacts[] = [
      { ...HEADLESS[0]!, totalGb: 79.15 },
      { ...HEADLESS[1]!, totalGb: 23.98 },
    ];
    // Both fit it; the smaller card takes it, leaving the big one free.
    assert.deepEqual(choosePlacement(20 * GIB, uneven).placement?.gpuIds, [1]);
    assert.deepEqual(choosePlacement(40 * GIB, uneven).placement?.gpuIds, [0]);
    assert.equal(choosePlacement(40 * GIB, HEADLESS).placement?.kind, "single");
  });

  it("will use the display GPU first when the user turns preferHeadless off", () => {
    const choice = choosePlacement(18 * GIB, TWO_IDENTICAL, { preferHeadless: false });
    // Tightest-first among all cards: the display GPU has the smaller budget.
    assert.deepEqual(choice.placement?.gpuIds, [0]);
  });

  it("says what is missing rather than throwing when there are no GPUs", () => {
    const choice = choosePlacement(8 * GIB, []);
    assert.equal(choice.placement, undefined);
    assert.equal(choice.verdict.shortfallGb.toFixed(0), "8");
  });
});

describe("bytesToGb", () => {
  it("converts by 1024³, the same unit the server reports totals in", () => {
    assert.equal(bytesToGb(GIB), 1);
    // The documented 192 K reference estimate for the 27B.
    assert.equal(bytesToGb(42625401988).toFixed(1), "39.7");
  });
});
