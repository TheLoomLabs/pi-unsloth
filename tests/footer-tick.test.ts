import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";

import { state } from "../src/state.ts";
import {
  STATUS_KEY,
  footerRunning,
  footerTarget,
  onPaint,
  paint,
  setFooterVisible,
  startFooter,
  stopFooter,
  type FooterContext,
} from "../src/ui/footer.ts";

/**
 * A stand-in for Pi's context. The 4 s tick never fires inside a test, so
 * nothing here reaches the network — this is about the timer's lifecycle, which
 * is a standing cross-cutting gate.
 */
function fakeContext(hasUI = true): FooterContext & { statuses: (string | undefined)[] } {
  const statuses: (string | undefined)[] = [];
  return {
    hasUI,
    statuses,
    ui: {
      setStatus(key: string, text: string | undefined) {
        assert.equal(key, STATUS_KEY);
        statuses.push(text);
      },
      theme: { fg: (_color: string, text: string) => text },
    },
    modelRegistry: {
      getApiKeyForProvider: async () => undefined,
    },
  } as FooterContext & { statuses: (string | undefined)[] };
}

afterEach(() => {
  stopFooter(fakeContext());
  state.server = "unknown";
  state.activeModelLabel = undefined;
  state.gpus = [];
  state.footerVisible = true;
});

describe("footer tick lifecycle", () => {
  it("starts one timer and clears it on shutdown", () => {
    const ctx = fakeContext();
    startFooter(ctx);
    assert.equal(footerRunning(), true);
    stopFooter(ctx);
    assert.equal(footerRunning(), false);
  });

  it("is idempotent: a second start does not add a second timer", () => {
    const ctx = fakeContext();
    startFooter(ctx);
    startFooter(ctx);
    assert.equal(footerRunning(), true);
    stopFooter(ctx);
    assert.equal(footerRunning(), false);
  });

  it("survives a shutdown with no session, and a second shutdown", () => {
    const ctx = fakeContext();
    stopFooter(ctx);
    stopFooter(ctx);
    assert.equal(footerRunning(), false);
  });

  it("clears the status line on shutdown so nothing survives /new", () => {
    const ctx = fakeContext();
    state.server = "offline";
    startFooter(ctx);
    stopFooter(ctx);
    assert.equal(ctx.statuses.at(-1), undefined);
  });

  it("starts no timer and draws nothing without a UI (pi -p)", () => {
    const ctx = fakeContext(false);
    startFooter(ctx);
    assert.equal(footerRunning(), false);
    paint(ctx);
    assert.deepEqual(ctx.statuses, []);
  });
});

/**
 * The overlays draw from the same state as the footer and redraw when it does,
 * rather than each running a timer and a poll of its own. That is what keeps
 * the "every timer cleared in session_shutdown" gate to two owners however many
 * overlays exist.
 */
describe("redrawing with the footer", () => {
  it("tells subscribers to redraw whenever the line is repainted", () => {
    const ctx = fakeContext();
    let redraws = 0;
    const stop = onPaint(() => {
      redraws++;
    });
    paint(ctx);
    paint(ctx);
    assert.equal(redraws, 2);
    stop();
    paint(ctx);
    assert.equal(redraws, 2);
  });

  it("still repaints the status line when a subscriber throws", () => {
    const ctx = fakeContext();
    const stop = onPaint(() => {
      throw new Error("a disposed overlay");
    });
    state.server = "offline";
    paint(ctx);
    assert.equal(ctx.statuses.at(-1), "◌ unsloth offline");
    stop();
  });

  it("drops every subscriber on shutdown, so none survives /new", () => {
    const ctx = fakeContext();
    let redraws = 0;
    onPaint(() => {
      redraws++;
    });
    stopFooter(ctx);
    paint(ctx);
    assert.equal(redraws, 0);
  });

  it("notifies subscribers even without a UI, so a caller cannot rely on order", () => {
    const ctx = fakeContext(false);
    let redraws = 0;
    const stop = onPaint(() => {
      redraws++;
    });
    paint(ctx);
    assert.equal(redraws, 1);
    stop();
  });
});

describe("switching the footer off", () => {
  it("clears the line rather than leaving the last one stale", () => {
    const ctx = fakeContext();
    state.server = "offline";
    startFooter(ctx);
    assert.notEqual(ctx.statuses.at(-1), undefined);

    setFooterVisible(ctx, false);
    assert.equal(ctx.statuses.at(-1), undefined);
  });

  it("stops the poll, which is the other half of what off means", () => {
    const ctx = fakeContext();
    startFooter(ctx);
    assert.equal(footerRunning(), true);

    setFooterVisible(ctx, false);
    assert.equal(footerRunning(), false);
  });

  it("keeps the poll alive for an overlay that is open", () => {
    // The panel's bars draw from this tick. Hiding the status line must not
    // freeze them — the poll follows the views, not the line.
    const ctx = fakeContext();
    startFooter(ctx);
    const stop = onPaint(() => {}, ctx);

    setFooterVisible(ctx, false);
    assert.equal(footerRunning(), true);

    stop();
    assert.equal(footerRunning(), false);
  });

  it("starts the poll for an overlay opened while it is off", () => {
    const ctx = fakeContext();
    state.footerVisible = false;
    startFooter(ctx);
    assert.equal(footerRunning(), false);

    const stop = onPaint(() => {}, ctx);
    assert.equal(footerRunning(), true);
    stop();
  });

  it("still tells an overlay to redraw while the line is hidden", () => {
    const ctx = fakeContext();
    let redraws = 0;
    const stop = onPaint(() => {
      redraws++;
    }, ctx);
    setFooterVisible(ctx, false);
    paint(ctx);
    assert.ok(redraws >= 2);
    stop();
  });

  it("draws again, from current state, when switched back on", () => {
    const ctx = fakeContext();
    state.server = "offline";
    startFooter(ctx);
    setFooterVisible(ctx, false);
    setFooterVisible(ctx, true);
    assert.notEqual(ctx.statuses.at(-1), undefined);
    assert.equal(footerRunning(), true);
  });
});

describe("footerTarget", () => {
  it("toggles when the subcommand names nothing", () => {
    assert.equal(footerTarget(undefined, true), false);
    assert.equal(footerTarget(undefined, false), true);
  });

  it("is absolute when it names a state, so `on` twice is not a toggle", () => {
    assert.equal(footerTarget("on", true), true);
    assert.equal(footerTarget("off", false), false);
    assert.equal(footerTarget("ON", false), true);
  });

  it("refuses a word it does not know rather than guessing at a toggle", () => {
    assert.equal(footerTarget("maybe", true), undefined);
    assert.equal(footerTarget("0", true), undefined);
  });
});
