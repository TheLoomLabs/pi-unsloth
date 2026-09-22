import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";

import {
  beginSession,
  ctxLive,
  endSession,
  hasUI,
  sessionAlive,
  sessionSignal,
  type MaybeStaleContext,
} from "../src/session.ts";

/**
 * A ctx that has been retired, as Pi retires one: every property throws, and
 * the message is the one Pi actually raises.
 */
function staleCtx(): MaybeStaleContext {
  return {
    get hasUI(): boolean {
      throw new Error("This extension ctx is stale after session replacement or reload.");
    },
  };
}

/** A ctx from a live session, in the two shapes it comes in. */
function liveCtx(withUi: boolean): MaybeStaleContext {
  return { hasUI: withUi };
}

afterEach(() => {
  endSession();
});

describe("the session window", () => {
  it("is closed before the first session and open during one", () => {
    assert.equal(sessionAlive(), false);
    assert.equal(sessionSignal(), undefined);

    const alive = beginSession();

    assert.equal(sessionAlive(), true);
    assert.equal(alive.aborted, false);
    assert.equal(sessionSignal(), alive);
  });

  it("aborts the signal background work is holding when the session ends", () => {
    const alive = beginSession();
    endSession();

    assert.equal(alive.aborted, true);
    assert.equal(sessionAlive(), false);
  });

  it("ends the previous window when a session starts without a shutdown", () => {
    // `/reload` can land a second session_start before the first shutdown; work
    // from the old session must not be left holding a signal that never fires.
    const first = beginSession();
    const second = beginSession();

    assert.equal(first.aborted, true);
    assert.equal(second.aborted, false);
  });

  it("can be closed twice, and closed having never opened", () => {
    endSession();
    beginSession();
    endSession();
    endSession();

    assert.equal(sessionAlive(), false);
  });
});

describe("probing a ctx that may be stale", () => {
  it("reports a retired ctx as dead instead of throwing", () => {
    const ctx = staleCtx();

    assert.equal(ctxLive(ctx), false);
    assert.equal(hasUI(ctx), false);
  });

  it("passes a live ctx through unchanged", () => {
    assert.equal(ctxLive(liveCtx(true)), true);
    assert.equal(hasUI(liveCtx(true)), true);

    // A live print-mode ctx: alive, but nothing to draw on. The two answers
    // have to stay distinguishable — one drops the message, the other sends it
    // to stderr.
    assert.equal(ctxLive(liveCtx(false)), true);
    assert.equal(hasUI(liveCtx(false)), false);
  });
});
