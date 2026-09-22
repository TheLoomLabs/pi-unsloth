import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";

import { report } from "../src/supervisor.ts";
import { resetState } from "../src/state.ts";
import { footerRunning, paint, startFooter, stopFooter, type FooterContext } from "../src/ui/footer.ts";

/**
 * The regression these guard.
 *
 * Background work outlives the session that started it — the startup chain, the
 * 4 s tick, an ensure still loading a model. Pi retires the ctx it was handed
 * on `/new`, `/reload`, a fork, or the end of a `-p` run, and from then on
 * every property access on it throws. Reaching one of those reads from a
 * floating promise is an unhandled rejection, and Node exits the editor.
 *
 * So: no code path that a timer or a detached promise can reach may read a ctx
 * raw. These tests drive the sites that can, with a ctx that throws the way a
 * retired one does.
 */
function staleFooterContext(): FooterContext {
  const boom = (): never => {
    throw new Error("This extension ctx is stale after session replacement or reload.");
  };
  return {
    get hasUI(): boolean {
      return boom();
    },
    get ui(): never {
      return boom();
    },
    get modelRegistry(): never {
      return boom();
    },
  } as unknown as FooterContext;
}

/** A live print-mode ctx, to prove the guards did not flatten the other answer. */
function printModeContext(): { ctx: never; errors: string[] } {
  return { ctx: { hasUI: false } as never, errors: [] };
}

afterEach(() => {
  stopFooter({ hasUI: false } as unknown as FooterContext);
  resetState();
});

describe("a retired ctx reaching the footer", () => {
  it("makes paint a skipped repaint, not a throw", () => {
    assert.doesNotThrow(() => paint(staleFooterContext()));
  });

  it("makes startFooter a no-op rather than starting a tick", () => {
    assert.doesNotThrow(() => startFooter(staleFooterContext()));
    assert.equal(footerRunning(), false);
  });

  it("lets stopFooter still clear the timers it owns", () => {
    // The shutdown path itself can arrive holding a ctx Pi has already retired;
    // it has no line left to clear, but the tick is still ours to stop.
    assert.doesNotThrow(() => stopFooter(staleFooterContext()));
    assert.equal(footerRunning(), false);
  });
});

describe("a retired ctx reaching report()", () => {
  it("drops the line instead of throwing or printing", () => {
    const written: string[] = [];
    const original = process.stderr.write.bind(process.stderr);
    process.stderr.write = ((chunk: string) => {
      written.push(String(chunk));
      return true;
    }) as typeof process.stderr.write;

    try {
      assert.doesNotThrow(() => report(staleFooterContext() as never, "server is up", "info"));
    } finally {
      process.stderr.write = original;
    }

    // Nothing is left to tell: the session that would have shown this is gone,
    // and stderr in a live TUI prints straight through the render.
    assert.deepEqual(written, []);
  });

  it("still sends a live print-mode line to stderr", () => {
    const { ctx } = printModeContext();
    const written: string[] = [];
    const original = process.stderr.write.bind(process.stderr);
    process.stderr.write = ((chunk: string) => {
      written.push(String(chunk));
      return true;
    }) as typeof process.stderr.write;

    try {
      report(ctx, "server is up", "info");
    } finally {
      process.stderr.write = original;
    }

    assert.deepEqual(written, ["server is up\n"]);
  });
});
