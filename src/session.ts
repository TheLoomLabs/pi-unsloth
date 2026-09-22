/**
 * How long a `ctx` is worth touching.
 *
 * Pi hands every handler an `ExtensionContext` bound to the session that was
 * live when the handler fired. Once that session is replaced — `/new`,
 * `/reload`, a fork, or simply the end of a `-p` run — *every* property on the
 * old ctx throws, including `hasUI`. A handler that has already returned cannot
 * be hurt by that, but background work started from one can: it is still
 * holding the ctx it captured, and its next paint is a throw inside a floating
 * promise, which Node turns into an unhandled rejection and the process exits.
 *
 * So the rule this module exists to enforce: **background work touches a ctx
 * only while the session it came from is still the live one.** Two halves:
 *
 *   - `beginSession` / `endSession` mark the window, and the signal in between
 *     is what long chains check across their awaits and pass to their fetches,
 *     so the work *stops* rather than merely failing quietly;
 *   - `ctxLive` / `hasUI` make the reads themselves total, because no abort
 *     check can close the gap between a timer firing and the ctx it captured
 *     going stale one tick later.
 *
 * The probes cost a try/catch around a property read. That is the only way to
 * ask: Pi exposes no "is this still valid" flag, and the answer arrives as an
 * exception.
 */

/** The live session's lifetime, or `undefined` between sessions. */
let lifetime: AbortController | undefined;

/**
 * Open a session window and return the signal that closes with it.
 *
 * Idempotent per session in the sense that matters: a second `session_start`
 * without an intervening shutdown ends the previous window first, so work from
 * the old one cannot outlive it holding a signal that never aborts.
 */
export function beginSession(): AbortSignal {
  endSession();
  lifetime = new AbortController();
  return lifetime.signal;
}

/**
 * Close the window. Idempotent, and safe before any session has started —
 * `session_shutdown` fires for sessions that never got as far as `session_start`.
 */
export function endSession(): void {
  lifetime?.abort();
  lifetime = undefined;
}

/** Is a session window open right now? */
export function sessionAlive(): boolean {
  return lifetime !== undefined && !lifetime.signal.aborted;
}

/** The live session's signal, for work that wants to be cancelled with it. */
export function sessionSignal(): AbortSignal | undefined {
  return lifetime?.signal;
}

/** The narrowest shape the probes need, so tests can pass a fake that throws. */
export interface MaybeStaleContext {
  readonly hasUI: boolean;
}

/**
 * Is this ctx still attached to a live session?
 *
 * Ask before *any* use of a ctx that was captured earlier — a timer callback, a
 * `.then`, anything after an await. A dead ctx is not an error to report: the
 * session it belonged to is gone and there is nobody left to tell.
 */
export function ctxLive(ctx: MaybeStaleContext): boolean {
  try {
    void ctx.hasUI;
    return true;
  } catch {
    return false;
  }
}

/**
 * `ctx.hasUI`, answering `false` instead of throwing when the ctx is stale.
 *
 * Every caller of `hasUI` already means "may I draw?", and a stale ctx's honest
 * answer to that is no. Using this in place of the raw property turns the whole
 * class of late-paint crashes into a skipped repaint.
 */
export function hasUI(ctx: MaybeStaleContext): boolean {
  try {
    return ctx.hasUI;
  } catch {
    return false;
  }
}
