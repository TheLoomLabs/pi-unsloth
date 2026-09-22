/**
 * The one blocking moment in the extension: the first prompt against a model
 * the server has not loaded yet.
 *
 * Rules it exists to keep:
 *   - never wait when there is nothing to wait for — a ready model must not
 *     cost a visible overlay flash;
 *   - always escapable;
 *   - never draw anything outside the TUI.
 *
 * Escape abandons the *wait*, not the load. Killing a load halfway through
 * would leave the GPUs occupied by a model nobody can use; letting it finish in
 * the background means the next prompt is fast, and this one goes to Unsloth's
 * auto-switch, which is exactly what it is for.
 */

import { BorderedLoader } from "@earendil-works/pi-coding-agent";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { cancelEnsure } from "../supervisor.ts";

/**
 * How long a load may take before it is worth drawing anything. An already
 * resident model settles well inside this, so the common case shows nothing.
 */
const GRACE_MS = 300;

export type WaitOutcome = "ready" | "cancelled";

/** Resolve `true` if `promise` settles within `ms`. */
function settlesWithin(promise: Promise<unknown>, ms: number): Promise<boolean> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(false), ms);
    timer.unref?.();
    promise.then(
      () => {
        clearTimeout(timer);
        resolve(true);
      },
      () => {
        clearTimeout(timer);
        resolve(true);
      },
    );
  });
}

/**
 * Wait for an in-flight ensure, showing an escapable loader if it is slow.
 *
 * Returns `"cancelled"` only when the user pressed escape; the turn continues
 * either way, because `before_agent_start` cannot refuse a prompt.
 */
export async function waitForModel(
  ctx: ExtensionContext,
  label: string,
  promise: Promise<void>,
): Promise<WaitOutcome> {
  if (await settlesWithin(promise, GRACE_MS)) return "ready";

  if (ctx.mode !== "tui") {
    // No overlays outside the TUI: one line on stderr keeps `pi -p` pipelines
    // clean while still explaining the pause.
    process.stderr.write(`unsloth: waiting for ${label} to load…\n`);
    await promise.catch(() => {});
    return "ready";
  }

  const outcome = await ctx.ui.custom<WaitOutcome>((tui, theme, _keybindings, done) => {
    const loader = new BorderedLoader(tui, theme, `Loading ${label} — esc to continue without waiting`);
    loader.onAbort = () => done("cancelled");
    promise.then(
      () => done("ready"),
      () => done("ready"),
    );
    return loader;
  });

  if (outcome === "cancelled") {
    cancelEnsure();
    ctx.ui.notify("Continuing without waiting — Unsloth loads on demand", "warning");
  }
  return outcome;
}
