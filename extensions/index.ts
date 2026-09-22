/**
 * pi-unsloth — Unsloth Studio as a Pi provider.
 *
 * This file is only the wiring: registration, and nothing else. No network
 * calls, timers or other background work run here, because an extension
 * factory also runs in invocations that never start a session
 * (pi docs, extensions.md → "Long-lived resources and shutdown").
 */

import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { UnslothClient, checkHealth } from "../src/api/client.ts";
import { getServerVersions } from "../src/api/system.ts";
import { sizingSupport } from "../src/hardware/version.ts";
import { createUnslothProvider } from "../src/provider.ts";
import {
  ensureServer,
  isOurModel,
  pendingEnsure,
  refreshCatalogue,
  report,
  resetSupervisor,
  shortLabel,
  startEnsure,
  unloadAndReport,
} from "../src/supervisor.ts";
import { autoUnloadOnExit, footerEnabled } from "../src/settings.ts";
import { rememberPolicy } from "../src/hardware/profile.ts";
import { beginSession, endSession } from "../src/session.ts";
import { resetState, state } from "../src/state.ts";
import { footerTarget, paint, setFooterVisible, startFooter, stopFooter } from "../src/ui/footer.ts";
import { openPanel } from "../src/ui/panel.ts";
import { samplingByName } from "../src/ui/sampling.ts";
import { sizeByName } from "../src/ui/sizer.ts";
import { offerSetup, runWizard } from "../src/ui/wizard.ts";
import { waitForModel } from "../src/ui/waiting.ts";

export default function (pi: ExtensionAPI): void {
  pi.registerProvider(createUnslothProvider());

  pi.registerCommand("unsloth", {
    description:
      "Unsloth Studio panel; `add <model>` sizes a model, `sampling <model>` sets its sampling defaults, `setup` re-runs the wizard, `footer` shows or hides the status line, `off` frees the GPUs, `status` prints one line",
    handler: async (args: string, ctx: ExtensionCommandContext) => {
      const trimmed = args.trim();
      // `add` and `footer` are the subcommands that take an argument, so they
      // are matched on the first word rather than on the whole string.
      const [word, ...rest] = trimmed.split(/\s+/);
      const head = (word ?? "").toLowerCase();
      if (head === "add") {
        await sizeByName(ctx, rest.join(" "));
        return;
      }
      if (head === "sampling") {
        await samplingByName(ctx, rest.join(" "));
        return;
      }
      if (head === "footer") {
        const target = footerTarget(rest[0], state.footerVisible);
        if (target === undefined) {
          report(ctx, "Usage: /unsloth footer [on|off]", "warning");
          return;
        }
        applyFooter(ctx, target);
        return;
      }

      switch (trimmed.toLowerCase()) {
        case "":
          await openPanel(ctx);
          return;
        case "off":
          await unloadAndReport(ctx);
          return;
        case "status":
          await status(ctx);
          return;
        case "setup":
          await runWizard(ctx);
          return;
        default:
          report(ctx, "Usage: /unsloth [add <model>|sampling <model>|setup|footer [on|off]|off|status]", "warning");
      }
    },
  });

  pi.registerShortcut("ctrl+alt+u", {
    description: "Unsloth: open the panel",
    handler: async (ctx: ExtensionContext) => {
      await openPanel(ctx);
    },
  });

  pi.registerShortcut("ctrl+alt+f", {
    description: "Unsloth: show or hide the status line",
    handler: (ctx: ExtensionContext) => {
      applyFooter(ctx, !state.footerVisible);
    },
  });

  pi.registerShortcut("ctrl+alt+o", {
    description: "Unsloth: unload and free the GPUs",
    handler: async (ctx: ExtensionContext) => {
      await unloadAndReport(ctx);
    },
  });

  // Background work starts here, never in the factory.
  pi.on("session_start", async (_event, ctx) => {
    // Read once, here: `state` is the session's answer from now on, so the
    // toggle takes effect on the keystroke rather than on the next read.
    state.footerVisible = footerEnabled();
    // Opens the window this session's background work runs inside.
    // `session_shutdown` closes it, and the chain below stops at its next
    // await rather than carrying on against a ctx Pi has already retired.
    const alive = beginSession();
    startFooter(ctx);
    // Not awaited: startup can take a minute and the editor must stay usable
    // throughout. The footer is the progress report.
    void (async () => {
      await ensureServer(ctx, { signal: alive });
      if (alive.aborted) return;
      await refreshCatalogue(ctx, { signal: alive });
      if (alive.aborted) return;
      // The session may already have a model — chosen with `--model`, or
      // restored with the session — and `model_select` does not always fire
      // for those. It is ours to keep ready exactly as a switch would be.
      const model = ctx.model;
      if (model && isOurModel(model)) startEnsure(ctx, model);
      else paint(ctx);
      if (alive.aborted) return;
      // Last, and after the ensure is already running in the background: the
      // wizard owns the input while it is open, so nothing may wait behind it.
      await offerSetup(ctx);
    })().catch((error: unknown) => {
      // The backstop, not the fix: every step above already handles its own
      // failures, and the abort checks are what keep this chain off a retired
      // ctx. What is left is the race no check can close — the session ending
      // between one line and the next — and an unhandled rejection here would
      // take the whole editor down with it.
      report(ctx, `Unsloth startup stopped: ${error instanceof Error ? error.message : String(error)}`, "warning");
    });
  });

  // Every model switch, from every source: auto-switch would silently drop
  // speculative decoding, so the tuned settings are re-applied explicitly.
  pi.on("model_select", async (event, ctx) => {
    if (!isOurModel(event.model)) {
      // Switching away is not an unload — another window may still want the
      // model — but the footer must stop claiming this one is ours.
      state.activeModelId = undefined;
      state.activeModelLabel = undefined;
      state.activeContextWindow = undefined;
      paint(ctx);
      return;
    }
    startEnsure(ctx, event.model);
  });

  // The one place allowed to wait — and only when the answer would otherwise
  // be a request to a server that has not loaded the model yet.
  pi.on("before_agent_start", async (_event, ctx) => {
    const model = ctx.model;
    if (!isOurModel(model) || !model) return;
    if (!pendingEnsure(model.id)) startEnsure(ctx, model);
    const promise = pendingEnsure(model.id);
    if (!promise) return;
    await waitForModel(ctx, shortLabel(model), promise);
  });

  // Pause the VRAM poll while a turn streams.
  pi.on("agent_start", async (_event, ctx) => {
    state.streaming = true;
    paint(ctx);
  });

  pi.on("agent_settled", async (_event, ctx) => {
    state.streaming = false;
    paint(ctx);
  });

  // Idempotent, and the only place timers are cleared.
  pi.on("session_shutdown", async (event, ctx) => {
    // First, and before the await below: this is what stops the startup chain
    // and anything else holding this session's signal, so nothing new is
    // started against a ctx that is about to be retired.
    endSession();
    // Only a real exit frees the GPUs: `/new` and `/reload` also land here, and
    // unloading on those would cost a reload the user never asked for.
    if (event.reason === "quit" && autoUnloadOnExit()) {
      await unloadAndReport(ctx, { quiet: true });
    }
    stopFooter(ctx);
    resetSupervisor();
    resetState();
  });
}

/** `/unsloth status` — one line about the server and the catalogue. */
/**
 * Show or hide the status line, and remember which.
 *
 * The preference is stored rather than merely applied: a user who turned the
 * footer off did not mean "until I restart Pi". A profile that cannot be
 * written is not a failure worth stopping for — the toggle still took effect,
 * and the sentence says which half worked.
 */
function applyFooter(ctx: ExtensionContext, visible: boolean): void {
  setFooterVisible(ctx, visible);

  const written = rememberPolicy("footer", visible);
  const where = written.ok ? "" : ` — could not save the preference: ${written.error ?? "unknown error"}`;
  report(
    ctx,
    visible ? `✓ Unsloth footer on${where}` : `○ Unsloth footer off — ctrl+alt+f brings it back${where}`,
    written.ok ? "info" : "warning",
  );
}

async function status(ctx: ExtensionCommandContext): Promise<void> {
  const client = new UnslothClient();
  const health = await checkHealth(client);
  const catalogue =
    state.catalogueSize > 0
      ? ` · ${state.catalogueSize} model${state.catalogueSize === 1 ? "" : "s"}${
          state.catalogueFromCache ? " (cached)" : ""
        }`
      : "";

  switch (health.state) {
    case "up": {
      // The sizing gate is a property of the server, so the one line about the
      // server is where it belongs — the alternative is finding out inside the
      // wizard.
      const versions = await getServerVersions(client).catch(() => undefined);
      const sizing = sizingSupport(versions?.unsloth);
      const gate = sizing.enabled ? "" : ` · ⚠ ${sizing.reason}`;
      report(ctx, `✓ Unsloth up at ${client.baseUrl} (${health.latencyMs} ms)${catalogue}${gate}`, "info");
      return;
    }
    case "unauthorized":
      report(ctx, `⚠ Unsloth at ${client.baseUrl} rejected the API key — run /login`, "warning");
      return;
    case "down":
      report(
        ctx,
        `◌ Unsloth offline at ${client.baseUrl}${health.detail ? ` — ${health.detail}` : ""}${catalogue}`,
        "warning",
      );
      return;
  }
}
