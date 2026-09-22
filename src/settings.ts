/**
 * The policies the extension reads out of `~/.pi/agent/unsloth.json`, plus
 * their environment overrides.
 *
 * `src/hardware/profile.ts` owns that file — its schema, its migration and the
 * only code that writes it. Everything here is a best-effort *read* of a
 * handful of scalars: an absent, unreadable or corrupt profile must leave every
 * policy at its default rather than break the session, so nothing below throws.
 *
 * Resolution order for every policy: environment → profile → default.
 */

import { resolveEndpoint } from "./api/client.ts";
import { sameEndpoint } from "./endpoint.ts";
import { readRawProfile } from "./hardware/profile.ts";

export { PROFILE_FILENAME } from "./hardware/profile.ts";

/** The binary Unsloth Studio installs. Overridable — see `launchCommand()`. */
const DEFAULT_LAUNCH_COMMAND = ["unsloth-studio"] as const;

/**
 * Read the profile as a plain object. `{}` for anything that is not one.
 *
 * Delegated to `src/hardware/profile.ts`, which owns the file: one reader means
 * "absent", "unreadable" and "not JSON" cannot come to mean different things in
 * two places.
 */
export function readProfile(): Record<string, unknown> {
  return readRawProfile();
}

/**
 * A scalar from the profile, looked for at the top level and under `policy`.
 *
 * The documented schema groups the policies under `policy`; the flat spelling
 * is what M2 shipped and what an existing hand-written file may hold. Both are
 * read, the top level first, so neither breaks.
 */
function profileValue(key: string): unknown {
  const profile = readProfile();
  if (profile[key] !== undefined) return profile[key];
  const policy = profile["policy"];
  if (policy && typeof policy === "object" && !Array.isArray(policy)) {
    return (policy as Record<string, unknown>)[key];
  }
  return undefined;
}

/**
 * Parse an environment variable as a boolean.
 *
 * Only the unambiguous spellings count. Anything else — including the empty
 * string — is treated as "not set", so `UNSLOTH_X=` never silently means false.
 */
export function envBoolean(name: string): boolean | undefined {
  const raw = process.env[name]?.trim().toLowerCase();
  if (raw === undefined || raw === "") return undefined;
  if (["1", "true", "yes", "on"].includes(raw)) return true;
  if (["0", "false", "no", "off"].includes(raw)) return false;
  return undefined;
}

/** Base URL of the Unsloth Studio server, when the profile names one. */
export function profileBaseUrl(): string | undefined {
  const value = profileValue("baseUrl");
  return typeof value === "string" && value.trim() !== "" ? value.trim() : undefined;
}

/**
 * May the extension start Unsloth Studio itself? Default **on** — autostart is
 * the feature that replaces `pil`.
 */
export function autostartEnabled(): boolean {
  return envBoolean("UNSLOTH_AUTOSTART") ?? asBoolean(profileValue("autostart")) ?? true;
}

/**
 * Free the GPUs when Pi exits? Default **off**: a second Pi window, or anything
 * else using the server, would lose its model without having asked for that.
 */
export function autoUnloadOnExit(): boolean {
  return envBoolean("UNSLOTH_AUTO_UNLOAD_ON_EXIT") ?? asBoolean(profileValue("autoUnloadOnExit")) ?? false;
}

/**
 * Draw the status line? Default **on** — it is the thing most of this UI is
 * for, and a user who disagrees says so once.
 *
 * `off` costs the 4 s VRAM poll as well as the line, so it is worth having for
 * a slow link or a narrow terminal, not only for taste.
 */
export function footerEnabled(): boolean {
  return envBoolean("UNSLOTH_FOOTER") ?? asBoolean(profileValue("footer")) ?? true;
}

/**
 * How to start the server.
 *
 * Accepts a string (split on whitespace) or an argv array, from either the
 * environment or the profile, so a user whose install is not on `PATH` — or who
 * launches it through a wrapper — is not stuck with the default.
 */
export function launchCommand(): string[] {
  const fromEnv = commandFrom(process.env["UNSLOTH_LAUNCH_COMMAND"]);
  if (fromEnv) return fromEnv;
  const fromProfile = commandFrom(profileValue("launchCommand"));
  if (fromProfile) return fromProfile;
  return [...DEFAULT_LAUNCH_COMMAND];
}

function commandFrom(value: unknown): string[] | undefined {
  if (typeof value === "string") {
    const parts = value.trim().split(/\s+/).filter((part) => part !== "");
    return parts.length > 0 ? parts : undefined;
  }
  if (Array.isArray(value)) {
    const parts = value.filter((part): part is string => typeof part === "string" && part.trim() !== "");
    return parts.length > 0 ? parts : undefined;
  }
  return undefined;
}

function asBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

/**
 * Does the profile's *hardware* describe the server currently configured?
 *
 * Policies are the user's and travel with them; GPU entries are one machine's,
 * and reading another server's display flags would tag the wrong card in the
 * panel. A profile written before the endpoint was recorded states nothing to
 * contradict, so it is taken at face value.
 */
export function profileHardwareApplies(): boolean {
  const recorded = profileBaseUrl();
  if (recorded === undefined) return true;
  return sameEndpoint(recorded, resolveEndpoint().baseUrl);
}

/**
 * GPUs the profile says drive a display, or `undefined` when it does not say.
 *
 * `src/hardware/profile.ts` (M4) owns writing this; the wizard's `d` key is
 * what puts it there. Reading it here — best-effort, like everything else in
 * this file — means a user correction is honoured by the panel as soon as it
 * exists, without the panel having to know about the wizard.
 *
 * The distinction between "no profile" and "a profile that flags nothing"
 * matters: a headless box legitimately has zero display GPUs, and that is an
 * answer, not a missing one.
 */
export function profileDisplayGpus(): number[] | undefined {
  if (!profileHardwareApplies()) return undefined;
  const gpus = profileValue("gpus");
  if (!Array.isArray(gpus)) return undefined;

  const flagged: number[] = [];
  let stated = false;
  for (const entry of gpus) {
    if (typeof entry !== "object" || entry === null || Array.isArray(entry)) continue;
    const record = entry as Record<string, unknown>;
    const index = record["index"];
    const display = record["display"];
    if (typeof display !== "boolean" || typeof index !== "number" || !Number.isFinite(index)) continue;
    stated = true;
    if (display) flagged.push(index);
  }
  return stated ? flagged : undefined;
}
