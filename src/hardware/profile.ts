/**
 * The machine profile — `~/.pi/agent/unsloth.json`.
 *
 * This file owns the whole cycle: read, migrate, write. Its job is to hold the
 * answers that cannot be measured again for free — which GPU drives the
 * monitor, how much headroom the user wants on each card — and to hold the
 * user's *corrections* of what was measured, which is what makes a wrong guess
 * recoverable rather than permanent.
 *
 * Three rules:
 *
 *   - **Never throw.** A profile that is absent, unreadable or not JSON leaves
 *     every caller on its default; the session must start either way.
 *   - **Never clobber what we do not understand.** The file is documented as
 *     hand-editable, so unknown keys — at the top level, inside `policy`, and
 *     inside each GPU entry — survive a write untouched.
 *   - **Write atomically.** A half-written profile read by the next session is
 *     worse than none at all, so the write goes to a temporary file and is
 *     renamed over the target.
 */

import { existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { getAgentDir } from "@earendil-works/pi-coding-agent";

import { sameEndpoint } from "../endpoint.ts";
import { type SamplingEntry, type SamplingStore, normaliseStore } from "../sampling.ts";
import { defaultHeadroomGb } from "./budget.ts";

/** The file, in the directory `getAgentDir` names. */
export const PROFILE_FILENAME = "unsloth.json";

/** Schema version written today. Bump only with a migration below. */
export const PROFILE_VERSION = 1;

export interface ProfileGpu {
  /** The id `/api/system` gives it, which is also the id `gpu_ids` accepts. */
  index: number;
  name?: string;
  totalGiB?: number;
  /** VRAM in use with nothing loaded — the display signal, as measured. */
  idleUsedGiB?: number;
  display: boolean;
  /** The one user-tunable number. */
  headroomGiB: number;
  /** Why we believe the display flag: `idle-vram`, `drm-connector:…`, `user`. */
  displayEvidence: string[];
  [key: string]: unknown;
}

export interface ProfileCalibration {
  estimateDeltaGiB?: number;
  samples?: number;
  [key: string]: unknown;
}

export interface ProfilePolicy {
  preferHeadless?: boolean;
  autoUnloadOnExit?: boolean;
  kvDtype?: string;
  ctxStepTokens?: number;
  [key: string]: unknown;
}

export interface MachineProfile {
  version: number;
  baseUrl?: string;
  detectedAt?: string;
  unslothVersion?: string;
  backend?: string;
  gpus: ProfileGpu[];
  calibration?: ProfileCalibration;
  policy?: ProfilePolicy;
  /**
   * Per-model sampling defaults, keyed by Pi's catalogue id (`src/sampling.ts`).
   *
   * Taste, not hardware: it rides beside `policy` rather than beside `gpus`,
   * and survives pointing the client at a different server for the same reason
   * a headroom preference does not.
   */
  sampling?: SamplingStore;
  /** Anything a hand edit or a later version added. Preserved verbatim. */
  [key: string]: unknown;
}

export function profilePath(): string {
  return join(getAgentDir(), PROFILE_FILENAME);
}

export function profileExists(): boolean {
  try {
    return existsSync(profilePath());
  } catch {
    return false;
  }
}

/**
 * Has the setup wizard actually run here?
 *
 * Not the same question as "does the file exist". The profile is also written
 * by things that are not the wizard — the footer toggle's one policy key, a
 * hand edit — and a file holding nothing but a preference is not a machine
 * that has been through setup. GPU data is what the wizard produces, and it
 * refuses to save without any, so its presence is the honest signal.
 */
export function setupHasRun(): boolean {
  if (!profileExists()) return false;
  return (readMachineProfile().profile?.gpus.length ?? 0) > 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function num(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function str(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() !== "" ? value.trim() : undefined;
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];
}

export interface ProfileRead {
  /** The profile as this version understands it. `undefined` when there is none. */
  profile: MachineProfile | undefined;
  /** The file existed but could not be used. The caller says so; nothing throws. */
  corrupt: boolean;
  /** The on-disk schema version differed and was brought forward in memory. */
  migrated: boolean;
}

/**
 * The raw object on disk, or `{}`.
 *
 * `src/settings.ts` reads its policy scalars through this so that there is one
 * reader of the file in the codebase rather than two that can disagree about
 * what "unreadable" means.
 */
export function readRawProfile(): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(readFileSync(profilePath(), "utf8"));
    if (isRecord(parsed)) return parsed;
  } catch {
    // Absent, unreadable, or not JSON — every caller has a default.
  }
  return {};
}

/**
 * Bring an on-disk object up to the current schema.
 *
 * Pure, so a migration is a test rather than a thing you find out about on
 * someone else's machine. Unknown keys ride along untouched; known ones are
 * normalised, and a field that is the wrong type is dropped rather than
 * repaired, because a repaired guess is indistinguishable from a measurement.
 */
export function migrateProfile(raw: Record<string, unknown>): MachineProfile {
  const gpusRaw = Array.isArray(raw["gpus"]) ? raw["gpus"] : [];
  const gpus: ProfileGpu[] = [];
  for (const entry of gpusRaw) {
    if (!isRecord(entry)) continue;
    const index = num(entry["index"]);
    if (index === undefined) continue;
    const display = entry["display"] === true;
    const gpu: ProfileGpu = {
      ...entry,
      index,
      display,
      headroomGiB: num(entry["headroomGiB"]) ?? defaultHeadroomGb(display),
      displayEvidence: strings(entry["displayEvidence"]),
    };
    // The spread carried every key across, hand-added ones included. A *known*
    // key of the wrong type is dropped rather than coerced: "lots" is not a
    // measurement, and a repaired guess is indistinguishable from one.
    const name = str(entry["name"]);
    if (name === undefined) delete gpu.name;
    else gpu.name = name;
    const total = num(entry["totalGiB"]);
    if (total === undefined) delete gpu.totalGiB;
    else gpu.totalGiB = total;
    const idle = num(entry["idleUsedGiB"]);
    if (idle === undefined) delete gpu.idleUsedGiB;
    else gpu.idleUsedGiB = idle;
    gpus.push(gpu);
  }

  const profile: MachineProfile = { ...raw, version: PROFILE_VERSION, gpus };
  for (const key of ["baseUrl", "detectedAt", "unslothVersion", "backend"] as const) {
    const value = str(raw[key]);
    if (value === undefined) delete profile[key];
    else profile[key] = value;
  }
  if (isRecord(raw["calibration"])) profile.calibration = raw["calibration"] as ProfileCalibration;
  else delete profile["calibration"];
  if (isRecord(raw["policy"])) profile.policy = raw["policy"] as ProfilePolicy;
  else delete profile["policy"];

  // Normalised rather than carried across verbatim: a sampler outside its own
  // range, or one the server does not model, would be merged into every
  // request body for that model and rejected there — a long way from the hand
  // edit that caused it.
  const sampling = normaliseStore(raw["sampling"]);
  if (Object.keys(sampling).length > 0) profile.sampling = sampling;
  else delete profile["sampling"];
  return profile;
}

/** Read and migrate. Never throws — a corrupt file is a state, not an error. */
export function readMachineProfile(): ProfileRead {
  let text: string;
  try {
    text = readFileSync(profilePath(), "utf8");
  } catch {
    return { profile: undefined, corrupt: false, migrated: false };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { profile: undefined, corrupt: true, migrated: false };
  }
  if (!isRecord(parsed)) return { profile: undefined, corrupt: true, migrated: false };

  const onDisk = num(parsed["version"]);
  return {
    profile: migrateProfile(parsed),
    corrupt: false,
    migrated: onDisk !== PROFILE_VERSION,
  };
}

/**
 * Store one policy flag, leaving everything else exactly as it is.
 *
 * Deliberately narrow. `writeMachineProfile` is the wizard's whole-profile
 * write; this is for a single preference the user flipped with a keystroke,
 * and it must not invent GPU data, a `detectedAt`, or anything else it did not
 * measure.
 */
export function rememberPolicy(key: string, value: unknown): WriteResult {
  const existing = readMachineProfile().profile;
  const policy: ProfilePolicy = { ...existing?.policy, [key]: value };
  return writeMachineProfile({
    ...(existing ?? { version: PROFILE_VERSION, gpus: [] }),
    policy,
  });
}

/**
 * Every model's stored sampling defaults. `{}` when there is no profile.
 *
 * Best-effort like every other read of this file: the catalogue is built on
 * every refresh and must never depend on a file being there.
 */
export function readSamplingStore(): SamplingStore {
  return normaliseStore(readRawProfile()["sampling"]);
}

/**
 * Store (or drop) one model's sampling defaults, leaving everything else alone.
 *
 * The narrow sibling of `rememberPolicy`, for the same reason: this is one
 * screen's decision about one model, and it must not invent GPU data or a
 * `detectedAt` it did not measure. Passing `undefined` removes the entry —
 * clearing every field is a real choice, and it means "let the server decide",
 * which is exactly an absent entry.
 */
export function rememberSampling(modelId: string, entry: SamplingEntry | undefined): WriteResult {
  const existing = readMachineProfile().profile;
  const sampling: SamplingStore = { ...existing?.sampling };
  if (entry) sampling[modelId] = entry;
  else delete sampling[modelId];

  // Written even when empty: `writeMachineProfile` merges over what is on
  // disk, so an omitted block would leave the old one there and the removal
  // would not stick.
  return writeMachineProfile({ ...(existing ?? { version: PROFILE_VERSION, gpus: [] }), sampling });
}

/**
 * The profile as it applies to `baseUrl`.
 *
 * GPU entries describe one machine: indices, display flags, idle readings and
 * the headroom the user set on each card. Point the extension at a different
 * server and every one of those is about hardware that is no longer being
 * talked to — and a 3 GiB headroom set for a 24 GiB card, applied to a 12 GiB
 * one, is the kind of quiet wrong answer that ends in an OOM. So the GPU list
 * is **discarded rather than reused**, and the wizard opens on freshly detected
 * hardware.
 *
 * `policy`, `calibration` and `sampling` survive: a preference is the user's,
 * not the machine's, calibration carries the server version it was measured
 * against, which is what ages *it* out, and a model's sampling defaults are a
 * property of the model, which is the same model on either server.
 *
 * A profile with no `baseUrl` recorded is from before the endpoint was written
 * down, when there was only ever one. It is taken at face value — inventing a
 * mismatch would throw away good data on every upgrade.
 */
export function profileForEndpoint(
  profile: MachineProfile | undefined,
  baseUrl: string,
): MachineProfile | undefined {
  if (!profile) return undefined;
  if (profile.baseUrl === undefined) return profile;
  if (sameEndpoint(profile.baseUrl, baseUrl)) return profile;
  return { ...profile, gpus: [] };
}

export interface WriteResult {
  ok: boolean;
  path: string;
  error: string | undefined;
}

/**
 * Write the profile, preserving anything on disk this version does not know.
 *
 * The merge is deliberate: the file is documented as hand-editable, and a user
 * who added a key is owed it back. Only what the wizard actually collected is
 * overwritten.
 */
export function writeMachineProfile(profile: MachineProfile): WriteResult {
  const path = profilePath();
  const merged: MachineProfile = { ...readRawProfile(), ...profile, version: PROFILE_VERSION };
  const temporary = `${path}.tmp`;
  try {
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(temporary, `${JSON.stringify(merged, undefined, 2)}\n`, "utf8");
    renameSync(temporary, path);
    return { ok: true, path, error: undefined };
  } catch (error) {
    try {
      unlinkSync(temporary);
    } catch {
      // Nothing to clean up, or nothing we can do about it.
    }
    return { ok: false, path, error: error instanceof Error ? error.message : String(error) };
  }
}
