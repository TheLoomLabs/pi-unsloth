/**
 * The version gate on sizing.
 *
 * Sizing is the one feature that cannot degrade gracefully: every number it
 * produces comes from `POST /api/inference/estimate-memory`, and a server too
 * old to offer that endpoint leaves exactly one alternative — hand-rolled GGUF
 * header parsing and a KV-bytes-per-element table. That is the code this
 * project exists to delete, so the answer is to **disable sizing and say so**,
 * naming the version required.
 *
 * Nothing else is gated. A pre-2026.9 server still lists models, loads, unloads
 * and reports its GPUs, and all of that keeps working.
 */

/**
 * The oldest Unsloth Studio whose estimator this extension will talk to.
 *
 * Not a fact about anyone's machine: it is the release the API reference was
 * verified against, written as the series rather than the patch so that any
 * 2026.9.x satisfies it.
 */
export const MIN_SIZING_VERSION = "2026.9";

/**
 * Split a version into numbers, stopping at the first part that is not one.
 *
 * Unsloth uses calendar versions (`2026.9.7`), but a pre-release or a local
 * build may append anything at all, so `2026.9.7rc1` compares as `2026.9.7`
 * rather than as nothing.
 */
export function parseVersion(version: string | undefined): number[] {
  if (typeof version !== "string") return [];
  const parts: number[] = [];
  for (const raw of version.trim().split(".")) {
    const match = /^\d+/.exec(raw.trim());
    if (!match) break;
    parts.push(Number(match[0]));
  }
  return parts;
}

/**
 * `-1` / `0` / `1`, comparing component-wise.
 *
 * A missing component counts as zero, so `2026.9` and `2026.9.0` are the same
 * version and `2026.9.7` is newer than both.
 */
export function compareVersions(left: string | undefined, right: string | undefined): number {
  const a = parseVersion(left);
  const b = parseVersion(right);
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const diff = (a[i] ?? 0) - (b[i] ?? 0);
    if (diff !== 0) return diff < 0 ? -1 : 1;
  }
  return 0;
}

export interface SizingSupport {
  enabled: boolean;
  /** Why not, in the words the UI shows. Absent when sizing is available. */
  reason: string | undefined;
  /** What the server said it is, when it said anything. */
  version: string | undefined;
}

/**
 * May this server be sized against?
 *
 * An **unreported** version is treated exactly like an old one, deliberately.
 * `versions.unsloth` is a field of a local, always-cheap endpoint; a server
 * that does not carry it is a server from before it existed, and guessing that
 * it has the estimator anyway is the one mistake the gate exists to prevent.
 * The two cases say different things, because they need different fixes.
 */
export function sizingSupport(version: string | undefined): SizingSupport {
  if (version === undefined || parseVersion(version).length === 0) {
    return {
      enabled: false,
      reason: `sizing needs Unsloth ≥ ${MIN_SIZING_VERSION}, and this server does not report a version`,
      version,
    };
  }
  if (compareVersions(version, MIN_SIZING_VERSION) < 0) {
    return {
      enabled: false,
      reason: `sizing needs Unsloth ≥ ${MIN_SIZING_VERSION} — this server reports ${version}`,
      version,
    };
  }
  return { enabled: true, reason: undefined, version };
}
