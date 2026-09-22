import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, afterEach, before, beforeEach, describe, it } from "node:test";

import {
  PROFILE_FILENAME,
  PROFILE_VERSION,
  migrateProfile,
  profileExists,
  profileForEndpoint,
  profilePath,
  readMachineProfile,
  readRawProfile,
  rememberPolicy,
  setupHasRun,
  writeMachineProfile,
  type MachineProfile,
} from "../src/hardware/profile.ts";
import { DISPLAY_HEADROOM_GB, HEADLESS_HEADROOM_GB } from "../src/hardware/budget.ts";
import { autoUnloadOnExit, footerEnabled } from "../src/settings.ts";

const AGENT_DIR_ENV = "PI_CODING_AGENT_DIR";

let dir: string;
let saved: string | undefined;

function write(contents: string): void {
  writeFileSync(join(dir, PROFILE_FILENAME), contents);
}

before(() => {
  saved = process.env[AGENT_DIR_ENV];
  dir = mkdtempSync(join(tmpdir(), "pi-unsloth-profile-"));
});

beforeEach(() => {
  process.env[AGENT_DIR_ENV] = dir;
});

afterEach(() => {
  rmSync(join(dir, PROFILE_FILENAME), { force: true });
});

after(() => {
  if (saved === undefined) delete process.env[AGENT_DIR_ENV];
  else process.env[AGENT_DIR_ENV] = saved;
  rmSync(dir, { recursive: true, force: true });
});

/** The profile documents, as the wizard would write it. */
const REFERENCE: MachineProfile = {
  version: PROFILE_VERSION,
  baseUrl: "http://127.0.0.1:8888",
  detectedAt: "2026-09-21T12:00:00.000Z",
  unslothVersion: "2026.9.7",
  backend: "rocm",
  gpus: [
    {
      index: 0,
      name: "AMD Radeon Graphics",
      totalGiB: 23.98,
      idleUsedGiB: 1.21,
      display: true,
      headroomGiB: DISPLAY_HEADROOM_GB,
      displayEvidence: ["idle-vram", "drm-connector:card1-DP-3"],
    },
    {
      index: 1,
      name: "AMD Radeon Graphics",
      totalGiB: 23.98,
      idleUsedGiB: 0.03,
      display: false,
      headroomGiB: HEADLESS_HEADROOM_GB,
      displayEvidence: [],
    },
  ],
};

describe("readMachineProfile", () => {
  it("reports no profile, and no corruption, when there is no file", () => {
    assert.deepEqual(readMachineProfile(), { profile: undefined, corrupt: false, migrated: false });
    assert.equal(profileExists(), false);
  });

  it("survives a corrupt file without throwing, and says it is corrupt", () => {
    write("{ this is not json");
    const read = readMachineProfile();
    assert.equal(read.profile, undefined);
    assert.equal(read.corrupt, true);
    // The whole point: a session must still start, and every policy falls back.
    assert.equal(autoUnloadOnExit(), false);
    assert.deepEqual(readRawProfile(), {});
  });

  it("treats a JSON array as corrupt rather than as an empty profile", () => {
    write("[1, 2, 3]");
    assert.equal(readMachineProfile().corrupt, true);
  });

  it("round-trips what the wizard writes", () => {
    assert.equal(writeMachineProfile(REFERENCE).ok, true);
    const read = readMachineProfile();
    assert.equal(read.corrupt, false);
    assert.equal(read.migrated, false);
    assert.deepEqual(read.profile?.gpus, REFERENCE.gpus);
    assert.equal(read.profile?.backend, "rocm");
  });
});

describe("migrateProfile", () => {
  it("brings a file with no version forward, filling the headroom defaults", () => {
    const migrated = migrateProfile({ gpus: [{ index: 0, display: true }, { index: 1, display: false }] });
    assert.equal(migrated.version, PROFILE_VERSION);
    assert.equal(migrated.gpus[0]?.headroomGiB, DISPLAY_HEADROOM_GB);
    assert.equal(migrated.gpus[1]?.headroomGiB, HEADLESS_HEADROOM_GB);
    assert.deepEqual(migrated.gpus[0]?.displayEvidence, []);
  });

  it("drops a GPU entry with no index — there is nothing to apply it to", () => {
    const migrated = migrateProfile({ gpus: [{ display: true }, "nonsense", { index: 2, display: true }] });
    assert.deepEqual(migrated.gpus.map((gpu) => gpu.index), [2]);
  });

  it("drops a field of the wrong type rather than repairing it into a fact", () => {
    const migrated = migrateProfile({ backend: 7, gpus: [{ index: 0, display: false, totalGiB: "lots" }] });
    assert.equal(migrated.backend, undefined);
    assert.equal(migrated.gpus[0]?.totalGiB, undefined);
  });

  it("keeps hand-added keys, at the top level and inside a GPU entry", () => {
    const migrated = migrateProfile({
      note: "edited by hand",
      policy: { preferHeadless: false, custom: 1 },
      gpus: [{ index: 0, display: false, mine: "keep me" }],
    });
    assert.equal(migrated["note"], "edited by hand");
    assert.equal(migrated.policy?.["custom"], 1);
    assert.equal(migrated.gpus[0]?.["mine"], "keep me");
  });

  it("says it migrated when the on-disk version is not the current one", () => {
    write(JSON.stringify({ version: 0, gpus: [{ index: 0, display: true }] }));
    const read = readMachineProfile();
    assert.equal(read.migrated, true);
    assert.equal(read.profile?.version, PROFILE_VERSION);
  });
});

describe("writeMachineProfile", () => {
  it("preserves keys already on disk that this version knows nothing about", () => {
    write(JSON.stringify({ note: "mine", policy: { kvDtype: "q8_0" }, gpus: [] }));
    assert.equal(writeMachineProfile({ version: PROFILE_VERSION, gpus: REFERENCE.gpus }).ok, true);
    const written = JSON.parse(readFileSync(profilePath(), "utf8")) as Record<string, unknown>;
    assert.equal(written["note"], "mine");
    assert.deepEqual(written["policy"], { kvDtype: "q8_0" });
    assert.equal((written["gpus"] as unknown[]).length, 2);
  });

  it("writes JSON a human can edit, which is what the file is for", () => {
    writeMachineProfile(REFERENCE);
    const text = readFileSync(profilePath(), "utf8");
    assert.ok(text.includes("\n  \"gpus\""), text.slice(0, 80));
    assert.ok(text.endsWith("\n"));
  });

  it("reports a write it could not do instead of throwing into the session", () => {
    const readOnly = mkdtempSync(join(tmpdir(), "pi-unsloth-ro-"));
    process.env[AGENT_DIR_ENV] = readOnly;
    chmodSync(readOnly, 0o500);
    try {
      const result = writeMachineProfile(REFERENCE);
      assert.equal(result.ok, false);
      assert.ok(result.error);
    } finally {
      chmodSync(readOnly, 0o700);
      rmSync(readOnly, { recursive: true, force: true });
      process.env[AGENT_DIR_ENV] = dir;
    }
  });
});

describe("profileForEndpoint", () => {
  const profile = {
    version: 1,
    baseUrl: "http://127.0.0.1:8888",
    gpus: [{ index: 0, display: true, headroomGiB: 3, displayEvidence: ["idle-vram"] }],
    policy: { autoUnloadOnExit: true },
    calibration: { estimateDeltaGiB: 0.62, samples: 2 },
  };

  it("keeps everything while the endpoint is the one it was detected against", () => {
    const applied = profileForEndpoint(profile, "http://127.0.0.1:8888/");
    assert.equal(applied?.gpus.length, 1);
  });

  it("discards the GPUs when the endpoint is a different machine", () => {
    // They describe hardware that is no longer being talked to. A headroom set
    // for a 24 GiB card, applied to a 12 GiB one, is an OOM waiting to happen.
    const applied = profileForEndpoint(profile, "http://192.168.1.40:8888");
    assert.deepEqual(applied?.gpus, []);
  });

  it("keeps the policy and the calibration, which are not about the machine", () => {
    const applied = profileForEndpoint(profile, "http://192.168.1.40:8888");
    assert.deepEqual(applied?.policy, { autoUnloadOnExit: true });
    assert.equal(applied?.calibration?.estimateDeltaGiB, 0.62);
  });

  it("trusts a profile written before the endpoint was recorded", () => {
    const old = { version: 1, gpus: profile.gpus };
    assert.equal(profileForEndpoint(old, "http://192.168.1.40:8888")?.gpus.length, 1);
  });

  it("has nothing to say about a profile that does not exist", () => {
    assert.equal(profileForEndpoint(undefined, "http://127.0.0.1:8888"), undefined);
  });
});

describe("rememberPolicy", () => {
  it("creates a profile holding just the preference, and nothing invented", () => {
    const result = rememberPolicy("footer", false);
    assert.equal(result.ok, true);
    const written = JSON.parse(readFileSync(profilePath(), "utf8")) as MachineProfile;
    assert.equal(written.policy?.["footer"], false);
    // No measurement it never took: no detectedAt, no GPU entries.
    assert.deepEqual(written.gpus, []);
    assert.equal(written.detectedAt, undefined);
    assert.equal(footerEnabled(), false);
  });

  it("leaves every other policy, and the GPUs, exactly as they were", () => {
    write(
      JSON.stringify({
        version: PROFILE_VERSION,
        gpus: [{ index: 0, display: true, headroomGiB: 3, displayEvidence: ["user"] }],
        policy: { autoUnloadOnExit: true, kvDtype: "q8_0" },
      }),
    );

    rememberPolicy("footer", false);

    const written = JSON.parse(readFileSync(profilePath(), "utf8")) as MachineProfile;
    assert.equal(written.policy?.autoUnloadOnExit, true);
    assert.equal(written.policy?.kvDtype, "q8_0");
    assert.equal(written.policy?.["footer"], false);
    assert.equal(written.gpus.length, 1);
    assert.equal(written.gpus[0]?.headroomGiB, 3);
  });

  it("is what the toggle reads back", () => {
    rememberPolicy("footer", false);
    assert.equal(footerEnabled(), false);
    rememberPolicy("footer", true);
    assert.equal(footerEnabled(), true);
  });
});

describe("setupHasRun", () => {
  it("is false for a profile that holds only a preference", () => {
    // Otherwise switching the footer off on a fresh machine would silently
    // suppress the first-run hint for ever.
    rememberPolicy("footer", false);
    assert.equal(profileExists(), true);
    assert.equal(setupHasRun(), false);
  });

  it("is true once the wizard has written GPUs", () => {
    write(
      JSON.stringify({
        version: PROFILE_VERSION,
        gpus: [{ index: 0, display: true, headroomGiB: 3, displayEvidence: ["user"] }],
      }),
    );
    assert.equal(setupHasRun(), true);
  });

  it("is false when there is no profile at all", () => {
    assert.equal(setupHasRun(), false);
  });
});
