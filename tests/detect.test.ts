import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, beforeEach, describe, it } from "node:test";

import { DISPLAY_IDLE_GB, displayGpus, displayGpusFromIdle, mergeIdleVram } from "../src/hardware/detect.ts";
import { PROFILE_FILENAME, profileDisplayGpus } from "../src/settings.ts";

const AGENT_DIR_ENV = "PI_CODING_AGENT_DIR";

let dir: string;
let saved: string | undefined;

function writeProfile(contents: string): void {
  writeFileSync(join(dir, PROFILE_FILENAME), contents);
}

before(() => {
  saved = process.env[AGENT_DIR_ENV];
  dir = mkdtempSync(join(tmpdir(), "pi-unsloth-detect-"));
});

beforeEach(() => {
  process.env[AGENT_DIR_ENV] = dir;
  writeProfile("{}");
});

after(() => {
  if (saved === undefined) delete process.env[AGENT_DIR_ENV];
  else process.env[AGENT_DIR_ENV] = saved;
  rmSync(dir, { recursive: true, force: true });
});

/** The reference box at idle: one card holds the compositor, one does not. */
const IDLE = [
  { index: 0, usedGb: 1.21, totalGb: 23.98 },
  { index: 1, usedGb: 0.03, totalGb: 23.98 },
];

describe("displayGpusFromIdle", () => {
  it("picks the card holding framebuffers, not the bigger one", () => {
    assert.deepEqual(displayGpusFromIdle(IDLE), [0]);
  });

  it("finds none on a headless box, which is an answer and not an error", () => {
    assert.deepEqual(displayGpusFromIdle(IDLE.map((gpu) => ({ ...gpu, usedGb: 0.02 }))), []);
  });

  it("reads the threshold, exclusively", () => {
    assert.deepEqual(displayGpusFromIdle([{ index: 0, usedGb: DISPLAY_IDLE_GB, totalGb: 24 }]), []);
    assert.deepEqual(displayGpusFromIdle([{ index: 0, usedGb: DISPLAY_IDLE_GB + 0.01, totalGb: 24 }]), [0]);
  });
});

describe("displayGpus", () => {
  it("says nothing at all before anything has been measured at idle", () => {
    assert.equal(displayGpus([]), undefined);
  });

  it("uses the measurement when the profile does not say", () => {
    assert.deepEqual(displayGpus(IDLE), [0]);
  });

  it("lets a corrected profile override the measurement", () => {
    writeProfile(JSON.stringify({ gpus: [{ index: 0, display: false }, { index: 1, display: true }] }));
    assert.deepEqual(displayGpus(IDLE), [1]);
  });

  it("treats a profile that flags nothing as a headless answer, not a missing one", () => {
    writeProfile(JSON.stringify({ gpus: [{ index: 0, display: false }, { index: 1, display: false }] }));
    assert.deepEqual(displayGpus(IDLE), []);
  });
});

describe("profileDisplayGpus", () => {
  it("says nothing for a profile with no topology in it yet", () => {
    assert.equal(profileDisplayGpus(), undefined);
    writeProfile(JSON.stringify({ gpus: "soon" }));
    assert.equal(profileDisplayGpus(), undefined);
  });

  it("ignores entries that do not state both an index and a flag", () => {
    writeProfile(JSON.stringify({ gpus: [{ index: 0 }, { display: true }, null, { index: 1, display: true }] }));
    assert.deepEqual(profileDisplayGpus(), [1]);
  });
});

describe("mergeIdleVram", () => {
  it("keeps the lowest figure seen, not the most recent", () => {
    // The reference box, unloaded: the first reading catches the tail of a
    // 38 GiB release, the second is the real floor.
    const poisoned = [
      { index: 0, usedGb: 21.1, totalGb: 23.98 },
      { index: 1, usedGb: 18.4, totalGb: 23.98 },
    ];
    const settled = [
      { index: 0, usedGb: 1.6, totalGb: 23.98 },
      { index: 1, usedGb: 0.03, totalGb: 23.98 },
    ];
    assert.deepEqual(mergeIdleVram(poisoned, settled), settled);
    assert.deepEqual(mergeIdleVram(settled, poisoned), settled);
    assert.deepEqual(displayGpusFromIdle(mergeIdleVram(poisoned, settled)), [0]);
  });

  it("does not forget a GPU that one query failed to mention", () => {
    const held = [{ index: 0, usedGb: 1.6, totalGb: 24 }, { index: 1, usedGb: 0.03, totalGb: 24 }];
    assert.deepEqual(mergeIdleVram(held, [{ index: 1, usedGb: 0.02, totalGb: 24 }]), [
      { index: 0, usedGb: 1.6, totalGb: 24 },
      { index: 1, usedGb: 0.02, totalGb: 24 },
    ]);
  });

  it("accepts a first observation as-is", () => {
    const seen = [{ index: 1, usedGb: 0.03, totalGb: 24 }];
    assert.deepEqual(mergeIdleVram([], seen), seen);
  });
});
