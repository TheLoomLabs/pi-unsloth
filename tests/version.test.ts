import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { MIN_SIZING_VERSION, compareVersions, parseVersion, sizingSupport } from "../src/hardware/version.ts";

describe("parseVersion", () => {
  it("reads the calendar version the server reports", () => {
    assert.deepEqual(parseVersion("2026.9.7"), [2026, 9, 7]);
  });

  it("keeps the numeric prefix of a pre-release rather than discarding it", () => {
    assert.deepEqual(parseVersion("2026.9.7rc1"), [2026, 9, 7]);
  });

  it("stops at the first part that is not a number", () => {
    assert.deepEqual(parseVersion("2026.dev.3"), [2026]);
    assert.deepEqual(parseVersion("dev"), []);
    assert.deepEqual(parseVersion(undefined), []);
  });
});

describe("compareVersions", () => {
  it("treats a missing component as zero", () => {
    assert.equal(compareVersions("2026.9", "2026.9.0"), 0);
    assert.equal(compareVersions("2026.9.7", "2026.9"), 1);
  });

  it("compares numerically, not as text", () => {
    assert.equal(compareVersions("2026.10", "2026.9"), 1);
    assert.equal(compareVersions("2026.9", "2026.10"), -1);
  });
});

describe("sizingSupport", () => {
  it("allows the version the API reference was verified against", () => {
    const support = sizingSupport("2026.9.7");
    assert.equal(support.enabled, true);
    assert.equal(support.reason, undefined);
  });

  it("refuses an older server, naming the version required", () => {
    const support = sizingSupport("2026.8.4");
    assert.equal(support.enabled, false);
    assert.ok(support.reason?.includes(MIN_SIZING_VERSION));
    assert.ok(support.reason?.includes("2026.8.4"), support.reason);
  });

  it("refuses a server that reports no version at all, and says which it is", () => {
    for (const version of [undefined, "", "unknown"]) {
      const support = sizingSupport(version);
      assert.equal(support.enabled, false);
      assert.ok(support.reason?.includes("does not report a version"), support.reason);
    }
  });

  it("carries the version through so the UI can show what it found", () => {
    assert.equal(sizingSupport("2026.8.4").version, "2026.8.4");
  });
});
