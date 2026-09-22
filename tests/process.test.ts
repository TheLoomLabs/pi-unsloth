import assert from "node:assert/strict";
import { homedir } from "node:os";
import { describe, it } from "node:test";

import { launchDirectory, startServerProcess } from "../src/process.ts";

/**
 * A stand-in for `child_process.spawn` that records its arguments and hands
 * back the two methods `startServerProcess` uses. Typed loosely on purpose:
 * the real signature is enormous and none of it is under test here.
 */
function recordingSpawn(): {
  calls: Array<{ binary: string; args: readonly string[]; options: { cwd?: string; detached?: boolean; stdio?: unknown } }>;
  spawn: (binary: string, args: readonly string[], options: Record<string, unknown>) => unknown;
} {
  const calls: Array<{ binary: string; args: readonly string[]; options: { cwd?: string; detached?: boolean; stdio?: unknown } }> = [];
  return {
    calls,
    spawn: (binary, args, options) => {
      calls.push({ binary, args, options });
      return { on: () => {}, unref: () => {} };
    },
  };
}

describe("startServerProcess", () => {
  it("starts the server from a directory that does not move with the session", () => {
    // Unsloth resolves `models_dir` relative to the server's cwd. Inheriting
    // Pi's would point the server at `<the user's project>/models`, and every
    // model they actually have would report as "not on disk".
    const recorder = recordingSpawn();
    const result = startServerProcess(["unsloth-studio", "--api-only"], {
      spawnProcess: recorder.spawn as never,
    });

    assert.equal(result.started, true);
    assert.equal(recorder.calls.length, 1);
    assert.equal(recorder.calls[0]!.options.cwd, launchDirectory());
    assert.notEqual(recorder.calls[0]!.options.cwd, process.cwd());
  });

  it("stays detached and silent, so it outlives the session without touching the TUI", () => {
    const recorder = recordingSpawn();
    startServerProcess(["unsloth-studio"], { spawnProcess: recorder.spawn as never });

    assert.equal(recorder.calls[0]!.options.detached, true);
    assert.equal(recorder.calls[0]!.options.stdio, "ignore");
  });

  it("passes the command through unchanged", () => {
    const recorder = recordingSpawn();
    startServerProcess(["a-wrapper", "studio", "--api-only"], { spawnProcess: recorder.spawn as never });

    assert.equal(recorder.calls[0]!.binary, "a-wrapper");
    assert.deepEqual(recorder.calls[0]!.args, ["studio", "--api-only"]);
  });

  it("refuses an empty command rather than spawning nothing", () => {
    const recorder = recordingSpawn();
    const result = startServerProcess([], { spawnProcess: recorder.spawn as never });

    assert.equal(result.started, false);
    assert.match(result.error ?? "", /no launch command/);
    assert.equal(recorder.calls.length, 0);
  });

  it("reports a spawn that throws instead of taking the session down", () => {
    const result = startServerProcess(["unsloth-studio"], {
      spawnProcess: (() => {
        throw new Error("ENOENT");
      }) as never,
    });

    assert.equal(result.started, false);
    assert.equal(result.error, "ENOENT");
  });
});

describe("launchDirectory", () => {
  it("is the home directory — the one path that is not per-project", () => {
    assert.equal(launchDirectory(), homedir());
  });
});
