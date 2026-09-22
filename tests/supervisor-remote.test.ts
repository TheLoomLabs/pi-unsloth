import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import { unloadAll, unloadAndReport, unloadMessage, type ProcessLayer, type UnloadResult } from "../src/supervisor.ts";
import { resetState } from "../src/state.ts";

/**
 * A process layer that does nothing and remembers being asked.
 *
 * The assertion these tests exist for is an *absence* — no enumeration, no
 * signal — and the only honest way to observe that is a layer that counts its
 * own calls. The real one cannot be used: its effect is terminating something.
 */
function countingProcesses(): { layer: ProcessLayer; calls: string[] } {
  const calls: string[] = [];
  return {
    calls,
    layer: {
      supported: () => {
        calls.push("supported");
        return true;
      },
      list: async () => {
        calls.push("list");
        return [];
      },
      terminate: async () => {
        calls.push("terminate");
        return { found: [], survivors: [] };
      },
    },
  };
}

/**
 * Enough of an `ExtensionContext` for the unload path: somewhere to put a
 * notification, and a registry with no key. The endpoint is unreachable in
 * every test here, which is the point — the rule must hold for a server that
 * cannot be talked to at all.
 */
function stubContext(): { ctx: never; messages: string[] } {
  const messages: string[] = [];
  const ctx = {
    hasUI: true,
    mode: "tui",
    ui: {
      notify: (text: string) => messages.push(text),
      setStatus: () => {},
      theme: { fg: (_token: string, text: string) => text, bold: (text: string) => text },
    },
    modelRegistry: { getApiKeyForProvider: async () => undefined },
  };
  return { ctx: ctx as never, messages };
}

/**
 * `.invalid` is reserved and never resolves, so these tests fail their DNS
 * lookup immediately rather than waiting on a timeout — and the host is not
 * loopback, which is the property under test.
 */
const REMOTE = "http://gpubox.invalid:8888";
const LOCAL = "http://127.0.0.1:59999";

let previous: string | undefined;

beforeEach(() => {
  previous = process.env["UNSLOTH_BASE_URL"];
  resetState();
});

afterEach(() => {
  if (previous === undefined) delete process.env["UNSLOTH_BASE_URL"];
  else process.env["UNSLOTH_BASE_URL"] = previous;
});

describe("unloadAll against a remote endpoint", () => {
  it("sends no signal and enumerates no process on this machine", async () => {
    // The failure this guards: on 2026-09-21 an unload aimed at a stub server
    // terminated the real llama-server on the reference box.
    process.env["UNSLOTH_BASE_URL"] = REMOTE;
    const { ctx } = stubContext();
    const processes = countingProcesses();

    const result = await unloadAll(ctx, { processes: processes.layer });

    assert.deepEqual(processes.calls, []);
    assert.deepEqual(result.stopped, []);
    assert.deepEqual(result.survivors, []);
  });

  it("reports where the VRAM is still held, by name", async () => {
    process.env["UNSLOTH_BASE_URL"] = REMOTE;
    const { ctx } = stubContext();

    const result = await unloadAll(ctx, { processes: countingProcesses().layer });

    assert.equal(result.heldAt, "gpubox.invalid:8888");
  });

  it("consults the process layer when the endpoint is this machine", async () => {
    // The mirror image, so the first test cannot pass by the layer simply
    // never being reached.
    process.env["UNSLOTH_BASE_URL"] = LOCAL;
    const { ctx } = stubContext();
    const processes = countingProcesses();

    const result = await unloadAll(ctx, { processes: processes.layer });

    assert.ok(processes.calls.includes("supported"));
    assert.ok(processes.calls.includes("terminate"));
    assert.equal(result.heldAt, undefined);
  });
});

describe("unloadAndReport against a remote endpoint", () => {
  it("names the host rather than claiming this machine freed anything", async () => {
    process.env["UNSLOTH_BASE_URL"] = REMOTE;
    const { ctx, messages } = stubContext();

    // A model the server said was loaded is what makes it a report rather
    // than "nothing loaded"; here the server cannot be reached, so the
    // honest line is the nothing-loaded one.
    const result = await unloadAndReport(ctx, { processes: countingProcesses().layer });

    assert.equal(result.heldAt, "gpubox.invalid:8888");
    assert.equal(messages.length, 1);
    assert.match(messages[0]!, /Nothing loaded/);
  });
});

describe("unloadMessage", () => {
  const nothing: UnloadResult = {
    unloaded: [],
    stopped: [],
    survivors: [],
    freedGb: undefined,
    error: undefined,
    heldAt: undefined,
  };

  it("says where the weights still are, and what would free them", () => {
    const text = unloadMessage({ ...nothing, unloaded: ["/models/a"], heldAt: "gpubox:8888" }).text;
    assert.match(text, /Unloaded on gpubox:8888/);
    assert.match(text, /llama-server is stopped on that machine/);
    // Never "this machine": the one person who can act is over there.
    assert.doesNotMatch(text, /this machine\b(?! is stopped)/);
  });

  it("reports a local unload with the figure it actually measured", () => {
    assert.equal(unloadMessage({ ...nothing, stopped: [11170], freedGb: 42.6 }).text, "✓ Unloaded. 42.6 GiB freed.");
    assert.equal(unloadMessage({ ...nothing, stopped: [11170] }).text, "✓ Unloaded.");
  });

  it("puts survivors ahead of everything else, remote included", () => {
    // A process still holding VRAM is the one thing worth interrupting for.
    const line = unloadMessage({ ...nothing, unloaded: ["/models/a"], survivors: [42], heldAt: "gpubox:8888" });
    assert.equal(line.level, "warning");
    assert.match(line.text, /still running \(42\)/);
  });

  it("says nothing was loaded when nothing was", () => {
    assert.match(unloadMessage(nothing).text, /Nothing loaded/);
    assert.match(unloadMessage({ ...nothing, error: "no route" }).text, /Nothing loaded — no route/);
  });
});
