import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  endpointHost,
  endpointLabel,
  isLocalEndpoint,
  isLoopbackHost,
  normaliseEndpoint,
  sameEndpoint,
  thisMachine,
  type Machine,
} from "../src/endpoint.ts";

/**
 * A machine with one LAN address and a name, standing in for the box the
 * extension happens to be running on. No test may depend on the real one:
 * the whole point of the rule is the addresses it is *not* being run at.
 */
const MACHINE: Machine = {
  addresses: ["127.0.0.1", "::1", "192.168.1.10", "fe80::1"],
  hostname: "workstation.lan",
};

describe("isLoopbackHost", () => {
  it("accepts every spelling of this machine's own loopback", () => {
    for (const host of ["localhost", "127.0.0.1", "127.1.2.3", "::1", "0:0:0:0:0:0:0:1", "[::1]"]) {
      assert.equal(isLoopbackHost(host), true, host);
    }
  });

  it("accepts the addresses a server binds to, which reach this machine", () => {
    assert.equal(isLoopbackHost("0.0.0.0"), true);
    assert.equal(isLoopbackHost("::"), true);
  });

  it("rejects a LAN address, which may be anyone", () => {
    assert.equal(isLoopbackHost("192.168.1.40"), false);
    assert.equal(isLoopbackHost("10.0.0.1"), false);
  });
});

describe("isLocalEndpoint", () => {
  it("is true for loopback, whatever the port or scheme", () => {
    assert.equal(isLocalEndpoint("http://127.0.0.1:8888", MACHINE), true);
    assert.equal(isLocalEndpoint("http://localhost:8888", MACHINE), true);
    assert.equal(isLocalEndpoint("https://localhost", MACHINE), true);
    assert.equal(isLocalEndpoint("http://[::1]:8888", MACHINE), true);
  });

  it("is true for this machine's own LAN address", () => {
    // The one that would otherwise cost a user `/unsloth off` for typing the
    // address their server actually binds to.
    assert.equal(isLocalEndpoint("http://192.168.1.10:8888", MACHINE), true);
  });

  it("is true for this machine's own name, with or without the search domain", () => {
    assert.equal(isLocalEndpoint("http://workstation.lan:8888", MACHINE), true);
    assert.equal(isLocalEndpoint("http://workstation:8888", MACHINE), true);
  });

  it("is false for another machine on the same subnet", () => {
    assert.equal(isLocalEndpoint("http://192.168.1.40:8888", MACHINE), false);
    assert.equal(isLocalEndpoint("http://gpubox.lan:8888", MACHINE), false);
  });

  it("is false for anything that is not an address, because that touches nothing", () => {
    assert.equal(isLocalEndpoint("", MACHINE), false);
    assert.equal(isLocalEndpoint("not a url", MACHINE), false);
  });

  it("reads the real machine without throwing or resolving anything", () => {
    const machine = thisMachine();
    assert.ok(Array.isArray(machine.addresses));
    assert.equal(isLocalEndpoint("http://127.0.0.1:8888", machine), true);
  });
});

describe("normaliseEndpoint", () => {
  it("assumes the documented port for a bare host", () => {
    assert.equal(normaliseEndpoint("192.168.1.40"), "http://192.168.1.40:8888");
    assert.equal(normaliseEndpoint("gpubox"), "http://gpubox:8888");
  });

  it("keeps a port that was given", () => {
    assert.equal(normaliseEndpoint("192.168.1.40:9000"), "http://192.168.1.40:9000");
  });

  it("keeps a scheme that was given, and leaves https on its own port", () => {
    // An https address is a proxy in front of something; 443 is its business.
    assert.equal(normaliseEndpoint("https://unsloth.example.com"), "https://unsloth.example.com");
    assert.equal(normaliseEndpoint("http://gpubox:8888/"), "http://gpubox:8888");
  });

  it("refuses what cannot be an address, so the field stays open", () => {
    assert.equal(normaliseEndpoint(""), undefined);
    assert.equal(normaliseEndpoint("   "), undefined);
    assert.equal(normaliseEndpoint("ftp://gpubox"), undefined);
    assert.equal(normaliseEndpoint("http://"), undefined);
  });
});

describe("sameEndpoint", () => {
  it("sees through a trailing slash and a missing default port", () => {
    assert.equal(sameEndpoint("http://gpubox:8888", "http://gpubox:8888/"), true);
    assert.equal(sameEndpoint("http://gpubox", "http://gpubox:8888"), true);
  });

  it("separates two machines, and two ports on one machine", () => {
    assert.equal(sameEndpoint("http://gpubox:8888", "http://other:8888"), false);
    assert.equal(sameEndpoint("http://gpubox:8888", "http://gpubox:9000"), false);
  });

  it("treats an absent address as matching nothing it could be compared with", () => {
    assert.equal(sameEndpoint(undefined, "http://gpubox:8888"), false);
    assert.equal(sameEndpoint("http://gpubox:8888", undefined), false);
  });
});

describe("endpointHost / endpointLabel", () => {
  it("names the host for a message, without the scheme nobody reads", () => {
    assert.equal(endpointLabel("http://192.168.1.40:8888"), "192.168.1.40:8888");
    assert.equal(endpointHost("http://192.168.1.40:8888"), "192.168.1.40");
    assert.equal(endpointHost("nonsense"), undefined);
  });
});
