import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { SystemGpu } from "../src/api/system.ts";
import { buildTopology, toFacts, usableGb, type TopologyInputs } from "../src/hardware/detect.ts";
import { DISPLAY_HEADROOM_GB, HEADLESS_HEADROOM_GB } from "../src/hardware/budget.ts";
import type { DrmCard } from "../src/hardware/connectors.ts";
import { PROFILE_VERSION, type MachineProfile } from "../src/hardware/profile.ts";

/** `/api/system` on the reference box, iGPU included — it is device 2. */
const SYSTEM: SystemGpu = {
  available: true,
  backend: "rocm",
  devices: [
    {
      index: 0,
      name: "AMD Radeon Graphics",
      memoryTotalGb: 23.98,
      vramUsedGb: 1.21,
      vramFreeGb: 22.77,
      sharedMemory: false,
      unifiedMemory: false,
    },
    {
      index: 1,
      name: "AMD Radeon Graphics",
      memoryTotalGb: 23.98,
      vramUsedGb: 0.03,
      vramFreeGb: 23.95,
      sharedMemory: false,
      unifiedMemory: false,
    },
    {
      index: 2,
      name: "AMD Ryzen 9 7900X 12-Core Processor",
      memoryTotalGb: 30.34,
      vramUsedGb: 0,
      vramFreeGb: 30.34,
      sharedMemory: true,
      unifiedMemory: true,
    },
  ],
};

const IDLE = [
  { index: 0, usedGb: 1.21, totalGb: 23.98 },
  { index: 1, usedGb: 0.03, totalGb: 23.98 },
];

const CARDS: DrmCard[] = [
  { name: "card1", pciAddress: "0000:03:00.0", connectors: [{ name: "card1-DP-3", status: "connected" }] },
  { name: "card0", pciAddress: "0000:07:00.0", connectors: [{ name: "card0-DP-4", status: "disconnected" }] },
  { name: "card2", pciAddress: "0000:7e:00.0", connectors: [{ name: "card2-DP-7", status: "disconnected" }] },
];

function inputs(overrides: Partial<TopologyInputs> = {}): TopologyInputs {
  return {
    system: SYSTEM,
    idle: IDLE,
    profile: undefined,
    cards: CARDS,
    unslothVersion: "2026.9.7",
    ...overrides,
  };
}

describe("buildTopology — step 1, enumeration", () => {
  it("yields exactly the compute GPUs, counting the integrated one as ignored", () => {
    const topology = buildTopology(inputs());
    assert.deepEqual(topology.gpus.map((gpu) => gpu.index), [0, 1]);
    assert.equal(topology.ignoredCount, 1);
    assert.equal(topology.degraded, false);
    assert.equal(topology.backend, "rocm");
  });

  it("degrades — with a reason — when there is no topology to have", () => {
    for (const [patch, needle] of [
      [{ system: undefined }, "/api/system"],
      [{ system: { ...SYSTEM, devices: [] } }, "no GPUs at all"],
      [{ system: { ...SYSTEM, devices: [SYSTEM.devices[2]!] } }, "only integrated graphics"],
      [{ system: { ...SYSTEM, available: false } }, "no usable acceleration"],
    ] as Array<[Partial<TopologyInputs>, string]>) {
      const topology = buildTopology(inputs(patch));
      assert.equal(topology.degraded, true, needle);
      assert.ok(topology.reason?.includes(needle), `${topology.reason} ∌ ${needle}`);
      // Degraded means sizing is off, and says the same thing the panel does.
      assert.equal(topology.sizing.enabled, false);
      assert.equal(topology.sizing.reason, topology.reason);
    }
  });
});

describe("buildTopology — step 2, the display GPU", () => {
  it("picks the card holding framebuffers and records both signals", () => {
    const topology = buildTopology(inputs());
    assert.equal(topology.displaySource, "idle-vram");
    assert.equal(topology.gpus[0]?.display, true);
    assert.deepEqual(topology.gpus[0]?.evidence, ["idle-vram", "drm-connector:card1-DP-3"]);
    assert.equal(topology.gpus[1]?.display, false);
    assert.deepEqual(topology.gpus[1]?.evidence, []);
  });

  it("says nothing at all before anything has been measured at idle", () => {
    const topology = buildTopology(inputs({ idle: [] }));
    assert.equal(topology.displaySource, "none");
    assert.equal(topology.gpus[0]?.display, false);
    assert.deepEqual(topology.gpus[0]?.evidence, []);
    // …but the connector is still visible, which is what lets the wizard say
    // "headless by VRAM — but card1-DP-3 is connected".
    assert.deepEqual(topology.gpus[0]?.connected, ["card1-DP-3"]);
    assert.equal(topology.gpus[0]?.idleUsedGb, undefined);
  });

  it("lets a corrected profile overrule the measurement, and keeps its evidence", () => {
    const profile: MachineProfile = {
      version: PROFILE_VERSION,
      gpus: [
        { index: 0, display: false, headroomGiB: 0.5, displayEvidence: [] },
        { index: 1, display: true, headroomGiB: 3, displayEvidence: ["user"] },
      ],
    };
    const topology = buildTopology(inputs({ profile }));
    assert.equal(topology.displaySource, "profile");
    assert.equal(topology.gpus[0]?.display, false);
    assert.equal(topology.gpus[1]?.display, true);
    assert.deepEqual(topology.gpus[1]?.evidence, ["user"]);
    assert.equal(topology.gpus[1]?.headroomGb, 3);
  });

  it("never lets the connectors overrule the measurement, only join it", () => {
    // Connectors say card1 (→ GPU 0) has a monitor; idle VRAM says GPU 1 does.
    const idle = [
      { index: 0, usedGb: 0.02, totalGb: 23.98 },
      { index: 1, usedGb: 1.4, totalGb: 23.98 },
    ];
    const topology = buildTopology(inputs({ idle }));
    assert.equal(topology.gpus[0]?.display, false);
    assert.equal(topology.gpus[1]?.display, true);
    // The disagreement stays visible rather than being resolved silently.
    assert.deepEqual(topology.gpus[0]?.connected, ["card1-DP-3"]);
    assert.deepEqual(topology.gpus[1]?.evidence, ["idle-vram"]);
  });

  it("explains why there was no cross-check when the lists cannot be lined up", () => {
    const topology = buildTopology(inputs({ cards: CARDS.slice(0, 2) }));
    assert.ok(topology.connectorNote?.includes("2 DRM cards for 3 GPUs"), topology.connectorNote);
    assert.deepEqual(topology.gpus[0]?.evidence, ["idle-vram"]);
  });
});

describe("buildTopology — step 3, budgets", () => {
  it("gives the display GPU the larger headroom by default", () => {
    const topology = buildTopology(inputs());
    assert.equal(topology.gpus[0]?.headroomGb, DISPLAY_HEADROOM_GB);
    assert.equal(topology.gpus[1]?.headroomGb, HEADLESS_HEADROOM_GB);
    assert.equal(usableGb(topology.gpus[0]!).toFixed(2), "19.77");
    assert.equal(usableGb(topology.gpus[1]!).toFixed(2), "23.45");
  });

  it("treats an unmeasured card as holding nothing rather than guessing", () => {
    const topology = buildTopology(inputs({ idle: [] }));
    assert.equal(toFacts(topology.gpus[0]!).idleUsedGb, 0);
  });
});

describe("buildTopology — the version gate", () => {
  it("passes the gate through from the server's own version", () => {
    assert.equal(buildTopology(inputs()).sizing.enabled, true);
    const old = buildTopology(inputs({ unslothVersion: "2026.8.4" }));
    assert.equal(old.sizing.enabled, false);
    assert.ok(old.sizing.reason?.includes("2026.9"));
    // …and everything else still works, which is the point of gating.
    assert.equal(old.degraded, false);
    assert.equal(old.gpus.length, 2);
  });
});

describe("buildTopology — a server on another machine", () => {
  it("consults no DRM connector, because they belong to the wrong machine", () => {
    // Same sysfs, same GPUs; the only difference is where the server is. A
    // monitor plugged in *here* says nothing about a card over there.
    const topology = buildTopology(inputs({ remote: true }));
    assert.deepEqual(topology.gpus[0]?.connected, []);
    assert.deepEqual(topology.gpus[0]?.evidence, ["idle-vram"]);
    assert.equal(topology.remote, true);
  });

  it("still identifies the display GPU, because idle VRAM is the server's own", () => {
    const topology = buildTopology(inputs({ remote: true }));
    assert.equal(topology.displaySource, "idle-vram");
    assert.equal(topology.gpus[0]?.display, true);
    assert.equal(topology.gpus[1]?.display, false);
  });

  it("raises no cross-check note — that would be a complaint about this machine", () => {
    const mismatched = buildTopology(inputs({ remote: true, cards: [CARDS[0]!] }));
    assert.equal(mismatched.connectorNote, undefined);
    // Locally the same input does have something to say.
    assert.ok(buildTopology(inputs({ cards: [CARDS[0]!] })).connectorNote);
  });

  it("carries the flag through a degraded topology, where the UI still needs it", () => {
    const topology = buildTopology(inputs({ remote: true, system: undefined }));
    assert.equal(topology.degraded, true);
    assert.equal(topology.remote, true);
  });

  it("is local unless it is told otherwise", () => {
    assert.equal(buildTopology(inputs()).remote, false);
  });
});
