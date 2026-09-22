import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, describe, it } from "node:test";

import { connectedOutputs, connectorEvidence, crossCheck, readDrmCards } from "../src/hardware/connectors.ts";

let root: string;

/**
 * The reference box's sysfs, rebuilt in a temporary directory: `card1` is on
 * the lower PCI bus and holds the connected output, while `card0` — the *lower*
 * card number — is the headless one. The inversion is the whole point.
 */
before(() => {
  const dir = mkdtempSync(join(tmpdir(), "pi-unsloth-drm-"));
  root = join(dir, "drm");
  const cards: Array<[string, string, Array<[string, string]>]> = [
    ["card0", "0000:07:00.0", [["card0-DP-4", "disconnected"], ["card0-HDMI-A-2", "disconnected"]]],
    ["card1", "0000:03:00.0", [["card1-DP-3", "connected"], ["card1-HDMI-A-1", "disconnected"]]],
    ["card2", "0000:7e:00.0", [["card2-DP-7", "disconnected"]]],
  ];
  for (const [card, address, connectors] of cards) {
    mkdirSync(join(root, card), { recursive: true });
    const device = join(root, address);
    mkdirSync(device, { recursive: true });
    symlinkSync(device, join(root, card, "device"));
    for (const [name, status] of connectors) {
      mkdirSync(join(root, name), { recursive: true });
      writeFileSync(join(root, name, "status"), `${status}\n`);
    }
  }
  // A card the kernel exposes with no PCI address behind it, which must not
  // shift the mapping by taking a slot in the ordering.
  mkdirSync(join(root, "card9"), { recursive: true });
});

after(() => {
  rmSync(root, { recursive: true, force: true });
});

describe("readDrmCards", () => {
  it("orders cards by PCI address, not by card number", () => {
    assert.deepEqual(
      readDrmCards(root).map((card) => card.name),
      ["card1", "card0", "card2"],
    );
  });

  it("finds the connected output the reference box has", () => {
    const first = readDrmCards(root)[0];
    assert.equal(first?.name, "card1");
    assert.deepEqual(connectedOutputs(first!), ["card1-DP-3"]);
  });

  it("drops a card with no PCI address rather than guessing where it sits", () => {
    assert.equal(readDrmCards(root).some((card) => card.name === "card9"), false);
  });

  it("returns nothing at all, and does not throw, for a directory that is not there", () => {
    assert.deepEqual(readDrmCards(join(root, "nope")), []);
  });
});

describe("crossCheck", () => {
  it("agrees with the idle-VRAM signal on the reference box", () => {
    // `/api/system` indices 0, 1, 2 — the iGPU included, because it has a card.
    const check = crossCheck([0, 1, 2], readDrmCards(root));
    assert.equal(check.note, undefined);
    assert.deepEqual(check.connectedIndices, [0]);
    assert.equal(check.cards.get(0)?.name, "card1");
    assert.equal(check.cards.get(1)?.name, "card0");
  });

  it("produces the evidence string the profile records", () => {
    const check = crossCheck([0, 1, 2], readDrmCards(root));
    assert.deepEqual(connectorEvidence(0, check), ["drm-connector:card1-DP-3"]);
    assert.deepEqual(connectorEvidence(1, check), []);
  });

  it("refuses to line up lists of different lengths rather than shifting them", () => {
    // Dropping the iGPU from one list but not the other is the off-by-one this
    // guard exists for: it would name card1 as GPU 1.
    const check = crossCheck([0, 1], readDrmCards(root));
    assert.equal(check.cards.size, 0);
    assert.deepEqual(check.connectedIndices, []);
    assert.ok(check.note?.includes("3 DRM cards for 2 GPUs"), check.note);
  });

  it("says so when there are no cards to check against", () => {
    const check = crossCheck([0], []);
    assert.equal(check.note, "no DRM cards readable");
    assert.deepEqual(connectorEvidence(0, check), []);
  });
});
