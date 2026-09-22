/**
 * The Linux DRM cross-check — a **second opinion** on which GPU drives the
 * monitor.
 *
 * A connector's `status` file under `/sys/class/drm` says which outputs have
 * something plugged into them, which is a fact about the machine rather than an
 * inference. What *is* an inference is tying a DRM card back to a GPU index: it
 * rests on the backend enumerating devices by ascending PCI bus, which held on
 * the reference box (`card1` = `0000:03:00.0` = GPU 0) and is exactly the kind
 * of assumption that breaks on someone else's hardware.
 *
 * So this file **confirms, it never overrides**. The idle-VRAM measurement
 * decides; the connectors add evidence to that conclusion, and where they
 * disagree they say so and leave the correction to the user's `d` key. Nothing
 * here is consulted on any platform but Linux, and nothing here throws: an
 * unreadable sysfs simply produces no evidence.
 */

import { readFileSync, readdirSync, realpathSync } from "node:fs";
import { basename, join } from "node:path";

/** Where the kernel exposes DRM devices. A parameter only so tests can fake it. */
export const DRM_ROOT = "/sys/class/drm";

/** `0000:03:00.0` — domain:bus:device.function, fixed width and hex. */
const PCI_ADDRESS = /^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d$/i;

const CARD = /^card\d+$/;
const CONNECTOR = /^(card\d+)-(.+)$/;

export interface DrmConnector {
  /** `card1-DP-3`, exactly as sysfs names it — what the evidence string says. */
  name: string;
  /** `connected` | `disconnected` | `unknown`, in the kernel's own words. */
  status: string;
}

export interface DrmCard {
  /** `card1`. Never used as a GPU id — see the note at the top of this file. */
  name: string;
  /** PCI address from the `device` link. The only thing cards are ordered by. */
  pciAddress: string;
  connectors: DrmConnector[];
}

function read(path: string): string | undefined {
  try {
    return readFileSync(path, "utf8").trim();
  } catch {
    return undefined;
  }
}

function pciAddressOf(root: string, card: string): string | undefined {
  try {
    const address = basename(realpathSync(join(root, card, "device")));
    return PCI_ADDRESS.test(address) ? address : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Every DRM card with a PCI address, ordered by that address.
 *
 * Cards without one are dropped rather than guessed at: a card that cannot be
 * placed on the bus cannot be lined up against a GPU index either, and a
 * partial list would produce a confident wrong mapping.
 */
export function readDrmCards(root: string = DRM_ROOT): DrmCard[] {
  if (process.platform !== "linux" && root === DRM_ROOT) return [];

  let entries: string[];
  try {
    entries = readdirSync(root);
  } catch {
    return [];
  }

  const cards = new Map<string, DrmCard>();
  for (const entry of entries.filter((name) => CARD.test(name))) {
    const pciAddress = pciAddressOf(root, entry);
    if (pciAddress === undefined) continue;
    cards.set(entry, { name: entry, pciAddress, connectors: [] });
  }

  for (const entry of entries) {
    const match = CONNECTOR.exec(entry);
    if (!match) continue;
    const card = cards.get(match[1] ?? "");
    if (!card) continue;
    const status = read(join(root, entry, "status"));
    if (status === undefined) continue;
    card.connectors.push({ name: entry, status });
  }

  return [...cards.values()].sort((a, b) => a.pciAddress.localeCompare(b.pciAddress));
}

/** The outputs of this card with something plugged into them. */
export function connectedOutputs(card: DrmCard): string[] {
  return card.connectors.filter((connector) => connector.status === "connected").map((connector) => connector.name);
}

export interface ConnectorCheck {
  /** The card lined up against each GPU index, when they could be lined up. */
  cards: Map<number, DrmCard>;
  /** GPU indices whose card has a live output. Empty is an answer; see `note`. */
  connectedIndices: number[];
  /** Why there is no mapping, when there is none. */
  note: string | undefined;
}

/**
 * Line DRM cards up against GPU indices, or explain why they cannot be.
 *
 * `deviceIndices` is **every** device `/api/system` reports, integrated
 * graphics included: an iGPU has a DRM card too, and dropping it from one list
 * but not the other is how the mapping goes off by one.
 *
 * The rule is deliberately dull — both lists sorted, then matched in order,
 * since the backend enumerates by ascending PCI bus. A mismatch in length
 * means the assumption does not hold here (a card without a PCI address, a GPU
 * the kernel does not expose, a container), and the honest answer is then no
 * evidence rather than a shifted one.
 */
export function crossCheck(deviceIndices: readonly number[], cards: readonly DrmCard[]): ConnectorCheck {
  const empty = { cards: new Map<number, DrmCard>(), connectedIndices: [] };
  if (cards.length === 0) return { ...empty, note: "no DRM cards readable" };
  if (cards.length !== deviceIndices.length) {
    return { ...empty, note: `${cards.length} DRM cards for ${deviceIndices.length} GPUs — cannot line them up` };
  }

  const ordered = [...deviceIndices].sort((a, b) => a - b);
  const mapped = new Map<number, DrmCard>();
  const connectedIndices: number[] = [];
  ordered.forEach((index, position) => {
    const card = cards[position];
    if (!card) return;
    mapped.set(index, card);
    if (connectedOutputs(card).length > 0) connectedIndices.push(index);
  });

  return { cards: mapped, connectedIndices, note: undefined };
}

/**
 * The evidence line for one GPU: `drm-connector:card1-DP-3`.
 *
 * Only ever produced for a card with a live output, and only ever *added* to a
 * conclusion the VRAM measurement already reached.
 */
export function connectorEvidence(index: number, check: ConnectorCheck): string[] {
  const card = check.cards.get(index);
  if (!card) return [];
  return connectedOutputs(card).map((name) => `drm-connector:${name}`);
}
