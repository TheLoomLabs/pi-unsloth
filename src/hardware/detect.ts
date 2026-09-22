/**
 * Which GPUs are driving a display.
 *
 * This is the one thing `/api/system` does not label, and the one that matters
 * most: sizing a model onto the card driving the monitor is how you freeze
 * someone's desktop.
 *
 * M3 implements only the **primary signal** — idle VRAM — because that is all
 * the panel needs to tag a bar, and because it is a direct measurement rather
 * than an inference. M4 owns the rest of this file's eventual job: the Linux
 * DRM connector cross-check, the wizard that lets a wrong guess be corrected,
 * and the profile that remembers the answer. The profile is already read here,
 * so the moment M4's wizard writes one it takes effect.
 */

import type { UnslothClient } from "../api/client.ts";
import { isLocalEndpoint } from "../endpoint.ts";
import { listLoadedModels } from "../api/models.ts";
import { computeDevices, getServerVersions, getSystemGpu, type SystemGpu } from "../api/system.ts";
import { profileDisplayGpus } from "../settings.ts";
import { state, type GpuUsage } from "../state.ts";
import { budgetGb, defaultHeadroomGb, type GpuFacts } from "./budget.ts";
import { connectedOutputs, connectorEvidence, crossCheck, readDrmCards, type DrmCard } from "./connectors.ts";
import { profileForEndpoint, readMachineProfile, type MachineProfile, type ProfileGpu } from "./profile.ts";
import { sizingSupport, type SizingSupport } from "./version.ts";

/**
 * Idle VRAM above which a GPU is holding a compositor's framebuffers.
 *
 * Not a fact about anyone's hardware — it is a threshold sitting between the
 * ~0.03 GiB a headless card reports and the ~1.2 GiB a desktop one does. It is
 * deliberately nearer the floor: calling a headless GPU "display" costs
 * headroom, the other way round costs a desktop.
 */
export const DISPLAY_IDLE_GB = 0.3;

/**
 * Display GPUs according to VRAM measured **with nothing loaded**.
 *
 * Reading this while a model is resident is meaningless, which is why the
 * caller passes an idle snapshot rather than live figures.
 */
export function displayGpusFromIdle(idle: readonly GpuUsage[]): number[] {
  return idle.filter((gpu) => gpu.usedGb > DISPLAY_IDLE_GB).map((gpu) => gpu.index);
}

/**
 * The answer the UI should show: what the user told us, else what we measured.
 *
 * A profile written by the wizard always wins — it is the user correcting the
 * guess, and a guess that cannot be corrected is the failure mode this project
 * exists to avoid. `undefined` means "no idea yet", and the UI then says
 * nothing rather than tagging the wrong card.
 */
export function displayGpus(idle: readonly GpuUsage[]): number[] | undefined {
  const stated = profileDisplayGpus();
  if (stated) return stated;
  return idle.length > 0 ? displayGpusFromIdle(idle) : undefined;
}

/**
 * Fold a new idle observation into the one already held, per GPU.
 *
 * Taking the **lowest** figure seen, rather than the latest, for the same
 * reason the unload settles on a minimum: with nothing loaded nothing is
 * allocating, so a higher reading is memory that has not come back yet, not
 * memory in use. Without this, a panel opened seconds after unloading a 38 GiB
 * model records the tail of that release as "idle" and then tags every card as
 * display-attached.
 *
 * A GPU absent from `observed` keeps its previous figure: one failed query is
 * not evidence that a card went away.
 */
export function mergeIdleVram(previous: readonly GpuUsage[], observed: readonly GpuUsage[]): GpuUsage[] {
  const merged = new Map(previous.map((gpu) => [gpu.index, gpu]));
  for (const gpu of observed) {
    const held = merged.get(gpu.index);
    merged.set(gpu.index, held && held.usedGb <= gpu.usedGb ? held : gpu);
  }
  return [...merged.values()].sort((a, b) => a.index - b.index);
}

/**
 * ---------------------------------------------------------------------------
 * The detection pipeline (3).
 *
 * Everything above this line is the primary display signal on its own, which is
 * all the panel needed in M3. What follows assembles the whole answer: which
 * devices are compute GPUs, which of them drives a display and on what
 * evidence, how much room each one has, and whether this server can be sized
 * against at all.
 *
 * The assembly is a pure function of its inputs, deliberately — the interesting
 * cases are a headless box, a single-GPU desktop, mismatched cards and a
 * corrected profile, and none of those can be produced by owning this machine.
 * ---------------------------------------------------------------------------
 */

/** One compute GPU, with every conclusion this machine supports and its source. */
export interface TopologyGpu {
  index: number;
  name: string | undefined;
  totalGb: number;
  /** Live VRAM, whatever is loaded right now. */
  usedGb: number;
  /** VRAM with nothing loaded. `undefined` when it has never been observed. */
  idleUsedGb: number | undefined;
  display: boolean;
  headroomGb: number;
  /** Why `display` is what it is: `user`, `idle-vram`, `drm-connector:…`. */
  evidence: string[];
  /** Live outputs sysfs sees on this card — the cross-check, not the verdict. */
  connected: string[];
}

/** Where the display verdict came from. `none` means nothing has said yet. */
export type DisplaySource = "profile" | "idle-vram" | "none";

export interface Topology {
  backend: string | undefined;
  /**
   * The server is on another machine (src/endpoint.ts).
   *
   * Carried on the topology because it changes what the *evidence* can be, not
   * only what the UI says: the DRM cross-check reads this machine's sysfs, so
   * against a remote server it is not consulted and contributes nothing.
   */
  remote: boolean;
  /** Compute GPUs only; integrated graphics are counted in `ignoredCount`. */
  gpus: TopologyGpu[];
  ignoredCount: number;
  displaySource: DisplaySource;
  /** Why the DRM cross-check could not be run here, when it could not. */
  connectorNote: string | undefined;
  /**
   * True when the whole extension is running without a topology: the panel,
   * load and unload still work, and everything that needs to know where a model
   * goes does not.
   */
  degraded: boolean;
  reason: string | undefined;
  sizing: SizingSupport;
  unslothVersion: string | undefined;
}

export interface TopologyInputs {
  /** `undefined` when `/api/system` could not be asked at all. */
  system: SystemGpu | undefined;
  /** Idle VRAM observed this session. */
  idle: readonly GpuUsage[];
  profile: MachineProfile | undefined;
  cards: readonly DrmCard[];
  unslothVersion: string | undefined;
  /** Defaults to local: every existing caller is a server on this machine. */
  remote?: boolean;
}

function profileGpu(profile: MachineProfile | undefined, index: number): ProfileGpu | undefined {
  return profile?.gpus.find((gpu) => gpu.index === index);
}

/**
 * The display flags the profile states, or `undefined` when it states none.
 *
 * The same answer `profileDisplayGpus()` gives the panel, read out of the
 * profile the caller passed rather than off disk — `buildTopology` is a pure
 * function, and a pure function that secretly reads a file is neither.
 */
function statedDisplayGpus(profile: MachineProfile | undefined): number[] | undefined {
  if (!profile || profile.gpus.length === 0) return undefined;
  return profile.gpus.filter((gpu) => gpu.display).map((gpu) => gpu.index);
}

/**
 * Assemble a topology from measurements already taken.
 *
 * The display verdict follows one rule: **the user, then the measurement, then
 * nothing.** A profile entry wins because it is a correction of exactly this
 * conclusion; idle VRAM is the measurement; and where neither exists the answer
 * is `false` with no evidence, which the UI draws as "not measured" rather than
 * as "headless" — a guess with no evidence behind it is the thing this pipeline
 * is built to avoid.
 *
 * The DRM connectors never enter that rule. They only ever add evidence to a
 * verdict, or sit visibly beside one they disagree with (`connected`).
 */
export function buildTopology(inputs: TopologyInputs): Topology {
  const sizing = sizingSupport(inputs.unslothVersion);
  const remote = inputs.remote === true;
  const empty = (reason: string): Topology => ({
    backend: inputs.system?.backend,
    remote,
    gpus: [],
    ignoredCount: 0,
    displaySource: "none",
    connectorNote: undefined,
    degraded: true,
    reason,
    sizing: { enabled: false, reason, version: inputs.unslothVersion },
    unslothVersion: inputs.unslothVersion,
  });

  if (!inputs.system) return empty("the server did not answer /api/system");
  const devices = inputs.system.devices;
  if (devices.length === 0) return empty("the server reports no GPUs at all");
  const compute = computeDevices(devices);
  if (compute.length === 0) {
    return empty(`the server reports only integrated graphics (${devices.length} device${devices.length === 1 ? "" : "s"})`);
  }
  if (!inputs.system.available) return empty("the server reports no usable acceleration");

  // The cross-check lines up *every* device against a DRM card, integrated
  // graphics included: an iGPU has a card too, and dropping it from one list
  // and not the other is how the mapping goes off by one.
  //
  // Against a remote server it is not run at all: `/sys/class/drm` describes
  // the machine this code runs on, and a monitor plugged in here says nothing
  // about a GPU in another building.
  const check = crossCheck(devices.map((device) => device.index), remote ? [] : inputs.cards);

  const stated = statedDisplayGpus(inputs.profile);
  const measured = inputs.idle.length > 0 ? displayGpusFromIdle(inputs.idle) : undefined;
  const displaySource: DisplaySource = stated ? "profile" : measured ? "idle-vram" : "none";
  const displayIndices = stated ?? measured ?? [];

  const idleByIndex = new Map(inputs.idle.map((gpu) => [gpu.index, gpu.usedGb]));

  const gpus = compute.map((device): TopologyGpu => {
    const saved = profileGpu(inputs.profile, device.index);
    const display = displayIndices.includes(device.index);
    const card = check.cards.get(device.index);
    const connected = card ? connectedOutputs(card) : [];

    // Evidence is kept as the profile recorded it — that is the record of why
    // the user was shown what they accepted — and fresh connector evidence is
    // added to it rather than replacing it. A profile entry with nothing behind
    // it is the user's own correction, so it says so.
    const evidence = new Set<string>();
    if (display) {
      if (displaySource === "profile") {
        for (const item of saved?.displayEvidence ?? []) evidence.add(item);
        if (evidence.size === 0) evidence.add("user");
      } else {
        evidence.add("idle-vram");
      }
      for (const item of connectorEvidence(device.index, check)) evidence.add(item);
    }

    return {
      index: device.index,
      name: device.name,
      totalGb: device.memoryTotalGb ?? 0,
      usedGb: device.vramUsedGb ?? 0,
      idleUsedGb: idleByIndex.get(device.index) ?? saved?.idleUsedGiB,
      display,
      headroomGb: saved?.headroomGiB ?? defaultHeadroomGb(display),
      evidence: [...evidence],
      connected,
    };
  });

  return {
    backend: inputs.system.backend,
    gpus,
    ignoredCount: devices.length - compute.length,
    displaySource,
    // No note either: the wizard says the cross-check is off because the server
    // is elsewhere, which is the useful sentence. "2 DRM cards for 2 GPUs"
    // would be a complaint about the wrong machine.
    connectorNote: remote ? undefined : check.note,
    remote,
    degraded: false,
    reason: undefined,
    sizing,
    unslothVersion: inputs.unslothVersion,
  };
}

/** What the fit rule needs, from what detection found. */
export function toFacts(gpu: TopologyGpu): GpuFacts {
  return {
    index: gpu.index,
    totalGb: gpu.totalGb,
    // An unmeasured card is treated as holding nothing, which is what an
    // unmeasured *headless* card does hold. A display card without a
    // measurement is flagged by the UI rather than silently over-budgeted.
    idleUsedGb: gpu.idleUsedGb ?? 0,
    display: gpu.display,
    headroomGb: gpu.headroomGb,
  };
}

/** GiB a model may use on this GPU, after idle and headroom. */
export function usableGb(gpu: TopologyGpu): number {
  return budgetGb(toFacts(gpu));
}

type Call = { signal?: AbortSignal; timeoutMs?: number };

/**
 * Ask the machine everything the pipeline needs, then assemble it.
 *
 * Nothing here throws: a query that fails costs the conclusion it fed, and the
 * result says which. The five questions run in parallel because four of them
 * are local reads and the fifth is a request to a server on this machine.
 *
 * Idle VRAM is refreshed on the same rule the panel uses — only when nothing is
 * resident, and merged rather than replaced (`mergeIdleVram`), because "no
 * model loaded" and "the driver has finished handing the memory back" are not
 * the same instant.
 */
export async function detectTopology(client: UnslothClient, call: Call = {}): Promise<Topology> {
  const [system, versions, loaded] = await Promise.all([
    getSystemGpu(client, call).then(
      (result) => result,
      () => undefined,
    ),
    getServerVersions(client, call).then(
      (result) => result.unsloth,
      () => undefined,
    ),
    listLoadedModels(client, call).then(
      (result) => result,
      () => [],
    ),
  ]);

  if (system && loaded.length === 0) {
    const live = computeDevices(system.devices).map((device) => ({
      index: device.index,
      usedGb: device.vramUsedGb ?? 0,
      totalGb: device.memoryTotalGb ?? 0,
    }));
    if (live.length > 0) state.idleGpus = mergeIdleVram(state.idleGpus, live);
  }

  const remote = !isLocalEndpoint(client.baseUrl);

  return buildTopology({
    system,
    idle: state.idleGpus,
    // GPU data recorded against a different server describes hardware we are no
    // longer talking to.
    profile: profileForEndpoint(readMachineProfile().profile, client.baseUrl),
    cards: remote ? [] : readDrmCards(),
    remote,
    unslothVersion: versions,
  });
}
