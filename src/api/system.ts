/**
 * GPU topology and live VRAM.
 *
 * Reference:. `GET /api/system` is the single source of truth:
 * `devices[].index` **is** the integer that `gpu_ids` / `selected_gpu_ids`
 * accept elsewhere, which is what removes the ROCm-ordinal-vs-DRM-card mapping
 * problem entirely.
 *
 * M2 needs only the live figures the footer shows. The detection pipeline that
 * turns these devices into budgets and a display flag is M4; the one rule
 * borrowed early is the iGPU filter, because a footer that added 30 GB of
 * system RAM to the VRAM total would be lying.
 */

import type { RequestOptions, UnslothClient } from "./client.ts";
import type { GpuUsage } from "../state.ts";

type Call = Pick<RequestOptions, "signal" | "timeoutMs">;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function str(value: unknown): string | undefined {
  return typeof value === "string" && value !== "" ? value : undefined;
}

function num(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/** One entry of `/api/system` → `.gpu.devices`. */
export interface GpuDevice {
  /** The id `gpu_ids` accepts. Not an array position — never treat it as one. */
  index: number;
  name: string | undefined;
  memoryTotalGb: number | undefined;
  /** Live: with nothing loaded this is the desktop's own consumption. */
  vramUsedGb: number | undefined;
  vramFreeGb: number | undefined;
  /** `true` on an iGPU/APU, which borrows system RAM rather than owning VRAM. */
  sharedMemory: boolean;
  unifiedMemory: boolean;
}

export interface SystemGpu {
  /** The server's own verdict on whether it has usable acceleration at all. */
  available: boolean;
  /** `"rocm"` | `"cuda"`. Diagnostics only — never branch on a GPU name. */
  backend: string | undefined;
  devices: GpuDevice[];
}

function parseDevice(value: unknown): GpuDevice | undefined {
  if (!isRecord(value)) return undefined;
  const index = num(value["index"]);
  if (index === undefined) return undefined;
  return {
    index,
    name: str(value["name"]),
    memoryTotalGb: num(value["memory_total_gb"]),
    vramUsedGb: num(value["vram_used_gb"]),
    vramFreeGb: num(value["vram_free_gb"]),
    sharedMemory: value["shared_memory"] === true,
    unifiedMemory: value["unified_memory"] === true,
  };
}

/** `GET /api/system` → `.gpu`. */
export async function getSystemGpu(client: UnslothClient, call: Call = {}): Promise<SystemGpu> {
  const body = await client.get<unknown>("/api/system", call);
  const gpu = isRecord(body) ? body["gpu"] : undefined;
  if (!isRecord(gpu)) return { available: false, backend: undefined, devices: [] };
  const raw = Array.isArray(gpu["devices"]) ? gpu["devices"] : [];
  return {
    available: gpu["available"] === true,
    backend: str(gpu["backend"]),
    devices: raw.map(parseDevice).filter((device): device is GpuDevice => device !== undefined),
  };
}

/**
 * The GPUs a model can actually be placed on.
 *
 * Integrated graphics are dropped by the flags the server already sets, not by
 * guessing from a VRAM-size threshold or a name string.
 */
export function computeDevices(devices: readonly GpuDevice[]): GpuDevice[] {
  return devices.filter((device) => !device.unifiedMemory && !device.sharedMemory);
}

/** Versions `/api/system/hardware` reports. Diagnostics, and the sizing gate. */
export interface ServerVersions {
  /** `versions.unsloth` — the one the sizing gate reads. */
  unsloth: string | undefined;
  torch: string | undefined;
  rocm: string | undefined;
  cuda: string | undefined;
}

/**
 * `GET /api/system/hardware` → `.versions`.
 *
 * ⚠ The rest of that payload is **not** capacity maths: `gpu.vram_total_gb`
 * there reports one card, not the sum. Only the versions are read here, and
 * only two things use them — the wizard's server line and the version gate on
 * sizing.
 */
export async function getServerVersions(client: UnslothClient, call: Call = {}): Promise<ServerVersions> {
  const body = await client.get<unknown>("/api/system/hardware", call);
  const versions = isRecord(body) ? body["versions"] : undefined;
  if (!isRecord(versions)) return { unsloth: undefined, torch: undefined, rocm: undefined, cuda: undefined };
  return {
    unsloth: str(versions["unsloth"]),
    torch: str(versions["torch"]),
    rocm: str(versions["rocm"]),
    cuda: str(versions["cuda"]),
  };
}

/**
 * Live VRAM per compute GPU, in the shape the UI and the fit rule want.
 *
 * One reader, because three callers need exactly this and a second spelling of
 * "which devices count" is how an iGPU ends up in one total and not another:
 * the footer's tick, the unload's settle, and the verification measurement
 * (src/sizing/verify.ts) all come through here.
 */
export async function readComputeUsage(client: UnslothClient, call: Call = {}): Promise<GpuUsage[]> {
  const gpu = await getSystemGpu(client, call);
  return computeDevices(gpu.devices).map((device) => ({
    index: device.index,
    usedGb: device.vramUsedGb ?? 0,
    totalGb: device.memoryTotalGb ?? 0,
  }));
}
