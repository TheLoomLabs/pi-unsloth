/**
 * Tier B — empirical verification, and the calibration it leaves behind.
 *
 * Dry-run estimates are a model, and models are wrong at the margins. This is
 * the one place the extension finds out by how much: it loads the sized
 * configuration **once**, reads what the GPUs actually hold, unloads, and
 * stores the difference so later estimates are corrected by a measurement
 * rather than by a fudge factor.
 *
 * Every constraint here exists because this is the only step that can hurt:
 *
 *   - **Never automatic.** The user presses `v`.
 *   - **Never an OOM-probing loop.** One load, one measurement, one unload. No
 *     "try, crash, back off" — the search already happened against the free
 *     estimator, and this only confirms a value believed to fit.
 *   - **A second confirmation when the display GPU is involved**, because a GPU
 *     OOM on the card driving the monitor can take the desktop session with it.
 *   - **Nothing is left loaded.** Whatever happens, including a failure or an
 *     abandoned wait, the GPUs are freed on the way out.
 *
 * The calibration is per machine, per backend, and stamped with the Unsloth
 * version: `compute_bytes` behaviour moves with llama.cpp, so a delta learned
 * against one server is not evidence about the next one.
 */

import type { ExtensionContext } from "@earendil-works/pi-coding-agent";

import type { UnslothClient } from "../api/client.ts";
import type { MemoryEstimate } from "../api/estimate.ts";
import { readComputeUsage } from "../api/system.ts";
import { bytesToGb, type GpuFacts } from "../hardware/budget.ts";
import {
  readMachineProfile,
  writeMachineProfile,
  type MachineProfile,
  type ProfileCalibration,
} from "../hardware/profile.ts";
import { mergeIdleVram } from "../hardware/detect.ts";
import { runTunedLoad, unloadAll } from "../supervisor.ts";
import { state, type GpuUsage } from "../state.ts";
import { showLoadProgress } from "../ui/progress.ts";
import { buildOverrideEntry, type SizingConfig, type SizingTarget } from "./search.ts";

/** Short budget for the questions asked *about* a load, not the load itself. */
const QUERY_TIMEOUT_MS = 10_000;

/**
 * How long to wait for a loaded model's VRAM to *appear*.
 *
 * ⚠ It does not appear when the load returns. Measured on the reference box:
 * `POST /api/inference/load` answered after 3.6 s with `load-progress` already
 * reporting `phase: ready, fraction: 1` — and both cards still at their idle
 * figures. The 5.92 GiB showed up about four seconds *later*.
 *
 * This is M2's unload bug in a mirror: memory is no more granted synchronously
 * than it is released. So the measurement waits for the figure to stop rising,
 * exactly as the unload waits for it to stop falling.
 */
const MEASURE_TIMEOUT_MS = 30_000;
const MEASURE_POLL_MS = 500;
/** A rise smaller than this is the desktop breathing, not weights arriving. */
const MEASURE_EPSILON_GB = 0.05;
/** Polls without a rise that mean the allocation has landed. */
const MEASURE_STABLE_POLLS = 6;

/** Samples beyond this stop moving the mean; the machine is not that variable. */
const MAX_SAMPLES = 8;

/**
 * What a calibration is evidence *about*.
 *
 * Both halves matter and neither is inferred: a delta measured on ROCm says
 * nothing about a CUDA build, and a delta measured against one Unsloth says
 * nothing about the next, because the estimator's compute-buffer arithmetic
 * moves with llama.cpp.
 */
export interface CalibrationStamp {
  unslothVersion: string | undefined;
  backend: string | undefined;
}

export interface Calibration {
  deltaGb: number;
  samples: number;
  stamp: CalibrationStamp;
}

function num(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function str(value: unknown): string | undefined {
  return typeof value === "string" && value !== "" ? value : undefined;
}

/**
 * The stored calibration, **if it is still evidence about this server**.
 *
 * A stamp that does not match is dropped rather than reused: a stale
 * calibration is not a smaller error than no calibration, it is an error with
 * a measurement's confidence attached. An entry with no stamp at all is also
 * dropped — it was written before this version knew what it was measuring, and
 * "probably still true" is not a measurement either.
 */
export function readCalibration(
  profile: MachineProfile | undefined,
  stamp: CalibrationStamp,
): Calibration | undefined {
  const stored = profile?.calibration;
  if (!stored) return undefined;
  const deltaGb = num(stored["estimateDeltaGiB"]);
  if (deltaGb === undefined) return undefined;
  const version = str(stored["unslothVersion"]);
  const backend = str(stored["backend"]);
  if (version === undefined || version !== stamp.unslothVersion) return undefined;
  if (backend !== undefined && stamp.backend !== undefined && backend !== stamp.backend) return undefined;
  return { deltaGb, samples: num(stored["samples"]) ?? 1, stamp: { unslothVersion: version, backend } };
}

/**
 * Fold one measurement into what is already known.
 *
 * A running mean rather than the latest reading: two loads of the same model
 * differ by fragmentation and by whatever the desktop was doing, and the
 * average of the two is a better prior than the more recent accident. The
 * sample count is capped so an old machine's calibration can still move when
 * the truth does.
 */
export function foldCalibration(previous: Calibration | undefined, sampleGb: number, stamp: CalibrationStamp): Calibration {
  if (!previous) return { deltaGb: sampleGb, samples: 1, stamp };
  const weight = Math.min(previous.samples, MAX_SAMPLES - 1);
  return {
    deltaGb: (previous.deltaGb * weight + sampleGb) / (weight + 1),
    samples: Math.min(previous.samples + 1, MAX_SAMPLES),
    stamp,
  };
}

/** The profile entry a calibration becomes. Rounded: this is not a measurement to eight places. */
export function calibrationEntry(calibration: Calibration, existing: ProfileCalibration | undefined): ProfileCalibration {
  return {
    ...existing,
    estimateDeltaGiB: Math.round(calibration.deltaGb * 100) / 100,
    samples: calibration.samples,
    unslothVersion: calibration.stamp.unslothVersion,
    backend: calibration.stamp.backend,
    measuredAt: new Date().toISOString(),
  };
}

/**
 * VRAM the model itself is holding: what the cards report, minus what they
 * held with nothing loaded.
 *
 * Per GPU and clamped at zero, because a card that reports *less* than its own
 * idle baseline has not lent the model negative memory — it has a baseline that
 * was taken while something else was still resident.
 */
export function usageAboveIdle(live: readonly GpuUsage[], idle: readonly GpuUsage[]): number {
  const baseline = new Map(idle.map((gpu) => [gpu.index, gpu.usedGb]));
  return live.reduce((sum, gpu) => sum + Math.max(0, gpu.usedGb - (baseline.get(gpu.index) ?? 0)), 0);
}

export interface Measurement {
  /** GiB the model is holding across the compute GPUs, above idle. */
  measuredGb: number;
  /** GiB the estimate said it would hold. */
  estimatedGb: number;
  /** Measured − estimated. Positive means the estimator was optimistic. */
  deltaGb: number;
  live: GpuUsage[];
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    timer.unref?.();
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      resolve();
    }, { once: true });
  });
}

/**
 * Wait for the model's VRAM to appear, and report the **highest** figure seen.
 *
 * The opposite of the unload's settle in every particular, and for the same
 * underlying reason. There, nothing was allocating, so the *smallest* reading
 * was the truest one; here the weights are still arriving, so the largest is.
 * There, a plateau before the release began looked exactly like the floor;
 * here a plateau before the allocation lands looks exactly like a model that
 * costs nothing — which is what it reported the first time this ran against
 * real hardware: `0.0 GiB held` for a model holding 5.9 GiB.
 *
 * So the plateau rule only applies **after a rise has been seen**. If nothing
 * rises within the window, this returns `undefined` — no measurement — rather
 * than handing back a delta that says the estimator over-charges by the whole
 * size of the model. A calibration is a measurement or it is nothing.
 */
export async function measureLoaded(
  client: UnslothClient,
  estimate: MemoryEstimate,
  idle: readonly GpuUsage[],
  // `pollMs`/`windowMs` exist so the loop itself is a test rather than a
  // thirty-second wait: the behaviour under test is *when* it concludes.
  options: { signal?: AbortSignal; pollMs?: number; windowMs?: number } = {},
): Promise<Measurement | undefined> {
  const call = options.signal ? { signal: options.signal, timeoutMs: QUERY_TIMEOUT_MS } : { timeoutMs: QUERY_TIMEOUT_MS };
  const pollMs = options.pollMs ?? MEASURE_POLL_MS;
  const deadline = Date.now() + (options.windowMs ?? MEASURE_TIMEOUT_MS);

  let highest: { used: number; live: GpuUsage[] } | undefined;
  let stable = 0;
  while (Date.now() < deadline && !options.signal?.aborted) {
    const live = await readComputeUsage(client, call).catch(() => undefined);
    if (live && live.length > 0) {
      const used = usageAboveIdle(live, idle);
      if (!highest || used > highest.used + MEASURE_EPSILON_GB) {
        highest = { used, live };
        stable = 0;
      } else {
        stable++;
      }
    }
    // A figure that has stopped climbing *and* has actually climbed is the
    // model's own memory. Before either, keep looking.
    if (stable >= MEASURE_STABLE_POLLS && highest && highest.used > MEASURE_EPSILON_GB) break;
    await sleep(pollMs, options.signal);
  }

  if (!highest || highest.used <= MEASURE_EPSILON_GB) return undefined;
  const estimatedGb = bytesToGb(estimate.gpuBytes);
  return { measuredGb: highest.used, estimatedGb, deltaGb: highest.used - estimatedGb, live: highest.live };
}

export type VerificationOutcome = "measured" | "load-failed" | "no-measurement" | "cancelled";

export interface VerificationResult {
  outcome: VerificationOutcome;
  measurement: Measurement | undefined;
  calibration: Calibration | undefined;
  /**
   * The calibration reached the machine profile.
   *
   * False when there is no profile to put it in — this file will not *create*
   * one: the profile is the record of what the user accepted in the wizard,
   * and a verification is not a setup. (Creating one no longer suppresses the
   * first-run offer, which asks for GPU data rather than for a file, but
   * inventing a profile out of a measurement would still be wrong.)
   */
  saved: boolean;
  /** The server's own words when the load failed. */
  error: string | undefined;
}

export interface VerificationRequest {
  target: SizingTarget;
  label: string;
  config: SizingConfig;
  estimate: MemoryEstimate;
  stamp: CalibrationStamp;
  signal?: AbortSignal;
}

/** Does this configuration put weights on a card that is driving a monitor? */
export function touchesDisplay(config: SizingConfig, gpus: readonly GpuFacts[]): boolean {
  return config.placement.gpuIds.some((id) => gpus.some((gpu) => gpu.index === id && gpu.display));
}

/**
 * Load once, measure, unload, and store what was learned.
 *
 * The configuration is loaded from an override built **in memory** — nothing is
 * written to the server's settings by verifying, because the user has not
 * pressed `⏎` yet and a verification that quietly changed their auto-switch
 * config would be a worse surprise than a wrong estimate.
 *
 * The unload at the end is unconditional. A verification that fails half way
 * and leaves 30 GiB resident has cost the user exactly what it was meant to
 * protect them from.
 */
export async function runVerification(
  ctx: ExtensionContext,
  client: UnslothClient,
  request: VerificationRequest,
): Promise<VerificationResult> {
  const failure = (outcome: VerificationOutcome, error?: string): VerificationResult => ({
    outcome,
    measurement: undefined,
    calibration: undefined,
    saved: false,
    error,
  });

  // The baseline the measurement is against, and the reason this always starts
  // by freeing the GPUs. "What the model holds" is live VRAM minus what the
  // cards hold with nothing loaded, so the baseline has to be *this* machine
  // right now, with nothing else resident: an earlier reading taken while
  // something else was loaded, or a model still being released, measures the
  // wrong thing in the wrong direction. One unload before is not a retry
  // ladder; it is what makes the one measurement mean anything.
  await unloadAll(ctx, request.signal ? { signal: request.signal } : {});

  // ⚠ The unload only records an idle reading when it *saw memory come back*,
  // and it cannot see that when the GPUs were already free — which is exactly
  // the case here whenever the last thing this session did was unload. Then the
  // baseline is empty, every card counts from zero, and the desktop's own 1.9
  // GiB is billed to the model. Observed on the reference box: 7.8 GiB "held"
  // for a model holding 5.9.
  //
  // So take the reading directly when there is none. Nothing is loaded at this
  // point, which is the one moment a live reading *is* an idle reading — the
  // same thing the display-GPU signal wants.
  let idle = [...state.idleGpus];
  if (idle.length === 0) {
    idle = await readComputeUsage(
      client,
      request.signal ? { signal: request.signal, timeoutMs: QUERY_TIMEOUT_MS } : { timeoutMs: QUERY_TIMEOUT_MS },
    ).catch(() => [] as GpuUsage[]);
    if (idle.length > 0) state.idleGpus = mergeIdleVram(state.idleGpus, idle);
  }
  // Without a baseline there is no measurement, only a number.
  if (idle.length === 0) return failure("no-measurement");

  const override = buildOverrideEntry(request.config);
  const work = runTunedLoad(
    ctx,
    client,
    request.label,
    { modelPath: request.target.modelPath, quant: request.target.quant, override },
    request.signal ? { signal: request.signal } : {},
  );

  try {
    await showLoadProgress(ctx, request.label, override, work);
    const outcome = await work;
    if (!outcome.ok) return failure(outcome.error ? "load-failed" : "cancelled", outcome.error);

    const measurement = await measureLoaded(client, request.estimate, idle, request.signal ? { signal: request.signal } : {});
    if (!measurement) return failure("no-measurement");

    const existing = readMachineProfile();
    const calibration = foldCalibration(
      readCalibration(existing.profile, request.stamp),
      measurement.deltaGb,
      request.stamp,
    );
    const saved =
      existing.profile !== undefined &&
      writeMachineProfile({
        ...existing.profile,
        calibration: calibrationEntry(calibration, existing.profile.calibration),
      }).ok;
    return { outcome: "measured", measurement, calibration, saved, error: undefined };
  } finally {
    // One load, one measurement, then unload — whatever happened in between.
    await unloadAll(ctx, request.signal ? { signal: request.signal } : {});
  }
}
