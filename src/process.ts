/**
 * The local half of the server lifecycle: starting Unsloth Studio, and
 * stopping the `llama-server` process that actually holds the VRAM.
 *
 * Why this file exists at all, when everything else in this extension asks the
 * server rather than the machine:
 *
 *   - `POST /api/inference/unload` clears Unsloth's "active model" state but
 *     leaves `llama-server` running with the full weights resident (verified
 *     over 60 s of polling — SETTINGS.md in the Unsloth-Api repo). Freeing the
 *     GPUs means ending that process. There is no endpoint for it.
 *   - Unsloth reports a **stale** active model after that process dies, so the
 *     only honest answer to "is a model really resident?" is a process check.
 *
 * Nothing here knows anything about the hardware — no card index, no VRAM
 * figure, no GPU name. It knows one process name and two signals.
 */

import { spawn } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";

/**
 * The inference process Unsloth starts. Not a hardware constant: it is the
 * llama.cpp binary's name, the same string `piloff` matched on.
 */
const INFERENCE_PROCESS = "llama-server";

/** How long a terminate round waits for a polite exit before escalating. */
const TERM_GRACE_MS = 8_000;
/** How long to wait after SIGKILL for the supervisor to (not) respawn. */
const RESPAWN_GRACE_MS = 3_000;
/** Terminate rounds. The supervisor respawns, so one round is not enough. */
const TERMINATE_ROUNDS = 3;

export interface ProcessInfo {
  pid: number;
  /** Full command line, NUL separators already normalised to spaces. */
  command: string;
}

/**
 * Is this command line a loaded inference server?
 *
 * Both halves matter: the name alone also matches a server started with no
 * model, while ` -m ` is what says weights are resident. Pure, so the matching
 * rule is testable without processes.
 */
export function isInferenceCommand(command: string): boolean {
  return command.includes(INFERENCE_PROCESS) && command.includes(" -m ");
}

/**
 * Process enumeration is the one genuinely platform-specific thing here. Linux
 * is read directly from `/proc`; everything else shells out to `ps`, which is
 * untested.
 */
export function processListingSupported(): boolean {
  return process.platform === "linux" || process.platform === "darwin";
}

function listFromProc(): ProcessInfo[] {
  const found: ProcessInfo[] = [];
  let entries: string[];
  try {
    entries = readdirSync("/proc");
  } catch {
    return found;
  }
  for (const entry of entries) {
    if (!/^\d+$/.test(entry)) continue;
    let command: string;
    try {
      command = readFileSync(`/proc/${entry}/cmdline`, "utf8").replace(/\0/g, " ").trim();
    } catch {
      // The process exited between readdir and read, or is not ours to see.
      continue;
    }
    if (command !== "" && isInferenceCommand(command)) {
      found.push({ pid: Number(entry), command });
    }
  }
  return found;
}

async function listFromPs(): Promise<ProcessInfo[]> {
  const output = await new Promise<string>((resolve) => {
    try {
      const child = spawn("ps", ["-axo", "pid=,args="], { stdio: ["ignore", "pipe", "ignore"] });
      let stdout = "";
      child.stdout.setEncoding("utf8");
      child.stdout.on("data", (chunk: string) => {
        stdout += chunk;
      });
      child.on("error", () => resolve(""));
      child.on("close", () => resolve(stdout));
    } catch {
      resolve("");
    }
  });

  const found: ProcessInfo[] = [];
  for (const line of output.split("\n")) {
    const match = /^\s*(\d+)\s+(.*)$/.exec(line);
    if (!match) continue;
    const command = match[2]!.trim();
    if (isInferenceCommand(command)) found.push({ pid: Number(match[1]), command });
  }
  return found;
}

/** Every resident inference process. Empty when none, or when we cannot look. */
export async function listInferenceProcesses(): Promise<ProcessInfo[]> {
  if (process.platform === "linux") return listFromProc();
  if (process.platform === "darwin") return listFromPs();
  return [];
}

/** Is a model really resident? See the file header on stale server state. */
export async function isInferenceRunning(): Promise<boolean> {
  return (await listInferenceProcesses()).length > 0;
}

/** A cancellable, unref'd sleep. Never keeps the process alive by itself. */
function delay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal?.aborted) return resolve();
    const timer = setTimeout(finish, ms);
    timer.unref?.();
    signal?.addEventListener("abort", finish, { once: true });
    function finish(): void {
      clearTimeout(timer);
      signal?.removeEventListener("abort", finish);
      resolve();
    }
  });
}

function signalPid(pid: number, signal: NodeJS.Signals): void {
  try {
    process.kill(pid, signal);
  } catch {
    // Already gone, or not ours to signal. Both are answered by the next poll.
  }
}

export interface TerminateOptions {
  signal?: AbortSignal;
  /** Test seams. The defaults are the ones `piloff` arrived at empirically. */
  rounds?: number;
  graceMs?: number;
  respawnGraceMs?: number;
}

export interface TerminateResult {
  /** Processes that were running when we started. */
  found: number[];
  /** Still running when we gave up. Empty means the VRAM is free. */
  survivors: number[];
}

/**
 * End every inference process, and keep ending the ones Unsloth respawns.
 *
 * SIGTERM, wait, SIGKILL, then look again — because Unsloth's supervisor starts
 * a replacement, and a single round leaves the GPUs exactly as busy as before.
 */
export async function terminateInferenceProcesses(options: TerminateOptions = {}): Promise<TerminateResult> {
  const rounds = options.rounds ?? TERMINATE_ROUNDS;
  const graceMs = options.graceMs ?? TERM_GRACE_MS;
  const respawnGraceMs = options.respawnGraceMs ?? RESPAWN_GRACE_MS;
  const found = new Set<number>();

  for (let round = 0; round < rounds; round++) {
    if (options.signal?.aborted) break;
    const running = await listInferenceProcesses();
    if (running.length === 0) break;
    for (const entry of running) found.add(entry.pid);

    for (const entry of running) signalPid(entry.pid, "SIGTERM");

    const deadline = Date.now() + graceMs;
    while (Date.now() < deadline) {
      await delay(500, options.signal);
      if (options.signal?.aborted) break;
      if ((await listInferenceProcesses()).length === 0) break;
    }

    for (const entry of await listInferenceProcesses()) signalPid(entry.pid, "SIGKILL");
    await delay(respawnGraceMs, options.signal);
  }

  return { found: [...found], survivors: (await listInferenceProcesses()).map((entry) => entry.pid) };
}

export interface SpawnResult {
  started: boolean;
  /** Why not, in the words of whatever refused. */
  error: string | undefined;
}

/**
 * Where the server is started from.
 *
 * Not cosmetic. Unsloth resolves its `models_dir` **relative to the server's
 * working directory**, so a server that inherits Pi's cwd scans
 * `<whatever project the user opened>/models`. Every model in their real
 * models folder then disappears from `/api/models/local`, and this extension
 * reports the one thing it can see — accurately, and very confusingly — as
 * "not on disk". Observed on the reference box on 2026-09-22, with the 27B
 * intact in `~/models` the whole time.
 *
 * The home directory is the one location that does not move with the session.
 * A user who wants a different models folder registers it through
 * `POST /api/models/scan-folders`, which is stored server-side and does not
 * depend on cwd at all.
 */
export function launchDirectory(): string {
  return homedir();
}

/** Test seam: the spawn used by `startServerProcess`. */
export interface StartOptions {
  spawnProcess?: typeof spawn;
}

/**
 * Start Unsloth Studio, detached.
 *
 * Detached and fully disconnected on purpose: the server must outlive this Pi
 * session (a second window should find it already up), and it must never
 * inherit our stdio, where its output would corrupt the TUI. It must not
 * inherit our **cwd** either — see `launchDirectory`.
 */
export function startServerProcess(command: readonly string[], options: StartOptions = {}): SpawnResult {
  const spawnProcess = options.spawnProcess ?? spawn;
  const [binary, ...args] = command;
  if (!binary) return { started: false, error: "no launch command configured" };
  try {
    const child = spawnProcess(binary, args, { detached: true, stdio: "ignore", cwd: launchDirectory() });
    // A spawn failure (ENOENT) surfaces asynchronously; without a listener it
    // would be an unhandled 'error' event and take the session down with it.
    child.on("error", () => {});
    child.unref();
    return { started: true, error: undefined };
  } catch (error) {
    return { started: false, error: error instanceof Error ? error.message : String(error) };
  }
}
