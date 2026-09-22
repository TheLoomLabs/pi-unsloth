/**
 * In-memory, session-scoped state. Deliberately not persisted: everything here
 * is an observation about *right now*, and a stale observation is worse than
 * none. Durable settings live in the machine profile.
 *
 * One writer per field, roughly: the provider owns the catalogue fields, the
 * supervisor owns the server and model fields, the footer only reads.
 */

/**
 * `offline` is "not running, and nothing is being done about it"; `unreachable`
 * is "we tried to bring it up and could not". The footer draws them
 * differently on purpose — the first is a dim fact, the second needs the user.
 */
export type ServerState = "unknown" | "starting" | "up" | "offline" | "unreachable" | "unauthorized";

/** Live VRAM for one compute GPU, as the footer wants to print it. */
export interface GpuUsage {
  index: number;
  usedGb: number;
  totalGb: number;
}

/**
 * A load in flight.
 *
 * The footer needs only the label and the fraction (`◐ loading … 68%`); the
 * progress overlay needs the phase and the byte counts too. They share one
 * record so there is one poller feeding both, rather than a second one opening
 * whenever the overlay does.
 */
export interface LoadingState {
  label: string;
  /** 0–1, absent until the server reports a total. */
  fraction: number | undefined;
  /**
   * `resolving` → `loading` → `warming up` → `ready`, in the server's words.
   *
   * Optional, unlike `fraction`: the footer never shows these, so a caller that
   * only feeds the footer should not have to spell out three absences.
   */
  phase?: string | undefined;
  loadedBytes?: number | undefined;
  totalBytes?: number | undefined;
}

export interface SessionState {
  /** Last thing we learned about the server, for the footer and `/unsloth`. */
  server: ServerState;
  /** The server's own explanation when it is not up. */
  serverDetail: string | undefined;
  /** How many models the last successful refresh produced. */
  catalogueSize: number;
  /** True when the catalogue on show came from cache rather than the server. */
  catalogueFromCache: boolean;

  /** Pi id of the model the supervisor last ensured, when it is one of ours. */
  activeModelId: string | undefined;
  /** Short display name for the footer — never the whole `name · quant · ctx`. */
  activeModelLabel: string | undefined;
  /** Context the catalogue believes this model has, for the footer's `192K`. */
  activeContextWindow: number | undefined;
  /** Set while an explicit load is running. */
  loading: LoadingState | undefined;
  /** True between `agent_start` and `agent_settled`; pauses the VRAM tick. */
  streaming: boolean;
  /**
   * Is the status line drawn?
   *
   * Session state rather than a setting read on every paint: the toggle has to
   * take effect on the keystroke, and the stored preference is only where it
   * starts from.
   */
  footerVisible: boolean;
  /** Compute GPUs only — an iGPU's shared system RAM is not VRAM. */
  gpus: GpuUsage[];
  /**
   * The same figures, last seen with **nothing loaded**.
   *
   * This is the display-GPU signal, and it is only meaningful at idle — once
   * weights are resident every card looks busy. The panel refreshes it whenever
   * it opens and finds nothing loaded, so the tag survives the rest of the
   * session.
   */
  idleGpus: GpuUsage[];
}

export function createState(): SessionState {
  return {
    server: "unknown",
    serverDetail: undefined,
    catalogueSize: 0,
    catalogueFromCache: false,
    activeModelId: undefined,
    activeModelLabel: undefined,
    activeContextWindow: undefined,
    loading: undefined,
    footerVisible: true,
    streaming: false,
    gpus: [],
    idleGpus: [],
  };
}

/** Module-level singleton: one Pi process, one Unsloth server. */
export const state: SessionState = createState();

/** Reset to first-run conditions. Used by `session_shutdown` so `/new` starts clean. */
export function resetState(): void {
  Object.assign(state, createState());
}
