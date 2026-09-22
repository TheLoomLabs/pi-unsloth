/**
 * The Pi provider: every downloaded Unsloth model in `/model`, with no
 * `models.json` to hand-edit.
 *
 * Two jobs, kept apart on purpose:
 *   - `buildModels()` is a pure mapping from API payloads to Pi models, so the
 *     ladders below can be tested without a server;
 *   - `createUnslothProvider()` wires that into pi-ai's `createProvider`.
 */

import type { Api, Model, ModelsStoreEntry, ThinkingLevelMap } from "@earendil-works/pi-ai";
import { createProvider, envApiKeyAuth } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/compat";

import { UnslothClient, UnslothUnreachableError, resolveEndpoint, type UnslothClientOptions } from "./api/client.ts";
import { findOverride, getOverrides, type Overrides } from "./api/lifecycle.ts";
import {
  getInferenceStatus,
  listCachedGguf,
  listLocalModels,
  listModels,
  type CachedGguf,
  type CatalogueEntry,
  type LoadedStatus,
  type LocalModel,
} from "./api/models.ts";
import { readSamplingStore } from "./hardware/profile.ts";
import { hasParams, type SamplingStore } from "./sampling.ts";
import { state } from "./state.ts";

export const PROVIDER_ID = "unsloth";
export const PROVIDER_NAME = "Unsloth Studio";

/**
 * Last rung of the context ladder. Deliberately small: a context window Pi
 * believes in but the server does not honour truncates a conversation
 * mid-flight, whereas an under-estimate only wastes headroom.
 */
export const DEFAULT_CONTEXT_WINDOW = 8192;

/** Generation cap. Not a server-side number — there is no endpoint for it. */
export const DEFAULT_MAX_TOKENS = 8192;

/** Pi's effort levels, minus `off` which is handled separately below. */
const EFFORT_LEVELS = ["minimal", "low", "medium", "high", "xhigh", "max"] as const;

/** Local inference costs nothing per token. */
const FREE = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } as const;

export interface CatalogueInputs {
  baseUrl: string;
  entries: readonly CatalogueEntry[];
  local: readonly LocalModel[];
  cached: readonly CachedGguf[];
  overrides: Overrides;
  /** Capabilities of the loaded model, when one is loaded. */
  status: LoadedStatus | undefined;
  /** Previously published models, used to remember observed capabilities. */
  remembered: readonly Model<Api>[];
  /**
   * Per-model sampling defaults from the profile (`src/sampling.ts`).
   *
   * An input rather than a read, so the mapping stays pure and the interesting
   * case — a stored entry for a model the server no longer has — is a test.
   */
  sampling?: SamplingStore;
}

/**
 * `196608` → `"192K"`. Used only for display names.
 */
export function formatContext(tokens: number): string {
  if (tokens >= 1024 * 1024) {
    const millions = tokens / (1024 * 1024);
    return `${Number.isInteger(millions) ? millions : millions.toFixed(1)}M`;
  }
  if (tokens >= 1024) return `${Math.round(tokens / 1024)}K`;
  return String(tokens);
}

/**
 * Map the server's `reasoning_effort_levels` onto Pi's thinking levels.
 *
 * A level the server does not advertise is `null` — Pi then hides it from the
 * picker instead of sending an effort the server would reject.
 *
 * `off` is deliberately left **unset** rather than null: unset makes Pi send no
 * `reasoning_effort` at all, which is how this server turns thinking off,
 * whereas `null` would remove "off" from the picker altogether.
 */
export function buildThinkingLevelMap(levels: readonly string[] | undefined): ThinkingLevelMap | undefined {
  if (!levels) return undefined;
  const advertised = new Set(levels.map((level) => level.toLowerCase()));
  const map: ThinkingLevelMap = {};
  for (const level of EFFORT_LEVELS) {
    map[level] = advertised.has(level) ? level : null;
  }
  return map;
}

/** Resolve a catalogue entry to its filesystem identity. */
export function findLocal(entry: CatalogueEntry, local: readonly LocalModel[]): LocalModel | undefined {
  return (
    local.find((candidate) => candidate.id === entry.id) ??
    local.find((candidate) => candidate.repoId === entry.id) ??
    local.find((candidate) => candidate.displayName === entry.id) ??
    (entry.displayName ? local.find((candidate) => candidate.displayName === entry.displayName) : undefined)
  );
}

export interface Capabilities {
  reasoning: boolean;
  vision: boolean;
  thinkingLevelMap: ThinkingLevelMap | undefined;
}

/**
 * Capability ladder.
 *
 * `/api/inference/status` describes the **loaded** model only, and there is no
 * per-model equivalent for the rest of the catalogue. So capabilities observed
 * while a model was loaded are remembered in the persisted catalogue and reused
 * once it is cold. Never observed and no other signal → assume nothing: a model
 * wrongly marked non-reasoning still answers, whereas one wrongly marked
 * reasoning gets sent an effort the server may reject.
 */
export function resolveCapabilities(
  entry: CatalogueEntry,
  status: LoadedStatus | undefined,
  cached: CachedGguf | undefined,
  remembered: Model<Api> | undefined,
): Capabilities {
  if (entry.loaded && status) {
    return {
      reasoning: status.supportsReasoning ?? false,
      vision: status.isVision ?? cached?.hasVision ?? false,
      thinkingLevelMap: buildThinkingLevelMap(status.reasoningEffortLevels),
    };
  }
  return {
    reasoning: remembered?.reasoning ?? false,
    vision: cached?.hasVision ?? remembered?.input.includes("image") ?? false,
    thinkingLevelMap: remembered?.thinkingLevelMap,
  };
}

/**
 * Context ladder: tuned override → configured (loaded) → native → default,
 * clamped to whatever ceiling the server reports.
 */
export function resolveContextWindow(entry: CatalogueEntry, overrides: Overrides, local: LocalModel | undefined): number {
  const override = findOverride(overrides, [local?.repoId, local?.path, entry.id], entry.quant);
  const tuned = typeof override?.custom_context_length === "number" ? override.custom_context_length : undefined;
  const chosen = tuned ?? entry.contextLength ?? entry.nativeContextLength ?? DEFAULT_CONTEXT_WINDOW;
  const ceiling = entry.maxContextLength ?? entry.nativeContextLength;
  const clamped = ceiling ? Math.min(chosen, ceiling) : chosen;
  return clamped > 0 ? clamped : DEFAULT_CONTEXT_WINDOW;
}

/** Pure mapping from API payloads to the Pi catalogue. */
export function buildModels(inputs: CatalogueInputs): Model<"openai-completions">[] {
  const rememberedById = new Map(inputs.remembered.map((model) => [model.id, model]));
  const models: Model<"openai-completions">[] = [];

  for (const entry of inputs.entries) {
    const local = findLocal(entry, inputs.local);
    const cached = inputs.cached.find(
      (candidate) => candidate.repoId === entry.id || candidate.repoId === local?.repoId,
    );
    const capabilities = resolveCapabilities(entry, inputs.status, cached, rememberedById.get(entry.id));
    const contextWindow = resolveContextWindow(entry, inputs.overrides, local);
    // The one field on a Pi model that no endpoint can answer for us, so it is
    // published only when the user (or a model card, through them) has said
    // something. An absent `samplingParams` leaves the server's own defaults
    // alone, which is the honest default.
    const sampling = inputs.sampling?.[entry.id]?.params;

    const label = [
      entry.displayName ?? local?.displayName ?? entry.id,
      entry.quant,
      formatContext(contextWindow),
    ]
      .filter((part): part is string => Boolean(part))
      .join(" · ");

    models.push({
      id: entry.id,
      name: label,
      api: "openai-completions",
      provider: PROVIDER_ID,
      // Unsloth serves the OpenAI-compatible surface under /v1.
      baseUrl: `${inputs.baseUrl}/v1`,
      reasoning: capabilities.reasoning,
      input: capabilities.vision ? ["text", "image"] : ["text"],
      cost: { ...FREE },
      contextWindow,
      maxTokens: Math.min(DEFAULT_MAX_TOKENS, contextWindow),
      // llama.cpp's OpenAI surface takes `system`, not `developer`.
      compat: { supportsDeveloperRole: false },
      ...(capabilities.thinkingLevelMap ? { thinkingLevelMap: capabilities.thinkingLevelMap } : {}),
      ...(sampling && hasParams(sampling) ? { samplingParams: { ...sampling } } : {}),
    });
  }

  return models;
}

/**
 * Fetch everything the mapping needs.
 *
 * The catalogue itself is required; every other endpoint is an enrichment, so
 * one of them being unavailable costs a capability, not the model list.
 */
export async function fetchCatalogue(
  client: UnslothClient,
  stored: Readonly<ModelsStoreEntry> | undefined,
  signal: AbortSignal,
): Promise<Model<"openai-completions">[]> {
  const call = { signal };
  const entries = await listModels(client, call);

  const [local, cached, overrides, status] = await Promise.all([
    listLocalModels(client, call).then((result) => result.models, () => []),
    listCachedGguf(client, call).then(
      (result) => result,
      () => [],
    ),
    getOverrides(client, call).then(
      (result) => result,
      () => ({}) as Overrides,
    ),
    entries.some((entry) => entry.loaded)
      ? getInferenceStatus(client, call).then(
          (result) => result,
          () => undefined,
        )
      : Promise.resolve(undefined),
  ]);

  return buildModels({
    baseUrl: client.baseUrl,
    entries,
    local,
    cached,
    overrides,
    status,
    remembered: stored?.models ?? [],
    sampling: readSamplingStore(),
  });
}

/**
 * A client bound to this Pi session.
 *
 * Pi's credential store is the source of truth for the key — that is where
 * `/login` writes — so it is asked first, on every request rather than once, so
 * that logging in mid-session takes effect without a restart. The endpoint
 * resolution in `client.ts` (env, then `auth.json`) remains the backstop for
 * contexts where the registry has nothing.
 */
export function createSessionClient(
  ctx: { modelRegistry: { getApiKeyForProvider(provider: string): Promise<string | undefined> } },
  /** Overrides the resolved endpoint — the wizard testing an address the user just typed. */
  baseUrl?: string,
): UnslothClient {
  return new UnslothClient({
    ...(baseUrl !== undefined ? { baseUrl } : {}),
    apiKey: async () => {
      try {
        const key = await ctx.modelRegistry.getApiKeyForProvider(PROVIDER_ID);
        if (key) return key;
      } catch {
        // Registry not ready, or no credential — fall through to the file.
      }
      return resolveEndpoint().apiKey;
    },
  });
}

export interface ProviderOptions {
  /** Overrides the resolved endpoint. Used by tests and by `UNSLOTH_BASE_URL`. */
  baseUrl?: string;
}

/**
 * Build the provider.
 *
 * `createProvider` restores `context.stored` before fetching and persists the
 * result afterwards, which is what keeps `/model` populated when the server is
 * down. Streaming is pi's own OpenAI-completions implementation.
 */
export function createUnslothProvider(options: ProviderOptions = {}) {
  const baseUrl = new UnslothClient(options.baseUrl !== undefined ? { baseUrl: options.baseUrl } : {}).baseUrl;

  return createProvider<"openai-completions">({
    id: PROVIDER_ID,
    name: PROVIDER_NAME,
    baseUrl: `${baseUrl}/v1`,
    auth: { apiKey: envApiKeyAuth(`${PROVIDER_NAME} API key`, ["UNSLOTH_API_KEY"]) },
    models: [],
    api: openAICompletionsApi(),
    fetchModels: async (context) => {
      const clientOptions: UnslothClientOptions = {};
      if (options.baseUrl !== undefined) clientOptions.baseUrl = options.baseUrl;
      // Pi owns the credential; prefer it over anything read off disk.
      const credential = context.credential;
      if (credential?.type === "api_key" && typeof credential.key === "string" && credential.key !== "") {
        clientOptions.apiKey = credential.key;
      }
      const client = new UnslothClient(clientOptions);

      try {
        const models = await fetchCatalogue(client, context.stored, context.signal);
        state.server = "up";
        state.serverDetail = undefined;
        state.catalogueSize = models.length;
        state.catalogueFromCache = false;
        return models;
      } catch (error) {
        if (context.signal.aborted) throw error;

        // The server being down is an expected state, not a failure: keep the
        // last known catalogue so `/model` still works offline.
        const previous = (context.stored?.models ?? []).filter(
          (model): model is Model<"openai-completions"> =>
            model.provider === PROVIDER_ID && model.api === "openai-completions",
        );
        state.server = error instanceof UnslothUnreachableError ? "offline" : "unauthorized";
        state.serverDetail = error instanceof Error ? error.message : String(error);
        state.catalogueSize = previous.length;
        state.catalogueFromCache = true;
        return previous;
      }
    },
  });
}
