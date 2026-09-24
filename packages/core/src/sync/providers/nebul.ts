import { existsSync, readdirSync } from "node:fs";
import path from "node:path";
import { z } from "zod";

import type { ExistingModel, SyncProvider, SyncedModel } from "../index.js";
import { MissingReasoningOptionsError } from "../missing-reasoning-options.js";
import { factorBaseModel, modelMetadata } from "./openrouter.js";

const API_ENDPOINT = "https://api.inference.nebul.io/v1/model/info";
const MODELS_DIR = path.join(import.meta.dirname, "..", "..", "..", "..", "..", "models");

// Served org prefix -> models/ metadata namespace (HF org names differ from catalog labs).
// Keys are lowercase; lookups normalize the org the same way (Hugging Face orgs are
// case-insensitive in URLs, so e.g. "qwen/Qwen3.8-27B-FP8" is a valid ID shape).
const ORG_TO_MODEL_PROVIDER: Record<string, string | undefined> = {
  "deepseek-ai": "deepseek",
  google: "google",
  "meta-models": "meta",
  mistralai: "mistral",
  moonshotai: "moonshotai",
  nvidia: "nvidia",
  openai: "openai",
  qwen: "alibaba",
  "zai-org": "zhipuai",
};

// Served IDs whose canonical metadata lives under a differently-named lab entry.
const BASE_MODEL_ALIASES: Record<string, string | undefined> = {
  "mistralai/Mistral-Large-3-675B-Instruct-2512": "mistral/mistral-large-2512",
};

// Catalog scope is general chat models. The public catalog also lists specialized
// document-OCR models by name, and flags internal-only or safety-infrastructure
// entries via display_tags; keep all of those out (not coding/chat targets, and
// models.dev carries no matching lab metadata for them).
const OUT_OF_SCOPE_PATTERNS = [/OCR/i];
const OUT_OF_SCOPE_TAGS = new Set(["Guard Model", "Content Safety", "Private", "Internal"]);

// Fail-closed floor against partial catalog faults. The in-scope chat catalog is
// ~14 models as of 2026-09-24; a truncated response (per-lab serving outage,
// half-written deploy) that still passes the non-empty checks should not be
// treated as the real catalog. Defense in depth: the provider also runs with
// deleteMissing: false, so even a bad catalog cannot prune curated local files.
// Any catalog showing less than half the known-good size is treated as
// structurally incomplete. Raise this deliberately as the catalog grows.
const MIN_CHAT_MODELS = 6;

const EffortValues = z.enum(["none", "minimal", "low", "medium", "high", "xhigh", "max", "default"]);

const ModelInfo = z.object({
  description: z.string().nullable().optional(),
  huggingface_id: z.string().nullable().optional(),
  input_cost_per_1m_tokens: z.number().nullable().optional(),
  output_cost_per_1m_tokens: z.number().nullable().optional(),
  cache_read_input_cost_per_1m_tokens: z.number().nullable().optional(),
  display_tags: z.array(z.string()).nullable().optional(),
  max_input_tokens: z.number().nullable().optional(),
  mode: z.string().nullable(),
  model_type: z.string().nullable(),
  reasoning_efforts: z.array(EffortValues).nullable().optional(),
  superseded_by_model_name: z.string().nullable().optional(),
}).passthrough();

export const NebulEntry = z.object({
  model_info: ModelInfo,
  model_name: z.string().min(1),
}).passthrough();

export const NebulResponse = z.object({
  data: z.array(NebulEntry),
}).passthrough();

export type NebulEntry = z.infer<typeof NebulEntry>;

export const nebul = {
  id: "nebul",
  name: "Nebul",
  modelsDir: "providers/nebul/models",
  // Nebul is a curated provider: only the hand-authored flagship models ship.
  // The catalog is authoritative for their live cost/context, but must never
  // grow or shrink the local set: skipCreates keeps any other in-scope chat
  // model out, and deleteMissing: false keeps a curated model that drops out of
  // the catalog (zai-org/GLM-5.3 has been intermittently absent) instead of
  // silently removing it.
  skipCreates: true,
  deleteMissing: false,
  // Skipped reasoners (fail-closed below) and out-of-catalog chat models are
  // expected; opening missing-model issues for them would request models this
  // provider deliberately does not curate.
  trackMissingModels: false,
  async fetchModels() {
    const response = await fetch(API_ENDPOINT);
    if (!response.ok) {
      throw new Error(`Nebul models request failed: ${response.status} ${response.statusText}`);
    }
    return response.json();
  },
  parseModels(raw) {
    const data = NebulResponse.parse(raw).data;
    // An empty catalog is an upstream fault; syncing it would delete every
    // local model file via the delete-missing pass, so fail loudly instead.
    if (data.length === 0) {
      throw new Error("Nebul returned an empty model catalog");
    }
    // Same failure mode if the response shape drifts and no entry matches the
    // chat-model filter anymore (e.g. renamed model_type/mode values).
    if (!data.some(isCatalogChatModel)) {
      throw new Error("Nebul returned no usable chat models");
    }
    const chatCount = data.filter(isCatalogChatModel).length;
    if (chatCount < MIN_CHAT_MODELS) {
      throw new Error(
        `Nebul returned only ${chatCount} usable chat models (expected at least ${MIN_CHAT_MODELS}); treating the catalog as a partial fault and skipping this run`,
      );
    }
    return data;
  },
  // Unauthenticated /v1/model/info is authoritative for the curated entries'
  // live cost/context. Because the provider runs with skipCreates and
  // deleteMissing: false, translateModel only ever refreshes existing files; any
  // other in-scope chat model is reported via skippedNotice, and a curated model
  // missing from the catalog is retained and reported via missingNotice. Whole-
  // catalog faults still fail closed in parseModels.
  translateModel(entry, context) {
    if (!isCatalogChatModel(entry)) return undefined;
    const id = entry.model_name;
    const info = entry.model_info;
    const existing = context.existing(id);
    // Existing entries must survive incomplete source data — a transient null
    // price or an unresolved alias would otherwise delete the hand-authored
    // TOML on the next run. They keep their authored base_model and cost/limit;
    // only brand-new models need a fully-priced, resolvable source entry.
    const baseModel = existing?.base_model ?? resolveBaseModel(id, info.huggingface_id ?? undefined);
    const cost = info.input_cost_per_1m_tokens != null && info.output_cost_per_1m_tokens != null
      ? {
          input: info.input_cost_per_1m_tokens,
          output: info.output_cost_per_1m_tokens,
          cache_read: info.cache_read_input_cost_per_1m_tokens ?? undefined,
        }
      : existing?.cost;
    const limit = info.max_input_tokens != null ? { context: info.max_input_tokens } : existing?.limit;
    if (existing === undefined && (baseModel === undefined || cost === undefined || limit === undefined)) return undefined;
    // A hand-authored reasoning = false marks a served ID whose lab model reasons
    // but which this host runs with thinking disabled (the catalog reports
    // supports_reasoning = false and no reasoning_efforts). Keep the override and
    // suppress the control/trace machinery entirely: no reasoning_options to
    // require, and no interleaved side channel when no traces are returned.
    const reasoningDisabled = existing?.reasoning === false;
    // Fail closed unless caller control is probe-verified and hand-authored:
    // publishing the catalog's advertised reasoning_efforts unreviewed would
    // sync proven-wrong controls (2026-09-23: it advertised low|medium|high|max
    // for one model, whose engine rejects every value but high). The runner
    // skips the ID so the options can be hand-authored from live probes.
    const isReasoner = !reasoningDisabled && (baseModel !== undefined
      ? modelMetadata(baseModel).reasoning === true
      : existing?.reasoning === true);
    if (isReasoner && existing?.reasoning_options === undefined) {
      throw new MissingReasoningOptionsError(
        id,
        `${id} is a reasoning model, but the catalog entry has no probe-verified reasoning_options; hand-author them instead of trusting the advertised reasoning_efforts`,
      );
    }
    const values = reasoningDisabled
      ? { reasoning: false, interleaved: undefined, reasoning_options: undefined, cost, limit }
      : { interleaved: existing?.interleaved, reasoning_options: existing?.reasoning_options, cost, limit };
    if (baseModel !== undefined) {
      return {
        id,
        model: factorBaseModel(baseModel, values, limit) as SyncedModel,
      };
    }
    // Existing standalone definition whose served alias no longer resolves:
    // keep the authored fields, refreshing only what /model/info still provides.
    return { id, model: { ...existing, ...values } as SyncedModel };
  },
  // Only report in-scope chat models whose base_model could not be resolved; filtered
  // entries (embeddings, rerankers, out-of-scope specialized models, superseded IDs) skip silently.
  sourceID(entry: NebulEntry) {
    return isCatalogChatModel(entry) ? entry.model_name : undefined;
  },
  skippedNotice(ids) {
    if (ids.length === 0) return [];
    return [
      `Nebul serves these in-scope chat models, but only the curated catalog is shipped (skipCreates); not added:`,
      ids.map((id) => `\`${id}\``).join(", "),
    ];
  },
  missingNotice(paths) {
    if (paths.length === 0) return [];
    return [
      `Nebul models absent from the source catalog were retained, not deleted:`,
      paths.map((p) => `\`${p.replace(/\.toml$/, "")}\``).join(", "),
    ];
  },
} satisfies SyncProvider<NebulEntry>;

function isCatalogChatModel(entry: NebulEntry): boolean {
  const info = entry.model_info;
  return info.model_type === "llm" && info.mode === "chat"
    && info.superseded_by_model_name == null && !OUT_OF_SCOPE_PATTERNS.some((pattern) => pattern.test(entry.model_name))
    && !(info.display_tags ?? []).some((tag) => OUT_OF_SCOPE_TAGS.has(tag));
}

// Nebul documents exactly one reasoning control: reasoning_effort. Authored
// options are the only options ever synced: they are live-probe evidence for
// what the served engine accepts, while the catalog's advertised
// reasoning_efforts are probe-proven unreliable (2026-09-23: it advertised
// low|medium|high|max for one model, whose engine rejects every value but
// high). The advertised list is therefore never copied into an entry — a
// reasoner with nothing authored is rejected above, and a non-reasoner carries
// no options. Lab-style toggles or budgets are not supported on this API
// unless a probe of this host shows them.

function resolveBaseModel(servedID: string, huggingfaceID: string | undefined): string | undefined {
  return baseModelCandidates(servedID, huggingfaceID).find(canonicalExists);
}

// existsSync is case-insensitive on Windows/macOS; verify the real on-disk filename case
// so the resolved base_model matches the canonical metadata exactly (and CI on Linux).
function canonicalExists(candidate: string): boolean {
  const file = path.join(MODELS_DIR, `${candidate}.toml`);
  if (!existsSync(file)) return false;
  try {
    return readdirSync(path.dirname(file)).includes(path.basename(file));
  } catch {
    return false;
  }
}

function baseModelCandidates(servedID: string, huggingfaceID: string | undefined): string[] {
  const alias = BASE_MODEL_ALIASES[servedID];
  const servedCandidate = mapOrgToCandidate(servedID);
  const hfCandidate = huggingfaceID === undefined ? undefined : mapOrgToCandidate(huggingfaceID);
  return [
    ...new Set([alias, servedCandidate, hfCandidate, ...quantizationStripped(hfCandidate), ...quantizationStripped(servedCandidate)]).values(),
  ].filter((candidate): candidate is string => candidate !== undefined);
}

function mapOrgToCandidate(id: string): string | undefined {
  const [org, ...modelParts] = id.split("/");
  if (org === undefined || modelParts.length === 0) return undefined;
  const provider = ORG_TO_MODEL_PROVIDER[org.toLowerCase()];
  if (provider === undefined) return undefined;
  return `${provider}/${modelParts.join("/").toLowerCase()}`;
}

// Hosts serve quantized checkpoints (e.g. -FP8, -BF16) of weights whose canonical
// metadata is published for the base precision; try those names without the suffix.
// NVIDIA also prefixes checkpoints with "NVIDIA-", which the metadata names drop.
function quantizationStripped(candidate: string | undefined): string[] {
  if (candidate === undefined) return [];
  const withoutQuant = candidate.replace(/-(fp8|bf16|fp4|int8)$/i, "");
  const withoutPrefix = withoutQuant.replace(/nvidia-/, "");
  return withoutQuant === candidate ? [] : [...new Set([withoutQuant, withoutPrefix])].filter((value) => value !== candidate);
}
