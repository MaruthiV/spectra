/**
 * Shared types for @spectra/runtime.
 *
 * Notes:
 * - `TVMHandle`, `TVMTensor`, `TVMPackedFunc` are local stubs for tvmjs types.
 *   They will be replaced with the real `tvmjs` exports once we wire the loader.
 * - The MVP target is single-request `small_draft` mode with chain (no tree).
 *   `treeWidth` and tree-related fields are present in types so we can extend
 *   to Medusa-style verification later without churning the API.
 */

/** Opaque handle to a loaded TVM-compiled WebGPU model lib. */
export interface TVMHandle {
  readonly __brand: "TVMHandle";
}

/** Opaque handle to a tvmjs `Tensor` allocated on a WebGPU device. */
export interface TVMTensor {
  readonly __brand: "TVMTensor";
  readonly shape: readonly number[];
  readonly dtype: string;
}

/** Opaque handle to a `tvmjs.PackedFunc`. */
export interface TVMPackedFunc {
  readonly __brand: "TVMPackedFunc";
  // Indexed-args call. Will be typed properly once we wire tvmjs.
  (...args: unknown[]): unknown;
}

/** Configuration for one model in the spec pair. */
export interface ModelConfig {
  /** HF or local path with `mlc-chat-config.json` + `tokenizer.json`. */
  readonly modelPath: string;
  /** Path to the compiled `.wasm` lib. */
  readonly libPath: string;
  /** Display name for telemetry. */
  readonly name: string;
  /** Quantization label (q4f16_1 etc.) — informational. */
  readonly quantization: string;
}

/** Configuration for the spec decoding loop. */
export interface SpecConfig {
  /** Drafts per spec step (γ). MVP: 4. Tunable. */
  readonly draftLength: number;
  /** Tree width (Medusa branching). MVP: 1 (chain). */
  readonly treeWidth: number;
  /** Top-p for both target and draft sampling. */
  readonly topP: number;
  /** Temperature. T=0 = greedy verify (sharper acceptance). */
  readonly temperature: number;
  /** Max generated tokens per request. */
  readonly maxTokens: number;
  /** Random seed for reproducibility. Undefined = nondeterministic. */
  readonly seed?: number;
}

/**
 * Default spec configuration suitable for MVP.
 *
 * `draftLength: 2` (not 4) is chosen based on the Phase 2 measurement
 * (sim_8prompts_48tok.json): α ≈ 0.455 on Qwen2.5-1.5B / 0.5B Instruct pair,
 * for which γ=2 is the speedup-optimal value across all plausible browser
 * cost ratios (1.11× – 1.33×). γ=4 only wins when α ≥ 0.7 (e.g. with a
 * custom-distilled draft head). See `phase0_findings.md`.
 */
export const DEFAULT_SPEC_CONFIG: SpecConfig = {
  draftLength: 2,
  treeWidth: 1,
  topP: 0.9,
  temperature: 0.0,
  maxTokens: 256,
};

/** A single token-decoding event observed by callers (UI streaming, telemetry). */
export interface TokenEvent {
  /** Newly committed token ID. */
  readonly tokenId: number;
  /** Decoded UTF-8 fragment for this token. May be empty for incomplete code points. */
  readonly text: string;
  /** Was this token produced by accepting a draft, or by a target-side resample after a rejection? */
  readonly source: "draft-accepted" | "target-resample" | "target-bonus" | "prefill";
  /** Acceptance rate observed over the most recent spec round (0–1). 0 if not from a spec round. */
  readonly acceptanceThisRound: number;
  /** Cumulative accepted-draft / total-drafts ratio across the request so far. */
  readonly cumulativeAcceptance: number;
  /** Wall-clock decode time for this token (ms). */
  readonly latencyMs: number;
}

/** End-of-generation summary statistics. */
export interface GenerationStats {
  readonly tokensGenerated: number;
  /** Wall-clock generate duration (ms), excluding prefill. */
  readonly decodeMs: number;
  readonly prefillMs: number;
  readonly tokensPerSecondDecode: number;
  readonly cumulativeAcceptance: number;
  readonly draftsProposed: number;
  readonly draftsAccepted: number;
  readonly specRounds: number;
}
