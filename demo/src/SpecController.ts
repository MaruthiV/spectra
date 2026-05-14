/**
 * SpecController — browser-side chain-mode speculative decoding (γ=2, greedy).
 *
 * Implements Leviathan-2023 single-token speculative decoding using
 * @mlc-ai/web-llm's Spectra fork additions:
 *   - target.spectraPrefillMultiPosition(tokens, positions) for multi-position logits
 *   - target.spectraTruncateKVCache(n) for rollback on rejection
 *   - draft  ditto
 *
 * MVP semantics (greedy, T=0):
 *   - Draft autoregressively proposes γ tokens.
 *   - Target forwards γ drafts in one batch, returning γ logits.
 *   - Verify position i by comparing target_argmax(logits[i-1]) with draft[i].
 *     For position 0, use the stashed target-last-position logit from before
 *     this round (set by init() and refreshed after each round).
 *   - On first reject k: commit drafts[0..k-1] + target-resample at k.
 *     On full accept: commit drafts[0..γ-1] + bonus from target_argmax(logits[γ-1]).
 *
 * Open simplifications carried over from the Python simulator
 * (see scripts/sim/spec_decode_sim.py):
 *   - Greedy only (T=0). Sampled spec decoding uses rejection_sampling
 *     with probs ratios; we'll add it in P4.7.
 *   - Chain mode only (treeWidth = 1). Tree verification is Phase 7.
 *
 * Correctness oracle: outputs at T=0 must match the target model's greedy
 * decode at ≥99% of positions on MT-Bench-20.
 */

import type * as webllm from "@mlc-ai/web-llm";

export interface SpecConfig {
  readonly draftLength: number;   // γ
  readonly maxTokens: number;
  readonly eosTokenId?: number;
  /** Logger callback for per-step events. */
  readonly onStep?: (e: SpecStepEvent) => void;
}

export interface SpecStepEvent {
  readonly round: number;
  readonly drafts: number[];
  readonly accepted: number;        // 0..γ
  readonly committed: number[];     // accepted drafts + resample/bonus
  readonly latencyMs: number;
}

export interface SpecResult {
  readonly tokens: number[];        // all committed tokens after prefill
  readonly rounds: number;
  readonly drafted: number;
  readonly accepted: number;
  readonly cumulativeAcceptance: number;
  /** Wall-clock decoding time in milliseconds (excludes prefill). */
  readonly decodeMs: number;
  /** Wall-clock prefill time in milliseconds. */
  readonly prefillMs: number;
  /** Mean decode tok/s (accepted + resampled / decodeMs * 1000). */
  readonly tokensPerSecond: number;
}

/** Greedy argmax of a Float32Array (logits → token id). */
function argmax(arr: Float32Array): number {
  let best = 0;
  let bestV = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > bestV) { bestV = arr[i]; best = i; }
  }
  return best;
}

export class SpecController {
  private cfg: SpecConfig;
  private target: webllm.LLMChatPipeline;
  private draft: webllm.LLMChatPipeline;

  /** Target's last-position logit, updated after each spec round. */
  private targetLastLogit: Float32Array | null = null;

  /** Draft's last-position logit. */
  private draftLastLogit: Float32Array | null = null;

  constructor(
    target: webllm.LLMChatPipeline,
    draft: webllm.LLMChatPipeline,
    cfg: SpecConfig,
  ) {
    this.target = target;
    this.draft = draft;
    this.cfg = cfg;
  }

  /**
   * Prime both models with the prompt. Stashes the last-position logit from
   * each as the starting point for the spec loop.
   */
  async prefill(promptTokens: number[]): Promise<{ prefillMs: number }> {
    const t0 = performance.now();
    const lastPos = promptTokens.length - 1;
    console.log("[Spectra] prefilling target...");
    this.targetLastLogit = await this.target.spectraPrefillMultiPosition(promptTokens, [lastPos]);
    console.log("[Spectra] target prefill OK, draft logit len =", this.targetLastLogit.length);
    console.log("[Spectra] prefilling draft...");
    this.draftLastLogit = await this.draft.spectraPrefillMultiPosition(promptTokens, [lastPos]);
    console.log("[Spectra] draft prefill OK, draft logit len =", this.draftLastLogit.length);
    return { prefillMs: performance.now() - t0 };
  }

  /**
   * Run one spec round.
   *
   * Precondition: targetLastLogit and draftLastLogit are set (from previous
   * round's bonus or from prefill). Both KV caches are at the same length L.
   *
   * Postcondition: both KVs grown by (accepted + 1); targetLastLogit and
   * draftLastLogit refreshed to the new last-position logits.
   */
  async step(): Promise<SpecStepEvent & { eos: boolean }> {
    if (this.targetLastLogit === null || this.draftLastLogit === null) {
      throw new Error("SpecController.prefill() must be called before step()");
    }
    const γ = this.cfg.draftLength;
    const t0 = performance.now();

    // 1. Draft autoregressive proposal: d[0] from draftLastLogit, then γ-1 more
    //    via one-token forwards on draft model. After loop: draft KV at L+γ-1.
    const drafts: number[] = [argmax(this.draftLastLogit)];
    for (let i = 1; i < γ; i++) {
      const logits = await this.draft.spectraPrefillMultiPosition([drafts[i - 1]], [0]);
      drafts.push(argmax(logits));
    }

    // 2. Target verifies γ drafts in one batch. Feeds γ tokens; gets γ logits
    //    at chunk positions 0..γ-1. Chunk position i predicts the token AFTER
    //    drafts[i] (so cp 0 predicts draft[1], cp γ-1 predicts bonus).
    //    After this: target KV at L+γ.
    const positions = Array.from({ length: γ }, (_, i) => i);
    const tgtLogitsAll = await this.target.spectraPrefillMultiPosition(drafts, positions);
    const vocab = this.target.spectraGetVocabSize();
    const sliceRow = (rowIdx: number): Float32Array =>
      tgtLogitsAll.subarray(rowIdx * vocab, (rowIdx + 1) * vocab);

    // 3. Verify (greedy):
    //    - drafts[0] vs argmax(targetLastLogit)  (stashed from prior round)
    //    - drafts[i] vs argmax(tgtLogitsAll[i-1]) for i=1..γ-1
    let accepted = 0;
    if (argmax(this.targetLastLogit) === drafts[0]) {
      accepted = 1;
      for (let i = 1; i < γ; i++) {
        if (argmax(sliceRow(i - 1)) === drafts[i]) accepted++;
        else break;
      }
    }

    // 4. Determine the last committed token (bonus on full accept; resample on partial).
    let lastCommit: number;
    if (accepted === γ) {
      // Full accept → bonus from logit at chunk pos γ-1.
      lastCommit = argmax(sliceRow(γ - 1));
    } else {
      // Partial reject at position `accepted`. The target's logit predicting
      // that position is:
      //   - if accepted == 0: the stashed targetLastLogit
      //   - else: sliceRow(accepted - 1)
      const rejLogits =
        accepted === 0 ? this.targetLastLogit : sliceRow(accepted - 1);
      lastCommit = argmax(rejLogits);
    }
    const committed: number[] = drafts.slice(0, accepted).concat([lastCommit]);

    // 5. Sync KVs to length L+accepted, then forward `lastCommit` through both.
    //    Target: currently at L+γ. Rollback (γ - accepted) → L+accepted.
    //    Draft:  currently at L+γ-1. Difference to L+accepted = (γ-1) - accepted.
    //      - If positive: rollback draft by that amount.
    //      - If zero (i.e. accepted == γ-1): already in sync.
    //      - If negative (i.e. accepted == γ, full accept): feed drafts[γ-1]
    //        through draft to reach L+γ (= L+accepted).
    if (γ - accepted > 0) await this.target.spectraTruncateKVCache(γ - accepted);
    const draftDiff = γ - 1 - accepted;
    if (draftDiff > 0) {
      await this.draft.spectraTruncateKVCache(draftDiff);
    } else if (draftDiff < 0) {
      // Only possible when accepted === γ. Feed |diff| = 1 token = drafts[γ-1].
      await this.draft.spectraPrefillMultiPosition([drafts[γ - 1]], [0]);
    }

    // 6. Both KVs now at L+accepted. Feed lastCommit through both → both at
    //    L+accepted+1 = L + committed.length, and get the correct next-position
    //    logits for round N+1.
    this.targetLastLogit = await this.target.spectraPrefillMultiPosition([lastCommit], [0]);
    this.draftLastLogit = await this.draft.spectraPrefillMultiPosition([lastCommit], [0]);

    const latencyMs = performance.now() - t0;
    const eosId = this.cfg.eosTokenId;
    const eos = eosId !== undefined && committed.some((t) => t === eosId);

    const evt: SpecStepEvent & { eos: boolean } = {
      round: -1, // caller fills in
      drafts,
      accepted,
      committed,
      latencyMs,
      eos,
    };
    this.cfg.onStep?.(evt);
    return evt;
  }

  /**
   * Top-level generation. Prefills, then runs spec rounds until maxTokens
   * or EOS.
   */
  async generate(promptTokens: number[]): Promise<SpecResult> {
    const { prefillMs } = await this.prefill(promptTokens);
    const tokens: number[] = [];
    let rounds = 0;
    let drafted = 0;
    let accepted = 0;
    const tDecodeStart = performance.now();

    while (tokens.length < this.cfg.maxTokens) {
      const evt = await this.step();
      rounds += 1;
      drafted += evt.drafts.length;
      accepted += evt.accepted;
      for (const t of evt.committed) {
        if (tokens.length >= this.cfg.maxTokens) break;
        tokens.push(t);
      }
      if (evt.eos) break;
    }
    const decodeMs = performance.now() - tDecodeStart;
    const cumulativeAcceptance = drafted > 0 ? accepted / drafted : 0;
    const tokensPerSecond = decodeMs > 0 ? (tokens.length / decodeMs) * 1000 : 0;
    return {
      tokens,
      rounds,
      drafted,
      accepted,
      cumulativeAcceptance,
      decodeMs,
      prefillMs,
      tokensPerSecond,
    };
  }
}
