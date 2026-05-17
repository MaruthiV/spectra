/**
 * EagleSpecController — EAGLE-3 speculative decoding for Spectra.
 *
 * Differs from SpecController.ts (which is EAGLE-1-style with a separate
 * Qwen2.5-0.5B draft):
 *
 *   • The "draft" is a tiny EAGLE-3 head that consumes the TARGET's multi-layer
 *     hidden state (concat of layer 1, mid=N/2-1, late=N-4 — for Qwen3-1.7B's
 *     28 layers, that's [1, 13, 24]) PLUS the previous token's embedding.
 *   • Target exposes the multi-layer hidden via `spectraBatchPrefillWithAux`
 *     (added in the web-llm fork; underlying PackedFunc `batch_prefill_with_aux`).
 *   • Head outputs over a DRAFT VOCAB SUBSET (32K, vs target's 152K). A d2t
 *     buffer maps draft vocab idx → target vocab offset for verify.
 *
 * V1 design (simpler, slower):
 *   - Each spec round we re-prefill the head from scratch on the growing
 *     "committed + drafted-so-far" sequence (γ separate head prefills per round).
 *   - Target maintains KV across rounds (cheap), but the head's KV is reset
 *     per draft step. Optimisable later via head decode + cache.
 *
 * Per spec round at γ=2:
 *   1. (Head) prefill on (committed) → draft[0]
 *   2. (Head) prefill on (committed + [draft[0]]) → draft[1]
 *   3. (Target) batch_prefill_with_aux on (drafts) with positions [0..γ-1]
 *      to get target logits + aux_hidden at the γ new positions.
 *   4. Greedy verify: compare argmax(target_logits[i]) to draft[i]; on first
 *      mismatch, take target's pick as the bonus token.
 *   5. KV rollback target by (γ - accepted); update committed + aux cache.
 *
 * The d2t mapping: head's argmax returns a draft_vocab_idx. The corresponding
 * target_vocab_idx is `draft_vocab_idx + d2t[draft_vocab_idx]`. This is the
 * predicted token in the target's vocab.
 */

import type * as webllm from "@mlc-ai/web-llm";

export interface EagleSpecConfig {
  draftLength: number;        // γ
  maxTokens: number;
  eosTokenId: number;
  onStep?: (e: EagleStepEvent) => void;
}

export interface EagleStepEvent {
  round: number;
  drafts: number[];           // mapped to target vocab indices
  accepted: number;
  committed: number[];        // new tokens this round (accepted drafts + bonus)
  timing: EagleTiming;
}

export interface EagleTiming {
  headDraftMs: number;
  targetVerifyMs: number;
  kvOpsMs: number;
  totalMs: number;
}

export interface EagleSpecResult {
  tokens: number[];
  rounds: number;
  cumulativeAcceptance: number;
  decodeMs: number;
  tokensPerSecond: number;
  meanTiming: EagleTiming;
}

function argmaxF32(arr: Float32Array, offset = 0, length = arr.length): number {
  let bestIdx = offset;
  let bestVal = arr[offset];
  const end = offset + length;
  for (let i = offset + 1; i < end; i++) {
    if (arr[i] > bestVal) {
      bestVal = arr[i];
      bestIdx = i;
    }
  }
  return bestIdx - offset;
}

export class EagleSpecController {
  /** Cached aux_hidden_concat (target's multi-layer hidden) for all committed positions.
   *  Flat Float32Array of length `committed.length * 3 * hiddenSize`. */
  private auxCache: Float32Array | null = null;
  private auxRowDim: number = 0;       // 3 * hiddenSize
  private hiddenSize: number = 0;      // H, derived from auxRowDim/3
  private committed: number[] = [];
  /** Total tokens currently in target's KV cache. */
  private targetKVLen: number = 0;

  constructor(
    private target: webllm.LLMChatPipeline,
    private eagleHead: webllm.LLMChatPipeline,
    /** d2t buffer: draft_vocab_idx → (target_vocab_idx - draft_vocab_idx) offset */
    private d2t: Int32Array,
    private targetVocabSize: number,
    private cfg: EagleSpecConfig,
  ) {}

  /** Map a draft vocab index back to a target vocab index. */
  private draftToTarget(draftIdx: number): number {
    if (draftIdx < 0 || draftIdx >= this.d2t.length) {
      throw new Error(`draftToTarget: idx ${draftIdx} out of range [0, ${this.d2t.length})`);
    }
    return draftIdx + this.d2t[draftIdx];
  }

  /** Initial prefill: feed prompt to target, cache aux + logits at last position. */
  async prefill(promptTokens: number[]): Promise<void> {
    // Get target logits + aux at all positions. logit_positions=[last] for the
    // initial-token prediction we don't actually use (we'll use the head instead).
    const { auxHidden, auxHiddenDim } = await this.target.spectraBatchPrefillWithAux(
      promptTokens,
      [promptTokens.length - 1],
    );
    this.committed = [...promptTokens];
    this.auxCache = auxHidden;          // length = promptTokens.length * 3H
    this.auxRowDim = auxHiddenDim;      // 3H
    this.hiddenSize = Math.floor(auxHiddenDim / 3);
    this.targetKVLen = promptTokens.length;
  }

  /** One spec round: γ head drafts, 1 target verify, commit. */
  async step(roundIdx: number): Promise<EagleStepEvent & { eos: boolean }> {
    if (!this.auxCache) throw new Error("EagleSpecController.step before prefill");
    const γ = this.cfg.draftLength;
    const tStart = performance.now();
    const timing: EagleTiming = {
      headDraftMs: 0, targetVerifyMs: 0, kvOpsMs: 0, totalMs: 0,
    };

    // ---- 1. γ head prefills to predict drafts -----
    const tDraft0 = performance.now();
    const drafts: number[] = [];
    let curSeqTokens = [...this.committed];
    // Aux row to use for newly drafted positions. EAGLE-3 inference reuses
    // the LAST target-known aux as the "carrier" when the head sees a new
    // position whose true target_hidden is unknown.
    const lastAuxRow = this.auxCache.slice(
      (this.committed.length - 1) * this.auxRowDim,
      this.committed.length * this.auxRowDim,
    );
    let curAux = new Float32Array(this.auxCache);  // copy of committed aux
    for (let i = 0; i < γ; i++) {
      // Build inputs for head prefill on curSeqTokens.
      const inputEmb = await this.target.spectraEmbedTokens(curSeqTokens);
      const headLogits = await this.eagleHead.spectraEagle3Prefill(
        inputEmb,
        curAux,
        curSeqTokens.length,
        this.hiddenSize,
      );
      // Head logits: Float32Array of length draft_vocab_size (=32000).
      const draftIdx = argmaxF32(headLogits);
      const targetIdx = this.draftToTarget(draftIdx);
      drafts.push(targetIdx);

      // Prepare inputs for the NEXT draft step (only if i < γ-1).
      if (i < γ - 1) {
        curSeqTokens = [...curSeqTokens, targetIdx];
        // Append a copy of lastAuxRow as the aux for the new (drafted) position.
        const newAux = new Float32Array(curAux.length + this.auxRowDim);
        newAux.set(curAux);
        newAux.set(lastAuxRow, curAux.length);
        curAux = newAux;
      }
    }
    timing.headDraftMs = performance.now() - tDraft0;

    // ---- 2. Target verify ----
    const tVerify0 = performance.now();
    const positions = drafts.map((_, i) => i);  // 0..γ-1
    const { logits: targetLogitsFlat, auxHidden: newAuxFlat } =
      await this.target.spectraBatchPrefillWithAux(drafts, positions);
    this.targetKVLen += γ;
    timing.targetVerifyMs = performance.now() - tVerify0;

    // ---- 3. Greedy verify ----
    let accepted = 0;
    for (let i = 0; i < γ; i++) {
      const targetPick = argmaxF32(
        targetLogitsFlat,
        i * this.targetVocabSize,
        this.targetVocabSize,
      );
      if (targetPick === drafts[i]) {
        accepted += 1;
      } else {
        break;
      }
    }
    // Determine the +1 "bonus" token: target's pick at position `accepted`
    // (which is either the first mismatch or, if all accepted, the very next).
    const bonusPos = accepted;  // 0..γ
    const bonusToken = argmaxF32(
      targetLogitsFlat,
      bonusPos * this.targetVocabSize,
      this.targetVocabSize,
    );
    // If accepted === γ, bonusPos === γ would be out of our verified range
    // (we only fed γ tokens). In that case we just take target's pick at γ-1's
    // logits (which already predicts γ-th position).
    // For simplicity at V1, when accepted=γ: bonusToken comes from the last
    // verify logit row, which IS the target's prediction for position γ.
    const finalBonus = accepted === γ
      ? argmaxF32(targetLogitsFlat, (γ - 1) * this.targetVocabSize, this.targetVocabSize)
      : bonusToken;

    const newCommits: number[] = drafts.slice(0, accepted);
    newCommits.push(finalBonus);

    // ---- 4. Rollback target KV by (γ - accepted) ----
    const tKV0 = performance.now();
    const rollback = γ - accepted;
    if (rollback > 0) {
      await this.target.spectraTruncateKVCache(rollback);
      this.targetKVLen -= rollback;
    }
    timing.kvOpsMs = performance.now() - tKV0;

    // ---- 5. Update committed + aux cache ----
    // Append accepted drafts' aux from the verify step.
    this.committed.push(...drafts.slice(0, accepted));
    const newAuxAppend = newAuxFlat.subarray(0, accepted * this.auxRowDim);
    const updatedAux = new Float32Array(this.auxCache.length + newAuxAppend.length);
    updatedAux.set(this.auxCache);
    updatedAux.set(newAuxAppend, this.auxCache.length);
    this.auxCache = updatedAux;
    // We do NOT include bonus token's aux because target didn't see it (we'd
    // need another forward pass). Reuse the last available aux next round.
    this.committed.push(finalBonus);

    timing.totalMs = performance.now() - tStart;

    const eos = newCommits.includes(this.cfg.eosTokenId);
    return {
      round: roundIdx,
      drafts,
      accepted,
      committed: newCommits,
      timing,
      eos,
    };
  }

  async generate(promptTokens: number[]): Promise<EagleSpecResult> {
    // Reset both pipelines.
    this.target.resetChat();
    this.eagleHead.resetChat();
    this.committed = [];
    this.auxCache = null;
    this.targetKVLen = 0;

    const t0 = performance.now();
    await this.prefill(promptTokens);

    let rounds = 0;
    let totalDrafts = 0;
    let totalAccepted = 0;
    const generated: number[] = [];
    const timings: EagleTiming[] = [];

    while (generated.length < this.cfg.maxTokens) {
      const ev = await this.step(rounds);
      rounds += 1;
      totalDrafts += ev.drafts.length;
      totalAccepted += ev.accepted;
      generated.push(...ev.committed);
      timings.push(ev.timing);
      this.cfg.onStep?.(ev);
      if (ev.eos) break;
    }
    const decodeMs = performance.now() - t0;

    const meanTiming: EagleTiming = {
      headDraftMs: timings.reduce((s, t) => s + t.headDraftMs, 0) / timings.length,
      targetVerifyMs: timings.reduce((s, t) => s + t.targetVerifyMs, 0) / timings.length,
      kvOpsMs: timings.reduce((s, t) => s + t.kvOpsMs, 0) / timings.length,
      totalMs: timings.reduce((s, t) => s + t.totalMs, 0) / timings.length,
    };

    return {
      tokens: generated,
      rounds,
      cumulativeAcceptance: totalAccepted / Math.max(totalDrafts, 1),
      decodeMs,
      tokensPerSecond: generated.length / (decodeMs / 1000),
      meanTiming,
    };
  }
}
