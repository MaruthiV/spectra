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
  /** Total tokens currently in target's KV cache (= committed.length, invariant). */
  private targetKVLen: number = 0;
  /** Bonus token from prior round whose aux/KV is NOT yet materialized.
   *  Materialized at the start of next step via a 1-token target prefill. */
  private pendingBonus: number | null = null;
  /** V2: how many positions are currently filled in the HEAD's KV cache.
   *  At round R start: should equal `committed.length` for round R-1
   *  (i.e. head saw inputs for the first C_{R-1} positions). */
  private headKVLen: number = 0;

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
    const ret = await this.target.spectraBatchPrefillWithAux(
      promptTokens,
      [promptTokens.length - 1],
    );
    console.log("[EagleSpec] prefill ret:", {
      hasLogits: !!ret?.logits,
      logitsLen: ret?.logits?.length,
      hasAux: !!ret?.auxHidden,
      auxLen: ret?.auxHidden?.length,
      auxDim: ret?.auxHiddenDim,
      promptLen: promptTokens.length,
    });
    if (!ret || !ret.auxHidden || !ret.auxHiddenDim) {
      throw new Error("[EagleSpec] spectraBatchPrefillWithAux returned bad shape: " + JSON.stringify(ret));
    }
    this.committed = [...promptTokens];
    this.auxCache = ret.auxHidden;
    this.auxRowDim = ret.auxHiddenDim;
    this.hiddenSize = Math.floor(ret.auxHiddenDim / 3);
    this.targetKVLen = promptTokens.length;
    // The prefill's last logit row is the target's next-token prediction.
    // That's our first pendingBonus; it'll be materialized in step 0.
    this.pendingBonus = argmaxF32(ret.logits, 0, this.targetVocabSize);
    console.log("[EagleSpec] cached: committed.len=", this.committed.length,
      "auxCache.len=", this.auxCache.length, "auxRowDim=", this.auxRowDim,
      "hiddenSize=", this.hiddenSize, "pendingBonus=", this.pendingBonus);
  }

  /** Get embed for ONE token as Float32Array of length hiddenSize. */
  private async embedOne(token: number): Promise<Float32Array> {
    return await this.target.spectraEmbedTokens([token]);
  }

  /** One spec round V2 (carry-over verify + head KV reuse).
   *  - At round R≥1 start: head_kv has C_{R-1} positions. Extend by (1+accepted_prev)
   *    via decode with TRUE aux (from prior verify) to reach C_R positions. Last
   *    extension's logit = draft[0].
   *  - At round 0 (head_kv == 0): full prefill of length C_0 → head_kv = C_0;
   *    prefill's last logit = draft[0].
   *  - Steps 1..γ-1: decode 1 token each with PLACEHOLDER aux.
   *  - End of round: truncate head_kv by (γ-1) to drop placeholder positions. */
  async step(roundIdx: number): Promise<EagleStepEvent & { eos: boolean }> {
    if (!this.auxCache) throw new Error("EagleSpecController.step before prefill");
    if (this.pendingBonus === null) throw new Error("step: pendingBonus null (prefill not run?)");
    const γ = this.cfg.draftLength;
    const tStart = performance.now();
    const timing: EagleTiming = {
      headDraftMs: 0, targetVerifyMs: 0, kvOpsMs: 0, totalMs: 0,
    };
    const draftVocabSize = this.d2t.length;
    const C = this.committed.length;

    // ---- 1. Head drafts (V2: persistent KV) ----
    // Head input alignment (AngelSlim): at head position i, see
    //   embed(token_{i+1}) + aux(token_i)
    // For round 0 (head_kv empty): full prefill of length C, last logit = draft[0].
    // For round R≥1 (head_kv = C_prev): extend by 1+accepted_prev decodes with TRUE
    //   aux from prior verify; final decode's logit = draft[0]. Then decodes for [1..γ-1].
    const tDraft0 = performance.now();
    const drafts: number[] = [];
    let lastLogits: Float32Array;

    if (this.headKVLen === 0) {
      // Round 0: full prefill. embedTokens = committed[1:] + pendingBonus, aux = auxCache.
      const embedTokens = [...this.committed.slice(1), this.pendingBonus];
      const headLen = embedTokens.length;  // = C
      const inputEmb = await this.target.spectraEmbedTokens(embedTokens);
      lastLogits = await this.eagleHead.spectraEagle3Prefill(
        inputEmb, this.auxCache, headLen, this.hiddenSize,
      );
      this.headKVLen = headLen;
      // Extract LAST position's logits row.
      const lastOff = (headLen - 1) * draftVocabSize;
      const firstDraftIdx = argmaxF32(lastLogits, lastOff, draftVocabSize);
      drafts.push(this.draftToTarget(firstDraftIdx));
    } else {
      // Round R≥1: extend head_kv by (C - this.headKVLen) decodes with TRUE aux.
      const needed = C - this.headKVLen;  // = 1 + accepted_prev
      if (needed <= 0) throw new Error(`V2 head extension impossible: C=${C} headKVLen=${this.headKVLen}`);
      // For each new head position k in [headKVLen, C-1]:
      //   embed = committed[k+1] if k < C-1 else pendingBonus
      //   aux   = committed[k] (TRUE, from auxCache)
      let logitsBuf: Float32Array | null = null;
      for (let k = this.headKVLen; k < C; k++) {
        const embedTok = (k < C - 1) ? this.committed[k + 1] : this.pendingBonus!;
        const embed = await this.target.spectraEmbedTokens([embedTok]);
        const auxRow = this.auxCache.slice(k * this.auxRowDim, (k + 1) * this.auxRowDim);
        logitsBuf = await this.eagleHead.spectraEagle3Decode(embed, auxRow, this.hiddenSize);
      }
      this.headKVLen = C;
      lastLogits = logitsBuf!;  // last decode's logits (= step 0's draft prediction)
      const firstDraftIdx = argmaxF32(lastLogits, 0, draftVocabSize);
      drafts.push(this.draftToTarget(firstDraftIdx));
    }

    // Steps 1..γ-1: decode each with PLACEHOLDER aux.
    // At step i (i≥1), the new head position k has embed=drafts[i-1], aux=placeholder for
    // position k (would be aux of pendingBonus for step 1, or prior draft for step ≥2).
    // We use lastAuxRow (last TRUE row in auxCache) as the carrier placeholder.
    const lastTrueAuxRow = this.auxCache.slice(
      (this.committed.length - 1) * this.auxRowDim,
      this.committed.length * this.auxRowDim,
    );
    for (let i = 1; i < γ; i++) {
      const prevDraft = drafts[i - 1];
      const embed = await this.target.spectraEmbedTokens([prevDraft]);
      const stepLogits = await this.eagleHead.spectraEagle3Decode(
        embed, lastTrueAuxRow, this.hiddenSize,
      );
      this.headKVLen += 1;
      const draftIdx = argmaxF32(stepLogits, 0, draftVocabSize);
      drafts.push(this.draftToTarget(draftIdx));
    }
    timing.headDraftMs = performance.now() - tDraft0;

    // ---- 2. Target verify with carry-over: [pendingBonus, ...drafts] = γ+1 tokens ----
    // - LOCAL position 0 in verify = pendingBonus → logit predicts next-after-pendingBonus = compare to draft[0]
    // - LOCAL position i = draft[i-1]              → logit predicts next-after-draft[i-1] = compare to draft[i]
    // - LOCAL position γ = draft[γ-1]              → logit predicts after-all-drafts = NEW pendingBonus
    const tVerify0 = performance.now();
    const priorBonus = this.pendingBonus;
    const verifyInput = [priorBonus, ...drafts];                  // length γ+1
    const verifyPositions = verifyInput.map((_, i) => i);         // [0..γ]
    const { logits: targetLogitsFlat, auxHidden: newAuxFlat } =
      await this.target.spectraBatchPrefillWithAux(verifyInput, verifyPositions);
    this.targetKVLen += verifyInput.length;                       // +γ+1
    timing.targetVerifyMs = performance.now() - tVerify0;

    // ---- 3. Greedy verify ----
    let accepted = 0;
    for (let i = 0; i < γ; i++) {
      const targetPick = argmaxF32(
        targetLogitsFlat, i * this.targetVocabSize, this.targetVocabSize,
      );
      if (targetPick === drafts[i]) accepted += 1;
      else break;
    }
    // New bonus = target.argmax at the (accepted)-th logit if rejected mid-way,
    // OR the γ-th logit if all accepted (= prediction after the last draft).
    const newBonusRowIdx = accepted < γ ? accepted : γ;
    const newBonus = argmaxF32(
      targetLogitsFlat, newBonusRowIdx * this.targetVocabSize, this.targetVocabSize,
    );

    // ---- 4. Rollback target KV by (γ - accepted) ----
    // Keep [priorBonus, drafts[0..accepted-1]] in KV (1 + accepted tokens).
    const tKV0 = performance.now();
    const rollback = γ - accepted;
    if (rollback > 0) {
      await this.target.spectraTruncateKVCache(rollback);
      this.targetKVLen -= rollback;
    }
    timing.kvOpsMs = performance.now() - tKV0;

    // ---- 5. Commit priorBonus + accepted drafts (aux from verify rows 0..accepted) ----
    this.committed.push(priorBonus);
    this.committed.push(...drafts.slice(0, accepted));
    const commitRows = 1 + accepted;
    const commitAux = newAuxFlat.subarray(0, commitRows * this.auxRowDim);
    const updatedAux = new Float32Array(this.auxCache.length + commitAux.length);
    updatedAux.set(this.auxCache);
    updatedAux.set(commitAux, this.auxCache.length);
    this.auxCache = updatedAux;
    this.pendingBonus = newBonus;

    // ---- 6. V2: truncate head KV by (γ-1) to drop placeholder draft positions ----
    // The first head op of round R extended by (1 + accepted_prev) with TRUE aux.
    // The subsequent (γ-1) decodes used placeholder aux. Drop them so the next
    // round starts from a head_kv that exactly mirrors committed.length.
    const headTrunc = γ - 1;
    if (headTrunc > 0) {
      this.eagleHead.spectraTruncateKVCache(headTrunc);
      this.headKVLen -= headTrunc;
    }

    timing.totalMs = performance.now() - tStart;
    const committedThisRound = [priorBonus, ...drafts.slice(0, accepted)];
    const eos = committedThisRound.includes(this.cfg.eosTokenId);
    return {
      round: roundIdx, drafts, accepted,
      committed: committedThisRound, timing, eos,
    };
  }

  async generate(promptTokens: number[]): Promise<EagleSpecResult> {
    // Reset both pipelines.
    this.target.resetChat();
    this.eagleHead.resetChat();
    this.committed = [];
    this.auxCache = null;
    this.targetKVLen = 0;
    this.pendingBonus = null;
    this.headKVLen = 0;

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
