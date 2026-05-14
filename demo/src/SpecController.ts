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

// We need tvmjs's Tensor type. Webllm's lib re-exports it through its types,
// but to avoid pulling tvmjs as a peer dep, we'll use a structural type.
type GPUTensor = {
  toArray(): unknown;
  view(shape: number[], dtype?: string): GPUTensor;
  dispose(): void;
  readonly shape: number[];
};

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
    const tgtLogits = await this.target.spectraPrefillMultiPosition(promptTokens, [lastPos]) as unknown as GPUTensor;
    const draftLogits = await this.draft.spectraPrefillMultiPosition(promptTokens, [lastPos]) as unknown as GPUTensor;
    // Copy to CPU; dispose GPU tensors.
    this.targetLastLogit = new Float32Array(tgtLogits.toArray() as Float32Array);
    this.draftLastLogit = new Float32Array(draftLogits.toArray() as Float32Array);
    tgtLogits.dispose();
    draftLogits.dispose();
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
    //    via one-token forwards on draft model.
    const drafts: number[] = [];
    drafts.push(argmax(this.draftLastLogit));
    let newDraftLastLogit: Float32Array | null = this.draftLastLogit;
    for (let i = 1; i < γ; i++) {
      const t = await this.draft.spectraPrefillMultiPosition([drafts[i - 1]], [0]) as unknown as GPUTensor;
      newDraftLastLogit = new Float32Array(t.toArray() as Float32Array);
      t.dispose();
      drafts.push(argmax(newDraftLastLogit));
    }
    // ── At this point draft KV has grown by (γ-1) and its last logit predicts
    //    AFTER draft[γ-2]. We still owe draft KV one more advance (feeding
    //    draft[γ-1]) so that draft is synced to target after the round.
    //    That happens AFTER we decide accepted count, in the rollback step.

    // 2. Target verifies all γ drafts in one batch: feed all γ tokens, get γ
    //    logits. Logit at chunk-position i predicts the token AFTER drafts[i].
    //    - chunk-position 0   = predicts draft[1] (compare to target_argmax)
    //    - chunk-position γ-1 = predicts bonus (the resample candidate after full accept)
    const positions = Array.from({ length: γ }, (_, i) => i);
    const tgtLogitsT = await this.target.spectraPrefillMultiPosition(drafts, positions) as unknown as GPUTensor;
    const tgtLogitsAll = new Float32Array(tgtLogitsT.toArray() as Float32Array);
    tgtLogitsT.dispose();
    // tgtLogitsAll is shape [γ, vocab] flattened to a length γ*vocab Float32Array.
    const vocab = this.target.spectraGetVocabSize();
    // Slice helper:
    const sliceRow = (rowIdx: number): Float32Array =>
      tgtLogitsAll.subarray(rowIdx * vocab, (rowIdx + 1) * vocab);

    // 3. Verify:
    //    - draft[0] vs argmax(targetLastLogit)  (stashed from prior round)
    //    - draft[i] vs argmax(tgtLogitsAll[i-1]) for i=1..γ-1
    //    Bonus = argmax(tgtLogitsAll[γ-1])
    let accepted = 0;
    const targetArgmaxes: number[] = [];
    {
      const tgtPred = argmax(this.targetLastLogit);
      targetArgmaxes.push(tgtPred);
      if (tgtPred === drafts[0]) {
        accepted = 1;
        for (let i = 1; i < γ; i++) {
          const pred = argmax(sliceRow(i - 1));
          targetArgmaxes.push(pred);
          if (pred === drafts[i]) accepted++;
          else break;
        }
      }
    }
    // committed = accepted drafts + (resample or bonus)
    const committed: number[] = drafts.slice(0, accepted);
    let nextTargetLastLogit: Float32Array;
    if (accepted === γ) {
      // Full accept → bonus token from final position.
      const bonusLogits = sliceRow(γ - 1);
      committed.push(argmax(bonusLogits));
      nextTargetLastLogit = new Float32Array(bonusLogits);  // copy for next round
    } else {
      // Reject at position `accepted` → resample from target's prediction
      // at that position. Target's logit predicting that position is:
      //   - if accepted == 0: targetLastLogit (the stashed one)
      //   - else: sliceRow(accepted - 1)
      const rejLogits =
        accepted === 0 ? this.targetLastLogit! : new Float32Array(sliceRow(accepted - 1));
      committed.push(argmax(rejLogits));
      // For next round, target's last logit prediction is the resample logit
      // (we'll re-feed the resampled token in the next round's draft phase if
      // it propagates through the draft model).
      nextTargetLastLogit = new Float32Array(rejLogits);
    }

    // 4. Rollback target KV:
    //    - Target KV grew by γ (we fed γ tokens).
    //    - We want it to grow by (accepted + 1) net. Rollback (γ - accepted - 1)
    //      if accepted < γ; on full accept, we have γ in KV, but the bonus is
    //      not yet in KV (it's just a logit). Actual KV growth so far = γ.
    //      The bonus token has NOT been processed; it just has a logit.
    //      Next round will start by feeding the bonus through the draft. The
    //      bonus will go through target via the next round's drafts. So we
    //      DON'T add the bonus to target KV here. KV growth from this round
    //      = accepted (the matched drafts; we keep their KV).
    //    - On reject at position k=accepted: drafts[0..k-1] are accepted and
    //      remain in KV. drafts[k..γ-1] are wrong and must be rolled back.
    //      Plus the resampled token at position k is NOT yet in KV (we'll feed
    //      it next round). So rollback target by γ - accepted.
    {
      const rollback = γ - accepted;
      if (rollback > 0) await this.target.spectraTruncateKVCache(rollback);
    }

    // 5. Update draft KV to match. Draft fed (γ-1) tokens during the proposal
    //    loop. Net effect after rollback should be: draft KV +accepted.
    //    Currently draft KV = original + (γ-1). We need to:
    //      - If accepted >= γ-1: draft KV is at original + γ-1; need to grow
    //        by (accepted - (γ-1)) more. For γ=2, accepted=2 (full) means
    //        we need +1 more (the original target bonus, but we don't have
    //        the bonus through the draft yet either — let me re-check).
    //
    //    Actually let me re-derive. After step:
    //      target KV = original + accepted (after rollback)
    //      draft  KV should also = original + accepted (for next round sync)
    //
    //    Draft fed γ-1 tokens during proposal (drafts[0]..drafts[γ-2] each one-token forward).
    //    So draft KV = original + (γ-1).
    //
    //    To get draft KV to original + accepted:
    //      diff = accepted - (γ-1)
    //      if diff > 0: feed `diff` more tokens to draft. Specifically:
    //         - drafts[γ-1] (if not already fed) and/or the committed[accepted] (bonus/resample)
    //         - For γ=2 and accepted=2 (full): drafts fed = 1 (only drafts[0]).
    //           We need draft KV at +2 → feed drafts[1] (which was accepted) → +1 → now +2. Good.
    //         - For γ=2 and accepted=1: drafts fed = 1. We need +1. Good as-is.
    //         - For γ=2 and accepted=0: drafts fed = 1. We need 0. Rollback draft by 1.
    //      if diff == 0: no-op.
    //      if diff < 0: rollback draft by -diff.
    {
      const fed = γ - 1;
      const need = accepted;
      const diff = need - fed;
      if (diff > 0) {
        // Feed `diff` more tokens to draft. They come from accepted drafts that
        // haven't been fed yet (drafts[γ-1] is the candidate; for γ=2 this is drafts[1]).
        const tokensToFeed = drafts.slice(γ - 1, γ - 1 + diff);
        let lastDraftLogitsT: GPUTensor | null = null;
        for (const tok of tokensToFeed) {
          if (lastDraftLogitsT) lastDraftLogitsT.dispose();
          lastDraftLogitsT = await this.draft.spectraPrefillMultiPosition([tok], [0]) as unknown as GPUTensor;
        }
        if (lastDraftLogitsT) {
          newDraftLastLogit = new Float32Array(lastDraftLogitsT.toArray() as Float32Array);
          lastDraftLogitsT.dispose();
        }
      } else if (diff < 0) {
        await this.draft.spectraTruncateKVCache(-diff);
        // After rollback the "last logit" we had cached is no longer valid for
        // the rolled-back position. The correct draft-last-logit is now the
        // logit at draft KV position (original + accepted - 1). We don't have
        // it cached; will need to recompute by feeding the last accepted draft
        // again at next round's start.
        //
        // Simplest correct fix: in the diff < 0 case (accepted < γ-1), the
        // committed sequence is drafts[0..accepted-1] + resample. We just had
        // draft KV roll back to position (original + accepted). Now we feed
        // the resample (committed[accepted]) at the start of the NEXT round
        // (treating it like prev_committed for the next draft cycle).
        // For correct draft prediction, we need a fresh draft logit at the
        // position AFTER committed[accepted].
        //
        // To handle this cleanly: feed the resample to draft NOW.
        const resample = committed[committed.length - 1];
        const t = await this.draft.spectraPrefillMultiPosition([resample], [0]) as unknown as GPUTensor;
        newDraftLastLogit = new Float32Array(t.toArray() as Float32Array);
        t.dispose();
      } else {
        // diff == 0 → draft KV is already at the right place.
        // If accepted < γ, our cached newDraftLastLogit is from drafts[γ-2]'s
        // prediction. After commit, we'd want the logit at draft KV's NEW
        // last position, which is (original + accepted - 1).
        // For accepted == γ-1 (which is when diff==0), draft KV currently is
        // at original+(γ-1) = original + accepted. So newDraftLastLogit is
        // correct as-is — it predicts the next token after drafts[γ-2] which
        // is exactly position (original + accepted).
        //
        // Hmm but what does it predict? It predicts the NEXT token. So it's
        // valid as the start of the next round's draft loop.
      }
    }

    // 6. Update state for next round.
    this.targetLastLogit = nextTargetLastLogit;
    this.draftLastLogit = newDraftLastLogit;

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
