/**
 * SpecController — browser-side chain-mode speculative decoding.
 *
 * Implements Leviathan-2023 single-token speculative decoding using
 * the Spectra fork of @mlc-ai/web-llm:
 *   - target.spectraPrefillMultiPosition(tokens, positions) for multi-position logits
 *   - target.spectraTruncateKVCache(n) for rollback on rejection
 *   - draft  ditto
 *
 * Key optimization (P4.7-A3 carry-over):
 *   Instead of (target verify γ tokens) + (target single-forward of lastCommit)
 *   per round, we BATCH lastCommit with the next round's drafts.
 *
 *     Round 0   : feed γ drafts to target.        Get γ logits. Verify.
 *     Round N≥1 : feed [lastCommit, γ drafts].    Get γ+1 logits.
 *                 Logit at chunk-pos 0 = predicts d[0] of this round.
 *                 Logit at chunk-pos i (i≥1) = predicts d[i].
 *                 Logit at chunk-pos γ = bonus.
 *
 *   This eliminates one target single-forward per round (saves ~7-15ms on M3 Pro
 *   for the 1.5B target).
 *
 * MVP semantics: greedy only (T=0), chain only (treeWidth = 1).
 */

import type * as webllm from "@mlc-ai/web-llm";

export interface SpecConfig {
  readonly draftLength: number;   // γ
  readonly maxTokens: number;
  readonly eosTokenId?: number;
  readonly onStep?: (e: SpecStepEvent) => void;
}

export interface SpecTiming {
  /** Time spent in the draft proposal loop (γ-1 single forwards). */
  draftLoopMs: number;
  /** Time spent in the target's batched multi-position forward. */
  targetVerifyMs: number;
  /** Time spent in KV truncations (target + draft). */
  kvOpsMs: number;
  /** Time spent in CPU-side argmax over γ × vocab. */
  argmaxMs: number;
  /** Time spent feeding lastCommit to the draft model (the per-round overhead we still pay; target uses carry-over). */
  draftLastCommitMs: number;
  /** Total wall-clock time for this round. */
  totalMs: number;
}

export interface SpecStepEvent {
  readonly round: number;
  readonly drafts: number[];
  readonly accepted: number;        // 0..γ
  readonly committed: number[];     // accepted drafts + resample/bonus
  readonly timing: SpecTiming;
}

export interface SpecResult {
  readonly tokens: number[];        // all committed tokens after prefill
  readonly rounds: number;
  readonly drafted: number;
  readonly accepted: number;
  readonly cumulativeAcceptance: number;
  readonly decodeMs: number;
  readonly prefillMs: number;
  readonly tokensPerSecond: number;
  /** Aggregated per-section timing (mean ms across rounds). */
  readonly meanTiming: SpecTiming;
}

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

  /** Target's last-position logit, valid ONLY at the start of round 0 (after prefill). */
  private targetLastLogit: Float32Array | null = null;

  /** Draft's last-position logit, refreshed each round (we still pay the lastCommit-forward on draft side). */
  private draftLastLogit: Float32Array | null = null;

  /** The token to prepend to round N≥1's target-verify batch (the carry-over). */
  private pendingLastCommit: number | null = null;

  constructor(
    target: webllm.LLMChatPipeline,
    draft: webllm.LLMChatPipeline,
    cfg: SpecConfig,
  ) {
    this.target = target;
    this.draft = draft;
    this.cfg = cfg;
  }

  /** Prefill both models with the prompt. */
  async prefill(promptTokens: number[]): Promise<{ prefillMs: number }> {
    const t0 = performance.now();
    const lastPos = promptTokens.length - 1;
    this.targetLastLogit = await this.target.spectraPrefillMultiPosition(promptTokens, [lastPos]);
    this.draftLastLogit = await this.draft.spectraPrefillMultiPosition(promptTokens, [lastPos]);
    this.pendingLastCommit = null;  // round 0 won't carry-over
    return { prefillMs: performance.now() - t0 };
  }

  async step(): Promise<SpecStepEvent & { eos: boolean }> {
    if (this.draftLastLogit === null) {
      throw new Error("SpecController.prefill() must be called before step()");
    }
    const γ = this.cfg.draftLength;
    const tStart = performance.now();
    const timing: SpecTiming = {
      draftLoopMs: 0, targetVerifyMs: 0, kvOpsMs: 0,
      argmaxMs: 0, draftLastCommitMs: 0, totalMs: 0,
    };

    // 1. Draft autoregressive proposal: d[0] from draftLastLogit, then γ-1 more
    //    via single-token forwards. After loop: draft KV grew by γ-1.
    const tDraft0 = performance.now();
    const drafts: number[] = [argmax(this.draftLastLogit)];
    for (let i = 1; i < γ; i++) {
      const logits = await this.draft.spectraPrefillMultiPosition([drafts[i - 1]], [0]);
      drafts.push(argmax(logits));
    }
    timing.draftLoopMs = performance.now() - tDraft0;

    // 2. Target verify. CARRY-OVER OPTIMIZATION:
    //    - Round 0 (pendingLastCommit === null): feed γ drafts. Get γ logits.
    //      Verify d[0] vs targetLastLogit (from prefill); d[i] vs sliceRow(i-1).
    //    - Round N≥1 (pendingLastCommit set): feed [pendingLastCommit, ...drafts].
    //      Get γ+1 logits. Verify d[0] vs sliceRow(0); d[i] vs sliceRow(i).
    //      The logit at sliceRow(0) is the prediction AFTER pendingLastCommit
    //      = predicts position L+committed_total_so_far+1 = d[0].
    const tVerify0 = performance.now();
    let tgtInputTokens: number[];
    let nLogitRows: number;
    if (this.pendingLastCommit === null) {
      tgtInputTokens = drafts;
      nLogitRows = γ;
    } else {
      tgtInputTokens = [this.pendingLastCommit, ...drafts];
      nLogitRows = γ + 1;
    }
    const positions = Array.from({ length: nLogitRows }, (_, i) => i);
    const tgtLogitsAll = await this.target.spectraPrefillMultiPosition(tgtInputTokens, positions);
    timing.targetVerifyMs = performance.now() - tVerify0;
    const vocab = this.target.spectraGetVocabSize();
    const sliceRow = (rowIdx: number): Float32Array =>
      tgtLogitsAll.subarray(rowIdx * vocab, (rowIdx + 1) * vocab);

    // 3. Verify (greedy). The "predict d[i]" logit row depends on whether we
    //    carried over a token:
    //      - Round 0:   d[0] vs argmax(targetLastLogit), d[i≥1] vs argmax(sliceRow(i-1))
    //      - Round N≥1: d[0] vs argmax(sliceRow(0)),      d[i≥1] vs argmax(sliceRow(i))
    const tArg0 = performance.now();
    const carryOver = this.pendingLastCommit !== null;
    const predForDraft = (i: number): Float32Array =>
      carryOver ? sliceRow(i) : (i === 0 ? this.targetLastLogit! : sliceRow(i - 1));
    let accepted = 0;
    if (argmax(predForDraft(0)) === drafts[0]) {
      accepted = 1;
      for (let i = 1; i < γ; i++) {
        if (argmax(predForDraft(i)) === drafts[i]) accepted++;
        else break;
      }
    }

    // 4. Determine the last committed token (bonus on full accept; resample on partial).
    //    "Bonus row" is the LAST sliceRow (γ in carry-over mode, γ-1 in round 0).
    let lastCommit: number;
    if (accepted === γ) {
      const bonusRowIdx = carryOver ? γ : γ - 1;
      lastCommit = argmax(sliceRow(bonusRowIdx));
    } else {
      // Partial reject at position `accepted`. Logit predicting that position is predForDraft(accepted).
      lastCommit = argmax(predForDraft(accepted));
    }
    timing.argmaxMs = performance.now() - tArg0;
    const committed: number[] = drafts.slice(0, accepted).concat([lastCommit]);

    // 5. KV sync. After verify, target KV holds γ new positions in round 0, or γ+1 new positions
    //    in carry-over mode (the leading position is pendingLastCommit, which is COMMITTED state
    //    from the prior round and must NOT be popped). We want target KV to end with: the carryOver
    //    token (if any) + `accepted` matched drafts. That means popping γ - accepted positions in
    //    BOTH modes. (lastCommit is deferred to next round's carry-over and not in KV yet.)
    //    Draft fed γ-1 tokens; need draft KV at L+accepted. Difference: γ-1 - accepted.
    const tKV0 = performance.now();
    const targetRollback = γ - accepted;
    if (targetRollback > 0) await this.target.spectraTruncateKVCache(targetRollback);
    const draftDiff = γ - 1 - accepted;
    if (draftDiff > 0) {
      await this.draft.spectraTruncateKVCache(draftDiff);
    } else if (draftDiff < 0) {
      // Only when accepted === γ. Feed drafts[γ-1] to draft to reach L+γ.
      await this.draft.spectraPrefillMultiPosition([drafts[γ - 1]], [0]);
    }
    timing.kvOpsMs = performance.now() - tKV0;

    // 6. Stash lastCommit for next round's target carry-over.
    //    For draft, we still need to advance KV by 1 (feed lastCommit) and
    //    refresh draftLastLogit so the next round's draft loop can start.
    this.pendingLastCommit = lastCommit;
    const tFwd0 = performance.now();
    this.draftLastLogit = await this.draft.spectraPrefillMultiPosition([lastCommit], [0]);
    timing.draftLastCommitMs = performance.now() - tFwd0;

    timing.totalMs = performance.now() - tStart;
    const eosId = this.cfg.eosTokenId;
    const eos = eosId !== undefined && committed.some((t) => t === eosId);
    return {
      round: -1, drafts, accepted, committed, timing, eos,
    };
  }

  async generate(promptTokens: number[]): Promise<SpecResult> {
    const { prefillMs } = await this.prefill(promptTokens);
    const tokens: number[] = [];
    let rounds = 0;
    let drafted = 0;
    let accepted = 0;
    const accumTiming: SpecTiming = {
      draftLoopMs: 0, targetVerifyMs: 0, kvOpsMs: 0,
      argmaxMs: 0, draftLastCommitMs: 0, totalMs: 0,
    };
    const tDecodeStart = performance.now();

    while (tokens.length < this.cfg.maxTokens) {
      const rawEvt = await this.step();
      const evt = { ...rawEvt, round: rounds };
      this.cfg.onStep?.(evt);
      rounds += 1;
      drafted += evt.drafts.length;
      accepted += evt.accepted;
      for (const t of evt.committed) {
        if (tokens.length >= this.cfg.maxTokens) break;
        tokens.push(t);
      }
      // Accumulate timings.
      accumTiming.draftLoopMs += evt.timing.draftLoopMs;
      accumTiming.targetVerifyMs += evt.timing.targetVerifyMs;
      accumTiming.kvOpsMs += evt.timing.kvOpsMs;
      accumTiming.argmaxMs += evt.timing.argmaxMs;
      accumTiming.draftLastCommitMs += evt.timing.draftLastCommitMs;
      accumTiming.totalMs += evt.timing.totalMs;
      if (evt.eos) break;
    }

    const decodeMs = performance.now() - tDecodeStart;
    const cumulativeAcceptance = drafted > 0 ? accepted / drafted : 0;
    const tokensPerSecond = decodeMs > 0 ? (tokens.length / decodeMs) * 1000 : 0;
    const meanTiming: SpecTiming = {
      draftLoopMs: accumTiming.draftLoopMs / Math.max(rounds, 1),
      targetVerifyMs: accumTiming.targetVerifyMs / Math.max(rounds, 1),
      kvOpsMs: accumTiming.kvOpsMs / Math.max(rounds, 1),
      argmaxMs: accumTiming.argmaxMs / Math.max(rounds, 1),
      draftLastCommitMs: accumTiming.draftLastCommitMs / Math.max(rounds, 1),
      totalMs: accumTiming.totalMs / Math.max(rounds, 1),
    };
    return {
      tokens, rounds, drafted, accepted, cumulativeAcceptance,
      decodeMs, prefillMs, tokensPerSecond, meanTiming,
    };
  }
}
