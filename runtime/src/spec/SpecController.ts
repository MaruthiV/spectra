/**
 * SpecController — TS port of mlc-llm's BatchDraftAction + BatchVerifyAction.
 *
 * Per spec round (single request, MVP):
 *   1. Draft model produces γ tokens via repeated decode + sample_with_top_p.
 *   2. Target model runs batch_prefill on prefix + drafts (γ logit positions).
 *   3. Call batch_verify TIR PackedFunc with (draft_probs, target_probs, tree).
 *   4. Commit accepted prefix + bonus token; rollback rejected suffix on draft KV.
 *
 * For MVP: chain only (treeWidth = 1). Tree variant = Phase 7.
 *
 * Status: skeleton. step() throws "not implemented".
 */

import type { TVMHandle, SpecConfig } from "../types.js";

export interface SpecStepResult {
  readonly committed: number[];
  readonly rolledBack: number;
  /** Acceptance count this round (0..γ). */
  readonly acceptedCount: number;
  /** Wall-clock duration for this round (ms). */
  readonly latencyMs: number;
}

export class SpecController {
  constructor(
    private readonly target: TVMHandle,
    private readonly draft: TVMHandle,
    private readonly cfg: SpecConfig,
  ) {}

  /** Run one spec round starting from the previously committed token. */
  async step(prevToken: number): Promise<SpecStepResult> {
    void prevToken;
    void this.target;
    void this.draft;
    void this.cfg;
    throw new Error("SpecController.step(): not implemented (stub)");
  }

  /** Reset both KV caches to the prefix length (for re-run on a fresh prompt). */
  async reset(prefixLen: number): Promise<void> {
    void prefixLen;
    throw new Error("SpecController.reset(): not implemented (stub)");
  }
}
