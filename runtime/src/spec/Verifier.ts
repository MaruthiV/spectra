/**
 * Verifier — wraps the `batch_verify` TVM PackedFunc baked into the
 * WebGPU-compiled target model lib.
 *
 * Implementation will:
 *   - resolve `batch_verify` from the loaded model's PackedFunc table
 *   - allocate workspace tensors (draft_probs, draft_tokens, tree, uniform_samples)
 *   - on each call, copy host-side tree structure to device, invoke the
 *     PackedFunc, and read back accepted_count.
 *
 * Status: skeleton.
 */

import type { TVMHandle, TVMTensor, SpecConfig } from "../types.js";

export interface VerifyInput {
  readonly draftProbs: TVMTensor; // shape: (γ × treeW, vocab) f16
  readonly draftTokens: TVMTensor; // shape: (γ × treeW,) i32
  readonly targetProbs: TVMTensor; // shape: (γ × treeW, vocab) f16
  readonly tree: {
    readonly firstChild: TVMTensor;
    readonly nextSibling: TVMTensor;
    readonly parentPtr: TVMTensor;
  };
  readonly uniformSamples: TVMTensor; // shape: (γ × treeW,) f32 in [0,1)
}

export interface VerifyResult {
  readonly acceptedCount: number;
  /**
   * The bonus / resample token sampled from the target model after the last
   * accepted position. Always present — even on full-acceptance, we sample
   * one bonus token from the target_probs at position γ.
   */
  readonly nextTokenId: number;
}

export class Verifier {
  constructor(
    private readonly target: TVMHandle,
    private readonly cfg: SpecConfig,
  ) {}

  /** Resolve the batch_verify PackedFunc from the loaded target lib. */
  async init(): Promise<void> {
    void this.target;
    void this.cfg;
    throw new Error("Verifier.init(): not implemented (stub)");
  }

  /** Run rejection-sampling verification on a chain (MVP) or tree (Phase 7). */
  async verify(input: VerifyInput): Promise<VerifyResult> {
    void input;
    throw new Error("Verifier.verify(): not implemented (stub)");
  }
}
