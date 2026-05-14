/**
 * CacheManager — manages KV cache state for two models (target + draft).
 *
 * Wraps the TVM-compiled paged KV cache exposed by mlc-llm's `create_tir_paged_kv_cache`.
 * Provides:
 *   - prefill(model, tokens): grow KV by N positions
 *   - rollbackKV(model, n): truncate the most recent n positions (after rejection)
 *   - clear(): reset both caches (between requests)
 *
 * The exact PackedFunc names for KV ops (e.g. `kv_cache_pop_n`) will be
 * resolved at init from each model's lib. mlc-llm exposes these via the
 * compiled module's PackedFunc table.
 *
 * Status: skeleton.
 */

import type { TVMHandle } from "../types.js";

export type ModelRole = "target" | "draft";

export class CacheManager {
  constructor(
    private readonly target: TVMHandle,
    private readonly draft: TVMHandle,
  ) {}

  /** Initialize KV caches for both models. */
  async init(maxContext: number): Promise<void> {
    void this.target;
    void this.draft;
    void maxContext;
    throw new Error("CacheManager.init(): not implemented (stub)");
  }

  /** Truncate the most-recent N positions of the named model's KV. */
  async rollbackKV(model: ModelRole, n: number): Promise<void> {
    void model;
    void n;
    throw new Error("CacheManager.rollbackKV(): not implemented (stub)");
  }

  /** Reset both caches to length 0. */
  async clear(): Promise<void> {
    throw new Error("CacheManager.clear(): not implemented (stub)");
  }
}
