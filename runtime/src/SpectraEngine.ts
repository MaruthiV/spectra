/**
 * SpectraEngine — top-level entry point for browser-side speculative decoding.
 *
 * Loads two compiled WebGPU model libs (target + draft) via the MLC loader,
 * holds shared tvmjs runtime state, and exposes `generate()` returning a
 * stream of decoded tokens.
 *
 * Status: skeleton. All methods throw "not implemented".
 */

import { MLCLoader } from "./loader/MLCLoader.js";
import { SpecController } from "./spec/SpecController.js";
import type {
  ModelConfig,
  SpecConfig,
  TokenEvent,
  GenerationStats,
  TVMHandle,
} from "./types.js";

export interface SpectraEngineConfig {
  readonly target: ModelConfig;
  readonly draft: ModelConfig;
  readonly spec: SpecConfig;
}

export class SpectraEngine {
  private readonly cfg: SpectraEngineConfig;
  private targetHandle?: TVMHandle;
  private draftHandle?: TVMHandle;
  private controller?: SpecController;

  constructor(cfg: SpectraEngineConfig) {
    this.cfg = cfg;
  }

  /** Load both model libs into a shared WebGPU context. */
  async load(): Promise<void> {
    const loader = new MLCLoader();
    this.targetHandle = await loader.load(this.cfg.target);
    this.draftHandle = await loader.load(this.cfg.draft);
    this.controller = new SpecController(
      this.targetHandle,
      this.draftHandle,
      this.cfg.spec,
    );
  }

  /**
   * Generate tokens for a prompt, yielding each token as it commits.
   * Wraps the spec loop. Returns final stats once EOS / maxTokens hit.
   */
  async *generate(prompt: string): AsyncGenerator<TokenEvent, GenerationStats> {
    if (!this.controller) {
      throw new Error("SpectraEngine.load() must be called before generate()");
    }
    // Suppress unused-private warnings until implementation lands.
    void prompt;
    void this.controller;
    throw new Error("SpectraEngine.generate(): not implemented (stub)");
  }

  /** Free GPU buffers. Idempotent. */
  async dispose(): Promise<void> {
    this.targetHandle = undefined;
    this.draftHandle = undefined;
    this.controller = undefined;
  }
}
