/**
 * MLCLoader — loads a Spectra-compiled `.wasm` model lib via tvmjs.
 *
 * Reads the model's `mlc-chat-config.json`, fetches the `.wasm` and the
 * quantized parameter shards (described by `ndarray-cache.json`), instantiates
 * the WebGPU runtime, and returns an opaque TVMHandle ready for use by
 * SpectraEngine.
 *
 * Status: skeleton. load() throws.
 */

import type { TVMHandle, ModelConfig } from "../types.js";

export class MLCLoader {
  /** Load one model and return an opaque handle. */
  async load(cfg: ModelConfig): Promise<TVMHandle> {
    void cfg;
    throw new Error("MLCLoader.load(): not implemented (stub)");
  }
}
