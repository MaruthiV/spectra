/**
 * @spectra/runtime — public API.
 *
 * This is a skeleton. Nothing here actually runs yet. The implementations
 * are stubs that throw. See ../docs/SPEC_CONTROLLER_DESIGN.md.
 */

export { SpectraEngine } from "./SpectraEngine.js";
export { SpecController } from "./spec/SpecController.js";
export { Verifier } from "./spec/Verifier.js";
export { CacheManager } from "./kv/CacheManager.js";
export { MLCLoader } from "./loader/MLCLoader.js";
export { Profiler } from "./perf/Profiler.js";
export { StepLogger } from "./logging/StepLogger.js";

export type {
  TVMHandle,
  TVMTensor,
  TVMPackedFunc,
  ModelConfig,
  SpecConfig,
  TokenEvent,
  GenerationStats,
} from "./types.js";

export { DEFAULT_SPEC_CONFIG } from "./types.js";
