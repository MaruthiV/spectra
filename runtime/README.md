# @spectra/runtime

> Browser-native speculative decoding for small open LLMs in WebGPU.

**Status: skeleton only. No implementation yet.** This package will host the
TypeScript port of mlc-llm's `BatchDraftAction` + `BatchVerifyAction`
orchestration, calling the WebGPU-compiled `batch_prefill` + `batch_verify`
TIR ops via tvmjs PackedFuncs.

## Design

See `../docs/SPEC_CONTROLLER_DESIGN.md` for the full design.

## Layout

- `src/SpectraEngine.ts` — top-level engine; loads target + draft `.wasm`
- `src/spec/SpecController.ts` — drives the spec decoding loop
- `src/spec/Verifier.ts` — wraps the WebGPU `batch_verify` PackedFunc
- `src/kv/CacheManager.ts` — two-model KV cache + rollback
- `src/loader/MLCLoader.ts` — wraps tvmjs `instantiateWasmFromCache`
- `src/decoder/Sampling.ts` — greedy / top-p / top-k primitives
- `src/perf/Profiler.ts` — per-step latency telemetry
- `src/logging/StepLogger.ts` — structured per-step logs (for debugging + viz)

## Build

```bash
pnpm install
pnpm typecheck
pnpm build
```

(Tests + browser demo come later — see `../demo/` once it exists.)
