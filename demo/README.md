# spectra/demo

Phase 3 smoke test for Spectra. Two-button page:

1. **Run with stock prebuilt MLC config** — loads `Qwen2.5-0.5B-Instruct-q4f16_1-MLC`
   from mlc-ai's prebuilt list via `@mlc-ai/web-llm`. Validates that the
   environment + WebGPU + web-llm runtime all work on M3 Pro Chrome. Greedy
   decode, max 64 tokens, 3 prompts.
2. **Run with Spectra-compiled .wasm** — same workflow, but the model_lib URL
   points at our `spectra-qwen2_5_0_5b_webgpu.wasm` (compiled with
   `--enable-subgroups` and Spectra-tuned overrides). Confirms our compile
   output is loadable end-to-end and gives us a baseline tok/s under the
   Spectra-tuned binary.

Both runs use the same q4f16_1 weights from the official HF repo.

## How to run

```bash
cd spectra/demo
pnpm install
# Copy the Spectra-compiled .wasm into public/ (one-time, before first dev run)
mkdir -p public
cp ../build-deps/models/spectra-qwen2_5_0_5b_webgpu.wasm public/
pnpm dev
# Open http://localhost:5173 in Chrome 119+ (Apple Silicon Safari 18+ also works)
```

## What to look for

The page logs both prefill and decode tok/s for each of the three prompts.
After both runs, compare the **mean decode tok/s**:

- **Stock should match WebLLM hosted demo numbers** (~150–200 tok/s for the 0.5B
  on M3 Pro Chrome)
- **Spectra-compiled should be within ±10%** of stock — same model, same weights,
  same kernels modulo the Spectra-tuning overrides. If significantly slower, our
  overrides may have hurt; if same speed, we're cleared to proceed to the
  SpecController work.

## Known limitations

- The "Dump runtime PackedFunc table" button does not work — that introspection
  isn't part of `@mlc-ai/web-llm`'s public API. We confirmed `batch_verify` is
  in the binary via `strings | grep` during Phase 0b.
- The CORS headers in `vite.config.ts` (COEP/COOP) are conservative; remove
  if they cause weight-fetch issues against HuggingFace.
