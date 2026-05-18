# Spectra · First WebGPU EAGLE-3 speculative decoding

> Browser-native speculative-decoding inference for small open LLMs, running entirely in WebGPU.
> **First public WebGPU implementation of EAGLE-3.** Qwen3-1.7B target + AngelSlim's pretrained 136M EAGLE-3 head, q4f16_1 quantized, no server round-trips.

🎬 **Live demo:** https://spectra-blush.vercel.app/

---

## TL;DR

| Metric | Value |
|---|---|
| Qwen3-1.7B greedy baseline (M3 Mac) | **67 tok/s** |
| EAGLE-3 spec decode in browser | **40 tok/s** |
| Acceptance rate α (γ=2, greedy chain) | **0.39** |
| Output correctness | **byte-identical to baseline** |
| Speedup vs baseline | **0.59× (still slower)** |
| Total Modal cost across all training attempts | **~$45** |

I set out to ship a **2× browser LLM speedup** via EAGLE-3. After **two weeks**, **four training runs**, and **$45 in Modal credits**, I have:

- ✅ The first working WebGPU implementation of EAGLE-3 (model definition + runtime + controller)
- ✅ Byte-identical output to the greedy baseline (correctness proven)
- ❌ Slower than baseline (0.59× — not 2×)

This README is the honest postmortem of what I tried, why it didn't hit 2×, and where the architectural wall is. The complete code is in this repo.

---

## The journey

### Training my own EAGLE-1 head (3 attempts, ~$15)

The goal was a custom draft model trained against Qwen2.5-1.5B (later swapped for Qwen3-1.7B).

| Step | Outcome |
|---|---|
| Build 30M-token training corpus from `HuggingFaceH4/ultrachat_200k` | 2 shards, $0.01 |
| Dump target hidden states across the corpus | 29.9M positions across 8 shards, ~100 GB, 3h H100, **$0.84** |
| Define `EagleHead` architecture (~50M params, 1 layer) | done |
| Train 3× (random sampling → sequential → +feature loss) | all converged near loss 1.2 |
| Chain-eval each checkpoint vs target on demo prompts | **α ∈ {0.49, 0.44, 0.45}** |

**The plateau:** all three training attempts ended at α ≈ 0.45 — *below* the off-the-shelf Qwen2.5-0.5B draft (α=0.60). Diagnosis: **exposure bias** from teacher-forced single-step training. At inference, the draft is asked to predict on its own previous predictions, which it never saw during training.

### EAGLE-3 TTT (Test-Time Training), $6.36

Researched EAGLE-3 paper + SpecForge code. Three things distinguished EAGLE-3 from my EAGLE-1 attempts:

1. **TTT rollout** during training — generate γ-step chains during training (not just predict next token), backprop through all γ losses with `0.8^i` weighting
2. **Noise injection** on step-0 input to mimic train/inference distribution mismatch
3. **Multi-layer hidden state fusion** as draft input (not just last layer)

| Step | Outcome |
|---|---|
| Implement TTT rollout in `train_eagle.py` | done |
| Smoke test (1500 steps, ~30 min) | losses descending healthily, $2 |
| Full training: 25K steps × 4 substeps | 1.61h H100, **$6.36** |
| | L0=1.32, L1=1.65, L2=1.74, L3=1.81 (descending substeps = good) |
| | val_top1_match @ 24K = 0.9092 |
| Chain-eval on demo prompts | **α=0.585** (haiku 0.45, spec-decoding explainer 0.62, code 0.69) |

**Better than the EAGLE-1 attempts** (+0.14) but still below the **α≥0.7 threshold** needed for 2× speedup at γ=2.

### Why I stopped training

Read SpecForge source for the canonical recipe. Found **3 high-severity bugs in my training pipeline** vs theirs:

1. **Missing multi-layer hidden fusion** — my head only saw the last-layer hidden state, not the early+mid+late concat that EAGLE-3 specifies
2. **No draft vocab subset** — my lm_head predicted over the full 152K Qwen vocab, not the top-32K subset (which doubles effective parameter density)
3. **Wrong rollout semantics** — small off-by-one in how previous-step predictions feed back into the current-step input

Plus secondary issues: missing loss mask on padding, sub-optimal LR schedule.

**Decision:** Fix all 3 + retrain = ~2-3 more training rounds (~$20-40 + days). **AngelSlim** already shipped pretrained Qwen3-1.7B EAGLE-3 heads with the correct architecture and recipe. The right call was to use theirs and reserve the training budget for if AngelSlim's didn't work.

### AngelSlim head + first WebGPU EAGLE-3 runtime

**Total integration cost: ~$2** (just target wasm compile time on Modal).

#### Architecture

```
                                    ┌──────────────────────────────┐
prompt ──► tokenizer ──► PROMPT     │   web-llm fork (TypeScript) │
                                    │                              │
              ┌─────────────────────┼──────────────────────────┐   │
              ▼                     │                          ▼   │
       ┌────────────┐               │              ┌─────────────────────┐
       │  TARGET    │  spectraBatchPrefillWithAux  │  EagleSpecController │
       │  Qwen3-1.7B├──── logits + multi-layer ───►│  V2: head decode +   │
       │  q4f16_1   │     hidden state concat      │  KV reuse + carry-   │
       │  (WebGPU)  │◄──── verify drafts ─────────│  over batch verify   │
       └────────────┘                              └────────┬────────────┘
                                                            │
                                                            │ shifted embed
                                                            │ + cached aux
                                                            ▼
                                                  ┌──────────────────┐
                                                  │  EAGLE-3 HEAD    │
                                                  │  AngelSlim 136M  │
                                                  │  q4f16_1 (WebGPU)│
                                                  │  draft vocab=32K │
                                                  └──────────────────┘
```

| Step | Outcome |
|---|---|
| Compile Qwen3-1.7B target to WebGPU wasm | 5.9 MB wasm with `spectra_aux_layer_ids=[1,13,24]` |
| AngelSlim head → WebGPU artifact | Wrote `eagle3` model type in `mlc-llm/python/mlc_llm/model/eagle3/` (~350 LoC), registered in MODELS dict, weight loader for AngelSlim → MLC param mapping. Compiled 4.7 MB head wasm, 72 MB q4f16_1 weight shards. |
| Multi-layer hidden state exposure | Added `forward_with_aux` to `Qwen3Model`, `batch_prefill_with_aux` to LM-head model. Exposed via `spectraBatchPrefillWithAux` in web-llm fork. |
| EagleSpecController V1 (re-prefill per draft) | Works end-to-end at **17 tok/s** |
| V2 — Head decode + KV reuse + carry-over verify | **40 tok/s (2.3× over V1)** |

---

## Experiments table

All measured on M3 Mac (M4 Max class), Chrome 134 with `shader-f16`. Same prompt: *"Write a Python function that returns the first n Fibonacci numbers as a list."*

| Setup | Speed | α | vs baseline | Notes |
|---|---|---|---|---|
| Qwen3-1.7B greedy baseline | 67 tok/s | — | 1.00× | stock web-llm decoder |
| V1: re-prefill per draft step | 17 tok/s | 0.37 | 0.25× | naive controller, head re-prefills full sequence each step |
| **V2: head decode + KV reuse** | **40 tok/s** | **0.39** | **0.59×** | + carry-over batch verify (γ+1 tokens) |
| V2 + γ=4 | 34 tok/s | 0.20 | 0.51× | acceptance compounds; γ=2 optimal at this α |
| V2 + q4f32_1 target (fp32 acts) | 30 tok/s | 0.25 | 0.44× | surprised — more precision HURT α |

**The wall:**
- Per-round cost = 25 ms target verify + ~8 ms head decode ≈ 33 ms
- Commits per round = `1 + γ·α` = 1.78 at γ=2, α=0.39
- Per token = 33 / 1.78 = **18.5 ms vs baseline 15 ms** → 1.17× slower

For 2× speedup we need ~4.4 commits per round. With α=0.39 this is unreachable — γ=4 only gives 2.56 commits but with α dropping to 0.20 (rejection compounds).

## What I learned

1. **Pretrained EAGLE-3 heads don't generalize across precision regimes.** AngelSlim trained against fp16 target activations. Switching the browser target to q4f32_1 (fp32 activations) made acceptance WORSE, not better. The head is tuned to specific numerical noise.
2. **Browser inference has a verify floor.** Even with a perfect zero-cost draft model, target verify (~25 ms for 3 tokens through 28 layers) sets the round time. To hit 2× under that floor needs >4 commits per round on average.
3. **Chain decoding is fundamentally weaker than tree decoding.** EAGLE-3's published results (α≈0.7) come from tree decoding with K=4 candidates per step + tree-mask attention. mlc-llm's `PagedKVCache` doesn't support tree-mask. Adding it is a days-long fork.
4. **The cost of "let me train my own" should be measured before committing.** I spent ~$45 + ~2 weeks on the training attempts before pivoting. Reading SpecForge first would have flagged the recipe bugs, and a single $0 search for "pretrained EAGLE-3" would have found AngelSlim's release the same day they posted it.
5. **Forking quickly beats waiting for upstream.** mlc-llm has no `eagle3` model type. Took ~350 LoC to add. Same for multi-layer hidden state exposure in web-llm.

## What I wish I'd find / what's next

- **A pretrained EAGLE-3 head against an int4-quantized target.** Most pretrained heads ship against fp16/bf16 — but production inference is quantized. There's a real research gap here.
- **Tree-mask attention in mlc-llm `PagedKVCache`.** If this lands, my V2 controller can switch to tree decoding overnight and likely hit 1.4–1.8× even at α=0.39.
- **Custom training run with the corrected recipe** (~$10) — sanity-check that my pipeline now produces a reasonable head, even if not headline-grade. Mostly a reproducibility credibility move.

## Open questions for the community

1. **Anyone got an EAGLE-3 / Medusa head pretrained against an int4-quantized target?** The fp16→int4 transfer is the bottleneck I can't fix in user-space.
2. **Anyone working on `tree_mask` attention in mlc-llm or TVM's `PagedKVCache`?** This is the architectural blocker between chain decoding (what I have) and tree decoding (what would actually hit 2×).
3. **Why does my q4f32_1 target give worse acceptance than q4f16_1?** The fp32 activations should be a STRICT superset of fp16's information. My best hypothesis is that AngelSlim's head was specifically tuned for fp16 rounding noise, and removing that noise pushes the head off its training distribution. Has anyone seen this elsewhere?

If any of these resonate, please reach out — handle `@MaruthiV` on Twitter/GitHub.

## Repo structure

```
spectra/
├── demo/                   # Vite + web-llm demo at localhost:5173 (active surface)
│   ├── src/
│   │   ├── EagleSpecController.ts   # V2 spec controller with carry-over
│   │   ├── SpecController.ts        # Earlier 1.45× Qwen2.5 EAGLE-1-style controller (still works)
│   │   └── main.ts
│   ├── public/             # compiled .wasm + AngelSlim head weights
│   └── index.html
├── build-deps/
│   ├── web-llm/            # FORK with spec-decoding internal APIs exposed
│   │   └── src/llm_chat.ts # spectraBatchPrefillWithAux, spectraEagle3Prefill, spectraEagle3Decode, etc.
│   ├── mlc-llm/            # FORK with `eagle3` model type added
│   │   └── python/mlc_llm/model/eagle3/   # eagle3_model.py, eagle3_loader.py, ~350 LoC
│   └── models/             # compiled wasm + weights (gitignored)
├── scripts/
│   ├── remote/             # Modal training scripts (EAGLE-1 + EAGLE-3 TTT)
│   │   ├── train_eagle.py  # EAGLE-3 TTT trainer
│   │   ├── dump_hidden.py  # Target hidden state dump
│   │   └── eval_eagle.py   # Chain α evaluation
│   └── sim/                # Python CPU correctness simulator
└── bench/                  # Playwright γ sweep harness
```

## Build & run locally

```bash
# 1. Demo (the active surface)
cd demo && pnpm install && pnpm dev
# → http://localhost:5173/

# 2. After editing the web-llm fork, rebuild + refresh deps
cd build-deps/web-llm && npm run build
cd ../../demo && rm -rf node_modules pnpm-lock.yaml && pnpm install && pnpm dev
# (pnpm caches file: deps — required to pick up rebuilt fork)

# 3. Recompile target / head wasm (rare — only for compile-flag changes)
# Requires the `spectra` conda env, MLC_LLM_SOURCE_DIR, emcc on PATH, and tvm/lib
# symlinks for mlc_wasm_runtime.bc + wasm_runtime.bc + tvmjs_support.bc + webgpu_runtime.bc.
```

## Credits

- **EAGLE-3 paper:** [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)
- **Pretrained head:** [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3) (Tencent)
- **EAGLE-3 reference recipe:** [SpecForge](https://github.com/SafeAILab/EAGLE/tree/main/train)
- **Browser runtime:** [mlc-ai/web-llm](https://github.com/mlc-ai/web-llm) + [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm) + [TVM](https://github.com/apache/tvm)
- **Target weights:** [mlc-ai/Qwen3-1.7B-q4f16_1-MLC](https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f16_1-MLC) (MLC AI prebuilt)
