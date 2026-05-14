# `scripts/sim/` — Python correctness oracle

The simulator (`spec_decode_sim.py`) is the **ground-truth oracle** for the
TypeScript `SpecController` in `runtime/src/spec/`. It runs single-token
classic speculative decoding on CPU using HuggingFace transformers, in fp32,
deterministically (with `--seed`).

The TS implementation must match it within fp16 noise:
- ≥99% token-level match at T=0 on `bench/prompts/mt-bench-20.json`
- KL ≤ 0.02 at T=1 on the same prompt set

## Usage

From the spectra/ root, in the `spectra` conda env:

```bash
conda activate spectra
python scripts/sim/spec_decode_sim.py \
    --target Qwen/Qwen2.5-1.5B-Instruct \
    --draft Qwen/Qwen2.5-0.5B-Instruct \
    --gamma 4 \
    --temperature 0.0 \
    --top-p 1.0 \
    --max-tokens 128 \
    --prompts bench/prompts/mt-bench-20.json \
    --output bench/results/sim_mt_bench_20.json \
    --seed 42 \
    --limit 5
```

The `--limit 5` cap is handy for first sanity runs (CPU full-prefill per
spec round = slow; a 128-token completion takes minutes per prompt).

## Output schema

```jsonc
{
  "args": { ... },
  "traces": [
    {
      "prompt": "...",
      "prompt_token_ids": [...],
      "completion": "...",
      "committed_token_ids": [...],
      "steps": [
        {
          "round": 0,
          "drafts_proposed": 4,
          "drafts_accepted": 3,
          "per_position_accept": [true, true, true, false],
          "per_position_target_top": [...],
          "per_position_draft_top": [...],
          "bonus_token_id": 1234,
          "latency_ms": 1234.5
        },
        ...
      ],
      "cumulative_acceptance": 0.612,
      "decode_ms": 12345.6,
      "prefill_ms": 234.5,
      "tokens_per_second_decode": 0.93
    }
  ],
  "summary": { "n_prompts": 5, "mean_acceptance": 0.6, "mean_tokens_per_second_decode": 0.9 }
}
```

## Caveats

- **CPU performance is intentional.** Don't measure tok/s here for any
  comparison purpose. The browser TS impl will be much faster.
- **Re-prefills every spec round.** Simulator simplification — HF's
  `past_key_values` lacks a clean rollback API. Real impl uses paged KV
  with O(branches) rollback. Correctness is unaffected.
- **Vocab-size mismatch is a hard error.** Spec decoding requires identical
  vocab between target and draft (Phase 0a A5 confirmed Qwen2.5 0.5B/1.5B match).
