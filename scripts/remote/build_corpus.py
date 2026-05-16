"""C2 — Build + tokenize the EAGLE training corpus.

Downloads `HuggingFaceH4/ultrachat_200k` (high-quality multi-turn chat distillation
corpus, freely available, ~600 MB raw), applies Qwen2.5's chat template, tokenizes
with Qwen2.5's tokenizer, packs into binary int32 shards, and writes to the
`spectra-eagle-corpus` Modal volume.

Target: ~30M tokens. UltraChat's avg conversation ≈ 500 tokens after the chat
template, so we'll cap at ~60K conversations.

Cost: CPU-only Modal instance. ~15 min wall, <$0.01.

Run:
    modal run scripts/remote/build_corpus.py::build

Output layout on volume `spectra-eagle-corpus`:
    /vol/manifest.json    — {n_shards, n_tokens, vocab_size, tokenizer, dataset, created_at}
    /vol/shard_000.bin    — int32 token IDs, raw little-endian
    /vol/shard_001.bin
    /vol/...
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from modal_eagle import app, corpus_vol, image

# Constants
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
TOKENIZER_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_TOKENS = 30_000_000           # stop after we hit this many tokens
SHARD_TOKENS = 16_000_000            # 16M tokens per shard ≈ 64 MB per .bin (int32)
MAX_CONV_TOKENS = 4096               # skip conversations longer than this
VOLUME_PATH = "/vol"


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    volumes={VOLUME_PATH: corpus_vol},
)
def build() -> dict:
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer

    t0 = time.time()
    out_dir = Path(VOLUME_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[corpus] loading tokenizer: {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    vocab_size = tokenizer.vocab_size
    print(f"[corpus] vocab_size={vocab_size}, eos_token_id={tokenizer.eos_token_id}")

    print(f"[corpus] loading dataset: {DATASET_ID} split={DATASET_SPLIT}")
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)

    # Pack conversations into shards. Each conversation gets EOS appended.
    shard_buf: list[int] = []
    shard_idx = 0
    total_tokens = 0
    total_convs = 0
    skipped_long = 0

    def flush_shard():
        nonlocal shard_buf, shard_idx
        if not shard_buf:
            return
        arr = np.asarray(shard_buf, dtype=np.int32)
        path = out_dir / f"shard_{shard_idx:03d}.bin"
        arr.tofile(path)
        print(f"[corpus] wrote {path.name}: {len(arr):,} tokens ({arr.nbytes/(1024**2):.1f} MB)")
        shard_idx += 1
        shard_buf = []

    for example in ds:
        if total_tokens >= TARGET_TOKENS:
            break
        # ultrachat_200k schema: 'messages' = list of {'role': str, 'content': str}
        messages = example.get("messages")
        if not messages:
            continue
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) > MAX_CONV_TOKENS:
            skipped_long += 1
            continue
        # Append EOS as a separator so the model sees clean conversation boundaries.
        token_ids.append(tokenizer.eos_token_id)

        shard_buf.extend(token_ids)
        total_tokens += len(token_ids)
        total_convs += 1

        if len(shard_buf) >= SHARD_TOKENS:
            flush_shard()

        if total_convs % 5000 == 0:
            elapsed = time.time() - t0
            print(
                f"[corpus] convs={total_convs:>6}  tokens={total_tokens:>10,}  "
                f"shards={shard_idx}  skipped_long={skipped_long}  elapsed={elapsed:.0f}s"
            )

    flush_shard()

    manifest = {
        "dataset": DATASET_ID,
        "dataset_split": DATASET_SPLIT,
        "tokenizer": TOKENIZER_ID,
        "vocab_size": int(vocab_size),
        "eos_token_id": int(tokenizer.eos_token_id),
        "n_shards": shard_idx,
        "n_tokens": total_tokens,
        "n_conversations": total_convs,
        "skipped_long": skipped_long,
        "shard_tokens_target": SHARD_TOKENS,
        "max_conv_tokens": MAX_CONV_TOKENS,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    corpus_vol.commit()
    print("\n[corpus] manifest:")
    print(json.dumps(manifest, indent=2))
    return manifest


@app.local_entrypoint()
def corpus_main() -> None:
    print(f"Building EAGLE training corpus from {DATASET_ID} (target ~{TARGET_TOKENS:,} tokens)…")
    manifest = build.remote()
    print("\n--- Modal returned ---")
    for k, v in manifest.items():
        print(f"  {k}: {v}")
    print(f"\n✓ C2 corpus built. {manifest['n_shards']} shards, {manifest['n_tokens']:,} tokens.")
