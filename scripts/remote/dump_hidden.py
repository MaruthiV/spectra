"""C3 — Dump target model's hidden states + top-K logits over the training corpus.

Reads the int32 corpus shards from `spectra-eagle-corpus`, packs tokens into
fixed-length 1024-token windows, runs Qwen2.5-1.5B-Instruct (fp16, flash-attn)
on each batch, and writes per-token (hidden_state[t], target_topK_logits[t])
tuples to `spectra-eagle-dump`.

Why we save what we save:
  - hidden_state[t]   — input feature for EAGLE head at training position t+1
  - token_id[t]       — also input feature (combined with hidden via fc_merge)
  - topK ids+logits[t]— the target's prediction for position t+1 (KD label)
At training time, sample (hiddens[i], tokens[i]) → predict topK_logits[i].

Cost ceiling: Qwen-1.5B fp16 + flash-attn on H100, batch=32 × seq=1024.
  Compute: ~30M tokens / ~50k tok/s = ~10 min. Wall + IO + writes: ~30-60 min.
  $ : ~0.6h × $3.95 = $2-3.

Storage: 30M tokens × (1536 fp16 + K*int32 + K*fp16) ≈ ~100 GB on volume.

Run:
    modal run scripts/remote/dump_hidden.py::dump

Output (per dump shard ≈ 4M tokens):
    /vol/manifest.json     — {n_shards, n_tokens, hidden_dim, topk, ...}
    /vol/shard_000_tokens.bin       int32  [N]
    /vol/shard_000_hiddens.bin      fp16   [N, 1536]
    /vol/shard_000_topk_ids.bin     int32  [N, 64]
    /vol/shard_000_topk_logits.bin  fp16   [N, 64]
    /vol/shard_001_*.bin
    ...
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from modal_eagle import app, corpus_vol, dump_vol, image

CORPUS_PATH = "/corpus"
DUMP_PATH = "/dump"

TARGET_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SEQ_LEN = 1024
BATCH_SIZE = 32                  # fits comfortably on H100 80GB for 1.5B fp16
TOPK = 64                        # number of (token_id, logit) pairs to save per position
DUMP_SHARD_TOKENS = 4_000_000    # ~14 GB per shard (1536*2 + 64*6 = 3456 B/token)


@app.function(
    image=image,
    gpu="H100",
    timeout=2 * 60 * 60,         # 2h hard cap (expected ~30-60 min)
    volumes={CORPUS_PATH: corpus_vol, DUMP_PATH: dump_vol},
)
def dump() -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    corpus_dir = Path(CORPUS_PATH)
    dump_dir = Path(DUMP_PATH)
    dump_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load corpus manifest + shards -----
    corpus_manifest = json.loads((corpus_dir / "manifest.json").read_text())
    shard_files = sorted(corpus_dir.glob("shard_*.bin"))
    print(f"[dump] corpus: {corpus_manifest['n_tokens']:,} tokens across {len(shard_files)} shards")
    print(f"[dump] tokenizer={corpus_manifest['tokenizer']}, vocab={corpus_manifest['vocab_size']}")

    # Concatenate corpus shards into one flat token array (we'll re-pack into SEQ_LEN windows).
    token_arrays = [np.fromfile(p, dtype=np.int32) for p in shard_files]
    all_tokens = np.concatenate(token_arrays)
    n_tokens = len(all_tokens)
    print(f"[dump] loaded {n_tokens:,} tokens into memory ({all_tokens.nbytes/(1024**2):.1f} MB)")

    # Trim to multiple of SEQ_LEN for clean batching.
    usable_tokens = (n_tokens // SEQ_LEN) * SEQ_LEN
    all_tokens = all_tokens[:usable_tokens]
    n_seqs = usable_tokens // SEQ_LEN
    print(f"[dump] using {usable_tokens:,} tokens → {n_seqs:,} sequences of len {SEQ_LEN}")
    print(f"[dump] {n_tokens - usable_tokens} tokens trimmed from tail")

    # Reshape into (n_seqs, SEQ_LEN). int64 for the model.
    sequences = all_tokens.reshape(n_seqs, SEQ_LEN)

    # ----- Load target model on GPU -----
    print(f"[dump] loading {TARGET_MODEL_ID} in fp16…")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )
    model.eval()
    H = model.config.hidden_size
    V = model.config.vocab_size
    print(f"[dump] model loaded; hidden={H}, vocab={V}")

    # ----- Pre-allocate per-shard buffers in numpy (CPU) and flush to disk on overflow. -----
    shard_idx = 0
    shard_buf = {
        "tokens": np.empty((DUMP_SHARD_TOKENS,), dtype=np.int32),
        "hiddens": np.empty((DUMP_SHARD_TOKENS, H), dtype=np.float16),
        "topk_ids": np.empty((DUMP_SHARD_TOKENS, TOPK), dtype=np.int32),
        "topk_logits": np.empty((DUMP_SHARD_TOKENS, TOPK), dtype=np.float16),
    }
    shard_fill = 0
    total_dumped = 0

    def flush_shard():
        nonlocal shard_idx, shard_fill
        if shard_fill == 0:
            return
        prefix = dump_dir / f"shard_{shard_idx:03d}"
        for k, arr in shard_buf.items():
            arr[:shard_fill].tofile(f"{prefix}_{k}.bin")
        size_mb = sum(arr[:shard_fill].nbytes for arr in shard_buf.values()) / (1024**2)
        print(f"[dump] wrote shard_{shard_idx:03d}: {shard_fill:,} positions ({size_mb:.1f} MB)")
        shard_idx += 1
        shard_fill = 0
        dump_vol.commit()           # flush volume to durable storage after each shard

    # ----- Streaming forward pass over all sequences in batches -----
    n_batches = (n_seqs + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[dump] starting forward pass: {n_batches:,} batches of up to {BATCH_SIZE} sequences")
    fwd_start = time.time()
    with torch.no_grad():
        for batch_idx in range(n_batches):
            i0 = batch_idx * BATCH_SIZE
            i1 = min((batch_idx + 1) * BATCH_SIZE, n_seqs)
            batch_np = sequences[i0:i1]               # (B, T) int32
            batch = torch.from_numpy(batch_np.astype(np.int64)).to("cuda")  # (B, T) int64

            outputs = model(
                input_ids=batch,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]        # (B, T, H)  fp16
            logits = outputs.logits                   # (B, T, V)  fp16

            # Top-K of the target's logits, position-wise.
            topk_logits_t, topk_ids_t = torch.topk(logits, TOPK, dim=-1)  # (B, T, K)

            # Flatten (B, T) → (B*T) so we can pack into the shard buffer.
            B, T = batch.shape
            n_new = B * T
            flat_tokens = batch.view(-1).to(torch.int32).cpu().numpy()
            flat_hidden = hidden.reshape(-1, H).cpu().numpy()
            flat_topk_ids = topk_ids_t.reshape(-1, TOPK).to(torch.int32).cpu().numpy()
            flat_topk_logits = topk_logits_t.reshape(-1, TOPK).cpu().numpy()

            # Spill into shard buffer (handles cross-shard boundaries by flushing).
            written = 0
            while written < n_new:
                free = DUMP_SHARD_TOKENS - shard_fill
                take = min(free, n_new - written)
                slc = slice(shard_fill, shard_fill + take)
                src = slice(written, written + take)
                shard_buf["tokens"][slc] = flat_tokens[src]
                shard_buf["hiddens"][slc] = flat_hidden[src]
                shard_buf["topk_ids"][slc] = flat_topk_ids[src]
                shard_buf["topk_logits"][slc] = flat_topk_logits[src]
                shard_fill += take
                written += take
                if shard_fill == DUMP_SHARD_TOKENS:
                    flush_shard()
            total_dumped += n_new

            if batch_idx % 20 == 0 or batch_idx == n_batches - 1:
                elapsed = time.time() - fwd_start
                throughput = total_dumped / max(elapsed, 1e-3)
                eta = (n_seqs * SEQ_LEN - total_dumped) / max(throughput, 1)
                print(
                    f"[dump] batch {batch_idx + 1:>4}/{n_batches}  "
                    f"dumped={total_dumped:>9,}  shards={shard_idx}  fill={shard_fill:>7,}  "
                    f"throughput={throughput:>6.0f} tok/s  eta={eta:>5.0f}s"
                )

    flush_shard()

    manifest = {
        "target_model": TARGET_MODEL_ID,
        "corpus_manifest": corpus_manifest,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "topk": TOPK,
        "hidden_dim": int(H),
        "vocab_size": int(V),
        "n_positions": total_dumped,
        "n_shards": shard_idx,
        "shard_size_positions": DUMP_SHARD_TOKENS,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (dump_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    dump_vol.commit()
    print("\n[dump] manifest:")
    print(json.dumps(manifest, indent=2))
    return manifest


@app.local_entrypoint()
def dump_main() -> None:
    print(f"Dumping {TARGET_MODEL_ID} hidden states over corpus…")
    manifest = dump.remote()
    print("\n--- Modal returned ---")
    for k, v in manifest.items():
        if k == "corpus_manifest":
            continue
        print(f"  {k}: {v}")
    print(f"\n✓ C3 dump complete. {manifest['n_shards']} shards, {manifest['n_positions']:,} positions.")
