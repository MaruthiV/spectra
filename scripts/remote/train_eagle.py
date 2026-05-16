"""C5 — Train the EAGLE draft head on the C3 hidden-state dump.

Loss = top-K KL divergence between EAGLE's predicted distribution and the
target's saved top-K distribution at each training position.

Data shape (per dump shard from C3):
    tokens[N]              int32        input token at position t
    hiddens[N, H]          fp16         target's last-layer hidden at position t
    topk_ids[N, K]         int32        target's top-K token ids predicted for position t+1
    topk_logits[N, K]      fp16         target's top-K logits at those ids

Training pairs: for each position i (skip i == 0 within a sequence boundary, but
we ignore boundaries for MVP — small noise from inter-sequence transitions):
    INPUT  = (hiddens[i-1], tokens[i-1])
    LABEL  = softmax(topk_logits[i-1])  over topk_ids[i-1]
EAGLE head produces full-vocab logits; we gather at topk_ids[i-1] and softmax,
then cross-entropy against the label distribution.

Cost: H100 with 1-layer EAGLE head, batch 32 × 1024, ~50K steps ≈ 8–14 hours.
At $3.95/hr that's ~$32-55 per run. Plan for 1-2 runs total.

Run:
    modal run scripts/remote/train_eagle.py::train --steps 50000

Checkpoint output:
    /vol/ckpt_step_005000/{model.safetensors, optim.pt, meta.json}
    /vol/ckpt_step_010000/...
    /vol/ckpt_latest/ (symlink-ish — directory copy of most recent)
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import modal

from modal_eagle import app, ckpt_vol, dump_vol, image

DUMP_PATH = "/dump"
CKPT_PATH = "/ckpt"

# Default training config — overridable via CLI flags.
DEFAULT_STEPS = 50_000
DEFAULT_BATCH = 32                # sequences per micro-batch
DEFAULT_SEQ = 1024                # positions per training "minibatch row" (matches C3 SEQ_LEN)
DEFAULT_LR = 3e-4
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
CKPT_EVERY = 5_000
EVAL_EVERY = 1_000
EVAL_TAKE = 8                     # 8 batches of held-out data for quick val-α
SHARD_HELD_OUT = 0                # use dump shard 0's last EVAL_TAKE*BATCH positions as held-out


@app.function(
    image=image,
    gpu="H100",
    timeout=18 * 60 * 60,             # 18h hard cap (expect ~12h max per run)
    volumes={DUMP_PATH: dump_vol, CKPT_PATH: ckpt_vol},
)
def train(
    steps: int = DEFAULT_STEPS,
    batch: int = DEFAULT_BATCH,
    seq: int = DEFAULT_SEQ,
    lr: float = DEFAULT_LR,
    resume_from: str | None = None,
) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW

    from eagle_head import build_eagle_head

    t0 = time.time()
    dump_dir = Path(DUMP_PATH)
    ckpt_dir = Path(CKPT_PATH)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load dump manifest + memmap all shards -----
    dump_manifest = json.loads((dump_dir / "manifest.json").read_text())
    H = dump_manifest["hidden_dim"]
    K = dump_manifest["topk"]
    n_shards = dump_manifest["n_shards"]
    target_model_id = dump_manifest["target_model"]
    print(f"[train] dump: {dump_manifest['n_positions']:,} positions, H={H}, K={K}, {n_shards} shards")

    # Each shard maps onto 4 memmap'd arrays. We treat the corpus as a single
    # flat sequence; sampling indices are uniform across the full position pool.
    shard_arrays = []
    for s in range(n_shards):
        prefix = dump_dir / f"shard_{s:03d}"
        sz = None
        for k in ("tokens", "hiddens", "topk_ids", "topk_logits"):
            p = prefix.parent / f"{prefix.name}_{k}.bin"
            n_bytes = p.stat().st_size
            if k == "tokens":
                sz = n_bytes // 4
                tokens = np.memmap(p, dtype=np.int32, mode="r", shape=(sz,))
            elif k == "hiddens":
                hiddens = np.memmap(p, dtype=np.float16, mode="r", shape=(sz, H))
            elif k == "topk_ids":
                topk_ids = np.memmap(p, dtype=np.int32, mode="r", shape=(sz, K))
            elif k == "topk_logits":
                topk_logits = np.memmap(p, dtype=np.float16, mode="r", shape=(sz, K))
        shard_arrays.append({
            "size": sz,
            "tokens": tokens,
            "hiddens": hiddens,
            "topk_ids": topk_ids,
            "topk_logits": topk_logits,
        })

    # Flat global indexing: precompute shard offsets as a numpy array for digitize.
    shard_offsets_np = np.zeros(n_shards + 1, dtype=np.int64)
    for s_idx, s in enumerate(shard_arrays):
        shard_offsets_np[s_idx + 1] = shard_offsets_np[s_idx] + s["size"]
    total_positions = int(shard_offsets_np[-1])

    # Held-out tail of shard 0 for fast eval. We exclude these from training samples.
    eval_count = min(EVAL_TAKE * batch * seq, shard_arrays[SHARD_HELD_OUT]["size"] // 10)
    held_start_local = shard_arrays[SHARD_HELD_OUT]["size"] - eval_count
    held_end_local = shard_arrays[SHARD_HELD_OUT]["size"]
    held_start_global = int(shard_offsets_np[SHARD_HELD_OUT]) + held_start_local
    print(f"[train] held-out: shard {SHARD_HELD_OUT}, global positions {held_start_global:,}..{held_start_global + eval_count:,}")

    def sample_indices(B_total: int) -> np.ndarray:
        """Sample B_total global positions, skipping the held-out region."""
        idx = np.random.randint(0, total_positions - eval_count, size=B_total, dtype=np.int64)
        # Shift indices ≥ held_start_global to skip the held-out window.
        idx[idx >= held_start_global] += eval_count
        return idx

    def gather_batch(flat_idx: np.ndarray) -> dict:
        """Vectorised gather: bucket indices by shard with np.digitize, then
        slice into each shard's memmap via fancy indexing — no Python per-index loop."""
        # Which shard does each global index belong to?
        # digitize returns 1..n_shards; subtract 1 for 0-based shard idx.
        s_of = np.digitize(flat_idx, shard_offsets_np[1:], right=False)
        s_of = np.clip(s_of, 0, n_shards - 1)
        local = flat_idx - shard_offsets_np[s_of]

        out_tokens = np.empty(len(flat_idx), dtype=np.int32)
        out_hidden = np.empty((len(flat_idx), H), dtype=np.float16)
        out_topk_ids = np.empty((len(flat_idx), K), dtype=np.int32)
        out_topk_logits = np.empty((len(flat_idx), K), dtype=np.float16)

        for s_idx in range(n_shards):
            mask = s_of == s_idx
            if not mask.any():
                continue
            local_idx = local[mask]
            sh = shard_arrays[s_idx]
            out_tokens[mask] = sh["tokens"][local_idx]
            out_hidden[mask] = sh["hiddens"][local_idx]
            out_topk_ids[mask] = sh["topk_ids"][local_idx]
            out_topk_logits[mask] = sh["topk_logits"][local_idx]

        return {
            "tokens": out_tokens, "hidden": out_hidden,
            "topk_ids": out_topk_ids, "topk_logits": out_topk_logits,
        }

    # ----- Build EAGLE head (fp32 weights for AdamW stability, bf16 forward via autocast) -----
    print(f"[train] building EAGLE head from {target_model_id} (fp32 weights + bf16 autocast, frozen embed)…")
    head = build_eagle_head(target_model_id, freeze_embed=True, dtype=torch.float32).to("cuda")
    head.train()

    trainable = [p for p in head.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[train] trainable params: {n_trainable:,}")

    # ----- Optimizer + LR schedule -----
    optim = AdamW(trainable, lr=lr, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)

    def lr_at(step: int) -> float:
        if step < WARMUP_STEPS:
            return lr * (step + 1) / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(steps - WARMUP_STEPS, 1)
        return 0.5 * lr * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    # ----- Optional resume -----
    start_step = 0
    if resume_from:
        from safetensors.torch import load_file
        ck_dir = ckpt_dir / resume_from
        head.load_state_dict(load_file(ck_dir / "model.safetensors"), strict=False)
        meta = json.loads((ck_dir / "meta.json").read_text())
        start_step = meta["step"]
        print(f"[train] resumed from {resume_from} at step {start_step}")

    # ----- Training loop. We sample TRAINING POSITIONS, not sequences. Each
    #   "step" feeds a (B, T) shape where B*T positions are drawn independently
    #   and arranged contiguously in T dim with FAKE position_ids=arange(T).
    #   This is fine for the EAGLE head since the decoder layer's RoPE only
    #   matters relatively and we randomise positions anyway. -----
    losses = []
    print(f"[train] starting from step {start_step} → {steps}; batch={batch} × seq={seq}")
    for step in range(start_step, steps):
        cur_lr = lr_at(step)
        for g in optim.param_groups:
            g["lr"] = cur_lr

        # Sample B*T positions; arrange into (B, T).
        flat_idx = sample_indices(batch * seq)
        b = gather_batch(flat_idx)

        # Stage data in fp32 on GPU; autocast casts internals to bf16.
        hidden = torch.from_numpy(b["hidden"]).to("cuda", dtype=torch.float32).view(batch, seq, H)
        token = torch.from_numpy(b["tokens"].astype(np.int64)).to("cuda").view(batch, seq)
        topk_ids = torch.from_numpy(b["topk_ids"].astype(np.int64)).to("cuda").view(batch, seq, K)
        topk_logits = torch.from_numpy(b["topk_logits"]).to("cuda", dtype=torch.float32).view(batch, seq, K)

        # Forward in bf16 via autocast (weights stay fp32 for AdamW stability).
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            eagle_logits_topk = head.forward_topk(
                hidden_prev=hidden, token_prev=token, topk_ids=topk_ids,
            )                                                            # (B, T, K)

        # Top-K KL: distill the target's distribution shape (over its top-K) into
        # EAGLE's distribution at those same K indices. Cast to fp32 for log/exp stability.
        log_p_eagle = F.log_softmax(eagle_logits_topk.float(), dim=-1)
        p_target = F.softmax(topk_logits.float(), dim=-1)
        # KL(target || eagle) = sum p_target * (log p_target - log p_eagle).
        # Drop the constant H(target) term — gradient-equivalent and one less compute.
        loss = -(p_target * log_p_eagle).sum(dim=-1).mean()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optim.step()

        losses.append(loss.item())

        if step % 100 == 0:
            avg = sum(losses[-100:]) / max(len(losses[-100:]), 1)
            elapsed = time.time() - t0
            tokens_seen = (step - start_step + 1) * batch * seq
            throughput = tokens_seen / max(elapsed, 1e-3)
            print(
                f"[train] step {step:>6}  loss={loss.item():.4f}  avg100={avg:.4f}  "
                f"lr={cur_lr:.2e}  thru={throughput:>5.0f} pos/s  elapsed={elapsed:.0f}s"
            )

        if step > 0 and step % EVAL_EVERY == 0:
            head.eval()
            with torch.no_grad():
                acc_correct = 0
                acc_total = 0
                # Pull held-out positions as GLOBAL indices, then gather as usual.
                held_global = np.arange(held_start_global, held_start_global + eval_count, dtype=np.int64)
                for _ in range(EVAL_TAKE):
                    e_idx = np.random.choice(held_global, size=batch * seq, replace=False)
                    b = gather_batch(e_idx)
                    h = torch.from_numpy(b["hidden"]).to("cuda", dtype=torch.float32).view(batch, seq, H)
                    tk = torch.from_numpy(b["tokens"].astype(np.int64)).to("cuda").view(batch, seq)
                    tids = torch.from_numpy(b["topk_ids"].astype(np.int64)).to("cuda").view(batch, seq, K)
                    # Use selective gather and find argmax over the top-K subset
                    # (target's top-1 is always in its top-K; if EAGLE's true argmax
                    # is outside K, this slightly UNDERcounts α — but eval is fast).
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        el_topk = head.forward_topk(hidden_prev=h, token_prev=tk, topk_ids=tids)  # (B, T, K)
                    eagle_topk_pick = el_topk.argmax(dim=-1)              # (B, T) index into K
                    eagle_pred = torch.gather(tids, dim=-1, index=eagle_topk_pick.unsqueeze(-1)).squeeze(-1)
                    target_top1 = tids[:, :, 0]                            # (B, T) — target's #1 pick
                    acc_correct += (eagle_pred == target_top1).sum().item()
                    acc_total += eagle_pred.numel()
                val_alpha = acc_correct / max(acc_total, 1)
            head.train()
            print(f"[train] >>> step {step}  val_top1_match (α proxy) = {val_alpha:.4f} on {acc_total:,} positions")

        if step > 0 and step % CKPT_EVERY == 0:
            from safetensors.torch import save_file
            ck_dir = ckpt_dir / f"ckpt_step_{step:06d}"
            ck_dir.mkdir(parents=True, exist_ok=True)
            save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, ck_dir / "model.safetensors")
            (ck_dir / "meta.json").write_text(json.dumps({
                "step": step, "loss": loss.item(), "lr": cur_lr,
                "batch": batch, "seq": seq, "target_model": target_model_id,
            }, indent=2))
            ckpt_vol.commit()
            print(f"[train] saved checkpoint {ck_dir.name}")

    summary = {
        "steps_run": steps - start_step,
        "final_loss": losses[-1] if losses else None,
        "avg_loss_last_100": sum(losses[-100:]) / max(len(losses[-100:]), 1),
        "elapsed_seconds": round(time.time() - t0, 1),
        "ckpt_dir": str(ckpt_dir),
    }
    print("\n[train] training complete:")
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def train_main(steps: int = DEFAULT_STEPS) -> None:
    print(f"Launching EAGLE training: {steps} steps…")
    summary = train.remote(steps=steps)
    print("\n--- Modal returned ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
