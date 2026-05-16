"""C6 — Validate the trained EAGLE head's acceptance rate (α) before browser export.

Loads a trained EAGLE head checkpoint from `spectra-eagle-ckpt`, plus the target
Qwen2.5-1.5B-Instruct, runs the speculative-decoding loop on a small prompt set,
and reports per-prompt + mean α.

This is the **hard gate before C7-C10**: if α < 0.70 on these prompts, the head
isn't good enough to bother exporting / plumbing through the browser. If 0.70 ≤
α < 0.75 we'd consider more training; ≥ 0.75 → ship it.

Implementation note: for sim simplicity we use a "fresh-target-hidden" mode for
EAGLE's draft loop — at each draft step we re-run the target to provide hidden
states. This slightly OVERESTIMATES α vs the production browser path (where
EAGLE uses its own running hidden state after the first draft), so a passing
sim score is a necessary but not sufficient condition. The browser bench (C10)
is the ground truth.

Cost: a few minutes on H100, ~$1.

Run:
    modal run scripts/remote/eval_eagle.py::evaluate --checkpoint ckpt_step_050000
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from modal_eagle import app, ckpt_vol, image

CKPT_PATH = "/ckpt"

# Same 3 prompts the browser bench uses — direct comparability to the 1.45× baseline.
EVAL_PROMPTS = [
    "Compose a haiku about WebGPU running an LLM in the browser.",
    "Explain rejection-sampling speculative decoding in one paragraph.",
    "Write five Python list comprehensions, one per line, that each filter primes from a list.",
]

TARGET_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
GENERATE_TOKENS = 64
DEFAULT_GAMMA = 2


@app.function(
    image=image,
    gpu="H100",
    timeout=30 * 60,
    volumes={CKPT_PATH: ckpt_vol},
)
def evaluate(checkpoint: str, gamma: int = DEFAULT_GAMMA) -> dict:
    """Run α evaluation on the held-out prompts.

    Args:
        checkpoint: subdir under /ckpt, e.g. "ckpt_step_050000".
        gamma: draft length per spec round.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file

    from eagle_head import build_eagle_head

    t0 = time.time()
    ck_dir = Path(CKPT_PATH) / checkpoint
    if not ck_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ck_dir}")
    meta = json.loads((ck_dir / "meta.json").read_text())
    print(f"[eval] checkpoint: {checkpoint}  step={meta['step']}  loss={meta['loss']:.4f}")

    # ----- Load target + EAGLE head -----
    print(f"[eval] loading target {TARGET_MODEL_ID} in fp16…")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )
    target.eval()
    print(f"[eval] loading EAGLE head…")
    head = build_eagle_head(TARGET_MODEL_ID, freeze_embed=True, dtype=torch.float32).to("cuda")
    head.load_state_dict(load_file(ck_dir / "model.safetensors"), strict=False)
    head.eval()

    # ----- For each prompt, run a γ-spec loop until we've generated GENERATE_TOKENS tokens -----
    per_prompt = []
    for prompt_idx, prompt in enumerate(EVAL_PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        # prompt_ids: (1, L)
        committed = prompt_ids[0].tolist()
        prefix_len = len(committed)

        n_drafted = 0
        n_accepted = 0
        n_rounds = 0
        gen_count = 0

        while gen_count < GENERATE_TOKENS:
            # Run target on current committed sequence — get hidden states + next-token logits.
            inp = torch.tensor([committed], device="cuda")           # (1, T)
            with torch.no_grad():
                out_t = target(input_ids=inp, output_hidden_states=True, use_cache=False, return_dict=True)
            target_hidden = out_t.hidden_states[-1]                 # (1, T, H) — fp16
            target_logits = out_t.logits                            # (1, T, V) — fp16
            # Target's "next-token" prediction (after the full prefix) for verify of d[0].
            target_pred_for_next = target_logits[0, -1].argmax().item()

            # Build drafts. drafts[i] is EAGLE's prediction for position prefix+i.
            # We use the simple "re-run-target-per-step" sim mode (overestimates α slightly).
            drafts: list[int] = []
            for i in range(gamma):
                # Sequence so far this round: committed + drafts[:i]
                if i == 0:
                    full_tokens = committed
                    full_hidden = target_hidden[0]                  # (T, H)
                else:
                    # Re-run target on (committed + drafts[:i]) to get its hidden at the new positions.
                    extended = torch.tensor([committed + drafts[:i]], device="cuda")
                    with torch.no_grad():
                        out_ext = target(input_ids=extended, output_hidden_states=True, use_cache=False, return_dict=True)
                    full_hidden = out_ext.hidden_states[-1][0]      # (T+i, H)
                    full_tokens = committed + drafts[:i]

                hp = full_hidden.unsqueeze(0).to(torch.float32)     # (1, T+i, H)
                tp = torch.tensor([full_tokens], device="cuda")     # (1, T+i)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    eagle_logits = head(hidden_prev=hp, token_prev=tp)  # (1, T+i, V)
                drafts.append(eagle_logits[0, -1].argmax().item())

            # Verify: target accepts d[0] if it equals target's argmax-at-position(prefix-1+0)
            #         d[i] accepted if it equals target's argmax after seeing d[0..i-1]
            # We already have target's logits over the (prefix-1) position via target_pred_for_next.
            # For d[i>=1] we'd need to re-run target on (committed + drafts[:i]) → we'll piggyback
            # on the eagle re-runs above by also storing target's argmax-at-extended-last.
            #
            # To keep this readable we'll re-verify cleanly with a final target call:
            verify_input = torch.tensor([committed + drafts], device="cuda")
            with torch.no_grad():
                out_v = target(input_ids=verify_input, use_cache=False, return_dict=True)
            # Target's argmax at position prefix-1+i → predicts position prefix+i, i.e. d[i].
            target_picks = out_v.logits[0, prefix_len - 1 : prefix_len - 1 + gamma + 1].argmax(dim=-1)  # (γ+1,)
            target_picks_list = target_picks.tolist()

            accepted_this_round = 0
            for i in range(gamma):
                if drafts[i] == target_picks_list[i]:
                    accepted_this_round += 1
                else:
                    break

            n_drafted += gamma
            n_accepted += accepted_this_round
            n_rounds += 1

            # Commit accepted drafts + bonus token.
            new_commits = drafts[:accepted_this_round]
            if accepted_this_round == gamma:
                # Full accept → bonus is target's pick at position prefix+gamma.
                bonus = target_picks_list[gamma]
            else:
                # Partial reject at position `accepted_this_round` → use target's pick there.
                bonus = target_picks_list[accepted_this_round]
            new_commits.append(bonus)
            committed.extend(new_commits)
            prefix_len = len(committed)
            gen_count += len(new_commits)
            if tokenizer.eos_token_id in new_commits:
                break

        alpha = n_accepted / max(n_drafted, 1)
        gen_text = tokenizer.decode(committed[len(prompt_ids[0]):])
        per_prompt.append({
            "prompt": prompt[:60],
            "tokens": gen_count,
            "rounds": n_rounds,
            "accepted": n_accepted,
            "drafted": n_drafted,
            "alpha": round(alpha, 4),
            "sample": gen_text[:160].replace("\n", " "),
        })
        print(f"[eval] prompt {prompt_idx+1}: tokens={gen_count} rounds={n_rounds} α={alpha:.3f}")
        print(f"       sample: {gen_text[:120]!r}")

    mean_alpha = sum(p["alpha"] for p in per_prompt) / len(per_prompt)
    result = {
        "checkpoint": checkpoint,
        "checkpoint_step": meta["step"],
        "checkpoint_loss": meta["loss"],
        "gamma": gamma,
        "n_tokens_per_prompt": GENERATE_TOKENS,
        "per_prompt": per_prompt,
        "mean_alpha": round(mean_alpha, 4),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print(f"\n[eval] mean α = {mean_alpha:.4f}")
    return result


@app.local_entrypoint()
def eval_main(checkpoint: str = "ckpt_step_050000", gamma: int = DEFAULT_GAMMA) -> None:
    print(f"Evaluating EAGLE head {checkpoint} at γ={gamma}…")
    result = evaluate.remote(checkpoint=checkpoint, gamma=gamma)
    print("\n--- Modal returned ---")
    print(json.dumps(result, indent=2))

    # Gate decision
    alpha = result["mean_alpha"]
    if alpha >= 0.75:
        print(f"\n✅ α={alpha:.3f} ≥ 0.75 — PROCEED to C7 (mlc-llm export).")
    elif alpha >= 0.70:
        print(f"\n⚠️  α={alpha:.3f} in 0.70-0.75 — consider more training before C7.")
    else:
        print(f"\n❌ α={alpha:.3f} < 0.70 — investigate training (data, lr, architecture) before C7.")
