"""Python speculative-decoding simulator.

Ground-truth oracle for the TypeScript SpecController in ../runtime/.

Implements classic single-token speculative decoding (Leviathan et al., 2023:
arxiv 2211.17192) using HuggingFace transformers on CPU. Loads a target/draft
pair, runs the spec loop, and writes per-step traces.

Why CPU and not WebGPU? Because correctness is the goal here — fp32 on CPU
gives a deterministic oracle. The browser side runs fp16 on GPU and will
diverge at fp16 boundaries; the TS impl is acceptable as long as it matches
this simulator distributionally (KL <= 0.02 at T=1) and at >=99% of token
positions at T=0 on MT-Bench-20.

Usage:
    python spec_decode_sim.py \\
        --target Qwen/Qwen2.5-1.5B-Instruct \\
        --draft Qwen/Qwen2.5-0.5B-Instruct \\
        --gamma 4 \\
        --temperature 0.0 \\
        --top-p 1.0 \\
        --prompts ../../bench/prompts/mt-bench-20.json \\
        --max-tokens 256 \\
        --seed 42 \\
        --output ../../bench/results/sim_mt_bench_20.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("spec_decode_sim")


@dataclass
class StepRecord:
    """One spec round's audit trail."""
    round: int
    drafts_proposed: int
    drafts_accepted: int
    per_position_accept: list[bool]
    per_position_target_top: list[int]   # argmax of target probs at each pos
    per_position_draft_top: list[int]    # draft sampled token at each pos
    bonus_token_id: int                  # token sampled from target at the rejection / final position
    latency_ms: float


@dataclass
class GenerationTrace:
    """Per-prompt full trace."""
    prompt: str
    prompt_token_ids: list[int]
    completion: str
    committed_token_ids: list[int]
    steps: list[StepRecord]
    cumulative_acceptance: float
    decode_ms: float
    prefill_ms: float
    tokens_per_second_decode: float


def _import_torch_lazy():
    """Defer torch import so --help works without torch installed."""
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: F401
        return True
    except ImportError as e:
        logger.error("Missing dependency: %s. Install with `pip install torch transformers`.", e)
        return False


def speculative_step(
    target_model,
    draft_model,
    prefix_ids,           # torch.LongTensor (1, L) — prompt + committed so far
    gamma: int,
    temperature: float,
    top_p: float,
    rng,                  # numpy random state for uniform samples
):
    """Run one spec round. Returns (accepted_count, committed_token_ids,
    bonus_token_id, per_position_accept, per_position_target_top, per_position_draft_top).

    Implements Leviathan et al. Algorithm 1: rejection sampling with
    probability min(1, p_target / p_draft).

    Simulator simplification: full-sequence forward each call (no KV reuse).
    Slow but unambiguous. The TS impl uses paged KV with rollback for speed;
    correctness equivalence is checked at the token-position level.
    """
    import torch
    import torch.nn.functional as F

    L = int(prefix_ids.shape[1])  # length of prompt + committed
    if L < 1:
        raise ValueError("prefix_ids must contain at least one token")

    # 1. Draft model proposes gamma tokens autoregressively.
    #    Each iteration: forward on the full growing sequence, take last-position
    #    logits, sample, append.
    draft_tokens: list[int] = []
    draft_probs_list: list[Any] = []  # list of (vocab,) prob tensors at each draft position
    cur_seq = prefix_ids
    for _ in range(gamma):
        out = draft_model(input_ids=cur_seq, use_cache=False)
        logits = out.logits[0, -1]  # (vocab,)
        if temperature == 0.0:
            probs = F.softmax(logits, dim=-1)
            draft_token = int(torch.argmax(logits).item())
        else:
            probs = _top_p_softmax(logits, temperature, top_p)
            r = rng.random()
            cdf = torch.cumsum(probs, dim=-1)
            draft_token = int(torch.searchsorted(cdf, torch.tensor(r)).item())
        draft_tokens.append(draft_token)
        draft_probs_list.append(probs.detach())
        cur_seq = torch.cat(
            [cur_seq, torch.tensor([[draft_token]], dtype=torch.long)], dim=1
        )

    # 2. Target model verifies all gamma drafts in one forward pass.
    #    Feed [prefix, drafts[0], ..., drafts[gamma-1]] (length L + gamma).
    #    The target's logit at position (L + i - 1) predicts drafts[i] for
    #    i in 0..gamma-1; logit at position (L + gamma - 1) is the bonus
    #    distribution for after-full-acceptance.
    target_input = torch.cat(
        [prefix_ids, torch.tensor([draft_tokens], dtype=torch.long)], dim=1
    )
    out_t = target_model(input_ids=target_input, use_cache=False)
    # Slice the gamma+1 relevant positions: [L-1, L, ..., L+gamma-1]
    target_logits_all = out_t.logits[0, L - 1 : L + gamma]  # (gamma+1, vocab)

    # 3. Rejection sampling per position.
    accepted_count = 0
    committed: list[int] = []
    per_position_accept: list[bool] = []
    per_position_target_top: list[int] = []
    per_position_draft_top: list[int] = []

    for i in range(gamma):
        if temperature == 0.0:
            # Greedy verify: accept iff draft matches target argmax.
            target_top = int(torch.argmax(target_logits_all[i]).item())
            per_position_target_top.append(target_top)
            per_position_draft_top.append(draft_tokens[i])
            if target_top == draft_tokens[i]:
                accepted_count += 1
                committed.append(draft_tokens[i])
                per_position_accept.append(True)
            else:
                # Rejection: take the target argmax instead.
                committed.append(target_top)
                per_position_accept.append(False)
                # Stop accepting further drafts from this round.
                break
        else:
            # Sampling verify: rejection sampling Algorithm 1.
            target_probs = _top_p_softmax(target_logits_all[i], temperature, top_p)
            draft_probs = draft_probs_list[i]
            tok = draft_tokens[i]
            p_target = float(target_probs[tok].item())
            p_draft = float(draft_probs[tok].item())
            per_position_target_top.append(int(torch.argmax(target_probs).item()))
            per_position_draft_top.append(tok)
            r = rng.random()
            accept_prob = min(1.0, p_target / max(p_draft, 1e-12))
            if r < accept_prob:
                accepted_count += 1
                committed.append(tok)
                per_position_accept.append(True)
            else:
                # Sample from corrected distribution: max(0, p_target - p_draft) renormalized.
                corrected = torch.clamp(target_probs - draft_probs, min=0.0)
                s = float(corrected.sum().item())
                if s < 1e-12:
                    resampled = int(torch.argmax(target_probs).item())
                else:
                    corrected /= s
                    cdf = torch.cumsum(corrected, dim=-1)
                    r2 = rng.random()
                    resampled = int(torch.searchsorted(cdf, torch.tensor(r2)).item())
                committed.append(resampled)
                per_position_accept.append(False)
                break

    # 4. Bonus token sampling on full-accept.
    if accepted_count == gamma:
        if temperature == 0.0:
            bonus = int(torch.argmax(target_logits_all[gamma]).item())
        else:
            target_probs = _top_p_softmax(target_logits_all[gamma], temperature, top_p)
            cdf = torch.cumsum(target_probs, dim=-1)
            r = rng.random()
            bonus = int(torch.searchsorted(cdf, torch.tensor(r)).item())
        committed.append(bonus)
        bonus_token_id = bonus
    else:
        # `committed` already includes the resampled / target-argmax token
        # at the rejection position. No bonus on partial accept.
        bonus_token_id = committed[-1]

    return (
        accepted_count,
        committed,
        bonus_token_id,
        per_position_accept,
        per_position_target_top,
        per_position_draft_top,
    )


def _top_p_softmax(logits, temperature: float, top_p: float):
    import torch
    import torch.nn.functional as F
    scaled = logits / max(temperature, 1e-6)
    probs = F.softmax(scaled, dim=-1)
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative > top_p
    # Keep at least one token: shift mask right by 1.
    mask = torch.cat([torch.zeros(1, dtype=torch.bool), mask[:-1]])
    sorted_probs[mask] = 0.0
    s = sorted_probs.sum()
    sorted_probs = sorted_probs / s if s > 0 else sorted_probs
    out = torch.zeros_like(probs)
    out.scatter_(0, sorted_idx, sorted_probs)
    return out


def run_one_prompt(
    target_model, draft_model, tokenizer, prompt: str, args, rng
) -> GenerationTrace:
    import torch

    tprefill0 = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    _prompt_len = int(input_ids.shape[1])

    # Simulator simplification: we run a full-sequence forward each spec
    # round (no KV reuse). This is slow but unambiguous; the TS impl will
    # use paged KV with rollback for speed.
    with torch.no_grad():
        # Initial prefill (just to time it).
        _ = target_model(input_ids=input_ids, use_cache=False)
        _ = draft_model(input_ids=input_ids, use_cache=False)
    tprefill_ms = (time.perf_counter() - tprefill0) * 1000.0

    committed: list[int] = []
    steps: list[StepRecord] = []
    eos_id = tokenizer.eos_token_id

    tdec0 = time.perf_counter()
    round_idx = 0
    while len(committed) < args.max_tokens:
        ts0 = time.perf_counter()
        # Re-prefill from prompt + committed (simulator simplification).
        full_seq = torch.cat(
            [input_ids, torch.tensor([committed], dtype=torch.long)], dim=1
        ) if committed else input_ids

        with torch.no_grad():
            (
                accepted,
                round_committed,
                bonus,
                per_acc,
                per_tgt,
                per_dft,
            ) = speculative_step(
                target_model,
                draft_model,
                prefix_ids=full_seq,
                gamma=args.gamma,
                temperature=args.temperature,
                top_p=args.top_p,
                rng=rng,
            )
        ts_ms = (time.perf_counter() - ts0) * 1000.0
        committed.extend(round_committed)
        steps.append(StepRecord(
            round=round_idx,
            drafts_proposed=args.gamma,
            drafts_accepted=accepted,
            per_position_accept=per_acc,
            per_position_target_top=per_tgt,
            per_position_draft_top=per_dft,
            bonus_token_id=bonus,
            latency_ms=ts_ms,
        ))
        round_idx += 1
        if eos_id is not None and any(t == eos_id for t in round_committed):
            break

    tdec_ms = (time.perf_counter() - tdec0) * 1000.0
    completion = tokenizer.decode(committed[: args.max_tokens], skip_special_tokens=True)

    proposed = sum(s.drafts_proposed for s in steps)
    accepted = sum(s.drafts_accepted for s in steps)
    cum_acc = accepted / proposed if proposed > 0 else 0.0
    tps = (len(committed) / tdec_ms * 1000.0) if tdec_ms > 0 else 0.0

    return GenerationTrace(
        prompt=prompt,
        prompt_token_ids=input_ids[0].tolist(),
        completion=completion,
        committed_token_ids=committed,
        steps=steps,
        cumulative_acceptance=cum_acc,
        decode_ms=tdec_ms,
        prefill_ms=tprefill_ms,
        tokens_per_second_decode=tps,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Spectra speculative-decoding correctness simulator")
    p.add_argument("--target", required=True, help="HF target model id")
    p.add_argument("--draft", required=True, help="HF draft model id")
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--prompts", required=True, help="Path to JSON array of prompts")
    p.add_argument("--output", required=True, help="Write traces JSON here")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Cap number of prompts")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not _import_torch_lazy():
        return 2

    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    logger.info("Loading target model: %s", args.target)
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.float32, trust_remote_code=True
    )
    target_model.eval()

    logger.info("Loading draft model: %s", args.draft)
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft, torch_dtype=torch.float32, trust_remote_code=True
    )
    draft_model.eval()

    # Vocab size sanity check (R19 in the dev plan).
    if target_model.config.vocab_size != draft_model.config.vocab_size:
        logger.error(
            "vocab_size mismatch: target=%d draft=%d (mlc-llm spec decode requires match)",
            target_model.config.vocab_size, draft_model.config.vocab_size,
        )
        return 3

    prompts = json.loads(Path(args.prompts).read_text())
    if not isinstance(prompts, list):
        logger.error("--prompts file must be a JSON array of strings")
        return 4
    if args.limit:
        prompts = prompts[: args.limit]

    traces: list[GenerationTrace] = []
    for i, pr in enumerate(prompts):
        if not isinstance(pr, str):
            pr = pr.get("prompt") if isinstance(pr, dict) else str(pr)
        logger.info("[%d/%d] prompt: %r", i + 1, len(prompts), pr[:80])
        tr = run_one_prompt(target_model, draft_model, tokenizer, pr, args, rng)
        logger.info(
            "  -> %d tokens, decode_ms=%.1f, tok/s=%.2f, accept=%.3f",
            len(tr.committed_token_ids),
            tr.decode_ms,
            tr.tokens_per_second_decode,
            tr.cumulative_acceptance,
        )
        traces.append(tr)

    out = {
        "args": vars(args),
        "traces": [
            {
                **{k: v for k, v in dataclasses.asdict(t).items() if k != "steps"},
                "steps": [dataclasses.asdict(s) for s in t.steps],
            }
            for t in traces
        ],
        "summary": {
            "n_prompts": len(traces),
            "mean_acceptance": (
                sum(t.cumulative_acceptance for t in traces) / max(len(traces), 1)
            ),
            "mean_tokens_per_second_decode": (
                sum(t.tokens_per_second_decode for t in traces) / max(len(traces), 1)
            ),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    logger.info(
        "Wrote %d traces to %s. Mean acceptance: %.3f. Mean tok/s: %.2f",
        len(traces),
        args.output,
        out["summary"]["mean_acceptance"],
        out["summary"]["mean_tokens_per_second_decode"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
