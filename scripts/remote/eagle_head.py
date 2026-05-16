"""C4 — EAGLE draft-head architecture.

Pure PyTorch module (no Modal). Defines `EagleHead`, the small draft model that
predicts position t given (target_hidden_state[t-1], target_token[t-1]).

Design (matches EAGLE-1 / Li et al. 2024, scaled to Qwen2.5-1.5B):
  - feature merger: Linear(2H → H, no bias) over [hidden_prev, token_embed_prev]
  - 1× Qwen2DecoderLayer (rotary + RMSNorm + GQA, same hyperparams as target)
  - lm_head shared with target Qwen2.5-1.5B (frozen weights)
  - token embedding initialised from target's embedding (trainable)

Parameter count (with hidden=1536, intermediate=8960, num_heads=12 KV=2):
  - fc_merge:    1536 × 3072  =  4.7M
  - decoder layer: ~25M (attention + MLP)
  - embed (shared): 151936 × 1536 = 233M (trainable but enormous)
  - lm_head (shared, frozen): 151936 × 1536 = 233M (no grad)
  Trainable: ~263M if embed is unfrozen; ~30M if embed frozen too.

Sanity check (run locally, no GPU/network):
    python scripts/remote/eagle_head.py
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def build_eagle_head(
    target_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    freeze_embed: bool = False,
    dtype: torch.dtype = torch.float32,
) -> "EagleHead":
    """Construct an EagleHead from the target model's config + shared weights.

    Loads the target's tokenizer/embedding/lm_head once on CPU. Cheap (~1.5 GB
    of RAM); designed to be called inside a Modal job after the dataset is ready.
    """
    from transformers import AutoModelForCausalLM, AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    config = AutoConfig.from_pretrained(target_model_id)
    target = AutoModelForCausalLM.from_pretrained(target_model_id, torch_dtype=dtype)

    head = EagleHead(
        config=config,
        decoder_layer_cls=Qwen2DecoderLayer,
        embed_weight=target.model.embed_tokens.weight.detach().clone(),
        lm_head_weight=target.lm_head.weight.detach().clone(),
        freeze_embed=freeze_embed,
        dtype=dtype,
    )
    del target
    return head


class EagleHead(nn.Module):
    """EAGLE-style 1-layer draft head.

    Forward signature:
        logits = head(hidden_prev, token_prev, position_ids=None)
    Where:
        hidden_prev:  (B, T, H) — target's last-layer hidden at positions [0..T-1]
        token_prev:   (B, T)    — actual tokens at positions [0..T-1]
        position_ids: (B, T)    — absolute positions for RoPE (defaults to range(T))
    Returns:
        logits: (B, T, V) — distribution over next token at positions [1..T]
    """

    def __init__(
        self,
        config,
        decoder_layer_cls,
        embed_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        freeze_embed: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.config = config
        H = config.hidden_size
        V = config.vocab_size

        # Feature merger: combine target's hidden state with token embedding.
        self.fc_merge = nn.Linear(2 * H, H, bias=False, dtype=dtype)

        # Single decoder layer (matches target's architecture exactly).
        # transformers' Qwen2DecoderLayer requires (config, layer_idx).
        self.decoder = decoder_layer_cls(config, layer_idx=0)
        if dtype != torch.float32:
            self.decoder = self.decoder.to(dtype)

        # RoPE module (need to compute cos/sin for position_embeddings).
        # transformers exposes this as model.model.rotary_emb on Qwen2.
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # Token embedding: initialised from target, optionally frozen.
        self.embed_tokens = nn.Embedding(V, H, dtype=dtype)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_weight.to(dtype))
        if freeze_embed:
            self.embed_tokens.weight.requires_grad_(False)

        # LM head: shared with target, frozen (knowledge distillation target).
        self.lm_head = nn.Linear(H, V, bias=False, dtype=dtype)
        with torch.no_grad():
            self.lm_head.weight.copy_(lm_head_weight.to(dtype))
        self.lm_head.weight.requires_grad_(False)

    def _trunk(
        self,
        hidden_prev: torch.Tensor,
        token_prev: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Shared body: returns the post-decoder hidden states (B, T, H)."""
        B, T, _ = hidden_prev.shape
        device = hidden_prev.device

        if position_ids is None:
            position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        emb = self.embed_tokens(token_prev)
        merged = self.fc_merge(torch.cat([hidden_prev, emb], dim=-1))
        cos, sin = self.rotary_emb(merged, position_ids)
        position_embeddings = (cos, sin)

        if attention_mask is None:
            mask_val = torch.finfo(merged.dtype).min
            causal = torch.full((T, T), mask_val, device=device, dtype=merged.dtype)
            causal = torch.triu(causal, diagonal=1)
            attention_mask = causal.view(1, 1, T, T).expand(B, 1, T, T)

        decoder_out = self.decoder(
            hidden_states=merged,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        return decoder_out[0] if isinstance(decoder_out, tuple) else decoder_out

    def forward(
        self,
        hidden_prev: torch.Tensor,
        token_prev: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full-vocab forward. (B, T, V). Use forward_topk during training to avoid OOM."""
        hidden = self._trunk(hidden_prev, token_prev, position_ids, attention_mask)
        return self.lm_head(hidden)

    def forward_topk(
        self,
        hidden_prev: torch.Tensor,
        token_prev: torch.Tensor,
        topk_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute logits only at the requested top-K vocab indices, per position.

        This avoids materialising the (B, T, V≈152k) tensor — saves ~20 GB at
        B=32 T=1024 and unlocks bf16 + bigger batches without OOM.

        Args:
            topk_ids: (B, T, K) int64 vocab indices to score against.

        Returns:
            (B, T, K) logits — equivalent to gather(lm_head(trunk), -1, topk_ids).
        """
        hidden = self._trunk(hidden_prev, token_prev, position_ids, attention_mask)
        # lm_head.weight: (V, H). Gather rows at topk_ids → (B, T, K, H).
        # Then dot with hidden[..., None, :] → (B, T, K, 1) → squeeze.
        B, T, H = hidden.shape
        K = topk_ids.shape[-1]
        W = self.lm_head.weight                                 # (V, H)
        W_topk = W[topk_ids.reshape(-1)].reshape(B, T, K, H)    # (B, T, K, H)
        # einsum is faster + memory-friendlier than expand + matmul.
        return torch.einsum("btkh,bth->btk", W_topk, hidden)


def _trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _total_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Local smoke test — run with `python scripts/remote/eagle_head.py`. Doesn't
# touch Modal, doesn't need GPU. Verifies shapes + that the forward path runs.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building EagleHead from Qwen/Qwen2.5-1.5B-Instruct (will download ~3GB if not cached)…")
    head = build_eagle_head(freeze_embed=False, dtype=torch.float32)
    head.eval()

    print(f"  total params:     {_total_param_count(head):>12,}")
    print(f"  trainable params: {_trainable_param_count(head):>12,}")
    print(f"  config: hidden={head.config.hidden_size}, vocab={head.config.vocab_size}")

    B, T = 2, 16
    H = head.config.hidden_size
    V = head.config.vocab_size

    hidden_prev = torch.randn(B, T, H)
    token_prev = torch.randint(0, V, (B, T))
    with torch.no_grad():
        logits = head(hidden_prev, token_prev)

    assert logits.shape == (B, T, V), f"bad output shape: {logits.shape}"
    print(f"\n✓ Forward OK. Input (B={B}, T={T}, H={H}); output logits {tuple(logits.shape)}.")
    print(f"  logits sample: mean={logits.mean().item():.3f} std={logits.std().item():.3f}")
