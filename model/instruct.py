"""
model/instruct.py — Sovereign MiMo-4B Instruct Model

Decoder-only transformer for instruction following.
Architecture: 24 layers, 2048 hidden, 16 heads (GQA: 4 KV), SwiGLU, RoPE, RMSNorm.

Usage:
    from model.instruct import SovereignMiMo4B, ModelConfig
    config = ModelConfig()
    model = SovereignMiMo4B(config)
    output = model.generate(input_ids, max_new_tokens=256)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 128
    d_ff: int = 5504
    vocab_size: int = 32000
    max_seq_len: int = 8192
    rope_theta: float = 10000.0
    rope_dim: int = 128
    norm_eps: float = 1e-5
    dropout: float = 0.0

    @classmethod
    def from_json(cls, path: str = "config/architecture.json") -> "ModelConfig":
        with open(path) as f:
            data = json.load(f)
        dims = data["dimensions"]
        rope = data["rope"]
        norm = data["norm"]
        return cls(
            d_model=dims["d_model"],
            n_layers=dims["n_layers"],
            n_heads=dims["n_heads"],
            n_kv_heads=dims["n_kv_heads"],
            head_dim=dims["head_dim"],
            d_ff=dims["d_ff"],
            vocab_size=dims["vocab_size"],
            max_seq_len=dims["max_seq_len"],
            rope_theta=rope["theta"],
            rope_dim=rope["dimension_count"],
            norm_eps=norm["epsilon"],
        )


# ── RMSNorm ───────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# ── Rotary Position Embedding ────────────────────────────────────────────────

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:x_complex.shape[-2]].to(x_complex.device)
    x_rot = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rot.type_as(x)


# ── Grouped-Query Attention ───────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


# ── SwiGLU MLP ────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ff_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ff = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.ff(self.ff_norm(x))
        return x


# ── Sovereign MiMo-4B Instruct ───────────────────────────────────────────────

class SovereignMiMo4B(nn.Module):
    """
    Sovereign MiMo-4B Instruct: decoder-only transformer for instruction following.

    Forward pass returns logits for next-token prediction.
    Use generate() for autoregressive text generation.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.rope_dim, config.max_seq_len, config.rope_theta),
            persistent=False,
        )

        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        B, L = input_ids.shape
        device = input_ids.device

        x = self.embed_tokens(input_ids)
        x = self.embed_dropout(x)

        freqs_cis = self.freqs_cis[:L].to(device)
        mask = self.causal_mask[:, :, :L, :L].to(device)

        for layer in self.layers:
            x = layer(x, freqs_cis, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_token: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: (B, L) prompt tokens
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus sampling threshold
            stop_token: optional token to stop generation

        Returns:
            (B, L + generated) generated tokens
        """
        self.eval()
        device = input_ids.device

        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            out = self.forward(idx_cond)
            logits = out["logits"][:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if stop_token is not None and (next_token == stop_token).all():
                break

        return input_ids


# ── Count parameters ──────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "total_b": round(total / 1e9, 2),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = ModelConfig()
    model = SovereignMiMo4B(config)
    stats = count_params(model)

    print(f"Sovereign MiMo-4B Instruct")
    print(f"  Parameters: {stats['total']:,} ({stats['total_b']}B)")
    print(f"  Layers: {config.n_layers}")
    print(f"  Hidden: {config.d_model}")
    print(f"  Heads: {config.n_heads}Q/{config.n_kv_heads}KV")
    print(f"  FF: {config.d_ff} (SwiGLU)")
    print(f"  Vocab: {config.vocab_size}")
    print(f"  Max seq: {config.max_seq_len}")
    print(f"  Mode: Instruct (text generation)")
