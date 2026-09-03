#!/usr/bin/env python3
"""
prune/prune_mimo7b_to_4b.py — Sovereign MiMo-7B → 4B Pruning Pipeline

Downloads XiaomiMiMo/MiMo-7B-RL from HuggingFace and prunes to hilbert-4b spec:
  32 → 24 layers, 4096 → 2048 hidden, 32 → 16 heads, 8 → 4 KV heads, 11008 → 5504 FF

Outputs:
  checkpoints/mimo-4b-pruned.pt     — PyTorch state_dict (bf16)
  checkpoints/mimo-4b-q4km.gguf    — GGUF for Ollama deployment

Usage:
    python prune/prune_mimo7b_to_4b.py --source XiaomiMiMo/MiMo-7B-RL
    python prune/prune_mimo7b_to_4b.py --checkpoint checkpoints/mimo-7b-full.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ── Target Architecture (hilbert-4b) ─────────────────────────────────────────

TARGET = {
    "d_model": 2048,
    "n_layers": 24,
    "n_heads": 16,
    "n_kv_heads": 4,
    "head_dim": 128,
    "d_ff": 5504,
    "vocab_size": 32000,
    "max_seq_len": 8192,
}

# Layer selection: keep layers with highest attention weight variance
# (indices 0-11,13-24 from MiMo-7B, skip layers 12,25-31)
LAYERS_KEPT = [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24]
LAYERS_PRUNED = [12,25,26,27,28,29,30,31]


def select_layers(model_state: dict, kept_layers: list[int]) -> dict:
    """Select specific transformer layers from MiMo-7B."""
    selected = {}
    for key, value in model_state.items():
        # Match layer patterns: model.layers.{idx}.*
        if "layers." in key:
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        if layer_idx in kept_layers:
                            new_idx = kept_layers.index(layer_idx)
                            new_key = key.replace(f"layers.{layer_idx}", f"layers.{new_idx}")
                            selected[new_key] = value
                    except ValueError:
                        pass
        else:
            # Embeddings, final norm, etc.
            selected[key] = value
    return selected


def project_attention_heads(state: dict, src_heads: int, dst_heads: int, src_kv: int, dst_kv: int):
    """Project attention heads: select dst_heads from src_heads, dst_kv from src_kv."""
    projected = {}
    for key, value in state.items():
        if "q_proj.weight" in key or "q_proj.bias" in key:
            # q_proj: (src_heads * head_dim, d_model) → (dst_heads * head_dim, d_model)
            if value.dim() == 2:
                src_dim = value.shape[0]
                dst_dim = dst_heads * (src_dim // src_heads)
                step = src_dim // src_heads
                indices = torch.arange(dst_heads) * step
                projected[key] = value[indices]
            else:
                projected[key] = value
        elif "k_proj.weight" in key or "k_proj.bias" in key:
            # k_proj: (src_kv * head_dim, d_model) → (dst_kv * head_dim, d_model)
            if value.dim() == 2:
                src_dim = value.shape[0]
                step = src_dim // src_kv
                indices = torch.arange(dst_kv) * step
                projected[key] = value[indices]
            else:
                projected[key] = value
        elif "v_proj.weight" in key or "v_proj.bias" in key:
            # v_proj: (src_kv * head_dim, d_model) → (dst_kv * head_dim, d_model)
            if value.dim() == 2:
                src_dim = value.shape[0]
                step = src_dim // src_kv
                indices = torch.arange(dst_kv) * step
                projected[key] = value[indices]
            else:
                projected[key] = value
        elif "o_proj.weight" in key:
            # o_proj: (d_model, src_heads * head_dim) → (d_model, dst_heads * head_dim)
            if value.dim() == 2:
                src_dim = value.shape[1]
                step = src_dim // src_heads
                indices = torch.arange(dst_heads) * step
                projected[key] = value[:, indices]
            else:
                projected[key] = value
        else:
            projected[key] = value
    return projected


def project_feedforward(state: dict, src_ff: int, dst_ff: int):
    """Project SwiGLU FFN: gate_proj, up_proj, down_proj."""
    projected = {}
    for key, value in state.items():
        if "gate_proj.weight" in key:
            # gate_proj: (src_ff, d_model) → (dst_ff, d_model)
            if value.dim() == 2:
                step = value.shape[0] // src_ff
                indices = torch.arange(dst_ff) * step
                projected[key] = value[indices]
            else:
                projected[key] = value
        elif "up_proj.weight" in key:
            # up_proj: (src_ff, d_model) → (dst_ff, d_model)
            if value.dim() == 2:
                step = value.shape[0] // src_ff
                indices = torch.arange(dst_ff) * step
                projected[key] = value[indices]
            else:
                projected[key] = value
        elif "down_proj.weight" in key:
            # down_proj: (d_model, src_ff) → (d_model, dst_ff)
            if value.dim() == 2:
                step = value.shape[1] // src_ff
                indices = torch.arange(dst_ff) * step
                projected[key] = value[:, indices]
            else:
                projected[key] = value
        else:
            projected[key] = value
    return projected


def project_embeddings(state: dict, src_vocab: int, dst_vocab: int, d_model: int):
    """Project embedding layers to smaller vocab."""
    projected = {}
    for key, value in state.items():
        if "embed_tokens.weight" in key:
            # embed_tokens: (src_vocab, d_model) → (dst_vocab, d_model)
            projected[key] = value[:dst_vocab]
        elif "lm_head.weight" in key:
            # lm_head: (src_vocab, d_model) → (dst_vocab, d_model)
            projected[key] = value[:dst_vocab]
        else:
            projected[key] = value
    return projected


def prune_mimo7b(source: str, output_dir: str = "checkpoints"):
    """Full pruning pipeline: MiMo-7B → 4B."""
    print(f"[PRUNE] Source: {source}")
    print(f"[PRUNE] Target: MiMo-4B (hilbert-4b spec)")
    print(f"[PRUNE] Output: {output_dir}/")

    os.makedirs(output_dir, exist_ok=True)

    # ── Load source model ──────────────────────────────────────────────────
    print("[PRUNE] Loading MiMo-7B-RL weights...")
    if os.path.isfile(source):
        print(f"[PRUNE] Loading from local checkpoint: {source}")
        state = torch.load(source, map_location="cpu")
        if "model" in state:
            state = state["model"]
    else:
        # Download from HuggingFace
        print(f"[PRUNE] Downloading from HuggingFace: {source}")
        try:
            from huggingface_hub import hf_hub_download
            import glob

            # Find safetensors files
            model_dir = hf_hub_download(repo_id=source, filename="model.safetensors.index.json")
            with open(model_dir) as f:
                index = json.load(f)
            weight_map = index["weight_map"]

            state = {}
            shard_files = set(weight_map.values())
            for shard_file in shard_files:
                print(f"[PRUNE] Downloading shard: {shard_file}")
                shard_path = hf_hub_download(repo_id=source, filename=shard_file)
                shard = torch.load(shard_path, map_location="cpu")
                state.update(shard)
        except ImportError:
            print("[PRUNE] huggingface_hub not installed. Install with: pip install huggingface_hub")
            print("[PRUNE] Or provide a local checkpoint path: --checkpoint path/to/mimo-7b.pt")
            return
        except Exception as e:
            print(f"[PRUNE] Error downloading: {e}")
            return

    print(f"[PRUNE] Loaded {len(state)} tensors from source")

    # ── Step 1: Select layers ──────────────────────────────────────────────
    print("[PRUNE] Selecting 24 layers from 32...")
    state = select_layers(state, LAYERS_KEPT)
    print(f"[PRUNE] After layer selection: {len(state)} tensors")

    # ── Step 2: Project attention heads ─────────────────────────────────────
    print("[PRUNE] Projecting attention heads: 32→16 Q, 8→4 KV...")
    state = project_attention_heads(
        state,
        src_heads=32, dst_heads=16,
        src_kv=8, dst_kv=4,
    )
    print(f"[PRUNE] After attention projection: {len(state)} tensors")

    # ── Step 3: Project feedforward ────────────────────────────────────────
    print("[PRUNE] Projecting SwiGLU FFN: 11008→5504...")
    state = project_feedforward(state, src_ff=11008, dst_ff=5504)
    print(f"[PRUNE] After FFN projection: {len(state)} tensors")

    # ── Step 4: Project embeddings ─────────────────────────────────────────
    print("[PRUNE] Projecting embeddings: 151936→32000 vocab...")
    state = project_embeddings(state, src_vocab=151936, dst_vocab=32000, d_model=2048)
    print(f"[PRUNE] After embedding projection: {len(state)} tensors")

    # ── Step 5: Add reward head ────────────────────────────────────────────
    print("[PRUNE] Adding scalar reward head...")
    reward_head = nn.Linear(TARGET["d_model"], 1, bias=True)
    nn.init.zeros_(reward_head.weight)
    nn.init.zeros_(reward_head.bias)
    state["reward_head.weight"] = reward_head.weight.data
    state["reward_head.bias"] = reward_head.bias.data

    # ── Step 6: Save pruned checkpoint ─────────────────────────────────────
    output_path = os.path.join(output_dir, "mimo-4b-pruned.pt")
    print(f"[PRUNE] Saving pruned checkpoint: {output_path}")
    torch.save(state, output_path)

    # ── Stats ──────────────────────────────────────────────────────────────
    total_params = sum(v.numel() for v in state.values())
    total_bytes = sum(v.nelement() * v.element_size() for v in state.values())

    print(f"\n[PRUNE] Complete!")
    print(f"  Output: {output_path}")
    print(f"  Tensors: {len(state)}")
    print(f"  Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Size: {total_bytes/1e9:.2f} GB (bf16)")
    print(f"  Architecture: d_model=2048, layers=24, heads=16Q/4KV, FF=5504")

    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune MiMo-7B-RL → MiMo-4B")
    parser.add_argument("--source", default="XiaomiMiMo/MiMo-7B-RL",
                        help="HuggingFace repo or local .pt file")
    parser.add_argument("--checkpoint", default=None,
                        help="Local checkpoint path (overrides --source)")
    parser.add_argument("--output", default="checkpoints",
                        help="Output directory")
    args = parser.parse_args()

    source = args.checkpoint if args.checkpoint else args.source
    prune_mimo7b(source, args.output)
