#!/usr/bin/env python3
"""
create_draft_model.py — Generate draft model binary for Kimi K3 Speculative Decoding

Creates a tiny 1-layer dense FFN draft model (dim=256, inter=1024) that shares
embeddings with the target model. Used for speculative decoding acceleration.

Usage:
    python create_draft_model.py --target model.bin --output draft.bin
    python create_draft_model.py --target model.bin --output draft.bin --dim 256 --layers 1
"""

import argparse
import struct
import numpy as np
from pathlib import Path


def create_draft_bin(
    target_bin: str,
    draft_bin: str,
    dim: int = 256,
    layers: int = 1,
    inter: int = 1024,
):
    """Create a draft model binary with random weights (for testing)."""
    # Read target header to get vocab_size and max_seq_len
    with open(target_bin, "rb") as f:
        header_data = f.read(40)
        magic, version, t_dim, t_layers, t_experts, t_topk, vocab, max_seq, t_inter, tied = struct.unpack(
            "<4siiiiiiiiibbb", header_data
        )

    if magic != b"K3M1":
        raise ValueError(f"Invalid target magic: {magic}")

    print(f"Target: dim={t_dim} layers={t_layers} vocab={vocab} max_seq={max_seq}")
    print(f"Draft:  dim={dim} layers={layers} inter={inter}")

    rng = np.random.default_rng(42)

    with open(draft_bin, "wb") as f:
        # Draft header: magic("K3D1") version(1) + K3Config struct
        f.write(b"K3D1")
        f.write(struct.pack("<i", 1))  # version
        f.write(struct.pack("<iiiiiiii",
            dim,           # dim
            layers,        # num_layers
            inter,         # intermediate_dim
            0,             # num_experts (dense FFN, no MoE)
            0,             # top_k
            vocab,         # vocab_size (shared with target)
            max_seq,       # max_seq_len
        ))
        f.write(b"\x00" * 4)  # tied_embeddings + pad

        # ln_f_weight [dim]
        f.write(rng.standard_normal(dim).astype(np.float32).tobytes())
        # output_weight [vocab, dim]
        f.write(rng.standard_normal((vocab, dim)).astype(np.float32).tobytes())

        for l in range(layers):
            # attn_ln_weight [dim]
            f.write(rng.standard_normal(dim).astype(np.float32).tobytes())
            # qkv_weight [3*dim, dim]
            f.write(rng.standard_normal((3 * dim, dim)).astype(np.float32).tobytes())
            # attn_out_weight [dim, dim]
            f.write(rng.standard_normal((dim, dim)).astype(np.float32).tobytes())
            # ffn_ln_weight [dim]
            f.write(rng.standard_normal(dim).astype(np.float32).tobytes())
            # ffn_w1 [inter, dim]
            f.write(rng.standard_normal((inter, dim)).astype(np.float32).tobytes())
            # ffn_w2 [dim, inter]
            f.write(rng.standard_normal((dim, inter)).astype(np.float32).tobytes())

    size_mb = Path(draft_bin).stat().st_size / (1024 * 1024)
    print(f"Created {draft_bin} ({size_mb:.1f} MB)")
    print(f"  dim={dim} layers={layers} inter={inter} vocab={vocab}")
    print(f"  Shares embeddings with target (token_emb + pos_emb)")


def main():
    parser = argparse.ArgumentParser(description="Create draft model binary")
    parser.add_argument("--target", required=True, help="Target model.bin path")
    parser.add_argument("--output", default="draft.bin", help="Output draft.bin path")
    parser.add_argument("--dim", type=int, default=256, help="Draft hidden dim")
    parser.add_argument("--layers", type=int, default=1, help="Draft layers")
    parser.add_argument("--inter", type=int, default=1024, help="Draft intermediate dim")
    args = parser.parse_args()

    create_draft_bin(args.target, args.output, args.dim, args.layers, args.inter)


if __name__ == "__main__":
    main()
