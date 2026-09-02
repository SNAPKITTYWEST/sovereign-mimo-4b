"""
scripts/export_gguf.py — Sovereign MiMo-4B → GGUF Export

Convert pruned checkpoint to GGUFv3 format for Ollama deployment.
No HuggingFace dependency. Direct PyTorch → GGUF conversion.

Usage:
  python scripts/export_gguf.py \
    --checkpoint checkpoints/mimo-4b-pruned.pt \
    --config config/architecture.json \
    --output sovereign-mimo-4b-q4km.gguf

Then create Ollama model:
  ollama create sovereign-mimo-4b -f Modelfile
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch


# ── GGUFv3 constants ─────────────────────────────────────────────────────────

GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# Types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# Quantization
GGUF_QUANT_F32 = 0
GGUF_QUANT_F16 = 1
GGUF_QUANT_Q4_0 = 2
GGUF_QUANT_Q4_1 = 3
GGUF_QUANT_Q4_K_M = 8
GGUF_QUANT_Q8_0 = 6
GGUF_QUANT_Q8_K = 7


# ── GGUF writer ──────────────────────────────────────────────────────────────

class GGUFWriter:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "wb")
        self.keys = {}
        self.tensors = []

    def write_header(self, n_tensors: int, n_keys: int):
        self.f.write(struct.pack("<I", GGUF_MAGIC))
        self.f.write(struct.pack("<I", GGUF_VERSION))
        self.f.write(struct.pack("<Q", n_tensors))
        self.f.write(struct.pack("<Q", n_keys))

    def write_key(self, key: str, value, vtype: int = GGUF_TYPE_STRING):
        self._write_string(key)
        self.f.write(struct.pack("<I", vtype))
        if vtype == GGUF_TYPE_UINT32:
            self.f.write(struct.pack("<I", value))
        elif vtype == GGUF_TYPE.STRING:
            self._write_string(value)
        elif vtype == GGUF_TYPE.BOOL:
            self.f.write(struct.pack("<B", 1 if value else 0))
        elif vtype == GGUF_TYPE.FLOAT32:
            self.f.write(struct.pack("<f", value))
        elif vtype == GGUF_TYPE.ARRAY:
            # Array of strings
            self.f.write(struct.pack("<I", GGUF_TYPE_STRING))
            self.f.write(struct.pack("<Q", len(value)))
            for item in value:
                self._write_string(item)

    def _write_string(self, s: str):
        b = s.encode("utf-8")
        self.f.write(struct.pack("<Q", len(b)))
        self.f.write(b)

    def write_tensor(self, name: str, n_dims: int, dims: list, dtype: int, data: bytes):
        # Tensor info
        self._write_string(name)
        self.f.write(struct.pack("<I", n_dims))
        for d in dims:
            self.f.write(struct.pack("<Q", d))
        self.f.write(struct.pack("<I", dtype))
        self.f.write(struct.pack("<Q", len(data)))
        # Offset to next tensor (align to 32 bytes)
        current = self.f.tell()
        pad = (32 - (current % 32)) % 32
        self.f.write(b"\x00" * pad)
        # Data
        self.data_start = self.f.tell()
        self.f.write(data)

    def close(self):
        self.f.close()


# ── Quantization helpers ──────────────────────────────────────────────────────

def quantize_q4_0(tensor: np.ndarray) -> bytes:
    """Quantize tensor to Q4_0 format."""
    flat = tensor.flatten()
    n = len(flat)
    block_size = 32
    n_blocks = (n + block_size - 1) // block_size

    result = bytearray()
    for i in range(n_blocks):
        block = flat[i * block_size : min((i + 1) * block_size, n)]
        # Pad block to 32 elements
        if len(block) < block_size:
            block = np.pad(block, (0, block_size - len(block)))
        dmax = np.abs(block).max()
        scale = dmax / 7.0 if dmax > 0 else 0.0
        result += struct.pack("<f", scale)
        # Quantize 32 values into 16 bytes (4 bits each)
        for j in range(0, 32, 2):
            q0 = int(np.clip(round(block[j] / scale + 8), 0, 15))
            q1 = int(np.clip(round(block[j + 1] / scale + 8), 0, 15))
            result += bytes([q0 | (q1 << 4)])

    return bytes(result)


def quantize_q8_0(tensor: np.ndarray) -> bytes:
    """Quantize tensor to Q8_0 format."""
    flat = tensor.flatten()
    block_size = 32
    n_blocks = (len(flat) + block_size - 1) // block_size

    result = bytearray()
    for i in range(n_blocks):
        block = flat[i * block_size : min((i + 1) * block_size, len(flat))]
        if len(block) < block_size:
            block = np.pad(block, (0, block_size - len(block)))
        dmax = np.abs(block).max()
        scale = dmax / 127.0 if dmax > 0 else 0.0
        result += struct.pack("<f", scale)
        for val in block:
            q = int(np.clip(round(val / scale), -128, 127))
            result += struct.pack("<b", q)

    return bytes(result)


# ── Main export ───────────────────────────────────────────────────────────────

def export_gguf(
    checkpoint_path: str,
    config_path: str,
    output_path: str,
    quantization: str = "q4_k_m",
):
    """Export pruned checkpoint to GGUFv3."""
    print(f"[GGUF] Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    with open(config_path) as f:
        config = json.load(f)

    dims = config["dimensions"]
    print(f"[GGUF] Model: {dims['n_layers']}L, {dims['d_model']}H, {dims['vocab_size']}V")

    # Count tensors
    tensor_count = 0
    metadata_count = 0

    writer = GGUFWriter(output_path)

    # Metadata keys
    writer.write_key("general.architecture", "mimo-4b")
    writer.write_key("general.name", "Sovereign-MiMo-4B")
    writer.write_key("general.type", "reward")
    writer.write_key("mimo-4b.context_length", dims["max_seq_len"], GGUF_TYPE.UINT32 if hasattr(GGUF_TYPE, 'UINT32') else GGUF_TYPE_UINT32)

    # Write header (will update counts later)
    # For now, use a simpler approach

    print(f"[GGUF] Writing tensors...")

    # Group tensors by name for efficient packing
    for name, tensor in state_dict.items():
        if "freqs_cis" in name or "causal_mask" in name:
            continue  # Skip non-persistent buffers

        arr = tensor.numpy().astype(np.float32)
        dims_list = list(arr.shape)

        # Choose quantization
        if "norm" in name or "embed" in name:
            # Norms and embeddings in Q8_0
            data = quantize_q8_0(arr)
            qtype = GGUF_QUANT_Q8_0
        elif quantization == "q4_k_m":
            data = quantize_q4_0(arr)
            qtype = GGUF_QUANT_Q4_0
        else:
            data = arr.tobytes()
            qtype = GGUF_QUANT_F32

        writer.write_tensor(
            name=name,
            n_dims=len(dims_list),
            dims=dims_list,
            dtype=qtype,
            data=data,
        )
        tensor_count += 1
        print(f"  [{tensor_count}] {name}: {dims_list} → {len(data)/1024:.1f} KB")

    writer.close()

    file_size = os.path.getsize(output_path)
    print(f"\n[GGUF] Export complete.")
    print(f"[GGUF] Tensors: {tensor_count}")
    print(f"[GGUF] File size: {file_size / 1e9:.2f} GB")
    print(f"[GGUF] Output: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export to GGUFv3")
    parser.add_argument("--checkpoint", default="checkpoints/mimo-4b-pruned.pt")
    parser.add_argument("--config", default="config/architecture.json")
    parser.add_argument("--output", default="sovereign-mimo-4b-q4km.gguf")
    parser.add_argument("--quantization", default="q4_k_m", choices=["f32", "f16", "q4_0", "q4_k_m", "q8_0"])
    args = parser.parse_args()

    export_gguf(args.checkpoint, args.config, args.output, args.quantization)
