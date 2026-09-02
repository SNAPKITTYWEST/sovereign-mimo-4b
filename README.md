# Sovereign MiMo-4B

[![License: Tri](https://img.shields.io/badge/license-AGPL%20%7C%20BSL%201.1%20%7C%20MIT-blue)](LICENSE)
[![Params](https://img.shields.io/badge/Params-~4B-orange.svg)](config/architecture.json)
[![GPU](https://img.shields.io/badge/GPU-RTX%203080-76b900.svg)](#hardware)
[![Ollama](https://img.shields.io/badge/Deploy-Ollama%20Local-000000.svg)](#deploy)

Pruned reward model built from MiMo-7B. Sovereign stack integration: FSM from DEVFLOW-FINANCE + ERE gates from bert-agent.

**No checkpoints. No cloud. Everything runs local on BBQBADDIE.**

---

## What This Is

A 4B parameter code reward model pruned from MiMo-7B-RL to match the hilbert-4b architecture spec. Every score passes through ERE P1-P5 gates. Every decision is sealed to a WORM chain.

```
MiMo-7B (32L, 4096H, 32H) → prune → Sovereign MiMo-4B (24L, 2048H, 16H GQA:4)
                                        ↓
                              GGUF Q4_K_M (~2.5 GB)
                                        ↓
                              Ollama local on RTX 3080
                                        ↓
                              FSM: PREFLIGHT → REASON → SEAL → RESPOND
                                        ↓
                              ERE: P1-P5 gates on every score
                                        ↓
                              WORM: SHA-256 chain, append-only
```

---

## Quick Start

```bash
# Prune MiMo-7B → 4B
python -m prune.cut

# Export to GGUF
python scripts/export_gguf.py

# Deploy to Ollama
ollama create sovereign-mimo-4b -f Modelfile
ollama run sovereign-mimo-4b

# Or use Python inference
python -m inference.engine
```

---

## Architecture

See [config/architecture.json](config/architecture.json) and [MODEL_CARD.md](MODEL_CARD.md).

| Parameter | Value |
|-----------|-------|
| Layers | 24 |
| Hidden | 2048 |
| Heads | 16Q / 4KV (GQA) |
| FFN | 5504 (SwiGLU) |
| Norm | RMSNorm |
| Position | RoPE (θ=10000) |
| Vocab | 32000 |
| Max Seq | 8192 |
| Reward Head | linear(2048, 1) → sigmoid |

---

## Sovereign Stack

| Component | Source | Pattern |
|-----------|--------|---------|
| FSM | DEVFLOW-FINANCE | 7-stage pipeline with HMAC-SHA256 seals |
| ERE Gates | bert-agent | P1-P5 verification, halt on failure |
| WORM Chain | DEVFLOW-FINANCE | Append-only SHA-256 audit chain |
| CUDA Kernels | hilbert | Ampere-ready (RMSNorm, FlashAttn, SwiGLU, RoPE) |
| GGUF Export | hilbert | Q4_K_M for llama.cpp / Ollama |

---

## Hardware

| Component | BBQBADDIE Spec |
|-----------|---------------|
| GPU | NVIDIA RTX 3080 (10 GB) |
| CPU | AMD Ryzen 7 7700X (8C/16T) |
| RAM | 32 GB |
| Storage | 2.33 TB free |

VRAM usage: ~3.3 GB (model + KV cache + overhead). Fits in 10 GB.

---

## License

Tri-license — choose any one:

- **AGPL-3.0** for open source / community use
- **BSL 1.1 → MIT** for commercial / production use
- **MIT** after 2029-01-01

Copyright (C) 2026 Ahmad Ali Parr, Jessica L. Williams / SNAPKITTYWEST
Bel Esprit D'Accord Irrevocable Trust

---

*Built on BBQBADDIE. No cloud. No vendor. Sovereign.*
