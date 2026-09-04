# Sovereign MiMo-4B — Model Card

## Model Details

| Property | Value |
|----------|-------|
| **Name** | Sovereign MiMo-4B |
| **Type** | Instruct model (decoder-only transformer) |
| **Architecture** | GQA + SwiGLU + RMSNorm + RoPE |
| **Parameters** | ~4B |
| **Source** | Pruned from MiMo-7B-RL (XiaomiMiMo) |
| **Target** | Hilbert-4b architecture spec |
| **License** | Tri-license: AGPL-3.0 / BSL-1.1 / Sovereign Source v1.0 |
| **Author** | Ahmad Ali Parr, Jessica L. Williams / SNAPKITTYWEST |
| **Hardware** | BBQBADDIE — RTX 3080 (10GB), Ryzen 7 7700X, 32GB RAM |

---

## Architecture

| Parameter | MiMo-7B (Source) | Sovereign MiMo-4B (Target) |
|-----------|------------------|---------------------------|
| Layers | 32 | 24 |
| Hidden Dim | 4096 | 2048 |
| Attention Heads | 32 | 16 |
| KV Heads | 8 | 4 (GQA) |
| Head Dim | 128 | 128 |
| FFN Dim | 11008 | 5504 |
| Activation | SwiGLU | SwiGLU |
| Norm | RMSNorm | RMSNorm |
| Position | RoPE | RoPE (θ=10000) |
| Vocab | 152064 | 32000 |
| Max Seq | 32768 | 8192 |
| Output Head | Linear(4096, 152064) | Linear(2048, 32000) |

---

## Pruning Strategy

```
MiMo-7B (32 layers, 4096 hidden, 32 heads)
    │
    ├── Layer Selection: keep 24 of 32 (drop 12,25-31)
    ├── Head Pruning: 32Q/8KV → 16Q/4KV (first N heads)
    ├── Hidden Projection: 4096 → 2048 (truncated SVD)
    ├── MLP Projection: 11008 → 5504 (direct slice)
    ├── Embedding Truncation: 152064 → 32000 vocab
    └── LM Head: linear(2048, 32000) for text generation
    │
    ▼
Sovereign MiMo-4B (24 layers, 2048 hidden, 16 heads)
```

---

## Capabilities

General-purpose instruction following:

| Task | Example |
|------|---------|
| Code Generation | "Write a Python function to sort a list" |
| Question Answering | "Explain what a hash table is" |
| Reasoning | "What is 15 * 37 + 82?" |
| Task Execution | "Create a SQL query to find active users" |
| Text Analysis | "Summarize this paragraph" |
| Translation | "Translate this to Spanish" |

---

## Generation Parameters

| Parameter | Default | Range |
|-----------|---------|-------|
| Temperature | 0.7 | 0.0 – 2.0 |
| Top-K | 50 | 1 – 100 |
| Top-P | 0.9 | 0.0 – 1.0 |
| Max Tokens | 512 | 1 – 8192 |
| Repeat Penalty | 1.1 | 1.0 – 2.0 |

---

## Sovereign Stack Integration

### FSM Pattern (from DEVFLOW-FINANCE)

```
IDLE → PREFLIGHT → REASONING → SEALING → RESPONDING → IDLE
           │                                        │
     PREFLIGHT_FAILED                        ERE_HALT (terminal)
```

### Three-Pillar Preflight

| Pillar | Check | Failure |
|--------|-------|---------|
| P1: SEAL | Deterministic nonce (SHA-256) | Query dropped |
| P2: CHAIN | Payload integrity (no nulls) | Query dropped |
| P3: IDENTITY | Agent verified against registry | Query dropped |

### ERE Gate Protocol (from bert-agent)

| Gate | Check | Failure |
|------|-------|---------|
| P1 | No secrets in output | Output suppressed |
| P2 | No eval/code injection | Output suppressed |
| P3 | Loop safety | Output suppressed |
| P4 | No telemetry beacons | Output suppressed |
| P5 | SHA-256 audit seal | Seal not generated |

### WORM Chain

Every output is appended to an append-only audit chain:
- SHA-256 content hash
- Previous entry hash (chain linking)
- ERE seal (P5)
- Timestamp, agent ID, intent, output hash, verdict

---

## Deployment

### Prune MiMo-7B → 4B

```bash
python -m prune.cut \
  --source XiaomiMiMo/MiMo-7B-RL \
  --output checkpoints/mimo-4b-instruct.pt \
  --config config/architecture.json
```

### Export to GGUF

```bash
python scripts/export_gguf.py \
  --checkpoint checkpoints/mimo-4b-instruct.pt \
  --config config/architecture.json \
  --output sovereign-mimo-4b-q4km.gguf
```

### Deploy to Ollama

```bash
ollama create sovereign-mimo-4b -f Modelfile
ollama run sovereign-mimo-4b
```

### Inference (Python)

```python
from inference.engine import InferenceEngine

engine = InferenceEngine("checkpoints/mimo-4b-instruct.pt")
result = engine.generate("Write a Python function to compute fibonacci numbers")

print(result.text)          # "def fibonacci(n):\n    ..."
print(result.ere_gates)     # {"P1": True, "P2": True, ...}
print(result.hash)          # "a3f8d2c1..."
```

---

## Hardware Requirements

| Component | Requirement | BBQBADDIE |
|-----------|-------------|-----------|
| GPU | 10GB+ VRAM | RTX 3080 (10GB) ✅ |
| RAM | 16GB+ | 32GB ✅ |
| Storage | 5GB+ | 2.33TB free ✅ |
| CPU | 4+ cores | Ryzen 7 7700X (8C) ✅ |

### VRAM Usage

| Component | Size |
|-----------|------|
| Model weights (Q4_K_M) | ~2.5 GB |
| KV cache (8192 ctx) | ~0.5 GB |
| Inference overhead | ~0.3 GB |
| **Total** | **~3.3 GB** (fits in 10GB) |

---

## Design Decisions

| Decision | Why |
|----------|-----|
| Layer selection over random pruning | Preserves learned representations in early/mid layers |
| Head selection over head pruning | GQA requires clean head structure |
| SVD projection for hidden dim | Preserves 99% of weight energy |
| InstructBERT tokenizer | Better instruction following than BPE |
| LM head for text generation | Standard autoregressive generation |
| ERE gates on every output | No unverified output leaves the system |
| WORM chain | Every decision is cryptographically auditable |
| GGUF Q4_K_M | Fits RTX 3080 with room for KV cache |
| Ollama local deployment | Zero cloud dependency, sovereign compute |

---

## Provenance

| Component | Source | License |
|-----------|--------|---------|
| Architecture | Hilbert-4b (SNAPKITTYWEST) | BSL-1.1 / AGPL / MPL |
| Base weights | MiMo-7B-RL (XiaomiMiMo) | Apache 2.0 |
| FSM pattern | DEVFLOW-FINANCE (SNAPKITTYWEST) | FSL-1.1 |
| ERE gates | bert-agent (SNAPKITTYWEST) | AGPL / BSL / MIT |
| CUDA kernels | hilbert (SNAPKITTYWEST) | BSL-1.1 / AGPL / MPL |
| Training data | Public domain only | Permissive |

---

## Citation

```bibtex
@misc{sovereign-mimo-4b-2026,
  title={Sovereign MiMo-4B: Instruct Model with Sovereign Stack Integration},
  author={Ahmad Ali Parr, Jessica L. Williams},
  year={2026},
  note={Pruned from MiMo-7B-RL, hilbert-4b architecture, DEVFLOW-FINANCE FSM + bert-agent ERE gates},
  url={https://github.com/SNAPKITTYWEST/sovereign-mimo-4b}
}
```

---

*Built on BBQBADDIE. No cloud. No vendor. Sovereign.*
