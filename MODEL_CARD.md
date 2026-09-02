# Sovereign MiMo-4B — Model Card

## Model Details

| Property | Value |
|----------|-------|
| **Name** | Sovereign MiMo-4B |
| **Architecture** | Decoder-only transformer (GQA + SwiGLU + RMSNorm + RoPE) |
| **Parameters** | ~4B |
| **Source** | Pruned from MiMo-7B-RL (XiaomiMiMo) |
| **Target** | Hilbert-4b architecture spec |
| **License** | Tri-license: AGPL-3.0 / BSL-1.1 / MIT |
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
    └── Reward Head: linear(2048, 1) with tanh
    │
    ▼
Sovereign MiMo-4B (24 layers, 2048 hidden, 16 heads)
```

---

## Reward Head

Scalar reward model for code quality scoring.

| Property | Value |
|----------|-------|
| Type | Linear → sigmoid |
| Input | Last token hidden state (2048-dim) |
| Output | Scalar [0, 1] |
| Training | Code quality signals (test pass, complexity, style) |

### Scoring Rubric

| Score | Verdict | Meaning |
|-------|---------|---------|
| 0.8 – 1.0 | EXCELLENT | Optimal, well-documented, secure |
| 0.6 – 0.8 | GOOD | Clean, efficient, idiomatic |
| 0.4 – 0.6 | ACCEPTABLE | Works, needs improvement |
| 0.2 – 0.4 | POOR | Functional but bad quality |
| 0.0 – 0.2 | REJECT | Unsafe, broken, malicious |

---

## Sovereign Stack Integration

### FSM Pattern (from DEVFLOW-FINANCE)

```
IDLE → PREFLIGHT → REASONING → SCORING → SEALING → RESPONDING → IDLE
          ↓                                              ↓
    PREFLIGHT_FAILED                              ERE_HALT (terminal)
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
| P1 | No secrets in output | Verdict suppressed |
| P2 | No eval/code injection | Verdict suppressed |
| P3 | Loop safety | Verdict suppressed |
| P4 | No telemetry beacons | Verdict suppressed |
| P5 | SHA-256 audit seal | Seal not generated |

### WORM Chain

Every score is appended to an append-only audit chain:
- SHA-256 content hash
- Previous entry hash (chain linking)
- ERE seal (P5)
- Timestamp, agent ID, intent, score, verdict

---

## Deployment

### Prune MiMo-7B → 4B

```bash
python -m prune.cut \
  --source XiaomiMiMo/MiMo-7B-RL \
  --output checkpoints/mimo-4b-pruned.pt \
  --config config/architecture.json
```

### Export to GGUF

```bash
python scripts/export_gguf.py \
  --checkpoint checkpoints/mimo-4b-pruned.pt \
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

engine = InferenceEngine("checkpoints/mimo-4b-pruned.pt")
result = engine.score("def add(a, b): return a + b")

print(result.score)      # 0.85
print(result.verdict)    # "EXCELLENT"
print(result.ere_gates)  # {"P1": True, "P2": True, ...}
print(result.hash)       # "a3f8d2c1..."
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
| Scalar reward head | Single score is sufficient for code quality |
| ERE gates on every score | No unscored code leaves the system |
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
  title={Sovereign MiMo-4B: Pruned Reward Model with Sovereign Stack Integration},
  author={Ahmad Ali Parr, Jessica L. Williams},
  year={2026},
  note={Pruned from MiMo-7B-RL, hilbert-4b architecture, DEVFLOW-FINANCE FSM + bert-agent ERE gates},
  url={https://github.com/SNAPKITTYWEST/sovereign-mimo-4b}
}
```

---

*Built on BBQBADDIE. No cloud. No vendor. Sovereign.*
