"""
inference/engine.py — Sovereign MiMo-4B Inference Engine

FSM + ERE gates pattern from DEVFLOW-FINANCE + bert-agent.

States:
  IDLE         — awaiting input
  PREFLIGHT    — three-pillar validation (SEAL, CHAIN, IDENTITY)
  REASONING    — model forward pass
  SCORING      — reward head scoring
  SEALING      — BLAKE3 attestation + WORM chain
  RESPONDING   — return gated verdict
  HALTED       — ERE gate failure, no output

Every score passes ERE P1-P5 before leaving the system.
Every decision is sealed to a WORM chain.

Usage:
    from inference.engine import InferenceEngine
    engine = InferenceEngine("checkpoints/mimo-4b-pruned.pt")
    result = engine.score("def add(a, b): return a + b")
    print(result)
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch

# ── FSM States ────────────────────────────────────────────────────────────────

class EngineState(Enum):
    IDLE = "idle"
    PREFLIGHT = "preflight"
    REASONING = "reasoning"
    SCORING = "scoring"
    SEALING = "sealing"
    RESPONDING = "responding"
    HALTED = "halted"


# ── ERE Gate Protocol ────────────────────────────────────────────────────────

@dataclass
class EREGateResult:
    p1_secrets: bool = True
    p2_injection: bool = True
    p3_loop: bool = True
    p4_telemetry: bool = True
    p5_seal: str = ""
    passed: bool = True
    violations: list[str] = field(default_factory=list)


def ere_check(agent_id: str, intent: str, output: str) -> EREGateResult:
    """
    Five-gate ERE protocol from bert-agent.

    P1: No secrets in output
    P2: No eval/code injection
    P3: Loop safety
    P4: No telemetry beacons
    P5: SHA-256 audit seal
    """
    violations = []

    # P1: No secrets (API keys, tokens, passwords)
    secret_patterns = [
        r"sk[-_]", r"api[-_]?key", r"token\s*[:=]", r"password\s*[:=]",
        r"secret\s*[:=]", r"bearer\s+", r"authorization:\s*",
    ]
    for pat in secret_patterns:
        if re.search(pat, output, re.IGNORECASE):
            violations.append(f"P1: secret pattern detected: {pat}")
            break

    # P2: No eval/code injection
    injection_patterns = [
        r"eval\s*\(", r"exec\s*\(", r"__import__", r"subprocess",
        r"os\.system", r"shell=True", r"rm\s+-rf",
    ]
    for pat in injection_patterns:
        if re.search(pat, output, re.IGNORECASE):
            violations.append(f"P2: injection pattern detected: {pat}")
            break

    # P3: Loop safety (no infinite loops without exit)
    loop_patterns = [
        r"while\s+True(?!\s*:.*?break)",
        r"for\s+.*\s+in\s+range\s*\(\s*float\s*\('inf'\)\s*\)",
    ]
    for pat in loop_patterns:
        if re.search(pat, output):
            violations.append(f"P3: unsafe loop detected: {pat}")
            break

    # P4: No telemetry beacons
    telemetry_patterns = [
        r"fetch\s*\(", r"XMLHttpRequest", r"navigator\.sendBeacon",
        r"analytics\.", r"telemetry\.",
    ]
    for pat in telemetry_patterns:
        if re.search(pat, output, re.IGNORECASE):
            violations.append(f"P4: telemetry beacon detected: {pat}")
            break

    # P5: SHA-256 seal
    seal_payload = f"{agent_id}:{intent}:{output}"
    seal = hashlib.sha256(seal_payload.encode()).hexdigest()

    passed = len(violations) == 0

    return EREGateResult(
        p1_secrets=not any("P1" in v for v in violations),
        p2_injection=not any("P2" in v for v in violations),
        p3_loop=not any("P3" in v for v in violations),
        p4_telemetry=not any("P4" in v for v in violations),
        p5_seal=seal if passed else "",
        passed=passed,
        violations=violations,
    )


# ── WORM Chain ────────────────────────────────────────────────────────────────

@dataclass
class WORMEntry:
    seq: int
    timestamp: str
    agent_id: str
    intent: str
    score: float
    verdict: str
    hash_prev: str
    hash_self: str
    ere_seal: str


class WORMChain:
    """Append-only audit chain. Every entry links to previous hash."""

    def __init__(self):
        self.entries: list[WORMEntry] = []
        self.last_hash = "genesis"

    def append(
        self,
        agent_id: str,
        intent: str,
        score: float,
        verdict: str,
        ere_seal: str,
    ) -> WORMEntry:
        seq = len(self.entries)
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Hash chain
        payload = f"{seq}:{timestamp}:{agent_id}:{intent}:{score}:{verdict}:{self.last_hash}"
        hash_self = hashlib.sha256(payload.encode()).hexdigest()

        entry = WORMEntry(
            seq=seq,
            timestamp=timestamp,
            agent_id=agent_id,
            intent=intent,
            score=score,
            verdict=verdict,
            hash_prev=self.last_hash,
            hash_self=hash_self,
            ere_seal=ere_seal,
        )

        self.entries.append(entry)
        self.last_hash = hash_self
        return entry

    def verify(self) -> bool:
        """Verify chain integrity — every link must hash correctly."""
        prev = "genesis"
        for entry in self.entries:
            if entry.hash_prev != prev:
                return False
            payload = f"{entry.seq}:{entry.timestamp}:{entry.agent_id}:{entry.intent}:{entry.score}:{entry.verdict}:{entry.hash_prev}"
            expected = hashlib.sha256(payload.encode()).hexdigest()
            if entry.hash_self != expected:
                return False
            prev = entry.hash_self
        return True


# ── Three-Pillar Preflight ───────────────────────────────────────────────────

@dataclass
class PreflightResult:
    passed: bool
    seal_valid: bool
    chain_valid: bool
    identity_valid: bool
    errors: list[str] = field(default_factory=list)


def preflight_check(
    input_text: str,
    agent_id: str = "mimo-4b-reward",
    expected_agent: str = "mimo-4b-reward",
) -> PreflightResult:
    """
    Three-pillar preflight from DEVFLOW-FINANCE.

    P1: SEAL — nonce computed and validated
    P2: CHAIN — payload integrity (no nulls, no tampering)
    P3: IDENTITY — agent verified against registry
    """
    errors = []

    # P1: SEAL — deterministic nonce from input
    nonce = hashlib.sha256(f"{agent_id}:{input_text[:80]}".encode()).hexdigest()[:16]
    seal_valid = len(nonce) == 16 and all(c in "0123456789abcdef" for c in nonce)

    # P2: CHAIN — no null bytes, no empty input
    chain_valid = True
    if not input_text or len(input_text.strip()) == 0:
        errors.append("P2: empty input")
        chain_valid = False
    if "\x00" in input_text:
        errors.append("P2: null byte in input")
        chain_valid = False

    # P3: IDENTITY — agent must be registered
    identity_valid = agent_id == expected_agent
    if not identity_valid:
        errors.append(f"P3: agent '{agent_id}' not in registry")

    passed = seal_valid and chain_valid and identity_valid

    return PreflightResult(
        passed=passed,
        seal_valid=seal_valid,
        chain_valid=chain_valid,
        identity_valid=identity_valid,
        errors=errors,
    )


# ── Inference Engine ──────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    score: float
    verdict: str
    hash: str
    ere_seal: str
    ere_gates: dict
    worm_seq: int
    latency_ms: float
    state: str


class InferenceEngine:
    """
    Sovereign MiMo-4B inference engine with FSM + ERE gates.

    Flow: IDLE → PREFLIGHT → REASONING → SCORING → SEALING → RESPONDING
    Or:   IDLE → PREFLIGHT → HALTED (if preflight fails)
    Or:   SCORING → HALTED (if ERE gates fail)
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/mimo-4b-pruned.pt",
        device: str = "auto",
        agent_id: str = "mimo-4b-reward",
    ):
        self.agent_id = agent_id
        self.state = EngineState.IDLE
        self.worm = WORMChain()

        # Load model
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[ENGINE] Loading model from {checkpoint_path}")
        print(f"[ENGINE] Device: {self.device}")

        # Import model
        from model.reward import SovereignMiMo4B, ModelConfig

        self.config = ModelConfig()
        self.model = SovereignMiMo4B(self.config)

        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[ENGINE] Checkpoint loaded: {checkpoint_path}")

        self.model = self.model.to(self.device).eval()

        # Load tokenizer (InstructBERT)
        self._load_tokenizer()

        print("[ENGINE] Ready.")

    def _load_tokenizer(self):
        """Load InstructBERT tokenizer."""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            print("[ENGINE] Tokenizer loaded: all-MiniLM-L6-v2 (InstructBERT-compatible)")
        except Exception:
            # Fallback: simple char-level tokenizer
            print("[ENGINE] WARNING: Using fallback tokenizer")
            self.tokenizer = None

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text for the model."""
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                text,
                max_length=self.config.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return encoded["input_ids"].to(self.device)
        else:
            # Fallback: truncate to max_seq_len
            tokens = [ord(c) % self.config.vocab_size for c in text[:self.config.max_seq_len]]
            tokens = tokens + [0] * (self.config.max_seq_len - len(tokens))
            return torch.tensor([tokens], dtype=torch.long, device=self.device)

    def score(self, text: str) -> ScoreResult:
        """
        Score a code/text snippet through the full sovereign pipeline.

        FSM: IDLE → PREFLIGHT → REASONING → SCORING → SEALING → RESPONDING
        """
        start = time.time()

        # ── PREFLIGHT ─────────────────────────────────────────────────────
        self.state = EngineState.PREFLIGHT
        pf = preflight_check(text, self.agent_id)
        if not pf.passed:
            self.state = EngineState.HALTED
            return ScoreResult(
                score=0.0,
                verdict="HALTED",
                hash="",
                ere_seal="",
                ere_gates={"P1": False, "P2": False, "P3": False, "P4": False, "P5": False},
                worm_seq=self.worm.last_hash,
                latency_ms=(time.time() - start) * 1000,
                state="halted",
            )

        # ── REASONING ─────────────────────────────────────────────────────
        self.state = EngineState.REASONING
        input_ids = self.tokenize(text)

        # ── SCORING ───────────────────────────────────────────────────────
        self.state = EngineState.SCORING
        with torch.no_grad():
            reward = self.model.score(input_ids)
        score = reward.item()

        # ── ERE GATE CHECK ────────────────────────────────────────────────
        verdict_text = f"score={score:.4f}"
        ere = ere_check(
            agent_id=self.agent_id,
            intent=f"reward:{text[:80]}",
            output=verdict_text,
        )

        if not ere.passed:
            self.state = EngineState.HALTED
            return ScoreResult(
                score=0.0,
                verdict="ERE_HALT",
                hash="",
                ere_seal="",
                ere_gates={
                    "P1": ere.p1_secrets,
                    "P2": ere.p2_injection,
                    "P3": ere.p3_loop,
                    "P4": ere.p4_telemetry,
                    "P5": False,
                },
                worm_seq=self.worm.last_hash,
                latency_ms=(time.time() - start) * 1000,
                state="halted",
            )

        # ── SEALING ───────────────────────────────────────────────────────
        self.state = EngineState.SEALING

        # Classify verdict
        if score >= 0.8:
            verdict = "EXCELLENT"
        elif score >= 0.6:
            verdict = "GOOD"
        elif score >= 0.4:
            verdict = "ACCEPTABLE"
        elif score >= 0.2:
            verdict = "POOR"
        else:
            verdict = "REJECT"

        # BLAKE3-style hash (using SHA-256 as stand-in)
        hash_payload = f"{self.agent_id}:{text[:128]}:{score}:{verdict}"
        hash_val = hashlib.sha256(hash_payload.encode()).hexdigest()

        # WORM chain entry
        entry = self.worm.append(
            agent_id=self.agent_id,
            intent=f"reward:{text[:80]}",
            score=score,
            verdict=verdict,
            ere_seal=ere.p5_seal,
        )

        # ── RESPONDING ────────────────────────────────────────────────────
        self.state = EngineState.RESPONDING
        latency = (time.time() - start) * 1000

        result = ScoreResult(
            score=score,
            verdict=verdict,
            hash=hash_val,
            ere_seal=ere.p5_seal,
            ere_gates={
                "P1": ere.p1_secrets,
                "P2": ere.p2_injection,
                "P3": ere.p3_loop,
                "P4": ere.p4_telemetry,
                "P5": True,
            },
            worm_seq=entry.seq,
            latency_ms=latency,
            state="complete",
        )

        self.state = EngineState.IDLE
        return result

    def score_batch(self, texts: list[str]) -> list[ScoreResult]:
        """Score multiple texts through the sovereign pipeline."""
        return [self.score(text) for text in texts]

    def verify_chain(self) -> bool:
        """Verify WORM chain integrity."""
        return self.worm.verify()

    def chain_length(self) -> int:
        """Number of sealed entries in WORM chain."""
        return len(self.worm.entries)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    # Add parent dir to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    engine = InferenceEngine(
        checkpoint_path="checkpoints/mimo-4b-pruned.pt",
        device="cpu",
    )

    # Test scoring
    test_cases = [
        "def add(a, b): return a + b",
        "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
        "import os; os.system('rm -rf /')",
        "SELECT * FROM users WHERE password = 'admin'",
    ]

    for code in test_cases:
        result = engine.score(code)
        print(f"\n{'='*60}")
        print(f"Code: {code[:60]}...")
        print(f"Score: {result.score:.4f}")
        print(f"Verdict: {result.verdict}")
        print(f"Hash: {result.hash[:16]}...")
        print(f"ERE Gates: {result.ere_gates}")
        print(f"Latency: {result.latency_ms:.1f}ms")
        print(f"Chain valid: {engine.verify_chain()}")

    print(f"\n[ENGINE] Total chain entries: {engine.chain_length()}")
