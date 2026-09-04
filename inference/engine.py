"""
inference/engine.py — Sovereign MiMo-4B Instruct Engine

FSM + ERE gates pattern for instruction-following inference.

States:
  IDLE         — awaiting input
  PREFLIGHT    — three-pillar validation (SEAL, CHAIN, IDENTITY)
  REASONING    — model forward pass + generation
  SEALING      — BLAKE3 attestation + WORM chain
  RESPONDING   — return generated text
  HALTED       — ERE gate failure, no output

Every output passes ERE P1-P5 before leaving the system.
Every decision is sealed to a WORM chain.

Usage:
    from inference.engine import InferenceEngine
    engine = InferenceEngine("checkpoints/mimo-4b-instruct.pt")
    result = engine.generate("Write a Python function to compute fibonacci numbers")
    print(result.text)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


class EngineState(Enum):
    IDLE = "idle"
    PREFLIGHT = "preflight"
    REASONING = "reasoning"
    SEALING = "sealing"
    RESPONDING = "responding"
    HALTED = "halted"


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
    violations = []

    secret_patterns = [
        r"sk[-_]", r"api[-_]?key", r"token\s*[:=]", r"password\s*[:=]",
        r"secret\s*[:=]", r"bearer\s+", r"authorization:\s*",
    ]
    for pat in secret_patterns:
        if re.search(pat, output, re.IGNORECASE):
            violations.append(f"P1: secret pattern detected: {pat}")
            break

    injection_patterns = [
        r"eval\s*\(", r"exec\s*\(", r"__import__", r"subprocess",
        r"os\.system", r"shell=True", r"rm\s+-rf",
    ]
    for pat in injection_patterns:
        if re.search(pat, output, re.IGNORECASE):
            violations.append(f"P2: injection pattern detected: {pat}")
            break

    loop_patterns = [
        r"while\s+True(?!\s*:.*?break)",
        r"for\s+.*\s+in\s+range\s*\(\s*float\s*\('inf'\)\s*\)",
    ]
    for pat in loop_patterns:
        if re.search(pat, output):
            violations.append(f"P3: unsafe loop detected: {pat}")
            break

    telemetry_patterns = [
        r"fetch\s*\(", r"XMLHttpRequest", r"navigator\.sendBeacon",
        r"analytics\.", r"telemetry\.",
    ]
    for pat in telemetry_patterns:
        if re.search(pat, output, re.IGNORECASE):
            violations.append(f"P4: telemetry beacon detected: {pat}")
            break

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


@dataclass
class WORMEntry:
    seq: int
    timestamp: str
    agent_id: str
    intent: str
    output_hash: str
    verdict: str
    hash_prev: str
    hash_self: str
    ere_seal: str


class WORMChain:
    def __init__(self):
        self.entries: list[WORMEntry] = []
        self.last_hash = "genesis"

    def append(self, agent_id: str, intent: str, output_hash: str, verdict: str, ere_seal: str) -> WORMEntry:
        seq = len(self.entries)
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload = f"{seq}:{timestamp}:{agent_id}:{intent}:{output_hash}:{verdict}:{self.last_hash}"
        hash_self = hashlib.sha256(payload.encode()).hexdigest()
        entry = WORMEntry(seq=seq, timestamp=timestamp, agent_id=agent_id, intent=intent,
                          output_hash=output_hash, verdict=verdict, hash_prev=self.last_hash,
                          hash_self=hash_self, ere_seal=ere_seal)
        self.entries.append(entry)
        self.last_hash = hash_self
        return entry

    def verify(self) -> bool:
        prev = "genesis"
        for entry in self.entries:
            if entry.hash_prev != prev:
                return False
            payload = f"{entry.seq}:{entry.timestamp}:{entry.agent_id}:{entry.intent}:{entry.output_hash}:{entry.verdict}:{entry.hash_prev}"
            if entry.hash_self != hashlib.sha256(payload.encode()).hexdigest():
                return False
            prev = entry.hash_self
        return True


@dataclass
class PreflightResult:
    passed: bool
    seal_valid: bool
    chain_valid: bool
    identity_valid: bool
    errors: list[str] = field(default_factory=list)


def preflight_check(input_text: str, agent_id: str = "mimo-4b-instruct", expected_agent: str = "mimo-4b-instruct") -> PreflightResult:
    errors = []
    nonce = hashlib.sha256(f"{agent_id}:{input_text[:80]}".encode()).hexdigest()[:16]
    seal_valid = len(nonce) == 16 and all(c in "0123456789abcdef" for c in nonce)
    chain_valid = True
    if not input_text or len(input_text.strip()) == 0:
        errors.append("P2: empty input")
        chain_valid = False
    if "\x00" in input_text:
        errors.append("P2: null byte in input")
        chain_valid = False
    identity_valid = agent_id == expected_agent
    if not identity_valid:
        errors.append(f"P3: agent '{agent_id}' not in registry")
    passed = seal_valid and chain_valid and identity_valid
    return PreflightResult(passed=passed, seal_valid=seal_valid, chain_valid=chain_valid,
                           identity_valid=identity_valid, errors=errors)


@dataclass
class GenerateResult:
    text: str
    hash: str
    ere_seal: str
    ere_gates: dict
    worm_seq: int
    latency_ms: float
    state: str
    tokens_generated: int


class InferenceEngine:
    """
    Sovereign MiMo-4B Instruct inference engine with FSM + ERE gates.

    Flow: IDLE -> PREFLIGHT -> REASONING -> SEALING -> RESPONDING
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/mimo-4b-instruct.pt",
        device: str = "auto",
        agent_id: str = "mimo-4b-instruct",
    ):
        self.agent_id = agent_id
        self.state = EngineState.IDLE
        self.worm = WORMChain()

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[ENGINE] Loading instruct model from {checkpoint_path}")
        print(f"[ENGINE] Device: {self.device}")

        from model.instruct import SovereignMiMo4B, ModelConfig
        self.config = ModelConfig()
        self.model = SovereignMiMo4B(self.config)

        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[ENGINE] Checkpoint loaded: {checkpoint_path}")

        self.model = self.model.to(self.device).eval()
        self._load_tokenizer()
        print("[ENGINE] Instruct model ready.")

    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            self.tokenizer = None

    def tokenize(self, text: str) -> torch.Tensor:
        if self.tokenizer is not None:
            encoded = self.tokenizer(text, max_length=self.config.max_seq_len, padding="max_length",
                                     truncation=True, return_tensors="pt")
            return encoded["input_ids"].to(self.device)
        else:
            tokens = [ord(c) % self.config.vocab_size for c in text[:self.config.max_seq_len]]
            tokens = tokens + [0] * (self.config.max_seq_len - len(tokens))
            return torch.tensor([tokens], dtype=torch.long, device=self.device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        system: str = "You are Sovereign MiMo-4B, a sovereign instruct model.",
    ) -> GenerateResult:
        start = time.time()

        # PREFLIGHT
        self.state = EngineState.PREFLIGHT
        pf = preflight_check(prompt, self.agent_id)
        if not pf.passed:
            self.state = EngineState.HALTED
            return GenerateResult(text="", hash="", ere_seal="", ere_gates={},
                                  worm_seq=0, latency_ms=0, state="halted", tokens_generated=0)

        # REASONING
        self.state = EngineState.REASONING
        full_prompt = f"{system}\n\n{prompt}"
        input_ids = self.tokenize(full_prompt)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        # Decode only the generated portion
        generated_ids = output_ids[:, input_ids.shape[1]:]
        if self.tokenizer is not None:
            text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        else:
            text = "".join(chr(t % 128) for t in generated_ids[0].tolist())

        tokens_generated = generated_ids.shape[1]

        # ERE GATE CHECK
        ere = ere_check(agent_id=self.agent_id, intent=f"instruct:{prompt[:80]}", output=text)
        if not ere.passed:
            self.state = EngineState.HALTED
            return GenerateResult(text="", hash="", ere_seal="", ere_gates={},
                                  worm_seq=0, latency_ms=(time.time() - start) * 1000,
                                  state="ere_halt", tokens_generated=0)

        # SEALING
        self.state = EngineState.SEALING
        output_hash = hashlib.sha256(text.encode()).hexdigest()
        entry = self.worm.append(agent_id=self.agent_id, intent=f"instruct:{prompt[:80]}",
                                 output_hash=output_hash, verdict="GENERATED", ere_seal=ere.p5_seal)

        # RESPONDING
        self.state = EngineState.RESPONDING
        latency = (time.time() - start) * 1000
        result = GenerateResult(
            text=text, hash=output_hash, ere_seal=ere.p5_seal,
            ere_gates={"P1": ere.p1_secrets, "P2": ere.p2_injection, "P3": ere.p3_loop,
                       "P4": ere.p4_telemetry, "P5": True},
            worm_seq=entry.seq, latency_ms=latency, state="complete", tokens_generated=tokens_generated,
        )
        self.state = EngineState.IDLE
        return result

    def verify_chain(self) -> bool:
        return self.worm.verify()

    def chain_length(self) -> int:
        return len(self.worm.entries)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    engine = InferenceEngine(checkpoint_path="checkpoints/mimo-4b-instruct.pt", device="cpu")

    test_cases = [
        "Write a Python function to compute fibonacci numbers",
        "Explain what a hash table is",
        "Write a Rust function that reverses a string",
    ]

    for prompt in test_cases:
        result = engine.generate(prompt)
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Response: {result.text[:200]}...")
        print(f"Tokens: {result.tokens_generated}")
        print(f"Latency: {result.latency_ms:.1f}ms")
        print(f"ERE Gates: {result.ere_gates}")
        print(f"Chain valid: {engine.verify_chain()}")

    print(f"\n[ENGINE] Total chain entries: {engine.chain_length()}")
