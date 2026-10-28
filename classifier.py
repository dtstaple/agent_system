"""
classifier.py - the reasoning agent (LLM).

Takes the Evidence bundle the investigator assembled and decides:
  - is this an attack or benign?
  - if attack, which type?
  - confidence 0-1
  - WHICH evidence IDs justify the call (groundedness)
  - a short rationale (kept for the paper; the gate never reads it)

Safety design:
  - LLM output is parsed into a Pydantic model (VerdictModel) FIRST. A malformed or
    prompt-injected response fails validation here and never enters AlertState.
  - cited_evidence_ids are checked against the real evidence; hallucinated IDs are
    dropped and flagged, so groundedness is measurable downstream.
  - The prompt shows ONLY the investigator's evidence. Ground truth is never present.

Writes exactly one state key: "verdict".
"""

from __future__ import annotations
import json
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from anthropic import Anthropic

from state import AlertState, Evidence, Verdict

load_dotenv()

MODEL = "claude-sonnet-4-6"   # capable + consistent for the study
_client = Anthropic()         # reads ANTHROPIC_API_KEY from env

# the label vocabulary the classifier must choose from -- matches labels.csv categories
ATTACK_TYPES = [
    "reconnaissance",     # port_scan, host_sweep
    "dos",                # syn/udp/icmp flood
    "brute_force",        # mqtt_brute
    "data_integrity",     # stm32_value_drift
    "c2_beacon",          # stm32_beacon
    "botnet_dos",         # stm32_rate_spike
    "benign",             # not an attack
]


# ---------- Pydantic validation boundary (untrusted LLM output lands here) ----------
class VerdictModel(BaseModel):
    is_attack: bool
    attack_type: Literal[tuple(ATTACK_TYPES)]  # type: ignore
    confidence: float = Field(ge=0.0, le=1.0)
    cited_evidence_ids: list[str]
    rationale: str = Field(max_length=1000)


SYSTEM = """You are a security reasoning agent for an IoT gateway. You are given \
evidence about one traffic window from one device: its current feature values, the \
device's normal baseline for comparison, its identity, and recent history.

Decide whether this window represents an ATTACK or BENIGN behavior. If an attack, \
choose the single best type from the allowed list.

Rules you must follow:
- Base your decision ONLY on the evidence provided. Do not invent values.
- You MUST cite the evidence ids (the "id" field of each item) that justify your \
decision in cited_evidence_ids. Cite the specific ids whose values drove your call.
- Compare current values to their matching baseline.* ids to judge deviation.
- A device with normal traffic rate/size but an abnormal val_mean indicates data \
integrity / sensor spoofing, not a network attack.
- A device that went silent (msg_count/pkt_count near zero vs a nonzero baseline) \
may indicate a DoS that knocked it offline.
- Respond with ONLY a JSON object, no prose outside it, matching:
  {"is_attack": bool, "attack_type": one of %s,
   "confidence": 0..1, "cited_evidence_ids": [ids], "rationale": "short"}""" % ATTACK_TYPES


def _format_evidence(ev: Evidence) -> str:
    lines = [f'  {it["id"]} = {it["value"]}' for it in ev["items"]]
    lines.append(f'  device_enrolled = {ev["enrolled"]}')
    return "\n".join(lines)


def _call_llm(evidence_text: str) -> str:
    msg = _client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM,
        messages=[{"role": "user",
                   "content": f"Evidence for this window:\n{evidence_text}\n\nReturn the JSON verdict."}],
    )
    # concatenate any text blocks
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def _parse(raw: str) -> VerdictModel | None:
    """Strip code fences, parse JSON, validate with Pydantic. None on any failure."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    try:
        data = json.loads(s)
        return VerdictModel(**data)
    except (json.JSONDecodeError, ValidationError, IndexError, TypeError):
        return None


# ---------- the node ----------
def classify(state: AlertState) -> dict:
    """LangGraph node. Reads state['evidence'], returns {'verdict': Verdict, 'audit': [...]}."""
    ev: Evidence = state["evidence"]
    valid_ids = {it["id"] for it in ev["items"]}
    evidence_text = _format_evidence(ev)

    raw = _call_llm(evidence_text)
    vm = _parse(raw)

    if vm is None:
        # validation failed (malformed or injected) -> safe default, flagged
        verdict = Verdict(is_attack=False, attack_type="benign", confidence=0.0,
                          cited_evidence_ids=[], rationale="LLM output failed validation")
        audit = [{"node": "classifier", "status": "parse_failed", "raw_preview": raw[:200]}]
        return {"verdict": verdict, "audit": audit}

    # groundedness check: drop any cited id that isn't real evidence
    real = [cid for cid in vm.cited_evidence_ids if cid in valid_ids]
    hallucinated = [cid for cid in vm.cited_evidence_ids if cid not in valid_ids]

    verdict = Verdict(
        is_attack=vm.is_attack,
        attack_type=vm.attack_type,
        confidence=vm.confidence,
        cited_evidence_ids=real,
        rationale=vm.rationale,
    )
    audit = [{
        "node": "classifier",
        "status": "ok",
        "is_attack": vm.is_attack,
        "attack_type": vm.attack_type,
        "confidence": vm.confidence,
        "cited_valid": real,
        "cited_hallucinated": hallucinated,   # groundedness signal for the paper
    }]
    return {"verdict": verdict, "audit": audit}


# ---------- standalone test ----------
if __name__ == "__main__":
    from investigator import investigate
    from state import Alert
    # same value_drift window the investigator test used
    alert = Alert(alert_id="stm32-01@1792886100", device="stm32-01",
                  window_start="1792886100", window_end="1792886110")
    ev_out = investigate({"alert": alert})
    result = classify({"evidence": ev_out["evidence"]})
    import pprint
    pprint.pprint(result["verdict"])
    print("\naudit:", result["audit"][0])