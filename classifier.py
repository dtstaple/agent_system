"""
classifier.py - the reasoning agent (LLM).

Takes the Evidence bundle the investigator assembled and either:
  - requests more evidence: ONE call from a fixed, read-only tool menu
    (get_more_history / get_other_source / get_baseline_detail), which the
    deterministic investigator executes -- the LLM chooses, trusted code fetches; or
  - returns a verdict via the submit_verdict tool (always offered; free text is
    never parsed -- a response without a tool call is a validation failure):
      - is this an attack or benign?
      - if attack, which type?
      - confidence 0-1
      - WHICH evidence IDs justify the call (groundedness)
      - a short rationale (kept for the paper; the gate never reads it)

Safety design:
  - LLM output is parsed into Pydantic models (VerdictModel / tool arg models) FIRST.
    A malformed or prompt-injected response fails validation here, becomes
    status="validation_failed" (is_attack=None, never "benign"), and the graph
    routes it to human review.
  - Tools take no device argument; the investigator injects the alert's device.
  - Tools are only offered while investigation_round < MAX_INVESTIGATION_ROUNDS.
  - cited_evidence_ids are checked against the real evidence (including tool
    results); hallucinated IDs are dropped and flagged, so groundedness is
    measurable downstream.
  - The prompt shows ONLY the investigator's evidence. Ground truth is never present.

Writes: "verdict", "evidence_request" (and appends "audit").
"""

from __future__ import annotations
import json
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from anthropic import Anthropic

from state import (AlertState, Evidence, EvidenceRequest, Verdict,
                   HISTORY_MAX_N, MAX_INVESTIGATION_ROUNDS)
from investigator import MQTT_COLS, FEAT_COLS

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

ALLOWED_COLUMNS = list(dict.fromkeys(MQTT_COLS + FEAT_COLS))  # dedup, keep order


# ---------- Pydantic validation boundary (untrusted LLM output lands here) ----------
class VerdictModel(BaseModel):
    is_attack: bool
    attack_type: Literal[tuple(ATTACK_TYPES)]  # type: ignore
    confidence: float = Field(ge=0.0, le=1.0)
    cited_evidence_ids: list[str]
    rationale: str = Field(max_length=1000)

    @model_validator(mode="after")
    def _consistent(self):
        if self.is_attack != (self.attack_type != "benign"):
            raise ValueError("is_attack contradicts attack_type")
        return self


class _ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

class HistoryArgs(_ToolArgs):
    n: int = Field(ge=1, le=HISTORY_MAX_N)

class OtherSourceArgs(_ToolArgs):
    pass

class BaselineDetailArgs(_ToolArgs):
    column: Literal[tuple(ALLOWED_COLUMNS)]  # type: ignore

TOOL_ARG_MODELS: dict[str, type[_ToolArgs]] = {
    "get_more_history": HistoryArgs,
    "get_other_source": OtherSourceArgs,
    "get_baseline_detail": BaselineDetailArgs,
}

TOOL_SPECS = [
    {"name": "get_more_history",
     "description": f"Fetch the n windows before this one (1..{HISTORY_MAX_N}) for every "
                    "source already in the evidence. Use to check whether the deviation is "
                    "sustained or a one-off.",
     "input_schema": {"type": "object",
                      "properties": {"n": {"type": "integer", "minimum": 1, "maximum": HISTORY_MAX_N}},
                      "required": ["n"], "additionalProperties": False}},
    {"name": "get_other_source",
     "description": "Fetch this device's secondary data source for the same window "
                    "(e.g. MQTT behavior for a device whose bundle has network features). "
                    "Returns an explicit 'unavailable' item if the device has none.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "get_baseline_detail",
     "description": "Fetch mean, std, p5 and p95 of one column over this device's "
                    "clean baseline, to judge whether a value is outside normal spread.",
     "input_schema": {"type": "object",
                      "properties": {"column": {"type": "string", "enum": ALLOWED_COLUMNS}},
                      "required": ["column"], "additionalProperties": False}},
]

# the only way to conclude -- always offered, even at the round cap
SUBMIT_VERDICT_SPEC = {
    "name": "submit_verdict",
    "description": "Submit your final verdict for this window. Call this once you have "
                   "enough evidence (or no tool rounds remain).",
    "input_schema": {"type": "object",
                     "properties": {
                         "is_attack": {"type": "boolean"},
                         "attack_type": {"type": "string", "enum": ATTACK_TYPES},
                         "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                         "cited_evidence_ids": {"type": "array", "items": {"type": "string"}},
                         "rationale": {"type": "string", "maxLength": 1000}},
                     "required": ["is_attack", "attack_type", "confidence",
                                  "cited_evidence_ids", "rationale"],
                     "additionalProperties": False},
}


SYSTEM = """You are a security reasoning agent for an IoT gateway. You are given \
evidence about one traffic window from one device: its current feature values from ONE \
primary data source, the device's normal baseline for comparison, its identity, and \
recent history. The evidence may also include results of tools you called earlier \
(ids starting with "tool.rN.").

Decide whether this window represents an ATTACK or BENIGN behavior. If an attack, \
choose the single best type from the allowed list.

Every response must be exactly ONE tool call. If the evidence is not enough to decide \
confidently and evidence tools are available, call one of them. Otherwise call \
submit_verdict.

Rules you must follow:
- Base your decision ONLY on the evidence provided. Do not invent values.
- You MUST cite the evidence ids (the "id" field of each item) that justify your \
decision in cited_evidence_ids. Cite the specific ids whose values drove your call.
- Compare current values to their matching baseline ids to judge deviation.
- An item marked unavailable or refused is real evidence that the data does not exist; \
you may cite it.
- A device with normal traffic rate/size but an abnormal val_mean indicates data \
integrity / sensor spoofing, not a network attack.
- A device that went silent (msg_count/pkt_count near zero vs a nonzero baseline) \
may indicate a DoS that knocked it offline.
- is_attack must be false exactly when attack_type is "benign"."""


def _format_evidence(ev: Evidence) -> str:
    lines = [f'  {it["id"]} = {it["value"]}' for it in ev["items"]]
    lines.append(f'  device_enrolled = {ev["enrolled"]}')
    return "\n".join(lines)


def _format_requests(state: AlertState) -> str:
    hist = state.get("tool_history", [])
    if not hist:
        return "  (none)"
    return "\n".join(f'  round {r["request"]["round"]}: {r["request"]["tool"]}'
                     f'({json.dumps(r["request"]["args"])}) -> {r["status"]}' for r in hist)


def _call_llm(user_text: str, offer_tools: bool):
    """Fresh single-turn call each round: the prompt is rebuilt from state, so no
    message history lives in AlertState. tool_choice "any" forces a tool call, so
    the model can't answer in free text; submit_verdict is always in the menu."""
    tools = (TOOL_SPECS if offer_tools else []) + [SUBMIT_VERDICT_SPEC]
    return _client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_text}],
        tools=tools,
        tool_choice={"type": "any"},
    )


def _failed(raw: str, why: str) -> dict:
    """validation_failed -> human review. is_attack=None so nothing can read it as benign."""
    verdict = Verdict(status="validation_failed", is_attack=None, attack_type=None,
                      confidence=0.0, cited_evidence_ids=[],
                      rationale="LLM output failed validation")
    audit = [{"node": "classifier", "status": "validation_failed", "reason": why,
              "raw_preview": raw[:2000]}]
    return {"verdict": verdict, "evidence_request": None, "audit": audit}


# ---------- the node ----------
def classify(state: AlertState) -> dict:
    """LangGraph node. Reads state['evidence'] (+ tool_history), returns either
    {'evidence_request': ...} or {'verdict': ..., 'evidence_request': None}, plus audit."""
    ev: Evidence = state["evidence"]
    valid_ids = {it["id"] for it in ev["items"]}   # includes tool results already merged in
    rnd = state.get("investigation_round", 0)
    rounds_left = MAX_INVESTIGATION_ROUNDS - rnd
    offer_tools = rounds_left > 0

    user_text = (f"Evidence for this window:\n{_format_evidence(ev)}\n\n"
                 f"Evidence requests so far:\n{_format_requests(state)}\n\n"
                 f"Tool rounds remaining: {rounds_left}\n\n"
                 + ("Call one evidence tool, or submit_verdict." if offer_tools
                    else "No evidence tools remain. Call submit_verdict."))

    msg = _call_llm(user_text, offer_tools)
    tool_uses = [b for b in msg.content if b.type == "tool_use"]
    raw = "".join(b.text for b in msg.content if b.type == "text").strip()

    if not tool_uses:
        return _failed(raw, "no tool call")

    first = tool_uses[0]   # one call per round; extras are logged, never run
    raw_call = json.dumps({"tool": first.name, "input": first.input})
    ignored = [json.dumps({"tool": b.name, "input": b.input}) for b in tool_uses[1:]]

    # ---- verdict path ----
    if first.name == "submit_verdict":
        try:
            vm = VerdictModel(**first.input)
        except (ValidationError, TypeError):
            return _failed(raw_call, "invalid verdict")
        return _verdict(vm, valid_ids, rnd, ignored)

    # ---- evidence request path ----
    if not offer_tools:
        return _failed(raw_call, "tool call after round cap")
    model = TOOL_ARG_MODELS.get(first.name)
    if model is None:
        return _failed(raw_call, "unknown tool")
    try:
        args = model(**first.input).model_dump()
    except (ValidationError, TypeError):
        return _failed(raw_call, "invalid tool args")

    req = EvidenceRequest(tool=first.name, args=args, round=rnd + 1)
    audit = [{"node": "classifier", "status": "tool_request", "round": rnd + 1,
              "tool": first.name, "args": args, "ignored_extra_calls": ignored,
              "text_preview": raw[:2000]}]
    return {"evidence_request": req, "audit": audit}


def _verdict(vm: VerdictModel, valid_ids: set[str], rnd: int, ignored: list[str]) -> dict:

    # groundedness check: drop any cited id that isn't real evidence
    real = [cid for cid in vm.cited_evidence_ids if cid in valid_ids]
    hallucinated = [cid for cid in vm.cited_evidence_ids if cid not in valid_ids]

    verdict = Verdict(
        status="ok",
        is_attack=vm.is_attack,
        attack_type=vm.attack_type,
        confidence=vm.confidence,
        cited_evidence_ids=real,
        rationale=vm.rationale,
    )
    audit = [{
        "node": "classifier",
        "status": "ok",
        "round": rnd,
        "is_attack": vm.is_attack,
        "attack_type": vm.attack_type,
        "confidence": vm.confidence,
        "cited_valid": real,
        "cited_hallucinated": hallucinated,   # groundedness signal for the paper
        "ignored_extra_calls": ignored,
    }]
    return {"verdict": verdict, "evidence_request": None, "audit": audit}


# ---------- standalone test ----------
if __name__ == "__main__":
    from investigator import investigate
    from state import Alert
    # same value_drift window the investigator test used (single classifier step, no loop)
    alert = Alert(alert_id="stm32-01@1792886100", device="stm32-01",
                  window_start="1792886100", window_end="1792886110")
    state = {"alert": alert, **investigate({"alert": alert})}
    result = classify(state)
    import pprint
    pprint.pprint({k: v for k, v in result.items() if k != "audit"})
    print("\naudit:", result["audit"][0])
