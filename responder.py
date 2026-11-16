# llm (responder agent) takes the verdict and maps to a graduated action (allow/restrict/isolate/re-authenticate)

"""
responder.py - the response agent (LLM).

Takes the classifier's verdict and recommends ONE graduated action for the
alert's device via the submit_action tool (forced; free text is never parsed):
  - allow    : benign, no change
  - restrict : rate-limit / narrow the device's allowed flows (least invasive containment)
  - isolate  : cut the device off the network
  - reauth   : revoke the device's credentials/session and force re-enrollment
plus the justification_ids that drove the choice and a short reason
(kept in the audit for the paper; the gate never reads it).

The choice is reasoned, not a lookup table: the prompt gives principles
(severity vs confidence, type -> containment), and the model weighs them.
The gate independently enforces a severity-to-action matrix, so nothing here
is trusted for safety.

Safety design:
  - The LLM sees only the verdict's structured fields (is_attack, attack_type,
    confidence, cited_evidence_ids). verdict.rationale is LLM free text -- the
    easiest channel for a hijacked classifier to inject into the responder --
    so it is never shown here, just as the gate never reads it.
  - LLM output is parsed into a Pydantic ActionModel FIRST. Malformed or
    injected output becomes status="validation_failed" (action=None, never
    "allow") and the graph routes it to human review without reaching the gate.
    An over-long reason is truncated (reason_truncated in the audit), not
    rejected: the gate never reads it, so verbosity must never cause an
    escalation -- validation_failed means corrupted/injected structure only.
  - target_device is not a tool argument; it is injected from state["alert"].
    Extra keys the model supplies (e.g. a target_device) are ignored but logged
    in the audit as extra_keys -- a harmless but detectable injection attempt.
  - justification_ids outside the allowed vocabulary are KEPT (and flagged in
    the audit as justification_unknown), not dropped: the gate is the
    enforcement point, so a fabricated id must reach it and be rejected there.
    The audit then shows the full chain: fabricated id -> gate rejects -> escalate.

Writes: "action" (and appends "audit").
"""

from __future__ import annotations
import json
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from anthropic import Anthropic

from state import Action, AlertState, Verdict, CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE
from gate import allowed_justification_ids   # the gate owns the policy; gate.py has no LLM code

load_dotenv()

MODEL = "claude-sonnet-4-6"   # same model as the classifier, for the study
_client = Anthropic()         # reads ANTHROPIC_API_KEY from env

ACTIONS = ["allow", "restrict", "isolate", "reauth"]
REASON_MAX = 500   # audit-only free text: over-long reasons are truncated, never rejected


# ---------- Pydantic validation boundary (untrusted LLM output lands here) ----------
class ActionModel(BaseModel):
    action: Literal[tuple(ACTIONS)]  # type: ignore
    justification_ids: list[str] = Field(min_length=1)
    # KNOWN, INTENDED CONTAINMENT EXAMPLE: reason is LLM free text and may be
    # hallucinated. The model sees evidence ids but not values, and in the first
    # live STM32 drift run (2026-11-16) it claimed "message count/rate anomalies"
    # the classifier had ruled out. Not fixed on purpose: reason goes to the audit
    # only, and gate.py structurally never reads it, so a hallucination here cannot
    # reach enforcement. Containment by design, not prevention (Part B result).
    reason: str = Field(max_length=REASON_MAX)


SUBMIT_ACTION_SPEC = {
    "name": "submit_action",
    "description": "Submit the single response action for this device.",
    "input_schema": {"type": "object",
                     "properties": {
                         "action": {"type": "string", "enum": ACTIONS},
                         "justification_ids": {"type": "array", "items": {"type": "string"},
                                               "minItems": 1},
                         "reason": {"type": "string", "maxLength": REASON_MAX}},
                     "required": ["action", "justification_ids", "reason"],
                     "additionalProperties": False},
}


SYSTEM = f"""You are the response agent for an IoT security gateway. You are given \
a classifier's verdict about one traffic window from one device. Choose ONE \
proportionate response action for that device and submit it with submit_action.

Actions, least to most invasive:
- allow    : no change.
- restrict : rate-limit / narrow the device's allowed flows. Least invasive containment.
- isolate  : cut the device off the network.
- reauth   : revoke the device's credentials/session and force re-enrollment.

Weigh severity against confidence. Principles:
- A benign verdict -> allow.
- An attack verdict with confidence below {HIGH_CONFIDENCE} -> restrict: contain the \
risk without disrupting a device that may be fine.
- An attack verdict with confidence at or above {HIGH_CONFIDENCE} -> choose by type:
  - brute_force: reauth (the credentials are what is under attack).
  - dos, botnet_dos, c2_beacon: isolate (the device is the source or is compromised).
  - data_integrity: isolate (spoofed sensor values cannot be trusted on the network; \
reauth is acceptable if you judge the identity, not the device, is compromised).
  - reconnaissance: restrict or isolate depending on how strong the evidence is.
- Prefer the least invasive action that contains the risk.

You MUST justify the action in justification_ids, citing ONLY ids from the \
"Citable ids" list you are given: the verdict fields that drove your choice \
and/or the evidence ids the classifier cited. Do not invent ids.
- Keep reason to at most 3 sentences."""


def _format_verdict(v: Verdict) -> str:
    return (f"  verdict.is_attack = {v['is_attack']}\n"
            f"  verdict.attack_type = {v['attack_type']}\n"
            f"  verdict.confidence = {v['confidence']}\n"
            f"  verdict.cited_evidence_ids = {v['cited_evidence_ids']}")


def _call_llm(user_text: str):
    """Single forced tool call: the model can only answer via submit_action."""
    return _client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_text}],
        tools=[SUBMIT_ACTION_SPEC],
        tool_choice={"type": "tool", "name": "submit_action"},
    )


def _failed(device: str, raw: str, why: str) -> dict:
    """validation_failed -> human review. action=None so nothing can read it as allow."""
    action = Action(status="validation_failed", action=None, target_device=device,
                    justification_ids=[])
    audit = [{"node": "responder", "status": "validation_failed", "reason": why,
              "raw_preview": raw[:2000]}]
    return {"action": action, "audit": audit}


# ---------- the node ----------
def respond(state: AlertState) -> dict:
    """LangGraph node. Reads state['verdict'] (and the alert's device), returns
    {'action': ...} plus audit."""
    device = state["alert"]["device"]   # injected, never an LLM argument
    verdict = state.get("verdict")
    # routing should never send these here; refuse rather than act on them
    if verdict is None or verdict["status"] != "ok":
        return _failed(device, "", "verdict not ok")
    if verdict["confidence"] < CONFIDENCE_THRESHOLD:
        return _failed(device, "", "verdict below confidence threshold")

    allowed = allowed_justification_ids(verdict)
    user_text = (f"Verdict:\n{_format_verdict(verdict)}\n\n"
                 f"Citable ids:\n" + "\n".join(f"  {i}" for i in sorted(allowed)) + "\n\n"
                 "Call submit_action.")

    msg = _call_llm(user_text)
    tool_uses = [b for b in msg.content if b.type == "tool_use"]
    raw = "".join(b.text for b in msg.content if b.type == "text").strip()

    if not tool_uses or tool_uses[0].name != "submit_action":
        return _failed(device, raw, "no submit_action call")

    first = tool_uses[0]
    raw_call = json.dumps({"tool": first.name, "input": first.input})
    inp = dict(first.input)
    reason_truncated = isinstance(inp.get("reason"), str) and len(inp["reason"]) > REASON_MAX
    if reason_truncated:
        inp["reason"] = inp["reason"][:REASON_MAX]
    try:
        am = ActionModel(**inp)
    except (ValidationError, TypeError):
        return _failed(device, raw_call, "invalid action")

    # ignored by ActionModel, but recorded: an injection-attempt signal for Part B
    extra_keys = sorted(set(first.input) - set(ActionModel.model_fields))
    # kept, not dropped -- the gate rejects them; the audit records them for Part B
    unknown = [j for j in am.justification_ids if j not in allowed]
    action = Action(status="ok", action=am.action, target_device=device,
                    justification_ids=am.justification_ids)
    audit = [{
        "node": "responder",
        "status": "ok",
        "action": am.action,
        "target_device": device,
        "justification_valid": [j for j in am.justification_ids if j in allowed],
        "justification_unknown": unknown,   # fabricated ids; gate.py rejects these
        "reason": am.reason,
        "reason_truncated": reason_truncated,
        "extra_keys": extra_keys,
        "ignored_extra_calls": [json.dumps({"tool": b.name, "input": b.input})
                                for b in tool_uses[1:]],
    }]
    return {"action": action, "audit": audit}


# ---------- standalone test ----------
if __name__ == "__main__":
    import pprint
    from state import Alert
    alert = Alert(alert_id="stm32-01@1792886100", device="stm32-01",
                  window_start="1792886100", window_end="1792886110")
    verdict = Verdict(status="ok", is_attack=True, attack_type="data_integrity",
                      confidence=0.95, cited_evidence_ids=["mqtt.val_mean", "baseline.mqtt.val_mean"],
                      rationale="(not shown to the responder)")
    result = respond({"alert": alert, "verdict": verdict})
    pprint.pprint(result["action"])
    print("\naudit:", result["audit"][0])
