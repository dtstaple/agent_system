# no LLM, the safety centerpiece, this validates the recommended action against a capability policy: is the target enrolled,
# does the action match severity, does the cited evidence actually support it,
# approves > executes; rejects > human review. this is where the safety is enforced.

"""
gate.py - the reference monitor (NO LLM, fully deterministic).

Checks the responder's action against a capability policy before anything is
allowed to touch the firewall. Every rule is evaluated (no short-circuit) so
violated_rules lists every failure, not just the first:
  verdict_ok              verdict.status == "ok"
  verdict_confident       verdict.confidence >= CONFIDENCE_THRESHOLD
  action_ok               action.status == "ok"
  enrolled                evidence.enrolled is True
  target_matches_alert    action.target_device == alert.device
  proportionate           action is in SEVERITY_MATRIX for the verdict's severity
  justification_grounded  justification_ids non-empty, and each is a verdict field
                          ref or a verdict-cited evidence id that ALSO exists in
                          evidence.items (re-checked here, not trusted from the classifier)

SAFETY PROPERTY: every rule reads only structured fields that trusted code wrote
or validated -- enums, numbers, booleans, id sets. The gate never reads LLM free
text (verdict.rationale, the responder's reason). So the policy holds even if
BOTH LLMs upstream are fully hijacked: the worst a hijacked agent can do is pick
an action that is inside the policy anyway, or be rejected to human review.
HIGH_CONFIDENCE is a tuned threshold; this gate is the safety floor regardless
of its value.

Approve -> the execute node is the only code with firewall authority. In dry_run
(default) it logs the exact nft command it would run; GATE_MODE=enforce is a stub
until the Pi's table/chain names and per-device identification exist (Part D).

Writes: "gate_decision" (and appends "audit"). execute writes only "audit".
"""

from __future__ import annotations
import os

from state import (Action, AlertState, Evidence, GateDecision, Verdict,
                   CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE)

# verdict fields an action may cite (besides the verdict's cited evidence ids)
VERDICT_FIELD_IDS = ["verdict.is_attack", "verdict.attack_type", "verdict.confidence"]

# severity -> allowed actions. restrict (least invasive containment) is always
# acceptable for an attack; allow is never acceptable for one.
BENIGN_ACTIONS = {"allow"}
LOW_CONF_ATTACK_ACTIONS = {"restrict"}           # CONFIDENCE_THRESHOLD <= conf < HIGH_CONFIDENCE
HIGH_CONF_ATTACK_ACTIONS = {                     # conf >= HIGH_CONFIDENCE, by attack type
    "brute_force":    {"restrict", "reauth"},
    "dos":            {"restrict", "isolate"},
    "botnet_dos":     {"restrict", "isolate"},
    "c2_beacon":      {"restrict", "isolate"},
    "data_integrity": {"restrict", "isolate", "reauth"},   # permissive: agent keeps genuine choice
    "reconnaissance": {"restrict", "isolate"},
}   # unknown type -> empty set -> reject

GATE_MODE = os.environ.get("GATE_MODE", "dry_run")   # "dry_run" | "enforce"

# PLACEHOLDERS: table/set names and device identifiers are fixed on the Pi in Part D.
# The STM32 is serial-bridged (no MAC), so per-device identification is still open.
NFT_TABLE = "inet zt_gateway"
NFT_COMMANDS = {
    "restrict": "nft add element {table} restricted {{ {dev} }}",
    "isolate":  "nft add element {table} isolated {{ {dev} }}",
    "reauth":   "nft add element {table} reauth_pending {{ {dev} }}",
}


def allowed_justification_ids(verdict: Verdict, evidence_ids: set[str] | None = None) -> set[str]:
    """The only ids an action may cite: verdict field refs + the verdict's cited
    evidence ids. With evidence_ids (the gate), cited ids must also be real evidence;
    the responder calls it without, since it reads only the verdict."""
    cited = set(verdict["cited_evidence_ids"])
    if evidence_ids is not None:
        cited &= evidence_ids
    return set(VERDICT_FIELD_IDS) | cited


def allowed_actions(verdict: Verdict) -> set[str]:
    if verdict["status"] != "ok" or verdict["is_attack"] is None:
        return set()
    if not verdict["is_attack"]:
        return BENIGN_ACTIONS
    if verdict["confidence"] < HIGH_CONFIDENCE:
        return LOW_CONF_ATTACK_ACTIONS
    return HIGH_CONF_ATTACK_ACTIONS.get(verdict["attack_type"], set())


# ---------- the node ----------
def gate(state: AlertState) -> dict:
    """LangGraph node. Reads verdict, action, evidence, alert; returns
    {'gate_decision': ...} plus audit."""
    verdict: Verdict | None = state.get("verdict")
    action: Action | None = state.get("action")
    evidence: Evidence | None = state.get("evidence")
    alert = state.get("alert")
    violated: list[str] = []

    if verdict is None or verdict["status"] != "ok":
        violated.append("verdict_ok")
    if verdict is None or verdict["confidence"] < CONFIDENCE_THRESHOLD:
        violated.append("verdict_confident")
    if action is None or action["status"] != "ok":
        violated.append("action_ok")
    if evidence is None or evidence.get("enrolled") is not True:
        violated.append("enrolled")
    if action is None or alert is None or action["target_device"] != alert["device"]:
        violated.append("target_matches_alert")

    permitted = allowed_actions(verdict) if verdict is not None else set()
    if action is None or action["action"] not in permitted:
        violated.append("proportionate")

    evidence_ids = {it["id"] for it in evidence["items"]} if evidence else set()
    citable = allowed_justification_ids(verdict, evidence_ids) if verdict else set()
    just = action["justification_ids"] if action else []
    rejected_ids = [j for j in just if j not in citable]
    if not just or rejected_ids:
        violated.append("justification_grounded")

    outcome = "reject" if violated else "approve"
    decision = GateDecision(outcome=outcome, violated_rules=violated)
    audit = [{
        "node": "gate",
        "outcome": outcome,
        "violated_rules": violated,
        "action": action["action"] if action else None,
        "permitted_actions": sorted(permitted),       # the matrix entry checked against
        "rejected_justification_ids": rejected_ids,   # fabricated ids the gate caught
    }]
    return {"gate_decision": decision, "audit": audit}


def execute(state: AlertState) -> dict:
    """Terminal node, reached only on gate approve. The only code with firewall authority."""
    action = state["action"]
    if action["action"] == "allow":
        cmd = None   # nothing to change
    else:
        cmd = NFT_COMMANDS[action["action"]].format(table=NFT_TABLE, dev=action["target_device"])

    if GATE_MODE == "enforce" and cmd is not None:
        # Part D: subprocess.run(["nft", ...], check=True) once the Pi's table/chain
        # names and per-device identification (STM32 has no MAC) are known.
        raise NotImplementedError("GATE_MODE=enforce is not wired yet (Part D)")

    return {"audit": [{"node": "execute", "mode": GATE_MODE, "action": action["action"],
                       "target_device": action["target_device"], "nft_command": cmd}]}
