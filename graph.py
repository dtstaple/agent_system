#LangGraph Wiring, connects the nodes, adds the two conditional edges (low confidence loop, gate reject loop)
#
# Routing after the classifier:
#   validation_failed (bad parse / unknown tool / bad args)  -> human_review
#   evidence_request and round cap not reached               -> investigator (runs the tool)
#   evidence_request at/after the cap                        -> human_review
#   verdict below CONFIDENCE_THRESHOLD                       -> human_review
#   verdict at/above CONFIDENCE_THRESHOLD                    -> responder
# Routing after the responder:
#   action validation_failed                                 -> human_review (never reaches the gate)
#   action ok                                                -> gate
# Routing after the gate:
#   approve                                                  -> execute (dry-run nft) -> END
#   reject                                                   -> human_review
# human_review is the single "don't act autonomously" terminal: validation failure
# (classifier or responder), low confidence, and gate rejection all end there.

from __future__ import annotations
from typing import Literal

from langgraph.graph import END, StateGraph

from state import (AlertState, Escalation, EscalationReason,
                   CONFIDENCE_THRESHOLD, MAX_INVESTIGATION_ROUNDS)
from investigator import investigate
from classifier import classify
from responder import respond
from gate import gate, execute


def _escalation_reason(state: AlertState) -> EscalationReason:
    if state.get("gate_decision", {}).get("outcome") == "reject":
        return "gate_reject"
    verdict = state.get("verdict")
    if verdict is None or verdict["status"] == "validation_failed":
        return "validation_failed"
    action = state.get("action")
    if action is not None and action["status"] == "validation_failed":
        return "validation_failed"
    return "low_confidence"


def human_review(state: AlertState) -> dict:
    """Terminal node: record why the alert was escalated instead of acted on."""
    alert_id = state["alert"]["alert_id"]
    reason = _escalation_reason(state)
    entry = {"node": "human_review", "alert_id": alert_id, "reason": reason}
    if reason == "gate_reject":
        entry["violated_rules"] = state["gate_decision"]["violated_rules"]
    return {"escalation": Escalation(reason=reason, alert_id=alert_id), "audit": [entry]}


def route_after_classifier(state: AlertState) -> Literal["investigator", "human_review", "responder"]:
    if state.get("evidence_request") is not None:
        if state.get("investigation_round", 0) < MAX_INVESTIGATION_ROUNDS:
            return "investigator"
        return "human_review"
    verdict = state.get("verdict")
    if verdict is None or verdict["status"] != "ok":
        return "human_review"
    if verdict["confidence"] < CONFIDENCE_THRESHOLD:
        return "human_review"
    return "responder"


def route_after_responder(state: AlertState) -> Literal["gate", "human_review"]:
    action = state.get("action")
    if action is None or action["status"] != "ok":
        return "human_review"
    return "gate"


def route_after_gate(state: AlertState) -> Literal["execute", "human_review"]:
    if state.get("gate_decision", {}).get("outcome") == "approve":
        return "execute"
    return "human_review"


def build_graph():
    g = StateGraph(AlertState)
    g.add_node("investigator", investigate)
    g.add_node("classifier", classify)
    g.add_node("responder", respond)
    g.add_node("gate", gate)
    g.add_node("execute", execute)
    g.add_node("human_review", human_review)

    g.set_entry_point("investigator")
    g.add_edge("investigator", "classifier")
    g.add_conditional_edges("classifier", route_after_classifier,
                            {"investigator": "investigator",
                             "human_review": "human_review",
                             "responder": "responder"})
    g.add_conditional_edges("responder", route_after_responder,
                            {"gate": "gate", "human_review": "human_review"})
    g.add_conditional_edges("gate", route_after_gate,
                            {"execute": "execute", "human_review": "human_review"})
    g.add_edge("execute", END)
    g.add_edge("human_review", END)
    return g.compile()


# ---------- standalone test ----------
if __name__ == "__main__":
    import pprint
    from state import Alert
    # value_drift attack window on the STM32 (from labels.csv: ~1792886067)
    alert = Alert(alert_id="stm32-01@1792886100", device="stm32-01",
                  window_start="1792886100", window_end="1792886110")
    final = build_graph().invoke({"alert": alert})
    print("tool_history:")
    pprint.pprint(final.get("tool_history", []))
    print("\nverdict:")
    pprint.pprint(final.get("verdict"))
    print("\naction:")
    pprint.pprint(final.get("action"))
    print("\ngate_decision:", final.get("gate_decision"))
    print("escalation:", final.get("escalation"))
    print("\naudit:")
    for a in final["audit"]:
        print(" ", a)
