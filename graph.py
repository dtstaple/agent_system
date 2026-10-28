#LangGraph Wiring, connects the nodes, adds the two conditional edges (low confidence loop, gate reject loop)
#
# Routing after the classifier:
#   validation_failed (bad parse / unknown tool / bad args)  -> human_review
#   evidence_request and round cap not reached               -> investigator (runs the tool)
#   evidence_request at/after the cap                        -> human_review
#   verdict below CONFIDENCE_THRESHOLD                       -> human_review
#   verdict at/above CONFIDENCE_THRESHOLD                    -> responder (END until it exists)
# human_review is the single "don't act autonomously" terminal: validation failure,
# low confidence, and (once gate.py exists) gate rejection all end there.

from __future__ import annotations
from typing import Literal

from langgraph.graph import END, StateGraph

from state import (AlertState, Escalation, EscalationReason,
                   CONFIDENCE_THRESHOLD, MAX_INVESTIGATION_ROUNDS)
from investigator import investigate
from classifier import classify


def _escalation_reason(state: AlertState) -> EscalationReason:
    if state.get("gate_decision", {}).get("outcome") == "reject":
        return "gate_reject"
    verdict = state.get("verdict")
    if verdict is None or verdict["status"] == "validation_failed":
        return "validation_failed"
    return "low_confidence"


def human_review(state: AlertState) -> dict:
    """Terminal node: record why the alert was escalated instead of acted on."""
    alert_id = state["alert"]["alert_id"]
    reason = _escalation_reason(state)
    return {"escalation": Escalation(reason=reason, alert_id=alert_id),
            "audit": [{"node": "human_review", "alert_id": alert_id, "reason": reason}]}


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


def build_graph():
    g = StateGraph(AlertState)
    g.add_node("investigator", investigate)
    g.add_node("classifier", classify)
    g.add_node("human_review", human_review)

    g.set_entry_point("investigator")
    g.add_edge("investigator", "classifier")
    g.add_conditional_edges("classifier", route_after_classifier,
                            {"investigator": "investigator",
                             "human_review": "human_review",
                             "responder": END})   # TODO: responder -> gate once built
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
    print("\nescalation:", final.get("escalation"))
    print("\naudit:")
    for a in final["audit"]:
        print(" ", a)
