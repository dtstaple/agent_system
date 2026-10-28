#defines AlertState, the dict that flows through the graph. Holds the window data, retrieve evidence,
# classifier verdict, the responder action, and the gate descision, every node reads/writes it
#
# Each top-level key has exactly one writer so write scope can be audited in the adversarial runs:
#   investigator -> evidence, tool_history, investigation_round
#   classifier   -> verdict, evidence_request
#   responder    -> action
#   gate         -> gate_decision
#   human_review -> escalation
# Ground truth (labels.csv) is never in here -- replay.py joins it on alert_id after the run.

import operator
from typing import Annotated, Any, Literal, TypedDict

MAX_INVESTIGATION_ROUNDS = 2  # max tool executions in the classifier <-> investigator loop
CONFIDENCE_THRESHOLD = 0.6    # verdicts below this escalate to human review
HISTORY_MAX_N = 10            # upper bound on get_more_history(n)

ActionType = Literal["allow", "restrict", "isolate", "reauth"]
GateOutcome = Literal["approve", "reject"]
EvidenceSource = Literal["features", "mqtt", "baseline", "identity", "history", "tool"]

# the classifier's fixed, read-only tool menu -- executed deterministically by the investigator
ToolName = Literal["get_more_history", "get_other_source", "get_baseline_detail"]
ToolStatus = Literal["ok", "unavailable", "refused"]
VerdictStatus = Literal["ok", "validation_failed"]
EscalationReason = Literal["validation_failed", "low_confidence", "gate_reject"]


class Alert(TypedDict):
    alert_id: str        # f"{device}@{window_start}"
    device: str          # e.g. "esp32-01"
    window_start: str    # unix ts, one ~10s window
    window_end: str


class EvidenceItem(TypedDict):
    id: str              # stable, e.g. "features.pkt_rate" -- what the classifier must cite
    source: EvidenceSource
    key: str
    value: Any


class Evidence(TypedDict):
    items: list[EvidenceItem]
    enrolled: bool       # device is in the identity registry; gate checks this directly


class EvidenceRequest(TypedDict):
    tool: ToolName
    args: dict[str, Any]  # never contains a device -- the investigator injects it from the alert
    round: int            # the round this request will run as (1-based)


class ToolResult(TypedDict):
    request: EvidenceRequest
    status: ToolStatus
    item_ids: list[str]   # evidence ids this call added (citable)


class Verdict(TypedDict):
    status: VerdictStatus
    is_attack: bool | None   # None when validation failed -- never read as "benign"
    attack_type: str | None
    confidence: float    # 0.0-1.0
    cited_evidence_ids: list[str]
    rationale: str       # stored for the paper; the gate never reads it


class Action(TypedDict):
    action: ActionType
    target_device: str
    justification_ids: list[str]


class GateDecision(TypedDict):
    outcome: GateOutcome
    violated_rules: list[str]


class Escalation(TypedDict):
    reason: EscalationReason
    alert_id: str


class AlertState(TypedDict, total=False):
    alert: Alert
    evidence: Evidence
    evidence_request: EvidenceRequest | None   # pending tool call; None once a verdict is returned
    tool_history: Annotated[list[ToolResult], operator.add]  # full request/result record
    verdict: Verdict
    action: Action
    gate_decision: GateDecision
    escalation: Escalation
    investigation_round: int   # number of tool executions so far
    audit: Annotated[list[dict], operator.add]  # each node appends; LangGraph concatenates
