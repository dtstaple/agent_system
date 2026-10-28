#defines AlertState, the dict that flows through the graph. Holds the window data, retrieve evidence,
# classifier verdict, the responder action, and the gate descision, every node reads/writes it
#
# Each node writes exactly one top-level key (investigator->evidence, classifier->verdict,
# responder->action, gate->gate_decision) so write scope can be audited in the adversarial runs.
# Ground truth (labels.csv) is never in here -- replay.py joins it on alert_id after the run.

import operator
from typing import Annotated, Any, Literal, TypedDict

MAX_INVESTIGATION_ROUNDS = 2  # bounds the low-confidence classifier -> investigator loop

ActionType = Literal["allow", "restrict", "isolate", "reauth"]
GateOutcome = Literal["approve", "reject"]
EvidenceSource = Literal["features", "mqtt", "baseline", "identity", "history"]


class Alert(TypedDict):
    alert_id: str        # f"{device}@{window_start}"
    device: str          # e.g. "esp32-01"
    window_start: str    # ISO timestamp, one ~10s window
    window_end: str


class EvidenceItem(TypedDict):
    id: str              # stable, e.g. "features.pkt_rate" -- what the classifier must cite
    source: EvidenceSource
    key: str
    value: Any


class Evidence(TypedDict):
    items: list[EvidenceItem]
    enrolled: bool       # device is in the identity registry; gate checks this directly


class Verdict(TypedDict):
    is_attack: bool
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


class AlertState(TypedDict, total=False):
    alert: Alert
    evidence: Evidence
    verdict: Verdict
    action: Action
    gate_decision: GateDecision
    investigation_round: int
    audit: Annotated[list[dict], operator.add]  # each node appends; LangGraph concatenates
