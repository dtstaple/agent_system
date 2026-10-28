# Project Map

## Purpose
Agent layer for an IoT security testbed (Raspberry Pi zero-trust gateway; ESP32 WiFi sensor, STM32 serial-bridged sensor). Agents investigate anomaly alerts and recommend/enforce graduated responses. Research question: safety — resistance to prompt injection and hallucination, and whether a deterministic gate holds when the reasoning agent is hijacked. Class final project / paper.

## Requirements
- Investigator (no LLM) grounds everything in real CSV evidence.
- Classifier (LLM): attack vs benign, type, confidence, must cite evidence; structured JSON.
- Responder (LLM): graduated action (allow / restrict / isolate / re-auth).
- Gate (no LLM): deterministic least-privilege check; only the gate may touch the firewall; reject -> human review.
- Conditional edges: low confidence -> back to investigator; gate reject -> human review.
- Experiments: Part A correctness/consistency/groundedness (replay.py); Part B prompt injection, misinformation, hallucination (adversarial.py); Part C access control.

## Components
Implemented (commits 15427e2, f43bc60; verified 2026-10-27):
- state.py: Alert (alert_id = device@window_start, one ~10s window), EvidenceItem/Evidence (+ enrolled), Verdict, Action, GateDecision, AlertState with append-only `audit`; MAX_INVESTIGATION_ROUNDS = 2. Ground truth excluded; replay joins on alert_id.
- investigator.py: deterministic; DEVICE_REGISTRY (MAC mapping), nearest-row lookup within 5s, baseline means, last 3 history values. Same input -> same evidence.
- classifier.py: single Anthropic messages call (claude-sonnet-4-6), JSON -> Pydantic VerdictModel, drops/flags hallucinated citation ids; parse failure -> benign, confidence 0.
Updated 2026-10-28: state.py (EvidenceRequest, ToolResult, Escalation, Verdict.status), investigator.py (two modes + 3 tools), classifier.py (native tool use, validation_failed), graph.py (wired). Skeleton only: responder.py, gate.py, replay.py, adversarial.py. Dependencies: req.txt (langgraph, langchain-anthropic, anthropic, pydantic, pandas, onnxruntime, ...).

## Main Flow
(planned, from learner's description; not implemented)
alert(window, device) -> investigator -> classifier --low conf--> investigator
                                       classifier -> responder -> gate --approve--> execute (firewall: ? real or simulated)
                                                                   gate --reject--> human review
Where alerts come from (ONNX detector output vs labels.csv windows): ?

Implemented (2026-10-28): classifier evidence-request loop in the graph (graph.py: investigator -> classifier -> route_after_classifier -> investigator | human_review | END[responder placeholder]).
investigator(initial bundle) -> classifier --tool request--> investigator(executes fixed read-only tool) -> classifier ...
  menu: get_more_history(n<=10), get_other_source(), get_baseline_detail(column in MQTT_COLS/FEAT_COLS)
  no device arg (runner injects alert device); unavailable -> explicit evidence item; results join valid citation ids
  cap MAX_INVESTIGATION_ROUNDS=2; cap hit + low confidence -> human_review (same terminal as gate reject)
  classifier writes a request key distinct from verdict; every call logged to audit (post-hoc analysis only)
  initial bundle single primary source (ESP32 features, STM32 mqtt); low conf w/o request -> human_review; validation_failed (is_attack=None) -> human_review.
  verdict only via submit_verdict tool (always offered), tool_choice any; no tool call -> validation_failed.

## Data and Trust Boundaries
data/ (git-ignored CSVs, verified):
- labels.csv: 12 attack windows (label, device, category, intensity, start, end). Devices: esp32-01, stm32-01.
- features.csv / baseline_features.csv: ~10s network-flow windows keyed by ts + mac (only esp32_sensor, f0:16:1d:51:4d:34).
- mqtt_behavior.csv / baseline_mqtt.csv: ~10s MQTT windows keyed by ts + device (esp32-01, stm32-01).
- No ONNX anomaly score column observed in these CSVs.
Trust boundary (planned): LLM outputs untrusted; gate is the only component with firewall authority.

## Build and Deployment
No dependency files, run commands, or config present yet.

## Unknowns
- Alert source and identity (detector output? labeled windows? sliding windows?).
- MAC <-> device-name mapping (identity source).
- Device "baseline" definition and "history" definition.
- LLM provider/model; LangGraph version; dependency management.
- Whether firewall actions are real (nftables on the Pi) or simulated in replay.
