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
All files are comment-only skeletons (commit b8424a2): state.py, investigator.py, classifier.py, responder.py, gate.py, graph.py, replay.py, adversarial.py (empty), responder.py.

## Main Flow
(planned, from learner's description; not implemented)
alert(window, device) -> investigator -> classifier --low conf--> investigator
                                       classifier -> responder -> gate --approve--> execute (firewall: ? real or simulated)
                                                                   gate --reject--> human review
Where alerts come from (ONNX detector output vs labels.csv windows): ?

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
