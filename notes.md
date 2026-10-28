# Notes

## TL;DR — where it's at
- **Working on real data:** investigator → classifier evidence loop, wired in `graph.py`.
  - Investigator: deterministic, single-source initial bundle (ESP32 → features, STM32 → MQTT), runs a fixed read-only tool menu (`get_more_history`, `get_other_source`, `get_baseline_detail`).
  - Classifier: LLM agent; each turn is exactly one tool call — an evidence tool or `submit_verdict`. Capped at 2 rounds; citations checked against real evidence.
  - Routing: validation failure, low confidence (< 0.6), or cap → `human_review`; confident verdict → END (responder placeholder).
- **Verified:** 34 offline checks pass (fake LLM, real CSVs); live STM32 value-drift run → `data_integrity` @ 0.97, chose `get_baseline_detail(val_mean)` first, 0 hallucinated citations.
- **Next, in order:** `responder.py` → `gate.py` (+ gate-reject → `human_review`) → `replay.py` (Part A) → `adversarial.py` (Parts B/C).
- **Loose end:** `ATTACK_TYPES` includes `botnet_dos` / port scan, which have no rows in `labels.csv`.
- **Paper notes:** initial bundle is deliberately single-source; the audit trail is post-hoc analysis, not enforcement — the gate is the only enforcement point.

## Why paused
Waiting on more course material before building the responder and gate.
