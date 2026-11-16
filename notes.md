# Notes

## TL;DR — where it's at
- **Pipeline complete (2026-11-16):** investigator ⇄ classifier → responder → gate → execute (dry-run) | human_review, wired in `graph.py`.
  - Investigator: deterministic, single-source initial bundle (ESP32 → features, STM32 → MQTT), fixed read-only tool menu.
  - Classifier (LLM): one tool call per turn — evidence tool or `submit_verdict`; cap 2 rounds; citations checked.
  - Responder (LLM): forced `submit_action` → allow / restrict / isolate / reauth + `justification_ids`; device injected from the alert; never sees `verdict.rationale`.
  - Gate (no LLM): 7 rules (verdict_ok, verdict_confident, action_ok, enrolled, target_matches_alert, proportionate, justification_grounded), severity-to-action matrix, re-checks grounding against evidence itself. Holds even if both LLMs are hijacked.
  - Execute: dry-run logs the exact `nft` command; `GATE_MODE=enforce` is a stub until Part D (Pi table/chain names; STM32 is serial-bridged, no MAC).
  - Thresholds: `CONFIDENCE_THRESHOLD = 0.6` (escalate below), `HIGH_CONFIDENCE = 0.85` (tuned; isolate/reauth allowed at/above). The gate is the safety floor regardless of the tuned value.
- **Verified:** `python tests/checks.py` — 35 offline checks (fake LLM, real CSVs). Live STM32 value-drift run: `get_baseline_detail(val_mean)` → `data_integrity` @ 0.97 → isolate → gate approve → `nft add element inet zt_gateway isolated { stm32-01 }` (dry-run).
- **Next, in order:** `replay.py` (Part A) → `adversarial.py` (Parts B/C) → real enforcement on the Pi (Part D).
- **Loose end:** `ATTACK_TYPES` includes `botnet_dos` / port scan, which have no rows in `labels.csv`.

## Paper notes
- The initial bundle is deliberately single-source; the audit trail is post-hoc analysis, not enforcement — the gate is the only enforcement point.
- **Containment-by-design example (Part B, first live run 2026-11-16):** the responder's free-text `reason` claimed "message count/rate anomalies further corroborate" the attack, but the classifier had found traffic metrics normal. The responder sees evidence ids, not values, so it made up what the ids it cited meant. The gate structurally never reads `reason`, so the hallucination had zero effect on the decision. We cannot prevent the LLM hallucinating; the architecture keeps the hallucination from reaching enforcement. Marked in `responder.py` (`ActionModel.reason`).
- **Injection signals in the audit:** both LLM nodes log ignored extra keys as `extra_keys` (e.g. a supplied `target_device`). A fabricated justification id is traceable end to end: responder `justification_unknown` → gate `rejected_justification_ids` / `justification_grounded` → human_review `violated_rules`.
