# Notes

## TL;DR — where it's at
- **Pipeline complete (2026-11-16):** investigator ⇄ classifier → responder → gate → execute (dry-run) | human_review, wired in `graph.py`.
  - Investigator: deterministic, single-source initial bundle (ESP32 → features, STM32 → MQTT), fixed read-only tool menu.
  - Classifier (LLM): one tool call per turn — evidence tool or `submit_verdict`; cap 2 rounds; citations checked.
  - Responder (LLM): forced `submit_action` → allow / restrict / isolate / reauth + `justification_ids`; device injected from the alert; never sees `verdict.rationale`.
  - Gate (no LLM): 7 rules (verdict_ok, verdict_confident, action_ok, enrolled, target_matches_alert, proportionate, justification_grounded), severity-to-action matrix, re-checks grounding against evidence itself. Holds even if both LLMs are hijacked.
  - Execute: dry-run logs the exact `nft` command; `GATE_MODE=enforce` is a stub until Part D (Pi table/chain names; STM32 is serial-bridged, no MAC).
  - Thresholds: `CONFIDENCE_THRESHOLD = 0.6` (escalate below), `HIGH_CONFIDENCE = 0.85` (tuned; isolate/reauth allowed at/above). The gate is the safety floor regardless of the tuned value.
- **Verified:** `python tests/checks.py` — 56 offline checks (fake LLM, real CSVs). Live STM32 value-drift run: `get_baseline_detail(val_mean)` → `data_integrity` @ 0.97 → isolate → gate approve → `nft add element inet zt_gateway isolated { stm32-01 }` (dry-run).
- **Part A done** (`replay.py`, results below). **Next:** `adversarial.py` (Parts B/C) → real enforcement on the Pi (Part D).
- **Loose end:** `ATTACK_TYPES` includes `botnet_dos` / port scan, which have no rows in `labels.csv`.

## Paper notes
- The initial bundle is deliberately single-source; the audit trail is post-hoc analysis, not enforcement — the gate is the only enforcement point.
- **Containment-by-design example (Part B, first live run 2026-11-16):** the responder's free-text `reason` claimed "message count/rate anomalies further corroborate" the attack, but the classifier had found traffic metrics normal. The responder sees evidence ids, not values, so it made up what the ids it cited meant. The gate structurally never reads `reason`, so the hallucination had zero effect on the decision. We cannot prevent the LLM hallucinating; the architecture keeps the hallucination from reaching enforcement. Marked in `responder.py` (`ActionModel.reason`).
- **Injection signals in the audit:** both LLM nodes log ignored extra keys as `extra_keys` (e.g. a supplied `target_device`). A fabricated justification id is traceable end to end: responder `justification_unknown` → gate `rejected_justification_ids` / `justification_grounded` → human_review `violated_rules`.

## Part A results (`results/part_a.{txt,json}`, `python replay.py --full`, 2026-11-16)
- **Headline:** value_drift detected 6/6 windows, 100% [61.0, 100.0], all typed `data_integrity` (both intensities, 3/3 each) vs behavioral IF 0% [0.0, 10.7] (0/32).
- 60 alerts (36 attack windows = 12 attacks x early/mid/late, 24 normal), 84 graph runs, 253 calls, 0 errors, 0 escalations.
- Recall 32/36 = 89% [74.7, 95.6] | precision 0.842 | accuracy 0.833 | FP on normals 6/24 = 25% [12.0, 44.9] (network IF session FP 49.7%) | type correct among detected 21/32 = 66%.
- Groundedness 1.000 over 84 verdicts, 0 hallucinated citations, 0 fabricated justifications, 0 extra keys.
- Gate approved 36/36 attack actions; 0 rejections (no rule was ever violated by honest runs -- Part B tests the rejections).
- Consistency (6 windows x 5): mean agreement 0.933, 1/6 is_attack flips (stm32_beacon, agreement 0.6), 0/6 type flips.
- Misses: mqtt_brute 3/6 detected (called benign -> allow -> approved), stm32_beacon 2/3 (one window called benign). Network IF on mqtt_brute: 50% -- same weak spot.

## Part A limitations / findings (honest results -- kept, not fixed away)
- **The gate enforces proportionality-to-verdict, not verdict-correctness.** 5/12 NORMAL ESP32 windows were classified `dos` and isolated, and the gate approved every one (also seen in the small run: `esp32-01@1792882852.45`, isolated in 5/5 runs). Every rule passed: the device is enrolled, isolate is proportionate to a high-confidence dos verdict, and the justification cites real evidence. A confidently wrong classifier passes the gate. Same mechanism in the other direction: 3 mqtt_brute windows called benign -> allow -> approved (missed attacks). Future work: independent cross-checks of the verdict (detector score agreement, second-source corroboration) before high-impact actions.
- **Detection works; fine-grained typing does not.** Every ESP32 network attack was typed `dos` -- including `mqtt_brute` and `host_sweep` -- and the STM32 `stm32_beacon` too. Only value_drift got its own type. Type-correct 21/32. Type matters because it selects isolate vs reauth: brute-force attacks got `isolate`, never `reauth`.
- **Consistency is not correctness.** Repeated verdicts were near-identical (mean agreement 0.933, confidence variance ~0 on 5/6 windows) -- including repeating the same wrong answers (beacon -> dos, the normal -> dos FP in the small run, 5/5). High run-to-run agreement shows the model is stable on this evidence, not that it is right. The one instability is the most worrying kind: stm32_beacon flipped is_attack (attack/benign) across runs at mean confidence 0.67 -- a true/benign flip, not a type flip -- which is exactly the low-margin case where a single run can't be trusted.
- **Fixed (a bug, not a finding):** the first small run escalated 19 responder actions as `validation_failed` because the free-text `reason` exceeded 500 chars (the API does not enforce `maxLength`). That broke the invariant that `validation_failed` means corrupted/injected structure, never verbosity. `reason` and `rationale` are now truncated (logged as `reason_truncated` / `rationale_truncated`), and both prompts ask for <= 3 sentences.
