# Learning Progress

## LangGraph state
- Introduced (AI explained): nodes as functions, shared TypedDict state, partial-update returns, conditional edges read state; TypedDict does not enforce scoping.
- Learner reasoning: proposed scoping each node to its job to limit hallucination; initial alert = "a frame of network traffic," fields = packet count + device ID.
- Needs reinforcement: data is ~10s windows, not frames; distinction between limiting write scope vs. preventing hallucination (citation checks); state must carry downstream outputs, not just evidence.

- 2026-10-27: state.py, investigator.py, classifier.py were implemented outside this learning loop (commits 15427e2, f43bc60); authorship of the design reasoning not recorded. The AlertState checkpoint questions are answered by the committed code (10s-window alert_id, one key per node, ground truth joined in replay.py) — not counted as demonstrated learner reasoning.

## Agent vs. workflow (agentic claim)
- Introduced (AI explained, 2026-10-27 review): workflow vs. agent distinction (predefined code paths vs. LLM choosing its own observations/tools in a loop); current pipeline is a workflow with two single-shot LLM steps; low-confidence loop re-fetches identical evidence; investigation_round never written; fail-to-benign on parse failure is fail-open; tool use widens prompt-injection surface, which Part B needs.
- Couldn't verify the cited Jamshidi 2026 / Chou 2026 papers via web search; asked the learner for titles.

- Learner reasoning (2026-10-28): chose B. Fixed read-only menu get_more_history(n), get_other_source(), get_baseline_detail(column); investigator executes deterministically so groundedness holds; cap at MAX_INVESTIGATION_ROUNDS=2; device-scoped; every call logged to audit; gate validates final verdict+action, not tool calls, with audit exposing suspicious patterns. Correctly separated "LLM chooses" from "trusted code fetches".
- Gaps raised by AI: device scope must be enforced by executor binding (not an LLM arg); bounds on n and column whitelist; STM32 has no other source; new evidence ids must join the citation-valid set; audit pattern detection is post-hoc, not enforcement.

- Learner decisions (2026-10-28): loop lives in the GRAPH so every request/result is a visible state transition for adversarial.py; classifier may write a distinct request key (single writer per key); final state shows request/result history, verdict, audit. Cap reached with low confidence -> human review (same terminal as gate reject), reasoning: escalating under uncertainty is the safe behavior vs. forced guessing. Accepted all four AI additions (n<=10, column allowlist, explicit unavailable item, tool-result ids join valid_ids) and device injection by runner. Paper will state audit is post-hoc; gate is the only enforcement point.

- Learner decisions (2026-10-28, round 2): low confidence + no tool request -> human_review immediately (loop only on explicit request; avoids no-op loop). Unparseable/injected output -> validation_failed -> human_review, never benign; keep raw output in audit. human_review is the single escalation terminal (gate reject, low-confidence, validation failure).

- Learner decision (2026-10-28): narrow initial bundle to one primary source per device (ESP32 -> features, STM32 -> mqtt) so get_other_source() is a real agentic choice; reasoning: makes "does the classifier recognize it needs the second source?" a Part A result; paper will state the bundle is deliberately single-source. Accepted all AI proposed additions (threshold 0.6, one tool call/round, round-prefixed ids, executor re-validation, raw SDK, offline tests).
- Implemented 2026-10-28 (state.py, investigator.py, classifier.py, graph.py); 28 offline checks passed; one live run escalated as validation_failed because the model wrote prose before its JSON verdict.

- Learner decision (2026-10-28): verdict returned only via a submit_verdict tool (always offered, including at cap); free text never parsed. Reasoning: verbosity can't cause false escalation, so validation_failed means corrupted/injected structured output, keeping the Part B claim precise. Learner assumed "SDK guarantees valid JSON" -- AI clarified the input arrives as parsed JSON but the schema isn't enforced without strict tool use; Pydantic is the actual enforcement.
- Implemented 2026-10-28 with AI addition tool_choice={"type":"any"} (forces a tool call). 34 offline checks pass; live STM32 drift run: classifier requested get_baseline_detail(val_mean), then submitted data_integrity @0.97, 12 valid citations, 0 hallucinated, no escalation.
