# Learning Progress

## LangGraph state
- Introduced (AI explained): nodes as functions, shared TypedDict state, partial-update returns, conditional edges read state; TypedDict does not enforce scoping.
- Learner reasoning: proposed scoping each node to its job to limit hallucination; initial alert = "a frame of network traffic," fields = packet count + device ID.
- Needs reinforcement: data is ~10s windows, not frames; distinction between limiting write scope vs. preventing hallucination (citation checks); state must carry downstream outputs, not just evidence.

## Pending decision
Decision: AlertState shape (state.py) — Build checkpoint, stage: awaiting reasoning (round 2).
Asked: (1) alert unit — labeled attack vs single 10s window vs other, and how benign baseline windows enter replay; (2) per-node output fields; (3) where labels.csv ground truth lives so LLM nodes can't see it.
