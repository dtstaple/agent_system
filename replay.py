# feeds your labeled windows through the graph. Runs Part A experiments: correctness vs labels.csv, consistency across runs, groundedness of citations

"""
replay.py - Part A experiment harness (agent reliability vs ground truth).

Builds Alerts for labeled attack windows and for NORMAL windows (negatives),
runs each through the full graph (build_graph()), and scores the final states:
  - correctness      : verdict vs label -- detection rate per attack, accuracy,
                       precision/recall, FP rate on normals; value_drift headline
  - consistency      : a subset (one window per attack type + one normal) run N
                       times -- agreement with the modal answer, is_attack flips
                       vs attack_type flips, confidence variance
  - groundedness     : fraction of cited_evidence_ids that were real evidence
  - proportionality  : gate approve/reject on true attacks
and contrasts agent detection with the ML detector baselines (FINDINGS.txt).

Ground truth is never passed into the pipeline: the graph only ever receives
{"alert": Alert}. Labels live in a dict keyed by alert_id that only the scoring
code reads, joined AFTER the run.

Cost-aware: every LLM call is counted (with tokens) and printed per alert; a
hard --max-calls budget is never exceeded (an alert only starts if its worst
case fits).

Run:  python replay.py            small run  -> results/scratch_part_a_small.{txt,json}
      python replay.py --full     paper run  -> results/part_a.{txt,json}
      python replay.py --rescore results/part_a.json   rebuild the report, no API calls
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from collections import Counter

from dotenv import load_dotenv
from tabulate import tabulate

import classifier
import responder
from graph import build_graph
from investigator import DATA_DIR, DEVICE_REGISTRY
from state import Alert, MAX_INVESTIGATION_ROUNDS

load_dotenv()

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
NORMAL_MARGIN_S = 60   # a normal window must be at least this far from every labeled range
WINDOW_S = 10
# worst case per alert: one classifier call per tool round + the verdict call + the responder
MAX_CALLS_PER_ALERT = MAX_INVESTIGATION_ROUNDS + 2

# ML detector baselines, copied from ../ztgw-ml/FINDINGS.txt (rigorous re-evaluation:
# chronological split, threshold tuned on validation, 95% CIs). Rates are over ALL
# attack windows; the agent's are over sampled windows -- directional, not paired.
NETWORK_IF = {"host_sweep": 100.0, "syn_flood": 100.0, "udp_flood": 100.0,
              "icmp_flood": 76.9, "mqtt_brute": 50.0}
NETWORK_IF_OVERALL = (84.3, 78.1, 89.3)
NETWORK_IF_SESSION_FP = 49.71
BEHAVIORAL_IF = {"stm32_beacon": (100.0, 83.2, 100.0), "stm32_value_drift": (0.0, 0.0, 10.7)}
BEHAVIORAL_IF_OVERALL = (37.3, 25.3, 51.0)

SMALL = {"windows_per_attack": 1, "normals": 6, "max_calls": 180}   # ~3.6 calls/run observed
FULL = {"windows_per_attack": 3, "normals": 24, "max_calls": 400}


# ---------- alert construction (no LLM, no ground truth leaves this section) ----------
def _read(fname: str) -> list[dict]:
    with open(os.path.join(DATA_DIR, fname), newline="") as f:
        return list(csv.DictReader(f))


def load_labels() -> list[dict]:
    return _read("labels.csv")


def primary_rows(device: str) -> list[dict]:
    """The device's primary-source rows in the attack session, sorted by ts
    (ESP32 -> network features by MAC, STM32 -> MQTT by device), as the investigator uses."""
    if device == "stm32-01":
        rows = [r for r in _read("mqtt_behavior.csv") if r["device"] == device]
    else:
        mac = DEVICE_REGISTRY[device]["mac"]
        rows = [r for r in _read("features.csv") if r["mac"] == mac]
    return sorted(rows, key=lambda r: float(r["ts"]))


def make_alert(device: str, ts: str) -> Alert:
    # window_start is the row's own ts, so the investigator's nearest-row match always hits
    return Alert(alert_id=f"{device}@{ts}", device=device, window_start=ts,
                 window_end=str(round(float(ts) + WINDOW_S, 2)))


def pick_windows(rows: list[dict], k: int) -> list[dict]:
    """k=1 -> middle window; k=3 -> early/mid/late (quartiles). Deduplicated, in order."""
    if not rows:
        return []
    idx = [len(rows) // 2] if k == 1 else [len(rows) // 4, len(rows) // 2, (3 * len(rows)) // 4]
    return [rows[i] for i in sorted(set(idx))]


def attack_cases(labels: list[dict], k: int) -> list[tuple[Alert, dict]]:
    cases = []
    for lab in labels:
        s, e = float(lab["start"]), float(lab["end"])
        inside = [r for r in primary_rows(lab["device"]) if s <= float(r["ts"]) <= e]
        for r in pick_windows(inside, k):
            cases.append((make_alert(lab["device"], r["ts"]), lab))
    return cases


def is_normal_ts(ts: float, labels: list[dict]) -> bool:
    return all(ts < float(l["start"]) - NORMAL_MARGIN_S or ts > float(l["end"]) + NORMAL_MARGIN_S
               for l in labels)


def normal_cases(labels: list[dict], n: int, seed: int) -> list[tuple[Alert, dict]]:
    """n normal windows from the ATTACK session (same session the detectors' session-FP
    is measured on), split across both devices, sampled with a fixed seed."""
    rng = random.Random(seed)
    devices = ["esp32-01", "stm32-01"]
    cases = []
    for i, dev in enumerate(devices):
        pool = [r for r in primary_rows(dev) if is_normal_ts(float(r["ts"]), labels)]
        want = n // len(devices) + (1 if i < n % len(devices) else 0)
        for r in sorted(rng.sample(pool, min(want, len(pool))), key=lambda r: float(r["ts"])):
            cases.append((make_alert(dev, r["ts"]), {"label": "normal", "device": dev,
                                                     "category": "benign", "intensity": ""}))
    return cases


def consistency_subset(attacks: list[tuple[Alert, dict]], normals: list[tuple[Alert, dict]]) -> list[str]:
    """One window per attack category (the first label of that category; its middle
    window in --full) + one normal. Returns alert_ids."""
    chosen, seen = [], set()
    by_label: dict[int, list[str]] = {}
    for alert, lab in attacks:
        by_label.setdefault(id(lab), []).append(alert["alert_id"])
    for alert, lab in attacks:
        if lab["category"] not in seen:
            seen.add(lab["category"])
            ids = by_label[id(lab)]
            chosen.append(ids[len(ids) // 2])
    if normals:
        chosen.append(normals[0][0]["alert_id"])
    return chosen


# ---------- cost tracking: wrap the two LLM call sites at runtime ----------
class Meter:
    def __init__(self):
        self.calls = 0
        self.in_tok = 0
        self.out_tok = 0

    def wrap(self, module) -> None:
        inner = module._call_llm
        def counted(*a, **k):
            msg = inner(*a, **k)
            self.calls += 1
            usage = getattr(msg, "usage", None)
            self.in_tok += getattr(usage, "input_tokens", 0) or 0
            self.out_tok += getattr(usage, "output_tokens", 0) or 0
            return msg
        module._call_llm = counted

    def line(self) -> str:
        return f"calls {self.calls} | tok in {self.in_tok/1000:.1f}k out {self.out_tok/1000:.1f}k"


# ---------- running ----------
def run_alert(graph, alert: Alert) -> dict:
    """The pipeline sees ONLY the alert. Returns the parts of the final state we score."""
    try:
        final = graph.invoke({"alert": alert})
    except Exception as e:   # API/network error: record it, keep going
        return {"error": f"{type(e).__name__}: {e}"}
    keep = ("verdict", "action", "gate_decision", "escalation", "tool_history", "audit")
    return {k: final.get(k) for k in keep}


def run_experiment(attacks, normals, consistency_ids, n_runs, max_calls, meter, graph=None,
                   log=print) -> tuple[list[dict], bool]:
    """Correctness pass (run 1 for every alert), then runs 2..N for the consistency subset.
    Returns (records, budget_stopped)."""
    graph = graph or build_graph()
    truth = {a["alert_id"]: lab for a, lab in attacks + normals}   # joined after each run
    alerts = {a["alert_id"]: a for a, _ in attacks + normals}
    plan = [(aid, 1) for aid in alerts] + [(aid, r) for r in range(2, n_runs + 1)
                                           for aid in consistency_ids]
    records = []
    for i, (aid, run) in enumerate(plan, 1):
        if meter.calls + MAX_CALLS_PER_ALERT > max_calls:
            log(f"budget: stopping before {aid} run {run} ({meter.calls}/{max_calls} calls used)")
            return records, True
        out = run_alert(graph, alerts[aid])
        lab = truth[aid]
        rec = {"alert": alerts[aid], "run": run, "consistency": aid in consistency_ids,
               "truth": {"label": lab["label"], "category": lab["category"],
                         "intensity": lab["intensity"], "is_attack": lab["label"] != "normal"},
               **out}
        records.append(rec)
        log(f"[{i:>3}/{len(plan)}] {aid} run {run} ({lab['label']}) -> {_short(rec)} | {meter.line()}")
    return records, False


def _short(rec: dict) -> str:
    if rec.get("error"):
        return "ERROR"
    v, a, g = rec.get("verdict") or {}, rec.get("action") or {}, rec.get("gate_decision") or {}
    if v.get("status") != "ok":
        s = "validation_failed"
    else:
        s = f"{v['attack_type']} {v['confidence']:.2f}"
    if a:
        s += f" {a.get('action')}/{g.get('outcome', '-')}"
    if rec.get("escalation"):
        s += f" [esc: {rec['escalation']['reason']}]"
    return s


# ---------- scoring (pure functions over records; --rescore uses these alone) ----------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Rate and Wilson 95% interval, in percent."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h))


def fmt_rate(k: int, n: int) -> str:
    if n == 0:
        return "-"
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} {p:.0f}% [{lo:.1f}, {hi:.1f}]"


def predicted(rec: dict) -> tuple[bool | None, str | None]:
    """(is_attack, attack_type); (None, None) when there is no usable verdict --
    validation_failed or an error is never read as benign."""
    v = rec.get("verdict")
    if rec.get("error") or not v or v.get("status") != "ok":
        return None, None
    return v["is_attack"], v["attack_type"]


def correctness(records: list[dict]) -> dict:
    prim = [r for r in records if r["run"] == 1]
    tp = fp = tn = fn = 0
    for r in prim:
        pa, _ = predicted(r)
        if r["truth"]["is_attack"]:
            tp += pa is True
            fn += pa is not True          # None (failed) counts as a miss
        else:
            fp += pa is True
            tn += pa is False             # None is neither a correct benign nor an FP
    attacks = [r for r in prim if r["truth"]["is_attack"]]
    detected = [r for r in attacks if predicted(r)[0] is True]
    type_ok = sum(predicted(r)[1] == r["truth"]["category"] for r in detected)
    return {"n": len(prim), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / len(prim) if prim else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "n_attacks": len(attacks), "n_normals": len(prim) - len(attacks),
            "type_correct": type_ok, "n_detected": len(detected),
            "failed": sum(predicted(r)[0] is None for r in prim)}


def per_label_rows(records: list[dict]) -> list[list]:
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        if r["run"] == 1:
            t = r["truth"]
            groups.setdefault((t["label"], t["intensity"], r["alert"]["device"]), []).append(r)
    rows = []
    for (label, intensity, device), rs in groups.items():
        is_att = rs[0]["truth"]["is_attack"]
        k = sum(predicted(r)[0] is (True if is_att else False) for r in rs)
        types = Counter(str(predicted(r)[1]) for r in rs)
        actions = Counter(f"{(r.get('action') or {}).get('action')}/"
                          f"{(r.get('gate_decision') or {}).get('outcome', '-')}" for r in rs)
        esc = Counter(r["escalation"]["reason"] for r in rs if r.get("escalation"))
        if device == "stm32-01":
            net = "n/a (invisible)"
        else:
            net = f"{NETWORK_IF[label]:.1f}%" if label in NETWORK_IF else "-"
        beh = (f"{BEHAVIORAL_IF[label][0]:.1f}% [{BEHAVIORAL_IF[label][1]}, {BEHAVIORAL_IF[label][2]}]"
               if label in BEHAVIORAL_IF else "-")
        rows.append([label + (f" ({intensity})" if intensity else ""), device,
                     ("detected " if is_att else "benign ") + fmt_rate(k, len(rs)),
                     _counts(types), _counts(actions), _counts(esc) or "-", net, beh])
    return rows


def _counts(c: Counter) -> str:
    return ", ".join(f"{k}×{v}" if v > 1 else k for k, v in c.most_common())


def value_drift(records: list[dict]) -> dict:
    vd = [r for r in records if r["run"] == 1 and r["truth"]["label"] == "stm32_value_drift"]
    out = {"all": (sum(predicted(r)[0] is True for r in vd), len(vd)),
           "type_ok": sum(predicted(r)[1] == "data_integrity" for r in vd)}
    for inten in sorted({r["truth"]["intensity"] for r in vd}):
        sub = [r for r in vd if r["truth"]["intensity"] == inten]
        out[inten] = (sum(predicted(r)[0] is True for r in sub), len(sub))
    return out


def consistency(records: list[dict]) -> list[dict]:
    by_id: dict[str, list[dict]] = {}
    for r in records:
        if r["consistency"]:
            by_id.setdefault(r["alert"]["alert_id"], []).append(r)
    out = []
    for aid, rs in by_id.items():
        answers = [predicted(r) for r in rs]
        modal, modal_n = Counter(answers).most_common(1)[0]
        attack_vals = {a[0] for a in answers}
        types_when_attack = {a[1] for a in answers if a[0] is True}
        confs = [r["verdict"]["confidence"] for r in rs if predicted(r)[0] is not None]
        acts = Counter((r.get("action") or {}).get("action") for r in rs)
        gates = Counter((r.get("gate_decision") or {}).get("outcome") for r in rs)
        out.append({
            "alert_id": aid, "label": rs[0]["truth"]["label"], "runs": len(rs),
            "modal": modal, "agreement": modal_n / len(rs),
            # a true/benign flip is more serious than a type flip, so they are separate
            "is_attack_flip": len(attack_vals - {None}) > 1,
            "failed_runs": sum(a[0] is None for a in answers),
            "type_flip": len(types_when_attack) > 1,
            "conf_mean": statistics.fmean(confs) if confs else float("nan"),
            "conf_var": statistics.pvariance(confs) if len(confs) > 1 else 0.0,
            "action_agreement": acts.most_common(1)[0][1] / len(rs),
            "gate_agreement": gates.most_common(1)[0][1] / len(rs),
        })
    return out


def groundedness(records: list[dict]) -> dict:
    ratios, hallucinated, unknown_just, gate_rejected, extra = [], [], [], [], []
    truncated = Counter()
    for r in records:
        aid = f'{r["alert"]["alert_id"]}#{r["run"]}'
        for a in r.get("audit") or []:
            if a["node"] == "classifier" and a["status"] == "ok":
                valid, bad = a["cited_valid"], a["cited_hallucinated"]
                if valid or bad:
                    ratios.append(len(valid) / (len(valid) + len(bad)))
                hallucinated += [(aid, x) for x in bad]
            truncated["rationale"] += bool(a.get("rationale_truncated"))
            truncated["reason"] += bool(a.get("reason_truncated"))
            if a["node"] in ("classifier", "responder") and a.get("extra_keys"):
                extra += [(aid, a["node"], x) for x in a["extra_keys"]]
            if a["node"] == "responder" and a.get("justification_unknown"):
                unknown_just += [(aid, x) for x in a["justification_unknown"]]
            if a["node"] == "gate" and a.get("rejected_justification_ids"):
                gate_rejected += [(aid, x) for x in a["rejected_justification_ids"]]
    return {"verdicts": len(ratios),
            "mean": statistics.fmean(ratios) if ratios else float("nan"),
            "min": min(ratios) if ratios else float("nan"),
            "hallucinated": hallucinated, "justification_unknown": unknown_just,
            "gate_rejected_ids": gate_rejected, "extra_keys": extra,
            "rationale_truncated": truncated["rationale"], "reason_truncated": truncated["reason"]}


def proportionality(records: list[dict]) -> dict:
    prim = [r for r in records if r["run"] == 1]
    att = [r for r in prim if r["truth"]["is_attack"]]
    gated = [r for r in att if r.get("gate_decision")]
    norm = [r for r in prim if not r["truth"]["is_attack"]]
    return {"attacks": len(att), "reached_gate": len(gated),
            "approved": sum(r["gate_decision"]["outcome"] == "approve" for r in gated),
            "rejected": sum(r["gate_decision"]["outcome"] == "reject" for r in gated),
            "actions": Counter(r["action"]["action"] for r in gated),
            "violated": Counter(v for r in gated for v in r["gate_decision"]["violated_rules"]),
            "escalated_before_gate": Counter(r["escalation"]["reason"] for r in att
                                             if r.get("escalation") and not r.get("gate_decision")),
            "normals": len(norm),
            "normal_allow_approved": sum((r.get("action") or {}).get("action") == "allow"
                                         and r["gate_decision"]["outcome"] == "approve"
                                         for r in norm if r.get("gate_decision"))}


# ---------- report ----------
def report(records: list[dict], meta: dict) -> str:
    out: list[str] = []
    p = out.append
    vd = value_drift(records)
    k, n = vd["all"]
    bp, blo, bhi = BEHAVIORAL_IF["stm32_value_drift"]
    rate = wilson(k, n)
    p("=" * 78)
    p(f"HEADLINE  value_drift: agent {rate[0]:.0f}% [{rate[1]:.1f}, {rate[2]:.1f}] ({k}/{n} windows)"
      f"  vs  behavioral detector {bp:.0f}% [{blo}, {bhi}] (0/32)")
    p("          " + "  |  ".join(f"{i}: {fmt_rate(*vd[i])}" for i in vd if i not in ("all", "type_ok"))
      + f"  |  typed data_integrity: {vd['type_ok']}/{n}")
    p("=" * 78)
    p(f"mode {meta['mode']} | N={meta['n_runs']} | seed {meta['seed']} | model {meta['model']} | "
      f"calls {meta['calls']} | tokens in {meta['in_tok']} out {meta['out_tok']}"
      + (" | STOPPED AT BUDGET (partial)" if meta.get("budget_stopped") else ""))
    errors = sum(1 for r in records if r.get("error"))
    if errors:
        p(f"!! {errors} run(s) errored (API) -- counted as misses")

    c = correctness(records)
    p("\n## Correctness (run 1 of every alert)")
    p(tabulate([["alerts", c["n"], f"{c['n_attacks']} attack / {c['n_normals']} normal"],
                ["accuracy", f"{c['accuracy']:.3f}", "failed verdicts count as wrong"],
                ["precision", f"{c['precision']:.3f}", f"TP {c['tp']} FP {c['fp']}"],
                ["recall (detection)", f"{c['recall']:.3f}", fmt_rate(c["tp"], c["n_attacks"])],
                ["FP rate on normals", fmt_rate(c["fp"], c["n_normals"]),
                 f"network IF session FP {NETWORK_IF_SESSION_FP}%"],
                ["attack type correct", fmt_rate(c["type_correct"], c["n_detected"]), "among detected"],
                ["no usable verdict", c["failed"], "validation_failed / error"]],
               headers=["metric", "value", "note"]))

    p("\n## Agent vs ML detectors, per label")
    p(tabulate(per_label_rows(records),
               headers=["label", "device", "agent", "agent type", "action/gate", "escalated",
                        "network IF", "behavioral IF"]))
    p(f"detector overall: network IF {NETWORK_IF_OVERALL[0]}% [{NETWORK_IF_OVERALL[1]}, "
      f"{NETWORK_IF_OVERALL[2]}] | behavioral IF {BEHAVIORAL_IF_OVERALL[0]}% "
      f"[{BEHAVIORAL_IF_OVERALL[1]}, {BEHAVIORAL_IF_OVERALL[2]}]")
    p("note: detector rates are over ALL attack windows (FINDINGS.txt); agent rates are over "
      "sampled windows,\n      so intervals are wide and the comparison is directional, not paired.")

    cons = consistency(records)
    p(f"\n## Consistency (subset re-run, agreement vs modal answer)")
    if cons:
        p(tabulate([[x["label"], x["alert_id"], x["runs"],
                     f"{x['modal'][0]}/{x['modal'][1]}", f"{x['agreement']:.2f}",
                     "YES" if x["is_attack_flip"] else "no", "YES" if x["type_flip"] else "no",
                     x["failed_runs"], f"{x['conf_mean']:.3f}", f"{x['conf_var']:.5f}",
                     f"{x['action_agreement']:.2f}", f"{x['gate_agreement']:.2f}"] for x in cons],
                   headers=["label", "alert", "runs", "modal (attack/type)", "agree",
                            "is_attack flip", "type flip", "failed", "conf mean", "conf var",
                            "action agree", "gate agree"]))
        p(f"mean agreement {statistics.fmean(x['agreement'] for x in cons):.3f} | "
          f"is_attack flips {sum(x['is_attack_flip'] for x in cons)}/{len(cons)} | "
          f"type flips {sum(x['type_flip'] for x in cons)}/{len(cons)}")
    else:
        p("(skipped)")

    g = groundedness(records)
    p("\n## Groundedness (all runs)")
    p(f"verdicts with citations {g['verdicts']} | mean valid fraction {g['mean']:.3f} | "
      f"min {g['min']:.3f} | hallucinated ids {len(g['hallucinated'])}")
    for aid, x in g["hallucinated"]:
        p(f"  HALLUCINATED {aid}: {x}")
    p(f"responder justification_unknown {len(g['justification_unknown'])} | "
      f"gate rejected_justification_ids {len(g['gate_rejected_ids'])} | "
      f"extra_keys {len(g['extra_keys'])}")
    p(f"truncated (audit-only free text, not escalated): rationale {g['rationale_truncated']} | "
      f"reason {g['reason_truncated']}")
    for item in g["justification_unknown"] + g["extra_keys"]:
        p(f"  FLAG {item}")

    pr = proportionality(records)
    p("\n## Action proportionality (run 1, true attacks)")
    p(f"attacks {pr['attacks']} | reached gate {pr['reached_gate']} | "
      f"approved {fmt_rate(pr['approved'], pr['reached_gate'])} | rejected {pr['rejected']}")
    p(f"actions: {_counts(pr['actions']) or '-'} | violated rules: {_counts(pr['violated']) or '-'} | "
      f"escalated before gate: {_counts(pr['escalated_before_gate']) or '-'}")
    p(f"normals: allow+approved {pr['normal_allow_approved']}/{pr['normals']}")
    return "\n".join(out)


def write_outputs(prefix: str, records: list[dict], meta: dict, text: str) -> None:
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    with open(prefix + ".json", "w") as f:
        json.dump({"meta": meta, "records": records}, f, indent=1, default=str)
    with open(prefix + ".txt", "w") as f:
        f.write(text + "\n")


# ---------- main ----------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Part A replay experiments")
    ap.add_argument("--full", action="store_true", help="3 windows/attack + 24 normals (~300 calls)")
    ap.add_argument("--n", type=int, default=5, help="consistency runs per subset window")
    ap.add_argument("--max-calls", type=int, help="hard API-call budget (default 180 small / 400 full)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-consistency", action="store_true")
    ap.add_argument("--out", help="output prefix (default results/part_a or results/scratch_part_a_small)")
    ap.add_argument("--rescore", metavar="JSON", help="rebuild the report from saved records; no API calls")
    args = ap.parse_args(argv)

    if args.rescore:
        with open(args.rescore) as f:
            saved = json.load(f)
        text = report(saved["records"], saved["meta"])
        print(text)
        with open(os.path.splitext(args.rescore)[0] + ".txt", "w") as f:
            f.write(text + "\n")
        return

    cfg = FULL if args.full else SMALL
    max_calls = args.max_calls or cfg["max_calls"]
    prefix = args.out or os.path.join(RESULTS_DIR, "part_a" if args.full else "scratch_part_a_small")
    n_runs = 1 if args.skip_consistency else args.n

    labels = load_labels()
    attacks = attack_cases(labels, cfg["windows_per_attack"])
    normals = normal_cases(labels, cfg["normals"], args.seed)
    cons_ids = [] if n_runs <= 1 else consistency_subset(attacks, normals)
    total = len(attacks) + len(normals) + len(cons_ids) * (n_runs - 1)
    print(f"{len(attacks)} attack + {len(normals)} normal alerts, consistency {len(cons_ids)}×{n_runs} "
          f"-> {total} graph runs, budget {max_calls} calls (≤{MAX_CALLS_PER_ALERT}/run)")

    meter = Meter()
    meter.wrap(classifier)
    meter.wrap(responder)
    t0 = time.time()
    records, stopped = run_experiment(attacks, normals, cons_ids, n_runs, max_calls, meter)
    meta = {"mode": "full" if args.full else "small", "n_runs": n_runs, "seed": args.seed,
            "model": classifier.MODEL, "calls": meter.calls, "in_tok": meter.in_tok,
            "out_tok": meter.out_tok, "budget_stopped": stopped, "max_calls": max_calls,
            "seconds": round(time.time() - t0, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    text = report(records, meta)
    print("\n" + text)
    write_outputs(prefix, records, meta, text)
    print(f"\nwrote {prefix}.txt and {prefix}.json")


if __name__ == "__main__":
    main()
