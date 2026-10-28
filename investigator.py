#This has no LLM behavior, this file is given a window timestamp + device, pulls real evidence from the CSV
# also the feeature vector, devices normal baseline, its identity, recent history, returns a structured evidence bundle
# this is the groundness anchor, everything downstream only gets real dtaa to ensure no hallucination.

"""
investigator.py - the investigation tool-service (NO LLM, fully deterministic).

Two modes, one node:
  1. Initial bundle: given an Alert (device + window timestamp), retrieve real
     evidence from the device's PRIMARY source only (ESP32 -> network features,
     STM32 -> MQTT behavior). The bundle is deliberately single-source so the
     classifier has to decide whether it needs more.
  2. Tool execution: run the classifier's pending evidence_request from the fixed,
     read-only menu (get_more_history / get_other_source / get_baseline_detail).
     The LLM only chooses which tool; this code fetches the data. Tools take no
     device argument -- the device always comes from state["alert"].

Every value here is pulled from real data, never generated -- this is the
groundedness anchor: the classifier can only reason over, and cite, what this
node actually found.

Writes: "evidence", "tool_history", "investigation_round" (and appends "audit").

Data sources (see headers in data/):
  mqtt_behavior.csv  : keyed on `device`  -> behavioral features (both devices)
  features.csv       : keyed on `mac`     -> network features (ESP32 only)
  baseline_*.csv     : same schemas, clean-normal reference for deltas
"""

from __future__ import annotations
import csv
import os
import statistics
from typing import Any

from state import (Alert, AlertState, Evidence, EvidenceItem, EvidenceRequest,
                   ToolResult, HISTORY_MAX_N, MAX_INVESTIGATION_ROUNDS)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# device identity registry -- the gate checks `enrolled` directly.
# maps logical device name -> its network identity (MAC for the ESP32) and its sources.
DEVICE_REGISTRY = {
    "esp32-01": {"mac": "f0:16:1d:51:4d:34", "class": "esp32_sensor", "bridged": False,
                 "primary": "features", "secondary": "mqtt"},
    "stm32-01": {"mac": None, "class": "stm32_sensor", "bridged": True,   # serial-bridged
                 "primary": "mqtt", "secondary": None},                   # no network source
}

# which behavioral columns we surface as evidence, per source
MQTT_COLS = ["msg_count", "bytes_total", "msg_size_mean", "iat_mean",
             "rate_per_sec", "unique_topics", "val_mean", "val_std"]
FEAT_COLS = ["pkt_count", "bytes_total", "up_down_ratio", "pkt_size_mean",
             "iat_mean", "frac_tcp", "frac_udp", "frac_icmp", "frac_arp",
             "syn_count", "rst_count", "uniq_dst_ip", "uniq_dst_port", "new_dst_count"]

# per-source file layout: which file, which baseline, which column identifies the device
SOURCES = {
    "mqtt":     {"file": "mqtt_behavior.csv", "baseline": "baseline_mqtt.csv",
                 "match_col": "device", "cols": MQTT_COLS, "hist_col": "rate_per_sec"},
    "features": {"file": "features.csv", "baseline": "baseline_features.csv",
                 "match_col": "mac", "cols": FEAT_COLS, "hist_col": "pkt_count"},
}

HISTORY_WINDOWS = 3   # how many prior windows the initial bundle includes as context
MATCH_TOLERANCE_S = 5  # a window row must be within this many seconds of the alert ts


# ---------- CSV loading (cached so replay doesn't re-read per alert) ----------
_cache: dict[str, list[dict]] = {}

def _load(fname: str) -> list[dict]:
    if fname not in _cache:
        path = os.path.join(DATA_DIR, fname)
        with open(path, newline="") as f:
            _cache[fname] = list(csv.DictReader(f))
    return _cache[fname]

def _fnum(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---------- row lookup ----------
def _match_val(device: str, src: str) -> str | None:
    """The value identifying this device in a source's key column (device name or MAC)."""
    if src == "mqtt":
        return device
    return DEVICE_REGISTRY.get(device, {}).get("mac")

def _device_rows(fname: str, src: str, device: str) -> list[dict]:
    mv = _match_val(device, src)
    if mv is None:
        return []
    col = SOURCES[src]["match_col"]
    return [r for r in _load(fname) if r.get(col) == mv]

def _current_row(device: str, ts: float, src: str) -> dict | None:
    """The source's row nearest ts, if within MATCH_TOLERANCE_S."""
    rows = _device_rows(SOURCES[src]["file"], src, device)
    row = min(rows, key=lambda r: abs(_fnum(r["ts"]) - ts), default=None)
    if row and abs(_fnum(row["ts"]) - ts) <= MATCH_TOLERANCE_S:
        return row
    return None

def _prior_rows(device: str, ts: float, src: str, n: int) -> list[dict]:
    rows = sorted(_device_rows(SOURCES[src]["file"], src, device), key=lambda r: _fnum(r["ts"]))
    return [r for r in rows if _fnum(r["ts"]) < ts][-n:]

def _baseline_values(device: str, src: str, col: str) -> list[float]:
    rows = _device_rows(SOURCES[src]["baseline"], src, device)
    return [_fnum(r[col]) for r in rows if col in r]

def _baseline_stats(device: str, src: str) -> dict[str, float]:
    """Mean of each column over this device's clean-normal baseline."""
    out = {}
    for c in SOURCES[src]["cols"]:
        vals = _baseline_values(device, src, c)
        out[c] = statistics.fmean(vals) if vals else 0.0
    return out


def _source_items(device: str, ts: float, src: str, prefix: str = "") -> list[EvidenceItem]:
    """Current window values + baseline means for one source. prefix namespaces tool results."""
    row = _current_row(device, ts, src)
    if not row:
        return []
    base = _baseline_stats(device, src)
    items = []
    for c in SOURCES[src]["cols"]:
        items.append(EvidenceItem(id=f"{prefix}{src}.{c}", source=src, key=c,
                                  value=round(_fnum(row[c]), 3)))
        items.append(EvidenceItem(id=f"{prefix}baseline.{src}.{c}", source="baseline",
                                  key=f"normal_{c}", value=round(base[c], 3)))
    return items

def _sources_in_evidence(ev: Evidence) -> set[str]:
    return {it["source"] for it in ev["items"] if it["source"] in SOURCES}

def _status_item(prefix: str, status: str, key: str, reason: str) -> EvidenceItem:
    """An explicit, citable record that a tool returned nothing usable."""
    return EvidenceItem(id=f"{prefix}{status}.{key}", source="tool", key=key, value=reason)


# ---------- the fixed tool menu (device is injected, never an argument) ----------
def get_more_history(device: str, ts: float, ev: Evidence, prefix: str,
                     n: Any) -> tuple[str, list[EvidenceItem]]:
    """The n windows before this one, for each source already in the evidence."""
    if type(n) is not int or not 1 <= n <= HISTORY_MAX_N:
        return "refused", [_status_item(prefix, "refused", "history",
                                        f"n must be an integer 1..{HISTORY_MAX_N}")]
    items = []
    for src in sorted(_sources_in_evidence(ev)):
        prior = _prior_rows(device, ts, src, n)
        for c in SOURCES[src]["cols"]:
            items.append(EvidenceItem(id=f"{prefix}history.{src}.{c}", source="history",
                                      key=f"prior_{n}_{c}",
                                      value=[round(_fnum(r[c]), 3) for r in prior]))
    if not items:
        return "unavailable", [_status_item(prefix, "unavailable", "history", "no prior windows")]
    return "ok", items

def get_other_source(device: str, ts: float, ev: Evidence, prefix: str) -> tuple[str, list[EvidenceItem]]:
    """The device's secondary source for this window."""
    secondary = DEVICE_REGISTRY.get(device, {}).get("secondary")
    if secondary is None:
        return "unavailable", [_status_item(prefix, "unavailable", "other_source",
                                            "device has no secondary source")]
    if secondary in _sources_in_evidence(ev):
        return "refused", [_status_item(prefix, "refused", "other_source",
                                        f"{secondary} already in evidence")]
    items = _source_items(device, ts, secondary, prefix)
    if not items:
        return "unavailable", [_status_item(prefix, "unavailable", f"{secondary}",
                                            f"no {secondary} row within {MATCH_TOLERANCE_S}s")]
    return "ok", items

def get_baseline_detail(device: str, ts: float, ev: Evidence, prefix: str,
                        column: Any) -> tuple[str, list[EvidenceItem]]:
    """Mean / std / p5 / p95 of one column over this device's baseline."""
    if not isinstance(column, str) or column not in set(MQTT_COLS) | set(FEAT_COLS):
        return "refused", [_status_item(prefix, "refused", "baseline_detail",
                                        "column not in allowlist")]
    srcs = [s for s in sorted(_sources_in_evidence(ev)) if column in SOURCES[s]["cols"]]
    if not srcs:
        return "unavailable", [_status_item(prefix, "unavailable", f"baseline_detail.{column}",
                                            "no source in evidence has this column")]
    items = []
    for src in srcs:
        vals = _baseline_values(device, src, column)
        if len(vals) < 2:
            continue
        q = statistics.quantiles(vals, n=20)   # 5% steps: q[0]=p5, q[-1]=p95
        stats = {"mean": statistics.fmean(vals), "std": statistics.stdev(vals),
                 "p5": q[0], "p95": q[-1]}
        for name, v in stats.items():
            items.append(EvidenceItem(id=f"{prefix}baseline_detail.{src}.{column}.{name}",
                                      source="baseline", key=f"normal_{column}_{name}",
                                      value=round(v, 3)))
    if not items:
        return "unavailable", [_status_item(prefix, "unavailable", f"baseline_detail.{column}",
                                            "not enough baseline rows")]
    return "ok", items


# allowed args per tool -- anything else is refused, even if the classifier's schema passed it
TOOL_ARGS = {"get_more_history": {"n"}, "get_other_source": set(), "get_baseline_detail": {"column"}}
TOOLS = {"get_more_history": get_more_history, "get_other_source": get_other_source,
         "get_baseline_detail": get_baseline_detail}


def _run_tool(state: AlertState) -> dict:
    alert: Alert = state["alert"]
    ev: Evidence = state["evidence"]
    req: EvidenceRequest | None = state.get("evidence_request")
    rnd = state.get("investigation_round", 0) + 1
    prefix = f"tool.r{rnd}."
    device, ts = alert["device"], float(alert["window_start"])

    if req is None:
        raise ValueError("investigator called with evidence but no pending evidence_request")

    tool, args = req["tool"], req.get("args", {})
    # second validation on the trusted side: cap, known tool, exact arg names
    if rnd > MAX_INVESTIGATION_ROUNDS:
        status, new = "refused", [_status_item(prefix, "refused", "cap", "round cap reached")]
    elif tool not in TOOLS or set(args) != TOOL_ARGS[tool]:
        status, new = "refused", [_status_item(prefix, "refused", "request", "unknown tool or bad args")]
    else:
        status, new = TOOLS[tool](device, ts, ev, prefix, **args)

    ids = [it["id"] for it in new]
    result = ToolResult(request=req, status=status, item_ids=ids)
    audit = [{"node": "investigator", "mode": "tool", "alert_id": alert["alert_id"],
              "round": rnd, "tool": tool, "args": args, "status": status, "new_item_ids": ids}]
    return {"evidence": Evidence(items=ev["items"] + new, enrolled=ev["enrolled"]),
            "tool_history": [result], "investigation_round": rnd, "audit": audit}


def _initial_bundle(state: AlertState) -> dict:
    alert: Alert = state["alert"]
    device = alert["device"]
    ts = float(alert["window_start"])   # alert carries the unix ts of the window

    reg = DEVICE_REGISTRY.get(device, {})
    enrolled = device in DEVICE_REGISTRY
    primary = reg.get("primary", "mqtt")

    items: list[EvidenceItem] = []

    # identity evidence
    items.append(EvidenceItem(id="identity.device", source="identity",
                              key="device", value=device))
    items.append(EvidenceItem(id="identity.bridged", source="identity",
                              key="bridged", value=reg.get("bridged", False)))

    # primary source only: current window + baseline means
    src_items = _source_items(device, ts, primary)
    items.extend(src_items)

    # recent history (context: was it normal just before?)
    col = SOURCES[primary]["hist_col"]
    hist = [round(_fnum(r[col]), 3) for r in _prior_rows(device, ts, primary, HISTORY_WINDOWS)]
    items.append(EvidenceItem(id="history.recent", source="history",
                              key=f"prior_{HISTORY_WINDOWS}_{col}", value=hist))

    evidence = Evidence(items=items, enrolled=enrolled)

    audit = [{
        "node": "investigator",
        "mode": "initial",
        "alert_id": alert["alert_id"],
        "primary_source": primary,
        "evidence_count": len(items),
        "has_primary": bool(src_items),
    }]
    return {"evidence": evidence, "investigation_round": 0, "audit": audit}


# ---------- the node ----------
def investigate(state: AlertState) -> dict:
    """LangGraph node. First call builds the single-source bundle; later calls run the
    classifier's pending evidence_request."""
    if "evidence" not in state:
        return _initial_bundle(state)
    return _run_tool(state)


# ---------- standalone test ----------
if __name__ == "__main__":
    # value_drift attack window on the STM32 (from labels.csv: ~1792886067)
    test_alert = Alert(alert_id="stm32-01@1792886100",
                       device="stm32-01",
                       window_start="1792886100",
                       window_end="1792886110")
    out = investigate({"alert": test_alert})
    print(f"enrolled: {out['evidence']['enrolled']}")
    print(f"evidence items: {len(out['evidence']['items'])}\n")
    for it in out["evidence"]["items"]:
        print(f"  {it['id']:28s} = {it['value']}")
    print(f"\naudit: {out['audit']}")
