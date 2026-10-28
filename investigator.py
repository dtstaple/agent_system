#This has no LLM behavior, this file is given a window timestamp + device, pulls real evidence from the CSV
# also the feeature vector, devices normal baseline, its identity, recent history, returns a structured evidence bundle
# this is the groundness anchor, everything downstream only gets real dtaa to ensure no hallucination. 

"""
investigator.py - the investigation agent (NO LLM, fully deterministic).

Given an Alert (device + window timestamp), it retrieves real evidence from the
collected CSVs and assembles an Evidence bundle. Every value here is pulled from
real data, never generated -- this is the groundedness anchor: the classifier can
only reason over, and cite, what this node actually found.

Writes exactly one state key: "evidence".

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

from state import Alert, Evidence, EvidenceItem, AlertState

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# device identity registry -- the gate checks `enrolled` directly.
# maps logical device name -> its network identity (MAC for the ESP32).
DEVICE_REGISTRY = {
    "esp32-01": {"mac": "f0:16:1d:51:4d:34", "class": "esp32_sensor", "bridged": False},
    "stm32-01": {"mac": None, "class": "stm32_sensor", "bridged": True},  # serial-bridged
}

# which behavioral columns we surface as evidence, per source
MQTT_COLS = ["msg_count", "bytes_total", "msg_size_mean", "iat_mean",
             "rate_per_sec", "unique_topics", "val_mean", "val_std"]
FEAT_COLS = ["pkt_count", "bytes_total", "up_down_ratio", "pkt_size_mean",
             "iat_mean", "frac_tcp", "frac_udp", "frac_icmp", "frac_arp",
             "syn_count", "rst_count", "uniq_dst_ip", "uniq_dst_port", "new_dst_count"]

HISTORY_WINDOWS = 3   # how many prior windows to include as context


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
def _window_row(device: str, ts: float) -> tuple[dict | None, str]:
    """Return (row, source) for the window at ts. ESP32 has both network+behavioral;
    pick the source that has a row within 5s of ts."""
    # behavioral (both devices)
    mqtt = [r for r in _load("mqtt_behavior.csv") if r["device"] == device]
    mrow = min(mqtt, key=lambda r: abs(_fnum(r["ts"]) - ts), default=None)
    if mrow and abs(_fnum(mrow["ts"]) - ts) <= 5:
        return mrow, "mqtt"
    return None, "mqtt"

def _feature_row(device: str, ts: float) -> dict | None:
    """ESP32-only network features, matched by MAC and nearest ts."""
    reg = DEVICE_REGISTRY.get(device, {})
    mac = reg.get("mac")
    if not mac:
        return None
    feats = [r for r in _load("features.csv") if r["mac"] == mac]
    frow = min(feats, key=lambda r: abs(_fnum(r["ts"]) - ts), default=None)
    if frow and abs(_fnum(frow["ts"]) - ts) <= 5:
        return frow
    return None

def _baseline_stats(fname: str, match_col: str, match_val: str, cols: list[str]) -> dict[str, float]:
    """Mean of each column over this device's clean-normal baseline."""
    rows = [r for r in _load(fname) if r.get(match_col) == match_val]
    out = {}
    for c in cols:
        vals = [_fnum(r[c]) for r in rows if c in r]
        out[c] = statistics.fmean(vals) if vals else 0.0
    return out

def _history(device: str, ts: float, source: str) -> list[float]:
    """rate_per_sec (mqtt) or pkt_count (features) for the HISTORY_WINDOWS before ts."""
    if source == "mqtt":
        rows = sorted((r for r in _load("mqtt_behavior.csv") if r["device"] == device),
                      key=lambda r: _fnum(r["ts"]))
        col = "rate_per_sec"
    else:
        reg = DEVICE_REGISTRY.get(device, {})
        rows = sorted((r for r in _load("features.csv") if r["mac"] == reg.get("mac")),
                      key=lambda r: _fnum(r["ts"]))
        col = "pkt_count"
    prior = [r for r in rows if _fnum(r["ts"]) < ts][-HISTORY_WINDOWS:]
    return [round(_fnum(r[col]), 3) for r in prior]


# ---------- the node ----------
def investigate(state: AlertState) -> dict:
    """LangGraph node. Reads state['alert'], returns {'evidence': Evidence, 'audit': [...]}."""
    alert: Alert = state["alert"]
    device = alert["device"]
    ts = float(alert["window_start"])   # alert carries the unix ts of the window

    reg = DEVICE_REGISTRY.get(device, {})
    enrolled = device in DEVICE_REGISTRY

    items: list[EvidenceItem] = []

    # identity evidence
    items.append(EvidenceItem(id="identity.device", source="identity",
                              key="device", value=device))
    items.append(EvidenceItem(id="identity.bridged", source="identity",
                              key="bridged", value=reg.get("bridged", False)))

    # behavioral window + baseline deltas (both devices)
    mrow, _ = _window_row(device, ts)
    if mrow:
        base = _baseline_stats("baseline_mqtt.csv", "device", device, MQTT_COLS)
        for c in MQTT_COLS:
            cur = round(_fnum(mrow[c]), 3)
            items.append(EvidenceItem(id=f"mqtt.{c}", source="mqtt", key=c, value=cur))
            items.append(EvidenceItem(id=f"baseline.mqtt.{c}", source="baseline",
                                      key=f"normal_{c}", value=round(base[c], 3)))

    # network window + baseline deltas (ESP32 only)
    frow = _feature_row(device, ts)
    if frow:
        base = _baseline_stats("baseline_features.csv", "mac", reg.get("mac"), FEAT_COLS)
        for c in FEAT_COLS:
            cur = round(_fnum(frow[c]), 3)
            items.append(EvidenceItem(id=f"features.{c}", source="features", key=c, value=cur))
            items.append(EvidenceItem(id=f"baseline.features.{c}", source="baseline",
                                      key=f"normal_{c}", value=round(base[c], 3)))

    # recent history (context: was it normal just before?)
    src = "mqtt" if mrow else "features"
    hist = _history(device, ts, src)
    items.append(EvidenceItem(id="history.recent", source="history",
                              key="prior_windows", value=hist))

    evidence = Evidence(items=items, enrolled=enrolled)

    audit = [{
        "node": "investigator",
        "alert_id": alert["alert_id"],
        "evidence_count": len(items),
        "has_network": frow is not None,
        "has_behavioral": mrow is not None,
    }]
    return {"evidence": evidence, "audit": audit}


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