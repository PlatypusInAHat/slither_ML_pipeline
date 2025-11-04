# src/slither_tools/report_parser.py
import json
from pathlib import Path

IMPACT_RANK = {"High": 3, "Medium": 2, "Low": 1, "Informational": 0}

def parse_slither_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {}
    detectors = data.get("results", {}).get("detectors", [])
    per_file = {}
    for det in detectors:
        label = det.get("check", "")
        for elem in det.get("elements", []):
            func = elem.get("name")
            sm = elem.get("source_mapping", {})
            filename = sm.get("filename_relative") or sm.get("filename_absolute")
            if not func or not filename:
                continue
            cur = per_file.setdefault(filename, {})
            rank = IMPACT_RANK.get(det.get("impact", "Low"), 0)
            old = cur.get(func)
            if old is None or rank > old[1]:
                cur[func] = (label, rank)
    for fname in list(per_file.keys()):
        per_file[fname] = {fn: lr[0] for fn, lr in per_file[fname].items()}
    return per_file
