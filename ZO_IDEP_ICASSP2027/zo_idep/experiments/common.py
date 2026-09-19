"""Shared experiment utilities."""
from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zo_idep.src.utils import append_jsonl, jsonable, write_json

RAW = ROOT / "zo_idep" / "results" / "raw"
SUMMARY = ROOT / "zo_idep" / "results" / "summary"
FIGURES = ROOT / "zo_idep" / "results" / "figures"
PAPER_FIGURES = ROOT / "figures"


def compact_result(result):
    return {key: jsonable(value) for key, value in result.items()
            if key not in {"trace", "endpoint"}}


def best_chi(result):
    values = [row["chi"] for row in result["trace"]
              if not math.isnan(row["chi"])]
    return float(max(values)) if values else float("nan")


def endpoint_gaps(left, right):
    difference = np.asarray(left) - np.asarray(right)
    return float(np.max(np.abs(difference))), float(np.linalg.norm(difference))


def saving(adaptive_queries, full_queries):
    return float(1.0 - adaptive_queries / full_queries) if full_queries else 0.0


def reset_jsonl(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def save_run(path: Path, run, config):
    append_jsonl(path, {"config": config, "result": run})


def aggregate_by(records, group_keys, value_keys):
    groups = {}
    for record in records:
        key = tuple(record[name] for name in group_keys)
        groups.setdefault(key, []).append(record)
    output = []
    for key, rows in sorted(groups.items(), key=lambda item: item[0]):
        aggregate = {name: value for name, value in zip(group_keys, key)}
        aggregate["runs"] = len(rows)
        for name in value_keys:
            values = np.asarray([row[name] for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            aggregate[name + "_count"] = int(finite.size)
            aggregate[name + "_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            aggregate[name + "_std"] = (float(np.std(finite, ddof=1))
                                         if finite.size > 1 else 0.0)
            aggregate[name + "_se"] = (aggregate[name + "_std"] / np.sqrt(finite.size)
                                        if finite.size > 1 else 0.0)
            aggregate[name + "_max"] = float(np.max(finite)) if finite.size else float("nan")
        output.append(aggregate)
    return output


def write_summary(name: str, records, summary):
    write_json(SUMMARY / f"{name}.json", {"records": records, "summary": summary})
