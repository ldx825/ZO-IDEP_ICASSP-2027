"""Serialization, deterministic configuration, and summary helpers."""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(jsonable(payload), sort_keys=True) + "\n")


def summarize(records, keys):
    output = {"runs": len(records)}
    for key in keys:
        values = np.asarray([record[key] for record in records], dtype=float)
        output[key + "_mean"] = float(np.mean(values))
        output[key + "_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        output[key + "_se"] = (output[key + "_std"] / np.sqrt(values.size)
                                if values.size > 1 else 0.0)
        output[key + "_max"] = float(np.max(values))
    return output
