import json
from pathlib import Path
from typing import Dict

import numpy as np


def append_percentiles(metrics_path: str | Path, summary_path: str | Path) -> Dict[str, float]:
    """Append median and p90 metrics to an existing summary file.

    Parameters
    ----------
    metrics_path: str or Path
        JSONL file containing per-query metrics with fields ``em`` and ``f1``.
    summary_path: str or Path
        Path to the summary JSON file to update.

    Returns
    -------
    Dict[str, float]
        The computed statistics added to the summary file.
    """
    metrics_path = Path(metrics_path)
    summary_path = Path(summary_path)

    if not metrics_path.exists():
        return {}

    records = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        return {}

    f1s = [r.get("f1", 0.0) for r in records]
    ems = [r.get("em", 0.0) for r in records]

    stats: Dict[str, float] = {}
    if f1s:
        stats["median_f1"] = float(np.median(f1s))
        stats["p90_f1"] = float(np.percentile(f1s, 90))
    if ems:
        stats["median_em"] = float(np.median(ems))
        stats["p90_em"] = float(np.percentile(ems, 90))

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}
    summary.update(stats)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return stats