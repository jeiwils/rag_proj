"""Compute active-parameter tokens (APT) for result directories.

APT is defined as the product of the number of active parameters in a
model and the number of tokens processed. Token usage is read from the
``token_usage.json`` files produced by pipeline runs while the active
parameter count is inferred from the model name.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Sequence

from .utils import (
    get_result_dirs,
    rag_run_paths,
    load_token_usage,
    parse_traversal_run_dir,
)


def _active_params_from_model(model: str) -> float:
    """Return the number of active parameters for ``model``.

    Model names containing patterns like ``7b`` or ``14b`` are interpreted as
    billions of parameters. Mixture-of-experts model names such as ``2x7b``
    indicate ``experts`` times ``per_expert`` billions of active parameters.
    The returned value is the actual parameter count (not divided by 1e9).
    """
    m = model.lower()
    moe_match = re.search(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)b", m)
    if moe_match:
        experts = float(moe_match.group(1))
        per_expert = float(moe_match.group(2))
        return experts * per_expert * 1e9
    match = re.search(r"(\d+(?:\.\d+)?)b", m)
    if match:
        return float(match.group(1)) * 1e9
    raise ValueError(f"Unable to determine active parameters from model name: {model}")


def apt_for_result(result_dir: Path) -> float:
    """Compute APT for a single result directory.

    Parameters
    ----------
    result_dir:
        Path to a result directory such as
        ``data/results/{model}/{dataset}/{split}/{variant}``.

    Returns
    -------
    float
        Active-parameter tokens for the run.
    """
    model, dataset, split, seed = parse_traversal_run_dir(result_dir)
    mode = result_dir.name.rsplit("_seed", 1)[0]
    usage_path = rag_run_paths(model, dataset, split, seed, mode)["answers"]["token_usage"]
    data = load_token_usage(usage_path)["global"]
    tokens = float(data.get("tokens_total", 0) or 0)
    params = _active_params_from_model(model)
    return params * tokens


def summarize_apts(result_dirs: Sequence[Path]) -> Dict[Path, float]:
    """Return APT values for each directory in ``result_dirs``."""
    return {rd: apt_for_result(rd) for rd in result_dirs}


__all__ = ["apt_for_result", "summarize_apts"]


if __name__ == "__main__":
    for directory in get_result_dirs(required="token_usage.json"):
        apt = apt_for_result(directory)
        print(f"{directory}: {apt:.3e} param-tokens")