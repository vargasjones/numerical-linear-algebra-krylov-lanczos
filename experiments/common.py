from __future__ import annotations

import json
from functools import wraps
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


def first_below(values, tol, *, start_index=1):
    for i, value in enumerate(values, start=start_index):
        if np.isfinite(value) and value <= tol:
            return int(i)
    return None


def single_threaded(function):
    """Run one experiment with one BLAS/LAPACK thread for reproducibility."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with threadpool_limits(limits=1):
            return function(*args, **kwargs)
    return wrapped


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def save_summary(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(data), indent=2), encoding="utf-8")


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
        "lines.linewidth": 1.8,
    })
    return plt
