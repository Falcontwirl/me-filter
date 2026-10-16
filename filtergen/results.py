"""Saving records and summarizing them with bootstrap confidence intervals."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from filtergen.upstream import ROOT

RESULTS = ROOT / "results"


def bootstrap_ci(values, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for the mean."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key), n=len(g))
        for m in metrics:
            if m not in g or g[m].isna().all():
                continue
            vals = g[m].dropna().to_numpy()
            lo, hi = bootstrap_ci(vals)
            row.update({m: vals.mean(), f"{m}_sd": vals.std(ddof=1) if len(vals) > 1 else 0.0,
                        f"{m}_lo": lo, f"{m}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)
