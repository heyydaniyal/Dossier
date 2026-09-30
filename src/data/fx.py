"""D6: USD conversion with the dataset's own fixed exchange rates, derived on TRAIN only.

Rates: amount_paid * usd[pay] == amount_received * usd[recv] for every cross-currency txn, so
log(received/paid) = log usd[pay] - log usd[recv]. Per currency pair we take the median of that
log-ratio (robust to cent rounding and to tiny Bitcoin amounts), then solve all pairs jointly by
weighted least squares with log usd['US Dollar'] = 0. No external FX (D6).
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src.data.transactions import DataError, visible

REPORTING = "US Dollar"


def _sig(x: float, n: int = 10) -> float:
    return float(f"{x:.{n}g}")


def derive_usd_per_unit(lf: pl.LazyFrame, start: datetime, end: datetime) -> dict:
    """Fit usd_per_unit on transactions with start <= ts < end (TRAIN)."""
    win = visible(lf, end, start)
    seen = (
        pl.concat(
            [
                win.select(pl.col("payment_currency").alias("c")),
                win.select(pl.col("receiving_currency").alias("c")),
            ]
        )
        .unique()
        .collect()["c"]
        .sort()
        .to_list()
    )
    pairs = (
        win.filter(pl.col("payment_currency") != pl.col("receiving_currency"))
        .select(
            "payment_currency",
            "receiving_currency",
            (pl.col("amount_received") / pl.col("amount_paid")).log().alias("lr"),
        )
        .group_by("payment_currency", "receiving_currency")
        .agg(pl.col("lr").median().alias("m"), pl.len().alias("n"))
        .sort("payment_currency", "receiving_currency")
        .collect()
    )
    if REPORTING not in seen:
        raise DataError("no US Dollar transactions in the fitting period")
    others = [c for c in seen if c != REPORTING]
    idx = {c: i for i, c in enumerate(others)}
    # connectivity: every currency must be linked (directly or via others) to USD
    adj: dict[str, set[str]] = {c: set() for c in seen}
    for a, b in zip(pairs["payment_currency"], pairs["receiving_currency"], strict=True):
        adj[a].add(b)
        adj[b].add(a)
    reach, stack = {REPORTING}, [REPORTING]
    while stack:
        for nb in adj[stack.pop()] - reach:
            reach.add(nb)
            stack.append(nb)
    missing = sorted(set(seen) - reach)
    if missing:
        raise DataError(f"currencies with no cross-currency path to USD in TRAIN: {missing}")

    a_mat = np.zeros((pairs.height, len(others)))
    y = np.zeros(pairs.height)
    w = np.sqrt(pairs["n"].to_numpy().astype(float))
    for r, (p, q, m) in enumerate(
        zip(pairs["payment_currency"], pairs["receiving_currency"], pairs["m"], strict=True)
    ):
        if p != REPORTING:
            a_mat[r, idx[p]] += 1.0
        if q != REPORTING:
            a_mat[r, idx[q]] -= 1.0
        y[r] = m
    x, *_ = np.linalg.lstsq(a_mat * w[:, None], y * w, rcond=None)
    resid = np.abs(a_mat @ x - y)
    rates = {REPORTING: 1.0} | {c: _sig(math.exp(x[idx[c]])) for c in others}
    return {
        "reporting_currency": REPORTING,
        "fitted_on": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "n_pairs": pairs.height,
        "n_cross_currency_txns": int(pairs["n"].sum()),
        "max_abs_log_residual": _sig(float(resid.max()) if resid.size else 0.0),
        "usd_per_unit": dict(sorted(rates.items())),
    }


def save(fx: dict, path: Path) -> None:
    header = (
        "# P2: usd_per_unit derived on TRAIN from the dataset's own cross-currency txns (D6).\n"
        "# Written by the P2 driver (run_p2); do not edit by hand. P3+ read this file.\n"
    )
    path.write_bytes((header + yaml.safe_dump(fx, sort_keys=False)).encode("utf-8"))  # LF on any OS


def load(path: Path) -> dict[str, float]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["usd_per_unit"]


def with_usd(lf: pl.LazyFrame, usd_per_unit: dict[str, float]) -> pl.LazyFrame:
    """Add usd_paid / usd_received. An unknown currency raises (replace_strict)."""
    m = pl.Series(list(usd_per_unit.values()), dtype=pl.Float64)
    keys = list(usd_per_unit.keys())
    return lf.with_columns(
        (
            pl.col("amount_paid")
            * pl.col("payment_currency").replace_strict(keys, m, return_dtype=pl.Float64)
        ).alias("usd_paid"),
        (
            pl.col("amount_received")
            * pl.col("receiving_currency").replace_strict(keys, m, return_dtype=pl.Float64)
        ).alias("usd_received"),
    )
