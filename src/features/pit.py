"""Point-in-time check helpers shared by tests/test_p3_features.py and scripts/p3/run_p3.py check.

Runtime-safe: label-free transformations of the label-free transaction frame.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import polars as pl


def scramble_outside_window(lf: pl.LazyFrame, as_of: datetime, lookback: timedelta) -> pl.LazyFrame:
    """Rewrite every row OUTSIDE [as_of - lookback, as_of): amounts, formats and receivers change
    (receiver := sender, i.e. self-transfers). A pipeline that reads any of these rows changes."""
    outside = (pl.col("timestamp") >= as_of) | (pl.col("timestamp") < as_of - lookback)
    return lf.with_columns(
        pl.when(outside).then(pl.col("amount_paid") * 7.3 + 1).otherwise(pl.col("amount_paid"))
        .alias("amount_paid"),
        pl.when(outside).then(pl.col("amount_received") * 7.3 + 1)
        .otherwise(pl.col("amount_received")).alias("amount_received"),
        pl.when(outside).then(pl.lit("Cash")).otherwise(pl.col("payment_format"))
        .alias("payment_format"),
        pl.when(outside).then(pl.col("from_key")).otherwise(pl.col("to_key")).alias("to_key"),
    )  # fmt: skip


def close(a: float, b: float, abs_tol: float = 1e-9, rel_tol: float = 1e-9) -> bool:
    """Equality for feature values; NaN equals NaN (one missing-value convention)."""
    if a is None or b is None:
        return a is None and b is None
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def mismatches(a: pl.DataFrame, b: pl.DataFrame, names: list[str], **tol) -> list[str]:
    """Feature cells that differ between two frames with the same alert_ids (sorted)."""
    a, b = a.sort("alert_id"), b.sort("alert_id")
    if a["alert_id"].to_list() != b["alert_id"].to_list():
        return ["<alert_id sets differ>"]
    out = []
    for c in names:
        for x, y, aid in zip(a[c].to_list(), b[c].to_list(), a["alert_id"].to_list(), strict=True):
            if not close(x, y, **tol):
                out.append(f"{aid}:{c}: {x} vs {y}")
    return out
