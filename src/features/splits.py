"""P3 task 1-2: the frozen P2 split applied to feature rows; unseen-account slice; agent-eval rule.

Kept out of src/data/periods.py on purpose: that file is part of the P2 cache key (open risk 28).
Runtime-safe: everything here reads only the runtime alert store columns (period, account_key).
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from src.data.periods import ORDER, Split

FITTED = ("TRAIN", "VALIDATION", "CALIBRATION")
SPLIT_PERIODS = (*FITTED, "TEST")


class SplitViolation(RuntimeError):
    """A feature row or an evaluation alert breaks the frozen temporal split."""


def assign_periods(df: pl.DataFrame, split: Split) -> pl.DataFrame:
    """Period of each row by its window_start (half-open intervals). Rows outside the four split
    periods (burn-in, embargo, tail) are refused, never silently dropped."""
    names = [split.period_of(ws) for ws in df["window_start"].to_list()]
    bad = [n for n in names if n not in SPLIT_PERIODS]
    if bad:
        raise SplitViolation(
            f"{len(bad)} rows outside TRAIN/VALIDATION/CALIBRATION/TEST: {set(bad)}"
        )
    out = df.with_columns(pl.Series("split_period", names, dtype=pl.String))
    if "period" in df.columns and (out["period"] != out["split_period"]).any():
        raise SplitViolation("stored period disagrees with the frozen split")
    return out.drop("period") if "period" in df.columns else out


def check_boundaries(df: pl.DataFrame, split: Split) -> dict:
    """Split test, two distinct rules for consecutive periods P < Q (raises on any violation):
      (1) phase rule:  max(as_of in P) + embargo  <  min(as_of in Q)        (strict)
      (2) window rule: min(as_of in Q) - L_max   >=  max(as_of in P)
    (2) says Q's feature windows never reach P's label windows. With the frozen split
    (embargo = L_max = 24 h, measured gap 48 h) both hold with a margin; they differ when the
    embargo and L_max differ, so each is tested on its own. Returns the per-pair margins."""
    per = (
        df.group_by("split_period")
        .agg(pl.col("as_of").min().alias("lo"), pl.col("as_of").max().alias("hi"))
        .to_dicts()
    )
    lim = {r["split_period"]: (r["lo"], r["hi"]) for r in per}
    present = [p for p in SPLIT_PERIODS if p in lim]
    out = {}
    for p, q in zip(present, present[1:], strict=False):
        emb = [e for e in ORDER[ORDER.index(p) + 1 : ORDER.index(q)] if e.startswith("EMBARGO")]
        embargo = sum((split.periods[e][1] - split.periods[e][0] for e in emb), timedelta(0))
        gap = lim[q][0] - lim[p][1]
        if not lim[p][1] + embargo < lim[q][0]:
            raise SplitViolation(f"{p} -> {q}: max as_of + embargo >= min as_of of {q}")
        if not lim[q][0] - split.l_max >= lim[p][1]:
            raise SplitViolation(f"{q} feature windows reach into {p}")
        out[f"{p}->{q}"] = {"gap_hours": gap.total_seconds() / 3600, "embargo_hours":
                            embargo.total_seconds() / 3600}  # fmt: skip
    return out


def unseen_account_mask(alerts: pl.DataFrame) -> pl.Series:
    """Regime B: TEST alerts of accounts with NO alert in TRAIN (runtime alert store only)."""
    train_accounts = alerts.filter(pl.col("period") == "TRAIN")["account_key"].unique()
    return (pl.col("period") == "TEST") & ~pl.col("account_key").is_in(train_accounts.to_list())


def unseen_test_slice(alerts: pl.DataFrame) -> pl.DataFrame:
    return alerts.filter(unseen_account_mask(alerts))


def require_test_period(alerts: pl.DataFrame) -> pl.DataFrame:
    """P3 task 2 (FROZEN): every alert investigated by agents during evaluation comes from TEST,
    so the model is out-of-sample on each one. The evaluation harness calls this on its cases."""
    if "period" not in alerts.columns:
        raise SplitViolation("alerts need their stored period")
    bad = alerts.filter(pl.col("period") != "TEST")
    if bad.height:
        raise SplitViolation(f"{bad.height} agent-evaluation alerts are not from TEST")
    return alerts
