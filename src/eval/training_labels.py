"""The ONE way model-fitting code (P3 permutation test, P4 model, P5 calibration) gets labels.

Open risk 26 (P3 task 0), decided 2026-10-06: labels are read only here, in the evaluation
harness. Offline drivers in scripts/p3 (and later scripts/p4, scripts/p5) may import this module
and src.eval.identity_guard, nothing else from src.eval (tests/test_eval_isolation.py). src/features
and src/models stay label-free and receive y as an argument.

TEST is refused BEFORE any file is opened: the model holdout is evaluated once, in P5 task 8, by a
separate, explicit function that does not exist yet.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.eval.alert_labels import LABEL_DEFINITION_ID

FIT_PERIODS = ("TRAIN", "VALIDATION", "CALIBRATION")


class HoldoutAccessError(PermissionError):
    """Labels of a holdout period were requested by fitting code."""


def load_fit_labels(periods: list[str], variant: str, store_root: Path) -> pl.DataFrame:
    """alert_id, period, y (0/1 Int8) for every alert of the requested pre-TEST periods."""
    periods = list(periods)
    bad = [p for p in periods if p not in FIT_PERIODS]
    if bad or not periods:
        raise HoldoutAccessError(
            f"labels may be loaded only for {FIT_PERIODS}; asked for {periods}"
        )
    base = Path(store_root) / variant
    alerts = (
        pl.scan_parquet(base / "runtime" / "alerts.parquet")
        .filter(pl.col("period").is_in(periods))
        .select("alert_id", "period")
        .collect()
    )
    lab = (
        pl.scan_parquet(base / "eval" / "alert_labels.parquet")
        .select("alert_id", "is_true_positive", "label_definition_id")
        .join(alerts.lazy(), on="alert_id", how="semi")
        .collect()
    )
    if lab.height != alerts.height:
        raise RuntimeError(f"{alerts.height - lab.height} requested alerts have no label row")
    defs = set(lab["label_definition_id"].unique().to_list())
    if defs != {LABEL_DEFINITION_ID}:
        raise RuntimeError(
            f"unexpected label definition(s) {defs}; frozen is {LABEL_DEFINITION_ID}"
        )
    return (
        alerts.join(lab.select("alert_id", "is_true_positive"), on="alert_id")
        .select("alert_id", "period", pl.col("is_true_positive").cast(pl.Int8).alias("y"))
        .sort("alert_id")
    )
