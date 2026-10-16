"""The ONE way model-fitting code (P3 permutation test, P4 model, P5 calibration) gets labels.

Open risk 26 (P3 task 0), decided 2026-10-06: labels are read only here, in the evaluation
harness. Offline drivers in scripts/p3 and scripts/p4 (allowance dated 2026-10-12; scripts/p5 not
yet) may import this module and src.eval.identity_guard, nothing else from src.eval
(tests/test_eval_isolation.py). src/features
and src/models stay label-free and receive y as an argument.

TEST is refused BEFORE any file is opened: the model holdout is evaluated once, in P5 task 8, by a
separate, explicit function that does not exist yet.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.eval.alert_labels import KEY_COLS, LABEL_DEFINITION_ID

FIT_PERIODS = ("TRAIN", "VALIDATION", "CALIBRATION")


class HoldoutAccessError(PermissionError):
    """Labels of a holdout period were requested by fitting code."""


def _refuse_holdout(periods: list[str]) -> list[str]:
    periods = list(periods)
    bad = [p for p in periods if p not in FIT_PERIODS]
    if bad or not periods:
        raise HoldoutAccessError(
            f"labels may be loaded only for {FIT_PERIODS}; asked for {periods}"
        )
    return periods


# Reporting strata (P4: per-typology recall, laundering-share strata, fan-in vs other rules).
# Built HERE so truth-derived fields never leave the evaluation module: callers receive only stratum
# NAMES per positive alert. Never model inputs: P4 builds X strictly from registry feature names.
DETAIL_COLUMNS = ("n_laundering_txns", "has_unattributed", "typologies")


def share_bin(share: float, edges: list[float]) -> str:
    """'share_[lo,hi)' for the bin containing share; the last bin is open-ended."""
    for lo, hi in zip(edges, [*edges[1:], None], strict=True):
        if share >= lo and (hi is None or share < hi):
            return f"share_[{lo:.2f},{'inf' if hi is None else f'{hi:.2f}'})"
    raise ValueError(f"share {share} below the first edge {edges[0]}")


def _label_details(periods: list[str], variant: str, store_root: Path) -> pl.DataFrame:
    y = load_fit_labels(periods, variant, store_root)
    det = (
        pl.scan_parquet(Path(store_root) / variant / "eval" / "alert_labels.parquet")
        .select("alert_id", *DETAIL_COLUMNS)
        .join(y.lazy().select("alert_id"), on="alert_id", how="semi")
        .collect()
    )
    out = y.join(det, on="alert_id", how="left")
    if out.height != y.height or out["n_laundering_txns"].null_count():
        raise RuntimeError("label details missing for some requested alerts")
    if ((out["y"] == 1) != (out["n_laundering_txns"] > 0)).any():
        raise RuntimeError("label details disagree with the frozen label")
    return out


def load_fit_strata(
    periods: list[str], variant: str, store_root: Path, strata_cfg: dict
) -> pl.DataFrame:
    """alert_id, period, y, strata (list of stratum names; empty for negatives) for pre-TEST
    alerts. strata_cfg = configs/p4_diagnosis.yaml -> strata. TEST is refused before any file is
    opened. Strata: laundering share of the account-day's legs (bins), fan-in rule fired or not,
    each pattern typology, unattributed laundering."""
    periods = _refuse_holdout(periods)
    det = _label_details(periods, variant, store_root)
    alerts = (
        pl.scan_parquet(Path(store_root) / variant / "runtime" / "alerts.parquet")
        .select("alert_id", "n_txn", "triggered_rules")
        .join(det.lazy().select("alert_id"), on="alert_id", how="semi")
        .collect()
    )
    j = det.join(alerts, on="alert_id", how="left")
    if j["n_txn"].null_count():
        raise RuntimeError("alerts missing from the runtime store")
    names = []
    for r in j.iter_rows(named=True):
        if r["y"] != 1:
            names.append([])
            continue
        share = r["n_laundering_txns"] / r["n_txn"]
        k = [share_bin(share, strata_cfg["laundering_share_bins"])]
        fan = strata_cfg["fan_in_rule"] in (r["triggered_rules"] or [])
        k.append("rule_fan_in" if fan else "rule_other")
        k += [f"typ_{t}" for t in strata_cfg["typologies"] if t in (r["typologies"] or [])]
        if strata_cfg.get("unattributed") and r["has_unattributed"]:
            k.append("unattributed")
        names.append(k)
    return (
        j.select("alert_id", "period", "y")
        .with_columns(pl.Series("strata", names, dtype=pl.List(pl.String)))
        .sort("alert_id")
    )


def load_fit_labels(periods: list[str], variant: str, store_root: Path) -> pl.DataFrame:
    """alert_id, period, y (0/1 Int8) for every alert of the requested pre-TEST periods."""
    periods = _refuse_holdout(periods)
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


def load_fit_laundering_txns(
    periods: list[str], variant: str, interim_dir: Path, split
) -> pl.DataFrame:
    """Raw key columns (alert_labels.KEY_COLS, timestamps UTC) of every laundering transaction with
    period start <= ts < period end, for the requested pre-TEST periods.

    P4 step 0b ONLY (configs/p4_leg_check.yaml, approved by Dani 2026-10-16): payment labels as the
    target of a DIAGNOSTIC payment model. Dated deviation from the frozen ontology ("transaction
    labels only derive alert labels"); nothing fitted from these labels is kept. TEST is refused
    before any file is opened."""
    periods = _refuse_holdout(periods)
    rng = [split.periods[p] for p in periods]
    cond = pl.lit(False)
    for a, b in rng:
        cond = cond | ((pl.col("timestamp") >= a) & (pl.col("timestamp") < b))
    out = (
        pl.scan_parquet(Path(interim_dir) / f"{variant}_trans.parquet")
        .select([*KEY_COLS, "is_laundering"])
        .with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
        .filter(cond & (pl.col("is_laundering") == 1))
        .select(KEY_COLS)
        .collect()
    )
    if out.height and out.select(KEY_COLS).is_duplicated().any():
        raise RuntimeError("laundering transactions are not unique by their key columns")
    return out
