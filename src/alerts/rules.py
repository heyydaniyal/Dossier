"""Rule-based alert layer: account-day statistics and rule firing. Runtime-safe (no labels).

Every statistic for an alert with as_of T is computed from transactions with
T - L_max <= ts < T (src.data.transactions.visible). Both roles count: the account as sender
(out-legs) and as receiver (in-legs). Self-transfers (same bank and account) are excluded.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import yaml

from src.data.periods import ROOT, Split
from src.data.transactions import visible

RULES_CONFIG = ROOT / "configs" / "p2_rules.yaml"
THRESHOLDS_FILE = ROOT / "configs" / "p2_rule_thresholds.yaml"

# statistics used by rules, in a fixed order
RULE_STATS = [
    "max_txn_usd",
    "volume_usd",
    "n_near_threshold",
    "n_distinct_senders",
    "n_distinct_receivers",
    "pass_through_usd",
    "high_risk_channel_usd",
    "n_cross_currency",
    "n_round_amounts",
]
# descriptive context carried on the alert (observable, not used for firing)
CONTEXT_STATS = ["n_txn", "in_usd", "out_usd"]
ALL_STATS = RULE_STATS + CONTEXT_STATS


def load_rules(path: Path = RULES_CONFIG) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    stats = [r["stat"] for r in cfg["rules"].values()]
    if sorted(stats) != sorted(RULE_STATS) or len(set(stats)) != len(stats):
        raise ValueError(f"each rule needs exactly one of {RULE_STATS}, got {stats}")
    if not 6 <= len(cfg["rules"]) <= 10:
        raise ValueError("the rule set must have 6-10 rules (P2 task 3)")
    return cfg


def rule_ids(cfg: dict) -> list[str]:
    return list(cfg["rules"])  # config order (never ID order of accounts)


# ---------------------------------------------------------------- legs


def legs(lf: pl.LazyFrame, cfg: dict) -> pl.LazyFrame:
    """One row per (transaction, role). Needs usd_paid / usd_received (src.data.fx.with_usd)."""
    r = cfg["rules"]
    channels = r["R07_HIGH_RISK_CHANNEL"]["channels"]
    base = lf.filter(pl.col("from_key") != pl.col("to_key")).with_columns(
        (pl.col("payment_currency") != pl.col("receiving_currency")).alias("cross"),
        pl.col("payment_format").is_in(channels).alias("high_risk"),
    )
    common = ["timestamp", "cross", "high_risk"]
    out_legs = base.select(
        pl.col("from_key").alias("account_key"),
        pl.col("to_key").alias("counterparty"),
        pl.lit("out").alias("direction"),
        pl.col("usd_paid").alias("usd"),
        pl.col("amount_paid").alias("amount"),
        *common,
    )
    in_legs = base.select(
        pl.col("to_key").alias("account_key"),
        pl.col("from_key").alias("counterparty"),
        pl.lit("in").alias("direction"),
        pl.col("usd_received").alias("usd"),
        pl.col("amount_received").alias("amount"),
        *common,
    )
    return pl.concat([out_legs, in_legs])


def _stats_exprs(cfg: dict) -> list[pl.Expr]:
    r = cfg["rules"]
    band = r["R03_STRUCTURING"]["band_usd"]
    lo_ratio, hi_ratio = r["R06_RAPID_IN_OUT"]["out_in_ratio"]
    unit = r["R09_ROUND_AMOUNTS"]["round_unit"]
    min_amt = r["R09_ROUND_AMOUNTS"]["min_amount"]
    is_in = pl.col("direction") == "in"
    is_out = pl.col("direction") == "out"
    is_round = (pl.col("amount") >= min_amt) & (
        ((pl.col("amount") / unit).round(0) * unit - pl.col("amount")).abs() < 1e-6
    )
    return [
        pl.len().cast(pl.Int64).alias("n_txn"),
        pl.col("usd").max().alias("max_txn_usd"),
        pl.col("usd").sum().alias("volume_usd"),
        ((pl.col("usd") >= band["lo"]) & (pl.col("usd") < band["hi_exclusive"]))
        .sum()
        .cast(pl.Int64)
        .alias("n_near_threshold"),
        pl.col("counterparty").filter(is_in).n_unique().cast(pl.Int64).alias("n_distinct_senders"),
        pl.col("counterparty")
        .filter(is_out)
        .n_unique()
        .cast(pl.Int64)
        .alias("n_distinct_receivers"),
        pl.col("usd").filter(is_in).sum().alias("in_usd"),
        pl.col("usd").filter(is_out).sum().alias("out_usd"),
        pl.col("timestamp").filter(is_in).min().alias("_first_in"),
        pl.col("timestamp").filter(is_out).max().alias("_last_out"),
        pl.col("usd").filter(pl.col("high_risk")).sum().alias("high_risk_channel_usd"),
        pl.col("cross").sum().cast(pl.Int64).alias("n_cross_currency"),
        is_round.sum().cast(pl.Int64).alias("n_round_amounts"),
    ], (lo_ratio, hi_ratio)


def account_stats(legs_lf: pl.LazyFrame, cfg: dict) -> pl.LazyFrame:
    exprs, (lo, hi) = _stats_exprs(cfg)
    ratio = pl.col("out_usd") / pl.col("in_usd")
    pass_through = (
        pl.when(
            (pl.col("in_usd") > 0)
            & (pl.col("out_usd") > 0)
            & (ratio >= lo)
            & (ratio <= hi)
            & (pl.col("_last_out") > pl.col("_first_in"))
        )
        .then(pl.min_horizontal("in_usd", "out_usd"))
        .otherwise(0.0)
    )
    return (
        legs_lf.group_by("account_key")
        .agg(exprs)
        .with_columns(pass_through.alias("pass_through_usd"))
        .drop("_first_in", "_last_out")
    )


def day_stats(
    trans_usd: pl.LazyFrame, window_start: datetime, split: Split, cfg: dict
) -> pl.DataFrame:
    """All active accounts' statistics for one daily window (point-in-time by construction)."""
    as_of = split.as_of(window_start)
    lf = visible(trans_usd, as_of, split.lookback_start(as_of))
    st = account_stats(legs(lf, cfg), cfg).collect()
    return st.select(
        "account_key",
        pl.lit(window_start).alias("window_start"),
        pl.lit(as_of).alias("as_of"),
        *[pl.col(c).fill_null(0) for c in ALL_STATS],
    ).sort("account_key")  # deterministic storage order only; never used as signal


# ---------------------------------------------------------------- firing


def _threshold_expr(rule: str, spec: dict) -> pl.Expr:
    if "by_peer" in spec:
        keys = list(spec["by_peer"])
        vals = pl.Series(list(spec["by_peer"].values()), dtype=pl.Float64)
        return pl.col("peer_group").replace_strict(
            keys, vals, default=spec["default"], return_dtype=pl.Float64
        )
    return pl.lit(float(spec["threshold"]))


def fire(stats: pl.DataFrame, thresholds: dict, cfg: dict) -> pl.DataFrame:
    """Add fired_<rule> booleans, triggered_rules (config order) and n_rules_triggered."""
    rules = cfg["rules"]
    active = [
        r for r in rule_ids(cfg) if r in thresholds["rules"] and thresholds["rules"][r]["active"]
    ]
    cols = []
    for r in active:
        stat = pl.col(rules[r]["stat"])
        cols.append(
            ((stat > 0) & (stat >= _threshold_expr(r, thresholds["rules"][r]))).alias(f"fired_{r}")
        )
    out = stats.with_columns(cols)
    trig = pl.concat_list(
        [pl.when(pl.col(f"fired_{r}")).then(pl.lit(r)).otherwise(None) for r in active]
    ).list.drop_nulls()
    return out.with_columns(trig.alias("triggered_rules")).with_columns(
        pl.col("triggered_rules").list.len().cast(pl.Int64).alias("n_rules_triggered")
    )


def load_thresholds(path: Path = THRESHOLDS_FILE) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
