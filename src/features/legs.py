"""P4 step 0b: payment-level (leg) features for alerted accounts. Runtime-safe: no label is read.

One row per (alert, non-self transaction of the alerted account in its visible window), with the
same point-in-time rule as every P3 feature:  as_of - lookback <= ts < as_of
(src.features.compute.window_frame -> src.data.transactions.visible).

Definitions reuse the frozen P2 rule definitions (src.alerts.rules.legs; near-threshold band and
round unit from configs/p2_rules.yaml). Counterparty context (lever 2) is computed over ALL legs of
the same visible window. No wall-clock field (hour, weekday) and no identifier is a feature (P3-7,
D5). The transaction's raw key columns are carried (not features) so the evaluation module can
attach payment labels by exact key match.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from src.alerts import rules
from src.data.transactions import RUNTIME_COLUMNS
from src.features import compute as fc

KEY_COLUMNS = list(RUNTIME_COLUMNS)  # the transaction's raw columns: join key, never features

OWN = [
    "log10_usd",
    "is_out",
    "cross_currency",
    "cross_bank",
    "near_threshold",
    "round_amount",
    "log10_ratio_to_account_median",
    "share_of_account_volume",
    "log10_account_n_legs",
]
COUNTERPARTY = [
    "log10_cp_n_legs",
    "log10_cp_out_deg",
    "log10_cp_in_deg",
    "log10_pair_n_txn",
    "log10_same_amount_n",
]


def format_names(ctx: fc.FeatureContext) -> list[str]:
    return [f"leg_fmt_{fc.slug(f)}" for f in ctx.formats]


def leg_feature_names(ctx: fc.FeatureContext, with_format: bool = True) -> list[str]:
    return [*OWN, *COUNTERPARTY, *(format_names(ctx) if with_format else [])]


def _txns(win: pl.DataFrame) -> pl.DataFrame:
    """Non-self transactions of the window, one row each, with a window-local index."""
    t = win.filter(pl.col("from_key") != pl.col("to_key")).with_row_index("_t")
    same = t.group_by("amount_paid", "payment_currency").agg(pl.len().alias("_same_n"))
    return t.join(same, on=["amount_paid", "payment_currency"], how="left")


def legs_for_window(
    trans_usd: pl.LazyFrame, as_of: datetime, alerts: pl.DataFrame, ctx: fc.FeatureContext
) -> pl.DataFrame:
    """Leg rows for the alerts of ONE as_of. alerts: alert_id, account_key, as_of."""
    if alerts["as_of"].n_unique() != 1 or alerts["as_of"][0] != as_of:
        raise fc.FeatureError("legs_for_window needs the alerts of exactly this as_of")
    if alerts["account_key"].n_unique() != alerts.height:
        raise fc.FeatureError("one alert per account per window")
    win = fc.window_frame(trans_usd, as_of, ctx)
    fc._check_formats(win, ctx)
    t = _txns(win)
    cfg = ctx.rules_cfg["rules"]
    band = cfg["R03_STRUCTURING"]["band_usd"]
    unit = cfg["R09_ROUND_AMOUNTS"]["round_unit"]
    min_amt = cfg["R09_ROUND_AMOUNTS"]["min_amount"]
    # legs of every account (counterparty context), built by the frozen P2 leg definition
    lg = rules.legs(t.lazy(), ctx.rules_cfg).collect()
    if lg.height != 2 * t.height:
        raise fc.FeatureError("leg count != 2 x non-self transactions")
    lg = lg.with_columns(pl.concat([t["_t"], t["_t"]]).alias("_t"))
    # rules.legs emits out-legs then in-legs, each in transaction order: verify the pairing
    chk = lg.join(t.select("_t", "from_key", "to_key"), on="_t")
    want = (
        pl.when(pl.col("direction") == "out").then(pl.col("from_key")).otherwise(pl.col("to_key"))
    )
    other = (
        pl.when(pl.col("direction") == "out").then(pl.col("to_key")).otherwise(pl.col("from_key"))
    )
    if (
        chk.height != lg.height
        or (chk["account_key"] != chk.select(want).to_series()).any()
        or (chk["counterparty"] != chk.select(other).to_series()).any()
    ):
        raise fc.FeatureError("leg <-> transaction pairing broken")
    acct = lg.group_by("account_key").agg(
        pl.len().alias("_n_legs"),
        pl.col("counterparty").filter(pl.col("direction") == "out").n_unique().alias("_out_deg"),
        pl.col("counterparty").filter(pl.col("direction") == "in").n_unique().alias("_in_deg"),
        pl.col("usd").median().alias("_med_usd"),
        pl.col("usd").sum().alias("_vol_usd"),
    )
    pair = lg.group_by("account_key", "counterparty").agg(pl.len().alias("_pair_n"))
    targets = alerts.select("alert_id", "account_key")
    mine = lg.join(targets, on="account_key", how="inner")
    cp = acct.select(
        pl.col("account_key").alias("counterparty"),
        pl.col("_n_legs").alias("_cp_n"),
        pl.col("_out_deg").alias("_cp_out"),
        pl.col("_in_deg").alias("_cp_in"),
    )
    j = (
        mine.join(acct, on="account_key")
        .join(cp, on="counterparty")
        .join(pair, on=["account_key", "counterparty"])
        .join(t.select("_t", "_same_n", *KEY_COLUMNS), on="_t")
    )
    lg10 = lambda e: e.cast(pl.Float64).log10()  # noqa: E731
    out = j.select(
        "alert_id",
        *KEY_COLUMNS,
        lg10(pl.col("usd")).alias("log10_usd"),
        (pl.col("direction") == "out").cast(pl.Float64).alias("is_out"),
        pl.col("cross").cast(pl.Float64).alias("cross_currency"),
        (pl.col("from_bank") != pl.col("to_bank")).cast(pl.Float64).alias("cross_bank"),
        ((pl.col("usd") >= band["lo"]) & (pl.col("usd") < band["hi_exclusive"]))
        .cast(pl.Float64)
        .alias("near_threshold"),
        (
            (pl.col("amount") >= min_amt)
            & (((pl.col("amount") / unit).round(0) * unit - pl.col("amount")).abs() < 1e-6)
        )
        .cast(pl.Float64)
        .alias("round_amount"),
        lg10(pl.col("usd") / pl.col("_med_usd")).alias("log10_ratio_to_account_median"),
        (pl.col("usd") / pl.col("_vol_usd")).alias("share_of_account_volume"),
        lg10(pl.col("_n_legs")).alias("log10_account_n_legs"),
        lg10(pl.col("_cp_n")).alias("log10_cp_n_legs"),
        lg10(pl.col("_cp_out") + 1).alias("log10_cp_out_deg"),
        lg10(pl.col("_cp_in") + 1).alias("log10_cp_in_deg"),
        lg10(pl.col("_pair_n")).alias("log10_pair_n_txn"),
        lg10(pl.col("_same_n")).alias("log10_same_amount_n"),
        *[
            (pl.col("payment_format") == f).cast(pl.Float64).alias(n)
            for f, n in zip(ctx.formats, format_names(ctx), strict=True)
        ],
    )
    if out["alert_id"].n_unique() != alerts.height:
        raise fc.FeatureError("an alerted account has no visible non-self leg in its window")
    return out


def batch_legs(
    trans_usd: pl.LazyFrame, alerts: pl.DataFrame, ctx: fc.FeatureContext
) -> pl.DataFrame:
    parts = [
        legs_for_window(trans_usd, a, alerts.filter(pl.col("as_of") == a), ctx)
        for a in sorted(alerts["as_of"].unique().to_list())
    ]
    if not parts:
        raise fc.FeatureError("no alerts")
    return pl.concat(parts)


def leg_matrix(df: pl.DataFrame, names: list[str]) -> np.ndarray:
    return df.select(names).to_numpy().astype(np.float64)
