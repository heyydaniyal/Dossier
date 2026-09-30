"""Alert labels (P2 task 2). EVALUATION ONLY: reads the ground-truth column and the patterns file.

Definition (FROZEN in P2, id P2-v1): an alert is a TRUE POSITIVE iff its account is the sender or
the receiver of at least one laundering transaction (Is Laundering = 1) with
window_start <= ts < as_of. Typologies = the pattern typologies of those transactions;
laundering transactions in no pattern ("natural" laundering, IT-AML §3.4) are flagged
has_unattributed and add no typology.

Edge cases (documented in docs/P2_ALERT_LAYER.md §3):
  - a multi-day attempt: each daily alert is labelled by that day's transactions only; an
    account alerted on a day between its laundering transactions is a FALSE positive that day;
  - pass-through accounts: positive on the days they receive or send a laundering transaction;
  - an attempt crossing a period boundary: labelled per day, so both periods see positives.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.eval.labels import AlertLabel

LABEL_DEFINITION_ID = "P2-v1"
KEY_COLS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
]
EVAL_LABEL_COLUMNS = [
    "alert_id",
    "is_true_positive",
    "typologies",
    "label_definition_id",
    "n_laundering_txns",
    "has_unattributed",
    "attempt_ids",
]


def laundering_legs(trans_path: Path, patterns_path: Path, span_start, span_end) -> pl.DataFrame:
    """(account_key, window_start, row_id, attempt_id, typology) per laundering leg in span."""
    t = (
        pl.scan_parquet(trans_path)
        .select(["row_id", *KEY_COLS, "is_laundering"])
        .filter(pl.col("is_laundering") == 1)
    )
    p = pl.scan_parquet(patterns_path).select([*KEY_COLS, "attempt_id", "typology"])
    txn = (
        t.join(p, on=KEY_COLS, how="left")
        .with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
        .filter((pl.col("timestamp") >= span_start) & (pl.col("timestamp") < span_end))
        .with_columns(pl.col("timestamp").dt.truncate("1d").alias("window_start"))
        .collect()
    )
    if txn["row_id"].n_unique() != txn.height:
        raise RuntimeError("a laundering transaction matched more than one pattern row")
    cols = ["window_start", "row_id", "attempt_id", "typology", "timestamp"]
    out_leg = txn.select(
        pl.concat_str([pl.col("from_bank"), pl.lit("|"), pl.col("from_account")]).alias(
            "account_key"
        ),
        *cols,
    )
    in_leg = txn.select(
        pl.concat_str([pl.col("to_bank"), pl.lit("|"), pl.col("to_account")]).alias("account_key"),
        *cols,
    )
    return (
        pl.concat([out_leg, in_leg])
        .unique(["account_key", "row_id"])
        .sort("window_start", "row_id", "account_key")
    )


def positive_account_days(legs: pl.DataFrame) -> pl.DataFrame:
    return (
        legs.group_by("account_key", "window_start")
        .agg(
            pl.col("row_id").n_unique().cast(pl.Int64).alias("n_laundering_txns"),
            pl.col("typology").drop_nulls().unique().sort().alias("typologies"),
            pl.col("typology").is_null().any().alias("has_unattributed"),
            pl.col("attempt_id").drop_nulls().unique().sort().alias("attempt_ids"),
        )
        .sort("window_start", "account_key")
    )


def label_alerts(alerts: pl.DataFrame, pos: pl.DataFrame) -> pl.DataFrame:
    """alerts: alert_id, account_key, window_start. Every alert gets exactly one label row."""
    j = alerts.select("alert_id", "account_key", "window_start").join(
        pos, on=["account_key", "window_start"], how="left"
    )
    out = j.select(
        "alert_id",
        pl.col("n_laundering_txns").is_not_null().alias("is_true_positive"),
        pl.col("typologies").fill_null(pl.lit([], dtype=pl.List(pl.String))),
        pl.lit(LABEL_DEFINITION_ID).alias("label_definition_id"),
        pl.col("n_laundering_txns").fill_null(0),
        pl.col("has_unattributed").fill_null(False),
        pl.col("attempt_ids").fill_null(pl.lit([], dtype=pl.List(pl.Int32))),
    ).sort("alert_id")
    if out.height != alerts.height:
        raise RuntimeError("label join changed the number of alerts")
    return out.select(EVAL_LABEL_COLUMNS)


def to_alert_label(row: dict) -> AlertLabel:
    return AlertLabel(
        alert_id=row["alert_id"],
        is_true_positive=row["is_true_positive"],
        typologies=list(row["typologies"]),
        label_definition_id=row["label_definition_id"],
    )
