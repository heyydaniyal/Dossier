"""Alerts from fired rules, in the frozen Alert contract. Runtime-safe (no labels).

Alert unit (FROZEN in P2): one alert per (account_key, daily window) on which >= 1 rule fired.
Windows are the disjoint calendar days, so alerts of one account never overlap; an account
flagged on consecutive days has one alert per day. The account is evaluated in BOTH roles.
alert_id is an opaque keyed hash: it carries no account-ID order (D5).
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from src.alerts.rules import ALL_STATS
from src.contracts.models import Alert
from src.data.hashing import keyed_hex

ALERT_ID_PURPOSE = "alert_id"
ALERT_COLUMNS = [
    "alert_id",
    "account_key",
    "window_start",
    "window_end",
    "as_of",
    "created_at",
    "period",
    "triggered_rules",
    "n_rules_triggered",
    "peer_group",
    *ALL_STATS,
]


def alert_id(variant: str, account_key: str, window_start, seed: int) -> str:
    return "A" + keyed_hex(
        f"{variant}|{account_key}|{window_start.date().isoformat()}", ALERT_ID_PURPOSE, seed
    )


def alerts_from_fired(fired: pl.DataFrame, variant: str, period: str, seed: int) -> pl.DataFrame:
    a = fired.filter(pl.col("n_rules_triggered") > 0)
    ids = [
        alert_id(variant, k, ws, seed)
        for k, ws in zip(a["account_key"], a["window_start"], strict=True)
    ]
    a = a.with_columns(
        pl.Series("alert_id", ids, dtype=pl.String),
        (pl.col("as_of") - timedelta(seconds=1)).alias("window_end"),
        pl.col("as_of").alias("created_at"),
        pl.lit(period).alias("period"),
    )
    return a.select(ALERT_COLUMNS)


def to_alert(row: dict) -> Alert:
    feats: dict = {c: row[c] for c in ALL_STATS}
    feats["n_rules_triggered"] = row["n_rules_triggered"]
    feats["peer_group"] = row["peer_group"]
    return Alert(
        alert_id=row["alert_id"],
        account_key=row["account_key"],
        window_start=row["window_start"],
        window_end=row["window_end"],
        as_of=row["as_of"],
        triggered_rules=list(row["triggered_rules"]),
        rule_features=feats,
        created_at=row["created_at"],
    )


def validate_alerts(df: pl.DataFrame) -> int:
    """Validate every row against the frozen Alert contract; also one alert per account-day."""
    dup = df.group_by("account_key", "window_start").len().filter(pl.col("len") > 1)
    if dup.height:
        raise ValueError(f"{dup.height} account-days with more than one alert")
    if df["alert_id"].n_unique() != df.height:
        raise ValueError("alert_id collision")
    for row in df.iter_rows(named=True):
        to_alert(row)
    return df.height
