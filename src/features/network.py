"""Known-flagged counterparties: NOT a model feature (MDP, P3 task 4). P10's Network tool wraps it.

Point-in-time on both inputs:
  - edges: transactions with  as_of - lookback <= ts < as_of  (src.data.transactions.visible);
  - flags: simulated dispositions CONFIRMED and CLOSED strictly before as_of
    (src.data.stores.dispositions_visible). The dispositions store holds pre-TEST alerts only,
    and its records carry only {alert_id, account_key, disposition, closed_at}.
Never the label vector: the flags are the noisy simulated analyst decisions of P2
(confirmed_suspicious precision 56.5%), and the output says so.
Runtime-safe: no ground truth is read here.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from src.data.transactions import visible

CONFIRMED = "confirmed_suspicious"


def flagged_counterparties(
    trans: pl.LazyFrame,
    dispositions_visible: pl.LazyFrame,
    account_key: str,
    as_of: datetime,
    lookback: timedelta,
) -> dict:
    """dispositions_visible must come from src.data.stores.dispositions_visible(as_of); it is
    re-filtered here (closed_at < as_of) so a wrong caller cannot widen it."""
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    win = visible(trans, as_of, as_of - lookback).filter(pl.col("from_key") != pl.col("to_key"))
    cps = (
        pl.concat(
            [
                win.filter(pl.col("from_key") == account_key).select(
                    pl.col("to_key").alias("counterparty")
                ),
                win.filter(pl.col("to_key") == account_key).select(
                    pl.col("from_key").alias("counterparty")
                ),
            ]
        )
        .unique()
        .collect()
    )
    flagged = (
        dispositions_visible.filter(
            (pl.col("closed_at") < as_of) & (pl.col("disposition") == CONFIRMED)
        )
        .select(pl.col("account_key").alias("counterparty"))
        .unique()
        .collect()
    )
    hit = cps.join(flagged, on="counterparty").sort("counterparty")["counterparty"].to_list()
    return {
        "account_key": account_key,
        "as_of": as_of.isoformat(),
        "lookback_hours": lookback.total_seconds() / 3600.0,
        "n_counterparties": cps.height,
        "n_flagged_counterparties": len(hit),
        "flagged_counterparties": hit,
        "caveat": "flags are simulated analyst dispositions (noisy), not ground truth",
    }
