"""Identity guard v2 (proposal): seen/unseen alert and positive counts per rolling-origin fold.

Reads ONLY alerts with period in {TRAIN, VALIDATION} and window_start < 2022-09-08 UTC (filtered
lazily before collect, then asserted) and their labels. Never touches CALIBRATION/TEST rows.

Folds (daily windows, 1-day embargo between the last fit day and the evaluated day):
  F1  fit 09-02         -> eval 09-04
  F2  fit 09-02..09-03  -> eval 09-05
  F3  fit 09-02..09-05  -> eval 09-07 (VALIDATION)
seen = the alert's account_key has an alert on one of the fold's FIT days (the model's training
rows, if the model trains on alerts). 'gap' = unseen, but the account has an alert on an embargo
day (not trained on; shown only to judge how sensitive 'seen' is to the definition).

Run from the repo root:
  uv run python scripts/p2/review/fold_counts_train_val.py [--base data/p2/HI-Medium]
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

CUTOFF = datetime(2022, 9, 8, tzinfo=UTC)
PERIODS = ["TRAIN", "VALIDATION"]
FOLDS = [
    ("F1", [date(2022, 9, 2)], date(2022, 9, 4)),
    ("F2", [date(2022, 9, 2), date(2022, 9, 3)], date(2022, 9, 5)),
    ("F3", [date(2022, 9, 2) + timedelta(days=i) for i in range(4)], date(2022, 9, 7)),
]


def load(base: Path) -> pl.DataFrame:
    alerts = (
        pl.scan_parquet(base / "runtime" / "alerts.parquet")
        .filter(pl.col("period").is_in(PERIODS) & (pl.col("window_start") < CUTOFF))
        .select("alert_id", "account_key", "period", "window_start")
    )
    labels = pl.scan_parquet(base / "eval" / "alert_labels.parquet").select(
        "alert_id", "is_true_positive"
    )
    df = (
        alerts.join(labels, on="alert_id", how="left")  # labels restricted to the filtered alerts
        .with_columns(pl.col("window_start").dt.date().alias("day"))
        .collect()
    )
    assert df.height > 0, "no TRAIN/VALIDATION alerts found"
    assert set(df["period"].unique().to_list()) <= set(PERIODS), "non-TRAIN/VALIDATION row read"
    assert df["window_start"].max() < CUTOFF, "row at/after 2022-09-08 read"
    assert df["is_true_positive"].null_count() == 0, "alert without label"
    assert df["alert_id"].n_unique() == df.height, "duplicate alert_id"
    assert df.group_by("account_key", "day").len()["len"].max() == 1, "two alerts per account-day"
    return df


def fold_table(df: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for name, fit_days, eval_day in FOLDS:
        fit_acc = df.filter(pl.col("day").is_in(fit_days))["account_key"].unique()
        gap_days = [d for d in df["day"].unique().to_list() if max(fit_days) < d < eval_day]
        gap_acc = df.filter(pl.col("day").is_in(gap_days))["account_key"].unique()
        ev = df.filter(pl.col("day") == eval_day).with_columns(
            pl.col("account_key").is_in(fit_acc.implode()).alias("seen"),
            pl.col("account_key").is_in(gap_acc.implode()).alias("in_gap"),
        )
        assert ev.height > 0, f"{name}: no alerts on eval day {eval_day}"
        for sub, m in (("seen", pl.col("seen")), ("unseen", ~pl.col("seen"))):
            e = ev.filter(m)
            n, p = e.height, int(e["is_true_positive"].sum())
            gap = e.filter(pl.col("in_gap")) if sub == "unseen" else e.clear()
            rows.append(
                {
                    "fold": name,
                    "fit_days": f"{fit_days[0]:%m-%d}..{fit_days[-1]:%m-%d}",
                    "eval_day": f"{eval_day:%m-%d}",
                    "subset": sub,
                    "alerts": n,
                    "positives": p,
                    "prevalence": round(p / n, 4) if n else None,
                    "unseen_alerted_in_gap": gap.height if sub == "unseen" else None,
                    "unseen_pos_alerted_in_gap": (
                        int(gap["is_true_positive"].sum()) if sub == "unseen" else None
                    ),
                }
            )
    return pl.DataFrame(rows)


def recurrence(df: pl.DataFrame) -> pl.DataFrame:
    """Eval-day alerts whose account is evaluated on more than one fold's eval day: if these
    carry a material share of positives, use the account-level (cluster) bootstrap."""
    ev = df.filter(pl.col("day").is_in([e for _, _, e in FOLDS]))
    k = ev.group_by("account_key").agg(pl.len().alias("n_days"))
    ev = ev.join(k, on="account_key")
    return (
        ev.group_by(pl.col("n_days") > 1)
        .agg(
            pl.col("account_key").n_unique().alias("accounts"),
            pl.len().alias("eval_alerts"),
            pl.col("is_true_positive").sum().alias("eval_positives"),
        )
        .rename({"n_days": "account_on_2plus_eval_days"})
        .sort("account_on_2plus_eval_days")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("data/p2/HI-Medium"))
    args = ap.parse_args()
    df = load(args.base)
    t = fold_table(df)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(t)
        print(
            t.group_by("subset")
            .agg(pl.col("alerts").sum(), pl.col("positives").sum())
            .sort("subset")
            .with_columns((pl.col("positives") / pl.col("alerts")).round(4).alias("prevalence"))
        )
        print(recurrence(df))


if __name__ == "__main__":
    main()
