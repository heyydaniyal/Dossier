"""Reviewer A, item 10c: do the bank IDs written with two zero-paddings split any real account?

TRAIN/VALIDATION ONLY: transactions with ts in [2022-09-02, 2022-09-06) U [2022-09-07, 2022-09-08)
UTC, and alerts of periods TRAIN / VALIDATION (window_start < 2022-09-08). Filters are applied
before any aggregation; the script asserts that no collected row is at or after 2022-09-08.
The label column is never selected (src.data.transactions.scan_transactions allow-list).
The full list of 61 collided bank IDs (P1, whole file) cannot be rebuilt without reading TEST
rows, so this script reports the collisions visible in TRAIN/VALIDATION.

Run from the repo root:
    uv run python scripts/p2/review/bank_padding_diagnostic.py [--interim-dir data/interim]
        [--alerts data/p2/HI-Medium/runtime/alerts.parquet] [--variant HI-Medium]
Prints, per normalised bank ID with >= 2 spellings in TRAIN/VALIDATION:
  spellings, accounts per spelling, account numbers seen under BOTH spellings (the dangerous case),
  txns per spelling, alerts per spelling, accounts-file bank names behind each spelling;
plus global checks: distinct keys before/after normalisation, cross-spelling "self-transfers",
receivers whose distinct-sender count would shrink after normalisation (R04), and accounts with
two alerts on one day under two spellings.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src.data.accounts import norm_bank  # noqa: E402
from src.data.transactions import interim_path, scan_transactions  # noqa: E402

TRAIN = (datetime(2022, 9, 2, tzinfo=UTC), datetime(2022, 9, 6, tzinfo=UTC))
VALID = (datetime(2022, 9, 7, tzinfo=UTC), datetime(2022, 9, 8, tzinfo=UTC))
HARD_LIMIT = datetime(2022, 9, 8, tzinfo=UTC)


def in_periods(col: str) -> pl.Expr:
    c = pl.col(col)
    return ((c >= TRAIN[0]) & (c < TRAIN[1])) | ((c >= VALID[0]) & (c < VALID[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim-dir", type=Path, default=ROOT / "data" / "interim")
    ap.add_argument("--variant", default="HI-Medium")
    ap.add_argument("--alerts", type=Path, default=None)
    args = ap.parse_args()
    alerts_path = args.alerts or ROOT / "data" / "p2" / args.variant / "runtime" / "alerts.parquet"
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_fmt_str_lengths(60)

    t = (
        scan_transactions(interim_path(args.interim_dir, args.variant, "trans"))
        .filter(in_periods("timestamp"))
        .select("timestamp", "from_bank", "from_account", "to_bank", "to_account")
        .collect()
    )
    assert t.height > 0
    assert t["timestamp"].max() < HARD_LIMIT, "row at/after 2022-09-08 loaded"
    assert t.filter(~in_periods("timestamp")).height == 0, "embargo/other row loaded"
    print(f"TRAIN+VALIDATION txns: {t.height:,} ({t['timestamp'].min()} .. {t['timestamp'].max()})")

    t = t.with_columns(
        norm_bank(pl.col("from_bank")).alias("from_norm"),
        norm_bank(pl.col("to_bank")).alias("to_norm"),
        pl.col("timestamp").dt.truncate("1d").alias("day"),
    )
    nodes = pl.concat(
        [
            t.select(pl.col("from_bank").alias("bank"), pl.col("from_account").alias("account")),
            t.select(pl.col("to_bank").alias("bank"), pl.col("to_account").alias("account")),
        ]
    ).unique()
    nodes = nodes.with_columns(norm_bank(pl.col("bank")).alias("bank_norm"))
    n_raw = nodes.height
    n_norm = nodes.select("bank_norm", "account").unique().height
    print(
        f"\nA. distinct (bank, account): raw {n_raw:,} | after stripping zeros {n_norm:,} "
        f"| merged by normalisation {n_raw - n_norm:,}  (0 => no account has two spellings)"
    )

    spell = nodes.select("bank", "bank_norm").unique()
    coll = (
        spell.group_by("bank_norm")
        .agg(pl.col("bank").sort().alias("spellings"), pl.len())
        .filter(pl.col("len") > 1)
        .sort("bank_norm")
    )
    print(
        f"\nB. normalised bank IDs with >= 2 spellings in TRAIN/VALIDATION: {coll.height} "
        f"(P1, whole file: 61)"
    )
    if coll.height == 0:
        return
    cb = set(coll["bank_norm"].to_list())

    # accounts under both spellings of the same normalised bank
    both = (
        nodes.filter(pl.col("bank_norm").is_in(cb))
        .group_by("bank_norm", "account")
        .agg(pl.col("bank").n_unique().alias("n_spellings"))
    )
    per_bank_acc = (
        nodes.filter(pl.col("bank_norm").is_in(cb))
        .group_by("bank_norm", "bank")
        .agg(pl.col("account").n_unique().alias("n_accounts"))
    )
    txn_from = (
        t.filter(pl.col("from_norm").is_in(cb))
        .group_by(pl.col("from_norm").alias("bank_norm"), pl.col("from_bank").alias("bank"))
        .len("n_out")
    )
    txn_to = (
        t.filter(pl.col("to_norm").is_in(cb))
        .group_by(pl.col("to_norm").alias("bank_norm"), pl.col("to_bank").alias("bank"))
        .len("n_in")
    )

    acc = pl.read_parquet(
        interim_path(args.interim_dir, args.variant, "accounts"),
        columns=["bank_id", "account_number", "bank_name"],
    )
    names = (
        nodes.filter(pl.col("bank_norm").is_in(cb))
        .join(
            acc.rename({"bank_id": "bank_norm", "account_number": "account"}),
            on=["bank_norm", "account"],
            how="left",
        )
        .group_by("bank_norm", "bank")
        .agg(
            pl.col("bank_name").unique().sort().alias("bank_names"),
            pl.col("bank_name").is_null().sum().alias("n_unjoined"),
        )
    )
    acc_names_per_id = (
        acc.filter(pl.col("bank_id").is_in(cb))
        .group_by("bank_id")
        .agg(pl.col("bank_name").n_unique().alias("n_names_in_accounts_file"))
        .rename({"bank_id": "bank_norm"})
    )

    alerts = None
    if alerts_path.is_file():
        alerts = (
            pl.scan_parquet(alerts_path)
            .filter(pl.col("window_start") < HARD_LIMIT)
            .filter(pl.col("period").is_in(["TRAIN", "VALIDATION"]))
            .select("account_key", "window_start", "period", "triggered_rules")
            .collect()
        )
        assert alerts.height == 0 or alerts["window_start"].max() < HARD_LIMIT
        alerts = alerts.with_columns(
            pl.col("account_key").str.split_exact("|", 1).struct.field("field_0").alias("bank"),
            pl.col("account_key").str.split_exact("|", 1).struct.field("field_1").alias("account"),
        ).with_columns(norm_bank(pl.col("bank")).alias("bank_norm"))
        al = (
            alerts.filter(pl.col("bank_norm").is_in(cb))
            .group_by("bank_norm", "bank")
            .agg(
                pl.len().alias("n_alerts"),
                (pl.col("period") == "TRAIN").sum().alias("n_alerts_train"),
                (pl.col("period") == "VALIDATION").sum().alias("n_alerts_validation"),
            )
        )
    else:
        print(f"   (alerts file {alerts_path} not found: alert columns skipped)")
        al = pl.DataFrame(schema={"bank_norm": pl.String, "bank": pl.String, "n_alerts": pl.UInt32})

    tab = (
        per_bank_acc.join(txn_from, on=["bank_norm", "bank"], how="left")
        .join(txn_to, on=["bank_norm", "bank"], how="left")
        .join(names, on=["bank_norm", "bank"], how="left")
        .join(al, on=["bank_norm", "bank"], how="left")
        .join(
            both.group_by("bank_norm").agg(
                (pl.col("n_spellings") > 1).sum().alias("n_accounts_under_both")
            ),
            on="bank_norm",
        )
        .join(acc_names_per_id, on="bank_norm", how="left")
        .fill_null(0)
        .sort("bank_norm", "bank")
    )
    print("\nC. per spelling of each collided bank ID (TRAIN/VALIDATION):")
    print(tab)
    print("\n   summary:")
    print(
        f"   collided IDs whose spellings share >= 1 account number: "
        f"{tab.filter(pl.col('n_accounts_under_both') > 0)['bank_norm'].n_unique()}"
    )
    print(
        f"   account numbers under both spellings (total): {int((both['n_spellings'] > 1).sum())}"
    )
    nn = tab.group_by("bank_norm").agg(
        pl.col("bank_names").explode(empty_as_null=True).n_unique().alias("k")
    )
    print(
        f"   collided IDs whose two spellings map to DIFFERENT bank names: "
        f"{nn.filter(pl.col('k') > 1).height} of {nn.height}"
    )
    print(f"   txns touching a collided bank: {int(tab['n_out'].sum() + tab['n_in'].sum()):,} legs")
    if "n_alerts" in tab.columns:
        print(
            f"   TRAIN/VALIDATION alerts on accounts of a collided bank: "
            f"{int(tab['n_alerts'].sum()):,}"
        )

    # D. global consequences of the D2 key
    x_self = t.filter(
        (pl.col("from_norm") == pl.col("to_norm"))
        & (pl.col("from_account") == pl.col("to_account"))
        & (pl.col("from_bank") != pl.col("to_bank"))
    ).height
    print(f"\nD1. cross-spelling self-transfers (kept as normal txns by the D2 key): {x_self}")
    fan_raw = (
        t.filter(
            (pl.col("from_bank") + "|" + pl.col("from_account"))
            != (pl.col("to_bank") + "|" + pl.col("to_account"))
        )
        .group_by("day", "to_bank", "to_account")
        .agg(
            (pl.col("from_bank") + "|" + pl.col("from_account")).n_unique().alias("raw"),
            (pl.col("from_norm") + "|" + pl.col("from_account")).n_unique().alias("norm"),
        )
    )
    print(
        f"D2. receiver-days whose distinct-sender count shrinks after normalisation (R04): "
        f"{fan_raw.filter(pl.col('raw') != pl.col('norm')).height}"
    )
    if alerts is not None:
        dbl = (
            alerts.group_by("bank_norm", "account", "window_start")
            .agg(pl.col("bank").n_unique().alias("k"))
            .filter(pl.col("k") > 1)
        )
        print(f"D3. account-days alerted under two spellings (double alerts): {dbl.height}")


if __name__ == "__main__":
    main()
