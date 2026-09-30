"""P2 discovery: what do bank names in *_accounts.csv actually look like? Label-free aggregates.

The first real run showed that many bank names do not follow '<Location> Bank #<n>'
(e.g. 'Hearthstone Bancorp'). Before changing the parser, print the real formats.

Usage (repo root):  uv run python -m scripts.p2.discover_banks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

pl.Config.set_tbl_rows(60)
pl.Config.set_fmt_str_lengths(60)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--accounts", type=Path, default=Path("data/interim/HI-Medium_accounts.parquet")
    )
    args = ap.parse_args(argv)
    a = pl.read_parquet(args.accounts, columns=["bank_id", "bank_name"])
    print(f"account rows: {a.height:,}")
    print(f"distinct bank_id: {a['bank_id'].n_unique():,}")
    print(f"distinct bank_name: {a['bank_name'].n_unique():,}")
    per_id = a.group_by("bank_id").agg(pl.col("bank_name").n_unique().alias("k"))
    print(f"bank_ids with more than one name: {(per_id['k'] > 1).sum():,}")

    a = a.with_columns(
        pl.col("bank_name").str.contains(r"#\d+\s*$").alias("has_number"),
        pl.col("bank_name").str.contains(r"Bank #\d+\s*$").alias("bank_hash_form"),
        pl.col("bank_name").str.split(" ").list.last().alias("last_word"),
    )
    print("\nform counts (accounts):")
    print(a.group_by("has_number", "bank_hash_form").len().sort("len", descending=True))

    print("\n'<X> Bank #n' form: top 60 X by accounts")
    x = a.filter(pl.col("bank_hash_form")).with_columns(
        pl.col("bank_name").str.extract(r"^(.*?)\s*Bank #\d+\s*$", 1).alias("prefix")
    )
    print(x.group_by("prefix").len().sort(["len", "prefix"], descending=[True, False]).head(60))
    print(f"distinct prefixes: {x['prefix'].n_unique():,}")

    other = a.filter(~pl.col("bank_hash_form"))
    print(f"\nother forms: {other.height:,} accounts, {other['bank_name'].n_unique():,} names")
    print("top 60 names by accounts:")
    print(
        other.group_by("bank_name")
        .len()
        .sort(["len", "bank_name"], descending=[True, False])
        .head(60)
    )
    print("last word of other names (top 30):")
    print(
        other.group_by("last_word")
        .len()
        .sort(["len", "last_word"], descending=[True, False])
        .head(30)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
