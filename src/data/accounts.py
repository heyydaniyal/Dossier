"""Real account attributes from *_accounts.csv (P1 Parquet): entity type, bank location, country.

Join rule (D2): accounts file on (bank with leading zeros stripped, '' -> '0', account).
ID VALUES are never returned as attributes (D5); only descriptors derived from names are.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.data.transactions import DataError

# Bank names look like "<Location> Bank #<n>" (DATA_CARD §9: foreign banks carry a country,
# US banks a city). Location = text before "#<n>", with a trailing " Bank" removed.
_BEFORE_NUMBER = r"^(.*?)\s*#\d+\s*$"
DOMESTIC = "United States"


def norm_bank(e: pl.Expr) -> pl.Expr:
    s = e.str.strip_chars_start("0")
    return pl.when(s == "").then(pl.lit("0")).otherwise(s)


def split_key(keys: pl.DataFrame) -> pl.DataFrame:
    """account_key 'bank|account' -> bank, account (split on the single '|')."""
    parts = pl.col("account_key").str.split_exact("|", 1)
    return keys.select(
        "account_key",
        parts.struct.field("field_0").alias("bank"),
        parts.struct.field("field_1").alias("account"),
    )


def bank_location(name: pl.Expr) -> pl.Expr:
    return name.str.extract(_BEFORE_NUMBER, 1).str.replace(r"\s*Bank$", "").str.strip_chars()


def account_attributes(
    accounts_path: Path, keys: pl.DataFrame, foreign_countries: list[str]
) -> pl.DataFrame:
    """One row per account_key in `keys` with entity_type, bank_location, bank_country.

    Fails loudly if any key is missing from the accounts file or any bank name does not parse
    (P1 verified 100% coverage; if that changes, it must not pass silently).
    """
    k = split_key(keys.select("account_key").unique()).with_columns(
        norm_bank(pl.col("bank")).alias("bank_norm")
    )
    acc = pl.read_parquet(
        accounts_path, columns=["bank_id", "account_number", "bank_name", "entity_type"]
    ).rename({"bank_id": "bank_norm", "account_number": "account"})
    out = k.join(acc, on=["bank_norm", "account"], how="left")
    if out["entity_type"].null_count():
        raise DataError(f"{out['entity_type'].null_count()} account keys not in the accounts file")
    foreign = sorted(set(foreign_countries))
    out = out.with_columns(bank_location(pl.col("bank_name")).alias("bank_location"))
    bad = out.filter(pl.col("bank_location").is_null() | (pl.col("bank_location") == ""))
    if bad.height:
        ex = bad["bank_name"].head(5).to_list()
        raise DataError(f"{bad.height} bank names do not parse as '<Location> Bank #<n>': {ex}")
    return out.select(
        "account_key",
        "entity_type",
        "bank_location",
        pl.when(pl.col("bank_location").is_in(foreign))
        .then(pl.col("bank_location"))
        .otherwise(pl.lit(DOMESTIC))
        .alias("bank_country"),
    )
