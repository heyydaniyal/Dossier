"""Real account attributes from *_accounts.csv (P1 Parquet): entity type, bank location, country.

Join rule (D2): accounts file on (bank with leading zeros stripped, '' -> '0', account).
ID VALUES are never returned as attributes (D5); only descriptors derived from names are.

Bank-name formats, as PRINTED by scripts/p2/discover_banks.py on HI-Medium (2026-09-30):
  - '<X> Bank #<n>': 1,312,489 accounts, 33 distinct X. 32 are country names; one is
    'Crytpo' (the data's own spelling), i.e. crypto platforms with no jurisdiction.
  - any other name (no '#<n>'): 775,297 accounts, 468 names such as 'Savings Bank of Seattle',
    'First Bank of Miami', 'Hearthstone Bancorp': US-style named banks -> United States.
An '<X> Bank #<n>' name whose X is neither a declared country nor a declared crypto prefix is an
unexpected format and fails loudly (never silently mapped to the United States).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.data.transactions import DataError

_NUMBERED = r"^(.*?)\s*Bank\s*#\d+\s*$"
DOMESTIC = "United States"
CRYPTO = "Crypto"


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
    """X for '<X> Bank #<n>' names, null for named (domestic) banks."""
    return name.str.extract(_NUMBERED, 1).str.strip_chars()


def account_attributes(
    accounts_path: Path,
    keys: pl.DataFrame,
    foreign_countries: list[str],
    crypto_prefixes: list[str],
) -> pl.DataFrame:
    """One row per account_key in `keys` with entity_type, bank_location, bank_country.

    bank_location: X of '<X> Bank #<n>' (country or crypto prefix), or 'United States' for
    named banks. bank_country: the country, 'Crypto', or 'United States'.
    Fails loudly if any key is missing from the accounts file or a numbered prefix is unknown.
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
    if out["bank_name"].null_count():
        raise DataError("accounts without a bank name")
    foreign = sorted(set(foreign_countries))
    crypto = sorted(set(crypto_prefixes))
    out = out.with_columns(bank_location(pl.col("bank_name")).alias("_x"))
    unknown = out.filter(
        pl.col("_x").is_not_null() & ~pl.col("_x").is_in(foreign) & ~pl.col("_x").is_in(crypto)
    )
    if unknown.height:
        ex = sorted(unknown["_x"].unique().to_list())[:10]
        raise DataError(f"{unknown.height} accounts at '<X> Bank #n' banks with unknown X: {ex}")
    return out.select(
        "account_key",
        "entity_type",
        pl.col("_x").fill_null(pl.lit(DOMESTIC)).alias("bank_location"),
        pl.when(pl.col("_x").is_null())
        .then(pl.lit(DOMESTIC))
        .when(pl.col("_x").is_in(crypto))
        .then(pl.lit(CRYPTO))
        .otherwise(pl.col("_x"))
        .alias("bank_country"),
    )
