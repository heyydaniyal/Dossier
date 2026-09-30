"""Label-free transaction access for everything that is not the evaluation harness.

The runtime-side loader selects an explicit ALLOW-LIST of columns. The ground-truth column is
never named in this module (tests/test_p2_firewall.py scans the source for it), so no code that
imports from here can obtain it by accident.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml

from src.data.periods import ROOT

# Allow-list of transaction columns visible to rules, features, tools and the app.
RUNTIME_COLUMNS = [
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
DATA_SOURCES = ROOT / "configs" / "data_sources.yaml"


class DataError(RuntimeError):
    """An input no longer matches its frozen fingerprint. Never swallowed."""


def interim_path(interim_dir: Path, variant: str, kind: str) -> Path:
    return Path(interim_dir) / f"{variant}_{kind}.parquet"


def verify_interim(interim_dir: Path, variant: str, sources: Path = DATA_SOURCES) -> None:
    """The P1 Parquet must come from the frozen raw files (sidecar sha256 == frozen sha256)."""
    frozen = yaml.safe_load(sources.read_text(encoding="utf-8"))["files"]
    raw = {"trans": "Trans.csv", "accounts": "accounts.csv", "patterns": "Patterns.txt"}
    for kind, suffix in raw.items():
        p = interim_path(interim_dir, variant, kind)
        meta = p.with_suffix(p.suffix + ".meta.json")
        if not (p.is_file() and meta.is_file()):
            raise DataError(f"missing {p} or its sidecar; run the P1 audit conversion first")
        m = json.loads(meta.read_text(encoding="utf-8"))
        want = frozen[f"{variant}_{suffix}"]["sha256"]
        if m.get("source_sha256") != want:
            raise DataError(f"{p.name} was built from a different raw file than the frozen one")


def account_key(bank: pl.Expr, account: pl.Expr) -> pl.Expr:
    """D2: 'bank|account', bank string exactly as written in the transactions (zeros kept)."""
    return pl.concat_str([bank, pl.lit("|"), account])


def scan_transactions(path: Path) -> pl.LazyFrame:
    """Transactions WITHOUT the ground-truth column; timestamps as UTC (D3); account keys (D2)."""
    return (
        pl.scan_parquet(path)
        .select(RUNTIME_COLUMNS)
        .with_columns(
            pl.col("timestamp").dt.replace_time_zone("UTC"),
            account_key(pl.col("from_bank"), pl.col("from_account")).alias("from_key"),
            account_key(pl.col("to_bank"), pl.col("to_account")).alias("to_key"),
        )
    )


def visible(lf: pl.LazyFrame, as_of: datetime, lookback_start: datetime) -> pl.LazyFrame:
    """Point-in-time filter (frozen tie rule): lookback_start <= ts < as_of (ties excluded)."""
    if as_of.tzinfo is None or lookback_start.tzinfo is None:
        raise ValueError("as_of and lookback_start must be timezone-aware")
    if not lookback_start < as_of:
        raise ValueError("lookback_start must be before as_of")
    return lf.filter((pl.col("timestamp") >= lookback_start) & (pl.col("timestamp") < as_of))
