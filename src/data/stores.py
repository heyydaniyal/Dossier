"""The ONE way runtime code (rules, features, tools, agents, app) reaches P2 data stores.

Review findings C-2 / M-1 (P3 task 0): runtime code must not be able to read the evaluation store
(labels, laundering legs, planting, agent split) or the devtools store (AGENT-DEV ids, whose
membership is truth-derived). This module only knows the runtime store and refuses anything else;
tests/test_eval_isolation.py bans eval/devtools path literals in runtime code, so going around it
is caught by the test suite.

Runtime-safe: no ground truth is read here.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from src.data.periods import ROOT

DEFAULT_STORE_ROOT = ROOT / "data" / "p2"
# Allow-list of the agent-side store (docs/P2_ALERT_LAYER.md §11)
RUNTIME_STORES = ("alerts.parquet", "kyc.parquet", "dispositions.parquet")


class StoreAccessError(PermissionError):
    """A runtime caller asked for something outside the runtime store."""


def runtime_path(name: str, variant: str = "HI-Medium", root: Path = DEFAULT_STORE_ROOT) -> Path:
    if name not in RUNTIME_STORES:
        raise StoreAccessError(f"{name!r} is not a runtime store (allowed: {RUNTIME_STORES})")
    p = Path(root) / variant / "runtime" / name
    if p.resolve().parent != (Path(root) / variant / "runtime").resolve():
        raise StoreAccessError(f"{name!r} resolves outside the runtime store")
    return p


def scan_runtime(name: str, variant: str = "HI-Medium", root: Path = DEFAULT_STORE_ROOT):
    return pl.scan_parquet(runtime_path(name, variant, root))


def dispositions_visible(
    as_of: datetime, variant: str = "HI-Medium", root: Path = DEFAULT_STORE_ROOT
) -> pl.LazyFrame:
    """Historical dispositions an investigation with this as_of may see: closed STRICTLY before
    as_of (frozen tie rule). Review m-2: closed_at can fall inside TEST (up to 168 h after a
    CALIBRATION alert), so every consumer (P3 graph features, P10 network, P12 memory) must use
    this loader, never the raw file."""
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    return scan_runtime("dispositions.parquet", variant, root).filter(pl.col("closed_at") < as_of)
