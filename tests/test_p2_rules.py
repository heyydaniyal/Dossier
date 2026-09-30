"""P2 task 3: rule statistics are point-in-time correct, match a slow reference, and fire right.

Test required by the phase: rule computation for any alert uses no transaction at or after as_of.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.alerts import rules
from src.contracts.models import is_visible
from src.data import fx
from src.data.periods import load_split

ROOT = Path(__file__).resolve().parents[1]
CFG = rules.load_rules()
SPLIT = load_split()
USD = {"US Dollar": 1.0, "Euro": 1.1, "Bitcoin": 20000.0}
WS = datetime(2022, 9, 5, tzinfo=UTC)
AS_OF = SPLIT.as_of(WS)


def _frame(rows: list[dict]) -> pl.LazyFrame:
    df = pl.DataFrame(
        rows,
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "from_bank": pl.String,
            "from_account": pl.String,
            "to_bank": pl.String,
            "to_account": pl.String,
            "amount_received": pl.Float64,
            "receiving_currency": pl.String,
            "amount_paid": pl.Float64,
            "payment_currency": pl.String,
            "payment_format": pl.String,
        },
        orient="row",
    )
    df = df.with_columns(
        pl.concat_str([pl.col("from_bank"), pl.lit("|"), pl.col("from_account")]).alias("from_key"),
        pl.concat_str([pl.col("to_bank"), pl.lit("|"), pl.col("to_account")]).alias("to_key"),
    )
    return fx.with_usd(df.lazy(), USD)


def _tx(ts, s, r, amt, fmt="ACH", pc="US Dollar", rc="US Dollar", recv=None):
    recv = amt * USD[pc] / USD[rc] if recv is None else recv
    return {
        "timestamp": ts,
        "from_bank": s.split("|")[0],
        "from_account": s.split("|")[1],
        "to_bank": r.split("|")[0],
        "to_account": r.split("|")[1],
        "amount_received": recv,
        "receiving_currency": rc,
        "amount_paid": amt,
        "payment_currency": pc,
        "payment_format": fmt,
    }


def _stats(rows):
    return rules.day_stats(_frame(rows), WS, SPLIT, CFG)


def _one(st: pl.DataFrame, key: str) -> dict:
    r = st.filter(pl.col("account_key") == key)
    return r.row(0, named=True) if r.height else {}


A, B, C = "010|A1", "020|B1", "030|C1"


# ---------------------------------------------------------------- point in time


def test_nothing_at_or_after_as_of_is_visible():
    base = [_tx(WS + timedelta(hours=3), A, B, 500.0)]
    ref = _stats(base)
    polluted = base + [
        _tx(AS_OF, A, C, 1e9),  # tie with as_of: excluded (frozen tie rule)
        _tx(AS_OF + timedelta(minutes=1), A, C, 1e9),
        _tx(AS_OF + timedelta(days=5), C, A, 1e9, "Cash"),
        _tx(WS - timedelta(minutes=1), A, C, 1e9),  # before lookback start: excluded
    ]
    got = _stats(polluted)
    assert got.drop("as_of", "window_start").equals(ref.drop("as_of", "window_start"))


def test_boundary_minutes_inside_the_window_are_visible():
    rows = [_tx(WS, A, B, 100.0), _tx(AS_OF - timedelta(minutes=1), A, C, 200.0)]
    r = _one(_stats(rows), A)
    assert r["n_txn"] == 2 and r["volume_usd"] == pytest.approx(300.0)


def _reference(rows: list[dict], key: str) -> dict:
    """Slow, obviously-correct per-alert computation using the contract's is_visible."""
    lb = SPLIT.lookback_start(AS_OF)
    legs = []
    for t in rows:
        if not (is_visible(t["timestamp"], AS_OF) and t["timestamp"] >= lb):
            continue
        s = f"{t['from_bank']}|{t['from_account']}"
        r = f"{t['to_bank']}|{t['to_account']}"
        if s == r:
            continue
        cross = t["payment_currency"] != t["receiving_currency"]
        hr = t["payment_format"] in ("Cash", "Bitcoin")
        if s == key:
            usd = t["amount_paid"] * USD[t["payment_currency"]]
            legs.append(("out", r, usd, t["amount_paid"], cross, hr, t["timestamp"]))
        if r == key:
            usd = t["amount_received"] * USD[t["receiving_currency"]]
            legs.append(("in", s, usd, t["amount_received"], cross, hr, t["timestamp"]))
    if not legs:
        return {}
    ins = [x for x in legs if x[0] == "in"]
    outs = [x for x in legs if x[0] == "out"]
    in_usd, out_usd = sum(x[2] for x in ins), sum(x[2] for x in outs)
    pt = 0.0
    if ins and outs and 0.8 <= out_usd / in_usd <= 1.25:
        if max(x[6] for x in outs) > min(x[6] for x in ins):
            pt = min(in_usd, out_usd)
    return {
        "n_txn": len(legs),
        "max_txn_usd": max(x[2] for x in legs),
        "volume_usd": in_usd + out_usd,
        "n_near_threshold": sum(9000 <= x[2] < 10000 for x in legs),
        "n_distinct_senders": len({x[1] for x in ins}),
        "n_distinct_receivers": len({x[1] for x in outs}),
        "pass_through_usd": pt,
        "high_risk_channel_usd": sum(x[2] for x in legs if x[5]),
        "n_cross_currency": sum(x[4] for x in legs),
        "n_round_amounts": sum(
            x[3] >= 1000 and abs(round(x[3] / 1000) * 1000 - x[3]) < 1e-6 for x in legs
        ),
        "in_usd": in_usd,
        "out_usd": out_usd,
    }


def test_batch_equals_reference_on_random_data():
    rng = np.random.default_rng(42)
    accts = [f"{b:03d}|X{i}" for i, b in enumerate(rng.integers(1, 5, 25))]
    rows = []
    for _ in range(600):
        s, r = rng.choice(len(accts), 2)
        ts = WS + timedelta(minutes=int(rng.integers(-600, 24 * 60 + 600)))  # spills both sides
        pc = str(rng.choice(["US Dollar", "Euro", "Bitcoin"], p=[0.7, 0.2, 0.1]))
        rc = pc if rng.random() < 0.8 else "US Dollar"
        amt = float(rng.choice([1000.0, 2000.0, 9500.0, round(float(rng.lognormal(7, 1.5)), 2)]))
        fmt = str(rng.choice(["ACH", "Cash", "Wire", "Cheque", "Bitcoin"]))
        rows.append(_tx(ts, accts[s], accts[r], amt, fmt, pc, rc))
    st = _stats(rows)
    checked = 0
    for k in accts:
        ref, got = _reference(rows, k), _one(st, k)
        assert bool(ref) == bool(got), k
        for c, v in ref.items():
            assert got[c] == pytest.approx(v, rel=1e-9, abs=1e-9), (k, c)
        checked += bool(ref)
    assert checked > 15


# ---------------------------------------------------------------- statistic semantics


def test_self_transfers_are_ignored_and_both_roles_count():
    rows = [
        _tx(WS + timedelta(hours=1), A, A, 50000.0, "Reinvestment"),
        _tx(WS + timedelta(hours=2), B, A, 100.0),
        _tx(WS + timedelta(hours=3), A, C, 70.0),
    ]
    r = _one(_stats(rows), A)
    assert r["n_txn"] == 2 and r["max_txn_usd"] == 100.0
    assert r["n_distinct_senders"] == 1 and r["n_distinct_receivers"] == 1


def test_structuring_band_edges_and_round_amounts_in_original_currency():
    rows = [
        _tx(WS + timedelta(hours=1), A, B, 9000.0),  # in band (inclusive)
        _tx(WS + timedelta(hours=2), A, B, 9999.99),  # in band
        _tx(WS + timedelta(hours=3), A, B, 10000.0),  # out of band (exclusive)
        _tx(WS + timedelta(hours=4), A, C, 1000.0, pc="Euro", rc="Euro"),  # 1,000 EUR is round
    ]
    r = _one(_stats(rows), A)
    assert r["n_near_threshold"] == 2
    assert r["n_round_amounts"] == 3  # 9000, 10000 (USD) and 1000 (EUR); 9999.99 is not
    assert r["max_txn_usd"] == pytest.approx(10000.0)


def test_rapid_in_out_needs_ratio_and_order():
    in_then_out = [
        _tx(WS + timedelta(hours=1), B, A, 10000.0),
        _tx(WS + timedelta(hours=5), A, C, 9500.0),
    ]
    out_then_in = [
        _tx(WS + timedelta(hours=5), B, A, 10000.0),
        _tx(WS + timedelta(hours=1), A, C, 9500.0),
    ]
    ratio_off = [
        _tx(WS + timedelta(hours=1), B, A, 10000.0),
        _tx(WS + timedelta(hours=5), A, C, 5000.0),
    ]
    assert _one(_stats(in_then_out), A)["pass_through_usd"] == pytest.approx(9500.0)
    assert _one(_stats(out_then_in), A)["pass_through_usd"] == 0.0
    assert _one(_stats(ratio_off), A)["pass_through_usd"] == 0.0


# ---------------------------------------------------------------- firing


def _thr(**over) -> dict:
    t = {"rules": {r: {"active": True, "threshold": 1e18} for r in rules.rule_ids(CFG)}}
    t["rules"]["R02_PEER_VOLUME_OUTLIER"] = {
        "active": True,
        "by_peer": {"Corporation": 1e18},
        "default": 1e18,
    }
    for k, v in over.items():
        t["rules"][k] = v
    return t


def test_fire_threshold_semantics_and_order():
    st = pl.DataFrame(
        {"account_key": ["a", "b", "c"], "peer_group": ["Corporation", "Individual", "Corporation"]}
        | {c: [0, 0, 0] for c in rules.ALL_STATS}
    ).with_columns(
        pl.Series("n_distinct_senders", [5, 4, 9]),
        pl.Series("volume_usd", [50.0, 50.0, 10.0]),
        pl.Series("max_txn_usd", [0.0, 0.0, 99.0]),
    )
    thr = _thr(
        R04_FAN_IN={"active": True, "threshold": 5.0},
        R02_PEER_VOLUME_OUTLIER={
            "active": True,
            "by_peer": {"Corporation": 100.0},
            "default": 40.0,
        },
        R01_LARGE_SINGLE_TXN={"active": False, "threshold": 1.0},
    )
    f = rules.fire(st, thr, CFG)
    trig = dict(zip(f["account_key"], f["triggered_rules"].to_list(), strict=True))
    assert trig["a"] == ["R04_FAN_IN"]  # >= threshold fires; Corporation peer threshold 100 not met
    assert trig["b"] == ["R02_PEER_VOLUME_OUTLIER"]  # Individual -> default threshold 40
    assert trig["c"] == ["R04_FAN_IN"]  # R01 inactive
    assert "fired_R01_LARGE_SINGLE_TXN" not in f.columns


def test_rule_set_size_and_one_stat_per_rule():
    assert 6 <= len(CFG["rules"]) <= 10
    assert sorted(r["stat"] for r in CFG["rules"].values()) == sorted(rules.RULE_STATS)


# ---------------------------------------------------------------- no labels anywhere near the rules

TRUTH_TOKENS = ("is_laundering", "Is Laundering", "typolog", "attempt_id", "src.eval", "scripts.")


@pytest.mark.parametrize(
    "py",
    sorted((ROOT / "src" / "alerts").glob("*.py")) + sorted((ROOT / "src" / "data").glob("*.py")),
    ids=lambda p: p.name,
)
def test_rule_and_data_code_never_names_ground_truth(py: Path):
    text = py.read_text(encoding="utf-8")
    ast.parse(text)
    for tok in TRUTH_TOKENS:
        assert tok not in text, f"{py.name} mentions {tok!r}"


def test_runtime_loader_never_returns_the_label(tmp_path: Path):
    from src.data.transactions import RUNTIME_COLUMNS, scan_transactions

    p = tmp_path / "t.parquet"
    pl.DataFrame(
        {
            c: ["x"]
            for c in RUNTIME_COLUMNS
            if c not in ("timestamp", "amount_received", "amount_paid")
        }
        | {"timestamp": [datetime(2022, 9, 5)], "amount_received": [1.0], "amount_paid": [1.0]}
        | {"is_" + "laundering": [1], "row_id": [0]}
    ).write_parquet(p)
    cols = scan_transactions(p).collect_schema().names()
    assert "is_" + "laundering" not in cols and "row_id" not in cols
    assert scan_transactions(p).collect()["timestamp"].dtype == pl.Datetime("us", "UTC")
