"""End-to-end P2 on the fixture: labels, split, stores, firewall, regeneration.

Tests required by the phase covered here:
  - agent-side stores contain no label or typology fields
  - regeneration with the same seed is byte-identical
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import polars as pl

from src.alerts.build import ALERT_COLUMNS, to_alert
from src.contracts.models import FORBIDDEN_FIELDS, HistoricalDisposition
from src.data import kyc as kycmod
from src.data.kyc import KYC_COLUMNS
from src.eval.alert_labels import EVAL_LABEL_COLUMNS, to_alert_label
from tests.p2_fixture import L, N, key

D = lambda day: datetime(2022, 9, day, tzinfo=UTC)  # noqa: E731

# exact allow-lists of the agent-side (runtime) stores
RUNTIME_SCHEMAS = {
    "alerts.parquet": ALERT_COLUMNS,
    "kyc.parquet": KYC_COLUMNS,
    "dispositions.parquet": ["alert_id", "account_key", "disposition", "closed_at"],
    "agent_dev_alert_ids.parquet": ["alert_id"],
}
TRUTHY = {
    "typology",
    "typologies",
    "attempt_id",
    "attempt_ids",
    "has_unattributed",
    "planted",
    "laundering_account",
    "n_laundering_txns",
    "is_pos",
    "agent_group",
    "component",
}


def _labels(r) -> pl.DataFrame:
    a = pl.read_parquet(r["base"] / "runtime" / "alerts.parquet")
    lab = pl.read_parquet(r["base"] / "eval" / "alert_labels.parquet")
    return a.join(lab, on="alert_id")


def _alert(df: pl.DataFrame, k: str, day: int) -> dict | None:
    m = df.filter((pl.col("account_key") == k) & (pl.col("window_start") == D(day)))
    return m.row(0, named=True) if m.height else None


# ---------------------------------------------------------------- stores and firewall


def test_runtime_stores_have_exact_allow_listed_columns(p2_run):
    rt = p2_run["base"] / "runtime"
    assert sorted(p.name for p in rt.glob("*.parquet")) == sorted(RUNTIME_SCHEMAS)
    for name, cols in RUNTIME_SCHEMAS.items():
        got = pl.read_parquet_schema(rt / name).names()
        assert got == cols, name
        low = {c.lower() for c in got}
        assert not (low & FORBIDDEN_FIELDS), name
        assert not (low & TRUTHY), name


def test_runtime_store_values_carry_no_truth(p2_run):
    """No string cell in the runtime stores names a typology or the label (defence in depth)."""
    tokens = (
        "FAN-IN",
        "FAN-OUT",
        "CYCLE",
        "GATHER",
        "SCATTER",
        "BIPARTITE",
        "STACK",
        "RANDOM",
        "UNATTRIBUTED",
    )
    for f in (p2_run["base"] / "runtime").glob("*.parquet"):
        df = pl.read_parquet(f)
        for c, dt in df.schema.items():
            if dt == pl.String:
                vals = set(df[c].drop_nulls().unique().to_list())
                assert not any(t in v for v in vals for t in tokens), (f.name, c)


def test_eval_store_holds_the_labels(p2_run):
    lab = pl.read_parquet(p2_run["base"] / "eval" / "alert_labels.parquet")
    assert lab.columns == EVAL_LABEL_COLUMNS
    for row in lab.iter_rows(named=True):
        to_alert_label(row)  # frozen AlertLabel contract


def test_every_alert_satisfies_the_frozen_contract(p2_run):
    a = pl.read_parquet(p2_run["base"] / "runtime" / "alerts.parquet")
    assert a.height > 0
    for row in a.iter_rows(named=True):
        al = to_alert(row)
        assert al.window_end < al.as_of
    assert a.group_by("account_key", "window_start").len()["len"].max() == 1
    assert set(a["period"].unique()) <= {"TRAIN", "VALIDATION", "CALIBRATION", "TEST"}
    assert a["alert_id"].str.contains(r"\|").sum() == 0  # opaque: no account id inside


def test_no_alert_in_burn_in_embargo_or_tail(p2_run):
    a = pl.read_parquet(p2_run["base"] / "runtime" / "alerts.parquet")
    days = {d.day for d in a["window_start"].dt.date().to_list()}
    assert not days & {1, 6, 8, 12, 17, 18}


# ---------------------------------------------------------------- labels (task 2)


def test_labels_follow_the_definition(p2_run):
    lab = pl.read_parquet(p2_run["base"] / "eval" / "positive_account_days.parquet")
    pos = {(r["account_key"], r["window_start"].day) for r in lab.iter_rows(named=True)}
    # FAN-IN 09-03: collector and all 8 senders are positive that day (both roles)
    assert {(key(L(i)), 3) for i in range(0, 9)} <= pos
    # CYCLE across CAL/embargo/TEST: each leg labels its own day only
    assert (key(L(20)), 11) in pos and (key(L(21)), 11) in pos
    assert (key(L(21)), 12) in pos and (key(L(22)), 12) in pos
    assert (key(L(22)), 13) in pos and (key(L(20)), 13) in pos
    assert (key(L(20)), 12) not in pos  # between its laundering days: not positive
    # tail laundering (09-17) is excluded (D1)
    assert not any(d == 17 for _, d in pos)
    # legitimate structuring is not a positive
    assert (key(N(5)), 5) not in pos
    row = lab.filter((pl.col("account_key") == key(L(40))) & (pl.col("window_start") == D(4))).row(
        0, named=True
    )
    assert row["typologies"] == [] and row["has_unattributed"] and row["n_laundering_txns"] == 2
    row = lab.filter((pl.col("account_key") == key(L(0))) & (pl.col("window_start") == D(3))).row(
        0, named=True
    )
    assert (
        row["typologies"] == ["FAN-IN"]
        and not row["has_unattributed"]
        and row["n_laundering_txns"] == 8
    )


def test_alert_labels_join(p2_run):
    j = _labels(p2_run)
    fan_in = _alert(j, key(L(0)), 3)
    assert (
        fan_in is not None
        and fan_in["is_true_positive"]
        and "R04_FAN_IN" in fan_in["triggered_rules"]
    )
    struct = _alert(j, key(N(5)), 5)
    assert struct is not None and not struct["is_true_positive"]
    assert "R03_STRUCTURING" in struct["triggered_rules"]
    assert j["is_true_positive"].sum() >= 1 and (~j["is_true_positive"]).sum() >= 1


# ---------------------------------------------------------------- AGENT-DEV / AGENT-TEST


def test_group_split_covers_test_only_and_is_disjoint(p2_run):
    a = pl.read_parquet(p2_run["base"] / "runtime" / "alerts.parquet")
    s = pl.read_parquet(p2_run["base"] / "eval" / "agent_split.parquet")
    test_ids = set(a.filter(pl.col("period") == "TEST")["alert_id"])
    assert set(s["alert_id"]) == test_ids
    assert s.group_by("account_key").agg(pl.col("agent_group").n_unique())["agent_group"].max() == 1
    dev = set(
        pl.read_parquet(p2_run["base"] / "runtime" / "agent_dev_alert_ids.parquet")["alert_id"]
    )
    assert dev == set(s.filter(pl.col("agent_group") == "AGENT-DEV")["alert_id"])


def test_group_split_links_pattern_instances_and_is_order_free():
    from src.eval.group_split import check_disjoint, split_test_alerts

    alerts = pl.DataFrame(
        {"alert_id": [f"a{i}" for i in range(6)], "account_key": ["x", "y", "z", "u", "v", "w"]}
    )
    legs = pl.DataFrame(
        {
            "account_key": ["x", "y", "z", "u", "q"],
            "window_start": [D(13)] * 5,
            "row_id": [1, 1, 2, 3, 3],
            "attempt_id": [7, 7, None, None, None],
            "typology": ["CYCLE", "CYCLE", None, None, None],
            "timestamp": [D(13)] * 5,
        },
        schema_overrides={"attempt_id": pl.Int32},
    )
    for frac in (0.01, 0.5, 0.99):
        s = split_test_alerts(alerts, legs, D(13), D(17), "k", frac)
        g = dict(zip(s["account_key"], s["agent_group"], strict=True))
        assert g["x"] == g["y"]  # same attempt -> same group
        check_disjoint(s, legs)
        s2 = split_test_alerts(alerts.reverse(), legs.reverse(), D(13), D(17), "k", frac)
        assert s2.equals(s)  # independent of row order


# ---------------------------------------------------------------- dispositions (task 9)


def test_dispositions_only_pre_test_and_after_as_of(p2_run):
    a = pl.read_parquet(p2_run["base"] / "runtime" / "alerts.parquet")
    d = pl.read_parquet(p2_run["base"] / "runtime" / "dispositions.parquet")
    j = d.join(a.select("alert_id", "period", "as_of"), on="alert_id", how="left")
    assert j["period"].null_count() == 0
    assert set(j["period"]) <= {"TRAIN", "VALIDATION", "CALIBRATION"}
    assert d.height == a.filter(pl.col("period") != "TEST").height
    assert (j["closed_at"] > j["as_of"]).all()
    for row in d.iter_rows(named=True):
        HistoricalDisposition(**row)


# ---------------------------------------------------------------- KYC


def test_kyc_covers_every_account_and_has_no_planting_trace(p2_run):
    k = pl.read_parquet(p2_run["base"] / "runtime" / "kyc.parquet")
    keys = pl.read_parquet(p2_run["base"] / "work" / "account_keys.parquet")
    assert set(k["account_key"]) == set(keys["account_key"])
    assert k["account_key"].n_unique() == k.height
    assert k.null_count().sum_horizontal().item() == 0
    assert set(k["bank_country"]) == {"United States", "Spain", "Japan", "Crypto"}
    assert set(k["bank_location"]) == {"United States", "Spain", "Japan", "Crytpo"}
    us = k.filter(pl.col("bank_country") == "United States")["country_risk"].unique().to_list()
    assert us == ["low"]
    crypto = k.filter(pl.col("bank_country") == "Crypto")["country_risk"].unique().to_list()
    assert crypto == [kycmod.load_kyc_config()["country_risk"]["crypto_tier"]]


# ---------------------------------------------------------------- results JSON: TEST exposure


def test_results_expose_test_only_as_counts(p2_run):
    doc = p2_run["doc"]
    rm = doc["build"]["rule_metrics"]
    assert set(rm) == {"TRAIN", "VALIDATION", "CALIBRATION"}
    assert "TEST" not in doc["calibration"]["train"]
    c = doc["build"]["feasibility"]["counts"]["TEST"]
    assert set(c) == {"n_alerts", "n_positive_alerts", "n_days"}
    assert "positive_account_days_per_period" in doc
    thr_sha = doc["build"]["thresholds_sha256_before_any_test_count"]
    assert (
        thr_sha
        == hashlib.sha256((p2_run["configs"] / "p2_rule_thresholds.yaml").read_bytes()).hexdigest()
    )


# ---------------------------------------------------------------- reproducibility


def _hashes(r) -> dict:
    files = [
        *r["base"].joinpath("runtime").glob("*.parquet"),
        *r["base"].joinpath("eval").glob("*.parquet"),
        *r["configs"].glob("*.yaml"),
    ]
    return {f"{f.parent.name}/{f.name}": hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def test_regeneration_is_byte_identical(p2_run, p2_run_again):
    h1, h2 = _hashes(p2_run), _hashes(p2_run_again)
    assert len(h1) == 11  # 4 runtime + 5 eval stores + 2 configs
    assert h1 == h2
    d1 = {k: v for k, v in p2_run["doc"].items() if k != "run_info"}
    d2 = {k: v for k, v in p2_run_again["doc"].items() if k != "run_info"}
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)
