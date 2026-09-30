"""P2 facts on the REAL data (HI-Medium): every claim in docs/P2_ALERT_LAYER.md is checked against
docs/p2/p2_results.json, produced by `uv run python -m scripts.p2.run_p2` on Dani's laptop.

Skipped until that file exists. Once it exists, a failure here means a P2 gate is not met.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.alerts.build import ALERT_COLUMNS
from src.data.kyc import KYC_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
RES_PATH = ROOT / "docs" / "p2" / "p2_results.json"
RES = json.loads(RES_PATH.read_text(encoding="utf-8")) if RES_PATH.is_file() else {}
# skipped until the full run (calibrate + build) has produced the results file
pytestmark = pytest.mark.skipif("build" not in RES, reason="P2 build results not produced yet")
RULES = yaml.safe_load((ROOT / "configs" / "p2_rules.yaml").read_text(encoding="utf-8"))
KYC = yaml.safe_load((ROOT / "configs" / "p2_kyc.yaml").read_text(encoding="utf-8"))
DISP = yaml.safe_load((ROOT / "configs" / "p2_dispositions.yaml").read_text(encoding="utf-8"))


def test_results_are_for_the_primary_variant_and_complete():
    assert RES["variant"] == "HI-Medium"
    for k in ("stats", "calibration", "build", "stores", "split"):
        assert k in RES, k


def test_thresholds_file_is_the_one_used():
    f = ROOT / "configs" / "p2_rule_thresholds.yaml"
    assert (
        hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        == RES["build"]["thresholds_sha256_before_any_test_count"]
    )
    thr = yaml.safe_load(f.read_text(encoding="utf-8"))
    assert thr["calibrated_on"]["period"] == "TRAIN"


def test_alert_precision_is_realistic_and_recall_below_one():
    lo, hi = RULES["calibration"]["precision_band"]
    tr = RES["build"]["rule_metrics"]["TRAIN"]
    assert lo <= tr["precision"] <= hi
    assert tr["recall"] < RULES["calibration"]["max_layer_recall"]
    for per in ("VALIDATION", "CALIBRATION"):  # reported for every pre-TEST period
        assert RES["build"]["rule_metrics"][per]["precision"] is not None
    assert set(RES["build"]["rule_metrics"]) == {"TRAIN", "VALIDATION", "CALIBRATION"}


def test_no_rule_is_near_perfect_on_train():
    ts = RULES["calibration"]["too_strong"]
    for r, m in RES["build"]["rule_metrics"]["TRAIN"]["per_rule"].items():
        assert not (
            (m["precision"] or 0) > ts["precision_above"]
            and (m["recall"] or 0) > ts["recall_above"]
        ), r
    assert len(RES["build"]["rule_metrics"]["TRAIN"]["per_rule"]) >= 6


SPLIT = yaml.safe_load((ROOT / "configs" / "p2_split.yaml").read_text(encoding="utf-8"))


def test_feasibility_gates_pass_or_are_documented_deviations():
    f = RES["build"]["feasibility"]
    accepted = {d["gate"]: d for d in SPLIT["feasibility"].get("accepted_deviations") or []}
    failing = [g for g, ok in f["gates"].items() if not ok]
    for g in failing:
        assert g in accepted, f"gate {g} fails and is not a documented deviation"
        per = g.removesuffix("_min_positive")
        # the deviation must record exactly the measured value (no silent drift)
        assert f["counts"][per]["n_positive_alerts"] == accepted[g]["measured_positive_alerts"]
        assert accepted[g]["decided_by"] and accepted[g]["reason"]
    assert set(f["counts"]["TEST"]) == {"n_alerts", "n_positive_alerts", "n_days"}  # counts only


def test_kyc_leakage_within_pre_declared_ceilings():
    lk = RES["build"]["kyc"]["leakage"]
    assert lk["status"] == "measured"
    c = KYC["ceilings"]
    prev = lk["val_prevalence"]
    assert lk["pr_auc"]["kyc_all"] <= c["kyc_only_pr_auc_max_multiple_of_prev"] * prev + 1e-12
    for k, v in lk["checks"].items():
        assert v["pass"], (k, v)


def test_kyc_v2_planting_rates_recorded():
    k = RES["build"]["kyc"]
    assert k["version"] == 2
    p = k["planting"]
    assert p["method"] == KYC["planting"]["method"] == "burn_in_behaviour"
    for c in ("realised_rate_laundering", "realised_rate_legitimate"):
        assert p[c] is not None and 0 < p[c] < 1, c
    assert "TEST" not in p["scope"]


def test_dispositions_pre_test_only_with_errors():
    d = RES["build"]["dispositions"]
    assert set(d["by_period"]) == set(DISP["periods"]) == {"TRAIN", "VALIDATION", "CALIBRATION"}
    assert 0 < 1 - d["overall"]["accuracy"] < 0.2
    assert (
        d["hard_H1_shapeless_laundering"]["false_negative_rate"]
        > d["easy_laundering"]["false_negative_rate"]
    )


def test_runtime_store_schemas():
    s = RES["stores"]
    assert s["runtime/alerts.parquet"]["columns"] == ALERT_COLUMNS
    assert s["runtime/kyc.parquet"]["columns"] == KYC_COLUMNS
    assert s["runtime/dispositions.parquet"]["columns"] == [
        "alert_id",
        "account_key",
        "disposition",
        "closed_at",
    ]
