"""Facts of the real-data P4 step-0 runs (docs/p4/p4_diagnosis.json, docs/p4/p4_leg_check.json,
written on Dani's laptop). Each file's tests are skipped until it is committed. Then:
the run read TRAIN only, used the P3-recorded TRAIN matrix and the configs as committed now (a
config edited after the run fails here), and every number in docs/p4/P4_MODEL.md equals the JSON.
"""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from src.data.periods import ROOT

DOC = (ROOT / "docs" / "p4" / "P4_MODEL.md").read_text(encoding="utf-8")
DIAG = ROOT / "docs" / "p4" / "p4_diagnosis.json"
LEG = ROOT / "docs" / "p4" / "p4_leg_check.json"
P3 = json.loads((ROOT / "docs" / "p3" / "p3_results.json").read_text(encoding="utf-8"))
TRAIN_SHA = P3["features"]["periods"]["TRAIN"]["parquet_sha256"]


def _lf_sha(p) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _f3(x: float) -> str:
    return f"{x:.3f}"


def _in_doc(*xs: float) -> None:
    for x in xs:
        assert _f3(x) in DOC, f"{_f3(x)} not stated in P4_MODEL.md"


def _signed(x: float) -> str:
    return ("+" if x >= 0 else "−") + f"{abs(x):.3f}"


# ------------------------------------------------------------------ step 0 (diagnosis)

diag = pytest.mark.skipif(not DIAG.exists(), reason="p4_diagnosis.json not committed")


@pytest.fixture(scope="module")
def d0() -> dict:
    return json.loads(DIAG.read_text(encoding="utf-8"))


@diag
def test_diagnosis_inputs(d0):
    i = d0["inputs"]
    assert i["periods_read"] == ["TRAIN"]
    assert i["train_parquet_sha256"] == TRAIN_SHA
    assert i["config_sha256"] == _lf_sha(ROOT / "configs" / "p4_diagnosis.yaml")
    assert i["n_features"] == {"rule": 19, "all": 61}


@diag
def test_diagnosis_verdicts_and_doc_numbers(d0):
    for k in ("rule", "all"):
        r = d0["recognition"][k]
        assert r["verdict"] == "mixed", k
        assert r["control_shortfall_share"] == 0.0
        assert (r["n_seen_pos"], r["n_unseen_pos"]) == (40, 328)
        p = r["pooled"]
        _in_doc(p["full"], p["control"], p["disjoint"], p["label_memory"], p["unseen_full"])
        lo, hi = r["ci"]["control_minus_disjoint"]
        assert f"CI {_f3(lo)}–{_f3(hi)}" in DOC, k
        # share of the seen gap lost without the account's own rows (stated as a percentage)
        lost = (p["control"] - p["disjoint"]) / (p["control"] - p["unseen_full"])
        assert f"{round(100 * lost)}%" in DOC, (k, lost)
    u = d0["recognition"]["all"]["ci"]["unseen_full"]
    assert f"[{_f3(u[0])}, {_f3(u[1])}]" in DOC
    assert d0["univariate"]["n_ci_lower_above_0_55"] == 0
    assert d0["univariate"]["top"][0]["feature"] == "amt_std_usd"
    _in_doc(d0["univariate"]["top"][0]["auc_unseen"])
    s = d0["strata"]["strata"]
    for k in ("share_[0.00,0.10)", "share_[0.10,0.50)", "share_[0.50,inf)", "rule_fan_in",
              "unattributed"):  # fmt: skip
        _in_doc(s[k]["auc_unseen"])
    assert d0["strata"]["total_unseen_pos"] == 328


# ------------------------------------------------------------------ step 0b (payment check)

leg = pytest.mark.skipif(not LEG.exists(), reason="p4_leg_check.json not committed")


@pytest.fixture(scope="module")
def d1() -> dict:
    return json.loads(LEG.read_text(encoding="utf-8"))


@leg
def test_leg_check_inputs(d1):
    i = d1["inputs"]
    assert i["periods_read"] == ["TRAIN"]
    assert i["train_parquet_sha256"] == TRAIN_SHA
    assert i["config_sha256"] == _lf_sha(ROOT / "configs" / "p4_leg_check.yaml")
    assert i["protocol_sha256"] == _lf_sha(ROOT / "configs" / "p4_protocol.yaml")
    assert i["lightgbm_version"] == "4.7.0"


@leg
def test_leg_check_verdicts_and_doc_numbers(d1):
    a, b = d1["primary"], d1["without_payment_format"]
    assert (a["verdict"], b["verdict"]) == ("inconclusive", "no_signal")
    for r in (a, b):
        assert r["n_unseen_pos"] == 328
        _in_doc(r["auc_payment_max"], r["auc_baseline"])
        assert _signed(r["gain"]) in DOC
        lo, hi = r["gain_ci"]
        assert f"[{_signed(lo)}, {_signed(hi)}]" in DOC
    w = d1["within_alert"]
    _in_doc(w["top_payment_hit_rate"], w["chance_rate"], w["mean_within_alert_auc"])
    assert str(w["n_alerts"]) in DOC
    for x in d1["payment_level"]:
        _in_doc(x["auc_payment_level_unseen"])
    for c in d1["counts"]:
        for k in ("fit_legs_sampled", "eval_legs"):
            assert f"{c[k]:,}" in DOC, (c["fold"], k)
    n_eval = sum(c["eval_alerts"] for c in d1["counts"])
    assert f"{n_eval:,}" in DOC


def test_doc_records_the_decisions():
    for k in ("P4-1", "P4-2", "P4-3", "P4-4", "P4-5", "P4-6", "P4-7", "P4-8", "P4-9"):
        assert re.search(rf"\| {k} \|", DOC), k


RUN1 = ROOT / "docs" / "p4" / "p4_cv_run1_format_candidates.json"


@pytest.mark.skipif(not RUN1.exists(), reason="first CV run not kept")
def test_first_cv_run_supports_decision_p4_9():
    d = json.loads(RUN1.read_text(encoding="utf-8"))
    by = {r["id"]: r for r in d["configs"] + d["sensitivity"]}
    w, wo = by["lightgbm_l7_m50_none"], by["lightgbm_l7_m50_none__no_format"]
    assert w["feature_set"] == "all" and wo["feature_set"] == "no_format"
    _in_doc(w["cv_pr_auc"], wo["cv_pr_auc"], w["cv_pr_auc_seen"], wo["cv_pr_auc_seen"])
    assert d["inputs"]["periods_read"] == ["TRAIN"]
