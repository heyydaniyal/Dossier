"""Facts of the real-data P3 run (docs/p3/p3_results.json, written by scripts/p3/run_p3.py on Dani's
laptop). Skipped until that file is committed; from then on every declared criterion is asserted,
and a config edited after the run makes this test fail (hashes are compared with the files now).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts.p3 import run_p3
from src.features import compute as fc
from src.features import registry as reg
from tests.test_p3_features import ROOT

RES = ROOT / "docs" / "p3" / "p3_results.json"
P2 = json.loads((ROOT / "docs" / "p2" / "p2_results.json").read_text(encoding="utf-8"))
COUNTS = P2["build"]["feasibility"]["counts"]
CFG = fc.load_features_config()

pytestmark = pytest.mark.skipif(
    not RES.exists(), reason="real-data P3 run pending (scripts/p3/run_p3.py on Dani's laptop)"
)


@pytest.fixture(scope="module")
def doc() -> dict:
    return json.loads(RES.read_text(encoding="utf-8"))


def _lf_sha(p) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_exit_gate_sections_present(doc):
    for s in ("inputs", "features", "check", "permutation", "guard"):
        assert s in doc, f"stage {s} has not been run"


def test_inputs_equal_the_files_now(doc):
    for name, path in run_p3.CONFIGS.items():
        assert doc["inputs"]["configs_sha256"][name] == _lf_sha(path), f"{name} changed after run"
    assert doc["inputs"]["feature_registry_sha256"] == reg.registry_hash(fc.default_context())


def test_feature_matrices_cover_every_alert_of_each_period(doc):
    f = doc["features"]
    for p in ("TRAIN", "VALIDATION", "CALIBRATION"):
        rec = f["periods"][p]
        assert rec["rows"] == COUNTS[p]["n_alerts"], p
        assert rec["integrity"]["rows_joined"] == rec["rows"]
        assert rec["integrity"]["rule_stat_mismatches"] == 0
        assert rec["integrity"]["triggered_rules_mismatches"] == 0
    for pair, m in f["boundaries"].items():
        assert m["gap_hours"] > m["embargo_hours"], pair
    if "TEST" in f["periods"]:
        assert f["periods"]["TEST"]["rows"] == COUNTS["TEST"]["n_alerts"]
        assert set(f["periods"]["TEST"]) == {"rows", "parquet_sha256", "seconds", "integrity_ok"}
        assert (
            f["regime_b_unseen_test_alerts"] == COUNTS["TEST_regime_B_unseen_accounts"]["n_alerts"]
        )
        log = (ROOT / "docs" / "holdout_exposure_log.md").read_text(encoding="utf-8")
        assert "run_p3 features --include-test" in log, "TEST read must be in the exposure log"


def test_point_in_time_check_on_real_data(doc):
    c = doc["check"]
    pc = CFG["pit_check"]
    assert c["passed"]
    assert c["n_sampled"] == pc["n_alerts_batch"] and c["n_pure"] == pc["n_alerts_pure"]
    assert c["deletion"] == c["mutation"] == c["stored_vs_recomputed"] == c["pure_vs_batch"] == 0


def test_shuffled_label_test(doc):
    r = doc["permutation"]
    pt = CFG["permutation_test"]
    assert r["n_permutations"] == pt["n_permutations"]
    assert r["features"] == len(reg.model_features(fc.default_context()))
    assert r["null_mean_over_prevalence"] <= pt["pass_if"]["null_mean_over_prevalence_max"]
    assert r["real_statistic"] > r["null_max"]
    assert r["empirical_p"] == pytest.approx(1 / (pt["n_permutations"] + 1))
    assert r["passed"]


def test_identity_guard_every_candidate_group(doc):
    g = doc["guard"]
    assert g["base"] == CFG["groups"]["base"]
    assert set(g["summary"]) == set(CFG["groups"]["candidates"])
    for name, r in g["groups"].items():
        assert r["config_is_frozen"], name
        assert g["summary"][name] == {"verdict": r["verdict"], "cleared": r["cleared"]}
    want = [g["base"], *[k for k, v in g["summary"].items() if v["cleared"]]]
    assert g["cleared_groups"] == want
