"""scripts/p3/run_p3.py end to end on the P2 fixture (same code path Dani runs on HI-Medium).

The fixture is far too small for the statistics (the guard ends as insufficient_data, and the
fixture's F2 eval day has no positive), so this test checks the wiring and the stop conditions:
  - feature matrices per period, integrity with the P2 alert store, split boundaries;
  - TEST only with --include-test, with counts and hashes only, and the regime-B count equal to
    the P2 record;
  - the real-data point-in-time check finds zero mismatches;
  - the permutation stage feeds the right rows and blocks, and refuses an eval day with one class;
  - results JSON holds aggregates only (no account key, no alert id).
The stages build on each other's outputs, so the tests run in file order (as the driver does).
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from scripts.p3 import run_p3
from src.features import compute as fc


@pytest.fixture(scope="module")
def drv(p2_run, tmp_path_factory):
    r = p2_run["root"]
    out = tmp_path_factory.mktemp("p3drv")
    base = [
        "--variant", "FIX", "--interim", str(p2_run["info"]["interim"]),
        "--store-root", str(r / "out"), "--out-root", str(out), "--results", str(out / "res.json"),
        "--rules", str(p2_run["cp"].rules),
        "--thresholds", str(p2_run["configs"] / "p2_rule_thresholds.yaml"),
        "--fx", str(p2_run["configs"] / "p2_fx_usd_per_unit.yaml"),
        "--sources", str(p2_run["info"]["sources"]), "--p2-results", str(r / "results.json"),
    ]  # fmt: skip
    env = run_p3.Env(run_p3.parse(["features", *base]))
    env.cfg["pit_check"] |= {"n_alerts_batch": 40, "n_alerts_pure": 6}
    env.cfg["permutation_test"]["n_permutations"] = 4
    return {"base": base, "env": env, "out": out}


def _doc(drv) -> dict:
    return json.loads((drv["out"] / "res.json").read_text(encoding="utf-8"))


def test_features_stage_pre_test(drv, p2_run):
    run_p3.main(["features", *drv["base"]], env=drv["env"])
    d = _doc(drv)["features"]
    assert set(d["periods"]) == {"TRAIN", "VALIDATION", "CALIBRATION"}
    alerts = pl.read_parquet(p2_run["base"] / "runtime" / "alerts.parquet")
    for p, rec in d["periods"].items():
        m = pl.read_parquet(drv["env"].feat_dir / f"{p}.parquet")
        assert rec["rows"] == m.height == alerts.filter(pl.col("period") == p).height
        assert rec["integrity"]["rule_stat_mismatches"] == 0
        assert rec["integrity"]["triggered_rules_mismatches"] == 0
        assert m.columns == [*fc.ALERT_KEYS, *fc.feature_names(drv["env"].ctx), "split_period"]
        assert (m["split_period"] == p).all()
    assert not (drv["env"].feat_dir / "TEST.parquet").exists()
    assert set(d["boundaries"]) == {"TRAIN->VALIDATION", "VALIDATION->CALIBRATION"}


def test_check_stage_finds_no_mismatch(drv):
    run_p3.main(["check", *drv["base"]], env=drv["env"])
    c = _doc(drv)["check"]
    assert c["passed"] and c["n_sampled"] == 40 and c["n_pure"] == 6
    assert c["deletion"] == c["mutation"] == c["stored_vs_recomputed"] == c["pure_vs_batch"] == 0


def test_permutation_stage_refuses_a_one_class_eval_day(drv):
    with pytest.raises(ValueError, match="both classes"):  # fixture F2 eval day (09-05): 0 pos
        run_p3.main(["permutation", *drv["base"]], env=drv["env"])


def test_permutation_stage_wiring(drv, monkeypatch):
    """Same stage with fixture days that have both classes: rows, blocks and result shape."""
    specs = [
        {"name": "F1", "fit_days": ["2022-09-02", "2022-09-03"], "eval_day": "2022-09-04"},
        {"name": "F2", "fit_days": ["2022-09-02", "2022-09-03"], "eval_day": "2022-09-04"},
    ]
    monkeypatch.setattr(run_p3, "_fold_specs", lambda env, names: specs)
    r = run_p3.stage_permutation(drv["env"])
    assert r["n_permutations"] == 4 and len(r["per_fold"]) == 2
    tr = pl.read_parquet(drv["env"].feat_dir / "TRAIN.parquet").with_columns(
        pl.col("window_start").dt.date().cast(pl.String).alias("d")
    )
    assert r["per_fold"][0]["n_fit"] == tr.filter(pl.col("d").is_in(specs[0]["fit_days"])).height
    assert r["per_fold"][0]["n_eval"] == tr.filter(pl.col("d") == "2022-09-04").height
    assert r["features"] == len(run_p3.reg.model_features(drv["env"].ctx))


def test_guard_stage_runs_every_candidate_group(drv):
    run_p3.main(["guard", *drv["base"]], env=drv["env"])
    g = _doc(drv)["guard"]
    assert set(g["summary"]) == set(drv["env"].cfg["groups"]["candidates"])
    # the fixture is far below N_min = 30 positives per subset: nothing may be cleared
    assert all(v["verdict"] == "insufficient_data" for v in g["summary"].values())
    assert g["cleared_groups"] == ["rule"]


def test_test_matrix_only_with_flag_and_counts_only(drv, p2_run, capsys):
    run_p3.main(["features", "--include-test", *drv["base"]], env=drv["env"])
    d = _doc(drv)["features"]
    t = d["periods"]["TEST"]
    assert set(t) == {"rows", "parquet_sha256", "seconds", "integrity_ok"}  # no statistics
    rb = json.loads((p2_run["root"] / "results.json").read_text(encoding="utf-8"))
    want = rb["build"]["feasibility"]["counts"]["TEST_regime_B_unseen_accounts"]["n_alerts"]
    assert d["regime_b_unseen_test_alerts"] == want
    assert "CALIBRATION->TEST" in d["boundaries"]
    assert "| 5 |" in capsys.readouterr().out  # the exposure-log row is printed for Dani
    with pytest.raises(SystemExit):
        run_p3.main(["all", "--include-test", *drv["base"]], env=drv["env"])


def test_results_hold_aggregates_only(drv, p2_run):
    text = (drv["out"] / "res.json").read_text(encoding="utf-8")
    alerts = pl.read_parquet(p2_run["base"] / "runtime" / "alerts.parquet")
    for k in alerts["account_key"].unique().to_list() + alerts["alert_id"].to_list():
        assert k not in text
