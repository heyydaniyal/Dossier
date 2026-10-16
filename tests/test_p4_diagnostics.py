"""P4 step 0 diagnosis: known answers for the label-free module (src/models/diagnostics.py), the
label-detail loader, and the driver end to end on the P2 fixture.

The two simulated worlds are the point of the probe:
  recognition world  labels attach to ACCOUNTS (repeat launderers) and features carry only a stable
                     per-account fingerprint -> seen AUC high, disjoint AUC at chance;
  population world   labels follow the DAY's features for busy accounts (the kind that recur), new
                     accounts carry no signal -> seen AUC high even when the seen accounts' own
                     rows are removed.
If the probe could not tell these apart, its verdict on the real data would mean nothing.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from src.eval import training_labels as tl
from src.models import diagnostics as dg

CFG = {"max_control_shortfall": 0.25}  # = configs/p4_diagnosis.yaml (checked below)

# ------------------------------------------------------------------ simulated worlds


def _new_accounts(rng, day, n, d, prev=0.07):
    keys = np.array([f"new{day}_{i}" for i in range(n)], dtype=object)
    return keys, rng.normal(size=(n, d)), (rng.random(n) < prev).astype(int)


def _world(kind: str, seed: int = 0, n_busy=3000, n_new=1500, d=3):
    """Two fit days + one eval day. Returns a FoldData."""
    rng = np.random.default_rng(seed)
    finger = rng.normal(size=(n_busy, d))
    repeat = rng.random(n_busy) < 0.05  # recognition world: these launder whenever active
    days = []
    for day in range(3):
        active = np.flatnonzero(rng.random(n_busy) < 0.5)
        keys = np.array([f"busy{i}" for i in active], dtype=object)
        if kind == "recognition":
            X = finger[active] + 0.01 * rng.normal(size=(len(active), d))
            y = repeat[active].astype(int)
        else:  # population: signal is in the day's own features
            X = rng.normal(size=(len(active), d))
            y = (rng.random(len(active)) < 1 / (1 + np.exp(-(-4.0 + 3.0 * X[:, 0])))).astype(int)
        nk, nX, ny = _new_accounts(rng, day, n_new, d)
        days.append((np.concatenate([keys, nk]), np.vstack([X, nX]), np.concatenate([y, ny])))
    fit_k = np.concatenate([days[0][0], days[1][0]])
    fit_X = np.vstack([days[0][1], days[1][1]])
    fit_y = np.concatenate([days[0][2], days[1][2]])
    ek, eX, ey = days[2]
    return dg.FoldData("S", fit_X, fit_y, fit_k, eX, ey, ek)


def _knn(seed):  # memorises rows: the extreme recogniser
    return KNeighborsClassifier(n_neighbors=1)


def _lr(seed):
    return LogisticRegression(max_iter=500)


def test_probe_detects_recognition():
    f = _world("recognition", seed=1)
    r = dg.recognition_probe(
        [f], _knn, [0], n_boot=300, boot_seed=1, ci=0.95, min_seen_pos=30, **CFG
    )
    p = r["pooled"]
    assert p["full"] > 0.95 and p["control"] > 0.95
    assert abs(p["disjoint"] - 0.5) < 0.1
    assert p["label_memory"] > 0.95  # labels persist per account
    assert r["verdict"] == "recognition"


def test_probe_detects_population_difference():
    f = _world("population", seed=2)
    r = dg.recognition_probe(
        [f], _lr, [0], n_boot=300, boot_seed=2, ci=0.95, min_seen_pos=30, **CFG
    )
    p = r["pooled"]
    assert p["full"] > 0.8 and p["disjoint"] > 0.8
    assert p["unseen_full"] < p["disjoint"] - 0.2  # busy accounts not active in the fit days carry
    assert abs(p["label_memory"] - 0.5) < 0.15  # signal too, so unseen is above 0.5; no persistence
    assert r["verdict"] == "population"


def test_probe_reports_insufficient_data_when_seen_positives_are_few():
    f = _world("population", seed=3)
    r = dg.recognition_probe(
        [f], _lr, [0], n_boot=50, boot_seed=3, ci=0.95, min_seen_pos=10**6, **CFG
    )
    assert r["verdict"] == "insufficient_data"


def test_control_arm_removes_the_same_rows_and_positives_outside_the_eval_accounts():
    f = _world("recognition", seed=4)
    s = dg.fold_scores(f, _knn, [0], seed_base=4)
    own = np.isin(f.acct_fit, f.acct_eval[f.seen])
    assert s["removed"] == {
        "rows": int(own.sum()),
        "positives": int(f.y_fit[own].sum()),
        "control_positives_from_own_rows": 0,
    }
    n_pos = s["removed"]["positives"]
    rm, short = dg._control_rows(f, n_pos, s["removed"]["rows"] - n_pos, np.random.default_rng(0))
    assert short == 0
    assert len(set(rm.tolist())) == s["removed"]["rows"]
    assert not np.isin(f.acct_fit[rm], f.acct_eval).any()
    assert int(f.y_fit[rm].sum()) == s["removed"]["positives"]


def _repeat_launderer_world(seed: int, signal: str):
    """Seen eval accounts hold 150 of 170 fit positives; only 20 positives lie on accounts absent
    from the eval day, so the matched control cannot be built from them alone (the first version
    crashed here). signal: 'fingerprint' (recognition) or 'day' (population)."""
    rng = np.random.default_rng(seed)
    nf, ne = 3000, 1000
    kf = np.array([f"a{i}" for i in range(nf)], dtype=object)
    finger = rng.normal(size=(nf, 3))
    yf = np.zeros(nf, int)
    yf[:150] = 1
    yf[2900:2920] = 1
    ke = np.concatenate([kf[:300], np.array([f"n{i}" for i in range(ne - 300)], dtype=object)])
    if signal == "fingerprint":  # the same accounts launder again; features = stable fingerprint
        Xf = finger + 0.01 * rng.normal(size=finger.shape)
        Xe = np.vstack([finger[:300], rng.normal(size=(ne - 300, 3))])
        ye = np.r_[yf[:300], (rng.random(ne - 300) < 0.06).astype(int)]
    else:  # labels follow the day's own features, for everyone
        Xf = rng.normal(size=(nf, 3))
        yf = (rng.random(nf) < 1 / (1 + np.exp(-(-3.0 + 2.5 * Xf[:, 0])))).astype(int)
        yf[:150] = 1
        yf[300:] = 0  # as in the fingerprint world: only 20 positives outside the eval accounts
        yf[2900:2920] = 1
        Xf[:150, 0] += 1.5
        Xf[2900:2920, 0] += 1.5
        Xe = rng.normal(size=(ne, 3))
        ye = (rng.random(ne) < 1 / (1 + np.exp(-(-3.0 + 2.5 * Xe[:, 0])))).astype(int)
    return dg.FoldData("R", Xf, yf, kf, Xe, ye, ke)


def test_control_arm_never_crashes_and_records_the_top_up():
    f = _repeat_launderer_world(0, "fingerprint")
    s = dg.fold_scores(f, _knn, [0, 1], seed_base=0)
    own = np.isin(f.acct_fit, f.acct_eval[f.seen])
    need = int(f.y_fit[own].sum())
    outside = int(f.y_fit[~np.isin(f.acct_fit, f.acct_eval)].sum())
    assert s["removed"]["positives"] == need
    assert s["removed"]["control_positives_from_own_rows"] == need - outside > 0
    rm, short = dg._control_rows(f, need, s["removed"]["rows"] - need, np.random.default_rng(3))
    assert len(set(rm.tolist())) == s["removed"]["rows"] and int(f.y_fit[rm].sum()) == need
    assert short == need - outside


def test_recognition_survives_a_large_top_up():
    """The top-up biases control toward disjoint (against recognition): strong recognition must
    still be called recognition."""
    f = _repeat_launderer_world(1, "fingerprint")
    r = dg.recognition_probe(
        [f], _knn, [0], n_boot=300, boot_seed=1, ci=0.95, min_seen_pos=30, **CFG
    )
    assert r["control_shortfall_share"] > 0.25
    assert r["verdict"] == "recognition"


def test_other_verdicts_become_control_unmatched_with_a_large_top_up():
    f = _repeat_launderer_world(2, "day")
    r = dg.recognition_probe(
        [f], _lr, [0], n_boot=300, boot_seed=2, ci=0.95, min_seen_pos=30, **CFG
    )
    assert r["control_shortfall_share"] > 0.25
    assert r["verdict"] == "control_unmatched"
    r2 = dg.recognition_probe(
        [f], _lr, [0], n_boot=300, boot_seed=2, ci=0.95, min_seen_pos=30, max_control_shortfall=1.0
    )
    assert r2["verdict"] != "control_unmatched"


def test_cfg_matches_the_declared_config():
    from scripts.p4.diagnose import load_config

    assert load_config()["recognition"]["max_control_shortfall"] == CFG["max_control_shortfall"]


@pytest.mark.parametrize(
    ("diff_ci", "disj", "disj_ci", "ctrl", "unseen", "want"),
    [
        ([0.10, 0.30], 0.55, [0.45, 0.65], 0.90, 0.52, "recognition"),
        ([-0.02, 0.30], 0.55, [0.45, 0.65], 0.90, 0.52, "mixed"),  # drop not credible (CI spans 0)
        ([0.10, 0.30], 0.80, [0.70, 0.88], 0.90, 0.52, "mixed"),  # credible drop, < half the gap
        ([-0.03, 0.04], 0.88, [0.80, 0.94], 0.90, 0.52, "population"),
        ([-0.03, 0.08], 0.88, [0.80, 0.94], 0.90, 0.52, "mixed"),  # cannot rule out a drop
        ([-0.03, 0.04], 0.60, [0.48, 0.70], 0.62, 0.52, "mixed"),  # disjoint not above unseen
        (None, 0.6, [0.5, 0.7], 0.9, 0.5, "insufficient_data"),
    ],
)
def test_decision_table(diff_ci, disj, disj_ci, ctrl, unseen, want):
    assert dg.decide_recognition(diff_ci, disj, disj_ci, ctrl, unseen) == want
    # a control CI well above the unseen level never changes the verdict
    assert dg.decide_recognition(diff_ci, disj, disj_ci, ctrl, unseen, [0.8, 0.95]) == want


def test_no_gap_when_seen_accounts_are_not_ranked_better():
    assert (
        dg.decide_recognition([-0.05, 0.05], 0.5, [0.45, 0.55], 0.51, 0.5, [0.46, 0.56]) == "no_gap"
    )


def test_pure_noise_world_is_no_gap():
    rng = np.random.default_rng(9)
    keys = np.array([f"k{i}" for i in range(6000)], dtype=object)
    f = dg.FoldData("N", rng.normal(size=(4000, 3)), (rng.random(4000) < 0.05).astype(int),
                    keys[:4000], rng.normal(size=(3000, 3)), (rng.random(3000) < 0.05).astype(int),
                    np.concatenate([keys[:1500], keys[4000:5500]]))  # fmt: skip
    r = dg.recognition_probe(
        [f], _lr, [0], n_boot=300, boot_seed=9, ci=0.95, min_seen_pos=30, **CFG
    )
    assert r["verdict"] == "no_gap"


# ------------------------------------------------------------------ metrics and validation


def test_auc_equals_sklearn_with_ties_and_none_for_one_class():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = rng.integers(0, 5, 500).astype(float)  # many ties
    assert dg.auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert dg.auc(np.zeros(5, int), np.arange(5.0)) is None


def test_pooled_is_positive_weighted_and_skips_undefined():
    assert dg.pooled([0.6, None, 0.9], [10, 5, 30]) == pytest.approx((6 + 27) / 40)
    assert dg.pooled([None], [3]) is None


def test_fold_validation():
    k = np.array(["a", "a"], dtype=object)
    with pytest.raises(ValueError, match="one eval row per account"):
        dg.FoldData(
            "x", np.zeros((2, 1)), np.array([0, 1]), k, np.zeros((2, 1)), np.array([0, 1]), k
        )
    with pytest.raises(ValueError, match="positives only"):
        dg.FoldData("x", np.zeros((2, 1)), np.array([0, 1]), np.array(["a", "b"], dtype=object),
                    np.zeros((2, 1)), np.array([0, 1]), np.array(["a", "b"], dtype=object),
                    strata={"s": np.array([True, False])})  # fmt: skip


def test_univariate_orientation_nan_and_point_equal_sklearn():
    rng = np.random.default_rng(5)
    n = 2000
    y = (rng.random(n) < 0.1).astype(int)
    good = y * 1.0 + rng.normal(size=n)  # positive orientation
    bad = -good  # same signal, negative orientation
    noisy = rng.normal(size=n)
    noisy[:200] = np.nan
    X = np.column_stack([good, bad, noisy])
    keys = np.array([f"a{i}" for i in range(n)], dtype=object)
    f = dg.FoldData("F", X[:10], np.r_[np.zeros(5, int), np.ones(5, int)],
                    np.array([f"z{i}" for i in range(10)], dtype=object), X, y, keys)  # fmt: skip
    u = dg.univariate([f], ["good", "bad", "noisy"], n_boot=100, boot_seed=0, ci=0.95, top_k=3)
    by = {r["feature"]: r for r in u["all"]}
    want = roc_auc_score(y, good)
    assert by["good"]["orientation"] == "+" and by["good"]["auc_unseen"] == pytest.approx(want)
    assert by["bad"]["orientation"] == "-" and by["bad"]["auc_unseen"] == pytest.approx(want)
    lo, hi = by["good"]["ci_unseen"]
    assert lo < want < hi and by["bad"]["ci_unseen"] == pytest.approx([lo, hi], abs=1e-9)
    assert by["noisy"]["nan_share_eval"] == pytest.approx(0.1)
    ref = roc_auc_score(y, np.where(np.isnan(noisy), -1e300, noisy))  # NaN ranked lowest
    assert by["noisy"]["auc_unseen"] == pytest.approx(max(ref, 1.0 - ref))
    assert u["top"][0]["feature"] in ("good", "bad")


def test_strata_auc_known_answer():
    rng = np.random.default_rng(6)
    n = 3000
    y = (rng.random(n) < 0.1).astype(int)
    easy = y.astype(bool) & (rng.random(n) < 0.5)
    hard = y.astype(bool) & ~easy
    score = rng.random(n) + 2.0 * easy  # easy positives ranked on top, hard ones at random
    keys = np.array([f"a{i}" for i in range(n)], dtype=object)
    f = dg.FoldData("F", np.zeros((4, 1)), np.array([0, 1, 0, 1]),
                    np.array(["z1", "z2", "z3", "z4"], dtype=object), np.zeros((n, 1)), y, keys,
                    strata={"easy": easy, "hard": hard})  # fmt: skip
    r = dg.strata_auc([f], {"F": [score]}, min_pos=15)["strata"]
    assert r["easy"]["auc_unseen"] == pytest.approx(1.0)
    assert abs(r["hard"]["auc_unseen"] - 0.5) < 0.06
    assert r["easy"]["n_pos_unseen"] + r["hard"]["n_pos_unseen"] == int(y.sum())
    r2 = dg.strata_auc([f], {"F": [score]}, min_pos=10**6)["strata"]
    assert r2["easy"]["auc_unseen"] is None and r2["easy"]["n_pos_unseen"] > 0


# ------------------------------------------------------------------ label details (eval module)


STRATA_CFG = {
    "laundering_share_bins": [0.0, 0.1, 0.5],
    "fan_in_rule": "R04_FAN_IN",
    "typologies": ["FAN-IN", "FAN-OUT", "CYCLE"],
    "unattributed": True,
}


def test_strata_refuse_test_before_opening_anything(tmp_path):
    with pytest.raises(tl.HoldoutAccessError):
        tl.load_fit_strata(["TRAIN", "TEST"], "NOPE", tmp_path, STRATA_CFG)  # path does not exist


def test_strata_known_answers_on_fixture(p2_run):
    """Fixture TRAIN truth: attempt 0 FAN-IN on 09-03 (L01..L08 -> L00) and unattributed
    laundering L40 -> L41 on 09-04. Only positives get strata; only names leave the eval module."""
    root = p2_run["root"] / "out"
    d = tl.load_fit_strata(["TRAIN"], "FIX", root, STRATA_CFG)
    y = tl.load_fit_labels(["TRAIN"], "FIX", root)
    assert d.select("alert_id", "y").equals(y.select("alert_id", "y"))
    assert d.columns == ["alert_id", "period", "y", "strata"]
    assert (d.filter(pl.col("y") == 0)["strata"].list.len() == 0).all()
    pos = d.filter(pl.col("y") == 1)
    assert pos.height and (pos["strata"].list.len() >= 2).all()  # a share bin and a rule group
    flat = [k for x in pos["strata"].to_list() for k in x]
    assert "typ_FAN-IN" in flat and "unattributed" in flat
    assert "typ_FAN-OUT" not in flat and "typ_CYCLE" not in flat  # those attempts are not in TRAIN
    alerts = pl.read_parquet(p2_run["base"] / "runtime" / "alerts.parquet")
    j = pos.join(alerts.select("alert_id", "triggered_rules"), on="alert_id")
    fan = [STRATA_CFG["fan_in_rule"] in r for r in j["triggered_rules"].to_list()]
    group = [("rule_fan_in" in x) for x in j["strata"].to_list()]
    other = [("rule_other" in x) for x in j["strata"].to_list()]
    assert group == fan and other == [not f for f in fan]
    assert any(fan)  # the fixture's TRAIN fan-in hub is alerted by R04


def test_share_bins():
    e = [0.0, 0.1, 0.5]
    assert tl.share_bin(0.0, e) == "share_[0.00,0.10)"
    assert tl.share_bin(0.1, e) == "share_[0.10,0.50)"
    assert tl.share_bin(0.5, e) == "share_[0.50,inf)"
    assert tl.share_bin(3.0, e) == "share_[0.50,inf)"


# ------------------------------------------------------------------ driver on the P2 fixture


@pytest.fixture(scope="module")
def p4_fix(p2_run, tmp_path_factory, monkeypatch_module):
    """P3 feature matrices for the fixture, then a fixture-sized diagnosis config. The fixture's
    real fold F1 fit day (09-02) has no positive, so the folds are patched to days that do."""
    from scripts.p3 import run_p3
    from scripts.p4 import diagnose

    r = p2_run["root"]
    out = tmp_path_factory.mktemp("p4")
    p3_args = [
        "features", "--variant", "FIX", "--interim", str(p2_run["info"]["interim"]),
        "--store-root", str(r / "out"), "--out-root", str(out / "p3"),
        "--results", str(out / "p3.json"), "--rules", str(p2_run["cp"].rules),
        "--thresholds", str(p2_run["configs"] / "p2_rule_thresholds.yaml"),
        "--fx", str(p2_run["configs"] / "p2_fx_usd_per_unit.yaml"),
        "--sources", str(p2_run["info"]["sources"]), "--p2-results", str(r / "results.json"),
    ]  # fmt: skip
    run_p3.main(p3_args)
    cfg = diagnose.load_config()
    cfg["seeds"] = [1]
    cfg["recognition"] |= {"n_boot": 20, "min_seen_pos": 1}
    cfg["univariate"] |= {"n_boot": 10}
    cfg["strata"]["min_stratum_pos"] = 1
    cfg["recognition"]["max_control_shortfall"] = 0.123  # must reach the probe unchanged
    import yaml

    cpath = out / "diag.yaml"
    cpath.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    folds = [
        {"name": "F1", "fit_days": ["2022-09-02", "2022-09-03"], "eval_day": "2022-09-04"},
        {"name": "F2", "fit_days": ["2022-09-02", "2022-09-03", "2022-09-04"],
         "eval_day": "2022-09-05"},
    ]  # fmt: skip
    monkeypatch_module.setattr(diagnose, "load_v2_config", lambda: {"folds": folds})
    seen_periods = []
    real = diagnose.load_fit_strata

    def spy(periods, *a, **k):
        seen_periods.append(list(periods))
        return real(periods, *a, **k)

    monkeypatch_module.setattr(diagnose, "load_fit_strata", spy)
    probe_kwargs = []
    real_probe = diagnose.dg.recognition_probe

    def probe_spy(*a, **k):
        probe_kwargs.append(k)
        return real_probe(*a, **k)

    monkeypatch_module.setattr(diagnose.dg, "recognition_probe", probe_spy)
    args = [
        "--variant", "FIX", "--config", str(cpath), "--features-root", str(out / "p3"),
        "--store-root", str(r / "out"), "--p3-results", str(out / "p3.json"),
        "--results", str(out / "diag.json"), "--rules", str(p2_run["cp"].rules),
        "--thresholds", str(p2_run["configs"] / "p2_rule_thresholds.yaml"),
    ]  # fmt: skip
    return {"args": args, "out": out, "periods": seen_periods, "diagnose": diagnose,
            "probe_kwargs": probe_kwargs}  # fmt: skip


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_driver_end_to_end_on_fixture(p4_fix):
    d = p4_fix["diagnose"]
    assert d.main(p4_fix["args"]) == 0
    doc = json.loads((p4_fix["out"] / "diag.json").read_text(encoding="utf-8"))
    assert p4_fix["periods"] and all(p == ["TRAIN"] for p in p4_fix["periods"])
    assert p4_fix["probe_kwargs"]
    assert all(k["max_control_shortfall"] == 0.123 for k in p4_fix["probe_kwargs"])
    assert doc["inputs"]["periods_read"] == ["TRAIN"]
    assert set(doc["recognition"]) == {"rule", "all"}
    assert doc["inputs"]["n_features"]["all"] > doc["inputs"]["n_features"]["rule"]
    for r in doc["recognition"].values():
        assert r["verdict"] in {
            "no_gap",
            "recognition",
            "population",
            "mixed",
            "insufficient_data",
            "control_unmatched",
        }
        assert [x["fold"] for x in r["per_fold"]] == ["F1", "F2"]
    assert doc["univariate"]["n_features"] == doc["inputs"]["n_features"]["all"]
    assert doc["strata"]["total_unseen_pos"] >= 1


def test_driver_output_holds_aggregates_only(p4_fix, p2_run):
    text = (p4_fix["out"] / "diag.json").read_text(encoding="utf-8")
    alerts = pl.read_parquet(p2_run["base"] / "runtime" / "alerts.parquet")
    for col in ("account_key", "alert_id"):
        for v in alerts[col].unique().to_list()[:500]:
            assert v not in text, (col, v)


def test_driver_refuses_a_matrix_p3_did_not_record(p4_fix):
    import shutil

    d = p4_fix["diagnose"]
    feat = p4_fix["out"] / "p3" / "FIX" / "features" / "TRAIN.parquet"
    backup = feat.with_suffix(".bak")
    shutil.copy(feat, backup)
    try:
        m = pl.read_parquet(feat)
        m.with_columns(pl.col("n_txn") + 1).write_parquet(feat)
        with pytest.raises(SystemExit, match="P3 record"):
            d.main(p4_fix["args"])
    finally:
        shutil.move(backup, feat)


def test_strata_count_unseen_positives_only():
    """A stratum's positives on SEEN accounts must not enter the unseen stratum statistics."""
    n = 400
    y = np.zeros(n, int)
    y[:40] = 1
    keys = np.array([f"a{i}" for i in range(n)], dtype=object)
    fit_keys = keys[:20]  # the first 20 positives are seen
    f = dg.FoldData("F", np.zeros((20, 1)), np.r_[np.zeros(10, int), np.ones(10, int)], fit_keys,
                    np.zeros((n, 1)), y, keys, strata={"all_pos": y.astype(bool)})  # fmt: skip
    score = np.r_[np.full(20, -1.0), np.full(20, 2.0), np.zeros(n - 40)]  # seen pos ranked lowest
    r = dg.strata_auc([f], {"F": [score]}, min_pos=1)
    assert r["total_unseen_pos"] == 20
    assert r["strata"]["all_pos"]["n_pos_unseen"] == 20
    assert r["strata"]["all_pos"]["auc_unseen"] == pytest.approx(1.0)
