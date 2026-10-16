"""P4 models: metrics by known answers, model families (additive explanations, exact round trip),
load-by-hash, the declared search space and selection rule, stop rules, the identity-guard wiring,
and both driver stages end to end on the P2 fixture (TRAIN/VALIDATION only, VALIDATION once)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.p4 import models as drv
from src.data.periods import ROOT
from src.models import families as fam
from src.models import metrics as mt
from src.models import store

# ------------------------------------------------------------------ metrics


def test_precision_at_recall_and_capacity_known_answers():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    s = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], float)
    # recall >= 2/3 first reached at rank 3 (precision 2/3); recall 1 at rank 6 (precision 1/2)
    assert mt.precision_at_recall(y, s, 0.6) == pytest.approx(2 / 3)
    assert mt.precision_at_recall(y, s, 1.0) == pytest.approx(0.5)
    assert mt.recall_at_capacity(y, s, 0.3) == pytest.approx(2 / 3)  # top 3
    assert mt.recall_at_capacity(y, s, 0.1) == pytest.approx(1 / 3)


def test_recall_at_capacity_splits_ties_and_ignores_row_order():
    y = np.array([1, 0, 0, 0])
    s = np.array(
        [1.0, 1.0, 1.0, 1.0]
    )  # all tied: reviewing 2 of 4 finds the positive half the time
    assert mt.recall_at_capacity(y, s, 0.5) == pytest.approx(0.5)
    assert mt.recall_at_capacity(y[::-1], s, 0.5) == pytest.approx(0.5)


def test_cost_threshold_equals_brute_force():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.08).astype(int)
    s = np.round(y * 0.8 + rng.normal(size=400), 1)  # ties on purpose
    for ratio in (1, 5, 20, 50):
        r = mt.cost_threshold(y, s, ratio)
        best = min(
            [(ratio * y.sum(), np.inf)]
            + [(ratio * (y.sum() - y[s >= t].sum()) + (1 - y[s >= t]).sum(), t) for t in set(s)],
            key=lambda x: (x[0], -x[1]),
        )
        assert r["cost"] == pytest.approx(best[0])
        assert r["threshold"] == best[1]
        assert r["n_flagged"] == int((s >= r["threshold"]).sum())


def test_cost_threshold_tie_goes_to_fewer_alerts():
    # ratio 1: flag nothing costs 1 (one FN); flag both costs 1 (one FP) -> flag nothing
    r = mt.cost_threshold(np.array([0, 1]), np.array([2.0, 1.0]), 1)
    assert r["threshold"] == float("inf") and r["n_flagged"] == 0


def test_pr_auc_roc_brier_match_sklearn():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.1).astype(int)
    s = rng.random(300) + y * 0.3
    assert mt.pr_auc(y, s) == pytest.approx(average_precision_score(y, s))
    assert mt.roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert mt.brier(y, np.clip(s, 0, 1)) == pytest.approx(np.mean((np.clip(s, 0, 1) - y) ** 2))
    assert mt.pr_auc(np.zeros(5, int), s[:5]) is None


# ------------------------------------------------------------------ families


FEATS = ["fired_R01", "fired_R04", "volume_usd", "share_x", "lag"]


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    f1 = (rng.random(n) < 0.1).astype(float)
    f4 = (rng.random(n) < 0.6).astype(float)
    vol = np.exp(rng.normal(8, 2, n))
    share = rng.random(n)
    lag = np.where(rng.random(n) < 0.6, np.nan, rng.exponential(30, n))
    z = -3.5 + 1.2 * f1 + 0.8 * f4 + 0.3 * (np.log1p(vol) - 8) + np.where(np.isnan(lag), 0.5, 0)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return np.column_stack([f1, f4, vol, share, lag]), y


LGB_P = {"learning_rate": 0.1, "n_estimators": 300, "num_leaves": 7, "min_child_samples": 20,
         "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
         "deterministic": True, "force_row_wise": True, "num_threads": 2, "verbose": -1,
         "random_state": 1, "class_weight": "none"}  # fmt: skip


def _models():
    return [
        fam.RulesCount(FEATS),
        fam.RulesWeighted(FEATS, params={"smoothing": 1.0}),
        fam.Logistic(FEATS, params={"C": 0.1, "class_weight": "none", "max_iter": 2000}),
        fam.LightGBM(FEATS, params=dict(LGB_P), early_stopping_rounds=30),
    ]


@pytest.mark.parametrize("i", range(4))
def test_contributions_are_additive_and_round_trip_is_exact(i, tmp_path):
    X, y = _data()
    m = _models()[i]
    m.fit(X[:2000], y[:2000], eval_set=(X[2000:], y[2000:]) if m.family == "lightgbm" else None)
    c, b = m.contributions(X)
    np.testing.assert_allclose(c.sum(axis=1) + b, m.margin(X), atol=1e-9)
    h = store.save(m, {"t": 1}, tmp_path)
    r, meta = store.load(h, tmp_path)
    assert meta == {"t": 1} and r.family == m.family
    np.testing.assert_array_equal(r.margin(X), m.margin(X))


def test_logistic_margin_equals_an_independent_computation():
    """Logistic.margin is computed from contributions; check it against the design written out
    here (signed log1p, fit-row mean/sd, NaN -> 0 plus a missing flag, coef, flag coef)."""
    X, y = _data()
    m = fam.Logistic(FEATS, params={"C": 0.5, "class_weight": "none", "max_iter": 2000}).fit(X, y)
    T = np.sign(X) * np.log1p(np.abs(X))
    mu, sd = np.nanmean(T, axis=0), np.nanstd(T, axis=0)
    np.testing.assert_allclose(m.state["mean"], mu)
    np.testing.assert_allclose(m.state["scale"], sd)
    Z = (T - mu) / sd
    miss = np.isnan(Z)
    Z = np.where(miss, 0.0, Z)
    want = Z @ np.array(m.state["coef"]) + m.state["intercept"]
    want += miss[:, m.state["flag_cols"]] @ np.array(m.state["coef_flag"])
    np.testing.assert_allclose(m.margin(X), want, atol=1e-9)
    assert abs(m.state["coef_flag"][0]) > 0.1  # the missing flag carries the planted signal


def test_rule_scores_known_answers():
    X, y = _data()
    rc = fam.RulesCount(FEATS).fit(X, y)
    np.testing.assert_array_equal(rc.margin(X), X[:, 0] + X[:, 1])
    rw = fam.RulesWeighted(FEATS, params={"smoothing": 1.0}).fit(X, y)
    prev = (y.sum() + 1) / (len(y) + 2)
    on = X[:, 0] == 1
    assert rw.weights["fired_R01"] == pytest.approx(
        np.log(((y[on].sum() + 1) / (on.sum() + 2)) / prev)
    )
    assert rw.weights["fired_R01"] > 0  # a rule with lift > 1 adds to the score
    assert set(rw.weights) == {"fired_R01", "fired_R04"}


def test_logistic_handles_nan_and_learns_the_direction():
    X, y = _data()
    m = fam.Logistic(FEATS, params={"C": 1.0, "class_weight": "none", "max_iter": 2000}).fit(X, y)
    assert m.state["flag_cols"] == [4]  # only 'lag' has NaN
    assert np.isfinite(m.margin(X)).all()
    assert m.state["coef"][0] > 0 and m.state["coef"][1] > 0


def test_lightgbm_early_stops_on_the_given_eval_set_and_refits_with_fixed_trees():
    X, y = _data()
    m = fam.LightGBM(FEATS, params=dict(LGB_P), early_stopping_rounds=30)
    m.fit(X[:2000], y[:2000], eval_set=(X[2000:], y[2000:]))
    assert 1 <= m.best_iteration < 300
    m2 = fam.LightGBM(FEATS, params={**LGB_P, "n_estimators": 17}).fit(X[:2000], y[:2000])
    assert m2.best_iteration == 17


def test_class_weight_balanced_changes_the_model():
    X, y = _data()
    a = fam.Logistic(FEATS, params={"C": 1.0, "class_weight": "none", "max_iter": 2000}).fit(X, y)
    b = fam.Logistic(FEATS, params={"C": 1.0, "class_weight": "balanced", "max_iter": 2000}).fit(
        X, y
    )
    assert b.state["intercept"] > a.state["intercept"] + 1.0  # weighting shifts the margin up


# ------------------------------------------------------------------ load by hash


def test_store_refuses_tampering_prefixes_and_line_ending_conversion(tmp_path):
    X, y = _data()
    m = fam.RulesWeighted(FEATS, params={"smoothing": 1.0}).fit(X, y)
    h = store.save(m, {}, tmp_path)
    p = tmp_path / f"model_{h[:16]}.json"
    with pytest.raises(store.ModelHashError):
        store.load(h[:16], tmp_path)
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))  # what core.autocrlf would do
    with pytest.raises(store.ModelHashError, match="do not match"):
        store.load(h, tmp_path)
    with pytest.raises(store.ModelHashError, match="no model file"):
        store.load("0" * 64, tmp_path)


def test_gitattributes_keeps_model_bytes_exact():
    assert "models/** -text" in (ROOT / ".gitattributes").read_text(encoding="utf-8")


def test_store_refuses_oversized_models(tmp_path):
    X, y = _data()
    m = fam.RulesCount(FEATS).fit(X, y)
    with pytest.raises(store.ModelHashError, match="exceeds"):
        store.save(m, {}, tmp_path, max_bytes=10)


# ------------------------------------------------------------------ search space, selection, stops


def test_declared_search_space():
    cfg = drv.load_yaml(drv.CONFIG)
    proto = drv.load_yaml(drv.PROTOCOL)
    specs = drv.enumerate_configs(cfg)
    assert len({s["id"] for s in specs}) == len(specs) == 16
    total = len(specs) + len(cfg["cv"]["sensitivity"])
    assert total == 18 <= proto["data_access"]["tuning"]["max_configs"]
    assert {drv.family_group(s["family"]) for s in specs} == set(drv.CANDIDATE_ORDER)
    assert drv.CANDIDATE_ORDER == ["rules", "logistic", "lightgbm"]
    assert proto["selection"]["simplicity_order"] == [
        "rules_only",
        "logistic_regression",
        "lightgbm",
    ]
    assert 3 + len(cfg["cv"]["sensitivity"]) <= proto["data_access"]["validation"]["max_finalists"]


def _cands(ap):
    return {g: {"id": g, "all": {"pr_auc": ap[g]}} for g in ap}


def test_selection_tie_goes_to_the_simpler_candidate():
    rng = np.random.default_rng(4)
    y = (rng.random(3000) < 0.05).astype(int)
    s0 = rng.normal(size=3000)
    strong = s0 + 3.0 * y
    scores = {"rules": s0 + 0.3 * y, "logistic": strong, "lightgbm": strong.copy()}
    ap = {g: mt.pr_auc(y, v) for g, v in scores.items()}
    sel = drv.select(_cands(ap), y, scores, {"n_boot": 200, "seed": 1, "ci": 0.95})
    assert sel["chosen"] == "logistic"


@pytest.mark.parametrize(
    ("shift", "want"),
    [({"logistic": 0.0, "lightgbm": 0.0}, "rules"),       # equal skill: CI spans 0 -> simplest
     ({"logistic": 2.5, "lightgbm": 0.0}, "logistic"),    # only logistic credibly better
     ({"logistic": 1.5, "lightgbm": 3.0}, "lightgbm")],   # both pass: highest PR-AUC wins
)  # fmt: skip
def test_selection_rule(shift, want):
    rng = np.random.default_rng(3)
    n = 4000
    y = (rng.random(n) < 0.05).astype(int)
    common = rng.normal(size=n)

    def score(d):  # half shared, half own noise: candidates differ by chance as real models do
        return 0.5 * common + 0.866 * rng.normal(size=n) + (0.6 + d) * y

    scores = {"rules": score(0.0), **{g: score(d) for g, d in shift.items()}}
    ap = {g: mt.pr_auc(y, s) for g, s in scores.items()}
    b = {"n_boot": 300, "seed": 1, "ci": 0.95}
    sel = drv.select(_cands(ap), y, scores, b)
    assert sel["chosen"] == want, sel


SR = {"validation_pr_auc_above": 0.95, "top_feature_share_of_abs_contribution_above": 0.5}


def test_stop_rules():
    ok = {"top": [{"feature": "a"}], "top_share": 0.2}
    dom = {"top": [{"feature": "a"}], "top_share": 0.7}
    assert drv.stop_reason(0.4, True, ok, ["a"], ["a", "b"], SR) is None
    assert "implausibly high" in drv.stop_reason(0.96, True, ok, ["a"], ["a"], SR)
    assert "holds 70%" in drv.stop_reason(0.4, True, dom, ["a"], ["a"], SR)
    assert drv.stop_reason(0.4, False, dom, ["a"], ["a"], SR) is None  # rule scores exempt
    assert "registry" in drv.stop_reason(0.4, True, ok, ["account_id"], ["a"], SR)


def test_explain_is_in_log_odds_and_reports_the_top_share():
    X, y = _data()
    m = fam.Logistic(FEATS, params={"C": 1.0, "class_weight": "none", "max_iter": 2000}).fit(X, y)
    ex = drv.explain(m, X, FEATS, 3)
    assert ex["output_space"] == "log-odds margin of the uncalibrated model"
    assert len(ex["top"]) == 3 and 0 < ex["top_share"] <= 1
    assert sum(t["share"] for t in drv.explain(m, X, FEATS, 5)["top"]) == pytest.approx(1.0)


def test_period_loader_refuses_calibration_and_test():
    env = drv.Env(drv.parse(["cv"]))
    for p in ("CALIBRATION", "TEST"):
        with pytest.raises(SystemExit, match="TRAIN and VALIDATION only"):
            env.period(p)


def test_cv_and_learning_curves_pass_the_eval_fold_to_lightgbm(monkeypatch):
    env = drv.Env(drv.parse(["cv"]))
    rng = np.random.default_rng(6)
    cols = env.sets["all"]

    def frame(n):
        X = rng.normal(size=(n, len(cols)))
        y = (rng.random(n) < 0.1).astype(int)
        return pl.DataFrame({"y": y, "account_key": [f"k{i}" for i in range(n)],
                             **{c: X[:, j] for j, c in enumerate(cols)}})  # fmt: skip

    fs = [{"name": "F1", "fit": frame(300), "ev": frame(200), "seen": np.zeros(200, bool)}]
    seen = []
    real = fam.LightGBM.fit

    def spy(self, X, y, eval_set=None):
        seen.append(eval_set is not None and len(eval_set[1]) == 200)
        return real(self, X, y, eval_set=eval_set)

    monkeypatch.setattr(fam.LightGBM, "fit", spy)
    spec = {"id": "x", "family": "lightgbm", "feature_set": "all",
            "params": {"num_leaves": 7, "min_child_samples": 20,
                       "class_weight": "none"}}  # fmt: skip
    env.lgb_fixed["n_estimators"] = 30
    drv.run_config(env, spec, fs)
    drv.learning_curve(env, spec, fs)
    assert seen and all(seen)


def test_validate_refuses_a_saved_model_that_does_not_reproduce_its_scores(p4m, monkeypatch):
    _ensure_cv(p4m)
    _fresh_validation(p4m)
    monkeypatch.setattr(drv, "guard", lambda *a, **k: {"verdict": "pass", "cleared": True})
    real = store.load

    class Off:
        def __init__(self, m):
            self.m = m

        def margin(self, X):
            return self.m.margin(X) + 1e-3

    monkeypatch.setattr(drv.store, "load", lambda h, d: (Off(real(h, d)[0]), {}))
    with pytest.raises(SystemExit, match="does not reproduce"):
        drv.main(["validate", *p4m["args"]])
    _fresh_validation(p4m)


def test_identity_guard_wiring_on_synthetic_days(monkeypatch):
    """guard() feeds the frozen v2 evaluator: real fold days, one row per account per eval day,
    seen = account in that fold's fit rows."""
    from src.eval import identity_guard as ig

    monkeypatch.setattr(drv, "load_v2_config", ig.load_v2_config)  # the frozen folds, unpatched
    env = drv.Env(drv.parse(["validate"]))
    rng = np.random.default_rng(5)
    days = [date(2022, 9, d) for d in (2, 3, 4, 5, 7)]
    accts = [f"B|{i:05d}" for i in range(1500)]
    rows = []
    for d in days:
        for k in rng.choice(accts, 700, replace=False):
            rows.append({"account_key": k, "day": d})
    df = pl.DataFrame(rows)
    n = df.height
    cols = env.sets["all"]
    X = rng.normal(size=(n, len(cols)))
    for j, c in enumerate(cols):
        if c.startswith("fired_"):
            X[:, j] = (rng.random(n) < 0.3).astype(float)
    z = -3.0 + 1.5 * X[:, cols.index(env.rule_cols[0])]
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    df = df.with_columns(
        pl.Series("y", y),
        pl.col("day").map_elements(lambda d: datetime(d.year, d.month, d.day, tzinfo=UTC),
                                   return_dtype=pl.Datetime("us", "UTC")).alias("window_start"),
        *[pl.Series(c, X[:, j]) for j, c in enumerate(cols)],
    )  # fmt: skip
    spec = {"family": "logistic", "params": {"C": 0.1, "class_weight": "none", "max_iter": 2000},
            "feature_set": "all"}  # fmt: skip
    g = drv.guard(env, spec, None, df)
    assert g["version"] == 2 and g["verdict"] in {
        "pass", "no_material_gain", "insufficient_data", "inconclusive", "account_recognition",
        "harms_unseen"}  # fmt: skip
    assert isinstance(g["cleared"], bool)


# ------------------------------------------------------------------ driver on the P2 fixture


@pytest.fixture(scope="module")
def p4m(p2_run, tmp_path_factory):
    from scripts.p3 import run_p3

    r = p2_run["root"]
    out = tmp_path_factory.mktemp("p4m")
    common = ["--variant", "FIX", "--store-root", str(r / "out"),
              "--rules", str(p2_run["cp"].rules),
              "--thresholds", str(p2_run["configs"] / "p2_rule_thresholds.yaml")]  # fmt: skip
    run_p3.main([
        "features", *common, "--interim", str(p2_run["info"]["interim"]),
        "--out-root", str(out / "p3"), "--results", str(out / "p3.json"),
        "--fx", str(p2_run["configs"] / "p2_fx_usd_per_unit.yaml"),
        "--sources", str(p2_run["info"]["sources"]), "--p2-results", str(r / "results.json"),
    ])  # fmt: skip
    import yaml

    cfg = drv.load_yaml(drv.CONFIG)
    cfg["validate"]["bootstrap"]["n_boot"] = 30
    cfg["cv"]["families"]["lightgbm"]["fixed"] |= {"n_estimators": 60}
    cfg["cv"]["families"]["lightgbm"]["grid"]["min_child_samples"] = [2, 5]
    cfg["cv"]["permutation_importance"]["repeats"] = 1
    cfg["validate"]["permutation_importance"]["repeats"] = 1
    strict = out / "models_strict.yaml"  # declared stop rules (fixture PR-AUC is 1.0 -> stop)
    strict.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    cfg["validate"]["stop_rules"]["validation_pr_auc_above"] = 1.01  # fixture only: reach the
    cpath = out / "models.yaml"  # guard and the save path (the tiny fixture separates perfectly)
    cpath.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    mp = pytest.MonkeyPatch()
    folds = [
        {"name": "F1", "fit_days": ["2022-09-02", "2022-09-03"], "eval_day": "2022-09-04"},
        {"name": "F2", "fit_days": ["2022-09-02", "2022-09-03", "2022-09-04"],
         "eval_day": "2022-09-05"},
    ]  # fmt: skip
    mp.setattr(drv, "load_v2_config", lambda: {"folds": folds})
    asked = []
    for name in ("load_fit_labels", "load_fit_strata"):
        real = getattr(drv, name)

        def spy(periods, *a, _real=real, **k):
            asked.append(list(periods))
            return _real(periods, *a, **k)

        mp.setattr(drv, name, spy)
    args = [*common, "--config", str(cpath), "--features-root", str(out / "p3"),
            "--p3-results", str(out / "p3.json"), "--cv-results", str(out / "cv.json"),
            "--results", str(out / "val.json"), "--model-card", str(out / "card.md"),
            "--model-dir", str(out / "models"),
            "--precheck-results", str(out / "pre.json")]  # fmt: skip
    yield {"args": args, "out": out, "asked": asked, "mp": mp, "cpath": cpath, "strict": strict}
    mp.undo()


def _ensure_cv(p4m):
    if not (p4m["out"] / "cv.json").exists():
        assert drv.main(["cv", *p4m["args"]]) == 0


def _fresh_validation(p4m):
    v = p4m["out"] / "val.json"
    if v.exists():
        v.unlink()


def test_cv_stage_on_fixture(p4m):
    p4m["asked"].clear()
    assert drv.main(["cv", *p4m["args"]]) == 0
    doc = json.loads((p4m["out"] / "cv.json").read_text(encoding="utf-8"))
    assert doc["inputs"]["periods_read"] == ["TRAIN"]
    assert doc["inputs"]["n_configs"] == 18
    roles = [f["role"] for f in doc["finalists"]]
    assert roles == ["candidate"] * 3 + ["sensitivity"] * 2
    assert [drv.family_group(f["family"]) for f in doc["finalists"][:3]] == drv.CANDIDATE_ORDER
    assert all(f["feature_set"] == "no_format" for f in doc["finalists"][3:])
    lgb_f = next(f for f in doc["finalists"] if f["family"] == "lightgbm")
    assert 1 <= lgb_f["n_estimators"] < 60  # early-stopped on the eval fold (cap 60 here)
    assert all(p == ["TRAIN"] for p in p4m["asked"])
    assert {"logistic", "lightgbm"} == set(doc["learning_curves"])


def test_validate_stops_on_implausibly_high_pr_auc_with_the_declared_rule(p4m):
    o = p4m["out"]
    over = {"--config": p4m["strict"], "--cv-results": o / "cv_strict.json",
            "--results": o / "val_strict.json", "--model-dir": o / "models_strict"}  # fmt: skip
    args = list(p4m["args"])
    for i in range(0, len(args) - 1):
        if args[i] in over:
            args[i + 1] = str(over[args[i]])
    assert drv.main(["cv", *args]) == 0
    assert drv.main(["validate", *args]) == 2
    doc = json.loads((o / "val_strict.json").read_text(encoding="utf-8"))
    assert "implausibly high" in doc["stopped"] and doc["model_sha256"] is None
    assert not (o / "models_strict").exists()


def test_validate_stops_when_the_guard_objects_and_saves_nothing(p4m, monkeypatch):
    _ensure_cv(p4m)
    _fresh_validation(p4m)
    monkeypatch.setattr(drv, "guard", lambda *a, **k: {"verdict": "account_recognition",
                                                     "cleared": False})  # fmt: skip
    p4m["asked"].clear()
    mdir = p4m["out"] / "models"
    before = set(mdir.iterdir()) if mdir.exists() else set()
    code = drv.main(["validate", *p4m["args"]])
    doc = json.loads((p4m["out"] / "val.json").read_text(encoding="utf-8"))
    chosen_family = doc["finalists"][doc["chosen"]]["family"]
    if chosen_family in ("logistic", "lightgbm"):
        assert code == 2 and "identity guard" in doc["stopped"]
        assert doc["model_sha256"] is None
        after = set(mdir.iterdir()) if mdir.exists() else set()
        assert after == before  # nothing saved
    else:  # rule score chosen on the fixture: the guard does not apply
        assert doc["identity_guard"]["verdict"] == "not_applicable"
    assert {tuple(p) for p in p4m["asked"]} <= {("TRAIN",), ("VALIDATION",)}
    (p4m["out"] / "val.json").unlink()


def test_validate_end_to_end_saves_by_hash_and_refuses_reopening(p4m, p2_run, monkeypatch):
    _ensure_cv(p4m)
    _fresh_validation(p4m)
    monkeypatch.setattr(drv, "guard", lambda *a, **k: {"verdict": "pass", "cleared": True})
    p4m["asked"].clear()
    assert drv.main(["validate", *p4m["args"]]) == 0
    doc = json.loads((p4m["out"] / "val.json").read_text(encoding="utf-8"))
    assert doc["inputs"]["periods_read"] == ["TRAIN", "VALIDATION"]
    assert {tuple(p) for p in p4m["asked"]} <= {("TRAIN",), ("VALIDATION",)}
    assert len(doc["finalists"]) == 5 and doc["chosen"] in doc["finalists"]
    for r in doc["finalists"].values():
        assert {"all", "seen", "unseen"} <= set(r) and "pr_auc" in r["all"]["ci"]
        assert r["seen"]["n"] + r["unseen"]["n"] == r["all"]["n"]
        assert r["seen"]["n"] > 0 and r["unseen"]["n"] > 0  # fixture: 24 seen, 5 unseen
        assert (r["all"]["brier"] is not None) == r["raw_score_is_probability"]
        if r["raw_score_is_probability"]:
            assert 0 <= r["all"]["brier"] <= 0.25
    h = doc["model_sha256"]
    model, meta = store.load(h, p4m["out"] / "models")
    assert meta["explain_output_space"] == "log-odds margin of the uncalibrated model"
    assert meta["recalibration_required_in_p5"] is True
    assert set(doc["operating_points"]) == {"20", "5", "50"}
    card = (p4m["out"] / "card.md").read_text(encoding="utf-8")
    assert h in card and "log-odds" in card
    # aggregates only
    text = (p4m["out"] / "val.json").read_text(encoding="utf-8")
    alerts = pl.read_parquet(p2_run["base"] / "runtime" / "alerts.parquet")
    for v in alerts["alert_id"].to_list() + alerts["account_key"].to_list():
        assert v not in text
    # VALIDATION is opened once
    with pytest.raises(SystemExit, match="opened once"):
        drv.main(["validate", *p4m["args"]])


def test_validate_refuses_configs_changed_after_cv(p4m):
    _ensure_cv(p4m)
    _fresh_validation(p4m)
    original = p4m["cpath"].read_text(encoding="utf-8")
    try:
        p4m["cpath"].write_text(original + "\n# edited after cv\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="changed after the CV run"):
            drv.main(["validate", *p4m["args"]])
    finally:
        p4m["cpath"].write_text(original, encoding="utf-8")


# ------------------------------------------------------------------ precheck (TRAIN only)


@pytest.mark.parametrize(
    ("worst", "want"),
    [(0.10, "clear"), (0.399, "clear"), (0.40, "near_stop_rule"), (0.50, "near_stop_rule"),
     (0.501, "stop_rule_would_fire")],
)  # fmt: skip
def test_precheck_reading(worst, want):
    assert drv.precheck_reading(worst, 0.40, 0.50) == want


def test_precheck_stop_level_equals_the_declared_validate_stop_rule():
    pc = drv.load_yaml(drv.PRECHECK_CONFIG)
    cfg = drv.load_yaml(drv.CONFIG)
    assert (
        pc["stop"] == cfg["validate"]["stop_rules"]["top_feature_share_of_abs_contribution_above"]
    )
    assert pc["near_stop"] < pc["stop"]


def test_precheck_on_fixture_reads_train_only_and_never_opens_validation(p4m):
    _ensure_cv(p4m)
    _fresh_validation(p4m)
    p4m["asked"].clear()
    assert drv.main(["precheck", *p4m["args"]]) == 0
    doc = json.loads((p4m["out"] / "pre.json").read_text(encoding="utf-8"))
    assert doc["inputs"]["periods_read"] == ["TRAIN"]
    assert p4m["asked"] and all(p == ["TRAIN"] for p in p4m["asked"])
    assert not (p4m["out"] / "val.json").exists()
    lf = doc["learned_finalists"]
    assert {v["family"] for v in lf.values()} == {"logistic", "lightgbm"}
    assert sorted(v["role"] for v in lf.values()) == ["candidate"] * 2 + ["sensitivity"] * 2
    for v in lf.values():
        assert [p["fold"] for p in v["per_fold"]] == ["F1", "F2"]
        for p in v["per_fold"]:
            assert 0 < p["top_share"] <= 1
            assert p["top"][0]["share"] == pytest.approx(p["top_share"])
        assert v["max_top_share"] == max(p["top_share"] for p in v["per_fold"])
    cand = max(v["max_top_share"] for v in lf.values() if v["role"] == "candidate")
    assert doc["worst_candidate_top_share"] == cand
    assert doc["reading"] == drv.precheck_reading(cand, 0.40, 0.50)
    assert set(doc["rules_weighted"]) == {"F1", "F2"}
    for v in doc["rules_weighted"].values():
        assert all(k.startswith("fired_") for k in v["rules"])


def test_precheck_refuses_configs_changed_after_cv(p4m):
    _ensure_cv(p4m)
    original = p4m["cpath"].read_text(encoding="utf-8")
    try:
        p4m["cpath"].write_text(original + "\n# edited after cv\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="changed after the CV run"):
            drv.main(["precheck", *p4m["args"]])
    finally:
        p4m["cpath"].write_text(original, encoding="utf-8")
