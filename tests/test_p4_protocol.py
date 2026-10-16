"""P4 protocol (configs/p4_protocol.yaml): the declared rules are consistent with the frozen split,
and the LightGBM properties the protocol relies on actually hold for the pinned version."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from importlib.metadata import version

import lightgbm as lgb
import numpy as np
import pytest
import yaml
from sklearn.metrics import average_precision_score

from src.data.periods import ROOT, load_split
from src.eval.identity_guard import load_v2_config

PROTOCOL = yaml.safe_load((ROOT / "configs" / "p4_protocol.yaml").read_text(encoding="utf-8"))


def _day(d: str) -> datetime:
    return datetime.fromisoformat(d).replace(tzinfo=UTC)


def test_tuning_folds_live_entirely_inside_train():
    split = load_split()
    folds = {f["name"]: f for f in load_v2_config()["folds"]}
    names = PROTOCOL["data_access"]["tuning"]["folds"]
    assert names, "no tuning folds declared"
    for n in names:
        f = folds[n]
        for d in [*f["fit_days"], f["eval_day"]]:
            assert split.period_of(_day(d)) == "TRAIN", (n, d)
        assert _day(f["eval_day"]) > max(_day(d) for d in f["fit_days"])  # eval is time-later


def test_access_rules():
    a = PROTOCOL["data_access"]
    assert a["fit"] == ["TRAIN"]
    assert set(a["never"]) == {"CALIBRATION", "TEST"}
    assert a["validation"]["opened"] == "once"
    assert 1 <= a["validation"]["max_finalists"] <= 5
    assert a["tuning"]["max_configs"] <= 20
    assert "random" not in a["tuning"]["early_stopping"].split("never")[0]


def test_selection_rule_starts_from_the_simplest_model():
    s = PROTOCOL["selection"]
    assert s["simplicity_order"][0] == "rules_only"
    assert s["metric"] == "pr_auc"
    assert set(s["report_for"]) == {"all", "seen", "unseen"}


def test_pinned_version_matches_installed_and_pyproject():
    want = PROTOCOL["library"]["version"]
    assert version("lightgbm") == want
    py = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(rf'"lightgbm=={re.escape(want)}"', py)


def test_task_10_deviation_is_recorded():
    d = PROTOCOL["deviations"]["task_10_li_small"]
    assert d["decision"] == "not done in P4"
    p1 = yaml.safe_load((ROOT / "configs" / "p1_data_decisions.yaml").read_text(encoding="utf-8"))
    assert p1["variants"]["cross_variant"] == "LI-Medium"
    src = yaml.safe_load((ROOT / "configs" / "data_sources.yaml").read_text(encoding="utf-8"))
    assert not any(k.startswith("LI-Small") for k in src["files"])


# ------------------------------------------------------------------ LightGBM properties relied on


def _data(seed: int = 0, n: int = 6000):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12))
    X[rng.random(X.shape) < 0.3] = np.nan  # NaN handled natively (P3 features use NaN)
    z = -3.5 + 1.5 * np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 1]) * np.clip(X[:, 2], -2, 2)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    k = n * 2 // 3
    return X[:k], y[:k], X[k:], y[k:]


def _params(threads: int) -> dict:
    p = dict(PROTOCOL["library"]["fixed_params"])
    p.update(num_threads=threads, random_state=PROTOCOL["library"]["seed"], learning_rate=0.05,
             num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=1.0, n_estimators=600)  # fmt: skip
    return p


def _fit(threads: int):
    Xf, yf, Xe, ye = _data()
    m = lgb.LGBMClassifier(**_params(threads))
    m.fit(Xf, yf, eval_X=(Xe,), eval_y=(ye,), eval_metric="average_precision",
          callbacks=[lgb.early_stopping(40, first_metric_only=True, verbose=False)])  # fmt: skip
    return m, Xe, ye


@pytest.fixture(scope="module")
def fitted():
    return {t: _fit(t) for t in (1, 2)}


def test_identical_predictions_across_thread_counts(fitted):
    (m1, Xe, _), (m2, _, _) = fitted[1], fitted[2]
    assert m1.best_iteration_ == m2.best_iteration_
    np.testing.assert_array_equal(m1.predict_proba(Xe, raw_score=True),
                                  m2.predict_proba(Xe, raw_score=True))  # fmt: skip


def test_early_stopping_is_driven_by_the_given_time_later_fold(fitted):
    """best_iteration_ = argmax over iterations of AP on the eval fold we passed in (not a random
    internal hold-out)."""
    m, Xe, ye = fitted[1]
    last = m.booster_.current_iteration()
    ap = [average_precision_score(ye, m.predict_proba(Xe, raw_score=True, num_iteration=i))
          for i in range(1, last + 1)]  # fmt: skip
    assert m.best_iteration_ == int(np.argmax(ap)) + 1


def test_native_shap_is_additive_in_the_log_odds_margin(fitted):
    m, Xe, _ = fitted[1]
    c = m.predict_proba(Xe[:300], pred_contrib=True)
    raw = m.predict_proba(Xe[:300], raw_score=True)
    assert c.shape == (300, Xe.shape[1] + 1)  # one column per feature + expected value
    np.testing.assert_allclose(c.sum(axis=1), raw, atol=1e-9)
