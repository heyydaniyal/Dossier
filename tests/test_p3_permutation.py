"""Shuffled-label test machinery (src/models/permutation.py) on known-answer synthetic data.

The real-data run lives in scripts/p3/run_p3.py (Dani's laptop) and its result is asserted by
tests/test_p3_results_facts.py. Here we prove the machinery itself:
  - a clean pipeline passes: null mean ~ prevalence, real model far outside the null;
  - a LEAKING pipeline (naive target encoding that uses the eval day's real labels) is caught:
    its null sits far above the prevalence;
  - permutation keeps every block's prevalence; results are deterministic in the seed.
"""

from __future__ import annotations

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

from src.models.permutation import Fold, permute_within_blocks, run
from tests.test_p3_features import ROOT

CFG = yaml.safe_load((ROOT / "configs" / "p3_features.yaml").read_text(encoding="utf-8"))
BOUND = CFG["permutation_test"]["pass_if"]["null_mean_over_prevalence_max"]


def _world(seed: int = 0):
    """Two folds shaped like F1/F2 (scaled down): fit days -> one eval day; signal in x0."""
    rng = np.random.default_rng(seed)
    folds, data = [], {}
    for name, n_fit_days, n_eval in (("F1", 1, 1500), ("F2", 2, 2000)):
        nf = 1200 * n_fit_days
        y_fit = (rng.random(nf) < 0.05).astype(int)
        y_eval = (rng.random(n_eval) < 0.06).astype(int)
        y_eval[:5] = 1
        x_fit = rng.normal(size=(nf, 4))
        x_fit[:, 0] += 1.5 * y_fit  # noqa: E702
        x_eval = rng.normal(size=(n_eval, 4))
        x_eval[:, 0] += 1.5 * y_eval  # noqa: E702
        cat_fit = rng.integers(0, 300, nf)  # a high-cardinality "account-like" category
        cat_eval = rng.integers(0, 300, n_eval)
        blocks = np.repeat(np.arange(n_fit_days), 1200)
        folds.append(Fold(name, y_fit, blocks, y_eval))
        data[name] = (x_fit, x_eval, cat_fit, cat_eval)
    return folds, data


def _model():
    return LogisticRegression(max_iter=500)


def test_clean_pipeline_passes():
    folds, data = _world()
    clean = lambda f, y: data[f.name][:2]  # noqa: E731  (features ignore labels)
    r = run(folds, clean, _model, n_permutations=60, seed=1, null_mean_over_prevalence_max=BOUND)
    assert r["passed"], r["checks"]
    assert 0.9 < r["null_mean_over_prevalence"] < 1.15
    assert r["real_over_prevalence"] > 3 and r["empirical_p"] == 1 / 61


def test_leaking_pipeline_is_caught():
    """Naive target encoding computed on fit AND eval rows with the eval day's REAL labels: the
    classic leak. Its null distribution stays far above the prevalence -> the test fails."""
    folds, data = _world()

    def leaky(f, y_fit):
        x_fit, x_eval, c_fit, c_eval = data[f.name]
        cats = np.concatenate([c_fit, c_eval])
        ys = np.concatenate([y_fit, f.y_eval])  # LEAK: eval labels used to encode eval rows
        enc = np.bincount(cats, weights=ys, minlength=300) / np.maximum(
            np.bincount(cats, minlength=300), 1
        )
        return (np.c_[x_fit[:, 1:], enc[c_fit]], np.c_[x_eval[:, 1:], enc[c_eval]])

    r = run(folds, leaky, _model, n_permutations=40, seed=1, null_mean_over_prevalence_max=BOUND)
    assert not r["checks"]["null_mean_close_to_prevalence"]
    assert r["null_mean_over_prevalence"] > 2
    assert not r["passed"]


def test_permutation_preserves_block_prevalence():
    rng = np.random.default_rng(3)
    y = (rng.random(1000) < 0.1).astype(int)
    blocks = np.repeat([0, 1, 2, 3], 250)
    p = permute_within_blocks(y, blocks, rng)
    for b in range(4):
        assert p[blocks == b].sum() == y[blocks == b].sum()
    assert not np.array_equal(p, y)


def test_deterministic_in_seed():
    folds, data = _world()
    clean = lambda f, y: data[f.name][:2]  # noqa: E731
    a = run(folds, clean, _model, n_permutations=10, seed=7, null_mean_over_prevalence_max=BOUND)
    b = run(folds, clean, _model, n_permutations=10, seed=7, null_mean_over_prevalence_max=BOUND)
    assert a == b


def test_declared_criteria_match_the_simulation():
    """Pass criteria are fixed in the config before the real run (validated 2026-10-06)."""
    pt = CFG["permutation_test"]
    assert pt["n_permutations"] >= 100
    assert pt["folds"] == ["F1", "F2"] and pt["permute_within"] == "day"
    assert pt["pass_if"] == {"null_mean_over_prevalence_max": 1.15, "real_above_null_max": True}
