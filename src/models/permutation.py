"""Shuffled-label (permutation) leakage test. Label-free module: labels arrive as arguments.

Procedure (configs/p3_features.yaml -> permutation_test):
  for each permutation b: shuffle ALL labels within temporal blocks (each fit day, and the eval
  day), so every block keeps its prevalence; rebuild the fold's features through the SAME pipeline
  callable, which sees only the shuffled labels (our pipeline ignores them; a leaking one -- naive
  target encoding, eval rows reused in the fit -- would not); refit the fixed model; score the eval
  day; compute AP against the SHUFFLED eval labels.
The statistic is the positive-weighted mean of per-fold AP. If the pipeline is clean, shuffled
labels are independent of everything the model sees, so the null sits at chance (~prevalence);
a null well above it means label information reaches the features or the model by another route.

Why eval labels are shuffled too (found 2026-10-06 by tests/test_p3_permutation.py, BEFORE any
real run): with only the fit labels shuffled and real eval labels, a model fitted on noise is a
random function of INFORMATIVE features, and AP is convex in the correlation, so a clean pipeline's
null sat at 1.38x prevalence (random scores: 1.05x) -- a false alarm, not a leak.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class Fold:
    name: str
    y_fit: np.ndarray  # 0/1
    blocks_fit: np.ndarray  # temporal block id per fit row (day)
    y_eval: np.ndarray  # 0/1; the eval day is one block


# pipeline(fold, y_fit) -> (X_fit, X_eval). The real one ignores both y_fit and fold.y_eval; in a
# permutation round `fold` carries the SHUFFLED eval labels, so a pipeline that peeks at them is
# caught.
Pipeline = Callable[[Fold, np.ndarray], tuple[np.ndarray, np.ndarray]]
ModelFactory = Callable[[], object]


def permute_within_blocks(
    y: np.ndarray, blocks: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    out = y.copy()
    for b in np.unique(blocks):
        idx = np.flatnonzero(blocks == b)
        out[idx] = y[rng.permutation(idx)]
    return out


def _stat(folds: list[Fold], preds: list[np.ndarray]) -> float:
    w = np.array([f.y_eval.sum() for f in folds], dtype=float)
    ap = np.array([average_precision_score(f.y_eval, p) for f, p in zip(folds, preds, strict=True)])
    return float(np.dot(w / w.sum(), ap))


def _fit_predict(folds, pipeline, model_factory) -> list[np.ndarray]:
    preds = []
    for f in folds:
        y = f.y_fit
        x_fit, x_eval = pipeline(f, y)
        if len(np.unique(y)) < 2:
            preds.append(np.zeros(len(f.y_eval)))
            continue
        m = model_factory()
        m.fit(x_fit, y)
        preds.append(m.predict_proba(x_eval)[:, 1])
    return preds


def run(
    folds: list[Fold],
    pipeline: Pipeline,
    model_factory: ModelFactory,
    n_permutations: int,
    seed: int,
    null_mean_over_prevalence_max: float,
    real_above_null_max: bool = True,
) -> dict:
    for f in folds:
        if not (f.y_eval.sum() > 0 and (1 - f.y_eval).sum() > 0):
            raise ValueError(f"fold {f.name}: eval day needs both classes")
        if len(f.y_fit) != len(f.blocks_fit):
            raise ValueError(f"fold {f.name}: one block id per fit row")
    real = _stat(folds, _fit_predict(folds, pipeline, model_factory))
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for b in range(n_permutations):
        shuffled = [
            replace(
                f,
                y_fit=permute_within_blocks(f.y_fit, f.blocks_fit, rng),
                y_eval=f.y_eval[rng.permutation(len(f.y_eval))],
            )
            for f in folds
        ]
        null[b] = _stat(shuffled, _fit_predict(shuffled, pipeline, model_factory))
    w = np.array([f.y_eval.sum() for f in folds], dtype=float)
    prevalence = float(np.dot(w / w.sum(), [f.y_eval.mean() for f in folds]))
    p = (1 + int((null >= real).sum())) / (n_permutations + 1)
    checks = {
        "null_mean_close_to_prevalence": bool(
            null.mean() <= null_mean_over_prevalence_max * prevalence
        ),
        "real_above_null_max": bool(real > null.max()) if real_above_null_max else True,
    }
    return {
        "real_statistic": real,
        "prevalence": prevalence,
        "null_mean": float(null.mean()),
        "null_mean_over_prevalence": float(null.mean() / prevalence),
        "null_p99": float(np.quantile(null, 0.99)),
        "null_max": float(null.max()),
        "real_over_prevalence": real / prevalence,
        "empirical_p": p,
        "n_permutations": n_permutations,
        "per_fold": [
            {
                "name": f.name,
                "n_fit": int(len(f.y_fit)),
                "pos_fit": int(f.y_fit.sum()),
                "n_eval": int(len(f.y_eval)),
                "pos_eval": int(f.y_eval.sum()),
            }
            for f in folds
        ],  # fmt: skip
        "checks": checks,
        "passed": all(checks.values()),
    }
