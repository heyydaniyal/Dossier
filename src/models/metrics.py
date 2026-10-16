"""P4 evaluation metrics under class imbalance. Label-free module: labels arrive as arguments.

Headline: PR-AUC (average precision). Also precision at fixed recall, recall at a review capacity,
Brier score, a cost-based operating point; ROC-AUC is secondary only. Bootstrap CIs resample rows;
callers pass one row per account per day (VALIDATION is one day), so this resamples accounts.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _ok(y: np.ndarray) -> bool:
    return 0 < int(np.sum(y)) < len(y)


def pr_auc(y, s) -> float | None:
    return float(average_precision_score(y, s)) if _ok(y) else None


def roc_auc(y, s) -> float | None:
    return float(roc_auc_score(y, s)) if _ok(y) else None


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def precision_at_recall(y, s, r: float) -> float | None:
    """Highest precision among thresholds whose recall is >= r."""
    if not _ok(y):
        return None
    prec, rec, _ = precision_recall_curve(y, s)
    ok = rec >= r - 1e-12
    return float(prec[ok].max()) if ok.any() else None


def recall_at_capacity(y, s, share: float) -> float | None:
    """Recall when the top ceil(share * n) alerts by score are reviewed. Ties at the cut are split
    in proportion (expected recall under random tie-breaking), so the value never depends on row
    order."""
    y, s = np.asarray(y), np.asarray(s, float)
    npos = int(y.sum())
    if npos == 0:
        return None
    k = math.ceil(share * len(s))
    if k <= 0:
        return 0.0
    t = np.sort(s)[::-1][k - 1]
    above = s > t
    tied = s == t
    room = k - int(above.sum())
    tp = int(y[above].sum()) + (room / int(tied.sum())) * int(y[tied].sum())
    return float(tp / npos)


def cost_threshold(y, s, ratio: float) -> dict:
    """Threshold t (flag iff s >= t) minimising ratio * FN + FP. Candidate thresholds are the
    observed scores plus 'flag nothing'. Ties in cost -> the higher threshold (fewer alerts)."""
    y, s = np.asarray(y), np.asarray(s, float)
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    npos = int(y.sum())
    # flagging the first i rows, at boundaries between distinct scores
    tp = np.concatenate([[0], np.cumsum(ys)])
    fp = np.concatenate([[0], np.cumsum(1 - ys)])
    idx = [0] + [i for i in range(1, len(ss) + 1) if i == len(ss) or ss[i] != ss[i - 1]]
    best = None
    for i in idx:
        fn, f = npos - tp[i], fp[i]
        c = ratio * fn + f
        if best is None or c < best[0] - 1e-9:
            best = (c, i)
    c, i = best
    thr = float("inf") if i == 0 else float(ss[i - 1])
    return {
        "threshold": thr,
        "n_flagged": int(i),
        "tp": int(tp[i]),
        "fp": int(fp[i]),
        "fn": int(npos - tp[i]),
        "recall": float(tp[i] / npos) if npos else None,
        "precision": float(tp[i] / i) if i else None,
        "cost": float(c),
        "ratio": ratio,
    }


def percentile_ci(rep: np.ndarray, level: float) -> list[float] | None:
    rep = np.asarray(rep, float)
    rep = rep[np.isfinite(rep)]
    if not len(rep):
        return None
    a = (1 - level) / 2
    return [float(np.quantile(rep, a)), float(np.quantile(rep, 1 - a))]


def bootstrap(
    metric: Callable[[np.ndarray], float | None], n: int, n_boot: int, seed: int, level: float
) -> tuple[list[float] | None, int]:
    """metric(idx) on resampled row indices; returns (CI, number of undefined replicates)."""
    rng = np.random.default_rng(seed)
    rep = np.full(n_boot, np.nan)
    for b in range(n_boot):
        v = metric(rng.integers(0, n, n))
        if v is not None:
            rep[b] = v
    return percentile_ci(rep, level), int(np.isnan(rep).sum())


def summary(y, s, capacity: list[float], recalls: list[float], probability: bool) -> dict:
    """Point metrics for one subset."""
    y, s = np.asarray(y), np.asarray(s, float)
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "pr_auc": pr_auc(y, s),
        "roc_auc": roc_auc(y, s),
        **{f"precision_at_recall_{r}": precision_at_recall(y, s, r) for r in recalls},
        **{f"recall_at_capacity_{c}": recall_at_capacity(y, s, c) for c in capacity},
    }
    out["brier"] = brier(y, s) if probability else None
    return out
