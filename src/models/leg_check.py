"""P4 step 0b: does any per-payment signal exist for NEW accounts? Label-free module: payment and
alert labels arrive as arguments (scripts/p4/leg_check.py gets them from src.eval). Protocol and
verdict: configs/p4_leg_check.yaml (declared before the real run).

Per fold: a payment model is fitted on the fit days' legs (payment label = laundering or not); each
eval-day alert is scored by its MAXIMUM payment score; on UNSEEN accounts that alert score is
compared, paired, with the step-0 baseline alert model. AUC is a ranking diagnostic here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from src.models import diagnostics as dg

ModelFactory = Callable[[], object]


@dataclass
class LegFold:
    name: str
    X_fit: np.ndarray
    y_fit: np.ndarray  # payment labels, fit days
    w_fit: np.ndarray  # 1 / sampling rate
    X_eval: np.ndarray  # every leg of every eval-day alert
    leg_alert: np.ndarray  # eval leg -> alert row index
    y_leg_eval: np.ndarray  # payment labels of eval legs (within-alert metrics only)
    y_alert: np.ndarray
    seen: np.ndarray  # bool per eval alert
    baseline: np.ndarray  # step-0 alert model score per eval alert
    strata: dict[str, np.ndarray] = field(default_factory=dict)  # positives per stratum

    def __post_init__(self) -> None:
        n = len(self.y_alert)
        if not (len(self.seen) == len(self.baseline) == n):
            raise ValueError(f"{self.name}: alert arrays differ in length")
        if len(self.leg_alert) != len(self.X_eval) or len(self.y_leg_eval) != len(self.X_eval):
            raise ValueError(f"{self.name}: eval leg arrays differ in length")
        if len(self.X_fit) != len(self.y_fit) or len(self.w_fit) != len(self.y_fit):
            raise ValueError(f"{self.name}: fit arrays differ in length")
        if np.bincount(self.leg_alert, minlength=n).min() == 0:
            raise ValueError(f"{self.name}: an eval alert has no leg")
        # P2-v1: an alert is positive iff >= 1 of its payments is laundering
        mx = np.zeros(n, dtype=np.int64)
        np.maximum.at(mx, self.leg_alert, self.y_leg_eval.astype(np.int64))
        if (mx != self.y_alert).any():
            raise ValueError(f"{self.name}: alert labels disagree with their payment labels")


def max_by_alert(scores: np.ndarray, leg_alert: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, -np.inf)
    np.maximum.at(out, leg_alert, scores)
    return out


def leg_scores(f: LegFold, factory: ModelFactory, cols: np.ndarray | None = None) -> np.ndarray:
    Xf = f.X_fit if cols is None else f.X_fit[:, cols]
    Xe = f.X_eval if cols is None else f.X_eval[:, cols]
    m = factory()
    m.fit(Xf, f.y_fit, sample_weight=f.w_fit)
    return m.predict_proba(Xe)[:, 1]


def decide(diff_ci: list[float] | None, material: float) -> str:
    if diff_ci is None:
        return "insufficient_data"
    if diff_ci[0] >= material:
        return "signal"
    if diff_ci[1] < material:
        return "no_signal"
    return "inconclusive"


def compare(
    folds: list[LegFold],
    alert_scores: dict[str, np.ndarray],
    n_boot: int,
    seed: int,
    ci: float,
    material: float,
    min_pos: int,
) -> dict:
    """Pooled positive-weighted AUC on UNSEEN alerts: payment-max vs baseline, paired bootstrap
    over unseen eval accounts (one row per account per eval day), within fold."""
    idx = {f.name: np.flatnonzero(~f.seen) for f in folds}
    w = [int(f.y_alert[idx[f.name]].sum()) for f in folds]

    def pooled(sel: dict[str, np.ndarray]):
        a, b, ws = [], [], []
        for f in folds:
            i = sel[f.name]
            a.append(dg.auc(f.y_alert[i], alert_scores[f.name][i]))
            b.append(dg.auc(f.y_alert[i], f.baseline[i]))
            ws.append(int(f.y_alert[i].sum()))
        return dg.pooled(a, ws), dg.pooled(b, ws)

    pm, pb = pooled(idx)
    rng = np.random.default_rng(seed)
    rep = np.full(n_boot, np.nan)
    for r in range(n_boot):
        sel = {k: rng.choice(v, len(v), replace=True) for k, v in idx.items()}
        a, b = pooled(sel)
        if a is not None and b is not None:
            rep[r] = a - b
    per_fold = [
        {
            "fold": f.name,
            "n_unseen": int(len(idx[f.name])),
            "pos_unseen": wf,
            "auc_payment_max": dg.auc(f.y_alert[idx[f.name]], alert_scores[f.name][idx[f.name]]),
            "auc_baseline": dg.auc(f.y_alert[idx[f.name]], f.baseline[idx[f.name]]),
        }
        for f, wf in zip(folds, w, strict=True)
    ]
    out = {
        "per_fold": per_fold,
        "n_unseen_pos": int(sum(w)),
        "auc_payment_max": pm,
        "auc_baseline": pb,
        "gain": None if pm is None or pb is None else pm - pb,
        "gain_ci": dg.percentile_ci(rep, ci),
        "n_boot_undefined": int(np.isnan(rep).sum()),
    }
    if out["n_unseen_pos"] < min_pos or out["n_boot_undefined"] > 0.01 * n_boot:
        out["verdict"] = "insufficient_data"
    else:
        out["verdict"] = decide(out["gain_ci"], material)
    return out


def within_alert(folds: list[LegFold], leg_scores_by_fold: dict[str, np.ndarray]) -> dict:
    """Unseen POSITIVE alerts with >= 1 normal payment: is the top-scored payment a laundering one?
    Compared with the chance rate (laundering share of the alert's payments). Ties at the top are
    split evenly. Also the mean within-alert AUC (laundering vs normal payments)."""
    hits, chance, aucs, n = 0.0, 0.0, [], 0
    for f in folds:
        s = leg_scores_by_fold[f.name]
        pos = np.flatnonzero(~f.seen & (f.y_alert == 1))
        order = np.argsort(f.leg_alert, kind="stable")
        bounds = np.searchsorted(f.leg_alert[order], np.arange(len(f.y_alert) + 1))
        for a in pos:
            legs = order[bounds[a] : bounds[a + 1]]
            y, sc = f.y_leg_eval[legs], s[legs]
            if y.sum() == len(y):
                continue
            top = sc == sc.max()
            hits += y[top].mean()
            chance += y.mean()
            aucs.append(dg.auc(y, sc))
            n += 1
    return {
        "n_alerts": n,
        "top_payment_hit_rate": hits / n if n else None,
        "chance_rate": chance / n if n else None,
        "mean_within_alert_auc": float(np.mean(aucs)) if aucs else None,
    }


def strata(
    folds: list[LegFold], alert_scores: dict[str, np.ndarray], min_pos: int
) -> dict[str, dict]:
    """Each stratum's unseen positives vs all unseen negatives: payment-max and baseline AUC."""
    names = sorted({k for f in folds for k in f.strata})
    out = {}
    for k in names:
        a, b, w = [], [], []
        for f in folds:
            u = ~f.seen
            m = f.strata.get(k, np.zeros_like(u)) & u
            i = np.flatnonzero(m | (u & (f.y_alert == 0)))
            ok = m.any()
            a.append(dg.auc(f.y_alert[i], alert_scores[f.name][i]) if ok else None)
            b.append(dg.auc(f.y_alert[i], f.baseline[i]) if ok else None)
            w.append(int(m.sum()))
        n = int(sum(w))
        out[k] = {
            "n_pos_unseen": n,
            "auc_payment_max": dg.pooled(a, w) if n >= min_pos else None,
            "auc_baseline": dg.pooled(b, w) if n >= min_pos else None,
        }
    return out
