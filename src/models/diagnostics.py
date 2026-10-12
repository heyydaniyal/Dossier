"""P4 step 0: diagnosis of the seen/unseen gap (open risk 36). Label-free module: labels, account
keys and strata arrive as arguments from the driver (scripts/p4/diagnose.py), which gets them from
src.eval.training_labels. Protocol and verdict rules: configs/p4_diagnosis.yaml (declared before
the real run).

Three questions, each of which changes what P4 does next:
  Q-A recognition_probe  Is the high AUC on SEEN accounts (alerted in the fit rows) recognition of
                         those accounts, or a population difference? Arms: full / disjoint (the seen
                         accounts' own fit rows removed) / control (the same number of rows and
                         positives removed from accounts absent from the eval day), plus a pure
                         label-memory score.
  Q-B univariate         Does any single feature carry signal on UNSEEN accounts?
  Q-C strata_auc         Which unseen positives does the model rank (laundering share, typology,
                         fan-in vs other rules)?

AUC here is a diagnostic of ranking (as in the P3 guard), not a headline metric (P4 headlines are
PR-AUC). One row per account per eval day is required (the bootstrap resamples accounts).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

ModelFactory = Callable[[int], object]  # seed -> unfitted classifier with fit / predict_proba


@dataclass
class FoldData:
    name: str
    X_fit: np.ndarray
    y_fit: np.ndarray
    acct_fit: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    acct_eval: np.ndarray
    # stratum name -> boolean mask over eval rows marking the POSITIVES that belong to it
    strata: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for a in ("y_fit", "y_eval"):
            v = np.asarray(getattr(self, a))
            if not np.isin(v, (0, 1)).all():
                raise ValueError(f"{self.name}: {a} must be 0/1")
            setattr(self, a, v.astype(np.int8))
        if len(set(self.acct_eval.tolist())) != len(self.acct_eval):
            raise ValueError(f"{self.name}: one eval row per account is required")
        if not (len(self.X_fit) == len(self.y_fit) == len(self.acct_fit)):
            raise ValueError(f"{self.name}: fit arrays differ in length")
        if not (len(self.X_eval) == len(self.y_eval) == len(self.acct_eval)):
            raise ValueError(f"{self.name}: eval arrays differ in length")
        for k, m in self.strata.items():
            if m.shape != self.y_eval.shape or (m & (self.y_eval == 0)).any():
                raise ValueError(f"{self.name}: stratum {k} must mark eval positives only")

    @property
    def seen(self) -> np.ndarray:
        fit = set(self.acct_fit.tolist())
        return np.array([a in fit for a in self.acct_eval.tolist()], dtype=bool)


# ------------------------------------------------------------------ metrics


def auc(y: np.ndarray, s: np.ndarray) -> float | None:
    """ROC-AUC with average ranks for ties (= sklearn); None if one class is absent."""
    y = np.asarray(y)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return None
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def ap(y: np.ndarray, s: np.ndarray) -> float | None:
    return float(average_precision_score(y, s)) if 0 < int(np.sum(y)) < len(y) else None


def pooled(values: list[float | None], weights: list[float]) -> float | None:
    """Positive-weighted mean over folds, skipping undefined folds."""
    pairs = [(v, w) for v, w in zip(values, weights, strict=True) if v is not None and w > 0]
    if not pairs:
        return None
    return float(sum(v * w for v, w in pairs) / sum(w for _, w in pairs))


def percentile_ci(rep: np.ndarray, level: float) -> list[float] | None:
    rep = rep[np.isfinite(rep)]
    if len(rep) == 0:
        return None
    a = (1.0 - level) / 2.0
    return [float(np.quantile(rep, a)), float(np.quantile(rep, 1.0 - a))]


def _nan_lowest(x: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(x), -np.inf, x)


# ------------------------------------------------------------------ Q-A recognition probe


def _fit_scores(factory: ModelFactory, seed: int, X, y, X_eval) -> np.ndarray:
    m = factory(seed)
    m.fit(X, y)
    return m.predict_proba(X_eval)[:, 1]


def _control_rows(f: FoldData, n_pos: int, n_neg: int, rng: np.random.Generator) -> np.ndarray:
    """Fit-row indices to REMOVE for the control arm: n_pos positives and n_neg negatives from
    accounts that are absent from the eval day."""
    eval_accts = set(f.acct_eval.tolist())
    pool = np.array([a not in eval_accts for a in f.acct_fit.tolist()], dtype=bool)
    pos = np.flatnonzero(pool & (f.y_fit == 1))
    neg = np.flatnonzero(pool & (f.y_fit == 0))
    if len(pos) < n_pos or len(neg) < n_neg:
        raise ValueError(f"{f.name}: not enough rows outside the eval accounts for the control arm")
    return np.concatenate(
        [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
    )


def _label_memory(f: FoldData) -> np.ndarray:
    """Per eval row: the account's mean label over the fit rows (NaN if unseen)."""
    s, n = {}, {}
    for a, y in zip(f.acct_fit.tolist(), f.y_fit.tolist(), strict=True):
        s[a] = s.get(a, 0) + y
        n[a] = n.get(a, 0) + 1
    return np.array([s[a] / n[a] if a in n else np.nan for a in f.acct_eval.tolist()])


def fold_scores(f: FoldData, factory: ModelFactory, seeds: list[int], seed_base: int) -> dict:
    """Eval-row scores per arm and seed: full, disjoint, control (each a list over seeds)."""
    seen = f.seen
    seen_accts = set(f.acct_eval[seen].tolist())
    own = np.array([a in seen_accts for a in f.acct_fit.tolist()], dtype=bool)
    n_pos_rm, n_neg_rm = int((own & (f.y_fit == 1)).sum()), int((own & (f.y_fit == 0)).sum())
    out: dict[str, list[np.ndarray]] = {"full": [], "disjoint": [], "control": []}
    for i, seed in enumerate(seeds):
        out["full"].append(_fit_scores(factory, seed, f.X_fit, f.y_fit, f.X_eval))
        keep = ~own
        out["disjoint"].append(_fit_scores(factory, seed, f.X_fit[keep], f.y_fit[keep], f.X_eval))
        rm = _control_rows(f, n_pos_rm, n_neg_rm, np.random.default_rng(seed_base + i))
        keep = np.ones(len(f.y_fit), dtype=bool)
        keep[rm] = False
        out["control"].append(_fit_scores(factory, seed, f.X_fit[keep], f.y_fit[keep], f.X_eval))
    out["removed"] = {"rows": n_pos_rm + n_neg_rm, "positives": n_pos_rm}
    return out


def _mean_auc(y, score_list, idx=None) -> float | None:
    vals = [auc(y if idx is None else y[idx], s if idx is None else s[idx]) for s in score_list]
    return None if any(v is None for v in vals) else float(np.mean(vals))


def decide_recognition(diff_ci, disj_point, disj_ci, ctrl_point, unseen_point, ctrl_ci=None) -> str:
    """configs/p4_diagnosis.yaml -> recognition verdict (first match wins)."""
    if None in (diff_ci, disj_point, disj_ci, ctrl_point, unseen_point):
        return "insufficient_data"
    if ctrl_ci is not None and ctrl_ci[0] <= unseen_point:
        return "no_gap"
    if diff_ci[0] > 0 and disj_point <= unseen_point + 0.5 * (ctrl_point - unseen_point):
        return "recognition"
    if diff_ci[1] < 0.05 and disj_ci[0] > unseen_point:
        return "population"
    return "mixed"


def recognition_probe(
    folds: list[FoldData],
    factory: ModelFactory,
    seeds: list[int],
    n_boot: int,
    boot_seed: int,
    ci: float,
    min_seen_pos: int,
    scores: dict[str, dict] | None = None,
) -> dict:
    scores = scores or {f.name: fold_scores(f, factory, seeds, boot_seed) for f in folds}
    per_fold, w_seen, w_unseen = [], [], []
    pts: dict[str, list] = {k: [] for k in ("full", "disjoint", "control", "label_memory",
                                            "unseen_full", "ap_seen", "ap_unseen")}  # fmt: skip
    for f in folds:
        s, seen = scores[f.name], f.seen
        si, ui = np.flatnonzero(seen), np.flatnonzero(~seen)
        lm = _label_memory(f)
        rec = {
            "fold": f.name,
            "n_fit": int(len(f.y_fit)),
            "pos_fit": int(f.y_fit.sum()),
            "n_seen": int(len(si)),
            "pos_seen": int(f.y_eval[si].sum()),
            "n_unseen": int(len(ui)),
            "pos_unseen": int(f.y_eval[ui].sum()),
            "removed_for_disjoint": s["removed"],
            "auc_seen_full": _mean_auc(f.y_eval, s["full"], si),
            "auc_seen_disjoint": _mean_auc(f.y_eval, s["disjoint"], si),
            "auc_seen_control": _mean_auc(f.y_eval, s["control"], si),
            "auc_seen_label_memory": auc(f.y_eval[si], lm[si]),
            "auc_unseen_full": _mean_auc(f.y_eval, s["full"], ui),
            "ap_seen_full": _mean_ap(f.y_eval[si], [x[si] for x in s["full"]]),
            "ap_unseen_full": _mean_ap(f.y_eval[ui], [x[ui] for x in s["full"]]),
        }
        rec["prevalence_seen"] = rec["pos_seen"] / rec["n_seen"] if rec["n_seen"] else None
        rec["prevalence_unseen"] = rec["pos_unseen"] / rec["n_unseen"] if rec["n_unseen"] else None
        per_fold.append(rec)
        w_seen.append(rec["pos_seen"])
        w_unseen.append(rec["pos_unseen"])
        for k, src in (("full", "auc_seen_full"), ("disjoint", "auc_seen_disjoint"),
                       ("control", "auc_seen_control"), ("label_memory", "auc_seen_label_memory"),
                       ("ap_seen", "ap_seen_full")):  # fmt: skip
            pts[k].append(rec[src])
        pts["unseen_full"].append(rec["auc_unseen_full"])
        pts["ap_unseen"].append(rec["ap_unseen_full"])

    pooled_pts = {k: pooled(v, w_unseen if k in ("unseen_full", "ap_unseen") else w_seen)
                  for k, v in pts.items()}  # fmt: skip

    # paired bootstrap: resample seen (and unseen) eval accounts within each fold
    rng = np.random.default_rng(boot_seed)
    reps = {k: np.full(n_boot, np.nan) for k in ("diff", "disjoint", "unseen_full", "control")}
    for b in range(n_boot):
        vals = {k: [] for k in ("disjoint", "control", "unseen_full")}
        ws, wu = [], []
        for f in folds:
            s, seen = scores[f.name], f.seen
            si, ui = np.flatnonzero(seen), np.flatnonzero(~seen)
            bs = rng.choice(si, len(si), replace=True) if len(si) else si
            bu = rng.choice(ui, len(ui), replace=True) if len(ui) else ui
            vals["disjoint"].append(_mean_auc(f.y_eval, s["disjoint"], bs) if len(bs) else None)
            vals["control"].append(_mean_auc(f.y_eval, s["control"], bs) if len(bs) else None)
            vals["unseen_full"].append(_mean_auc(f.y_eval, s["full"], bu) if len(bu) else None)
            ws.append(int(f.y_eval[bs].sum()) if len(bs) else 0)
            wu.append(int(f.y_eval[bu].sum()) if len(bu) else 0)
        d, c, u = (
            pooled(vals["disjoint"], ws),
            pooled(vals["control"], ws),
            pooled(vals["unseen_full"], wu),
        )
        if d is not None and c is not None:
            reps["diff"][b], reps["disjoint"][b], reps["control"][b] = c - d, d, c
        if u is not None:
            reps["unseen_full"][b] = u
    out = {
        "per_fold": per_fold,
        "pooled": pooled_pts,
        "ci": {
            "control_minus_disjoint": percentile_ci(reps["diff"], ci),
            "disjoint": percentile_ci(reps["disjoint"], ci),
            "control": percentile_ci(reps["control"], ci),
            "unseen_full": percentile_ci(reps["unseen_full"], ci),
        },
        "n_seen_pos": int(sum(w_seen)),
        "n_unseen_pos": int(sum(w_unseen)),
        "n_boot_undefined": int(np.isnan(reps["diff"]).sum()),
    }
    p = pooled_pts
    out["control_minus_disjoint"] = (
        None if p["control"] is None or p["disjoint"] is None else p["control"] - p["disjoint"]
    )
    if out["n_seen_pos"] < min_seen_pos or out["n_boot_undefined"] > 0.01 * n_boot:
        out["verdict"] = "insufficient_data"
    else:
        out["verdict"] = decide_recognition(
            out["ci"]["control_minus_disjoint"], p["disjoint"], out["ci"]["disjoint"],
            p["control"], p["unseen_full"], out["ci"]["control"],
        )  # fmt: skip
    return out


def _mean_ap(y, score_list) -> float | None:
    vals = [ap(y, s) for s in score_list]
    return None if any(v is None for v in vals) else float(np.mean(vals))


# ------------------------------------------------------------------ Q-B univariate signal


def univariate(
    folds: list[FoldData], names: list[str], n_boot: int, boot_seed: int, ci: float, top_k: int
) -> dict:
    """Per feature: pooled AUC on unseen eval rows (NaN ranked lowest), oriented so that >= 0.5,
    with an account-bootstrap CI; the same feature's pooled AUC on seen rows (point only)."""
    rng = np.random.default_rng(boot_seed)
    unseen_idx = [np.flatnonzero(~f.seen) for f in folds]
    seen_idx = [np.flatnonzero(f.seen) for f in folds]
    boots = [[rng.choice(u, len(u), replace=True) for _ in range(n_boot)] for u in unseen_idx]
    rows = []
    for j, name in enumerate(names):
        xs = [_nan_lowest(f.X_eval[:, j]) for f in folds]
        wu = [int(f.y_eval[u].sum()) for f, u in zip(folds, unseen_idx, strict=True)]
        ws = [int(f.y_eval[s].sum()) for f, s in zip(folds, seen_idx, strict=True)]
        pu = pooled(
            [auc(f.y_eval[u], x[u]) for f, x, u in zip(folds, xs, unseen_idx, strict=True)], wu
        )
        ps = pooled(
            [auc(f.y_eval[s], x[s]) for f, x, s in zip(folds, xs, seen_idx, strict=True)], ws
        )
        rep = np.full(n_boot, np.nan)
        for b in range(n_boot):
            v, w = [], []
            for f, x, bb in zip(folds, xs, boots, strict=True):
                idx = bb[b]
                v.append(auc(f.y_eval[idx], x[idx]))
                w.append(int(f.y_eval[idx].sum()))
            r = pooled(v, w)
            rep[b] = np.nan if r is None else r
        sign = 1 if pu is None or pu >= 0.5 else -1
        c = percentile_ci(rep, ci)
        if sign < 0 and c is not None:
            c = [1.0 - c[1], 1.0 - c[0]]
        rows.append({
            "feature": name,
            "orientation": "+" if sign > 0 else "-",
            "auc_unseen": None if pu is None else (pu if sign > 0 else 1.0 - pu),
            "ci_unseen": c,
            "auc_seen_same_orientation": None if ps is None else (ps if sign > 0 else 1.0 - ps),
            "nan_share_eval": float(np.mean([np.isnan(f.X_eval[:, j]).mean() for f in folds])),
        })  # fmt: skip
    rows.sort(key=lambda r: -(r["auc_unseen"] or 0.0))
    return {
        "top": rows[:top_k],
        "n_features": len(names),
        "n_ci_lower_above_0_55": sum(
            1 for r in rows if r["ci_unseen"] and r["ci_unseen"][0] > 0.55
        ),
        "all": rows,
    }


# ------------------------------------------------------------------ Q-C strata


def strata_auc(
    folds: list[FoldData], full_scores: dict[str, list[np.ndarray]], min_pos: int
) -> dict:
    """Each stratum's UNSEEN positives against all unseen negatives of the same fold (model score
    = the full arm, mean AUC over seeds), pooled positive-weighted."""
    names = sorted({k for f in folds for k in f.strata})
    total_pos = sum(int(f.y_eval[~f.seen].sum()) for f in folds)
    out = {}
    for k in names:
        vals, w = [], []
        for f in folds:
            u = ~f.seen
            m = f.strata.get(k, np.zeros_like(u)) & u
            idx = np.flatnonzero(m | (u & (f.y_eval == 0)))
            vals.append(_mean_auc(f.y_eval, full_scores[f.name], idx) if m.any() else None)
            w.append(int(m.sum()))
        n = int(sum(w))
        out[k] = {
            "n_pos_unseen": n,
            "share_of_unseen_pos": n / total_pos if total_pos else None,
            "auc_unseen": pooled(vals, w) if n >= min_pos else None,
        }
    return {"total_unseen_pos": total_pos, "strata": out}
