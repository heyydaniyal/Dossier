"""Identity (account-recognition) guard. Evaluation-side: it reads labels, so it lives in src/eval.

TWO VERSIONS
  v1 `evaluate` / `cleared` (below, P2): kept unchanged as the record of how KYC was judged in P2.
  v2 `evaluate_v2` (further below; configs/p3_identity_guard.yaml): BINDING for P3-P5, approved
     by Dani 2026-10-05 after the independent P2 review (finding C-1) showed v1 is biased by its
     prevalence scaling and has no power.

--- v1 (P2) ---

Why (P2, 2026-09-30): the synthetic KYC failed ceilings C1/C2 on all VALIDATION alerts but showed
no signal on VALIDATION alerts of accounts never alerted in TRAIN. A model can learn to RECOGNISE
accounts that recur between TRAIN and VALIDATION from any stable per-account field combination.
That looks like skill on regime A and disappears on new accounts (regime B).

Raw lift cannot be compared between seen and unseen accounts: they are different populations
(even the transaction-only model has lift 4.0x on all VALIDATION vs 1.43x on unseen). So the guard
compares the INCREMENTAL gain of a feature group, measured separately inside each subset:

  gain_s = AP(base + group on subset s) - AP(base on subset s),  s in {seen, unseen}
  normalised by the subset's own prevalence:  g_s = gain_s / prev_s

Verdict (thresholds pre-declared in configs/p2_kyc.yaml -> identity_guard):
  insufficient_data     either subset has < min_positives positives: the group is NOT cleared
  no_material_gain      g_seen < materiality: nothing to recognise (group adds nothing on seen)
  account_recognition   g_seen >= materiality and g_unseen < min_unseen_to_seen_ratio * g_seen
  pass                  otherwise
Both predictions must come from ONE fit per model (same TRAIN), evaluated on the two subsets.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np
import yaml
from sklearn.metrics import average_precision_score

from src.data.hashing import keyed_u64

VERDICTS = ("pass", "account_recognition", "no_material_gain", "insufficient_data")


def _sig(x: float) -> float:
    return float(f"{float(x):.10g}")


def evaluate(
    y: np.ndarray,
    pred_base: np.ndarray,
    pred_with_group: np.ndarray,
    seen: np.ndarray,
    cfg: dict,
) -> dict:
    """y, predictions and the seen-in-TRAIN mask are aligned VALIDATION rows (alert level)."""
    y = np.asarray(y).astype(int)
    seen = np.asarray(seen).astype(bool)
    pb, pw = np.asarray(pred_base, dtype=float), np.asarray(pred_with_group, dtype=float)
    if not (len(y) == len(pb) == len(pw) == len(seen)):
        raise ValueError("identity guard: inputs are not aligned")
    out: dict = {"thresholds": dict(cfg)}
    for name, m in (("seen", seen), ("unseen", ~seen)):
        n_pos = int(y[m].sum())
        s = {"n": int(m.sum()), "n_pos": n_pos}
        if n_pos >= cfg["min_positives"] and n_pos < m.sum():
            prev = float(y[m].mean())
            ab = average_precision_score(y[m], pb[m])
            aw = average_precision_score(y[m], pw[m])
            s |= {
                "prevalence": _sig(prev),
                "ap_base": _sig(ab),
                "ap_with_group": _sig(aw),
                "gain": _sig(aw - ab),
                "gain_over_prev": _sig((aw - ab) / prev),
            }
        out[name] = s
    if "gain_over_prev" not in out["seen"] or "gain_over_prev" not in out["unseen"]:
        return out | {"verdict": "insufficient_data"}
    gs, gu = out["seen"]["gain_over_prev"], out["unseen"]["gain_over_prev"]
    if gs < cfg["materiality_gain_over_prev"]:
        verdict = "no_material_gain"
    elif gu < cfg["min_unseen_to_seen_ratio"] * gs:
        verdict = "account_recognition"
    else:
        verdict = "pass"
    return out | {"verdict": verdict}


def cleared(result: dict) -> bool:
    """A feature group may enter a model only if it passes, or adds nothing at all."""
    return result["verdict"] in ("pass", "no_material_gain")


# ================================================================ v2 (P3-P5, binding)

GUARD_V2_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "p3_identity_guard.yaml"
VERDICT_ORDER_V2 = (
    "insufficient_data",
    "harms_unseen",
    "account_recognition",
    "pass",
    "inconclusive",
    "no_material_gain",
)
_PPF = np.vectorize(NormalDist().inv_cdf)
_EPS = 1e-6


def _canonical(cfg: dict) -> str:
    return json.dumps({k: v for k, v in cfg.items() if k != "_sha256"}, sort_keys=True, default=str)


def load_v2_config(path: Path = GUARD_V2_CONFIG) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg["version"] != 2 or cfg["statistic"] != "delta_dprime":
        raise ValueError("identity guard v2 config expected")
    if tuple(cfg["verdicts"]) != VERDICT_ORDER_V2:
        raise ValueError(f"verdict order must be {VERDICT_ORDER_V2}")
    cfg["_sha256"] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return cfg


def check_folds(cfg: dict, split) -> list[dict]:
    """Every fold fits and evaluates inside TRAIN + VALIDATION only, with an embargo of at least
    L_max between the last fit day and the evaluated day (derived from the split). Raises."""
    allowed = {d.date() for p in ("TRAIN", "VALIDATION") for d in split.window_starts(p)}
    min_gap = timedelta(days=1) + split.l_max  # last fit day ends +1 day; then the embargo
    out = []
    for f in cfg["folds"]:
        fit = [date.fromisoformat(d) for d in f["fit_days"]]
        ev = date.fromisoformat(f["eval_day"])
        if not set(fit) | {ev} <= allowed:
            raise ValueError(f"fold {f['name']} uses a day outside TRAIN/VALIDATION")
        if ev - max(fit) < min_gap:
            raise ValueError(f"fold {f['name']}: no embargo day between fit and eval")
        out.append({"name": f["name"], "fit_days": fit, "eval_day": ev})
    return out


def _probit(a: np.ndarray) -> np.ndarray:
    """Phi^-1 of AUC (clipped); an undefined AUC (empty resample) stays NaN, silently."""
    a = np.asarray(a, float)
    nan = np.isnan(a)
    return np.where(nan, np.nan, _PPF(np.clip(np.where(nan, 0.5, a), _EPS, 1 - _EPS)))


def _cum(scores: np.ndarray, w: np.ndarray):
    order = np.argsort(scores, kind="stable")
    c = np.concatenate([np.zeros((w.shape[0], 1)), np.cumsum(w[:, order], axis=1)], axis=1)
    return scores[order], c


def weighted_auc_ap(sp, sn, wp, wn):
    """ROC-AUC (Mann-Whitney, ties count 1/2) and average precision (sklearn definition, tied
    scores grouped) for each row of the weights wp (B x n_pos) and wn (B x n_neg). With integer
    weights this equals the unweighted metric on the rows repeated that many times (tested).
    A row whose positive or negative weights are all zero gives NaN."""
    sns, cn = _cum(sn, wn)
    sps, cp = _cum(sp, wp)
    lo_n = np.searchsorted(sns, sp, side="left")
    hi_n = np.searchsorted(sns, sp, side="right")
    lo_p = np.searchsorted(sps, sp, side="left")
    tot_p, tot_n = cp[:, -1:], cn[:, -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (wp * 0.5 * (cn[:, lo_n] + cn[:, hi_n])).sum(1) / (tot_p[:, 0] * tot_n[:, 0])
        tp_ge = tot_p - cp[:, lo_p]
        fp_ge = tot_n - cn[:, lo_n]
        den = tp_ge + fp_ge
        prec = np.divide(tp_ge, den, out=np.zeros_like(den), where=den > 0)
        ap = (wp * prec).sum(1) / tot_p[:, 0]
    return auc, ap


def _account_weights(folds: list[dict], n_boot: int, seed: int) -> list[np.ndarray]:
    """One account-level resample per replicate, shared by every fold and both subsets (an
    account on several eval days keeps one multiplicity). Accounts are indexed by a keyed hash,
    never by ID order (D5)."""
    keys = [np.asarray(f["account_key"]).astype(str) for f in folds]
    uniq = np.unique(np.concatenate(keys))
    h = keyed_u64(uniq.tolist(), "identity_guard_v2_bootstrap", seed)
    rank = np.argsort(np.argsort(h, kind="stable"), kind="stable")
    pos = dict(zip(uniq.tolist(), rank.tolist(), strict=True))
    n = len(uniq)
    rng = np.random.default_rng([seed, 99])
    idx = rng.integers(0, n, size=(n_boot, n)) + (np.arange(n_boot) * n)[:, None]
    c = np.bincount(idx.ravel(), minlength=n_boot * n).reshape(n_boot, n).astype(float)
    return [c[:, [pos[k] for k in ks]] for ks in keys]


def percentile_ci(rep: np.ndarray, level: float) -> list[float]:
    a = (1 - level) / 2
    lo, hi = np.nanquantile(rep, [a, 1 - a])
    return [float(lo), float(hi)]


def decide(seen_ci, unseen_ci, contrast_ci) -> str:
    """The verdict table of configs/p3_identity_guard.yaml (after the insufficient_data checks)."""
    if unseen_ci[1] < 0:
        return "harms_unseen"
    if seen_ci[0] > 0 and contrast_ci[1] < 0:
        return "account_recognition"
    if unseen_ci[0] > 0:
        return "pass"
    if seen_ci[0] > 0:
        return "inconclusive"
    return "no_material_gain"


def _validate_folds(folds: list[dict], spec: list[dict]) -> None:
    if [f.get("name") for f in folds] != [s["name"] for s in spec]:
        raise ValueError(f"expected folds {[s['name'] for s in spec]} in this order")
    for f, sp in zip(folds, spec, strict=True):
        n = len(f["y"])
        keys = ("window_start", "pred_base", "pred_with_group", "seen", "account_key")
        if not all(len(f[k]) == n for k in keys):
            raise ValueError(f"fold {sp['name']}: arrays are not aligned")
        y = np.asarray(f["y"])
        if not np.isin(y, (0, 1)).all():
            raise ValueError(f"fold {sp['name']}: y must be 0/1")
        for k in ("pred_base", "pred_with_group"):
            if not np.isfinite(np.asarray(f[k], float)).all():
                raise ValueError(f"fold {sp['name']}: {k} has NaN or infinite values")
        if len(set(np.asarray(f["account_key"]).astype(str).tolist())) != n:
            raise ValueError(f"fold {sp['name']}: account_key must be unique per eval day")
        days = {(d.date() if hasattr(d, "date") else d) for d in f["window_start"]}
        if days != {sp["eval_day"]}:
            raise ValueError(f"fold {sp['name']}: rows must all be from {sp['eval_day']}")


def evaluate_v2(
    folds: list[dict],
    cfg: dict | None = None,
    split=None,
    allow_unfrozen_config: bool = False,
) -> dict:
    """folds: one dict per configured fold, in config order, with aligned arrays for that fold's
    EVALUATED day: name, window_start, y (0/1), pred_base, pred_with_group (scores of the two
    models fitted on the fold's fit rows), seen (account in those fit rows), account_key (unique).
    Never raises on thin data: such cases end as insufficient_data, with the reason.
    """
    frozen = load_v2_config()
    cfg = cfg or frozen
    is_frozen = _canonical(cfg) == _canonical(frozen)
    if not is_frozen and not allow_unfrozen_config:
        raise ValueError("identity guard v2: config differs from configs/p3_identity_guard.yaml")
    if split is None:
        from src.data.periods import load_split

        split = load_split()
    spec = check_folds(cfg, split)
    _validate_folds(folds, spec)
    bs, th = cfg["bootstrap"], cfg["thresholds"]
    if bs["unit"] != "account":
        raise ValueError("v2 resamples accounts")
    b = bs["n_boot"]
    accw = _account_weights(folds, b, bs["seed"])
    out: dict = {
        "version": 2,
        "config_sha256": hashlib.sha256(_canonical(cfg).encode()).hexdigest(),
        "config_is_frozen": is_frozen,
        "statistic": cfg["statistic"],
        "thresholds": dict(th),
    }
    reps: dict[str, np.ndarray] = {}
    sq_pts: dict[str, float] = {}
    for sub in ("seen", "unseen"):
        g_rep, g_pt, g_sq, w, base_auc, d_ap, per_fold = [], [], [], [], [], [], []
        for fi, f in enumerate(folds):
            y = np.asarray(f["y"]).astype(int)
            m = np.asarray(f["seen"]).astype(bool)
            m = m if sub == "seen" else ~m
            yy = y[m]
            pb = np.asarray(f["pred_base"], float)[m]
            pw = np.asarray(f["pred_with_group"], float)[m]
            p, n = int(yy.sum()), int((1 - yy).sum())
            rec: dict = {"fold": spec[fi]["name"], "n": p + n, "n_pos": p}
            per_fold.append(rec)
            if p == 0 or n == 0:
                rec["dropped"] = "one class only"
                continue
            pos, neg = yy == 1, yy == 0
            one_p, one_n = np.ones((1, p)), np.ones((1, n))
            ab1, apb1 = weighted_auc_ap(pb[pos], pb[neg], one_p, one_n)
            aw1, apw1 = weighted_auc_ap(pw[pos], pw[neg], one_p, one_n)
            if ab1[0] >= th["base_auc_max"]:
                rec |= {"dropped": "base AUC at/above base_auc_max", "base_auc": float(ab1[0])}
                continue
            ws = accw[fi][:, m]
            ab, _ = weighted_auc_ap(pb[pos], pb[neg], ws[:, pos], ws[:, neg])
            aw, _ = weighted_auc_ap(pw[pos], pw[neg], ws[:, pos], ws[:, neg])
            g_rep.append(np.sqrt(2) * (_probit(aw) - _probit(ab)))
            gp = float(np.sqrt(2) * (_probit(aw1) - _probit(ab1))[0])
            g_pt.append(gp)
            g_sq.append(float(2 * (_probit(aw1) ** 2 - _probit(ab1) ** 2)[0]))
            base_auc.append(float(ab1[0]))
            d_ap.append(float(apw1[0] - apb1[0]))
            w.append(p)
            rec |= {"gain_dprime": gp, "base_auc": float(ab1[0])}
        s: dict = {"n_pos_used": int(sum(w)), "per_fold": per_fold}
        if w:
            wv = np.asarray(w, float)
            G = np.vstack(g_rep)  # folds x replicates
            ok = ~np.isnan(G)
            num = np.where(ok, G * wv[:, None], 0.0).sum(0)
            den = np.where(ok, wv[:, None], 0.0).sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                rep = num / den  # a fold left empty by a resample is skipped in that replicate
            wn = wv / wv.sum()
            reps[sub] = rep
            sq_pts[sub] = float(np.dot(wn, g_sq))
            s |= {
                "gain_dprime": float(np.dot(wn, g_pt)),
                "ci": percentile_ci(rep, bs["ci"]),
                "base_auc": float(np.dot(wn, base_auc)),
                "delta_ap_report_only": float(np.dot(wn, d_ap)),
                "n_boot_undefined": int(np.isnan(rep).sum()),
            }
        out[sub] = s

    def done(v: str, reason: str | None = None) -> dict:
        r = out | {"verdict": v, "cleared": cfg["verdicts"][v]["cleared"]}
        return r | ({"insufficient_reason": reason} if reason else {})

    for sub, key in (("seen", "n_min_pos_seen"), ("unseen", "n_min_pos_unseen")):
        if out[sub]["n_pos_used"] < th[key]:
            return done("insufficient_data", f"{sub}: {out[sub]['n_pos_used']} usable positives")
        if out[sub]["n_boot_undefined"] > 0.01 * b:
            return done("insufficient_data", f"{sub}: too many undefined bootstrap replicates")
    contrast = reps["unseen"] - th["ratio"] * reps["seen"]
    out["contrast_unseen_minus_ratio_seen"] = {
        "point": out["unseen"]["gain_dprime"] - th["ratio"] * out["seen"]["gain_dprime"],
        "ci": percentile_ci(contrast, bs["ci"]),
    }
    # Report only (independent check, 2026-10-05): if the group's signal is INDEPENDENT of the
    # base score, d'^2 (not d') adds up, and a high base level squeezes the seen d' gain. The
    # same contrast on the d'^2 scale (ratio squared) shows when that matters.
    sq = sq_pts["unseen"] - th["ratio"] ** 2 * sq_pts["seen"]
    out["report_only_dprime_squared"] = {
        "seen_gain": sq_pts["seen"],
        "unseen_gain": sq_pts["unseen"],
        "contrast_unseen_minus_ratio2_seen": sq,
        "contrast_sign_disagrees": (sq < 0)
        != (out["contrast_unseen_minus_ratio_seen"]["point"] < 0),
    }
    v = decide(
        out["seen"]["ci"], out["unseen"]["ci"], out["contrast_unseen_minus_ratio_seen"]["ci"]
    )
    return done(v)
