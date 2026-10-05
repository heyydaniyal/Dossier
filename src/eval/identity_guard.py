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


def load_v2_config(path: Path = GUARD_V2_CONFIG) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg["version"] != 2 or cfg["statistic"] != "delta_dprime":
        raise ValueError("identity guard v2 config expected")
    if tuple(cfg["verdicts"]) != VERDICT_ORDER_V2:
        raise ValueError(f"verdict order must be {VERDICT_ORDER_V2}")
    cfg["_sha256"] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return cfg


def check_folds(cfg: dict, split) -> list[dict]:
    """Every fold fits and evaluates inside TRAIN + VALIDATION only, with at least one full day
    (the embargo) between the last fit day and the evaluated day. Raises otherwise."""
    allowed = {
        d.date() for p in ("TRAIN", "VALIDATION") for d in split.window_starts(p)
    }  # window starts are the calendar days of these periods
    out = []
    for f in cfg["folds"]:
        fit = [date.fromisoformat(d) for d in f["fit_days"]]
        ev = date.fromisoformat(f["eval_day"])
        if not set(fit) | {ev} <= allowed:
            raise ValueError(f"fold {f['name']} uses a day outside TRAIN/VALIDATION")
        if ev - max(fit) < timedelta(days=2):
            raise ValueError(f"fold {f['name']}: no embargo day between fit and eval")
        out.append({"name": f["name"], "fit_days": fit, "eval_day": ev})
    return out


def _probit(a: np.ndarray) -> np.ndarray:
    return _PPF(np.clip(a, _EPS, 1 - _EPS))


def _cum(scores: np.ndarray, w: np.ndarray):
    order = np.argsort(scores, kind="stable")
    c = np.concatenate([np.zeros((w.shape[0], 1)), np.cumsum(w[:, order], axis=1)], axis=1)
    return scores[order], c


def weighted_auc_ap(sp, sn, wp, wn):
    """ROC-AUC (Mann-Whitney, ties count 1/2) and average precision (sklearn definition, tied
    scores grouped) for each row of the weights wp (B x n_pos) and wn (B x n_neg). With integer
    weights this equals the unweighted metric on the rows repeated that many times (tested)."""
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
    account on several eval days keeps one multiplicity). Order-free: accounts are ranked by a
    keyed hash, never by ID (D5)."""

    keys = [np.asarray(f["account_key"]).astype(str) for f in folds]
    allk = np.concatenate(keys)
    uniq = np.unique(allk)
    h = keyed_u64(uniq.tolist(), "identity_guard_v2_bootstrap", seed)
    rank = np.argsort(np.argsort(h, kind="stable"), kind="stable")  # position in hash order
    pos = dict(zip(uniq.tolist(), rank.tolist(), strict=True))
    n = len(uniq)
    rng = np.random.default_rng([seed, 99])
    idx = rng.integers(0, n, size=(n_boot, n)) + (np.arange(n_boot) * n)[:, None]
    c = np.bincount(idx.ravel(), minlength=n_boot * n).reshape(n_boot, n).astype(float)
    return [c[:, [pos[k] for k in ks]] for ks in keys]


def evaluate_v2(folds: list[dict], cfg: dict | None = None) -> dict:
    """folds: one dict per configured fold, with aligned arrays for that fold's EVALUATED day:
    y (0/1), pred_base, pred_with_group (scores of the two models fitted on the fold's fit rows),
    seen (account in those fit rows), account_key. Returns the verdict and every number behind it.
    """
    cfg = cfg or load_v2_config()
    bs, th = cfg["bootstrap"], cfg["thresholds"]
    if bs["unit"] != "account":
        raise ValueError("v2 resamples accounts")
    if len(folds) != len(cfg["folds"]):
        raise ValueError(f"expected {len(cfg['folds'])} folds, got {len(folds)}")
    for f in folds:
        n = len(f["y"])
        if not all(len(f[k]) == n for k in ("pred_base", "pred_with_group", "seen", "account_key")):
            raise ValueError("identity guard v2: fold arrays are not aligned")
    b, alpha = bs["n_boot"], (1 - bs["ci"]) / 2
    accw = _account_weights(folds, b, bs["seed"])
    out: dict = {
        "version": 2,
        "config_sha256": cfg.get("_sha256"),
        "statistic": cfg["statistic"],
        "thresholds": dict(th),
    }
    reps: dict[str, np.ndarray] = {}
    for sub in ("seen", "unseen"):
        g_rep, g_pt, w, base_auc, d_ap, npos, nall, per_fold = [], [], [], [], [], 0, 0, []
        for fi, f in enumerate(folds):
            y = np.asarray(f["y"]).astype(int)
            m = np.asarray(f["seen"]).astype(bool)
            m = m if sub == "seen" else ~m
            yy = y[m]
            pb = np.asarray(f["pred_base"], float)[m]
            pw = np.asarray(f["pred_with_group"], float)[m]
            p, n = int(yy.sum()), int((1 - yy).sum())
            npos, nall = npos + p, nall + p + n
            per_fold.append({"n": p + n, "n_pos": p})
            if p == 0 or n == 0:
                continue
            pos, neg = yy == 1, yy == 0
            ws = accw[fi][:, m]
            ab, _ = weighted_auc_ap(pb[pos], pb[neg], ws[:, pos], ws[:, neg])
            aw, _ = weighted_auc_ap(pw[pos], pw[neg], ws[:, pos], ws[:, neg])
            g_rep.append(np.sqrt(2) * (_probit(aw) - _probit(ab)))
            one_p, one_n = np.ones((1, p)), np.ones((1, n))
            ab1, apb1 = weighted_auc_ap(pb[pos], pb[neg], one_p, one_n)
            aw1, apw1 = weighted_auc_ap(pw[pos], pw[neg], one_p, one_n)
            g_pt.append(float(np.sqrt(2) * (_probit(aw1) - _probit(ab1))[0]))
            base_auc.append(float(ab1[0]))
            d_ap.append(float(apw1[0] - apb1[0]))
            w.append(p)
        s: dict = {"n": nall, "n_pos": npos, "per_fold": per_fold}
        if w:
            wn = np.asarray(w, float) / sum(w)
            rep = np.sum([wi * g for wi, g in zip(wn, g_rep, strict=True)], axis=0)
            n_bad = int(np.isnan(rep).sum())
            if n_bad > 0.01 * b:
                raise RuntimeError(f"{sub}: {n_bad} of {b} bootstrap replicates undefined")
            lo, hi = np.nanquantile(rep, [alpha, 1 - alpha])
            reps[sub] = rep
            s |= {
                "gain_dprime": float(np.dot(wn, g_pt)),
                "ci": [float(lo), float(hi)],
                "base_auc": float(np.dot(wn, base_auc)),
                "delta_ap_report_only": float(np.dot(wn, d_ap)),
                "n_boot_undefined": n_bad,
            }
        out[sub] = s

    def done(v: str) -> dict:
        return out | {"verdict": v, "cleared": cfg["verdicts"][v]["cleared"]}

    if (
        out["seen"]["n_pos"] < th["n_min_pos_seen"]
        or out["unseen"]["n_pos"] < th["n_min_pos_unseen"]
        or "seen" not in reps
        or "unseen" not in reps
        or out["seen"]["base_auc"] >= th["seen_base_auc_max"]
    ):
        return done("insufficient_data")
    contrast = reps["unseen"] - th["ratio"] * reps["seen"]
    c_lo, c_hi = np.nanquantile(contrast, [alpha, 1 - alpha])
    out["contrast_unseen_minus_ratio_seen"] = {
        "point": out["unseen"]["gain_dprime"] - th["ratio"] * out["seen"]["gain_dprime"],
        "ci": [float(c_lo), float(c_hi)],
    }
    s_lo = out["seen"]["ci"][0]
    u_lo, u_hi = out["unseen"]["ci"]
    if u_hi < 0:
        return done("harms_unseen")
    if s_lo > 0 and c_hi < 0:
        return done("account_recognition")
    if u_lo > 0:
        return done("pass")
    if s_lo > 0:
        return done("inconclusive")
    return done("no_material_gain")
