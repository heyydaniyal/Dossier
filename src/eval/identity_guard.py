"""Identity (account-recognition) guard. Evaluation-side: it reads labels, so it lives in src/eval.

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

import numpy as np
from sklearn.metrics import average_precision_score

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
