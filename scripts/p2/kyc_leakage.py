"""P2 task 6: KYC leakage tests against ceilings DECLARED in configs/p2_kyc.yaml before measuring.

Alert level. Fit on TRAIN alerts, evaluate on VALIDATION alerts. PR-AUC = average precision.
  C1  KYC-only (all attributes)                       <= c1 * prev
  C2  PR-AUC(all KYC) - PR-AUC(real/data-derived KYC) <= c2 * prev   (synthetic attributes)
  C3  PR-AUC(txn + all KYC) - PR-AUC(txn only)        <= c3 * prev   (task 6 incremental value)
  C4  PR-AUC(txn + all KYC) - PR-AUC(txn + real KYC)  <= c4 * prev   (synthetic interaction leakage)
The gate is the point estimate; paired bootstrap 95% intervals are reported alongside.
unseen_only=True repeats the test on VALIDATION alerts of accounts with no TRAIN alert (report
only, not a gate): it separates account memorisation from a planted clue.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from src.alerts.rules import ALL_STATS

MIN_POSITIVES = 20


def _sig(x: float) -> float:
    return float(f"{float(x):.10g}")


def _design(
    df: pl.DataFrame, rule_ids: list[str], kyc_as_of: datetime | None = None
) -> pl.DataFrame:
    return df.with_columns(
        *[
            pl.col("triggered_rules").list.contains(r).cast(pl.Int8).alias(f"fired_{r}")
            for r in rule_ids
        ],
    )


def _matrix(tr: pl.DataFrame, va: pl.DataFrame, num: list[str], cat: list[str]):
    cols_tr, cols_va, is_cat = [], [], []
    for c in num:
        cols_tr.append(tr[c].cast(pl.Float64).to_numpy())
        cols_va.append(va[c].cast(pl.Float64).to_numpy())
        is_cat.append(False)
    for c in cat:
        cats = sorted(tr[c].drop_nulls().unique().to_list())
        if len(cats) > 250:
            raise RuntimeError(f"{c}: {len(cats)} categories (> 250)")
        m = {v: float(i) for i, v in enumerate(cats)}
        cols_tr.append(np.array([m.get(v, np.nan) for v in tr[c].to_list()]))
        cols_va.append(np.array([m.get(v, np.nan) for v in va[c].to_list()]))
        is_cat.append(True)
    return np.column_stack(cols_tr), np.column_stack(cols_va), np.array(is_cat)


def _fit_predict(xtr, ytr, xva, is_cat) -> np.ndarray:
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_leaf_nodes=31,
        early_stopping=False,
        random_state=0,
        categorical_features=is_cat if is_cat.any() else None,
    )
    clf.fit(xtr, ytr)
    return clf.predict_proba(xva)[:, 1]


def leakage_tests(
    alerts: pl.DataFrame,
    labels: pl.DataFrame,
    kyc: pl.DataFrame,
    cfg: dict,
    rule_ids: list[str],
    unseen_only: bool = False,
) -> dict:
    groups = cfg["attribute_groups"]
    as_of = datetime.fromisoformat(cfg["kyc_as_of"])
    d = (
        alerts.filter(pl.col("period").is_in(["TRAIN", "VALIDATION"]))
        .join(labels.select("alert_id", "is_true_positive"), on="alert_id", how="left")
        .join(kyc, on="account_key", how="left")
        .sort("alert_id")
    )
    d = _design(d, rule_ids, as_of)
    tr, va = d.filter(pl.col("period") == "TRAIN"), d.filter(pl.col("period") == "VALIDATION")
    if unseen_only:
        va = va.join(tr.select("account_key").unique(), on="account_key", how="anti")
    ytr, yva = (
        tr["is_true_positive"].to_numpy().astype(int),
        va["is_true_positive"].to_numpy().astype(int),
    )
    res: dict = {
        "n_train": tr.height,
        "n_val": va.height,
        "n_pos_train": int(ytr.sum()),
        "n_pos_val": int(yva.sum()),
    }
    if ytr.sum() < MIN_POSITIVES or yva.sum() < MIN_POSITIVES:
        return res | {"status": "insufficient_data"}
    prev = float(yva.mean())

    txn_num = [*ALL_STATS, "n_rules_triggered", *[f"fired_{r}" for r in rule_ids]]
    numeric = set(groups.get("numeric") or [])
    cat_real = [c for c in groups["real_or_data_derived"] if c not in numeric]
    syn = groups["synthetic"]
    cat_syn = [c for c in syn if c not in numeric]
    num_syn = [c for c in syn if c in numeric]
    specs = {
        "kyc_all": (num_syn, cat_real + cat_syn),
        "kyc_real": ([], cat_real),
        "txn": (txn_num, []),
        "txn_kyc_all": (txn_num + num_syn, cat_real + cat_syn),
        "txn_kyc_real": (txn_num, cat_real),
    }
    preds = {}
    for name, (num, cat) in specs.items():
        xtr, xva, is_cat = _matrix(tr, va, num, cat)
        preds[name] = _fit_predict(xtr, ytr, xva, is_cat)
    ap = {k: _sig(average_precision_score(yva, p)) for k, p in preds.items()}

    diffs = {
        "C1_kyc_only": ("kyc_all", None),
        "C2_synthetic_increment": ("kyc_all", "kyc_real"),
        "C3_txn_plus_kyc_increment": ("txn_kyc_all", "txn"),
        "C4_txn_synthetic_increment": ("txn_kyc_all", "txn_kyc_real"),
    }
    c = cfg["ceilings"]
    limits = {
        "C1_kyc_only": c["kyc_only_pr_auc_max_multiple_of_prev"] * prev,
        "C2_synthetic_increment": c["synthetic_increment_max_multiple_of_prev"] * prev,
        "C3_txn_plus_kyc_increment": c["txn_plus_kyc_increment_max_multiple_of_prev"] * prev,
        "C4_txn_synthetic_increment": c["txn_synthetic_increment_max_multiple_of_prev"] * prev,
    }
    rng = np.random.default_rng(0)
    boot = {k: [] for k in diffs}
    n = len(yva)
    for _ in range(c["bootstrap_reps"]):
        idx = rng.integers(0, n, n)
        if yva[idx].sum() == 0:
            continue
        a = {k: average_precision_score(yva[idx], p[idx]) for k, p in preds.items()}
        for k, (x, y) in diffs.items():
            boot[k].append(a[x] - (a[y] if y else 0.0))
    checks = {}
    for k, (x, y) in diffs.items():
        val = ap[x] - (ap[y] if y else 0.0)
        lo, hi = np.percentile(boot[k], [2.5, 97.5]) if boot[k] else (None, None)
        checks[k] = {
            "value": _sig(val),
            "ceiling": _sig(limits[k]),
            "pass": bool(val <= limits[k]),
            "ci95": [_sig(lo), _sig(hi)] if lo is not None else None,
            "value_over_prev": _sig(val / prev),
        }
    return res | {
        "status": "measured",
        "val_prevalence": _sig(prev),
        "pr_auc": ap,
        "checks": checks,
        "all_pass": all(v["pass"] for v in checks.values()),
    }
