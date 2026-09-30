"""P2 diagnostic: WHY does KYC alone predict laundering (C1, C2 failed)? TRAIN and VALIDATION only.

Scope (Dani's rule, 2026-09-30: nothing touches TEST without asking first):
  - alerts of TRAIN and VALIDATION only (runtime store, filtered on read);
  - labels of exactly those alerts (evaluation store, semi-joined on read; no other row kept);
  - the KYC store (no labels) and the configs.
The planting record is NOT read (its laundering flag uses every period, including TEST).

Measures:
  A. KYC-only PR-AUC (fit TRAIN, evaluate VALIDATION) adding one attribute at a time.
  B. Rules-only vs rules + KYC (confirms whether KYC adds anything beyond which rules fired).
  C. How often the account's sector matches an explanation archetype of the alert's own rules,
     by label and by rule (planting should make this equal for real and false alerts).
  D. 'Future alert' hint: among accounts with NO TRAIN alert, does an archetype sector or a
     high declared activity band raise the chance of a VALIDATION alert / true VALIDATION alert?

Usage (repo root): uv run python -m scripts.p2.diagnose_kyc
Writes docs/p2/p2_kyc_diagnostics.json (aggregates only).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from scripts.p2 import kyc_leakage as kl
from scripts.p2.run_p2 import ROOT, Paths, _sig, write_text_lf
from src.alerts import rules
from src.data import kyc as kycmod

PERIODS = ("TRAIN", "VALIDATION")


def _load(P: Paths) -> tuple[pl.DataFrame, pl.DataFrame]:
    alerts = pl.scan_parquet(P.alerts).filter(pl.col("period").is_in(PERIODS)).collect()
    if set(alerts["period"].unique()) - set(PERIODS):
        raise RuntimeError("non TRAIN/VALIDATION alert loaded")
    labels = (
        pl.scan_parquet(P.labels)
        .join(alerts.select("alert_id").lazy(), on="alert_id", how="semi")
        .select("alert_id", "is_true_positive")
        .collect()
    )
    if labels.height != alerts.height:
        raise RuntimeError("label rows do not match TRAIN/VALIDATION alerts")
    return alerts, labels


def _ap(tr: pl.DataFrame, va: pl.DataFrame, num: list[str], cat: list[str]) -> float:
    xtr, xva, is_cat = kl._matrix(tr, va, num, cat)
    ytr = tr["is_true_positive"].to_numpy().astype(int)
    yva = va["is_true_positive"].to_numpy().astype(int)
    return _sig(average_precision_score(yva, kl._fit_predict(xtr, ytr, xva, is_cat)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="HI-Medium")
    ap.add_argument("--out-root", type=Path, default=ROOT / "data" / "p2")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "p2" / "p2_kyc_diagnostics.json")
    args = ap.parse_args(argv)

    kcfg = kycmod.load_kyc_config()
    rcfg = rules.load_rules()
    rule_ids = rules.rule_ids(rcfg)
    P = Paths(args.out_root, args.variant, ROOT / "configs")
    alerts, labels = _load(P)
    kyc = pl.read_parquet(P.kyc)
    arche = kcfg["planting"]["archetypes"]
    arche_all = sorted(
        {s for v in arche.values() for s in v} | {kcfg["planting"]["individual_occupation"]}
    )

    d = alerts.join(labels, on="alert_id").join(kyc, on="account_key", how="left")
    d = kl._design(d, rule_ids, datetime.fromisoformat(kcfg["kyc_as_of"]))
    match = [
        any(s in arche.get(r, []) for r in trig) or s == kcfg["planting"]["individual_occupation"]
        for s, trig in zip(d["sector_or_occupation"], d["triggered_rules"].to_list(), strict=True)
    ]
    d = d.with_columns(pl.Series("sector_matches_rule", np.array(match, dtype=np.int8)))
    tr, va = d.filter(pl.col("period") == "TRAIN"), d.filter(pl.col("period") == "VALIDATION")
    prev = float(va["is_true_positive"].mean())
    out: dict = {"scope": "TRAIN and VALIDATION only", "val_prevalence": _sig(prev)}

    # ---- A. KYC-only ablation
    real = ["entity_type", "bank_country", "country_risk"]
    steps = [
        ("real (entity type, country, country risk)", [], real),
        ("+ expected_activity_band", [], [*real, "expected_activity_band"]),
        ("+ sector_or_occupation", [], [*real, "expected_activity_band", "sector_or_occupation"]),
        (
            "+ customer_risk_rating",
            [],
            [*real, "expected_activity_band", "sector_or_occupation", "customer_risk_rating"],
        ),
        (
            "+ onboarding_age_days (= all KYC)",
            ["onboarding_age_days"],
            [*real, "expected_activity_band", "sector_or_occupation", "customer_risk_rating"],
        ),
        ("sector_or_occupation alone", [], ["sector_or_occupation"]),
        ("expected_activity_band alone", [], ["expected_activity_band"]),
        ("sector_matches_rule flag alone", ["sector_matches_rule"], []),
    ]
    out["A_kyc_only_pr_auc"] = {}
    for name, num, cat in steps:
        v = _ap(tr, va, num, cat)
        out["A_kyc_only_pr_auc"][name] = {"pr_auc": v, "x_prev": _sig(v / prev)}

    # ---- B. rules vs rules + KYC
    fired = [f"fired_{r}" for r in rule_ids]
    all_cat = [*real, "expected_activity_band", "sector_or_occupation", "customer_risk_rating"]
    b = {
        "fired rules only": _ap(tr, va, fired, []),
        "fired rules + all KYC": _ap(tr, va, [*fired, "onboarding_age_days"], all_cat),
        "fired rules + sector_matches_rule": _ap(tr, va, [*fired, "sector_matches_rule"], []),
    }
    out["B_rules_vs_rules_plus_kyc"] = {
        k: {"pr_auc": v, "x_prev": _sig(v / prev)} for k, v in b.items()
    }

    # ---- C. archetype match rate by label and rule (TRAIN + VALIDATION)
    both = pl.concat([tr, va])
    out["C_sector_matches_rule_rate"] = {
        "true_alerts": _sig(both.filter(pl.col("is_true_positive"))["sector_matches_rule"].mean()),
        "false_alerts": _sig(
            both.filter(~pl.col("is_true_positive"))["sector_matches_rule"].mean()
        ),
        "by_rule": {
            r: {
                "true_rate": _sig(
                    both.filter(pl.col(f"fired_{r}").cast(pl.Boolean) & pl.col("is_true_positive"))[
                        "sector_matches_rule"
                    ].mean()
                    or 0
                ),
                "false_rate": _sig(
                    both.filter(
                        pl.col(f"fired_{r}").cast(pl.Boolean) & ~pl.col("is_true_positive")
                    )["sector_matches_rule"].mean()
                    or 0
                ),
                "alert_precision": _sig(
                    both.filter(pl.col(f"fired_{r}").cast(pl.Boolean))["is_true_positive"].mean()
                    or 0
                ),
            }
            for r in rule_ids
            if both[f"fired_{r}"].sum() > 0
        },
    }

    # ---- D. future-alert hint (accounts with no TRAIN alert)
    train_accts = tr.select("account_key").unique()
    val_alerted = (
        va.select("account_key", "is_true_positive")
        .group_by("account_key")
        .agg(pl.col("is_true_positive").any().alias("val_true"))
    )
    pool = (
        kyc.join(train_accts, on="account_key", how="anti")
        .select("account_key", "sector_or_occupation", "expected_activity_band")
        .join(val_alerted, on="account_key", how="left")
        .with_columns(
            pl.col("val_true").is_not_null().alias("val_alerted"),
            pl.col("val_true").fill_null(False),
            pl.col("sector_or_occupation").is_in(arche_all).alias("archetype_sector"),
            pl.col("expected_activity_band").is_in(["high", "very_high"]).alias("high_band"),
        )
    )
    dd = {"n_accounts_without_train_alert": pool.height}
    for flag in ("archetype_sector", "high_band"):
        yes, no = pool.filter(pl.col(flag)), pool.filter(~pl.col(flag))
        dd[flag] = {
            "share_of_accounts": _sig(yes.height / pool.height),
            "val_alert_rate_if_flag": _sig(yes["val_alerted"].mean()),
            "val_alert_rate_if_not": _sig(no["val_alerted"].mean()),
            "val_alert_lift": _sig(yes["val_alerted"].mean() / no["val_alerted"].mean()),
            "val_true_alert_rate_if_flag": _sig(yes["val_true"].mean()),
            "val_true_alert_rate_if_not": _sig(no["val_true"].mean()),
            "val_true_alert_lift": _sig(yes["val_true"].mean() / max(no["val_true"].mean(), 1e-12)),
        }
    out["D_future_alert_hint"] = dd

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(args.out, json.dumps(_sig(out), indent=2) + "\n")
    print(json.dumps(_sig(out), indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
