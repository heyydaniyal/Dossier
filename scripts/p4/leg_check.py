"""P4 step 0b: per-payment signal check for NEW accounts (configs/p4_leg_check.yaml).

TRAIN ONLY (folds F1, F2). Reads: TRAIN transactions (point-in-time windows), the runtime alert
store (TRAIN alerts), the P3 TRAIN feature matrix (hash-checked against the P3 record), and through
src.eval.training_labels: TRAIN alert labels/strata and TRAIN laundering transactions (the one-off
payment-label use approved 2026-10-16). Output: docs/p4/p4_leg_check.json, aggregates only.

    uv run python -m scripts.p4.leg_check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.p4 import diagnose as d0  # noqa: E402
from src.alerts import rules  # noqa: E402
from src.data import fx as fxmod  # noqa: E402
from src.data.hashing import keyed_u64  # noqa: E402
from src.data.periods import ROOT, load_split  # noqa: E402
from src.data.stores import scan_runtime  # noqa: E402
from src.data.transactions import interim_path, scan_transactions, verify_interim  # noqa: E402
from src.eval.identity_guard import load_v2_config  # noqa: E402
from src.eval.training_labels import load_fit_laundering_txns  # noqa: E402
from src.features import compute as fc  # noqa: E402
from src.features import legs as fl  # noqa: E402
from src.features import registry as reg  # noqa: E402
from src.models import leg_check as lc  # noqa: E402

VARIANT = "HI-Medium"
CONFIG = ROOT / "configs" / "p4_leg_check.yaml"
PROTOCOL = ROOT / "configs" / "p4_protocol.yaml"
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "docs" / "p4" / "p4_leg_check.json"
PERIODS = ["TRAIN"]  # the only period this script may read


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def label_legs(legs: pl.DataFrame, laund: pl.DataFrame) -> pl.DataFrame:
    lab = laund.select(fl.KEY_COLUMNS).with_columns(pl.lit(1, pl.Int8).alias("y_leg"))
    out = legs.join(lab, on=fl.KEY_COLUMNS, how="left").with_columns(pl.col("y_leg").fill_null(0))
    if out.height != legs.height:
        raise SystemExit("payment-label join changed the number of legs")
    return out


def leg_hash(legs: pl.DataFrame, key: str, seed: int) -> np.ndarray:
    """Keyed hash of the transaction's raw columns: an order no ID or file order can leak (D5)."""
    s = legs.select(pl.concat_str([pl.col(c).cast(pl.String) for c in fl.KEY_COLUMNS],
                                  separator="\x1f")).to_series().to_list()  # fmt: skip
    return keyed_u64(s, key, seed)


def sample_fit_legs(legs: pl.DataFrame, cap: int, key: str, seed: int) -> pl.DataFrame:
    """All laundering legs + at most `cap` normal legs per alert (smallest keyed hash), weighted by
    1 / sampling rate. Deterministic row order: (alert_id, keyed hash)."""
    df = legs.with_columns(pl.Series("_h", leg_hash(legs, key, seed))).sort("alert_id", "_h")
    df = df.with_columns(
        (pl.col("y_leg") == 0).cum_sum().over("alert_id").alias("_k"),
        (pl.col("y_leg") == 0).sum().over("alert_id").alias("_n_norm"),
    )
    keep = df.filter((pl.col("y_leg") == 1) | (pl.col("_k") <= cap))
    w = (
        pl.when(pl.col("y_leg") == 1)
        .then(1.0)
        .otherwise(pl.col("_n_norm").cast(pl.Float64) / pl.min_horizontal(pl.col("_n_norm"), cap))
    )
    return keep.with_columns(w.alias("w")).drop("_k", "_n_norm")


def run(a: argparse.Namespace) -> dict:
    t0 = time.time()
    cfg, dcfg, pcfg = load_yaml(a.config), d0.load_config(a.diag_config), load_yaml(a.protocol)
    split = load_split()
    ctx = fc.make_context(
        rules.load_rules(a.rules), rules.load_thresholds(a.thresholds), fc.load_features_config()
    )
    # P3 TRAIN matrix must be the recorded one
    feat_dir = a.features_root / a.variant / "features"
    p3 = json.loads(a.p3_results.read_text(encoding="utf-8"))
    got = d0.sha256_file(feat_dir / "TRAIN.parquet")
    if got != p3["features"]["periods"]["TRAIN"]["parquet_sha256"]:
        raise SystemExit("TRAIN.parquet differs from the P3 record")
    alerts_x = d0.load_train(feat_dir, a.variant, a.store_root, dcfg["strata"])
    cols = reg.model_features(ctx)
    # transactions (label-free) and TRAIN laundering transactions (eval module)
    verify_interim(a.interim, a.variant, a.sources)
    trans = fxmod.with_usd(
        scan_transactions(interim_path(a.interim, a.variant, "trans")), fxmod.load(a.fx)
    )
    laund = load_fit_laundering_txns(PERIODS, a.variant, a.interim, split)
    store = scan_runtime("alerts.parquet", a.variant, a.store_root).filter(
        pl.col("period") == "TRAIN"
    ).select("alert_id", "account_key", "as_of").collect()  # fmt: skip

    lib = pcfg["library"]
    params = {**cfg["model"]["params"], **lib["fixed_params"], "random_state": lib["seed"]}
    names_all = fl.leg_feature_names(ctx, with_format=True)
    nofmt = np.array([i for i, n in enumerate(names_all) if not n.startswith("leg_fmt_")])
    spec = {f["name"]: f for f in load_v2_config()["folds"]}
    folds, counts = [], []
    for name in cfg["folds"]:
        s = spec[name]
        fit_days = [date.fromisoformat(x) for x in s["fit_days"]]
        eval_day = date.fromisoformat(s["eval_day"])
        fit_a = alerts_x.filter(pl.col("day").is_in(fit_days))
        ev_a = alerts_x.filter(pl.col("day") == eval_day)
        # baseline = step-0 full model (HGB, 61 features, mean over seeds)
        base = np.mean([
            HistGradientBoostingClassifier(random_state=sd)
            .fit(fit_a.select(cols).to_numpy(), fit_a["y"].to_numpy())
            .predict_proba(ev_a.select(cols).to_numpy())[:, 1]
            for sd in dcfg["seeds"]
        ], axis=0)  # fmt: skip
        st = store.join(fit_a.select("alert_id"), on="alert_id", how="semi")
        se = store.join(ev_a.select("alert_id"), on="alert_id", how="semi")
        fit_legs = label_legs(fl.batch_legs(trans, st, ctx), laund)
        fit_legs = sample_fit_legs(
            fit_legs, cfg["model"]["max_normal_legs_per_alert"], cfg["model"]["sample_key"],
            lib["seed"],
        )  # fmt: skip
        ev_legs = label_legs(fl.batch_legs(trans, se, ctx), laund)
        ev_legs = ev_legs.with_columns(
            pl.Series("_h", leg_hash(ev_legs, cfg["model"]["sample_key"], lib["seed"]))
        ).sort("alert_id", "_h")
        pos = {aid: i for i, aid in enumerate(ev_a["alert_id"].to_list())}
        seen_keys = set(fit_a["account_key"].to_list())
        folds.append(lc.LegFold(
            name=name,
            X_fit=fl.leg_matrix(fit_legs, names_all), y_fit=fit_legs["y_leg"].to_numpy(),
            w_fit=fit_legs["w"].to_numpy(),
            X_eval=fl.leg_matrix(ev_legs, names_all),
            leg_alert=np.array([pos[x] for x in ev_legs["alert_id"].to_list()]),
            y_leg_eval=ev_legs["y_leg"].to_numpy(), y_alert=ev_a["y"].to_numpy(),
            seen=np.array([k in seen_keys for k in ev_a["account_key"].to_list()]),
            baseline=base, strata=d0.strata_masks(ev_a),
        ))  # fmt: skip
        counts.append({
            "fold": name, "fit_alerts": fit_a.height, "fit_legs_sampled": fit_legs.height,
            "fit_positive_payments": int(fit_legs["y_leg"].sum()),
            "eval_alerts": ev_a.height, "eval_legs": ev_legs.height,
            "eval_positive_payments": int(ev_legs["y_leg"].sum()),
        })  # fmt: skip
        print(f"{name}: {counts[-1]}", flush=True)

    def factory():
        return lgb.LGBMClassifier(**params)

    cc = cfg["comparison"]
    leg_s = {f.name: lc.leg_scores(f, factory) for f in folds}
    leg_s_nofmt = {f.name: lc.leg_scores(f, factory, nofmt) for f in folds}
    amax = {f.name: lc.max_by_alert(leg_s[f.name], f.leg_alert, len(f.y_alert)) for f in folds}
    amax_nofmt = {
        f.name: lc.max_by_alert(leg_s_nofmt[f.name], f.leg_alert, len(f.y_alert)) for f in folds
    }
    args = (cc["n_boot"], cc["boot_seed"], cc["ci"], cc["material_gain_auc"], cfg["min_pos_unseen"])
    leg_auc = [
        {"fold": f.name, "auc_payment_level_unseen": lc.dg.auc(
            f.y_leg_eval[~f.seen[f.leg_alert]], leg_s[f.name][~f.seen[f.leg_alert]])}
        for f in folds
    ]  # fmt: skip
    return {
        "inputs": {
            "config_sha256": d0.sha256_text(a.config),
            "protocol_sha256": d0.sha256_text(a.protocol),
            "train_parquet_sha256": got,
            "periods_read": PERIODS,
            "leg_features": names_all,
            "lightgbm_version": lgb.__version__,
        },
        "counts": counts,
        "primary": lc.compare(folds, amax, *args),
        "without_payment_format": lc.compare(folds, amax_nofmt, *args),
        "within_alert": lc.within_alert(folds, leg_s),
        "payment_level": leg_auc,
        "strata": lc.strata(folds, amax, dcfg["strata"]["min_stratum_pos"]),
        "finished_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seconds": round(time.time() - t0),
    }


def summary(doc: dict) -> str:
    def f(x):
        return "  n/a" if x is None else f"{x:.3f}"

    L = []
    for k in ("primary", "without_payment_format"):
        r = doc[k]
        L.append(f"[{k}] verdict={r['verdict']}  unseen pos={r['n_unseen_pos']}")
        L.append(f"   AUC unseen: payment-max {f(r['auc_payment_max'])}  baseline "
                 f"{f(r['auc_baseline'])}  gain {f(r['gain'])} CI {r['gain_ci']}")  # fmt: skip
    w = doc["within_alert"]
    L.append(f"within positive unseen alerts (n={w['n_alerts']}): top payment is a positive one "
             f"{f(w['top_payment_hit_rate'])} vs chance {f(w['chance_rate'])}; "
             f"mean within-alert AUC {f(w['mean_within_alert_auc'])}")  # fmt: skip
    for x in doc["payment_level"]:
        L.append(f"payment-level AUC unseen {x['fold']}: {f(x['auc_payment_level_unseen'])}")
    L.append("strata (unseen positives vs unseen negatives):")
    for k, v in doc["strata"].items():
        L.append(f"   {k:<26} n={v['n_pos_unseen']:<5} payment-max {f(v['auc_payment_max'])} "
                 f"baseline {f(v['auc_baseline'])}")  # fmt: skip
    for c in doc["counts"]:
        L.append(f"   {c}")
    return "\n".join(L)


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--diag-config", type=Path, default=d0.CONFIG)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    ap.add_argument("--interim", type=Path, default=INTERIM)
    ap.add_argument("--features-root", type=Path, default=d0.FEATURES_ROOT)
    ap.add_argument("--store-root", type=Path, default=d0.STORE_ROOT)
    ap.add_argument("--p3-results", type=Path, default=d0.P3_RESULTS)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--rules", type=Path, default=rules.RULES_CONFIG)
    ap.add_argument("--thresholds", type=Path, default=rules.THRESHOLDS_FILE)
    ap.add_argument("--fx", type=Path, default=fc.FX_FILE)
    ap.add_argument("--sources", type=Path, default=ROOT / "configs" / "data_sources.yaml")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse(argv)
    doc = run(a)
    a.results.parent.mkdir(parents=True, exist_ok=True)
    a.results.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(summary(doc))
    print(f"\nwrote {a.results} ({doc['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
