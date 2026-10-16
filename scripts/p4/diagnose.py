"""P4 step 0: diagnose the seen/unseen gap before any model choice or lever (open risk 36).

TRAIN ONLY (folds F1, F2). VALIDATION, CALIBRATION and TEST are never read: labels come from
src.eval.training_labels with periods=["TRAIN"], and only data/p3/<variant>/features/TRAIN.parquet
is opened. Protocol and verdict rules: configs/p4_diagnosis.yaml (declared before this run).
Output: docs/p4/p4_diagnosis.json (aggregates only: no account key, no alert id).

    uv run python -m scripts.p4.diagnose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")

from src.alerts import rules  # noqa: E402
from src.data.periods import ROOT  # noqa: E402
from src.eval.identity_guard import load_v2_config  # noqa: E402
from src.eval.training_labels import load_fit_strata  # noqa: E402
from src.features import compute as fc  # noqa: E402
from src.features import registry as reg  # noqa: E402
from src.models import diagnostics as dg  # noqa: E402

VARIANT = "HI-Medium"
CONFIG = ROOT / "configs" / "p4_diagnosis.yaml"
FEATURES_ROOT = ROOT / "data" / "p3"
STORE_ROOT = ROOT / "data" / "p2"
P3_RESULTS = ROOT / "docs" / "p3" / "p3_results.json"
RESULTS = ROOT / "docs" / "p4" / "p4_diagnosis.json"
PERIODS = ["TRAIN"]  # the only period this script may read


def sha256_text(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def strata_masks(ev: pl.DataFrame) -> dict[str, np.ndarray]:
    """Masks over eval rows marking POSITIVES per stratum (built in src.eval; reporting only)."""
    lists = ev["strata"].to_list()
    names = sorted({k for x in lists for k in x})
    return {k: np.array([k in x for x in lists], dtype=bool) for k in names}


def build_folds(df: pl.DataFrame, cols: list[str], cfg: dict) -> list[dg.FoldData]:
    spec = {f["name"]: f for f in load_v2_config()["folds"]}
    folds = []
    for name in cfg["folds"]:
        s = spec[name]
        fit_days = [date.fromisoformat(d) for d in s["fit_days"]]
        eval_day = date.fromisoformat(s["eval_day"])
        fit = df.filter(pl.col("day").is_in(fit_days))
        ev = df.filter(pl.col("day") == eval_day)
        if fit.is_empty() or ev.is_empty():
            raise SystemExit(f"fold {name}: no rows (fit {fit.height}, eval {ev.height})")
        folds.append(dg.FoldData(
            name=name,
            X_fit=fit.select(cols).to_numpy(), y_fit=fit["y"].to_numpy(),
            acct_fit=fit["account_key"].to_numpy(),
            X_eval=ev.select(cols).to_numpy(), y_eval=ev["y"].to_numpy(),
            acct_eval=ev["account_key"].to_numpy(),
            strata=strata_masks(ev),
        ))  # fmt: skip
    return folds


def load_train(feat_dir: Path, variant: str, store_root: Path, strata_cfg: dict) -> pl.DataFrame:
    x = pl.read_parquet(feat_dir / "TRAIN.parquet")
    if set(x["split_period"].unique().to_list()) != {"TRAIN"}:
        raise SystemExit("TRAIN.parquet holds rows of another period")
    lab = load_fit_strata(PERIODS, variant, store_root, strata_cfg)
    j = x.join(lab.drop("period"), on="alert_id", how="inner")
    if j.height != x.height:
        raise SystemExit("feature rows and label rows do not match")
    return j.with_columns(pl.col("window_start").dt.date().alias("day"))


def run(a: argparse.Namespace) -> dict:
    t0 = time.time()
    cfg = load_config(a.config)
    if cfg.get("model") != "HistGradientBoostingClassifier":
        raise SystemExit("configs/p4_diagnosis.yaml: unsupported model")
    ctx = fc.make_context(
        rules.load_rules(a.rules), rules.load_thresholds(a.thresholds), fc.load_features_config()
    )
    feat_dir = a.features_root / a.variant / "features"
    # the matrix must be the one P3 recorded (same features, same rows)
    p3 = json.loads(a.p3_results.read_text(encoding="utf-8"))
    want = p3["features"]["periods"]["TRAIN"]["parquet_sha256"]
    got = sha256_file(feat_dir / "TRAIN.parquet")
    if got != want:
        raise SystemExit(f"TRAIN.parquet sha256 {got[:16]} != P3 record {want[:16]}")
    df = load_train(feat_dir, a.variant, a.store_root, cfg["strata"])
    all_cols = reg.model_features(ctx)
    sets = {"rule": reg.model_features(ctx, ["rule"]), "all": all_cols}
    if set(sets["rule"]) - set(all_cols):
        raise SystemExit("rule group is not a subset of the model features")

    def factory(seed: int):
        return HistGradientBoostingClassifier(random_state=seed)

    seeds = list(cfg["seeds"])
    rc, uc = cfg["recognition"], cfg["univariate"]
    out: dict = {
        "inputs": {
            "config_sha256": sha256_text(a.config),
            "feature_registry_sha256": reg.registry_hash(ctx),
            "train_parquet_sha256": got,
            "periods_read": PERIODS,
            "n_features": {k: len(v) for k, v in sets.items()},
        },
        "recognition": {},
    }
    full_all = None
    for name, cols in sets.items():
        folds = build_folds(df, cols, cfg)
        scores = {f.name: dg.fold_scores(f, factory, seeds, rc["boot_seed"]) for f in folds}
        out["recognition"][name] = dg.recognition_probe(
            folds, factory, seeds, rc["n_boot"], rc["boot_seed"], rc["ci"], rc["min_seen_pos"],
            scores=scores, max_control_shortfall=rc["max_control_shortfall"],
        )  # fmt: skip
        print(f"recognition [{name}]: {out['recognition'][name]['verdict']}", flush=True)
        if name == "all":
            full_all = (folds, {k: v["full"] for k, v in scores.items()})
    folds, full = full_all
    uni = dg.univariate(folds, all_cols, uc["n_boot"], uc["boot_seed"], rc["ci"], uc["top_k"])
    out["univariate"] = uni
    out["strata"] = dg.strata_auc(folds, full, cfg["strata"]["min_stratum_pos"])
    out["finished_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["seconds"] = round(time.time() - t0)
    return out


def summary(doc: dict) -> str:
    def f(x):
        return "  n/a" if x is None else f"{x:.3f}"

    lines = []
    for name, r in doc["recognition"].items():
        p, c = r["pooled"], r["ci"]
        lines.append(f"Q-A [{name}] verdict={r['verdict']}  seen pos={r['n_seen_pos']}  "
                     f"control top-up {r['control_shortfall_share']:.0%}  "
                     f"unseen pos={r['n_unseen_pos']}")  # fmt: skip
        lines.append(f"   AUC seen: full {f(p['full'])}  control {f(p['control'])}  "
                     f"disjoint {f(p['disjoint'])}  "
                     f"label-memory {f(p['label_memory'])}")  # fmt: skip
        lines.append(f"   AUC unseen full {f(p['unseen_full'])} CI {c['unseen_full']}; "
                     f"control-disjoint CI {c['control_minus_disjoint']}")  # fmt: skip
        for x in r["per_fold"]:
            lines.append(f"   {x['fold']}: prevalence seen {f(x['prevalence_seen'])} "
                         f"unseen {f(x['prevalence_unseen'])}")  # fmt: skip
    u = doc["univariate"]
    lines.append(f"Q-B features with unseen AUC CI lower > 0.55: {u['n_ci_lower_above_0_55']}"
                 f" of {u['n_features']}")  # fmt: skip
    for r in u["top"][:10]:
        lines.append(f"   {r['feature']:<28}{r['orientation']} unseen {f(r['auc_unseen'])} "
                     f"CI {r['ci_unseen']}  seen {f(r['auc_seen_same_orientation'])}")  # fmt: skip
    s = doc["strata"]
    lines.append(f"Q-C unseen positives: {s['total_unseen_pos']}")
    for k, v in s["strata"].items():
        lines.append(f"   {k:<26} n={v['n_pos_unseen']:<5} share={f(v['share_of_unseen_pos'])} "
                     f"AUC={f(v['auc_unseen'])}")  # fmt: skip
    return "\n".join(lines)


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--features-root", type=Path, default=FEATURES_ROOT)
    ap.add_argument("--store-root", type=Path, default=STORE_ROOT)
    ap.add_argument("--p3-results", type=Path, default=P3_RESULTS)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--rules", type=Path, default=rules.RULES_CONFIG)
    ap.add_argument("--thresholds", type=Path, default=rules.THRESHOLDS_FILE)
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
