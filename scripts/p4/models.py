"""Phase 4 detection models (configs/p4_models.yaml under the frozen configs/p4_protocol.yaml).

Stages, run in order on Dani's laptop (precheck added 2026-10-16, configs/p4_precheck.yaml):
  cv        TRAIN ONLY. Time-series CV (folds F1, F2), all configurations of the declared search
            space, learning curves, fold stability, seen/unseen, permutation importance. Writes
            docs/p4/p4_cv.json including the FINALISTS. Commit it before running validate.
  precheck  TRAIN ONLY. Explanation shares of the learned finalists on the CV folds (would the
            "one feature dominates" stop rule fire?) and the rules_weighted weights.
  validate  Opens VALIDATION ONCE (refused if docs/p4/p4_validation.json exists). Fits the finalists
            on TRAIN, evaluates them on VALIDATION with bootstrap CIs, applies the declared
            selection
            rule and stop rules, runs identity guard v2 on the chosen feature set, sets the
            cost-based operating point, saves the model BY HASH (models/p4/) and writes
            docs/p4/p4_validation.json and docs/p4/MODEL_CARD.md.
CALIBRATION and TEST are never read. Labels only via src.eval.training_labels. Aggregates only.

    uv run python -m scripts.p4.models cv
    uv run python -m scripts.p4.models validate
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

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")

from src.alerts import rules  # noqa: E402
from src.data.hashing import keyed_u64  # noqa: E402
from src.data.periods import ROOT, load_split  # noqa: E402
from src.eval.identity_guard import evaluate_v2, load_v2_config  # noqa: E402
from src.eval.training_labels import load_fit_labels, load_fit_strata  # noqa: E402
from src.features import compute as fc  # noqa: E402
from src.features import registry as reg  # noqa: E402
from src.models import families as fam  # noqa: E402
from src.models import metrics as mt  # noqa: E402
from src.models import store  # noqa: E402

VARIANT = "HI-Medium"
CONFIG = ROOT / "configs" / "p4_models.yaml"
PROTOCOL = ROOT / "configs" / "p4_protocol.yaml"
DIAG_CONFIG = ROOT / "configs" / "p4_diagnosis.yaml"  # strata definitions (reporting only)
FEATURES_ROOT = ROOT / "data" / "p3"
STORE_ROOT = ROOT / "data" / "p2"
P3_RESULTS = ROOT / "docs" / "p3" / "p3_results.json"
CV_RESULTS = ROOT / "docs" / "p4" / "p4_cv.json"
PRECHECK_CONFIG = ROOT / "configs" / "p4_precheck.yaml"
PRECHECK_RESULTS = ROOT / "docs" / "p4" / "p4_precheck.json"
VAL_RESULTS = ROOT / "docs" / "p4" / "p4_validation.json"
MODEL_CARD = ROOT / "docs" / "p4" / "MODEL_CARD.md"
CANDIDATE_ORDER = ["rules", "logistic", "lightgbm"]  # configs/p4_protocol.yaml simplicity_order


def sha_text(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- environment


class Env:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.cfg, self.protocol = load_yaml(a.config), load_yaml(a.protocol)
        self.diag = load_yaml(a.diag_config)
        self.ctx = fc.make_context(
            rules.load_rules(a.rules),
            rules.load_thresholds(a.thresholds),
            fc.load_features_config(),
        )
        allf = reg.model_features(self.ctx)
        fmt = set(fc.group_names(self.ctx)["format_mix"])
        self.sets = {"all": allf, "no_format": [f for f in allf if f not in fmt]}
        cfs, sfs = self.cfg["candidate_feature_set"], self.cfg["sensitivity_feature_set"]
        if cfs not in self.sets or sfs not in self.sets or cfs == sfs:
            raise SystemExit("candidate and sensitivity feature sets must be two different sets")
        self.rule_cols = reg.model_features(self.ctx, ["rule"])
        self.feat_dir = a.features_root / a.variant / "features"
        self.p3 = json.loads(a.p3_results.read_text(encoding="utf-8"))
        lib = self.protocol["library"]
        self.lgb_fixed = {
            **self.cfg["cv"]["families"]["lightgbm"]["fixed"],
            **lib["fixed_params"],
            "random_state": lib["seed"],
        }

    def inputs(self) -> dict:
        return {
            "models_config_sha256": sha_text(self.a.config),
            "protocol_sha256": sha_text(self.a.protocol),
            "feature_registry_sha256": reg.registry_hash(self.ctx),
        }

    def period(self, p: str, strata: bool = False) -> pl.DataFrame:
        if p not in ("TRAIN", "VALIDATION"):
            raise SystemExit(f"P4 reads TRAIN and VALIDATION only, not {p}")
        path = self.feat_dir / f"{p}.parquet"
        want = self.p3["features"]["periods"][p]["parquet_sha256"]
        if sha_file(path) != want:
            raise SystemExit(f"{p}.parquet differs from the P3 record")
        x = pl.read_parquet(path)
        if set(x["split_period"].unique().to_list()) != {p}:
            raise SystemExit(f"{p}.parquet holds rows of another period")
        if strata:
            lab = load_fit_strata([p], self.a.variant, self.a.store_root, self.diag["strata"])
        else:
            lab = load_fit_labels([p], self.a.variant, self.a.store_root)
        j = x.join(lab.drop("period"), on="alert_id", how="inner")
        if j.height != x.height:
            raise SystemExit(f"{p}: feature rows and label rows do not match")
        fired = [c for c in self.sets["all"] if c.startswith("fired_")]
        if (j.select(pl.sum_horizontal(fired)).to_series() != j["n_rules_triggered"]).any():
            raise SystemExit(f"{p}: n_rules_triggered != number of fired_* flags")
        return j.with_columns(pl.col("window_start").dt.date().alias("day"))


# ---------------------------------------------------------------- configurations


def enumerate_configs(cfg: dict) -> list[dict]:
    """The declared search space (sensitivity configs are added after the CV choice)."""
    fams = cfg["cv"]["families"]
    fs = cfg["candidate_feature_set"]
    out = [
        {"id": "rules_count", "family": "rules_count", "params": {}, "feature_set": fs},
        {
            "id": "rules_weighted",
            "family": "rules_weighted",
            "params": dict(fams["rules_weighted"]),
            "feature_set": fs,
        },
    ]
    lr = fams["logistic"]
    for c in lr["C"]:
        for cw in lr["class_weight"]:
            out.append(
                {
                    "id": f"logistic_C{c}_{cw}",
                    "family": "logistic",
                    "params": {"C": c, "class_weight": cw, "max_iter": lr["max_iter"]},
                    "feature_set": fs,
                }
            )
    g = fams["lightgbm"]["grid"]
    for nl in g["num_leaves"]:
        for mc in g["min_child_samples"]:
            for cw in g["class_weight"]:
                out.append(
                    {
                        "id": f"lightgbm_l{nl}_m{mc}_{cw}",
                        "family": "lightgbm",
                        "params": {"num_leaves": nl, "min_child_samples": mc, "class_weight": cw},
                        "feature_set": fs,
                    }
                )
    return out


def build(env: Env, spec: dict, n_estimators: int | None = None):
    feats = env.sets[spec["feature_set"]]
    f = spec["family"]
    if f == "rules_count":
        return fam.RulesCount(feats)
    if f == "rules_weighted":
        return fam.RulesWeighted(feats, params=dict(spec["params"]))
    if f == "logistic":
        return fam.Logistic(feats, params=dict(spec["params"]))
    if f == "lightgbm":
        p = {**env.lgb_fixed, **spec["params"]}
        if n_estimators is not None:
            p["n_estimators"] = int(n_estimators)
        es = env.cfg["cv"]["families"]["lightgbm"]["early_stopping_rounds"]
        return fam.LightGBM(feats, params=p, early_stopping_rounds=es)
    raise SystemExit(f"unknown family {f}")


def family_group(f: str) -> str:
    return "rules" if f.startswith("rules") else f


def simplicity_key(spec: dict) -> tuple:
    """Ties in CV PR-AUC go to the simpler config (configs/p4_models.yaml within_family_choice)."""
    p = spec["params"]
    return (
        p.get("class_weight", "none") != "none",
        p.get("C", 0.0),
        p.get("num_leaves", 0),
        -p.get("min_child_samples", 0),
        spec["family"] != "rules_count",
    )


# ---------------------------------------------------------------- stage cv


def folds(env: Env, df: pl.DataFrame) -> list[dict]:
    spec = {f["name"]: f for f in load_v2_config()["folds"]}
    out = []
    for name in env.protocol["data_access"]["tuning"]["folds"]:
        s = spec[name]
        fit = df.filter(pl.col("day").is_in([date.fromisoformat(d) for d in s["fit_days"]]))
        ev = df.filter(pl.col("day") == date.fromisoformat(s["eval_day"]))
        if fit.is_empty() or ev.is_empty():
            raise SystemExit(f"fold {name}: empty fit or eval rows")
        seen = set(fit["account_key"].to_list())
        out.append(
            {
                "name": name,
                "fit": fit,
                "ev": ev,
                "seen": np.array([k in seen for k in ev["account_key"].to_list()]),
            }
        )
    return out


def fold_metrics(y, s, seen) -> dict:
    return {
        "n": int(len(y)),
        "pos": int(y.sum()),
        "pr_auc": mt.pr_auc(y, s),
        "roc_auc": mt.roc_auc(y, s),
        "pr_auc_seen": mt.pr_auc(y[seen], s[seen]),
        "pr_auc_unseen": mt.pr_auc(y[~seen], s[~seen]),
        "pos_seen": int(y[seen].sum()),
        "pos_unseen": int(y[~seen].sum()),
    }


def pooled(per_fold: list[dict], key: str, wkey: str = "pos") -> float | None:
    pairs = [(f[key], f[wkey]) for f in per_fold if f[key] is not None and f[wkey] > 0]
    return float(sum(v * w for v, w in pairs) / sum(w for _, w in pairs)) if pairs else None


def run_config(env: Env, spec: dict, fs: list[dict]) -> dict:
    cols = env.sets[spec["feature_set"]]
    per = []
    for f in fs:
        Xf, yf = f["fit"].select(cols).to_numpy(), f["fit"]["y"].to_numpy()
        Xe, ye = f["ev"].select(cols).to_numpy(), f["ev"]["y"].to_numpy()
        m = build(env, spec)
        m.fit(Xf, yf, eval_set=(Xe, ye) if spec["family"] == "lightgbm" else None)
        rec = fold_metrics(ye, m.margin(Xe), f["seen"])
        rec["fold"] = f["name"]
        rec["train_pr_auc"] = mt.pr_auc(yf, m.margin(Xf))
        if spec["family"] == "lightgbm":
            rec["best_iteration"] = m.best_iteration
        per.append(rec)
    out = {**spec, "per_fold": per}
    for k in ("pr_auc", "train_pr_auc", "roc_auc"):
        out[f"cv_{k}"] = pooled(per, k)
    out["cv_pr_auc_seen"] = pooled(per, "pr_auc_seen", "pos_seen")
    out["cv_pr_auc_unseen"] = pooled(per, "pr_auc_unseen", "pos_unseen")
    vals = [p["pr_auc"] for p in per if p["pr_auc"] is not None]
    out["fold_spread"] = float(max(vals) - min(vals)) if len(vals) > 1 else None
    if spec["family"] == "lightgbm":
        out["n_estimators"] = int(round(np.mean([p["best_iteration"] for p in per])))
    return out


def learning_curve(env: Env, spec: dict, fs: list[dict]) -> list[dict]:
    lc = env.cfg["cv"]["learning_curve"]
    cols = env.sets[spec["feature_set"]]
    out = []
    for frac in lc["fractions"]:
        per = []
        for f in fs:
            keys = f["fit"]["account_key"].to_list()
            u = keyed_u64(keys, lc["sample_key"], 0).astype(np.float64) / 2.0**64
            sub = f["fit"].filter(pl.Series(u < frac))
            m = build(env, spec)
            Xe, ye = f["ev"].select(cols).to_numpy(), f["ev"]["y"].to_numpy()
            ys = sub["y"].to_numpy()
            if ys.sum() == 0 or ys.sum() == len(ys):
                per.append({"fold": f["name"], "pr_auc": None, "train_pr_auc": None, "pos": 0})
                continue
            Xs = sub.select(cols).to_numpy()
            m.fit(Xs, ys, eval_set=(Xe, ye) if spec["family"] == "lightgbm" else None)
            per.append(
                {
                    "fold": f["name"],
                    "n_fit": sub.height,
                    "pos": int(ye.sum()),
                    "pr_auc": mt.pr_auc(ye, m.margin(Xe)),
                    "train_pr_auc": mt.pr_auc(ys, m.margin(Xs)),
                }
            )
        out.append(
            {
                "fraction": frac,
                "per_fold": per,
                "pr_auc": pooled(per, "pr_auc"),
                "train_pr_auc": pooled(per, "train_pr_auc"),
            }
        )
    return out


def permutation_importance(model, X, y, names, repeats: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    base = mt.pr_auc(y, model.margin(X))
    if base is None:  # one class on this day: importance undefined
        return [{"feature": n, "mean_drop": None} for n in names]
    out = []
    for j, n in enumerate(names):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            drops.append(base - mt.pr_auc(y, model.margin(Xp)))
        out.append({"feature": n, "mean_drop": float(np.mean(drops))})
    return sorted(out, key=lambda r: -r["mean_drop"])


def stage_cv(env: Env) -> dict:
    t0 = time.time()
    df = env.period("TRAIN")
    fs = folds(env, df)
    specs = enumerate_configs(env.cfg)
    results = []
    for s in specs:
        results.append(run_config(env, s, fs))
        r = results[-1]
        print(
            f"cv {s['id']:<32} PR-AUC {f3(r['cv_pr_auc'])}  unseen {f3(r['cv_pr_auc_unseen'])}",
            flush=True,
        )
    best = {}
    for g in CANDIDATE_ORDER:
        rs = [r for r in results if family_group(r["family"]) == g]
        best[g] = min(rs, key=lambda r: (-(r["cv_pr_auc"] or -1.0), simplicity_key(r)))
    sens = []
    for g in env.cfg["cv"]["sensitivity"]:
        s = {k: best[g][k] for k in ("family", "params")}
        sfs = env.cfg["sensitivity_feature_set"]
        s |= {"id": f"{best[g]['id']}__{sfs}", "feature_set": sfs}
        sens.append(run_config(env, s, fs))
        print(f"cv {s['id']:<32} PR-AUC {f3(sens[-1]['cv_pr_auc'])}", flush=True)
    n_configs = len(results) + len(sens)
    if n_configs > env.protocol["data_access"]["tuning"]["max_configs"]:
        raise SystemExit(f"{n_configs} configurations exceed the protocol maximum")

    def fin(r, role):
        d = {k: r[k] for k in ("id", "family", "params", "feature_set")}
        d["role"] = role
        if r["family"] == "lightgbm":
            d["n_estimators"] = r["n_estimators"]
        return d

    finalists = [fin(best[g], "candidate") for g in CANDIDATE_ORDER]
    finalists += [fin(r, "sensitivity") for r in sens]
    if len(finalists) > env.protocol["data_access"]["validation"]["max_finalists"]:
        raise SystemExit("too many finalists")
    curves = {g: learning_curve(env, best[g], fs) for g in ("logistic", "lightgbm")}
    pi = env.cfg["cv"]["permutation_importance"]
    imp = {}
    for g in ("logistic", "lightgbm"):
        cols = env.sets[best[g]["feature_set"]]
        per = {}
        for f in fs:
            m = build(env, best[g])
            Xe, ye = f["ev"].select(cols).to_numpy(), f["ev"]["y"].to_numpy()
            m.fit(
                f["fit"].select(cols).to_numpy(),
                f["fit"]["y"].to_numpy(),
                eval_set=(Xe, ye) if g == "lightgbm" else None,
            )
            per[f["name"]] = {
                r["feature"]: r["mean_drop"]
                for r in permutation_importance(m, Xe, ye, cols, pi["repeats"], pi["seed"])
            }
        names = env.sets[best[g]["feature_set"]]
        rows = []
        for n in names:
            r = {"feature": n, **{k: per[k][n] for k in per}}
            vals = [r[k] for k in per if r[k] is not None]
            r["sign_flip"] = len({np.sign(v) for v in vals} - {0.0}) > 1
            r["mean_drop"] = float(np.mean(vals)) if vals else None
            rows.append(r)
        rows.sort(key=lambda r: -(r["mean_drop"] if r["mean_drop"] is not None else -np.inf))
        imp[g] = {"top": rows[:15], "n_sign_flip": sum(r["sign_flip"] for r in rows)}
    return {
        "inputs": {
            **env.inputs(),
            "train_parquet_sha256": sha_file(env.feat_dir / "TRAIN.parquet"),
            "periods_read": ["TRAIN"],
            "n_configs": n_configs,
        },
        "configs": results,
        "sensitivity": sens,
        "best_per_family": {g: best[g]["id"] for g in CANDIDATE_ORDER},
        "finalists": finalists,
        "learning_curves": curves,
        "permutation_importance": imp,
        "finished_utc": now(),
        "seconds": round(time.time() - t0),
    }


# ---------------------------------------------------------------- stage validate


def evaluate_subset(y, s, probability: bool, vcfg: dict, with_ci: bool) -> dict:
    m = vcfg["metrics"]
    out = mt.summary(y, s, m["review_capacity_share"], m["precision_at_recall"], probability)
    if not with_ci or len(y) == 0:
        return out
    b = vcfg["bootstrap"]
    fns = {
        "pr_auc": mt.pr_auc,
        "roc_auc": mt.roc_auc,
        "recall_at_capacity_0.1": lambda yy, ss: mt.recall_at_capacity(yy, ss, 0.1),
        "precision_at_recall_0.8": lambda yy, ss: mt.precision_at_recall(yy, ss, 0.8),
    }
    if probability:
        fns["brier"] = mt.brier
    out["ci"] = {
        k: mt.bootstrap(lambda i, f=f: f(y[i], s[i]), len(y), b["n_boot"], b["seed"], b["ci"])[0]
        for k, f in fns.items()
    }
    return out


def paired_gain_ci(y, s_complex, s_simple, b: dict) -> list[float] | None:
    def g(i):
        a, c = mt.pr_auc(y[i], s_complex[i]), mt.pr_auc(y[i], s_simple[i])
        return None if a is None or c is None else a - c

    return mt.bootstrap(g, len(y), b["n_boot"], b["seed"], b["ci"])[0]


def select(results: dict, y: np.ndarray, scores: dict, b: dict) -> dict:
    """configs/p4_protocol.yaml selection: the simplest finalist wins unless a more complex one's
    PR-AUC gain over it has a paired bootstrap CI lower bound > 0; among those that pass, the
    highest PR-AUC wins."""
    cands = [results[g] for g in CANDIDATE_ORDER]
    simple = cands[0]
    tests = []
    for c in cands[1:]:
        ci = paired_gain_ci(y, scores[c["id"]], scores[simple["id"]], b)
        gain = c["all"]["pr_auc"] - simple["all"]["pr_auc"]
        tests.append(
            {
                "candidate": c["id"],
                "vs": simple["id"],
                "gain": gain,
                "gain_ci": ci,
                "passes": ci is not None and ci[0] > 0,
            }
        )
    passing = [c for c, t in zip(cands[1:], tests, strict=True) if t["passes"]]
    # highest PR-AUC among those that pass; an exact tie goes to the simpler (earlier) candidate
    chosen = (
        min(passing, key=lambda c: (-c["all"]["pr_auc"], cands.index(c))) if passing else simple
    )
    return {"tests": tests, "chosen": chosen["id"]}


def guard(env: Env, spec: dict, n_est: int | None, df: pl.DataFrame) -> dict:
    """Identity guard v2 (frozen config): base = rule group, with = the chosen feature set; the
    chosen family and params; folds F1-F3 (F3 eval day = VALIDATION)."""
    base_spec = {**spec, "feature_set": "_rule"}
    env.sets["_rule"] = env.rule_cols
    out_folds = []
    for s in load_v2_config()["folds"]:
        fit = df.filter(pl.col("day").is_in([date.fromisoformat(d) for d in s["fit_days"]]))
        ev = df.filter(pl.col("day") == date.fromisoformat(s["eval_day"]))
        seen = set(fit["account_key"].to_list())
        preds = {}
        for k, sp in (("pred_base", base_spec), ("pred_with_group", spec)):
            cols = env.sets[sp["feature_set"]]
            m = build(env, sp, n_estimators=n_est)
            m.fit(fit.select(cols).to_numpy(), fit["y"].to_numpy())
            preds[k] = m.margin(ev.select(cols).to_numpy())
        out_folds.append(
            {
                "name": s["name"],
                "window_start": ev["window_start"].to_list(),
                "y": ev["y"].to_numpy(),
                **preds,
                "seen": np.array([k in seen for k in ev["account_key"].to_list()]),
                "account_key": ev["account_key"].to_list(),
            }
        )
    return evaluate_v2(out_folds, split=load_split())


def explain(model, X, names, k: int) -> dict:
    c, _ = model.contributions(X)
    a = np.abs(c).mean(axis=0)
    order = np.argsort(-a)
    total = float(a.sum())
    return {
        "output_space": "log-odds margin of the uncalibrated model",
        "top": [
            {
                "feature": names[j],
                "mean_abs": float(a[j]),
                "mean_signed": float(c[:, j].mean()),
                "share": float(a[j] / total) if total else None,
            }
            for j in order[:k]
        ],
        "top_share": float(a[order[0]] / total) if total else None,
    }


def precheck_reading(worst: float, near: float, stop: float) -> str:
    """configs/p4_precheck.yaml reading of the worst candidate top-feature share."""
    if worst > stop:
        return "stop_rule_would_fire"
    if worst >= near:
        return "near_stop_rule"
    return "clear"


def stage_precheck(env: Env) -> dict:
    """TRAIN ONLY (configs/p4_precheck.yaml): explanation shares of each learned finalist on the CV
    folds' eval days, fitted exactly as `validate` fits it; rules_weighted weights per fold."""
    t0 = time.time()
    pc = load_yaml(env.a.precheck_config)
    if (
        pc["stop"]
        != env.cfg["validate"]["stop_rules"]["top_feature_share_of_abs_contribution_above"]
    ):
        raise SystemExit("precheck stop level differs from the declared validate stop rule")
    cv = json.loads(env.a.cv_results.read_text(encoding="utf-8"))
    if {k: cv["inputs"][k] for k in env.inputs()} != env.inputs():
        raise SystemExit("configs changed after the CV run: the finalists no longer match them")
    fs = folds(env, env.period("TRAIN"))
    if [f["name"] for f in fs] != pc["folds"]:
        raise SystemExit("precheck folds differ from the declared ones")
    out_f = {}
    for spec in cv["finalists"]:
        if spec["family"] not in ("logistic", "lightgbm"):
            continue
        cols = env.sets[spec["feature_set"]]
        per = []
        for f in fs:
            m = build(env, spec, n_estimators=spec.get("n_estimators"))
            m.fit(f["fit"].select(cols).to_numpy(), f["fit"]["y"].to_numpy())
            ex = explain(m, f["ev"].select(cols).to_numpy(), cols, pc["top_k"])
            per.append({"fold": f["name"], "top_share": ex["top_share"], "top": ex["top"]})
        out_f[spec["id"]] = {
            "role": spec["role"],
            "family": spec["family"],
            "feature_set": spec["feature_set"],
            "per_fold": per,
            "max_top_share": max(p["top_share"] for p in per),
        }
    cand = [v["max_top_share"] for v in out_f.values() if v["role"] == "candidate"]
    worst = max(cand) if cand else 0.0
    reading = precheck_reading(worst, pc["near_stop"], pc["stop"])
    rw = {}
    rw_spec = {
        "id": "rules_weighted",
        "family": "rules_weighted",
        "feature_set": "all",
        "params": dict(env.cfg["cv"]["families"]["rules_weighted"]),
    }
    cols = env.sets["all"]
    for f in fs:
        m = build(env, rw_spec)
        X, y = f["fit"].select(cols).to_numpy(), f["fit"]["y"].to_numpy()
        m.fit(X, y)
        rows = {}
        for r, w in m.weights.items():
            on = X[:, cols.index(r)] == 1
            rows[r] = {
                "weight": w,
                "fired": int(on.sum()),
                "precision": float(y[on].mean()) if on.any() else None,
            }
        rw[f["name"]] = {"fit_prevalence": float(y.mean()), "rules": rows}
    return {
        "inputs": {
            **env.inputs(),
            "precheck_config_sha256": sha_text(env.a.precheck_config),
            "cv_results_sha256": sha_text(env.a.cv_results),
            "periods_read": ["TRAIN"],
        },
        "learned_finalists": out_f,
        "worst_candidate_top_share": worst,
        "reading": reading,
        "rules_weighted": rw,
        "finished_utc": now(),
        "seconds": round(time.time() - t0),
    }


def precheck_summary(doc: dict) -> str:
    L = [
        f"reading: {doc['reading']}  (worst candidate top share "
        f"{f3(doc['worst_candidate_top_share'])}; near 0.40, stop 0.50)"
    ]
    for k, v in doc["learned_finalists"].items():
        for p in v["per_fold"]:
            tops = ", ".join(f"{t['feature']} {t['share']:.0%}" for t in p["top"][:4])
            L.append(f"  {k:<36} [{v['role']}] {p['fold']}: top share {f3(p['top_share'])}  {tops}")
    for fold, v in doc["rules_weighted"].items():
        L.append(
            f"rules_weighted {fold} (fit prevalence {f3(v['fit_prevalence'])}): "
            + ", ".join(
                f"{r} w={f3(x['weight'])} n={x['fired']} prec={f3(x['precision'])}"
                for r, x in v["rules"].items()
            )
        )
    return "\n".join(L)


def stop_reason(
    pr_auc: float | None, learned: bool, ex: dict, cols: list[str], registry: list[str], sr: dict
) -> str | None:
    """configs/p4_models.yaml validate.stop_rules (except the identity guard), in order."""
    if pr_auc is not None and pr_auc > sr["validation_pr_auc_above"]:
        return f"VALIDATION PR-AUC {pr_auc:.3f} is implausibly high"
    if learned and ex["top_share"] > sr["top_feature_share_of_abs_contribution_above"]:
        top = ex["top"][0]["feature"]
        return f"one feature ({top}) holds {ex['top_share']:.0%} of the explanation"
    if not set(cols) <= set(registry):
        return "a model input is not a registry feature"
    return None


def stage_validate(env: Env) -> tuple[dict, int]:
    t0 = time.time()
    if env.a.results.exists():
        raise SystemExit(
            f"{env.a.results} exists: VALIDATION is opened once (configs/p4_models.yaml)"
        )
    cv = json.loads(env.a.cv_results.read_text(encoding="utf-8"))
    if {k: cv["inputs"][k] for k in env.inputs()} != env.inputs():
        raise SystemExit("configs changed after the CV run: the finalists no longer match them")
    vcfg = env.cfg["validate"]
    m_cfg = vcfg["metrics"]
    if 0.1 not in m_cfg["review_capacity_share"] or 0.8 not in m_cfg["precision_at_recall"]:
        raise SystemExit("headline metrics (10% capacity, precision at recall 0.8) not declared")
    tr, va = env.period("TRAIN"), env.period("VALIDATION", strata=True)
    yv = va["y"].to_numpy()
    tr_keys = set(tr["account_key"].to_list())
    seen = np.array([k in tr_keys for k in va["account_key"].to_list()])
    results, scores, models = {}, {}, {}
    for f in cv["finalists"]:
        cols = env.sets[f["feature_set"]]
        m = build(env, f, n_estimators=f.get("n_estimators"))
        m.fit(tr.select(cols).to_numpy(), tr["y"].to_numpy())
        s = m.margin(va.select(cols).to_numpy())
        probability = (
            f["family"] in ("logistic", "lightgbm")
            and f["params"].get("class_weight", "none") == "none"
        )
        r = {
            "id": f["id"],
            "family": f["family"],
            "role": f["role"],
            "feature_set": f["feature_set"],
            "params": f["params"],
            "n_estimators": f.get("n_estimators"),
            "raw_score_is_probability": probability,
            "train_pr_auc": mt.pr_auc(tr["y"].to_numpy(), m.margin(tr.select(cols).to_numpy())),
        }
        score_for_brier = m.proba(va.select(cols).to_numpy()) if probability else s
        r["all"] = evaluate_subset(yv, s, False, vcfg, with_ci=True)
        if probability:
            r["all"]["brier"] = mt.brier(yv, score_for_brier)
            r["all"]["ci"]["brier"] = mt.bootstrap(
                lambda i, p=score_for_brier: mt.brier(yv[i], p[i]),
                len(yv),
                vcfg["bootstrap"]["n_boot"],
                vcfg["bootstrap"]["seed"],
                vcfg["bootstrap"]["ci"],
            )[0]
        for name, mask in (("seen", seen), ("unseen", ~seen)):
            r[name] = evaluate_subset(yv[mask], s[mask], False, vcfg, with_ci=False)
            b = vcfg["bootstrap"]
            ys, ss = yv[mask], s[mask]
            r[name]["ci"] = {
                "pr_auc": mt.bootstrap(
                    lambda i, ys=ys, ss=ss: mt.pr_auc(ys[i], ss[i]),
                    len(ys),
                    b["n_boot"],
                    b["seed"],
                    b["ci"],
                )[0]
            }
        r["overfit_gap_pr_auc"] = (
            r["train_pr_auc"] - r["all"]["pr_auc"] if r["all"]["pr_auc"] is not None else None
        )
        results[f["id"]] = r
        scores[f["id"]] = s
        models[f["id"]] = m
        print(
            f"validate {f['id']:<34} PR-AUC {f3(r['all']['pr_auc'])} CI {r['all']['ci']['pr_auc']}",
            flush=True,
        )
    by_group = {family_group(r["family"]): r for r in results.values() if r["role"] == "candidate"}
    sel = select(by_group, yv, scores, vcfg["bootstrap"])
    chosen = results[sel["chosen"]]
    spec = next(f for f in cv["finalists"] if f["id"] == chosen["id"])
    model = models[chosen["id"]]
    cols = env.sets[spec["feature_set"]]
    Xv = va.select(cols).to_numpy()
    doc = {
        "inputs": {
            **env.inputs(),
            "cv_results_sha256": sha_text(env.a.cv_results),
            "periods_read": ["TRAIN", "VALIDATION"],
        },
        "finalists": results,
        "selection": sel,
        "chosen": chosen["id"],
    }
    # ---- stop rules (configs/p4_models.yaml validate.stop_rules), in order
    sr = vcfg["stop_rules"]
    learned = spec["family"] in ("logistic", "lightgbm")
    ex = explain(model, Xv, cols, vcfg["explain"]["top_k"])
    doc["explain"] = ex
    stop = stop_reason(chosen["all"]["pr_auc"], learned, ex, cols, reg.model_features(env.ctx), sr)
    if stop is None and learned:
        gv = guard(env, spec, spec.get("n_estimators"), pl.concat([tr, va.drop("strata")]))
        doc["identity_guard"] = gv
        if not gv["cleared"]:
            stop = f"identity guard v2: {gv['verdict']}"
    elif stop is None:
        doc["identity_guard"] = {"verdict": "not_applicable", "reason": "rule features only"}
    if stop:
        doc |= {
            "stopped": stop,
            "model_sha256": None,
            "finished_utc": now(),
            "seconds": round(time.time() - t0),
        }
        return doc, 2
    # ---- operating point, typologies, importance
    cr = [vcfg["cost_ratio_fn_fp"], *vcfg["cost_ratio_sensitivity"]]
    ops = {str(r): mt.cost_threshold(yv, scores[chosen["id"]], r) for r in cr}
    doc["operating_points"] = ops
    s_ch = scores[chosen["id"]]
    thr = ops[str(vcfg["cost_ratio_fn_fp"])]["threshold"]
    k = int(np.ceil(0.10 * len(s_ch)))
    top = np.zeros(len(s_ch), bool)
    top[np.argsort(-s_ch, kind="stable")[:k]] = True
    strata = va["strata"].to_list()
    typ = sorted({t for x in strata for t in x if t.startswith("typ_")})
    doc["per_typology_recall"] = {
        t: {
            "n_pos": int(sum(t in x for x in strata)),
            "recall_at_cost_threshold": float(
                np.mean([s_ch[i] >= thr for i, x in enumerate(strata) if t in x])
            ),
            "recall_at_capacity_0.1": float(
                np.mean([top[i] for i, x in enumerate(strata) if t in x])
            ),
        }
        for t in typ
    }
    if learned:
        pi = vcfg["permutation_importance"]
        doc["permutation_importance_validation"] = permutation_importance(
            model, Xv, yv, cols, pi["repeats"], pi["seed"]
        )[:15]
    # ---- save BY HASH
    meta = {
        "model_id": f"p4-{spec['family']}",
        "family": spec["family"],
        "feature_set": spec["feature_set"],
        "features": cols,
        "feature_registry_sha256": reg.registry_hash(env.ctx),
        "trained_on": "TRAIN 2022-09-02..2022-09-05 (HI-Medium); never refitted with VALIDATION",
        "explain_output_space": "log-odds margin of the uncalibrated model",
        "raw_score_is_probability": chosen["raw_score_is_probability"],
        "recalibration_required_in_p5": True,
        "class_weight": spec["params"].get("class_weight", "none"),
        "cv_results_sha256": doc["inputs"]["cv_results_sha256"],
        "protocol_sha256": doc["inputs"]["protocol_sha256"],
    }
    h = store.save(model, meta, env.a.model_dir, env.cfg["save"]["max_bytes"])
    reloaded, _ = store.load(h, env.a.model_dir)
    if not np.allclose(reloaded.margin(Xv), s_ch, atol=1e-9, rtol=0):
        raise SystemExit("the saved model does not reproduce its VALIDATION scores")
    doc |= {
        "model_sha256": h,
        "model_meta": meta,
        "stopped": None,
        "finished_utc": now(),
        "seconds": round(time.time() - t0),
    }
    return doc, 0


# ---------------------------------------------------------------- reporting


def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def cv_summary(doc: dict) -> str:
    L = [f"configurations run: {doc['inputs']['n_configs']}"]
    for r in sorted(doc["configs"] + doc["sensitivity"], key=lambda r: -(r["cv_pr_auc"] or 0)):
        L.append(
            f"  {r['id']:<36} PR-AUC {f3(r['cv_pr_auc'])}  seen {f3(r['cv_pr_auc_seen'])}  "
            f"unseen {f3(r['cv_pr_auc_unseen'])}  train {f3(r['cv_train_pr_auc'])}  "
            f"fold spread {f3(r['fold_spread'])}"
        )
    L.append("finalists: " + ", ".join(f"{f['id']} ({f['role']})" for f in doc["finalists"]))
    for g, c in doc["learning_curves"].items():
        L.append(
            f"learning curve {g}: "
            + "  ".join(f"{x['fraction']}: {f3(x['pr_auc'])}/{f3(x['train_pr_auc'])}" for x in c)
        )
    for g, v in doc["permutation_importance"].items():
        L.append(
            f"permutation importance {g} (sign flips {v['n_sign_flip']}): "
            + ", ".join(r["feature"] for r in v["top"][:5])
        )
    return "\n".join(L)


def val_summary(doc: dict) -> str:
    L = []
    for r in doc["finalists"].values():
        a = r["all"]
        L.append(
            f"{r['id']:<36} [{r['role']}] PR-AUC {f3(a['pr_auc'])} CI {a['ci']['pr_auc']}  "
            f"seen {f3(r['seen']['pr_auc'])} unseen {f3(r['unseen']['pr_auc'])}  "
            f"recall@10% {f3(a['recall_at_capacity_0.1'])}  train {f3(r['train_pr_auc'])}"
        )
    for t in doc["selection"]["tests"]:
        L.append(
            f"selection: {t['candidate']} vs {t['vs']} gain {f3(t['gain'])} CI {t['gain_ci']} "
            f"passes={t['passes']}"
        )
    L.append(f"chosen: {doc['chosen']}")
    g = doc.get("identity_guard", {})
    L.append(f"identity guard v2: {g.get('verdict')}")
    L.append(
        f"explanation top feature share {f3(doc['explain']['top_share'])}: "
        + ", ".join(t["feature"] for t in doc["explain"]["top"][:5])
    )
    if doc.get("stopped"):
        L.append(f"STOPPED: {doc['stopped']} -> no model saved; tell Claude")
        return "\n".join(L)
    op = doc["operating_points"]
    for k, v in op.items():
        L.append(
            f"cost {k}:1 -> flag {v['n_flagged']}  recall {f3(v['recall'])}  "
            f"precision {f3(v['precision'])}"
        )
    L.append(f"model saved: {doc['model_sha256']}")
    return "\n".join(L)


def model_card(doc: dict, cv: dict) -> str:
    ch = doc["finalists"][doc["chosen"]]
    a, op = ch["all"], doc["operating_points"]
    rows = "\n".join(
        f"| {r['id']} | {r['role']} | {f3(r['all']['pr_auc'])} {r['all']['ci']['pr_auc']} | "
        f"{f3(r['seen']['pr_auc'])} | {f3(r['unseen']['pr_auc'])} | {f3(r['all']['roc_auc'])} |"
        for r in doc["finalists"].values()
    )
    ops = "\n".join(
        f"| {k}:1 | {v['n_flagged']} | {f3(v['recall'])} | {f3(v['precision'])} |"
        for k, v in op.items()
    )
    top = "\n".join(
        f"| {t['feature']} | {f3(t['mean_abs'])} | {f3(t['mean_signed'])} |"
        for t in doc["explain"]["top"]
    )
    meta = doc["model_meta"]
    n_feat, reg16 = len(meta["features"]), meta["feature_registry_sha256"][:16]
    is_prob = ch["raw_score_is_probability"]
    return f"""# Model card — {meta["model_id"]}

Generated by `scripts/p4/models.py validate` (do not edit by hand).

- **Model hash (load by this only):** `{doc["model_sha256"]}`
- **Family / features:** {ch["family"]}, feature set `{ch["feature_set"]}` \
({n_feat} features, registry `{reg16}…`)
- **Data:** IBM AML HI-Medium (synthetic). Fitted on TRAIN (2022-09-02 … 09-05) only; tuned by
  time-series CV inside TRAIN ({cv["inputs"]["n_configs"]} configurations); compared once on
  VALIDATION (2022-09-07). CALIBRATION and TEST not read.
- **Output:** raw score = log-odds margin of the uncalibrated model; raw score is a probability:
  {is_prob}. **P5 recalibrates on CALIBRATION before any probability is used.**
- **Explanations:** contributions in log-odds (margin) units, never probability units.

## VALIDATION (one day; 95% account-bootstrap CIs)

| model | role | PR-AUC [CI] | seen | unseen (regime-B analogue) | ROC-AUC (secondary) |
|---|---|---|---|---|---|
{rows}

Chosen: **{doc["chosen"]}** — precision at recall 0.8 {f3(a["precision_at_recall_0.8"])}, \
recall at 10% review capacity {f3(a["recall_at_capacity_0.1"])}.

## Operating point (evaluation device; thresholds belong to compliance)

| FN:FP cost | alerts flagged | recall | precision |
|---|---|---|---|
{ops}

## Top explanation features (VALIDATION, mean |contribution|, log-odds)

| feature | mean abs | mean signed |
|---|---|---|
{top}

## Limitations

- Synthetic data from one generator; nothing here is real-world validation.
- New accounts (regime B, 85% of TEST positives) are ranked close to chance; step 0 and 0b
  (docs/p4/P4_MODEL.md) show this is a property of the data at a 24 h look-back, not a tuning gap.
- Label P2-v1 counts coincidental positives (58.7% on VALIDATION), so rule-like features look
  better than their causal contribution.
- VALIDATION is a single weekday; CIs ignore that one laundering attempt spans several accounts.
- The chosen model excludes payment format (decision P4-9): ACH is a generator signature here. The
  with-format sensitivity rows show what it would add on this synthetic data.
"""


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cv", "precheck", "validate"])
    ap.add_argument("--precheck-config", type=Path, default=PRECHECK_CONFIG)
    ap.add_argument("--precheck-results", type=Path, default=PRECHECK_RESULTS)
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    ap.add_argument("--diag-config", type=Path, default=DIAG_CONFIG)
    ap.add_argument("--features-root", type=Path, default=FEATURES_ROOT)
    ap.add_argument("--store-root", type=Path, default=STORE_ROOT)
    ap.add_argument("--p3-results", type=Path, default=P3_RESULTS)
    ap.add_argument("--cv-results", type=Path, default=CV_RESULTS)
    ap.add_argument("--results", type=Path, default=VAL_RESULTS)
    ap.add_argument("--model-card", type=Path, default=MODEL_CARD)
    ap.add_argument("--model-dir", type=Path, default=store.MODEL_DIR)
    ap.add_argument("--rules", type=Path, default=rules.RULES_CONFIG)
    ap.add_argument("--thresholds", type=Path, default=rules.THRESHOLDS_FILE)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse(argv)
    env = Env(a)
    if a.stage == "cv":
        doc = stage_cv(env)
        a.cv_results.parent.mkdir(parents=True, exist_ok=True)
        a.cv_results.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(cv_summary(doc))
        print(f"\nwrote {a.cv_results} ({doc['seconds']} s). Commit it BEFORE `validate`.")
        return 0
    if a.stage == "precheck":
        doc = stage_precheck(env)
        a.precheck_results.parent.mkdir(parents=True, exist_ok=True)
        a.precheck_results.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(precheck_summary(doc))
        print(f"\nwrote {a.precheck_results} ({doc['seconds']} s). VALIDATION was not read.")
        return 0
    doc, code = stage_validate(env)
    a.results.parent.mkdir(parents=True, exist_ok=True)
    a.results.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    if code == 0:
        cv = json.loads(a.cv_results.read_text(encoding="utf-8"))
        a.model_card.write_text(model_card(doc, cv), encoding="utf-8")
    print(val_summary(doc))
    print(f"\nwrote {a.results} ({doc['seconds']} s)")
    return code


if __name__ == "__main__":
    sys.exit(main())
