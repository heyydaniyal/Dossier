"""Phase 3 driver (Dani's laptop; real data is local). Aggregates only -> docs/p3/p3_results.json.

Stages (run in this order; `all` = features (pre-TEST) -> check -> permutation -> guard):
  features [--include-test]  feature matrices per period -> data/p3/<variant>/features/<P>.parquet
                             + integrity vs the P2 alert store + split boundary check.
                             TEST is built ONLY with --include-test (Dani's rule: ask before
                             anything reads TEST). It reads TEST transactions and alerts, never
                             labels, and writes only row counts and hashes for TEST.
  check                      real-data point-in-time checks on a keyed-hash sample of pre-TEST
                             alerts: deletion-recompute, outside-window mutation, pure == batch,
                             stored matrix == recomputation.
  permutation                shuffled-label test (configs/p3_features.yaml), TRAIN only.
  guard                      identity guard v2 per candidate feature group (TRAIN + VALIDATION).

Labels: only through src.eval.training_labels (refuses TEST). No TEST label is read in P3.

    uv run python -m scripts.p3.run_p3 features
    uv run python -m scripts.p3.run_p3 check
    uv run python -m scripts.p3.run_p3 permutation
    uv run python -m scripts.p3.run_p3 guard
    uv run python -m scripts.p3.run_p3 features --include-test     (only with Dani's OK)
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
from sklearn.ensemble import HistGradientBoostingClassifier

from src.alerts import rules
from src.data import fx as fxmod
from src.data.hashing import keyed_u64
from src.data.periods import ROOT, load_split
from src.data.stores import scan_runtime
from src.data.transactions import interim_path, scan_transactions, verify_interim
from src.eval.identity_guard import evaluate_v2, load_v2_config
from src.eval.training_labels import load_fit_labels
from src.features import compute as fc
from src.features import registry as reg
from src.features import splits
from src.features.pit import close, mismatches, scramble_outside_window
from src.models.permutation import Fold
from src.models.permutation import run as run_permutation

VARIANT = "HI-Medium"
INTERIM = ROOT / "data" / "interim"
STORE_ROOT = ROOT / "data" / "p2"
OUT_ROOT = ROOT / "data" / "p3"
RESULTS = ROOT / "docs" / "p3" / "p3_results.json"
PRE_TEST = ("TRAIN", "VALIDATION", "CALIBRATION")
CONFIGS = {
    "p3_features.yaml": fc.FEATURES_CONFIG,
    "p3_identity_guard.yaml": ROOT / "configs" / "p3_identity_guard.yaml",
    "p2_split.yaml": ROOT / "configs" / "p2_split.yaml",
    "p2_rules.yaml": rules.RULES_CONFIG,
    "p2_rule_thresholds.yaml": rules.THRESHOLDS_FILE,
    "p2_fx_usd_per_unit.yaml": fc.FX_FILE,
    "data_sources.yaml": ROOT / "configs" / "data_sources.yaml",
}
P2_RESULTS = ROOT / "docs" / "p2" / "p2_results.json"


def regime_b_p2(path: Path) -> int:
    """TEST alerts of accounts with no TRAIN alert, as recorded by P2 (10,265 on HI-Medium)."""
    d = json.loads(path.read_text(encoding="utf-8"))
    return int(d["build"]["feasibility"]["counts"]["TEST_regime_B_unseen_accounts"]["n_alerts"])


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_results(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")


class Env:
    def __init__(self, a: argparse.Namespace):
        self.variant = a.variant
        self.split = load_split()
        self.cfg = fc.load_features_config()
        self.configs = CONFIGS | {
            "p2_rules.yaml": a.rules,
            "p2_rule_thresholds.yaml": a.thresholds,
            "p2_fx_usd_per_unit.yaml": a.fx,
            "data_sources.yaml": a.sources,
        }
        self.ctx = fc.make_context(
            rules.load_rules(a.rules), rules.load_thresholds(a.thresholds), self.cfg, self.split
        )
        verify_interim(a.interim, a.variant, a.sources)
        self.raw = scan_transactions(interim_path(a.interim, a.variant, "trans"))
        self.fx = fxmod.load(a.fx)
        self.trans_usd = fxmod.with_usd(self.raw, self.fx)
        self.store_root = a.store_root
        self.feat_dir = a.out_root / a.variant / "features"
        self.results = a.results
        self.p2_results = a.p2_results

    def alerts(self) -> pl.DataFrame:
        return scan_runtime("alerts.parquet", self.variant, self.store_root).collect()

    def matrix(self, period: str) -> pl.DataFrame:
        p = self.feat_dir / f"{period}.parquet"
        if not p.exists():
            raise SystemExit(f"{p} missing: run the features stage first")
        return pl.read_parquet(p)

    def inputs(self) -> dict:
        return {
            "configs_sha256": {k: sha256(v) for k, v in self.configs.items()},
            "feature_registry_sha256": reg.registry_hash(self.ctx),
            "n_model_features": len(reg.model_features(self.ctx)),
        }


# ------------------------------------------------------------------ features


def _integrity(df: pl.DataFrame, store: pl.DataFrame) -> dict:
    j = df.join(store, on="alert_id", suffix="_p2")
    bad_stats = 0
    for c in rules.ALL_STATS:
        bad_stats += len(
            [1 for a, b in zip(j[c].to_list(), j[f"{c}_p2"].to_list(), strict=True)
             if abs(a - float(b)) > 1e-9 * max(1.0, abs(a))]
        )  # fmt: skip
    bad_rules = int((j["_triggered"] != j["triggered_rules"]).sum())
    return {"rows_joined": j.height, "rule_stat_mismatches": bad_stats,
            "triggered_rules_mismatches": bad_rules}  # fmt: skip


def stage_features(env: Env, include_test: bool) -> dict:
    periods = [*PRE_TEST, *(["TEST"] if include_test else [])]
    alerts = env.alerts()
    env.feat_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {"periods": {}}
    frames = []
    for p in periods:
        t0 = time.time()
        a = alerts.filter(pl.col("period") == p)
        df = fc.batch_features(env.trans_usd, a, env.ctx)
        integ = _integrity(df, a)
        if (
            integ["rows_joined"] != a.height
            or integ["rule_stat_mismatches"]
            or integ["triggered_rules_mismatches"]
        ):
            raise SystemExit(
                f"{p}: recomputed rule features disagree with the P2 alert store: {integ}"
            )
        df = splits.assign_periods(
            df.join(a.select("alert_id", "period"), on="alert_id"), env.split
        )
        path = env.feat_dir / f"{p}.parquet"
        df.drop("_triggered").write_parquet(path)
        rec = {
            "rows": df.height,
            "parquet_sha256": file_sha(path),
            "seconds": round(time.time() - t0),
        }
        if p != "TEST":  # TEST: counts and hashes only
            names = fc.feature_names(env.ctx)
            rec["integrity"] = integ
            rec["nan_share"] = {c: round(float(df[c].is_nan().mean()), 6) for c in names
                                if df[c].is_nan().any()}  # fmt: skip
        else:
            rec["integrity_ok"] = True
        out["periods"][p] = rec
        frames.append(df.select("alert_id", "split_period", "as_of"))
        print(f"{p}: {df.height} rows, {rec['seconds']} s", flush=True)
    out["boundaries"] = splits.check_boundaries(pl.concat(frames), env.split)
    if include_test:
        n_unseen = splits.unseen_test_slice(alerts).height
        out["regime_b_unseen_test_alerts"] = n_unseen
        want = regime_b_p2(env.p2_results)
        if n_unseen != want:
            raise SystemExit(f"unseen-account TEST slice {n_unseen} != P2 record {want}")
        print(
            "\nAppend to docs/holdout_exposure_log.md BEFORE committing anything from this run:\n"
            f"| 5 | {now()} | P3 | TEST | `run_p3 features --include-test`: TEST transactions and "
            "TEST alert ids/rule stats (no labels); row counts and file hashes only | TEST feature "
            "matrix (runtime input for agents; P5 task 8) | feature registry "
            f"{reg.registry_hash(env.ctx)[:16]} |\n"
        )
    return out


# ------------------------------------------------------------------ point-in-time check


def _sample(alerts: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    h = keyed_u64(alerts["alert_id"].to_list(), "p3_pit_sample", seed)
    return alerts.with_columns(pl.Series("_h", h)).sort("_h").head(n).drop("_h")


def stage_check(env: Env) -> dict:
    pc = env.cfg["pit_check"]
    tol = {"abs_tol": pc["abs_tol"], "rel_tol": pc["rel_tol"]}
    names = fc.feature_names(env.ctx)
    alerts = env.alerts().filter(pl.col("period").is_in(PRE_TEST))
    sample = _sample(alerts, pc["n_alerts_batch"], pc["sample_seed"])
    stored = pl.concat([env.matrix(p) for p in PRE_TEST])
    res = {"n_sampled": sample.height, "n_windows": 0, "deletion": 0, "mutation": 0,
           "stored_vs_recomputed": 0, "pure_vs_batch": 0, "examples": []}  # fmt: skip
    full_parts = []
    for as_of in sorted(sample["as_of"].unique().to_list()):
        a = sample.filter(pl.col("as_of") == as_of)
        full = fc.features_for_window(env.trans_usd, as_of, a, env.ctx)
        past = fxmod.with_usd(env.raw.filter(pl.col("timestamp") < as_of), env.fx)
        deleted = fc.features_for_window(past, as_of, a, env.ctx)
        mut = fxmod.with_usd(scramble_outside_window(env.raw, as_of, env.ctx.lookback), env.fx)
        mutated = fc.features_for_window(mut, as_of, a, env.ctx)
        st = stored.filter(pl.col("alert_id").is_in(a["alert_id"].to_list()))
        for key, other in (
            ("deletion", deleted),
            ("mutation", mutated),
            ("stored_vs_recomputed", st),
        ):
            m = mismatches(full, other, names, **tol)
            res[key] += len(m)
            res["examples"] += [f"{key}: {x}" for x in m[:3]]
        full_parts.append(full)
        res["n_windows"] += 1
        print(f"check {as_of.date()}: {a.height} alerts", flush=True)
    full = pl.concat(full_parts)
    for row in sample.head(pc["n_alerts_pure"]).iter_rows(named=True):
        pure = fc.account_features(env.trans_usd, row["account_key"], row["as_of"],
                                   row["peer_group"], env.ctx)  # fmt: skip
        b = full.filter(pl.col("alert_id") == row["alert_id"]).row(0, named=True)
        bad = [c for c in names if not close(pure[c], b[c], **tol)]
        res["pure_vs_batch"] += len(bad)
        res["examples"] += [f"pure: {row['alert_id']}:{c}" for c in bad[:3]]
    res["n_pure"] = min(pc["n_alerts_pure"], sample.height)
    res["passed"] = all(res[k] == 0 for k in ("deletion", "mutation", "stored_vs_recomputed",
                                               "pure_vs_batch"))  # fmt: skip
    return res


# ------------------------------------------------------------------ labelled stages


def _with_labels(env: Env, periods: list[str]) -> pl.DataFrame:
    x = pl.concat([env.matrix(p) for p in periods])
    y = load_fit_labels(periods, env.variant, env.store_root)
    j = x.join(y.select("alert_id", "y"), on="alert_id")
    if j.height != x.height:
        raise SystemExit("feature rows and label rows do not match")
    return j.with_columns(pl.col("window_start").dt.date().alias("day"))


def _model(env: Env):
    return HistGradientBoostingClassifier(**env.cfg["permutation_test"]["model_params"])


def _fold_specs(env: Env, names: list[str]) -> list[dict]:
    spec = {f["name"]: f for f in load_v2_config()["folds"]}
    return [spec[n] for n in names]


def stage_permutation(env: Env) -> dict:
    pt = env.cfg["permutation_test"]
    df = _with_labels(env, ["TRAIN"])
    cols = reg.model_features(env.ctx)
    folds, xs = [], {}
    for s in _fold_specs(env, pt["folds"]):
        fit_days = [date.fromisoformat(d) for d in s["fit_days"]]
        fit = df.filter(pl.col("day").is_in(fit_days))
        ev = df.filter(pl.col("day") == date.fromisoformat(s["eval_day"]))
        blocks = np.array([fit_days.index(d) for d in fit["day"].to_list()])
        folds.append(Fold(s["name"], fit["y"].to_numpy(), blocks, ev["y"].to_numpy()))
        xs[s["name"]] = (fit.select(cols).to_numpy(), ev.select(cols).to_numpy())
    t0 = time.time()
    r = run_permutation(
        folds,
        lambda f, y: xs[f.name],  # the real pipeline: features never depend on labels
        lambda: _model(env),
        n_permutations=pt["n_permutations"],
        seed=pt["seed"],
        null_mean_over_prevalence_max=pt["pass_if"]["null_mean_over_prevalence_max"],
        real_above_null_max=pt["pass_if"]["real_above_null_max"],
    )
    r["seconds"] = round(time.time() - t0)
    r["features"] = len(cols)
    return r


def stage_guard(env: Env) -> dict:
    g = env.cfg["groups"]
    df = _with_labels(env, ["TRAIN", "VALIDATION"])
    groups = fc.group_names(env.ctx)
    base_cols = groups[g["base"]]
    specs = _fold_specs(env, [f["name"] for f in load_v2_config()["folds"]])
    out: dict = {"base": g["base"], "groups": {}}
    for cand in g["candidates"]:
        with_cols = base_cols + groups[cand]
        folds = []
        for s in specs:
            fit = df.filter(pl.col("day").is_in([date.fromisoformat(d) for d in s["fit_days"]]))
            ev = df.filter(pl.col("day") == date.fromisoformat(s["eval_day"]))
            seen_keys = set(fit["account_key"].to_list())
            preds = {}
            for k, cols in (("pred_base", base_cols), ("pred_with_group", with_cols)):
                m = _model(env).fit(fit.select(cols).to_numpy(), fit["y"].to_numpy())
                preds[k] = m.predict_proba(ev.select(cols).to_numpy())[:, 1]
            folds.append({
                "name": s["name"], "window_start": ev["window_start"].to_list(),
                "y": ev["y"].to_numpy(), **preds,
                "seen": np.array([k in seen_keys for k in ev["account_key"].to_list()]),
                "account_key": ev["account_key"].to_list(),
            })  # fmt: skip
        r = evaluate_v2(folds, split=env.split)
        out["groups"][cand] = r
        print(f"guard {cand}: {r['verdict']} (cleared={r['cleared']})", flush=True)
    out["summary"] = {k: {"verdict": v["verdict"], "cleared": v["cleared"]}
                      for k, v in out["groups"].items()}  # fmt: skip
    out["cleared_groups"] = [g["base"], *[k for k, v in out["summary"].items() if v["cleared"]]]
    return out


# ------------------------------------------------------------------ main


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["features", "check", "permutation", "guard", "all"])
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--interim", type=Path, default=INTERIM)
    ap.add_argument("--store-root", type=Path, default=STORE_ROOT)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--results", type=Path, default=RESULTS)
    # real configs by default; overridden only by the fixture smoke test
    ap.add_argument("--rules", type=Path, default=rules.RULES_CONFIG)
    ap.add_argument("--thresholds", type=Path, default=rules.THRESHOLDS_FILE)
    ap.add_argument("--fx", type=Path, default=fc.FX_FILE)
    ap.add_argument("--sources", type=Path, default=CONFIGS["data_sources.yaml"])
    ap.add_argument("--p2-results", type=Path, default=P2_RESULTS)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None, env: Env | None = None) -> int:
    a = parse(argv)
    if a.stage == "all" and a.include_test:
        raise SystemExit("--include-test only with the features stage, on its own")
    env = env or Env(a)
    doc = load_results(a.results)
    doc["inputs"] = env.inputs()
    stages = ["features", "check", "permutation", "guard"] if a.stage == "all" else [a.stage]
    for s in stages:
        t0 = time.time()
        if s == "features":
            r = stage_features(env, a.include_test)
        elif s == "check":
            r = stage_check(env)
        elif s == "permutation":
            r = stage_permutation(env)
        else:
            r = stage_guard(env)
        r["finished_utc"] = now()
        r["seconds_total"] = round(time.time() - t0)
        doc[s] = r
        save_results(doc, a.results)
        verdict = r.get("passed", r.get("cleared_groups", "done"))
        print(f"{s}: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
