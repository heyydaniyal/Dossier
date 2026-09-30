"""Phase 2 driver: boundaries -> stats -> calibration (TRAIN) -> alerts -> labels -> feasibility
-> AGENT-DEV/TEST split -> KYC v2 (+ planting) -> leakage tests -> dispositions -> results JSON.

Usage (repo root, on the machine that holds data/interim from P1):
    uv run python -m scripts.p2.run_p2 calibrate     # stats, FX, thresholds on TRAIN (no TEST)
    uv run python -m scripts.p2.run_p2 kyc           # KYC v2 + leakage tests only (no TEST)
    uv run python -m scripts.p2.run_p2 build         # everything; READS TEST (counts, split)
    uv run python -m scripts.p2.run_p2 all --verify  # regenerate everything elsewhere (READS TEST)
Rule (Dani, 2026-09-30): ask before running any stage that reads TEST.

Order is enforced: `build` refuses to run without configs/p2_rule_thresholds.yaml and records its
sha256 BEFORE any TEST-period count is computed (the holdout's single permitted exposure).
The results JSON holds aggregates only. TEST appears only as counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import yaml

from scripts.p2 import calibrate as cal
from scripts.p2 import dispositions as disp
from scripts.p2 import kyc_leakage
from src.alerts import build, rules
from src.data import fx as fxmod
from src.data import kyc as kycmod
from src.data.accounts import DOMESTIC, account_attributes
from src.data.periods import PRE_TEST_PERIODS, ROOT, SPLIT_CONFIG, Split, load_split
from src.data.transactions import DATA_SOURCES, interim_path, scan_transactions, verify_interim
from src.eval import alert_labels as al
from src.eval import group_split as gs

P2_VERSION = "1.0.0"
PRIMARY = "HI-Medium"
DEFAULT_RESULTS = ROOT / "docs" / "p2" / "p2_results.json"
FX_FILE = ROOT / "configs" / "p2_fx_usd_per_unit.yaml"
DISP_CONFIG = ROOT / "configs" / "p2_dispositions.yaml"


# ---------------------------------------------------------------- helpers


def sha256(p: Path) -> str:
    """sha256 of a file; text files (.yaml/.json/.py/.md) are hashed with LF line endings so the
    hash is the same on Windows (CRLF checkout) and Linux."""
    b = Path(p).read_bytes()
    if Path(p).suffix in (".yaml", ".yml", ".json", ".py", ".md"):
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def write_text_lf(path: Path, text: str) -> None:
    """Always LF, whatever the OS (Path.write_text would write CRLF on Windows)."""
    path.write_bytes(text.encode("utf-8"))


def _sig(o):
    if isinstance(o, float):
        return None if (o != o or o in (float("inf"), float("-inf"))) else float(f"{o:.10g}")
    if isinstance(o, dict):
        return {str(k): _sig(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_sig(v) for v in o]
    return o


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd", statistics=False)


class Paths:
    def __init__(self, out_root: Path, variant: str, configs_dir: Path):
        self.root = out_root / variant
        self.work = self.root / "work"
        self.runtime = self.root / "runtime"
        self.eval = self.root / "eval"
        self.configs = configs_dir
        self.stats_dir = self.work / "account_day_stats"
        self.burn_in = self.work / "burn_in_stats.parquet"
        self.keys = self.work / "account_keys.parquet"
        self.pos = self.eval / "positive_account_days.parquet"
        self.legs = self.eval / "laundering_legs.parquet"
        self.alerts = self.runtime / "alerts.parquet"
        self.kyc = self.runtime / "kyc.parquet"
        self.disp = self.runtime / "dispositions.parquet"
        self.dev_ids = self.runtime / "agent_dev_alert_ids.parquet"
        self.labels = self.eval / "alert_labels.parquet"
        self.split = self.eval / "agent_split.parquet"
        self.planting = self.eval / "planting.parquet"
        self.fx = configs_dir / "p2_fx_usd_per_unit.yaml"
        self.thresholds = configs_dir / "p2_rule_thresholds.yaml"

    def stats_file(self, ws: datetime) -> Path:
        return self.stats_dir / f"{ws.date().isoformat()}.parquet"


def _step(msg: str) -> None:
    print(f"  {msg} ...", flush=True)


# ---------------------------------------------------------------- stage: stats


def burn_in_stats(trans_usd: pl.LazyFrame, split: Split, rcfg: dict) -> pl.DataFrame:
    """Every rule statistic for the burn-in day (2022-09-01): label-free, before every window."""
    bi = rules.day_stats(trans_usd, split.burn_in[0], split, rcfg)
    return bi.select("account_key", *rules.ALL_STATS)


def stage_stats(
    variant: str,
    interim: Path,
    P: Paths,
    split: Split,
    rcfg: dict,
    kcfg: dict,
    fit_fx: bool,
    sources: Path,
) -> dict:
    verify_interim(interim, variant, sources)
    trans = scan_transactions(interim_path(interim, variant, "trans"))
    info: dict = {}
    if fit_fx:
        _step("FX: usd_per_unit on TRAIN")
        a, b = split.periods["TRAIN"]
        fx = fxmod.derive_usd_per_unit(trans, a, b)
        fxmod.save(fx, P.fx)
        info["fx"] = {k: v for k, v in fx.items() if k != "usd_per_unit"}
    usd = fxmod.load(P.fx)
    tu = fxmod.with_usd(trans, usd)

    _step("account keys in the usable span")
    s0, s1 = split.span
    inspan = trans.filter((pl.col("timestamp") >= s0) & (pl.col("timestamp") < s1))
    pairs = pl.concat(
        [
            inspan.select(
                pl.col("from_bank").alias("b"), pl.col("from_account").alias("a")
            ).unique(),
            inspan.select(pl.col("to_bank").alias("b"), pl.col("to_account").alias("a")).unique(),
        ]
    ).unique()
    keys = (
        pairs.select(pl.concat_str([pl.col("b"), pl.lit("|"), pl.col("a")]).alias("account_key"))
        .collect()
        .sort("account_key")
    )
    attrs = account_attributes(
        interim_path(interim, variant, "accounts"),
        keys,
        kcfg["foreign_countries"],
        kcfg["crypto_prefixes"],
    ).sort("account_key")
    write_parquet(attrs, P.keys)
    peer = attrs.select("account_key", pl.col("entity_type").alias("peer_group"))

    _step("burn-in statistics (KYC activity band and v2 planting)")
    write_parquet(burn_in_stats(tu, split, rcfg), P.burn_in)

    P.stats_dir.mkdir(parents=True, exist_ok=True)
    n_active = {}
    for ws in split.window_starts():
        _step(f"account-day statistics {ws.date()}")
        st = rules.day_stats(tu, ws, split, rcfg).join(peer, on="account_key", how="left")
        if st["peer_group"].null_count():
            raise RuntimeError("active account without accounts-file attributes")
        st = st.with_columns(pl.lit(split.period_of(ws)).alias("period"))
        write_parquet(st, P.stats_file(ws))
        n_active[ws.date().isoformat()] = st.height
    info["n_active_account_days"] = n_active
    info["n_accounts_in_span"] = keys.height
    loc = (
        attrs.group_by("bank_location", "bank_country")
        .len()
        .sort(["len", "bank_location"], descending=[True, False])
    )
    info["bank_locations"] = {
        "n_distinct_locations": loc.height,
        "n_accounts_domestic": int(loc.filter(pl.col("bank_country") == DOMESTIC)["len"].sum()),
        "n_accounts_foreign": int(loc.filter(pl.col("bank_country") != DOMESTIC)["len"].sum()),
        "foreign_countries_found": sorted(
            loc.filter(pl.col("bank_country") != DOMESTIC)["bank_country"].unique().to_list()
        ),
        "top_300": [
            {"location": r["bank_location"], "country": r["bank_country"], "n_accounts": r["len"]}
            for r in loc.head(300).iter_rows(named=True)
        ],
    }
    return info


def _labelled_stats(P: Paths, split: Split, period: str, pos: pl.DataFrame) -> pl.DataFrame:
    """Rule statistics + peer group + is_pos for one period, built day by day (memory-lean)."""
    keep = ["account_key", "window_start", "peer_group", *rules.RULE_STATS]
    parts = []
    for ws in split.window_starts(period):
        day = pl.read_parquet(P.stats_file(ws), columns=keep)
        p = pos.filter(pl.col("window_start") == ws).select(
            "account_key", pl.lit(True).alias("is_pos")
        )
        parts.append(
            day.join(p, on="account_key", how="left")
            .with_columns(
                pl.col("is_pos").fill_null(False), pl.col("peer_group").cast(pl.Categorical)
            )
            .drop("account_key", "window_start")
        )
    return pl.concat(parts).with_columns(pl.col("peer_group").cast(pl.String))


# ---------------------------------------------------------------- stage: labels (eval)


def stage_positive_days(variant: str, interim: Path, P: Paths, split: Split) -> pl.DataFrame:
    _step("laundering legs and positive account-days (evaluation store)")
    legs = al.laundering_legs(
        interim_path(interim, variant, "trans"),
        interim_path(interim, variant, "patterns"),
        *split.span,
    )
    write_parquet(legs, P.legs)
    pos = al.positive_account_days(legs)
    write_parquet(pos, P.pos)
    return pos


def _n_pos(pos: pl.DataFrame, split: Split, period: str) -> int:
    a, b = split.periods[period]
    return pos.filter((pl.col("window_start") >= a) & (pl.col("window_start") < b)).height


# ---------------------------------------------------------------- stage: calibrate


def stage_calibrate(P: Paths, split: Split, rcfg: dict, pos: pl.DataFrame, cp: ConfigPaths) -> dict:
    _step("threshold calibration on TRAIN")
    tr = _labelled_stats(P, split, "TRAIN", pos)
    res = cal.calibrate(tr, rcfg, _n_pos(pos, split, "TRAIN"))
    a, b = split.periods["TRAIN"]
    doc = {
        "version": 1,
        "status": "CALIBRATED_ON_TRAIN",
        "calibrated_on": {
            "period": "TRAIN",
            "start": a.isoformat(),
            "end_exclusive": b.isoformat(),
        },
        "inputs_sha256": {
            "p2_rules.yaml": sha256(cp.rules),
            "p2_split.yaml": sha256(cp.split),
            "p2_fx_usd_per_unit.yaml": sha256(P.fx),
        },
        "method": res["method"],
        "tau_global": res["tau_global"],
        "rules": res["rules"],
        "actions": res["actions"],
    }
    header = (
        "# P2: rule thresholds calibrated on TRAIN by scripts.p2.run_p2 calibrate. Do not edit.\n"
        "# Metrics of this calibration are in docs/p2/p2_results.json (calibration).\n"
    )
    write_text_lf(P.thresholds, header + yaml.safe_dump(_sig(doc), sort_keys=False))
    return _sig(
        {
            "tau_curve": res["tau_curve"],
            "actions": res["actions"],
            "method": res["method"],
            "tau_global": res["tau_global"],
            "train": res["train_metrics"],
        }
    )


# ---------------------------------------------------------------- stage: build


def stage_build(
    variant: str,
    P: Paths,
    split: Split,
    rcfg: dict,
    kcfg: dict,
    dcfg: dict,
    pos: pl.DataFrame,
    seed: int,
) -> dict:
    if not P.thresholds.is_file():
        raise RuntimeError("thresholds file missing: run `calibrate` first (TRAIN only)")
    thr = rules.load_thresholds(P.thresholds)
    out: dict = {"thresholds_sha256_before_any_test_count": sha256(P.thresholds)}

    # ---- alerts (runtime store)
    parts = []
    for ws in split.window_starts():
        st = pl.read_parquet(P.stats_file(ws))
        f = rules.fire(st, thr, rcfg)
        parts.append(build.alerts_from_fired(f, variant, split.period_of(ws), seed))
    alerts = pl.concat(parts).sort("window_start", "alert_id")
    _step(f"validating {alerts.height:,} alerts against the frozen contract")
    build.validate_alerts(alerts)
    write_parquet(alerts, P.alerts)
    out["alerts_per_day"] = {
        r["d"]: r["len"]
        for r in alerts.group_by(pl.col("window_start").dt.date().cast(pl.String).alias("d"))
        .len()
        .sort("d")
        .iter_rows(named=True)
    }

    # ---- labels (evaluation store)
    labels = al.label_alerts(alerts, pos)
    for row in labels.iter_rows(named=True):  # every label must satisfy the frozen AlertLabel
        al.to_alert_label(row)
    write_parquet(labels, P.labels)
    lab = alerts.select(
        "alert_id", "account_key", "period", "window_start", "triggered_rules", "n_rules_triggered"
    ).join(labels, on="alert_id")

    # ---- per-period metrics: rules on TRAIN and VALIDATION; layer on CALIBRATION; TEST counts only
    metrics = {}
    for per in ("TRAIN", "VALIDATION"):
        st = _labelled_stats(P, split, per, pos)
        metrics[per] = cal.evaluate(st, thr, rcfg, _n_pos(pos, split, per))
    c = lab.filter(pl.col("period") == "CALIBRATION")
    n_pos_c = _n_pos(pos, split, "CALIBRATION")
    metrics["CALIBRATION"] = {
        "n_alerts": c.height,
        "n_true_alerts": int(c["is_true_positive"].sum()),
        "precision": c["is_true_positive"].mean(),
        "recall": int(c["is_true_positive"].sum()) / n_pos_c if n_pos_c else None,
        "n_positive_account_days": n_pos_c,
    }
    out["rule_metrics"] = metrics

    # ---- task-0 feasibility (aggregate counts; TEST exposure = counts only)
    counts = {
        per: {
            "n_alerts": int((lab["period"] == per).sum()),
            "n_positive_alerts": int(lab.filter(pl.col("period") == per)["is_true_positive"].sum()),
            "n_days": len(split.window_starts(per)),
        }
        for per in ("TRAIN", "VALIDATION", "CALIBRATION", "TEST")
    }
    t = lab.filter(pl.col("period") == "TEST")
    train_accts = lab.filter(pl.col("period") == "TRAIN")["account_key"].unique()
    tb = t.filter(~pl.col("account_key").is_in(train_accts.implode()))
    counts["TEST_regime_B_unseen_accounts"] = {
        "n_alerts": tb.height,
        "n_positive_alerts": int(tb["is_true_positive"].sum()),
    }

    # ---- AGENT-DEV / AGENT-TEST split (evaluation store)
    _step("AGENT-DEV / AGENT-TEST group split")
    legs = pl.read_parquet(P.legs)
    scfg = split.raw["agent_split"]
    ts0, ts1 = split.periods["TEST"]
    assign = gs.split_test_alerts(
        t.select("alert_id", "account_key"),
        legs,
        ts0,
        ts1,
        scfg["hash_key"],
        scfg["fraction_agent_test"],
    )
    disj = gs.check_disjoint(assign, legs)
    write_parquet(assign, P.split)
    write_parquet(
        assign.filter(pl.col("agent_group") == gs.AGENT_DEV).select("alert_id"), P.dev_ids
    )
    ga = assign.join(t.select("alert_id", "is_true_positive"), on="alert_id")
    comp = ga.group_by("component").agg(
        pl.len().alias("n"), pl.col("is_true_positive").sum().alias("npos")
    )
    n_test_pos = max(int(t["is_true_positive"].sum()), 1)
    counts["TEST_agent_groups"] = {
        g: {
            "n_alerts": int((ga["agent_group"] == g).sum()),
            "n_positive_alerts": int(
                ga.filter(pl.col("agent_group") == g)["is_true_positive"].sum()
            ),
        }
        for g in (gs.AGENT_DEV, gs.AGENT_TEST)
    }
    counts["TEST_components"] = {
        "n_components": comp.height,
        "largest_component_alerts": int(comp["n"].max()),
        "largest_component_share_of_positive_alerts": _sig(int(comp["npos"].max()) / n_test_pos),
        "n_components_with_positive": int((comp["npos"] > 0).sum()),
    } | disj
    fz = split.raw["feasibility"]
    gates = {
        f"{p}_min_positive": counts[p]["n_positive_alerts"] >= m
        for p, m in fz["min_positive_alerts"].items()
    }
    for g in (gs.AGENT_DEV, gs.AGENT_TEST):
        gates[f"{g}_min_positive"] = (
            counts["TEST_agent_groups"][g]["n_positive_alerts"]
            >= fz["min_positive_alerts_per_agent_group"]
        )
    gates["largest_component_share"] = (
        counts["TEST_components"]["largest_component_share_of_positive_alerts"]
        <= fz["max_share_of_test_positives_in_one_component"]
    )
    counts["regime_B_underpowered"] = (
        counts["TEST_regime_B_unseen_accounts"]["n_positive_alerts"]
        < fz["regime_b_min_positive_alerts"]
    )
    accepted = {d["gate"]: d for d in fz.get("accepted_deviations") or []}
    for g in accepted:
        if g not in gates:
            raise RuntimeError(f"accepted deviation for unknown gate {g}")
    failing = sorted(g for g, ok in gates.items() if not ok)
    out["feasibility"] = {
        "counts": counts,
        "gates": gates,
        "all_pass": not failing,
        "failing_gates": failing,
        "accepted_deviations": sorted(g for g in failing if g in accepted),
        "all_pass_with_accepted_deviations": all(g in accepted for g in failing),
    }

    # ---- KYC v2 (runtime store) + planting record (eval store); pre-TEST alerts only
    pre = alerts.filter(pl.col("period").is_in(PRE_TEST_PERIODS))
    out["kyc"] = kyc_block(
        P, split, rcfg, kcfg, pre, labels.join(pre.select("alert_id"), on="alert_id", how="semi")
    )

    # ---- dispositions (runtime store)
    _step("simulated dispositions (TRAIN, VALIDATION, CALIBRATION)")
    d, audit = disp.simulate(alerts, labels, dcfg)
    disp.validate(d)
    write_parquet(d, P.disp)
    out["dispositions"] = audit | {"n": d.height}
    return out


def kyc_block(
    P: Paths,
    split: Split,
    rcfg: dict,
    kcfg: dict,
    pre_alerts: pl.DataFrame,
    pre_labels: pl.DataFrame,
) -> dict:
    """KYC v2 generation (label-free) + measurement on pre-TEST alerts only.

    pre_alerts / pre_labels must contain TRAIN, VALIDATION, CALIBRATION rows only: nothing in this
    block reads TEST (Dani's rule, 2026-09-30)."""
    if set(pre_alerts["period"].unique()) - set(PRE_TEST_PERIODS):
        raise RuntimeError("kyc_block received non pre-TEST alerts")
    _step("synthetic KYC v2 (label-free)")
    attrs = pl.read_parquet(P.keys)
    bi = pl.read_parquet(P.burn_in)
    base = kycmod.base_kyc(attrs, bi, kcfg)
    planted, record = kycmod.plant_from_behaviour(base, bi, kcfg)
    kyc = kycmod.derive_risk(planted, kcfg)
    write_parquet(kyc, P.kyc)
    write_parquet(record, P.planting)

    # realised planting rates among pre-TEST alerted accounts, real vs false (task 7)
    acc = (
        pre_alerts.select("alert_id", "account_key", "triggered_rules")
        .join(pre_labels.select("alert_id", "is_true_positive"), on="alert_id")
        .group_by("account_key")
        .agg(
            pl.col("is_true_positive").any().alias("laundering_account"),
            pl.col("triggered_rules")
            .list.explode(keep_nulls=False, empty_as_null=False)
            .unique()
            .alias("rules"),
        )
        .join(record, on="account_key", how="left")
        .with_columns(
            pl.col("planted").fill_null(False),
            pl.col("candidate").fill_null(False),
            pl.col("rules")
            .list.contains(pl.col("dominant_behaviour"))
            .fill_null(False)
            .alias("same"),
        )
        .with_columns((pl.col("planted") & pl.col("same")).alias("explains_own_alert"))
    )

    def rate(df: pl.DataFrame, c: str) -> float | None:
        return _sig(df[c].mean()) if df.height else None

    lau, leg = acc.filter(pl.col("laundering_account")), acc.filter(~pl.col("laundering_account"))
    planting = {
        "method": kcfg["planting"]["method"],
        "scope": "accounts with >= 1 TRAIN/VALIDATION/CALIBRATION alert",
        "n_planted_all_accounts": int(record["planted"].sum()),
        "n_candidates_all_accounts": int(record["candidate"].sum()),
        "n_alerted_accounts": acc.height,
        "n_laundering_alerted_accounts": lau.height,
        "n_legitimate_alerted_accounts": leg.height,
        "realised_rate_laundering": rate(lau, "planted"),
        "realised_rate_legitimate": rate(leg, "planted"),
        "explains_own_alert_rate_laundering": rate(lau, "explains_own_alert"),
        "explains_own_alert_rate_legitimate": rate(leg, "explains_own_alert"),
    }
    block = {
        "version": 2,
        "n_accounts": kyc.height,
        "planting": planting,
        "distributions": {
            c: dict(sorted(Counter(kyc[c].to_list()).items()))
            for c in (
                "entity_type",
                "country_risk",
                "expected_activity_band",
                "customer_risk_rating",
            )
        },
        "n_sectors": kyc["sector_or_occupation"].n_unique(),
    }
    _step("KYC leakage tests (TRAIN -> VALIDATION)")
    ids = rules.rule_ids(rcfg)
    block["leakage"] = kyc_leakage.leakage_tests(pre_alerts, pre_labels, kyc, kcfg, ids)
    block["leakage_unseen_accounts_report_only"] = kyc_leakage.leakage_tests(
        pre_alerts, pre_labels, kyc, kcfg, ids, unseen_only=True
    )
    _step("identity guard on KYC (seen vs unseen VALIDATION accounts)")
    block["identity_guard"] = kyc_leakage.kyc_identity_guard(pre_alerts, pre_labels, kyc, kcfg, ids)
    return block


def stage_kyc(
    variant: str, interim: Path, P: Paths, split: Split, rcfg: dict, kcfg: dict, cp
) -> dict:
    """Regenerate KYC v2 and its leakage tests WITHOUT reading TEST (alerts filtered on read,
    labels semi-joined on read)."""
    need = {"account_key", *rules.ALL_STATS}
    if not P.burn_in.is_file() or not need <= set(pl.read_parquet_schema(P.burn_in).names()):
        verify_interim(interim, variant, cp.sources)
        tu = fxmod.with_usd(
            scan_transactions(interim_path(interim, variant, "trans")), fxmod.load(P.fx)
        )
        _step("burn-in statistics (KYC activity band and v2 planting)")
        write_parquet(burn_in_stats(tu, split, rcfg), P.burn_in)
    pre = pl.scan_parquet(P.alerts).filter(pl.col("period").is_in(PRE_TEST_PERIODS)).collect()
    labels = (
        pl.scan_parquet(P.labels)
        .join(pre.select("alert_id").lazy(), on="alert_id", how="semi")
        .select("alert_id", "is_true_positive")
        .collect()
    )
    if labels.height != pre.height:
        raise RuntimeError("pre-TEST labels do not match pre-TEST alerts")
    return kyc_block(P, split, rcfg, kcfg, pre, labels)


def store_fingerprints(P: Paths) -> dict:
    files = sorted([*P.runtime.glob("*.parquet"), *P.eval.glob("*.parquet")])
    return {
        f"{f.parent.name}/{f.name}": {
            "sha256": sha256(f),
            "columns": pl.read_parquet_schema(f).names(),
            "rows": pl.scan_parquet(f).select(pl.len()).collect().item(),
        }
        for f in files
    }


# ---------------------------------------------------------------- main


@dataclass(frozen=True)
class ConfigPaths:
    split: Path = SPLIT_CONFIG
    rules: Path = rules.RULES_CONFIG
    kyc: Path = kycmod.KYC_CONFIG
    dispositions: Path = DISP_CONFIG
    sources: Path = DATA_SOURCES


def run(
    stage: str,
    variant: str,
    interim: Path,
    out_root: Path,
    configs_dir: Path,
    results: Path,
    seed: int,
    cp: ConfigPaths | None = None,
    primary: str = PRIMARY,
) -> dict:
    cp = cp or ConfigPaths()
    split = load_split(cp.split)
    rcfg = rules.load_rules(cp.rules)
    kcfg = kycmod.load_kyc_config(cp.kyc)
    dcfg = yaml.safe_load(cp.dispositions.read_text(encoding="utf-8"))
    P = Paths(out_root, variant, configs_dir)
    t0 = time.perf_counter()
    doc: dict = {"p2_version": P2_VERSION, "variant": variant}
    if results.is_file():
        prev = json.loads(results.read_text(encoding="utf-8"))
        if prev.get("variant") == variant:
            doc = prev
    doc["split"] = {
        "status": split.status,
        "l_max_hours": split.l_max.total_seconds() / 3600,
        "periods": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in split.periods.items()},
    }
    if stage in ("calibrate", "all"):
        if variant != primary:
            raise RuntimeError("calibration runs on the primary variant only")
        doc["stats"] = stage_stats(variant, interim, P, split, rcfg, kcfg, True, cp.sources)
        pos = stage_positive_days(variant, interim, P, split)
        # pre-TEST periods only here; the TEST aggregate is computed in `build`, after thresholds
        doc["positive_account_days_per_period"] = {
            p: _n_pos(pos, split, p) for p in PRE_TEST_PERIODS
        }
        doc["calibration"] = stage_calibrate(P, split, rcfg, pos, cp)
    if stage == "kyc":
        doc.setdefault("build", {})["kyc"] = stage_kyc(variant, interim, P, split, rcfg, kcfg, cp)
        fps = store_fingerprints(P)
        doc.setdefault("stores", {})
        for k in ("runtime/kyc.parquet", "eval/planting.parquet"):
            doc["stores"][k] = fps[k]
    if stage in ("build", "all"):
        if not P.stats_dir.is_dir():
            doc["stats"] = stage_stats(variant, interim, P, split, rcfg, kcfg, False, cp.sources)
        pos = (
            pl.read_parquet(P.pos)
            if P.pos.is_file()
            else stage_positive_days(variant, interim, P, split)
        )
        doc["build"] = stage_build(variant, P, split, rcfg, kcfg, dcfg, pos, seed)
        doc["positive_account_days_per_period"] = {
            p: _n_pos(pos, split, p) for p in (*PRE_TEST_PERIODS, "TEST")
        }
        doc["stores"] = store_fingerprints(P)
    doc["configs_sha256"] = {
        f.name: sha256(f)
        for f in sorted([*(ROOT / "configs").glob("p2_*.yaml"), *configs_dir.glob("p2_*.yaml")])
    }
    doc["run_info"] = {
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "polars": pl.__version__,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "code_sha256": {
            f"{f.parent.name}/{f.name}": sha256(f)
            for f in sorted(
                [
                    *(ROOT / "scripts" / "p2").glob("*.py"),
                    *(ROOT / "src" / "alerts").glob("*.py"),
                    *(ROOT / "src" / "data").glob("*.py"),
                    *(ROOT / "src" / "eval").glob("*.py"),
                ]
            )
        },
    }
    doc = _sig(doc)
    results.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(results, json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return doc


def compare_runs(a: Path, b: Path) -> list[str]:
    diffs = []
    fa = {p.relative_to(a): p for p in a.rglob("*") if p.is_file()}
    fb = {p.relative_to(b): p for p in b.rglob("*") if p.is_file()}
    for k in sorted(set(fa) | set(fb)):
        if k not in fa or k not in fb:
            diffs.append(f"{k}: only in one run")
        elif sha256(fa[k]) != sha256(fb[k]):
            diffs.append(f"{k}: bytes differ")
    return diffs


def compare_thresholds(original: Path, regenerated: Path) -> tuple[list[str], list[str]]:
    """Thresholds file check for --verify.

    Everything calibration PRODUCED (method, taus, thresholds, actions, calibrated_on) must be
    identical. `inputs_sha256` records the config files as they were at calibration time; a config
    edited afterwards (e.g. p2_split.yaml gaining accepted_deviations or status FROZEN) changes that
    record without changing any threshold. Such changes are reported as notes, never hidden."""
    a = yaml.safe_load(original.read_text(encoding="utf-8"))
    b = yaml.safe_load(regenerated.read_text(encoding="utf-8"))
    ia, ib = a.pop("inputs_sha256", {}), b.pop("inputs_sha256", {})
    diffs = [
        f"configs/p2_rule_thresholds.yaml: '{k}' differs"
        for k in sorted(set(a) | set(b))
        if a.get(k) != b.get(k)
    ]
    notes = [
        f"configs/{k} changed after calibration (recorded {str(ia.get(k))[:12]}, now "
        f"{str(ib.get(k))[:12]}); calibration outputs are identical"
        for k in sorted(set(ia) | set(ib))
        if ia.get(k) != ib.get(k)
    ]
    return diffs, notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dossier P2: alerts, labels, KYC, dispositions")
    ap.add_argument("stage", choices=["calibrate", "kyc", "build", "all"])
    ap.add_argument("--variant", default=PRIMARY)
    ap.add_argument("--interim-dir", type=Path, default=ROOT / "data" / "interim")
    ap.add_argument("--out-root", type=Path, default=ROOT / "data" / "p2")
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="regenerate everything elsewhere; outputs must be byte-identical",
    )
    args = ap.parse_args(argv)
    print(f"Dossier P2 v{P2_VERSION} (polars {pl.__version__})")
    print(f"stage={args.stage} variant={args.variant}")
    if not args.verify:
        doc = run(
            args.stage,
            args.variant,
            args.interim_dir,
            args.out_root,
            ROOT / "configs",
            args.results,
            args.seed,
        )
        print(
            json.dumps(
                {
                    "feasibility": doc.get("build", {}).get("feasibility", {}).get("gates"),
                    "kyc_leakage_all_pass": doc.get("build", {})
                    .get("kyc", {})
                    .get("leakage", {})
                    .get("all_pass"),
                },
                indent=2,
            )
        )
        print(f"wrote {args.results}")
        return 0
    vroot = ROOT / "data" / "p2_verify"
    vcfg = vroot / "configs"
    if vroot.exists():
        shutil.rmtree(vroot)
    vcfg.mkdir(parents=True)
    for f in ("p2_fx_usd_per_unit.yaml", "p2_rule_thresholds.yaml"):
        if (ROOT / "configs" / f).is_file() and args.stage == "build":
            shutil.copy(ROOT / "configs" / f, vcfg / f)
    run(
        "all" if args.stage == "all" else args.stage,
        args.variant,
        args.interim_dir,
        vroot,
        vcfg,
        vroot / "p2_results.verify.json",
        args.seed,
    )
    diffs = compare_runs(args.out_root / args.variant / "runtime", vroot / args.variant / "runtime")
    diffs += compare_runs(args.out_root / args.variant / "eval", vroot / args.variant / "eval")
    f = "p2_fx_usd_per_unit.yaml"
    if (vcfg / f).is_file() and sha256(vcfg / f) != sha256(ROOT / "configs" / f):
        diffs.append(f"configs/{f}: bytes differ")
    notes: list[str] = []
    f = "p2_rule_thresholds.yaml"
    if (vcfg / f).is_file():
        d, notes = compare_thresholds(ROOT / "configs" / f, vcfg / f)
        diffs += d
    if diffs:
        print(f"REGENERATION FAILED: {len(diffs)} differences")
        for d in diffs[:40]:
            print("  " + d)
        return 1
    print(
        "REPRODUCED: every data file and the FX table are byte-identical; "
        "thresholds file identical in every calibrated value"
    )
    for n in notes:
        print("  note: " + n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
