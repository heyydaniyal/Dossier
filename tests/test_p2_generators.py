"""P2 tasks 6, 7, 9 and the FX/calibration units: KYC generator audit, planting balance,
leakage ceilings catch a planted clue, disposition error model, FX recovery, calibration logic.

Test required by the phase: KYC-only classifier PR-AUC (VALIDATION) at or below the ceiling
declared before measurement -> the mechanism is tested here on synthetic data; the real value
is asserted in tests/test_p2_results_facts.py once docs/p2/p2_results.json exists.
"""

from __future__ import annotations

import ast
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scripts.p2 import calibrate as cal
from scripts.p2 import dispositions as disp
from scripts.p2 import kyc_leakage
from scripts.p2 import plant as planting
from src.alerts import rules
from src.data import fx
from src.data import kyc as kycmod
from src.data.transactions import DataError

ROOT = Path(__file__).resolve().parents[1]
KCFG = kycmod.load_kyc_config()
RCFG = rules.load_rules()
DCFG = __import__("yaml").safe_load((ROOT / "configs" / "p2_dispositions.yaml").read_text())
D = lambda day: datetime(2022, 9, day, tzinfo=UTC)  # noqa: E731


def _attrs(n: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    types = ["Corporation", "Partnership", "Sole Proprietorship", "Individual", "Country"]
    return pl.DataFrame(
        {
            "account_key": [f"{i % 7:03d}|K{i:06d}" for i in range(n)],
            "entity_type": rng.choice(types, n, p=[0.3, 0.3, 0.3, 0.05, 0.05]).tolist(),
            "bank_location": rng.choice(["Chicago", "Spain", "Japan"], n).tolist(),
        }
    ).with_columns(
        pl.when(pl.col("bank_location") == "Chicago")
        .then(pl.lit("United States"))
        .otherwise(pl.col("bank_location"))
        .alias("bank_country")
    )


def _burn(attrs: pl.DataFrame, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    k = attrs.sample(fraction=0.6, seed=seed)
    return k.select("account_key").with_columns(
        pl.Series("volume_usd", rng.lognormal(8, 2, k.height))
    )


# ---------------------------------------------------------------- KYC generator audit (task 6)


def test_kyc_generator_source_never_touches_ground_truth():
    src = (ROOT / "src" / "data" / "kyc.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [
        m
        for m in imports
        if m and (m.startswith("src.eval") or m.startswith("scripts") or m.startswith("src.alerts"))
    ]
    body = src.split('"""', 2)[2]  # skip the module docstring, which explains what is forbidden
    for tok in ("is_laundering", "laundering", "pattern", "typolog", "is_true_positive", "alert"):
        assert tok not in body.lower(), tok


def test_planting_is_the_only_label_reader_and_is_declared():
    src = (ROOT / "scripts" / "p2" / "plant.py").read_text(encoding="utf-8")
    assert "DECLARED FIREWALL EXCEPTION" in src
    assert "is_true_positive" in src


def test_kyc_is_deterministic_and_order_free():
    a, b = _attrs(3000), None
    b = _burn(a)
    k1 = kycmod.derive_risk(kycmod.base_kyc(a, b, KCFG), KCFG)
    k2 = kycmod.derive_risk(
        kycmod.base_kyc(a.reverse(), b.sample(fraction=1.0, shuffle=True, seed=3), KCFG), KCFG
    )
    assert k1.equals(k2)
    assert k1.columns == kycmod.KYC_COLUMNS
    assert k1.null_count().sum_horizontal().item() == 0
    assert (k1["onboarding_date"] <= date(2022, 8, 31)).all()
    other = dict(KCFG, seed=KCFG["seed"] + 1)
    k3 = kycmod.derive_risk(kycmod.base_kyc(a, b, other), other)
    assert not k1.equals(k3)  # the seed matters


def test_activity_band_follows_burn_in_rank_within_entity_type():
    a = _attrs(20000)
    b = _burn(a)
    k = kycmod.base_kyc(a, b, dict(KCFG, activity_band=dict(KCFG["activity_band"], noise_prob=0.0)))
    j = k.join(b, on="account_key", how="left")
    act = j.filter(pl.col("volume_usd").is_not_null())
    med = (
        act.group_by("expected_activity_band").agg(pl.col("volume_usd").median()).sort("volume_usd")
    )
    assert med["expected_activity_band"].to_list() == ["low", "medium", "high", "very_high"]
    share_low = (act["expected_activity_band"] == "low").mean()
    assert 0.45 < share_low < 0.55


# ---------------------------------------------------------------- planting (task 7)


def _alerted_population(n: int = 20000, prev: float = 0.05, seed: int = 5):
    rng = np.random.default_rng(seed)
    a = _attrs(n, seed)
    rule = rng.choice(rules.rule_ids(RCFG), n).tolist()
    alerts = pl.DataFrame(
        {
            "alert_id": [f"A{i}" for i in range(n)],
            "account_key": a["account_key"],
            "window_start": [D(2 + int(x)) for x in rng.integers(0, 4, n)],
            "triggered_rules": [[r] for r in rule],
        }
    )
    labels = pl.DataFrame(
        {"alert_id": alerts["alert_id"], "is_true_positive": rng.random(n) < prev}
    )
    return a, alerts, labels


def test_planting_is_label_balanced_and_leaves_no_trace():
    a, alerts, labels = _alerted_population()
    base = kycmod.base_kyc(a, _burn(a), KCFG)
    planted, record, rates = planting.plant(base, alerts, labels, KCFG)
    assert rates["realised_rate_laundering"] == pytest.approx(0.30, abs=0.03)
    assert rates["realised_rate_legitimate"] == pytest.approx(0.30, abs=0.01)
    kyc = kycmod.derive_risk(planted, KCFG)
    assert kyc.columns == kycmod.KYC_COLUMNS  # no planted flag, no label
    changed = kyc.join(kycmod.derive_risk(base, KCFG), on="account_key", suffix="_b").filter(
        pl.col("sector_or_occupation") != pl.col("sector_or_occupation_b")
    )
    assert changed.height > 0
    # planted sectors match the rule archetype
    r = record.filter(pl.col("planted")).join(kyc, on="account_key")
    ok = [
        s in KCFG["planting"]["archetypes"][p]
        or s in (KCFG["planting"]["individual_occupation"], kycmod.GOVERNMENT)
        for s, p in zip(r["sector_or_occupation"], r["primary_rule"], strict=True)
    ]
    assert all(ok)


# ---------------------------------------------------------------- leakage ceilings (task 6)


def _leak_data(n: int = 12000, leak: bool = False):
    rng = np.random.default_rng(11)
    a, alerts, labels = _alerted_population(n, prev=0.08, seed=11)
    y = labels["is_true_positive"].to_numpy()
    alerts = alerts.with_columns(
        pl.Series("period", np.where(np.arange(n) < n * 0.6, "TRAIN", "VALIDATION")),
        pl.lit(1).alias("n_rules_triggered"),
        *[
            pl.Series(c, rng.lognormal(5, 1, n) + (y * 50 if c == "volume_usd" else 0))
            for c in rules.ALL_STATS
        ],
    )
    kyc = kycmod.derive_risk(kycmod.base_kyc(a, _burn(a), KCFG), KCFG)
    if leak:  # plant a clue: laundering accounts get a rare sector
        clue = dict(zip(alerts["account_key"], y, strict=True))
        kyc = kyc.with_columns(
            pl.when(pl.col("account_key").replace_strict(clue, default=False))
            .then(pl.lit("crypto_exchange"))
            .otherwise(pl.lit("manufacturing"))
            .alias("sector_or_occupation")
        )
    return alerts, labels, kyc


def test_ceilings_pass_on_clean_kyc():
    alerts, labels, kyc = _leak_data()
    r = kyc_leakage.leakage_tests(alerts, labels, kyc, KCFG, rules.rule_ids(RCFG))
    assert r["status"] == "measured"
    assert r["checks"]["C1_kyc_only"]["pass"], r
    assert r["checks"]["C2_synthetic_increment"]["pass"], r


def test_ceilings_catch_a_planted_clue():
    alerts, labels, kyc = _leak_data(leak=True)
    r = kyc_leakage.leakage_tests(alerts, labels, kyc, KCFG, rules.rule_ids(RCFG))
    assert not r["checks"]["C1_kyc_only"]["pass"]
    assert not r["checks"]["C2_synthetic_increment"]["pass"]
    assert not r["checks"]["C4_txn_synthetic_increment"]["pass"]
    assert not r["all_pass"]


# ---------------------------------------------------------------- dispositions (task 9)


def _disp_population(n: int = 40000):
    rng = np.random.default_rng(3)
    nr = rng.choice([1, 2, 3], n, p=[0.6, 0.3, 0.1])
    per = rng.choice(["TRAIN", "VALIDATION", "CALIBRATION", "TEST"], n)
    tp = rng.random(n) < 0.2
    typ = rng.choice(["FAN-IN", "RANDOM", "NONE"], n)
    alerts = pl.DataFrame(
        {
            "alert_id": [f"A{i}" for i in range(n)],
            "account_key": [f"001|K{i}" for i in range(n)],
            "period": per,
            "as_of": [D(5)] * n,
            "triggered_rules": [["R04_FAN_IN"] * int(k) for k in nr],
            "n_rules_triggered": nr,
        }
    )
    labels = pl.DataFrame(
        {
            "alert_id": alerts["alert_id"],
            "is_true_positive": tp,
            "typologies": [
                ([t] if (p and t != "NONE") else []) for p, t in zip(tp, typ, strict=True)
            ],
        }
    )
    return alerts, labels


def test_dispositions_realise_the_declared_error_model():
    alerts, labels = _disp_population()
    d, audit = disp.simulate(alerts, labels, DCFG)
    assert disp.validate(d) == (alerts["period"] != "TEST").sum()
    one = audit["by_n_rules"]["1"]
    em = DCFG["error_model"]
    easy = audit["easy_laundering"]
    assert audit["by_period"].keys() == {"TRAIN", "VALIDATION", "CALIBRATION"}
    # anchoring: more rules -> more confirmations for the same truth
    assert audit["by_n_rules"]["3+"]["confirm_rate"] > one["confirm_rate"]
    assert audit["by_n_rules"]["3+"]["false_positive_rate"] > one["false_positive_rate"]
    # hard cases are missed more often than easy ones
    assert (
        audit["hard_H1_shapeless_laundering"]["false_negative_rate"] > easy["false_negative_rate"]
    )
    # overall accuracy < 1 (errors exist) and near the declared rates for single-rule alerts
    assert audit["overall"]["accuracy"] < 1
    assert one["false_positive_rate"] == pytest.approx(em["false_confirm_rate"], abs=0.006)
    j = d.join(alerts.select("alert_id", "as_of"), on="alert_id")
    assert (j["closed_at"] > j["as_of"]).all()
    assert (j["closed_at"] - j["as_of"]).max() <= timedelta(
        hours=DCFG["handling_delay_hours"]["max"]
    )


def test_disposition_generator_never_uses_scores_or_agents():
    src = (ROOT / "scripts" / "p2" / "dispositions.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    mods += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [
        m for m in mods if m.startswith(("src.models", "src.agents", "src.tools", "src.features"))
    ]
    assert "score" not in src.split('"""', 2)[2].lower()


# ---------------------------------------------------------------- FX (D6 on TRAIN)


def _fx_frame(rows):
    return (
        pl.DataFrame(
            rows,
            schema=[
                "timestamp",
                "payment_currency",
                "receiving_currency",
                "amount_paid",
                "amount_received",
            ],
            orient="row",
        )
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        .lazy()
    )


def test_fx_recovers_fixed_rates_from_train_only():
    rows = []
    for i in range(50):
        ts = D(3) + timedelta(minutes=i)
        rows += [
            (ts, "Euro", "US Dollar", 100.0 + i, (100.0 + i) * 1.1),
            (ts, "Yen", "Euro", 1000.0 + i, (1000.0 + i) * 0.0075 / 1.1),  # Yen only via Euro
            (ts, "US Dollar", "US Dollar", 5.0, 5.0),
        ]
    rows.append((D(10), "Euro", "US Dollar", 1.0, 999.0))  # outside TRAIN: ignored
    r = fx.derive_usd_per_unit(_fx_frame(rows), D(2), D(6))
    assert r["usd_per_unit"]["US Dollar"] == 1.0
    assert r["usd_per_unit"]["Euro"] == pytest.approx(1.1, rel=1e-6)
    assert r["usd_per_unit"]["Yen"] == pytest.approx(0.0075, rel=1e-4)


def test_fx_fails_loudly_without_a_path_to_usd_and_on_unknown_currency():
    rows = [(D(3), "Euro", "Euro", 1.0, 1.0), (D(3), "US Dollar", "US Dollar", 1.0, 1.0)]
    with pytest.raises(DataError, match="no cross-currency path"):
        fx.derive_usd_per_unit(_fx_frame(rows), D(2), D(6))
    lf = pl.DataFrame(
        {
            "amount_paid": [1.0],
            "payment_currency": ["Zloty"],
            "amount_received": [1.0],
            "receiving_currency": ["US Dollar"],
        }
    ).lazy()
    with pytest.raises(pl.exceptions.InvalidOperationError):
        fx.with_usd(lf, {"US Dollar": 1.0}).collect()


# ---------------------------------------------------------------- calibration procedure


def _cal_stats(n: int = 50000):
    rng = np.random.default_rng(9)
    pos = rng.random(n) < 0.01
    df = pl.DataFrame({"peer_group": rng.choice(["Corporation", "Partnership"], n)})
    cols = {}
    for c in rules.ALL_STATS:
        base = rng.poisson(1, n).astype(float) if c.startswith("n_") else rng.lognormal(7, 1, n)
        cols[c] = base + np.where(pos & (rng.random(n) < 0.3), base * 20 + 20, 0)
    return df.with_columns(
        *[pl.Series(k, v) for k, v in cols.items()], pl.Series("is_pos", pos)
    ), int(pos.sum())


GLOBAL = dict(RCFG, calibration=dict(RCFG["calibration"], method="global_tau"))


def test_revision0_global_tau_hits_the_band():
    st, npos = _cal_stats()
    res = cal.calibrate(st, GLOBAL, npos)
    lo, hi = RCFG["calibration"]["precision_band"]
    assert lo <= res["train_metrics"]["precision"] <= hi
    assert res["method"] == "global_tau" and res["tau_global"] in RCFG["calibration"]["tau_grid"]
    for r, spec in res["rules"].items():
        assert (spec.get("threshold") or spec.get("default")) >= RCFG["rules"][r]["floor"]


def test_revision0_refuses_when_band_unreachable():
    st, npos = _cal_stats()
    bad = dict(
        GLOBAL,
        calibration=dict(GLOBAL["calibration"], precision_band=[0.5, 0.6], precision_target=0.55),
    )
    with pytest.raises(cal.CalibrationError, match="no tau"):
        cal.calibrate(st, bad, npos)


def _per_rule_stats(n: int = 200000):
    """Signal planted in R04 (strong), R05 (weak) and nowhere else; others pure noise."""
    rng = np.random.default_rng(21)
    pos = rng.random(n) < 0.003
    cols = {c: rng.lognormal(7, 1, n) for c in rules.ALL_STATS}
    for c in rules.ALL_STATS:
        if c.startswith("n_"):
            cols[c] = rng.poisson(0.3, n).astype(float)
    # R04: 60% of positives at 8 senders, plus ~1.6% of negatives -> precision ~10%
    fan_in = np.where(rng.random(n) < 0.016, 8.0, rng.poisson(0.5, n))
    cols["n_distinct_senders"] = np.where(pos & (rng.random(n) < 0.6), 8.0, fan_in)
    # R05: 20% of positives at 7 receivers, plus ~1% of negatives -> precision ~6%
    fan_out = np.where(rng.random(n) < 0.01, 7.0, rng.poisson(0.8, n))
    cols["n_distinct_receivers"] = np.where(pos & (rng.random(n) < 0.2), 7.0, fan_out)
    df = pl.DataFrame({"peer_group": ["Corporation"] * n})
    return df.with_columns(
        *[pl.Series(k, v) for k, v in cols.items()], pl.Series("is_pos", pos)
    ), int(pos.sum())


def test_revision1_picks_the_loosest_level_meeting_the_target_and_drops_noise():
    st, npos = _per_rule_stats()
    cfg = dict(RCFG, calibration=dict(RCFG["calibration"], min_active_rules=1))
    res = cal.calibrate(st, cfg, npos)
    assert res["method"] == "per_rule" and res["tau_global"] is None
    act = {x["rule"]: x for x in res["actions"] if x["action"].startswith(("select", "drop"))}
    assert act["R04_FAN_IN"]["action"] == "select"
    for noise in ("R06_RAPID_IN_OUT", "R08_CROSS_CURRENCY_CHURN", "R09_ROUND_AMOUNTS"):
        assert act[noise]["action"].startswith("drop"), noise
        assert not res["rules"][noise]["active"]
    # the chosen level meets the target and nothing looser does
    target = RCFG["calibration"]["precision_target"]
    n_min = RCFG["calibration"]["per_rule_min_true_alerts"]
    tab = [t for t in res["tau_curve"] if t["rule"] == "R04_FAN_IN"]
    chosen = act["R04_FAN_IN"]["tau"]
    ok = [
        t
        for t in tab
        if t["precision"] is not None and t["precision"] >= target and t["n_true"] >= n_min
    ]
    assert chosen == min(t["tau"] for t in ok)
    lo, hi = RCFG["calibration"]["precision_band"]
    assert lo <= res["train_metrics"]["precision"] <= hi


def test_revision1_refuses_with_too_few_rules():
    st, npos = _per_rule_stats()
    with pytest.raises(cal.CalibrationError, match="fewer than 6 active rules"):
        cal.calibrate(st, RCFG, npos)


def test_calibration_firing_equals_runtime_firing():
    """The numpy firing used for calibration must match src.alerts.rules.fire exactly."""
    st, npos = _cal_stats(20000)
    a = cal.Arrays(st, RCFG)
    taus = dict.fromkeys(rules.rule_ids(RCFG), 0.99)
    active = {r: r != "R08_CROSS_CURRENCY_CHURN" for r in rules.rule_ids(RCFG)}
    thr = cal.thresholds_at(a, RCFG, taus, active)
    fast = cal.fired(a, thr, RCFG)
    slow = rules.fire(st, thr, RCFG)
    assert set(fast) == {r for r in active if active[r]}
    for r, v in fast.items():
        assert (slow[f"fired_{r}"].to_numpy() == v).all(), r


# ---------------------------------------------------------------- bank-name formats (real, printed)


def _accounts(tmp_path, names: list[str]) -> tuple:
    from src.data.accounts import account_attributes

    rows = [(str(i + 1), f"ACC{i}", n, "Corporation") for i, n in enumerate(names)]
    p = tmp_path / "acc.parquet"
    pl.DataFrame(
        rows, schema=["bank_id", "account_number", "bank_name", "entity_type"], orient="row"
    ).write_parquet(p)
    keys = pl.DataFrame({"account_key": [f"0{i + 1}|ACC{i}" for i in range(len(names))]})
    return account_attributes, p, keys


def test_bank_name_forms_map_to_country(tmp_path):
    fn, p, keys = _accounts(
        tmp_path,
        [
            "Spain Bank #16393",
            "Saudi Arabia Bank #7",
            "Crytpo Bank #3",
            "Hearthstone Bancorp",
            "Savings Bank of Seattle",
        ],
    )
    out = fn(p, keys, KCFG["foreign_countries"], KCFG["crypto_prefixes"]).sort("account_key")
    assert out["bank_country"].to_list() == [
        "Spain",
        "Saudi Arabia",
        "Crypto",
        "United States",
        "United States",
    ]
    assert out["bank_location"].to_list() == [
        "Spain",
        "Saudi Arabia",
        "Crytpo",
        "United States",
        "United States",
    ]


def test_unknown_numbered_prefix_fails_loudly(tmp_path):
    fn, p, keys = _accounts(tmp_path, ["Atlantis Bank #1", "Bank of Miami"])
    with pytest.raises(DataError, match="unknown X"):
        fn(p, keys, KCFG["foreign_countries"], KCFG["crypto_prefixes"])
