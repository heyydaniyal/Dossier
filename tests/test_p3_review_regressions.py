"""Regression tests from the independent P2 review (P3 task 0). Each closes a gap where a real bug
could be introduced with the whole P2 suite still green; each was proven by a mutation that the
old suite missed and this test catches (mutations listed per test).

  M-2  KYC must use burn-in (2022-09-01) transactions only
  M-7  the simulated-disposition error model must realise the FROZEN rates, cell by cell
  M-8  naive timestamps are UTC (D3) end to end: day boundaries in loaders, rules and labels
  M-9  the FX table is fitted on TRAIN only
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from scripts.p2 import dispositions as disp
from src.alerts import rules
from src.data import fx
from src.data.periods import load_split
from src.data.transactions import scan_transactions
from tests import p2_fixture as fxt

ROOT = Path(__file__).resolve().parents[1]
D = lambda day, h=0, m=0: datetime(2022, 9, day, h, m, tzinfo=UTC)  # noqa: E731


# ---------------------------------------------------------------- M-2 KYC from burn-in only


def test_kyc_depends_on_burn_in_transactions_only(p2_run, tmp_path):
    """Perturb every transaction at/after 2022-09-02 00:00 (amounts x7, payment formats rotated),
    drop the cached burn-in statistics and the KYC, re-run `run_p2 kyc`: kyc.parquet must be
    byte-identical. The P2 test only removed TEST alerts/labels and reused the cache, so building
    KYC from TRAIN or even TEST transactions passed it.
    Mutations caught: run_p2.burn_in_stats using split.burn_in[1] / split.periods['TEST'][0]."""
    from scripts.p2 import run_p2

    src = p2_run["root"]
    dst = tmp_path / "copy"
    shutil.copytree(src / "out", dst / "out")
    shutil.copytree(src / "configs", dst / "configs")
    interim = dst / "interim"
    shutil.copytree(p2_run["info"]["interim"], interim)
    tp = interim / "FIX_trans.parquet"
    t = pl.read_parquet(tp)
    later = pl.col("timestamp") >= datetime(2022, 9, 2)
    fmts = ["Cash", "Bitcoin", "ACH", "Wire", "Cheque", "Credit Card", "Reinvestment"]
    rot = {f: fmts[(i + 1) % len(fmts)] for i, f in enumerate(fmts)}
    t = t.with_columns(
        pl.when(later).then(pl.col("amount_paid") * 7).otherwise(pl.col("amount_paid")),
        pl.when(later).then(pl.col("amount_received") * 7).otherwise(pl.col("amount_received")),
        pl.when(later)
        .then(pl.col("payment_format").replace_strict(rot, default=pl.lit("Cash")))
        .otherwise(pl.col("payment_format")),
    )
    assert t.filter(later).height > 0.9 * t.height  # the perturbation touches almost everything
    t.write_parquet(tp)
    base = dst / "out" / "FIX"
    (base / "work" / "burn_in_stats.parquet").unlink()
    # second-pass review: also remove the cached per-day statistics, so KYC code that read them
    # (instead of the burn-in day) would crash here rather than see unperturbed values
    shutil.rmtree(base / "work" / "account_day_stats")
    (base / "runtime" / "kyc.parquet").unlink()
    run_p2.run(
        "kyc",
        "FIX",
        interim,
        dst / "out",
        dst / "configs",
        dst / "results.json",
        20260930,
        cp=p2_run["cp"],
        primary="FIX",
    )
    got = hashlib.sha256((base / "runtime" / "kyc.parquet").read_bytes()).hexdigest()
    want = hashlib.sha256((p2_run["base"] / "runtime" / "kyc.parquet").read_bytes()).hexdigest()
    assert got == want


def test_kyc_test_is_sensitive_to_burn_in_changes(p2_run, tmp_path):
    """Control for the test above: perturbing the BURN-IN day must change the KYC (otherwise the
    test above would pass vacuously)."""
    from scripts.p2 import run_p2

    src = p2_run["root"]
    dst = tmp_path / "copy"
    shutil.copytree(src / "out", dst / "out")
    shutil.copytree(src / "configs", dst / "configs")
    interim = dst / "interim"
    shutil.copytree(p2_run["info"]["interim"], interim)
    tp = interim / "FIX_trans.parquet"
    t = pl.read_parquet(tp)
    burn = pl.col("timestamp") < datetime(2022, 9, 2)
    t.with_columns(
        pl.when(burn).then(pl.col("amount_paid") * 50).otherwise(pl.col("amount_paid")),
        pl.when(burn).then(pl.col("amount_received") * 50).otherwise(pl.col("amount_received")),
    ).write_parquet(tp)
    base = dst / "out" / "FIX"
    (base / "work" / "burn_in_stats.parquet").unlink()
    (base / "runtime" / "kyc.parquet").unlink()
    run_p2.run(
        "kyc", "FIX", interim, dst / "out", dst / "configs", dst / "results.json", 20260930,
        cp=p2_run["cp"], primary="FIX",
    )  # fmt: skip
    got = hashlib.sha256((base / "runtime" / "kyc.parquet").read_bytes()).hexdigest()
    want = hashlib.sha256((p2_run["base"] / "runtime" / "kyc.parquet").read_bytes()).hexdigest()
    assert got != want


# ---------------------------------------------------------------- M-7 disposition error model

DISP_PATH = ROOT / "configs" / "p2_dispositions.yaml"
DCFG = yaml.safe_load(DISP_PATH.read_text(encoding="utf-8"))
# FROZEN values (PROJECT_STATE 2026-09-30, configs/p2_dispositions.yaml sha256 698144e3378fc7be).
FROZEN = {
    "sensitivity": 0.85,
    "hard_sensitivity": 0.70,
    "false_confirm_rate": 0.02,
    "near_threshold_false_confirm_rate": 0.05,
    "anchoring_beta": 0.5,
    "hard_typologies": ["RANDOM", "BIPARTITE"],
    "near_threshold_rule": "R03_STRUCTURING",
    "median_closed": 18,
    "median_confirmed": 48,
    "sigma": 0.6,
    "clip": (1, 168),
}


def test_disposition_config_is_the_frozen_one():
    b = DISP_PATH.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(b).hexdigest().startswith("698144e3378fc7be")
    em, hc, d = DCFG["error_model"], DCFG["error_model"]["hard_cases"], DCFG["handling_delay_hours"]
    assert em["sensitivity"] == FROZEN["sensitivity"]
    assert em["false_confirm_rate"] == FROZEN["false_confirm_rate"]
    assert em["anchoring_beta"] == FROZEN["anchoring_beta"]
    assert hc["hard_sensitivity"] == FROZEN["hard_sensitivity"]
    assert hc["near_threshold_false_confirm_rate"] == FROZEN["near_threshold_false_confirm_rate"]
    assert hc["hard_typologies"] == FROZEN["hard_typologies"]
    assert hc["near_threshold_rule"] == FROZEN["near_threshold_rule"]
    assert d["closed_legitimate"] == {"median": 18, "sigma": 0.6}
    assert d["confirmed_suspicious"] == {"median": 48, "sigma": 0.6}
    assert (d["min"], d["max"]) == FROZEN["clip"]


def _expected(p0: float, n_rules: int) -> float:
    lg = np.log(p0 / (1 - p0)) + FROZEN["anchoring_beta"] * (n_rules - 1)
    return float(1 / (1 + np.exp(-lg)))


# cell -> (is_true_positive, typologies, extra rule, base P(confirm))
CELLS = {
    "easy_attributed": (True, ["FAN-IN"], None, FROZEN["sensitivity"]),
    "easy_mixed_with_hard": (True, ["FAN-IN", "RANDOM"], None, FROZEN["sensitivity"]),
    "hard_random": (True, ["RANDOM"], None, FROZEN["hard_sensitivity"]),
    "hard_bipartite": (True, ["BIPARTITE"], None, FROZEN["hard_sensitivity"]),
    "hard_unattributed_only": (True, [], None, FROZEN["hard_sensitivity"]),
    "legit": (False, [], None, FROZEN["false_confirm_rate"]),
    "legit_near_threshold": (
        False,
        [],
        "R03_STRUCTURING",
        FROZEN["near_threshold_false_confirm_rate"],
    ),
    # H2 applies to legitimate alerts only: a laundering R03 alert is not easier or harder
    "laundering_near_threshold": (True, ["FAN-IN"], "R03_STRUCTURING", FROZEN["sensitivity"]),
}
N_PER_CELL = 6000


def test_dispositions_realise_the_frozen_rates_per_cell():
    """Known-answer test per cell x number of rules: realised confirm rate within 4.5 binomial SE
    of sigmoid(logit p0 + beta (n - 1)) from the FROZEN values. Mutations caught (all left the P2
    suite green): hard 0.70 -> 0.5; R03 rate -> 0.02; beta 0.5 -> 1.5; unattributed-only not hard;
    sensitivity 0.85 -> 0.97."""
    ids, keys, per, rules_, nr, tp, typ, cell_of = [], [], [], [], [], [], [], []
    for cell, (is_tp, ty, extra, _) in CELLS.items():
        for n in (1, 2, 3):
            for i in range(N_PER_CELL):
                base = ["R04_FAN_IN", "R02_PEER_VOLUME_OUTLIER", "R05_FAN_OUT"][: n - bool(extra)]
                ids.append(f"A|{cell}|{n}|{i}")
                keys.append(f"001|K{i}")
                per.append("TRAIN")
                rules_.append(base + ([extra] if extra else []))
                nr.append(n)
                tp.append(is_tp)
                typ.append(ty)
                cell_of.append((cell, n))
    alerts = pl.DataFrame(
        {
            "alert_id": ids,
            "account_key": keys,
            "period": per,
            "as_of": [D(5)] * len(ids),
            "triggered_rules": rules_,
            "n_rules_triggered": nr,
        }
    )
    labels = pl.DataFrame({"alert_id": ids, "is_true_positive": tp, "typologies": typ})
    d, _ = disp.simulate(alerts, labels, DCFG)
    conf = dict(zip(d["alert_id"], d["disposition"] == "confirmed_suspicious", strict=True))
    for cell, (_, _, _, p0) in CELLS.items():
        for n in (1, 2, 3):
            got = np.mean([conf[f"A|{cell}|{n}|{i}"] for i in range(N_PER_CELL)])
            want = _expected(p0, n)
            se = np.sqrt(want * (1 - want) / N_PER_CELL)
            assert abs(got - want) <= 4.5 * se, (cell, n, got, want)


def test_disposition_delays_realise_the_frozen_lognormal():
    n = 20000
    alerts = pl.DataFrame(
        {
            "alert_id": [f"A{i}" for i in range(n)],
            "account_key": ["001|K"] * n,
            "period": ["CALIBRATION"] * n,
            "as_of": [D(10)] * n,
            "triggered_rules": [["R04_FAN_IN"]] * n,
            "n_rules_triggered": [1] * n,
        }
    )
    labels = pl.DataFrame(
        {"alert_id": alerts["alert_id"], "is_true_positive": [i % 2 == 0 for i in range(n)],
         "typologies": [["FAN-IN"] if i % 2 == 0 else [] for i in range(n)]}
    )  # fmt: skip
    d, _ = disp.simulate(alerts, labels, DCFG)
    h = (d["closed_at"] - D(10)).dt.total_minutes().to_numpy() / 60
    c = (d["disposition"] == "confirmed_suspicious").to_numpy()
    assert np.median(h[c]) == pytest.approx(FROZEN["median_confirmed"], rel=0.04)
    assert np.median(h[~c]) == pytest.approx(FROZEN["median_closed"], rel=0.04)
    s = np.std(np.log(h[~c][(h[~c] > 2) & (h[~c] < 160)]))
    assert s == pytest.approx(FROZEN["sigma"], rel=0.1)
    assert h.min() >= FROZEN["clip"][0] and h.max() <= FROZEN["clip"][1]


# ---------------------------------------------------------------- M-8 UTC end to end (D3)


def _mini_raw(raw: Path) -> None:
    """Transactions at 23:59 and 00:00 around day boundaries; amounts identify each row."""
    a, b, c, e = fxt.L(0), fxt.L(1), fxt.N(0), fxt.N(1)
    rows = [
        fxt._usd("2022/09/01 12:00", c, e, 10.0),  # burn-in
        fxt._usd("2022/09/12 23:59", a, b, 111.0, "ACH", 1),  # laundering, last minute of 09-12
        fxt._usd("2022/09/13 00:00", a, b, 222.0, "ACH", 1),  # laundering, first minute of 09-13
        fxt._usd("2022/09/12 23:59", c, e, 333.0),
        fxt._usd("2022/09/13 00:00", c, e, 444.0),
        fxt._usd("2022/09/16 23:59", a, b, 555.0, "ACH", 1),  # last usable minute (D1)
        fxt._usd("2022/09/17 00:00", a, b, 666.0, "ACH", 1),  # first tail minute: excluded
    ]
    raw.mkdir(parents=True)
    (raw / "FIX_Trans.csv").write_text(fxt.HEADER + "\n" + "\n".join(rows) + "\n", "utf-8")
    pat = ["BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 1-degree Fan-Out", rows[1],
           "END LAUNDERING ATTEMPT - FAN-OUT", ""]  # fmt: skip
    (raw / "FIX_Patterns.txt").write_text("\n".join(pat) + "\n", encoding="utf-8")
    names = dict(fxt.BANKS)
    acc = [
        f"{names[bk]},{bk.lstrip('0') or '0'},{ac},E{j:05d},Corporation #{j}"
        for j, (bk, ac) in enumerate([a, b, c, e])
    ]
    (raw / "FIX_accounts.csv").write_text(fxt.ACC_HEADER + "\n" + "\n".join(acc) + "\n", "utf-8")


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("utc")
    _mini_raw(tmp / "raw")
    fxt.convert(tmp / "raw", tmp / "interim", tmp / "sources.yaml")
    return tmp / "interim"


def test_loader_keeps_wall_clock_as_utc(mini):
    t = scan_transactions(mini / "FIX_trans.parquet").collect()
    assert t["timestamp"].dtype == pl.Datetime("us", "UTC")
    got = {(r["amount_paid"], r["timestamp"]) for r in t.iter_rows(named=True)}
    assert (111.0, D(12, 23, 59)) in got and (222.0, D(13)) in got
    assert (555.0, D(16, 23, 59)) in got and (666.0, D(17)) in got


def test_rules_put_23_59_and_00_00_in_different_days(mini):
    """Mutation caught: transactions.py localising to Europe/Lisbon then converting to UTC
    (a 1 h shift moves the 00:00 row into the previous day)."""
    split, rcfg = load_split(), rules.load_rules()
    tu = fx.with_usd(scan_transactions(mini / "FIX_trans.parquet"), {"US Dollar": 1.0})
    c = fxt.key(fxt.N(0))
    d12 = rules.day_stats(tu, D(12), split, rcfg).filter(pl.col("account_key") == c)
    d13 = rules.day_stats(tu, D(13), split, rcfg).filter(pl.col("account_key") == c)
    assert d12["max_txn_usd"].to_list() == [333.0] and d12["n_txn"].to_list() == [1]
    assert d13["max_txn_usd"].to_list() == [444.0] and d13["n_txn"].to_list() == [1]


def test_labels_put_23_59_and_00_00_in_different_days_and_drop_the_tail(mini):
    """Mutation caught: alert_labels.py shifting the label timestamps by 1 h (rules and labels
    would then disagree on the day)."""
    from src.eval.alert_labels import laundering_legs

    legs = laundering_legs(mini / "FIX_trans.parquet", mini / "FIX_patterns.parquet", D(1), D(17))
    a = fxt.key(fxt.L(0))
    days = sorted(legs.filter(pl.col("account_key") == a)["window_start"].to_list())
    assert days == [D(12), D(13), D(16)]  # 23:59 -> 09-12, 00:00 -> 09-13, tail 09-17 excluded


# ---------------------------------------------------------------- M-9 FX on TRAIN only


def _fx_rows(rows):
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
        )  # fmt: skip
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        .lazy()
    )


def test_fx_ignores_a_majority_of_rows_outside_the_window():
    """More rows OUTSIDE the window (before and after, different rate) than inside: the fitted
    rate must still be the inside one. Mutation caught: fx.derive_usd_per_unit ignoring the
    window (the P2 test had 1 outside row against 50 inside, which cannot move a median)."""
    rows = []
    for i in range(30):
        rows.append((D(3) + timedelta(minutes=i), "Euro", "US Dollar", 100.0 + i, (100 + i) * 1.1))
    for i in range(100):
        for day in (1, 6, 7, 13, 16):
            rows.append((D(day) + timedelta(minutes=i), "Euro", "US Dollar", 50.0, 50.0 * 1.3))
    r = fx.derive_usd_per_unit(_fx_rows(rows), D(2), D(6))
    assert r["usd_per_unit"]["Euro"] == pytest.approx(1.1, rel=1e-9)
    assert r["fitted_on"] == {"start": D(2).isoformat(), "end_exclusive": D(6).isoformat()}
    with pytest.raises(AssertionError):  # the control: fitting on the span gives the wrong rate
        r2 = fx.derive_usd_per_unit(_fx_rows(rows), D(1), D(17))
        assert r2["usd_per_unit"]["Euro"] == pytest.approx(1.1, rel=1e-9)


def test_pipeline_fits_fx_on_exactly_the_train_period(p2_run):
    """Mutation caught: run_p2.stage_stats fitting on split.span instead of TRAIN."""
    split = load_split()
    a, b = split.periods["TRAIN"]
    got = yaml.safe_load((p2_run["configs"] / "p2_fx_usd_per_unit.yaml").read_text("utf-8"))
    assert got["fitted_on"] == {"start": a.isoformat(), "end_exclusive": b.isoformat()}
    assert p2_run["doc"]["stats"]["fx"]["fitted_on"] == got["fitted_on"]
    real = yaml.safe_load((ROOT / "configs" / "p2_fx_usd_per_unit.yaml").read_text("utf-8"))
    assert real["fitted_on"] == got["fitted_on"]  # the committed table too


# ---------------------------------------------------------------- m-6 verify, m-9 cache manifest


def test_cache_manifest_matches_and_stale_cache_is_refused(p2_run, tmp_path):
    from scripts.p2 import run_p2
    from src.data import kyc as kycmod

    split, rcfg = load_split(p2_run["cp"].split), rules.load_rules(p2_run["cp"].rules)
    kcfg = kycmod.load_kyc_config()
    P = run_p2.Paths(p2_run["root"] / "out", "FIX", p2_run["configs"])
    key = run_p2.cache_key(split, rcfg, P.fx, p2_run["cp"].sources, "FIX", kcfg)
    assert run_p2.check_cache(P, key) == "match"
    # a rule parameter change makes the cached statistics stale -> loud failure
    rcfg2 = yaml.safe_load(yaml.safe_dump(rcfg))
    rcfg2["rules"]["R03_STRUCTURING"]["band_usd"]["lo"] = 8000
    with pytest.raises(run_p2.StaleCacheError):
        run_p2.check_cache(
            P, run_p2.cache_key(split, rcfg2, P.fx, p2_run["cp"].sources, "FIX", kcfg)
        )
    # an interrupted stats run leaves an incomplete manifest -> refused
    R = run_p2.Paths(tmp_path / "r", "FIX", p2_run["configs"])
    R.work.mkdir(parents=True)
    R.cache_manifest.write_text(json.dumps({"cache_key": key, "complete": False}), "utf-8")
    with pytest.raises(run_p2.StaleCacheError, match="incomplete"):
        run_p2.check_cache(R, key)
    # a cache from before the manifest existed is recorded, not silently trusted forever
    Q = run_p2.Paths(tmp_path, "FIX", p2_run["configs"])
    Q.stats_dir.mkdir(parents=True)
    assert run_p2.check_cache(Q, key) == "created"
    assert run_p2.check_cache(Q, key) == "match"


def test_verify_build_and_all_reproduce_the_fixture(p2_run, tmp_path):
    """Second-pass review M1: `build --verify` used to fail on keys a build never writes. Run the
    real verify() for both stages on a copy of the fixture: no differences."""
    from scripts.p2 import run_p2

    dst = tmp_path / "copy"
    shutil.copytree(p2_run["root"] / "out", dst / "out")
    shutil.copytree(p2_run["root"] / "configs", dst / "configs")
    shutil.copyfile(p2_run["root"] / "results.json", dst / "results.json")
    for stage in ("build", "all"):
        diffs, _ = run_p2.verify(
            stage, "FIX", p2_run["info"]["interim"], dst / "out", dst / "configs",
            dst / "results.json", 20260930, tmp_path / f"v_{stage}", cp=p2_run["cp"],
            primary="FIX",
        )  # fmt: skip
        assert diffs == [], (stage, diffs[:5])
    # and a tampered store is caught
    k = dst / "out" / "FIX" / "runtime" / "kyc.parquet"
    pl.read_parquet(k).head(5).write_parquet(k)
    diffs, _ = run_p2.verify(
        "build", "FIX", p2_run["info"]["interim"], dst / "out", dst / "configs",
        dst / "results.json", 20260930, tmp_path / "v_bad", cp=p2_run["cp"], primary="FIX",
    )  # fmt: skip
    assert any("kyc.parquet" in d for d in diffs)


def test_verify_compares_results_json_and_fingerprints(p2_run, p2_run_again, tmp_path):
    import json

    from scripts.p2 import run_p2

    a, b = p2_run["doc"], p2_run_again["doc"]
    assert run_p2.compare_results(a, b) == []
    c = json.loads(json.dumps(b))
    c["build"]["dispositions"]["overall"]["accuracy"] = 0.5
    c["run_info"]["elapsed_s"] = 999  # ignored
    diffs = run_p2.compare_results(a, c)
    assert len(diffs) == 1 and "dispositions.overall.accuracy" in diffs[0]
    P = run_p2.Paths(p2_run["root"] / "out", "FIX", p2_run["configs"])
    assert run_p2.check_fingerprints(a, P) == []
    d = json.loads(json.dumps(a))
    d["stores"]["runtime/kyc.parquet"]["sha256"] = "0" * 64
    assert run_p2.check_fingerprints(d, P) == [
        "store runtime/kyc.parquet: results JSON fingerprint does not match the file on disk"
    ]


def test_exposure_log_is_well_formed():
    """docs/holdout_exposure_log.md (review M-5): numbered rows in order, every field filled,
    times non-decreasing (PENDING allowed only for the last rows)."""
    text = (ROOT / "docs" / "holdout_exposure_log.md").read_text(encoding="utf-8")
    rows = [
        [c.strip() for c in ln.strip().strip("|").split("|")]
        for ln in text.splitlines()
        if ln.startswith("| ") and ln[2:3].isdigit()
    ]
    assert len(rows) >= 4
    assert [int(r[0]) for r in rows] == list(range(1, len(rows) + 1))
    assert all(len(r) == 7 and all(r) for r in rows)
    times = [r[1] for r in rows]
    done = [t for t in times if t != "PENDING"]
    assert times[: len(done)] == done, "PENDING rows must come last"
    assert done == sorted(done)
