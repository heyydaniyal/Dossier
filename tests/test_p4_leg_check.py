"""P4 step 0b (per-payment signal check): payment features are point-in-time and correct by known
answers; the payment-label loader is TRAIN-scoped and refuses TEST; sampling is deterministic and
re-weighted; the verdict separates a world WITH per-payment signal from one WITHOUT; the driver
runs end to end on the P2 fixture, reads TRAIN only and writes aggregates only."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression

from src.data import fx as fxmod
from src.data.periods import load_split
from src.data.transactions import interim_path, scan_transactions
from src.eval import training_labels as tl
from src.features import compute as fc
from src.features import legs as fl
from src.features.pit import scramble_outside_window
from src.models import leg_check as lc

UTC_ = UTC


def T(h: int, m: int = 0, day: int = 3) -> datetime:
    return datetime(2022, 9, day, h, m, tzinfo=UTC)


def _frame(rows: list[tuple]) -> pl.LazyFrame:
    """rows: (ts, from_bank, from_acct, to_bank, to_acct, usd, fmt, paid_ccy, recv_ccy, amount)."""
    recs = []
    for ts, fb, fa, tb, ta, usd, fmt, pc, rc, amt in rows:
        recs.append({
            "timestamp": ts, "from_bank": fb, "from_account": fa, "to_bank": tb, "to_account": ta,
            "amount_received": amt, "receiving_currency": rc, "amount_paid": amt,
            "payment_currency": pc, "payment_format": fmt,
            "from_key": f"{fb}|{fa}", "to_key": f"{tb}|{ta}", "usd_paid": usd, "usd_received": usd,
        })  # fmt: skip
    return pl.DataFrame(recs).lazy()


@pytest.fixture(scope="module")
def ctx():
    return fc.default_context()


def test_known_answers(ctx):
    """Hub H receives from A (x2, same amount), B, C; pays D. B also pays C and E. Self-transfer
    and rows outside the window are ignored."""
    rows = [
        (T(1), "1", "A", "1", "H", 1000.0, "ACH", "US Dollar", "US Dollar", 1000.0),
        (T(2), "1", "A", "1", "H", 1000.0, "ACH", "US Dollar", "US Dollar", 1000.0),
        (T(3), "2", "B", "1", "H", 9500.0, "Wire", "US Dollar", "US Dollar", 9500.0),
        (T(4), "3", "C", "1", "H", 250.0, "Cash", "US Dollar", "US Dollar", 250.0),
        (T(5), "1", "H", "4", "D", 5000.0, "ACH", "US Dollar", "Euro", 5000.0),
        (T(6), "2", "B", "3", "C", 10.0, "Cheque", "US Dollar", "US Dollar", 10.0),
        (T(7), "2", "B", "5", "E", 10.0, "Cheque", "US Dollar", "US Dollar", 10.0),
        (T(8), "1", "H", "1", "H", 777.0, "Reinvestment", "US Dollar", "US Dollar", 777.0),
        (T(9, day=2), "9", "Z", "1", "H", 1.0, "ACH", "US Dollar", "US Dollar", 1.0),  # before
        (T(0, day=4), "9", "Z", "1", "H", 1.0, "ACH", "US Dollar", "US Dollar", 1.0),  # = as_of
    ]
    as_of = T(0, day=4)
    al = pl.DataFrame({"alert_id": ["AH"], "account_key": ["1|H"], "as_of": [as_of]})
    out = fl.legs_for_window(_frame(rows), as_of, al, ctx)
    assert out.height == 5  # 4 in-legs + 1 out-leg; self-transfer and outside rows excluded
    by = {
        (r["from_account"], r["to_account"], r["timestamp"]): r for r in out.iter_rows(named=True)
    }
    a1 = by[("A", "H", T(1))]
    b = by[("B", "H", T(3))]
    d = by[("H", "D", T(5))]
    assert a1["is_out"] == 0 and d["is_out"] == 1
    assert a1["log10_pair_n_txn"] == pytest.approx(np.log10(2))  # A paid H twice
    assert a1["log10_same_amount_n"] == pytest.approx(np.log10(2))  # 1000 USD appears twice
    assert b["log10_same_amount_n"] == pytest.approx(0.0)
    assert b["near_threshold"] == 1 and a1["near_threshold"] == 0  # 9,500 in [9,000, 10,000)
    assert a1["round_amount"] == 1 and by[("C", "H", T(4))]["round_amount"] == 0
    assert b["cross_bank"] == 1 and a1["cross_bank"] == 0
    assert d["cross_currency"] == 1 and a1["cross_currency"] == 0
    assert b["log10_cp_out_deg"] == pytest.approx(np.log10(3 + 1))  # B pays H, C, E
    assert b["log10_cp_n_legs"] == pytest.approx(np.log10(3))
    assert a1["log10_cp_out_deg"] == pytest.approx(np.log10(1 + 1))
    assert d["log10_cp_in_deg"] == pytest.approx(np.log10(1 + 1))
    usd = [1000, 1000, 9500, 250, 5000]
    assert b["share_of_account_volume"] == pytest.approx(9500 / sum(usd))
    assert b["log10_ratio_to_account_median"] == pytest.approx(np.log10(9500 / 1000))
    assert a1["log10_account_n_legs"] == pytest.approx(np.log10(5))
    assert b["leg_fmt_wire"] == 1 and b["leg_fmt_ach"] == 0
    assert set(fl.leg_feature_names(ctx)) <= set(out.columns)


def test_no_wall_clock_or_identifier_features(ctx):
    names = fl.leg_feature_names(ctx)
    for bad in ("hour", "weekday", "minute", "timestamp", "_id", "key"):  # P3-7, D5
        assert not any(bad in n for n in names), bad
    ids = {"from_bank", "from_account", "to_bank", "to_account", "from_key", "to_key",
           "account_key", "counterparty", "alert_id"}  # fmt: skip
    assert set(names).isdisjoint(ids | set(fl.KEY_COLUMNS))


@pytest.fixture(scope="module")
def fix_trans(p2_run):
    raw = scan_transactions(interim_path(p2_run["info"]["interim"], "FIX", "trans"))
    fx = fxmod.load(p2_run["configs"] / "p2_fx_usd_per_unit.yaml")
    alerts = pl.read_parquet(p2_run["base"] / "runtime" / "alerts.parquet")
    return raw, fx, alerts


def _legs_on(raw, fx, alerts, ctx, as_of):
    a = alerts.filter(pl.col("as_of") == as_of).select("alert_id", "account_key", "as_of")
    return fl.legs_for_window(fxmod.with_usd(raw, fx), as_of, a, ctx).sort(
        "alert_id", *fl.KEY_COLUMNS
    )


def test_point_in_time_deletion_and_outside_mutation(fix_trans, ctx):
    raw, fx, alerts = fix_trans
    names = fl.leg_feature_names(ctx)
    for as_of in sorted(alerts["as_of"].unique().to_list())[:4]:
        full = _legs_on(raw, fx, alerts, ctx, as_of)
        past = _legs_on(raw.filter(pl.col("timestamp") < as_of), fx, alerts, ctx, as_of)
        mut = _legs_on(scramble_outside_window(raw, as_of, ctx.lookback), fx, alerts, ctx, as_of)
        for other in (past, mut):
            assert other.select(names).equals(full.select(names)), as_of


def test_inside_window_mutation_does_change_features(fix_trans, ctx):
    raw, fx, alerts = fix_trans
    as_of = sorted(alerts["as_of"].unique().to_list())[0]
    full = _legs_on(raw, fx, alerts, ctx, as_of)
    inside = (pl.col("timestamp") < as_of) & (pl.col("timestamp") >= as_of - timedelta(hours=24))
    mut = raw.with_columns(
        pl.when(inside).then(pl.col("amount_paid") * 3.1).otherwise(pl.col("amount_paid"))
        .alias("amount_paid")
    )  # fmt: skip
    changed = _legs_on(mut, fx, alerts, ctx, as_of)
    assert not changed.select("log10_usd").equals(full.select("log10_usd"))


# ------------------------------------------------------------------ payment-label loader


def test_laundering_txns_refuse_test_before_opening(tmp_path):
    with pytest.raises(tl.HoldoutAccessError):
        tl.load_fit_laundering_txns(["TRAIN", "TEST"], "NOPE", tmp_path, load_split())


def test_laundering_txns_are_train_scoped_and_exact(p2_run):
    split = load_split()
    interim = p2_run["info"]["interim"]
    got = tl.load_fit_laundering_txns(["TRAIN"], "FIX", interim, split)
    a, b = split.periods["TRAIN"]
    truth = (
        pl.read_parquet(interim / "FIX_trans.parquet")
        .with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
        .filter(
            (pl.col("is_laundering") == 1) & (pl.col("timestamp") >= a) & (pl.col("timestamp") < b)
        )  # fmt: skip
    )
    assert got.height == truth.height > 0
    assert got["timestamp"].min() >= a and got["timestamp"].max() < b
    assert "is_laundering" not in got.columns
    # fixture truth: FAN-IN 09-03 (8 txns) + unattributed 09-04 (2 txns)
    assert got.height == 10


# ------------------------------------------------------------------ sampling


def test_sampling_keeps_every_laundering_leg_caps_normals_and_reweights():
    from scripts.p4.leg_check import sample_fit_legs

    rng = np.random.default_rng(0)
    n = 900
    df = pl.DataFrame({
        "alert_id": ["A"] * 600 + ["B"] * 300,
        **{c: [f"{c}{i}" for i in range(n)] for c in fl.KEY_COLUMNS},
        "y_leg": (rng.random(n) < 0.05).astype(np.int8),
    })  # fmt: skip
    s1 = sample_fit_legs(df, 100, "k", 1)
    s2 = sample_fit_legs(df.sample(fraction=1.0, shuffle=True, seed=3), 100, "k", 1)
    assert s1.equals(s2)  # input order does not matter
    assert int(s1["y_leg"].sum()) == int(df["y_leg"].sum())
    for a in ("A", "B"):
        sub = s1.filter(pl.col("alert_id") == a)
        n_norm = int((df.filter(pl.col("alert_id") == a)["y_leg"] == 0).sum())
        assert int((sub["y_leg"] == 0).sum()) == min(100, n_norm)
        assert sub.filter(pl.col("y_leg") == 0)["w"].sum() == pytest.approx(n_norm)


# ------------------------------------------------------------------ verdict on simulated worlds


def _world(signal: bool, seed: int):
    """Unseen alerts; each has 5-40 payments. Positives hold 1-3 laundering payments. With signal,
    a laundering payment's first feature is shifted (+2.5 SD); without, it is drawn like the rest.
    The baseline alert score is weak (AUC ~0.56) in both worlds."""
    rng = np.random.default_rng(seed)

    def day(n_alerts):
        y = (rng.random(n_alerts) < 0.07).astype(int)
        X, legal, ylg = [], [], []
        for i in range(n_alerts):
            k = int(rng.integers(5, 40))
            yl = np.zeros(k, int)
            if y[i]:
                yl[rng.choice(k, int(rng.integers(1, 4)), replace=False)] = 1
            x = rng.normal(size=(k, 4))
            if signal:
                x[yl == 1, 0] += 2.5
            X.append(x)
            legal += [i] * k
            ylg.append(yl)
        return y, np.vstack(X), np.array(legal), np.concatenate(ylg)

    yf_a, Xf, _, yf = day(1500)
    ya, Xe, la, yle = day(1500)
    base = ya * 0.25 + rng.normal(size=len(ya))
    return lc.LegFold("S", Xf, yf, np.ones(len(yf)), Xe, la, yle, ya,
                      np.zeros(len(ya), bool), base)  # fmt: skip


def _lr():
    return LogisticRegression(max_iter=500)


@pytest.mark.parametrize(("signal", "want"), [(True, "signal"), (False, "no_signal")])
def test_verdict_separates_worlds(signal, want):
    f = _world(signal, 1 if signal else 2)
    s = {"S": lc.max_by_alert(lc.leg_scores(f, _lr), f.leg_alert, len(f.y_alert))}
    r = lc.compare([f], s, n_boot=300, seed=1, ci=0.95, material=0.03, min_pos=30)
    assert r["verdict"] == want, r
    w = lc.within_alert([f], {"S": lc.leg_scores(f, _lr)})
    if signal:
        assert w["top_payment_hit_rate"] > 2 * w["chance_rate"]
    else:
        assert abs(w["mean_within_alert_auc"] - 0.5) < 0.06


def test_only_unseen_alerts_are_compared():
    """Seen alerts with perfect payment scores must not lift the unseen comparison."""
    f = _world(False, 4)
    rng = np.random.default_rng(4)
    seen = rng.random(len(f.y_alert)) < 0.5
    f2 = lc.LegFold("S", f.X_fit, f.y_fit, f.w_fit, f.X_eval, f.leg_alert, f.y_leg_eval,
                    f.y_alert, seen, f.baseline)  # fmt: skip
    s = lc.max_by_alert(lc.leg_scores(f2, _lr), f2.leg_alert, len(f2.y_alert))
    s = np.where(seen, f2.y_alert * 10.0, s)  # seen alerts ranked perfectly
    r = lc.compare([f2], {"S": s}, n_boot=200, seed=1, ci=0.95, material=0.03, min_pos=10)
    assert r["n_unseen_pos"] == int(f2.y_alert[~seen].sum())
    u = ~seen
    assert r["auc_payment_max"] == pytest.approx(lc.dg.auc(f2.y_alert[u], s[u]))
    assert r["verdict"] != "signal"  # perfect SEEN ranking must not create a signal
    w = lc.within_alert([f2], {"S": lc.leg_scores(f2, _lr)})
    norm = np.bincount(f2.leg_alert, weights=1 - f2.y_leg_eval, minlength=len(f2.y_alert))
    assert w["n_alerts"] == int(((f2.y_alert == 1) & ~seen & (norm > 0)).sum())


def test_decide_table():
    assert lc.decide([0.03, 0.10], 0.03) == "signal"
    assert lc.decide([0.029, 0.10], 0.03) == "inconclusive"
    assert lc.decide([-0.05, 0.029], 0.03) == "no_signal"
    assert lc.decide(None, 0.03) == "insufficient_data"


def test_fold_refuses_alert_labels_that_disagree_with_payment_labels():
    f = _world(True, 3)
    bad = f.y_alert.copy()
    bad[np.flatnonzero(bad == 1)[0]] = 0
    with pytest.raises(ValueError, match="disagree"):
        lc.LegFold("S", f.X_fit, f.y_fit, f.w_fit, f.X_eval, f.leg_alert, f.y_leg_eval, bad,
                   f.seen, f.baseline)  # fmt: skip


def test_max_by_alert():
    s = np.array([0.1, 0.9, 0.3, 0.2])
    assert lc.max_by_alert(s, np.array([0, 0, 1, 1]), 2).tolist() == [0.9, 0.3]


# ------------------------------------------------------------------ driver on the P2 fixture


@pytest.fixture(scope="module")
def p4b(p2_run, tmp_path_factory):
    from scripts.p3 import run_p3
    from scripts.p4 import diagnose as d0
    from scripts.p4 import leg_check as drv

    r = p2_run["root"]
    out = tmp_path_factory.mktemp("p4b")
    common = [
        "--variant", "FIX", "--store-root", str(r / "out"), "--rules", str(p2_run["cp"].rules),
        "--thresholds", str(p2_run["configs"] / "p2_rule_thresholds.yaml"),
    ]  # fmt: skip
    run_p3.main([
        "features", *common, "--interim", str(p2_run["info"]["interim"]),
        "--out-root", str(out / "p3"), "--results", str(out / "p3.json"),
        "--fx", str(p2_run["configs"] / "p2_fx_usd_per_unit.yaml"),
        "--sources", str(p2_run["info"]["sources"]), "--p2-results", str(r / "results.json"),
    ])  # fmt: skip
    import yaml

    cfg = yaml.safe_load(drv.CONFIG.read_text(encoding="utf-8"))
    cfg["comparison"]["n_boot"] = 20
    cfg["min_pos_unseen"] = 1
    cfg["model"]["params"] |= {"n_estimators": 20, "min_child_samples": 2}
    (out / "leg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    dc = d0.load_config()
    dc["seeds"] = [1]
    dc["strata"]["min_stratum_pos"] = 1
    (out / "diag.yaml").write_text(yaml.safe_dump(dc), encoding="utf-8")
    mp = pytest.MonkeyPatch()
    folds = [
        {"name": "F1", "fit_days": ["2022-09-02", "2022-09-03"], "eval_day": "2022-09-04"},
        {"name": "F2", "fit_days": ["2022-09-02", "2022-09-03", "2022-09-04"],
         "eval_day": "2022-09-05"},
    ]  # fmt: skip
    mp.setattr(drv, "load_v2_config", lambda: {"folds": folds})
    asked = []
    real = drv.load_fit_laundering_txns

    def spy(periods, *a, **k):
        asked.append(list(periods))
        return real(periods, *a, **k)

    mp.setattr(drv, "load_fit_laundering_txns", spy)
    args = [
        *common, "--interim", str(p2_run["info"]["interim"]),
        "--config", str(out / "leg.yaml"), "--diag-config", str(out / "diag.yaml"),
        "--features-root", str(out / "p3"), "--p3-results", str(out / "p3.json"),
        "--results", str(out / "leg.json"),
        "--fx", str(p2_run["configs"] / "p2_fx_usd_per_unit.yaml"),
        "--sources", str(p2_run["info"]["sources"]),
    ]  # fmt: skip
    yield {"drv": drv, "args": args, "out": out, "asked": asked}
    mp.undo()


def test_driver_end_to_end_on_fixture(p4b):
    """The run completes, so every eval alert's label equals the max of its payment labels
    (LegFold refuses otherwise); payment labels are requested for TRAIN only."""
    assert p4b["drv"].main(p4b["args"]) == 0
    doc = json.loads((p4b["out"] / "leg.json").read_text(encoding="utf-8"))
    assert p4b["asked"] == [["TRAIN"]]
    assert doc["inputs"]["periods_read"] == ["TRAIN"]
    assert doc["primary"]["verdict"] in {"signal", "no_signal", "inconclusive", "insufficient_data"}
    assert [c["fold"] for c in doc["counts"]] == ["F1", "F2"]
    assert sum(c["fit_positive_payments"] for c in doc["counts"]) > 0


def test_driver_output_holds_aggregates_only(p4b, p2_run):
    text = (p4b["out"] / "leg.json").read_text(encoding="utf-8")
    alerts = pl.read_parquet(p2_run["base"] / "runtime" / "alerts.parquet")
    for col in ("account_key", "alert_id"):
        for v in alerts[col].unique().to_list():
            assert v not in text, (col, v)
    for v in alerts["account_key"].str.split("|").list.get(1).unique().to_list():
        assert f'"{v}"' not in text
