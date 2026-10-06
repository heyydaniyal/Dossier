"""P3 features and leakage suite on the P2 fixture (real schema, real P2 pipeline, known data).

Leakage suite required by the phase, covered here:
  - deletion-recompute: features(full data) == features(data before as_of), every feature type
    (scalar, rolling, peer, entity aggregate, graph); KYC-derived features do not exist (frozen);
    the anomaly score does not exist yet (P5 adds it to this test)
  - lookback: mutating every row OUTSIDE [as_of - 24 h, as_of) changes nothing (so no feature
    looks back further than L_max), and mutating a row INSIDE does change features (control)
  - pure per-account path == vectorised batch path
  - registry: names == pipeline columns; lookbacks <= L_max; inputs label-free; no KYC/ID inputs
  - split: max(as_of) + embargo <= min(as_of) of the next period; no feature window crosses
  - graph: every edge used has first_ts >= as_of - L_max and last_ts < as_of; known-answer graph
"""

from __future__ import annotations

import ast
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from src.alerts import rules
from src.data.fx import load as load_fx
from src.data.fx import with_usd
from src.data.periods import load_split
from src.data.transactions import interim_path, scan_transactions
from src.features import compute as fc
from src.features import registry as reg
from src.features import splits
from src.features.network import flagged_counterparties
from src.features.pit import scramble_outside_window

ROOT = Path(__file__).resolve().parents[1]
D = lambda day, h=0, m=0: datetime(2022, 9, day, h, m, tzinfo=UTC)  # noqa: E731


# ---------------------------------------------------------------- helpers


@pytest.fixture(scope="module")
def env(p2_run):
    split = load_split(p2_run["cp"].split)
    rcfg = yaml.safe_load(Path(p2_run["cp"].rules).read_text(encoding="utf-8"))
    thr = rules.load_thresholds(p2_run["configs"] / "p2_rule_thresholds.yaml")
    ctx = fc.make_context(rcfg, thr, fc.load_features_config(), split)
    fx = load_fx(p2_run["configs"] / "p2_fx_usd_per_unit.yaml")
    raw = scan_transactions(interim_path(p2_run["info"]["interim"], "FIX", "trans"))
    alerts = pl.read_parquet(p2_run["base"] / "runtime" / "alerts.parquet")
    feats = fc.batch_features(with_usd(raw, fx), alerts, ctx)
    return {"split": split, "ctx": ctx, "fx": fx, "raw": raw, "alerts": alerts, "feats": feats}


def _usd(lf, env):
    return with_usd(lf, env["fx"])


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _assert_frames_equal(a: pl.DataFrame, b: pl.DataFrame, names: list[str]) -> None:
    assert a["alert_id"].to_list() == b["alert_id"].to_list()
    for c in names:
        for x, y, aid in zip(a[c].to_list(), b[c].to_list(), a["alert_id"].to_list(), strict=True):
            assert _close(x, y), f"{c} differs for {aid}: {x} vs {y}"


# ---------------------------------------------------------------- registry


def test_registry_matches_pipeline_columns(env):
    names = fc.feature_names(env["ctx"])
    assert [f.name for f in reg.build_registry(env["ctx"])] == names
    assert env["feats"].columns == [*fc.ALERT_KEYS, *names, "_triggered"]
    assert len(set(names)) == len(names)


def test_registry_lookbacks_inputs_and_names(env):
    lmax_h = env["split"].l_max.total_seconds() / 3600
    banned = ("label", "laundering", "typolog", "pattern", "attempt", "planted", "disposition",
              "kyc", "sector", "entity", "bank", "account_id", "row_id", "alert_id", "period",
              "peer_group", "onboarding", "risk_rating", "country", "weekday",
              "hour_of_day")  # fmt: skip
    for f in reg.build_registry(env["ctx"]):
        assert f.lookback_hours <= lmax_h, f.name
        assert set(f.inputs) <= reg.ALLOWED_INPUTS, f.name
        assert "dispositions_visible" not in f.inputs, f"{f.name}: flags are tool-only (MDP)"
        assert not any(b in f.name.lower() for b in banned), f.name
        assert f.as_of_tests and f.owner and f.definition, f.name
        for t in f.as_of_tests:
            assert t in globals(), f"{f.name} cites a test that does not exist: {t}"
    for f in reg.TOOL_ONLY.values():
        assert not f.model and f.name not in fc.feature_names(env["ctx"])
        for t in f.as_of_tests:
            assert t in globals()
    assert len(reg.registry_hash(env["ctx"])) == 64


def test_feature_code_never_touches_kyc_labels_or_eval():
    """Registry check, by source: src/features imports no KYC module, names no KYC/eval store,
    and never selects the label column (the isolation scan covers eval imports too)."""
    for py in (ROOT / "src" / "features").glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom | ast.Import):
                mods = [node.module or ""] if isinstance(node, ast.ImportFrom) else [
                    a.name for a in node.names]  # fmt: skip
                for m in mods:
                    banned_mod = m.startswith(("src.data.kyc", "src.data.accounts", "src.eval"))
                    assert not banned_mod, f"{py.name} imports {m}"
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "kyc.parquet" not in node.value, py.name


def test_lookback_longer_than_lmax_is_refused(env):
    cfg = fc.load_features_config() | {"lookback_hours": 25}
    with pytest.raises(fc.FeatureError, match="exceeds the frozen L_max"):
        fc.make_context(env["ctx"].rules_cfg, env["ctx"].thresholds, cfg, env["split"])


# ---------------------------------------------------------------- integrity vs P2


def test_recomputed_rule_features_equal_the_alert_store(env):
    """The rule group is recomputed from transactions, not copied: it must reproduce every stored
    P2 statistic and the stored triggered rules exactly."""
    j = env["feats"].join(env["alerts"], on="alert_id", suffix="_p2")
    assert j.height == env["alerts"].height
    for c in rules.ALL_STATS:
        for a, b in zip(j[c].to_list(), j[f"{c}_p2"].to_list(), strict=True):
            assert _close(a, float(b)), c
    assert j["_triggered"].to_list() == j["triggered_rules"].to_list()
    assert (j["n_rules_triggered"] == j["n_rules_triggered_p2"].cast(pl.Float64)).all()


# ---------------------------------------------------------------- deletion / mutation / pure


def test_deletion_recompute_batch(env):
    """features(full data) == features(only rows with ts < as_of), alert by alert."""
    names = fc.feature_names(env["ctx"])
    parts = []
    for as_of in sorted(env["alerts"]["as_of"].unique().to_list()):
        past = env["raw"].filter(pl.col("timestamp") < as_of)
        a = env["alerts"].filter(pl.col("as_of") == as_of)
        parts.append(fc.features_for_window(_usd(past, env), as_of, a, env["ctx"]))
    _assert_frames_equal(env["feats"], pl.concat(parts).sort("alert_id"), names)


def _with_edge_rows(lf: pl.LazyFrame, keys: list[str], as_of, lookback) -> pl.LazyFrame:
    """Append transactions at ts == as_of and ts == as_of - lookback - 1 min between alerted
    accounts (fan-in, cycles, structuring amounts): all must be invisible."""
    schema = lf.collect_schema()
    rows = []
    for t in (as_of, as_of - lookback - timedelta(minutes=1)):
        for i, k in enumerate(keys):
            for j in range(1, 4):
                rows.append({"timestamp": t, "from_key": keys[(i + j) % len(keys)], "to_key": k,
                             "amount_paid": 9500.0, "amount_received": 9500.0,
                             "payment_currency": "US Dollar", "receiving_currency": "US Dollar",
                             "payment_format": "Cash"})  # fmt: skip
    extra = pl.DataFrame(rows).with_columns(
        pl.col("from_key").str.split("|").list.get(0).alias("from_bank"),
        pl.col("from_key").str.split("|").list.get(1).alias("from_account"),
        pl.col("to_key").str.split("|").list.get(0).alias("to_bank"),
        pl.col("to_key").str.split("|").list.get(1).alias("to_account"),
    )
    extra = extra.select([pl.col(c).cast(schema[c]) for c in schema.names()])
    return pl.concat([lf, extra.lazy()])


def test_outside_window_mutation_changes_nothing(env):
    names = fc.feature_names(env["ctx"])
    lb = env["ctx"].lookback
    for as_of in sorted(env["alerts"]["as_of"].unique().to_list()):
        a = env["alerts"].filter(pl.col("as_of") == as_of)
        lf = scramble_outside_window(env["raw"], as_of, lb)
        lf = _with_edge_rows(lf, a["account_key"].to_list(), as_of, lb)
        got = fc.features_for_window(_usd(lf, env), as_of, a, env["ctx"]).sort("alert_id")
        want = env["feats"].filter(pl.col("as_of") == as_of).sort("alert_id")
        _assert_frames_equal(want, got, names)


def test_inside_window_mutation_does_change_features(env):
    """Control: the mutation test is not vacuous -- the same scramble applied INSIDE the window
    changes features of the alerted accounts."""
    as_of = sorted(env["alerts"]["as_of"].unique().to_list())[0]
    a = env["alerts"].filter(pl.col("as_of") == as_of)
    lb = env["ctx"].lookback
    lf = env["raw"].with_columns(
        pl.when((pl.col("timestamp") >= as_of - lb) & (pl.col("timestamp") < as_of))
        .then(pl.col("amount_paid") * 3).otherwise(pl.col("amount_paid")).alias("amount_paid"),
    )  # fmt: skip
    got = fc.features_for_window(_usd(lf, env), as_of, a, env["ctx"])
    want = env["feats"].filter(pl.col("as_of") == as_of).sort("alert_id")
    assert not np.allclose(got.sort("alert_id")["out_usd"].to_numpy(), want["out_usd"].to_numpy())


def test_pure_path_equals_batch_path(env):
    names = fc.feature_names(env["ctx"])
    usd = _usd(env["raw"], env)
    for row in env["alerts"].sort("alert_id").iter_rows(named=True):
        pure = fc.account_features(usd, row["account_key"], row["as_of"], row["peer_group"],
                                   env["ctx"])  # fmt: skip
        b = env["feats"].filter(pl.col("alert_id") == row["alert_id"]).row(0, named=True)
        for c in names:
            assert _close(pure[c], b[c]), f"{c} for {row['alert_id']}: pure {pure[c]} batch {b[c]}"


def test_tie_rule_at_window_edges(env):
    """A row at ts == as_of is invisible; a row at ts == as_of - L_max is visible (P0 rule)."""
    as_of = D(5)
    lb = env["ctx"].lookback
    k = env["alerts"]["account_key"][0]
    lf = _with_edge_rows(env["raw"].filter(pl.lit(False)), [k, "001|X1", "001|X2"], as_of, lb)
    lf2 = lf.with_columns(
        pl.when(pl.col("timestamp") == as_of - lb - timedelta(minutes=1))
        .then(pl.lit(as_of - lb)).otherwise(pl.col("timestamp")).alias("timestamp")
    )  # fmt: skip
    win = fc.window_frame(_usd(lf2, env), as_of, env["ctx"])
    assert win.height == 9 and (win["timestamp"] == as_of - lb).all()


# ---------------------------------------------------------------- graph


def test_graph_edges_are_inside_the_window(env):
    usd = _usd(env["raw"], env)
    for as_of in sorted(env["alerts"]["as_of"].unique().to_list()):
        e = fc.window_edges(fc.window_frame(usd, as_of, env["ctx"]))
        assert e.height > 0
        assert (e["first_ts"] >= as_of - env["split"].l_max).all()
        assert (e["last_ts"] < as_of).all()
        assert (e["from_key"] != e["to_key"]).all()


def _tiny_frame(edges: list[tuple[str, str, int]]) -> pl.LazyFrame:
    rows = [{"timestamp": D(5, 10, m), "from_bank": s.split("|")[0],
             "from_account": s.split("|")[1], "to_bank": d.split("|")[0],
             "to_account": d.split("|")[1], "amount_received": 100.0 + m,
             "receiving_currency": "US Dollar", "amount_paid": 100.0 + m,
             "payment_currency": "US Dollar", "payment_format": "ACH",
             "from_key": s, "to_key": d} for s, d, m in edges]  # fmt: skip
    return pl.DataFrame(rows).with_columns(pl.col("timestamp").dt.cast_time_unit("us")).lazy()


def test_graph_known_answers(env):
    """Hand-built day: A->B->C->A (3-cycle), A<->D (reciprocal), E->A, E->B, F->G (separate
    component), A->A (self-transfer, ignored by the graph, counted as n_self_transfers)."""
    A, B, C, Dk, E, F, G = (f"001|{x}" for x in "ABCDEFG")
    edges = [(A, B, 1), (B, C, 2), (C, A, 3), (A, Dk, 4), (Dk, A, 5), (E, A, 6), (E, B, 7),
             (F, G, 8), (A, A, 9), (A, B, 10)]  # fmt: skip
    lf = with_usd(_tiny_frame(edges), {"US Dollar": 1.0})
    alerts = pl.DataFrame({"alert_id": ["a1", "a2"], "account_key": [A, B],
                           "window_start": [D(5), D(5)], "as_of": [D(6), D(6)],
                           "peer_group": ["Corporation", "Corporation"]})  # fmt: skip
    out = fc.batch_features(lf, alerts, env["ctx"]).sort("alert_id")
    a = out.row(0, named=True)
    assert a["cycles3"] == 1 and a["reciprocal_cp"] == 1
    assert a["n_distinct_senders"] == 3 and a["n_distinct_receivers"] == 2  # C, D, E / B, D
    assert a["reach2_out"] == 1  # B->C ; D->A is A itself (excluded)
    assert a["reach2_in"] == 1  # senders of C={B}, of D={A} (itself, excluded), of E={}
    assert a["senders_mean_outdeg"] == pytest.approx((1 + 1 + 2) / 3)  # C:{A} D:{A} E:{A,B}
    assert a["receivers_mean_indeg"] == pytest.approx((2 + 1) / 2)  # B:{A,E} D:{A}
    assert a["wcc_log10_size"] == pytest.approx(math.log10(5))  # A,B,C,D,E
    assert a["n_self_transfers"] == 1 and a["max_txn_one_cp"] == 2 and a["n_txn"] == 6
    for row in alerts.iter_rows(named=True):
        pure = fc.account_features(lf, row["account_key"], row["as_of"], "Corporation", env["ctx"])
        got = out.filter(pl.col("alert_id") == row["alert_id"]).row(0, named=True)
        for c in fc.feature_names(env["ctx"]):
            assert _close(pure[c], got[c]), c


def test_unknown_payment_format_fails_loudly(env):
    A, B = "001|A", "001|B"
    lf = with_usd(_tiny_frame([(A, B, 1)]), {"US Dollar": 1.0}).with_columns(
        pl.lit("Carrier Pigeon").alias("payment_format")
    )
    al = pl.DataFrame({"alert_id": ["x"], "account_key": [A], "window_start": [D(5)],
                       "as_of": [D(6)], "peer_group": ["Corporation"]})  # fmt: skip
    with pytest.raises(fc.FeatureError, match="payment formats"):
        fc.batch_features(lf, al, env["ctx"])


# ---------------------------------------------------------------- known-flagged neighbours


def test_flagged_neighbours_point_in_time():
    """Only CONFIRMED dispositions CLOSED strictly before as_of, and only edges before as_of."""
    A, B, C, Dk = (f"001|{x}" for x in "ABCD")
    lf = _tiny_frame([(B, A, 1), (A, C, 2), (Dk, A, 3)])
    late = _tiny_frame([(A, Dk, 1)]).with_columns(pl.lit(D(6)).dt.cast_time_unit("us")
                                                  .alias("timestamp"))  # fmt: skip
    lf = pl.concat([lf, late])
    disp = pl.DataFrame({
        "alert_id": ["d1", "d2", "d3", "d4"],
        "account_key": [B, C, Dk, Dk],
        "disposition": ["confirmed_suspicious", "confirmed_suspicious", "closed_legitimate",
                        "confirmed_suspicious"],
        "closed_at": [D(5, 9), D(6), D(5, 1), D(6, 0, 1)],
    }).lazy()  # fmt: skip
    r = flagged_counterparties(lf, disp, A, D(6), timedelta(hours=24))
    assert r["flagged_counterparties"] == [B]  # C closed AT as_of (tie excluded); D: legit/late
    assert r["n_counterparties"] == 3  # B, C, D (the A->D edge at as_of is invisible)


# ---------------------------------------------------------------- split


def test_split_boundaries_on_alerts(env):
    df = splits.assign_periods(env["alerts"], env["split"])
    margins = splits.check_boundaries(df, env["split"])
    assert set(margins) == {"TRAIN->VALIDATION", "VALIDATION->CALIBRATION", "CALIBRATION->TEST"}
    for m in margins.values():
        assert m["gap_hours"] >= m["embargo_hours"] + 24


def _shift(df: pl.DataFrame, period: str, hours: float) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("split_period") == period)
        .then(pl.col("as_of") + timedelta(hours=hours))
        .otherwise(pl.col("as_of"))
        .alias("as_of")
    )


def test_split_phase_rule_violation_is_caught(env):
    """Rule (1): VALIDATION moved 24 h earlier -> gap == embargo, not strictly greater."""
    df = splits.assign_periods(env["alerts"], env["split"])
    with pytest.raises(splits.SplitViolation, match="embargo"):
        splits.check_boundaries(_shift(df, "VALIDATION", -24), env["split"])


def test_split_window_rule_violation_is_caught(env):
    """Rule (2) on its own: an L_max longer than the 48 h gap passes rule (1) but feature
    windows would reach the previous period's label windows."""
    from dataclasses import replace

    df = splits.assign_periods(env["alerts"], env["split"])
    long_l = replace(env["split"], l_max=timedelta(hours=49))
    with pytest.raises(splits.SplitViolation, match="feature windows"):
        splits.check_boundaries(df, long_l)


def test_rows_outside_the_split_are_refused(env):
    emb = env["alerts"].head(1).with_columns(pl.lit(D(6)).alias("window_start")).drop("period")
    with pytest.raises(splits.SplitViolation, match="outside"):
        splits.assign_periods(emb, env["split"])


def test_unseen_account_slice(env):
    a = env["alerts"]
    train = set(a.filter(pl.col("period") == "TRAIN")["account_key"].to_list())
    want = {r["alert_id"] for r in a.iter_rows(named=True)
            if r["period"] == "TEST" and r["account_key"] not in train}  # fmt: skip
    got = set(splits.unseen_test_slice(a)["alert_id"].to_list())
    assert got == want and got  # the fixture has unseen TEST alerts
    assert len(got) < a.filter(pl.col("period") == "TEST").height or not train


def test_agent_evaluation_alerts_must_come_from_test(env):
    a = env["alerts"]
    splits.require_test_period(a.filter(pl.col("period") == "TEST"))
    with pytest.raises(splits.SplitViolation, match="not from TEST"):
        splits.require_test_period(a.filter(pl.col("period").is_in(["TEST", "CALIBRATION"])))


def test_committed_registry_matches_code():
    """docs/p3/feature_registry.{json,md} are generated from the code; must be current."""
    from scripts.p3.export_registry import OUT, render

    js, md = render(fc.default_context())
    assert (OUT / "feature_registry.json").read_text(encoding="utf-8") == js
    assert (OUT / "feature_registry.md").read_text(encoding="utf-8") == md


def test_fitted_inputs_are_train_only():
    """P3 fits no scaler, encoder or target encoding (none exists in src/features). The only fitted
    inputs the features use are P2's FX table and rule thresholds: both must be fitted on TRAIN."""
    split = load_split()
    tr = split.periods["TRAIN"]
    fx = yaml.safe_load((ROOT / "configs" / "p2_fx_usd_per_unit.yaml").read_text(encoding="utf-8"))
    assert datetime.fromisoformat(fx["fitted_on"]["start"]) >= tr[0]
    assert datetime.fromisoformat(fx["fitted_on"]["end_exclusive"]) <= tr[1]
    th = rules.load_thresholds()
    assert th["calibrated_on"]["period"] == "TRAIN"
    assert datetime.fromisoformat(th["calibrated_on"]["end_exclusive"]) <= tr[1]
    banned = ("fit_transform", "StandardScaler", "OneHotEncoder", "TargetEncoder", ".fit(")
    for py in (ROOT / "src" / "features").glob("*.py"):
        src = py.read_text(encoding="utf-8")
        assert not any(b in src for b in banned), f"{py.name} fits something"
