"""AGENT-DEV / AGENT-TEST split v2 (P3 task 0; independent P2 review findings M-6 and C-2).

Each gap found by the review gets a test that FAILS under v1 and passes under the variant meant to
close it:
  (a) hub without a TEST alert joining two TEST-alerted accounts      -> closed by C only
  (b) attempt member whose legs are only in the D1 tail (>= 09-17)    -> closed by A, B, C
  (c) unattributed laundering link dated before TEST                  -> closed by B, C
plus the pre-declared selection rule, cover/disjointness/order properties on random graphs, and
the parsers that read the raw patterns and transactions.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime

import polars as pl
import pytest
import yaml

from scripts.p2.run_p2 import SPLIT_V2_CONFIG, choose_variant
from src.eval import group_split as gs

D = lambda day: datetime(2022, 9, day, tzinfo=UTC)  # noqa: E731
T0, T1 = D(13), D(17)
A, B, C = gs.VARIANTS
FRACTIONS = [i / 200 for i in range(1, 200)]


def att_df(rows):  # (attempt_id, account_key)
    return pl.DataFrame(
        {"node": [f"att:{r[0]}" for r in rows], "account_key": [r[1] for r in rows]},
        schema={"node": pl.String, "account_key": pl.String},
    )


def un_df(rows):  # (row_id, account_key, ts)
    return pl.DataFrame(
        {
            "node": [f"txn:{r[0]}" for r in rows],
            "account_key": [r[1] for r in rows],
            "timestamp": [r[2] for r in rows],
        },
        schema={"node": pl.String, "account_key": pl.String, "timestamp": pl.Datetime("us", "UTC")},
    )


EMPTY_ATT = att_df([("0", "_none_")]).clear()
EMPTY_UN = un_df([(0, "_none_", D(2))]).clear()


def alerts_df(keys):
    return pl.DataFrame({"alert_id": [f"a_{k}" for k in keys], "account_key": keys})


def groups(s):
    return dict(zip(s["account_key"], s["agent_group"], strict=True))


def split(variant, alerts, att, un, frac):
    m = gs.memberships(variant, att, un, set(alerts["account_key"]), T0, T1)
    return gs.split_v2(alerts, m, "k", frac)


def ever_split(variant, alerts, att, un, a="x", b="z") -> bool:
    """True if some fraction puts accounts a and b in different groups."""
    for f in FRACTIONS:
        g = groups(split(variant, alerts, att, un, f))
        if g[a] != g[b]:
            return True
    return False


# ---------------------------------------------------------------- the three gaps


def test_gap_a_hub_without_test_alert_is_joined_only_by_variant_c():
    # x -A1- y (no TEST alert) -A2- z
    att = att_df([(1, "x"), (1, "y"), (2, "y"), (2, "z")])
    al = alerts_df(["x", "z"])
    assert ever_split(A, al, att, EMPTY_UN) and ever_split(B, al, att, EMPTY_UN)
    assert not ever_split(C, al, att, EMPTY_UN)


def test_gap_b_tail_only_attempt_member_is_joined_by_every_v2_variant():
    # attempt 3 = x (legs in TEST) + z (legs only on 09-18, in the excluded tail). v2 takes
    # membership from the patterns file, which has no date cut, so the attempt links x and z.
    att = att_df([(3, "x"), (3, "z")])
    al = alerts_df(["x", "z"])
    for v in gs.VARIANTS:
        assert not ever_split(v, al, att, EMPTY_UN), v
        for f in (0.1, 0.5, 0.9):
            gs.check_disjoint_v2(split(v, al, att, EMPTY_UN, f), att, EMPTY_UN)


def test_gap_b_v1_misses_it_and_v2_check_sees_it():
    """The v1 assignment can put x and z apart; the v2 check on FULL memberships raises."""
    legs_in_span = pl.DataFrame(
        {
            "account_key": ["x"],
            "window_start": [D(15)],
            "row_id": [1],
            "attempt_id": [3],
            "typology": ["X"],
            "timestamp": [D(15)],
        },
        schema_overrides={"attempt_id": pl.Int32, "row_id": pl.Int64},
    )
    al = alerts_df(["x", "z"])
    att = att_df([(3, "x"), (3, "z")])
    for f in FRACTIONS:
        s = gs.split_test_alerts(al, legs_in_span, T0, T1, "k", f)
        if len(set(groups(s).values())) == 2:
            gs.check_disjoint(s, legs_in_span)  # v1 check is blind
            with pytest.raises(AssertionError, match="pattern instance"):
                gs.check_disjoint_v2(s, att, EMPTY_UN)
            return
    pytest.fail("no fraction split x and z under v1")


def test_gap_c_pre_test_unattributed_link_is_joined_by_b_and_c_only():
    un = un_df([(5, "x", D(11)), (5, "z", D(11))])  # CALIBRATION-dated laundering x -> z
    al = alerts_df(["x", "z"])
    assert ever_split(A, al, EMPTY_ATT, un)
    assert not ever_split(B, al, EMPTY_ATT, un)
    assert not ever_split(C, al, EMPTY_ATT, un)


def test_unattributed_link_in_test_is_joined_by_every_variant():
    un = un_df([(5, "x", D(14)), (5, "z", D(14))])
    al = alerts_df(["x", "z"])
    for v in gs.VARIANTS:
        assert not ever_split(v, al, EMPTY_ATT, un), v


def test_singleton_keeps_its_v1_group():
    """Same hash key and fraction: a component with the same members gets the same group."""
    al = alerts_df(["x", "y", "z"])
    legs = pl.DataFrame(
        schema={
            "account_key": pl.String,
            "window_start": pl.Datetime("us", "UTC"),
            "row_id": pl.Int64,
            "attempt_id": pl.Int32,
            "typology": pl.String,
            "timestamp": pl.Datetime("us", "UTC"),
        }
    )
    for f in (0.2, 0.5, 0.8):
        v1 = gs.split_test_alerts(al, legs, T0, T1, "k", f)
        for v in gs.VARIANTS:
            assert split(v, al, EMPTY_ATT, EMPTY_UN, f).equals(v1)


# ---------------------------------------------------------------- selection rule


def test_selection_rule_takes_first_passing_variant_in_declared_order():
    order = yaml.safe_load(SPLIT_V2_CONFIG.read_text(encoding="utf-8"))["preference_order"]
    assert order == [C, B, A]  # most conservative first (declared 2026-10-05)
    assert choose_variant(order, {A: True, B: True, C: True}) == C
    assert choose_variant(order, {A: True, B: True, C: False}) == B
    assert choose_variant(order, {A: True, B: False, C: False}) == A
    assert choose_variant(order, {A: False, B: False, C: False}) is None
    with pytest.raises(RuntimeError):
        choose_variant([C, B], {A: True, B: True, C: True})


def test_config_is_declared_and_keeps_the_frozen_key_and_fraction():
    cfg = yaml.safe_load(SPLIT_V2_CONFIG.read_text(encoding="utf-8"))
    assert cfg["version"] == 2 and cfg["declared_on"] == "2026-10-05"
    assert set(cfg["variants"]) == set(gs.VARIANTS)
    assert cfg["outputs"]["devtools_dev_ids"].startswith("devtools/")
    assert cfg["outputs"]["removed_from_runtime"].startswith("runtime/")
    assert cfg["group_key"] == {"source": "env", "env_var": "DOSSIER_AGENT_SPLIT_KEY"}


def test_group_key_is_secret_and_only_its_hash_is_recorded(monkeypatch, tmp_path):
    from scripts.p2 import run_p2
    from src.data.periods import load_split

    cfg = yaml.safe_load(SPLIT_V2_CONFIG.read_text(encoding="utf-8"))
    split = load_split()
    monkeypatch.delenv("DOSSIER_AGENT_SPLIT_KEY", raising=False)
    monkeypatch.setattr(run_p2, "ROOT", tmp_path)  # no .env there
    with pytest.raises(RuntimeError, match="DOSSIER_AGENT_SPLIT_KEY"):
        run_p2.agent_split_key(cfg, split)
    monkeypatch.setenv("DOSSIER_AGENT_SPLIT_KEY", "short")
    with pytest.raises(RuntimeError):
        run_p2.agent_split_key(cfg, split)
    secret = "f" * 64
    monkeypatch.setenv("DOSSIER_AGENT_SPLIT_KEY", secret)
    key, rec = run_p2.agent_split_key(cfg, split)
    assert key == secret and rec["source"] == "env" and secret not in json.dumps(rec)
    monkeypatch.delenv("DOSSIER_AGENT_SPLIT_KEY")
    (tmp_path / ".env").write_text(f"OTHER=1\nDOSSIER_AGENT_SPLIT_KEY={secret}\n", "utf-8")
    assert run_p2.agent_split_key(cfg, split)[0] == secret  # read from .env as well


def test_public_key_leaks_linkage_and_secret_key_does_not():
    """Why the key must be secret: with the public key, an account whose group differs from its
    singleton group is provably in a multi-account (laundering-linked) component."""
    keys = [f"001|K{i}" for i in range(400)]
    att = att_df([(i // 4, k) for i, k in enumerate(keys[:200])])  # 50 attempts x 4 accounts
    al = alerts_df(keys)

    def exposed(split_key, guess_key):
        m = gs.memberships(C, att, EMPTY_UN, set(keys), T0, T1)
        s = gs.split_v2(al, m, split_key, 0.5)
        single = gs.split_v2(al, m.clear(), guess_key, 0.5)  # what each would get alone
        g, g1 = groups(s), groups(single)
        return {k for k in keys if g[k] != g1[k]}

    pub = exposed("dossier-p2-agent-split-v1", "dossier-p2-agent-split-v1")
    assert pub and pub <= set(keys[:200])  # every exposed account is linked: precision 1.0
    sec = exposed("s" * 64, "dossier-p2-agent-split-v1")  # attacker only knows the public key
    linked = len(sec & set(keys[:200])) / len(sec)
    assert 0.3 < linked < 0.7  # ~ base rate 0.5: no information


# ---------------------------------------------------------------- properties on random graphs


def test_property_cover_disjoint_order_free_and_nested():
    rnd = random.Random(0)
    for trial in range(150):
        n = rnd.randint(2, 40)
        keys = [f"{rnd.randint(0, 9):03d}|K{i}" for i in range(n)]
        alerted = rnd.sample(keys, rnd.randint(1, n))
        att = (
            att_df(
                [
                    (a, m)
                    for a in range(rnd.randint(0, 10))
                    for m in rnd.sample(keys, rnd.randint(1, 5))
                ]
            )
            if n >= 5
            else EMPTY_ATT
        )
        un_rows = []
        for r in range(rnd.randint(0, 10)):
            x, y = rnd.sample(keys, 2)
            ts = D(rnd.choice([1, 4, 9, 12, 13, 15]))
            un_rows += [(r, x, ts), (r, y, ts)]
        un = un_df(un_rows) if un_rows else EMPTY_UN
        al = alerts_df(alerted)
        frac = rnd.random()
        comps = {}
        for v in gs.VARIANTS:
            s = split(v, al, att, un, frac)
            assert s.height == al.height and set(s["alert_id"]) == set(al["alert_id"])
            gs.check_disjoint_v2(s, att, un)  # never an attempt in both groups
            m = gs.memberships(v, att, un, set(alerted), T0, T1)
            s2 = gs.split_v2(
                al.sample(fraction=1, shuffle=True, seed=trial),
                m.sample(fraction=1, shuffle=True, seed=trial),
                "k",
                frac,
            )
            assert s2.equals(s)  # independent of row order
            comps[v] = {frozenset(g["account_key"]) for _, g in s.group_by("component")}
        # graphs are nested A <= B <= C: every component of a smaller graph lies inside one
        # component of the larger graph
        for small, big in ((A, B), (B, C)):
            for c in comps[small]:
                assert any(c <= d for d in comps[big]), (trial, small, big)


# ---------------------------------------------------------------- parsers on the fixture


def test_parsers_read_every_attempt_and_unattributed_txn_including_the_tail(p2_run):
    from src.data.transactions import interim_path
    from tests.p2_fixture import L, key

    interim = p2_run["info"]["interim"]
    att = gs.attempt_members(interim_path(interim, "FIX", "patterns"))
    un = gs.unattributed_members(
        interim_path(interim, "FIX", "trans"), interim_path(interim, "FIX", "patterns")
    )
    assert att["node"].n_unique() == 5  # 5 attempts in the fixture, incl. the tail attempt
    tail_attempt = {
        n for n, g in att.group_by("node") if set(g["account_key"]) == {key(L(0)), key(L(1))}
    }
    assert len(tail_attempt) == 1  # the 09-17 FAN-OUT L00 -> L01 is present (no date cut)
    assert set(un["account_key"]) == {key(L(i)) for i in (40, 41, 42, 43)}
    assert un["node"].n_unique() == 3  # 2 x (L40 -> L41) on 09-04, 1 x (L42 -> L43) on 09-15
    assert un["timestamp"].dtype == pl.Datetime("us", "UTC")


def test_fixture_build_records_every_variant_and_the_choice(p2_run):
    rep = p2_run["doc"]["build"]["agent_split"]
    assert set(rep["variants"]) == set(gs.VARIANTS)
    assert rep["chosen_variant"] == C  # fixture gates relaxed in conftest -> most conservative
    for v in gs.VARIANTS:
        d = rep["variants"][v]["disjointness_on_full_memberships"]
        assert d["n_attempt_nodes_in_both_groups"] == 0


# ---------------------------------------------------------------- the one-off `split` stage


def test_split_stage_rebuilds_on_a_p2v1_build(p2_run, tmp_path):
    """Recreate the laptop state after P2 (v1 split, dev ids in the runtime store), run the
    one-off `split` stage, and check: v1 archived, runtime cleaned, devtools written, original
    feasibility kept, alerts moved reported, no other store touched."""
    import hashlib
    import json
    import shutil

    from scripts.p2 import run_p2

    src = p2_run["root"]
    dst = tmp_path / "copy"
    shutil.copytree(src / "out", dst / "out")
    shutil.copytree(src / "configs", dst / "configs")
    base = dst / "out" / "FIX"
    # P2 v1 state
    a = pl.read_parquet(base / "runtime" / "alerts.parquet").filter(pl.col("period") == "TEST")
    legs = pl.read_parquet(base / "eval" / "laundering_legs.parquet")
    v1 = gs.split_test_alerts(a.select("alert_id", "account_key"), legs, T0, T1, "k", 0.5)
    v1.write_parquet(base / "eval" / "agent_split.parquet")
    v1.filter(pl.col("agent_group") == gs.AGENT_DEV).select("alert_id").write_parquet(
        base / "runtime" / "agent_dev_alert_ids.parquet"
    )
    shutil.rmtree(base / "devtools")
    doc = json.loads(json.dumps(p2_run["doc"]))
    doc["build"].pop("agent_split")  # the P2 v1 results JSON has no agent_split block
    doc["build"]["feasibility"]["counts"].pop("TEST_components_disjointness")
    doc["stores"].pop("devtools/agent_dev_alert_ids.parquet")
    (dst / "results.json").write_text(json.dumps(doc), encoding="utf-8")
    untouched = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*base.joinpath("runtime").glob("*.parquet"), *base.joinpath("eval").glob("*")]
        if p.name not in ("agent_split.parquet", "agent_dev_alert_ids.parquet")
    }
    old_feas = doc["build"]["feasibility"]

    out = run_p2.run(
        "split",
        "FIX",
        p2_run["info"]["interim"],
        dst / "out",
        dst / "configs",
        dst / "results.json",
        20260930,
        cp=p2_run["cp"],
        primary="FIX",
    )
    assert not (base / "runtime" / "agent_dev_alert_ids.parquet").exists()
    assert (base / "devtools" / "agent_dev_alert_ids.parquet").is_file()
    assert pl.read_parquet(base / "archive" / "agent_split_p2v1.parquet").equals(v1)
    new = pl.read_parquet(base / "eval" / "agent_split.parquet")
    assert set(new["alert_id"]) == set(a["alert_id"])
    b = out["build"]
    orig = out["p2v1_original"]
    assert orig["feasibility"] == json.loads(json.dumps(old_feas))
    assert orig["run_info"] == json.loads(json.dumps(p2_run["doc"]["run_info"]))
    assert "moved_to_other_group_in_v2" in orig
    assert b["agent_split"]["chosen_variant"] == C
    assert "archive/agent_split_p2v1.parquet" in out["stores"]
    assert "devtools/agent_dev_alert_ids.parquet" in out["stores"]
    assert "runtime/agent_dev_alert_ids.parquet" not in out["stores"]
    for p, h in untouched.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == h, p.name


def test_split_stage_stops_and_keeps_diagnostics_when_no_variant_passes(p2_run, tmp_path):
    import json
    import shutil

    from scripts.p2 import run_p2

    dst = tmp_path / "copy"
    shutil.copytree(p2_run["root"] / "out", dst / "out")
    shutil.copytree(p2_run["root"] / "configs", dst / "configs")
    (dst / "results.json").write_text(json.dumps(p2_run["doc"]), encoding="utf-8")
    strict = tmp_path / "split.yaml"  # the REAL gates: the fixture cannot pass them
    shutil.copyfile(SPLIT_V2_CONFIG.parent / "p2_split.yaml", strict)
    cp = run_p2.ConfigPaths(
        rules=p2_run["cp"].rules,
        sources=p2_run["cp"].sources,
        split=strict,
        agent_split_v2=p2_run["cp"].agent_split_v2,
    )
    before = (dst / "out" / "FIX" / "eval" / "agent_split.parquet").read_bytes()
    with pytest.raises(run_p2.SplitSelectionError):
        run_p2.run(
            "split", "FIX", p2_run["info"]["interim"], dst / "out", dst / "configs",
            dst / "results.json", 20260930, cp=cp, primary="FIX",
        )  # fmt: skip
    rep = json.loads((dst / "p2_agent_split_v2_no_variant_passed.json").read_text("utf-8"))
    assert rep["chosen_variant"] is None and set(rep["variants"]) == set(gs.VARIANTS)
    assert (dst / "out" / "FIX" / "eval" / "agent_split.parquet").read_bytes() == before


def test_variant_a_equals_v1_when_no_member_is_outside_the_cut_legs():
    """Second-pass review: A must reproduce v1 exactly whenever the full memberships coincide with
    v1's in-span legs (no tail-only members)."""
    rnd = random.Random(7)
    for trial in range(200):
        n = rnd.randint(2, 30)
        keys = [f"{rnd.randint(0, 9):03d}|K{i}" for i in range(n)]
        alerted = rnd.sample(keys, rnd.randint(1, n))
        rows, rid = [], 0
        for a in range(rnd.randint(0, 8)):
            for m in rnd.sample(keys, rnd.randint(1, min(4, n))):
                rid += 1
                rows.append((m, rid, a, D(rnd.choice([3, 9, 13, 15]))))
        for _ in range(rnd.randint(0, 6)):
            x, y = rnd.sample(keys, 2)
            rid += 1
            ts = D(rnd.choice([9, 13, 15]))
            rows += [(x, rid, None, ts), (y, rid, None, ts)]
        legs = pl.DataFrame(
            {
                "account_key": [r[0] for r in rows],
                "window_start": [r[3] for r in rows],
                "row_id": [r[1] for r in rows],
                "attempt_id": [r[2] for r in rows],
                "typology": [None] * len(rows),
                "timestamp": [r[3] for r in rows],
            },
            schema={
                "account_key": pl.String,
                "window_start": pl.Datetime("us", "UTC"),
                "row_id": pl.Int64,
                "attempt_id": pl.Int32,
                "typology": pl.String,
                "timestamp": pl.Datetime("us", "UTC"),
            },
        )
        att = att_df([(r[2], r[0]) for r in rows if r[2] is not None]) if rows else EMPTY_ATT
        un = un_df([(r[1], r[0], r[3]) for r in rows if r[2] is None]) if rows else EMPTY_UN
        al = alerts_df(alerted)
        f = rnd.random()
        v1 = gs.split_test_alerts(al, legs, T0, T1, "k", f)
        assert split(A, al, att, un, f).equals(v1), trial


def test_disjointness_check_raises_on_a_linked_unattributed_txn_in_both_groups():
    s = pl.DataFrame(
        {
            "alert_id": ["a_x", "a_z"],
            "account_key": ["x", "z"],
            "agent_group": [gs.AGENT_DEV, gs.AGENT_TEST],
            "component": ["1", "2"],
        }
    )
    pre = un_df([(5, "x", D(11)), (5, "z", D(11))])  # CALIBRATION-dated link
    gs.check_disjoint_v2(s, EMPTY_ATT, pre, A, T0, T1)  # A does not link pre-TEST: allowed
    for v in (B, C):
        with pytest.raises(AssertionError, match="unattributed"):
            gs.check_disjoint_v2(s, EMPTY_ATT, pre, v, T0, T1)
    intest = un_df([(6, "x", D(14)), (6, "z", D(14))])
    with pytest.raises(AssertionError, match="unattributed"):
        gs.check_disjoint_v2(s, EMPTY_ATT, intest, A, T0, T1)
