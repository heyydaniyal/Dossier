"""Identity guard v2 (configs/p3_identity_guard.yaml; approved by Dani 2026-10-05, review C-1).

- the weighted AUC / AP equal scikit-learn (unit weights) and row repetition (integer weights);
- each scenario gets the verdict it was designed for, at realistic sizes (3 folds, ~90 seen and
  ~550 unseen positives): no effect, equal real effect, seen-only effect, harm to unseen;
- thresholds, verdict order and folds are the approved ones; folds stay in TRAIN/VALIDATION with
  an embargo day;
- deterministic, and independent of row order and of account-ID order (D5);
- v1 (P2 record) still gives the recorded KYC verdict logic.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.periods import load_split
from src.eval import identity_guard as ig

CFG = ig.load_v2_config()
FAST = {**CFG, "bootstrap": {**CFG["bootstrap"], "n_boot": 400}}
EVAL_DAYS = [datetime.fromisoformat(f["eval_day"]).replace(tzinfo=UTC) for f in CFG["folds"]]
NAMES = [f["name"] for f in CFG["folds"]]


def run(folds, cfg=FAST):
    return ig.evaluate_v2(folds, cfg, allow_unfrozen_config=cfg is not CFG)


# per fold: (seen n, seen pos, unseen n, unseen pos); ~ 3x the VALIDATION subset sizes in total
SIZES = [(1600, 25, 2400, 180), (1700, 30, 2500, 185), (1692, 35, 2466, 185)]
MU_SEEN, MU_UNSEEN = 2.5, 0.55  # base separations tuned to AP ~0.64 seen / ~0.11 unseen


def _folds(add_seen: float, add_unseen: float, seed: int, rho: float = 0.95) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for fi, (ns, ps, nu, pu) in enumerate(SIZES):
        y = np.r_[np.ones(ps), np.zeros(ns - ps), np.ones(pu), np.zeros(nu - pu)].astype(int)
        seen = np.r_[np.ones(ns, bool), np.zeros(nu, bool)]
        mu = np.where(seen, MU_SEEN, MU_UNSEEN)
        add = np.where(seen, add_seen, add_unseen)
        e1 = rng.standard_normal(len(y))
        e2 = rho * e1 + np.sqrt(1 - rho**2) * rng.standard_normal(len(y))
        out.append(
            {
                "name": NAMES[fi],
                "window_start": [EVAL_DAYS[fi]] * len(y),
                "y": y,
                "pred_base": y * mu + e1,
                "pred_with_group": y * (mu + add) + e2,
                "seen": seen,
                "account_key": np.array([f"001|F{fi}A{i}" for i in range(len(y))]),
            }
        )
    return out


# ---------------------------------------------------------------- metric correctness


def test_weighted_metrics_match_sklearn_and_row_repetition():
    rng = np.random.default_rng(0)
    for _ in range(20):
        sp = np.round(rng.normal(1, 1, 40), 1)  # rounding creates ties
        sn = np.round(rng.normal(0, 1, 300), 1)
        auc, ap = ig.weighted_auc_ap(sp, sn, np.ones((1, 40)), np.ones((1, 300)))
        y = np.r_[np.ones(40), np.zeros(300)]
        s = np.r_[sp, sn]
        assert auc[0] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
        assert ap[0] == pytest.approx(average_precision_score(y, s), abs=1e-12)
        wp, wn = rng.integers(0, 3, (1, 40)), rng.integers(0, 3, (1, 300))
        wp[0, 0], wn[0, 0] = 1, 1  # keep both classes present
        auc_w, ap_w = ig.weighted_auc_ap(sp, sn, wp.astype(float), wn.astype(float))
        yr = np.r_[np.ones(int(wp.sum())), np.zeros(int(wn.sum()))]
        sr = np.r_[np.repeat(sp, wp[0]), np.repeat(sn, wn[0])]
        assert auc_w[0] == pytest.approx(roc_auc_score(yr, sr), abs=1e-12)
        assert ap_w[0] == pytest.approx(average_precision_score(yr, sr), abs=1e-12)


# ---------------------------------------------------------------- verdicts by scenario


@pytest.mark.parametrize(
    "add_seen,add_unseen,allowed",
    [
        (0.0, 0.0, {"no_material_gain", "pass"}),  # no effect: cleared, never blocked
        (0.6, 0.6, {"pass"}),  # same real effect on both subsets (v1 blocked this ~99%)
        (0.8, 0.0, {"account_recognition"}),  # helps only accounts the model has seen
        (0.8, 0.1, {"account_recognition"}),  # partial recognition, unseen << 0.5 x seen
        (0.0, -0.4, {"harms_unseen"}),  # hurts new accounts (v1 cleared this ~50%)
    ],
)
def test_each_scenario_gets_its_verdict(add_seen, add_unseen, allowed):
    for seed in (1, 2, 3):
        r = run(_folds(add_seen, add_unseen, seed))
        assert r["verdict"] in allowed, (add_seen, add_unseen, seed, r["verdict"])
        assert r["cleared"] == FAST["verdicts"][r["verdict"]]["cleared"]


def test_null_groups_are_rarely_blocked():
    """Operating characteristic (small version of the approved simulation): over 30 no-effect
    groups at least 25 are cleared (simulation: ~94%)."""
    cleared = sum(run(_folds(0.0, 0.0, 100 + s))["cleared"] for s in range(30))
    assert cleared >= 25


def test_insufficient_data_rules():
    f = _folds(0.8, 0.0, 5)
    few = [dict(x) for x in f]
    for x in few:  # keep only 9 seen positives per fold -> 27 pooled < 30
        keep = ~(x["seen"] & (x["y"] == 1)) | (np.cumsum(x["seen"] & (x["y"] == 1)) <= 9)
        for k in ("y", "pred_base", "pred_with_group", "seen", "account_key", "window_start"):
            x[k] = np.asarray(x[k])[keep]
    assert run(few)["verdict"] == "insufficient_data"
    near_perfect = [dict(x) for x in f]
    for x in near_perfect:  # seen base AUC ~ 1 -> d' unstable
        x["pred_base"] = np.where(x["seen"], x["y"] * 50.0, x["pred_base"])
    r = run(near_perfect)
    assert r["verdict"] == "insufficient_data"
    assert all(pf.get("dropped") for pf in r["seen"]["per_fold"])
    with pytest.raises(ValueError, match="folds"):
        run(f[:2])
    bad = [dict(x) for x in f]
    bad[0]["seen"] = bad[0]["seen"][:-1]
    with pytest.raises(ValueError, match="aligned"):
        run(bad)


# ---------------------------------------------------------------- determinism and D5


def test_deterministic_and_independent_of_row_and_id_order():
    f = _folds(0.8, 0.1, 9)
    a = run(f)
    assert run(f) == a
    rng = np.random.default_rng(3)
    shuffled = []
    for x in f:
        p = rng.permutation(len(x["y"]))
        shuffled.append({k: np.asarray(v)[p] for k, v in x.items() if k != "name"})
    for x, y in zip(shuffled, f, strict=True):
        x["name"] = y["name"]
    b = run(shuffled)
    assert b["verdict"] == a["verdict"]
    assert b["seen"]["ci"] == pytest.approx(a["seen"]["ci"], abs=1e-12)
    renamed = [dict(x) for x in f]  # relabel accounts monotonically: ID order must not matter
    for x in renamed:
        x["account_key"] = np.array([k.replace("001|", "999|Z") for k in x["account_key"]])
    c = run(renamed)
    assert c["verdict"] == a["verdict"]


def test_accounts_recurring_across_folds_share_one_multiplicity():
    f = _folds(0.0, 0.0, 11)
    for x in f:  # the same 500 accounts appear on every eval day
        x["account_key"] = np.array([f"001|A{i % 500}" for i in range(len(x["y"]))])
    w = ig._account_weights(f, 50, 7)
    k0, k1 = f[0]["account_key"], f[1]["account_key"]
    i0 = int(np.where(k0 == "001|A3")[0][0])
    i1 = int(np.where(k1 == "001|A3")[0][0])
    assert np.array_equal(w[0][:, i0], w[1][:, i1])


# ---------------------------------------------------------------- the approved configuration


def test_config_is_the_approved_one():
    raw = ig.GUARD_V2_CONFIG.read_bytes().replace(b"\r\n", b"\n")
    assert CFG["_sha256"] == hashlib.sha256(raw).hexdigest()
    assert CFG["status"] == "FROZEN" and CFG["approved_by"] == "Dani"
    assert CFG["approved_on"] == "2026-10-05"
    assert CFG["thresholds"] == {
        "n_min_pos_seen": 30,
        "n_min_pos_unseen": 30,
        "ratio": 0.5,
        "base_auc_max": 0.995,
    }
    assert CFG["bootstrap"] == {"unit": "account", "n_boot": 2000, "seed": 20260930, "ci": 0.95}
    assert tuple(CFG["verdicts"]) == ig.VERDICT_ORDER_V2
    cleared = {v for v, d in CFG["verdicts"].items() if d["cleared"]}
    assert cleared == {"pass", "no_material_gain"}


def test_folds_stay_in_train_validation_with_an_embargo_day():
    folds = ig.check_folds(CFG, load_split())
    assert [f["name"] for f in folds] == ["F1", "F2", "F3"]
    bad = {**CFG, "folds": [{"name": "X", "fit_days": ["2022-09-02"], "eval_day": "2022-09-03"}]}
    with pytest.raises(ValueError, match="embargo"):
        ig.check_folds(bad, load_split())
    cal = {**CFG, "folds": [{"name": "X", "fit_days": ["2022-09-02"], "eval_day": "2022-09-09"}]}
    with pytest.raises(ValueError, match="outside"):
        ig.check_folds(cal, load_split())


def test_v1_is_kept_unchanged_for_the_p2_record():
    from src.eval.identity_guard import evaluate

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    r = evaluate(y, rng.random(400), rng.random(400), rng.random(400) < 0.5, {
        "min_positives": 20, "materiality_gain_over_prev": 0.05, "min_unseen_to_seen_ratio": 0.5,
    })  # fmt: skip
    assert r["verdict"] in ig.VERDICTS


# ---------------------------------------------------------------- added after the independent check


def test_bootstrap_is_paired_identical_predictions_give_zero_gain_exactly():
    f = _folds(0.0, 0.0, 21)
    for x in f:
        x["pred_with_group"] = x["pred_base"].copy()
    r = run(f)
    assert r["seen"]["ci"] == [0.0, 0.0] and r["unseen"]["ci"] == [0.0, 0.0]
    assert r["verdict"] == "no_material_gain"


def test_ratio_is_half_not_one():
    """Unseen gain 0.75 x seen gain (> 0.5): must pass, not be called recognition."""
    for seed in (1, 2):
        assert run(_folds(0.8, 0.6, seed))["verdict"] == "pass"


def test_n_min_boundary_is_inclusive_at_30():
    def cut(k):
        f = _folds(0.0, 0.0, 31)
        out = []
        for i, x in enumerate(f):  # keep k seen positives in F3 only, none elsewhere
            sp = x["seen"] & (x["y"] == 1)
            keep = ~sp | ((np.cumsum(sp) <= k) & (i == 2))
            out.append({**{kk: np.asarray(v)[keep] for kk, v in x.items() if kk != "name"},
                        "name": x["name"]})  # fmt: skip
        return run(out)

    assert cut(29)["verdict"] == "insufficient_data"
    assert cut(30)["verdict"] != "insufficient_data"


def test_decision_table():
    pos, zero, neg = [0.1, 0.5], [-0.1, 0.1], [-0.5, -0.1]
    assert ig.decide(zero, neg, neg) == "harms_unseen"
    assert ig.decide(pos, zero, neg) == "account_recognition"
    assert ig.decide(zero, zero, neg) == "no_material_gain"  # seen not credibly > 0
    assert ig.decide(pos, pos, neg) == "account_recognition"  # recognition before pass
    assert ig.decide(zero, pos, zero) == "pass"
    assert ig.decide(pos, zero, zero) == "inconclusive"


def test_pooling_is_positive_weighted_and_ci_level_comes_from_config():
    r = run(_folds(0.5, 0.2, 41))
    for sub in ("seen", "unseen"):
        pf = [x for x in r[sub]["per_fold"] if "gain_dprime" in x]
        w = np.array([x["n_pos"] for x in pf], float)
        g = np.array([x["gain_dprime"] for x in pf])
        assert r[sub]["gain_dprime"] == pytest.approx(float(np.dot(w / w.sum(), g)), abs=1e-12)
    rep = np.linspace(0, 1, 1001)
    assert ig.percentile_ci(rep, 0.95) == pytest.approx([0.025, 0.975])
    assert ig.percentile_ci(rep, 0.90) == pytest.approx([0.05, 0.95])


def test_thin_fold_never_crashes():
    f = _folds(0.8, 0.0, 51)
    x = f[0]  # F1 seen subset cut to 2 positives
    sp = x["seen"] & (x["y"] == 1)
    keep = ~sp | (np.cumsum(sp) <= 2)
    f[0] = {**{k: np.asarray(v)[keep] for k, v in x.items() if k != "name"}, "name": x["name"]}
    r = run(f)
    assert r["verdict"] in ig.VERDICT_ORDER_V2  # no exception


def test_inputs_are_validated():
    f = _folds(0.0, 0.0, 61)
    bad = [dict(x) for x in f]
    bad[1]["pred_with_group"] = bad[1]["pred_with_group"].copy()
    bad[1]["pred_with_group"][3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run(bad)
    dup = [dict(x) for x in f]
    dup[0]["account_key"] = dup[0]["account_key"].copy()
    dup[0]["account_key"][1] = dup[0]["account_key"][0]
    with pytest.raises(ValueError, match="unique"):
        run(dup)
    day = [dict(x) for x in f]
    day[2]["window_start"] = [datetime(2022, 9, 9, tzinfo=UTC)] * len(day[2]["y"])  # CALIBRATION
    with pytest.raises(ValueError, match="2022-09-07"):
        run(day)
    order = [f[1], f[0], f[2]]
    with pytest.raises(ValueError, match="order"):
        run(order)
    with pytest.raises(ValueError, match="differs"):
        ig.evaluate_v2(f, FAST)  # an edited config is refused unless explicitly allowed
    loose = {**CFG, "thresholds": {**CFG["thresholds"], "n_min_pos_seen": 1}}
    r = ig.evaluate_v2(f, loose, allow_unfrozen_config=True)
    assert r["config_is_frozen"] is False and r["config_sha256"] != CFG["_sha256"]
    assert date(2022, 9, 7) == EVAL_DAYS[2].date()


def test_d_prime_squared_contrast_is_reported_not_deciding():
    r = run(_folds(0.8, 0.0, 71))
    rep = r["report_only_dprime_squared"]
    assert set(rep) == {
        "seen_gain",
        "unseen_gain",
        "contrast_unseen_minus_ratio2_seen",
        "contrast_sign_disagrees",
    }
    assert r["verdict"] == "account_recognition"
