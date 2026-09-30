"""Identity (account-recognition) guard: verdict logic on constructed data with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval import identity_guard as ig

CFG = {"min_positives": 20, "materiality_gain_over_prev": 0.05, "min_unseen_to_seen_ratio": 0.5}


def _data(n: int = 4000, prev: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prev).astype(int)
    seen = rng.random(n) < 0.5
    base = y * 0.3 + rng.random(n)  # weak shared signal
    return rng, y, seen, base


def test_gain_only_on_seen_accounts_is_account_recognition():
    rng, y, seen, base = _data()
    with_group = base + np.where(seen, y * 2.0, 0.0)  # sharp help, seen accounts only
    r = ig.evaluate(y, base, with_group, seen, CFG)
    assert r["verdict"] == "account_recognition"
    assert r["seen"]["gain"] > 0 and abs(r["unseen"]["gain"]) < 1e-12
    assert not ig.cleared(r)


def test_gain_on_both_subsets_passes():
    rng, y, seen, base = _data()
    with_group = base + y * 2.0
    r = ig.evaluate(y, base, with_group, seen, CFG)
    assert r["verdict"] == "pass" and ig.cleared(r)


def test_no_gain_is_cleared_as_no_material_gain():
    rng, y, seen, base = _data()
    r = ig.evaluate(y, base, base.copy(), seen, CFG)
    assert r["verdict"] == "no_material_gain" and ig.cleared(r)


def test_too_few_unseen_positives_is_not_cleared():
    rng, y, seen, base = _data()
    seen = seen | (y == 1)  # every positive is a seen account -> unseen has 0 positives
    r = ig.evaluate(y, base, base + y, seen, CFG)
    assert r["verdict"] == "insufficient_data" and not ig.cleared(r)


def test_verdict_switches_exactly_at_the_declared_ratio():
    _, y, seen, base = _data(n=20000, seed=1)
    with_group = base + y * np.where(seen, 0.4, 0.2)  # unseen accounts helped less
    r = ig.evaluate(y, base, with_group, seen, CFG)
    ratio = r["unseen"]["gain_over_prev"] / r["seen"]["gain_over_prev"]
    assert 0 < ratio < 1 and r["seen"]["gain_over_prev"] >= CFG["materiality_gain_over_prev"]
    below = CFG | {"min_unseen_to_seen_ratio": ratio - 0.01}
    above = CFG | {"min_unseen_to_seen_ratio": ratio + 0.01}
    assert ig.evaluate(y, base, with_group, seen, below)["verdict"] == "pass"
    assert ig.evaluate(y, base, with_group, seen, above)["verdict"] == "account_recognition"


def test_misaligned_inputs_fail_loudly():
    _, y, seen, base = _data()
    with pytest.raises(ValueError):
        ig.evaluate(y, base[:-1], base, seen, CFG)


def test_verdict_set_is_closed():
    _, y, seen, base = _data()
    assert ig.evaluate(y, base, base + y, seen, CFG)["verdict"] in ig.VERDICTS
