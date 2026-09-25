from pathlib import Path

from scripts.cost_model import Cfg, allocate, build_cfgs, load, main, usage_fn

EV = {
    "n_alerts_target": 100,
    "n_alerts_min": 50,
    "repeats_target": 5,
    "repeats_min": 3,
    "contingency": 0.25,
}
CFGS = [
    Cfg("full", 10, 1, 1, True),
    Cfg("single", 5, 1, 1, True),
    Cfg("a1", 10, 1, 1, False),
    Cfg("a2", 10, 1, 1, False),
]
Q = usage_fn({"mode": "quota"}, 1.0)


def test_runs_on_repo_config(capsys):
    assert main(["--config", "configs/cost_model.yaml"]) == 0
    assert "AFFORDABLE" in capsys.readouterr().out


def test_contingency_never_spent():
    d, _ = allocate(CFGS, Q, 100_000, EV)
    assert d.usage <= 100_000 * 0.75


def test_h1_first_then_ablations_in_order():
    # H1 at target = 15*100*5 = 7500; each ablation 5000. usable 12,600 -> H1 + a1 only
    d, notes = allocate(CFGS, Q, 16_800, EV)
    assert d.configs == ("full", "single", "a1") and (d.n, d.repeats) == (100, 5)
    assert any("MDP REQUIRES" in n for n in notes)


def test_reduces_repeats_before_alerts_and_respects_floors():
    d, _ = allocate(CFGS, Q, 15 * 100 * 3 / 0.75, EV)
    assert (d.n, d.repeats) == (100, 3)
    d, notes = allocate(CFGS, Q, 10, EV)
    assert d is None and any("UNAFFORDABLE" in n for n in notes)


def test_price_mode_arithmetic():
    use = usage_fn({"mode": "price", "usd_per_m_in": 1.0, "usd_per_m_out": 2.0}, 1.0)
    # 1 call * 1 alert * 1 repeat, 1e6 in + 1e6 out tokens -> $3
    assert abs(use(Cfg("x", 1, 1_000_000, 1_000_000, True), 1, 1) - 3.0) < 1e-9


def test_config_has_h1_pair():
    spec = load(Path("configs/cost_model.yaml"))
    assert {c.name for c in build_cfgs(spec) if c.h1} == {"full_system", "single_agent"}
