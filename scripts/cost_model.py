"""Dossier cost / quota model (Phase 0, task 3).

Two resource modes:
  quota -> free tier; the binding resource is requests/day (RPD, RPM, TPM), not euros.
  price -> paid tier; the binding resource is euros.

Allocation rule (phase spec):
  1. H1 = full_system + single_agent at the target Tier A size and target repeats.
  2. Secondary ablations, in config order, with what remains.
  3. 25% contingency is never allocated.
  4. If H1 does not fit, reduce: (dev on cheaper model) -> fewer configs -> fewer repeats
     (>= repeats_min) -> fewer alerts (>= n_alerts_min, the P8 power-pilot N).
Rules-only and model-only baselines make no LLM calls and are always included.

Usage: uv run python scripts/cost_model.py [--config configs/cost_model.yaml]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

MDP_MIN_ABLATIONS = 3


@dataclass(frozen=True)
class Cfg:
    name: str
    calls: int
    in_tok: int
    out_tok: int
    h1: bool

    def requests(self, n: int, r: int) -> int:
        return self.calls * n * r

    def tokens(self, n: int, r: int) -> tuple[int, int]:
        k = self.requests(n, r)
        return k * self.in_tok, k * self.out_tok


@dataclass(frozen=True)
class Design:
    n: int
    repeats: int
    configs: tuple[str, ...]
    usage: float  # requests (quota) or EUR (price)


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def build_cfgs(spec: dict) -> list[Cfg]:
    return [
        Cfg(k, v["calls"], v["in_tok"], v["out_tok"], bool(v.get("h1", False)))
        for k, v in spec["configurations"].items()
    ]


def usage_fn(prov: dict, eur_per_usd: float):
    if prov["mode"] == "quota":
        return lambda c, n, r: float(c.requests(n, r))
    pin, pout = prov["usd_per_m_in"], prov["usd_per_m_out"]

    def eur(c: Cfg, n: int, r: int) -> float:
        ti, to = c.tokens(n, r)
        return (ti * pin + to * pout) / 1e6 * eur_per_usd

    return eur


def daily_request_capacity(prov: dict, cfgs: list[Cfg]) -> float:
    """Effective requests/day = min(RPD, RPM*1440, TPM*1440 / avg tokens per request)."""
    avg_tok = sum(c.in_tok + c.out_tok for c in cfgs) / len(cfgs)
    return min(prov["rpd"], prov["rpm"] * 1440, prov["tpm"] * 1440 / avg_tok)


def allocate(cfgs: list[Cfg], use, capacity: float, ev: dict) -> tuple[Design | None, list[str]]:
    """Largest design within capacity*(1-contingency), following the spec's priority order."""
    usable = capacity * (1 - ev["contingency"])
    notes: list[str] = []
    h1 = [c for c in cfgs if c.h1]
    abl = [c for c in cfgs if not c.h1]

    def cost(cs, n, r):
        return sum(use(c, n, r) for c in cs)

    # Step 1: H1 at target; shrink repeats, then N, down to the floors.
    n, r = ev["n_alerts_target"], ev["repeats_target"]
    while cost(h1, n, r) > usable:
        if r > ev["repeats_min"]:
            r -= 1
            notes.append(f"H1 does not fit -> repeats reduced to {r}")
        elif n > ev["n_alerts_min"]:
            n = max(ev["n_alerts_min"], int(n * 0.9))
            notes.append(f"H1 does not fit -> alerts reduced to {n}")
        else:
            notes.append("H1 does not fit even at the floors (N_min, 3 repeats): UNAFFORDABLE")
            return None, notes
    used = cost(h1, n, r)
    chosen = [c.name for c in h1]
    # Step 2: ablations in priority order at the same N and repeats.
    for c in abl:
        u = use(c, n, r)
        if used + u <= usable:
            chosen.append(c.name)
            used += u
        else:
            notes.append(f"ablation '{c.name}' dropped (needs {u:,.1f}, left {usable - used:,.1f})")
    if len(chosen) - len(h1) < MDP_MIN_ABLATIONS:
        notes.append(
            f"MDP REQUIRES >= {MDP_MIN_ABLATIONS} ablations; "
            f"this design has {len(chosen) - len(h1)}"
        )
    return Design(n, r, tuple(chosen), used), notes


def min_capacity_for(cfgs, use, n, r, n_abl, contingency) -> float:
    cs = [c for c in cfgs if c.h1] + [c for c in cfgs if not c.h1][:n_abl]
    return sum(use(c, n, r) for c in cs) / (1 - contingency)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cost_model.yaml")
    a = ap.parse_args(argv)
    spec = load(Path(a.config))
    ev, sch, provs = spec["evaluation"], spec["schedule"], spec["providers"]
    fx = provs["eur_per_usd"]
    cfgs = build_cfgs(spec)
    nT, rT = ev["n_alerts_target"], ev["repeats_target"]
    nM, rM = ev["n_alerts_min"], ev["repeats_min"]

    print("=" * 78)
    print("DOSSIER COST / QUOTA MODEL  (all inputs are assumptions until measured in P8)")
    print("=" * 78)
    print(
        f"Tier A target N={nT} (floor {nM}), repeats target {rT} (floor {rM}), "
        f"contingency {ev['contingency']:.0%}, dev multiplier {ev['dev_multiplier']}x "
        f"(ASSUMPTION), dev cache hit {ev['dev_cache_hit_rate']:.0%}"
    )
    print(
        f"Eval window {sch['eval_days']} days, dev window {sch['dev_days']} days, "
        f"deadline {sch['deadline']}, budget EUR {spec['budget_eur']}\n"
    )

    # ---- per-config table
    print(f"{'config':<18}{'calls/alert':>12}{'req @target':>13}{'req @floor':>12}")
    for c in cfgs:
        print(f"{c.name:<18}{c.calls:>12}{c.requests(nT, rT):>13,}{c.requests(nM, rM):>12,}")
    full_target = sum(c.requests(nT, rT) for c in cfgs)
    print(
        f"{'ALL configs':<18}{'':>12}{full_target:>13,}"
        f"{sum(c.requests(nM, rM) for c in cfgs):>12,}\n"
    )

    # ---- free tier (quota)
    free = provs["gemini_free"]
    use_q = usage_fn(free, fx)
    per_day = daily_request_capacity(free, cfgs)
    eval_cap = per_day * sch["eval_days"]
    print(
        f"[FREE TIER: {free['model_id']}]  effective {per_day:,.0f} req/day "
        f"(RPD {free['rpd']}, RPM {free['rpm']}, TPM {free['tpm']:,} -- UNVERIFIED)"
    )
    print(
        f"Eval capacity: {eval_cap:,.0f} requests over {sch['eval_days']} days; "
        f"usable after contingency {eval_cap * (1 - ev['contingency']):,.0f}"
    )
    d, notes = allocate(cfgs, use_q, eval_cap, ev)
    for s in notes:
        print("  -", s)
    if d:
        print(
            f"  => AFFORDABLE (free): N={d.n}, repeats={d.repeats}, configs={list(d.configs)}, "
            f"{d.usage:,.0f} requests"
        )
    for k in (0, MDP_MIN_ABLATIONS):
        need = min_capacity_for(cfgs, use_q, nM, rM, k, ev["contingency"])
        print(
            f"  Days needed at floor (N={nM}, r={rM}) for H1 + {k} ablations: "
            f"{math.ceil(need / per_day)} days"
        )
    dev_need = full_target * ev["dev_multiplier"] * (1 - ev["dev_cache_hit_rate"])
    dev_cap = per_day * sch["dev_days"]
    print(
        f"Dev need ~{dev_need:,.0f} req vs dev capacity {dev_cap:,.0f} "
        f"({'OK' if dev_need <= dev_cap else 'SHORT'}; {dev_need / dev_cap:.0%} of daily cap "
        f"over the dev window)\n"
    )

    # ---- paid scenarios: free dev + paid final eval
    for key in ("gemini_flash_lite_paid", "gemini_flash_paid"):
        p = provs[key]
        use_p = usage_fn(p, fx)
        tgt = sum(use_p(c, nT, rT) for c in cfgs)
        flo = min_capacity_for(cfgs, use_p, nM, rM, MDP_MIN_ABLATIONS, 0)
        print(
            f"[PAID FINAL EVAL: {p['model_id']}]  "
            f"all {len(cfgs)} LLM configs at target: EUR {tgt:,.2f}; "
            f"MDP floor (H1+3 abl, N={nM}, r={rM}): EUR {flo:,.2f}; "
            f"budget incl. 25% contingency, dev on free tier: "
            f"EUR {tgt / (1 - ev['contingency']):,.2f} (target) / "
            f"EUR {flo / (1 - ev['contingency']):,.2f} (floor)"
        )
        full = tgt * (1 + ev["dev_multiplier"])
        print(
            f"   if dev ALSO paid on this model (3x): EUR {full / (1 - ev['contingency']):,.2f} "
            f"incl. contingency"
        )
    print()

    # ---- sensitivity: days needed on free tier
    print("Sensitivity (free tier): eval days needed for H1 + 3 ablations, incl. contingency")
    mults = (0.5, 1.0, 1.5, 2.0)
    print(f"{'N / calls x':<14}" + "".join(f"{m:>8}" for m in mults))
    for n in (50, 100, 150, 200, 300):
        row = []
        for m in mults:
            scaled = [
                Cfg(c.name, max(1, round(c.calls * m)), c.in_tok, c.out_tok, c.h1) for c in cfgs
            ]
            need = min_capacity_for(scaled, use_q, n, rM, MDP_MIN_ABLATIONS, ev["contingency"])
            row.append(math.ceil(need / per_day))
        print(f"{n:<14}" + "".join(f"{x:>8}" for x in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
