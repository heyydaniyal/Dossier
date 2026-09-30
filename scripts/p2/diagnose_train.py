"""P2 diagnostic on TRAIN ONLY: how much signal does each rule carry, at every threshold level?

Why: the pre-declared calibration (one global tau, precision closest to 5%) picked tau=0.99995,
which is inside the precision band but alerts on only 0.9% of TRAIN positive account-days
(~355 alerts/day). That cannot meet the pre-TEST feasibility gates. Before revising the procedure,
print the per-rule facts. Reads TRAIN statistics and TRAIN labels only (data-access matrix:
threshold selection uses TRAIN). Never reads VALIDATION, CALIBRATION or TEST.

Usage (repo root, after `run_p2 calibrate`):
    uv run python -m scripts.p2.diagnose_train
Writes docs/p2/p2_train_diagnostics.json (aggregates only) and prints a summary table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from scripts.p2 import calibrate as cal
from scripts.p2.run_p2 import ROOT, Paths, _labelled_stats, _n_pos, _sig, write_text_lf
from src.alerts import rules
from src.data.periods import load_split

TAUS = [0.99999, 0.99995, 0.9999, 0.9998, 0.9995, 0.999, 0.998, 0.995, 0.99, 0.98, 0.95, 0.9, 0.0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="HI-Medium")
    ap.add_argument("--out-root", type=Path, default=ROOT / "data" / "p2")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "p2" / "p2_train_diagnostics.json")
    args = ap.parse_args(argv)

    split = load_split()
    cfg = rules.load_rules()
    P = Paths(args.out_root, args.variant, ROOT / "configs")
    pos = pl.read_parquet(P.pos)
    st = _labelled_stats(P, split, "TRAIN", pos)  # TRAIN only
    n_pos = _n_pos(pos, split, "TRAIN")
    a = cal.Arrays(st, cfg)
    base = float(a.pos.sum()) / a.n  # positive rate among ACTIVE account-days (the rule population)
    ids = rules.rule_ids(cfg)
    out: dict = {
        "scope": "TRAIN only",
        "n_active_account_days": a.n,
        "n_positive_account_days": n_pos,
        "n_positive_active_account_days": int(a.pos.sum()),
        "base_rate": _sig(base),
        "per_rule": {},
    }
    print(f"TRAIN: {a.n:,} active account-days, {n_pos:,} positive, base rate {base:.4%}\n")
    head = ["rule".ljust(26), "tau".rjust(8), "thr".rjust(12), "alerts/day".rjust(11)]
    print(" ".join(head + ["prec".rjust(7), "lift".rjust(6), "recall".rjust(7)]))
    for r in ids:
        rows = []
        for tau in TAUS:
            only = {x: x == r for x in ids}
            thr = cal.thresholds_at(a, cfg, dict.fromkeys(ids, tau), only)
            m = cal.evaluate_arrays(a, thr, cfg, n_pos, detail=False)["per_rule"][r]
            spec = thr["rules"][r]
            t = spec.get("threshold", spec.get("default"))
            prec = m["precision"]
            row = {
                "tau": tau,
                "threshold": t,
                "by_peer": spec.get("by_peer"),
                "n_alerts": m["n_alerts"],
                "n_true": m["n_true"],
                "precision": prec,
                "lift": _sig(prec / base) if prec is not None else None,
                "recall": m["recall"],
            }
            rows.append(row)
            print(
                f"{r:26s} {tau:8.5f} {t:12.6g} {m['n_alerts'] / 4:11.0f} "
                f"{(prec or 0):7.2%} {(row['lift'] or 0):6.1f} {(m['recall'] or 0):7.2%}"
            )
        print()
        out["per_rule"][r] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(args.out, json.dumps(_sig(out), indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
