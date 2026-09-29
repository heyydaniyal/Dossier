"""Phase 1 data audit driver.

Runs, per variant: input verification + Parquet conversion, then the analyses in
scripts/audit/analyses.py. Asserts internal consistency of every reported fact, rounds floats
to 10 significant digits (streaming aggregation order must not change the file), and writes
one JSON of AGGREGATES ONLY (safe to commit).

Usage (repo root):
    uv run python -m scripts.audit.run_audit                      # first run: writes results
    uv run python -m scripts.audit.run_audit --verify             # re-run: must reproduce exactly

Options: --variants HI-Small (fast dev run), --force-convert (rebuild Parquet),
         --skip-trees (skip the separability trees; for quick dev runs only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from scripts.audit import analyses as an
from scripts.audit.convert import AuditError, convert_variant, load_sources

AUDIT_VERSION = "1.0.0"
DEFAULT_VARIANTS = ("HI-Medium", "LI-Medium", "HI-Small")
DEFAULT_OUT = Path("docs/audit/p1_audit_results.json")


def _round(o):
    if isinstance(o, float):
        # NaN/inf are not valid JSON: report them as null (never silently as a number)
        return None if (o != o or o in (float("inf"), float("-inf"))) else float(f"{o:.10g}")
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_round(v) for v in o]
    return o


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AuditError(f"fact check failed: {msg}")


def audit_variant(
    variant: str, raw_dir: Path, interim: Path, sources: dict, force: bool, skip_trees: bool
) -> dict:
    t0 = time.perf_counter()
    paths = convert_variant(variant, raw_dir, interim, sources, force=force)
    t = pl.scan_parquet(paths["trans"])
    a = pl.scan_parquet(paths["accounts"])
    p = pl.read_parquet(paths["patterns"])
    src = sources[f"{variant}_Trans.csv"]
    psrc = sources[f"{variant}_Patterns.txt"]
    r: dict = {"inputs": {k: str(v.name) for k, v in paths.items()}}

    step = lambda s: print(f"  [{variant}] {s} ...", flush=True)  # noqa: E731
    step("time audit")
    r["time"], bounds = an.time_audit(t)
    step("schema audit")
    r["schema"] = an.schema_audit(t)
    step("duplicates")
    r["duplicates"] = an.duplicates(t)
    step("keys")
    r["keys"] = an.key_audit(t, a)
    step("labels by period")
    r["labels_by_period"] = an.labels_by_period(t, bounds)
    step("patterns vs transactions")
    r["patterns_vs_trans"] = an.patterns_vs_trans(t, p, bounds)
    step("typologies")
    r["typologies"] = an.typology_audit(p, bounds)
    step("laundering accounts")
    r["laundering_accounts"] = an.laundering_accounts(t, p, bounds)
    step("label rates by category (pre-holdout)")
    r["label_rates_pre_holdout"] = an.label_rates_by_category(t, bounds)
    step("currency rates (pre-holdout)")
    r["currency_pre_holdout"] = an.currency_rates(t, bounds["span_start"], bounds["holdout_start"])
    if not skip_trees:
        step("trivial separability (pre-holdout)")
        r["separability"] = an.separability(t, bounds)
    step("identifier leakage (pre-holdout)")
    r["identifier_leakage"] = an.identifier_leakage(t, a, bounds)
    step("memory footprint")
    r["footprint"] = an.footprint(t, r["schema"]["n_rows"])
    r["footprint"]["parquet_mb"] = {k: v.stat().st_size / 1e6 for k, v in paths.items()}

    # ---- every stated fact must be internally consistent
    s = r["schema"]
    check(
        s["n_rows"] == src["n_data_rows"], f"{variant} rows {s['n_rows']} != {src['n_data_rows']}"
    )
    check(all(v == 0 for v in s["nulls"].values()), f"{variant} nulls present: {s['nulls']}")
    check(s["n_label_not_0_1"] == 0, f"{variant} label outside {{0,1}}")
    check(sum(d["n"] for d in r["time"]["daily"]) == s["n_rows"], f"{variant} daily counts")
    lp = r["labels_by_period"]
    check(sum(v["n"] for v in lp.values()) == s["n_rows"], f"{variant} period rows")
    check(
        sum(v["n_laundering"] for v in lp.values()) == s["n_laundering"], f"{variant} period labels"
    )
    pv = r["patterns_vs_trans"]
    check(pv["n_pattern_txns"] == psrc["n_pattern_txns"], f"{variant} pattern txns")
    check(r["typologies"]["n_attempts"] == psrc["n_attempts"], f"{variant} attempts")
    check(
        pv["n_laundering_rows_attributed_to_a_pattern"] + pv["n_laundering_rows_not_in_any_pattern"]
        == s["n_laundering"],
        f"{variant} attribution sums",
    )
    check(pv["n_laundering_rows_total"] == s["n_laundering"], f"{variant} laundering total")
    r["_elapsed_s"] = round(time.perf_counter() - t0, 1)
    return r


def compare(new: dict, old: dict, path: str = "") -> list[str]:
    diffs: list[str] = []
    if isinstance(new, dict) and isinstance(old, dict):
        for k in sorted(set(new) | set(old)):
            if k in ("run_info", "_elapsed_s"):
                continue
            if k not in new or k not in old:
                diffs.append(f"{path}/{k}: present in only one run")
            else:
                diffs += compare(new[k], old[k], f"{path}/{k}")
    elif isinstance(new, list) and isinstance(old, list):
        if len(new) != len(old):
            diffs.append(f"{path}: length {len(new)} != {len(old)}")
        else:
            for i, (x, y) in enumerate(zip(new, old, strict=True)):
                diffs += compare(x, y, f"{path}[{i}]")
    elif new != old:
        diffs.append(f"{path}: {old!r} -> {new!r}")
    return diffs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dossier P1 data audit")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    ap.add_argument("--sources", type=Path, default=Path("configs/data_sources.yaml"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    ap.add_argument(
        "--verify", action="store_true", help="compare to existing --out; do not overwrite"
    )
    ap.add_argument("--force-convert", action="store_true")
    ap.add_argument("--skip-trees", action="store_true")
    args = ap.parse_args(argv)

    sources = load_sources(args.sources)
    print(f"Dossier P1 audit v{AUDIT_VERSION} (polars {pl.__version__})")
    t0 = time.perf_counter()
    results: dict = {"audit_version": AUDIT_VERSION, "variants": {}}
    timings = {}
    for v in args.variants:
        print(f"\n== {v}")
        res = audit_variant(
            v, args.raw_dir, args.interim_dir, sources, args.force_convert, args.skip_trees
        )
        timings[v] = res.pop("_elapsed_s")
        results["variants"][v] = res
    results = _round(results)
    results["run_info"] = {
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "polars": pl.__version__,
        "elapsed_s_per_variant": timings,
        "elapsed_s_total": round(time.perf_counter() - t0, 1),
        "code_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(Path(__file__).parent.glob("*.py"))
        },
    }

    if args.verify:
        old = json.loads(args.out.read_text(encoding="utf-8"))
        diffs = compare(results, old)
        tmp = args.out.with_suffix(".verify.json")
        tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if diffs:
            print(f"\nREPRODUCIBILITY FAILED: {len(diffs)} differences (new run in {tmp})")
            for d in diffs[:40]:
                print("  " + d)
            return 1
        print(f"\nREPRODUCED: every number matches {args.out}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out} in {time.perf_counter() - t0:,.0f} s")
    for v, res in results["variants"].items():
        tm, s = res["time"], res["schema"]
        sep = res.get("separability", {}).get("models", {})
        d1 = sep.get("depth_1", {}).get("pr_auc_eval")
        d3 = sep.get("depth_3", {}).get("pr_auc_eval")
        print(
            f"  {v}: {s['n_rows']:,} rows | complete days {tm['complete_span_first_day']}.."
            f"{tm['complete_span_last_day']} ({tm['n_complete_days']}) | laundering rate "
            f"{s['laundering_rate_overall']:.5f} | holdout from "
            f"{tm['provisional_holdout']['holdout_start']}"
            f" | tree PR-AUC d1={d1} d3={d3}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
