"""Review check 10a (label generosity) -- TRAIN + VALIDATION ONLY. Aggregates only.

Question: label P2-v1 makes an alert TRUE if its account touched ANY laundering txn that day, even
if the rule that fired measures something unrelated. For every TRUE alert and every rule that fired
on it, this script measures
  (a) does the rule's statistic actually involve >= 1 laundering txn?
  (b) would the rule still fire if the account's laundering txns of that day were removed?
      (counterfactual: recompute the statistic on the non-laundering legs with the repo's own
       src.alerts.rules.account_stats, then fire with the repo's src.alerts.rules.fire and the
       frozen thresholds) -> "coincidental" true positive for that rule.

Data access (hard limits, asserted):
  - transactions: only rows with 2022-09-02 00:00 UTC <= timestamp < 2022-09-08 00:00 UTC are
    ever collected (filter applied lazily before anything else);
  - alerts: only period in {TRAIN, VALIDATION} and window_start < 2022-09-08 (filtered before
    collect); labels: semi-joined to those alert_ids before collect. No TEST/CALIBRATION row is
    materialised. No eval store other than alert_labels.parquet is read.

Uses the repo's own definitions: FX (src.data.fx), point-in-time window (src.data.transactions.
visible, src.data.periods.Split), statistics (src.alerts.rules.account_stats) and firing
(src.alerts.rules.fire). The legs are rebuilt here only to carry the laundering flag; a built-in
check asserts that the recomputed statistics equal the statistics stored on every alert.

Run from the repo root (Windows or Linux):
    uv run python <path>\\label_generosity_train_val.py
    (options: --variant, --interim-dir, --out-root, --configs-dir, --rules, --split, --out)
Output: a table on stdout and a JSON file (default: label_generosity_train_val.json in the cwd).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Windows writes piped output as cp1252, which cannot encode polars table borders
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path.cwd()))  # the script may live outside the repo; run from repo root

import polars as pl  # noqa: E402

from src.alerts import rules  # noqa: E402
from src.data import fx as fxmod  # noqa: E402
from src.data.periods import load_split  # noqa: E402
from src.data.transactions import RUNTIME_COLUMNS, account_key, visible  # noqa: E402

START = datetime(2022, 9, 2, tzinfo=UTC)
CUTOFF = datetime(2022, 9, 8, tzinfo=UTC)  # nothing at or after this instant is ever collected
PERIODS = ("TRAIN", "VALIDATION")
LABEL = "is_" + "laundering"

# statistic -> column (computed below) that says "the statistic involves >= 1 laundering txn"
INVOLVES = {
    "max_txn_usd": "inv_max_leg_is_laundering",
    "volume_usd": "inv_laundering_value",
    "n_near_threshold": "inv_n_near_threshold_laundering",
    "n_distinct_senders": "inv_n_laundering_senders",
    "n_distinct_receivers": "inv_n_laundering_receivers",
    "pass_through_usd": "inv_laundering_value",
    "high_risk_channel_usd": "inv_laundering_high_risk_value",
    "n_cross_currency": "inv_n_laundering_cross",
    "n_round_amounts": "inv_n_laundering_round",
}
# statistic -> (laundering part, whole) for a "share of the statistic" summary
SHARE = {
    "volume_usd": ("inv_laundering_value", "volume_usd"),
    "high_risk_channel_usd": ("inv_laundering_high_risk_value", "high_risk_channel_usd"),
    "n_near_threshold": ("inv_n_near_threshold_laundering", "n_near_threshold"),
    "n_distinct_senders": ("inv_n_laundering_senders", "n_distinct_senders"),
    "n_distinct_receivers": ("inv_n_laundering_receivers", "n_distinct_receivers"),
}


def scan_trans_with_flag(path: Path) -> pl.LazyFrame:
    """Same columns/keys/UTC handling as src.data.transactions.scan_transactions, plus the flag.
    The time filter is the FIRST operation after the column selection."""
    return (
        pl.scan_parquet(path)
        .select([*RUNTIME_COLUMNS, LABEL])
        .with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
        .filter((pl.col("timestamp") >= START) & (pl.col("timestamp") < CUTOFF))
        .with_columns(
            account_key(pl.col("from_bank"), pl.col("from_account")).alias("from_key"),
            account_key(pl.col("to_bank"), pl.col("to_account")).alias("to_key"),
            (pl.col(LABEL) == 1).alias("laund"),
        )
        .drop(LABEL)
    )


def legs_with_flag(lf: pl.LazyFrame, cfg: dict) -> pl.LazyFrame:
    """Exactly src.alerts.rules.legs, plus the per-txn laundering flag (checked via stats below)."""
    r = cfg["rules"]
    channels = r["R07_HIGH_RISK_CHANNEL"]["channels"]
    base = lf.filter(pl.col("from_key") != pl.col("to_key")).with_columns(
        (pl.col("payment_currency") != pl.col("receiving_currency")).alias("cross"),
        pl.col("payment_format").is_in(channels).alias("high_risk"),
    )
    common = ["timestamp", "cross", "high_risk", "laund"]
    out_legs = base.select(
        pl.col("from_key").alias("account_key"),
        pl.col("to_key").alias("counterparty"),
        pl.lit("out").alias("direction"),
        pl.col("usd_paid").alias("usd"),
        pl.col("amount_paid").alias("amount"),
        *common,
    )
    in_legs = base.select(
        pl.col("to_key").alias("account_key"),
        pl.col("from_key").alias("counterparty"),
        pl.lit("in").alias("direction"),
        pl.col("usd_received").alias("usd"),
        pl.col("amount_received").alias("amount"),
        *common,
    )
    return pl.concat([out_legs, in_legs])


def involvement(legs: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    r = cfg["rules"]
    band = r["R03_STRUCTURING"]["band_usd"]
    unit = r["R09_ROUND_AMOUNTS"]["round_unit"]
    min_amt = r["R09_ROUND_AMOUNTS"]["min_amount"]
    lau = pl.col("laund")
    near = (pl.col("usd") >= band["lo"]) & (pl.col("usd") < band["hi_exclusive"])
    is_round = (pl.col("amount") >= min_amt) & (
        ((pl.col("amount") / unit).round(0) * unit - pl.col("amount")).abs() < 1e-6
    )
    return legs.group_by("account_key").agg(
        (lau & (pl.col("usd") == pl.col("usd").max())).any().alias("inv_max_leg_is_laundering"),
        pl.col("usd").filter(lau).sum().alias("inv_laundering_value"),
        (lau & near).sum().cast(pl.Int64).alias("inv_n_near_threshold_laundering"),
        pl.col("counterparty")
        .filter(lau & (pl.col("direction") == "in"))
        .n_unique()
        .cast(pl.Int64)
        .alias("inv_n_laundering_senders"),
        pl.col("counterparty")
        .filter(lau & (pl.col("direction") == "out"))
        .n_unique()
        .cast(pl.Int64)
        .alias("inv_n_laundering_receivers"),
        pl.col("usd")
        .filter(lau & pl.col("high_risk"))
        .sum()
        .alias("inv_laundering_high_risk_value"),
        (lau & pl.col("cross")).sum().cast(pl.Int64).alias("inv_n_laundering_cross"),
        (lau & is_round).sum().cast(pl.Int64).alias("inv_n_laundering_round"),
        lau.sum().cast(pl.Int64).alias("n_laundering_legs_seen_by_rules"),
    )


def pct(num: int, den: int) -> float | None:
    return round(100.0 * num / den, 2) if den else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="HI-Medium")
    ap.add_argument("--interim-dir", type=Path, default=Path("data") / "interim")
    ap.add_argument("--out-root", type=Path, default=Path("data") / "p2")
    ap.add_argument("--configs-dir", type=Path, default=Path("configs"))
    ap.add_argument("--rules", type=Path, default=Path("configs") / "p2_rules.yaml")
    ap.add_argument("--split", type=Path, default=Path("configs") / "p2_split.yaml")
    ap.add_argument("--out", type=Path, default=Path("label_generosity_train_val.json"))
    a = ap.parse_args(argv)

    split = load_split(a.split)
    rcfg = rules.load_rules(a.rules)
    thr = rules.load_thresholds(a.configs_dir / "p2_rule_thresholds.yaml")
    usd = fxmod.load(a.configs_dir / "p2_fx_usd_per_unit.yaml")
    for p in PERIODS:
        s, e = split.periods[p]
        assert START <= s and e <= CUTOFF, f"{p} {s}..{e} is outside [{START}, {CUTOFF})"
    active = [r for r in rules.rule_ids(rcfg) if thr["rules"].get(r, {}).get("active")]
    stat_of = {r: rcfg["rules"][r]["stat"] for r in active}

    base = a.out_root / a.variant
    # ---- alerts: TRAIN/VALIDATION only, filtered BEFORE collect
    alerts = (
        pl.scan_parquet(base / "runtime" / "alerts.parquet")
        .filter(
            pl.col("period").is_in(list(PERIODS))
            & (pl.col("window_start") >= START)
            & (pl.col("window_start") < CUTOFF)
        )
        .collect()
    )
    assert alerts.height > 0
    assert alerts["window_start"].max() < CUTOFF and alerts["as_of"].max() <= CUTOFF
    assert set(alerts["period"].unique()) <= set(PERIODS)
    labels = (
        pl.scan_parquet(base / "eval" / "alert_labels.parquet")
        .join(alerts.select("alert_id").lazy(), on="alert_id", how="semi")
        .select("alert_id", "is_true_positive")
        .collect()
    )
    assert labels.height == alerts.height, "labels do not match the TRAIN/VALIDATION alerts"
    tp = alerts.join(labels, on="alert_id").filter(pl.col("is_true_positive"))

    trans = fxmod.with_usd(scan_trans_with_flag(a.interim_dir / f"{a.variant}_trans.parquet"), usd)

    rows = []
    max_ts_seen = None
    for per in PERIODS:
        for ws in split.window_starts(per):
            as_of = split.as_of(ws)
            assert as_of <= CUTOFF
            keys = tp.filter(pl.col("window_start") == ws)["account_key"].to_list()
            if not keys:
                continue
            day = visible(trans, as_of, split.lookback_start(as_of))
            lg = legs_with_flag(day, rcfg).filter(pl.col("account_key").is_in(keys)).collect()
            if lg.height:
                m = lg["timestamp"].max()
                max_ts_seen = m if max_ts_seen is None else max(max_ts_seen, m)
                assert m < CUTOFF
            full = rules.account_stats(lg.lazy(), rcfg).collect()
            cf = rules.account_stats(lg.filter(~pl.col("laund")).lazy(), rcfg).collect()
            inv = involvement(lg, rcfg)
            cf = cf.select("account_key", *[pl.col(c).alias(f"cf_{c}") for c in rules.RULE_STATS])
            full = full.select(
                "account_key", *[pl.col(c).alias(f"re_{c}") for c in rules.RULE_STATS]
            )
            rows.append(
                tp.filter(pl.col("window_start") == ws)
                .join(full, on="account_key", how="left")
                .join(cf, on="account_key", how="left")
                .join(inv, on="account_key", how="left")
            )
    d = pl.concat(rows, how="vertical_relaxed")
    assert d.height == tp.height
    fill = [c for c in d.columns if c.startswith(("re_", "cf_", "inv_"))]
    d = d.with_columns(
        [pl.col(c).fill_null(False if d[c].dtype == pl.Boolean else 0) for c in fill]
    ).with_columns(pl.col("n_laundering_legs_seen_by_rules").fill_null(0))

    # ---- built-in check: recomputed statistics == the statistics stored on the alerts
    mismatch = {}
    for c in rules.RULE_STATS:
        x, y = d[c].cast(pl.Float64), d[f"re_{c}"].cast(pl.Float64)
        bad = ((x - y).abs() > 1e-9 * (x.abs() + 1.0)).sum()
        if bad:
            mismatch[c] = int(bad)
    assert not mismatch, f"recomputed statistics differ from stored alert statistics: {mismatch}"

    # ---- counterfactual firing with the repo's own fire() and the frozen thresholds
    cfst = d.select(
        "alert_id", "peer_group", *[pl.col(f"cf_{c}").alias(c) for c in rules.RULE_STATS]
    )
    cff = rules.fire(cfst, thr, rcfg).select(
        "alert_id", *[pl.col(f"fired_{r}").alias(f"cf_fired_{r}") for r in active]
    )
    d = d.join(cff, on="alert_id")
    orig = rules.fire(d.select("alert_id", "peer_group", *rules.RULE_STATS), thr, rcfg).select(
        "alert_id", "triggered_rules"
    )
    chk = d.select("alert_id", "triggered_rules").join(orig, on="alert_id", suffix="_re")
    assert (chk["triggered_rules"] == chk["triggered_rules_re"]).all(), "re-firing differs"

    out: dict = {
        "scope": "TRUE alerts of TRAIN and VALIDATION only (label P2-v1)",
        "data_window": [START.isoformat(), CUTOFF.isoformat()],
        "max_transaction_timestamp_loaded": max_ts_seen.isoformat() if max_ts_seen else None,
        "stats_recomputation_check": "recomputed statistics equal stored alert statistics",
        "definitions": {
            "involves": "the fired rule's statistic includes >= 1 laundering txn of that day "
            "(R01: the largest leg is laundering; R02: laundering value > 0; R03: >= 1 "
            "laundering txn in the band; R04/R05: >= 1 distinct sender/receiver of a "
            "laundering txn; R07: laundering Cash/Bitcoin value > 0)",
            "coincidental": "the rule still fires on the same account-day when all laundering "
            "txns are removed (repo account_stats + fire, frozen thresholds)",
        },
        "per_period": {},
    }
    for per in PERIODS:
        p = d.filter(pl.col("period") == per)
        n = p.height
        blk: dict = {
            "n_true_alerts": n,
            "n_true_alerts_no_laundering_leg_visible_to_rules": int(
                (p["n_laundering_legs_seen_by_rules"] == 0).sum()
            ),
            "per_rule": {},
        }
        all_coinc = pl.lit(True)
        any_still = pl.lit(False)
        none_involves = pl.lit(True)
        for r in active:
            st = stat_of[r]
            fired = pl.col("triggered_rules").list.contains(r)
            invc = INVOLVES[st]
            inv = pl.col(invc) if p[invc].dtype == pl.Boolean else (pl.col(invc) > 0)
            still = pl.col(f"cf_fired_{r}")
            all_coinc = all_coinc & (~fired | still)
            any_still = any_still | (fired & still)
            none_involves = none_involves & (~fired | ~inv)
            f = p.filter(fired)
            k = f.height
            n_inv = int(f.select(inv.sum()).item()) if k else 0
            n_still = int(f.select(still.sum()).item()) if k else 0
            rb = {
                "n_true_alerts_fired": k,
                "pct_statistic_involves_laundering": pct(n_inv, k),
                "pct_coincidental_still_fires_without_laundering": pct(n_still, k),
            }
            if st in SHARE and k:
                num, den = SHARE[st]
                sh = f.select(
                    (pl.col(num).cast(pl.Float64) / pl.col(den).cast(pl.Float64)).alias("s")
                )["s"]
                rb["laundering_share_of_statistic_median"] = round(float(sh.median()), 4)
                rb["laundering_share_of_statistic_mean"] = round(float(sh.mean()), 4)
            blk["per_rule"][r] = rb
        lay = p.select(
            all_coinc.sum().alias("all"),
            any_still.sum().alias("any"),
            none_involves.sum().alias("ni"),
        ).row(0, named=True)
        blk["layer"] = {
            "pct_coincidental_for_every_fired_rule": pct(int(lay["all"]), n),
            "pct_alert_still_raised_without_laundering_any_rule": pct(int(lay["any"]), n),
            "pct_no_fired_statistic_involves_laundering": pct(int(lay["ni"]), n),
        }
        out["per_period"][per] = blk

    a.out.write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(f"wrote {a.out}  (max txn timestamp loaded: {out['max_transaction_timestamp_loaded']})")
    for per, blk in out["per_period"].items():
        print(
            f"\n{per}: {blk['n_true_alerts']} true alerts "
            f"({blk['n_true_alerts_no_laundering_leg_visible_to_rules']} with no laundering leg "
            f"visible to the rules)"
        )
        print(f"  {'rule':<26}{'n':>6}{'%involves':>11}{'%coincid.':>11}{'share_med':>11}")
        for r, rb in blk["per_rule"].items():
            print(
                f"  {r:<26}{rb['n_true_alerts_fired']:>6}"
                f"{str(rb['pct_statistic_involves_laundering']):>11}"
                f"{str(rb['pct_coincidental_still_fires_without_laundering']):>11}"
                f"{str(rb.get('laundering_share_of_statistic_median', '')):>11}"
            )
        print("  layer:", blk["layer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
