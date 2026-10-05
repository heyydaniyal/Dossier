"""Reviewer A, item 10b: why does the TRAIN FX fit have max |log residual| 0.0125?

TRAIN ONLY: reads transactions with 2022-09-02 00:00 <= ts < 2022-09-06 00:00 UTC (the TRAIN
period, configs/p2_split.yaml). The filter is applied before any aggregation, and the script
asserts that no collected row is at or after 2022-09-06 (hence never >= 2022-09-08).
The label column is never selected (src.data.transactions.scan_transactions allow-list).

Run from the repo root on the machine that holds data/interim:
    uv run python scripts/p2/review/fx_residual_diagnostic.py [--interim-dir ...] [--variant ...]
Prints:
  1. the committed table vs a re-fit (must match configs/p2_fx_usd_per_unit.yaml),
  2. every currency pair: n, median log-ratio, fitted log-ratio, residual, IQR of the log-ratio,
     share of rows sitting exactly on the modal ratio (rounding pile-ups), sorted by |residual|,
  3. the 25 individual rows that deviate most from the fitted rate (amounts, currencies, format),
  4. significant-digit profile of Bitcoin amounts by direction,
  5. alternative Bitcoin rates: joint fit (committed), BTC-paying pairs only, BTC->USD only,
     and a re-fit that drops fiat->Bitcoin pairs; relative change of every currency.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src.data import fx as fxmod  # noqa: E402
from src.data.transactions import interim_path, scan_transactions  # noqa: E402

T0 = datetime(2022, 9, 2, tzinfo=UTC)
T1 = datetime(2022, 9, 6, tzinfo=UTC)
HARD_LIMIT = datetime(2022, 9, 8, tzinfo=UTC)
assert T1 <= HARD_LIMIT


def fit(pairs: pl.DataFrame) -> tuple[dict[str, float], np.ndarray]:
    cur = sorted(set(pairs["payment_currency"]) | set(pairs["receiving_currency"]))
    others = [c for c in cur if c != "US Dollar"]
    idx = {c: i for i, c in enumerate(others)}
    a = np.zeros((pairs.height, len(others)))
    y = pairs["m"].to_numpy()
    w = np.sqrt(pairs["n"].to_numpy().astype(float))
    cur = zip(pairs["payment_currency"], pairs["receiving_currency"], strict=True)
    for r, (p, q) in enumerate(cur):
        if p != "US Dollar":
            a[r, idx[p]] += 1.0
        if q != "US Dollar":
            a[r, idx[q]] -= 1.0
    x, *_ = np.linalg.lstsq(a * w[:, None], y * w, rcond=None)
    rates = {"US Dollar": 1.0} | {c: math.exp(x[idx[c]]) for c in others}
    return rates, a @ x - y


def sig_digits(values: np.ndarray) -> np.ndarray:
    """Smallest number of significant digits that reproduces each amount (to 1e-9 relative)."""
    out = np.full(values.shape, 16, dtype=np.int64)
    mag = np.floor(np.log10(values))
    for k in range(15, 0, -1):
        scale = 10.0 ** (k - 1 - mag)
        ok = np.abs(np.round(values * scale) / scale - values) <= 1e-9 * values
        out[ok] = k
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim-dir", type=Path, default=ROOT / "data" / "interim")
    ap.add_argument("--variant", default="HI-Medium")
    args = ap.parse_args()
    pl.Config.set_tbl_rows(250)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_fmt_str_lengths(40)

    lf = scan_transactions(interim_path(args.interim_dir, args.variant, "trans"))
    train = (
        lf.filter((pl.col("timestamp") >= T0) & (pl.col("timestamp") < T1))
        .filter(pl.col("payment_currency") != pl.col("receiving_currency"))
        .select(
            "timestamp",
            "payment_currency",
            "receiving_currency",
            "amount_paid",
            "amount_received",
            "payment_format",
        )
        .collect()
    )
    assert train.height > 0
    assert train["timestamp"].max() < T1 and train["timestamp"].min() >= T0, "non-TRAIN row loaded"
    assert train["timestamp"].max() < HARD_LIMIT
    print(
        f"TRAIN cross-currency rows: {train.height:,}  ts range {train['timestamp'].min()} .. "
        f"{train['timestamp'].max()}"
    )

    # 1. committed table vs re-fit with the production function
    committed = fxmod.load(ROOT / "configs" / "p2_fx_usd_per_unit.yaml")
    refit = fxmod.derive_usd_per_unit(lf, T0, T1)
    print("\n1. committed vs re-fit (production code), relative difference:")
    for c in sorted(set(committed) | set(refit["usd_per_unit"])):
        a, b = committed.get(c), refit["usd_per_unit"].get(c)
        rel = f"{b / a - 1:+.2e}" if a and b else "MISSING"
        print(f"   {c:<18} {a!s:>16} {b!s:>16} {rel}")
    print(f"   n_pairs {refit['n_pairs']}  max_abs_log_residual {refit['max_abs_log_residual']}")

    t = train.with_columns((pl.col("amount_received") / pl.col("amount_paid")).log().alias("lr"))
    pairs = (
        t.group_by("payment_currency", "receiving_currency")
        .agg(
            pl.col("lr").median().alias("m"),
            pl.len().alias("n"),
            (pl.col("lr").quantile(0.75) - pl.col("lr").quantile(0.25)).alias("iqr_lr"),
            pl.col("lr").min().alias("lr_min"),
            pl.col("lr").max().alias("lr_max"),
            (pl.col("lr").round(6).mode().first()).alias("mode_lr"),
        )
        .sort("payment_currency", "receiving_currency")
    )
    mode_share = (
        t.join(
            pairs.select("payment_currency", "receiving_currency", "mode_lr"),
            on=["payment_currency", "receiving_currency"],
        )
        .group_by("payment_currency", "receiving_currency")
        .agg((pl.col("lr").round(6) == pl.col("mode_lr")).mean().alias("share_at_mode"))
    )
    rates, resid = fit(pairs)
    pairs = (
        pairs.with_columns(pl.Series("resid", resid), (pl.col("m") + pl.Series(resid)).alias("fit"))
        .join(mode_share, on=["payment_currency", "receiving_currency"])
        .with_columns(pl.col("resid").abs().alias("abs_resid"))
        .sort("abs_resid", descending=True)
    )
    print("\n2. per-pair residuals (log units; 0.01 = 1%), sorted by |residual|:")
    print(
        pairs.select(
            "payment_currency",
            "receiving_currency",
            "n",
            "m",
            "fit",
            "resid",
            "iqr_lr",
            "lr_min",
            "lr_max",
            "share_at_mode",
        )
    )
    print("   residual quantiles:", np.quantile(np.abs(resid), [0.5, 0.9, 0.99, 1.0]).round(6))
    print("   |resid| > 1e-3 by whether Bitcoin is involved:")
    print(
        pairs.with_columns(
            (
                pl.col("payment_currency").eq("Bitcoin")
                | pl.col("receiving_currency").eq("Bitcoin")
            ).alias("btc"),
            (pl.col("abs_resid") > 1e-3).alias("big"),
        )
        .group_by("btc", "big")
        .agg(pl.len(), pl.col("n").sum().alias("rows"))
        .sort("btc", "big")
    )

    # 3. worst individual rows vs the fitted rate
    lr_fit = {c: math.log(v) for c, v in rates.items()}
    worst = (
        t.with_columns(
            (
                pl.col("payment_currency").replace_strict(lr_fit, return_dtype=pl.Float64)
                - pl.col("receiving_currency").replace_strict(lr_fit, return_dtype=pl.Float64)
            ).alias("lr_fit")
        )
        .with_columns((pl.col("lr") - pl.col("lr_fit")).alias("dev"))
        .with_columns(pl.col("dev").abs().alias("abs_dev"))
        .sort("abs_dev", descending=True)
    )
    print("\n3. 25 rows furthest from the fitted rate:")
    print(
        worst.head(25).select(
            "timestamp",
            "payment_currency",
            "amount_paid",
            "receiving_currency",
            "amount_received",
            "payment_format",
            "dev",
        )
    )
    print("   share of rows with |dev| > 1e-3:", round(float((worst["abs_dev"] > 1e-3).mean()), 5))

    # 4. Bitcoin amount precision
    print("\n4. significant digits of Bitcoin amounts (how coarsely BTC is rounded):")
    for side, col in (("BTC received", "amount_received"), ("BTC paid", "amount_paid")):
        cur = "receiving_currency" if col == "amount_received" else "payment_currency"
        b = t.filter(pl.col(cur) == "Bitcoin")
        b = b.with_columns(pl.Series("sig_digits", sig_digits(b[col].to_numpy())))
        print(f"   {side}: n={b.height:,}")
        print(b.group_by("sig_digits").len().sort("sig_digits"))

    # 5. alternative Bitcoin rates
    def rate_from(df: pl.DataFrame) -> float | None:
        return None if df.height == 0 else float(np.exp(df["m"].median()))

    btc_paid = pairs.filter(pl.col("payment_currency") == "Bitcoin")
    alt = [
        math.exp(m + math.log(rates[q]))
        for m, q in zip(btc_paid["m"], btc_paid["receiving_currency"], strict=True)
    ]
    direct = pairs.filter(
        (pl.col("payment_currency") == "Bitcoin") & (pl.col("receiving_currency") == "US Dollar")
    )
    print("\n5. Bitcoin USD rate:")
    print(f"   joint WLS (this re-fit)            {rates['Bitcoin']:.2f}")
    print(f"   committed table                    {committed['Bitcoin']:.2f}")
    print(f"   BTC->USD median only               {rate_from(direct)}")
    print(f"   BTC-paying pairs, median of rates  {float(np.median(alt)) if alt else None}")
    no_fiat_to_btc = pairs.filter(pl.col("receiving_currency") != "Bitcoin")
    rates2, resid2 = fit(no_fiat_to_btc)
    print(
        f"   re-fit without fiat->BTC pairs     {rates2['Bitcoin']:.2f}  "
        f"(max |resid| {np.abs(resid2).max():.2e})"
    )
    print("   relative change of every currency when fiat->BTC pairs are dropped:")
    for c in sorted(rates):
        print(f"     {c:<18} {rates2[c] / rates[c] - 1:+.2e}")


if __name__ == "__main__":
    main()
