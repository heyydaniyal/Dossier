"""Review check (TRAIN and VALIDATION rows ONLY: window_start / timestamp < 2022-09-08).

Run from the repo root on the machine that holds data/p2:
    uv run python scripts/p2/review/rb_check_train_val.py  > rb_check_output.txt

Outputs (aggregates only):
  1. composition of positive alerts behind the disposition "hard" class (unattributed-only /
     RANDOM-BIPARTITE / easy), split by fan-in (R04) vs other rules;
  2. share of laundering legs carrying an attempt_id (sanity of the patterns join);
  3. identity guard on KYC with paired-bootstrap CIs for the seen / unseen gains;
  4. identity guard on a PURE-NOISE feature group (20 seeds): how often does a group with no
     information get 'account_recognition'?
Never reads CALIBRATION or TEST: every frame is filtered lazily before collect and asserted.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from scripts.p2 import kyc_leakage as kl  # noqa: E402
from src.alerts import rules  # noqa: E402
from src.data import kyc as kycmod  # noqa: E402
from src.data.hashing import keyed_u64, uniforms  # noqa: E402
from src.eval import identity_guard as ig  # noqa: E402

CUTOFF = datetime(2022, 9, 8, tzinfo=UTC)  # VALIDATION end; nothing at or after this is loaded

BASE = Path(os.environ.get("RB_BASE", ROOT / "data" / "p2" / "HI-Medium"))
HARD = {"RANDOM", "BIPARTITE"}


def guard(df: pl.DataFrame, col: str) -> None:
    if df.height and df[col].max() >= CUTOFF:
        raise SystemExit(f"ABORT: a row with {col} >= {CUTOFF} was loaded")


alerts = (
    pl.scan_parquet(BASE / "runtime" / "alerts.parquet")
    .filter(pl.col("window_start") < CUTOFF)
    .filter(pl.col("period").is_in(["TRAIN", "VALIDATION"]))
    .collect()
)
guard(alerts, "window_start")
labels = (
    pl.scan_parquet(BASE / "eval" / "alert_labels.parquet")
    .join(alerts.select("alert_id").lazy(), on="alert_id", how="semi")
    .collect()
)
assert labels.height == alerts.height
legs = (
    pl.scan_parquet(BASE / "eval" / "laundering_legs.parquet")
    .filter(pl.col("timestamp") < CUTOFF)
    .filter(pl.col("window_start") >= datetime(2022, 9, 2, tzinfo=UTC))  # drop burn-in
    .collect()
)
guard(legs, "timestamp")
guard(legs, "window_start")

# ---- 1. hard-case composition
j = alerts.select("alert_id", "period", "triggered_rules").join(labels, on="alert_id")
p = j.filter(pl.col("is_true_positive"))


def cat(ty: list[str], unat: bool) -> str:
    s = set(ty)
    if not s:
        return "unattributed_only"
    if s <= HARD:
        return "random_bipartite" + ("+unattributed" if unat else "")
    return "easy"


p = p.with_columns(
    pl.struct("typologies", "has_unattributed")
    .map_elements(lambda r: cat(r["typologies"], r["has_unattributed"]), return_dtype=pl.String)
    .alias("cat"),
    pl.col("triggered_rules").list.contains("R04_FAN_IN").alias("fan_in"),
)
print("1. positive alerts by hard-class category (TRAIN+VALIDATION)")
print(p.group_by("period", "fan_in", "cat").len().sort("period", "fan_in", "cat"))
print("   share hard:", round(1 - (p["cat"] == "easy").mean(), 4), "n_pos:", p.height)

# ---- 2. patterns join sanity
print(
    "2. laundering legs with attempt_id (TRAIN+VALIDATION windows):",
    round(legs["attempt_id"].is_not_null().mean(), 4),
    "of",
    legs.height,
)
print(legs.group_by("typology").len().sort("len", descending=True))

# ---- 3 / 4. identity guard
kcfg = kycmod.load_kyc_config()
kyc = pl.read_parquet(BASE / "runtime" / "kyc.parquet")
rcfg = rules.load_rules()
ids = rules.rule_ids(rcfg)
tr, va = kl._prepare(alerts, labels, kyc, kcfg, ids)
assert set(tr["period"]) == {"TRAIN"} and set(va["period"]) == {"VALIDATION"}
seen = np.isin(va["account_key"].to_numpy(), tr["account_key"].unique().to_numpy())
ytr = tr["is_true_positive"].to_numpy().astype(int)
yva = va["is_true_positive"].to_numpy().astype(int)
specs = kl._feature_sets(kcfg, ids)
pred = {}
for name in ("txn", "txn_kyc_all"):
    xtr, xva, ic = kl._matrix(tr, va, *specs[name])
    pred[name] = kl._fit_predict(xtr, ytr, xva, ic)
th = kcfg["identity_guard"]["thresholds"]
res = ig.evaluate(yva, pred["txn"], pred["txn_kyc_all"], seen, th)
print(
    "3. KYC guard point estimate:",
    res["verdict"],
    {s: res[s].get("gain_over_prev") for s in ("seen", "unseen")},
)
rng = np.random.default_rng(0)
boot = {"seen": [], "unseen": []}
from sklearn.metrics import average_precision_score as aps  # noqa: E402

for _ in range(2000):
    for s, m in (("seen", seen), ("unseen", ~seen)):
        idx = rng.choice(np.flatnonzero(m), m.sum(), replace=True)
        if yva[idx].sum() == 0 or yva[idx].sum() == len(idx):
            continue
        g = aps(yva[idx], pred["txn_kyc_all"][idx]) - aps(yva[idx], pred["txn"][idx])
        boot[s].append(g / yva[idx].mean())
for s in boot:
    if not boot[s]:
        print(f"   {s}: no bootstrap samples")
        continue
    lo, hi = np.percentile(boot[s], [2.5, 97.5])
    print(f"   {s} gain/prev 95% CI [{lo:.3f}, {hi:.3f}]  (n_boot {len(boot[s])})")

print("4. guard verdicts for a pure-noise feature group (20 seeds)")
c = Counter()
xtr0, xva0, ic0 = kl._matrix(tr, va, *specs["txn"])
for seed in range(20):
    ntr = uniforms(keyed_u64(tr["alert_id"].to_list(), "rb-noise", seed), "x")
    nva = uniforms(keyed_u64(va["alert_id"].to_list(), "rb-noise", seed), "x")
    pw = kl._fit_predict(
        np.column_stack([xtr0, ntr]), ytr, np.column_stack([xva0, nva]), np.r_[ic0, False]
    )
    r = ig.evaluate(yva, pred["txn"], pw, seen, th)
    c[r["verdict"]] += 1
    print(
        f"   seed {seed}: {r['verdict']}  seen {r['seen'].get('gain_over_prev')}  "
        f"unseen {r['unseen'].get('gain_over_prev')}"
    )
print("   verdict counts:", dict(c))
