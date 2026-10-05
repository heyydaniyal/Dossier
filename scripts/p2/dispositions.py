"""P2 task 9: simulated historical analyst dispositions. Offline; reads alert labels.

Inputs are ground truth + observable alert features (number of triggered rules, which rules)
+ seeded keyed-hash noise. Never model scores or agent outputs (no import of src.models,
src.agents, src.tools: checked by
tests/test_p2_generators.py::test_disposition_generator_never_uses_scores_or_agents).
Output rows carry exactly the HistoricalDisposition fields.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import polars as pl

from src.contracts.models import HistoricalDisposition
from src.data.hashing import keyed_u64, uniforms

DISPOSITION_COLUMNS = ["alert_id", "account_key", "disposition", "closed_at"]


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def simulate(alerts: pl.DataFrame, labels: pl.DataFrame, cfg: dict) -> tuple[pl.DataFrame, dict]:
    em = cfg["error_model"]
    hc = em["hard_cases"]
    a = (
        alerts.filter(pl.col("period").is_in(cfg["periods"]))
        .select(
            "alert_id", "account_key", "period", "as_of", "triggered_rules", "n_rules_triggered"
        )
        .join(
            labels.select("alert_id", "is_true_positive", "typologies"), on="alert_id", how="left"
        )
        .sort("alert_id")
    )
    if a["is_true_positive"].null_count():
        raise RuntimeError("alerts without labels")
    hard_set = set(hc["hard_typologies"])
    tp = a["is_true_positive"].to_numpy()
    hard = np.array(
        [
            bool(t) and set(ty) <= hard_set
            for t, ty in zip(tp, a["typologies"].to_list(), strict=True)
        ]
    )
    near = a["triggered_rules"].list.contains(hc["near_threshold_rule"]).to_numpy()
    p0 = np.where(
        tp,
        np.where(hard, hc["hard_sensitivity"], em["sensitivity"]),
        np.where(near, hc["near_threshold_false_confirm_rate"], em["false_confirm_rate"]),
    ).astype(float)
    nr = a["n_rules_triggered"].to_numpy().astype(float)
    p = 1.0 / (1.0 + np.exp(-(_logit(p0) + em["anchoring_beta"] * (nr - 1.0))))

    h = keyed_u64(a["alert_id"].to_list(), "disposition", cfg["seed"])
    confirmed = uniforms(h, "confirm") < p

    d = cfg["handling_delay_hours"]
    nd = NormalDist()
    ud = np.clip(uniforms(h, "delay"), 1e-12, 1 - 1e-12)
    z = np.array([nd.inv_cdf(float(x)) for x in ud])
    med = np.where(confirmed, d["confirmed_suspicious"]["median"], d["closed_legitimate"]["median"])
    sig = np.where(confirmed, d["confirmed_suspicious"]["sigma"], d["closed_legitimate"]["sigma"])
    hours = np.clip(med * np.exp(sig * z), d["min"], d["max"])
    minutes = np.floor(hours * 60).astype(np.int64)

    out = a.select(
        "alert_id",
        "account_key",
        pl.Series(
            "disposition",
            np.where(confirmed, "confirmed_suspicious", "closed_legitimate").tolist(),
            dtype=pl.String,
        ),
        (pl.col("as_of") + pl.Series(minutes).cast(pl.Int64) * pl.duration(minutes=1)).alias(
            "closed_at"
        ),
    )
    if (out["closed_at"] <= a["as_of"]).any():
        raise RuntimeError("a disposition closes at or before its alert's as_of")

    correct = confirmed == tp

    def grp(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        if not n:
            return {"n": 0}
        return {
            "n": n,
            "n_true_positive": int(tp[mask].sum()),
            "accuracy": _sig(correct[mask].mean()),
            "confirm_rate": _sig(confirmed[mask].mean()),
            "false_negative_rate": _sig((~confirmed[mask & tp]).mean())
            if (mask & tp).any()
            else None,
            "false_positive_rate": _sig(confirmed[mask & ~tp].mean())
            if (mask & ~tp).any()
            else None,
        }

    period = a["period"].to_numpy()
    audit = {
        "overall": grp(np.ones(len(tp), bool)),
        "by_period": {pp: grp(period == pp) for pp in cfg["periods"]},
        "hard_H1_shapeless_laundering": grp(hard),
        "easy_laundering": grp(tp & ~hard),
        "near_threshold_H2": grp(near),
        "by_n_rules": {
            "1": grp(nr == 1),
            "2": grp(nr == 2),
            "3+": grp(nr >= 3),
        },
        "confirmed_precision": _sig(tp[confirmed].mean()) if confirmed.any() else None,
        "delay_hours_median": {
            "confirmed_suspicious": _sig(float(np.median(minutes[confirmed])) / 60)
            if confirmed.any()
            else None,
            "closed_legitimate": _sig(float(np.median(minutes[~confirmed])) / 60)
            if (~confirmed).any()
            else None,
        },
    }
    return out.select(DISPOSITION_COLUMNS), audit


def _sig(x: float) -> float:
    return float(f"{float(x):.10g}")


def validate(df: pl.DataFrame) -> int:
    if df.columns != DISPOSITION_COLUMNS:
        raise RuntimeError(f"disposition columns {df.columns} != {DISPOSITION_COLUMNS}")
    for row in df.iter_rows(named=True):
        HistoricalDisposition(**row)
    return df.height
