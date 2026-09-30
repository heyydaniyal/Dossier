"""P2 task 7: planted innocent explanations. DECLARED FIREWALL EXCEPTION: reads alert labels.

Label-balanced by construction: a legitimate alerted account and a laundering alerted account
have the same probability (configs/p2_kyc.yaml) of receiving a profile that explains the
behaviour behind their FIRST alert (earliest window; its first triggered rule in config order).
The planting record (who was planted, and whether they launder) goes to the EVALUATION store
only. The KYC store keeps no trace of planting.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.data.hashing import choose, keyed_u64, uniforms


def plant(
    kyc: pl.DataFrame, alerts: pl.DataFrame, labels: pl.DataFrame, cfg: dict
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """kyc: base_kyc output (with _h). alerts: alert_id, account_key, window_start, triggered_rules.
    labels: alert_id, is_true_positive. Returns (kyc, planting_record, rates)."""
    pc = cfg["planting"]
    seed = cfg["seed"]
    a = alerts.select("alert_id", "account_key", "window_start", "triggered_rules").join(
        labels.select("alert_id", "is_true_positive"), on="alert_id", how="left"
    )
    if a["is_true_positive"].null_count():
        raise RuntimeError("alerts without labels")
    acc = (
        a.sort("window_start")
        .group_by("account_key", maintain_order=True)
        .agg(
            pl.col("triggered_rules").first().list.first().alias("primary_rule"),
            pl.col("is_true_positive").any().alias("laundering_account"),
        )
        .sort("account_key")
    )
    keys = acc["account_key"].to_list()
    h = keyed_u64(keys, "plant", seed)
    u = uniforms(h, "plant")
    lau = acc["laundering_account"].to_numpy()
    rate = np.where(lau, pc["rate_laundering_alerted"], pc["rate_legitimate_alerted"])
    planted = u < rate

    arche = pc["archetypes"]
    missing = set(acc["primary_rule"].unique().to_list()) - set(arche)
    if missing:
        raise RuntimeError(f"no archetypes for rules {missing}")
    ua = uniforms(h, "archetype")
    new_sector = np.empty(len(keys), dtype=object)
    for rule in arche:
        m = (acc["primary_rule"] == rule).to_numpy()
        if m.any():
            new_sector[m] = choose(ua[m], arche[rule], [1.0] * len(arche[rule]))
    bp = pc["planted_band_probs"]
    new_band = choose(uniforms(h, "band"), list(bp), list(bp.values()))

    rec = acc.with_columns(
        pl.Series("planted", planted),
        pl.Series("planted_sector", new_sector.tolist(), dtype=pl.String),
        pl.Series("planted_band", new_band.tolist(), dtype=pl.String),
    )
    j = kyc.join(
        rec.select("account_key", "planted", "planted_sector", "planted_band"),
        on="account_key",
        how="left",
    )
    is_p = pl.col("planted").fill_null(False)
    ind = pl.col("entity_type").is_in(cfg["individual_types"])
    pub = pl.col("entity_type").is_in(cfg["public_sector_types"])
    j = j.with_columns(
        pl.when(is_p & ind)
        .then(pl.lit(pc["individual_occupation"]))
        .when(is_p & ~pub)
        .then(pl.col("planted_sector"))
        .otherwise(pl.col("sector_or_occupation"))
        .alias("sector_or_occupation"),
        pl.when(is_p)
        .then(pl.col("planted_band"))
        .otherwise(pl.col("expected_activity_band"))
        .alias("expected_activity_band"),
    ).drop("planted", "planted_sector", "planted_band")

    def _rate(mask: np.ndarray) -> float | None:
        return float(planted[mask].mean()) if mask.any() else None

    rates = {
        "n_alerted_accounts": len(keys),
        "n_laundering_alerted_accounts": int(lau.sum()),
        "n_legitimate_alerted_accounts": int((~lau).sum()),
        "declared_rate_laundering": pc["rate_laundering_alerted"],
        "declared_rate_legitimate": pc["rate_legitimate_alerted"],
        "realised_rate_laundering": _rate(lau),
        "realised_rate_legitimate": _rate(~lau),
        "n_planted": int(planted.sum()),
    }
    record = rec.select(
        "account_key",
        "laundering_account",
        "primary_rule",
        "planted",
        "planted_sector",
        "planted_band",
    )
    return j, record, rates
