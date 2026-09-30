"""Synthetic KYC profiles (P2 tasks 6-7), KYC v2 (approved by Dani 2026-09-30).

Runtime-safe and fully LABEL-FREE: no access to labels, patterns, rule outcomes or the evaluation
package. Every synthetic draw is a keyed hash of (seed, stream, account_key): independent of
labels, of row order and of ID order.

Inputs are only real account attributes and the BURN-IN day's transaction statistics
(2022-09-01, before every window), so nothing here can carry future information.

v2 changes (reasons in configs/p2_kyc.yaml):
  - onboarding is stored as a YEAR: a day-level random date acted as an account fingerprint;
  - planted explanations follow each account's burn-in-day behaviour (plant_from_behaviour),
    no longer later rule outcomes plus labels: the firewall exception is gone.
tests/test_p2_generators.py audits this file's source for any ground-truth access.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src.data.hashing import choose, keyed_u64, uniforms
from src.data.periods import ROOT

KYC_CONFIG = ROOT / "configs" / "p2_kyc.yaml"
GOVERNMENT = "government_public_body"
RISK_POINTS = {"low": 0, "medium": 1, "high": 2}
KYC_COLUMNS = [
    "account_key",
    "entity_type",
    "bank_location",
    "bank_country",
    "country_risk",
    "sector_or_occupation",
    "expected_activity_band",
    "onboarding_year",
    "customer_risk_rating",
    "kyc_as_of",
]


def load_kyc_config(path: Path = KYC_CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sector_risk(cfg: dict) -> dict[str, str]:
    m = {k: v["risk"] for k, v in cfg["sectors"].items()}
    m |= {k: v["risk"] for k, v in cfg["occupations"].items()}
    m[GOVERNMENT] = "low"
    return m


def _draw(u: np.ndarray, table: dict) -> np.ndarray:
    names = list(table)
    return choose(u, names, [table[n]["w"] for n in names])


def country_tiers(countries: list[str], cfg: dict) -> dict[str, str]:
    cr = cfg["country_risk"]
    uniq = sorted(set(countries))
    if not uniq:
        return {}
    u = uniforms(keyed_u64(uniq, "country_risk", cfg["seed"]), "tier")
    probs = cr["foreign_tier_probs"]
    tiers = choose(u, list(probs), list(probs.values()))
    out = {c: str(t) for c, t in zip(uniq, tiers, strict=True)}
    if cr["domestic"] in out:
        out[cr["domestic"]] = cr["domestic_tier"]
    if cr.get("crypto") in out:
        out[cr["crypto"]] = cr["crypto_tier"]
    return out


def base_kyc(attrs: pl.DataFrame, burn_in_volume: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """attrs: account_key, entity_type, bank_location, bank_country (one row per account).
    burn_in_volume: account_key, volume_usd over the burn-in day (transactions only)."""
    if attrs["account_key"].n_unique() != attrs.height:
        raise ValueError("attrs must have one row per account_key")
    seed = cfg["seed"]
    df = attrs.join(
        burn_in_volume.select("account_key", "volume_usd"), on="account_key", how="left"
    )
    h = keyed_u64(df["account_key"].to_list(), "kyc", seed)
    n = df.height

    # sector / occupation
    et = df["entity_type"].to_numpy()
    is_ind = np.isin(et, cfg["individual_types"])
    is_pub = np.isin(et, cfg["public_sector_types"])
    sec = _draw(uniforms(h, "sector"), cfg["sectors"])
    occ = _draw(uniforms(h, "occupation"), cfg["occupations"])
    sector = np.where(is_pub, GOVERNMENT, np.where(is_ind, occ, sec)).astype(object)

    # expected activity band from burn-in activity, within entity type
    ab = cfg["activity_band"]
    bands = ab["bands"]
    # percentile rank among the accounts of the same entity type that were active on burn-in day
    pct = (
        df.select(
            (
                pl.col("volume_usd").rank("average").over("entity_type")
                / pl.col("volume_usd").count().over("entity_type")
            ).cast(pl.Float64)
        )
        .to_series()
        .to_numpy()
    )
    active = ~np.isnan(pct)
    pctf = np.where(active, pct, 0.0)
    idx = np.zeros(n, dtype=np.int64)
    for c in ab["cuts"]:
        idx += (pctf > c).astype(np.int64)
    shift = np.where(
        uniforms(h, "band_noise") < ab["noise_prob"],
        np.where(uniforms(h, "band_dir") < 0.5, -1, 1),
        0,
    )
    idx = np.clip(idx + shift, 0, len(bands) - 1)
    prior = ab["prior_if_inactive"]
    prior_draw = choose(uniforms(h, "band_prior"), list(prior), list(prior.values()))
    band = np.where(active, np.asarray(bands, dtype=object)[idx], prior_draw).astype(object)

    # onboarding: a day is drawn, only its YEAR is stored (a day acts as an account fingerprint)
    ob = cfg["onboarding"]
    d0, d1 = date.fromisoformat(ob["start"]), date.fromisoformat(ob["end"])
    span = (d1 - d0).days + 1
    offs = np.floor(uniforms(h, "onboard") * span).astype(np.int64)
    onboard_year = [(d0 + timedelta(days=int(o))).year for o in offs]

    tiers = country_tiers(df["bank_country"].to_list(), cfg)
    out = df.select("account_key", "entity_type", "bank_location", "bank_country").with_columns(
        pl.col("bank_country").replace_strict(tiers, return_dtype=pl.String).alias("country_risk"),
        pl.Series("sector_or_occupation", sector.tolist(), dtype=pl.String),
        pl.Series("expected_activity_band", band.tolist(), dtype=pl.String),
        pl.Series("onboarding_year", onboard_year, dtype=pl.Int32),
        pl.Series("_h", h, dtype=pl.UInt64),
    )
    return out


def plant_from_behaviour(
    kyc: pl.DataFrame, burn_in_stats: pl.DataFrame, cfg: dict
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Task 7, v2: planted innocent explanations from BURN-IN behaviour only (no ground truth).

    For every account active on the burn-in day, each behaviour statistic is turned into a
    within-day percentile (0 if the statistic is 0). The account's dominant behaviour is the one
    with the highest percentile (ties: config order). Accounts whose dominant percentile is at
    least candidate_percentile are candidates; each candidate receives, with planting_prob, a
    sector (or occupation) whose normal business looks like that behaviour, plus a high declared
    activity band. Returns (kyc, planting_record); the record goes to the evaluation store only.
    """
    pc = cfg["planting"]
    behaviours = list(pc["behaviour_stats"])
    b = burn_in_stats.select("account_key", *[pc["behaviour_stats"][r] for r in behaviours])
    pct_cols = []
    for r in behaviours:
        col = pc["behaviour_stats"][r]
        pct_cols.append(
            pl.when(pl.col(col) > 0)
            .then(pl.col(col).rank("average") / pl.len())
            .otherwise(0.0)
            .cast(pl.Float64)
            .alias(f"_p_{r}")
        )
    b = b.with_columns(pct_cols)
    mat = b.select([f"_p_{r}" for r in behaviours]).to_numpy()
    if mat.size:
        best = mat.argmax(axis=1)  # first maximum = config order on ties
        best_p = mat[np.arange(mat.shape[0]), best]
    else:
        best, best_p = np.zeros(0, dtype=np.int64), np.zeros(0)
    cand = best_p >= pc["candidate_percentile"]
    h = keyed_u64(b["account_key"].to_list(), "plant_v2", cfg["seed"])
    planted = cand & (uniforms(h, "plant") < pc["planting_prob"])
    dominant = (
        np.asarray(behaviours, dtype=object)[best] if mat.size else np.array([], dtype=object)
    )
    arche = pc["archetypes"]
    ua = uniforms(h, "archetype")
    new_sector = np.empty(b.height, dtype=object)
    for r in behaviours:
        m = dominant == r
        if m.any():
            new_sector[m] = choose(ua[m], arche[r], [1.0] * len(arche[r]))
    bp = pc["planted_band_probs"]
    new_band = choose(uniforms(h, "band"), list(bp), list(bp.values()))
    record = b.select("account_key").with_columns(
        pl.Series("candidate", cand),
        pl.Series("planted", planted),
        pl.Series("dominant_behaviour", dominant.tolist(), dtype=pl.String),
        pl.Series("dominant_percentile", best_p, dtype=pl.Float64),
        pl.Series("planted_sector", new_sector.tolist(), dtype=pl.String),
        pl.Series("planted_band", new_band.tolist(), dtype=pl.String),
    )
    j = kyc.join(
        record.select("account_key", "planted", "planted_sector", "planted_band"),
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
    return j, record.sort("account_key")


def derive_risk(kyc: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Customer risk rating = f(sector risk, country risk, declared activity) + seeded noise.
    Called AFTER planting so the rating is consistent with the final profile."""
    rr = cfg["customer_risk_rating"]
    srisk = sector_risk(cfg)
    h = kyc["_h"].to_numpy()
    pts = (
        kyc["sector_or_occupation"].replace_strict(srisk).replace_strict(RISK_POINTS).to_numpy()
        + kyc["country_risk"].replace_strict(RISK_POINTS).to_numpy()
        + (kyc["expected_activity_band"] == "very_high").to_numpy().astype(np.int64)
        * rr["very_high_activity_point"]
    ).astype(np.int64)
    noise = np.where(
        uniforms(h, "rating_noise") < rr["noise_prob"],
        np.where(uniforms(h, "rating_dir") < 0.5, -1, 1),
        0,
    )
    pts = pts + noise
    rating = np.where(
        pts >= rr["cuts"]["high_at"],
        "high",
        np.where(pts >= rr["cuts"]["medium_at"], "medium", "low"),
    )
    as_of = datetime.fromisoformat(cfg["kyc_as_of"])
    return (
        kyc.with_columns(
            pl.Series("customer_risk_rating", rating.tolist(), dtype=pl.String),
            pl.lit(as_of).alias("kyc_as_of"),
        )
        .sort("_h")
        .select(KYC_COLUMNS)
    )
