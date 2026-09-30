"""Synthetic KYC profiles (P2 task 6). Runtime-safe: NO access to labels, patterns or alerts.

Every synthetic draw is a keyed hash of (seed, stream, account_key) -> independent of labels,
of row order and of ID order. The only label-dependent step, planted explanations (task 7),
lives in scripts/p2/plant.py and calls `derive_risk` afterwards.
tests/test_p2_kyc.py audits this file's source: it must not mention the label column, the
patterns file, the evaluation package or alerts.
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
    "onboarding_date",
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
    burn_in_volume: account_key, volume_usd over the burn-in day (label-free transactions)."""
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

    # onboarding date
    ob = cfg["onboarding"]
    d0, d1 = date.fromisoformat(ob["start"]), date.fromisoformat(ob["end"])
    span = (d1 - d0).days + 1
    offs = np.floor(uniforms(h, "onboard") * span).astype(np.int64)
    onboard = [d0 + timedelta(days=int(o)) for o in offs]

    tiers = country_tiers(df["bank_country"].to_list(), cfg)
    out = df.select("account_key", "entity_type", "bank_location", "bank_country").with_columns(
        pl.col("bank_country").replace_strict(tiers, return_dtype=pl.String).alias("country_risk"),
        pl.Series("sector_or_occupation", sector.tolist(), dtype=pl.String),
        pl.Series("expected_activity_band", band.tolist(), dtype=pl.String),
        pl.Series("onboarding_date", onboard, dtype=pl.Date),
        pl.Series("_h", h, dtype=pl.UInt64),
    )
    return out


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
