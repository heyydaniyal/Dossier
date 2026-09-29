"""Validates each FROZEN Phase-1 decision (configs/p1_data_decisions.yaml) against its evidence.

A decision is justified only while its evidence holds. If a re-run of the audit changed the
evidence, these tests fail and the decision must be revisited, not silently kept.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEC = yaml.safe_load((ROOT / "configs/p1_data_decisions.yaml").read_text(encoding="utf-8"))
RES = json.loads((ROOT / "docs/audit/p1_audit_results.json").read_text(encoding="utf-8"))
MEDIUM = [RES["variants"][v] for v in ("HI-Medium", "LI-Medium")]


def _naive(s: str) -> str:
    return datetime.fromisoformat(s).replace(tzinfo=None).isoformat()


def test_config_is_well_formed():
    assert DEC["span"]["timezone"] == "UTC"
    for k in ("start", "end_exclusive"):
        assert datetime.fromisoformat(DEC["span"][k]).tzinfo is not None
    assert DEC["account_key"]["format"] == "{bank}|{account}"
    assert DEC["variants"]["primary"] == "HI-Medium"


def test_d1_tail_exclusion_is_justified():
    """Span = the complete-volume days; the tail after it is a near-pure laundering artefact."""
    for d in MEDIUM:
        t = d["time"]
        assert (
            _naive(DEC["span"]["end_exclusive"]) == t["provisional_holdout"]["span_end_exclusive"]
        )
        assert t["complete_span_first_day"] == DEC["span"]["start"][:10]
        tail = d["labels_by_period"]["after_span"]["rate"]
        normal = d["labels_by_period"]["pre_holdout"]["rate"]
        # the rule "timestamp >= span end" alone would be > 500x better than chance: a fake signal
        assert tail / normal > 500
        # and the tail is tiny in volume (< 0.03% of rows): excluding it costs no normal activity
        assert d["labels_by_period"]["after_span"]["n"] / d["schema"]["n_rows"] < 0.0003


def test_d2_composite_key_is_required_and_lossless():
    for d in MEDIUM:
        k = d["keys"]
        assert k["account_number_alone_is_unique"] is False  # key must include the bank
        assert k["n_pairs_after_bank_normalisation"] == k["n_distinct_bank_account_pairs"]
        assert k["trans_pairs_missing_from_accounts_file"] == 0  # join rule is complete


def test_d3_timestamps_have_no_timezone_and_minute_resolution():
    for d in MEDIUM:
        assert d["time"]["timestamp_resolution"].startswith("minute")
        assert "+" not in d["time"]["ts_min"] and "Z" not in d["time"]["ts_min"]  # naive in data


def test_d4_holdout_is_last_quarter_and_evaluable():
    for d in MEDIUM:
        ph = d["time"]["provisional_holdout"]
        assert _naive(DEC["provisional_holdout"]["test_start_not_before"]) == ph["holdout_start"]
        assert ph["holdout_days"] == 4 and ph["pre_holdout_days"] == 12
        # evaluable: well over 1,000 positive transactions in the holdout (aggregate count only)
        assert d["labels_by_period"]["holdout"]["n_laundering"] > 4_000


def test_d5_identifier_ban_is_justified():
    for d in MEDIUM:
        il = d["identifier_leakage"]
        assert il["contiguity"]["lift"] > 5  # neighbouring IDs share labels far above chance
        rates = [r["rate"] for r in il["txn_rate_by_file_position_decile"]]
        assert max(rates) / min(rates) > 3  # file position correlates with the label
    banned = set(DEC["banned_as_signal"])
    assert {"account_number_value", "row_id", "file_order"} <= banned


def test_d6_usd_conversion_is_needed_and_well_defined():
    for d in MEDIUM:
        s = d["schema"]
        non_usd = 1 - s["payment_currency_counts"]["US Dollar"] / s["n_rows"]
        assert non_usd > 0.5  # most amounts are not in USD: raw amounts are not comparable
        usd = d["currency_pre_holdout"]["usd_per_unit"]
        assert len(usd) == 15  # every currency has a rate derived from the data
        fiat = {c: v for c, v in usd.items() if c not in ("Bitcoin", "US Dollar")}
        assert all(v["direct_vs_inverse_rel_diff"] <= 1e-5 for v in fiat.values())  # fixed rates
    assert DEC["currency"]["external_fx"] == "forbidden"
