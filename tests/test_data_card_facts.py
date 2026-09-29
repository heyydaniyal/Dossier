"""Every number stated in docs/DATA_CARD.md, asserted against the committed audit results.

If the audit is re-run and any of these change, this test fails and the data card must be
re-checked. Needs no raw data: it reads docs/audit/p1_audit_results.json only.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "docs/audit/p1_audit_results.json").read_text(encoding="utf-8"))
V = RES["variants"]
HI, LI, SM = V["HI-Medium"], V["LI-Medium"], V["HI-Small"]


def fmt(d: dict, name: str) -> dict:
    return next(
        r for r in d["label_rates_pre_holdout"]["payment_format"] if r["payment_format"] == name
    )


def test_row_counts_and_labels():
    assert HI["schema"]["n_rows"] == 31_898_238 and LI["schema"]["n_rows"] == 31_251_483
    assert SM["schema"]["n_rows"] == 5_078_345
    assert HI["schema"]["n_laundering"] == 35_230 and LI["schema"]["n_laundering"] == 16_041
    assert round(HI["schema"]["laundering_rate_overall"] * 100, 3) == 0.110
    assert round(LI["schema"]["laundering_rate_overall"] * 100, 3) == 0.051


def test_schema_facts():
    for d in (HI, LI, SM):
        s = d["schema"]
        assert all(v == 0 for v in s["nulls"].values())
        assert all(v == 0 for v in s["empty_strings"].values())
        assert s["n_same_currency_amount_mismatch"] == 0
        assert (
            s["amounts"]["amount_paid__n_zero"] == 0
            and s["amounts"]["amount_paid__n_negative"] == 0
        )
        assert (
            s["self_transfer_by_format"]["Reinvestment"]
            == s["payment_format_counts"]["Reinvestment"]
        )
        assert len(s["payment_currency_counts"]) == 15 and len(s["payment_format_counts"]) == 7
    assert (
        HI["schema"]["n_cross_currency"] == 485_144 and HI["schema"]["n_self_transfer"] == 2_561_860
    )
    assert HI["schema"]["amounts"]["amount_paid__median"] == 1471.54
    assert HI["duplicates"]["exact_duplicate_extra_rows"] == 20
    assert LI["duplicates"]["exact_duplicate_extra_rows"] == 14
    for d in (HI, LI, SM):
        assert d["duplicates"]["label_conflict_groups"] == 0
        assert d["duplicates"]["exact_duplicate_groups_laundering"] == 0


def test_account_key_facts():
    k = HI["keys"]
    assert (
        k["account_number_alone_is_unique"] is False
        and k["n_account_numbers_in_multiple_banks"] == 24
    )
    assert LI["keys"]["n_account_numbers_in_multiple_banks"] == 34
    for d in (HI, LI, SM):
        kk = d["keys"]
        assert kk["n_pairs_after_bank_normalisation"] == kk["n_distinct_bank_account_pairs"]
        assert kk["trans_pairs_missing_from_accounts_file"] == 0
        assert kk["n_bank_ids_with_leading_zero"] == kk["n_distinct_bank_ids_raw"]
        assert kk["accounts_file_key_unique"] is True
    assert k["n_distinct_bank_account_pairs"] == 2_077_023
    assert k["accounts_file_pairs_never_transacting"] == 10_763
    assert k["n_bank_id_collisions_after_stripping_zeros"] == 61
    assert LI["keys"]["n_bank_id_collisions_after_stripping_zeros"] == 67


def test_time_facts():
    for d in (HI, LI):
        t = d["time"]
        assert t["complete_span_first_day"] == "2022-09-01"
        assert t["complete_span_last_day"] == "2022-09-16"
        assert t["n_complete_days"] == 16 and t["incomplete_days_inside_span"] == 0
        assert t["provisional_holdout"]["holdout_start"] == "2022-09-13T00:00:00"
        assert t["provisional_holdout"]["pre_holdout_days"] == 12
        assert t["file_sorted_by_time"] is False
    assert (
        HI["time"]["ts_max"] == "2022-09-28T15:58:00"
        and LI["time"]["ts_max"] == "2022-09-27T14:58:00"
    )
    assert SM["time"]["complete_span_last_day"] == "2022-09-10"
    assert SM["time"]["provisional_holdout"]["holdout_start"] == "2022-09-08T00:00:00"
    assert HI["time"]["n_distinct_timestamps"] == 26_086
    assert HI["time"]["max_txns_same_minute"] == 44_215
    assert HI["time"]["share_rows_sharing_timestamp"] > 0.99997
    assert HI["time"]["n_time_inversions_in_file_order"] == 15_174_709
    d1 = HI["time"]["daily"]
    assert (d1[0]["n"], d1[0]["n_laundering"]) == (4_465_985, 1_056)
    assert d1[15]["day"] == "2022-09-16" and d1[15]["n_laundering"] == 2_416
    assert HI["time"]["hourly_volume_first_day"][0] == 1_380_940


def test_weekend_rate_artefact():
    def rate(days):
        return sum(x["n_laundering"] for x in days) / sum(x["n"] for x in days)

    pre = [x for x in HI["time"]["daily"] if "2022-09-02" <= x["day"] < "2022-09-13"]
    we = [x for x in pre if date.fromisoformat(x["day"]).weekday() >= 5]
    wd = [x for x in pre if date.fromisoformat(x["day"]).weekday() < 5]
    assert round(rate(we) / rate(wd), 1) == 2.3


def test_tail_after_span():
    a = HI["labels_by_period"]["after_span"]
    assert (a["n"], a["n_laundering"]) == (6_987, 4_084) and round(a["rate"], 3) == 0.585
    assert round(a["n_laundering"] / HI["schema"]["n_laundering"], 3) == 0.116
    b = LI["labels_by_period"]["after_span"]
    assert (b["n"], b["n_laundering"]) == (1_370, 771)
    assert len(HI["time"]["incomplete_days_outside_span"]) == 12
    assert len(LI["time"]["incomplete_days_outside_span"]) == 11
    assert SM["labels_by_period"]["after_span"]["n"] == 1_108


def test_period_label_counts():
    p = HI["labels_by_period"]
    assert (p["pre_holdout"]["n"], p["pre_holdout"]["n_laundering"]) == (23_079_383, 21_939)
    assert (p["holdout"]["n"], p["holdout"]["n_laundering"]) == (8_811_868, 9_207)
    q = LI["labels_by_period"]
    assert (q["pre_holdout"]["n"], q["pre_holdout"]["n_laundering"]) == (22_620_673, 11_035)
    assert (q["holdout"]["n"], q["holdout"]["n_laundering"]) == (8_629_440, 4_235)


def test_patterns_facts():
    for d, n_ptx, n_unattr in ((HI, 22_743, 12_487), (LI, 3_909, 12_132)):
        pv = d["patterns_vs_trans"]
        assert pv["n_pattern_txns_matched_in_trans"] == n_ptx == pv["n_pattern_txns"]
        assert (
            pv["n_matched_trans_rows_label_0"] == 0 and pv["n_trans_rows_in_multiple_attempts"] == 0
        )
        assert pv["n_laundering_rows_not_in_any_pattern"] == n_unattr
    ty = HI["typologies"]
    assert ty["n_attempts"] == 2_756 and LI["typologies"]["n_attempts"] == 456
    assert ty["attempt_duration_h"]["median"] == pytest.approx(110.5166667)
    assert ty["attempt_duration_h"]["max"] == pytest.approx(304.6666667)
    assert (
        ty["n_attempts_extending_past_span_end"] == 738
        and ty["n_attempts_crossing_holdout_start"] == 691
    )
    by = {
        r["typology"]: (r["n_attempts"], r["n_txns"], round(r["median_duration_h"], 2))
        for r in ty["by_typology"]
    }
    assert by == {
        "BIPARTITE": (369, 2135, 42.55),
        "CYCLE": (367, 2235, 121.82),
        "FAN-IN": (355, 2315, 119.17),
        "FAN-OUT": (345, 2128, 115.55),
        "GATHER-SCATTER": (322, 4289, 227.36),
        "RANDOM": (331, 1667, 93.5),
        "SCATTER-GATHER": (331, 3988, 131.42),
        "STACK": (336, 3986, 102.56),
    }
    la = HI["laundering_accounts"]
    assert la["n_accounts_in_laundering_txns"] == 41_857
    assert (
        la["n_pattern_accounts_in_multiple_attempts"] == 1_041
        and la["max_attempts_per_account"] == 57
    )
    assert la["n_pattern_accounts_in_multiple_typologies"] == 944


def test_currency_facts():
    for d in (HI, LI):
        c = d["currency_pre_holdout"]
        fiat = [
            p
            for p in c["pairs"]
            if "Bitcoin" not in (p["payment_currency"], p["receiving_currency"])
        ]
        assert max(p["rel_iqr"] for p in fiat) <= 0.0026
        for ccy, v in c["usd_per_unit"].items():
            if ccy not in ("Bitcoin", "US Dollar"):
                assert v["direct_vs_inverse_rel_diff"] <= 6e-6, ccy
        assert 11_870 <= c["usd_per_unit"]["Bitcoin"]["usd_per_unit"] <= 11_882
    cc = next(
        r for r in HI["label_rates_pre_holdout"]["cross_currency"] if r["cross_currency"] == "true"
    )
    assert (cc["n"], cc["n_laundering"]) == (339_144, 0)


def test_separability_facts():
    s = HI["separability"]
    assert s["fit_window"] == ["2022-09-01T00:00:00", "2022-09-09T00:00:00"]
    assert s["eval_window"] == ["2022-09-09T00:00:00", "2022-09-13T00:00:00"]
    assert round(s["models"]["depth_1"]["pr_auc_eval"], 4) == 0.0082
    assert round(s["models"]["depth_3"]["pr_auc_eval"], 4) == 0.0211
    assert s["models"]["depth_1"]["tree"][0]["split"].startswith("payment_format=ACH")
    assert round(s["models"]["depth_3"]["pr_auc_over_prevalence"], 1) == 15.5
    assert (
        max(n["positive_rate"] for n in s["models"]["depth_3"]["tree"]) < 0.016
    )  # far from separable
    t = LI["separability"]
    assert round(t["models"]["depth_3"]["pr_auc_eval"], 4) == 0.0061
    assert (
        round(fmt(HI, "ACH")["n_laundering"] / HI["label_rates_pre_holdout"]["n_laundering"], 2)
        == 0.86
    )
    assert (
        round(fmt(LI, "ACH")["n_laundering"] / LI["label_rates_pre_holdout"]["n_laundering"], 2)
        == 0.72
    )
    for d in (HI, LI, SM):
        assert fmt(d, "Wire")["n_laundering"] == 0 and fmt(d, "Reinvestment")["n_laundering"] == 0


def test_identifier_leakage_facts():
    il = HI["identifier_leakage"]
    assert round(il["contiguity"]["lift"], 1) == 7.0
    assert round(LI["identifier_leakage"]["contiguity"]["lift"], 1) == 14.9
    for d in (HI, LI, SM):
        eight = next(
            r for r in d["identifier_leakage"]["by_last_hex_digit"] if r["last_hex"] == "8"
        )
        assert eight["n"] == eight["n_laundering"] == 7
    rates = [r["rate"] for r in il["txn_rate_by_file_position_decile"]]
    assert round(max(rates) / min(rates), 1) == 5.7
    assert round(il["pr_auc_negated_account_id_as_score"], 3) == 0.038


def test_kyc_and_footprint_facts():
    k = HI["keys"]
    assert k["n_entities"] == 668_138 and k["n_entities_with_multiple_accounts"] == 210_275
    assert k["max_accounts_per_entity"] == 8_638
    assert k["entity_type_counts"] == {
        "Corporation": 679_329,
        "Country": 4_738,
        "Direct": 27,
        "Individual": 3_050,
        "Partnership": 732_828,
        "Sole Proprietorship": 667_814,
    }
    assert HI["identifier_leakage"]["n_accounts_without_kyc_row"] == 0
    f = HI["footprint"]
    assert round(f["parquet_mb"]["trans"]) == 742
    assert (
        1.7 < f["est_in_memory_gb_categorical"] < 1.9 and 2.6 < f["est_in_memory_gb_strings"] < 2.8
    )


def test_remaining_card_numbers():
    # time table
    assert round(LI["labels_by_period"]["after_span"]["rate"], 3) == 0.563
    assert (
        round(
            LI["labels_by_period"]["after_span"]["n_laundering"] / LI["schema"]["n_laundering"], 3
        )
        == 0.048
    )
    assert round(SM["labels_by_period"]["after_span"]["rate"], 3) == 0.591
    assert len(SM["time"]["incomplete_days_outside_span"]) == 8
    assert SM["time"]["n_complete_days"] == 10
    # labels table (LI)
    la = LI["laundering_accounts"]
    assert la["n_accounts_in_laundering_txns"] == 23_510 and la["n_accounts_total"] == 2_032_095
    assert (la["n_pattern_accounts_in_multiple_attempts"], la["max_attempts_per_account"]) == (
        181,
        6,
    )
    assert la["n_pattern_accounts_in_multiple_typologies"] == 142
    assert HI["laundering_accounts"]["n_accounts_total"] == 2_077_023
    # separability table
    assert round(HI["separability"]["eval_prevalence"] * 100, 3) == 0.136
    assert round(LI["separability"]["eval_prevalence"] * 100, 3) == 0.063
    assert round(HI["separability"]["models"]["depth_1"]["pr_auc_over_prevalence"], 1) == 6.1
    assert round(LI["separability"]["models"]["depth_1"]["pr_auc_eval"], 4) == 0.0030
    assert round(LI["separability"]["models"]["depth_1"]["pr_auc_over_prevalence"], 1) == 4.7
    assert round(LI["separability"]["models"]["depth_3"]["pr_auc_over_prevalence"], 1) == 9.6
    assert LI["separability"]["models"]["depth_1"]["tree"][0]["split"].startswith(
        "payment_format=ACH"
    )
    leaf = lambda d: max(n["positive_rate"] for n in d["separability"]["models"]["depth_3"]["tree"])  # noqa: E731
    assert round(leaf(HI) * 100, 2) == 1.57 and round(leaf(LI) * 100, 2) == 0.57
    assert round(fmt(HI, "ACH")["lift"], 1) == 7.2 and round(fmt(LI, "ACH")["lift"], 1) == 6.1
    h0 = HI["label_rates_pre_holdout"]["hour"][0]
    assert h0["hour"] == "0" and round(h0["lift"], 2) == 0.27
    st = next(
        r for r in HI["label_rates_pre_holdout"]["self_transfer"] if r["self_transfer"] == "true"
    )
    assert st["n_laundering"] == 42
    # leakage details
    il = HI["identifier_leakage"]
    assert round(il["contiguity"]["p_next_id_laundering_given_laundering"] * 100, 1) == 9.5
    assert round(il["account_prevalence"] * 100, 2) == 1.35
    top = il["id_prefix2_max_lift_with_support"][0]
    assert top["prefix2"] == "80" and round(top["lift"], 1) == 2.0
    assert round(il["pr_auc_negated_account_id_as_score"] / il["account_prevalence"], 1) == 2.8
    assert round(il["bank_max_lift_with_support"][0]["lift"]) == 6
    rates = [r["rate"] for r in il["txn_rate_by_file_position_decile"]]
    assert round(min(rates) * 100, 3) == 0.035 and round(max(rates) * 100, 3) == 0.197
    # runtime on the team laptop
    el = RES["run_info"]["elapsed_s_per_variant"]
    assert el["HI-Medium"] < 400 and el["LI-Medium"] < 400
