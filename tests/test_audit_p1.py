"""Tests for the P1 audit (convert + analyses + driver) on a fixture with known answers."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import yaml

from scripts.audit import convert as cv
from scripts.audit import run_audit as ra
from tests import p1_fixture as fx


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("p1")
    raw = root / "raw"
    sources = fx.build(raw)
    cfg = root / "data_sources.yaml"
    cfg.write_text(yaml.safe_dump({"files": sources}), encoding="utf-8")
    out = root / "results.json"
    rc = ra.main(
        [
            "--raw-dir", str(raw),
            "--interim-dir", str(root / "interim"),
            "--sources", str(cfg),
            "--out", str(out),
            "--variants", "HI-Small",
        ]
    )  # fmt: skip
    assert rc == 0
    res = json.loads(out.read_text(encoding="utf-8"))["variants"]["HI-Small"]
    return {"root": root, "raw": raw, "cfg": cfg, "out": out, "res": res, "sources": sources}


# ---------------------------------------------------------------- convert


def test_patterns_parse(ws):
    df = cv.parse_patterns(ws["raw"] / "HI-Small_Patterns.txt")
    assert df.height == 6 and df["attempt_id"].n_unique() == 2
    assert df.filter(pl.col("attempt_id") == 0)["typology"].unique().to_list() == ["FAN-OUT"]
    assert df["typology_param"].unique().sort().to_list() == [3]
    assert df.schema["timestamp"] == pl.Datetime("us") and df.schema["amount_paid"] == pl.Float64


@pytest.mark.parametrize(
    "text,err",
    [
        ("BEGIN LAUNDERING ATTEMPT - A\nBEGIN LAUNDERING ATTEMPT - B\n", "nested"),
        ("BEGIN LAUNDERING ATTEMPT - A\nEND LAUNDERING ATTEMPT - B\n", "does not close"),
        ("BEGIN LAUNDERING ATTEMPT - A:  Max 3 widgets\n", "unknown parameter"),
        ("2022/09/01 00:00,1,A,2,B,1,USD,1,USD,ACH,1\n", "outside a block"),
        ("BEGIN LAUNDERING ATTEMPT - A\n", "unclosed"),
    ],
)
def test_patterns_structure_errors(tmp_path: Path, text: str, err: str):
    p = tmp_path / "x.txt"
    p.write_text(text)
    with pytest.raises(cv.AuditError, match=err):
        cv.parse_patterns(p)


def test_verify_input_rejects_changed_file(ws, tmp_path: Path):
    src = ws["raw"] / "HI-Small_accounts.csv"
    exp = dict(ws["sources"][src.name])
    exp["sha256"] = "0" * 64
    with pytest.raises(cv.AuditError, match="sha256"):
        cv.verify_input(src, exp)
    exp = dict(ws["sources"][src.name], size_bytes=1)
    with pytest.raises(cv.AuditError, match="size"):
        cv.verify_input(src, exp)


def test_header_mismatch_raises(tmp_path: Path):
    p = tmp_path / "t.csv"
    p.write_text("Timestamp,From Bank,Account\n")
    with pytest.raises(cv.AuditError, match="header"):
        cv.scan_trans_csv(p)


def test_trans_parquet_typed_and_row_index(ws):
    t = pl.read_parquet(ws["root"] / "interim" / "HI-Small_trans.parquet")
    assert t.height == fx.N_ROWS
    assert t["row_id"].to_list() == list(range(fx.N_ROWS))
    assert t.schema["from_bank"] == pl.String  # leading zeros preserved
    assert "010" in t["from_bank"].to_list()


# ---------------------------------------------------------------- analyses (known answers)


def test_time_and_holdout(ws):
    tm = ws["res"]["time"]
    assert tm["n_complete_days"] == 8
    assert tm["complete_span_first_day"] == "2022-09-01"
    assert tm["complete_span_last_day"] == "2022-09-08"
    assert tm["provisional_holdout"]["holdout_start"] == "2022-09-07T00:00:00"
    assert tm["incomplete_days_outside_span"] == [{"day": "2022-09-10", "n": 2, "n_laundering": 2}]
    assert tm["file_sorted_by_time"] is False


def test_schema(ws):
    s = ws["res"]["schema"]
    assert s["n_rows"] == fx.N_ROWS and s["n_laundering"] == fx.N_LAUNDERING
    assert s["n_self_transfer"] == 8 and s["n_cross_currency"] == 8
    assert all(v == 0 for v in s["nulls"].values())


def test_duplicates(ws):
    d = ws["res"]["duplicates"]
    assert d["exact_duplicate_groups"] == 1 and d["exact_duplicate_extra_rows"] == 1
    assert d["label_conflict_groups"] == 1 and d["label_conflict_rows"] == 2


def test_keys(ws):
    k = ws["res"]["keys"]
    assert k["account_number_alone_is_unique"] is False
    assert k["n_account_numbers_in_multiple_banks"] == 1  # 800A1 at 010 and 020
    assert k["n_bank_id_collisions_after_stripping_zeros"] == 1  # "010" vs "10"
    assert k["n_distinct_bank_account_pairs"] == 14
    assert k["trans_pairs_found_in_accounts_file"] == 6
    assert k["accounts_file_key_unique"] is True
    assert k["entity_type_counts"] == {"Corporation": 4, "Partnership": 1, "Sole Proprietorship": 1}


def test_labels_by_period(ws):
    lp = ws["res"]["labels_by_period"]
    assert lp["after_span"] == {"n": 2, "n_laundering": 2, "rate": 1.0}
    assert sum(v["n"] for v in lp.values()) == fx.N_ROWS


def test_patterns_vs_trans(ws):
    pv = ws["res"]["patterns_vs_trans"]
    assert pv["n_pattern_txns_matched_in_trans"] == 5
    assert pv["n_pattern_txns_not_in_trans"] == 1
    assert pv["pattern_txns_not_in_trans_by_period"] == {"after_span": 1}
    assert pv["n_laundering_rows_attributed_to_a_pattern"] == 5
    assert pv["n_laundering_rows_not_in_any_pattern"] == 3


def test_typologies(ws):
    ty = ws["res"]["typologies"]
    assert ty["n_attempts"] == 2
    assert ty["n_attempts_extending_past_span_end"] == 1  # the CYCLE
    by = {r["typology"]: r for r in ty["by_typology"]}
    assert by["FAN-OUT"]["n_txns"] == 3 and by["FAN-OUT"]["max_duration_h"] == 2.0


def test_label_rates_are_pre_holdout_only(ws):
    lr = ws["res"]["label_rates_pre_holdout"]
    assert lr["scope"] == "pre_holdout"
    ach = next(r for r in lr["payment_format"] if r["payment_format"] == "ACH")
    assert ach["n"] == 5 and ach["rate"] == 1.0  # holdout/tail ACH rows excluded


def test_currency_rates_fixed(ws):
    c = ws["res"]["currency_pre_holdout"]
    assert c["usd_per_unit"]["Euro"]["usd_per_unit"] == pytest.approx(1.1)
    assert c["max_rel_iqr"] == 0.0


def test_separability_detects_planted_artefact(ws):
    sep = ws["res"]["separability"]
    assert sep["fit_window"] == ["2022-09-01T00:00:00", "2022-09-05T00:00:00"]
    assert sep["eval_window"] == ["2022-09-05T00:00:00", "2022-09-07T00:00:00"]
    assert sep["n_eval_positive"] == 2
    assert sep["models"]["depth_1"]["pr_auc_eval"] == 1.0
    assert sep["single_feature_group_depth3"]["payment_format"]["pr_auc_eval"] == 1.0
    root_split = sep["models"]["depth_1"]["tree"][0]["split"]
    assert root_split.startswith("payment_format=ACH")


def test_identifier_leakage_runs(ws):
    il = ws["res"]["identifier_leakage"]
    assert il["scope"] == "pre_holdout accounts" and il["n_account_ids_not_hex"] == 0
    assert 0.0 <= il["pr_auc_account_id_as_score"] <= 1.0


# ---------------------------------------------------------------- driver


def test_verify_reproduces_and_detects_change(ws):
    args = [
        "--raw-dir", str(ws["raw"]), "--interim-dir", str(ws["root"] / "interim"),
        "--sources", str(ws["cfg"]), "--out", str(ws["out"]), "--variants", "HI-Small",
    ]  # fmt: skip
    assert ra.main([*args, "--verify"]) == 0
    data = json.loads(ws["out"].read_text(encoding="utf-8"))
    data["variants"]["HI-Small"]["schema"]["n_rows"] += 1
    tampered = ws["root"] / "tampered.json"
    tampered.write_text(json.dumps(data), encoding="utf-8")
    args[args.index("--out") + 1] = str(tampered)
    assert ra.main([*args, "--verify"]) == 1


def test_fact_check_fails_on_wrong_frozen_count(ws, tmp_path: Path):
    bad = yaml.safe_load(ws["cfg"].read_text(encoding="utf-8"))
    bad["files"]["HI-Small_Patterns.txt"]["n_attempts"] = 3
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(cv.AuditError, match="attempts"):
        ra.main(
            [
                "--raw-dir", str(ws["raw"]), "--interim-dir", str(tmp_path / "i"),
                "--sources", str(cfg), "--out", str(tmp_path / "o.json"), "--variants", "HI-Small",
            ]
        )  # fmt: skip
