"""Tests for scripts.audit.discover on small synthetic files with known answers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit import discover as d

TRANS = (
    "Timestamp,Bank,Account,Bank,Account,Amount,Flag\r\n"
    "2022/01/01 00:00,1,A1,2,B1,10.5,0\r\n"
    '2022/01/01 00:05,1,A1,2,"B,2",20,1\r\n'  # quoted comma: still 7 fields
    "2022/01/01 00:06,1,A1,2\r\n"  # malformed: 4 fields
    "2022/01/02 23:59,3,C1,3,C1,5,0"  # no trailing newline
)

PATTERNS = (
    "BEGIN LAUNDERING ATTEMPT - FAN-OUT\n"
    "2022/01/01 00:05,1,A1,2,B2,20,1\n"
    "2022/01/01 00:07,1,A1,2,B3,20,1\n"
    "END LAUNDERING ATTEMPT - FAN-OUT\n"
    "\n"
    "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 4 hops\n"
    "2022/01/01 01:00,1,X,2,Y,5,1\n"
    "END LAUNDERING ATTEMPT - CYCLE:  Max 4 hops\n"
    "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 12 hops\n"
    "2022/01/01 02:00,1,X,2,Y,5,1\n"
    "END LAUNDERING ATTEMPT - CYCLE:  Max 12 hops\n"
)


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    r = tmp_path / "raw"
    r.mkdir()
    (r / "HI-Small_Trans.csv").write_bytes(TRANS.encode())
    (r / "HI-Small_Patterns.txt").write_bytes(PATTERNS.encode())
    (r / "HI-Small_accounts.csv").write_bytes(b"Bank Name,Bank ID\nFirst,1\n")
    (r / "notes.txt").write_bytes(b"x")
    return r


def test_bytes_scan_matches_hashlib_and_counts(raw: Path):
    p = raw / "HI-Small_Trans.csv"
    b = d.scan_bytes(p)
    assert b["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert b["size_bytes"] == p.stat().st_size
    assert b["n_lf"] == 4 and b["n_lines"] == 5  # last line has no newline
    assert b["line_ending"] == "CRLF" and not b["ends_with_newline"]
    assert b["utf8_valid"] and not b["utf8_bom"]


def test_crlf_split_across_chunks(tmp_path: Path, monkeypatch):
    p = tmp_path / "x.csv"
    p.write_bytes(b"ab\r\ncd\r\n")
    monkeypatch.setattr(d, "CHUNK", 3)  # chunk boundary falls between \r and \n
    b = d.scan_bytes(p)
    assert b["n_crlf"] == 2 and b["line_ending"] == "CRLF"


def test_invalid_utf8_and_bom_detected(tmp_path: Path):
    p = tmp_path / "bad.csv"
    p.write_bytes(b"\xef\xbb\xbfa,b\n\xff\xfe,1\n")
    b = d.scan_bytes(p)
    assert b["utf8_bom"] and not b["utf8_valid"] and "offset" in b["utf8_error"]


def test_csv_profile(raw: Path):
    c = d.csv_profile(raw / "HI-Small_Trans.csv")
    assert c["header_n_fields"] == 7
    assert c["duplicate_header_names"] == ["Account", "Bank"]
    assert c["n_data_rows"] == 4
    assert c["field_count_distribution"] == {"4": 1, "7": 3}
    assert c["n_rows_wrong_field_count"] == 1
    assert c["wrong_field_count_examples"][0]["data_row"] == 3


def test_patterns_profile(raw: Path):
    pp = d.patterns_profile(raw / "HI-Small_Patterns.txt")
    assert pp["n_begin"] == 3 and pp["n_end"] == 3
    assert pp["line_kinds"]["blank"] == 1
    assert pp["unclosed_blocks_at_eof"] == 0 and pp["max_nesting_depth"] == 1
    assert pp["end_without_begin"] == 0 and pp["begin_end_template_mismatch"] == 0
    assert pp["transaction_lines_outside_blocks"] == 0
    assert pp["block_size"] == {
        "n_blocks": 3,
        "min": 1,
        "median": 1,
        "max": 2,
        "total_txn_in_blocks": 4,
    }
    # digits masked: the two CYCLE headers collapse into one template
    templates = {t["template"]: t["count"] for t in pp["begin_templates"]}
    assert templates == {
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT": 1,
        "BEGIN LAUNDERING ATTEMPT - CYCLE: Max # hops": 2,
    }


def test_patterns_structure_errors(tmp_path: Path):
    p = tmp_path / "p.txt"
    p.write_text(
        "2022/01/01 00:00,1,A,2,B,1,1\n"  # outside any block
        "END LAUNDERING ATTEMPT - X\n"  # END without BEGIN
        "BEGIN LAUNDERING ATTEMPT - Y\n"
        "BEGIN LAUNDERING ATTEMPT - Z\n"  # nested
        "END LAUNDERING ATTEMPT - Y\n"  # closes Z: mismatch
    )
    pp = d.patterns_profile(p)
    assert pp["transaction_lines_outside_blocks"] == 1
    assert pp["end_without_begin"] == 1
    assert pp["max_nesting_depth"] == 2
    assert pp["unclosed_blocks_at_eof"] == 1
    assert pp["begin_end_template_mismatch"] == 1


def test_discover_missing_extra_and_reproducible(raw: Path, tmp_path: Path):
    r1 = d.discover(raw)
    assert "HI-Medium_Trans.csv" in r1["missing_files"]
    assert "HI-Small_Trans.csv" not in r1["missing_files"]
    assert r1["extra_files_in_raw_dir"] == ["notes.txt"]
    r2 = d.discover(raw)
    r1.pop("run_info")
    r2.pop("run_info")
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_head_tail(raw: Path):
    s = d.head_tail(raw / "HI-Small_Trans.csv")
    assert s["head"][0].startswith("Timestamp")
    assert s["tail"][-1] == "2022/01/02 23:59,3,C1,3,C1,5,0"


def test_main_writes_json_and_flags_missing(raw: Path, tmp_path: Path):
    out = tmp_path / "o" / "discover.json"
    rc = d.main(["--raw-dir", str(raw), "--out", str(out)])
    assert rc == 1  # Medium files missing in this fixture
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data["files"]) == {
        "HI-Small_Trans.csv",
        "HI-Small_Patterns.txt",
        "HI-Small_accounts.csv",
    }
