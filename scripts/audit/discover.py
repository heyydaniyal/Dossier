"""Phase 1, step 1: file discovery. Look inside the raw files before assuming anything.

Records, for every raw IBM AML file in data/raw/:
  - size, SHA-256, line count, line endings, UTF-8 validity, BOM
  - first and last raw lines (so the real header and timestamp format are seen, not assumed)
  - CSV files: parsed header, duplicate column names, field-count distribution over ALL rows
  - Patterns files: line census (BEGIN / END / transaction / blank / other), BEGIN/END
    balance and nesting, block sizes, and header templates (digits masked) with counts

Standard library only, streaming, constant memory: safe on a 12 GB laptop with multi-GB files.
Nothing here depends on the column names, so it cannot be wrong about the schema.

Usage (from the repo root):
    uv run python -m scripts.audit.discover
    uv run python -m scripts.audit.discover --raw-dir data/raw --out artifacts/p1/discover.json

Output: artifacts/p1/discover.json (gitignored: it holds a few raw sample rows).
Everything outside "run_info" is deterministic: re-running must give identical JSON.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
import platform
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

SCRIPT_VERSION = "1.0.0"
EXPECTED_VARIANTS = ("HI-Medium", "LI-Medium", "HI-Small")
KINDS = ("Trans.csv", "Patterns.txt", "accounts.csv")
CHUNK = 8 * 1024 * 1024
N_HEAD = 12
N_TAIL = 6
TAIL_BYTES = 64 * 1024
MAX_TEMPLATES = 60
MAX_EXAMPLES = 3
PROGRESS_EVERY = 5_000_000

csv.field_size_limit(10_000_000)


def expected_files(variants: tuple[str, ...] = EXPECTED_VARIANTS) -> list[str]:
    return [f"{v}_{k}" for v in variants for k in KINDS]


# ---------------------------------------------------------------- byte-level scan


def scan_bytes(path: Path) -> dict:
    """One binary pass: hash, size, newline counts, UTF-8 validity. Constant memory."""
    h = hashlib.sha256()
    size = n_lf = n_crlf = 0
    first = b""
    prev_last = b""
    last_byte = b""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    utf8_error: str | None = None
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            if not first:
                first = chunk[:3]
            h.update(chunk)
            n_lf += chunk.count(b"\n")
            n_crlf += chunk.count(b"\r\n")
            if prev_last == b"\r" and chunk[:1] == b"\n":  # CRLF split across chunks
                n_crlf += 1
            if utf8_error is None:
                try:
                    decoder.decode(chunk)
                except UnicodeDecodeError as e:
                    utf8_error = f"invalid UTF-8 near byte offset {size + e.start}"
            size += len(chunk)
            prev_last = chunk[-1:]
            last_byte = chunk[-1:]
    if utf8_error is None:
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            utf8_error = "truncated UTF-8 sequence at end of file"
    ends_nl = last_byte == b"\n"
    n_lines = n_lf + (0 if (ends_nl or size == 0) else 1)
    if n_lf == 0:
        ending = "none"
    elif n_crlf == n_lf:
        ending = "CRLF"
    elif n_crlf == 0:
        ending = "LF"
    else:
        ending = "MIXED"
    return {
        "size_bytes": size,
        "sha256": h.hexdigest(),
        "n_lines": n_lines,
        "n_lf": n_lf,
        "n_crlf": n_crlf,
        "line_ending": ending,
        "ends_with_newline": ends_nl,
        "utf8_bom": first.startswith(codecs.BOM_UTF8),
        "utf8_valid": utf8_error is None,
        "utf8_error": utf8_error,
    }


def head_tail(path: Path, n_head: int = N_HEAD, n_tail: int = N_TAIL) -> dict:
    head: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for line in f:
            head.append(line.rstrip("\r\n"))
            if len(head) >= n_head:
                break
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(max(0, size - TAIL_BYTES))
        tail_raw = f.read()
    lines = tail_raw.decode("utf-8", errors="replace").splitlines()
    if size > TAIL_BYTES and lines:
        lines = lines[1:]  # first line of the window is probably cut
    tail = [ln for ln in lines if ln != ""][-n_tail:] if lines else []
    return {"head": head, "tail": tail}


# ---------------------------------------------------------------- CSV profile


def csv_profile(path: Path, label: str = "") -> dict:
    """Parse every row with the csv module; count fields per row. No type assumptions."""
    field_counts: Counter[int] = Counter()
    bad_examples: list[dict] = []
    n_rows = 0
    max_row_chars = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return {"header": [], "n_data_rows": 0, "empty": True}
        n_fields = len(header)
        for row in reader:
            n_rows += 1
            k = len(row)
            field_counts[k] += 1
            chars = sum(len(x) for x in row)
            if chars > max_row_chars:
                max_row_chars = chars
            if k != n_fields and len(bad_examples) < MAX_EXAMPLES:
                bad_examples.append({"data_row": n_rows, "n_fields": k, "row": row[:20]})
            if n_rows % PROGRESS_EVERY == 0:
                print(f"    {label}: {n_rows:,} rows parsed", flush=True)
    dup = sorted(name for name, c in Counter(header).items() if c > 1)
    return {
        "header": header,
        "header_n_fields": n_fields,
        "duplicate_header_names": dup,
        "header_whitespace_issues": [h for h in header if h != h.strip()],
        "n_data_rows": n_rows,
        "field_count_distribution": {str(k): v for k, v in sorted(field_counts.items())},
        "n_rows_wrong_field_count": sum(v for k, v in field_counts.items() if k != n_fields),
        "wrong_field_count_examples": bad_examples,
        "max_row_chars": max_row_chars,
        "empty": False,
    }


# ---------------------------------------------------------------- patterns profile

_DIGITS = re.compile(r"\d+")


def _template(line: str) -> str:
    return _DIGITS.sub("#", " ".join(line.split()))


def patterns_profile(path: Path) -> dict:
    """Census of the patterns file. Only structure, no assumption about its column meaning."""
    kinds: Counter[str] = Counter()
    begin_t: Counter[str] = Counter()
    end_t: Counter[str] = Counter()
    begin_ex: dict[str, list[str]] = {}
    end_ex: dict[str, list[str]] = {}
    txn_fields: Counter[int] = Counter()
    other_ex: list[str] = []
    block_sizes: list[int] = []
    depth = 0
    max_depth = 0
    txn_outside = 0
    end_without_begin = 0
    begin_end_mismatch = 0
    current = 0
    current_begin_kind = ""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                kinds["blank"] += 1
                continue
            up = line.upper()
            if up.startswith("BEGIN"):
                kinds["begin"] += 1
                t = _template(line)
                begin_t[t] += 1
                begin_ex.setdefault(t, [])
                if len(begin_ex[t]) < MAX_EXAMPLES:
                    begin_ex[t].append(line)
                depth += 1
                max_depth = max(max_depth, depth)
                current = 0
                current_begin_kind = t[len("BEGIN") :].strip()
            elif up.startswith("END"):
                kinds["end"] += 1
                t = _template(line)
                end_t[t] += 1
                end_ex.setdefault(t, [])
                if len(end_ex[t]) < MAX_EXAMPLES:
                    end_ex[t].append(line)
                if depth == 0:
                    end_without_begin += 1
                else:
                    depth -= 1
                    block_sizes.append(current)
                    if t[len("END") :].strip() != current_begin_kind:
                        begin_end_mismatch += 1
            else:
                row = next(csv.reader([line]))
                if len(row) >= 3:
                    kinds["transaction_like"] += 1
                    txn_fields[len(row)] += 1
                    if depth == 0:
                        txn_outside += 1
                    else:
                        current += 1
                else:
                    kinds["other"] += 1
                    if len(other_ex) < MAX_EXAMPLES:
                        other_ex.append(line)
    top_b = begin_t.most_common(MAX_TEMPLATES)
    top_e = end_t.most_common(MAX_TEMPLATES)
    bs = sorted(block_sizes)
    return {
        "line_kinds": dict(sorted(kinds.items())),
        "n_begin": kinds["begin"],
        "n_end": kinds["end"],
        "unclosed_blocks_at_eof": depth,
        "max_nesting_depth": max_depth,
        "end_without_begin": end_without_begin,
        "begin_end_template_mismatch": begin_end_mismatch,
        "transaction_lines_outside_blocks": txn_outside,
        "transaction_field_count_distribution": {str(k): v for k, v in sorted(txn_fields.items())},
        "block_size": (
            {
                "n_blocks": len(bs),
                "min": bs[0],
                "median": bs[len(bs) // 2],
                "max": bs[-1],
                "total_txn_in_blocks": sum(bs),
            }
            if bs
            else {"n_blocks": 0}
        ),
        "n_distinct_begin_templates": len(begin_t),
        "begin_templates": [{"template": t, "count": c, "examples": begin_ex[t]} for t, c in top_b],
        "n_distinct_end_templates": len(end_t),
        "end_templates": [{"template": t, "count": c, "examples": end_ex[t]} for t, c in top_e],
        "other_line_examples": other_ex,
    }


# ---------------------------------------------------------------- driver


def profile_file(path: Path) -> dict:
    t0 = time.perf_counter()
    out: dict = {"file": path.name}
    print(f"  {path.name}: hashing and counting lines ...", flush=True)
    out["bytes"] = scan_bytes(path)
    out["sample"] = head_tail(path)
    if path.name.endswith(".csv"):
        print(f"  {path.name}: parsing every CSV row ...", flush=True)
        out["csv"] = csv_profile(path, label=path.name)
    elif path.name.endswith("_Patterns.txt"):
        print(f"  {path.name}: census of pattern blocks ...", flush=True)
        out["patterns"] = patterns_profile(path)
    out["_elapsed_s"] = round(time.perf_counter() - t0, 1)
    return out


def discover(raw_dir: Path, variants: tuple[str, ...] = EXPECTED_VARIANTS) -> dict:
    expected = expected_files(variants)
    present = sorted(p.name for p in raw_dir.iterdir() if p.is_file()) if raw_dir.is_dir() else []
    missing = [f for f in expected if f not in present]
    extra = [f for f in present if f not in expected]
    files: dict[str, dict] = {}
    timings: dict[str, float] = {}
    for name in expected:
        p = raw_dir / name
        if p.is_file():
            prof = profile_file(p)
            timings[name] = prof.pop("_elapsed_s")
            files[name] = prof
    return {
        "script": "scripts.audit.discover",
        "script_version": SCRIPT_VERSION,
        "expected_files": expected,
        "missing_files": missing,
        "extra_files_in_raw_dir": extra,
        "files": files,
        "run_info": {  # volatile: excluded from reproducibility comparison
            "started_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "raw_dir": str(raw_dir),
            "elapsed_s_per_file": timings,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/p1/discover.json"))
    args = ap.parse_args(argv)

    print(f"Dossier P1 discovery v{SCRIPT_VERSION} — raw dir: {args.raw_dir.resolve()}")
    t0 = time.perf_counter()
    result = discover(args.raw_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nSummary")
    for name, prof in result["files"].items():
        b = prof["bytes"]
        extra = ""
        if "csv" in prof:
            c = prof["csv"]
            extra = f" | {c['n_data_rows']:,} data rows, {c['header_n_fields']} cols"
            if c["n_rows_wrong_field_count"]:
                extra += f", {c['n_rows_wrong_field_count']:,} malformed"
        if "patterns" in prof:
            pp = prof["patterns"]
            extra = f" | {pp['n_begin']:,} BEGIN / {pp['n_end']:,} END"
        print(f"  {name:28s} {b['size_bytes'] / 1e6:10.1f} MB  {b['n_lines']:>12,} lines{extra}")
    if result["missing_files"]:
        print(f"\nMISSING: {', '.join(result['missing_files'])}")
    if result["extra_files_in_raw_dir"]:
        print(f"Extra files (ignored): {', '.join(result['extra_files_in_raw_dir'])}")
    print(f"\nWrote {args.out} in {time.perf_counter() - t0:,.0f} s")
    return 1 if result["missing_files"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
