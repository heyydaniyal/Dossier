"""P1: convert raw IBM AML files to typed Parquet, verifying every input against its frozen hash.

Schemas below are the ones PRINTED by scripts.audit.discover (2026-09-29), not remembered ones.
Transactions: the raw header repeats "Account" (sender, receiver), so columns are renamed by
POSITION after asserting the exact raw header. Bank IDs and account numbers stay strings:
bank IDs carry leading zeros ("020") that an integer cast would destroy.

Streaming (Polars lazy + sink_parquet): the 3 GB CSVs never sit in RAM as a whole.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import polars as pl
import yaml

CONVERT_VERSION = "1.0.0"

RAW_TRANS_HEADER = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]
TRANS_COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]
RAW_ACCOUNTS_HEADER = ["Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name"]
ACCOUNTS_COLUMNS = ["bank_name", "bank_id", "account_number", "entity_id", "entity_name"]
TS_FORMAT = "%Y/%m/%d %H:%M"  # minute resolution, seen in the data: "2022/09/01 00:17"

BEGIN_PREFIX = "BEGIN LAUNDERING ATTEMPT - "
END_PREFIX = "END LAUNDERING ATTEMPT - "
_PARAM = re.compile(r"^Max (\d+)(?: hops|-degree Fan-(?:In|Out))$")


class AuditError(RuntimeError):
    """A verified fact about the input no longer holds. Never swallowed."""


# ---------------------------------------------------------------- input verification


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_sources(config: Path) -> dict:
    return yaml.safe_load(config.read_text(encoding="utf-8"))["files"]


def verify_input(path: Path, expected: dict) -> str:
    if not path.is_file():
        raise AuditError(f"missing input file: {path}")
    size = path.stat().st_size
    if size != expected["size_bytes"]:
        raise AuditError(f"{path.name}: size {size} != frozen {expected['size_bytes']}")
    digest = sha256_file(path)
    if digest != expected["sha256"]:
        raise AuditError(f"{path.name}: sha256 {digest} != frozen {expected['sha256']}")
    return digest


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


def assert_header(path: Path, expected: list[str]) -> None:
    got = read_header(path)
    if got != expected:
        raise AuditError(f"{path.name}: header {got} != expected {expected}")


# ---------------------------------------------------------------- transactions


def _typed_trans(lf: pl.LazyFrame) -> pl.LazyFrame:
    """String columns -> typed columns. strict=True: any unparsable value raises."""
    return lf.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime("us"), TS_FORMAT, strict=True),
        pl.col("amount_received").cast(pl.Float64, strict=True),
        pl.col("amount_paid").cast(pl.Float64, strict=True),
        pl.col("is_laundering").cast(pl.Int8, strict=True),
    )


def scan_trans_csv(path: Path) -> pl.LazyFrame:
    assert_header(path, RAW_TRANS_HEADER)
    lf = pl.scan_csv(
        path,
        has_header=True,
        new_columns=TRANS_COLUMNS,
        infer_schema=False,  # everything String first; we cast explicitly
    ).with_row_index("row_id")  # 0-based position in the raw file (data rows only)
    return _typed_trans(lf)


# ---------------------------------------------------------------- accounts


def scan_accounts_csv(path: Path) -> pl.LazyFrame:
    assert_header(path, RAW_ACCOUNTS_HEADER)
    return (
        pl.scan_csv(path, has_header=True, new_columns=ACCOUNTS_COLUMNS, infer_schema=False)
        .with_row_index("row_id")
        .with_columns(
            pl.col("entity_name").str.extract(r"^(.*) #\d+$", 1).alias("entity_type"),
        )
    )


# ---------------------------------------------------------------- patterns


def parse_patterns(path: Path) -> pl.DataFrame:
    """Parse BEGIN/END blocks. Structure verified in discovery: no nesting, balanced, 11 fields.

    The END line carries only the typology name; the BEGIN line may add a parameter
    ("CYCLE:  Max 12 hops", "FAN-IN:  Max 9-degree Fan-In"). Unknown formats raise.
    """
    rows: list[list] = []
    attempt = -1
    typology: str | None = None
    param: int | None = None
    idx = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(BEGIN_PREFIX):
                if typology is not None:
                    raise AuditError(f"{path.name}:{lineno}: nested BEGIN")
                body = line[len(BEGIN_PREFIX) :]
                name, _, rest = body.partition(":")
                rest = " ".join(rest.split())
                if rest:
                    m = _PARAM.match(rest)
                    if not m:
                        raise AuditError(f"{path.name}:{lineno}: unknown parameter {rest!r}")
                    param = int(m.group(1))
                else:
                    param = None
                typology = name.strip()
                attempt += 1
                idx = 0
            elif line.startswith(END_PREFIX):
                name = line[len(END_PREFIX) :].strip()
                if typology is None or name != typology:
                    raise AuditError(
                        f"{path.name}:{lineno}: END {name!r} does not close {typology!r}"
                    )
                typology = None
            else:
                if typology is None:
                    raise AuditError(f"{path.name}:{lineno}: transaction outside a block")
                fields = next(csv.reader([line]))
                if len(fields) != 11:
                    raise AuditError(f"{path.name}:{lineno}: {len(fields)} fields, expected 11")
                rows.append([attempt, typology, param, idx, *fields])
                idx += 1
    if typology is not None:
        raise AuditError(f"{path.name}: unclosed block at end of file")
    schema = {"attempt_id": pl.Int32, "typology": pl.String, "typology_param": pl.Int32}
    schema |= {"txn_index": pl.Int32} | {c: pl.String for c in TRANS_COLUMNS}
    df = pl.DataFrame(rows, schema=schema, orient="row")
    return _typed_trans(df.lazy()).collect()


# ---------------------------------------------------------------- conversion driver


def _sidecar(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".meta.json")


def _is_current(out: Path, src_sha: str) -> bool:
    meta = _sidecar(out)
    if not (out.is_file() and meta.is_file()):
        return False
    m = json.loads(meta.read_text(encoding="utf-8"))
    return m.get("source_sha256") == src_sha and m.get("convert_version") == CONVERT_VERSION


def _write_meta(out: Path, src: Path, src_sha: str, n_rows: int) -> None:
    meta = {
        "source": src.name,
        "source_sha256": src_sha,
        "convert_version": CONVERT_VERSION,
        "n_rows": n_rows,
    }
    _sidecar(out).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def convert_variant(
    variant: str, raw_dir: Path, out_dir: Path, sources: dict, force: bool = False
) -> dict[str, Path]:
    """Verify the three raw files of one variant and write trans/accounts/patterns Parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # transactions
    src = raw_dir / f"{variant}_Trans.csv"
    exp = sources[src.name]
    print(f"  {src.name}: verifying sha256 ...", flush=True)
    sha = verify_input(src, exp)
    out = out_dir / f"{variant}_trans.parquet"
    if force or not _is_current(out, sha):
        print(f"  {src.name}: converting to Parquet (streaming) ...", flush=True)
        scan_trans_csv(src).sink_parquet(out, compression="zstd")
    n = pl.scan_parquet(out).select(pl.len()).collect().item()
    if n != exp["n_data_rows"]:
        raise AuditError(f"{out.name}: {n} rows != frozen {exp['n_data_rows']}")
    _write_meta(out, src, sha, n)
    paths["trans"] = out

    # accounts
    src = raw_dir / f"{variant}_accounts.csv"
    exp = sources[src.name]
    print(f"  {src.name}: verifying and converting ...", flush=True)
    sha = verify_input(src, exp)
    out = out_dir / f"{variant}_accounts.parquet"
    if force or not _is_current(out, sha):
        scan_accounts_csv(src).sink_parquet(out, compression="zstd")
    n = pl.scan_parquet(out).select(pl.len()).collect().item()
    if n != exp["n_data_rows"]:
        raise AuditError(f"{out.name}: {n} rows != frozen {exp['n_data_rows']}")
    _write_meta(out, src, sha, n)
    paths["accounts"] = out

    # patterns
    src = raw_dir / f"{variant}_Patterns.txt"
    exp = sources[src.name]
    print(f"  {src.name}: verifying and parsing ...", flush=True)
    sha = verify_input(src, exp)
    out = out_dir / f"{variant}_patterns.parquet"
    if force or not _is_current(out, sha):
        parse_patterns(src).write_parquet(out, compression="zstd")
    df = pl.read_parquet(out)
    if df.height != exp["n_pattern_txns"]:
        raise AuditError(f"{out.name}: {df.height} txns != frozen {exp['n_pattern_txns']}")
    n_att = df["attempt_id"].n_unique()
    if n_att != exp["n_attempts"]:
        raise AuditError(f"{out.name}: {n_att} attempts != frozen {exp['n_attempts']}")
    _write_meta(out, src, sha, df.height)
    paths["patterns"] = out
    return paths
