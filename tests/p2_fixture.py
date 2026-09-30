"""Small IBM-AML-shaped dataset spanning the real P2 calendar, with KNOWN laundering, for P2 tests.

Raw files are written in the exact raw formats and converted with the P1 converter, so the
Parquet schema is the real one. Variant name 'FIX'.

Known facts (asserted by the tests):
  - normal activity every day 2022-09-01..16 among 80 normal accounts (seeded), a tiny tail on
    09-17 containing one laundering txn (excluded by D1)
  - attempt 0 FAN-IN   09-03 (TRAIN):     L01..L08 -> L00
  - attempt 1 FAN-OUT  09-07 (VALIDATION): L10 -> L11..L18
  - attempt 2 CYCLE    09-11 L20->L21 (CAL), 09-12 L21->L22 (embargo), 09-13 L22->L20 (TEST)
  - attempt 3 FAN-IN   09-14 (TEST):      L30..L36 -> L37
  - attempt 4 FAN-OUT  09-17 (tail):      L00 -> L01 (outside the span)
  - unattributed laundering: 09-04 L40->L41 twice 9,500 USD; 09-15 L42->L43 7,000 USD
  - legitimate structuring: N05 sends 3 x 9,600 USD on 09-05
  - Euro -> USD fixed at 1.1 USD per EUR; Bitcoin at 20,000 USD per BTC
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.audit.convert import CONVERT_VERSION, parse_patterns, scan_accounts_csv, scan_trans_csv

VARIANT = "FIX"
HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering"
)
ACC_HEADER = "Bank Name,Bank ID,Account Number,Entity ID,Entity Name"
# the three real bank-name forms (discover_banks, 2026-09-30): named US bank, '<Country> Bank #n',
# and the data's own 'Crytpo Bank #n'
BANKS = [
    ("001", "Savings Bank of Seattle"),
    ("0020", "Spain Bank #20"),
    ("030", "Hearthstone Bancorp"),
    ("04", "Japan Bank #4"),
    ("05", "Crytpo Bank #5"),
]
TYPES = ["Corporation", "Partnership", "Sole Proprietorship", "Individual"]
EUR, BTC = 1.1, 20000.0


def _acct(prefix: str, i: int) -> tuple[str, str]:
    bank = BANKS[i % len(BANKS)][0]
    return bank, f"{prefix}{i:04X}"


NORMAL = [_acct("8000", i) for i in range(80)]
LAUND = [_acct("9000", i) for i in range(45)]


def L(i: int) -> tuple[str, str]:
    return LAUND[i]


def N(i: int) -> tuple[str, str]:
    return NORMAL[i]


def key(a: tuple[str, str]) -> str:
    return f"{a[0]}|{a[1]}"


def _amt(x: float) -> str:
    return f"{x:.2f}" if x >= 1 else f"{x:.6f}"


def _row(ts, s, r, amt_recv, rc, amt_paid, pc, fmt, y) -> str:
    return f"{ts},{s[0]},{s[1]},{r[0]},{r[1]},{_amt(amt_recv)},{rc},{_amt(amt_paid)},{pc},{fmt},{y}"


def _usd(ts, s, r, usd, fmt="ACH", y=0) -> str:
    return _row(ts, s, r, usd, "US Dollar", usd, "US Dollar", fmt, y)


def build(raw: Path) -> dict:
    raw.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for d in range(1, 17):
        rng = np.random.default_rng(1000 + d)
        for _ in range(150):
            s, r = rng.choice(len(NORMAL), 2, replace=False)
            ts = f"2022/09/{d:02d} {int(rng.integers(0, 24)):02d}:{int(rng.integers(0, 60)):02d}"
            usd = float(np.round(np.exp(rng.normal(7.3, 1.0)), 2))
            fmt = str(rng.choice(["Cheque", "Credit Card", "ACH", "Cash", "Wire", "Bitcoin"]))
            k = rng.random()
            if k < 0.08:  # Euro paid, USD received (cross-currency)
                eur = round(usd / EUR, 2)
                rows.append(_row(ts, N(s), N(r), eur * EUR, "US Dollar", eur, "Euro", fmt, 0))
            elif k < 0.12:  # Euro -> Euro
                rows.append(_row(ts, N(s), N(r), usd, "Euro", usd, "Euro", fmt, 0))
            elif k < 0.14:  # Bitcoin
                btc = round(usd / BTC, 6)
                rows.append(_row(ts, N(s), N(r), btc, "Bitcoin", btc, "Bitcoin", "Bitcoin", 0))
            else:
                rows.append(_usd(ts, N(s), N(r), usd, fmt))
        for i in range(3):  # self-transfers
            a = N((d * 7 + i) % 80)
            rows.append(_usd(f"2022/09/{d:02d} 0{i}:10", a, a, 500.0, "Reinvestment"))
        # a USD -> Euro cross-currency transaction (links USD and EUR in both directions)
        rows.append(
            _row(
                f"2022/09/{d:02d} 12:00",
                N(d),
                N(d + 20),
                1000.0,
                "Euro",
                1100.0,
                "US Dollar",
                "Wire",
                0,
            )
        )
        # a Bitcoin -> USD conversion (links BTC to USD)
        rows.append(
            _row(
                f"2022/09/{d:02d} 13:00",
                N(d + 1),
                N(d + 30),
                1000.0,
                "US Dollar",
                0.05,
                "Bitcoin",
                "Bitcoin",
                0,
            )
        )
    # legitimate structuring
    for m in (10, 20, 40):
        rows.append(_usd(f"2022/09/05 09:{m}", N(5), N(6), 9600.0, "Cash"))

    pat: list[tuple[str, str, list[str]]] = []
    fan_in = [
        _usd(f"2022/09/03 1{i}:00", L(i), L(0), 4000.0 + 100 * i, "ACH", 1) for i in range(1, 9)
    ]
    pat.append(("FAN-IN", "Max 8-degree Fan-In", fan_in))
    fan_out = [
        _usd(f"2022/09/07 1{i % 10}:30", L(10), L(10 + i), 3000.0 + 50 * i, "ACH", 1)
        for i in range(1, 9)
    ]
    pat.append(("FAN-OUT", "Max 8-degree Fan-Out", fan_out))
    cycle = [
        _usd("2022/09/11 08:00", L(20), L(21), 20000.0, "ACH", 1),
        _usd("2022/09/12 08:00", L(21), L(22), 19800.0, "ACH", 1),
        _usd("2022/09/13 08:00", L(22), L(20), 19600.0, "ACH", 1),
    ]
    pat.append(("CYCLE", "Max 3 hops", cycle))
    fan_in2 = [
        _usd(f"2022/09/14 1{i - 30}:15", L(i), L(37), 5000.0 + 10 * i, "ACH", 1)
        for i in range(30, 37)
    ]
    pat.append(("FAN-IN", "Max 7-degree Fan-In", fan_in2))
    tail = [_usd("2022/09/17 03:00", L(0), L(1), 1234.0, "ACH", 1)]
    pat.append(("FAN-OUT", "Max 1-degree Fan-Out", tail))
    for _, _, txs in pat:
        rows.extend(txs)
    rows += [
        _usd("2022/09/04 10:00", L(40), L(41), 9500.0, "ACH", 1),
        _usd("2022/09/04 11:00", L(40), L(41), 9500.0, "ACH", 1),
        _usd("2022/09/15 10:00", L(42), L(43), 7000.0, "ACH", 1),
        _usd("2022/09/17 05:00", N(1), N(2), 100.0, "Cheque", 0),
    ]
    order = np.random.default_rng(7).permutation(len(rows))  # file is not time-sorted
    (raw / f"{VARIANT}_Trans.csv").write_text(
        HEADER + "\n" + "\n".join(rows[i] for i in order) + "\n", encoding="utf-8"
    )

    lines = []
    for typ, param, txs in pat:
        lines.append(f"BEGIN LAUNDERING ATTEMPT - {typ}:  {param}")
        lines.extend(txs)
        lines.append(f"END LAUNDERING ATTEMPT - {typ}")
        lines.append("")
    (raw / f"{VARIANT}_Patterns.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    bank_name = dict(BANKS)
    acc_rows = []
    for j, (b, a) in enumerate(NORMAL + LAUND):
        acc_rows.append(f"{bank_name[b]},{b.lstrip('0') or '0'},{a},E{j:05d},{TYPES[j % 4]} #{j}")
    (raw / f"{VARIANT}_accounts.csv").write_text(
        ACC_HEADER + "\n" + "\n".join(acc_rows) + "\n", encoding="utf-8"
    )
    return {"n_trans": len(rows)}


def convert(raw: Path, interim: Path, sources_out: Path) -> None:
    interim.mkdir(parents=True, exist_ok=True)
    files = {}
    jobs = {
        "trans": (f"{VARIANT}_Trans.csv", lambda p: scan_trans_csv(p).collect()),
        "accounts": (f"{VARIANT}_accounts.csv", lambda p: scan_accounts_csv(p).collect()),
        "patterns": (f"{VARIANT}_Patterns.txt", parse_patterns),
    }
    for kind, (name, fn) in jobs.items():
        src = raw / name
        sha = hashlib.sha256(src.read_bytes()).hexdigest()
        out = interim / f"{VARIANT}_{kind}.parquet"
        df = fn(src)
        df.write_parquet(out)
        meta = {
            "source": name,
            "source_sha256": sha,
            "convert_version": CONVERT_VERSION,
            "n_rows": df.height,
        }
        out.with_suffix(".parquet.meta.json").write_text(json.dumps(meta), encoding="utf-8")
        files[name] = {"sha256": sha}
    sources_out.write_text(yaml.safe_dump({"files": files}), encoding="utf-8")


def make(tmp: Path) -> dict:
    raw, interim = tmp / "raw", tmp / "interim"
    info = build(raw)
    convert(raw, interim, tmp / "data_sources.yaml")
    return info | {"interim": interim, "sources": tmp / "data_sources.yaml"}
