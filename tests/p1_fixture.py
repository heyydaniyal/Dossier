"""Tiny synthetic IBM-AML-shaped dataset with KNOWN answers, for testing the P1 audit.

Layout (all rows written in the exact raw formats seen in discovery):
  - 8 complete days 2022-09-01..08, 24 txns/day (one per hour, minute :00 / :30 alternating)
  - 1 incomplete tail day 2022-09-10 with 2 laundering txns (pattern truncation at the end)
  - banks "010" and "10" both exist -> 1 collision after stripping leading zeros
  - account "800A1" exists at banks "010" and "020" -> account number alone is NOT unique
  - Euro -> US Dollar at a fixed rate 1.1
  - 1 exact duplicate row, 1 label-conflict pair
  - laundering: all ACH (a planted perfectly separating artefact); amounts overlap normal ones
  - patterns: FAN-OUT (3 txns, day 2), CYCLE (2 txns: day 5 and the tail day 10),
    plus 1 pattern txn dated 2022-09-11 that is NOT in the transactions file;
    1 laundering txn (day 6) is in no pattern
Expected provisional holdout: 8 complete days -> start + floor(0.75*8)=6 days -> 2022-09-07.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering"
)
ACC_HEADER = "Bank Name,Bank ID,Account Number,Entity ID,Entity Name"


def _row(ts, fb, fa, tb, ta, ar, rc, ap, pc, fmt, y):
    return f"{ts},{fb},{fa},{tb},{ta},{ar},{rc},{ap},{pc},{fmt},{y}"


def build(raw: Path, variant: str = "HI-Small") -> dict:
    raw.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    normal_accts = [("010", "800B1"), ("020", "800B2"), ("10", "800B3"), ("0300", "800B4")]
    for d in range(1, 9):
        for h in range(24):
            ts = f"2022/09/{d:02d} {h:02d}:{'00' if h % 2 == 0 else '30'}"
            fb, fa = normal_accts[h % 4]
            tb, ta = normal_accts[(h + 1) % 4]
            if h == 5:  # self transfer (Reinvestment)
                rows.append(
                    _row(
                        ts,
                        fb,
                        fa,
                        fb,
                        fa,
                        "50.00",
                        "US Dollar",
                        "50.00",
                        "US Dollar",
                        "Reinvestment",
                        0,
                    )
                )
            elif h == 7:  # cross currency Euro -> USD at 1.1
                rows.append(
                    _row(ts, fb, fa, tb, ta, "110.00", "US Dollar", "100.00", "Euro", "Wire", 0)
                )
            else:
                amt = f"{100 + h + d:.2f}"
                rows.append(
                    _row(ts, fb, fa, tb, ta, amt, "US Dollar", amt, "US Dollar", "Cheque", 0)
                )
    # laundering (ACH) inside the normal days: pattern txns
    fan = [
        _row(
            "2022/09/02 03:15",
            "010",
            "800A1",
            "020",
            "800C1",
            "120.00",
            "US Dollar",
            "120.00",
            "US Dollar",
            "ACH",
            1,
        ),
        _row(
            "2022/09/02 04:15",
            "010",
            "800A1",
            "020",
            "800C2",
            "121.00",
            "US Dollar",
            "121.00",
            "US Dollar",
            "ACH",
            1,
        ),
        _row(
            "2022/09/02 05:15",
            "010",
            "800A1",
            "020",
            "800C3",
            "122.00",
            "US Dollar",
            "122.00",
            "US Dollar",
            "ACH",
            1,
        ),
    ]
    cyc_in = _row(
        "2022/09/05 10:15",
        "020",
        "800A1",
        "010",
        "800C9",
        "115.00",
        "US Dollar",
        "115.00",
        "US Dollar",
        "ACH",
        1,
    )
    cyc_tail = _row(
        "2022/09/10 10:15",
        "010",
        "800C9",
        "020",
        "800A1",
        "118.00",
        "US Dollar",
        "118.00",
        "US Dollar",
        "ACH",
        1,
    )
    cyc_missing = _row(
        "2022/09/11 09:00",
        "010",
        "800C9",
        "020",
        "800C8",
        "117.00",
        "US Dollar",
        "117.00",
        "US Dollar",
        "ACH",
        1,
    )
    unattributed = _row(
        "2022/09/06 11:15",
        "020",
        "800D1",
        "010",
        "800D2",
        "125.00",
        "US Dollar",
        "125.00",
        "US Dollar",
        "ACH",
        1,
    )
    tail2 = _row(
        "2022/09/10 11:15",
        "010",
        "800D5",
        "020",
        "800D6",
        "119.00",
        "US Dollar",
        "119.00",
        "US Dollar",
        "ACH",
        1,
    )
    dup = _row(
        "2022/09/03 12:45",
        "010",
        "800B1",
        "020",
        "800B2",
        "12.00",
        "US Dollar",
        "12.00",
        "US Dollar",
        "Cash",
        0,
    )
    conflict0 = _row(
        "2022/09/04 13:45",
        "010",
        "800B1",
        "020",
        "800B2",
        "13.00",
        "US Dollar",
        "13.00",
        "US Dollar",
        "Cash",
        0,
    )
    conflict1 = _row(
        "2022/09/04 13:45",
        "010",
        "800B1",
        "020",
        "800B2",
        "13.00",
        "US Dollar",
        "13.00",
        "US Dollar",
        "Cash",
        1,
    )
    rows += fan + [cyc_in, unattributed, dup, dup, conflict0, conflict1, cyc_tail, tail2]
    trans = raw / f"{variant}_Trans.csv"
    trans.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    patterns = raw / f"{variant}_Patterns.txt"
    patterns.write_text(
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 3-degree Fan-Out\n"
        + "\n".join(fan)
        + "\nEND LAUNDERING ATTEMPT - FAN-OUT\n\n"
        + "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 3 hops\n"
        + "\n".join([cyc_in, cyc_tail, cyc_missing])
        + "\nEND LAUNDERING ATTEMPT - CYCLE\n\n",
        encoding="utf-8",
    )

    acc_rows = [
        "First Bank of Testville,10,800B1,E0001,Corporation #1",
        "Spain Bank #7,20,800B2,E0002,Partnership #2",
        "First Bank of Testville,10,800B3,E0001,Corporation #1",
        "Bank of Nowhere,300,800B4,E0003,Sole Proprietorship #3",
        "First Bank of Testville,10,800A1,E0004,Corporation #4",
        "Spain Bank #7,20,800A1,E0004,Corporation #4",
    ]
    accounts = raw / f"{variant}_accounts.csv"
    accounts.write_text(ACC_HEADER + "\n" + "\n".join(acc_rows) + "\n", encoding="utf-8")

    def meta(p: Path, n_rows: int | None = None, **extra) -> dict:
        b = p.read_bytes()
        d = {
            "sha256": hashlib.sha256(b).hexdigest(),
            "size_bytes": len(b),
            "n_lines": b.count(b"\n"),
        }
        if n_rows is not None:
            d["n_data_rows"] = n_rows
        return d | extra

    return {
        trans.name: meta(trans, len(rows)),
        patterns.name: meta(patterns, n_attempts=2, n_pattern_txns=6),
        accounts.name: meta(accounts, len(acc_rows)),
    }


N_NORMAL = 8 * 24
N_LAUNDERING = 3 + 1 + 1 + 1 + 1 + 1  # fan(3) cyc_in unattributed conflict1 cyc_tail tail2
N_ROWS = N_NORMAL + 3 + 1 + 1 + 2 + 2 + 2
