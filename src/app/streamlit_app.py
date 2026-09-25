"""Dossier - placeholder alert queue (Phase 0 deployment check).

Purpose: retire deployment risk in week one. Shows PLACEHOLDER alerts only, validated through
the FROZEN v1 Alert contract so the deployed app proves the shared package imports on the host.
No data, no models, no agents yet.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # Streamlit puts only the script dir on sys.path
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.contracts import CONTRACTS_STATUS, CONTRACTS_VERSION, Alert  # noqa: E402


def placeholder_alerts() -> list[Alert]:
    t = datetime(2022, 9, 5, 12, tzinfo=UTC)
    rows = [
        ("PLACEHOLDER-001", "000|DEMO0001", ["R_FAN_OUT"]),
        ("PLACEHOLDER-002", "000|DEMO0002", ["R_STRUCTURING", "R_VELOCITY"]),
        ("PLACEHOLDER-003", "000|DEMO0003", ["R_CYCLE"]),
    ]
    return [
        Alert(
            alert_id=a,
            account_key=k,
            window_start=t - timedelta(days=1),
            window_end=t,
            as_of=t + timedelta(seconds=1),
            triggered_rules=r,
            rule_features={},
            created_at=t + timedelta(seconds=1),
        )
        for a, k, r in rows
    ]


def main() -> None:
    st.set_page_config(page_title="Dossier - Alert queue", layout="wide")
    st.title("Dossier")
    st.caption("Multi-agent AML alert investigation - NOVA IMS capstone. Decision support only.")
    st.warning("Phase 0 placeholder. The alerts below are fake and exist only to test deployment.")
    alerts = placeholder_alerts()
    st.subheader(f"Alert queue ({len(alerts)})")
    st.dataframe(
        [
            {
                "alert_id": a.alert_id,
                "account": a.account_key,
                "rules": ", ".join(a.triggered_rules),
                "as_of (UTC)": a.as_of.isoformat(),
                "status": "not investigated",
            }
            for a in alerts
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.info(
        "Closing an alert will require analyst confirmation. The system never files SARs, "
        "freezes accounts or sets thresholds."
    )
    st.caption(f"Contracts {CONTRACTS_STATUS} v{CONTRACTS_VERSION}")


if __name__ == "__main__":
    main()
