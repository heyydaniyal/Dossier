"""P2 facts on the REAL data (HI-Medium), checked against docs/p2/p2_results.json (produced by
`uv run python -m scripts.p2.run_p2` on Dani's laptop).

Two kinds of check:
  - gates and invariants (precision band, ceilings, deviations recorded with exact values);
  - DOC_FACTS: the measured numbers of docs/P2_ALERT_LAYER.md §6 (feasibility), §8 (KYC leakage,
    identity guard), §9 (planting), §13 (summary) and the §5 revision-2 actuals are rendered from
    the JSON and must appear verbatim in the doc (review finding M-3, P3 task 0: the P2 version of
    this file never opened the doc). Changing a number in either place fails the test.
    NOT covered: numbers quoted from the review (§10, §12, e.g. 83.5% hard share; they come from
    the review's arithmetic, not from the results JSON) and historical numbers from overwritten
    runs (§5, §8; marked as such in the doc).

P2 is closed, so a missing results file is a FAILURE, not a skip (review m-7).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.alerts.build import ALERT_COLUMNS
from src.data.kyc import KYC_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
RES_PATH = ROOT / "docs" / "p2" / "p2_results.json"
DOC_PATH = ROOT / "docs" / "P2_ALERT_LAYER.md"
RES = json.loads(RES_PATH.read_text(encoding="utf-8")) if RES_PATH.is_file() else {}
RULES = yaml.safe_load((ROOT / "configs" / "p2_rules.yaml").read_text(encoding="utf-8"))
KYC = yaml.safe_load((ROOT / "configs" / "p2_kyc.yaml").read_text(encoding="utf-8"))
DISP = yaml.safe_load((ROOT / "configs" / "p2_dispositions.yaml").read_text(encoding="utf-8"))


def test_results_file_exists_and_has_a_build():
    assert RES_PATH.is_file(), "docs/p2/p2_results.json is missing (P2 is closed: never skip)"
    assert "build" in RES


def test_results_are_for_the_primary_variant_and_complete():
    assert RES["variant"] == "HI-Medium"
    for k in ("stats", "calibration", "build", "stores", "split"):
        assert k in RES, k


def test_thresholds_file_is_the_one_used():
    f = ROOT / "configs" / "p2_rule_thresholds.yaml"
    assert (
        hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        == RES["build"]["thresholds_sha256_before_any_test_count"]
    )
    thr = yaml.safe_load(f.read_text(encoding="utf-8"))
    assert thr["calibrated_on"]["period"] == "TRAIN"


def test_alert_precision_is_realistic_and_recall_below_one():
    lo, hi = RULES["calibration"]["precision_band"]
    tr = RES["build"]["rule_metrics"]["TRAIN"]
    assert lo <= tr["precision"] <= hi
    assert tr["recall"] < RULES["calibration"]["max_layer_recall"]
    for per in ("VALIDATION", "CALIBRATION"):  # reported for every pre-TEST period
        assert RES["build"]["rule_metrics"][per]["precision"] is not None
    assert set(RES["build"]["rule_metrics"]) == {"TRAIN", "VALIDATION", "CALIBRATION"}


def test_no_rule_is_near_perfect_on_train():
    ts = RULES["calibration"]["too_strong"]
    for r, m in RES["build"]["rule_metrics"]["TRAIN"]["per_rule"].items():
        assert not (
            (m["precision"] or 0) > ts["precision_above"]
            and (m["recall"] or 0) > ts["recall_above"]
        ), r
    assert len(RES["build"]["rule_metrics"]["TRAIN"]["per_rule"]) >= 6


SPLIT = yaml.safe_load((ROOT / "configs" / "p2_split.yaml").read_text(encoding="utf-8"))


def test_feasibility_gates_pass_or_are_documented_deviations():
    f = RES["build"]["feasibility"]
    accepted = {d["gate"]: d for d in SPLIT["feasibility"].get("accepted_deviations") or []}
    failing = [g for g, ok in f["gates"].items() if not ok]
    for g in failing:
        assert g in accepted, f"gate {g} fails and is not a documented deviation"
        per = g.removesuffix("_min_positive")
        # the deviation must record exactly the measured value (no silent drift)
        assert f["counts"][per]["n_positive_alerts"] == accepted[g]["measured_positive_alerts"]
        assert accepted[g]["decided_by"] and accepted[g]["reason"]
    assert set(f["counts"]["TEST"]) == {"n_alerts", "n_positive_alerts", "n_days"}  # counts only


def test_kyc_leakage_within_ceilings_or_documented_with_guard():
    """A ceiling may fail only if (1) it is recorded in accepted_ceiling_failures with exactly the
    measured value, (2) the same test on unseen accounts passes every check (so the failure is
    account recognition, not a planted clue), and (3) the identity guard was applied to KYC."""
    k = RES["build"]["kyc"]
    lk = k["leakage"]
    assert lk["status"] == "measured"
    accepted = {a["check"]: a for a in KYC.get("accepted_ceiling_failures") or []}
    failing = [c for c, v in lk["checks"].items() if not v["pass"]]
    for c in failing:
        assert c in accepted, f"{c} fails and is not a documented ceiling failure"
        assert accepted[c]["measured_value"] == lk["checks"][c]["value"], c
        assert accepted[c]["decided_by"] and accepted[c]["reason"]
    for c in accepted:  # no stale entries: every accepted failure is a current failure
        assert c in failing, f"{c} is recorded as failed but passes now"
    if failing:
        un = k["leakage_unseen_accounts_report_only"]
        assert KYC["identity_guard"]["required_unseen_check_pass"]
        assert un["status"] == "measured" and un["all_pass"], un.get("checks")
        # pinned (review M-3): the docs and the "KYC is not a model feature" decision rest on it
        assert k["identity_guard"]["verdict"] == "account_recognition"


def test_identity_guard_recorded_with_declared_thresholds():
    g = RES["build"]["kyc"]["identity_guard"]
    assert g["thresholds"] == KYC["identity_guard"]["thresholds"]
    assert g["group"] == "kyc_all" and g["base"] == "txn"


def test_kyc_v2_planting_rates_recorded():
    k = RES["build"]["kyc"]
    assert k["version"] == 2
    assert KYC["planting"]["candidate_percentile"] == 0.95  # v2.1
    assert KYC["planting"]["planting_prob"] == 1.0
    p = k["planting"]
    assert p["method"] == KYC["planting"]["method"] == "burn_in_behaviour"
    for c in ("realised_rate_laundering", "realised_rate_legitimate"):
        assert p[c] is not None and 0 < p[c] < 1, c
    assert "TEST" not in p["scope"]


def test_dispositions_pre_test_only_with_errors():
    d = RES["build"]["dispositions"]
    assert set(d["by_period"]) == set(DISP["periods"]) == {"TRAIN", "VALIDATION", "CALIBRATION"}
    assert 0 < 1 - d["overall"]["accuracy"] < 0.2
    assert (
        d["hard_H1_shapeless_laundering"]["false_negative_rate"]
        > d["easy_laundering"]["false_negative_rate"]
    )


def test_runtime_store_schemas():
    s = RES["stores"]
    assert s["runtime/alerts.parquet"]["columns"] == ALERT_COLUMNS
    assert s["runtime/kyc.parquet"]["columns"] == KYC_COLUMNS
    assert s["runtime/dispositions.parquet"]["columns"] == [
        "alert_id",
        "account_key",
        "disposition",
        "closed_at",
    ]


# ---------------------------------------------------------------- doc claims vs JSON (M-3)


def _get(path: str):
    """JSON value at a dotted path. 'build.feasibility_doc_basis' = the ORIGINAL P2 build counts
    (the doc's §6 line describes the 2026-09-30 build): after the P3 split rebuild these live in
    p2v1_original.feasibility, before it in build.feasibility."""
    alias = (RES.get("p2v1_original", {}).get("feasibility") or RES["build"]["feasibility"])[
        "counts"
    ]
    o = RES
    for i, part in enumerate(path.split(".")):
        if i == 1 and path.startswith("build.feasibility_doc_basis."):
            o = alias
            continue
        o = o[part]
    return o


def _minus(x: str) -> str:
    return x.replace("-", "\u2212")  # the doc uses the typographic minus


FMT = {
    "int": lambda v: f"{v:,}",
    "pct1": lambda v: f"{100 * v:.1f}%",
    "pct2": lambda v: f"{100 * v:.2f}%",
    "f2": lambda v: _minus(f"{v:.2f}"),
    "sf2": lambda v: _minus(f"{v:+.2f}"),
    "f3": lambda v: _minus(f"{v:.3f}"),
    "f4": lambda v: _minus(f"{v:.4f}"),
    "h2": lambda v: f"{v:.2f} h",
    "str": str,
}
B = "build."
FE = B + "feasibility.counts."
KL = B + "kyc.leakage.checks."
KU = B + "kyc.leakage_unseen_accounts_report_only.checks."
KG = B + "kyc.identity_guard."
KP = B + "kyc.planting."
RM = B + "rule_metrics."
DI = B + "dispositions."
# (template with one {} per value, [(json path, format)]) -- each rendered line must be in the doc
DOC_FACTS = [
    (
        "TRAIN has **{}** positive alerts against a gate of 1,000",
        [(FE + "TRAIN.n_positive_alerts", "int")],
    ),
    (
        "positive alerts TRAIN {}, VALIDATION {}, CALIBRATION {}, TEST {} (AGENT-DEV {}, "
        "AGENT-TEST {}); largest TEST component {} of TEST positives; regime B {} positives",
        [
            (FE + "TRAIN.n_positive_alerts", "int"),
            (FE + "VALIDATION.n_positive_alerts", "int"),
            (FE + "CALIBRATION.n_positive_alerts", "int"),
            (FE + "TEST.n_positive_alerts", "int"),
            (B + "feasibility_doc_basis.TEST_agent_groups.AGENT-DEV.n_positive_alerts", "int"),
            (B + "feasibility_doc_basis.TEST_agent_groups.AGENT-TEST.n_positive_alerts", "int"),
            (
                B + "feasibility_doc_basis.TEST_components."
                "largest_component_share_of_positive_alerts",
                "pct2",
            ),
            (FE + "TEST_regime_B_unseen_accounts.n_positive_alerts", "int"),
        ],
    ),
    (
        "**Corrected (v2):** chosen variant `{}`; AGENT-DEV {} positive alerts ({} alerts), "
        "AGENT-TEST {} ({}); largest TEST component {} alerts, {} of TEST positives",
        [
            (B + "agent_split.chosen_variant", "str"),
            (FE + "TEST_agent_groups.AGENT-DEV.n_positive_alerts", "int"),
            (FE + "TEST_agent_groups.AGENT-DEV.n_alerts", "int"),
            (FE + "TEST_agent_groups.AGENT-TEST.n_positive_alerts", "int"),
            (FE + "TEST_agent_groups.AGENT-TEST.n_alerts", "int"),
            (FE + "TEST_components.largest_component_alerts", "int"),
            (FE + "TEST_components.largest_component_share_of_positive_alerts", "pct2"),
        ],
    ),
    *[
        (
            f"| {label} | {{}} | {{}} | {{}} | yes |",
            [
                (
                    B + f"agent_split.variants.{v}.TEST_agent_groups.AGENT-DEV.n_positive_alerts",
                    "int",
                ),
                (
                    B + f"agent_split.variants.{v}.TEST_agent_groups.AGENT-TEST.n_positive_alerts",
                    "int",
                ),
                (
                    B + f"agent_split.variants.{v}.TEST_components."
                    "largest_component_share_of_positive_alerts",
                    "pct2",
                ),
            ],
        )
        for label, v in (
            ("A attempts, full membership", "A_attempts_full"),
            ("B + unattributed any date", "B_plus_unattributed_any_date"),
        )
    ],
    (
        "{} TEST alerts ({} positives) changed group compared with v1",
        [
            ("p2v1_original.moved_to_other_group_in_v2.n_alerts", "int"),
            ("p2v1_original.moved_to_other_group_in_v2.n_positive_alerts", "int"),
        ],
    ),
    (
        "| C1 KYC-only | {} × prev ✗ (ceiling 2.0) | {} × prev ✓ |",
        [(KL + "C1_kyc_only.value_over_prev", "f2"), (KU + "C1_kyc_only.value_over_prev", "f2")],
    ),
    (
        "| C2 synthetic increment | {} × prev ✗ (ceiling 0.25) | {} × prev ✓ |",
        [
            (KL + "C2_synthetic_increment.value_over_prev", "f2"),
            (KU + "C2_synthetic_increment.value_over_prev", "f2"),
        ],
    ),
    (
        "| C3 txn + KYC \u2212 txn | {} × prev ✓ | {} × prev ✓ |",
        [
            (KL + "C3_txn_plus_kyc_increment.value_over_prev", "f2"),
            (KU + "C3_txn_plus_kyc_increment.value_over_prev", "f2"),
        ],
    ),
    (
        "| C4 synthetic interaction | {} × prev ✓ | {} × prev ✓ |",
        [
            (KL + "C4_txn_synthetic_increment.value_over_prev", "f2"),
            (KU + "C4_txn_synthetic_increment.value_over_prev", "f2"),
        ],
    ),
    (
        "PR-AUC ({} → {} on all VALIDATION)",
        [(B + "kyc.leakage.pr_auc.txn", "f3"), (B + "kyc.leakage.pr_auc.txn_kyc_all", "f3")],
    ),
    (
        "seen accounts ({} alerts, {} positives) gain {} × prev; unseen accounts ({} alerts, "
        "{} positives) gain {} × prev → **`{}`**",
        [
            (KG + "seen.n", "int"),
            (KG + "seen.n_pos", "int"),
            (KG + "seen.gain_over_prev", "sf2"),
            (KG + "unseen.n", "int"),
            (KG + "unseen.n_pos", "int"),
            (KG + "unseen.gain_over_prev", "sf2"),
            (KG + "verdict", "str"),
        ],
    ),
    (
        "({} accounts with a pre-TEST alert; {} accounts planted overall)",
        [(KP + "n_alerted_accounts", "int"), (KP + "n_planted_all_accounts", "int")],
    ),
    (
        "| | Laundering accounts ({}) | Legitimate accounts ({}) |",
        [
            (KP + "n_laundering_alerted_accounts", "int"),
            (KP + "n_legitimate_alerted_accounts", "int"),
        ],
    ),
    (
        "| Has an explaining profile | {} | {} |",
        [(KP + "realised_rate_laundering", "pct1"), (KP + "realised_rate_legitimate", "pct1")],
    ),
    (
        "| Profile explains its own alert's rule | {} | {} |",
        [
            (KP + "explains_own_alert_rate_laundering", "pct1"),
            (KP + "explains_own_alert_rate_legitimate", "pct1"),
        ],
    ),
    # §13 measured summary (added in P3 task 0 so PROJECT_STATE's P2 numbers have a checked source)
    *[
        (
            f"| {per} | {{}} | {{}} | {{}} | {{}} |",
            [
                (RM + f"{per}.n_alerts", "int"),
                (RM + f"{per}.n_true_alerts", "int"),
                (RM + f"{per}.precision", "pct2"),
                (RM + f"{per}.recall", "pct1"),
            ],
        )
        for per in ("TRAIN", "VALIDATION", "CALIBRATION")
    ],
    (
        "R04 fan-in: {} of {} TRAIN true alerts, precision {}",
        [
            (RM + "TRAIN.per_rule.R04_FAN_IN.n_true", "int"),
            (RM + "TRAIN.n_true_alerts", "int"),
            (RM + "TRAIN.per_rule.R04_FAN_IN.precision", "pct1"),
        ],
    ),
    (
        "R03 structuring: TRAIN precision {}, recall {}",
        [
            (RM + "TRAIN.per_rule.R03_STRUCTURING.precision", "pct1"),
            (RM + "TRAIN.per_rule.R03_STRUCTURING.recall", "pct1"),
        ],
    ),
    (
        "R01 large single txn: {} alerts that only R01 fired",
        [(RM + "TRAIN.per_rule.R01_LARGE_SINGLE_TXN.n_alerts_only_this_rule", "int")],
    ),
    (
        "R04 true alerts that only R04 fired: {}",
        [(RM + "TRAIN.per_rule.R04_FAN_IN.n_true_only_this_rule", "int")],
    ),
    (
        "FX: {} currency pairs on TRAIN, max |log residual| {}",
        [("stats.fx.n_pairs", "int"), ("stats.fx.max_abs_log_residual", "f4")],
    ),
    (
        "Dispositions: {} pre-TEST alerts, accuracy {}, false-negative rate {} (TRAIN {} / "
        "VALIDATION {} / CALIBRATION {}), false-positive rate {}",
        [
            (DI + "n", "int"),
            (DI + "overall.accuracy", "pct1"),
            (DI + "overall.false_negative_rate", "pct1"),
            (DI + "by_period.TRAIN.false_negative_rate", "pct1"),
            (DI + "by_period.VALIDATION.false_negative_rate", "pct1"),
            (DI + "by_period.CALIBRATION.false_negative_rate", "pct1"),
            (DI + "overall.false_positive_rate", "pct1"),
        ],
    ),
    (
        "hard (shapeless) laundering FN {} vs easy {}; by number of rules 1 / 2 / 3+: FN {} / {} / "
        "{}, FP {} / {} / {}; confirmed_suspicious precision {}; median delay {} (confirmed) / "
        "{} (closed)",
        [
            (DI + "hard_H1_shapeless_laundering.false_negative_rate", "pct1"),
            (DI + "easy_laundering.false_negative_rate", "pct1"),
            (DI + "by_n_rules.1.false_negative_rate", "pct1"),
            (DI + "by_n_rules.2.false_negative_rate", "pct1"),
            (DI + "by_n_rules.3+.false_negative_rate", "pct1"),
            (DI + "by_n_rules.1.false_positive_rate", "pct1"),
            (DI + "by_n_rules.2.false_positive_rate", "pct1"),
            (DI + "by_n_rules.3+.false_positive_rate", "pct1"),
            (DI + "confirmed_precision", "pct1"),
            (DI + "delay_hours_median.confirmed_suspicious", "h2"),
            (DI + "delay_hours_median.closed_legitimate", "h2"),
        ],
    ),
]


@pytest.mark.parametrize("template,paths", DOC_FACTS, ids=lambda x: str(x)[:40])
def test_doc_numbers_match_results_json(template, paths):
    doc = DOC_PATH.read_text(encoding="utf-8")
    rendered = template.format(*(FMT[f](_get(p)) for p, f in paths))
    assert rendered in doc, f"not in docs/P2_ALERT_LAYER.md: {rendered!r}"


def test_doc_revision2_actuals_are_derived_from_the_json():
    """§5 correction (review m-8): derived counts, rendered from the JSON."""
    tr = RES["build"]["rule_metrics"]["TRAIN"]
    r04 = tr["per_rule"]["R04_FAN_IN"]
    other = tr["n_true_alerts"] - r04["n_true_only_this_rule"]  # >= 1 rule other than R04
    no_r04 = tr["n_true_alerts"] - r04["n_true"]
    days = RES["build"]["feasibility"]["counts"]["TRAIN"]["n_days"]
    val = RES["build"]["rule_metrics"]["VALIDATION"]["n_true_alerts"]
    s = (
        f"{other} TRAIN true alerts carry at least one rule other than R04 "
        f"({other / days:.0f}/day) and {no_r04} carry no R04 at all "
        f"({no_r04 / days:.0f}/day); VALIDATION has {val} true alerts"
    )
    assert s in DOC_PATH.read_text(encoding="utf-8"), s


def test_agent_split_v2_state_matches_the_exposure_log():
    """Until `run_p2 split` has run on the real data, the results JSON has no v2 split and the
    exposure log's split row is PENDING. Afterwards the row must carry a time, and the v2 result
    must satisfy the rule it was built for (no skip either way)."""
    log = (ROOT / "docs" / "holdout_exposure_log.md").read_text(encoding="utf-8")
    row = next(ln for ln in log.splitlines() if "`run_p2 split`" in ln)
    rep = RES["build"].get("agent_split")
    if rep is None:
        assert "| PENDING |" in row
        assert "p2v1_original" not in RES
        return
    assert "| PENDING |" not in row, "log the split run's UTC time in docs/holdout_exposure_log.md"
    assert rep["chosen_variant"] is not None
    d = rep["variants"][rep["chosen_variant"]]
    assert d["disjointness_on_full_memberships"]["n_attempt_nodes_in_both_groups"] == 0
    assert d["passes_gates"]
    assert RES["build"]["feasibility"]["gates"]["largest_component_share"]
    assert "p2v1_original" in RES and "moved_to_other_group_in_v2" in RES["p2v1_original"]
    assert "runtime/agent_dev_alert_ids.parquet" not in RES["stores"]
    assert "devtools/agent_dev_alert_ids.parquet" in RES["stores"]


def test_doc_label_generosity_numbers_match_the_committed_json():
    """§12 (review 10a): every rule row and the layer figures come from the committed JSON."""
    lg = json.loads(
        (ROOT / "docs" / "p2" / "review" / "label_generosity_train_val.json").read_text("utf-8")
    )
    doc = DOC_PATH.read_text(encoding="utf-8")
    tr, va = lg["per_period"]["TRAIN"], lg["per_period"]["VALIDATION"]
    names = {
        "R01_LARGE_SINGLE_TXN": "R01 large single txn",
        "R02_PEER_VOLUME_OUTLIER": "R02 peer volume",
        "R03_STRUCTURING": "R03 structuring",
        "R04_FAN_IN": "R04 fan-in",
        "R05_FAN_OUT": "R05 fan-out",
        "R07_HIGH_RISK_CHANNEL": "R07 high-risk channel",
    }
    for rid, label in names.items():
        t, v = tr["per_rule"][rid], va["per_rule"][rid]
        row = (
            f"| {label} | {t['n_true_alerts_fired']} | {t['pct_statistic_involves_laundering']}% "
            f"| {t['pct_coincidental_still_fires_without_laundering']}% "
            f"| {v['n_true_alerts_fired']} "
            f"| {v['pct_coincidental_still_fires_without_laundering']}% |"
        )
        assert row in doc, row
    assert f"{tr['layer']['pct_coincidental_for_every_fired_rule']}% of TRAIN true alerts" in doc
    assert f"(VALIDATION {va['layer']['pct_coincidental_for_every_fired_rule']}%)" in doc
    assert tr["n_true_alerts"] == RES["build"]["rule_metrics"]["TRAIN"]["n_true_alerts"]
    assert va["n_true_alerts"] == RES["build"]["rule_metrics"]["VALIDATION"]["n_true_alerts"]
    assert lg["max_transaction_timestamp_loaded"] < "2022-09-08"  # TRAIN/VALIDATION only
