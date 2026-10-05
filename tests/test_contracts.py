from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

import src.contracts as C
from src.contracts import (
    Alert,
    CalibrationBand,
    CaseFile,
    Citation,
    Deliberation,
    EvidenceItem,
    ForbiddenFieldError,
    PolicyStep,
    Recommendation,
    ScoreResult,
    is_visible,
)

T0 = datetime(2022, 9, 5, 12, tzinfo=UTC)


def _alert(**kw):
    base = dict(
        alert_id="A1",
        account_key="011|8000ABCD",
        window_start=T0 - timedelta(days=1),
        window_end=T0,
        as_of=T0 + timedelta(seconds=1),
        triggered_rules=["R1"],
        rule_features={"n_tx": 12},
        created_at=T0 + timedelta(seconds=1),
    )
    base.update(kw)
    return Alert(**base)


def _ev(eid="E1", payload=None):
    return EvidenceItem(
        evidence_id=eid,
        source_tool="get_transactions",
        tool_call_id="TC1",
        as_of=T0,
        claim_type="volume",
        payload=payload or {"n": 3},
        human_summary="3 tx",
    )


def test_frozen_v1_marker():
    assert C.CONTRACTS_STATUS == "FROZEN" and C.CONTRACTS_VERSION == "1.1.0"


def test_alert_valid():
    assert _alert().account_key == "011|8000ABCD"


@pytest.mark.parametrize(
    "kw",
    [
        {"account_key": "8000ABCD"},
        {"as_of": T0},  # as_of must be strictly after window_end
        {"window_start": T0 + timedelta(hours=1)},
        {"triggered_rules": []},
        {"created_at": datetime(2022, 9, 5)},  # naive
        {"rule_features": {"is_laundering": 1}},
        {"extra_field": 1},
    ],
)
def test_alert_rejects(kw):
    with pytest.raises((ValidationError, ForbiddenFieldError)):
        _alert(**kw)


def test_point_in_time_strict_and_ties_excluded():
    assert is_visible(T0 - timedelta(microseconds=1), T0)
    assert not is_visible(T0, T0)
    with pytest.raises(ValueError):
        is_visible(datetime(2022, 9, 5), T0)


def test_evidence_firewall_nested():
    with pytest.raises((ValidationError, ForbiddenFieldError)):
        _ev(payload={"counterparties": [{"acct": "x", "Is_Laundering": 1}]})


def test_score_must_be_inside_band():
    band = CalibrationBand(lo=0.2, hi=0.4, n_support=50, reliability_flag="reliable")
    kw = dict(
        alert_id="A1",
        model_id="m",
        model_version_hash="h",
        calibrator_hash="c",
        raw_score=1.2,
        calibration_band=band,
        top_features=[],
        explain_output_space="log-odds of the uncalibrated model",
    )
    ScoreResult(score=0.3, **kw)
    with pytest.raises(ValidationError):
        ScoreResult(score=0.9, **kw)


def test_citation_exactly_one_target():
    with pytest.raises(ValidationError):
        Citation(claim_id="c1")
    with pytest.raises(ValidationError):
        Citation(claim_id="c1", evidence_id="E1", doc_section_id="S1")


def test_casefile_rejects_dangling_citation():
    kw = dict(
        alert_id="A1",
        recommendation=Recommendation.ESCALATE,
        evidence=[_ev()],
        deliberation=Deliberation(),
        narrative="n",
        policy_steps={},
        trajectory_ref="t",
    )
    CaseFile(citations=[Citation(claim_id="c", evidence_id="E1")], **kw)
    with pytest.raises(ValidationError):
        CaseFile(citations=[Citation(claim_id="c", evidence_id="E999")], **kw)


def test_policy_done_needs_evidence():
    with pytest.raises(ValidationError):
        PolicyStep(status="done")


def test_models_are_immutable():
    a = _alert()
    with pytest.raises(ValidationError):
        a.alert_id = "B"
