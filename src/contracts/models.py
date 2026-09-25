"""Dossier interface contracts.

STATUS: FROZEN v1 (2026-09-25).
Any change must: bump CONTRACTS_VERSION, add a dated entry to docs/CONTRACTS_CHANGELOG.md,
be announced to the whole team, and be logged in PROJECT_STATE.md.

Design notes
- All models forbid unknown fields (extra="forbid") so a typo or a smuggled field fails loudly.
- All timestamps must be timezone-aware (UTC). Naive datetimes are rejected.
- Point-in-time rule: data visible to a query with as_of T has timestamp < T (strict).
  Ties (timestamp == as_of) are EXCLUDED. See `is_visible`.
- AlertLabel is NOT defined here. It lives in src/eval/labels.py, which the agent runtime
  cannot import (enforced by tests/test_eval_isolation.py).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTRACTS_VERSION = "1.0.0"
CONTRACTS_STATUS = "FROZEN"
CONTRACTS_FROZEN_ON = "2026-09-25"

# Field names that must never appear anywhere in agent-visible payloads (ground-truth firewall).
FORBIDDEN_FIELDS: frozenset[str] = frozenset(
    {
        "is_laundering",
        "is_true_positive",
        "label",
        "labels",
        "pattern_id",
        "pattern_instance_id",
        "typology_truth",
        "gt_typology",
        "ground_truth",
        "label_definition_id",
    }
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_tz(v: datetime | None) -> datetime | None:
    if v is not None and (v.tzinfo is None or v.tzinfo.utcoffset(v) is None):
        raise ValueError("timestamps must be timezone-aware (UTC)")
    return v


def is_visible(event_ts: datetime, as_of: datetime) -> bool:
    """Point-in-time visibility. Strictly before as_of; ties excluded."""
    _require_tz(event_ts)
    _require_tz(as_of)
    return event_ts < as_of


def find_forbidden_fields(obj: Any, path: str = "") -> list[str]:
    """Recursively scan dicts/lists/models for forbidden keys. Returns offending paths."""
    hits: list[str] = []
    if isinstance(obj, BaseModel):
        obj = obj.model_dump()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if str(k).lower() in FORBIDDEN_FIELDS:
                hits.append(p)
            hits.extend(find_forbidden_fields(v, p))
    elif isinstance(obj, list | tuple):
        for i, v in enumerate(obj):
            hits.extend(find_forbidden_fields(v, f"{path}[{i}]"))
    return hits


class ForbiddenFieldError(RuntimeError):
    """Raised when ground-truth fields leak into agent-visible data. Fails the run."""


def assert_no_forbidden_fields(obj: Any) -> None:
    hits = find_forbidden_fields(obj)
    if hits:
        raise ForbiddenFieldError(f"ground-truth firewall violation at: {hits}")


# ---------------------------------------------------------------- enums
class Recommendation(StrEnum):
    CLOSE = "close"
    ESCALATE = "escalate"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOVEL_PATTERN = "novel_pattern"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CRITIC_UNRESOLVED = "critic_unresolved"


class Disposition(StrEnum):
    CONFIRMED_SUSPICIOUS = "confirmed_suspicious"
    CLOSED_LEGITIMATE = "closed_legitimate"


class PolicyStepStatus(StrEnum):
    DONE = "done"
    NOT_DONE = "not_done"
    NA = "n/a"


class ToolStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"


class HumanDecision(StrEnum):
    CONFIRM_CLOSE = "confirm_close"
    ESCALATE = "escalate"
    REQUEST_MORE_INFO = "request_more_info"
    OVERRIDE_CLOSE = "override_close"


# ---------------------------------------------------------------- alert
class Alert(_Strict):
    alert_id: str
    account_key: str = Field(description="composite 'bank|account'")
    window_start: datetime
    window_end: datetime
    as_of: datetime
    triggered_rules: list[str] = Field(min_length=1)
    rule_features: dict[str, float | int | str | bool] = Field(default_factory=dict)
    created_at: datetime

    _tz = field_validator("window_start", "window_end", "as_of", "created_at")(_require_tz)

    @field_validator("account_key")
    @classmethod
    def _composite(cls, v: str) -> str:
        parts = v.split("|")
        if len(parts) != 2 or not all(parts):
            raise ValueError("account_key must be 'bank|account'")
        return v

    @model_validator(mode="after")
    def _order(self) -> Alert:
        if not self.window_start <= self.window_end:
            raise ValueError("window_start must be <= window_end")
        if self.as_of <= self.window_end:
            # as_of is strictly after the window so every window transaction is visible (< as_of).
            raise ValueError("as_of must be strictly after window_end")
        assert_no_forbidden_fields(self.rule_features)
        return self


# ---------------------------------------------------------------- scoring
class CalibrationBand(_Strict):
    lo: float = Field(ge=0, le=1)
    hi: float = Field(ge=0, le=1)
    n_support: int = Field(ge=0)
    reliability_flag: Literal["reliable", "low_support", "out_of_range"]

    @model_validator(mode="after")
    def _lohi(self) -> CalibrationBand:
        if self.lo > self.hi:
            raise ValueError("lo must be <= hi")
        return self


class FeatureContribution(_Strict):
    name: str
    value: float | int | str | bool | None
    contribution: float


class ScoreResult(_Strict):
    alert_id: str
    model_id: str
    model_version_hash: str
    calibrator_hash: str
    score: float = Field(ge=0, le=1, description="calibrated probability")
    raw_score: float
    calibration_band: CalibrationBand
    top_features: list[FeatureContribution]
    explain_output_space: str = Field(
        description="space of the contributions, e.g. 'log-odds of the uncalibrated model'. "
        "NOT calibrated-probability units."
    )

    @model_validator(mode="after")
    def _band_contains(self) -> ScoreResult:
        b = self.calibration_band
        if not (b.lo <= self.score <= b.hi):
            raise ValueError("score must lie inside its calibration band")
        return self


# ---------------------------------------------------------------- evidence & agents
class EvidenceItem(_Strict):
    evidence_id: str
    source_tool: str
    tool_call_id: str
    as_of: datetime
    claim_type: str
    payload: dict[str, Any]
    human_summary: str

    _tz = field_validator("as_of")(_require_tz)

    @model_validator(mode="after")
    def _firewall(self) -> EvidenceItem:
        assert_no_forbidden_fields(self.payload)
        return self


class BudgetUsed(_Strict):
    steps: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_eur: float = Field(ge=0)


class AgentOutput(_Strict):
    agent_name: str
    run_id: str
    evidence_ids_used: list[str]
    conclusion: str = Field(description="value from the agent's own conclusion enum")
    rationale: str
    confidence_note: str
    budget_used: BudgetUsed


class Deliberation(_Strict):
    prosecution: AgentOutput | None = None
    defence: AgentOutput | None = None
    adjudication: AgentOutput | None = None
    argument_order: list[Literal["prosecution", "defence"]] = Field(default_factory=list)


class Citation(_Strict):
    claim_id: str
    evidence_id: str | None = None
    doc_section_id: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Citation:
        if (self.evidence_id is None) == (self.doc_section_id is None):
            raise ValueError("citation needs exactly one of evidence_id or doc_section_id")
        return self


class PolicyStep(_Strict):
    status: PolicyStepStatus
    evidence_id: str | None = None

    @model_validator(mode="after")
    def _done_needs_evidence(self) -> PolicyStep:
        if self.status == PolicyStepStatus.DONE and not self.evidence_id:
            raise ValueError("a 'done' policy step must cite an evidence_id")
        return self


class CaseFile(_Strict):
    alert_id: str
    recommendation: Recommendation
    evidence: list[EvidenceItem]
    deliberation: Deliberation
    narrative: str
    citations: list[Citation]
    policy_steps: dict[str, PolicyStep]
    critic_report: dict[str, Any] | None = None
    trajectory_ref: str

    @model_validator(mode="after")
    def _citations_resolve(self) -> CaseFile:
        ids = {e.evidence_id for e in self.evidence}
        dangling = [
            c.evidence_id for c in self.citations if c.evidence_id and c.evidence_id not in ids
        ]
        dangling += [
            s.evidence_id
            for s in self.policy_steps.values()
            if s.evidence_id and s.evidence_id not in ids
        ]
        if dangling:
            raise ValueError(f"citations reference unknown evidence ids: {dangling}")
        return self


# ---------------------------------------------------------------- logging
class ToolCall(_Strict):
    tool_call_id: str
    run_id: str
    tool_name: str
    tool_version: str
    as_of: datetime
    input: dict[str, Any]
    input_hash: str
    output_hash: str | None
    duration_ms: float = Field(ge=0)
    status: ToolStatus
    error_code: str | None = None

    _tz = field_validator("as_of")(_require_tz)


class TrajectoryRecord(_Strict):
    """One line of the trajectory JSONL (global rule 7)."""

    run_id: str
    config_id: str
    agent: str
    prompt_id: str | None
    prompt_hash: str | None
    model_id: str | None
    temperature: float | None
    inputs: dict[str, Any]
    outputs: dict[str, Any] | None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_tokens: int = Field(ge=0, default=0)
    output_tokens: int = Field(ge=0, default=0)
    cost_eur: float = Field(ge=0, default=0.0)
    latency_ms: float = Field(ge=0, default=0.0)
    error: str | None = None
    ts: datetime

    _tz = field_validator("ts")(_require_tz)


class GraphQuery(_Strict):
    """The ONLY way to obtain graph information. The tool builds G(as_of) internally."""

    account_key: str
    as_of: datetime
    lookback_hours: float = Field(gt=0)
    max_hops: int = Field(ge=1, le=3)

    _tz = field_validator("as_of")(_require_tz)


class HistoricalDisposition(_Strict):
    """The only truth-derived record allowed into the runtime (noisy, pre-TEST only)."""

    alert_id: str
    account_key: str
    disposition: Disposition
    closed_at: datetime

    _tz = field_validator("closed_at")(_require_tz)


class DecisionRecord(_Strict):
    case_id: str
    recommendation: Recommendation
    reliability_flag: Literal["reliable", "low_support", "out_of_range"]
    evidence_ids: list[str]
    agent_run_id: str
    human_decision: HumanDecision | None = None
    human_reason: str | None = None
    decided_at: datetime | None = None

    @model_validator(mode="after")
    def _human_gate(self) -> DecisionRecord:
        if self.human_decision is not None and (not self.human_reason or self.decided_at is None):
            raise ValueError("a human decision needs a reason and a timestamp")
        return self


class RunManifest(_Strict):
    run_id: str
    config_hash: str
    prompt_hashes: dict[str, str]
    model_ids: dict[str, str]
    model_revisions: dict[str, str] = Field(default_factory=dict)
    temperature: float
    top_p: float | None
    seeds: dict[str, int]
    dataset_hash: str
    feature_registry_hash: str | None = None
    model_hash: str | None = None
    calibrator_hash: str | None = None
    band_table_hash: str | None = None
    retrieval_index_hash: str | None = None
    tool_versions: dict[str, str]
    code_commit: str
    llm_cache_enabled: bool
    contracts_version: str = CONTRACTS_VERSION
    started_at: datetime

    _tz = field_validator("started_at")(_require_tz)
