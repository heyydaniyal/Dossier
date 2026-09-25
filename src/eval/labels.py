"""Evaluation-only ground-truth contracts. Part of FROZEN v1.

This module lives in src/eval. src/agents, src/tools, src/rag and src/app must never import it
(tests/test_eval_isolation.py enforces this).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AlertLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alert_id: str
    is_true_positive: bool
    typologies: list[str]
    label_definition_id: str
