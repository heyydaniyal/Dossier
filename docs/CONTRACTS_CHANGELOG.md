# Contracts changelog

## v1.1.0 — 2026-10-05 — FROZEN (additive, stricter)
Independent P2 review, finding m-4 (P3 task 0; approved by Dani 2026-10-05).
- `FORBIDDEN_FIELDS` gains the truth-derived columns of the P2 evaluation store: `attempt_id`, `attempt_ids`, `has_unattributed`, `n_laundering_txns`, `is_pos`, `planted`, `laundering_account`, `agent_group`. Before, only a test-local list covered them, so the P7 runtime evidence scanner would have let `attempt_ids` through.
- Deliberately NOT added: `typology`/`typologies` (the Typology agent may name typologies as its own reasoning) and `component` (too generic).
- `find_forbidden_fields` normalises keys before matching (trim, lowercase, spaces and hyphens → `_`), so the raw column name `Is Laundering` is caught too (second-pass review).
- No model or field changed. Nothing produced in P2 carries any of these keys in agent-visible payloads.
- Team announcement: required (Dani to post).

## v1.0.0 — 2026-09-25 — FROZEN
Finalised from draft v0 (Master Context). Differences from v0, all deliberate:
- `GraphQuery.lookback` → `lookback_hours` (float, >0): unit made explicit; `max_hops` bounded 1–3.
- `AgentOutput.budget_used` typed as `BudgetUsed` (steps, llm_calls, tokens, cost_eur).
- `DecisionRecord.human_decision` typed as enum `HumanDecision`; a decision requires reason + timestamp.
- Added `HistoricalDisposition` {alert_id, account_key, disposition, closed_at}: the only truth-derived record allowed in the runtime (ground-truth firewall).
- `RunManifest` adds `model_revisions`, `llm_cache_enabled` (rule 12) and `contracts_version`.
- `TrajectoryRecord` adds `ts`; token/cost/latency fields explicit.
- Validation added: tz-aware timestamps; `as_of > window_end`; `account_key` = `bank|account`; score inside its calibration band; citation cites exactly one of evidence/doc section; citations and policy steps must resolve to evidence in the case; `done` policy steps need evidence; forbidden ground-truth fields rejected recursively in `EvidenceItem.payload` and `Alert.rule_features`.
- Point-in-time tie rule: visible iff `event_ts < as_of`; ties excluded (`is_visible`).
- `AlertLabel` lives in `src/eval/labels.py`, not in the shared package.
