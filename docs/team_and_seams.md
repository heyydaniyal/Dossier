# Team seams and ownership (Phase 0)

**Status (2026-09-28):** Daniyal Ahmad owns all four seams. Joana Martins, Maiara Almada and Mariana Martins attend every end-of-phase walkthrough and write their own defence notes (docs/defence_notes/), because the grade requires every member to defend every part.

| Seam | Owns | Phases led | Interfaces it produces |
|---|---|---|---|
| Modelling | alert rules, features, detection model, calibration, anomaly | P2 (alerts), P3, P4, P5 | `Alert`, `ScoreResult`, feature registry, band table |
| Agents | runtime, tools, prompts, orchestration, RAG corpus | P7, P9, P10, P11, P12 | `EvidenceItem`, `AgentOutput`, `CaseFile`, `TrajectoryRecord`, `ToolCall`, `GraphQuery` |
| App | Streamlit app, deployment, human decision gate | P0 deploy, P13 | `DecisionRecord` |
| **Evaluation harness (first-class)** | splits audit, simulated dispositions, AGENT-DEV/AGENT-TEST carve, Tier A/B ground truth, metrics, power analysis, ablation runner, `RunManifest` | P1, P2 (split + dispositions), P6, P8 (pilot), P14 | `AlertLabel` (eval-only), `RunManifest`, metrics tables |

Hard rules
- The evaluation-harness owner builds AGENT-TEST cases and **must not write or edit prompts**. If staffing forces overlap, all AGENT-TEST cases are built and committed in P6, before any prompt work in P8.
- Contracts are FROZEN v1 (`src/contracts`, `src/eval/labels.py`). Changes: bump version, dated entry in `docs/CONTRACTS_CHANGELOG.md`, announce to all four, log in PROJECT_STATE.md.
- Every seam merges only with passing `uv run pytest`.
