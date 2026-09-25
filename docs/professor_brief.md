# Dossier — Professor alignment brief (Phase 0)

**Team:** Daniyal Ahmad, Joana Martins, Maiara Almada, Mariana Martins · NOVA IMS capstone · plan: 10 weeks, deadline 2026-12-04

## Problem
Bank transaction-monitoring rules generate large alert volumes, most of which are false positives. Each alert needs a human investigation: gather the account's activity, look at counterparties, check procedure, write up a case, decide. Dossier is a web app in which a multi-agent LLM system does that investigation and hands a **cited case file and a recommendation to a human analyst, who decides**. The system never files a SAR, freezes an account or sets thresholds.

## Research question
Does a structured multi-agent architecture (specialisation, adversarial deliberation, verification, runtime planning) produce better and more defensible investigative decisions than (a) rules alone, (b) the detection model alone, (c) a single LLM agent with the same tools and no structure — and what does each component contribute? Answered by **ablation**. A null or negative result will be reported as such.

## Data
IBM *Transactions for Anti-Money Laundering* (Altman et al., NeurIPS 2023 Datasets & Benchmarks; Kaggle). Fully synthetic, labelled. No real persons. Customer/KYC attributes do not exist in the data and will be synthesised and documented. Temporal train/validation/calibration/test split with embargo gaps; the agent test set is a held-out group split used once.

## Planned results table (the deliverable of the project)
Every cell is estimated on the held-out AGENT-TEST cases, ≥3 repeated runs, with 95% confidence intervals.

| Configuration | What it tests | Decision quality (escalation precision / recall, cost-weighted error) | Defensibility (citation validity, unsupported claims) | Cost / latency per case | Run-to-run variability |
|---|---|---|---|---|---|
| Rules only (no LLM) | baseline (a) | | n/a | | n/a |
| Detection model only (no LLM) | baseline (b) | | n/a | | n/a |
| Single agent, same tools (**H1** baseline) | baseline (c) | | | | |
| **Full multi-agent system (H1)** | the proposal | | | | |
| − Deliberation (no Prosecution/Defence/Adjudicator) | value of adversarial debate | | | | |
| − Critic | value of verification | | | | |
| − Policy retrieval | value of document grounding | | | | |
| Fixed pipeline, no runtime planning (**H6**) | value of the Orchestrator | | | | |

*Hypothesis numbering beyond H1/H6 is finalised in the evaluation design (P6). The number of ablations actually run depends on API quota/budget; the minimum is three.*

## Minimum Defensible Project (MDP)
Rule-generated alert layer · one calibrated detection model (as an agent tool) · agents: Orchestrator, Enrichment, Scoring, Policy (retrieval over regulation + procedure), Prosecution, Defence, Adjudicator, Narrative, Critic · single-agent baseline · ≥3 ablations · evaluation harness · deployed Streamlit app with alert queue, case file view and a human confirm/escalate gate.
**Stretch (only after the MDP works end to end):** Network agent, spawned Entity Investigators, Typology agent, episodic memory, anomaly model, cross-variant (HI→LI) generalisation.

## Questions for the professor
a. Is "measure what each agent contributes" a sufficient capstone?
b. Is a null or negative ablation result acceptable if rigorously reported?
c. Is there a required agent framework, LLM provider, or deployment target?
d. What is the grading rubric and its weighting (app / agents / evaluation / report)?
e. Are synthetic financial data acceptable, and is an ethics/data statement or form required?

*Current plan pending (c): Python, own lightweight orchestrator with native function calling, Google Gemini API (free tier for development), Streamlit Community Cloud.*
