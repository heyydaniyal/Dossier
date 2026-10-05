# Holdout and VALIDATION exposure log

Append-only. One row per event in which a holdout (TEST period, AGENT-TEST) or a selection period
(VALIDATION) was read for anything beyond its designated use. Started 2026-10-05 after the
independent P2 review (finding M-5): git alone cannot prove the order of events, so from now on
every exposure is written here, with the UTC time and the hash of the frozen artefact it must come
after, in the same commit as (or before) the result it produced.

Rules
- Add a row BEFORE committing anything computed from the exposure. Never edit or delete a row; a
  correction is a new row that names the row it corrects.
- Rows 1-3 were reconstructed on 2026-10-05 from PROJECT_STATE.md and the results JSONs; their
  times are as recorded there, not logged live.

| # | utc_time | phase | period | what was read | used for | must come after (artefact sha256) |
|---|---|---|---|---|---|---|
| 1 | 2026-09-29 | P1 | TEST (HI, LI) | aggregate laundering label counts per period (HI holdout 9,207 / 8,811,868 txns; LI 4,235) | feasibility only (permitted single exposure) | none (no model or threshold existed) |
| 2 | 2026-09-30 | P2 | VALIDATION | aggregate positive account-days (3,411), written by `calibrate`; used to project VALIDATION true alerts under revision 1 (~218) | choice of calibration revision 2 (review M-4) | none |
| 3 | 2026-09-30T11:38:02Z | P2 | TEST | aggregate counts by `build`: alerts, positive alerts, per agent group, regime B, components (an earlier build the same day did the same; its time was not recorded) | feasibility gates; TRAIN deviation decided after seeing them, explicitly without re-tuning rules | configs/p2_rule_thresholds.yaml 90f9a4995e3fa52a (recorded inside the results JSON only) |
| 4 | 2026-10-05T20:56:37Z | P3 task 0 | TEST | `run_p2 split`: TEST alert ids, TEST labels, laundering links of all dates; aggregate counts per split variant and alerts moved vs v1 | AGENT-DEV/AGENT-TEST split v2 (review M-6), approved by Dani 2026-10-05 | configs/p2_agent_split_v2.yaml (hash recorded in the results JSON `build.agent_split.config_sha256`) |
