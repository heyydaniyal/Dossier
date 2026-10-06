# P3 — Point-in-time features, splits and leakage tests

Phase 3 design and evidence. Real-data **numbers** come only from `docs/p3/p3_results.json`, written
by `scripts/p3/run_p3.py` on Dani's laptop and asserted by `tests/test_p3_results_facts.py`. Until
that file is committed, those tests are skipped and the exit gate is **not** met.

| File | Content |
|---|---|
| `configs/p3_features.yaml` | lookback, formats, peer stats, feature groups, permutation-test protocol and pass criteria, real-data check sizes. **Declared before any real run** |
| `src/features/compute.py` | batch path (polars + scipy.sparse) and pure per-account reference path |
| `src/features/registry.py` → `docs/p3/feature_registry.{md,json}` | 61 model features + 1 tool-only feature |
| `src/features/splits.py` | split applied to feature rows, boundary rules, unseen-account slice, agent-eval TEST rule |
| `src/features/network.py` | known-flagged counterparties (tool only, P10) |
| `src/features/pit.py` | shared point-in-time check helpers |
| `src/models/permutation.py` | shuffled-label test (label-free; labels passed in) |
| `src/eval/training_labels.py` | the only label loader for fitting code; refuses TEST |
| `scripts/p3/run_p3.py` | real-data driver |

## 1. Decisions (validated 2026-10-06 before implementing; approved by Dani: "go ahead")

| # | Decision | Evidence | Result |
|---|---|---|---|
| 1.1 | **No own-baseline features.** "Deviation from the account's own baseline" is replaced by within-day dynamics. | The frozen split rejects any lookback > 24 h (`load_split` with 36 h / 48 h → `L_max > window`). With daily windows, an alert's lookback is exactly its own day, so no earlier history is visible. | Holds. Reported as a limitation. The "stop and revisit boundaries" condition is not triggered: no feature needs more than 24 h. |
| 1.2 | **Peer group = all accounts active in the same visible window** (average-rank percentile). The phase prompt's "synthetic KYC sector" is not used. | `configs/p2_kyc.yaml`: "KYC fields are excluded from P3-P5 model features". That covers `entity_type` too. The sector is planted from burn-in behaviour and is a stable per-account attribute, which is the account-recognition channel P2 excluded. | Holds. The cross-sectional rank also absorbs the weekday/weekend volume swing (weekends ≈ 43% of weekday volume). |
| 1.3 | **No KYC model features** (phase task 3 lists them). | Same frozen decision. P2 also measured that adding KYC lowers PR-AUC (C3 < 0 on seen and unseen accounts). | Holds. Agents still read KYC as case context. |
| 1.4 | **Community feature = weakly-connected-component size** of the window graph, not a community-detection algorithm. | scipy 1.17.1 is already locked. A synthetic real-scale day (2M nodes, 2M edges, 20 hubs × 2,000 senders) takes 0.4 s and 0.22 GB. | Holds. No new dependency; scipy becomes explicit in `pyproject.toml` (lock metadata only, no new package). |
| 1.5 | **Label access only through `src/eval/training_labels.py`** (open risk 26). `scripts/p3` may import that module and `src.eval.identity_guard`, nothing else from `src.eval`. | The isolation scan (`tests/test_eval_isolation.py`) gives `scripts/p3` a narrow import allowance; every other check still applies. TEST is refused before any file is opened. | Holds; tested. |
| 1.6 | **Permutation protocol: shuffle ALL labels** within day blocks (fit days and the eval day) and score AP against the shuffled eval labels. **Pass:** null mean ≤ 1.15 × prevalence, and the real statistic above every null value. | Simulation at the real fold sizes (F1 eval 2,621 alerts / 170 positives; F2 4,109 / 198). Clean pipeline: null 1.02–1.04 × prevalence. Planted leak: 4.6 ×. **First draft corrected before any real run:** shuffling only the fit labels lifted a CLEAN pipeline to 1.38 ×, because a model fitted on noise is a random function of informative features and AP is convex in the correlation. With all labels shuffled, a clean pipeline with 20 strong, correlated features gave 1.036 ×. | Holds. The bound was tightened from my proposed 1.25 × to 1.15 ×. |
| 1.7 | **No wall-clock features** (hour of day, weekday, position in the window). | Hour 0 is a midnight batch (lift 0.27) and weekend laundering is 2.3 × weekday (DATA_CARD §4.6), both generator artefacts. TEST has no weekend. | Holds. `active_hours` counts hour buckets; it does not encode which hour. |
| 1.8 | **Payment-format shares are their own group** (`format_mix`), flagged as a generator signature. | ACH carries 86% of pre-holdout laundering (DATA_CARD §7). P4 runs a sensitivity check without it. | Holds. |
| 1.9 | **TEST feature matrix** is built only with `--include-test`. The script reads TEST transactions and alerts, never labels, and records only row counts and hashes. It prints exposure-log row 5 for Dani to append. | Dani's rule: ask before anything reads TEST. | Holds. Dani runs it consciously; `test_p3_results_facts` requires the log row if TEST was built. |

## 2. Split (tasks 1–2)

- Feature rows inherit the frozen P2 periods. `assign_periods` refuses burn-in, embargo and tail rows instead of dropping them, and checks the stored period against the split.
- **Boundary rules** (`check_boundaries`), checked separately because they differ when the embargo and L_max differ:
  1. `max(as_of in P) + embargo < min(as_of in Q)` (phase rule, strict);
  2. `min(as_of in Q) − L_max ≥ max(as_of in P)` (Q's feature windows never reach P's label windows).
  - Frozen split: the gap is 48 h, the embargo 24 h and L_max 24 h, so both rules hold with margin.
- **Unseen-account slice (regime B):** TEST alerts of accounts with no TRAIN alert, from the runtime alert store only. The TEST build asserts it equals the P2 record (10,265 alerts).
- **Task 2 (FROZEN):** `require_test_period` refuses any agent-evaluation alert that is not from TEST. The P6/P14 harness must call it on its case list.

## 3. Features (task 3; registry in `docs/p3/feature_registry.md`)

All 61 features see exactly `as_of − 24 h ≤ ts < as_of`. Legs are non-self transactions in both roles, matching the P2 rules.

| Group | n | What |
|---|---:|---|
| rule | 19 | The 12 frozen P2 statistics, `n_rules_triggered`, and 6 `fired_*` flags. **Recomputed** from transactions, not copied; the integrity check requires an exact match with the alert store. |
| behaviour | 18 | Amount level and dispersion; near-threshold, round-amount and cross-currency shares; currencies; counterparty count and concentration (HHI); repeat share; net flow; self-transfers |
| format_mix | 7 | Share of legs per payment format (generator signature) |
| dynamics | 5 | Active hour buckets, span, peak legs per hour, median inter-arrival, first-inflow → first-outflow lag |
| peer | 5 | Percentile of volume, legs, distinct senders, distinct receivers and largest leg among all accounts active in the window |
| graph | 7 | Reciprocal counterparties, directed 3-cycles, 2-hop in/out reach, counterparty degrees, component size (log10) |

- **Missing values** are NaN, never null. Examples: no inflow followed by an outflow, a single leg, no senders.
- **Known-flagged counterparties** (task 4) are not a model feature. They use edges in the window and only `confirmed_suspicious` dispositions closed **strictly** before as_of. P10 wraps this as the Network tool.
- **Fitted objects** (task 6): P3 fits no scaler, encoder or target encoding (a test scans `src/features` for this). The only fitted inputs are P2's FX table and rule thresholds, and a test checks both were fitted on TRAIN.

## 4. Leakage suite (all in pytest)

| Required test | Where | Covers |
|---|---|---|
| Deletion-recompute | `test_deletion_recompute_batch` (fixture, every alert); `run_p3 check` (2,000 real pre-TEST alerts) | every group: scalar, rolling, peer, entity aggregates, graph. KYC features do not exist; the anomaly score arrives in P5 and must be added then |
| Lookback ≤ L_max (measured, not declared) | `test_outside_window_mutation_changes_nothing`, plus rows exactly at `as_of` and at `as_of − 24 h − 1 min`; control: `test_inside_window_mutation_does_change_features` | every feature |
| Pure == batch | `test_pure_path_equals_batch_path` (fixture, every alert); `run_p3 check` (40 real alerts) | every feature; two independent implementations |
| Permutation (shuffled-label) | `tests/test_p3_permutation.py` (clean passes, leaking pipeline caught, block prevalence kept); `run_p3 permutation` (real, 200 permutations) | whole pipeline + model |
| Registry check | `test_registry_*`, `test_feature_code_never_touches_kyc_labels_or_eval`, `test_committed_registry_matches_code`, isolation scan | inputs, names, lookbacks, KYC / eval / label access |
| Split test | `test_split_boundaries_on_alerts`, `test_split_phase_rule_violation_is_caught`, `test_split_window_rule_violation_is_caught`, `test_rows_outside_the_split_are_refused` | boundaries |
| Graph test | `test_graph_edges_are_inside_the_window`, `test_graph_known_answers` | edges, cycles, reach, components |

**Mutation evidence (2026-10-06).** 17 planted bugs, each caught by the named tests:
- a future leak (+1 h);
- a 48 h lookback;
- peer ranks and the graph built over the whole dataset;
- a wrong 3-cycle count;
- self-transfers as edges;
- a flag tie leak (`<=`);
- a label loader that accepts TEST;
- a widened `scripts/p3` exemption;
- each split rule disabled or made non-strict;
- a label input in the registry;
- fit-only permutation;
- permutation ignoring day blocks;
- a KYC import;
- an agent-eval rule accepting CALIBRATION.

The first mutation run missed the split check: both rules then reduced to "gap ≥ 24 h". The rules were rewritten as the two distinct conditions above, each with its own test.

## 5. Real-data run (Dani's laptop)

```
uv sync
uv run python -m scripts.p3.run_p3 features       # TRAIN, VALIDATION, CALIBRATION matrices
uv run python -m scripts.p3.run_p3 check          # 2,000-alert deletion/mutation + 40 pure
uv run python -m scripts.p3.run_p3 permutation    # 200 shuffled-label refits, TRAIN only
uv run python -m scripts.p3.run_p3 guard          # identity guard v2 for 5 candidate groups
uv run pytest
```

- Then commit `docs/p3/p3_results.json`.
- **TEST matrix:** `run_p3 features --include-test` only with Dani's explicit go-ahead. Append the printed row to `docs/holdout_exposure_log.md` before committing.
- **Expected cost** (measured on synthetic real-scale data here; the laptop is slower):
  - batch features: about 5 s per 5,000-alert day plus the Parquet scan;
  - pure path: about 10 s per alert;
  - permutation: about 2–5 min;
  - peak memory: about 2 GB.

## 6. Identity guard and feature adoption

- The rule group is the base. Each candidate group (behaviour, format_mix, dynamics, peer, graph) is added on its own, using folds F1–F3 and the frozen v2 rules.
- `cleared_groups` in the results lists the groups the guard does not object to.
- Adoption is still P4's call, by time-series CV PR-AUC.
- A group with `account_recognition` or `harms_unseen` is not a model feature.
- `insufficient_data` or `inconclusive` mean "not cleared" and come back to Dani.

## 7. Limitations and items for P4

- **No own-history features:** the 24 h lookback sees one day of attempts that last about 4.6 days (open risk 4).
- **Peer ranks are cross-sectional within a day:** they normalise drift but carry no customer-type context, because KYC is excluded.
- **Generator signatures remain:** `format_mix`; the R07 (Cash/Bitcoin) statistic and flag; and R02's flag, which uses the entity-type peer group (a frozen P2 rule output).
- **Data size:** 61 features against 721 TRAIN positives (about 12 positives per feature). P4 should prefer a regularised or tree model and report CI width.
- **Storage order:** feature matrices are stored sorted by `alert_id`, an opaque keyed hash. They still must not be sliced unshuffled (D5).
