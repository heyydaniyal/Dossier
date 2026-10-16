# P4 — Detection model

Phase 4 design and evidence. Real-data numbers come only from results JSON files written on Dani's
laptop (`docs/p4/p4_diagnosis.json` first).

| File | Content |
|---|---|
| `configs/p4_diagnosis.yaml` | step 0 protocol and verdict rules, **declared before the real run** |
| `configs/p4_protocol.yaml` | model-selection protocol, **declared before any P4 model is fitted** |
| `src/models/diagnostics.py` | step 0, label-free |
| `src/eval/training_labels.py` | `load_fit_strata`: reporting strata built inside the eval module; refuses TEST |
| `scripts/p4/diagnose.py` | step 0 driver: TRAIN only, aggregates only |
| `configs/p4_leg_check.yaml` | step 0b protocol and verdict, **declared before the real run** |
| `src/features/legs.py` | payment-level features, point-in-time, label-free |
| `src/models/leg_check.py` | step 0b analysis, label-free |
| `scripts/p4/leg_check.py` | step 0b driver: TRAIN only, aggregates only |
| `configs/p4_models.yaml` | search space, metrics, cost ratio, stop rules, save rules — **declared before any model is fitted** |
| `src/models/metrics.py`, `families.py`, `store.py` | metrics; rules / logistic / LightGBM behind one interface; save and load **by hash** |
| `scripts/p4/models.py` | `cv` (TRAIN only) then `validate` (VALIDATION once): finalists, selection, stop rules, guard, operating point, model + card |

## 1. Decisions (validated 2026-10-16 before implementing; approved by Dani: "push ok go ahead")

| # | Decision | Evidence | Result |
|---|---|---|---|
| P4-1 | **Diagnose the seen/unseen gap first** (open risk 36), on TRAIN folds F1/F2 only | P3: base AUC 0.903 on seen accounts vs 0.519 on unseen. Known-answer worlds: recognition, population and pure noise are each classified correctly | Holds. Step 0 runs before any lever or model choice |
| P4-2 | **Task 10 (LI-Small) not done in P4** | LI-Small is not a frozen input (`configs/data_sources.yaml`). The variants decision (FROZEN 2026-09-29) names LI-Medium for regime C. LI-Medium is deferred until the MDP works end to end (2026-09-30, open risk 22) and needs a frozen-HI-thresholds mode that does not exist | Deviation from the phase prompt, recorded in `configs/p4_protocol.yaml`. P5 task 8 reports regime C as not evaluated unless the deferral is lifted |
| P4-3 | **LightGBM 4.7.0** (exact pin) | Adds 1 package (3.3 MB), with a Windows wheel in `uv.lock`. Early stopping uses the eval set we pass (the time-later fold). sklearn's HistGradientBoosting uses a *random* internal hold-out for early stopping and has **no row subsampling** (tasks 3 and 5 need both). Predictions are identical with 1 and 2 threads. Native `pred_contrib` equals shap's TreeExplainer exactly. **Correction:** shap also explains HistGradientBoosting exactly (additivity error 1e-14), so SHAP alone was not a reason; shap would add numba, llvmlite and pandas | Holds. `tests/test_p4_protocol.py` pins thread determinism, eval-fold early stopping and log-odds additivity for the pinned version |
| P4-4 | **Tune inside TRAIN; open VALIDATION once**, at most 5 finalists, with a declared rule: the simpler model wins unless the PR-AUC gain's paired account-bootstrap 95% CI excludes 0 | Simulation at 213 positives / 4,158 alerts: one look carries ±17–25% noise; best of 5 configs is 6–13% optimistic, best of 20 is 7–19%; two models are ranked correctly unless their true gap is below ≈ 0.03 PR-AUC. The simulation assumes independent rows, so the real spread is wider (accounts cluster). VALIDATION was already used in P2 (revision-2 count, KYC checks) and in the P3 guard (fold F3); this is allowed and listed in the protocol | Holds. `configs/p4_protocol.yaml` |
| P4-5 | **`scripts/p4` gets exactly the `scripts/p3` label-access allowance** (`src.eval.training_labels`, `src.eval.identity_guard`) | 4 planted widenings caught: one extra module, the whole `src.eval`, the driver importing the label builder, the label-free model module importing labels. `scripts/p5` is not granted ahead of time | Holds. `tests/test_eval_isolation.py` |
| P4-7 | **Step 0b: one-off use of transaction-level labels as the target of a DIAGNOSTIC payment model, TRAIN only** (approved by Dani 2026-10-16, "ok, lets go for your recommendation") | Step 0 Q-C contradicted the needle hypothesis (below). Before building levers 1–2 or deciding P4-6, test whether laundering payments of NEW accounts differ from normal ones at all | Dated deviation from the frozen ontology, scoped to this check: `src/eval/training_labels.load_fit_laundering_txns` refuses TEST; nothing fitted is kept; aggregates only |
| P4-8 | **Levers 1–2 not built; P4-6 closed: transaction labels stay "derive alert labels only" (frozen ontology unchanged)** (Dani 2026-10-16, "go with what you want to do") | Step 0b (§3): payment-max vs step-0 model on new accounts +0.003 AUC, CI [−0.039, +0.045] (inconclusive); without payment format −0.037, CI upper +0.005 (no_signal); inside a positive account the top payment is a laundering one 23.5% vs 14.2% by chance (within-account AUC 0.590). Per-stratum gains are descriptive, not pre-declared, partly the ACH signature, and the maximum over many payments is biased toward busy accounts (needles 0.410) | Go to baselines (rules → logistic regression → LightGBM) under `configs/p4_protocol.yaml` |
| P4-6 | **Transaction-level labels as training targets: CLOSED by P4-8** | Changes the FROZEN ontology ("transaction-level labels are used only to derive alert labels") and needs a new label API. It is consistent with label P2-v1: the maximum per-payment score estimates "did this account touch laundering today" | Decide after step 0 Q-C (are needle cases a large share of unseen positives, and at chance?) |

## 2. Step 0 — diagnosis (configs/p4_diagnosis.yaml)

- **Q-A, recognition or population?** Three arms, scored on the eval-day rows of seen accounts:
  - **full**: fitted on all fit rows;
  - **disjoint**: the seen accounts' own fit rows removed;
  - **control**: the same number of rows and positives removed from other accounts.
  - Also reported: **label memory**, which scores an account by its past labels alone.
- **Control-arm fix (2026-10-16, before any real run).** The first version crashed when repeat launderers hold most fit positives. The shortfall is now drawn from the seen accounts' own rows. This biases control toward disjoint, i.e. *against* recognition, so a `recognition` verdict stands. If more than 25% of the control's positives are borrowed this way, any other verdict becomes `control_unmatched`. Checked on simulated data: recognition survives an 87% top-up; a population world with the same top-up gives `control_unmatched`.
- **Q-B:** AUC of each feature on unseen accounts, with a 95% CI (screening only; 61 features tested).
- **Q-C:** model AUC per stratum of unseen positives: laundering share of the day's payments, fan-in vs other rules, typology, unattributed laundering.
- **Evidence:** 22 planted bugs caught in step 0 code and tests; metrics equal sklearn; the driver runs end to end on the fixture; the output holds aggregates only.

Run (Dani's laptop): `uv sync`, then `uv run python -m scripts.p4.diagnose`. Commit `docs/p4/p4_diagnosis.json`.

**Real-data result (Dani's laptop, 2026-10-16; numbers to be asserted by a facts test once the JSON
is committed).** Control top-up 0%; 40 seen / 328 unseen positives.

| Features | AUC seen: full / control / disjoint | label memory | AUC unseen | share of the seen gap lost without own rows | verdict |
|---|---|---|---|---|---|
| rule (19) | 0.923 / 0.921 / 0.782 | 0.884 | 0.512 [0.485, 0.540] | 34% (control−disjoint CI 0.091–0.183) | mixed |
| all (61) | 0.965 / 0.970 / 0.918 | 0.884 | 0.560 [0.534, 0.586] | 13% (CI 0.034–0.071) | mixed |

- Reading: the seen-account advantage is mostly a **population** difference (accounts alerted on several days are an easier population even when the model never saw them: disjoint 0.918), with a smaller **recognition** part. "Mixed" rather than "population" only because the CI upper (0.071) exceeds 0.05.
- Q-B: no feature has an unseen CI lower bound > 0.55 (best amt_std_usd 0.551); the model (0.560) already matches the best single feature → no under-use.
- Q-C (unseen positives 328): fan-in 90% (AUC 0.569); unattributed laundering 88% (0.549); laundering share < 10% n=140 AUC 0.612; 10–50% n=162 **0.500**; ≥ 50% n=26 0.656; typologies n ≤ 9 each.
- The declared Q-C reading did not anticipate this pattern (needles above chance, the middle at chance). It contradicts the needle hypothesis behind lever 1 → step 0b.

## 3. Step 0b — per-payment signal (configs/p4_leg_check.yaml)

- **Question:** on a new account's laundering day, do its laundering payments look different from its normal payments?
- **Payment features** (`src/features/legs.py`, same 24 h point-in-time window): amount, direction, cross-currency, cross-bank, near-threshold and round (frozen P2 definitions), ratio to the account's median payment, share of its volume, its number of payments; counterparty context that day (lever 2): the counterparty's payments, distinct receivers and senders, payments in the pair, how often the exact amount appears; payment format (reported with and without). No hour, weekday or identifier.
- **Model:** LightGBM, fixed (untuned) parameters, fitted on the fit days' payments with payment labels; normal payments capped at 300 per alert by keyed hash, re-weighted.
- **Score:** each eval-day alert = its maximum payment score (lever 1's best case); compared paired with the step-0 model on the same UNSEEN alerts.
- **Verdict (declared):** gain CI lower ≥ 0.03 AUC → `signal` (P4-6 returns with numbers; levers 1–2 worth building); CI upper < 0.03 → `no_signal` (levers 1–2 and P4-6 closed with evidence); else `inconclusive`.
- **Integrity:** every eval alert's label must equal the maximum of its payment labels (the run stops otherwise).
- **Evidence:**
  - known-answer payment features;
  - deletion, outside-window mutation and inside-window control;
  - label loader TRAIN-scoped and refusing TEST;
  - sampling order-independent and re-weighted;
  - simulated worlds (signal: payment-max 0.926 vs baseline 0.557, top payment laundering 0.89 vs chance 0.13; no signal: within-alert AUC 0.53) both classified correctly;
  - perfect seen-account scores cannot create a signal;
  - driver end to end on the fixture;
  - 15 planted bugs caught;
  - real-scale day (1.7M payments, 5,000 alerts incl. 20 hubs × 2,000 senders): 4 s, 2.2 GB peak.

Run (Dani's laptop): `uv run python -m scripts.p4.leg_check`. Commit `docs/p4/p4_leg_check.json`.

**Real-data result (Dani's laptop, 2026-10-16; asserted by `tests/test_p4_results_facts.py` once the
JSON is committed).** Integrity passed on all 6,730 eval alerts (alert label = max payment label).
Payments: F1 fit 147,685 sampled (603 laundering), eval 112,084 (277); F2 fit 183,307 (886), eval
248,632 (501). Unseen positives 328.

| | AUC unseen: payment-max | baseline | gain [95% CI] | verdict |
|---|---|---|---|---|
| primary | 0.573 | 0.570 | +0.003 [−0.039, +0.045] | **inconclusive** |
| without payment format | 0.533 | 0.570 | −0.037 [−0.080, +0.005] | **no_signal** |

- Payment-level AUC on new accounts: F1 0.712, F2 0.783 (laundering payments differ from *other accounts'* payments, mostly by format).
- Inside 321 positive new accounts: the top payment is a laundering one 0.235 vs chance 0.142; mean within-account AUC 0.590.
- Strata (descriptive): laundering share < 10% 0.410 vs 0.627; 10–50% 0.667 vs 0.505; ≥ 50% 0.873 vs 0.666; fan-in 0.542 vs 0.578; other rules (n=32) 0.860 vs 0.496; unattributed 0.531 vs 0.556.
- The baseline here (0.570) averages the seed scores before the AUC; step 0 (0.560) averaged per-seed AUCs. Same models.
- Decision P4-8 (§1).

## 4. Models (configs/p4_models.yaml)

**Stage `cv` (TRAIN only, folds F1/F2).**
- **Configurations (18, protocol max 20):**
  - rules: rule count, and the sum of each fired rule's log-lift;
  - logistic regression: C ∈ {0.01, 0.1, 1} × {no weighting, balanced};
  - LightGBM: leaves ∈ {7, 15} × min_child_samples ∈ {50, 200} × {no weighting, balanced}, early-stopped on each fold's eval day (average precision only);
  - the best logistic and best LightGBM again without payment format.
- **Reported per configuration:** pooled PR-AUC (all / seen / unseen), train PR-AUC (overfitting gap) and fold spread (stability).
- **Also reported:** learning curves over 25–100% of fit accounts, and permutation importance on both folds (sign flips counted).
- **Finalists:** the best per family (3 candidates) plus the two no-format models (report only).

**CV result (Dani's laptop, 2026-10-16; `docs/p4/p4_cv.json`).**
- Best pooled PR-AUC:
  - LightGBM (7 leaves, min_child 50, no weighting): 0.216 (seen 0.813, unseen 0.120; train 0.392; fold spread 0.017);
  - logistic (C 0.1, no weighting): 0.197;
  - rule count: 0.129.
- Class weighting lost in every family.
- Without payment format: LightGBM 0.138 (seen **0.458**), logistic 0.148. The LightGBM's top permutation feature is `share_fmt_ach`.
- **Consequence:** the declared "one feature > 50% of the explanation" stop rule might fire inside `validate`, after VALIDATION is opened.

**Stage `precheck` (added 2026-10-16, Dani "go ahead"; `configs/p4_precheck.yaml`, TRAIN only).**
- Fits each learned finalist exactly as `validate` will, on folds F1/F2, and reports explanation shares on the eval days.
- Reading of the worst candidate's top-feature share:
  - below 0.40: `validate` as declared;
  - 0.40–0.50: back to Dani;
  - above 0.50: the stop rule would fire, so the candidate set is decided now, on TRAIN evidence.
- Also prints the rules_weighted weights per fold.
- Kept out of `configs/p4_models.yaml` so the CV run stays valid.

**Stage `validate` (VALIDATION opened once; refused if `docs/p4/p4_validation.json` exists or any config changed after `cv`).**
- **Fit:** finalists fitted on all of TRAIN (LightGBM trees = mean of the folds' best iterations).
- **Metrics with 95% account-bootstrap CIs:** PR-AUC, precision at recall 0.8/0.9, recall at 5/10/20% review capacity, Brier (only where the raw score is a probability), ROC-AUC (secondary). Every number is given for all, seen and unseen accounts.
- **Selection:** the declared protocol rule.
- **Stop rules, in order:**
  1. VALIDATION PR-AUC > 0.95;
  2. one feature > 50% of the explanation (learned models only, since rule scores are R04-dominated by construction);
  3. any input that is not a registry feature;
  4. identity guard v2 not cleared on the chosen feature set (folds F1–F3).

  Any stop means no model is saved.
- **After the stop rules pass:**
  - cost-based operating point at FN:FP = 20:1 (sensitivities 5:1 and 50:1);
  - per-typology recall;
  - SHAP-equivalent contributions in log-odds;
  - permutation importance;
  - model saved **by hash** to `models/p4/`, then re-loaded and checked to reproduce its VALIDATION scores;
  - `docs/p4/MODEL_CARD.md` written.
- **Line endings:** `.gitattributes` keeps model bytes exact on Windows.

**Evidence (2026-10-16):**
- metrics match sklearn or a brute force (including ties and row order);
- every family's contributions sum to its margin;
- logistic margin matches an independent computation;
- save/load round trip is exact;
- tampering, prefixes and CRLF conversion are refused;
- the selection rule is checked on simulated scores, including the tie;
- each stop rule is tested;
- the guard wiring is checked against the frozen evaluator;
- CV and learning curves hand the eval fold to LightGBM;
- both stages run end to end on the fixture: VALIDATION read once, nothing saved on a stop, aggregates only;
- 21 planted bugs caught.

**Real-size timing (synthetic data, this workspace):** 16 configurations × 2 folds 13 s; VALIDATION CIs about 14 s per model.
