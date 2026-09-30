# P2 — Alert layer, labels, KYC, dispositions

Phase 2 design. The **numbers** come from `docs/p2/p2_results.json`, written by
`scripts/p2/run_p2.py` on Dani's laptop, and are checked by `tests/test_p2_results_facts.py`.
Nothing below is a measured number unless it cites that file.

Configs, each declared **before** the first run on real data (the commit date is the evidence):

| File | Content |
|---|---|
| `configs/p2_split.yaml` | L_max, windows, period boundaries, agent-split method, feasibility gates |
| `configs/p2_rules.yaml` | rule set, floors, τ grid, precision target, "too strong" criterion |
| `configs/p2_kyc.yaml` | KYC generator, planting rates, **leakage ceilings** |
| `configs/p2_dispositions.yaml` | analyst error model, delays |

Written by the run:
- `configs/p2_fx_usd_per_unit.yaml` (TRAIN)
- `configs/p2_rule_thresholds.yaml` (TRAIN)

## 1. Temporal split (task 0)

- **L_max = 24 h.** Every rule, and every P3–P5 feature, for an alert with `as_of` T sees only `T − 24 h ≤ ts < T`.
- **Why 24 h.** 12 pre-TEST days = 1 burn-in day + 3 embargoes + TRAIN + VALIDATION + CALIBRATION. At 24 h that leaves 8 days for the three fitted periods. At 48 h only 4 would be left.

| Period | Window days (UTC) | Days | Weekdays |
|---|---|---|---|
| burn-in | 09-01 | 1 | Thu (generator warm-up; KYC activity source) |
| TRAIN | 09-02 … 09-05 | 4 | Fri–Mon |
| embargo | 09-06 | 1 | |
| VALIDATION | 09-07 | 1 | Wed |
| embargo | 09-08 | 1 | |
| CALIBRATION | 09-09 … 09-11 | 3 | Fri–Sun |
| embargo | 09-12 | 1 | |
| TEST | 09-13 … 09-16 | 4 | Tue–Fri |

Accepted limitations:
- CALIBRATION is weekend-heavy, while TEST has no weekend. P5 checks calibration by weekday/weekend.
- The median attempt lasts 4.6 days, so one attempt can span an embargo. Regime A accepts this; regime B is the stricter check.

## 2. Alert unit (task 1, FROZEN at phase end)

- **One alert = (account_key, calendar day), created when ≥ 1 rule fires.**
- Timing: `window_start` = D 00:00, `window_end` = D 23:59:59, `as_of = created_at` = D+1 00:00.
- **Both roles.** Fan-in is visible only from the receiver side and fan-out only from the sender side, so evaluating one role would blind half the rules.
- Self-transfers (same bank and account) are excluded: they are internal book transfers.
- De-duplication: one alert per account per day, carrying every rule that fired. Windows are disjoint days, so alerts never overlap. An account flagged on 3 consecutive days has 3 alerts.
- There are no alerts on burn-in or embargo days.
- `alert_id` is an opaque keyed hash, so it carries no ID order (D5).

## 3. Alert label (task 2, evaluation store only, id `P2-v1`)

**True positive** ⇔ the account is sender or receiver of ≥ 1 transaction with `Is Laundering = 1`, where `window_start ≤ ts < as_of`.

`typologies` lists the pattern typologies of those transactions. Laundering found in no pattern is flagged `has_unattributed` (35% of HI laundering, DATA_CARD §5), so typology metrics use attributed alerts only.

Edge cases:
- **Multi-day attempt:** each day is labelled by its own transactions. An account alerted on a day *between* its laundering transactions is a false positive that day. This is deliberate: the alert asks "is this day's activity suspicious?", and investigators judge the window.
- **Pass-through accounts:** positive on every day they send or receive a laundering transaction, in either role.
- **Attempt crossing a period boundary:** labelled per day, so both periods contain positives from the same attempt. The fixture's CYCLE runs CAL → embargo → TEST.
- **Laundering self-transfer:** counts as participation, but self-transfers never trigger rules.
- **Tail (≥ 09-17):** excluded by D1.

## 4. Rules (task 3), designed before measurement

| id | statistic per account-day | floor | domain source |
|---|---|---|---|
| R01 | largest single payment (USD) | 10k | CTR-style large value |
| R02 | total value in+out vs **entity-type peer group** | 10k | peer-group analysis (replaces "vs own history": needs > 24 h) |
| R03 | # payments in [9,000, 10,000) USD | 2 | structuring / smurfing |
| R04 | # distinct senders | 5 | fan-in / collection account |
| R05 | # distinct receivers | 5 | fan-out / distribution |
| R06 | min(in, out) if out/in ∈ [0.8, 1.25] and an outflow follows the first inflow | 5k | rapid in-and-out / pass-through |
| R07 | value via Cash or Bitcoin | 10k | high-risk channels |
| R08 | # cross-currency payments | 3 | currency churn |
| R09 | # payments that are exact multiples of 1,000 (original currency) | 3 | round amounts |

- ACH is **not** treated as high-risk, although it carries 86% of this generator's laundering. Real banks treat ACH as a low-risk domestic channel.
- R08 is expected to be weak, because this generator produces no cross-currency laundering. It is kept as a realistic noisy rule.

## 5. Threshold calibration (task 4, TRAIN only)

`threshold_r = max(floor_r, TRAIN quantile_τ(stat_r))`, with R02 computed per entity type.

**Revision 0** (declared before any run, now replaced): one global τ for all rules, chosen as the value whose TRAIN precision is closest to the **5% target** within the band [2%, 10%]. Industry reports 90–99% false positives, i.e. 1–10% precision. On HI-Medium TRAIN (run of 2026-09-30) it chose τ = 0.99995:
- precision 6.3%, so formally inside the band;
- but only 0.9% of laundering account-days caught, about 22 true alerts/day, which cannot meet the feasibility gates.

The per-rule diagnostic (`docs/p2/p2_train_diagnostics.json`, TRAIN only) showed the cause: noise rules (R06, R08, R09) forced the shared knob so strict that the useful rules barely fired.

**Revision 1** (approved by Dani 2026-09-30):
1. Each rule gets its **own** τ: the loosest grid level at which the rule alone reaches TRAIN precision ≥ 5% with ≥ 10 true alerts. A rule with no such level is dropped. Expected from the diagnostic: R06, R08 and R09 drop; R01–R05 and R07 stay.
   - Declared openly: the minimum of 10 true alerts was chosen *after* seeing the diagnostic. With 20, R01 would drop and only 5 rules would remain, below the task's minimum of 6.
2. **Unchanged:**
   - **Too strong** means TRAIN precision > 25% **and** recall > 5%. Such a rule gets its τ loosened step by step, or is dropped. Every action is logged in the thresholds file.
   - The combined precision must land in [2%, 10%].
   - At least 6 rules must stay active.
   - The layer's recall must stay below 95%.
   - The feasibility gates still apply.
3. Expected on TRAIN, from the diagnostic and before overlaps:
   - about 3,350 alerts/day and about 6% precision;
   - about 8% of laundering account-days caught.
   - **Fan-in produces about 72% of the true alerts**, because most laundering account-days are single "legs" that look ordinary, and only hub accounts stand out.
   - Laundering the rules miss is out of scope: the system triages alerts, it does not search for un-alerted laundering. This is reported as a limitation.

**Revision 2** (approved by Dani 2026-09-30, decided on TRAIN only, before any build):
- **What revision 1 gave on TRAIN** (measured):
  - precision 5.0%, recall 6.4%, about 3,200 alerts/day;
  - **fan-in produced 90% of the true alerts**, only about 16 true non-fan-in alerts per day;
  - VALIDATION was projected at about 218 true alerts, against a gate of 200.
- **Cause:** revision 1 required *every rule* to reach the *layer* target of 5%, and only fan-in manages that at volume.
- **Change:** a rule must now reach the band **floor** of 2% on its own (`per_rule_min_precision`). The layer must still land in [2%, 10%], and everything else is unchanged.
- **Expected from the per-rule table**, before overlaps: about 3.5% precision, about 65 true non-fan-in alerts/day, VALIDATION about 290. This is checked on the real `calibrate` output **before** `build`.
- **Realism check:** real banks report 2.8% of alerts becoming a SAR (MBCA survey) and about 4% (BPI, largest US banks).

**Note for P6 (approved with revision 1, refined 2026-09-30):** the evaluation sample is **stratified by triggered rule**, at least into two groups, *fan-in* and *other rules*, and agent results are reported per group. Per-rule strata are used only where there are enough real cases. The exact allocation is set in P6.

Task-5 reporting is in `p2_results.json → build.rule_metrics`:
- TRAIN and VALIDATION, per rule: volume, precision, recall, alerts only this rule caught, and pairwise overlap P(b | a).
- CALIBRATION: layer level only.
- TEST: counts only, computed after the thresholds' sha256 was recorded.

## 6. Feasibility and freeze (task 0, second half)

- Gates, declared in advance, on positive alerts: TRAIN ≥ 1000, VALIDATION ≥ 200, CALIBRATION ≥ 300, TEST ≥ 400.
- Each agent group needs ≥ 150 positives, and the largest TEST component may hold ≤ 20% of TEST positives.
- The boundaries may be adjusted **once** if a gate fails; then `p2_split.yaml` → `status: FROZEN`.
- **Documented deviation (decided by Dani, 2026-09-30):** TRAIN has **721** positive alerts against a gate of 1,000. It is recorded in `p2_split.yaml → feasibility.accepted_deviations` with the reason.
  - Moving boundaries cannot reach 1,000: adding 09-06 gives about 920 and starves CALIBRATION.
  - Re-tuning the rules after the TEST feasibility counts were computed would be harder to defend.
  - 721 is enough for the P4 model.
  - The gate value itself is unchanged, and `tests/test_p2_results_facts.py` accepts only deviations recorded with the exact measured value.

## 7. AGENT-DEV / AGENT-TEST

- The graph's nodes are accounts that have TEST alerts.
- Two accounts are joined if they share a pattern attempt (any date) or are the two sides of an unattributed laundering transaction in TEST.
- Each component goes wholly to one group, decided by the keyed hash of the component. The target is 50/50.
- This is executed in `src/eval`. The agent side receives only `agent_dev_alert_ids.parquet`.

## 8. Synthetic KYC (task 6), KYC v2

| Attribute | Source | Label access |
|---|---|---|
| entity_type | REAL (accounts file) | none |
| bank_location, bank_country | REAL, from the bank name. Formats printed by `discover_banks` (2026-09-30): '<Country> Bank #n' (32 countries), 'Crytpo Bank #n' (crypto platforms, country 'Crypto'), or a named US bank (468 names, e.g. 'Savings Bank of Seattle'; United States). An unknown '#n' prefix stops the run | none |
| country_risk | synthetic tier per country, seeded; US = low; Crypto = high (decided 2026-09-30) | none |
| sector_or_occupation | seeded draw by entity type, then v2 planting (§9) | none |
| expected_activity_band | rank of **burn-in day** volume within entity type, 25% ± 1-band noise; inactive accounts use a prior; v2 planting may raise it | none |
| onboarding_year | a uniform day in 2005-01-01 … 2022-08-31 is drawn; **only its year is stored** (v2) | none |
| customer_risk_rating | f(sector risk, country risk, very-high activity) ± noise, computed after planting | none |

- `kyc_as_of` = 2022-09-02 00:00, which is before every alert.
- Every draw is a keyed hash of (seed, stream, account_key). The whole generator, including planting, lives in `src/data/kyc.py` and is audited for ground-truth access.

**Leakage ceilings (declared in advance, unchanged by v2).** Fit on TRAIN alerts, evaluate on VALIDATION alerts; `prev` = VALIDATION prevalence.

| Check | Ceiling |
|---|---|
| C1 KYC-only PR-AUC | ≤ 2.0 × prev |
| C2 synthetic increment | ≤ 0.25 × prev |
| C3 PR-AUC(txn + KYC) − PR-AUC(txn) | ≤ 0.5 × prev |
| C4 synthetic interaction increment | ≤ 0.25 × prev |

A report-only rerun uses VALIDATION accounts with no TRAIN alert.

**Why v2** (KYC v1 measured by the 2026-09-30 build):
- C1 = 2.6× and C2 = 1.27× failed; C3 and C4 passed.
- The TRAIN/VALIDATION diagnostic (`docs/p2/p2_kyc_diagnostics.json`) found two causes:
  1. **The day-level onboarding date acted as an account fingerprint.** Adding it took KYC-only PR-AUC from 1.70× to 2.63×: the model memorised TRAIN laundering accounts that recur in VALIDATION. The same class of problem as D5's ID leakage.
  2. **v1 planting leaked future information.** It planted profiles on accounts because of *later* alerts. Among accounts with no TRAIN alert, an explanation-type sector made a VALIDATION alert 1.9× more likely, and a high declared band 3.5×.

**v2 result and the identity guard (decided by Dani, 2026-09-30).**
- KYC v2 (label-free) still failed C1 (2.63× prev) and C2 (1.16× prev); C3 and C4 passed.
- The same test on VALIDATION alerts of accounts **never alerted in TRAIN** gave C1 = 1.01× prev (no signal), and every check passed. So KYC contains no planted clue. What fails is **account recognition**: a stable combination of per-account fields lets a model recognise accounts that recur between TRAIN and VALIDATION.
- **Final run (v2.1 planting, 2026-09-30):**

  | Check | All VALIDATION alerts | Unseen accounts only (report) |
  |---|---|---|
  | C1 KYC-only | 2.04 × prev ✗ (ceiling 2.0) | 0.96 × prev ✓ |
  | C2 synthetic increment | 0.49 × prev ✗ (ceiling 0.25) | −0.17 × prev ✓ |
  | C3 txn + KYC − txn | −0.65 × prev ✓ | −0.38 × prev ✓ |
  | C4 synthetic interaction | −0.14 × prev ✓ | −0.05 × prev ✓ |

  Adding KYC to the transaction features **lowers** PR-AUC (0.206 → 0.172 on all VALIDATION).
  Identity guard on KYC: seen accounts (1,692 alerts, 28 positives) gain +0.86 × prev; unseen accounts (2,466 alerts, 185 positives) gain −0.38 × prev → **`account_recognition`**. **Consequence: KYC fields are not model features in P3–P5.** The seen subset has only 28 positives, so the seen gain is noisy, but the decision does not depend on it: KYC adds nothing (C3 < 0) on either subset.
- Any stable per-account attribute (real or synthetic) can do this, so it cannot be "fixed" inside KYC without making KYC useless. The ceilings stay as declared, and the failures are recorded in `configs/p2_kyc.yaml → accepted_ceiling_failures` with the exact measured values of the final run. A results test accepts a failed ceiling only if it is recorded there **and** the unseen-accounts rerun passes every check.
- The real risk is a **model** that looks good by recognising accounts. It is blocked by a binding rule (`identity_guard` in `configs/p2_kyc.yaml`, code in `src/eval/identity_guard.py`):
  - For P3–P5, every model feature group is measured on VALIDATION separately for accounts **seen** in TRAIN and **unseen** accounts: gain = PR-AUC(base + group) − PR-AUC(base) in each subset, divided by that subset's prevalence. Raw lift is not compared, because seen and unseen accounts are different populations (the transaction-only model has 4.0× lift on all VALIDATION vs 1.43× on unseen).
  - Verdicts, with thresholds declared before first use: fewer than 20 positives in either subset → `insufficient_data` (not cleared); seen gain < 0.05 × prev → `no_material_gain`; unseen gain < 0.5 × seen gain → `account_recognition` (the group is removed or fixed); otherwise `pass`.
  - Model results are always reported for regime B (unseen accounts) next to regime A.
  - The guard is applied here to KYC on top of the transaction features (`build.kyc.identity_guard`). If KYC gets `account_recognition`, P3 may not use KYC fields as model features. Agents may still read KYC as case context: they do not learn across cases.

## 9. Planted innocent explanations (task 7), v2: from burn-in behaviour, label-free

- For every account active on the burn-in day (2022-09-01), each rule statistic becomes a within-day percentile. The account's **dominant behaviour** is the highest one.
- Accounts whose dominant percentile is ≥ 0.95 are candidates. Each candidate receives a sector explaining that behaviour (v2.1: every candidate, probability 1.0), plus a high or very-high declared band. Examples: fan-in → payment processor; many near-10k payments → convenience store; cash → restaurant.
- If the behaviour recurs later, the profile plausibly explains the later alert, **for launderers and legitimate customers alike**. The Defence has something real to argue, and nothing uses a label or a future date.
- **The v1 firewall exception is removed.**
- **v2.1 (Dani, 2026-09-30, declared before the rerun).** v2 used the top 1% and probability 0.5. Realised planted rates among pre-TEST alerted accounts were only 7.4% (legitimate) and 3.4% (laundering): too little for the Defence to work with. v2.1 widens candidates to the top 5% and plants every candidate. The new rates are measured, not promised.
- **v2.1 measured** (27,673 accounts with a pre-TEST alert; 204,481 accounts planted overall):

  | | Laundering accounts (1,406) | Legitimate accounts (26,267) |
  |---|---|---|
  | Has an explaining profile | 12.7% | 23.9% |
  | Profile explains its own alert's rule | 6.1% | 16.1% |

  Legitimate accounts get a profile about twice as often as laundering accounts. Planting is label-free, so this comes from the generator's own behaviour: accounts with a very pronounced burn-in behaviour are more often legitimate. It is realistic (most innocent explanations belong to innocent customers) and it means a profile is **weak** evidence for innocence, not proof. 1 in 16 laundering accounts carries a profile that explains its own alert: exactly the "plausible but wrong" case the Defence must not win automatically. P6 must include such cases in Tier B.
- **Realised rates are measured on pre-TEST alerted accounts** and reported as `build.kyc.planting`:
  - planted rate for real vs false alerted accounts;
  - "explains its own alert" rate.
- The planting record lives only in the evaluation store.
- `run_p2 kyc` regenerates KYC and its tests **without reading TEST**; a test proves it gives identical output with all TEST rows deleted.

## 10. Simulated dispositions (task 9)

Only TRAIN, VALIDATION and CALIBRATION alerts get a disposition. Records hold exactly `{alert_id, account_key, disposition, closed_at}`.

Probability that the analyst confirms the alert:

| Case | P(confirm) |
|---|---|
| Laundering | 0.85 |
| Laundering, H1 "shapeless": only unattributed, RANDOM or BIPARTITE transactions | 0.70 |
| Legitimate | 0.02 |
| Legitimate, H2 near-threshold (R03 fired) | 0.05 |
| **Anchoring** (applies to all) | logit += 0.5 × (n_rules − 1) |

- `closed_at` = `as_of` + lognormal delay: median 18 h if closed, 48 h if confirmed; clipped to 1–168 h. Time is compressed because the span is only 16 days.
- The inputs are ground truth, observable alert features and seeded noise. The generator never uses model scores (H4 circularity).
- Realised error rates, overall and per group, are in `build.dispositions`.

## 11. Stores

**Agent side** (`data/p2/HI-Medium/runtime/`), allow-listed columns:
- `alerts.parquet`
- `kyc.parquet`
- `dispositions.parquet`
- `agent_dev_alert_ids.parquet`

**Evaluation side** (`…/eval/`):
- `alert_labels.parquet`
- `positive_account_days.parquet`
- `laundering_legs.parquet`
- `agent_split.parquet`
- `planting.parquet`

Regeneration with the same seed is byte-identical: `run_p2 all --verify`, and `test_regeneration_is_byte_identical` on the fixture.
