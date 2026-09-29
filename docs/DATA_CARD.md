# DATA_CARD — IBM Transactions for Anti-Money Laundering (Medium variants)

Phase 1 output. Every number below was printed from the data by `scripts/audit/discover.py` and
`scripts/audit/run_audit.py` (results: `docs/audit/p1_audit_results.json`) and is re-checked by
`tests/test_data_card_facts.py`. Nothing here is from memory.

**Frozen decisions (confirmed 2026-09-29)** live in `configs/p1_data_decisions.yaml`, which later phases read.
`tests/test_p1_decisions.py` re-validates each one against the audit evidence:
D1 exclude rows from 2022-09-17 (§4) · D2 `account_key = bank|account` as written (§3) · D3 timestamps are UTC (§4) ·
D4 TEST starts no earlier than 2022-09-13 (§4) · D5 ID values, ID order and file order banned as signal (§8) ·
D6 amounts converted to USD with the data's own rates (§6).
Audit run: 2026-09-29, Dani's laptop (Windows 10, Python 3.11.4, polars 1.44.2).
Re-run with `--verify` on 2026-09-29: every number reproduced exactly.
**All items marked DECISION are FROZEN (confirmed by Dani, 2026-09-29).**

## 1. Provenance and licence

| Item | Value |
|---|---|
| Source | Kaggle "IBM Transactions for Anti Money Laundering (AML)", uploader ealtman2019 |
| Generator | IBM IT-AML simulator (Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models", NeurIPS 2023 Datasets & Benchmarks, arXiv:2306.16424) |
| Licence | **CDLA-Sharing-1.0** (Community Data License Agreement – Sharing, Version 1.0). Confirmed on the Kaggle page (licence text checked by Dani, 2026-09-29) and in IBM's `IBM/AML-Data` GitHub README |
| Nature | Fully synthetic. No real persons or institutions. Not real-world data |
| Variants used | **HI-Medium** (primary), **LI-Medium** (regime C: within-generator cross-variant generalisation, never "external validation"), HI-Small (fast tests only) |
| What the licence allows | **Use** of the data, including any analysis (§2.1). **Results** (models, metrics, aggregate statistics, the report) carry **no** obligations (§3.5) |
| If we ever publish data | Any published raw data or subset (e.g. the P13 demo subset on the public Streamlit app) must: be published under CDLA-Sharing-1.0 with the licence text or a link (§3.3); keep the attribution to IBM / the Kaggle source (§3.1c); mark files we changed or added to (e.g. synthesised KYC) with a prominent notice (§3.1b) |
| Our policy | Raw files are never committed to our repositories (size, and they contain ground truth) |

Input fingerprints are frozen in `configs/data_sources.yaml`; the audit refuses to run if any file differs.

| File | Rows (data) | SHA-256 (first 16) |
|---|---:|---|
| HI-Medium_Trans.csv | 31,898,238 | 3126afb8155e7c88 |
| HI-Medium_Patterns.txt | 2,756 attempts / 22,743 txns | d612cd7ed6f7761c |
| HI-Medium_accounts.csv | 2,087,786 | 0fb441d31df9ca08 |
| LI-Medium_Trans.csv | 31,251,483 | 0fc89584453c9747 |
| LI-Medium_Patterns.txt | 456 attempts / 3,909 txns | 6aa4c4041894eb27 |
| LI-Medium_accounts.csv | 2,040,823 | d553f80638c183dd |
| HI-Small_Trans.csv | 5,078,345 | b19d39f515523373 |

## 2. Schema (as printed, not assumed)

**Transactions** (`*_Trans.csv`): 11 columns, UTF-8, LF line endings, no BOM, no malformed rows, no nulls, no empty strings.
The raw header repeats `Account` (sender, receiver), so columns are renamed **by position**.

| Raw column | Our name | Type | Notes |
|---|---|---|---|
| Timestamp | timestamp | datetime | format `YYYY/MM/DD HH:MM`, **minute resolution, no timezone** |
| From Bank | from_bank | string | always has leading zeros (e.g. `020`); kept as string |
| Account | from_account | string | 9 hex characters |
| To Bank | to_bank | string | as above |
| Account | to_account | string | as above |
| Amount Received | amount_received | float | > 0 always; min 1e-06, max 8.16e12 (HI) |
| Receiving Currency | receiving_currency | string | 15 currencies |
| Amount Paid | amount_paid | float | > 0 always; median 1,471.54 (HI) |
| Payment Currency | payment_currency | string | 15 currencies |
| Payment Format | payment_format | string | Cheque, Credit Card, ACH, Cash, Reinvestment, Wire, Bitcoin |
| Is Laundering | is_laundering | int8 | 0/1 only — **ground truth, firewalled** |

**Accounts** (`*_accounts.csv`): `Bank Name, Bank ID, Account Number, Entity ID, Entity Name`.
**Patterns** (`*_Patterns.txt`): blocks `BEGIN LAUNDERING ATTEMPT - <TYPOLOGY>[: Max N hops | Max N-degree Fan-In/Out]`,
then transaction lines in the transactions format, then `END LAUNDERING ATTEMPT - <TYPOLOGY>`. Never nested, always
balanced. 8 typologies: BIPARTITE, CYCLE, FAN-IN, FAN-OUT, GATHER-SCATTER, RANDOM, SCATTER-GATHER, STACK.
**Ground truth, firewalled.**

Other schema facts (HI-Medium; LI-Medium alike):
- Same-currency transactions always have `amount_paid == amount_received` (0 exceptions).
- 485,144 cross-currency transactions (1.5%).
- 2,561,860 self-transfers (same bank and account); **every Reinvestment is a self-transfer**.
- 20 exact duplicate row pairs (LI: 14), none laundering; **no label conflicts** (no identical transactions with different labels). Kept as-is.

## 3. Account key (DECISION)

- An account number alone is **not unique**: 24 account numbers (LI: 34) exist at two different banks.
  → **`account_key = "<from/to bank string as in transactions>|<account>"`**, e.g. `020|800104D70`, matching the frozen contract format `bank|account`.
- Bank IDs in transactions all carry leading zeros; the accounts file has none. Stripping leading zeros merges **no** (bank, account) pairs (2,077,023 pairs before and after), and after stripping **100%** of transacting accounts are found in the accounts file (0 missing; 10,763 account rows never transact).
  → Join to the accounts file on `(strip_leading_zeros(bank), account)`.
- Open item for P2: 61 normalised bank IDs (LI: 67) occur with two different zero-paddings on *different* accounts. Whether they are one bank matters only for bank-level features. The key above never merges them.

## 4. Time (the constraint that shapes everything)

| | HI-Medium | LI-Medium | HI-Small |
|---|---|---|---|
| First / last timestamp | 2022-09-01 00:00 / 2022-09-28 15:58 | 2022-09-01 00:00 / 2022-09-27 14:58 | 2022-09-01 / 2022-09-18 |
| **Complete days** (volume ≥ 50% of median day; labels not used) | **2022-09-01 … 2022-09-16 (16 days)** | same (16 days) | 09-01 … 09-10 (10) |
| Tail after the complete span | 12 days, 6,987 txns, **58.5% laundering** | 11 days, 1,370 txns, 56.3% laundering | 8 days, 1,108 txns, 59.1% |
| Provisional holdout start (P1 rule: floor(0.75 × complete days)) | **2022-09-13 00:00** (12 pre-holdout days, 4 holdout days) | same | 2022-09-08 |

Findings:
1. **Laundering continues after normal activity stops.** After 2022-09-16 only pattern transactions remain (58% laundering). Using these days would make "late in the data" a near-perfect laundering signal.
   → **DECISION: rows with timestamp ≥ 2022-09-17 00:00 are excluded from every split.** Cost: 4,084 HI laundering txns (11.6% of all), 771 LI (4.8%).
2. **Only 16 usable days.** Laundering attempts last a median 110.5 h (4.6 days), p90 144 h, max 305 h (12.7 days) in HI-Medium; GATHER-SCATTER median 227.36 h. Real-bank lookbacks ("12 months of history") are impossible. All windows must be in **hours or a few days**. With 12 pre-holdout days holding TRAIN, VALIDATION, CALIBRATION and three embargoes ≥ L_max, **L_max above ~24–48 h leaves too little data**. P2 must set L_max with this arithmetic written down.
3. **Truncation at both ends.** 738 of 2,756 HI attempts (27%) extend past 2022-09-16; 691 cross the provisional holdout start. Day 1 has no history, so the first L_max hours are a burn-in with incomplete features.
4. **Minute-resolution timestamps; ties are the norm.** Only 26,086 distinct minutes; 99.997% of rows share their minute with another row (max 44,215 in one minute). The frozen tie rule (visible iff `event_ts < as_of`) is load-bearing. Timestamps carry no timezone → **interpreted as UTC** (contracts require tz-aware datetimes).
5. **The file is not time-sorted** (15.2 million inversions in file order).
6. **Volume artefacts:** day 1 (Thursday) has 4.47M txns (warm-up burst, 1.38M in its first hour); every day has a midnight spike; weekends have ~43% of weekday volume while laundering counts do not drop, so the **weekend laundering rate is 2.3× the weekday rate** (pre-holdout). Laundering per day also rises over the span (1,056 on day 1 → 2,416 on day 16, HI) → **prevalence drifts upward over time**, which matters for calibration.

## 5. Labels and patterns

| | HI-Medium | LI-Medium |
|---|---:|---:|
| Laundering txns / all | 35,230 / 31,898,238 (0.110%) | 16,041 / 31,251,483 (0.051%) |
| Pre-holdout (09-01 … 09-12) | 21,939 / 23,079,383 (0.095%) | 11,035 / 22,620,673 (0.049%) |
| Holdout (09-13 … 09-16) — aggregate only | 9,207 / 8,811,868 (0.104%) | 4,235 / 8,629,440 (0.049%) |
| Pattern txns found in transactions (all labelled 1) | 22,743 of 22,743 | 3,909 of 3,909 |
| **Laundering txns in no pattern** | **12,487 (35.4%)** | **12,132 (75.6%)** |
| Accounts in ≥1 laundering txn | 41,857 of 2,077,023 (2.0%) | 23,510 of 2,032,095 (1.2%) |
| Pattern accounts in several attempts / typologies | 1,041 (max 57 attempts) / 944 | 181 (max 6) / 142 |

- Labels are **transaction-level**. The patterns file is a subset: the IT-AML paper (§3.4) states some laundering happens "naturally" outside the 8 patterns. → **Alert labels derive from `is_laundering`; typology ground truth exists only for pattern-attributed transactions** (P6 must restrict typology metrics accordingly).
- No transaction belongs to two attempts, but accounts are reused across attempts and typologies. → P2's AGENT-DEV/AGENT-TEST group split must use connected components of accounts and attempts.
- The holdout has enough positives to be evaluable (the one permitted exposure: aggregate counts per period).

Typologies (HI-Medium, attempts / txns / median duration h): BIPARTITE 369 / 2,135 / 42.55 · CYCLE 367 / 2,235 / 121.82 · FAN-IN 355 / 2,315 / 119.17 · FAN-OUT 345 / 2,128 / 115.55 · GATHER-SCATTER 322 / 4,289 / 227.36 · RANDOM 331 / 1,667 / 93.50 · SCATTER-GATHER 331 / 3,988 / 131.42 · STACK 336 / 3,986 / 102.56.

## 6. Currency (DECISION)

- 15 currencies, including Yen, Rupee and Bitcoin → raw amounts are not comparable across currencies. Conversion **is needed** for amount features.
- The generator uses **fixed exchange rates**: from pre-holdout cross-currency pairs, every fiat currency's USD rate agrees with its inverse to ≤ 6e-6 relative difference; within-pair interquartile spread ≤ 0.26% (consistent with cent rounding). Bitcoin ≈ 11,871–11,881 USD (rounding noise from tiny BTC amounts).
  → **DECISION: convert to USD with a deterministic `usd_per_unit` table derived from the dataset's own cross-currency transactions.** No real-world FX. P3 re-derives it on TRAIN only (identical by construction, but that keeps the data-access matrix clean).
- In pre-holdout data, **no cross-currency transaction is laundering** (0 of 339,144).

## 7. Trivial-separability test (task 6)

Pre-holdout data only. Temporal split inside it: fit on 09-01 … 09-08, evaluate on 09-09 … 09-12. Features: raw amount, USD amount (log), payment currency, payment format, hour, self-transfer, cross-currency.

| | HI-Medium | LI-Medium |
|---|---|---|
| Eval prevalence | 0.136% | 0.063% |
| Depth-1 tree PR-AUC | 0.0082 (6.1× prevalence): split = ACH | 0.0030 (4.7×): split = ACH |
| Depth-3 tree PR-AUC | **0.0211 (15.5×)** | **0.0061 (9.6×)** |
| Purest leaf | ACH with USD amount 10^3.18–10^4.30 (≈ $1.5k–$20k): 1.57% laundering | 0.57% |

**Result: no feature nearly separates the classes. There is a real detection problem. No feature is excluded.**
Generator signatures to document (they are real in this data but may not transfer):
- **ACH carries 86% (LI: 72%) of pre-holdout laundering** (lift 7.2 / 6.1).
- Wire, Reinvestment, cross-currency and (almost) self-transfers carry **zero** laundering: "never-laundering" rules exist in this generator.
- Hour 0 has a laundering lift of 0.27 (midnight batch of normal transactions).

Recommendation for P4 (not decided here): report feature importance, and a sensitivity run without `payment_format`.

## 8. Source-level leakage hunt (task 7)

- **No column is a function of the label or of future information.** `Is Laundering` is the only label column; no label conflicts exist.
- **Identifier values leak the label (FOUND).** Pre-holdout, account level:
  - laundering accounts cluster in ID order: P(next account in ID order is laundering | this one is) = 9.5% vs 1.35% base (**lift 7.0**; LI **14.9**);
  - low IDs are over-represented (prefix `80`: lift 2.0); PR-AUC of the negated account ID as a score = 0.038 (2.8× base);
  - 7 accounts whose ID ends in `8` are **all** laundering, in every variant.
- **File order leaks the label.** The pre-holdout laundering rate per decile of file position ranges from 0.035% to 0.197% (5.7×).
- → **DECISION: account, entity and bank ID values, their ordering and the file row position are never used as features, never used to order or tie-break data, and never used to split or sample. `row_id` is a technical key only. Group splits use a keyed hash of `account_key`.**
- Bank identity: the top banks with ≥1,000 accounts have lift ≈ 6. A raw bank-ID feature would mostly memorise the generator; P3 should justify any bank-level feature.

## 9. KYC available (replaces the "no KYC" assumption)

For **every** transacting account: bank name (foreign banks carry a country, e.g. "Spain Bank #16393"; US banks carry a city), **Entity ID** (668,138 entities; 210,275 own several accounts, up to 8,638, almost all across several banks) and **entity type** from the entity name: Corporation 679,329 · Partnership 732,828 · Sole Proprietorship 667,814 · Country 4,738 · Individual 3,050 · Direct 27 (HI-Medium). Label signal is weak (lift 0.6–1.5 with support).
Not available: customer age, occupation, income, risk rating, onboarding date, addresses → synthesised in P2 if needed, and documented as synthetic.

## 10. Scope and size (task 8, DECISION)

- Full Medium fits: transactions Parquet 742 MB; in memory ≈ 1.8 GB (categorical) to 2.7 GB (strings). The full audit ran in ~6 min per Medium variant on the team laptop (12 GB RAM, 2 cores).
- → **No subsample for offline work.** All phases read the full Medium Parquet lazily with column and row filters.
- Any later subset (the Streamlit demo, P13) is cut **by time window, keeping whole accounts and whole laundering attempts**. Never random rows.

## 11. Limitations

- Synthetic data from one generator: all results are "within-generator"; LI-Medium is a cross-variant check, not external validation.
- 16 usable days: no seasonality, no long histories, lookbacks in hours or days.
- Minimal KYC; no free text (no payment descriptions or names). Prompt-injection tests on these fields must use injected test data.
- Generator artefacts listed in §4, §7 and §8 are real in this data but would not exist in a bank's data. Findings that rely on them must be flagged.
- Labels are clean generator truth. Real alert labels are noisy; P2's simulated dispositions add a documented noise model.
