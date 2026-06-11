# SACCO GL — World-Class Improvement Roadmap

15 evidence-based improvements benchmarked against Temenos T24, Mambu, Finflux, and Mifos X.

---

### I1. Journal Entry Reversal Engine
**Category:** Core Accounting Integrity
**Justification:** SASRA examination findings frequently cite absence of audit-trail-safe reversals. Reversals must create a new, linked counter-entry — not mutate the original. This is the foundational correction mechanism for all SACCO operations.
**Implementation:** `reverse_journal_entry(tenant_id, entry_id, reversal_date, reversed_by, reason)` — copies all lines with debit/credit swapped, sets `is_reversed=True` and `reversal_of` on both entries, updates running balances, blocks re-reversal of an already-reversed entry.
**Competitor Reference:** Temenos T24 AC module — every transaction creates a new amendment entry; original records are immutable.

---

### I2. Automated Accrual Engine
**Category:** Revenue Recognition / IFRS 9
**Justification:** SACCOs must accrue interest monthly per ICPAK. Manual end-of-month accruals are error-prone and a leading cause of restatements. Automated accrual posting cuts month-end close from days to minutes.
**Implementation:** `run_interest_accrual(tenant_id, period, accrual_rate_pct, posted_by)` — iterates outstanding loan balances, calculates daily accrual per loan type, posts DR Accrued Interest Receivable / CR Interest Income, returns itemised accrual schedule with provision impact.
**Competitor Reference:** Mambu Loan Product Engine accrues interest on a configurable schedule, posting to the GL automatically at EOD.

---

### I3. Depreciation Posting Engine
**Category:** Asset Management / IAS 16
**Justification:** Fixed asset depreciation is mandatory under ICPAK-aligned standards. Without automated depreciation, fixed assets overstate the balance sheet and violate IAS 16, risking SASRA regulatory censure.
**Implementation:** `post_depreciation(tenant_id, period, asset_schedule, posted_by)` — accepts list of `{asset_id, cost, accumulated, useful_life_months}`, computes straight-line monthly charge, posts DR Depreciation (5500) / CR Accumulated Depreciation (1305), returns per-asset schedule.
**Competitor Reference:** Finflux Fixed Assets module auto-generates depreciation journals at period-end using asset register data.

---

### I4. Multi-Currency GL Support
**Category:** Treasury / Regulatory
**Justification:** SACCOs with diaspora members transact in USD/GBP/EUR. SASRA requires FX positions to be disclosed. Single-currency GL cannot capture translation gains/losses, causing material balance sheet misstatements.
**Implementation:** `post_fx_revaluation(tenant_id, period, fx_rates, posted_by)` — revalues all foreign-currency accounts at closing rates, posts DR/CR FX Revaluation reserve, logs translation gain/loss per currency pair. Monetary amounts remain in KES with FCY memo fields.
**Competitor Reference:** Temenos T24 currency modules maintain parallel FCY ledger, with automatic revaluation at period-end.

---

### I5. SASRA Prudential Ratios Calculator
**Category:** Regulatory Compliance
**Justification:** SASRA's WOCCU-aligned PEARLS framework requires 12 mandatory ratios reported quarterly. SACCOs that cannot compute these in real time miss deadlines and incur fines of up to KES 500,000.
**Implementation:** `compute_sasra_pearls(tenant_id, period)` — computes P1 (institutional capital), E1 (delinquency rate), A1 (productive assets), R8 (net income), L1 (liquid assets), and 7 additional ratios from GL balances; returns structured dict with pass/fail against SASRA thresholds.
**Competitor Reference:** Mifos X SACCO plugin ships PEARLS dashboard consuming the core GL via REST.

---

### I6. Bulk Transaction Import with Rollback
**Category:** Operations / Data Integrity
**Justification:** SACCO migrations and batch salary deductions require posting hundreds of entries atomically. Partial failures leave the GL in an unbalanced state. Competitors implement unit-of-work patterns to guarantee all-or-nothing semantics.
**Implementation:** `post_bulk_transactions(tenant_id, transactions, posted_by, dry_run=False)` — validates all entries in a first pass (balance check, period-open check, account existence), posts all in a second pass only if validation passes; `dry_run=True` returns validation results without posting.
**Competitor Reference:** Mambu Batch API supports transactional bulk import with per-item error reporting and full rollback on any failure.

---

### I7. Audit Trail with Tamper-Evident Hashing
**Category:** Audit & Compliance
**Justification:** SASRA examinations require production of full audit trails. Mutable in-memory logs do not satisfy the evidence standard. Chained SHA-256 hashes on journal entries make post-hoc tampering detectable.
**Implementation:** `get_audit_trail(tenant_id, from_date, to_date, account_code)` — returns ordered journal entries enriched with `entry_hash` (SHA-256 of entry fields) and `chain_hash` (hash chained from predecessor); a `verify_chain` flag recomputes hashes and flags broken links.
**Competitor Reference:** T24 uses cryptographic journaling to satisfy Bank of Kenya AML record-keeping requirements.

---

### I8. Loan Portfolio Ageing Analysis
**Category:** Credit Risk / SASRA Reporting
**Justification:** SASRA requires monthly PAR (Portfolio at Risk) reporting by ageing bucket (current, 1-30, 31-90, 91-180, 181-365, >365 days). Without GL-level ageing, provision calculations are inaccurate and capital is under-reserved.
**Implementation:** `get_loan_portfolio_ageing(tenant_id, as_of_date, loan_balances)` — accepts `{loan_id, outstanding_principal, last_payment_date}` list, classifies each into PAR buckets, applies SASRA provisioning rates (1%, 5%, 25%, 50%, 100%), posts incremental provision adjustment, returns ageing schedule and required provision.
**Competitor Reference:** Finflux PAR engine computes bucket-level provisioning and posts adjustment entries automatically.

---

### I9. Intra-Period Reporting Snapshots
**Category:** Management Reporting
**Justification:** CFOs and SASRA examiners require point-in-time balance snapshots without closing the period. Current implementation re-scans all journals on every call — O(n) per query. Snapshots give O(1) lookups for common dates.
**Implementation:** `create_balance_snapshot(tenant_id, snapshot_date, label)` — captures current account balances into an immutable snapshot record keyed by date + label; `get_snapshot(tenant_id, label)` returns stored balances without scanning journals.
**Competitor Reference:** Mambu end-of-day snapshot service captures ledger state daily for instant historical balance lookups.

---

### I10. Automated Closing Entries
**Category:** Period Close / IFRS
**Justification:** IFRS-aligned accounting requires income/expense accounts to be zeroed to retained surplus at year-end. Without automated closing entries, the income statement accumulates across years, producing materially misstated financials.
**Implementation:** `post_closing_entries(tenant_id, year, closed_by)` — aggregates all income account balances (credit normal) and expense account balances (debit normal), posts a single closing journal that zeros each income/expense account to Retained Surplus (3300), validates that Retained Surplus movement equals the net surplus, locks the year.
**Competitor Reference:** Temenos T24 period-end suite includes automated journal generation for nominal account closure with GL lock-out.

---

### I11. Inter-Branch / Multi-Entity Elimination
**Category:** Consolidation
**Justification:** SACCO federations (e.g. apex cooperative societies) consolidate subsidiaries. Intra-group transactions must be eliminated. Without a due-to/due-from mechanism and elimination step, consolidated statements are overstated.
**Implementation:** `post_interentity_transfer(tenant_id, counterpart_tenant_id, amount, narrative, posted_by)` — posts DR Due-From-Entity (2500) in counterpart and CR Due-To-Entity (2400) in source simultaneously using a shared transaction ID; `eliminate_interentity(parent_tenant_id, child_ids, period)` nets and eliminates reciprocal balances.
**Competitor Reference:** Mifos X multi-office GL segregates branch books with head-office elimination at consolidation run.

---

### I12. Regulatory Return Generator (SASRA Form-6 / CBK)
**Category:** Regulatory Reporting
**Justification:** SASRA requires quarterly submission of Form 6 (Financial Condition Report). Manual extraction from the GL takes 3-5 days and is error-prone. Automated generation cuts this to minutes and removes transcription risk.
**Implementation:** `generate_sasra_form6(tenant_id, period)` — maps GL account balances to SASRA Form 6 line items using a configurable mapping table, validates balance sheet identity (Assets = Liabilities + Equity), produces a structured dict matching the SASRA XML/XLS submission format with all mandatory fields.
**Competitor Reference:** Finflux Regulatory Reporting module maintains a regulator-to-COA mapping table updated with each regulatory change.

---

### I13. Real-Time Liquidity Monitoring
**Category:** Treasury Risk
**Justification:** SASRA Prudential Guideline No. 4 requires SACCOs to maintain minimum liquid assets of 15% of deposits. Breaches attract daily penalties. Real-time liquidity dashboards prevent unplanned breaches caused by large withdrawals.
**Implementation:** `get_liquidity_position(tenant_id)` — computes liquid assets (Cash + Bank + near-liquid investments maturing ≤30 days), deposits base, liquidity ratio, distance to SASRA 15% minimum, and a projected breach date based on 30-day withdrawal run-rate; posts alert if ratio falls below 17% (2% buffer).
**Competitor Reference:** Temenos Treasury module provides intraday liquidity heat-maps with SASRA minimum thresholds hard-coded per jurisdiction.

---

### I14. Automated Withholding Tax on Interest
**Category:** Tax Compliance / KRA
**Justification:** KRA requires 15% withholding tax on interest paid to members. SACCOs that fail to deduct and remit WHT face penalties equal to the tax owed plus 20% surcharge. Automating the deduction at point of interest posting eliminates compliance gaps.
**Implementation:** `post_interest_earned_with_wht(tenant_id, account_id, gross_amount, period, wht_rate_pct, account_type, value_date, posted_by)` — deducts WHT from gross interest, posts DR Interest Expense (5100) / CR WHT Payable (2400) + CR Net Deposits (2100/2110); tracks cumulative WHT payable for P9 filing.
**Competitor Reference:** Mambu tax module applies jurisdiction-specific withholding rules at product level, posting tax entries automatically.

---

### I15. Configurable Approval Workflow for Large Transactions
**Category:** Internal Controls / Fraud Prevention
**Justification:** SACCOs lose millions annually to fraudulent or erroneous large-value postings. Industry best practice requires dual authorisation for transactions above a configurable threshold. Without this control, a single operator can post arbitrary amounts.
**Implementation:** `submit_for_approval(tenant_id, transaction_payload, submitted_by, threshold)` — if transaction amount exceeds threshold, creates a pending-approval record instead of posting; `approve_transaction(tenant_id, approval_id, approved_by)` validates approver != submitter rule, then posts to GL; `reject_transaction(tenant_id, approval_id, rejected_by, reason)` cancels the pending entry with audit record.
**Competitor Reference:** Finflux Maker-Checker module enforces four-eyes principle on configurable transaction types and amount thresholds.
