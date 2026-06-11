# General Ledger — World-Class Improvements

## I1. Accrual Reversal Scheduling
**Category**: Core Accounting  
**Justification**: Accrual entries (accrued interest, prepaid amortisation) must auto-reverse at the start of the next period. Without this, manual reversals are error-prone and frequently missed, distorting the next period's P&L. SAP S/4HANA, Oracle Financials, and Xero all implement scheduled accrual reversals as a first-class feature.  
**Implementation**: Add `reversal_date` and `auto_reverse` fields to journal entries. A `process_scheduled_reversals(period_id)` method scans for entries with `auto_reverse=True` and `reversal_date <= today`, posts mirror entries, and marks originals as `REVERSED`.  
**Competitor Reference**: SAP FI Accrual Engine; Oracle Period Close automation; QuickBooks Online recurring journal entries.

---

## I2. Cash Flow Statement Generation
**Category**: Financial Reporting  
**Justification**: The three mandatory financial statements are P&L, Balance Sheet, and Cash Flow. Most GL implementations stop at P&L + balance sheet. SASRA (Sacco Societies Regulatory Authority) requires statement of cash flows in annual returns. Missing this forces manual reconciliation.  
**Implementation**: Classify accounts into operating/investing/financing activities via a `cash_flow_class` field. `get_cash_flow_statement(from_date, to_date)` aggregates movements by class using the indirect method (net surplus + depreciation + working capital changes).  
**Competitor Reference**: Sage Intacct CF module; NetSuite Cash Flow report; QuickBooks Cash Flow Snapshot.

---

## I3. Multi-Currency Revaluation with Gain/Loss Posting
**Category**: FX Management  
**Justification**: The existing `revalue_foreign_accounts` method computes FX gain/loss but never posts it. This means the balance sheet is not restated and the P&L misses unrealised FX gains/losses — a IFRS 21 violation for any institution holding USD, GBP, or EUR balances.  
**Implementation**: Extend `revalue_foreign_accounts` to post balanced journal entries: debit/credit the foreign account for the revaluation delta, and post the offsetting entry to account `4400` (Other Income) or `5600` (Other Expenses) with a `FXREVAL` reference prefix.  
**Competitor Reference**: Oracle GL Revaluation; SAP FX Valuation Run; Sage 200 Revaluation Wizard.

---

## I4. Segment / Dimension Reporting
**Category**: Managerial Accounting  
**Justification**: Single-dimension COA cannot answer "what is the P&L by branch?" or "by product line?". SACCO regulators expect branch-level reporting. Adding cost-centre/segment tags to journal lines enables multi-dimensional slicing without doubling the COA.  
**Implementation**: Add optional `segment` and `cost_centre` fields to journal entry lines. `get_segment_pnl(segment, from_date, to_date)` filters lines by segment and aggregates income/expense. `get_segment_trial_balance(segment)` does the same for balance accounts.  
**Competitor Reference**: Sage Intacct Dimensions; Microsoft Dynamics 365 Business Central Dimensions; Xero Tracking Categories.

---

## I5. Intercompany Elimination Engine
**Category**: Consolidation  
**Justification**: Multi-entity SACCOs (FOSA + BOSA + investment subsidiaries) must eliminate intercompany balances for consolidated reporting. The current `settle_intercompany` is a stub returning `{"settled": True}` — it posts nothing.  
**Implementation**: Real implementation posts balanced entries using accounts `1400` (Interbank Receivable) and `2600` (Interbank Payable). `get_consolidation_report(entity_ids, as_of_date)` aggregates balances across tenants and subtracts elimination entries to produce a clean consolidated balance sheet.  
**Competitor Reference**: Oracle Hyperion HFM; SAP Group Reporting; Sage Intacct Multi-Entity Consolidation.

---

## I6. Straight-Line Depreciation Scheduler
**Category**: Asset Accounting  
**Justification**: Fixed assets on the COA (1300) require periodic depreciation entries debiting account 5500 and crediting 1310. Without automated depreciation schedules, institutions must manually post these every month — a source of omission errors that overstate asset values.  
**Implementation**: `register_asset(code, cost, useful_life_months, start_date)` records the asset schedule. `run_depreciation(period_id, posting_date)` computes monthly charge = cost / useful_life_months, posts the Dr 5500 / Cr 1310 entry, and marks the schedule period as processed.  
**Competitor Reference**: Sage Fixed Assets; Oracle Assets; Xero Fixed Asset Manager.

---

## I7. Budget vs Actual Variance Analysis
**Category**: Planning & Control  
**Justification**: No budget module means management cannot answer "are we on track?". SASRA also requires institutions to submit annual budgets. Without a budget store, the GL cannot produce variance reports that flag overspends.  
**Implementation**: `set_account_budget(code, period_id, budgeted_amount)` stores a budget figure. `get_budget_variance_report(period_id)` retrieves actual movements (from journal entries) vs budgeted amounts and computes variance % and over/under flags.  
**Competitor Reference**: Adaptive Insights (Workday); Sage Intacct Budgeting; QuickBooks Budgets vs Actuals.

---

## I8. Period Locking with Hard/Soft Lock Distinction
**Category**: Period Control  
**Justification**: The existing `close_period` permanently locks a period. In practice, finance teams need a "soft lock" (warn but allow with manager approval) and a "hard lock" (strictly deny all postings). Auditors frequently need to post audit adjustments to previously soft-locked periods without re-opening.  
**Implementation**: Add `lock_type: Literal["SOFT", "HARD"]` to period records. `close_period` accepts `lock_type` parameter. `post_journal_entry` raises `PostingToClosedPeriodError` on hard-locked periods and logs a `WARN` for soft-locked periods (allowing posting with an `override_soft_lock=True` flag).  
**Competitor Reference**: SAP Posting Period variants; Oracle Period Close statuses; Sage 200 Period Lock levels.

---

## I9. Aging Analysis for Receivables and Payables
**Category**: Credit Risk  
**Justification**: Loan portfolio health and creditor exposure both require aging buckets (Current, 30d, 60d, 90d, 90d+). This is mandatory for SASRA loan classification and provisioning calculations. Without it, the provision for loan losses (1130) has no automated basis.  
**Implementation**: `get_aging_report(account_code, as_of_date, buckets=[30,60,90])` scans sub-ledger lines for the given receivable/payable account, computes days-outstanding for each line using `as_of_date - posting_date`, and groups into the specified day buckets with totals.  
**Competitor Reference**: QuickBooks Accounts Receivable Aging; Sage 50 Aged Debtors; Oracle AR Aging Report.

---

## I10. Audit Log with Tamper-Evidence Chaining
**Category**: Compliance & Audit  
**Justification**: The existing `entry_hash` is computed per-entry but not chained. A sophisticated attacker could replace a journal entry and recompute its hash without breaking the chain. Blockchain-style hash chaining (each entry includes the previous entry hash) makes deletion or reordering cryptographically detectable.  
**Implementation**: On every `post_journal_entry`, set `prev_hash` = hash of the last posted entry. Include `prev_hash` in the `entry_data` used to compute `entry_hash`. `verify_audit_chain()` walks all entries in posting order and confirms each hash is consistent with its predecessor.  
**Competitor Reference**: Sovos audit log chaining; IBM Financial Controls; R3 Corda ledger immutability.

---

## I11. Recurring Journal Entry Templates
**Category**: Automation  
**Justification**: Institutions post the same entries every month: interest accruals, depreciation, management fees, regulatory levies. Manual re-entry is a source of errors and staff time waste. Template-based recurrences eliminate both.  
**Implementation**: `create_recurring_template(name, lines, frequency, next_run_date, period_template)` stores a template. `process_recurring_entries(as_of_date)` identifies templates due on or before `as_of_date`, posts the journal entry, and advances `next_run_date` by the frequency interval. Returns a summary of entries generated.  
**Competitor Reference**: Xero Repeating Journals; QuickBooks Memorised Transactions; Sage Business Cloud Recurring Journals.

---

## I12. Interbank Reconciliation Statement
**Category**: Reconciliation  
**Justification**: Cash accounts (1010, 1020) must be reconciled against external bank statements. The current `reconcile_period` only checks trial balance balance; it does not match individual transactions against a bank statement. Unexplained differences indicate fraud, errors, or timing differences.  
**Implementation**: `import_bank_statement(account_code, statement_lines)` stores bank statement entries. `reconcile_bank_account(account_code, statement_date)` matches GL entries to bank lines by amount + date, marks matched pairs, and returns a list of unmatched items on both sides (GL and bank).  
**Competitor Reference**: Xero Bank Reconciliation; QuickBooks Bank Feeds matching; Sage Bank Feeds auto-match.

---

## I13. Deferred Revenue / Prepaid Expense Amortisation
**Category**: IFRS Compliance  
**Justification**: IFRS 15 (Revenue Recognition) requires upfront fees (e.g., loan origination fees) to be deferred and recognised over the loan life. Without automated amortisation, institutions either immediately recognise all fee income (overstating early periods) or miss recognition entirely (understating later periods).  
**Implementation**: `register_deferred_item(account_code, total_amount, start_date, end_date, recognition_account)` stores amortisation schedules. `run_amortisation(period_id, posting_date)` computes the period's portion (straight-line), posts Dr deferred-liability / Cr income, and updates the remaining unamortised balance.  
**Competitor Reference**: NetSuite Revenue Management; Sage Intacct Revenue Recognition; Oracle Revenue Management Cloud.

---

## I14. Comparative Period Financial Statements
**Category**: Financial Reporting  
**Justification**: Regulators and boards require comparative financials (current vs prior period / prior year). Single-period reports provide no trend context. Without this, users must manually run two reports and align columns — a PDF-editing exercise that introduces transcription errors.  
**Implementation**: `get_comparative_pnl(current_from, current_to, prior_from, prior_to)` and `get_comparative_balance_sheet(current_date, prior_date)` fetch both periods' data in one call, compute variance amounts and variance percentages, and return a single merged data structure ready for rendering.  
**Competitor Reference**: QuickBooks Comparative P&L; Xero Comparative Balance Sheet; Sage 200 Period Comparison.

---

## I15. GL Integration Event Bus (NATS/Kafka Publish)
**Category**: Integration & Event Sourcing  
**Justification**: The design doc states "every state change generates a NATS event" but no publishing code exists. Downstream capabilities (loans, deposits, compliance, dashboards) need reliable event streams to maintain their own projections without polling the GL. Missing this creates tight coupling via direct calls and race conditions under load.  
**Implementation**: Inject an optional `event_bus` adapter into `GLService.__init__`. After every `post_journal_entry`, `close_period`, and `close_year`, call `await self._event_bus.publish(topic, payload)` with a structured CloudEvents envelope. Include `tenant_id`, `entry_id`, `reference`, and `total_debit`/`total_credit`. Gracefully no-op when `event_bus=None`.  
**Competitor Reference**: SAP Event Mesh GL postings; Oracle EBS Business Events; Confluent Kafka financial event streaming.
