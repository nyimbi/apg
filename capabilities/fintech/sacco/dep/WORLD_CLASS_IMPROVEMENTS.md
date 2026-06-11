# SACCO Deposits & Savings — World-Class Improvements

## Overview

Fifteen high-impact improvements derived from analysis of leading SACCO platforms (Mambu, Temenos Infinity, Musoni, Mifos X, KCB SACCO, Faulu, Stima SACCO) and SASRA regulatory requirements.

---

### I1. Tiered Interest Rate Engine
**Category:** Core Business Logic
**Justification:** Flat-rate interest leaves yield on the table and reduces member loyalty. All tier-1 SACCOs (Stima, Mwalimu) offer bracket-based rates that reward higher balances — this is a primary driver of deposit mobilisation.
**Implementation:** Add `interest_rate_tiers` field to products (list of `{min_balance, max_balance, rate_pa}`). During accrual, split account balance into brackets and apply the correct marginal rate to each portion before summing. Fallback to flat rate when no tiers defined.
**Competitor Reference:** Mambu `interestRateTiers`, Temenos T24 `INTEREST.RATE.TABLE`

---

### I2. Goal-Based Savings Targets
**Category:** Member Engagement / UX
**Justification:** Behavioural research (Stango & Zinman 2014, M-Shwari data) shows named goal accounts increase deposit frequency by 30–60%. Faulu Kenya and M-Shwari both offer goal accounts. SACCOs without this lose members to mobile wallets.
**Implementation:** `set_savings_goal(account_id, goal_name, target_amount, target_date)` persists goal metadata on the account. `get_savings_goal_progress` returns projected vs actual trajectory, months remaining, required monthly contribution to stay on track.
**Competitor Reference:** Kuda Bank goal pots, M-Shwari lock savings, Faulu "Save for a Purpose"

---

### I3. Standing Order / Recurring Deposit Scheduling
**Category:** Automation
**Justification:** Manual deposit collection is the #1 operational cost for SACCOs. Automated recurring instructions reduce defaults and teller load. Equity Bank SACCO and ICEA SACCO report 40% lower cost-per-deposit after standing order automation.
**Implementation:** `create_standing_order(account_id, amount, frequency, start_date, end_date, payment_method)` stores a schedule. `process_due_standing_orders(run_date)` scans all orders due on or before `run_date`, calls `deposit()` for each, and returns a batch result with success/failure counts.
**Competitor Reference:** Mifos `RecurringDepositAccount`, Musoni standing orders, Temenos `PERIODIC.PAYMENT`

---

### I4. Fixed Deposit Maturity Processing
**Category:** Product Lifecycle
**Justification:** Fixed deposits locked beyond maturity sit idle and accrue incorrect interest. SASRA compliance requires formal maturity notification and rollover/payout decisions within defined windows (typically 7 days).
**Implementation:** `process_matured_fixed_deposits(run_date, action)` where action is `rollover | payout | manual`. On payout: credits principal+interest to linked regular savings account. On rollover: resets maturity date by original term, updates opening balance to include capitalised interest.
**Competitor Reference:** Mambu `TermDepositAccount.maturityDate`, KCB SACCO fixed deposit renewal, Temenos `FIXED.DEPOSIT`

---

### I5. Dividend / Interest Capitalisation
**Category:** Financial Accuracy
**Justification:** Many SACCOs post interest to income rather than the member account. SASRA mandates that member savings interest be capitalised (credited to balance) and distinct from loan interest income. Incorrect posting distorts member equity and triggers SASRA queries.
**Implementation:** `capitalise_interest(account_id, period, approved_by)` moves `accrued_interest` from the staging field to `balance` via a formal `sacco_interest_capitalisation` transaction type. Generates a batch capitalisation run report for the AGM.
**Competitor Reference:** Temenos `CAPITALISE.INTEREST`, Mifos `SavingsAccount.postInterest()`

---

### I6. Withdrawal Notice Period Enforcement
**Category:** Compliance / Risk
**Justification:** Products with `withdrawal_notice_days > 0` currently store the field but never enforce it. A fixed deposit or notice account where members can withdraw immediately is a liquidity management failure and a regulatory breach (SASRA Prudential Guideline 4.2).
**Implementation:** `request_withdrawal_notice(account_id, amount, requester)` creates a `withdrawal_notice` record with `release_date = today + notice_days`. `process_released_notices(run_date)` finds notices whose `release_date <= run_date` and calls `withdraw()` automatically.
**Competitor Reference:** Musoni "Notice Period Savings", UK building society notice accounts

---

### I7. Dormancy Scoring & Automated Classification
**Category:** Risk Management
**Justification:** SASRA defines dormancy as 12 months with no member-initiated transactions (not system interest posts). Manual dormancy review is error-prone. Automating this with a score prevents regulatory findings and enables targeted reactivation campaigns.
**Implementation:** `run_dormancy_scan(as_of_date, inactivity_threshold_days, tenant_id)` iterates active accounts, computes `last_member_transaction_date`, marks dormant if threshold exceeded, emits `account_auto_dormant` event. Returns a report with counts by product and total dormant balance.
**Competitor Reference:** Mifos `SavingsAccountDormancyTracker`, Mambu dormancy handling

---

### I8. Deposit Limit & Velocity Controls
**Category:** AML / Compliance
**Justification:** FATF Recommendation 10 and CBK AML guidelines require SACCOs to flag unusual deposit patterns. Large single deposits (>KES 1M) and high-velocity sequences are primary red flags. Implementing controls in the deposit layer is cheaper than a separate AML system for tier-2/3 SACCOs.
**Implementation:** `set_deposit_controls(product_id, single_txn_limit, daily_limit, monthly_limit)` stores limits on the product. `deposit()` checks these limits before posting, raises `DepositLimitExceeded` with `control_type` and `limit_value` in the error. Flag transactions for manual review rather than hard-reject when configured.
**Competitor Reference:** Temenos `TRANSACTION.LIMIT`, CBK AML Guideline 2023, Mambu transaction limits

---

### I9. Inter-Account Transfer
**Category:** Operations
**Justification:** Members frequently need to move funds between their own accounts (e.g., regular savings → holiday savings) or to another member's account (group contributions). Without a transfer primitive, this becomes two separate transactions with no atomicity guarantee, creating reconciliation issues.
**Implementation:** `transfer(from_account_id, to_account_id, amount, narration, transferred_by)` atomically debits source and credits destination within the same tenant. Validates both accounts are active, checks minimum balance on source, creates two linked transactions with a shared `transfer_reference`. Cross-member transfers require additional `requires_approval` flag on the product.
**Competitor Reference:** Core banking transfer primitives; Mifos `SavingsAccountTransactionType.TRANSFER`

---

### I10. Projected Balance & Interest Calculator
**Category:** Member Self-Service
**Justification:** Member retention is strongly correlated with financial transparency (EY SACCO Study 2023). Providing a "what-if" calculator reduces member support calls, increases trust, and satisfies the Consumer Protection Act disclosure requirement that members understand the cost/benefit of their savings products.
**Implementation:** `project_balance(account_id, months_ahead, monthly_deposit, tenant_id)` returns month-by-month projected balance using the product's rate and compounding frequency, assuming `monthly_deposit` added at the start of each month. Returns `{"month": ..., "opening_balance": ..., "deposit": ..., "interest": ..., "closing_balance": ...}` per row.
**Competitor Reference:** Kuda Spend Forecast, Mambu account projections, Equity Bank savings calculator

---

### I11. Regulatory Reporting — SASRA SF01 Export
**Category:** Compliance
**Justification:** Every licensed SACCO must submit the SF01 (Deposits Schedule) to SASRA monthly. Generating this manually from raw transaction data is a source of errors and a recurring audit finding. Automating the SF01 extract eliminates this risk.
**Implementation:** `generate_sasra_sf01(period_year, period_month, tenant_id)` aggregates savings balances by product category, computes average balance, interest earned, and member counts in the prescribed SF01 format. Returns a structured dict mirroring the form columns, ready for PDF/XLSX rendering.
**Competitor Reference:** SASRA Supervisory Framework 2020, Mifos custom reports

---

### I12. Multi-Currency Savings Support
**Category:** Extensibility
**Justification:** Kenyan SACCOs in the diaspora segment (Kenya National Police Service SACCO diaspora, ICEA) accept USD and GBP deposits. Hardcoding KES means these SACCOs cannot use the platform. Adding currency-aware balances future-proofs the engine.
**Implementation:** `convert_currency(account_id, to_currency, exchange_rate, converted_by)` creates a currency conversion transaction, records `exchange_rate` and `from_currency`. `portfolio_summary` aggregates in the tenant's base currency using stored exchange rates. Each account carries `currency` and `base_currency_balance`.
**Competitor Reference:** Temenos multi-currency savings, Mambu FX module

---

### I13. Savings Group / Chama Account Support
**Category:** Product Expansion
**Justification:** ~60% of Kenyan SACCO members also participate in informal chama (savings group) structures. SACCOs that offer group accounts retain members who would otherwise fragment savings across multiple providers. Group accounts require shared ownership, contribution tracking per member, and quorum-based withdrawals.
**Implementation:** `create_group_account(group_id, member_ids, product_id, contribution_schedule)` opens a group savings account. `record_group_contribution(account_id, contributing_member_id, amount, ...)` deposits and logs individual contributions. `get_group_contribution_summary(account_id)` returns per-member totals and percentage of group balance.
**Competitor Reference:** Musoni group savings, Mifos `GroupSavingsAccount`, Cooperative Bank chama accounts

---

### I14. Penalty & Charge Engine
**Category:** Revenue / Compliance
**Justification:** SACCOs lose revenue and create inconsistent member treatment by applying charges manually. Minimum balance penalties, early withdrawal charges (fixed deposits), and account maintenance fees need to be applied systematically and recorded as proper charge transactions for financial statement accuracy.
**Implementation:** `apply_charge(account_id, charge_type, amount, narration, applied_by)` posts a `sacco_charge` debit transaction. `run_maintenance_charges(charge_date, product_id, amount)` batch-applies a fee to all active accounts on the product. `apply_early_withdrawal_penalty(account_id, break_date)` computes penalty as `principal × penalty_rate × remaining_days / 365`.
**Competitor Reference:** Mambu `ProductFee`, Mifos `SavingsProductCharge`, Temenos `CHARGES`

---

### I15. Real-Time Balance Notification Hooks
**Category:** Member Experience / Integration
**Justification:** CBK Consumer Protection Guidelines require SACCOs to notify members of every debit/credit within a "reasonable time" (interpreted as 24h by SASRA). SMS/push notifications sent immediately after deposit/withdrawal increase member trust and reduce fraud response time.
**Implementation:** `register_notification_hook(event_type, handler_url, secret, tenant_id)` stores a webhook registration. After each `deposit()`, `withdraw()`, `accrue_interest()` etc., `_fire_hooks(event_type, payload)` iterates matching hooks and posts a signed JSON payload. Includes `X-Signature-SHA256` header using HMAC-SHA256 of payload with the registered secret.
**Competitor Reference:** Stripe webhook pattern, Mambu webhooks, Flutterwave event hooks
