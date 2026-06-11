# FOSA World-Class Improvements

Fifteen high-impact improvements drawn from production SACCO/fintech practice and leading competitor benchmarks.

---

### I1. Real-Time Interest Accrual Engine
**Category**: Core Banking  
**Justification**: FOSA current accounts at Kenya Co-op Bank and Equity SACCO pay daily interest on minimum monthly balance. Current implementation only exposes `get_interest_earned` with no accrual engine, leaving revenue un-posted and member statements inaccurate.  
**Implementation**: `accrue_daily_interest(tenant_id, period_date)` — iterates active accounts, computes daily interest at the account's configured rate using minimum-balance-in-period, posts GL DR Interest Expense / CR Member Account, stores `interest_records` entry. Nightly batch job calls this; idempotent on `(account_id, period_date)`.  
**Competitor Reference**: Kenya Co-op Bank, Equity SACCO, NCBA Loop — all accrue daily interest on FOSA balances.

---

### I2. Cheque Lifecycle Management
**Category**: Payments  
**Justification**: Kenyan SACCOs (e.g., Stima, Mwalimu National) issue cheque books and process inward/outward cheques through the National Payment System. No cheque module exists currently, forcing manual GL adjustments.  
**Implementation**: `issue_cheque_book(tenant_id, account_id, leaves, series_start)`, `present_cheque(tenant_id, cheque_number, amount, payee)`, `return_cheque(tenant_id, cheque_number, reason)` — manages cheque register, posts GL entries for clearing float and final settlement, tracks bounced cheques and notifies member.  
**Competitor Reference**: Mwalimu National SACCO, Stima SACCO, I&M Bank — full cheque clearing integration.

---

### I3. RTGS / EFT Batch Payment Processing
**Category**: Payments  
**Justification**: High-value salary and supplier payments require RTGS submission. Current `BANK_TRANSFER` channel is a stub with no batch envelope, SWIFT reference, or settlement confirmation — not acceptable for CBK-regulated FOSA.  
**Implementation**: `create_eft_batch(tenant_id, debit_account_id, payments)`, `confirm_eft_settlement(tenant_id, batch_id, settlement_reference)` — assembles ISO 20022 pain.001-compatible batch, holds funds in suspense, releases on settlement confirmation. Includes batch reversal.  
**Competitor Reference**: KCB, Equity Bank EFT/RTGS via CBK KEPSS.

---

### I4. Loan Repayment Auto-Deduction
**Category**: Collections  
**Justification**: The primary FOSA use case for most members is having salary/deposits auto-deducted for BOSA loan repayments. Without this, FOSA is not connected to loan lifecycle, breaking the core SACCO value proposition.  
**Implementation**: `schedule_loan_deduction(tenant_id, account_id, loan_id, amount, deduction_date, frequency)`, `process_loan_deductions(tenant_id, processing_date)` — on due date, deducts from FOSA balance, credits loan repayment GL, posts loan payment event for the BOSA loan service to consume.  
**Competitor Reference**: Stima SACCO, Mwalimu National — automatic payroll deduction linked to FOSA.

---

### I5. Multi-Currency Accounts & FX Conversion
**Category**: International Banking  
**Justification**: Kenya-based SACCOs with diaspora members (e.g., KESALL in USA, KUSCCO diaspora chapters) need USD/EUR/GBP accounts. All balances are currently KES only.  
**Implementation**: `open_fosa_account` extended with `currency` already present but no FX engine. Add `fx_convert(tenant_id, source_account_id, target_currency, amount, fx_rate)` — posts GL with FX revaluation account, records exchange rate used, generates CBK-compliant FX transaction report.  
**Competitor Reference**: Equity Bank diaspora accounts, I&M Bank multi-currency.

---

### I6. Transaction Dispute & Chargeback Workflow
**Category**: Risk / Compliance  
**Justification**: CBK Prudential Guidelines require FOSA operators to have a formal dispute resolution process with SLA tracking (72h for M-PESA, 30 days for card). Currently no dispute model exists.  
**Implementation**: `raise_dispute(tenant_id, account_id, transaction_id, dispute_type, description)`, `resolve_dispute(tenant_id, dispute_id, resolution, resolved_by, reversal_amount)` — manages dispute lifecycle (raised → investigating → resolved/rejected), posts reversal GL on resolution, triggers member notification event.  
**Competitor Reference**: CBK Consumer Protection Guidelines, Equity Bank dispute portal.

---

### I7. KYC / AML Transaction Monitoring
**Category**: Compliance / AML  
**Justification**: CBK AML/CFT Guidelines (2023) and FATF Recommendation 10 require SACCOs to flag structuring, large cash transactions, and dormant-account activations. No AML screening exists.  
**Implementation**: `screen_transaction_aml(tenant_id, account_id, amount, channel, txn_type)` — checks: cash transactions > KES 1M (CBK CTR threshold), multiple transactions just below threshold (structuring), dormant account sudden large deposit. Returns `{risk_level, flags, requires_ctr}`. Called pre-commit inside `deposit`/`withdraw`.  
**Competitor Reference**: KCB, Co-op Bank — CBK-mandated AML systems, KFIU reporting.

---

### I8. Bulk Salary Processing (Payroll Credits)
**Category**: Corporate Banking  
**Justification**: SACCOs with employer partnerships (e.g., government, county, parastatal) receive payroll files and must credit hundreds of SALARY accounts atomically. No bulk credit method exists.  
**Implementation**: `process_salary_batch(tenant_id, batch_reference, salary_credits)` where `salary_credits: list[{account_id, amount, narration, employer_reference}]` — atomically credits accounts, posts GL per entry, generates batch settlement summary, handles partial failures with rollback-per-line and error report.  
**Competitor Reference**: Stima SACCO (KPLC payroll), Teachers SACCO (TSC payroll).

---

### I9. Account Freeze / Unfreeze with Audit
**Category**: Governance  
**Justification**: Current `withdraw` checks for `frozen` status but there is no method to programmatically freeze/unfreeze accounts. Court orders, fraud investigations, and deceased-member estates require formally tracked freeze actions with mandatory reason and approver.  
**Implementation**: `freeze_account(tenant_id, account_id, reason, freeze_type, ordered_by, court_reference)`, `unfreeze_account(tenant_id, account_id, unfreeze_reason, authorized_by)` — updates status, records full audit trail including `freeze_type` in `{court_order, fraud_investigation, member_request, deceased}`, emits compliance events.  
**Competitor Reference**: CBK Banking Supervision Guidelines — mandatory freeze audit trail.

---

### I10. Member Notification Dispatch
**Category**: Member Experience  
**Justification**: Leading mobile-first SACCOs (Sacco Societies, Mwalimu) send SMS/push notifications on every debit/credit above a threshold. Current service emits audit events but has no notification dispatch — members are unaware of transactions in real time.  
**Implementation**: `configure_notification_prefs(tenant_id, account_id, sms_enabled, push_enabled, email_enabled, min_amount_threshold)`, `dispatch_transaction_notification(tenant_id, txn)` — builds notification payload (masked account, amount, balance, channel), enqueues to notification service. Called at end of `deposit`/`withdraw`/`mpesa_cash_in`/`mpesa_cash_out`.  
**Competitor Reference**: Equity Bank, M-Pesa — real-time SMS per transaction.

---

### I11. End-of-Day Teller Reconciliation & Variance Detection
**Category**: Operations  
**Justification**: `get_teller_summary` returns a hardcoded `variance: 0` because no physical cash count input is modeled. CBK requires tellers to certify end-of-day balances with documented variance resolution.  
**Implementation**: `submit_teller_cash_count(tenant_id, teller_id, date_str, physical_count, denominations, submitted_by)`, `get_teller_reconciliation_report(tenant_id, teller_id, date_str)` — computes variance = physical_count − closing_float, flags over/short, stores denominations breakdown, marks teller session as reconciled.  
**Competitor Reference**: T24 (Temenos) teller module, Finacle teller reconciliation.

---

### I12. Fixed Deposit Maturity & Rollover Management
**Category**: Deposit Products  
**Justification**: `FIXED_DEPOSIT` account type exists but there is no maturity date enforcement, interest rate, or rollover logic. Fixed deposits are a core revenue product for SACCOs (typically 3–12 month terms at 8–12% p.a.).  
**Implementation**: `create_fixed_deposit(tenant_id, account_id, principal, term_months, interest_rate, maturity_action)` where `maturity_action` in `{rollover, credit_fosa, credit_bosa}`, `process_fixed_deposit_maturities(tenant_id, processing_date)` — calculates simple/compound interest, posts GL, executes maturity action, emits event.  
**Competitor Reference**: Co-op Bank fixed deposits, Equity SACCO term accounts.

---

### I13. Peer-to-Peer Member Transfers
**Category**: Payments  
**Justification**: Modern SACCOs (Sacco Link, M-Sacco) allow member-to-member transfers within the SACCO — analogous to bank transfers but internal, zero-fee, instant. Currently no intra-SACCO transfer method.  
**Implementation**: `transfer_between_members(tenant_id, source_account_id, dest_account_id, amount, reference, narration)` — validates both accounts active and same tenant, checks daily transfer limit, atomically debits source / credits destination, posts internal GL transfer, emits events for both accounts.  
**Competitor Reference**: M-Sacco, Sacco Link interoperability layer.

---

### I14. Regulatory Reporting (CBK Returns)
**Category**: Compliance  
**Justification**: SACCOs filing FOSA deposits with CBK (SASRA Directive on FOSA SACCOs) must submit monthly prudential returns: total deposits, dormancy ratios, overdraft exposure, M-PESA volumes. Manual extraction is error-prone.  
**Implementation**: `generate_cbk_monthly_return(tenant_id, year, month)` — aggregates portfolio data into CBK-mandated schedule format: total deposits, deposit by type, overdraft exposures, dormancy counts, channel transaction volumes, interest paid. Returns structured dict and emits `cbk_return_generated` audit event.  
**Competitor Reference**: SASRA FOSA SACCO prudential reporting framework.

---

### I15. Configurable Charge Schedules (Service Fee Engine)
**Category**: Revenue  
**Justification**: SACCOs charge maintenance fees, withdrawal fees, M-PESA fees, card fees, and statement fees. Hardcoded zero-fee model means revenue GL account (`GL_CHARGES_INCOME = "4300"`) is never posted, distorting financials.  
**Implementation**: `configure_charge_schedule(tenant_id, charge_type, amount, frequency, applicable_account_types)`, `apply_transaction_charge(tenant_id, account_id, charge_type, reference)` — looks up applicable charge, deducts from account, posts GL DR Member FOSA Deposits / CR Fee Income, records charge in audit trail. Charges are exempt if account balance < minimum balance threshold.  
**Competitor Reference**: Equity Bank service charge schedule, KCB SACCO tier-based fees.
