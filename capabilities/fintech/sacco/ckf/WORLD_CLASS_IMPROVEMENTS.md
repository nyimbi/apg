# SACCO Check-off — 15 World-Class Improvements

---

### I1. Bulk Reconciliation Across All Employers in Parallel
**Category:** Performance / Operations
**Justification:** SACCOs with 50+ check-off employers run month-end close serially today; that wall-clock time is unacceptable. Kenya's Mwalimu National SACCO processes 80,000+ payslips monthly — a sequential loop over employers takes minutes. asyncio.gather() over employer reconciliations reduces that to the latency of the slowest employer.
**Implementation:** Add `bulk_reconcile_all_employers(tenant_id, payroll_month, payroll_year)` that fans out `reconcile_check_off` calls with `asyncio.gather`, collects per-employer results into a `BulkReconciliationSummary` and returns aggregate pass/fail/short/over counts in one call.
**Competitor Reference:** Co-op Bank Kenya payroll portal; Sahl Finance SACCO software.

---

### I2. Demand Notice Generation with Audit Trail
**Category:** Compliance / Collections
**Justification:** Central Bank of Kenya Prudential Guidelines §16 require formal demand evidence before a SACCO can reclassify a loan as non-performing. Currently `send_remittance_reminder` just increments a counter. A structured `DemandNotice` record with notice number, served-at, served-by, legal basis and acknowledgement status is needed to satisfy auditors and the tribunal.
**Implementation:** Add `issue_demand_notice(tenant_id, employer_id, month, year, notice_text, issued_by)` creating a `DemandNotice` record, advancing status to `DEMAND_ISSUED`, and logging the audit event. Add `acknowledge_demand_notice(tenant_id, notice_id, acknowledged_by, acknowledged_at)`.
**Competitor Reference:** Orion SACCO ERP (Uganda); CIC Money Market Fund recovery module.

---

### I3. Partial Remittance Acceptance with Pro-Rata Allocation
**Category:** Financial Accuracy
**Justification:** Employers often remit a lump-sum slightly below the expected total. Treating any shortfall as a full rejection is operationally wrong — the SACCO should accept what was paid, allocate proportionally across loan/savings, and carry forward only the shortfall. This matches how Co-operative Bank's FOSA desk works.
**Implementation:** Add `accept_partial_remittance(tenant_id, employer_id, month, year, amount_received, allocation_method)` that pro-rates `amount_received` across members by their share of total expected, posts partial GL entries, and generates per-member shortfall arrears records.
**Competitor Reference:** Equity Bank Agency Banking settlement engine; Eclectics USSD SACCO module.

---

### I4. Automated Arrears Ageing Report
**Category:** Risk Management
**Justification:** Arrears that are 1–30 / 31–60 / 61–90 / 90+ days bucket differently under IFRS 9 Expected Credit Loss models. The current service tracks arrears as a list but provides no ageing view, making provisioning impossible. Every regulated SACCO in East Africa requires this for the Commissioner of Co-operatives' annual return.
**Implementation:** Add `get_arrears_ageing_report(tenant_id, as_at_date)` that buckets per-member shortfalls into standard ageing bands (30/60/90/90+) and returns ECL-ready totals per bucket, per employer, and sacco-wide.
**Competitor Reference:** Temenos T24 SACCO module; i-Sacco (Kenya Co-operative Savings & Credit Union).

---

### I5. Check-off Agreement Expiry Tracking and Renewal Workflow
**Category:** Contract Lifecycle Management
**Justification:** Check-off agreements typically have 1–3 year terms requiring renewal signatures. Most SACCOs lose check-off authority when agreements silently lapse. An expiry-aware system blocks schedule generation on expired agreements and surfaces a renewal queue.
**Implementation:** Add `expiry_date` and `renewal_status` fields to `Employer` model; add `get_expiring_agreements(tenant_id, days_ahead)` returning employers whose agreements expire within `days_ahead` days; add `renew_check_off_agreement(tenant_id, employer_id, new_agreement_date, renewed_by)`.
**Competitor Reference:** Wezatele SACCO system (Tanzania); Kuscco MFIS module.

---

### I6. Multi-Currency Support with FX Rate Pinning
**Category:** International / NGO SACCOs
**Justification:** Diaspora SACCOs (e.g., KENASACCO-UK, US-based Kenyan staff SACCOs) have members paid in GBP/USD whose deductions are remitted in KES. Without an FX-pinned rate per schedule, reconciliation produces phantom variances. The rate must be locked at schedule generation and used for all downstream reconciliation.
**Implementation:** Add `currency` and `fx_rate_to_base` fields to `CheckOffSchedule` and `RemittanceRecord`; add `set_period_fx_rate(tenant_id, employer_id, month, year, currency, rate, rate_source)` and normalise all monetary comparisons to base currency at reconciliation time.
**Competitor Reference:** Diaspora SACCO (Kenya); WorldRemit payroll integration specs.

---

### I7. Schedule Version Control and Amendment Log
**Category:** Audit / Correctness
**Justification:** Payroll departments frequently request corrections after a schedule is issued (member joins/leaves mid-month, salary change, ad-hoc loan top-up). The current system regenerates silently, losing the original. SACCO auditors require a version trail — "what was sent vs what was reconciled."
**Implementation:** Add `version` and `superseded_by` fields to `CheckOffSchedule`; instead of overwriting, `generate_check_off_schedule` creates a new version and links to the previous; add `get_schedule_versions(tenant_id, employer_id, month, year)` returning the amendment chain with change deltas.
**Competitor Reference:** SAP Payroll Reconciliation audit trail; Oracle HRMS check-off module.

---

### I8. Member Deduction Cap Enforcement (Salary Protection)
**Category:** Consumer Protection / Regulatory
**Justification:** The SACCO Societies Regulatory Authority (SASRA) Act §35 limits total deductions to 2/3 of basic salary. Generating schedules that violate this cap exposes the SACCO to member complaints and regulatory sanctions. The system must enforce the cap at schedule generation, not just report it.
**Implementation:** Add `max_deduction_pct` (default 66.67) to `MemberEmployerLink`; in `generate_check_off_schedule` compute capped total per member, flag `is_over_cap` on `ScheduleMemberEntry`, drop excess deductions with priority order: arrears > savings > interest > principal (configurable).
**Competitor Reference:** SASRA Prudential Guidelines 2023; KCB-SACCO product terms.

---

### I9. Automated Reconciliation Discrepancy Alerting
**Category:** Operations / Monitoring
**Justification:** Operations staff only discover short-payments at report time. An event-driven alert — triggered the moment reconciliation flags a discrepancy — shrinks the collection response window from days to hours. This is standard in tier-1 bank treasury operations.
**Implementation:** Add `register_alert_handler(handler: Callable[[ReconciliationAlert], Awaitable[None]])` on `CheckOffService`; after every `reconcile_check_off` that returns non-RECONCILED, fire a `ReconciliationAlert` event with employer details, variance, and recommended action, dispatching to all registered handlers.
**Competitor Reference:** Temenos Infinity alert centre; Mambu webhook events.

---

### I10. Employer Payment History Scoring
**Category:** Risk Intelligence
**Justification:** Not all non-payment is equal — a first-time miss is different from a serial offender. A compliance score (0–100) derived from rolling 12-month payment history enables the SACCO to tier response: auto-reminder vs. legal threat vs. agreement suspension. This is equivalent to a credit score applied to the employer relationship.
**Implementation:** Add `compute_employer_compliance_score(tenant_id, employer_id, lookback_months)` that scans the last N remittances, scores each (on-time=1, partial=0.5, late=0.25, default=0), weights by recency (exponential decay), returns a 0–100 score with a letter grade (A/B/C/D/F) and trend direction.
**Competitor Reference:** Creditinfo Kenya employer scoring; Metropol CRB trade credit model.

---

### I11. GL Account Configurability per Tenant
**Category:** Accounting Flexibility
**Justification:** The current hardcoded `GL_CHECKOFF_RECEIVABLE = "1310"` forces every tenant to use the same chart of accounts. Multi-tenant deployments (different SACCOs on one APG instance) have different CoA numbering schemes. SASRA-registered SACCOs must follow their own auditor-approved CoA.
**Implementation:** Add `configure_gl_accounts(tenant_id, checkoff_receivable, loan_ledger, savings_ledger, penalty_ledger)` persisting per-tenant GL config; all GL posting methods read tenant config first, falling back to module defaults. Validate account codes are non-empty strings.
**Competitor Reference:** QuickBooks multi-entity CoA mapping; Sage Intacct entity-level chart of accounts.

---

### I12. Salary Change Propagation with Effective Date
**Category:** Data Accuracy
**Justification:** When a member receives a pay rise, the basic salary on their employer link must be updated with an effective date — not retroactively — to avoid recalculating historical deduction caps. The current `add_member_employer_link` creates a new link on salary change, losing the history context. A dedicated salary revision path preserves the audit trail.
**Implementation:** Add `update_member_salary(tenant_id, member_id, new_salary, effective_date, change_reason)` that records a `SalaryRevisionLog` entry and updates the active link; add `get_salary_history(tenant_id, member_id)` returning the revision log.
**Competitor Reference:** Workday HCM salary change workflow; BambooHR compensation history.

---

### I13. Bulk Member Upload via CSV/Dict Batch
**Category:** Onboarding Efficiency
**Justification:** New check-off employers arrive with hundreds of employee records in a payroll export. The current API requires one `add_member_employer_link` call per member — O(n) round trips. A batch importer with validation, duplicate detection, and per-row error reporting matches what payroll integrations actually deliver.
**Implementation:** Add `bulk_link_members(tenant_id, employer_id, member_rows: list[dict], effective_date)` that validates each row against `MemberEmployerLink` schema, processes good rows, and returns `BulkLinkResult(processed, failed, errors_by_row)`.
**Competitor Reference:** Gusto payroll CSV import; Rippling HR bulk onboarding.

---

### I14. Period Rollover with Carry-Forward Logic
**Category:** Operational Automation
**Justification:** Each new payroll month requires staff to manually trigger schedule generation for every employer. A rollover operation copies forward any unresolved arrears, marks the previous period as closed, and opens the new period — matching how the Co-op Bank FOSA desk month-end close works.
**Implementation:** Add `rollover_to_next_period(tenant_id, payroll_month, payroll_year)` that: (1) asserts all posted employers are in POSTED status, (2) carries forward PARTIAL/SHORT_PAID amounts as arrears, (3) triggers `batch_process_all_employers` for the next month, (4) returns a `PeriodRolloverResult` with counts and total carried-forward arrears.
**Competitor Reference:** Navision SACCO end-of-period close; Sage Pastel month-end rollover.

---

### I15. Audit Trail with Immutable Event Log
**Category:** Compliance / Governance
**Justification:** Financial regulators (CBK, SASRA) require immutable audit logs showing who changed what and when. The current service mutates in-memory dicts with no history. An append-only event log — even in-memory for now — enables forensic reconstruction of any state and is a prerequisite for SOC 2 and ISO 27001 compliance audits.
**Implementation:** Add `AuditEvent(event_type, tenant_id, entity_type, entity_id, actor, before_state, after_state, occurred_at)` model; wrap all state-mutating methods to append an `AuditEvent` to `self._audit_log`; add `get_audit_trail(tenant_id, entity_type, entity_id, limit)` for retrieval; add `export_audit_log(tenant_id, from_date, to_date)` for compliance exports.
**Competitor Reference:** Temenos T24 audit module; Oracle Financials audit vault.
