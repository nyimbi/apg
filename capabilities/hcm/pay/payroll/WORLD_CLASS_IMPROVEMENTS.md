# Payroll Management — World Class Improvements

Fifteen targeted improvements that push `pay_payroll` from a solid African payroll engine toward Workday/SAP-grade sophistication.

---

### I1. Real-Time Pay Simulation & What-If Engine
**Category**: Employee Self-Service / Analytics
**Justification**: Employees and HR cannot currently model the impact of a salary change, a pension increase, or a new allowance before the official run. A live simulation endpoint eliminates the cycle of "try it on the test run, roll back, redo". Workday Compensation Planning and Sage People both expose this; APG can do it in-process with zero infrastructure.
**Implementation**: Add `async def simulate_pay_change(employee_id, deltas, tenant_id)` that clones the profile in-memory, applies a dict of proposed changes (`{"base_pay": 120_000, "pension_ee_pct": 10}`), runs the full `calculate_paye` + `calculate_statutory_deductions` waterfall, and returns a before/after diff object. No writes occur; the clone is discarded. Wire to a `/simulate` API endpoint secured with `pay_payroll:view`.
**Competitor**: Workday Payroll What-If, ADP SmartCompliance

---

### I2. Multi-Currency FX-Aware Payroll Runs
**Category**: Global Payroll
**Justification**: Multinationals pay expatriates in USD while computing statutory obligations in KES. The current engine hard-codes a single currency per pay group, forcing operators to maintain separate groups for USD-denominated staff — a maintenance tax. Supporting a `home_currency` / `payment_currency` split with daily FX rates stored in-service matches how SAP Global Payroll handles shadow payroll.
**Implementation**: Add `async def set_fx_rate(from_currency, to_currency, rate, effective_date, tenant_id)` storing rates in a new `fx_rates` dict. Extend `run_payroll` to detect `profile.payment_currency != pay_group.currency`, apply the rate for the net-pay figure sent to bank file, while all tax calculations use local currency. Emit `fx_rate_applied` audit events per employee.
**Competitor**: SAP Payroll FX Handling, Oracle HCM Global Payroll

---

### I3. Bulk Payroll Reversal & Reprocessing
**Category**: Payroll Operations / Compliance
**Justification**: Errors discovered post-posting require full reversal of journal entries, bank file cancellation, and reprocessing — currently a manual multi-step workaround. SAP provides `HRPAY_REVERSAL_POST` as a first-class transaction; APG has no equivalent. Omitting this forces auditors to accept manual corrections, which regulators in KE and NG reject.
**Implementation**: Add `async def reverse_payroll_run(run_id, reason, reversed_by, tenant_id)` that marks the run `reversed`, emits negating GL entries via the existing `gl_posting` mechanism (each debit becomes a credit and vice versa), voids associated bank files, and creates a new run in state `reversal_pending` linked via `reversed_run_id`. Reprocessing calls `run_payroll` with the corrected data and cross-references the original run.
**Competitor**: SAP Payroll Retroactive Accounting, Workday Retro Pay

---

### I4. Leave Liability Accrual Tracker
**Category**: Financial Reporting / Compliance
**Justification**: Under IAS 19, organisations must accrue the monetary value of unconsumed leave as a balance-sheet liability. Payroll already tracks leave encashment on exit; it does not track the rolling accrual monthly, making it impossible to produce an IAS 19-compliant note without exporting raw data to a spreadsheet. Sage X3 and Oracle Payroll both produce this automatically.
**Implementation**: Add `async def accrue_leave_liability(employee_id, leave_days_balance, period, tenant_id)` that fetches the employee's daily rate (base_pay / working_days), multiplies by leave_days_balance, stores the result in a new `leave_accruals` dict, and emits GL credit to a configurable leave-liability account. Add `async def get_leave_liability_summary(tenant_id)` that aggregates across all employees, useful for month-end close.
**Competitor**: Oracle Payroll Leave Accrual, Sage X3 HR Module

---

### I5. NATS-Based Real-Time Payroll Event Streaming
**Category**: Event-Driven Integration / Architecture
**Justification**: `_emit` currently appends to an in-memory list. This is fine for testing but gives production integrations no way to react to payroll events without polling. Publishing to NATS JetStream subjects (e.g. `payroll.run.posted`, `payroll.payslip.published`) lets downstream capabilities (HCM notifications, GL posting, finance dashboards) subscribe with at-least-once delivery, exactly the model used in the Bytewax+NATS reference architecture.
**Implementation**: Add optional `nats_client` injection in `__init__`. Wrap `_emit` to also call `await nats_client.publish(f"payroll.{event_type}", orjson.dumps(payload))` when the client is present. Gracefully fall back to in-memory when no client is provided. Add `async def replay_audit_events(tenant_id, from_seq, nats_client)` for recovery scenarios. Document subject taxonomy in `cap_spec.md`.
**Competitor**: Kafka-based payroll pipelines (ADP, Workday) — APG uses NATS+Bytewax instead.

---

### I6. Automated Tax Table Version Management
**Category**: Compliance / Regulatory
**Justification**: `PAYE_TABLES` is hard-coded in `service.py`. Every budget cycle (KRA, URA, TRA publish new rates in June/July) requires a code deploy. Competitors like SAP HR apply a `validity_period` to each tax table row, selecting the applicable table by effective date at run time. Missing a rate change exposes clients to under-deduction penalties.
**Implementation**: Convert `PAYE_TABLES` to a `paye_table_versions` dict keyed by `(country, valid_from)`. Add `async def upsert_tax_table(country, valid_from, bands, reliefs, tenant_id)` that stores an override in `self.tax_table_overrides`. Modify `calculate_paye` to select the most-recent table version whose `valid_from <= pay_period.start_date`. Ship a `seed_official_tables()` utility that loads the current hard-coded values as version `2025-07-01`.
**Competitor**: SAP Payroll Tax Variant, Workday Tax Effective Dating

---

### I7. P10 / Annual Tax Reconciliation Report (Kenya)
**Category**: Statutory Compliance
**Justification**: KRA requires employers to file P10 (employer annual reconciliation) and P9A (individual certificate) by end of February each year. The engine produces individual P9 forms but has no P10 aggregate. Without it, compliance officers manually sum 12 months of P9 data — error-prone and time-consuming. Sage Pastel and QuickBooks Payroll auto-generate P10.
**Implementation**: Add `async def generate_p10_report(tenant_id, tax_year, approved_by)` that aggregates all P9 forms for the tax year, totals gross pay, taxable income, and PAYE per employee and overall, formats the KRA P10 layout, and stores the result in a new `p10_reports` dict. Also surface a `generate_p9a_certificate(employee_id, tax_year, tenant_id)` that reuses `generate_p9_form` data but outputs the employee-facing P9A format.
**Competitor**: Sage Pastel Payroll P10/P9A, iTax KRA integration

---

### I8. Cost-Centre & Project Payroll Allocation
**Category**: Financial Reporting / Project Accounting
**Justification**: Organisations split payroll costs across departments, projects, and grants. The current GL journal posts everything to a single `gross_pay_expense` account. SAP CO-PA and Oracle Project Costing let payroll administrators define allocation rules so 40% of an engineer's cost goes to Project A and 60% to Department B. Without this, finance teams do manual journal re-allocations monthly.
**Implementation**: Add `async def set_cost_allocation(profile_id, allocations, effective_date, tenant_id)` where `allocations` is a list of `{"cost_centre": "CC01", "project_id": "P001", "pct": 40}` dicts summing to 100. Extend `gl_posting` to split the gross-pay debit line according to active allocations, emitting one GL line per allocation split. Store active allocations in `self.cost_allocations`.
**Competitor**: SAP CO-PA Payroll Split, Oracle Payroll Project Costing

---

### I9. Configurable Payroll Approval Workflow
**Category**: Governance / Compliance
**Justification**: Currently `approve_payroll_run` accepts any `approved_by` string. There is no enforcement of multi-level approval (Finance Manager → CFO for runs above a threshold), no time-boxed SLA, and no escalation path. Workday Payroll enforces configurable approval chains; ADP Workforce Now requires dual sign-off above configurable thresholds. Auditors cite this gap in payroll SOX controls.
**Implementation**: Add `async def configure_approval_policy(pay_group_id, levels, amount_threshold, sla_hours, tenant_id)` storing policy in `self.approval_policies`. Add `async def submit_for_approval(run_id, submitted_by, tenant_id)` that creates an `approval_request` record with SLA deadline. Modify `approve_payroll_run` to validate that the approver satisfies the required level and that all levels have been cleared before allowing posting.
**Competitor**: Workday Payroll Approval Chains, ADP Workforce Now Dual Sign-Off

---

### I10. Payroll Analytics & Variance Drill-Down
**Category**: Analytics / Operations
**Justification**: The existing `payroll_variance_report` compares two runs but returns only top-level numeric deltas. Decision-makers need drill-down: which employees drove the variance, which components changed, and whether the change is expected (promotion) or anomalous (data entry error). Ramco Payroll and Workday deliver drill-to-employee variance trees; APG delivers a flat dict.
**Implementation**: Extend `payroll_variance_report` to add `employee_variances: list` — each entry containing `employee_id`, `gross_delta`, `net_delta`, `component_deltas: dict`, `variance_flag` (`expected | anomalous | review_required`). Add `async def explain_variance(run_id_a, run_id_b, employee_id, tenant_id)` that returns a structured narrative: prior values, new values, and a computed explanation string. Flag anomalies where gross delta > 2 standard deviations from the run mean.
**Competitor**: Workday Payroll Analytics, Ramco Payroll Variance Intelligence

---

### I11. Shift & Roster-Aware Gross Computation
**Category**: Time & Attendance Integration
**Justification**: The engine receives pre-computed hours via `record_time_import`. It cannot natively interpret shift differentials (night allowance, weekend premium) or roster patterns (6-days-on / 2-off for mining/hospitality). This forces upstream time systems to do the gross computation, defeating the purpose of a centralised payroll engine. ADP iHCM and Sage 200 Payroll both apply shift differentials from roster data.
**Implementation**: Add `async def import_shift_schedule(employee_id, period, shifts, tenant_id)` where `shifts` is a list of `{"date": "2026-01-15", "type": "night", "hours": 8}` objects. Maintain a `shift_differentials` config dict per pay group with multipliers per shift type. During `run_payroll`, compute gross by summing `regular_hours * base_rate + Σ(shift_hours * base_rate * differential_multiplier)`.
**Competitor**: ADP iHCM Shift Differentials, Sage 200 Payroll Roster Integration

---

### I12. Defined-Benefit Pension Projection Engine
**Category**: Financial Planning / Employee Retention
**Justification**: Most African statutory pension schemes (NSSF, RSSB, NAPSA) are defined-contribution, but some employers supplement with defined-benefit (DB) schemes. The engine has no facility to project DB pension obligations using actuarial assumptions (discount rate, salary growth, mortality). IAS 19 requires DB obligations on the balance sheet; without projections, the finance team engages external actuaries for each year-end — expensive and slow.
**Implementation**: Add `async def project_pension_obligation(employee_id, scheme_params, projection_years, tenant_id)` where `scheme_params` carries `{"accrual_rate": 0.0167, "discount_rate": 0.10, "salary_growth": 0.05}`. Use the projected-unit-credit method: for each future year, project salary, accrue the unit credit, discount back to present value. Return `{"present_value_obligation": ..., "current_service_cost": ..., "interest_cost": ...}`.
**Competitor**: Oracle Payroll Pension Module, Willis Towers Watson actuarial models

---

### I13. Automatic Payslip Delivery via Multiple Channels
**Category**: Employee Experience / Compliance
**Justification**: `publish_payslip` marks a record published but does not deliver it. Employees in low-connectivity environments need payslips via SMS summary, WhatsApp PDF, or email. Zambian and Rwandan labour law mandates written pay advice; simply "publishing" to a portal fails employees without smartphone access. BambooHR and Gusto deliver via email and SMS automatically; APG produces no outbound delivery.
**Implementation**: Add `async def deliver_payslip(payslip_id, channels, tenant_id)` where `channels` is a list of `{"type": "email" | "sms" | "whatsapp", "address": "..."}` objects. Integrate with the `ntfy` capability adapter for actual delivery; in the absence of an adapter, store delivery requests in a new `payslip_deliveries` dict with status `queued`. Emit `payslip_delivery_queued` NATS events for downstream delivery workers. Include a compact SMS template (5 lines: name, period, gross, deductions, net).
**Competitor**: Gusto Payslip Delivery, BambooHR Pay Stub Email

---

### I14. AI-Assisted Anomaly Detection on Payroll Runs
**Category**: AI / Fraud Prevention
**Justification**: Ghost employees, duplicate payments, and salary manipulation are among the top payroll fraud vectors in Africa (per ACFE 2024 Report). The existing `ai_intelligence_engine.py` provides raw AI infra but is not wired into run-time anomaly gating. Integrating lightweight statistical checks (z-score outliers, sudden grade jumps, unrecognised bank accounts) before a run is approved prevents rather than detects fraud. Workday and Ceridian Dayforce both offer pre-approval anomaly gating.
**Implementation**: Add `async def detect_payroll_anomalies(run_id, tenant_id, sensitivity)` that runs four checks: (1) gross > mean + 3σ, (2) new bank account not seen in prior 3 runs, (3) component added < 24 hours before run freeze, (4) employee hired same day as payslip. Return structured `anomalies: list[{"employee_id", "check", "severity", "detail"}]`. Gate `approve_payroll_run` to require anomaly sign-off when `severity="high"` records exist.
**Competitor**: Workday Payroll Anomaly Detection, Ceridian Dayforce AI Audit

---

### I15. Pensionable Earnings Certificate & Benefit-in-Kind Reporting
**Category**: Statutory Compliance
**Justification**: Several African revenue authorities (KRA, FIRS, GRA) require employers to declare benefits-in-kind (company car, housing, medical) as part of the taxable income declaration. The engine currently ignores BIK completely — an employee with a company car valued at KES 10,000/month is under-declared by that amount every period. This creates contingent tax liability for the employer. SAP and Workday both compute BIK imputed income and include it in PAYE calculations.
**Implementation**: Add `async def register_benefit_in_kind(employee_id, bik_type, monthly_value, effective_date, tenant_id)` storing BIK entries in `self.benefits_in_kind`. Modify `calculate_paye` to add total BIK for the employee to `gross_monthly` before the taxable-income calculation (flagging it separately so the net-pay line is unaffected — BIK does not produce a cash payment). Add `async def generate_bik_report(tenant_id, period)` for the annual statutory return.
**Competitor**: SAP Payroll Benefits-in-Kind, Workday Imputed Income, HMRC P11D
