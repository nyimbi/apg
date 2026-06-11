# ESS World-Class Improvements

## 1. Leave Clash Detection
Before approving or even submitting a leave request, check for overlapping approved/pending leave for the same employee. Prevents double-booking and manager confusion. Implementation: `check_leave_conflicts(tenant_id, employee_id, start_date, end_date)` returns any colliding request IDs.

## 2. Leave Accrual Engine
Replace static balance defaults with a time-based accrual model: employees earn leave pro-rata per pay period. Supports part-time proration, probation lock-outs, and carry-over caps. Implementation: `accrue_leave(tenant_id, employee_id, period_end_date, accrual_rules)`.

## 3. Delegation / Acting-For Leave Handover
When a leave request names a `handover_to` employee, automatically create a delegation record granting that person access to the requestor's work queues for the leave period. Implementation: `create_leave_delegation(tenant_id, leave_request_id)`.

## 4. Multi-Level Leave Approval Workflow
Support configurable approval chains (e.g., direct manager → HR → Finance for long unpaid leave). Each step tracks approver, timestamp, and comments. Status moves through `pending_l1 → pending_l2 → approved`. Implementation: `advance_leave_approval_step(tenant_id, request_id, approver_id, step, comment)`.

## 5. Payslip Year-to-Date Aggregation
Compute cumulative YTD figures (gross, PAYE, NSSF, NHIF, net) across all payslips for an employee in a given year. Useful for tax certificates and employee queries. Implementation: `get_payslip_ytd(tenant_id, employee_id, year)`.

## 6. Payslip PDF Generation
Produce a structured payslip PDF (via `reportlab` or `weasyprint`) with employer branding, statutory deduction breakdowns, and QR code for verification. Implementation: `generate_payslip_pdf(tenant_id, payslip_id) -> bytes`.

## 7. Expense Claim Bulk Submit
Allow employees to submit all draft claims in a single call with optional total-limit validation (e.g., single-day per-diem cap). Reduces round-trips and enforces policy atomically. Implementation: `bulk_submit_expense_claims(tenant_id, employee_id, claim_ids, policy_limits)`.

## 8. Expense Policy Engine
Attach spend-limit rules per category/project_code (e.g., meals ≤ KES 2,000/day). Enforce on create/submit with a structured violation object rather than a bare exception. Implementation: `validate_expense_against_policy(tenant_id, claim)`.

## 9. Benefit Open-Enrollment Window
Allow HR to define an annual open-enrollment window; outside that window, enrolment changes are locked except for qualifying life events (marriage, birth, death). Implementation: `open_enrolment_window(tenant_id, start_date, end_date)` and `check_enrolment_eligibility(tenant_id, employee_id, event_type)`.

## 10. Training Completion Certificate Store
Persist issued certificates as first-class records (url, expiry, issuer, CPD credits). Support querying all certifications held by an employee, driving professional development dashboards. Implementation: `record_certificate(tenant_id, employee_id, cert_data)` and `list_employee_certificates(tenant_id, employee_id)`.

## 11. Bulk Leave Accrual Run
Process an entire tenant's accruals in one async batch (parallel per-employee coroutines), producing a structured run-report with counts of successes, errors, and balance changes. Implementation: `run_leave_accrual_batch(tenant_id, period_end_date, accrual_rules)`.

## 12. Document Attachment Tracking
Replace bare URL lists in leave/expense/training with structured `Attachment` objects carrying `filename`, `mime_type`, `size_bytes`, `uploaded_by`, and `uploaded_at`. Enables virus-scan hooks and retention policy enforcement. Implementation: `add_attachment(entity_type, entity_id, attachment_data)` and `list_attachments(entity_type, entity_id)`.

## 13. Employee Self-Service Notifications
Emit structured notification events (email, in-app, SMS) on key ESS state transitions: leave approved/rejected, expense paid, benefit near-expiry, training starting tomorrow. Backed by a notification preferences store per employee. Implementation: `send_ess_notification(tenant_id, employee_id, event_type, context)` and `upsert_notification_preferences(tenant_id, employee_id, prefs)`.

## 14. Compliance / Statutory Reporting
Aggregate statutory deduction totals (PAYE, NSSF, NHIF) per pay period across all employees for direct upload to KRA iTax, NSSF, and NHIF portals. Include validation against statutory rate tables. Implementation: `generate_statutory_report(tenant_id, period_month, period_year)`.

## 15. Time-Off-In-Lieu (TOIL) Tracker
Record overtime hours worked and convert them to compensatory leave credits at a configurable rate (e.g., 1.5× for weekend). Integrates with the accrual engine so TOIL balances appear alongside annual leave. Implementation: `record_toil(tenant_id, employee_id, overtime_date, hours_worked, rate)` and `get_toil_balance(tenant_id, employee_id)`.
