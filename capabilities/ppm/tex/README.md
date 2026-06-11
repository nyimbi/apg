# Time & Expense Management

## Overview
Time & Expense Management (tex) handles the complete employee time and expense lifecycle: weekly/bi-weekly/monthly timesheet entry with project and task linkage, expense claim submission with receipt enforcement above configurable thresholds, multi-step approval workflows, reimbursement processing via payroll or bank transfer, billing rate management per resource/project, and billable hour export to project accounting.

## Capability ID
`ppm_tex`

## Provides
| Service | Description |
|---------|-------------|
| timesheet_entry_and_management | Weekly, bi-weekly, and monthly timesheets with project/task linkage |
| expense_claim_workflow | Categorised expense claims with receipt enforcement above threshold |
| approval_workflow_engine | Configurable single, parallel, sequential, and committee approval flows |
| billable_hour_tracking | Billable vs non-billable split per project |
| reimbursement_processing | Payroll, bank transfer, expense card, cheque, and petty cash methods |
| project_time_reporting | Time-by-project, resource, and task reports |
| billing_rate_management | Standard, preferential, overtime, blended, and NTE rate types |
| compliance_and_policy_enforcement | Receipt thresholds, backdating justification, duplicate rejection |
| multi_currency_expense_management | 10 currencies including KES, NGN, GHS, ZAR |
| audit_trail_for_time_and_expenses | Immutable event stream for payroll and billing audit |

## Requires
| Capability | Reason |
|------------|--------|
| auth | Authentication |
| audl | Immutable audit trail for payroll compliance |
| mten | Tenant scoping |
| conf | Receipt threshold and policy configuration |
| ntfy | Approval request and rejection notifications |
| wflo | Multi-step timesheet and expense approval workflows |
| comp | Payroll compliance and labour law adherence |
| mqeb | Event streaming via Bytewax |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| timesheets.supported_period_types | 5 | daily, weekly, bi_weekly, semi_monthly, monthly |
| timesheets.supported_billable_statuses | 5 | billable, non_billable, not_to_exceed, pro_bono, internal |
| expenses.receipt_threshold_amount | 25.00 | USD amount above which receipt is required |
| expenses.supported_categories | 11 | Travel, meals, software, training, and more |
| governance.duplicate_expense_submission_denied | true | Fraud prevention |
| governance.backdated_entry_requires_justification | true | Payroll integrity |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /ppm-tex/timesheets/my | GET | Personal timesheet list | ppm_tex:timesheets |
| /ppm-tex/timesheets/entry | POST | Timesheet entry form | ppm_tex:timesheets |
| /ppm-tex/timesheets/approvals | GET/POST | Timesheet approval queue | ppm_tex:approve_timesheets |
| /ppm-tex/expenses/my | GET | Personal expense list | ppm_tex:expenses |
| /ppm-tex/expenses/claim | POST | Expense claim form | ppm_tex:expenses |
| /ppm-tex/expenses/approvals | GET/POST | Expense approval queue | ppm_tex:approve_expenses |
| /ppm-tex/billable | GET | Billable hour tracker | ppm_tex:billing |
| /ppm-tex/rates | GET/POST | Billing rate table | ppm_tex:rates |
| /ppm-tex/reimbursements | GET/POST | Reimbursement console | ppm_tex:reimburse |
| /ppm-tex/agents | GET/POST | Agent workbench | ppm_tex:admin |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| timesheet_project_required | project_present=False | deny |
| time_entry_hours_must_be_positive | hours_positive=False | deny |
| backdated_entry_requires_justification | backdated + no justification | deny |
| expense_above_threshold_requires_receipt | above threshold + no receipt | deny |
| duplicate_expense_denied | duplicate_expense_submission=True | deny |
| reimbursement_approval_required | approval_present=False | deny |
| billing_rate_approval_required | approval_present=False | deny |
| cross_tenant_time_access_denied | cross_tenant_access=True | deny |
| tex_batch_requires_bytewax | event_stream != "bytewax" | deny |

## Data Models
- **Timesheet** — id, resource_id, project_id, period_type, period_reference, status, reviewer_id
- **TimeEntry** — id, timesheet_id, project_id, task_id, entry_type, billable_status, hours, entry_date
- **ExpenseClaim** — id, resource_id, project_id, category, currency, amount, status, receipt_status
- **Reimbursement** — id, expense_claim_id, resource_id, method, amount, currency, approval_reference
- **BillingRate** — id, resource_id, project_id, rate_type, rate_amount, effective_date, approval_reference
- **TimesheetApproval** — id, timesheet_id, reviewer_id, status, comments
- **ExpenseApproval** — id, expense_claim_id, reviewer_id, status, comments
- **TexAgent** — id, name, runtime, role, scope

## Service Methods

### Core Workflow
| Method | Description |
|--------|-------------|
| `submit_timesheet(employee_id, project_id, task_id, hours, date_str, description)` | Create/extend weekly timesheet with a time entry |
| `approve_timesheet(timesheet_id, approver_id)` | Approve a submitted timesheet |
| `reject_timesheet(timesheet_id, reason)` | Reject with reason |
| `submit_expense(employee_id, project_id, category, amount, currency, receipt_metadata, date_str)` | Submit expense claim |
| `approve_expense(expense_id, approver_id)` | Approve expense claim |
| `reject_expense(expense_id, reason)` | Reject with reason |
| `reimburse_expense(expense_id, reimbursement_date, payment_method)` | Process reimbursement |
| `per_diem_calculation(employee_id, travel_days, destination_type)` | Compute per diem entitlement |

### Bulk & Parallel Operations
| Method | Description |
|--------|-------------|
| `bulk_submit_time_entries(timesheet_id, entry_specs)` | Add multiple time entries in one call |
| `bulk_approve_timesheets(timesheet_ids, approver_id, comments)` | Approve multiple timesheets concurrently via `asyncio.gather` |
| `bulk_import(records)` | Import records in bulk |
| `bulk_delete(record_ids)` | Delete records by ID list |

### Analytics & Reporting
| Method | Description |
|--------|-------------|
| `timesheet_analytics(period)` | Billable ratio, utilisation, submission rate |
| `expense_analytics(period)` | Total spend, reimbursement rate, by-category breakdown |
| `billing_rate_report()` | Mean rates by type |
| `resource_utilisation_heatmap(resource_id, year, month)` | Day-indexed utilisation for calendar UI widgets |
| `forecast_expense_spend(project_id, lookahead_days)` | Committed + trend-based spend forecast with risk level |
| `employee_expense_summary(employee_id, period)` | Employee-scoped expense totals by status and category |
| `generate_report(report_type, period)` | Summary report generation |

### Compliance & Governance
| Method | Description |
|--------|-------------|
| `tex_compliance_check()` | Receipt threshold compliance rate |
| `compliance_check()` | General policy compliance |
| `check_overtime(employee_id, period_reference)` | Detect daily/weekly overtime breaches at submission time |
| `query_audit_log(event_type, reference_id, limit)` | Queryable audit log with optional filters |

### Time Entry Management
| Method | Description |
|--------|-------------|
| `record_time_entry(...)` | Low-level time entry creation |
| `amend_time_entry(entry_id, new_hours, justification)` | Create amendment preserving original as immutable history |
| `list_time_entries(tenant_id, timesheet_id)` | List entries with optional timesheet filter |
| `billable_hours_summary(tenant_id, project_id)` | Billable vs non-billable split |

### Payroll & Accounting Integration
| Method | Description |
|--------|-------------|
| `lock_timesheet_for_payroll(timesheet_id, payroll_run_id)` | Lock approved timesheet post-payroll to prevent double-processing |
| `export_to_project_accounting(project_id, period, pac_adapter)` | Compute labour cost and bundle expense claims for ppm_pac |
| `export_timesheets(format)` | Export to JSON or CSV |
| `export_records(format)` | Generic record export |

### Operations
| Method | Description |
|--------|-------------|
| `health_check()` | Service health status |
| `dashboard_summary(tenant_id)` | Counts across all entity types |
| `search(query)` | Full-text search across records |
| `archive_record(record_id, reason)` | Archive with reason |
| `restore_record(record_id)` | Restore archived record |

## Streaming Events
- `timesheet_submitted`, `timesheet_approved`, `timesheet_rejected`, `timesheet_locked`, `time_entry_recorded`, `time_entry_amended`
- `expense_claim_submitted`, `expense_approved`, `expense_rejected`, `expense_forecast_generated`
- `reimbursement_processed`, `billable_hours_exported`, `expense_costs_exported`, `billing_rate_updated`
- `overtime_detected`, `bulk_timesheets_approved`, `time_entries_bulk_submitted`
- `policy_violation_detected`, `agent_registered`, `tex_compliance_check_run`

## Edge Cases Handled
- Duplicate expense detection: same resource, date, amount, and category within the same tenant is rejected
- Receipt threshold (default $25) is configurable per tenant; receipt status "pending_upload" fails the check above threshold
- Backdated time entries require explicit justification text; empty strings are rejected
- Timesheet status is automatically updated when the approval decision is recorded
- Billing rate changes require both approval reference and effective date to prevent retroactive rate manipulation
- Payroll-locked timesheets (`status="locked"`) reject further approval or amendment attempts
- Time entry amendments create new records with amendment linkage; originals are never mutated
- Overtime detection runs at O(n entries) per employee/period; suitable for real-time submission checks
- `bulk_approve_timesheets` uses `asyncio.gather`; individual failures are isolated and reported without aborting the batch

## Composability Notes
- Billable hours and expense claims feed **ppm_pac** via `export_to_project_accounting`
- Billing rates are sourced from or cross-referenced with **ppm_res** cost rates
- Approved timesheets trigger project progress updates in **ppm_pps**
- `resource_utilisation_heatmap` output feeds the **ppm_res** capacity planning dashboard
- Expense spend forecasts integrate with **ppm_pac** budget alert thresholds
