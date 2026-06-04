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

## Streaming Events
- `timesheet_submitted`, `timesheet_approved`, `timesheet_rejected`, `time_entry_recorded`
- `expense_claim_submitted`, `expense_approved`, `expense_rejected`
- `reimbursement_processed`, `billable_hours_exported`, `billing_rate_updated`
- `policy_violation_detected`, `agent_registered`

## Edge Cases Handled
- Duplicate expense detection: same resource, date, amount, and category within the same tenant is rejected
- Receipt threshold (default $25) is configurable per tenant; receipt status "pending_upload" fails the check above threshold
- Backdated time entries require explicit justification text; empty strings are rejected
- Timesheet status is automatically updated when the approval decision is recorded
- Billing rate changes require both approval reference and effective date to prevent retroactive rate manipulation

## Composability Notes
- Billable hours feed **ppm_pac** actual labour cost transactions
- Billing rates are sourced from or cross-referenced with **ppm_res** cost rates
- Approved timesheets trigger project progress updates in **ppm_pps**
- Expense data can feed project cost actuals in **ppm_pac** for complete project P&L
