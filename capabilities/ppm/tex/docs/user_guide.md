# Time & Expense Management

**Capability ID**: `ppm_tex` | **Domain**: `ppm` | **Version**: `1.0.0`

## Description

Time & Expense Management (tex) handles the complete employee time and expense lifecycle: weekly/bi-weekly/monthly timesheet entry with project and task linkage, expense claim submission with receipt enforcement above configurable thresholds, multi-step approval workflows, reimbursement processing via payroll or bank transfer, billing rate management per resource/project, and billable hour export to project accounting.

## Installation

```bash
pip install apg-ppm-tex
```

## Provides

- `timesheet_entry_and_management`
- `expense_claim_workflow`
- `approval_workflow_engine`
- `billable_hour_tracking`
- `reimbursement_processing`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/ppm-tex/dashboard` | `ppm_tex:view` | Overview |
| `/ppm-tex/timesheets/my` | `ppm_tex:timesheets` | Timesheets |
| `/ppm-tex/timesheets/entry` | `ppm_tex:timesheets` | Timesheets |
| `/ppm-tex/timesheets/approvals` | `ppm_tex:approve_timesheets` | Approvals |
| `/ppm-tex/expenses/my` | `ppm_tex:expenses` | Expenses |
| `/ppm-tex/expenses/claim` | `ppm_tex:expenses` | Expenses |
| `/ppm-tex/expenses/approvals` | `ppm_tex:approve_expenses` | Approvals |
| `/ppm-tex/billable` | `ppm_tex:billing` | Billing |

## Key Service Methods

- `describe()`
- `evaluate()`
- `submit_timesheet()`
- `approve_timesheet()`
- `reject_timesheet()`
- `submit_expense()`
- `approve_expense()`
- `reject_expense()`
- `reimburse_expense()`
- `per_diem_calculation()`

_(See `service.py` for complete API.)_

## Interoperability

`ppm_tex` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use ppm_tex;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PPM_TEX_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
