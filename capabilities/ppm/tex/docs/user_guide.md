# Time & Expense Management — User Guide

**Capability ID**: `ppm_tex` | **Domain**: `ppm` | **Version**: `1.1.0`

---

## Overview

`ppm_tex` manages the complete employee time and expense lifecycle: weekly/bi-weekly/monthly timesheet entry with project and task linkage, expense claim submission with receipt enforcement, multi-step approval workflows, reimbursement processing, billing rate management, per-diem calculation, and composable export to project accounting.

---

## Installation

```bash
pip install apg-ppm-tex
```

---

## Quick Start

```python
from capabilities.ppm.tex.service import TimeExpenseService

svc = TimeExpenseService(tenant_id="acme", actor_id="mgr_001")

# Submit a time entry
result = await svc.submit_timesheet(
    employee_id="emp_101",
    project_id="proj_alpha",
    task_id="task_design",
    hours=6.5,
    date_str="2026-06-11",
    description="UI mockup review",
)

# Submit an expense
claim = await svc.submit_expense(
    employee_id="emp_101",
    project_id="proj_alpha",
    category="meals",
    amount=42.00,
    currency="USD",
    receipt_metadata={"filename": "receipt_jun11.jpg", "size_kb": 120},
    date_str="2026-06-11",
)

# Approve both
await svc.approve_timesheet(result["timesheet_id"], approver_id="mgr_001")
await svc.approve_expense(claim["expense_id"], approver_id="mgr_001")

# Reimburse
await svc.reimburse_expense(
    claim["expense_id"],
    reimbursement_date="2026-06-15",
    payment_method="bank_transfer",
)
```

---

## Core Workflows

### Timesheet Lifecycle

```
draft → submitted → approved → locked (after payroll run)
                 ↘ rejected
```

1. `submit_timesheet(...)` — creates or extends a weekly timesheet; adds a `TimeEntry` record.
2. `approve_timesheet(timesheet_id, approver_id)` — moves status to `approved`.
3. `reject_timesheet(timesheet_id, reason)` — moves to `rejected`; reason stored in approval record.
4. `lock_timesheet_for_payroll(timesheet_id, payroll_run_id)` — irreversibly locks after payroll consumption; prevents double-processing.

### Expense Claim Lifecycle

```
submitted → approved → reimbursed
          ↘ rejected
```

1. `submit_expense(...)` — validates category, currency, receipt threshold, duplicate.
2. `approve_expense(expense_id, approver_id)` — moves to `approved`.
3. `reject_expense(expense_id, reason)` — moves to `rejected`.
4. `reimburse_expense(expense_id, reimbursement_date, payment_method)` — creates `Reimbursement` record; marks claim as `reimbursed`.

**Receipt policy**: expenses above `$25.00` (configurable) require receipt metadata. Claims without a receipt reference above this threshold are rejected at submission time.

---

## Bulk Operations

### Bulk Time Entry Submission

```python
result = await svc.bulk_submit_time_entries(
    timesheet_id="ts_emp101_proj_alpha_2026-06",
    entry_specs=[
        {"task_id": "task_design", "hours": 8, "work_date": "2026-06-09", "entry_type": "regular", "billable_status": "billable"},
        {"task_id": "task_review", "hours": 4, "work_date": "2026-06-10", "entry_type": "regular", "billable_status": "billable"},
    ],
)
print(result["created_count"])  # 2
```

### Bulk Timesheet Approval

Approves a list of timesheets concurrently using `asyncio.gather`. Individual failures are isolated.

```python
result = await svc.bulk_approve_timesheets(
    timesheet_ids=["ts_001", "ts_002", "ts_003"],
    approver_id="mgr_001",
    comments="period-end batch approval",
)
print(result["approved_count"])  # 3
print(result["failed"])          # {} (empty if all succeeded)
```

---

## Time Entry Amendments

Amendments preserve original records as immutable history. A new `TimeEntry` is created with a derived ID, linked to the original.

```python
amendment = await svc.amend_time_entry(
    entry_id="te_emp101_proj_alpha_2026-06-09_task_design",
    new_hours=7.5,
    justification="Incorrect hours logged; corrected after manager review",
)
print(amendment["original_hours"])  # 8.0
print(amendment["new_hours"])       # 7.5
```

---

## Overtime Detection

Check for daily (>8 h) or weekly (>40 h) overtime at submission time or on-demand.

```python
ot = await svc.check_overtime(
    employee_id="emp_101",
    period_reference="2026-06",
    daily_limit_hours=8.0,
    weekly_limit_hours=40.0,
)
for warning in ot["overtime_warnings"]:
    print(warning["type"], warning["date"], warning["excess"])
```

---

## Per Diem Calculation

```python
pd = await svc.per_diem_calculation(
    employee_id="emp_101",
    travel_days=5,
    destination_type="international",  # domestic | international | high_cost_city
)
print(pd["total_per_diem"])  # 750.0 (5 × $150)
```

Destination type maps automatically for known city names (`london`, `new york`, `tokyo`, `zurich`, `singapore`).

---

## Analytics

### Timesheet Analytics

```python
analytics = await svc.timesheet_analytics(period="2026-06")
print(analytics["billable_ratio_pct"])
print(analytics["submission_rate_pct"])
```

### Expense Analytics

```python
exp = await svc.expense_analytics(period="2026-06")
print(exp["total_spend"])
print(exp["by_category"])  # {"meals": 120.0, "travel": 450.0, ...}
```

### Resource Utilisation Heatmap

Returns a day-indexed dict for calendar heatmap widgets.

```python
heatmap = await svc.resource_utilisation_heatmap(
    resource_id="emp_101",
    year=2026,
    month=6,
    capacity_hours_per_day=8.0,
)
for day, data in heatmap["heatmap"].items():
    print(day, data["utilisation_pct"])
```

### Expense Spend Forecast

Extrapolates from a 3-month rolling average to estimate upcoming spend.

```python
forecast = await svc.forecast_expense_spend(
    project_id="proj_alpha",
    lookahead_days=30,
)
print(forecast["total_exposure"])
print(forecast["budget_risk_level"])  # low | medium | high
```

### Employee Expense Summary

```python
summary = await svc.employee_expense_summary(
    employee_id="emp_101",
    period="2026-06",
)
print(summary["by_status"])    # {"approved": 320.0, "submitted": 42.0}
print(summary["by_category"])  # {"meals": 120.0, "software": 242.0}
```

---

## Compliance

```python
compliance = await svc.tex_compliance_check()
print(compliance["compliance_rate_pct"])    # 95.0
print(compliance["missing_receipt_count"])  # 2
```

---

## Audit Log Queries

```python
# All approval events
events = await svc.query_audit_log(event_type="timesheet_approved", limit=50)

# All events for a specific timesheet
events = await svc.query_audit_log(reference_id="ts_emp101_proj_alpha_2026-06")
```

---

## Payroll Lock

After a payroll run consumes a timesheet, lock it to prevent further mutations.

```python
lock = await svc.lock_timesheet_for_payroll(
    timesheet_id="ts_emp101_proj_alpha_2026-06",
    payroll_run_id="payroll_2026_06",
)
print(lock["locked_at"])
```

Subsequent approval or amendment attempts on a locked timesheet raise `PermissionError`.

---

## Project Accounting Export

Export approved timesheets and expense claims to `ppm_pac`. Labour cost is computed by applying billing rates to billable hours.

```python
batch = await svc.export_to_project_accounting(
    project_id="proj_alpha",
    period="2026-06",
    pac_adapter=None,  # pass a real pac_adapter to call ingest_cost_batch
)
print(batch["total_labour_cost"])
print(batch["total_expense_cost"])
print(batch["grand_total"])
```

When `pac_adapter` is provided, `pac_adapter.ingest_cost_batch(batch)` is awaited automatically.

---

## Billing Rates

```python
svc.set_billing_rate(
    rate_id="rate_emp101_proj_alpha",
    tenant_id="acme",
    resource_id="emp_101",
    project_id="proj_alpha",
    rate_type="standard",
    rate_amount=120.0,
    currency="USD",
    effective_date="2026-01-01",
    approval_reference="cfo_approved",
)
```

---

## UI Routes

| Path | Permission | Description |
|------|-----------|-------------|
| `/ppm-tex/dashboard` | `ppm_tex:view` | Summary dashboard |
| `/ppm-tex/timesheets/my` | `ppm_tex:timesheets` | Personal timesheets |
| `/ppm-tex/timesheets/entry` | `ppm_tex:timesheets` | Submit time entry |
| `/ppm-tex/timesheets/approvals` | `ppm_tex:approve_timesheets` | Approval queue |
| `/ppm-tex/expenses/my` | `ppm_tex:expenses` | Personal expense list |
| `/ppm-tex/expenses/claim` | `ppm_tex:expenses` | Submit expense |
| `/ppm-tex/expenses/approvals` | `ppm_tex:approve_expenses` | Expense approval queue |
| `/ppm-tex/billable` | `ppm_tex:billing` | Billable hour tracker |
| `/ppm-tex/rates` | `ppm_tex:rates` | Billing rate table |
| `/ppm-tex/reimbursements` | `ppm_tex:reimburse` | Reimbursement console |

---

## Policy Enforcement

All operations run through the capability rule engine. Key rules:

| Rule | Trigger | Effect |
|------|---------|--------|
| `expense_above_threshold_requires_receipt` | amount > $25 + no receipt | deny |
| `duplicate_expense_denied` | same resource/date/amount/category | deny |
| `backdated_entry_requires_justification` | backdated=True + empty justification | deny |
| `time_entry_hours_must_be_positive` | hours ≤ 0 | deny |
| `reimbursement_approval_required` | missing approval_reference | deny |
| `timesheet_locked_for_payroll` | amend/approve on locked timesheet | deny |
| `tex_batch_requires_bytewax` | event_stream != "bytewax" | deny |

---

## Composability

```apg
use ppm_tex;
```

- Billable hours + expenses → **ppm_pac** via `export_to_project_accounting`
- Billing rates ↔ **ppm_res** cost rate catalogue
- Approved timesheets → **ppm_pps** progress tracking
- Utilisation heatmap → **ppm_res** capacity planning

---

## Further Reading

- `service.py` — Full business logic implementation
- `models.py` — Dataclass models (Timesheet, TimeEntry, ExpenseClaim, etc.)
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 planned enhancements with rationale
- `README.md` — Quick API reference
- `cap_spec.md` — Capability specification
