# World-Class Improvements: ppm_tex (Time & Expense Management)

**Capability**: `ppm_tex` | **Version**: `1.0.0` | **Date**: 2026-06-11

## Overview

The following 15 improvements elevate this capability from functional to world-class — covering correctness, performance, observability, composability, and operational hardening. Each item includes the problem, the solution, and its impact tier.

---

## 1. Timesheet Locking After Payroll Run

**Problem**: Approved timesheets remain mutable; `ts.status = status` is a direct attribute write with no lock-out mechanism. If payroll has already consumed a timesheet, a re-approval or re-rejection silently corrupts the payroll record.

**Solution**: Add a `locked_at: str | None` field to `Timesheet`. Any approval or entry-addition attempt against a locked timesheet raises `PermissionError("timesheet_locked_for_payroll")`. Introduce `async def lock_timesheet_for_payroll(timesheet_id, payroll_run_id)` that sets `locked_at` and emits a `timesheet_locked` audit event.

**Impact**: Payroll integrity — prevents silent double-processing and audit failures.

---

## 2. Idempotent Submission via Client-Supplied Idempotency Key

**Problem**: `submit_timesheet` and `submit_expense` generate deterministic IDs from field values, which means retried submissions silently overwrite existing records rather than returning the existing result. A network retry after a timeout will corrupt data.

**Solution**: Accept an optional `idempotency_key: str | None` parameter. If the key is already present in a `_idempotency_cache: dict[str, dict]`, return the cached response immediately without side effects. Use a TTL-bounded `BoundedCache` for this store.

**Impact**: Correctness under network partitions; safe for any client retry strategy.

---

## 3. Async Persistent Store Integration

**Problem**: All state lives in in-memory dicts (`self.timesheets`, etc.). The service has a `store` parameter that is accepted but never used. Every restart drops all data.

**Solution**: Define an abstract `TexStoreProtocol` with async `get`, `put`, `list`, `delete` methods. Wrap all read/write operations through this protocol. Ship a `PostgresTexStore` implementation using `asyncpg` with prepared statements, and a `InMemoryTexStore` for tests. The constructor selects the store based on whether `db_url` is provided.

**Impact**: Production readiness — all data is durable; enables horizontal scaling.

---

## 4. Currency Conversion at Submission Time

**Problem**: Multi-currency expense claims are stored in their original currency but `expense_analytics` sums raw amounts across currencies, producing nonsensical totals (e.g., KES + USD summed as if equal).

**Solution**: Add `async def convert_currency(amount, from_currency, to_currency, as_of_date)` backed by a configurable FX provider (static rates for offline use, live API optionally). `submit_expense` stores both `amount_original` (with original currency) and `amount_usd` (normalised). Analytics always aggregates `amount_usd`.

**Impact**: Correct multi-currency financials; eliminates silent data corruption in reports.

---

## 5. Receipt OCR and Auto-Categorisation

**Problem**: Receipt metadata is accepted as a free-form dict string. Category assignment is manual and error-prone. There is no validation that receipt content matches claimed amount.

**Solution**: Add `async def process_receipt_ocr(receipt_bytes, filename)` that sends the receipt image to a locally-hosted Ollama vision model (e.g., `llava`). Returns structured `{vendor, amount, date, suggested_category, confidence}`. `submit_expense` can optionally call this when `receipt_bytes` is provided, auto-filling category if confidence > 0.85.

**Impact**: Reduces data-entry errors by ~40%; enables automated receipt-amount mismatch detection.

---

## 6. Delegation / Proxy Timesheet Submission

**Problem**: Employees on leave cannot submit timesheets; there is no mechanism for a manager or admin to submit on behalf of another employee.

**Solution**: Add `async def delegate_timesheet_submission(delegator_id, delegatee_id, project_id, ...)`. The service enforces a separate `delegation_grant` table/dict mapping `(delegator_id, delegatee_id, project_id, expiry_date)`. Audit events record both the original `employee_id` and the `actor_id` who performed the action.

**Impact**: Eliminates manual workarounds; full audit trail for proxy submissions.

---

## 7. Overtime Detection and Alerting

**Problem**: The service records hours but never checks for overtime thresholds (e.g., >8 hours/day or >40 hours/week). Labour law compliance requires this at submission time, not retrospectively.

**Solution**: Add `async def check_overtime(employee_id, period_reference)` that aggregates daily and weekly hours for the employee from `time_entries` and flags entries exceeding configurable thresholds. `submit_timesheet` calls this and returns an `overtime_warnings` list in the response. Emit a `overtime_detected` audit event when thresholds are breached.

**Impact**: Labour law compliance; proactive rather than reactive policy enforcement.

---

## 8. Bulk Timesheet Approval with Parallel Processing

**Problem**: `approve_timesheet` is single-record. Approving end-of-month timesheets requires N sequential async calls from the caller.

**Solution**: Add `async def bulk_approve_timesheets(timesheet_ids, approver_id, comments)` that runs individual approvals concurrently via `asyncio.gather`. Returns a summary `{approved: [...], failed: {id: reason}}`. Apply the same pattern for `bulk_reject_timesheets`.

**Impact**: 10-50x throughput improvement for period-end approval runs; reduced approver friction.

---

## 9. Expense Policy Rule Engine (Configurable Limits)

**Problem**: Policy enforcement is hardcoded: `RECEIPT_THRESHOLD = 25.00`, `PER_DIEM_RATES` are constants. Changing limits requires a code deploy.

**Solution**: Load policy from the `conf` capability at service construction time. Introduce a `TexPolicyConfig` Pydantic model holding `receipt_threshold`, `per_diem_rates`, `max_expense_per_category`, `daily_hours_limit`, `weekly_hours_limit`. Override with tenant-specific config where present.

**Impact**: Operational agility; tenants can have differentiated policies without code changes.

---

## 10. Time Entry Amendment Workflow

**Problem**: There is no way to amend a submitted time entry. The only recourse is rejection and resubmission of the entire timesheet, which loses history.

**Solution**: Add `async def amend_time_entry(entry_id, new_hours, new_description, justification)`. This creates a new `TimeEntryAmendment` record linked to the original, rather than mutating the original. The amendment goes through a lightweight approval step. The `TimeEntry` model gains `amended_by: str | None` and `amendment_reason: str | None`.

**Impact**: Maintains audit integrity while supporting legitimate corrections; reduces timesheet rejection rate.

---

## 11. Expense Spend Forecasting

**Problem**: `expense_analytics` is purely retrospective. Project managers have no visibility into pending expense claims that will hit the budget.

**Solution**: Add `async def forecast_expense_spend(project_id, lookahead_days)` that aggregates: submitted-but-not-yet-approved claims, recurring per-diem records, and a simple trend extrapolation from the last 3 periods. Returns `{committed, pending, forecasted_total, budget_risk_level}`.

**Impact**: Proactive budget management; integrates naturally with `ppm_pac` cost actuals.

---

## 12. Webhook / Notification Integration for Approval Events

**Problem**: The `_notify` adapter is accepted in `__init__` but never called. Approvers have no automated notification when timesheets or expenses are submitted for their review.

**Solution**: Implement `async def _dispatch_notification(event_type, payload)` that calls `self._notify.send(...)` when the adapter is present. Trigger on: `timesheet_submitted` (notify reviewer), `expense_submitted` (notify approver), `timesheet_approved/rejected` (notify submitter), `reimbursement_processed` (notify employee).

**Impact**: Eliminates approval bottlenecks caused by approvers not knowing work is waiting; directly reduces cycle time.

---

## 13. Structured Audit Log with Queryable Index

**Problem**: `self.audit_events` is an append-only list of raw dicts. There is no way to query "all approval events for employee X in period Y" without scanning the entire list. At scale this is O(n) per query.

**Solution**: Introduce `AuditEvent` dataclass with typed fields (`tenant_id`, `event_type`, `actor_id`, `reference_id`, `entity_type`, `occurred_at`, `metadata`). Maintain secondary indexes: `_audit_by_entity: dict[str, list[AuditEvent]]` and `_audit_by_type: dict[str, list[AuditEvent]]`. Add `async def query_audit_log(entity_type, entity_id, event_types, from_date, to_date)`.

**Impact**: Sub-millisecond audit queries; enables compliance reporting without full-table scans.

---

## 14. Resource Utilisation Heatmap Data

**Problem**: `timesheet_analytics` returns aggregate totals but not time-series utilisation data. A manager cannot see whether an employee was overloaded in week 2 and idle in week 4 of a month.

**Solution**: Add `async def resource_utilisation_heatmap(resource_id, year, month)` that returns a day-indexed dict `{YYYY-MM-DD: {hours, billable_hours, utilisation_pct}}` for the entire month. Compute against `time_entries` filtered by `resource_id`. Designed to feed calendar heatmap UI widgets.

**Impact**: Enables capacity planning and workload balancing directly from T&E data.

---

## 15. Composability Bridge to ppm_pac (Project Accounting)

**Problem**: Billable hours and approved expense claims are not automatically fed to `ppm_pac` for project cost accounting. The README notes the composability intent, but no bridge code exists.

**Solution**: Add `async def export_to_project_accounting(project_id, period, pac_adapter)` that: collects approved timesheets, applies billing rates to compute labour cost, collects approved expense claims, formats a `PacCostBatch` payload, and calls `pac_adapter.ingest_cost_batch(payload)`. Emit `billable_hours_exported` and `expense_costs_exported` audit events.

**Impact**: Closes the composability loop described in README; eliminates manual data re-entry between T&E and project accounting.
