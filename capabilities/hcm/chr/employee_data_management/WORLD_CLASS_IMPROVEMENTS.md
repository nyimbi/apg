# World-Class Improvements — chr_employee_data_management

© 2025 Datacraft | Author: Nyimbi Odero

These 15 improvements elevate the capability from a functional in-memory service
to production-grade, composable, observable HR infrastructure.

---

## 1. Async-First Service Layer

**Status**: Partially done — `ml_attrition_predict` is async; all other methods
are sync. Adopting async throughout removes blocking I/O during DB and HTTP
adapter calls and lets the service compose cleanly in asyncio event loops used
by FastAPI, Starlette, and LangGraph agents.

**Action**: Convert every public method to `async def`. Keep sync shims only
where callers cannot tolerate async (CLI entry points, WSGI blueprints).

---

## 2. Pydantic v2 Input/Output Models

**Status**: All inputs and outputs are plain `dict[str, Any]`. No validation
enforced at the boundary — invalid emails, negative salaries, and future hire
dates only fail deep inside logic.

**Action**: Introduce Pydantic v2 models in `views.py` for every request and
response object. Use `model_config = ConfigDict(extra='forbid')`. Validate at
the public method signature, not inside ad-hoc conditionals.

---

## 3. Pluggable Persistence Adapter

**Status**: All state lives in in-memory dicts. Restarting the process drops
everything.

**Action**: Define an `EmployeeStore` protocol with async CRUD methods.
Provide an `InMemoryEmployeeStore` (current behaviour, tests) and a
`PostgresEmployeeStore` backed by asyncpg. Wire via constructor injection so
the service never depends on a concrete adapter.

---

## 4. Event-Driven Audit Bus

**Status**: `_audit_events` is a module-level list. There is no durable
delivery, replay, or fan-out.

**Action**: Replace `_emit` with an `EventBus` protocol. Provide an
`InMemoryEventBus` for tests and a `BytewaxEventBus` / `RedisStreamEventBus`
for production. Events should be CloudEvents-formatted and serialisable.

---

## 5. Structured Observability (OpenTelemetry)

**Status**: No traces, spans, or metrics exported. Hard to diagnose latency or
error rates in production.

**Action**: Wrap every public method with an OTEL span. Emit `employee.created`,
`paye.computed` counters and histograms. Provide a `MetricsCollector` shim that
is a no-op when OTEL is not configured.

---

## 6. Row-Level Multi-Tenancy Enforcement

**Status**: `tenant_id` filtering is applied manually in each method — easy to
miss in new methods.

**Action**: Centralise tenancy in the store adapter: `store.get(id, tenant_id)`
raises `TenantIsolationError` if the record's tenant does not match. Remove
per-method `tenant_id` checks and replace with a single guard decorator.

---

## 7. GDPR / Data-Residency Controls

**Status**: Personal information fields are stored as opaque dict. No
encryption, masking, or retention tagging.

**Action**: Add a `PIIField` wrapper that encrypts at rest using a per-tenant
AES-256 key. Tag every personal-info record with `retention_until`. Provide
`purge_expired_pii(tenant_id)` and `anonymise_employee(employee_id, tenant_id)`.

---

## 8. Position Vacancy Tracking

**Status**: Positions track `authorized_headcount` but no mechanism counts
filled vs. vacant slots.

**Action**: Add `compute_position_vacancies(tenant_id)` that compares
`authorized_headcount` to the number of active employees in each position.
Expose `vacant_positions` in `dashboard_summary`.

---

## 9. Leave Balance Engine

**Status**: `record_leave` stores leave records but never validates or reduces
a balance.

**Action**: Add `accrue_leave_balance` and `get_leave_balance` methods that
compute pro-rated annual entitlement from hire date, deduct approved leave, and
return the remaining balance. Enforce the Kenya Employment Act 21-day minimum.

---

## 10. Onboarding Workflow Orchestration

**Status**: There is no onboarding checklist or workflow state machine. New
hires are created but never guided through day-0 tasks.

**Action**: Add `create_onboarding_checklist(employee_id, template)` that
generates a sequence of tasks (IT setup, contract signing, orientation, etc.)
with due dates and owner assignments. Add `advance_onboarding_step` and
`onboarding_status` query methods.

---

## 11. Payroll Run Aggregation

**Status**: `compute_net_pay` operates on a single employee. Running payroll for
500 employees requires 500 individual calls with no transactional boundary.

**Action**: Add `run_payroll(tenant_id, period, employee_ids)` that executes
all pay computations in parallel (asyncio.gather), writes a payroll run record,
and returns a summary + per-employee payslip list. Include a dry-run mode.

---

## 12. Org Chart Flattened Search

**Status**: `org_chart_generate` returns the full hierarchy dict. Finding a
specific employee's chain of command requires recursive client-side traversal.

**Action**: Add `reporting_chain(employee_id, tenant_id)` that returns the
upward chain (employee → manager → … → CEO) and `span_of_control(manager_id,
tenant_id)` that returns total recursive headcount under a manager.

---

## 13. Headcount Budget vs. Actual

**Status**: `headcount_forecast` uses a simple linear model with no budget
baseline.

**Action**: Add `headcount_budget_vs_actual(tenant_id, budget: dict[str, int])`
that compares approved headcount budget (keyed by department) against actual
active headcount and returns over/under variance and utilisation rate per
department.

---

## 14. Contract Expiry Alerting

**Status**: Certifications track `expires_on` but there is no proactive
expiry scan. Contract employees are not tracked for end-date proximity.

**Action**: Add `scan_expiring_certifications(tenant_id, days_ahead)` and
`scan_expiring_contracts(tenant_id, days_ahead)` that return records due to
expire within the lookahead window, sortable by urgency. Feed output into the
existing data-quality issue workflow.

---

## 15. Self-Describing Semantic API

**Status**: `describe()` returns the raw capability contract dict. There is no
machine-readable schema consumers can introspect to discover endpoints and
models.

**Action**: Implement an OpenAPI 3.1 schema generator (`openapi_schema(tenant_id)`)
that derives paths, request/response schemas, and security requirements from the
existing method signatures and Pydantic models. Expose via `GET /schema` so
developer tooling and AI agents can discover the full API surface automatically.
