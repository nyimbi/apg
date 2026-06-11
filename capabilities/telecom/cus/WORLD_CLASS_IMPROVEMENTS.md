# Customer Management — World-Class Improvements

**Capability**: `telecom_cus` | **Path**: `capabilities/telecom/cus`
**Author**: Nyimbi Odero | **Company**: Datacraft | **Copyright**: © 2025

---

## 1. Customer 360 Unified Profile

**Current gap**: No single call assembles the full customer view.
**Improvement**: `get_customer_360(customer_id, tenant_id)` joins CusCustomer, all
CusKycDocument, active CusPlan, CusSim list, CusDevice list, open CusCase list,
last 10 CusLifecycleEvent, latest NPS score, and latest churn intervention into a
single response.  Eliminates N+1 calls from API consumers.
**Implemented in**: `service.py::get_customer_360`

---

## 2. SIM Swap Workflow with Fraud Safeguard

**Current gap**: `update_sim_status` is a raw field mutation with no fraud gate.
**Improvement**: `sim_swap(old_sim_id, new_sim_id, new_iccid, new_imsi, msisdn, reason,
tenant_id)` validates the old SIM is active, provisions the replacement, checks for a
30-day cooling-off window, and auto-escalates cooling-off violations to a
`fraud_report` case type.  Fires a `sim_swapped` lifecycle event unconditionally.
**Implemented in**: `service.py::sim_swap`

---

## 3. SLA Breach Monitoring

**Current gap**: SLA due dates are computed at `complaint_log` time but never checked.
**Improvement**: `get_sla_breaches(tenant_id)` scans all open/in-progress cases,
compares `sla_due_at` against current UTC, returns breached and at-risk cases (within
2 hours of breach), and emits a `sla_breach_detected` audit event per breach.
**Implemented in**: `service.py::get_sla_breaches`

---

## 4. Number Portability Request Workflow

**Current gap**: Number porting is not modelled.
**Improvement**: `request_number_port(customer_id, msisdn, donor_operator, tenant_id)`
creates a `portability_request` case, assigns a port reference, sets a 5-business-day
SLA, and emits a `number_ported` lifecycle event on completion.  Validates the MSISDN
belongs to the customer before accepting.
**Target**: `service.py::request_number_port` (pending implementation)

---

## 5. Bulk Customer Import with Idempotency

**Current gap**: `bulk_create` is a stub returning a count with no actual record
creation or deduplication.
**Improvement**: `bulk_import_customers(records, tenant_id, dry_run)` validates every
record, deduplicates by MSISDN, creates all valid customers, returns per-record
`created | skipped | failed` status, and supports `dry_run=True` for pre-flight
validation without side effects.
**Implemented in**: `service.py::bulk_import_customers`

---

## 6. Churn Probability Scoring Pipeline

**Current gap**: `churn_intervention` accepts an externally supplied probability with
no internal scoring.
**Improvement**: `score_churn_risk(customer_id, tenant_id)` computes a deterministic
0.0–1.0 risk score from: open complaint count, NPS category, days since last plan
activation, SIM swap frequency (90-day window), and plan type.  Emits a
`churn_risk_flagged` lifecycle event when score ≥ 0.65.
**Implemented in**: `service.py::score_churn_risk`

---

## 7. GDPR / POPIA Right-to-Erasure

**Current gap**: No data subject rights workflow.
**Improvement**: `request_data_erasure(customer_id, reason, requested_by, tenant_id)`
pseudonymises PII fields (name, msisdn) in all associated records, marks
`kyc_status = "erased"`, opens a `service_request` case with a 30-day compliance SLA,
and emits a `data_erasure_requested` lifecycle event.  Audit trail is preserved with
anonymised references.
**Implemented in**: `service.py::request_data_erasure`

---

## 8. Automated Dunning Workflow

**Current gap**: No payment-failure / dunning lifecycle.
**Improvement**: `trigger_dunning(customer_id, invoice_id, amount_due, days_overdue,
tenant_id)` selects the dunning step (reminder → warning → soft suspension →
deactivation) based on `days_overdue`, opens a `billing_query` case with escalation
notes, suspends service when overdue > 14 days, and emits appropriate lifecycle events.
Integrates with `telecom_bil` via event stream.
**Implemented in**: `service.py::trigger_dunning`

---

## 9. Case Escalation Engine

**Current gap**: Cases have an `escalated` status but no escalation logic.
**Improvement**: `escalate_case(case_id, escalation_reason, escalated_to_tier,
tenant_id)` validates the case is open or in_progress, records the escalation path,
assigns to the next-tier agent pool, resets SLA clock to 4 hours, and fires a
`case_escalated` audit event.  Supports tier_1 → tier_2 → specialist → management.
**Implemented in**: `service.py::escalate_case`

---

## 10. Customer Segmentation API

**Current gap**: No programmatic segmentation.
**Improvement**: `segment_customers(criteria, tenant_id, page, page_size)` accepts a
criteria dict (status, kyc_status, customer_type, plan_type, churn_risk_min,
churn_risk_max) and returns a paginated filtered list with total match count.  Used by
`telecom_ana` for campaign targeting and cohort analysis.
**Implemented in**: `service.py::segment_customers`

---

## 11. Idempotent Event Deduplication

**Current gap**: `_audit` appends unconditionally — replaying the same operation
produces duplicate audit events.
**Improvement**: Derive an `event_id` from `(tenant_id, event_type, reference_id,
minute_bucket)`.  The audit store checks for an existing record with the same
`event_id` before inserting, making every write operation safe to retry under
at-least-once delivery semantics.
**Target**: `service.py::_audit` (pending deduplication gate)

---

## 12. Structured Pydantic v2 Responses

**Current gap**: All methods return raw `dict[str, Any]`, losing type safety at
boundaries.
**Improvement**: Replace `dict[str, Any]` return types with Pydantic v2 response
models (`CusCustomerResponse`, `CusCaseResponse`, `KycCheckResult`) in `views.py`.
Service methods return model instances; the API layer calls `.model_dump(mode="json")`.
Enables OpenAPI schema generation and compile-time contract checking.
**Target**: `views.py` (pending Pydantic v2 response model definitions)

---

## 13. Async Database Persistence

**Current gap**: In-memory `dict` stores lose all data on process restart.
**Improvement**: Replace `self.customers: dict` with async calls to `database/store.py`
backed by an asyncpg connection pool.  `DatabaseStore` uses
`INSERT ... ON CONFLICT DO UPDATE` for upserts.  All service methods become fully
async.  State survives restarts and scales horizontally.
**Target**: `database/store.py` (pending asyncpg integration)

---

## 14. OpenTelemetry Tracing

**Current gap**: Audit events are append-only dicts with no distributed trace context.
**Improvement**: Instrument every public async method with
`opentelemetry.trace.get_tracer("telecom_cus")` spans.  Include `customer_id`,
`tenant_id`, and `operation` as span attributes.  Propagate `trace_id` into audit
events.  Enables end-to-end request tracing surfaced in Grafana Tempo.
**Target**: `service.py` (pending OTel instrumentation)

---

## 15. Tenant-Scoped Rate Limiting and Quota Enforcement

**Current gap**: No protection against runaway bulk operations or API abuse.
**Improvement**: `_check_quota(tenant_id, operation, count)` enforces configurable
per-tenant limits (max 500 customer creates/hour, max 50 bulk-import records/request).
Uses a sliding-window counter stored in Redis (or in-process `BoundedCache` for
single-node).  Returns `QuotaExceededError` with retry-after seconds.  Prevents
noisy-neighbour issues in multi-tenant deployments.
**Target**: `service.py::_check_quota` (pending Redis/BoundedCache integration)
