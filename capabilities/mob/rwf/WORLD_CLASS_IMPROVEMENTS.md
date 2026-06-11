# World-Class Improvements — Remote Workforce (mob_rwf)

## Summary

15 high-impact improvements to elevate mob_rwf from a governance runtime to a
production-grade field-worker platform with offline-first resilience, intelligent
scheduling, and deep operational observability.

---

## 1. Offline-First Sync Queue with Conflict Resolution

**Current gap**: All state lives in process memory; field workers with intermittent
connectivity have no path to record activity offline and sync later.

**Improvement**: Introduce `OfflineSyncQueue` — a append-only log of operations
stamped with a logical clock (Lamport/HLC). On reconnect, operations are replayed
in causal order; last-write-wins per field (with merge semantics for lists) resolves
conflicts. Expose `sync_offline_queue()` and `get_sync_status()` methods.

---

## 2. Geofence-Aware Task Assignment

**Current gap**: Tasks and equipment requisitions carry no spatial context; a worker
in Mombasa receives the same assignment queue as one in Nairobi.

**Improvement**: Attach a `GeoRegion` (lat/lon bounding box or named region) to each
task and worker profile. `assign_field_tasks()` filters candidates by geofence and
travel-time estimate. Integration point: `mob_map` GPS tracking capability.

---

## 3. Dynamic Route Optimisation (TSP via Nearest-Neighbour + 2-opt)

**Current gap**: `ml_field_worker_route_optimize` is a stub that delegates to a
missing `MLCapability`; it returns `{"ml_enhanced": False}` in all real environments.

**Improvement**: Implement a deterministic nearest-neighbour + 2-opt TSP solver as
the default path. Ollama-based scoring becomes an optional enhancement layer, not
the primary path. Expose `optimize_field_route(waypoints)` returning an ordered
itinerary with estimated drive times.

---

## 4. Task Dependency Graph with Critical-Path Scheduling

**Current gap**: Onboarding steps are a flat list; complex field missions with
prerequisite tasks are not modelled.

**Improvement**: Represent tasks as a DAG (`task_id → [dependency_ids]`). The
scheduler computes the critical path, surfaces blockers, and auto-gates tasks
whose dependencies are incomplete. Add `create_task_graph()`, `get_critical_path()`,
and `advance_task_graph()` methods.

---

## 5. Role-Based Field Certifications and Skill Matching

**Current gap**: Equipment requisitions and onboarding steps carry no skill/cert
requirements. A worker without a forklift licence can be assigned forklift tasks.

**Improvement**: Model `FieldCertification` (cert_type, expiry_date, issuer). Task
assignment checks certification validity. Expired certs raise a compliance incident
automatically. Add `record_certification()`, `list_certifications()`, and
`check_certification_validity()`.

---

## 6. Incident Escalation Engine with SLA Timers

**Current gap**: Incidents are raised and resolved manually; there is no escalation
path if an incident sits open past its SLA.

**Improvement**: Attach an `sla_minutes` field to each incident. A background
coroutine (`_run_sla_monitor()`) polls open incidents and emits `incident_escalated`
audit events when SLA breaches. Add `escalate_incident()` and
`get_incidents_breaching_sla()`.

---

## 7. Attendance and Shift Check-In / Check-Out

**Current gap**: No way to record when a field worker starts or ends a shift, which
is essential for payroll, compliance, and productivity attribution.

**Improvement**: Add `FieldShift` model with `check_in()`, `check_out()`,
`get_active_shifts()`, and `shift_summary()`. GPS coordinates captured at
check-in/out enable geo-verification of attendance.

---

## 8. Push Notification Fanout with Priority Routing

**Current gap**: The `ntfy` dependency is listed but never called; incidents and
compliance alerts are silent.

**Improvement**: Add `notify_field_worker()` that dispatches to the `ntfy`
capability with priority levels (`critical`, `high`, `normal`, `low`). Critical
messages (e.g., safety incidents) bypass do-not-disturb windows. Batch lower-priority
notifications into digest messages.

---

## 9. Configurable Compliance Check Cadence per Check Type

**Current gap**: All compliance checks use a hard-coded 30-day next-due interval.
Security posture checks may need daily; background verification checks are annual.

**Improvement**: Replace the constant with a `_COMPLIANCE_CADENCE_DAYS` mapping
keyed on `check_type`. Expose `set_compliance_cadence()` and
`get_overdue_compliance_checks()` for proactive monitoring.

---

## 10. Equipment Lifecycle Tracking (Maintenance, Depreciation, Loss)

**Current gap**: Equipment state machine stops at `returned`; no record of
maintenance events, depreciation schedules, or loss reports.

**Improvement**: Extend `EquipmentRequisitionResponse` with `maintenance_history`
and `asset_value`. Add `record_equipment_maintenance()`, `report_equipment_lost()`,
and `calculate_equipment_depreciation()`. Integrates with `fint` financial
capability for asset write-downs.

---

## 11. Tenant-Scoped Rate Limiting and Quota Enforcement

**Current gap**: No throttle on how many VPN provisions, equipment requests, or
compliance checks a tenant can submit in a time window.

**Improvement**: Introduce `TenantRateLimiter` using a sliding-window token-bucket
per `(tenant_id, operation)`. Exceed quota → `QuotaExceededError` with
retry-after semantics. Expose `get_quota_status()` for dashboard visibility.

---

## 12. Structured Audit Log Export (SIEM Integration)

**Current gap**: Audit events are stored as raw dicts in process memory with no
export path; compliance auditors cannot extract them.

**Improvement**: Add `export_audit_log()` returning CEF/JSON-L formatted events
filterable by time range, tenant, and event type. Support streaming export via
async generator for large log volumes. Include tamper-evident checksums per batch.

---

## 13. Bulk Operation API with Partial-Failure Semantics

**Current gap**: All mutation methods are single-entity; HR onboarding 200 new
starters requires 200 sequential API calls.

**Improvement**: Add `bulk_start_onboarding()`, `bulk_request_equipment()`, and
`bulk_record_compliance_checks()`. Each returns a `BulkOperationResult` with
`succeeded`, `failed`, and `partial_errors` lists. Operations are independent;
one failure does not roll back the batch.

---

## 14. Time-Zone-Aware Scheduling for Distributed Teams

**Current gap**: All timestamps use `datetime.utcnow()` with no time-zone
context; workers in multiple time zones see wall-clock gaps in productivity
windows and compliance schedules.

**Improvement**: Replace `datetime.utcnow()` with `datetime.now(tz=timezone.utc)`
throughout. Attach `timezone` to `WorkPolicy` effective/expiry dates and evaluate
schedule windows in the worker's local timezone. Add `get_team_timezone_summary()`.

---

## 15. Capability Health Check and Self-Test Endpoint

**Current gap**: No way to verify the service is healthy, all dependencies are
reachable, and domain rules are internally consistent without running a full
test suite.

**Improvement**: Add `health_check()` returning a structured `HealthReport` with
sub-checks: rule engine, storage backend, dependency capabilities (auth, audl, ntfy),
and schema version. Returns HTTP 200/503 depending on overall status. Used by
Kubernetes liveness/readiness probes.
