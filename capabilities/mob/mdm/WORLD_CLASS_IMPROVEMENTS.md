# Mobile Device Management — World-Class Improvements

**Capability**: `mob_mdm` | **Version**: 1.0.0 → 2.0.0 | **Author**: Nyimbi Odero | **© 2025 Datacraft**

---

## 1. Persistent Storage Backend

**Current state**: All state lives in in-process dicts — wiped on restart.

**Improvement**: Replace `dict[tuple, Model]` stores with async SQLAlchemy (asyncpg + PostgreSQL). Add `database/store.py` repository layer with connection pooling, row-level locking (`SELECT ... FOR UPDATE`) for wipe execution, and alembic migrations already scaffolded. Gives crash-safe persistence and horizontal scaling.

**Impact**: Production readiness. Required before any real deployment.

---

## 2. Streaming Event Bus Integration

**Current state**: `_audit_events` is a plain `list[dict]` — no external consumers.

**Improvement**: Emit typed CloudEvents to a Bytewax/Kafka topic via `mqeb` on every state transition (`device_enrolled`, `compliance_evaluated`, `wipe_completed`, etc.). Include schema registry validation. Consumers: `moni` (fleet health), `ntfy` (escalation), `comp` (regulatory framework).

**Impact**: Real-time dashboards, cross-capability orchestration, and decoupled alert pipelines.

---

## 3. Differential Compliance Engine

**Current state**: Single-shot evaluation — returns a binary compliant/non_compliant with raw findings list.

**Improvement**: Introduce a `ComplianceRule` registry (JSON-configurable per tenant). Each rule has `id`, `severity`, `condition_expr` (CEL/JSONPath), `remediation_hint`, and `grace_period_hours`. Evaluation returns diffs against the previous record — only changed findings trigger alerts, eliminating alert fatigue. Add `auto_remediate` flag for self-healing rules (e.g., push a profile if WiFi config is missing).

**Impact**: 80% reduction in spurious alerts; actionable remediation hints in every finding.

---

## 4. Device Health Score

**Current state**: Compliance is binary; no holistic device health signal.

**Improvement**: Compute a `health_score` (0–100) for each device as a weighted aggregate of: OS patch level, compliance state, last-seen recency, certificate expiry, encryption status, and jailbreak/root detection. Expose via `get_device_health_score()` and surface on the dashboard KPI card. Configurable weights per tenant.

**Impact**: Single number operations teams can SLA on; replaces 5-minute manual triage.

---

## 5. Bulk Enrolment with CSV Import

**Current state**: One device enrolled per API call.

**Improvement**: Add `bulk_enrol_devices(tenant_id, csv_bytes, created_by)` that parses a validated CSV (serial, type, platform, ownership), runs each enrolment in a `asyncio.TaskGroup`, collects per-row success/failure, and returns a `BulkEnrolmentResult` with counts and error details. Idempotent via serial number deduplication.

**Impact**: IT admins can onboard 1 000-device fleet in a single operation.

---

## 6. Certificate Lifecycle Management

**Current state**: `certificate_push` is a fire-and-forget stub returning a fingerprint prefix.

**Improvement**: Implement full certificate lifecycle: issue (SCEP/ACME), install, track expiry, auto-renew 30 days before expiry via a scheduled background task, revoke (OCSP/CRL). Add `CertificateRecord` model with `serial_number`, `issuer`, `not_before`, `not_after`, `status`. Emit `certificate_expiring_soon` alerts at 30 and 7 days.

**Impact**: Eliminates manual certificate rotation; prevents VPN/WiFi outages from cert expiry.

---

## 7. Geofencing and Location Policy Enforcement

**Current state**: `location_track` is a stub returning `null` coordinates.

**Improvement**: Integrate with GPS provider (structured adapter interface). Add `GeoFence` model (polygon/radius, `allowed` or `denied` semantics). On location update, evaluate all active geofences for the device and trigger `geofence_violation` alerts if outside allowed zones. Support `lost_mode` activation on geofence exit.

**Impact**: Physical security enforcement for regulated environments (healthcare, finance).

---

## 8. Application Inventory and Vulnerability Scanning

**Current state**: `app_blacklist_check` only checks against a caller-supplied list; no persistent app inventory.

**Improvement**: Maintain a per-device `InstalledApp` registry (bundle_id, version, install_date). On sync, cross-reference against a vulnerability feed (CVE lookup via NVD API adapter). Auto-raise `critical` severity alerts for CVE-scored apps. Add `get_app_inventory(device_id)` and `scan_app_vulnerabilities(device_id)` methods.

**Impact**: Proactive CVE exposure visibility without manual security reviews.

---

## 9. Policy Inheritance and Group Targeting

**Current state**: Policies are assigned one-to-one to individual devices.

**Improvement**: Add `DeviceGroup` model (dynamic query-based or static membership). Policies can target groups; assignment propagates to all current and future group members. Implement inheritance: group policy + device-level override with last-write-wins or priority-order semantics. Add `create_device_group`, `add_device_to_group`, `assign_policy_to_group`.

**Impact**: Manage 10 000 devices with 10 policies instead of 10 000 assignments.

---

## 10. Wipe Cancellation and Rollback Window

**Current state**: Pending wipes can only be executed — no cancellation path.

**Improvement**: Add `cancel_wipe(tenant_id, wipe_id, cancelled_by, reason)` that transitions `pending → cancelled` within a configurable rollback window (default 15 minutes). After window expiry, wipe auto-executes. Add `WipeRollbackWindow` audit entry. Require the same dual-approval chain for cancellation.

**Impact**: Safety net against accidental wipe requests; critical for regulated data handling.

---

## 11. RBAC Permission Enforcement at Service Layer

**Current state**: `_enforce` only evaluates `capability_contract` rules — no identity/role awareness.

**Improvement**: Thread a `caller_principal: str` and `caller_roles: list[str]` through all mutating methods. Check permissions against a `RolePermissionMap` (configurable per tenant). Raise `PermissionDenied` (not generic `ValueError`) with the specific missing permission. Log denied attempts to the audit trail.

**Impact**: Least-privilege enforcement; audit trail distinguishes authorised vs. blocked attempts.

---

## 12. Idempotent Operations via Request IDs

**Current state**: Duplicate API calls create duplicate records (e.g., two `enrol_device` calls with same serial).

**Improvement**: Accept optional `idempotency_key: str` on all Create payloads. Cache `(tenant_id, idempotency_key) → response` in a short-TTL store (Redis via `conf`). Return cached response on replay. Add serial-number uniqueness check in `enrol_device`.

**Impact**: Safe retries — eliminates phantom duplicate devices from flaky clients.

---

## 13. Scheduled Compliance Re-evaluation

**Current state**: `next_evaluation_at` is stored but nothing acts on it.

**Improvement**: Add a background scheduler task (APScheduler or Celery beat) that queries devices with `next_evaluation_at <= now` and runs `evaluate_compliance` with a system evaluator ID. Configurable interval per tenant (`compliance.evaluation_interval_minutes`). Emit `compliance_check_scheduled` events.

**Impact**: Continuous compliance posture without ops manual triggering.

---

## 14. MDM Protocol Adapters (Apple MDM / Android Enterprise)

**Current state**: Service is protocol-agnostic in-process only; no real device communication.

**Improvement**: Define an `MdmProtocolAdapter` ABC with `send_command`, `fetch_device_info`, `push_profile`, `initiate_wipe`. Provide `AppleMdmAdapter` (APNs + MDM protocol) and `AndroidEnterpriseAdapter` (FCM + EMM API) implementations. `MobileDeviceManagementService` takes adapter as a constructor dep-injection.

**Impact**: Real device control; adapters can be mocked in tests, swapped in production.

---

## 15. Comprehensive Observability and Structured Logging

**Current state**: `_audit_events` is a list of raw dicts; no structured logging, no metrics.

**Improvement**: Replace `_audit` with structured log emission via `structlog` (JSON format). Add Prometheus metrics: `mdm_devices_enrolled_total`, `mdm_compliance_evaluations_total{state}`, `mdm_wipe_requests_total{type}`, `mdm_alert_severity_total{severity}`. Expose `/metrics` endpoint and a health check at `/health` returning service version and db connectivity status.

**Impact**: Plug into any observability stack (Grafana, Datadog, ELK) with zero integration effort.
