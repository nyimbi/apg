# Citizen Services Portal — World-Class Improvements

**Capability**: `government_csr` | **Domain**: `government` | **Author**: Nyimbi Odero
**© 2025 Datacraft** | www.datacraft.co.ke

---

## 1. Async-Native Service Layer

**Current state**: All methods are synchronous; `ml_citizen_service_score` is the sole async method.
**Improvement**: Convert the entire public API to `async def`. Long-running I/O (DB writes, external identity registry calls, M-Pesa STK push) currently blocks the event loop when used inside async frameworks (FastAPI, Starlette, Litestar). Adopt `asyncpg` + `SQLAlchemy 2.x async` sessions; use `asyncio.gather` for concurrent sub-tasks within a single request.
**Impact**: 3–10× throughput improvement under concurrent load; unlocks streaming SSE status updates to citizen browsers without thread-pool overhead.

---

## 2. Persistent Storage with PostgreSQL + Alembic Migrations

**Current state**: All state lives in in-memory Python dicts; the `alembic/` scaffold exists but is unpopulated.
**Improvement**: Implement `database/store.py` as a proper async repository layer backed by PostgreSQL. Define all tables as SQLAlchemy `DeclarativeBase` mapped classes mirroring `models.py` dataclasses, with composite unique constraints (`tenant_id`, entity PK), JSONB `metadata` columns for extensible payloads, and `updated_at` triggers. Wire Alembic autogenerate.
**Impact**: Survives process restart; enables cross-replica consistency; JSONB indexes make `service_search` a native SQL full-text query instead of Python iteration.

---

## 3. Event-Driven Architecture via Domain Events

**Current state**: `_audit()` writes a flat list; no external event emission.
**Improvement**: Formalise `domain/events.py` with typed `CitizenEvent` dataclasses (`ApplicationSubmitted`, `PaymentCompleted`, `DocumentVerified`, etc.). Emit events to a configurable broker (Kafka / NATS / Redis Streams) after every state mutation. Consumers drive notifications, analytics aggregation, and cross-capability triggers (CSR → CAS escalation) without coupling service methods.
**Impact**: Decouples notification, analytics, and external integration from business logic; enables full audit replay and temporal queries.

---

## 4. Idempotency Keys and Duplicate Submission Prevention

**Current state**: `submit_application` creates a new record on every call with no deduplication.
**Improvement**: Accept an `idempotency_key` parameter (SHA-256 of `tenant_id + citizen_id + service_id + period`). On duplicate key, return the existing application rather than creating a new one. Persist idempotency keys with a 24-hour TTL in a Redis or PostgreSQL table with a unique index.
**Impact**: Eliminates citizen double-submissions on flaky mobile networks; required for any production e-government portal.

---

## 5. Structured SLA Tracking and Breach Alerting

**Current state**: `avg_processing_days: 7.5` is hardcoded; no SLA deadline tracking.
**Improvement**: Persist `sla_deadline` (= `submitted_at + service.sla_days`) on every application. Add an async background task (APScheduler / Celery beat) that queries applications where `now() > sla_deadline AND status NOT IN ('completed','cancelled')` and emits `SlaBreachEvent` events. Surface breach counts in `service_analytics`.
**Impact**: Directly actionable operational intelligence; satisfies government service charter obligations; drives accountability at team and officer level.

---

## 6. Role-Based Access Control (RBAC) with Fine-Grained Permissions

**Current state**: `_enforce()` delegates to `evaluate_capability_rules()` with a flat context dict; no actor-role mapping.
**Improvement**: Introduce a `PermissionContext(actor_id, role, tenant_id, resource, action)` model. Map roles (`citizen`, `clerk`, `supervisor`, `auditor`, `admin`) to allowed `(resource, action)` pairs in the capability contract. Enforce at the service method boundary using a decorator `@require_permission("applications", "write")`.
**Impact**: Prevents privilege escalation; makes permission logic auditable and testable independently of business logic.

---

## 7. AI-Powered Application Pre-Screening

**Current state**: No automated completeness or eligibility check before human review.
**Improvement**: Add `async pre_screen_application(application_id)` that calls a local Ollama model (e.g. `llama3.2`) to evaluate: (a) document completeness against the service checklist, (b) eligibility based on declared details, (c) fraud signal scoring. Return a structured `PreScreenResult(score, flags, recommended_action)`. Clerk review queue is sorted by pre-screen score.
**Impact**: Reduces clerk workload by 40–60% by surfacing incomplete or likely-rejected applications before manual review begins.

---

## 8. Multi-Factor OTP Authentication with TOTP Support

**Current state**: `citizen_portal_login` generates a deterministic OTP from `hash(id_number + phone)` — trivially predictable.
**Improvement**: Replace with cryptographically random 6-digit OTP stored hashed (bcrypt) in Redis with 10-minute TTL. Add TOTP (`pyotp`) as a second factor for high-assurance services. Implement lockout after 5 failed attempts with exponential backoff.
**Impact**: Eliminates OTP predictability; complies with e-Government authentication standards (e.g. NIST 800-63B AAL2).

---

## 9. Offline-First USSD / SMS Interface

**Current state**: `channel` field accepts `"ussd"` but the service has no USSD-specific flow.
**Improvement**: Add `async ussd_session_handler(session_id, phone, input_text)` implementing a state-machine (using `transitions` library) that guides citizens through application, payment, and status-check flows via AT commands. Persist session state in Redis. Map USSD menus to the same service methods via a thin translation layer.
**Impact**: Extends portal reach to feature-phone users (~40% of Kenyan population); same backend, zero code duplication.

---

## 10. Document OCR and Auto-Population

**Current state**: Documents are referenced by ID strings; contents are never inspected.
**Improvement**: Add `async extract_document_fields(document_id, document_type)` that calls a local Ollama vision model (`llava` / `minicpm-v`) or Tesseract OCR to extract structured fields (name, ID number, date of birth, expiry) from uploaded images. Auto-populate application fields and flag mismatches against declared details.
**Impact**: Reduces citizen data-entry errors; accelerates verification from 24 hours to near-real-time; foundation for biometric matching.

---

## 11. Payment Reconciliation and Revenue Dashboard

**Current state**: `service_analytics` sums `PaymentRecord.amount` with no breakdown; no reconciliation.
**Improvement**: Add `async reconcile_payments(period)` that groups payments by service, channel, and M-Pesa transaction reference, cross-checks against expected fees from `ServiceDefinition.fee_amount`, and flags discrepancies. Produce a structured `ReconciliationReport` compatible with treasury reporting formats (IPSAS).
**Impact**: Closes a critical gap between collected fees and credited revenue; automates manual treasury reconciliation work currently done in spreadsheets.

---

## 12. Service Catalogue Versioning and Deprecation

**Current state**: `ServiceDefinition` has no version field; fee or SLA changes silently overwrite historical data.
**Improvement**: Add `version: int` and `valid_from / valid_to` date range to `ServiceDefinition`. `register_service` creates a new version rather than mutating in place. Applications reference the service version active at submission time. `service_search` returns only currently active versions by default.
**Impact**: Immutable audit trail for service definition changes; required for fee dispute resolution and regulatory compliance.

---

## 13. Citizen Profile and History Aggregation

**Current state**: No cross-application citizen view; `citizen_id` is a plain string with no associated profile.
**Improvement**: Add `async get_citizen_profile(citizen_id)` that aggregates all applications, payments, verifications, notifications, and appointments for a citizen across service types. Include a computed `citizen_risk_score` based on past application completeness and compliance. Cache profile in Redis with a 5-minute TTL.
**Impact**: Enables personalised service recommendations; reduces repeat document submission (cite once, reuse across services); supports fraud pattern detection.

---

## 14. Webhook / Push Notification Integration

**Current state**: Notifications are recorded as in-memory `CitizenNotification` objects; no actual delivery.
**Improvement**: Add `async dispatch_notification(notification_id)` with pluggable delivery adapters: SMS (Africa's Talking), email (SMTP/SendGrid), WhatsApp Business API, and push (FCM). Implement retry with exponential backoff (max 3 attempts); record delivery receipts and failure reasons. Respect per-citizen channel preferences set via `notification_preference`.
**Impact**: Citizens actually receive status updates; reduces counter visits for status enquiries by an estimated 60–70% based on comparable deployments.

---

## 15. Compliance Reporting and Data Retention Policies

**Current state**: Audit events are an unbounded in-memory list; no data lifecycle management.
**Improvement**: Implement `async generate_compliance_report(period, standard)` supporting GDPR-equivalent (Kenya Data Protection Act 2019), IPSAS, and ISO 27001 report templates. Add configurable data retention policies per service type (e.g. land applications: 30 years; feedback: 2 years). Implement soft-delete with `deleted_at` timestamps; hard-delete triggered by retention schedule. Expose `async export_citizen_data(citizen_id)` for right-to-access requests.
**Impact**: Mandatory for government systems; eliminates risk of DPA 2019 enforcement action; builds citizen trust through transparent data governance.
