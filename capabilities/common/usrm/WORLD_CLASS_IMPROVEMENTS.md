# USRM - World-Class Improvement Roadmap

**Capability**: User Management (`usrm`) | **Domain**: `common`
**Date**: 2026-06-11 | **Author**: Nyimbi Odero

---

## 1. Persistent Storage via Repository Pattern

**Current**: All records live in in-memory dicts on `UsrmService`. Any restart loses all state.

**Improvement**: Introduce an `AbstractUsrmRepository` interface with a `PostgresUsrmRepository` implementation using SQLAlchemy async sessions. The service receives the repo via constructor injection, keeping business logic storage-agnostic. The in-memory dict store becomes `InMemoryUsrmRepository` for tests and probes only.

**Impact**: Production-grade durability, zero code change to callers.

---

## 2. Avatar Upload and Storage Pipeline

**Current**: `UserProfileRecord` has no avatar support. Profiles carry only flat `attributes` dict.

**Improvement**: Add `avatar_url: str | None` and `avatar_mime: str | None` to `UserProfileRecord`. Add `async upload_avatar(tenant_id, user_id, image_bytes, mime_type, actor)` that validates MIME type, enforces max-size policy (configurable per tenant), stores via an `AbstractBlobStore` adapter (local FS, S3, or MinIO), and updates the profile record atomically. Emit `avatar_updated` audit event.

**Impact**: Closes a highly visible gap in user profile completeness.

---

## 3. Rich Activity History Timeline

**Current**: `audit_user_activity()` filters `audit_events` by naive string prefix matching. No pagination, no date-range filter, no event-type filter.

**Improvement**: Replace with `async get_activity_timeline(tenant_id, user_id, *, since, until, event_types, limit, cursor)` returning a typed `ActivityPage` with `items`, `total`, `next_cursor`. Store events in a secondary `user_id → [event_id]` index on write to avoid full-scan O(N) filtering. Support server-side cursor pagination compatible with the APG grid views.

**Impact**: Enables compliance reporting, per-user audit dashboards, and GDPR subject-access-request exports.

---

## 4. Fine-Grained Preference Schema with Versioning

**Current**: `privacy_preferences` is an untyped `dict[str, str]`. Any key is accepted; history is lost on each `update_profile` call.

**Improvement**: Introduce a `PreferenceSnapshot` model with a `version: int`, `effective_from: datetime`, and a validated `values: dict[PreferenceKey, str]` where `PreferenceKey` is a string enum. `update_profile` appends a new snapshot rather than overwriting; `get_preferences(user_id, at=<timestamp>)` returns the snapshot effective at that moment. Enables consent audit trails required under GDPR Art. 7(1).

**Impact**: Legally defensible preference history; enables point-in-time compliance queries.

---

## 5. MFA Device Registry

**Current**: `mfa_enabled` is a single boolean with no tracking of which device types are enrolled.

**Improvement**: Add `MfaDeviceRecord` (id, user_id, device_type: `totp|webauthn|sms|email`, enrolled_at, last_used_at, revoked). Provide `async enroll_mfa_device(...)`, `async revoke_mfa_device(...)`, `async list_mfa_devices(user_id)`. Derive `mfa_enabled` dynamically from the count of non-revoked devices. Emit high-severity audit events for enroll/revoke.

**Impact**: Supports step-up authentication, phishing-resistant WebAuthn flows, and device-loss recovery workflows.

---

## 6. Self-Service Password Reset with Time-Bound Tokens

**Current**: `password_reset()` accepts an opaque `reset_token` string with no expiry enforcement or single-use guarantee.

**Improvement**: Add `ResetTokenRecord` (id, user_id, token_hash: str, expires_at, used_at). `async issue_password_reset_token(user_id, channel)` generates a cryptographically random token, stores its bcrypt hash, and sends it over the configured channel adapter. `password_reset()` validates the token against the hash, enforces expiry, and marks `used_at`. Replay of used/expired tokens raises `PermissionError("reset_token_invalid")`.

**Impact**: Eliminates token-reuse vulnerabilities; satisfies NIST SP 800-63B § 5.1.1 reset guidance.

---

## 7. Delegated Administration (Sub-Tenant Admins)

**Current**: All admin operations are flat — any call with a valid `tenant_id` and `actor` string is treated equally.

**Improvement**: Introduce `DelegationRecord` (delegator_id, delegate_id, scope: list[str], expires_at). Add `async create_delegation(...)`, `async revoke_delegation(...)`, `async check_delegation(actor, operation, tenant_id)`. The service injects delegation checks before mutating operations, raising `PermissionError("delegation_scope_insufficient")` when the caller lacks explicit delegation or higher-level admin role.

**Impact**: Enables helpdesk staff, departmental managers, and AI agents to perform scoped admin actions without full tenant-admin privileges.

---

## 8. User Attribute Schema Enforcement

**Current**: `attributes` in `UserProfileRecord` is `dict[str, str]` — no schema, no validation per tenant.

**Improvement**: Add `TenantAttributeSchema` (tenant_id, fields: list[AttributeField(name, type, required, max_length, regex_pattern)]). Add `async define_attribute_schema(...)` and enforce the schema inside `update_profile` before persisting. Return structured `ValidationError` with per-field messages on failure.

**Impact**: Enforces data quality at the capability boundary; prevents garbage in RBAC and reporting pipelines downstream.

---

## 9. Notification / Webhook Dispatch

**Current**: Audit events are only stored internally. No external notification or webhook fanout.

**Improvement**: Add `WebhookSubscriptionRecord` (tenant_id, url, secret_hash, event_types: list[str], active). Add `async subscribe_webhook(...)`, `async unsubscribe_webhook(...)`. After each `_record_event()` call, dispatch to matching active subscriptions via an `AbstractNotificationAdapter` (HTTP POST with HMAC-SHA256 signature header). Support retry with exponential backoff up to 3 attempts.

**Impact**: Unblocks real-time integration with SIEM, incident management, and compliance tools without polling the audit log.

---

## 10. Compliance Report Generation

**Current**: `user_analytics()` returns aggregate counts only. No structured compliance artifact.

**Improvement**: Add `async generate_compliance_report(tenant_id, *, standard: Literal["gdpr","soc2","iso27001"], period_start, period_end, requested_by)` that collects access reviews, deprovision records, MFA adoption, privilege assignments, and consent snapshots into a structured `ComplianceReport` Pydantic model. Emit a high-severity audit event. Allow export to JSON or signed PDF via a pluggable renderer adapter.

**Impact**: Transforms USRM from an operational runtime into a first-class compliance evidence store.

---

## 11. Async-Native Service with `asyncio.Lock` Isolation

**Current**: All mutating methods are synchronous (`def`). The few existing `async def` methods just `await`-nothing and call sync helpers. Concurrent coroutines can corrupt in-memory records.

**Improvement**: Wrap every per-record mutation with a per-`user_id` `asyncio.Lock` acquired from an `asyncio.Lock` registry dict. Convert all `def` mutating methods to `async def`. This ensures the in-memory backend is safe for concurrent async callers without requiring external locking infra.

**Impact**: Eliminates race conditions; prepares the service for high-concurrency async web frameworks (ASGI/FastAPI) that co-host multiple coroutines per event loop.

---

## 12. Session Management with Expiry and Revocation

**Current**: `session_revoke_all()` emits an audit event but has no session state to actually invalidate. There is no session registry.

**Improvement**: Add `UserSessionRecord` (id, user_id, tenant_id, created_at, expires_at, revoked_at, metadata: dict). Add `async create_session(...)`, `async validate_session(session_id)` (raises `PermissionError` if revoked/expired), `async revoke_session(session_id, actor)`, and `async revoke_all_sessions(user_id, actor)`. Derive `sessions_revoked` count from actual records.

**Impact**: Enables zero-trust session controls, forced re-authentication on privilege change, and audit-accurate session counts for compliance dashboards.

---

## 13. User Import from External Identity Providers (IdP Sync)

**Current**: `bulk_create_users()` accepts raw dicts with no IdP provenance or de-duplication by external ID.

**Improvement**: Add `async sync_from_idp(tenant_id, provider: str, idp_records: list[IdpUserRecord], actor)` where `IdpUserRecord` carries `external_id`, `email`, `display_name`, `groups`, `attributes`. The method upserts (create-or-update) using `external_id` as the stable key, maps IdP groups to USRM roles via a configurable `IdpRoleMappingRecord`, and returns a `SyncResult` with counts of created/updated/skipped/errored. Emits `idp_sync_completed` audit event.

**Impact**: Closes the Active Directory / LDAP / SCIM gap that blocks enterprise adoption.

---

## 14. Granular Audit Event Severity and SIEM Tagging

**Current**: `_record_event` accepts `severity` as an unconstrained `str`. Only two values (`"low"`, `"medium"`) appear in code; no SIEM tags or structured threat categories.

**Improvement**: Replace `severity: str` with `severity: Literal["info","low","medium","high","critical"]`. Add `siem_category: str | None` and `mitre_tactic: str | None` to `UserAuditEventRecord`. Tag high-risk operations (`impersonation_started`, `all_sessions_revoked`, `user_merged`) with the appropriate MITRE ATT&CK tactic IDs (e.g., `TA0004` Privilege Escalation). Validate on write via `AfterValidator`.

**Impact**: Events become directly ingestible by Splunk, Elastic SIEM, and Chronicle without a mapping layer.

---

## 15. Capability-Level Rate Limiting and Abuse Detection

**Current**: No rate limiting on any operation. A misconfigured agent or compromised actor can call `create_user` or `assign_role` in a tight loop.

**Improvement**: Integrate the existing `BoundedCache` import (already imported from `capabilities.common.reliability`) as a sliding-window rate limiter: track `(tenant_id, actor, operation)` call counts per minute in a `BoundedCache`. Raise `PermissionError("rate_limit_exceeded")` when the per-minute count exceeds a configurable threshold (default: 100 calls/min for writes, 500 for reads). Expose `async get_rate_limit_status(tenant_id, actor)` for observability.

**Impact**: First line of defense against both accidental loop bugs and credential-stuffing / insider-threat scenarios, directly leveraging existing reliability infrastructure.

---

*Improvements are ordered by implementation readiness and compliance impact. Items 1, 2, 3, 5, 6, and 12 form the MVP cluster for a production hardening sprint.*
