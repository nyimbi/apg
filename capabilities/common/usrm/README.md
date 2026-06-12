# USRM - User Management

USRM is the APG capability for governed user lifecycle management. It gives
generated applications a composable runtime for user identity, profiles,
consented invitations, role assignment, privileged MFA, access reviews, privacy
preferences, deprovisioning, bulk lifecycle actions, AI-assisted review, and
Bytewax lifecycle events.

Use USRM when an application needs a tenant-aware user directory with clear
guardrails for onboarding, access, privacy, and offboarding.

## What USRM Provides

- Tenant-scoped user directory.
- Profile and privacy preference records.
- Consent-backed invitations.
- Role assignment governance and privileged-role approval.
- Privileged MFA enforcement.
- Periodic access review evidence.
- Deprovisioning with access revocation and evidence references.
- Bulk lifecycle action review.
- Account lock/unlock with audit trail.
- Admin impersonation sessions with reason and duration.
- Permission grant/revoke (fine-grained, non-privileged roles).
- User group creation and membership management.
- Password reset and tenant-level password policy enforcement.
- Session revocation (all-sessions, per-user).
- User analytics (MFA adoption, activity aggregates).
- User export (JSON, with optional profiles).
- User merge (primary/secondary account consolidation).
- Bulk user creation and bulk deactivation.
- First-class USRM agents for Codex, Claude Code, OpenCode, and Pi review lanes.
- Bytewax lifecycle stream metadata.
- Dashboard, user directory, profile, lifecycle, access, privacy,
  deprovisioning, agent, policy, and settings view models.

## Quick Start

```python
from capabilities.common.usrm import UsrmService

service = UsrmService()

user = service.create_user(
    tenant_id="tenant-a",
    identity="jane@example.com",
    display_name="Jane Doe",
    email="jane@example.com",
    owner="identity-owner",
    profile_validated=True,
)

service.update_profile(
    tenant_id="tenant-a",
    user_id=user["id"],
    attributes={"department": "finance"},
    privacy_preferences={"analytics": "limited"},
    consent_notice_ref="consent://notice/1",
    updated_by="profile-admin",
)

service.invite_user(
    tenant_id="tenant-a",
    user_id=user["id"],
    channel="email",
    consent_notice_ref="consent://notice/1",
    invited_by="identity-owner",
)

service.assign_role(
    tenant_id="tenant-a",
    user_id=user["id"],
    role="finance-reviewer",
    scope="tenant",
    privileged=False,
    mfa_enabled=True,
    approved_by="access-owner",
)

summary = service.dashboard_summary("tenant-a")
print(summary["user_count"])
```

## Privileged Access

Privileged users and privileged role assignments require MFA. Privileged role
assignment also requires approval.

```python
service.assign_role(
    tenant_id="tenant-a",
    user_id=user["id"],
    role="tenant-admin",
    scope="tenant",
    privileged=True,
    mfa_enabled=True,
    approved_by="access-owner",
)
```

## Deprovisioning

Deprovisioning requires access revocation evidence and Bytewax lifecycle stream
metadata.

```python
deprovision = service.deprovision_user(
    tenant_id="tenant-a",
    user_id=user["id"],
    actor="identity-owner",
    access_revoked=True,
    evidence_ref="evidence://deprovision/1",
    event_stream="bytewax",
)

assert deprovision["status"] == "completed"
```

## USRM Agents

USRM treats user lifecycle review agents as governed composition elements.

```python
agent = service.register_usrm_agent(
    tenant_id="tenant-a",
    name="Access reviewer",
    runtime="codex",
    role="access_reviewer",
    scope="review roles, MFA, and access findings",
)

decision = service.validate_agent_user_action(
    tenant_id="tenant-a",
    agent_id=agent["id"],
    action="assign_role",
    privileged_scope=True,
)

assert decision["decision"] == "deny"
```

Privileged agent user actions require human approval:

```python
decision = service.validate_agent_user_action(
    tenant_id="tenant-a",
    agent_id=agent["id"],
    action="assign_role",
    privileged_scope=True,
    human_approval_ref="approval://agent/user",
)

assert decision["decision"] == "allow"
```

## Batch Lifecycle Guardrail

Bulk user lifecycle actions must use Bytewax stream coordination and require
review when the affected user count crosses the configured threshold.

```python
decision = service.validate_batch_user_lifecycle(
    tenant_id="tenant-a",
    affected_user_count=30,
    event_stream="bytewax",
    bulk_review_recorded=True,
)

assert decision["decision"] == "allow"
```

## New Methods

### Account lock and unlock

```python
import asyncio

await service.account_lock(
    tenant_id="tenant-a",
    user_id=user["id"],
    reason="suspicious_login",
    actor="security-admin",
)

await service.account_unlock(
    tenant_id="tenant-a",
    user_id=user["id"],
    actor="security-admin",
    justification="false positive confirmed",
)
```

### Admin impersonation

```python
session = await service.impersonate(
    tenant_id="tenant-a",
    admin_id="admin-001",
    target_user_id=user["id"],
    reason="support-ticket-1234",
    duration_minutes=15,
)
# session["session_id"] tracks the impersonation for audit
```

### Permission grant and revoke

```python
await service.permission_grant(
    tenant_id="tenant-a",
    user_id=user["id"],
    permission="report:read",
    scope="finance",
    granted_by="access-owner",
)

await service.permission_revoke(
    tenant_id="tenant-a",
    user_id=user["id"],
    permission="report:read",
    revoked_by="access-owner",
)
```

### Password reset and policy enforcement

```python
await service.password_reset(
    tenant_id="tenant-a",
    user_id=user["id"],
    reset_token="tok-abc123",
    new_password_hash="$2b$12$...",
    actor="user",
)

policy = await service.password_policy_enforce(
    tenant_id="tenant-a",
    min_length=14,
    require_uppercase=True,
    require_symbols=True,
    max_age_days=60,
    actor="security-admin",
)
```

### User analytics

```python
stats = await service.user_analytics(tenant_id="tenant-a", days=30)
print(stats["mfa_adoption_rate"])   # float 0.0–1.0
print(stats["privileged_users"])
```

## Deterministic Rules

USRM enforces:

- tenant context on all executable operations;
- unique identity, owner, and profile validation for user creation;
- consent notice and Bytewax stream metadata for invitations;
- privacy preference sync for profile updates;
- MFA for privileged users and roles;
- approval for privileged role assignment;
- reviewer attribution for access reviews;
- access revocation, evidence, and Bytewax metadata for deprovisioning;
- review and Bytewax coordination for bulk lifecycle actions;
- supported USRM-agent runtime and role;
- human approval for privileged agent actions.

## World-Class Enhancements (v2.0)

Planned improvements ordered by implementation readiness and compliance impact.
Items 1, 2, 3, 5, 6, and 12 form the MVP cluster for a production hardening sprint.

1. **Persistent Storage via Repository Pattern** — `AbstractUsrmRepository` with `PostgresUsrmRepository` (SQLAlchemy async); `InMemoryUsrmRepository` for tests.
2. **Avatar Upload Pipeline** — `upload_avatar(tenant_id, user_id, image_bytes, mime_type, actor)` via pluggable `AbstractBlobStore` (S3/MinIO/local).
3. **Rich Activity Timeline** — `get_activity_timeline(...)` with cursor pagination, date-range and event-type filters, O(1) lookup via secondary index.
4. **Versioned Preference Snapshots** — `PreferenceSnapshot` with `version`, `effective_from`, and typed `PreferenceKey` enum; point-in-time `get_preferences(user_id, at=...)`.
5. **MFA Device Registry** — `MfaDeviceRecord` supporting `totp|webauthn|sms|email`; `enroll_mfa_device`, `revoke_mfa_device`, `list_mfa_devices`.
6. **Self-Service Password Reset Tokens** — `ResetTokenRecord` with bcrypt-hashed, single-use, time-bound tokens; satisfies NIST SP 800-63B § 5.1.1.
7. **Delegated Administration** — `DelegationRecord` with scoped, expiring delegations; `create_delegation`, `revoke_delegation`, `check_delegation`.
8. **Attribute Schema Enforcement** — `TenantAttributeSchema` with per-field type, required, max-length, and regex rules enforced at `update_profile`.
9. **Webhook Dispatch** — `WebhookSubscriptionRecord` with HMAC-SHA256 signed HTTP POST fanout after each audit event; 3-attempt exponential backoff.
10. **Compliance Report Generation** — `generate_compliance_report(standard: "gdpr"|"soc2"|"iso27001", ...)` producing a structured `ComplianceReport`; JSON or signed PDF export.
11. **Async-Native with Per-Record Locking** — All mutating methods converted to `async def` with per-`user_id` `asyncio.Lock` to eliminate race conditions under ASGI concurrency.
12. **Session Management** — `UserSessionRecord` with `create_session`, `validate_session`, `revoke_session`, `revoke_all_sessions`; actual session state backing `session_revoke_all`.
13. **IdP Sync** — `sync_from_idp(provider, idp_records)` with `external_id`-keyed upsert, group-to-role mapping via `IdpRoleMappingRecord`, and `SyncResult` counts.
14. **Structured Audit Severity and SIEM Tagging** — Typed `severity: Literal["info","low","medium","high","critical"]`; `siem_category` and `mitre_tactic` fields on `UserAuditEventRecord`.
15. **Rate Limiting and Abuse Detection** — Sliding-window `BoundedCache` rate limiter per `(tenant_id, actor, operation)`; `get_rate_limit_status` for observability; default 100 writes/min, 500 reads/min.

## API Helpers

`api.py` provides payload-oriented helpers:

- `capability_status()`
- `create_user()`
- `update_profile()`
- `invite_user()`
- `assign_role()`
- `record_access_review()`
- `deprovision_user()`
- `bulk_suspend_users()`
- `register_usrm_agent()`
- `validate_agent_user_action()`
- `validate_batch_user_lifecycle()`
- `create_record()`
- `list_records()`
- `list_user_management()`

## UI Routes

- dashboard: `/usrm/dashboard`
- users: `/usrm/users`
- profiles: `/usrm/profiles`
- lifecycle: `/usrm/lifecycle`
- access: `/usrm/access`
- privacy: `/usrm/privacy`
- deprovisioning: `/usrm/deprovisioning`
- agents: `/usrm/agents`
- policy: `/usrm/policy`
- settings: `/usrm/settings`

## Bytewax Stream

USRM publishes lifecycle metadata for Bytewax:

- processor: `bytewax`
- stream: `apg.usrm.lifecycle`
- key: `tenant_id`

Events:

- `user_created`
- `profile_updated`
- `user_invited`
- `role_assigned`
- `access_review_recorded`
- `user_deprovisioned`
- `bulk_suspend_users`
- `usrm_agent_registered`
- `account_locked`
- `account_unlocked`
- `impersonation_started`
- `password_reset`
- `user_merged`

## Adapter Boundaries

The in-package service stores records in memory so generated applications,
tests, and publish-plan probes can execute without external infrastructure.
Production systems should attach identity stores, RBAC providers, MFA
providers, consent registries, access-review workflows, deprovisioning
automation, audit sinks, and Bytewax workers through APG adapters.

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/usrm/__init__.py capabilities/common/usrm/capability_contract.py capabilities/common/usrm/models.py capabilities/common/usrm/user_runtime.py capabilities/common/usrm/service.py capabilities/common/usrm/api.py capabilities/common/usrm/views.py capabilities/common/usrm/app.py capabilities/common/usrm/test_capability_contract.py capabilities/common/usrm/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/usrm/test_capability_contract.py capabilities/common/usrm/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/usrm --json
./.venv/bin/apg capabilities publish-plan capabilities/common/usrm --json
```
