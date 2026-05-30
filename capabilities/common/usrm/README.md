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
- First-class USRM agents for Codex, Claude Code, OpenCode, and Pi based review
  lanes.
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
