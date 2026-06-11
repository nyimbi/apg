# Composition Access Control — User Guide

© 2025 Datacraft | www.datacraft.co.ke

---

## Table of Contents

1. [Introduction](#introduction)
2. [Concepts](#concepts)
3. [Quick Start](#quick-start)
4. [Identity Providers](#identity-providers)
5. [Resources and Scopes](#resources-and-scopes)
6. [Policies](#policies)
7. [Grants](#grants)
8. [Just-In-Time (JIT) Privileged Access](#just-in-time-jit-privileged-access)
9. [RBAC and Role Inheritance](#rbac-and-role-inheritance)
10. [Self-Service Access Requests](#self-service-access-requests)
11. [Session Risk Control](#session-risk-control)
12. [Access Decisions](#access-decisions)
13. [Permission Matrix Export](#permission-matrix-export)
14. [Policy Simulation Sandbox](#policy-simulation-sandbox)
15. [Compliance Reports](#compliance-reports)
16. [Access Analytics](#access-analytics)
17. [Grant Expiry Reaper](#grant-expiry-reaper)
18. [AI Access Agents](#ai-access-agents)
19. [Audit Trail](#audit-trail)
20. [Streaming with Bytewax and NATS](#streaming-with-bytewax-and-nats)
21. [Business Rules Reference](#business-rules-reference)
22. [Composability and Integration](#composability-and-integration)
23. [Troubleshooting](#troubleshooting)

---

## Introduction

The Composition Access Control capability (`composition_access`) is the single authorisation hub for every capability in the APG composition layer. Instead of each capability maintaining its own permission tables, they all delegate authZ decisions here — producing one audit trail, one revocation point, and a uniform risk model.

This guide is aimed at three audiences:

- **Operators** — day-to-day grant management and access review.
- **Governance / Security** — policy authoring, compliance reporting, access reviews.
- **Developers** — integrating downstream capabilities and writing composition tests.

---

## Concepts

| Term | Meaning |
|------|---------|
| **Tenant** | An isolated organisational unit. All records are scoped to `tenant_id`. |
| **Provider** | An identity provider (OIDC, SAML, LDAP, API key, JWT, local) registered with the hub. |
| **Resource** | Any protected asset (capability endpoint, data object, API route) with one or more named scopes. |
| **Policy** | A rule that maps a resource + conditions → `allow` or `deny`. Policies require an owner and go through a draft → active lifecycle. |
| **Grant** | An explicit entitlement binding a subject (user, service, agent) to a resource at specific scopes, with optional expiry. |
| **Session** | A risk-scored authentication context. Risk > 74 triggers mandatory step-up auth. |
| **Decision** | A recorded authZ verdict (allow/deny/review) emitted to the event stream. |
| **Role** | A named set of scopes with optional parent-role inheritance for RBAC hierarchies. |
| **JIT Grant** | A time-bounded privileged grant that is pending until an independent approver activates it. |
| **Access Request** | A self-service entitlement request that flows through an approve/deny workflow before creating a grant. |
| **Reaper** | The background sweep that expires grants past their `expires_at` timestamp. |

---

## Quick Start

```python
from capabilities.composition.access.service import CompositionAccessService

svc = CompositionAccessService()

# 1. Register an identity provider
provider = svc.register_provider(
    provider_key="corp-oidc",
    tenant_id="acme",
    name="Corporate OIDC",
    provider_type="oidc",
    owner_id="alice",
    external=True,
    secret_reference="vault://secrets/corp-oidc",
    test_evidence="oidc-smoke-test-2025-06-01",
    metadata_validated=True,
)

# 2. Activate the provider
svc.activate_provider(provider["id"], actor_id="alice")

# 3. Register a protected resource
resource = svc.register_resource(
    resource_key="reports-api",
    tenant_id="acme",
    display_name="Reports API",
    owner_id="alice",
    scopes=["read", "write", "admin"],
    capability_id="composition_access",
)

# 4. Issue a grant
grant = svc.create_grant(
    grant_key="bob-reports-read",
    tenant_id="acme",
    subject_id="bob",
    resource_id=resource["id"],
    scopes=["read"],
    requested_by="bob",
    justification="Need read access to generate weekly report",
    approved_by="alice",
)

# 5. Check access at runtime
import asyncio
decision = asyncio.run(
    svc.check_access("acme", "bob", resource["id"], "GET", "read")
)
print(decision["decision"])  # "allow"
```

---

## Identity Providers

### Register a Provider

All provider registrations require an `owner_id`. External providers additionally require a `secret_reference` pointing to a vault/secret-manager path — plaintext credentials at the record level are blocked by the `provider_requires_secret_reference` rule.

```python
svc.register_provider(
    provider_key="saml-hr",
    tenant_id="acme",
    name="HR SAML",
    provider_type="saml",
    owner_id="iam-team",
    external=True,
    metadata_validated=True,
    secret_reference="vault://secrets/saml-hr",
    test_evidence="saml-roundtrip-2025-06-01",
)
```

### Activate a Provider

Activation validates that metadata is present and test evidence has been recorded:

```python
svc.activate_provider(
    provider_id="...",
    actor_id="iam-team",
    metadata_validated=True,
    test_evidence="saml-roundtrip-2025-06-01",
)
```

### Rotate a Secret Reference

When a vault secret path changes:

```python
await svc.rotate_secret(
    provider_id="...",
    actor_id="iam-team",
    new_secret_reference="vault://secrets/saml-hr-v2",
)
```

---

## Resources and Scopes

Resources represent any protected asset. Define scopes precisely — they are validated at grant time:

```python
resource = svc.register_resource(
    resource_key="billing-api",
    tenant_id="acme",
    display_name="Billing API",
    owner_id="payments-team",
    scopes=["read", "write", "admin"],
    capability_id="composition_billing",
    sensitive=True,  # triggers mandatory conditions on any policy
)
```

Sensitive resources require explicit conditions on any attached policy; the `sensitive_policy_requires_conditions` rule enforces this.

---

## Policies

### Create a Policy

```python
policy = svc.create_policy(
    policy_key="billing-write-mfa",
    tenant_id="acme",
    name="Billing Write — MFA Required",
    resource_id=resource["id"],
    owner_id="security-team",
    effect="allow",
    conditions={"mfa_verified": True},
    risk_level="standard",
)
```

### Simulate Before Activating High-Risk Policies

For `risk_level="high"`, simulation evidence is required before activation:

```python
simulation = await svc.simulate_policy(
    policy_id=policy["id"],
    sample_decisions=[
        {"mfa_verified": True, "decision": "allow"},
        {"mfa_verified": False, "decision": "deny"},
    ],
    actor_id="security-team",
)
# simulation["simulation_evidence"] is now attached to the policy record

svc.activate_policy(
    policy_id=policy["id"],
    actor_id="security-team",
    simulation_evidence=simulation["simulation_evidence"],
    reviewed_by="ciso",
)
```

---

## Grants

### Standard Grant

```python
grant = svc.create_grant(
    grant_key="carol-billing-read",
    tenant_id="acme",
    subject_id="carol",
    resource_id=resource["id"],
    scopes=["read"],
    requested_by="carol",
    justification="Monthly reconciliation task",
    approved_by="finance-lead",
    expires_at="2025-09-01T00:00:00+00:00",
)
```

### Privileged Grant Rules

Privileged grants (`privileged=True`) additionally require:
- A distinct approver (`approved_by != requested_by`)
- An explicit `expires_at`

### Suspend / Reinstate

```python
await svc.suspend_grant(grant["id"], actor_id="security-team", reason="Incident under investigation")
await svc.reinstate_grant(grant["id"], actor_id="security-team")
```

### Bulk Revoke

```python
result = await svc.bulk_revoke_grants(
    grant_ids=["g1", "g2", "g3"],
    actor_id="security-team",
    reason="Employee offboarding batch 2025-Q2",
)
print(result["revoked_count"])
```

---

## Just-In-Time (JIT) Privileged Access

JIT grants eliminate standing privilege. Access is granted for a bounded window only:

```python
# 1. Request JIT access
jit = await svc.request_jit_grant(
    tenant_id="acme",
    subject_id="dave",
    resource_id=resource["id"],
    scopes=["admin"],
    justification="Emergency schema migration — ticket #4421",
    duration_minutes=30,
    approver_id="cto",
    requested_by="dave",
)
# jit["status"] == "pending_jit_approval"

# 2. Approver activates (clock starts now)
activated = await svc.approve_jit_grant(jit["id"], approver_id="cto")
# activated["expires_at"] == now + 30 minutes
# activated["status"] == "active"
```

The reaper (see [Grant Expiry Reaper](#grant-expiry-reaper)) auto-expires the grant after the window closes.

---

## RBAC and Role Inheritance

### Create Roles with Inheritance

```python
# Base viewer role
viewer = await svc.create_role(
    tenant_id="acme",
    name="reports-viewer",
    scopes=["read"],
    description="Read-only access to reports",
    owner_id="iam-team",
)

# Operator inherits viewer
operator = await svc.create_role(
    tenant_id="acme",
    name="reports-operator",
    scopes=["read", "write"],
    description="Read+write access to reports",
    owner_id="iam-team",
    parent_role_id=viewer["id"],
)
```

### Assign a Role

```python
assignment = await svc.assign_role(
    tenant_id="acme",
    subject_id="eve",
    role_id=operator["id"],
    approver_id="iam-team",
    expires_at="2025-12-31T00:00:00+00:00",
    justification="Quarterly rotation",
)
```

### Resolve Effective Scopes

`resolve_effective_scopes` combines direct grants and every role in the inheritance chain:

```python
effective = await svc.resolve_effective_scopes(
    tenant_id="acme",
    subject_id="eve",
    resource_id=resource["id"],
)
print(effective["effective_scopes"])  # ["read", "write"]
```

---

## Self-Service Access Requests

Users submit requests; approvers decide asynchronously.

```python
# User submits
request = await svc.submit_access_request(
    tenant_id="acme",
    requester_id="frank",
    resource_id=resource["id"],
    scopes=["read"],
    justification="Need access to run Q2 analysis",
    expires_at="2025-09-01T00:00:00+00:00",
)
# request["status"] == "pending"

# Approver approves — grant is created automatically
approved = await svc.approve_access_request(
    request_id=request["id"],
    approver_id="finance-lead",
    comment="Approved for Q2 analysis window",
)
# approved["grant_id"] contains the new grant

# Or deny
denied = await svc.deny_access_request(
    request_id=request["id"],
    approver_id="finance-lead",
    reason="Access not required for this role",
)
```

---

## Session Risk Control

```python
session = svc.evaluate_session(
    session_key="frank-session-001",
    tenant_id="acme",
    subject_id="frank",
    provider_id=provider["id"],
    risk_score=45,          # 0-100; > 74 requires step-up
    step_up_completed=False,
)
print(session["status"])  # "verified"

# High-risk session
blocked_session = svc.evaluate_session(
    session_key="frank-session-002",
    tenant_id="acme",
    subject_id="frank",
    provider_id=provider["id"],
    risk_score=90,
    step_up_completed=False,  # will be "blocked"
)
```

Risk scores above 74 flip status to `blocked`. Passing `step_up_completed=True` (after the user completes MFA/WebAuthn) overrides the block.

---

## Access Decisions

Every authZ verdict must be recorded and streamed through Bytewax+NATS:

```python
decision = svc.record_decision(
    decision_key="frank-reports-get-001",
    tenant_id="acme",
    subject_id="frank",
    resource_id=resource["id"],
    action="GET",
    decision="allow",
    reason="active_grant_found",
    policy_ids=[policy["id"]],
    event_stream="bytewax",
)
```

The `event_stream` parameter must be `"bytewax"` — the `decision_requires_bytewax_stream` rule rejects any other value.

---

## Permission Matrix Export

Generates a cross-tenant snapshot of all active entitlements, useful for SOC 2 / ISO 27001 access reviews:

```python
matrix = await svc.export_permission_matrix(tenant_id="acme", format="json")
# matrix["matrix"] → {"frank": {"resource-id": ["read"]}, ...}

# For auditors
csv_matrix = await svc.export_permission_matrix(tenant_id="acme", format="csv")
# csv_matrix["content"] → "subject_id,resource_id,scopes\nfrank,..."
```

---

## Policy Simulation Sandbox

Run a proposed policy against historical decisions before activating it:

```python
result = await svc.simulate_policy(
    policy_id=policy["id"],
    sample_decisions=[
        {"mfa_verified": True, "subject_clearance": "confidential", "decision": "allow"},
        {"mfa_verified": False, "subject_clearance": "public", "decision": "deny"},
        {"mfa_verified": True, "subject_clearance": "public", "decision": "allow"},
    ],
    actor_id="security-team",
)
print(result["changed_count"])  # decisions the new policy would change
```

The `simulation_evidence` value is written back to the policy record and satisfies the `high_risk_policy_requires_simulation` gate.

---

## Compliance Reports

```python
report = await svc.access_compliance_report(tenant_id="acme", standard="ISO27001")
# {
#   "standard": "ISO27001",
#   "total_grants": 42,
#   "privileged_grants": 5,
#   "approved_privileged_grants": 5,
#   "expired_grants": 1,
#   "compliance_rate_pct": 100.0,
# }
```

Supported standards: `ISO27001`, `SOC2` (both produce the same structural report; the `standard` field is included in the audit event for external tooling).

---

## Access Analytics

```python
analytics = await svc.access_analytics(tenant_id="acme", period="last_30_days")
# {
#   "total_decisions": 3840,
#   "allow_count": 3712,
#   "deny_count": 128,
#   "allow_rate_pct": 96.67,
#   "top_subjects": [{"subject_id": "frank", "count": 420}, ...]
# }
```

---

## Grant Expiry Reaper

Call `reap_expired_grants` from a scheduler or background task:

```python
result = await svc.reap_expired_grants(tenant_id="acme")
print(result["reaped_count"])   # number of grants marked "expired"
print(result["grant_ids"])      # list of affected grant IDs
```

The reaper is idempotent — repeated calls on already-expired grants are no-ops. Each sweep emits a `grant_expired` audit event per revoked grant, and publishes to the Bytewax+NATS `apg.composition.access.lifecycle` stream.

**Recommended schedule**: every 5 minutes via APG's built-in task scheduler or an external cron. Example (pseudo-code):

```python
async def reaper_loop(svc, interval_seconds=300):
    while True:
        await svc.reap_expired_grants()
        await asyncio.sleep(interval_seconds)
```

---

## AI Access Agents

Access agents are AI workloads that operate within bounded scopes. All agents must use an approved runtime (`codex`, `claude_code`, `opencode`, `pi`) and an approved role.

```python
agent = svc.register_access_agent(
    tenant_id="acme",
    name="grant-reviewer-bot",
    runtime="claude_code",
    role="grant_reviewer",
    instructions="Review grant requests for policy compliance. Flag anomalies. Never approve.",
)

# Validate an agent-proposed action
svc.validate_agent_access_action(
    tenant_id="acme",
    agent_id=agent["id"],
    action="recommend_grant_approval",
    privileged_scope=False,
    human_approval_recorded=False,
)
```

Privileged agent actions (those involving `privileged_scope=True`) are blocked unless `human_approval_recorded=True`. The `max_autonomous_scope` ceiling is `"read_and_recommend"`.

---

## Audit Trail

Every state-changing operation appends an `AccessAuditEventRecord`. Query the in-memory trail:

```python
events = svc.audit_events(tenant_id="acme")
# [{"id": ..., "event_type": "grant_created", "entity_id": ..., ...}]
```

Export as CSV for external SIEM ingestion:

```python
log = await svc.export_access_log(tenant_id="acme", format="csv")
# log["content"] → CSV string
```

For durable persistence, wire the `emit_audit_event` path to the NATS JetStream subject `apg.access.audit.events` and consume with the Bytewax pipeline into PostgreSQL (see `database/schema.sql`).

---

## Streaming with Bytewax and NATS

All lifecycle events flow through the Bytewax stream processor subscribed to the NATS subjects:

| NATS Subject | Contents |
|---|---|
| `apg.composition.access.lifecycle` | Provider, resource, policy, grant, session, decision events |
| `apg.access.jit.requests` | JIT grant request and approval notifications |
| `apg.access.reviews.pending` | Periodic access review dispatch messages |
| `apg.access.requests.*` | Self-service request state changes (submitted/approved/denied) |
| `apg.access.audit.events` | Immutable audit events (JetStream persistent, DiscardNew) |
| `apg.risk.signals.>` | Inbound real-time risk signals for session rescoring |

Downstream capabilities subscribe to `apg.composition.access.lifecycle` to react to grant revocations and policy changes without polling the service directly.

---

## Business Rules Reference

The rule engine is deterministic. Each rule fires when **all** conditions in its `condition` dict match the evaluation context. The `_ne`, `_gt`, `_gte`, `_lt`, `_lte` suffixes allow numeric and inequality comparisons on keys.

| Rule Name | Key Condition | Effect |
|-----------|--------------|--------|
| `tenant_context_required` | `tenant_context_present: False` | deny |
| `cross_tenant_access_blocked` | `cross_tenant_access_attempted: True` | deny |
| `privilege_escalation_blocked` | `privilege_escalation_detected: True` | deny |
| `circuit_breaker_open_blocks_requests` | `circuit_breaker_state: "open"` | deny |
| `bulkhead_overflow_sheds_load` | `bulkhead_capacity_exceeded: True` | deny |
| `provider_requires_owner` | `provider_owner_assigned: False` | deny |
| `provider_requires_secret_reference` | `external_provider: True, secret_reference_present: False` | deny |
| `sensitive_policy_requires_conditions` | `sensitive_resource: True, policy_conditions_present: False` | deny |
| `high_risk_policy_requires_simulation` | `risk_level: "high", simulation_evidence_present: False` | require_review |
| `privileged_grant_requires_approval` | `privileged_scope: True, approval_recorded: False` | deny |
| `grant_requires_separation_of_duties` | `separation_of_duties_passed: False` | deny |
| `high_risk_session_requires_step_up` | `risk_score_gt: 74, step_up_completed: False` | deny |
| `decision_requires_bytewax_stream` | `event_stream_ne: "bytewax"` | deny |
| `privileged_agent_action_requires_human_approval` | `privileged_scope: True, human_approval_recorded: False` | deny |
| `service_mesh_identity_required` | `mesh_identity_verified: False` | deny |

---

## Composability and Integration

### Downstream Capabilities

Any capability that needs to gate a write operation calls:

```python
decision = await svc.check_access(
    tenant_id=tenant_id,
    subject_id=caller_id,
    resource_id=resource_id,
    action=http_method,
    scope=required_scope,
)
if decision["decision"] != "allow":
    raise PermissionError("access_denied")
```

### Flask-AppBuilder Blueprint Integration

Mount the capability blueprint in your FAB app:

```python
from capabilities.composition.access.blueprint import CompositionAccessBlueprint
appbuilder.add_view_no_menu(CompositionAccessBlueprint)
```

All FAB views enforce `composition_access:{view|govern|grant|admin|audit|operate}` permissions against the standard FAB role model.

---

## Troubleshooting

### `PermissionError: tenant_context_required`
Every call requires a non-empty `tenant_id`. Check that the caller is passing `tenant_id` and that it is not an empty string.

### `PermissionError: provider_secret_reference_required`
External providers must reference a vault path in `secret_reference`. Plaintext credentials are blocked by design.

### `ValueError: grant_scope_not_registered`
The requested scope is not in the resource's `scopes` list. Either add the scope to the resource via `register_resource` or use only declared scopes.

### `PermissionError: separation_of_duties_required`
The `requested_by` and `approved_by` fields are the same principal. An independent approver must be designated.

### Policy activation blocked: `policy_simulation_required`
Run `simulate_policy` first and pass the returned `simulation_evidence` to `activate_policy`.

### JIT grant stuck in `pending_jit_approval`
Only the `designated_approver` specified at request time can call `approve_jit_grant`. Check the `metadata.designated_approver` field on the grant record.

### Grants not expiring
Ensure the reaper is running. Call `await svc.reap_expired_grants(tenant_id)` on schedule. Check `expires_at` timestamps are in ISO-8601 UTC format.

### `ValueError: access_request_not_pending`
The request has already been decided. Check `request["status"]` — only `"pending"` requests can be approved or denied.
