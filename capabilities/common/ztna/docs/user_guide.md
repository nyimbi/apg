# Zero Trust Network Access — User Guide

**Capability ID**: `ztna` | **Domain**: `common` | **Version**: `1.0.0`

---

## Overview

`ztna` is APG's package-backed Zero Trust Network Access capability. It provides a tenant-scoped access broker for identity, device posture, protected resources, access requests, access reviews, governed sessions, risk reevaluation, and audit events. The service is intentionally dependency-light; live identity providers, MFA services, posture collectors, and audit sinks remain APG adapter boundaries.

---

## Installation

```bash
pip install apg-common-ztna
```

---

## Core Concepts

| Concept | Description |
|---|---|
| Identity | A verified subject (user or service account) with optional MFA and federation context. |
| Device | An endpoint with a trust score, posture evidence, attestation state, and compliance flag. |
| Resource | A protected application or service with an access level, sensitivity flag, and attached policy. |
| Access Request | A rule-evaluated request linking identity + device + resource. May be approved, denied, or routed to review. |
| Session | An active, auditable connection to a resource, anchored to an approved access request. |
| Agent | A first-class AI agent record assigned to a governed zero-trust scope. |
| Lifecycle Batch | A Bytewax-governed batch of zero-trust mutations, validated before execution. |

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/ztna/dashboard` | `ztna:view` | Overview |
| `/ztna/policies` | `ztna:manage_policies` | Policies |
| `/ztna/identities` | `ztna:manage_policies` | Identity |
| `/ztna/devices` | `ztna:manage_devices` | Devices |
| `/ztna/resources` | `ztna:manage_policies` | Resources |
| `/ztna/access` | `ztna:approve_access` | Access |
| `/ztna/sessions` | `ztna:view` | Operations |
| `/ztna/risk` | `ztna:view` | Operations |
| `/ztna/reviews` | `ztna:approve_access` | Reviews |
| `/ztna/agents` | `ztna:manage_policies` | Agents |
| `/ztna/lifecycle` | `ztna:manage_policies` | Lifecycle |
| `/ztna/audit` | `ztna:view` | Audit |
| `/ztna/settings` | `ztna:manage_policies` | Settings |

---

## Service Methods

### Identity Management

| Method | Description |
|---|---|
| `register_identity(...)` | Register a new identity with verification, privilege, MFA, and federation context. |
| `verify_identity(identity_id, actor_id, mfa_completed)` | Mark an identity as verified and optionally record MFA completion. |
| `list_identities(tenant_id)` | List all identities, optionally scoped to a tenant. |
| `async_register_identity(...)` | Async variant of `register_identity` for use in async adapters. |

### Device Management

| Method | Description |
|---|---|
| `register_device(...)` | Register a device with trust score, posture, managed state, attestation, and compliance. |
| `update_device_posture(device_id, ...)` | Update a device's trust score, posture, compliance, and attestation state. |
| `list_devices(tenant_id)` | List all devices, optionally scoped to a tenant. |
| `async_update_device_posture(...)` | Async variant of `update_device_posture` for continuous telemetry pipelines. |

### Resource Management

| Method | Description |
|---|---|
| `register_resource(...)` | Register a protected resource with access level, sensitivity, policy, and network segment. |
| `attach_resource_policy(resource_id, policy_id, actor_id)` | Attach a policy to a resource and mark it active. |
| `list_resources(tenant_id)` | List all resources, optionally scoped to a tenant. |

### Access Requests

| Method | Description |
|---|---|
| `request_access(identity_id, device_id, resource_id, ...)` | Evaluate and record an access request. Returns approved, denied, or review_required. |
| `approve_access_request(request_id, reviewer_id)` | Approve a pending access request (reviewer must differ from requester). |
| `list_access_requests(tenant_id)` | List all access requests, optionally scoped to a tenant. |
| `async_request_access(...)` | Async variant suitable for concurrent broker fan-out patterns. |

### Session Lifecycle

| Method | Description |
|---|---|
| `start_session(request_id, actor_id)` | Start a session from an approved access request. |
| `reevaluate_session(session_id, risk_score, ...)` | Reevaluate a session against current risk, identity, and device signals. |
| `close_session(session_id, actor_id)` | Close an active session. |
| `list_sessions(tenant_id)` | List all sessions, optionally scoped to a tenant. |
| `async_reevaluate_session(...)` | Async session reevaluation for telemetry pipeline triggers. |
| `async_bulk_reevaluate_sessions(tenant_id, risk_score)` | Re-evaluate all active tenant sessions concurrently via `asyncio.gather`. |
| `async_close_session(session_id, actor_id)` | Async session close for event-driven lifecycle handlers. |

### Policy Evaluation

| Method | Description |
|---|---|
| `evaluate(context)` | Run the rule engine against an arbitrary context dict. |
| `async_evaluate_policy(identity_id, resource_id, action, ...)` | Async policy evaluation for identity-resource-action triples without mutating session state. Returns matched rules and deny reasons. |

### Analytics and Reporting

| Method | Description |
|---|---|
| `dashboard_summary(tenant_id)` | Aggregated counts for identities, devices, resources, sessions, and agents. |
| `async_compliance_snapshot(tenant_id, actor_id)` | Async compliance snapshot for SIEM export — summary + device posture detail. |

### Agent and Lifecycle Management

| Method | Description |
|---|---|
| `register_zero_trust_agent(...)` | Register an AI agent with runtime, role, scope, owner, and purpose. |
| `validate_ztna_lifecycle_batch(...)` | Validate a Bytewax lifecycle batch before execution. |
| `list_zero_trust_agents(tenant_id)` | List all registered zero-trust agents. |
| `list_lifecycle_batches(tenant_id)` | List all lifecycle batches. |

### Audit

| Method | Description |
|---|---|
| `list_audit_events(tenant_id)` | Return all audit events, optionally scoped to a tenant. |

---

## Quick Start

```python
from capabilities.common.ztna.service import ZtnaService

service = ZtnaService()

# 1. Register and verify an identity
identity = service.register_identity(
	identity_key="analyst",
	tenant_id="tenant-a",
	subject_id="user-1",
	display_name="Analyst",
	verified=True,
	mfa_completed=True,
)

# 2. Register a managed device
device = service.register_device(
	device_key="laptop",
	tenant_id="tenant-a",
	identity_id=identity["id"],
	name="Managed Laptop",
	trust_score=0.92,
	managed=True,
	attested=True,
)

# 3. Register a resource and attach a policy
resource = service.register_resource(
	resource_key="crm",
	tenant_id="tenant-a",
	name="CRM Console",
	policy_attached=True,
	policy_id="crm-policy-v1",
)

# 4. Request access
request = service.request_access(
	identity["id"], device["id"], resource["id"],
	requested_by="user-1",
)
assert request["status"] == "approved"

# 5. Start a session
session = service.start_session(request["id"], actor_id="access-broker")

# 6. Reevaluate when risk changes
updated = service.reevaluate_session(session["id"], risk_score=0.35)

# 7. Close the session
service.close_session(session["id"], actor_id="access-broker")
```

---

## Async Usage

All async methods delegate to their sync counterparts and are safe to `await` in `asyncio`-based adapters, FastAPI route handlers, or concurrent fan-out patterns. No additional locking is required for single-process usage.

```python
import asyncio
from capabilities.common.ztna.service import ZtnaService

service = ZtnaService()

async def main():
	# Async identity registration
	identity = await service.async_register_identity(
		identity_key="svc-acct",
		tenant_id="tenant-b",
		subject_id="svc-1",
		display_name="Payment Service",
		verified=True,
		mfa_completed=True,
	)

	# Async policy evaluation — read-only, no session mutation
	decision = await service.async_evaluate_policy(
		identity_id=identity["id"],
		resource_id=resource_id,
		action="invoke",
	)
	if not decision["allowed"]:
		raise PermissionError(decision["deny_reasons"])

	# Bulk session reevaluation after a tenant-wide risk event
	results = await service.async_bulk_reevaluate_sessions(
		tenant_id="tenant-b",
		risk_score=0.7,
		actor_id="risk-engine",
	)

	# Compliance snapshot for SIEM export
	snapshot = await service.async_compliance_snapshot("tenant-b")
	print(snapshot["summary"])
	print(snapshot["posture"])

asyncio.run(main())
```

---

## Privileged Access

Privileged resource access enforces stricter guardrails:

- The identity must be verified.
- MFA must be complete.
- The resource must have a policy attached.
- The request must carry independent review or explicit just-in-time approval.
- Unmanaged privileged device access is routed to review.
- The reviewer cannot be the requester.

```python
# Privileged access request — routes to review_required
request = service.request_access(
	identity_id=privileged_identity_id,
	device_id=device_id,
	resource_id=privileged_resource_id,
	requested_by="admin-1",
	mfa_completed=True,
)
assert request["status"] == "review_required"

# A different actor must approve
approved = service.approve_access_request(request["id"], reviewer_id="reviewer-1")
assert approved["status"] == "approved"
```

---

## Microsegmentation

Resources carry a `network_segment` label that the rule engine uses to enforce
`microsegmentation_present` policy conditions. Register resources in named
segments to express network topology boundaries:

```python
resource = service.register_resource(
	resource_key="payments-db",
	tenant_id="tenant-a",
	name="Payments Database",
	network_segment="payments-dmz",
	sensitive=True,
	policy_attached=True,
	policy_id="payments-db-policy",
	access_level="privileged",
)
```

---

## AI Agent Composition

Zero-trust agents are first-class records, provider-neutral, and governed by the
same tenant isolation and audit trail as all other ZTNA entities.

```python
agent = service.register_zero_trust_agent(
	agent_id="zt-steward-1",
	tenant_id="tenant-a",
	name="Zero Trust Steward",
	runtime="claude_code",
	role="zero_trust_steward",
	scope="tenant:tenant-a",
	owner="security-platform",
	purpose="Review privileged access lifecycle batches",
	human_approval_required=True,
)
# Privileged role → status="pending_review" until human approves
```

Supported roles: `policy_reviewer`, `identity_context_reviewer`,
`device_posture_reviewer`, `resource_access_reviewer`, `session_risk_reviewer`,
`segmentation_reviewer`, `access_review_reviewer`, `lifecycle_batch_reviewer`,
`zero_trust_steward`.

---

## Lifecycle Batches

Bytewax-governed batches allow bulk zero-trust mutations with policy-controlled
accept/deny decisions. Only `bytewax` event stream is accepted; broker-specific
queues are denied.

```python
batch = service.validate_ztna_lifecycle_batch(
	tenant_id="tenant-a",
	event_stream="bytewax",
	mutation_count=5,
	operation="device_posture_batch",
)
assert batch["status"] == "accepted"
```

Supported operations: `identity_batch`, `device_posture_batch`,
`resource_batch`, `access_request_batch`, `session_batch`, `review_batch`,
`policy_batch`, `ztna_agent_batch`.

---

## Contract and Composition

```python
from capabilities.common.ztna.capability_contract import get_capability_contract

contract = get_capability_contract("tenant-a")
adapters = contract["configuration"]["adapters"]
rules = contract["rule_engine"]["rules"]
routes = contract["ui"]["routes"]
```

Key adapters: `generated_app_runtime` → `service.ZtnaService`,
`event_stream` → `bytewax`, `audit_sink` → `audl`,
`mfa_provider` → `mfau`, `identity_federation` → `idfd`,
`anomaly_detection` → `anom`.

---

## Configuration

All configuration keys are tenant-scoped. Override via the `conf` capability or
environment variables prefixed with `ZTNA_`.

---

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/ztna/service.py
./.venv/bin/pytest -q capabilities/common/ztna/tests/
./.venv/bin/python -c "from capabilities.common.ztna import app; r=app.self_test(); assert r['passed']"
```

---

## Further Reading

- `service.py` — Business logic and async methods
- `zero_trust_runtime.py` — Dataclass records and deterministic helpers
- `capability_contract.py` — Rule engine, UI routes, adapter contracts
- `models.py` — SQLAlchemy models
- `api.py` — REST API surface
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 strategic improvements roadmap
