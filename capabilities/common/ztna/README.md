# Zero Trust Network Access

`ztna` is APG's package-backed Zero Trust Network Access capability. It gives
generated applications a tenant-scoped access broker for identity, device
posture, protected resources, access requests, access reviews, governed
sessions, risk reevaluation, audit events, UI route metadata, and visual theme
metadata.

The package is intentionally dependency-light. It proves the lifecycle and
guardrails locally, while live identity providers, endpoint posture collectors,
MFA providers, policy engines, service meshes, gateways, audit sinks, and event
streams remain APG adapter boundaries.

## What It Provides

- Verified and privileged identity context.
- Device posture, trust score, management, attestation, compliance, and
  quarantine state.
- Protected resource registration with access level, sensitivity, policy
  attachment, and network segment metadata.
- Deterministic zero-trust access decisions.
- MFA and independent review guardrails for privileged access.
- High-risk access review and session reauthentication.
- Tenant isolation for all identity, device, resource, request, session, and
  audit records.
- Audit events for zero-trust state changes.
- UI view models for dashboard, identities, devices, resources, policies,
  access, sessions, risk, reviews, audit, and settings.
- Contract-derived semantic model, package manifest, release report, and
  publish-plan support.

## Main Files

| File | Purpose |
| --- | --- |
| `SPECIFICATION.md` | Functional, lifecycle, rule, UI, adapter, and acceptance specification. |
| `PLAN.md` | Implementation and review plan for this capability packet. |
| `capability_contract.py` | Executable configuration, rule engine, UI routes, theme, and adapter contract. |
| `zero_trust_runtime.py` | Dataclass records and deterministic ID/score helpers. |
| `service.py` | In-memory capability runtime and guardrail enforcement. |
| `api.py` | Dependency-light helper surface for generated applications. |
| `views.py` | View-model payloads for generated APG UIs. |
| `app.py` | Package entrypoint, semantic model, component manifest, and self-test. |

## Runtime Flow

1. Register or verify an identity.
2. Register a device for that identity with posture, trust, compliance, and
   attestation signals.
3. Register a protected resource and attach a resource policy.
4. Request access.
5. The rule engine allows, denies, or routes the request for review.
6. Approved requests can start sessions.
7. Sessions can be reevaluated as risk changes.
8. Sessions can be closed or revoked, with audit events recorded throughout.

## Python Usage

```python
from capabilities.common.ztna.service import ZtnaService

service = ZtnaService()

identity = service.register_identity(
	identity_key="analyst",
	tenant_id="tenant-a",
	subject_id="user-1",
	display_name="Analyst",
	verified=True,
)
device = service.register_device(
	device_key="laptop",
	tenant_id="tenant-a",
	identity_id=identity["id"],
	name="Managed Laptop",
	trust_score=0.94,
	managed=True,
	attested=True,
)
resource = service.register_resource(
	resource_key="crm",
	tenant_id="tenant-a",
	name="CRM Console",
	policy_attached=True,
	policy_id="crm-policy",
)

request = service.request_access(
	identity["id"],
	device["id"],
	resource["id"],
	requested_by="user-1",
)
session = service.start_session(request["id"], actor_id="access-broker")
```

## Privileged Access

Privileged resource access is stricter than standard access:

- the identity must be verified;
- MFA must be complete;
- the resource must have a policy;
- the request must have independent review or explicit just-in-time approval;
- unmanaged privileged device access is routed to review;
- the reviewer cannot be the requester.

```python
request = service.request_access(
	identity_id,
	device_id,
	privileged_resource_id,
	requested_by="admin-1",
	mfa_completed=True,
)
assert request["status"] == "review_required"

approved = service.approve_access_request(request["id"], reviewer_id="reviewer-1")
```

## Contract And Composition

Use the capability contract when composing APG applications:

```python
from capabilities.common.ztna.capability_contract import get_capability_contract

contract = get_capability_contract("tenant-a")
routes = contract["ui"]["routes"]
rules = contract["rule_engine"]["rules"]
adapters = contract["configuration"]["adapters"]
```

Important adapters:

- `generated_app_runtime`: `service.ZtnaService`
- `helper_runtime`: `zero_trust_runtime.py`
- `api_helpers`: `api.py`
- `view_models`: `views.py`
- `event_stream`: `bytewax`
- `authentication`: `auth`
- `mfa_provider`: `mfau`
- `audit_sink`: `audl`
- `identity_federation`: `idfd`
- `anomaly_detection`: `anom`

## UI Surfaces

The contract exposes these route names:

- `dashboard`
- `policies`
- `identities`
- `devices`
- `resources`
- `access`
- `sessions`
- `risk`
- `reviews`
- `audit`
- `settings`

`views.py` returns data-only models for these screens. The generated UI should
render the provided theme tokens and component names instead of hard-coding
colors or layout assumptions.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/ztna/__init__.py capabilities/common/ztna/capability_contract.py capabilities/common/ztna/zero_trust_runtime.py capabilities/common/ztna/models.py capabilities/common/ztna/service.py capabilities/common/ztna/api.py capabilities/common/ztna/views.py capabilities/common/ztna/app.py capabilities/common/ztna/test_capability_contract.py capabilities/common/ztna/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/ztna/test_capability_contract.py capabilities/common/ztna/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.ztna import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/ztna --json
./.venv/bin/apg capabilities publish-plan capabilities/common/ztna --json
```

Full repository audits are intentionally separate so focused capability work
can move quickly on battery.
