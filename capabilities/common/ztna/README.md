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
- First-class zero-trust AI-agent composition for policy, identity, device,
  resource, session-risk, segmentation, access-review, and lifecycle work.
- Bytewax lifecycle-batch validation for identity, device posture, resource,
  access, session, review, policy, and agent mutations.
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

agent = service.register_zero_trust_agent(
	agent_id="agent-zero-trust-steward",
	tenant_id="tenant-a",
	name="Zero Trust Steward",
	runtime="codex",
	role="zero_trust_steward",
	scope="tenant:tenant-a",
	owner="security-platform",
	purpose="review zero-trust lifecycle batches",
	human_approval_required=True,
)

batch = service.validate_ztna_lifecycle_batch(
	tenant_id="tenant-a",
	event_stream="bytewax",
	mutation_count=2,
	operation="ztna_agent_batch",
)
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
- `agent_adapter`: `aicr_provider_neutral_zero_trust_agent_adapter`

## Agent Composition

ZTNA agents are first-class records. They are provider-neutral and may use
`codex`, `claude_code`, `opencode`, or `pi` behind the AICR adapter contract.
Each agent requires tenant context, name, runtime, role, scope, owner, purpose,
and machine-contribution disclosure. Privileged roles such as resource access,
session risk, segmentation, access review, lifecycle batch, and zero-trust
steward agents enter `pending_review` unless human approval is recorded.

Supported roles:

- `policy_reviewer`
- `identity_context_reviewer`
- `device_posture_reviewer`
- `resource_access_reviewer`
- `session_risk_reviewer`
- `segmentation_reviewer`
- `access_review_reviewer`
- `lifecycle_batch_reviewer`
- `zero_trust_steward`

## Lifecycle Batches

ZTNA lifecycle batches are explicit Bytewax-governed records. The generated
runtime accepts `identity_batch`, `device_posture_batch`, `resource_batch`,
`access_request_batch`, `session_batch`, `review_batch`, `policy_batch`, and
`ztna_agent_batch` only when they contain mutations and declare
`event_stream="bytewax"`. Kafka or broker-core routing is intentionally denied.

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
- `agents`
- `lifecycle`
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
