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
- Deterministic zero-trust access decisions with matched-rule traces.
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
- **Async-native methods** for identity registration, access requests, session
  management, policy evaluation, posture telemetry ingestion, bulk session
  reevaluation, and compliance snapshots — enabling safe use in async adapters
  and concurrent broker fan-out patterns.

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

1. Register or verify an identity (sync or `async_register_identity`).
2. Register a device for that identity with posture, trust, compliance, and
   attestation signals (sync or `async_update_device_posture` for continuous
   telemetry).
3. Register a protected resource and attach a resource policy.
4. Request access (sync or `async_request_access`).
5. The rule engine allows, denies, or routes the request for review.
6. Approved requests can start sessions.
7. Sessions can be reevaluated as risk changes (sync or
   `async_reevaluate_session` / `async_bulk_reevaluate_sessions`).
8. Policy decisions can be evaluated independently via `async_evaluate_policy`.
9. Sessions can be closed or revoked (sync or `async_close_session`), with
   audit events recorded throughout.
10. Tenant-level compliance snapshots are available via
    `async_compliance_snapshot` for SIEM export and dashboard polling.

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

### Async Usage

All async methods are safe to use in `asyncio`-based adapters, FastAPI
handlers, or concurrent broker fan-outs.

```python
import asyncio
from capabilities.common.ztna.service import ZtnaService

service = ZtnaService()

async def main():
	identity = await service.async_register_identity(
		identity_key="analyst",
		tenant_id="tenant-a",
		subject_id="user-1",
		display_name="Analyst",
		verified=True,
		mfa_completed=True,
	)

	# Evaluate policy without mutating session state
	decision = await service.async_evaluate_policy(
		identity_id=identity["id"],
		resource_id=resource["id"],
		action="read",
	)
	assert decision["allowed"]

	# Re-evaluate all active sessions after a risk signal
	results = await service.async_bulk_reevaluate_sessions(
		tenant_id="tenant-a",
		risk_score=0.6,
	)

	# Compliance snapshot for SIEM export
	snapshot = await service.async_compliance_snapshot("tenant-a")
	print(snapshot["summary"]["active_session_count"])

asyncio.run(main())
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
`event_stream="bytewax"`. Broker-specific queue or broker-core routing is intentionally denied.

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

## World-Class Enhancements (v2.0)

Fifteen improvements that elevate this capability from a functional prototype to
a production-grade zero-trust broker:

1. **Async-native service layer** — All service methods converted to `async`
   with `asyncio.Lock` per entity map, eliminating blocking I/O contention for
   concurrent policy decisions.

2. **Policy-as-code engine (REGO / CEL)** — Structured evaluator accepts REGO
   (OPA) or CEL expressions per rule with hot-reload, enabling ABAC,
   time-windowed access, and geo-fencing without redeployment.

3. **Persistent storage adapter interface** — `ZtnaStorageAdapter` protocol
   with `MemoryAdapter`, `PostgresAdapter` (SQLAlchemy async), and
   `RedisAdapter`. Zero-change swap at deploy time.

4. **Continuous posture telemetry pipeline** — Streaming channel (Bytewax /
   Bytewax) continuously updates `trust_score` and auto-triggers session
   reevaluation when score drops below tenant-configured thresholds.

5. **Risk-adaptive ML scoring** — Pluggable `RiskScoringAdapter` delegates to
   a local Ollama model, ONNX ensemble, or rules engine for calibrated
   probability estimates instead of heuristic linear sums.

6. **JIT privileged access vaults** — Time-boxed credentials (TOTP seeds,
   ephemeral API keys, SSH certificates via Vault/SPIFFE) issued on approval,
   auto-expired on session close or TTL. Eliminates standing privileged access.

7. **Cryptographic device attestation (TPM / SPIFFE / SVID)** —
   `DeviceAttestationAdapter` validates TPM 2.0 quotes and SPIFFE X.509 SVIDs
   against tenant CA; `attested` and `trust_score` update only on verified
   evidence.

8. **Micro-segmentation graph engine** — Directed graph of allowed lateral
   paths between segments; segment-to-segment firewall rules; deny-by-default
   enforced unless an explicit path policy exists.

9. **OIDC / SAML identity federation** — `FederationAdapter` validates JWTs
   and assertions from Entra, Okta, and PingFederate; maps groups/roles claims
   to ZTNA identity attributes via tenant-specific claim rules.

10. **Distributed CloudEvents audit trail** — Append-only `CloudEvent`-formatted
    records with `traceparent` and hash-linked tamper evidence written to
    PostgreSQL, OpenSearch, or S3.

11. **Zero-trust session proxy with mTLS** — Identity-aware reverse proxy
    terminates mTLS client certificates (SVID-validated), attaches signed
    `X-ZTNA-Session` headers, and enforces per-session rate limits with
    automatic TCP teardown on risk breach.

12. **Behavioral analytics and insider threat detection** — Per-identity
    baseline profiles (access hours, resource sets, session durations) via
    EWMS; flags impossible travel, off-hours privileged access, and abnormal
    data volumes as risk signals.

13. **Self-service access request portal** — Flask-AppBuilder blueprint for
    resource discovery, business-justified access requests, approval status
    tracking, and mobile-friendly approver review queue with contextual risk
    details.

14. **Tenant Zero Trust maturity scoring** — `ztna_maturity_score` evaluates a
    tenant against the CISA ZT Maturity Model tiers (Traditional → Advanced →
    Optimal) across five pillars (Identity, Devices, Networks, Applications,
    Data) with per-pillar remediation recommendations.

15. **Adversarial integration test harness** — Scenario-driven YAML fixtures
    covering privilege escalation, cross-tenant leaks, session hijack replays,
    posture downgrade attacks, and concurrent approval races with injected
    clocks and full matched-rule trace assertions.

## New Methods

The eight async methods added to `ZtnaService` in v2.0:

### `async_evaluate_policy` — stateless policy dry-run

```python
decision = await service.async_evaluate_policy(
	identity_id=identity["id"],
	resource_id=resource["id"],
	action="read",
)
# {"allowed": True, "decision": "allow", "matched_rules": [...], "deny_reasons": []}
```

Resolves identity and resource, runs the full rule engine, and returns an
enriched decision payload without mutating any session state. Use this in
authorization middleware or policy audit tools.

### `async_bulk_reevaluate_sessions` — fan-out reevaluation

```python
results = await service.async_bulk_reevaluate_sessions(
	tenant_id="tenant-a",
	risk_score=0.6,        # applies to all active sessions
	actor_id="risk-engine",
)
# list of per-session reevaluation dicts; uses asyncio.gather internally
```

Fan-out across all active sessions for a tenant in one call. Called immediately
after a tenant-level policy change or identity revocation.

### `async_update_device_posture` — continuous telemetry ingestion

```python
updated = await service.async_update_device_posture(
	device_id=device["id"],
	trust_score=0.72,
	posture_present=True,
	compliant=True,
	attested=True,
	actor_id="uem-connector",
)
```

Designed to be called from a streaming posture telemetry consumer (Bytewax,
Bytewax worker). Updates `trust_score` in-place and emits an audit event; pair
with `async_bulk_reevaluate_sessions` to close the posture-to-access-decision
loop.

### `async_compliance_snapshot` — SIEM export

```python
snapshot = await service.async_compliance_snapshot(
	tenant_id="tenant-a",
	actor_id="compliance-job",
)
# {
#   "tenant_id": "tenant-a",
#   "generated_at": "...",
#   "summary": {"active_session_count": 3, ...},
#   "posture": {"total": 5, "compliant": 4, "avg_trust": 0.91, "by_status": {...}}
# }
```

Aggregates identity verification, device posture, resource policy coverage,
session counts, and review backlog into one dict. Poll this on a schedule to
feed audit dashboards or SIEM pipelines.

### `async_close_session` — event-driven lifecycle

```python
closed = await service.async_close_session(
	session_id=session["id"],
	actor_id="session-gc",
)
```

For use in event-driven session lifecycle handlers (e.g. JIT vault expiry
callbacks, TCP teardown hooks from the session proxy).

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
