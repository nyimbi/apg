# AUTH - Authentication And RBAC

AUTH is the APG identity, session, role, access-decision, privacy-budget, and
security-agent governance capability. It gives generated applications a
dependency-light control plane for registering tenant identities, defining
roles, approving privileged assignments, starting risk-aware sessions,
evaluating access, governing privacy-budget analytics, and composing AI
security agents into review workflows.

The package is intentionally usable without the production Flask API,
Flask-AppBuilder views, biometric engines, behavioral engines, cryptographic
providers, federated mesh services, durable databases, or live Bytewax workers.
Those systems attach through adapters after the capability contract and local
service have validated tenant boundaries, lifecycle state, rule evidence, and
audit surfaces.

## What AUTH Provides

- Tenant-qualified identity, role, role-approval, assignment, session, access,
  privacy-query, privacy-approval, security-agent, and audit-event stores.
- Privileged role assignment lifecycle with requester, justification, reviewer,
  decision, notes, and matching approval evidence.
- Privacy-budget override lifecycle with independent review before analytics
  can continue after budget exhaustion.
- Access decisions that combine role permissions, session evidence, MFA,
  federation trust, tenant membership, risk level, and inferred privileged
  tiers.
- Security-agent registration for `codex`, `claude_code`, `opencode`, and `pi`
  runtimes with explicit role, scope, disclosure, and policy evidence.
- Bytewax lifecycle stream metadata for batch AUTH mutation and generated
  application composition.
- API-helper and view-model modules for generated APG Python applications.
- UI route metadata for login, dashboard, roles, approvals, sessions, access,
  privacy, federation, security agents, audit, analytics, and metrics.
- Theme tokens and component metadata for compact security operations screens.

## Package Structure

- `SPECIFICATION.md` defines the current functional contract and acceptance
  criteria.
- `PLAN.md` records the implementation and review sequence for this packet.
- `cap_spec.md` points older tooling to the active specification.
- `capability_contract.py` declares configuration, rules, UI routes, theme,
  provides/requires metadata, and Bytewax stream metadata.
- `models.py` defines dependency-light AUTH records.
- `service.py` implements the executable lifecycle.
- `api_helpers.py` exposes generated-application call helpers.
- `view_models.py` exposes route-ready UI state.
- `app.py`, `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` provide package publication evidence.
- `tests/test_capability_contract.py` and `tests/test_package_contract.py`
  provide focused verification.

## Basic Usage

```python
from capabilities.common.auth.service import AuthService

service = AuthService()
tenant_id = "tenant-auth"

identity = service.register_identity(
    identity_id="alice",
    tenant_id=tenant_id,
    username="alice",
    email="alice@example.com",
    mfa_enabled=True,
    tenant_memberships=[tenant_id],
    privacy_budget=3.0,
)

role = service.define_role(
    role_id="admin",
    tenant_id=tenant_id,
    name="Administrator",
    permissions=["auth:admin", "auth:manage_roles", "auth:approve_roles"],
    tier="admin",
)

approval = service.request_role_assignment_approval(
    request_id="approve-alice-admin",
    tenant_id=tenant_id,
    identity_id=identity["id"],
    role_id=role["id"],
    requested_by="system",
    justification="Initial tenant administrator.",
)

service.decide_role_assignment_approval(
    request_id=approval["id"],
    tenant_id=tenant_id,
    reviewer="system",
    decision="approved",
    notes="Bootstrap administrator accepted.",
)

assignment = service.assign_role(
    assignment_id="alice-admin",
    tenant_id=tenant_id,
    identity_id=identity["id"],
    role_id=role["id"],
    assigned_by="system",
    approval_id=approval["id"],
)

assert assignment["role_id"] == "admin"
```

## Security-Agent Governance

AUTH treats AI security agents as governed participants, not hidden automation.
Agents must declare a supported runtime, supported role, explicit scope,
registration state, and contribution disclosure before they can appear in AUTH
review workflows.

```python
agent = service.register_security_agent(
    agent_id="role-review-agent",
    tenant_id=tenant_id,
    name="Role Review Agent",
    runtime="claude-code",
    role="role-reviewer",
    scope="Summarize privileged role requests for human reviewers.",
    contribution_disclosed=True,
    policy_ref="auth-agent-policy",
)

assert agent["runtime"] == "claude_code"
assert agent["role"] == "role_reviewer"
```

## Bytewax Guardrail

Batch AUTH mutation must be routed through the declared Bytewax lifecycle
stream. The dependency-light service validates the declared stream provider
before accepting batch mutation intent.

```python
service.validate_batch_auth_mutation(
    tenant_id=tenant_id,
    event_stream="bytewax",
    mutation_count=2,
)
```

## Composition Contract

`get_capability_contract()` returns the executable APG contract:

- `provides`: identity registry, role governance, session control, access
  decisions, privacy-budget governance, and security agents.
- `requires`: AUDL, MTEN, KEYM, and SECU.
- `configuration`: identity, role, session, federation, privacy,
  security-agent, governance, observability, adapter, UI, and theme settings.
- `rule_engine`: deterministic guardrails for identity, access, privacy,
  approval, security-agent, audit, tenant, and Bytewax lifecycle decisions.
- `ui`: route metadata for generated APG Python applications.
- `theme`: visual tokens and component metadata.
- `streaming`: Bytewax processor, topic, state collections, lifecycle events,
  and batch mutation guardrail.

## Verification

Focused checks for this package:

```bash
./.venv/bin/python -m py_compile capabilities/common/auth/models.py capabilities/common/auth/service.py capabilities/common/auth/api_helpers.py capabilities/common/auth/view_models.py capabilities/common/auth/capability_contract.py capabilities/common/auth/app.py capabilities/common/auth/tests/test_capability_contract.py capabilities/common/auth/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/auth/tests/test_capability_contract.py capabilities/common/auth/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.auth import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/auth --json
./.venv/bin/apg capabilities publish-plan capabilities/common/auth --json
```

Full repository suites, live web adapters, live identity providers, production
cryptographic engines, biometric capture, behavioral models, rendered browser
UI, live Bytewax workers, and load tests are separate integration concerns.
