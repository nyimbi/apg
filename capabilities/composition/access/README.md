# Access Control Integration Hub

Access Control Integration Hub is APG's composition-layer access capability. It lets generated applications and composed capabilities share a single model for identity providers, protected resources, policies, grants, session risk, access decisions, audit evidence, AI-agent review, UI surfaces, visual theme tokens, and Bytewax lifecycle events.

## What It Provides

- Identity-provider composition for local, OIDC, SAML, LDAP, API-key, and JWT providers.
- Resource-access registry for APG capabilities, routes, screens, workflows, datasets, and composed application boundaries.
- Policy orchestration with sensitive-resource and high-risk activation guardrails.
- Grant lifecycle management with privileged approval, expiry, justification, and separation-of-duties checks.
- Session risk evaluation with adaptive step-up enforcement.
- Access-decision audit records routed through Bytewax.
- First-class access agents for Codex, Claude Code, OpenCode, and Pi.
- APG UI contracts for dashboard, providers, resources, policies, grants, decisions, sessions, agents, audit, and settings.
- Theme tokens and compact visual contracts for security-operation screens.

## Key Files

- `SPECIFICATION.md` defines the full functional contract.
- `PLAN.md` records the implementation and review plan.
- `cap_spec.md` is the APG capability specification consumed by tooling and humans.
- `capability_contract.py` exposes the executable APG contract and deterministic rule engine.
- `models.py` defines dependency-light runtime records.
- `service.py` implements the lifecycle operations and guardrails.
- `api.py` exposes dependency-light API helper functions.
- `views.py` exposes UI model helpers.
- `app.py` exposes package self-test, component manifest, and semantic model helpers.

## Basic Usage

```python
from capabilities.composition.access import CompositionAccessService

service = CompositionAccessService()

provider = service.register_provider(
	provider_key="corp-oidc",
	tenant_id="tenant-a",
	name="Corporate OIDC",
	provider_type="oidc",
	owner_id="security-owner",
)

provider = service.activate_provider(
	provider_id=provider["id"],
	actor_id="security-owner",
	secret_reference="vault://tenant-a/oidc/client",
	test_evidence="oidc-discovery-validated",
)

resource = service.register_resource(
	resource_key="erp.orders.approve",
	tenant_id="tenant-a",
	display_name="Approve Orders",
	owner_id="orders-owner",
	scopes=["read", "approve"],
	capability_id="erp_orders",
	sensitive=True,
)

policy = service.create_policy(
	policy_key="orders-approval-policy",
	tenant_id="tenant-a",
	name="Orders approval policy",
	resource_id=resource["id"],
	owner_id="security-owner",
	effect="allow",
	conditions={"department": "finance", "mfa": True},
	risk_level="high",
)

service.activate_policy(
	policy_id=policy["id"],
	actor_id="security-owner",
	simulation_evidence="simulation-run-2026-05-30",
	reviewed_by="risk-reviewer",
)

grant = service.create_grant(
	grant_key="grant-finance-approval",
	tenant_id="tenant-a",
	subject_id="user-123",
	resource_id=resource["id"],
	scopes=["approve"],
	requested_by="manager-1",
	justification="month-end order approval coverage",
	privileged=True,
	approved_by="security-owner",
	expires_at="2026-06-30T23:59:59+00:00",
)

decision = service.record_decision(
	decision_key="decision-1",
	tenant_id="tenant-a",
	subject_id="user-123",
	resource_id=resource["id"],
	action="approve",
	decision="allow",
	reason="active_grant_and_policy_match",
	policy_ids=[policy["id"]],
	event_stream="bytewax",
)
```

## Guardrail Examples

Privileged grants must include an independent approver, expiry, and justification:

```python
service.create_grant(
	grant_key="blocked",
	tenant_id="tenant-a",
	subject_id="user-123",
	resource_id=resource["id"],
	scopes=["approve"],
	requested_by="manager-1",
	justification="",
	privileged=True,
)
```

This raises `PermissionError` with the matched rule names.

High-risk sessions require step-up:

```python
service.evaluate_session(
	session_key="session-1",
	tenant_id="tenant-a",
	subject_id="user-123",
	provider_id=provider["id"],
	risk_score=91,
	step_up_completed=False,
)
```

This raises `PermissionError` for adaptive step-up.

## AI Agent Composition

Access agents are first-class records:

```python
agent = service.register_access_agent(
	tenant_id="tenant-a",
	name="Grant Review Agent",
	runtime="codex",
	role="grant_reviewer",
	instructions="Review privileged grants before approval.",
)

service.validate_agent_access_action(
	tenant_id="tenant-a",
	agent_id=agent["id"],
	action="recommend_privileged_grant",
	privileged_scope=True,
	human_approval_recorded=True,
)
```

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`.

## UI and Theming

Use `views.py` helpers to drive APG screens:

- `dashboard_model()`
- `provider_console_model()`
- `policy_studio_model()`
- `grant_workbench_model()`
- `decision_explorer_model()`
- `agent_workbench_model()`
- `audit_console_model()`

The theme contract uses compact surfaces, 8px radius, restrained operational colors, and component-specific visual roles for provider, policy, grant, decision, session, and agent screens.

## Verification

Focused checks for this package:

```bash
./.venv/bin/python -m py_compile capabilities/composition/access/__init__.py capabilities/composition/access/capability_contract.py capabilities/composition/access/models.py capabilities/composition/access/service.py capabilities/composition/access/api.py capabilities/composition/access/views.py capabilities/composition/access/app.py capabilities/composition/access/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/composition/access/tests/test_package_contract.py
```

Full repository checks are intentionally deferred during battery-conscious development unless the touched slice requires them.
