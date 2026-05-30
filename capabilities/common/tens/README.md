# TENS - Tenants Legacy

TENS is the APG capability for legacy tenant compatibility and migration governance. It gives generated applications a composable runtime for legacy tenant registration, APG tenant mapping, access-boundary validation, migration approval, migration completion, deprecation planning, AI-assisted review, and Bytewax lifecycle events.

Use TENS when an application must keep legacy tenant data, identity, roles, or compatibility boundaries under control while moving to APG multi-tenant applications.

## What It Provides

- Legacy tenant registry with source-system lineage, owner, compatibility scope, activity age, and lifecycle status.
- Tenant mapping from legacy tenant IDs to APG tenant IDs with validation evidence.
- Access boundary validation with auth boundary, role mapping, tenant isolation, and privileged access review evidence.
- Migration plans with approval, rollback, and post-migration validation.
- Migration completion with Bytewax lifecycle routing.
- Deprecation planning for retiring legacy tenant compatibility.
- First-class TENS agents for Codex, Claude Code, OpenCode, and Pi based review lanes.
- APG Python UI view models for dashboard, registry, mappings, migrations, boundaries, deprecation, agents, policy, audit, and settings.
- Visual theme tokens for legacy tenant migration screens.
- Bytewax stream metadata for lifecycle events.

## Core Runtime

```python
from capabilities.common.tens import TensService

service = TensService()

legacy = service.register_legacy_tenant(
	tenant_id="tenant-a",
	legacy_tenant_id="legacy-001",
	source_system="erp-legacy",
	owner="tenant-owner",
	compatibility_scope="finance",
)

mapping = service.map_tenant(
	tenant_id="tenant-a",
	legacy_tenant_id=legacy["id"],
	apg_tenant_id="apg-tenant-001",
	validated_by="migration-lead",
	validation_ref="validation://mapping/1",
)

service.validate_access_boundary(
	tenant_id="tenant-a",
	legacy_tenant_id=legacy["id"],
	auth_boundary_ref="auth://boundary/1",
	role_mapping_ref="roles://mapping/1",
	isolation_validation_ref="isolation://tenant/1",
	privileged_review_ref="review://privileged/1",
	actor="security-lead",
)

migration = service.create_migration_plan(
	tenant_id="tenant-a",
	legacy_tenant_id=legacy["id"],
	mapping_id=mapping["id"],
	owner="migration-lead",
	approval_ref="approval://migration/1",
	rollback_plan_ref="rollback://tenant/1",
	post_migration_validation_ref="validation://post/1",
)

service.complete_migration(
	tenant_id="tenant-a",
	migration_id=migration["id"],
	actor="migration-lead",
	post_migration_validation_ref="validation://post/complete",
)
```

## AI Agent Composition

TENS treats tenant migration agents as governed composition elements.

```python
agent = service.register_tens_agent(
	tenant_id="tenant-a",
	name="Boundary reviewer",
	runtime="codex",
	role="boundary_reviewer",
	scope="review tenant isolation and privileged access evidence",
	owner="security-lead",
)

decision = service.validate_agent_tenant_action(
	tenant_id="tenant-a",
	agent_id=agent["id"],
	privileged_scope=True,
	human_approval_recorded=False,
)

assert decision["decision"] == "deny"
```

Supported runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles:

- `tenant_mapper`
- `boundary_reviewer`
- `migration_reviewer`
- `deprecation_reviewer`
- `compatibility_reviewer`
- `audit_reviewer`

## Rule Engine

The deterministic rule engine protects tenant migration operations:

- tenant context is mandatory;
- legacy tenant registration requires owner, source system, and compatibility scope;
- mappings require validation and Bytewax event routing;
- migrations require approval, rollback, post-validation, and Bytewax event routing;
- access boundaries require auth, role mapping, tenant isolation, and privileged review evidence;
- stale legacy tenants require review;
- agents require supported runtime and role;
- privileged agent-driven tenant actions require human approval;
- batch tenant mapping requires Bytewax coordination.

Rules are exposed through `evaluate_capability_rules()` and `TensService.evaluate()`.

## UI Surfaces

`views.py` exposes route-backed models for:

- dashboard: `/tens/dashboard`
- legacy tenant registry: `/tens/tenants`
- mapping workbench: `/tens/mappings`
- migration queue: `/tens/migrations`
- boundary review: `/tens/boundaries`
- deprecation: `/tens/deprecation`
- agents: `/tens/agents`
- policy: `/tens/policy`
- audit: `/tens/audit`
- settings: `/tens/settings`

These models are framework-neutral so APG generated Python applications can compose them into their UI shell.

## Event Stream

TENS publishes lifecycle metadata for Bytewax:

- processor: `bytewax`
- stream: `apg.tens.lifecycle`
- key: `tenant_id`

Events:

- `legacy_tenant_registered`
- `tenant_mapped`
- `boundary_validated`
- `migration_plan_created`
- `migration_completed`
- `deprecation_planned`
- `tens_agent_registered`

## Adapter Boundaries

The package does not directly call live identity providers, tenant catalogs, role stores, legacy directories, migration engines, approval systems, or audit sinks. Add those integrations as adapters around the stable service methods and stream metadata.

## Verification

Battery-conscious package verification:

```bash
./.venv/bin/python -m py_compile capabilities/common/tens/__init__.py capabilities/common/tens/capability_contract.py capabilities/common/tens/models.py capabilities/common/tens/tenant_runtime.py capabilities/common/tens/service.py capabilities/common/tens/api.py capabilities/common/tens/views.py capabilities/common/tens/app.py capabilities/common/tens/test_capability_contract.py capabilities/common/tens/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/tens/test_capability_contract.py capabilities/common/tens/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/tens --json
./.venv/bin/apg capabilities publish-plan capabilities/common/tens --json
```

Run broader checks only when battery and time allow.
