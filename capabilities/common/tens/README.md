# TENS - Tenants Legacy

TENS is the APG capability for legacy tenant compatibility and migration governance. It provides a composable runtime for legacy tenant registration, APG tenant mapping, access-boundary validation, migration approval, migration completion, deprecation planning, AI-assisted review, and Bytewax lifecycle events.

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
- Async lifecycle operations: archive, restore, merge, clone, suspend, reactivate, health check, export/import, subdomain assignment, resource quotas, usage reports, billing summaries, migration summary, and full-text search.

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
- `tenant_migration_queued`
- `tenant_archived`
- `tenant_restored`
- `tenants_merged`
- `data_isolation_verified`
- `subdomain_assigned`
- `tenant_suspended`
- `tenant_reactivated`
- `resource_quota_updated`

## World-Class Enhancements (v2.0)

1. **Async-First Service Layer** — All new methods are `async def`; slots cleanly into FastAPI, Starlette, and async Bytewax pipelines.
2. **Persistent Storage via SQLAlchemy Async ORM** — Replace in-memory dicts with `sqlalchemy.ext.asyncio` sessions backed by PostgreSQL; Alembic migrations for schema versioning.
3. **StorageBackend Protocol / Adapter Pattern** — `StorageBackend` protocol with `get`, `put`, `delete`, `list`, `query`; `InMemoryBackend` and `PostgresBackend` ships out of box; test doubles without mocks.
4. **Structured Event Publishing via CloudEvents + Bytewax** — Typed `TensEvent` CloudEvents envelope published to a configurable `EventSink` (in-memory queue, Bytewax topic, Redis Streams); full CloudEvents 1.0 fields.
5. **Pydantic v2 Input/Output Contracts** — Every public method wrapped with typed `Request`/`Response` models; eliminates `dict[str, Any]` surface, gives runtime validation, and auto-generates OpenAPI schemas.
6. **Deterministic Idempotency Keys** — Mutating operations accept optional `idempotency_key: str`; replay returns stored result rather than raising or duplicating.
7. **Tenant Lifecycle State Machine** — Explicit FSM enforcing valid transitions (`active → stale → mapped → migration_ready → migrated → deprecated → archived → restored → suspended → merged`); raises `InvalidTransitionError` on illegal moves.
8. **Bulk / Batch Operations with Partial Failure Reporting** — Batch variants for `register_legacy_tenant`, `map_tenant`, `validate_access_boundary`; returns `BatchResult` with `succeeded`, `failed`, and `summary` counters.
9. **Optimistic Locking / ETag Concurrency Control** — `version: int` on every persistent record; mutating methods accept `expected_version`; raises `ConcurrentModificationError` on conflict.
10. **Tenant Compliance Report** — `async compliance_report(tenant_id, framework="SOC2")` evaluates a tenant against a configurable checklist; returns `ComplianceReport` with per-control pass/fail and posture score.
11. **Cross-Tenant Dependency Graph** — `async dependency_graph(tenant_id)` traverses mappings, boundaries, migrations, and merge records to produce an adjacency list and topological sort for migration sequencing.
12. **Tenant Activity Scoring Model** — Configurable `ActivityScorer` replacing the binary `days_since_activity > 90` heuristic; weights API calls, event frequency, mapping age, and boundary recency into a 0–100 score.
13. **Audit Log Integrity with HMAC Chaining** — Each audit event stores `prev_hash` = HMAC-SHA256 of the previous event; `verify_audit_chain(tenant_id)` returns the first broken link; tamper-evident without a separate ledger service.
14. **Observability: Structured Logging + OpenTelemetry Spans** — `@otel_span("tens.<method>")` decorator emits spans with `tenant_id`, `operation`, and result status; structured JSON logs at DEBUG/INFO/WARNING; zero-cost without a configured tracer.
15. **Rate Limiting and Quota Enforcement Middleware** — `QuotaEnforcer` checks `max_api_calls` per tenant per rolling window before every mutating operation; raises `QuotaExceededError` with `limit`, `used`, `reset_at`; pluggable, default uses in-memory sliding windows.

## New Methods

All new methods are `async`. Await them inside an async context.

### tenant_health_check

```python
health = await service.tenant_health_check(
	tenant_id="tenant-a",
	legacy_tenant_id="legacy-001",
)
# {"health_status": "healthy", "has_mapping": True, "has_boundary": True, "is_stale": False, ...}
```

### tenant_suspend / tenant_reactivate

```python
suspension = await service.tenant_suspend(
	tenant_id="tenant-a",
	legacy_tenant_id="legacy-001",
	reason="compliance hold",
	actor="compliance-officer",
)

reactivation = await service.tenant_reactivate(
	tenant_id="tenant-a",
	legacy_tenant_id="legacy-001",
	actor="compliance-officer",
	reactivation_note="hold lifted after audit",
)
```

### tenant_archive / tenant_restore

```python
archive = await service.tenant_archive(
	tenant_id="tenant-a",
	legacy_tenant_id="legacy-001",
	archive_ref="s3://archives/legacy-001",
	actor="ops-lead",
)

restore = await service.tenant_restore(
	tenant_id="tenant-a",
	legacy_tenant_id="legacy-001",
	restore_from_ref="s3://archives/legacy-001",
	actor="ops-lead",
)
```

### tenant_merge

```python
merge = await service.tenant_merge(
	tenant_id="tenant-a",
	source_tenant_id="legacy-001",
	target_tenant_id="legacy-002",
	merge_strategy="union",
	actor="migration-lead",
)
# source status transitions to "merged"
```

### tenant_search / audit_search

```python
results = await service.tenant_search(
	tenant_id="tenant-a",
	query="erp",
	status_filter="active",
)

events = await service.audit_search(
	tenant_id="tenant-a",
	event_type_filter="boundary",
	actor_filter="security-lead",
)
```

### usage_report / resource_quota

```python
report = await service.usage_report(
	tenant_id="tenant-a",
	legacy_tenant_id="legacy-001",
	period_start="2026-01-01",
	period_end="2026-03-31",
)

quota = await service.resource_quota(
	tenant_id="tenant-a",
	legacy_tenant_id="legacy-001",
	quotas={"max_api_calls": 10000, "max_storage_mb": 500, "max_users": 50},
	actor="system",
)
```

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
