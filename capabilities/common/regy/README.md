# APG API/Service Registry (REGY)

REGY is APG's governed API and service registry. It lets generated
applications register services and instances, discover healthy endpoints,
govern API versions, publish registry evidence to gateway adapters, retire
services with impact evidence, and expose audit-ready registry decisions.
REGY also treats AI and automation agents as governed registry participants,
so tools such as Codex, Claude Code, OpenCode, Pi, and future runtimes compose
through policy-controlled adapters instead of informal side channels.

REGY is intentionally split into two layers:

- `registry_runtime.RegistryService`: dependency-light generated-application
  lifecycle service used by package tests, API helpers, UI view models, and
  semantic evidence.
- production registry modules such as `service.py`, `api.py`, `views.py`, and
  enhancement modules: adapter-backed runtime surfaces for APG auth,
  configuration, monitoring, audit, cache, gateway synchronization, and richer
  operational integrations.

## What REGY Provides

- Tenant-scoped service registration with owner, version, schema, health, and
  routing metadata.
- Service instance registration with endpoint, region, health probe, weight,
  and health status.
- Service discovery that prefers healthy instances and blocks cross-tenant
  discovery by default.
- API/service version governance with compatibility review and deprecation
  evidence.
- Gateway publication guardrails for registered services, healthy instances,
  and routing metadata.
- Retirement guardrails requiring impact review and gateway unpublish evidence.
- First-class registry-agent composition with supported runtimes, role
  guardrails, bounded scope, accountable owner, purpose, contribution
  disclosure, and human approval for privileged registry roles.
- Bytewax lifecycle batch validation for registry mutation streams.
- Durable review evidence for generated-app governance queues, including
  policy decisions, matched rules, review reasons, required actions, and
  persisted denial evidence for non-Bytewax lifecycle batches.
- Deterministic rule decisions that return `allow`, `deny`, or
  `require_review`.
- UI view models for dashboard, catalog, registration, discovery, instances,
  health, versions, contract reviews, gateway sync, retirements, audit, and
  settings, plus registry-agent roster and lifecycle-batch monitor surfaces.
- Contract-derived semantic model, package manifest, and release evidence.

## World-Class Enhancements (v2.0)

1. **Persistent Storage Adapter** — async `get/put/delete/scan` contract backed by PostgreSQL, Redis, or etcd; in-memory shim for tests.
2. **Event Sourcing** — every mutation appends an immutable event log enabling audit reconstruction, replay, and time-travel queries.
3. **TTL Lease Expiry** — background coroutine marks crashed instances `STOPPED` and emits `lease_expired` when heartbeat TTL passes.
4. **Structured Health Probes** — concurrent HTTP/TCP probes via `aiohttp`/`asyncio.open_connection` against each instance's configured `health_checks`.
5. **Dependency-Graph Impact Analysis** — full BFS/DFS traversal of `dependencies` + reverse `dependents` lookup for cascade-impact reports.
6. **Circuit-Breaker Enforcement** — `discover_services` filters `OPEN`/`FORCED_OPEN` instances; new `instance_select` records metrics at selection.
7. **LRU-TTL Discovery Cache** — bounded `cachetools.TTLCache` replaces unbounded dict; mutations do targeted key eviction instead of full `clear()`.
8. **Async Pub/Sub Notifications** — `asyncio.Queue`-backed bus delivers filtered change streams to subscribers without polling.
9. **Semver Constraint Enforcement** — evaluates `version_constraints` ranges (e.g. `>=1.2.0,<2.0.0`) against registered versions at discovery time.
10. **Geographic Routing** — region-scoring function ranks instances by `metadata['region']` proximity for `GEOGRAPHIC` load-balance strategy.
11. **Federated Conflict Detection** — reconciliation step surfaces `FEDERATED_CONFLICT` events for version drift and schema divergence across tenants.
12. **Prometheus / OTLP Metrics Export** — counters and gauges emitted as OTLP spans and Prometheus metrics via `MonitoringService` adapter.
13. **Bulk Health Ingestion** — `bulk_update_health` accepts batched payloads, applies them transactionally, emits one reconciled event per changed service.
14. **Policy-Driven Discovery ACL** — `authorization_policies` wired through `AuthService` adapter with fast allow-list cache for namespace/tag-level restrictions.
15. **Automated Stale-Service GC** — configurable background coroutine deregisters ephemeral/test services past retention threshold, honouring impact-review guardrail.

## New Methods

Three high-impact async methods added in `service.py`:

### `capability_search` — find services by capability tag

```python
svc = RegistryService(tenant_id="tenant-a")
await svc.initialize()

# Find all services advertising "payments" in production
hits = await svc.capability_search(capability="payments", environment="production")
# [{"id": "...", "name": "payments-api", "namespace": "finance", "environment": "production"}]
```

### `federation_registry` — ingest a remote tenant's catalog

```python
remote_svcs = [{"name": "fx-rates", "service_type": "rest_api", "environment": "production"}]

result = await svc.federation_registry(
    remote_tenant_id="tenant-b",
    remote_services=remote_svcs,
    federated_by="federation-agent",
)
# {"remote_tenant_id": "tenant-b", "federated_count": 1, "federated_ids": ["..."]}
# Services appear under namespace "federated:tenant-b" with tags ["federated", "remote_tenant:tenant-b"]
```

### `dependency_graph` — inspect a service's declared dependencies

```python
result = await svc.dependency_graph(service_id="orders")
# {"service_id": "orders", "service_name": "orders-api", "dependencies": ["payments", "inventory"]}
```

## Important Files

- `SPECIFICATION.md`: current REGY functional specification.
- `PLAN.md`: lifecycle packet implementation plan.
- `capability_contract.py`: configuration, rules, adapters, UI, and theme.
- `registry_runtime.py`: dependency-light generated-app lifecycle service.
- `api.py`: Flask REST API plus generated-app helper functions.
- `view_models.py`: generated UI data models.
- `views.py`: legacy Flask-AppBuilder runtime views.
- `app.py`: publishable package entrypoint and semantic model generator.
- `test_capability_contract.py`: focused rule, lifecycle, API, and UI tests.
- `tests/test_package_contract.py`: package contract and app evidence tests.

## Generated-App Usage

```python
from capabilities.common.regy.registry_runtime import RegistryService

registry = RegistryService()

registry.register_service(
    service_id="orders",
    tenant_id="tenant-a",
    name="orders",
    owner="platform",
    service_type="rest_api",
    environment="production",
    api_version="1.0.0",
    contract_schema_ref="schemas/orders-openapi.yaml",
    health_endpoint="/health",
    routing_metadata={"path": "/orders", "strategy": "weighted"},
    production_review_recorded=True,
    trace_propagation_configured=True,
)

registry.register_instance(
    instance_id="orders-1",
    tenant_id="tenant-a",
    service_id="orders",
    endpoint="https://orders.internal",
    region="edge-africa",
    health_probe="/health",
    weight=100,
)

publication = registry.publish_to_gateway(
    publication_id="orders-public",
    tenant_id="tenant-a",
    service_id="orders",
    route_path="/orders",
)

assert publication["status"] == "published"

agent = registry.register_registry_agent(
    agent_id="catalog-agent",
    tenant_id="tenant-a",
    name="Catalog Agent",
    runtime="codex",
    role="catalog_steward",
    scope="catalog hygiene",
    owner="registry-office",
    purpose="maintain service catalog metadata",
)

batch = registry.validate_regy_lifecycle_batch(
    tenant_id="tenant-a",
    event_stream="bytewax",
    mutation_count=4,
)

assert agent["status"] == "active"
assert batch["status"] == "accepted"
```

## Review Evidence

Every generated-app lifecycle record carries `policy_decision`,
`matched_rules`, `review_reasons`, and `review_evidence` fields so generated
registry consoles can render why a service, version, discovery request,
owner-transfer, registry-agent registration, gateway publication, or lifecycle
batch is allowed, denied, or awaiting review. `list_pending_reviews()` returns
the composed queue across services, instances, versions, gateway publications,
reviews, registry agents, and lifecycle batches. Denied non-Bytewax lifecycle
batches are stored with `status="denied"` and `required_processor="bytewax"`
before the guardrail raises `PermissionError`.

## UI Composition

```python
from capabilities.common.regy.registry_runtime import RegistryService
from capabilities.common.regy.view_models import (
    dashboard_model,
    lifecycle_batch_model,
    registry_agent_roster_model,
    service_catalog_model,
)

registry = RegistryService()
dashboard = dashboard_model(registry, "tenant-a")
catalog = service_catalog_model(registry, "tenant-a")
agents = registry_agent_roster_model(registry, "tenant-a")
lifecycle = lifecycle_batch_model(registry, "tenant-a")
```

## Production Runtime Boundary

The generated-app control plane does not run a live service mesh, ingress
controller, cache cluster, Bytewax worker, APG gateway, external monitor,
audit sink, or vendor-specific agent runtime. Production deployments bind
adapters for auth, configuration, monitoring, audit, cache, gateway
synchronization, external AI runtimes, and Bytewax event streaming. Those
adapters must honor REGY guardrail decisions before side effects.

## Focused Verification

```bash
./.venv/bin/python -m py_compile \
  capabilities/common/regy/__init__.py \
  capabilities/common/regy/capability_contract.py \
  capabilities/common/regy/models.py \
  capabilities/common/regy/registry_runtime.py \
  capabilities/common/regy/api.py \
  capabilities/common/regy/view_models.py \
  capabilities/common/regy/app.py \
  capabilities/common/regy/test_capability_contract.py \
  capabilities/common/regy/tests/test_package_contract.py

./.venv/bin/pytest -q \
  capabilities/common/regy/test_capability_contract.py \
  capabilities/common/regy/tests/test_package_contract.py

./.venv/bin/apg capabilities implementation-audit --root capabilities/common/regy --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/regy --strict --json
./.venv/bin/apg capabilities publish-plan capabilities/common/regy --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/common/regy --json
```
