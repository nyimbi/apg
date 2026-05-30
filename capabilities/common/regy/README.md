# APG API/Service Registry (REGY)

REGY is APG's governed API and service registry. It lets generated
applications register services and instances, discover healthy endpoints,
govern API versions, publish registry evidence to gateway adapters, retire
services with impact evidence, and expose audit-ready registry decisions.

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
- Deterministic rule decisions that return `allow`, `deny`, or
  `require_review`.
- UI view models for dashboard, catalog, registration, discovery, instances,
  health, versions, contract reviews, gateway sync, retirements, audit, and
  settings.
- Contract-derived semantic model, package manifest, and release evidence.

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
```

## UI Composition

```python
from capabilities.common.regy.registry_runtime import RegistryService
from capabilities.common.regy.view_models import dashboard_model, service_catalog_model

registry = RegistryService()
dashboard = dashboard_model(registry, "tenant-a")
catalog = service_catalog_model(registry, "tenant-a")
```

## Production Runtime Boundary

The generated-app control plane does not run a live service mesh, ingress
controller, cache cluster, Bytewax flow, APG gateway, external monitor, or
audit sink. Production deployments bind adapters for auth, configuration,
monitoring, audit, cache, gateway synchronization, and Bytewax event streaming.
Those adapters must honor REGY guardrail decisions before side effects.

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
./.venv/bin/apg capabilities publish-plan capabilities/common/regy --json
```
