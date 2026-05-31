# APG API Gateway & Management (APIG)

APIG is APG's governed API gateway control plane. It lets generated
applications register upstream services and API consumers, request routes,
enforce security and traffic guardrails, review high quotas and canary traffic
shifts, gate deployments, retire routes, and publish audit evidence.

APIG is intentionally split into two layers:

- `gateway_runtime.ApigService`: dependency-light generated-application
  lifecycle service used by package tests, API helpers, view models, and
  semantic evidence.
- production gateway modules such as `service.py`, `edge_engine.py`,
  `traffic_manager.py`, and `wasm_runtime.py`: adapter-backed runtime surfaces
  for reverse proxies, edge execution, traffic engines, APG service clients,
  and AI/optimization providers.

## What APIG Provides

- Tenant-scoped upstream service registration.
- API consumer registration and credential governance.
- Route request, activation, and retirement lifecycle.
- Guardrails for tenant context, upstream owner, HTTPS, health checks,
  consumer owner, credential rotation, RBAC approval, route owner, absolute
  paths, registered upstreams, HTTP methods, public auth, external mTLS, unsafe
  method threat policy, route rate limits, signed edge filters, high quota
  review, canary review, canary limits, rollback plans, deployment region,
  observability, production approval, policy review, and impact review.
- API helper functions for generated applications.
- UI view models for dashboard, routes, upstreams, consumers, traffic,
  security, edge filters, quota reviews, canary releases, deployments,
  analytics, gateway agents, lifecycle batches, audit, and settings.
- First-class gateway-agent records for AI and automation tools.
- Bytewax lifecycle-batch validation before generated applications apply
  batched APIG state changes.
- Contract-derived semantic model and package evidence.

## Important Files

- `SPECIFICATION.md`: current APIG functional specification.
- `PLAN.md`: lifecycle packet implementation plan.
- `capability_contract.py`: configuration, rules, adapters, UI, and theme.
- `gateway_runtime.py`: dependency-light generated-app lifecycle service.
- `api.py`: generated-app API helper functions.
- `view_models.py`: generated UI data models.
- `views.py`: compatibility re-export for view models.
- `app.py`: publishable package entrypoint and semantic model generator.
- `test_capability_contract.py`: focused rule, lifecycle, API, and UI tests.
- `tests/test_package_contract.py`: package contract and app evidence tests.

## Generated-App Usage

```python
from capabilities.common.apig.gateway_runtime import ApigService

service = ApigService()

service.register_upstream(
    upstream_id="orders-api",
    tenant_id="tenant-a",
    name="Orders API",
    base_url="https://orders.internal",
    owner="platform",
)

service.register_consumer(
    consumer_id="orders-client",
    tenant_id="tenant-a",
    name="Orders Client",
    owner="integrations",
    credential_rotation_recorded=True,
)

route = service.request_route(
    route_id="orders-public",
    tenant_id="tenant-a",
    path="/orders",
    methods=["GET"],
    upstream_id="orders-api",
    consumer_id="orders-client",
    owner="api-team",
    route_exposure="public",
    auth_policy_attached=True,
    rate_limit_configured=True,
    requested_rps_limit=1000,
)

assert route["status"] == "active"

agent = service.register_gateway_agent(
    agent_id="security-agent",
    tenant_id="tenant-a",
    name="Security Agent",
    runtime="codex",
    role="security_policy_reviewer",
    scope="public and external route security policies",
    owner="security",
    purpose="review gateway security recommendations",
    human_approval_required=True,
)

batch = service.validate_apig_lifecycle_batch(
    tenant_id="tenant-a",
    event_stream="bytewax",
    mutation_count=4,
)

assert agent["status"] == "active"
assert batch["status"] == "accepted"
```

## UI Composition

```python
from capabilities.common.apig.gateway_runtime import ApigService
from capabilities.common.apig.view_models import dashboard_model, route_designer_model

service = ApigService()
dashboard = dashboard_model(service, "tenant-a")
routes = route_designer_model(service, "tenant-a")
```

## Production Runtime Boundary

The generated-app control plane does not configure a live reverse proxy or
execute edge code. Production deployments must bind adapters for reverse proxy,
service discovery, auth, credential vault, metrics, audit, cache, Bytewax event
streaming, edge runtime execution, and external AI/automation runtimes such as
Codex, Claude Code, OpenCode, and Pi. Those adapters must honor APIG guardrail
decisions before side effects.

## Focused Verification

```bash
./.venv/bin/python -m py_compile \
  capabilities/common/apig/__init__.py \
  capabilities/common/apig/capability_contract.py \
  capabilities/common/apig/models.py \
  capabilities/common/apig/gateway_runtime.py \
  capabilities/common/apig/api.py \
  capabilities/common/apig/view_models.py \
  capabilities/common/apig/views.py \
  capabilities/common/apig/app.py \
  capabilities/common/apig/test_capability_contract.py \
  capabilities/common/apig/tests/test_package_contract.py

./.venv/bin/pytest -q \
  capabilities/common/apig/test_capability_contract.py \
  capabilities/common/apig/tests/test_package_contract.py

./.venv/bin/apg capabilities implementation-audit --root capabilities/common/apig --json
./.venv/bin/apg capabilities publish-plan capabilities/common/apig --json
```
