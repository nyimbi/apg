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
- Durable review evidence on reviewable records: `policy_decision`,
  `matched_rules`, `review_reasons`, and `review_evidence`.
- Pending-review queues for routes, quota reviews, policies, traffic shifts,
  deployments, gateway agents, and lifecycle batches.
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

High-risk gateway changes preserve review evidence for operator consoles:

```python
request = service.request_route(
    route_id="orders-launch",
    tenant_id="tenant-a",
    path="/orders-launch",
    methods=["GET"],
    upstream_id="orders-api",
    owner="api-team",
    route_exposure="public",
    auth_policy_attached=True,
    requested_rps_limit=250000,
    justification="Launch traffic forecast requires elevated quota.",
)

assert request["route"]["status"] == "pending_quota_review"
assert request["route"]["policy_decision"] == "require_review"
assert request["quota_review"]["review_reasons"] == ["quota_review_required"]
assert service.list_pending_reviews("tenant-a")
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

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Monetary Precision for Billing & Quota Monetisation** [Correctness / Finance]
- **I2. Multi-Tier Token-Bucket Rate Limiting with Redis Lua Atomicity** [Performance / Correctness]
- **I3. mTLS Client Certificate Validation at the Edge** [Security]
- **I4. Semantic API Versioning with Backward-Compatibility Scoring** [API Lifecycle]
- **I5. Distributed Tracing Context Propagation (W3C Trace Context)** [Observability]
- **I6. Adaptive Circuit Breaker with Half-Open Probe Window** [Resilience]
- **I7. API Key Rotation with Zero-Downtime Dual-Active Window** [Security / Operations]
- **I8. Request Body Schema Validation Against OpenAPI Spec** [Data Quality / Security]
- **I9. Geographic Traffic Routing and Geo-Blocking** [Compliance / Performance]
- **I10. WebSocket and Server-Sent Events (SSE) Proxying** [Protocol Support]
- **I11. Policy-as-Code with OPA/Rego Integration** [Governance / Security]
- **I12. Response Caching with Content-Addressed Storage and ETags** [Performance]
- **I13. Canary Release Automation with Statistical Traffic Analysis** [Deployment Safety]
- **I14. GraphQL Gateway with Query Depth and Complexity Limiting** [Protocol Support / Security]
- **I15. Tenant Billing Dashboard with Decimal-Precise Usage Aggregation** [Monetisation / Multi-Tenancy]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
