# Integration API Management Capability

`int_api` is the APG capability packet for API registry, endpoint, policy,
consumer, key, subscription, deployment, gateway-route, analytics, and API-agent
lifecycles. It keeps the package boundary dependency-light so generated APG
applications can compose it immediately while production deployments attach
durable gateway, developer portal, discovery, analytics, security, and Bytewax
topology through adapters.

## What It Provides

- Tenant-scoped API registration with protocol, authentication, base path,
  upstream, owner, version, review, and rate-limit metadata.
- Endpoint registration with supported HTTP methods and route-level controls.
- Policy attachment for rate limits, quotas, authentication, transformation,
  CORS, IP filtering, and circuit breaker controls.
- Consumer onboarding with email validation, ownership, and external-consumer
  review.
- API key issuance with scopes and expiration.
- Subscription approval by plan.
- Deployment workflow for dev, test, stage, and production environments.
- Usage analytics with latency guardrails.
- First-class API-agent registration for Codex, Claude Code, OpenCode, and Pi
  review teams.
- APG UI route metadata, framework-neutral screen models, compact theme tokens,
  semantic metadata, package manifest, and release evidence.

## Package Layout

- `SPECIFICATION.md` defines records, workflows, rules, UI, events, adapter
  boundaries, and acceptance criteria.
- `PLAN.md` records the implementation and review plan for this lifecycle
  packet.
- `cap_spec.md` summarizes the current executable runtime contract.
- `capability_contract.py` exposes the executable APG contract and deterministic
  rule engine.
- `service.py` implements the dependency-light lifecycle service.
- `api.py` exposes composition helpers and legacy endpoint shims.
- `views.py` exposes framework-neutral screen models and legacy view shims.
- `app.py` exposes semantic model, component manifest, and self-test.
- `tests/test_package_contract.py` verifies the package contract, lifecycle,
  guardrails, API, views, and app surface.

## Runtime Lifecycle

1. Register APIs with base path, upstream, owner, protocol, auth, and rate
   limit.
2. Register endpoints under APIs.
3. Attach API policies.
4. Register consumers and issue scoped, expiring API keys.
5. Create approved subscriptions.
6. Approve APIs and deploy them to gateway environments.
7. Record usage analytics and review slow requests.
8. Register API agents that review, prepare, and recommend within explicit
   human-approval boundaries.

## Usage

```python
from capabilities.int.api import IntApiService

service = IntApiService()

api = service.register_api(
	"payments",
	"tenant-a",
	"payments",
	"Payments API",
	"/payments",
	"internal://payments",
	"api-owner",
)
endpoint = service.register_endpoint(
	"authorize",
	"tenant-a",
	api["id"],
	"/authorize",
	"POST",
)
service.attach_policy(
	"payments-rate-limit",
	"tenant-a",
	api["id"],
	"rate_limit",
	"Tenant rate limit",
	{"limit": 1000},
)
consumer = service.register_consumer(
	"checkout",
	"tenant-a",
	"Checkout App",
	"checkout@example.com",
	"consumer-owner",
)
service.issue_api_key(
	"checkout-key",
	"tenant-a",
	consumer["id"],
	"Checkout key",
	["payments:write"],
	"2026-12-31",
)
print(service.dashboard_summary("tenant-a"))
```

Generated APG applications can use `api.py`:

```python
from capabilities.int.api import api

status = api.capability_status("tenant-a")
records = api.list_records("apis", "tenant-a")
```

## Guardrails

- Tenant context is required.
- Write operations require policy context.
- APIs require name, title, base path, upstream, owner, supported protocol,
  supported auth type, and positive rate limit.
- External upstreams require review evidence.
- Endpoints require a same-tenant API, path, valid path format, and supported
  method.
- Policies require a same-tenant API, name, supported type, configuration, and
  nonnegative execution order.
- Consumers require name, valid email, owner, and review for external consumers.
- API keys require consumer, name, scopes, and expiration.
- Subscriptions require consumer, API, supported plan, and approval.
- Production deployments require approval and all deployments require API,
  supported environment, gateway route, and deployer.
- Usage records require API, status code, and nonnegative latency; slow requests
  require review.
- API batches and events require Bytewax metadata.
- API agents must use supported runtimes and roles.
- Privileged API-agent actions require recorded human approval.

## Integration Boundary

This package does not start a live gateway by default. Production deployments
should bind these concerns through adapters:

- identity, authorization, and tenant policy;
- audit vault and event replication;
- live gateway routing and service discovery;
- developer portal and application onboarding;
- analytics sinks and dashboards;
- notification and workflow routing;
- durable Bytewax topology and event sinks;
- AI-agent runtime orchestration.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/int/api/__init__.py capabilities/int/api/capability_contract.py capabilities/int/api/service.py capabilities/int/api/api.py capabilities/int/api/views.py capabilities/int/api/app.py capabilities/int/api/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/int/api/tests/test_package_contract.py
./.venv/bin/python capabilities/int/api/app.py
./.venv/bin/apg capabilities inspect int_api --json
./.venv/bin/apg capabilities publish-plan capabilities/int/api --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/int/api --json
```
