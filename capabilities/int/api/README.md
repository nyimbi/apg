# Integration API Management

## Overview

Integration API Management (`int_api`) is the central governance layer for the APG platform's API lifecycle. It provides a unified registry and control plane for defining, securing, versioning, and deploying APIs across all integration domains — tracking every API from initial draft through active production deployment and eventual retirement.

Beyond raw registration, the capability enforces a full policy-attached governance model: every write operation requires an associated policy, external upstreams and consumers trigger mandatory review workflows, and production deployments require explicit human approval. An AI agent subsystem (supporting Codex, Claude Code, OpenCode, and Pi runtimes) can automate review and preparation tasks, but all privileged actions remain gated on human sign-off. The package boundary is kept dependency-light so generated APG applications can compose it immediately while production deployments attach durable gateway, developer portal, discovery, analytics, security, and Bytewax topology through adapters.

## Capability ID

`int_api`  Version: 2.1.0

## Provides

| Service | Description |
|---------|-------------|
| api_registry_lifecycle | Full CRUD lifecycle for API definitions including versioning, protocol config, and upstream routing |
| api_endpoint_lifecycle | Per-endpoint path/method registration with auth, scoping, caching, and deprecation controls |
| api_policy_lifecycle | Policy attachment (rate limiting, auth, CORS, transformation, circuit breaker, etc.) with ordered execution |
| api_consumer_lifecycle | Developer/application consumer registration, approval workflows, and access control |
| api_key_lifecycle | API key issuance, scoping, expiration, rotation, and revocation |
| api_subscription_lifecycle | Consumer-to-API subscription management with plan tiers and billing model support |
| api_deployment_workflow | Deployment tracking across environments with rolling/blue-green/canary strategies and rollback |
| api_gateway_route_catalog | Gateway route registration and upstream route mapping for the API gateway |
| api_analytics_workflow | Per-API/endpoint/consumer usage metrics, latency tracking, and SLA monitoring |
| api_dashboard_service | Unified management dashboard aggregating status, traffic, and governance signals |
| api_agents | AI-assisted API design, policy review, security review, consumer review, and deployment review agents |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Authentication and authorization for all management operations |
| audl | Audit logging of all state changes across the API lifecycle |
| ntfy | Notifications for review requests, approvals, deployments, and threshold breaches |
| composition_events | Cross-capability event routing for lifecycle state changes |
| composition_config | Tenant-scoped configuration resolution and override propagation |
| policy_management | Policy definition storage and evaluation runtime |
| service_discovery | Upstream service registry for validating and resolving upstream URLs |
| developer_portal | Consumer-facing portal integration for API documentation and key management |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tenant_id | string | `"default"` | Tenant scope for all operations |
| apis.name_required | bool | `true` | Enforce non-empty API name on registration |
| apis.title_required | bool | `true` | Enforce non-empty human-readable title |
| apis.base_path_required | bool | `true` | Enforce base path presence (must start with `/`) |
| apis.upstream_required | bool | `true` | Enforce upstream URL on registration |
| apis.owner_required | bool | `true` | Enforce owner assignment on registration |
| apis.supported_protocols | list | `[rest, graphql, grpc, webhook]` | Allowed protocol types |
| apis.supported_auth_types | list | `[api_key, oauth2, jwt, mtls, none]` | Allowed authentication methods |
| apis.default_rate_limit_minimum | int | `1` | Minimum allowed rate limit value |
| apis.external_upstream_review_required | bool | `true` | Require upstream review for non-internal URLs |
| endpoints.auth_required_by_default | bool | `true` | New endpoints default to requiring authentication |
| policies.supported_policy_types | list | `[rate_limit, quota, auth, transform, cors, ip_filter, circuit_breaker]` | Allowed policy types |
| policies.execution_order_required | bool | `true` | Enforce execution order field on policy attachment |
| consumers.contact_email_required | bool | `true` | Enforce valid email on consumer registration |
| consumers.external_consumer_review_required | bool | `true` | Require review for external consumer registrations |
| api_keys.expiration_required | bool | `true` | All issued keys must carry an expiration timestamp |
| subscriptions.approval_required | bool | `true` | Subscriptions require explicit approval before activation |
| subscriptions.supported_plans | list | `[sandbox, standard, premium, internal]` | Allowed subscription plan tiers |
| deployments.supported_environments | list | `[dev, test, stage, prod]` | Allowed deployment target environments |
| deployments.production_approval_required | bool | `true` | Production deployments require recorded approval |
| analytics.latency_review_threshold_ms | int | `2000` | Requests exceeding this latency trigger a review rule |
| api_agents.max_autonomous_scope | string | `"review_prepare_and_recommend"` | Maximum scope an agent can act on autonomously |
| api_agents.human_approval_required | bool | `true` | Privileged agent actions require human approval |
| governance.segregation_of_duties | bool | `true` | Prevent the same actor from both creating and approving |
| observability.event_stream | string | `"apg.int.api.lifecycle"` | Bytewax stream for all lifecycle events |

## API Routes

| Name | Path | Permission | Group |
|------|------|------------|-------|
| dashboard | /int-api/dashboard | int_api:view | Overview |
| apis | /int-api/apis | int_api:manage_apis | APIs |
| endpoints | /int-api/endpoints | int_api:manage_endpoints | APIs |
| policies | /int-api/policies | int_api:manage_policies | Governance |
| consumers | /int-api/consumers | int_api:manage_consumers | Consumers |
| keys | /int-api/keys | int_api:manage_keys | Consumers |
| subscriptions | /int-api/subscriptions | int_api:manage_subscriptions | Consumers |
| deployments | /int-api/deployments | int_api:deploy | Gateway |
| analytics | /int-api/analytics | int_api:view_analytics | Analytics |
| agents | /int-api/agents | int_api:admin | Automation |
| settings | /int-api/settings | int_api:admin | Administration |

REST API prefix: `/int-api/api/v1`

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context present | deny — attach_tenant_context |
| api_write_requires_policy | Write op without policy attachment | deny — attach_operation_policy |
| api_requires_name | register_api with no name | deny — set_api_name |
| api_requires_title | register_api with no title | deny — set_api_title |
| api_requires_base_path | register_api with no base path | deny — set_base_path |
| api_base_path_format | base path does not start with `/` | deny — set_valid_base_path |
| api_requires_upstream | register_api with no upstream URL | deny — set_upstream_url |
| api_requires_owner | register_api with no owner | deny — assign_api_owner |
| api_protocol_supported | Protocol not in supported list | deny — select_supported_protocol |
| api_auth_supported | Auth type not in supported list | deny — select_supported_auth_type |
| api_rate_limit_positive | Rate limit <= 0 | deny — set_rate_limit |
| external_upstream_requires_review | External upstream, no review recorded | require_review — record_upstream_review |
| endpoint_requires_api | register_endpoint without API | deny — select_api |
| endpoint_requires_path | register_endpoint without path | deny — set_endpoint_path |
| endpoint_path_format | Endpoint path does not start with `/` | deny — set_valid_endpoint_path |
| endpoint_method_supported | HTTP method not in supported list | deny — select_supported_method |
| policy_requires_name | attach_policy with no name | deny — set_policy_name |
| policy_type_supported | Policy type not in supported list | deny — select_supported_policy_type |
| policy_requires_config | attach_policy without config | deny — set_policy_config |
| policy_execution_order_nonnegative | Execution order < 0 | deny — set_execution_order |
| consumer_requires_name | register_consumer with no name | deny — set_consumer_name |
| consumer_requires_email | register_consumer with no email | deny — set_contact_email |
| consumer_email_format | Email fails format validation | deny — set_valid_email |
| consumer_requires_owner | register_consumer with no owner | deny — assign_consumer_owner |
| external_consumer_requires_review | External consumer, no review recorded | require_review — record_consumer_review |
| api_key_requires_consumer | issue_api_key without consumer | deny — select_consumer |
| api_key_requires_scope | issue_api_key without scopes | deny — set_key_scope |
| api_key_requires_expiration | issue_api_key without expiry | deny — set_key_expiration |
| subscription_requires_approval | create_subscription without approval | deny — record_subscription_approval |
| subscription_plan_supported | Plan not in supported list | deny — select_supported_plan |
| production_deployment_requires_approval | prod deploy without approval | require_review — record_deployment_approval |
| deployment_requires_route | deploy_api without gateway route | deny — set_gateway_route |
| slow_usage_requires_review | Request latency >= 2000 ms, no review | require_review — record_latency_review |
| api_batch_requires_bytewax | Batch op not routed through Bytewax | deny — route_api_batch_to_bytewax |
| api_event_requires_bytewax | Event op not routed through Bytewax | deny — route_api_event_to_bytewax |
| api_agent_runtime_supported | Agent runtime not in approved list | deny — select_supported_agent_runtime |
| api_agent_role_supported | Agent role not in approved list | deny — select_supported_agent_role |
| privileged_api_agent_action_requires_human_approval | Privileged agent action, no human approval | deny — record_human_approval |

## Data Models

| Model | Key Fields |
|-------|-----------|
| AMAPI | api_id, api_name, api_title, version, protocol_type, base_path, upstream_url, status, auth_type, default_rate_limit, tenant_id |
| AMEndpoint | endpoint_id, api_id, path, method, auth_required, scopes_required, rate_limit_override, cache_enabled, deprecated |
| AMPolicy | policy_id, api_id, policy_name, policy_type, config (JSONB), execution_order, enabled, conditions, applies_to_endpoints |
| AMConsumer | consumer_id, consumer_name, organization, contact_email, status, approval_date, allowed_apis, ip_whitelist, global_rate_limit, tenant_id |
| AMAPIKey | key_id, consumer_id, key_name, key_hash, key_prefix, scopes, allowed_apis, active, expires_at, ip_restrictions |
| AMSubscription | subscription_id, consumer_id, api_id, subscription_name, plan_name, status, rate_limit, quota_limit, billing_model |
| AMDeployment | deployment_id, api_id, deployment_name, strategy, environment, from_version, to_version, status, progress_percentage, traffic_percentage |
| AMAnalytics | metric_id, api_id, endpoint_id, consumer_id, metric_name, metric_type, metric_value, time_bucket, aggregation_period, tenant_id |
| AMUsageRecord | record_id, request_id, consumer_id, api_id, endpoint_path, method, response_status, response_time_ms, billable, cost |

All SQLAlchemy models use the `am_` table prefix. Primary keys are UUID7 strings with type-specific prefixes (`api_`, `ep_`, `pol_`, `con_`, `key_`, `sub_`, `dep_`, `met_`, `usg_`).

## Streaming Events

Events emitted to the `apg.int.api.lifecycle` event stream via Bytewax, keyed by `tenant_id`.

| Event | Trigger |
|-------|---------|
| api_registered | New API successfully registered in the registry |
| endpoint_registered | New endpoint added to an existing API |
| policy_attached | Policy attached or updated on an API |
| consumer_registered | New consumer approved and registered |
| api_key_issued | API key generated for a consumer |
| subscription_created | Consumer subscription to an API activated |
| api_approved | API review approved and status moved to `approved` |
| api_deployed | API deployment completed successfully in an environment |
| usage_recorded | API call usage record persisted (batched via Bytewax) |
| api_agent_registered | New AI agent registered for a review/automation role |

Valid lifecycle states: `draft`, `active`, `approved`, `deployed`, `suspended`, `revoked`, `queued`, `blocked`

Streaming guardrails enforced as rules: `api_batch_requires_bytewax`, `api_event_requires_bytewax`, `privileged_api_agent_action_requires_human_approval`

## Edge Cases Handled

- **External upstream review bypass**: If `external_upstream` is flagged on registration but no review has been recorded, the rule engine returns `require_review` rather than `deny`, preserving the API in a reviewable state rather than hard-rejecting it.
- **Production deployment partial approval**: The `production_deployment_requires_approval` rule triggers `require_review`, not `deny` — the deployment record is created but gated until a human records approval, enabling async approval workflows without blocking creation.
- **Rate limit at exactly zero**: The `api_rate_limit_positive` rule uses `lte: 0` to catch both zero and negative values, since zero effective rate limit would silently block all traffic.
- **Policy execution order conflicts**: Multiple policies on one API execute in ascending `execution_order` value; the model enforces non-negative values but not uniqueness, allowing intentional parallel execution at the same priority level. Conflict resolution is the caller's responsibility.
- **Agent actions under `review_prepare_and_recommend` scope**: Agents are hard-limited to review, preparation, and recommendation operations autonomously; any action outside this scope requires a recorded human approval regardless of agent role.
- **Slow request review accumulation**: The `slow_usage_requires_review` rule fires on every usage record exceeding 2000 ms without a recorded review, meaning high-latency APIs will continuously generate review obligations until the latency issue is addressed or the threshold is adjusted per-tenant.
- **Consumer key expiration enforcement**: Key expiration is required at issuance; keys with no expiry are rejected by rule, preventing unbounded long-lived credentials.
- **Tenant isolation**: All model indexes include `tenant_id` as a composite key component; unique constraints on `api_name+version`, `consumer_name`, and `consumer+api_id` are scoped per tenant, allowing the same API names across different tenants.
- **Empty `applies_to_endpoints` semantics**: An empty list on `AMPolicy.applies_to_endpoints` means "apply to all endpoints" by convention. Non-empty lists restrict the policy to those specific endpoint IDs. This semantic default must be respected by the policy evaluation runtime.

## Composability

- **Upstream**: `auth` provides identity context for all operations; `composition_config` supplies tenant-specific configuration overrides; `service_discovery` validates upstream URLs before registration.
- **Downstream**: `developer_portal` consumes consumer, API, and key records to surface documentation and self-service access; `policy_management` receives policy definitions for runtime enforcement at the gateway; analytics sinks consume `AMUsageRecord` and `AMAnalytics` streams.
- **Peer**: `int_esb` (enterprise service bus) and `int_etl` (data pipeline) capabilities commonly deploy alongside `int_api` as they surface their own service endpoints through the API registry; `audl` is always co-deployed to satisfy the `audit_state_changes` governance requirement.

## Package Layout

| File | Purpose |
|------|---------|
| `capability_contract.py` | Executable APG contract and deterministic rule engine |
| `service.py` | Dependency-light lifecycle service implementation |
| `api.py` | Composition helpers and legacy endpoint shims |
| `views.py` | Framework-neutral screen models and legacy view shims |
| `models.py` | SQLAlchemy ORM models and Pydantic config models |
| `app.py` | Semantic model, component manifest, and self-test |
| `gateway.py` | Gateway route and upstream management |
| `monitoring.py` | Observability and metrics collection |
| `discovery.py` | Service discovery integration adapter |
| `integration.py` | External integration wiring helpers |
| `factory.py` | Component factory for adapter instantiation |
| `runner.py` | Runtime entry point and lifecycle hooks |
| `SPECIFICATION.md` | Records, workflows, rules, UI, events, adapter boundaries, acceptance criteria |
| `PLAN.md` | Implementation and review plan |
| `cap_spec.md` | Summary of the current executable runtime contract |
| `tests/test_package_contract.py` | Contract, lifecycle, guardrails, API, views, and app surface verification |

## Runtime Lifecycle

1. Register APIs with base path, upstream, owner, protocol, auth, and rate limit.
2. Register endpoints under APIs.
3. Attach API policies.
4. Register consumers and issue scoped, expiring API keys.
5. Create approved subscriptions.
6. Approve APIs and deploy them to gateway environments.
7. Record usage analytics and review slow requests.
8. Register API agents that review, prepare, and recommend within explicit human-approval boundaries.

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

## Development Notes

- The capability contract's rule engine is deterministic and stateless — `evaluate_capability_rules` evaluates all rules against a flat context dict and returns the most restrictive decision (`deny` > `require_review` > `allow`). Rules do not chain or have side effects.
- `models.py` imports `uuid7str` from `uuid_extensions` — this package is not on PyPI. The correct dependency is `uuid6` with a local shim re-exporting `uuid7str`. Update the import to `from uuid6 import uuid7` if `uuid_extensions` is not available in the target environment.
- `AMAnalytics` stores pre-aggregated metrics bucketed by `time_bucket` and `aggregation_period`. Raw per-request data lives in `AMUsageRecord`. Avoid querying `AMUsageRecord` for dashboard aggregates at scale — materialize into `AMAnalytics` via the Bytewax pipeline.
- The `api_agents` subsystem is intentionally constrained to `review_prepare_and_recommend` maximum autonomous scope. Any expansion of agent autonomy requires updating both the configuration and adding corresponding rules in `RULES`.
- This package does not start a live gateway by default. Production deployments bind these concerns through adapters: identity/authorization/tenant policy, audit vault and event replication, live gateway routing and service discovery, developer portal and application onboarding, analytics sinks and dashboards, notification and workflow routing, durable Bytewax topology and event sinks, and AI-agent runtime orchestration.
- Theme tokens (`int_api_control`) use a teal/purple palette (`#28536B` / `#6B5B95`) with compact density, suitable for data-dense management UIs. Tenant overrides are supported via `theme.allow_tenant_overrides: true`.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/int/api/__init__.py capabilities/int/api/capability_contract.py capabilities/int/api/service.py capabilities/int/api/api.py capabilities/int/api/views.py capabilities/int/api/app.py capabilities/int/api/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/int/api/tests/test_package_contract.py
./.venv/bin/python capabilities/int/api/app.py
./.venv/bin/apg capabilities inspect int_api --json
./.venv/bin/apg capabilities publish-plan capabilities/int/api --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/int/api --json
```
