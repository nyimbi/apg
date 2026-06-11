# int_api User Guide

© 2025 Datacraft. All rights reserved. Author: Nyimbi Odero <nyimbi@gmail.com>

---

## 1. Overview

`int_api` is the central governance layer for the APG platform's API lifecycle. It provides a
unified registry and control plane for:

- **Defining** APIs with protocol, authentication, upstream routing, and rate limits
- **Securing** them with policy chains (rate limit, auth, CORS, circuit breaker, etc.)
- **Consuming** them through scoped API keys and subscription plans
- **Deploying** them to named environments with approval gates
- **Monitoring** usage, latency SLAs, error budgets, and billing attribution

All writes require tenant context and policy attachment. External upstreams and external consumers
trigger mandatory review workflows. Production deployments require explicit human approval.

---

## 2. Quick Start

```python
import asyncio
from capabilities.int.api import IntApiService

svc = IntApiService(tenant_id="acme")

# 1. Register an API
api = svc.register_api(
    "payments",
    "acme",
    "payments",
    "Payments API",
    "/payments",
    "internal://payments-svc",
    "owner@acme.io",
    version="1.0.0",
    protocol="rest",
    auth_type="api_key",
    rate_limit_per_minute=5000,
)

# 2. Add an endpoint
ep = svc.register_endpoint("auth-ep", "acme", api["id"], "/authorize", "POST")

# 3. Attach a policy
svc.attach_policy(
    "rate-pol", "acme", api["id"],
    "rate_limit", "Global rate limit", {"limit": 5000}, execution_order=10,
)

# 4. Register a consumer and issue a key
consumer = svc.register_consumer("checkout", "acme", "Checkout App", "checkout@acme.io", "owner@acme.io")
key = svc.issue_api_key("ck-1", "acme", consumer["id"], "Checkout key", ["payments:write"], "2027-01-01")

# 5. Create a subscription
sub = svc.create_subscription("sub-1", "acme", consumer["id"], api["id"], "standard", "approver@acme.io")

# 6. Approve and deploy
svc.approve_api(api["id"], "acme", "approver@acme.io")
svc.deploy_api("dep-1", "acme", api["id"], "prod", "/gw/payments", "deployer@acme.io", "approver@acme.io")

print(svc.dashboard_summary("acme"))
```

---

## 3. Core Operations

### 3.1 API Registration

`register_api(api_id, tenant_id, name, title, base_path, upstream_url, owner_id, ...)`

Key constraints:
- `base_path` must start with `/`
- `protocol` must be one of `rest | graphql | grpc | webhook`
- `auth_type` must be one of `api_key | oauth2 | jwt | mtls | none`
- External `upstream_url` (http/https) triggers a `require_review` until `reviewed_by` is recorded

Status transitions: `draft → approved → deployed`

### 3.2 Endpoint Registration

`register_endpoint(endpoint_id, tenant_id, api_id, path, method, ...)`

- `path` must start with `/`
- `method` must be one of `GET | POST | PUT | PATCH | DELETE | HEAD | OPTIONS`
- `auth_required=True` by default

### 3.3 Policy Attachment

`attach_policy(policy_id, tenant_id, api_id, policy_type, name, config, execution_order)`

Supported policy types: `rate_limit | quota | auth | transform | cors | ip_filter | circuit_breaker`

Policies execute in ascending `execution_order`. Multiple policies at the same order value run in
parallel (conflict resolution is caller's responsibility).

### 3.4 Consumer Management

`register_consumer(consumer_id, tenant_id, name, contact_email, owner_id, external, reviewed_by)`

External consumers require a review before the record transitions to active.

### 3.5 API Key Issuance

`issue_api_key(key_id, tenant_id, consumer_id, name, scopes, expires_on)`

All keys must carry an expiration. The `key_prefix` field (format: `apg_<hex8>`) is safe to surface
in UIs; the full key material is never stored — only the prefix.

### 3.6 Subscriptions

`create_subscription(subscription_id, tenant_id, consumer_id, api_id, plan, approved_by)`

Supported plans: `sandbox | standard | premium | internal`

### 3.7 Deployment

`deploy_api(deployment_id, tenant_id, api_id, environment, gateway_route, deployed_by, approved_by)`

Supported environments: `dev | test | stage | prod`

Production deployments require `approved_by` to be set. The rule engine returns `require_review`
rather than `deny` — the deployment record is created but gated until approval is recorded.

---

## 4. Integration Lifecycle

### 4.1 Register an Integration

```python
integration = await svc.register_integration(
    name="CRM Sync",
    type="bidirectional",
    source="crm://salesforce",
    target="internal://contacts",
    config={"batch_size": 500},
    tenant_id="acme",
)
await svc.activate_integration(integration["id"], tenant_id="acme")
```

### 4.2 Sync

```python
# Immediate sync
run = await svc.sync_now(integration["id"], tenant_id="acme")

# Scheduled sync
schedule = await svc.schedule_sync(integration["id"], "0 * * * *", tenant_id="acme")

# Bulk sync for multiple integrations
results = await svc.bulk_sync([id1, id2, id3], tenant_id="acme")
```

### 4.3 Data Mapping

```python
mapping = await svc.create_mapping(
    "CRM to Contacts",
    source_schema={"first_name": "str", "last_name": "str", "email": "str"},
    target_schema={"given_name": "str", "family_name": "str", "email_address": "str"},
    tenant_id="acme",
)
# Auto-maps identical field names. Add explicit rules for renames.
```

### 4.4 Webhooks

```python
wh = await svc.register_webhook(
    integration["id"], "https://hooks.acme.io/crm",
    events=["contact.created", "contact.updated"],
    secret="super-secret",
    tenant_id="acme",
)
await svc.test_webhook(wh["id"], tenant_id="acme")
```

---

## 5. New Async Capabilities (v2.2.0)

### 5.1 Circuit Breaker

Tracks open/half-open/closed state per upstream. Wire `record_usage` error counts to trip the
breaker automatically; use `reset_circuit` to manually restore service.

```python
state = await svc.get_circuit_state("https://upstream.example.com", tenant_id="acme")
# state["state"] → "closed" | "open" | "half_open"

await svc.reset_circuit("https://upstream.example.com", tenant_id="acme")
```

### 5.2 API Key Rotation

Zero-downtime credential cycling with a configurable overlap window:

```python
result = await svc.rotate_api_key(old_key_id, overlap_seconds=600, tenant_id="acme")
# result["new_key"] → active immediately
# result["old_key_id"] → marked "rotating", deactivated after 600 s
```

### 5.3 OpenAPI Spec Validation and Diffing

```python
validation = await svc.validate_openapi_spec(spec_dict)
# {"valid": True, "errors": []}

diff = await svc.diff_openapi_spec(api_id, new_spec, tenant_id="acme")
# {"added_paths": [...], "removed_paths": [...], "breaking_change_detected": False}
```

### 5.4 SLA Error Budget

```python
budget = await svc.sla_budget(api_id, slo_target=0.999, window_hours=24, tenant_id="acme")
# {"budget_remaining": 42, "burn_rate": 0.12, "breached": False}
```

A `sla_budget_alert` audit event is emitted when `budget_remaining < 0`.

### 5.5 Schema Registry

Validate event payloads against registered JSON Schemas before they reach downstream consumers:

```python
await svc.register_event_schema(
    integration_id, "contact.created",
    schema={"type": "object", "required": ["id", "email"],
            "properties": {"id": {"type": "string"}, "email": {"type": "string"}}},
    tenant_id="acme",
)
result = await svc.validate_event_payload(integration_id, "contact.created", payload, tenant_id="acme")
# {"valid": True, "errors": []}
```

### 5.6 Dependency Graph

Understand blast radius before changing an API:

```python
graph = await svc.api_dependency_graph(api_id, tenant_id="acme")
# {
#   "dependent_consumers": ["consumer-abc", ...],
#   "active_deployments": [{"environment": "prod", ...}],
#   "safe_to_modify": False,
# }
```

### 5.7 Versioned Snapshots

```python
await svc.snapshot_api(api_id, label="v1.2.0-pre-migration", tenant_id="acme")
# ... make risky changes ...
await svc.restore_api_snapshot(api_id, "v1.2.0-pre-migration", restored_by="ops@acme.io", tenant_id="acme")
```

### 5.8 Multi-Region Deployment

```python
result = await svc.deploy_multi_region(
    api_id,
    regions=["eu-west-1", "us-east-1", "ap-southeast-1"],
    gateway_route="/gw/payments",
    deployed_by="cicd@acme.io",
    approved_by="ops@acme.io",
    tenant_id="acme",
)
# {"succeeded": [...], "failed": [], "success_count": 3}

health = await svc.regional_health_summary(api_id, tenant_id="acme")
```

### 5.9 Cost Attribution

```python
cost = await svc.compute_consumer_cost(
    consumer_id, "2026-06-01", "2026-06-30", tenant_id="acme"
)
# {"total_requests": 120000, "total_cost": 120.0, "currency": "USD"}
```

### 5.10 Adaptive Rate-Limit Tuning

```python
reco = await svc.recommend_rate_limit(api_id, lookback_hours=168, safety_factor=1.25, tenant_id="acme")
# {"p95_rpm": 800, "recommended_limit": 1000, "current_limit": 500}

await svc.apply_recommended_rate_limit(api_id, approved_by="ops@acme.io", tenant_id="acme")
```

---

## 6. Dashboard and Reporting

```python
# Overall API management summary
summary = svc.dashboard_summary("acme")

# Integration status
dash = await svc.integration_dashboard(tenant_id="acme")

# Error report for a specific integration
report = await svc.error_report(integration_id, "last_24h", tenant_id="acme")

# Data quality grading (A/B/C)
quality = await svc.data_quality_report(integration_id, "last_7d", tenant_id="acme")
```

---

## 7. AI Agent Subsystem

Register an AI agent for review and preparation tasks. Agents are hard-limited to
`review_prepare_and_recommend` scope; any privileged action requires recorded human approval.

```python
agent = svc.register_api_agent(
    tenant_id="acme",
    name="Security Reviewer",
    runtime="claude_code",
    role="security_reviewer",
    scope="review_prepare_and_recommend",
)

# Validate an agent action before execution
result = svc.validate_api_agent_action(
    "acme", agent["id"],
    action="recommend_policy_change",
    privileged_scope=False,
    human_approval_recorded=False,
)
```

Supported runtimes: `codex | claude_code | opencode | pi`
Supported roles: `api_designer | policy_reviewer | security_reviewer | consumer_reviewer |
deployment_reviewer | analytics_reviewer`

---

## 8. Audit Trail

Every state-changing operation appends an immutable audit event to `_audit_events`:

```python
events = svc.audit_events("acme")
# Returns all events for the tenant, ordered by emission time
```

Key event types: `api_registered`, `api_approved`, `api_deployed`, `api_restored`,
`api_key_issued`, `api_key_rotated`, `consumer_registered`, `subscription_created`,
`policy_attached`, `usage_recorded`, `sla_budget_alert`, `rate_limit_updated`, `integration_registered`

---

## 9. Business Rules Reference

The rule engine is deterministic and stateless. Rules evaluate to `deny`, `require_review`, or
`allow`. `deny` short-circuits immediately; `require_review` is the most restrictive non-deny
outcome.

Key governance rules to be aware of:

| Rule | When it fires |
|------|---------------|
| `tenant_context_required` | Any operation without a valid tenant_id |
| `api_write_requires_policy` | Any write operation that bypasses policy attachment |
| `external_upstream_requires_review` | Registering an API with an http/https upstream and no `reviewed_by` |
| `external_consumer_requires_review` | Registering an external consumer without a review |
| `production_deployment_requires_approval` | Deploying to `prod` without `approved_by` |
| `api_key_requires_expiration` | Issuing a key with no expiry date |
| `subscription_requires_approval` | Creating a subscription without `approved_by` |
| `slow_usage_requires_review` | Recording usage with latency >= 2000 ms without `reviewed_by` |
| `privileged_api_agent_action_requires_human_approval` | Agent privileged scope action without human approval |

---

## 10. Error Handling

| Exception | When raised |
|-----------|-------------|
| `PermissionError` | Rule engine returns `deny`; message contains the reason code(s) |
| `KeyError` | Record not found (format: `"record_type_not_found:id"`) |
| `ValueError` | Duplicate snapshot label; invalid field value |
| `APINotFoundError` | Explicit API lookup failure (subclass of `APIManagementError`) |
| `ConsumerNotFoundError` | Explicit consumer lookup failure |
| `AuthenticationError` | Authentication failure |
| `AuthorizationError` | Authorisation failure |
| `RateLimitExceededError` | Rate limit exceeded (raised by rate-limit backend when wired up) |

---

## 11. Composability

`int_api` publishes events to the `apg.int.api.lifecycle` Bytewax stream keyed by `tenant_id`.
Downstream capabilities that subscribe to this stream include:

- `audl` — persists all lifecycle events to the audit vault
- `ntfy` — sends notifications on approvals, deployments, and SLA budget alerts
- `developer_portal` — surfaces APIs, documentation, and key management to consumers
- `int_esb` / `int_etl` — register their own service endpoints through the API registry

Production adapters (gateway, identity, analytics sink, notification routing, durable Bytewax
topology) are attached via the adapter pattern defined in `factory.py`.

---

## 12. Configuration

Override defaults per-tenant at runtime:

```python
await svc.set_tenant_config(
    "acme",
    "analytics",
    {"latency_review_threshold_ms": 1000},
)
```

Or pass `overrides` to `get_capability_contract` for static configuration:

```python
from capabilities.int.api.capability_contract import get_capability_contract

contract = get_capability_contract("acme", overrides={
    "subscriptions": {"supported_plans": ["sandbox", "standard", "enterprise"]},
    "analytics": {"latency_review_threshold_ms": 500},
})
```

See `README.md → Configuration Reference` for the full parameter catalogue.
