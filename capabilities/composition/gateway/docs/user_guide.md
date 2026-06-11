# API Gateway — User Guide

© 2025 Datacraft. Author: Nyimbi Odero <nyimbi@gmail.com>

---

## 1. Introduction

The API Gateway (`composition_gateway`) is the single ingress and inter-service control plane for the APG platform. It enforces the invariant that **no traffic reaches a service without passing through a policy, a rate limiter, a circuit breaker, and a valid TLS certificate**.

All operations are tenant-scoped. Every mutating action emits an audit event to the Bytewax+NATS stream `apg.composition.gateway.lifecycle`.

---

## 2. Quick Start

```python
from capabilities.composition.gateway.service import CompositionGatewayService

gw = CompositionGatewayService()

# 1. Register a service
svc = gw.register_service(
    service_key="payments-v2",
    tenant_id="acme",
    name="Payments Service v2",
    owner_id="alice",
    endpoints=[{"host": "payments.internal", "port": 8080, "protocol": "http"}],
    health_check_path="/health",
    capability_id="fintech_payments",
    public_service=True,
)

# 2. Attach a policy (circuit breaker + rate limit required for public)
policy = gw.attach_policy(
    policy_key="payments-policy",
    tenant_id="acme",
    service_id=svc["id"],
    rate_limit_configured=True,
    circuit_breaker_configured=True,
    owner_id="alice",
)

# 3. Create a route
route = gw.create_route(
    route_key="payments-route",
    tenant_id="acme",
    service_id=svc["id"],
    path="/api/v2/payments",
    methods=["GET", "POST"],
    public_route=True,
    policy_id=policy["id"],
    approved_by="alice",
    tls_enabled=True,
    event_stream="bytewax",
)

print(gw.dashboard_summary("acme"))
```

---

## 3. Service Registration

### Requirements

Every service registration must include:

| Field | Constraint |
|-------|-----------|
| `owner_id` | Non-empty; identifies the accountable engineer |
| `endpoints` | At least one endpoint |
| `health_check_path` | Non-empty path (e.g. `/health`) |
| `capability_id` | Binds the service to an APG capability |

```python
svc = gw.register_service(
    service_key="analytics-api",
    tenant_id="acme",
    name="Analytics API",
    owner_id="bob",
    endpoints=[
        {"host": "analytics-a.internal", "port": 8080, "protocol": "http", "weight": 70, "zone": "us-east-1a"},
        {"host": "analytics-b.internal", "port": 8080, "protocol": "http", "weight": 30, "zone": "us-east-1b"},
    ],
    health_check_path="/health",
    capability_id="intel_analytics",
    public_service=False,
)
```

Adding `zone` and `region` to endpoint metadata enables locality-aware load balancing (see Section 9).

---

## 4. Route Management

### Creating a Route

```python
route = gw.create_route(
    route_key="analytics-internal",
    tenant_id="acme",
    service_id=svc["id"],
    path="/internal/analytics",
    methods=["GET"],
    public_route=False,     # internal — no TLS/policy/approval required
    event_stream="bytewax",
)
```

### Public Routes

Public routes require `tls_enabled=True`, `policy_id` (pointing to a policy with both rate limit and circuit breaker), and `approved_by`. Missing any of the three raises `PermissionError`.

### Canary Traffic Splitting

```python
shift = gw.shift_traffic(
    shift_key="payments-canary-10pct",
    tenant_id="acme",
    route_id=route["id"],
    weights={"payments-v2": 10, "payments-v1": 90},
    actor_id="alice",
    canary_shift=True,
    canary_evidence="PR-4422-load-test-report",
    event_stream="bytewax",
)
```

Canary shifts without `canary_evidence` produce a `require_review` decision rather than an outright denial, preserving emergency-rollback capability.

---

## 5. Policy Enforcement

Policies attach to services. The rule engine enforces:

- Public services require `rate_limit_configured=True` and `circuit_breaker_configured=True`.
- Internal services accept partial policies.

```python
policy = gw.attach_policy(
    policy_key="analytics-policy",
    tenant_id="acme",
    service_id=svc["id"],
    rate_limit_configured=True,
    circuit_breaker_configured=False,  # allowed for internal
    owner_id="bob",
)
```

---

## 6. Certificate Management

Certificate private keys **must** be stored in a vault. The `secret_reference` field holds the vault path; the `private_key_pem` column is reserved for legacy use only and triggers a policy denial if used without a vault reference.

```python
cert = gw.register_certificate(
    certificate_key="payments-tls",
    tenant_id="acme",
    domain="payments.acme.com",
    owner_id="alice",
    secret_reference="vault://acme/tls/payments-v2",
    expires_at="2027-06-01T00:00:00Z",
)
```

---

## 7. Dark Traffic Shadowing

Shadow mode copies a live request to a parallel shadow service and returns a divergence report. The primary response is never blocked.

```python
report = await gw.shadow_request(
    route_id=route["id"],
    tenant_id="acme",
    request_payload={
        "method": "POST",
        "path": "/api/v2/payments",
        "headers": {"content-type": "application/json"},
        "body": b'{"amount": 100}',
    },
    shadow_service_url="http://payments-v3.internal:8080",
    actor_id="alice",
)
# report["shadow_status_code"], report["shadow_body_hash"], report["shadow_latency_ms"]
```

The shadow subject `apg.gateway.shadow.<route_id>` is automatically published to NATS. A Bytewax processor can consume it to aggregate divergence metrics over time before promoting the shadow service to canary.

---

## 8. Predictive Auto-Scaling Signals

The gateway pre-computes scaling signals from recent metrics and emits them to NATS for consumption by orchestrators.

```python
signals = await gw.emit_scaling_signals(
    tenant_id="acme",
    window_seconds=300,
)
for signal in signals:
    print(signal["service_id"], signal["recommended_replicas"], signal["confidence"])
    # Each signal also contains stream_subject = "apg.gateway.autoscale.<service_id>"
```

Bytewax workers subscribe to `apg.gateway.autoscale.*` and forward signals to Kubernetes HPA or a custom autoscaler.

---

## 9. Locality-Aware Load Balancing

When endpoints carry `zone` and `region` metadata, the gateway can score them by locality affinity before selecting one.

```python
ranked_endpoints = await gw.check_locality_affinity(
    service_id=svc["id"],
    tenant_id="acme",
    requester_zone="us-east-1a",
    requester_region="us-east-1",
)
# ranked_endpoints[0] is the cheapest endpoint to reach from us-east-1a
# each entry has locality_cost and zone_penalty keys
best = ranked_endpoints[0]
```

Zone penalties:
- Same zone: 1.0 (no penalty)
- Same region, different AZ: 1.5
- Different region: 3.0

Effective cost = `(p50_latency_ms * zone_penalty) / weight`. Endpoints without zone metadata receive the cross-region penalty as a conservative default.

---

## 10. JSON Schema Payload Validation

Attach a JSON Schema to any route to reject malformed payloads at the perimeter.

```python
schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["amount", "currency"],
    "properties": {
        "amount": {"type": "number", "minimum": 0.01},
        "currency": {"type": "string", "enum": ["USD", "EUR", "KES"]},
    },
    "additionalProperties": False,
}

result = await gw.validate_request_payload(
    route_id=route["id"],
    tenant_id="acme",
    payload={"amount": -5, "currency": "GBP"},
    schema=schema,
)
# result["valid"] = False
# result["errors"] = ["amount: -5 is less than the minimum of 0.01",
#                      "currency: 'GBP' is not one of ['USD', 'EUR', 'KES']"]
```

Validation failures emit `payload_validation_failed` to the Bytewax stream for analytics.

---

## 11. Adaptive Timeout Budgeting and Deadline Propagation

### Computing the Request Budget

```python
budget = await gw.compute_request_budget(
    inbound_deadline_header=request.headers.get("x-request-deadline"),
    route_id=route["id"],
    tenant_id="acme",
    min_upstream_timeout_ms=50,
)

if budget["should_short_circuit"]:
    return Response(status=504, body="Gateway Timeout — deadline exceeded")

upstream_timeout = budget["upstream_timeout_ms"] / 1000  # convert to seconds
```

### Propagating the Deadline Downstream

```python
outbound_headers = {"content-type": "application/json"}
outbound_headers = await gw.propagate_deadline(
    outbound_headers=outbound_headers,
    remaining_ms=budget["remaining_ms"],
)
# outbound_headers now contains x-request-deadline and x-request-budget-ms
```

This ensures every hop in the call chain respects the original client's deadline, eliminating the "zombie work" anti-pattern where backend services do expensive computation after the client has already given up.

---

## 12. API Deprecation Management

### Marking a Route Deprecated

```python
updated_route = await gw.deprecate_route(
    route_id=route["id"],
    tenant_id="acme",
    deprecated_at="2026-03-01T00:00:00Z",
    sunset_at="2026-09-01T00:00:00Z",
    migration_guide_url="https://docs.acme.com/api/migrate-v2-to-v3",
    actor_id="alice",
)
```

### Injecting Deprecation Headers per Response

```python
headers = await gw.get_deprecation_headers(
    route_id=route["id"],
    tenant_id="acme",
)
# headers = {
#   "Deprecation": "2026-03-01T00:00:00Z",
#   "Sunset": "2026-09-01T00:00:00Z",
#   "Link": '<https://docs.acme.com/api/migrate-v2-to-v3>; rel="successor-version"',
# }

if headers.get("X-Gateway-Gone") == "410":
    return Response(
        status=410,
        headers=headers,
        body="This API version has been retired. See Link header for migration guide.",
    )

response.headers.update(headers)
```

The gateway emits `route_deprecated` and `route_sunset` events to the Bytewax stream. Configure `ntfy` to send deprecation warnings 30, 7, and 1 day before the sunset date.

---

## 13. Gateway Agents

Register AI agents with specific roles to assist with mesh architecture review, traffic analysis, and policy recommendations.

```python
agent = gw.register_gateway_agent(
    tenant_id="acme",
    name="mesh-architect-agent",
    runtime="apg_agent_runtime",
    role="mesh_architect",
    instructions="Review traffic patterns and recommend circuit breaker thresholds.",
)

# Validate before allowing an agent to execute a privileged action
decision = gw.validate_agent_gateway_action(
    tenant_id="acme",
    agent_id=agent["id"],
    action="update_circuit_breaker_threshold",
    privileged_scope=True,
    human_approval_recorded=True,  # required for privileged scope
)
# decision["decision"] in ("allow", "require_review", "deny")
```

Supported runtimes and roles are defined in `capability_contract.py` under `SUPPORTED_GATEWAY_AGENT_RUNTIMES` and `SUPPORTED_GATEWAY_AGENT_ROLES`.

---

## 14. Dashboard and Audit

```python
summary = gw.dashboard_summary("acme")
# {
#   "service_count": 3,
#   "route_count": 5,
#   "policy_count": 3,
#   "certificate_count": 2,
#   "traffic_shift_count": 1,
#   "gateway_agent_count": 1,
#   "audit_event_count": 42,
#   "streaming": {...},
# }

# Full audit log
events = gw.audit_events("acme")
for event in events:
    print(event["event_type"], event["entity_id"], event["created_at"])
```

---

## 15. Streaming Architecture

All gateway events are emitted to NATS and processed by Bytewax.

| NATS Subject | Producer | Consumer |
|---|---|---|
| `apg.composition.gateway.lifecycle` | gateway service | Bytewax auditor, ntfy, metrics |
| `apg.gateway.shadow.<route_id>` | `shadow_request()` | Bytewax divergence analyser |
| `apg.gateway.autoscale.<service_id>` | `emit_scaling_signals()` | Bytewax → HPA bridge |

Bytewax stream states: `draft → active → healthy → degraded → canary → blocked → retired → sunset`

---

## 16. Error Reference

| Error | Cause | Resolution |
|-------|-------|-----------|
| `PermissionError: tenant_context_required` | `tenant_id` is empty or None | Pass a non-empty tenant_id |
| `PermissionError: public_route_requires_policy` | Creating public route without `policy_id` | Call `attach_policy()` first |
| `PermissionError: public_route_requires_tls` | Creating public route without `tls_enabled=True` | Set `tls_enabled=True` |
| `PermissionError: public_route_requires_approval` | Creating public route without `approved_by` | Set `approved_by` to the approver's ID |
| `PermissionError: certificate_requires_secret_reference` | `secret_reference` is empty | Set a vault path in `secret_reference` |
| `PermissionError: public_service_requires_rate_limit` | Attaching policy to public service without rate limit | Set `rate_limit_configured=True` |
| `PermissionError: public_service_requires_circuit_breaker` | Attaching policy to public service without circuit breaker | Set `circuit_breaker_configured=True` |
| `ValueError: route_path_must_start_with_slash` | `path` does not start with `/` | Prefix path with `/` |
| `ValueError: service_tenant_mismatch` | Accessing a service belonging to a different tenant | Verify `service_id` belongs to `tenant_id` |
| `KeyError: unknown_service:<id>` | `service_id` not found | Check `list_services()` for valid IDs |
| `KeyError: unknown_route:<id>` | `route_id` not found | Check `list_routes()` for valid IDs |

---

## 17. Testing

Unit tests live in `tests/`. Run with:

```bash
uv run pytest -vxs tests/
```

Integration tests that require a live database and NATS use the `tests/ci/` path. Shadow traffic tests require a running `httpx`-compatible HTTP server; use `pytest-httpserver` for fixtures.
