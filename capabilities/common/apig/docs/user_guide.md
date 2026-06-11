# APIG User Guide

**Capability**: API Gateway & Management (`apig`)
**Domain**: common
**Author**: Nyimbi Odero — Datacraft
**Copyright**: © 2025 Datacraft

---

## Overview

APIG is the APG platform's governed API gateway control plane. It exposes two
service surfaces:

| Surface | File | Use case |
|---------|------|----------|
| `ApigService` | `gateway_runtime.py` | Generated-app lifecycle, tests, semantic evidence |
| `ProductionAPGIntelligentGatewayService` | `service.py` | Adapter-backed runtime: edge, WASM, AI, billing |

This guide covers the production service layer. For generated-app usage see
`README.md → Generated-App Usage`.

---

## Quick Start

```python
import asyncio
from capabilities.common.apig.service import ProductionAPGIntelligentGatewayService

async def main():
    svc = ProductionAPGIntelligentGatewayService(
        tenant_id="acme",
        user_id="ops-bot",
        config={
            "enable_wasm": False,
            "enable_ai": False,
        },
    )
    await svc.initialize()
    status = await svc.get_service_status()
    print(status["service"]["status"])   # healthy
    await svc.shutdown()

asyncio.run(main())
```

---

## Core Lifecycle Operations

### Create a Gateway

```python
from capabilities.common.apig.models import AgGatewayConfig, EnvironmentType

cfg = AgGatewayConfig(
    name="payments-gw",
    environment=EnvironmentType.PRODUCTION,
    created_by="ops",
    tenant_id="acme",
)
gateway = await svc.create_gateway(cfg)
print(gateway.id)
```

### Process a Request

```python
from capabilities.common.apig.models import AgHttpRequest, HttpMethod

req = AgHttpRequest(
    method=HttpMethod.GET,
    path="/v1/payments",
    client_ip="10.0.0.1",
    tenant_id="acme",
)
result = await svc.process_request(req, gateway_id=gateway.id)
print(result.response.status_code)
```

---

## Rate Limiting

### Token-Bucket Check (new)

`rate_limit_check` implements a token-bucket counter in process.  Replace the
backing store with Redis `EVAL` in production for atomicity across replicas.

```python
result = await svc.rate_limit_check(
    key="acme:consumer-x:payments-gw",
    capacity=100,        # max burst tokens
    refill_rate=10.0,    # tokens per second
    cost=1,              # tokens consumed by this request
)

if not result["allowed"]:
    # result["retry_after_ms"] tells the client when to retry
    raise RateLimitExceeded(result["retry_after_ms"])
```

Response fields: `allowed`, `remaining`, `capacity`, `refill_rate`,
`retry_after_ms`, `checked_at`.

### Quota Enforcement

```python
quota = await svc.quota_enforce(
    gateway_id=gateway.id,
    consumer_id="consumer-x",
    quota_limit=50_000,
    window_seconds=86400,   # daily
)
if quota["exhausted"]:
    ...  # serve 429
```

---

## API Key Management

### Onboard a Developer

```python
record = await svc.developer_onboard(
    developer_id="alice",
    app_name="mobile-app",
    scopes=["payments:read", "payments:write"],
    gateway_id=gateway.id,
)
api_key = record["api_key"]
```

### Zero-Downtime Key Rotation (new)

```python
rotation = await svc.rotate_api_key(
    developer_id="alice",
    app_name="mobile-app",
    overlap_seconds=3600,   # old key valid for 1 h after rotation
)
print(rotation["new_key"])
print(rotation["old_key_expires_at"])
```

### Validate a Key

Both the active key and the overlap-period previous key are accepted:

```python
record = await svc.validate_api_key(key=inbound_key)
if record is None:
    raise Unauthorized("invalid or expired API key")
```

---

## Security

### mTLS Certificate Validation (new)

```python
result = await svc.mtls_validate(
    gateway_id=gateway.id,
    route_id="payments-public",
    client_cert_pem=pem_string,
    trusted_cn_patterns=["*.acme.com", "payments-client"],
)
if not result["valid"]:
    raise Forbidden(result["reason"])
```

Requires `cryptography` for full decode (subject CN, expiry).  Falls back to
structural PEM check if not installed.

### Security Scan

```python
scan = await svc.security_scan_api(
    gateway_id=gateway.id,
    scan_type="owasp_top10",
)
# scan["findings"] lists unauthenticated routes
```

---

## Observability

### Inject W3C Trace Context (new)

Inject `traceparent` before forwarding to upstream so traces are correlated
end-to-end in Jaeger / Grafana Tempo:

```python
request = await svc.inject_trace_context(request)
# request.headers["traceparent"] is now set
```

### Metrics

```python
metrics = await svc.gateway_metrics(gateway_id=gateway.id)
print(metrics["avg_response_time_ms"])
```

---

## API Versioning & Lifecycle

### Register a Version

```python
await svc.api_version_manage(
    gateway_id=gateway.id,
    version="v2",
    status="active",
)
```

### Sunset a Version

```python
await svc.api_version_sunset(
    gateway_id=gateway.id,
    version="v1",
    sunset_date="2026-12-31",
    migration_guide_url="https://docs.acme.com/api/migrate-v1-v2",
)
```

### Compatibility Scoring (new)

Compare two OpenAPI specs before shipping a new version:

```python
score = await svc.version_compat_score(
    spec_old=old_openapi_dict,
    spec_new=new_openapi_dict,
)
print(score["score"])        # 0.0 – 1.0
print(score["breaking"])     # count of breaking changes
print(score["details"])      # list of change descriptions

if score["breaking"] > 0:
    raise ValueError("breaking API changes require a version bump")
```

---

## Traffic Management

### Traffic Split (Canary)

```python
split = await svc.traffic_split_apig(
    gateway_id=gateway.id,
    version_a="v2",
    version_b="v2-canary",
    split_pct_a=95,          # 95% stable, 5% canary
)
```

### Canary Statistical Analysis (new)

After running the canary for at least 10 minutes:

```python
analysis = await svc.canary_analyse(
    gateway_id=gateway.id,
    route_id="payments-public",
    canary_version="v2-canary",
    stable_version="v2",
    window_minutes=10,
)
# "promote" | "rollback" | "continue"
print(analysis["recommendation"])
print(analysis["error_rate_delta"])
print(analysis["statistically_significant"])
```

The recommendation uses a chi-squared test (p < 0.05) on error counts.
Rollback is triggered when canary error rate exceeds stable by more than 5 pp.

### Circuit Breaker

```python
cb = await svc.circuit_break(
    gateway_id=gateway.id,
    upstream_service="payments-backend",
    failure_threshold=5,
    recovery_timeout_seconds=30,
)
```

---

## Request Transformation

```python
await svc.request_transform(
    gateway_id=gateway.id,
    rule_name="add-version-header",
    match_path="/v1/*",
    add_headers={"X-API-Version": "1"},
    remove_headers=["X-Internal-Secret"],
)
```

---

## Request Body Validation (new)

Validate the request payload against a JSON Schema before forwarding:

```python
schema = {
    "type": "object",
    "required": ["amount", "currency"],
    "properties": {
        "amount":   {"type": "number", "minimum": 0},
        "currency": {"type": "string", "enum": ["KES", "USD", "EUR"]},
    },
}
validation = await svc.validate_request_body(request, schema)
if not validation["valid"]:
    return http_422(validation["errors"])
```

Requires `jsonschema` (draft-07).  Without it, a basic `isinstance(body, dict)`
check runs instead.

---

## Billing (new)

Generate a `Decimal`-precise billing statement for a tenant:

```python
from datetime import datetime, timezone

statement = await svc.billing_aggregate(
    tenant_id="acme",
    billing_period_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
    billing_period_end=datetime(2026, 6, 30, 23, 59, 59, tzinfo=timezone.utc),
    tier="pro",   # "free" | "pro" | "enterprise"
)
print(statement["total_due"])   # Decimal string, e.g. "49.3500"
print(statement["currency"])    # "USD"
```

Line items include the base plan fee and an overage charge computed with
`Decimal(overage_units) * Decimal(overage_per_1k)` — no floating-point
rounding.

---

## Documentation & Developer Portal

```python
# Generate OpenAPI skeleton from registered routes
doc = await svc.documentation_generate(gateway_id=gateway.id, output_format="openapi_3")

# Sync to external developer portal
await svc.developer_portal_sync(
    gateway_id=gateway.id,
    portal_url="https://developers.acme.com/api",
)
```

---

## Sandbox Environments

```python
sandbox = await svc.sandbox_env(
    gateway_id=gateway.id,
    sandbox_name="qa-environment",
    base_url="https://sandbox.acme.com",
)
```

---

## Focused Verification

```bash
# Syntax check
python -c "import py_compile; py_compile.compile('capabilities/common/apig/service.py')"

# Unit tests
uv run pytest -vxs capabilities/common/apig/tests/

# Contract tests
uv run pytest -q capabilities/common/apig/test_capability_contract.py
```

---

## Configuration Reference

| Key | Default | Description |
|-----|---------|-------------|
| `apg_base_url` | `http://localhost:8000` | APG platform base URL |
| `apg_api_key` | `demo-api-key` | APG platform API key |
| `redis_url` | `redis://localhost:6379` | Redis for rate limiting / cache |
| `ollama_url` | `http://localhost:11434` | Ollama for AI policy generation |
| `edge_location` | `default` | Edge node location label |
| `enable_wasm` | `True` | Enable WebAssembly runtime |
| `enable_ai` | `True` | Enable Ollama AI features |
| `max_wasm_modules` | `50` | WASM module registry limit |
| `circuit_breaker_threshold` | `5` | Failure count before open |
| `request_timeout` | `30` | Upstream request timeout (s) |

---

## Improvement Roadmap

See [WORLD_CLASS_IMPROVEMENTS.md](../WORLD_CLASS_IMPROVEMENTS.md) for 15 detailed
improvement proposals covering Redis-backed atomic rate limiting, OPA/Rego
policy-as-code, full mTLS enforcement, GraphQL protection, WebSocket proxying,
geographic routing, and more.
