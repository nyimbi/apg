# Platform Federation — User Guide

**Capability ID**: `plfd` | **Domain**: `common` | **Version**: `1.1.0`  
© 2025 Datacraft — Nyimbi Odero

---

## Overview

PLFD is the APG **Platform Federation** capability. It provides every APG
application with a tenant-scoped governance runtime covering:

- Foundation service registry and tiering
- Dependency posture tracking and active health probing
- Required baseline management (configuration, auth, audit, tenant)
- Readiness gates before production promotion
- Platform change queue with multi-stage approval and risk scoring
- Feature flags with gradual rollout and A/B experiment support
- Circuit breakers, rate limiters, and graceful degradation
- Canary release orchestration
- SLA contract enforcement
- Federated identity token exchange (cross-tenant)
- Capability sharing negotiation
- Baseline drift detection
- Comprehensive audit trail with Bytewax event streaming

All state is tenant-isolated. The service layer is pure Python with no
mandatory external dependencies — adapters for config stores, identity
providers, audit sinks, and Bytewax workers are bound by the host application.

---

## Installation

```bash
pip install apg-common-plfd
```

---

## Quick Start

```python
import asyncio
from capabilities.common.plfd import PlfdService

svc = PlfdService()

# Register a foundation service
svc.register_foundation_service(
    service_id="plfd-core",
    tenant_id="tenant-acme",
    name="Core Platform",
    owner="platform-team",
    tier="core",
    readiness_score=94,
    monitoring_enabled=True,
    rollback_plan_ref="rollback-v1",
    change_window_ref="cw-2026-q2",
)

# Attach a configuration baseline
svc.attach_baseline(
    baseline_id="base-cfg-001",
    tenant_id="tenant-acme",
    service_id="plfd-core",
    baseline_type="configuration",
    evidence_ref="evidence-cfg-001",
    approved_by="lead-engineer",
)

# Assess readiness
result = svc.assess_readiness("assess-001", "tenant-acme", "plfd-core")
print(result["status"])  # 'ready' or 'blocked'
```

---

## Foundation Services

### `register_foundation_service`

Registers a service in the platform registry with governance metadata.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `service_id` | str | Yes | Stable unique ID |
| `tenant_id` | str | Yes | Tenant scope |
| `name` | str | Yes | Human-readable name |
| `owner` | str | Yes | Owner identity |
| `tier` | str | Yes | `core` \| `shared` \| `extended` |
| `readiness_score` | float | Yes | 0–100 |
| `monitoring_enabled` | bool | No | Default False |
| `rollback_plan_ref` | str | No | Reference to rollback doc |
| `change_window_ref` | str | No | Reference to change window |

### `attach_baseline`

Attaches a typed baseline to a service. Required baseline types:
`configuration`, `tenant`, `auth`, `audit`.

### `assess_readiness`

Evaluates a service against all readiness gates. Returns `status: ready`
when all gates pass (dependencies healthy, baselines complete, monitoring
enabled, rollback plan present, change window set).

---

## Dependency Management

### `record_dependency` / `dependency_declare`

Declares a dependency edge between two registered services. Requires
`evidence_ref` to pass the dependency evidence rule.

```python
svc.record_dependency(
    dependency_id="dep-001",
    tenant_id="tenant-acme",
    source_service_id="plfd-core",
    target_service_id="auth-service",
    health_status="healthy",
    required=True,
    evidence_ref="dep-health-check-2026-06",
)
```

### `dependency_graph`

Returns the full dependency graph as nodes + edges in a format suitable for
graph visualisation. Includes DFS cycle detection.

### Async Dependency Probing

```python
# Active probe for a single dependency (plug in your HTTP check)
async def my_probe(dep: dict) -> str:
    # return 'healthy' | 'degraded' | 'unhealthy'
    return "healthy"

result = await svc.async_probe_dependency_health("tenant-acme", "dep-001", my_probe)

# Fan-out probe across all dependencies
report = await svc.async_probe_all_dependencies("tenant-acme", my_probe)
print(report["unhealthy"])
```

---

## Platform Changes

### `propose_platform_change`

Submits a change for approval. Requires owner and non-zero
`affected_capability_count`.

### `approve_platform_change`

Approves a change. All gates must pass: dependency health, approval recorded,
security review, change window, rollback plan, and Bytewax event stream.

### `async_score_change_risk`

Computes a composite risk score (0–100) and risk band before approval.

```python
risk = await svc.async_score_change_risk("change-001", "tenant-acme")
print(risk["risk_band"])          # 'low' | 'medium' | 'high' | 'critical'
print(risk["recommended_actions"])
```

Risk scoring components:

| Component | Max Points | Driver |
|-----------|-----------|--------|
| Blast radius | 40 | `affected_capability_count` / 50 |
| Dependency health | 30 | Ratio of unhealthy required deps |
| Review completeness | 20 | Missing broad/security review (10 pts each) |
| Rollback readiness | 10 | No rollback plan ref |

---

## Feature Flags

### `feature_flag_set` / `feature_flag_create`

Create or update a feature flag with optional rollout conditions.

```python
svc.feature_flag_set(
    flag_name="dark-mode",
    enabled=True,
    conditions={"regions": ["eu-west-1"]},
    tenant_id="tenant-acme",
    rollout_percentage=25.0,
)
```

### `feature_flag_check` / `feature_flag_evaluate`

Evaluate a flag for a request context. Deterministic for a given `user_id`
(hash-based percentage assignment).

```python
result = svc.feature_flag_check(
    flag_name="dark-mode",
    context={"user_id": "user-42", "region": "eu-west-1"},
    tenant_id="tenant-acme",
)
print(result["enabled"])  # True / False
```

### `ab_config_create`

Create an A/B experiment with multiple variants and a traffic split.

```python
svc.ab_config_create(
    tenant_id="tenant-acme",
    experiment_name="checkout-flow",
    variants=[{"name": "control", "config": {}}, {"name": "v2", "config": {"show_promo": True}}],
    traffic_split=[60.0, 40.0],
)
```

---

## Circuit Breakers

### `circuit_breaker_define`

Define a circuit breaker with custom thresholds.

```python
svc.circuit_breaker_define(
    tenant_id="tenant-acme",
    service_name="payments",
    failure_threshold=3,
    recovery_timeout_seconds=30,
)
```

### `circuit_breaker_status`

Return current state (closed / open / half-open) and failure metrics.

### `circuit_breaker_reset`

Manually reset to closed state. Requires `approved_by`.

---

## Canary Releases

```python
# Start a canary at 5% traffic
canary = await svc.async_canary_release_start(
    tenant_id="tenant-acme",
    service_name="api-gateway",
    canary_version="v2.1.0",
    baseline_version="v2.0.3",
    initial_traffic_pct=5.0,
)
canary_id = canary["canary_id"]

# Advance to 20%
await svc.async_canary_release_advance("tenant-acme", canary_id, 20.0)

# Promote to 100% (status → 'promoted')
await svc.async_canary_release_advance("tenant-acme", canary_id, 100.0)

# Or abort on error
await svc.async_canary_release_abort("tenant-acme", canary_id, "p99_latency_spike")
```

---

## SLA Contracts

```python
# Register an SLA
await svc.async_sla_contract_register(
    tenant_id="tenant-acme",
    service_name="api-gateway",
    availability_pct=99.9,
    latency_p99_ms=200.0,
    error_rate_pct=0.1,
    rpo_minutes=5,
    rto_minutes=30,
)

# Evaluate against live metrics
evaluation = await svc.async_sla_evaluate(
    tenant_id="tenant-acme",
    service_name="api-gateway",
    metrics_window={
        "observed_availability_pct": 99.95,
        "observed_latency_p99_ms": 185.0,
        "observed_error_rate_pct": 0.05,
    },
)
print(evaluation["compliant"])   # True
print(evaluation["breaches"])    # []
```

---

## Baseline Drift Detection

```python
drift = await svc.async_detect_baseline_drift(
    tenant_id="tenant-acme",
    service_id="plfd-core",
    live_config_snapshot={
        "db.max_connections": 150,   # was 100 in baseline
        "cache.ttl_seconds": 300,
    },
    drift_threshold=0.05,  # audit if >5% keys changed
)
print(drift["drift_detected"])   # True
print(drift["changed_keys"])     # ["db.max_connections"]
```

---

## Platform Federation

### Federated Token Exchange

Issue a capability-scoped assertion token allowing `source_tenant` to act on
behalf of `target_tenant` within declared scopes (OAuth2 RFC 8693 semantics).

```python
token = await svc.async_federated_token_exchange(
    source_tenant="tenant-acme",
    target_tenant="tenant-partner",
    scopes=["plfd:view", "regy:read"],
    issuer_token="<source-bearer-token>",
    requested_by="federation-service",
)
print(token["assertion_id"])
print(token["status"])   # 'issued'
```

### Capability Sharing Negotiation

```python
agreement = await svc.async_negotiate_capability_share(
    requester_tenant="tenant-acme",
    capability_id="regy",
    offered_capabilities=["plfd", "conf"],
    contract_version="1.0.0",
)
print(agreement["status"])   # 'accepted' | 'rejected'
```

---

## Configuration Management

### `platform_configuration`

Set a tenant-scoped configuration key for an environment. Supports dot-notation
keys and versioning.

```python
svc.platform_configuration(
    key="db.max_connections",
    value=100,
    environment="production",
    tenant_id="tenant-acme",
    data_type="int",
)
```

### `config_hot_reload`

Trigger a hot-reload of all (or selected) config keys for an environment.

### `env_promote`

Promote a config key from one environment to another (e.g. staging → production).

---

## Observability

### `platform_metrics_dashboard`

Returns a full snapshot of all platform sub-systems for a tenant.

### `platform_analytics`

Extends the metrics dashboard with A/B experiment counts and hot-reload history.

### `platform_health`

Single-call platform health status combining service health, circuit breaker
posture, and rate limiter counts.

### Audit Trail

Every mutating operation emits a `PlfdAuditEvent`. Query via:

```python
events = svc.list_audit_events("tenant-acme")
```

Events are structured with `event_type`, `subject_id`, `actor`, `decision`,
and `reasons` fields.

---

## AI Foundation Agents

Register AI agents before they participate in governance workflows.

```python
svc.register_plfd_agent(
    tenant_id="tenant-acme",
    name="Readiness Reviewer",
    runtime="claude_code",
    role="readiness_reviewer",
    scope="Review dependency, baseline, monitoring, and rollback gates",
)
```

Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`  
Supported roles: `foundation_reviewer`, `dependency_reviewer`,
`baseline_reviewer`, `readiness_reviewer`, `change_reviewer`,
`security_reviewer`

---

## Composition

`plfd` composes with:

| Capability | Purpose |
|-----------|---------|
| `conf` | Platform and tenant configuration storage |
| `mten` | Multi-tenant baseline requirements |
| `auth` | Identity, permissions, RBAC |
| `audl` | Durable audit evidence sink |
| `moni` / `hlth` | Health and monitoring evidence |
| `regy` | Platform service registry integration |
| `secu` | Security review evidence |
| `plgn` | Plugin registry governance |

Reference in `.apg` source:

```apg
use plfd;
```

Batch foundation mutations and change lifecycle events must use the `bytewax`
event-stream adapter.

---

## UI Routes

| Path | Permission | Component |
|------|-----------|-----------|
| `/plfd/dashboard` | `plfd:view` | PLFDDashboard |
| `/plfd/services` | `plfd:manage_services` | FoundationServices |
| `/plfd/dependencies` | `plfd:view` | DependencyMap |
| `/plfd/baselines` | `plfd:manage_baselines` | BaselineManager |
| `/plfd/readiness` | `plfd:view` | ReadinessGate |
| `/plfd/changes` | `plfd:approve_changes` | PlatformChangeQueue |
| `/plfd/agents` | `plfd:admin` | PLFDAgentPanel |
| `/plfd/governance` | `plfd:admin` | FoundationGovernance |
| `/plfd/audit` | `plfd:admin` | FoundationAuditTrail |
| `/plfd/settings` | `plfd:admin` | PLFDSettings |

---

## Verification

```bash
python -m py_compile capabilities/common/plfd/service.py
pytest -q capabilities/common/plfd/test_capability_contract.py
apg capabilities implementation-audit --root capabilities/common/plfd --json
```
