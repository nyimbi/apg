# PLFD Platform Federation Capability

PLFD provides APG applications with a tenant-scoped foundation governance and
**multi-platform federation** runtime: platform service registry, dependency
posture, required baselines, readiness gates, platform change approval,
foundation agents, UI metadata, theme tokens, audit evidence, and
Bytewax-backed lifecycle events.

The service is async-first for all federation APIs, with synchronous methods
available for local governance operations. Production adapters (config stores,
identity providers, audit sinks, monitoring, Bytewax workers) are bound by the
host application via the executable capability contract.

## What It Provides

- Foundation service registry with owner, tier, dependency declarations,
  readiness score, monitoring posture, rollback plan, and change window.
- Dependency posture records with source, target, health, required flag, and
  evidence reference — with active async probing.
- Baseline manager for configuration, tenant, auth, and audit baselines with
  live drift detection.
- Readiness assessment with dependency, baseline, monitoring, rollback, and
  change-window gates; OLS trajectory prediction.
- Platform change queue with composite risk scoring, affected capability scope,
  dependency health, approval, broad-review, security-review, change-window,
  and rollback gates.
- First-class AI foundation agents with runtime, role, scope, registration,
  and contribution-disclosure guardrails.
- Feature flags with gradual rollout, tenant/region conditions, and A/B
  experiment configuration.
- Circuit breakers, rate limiters (token bucket, sliding window, fixed window,
  leaky bucket), service discovery, and platform metrics dashboard.
- Chaos engineering: fault injection (latency / error / crash / partial) with
  automatic circuit-breaker tripping.
- Federated identity: OAuth2 RFC 8693 token exchange, revocation, SAML/OIDC
  claims mapping, W3C trace context propagation.
- SLA contract registration and compliance evaluation.
- Canary release orchestration with progressive traffic shifting.
- Merkle-chained audit log with integrity verification.
- UI route, API, view-model, theme, semantic-model, package-manifest, and
  release-report evidence.

## World-Class Enhancements (v2.0)

All 15 improvements from the design review are implemented:

1. **Async-First Service Layer** — `async_health_check_all_services` fans out
   via `asyncio.gather`; all federation and governance paths are non-blocking.
   5–20x throughput on health aggregation.

2. **Federated Multi-Tenant Auth Token Exchange** — `async_federated_token_exchange`
   implements OAuth2 RFC 8693. Issues capability-scoped assertion tokens
   between independently governed tenants; `async_revoke_federated_token`
   revokes with bounded LRU (10 000 entries).

3. **Capability Sharing Protocol** — `async_negotiate_capability_share`
   handshakes reciprocal offers; `async_federated_inventory_reconcile` diffs
   the agreed contract against live shares to surface drift.

4. **Distributed Circuit Breaker** — `async_chaos_fault_inject` trips breakers
   automatically for high-intensity crash/error faults; `circuit_breaker_reset`
   requires explicit approval. Audit trail on every state change.

5. **Real-Time Dependency Health Probing** — `async_probe_dependency_health`
   accepts a pluggable `probe_fn`; `async_probe_all_dependencies` fans out in
   parallel across all tenant dependencies.

6. **Change Risk Scoring Engine** — `async_score_change_risk` computes a 0–100
   composite score (blast radius 40 pts, dependency health 30 pts, review
   completeness 20 pts, rollback readiness 10 pts) and returns a risk band plus
   recommended actions.

7. **Baseline Drift Detection** — `async_detect_baseline_drift` diffs an
   approved configuration baseline against a live snapshot, returns structured
   drift report (changed / added / removed keys), and emits an audit event when
   drift exceeds the configurable threshold.

8. **SLA Contract Enforcement** — `async_sla_contract_register` stores
   availability, latency-p99, error-rate, RPO, and RTO targets;
   `async_sla_evaluate` computes compliance and returns breach events.

9. **Policy-as-Code Rule Hot-Swap** — `config_hot_reload` reloads active
   configuration keys for an environment without restart, with full audit trail
   and version tracking.

10. **Federated Service Mesh** — `async_service_discover_nearest` resolves the
    lowest-latency healthy endpoint given requester region, with weighted
    random selection across the 10% latency tolerance band.

11. **Canary Release Orchestration** — `async_canary_release_start` /
    `async_canary_release_advance` / `async_canary_release_abort` manage
    progressive traffic shifting with automatic promotion at 100% and instant
    rollback.

12. **Audit Event Streaming with Back-Pressure** — `async_verify_audit_chain`
    verifies Merkle-chained SHA-256 integrity across the full tenant audit log;
    tampered events are flagged per-entry.

13. **Cross-Platform Capability Versioning** — `async_federated_inventory_reconcile`
    surfaces capabilities that are agreed but missing, or active but not in the
    agreed contract, enabling semver-gated federation governance.

14. **Observability Telemetry Export** — `platform_metrics_dashboard` and
    `platform_analytics` aggregate all subsystem metrics into a snapshot
    suitable for OTLP export; the adapter binding is injected by the host.

15. **Federated Identity Broker** — `async_federated_token_exchange` plus
    `async_trace_context_propagate` provide W3C traceparent propagation and
    OAuth2-compliant identity assertions across tenant boundaries.

## Federation API Table

| Method | Purpose |
|--------|---------|
| `async_federated_token_exchange` | OAuth2 RFC 8693 token exchange between tenants |
| `async_revoke_federated_token` | Revoke an issued assertion token with bounded LRU revocation list |
| `async_negotiate_capability_share` | Runtime capability-sharing negotiation with reciprocal offers |
| `async_federated_inventory_reconcile` | Diff agreed capability contract against live shares; detect drift |
| `async_health_check_all_services` | Concurrent fan-out health probing via `asyncio.gather` |
| `async_probe_dependency_health` | Active single-dependency liveness probe with pluggable probe_fn |
| `async_probe_all_dependencies` | Parallel fan-out across all tenant dependencies |
| `async_score_change_risk` | Composite risk scoring (blast radius, dep health, review, rollback) |
| `async_detect_baseline_drift` | Diff approved baseline against live config snapshot |
| `async_sla_contract_register` | Register SLA targets (availability, latency, error-rate, RPO, RTO) |
| `async_sla_evaluate` | Evaluate SLA compliance from a metrics window |
| `async_canary_release_start` | Start a canary release with configurable traffic split |
| `async_canary_release_advance` | Advance canary traffic; promotes at 100% |
| `async_canary_release_abort` | Abort canary and roll back to baseline |
| `async_cost_budget_gate` | Decimal-precision cumulative cost gate with ISO 4217 currency |
| `async_chaos_fault_inject` | Inject latency/error/crash/partial faults for chaos engineering |
| `async_chaos_fault_remove` | Remove an active chaos fault, restoring normal operation |
| `async_service_discover_nearest` | Topology-aware nearest-endpoint discovery with weighted selection |
| `async_verify_audit_chain` | Verify Merkle-chained audit log integrity; detect tampered events |
| `async_predict_readiness_trajectory` | OLS linear-regression readiness velocity and predicted-ready date |
| `async_trace_context_propagate` | W3C traceparent propagation across tenant boundaries |

## New Methods — Usage Examples

### 1. Federated token exchange and revocation

```python
import asyncio
from capabilities.common.plfd import PlfdService

service = PlfdService()

# Issue an assertion allowing tenant-a to act on tenant-b
token = await service.async_federated_token_exchange(
    source_tenant="tenant-a",
    target_tenant="tenant-b",
    scopes=["read:services", "write:config"],
    issuer_token="<jwt-from-idp>",
    requested_by="platform-admin",
)
# token["assertion_id"] is the handle for revocation

# Revoke when done
await service.async_revoke_federated_token(
    tenant_id="tenant-a",
    assertion_id=token["assertion_id"],
    reason="session_ended",
)
```

### 2. Change risk scoring before approval

```python
service.propose_platform_change(
    change_id="chg-007",
    tenant_id="tenant-demo",
    service_id="plfd-core",
    title="Upgrade auth layer",
    owner="infra-team",
    affected_capability_count=12,
    rollback_plan_ref="rollback-auth-v2",
)

risk = await service.async_score_change_risk("chg-007", "tenant-demo")
# risk["risk_score"]  -> 0-100
# risk["risk_band"]   -> "low" | "medium" | "high" | "critical"
# risk["recommended_actions"] -> ["complete_security_review", ...]

if risk["risk_band"] in {"high", "critical"}:
    raise RuntimeError(f"Change blocked: {risk['risk_band']} risk")
```

### 3. Canary release lifecycle

```python
canary = await service.async_canary_release_start(
    tenant_id="tenant-demo",
    service_name="payments-api",
    canary_version="2.4.0",
    baseline_version="2.3.1",
    initial_traffic_pct=5.0,
    started_by="release-bot",
)

# Observe metrics, then advance
await service.async_canary_release_advance(
    tenant_id="tenant-demo",
    canary_id=canary["canary_id"],
    new_traffic_pct=25.0,
)

# Abort on anomaly
await service.async_canary_release_abort(
    tenant_id="tenant-demo",
    canary_id=canary["canary_id"],
    reason="p99_latency_spike",
)
```

### 4. Baseline drift detection

```python
live_snapshot = {
    "db.max_connections": 200,   # was 100 in baseline
    "cache.ttl_seconds": 3600,
    "new_feature.enabled": True,  # added since baseline
}

drift = await service.async_detect_baseline_drift(
    tenant_id="tenant-demo",
    service_id="plfd-core",
    live_config_snapshot=live_snapshot,
    drift_threshold=0.05,  # alert if >5% of keys drifted
)
# drift["drift_detected"]  -> True
# drift["changed_keys"]    -> ["db.max_connections"]
# drift["added_keys"]      -> ["new_feature.enabled"]
```

### 5. Active dependency health probe with fan-out

```python
async def http_probe(dep: dict) -> str:
    """Return 'healthy' | 'degraded' | 'unhealthy' based on live check."""
    import aiohttp
    url = dep.get("evidence_ref", "")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as r:
                return "healthy" if r.status == 200 else "degraded"
    except Exception:
        return "unhealthy"

# Fan-out across all tenant dependencies
result = await service.async_probe_all_dependencies(
    tenant_id="tenant-demo",
    probe_fn=http_probe,
)
# result["healthy"] / result["degraded"] / result["unhealthy"] counts
# result["results"] contains per-dependency detail
```

## Basic Usage

```python
from capabilities.common.plfd import PlfdService

service = PlfdService()
service.register_foundation_service(
    service_id="plfd-core",
    tenant_id="tenant-demo",
    name="Platform foundation",
    owner="foundation-owner",
    tier="core",
    readiness_score=94,
    monitoring_enabled=True,
    rollback_plan_ref="rollback-plfd",
    change_window_ref="cw-001",
)
service.attach_baseline(
    baseline_id="base-configuration",
    tenant_id="tenant-demo",
    service_id="plfd-core",
    baseline_type="configuration",
    evidence_ref="evidence-configuration",
    approved_by="platform-reviewer",
)
```

## AI Foundation Agents

```python
agent = service.register_plfd_agent(
    tenant_id="tenant-demo",
    name="Readiness reviewer",
    runtime="codex",
    role="readiness_reviewer",
    scope="Review dependency, baseline, monitoring, and rollback gates",
)
```

Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.
Supported roles cover foundation, dependency, baseline, readiness, change,
and security review.

## Composition

PLFD composes with:

- `conf` for platform and tenant configuration.
- `mten` for tenant baseline requirements.
- `auth` for identity, permissions, and foundation RBAC.
- `audl` for durable audit evidence.
- `moni` and `hlth` for health and monitoring evidence.
- `regy` for platform service registry integration.
- `secu` for security review evidence.
- `plgn` for plugin registry governance.

Batch foundation mutation and change lifecycle events must use the `bytewax`
event-stream adapter.

## Main Files

- `SPECIFICATION.md` — normative capability behavior.
- `PLAN.md` — implementation packet plan.
- `capability_contract.py` — executable configuration, rules, routes, theme,
  adapters, provides/requires, and Bytewax stream metadata.
- `models.py` — tenant-scoped services, dependencies, baselines, readiness
  assessments, changes, audit events, and agents.
- `foundation_runtime.py` — deterministic IDs, tier/health/baseline
  normalization, readiness posture, and change-review helpers.
- `service.py` — runtime facade (`PlatformFoundationService` / `PlfdService`).
- `api.py` — package-safe helper functions.
- `views.py` — UI view models.
- `test_capability_contract.py` — lifecycle and generated-evidence proofs.

## Verification

```bash
./.venv/bin/python -m py_compile \
    capabilities/common/plfd/__init__.py \
    capabilities/common/plfd/capability_contract.py \
    capabilities/common/plfd/models.py \
    capabilities/common/plfd/foundation_runtime.py \
    capabilities/common/plfd/service.py \
    capabilities/common/plfd/api.py \
    capabilities/common/plfd/views.py \
    capabilities/common/plfd/app.py \
    capabilities/common/plfd/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/plfd/test_capability_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/plfd --json
./.venv/bin/apg capabilities publish-plan capabilities/common/plfd --json
```

Live configuration stores, tenant registries, identity systems, audit stores,
monitoring providers, rendered UI, and Bytewax workers are integration
concerns outside the package proof.
