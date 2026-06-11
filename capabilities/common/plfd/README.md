# PLFD Platform Federation Capability

PLFD provides APG applications with a tenant-scoped foundation governance and
**multi-platform federation** runtime: platform service registry, dependency
posture, required baselines, readiness gates, platform change approval,
foundation agents, UI metadata, theme tokens, audit evidence, and
Bytewax-backed lifecycle events.

New in this release: async-first federation APIs covering federated identity
token exchange, capability sharing negotiation, active dependency probing,
change risk scoring, baseline drift detection, SLA contract enforcement, and
canary release orchestration.

The package stays dependency-light. Production configuration stores, tenant
registries, identity providers, audit sinks, monitoring systems, health checks,
security scanners, plugin registries, and Bytewax workers are represented as
APG adapters in the executable contract and are bound by the host application.

## What It Provides

- Foundation service registry with owner, tier, dependency declarations,
  readiness score, monitoring posture, rollback plan, and change window.
- Dependency posture records with source, target, health, required flag, and
  evidence reference.
- Baseline manager for configuration, tenant, auth, and audit baselines.
- Readiness assessment with dependency, baseline, monitoring, rollback, and
  change-window gates.
- Platform change queue with affected capability scope, dependency health,
  approval, broad-review, security-review, change-window, and rollback gates.
- First-class AI foundation agents with runtime, role, scope, registration,
  and contribution-disclosure guardrails.
- UI route, API, view-model, theme, semantic-model, package-manifest, and
  release-report evidence.

### Federation Features (async)

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

## Main Files

- `SPECIFICATION.md` defines the normative capability behavior.
- `PLAN.md` records the implementation packet plan.
- `capability_contract.py` is the executable source of configuration, rules,
  routes, theme, adapters, provides/requires, and Bytewax stream metadata.
- `models.py` defines tenant-scoped services, dependencies, baselines,
  readiness assessments, changes, audit events, and agents.
- `foundation_runtime.py` contains deterministic IDs, tier/health/baseline
  normalization, readiness posture, and change-review helpers.
- `service.py` implements the runtime facade.
- `api.py` exposes package-safe helper functions.
- `views.py` exposes UI view models.
- `test_capability_contract.py` proves lifecycle behavior and generated
  evidence.

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

Register AI agents before they assist with foundation governance:

```python
agent = service.register_plfd_agent(
    tenant_id="tenant-demo",
    name="Readiness reviewer",
    runtime="codex",
    role="readiness_reviewer",
    scope="Review dependency, baseline, monitoring, and rollback gates",
)
```

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`.
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

## Verification

Focused verification for this packet:

```bash
./.venv/bin/python -m py_compile capabilities/common/plfd/__init__.py capabilities/common/plfd/capability_contract.py capabilities/common/plfd/models.py capabilities/common/plfd/foundation_runtime.py capabilities/common/plfd/service.py capabilities/common/plfd/api.py capabilities/common/plfd/views.py capabilities/common/plfd/app.py capabilities/common/plfd/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/plfd/test_capability_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/plfd --json
./.venv/bin/apg capabilities publish-plan capabilities/common/plfd --json
```

Live configuration stores, tenant registries, identity systems, audit stores,
monitoring providers, rendered UI, and Bytewax workers are integration
concerns outside the package proof.
