# SBOX Sandbox/Testing Environment Capability

SBOX gives APG applications a tenant-scoped safe execution runtime: isolation
profiles, sandbox templates, controlled datasets, sandbox environments, test
runs, run completion evidence, sandbox governance agents, UI metadata, theme
tokens, audit trails, and Bytewax-backed lifecycle events.

The package stays dependency-light. Production container runtimes, deployment
systems, secret stores, data masking services, plugin test harnesses, logging
systems, audit sinks, and Bytewax workers are represented as APG adapters in
the executable contract and are bound by the host application.

## What It Provides

- Isolation profiles for network, data, secret, outbound access, approval, and
  masking posture.
- Sandbox template library with runtime, owner, TTL, plugin-test policy, and
  tags.
- Dataset manager with synthetic, fixture, masked, and reviewed production
  sample datasets.
- Sandbox registry with tenant, owner, template, isolation profile, TTL,
  datasets, risk score, lifecycle review, and state.
- Run monitor for unit, integration, plugin, agent, migration, and load runs
  with requested, passed, failed, blocked, log, and completion state.
- Mock service registry with per-route responses, auth, and chaos configuration.
- Domain event simulation with async subscription and backpressure-aware delivery.
- Chaos fault injection (latency, error_rate, partition, cpu_pressure) with
  time-series observation.
- Structured test scenario DSL with typed steps and per-step assertions.
- Differential dataset comparison with schema drift detection.
- Flakiness detection with variance-based scoring and quarantine recommendations.
- WASM module registry with SHA-256 integrity verification and supply-chain signing.
- Decimal-precise cost tracking with budget alert thresholds.
- Multi-dimension security posture reporting (grade A–D).
- Resource quota enforcement with configurable limits per tenant.
- First-class AI sandbox agents with runtime, role, scope, registration, and
  contribution-disclosure guardrails.
- UI route, API, view-model, theme, semantic-model, package-manifest, and
  release-report evidence.

## Main Files

- `SPECIFICATION.md` defines the normative capability behavior.
- `PLAN.md` records the implementation packet plan.
- `capability_contract.py` is the executable source of configuration, rules,
  routes, theme, adapters, provides/requires, and Bytewax stream metadata.
- `models.py` defines tenant-scoped isolation profiles, templates, datasets,
  sandboxes, runs, audit events, and agents.
- `sandbox_runtime.py` contains deterministic IDs, normalization helpers, risk
  scoring, sandbox state, run status, and policy summaries.
- `service.py` implements the runtime facade (`SandboxTestingService` / `SboxService`).
- `api.py` exposes package-safe helper functions.
- `views.py` exposes UI view models.
- `test_capability_contract.py` proves lifecycle behavior and generated evidence.

## Basic Usage

```python
from capabilities.common.sbox import SboxService

service = SboxService()
isolation = service.create_isolation_profile(
    tenant_id="tenant-demo",
    name="strict-network",
    level="strict",
    approved_by="security-reviewer",
)
template = service.create_template(
    tenant_id="tenant-demo",
    name="python-plugin-tests",
    runtime="python",
    owner="platform-owner",
)
dataset = service.register_dataset(
    tenant_id="tenant-demo",
    name="safe-fixture",
    dataset_type="fixture",
    owner="qa-owner",
    lineage="fixture://safe-fixture",
    retention_days=30,
)
sandbox = service.create_sandbox(
    tenant_id="tenant-demo",
    name="plugin-check",
    template=template["id"],
    isolation_profile_id=isolation["id"],
    owner_id="qa-owner",
    expiry_hours=4,
    dataset_ids=[dataset["id"]],
)
run = service.start_run(
    tenant_id="tenant-demo",
    sandbox_id=sandbox["id"],
    run_type="plugin",
    requested_by="qa-owner",
    tests_requested=12,
)
service.complete_run("tenant-demo", run["id"], tests_passed=12)
```

## Async Usage

```python
import asyncio
from capabilities.common.sbox import SboxService

svc = SboxService()

async def run_parallel_tests():
    sandbox = await svc.async_create_sandbox(
        name="async-ci-sandbox",
        template="python",
        owner_id="ci-bot",
        expiry_hours=2,
        tenant_id="tenant-ci",
        lifecycle_review_recorded=True,
    )
    result = await svc.async_parallel_scenario_run(
        sandbox_id=sandbox["id"],
        scenario_ids=["auth-flow", "payment-flow", "order-flow"],
        tenant_id="tenant-ci",
        run_type="integration",
        requested_by="ci-bot",
        tests_per_scenario=10,
        max_concurrency=4,
    )
    print(f"Passed: {result['passed_scenario_count']}/{result['scenario_count']}")
    posture = await svc.async_security_posture_report(
        sandbox_id=sandbox["id"],
        tenant_id="tenant-ci",
    )
    assert posture["posture_grade"] in {"A", "B"}, posture["recommendations"]

asyncio.run(run_parallel_tests())
```

## Synchronous API Reference

| Method | Description |
|--------|-------------|
| `create_isolation_profile(tenant_id, name, level, ...)` | Define network/secret/data isolation posture |
| `create_template(tenant_id, name, runtime, owner, ...)` | Register a sandbox template |
| `register_dataset(tenant_id, name, dataset_type, ...)` | Register a dataset with lineage and retention |
| `create_sandbox(name, template, owner_id, expiry_hours, ...)` | Provision a sandbox environment |
| `reset_sandbox(sandbox_id, tenant_id, ...)` | Clear mocks and data; return to ready state |
| `destroy_sandbox(sandbox_id, reason, ...)` | Permanently destroy a sandbox |
| `sandbox_status(sandbox_id, tenant_id)` | Detailed status including mock/data/run counts |
| `start_run(tenant_id, sandbox_id, run_type, ...)` | Initiate a test run |
| `complete_run(tenant_id, run_id, tests_passed, ...)` | Finalize a run with pass/fail counts |
| `expire_sandbox(tenant_id, sandbox_id, actor)` | Mark a sandbox expired |
| `load_test_data(sandbox_id, dataset_name, ...)` | Load a named dataset into a sandbox |
| `mock_service_register(sandbox_id, service_name, mock_config, ...)` | Register a mock service endpoint |
| `api_mock_advanced(sandbox_id, service_name, routes, ...)` | Register a mock with per-route responses and auth |
| `simulate_event(sandbox_id, event_type, payload, ...)` | Simulate a domain event; broadcasts to async subscribers |
| `run_test_scenario(sandbox_id, scenario_id, ...)` | Execute a named scenario |
| `parallel_test_run(sandbox_id, scenario_ids, ...)` | Run multiple scenarios and aggregate results |
| `test_data_generate(sandbox_id, schema, ...)` | Generate synthetic records matching a schema |
| `assertion_check(sandbox_id, run_id, assertions, ...)` | Evaluate assertions against run state |
| `cleanup_after_test(sandbox_id, ...)` | Remove data and mocks post-run |
| `chaos_inject(sandbox_id, fault_type, ...)` | Inject a chaos fault record |
| `load_simulate(sandbox_id, concurrent_users, ...)` | Project load metrics |
| `environment_snapshot(sandbox_id, ...)` | Capture sandbox state for audit or restore |
| `sandbox_cost_tracking(sandbox_id, ...)` | Record float-precision cost data |
| `sandbox_analytics(period, tenant_id)` | Aggregated tenant analytics |
| `coverage_report(sandbox_id, ...)` | Test coverage estimate from run history |
| `benchmark_run(sandbox_id, operation, ...)` | Micro-benchmark a named operation |
| `register_sbox_agent(tenant_id, name, runtime, role, ...)` | Register an AI sandbox agent |
| `list_sbox_agents(tenant_id)` | List registered agents |
| `dashboard_summary(tenant_id)` | Counts-at-a-glance across all resource types |
| `validate_batch_sandbox_mutation(event_stream)` | Validate that batch mutations route through Bytewax |
| `describe(tenant_id)` | Return full capability contract |
| `evaluate(context)` | Evaluate policy rules against a context dict |

## Async Method Reference

### Core async methods

| Method | Description |
|--------|-------------|
| `async_create_sandbox(...)` | Non-blocking sandbox provisioning |
| `async_start_run(...)` | Non-blocking run initiation |
| `async_complete_run(...)` | Non-blocking run finalization |
| `async_simulate_event(..., delivery_delay_ms)` | Event delivery with latency control |
| `async_parallel_scenario_run(..., max_concurrency)` | Concurrent scenario execution via `asyncio.Semaphore` |
| `async_chaos_inject_and_observe(...)` | Inject fault and collect time-series status observations |
| `async_load_and_validate_dataset(..., strict)` | Load and schema-validate records; raises on violation when strict=True |
| `async_snapshot_and_restore(...)` | Snapshot → reset → restore isolation primitive |
| `async_security_posture_report(...)` | Multi-dimension security scoring (grade A–D) |
| `async_quota_check(...)` | Resource usage vs. configurable limits; returns breach list |

### New async methods (v2.0 world-class improvements)

| Method | Category | Description |
|--------|----------|-------------|
| `async_guard_tenant(tenant_id)` | Security | Tenant validation guard — call at any async entry point |
| `async_cost_tracking_decimal(...)` | FinOps | Decimal-precise cost recording with budget alert threshold |
| `async_subscribe_events(sandbox_id, event_types)` | Events | Subscribe to sandbox events via `asyncio.Queue`; returns `(token, queue)` |
| `async_unsubscribe_events(token)` | Events | Unsubscribe and drain a subscription |
| `async_define_scenario(scenario_id, steps)` | Test DSL | Define a structured scenario with typed steps and per-step assertions |
| `async_execute_scenario(sandbox_id, scenario_id)` | Test DSL | Execute a defined scenario step-by-step with abort/continue control |
| `async_dataset_diff(sandbox_id, name_a, name_b)` | Data Quality | Structural diff of two loaded datasets: added/removed keys, changed values, schema drift |
| `async_flakiness_score(scenario_id)` | Reliability | Variance-based flakiness score (0–1) with quarantine recommendation |
| `async_register_wasm_module(name, bytes, signer_id)` | Security | SHA-256–verified WASM artifact registry with trust flag |
| `async_simulate_policy(context)` | Governance | Dry-run policy evaluation — no side effects |

## World-Class Enhancements (v2.0)

These 15 improvements address the gaps between a working sandbox and a
production-grade multi-tenant test platform.

| # | Name | Category | What Changed |
|---|------|----------|--------------|
| I1 | True Async-Native Service Layer | Architecture | Every hot path (`async_create_sandbox`, `async_start_run`, `async_simulate_event`, etc.) yields to the event loop; concurrent test orchestrators no longer need thread-pool bridging. |
| I2 | WASM Isolation Backend | Isolation | `async_register_wasm_module` stores SHA-256–verified artifacts; `trusted` flag gates execution; groundwork for per-sandbox Wasmtime/Extism linear-memory execution. |
| I3 | Snapshot/Restore with COW Semantics | State Management | `async_snapshot_and_restore` captures state, resets, and restores to the snapshot point; enables flaky-test bisection and hermetic test isolation. |
| I4 | Resource Quota Enforcement | Multi-Tenancy | `async_quota_check` reports breaches against per-tenant limits for sandboxes, active runs, and mock services; callers decide to block or warn. |
| I5 | Structured Test Scenario DSL | Test Orchestration | `async_define_scenario` / `async_execute_scenario` support `simulate_event`, `load_data`, and `assert` steps with per-step `on_failure` (abort/continue/retry) and static validation at definition time. |
| I6 | Pluggable Chaos Fault Scheduler | Chaos Engineering | `chaos_inject` / `async_chaos_inject_and_observe` records faults with duration; `simulate_event` broadcasts through subscriber queues even under active faults; time-series observations captured. |
| I7 | Dependency-Aware Sandbox Cloning | Developer Experience | `create_sandbox` auto-resolves or creates templates and isolation profiles; the full sandbox graph (template, isolation, datasets) is reproduced without manual rebuild. |
| I8 | Real-Time Event Bus with Async Subscription | Event Architecture | `async_subscribe_events` returns an `asyncio.Queue`; `simulate_event` broadcasts to all matching subscribers with backpressure (full queues drop silently); `async_unsubscribe_events` cleans up. |
| I9 | Execution Tracing and Latency Measurement | Observability | `async_simulate_event(delivery_delay_ms=N)` measures and returns `actual_delivery_latency_ms`; `async_chaos_inject_and_observe` produces timestamped observation series for flamegraph analysis. |
| I10 | Policy-as-Code Rule Engine with Dry-Run | Governance | `async_simulate_policy(context)` evaluates the full rule set against an arbitrary context dict and returns decision + human-readable summary with zero side effects. |
| I11 | Cross-Tenant Sandbox Federation | Platform Architecture | `tenant_id` isolation is enforced at every mutating entry point; `async_guard_tenant` wraps all async paths; audit events are written per-tenant. |
| I12 | Differential Dataset Comparison | Data Quality | `async_dataset_diff` produces `added_keys`, `removed_keys`, `changed_values`, `schema_drift`, and `record_count_delta`; supports tolerance thresholds; integrates with `assertion_check`. |
| I13 | Flakiness Detection and Quarantine | Test Reliability | `async_flakiness_score` computes variance over recent N runs (0=stable, 1=maximally flaky); scores ≥ 0.5 trigger a `quarantine` recommendation; ≥ 0.3 trigger `monitor`. |
| I14 | WASM Module Registry with Supply-Chain Signing | Security | `async_register_wasm_module` stores name, version, SHA-256, signer ID, and trusted flag; untrusted modules emit a `warning`-severity audit event on registration. |
| I15 | Decimal-Precise Cost Tracking with Budget Alerts | FinOps | `async_cost_tracking_decimal` uses `Decimal` with `ROUND_HALF_UP` throughout; cumulative spend checked against `monthly_budget`; a `budget_alert` audit event fires when `alert_threshold` fraction is crossed. |

## New Methods — Usage Examples

### 1. Async Event Bus Subscription

Test code can `await` event delivery instead of polling.

```python
import asyncio
from capabilities.common.sbox import SboxService

async def test_order_event():
    svc = SboxService()
    # ... create sandbox ...
    token, queue = await svc.async_subscribe_events(
        sandbox_id=sandbox["id"],
        event_types=["order.created"],
        tenant_id="tenant-ci",
    )
    await svc.async_simulate_event(
        sandbox_id=sandbox["id"],
        event_type="order.created",
        payload={"order_id": "ORD-001", "amount": 99.99},
        tenant_id="tenant-ci",
    )
    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event["event_type"] == "order.created"
    await svc.async_unsubscribe_events(token)
```

### 2. Structured Test Scenario DSL

Define scenarios with explicit steps and per-step assertion control.

```python
scenario = await svc.async_define_scenario(
    scenario_id="payment-auth-flow",
    tenant_id="tenant-ci",
    description="Auth, simulate payment event, assert delivered",
    steps=[
        {
            "action": "load_data",
            "target": "users",
            "params": {"dataset_name": "user-fixtures", "record_count": 5},
            "on_failure": "abort",
        },
        {
            "action": "simulate_event",
            "target": "payment-service",
            "params": {"event_type": "payment.initiated", "payload": {"amount": "50.00"}},
            "on_failure": "abort",
        },
        {
            "action": "assert",
            "assertion": {"field": "delivered", "expected": True},
            "on_failure": "continue",
        },
    ],
)
result = await svc.async_execute_scenario(
    sandbox_id=sandbox["id"],
    scenario_id="payment-auth-flow",
    tenant_id="tenant-ci",
    requested_by="qa-bot",
)
assert result["passed"], result["step_results"]
```

### 3. Decimal-Precise Cost Tracking with Budget Alert

```python
record = await svc.async_cost_tracking_decimal(
    sandbox_id=sandbox["id"],
    tenant_id="tenant-finance",
    resource_costs={"compute": "1.25", "storage": "0.03", "egress": "0.12"},
    currency="USD",
    monthly_budget="10.00",
    alert_threshold=0.8,   # fires budget_alert audit event at 80%
    recorded_by="billing-agent",
)
# total_cost is a Decimal string — no float rounding error
print(record["total_cost"])        # "1.40"
print(record["budget_status"])     # "ok" | "alert"
```

### 4. Differential Dataset Comparison

Validate ETL output or schema evolution between two dataset snapshots.

```python
await svc.async_load_and_validate_dataset(
    sandbox_id=sandbox["id"],
    dataset_name="users-v1",
    records=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
    schema={"id": "int", "name": "str"},
    tenant_id="tenant-ci",
)
await svc.async_load_and_validate_dataset(
    sandbox_id=sandbox["id"],
    dataset_name="users-v2",
    records=[{"id": 1, "name": "Alice", "email": "a@x.com"}, {"id": 3, "name": "Carol"}],
    schema={"id": "int", "name": "str", "email": "str"},
    tenant_id="tenant-ci",
    strict=False,
)
diff = await svc.async_dataset_diff(
    sandbox_id=sandbox["id"],
    dataset_name_a="users-v1",
    dataset_name_b="users-v2",
    tenant_id="tenant-ci",
)
# diff["added_keys"], diff["removed_keys"], diff["schema_drift"] ...
```

### 5. Flakiness Score and Quarantine Decision

```python
score = await svc.async_flakiness_score(
    scenario_id="payment-auth-flow",
    tenant_id="tenant-ci",
    window=20,
)
# score["flakiness_score"]:  0.0 = stable, 1.0 = maximally flaky
# score["recommendation"]:   "stable" | "monitor" | "quarantine"
if score["recommendation"] == "quarantine":
    print(f"Scenario is flaky ({score['flakiness_score']:.2f}). Exclude from CI gates.")
```

## AI Sandbox Agents

Register AI agents before they assist with sandbox governance:

```python
agent = service.register_sbox_agent(
    tenant_id="tenant-demo",
    name="Isolation reviewer",
    runtime="codex",
    role="isolation_reviewer",
    scope="Review network, data, secret, TTL, and Bytewax stream guardrails",
)
```

Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.
Supported roles cover isolation, dataset, run, plugin-test, security, and
lifecycle review.

## Composition

SBOX composes with:

- `plgn` for plugin test policies and extension validation.
- `secu` for isolation, network, data, and secret controls.
- `envm` for environment templates and lifecycle posture.
- `audl` for durable audit evidence.
- `depl`, `cicd`, and `logt` for execution, pipeline, and diagnostic
  integration.

Batch sandbox mutation and sandbox run lifecycle events must use the `bytewax`
event-stream adapter.

## Verification

Focused verification for this packet:

```bash
./.venv/bin/python -m py_compile \
    capabilities/common/sbox/__init__.py \
    capabilities/common/sbox/capability_contract.py \
    capabilities/common/sbox/models.py \
    capabilities/common/sbox/sandbox_runtime.py \
    capabilities/common/sbox/service.py \
    capabilities/common/sbox/api.py \
    capabilities/common/sbox/views.py \
    capabilities/common/sbox/app.py \
    capabilities/common/sbox/test_capability_contract.py

./.venv/bin/pytest -q capabilities/common/sbox/test_capability_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/sbox --json
./.venv/bin/apg capabilities publish-plan capabilities/common/sbox --json
```

Live container execution, external network enforcement, secret vault calls,
data masking engines, rendered UI, and Bytewax workers are integration
concerns outside the package proof.

---

*Datacraft © 2025 — Author: Nyimbi Odero*
