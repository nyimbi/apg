# Sandbox/Testing Environment — User Guide

**Capability ID**: `sbox` | **Domain**: `common` | **Version**: `1.2.0`
**Author**: Nyimbi Odero — Datacraft (c) 2025

---

## Overview

SBOX provides APG applications with a tenant-scoped safe execution runtime
for testing, sandboxing, chaos engineering, and code isolation. Every
operation is governance-enforced via policy evaluation, tenant-scoped, and
fully audited.

Key design decisions:
- Adapter/store pattern with no external runtime dependencies.
- Sync facade for simple scripts; async variants for concurrent pipelines.
- All mutating operations emit audit events with actor attribution.
- Risk scoring surfaces governance concerns before sandbox creation.

---

## Installation

```bash
pip install apg-common-sbox
```

---

## Provides

| Service | Description |
|---------|-------------|
| `sandbox_registry` | Tenant-scoped sandbox lifecycle management |
| `isolation_profiles` | Network, secret, and data isolation postures |
| `test_runs` | Run orchestration with pass/fail/blocked tracking |
| `synthetic_datasets` | Dataset generation and schema validation |
| `safety_policy` | Policy evaluation and risk scoring |
| `mock_services` | Per-route mock API registration |
| `chaos_faults` | Fault injection (latency, error_rate, partition) |
| `load_simulation` | Projected throughput and latency metrics |
| `cost_tracking` | Per-sandbox and per-period cost records |
| `audit_trail` | Append-only audit event log |

---

## Requires

| Capability | Purpose |
|------------|---------|
| `plgn` | Plugin test policies and extension validation |
| `secu` | Isolation, network, data, and secret controls |
| `envm` | Environment templates and lifecycle posture |
| `audl` | Durable audit evidence |

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/sbox/dashboard` | `sbox:view` | Overview |
| `/sbox/sandboxes` | `sbox:create` | Sandboxes |
| `/sbox/templates` | `sbox:create` | Templates |
| `/sbox/datasets` | `sbox:manage_policy` | Data |
| `/sbox/runs` | `sbox:run_tests` | Runs |
| `/sbox/agents` | `sbox:admin` | Operations |
| `/sbox/policies` | `sbox:manage_policy` | Governance |
| `/sbox/audit` | `sbox:admin` | Governance |

---

## Getting Started

### 1. Create an Isolation Profile

```python
from capabilities.common.sbox import SboxService

svc = SboxService()

isolation = svc.create_isolation_profile(
    tenant_id="tenant-a",
    name="ci-strict",
    level="strict",            # strict | standard | permissive
    approved_by="sec-team",
    outbound_network_allowed=False,
    secret_redaction_enabled=True,
    data_masking_enabled=True,
)
```

### 2. Register a Template

```python
template = svc.create_template(
    tenant_id="tenant-a",
    name="python-ci",
    runtime="python",          # python | node | go | wasm | docker
    owner="ci-owner",
    default_ttl_hours=4,
    tags=["ci", "python", "unit"],
)
```

### 3. Register a Dataset

```python
dataset = svc.register_dataset(
    tenant_id="tenant-a",
    name="fixture-orders",
    dataset_type="fixture",    # synthetic | fixture | masked | production_sample
    owner="qa-owner",
    lineage="fixture://orders-v3",
    retention_days=90,
    masked=True,
)
```

### 4. Create a Sandbox

```python
sandbox = svc.create_sandbox(
    name="order-service-ci",
    template=template["id"],
    owner_id="ci-bot",
    expiry_hours=4,
    tenant_id="tenant-a",
    isolation_profile_id=isolation["id"],
    dataset_ids=[dataset["id"]],
    lifecycle_review_recorded=True,
)
print(sandbox["state"])       # ready
print(sandbox["risk_score"])
```

### 5. Load Test Data and Register Mocks

```python
svc.load_test_data(
    sandbox_id=sandbox["id"],
    dataset_name="order-fixtures",
    tenant_id="tenant-a",
    data={"orders": [{"id": "o1", "amount": 99.99}]},
    record_count=1,
    loaded_by="ci-bot",
)

svc.mock_service_register(
    sandbox_id=sandbox["id"],
    service_name="payment-gateway",
    mock_config={
        "base_url": "http://mock-payment.internal",
        "response_map": {"/charge": {"status": "ok", "txn_id": "txn-001"}},
        "latency_ms": 30,
        "error_rate": 0.0,
    },
    tenant_id="tenant-a",
    registered_by="ci-bot",
)
```

### 6. Run Tests

```python
run = svc.start_run(
    tenant_id="tenant-a",
    sandbox_id=sandbox["id"],
    run_type="integration",
    requested_by="ci-bot",
    tests_requested=50,
)

result = svc.complete_run(
    tenant_id="tenant-a",
    run_id=run["id"],
    tests_passed=48,
    tests_failed=2,
    logs=["2 failures in payment retry logic."],
)
print(result["status"])    # failed
```

---

## Async Usage

### Parallel Scenario Execution

```python
import asyncio
from capabilities.common.sbox import SboxService

svc = SboxService()

async def ci_pipeline():
    sandbox = await svc.async_create_sandbox(
        name="parallel-ci",
        template="python",
        owner_id="ci-bot",
        expiry_hours=2,
        tenant_id="tenant-ci",
        lifecycle_review_recorded=True,
    )
    result = await svc.async_parallel_scenario_run(
        sandbox_id=sandbox["id"],
        scenario_ids=["auth", "checkout", "refund", "inventory"],
        tenant_id="tenant-ci",
        run_type="integration",
        requested_by="ci-bot",
        tests_per_scenario=20,
        max_concurrency=4,
    )
    print(f"{result['passed_scenario_count']}/{result['scenario_count']} scenarios passed")

asyncio.run(ci_pipeline())
```

### Event Delivery with Latency Control

```python
async def test_event_processing():
    event = await svc.async_simulate_event(
        sandbox_id=sandbox["id"],
        event_type="order.created",
        payload={"order_id": "o-999", "amount": 50.00},
        tenant_id="tenant-ci",
        triggered_by="test-harness",
        delivery_delay_ms=100,
    )
    print(event["actual_delivery_latency_ms"])
```

### Snapshot-and-Restore Test Isolation

```python
async def isolated_test(sandbox_id: str):
    restore = await svc.async_snapshot_and_restore(
        sandbox_id=sandbox_id,
        tenant_id="tenant-ci",
        snapshot_label="before-destructive-test",
        actor="test-runner",
    )
    assert restore["restored_state"] == "ready"
```

### Schema-Validated Dataset Loading

```python
records = [
    {"user_id": 1, "email": "a@b.com", "amount": 9.99, "active": True},
    {"user_id": 2, "email": "c@d.com", "amount": 19.99, "active": False},
]
result = await svc.async_load_and_validate_dataset(
    sandbox_id=sandbox["id"],
    dataset_name="users",
    records=records,
    schema={"user_id": "int", "email": "str", "amount": "float", "active": "bool"},
    tenant_id="tenant-ci",
    strict=True,
)
print(result["valid_count"])
```

---

## Chaos Engineering

### Inject a Fault

```python
fault = svc.chaos_inject(
    sandbox_id=sandbox["id"],
    tenant_id="tenant-a",
    fault_type="latency",
    target_service="payment-gateway",
    severity=0.3,
    duration_seconds=60,
    injected_by="chaos-team",
)
```

### Inject and Observe (Async)

```python
observation = await svc.async_chaos_inject_and_observe(
    sandbox_id=sandbox["id"],
    fault_type="error_rate",
    tenant_id="tenant-a",
    target_service="payment-gateway",
    severity=0.5,
    duration_seconds=10,
    observe_interval_seconds=2.0,
    injected_by="chaos-team",
)
# observation["observations"] is a time-series of sandbox_status snapshots
```

---

## Security Posture

```python
posture = await svc.async_security_posture_report(
    sandbox_id=sandbox["id"],
    tenant_id="tenant-a",
)
print(posture["posture_grade"])          # A | B | C | D
print(posture["overall_posture_score"])  # 0-100
for rec in posture["recommendations"]:
    print(rec)
```

Dimensions (0-100, higher = safer): `network_exposure`, `secret_surface`,
`data_sensitivity`, `ttl_risk`, `isolation_gap`.

---

## Resource Quota Checking

```python
quota = await svc.async_quota_check(
    tenant_id="tenant-a",
    max_sandboxes=20,
    max_active_runs=10,
    max_mock_services=50,
)
if not quota["within_quota"]:
    for breach in quota["breaches"]:
        print(f"{breach['resource']}: {breach['used']}/{breach['limit']}")
```

---

## Load Simulation

```python
load = svc.load_simulate(
    sandbox_id=sandbox["id"],
    tenant_id="tenant-a",
    concurrent_users=100,
    requests_per_second=50.0,
    duration_seconds=300,
    scenario="ramp_up",
)
print(f"p99 latency: {load['projected_p99_latency_ms']}ms")
```

---

## Benchmarking

```python
bench = svc.benchmark_run(
    sandbox_id=sandbox["id"],
    operation="order.create",
    iterations=1000,
    tenant_id="tenant-a",
)
print(f"p50: {bench['p50_ms']}ms  p99: {bench['p99_ms']}ms")
```

---

## Assertions

```python
check = svc.assertion_check(
    sandbox_id=sandbox["id"],
    run_id=run["id"],
    assertions=[
        {"field": "status", "expected": "passed"},
        {"field": "tests_passed", "expected": 48},
    ],
    tenant_id="tenant-a",
    checked_by="ci-bot",
)
print(check["all_passed"])
```

---

## Coverage Reports

```python
coverage = svc.coverage_report(
    sandbox_id=sandbox["id"],
    tenant_id="tenant-a",
    module_paths=["src/orders/", "src/payments/"],
)
print(f"Estimated coverage: {coverage['estimated_coverage_pct']}%")
```

---

## AI Sandbox Agents

```python
agent = svc.register_sbox_agent(
    tenant_id="tenant-a",
    name="Security reviewer",
    runtime="claude_code",
    role="security_reviewer",
    scope="Review network posture, secret handling, and TTL policy compliance.",
    contribution_disclosed=True,
)
```

---

## Audit Trail

```python
events = svc.list_audit_events(tenant_id="tenant-a")
for evt in events[-5:]:
    print(f"{evt['event_type']} | {evt['actor']} | {evt['severity']} | {evt['message']}")
```

---

## Analytics and Cost

```python
analytics = svc.sandbox_analytics(period="2026-06", tenant_id="tenant-a")
print(analytics["pass_rate"], analytics["total_cost"])

cost = svc.sandbox_cost_tracking(
    sandbox_id=sandbox["id"],
    tenant_id="tenant-a",
    period="2026-06",
    resource_costs={"compute": 0.40, "storage": 0.05, "network": 0.01},
)
print(cost["total_cost"])
```

---

## Cleanup

```python
svc.cleanup_after_test(
    sandbox_id=sandbox["id"],
    tenant_id="tenant-a",
    remove_data=True,
    remove_mocks=True,
    actor="ci-bot",
)
svc.destroy_sandbox(
    sandbox_id=sandbox["id"],
    reason="CI pipeline complete.",
    tenant_id="tenant-a",
    destroyed_by="ci-bot",
)
```

---

## New Features (v1.2.0)

### Decimal-Precise Cost Tracking with Budget Alerts

```python
cost = await svc.async_cost_tracking_decimal(
    sandbox_id=sandbox["id"],
    tenant_id="tenant-a",
    resource_costs={"compute": "0.12", "storage": "0.03"},
    monthly_budget="1.00",
    alert_threshold=0.8,   # fire budget_alert at 80% of budget
    currency="USD",
)
print(cost["total_cost"])       # "0.15" (exact Decimal arithmetic)
print(cost["budget_status"])    # "ok" or "alert"
```

### Real-Time Event Subscriptions

```python
token, queue = await svc.async_subscribe_events(
    sandbox_id=sandbox["id"],
    event_types=["order.created", "payment.failed"],
    tenant_id="tenant-a",
)
# simulate_event now broadcasts to all matching queues
svc.simulate_event(sandbox["id"], "order.created", {"id": "o-1"}, "tenant-a")
event = await asyncio.wait_for(queue.get(), timeout=1.0)
assert event["event_type"] == "order.created"
await svc.async_unsubscribe_events(token)
```

### Structured Test Scenario DSL

```python
await svc.async_define_scenario(
    "checkout-flow",
    steps=[
        {
            "action": "simulate_event",
            "target": "order-bus",
            "params": {"event_type": "order.created", "payload": {"id": "o-1"}},
            "on_failure": "abort",
        },
        {
            "action": "assert",
            "params": {},
            "assertion": {"field": "event_type", "expected": "order.created"},
            "on_failure": "continue",
        },
    ],
    tenant_id="tenant-a",
    description="Full checkout event flow",
)

result = await svc.async_execute_scenario(
    sandbox_id=sandbox["id"],
    scenario_id="checkout-flow",
    tenant_id="tenant-a",
    requested_by="ci-bot",
)
print(result["passed"])         # True / False
print(result["step_results"])   # per-step pass/fail detail
```

### Dataset Diff

```python
diff = await svc.async_dataset_diff(
    sandbox_id=sandbox["id"],
    dataset_name_a="baseline",
    dataset_name_b="after-migration",
    tenant_id="tenant-a",
    tolerance_record_count_pct=0.05,  # allow 5% count drift
)
print(diff["added_keys"])       # keys present in b but not a
print(diff["removed_keys"])     # keys present in a but not b
print(diff["schema_drift"])     # fields whose Python type changed
print(diff["within_tolerance"]) # True if record count delta within threshold
```

### Flakiness Detection

```python
score = await svc.async_flakiness_score(
    scenario_id="checkout-flow",
    tenant_id="tenant-a",
    window=20,              # consider last 20 runs
)
print(score["flakiness_score"])     # 0.0 (stable) – 1.0 (maximally flaky)
print(score["recommendation"])      # "stable" | "monitor" | "quarantine"
```

### WASM Module Registry

```python
with open("my_module.wasm", "rb") as f:
    module_bytes = f.read()

record = await svc.async_register_wasm_module(
    name="data-validator",
    module_bytes=module_bytes,
    signer_id="platform-team",
    tenant_id="tenant-a",
    version="2.1.0",
    trusted=False,          # set True after out-of-band signature verification
)
print(record["hash_sha256"])    # 64-char hex SHA-256
print(record["trusted"])        # False until admin approves
```

### Policy Dry-Run

```python
result = await svc.async_simulate_policy(
    context={
        "operation": "create_sandbox",
        "sandbox_owner_assigned": True,
        "ttl_hours": 168,
        "secret_access_requested": True,
        "secret_redaction_enabled": False,
    },
    tenant_id="tenant-a",
)
print(result["decision"])   # "allow" | "deny" | "require_review"
print(result["summary"])    # human-readable reason
```

### Tenant Guard

Use `async_guard_tenant` at the top of any custom async handler to get uniform
tenant validation and early PermissionError before touching state:

```python
async def my_handler(tenant_id: str, sandbox_id: str) -> dict:
    await svc.async_guard_tenant(tenant_id)
    ...
```

---

## Complete Async Method Reference

| Method | Description |
|--------|-------------|
| `async_create_sandbox(...)` | Non-blocking sandbox provisioning |
| `async_start_run(...)` | Non-blocking run initiation |
| `async_complete_run(...)` | Non-blocking run finalization |
| `async_simulate_event(..., delivery_delay_ms)` | Event with latency control |
| `async_parallel_scenario_run(..., max_concurrency)` | Concurrent scenarios via asyncio.gather |
| `async_chaos_inject_and_observe(...)` | Fault + time-series observations |
| `async_load_and_validate_dataset(..., strict)` | Load + schema validation |
| `async_snapshot_and_restore(...)` | Snapshot/reset/restore primitive |
| `async_security_posture_report(...)` | Multi-dimension security scoring |
| `async_quota_check(...)` | Resource usage vs. limits |
| `async_guard_tenant(tenant_id)` | Tenant validation guard |
| `async_cost_tracking_decimal(...)` | Decimal-precise costs with budget alerts |
| `async_subscribe_events(sandbox_id, event_types)` | Subscribe to sandbox events via asyncio.Queue |
| `async_unsubscribe_events(token)` | Unsubscribe and drain a subscription |
| `async_define_scenario(scenario_id, steps)` | Define typed scenario with per-step assertions |
| `async_execute_scenario(sandbox_id, scenario_id)` | Execute scenario step-by-step |
| `async_dataset_diff(sandbox_id, name_a, name_b)` | Structural diff of two datasets |
| `async_flakiness_score(scenario_id)` | Variance-based flakiness score with recommendation |
| `async_register_wasm_module(name, bytes, signer_id)` | SHA-256–verified WASM artifact registry |
| `async_simulate_policy(context)` | Dry-run policy evaluation without side effects |

---

## Further Reading

- `service.py` — Complete business logic (sync + async)
- `models.py` — Domain models
- `capability_contract.py` — Policy rules, routes, adapters
- `sandbox_runtime.py` — Deterministic IDs, risk scoring, normalization
- `api.py` — REST API helpers
- `views.py` — Flask-AppBuilder views
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 architectural improvement proposals
