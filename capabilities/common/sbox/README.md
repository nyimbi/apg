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
- `service.py` implements the runtime facade.
- `api.py` exposes package-safe helper functions.
- `views.py` exposes UI view models.
- `test_capability_contract.py` proves lifecycle behavior and generated
  evidence.

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
    template_id=template["id"],
    isolation_profile_id=isolation["id"],
    owner="qa-owner",
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

## Async Method Reference

| Method | Description |
|--------|-------------|
| `async_create_sandbox(...)` | Non-blocking sandbox provisioning |
| `async_start_run(...)` | Non-blocking run initiation |
| `async_complete_run(...)` | Non-blocking run finalization |
| `async_simulate_event(..., delivery_delay_ms)` | Event delivery with latency control |
| `async_parallel_scenario_run(..., max_concurrency)` | Concurrent scenario execution |
| `async_chaos_inject_and_observe(...)` | Inject fault and collect time-series observations |
| `async_load_and_validate_dataset(..., strict)` | Load and schema-validate records |
| `async_snapshot_and_restore(...)` | Snapshot/reset/restore isolation primitive |
| `async_security_posture_report(...)` | Multi-dimension security scoring |
| `async_quota_check(...)` | Resource quota enforcement |

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

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`.
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
./.venv/bin/python -m py_compile capabilities/common/sbox/__init__.py capabilities/common/sbox/capability_contract.py capabilities/common/sbox/models.py capabilities/common/sbox/sandbox_runtime.py capabilities/common/sbox/service.py capabilities/common/sbox/api.py capabilities/common/sbox/views.py capabilities/common/sbox/app.py capabilities/common/sbox/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/sbox/test_capability_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/sbox --json
./.venv/bin/apg capabilities publish-plan capabilities/common/sbox --json
```

Live container execution, external network enforcement, secret vault calls,
data masking engines, rendered UI, and Bytewax workers are integration
concerns outside the package proof.
