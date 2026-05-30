# DEPL - Deployment Management

DEPL is the APG capability for governed release, deployment, health-gate,
rollback, deployment-agent, audit, and deployment-evidence workflows. It gives
generated APG applications a tenant-aware deployment lifecycle that can be
composed with CI/CD, environment, logging, monitoring, notification, audit,
composition, and health capabilities.

The implementation is dependency-light and side-effect free. It records
deployment state, release evidence, rollout decisions, health gates, rollback
plans, AI deployment-agent governance, lifecycle streams, and audit events
without calling a live cloud provider, Kubernetes cluster, package registry,
ticketing system, notification provider, or observability backend.

## What DEPL Provides

- Deployment environments with tier, owner, policy, approvers, status, and
  tenant isolation.
- Release manifests with version, owner, artifact digest, artifact signature,
  change-ticket evidence, manifest payload, and creator.
- Tested rollback plans tied to release manifests.
- Health gates with check results, health-report references, log-trace links,
  recorded actor, and pass/fail state.
- Deployment plans for rolling, blue-green, and canary rollouts with approval,
  health, rollback, change-ticket, review, and status state.
- Deployment runs with deterministic fingerprints, trace evidence, health
  report references, actor, and completion timestamps.
- Rollback events that move runs and plans into rollback state with reason
  evidence.
- First-class AI deployment agents for Codex, Claude Code, OpenCode, Pi, and
  compatible runtime adapters.
- Deterministic rules for tenant context, release ownership, manifest evidence,
  artifact signatures, change tickets, health gates, production approvals,
  rollback plans, canary review, trace capture, AI deployment agents, state
  change audit, cross-tenant isolation, and Bytewax batch mutation streams.
- View models for dashboards, releases, deployments, rollouts, health gates,
  rollback, agents, audit, analytics, and settings.
- Theme metadata for APG Studio and generated Python applications.

## How To Use It

```python
from capabilities.common.depl.service import DeplService

service = DeplService()
tenant_id = "tenant-depl"

environment = service.register_environment(
    environment_id="prod",
    tenant_id=tenant_id,
    name="Production",
    tier="production",
    owner="platform-owner",
    policy="prod-change-policy",
    approvers=["release-approver"],
)

release = service.create_release(
    release_id="rel-2026-05",
    tenant_id=tenant_id,
    version="2026.05",
    owner="release-owner",
    manifest={"service": "erp-core", "version": "2026.05"},
    artifact_digest="sha256:artifact",
    artifact_signature="sigstore:signature",
    change_ticket="CHG-1001",
    created_by="release-owner",
)

agent = service.register_deployment_agent(
    tenant_id=tenant_id,
    agent_id="codex-rollout-reviewer",
    name="Codex Rollout Reviewer",
    runtime="codex",
    role="health_reviewer",
    scope="Review release health evidence and rollout readiness.",
    contribution_disclosed=True,
    policy_ref="policy:depl:agents:v1",
)

rollback = service.attach_rollback_plan(
    rollback_plan_id="rbp-2026-05",
    tenant_id=tenant_id,
    release_id=release["id"],
    owner="release-owner",
    steps=["switch traffic to previous slot", "restore previous artifact"],
    tested=True,
)

health = service.record_health_gate(
    health_gate_id="hlg-2026-05",
    tenant_id=tenant_id,
    release_id=release["id"],
    checks={"smoke": True, "latency": True, "error_budget": True},
    report_reference="health-report:1001",
    log_trace_link="trace:deploy-1001",
    recorded_by="sre",
)

plan = service.create_deployment_plan(
    plan_id="plan-2026-05",
    tenant_id=tenant_id,
    release_id=release["id"],
    environment_id=environment["id"],
    strategy="canary",
    requested_by="release-owner",
    approval_recorded=True,
    rollback_plan_id=rollback["id"],
    health_gate_id=health["id"],
    change_ticket="CHG-1001",
    canary_percent=10,
)

run = service.execute_deployment(
    run_id="run-2026-05",
    tenant_id=tenant_id,
    plan_id=plan["id"],
    actor="release-owner",
    log_trace_link="trace:deploy-1001",
    health_report_reference="health-report:1001",
)
```

Use `api.py` when composing generated application handlers, and use `views.py`
for framework-neutral screen state:

```python
from capabilities.common.depl.views import dashboard_model, deployment_agents_model

dashboard = dashboard_model(service, tenant_id)
agents = deployment_agents_model(service, tenant_id)
```

## Contract And Composition

`get_capability_contract()` publishes:

- configuration for releases, rollouts, evidence, deployment agents,
  governance, observability, adapters, UI, and theme;
- JSON-schema-style configuration requirements;
- deterministic rule engine;
- UI routes under `/depl`;
- theme tokens under `depl_release_ops`;
- Bytewax lifecycle-stream metadata.

DEPL depends on `logt`, `moni`, and `hlth`. Optional adapter boundaries are
`cicd`, `envm`, `ntfy`, `comp`, `bytewax`, and `audl`.

## Guardrail Summary

DEPL denies or requires review when:

- tenant context is missing;
- a release lacks owner, manifest, artifact signature, or change-ticket
  evidence;
- a health gate lacks recorded checks;
- a deployment lacks a passing health gate, rollback plan, or trace evidence;
- a production deployment lacks approval evidence;
- a large canary rollout lacks review evidence;
- an AI deployment agent is unregistered, uses an unsupported runtime or role,
  lacks explicit scope, or has undisclosed contributions;
- a deployment-plan state change lacks reason or audit evidence;
- a cross-tenant access attempt is detected;
- a batch deployment mutation does not declare Bytewax.

## Focused Verification

Battery-conscious DEPL checks:

```bash
./.venv/bin/python -m py_compile capabilities/common/depl/__init__.py capabilities/common/depl/models.py capabilities/common/depl/deployment_engine.py capabilities/common/depl/service.py capabilities/common/depl/api.py capabilities/common/depl/views.py capabilities/common/depl/capability_contract.py capabilities/common/depl/app.py capabilities/common/depl/test_capability_contract.py capabilities/common/depl/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/depl/test_capability_contract.py capabilities/common/depl/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/depl --json
./.venv/bin/apg capabilities publish-plan capabilities/common/depl --json
```
