# CICD - Continuous Integration and Delivery

CICD is the APG capability for governed build, test, package, scan, promotion,
and release-delivery workflows. It gives generated APG applications a
tenant-aware CI/CD lifecycle that can be composed with deployment, environment,
logging, secret, notification, audit, monitoring, and edge capabilities.

The implementation is dependency-light and side-effect free. It records
pipeline state, build evidence, artifact metadata, quality gates, promotions,
AI delivery-agent governance, lifecycle streams, and audit events without
calling a live source-control provider, build runner, container registry,
scanner, or deployment platform.

## What CICD Provides

- Pipeline definitions with owner, source policy, worker pool, stages, secret
  scope, cache policy, quality gate, parallelism, status, and review state.
- Build runs with commit reference, triggering actor, deterministic trace ID,
  log/trace capture evidence, secret scope, cache policy, and status.
- Build artifacts with name, version, deterministic digest, signature state,
  and availability status.
- Quality gate records with test, security scan, signature, approval, findings,
  and pass/fail state.
- Artifact promotions through source and target environments with approval,
  environment-policy, and separation-of-duties guardrails.
- First-class AI delivery agents for Codex, Claude Code, OpenCode, Pi, and
  compatible runtime adapters.
- Deterministic rules for tenant context, pipeline ownership, source policy,
  workers, stages, secret scope, cache policy, quality gates, trace capture,
  artifact signatures, promotion approval, environment policy, capacity review,
  delivery-agent governance, state-change audit, cross-tenant isolation, and
  Bytewax batch mutation streams.
- View models for dashboards, pipelines, builds, artifacts, quality gates,
  promotions, delivery agents, audit, analytics, and settings.
- Theme metadata for APG Studio and generated Python applications.

## How To Use It

```python
from capabilities.common.cicd.service import CicdService

service = CicdService()
tenant_id = "tenant-ci"

pipeline = service.create_pipeline(
    pipeline_id="orders-api",
    tenant_id=tenant_id,
    name="Orders API",
    owner="delivery-owner",
    source_ref="git://orders-api",
    worker_pool="python-workers",
    stages=["build", "test", "scan", "package"],
    secret_scope="orders-ci",
    cache_policy="python-cache",
    quality_gate="default-release",
)

agent = service.register_delivery_agent(
    tenant_id=tenant_id,
    agent_id="codex-release-reviewer",
    name="Codex Release Reviewer",
    runtime="codex",
    role="security_reviewer",
    scope="Review pipeline gates, release evidence, and promotion readiness.",
    contribution_disclosed=True,
    policy_ref="policy:cicd:agents:v1",
)

build = service.run_build(
    build_id="build-1",
    tenant_id=tenant_id,
    pipeline_id=pipeline["id"],
    commit_ref="abc123",
    triggered_by="developer",
)

artifact = service.publish_artifact(
    artifact_id="artifact-1",
    tenant_id=tenant_id,
    build_id=build["id"],
    name="orders-api",
    version="1.0.0",
    signed=True,
)

gate = service.record_quality_gate(
    gate_id="gate-1",
    tenant_id=tenant_id,
    artifact_id=artifact["id"],
    tests_passed=True,
    security_scan_passed=True,
    approval_recorded=True,
)

promotion = service.promote_artifact(
    promotion_id="promotion-1",
    tenant_id=tenant_id,
    artifact_id=artifact["id"],
    quality_gate_id=gate["id"],
    source_environment="staging",
    target_environment="production",
    requested_by="release-manager",
    approval_recorded=True,
    approver="release-approver",
)
```

Use `api.py` when composing generated application handlers, and use `views.py`
for framework-neutral screen state:

```python
from capabilities.common.cicd.views import dashboard_model, delivery_agents_model

dashboard = dashboard_model(service, tenant_id)
agents = delivery_agents_model(service, tenant_id)
```

## Contract And Composition

`get_capability_contract()` publishes:

- configuration for pipelines, builds, gates, delivery agents, governance,
  observability, adapters, UI, and theme;
- JSON-schema-style configuration requirements;
- deterministic rule engine;
- UI routes under `/cicd`;
- theme tokens under `cicd_pipeline_ops`;
- Bytewax lifecycle-stream metadata.

CICD depends on `depl`, `envm`, and `logt`. Optional adapter boundaries are
`scpt`, `ntfy`, `comp`, `edge`, `bytewax`, `audl`, and `moni`.

## Guardrail Summary

CICD denies or requires review when:

- tenant context is missing;
- a pipeline lacks owner, source policy, worker pool, stages, secret scope,
  cache policy, or quality gate policy;
- pipeline parallelism exceeds configured review expectations without capacity
  review evidence;
- a build lacks secret scope or trace/log capture evidence;
- a quality gate lacks security scan evidence;
- an artifact promotion lacks artifact signature, passing quality gate,
  approval evidence, environment policy, or separation of duties;
- an AI delivery agent is unregistered, uses an unsupported runtime or role,
  lacks explicit scope, or has undisclosed contributions;
- a pipeline state change lacks a reason or audit evidence;
- a cross-tenant access attempt is detected;
- a batch pipeline mutation does not declare Bytewax.

## Focused Verification

Battery-conscious CICD checks:

```bash
./.venv/bin/python -m py_compile capabilities/common/cicd/__init__.py capabilities/common/cicd/capability_contract.py capabilities/common/cicd/models.py capabilities/common/cicd/cicd_engine.py capabilities/common/cicd/service.py capabilities/common/cicd/api.py capabilities/common/cicd/views.py capabilities/common/cicd/app.py capabilities/common/cicd/test_capability_contract.py capabilities/common/cicd/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/cicd/test_capability_contract.py capabilities/common/cicd/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/cicd --json
./.venv/bin/apg capabilities publish-plan capabilities/common/cicd --json
```

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Real DORA Metrics Engine with Timestamp-Based Computation** [Observability / Engineering Excellence]
- **I2. Deployment Cost Tracking with Decimal Precision** [FinOps / Cost Governance]
- **I3. Pipeline-as-Code Schema Validation** [Developer Experience / Correctness]
- **I4. Approval Workflow with Multi-Party Sign-Off** [Governance / Compliance]
- **I5. Flaky Test Detection and Quarantine** [Build Intelligence]
- **I6. Environment Drift Detection** [Deployment Safety]
- **I7. Parallel Stage Execution with DAG Scheduling** [Performance / Pipeline Speed]
- **I8. Secret Rotation Integration with Vault** [Security / Zero-Trust]
- **I9. Artifact Provenance with SLSA Level 3 Evidence** [Supply Chain Security]
- **I10. Canary Analysis with Automated Pass/Fail Decision** [Progressive Delivery]
- **I11. Build Cache Efficiency Reporting** [Developer Experience / Cost]
- **I12. Policy-as-Code Gate Evaluation Engine** [Governance / Extensibility]
- **I13. Multi-Tenant Pipeline Isolation Audit** [Security / Compliance]
- **I14. Pipeline Performance Regression Detection** [Build Intelligence / SRE]
- **I15. GitOps Reconciliation Loop with Desired-State Diff** [Deployment Automation / Reliability]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
