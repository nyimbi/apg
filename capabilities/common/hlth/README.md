# HLTH - Health Checks and Diagnostics

HLTH is APG's tenant-scoped health checks and diagnostics capability. It gives
generated applications a dependency-light control plane for registering
components, recording health checks, maintaining baselines, opening alerts and
incidents, reviewing remediation, gating deployments, and publishing UI/theme
metadata.

The current packet can be composed without starting ML engines, Kubernetes
watchers, external observability backends, notification systems, ticketing
systems, remediation runners, or production databases. Those systems are
runtime adapters that must honor HLTH guardrail decisions.

## What HLTH Provides

- Tenant-aware component registration with owner, environment, criticality,
  dependency, and lifecycle status.
- Deterministic governance for health checks, component registration, health
  scoring, baseline freshness, prediction confidence, critical alerts,
  incident ownership, remediation approval, and deployment gates.
- Health check records with score, dimension, decision, status, and matched
  rule evidence.
- Baseline and prediction records for generated application workflows.
- Alert and incident lifecycle records.
- Remediation request and review workflows with runbook, approval, and
  independent reviewer evidence.
- Deployment gate decisions that block release while critical incidents remain
  unresolved unless explicitly waived.
- Generated-application view models for health dashboards and operations
  screens.
- Theme tokens and component metadata for health consoles.
- Contract-derived semantic-model and release evidence for APG publish tooling.

## Key Files

- `SPECIFICATION.md` - full functional and guardrail specification.
- `PLAN.md` - implementation plan and deferred runtime work.
- `capability_contract.py` - configuration, rule engine, UI routes, and theme.
- `service.py` - existing async health runtime plus `HlthService` control plane.
- `api.py` - generated-application helper functions and existing REST resources.
- `view_models.py` - generated-application UI model builders.
- `app.py` - APG package entrypoint and semantic model.
- `semantic_model.json` - publishable semantic-model evidence.
- `release_report.json` - focused release evidence.
- `tests/` - focused package and contract coverage.

## Direct Usage

```python
from capabilities.common.hlth.api import (
    register_component_record,
    record_health_check,
    create_baseline_record,
    request_prediction,
    request_remediation,
    decide_remediation,
    evaluate_deployment_gate,
)

component = register_component_record(
    tenant_id="tenant-a",
    component_id="orders-api",
    name="Orders API",
    component_type="service",
    environment="production",
    owner="platform",
    criticality="critical",
)

check = record_health_check(
    tenant_id="tenant-a",
    component_id="orders-api",
    dimension="availability",
    score=35,
    summary="Availability below threshold",
    runbook_id="orders-restore-availability",
    owner="platform",
    notification_route="pagerduty:orders",
)

baseline = create_baseline_record(
    tenant_id="tenant-a",
    component_id="orders-api",
    dimension="availability",
    expected_score=95,
    sample_count=200,
)

prediction = request_prediction(
    tenant_id="tenant-a",
    component_id="orders-api",
    baseline_id=baseline.baseline_id,
    predicted_score=62,
    confidence=0.82,
)

request = request_remediation(
    tenant_id="tenant-a",
    incident_id=check.incident_id,
    requester="platform",
    environment="production",
    runbook_id="orders-restore-availability",
    runbook_attached=True,
    production_approved=True,
    proposed_action="restart unhealthy workers",
    reason="availability score below threshold",
)

decision = decide_remediation(
    request_id=request.request_id,
    reviewer="sre-lead",
    decision="approved",
    notes="Runbook and blast radius reviewed.",
)

gate = evaluate_deployment_gate(
    tenant_id="tenant-a",
    deployment_id="orders-2026-05-30",
)
```

## Rule Evaluation

```python
from capabilities.common.hlth.capability_contract import evaluate_capability_rules

decision = evaluate_capability_rules({
    "tenant_context_present": True,
    "operation": "track_component_health",
    "component_registered": False,
    "component_id_present": True,
})

assert decision["decision"] == "deny"
assert "component_must_be_registered" in decision["matched_rules"]
```

## View Models

```python
from capabilities.common.hlth.api import SERVICE
from capabilities.common.hlth.view_models import dashboard_model, incident_model

dashboard = dashboard_model(SERVICE, tenant_id="tenant-a")
incidents = incident_model(SERVICE, tenant_id="tenant-a")
```

## Adapter Boundary

Production adapters should:

1. Register components before recording health checks.
2. Ask HLTH for guardrail decisions before accepting checks, predictions,
   remediation requests, and deployment gates.
3. Preserve tenant labels, component IDs, owners, and dependency metadata in
   backend storage.
4. Route critical alerts only when owner and notification route evidence exist.
5. Emit audit evidence through APG `audl` when available.
6. Treat HLTH remediation approvals as control-plane decisions, not execution.
7. Keep ML predictions, live probes, and discovery engines behind adapters that
   can be disabled or reviewed.

## Verification

Focused verification for this packet should use:

```bash
./.venv/bin/python -m py_compile \
  capabilities/common/hlth/capability_contract.py \
  capabilities/common/hlth/service.py \
  capabilities/common/hlth/api.py \
  capabilities/common/hlth/view_models.py \
  capabilities/common/hlth/app.py

./.venv/bin/pytest -q \
  capabilities/common/hlth/test_capability_contract.py \
  capabilities/common/hlth/tests/test_package_contract.py

./.venv/bin/apg capabilities publish-plan capabilities/common/hlth --json
```

Full repository tests, live probes, ML model training, production persistence,
rendered dashboard verification, and performance benchmarks are separate
runtime validation tasks.
