# MONI - Monitoring and Observability

MONI is APG's tenant-scoped monitoring and observability capability. It gives
generated applications a dependency-light control plane for registering signal
sources, governing metrics/logs/traces, managing SLOs, routing alerts,
correlating incidents, approving remediation, and publishing UI/theme metadata.

The current packet can be composed without starting OpenTelemetry collectors,
metrics databases, log stores, trace stores, notification systems, or incident
management tools. Those systems are runtime adapters that must honor MONI
guardrail decisions.

## What MONI Provides

- Tenant-aware telemetry source registration.
- Deterministic governance for signal ingestion, PII logs, trace metadata,
  cardinality exceptions, alert routes, incident ownership, retention, and
  remediation approval.
- Metric, log, and trace signal records with decision evidence.
- SLO records and alert/incident lifecycle records.
- Remediation request and approval workflows with independent reviewer evidence.
- Generated-application view models for dashboards and operations screens.
- Theme tokens and component metadata for signal consoles.
- Contract-derived semantic-model and release evidence for APG publish tooling.

## Key Files

- `SPECIFICATION.md` - full functional and guardrail specification.
- `PLAN.md` - implementation plan and deferred runtime work.
- `capability_contract.py` - configuration, rule engine, UI routes, and theme.
- `service.py` - existing async monitoring runtime plus `MoniService` control plane.
- `api.py` - direct helper functions for generated APG applications.
- `view_models.py` - generated-application UI model builders.
- `app.py` - APG package entrypoint and semantic model.
- `semantic_model.json` - publishable semantic-model evidence.
- `release_report.json` - focused release evidence.
- `tests/` - focused package and contract coverage.

## Direct Usage

```python
from capabilities.common.moni.api import (
    register_source_record,
    ingest_signal_record,
    create_slo_record,
    create_alert_record,
    request_remediation,
    decide_remediation,
)

source = register_source_record(
    tenant_id="tenant-a",
    source_id="orders-api",
    service_name="orders",
    environment="production",
    owner="platform",
    notification_route="pagerduty:orders",
)

signal = ingest_signal_record(
    tenant_id="tenant-a",
    source_id="orders-api",
    signal_type="metric",
    name="orders.request.latency_ms",
    value=275,
    labels={"route": "/orders"},
    cardinality=250,
)

slo = create_slo_record(
    tenant_id="tenant-a",
    service_name="orders",
    objective="p95 latency under 300ms",
    threshold=300,
    window_minutes=60,
    owner="platform",
    notification_route="pagerduty:orders",
)

alert = create_alert_record(
    tenant_id="tenant-a",
    source_id="orders-api",
    severity="critical",
    title="Orders latency SLO burn",
    notification_route="pagerduty:orders",
    owner="platform",
)

request = request_remediation(
    tenant_id="tenant-a",
    incident_id=alert.incident_id,
    requester="platform",
    environment="production",
    runbook_id="orders-scale-out",
    runbook_approved=True,
    proposed_action="scale orders workers",
    reason="latency burn rate",
)

decision = decide_remediation(
    request_id=request.request_id,
    reviewer="sre-lead",
    decision="approved",
    notes="Runbook is approved and capacity is available.",
)
```

## Rule Evaluation

```python
from capabilities.common.moni.capability_contract import evaluate_capability_rules

decision = evaluate_capability_rules({
    "tenant_context_present": True,
    "operation": "ingest_log",
    "source_registered": True,
    "log_contains_pii": True,
    "pii_redacted": False,
})

assert decision["decision"] == "deny"
assert "pii_logs_blocked" in decision["matched_rules"]
```

## View Models

```python
from capabilities.common.moni.api import SERVICE
from capabilities.common.moni.view_models import dashboard_model, incident_model

dashboard = dashboard_model(SERVICE, tenant_id="tenant-a")
incidents = incident_model(SERVICE, tenant_id="tenant-a")
```

## Adapter Boundary

Production adapters should:

1. Register telemetry sources before accepting signals.
2. Ask MONI for guardrail decisions before ingesting signals or executing
   remediation.
3. Preserve tenant labels and source IDs in backend storage.
4. Route critical alerts only through configured notification routes.
5. Emit audit evidence through APG `audl` when available.
6. Treat MONI remediation approvals as control-plane decisions, not execution.

## Verification

Focused verification for this packet should use:

```bash
./.venv/bin/python -m py_compile \
  capabilities/common/moni/capability_contract.py \
  capabilities/common/moni/service.py \
  capabilities/common/moni/api.py \
  capabilities/common/moni/view_models.py \
  capabilities/common/moni/app.py

./.venv/bin/pytest -q \
  capabilities/common/moni/tests/test_capability_contract.py \
  capabilities/common/moni/tests/test_package_contract.py

./.venv/bin/apg capabilities publish-plan capabilities/common/moni --json
```

Full repository tests, live telemetry adapters, production persistence, and
rendered dashboard verification are separate runtime validation tasks.
