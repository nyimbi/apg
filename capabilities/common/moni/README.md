# MONI - Monitoring and Observability

MONI is APG's tenant-scoped monitoring and observability capability. It gives
generated applications a dependency-light control plane for registering signal
sources, governing metrics/logs/traces, managing SLOs, routing alerts,
correlating incidents, approving remediation, composing monitoring AI agents,
validating Bytewax lifecycle batches, and publishing UI/theme metadata.

The capability operates without requiring OpenTelemetry collectors, metrics
databases, log stores, trace stores, notification systems, or incident management
tools at composition time. Those systems are runtime adapters that must honor
MONI guardrail decisions.

## What MONI Provides

- Tenant-aware telemetry source registration with per-source signal type controls.
- Deterministic guardrail evaluation for signal ingestion, PII logs, trace metadata,
  cardinality exceptions, alert routes, incident ownership, retention, and
  remediation approval.
- Metric, log, and trace signal records with full decision evidence.
- SLO records with threshold and window governance.
- Alert and incident lifecycle records with auto-incident creation for critical alerts.
- Remediation request and approval workflows with independent reviewer evidence.
- First-class monitoring-agent registration for Codex, Claude Code, opencode,
  Pi, and future runtime adapters.
- Durable review evidence across all review-required signals, remediation requests,
  privileged monitoring agents, denied lifecycle batches, alerts, incidents,
  and audit events.
- Pending-review queue composition for generated observability consoles.
- Bytewax-first lifecycle batch validation for metrics, SLOs, alerts,
  incidents, and monitoring-agent mutations.
- Generated-application view models for dashboards and operations screens.
- Theme tokens and component metadata for signal consoles.
- Contract-derived semantic-model and release evidence for APG publish tooling.
- Anomaly detection with configurable baseline learning and z-score thresholding.
- Multi-dimensional performance analytics with pattern extraction and recommendations.
- Predictive resource usage forecasting.
- Composite health scoring per tenant with alert-weighted severity model.

## Key Files

- `SPECIFICATION.md` - full functional and guardrail specification.
- `PLAN.md` - implementation plan and deferred runtime work.
- `WORLD_CLASS_IMPROVEMENTS.md` - 15 high-impact improvements targeting v2.0.
- `capability_contract.py` - configuration, rule engine, UI routes, and theme.
- `service.py` - async monitoring runtime (`MonitoringService`) plus `MoniService` control plane.
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
    register_monitoring_agent,
    validate_monitoring_lifecycle_batch,
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

agent = register_monitoring_agent(
    tenant_id="tenant-a",
    agent_id="slo-agent",
    name="SLO Reviewer",
    runtime="claude code",
    role="slo reviewer",
    scope="orders service SLOs",
    owner="sre-lead",
    purpose="review SLO burn and alert route quality",
    human_approval_required=True,
)

batch = validate_monitoring_lifecycle_batch(
    tenant_id="tenant-a",
    event_stream="bytewax",
    mutation_count=8,
)
```

Privileged monitoring agents that are otherwise valid but missing human
approval are stored as `pending_review` with
`policy_decision="require_review"`. Denied non-Bytewax lifecycle batches are
stored as `denied` before `PermissionError` is raised. Generated applications
can use `list_pending_reviews()` or `list_observability()` to compose one
operator review queue.

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

Bytewax is mandatory for lifecycle batches:

```python
decision = evaluate_capability_rules({
    "tenant_context_present": True,
    "operation": "validate_monitoring_lifecycle_batch",
    "event_stream": "legacy_broker",
})

assert decision["decision"] == "deny"
assert "bytewax_monitoring_stream_required" in decision["matched_rules"]
```

## View Models

```python
from capabilities.common.moni.api import SERVICE
from capabilities.common.moni.view_models import (
    dashboard_model,
    incident_model,
    monitoring_agent_roster_model,
    lifecycle_batch_model,
)

dashboard = dashboard_model(SERVICE, tenant_id="tenant-a")
incidents = incident_model(SERVICE, tenant_id="tenant-a")
agents = monitoring_agent_roster_model(SERVICE, tenant_id="tenant-a")
lifecycle = lifecycle_batch_model(SERVICE, tenant_id="tenant-a")
```

## New Methods

### `MonitoringService.detect_anomalies` — ML-based anomaly detection

Queries historical metrics and returns anomalous data points for a given metric
over a configurable lookback window. Baselines accumulate on every ingested
`MonitoringMetric` using `_update_metric_baseline`; anomaly detection uses z-score
comparison against the rolling mean/std.

```python
service = await create_monitoring_service()

anomalies = await service.detect_anomalies(
    metric_name="orders.request.latency_ms",
    tenant_id="tenant-a",
    lookback_hours=24,
)
# Returns: [{"timestamp": ..., "value": ..., "z_score": ..., "severity": ...}, ...]
```

### `MonitoringService.predict_resource_usage` — forecasting

Generates `forecast_hours`-ahead resource usage predictions from historical
data using the service's internal ML models.

```python
prediction = await service.predict_resource_usage(
    resource_type="cpu",
    tenant_id="tenant-a",
    forecast_hours=6,
)
# prediction["predicted_values"], prediction["confidence_interval"], prediction["trend"]
```

### `MonitoringService.analyze_performance` — performance analytics

Returns performance scores, extracted hourly/daily seasonal patterns, and
optimization recommendations for a named service over a configurable window.

```python
report = await service.analyze_performance(
    service_name="orders",
    tenant_id="tenant-a",
    analysis_hours=24,
)
# report["performance_scores"], report["patterns"], report["recommendations"]
```

### `MoniService.dashboard_summary` — operator overview

Returns a flat dict of tenant-scoped counts suitable for rendering an
observability overview screen. Zero runtime dependencies.

```python
from capabilities.common.moni.service import MoniService

svc = MoniService(tenant_id="tenant-a")
# ... register sources, ingest signals, create alerts ...
summary = svc.dashboard_summary("tenant-a")
# {
#   "source_count": 2,
#   "signal_count": 14,
#   "open_alert_count": 1,
#   "pending_remediation_count": 1,
#   "pending_review_count": 3,
#   "audit_event_count": 22,
#   ...
# }
```

### `MoniService.list_pending_reviews` — operator review queue

Returns all records awaiting human or operator review across signals, alerts,
incidents, remediation requests, monitoring agents, and lifecycle batches.
Compose directly into a generated console.

```python
pending = svc.list_pending_reviews("tenant-a")
for item in pending:
    print(item["status"], item.get("policy_decision"), item.get("review_reasons"))
```

## API Reference

| Function | Returns | Notes |
|---|---|---|
| `register_source` | `SignalSourceRecord` | Must precede signal ingestion for a source_id |
| `ingest_signal` | `SignalRecord` | Governs metric/log/trace; denies PII without redaction |
| `create_slo` | `SloRecord` | Requires positive threshold and window_minutes |
| `create_alert` | `AlertRecord` | Auto-opens `IncidentRecord` for critical severity |
| `create_incident` | `IncidentRecord` | Correlation record; attach alert_ids on creation |
| `request_remediation` | `RemediationRequestRecord` | Requires existing incident in same tenant |
| `decide_remediation` | `RemediationRequestRecord` | Reviewer must differ from requester |
| `register_monitoring_agent` | `MonitoringAgentRecord` | Privileged roles require human_approval_required=True |
| `validate_monitoring_lifecycle_batch` | `MonitoringLifecycleBatchRecord` | Raises PermissionError for non-Bytewax streams |
| `list_records` | `list[dict]` | Filtered by tenant_id; accepts optional record_type |
| `dashboard_summary` | `dict` | Flat count summary for generated dashboards |
| `list_pending_reviews` | `list[dict]` | Cross-collection review queue for operator consoles |

## World-Class Enhancements (v2.0)

The following 15 improvements are planned for v2.0, documented in full in
`WORLD_CLASS_IMPROVEMENTS.md`. Each item maps to a specific gap in the current
implementation.

1. **Distributed Tracing Context Propagation** — W3C `traceparent`/`tracestate`
   propagation across tenant boundaries; DAG of spans with flame-chart latency
   breakdowns. Closes the gap between MONI as a governance store and a
   first-class tracing backend.

2. **Adaptive Anomaly Detection with Concept Drift Handling** — Replace the
   rolling-deque mean/std baseline with EWMA statistics and CUSUM-based drift
   detection. Auto-resets baselines after confirmed regime changes to eliminate
   chronic alert fatigue post-deployment.

3. **SLO Burn Rate Alerting with Error Budget Tracking** — Implement Google
   SRE multi-window burn rate model (`burn_rate_1h`, `burn_rate_6h`,
   `error_budget_remaining_percent`) on `SloRecord`. Emit
   `slo.burn_rate_critical` when fast and slow windows both breach.

4. **On-Call Schedule Integration and Escalation Routing** — Tenant-scoped
   rotation schedules (primary, secondary, manager) with override windows.
   Compute current on-call at alert creation; auto-escalate after configurable
   acknowledgement windows with full audit trail.

5. **Cardinality Budget Enforcement per Tenant** — Per-tenant
   `max_series_per_tenant` budget with active series counting per metric name.
   Reject ingestion that would breach the budget; surface utilization in
   `dashboard_summary`.

6. **Metric Rollup and Downsampling Pipeline** — Rollup tiers
   (1m → 5m → 1h → 1d) with configurable aggregation functions including
   p50/p95/p99. `query_metrics` selects the appropriate resolution tier by
   requested time range. Extends effective retention ~100x.

7. **Multi-Dimensional Metric Correlation Engine** — Pearson and Spearman
   cross-correlation across configurable lag windows, segmented by tenant
   service topology. Surfaces top correlated metrics at incident creation to
   accelerate root-cause analysis.

8. **Structured Runbook Execution with Step Tracking** — Ordered runbook steps
   with preconditions, postcondition checks, rollback steps, and per-step
   `RunbookStepRecord` with stdout/stderr capture. Approval gates at step
   boundaries for production environments.

9. **Real-Time WebSocket Push for Dashboard Updates** — Per-connected-session
   asyncio `Queue` publishing delta updates when metrics or alert states change.
   Eliminates polling overhead; delivers sub-second dashboard freshness during
   active incidents.

10. **Composite Health Scoring with Weighted Signals** — Four golden signals
    model (latency, traffic, errors, saturation) with configurable weights per
    tenant service tier. Replaces the current alert-count subtraction heuristic
    in `_calculate_tenant_health_score`.

11. **Metric Pipeline Backpressure and Rejection Feedback** — Explicit
    high-water mark with `MONI_BACKPRESSURE` status on rejection. Rejection
    counters surfaced as first-class metrics. Eliminates silent data loss during
    ingestion spikes.

12. **Per-Tenant Retention Policy Enforcement** — Activate the existing
    `DataRetentionPolicy` model in the cleanup control plane. Per-tenant TTL
    overrides in `_cleanup_old_metrics`; emit `data.purged` audit events;
    expose retention budget in `dashboard_summary`.

13. **Chaos Injection Hooks for Resilience Testing** — `chaos_enabled` config
    flag (default `False`) that enables synthetic latency injection, simulated
    alert delivery failures, and artificial anomaly signals. Essential for
    continuous resilience validation without waiting for production incidents.

14. **OpenTelemetry Semantic Convention Validation** — Configurable convention
    registry validates signal names against OTel canonical forms. Emits
    `signal.convention_violation` audit events; provides a migration helper
    suggesting canonical alternatives.

15. **AI-Assisted Alert Triage with Explanation Generation** — Structured
    `triage_context` block appended to `AlertRecord` at creation time,
    incorporating correlated metrics, recent deployment events, and historical
    resolution patterns. When an Ollama model is available, generates ranked
    natural-language root-cause explanations (honors APG local-AI strategy).

## Adapter Boundary

Production adapters should:

1. Register telemetry sources before accepting signals.
2. Ask MONI for guardrail decisions before ingesting signals or executing
   remediation.
3. Preserve tenant labels and source IDs in backend storage.
4. Route critical alerts only through configured notification routes.
5. Emit audit evidence through APG `audl` when available.
6. Treat MONI remediation approvals as control-plane decisions, not execution.
7. Treat MONI monitoring-agent registrations as governance records, not as
   embedded runtime clients.
8. Route lifecycle mutation batches through Bytewax and preserve the
   `moni.lifecycle` event-time contract.
9. Preserve MONI policy evidence fields when moving records between durable
   storage, generated UIs, and runtime telemetry adapters.

## Verification

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
