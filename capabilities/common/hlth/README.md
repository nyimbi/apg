# HLTH - Health Checks and Diagnostics

HLTH is APG's tenant-scoped health checks and diagnostics capability. It gives
generated applications a dependency-light control plane for registering
components, recording health checks, maintaining baselines, opening alerts and
incidents, reviewing remediation, gating deployments, composing health AI
agents, validating Bytewax lifecycle batches, and publishing UI/theme metadata.
It also preserves durable policy and review evidence for generated health
review consoles.

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
- First-class health-agent registration for Codex, Claude Code, opencode, Pi,
  and future runtime adapters, including durable `pending_review` records for
  otherwise valid privileged agents awaiting human approval.
- Bytewax-first lifecycle batch validation for component, check, baseline,
  prediction, incident, and health-agent mutations, including persisted denial
  evidence before `PermissionError` on non-Bytewax batches.
- Pending-review queues and policy evidence fields for checks, predictions,
  alerts, incidents, remediation requests, deployment gates, health agents, and
  lifecycle batches.
- Generated-application view models for health dashboards and operations screens.
- Theme tokens and component metadata for health consoles.
- Contract-derived semantic-model and release evidence for APG publish tooling.
- Multi-dimensional health analysis across performance, security, availability,
  compliance, business process, and user experience dimensions.
- ML-backed anomaly detection, failure probability prediction, and ensemble
  health forecasting with configurable prediction windows.
- Enterprise multi-tenant isolation with quota enforcement, SLA compliance
  checking, and compliance report generation (SOC 2, ISO 27001).
- Advanced alert intelligence: false-positive filtering, frequency pattern
  analysis, contextual priority scoring, and autonomous remediation with
  blast-radius gating.

## World-Class Enhancements (v2.0)

Fifteen targeted improvements elevate the capability to production-grade quality.

1. **SLA Burn-Rate Tracking** — Multi-window (1h/6h/24h/72h) error-budget
   consumption tracking using the SRE workbook burn-rate algorithm. Fast-burn
   alerts fire when the 1h window exceeds 14x the allowed error rate.

2. **Dependency Health Propagation with Circuit Breaker State** — When a
   dependency enters UNHEALTHY, a weighted fan-out propagates a synthetic
   degraded score to all dependents. Circuit-breaker state (CLOSED/OPEN/
   HALF_OPEN) is carried so dependents distinguish "slow" from "tripped open."

3. **Percentile-Based Latency Health Scoring** — p50/p95/p99 percentile
   tracking per component using rolling histograms (t-digest / HDR histogram).
   Scoring targets percentile SLOs rather than means, catching p99 spikes that
   mean-based scoring misses.

4. **Composite Health Check Groups** — Checks can be grouped into logical
   units (e.g., "payment stack") with aggregate pass/fail logic:
   `ALL_MUST_PASS`, `MAJORITY_MUST_PASS`, or `AT_LEAST_ONE_MUST_PASS`.
   Enables deployment gates to block on logical service health.

5. **Adaptive Threshold Tuning via Bayesian Updating** — Static thresholds are
   replaced with Bayesian-updated posteriors. Each observation narrows the
   posterior predictive distribution toward the true baseline, preventing both
   early-deployment false alerts and long-running threshold drift.

6. **Correlated Root Cause Surfacing via Granger Causality** — When multiple
   components degrade simultaneously, Granger causality tests on the health
   time-series rank root-cause candidates with confidence scores and temporal
   lag estimates.

7. **On-Call Schedule Integration for Escalation Routing** — Reads PagerDuty /
   OpsGenie / JSON rotation files and routes escalations to the current on-call
   engineer via their preferred channel, with team-lead fallback.

8. **Canary and Blue/Green Deployment Health Differentiation** — Health checks
   are tagged with `deployment_slot` (canary/blue/green/stable). Real-time
   health delta comparison between canary and stable drives automatic promotion
   or rollback decisions.

9. **Cost-Aware Remediation Prioritisation** — Before triggering auto-
   remediation, estimates cost of inaction (revenue/min at current degradation)
   versus risk-adjusted cost of the action (blast radius × historical failure
   rate). Only positive-expected-value remediations execute.

10. **Health Evidence Ledger with Merkle Chaining** — Each `HlthCheckRecord`
    is hashed and chained to the previous hash. The chain root is stored in the
    tenant audit log, satisfying SOC 2 Type II and ISO 27001 tamper-evidence
    requirements.

11. **Synthetic Transaction Monitoring Integration** — Configurable HTTP/gRPC
    synthetic probes feed results back as `HealthMetric` objects with
    `source=synthetic`. Scoring distinguishes synthetic from real traffic so
    100% synthetic availability with 0% real-traffic availability scores
    correctly as unhealthy.

12. **MTTR and MTBF Calculation per Component** — Rolling mean-time-to-repair
    and mean-time-between-failures are maintained per component from incident
    open/close timestamps. Shortening MTBF triggers a predictive alert before
    the next failure.

13. **Kubernetes / Nomad Native Health Source Adapter** — Consumes pod
    ready/not-ready events, liveness probe failures, and OOMKill events from
    the cluster watch stream. Translates these into typed `HealthMetric` events
    without requiring application-side instrumentation.

14. **Health Score Caching with Cache Stampede Prevention** — Replaces the
    dict-based cache with a probabilistic early-expiry (PER) cache. Entries are
    probabilistically refreshed before expiry, eliminating thundering-herd
    latency spikes without sacrificing freshness.

15. **Multi-Region Health Aggregation with Split-Brain Detection** — Collects
    health scores from agents in multiple regions and aggregates with a
    configurable quorum policy (ANY/MAJORITY/ALL). Emits a split-brain alert
    when regions disagree rather than masking the disagreement behind an average.

## Key Files

- `SPECIFICATION.md` — full functional and guardrail specification.
- `PLAN.md` — implementation plan and deferred runtime work.
- `WORLD_CLASS_IMPROVEMENTS.md` — detailed design rationale for the 15 enhancements.
- `capability_contract.py` — configuration, rule engine, UI routes, and theme.
- `service.py` — async health runtime (`HealthService`) plus `HlthService` control plane.
- `api.py` — generated-application helper functions and existing REST resources.
- `view_models.py` — generated-application UI model builders.
- `app.py` — APG package entrypoint and semantic model.
- `semantic_model.json` — publishable semantic-model evidence.
- `release_report.json` — focused release evidence.
- `tests/` — focused package and contract coverage.

## API Reference

### Control Plane (`HlthService`)

| Method | Purpose |
|--------|---------|
| `register_component(...)` | Register a tenant-scoped component before any checks. |
| `record_health_check(...)` | Record governed health state; auto-creates critical alert evidence. |
| `create_baseline(...)` | Create health baseline evidence for prediction gating. |
| `request_prediction(...)` | Record a prediction decision against baseline evidence. |
| `create_alert(...)` | Open a governed alert record with severity and owner. |
| `create_incident(...)` | Open an incident linked to an alert. |
| `request_remediation(...)` | Submit remediation with runbook and blast-radius evidence. |
| `decide_remediation(...)` | Approve or reject a remediation request with reviewer evidence. |
| `evaluate_deployment_gate(...)` | Gate a deployment; blocks while critical incidents are open. |
| `register_health_agent(...)` | Register an AI agent for governed health operations. |
| `validate_health_lifecycle_batch(...)` | Validate Bytewax lifecycle batch; deny non-Bytewax streams. |
| `list_pending_reviews(...)` | Return all records in `pending_review` state for a tenant. |
| `dashboard_summary(...)` | Return counts and status summary for a tenant. |
| `list_records(...)` | Return all records for a tenant, optionally filtered by type. |

### Runtime (`HealthService`)

| Method | Purpose |
|--------|---------|
| `process_health_metric(metric, tenant_id)` | Ingest a `HealthMetric`; score, baseline, anomaly-detect, alert. |
| `get_component_health_status(component_id, tenant_id)` | Current `HealthStatus` for one component. |
| `calculate_component_health_score(component_id, tenant_id)` | Weighted composite score (0–100). |
| `perform_comprehensive_health_assessment(tenant_id)` | Full `HealthAssessmentResult` across all components. |
| `process_health_alert(alert)` | Filter, correlate, prioritise, and escalate an alert. |
| `generate_health_report(tenant_id, report_type)` | `HealthReport` in executive/operational/technical format. |
| `predict_component_health(component_id, tenant_id, window_hours)` | Trend-based health forecast. |
| `analyze_multi_dimensional_health(tenant_id, component_id)` | Performance, security, availability, compliance, UX, BizOps dimensions. |
| `predict_component_health_advanced(component_id, tenant_id, window_hours)` | ML ensemble prediction when engine is available. |
| `detect_health_anomalies(component_id, tenant_id, time_window_hours)` | ML anomaly detection. |
| `predict_failure_probability(component_id, tenant_id, window_hours)` | Failure probability estimate. |
| `generate_advanced_health_insights(tenant_id, time_window_hours)` | Analytics-engine insights. |
| `analyze_optimization_opportunities(tenant_id, component_id)` | Resource and performance optimization hints. |
| `check_sla_compliance(tenant_id, metric_type, current_value)` | Point-in-time SLA compliance check. |
| `generate_compliance_report(tenant_id, framework, days)` | Compliance report for SOC2/ISO27001/etc. |
| `process_health_metric_with_enterprise_features(metric, requesting_tenant_id)` | Metric ingestion with quota, boundary, and SLA checks. |
| `create_tenant_health_dashboard(tenant_id, dashboard_type, requesting_user_id)` | Enterprise dashboard with quota and isolation status. |

## Quick Start

```python
from capabilities.common.hlth.api import (
    register_component_record,
    record_health_check,
    create_baseline_record,
    request_prediction,
    request_remediation,
    decide_remediation,
    evaluate_deployment_gate,
    register_health_agent,
    validate_health_lifecycle_batch,
    list_pending_reviews,
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

agent = register_health_agent(
    tenant_id="tenant-a",
    agent_id="gate-agent",
    name="Deployment Gate Reviewer",
    runtime="claude code",
    role="deployment gate reviewer",
    scope="orders production health gates",
    owner="sre-lead",
    purpose="review critical deployment gates",
    human_approval_required=True,
)

batch = validate_health_lifecycle_batch(
    tenant_id="tenant-a",
    event_stream="bytewax",
    mutation_count=6,
)

pending = list_pending_reviews("tenant-a")
```

## New Methods

### Multi-Dimensional Health Analysis

```python
from capabilities.common.hlth.service import HealthService, HealthServiceConfig

service = HealthService(HealthServiceConfig())
await service.initialize()

# Returns scores across performance, security, availability,
# compliance, business-process, and user-experience dimensions,
# plus cross-dimensional correlation insights.
result = await service.analyze_multi_dimensional_health(
    tenant_id="tenant-a",
    component_id="orders-api",
    time_window_hours=24,
)
# result["dimension_scores"]["performance"] -> float
# result["cross_dimensional_insights"] -> list[str]
# result["overall_health_score"] -> float
```

### ML-Backed Failure Probability

```python
# Requires ML prediction engine to be initialised (advanced config).
result = await service.predict_failure_probability(
    component_id="orders-api",
    tenant_id="tenant-a",
    prediction_window_hours=48,
)
# result["failure_probability"] -> float (0-1)
# result["confidence"] -> float
# result["risk_factors"] -> list[str]
```

### Enterprise Metric Processing with Tenant Isolation

```python
from capabilities.common.hlth.models import HealthMetric

metric = HealthMetric(
    component_id="orders-api",
    tenant_id="tenant-a",
    metric_type="availability",
    value=0.82,
)

# Enforces quota, tenant boundary, and SLA checks before
# delegating to the standard processing pipeline.
result = await service.process_health_metric_with_enterprise_features(
    health_metric=metric,
    requesting_tenant_id="tenant-a",
)
# result["sla_compliance"]["compliant"] -> bool
# result["tenant_isolation_applied"] -> True
```

### SLA Compliance Check

```python
result = await service.check_sla_compliance(
    tenant_id="tenant-a",
    metric_type="availability",
    current_value=0.982,
)
# result["compliant"] -> bool
# result["sla_target"] -> float
# result["breach_margin"] -> float
```

### Compliance Report Generation

```python
report = await service.generate_compliance_report(
    tenant_id="tenant-a",
    framework="soc2",        # or "iso27001", "gdpr", "hipaa"
    time_period_days=30,
)
# report["controls_passed"] -> int
# report["controls_failed"] -> int
# report["evidence_records"] -> list[dict]
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

Bytewax is mandatory for lifecycle batches:

```python
decision = evaluate_capability_rules({
    "tenant_context_present": True,
    "operation": "validate_health_lifecycle_batch",
    "event_stream": "legacy_broker",
})

assert decision["decision"] == "deny"
assert "bytewax_health_stream_required" in decision["matched_rules"]
```

## View Models

```python
from capabilities.common.hlth.api import SERVICE
from capabilities.common.hlth.view_models import (
    dashboard_model,
    incident_model,
    health_agent_roster_model,
    lifecycle_batch_model,
)

dashboard = dashboard_model(SERVICE, tenant_id="tenant-a")
incidents = incident_model(SERVICE, tenant_id="tenant-a")
agents = health_agent_roster_model(SERVICE, tenant_id="tenant-a")
lifecycle = lifecycle_batch_model(SERVICE, tenant_id="tenant-a")
```

## Adapter Boundary

Production adapters must:

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
8. Treat HLTH health-agent registrations as governance records, not embedded
   runtime clients.
9. Preserve `policy_decision`, `matched_rules`, `review_reasons`, and
   `review_evidence` when moving records into external stores.
10. Route lifecycle mutation batches through Bytewax and preserve the
    `hlth.lifecycle` event-time contract.
11. Implement the 15 v2.0 enhancements (burn-rate, circuit-breaker propagation,
    percentile scoring, Bayesian thresholds, etc.) as runtime adapters behind
    the same guardrail interface — HLTH governs decisions, adapters execute them.

## Verification

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
