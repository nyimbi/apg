# Telecom Analytics — User Guide

**Capability ID**: `telecom_ana` | **Domain**: `telecom` | **Version**: `1.1.0`
**Copyright**: © 2025 Datacraft | **Author**: Nyimbi Odero

---

## Description

`telecom_ana` provides comprehensive network performance analytics, churn prediction, ARPU analysis, usage pattern analytics, revenue assurance, and 5G slice SLA tracking for telecom operators. The capability integrates ML model management to surface predictive insights, automated anomaly detection, customer segmentation, and competitive benchmarking across all network layers.

Version 1.1.0 adds nine world-class analytics methods: price elasticity modelling, spectrum efficiency per cell, concept drift detection, Markov-chain subscriber journey analysis, CDR revenue reconciliation, automated KPI root-cause analysis, predictive capacity hotspot detection, analytics DAG lineage tracking, and 5G slice SLA compliance.

---

## Installation

```bash
pip install apg-telecom-ana
```

---

## Quick Start

```python
import asyncio
from capabilities.telecom.ana.service import TelecomAnalyticsService

svc = TelecomAnalyticsService()

# Register a predictive model
svc.register_model(
    model_id="mdl-001",
    tenant_id="acme",
    model_type="classification",
    model_name="ChurnClassifierV3",
    version="3.1.0",
    validation_reference="val-report-2026-05",
    registered_by="data-science-team",
)

# Record a subscriber churn prediction
svc.record_churn_prediction(
    prediction_id="pred-001",
    tenant_id="acme",
    customer_id="cust-7890",
    risk_level="high",
    confidence_score=0.87,
    model_id="mdl-001",
    predicted_at="2026-06-01T08:00:00Z",
    features_reference="features-cust-7890-2026-05",
)

# Real-time churn risk with what-if override
result = asyncio.run(svc.churn_prediction(
    customer_id="cust-7890",
    tenant_id="acme",
    feature_overrides={"recent_complaint": True},
))
print(result["recommended_intervention"])  # → "immediate_retention_call"
```

---

## Provides

- `analytics_pipeline` — End-to-end analytics run orchestration
- `churn_prediction_workflow` — ML-driven subscriber churn scoring
- `arpu_analysis_workflow` — ARPU trend analysis and price-elasticity modelling
- `usage_pattern_workflow` — Subscriber usage profiling and segmentation
- `revenue_assurance_workflow` — Revenue leak detection, CDR reconciliation, and billing alignment
- `network_performance_analytics` — Per-layer KPI aggregation, trending, and spectrum efficiency
- `customer_segmentation_workflow` — Rule-based and ML segment definition
- `anomaly_detection_workflow` — Statistical and ML anomaly flagging
- `model_management_workflow` — Model registration, versioning, validation, and drift detection
- `5g_slice_sla_workflow` — Per-slice (eMBB/URLLC/mMTC) SLA compliance tracking
- `capacity_planning_workflow` — Predictive hotspot detection and demand forecasting
- `rca_workflow` — Automated KPI degradation root-cause analysis
- `journey_analytics_workflow` — Markov-chain subscriber lifecycle modelling
- `dag_orchestration_workflow` — Composable analytics DAG execution with lineage

---

## Requires

| Capability | Reason |
|------------|--------|
| `auth` | User authentication and permission checks |
| `audl` | Audit trail for all write operations |
| `mten` | Multi-tenancy context enforcement |
| `conf` | Runtime configuration management |
| `ntfy` | Breach and anomaly notifications |
| `nlpc` | NLP for search and text classification |
| `moni` | Operational monitoring |
| `mqeb` | Event stream via Bytewax |
| `schd` | Scheduled report and batch job triggers |
| `telecom_net` | Network performance data feed |
| `telecom_bil` | Revenue assurance reconciliation |

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/telecom-ana/dashboard` | `telecom_ana:view` | Overview |
| `/telecom-ana/analysis` | `telecom_ana:analysis` | Analysis |
| `/telecom-ana/metrics` | `telecom_ana:metrics` | Analysis |
| `/telecom-ana/churn` | `telecom_ana:churn` | Predictions |
| `/telecom-ana/revenue` | `telecom_ana:revenue` | Revenue |
| `/telecom-ana/segments` | `telecom_ana:segments` | Customers |
| `/telecom-ana/network` | `telecom_ana:network` | Network |
| `/telecom-ana/anomalies` | `telecom_ana:anomalies` | Monitoring |
| `/telecom-ana/spectrum` | `telecom_ana:network` | Network |
| `/telecom-ana/slices` | `telecom_ana:network` | 5G |
| `/telecom-ana/hotspots` | `telecom_ana:network` | Capacity |
| `/telecom-ana/rca` | `telecom_ana:analysis` | Root Cause |
| `/telecom-ana/reconcile` | `telecom_ana:revenue` | Revenue |
| `/telecom-ana/drift` | `telecom_ana:models` | Models |
| `/telecom-ana/journey` | `telecom_ana:churn` | Customers |
| `/telecom-ana/elasticity` | `telecom_ana:revenue` | Revenue |
| `/telecom-ana/dag` | `telecom_ana:admin` | Admin |

---

## Service Methods Reference

### Core Write Methods

#### `record_analysis_run(run_id, tenant_id, analysis_type, owner_id, time_granularity, start_time, end_time, evidence_reference)`
Register an analytics pipeline run. `analysis_type` must be one of `SUPPORTED_ANALYSIS_TYPES`.

#### `record_metric(metric_id, tenant_id, metric_type, metric_name, value, unit, baseline_value, aggregation_type, recorded_at)`
Ingest a KPI, counter, gauge, histogram, or derived metric. The `baseline_value` field is required and used by `kpi_root_cause_analysis()` and `model_drift_check()`.

#### `record_churn_prediction(prediction_id, tenant_id, customer_id, risk_level, confidence_score, model_id, predicted_at, features_reference)`
Store an ML churn risk prediction. Requires that `model_id` is already registered via `register_model()`. `confidence_score` must be in [0, 1].

#### `record_revenue_event(event_id, tenant_id, category, amount, currency, period, evidence_reference)`
Record a revenue assurance event. Negative `amount` values are treated as leakage candidates by `revenue_reconciliation()`.

#### `register_model(model_id, tenant_id, model_type, model_name, version, validation_reference, registered_by)`
Register a predictive model. Required before recording churn predictions or calling `model_drift_check()`.

---

### Analytics Query Methods (all async)

#### `network_traffic_analytics(period, segment, tenant_id)`
Aggregates `AnaNetworkAnalytics` records, computing per-layer throughput statistics, threshold breach rates, and dominant traffic patterns.

```python
result = await svc.network_traffic_analytics("2026-Q2", "enterprise", "acme")
# result["layer_stats"]["ran"]["breach_rate"]  → float
```

#### `subscriber_analytics(period, segment, tenant_id)`
Returns subscriber base stats: total count, churn risk distribution, high-risk count, segment membership, and active subscriber anomalies.

#### `revenue_analytics(tenant_id, period)`
Computes ARPU, total revenue, unique customer count, and per-category revenue breakdown from stored `AnaRevenueEvent` records.

#### `churn_prediction(customer_id, tenant_id, feature_overrides)`
Real-time churn risk for a single subscriber. Pass `feature_overrides` for what-if analysis:
```python
await svc.churn_prediction("cust-001", "acme", {"payment_default": True})
```

#### `anomaly_detection(metric_id, values, tenant_id, sigma_threshold)`
Detects point anomalies in a metric time series using the z-score method. Returns all points exceeding `sigma_threshold` standard deviations.

#### `churn_risk_scoring(customer_ids, tenant_id)`
Batch-scores a list of customer IDs. Returns a scored list with risk level and churn probability. Customers without existing predictions default to `low / 0.15`.

#### `forecast_demand(resource_type, horizon_days, tenant_id)`
Linear trend extrapolation over `horizon_days`. Returns daily forecast values, base value, and computed daily trend. See improvement #5 in `WORLD_CLASS_IMPROVEMENTS.md` for a foundation-model upgrade path.

#### `competitive_analytics(period, tenant_id)`
Compares own KPI means against stored competitor benchmarks across four dimensions: price_index, quality_score, coverage_pct, nps_score. Returns overall market position.

---

### World-Class Methods (all async, v1.1.0)

#### `arpu_elasticity(segment_id, price_change_pct, tenant_id)`
Estimates ARPU and total revenue impact of a proposed price change using a log-log elasticity model derived from historical revenue event variance. Returns `projected_arpu`, `revenue_delta_pct`, and a 90% confidence interval.

```python
result = await svc.arpu_elasticity("seg-premium", 10.0, "acme")
# result["revenue_delta_pct"]  → e.g. -13.2 (13.2% revenue fall for 10% price rise)
# result["ci_lower_pct"]       → -15.2
# result["ci_upper_pct"]       → -11.2
```

**Requirement**: The segment must be registered via `record_segment()` first.

---

#### `spectrum_efficiency_analytics(period, cell_ids, tenant_id)`
Computes bits-per-Hz efficiency per RAN cell by correlating throughput (`value`) against PRB utilisation (`threshold`) from stored network analytics records. Flags cells with `bps_per_hz < 2.0` as requiring radio parameter review.

```python
result = await svc.spectrum_efficiency_analytics("2026-Q2", tenant_id="acme")
# result["underperforming_cells"]  → list of dicts with cell_id and bps_per_hz
```

---

#### `model_drift_check(model_id, live_predictions, tenant_id, psi_threshold)`
Computes Population Stability Index (PSI) between live inference output distribution and a uniform baseline. PSI > `psi_threshold` (default 0.25) signals significant concept drift and triggers a `model_drift_detected` audit event.

```python
live_scores = [0.12, 0.78, 0.91, 0.05, ...]  # recent churn model outputs
result = await svc.model_drift_check("mdl-001", live_scores, "acme")
# result["drift_detected"]         → True/False
# result["recommended_action"]     → "schedule_retraining" | "immediate_retraining" | "monitor"
```

**Requirement**: `model_id` must be registered. Raises `ValueError` otherwise.

---

#### `subscriber_journey_analytics(cohort_start, cohort_end, tenant_id)`
Models the subscriber lifecycle as a Markov chain with states: `active → at_risk → churned → reacquired`. Derives transition probabilities from stored churn predictions and returns the steady-state distribution plus the highest-leverage intervention point.

```python
result = await svc.subscriber_journey_analytics("2026-01-01", "2026-06-01", "acme")
# result["transition_matrix"]  → nested dict of state transition probabilities
# result["top_intervention"]   → e.g. "loyalty_reward_trigger"
```

---

#### `revenue_reconciliation(period, billing_total, tenant_id)`
Reconciles CDR-sourced `AnaRevenueEvent` amounts for a period against the authoritative `billing_total`. Reports absolute and percentage gap. Flags negative-amount events as leakage candidates. Gap within ±0.5% is treated as reconciled.

```python
result = await svc.revenue_reconciliation("2026-05", 1_250_000.0, "acme")
# result["reconciled"]               → True/False
# result["gap_pct"]                  → float
# result["leakage_candidates"]       → list of {event_id, amount, category}
```

---

#### `kpi_root_cause_analysis(kpi_metric_id, degradation_threshold_pct, tenant_id)`
When a KPI's value has fallen below baseline by more than `degradation_threshold_pct`, scores all other tenant metrics by normalised deviation (a mutual-information proxy) and returns up to 5 ranked candidate root causes.

```python
result = await svc.kpi_root_cause_analysis("metric-ran-throughput", 15.0, "acme")
# result["degraded"]                 → True
# result["candidate_root_causes"]    → [{metric_name, deviation_pct, mi_proxy_score}, ...]
# result["top_cause"]                → "metric-ran-prb-util"
```

---

#### `predictive_capacity_hotspots(horizon_days, utilisation_alert_pct, tenant_id)`
Projects cell site PRB utilisation over `horizon_days` using linear trend extrapolation. Ranks cells projected to breach `utilisation_alert_pct` by revenue at risk (affected subscribers × mean ARPU).

```python
result = await svc.predictive_capacity_hotspots(90, 80.0, "acme")
# result["hotspot_count"]            → int
# result["ranked_hotspots"]          → [{cell_id, projected_utilisation_pct, revenue_at_risk}, ...]
# result["total_revenue_at_risk"]    → float
```

---

#### `analytics_dag_status(dag_id, tenant_id)`
Queries the audit trail for all events associated with `dag_id`. Derives completed / failed node counts and overall DAG status. Returns full lineage event list for regulatory reporting.

```python
result = await svc.analytics_dag_status("dag-weekly-churn-2026-24", "acme")
# result["overall_status"]           → "completed" | "failed" | "pending"
# result["lineage_events"]           → list of audit event dicts
```

---

#### `slice_sla_analytics(period, slice_type, tenant_id)`
Evaluates 5G network slice SLA compliance. Reads network analytics records tagged with the slice type and computes mean value, P99 value, and compliance rate against type-specific SLA thresholds:

| Slice | Threshold | Metric direction |
|-------|-----------|-----------------|
| `embb` | ≥ 100 Mbps throughput | higher is better |
| `urllc` | ≤ 1 ms P99 latency | lower is better |
| `mmtc` | ≥ 99.9% delivery rate | higher is better |

Compliance rate < 95% marks the slice as non-compliant.

```python
result = await svc.slice_sla_analytics("2026-Q2", "urllc", "acme")
# result["sla_compliant"]            → True/False
# result["p99_value"]                → float (ms for URLLC)
# result["compliance_pct"]           → float
```

---

## Interoperability

Reference this capability in `.apg` source files:

```apg
use telecom_ana;
```

Compose with billing reconciliation:

```apg
use telecom_ana;
use telecom_bil;

compose revenue_assurance {
    telecom_ana.revenue_reconciliation -> telecom_bil.post_reconciliation_adjustment;
}
```

---

## Configuration

All keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TELECOM_ANA_`.

| Key | Default | Description |
|-----|---------|-------------|
| `analysis.supported_analysis_types` | 10 types | Valid analysis type enum |
| `churn.supported_risk_levels` | low/medium/high/critical | Risk level enum |
| `revenue.supported_categories` | 8 categories | Revenue category enum |
| `models.supported_model_types` | 8 types | ML model type enum |
| `governance.cross_tenant_data_denied` | true | Block cross-tenant queries |
| `drift.psi_threshold` | 0.25 | PSI above which model drift is declared |
| `capacity.utilisation_alert_pct` | 80.0 | PRB % threshold for hotspot alerts |
| `reconciliation.gap_tolerance_pct` | 0.5 | Max gap% accepted as reconciled |

---

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | tenant_context_present=False | deny |
| analysis_type_supported | unsupported analysis_type | deny |
| churn_model_required | model_id not in registry | deny |
| confidence_score_invalid | score outside [0,1] | deny |
| unapproved_model_deployment_denied | agent deploys without human approval | deny |
| cross_tenant_data_denied | cross-tenant query by any identity | deny |
| ana_batch_requires_bytewax | batch processor not bytewax | deny |
| slice_type_constrained | slice_type not in {embb,urllc,mmtc} | ValueError |
| drift_requires_registered_model | model_id not found | ValueError |

---

## Streaming Events

All write operations emit CloudEvents to the `apg.telecom.ana.lifecycle` Bytewax stream:

```
analysis_run_recorded        metric_recorded
churn_prediction_recorded    revenue_assurance_event_recorded
segment_recorded             network_analytics_recorded
anomaly_detected             model_registered
report_generated             ana_agent_registered
model_drift_detected         revenue_reconciliation_mismatch
revenue_reconciliation_ok    kpi_rca_run
predictive_capacity_hotspots_run   slice_sla_analytics_run
arpu_elasticity_run          spectrum_efficiency_analytics_run
subscriber_journey_analytics_run
```

---

## Further Reading

- `service.py` — Business logic implementation (all methods)
- `models.py` — SQLAlchemy and Pydantic data models
- `api.py` — REST API endpoint definitions
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference and method index
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 architectural improvement proposals
- `cap_spec.md` — Formal capability specification
- `SPECIFICATION.md` — Detailed functional specification
