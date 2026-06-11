# Performance Management — User Guide

**Capability ID**: `telecom_per` | **Domain**: `telecom` | **Version**: `1.1.0`

## Description

Telecom network performance management covering KPI monitoring across all network layers,
SLA compliance tracking with breach alerting and penalty calculation, capacity utilisation
forecasting and heatmap analysis, trend analysis, anomaly detection, configurable threshold
management with adaptive suggestions, benchmark gap analysis, subscriber impact scoring,
and regulatory compliance evidence generation.

## Installation

```bash
pip install apg-telecom-per
```

## Quick Start

```python
from capabilities.telecom.per.service import TelecomPerformanceService
import asyncio

svc = TelecomPerformanceService()

# Record a KPI
svc.record_kpi(
    kpi_id="kpi-001",
    tenant_id="acme",
    kpi_category="radio",
    kpi_name="RSRP",
    value=-85.0,
    baseline_value=-90.0,
    unit="dBm",
    network_layer="cell-nairobi-001",
    recorded_at="2026-06-11T08:00:00Z",
)

# Detect anomalies
result = asyncio.run(svc.detect_kpi_anomalies("RSRP", lookback_days=30, tenant_id="acme"))
print(result["anomaly_count"], result["anomalies"])
```

## Provides

- `kpi_monitoring_workflow` — Per-layer KPI recording, status tracking, anomaly detection
- `sla_compliance_workflow` — SLA measurement, breach detection, penalty calculation
- `capacity_utilisation_workflow` — Utilisation state, congestion alerting, heatmap generation
- `trend_reporting_workflow` — Historical trend analysis and degradation detection
- `performance_reporting_workflow` — Multi-period, audience-targeted, and compliance reports
- `threshold_management_workflow` — Approval-gated configuration with adaptive suggestions
- `benchmark_analysis_workflow` — Gap analysis vs internal/regulatory/industry targets
- `per_agent_workflow` — Performance monitoring automation agents
- `subscriber_impact_workflow` — P1–P4 degradation event triage
- `intelligence_feed_workflow` — Cross-capability performance intelligence publishing

## Requires

| Capability | Reason |
|------------|--------|
| auth | Authentication |
| audl | Threshold change audit trail |
| mten | Tenant isolation |
| conf | Configuration |
| ntfy | SLA breach and threshold notifications |
| moni | Operational monitoring |
| mqeb | Event streaming |
| schd | Scheduled report delivery |
| nlpc | NL query interface for KPI search |

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/telecom-per/dashboard` | `telecom_per:view` | Overview |
| `/telecom-per/kpis` | `telecom_per:kpis` | KPIs |
| `/telecom-per/kpis/<id>` | `telecom_per:kpis` | KPIs |
| `/telecom-per/sla` | `telecom_per:sla` | SLA |
| `/telecom-per/capacity` | `telecom_per:capacity` | Capacity |
| `/telecom-per/trends` | `telecom_per:trends` | Analytics |
| `/telecom-per/thresholds` | `telecom_per:thresholds` | Configuration |
| `/telecom-per/benchmarks` | `telecom_per:benchmarks` | Analytics |
| `/telecom-per/alerts` | `telecom_per:alerts` | Operations |
| `/telecom-per/intelligence` | `telecom_per:intelligence` | Intelligence |

## Service Methods Reference

### Core Recording Methods (synchronous)

#### `record_kpi`
```python
svc.record_kpi(
    kpi_id="kpi-001",
    tenant_id="acme",
    kpi_category="radio",        # must be in SUPPORTED_KPI_CATEGORIES
    kpi_name="RSRP",
    value=-85.0,
    baseline_value=-90.0,
    unit="dBm",
    network_layer="cell-nb-001",
    recorded_at="2026-06-11T08:00:00Z",
)
```
Records a KPI measurement. `kpi_category` is normalised to lowercase. Raises `PermissionError`
if the category is unsupported or the baseline is absent.

#### `update_kpi_status`
```python
svc.update_kpi_status("kpi-001", "acme", "critical")
```
Updates the operational status. Allowed values: `nominal | warning | critical | inactive`.
Setting `critical` fires an audit event.

#### `record_sla_compliance`
```python
svc.record_sla_compliance(
    compliance_id="sla-001",
    tenant_id="acme",
    sla_type="availability",
    customer_id="cust-999",
    target_value=99.5,
    actual_value=98.2,
    period="2026-Q2",
    notification_sent=True,
)
```
Breach is auto-detected (`actual < target`). A breach without `notification_sent=True` is denied.

#### `set_threshold`
```python
svc.set_threshold(
    threshold_id="thr-001",
    tenant_id="acme",
    kpi_name="RSRP",
    network_layer="radio",
    warning_value=-95.0,
    critical_value=-110.0,
    action="alert",
    approval_reference="CHG-2026-001",
    set_by="noc-admin",
)
```
Approval reference is mandatory. Fires a `threshold_changed` audit event.

### Analytics Methods (async)

#### `detect_kpi_anomalies`
```python
result = await svc.detect_kpi_anomalies(
    kpi_name="RSRP",
    lookback_days=30,
    z_threshold=2.0,
    tenant_id="acme",
)
# result["anomalies"] — list of {kpi_id, value, z_score, direction, recorded_at}
# result["anomaly_count"] — integer
```
Uses IQR clipping before z-score computation to prevent extreme outliers from masking normal
anomalies. Typical starting threshold: 2.0σ for alerting, 3.0σ for critical escalation.

#### `correlate_degradation`
```python
result = await svc.correlate_degradation(
    kpi_ids=["kpi-001", "kpi-002", "kpi-003"],
    correlation_threshold=0.85,
    tenant_id="acme",
)
# result["high_correlation_pairs"] — [{kpi_a, kpi_b, correlation}]
# result["correlation_matrix"] — upper-triangular dict
```
Pearson correlation. Use to identify root-cause fan-out before suppressing alert storms.

#### `compute_sla_penalty`
```python
result = await svc.compute_sla_penalty(
    compliance_id="sla-001",
    sla_tier="gold",             # gold=2×, silver=1.5×, bronze=1×
    credit_rate_per_pct=100.0,
    currency="USD",
    tenant_id="acme",
)
# result["credit_amount"] — float in specified currency
# result["breach_pct"] — percentage breach magnitude
```
Outputs a `billing_event: "sla_credit"` field for telecom_bil consumption.

#### `suggest_threshold_updates`
```python
result = await svc.suggest_threshold_updates(
    tenant_id="acme",
    lookback_days=30,
    warning_percentile=0.95,
    critical_percentile=0.99,
)
# result["suggestions"] — [{threshold_id, kpi_name, current_warning, recommended_warning, ...}]
```
Only produces suggestions where >= 10 historical measurements exist. Apply changes via
`set_threshold` with a new approval reference.

#### `end_to_end_service_quality`
```python
result = await svc.end_to_end_service_quality(
    service_id="svc-mobile-data",
    period="2026-Q2",
    tenant_id="acme",
)
# result["e2e_score"] — float 0–1000
# result["quality_bucket"] — "excellent" | "good" | "fair" | "poor"
# result["layer_scores"] — {radio, core, transport}
```
Radio KPIs: RSRP, SINR. Core: latency, rtt, packet_loss. Transport: jitter, BER.
Missing layer data contributes a neutral 500-point sub-score.

#### `subscriber_impact_score`
```python
result = await svc.subscriber_impact_score(
    event_id="evt-outage-001",
    affected_cells=["cell-001", "cell-002"],
    affected_subscriber_count=5000,
    degradation_duration_minutes=30,
    tenant_id="acme",
)
# result["priority_tier"] — "P1" | "P2" | "P3" | "P4"
# result["recommended_action"] — string
# result["subscriber_impact_score"] — float (subscriber-minutes)
```
P1 (> 100k sub-min): create ITSM P1 incident. P2 (> 10k): P2 incident. P3 (> 1k): notify NOC.
P4 (≤ 1k): log and monitor.

#### `capacity_heatmap`
```python
result = await svc.capacity_heatmap(
    region="nairobi",
    granularity="daily",         # "hourly" | "daily" | "weekly"
    period="last_30_days",
    tenant_id="acme",
)
# result["cells"] — [[bucket_idx, resource_ref, utilisation_pct]]
# result["hotspot_resources"] — list of resources with peak > 85%
```
The `cells` array maps directly to a 2D heatmap: x=bucket_idx, y=resource_ref, colour=utilisation.

#### `generate_compliance_evidence`
```python
result = await svc.generate_compliance_evidence(
    regulator="CA_KENYA",
    standard="ITU_T_G826",
    period="2026-Q2",
    tenant_id="acme",
)
# result["sha256"] — tamper-evidence fingerprint
# result["sla_compliance"]["compliance_rate"] — float
# result["kpi_summary"] — {total, critical, warning}
```
Supported regulators: `BEREC`, `CA_KENYA`, `GSMA`. SHA-256 covers the full manifest payload.

#### `publish_performance_intelligence`
```python
result = await svc.publish_performance_intelligence(
    period="2026-Q2",
    tenant_id="acme",
)
# result["top_degrading_kpis"] — top-10 critical/warning KPIs
# result["sla_breach_hotspots"] — top-10 breached SLA records
# result["capacity_risk_nodes"] — top-10 congested/overloaded resources
# result["nps_detractors"] — top-10 NPS detractor responses
# result["stream_topic"] — "apg.intel.per_feed"
```
Published to `apg.intel.per_feed`. Downstream: `intel`, `telecom_ana`, `telecom_bil`.

### Alert Lifecycle (async)

```python
# Raise
alert = await svc.raise_performance_alert(
    kpi_id="kpi-001", alert_type="threshold_breach",
    severity="critical", value=45.0, threshold=30.0, tenant_id="acme",
)
# Acknowledge
await svc.acknowledge_alert(alert["id"], "noc-engineer-1", tenant_id="acme")
# Close
await svc.close_alert(alert["id"], "Root cause: RF interference resolved", tenant_id="acme")
```

### NPS Tracking (async)

```python
await svc.record_nps("cust-999", score=4, comment="slow data", tenant_id="acme")
nps_result = await svc.nps_analytics(tenant_id="acme", period="last_90_days")
# nps_result["nps"] — float, e.g. -22.5
```

## Batch Operations

```python
# Bulk import
result = await svc.bulk_import_kpis([
    {"kpi_id": "kpi-100", "name": "RSRP", "category": "radio", "unit": "dBm",
     "target_value": -90.0, "current_value": -85.0},
], tenant_id="acme")

# Export
csv_export = await svc.export_kpis(tenant_id="acme", format="csv")
```

## Dashboard Summary

```python
summary = svc.dashboard_summary("acme")
# summary keys: kpi_count, sla_compliance_count, sla_breach_count,
#               capacity_record_count, trend_count, threshold_count,
#               benchmark_count, report_count, agent_count,
#               alert_count, audit_event_count, streaming
```

## Interoperability

```apg
use telecom_per;
```

`telecom_per` integrates with:
- `telecom_bil` — SLA penalty credits via `compute_sla_penalty`
- `telecom_pro` — Capacity forecasts for resource reservation
- `telecom_qos` — Threshold configuration for QoS tuning
- `telecom_net` / `telecom_ana` — KPI raw data ingestion
- `intel` — Performance intelligence feed via `publish_performance_intelligence`

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment
variables prefixed with `TELECOM_PER_`.

| Key | Default | Description |
|-----|---------|-------------|
| kpis.collection_interval_seconds | 300 | KPI polling cadence |
| kpis.retention_days | 365 | KPI history window |
| capacity.utilisation_threshold_pct | 80 | Congestion alert level |
| capacity.forecast_horizon_days | 90 | Forecast planning window |
| sla_compliance.grace_period_minutes | 60 | Breach tolerance window |
| anomaly.z_threshold | 2.0 | Default z-score for anomaly detection |
| penalty.default_credit_rate | 100.0 | USD per breach percentage point |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| kpi_category_not_supported | unknown category | deny |
| kpi_baseline_required | no baseline value | deny |
| sla_breach_notification_required | breach without notification | deny |
| threshold_change_approval_required | no approval reference | deny |
| report_approval_required | no approval reference | deny |
| cross_tenant_data_denied | cross-tenant agent scope | deny |
| unapproved_threshold_change_denied | agent changes thresholds | deny |

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap for 15 world-class enhancements
- `cap_spec.md` — Full capability specification
- `SPECIFICATION.md` — Technical specification

## Copyright
© 2025 Datacraft — www.datacraft.co.ke | nyimbi@gmail.com
