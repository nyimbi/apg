# Store Intelligence — User Guide

**Capability ID**: `retail_sin` | **Domain**: `retail` | **Version**: `1.1.0`

---

## Overview

Store Intelligence (`retail_sin`) provides anonymised in-store analytics for retail operations. It covers foot traffic counting, zone dwell-time, heatmap generation, planogram compliance auditing, shelf-availability alerting, loss-prevention incident management, shopper journey attribution, KPI benchmarking, and staff demand forecasting. All personal data is anonymised at ingest; raw video storage and biometric identification are hard-denied by the rule engine.

---

## Installation

```bash
pip install apg-retail-sin
```

---

## Quick Start

```python
import asyncio
from apg_retail_sin import SinService
from apg_retail_sin.models import SinStoreCreate

async def main():
    svc = SinService(tenant_id="acme", actor_id="admin")

    store = await svc.create_store(SinStoreCreate(
        tenant_id="acme",
        store_code="NBO-001",
        name="Nairobi Flagship",
        store_format="superstore",
        address={"city": "Nairobi", "country": "KE"},
        latitude=-1.286,
        longitude=36.817,
        sqm_total=4500.0,
        sqm_selling=3200.0,
        created_by="admin",
    ))
    print(store.id)

asyncio.run(main())
```

---

## Core Concepts

### Stores and Zones

A **Store** is the top-level physical unit. Each store has `sqm_total`, `sqm_selling`, and a `store_format` (e.g. `superstore`, `convenience`, `express`). Zones subdivide a store into measurable areas (entrance, aisle, checkout, etc.) with polygon coordinates for spatial analysis.

### Sensors

Sensors are associated with zones. Supported `sensor_type` values include `infrared_beam`, `lidar`, `wifi_probe`, and `camera_count`. Sensors must heartbeat regularly; missing heartbeats are surfaced by `sensor_network_health()`.

### Traffic Counts

Traffic counts are anonymised entry/exit records at zone level. All counts are PII-stripped at ingest — only aggregate integers are stored, never individual identifiers.

---

## Feature Reference

### 1. Foot Traffic Analytics

Record and query anonymised entry/exit counts per zone.

```python
# Ingest a count
await svc.foot_traffic_record(
    store_id=store.id,
    timestamp="2026-06-01T09:00:00",
    count=142,
    zone="entrance",
)

# Query summary
summary = await svc.get_traffic_summary(
    tenant_id="acme",
    store_id=store.id,
    period_start=datetime(2026, 6, 1),
    period_end=datetime(2026, 6, 30),
)
# Returns: total_entries, total_exits, peak_occupancy, avg_dwell_seconds
```

### 2. Occupancy Capacity Compliance

Checks recorded occupancy peaks against fire-code limits (85% of `max_capacity`, warning at 80%).

```python
result = await svc.check_occupancy_compliance(store_id=store.id, period="2026-06")
# Returns: breach_count, warning_count, compliant, breach_records
```

If `max_capacity` is not set on the store, the system defaults to `sqm_total * 2` persons.

### 3. Sensor Network Health

Compute an overall health score for a store's sensor network.

```python
health = await svc.sensor_network_health(tenant_id="acme", store_id=store.id)
# health_score: 0-100
# status: "healthy" (>=80) | "degraded" (>=50) | "critical" (<50)
# uncovered_zone_ids: zones with no online sensor
```

Scores weight online ratio (50%), zone coverage (30%), and heartbeat recency (20%).

### 4. Planogram Compliance

Record manual or AI-assisted audits and track compliance scores over time.

```python
from apg_retail_sin.models import SinPlanogramAuditCreate

audit = await svc.record_planogram_audit(SinPlanogramAuditCreate(
    tenant_id="acme",
    store_id=store.id,
    zone_id=zone.id,
    planogram_id="PLG-BEVERAGES-001",
    audited_by="agent_vision_01",
    audit_method="image_ai",
    compliance_status="minor_deviation",
    deviation_details=[{"position": "A3", "expected": "cola_2L", "found": "cola_1.5L"}],
    created_by="system",
))

rate = await svc.get_store_compliance_rate(tenant_id="acme", store_id=store.id)
# Returns float 0.0-100.0
```

| `compliance_status` | Score |
|---|---|
| `compliant` | 100.0 |
| `minor_deviation` | 80.0 |
| `major_deviation` | 50.0 |
| `out_of_stock` | 0.0 |

### 5. Shelf Availability & Replenishment

```python
# Scan and raise OOS alerts
result = await svc.shelf_availability_scan(
    store_id=store.id,
    scan_date="2026-06-11",
    sku_gaps=["SKU-0042", "SKU-0099"],
)

# Trigger replenishment
await svc.trigger_replenishment(tenant_id="acme", alert_id=alert.id)

# Resolve when stock is restored
await svc.resolve_shelf_alert(
    tenant_id="acme", alert_id=alert.id,
    notes="Shelf replenished by team B", by="staff_007",
)
```

### 6. Loss Prevention Incident Management

A full incident lifecycle for shrinkage tracking.

```python
# Open incident
incident = await svc.report_lp_incident(
    store_id=store.id,
    zone_id=zone.id,
    sku="SKU-0042",
    incident_type="shoplifting",  # | staff_theft | admin_error | damage
    estimated_value_loss=450.0,
    sensor_ids=["sensor_cam_01"],
    notes="CCTV review scheduled",
)

# Escalate if required
await svc.escalate_lp_incident(
    incident_id=incident["id"],
    reason="Value exceeds KES 10,000 threshold",
)

# Close with confirmed outcome
await svc.close_lp_incident(
    incident_id=incident["id"],
    resolution="Stock recovered, suspect identified",
    confirmed_loss=0.0,
)

# Query
open_incidents = await svc.list_lp_incidents(store_id=store.id, investigation_status="open")
```

### 7. Conversion Funnel & Shopper Journey

Track shopper progression through stages and reconstruct journey paths.

```python
from apg_retail_sin.models import SinConversionEventCreate

await svc.record_conversion_event(SinConversionEventCreate(
    tenant_id="acme",
    store_id=store.id,
    session_id="sess_abc123",
    conversion_metric="browse_to_pickup",
    from_stage="browse",
    to_stage="pickup",
    converted=True,
    dwell_seconds=87.0,
    created_by="pos_agent",
))

# Reconstruct full journey
journey = await svc.stitch_shopper_journey(store_id=store.id, session_id="sess_abc123")
# Returns: ordered path, total_dwell_seconds, final_stage, converted
```

### 8. Heatmaps

```python
from apg_retail_sin.models import SinHeatmapCreate
from datetime import datetime

hm = await svc.create_heatmap(SinHeatmapCreate(
    tenant_id="acme",
    store_id=store.id,
    resolution="2m",
    period_start=datetime(2026, 6, 1),
    period_end=datetime(2026, 6, 7),
    grid_data=[[0.1, 0.4, 0.9], [0.3, 0.8, 0.5]],
    pii_masked=True,  # mandatory
    created_by="analytics_agent",
))

# Compare two heatmaps (before/after fixture change)
diff = await svc.compute_heatmap_diff(
    heatmap_id_before=hm_before.id,
    heatmap_id_after=hm_after.id,
)
# Returns: delta_grid, max_gain, max_loss, changed_cell_count
```

### 9. KPI Snapshots, Trends, and Benchmarking

```python
# Record a KPI snapshot
from apg_retail_sin.models import SinKpiSnapshotCreate

await svc.record_kpi_snapshot(SinKpiSnapshotCreate(
    tenant_id="acme",
    store_id=store.id,
    kpi_category="sales",
    period_type="weekly",
    period_start=datetime(2026, 6, 1),
    period_end=datetime(2026, 6, 7),
    kpi_values={"total_sales": 1_250_000.0, "transactions": 4_200},
    benchmark_type="peer_group",
    benchmark_values={"total_sales": 1_100_000.0},
    created_by="finance_agent",
))

# Detect trend
trend = await svc.detect_kpi_trends(store_id=store.id, kpi_metric="total_sales", n_periods=8)
# Returns: slope_per_period, r_squared, trend_direction, weeks_to_breach_zero

# Peer-group benchmark
bench = await svc.benchmark_peer_group(
    store_id=store.id,
    period="2026-06",
    kpi_metric="total_sales",
    min_peer_stores=5,
)
# Returns: percentile_rank, gap_to_median, gap_to_q3, ranking
```

### 10. Staff Demand Forecasting

```python
forecast = await svc.forecast_staffing_demand(
    store_id=store.id,
    forecast_weeks=2,
    traffic_to_staff_ratio=50.0,  # visitors per staff member per day
)
# Returns weekly_schedule keyed by day name (Mon..Sun)
# Each entry: avg_daily_traffic, recommended_headcount
```

### 11. Analytics Aggregate Methods

```python
# Zone-level heat map with hot/cold zone ranking
hm_analysis = await svc.heat_map_analytics(store_id=store.id, period="2026-06")

# Sales per sqm
density = await svc.sales_density(store_id=store.id, period="2026-06")

# Cross-store ranking by metric
ranking = await svc.store_ranking(period="2026-06", metric="total_sales")

# Full diagnostic report
report = await svc.store_diagnostics_report(store_id=store.id, period="2026-06")
```

---

## Business Rules

| Rule | Effect |
|---|---|
| `pii_anonymisation_required` | Traffic counts must be aggregate integers only |
| `raw_video_storage_denied` | Raw video payload → hard deny |
| `biometric_id_denied` | Any biometric field present → hard deny |
| `heatmap_pii_masking_required` | `pii_masked=False` on heatmap create → assertion error |
| `store_sqm_required` | `sqm_total <= 0` or `sqm_selling <= 0` → assertion error |
| `benchmark_min_peer_stores` | `<5` peer stores → error response, not exception |
| `lp_incident_type_validated` | Invalid `incident_type` → assertion error |

---

## Configuration Reference

All keys are tenant-scoped via the `conf` capability or `RETAIL_SIN_*` environment variables.

| Key | Default | Description |
|---|---|---|
| `traffic.counting_interval_seconds` | `60` | Sensor aggregation interval |
| `traffic.anonymisation_required` | `true` | PII stripped at ingest |
| `planogram.audit_frequency_hours` | `24` | Default audit schedule |
| `planogram.ai_compliance_check_enabled` | `true` | AI image analysis active |
| `shelf.oos_replenishment_sla_minutes` | `30` | Target replenishment time |
| `shelf.low_stock_threshold_pct` | `20` | % of par considered low |
| `heatmaps.retention_days` | `90` | Heatmap data TTL |
| `benchmarking.peer_group_min_stores` | `5` | Minimum peer group size |
| `occupancy.safety_margin_pct` | `85` | Capacity breach threshold |
| `occupancy.warning_margin_pct` | `80` | Capacity warning threshold |
| `staffing.default_traffic_to_staff_ratio` | `50` | Visitors per staff per day |

---

## Streaming Events

| Event | Trigger |
|---|---|
| `traffic_count_recorded` | `record_traffic_count()` |
| `capacity_limit_approaching` | Occupancy ≥ 80% of max |
| `planogram_audit_completed` | `record_planogram_audit()` |
| `planogram_deviation_detected` | `compliance_status != compliant` |
| `shelf_alert_raised` | `raise_shelf_alert()` |
| `shelf_alert_resolved` | `resolve_shelf_alert()` |
| `oos_replenishment_triggered` | `trigger_replenishment()` |
| `lp_incident_reported` | `report_lp_incident()` |
| `lp_incident_escalated` | `escalate_lp_incident()` |
| `conversion_event_recorded` | `record_conversion_event()` |
| `kpi_snapshot_published` | `record_kpi_snapshot()` |
| `heatmap_generated` | `create_heatmap()` |
| `sensor_network_degraded` | `sensor_network_health()` score < 50 |

---

## Composability

```apg
use retail_sin;
```

| Composing With | Integration Point |
|---|---|
| `retail_omc` | Journey stage events → conversion funnel |
| `retail_pos` | Session close events → conversion analytics |
| `retail_prm` | Planogram compliance data → promotion placement guidance |
| `retail_loy` | CLV segments → shopper journey attribution |
| `ntfy` | Critical shelf alerts, capacity warnings, LP escalations |
| `moni` | Sensor network degradation alerts |
| `schd` | Scheduled planogram audits and KPI reports |

---

## Further Reading

- `service.py` — Complete business logic implementation
- `models.py` — All Pydantic v2 data models
- `api.py` — REST API endpoint definitions
- `views.py` — Flask-AppBuilder views
- `WORLD_CLASS_IMPROVEMENTS.md` — Improvement roadmap (15 items)
- `SPECIFICATION.md` — Formal capability specification
- `cap_spec.md` — Capability contract metadata
