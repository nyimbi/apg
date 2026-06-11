# Asset Tracking — User Guide

**Capability ID**: `transport_tra` | **Domain**: `transport` | **Version**: `2.0.0`

---

## Description

The Asset Tracking capability delivers real-time GPS tracking, geofencing, cold-chain monitoring, container lifecycle management, multi-leg journey analytics, fleet safety scoring, and offline telemetry replay. It is the GPS telemetry backbone for the APG transport domain.

---

## Installation

```bash
pip install apg-transport-tra
```

---

## Quick Start

```python
from capabilities.transport.tra.service import AssetTrackingService

svc = AssetTrackingService(tenant_id="acme", actor_id="ops")

# Register a vehicle
await svc.register_tracked_asset(
    "VEH-001", "vehicle", "DEVICE-42",
    owner_id="fleet-mgr", tracking_technology="gps",
)

# Stream a location update and check geofences automatically
await svc.update_location("VEH-001", -1.286, 36.817, "2026-06-11T08:00:00Z", 72.5)

# Get the fleet map (clustered for large deployments)
clusters = await svc.fleet_map_clusters(geohash_precision=4)

# Journey leg segmentation
journey = await svc.journey_analytics("VEH-001", idle_threshold_minutes=30)
```

---

## Core Workflows

### 1. Real-Time GPS Tracking

Register assets and ingest location pings. Location updates trigger automatic geofence boundary checks when `check_geofences=True` (default).

```python
# Register
asset = await svc.register_tracked_asset(
    "TRK-002", "trailer", "BLE-DEVICE-09",
    owner_id="logistics-co", tracking_technology="bluetooth_ble",
)

# Ingest ping
update = await svc.update_location(
    "TRK-002", -1.295, 36.823, "2026-06-11T09:15:00Z", 0.0,
    heading_degrees=180.0, source="iot_sensor",
)
print(update["geofence_alerts"])  # list of triggered alerts
```

### 2. Geofencing

Create geofences and record discrete entry/exit events. Boundary definition format: `"lat,lng,radius_km"` for circles.

```python
gf = await svc.geofence_create(
    "Nairobi Depot", "-1.2921,36.8219,2.0",
    alert_on="both", geofence_type="depot",
)

event = await svc.geofence_event("VEH-001", gf["id"], "entry")
print(event["alert_raised"])  # True if alert_on_entry configured
```

### 3. Cold Chain Monitoring

Record temperature readings against standard profiles. Breach raises a high-severity alert automatically.

```python
reading = await svc.cold_chain_monitoring(
    "REEFER-01", temperature=9.5, humidity=85.0, standard="haccp",
)
print(reading["breached"])  # True — haccp range is 2–8 °C

# Compliance summary for a shipment period
summary = await svc.cold_chain_compliance_summary(
    "REEFER-01", "2026-Q2", standard="haccp",
)
print(summary["compliance_pct"], summary["certificate_eligible"])
```

**Supported standards**: `atp_agreement`, `haccp`, `gxp`, `who_guidelines`, `pda_technical_report`, `iata_live_animals`, `iata_perishables`

### 4. Container Tracking

Register and update containers with ISO number and seal management.

```python
svc.register_container("CONT-001", "acme", "TCKU1234567", "SEAL-99", "acme-fleet", "Mombasa Port", "2026-06-11T06:00:00Z")
await svc.container_tracking("CONT-001", "Nairobi ICD", "in_transit")
```

**Container dwell report** (detention risk):

```python
dwell = await svc.container_dwell_report("GF-MOMBASA", free_time_hours=72.0)
at_risk = [r for r in dwell["dwell_records"] if r["detention_risk"]]
```

### 5. Asset Utilisation

```python
util = await svc.asset_utilisation("VEH-001", "weekly")
print(util["utilisation_pct"], util["distance_km"])

# Fleet-wide benchmark
bench = await svc.fleet_utilisation_benchmark("weekly", cost_per_idle_hour=12.0)
print(bench["percentiles"])       # p25/p50/p75/p95
print(bench["total_idle_cost"])   # KES/USD at given rate
```

### 6. Journey Analytics

Segments location history into driving legs and stops. A stop begins when speed < 2 km/h for `idle_threshold_minutes` worth of consecutive pings (assumed 5-minute interval).

```python
journey = await svc.journey_analytics("VEH-001", idle_threshold_minutes=30)
for leg in journey["legs"]:
    print(f"Leg {leg['leg']}: {leg['distance_km']} km at {leg['avg_speed_kmh']} km/h")
for stop in journey["stops"]:
    print(f"Stop {stop['stop']}: {stop['dwell_minutes']} min at ({stop['lat']},{stop['lng']})")
```

### 7. Harsh Event Detection

Computes implied acceleration from consecutive speed readings. Raises alerts for harsh braking, harsh acceleration, and speeding.

```python
events = await svc.detect_harsh_events(
    "VEH-001",
    harsh_brake_g=0.3,
    harsh_accel_g=0.3,
    speed_limit_kmh=110.0,
)
print(events["event_count"], events["harsh_events"])
```

```python
# Speeding violation league table (fleet safety)
violations = await svc.speeding_violations(speed_limit_kmh=100.0, top_n=10)
for v in violations["results"]:
    print(v["asset_id"], v["violation_count"], v["max_speed_kmh"])
```

### 8. Location Anomaly Detection (Anti-Spoofing)

Before storing a suspicious ping, check for GPS spoofing or cloned trackers:

```python
check = await svc.detect_location_anomaly(
    "VEH-001", new_lat=-1.500, new_lng=37.100,
    new_timestamp="2026-06-11T10:00:00Z",
    max_plausible_speed_kmh=250.0,
)
if check["anomaly_detected"]:
    print(f"Spoofing suspected — implied {check['implied_speed_kmh']} km/h")
```

### 9. Offline Telemetry Replay

For assets returning from connectivity blackouts (tunnels, remote areas):

```python
buffered = [
    {"latitude": -1.290, "longitude": 36.820, "speed_kmh": 55.0,
     "heading_degrees": 45.0, "timestamp": "2026-06-11T07:00:00Z", "source": "gps"},
    # ... more pings
]
result = await svc.replay_buffered_telemetry("VEH-001", buffered)
print(result["accepted"], result["skipped_duplicates"], result["errors"])
```

### 10. Fleet Map (Clustered)

```python
# Large fleet — return geohash clusters to prevent browser overload
clusters = await svc.fleet_map_clusters(geohash_precision=4, active_only=True)

# Zoom in — individual features
view = await svc.fleet_map_view({"active_only": True, "max_age_minutes": 30})
```

### 11. Theft Response

```python
theft = await svc.theft_alert(
    "VEH-001", trigger="tamper_detected",
    last_known_location="-1.286,36.817",
)
# Asset is deactivated; critical alert raised; audit event emitted
```

### 12. Custom Tracking Agents

Register AI agents for automated tracking workflows:

```python
agent = svc.register_tracking_agent(
    "AGENT-01", "acme", "Cold Chain Watcher",
    runtime="claude_code", role="cold_chain_monitor", scope="all_reefers",
)
```

---

## Reporting

| Method | Output |
|--------|--------|
| `tracking_report(asset_id, period)` | Full report: pings, alerts, cold chain, geofences, theft |
| `tracking_analytics_detail(period)` | Fleet-level analytics by asset type and alert type |
| `alert_summary()` | Active alerts by type and severity |
| `cold_chain_compliance_report(period)` | Fleet-wide compliance rate |
| `dashboard_summary(tenant_id)` | KPI card counts |
| `export_tracking_data(period, format="csv")` | Export metadata + download ref |

---

## Permissions

| Permission | Access |
|-----------|--------|
| `transport_tra:view` | Dashboard, live map |
| `transport_tra:assets` | Asset CRUD |
| `transport_tra:geofencing` | Geofence CRUD |
| `transport_tra:alerts` | Alert management |
| `transport_tra:cold_chain` | Temperature records |
| `transport_tra:containers` | Container lifecycle |
| `transport_tra:utilisation` | Utilisation reports |
| `transport_tra:reports` | Report generation |
| `transport_tra:admin` | Agent workbench, settings |

---

## Business Rules (Enforced at Runtime)

| Rule | Behaviour |
|------|-----------|
| `tamper_alert_escalation_required` | Location update with `tamper_detected=True` is rejected; escalation required |
| `asset_unique_id_required` | Asset registration without unique device ID is denied |
| `geofence_area_required` | Geofence creation without boundary definition is denied |
| `container_iso_number_required` | Container registration without ISO number is denied |
| `cross_tenant_tracking_denied` | Write operations to another tenant's data are denied |
| `tracking_batch_requires_bytewax` | Batch telemetry must route through Bytewax stream |

---

## Composability

```apg
use transport_tra;
use transport_dis;   # dispatch receives live positions from tra
use transport_fle;   # fleet registry provides vehicle IDs
use transport_car;   # cargo custody chain links to container tracking
use comp;            # cold chain compliance feeds certificate generation
```

---

## Environment Variables

All config keys are tenant-scoped. Override via environment:

```
TRANSPORT_TRA_MONITORING_DATA_RETENTION_DAYS=730
TRANSPORT_TRA_GEOFENCING_MAX_GEOFENCES_PER_TENANT=1000
TRANSPORT_TRA_UTILISATION_IDLE_THRESHOLD_MINUTES=15
TRANSPORT_TRA_COLD_CHAIN_CONTINUOUS_LOGGING_ENABLED=true
```

---

## Further Reading

- `service.py` — Full business logic with all async methods
- `models.py` — Dataclass models (TrackedAsset, Geofence, ColdChainRecord, …)
- `capability_contract.py` — Supported types, rules, streaming config
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 improvement opportunities with technical rationale
- `tests/test_service.py` — Service-layer test suite
