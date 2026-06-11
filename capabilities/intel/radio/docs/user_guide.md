# Radio Intelligence (intel_radio) — User Guide

**Capability ID**: `intel_radio` | **Domain**: `intel` | **Version**: `1.2.0`

---

## Overview

`intel_radio` is the APG capability for lawful, passive radio intelligence
(RINT) collection, spectrum analysis, emitter identification, and RF direction
finding. It supports the full SIGINT workflow: authority registration, band plan
definition, receiver management, signal collection, transmission decoding,
geolocation, jamming assessment, ELINT/COMINT product generation, and
dissemination control.

All operations are tenant-scoped and governed by a deterministic rule engine
that enforces lawful authority, frequency bounds, evidence requirements, and
AI-agent guardrails.

---

## Installation

```bash
pip install apg-intel-radio
```

Or within the APG monorepo:

```bash
uv run pip install -e capabilities/intel/radio
```

---

## Architecture

```
RadioIntelligenceService(tenant_id, actor_id, *, auth, audit, notify, db_url, store)
    │
    ├── Governance: record_authority, record_band_plan, register_receiver
    ├── Collection: record_session, record_observation, bulk_observation_ingest
    ├── Analysis:   decode_transmission, identify_emitter, signal_classification_batch
    ├── Geolocation: radio_direction_finding, tdoa_geolocation, doppler_direction_finding,
    │                geo_emitter_tracking
    ├── Spectrum:   frequency_scan, spectrum_analysis, interference_detection,
    │               frequency_monitoring_schedule, frequency_deconfliction
    ├── ELINT/COMINT: elint_product, comms_intelligence_brief, radio_order_of_battle,
    │                 radio_intelligence_report
    ├── Safety:     register_exclusion_zone, jamming_assessment, frequency_compliance_audit
    ├── Graph:      signal_link_graph, comms_pattern_analysis
    └── Reporting:  dashboard_summary, export_observations, health_check, radio_analytics
```

The service follows the **adapter/store** pattern: inject `auth`, `audit`,
`notify`, `db_url`, or `store` collaborators without changing call sites.

---

## Core Workflow

### 1. Register Lawful Authority

Before any collection can proceed, register the governing authority:

```python
svc = RadioIntelligenceService("tenant-a", actor_id="analyst-1")

svc.record_authority(
    authority_id="auth-sigint-2026",
    tenant_id="tenant-a",
    authority_type="spectrum_license",          # or court_order, exec_directive, …
    scope_reference="national-sigint-framework",
    classification="confidential",
    approver_id="director-signals",
    expires_at="2027-06-30T00:00:00Z",
    evidence_reference="license-doc-001",
)
```

### 2. Define Band Plans

```python
svc.record_band_plan(
    band_id="bp-vhf",
    tenant_id="tenant-a",
    band_type="VHF",
    name="VHF Land Mobile",
    frequency_min_mhz=136.0,
    frequency_max_mhz=174.0,
    authority_id="auth-sigint-2026",
    evidence_reference="itu-vhf-plan",
)
```

### 3. Register Receivers

```python
svc.register_receiver(
    receiver_id="rx-nairobi-1",
    tenant_id="tenant-a",
    receiver_type="SDR",
    site_reference="nairobi-roof-site",
    custodian_id="technician-1",
    authority_id="auth-sigint-2026",
    calibration_reference="cal-2026-03-15",
    evidence_reference="inventory-rx-001",
)
```

### 4. Open a Collection Session

```python
svc.record_session(
    session_id="sess-001",
    tenant_id="tenant-a",
    band_id="bp-vhf",
    receiver_id="rx-nairobi-1",
    session_type="passive_intercept",
    started_at="2026-06-01T08:00:00Z",
    ended_at="2026-06-01T12:00:00Z",
    collection_plan_reference="plan-aor-nairobi",
    evidence_reference="session-log-001",
)
```

### 5. Ingest Observations

**Single observation:**

```python
svc.record_observation(
    observation_id="obs-001",
    tenant_id="tenant-a",
    session_id="sess-001",
    frequency_mhz=156.8,     # VHF Marine Distress Channel 16
    signal_type="FM",
    signal_fingerprint="sha256:abc123…",
    observed_at="2026-06-01T08:15:32Z",
    confidence_score=0.92,
    evidence_reference="iq-file-001",
)
```

**Bulk ingest (up to 5 000 records):**

```python
result = await svc.bulk_observation_ingest([
    {
        "observation_id": f"obs-{i:04d}",
        "session_id": "sess-001",
        "frequency_mhz": 136.0 + i * 0.025,
        "signal_type": "AM",
        "signal_fingerprint": f"fp-{i}",
        "observed_at": "2026-06-01T08:00:00Z",
        "confidence_score": 0.85,
        "evidence_reference": "bulk-ref",
    }
    for i in range(500)
])
# result["succeeded"] == 500
```

---

## Spectrum Operations

### Frequency Scan

```python
scan = await svc.frequency_scan(
    frequency_range=(118.0, 137.0),   # Aviation VHF band, MHz
    location="entebbe-airport-site",
    duration=120.0,                    # seconds
)
# scan["signals"] sorted by power_dbm descending
```

### Spectrum Occupancy Analysis

```python
analysis = await svc.spectrum_analysis(
    frequency_range=(118.0, 137.0),
    period="24h",
)
# analysis["occupancy_pct"], analysis["peak_frequency_mhz"]
```

### Monitoring Schedule

```python
schedule = await svc.frequency_monitoring_schedule(
    frequency_list=[121.5, 156.8, 243.0, 406.0],   # distress frequencies
    interval=5.0,                                    # every 5 minutes
)
```

### Interference Detection

```python
detection = await svc.interference_detection(target_frequency=156.8)
# detection["severity"]:  NONE | MEDIUM | HIGH
# detection["recommended_mitigation"]: NO_ACTION | NOTCH_FILTER | FREQUENCY_HOP
```

### Frequency Deconfliction

Check a proposed allocation before registering a band plan:

```python
check = await svc.frequency_deconfliction(
    proposed_frequency_mhz=145.5,
    proposed_bandwidth_khz=25.0,
)
# check["severity"]: CLEAR | WARNING | BLOCKED
# check["conflicts"]: list of overlapping band plans / ITU designations
# check["suggested_alternatives_mhz"]: [145.75, 145.25]
```

---

## Geolocation

### Bearing-Only Direction Finding

Requires 2+ receivers with known positions and measured bearings:

```python
fix = await svc.radio_direction_finding(
    signal_id="obs-001",
    receiver_positions=[
        {"lat": -1.286, "lon": 36.817, "bearing_deg": 045.0},
        {"lat": -1.300, "lon": 36.840, "bearing_deg": 315.0},
        {"lat": -1.265, "lon": 36.800, "bearing_deg": 090.0},
    ],
)
# fix["mean_bearing_deg"], fix["quality_score"] (0–1)
```

### TDOA Geolocation

More accurate fix using Time-Difference-of-Arrival (requires 3+ receivers):

```python
fix = await svc.tdoa_geolocation(
    signal_id="sig-xray-001",
    receiver_tdoa=[
        {"lat": -1.286, "lon": 36.817, "tdoa_us": 0.0},    # reference
        {"lat": -1.300, "lon": 36.840, "tdoa_us": 12.5},
        {"lat": -1.265, "lon": 36.800, "tdoa_us": 8.3},
        {"lat": -1.270, "lon": 36.850, "tdoa_us": 15.1},
    ],
)
# fix["estimated_lat"], fix["estimated_lon"]
# fix["cep_km"] — circular error probable
# fix["geometry_quality"] — 0 (poor) to 1 (excellent)
```

### Doppler Direction Finding

Single-site bearing estimate using rotating antenna array:

```python
fix = await svc.doppler_direction_finding(
    signal_id="sig-001",
    doppler_shifts_hz=[0.5, 2.1, 4.8, 6.2, 5.9, 3.4, 0.8, -1.2,
                       -3.5, -5.8, -6.1, -4.9, -2.3, -0.4, 1.1, 3.2],
    antenna_rotation_rpm=600.0,
    centre_frequency_mhz=145.5,
)
# fix["bearing_deg"], fix["confidence"]
```

### Emitter Tracking

Track a moving emitter across DF fixes:

```python
track = await svc.geo_emitter_tracking(
    emitter_id="emitter-alpha",
    fix_history=[
        {"lat": -1.290, "lon": 36.820, "timestamp_s": 1000.0},
        {"lat": -1.285, "lon": 36.825, "timestamp_s": 1060.0},
        {"lat": -1.280, "lon": 36.830, "timestamp_s": 1120.0},
    ],
)
# track["speed_kmh"], track["heading_deg"]
# track["predicted_lat"], track["predicted_lon"]
# track["mobility_class"]: STATIC | MOBILE | AIRBORNE
```

---

## Signal Analysis

### Transmission Decoding

Supported protocols: ADS-B, AIS, ACARS, APRS, DMR, P25, TETRA, DSTAR,
CW, RTTY, FSK, NOAA APT, SELCAL, Mode-S.

```python
decode = await svc.decode_transmission(
    signal_id="obs-001",
    protocol="ADS_B",
)
# decode["decoded_fields"]["icao"], ["callsign"], ["altitude_ft"], ["speed_kt"]
```

### Emitter Identification

```python
emitter = await svc.identify_emitter({
    "frequency_mhz": 9500.0,
    "modulation": "PULSE",
    "power_dbm": 60.0,
    "bandwidth_khz": 5000.0,
    "pulse_width_us": 2.0,
    "pri_us": 1000.0,
})
# emitter["emitter_class"]: SEARCH_RADAR | FIRE_CONTROL_RADAR | VHF_LAND_MOBILE | …
# emitter["confidence"]: 0.0–1.0
```

### Signal Anomaly Detection

```python
result = await svc.signal_anomaly_detection(
    session_id="sess-001",
    window_size=100,    # analyse the last 100 observations
)
# result["anomaly_count"]
# result["anomalies"]: [{observation_id, frequency_mhz, reasons: [CONFIDENCE_OUTLIER|FREQUENCY_OUTLIER]}]
```

### Batch Classification

```python
batch = await svc.signal_classification_batch(["obs-001", "obs-002", "obs-003"])
# batch["threat_distribution"]: {"NONE": 1, "LOW": 1, "MEDIUM": 1}
# batch["classifications"]: [{observation_id, signal_class, threat_level, frequency_mhz}]
```

### Communications Pattern Analysis

```python
patterns = await svc.comms_pattern_analysis(["sess-001", "sess-002"])
# patterns["traffic_pattern"]: BURST | CONTINUOUS
# patterns["reused_frequencies_mhz"]: [156.8]
# patterns["band_distribution"]: {"VHF": 45, "UHF": 12}
```

---

## ELINT / COMINT Products

### ELINT Product

```python
product = await svc.elint_product(
    emitter_id=emitter["emitter_id"],
    classification="secret",
    dissemination_marking="NATIONAL_ONLY",
)
# product["product_type"] == "ELINT"
# product["threat_level"]: CRITICAL | HIGH | MEDIUM | LOW | UNKNOWN
# product["df_fixes"]: associated direction-finding records
```

### COMINT Brief

```python
brief = await svc.comms_intelligence_brief(classification="confidential")
```

### Radio Intelligence Report

```python
report = await svc.radio_intelligence_report(classification="confidential")
# report["summary"] — counts of all RINT artefacts and analytics
```

### Electronic Order of Battle

```python
orbat = await svc.radio_order_of_battle(region="East Africa AOR")
# orbat["emitter_class_distribution"]: {"SEARCH_RADAR": 2, "VHF_LAND_MOBILE": 14}
```

### Signal Pattern Library

```python
library = await svc.signal_pattern_library()
# library["type_distribution"]: classification type breakdown
# library["mean_confidence"]: mean analyst confidence across all classifications
```

---

## Safety and Compliance

### Exclusion Zones

Prevent collection on protected frequencies within a geographic area:

```python
zone = await svc.register_exclusion_zone(
    zone_id="ez-diplomatic-quarter",
    name="Diplomatic Quarter — Protected",
    frequency_min_mhz=136.0,
    frequency_max_mhz=140.0,
    polygon_wkt="POLYGON((36.80 -1.30, 36.90 -1.30, 36.90 -1.20, 36.80 -1.20, 36.80 -1.30))",
    reason="Protected diplomatic communications per Vienna Convention",
)
```

### Jamming Assessment

```python
assessment = await svc.jamming_assessment(
    frequency_mhz=156.8,
    duration_s=120.0,
)
# assessment["intentional_jamming"]: True | False
# assessment["jamming_type"]: SPOT | SWEEP | BARRAGE | FOLLOW_ON | DECEPTIVE
# assessment["recommended_countermeasure"]
```

### Frequency Compliance Audit

```python
audit = await svc.frequency_compliance_audit()
# audit["compliant"]: True | False
# audit["issues"]: [{"band_id": …, "issue": INVALID_FREQUENCY_RANGE | MISSING_AUTHORITY}]
```

### AI Agent Action Validation

```python
svc.validate_agent_action(
    tenant_id="tenant-a",
    privileged_scope=True,
    human_approval_recorded=True,
    transmit_scope=False,       # False required — transmit is denied
    jamming_scope=False,        # False required
)
```

---

## Link Analysis

### Signal Link Graph

Build an adjacency graph for the `grph` capability:

```python
graph = await svc.signal_link_graph()
# graph["nodes"]: [{id, type: EMITTER|SESSION|OBSERVATION, label, frequency_mhz}]
# graph["edges"]: [{source, target, relation: CONTAINS|EMITS}]
# Compatible with NetworkX: nx.node_link_graph({"nodes": graph["nodes"], "links": graph["edges"]})
```

---

## Operational Dashboard

```python
summary = svc.dashboard_summary("tenant-a")
health = await svc.health_check()
analytics = await svc.radio_analytics()
calibration = await svc.receiver_calibration_status()
```

---

## Data Export

```python
export = await svc.export_observations(fmt="csv")   # or "json"
# export["record_count"], export["content_fingerprint"], export["export_id"]
```

---

## Cross-Border Monitoring

```python
result = await svc.cross_border_monitoring(
    border_region="Kenya-Somalia border AOR",
    frequencies=[156.8, 162.0, 167.5, 170.0],
)
# result["cross_border_suspected_count"]
# result["detections"]: [{frequency_mhz, bearing_deg, power_dbm, cross_border_suspected}]
```

---

## APG Composition

Reference `intel_radio` in `.apg` source files:

```apg
use intel_radio;
```

It integrates with:

| Capability | Integration |
|------------|-------------|
| `intel_threats` | Emitter threat escalation |
| `intel_correlation` | Cross-domain signal correlation |
| `intel_dashboard` | Live spectrum and ORBAT visualisation |
| `grph` | Signal link graph rendering |
| `auth` | Lawful authority enforcement |
| `audl` | Audit event streaming |
| `ntfy` | Alert and notification dispatch |
| `nlpc` | Natural-language tasking of collection |

---

## Configuration

All keys are tenant-scoped. Set via the `conf` capability or environment
variables prefixed `INTEL_RADIO_`:

| Key | Default | Description |
|-----|---------|-------------|
| `INTEL_RADIO_DB_URL` | `None` | PostgreSQL async connection string |
| `INTEL_RADIO_STREAM_TOPIC` | `apg.intel.radio.lifecycle` | Bytewax topic |
| `INTEL_RADIO_MAX_BATCH` | `5000` | Maximum bulk ingest batch size |
| `INTEL_RADIO_SCAN_CAP_SIGNALS` | `20` | Maximum signals per frequency scan |
| `INTEL_RADIO_EXCLUSION_STRICT` | `true` | Block (vs. warn) on exclusion zone hit |

---

## Further Reading

- `service.py` — Business logic implementation (1 700+ lines, 30+ async methods)
- `models.py` — Data models (RadioAuthority, RadioBandPlan, RadioReceiver, …)
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Supported types and rule engine
- `radio_runtime.py` — Validation helpers
- `WORLD_CLASS_IMPROVEMENTS.md` — Engineering improvement roadmap
- `README.md` — Quick reference
