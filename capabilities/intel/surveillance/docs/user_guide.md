# Digital Surveillance — User Guide

**Capability ID**: `intel_surveillance` | **Domain**: `intel` | **Version**: `1.2.0`

---

## Overview

`intel_surveillance` is an executable APG capability for lawful, defensive digital and
physical surveillance coordination workflows. It enforces compliance-first design: every
operation requires a valid legal authority, produces an immutable audit trail, and rejects
prohibited scopes at the rule-engine level.

---

## Installation

```bash
pip install apg-intel-surveillance
```

Or from source inside a `uv` workspace:

```bash
uv pip install -e capabilities/intel/surveillance
```

---

## Architecture

```
DigitalSurveillanceService
    ├── Constructor (tenant_id, actor_id, auth, audit, notify, db_url, store)
    ├── Sync CRUD: authorities → programs → assets → sensors → observations
    │                         → alerts → risks → referrals → disseminations
    │                         → reviews → agents
    ├── Async Operational
    │   ├── Target lifecycle: register / bulk_register / terminate
    │   ├── Collection: location_tracking / communication_metadata
    │   │              digital_footprint_analysis / cross_platform_correlation
    │   │              lawful_intercept
    │   ├── Physical surveillance: field_agent_tasking / observation_report_ingest
    │   ├── Analytics: pattern_of_life / associate_network / trajectory_analysis
    │   │             build_target_profile / cross_target_pattern_correlation
    │   │             behavioural_anomaly_detection / geofence_alert
    │   │             device_identifier_correlation
    │   ├── Evidence: register_evidence
    │   ├── Compliance: surveillance_audit / surveillance_compliance_report
    │   │              check_authority_renewals / surveillance_report
    │   ├── Batch: bulk_target_registration / ingest_observations_batch
    │   └── Ops: health_check / sensor_health_check / target_lifecycle_status
    │            surveillance_kpi_report / export_surveillance_data
    └── Internal: _enforce / _audit / _count / _fingerprint helpers
```

---

## Concepts

### Legal Authority

Every surveillance activity must be tied to a `SurveillanceAuthority`. Authorities carry
an `expires_at` date, a classification, and an evidence reference. Expired or missing
authorities cause `PermissionError` on all write operations.

### Surveillance Target

A target is an entity (person, device, or location) placed under surveillance. Targets must
be explicitly registered (`register_surveillance_target`) under a valid authority before any
collection methods can run against them.

### Physical vs. Digital Surveillance

- **Digital surveillance** collects signals automatically via sensors: location fixes, comm
  metadata, platform presence, and network traffic.
- **Physical surveillance** involves human field agents who submit observation reports
  (`observation_report_ingest`) linked to their assigned tasking (`field_agent_tasking`).
  Both paths feed the same analytics pipeline.

### Pattern of Life

A pattern-of-life (PoL) profile aggregates location centroid, communication timing, and
routine scores. Use `pattern_of_life()` to build a PoL for a single target, and
`cross_target_pattern_correlation()` to detect coordinated behaviour across multiple targets.

### Chain of Custody

`register_evidence()` creates an immutable evidence registry entry with SHA-256 hash
verification. Reference evidence IDs in all observation and authority records to maintain
legal admissibility.

---

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment
variables prefixed with `INTEL_SURVEILLANCE_`.

| Key | Description | Default |
|-----|-------------|---------|
| `INTEL_SURVEILLANCE_DB_URL` | PostgreSQL connection URL | in-memory |
| `INTEL_SURVEILLANCE_STREAM` | Bytewax stream topic | `apg.intel.surveillance.lifecycle` |
| `INTEL_SURVEILLANCE_BATCH_CAP` | Max batch size for bulk ops | `500` |
| `INTEL_SURVEILLANCE_POL_CACHE_TTL` | Pattern-of-life cache TTL (seconds) | `300` |

---

## Workflow Walkthrough

### 1. Establish Legal Authority

```python
svc = DigitalSurveillanceService(tenant_id="acme", actor_id="analyst-1")

auth = svc.record_authority(
    authority_id="auth-1",
    tenant_id="acme",
    authority_type="security_monitoring_authority",
    scope_reference="campus-north",
    classification="confidential",
    approver_id="legal-director",
    expires_at="2027-06-30",
    evidence_reference="court-order-2026-001",
)
```

### 2. Create a Programme and Register Assets

```python
prog = svc.record_program(
    "prog-1", "acme", "facility_monitoring", "Campus Watch",
    "high", "auth-1", "evid-prog-1",
)

asset = svc.record_asset(
    "asset-cam-01", "acme", "camera_system", "north-gate-camera",
    "facilities-team", "auth-1", "privacy-review-001", "evid-asset-1",
)

sensor = svc.register_sensor(
    "sen-01", "acme", "video", "asset-cam-01",
    "cam-stream-rtsp://192.168.1.10", "it-security", "calib-cert-2026", "evid-sen-1",
)
```

### 3. Register a Target

```python
reg = await svc.register_surveillance_target(
    target_id="tgt-001",
    authority_ref="auth-1",
    scope="campus-north",
    expiry="2027-06-30",
)
```

### 4. Digital Collection

```python
# GPS location fix
loc = await svc.location_tracking("tgt-001", "GPS")

# Communication metadata analysis
comm = await svc.communication_metadata("tgt-001", "2026-06")

# Digital footprint
footprint = await svc.digital_footprint_analysis("tgt-001")

# Cross-platform identity correlation
corr = await svc.cross_platform_correlation("tgt-001", ["TWITTER", "LINKEDIN", "TELEGRAM"])
```

### 5. Physical Surveillance — Field Agent Tasking

```python
# Register the field agent
svc.register_surveillance_agent(
    "agent-007", "acme", "Field Agent Alpha", "human", "field_collector", "campus-north",
)

# Task the agent
tasking = await svc.field_agent_tasking(
    target_id="tgt-001",
    agent_id="agent-007",
    observation_zone="north-gate",
    priority="HIGH",
    instructions="Observe and photograph movements every 30 minutes. Report via secure channel.",
)

# Agent submits a physical observation report
report = await svc.observation_report_ingest(
    target_id="tgt-001",
    agent_id="agent-007",
    report_text="Subject exited north gate at 14:32 carrying dark backpack. Entered blue Toyota KBZ-123.",
    media_refs=["photo-14-32-001.jpg", "photo-14-32-002.jpg"],
    location_lat=-1.2921,
    location_lon=36.8219,
)
```

### 6. Analytics

```python
# Pattern of life
pol = await svc.pattern_of_life("tgt-001", "2026-06")

# Trajectory analysis
traj = await svc.trajectory_analysis("tgt-001", window_hours=48)
print(f"Total distance: {traj['total_distance_km']} km, Mode legs: {traj['legs'][:3]}")

# Associate network
network = await svc.associate_network("tgt-001", depth=2)

# Unified target profile
profile = await svc.build_target_profile("tgt-001")

# Behavioural anomaly detection
anomaly = await svc.behavioural_anomaly_detection("tgt-001")
if anomaly["anomaly_level"] == "HIGH":
    print("High anomaly — escalate to analyst review")
```

### 7. Cross-Target Coordination Detection

```python
# Correlate patterns across multiple targets
cross_corr = await svc.cross_target_pattern_correlation(
    target_ids=["tgt-001", "tgt-002", "tgt-003"],
    period="2026-06",
)
for pair in cross_corr["coordinated_pairs"]:
    print(f"{pair['target_a']} <-> {pair['target_b']}: score {pair['correlation_score']}")
```

### 8. Geofencing

```python
geo = await svc.geofence_alert(
    target_id="tgt-001",
    fence_lat=-1.2833,
    fence_lon=36.8167,
    radius_km=2.0,
)
if geo["inside_fence"]:
    print("Target is inside restricted zone")
```

### 9. Evidence Chain of Custody

```python
evidence = await svc.register_evidence(
    evidence_id="evid-intercept-001",
    reference_url="s3://acme-evidence/intercept-2026-06-01.pcap",
    sha256_hash="a" * 64,  # real SHA-256 hex of the artefact
    custodian_id="forensics-team",
    expiry="2031-06-01",
)
```

### 10. Batch Observation Ingest

```python
result = await svc.ingest_observations_batch([
    {
        "observation_id": f"obs-{i}",
        "program_id": "prog-1",
        "sensor_id": "sen-01",
        "observation_type": "video",
        "observation_reference": f"clip-{i}.mp4",
        "content_fingerprint": f"fp-{i:04d}",
        "confidence_score": 0.85,
        "evidence_reference": "evid-intercept-001",
    }
    for i in range(200)
])
print(f"Ingested: {result['succeeded']}, Deduped: {result['skipped_dedup']}, Failed: {result['failed']}")
```

### 11. Compliance and Reporting

```python
# Authority renewal check
renewals = await svc.check_authority_renewals(days_ahead=30)
for alert in renewals["renewal_alerts"]:
    print(f"Authority {alert['authority_id']} {alert['status']} — expires {alert['expires_at']}")

# Per-target compliance audit
audit = await svc.surveillance_audit("tgt-001")
print("Compliant:", audit["compliant"], "Issues:", audit["compliance_issues"])

# Programme-level compliance
prog_compliance = await svc.surveillance_compliance_report()

# Classified target report
report = await svc.surveillance_report("tgt-001", "confidential")

# KPI report
kpis = await svc.surveillance_kpi_report()
```

### 12. Terminate Surveillance

```python
term = await svc.terminate_surveillance(
    target_id="tgt-001",
    reason="Surveillance programme concluded. Subject cleared.",
)
print("Terminated at:", term["terminated_at"])
```

---

## Guardrails

The rule engine (`capability_contract.py`) denies any operation that:

- Lacks a tenant context
- References a missing or expired authority
- Sets `covert_tracking_scope=True`
- Sets `stalking_scope=True`
- Sets `spyware_scope=True`
- Sets `credential_capture_scope=True`
- Sets `bypass_scope=True`
- Sets `biometric_identification_scope=True`
- Sets `exfiltration_scope=True`
- Lacks a privacy review for asset registration

Violations raise `PermissionError` with the rule name as the message.

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-surveillance/dashboard` | `intel_surveillance:view` | Overview |
| `/intel-surveillance/authorities` | `intel_surveillance:authorities` | Governance |
| `/intel-surveillance/programs` | `intel_surveillance:programs` | Planning |
| `/intel-surveillance/assets` | `intel_surveillance:assets` | Assets |
| `/intel-surveillance/sensors` | `intel_surveillance:sensors` | Collection |
| `/intel-surveillance/observations` | `intel_surveillance:observations` | Collection |
| `/intel-surveillance/alerts` | `intel_surveillance:alerts` | Analysis |
| `/intel-surveillance/risk` | `intel_surveillance:risk` | Analysis |
| `/intel-surveillance/targets` | `intel_surveillance:targets` | Operations |
| `/intel-surveillance/field-taskings` | `intel_surveillance:field_taskings` | Operations |
| `/intel-surveillance/profiles` | `intel_surveillance:profiles` | Analysis |
| `/intel-surveillance/compliance` | `intel_surveillance:compliance` | Governance |

---

## Interoperability

`intel_surveillance` integrates with other APG capabilities through the composition engine.
Reference this capability in `.apg` source files:

```apg
use intel_surveillance;
```

Compose with `intel_alerts` for alerting, `intel_reporting` for output packages, and
`intel_threats` for threat context enrichment.

---

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models (SQLAlchemy + Pydantic)
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Rule engine and guardrails
- `SPECIFICATION.md` — Detailed capability specification
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap for production-grade enhancements
- `CHANGELOG.md` — Version history
