# APG Digital Surveillance

`intel_surveillance` is an executable APG capability for lawful, defensive digital-surveillance
and physical-surveillance-coordination workflows. It can be composed into generated APG
applications that need facility monitoring, endpoint telemetry review, authorized public-safety
monitoring, fraud monitoring, asset protection, incident watch, field-agent tasking, or
compliance evidence.

## What It Provides

- Authority, program, monitored asset, sensor, observation, alert, risk assessment, referral,
  dissemination, review, and AI-agent workflows.
- Physical surveillance coordination: field-agent tasking, observation-report ingest, and
  location-fix ingestion from field reports.
- Advanced analytics: trajectory analysis, cross-target pattern correlation, unified target
  profiles, and associate-network mapping.
- Chain-of-custody evidence registry with SHA-256 artefact fingerprinting.
- Batch observation ingestion (up to 500 items, concurrent, with deduplication).
- Authority-renewal alerting with pluggable notification adapter.
- Deterministic rules that enforce tenant context, lawful authority, privacy review,
  calibration, evidence, approvals, Bytewax lifecycle routing, and AI-agent guardrails.
- API helpers and view models that generated Python applications can call without a web
  framework dependency.
- UI route metadata and compact theme tokens for generated application screens.
- A publishable `app.py` entrypoint with self-test and semantic-model output.

## Quick Start

```bash
./.venv/bin/python capabilities/intel/surveillance/app.py
./.venv/bin/pytest -q capabilities/intel/surveillance/tests/
./.venv/bin/apg capabilities inspect intel_surveillance --json
```

### Basic Usage

```python
from capabilities.intel.surveillance import DigitalSurveillanceService
import asyncio

svc = DigitalSurveillanceService(tenant_id="tenant-a", actor_id="analyst-1")

# 1. Record a legal authority
authority = svc.record_authority(
    "auth-1", "tenant-a", "security_monitoring_authority",
    "facility-scope", "confidential", "approver-1", "2027-12-31", "evid-auth",
)

# 2. Register a surveillance program
program = svc.record_program(
    "prog-1", "tenant-a", "facility_monitoring", "Campus Watch",
    "high", "auth-1", "evid-prog",
)

# 3. Register a target and collect intelligence
async def run():
    reg = await svc.register_surveillance_target(
        "tgt-001", "auth-1", "campus-north", "2027-12-31",
    )
    loc = await svc.location_tracking("tgt-001", "GPS")
    profile = await svc.build_target_profile("tgt-001")
    print(profile)

asyncio.run(run())
```

### Physical Surveillance

```python
# Register a field agent first
svc.register_surveillance_agent(
    "agent-007", "tenant-a", "Field Agent Alpha", "human", "field_collector", "campus-north",
)

# Task the agent
tasking = await svc.field_agent_tasking(
    "tgt-001", "agent-007", "north-gate", "HIGH",
    instructions="Observe and report movements every 30 minutes.",
)

# Ingest the agent's physical observation report
report = await svc.observation_report_ingest(
    "tgt-001", "agent-007",
    "Subject exited north gate at 14:32 carrying a dark backpack.",
    media_refs=["img-001.jpg"],
    location_lat=-1.2921, location_lon=36.8219,
)
```

### Batch Ingest

```python
result = await svc.ingest_observations_batch([
    {
        "observation_id": f"obs-{i}", "program_id": "prog-1",
        "sensor_id": "sen-1", "observation_type": "network_traffic",
        "observation_reference": f"pcap-{i}", "content_fingerprint": f"fp-{i}",
        "confidence_score": 0.9, "evidence_reference": "evid-batch",
    }
    for i in range(50)
])
print(result["succeeded"], "observations ingested")
```

## Service Methods

### Core CRUD (sync)

| Method | Description |
|--------|-------------|
| `record_authority()` | Register a legal surveillance authority |
| `record_program()` | Create a surveillance programme under an authority |
| `record_asset()` | Register a monitored asset |
| `register_sensor()` | Register a sensor attached to an asset |
| `record_observation()` | Record a single observation from a sensor |
| `record_alert()` | Raise an alert from an observation |
| `record_risk()` | Record a risk assessment against an alert |
| `record_referral()` | Refer an assessment to another party |
| `record_dissemination()` | Record intelligence dissemination |
| `record_review()` | Record a compliance review |
| `register_surveillance_agent()` | Register an AI or human surveillance agent |
| `validate_agent_action()` | Validate privileged agent scope |
| `validate_batch()` | Validate a batch pipeline request |
| `dashboard_summary()` | Operational dashboard counts |

### Async Operational (await)

| Method | Description |
|--------|-------------|
| `register_surveillance_target()` | Register a target under legal authority |
| `location_tracking()` | Collect a location fix for a target |
| `communication_metadata()` | Collect and analyse comm metadata |
| `digital_footprint_analysis()` | Analyse target's digital footprint |
| `cross_platform_correlation()` | Correlate presence across platforms |
| `pattern_of_life()` | Build a pattern-of-life profile |
| `associate_network()` | Map the associate network |
| `surveillance_audit()` | Generate compliance audit trail |
| `surveillance_report()` | Generate a classified target report |
| `terminate_surveillance()` | Terminate surveillance on a target |
| `bulk_target_registration()` | Bulk-register up to 100 targets |
| `geofence_alert()` | Check target entry/exit of a geofence |
| `device_identifier_correlation()` | Correlate IMEI/MAC/fingerprint to target |
| `surveillance_compliance_report()` | Programme-level compliance report |
| `behavioural_anomaly_detection()` | Detect behavioural anomalies |
| `export_surveillance_data()` | Export all data for a target |
| `health_check()` | Service health and operational metrics |
| `lawful_intercept()` | Initiate lawful channel intercept |
| `sensor_health_check()` | Report sensor calibration health |
| `target_lifecycle_status()` | Target lifecycle status summary |
| `surveillance_kpi_report()` | KPI report for the programme |
| **`field_agent_tasking()`** | Task a field agent with a physical assignment |
| **`observation_report_ingest()`** | Ingest a physical observation report |
| **`build_target_profile()`** | Unified target intelligence profile |
| **`trajectory_analysis()`** | Movement trajectory and mode-of-transport |
| **`register_evidence()`** | Chain-of-custody evidence registry |
| **`check_authority_renewals()`** | Identify expiring authorities |
| **`ingest_observations_batch()`** | Concurrent batch observation ingest |
| **`cross_target_pattern_correlation()`** | Correlate patterns across multiple targets |

## Guardrails

The capability is defensive and compliance-first. It does not implement covert tracking,
stalking, spyware, credential capture, bypass, biometric identification, exfiltration,
live sensor control, or unauthorized monitoring. AI-agent actions that request those scopes
are denied by the rule engine.

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/intel/surveillance/*.py
./.venv/bin/pytest -q capabilities/intel/surveillance/tests/
./.venv/bin/apg capabilities implementation-audit --root capabilities/intel/surveillance --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/surveillance --json
```

---

## World-Class Enhancements (v2.0)

- **I1.** World-Class Improvements — intel_surveillance
- **I2.** Persistent Storage via Async SQLAlchemy
- **I3.** Authoritative Authority Expiry Enforcement
- **I4.** Structured Event Streaming via Bytewax Producer
- **I5.** Pydantic v2 Request/Response Models for Every Public Method
- **I6.** RBAC Permission Check on Every Entry Point
- **I7.** Distributed Caching for
- **I8.** Physical Surveillance Coordination Module
- **I9.** Target Profile Builder
- **I10.** Time-Series Location History with Trajectory Analysis
- **I11.** Media Evidence Chain-of-Custody
- **I12.** Automated Legal Authority Renewal Workflow
- **I13.** Multi-Tenant Isolation Hardening
- **I14.** Sensor Calibration Scheduler
- **I15.** Async Batch Observation Ingestion with Deduplication

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
