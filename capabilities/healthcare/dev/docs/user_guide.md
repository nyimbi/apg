# Medical Device Management — User Guide

**Capability ID**: `healthcare_dev` | **Domain**: `healthcare` | **Version**: `1.1.0`

## Description

Medical device lifecycle management covering device inventory with FDA UDI tracking, preventive and corrective maintenance scheduling with work orders, calibration record management, adverse event reporting, chain-of-custody assignment, device loan management, decontamination/sterility tracking, fleet benchmarking, manufacturer quality scorecards, warranty lifecycle alerts, multi-jurisdiction regulatory profiles, and tamper-evident NATS-backed audit replay.

Enforces UDI requirements for Class II/III devices, blocks use of recalled or calibration-overdue devices, automatically escalates serious adverse events, and supports compliance profiles for FDA, EU MDR 2017/745, UKCA, Health Canada, and TGA jurisdictions.

## Installation

```bash
pip install apg-healthcare-dev
```

## Quick Start

```python
import asyncio
from apg_healthcare_dev.service import MedicalDeviceManagementService
from apg_healthcare_dev.models import DeviceCreate
from datetime import datetime, timedelta

async def main():
    svc = MedicalDeviceManagementService(tenant_id="hosp-001", actor_id="biomed-eng-1")

    # Register a Class II infusion pump — UDI required
    device = await svc.register_device(DeviceCreate(
        tenant_id="hosp-001",
        name="Baxter Sigma Spectrum 6.05",
        device_type="infusion_pump",
        device_class="class_ii",
        manufacturer="Baxter International",
        model_number="SIGMA-6.05",
        serial_number="SN-20240001",
        udi="(01)00810024021052(21)SN-20240001",
        udi_format="gs1",
        location="ICU-3",
        department="Intensive Care",
        warranty_expiry=datetime.utcnow() + timedelta(days=365 * 3),
        created_by="biomed-eng-1",
    ))
    print(device.id)

asyncio.run(main())
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NATS_URL` | NATS JetStream server URL for durable audit streaming | No |
| `AUDIT_HMAC_KEY` | HMAC-SHA256 key for audit event tamper detection | Recommended |
| `OLLAMA_BASE_URL` | Ollama server URL for ML anomaly detection | No |

## Provides

- `device_inventory_management`
- `maintenance_schedule_management`
- `calibration_record_tracking`
- `fda_udi_tracking`
- `adverse_event_reporting`
- `work_order_management`
- `device_lifecycle_management`
- `regulatory_submission_support`
- `chain_of_custody_tracking`
- `device_loan_management`
- `decontamination_record_tracking`
- `fleet_benchmarking`
- `manufacturer_quality_scorecard`
- `warranty_lifecycle_alerts`
- `multi_jurisdiction_regulatory_profiles`
- `durable_audit_replay`

## Requires

- `auth` — Role-based access for biomedical engineers and clinical staff
- `audl` — Audit trail for all device modifications
- `mten` — Multi-tenant isolation
- `conf` — Tenant configuration
- `ntfy` — Overdue calibration and recall alerts

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/healthcare-dev/dashboard` | `healthcare_dev:view` | Overview |
| `/healthcare-dev/inventory` | `healthcare_dev:inventory` | Devices |
| `/healthcare-dev/inventory/register` | `healthcare_dev:inventory_write` | Devices |
| `/healthcare-dev/inventory/<id>` | `healthcare_dev:inventory` | Devices |
| `/healthcare-dev/maintenance` | `healthcare_dev:maintenance` | Maintenance |
| `/healthcare-dev/work-orders` | `healthcare_dev:maintenance` | Maintenance |
| `/healthcare-dev/calibration` | `healthcare_dev:calibration` | Calibration |
| `/healthcare-dev/adverse-events` | `healthcare_dev:adverse_events` | Safety |
| `/healthcare-dev/assignments` | `healthcare_dev:assignment_write` | Operations |
| `/healthcare-dev/loans` | `healthcare_dev:loan_write` | Operations |
| `/healthcare-dev/decontamination` | `healthcare_dev:decontamination_write` | Safety |
| `/healthcare-dev/analytics/benchmark` | `healthcare_dev:analytics` | Analytics |
| `/healthcare-dev/analytics/manufacturer` | `healthcare_dev:analytics` | Analytics |
| `/healthcare-dev/warranty-alerts` | `healthcare_dev:inventory` | Compliance |
| `/healthcare-dev/regulatory-profile` | `healthcare_dev:compliance` | Compliance |
| `/healthcare-dev/audit` | `healthcare_dev:audit` | Compliance |

## Service Methods Reference

### Device Registration & Inventory

```python
# Register a device (UDI required for Class II/III)
device = await svc.register_device(DeviceCreate(...))

# Update operational status
device = await svc.update_device_status(tenant_id, device_id, "in_maintenance")

# Retrieve single device
device = await svc.get_device(tenant_id, device_id)

# List with optional filters
devices = await svc.list_devices(tenant_id, device_type="infusion_pump", status="active")

# Location-scoped inventory summary
inv = await svc.device_inventory("ICU-3", filters={"department": "Intensive Care"})

# UDI cross-tenant lookup
device = await svc.udi_lookup("(01)00810024021052(21)SN-20240001")
```

### Maintenance

```python
from apg_healthcare_dev.models import MaintenanceScheduleCreate

sched = await svc.schedule_maintenance(MaintenanceScheduleCreate(
    tenant_id="hosp-001",
    device_id=device.id,
    maintenance_type="preventive",
    scheduled_date=datetime.utcnow() + timedelta(days=30),
    assigned_to="tech-007",
    estimated_hours=2.5,
    instructions="Inspect pump mechanism, replace battery, clean tubing pathway.",
    created_by="biomed-eng-1",
))

# Complete a work order
completed = await svc.complete_maintenance(tenant_id, sched.id, notes="Battery replaced. All checks passed.")

# Upcoming PM schedule (devices due within 30 days)
schedule = await svc.preventive_maintenance_schedule(tenant_id)
```

### Calibration

```python
from apg_healthcare_dev.models import CalibrationRecordCreate

cal = await svc.record_calibration(CalibrationRecordCreate(
    tenant_id="hosp-001",
    device_id=device.id,
    calibrated_by="cal-lab-1",
    calibration_date=datetime.utcnow(),
    next_due_date=datetime.utcnow() + timedelta(days=365),
    certificate_reference="CERT-LAB-2026-0412",
    result="pass",
    notes="All channels within tolerance.",
    created_by="biomed-eng-1",
))

# Devices due for calibration within N days
alerts = await svc.calibration_due_alerts(tenant_id, days_ahead=30)
```

### Adverse Events

```python
from apg_healthcare_dev.models import AdverseEventCreate

event = await svc.report_adverse_event(AdverseEventCreate(
    tenant_id="hosp-001",
    device_id=device.id,
    event_type="malfunction",
    severity="serious",
    description="Pump over-infused by 12% over 4-hour period.",
    patient_id="PT-00421",
    occurred_at=datetime.utcnow(),
    reported_by="nurse-rn-42",
    immediate_action_taken="Pump removed from service. Patient assessed.",
    created_by="nurse-rn-42",
))
# Serious severity automatically sets device status to in_maintenance

# Close event with root cause and corrective action
closed = await svc.close_adverse_event(
    tenant_id, event.id,
    root_cause="Worn motor encoder causing speed drift.",
    corrective_action="Motor encoder replaced. Recalibrated and re-qualified.",
)
```

### Recall Management

```python
recall = await svc.recall_management(
    recall_id="FDA-RECALL-2026-00147",
    affected_devices=[device_id_1, device_id_2],
)
# All listed devices are set to status='recalled'
```

### Chain of Custody (New in v1.1)

```python
# Assign device to a nurse for a shift
assignment = await svc.assign_device(
    device_id=device.id,
    assignee_id="nurse-rn-42",
    shift_id="SHIFT-2026-06-11-DAY",
    location="ICU-3 Bay 7",
)

# Release at end of shift
released = await svc.release_device(
    assignment_id=assignment["id"],
    condition="Good — no damage observed.",
)
```

Raises `PolicyViolationError` if device is recalled, has overdue calibration, or is retired/out-of-service.

### Device Loan Management (New in v1.1)

```python
loan = await svc.create_device_loan(
    device_id=device.id,
    borrower_org="Nairobi County Hospital",
    loan_start=datetime.utcnow(),
    loan_end=datetime.utcnow() + timedelta(days=14),
    contact="biomedical@nch.go.ke",
)
# Device status set to 'on_loan'

returned = await svc.return_device_from_loan(
    loan_id=loan["id"],
    condition_notes="Minor scuff on casing. All functions normal.",
)
# Device status set to 'in_maintenance'; requalification_required=True
```

### Decontamination / Sterility (New in v1.1)

```python
record = await svc.record_decontamination(
    device_id=device.id,
    cycle_type="steam_autoclave",       # steam_autoclave | ethylene_oxide | hydrogen_peroxide
                                         # | chemical_disinfection | UV_disinfection
    steriliser_id="AUTOCLAVE-OR-2",
    cycle_number="CYC-20260611-047",
    result="pass",                       # pass | fail | inconclusive
    biological_indicator="BI-PASS",
)
# sal_classification: 'SAL_1e-6' for steam/EtO/H2O2; 'high_level_disinfection' otherwise
```

### Fleet Benchmarking (New in v1.1)

```python
benchmark = await svc.fleet_benchmark(
    tenant_id="hosp-001",
    device_type="infusion_pump",
    metric="adverse_event_rate",     # adverse_event_rate | calibration_pass_rate | uptime_pct
)
# Returns per-device z-scores; outlier=True where |z| > 2
# Example response excerpt:
# {
#   "fleet_mean": 0.4,
#   "fleet_stddev": 0.22,
#   "outlier_count": 1,
#   "devices": [
#     {"device_id": "...", "score": 1.2, "z_score": 3.6, "outlier": True, "outlier_direction": "high"},
#     ...
#   ]
# }
```

### Manufacturer Quality Scorecard (New in v1.1)

```python
scorecard = await svc.manufacturer_quality_scorecard(
    tenant_id="hosp-001",
    manufacturer="Baxter International",
)
# Returns quality_score (0-100) and quality_tier (A/B/C/D)
# Inputs: calibration pass rate, adverse event rate, serious adverse rate, active recalls
```

### Warranty Expiry Alerts (New in v1.1)

```python
alerts = await svc.warranty_expiry_alerts(tenant_id="hosp-001", days_ahead=90)
# Each alert includes days_remaining, expired bool, replacement_cost_tier (low/medium/high)
```

### Regulatory Profile Overlay (New in v1.1)

```python
# Supported jurisdictions: FDA, EU_MDR, UKCA, HC_CMDR, TGA
profile = await svc.regulatory_profile(tenant_id="hosp-001", jurisdiction="EU_MDR")
# Returns MDR regulation name, UDI standard, class nomenclature, MDR deadline days, PMS requirements
```

### Durable Audit Trail (New in v1.1)

```python
import os
os.environ["NATS_URL"] = "nats://localhost:4222"
os.environ["AUDIT_HMAC_KEY"] = "change-me-in-production"

# Publish a signed audit event (auto-called internally; also callable directly)
result = await svc.publish_audit_event({
    "tenant_id": "hosp-001",
    "event": "manual_audit_note",
    "entity_id": device.id,
    "note": "Annual JCIA inspection passed.",
})
# result["published_to_nats"] = True when NATS_URL is configured
# result["signature"] = HMAC-SHA256 hex digest

# Replay events for compliance inspection
from datetime import datetime, timedelta
events = await svc.replay_audit_events(
    tenant_id="hosp-001",
    from_ts=datetime.utcnow() - timedelta(days=30),
    to_ts=datetime.utcnow(),
)
# Each event has tampered=True if HMAC signature does not match
```

When `NATS_URL` is set, events are written to the `apg.healthcare.dev.audit.<tenant_id>` subject in a JetStream stream configured for 7-year retention (21 CFR Part 11 compliant). The in-memory fallback is always maintained as a secondary store.

## Analytics Methods

```python
# Device utilisation
util = await svc.device_utilisation_report(tenant_id, period="2026-Q2")

# Maintenance KPIs
maint = await svc.maintenance_analytics(tenant_id, period="2026-Q2")

# Adverse event trends by type and severity
trends = await svc.adverse_event_trend(tenant_id, period="2026-Q2")

# Full device KPI card
kpis = await svc.device_kpi_summary(tenant_id, period="2026-Q2")

# ISO 13485 QMS compliance score
iso = await svc.iso_13485_compliance(tenant_id)

# FDA 510(k) status check
fda = await svc.fda_510k_status(tenant_id, device_id)

# EU CE marking compliance
ce = await svc.ce_marking_check(tenant_id, device_id)

# Full device lifecycle history
lifecycle = await svc.device_lifecycle_report(tenant_id, device_id)

# Risk score (class + calibration + adverse event history)
risk = await svc.device_risk_assessment(tenant_id, device_id)
```

## Business Rules Summary

| Rule | Trigger | Effect |
|------|---------|--------|
| udi_required_for_class_ii_iii | register_device with class_ii/iii, no UDI | PolicyViolationError |
| recalled_device_blocked | assign_device or usage_log | PolicyViolationError |
| calibration_overdue_blocks_assignment | assign_device | PolicyViolationError |
| out_of_service_not_assignable | assign_device | PolicyViolationError |
| serious_adverse_event_auto_escalates | report_adverse_event, severity=serious+ | device set to in_maintenance |
| calibration_certificate_required | record_calibration, no cert ref | PolicyViolationError |
| loan_requires_active_device | create_device_loan | PolicyViolationError if not active/available |
| loan_return_triggers_requalification | return_device_from_loan | device set to in_maintenance, requalification_required=True |
| audit_events_hmac_signed | publish_audit_event | HMAC-SHA256 signature attached to every event |

## Streaming Architecture

The capability emits events on NATS JetStream subjects processed by a Bytewax pipeline:

```
apg.healthcare.dev.lifecycle    — device, maintenance, calibration, adverse event lifecycle
apg.healthcare.dev.audit.<tid>  — tamper-evident audit log (7-year retention)
apg.healthcare.dev.shadow.<did> — device shadow delta events (IoT integration, planned)
```

Bytewax consumers downstream of `apg.healthcare.dev.lifecycle` power real-time dashboards, calibration overdue detection, and recall impact assessments without polling the service.

## Configuration Reference

All keys are tenant-scoped and set via the `conf` capability or env vars prefixed `HEALTHCARE_DEV_`.

| Key | Default | Description |
|-----|---------|-------------|
| devices.udi_required_for_class_ii_iii | true | Require UDI for Class II/III |
| calibration.certificate_required | true | Block calibration without cert ref |
| calibration.overdue_alert_days | 7 | Days before due to trigger overdue alert |
| adverse_events.fda_mdr_reporting_threshold | serious | Severity threshold for MDR warning |
| audit.hmac_key_env | AUDIT_HMAC_KEY | Env var for HMAC signing key |
| streaming.nats_url_env | NATS_URL | Env var for NATS JetStream URL |

## Composability

```apg
use healthcare_dev;
```

- Adverse events feed `healthcare_reg` for FDA MDR submission tracking
- Calibration records consumed by `healthcare_lab` instrument management
- Maintenance schedules integrate with `schd` for PM reminders
- Manufacturer scorecards compose with procurement capabilities for vendor CAPA
- Audit replay integrates with `comp` for regulatory submission evidence packages

## Further Reading

- `service.py` — Complete business logic implementation
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and schemas
- `capability_contract.py` — Policy rules and supported value sets
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 planned enhancements with justifications
