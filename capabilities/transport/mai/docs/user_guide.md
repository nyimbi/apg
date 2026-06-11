# Vehicle Maintenance — User Guide

**Capability ID**: `transport_mai` | **Domain**: `transport` | **Version**: `2.0.0`

---

## Description

The Vehicle Maintenance capability manages preventive and corrective maintenance job scheduling, workshop bay allocation, parts inventory and ordering, warranty tracking, vehicle inspections with digital signature capture, and roadworthiness certificate management. It enforces pre-dispatch safety checks and blocks operation of expired-MOT or unsafe vehicles.

Version 2.0 adds world-class production enhancements: odometer-linked predictive alerts, breakdown SLA tracking, auto-reorder parts workflows, technician workload balancing, defect resolution with root-cause classification, compliance calendar, fleet TCO reporting, warranty claim filing, labour utilisation analytics, and supplier scorecards.

---

## Installation

```bash
pip install apg-transport-mai
```

---

## Quick Start

```python
from capabilities.transport.mai.service import VehicleMaintenanceService

svc = VehicleMaintenanceService(tenant_id="acme", actor_id="fleet_manager")

# Create a maintenance job
job = svc.create_job(
    "JOB-001", "acme", "VEH-100", "preventive", "medium",
    "TECH-01", "in_house", 3.0, "JC-2026-001",
)

# Check vehicle health
health = await svc.vehicle_health_score("VEH-100")
```

---

## Core Workflows

### 1. Preventive Maintenance Scheduling

Schedule a service by date and odometer trigger:

```python
result = await svc.schedule_service(
    vehicle_id="VEH-100",
    service_type="preventive",
    due_date="2026-09-01",
    due_km=85000,
    technician_id="TECH-02",
    priority="medium",
)
# Returns: {"schedule": {...}, "job": {...}, "due_date": ..., "due_km": ...}
```

Record actual odometer readings to keep predictive alerts accurate:

```python
await svc.record_odometer_reading("VEH-100", km=82340.5)
```

Trigger predictive alerts based on current odometer:

```python
alerts = await svc.predictive_maintenance_alert("VEH-100", current_odometer_km=82340.5)
# Returns: {"alerts": [{"service_type": "oil_change", "overdue": True, ...}], ...}
```

---

### 2. Breakdown Management with SLA Tracking

Log a breakdown event — automatically creates a critical-priority job and starts the SLA clock:

```python
breakdown = await svc.log_breakdown_event(
    vehicle_id="VEH-100",
    location="-1.2864,36.8172",
    breakdown_type="mechanical",
    sla_minutes=120,
    reported_by="DRIVER-07",
    description="Engine overheating, vehicle immobilised",
)
# auto_job is created immediately
```

Check for SLA breaches fleet-wide:

```python
breaches = await svc.check_sla_breaches()
# Returns: {"sla_breach_count": 2, "breaches": [...], ...}
```

---

### 3. Defect Logging and Resolution

Log a defect (auto-creates a job for critical/high severity):

```python
defect = await svc.log_defect(
    vehicle_id="VEH-100",
    defect_type="brake_fade",
    severity="high",
    reported_by="TECH-01",
    description="Brake pedal soft under heavy braking",
)
```

After repair, close the defect with root cause:

```python
resolved = await svc.resolve_defect(
    defect_id=defect["defect_id"],
    resolution_notes="Replaced brake fluid and bled system",
    root_cause_category="wear_and_tear",
    resolved_by="TECH-01",
    closing_job_id="JOB-002",
)
```

Detect systemic issues from recurring root causes:

```python
report = await svc.defect_recurrence_report("VEH-100")
# Returns: {"systemic_concerns": ["wear_and_tear"], "has_systemic_issues": True, ...}
```

---

### 4. Parts Inventory and Auto-Reorder

Check current inventory status for a part:

```python
inventory = await svc.parts_inventory_check("OIL-FILTER-5W30")
```

Set a minimum stock threshold to enable auto-reorder:

```python
await svc.set_parts_reorder_threshold(
    part_number="OIL-FILTER-5W30",
    min_qty=10,
    reorder_qty=50,
    supplier_id="SUP-AUTOPARTS-KE",
    parts_category="engine",
)
```

Trigger auto-reorder for all parts below threshold (run on a schedule):

```python
reorders = await svc.trigger_reorder_if_low()
# Returns: {"reorders_triggered": 2, "orders": [...], ...}
```

Record receipt and quality outcome when parts arrive:

```python
receipt = await svc.record_parts_receipt(
    order_id="PO-001",
    received_qty=48,
    quality_ok=True,
)
```

Get supplier performance scorecard:

```python
scorecard = await svc.supplier_scorecard("SUP-AUTOPARTS-KE", "2026-Q2")
# Returns: {"composite_score": 87.5, "rating": "good", "on_time_delivery_pct": 91.0, ...}
```

---

### 5. Warranty Management

Record a warranty:

```python
warranty = svc.record_warranty(
    "WRN-001", "acme", "VEH-100", "manufacturer",
    "Toyota Kenya", "2024-01-01", "2027-01-01",
)
```

File a structured warranty claim:

```python
claim = await svc.file_warranty_claim(
    warranty_id="WRN-001",
    job_id="JOB-003",
    defect_description="Gearbox bearing noise under load",
    evidence_refs=["photo://evidence/VEH100-gear-noise.jpg", "video://evidence/VEH100-noise.mp4"],
)
# Returns: {"claim_id": "WCL-...", "claim_status": "submitted", ...}
```

Check expiring warranties (within 90 days):

```python
expiring = await svc.warranty_expiry_check()
```

---

### 6. Inspections and Roadworthiness

Conduct a full roadworthiness check (inspects, checks defects, issues certificate if passed):

```python
result = await svc.roadworthiness_check(
    vehicle_id="VEH-100",
    inspector_id="INSP-KE-001",
    standard="ncop_kenya",
)
# Returns: {"passed": True, "certificate": {...}, "blocking_issues": [], ...}
```

**Note**: A vehicle with unresolved critical/high defects will fail this check. Call `resolve_defect()` first.

---

### 7. Compliance Calendar

Get a unified view of all upcoming maintenance and compliance deadlines:

```python
calendar = await svc.get_compliance_calendar(days_ahead=30)
# Returns: {"overdue": [...], "upcoming": [...], "overdue_count": 2, ...}
```

---

### 8. Fleet Analytics

Vehicle health score (0–100):

```python
health = await svc.vehicle_health_score("VEH-100")
# Returns: {"health_score": 75, "health_status": "fair", ...}
```

Fleet-wide health overview:

```python
overview = await svc.fleet_health_overview()
# Returns: {"avg_health_score": 82.3, "fleet_status": "good", ...}
```

Fleet Total Cost of Ownership report with replacement candidates:

```python
tco = await svc.fleet_tco_report(
    period="2026-H1",
    replacement_threshold_usd=8000.0,
)
# Returns: {"fleet_total_cost_usd": 45200.0, "replacement_candidates": [...], ...}
```

Technician workload:

```python
workload = await svc.get_technician_workload("TECH-01")
# Returns: {"open_job_count": 4, "backlog_hours": 18.5, "utilisation_status": "medium", ...}
```

Labour utilisation report:

```python
util = await svc.labour_utilisation_report("TECH-01", "2026-Q2")
# Returns: {"efficiency_pct": 94.2, "utilisation_status": "medium", ...}
```

Maintenance cost per km:

```python
cpk = await svc.cost_per_km("VEH-100", "2026-Q2", total_km=15000.0)
# Returns: {"cost_per_km_usd": 0.0312, "total_maintenance_cost_usd": 468.0, ...}
```

---

### 9. Work Orders (Multi-Defect Jobs)

Create a work order consolidating several defects:

```python
wo = await svc.create_work_order(
    vehicle_id="VEH-100",
    defects=["brake_fade", "oil_leak", "wiper_failure"],
    assigned_to="TECH-02",
    workshop_type="in_house",
    priority="high",
)
```

Complete a work order with parts used and labour:

```python
completed = await svc.complete_work_order(
    work_order_id=wo["work_order_id"],
    parts_used=[
        {"part_number": "BRAKE-PAD-F", "description": "Front brake pads", "quantity": 4, "unit_cost": 22.50},
        {"part_number": "OIL-SEAL-PCV", "description": "PCV oil seal", "quantity": 1, "unit_cost": 8.00},
    ],
    labour_hours=5.5,
    cost=175.0,
)
```

---

## Configuration Reference

All keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TRANSPORT_MAI_`.

| Key | Description | Default |
|-----|-------------|---------|
| `roadworthiness.fail_dispatch_on_expired` | Block dispatch if MOT expired | `true` |
| `inspections.digital_signature_required` | Require digital signature | `true` |
| `parts.reorder_alerts_enabled` | Enable low-stock alerts | `true` |
| `workshop.technician_skill_matching` | Match skills to jobs | `true` |

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/transport-maintenance/dashboard` | `transport_mai:view` | Overview |
| `/transport-maintenance/jobs` | `transport_mai:jobs` | Jobs |
| `/transport-maintenance/jobs/create` | `transport_mai:jobs_write` | Jobs |
| `/transport-maintenance/workshop` | `transport_mai:workshop` | Workshop |
| `/transport-maintenance/parts` | `transport_mai:parts` | Parts |
| `/transport-maintenance/warranty` | `transport_mai:warranty` | Warranty |
| `/transport-maintenance/inspections` | `transport_mai:inspections` | Compliance |
| `/transport-maintenance/roadworthiness` | `transport_mai:compliance` | Compliance |
| `/transport-maintenance/schedules` | `transport_mai:schedules` | Planning |
| `/transport-maintenance/reports` | `transport_mai:reports` | Reporting |
| `/transport-maintenance/agents` | `transport_mai:admin` | Automation |
| `/transport-maintenance/settings` | `transport_mai:admin` | Administration |

---

## Interoperability

```apg
use transport_mai;
```

- Receives vehicle IDs from `transport_fle`
- Maintenance schedules feed into `transport_sch` for planned downtime
- Parts reorder notifications route through `ntfy`
- Roadworthiness certificates validated by `transport_dis` pre-dispatch
- Warranty claims route through `wflo` for approval
- Breakdown events can trigger `transport_dis` fleet reassignment

---

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Rule engine and supported constants
- `WORLD_CLASS_IMPROVEMENTS.md` — Enhancement rationale and design notes
- `README.md` — Quick reference
