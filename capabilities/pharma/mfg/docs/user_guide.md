# Pharmaceutical Manufacturing — User Guide

**Capability ID**: `pharma_mfg` | **Domain**: `pharma` | **Version**: `1.1.0`

---

## Description

Manages pharmaceutical manufacturing operations from batch record creation through equipment qualification, yield management, deviation handling, line clearance, raw material management, and QP batch release. Enforces GMP compliance, electronic batch records, QP release signatures, and equipment qualification requirements at every production step.

Includes async methods for production scheduling, EBR step execution with dual-person integrity, environmental monitoring, CAPA lifecycle, material genealogy, SPC/CPV, and calibration management.

---

## Installation

```bash
pip install apg-pharma-mfg
```

---

## Quick Start

```python
from apg_pharma_mfg.service import PharmaceuticalManufacturingService
from apg_pharma_mfg.models import BatchRecordCreate

svc = PharmaceuticalManufacturingService(tenant_id="acme-pharma", actor_id="jane.doe")

# Register a production line
line = svc.register_line(
    tenant_id="acme-pharma",
    line_code="LINE-A1",
    name="Solid Oral Line A",
    manufacturing_type="solid_oral",
    created_by="jane.doe",
)
svc.clear_line(line.id, "acme-pharma", cleared_by="jane.doe")

# Create and start a batch
batch_create = BatchRecordCreate(
    tenant_id="acme-pharma",
    batch_number="BN-2026-001",
    product_id="PROD-PARACETAMOL-500",
    manufacturing_type="solid_oral",
    master_formula_reference="MFR-2026-001-v3",
    planned_quantity=100_000.0,
    unit_of_measure="tablets",
    created_by="jane.doe",
)
batch = svc.create_batch(batch_create)
batch = svc.start_batch(batch.id, "acme-pharma", line.id)
```

---

## Core Workflows

### Batch Record Management

```python
# Create → Start → Record Yield → Reconcile → Release
batch = svc.create_batch(payload)
batch = svc.start_batch(batch.id, tenant_id, line.id)

svc.record_yield(
    tenant_id=tenant_id,
    batch_id=batch.id,
    yield_type="granulation",
    step_name="wet_granulation",
    theoretical_quantity=105_000.0,
    actual_quantity=104_200.0,
    created_by="operator1",
)

svc.reconcile_batch_yield(batch.id, tenant_id)

batch = svc.release_batch(
    batch.id, tenant_id,
    qp_release_reference="QP-SIGN-2026-001",
    electronic_signature_reference="ESIG-123456",
)
```

### Equipment Qualification

```python
equip = svc.register_equipment(
    tenant_id=tenant_id,
    equipment_id="EQ-GRANULATOR-01",
    name="High-Shear Granulator 01",
    equipment_type="granulator",
    location="Room 102",
    created_by="eng.team",
    model="GHL-600",
    serial_number="SN-20240101",
)

# IQ → OQ → PQ lifecycle
for qtype in ("installation_qualification", "operational_qualification", "performance_qualification"):
    svc.qualify_equipment(
        equipment_id=equip.id,
        tenant_id=tenant_id,
        qualification_type=qtype,
        protocol_reference=f"PROT-{qtype.upper()}-001",
        report_reference=f"RPT-{qtype.upper()}-001",
        performed_by="qa.team",
    )
```

### Deviation Management

```python
dev = svc.raise_deviation(
    tenant_id=tenant_id,
    deviation_number="DEV-2026-042",
    deviation_type="process",
    severity="major",
    description="Granulation endpoint not achieved within specified time range.",
    raised_by="operator1",
    batch_id=batch.id,
    equipment_id=equip.id,
)

svc.close_deviation(
    deviation_id=dev.id,
    tenant_id=tenant_id,
    root_cause="Impeller speed too low due to incorrect parameter entry.",
    capa_reference="CAPA-2026-018",
)
```

### Raw Material Management

```python
from datetime import datetime, timedelta

mat = svc.receive_material(
    tenant_id=tenant_id,
    material_code="RM-PARACETAMOL-API",
    name="Paracetamol API",
    material_type="active_pharmaceutical_ingredient",
    vendor_id="VENDOR-CHEMCO-001",
    lot_number="LOT-20260101-A",
    quantity=500.0,
    unit_of_measure="kg",
    storage_condition="<25°C, 60% RH",
    vendor_qualified=True,
    created_by="warehouse.team",
    expiry_date=datetime.utcnow() + timedelta(days=730),
)

mat = svc.release_material(mat.id, tenant_id, qc_reference="QC-CERT-20260101-A")
```

---

## Async Methods

All async methods must be awaited. In a synchronous context use `asyncio.run()`.

### Production Scheduling

```python
import asyncio
from datetime import datetime, timezone

schedule = asyncio.run(svc.schedule_batch_production(
    batch_id=batch.id,
    line_id=line.id,
    start_dt=datetime(2026, 7, 1, 6, 0, tzinfo=timezone.utc),
    duration_hours=8.0,
    tenant_id=tenant_id,
    priority=3,
))
# schedule["status"] == "scheduled"
# Raises ValueError if the line has a conflicting booking
```

### Electronic Batch Record (EBR) Steps

Implements 21 CFR Part 11 dual-person integrity: the reviewer must differ from the operator.

```python
# Operator executes step
step = asyncio.run(svc.execute_ebr_step(
    batch_id=batch.id,
    step_number=1,
    operator_id="operator1",
    step_data={"weight_kg": 104.2, "moisture_pct": 2.1},
    tenant_id=tenant_id,
    step_name="Weighing and Dispensing",
))

# Reviewer verifies — must be different from operator
step = asyncio.run(svc.verify_ebr_step(
    batch_id=batch.id,
    step_number=1,
    reviewer_id="supervisor1",  # NOT operator1
    tenant_id=tenant_id,
    accepted=True,
    review_notes="Values within specification.",
))
# step["status"] == "completed"
```

### Environmental Monitoring

Out-of-limit samples automatically raise a GMP deviation.

```python
sample = asyncio.run(svc.record_environmental_sample(
    line_id=line.id,
    sample_point="HEPA-SUPPLY-001",
    parameter="particulates_0.5um",
    value=3_600.0,
    unit="particles/m³",
    limit_low=0.0,
    limit_high=3_520.0,
    sampled_by="em.technician",
    tenant_id=tenant_id,
))
# sample["out_of_limit"] == True
# sample["auto_deviation_id"] points to the auto-raised deviation
```

### CAPA Lifecycle

```python
capa = asyncio.run(svc.open_capa(
    deviation_id=dev.id,
    root_cause_category="process",
    actions=[
        {"description": "Retrain operators on parameter entry", "assignee_id": "training.mgr", "due_date": "2026-08-01"},
        {"description": "Add parameter range lock on SCADA", "assignee_id": "it.engineer", "due_date": "2026-09-01"},
    ],
    due_date=datetime(2026, 9, 30),
    owner_id="qa.manager",
    tenant_id=tenant_id,
))

capa = asyncio.run(svc.close_capa(
    capa_id=capa["id"],
    effectiveness_evidence="EFF-REVIEW-2026-018",
    tenant_id=tenant_id,
    closed_by="qa.manager",
))
```

### Material Genealogy

```python
# Dispense released material into batch
link = asyncio.run(svc.link_material_to_batch(
    material_id=mat.id,
    batch_id=batch.id,
    quantity_dispensed=95.5,
    dispense_reference="WEIGH-TICKET-2026-001",
    tenant_id=tenant_id,
    dispensed_by="operator1",
))

# Trace full genealogy for recall impact assessment
genealogy = asyncio.run(svc.trace_batch_genealogy(batch.id, tenant_id))
# genealogy["material_inputs"] — all raw lots
# genealogy["ebr_steps"] — ordered step records
# genealogy["deviations"] — linked deviations
```

### Statistical Process Control (SPC / CPV)

```python
spc = asyncio.run(svc.get_spc_data(
    product_id="PROD-PARACETAMOL-500",
    parameter="yield_pct",
    tenant_id=tenant_id,
    n_batches=30,
))
print(f"Mean: {spc['mean']}%, UCL: {spc['ucl']}%, LCL: {spc['lcl']}%")
print(f"WE Rule violations: {spc['violation_count']}")
print(f"Process capable: {spc['spc_capable']}")
```

### Calibration Management

```python
cal = asyncio.run(svc.record_calibration(
    equipment_id=equip.id,
    standard_reference="NIST-TRACEABLE-CERT-2026-0451",
    result="pass",
    calibrated_by="metrology.lab",
    next_due=datetime(2027, 7, 1),
    tenant_id=tenant_id,
    tolerance_pct=0.5,
    as_found=100.1,
    as_left=100.0,
))
# Equipment next_calibration_due and last_calibration_date are updated automatically
# Failed calibrations set equipment status to "out_of_service"
```

---

## GMP Compliance Check

```python
check = svc.gmp_compliance_check(
    facility_id="SITE-NAIROBI-01",
    period="2026-Q2",
    tenant_id=tenant_id,
    gmp_framework="eu_gmp",
    inspector_id="internal.auditor",
)
# check["compliant"] — True if score >= 70 and zero critical deviations
# check["compliance_score"] — 0–100 composite score
```

---

## Dashboard and Analytics

```python
# Live operations dashboard
summary = svc.dashboard_summary(tenant_id)

# Period KPIs
analytics = svc.batch_analytics("2026-Q2", tenant_id)
# analytics["right_first_time_pct"], analytics["average_yield_pct"],
# analytics["deviation_rate_per_batch"], analytics["deviations_by_type"]
```

---

## Configuration

All configuration keys are tenant-scoped. Override via the `conf` capability or environment variables prefixed `PHARMA_MFG_`.

| Key | Default | Description |
|-----|---------|-------------|
| `yield_management.yield_variance_threshold_pct` | `2.0` | Yield variance alert threshold |
| `yield_management.investigation_trigger_pct` | `5.0` | Variance requiring mandatory investigation |
| `equipment.requalification_trigger_months` | `12` | Equipment requalification cycle |
| `deviations.reporting_timeline_hours.critical` | `24` | Critical deviation reporting deadline |

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/pharma-mfg/dashboard` | `pharma_mfg:view` | Overview |
| `/pharma-mfg/batches` | `pharma_mfg:batches` | Production |
| `/pharma-mfg/batches/<id>` | `pharma_mfg:batches` | Production |
| `/pharma-mfg/batches/<id>/ebr` | `pharma_mfg:ebr` | Production |
| `/pharma-mfg/lines` | `pharma_mfg:lines` | Production |
| `/pharma-mfg/equipment` | `pharma_mfg:equipment` | Equipment |
| `/pharma-mfg/equipment/qualification` | `pharma_mfg:qualification` | Equipment |
| `/pharma-mfg/materials` | `pharma_mfg:materials` | Materials |

---

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| `batch_master_formula_required` | Batch created without master formula | Deny — attach master formula |
| `qp_release_required` | Batch released without QP signature | Deny — obtain QP signature |
| `equipment_qualification_required` | Equipment used without IQ/OQ/PQ | Deny — complete qualification |
| `line_clearance_required` | Batch started without line clearance | Deny — complete clearance |
| `deviation_investigation_required` | Deviation closed without root cause | Deny — complete investigation |
| `yield_reconciliation_required` | Batch closed without reconciliation | Deny — reconcile yield |
| `material_incoming_qc_required` | Material released without incoming QC | Deny — complete QC |
| `dual_person_ebr_required` | EBR step reviewed by same operator | Deny — assign different reviewer |
| `calibration_currency_required` | Equipment calibration overdue | Block use, emit audit event |
| `capa_effectiveness_required` | CAPA closed without effectiveness evidence | Deny — provide evidence |
| `material_released_before_dispense` | Material dispensed before QC release | Deny — release material first |

---

## Streaming Events

| Event | Trigger |
|-------|---------|
| `batch_started` | Batch moved to in_process |
| `batch_released` | QP signed off |
| `batch_rejected` | QP rejected batch |
| `equipment_qualified` | IQ/OQ/PQ completed |
| `equipment_calibration_failed` | Calibration result = fail |
| `deviation_raised` | New deviation created |
| `deviation_closed` | Investigation completed |
| `capa_opened` | CAPA record created |
| `capa_closed` | CAPA effectiveness confirmed |
| `yield_variance_exceeded` | Variance > 5% |
| `environmental_sample_recorded` | EM data point captured |
| `batch_production_scheduled` | Production slot booked |
| `ebr_step_executed` | Operator completed EBR step |
| `ebr_step_verified` | Reviewer approved EBR step |
| `material_dispensed_to_batch` | Raw lot linked to batch |
| `batch_genealogy_traced` | Traceability graph built |
| `spc_data_generated` | SPC/CPV calculation run |
| `calibration_recorded` | Calibration event logged |
| `gmp_compliance_check_completed` | Facility compliance assessed |

---

## Composability

| Downstream Capability | Integration |
|-----------------------|-------------|
| `pharma_dis` | Released batches trigger dispatch serialisation |
| `pharma_qms` | Deviations feed CAPA; equipment quals cited in validation lifecycle |
| `pharma_qms` CPV | SPC data feeds continuous process verification |
| `schd` | Calibration and requalification due dates fed to maintenance scheduler |
| `mqeb` | All lifecycle events streamed on `apg.pharma.mfg.lifecycle` |

---

## Edge Cases

- Equipment with expired calibration blocks `use_equipment()` even if IQ/OQ/PQ is current.
- Yield variance above 5% triggers mandatory investigation flag regardless of batch status.
- QP release requires both `qp_release_reference` **and** `electronic_signature_reference`; one is insufficient.
- Line cleaning status resets on batch start and must be re-verified before the next batch.
- Critical deviations trigger 24-hour reporting even when the batch is subsequently rejected.
- Environmental OOL samples auto-raise a `major` deviation linked to the production line.
- EBR steps are immutable once `completed`; re-execution raises `ValueError` (21 CFR Part 11).
- Material dispensing validates stock sufficiency and deducts quantity atomically.

---

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder view models
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 improvements roadmap
- `README.md` — Quick reference
