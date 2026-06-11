# Mine Production Operations — User Guide

**Capability ID**: `mining_pro` | **Domain**: `mining` | **Version**: `1.1.0`
**Copyright**: © 2025 Datacraft | **Contact**: nyimbi@gmail.com

---

## Overview

`mining_pro` is the central operational ledger for open-pit and underground mine production. It covers everything from blast design through ore delivery, giving mine managers, supervisors, and planners a single source of truth for daily, weekly, and monthly production performance.

Version 1.1 adds real-time truck dispatch, blast vibration compliance, block model grade reconciliation, Short-Interval Control (SIC), automated shift handovers, SMRP-compliant equipment availability, explosives reconciliation, and delay Pareto analysis.

---

## Installation

```bash
pip install apg-mining-pro
```

Or in a `pyproject.toml`:

```toml
dependencies = ["apg-mining-pro>=1.1.0"]
```

---

## Quick Start

```python
import asyncio
from capabilities.mining.pro.service import ProService

async def main():
    svc = ProService(tenant_id="site_alpha")

    # 1. Create a shift report
    from capabilities.mining.pro.models import (
        ShiftReportCreate, ShiftType, ProductionActivityCreate,
        MaterialType, OreTrackingMethod, DelayCreate
    )
    from datetime import datetime, timedelta

    now = datetime.utcnow()
    report = await svc.create_shift_report(
        ShiftReportCreate(
            tenant_id="site_alpha",
            shift_type=ShiftType.DAY,
            shift_date=now - timedelta(hours=1),
            shift_start=now - timedelta(hours=13),
            shift_end=now - timedelta(hours=1),
            mine_area="North Pit",
            supervisor_id="sup_001",
            operator_count=24,
            activities=[
                ProductionActivityCreate(
                    area="Bench 4020",
                    material_type=MaterialType.ORE,
                    planned_tonnes=5000.0,
                    actual_tonnes=4750.0,
                    grade_value=1.8,
                    grade_units="g/t Au",
                    tracking_method=OreTrackingMethod.WEIGHBRIDGE,
                )
            ],
            delays=[
                DelayCreate(
                    delay_category="equipment_breakdown",
                    duration_minutes=45.0,
                    equipment_id="CAT_793_07",
                    description="Right rear tyre puncture",
                )
            ],
        ),
        created_by="sup_001",
    )
    print(f"Shift report created: {report.id}")

asyncio.run(main())
```

---

## Core Workflows

### 1. Shift Report Lifecycle

Shift reports follow a strict status progression: `DRAFT → SUBMITTED → APPROVED`.

```python
# Submit for supervisor sign-off
submitted = await svc.submit_shift_report(report.id, supervisor_id="sup_002")

# Approve (mine manager or reviewer)
approved = await svc.approve_shift_report(report.id, reviewer_id="mgr_001")

# List reports for a date range and mine area
reports = await svc.list_shift_reports(
    mine_area="North Pit",
    date_from=datetime(2026, 6, 1),
    date_to=datetime(2026, 6, 30),
    status="approved",
)
```

**Rule**: Approved reports are immutable. Any modification attempt returns a `ValueError`.

---

### 2. Blast Management

Blasts move through a strict 8-stage state machine:
`planned → designed → drilled → charged → primed → fired → cleared → mucked`

Skipping any stage raises `ValueError`.

```python
from capabilities.mining.pro.models import BlastCreate, BlastType, MaterialType, BlastHoleCreate

blast = await svc.create_blast(
    BlastCreate(
        tenant_id="site_alpha",
        blast_name="NP-B4020-001",
        blast_type=BlastType.PRODUCTION,
        mine_area="North Pit",
        bench_level="4020",
        planned_date=datetime(2026, 6, 15, 7, 0),
        planned_tonnes=18000.0,
        planned_material_type=MaterialType.ORE,
        holes=[
            BlastHoleCreate(
                hole_id="H001",
                easting=412500.0,
                northing=8920000.0,
                elevation_m=4020.0,
                depth_m=10.5,
                diameter_mm=165.0,
                explosive_type="ANFO",
                explosive_mass_kg=95.0,
                stemming_m=2.5,
            )
        ],
        explosive_total_kg=4750.0,
        designer_id="eng_blast_01",
    ),
    created_by="eng_blast_01",
)

# Advance through state machine
await svc.update_blast(blast.id, BlastUpdate(status=BlastStatus.DESIGNED))
await svc.approve_blast_design(blast.id, approver_id="eng_senior_01")
await svc.update_blast(blast.id, BlastUpdate(status=BlastStatus.DRILLED))
await svc.update_blast(blast.id, BlastUpdate(status=BlastStatus.CHARGED))
await svc.update_blast(blast.id, BlastUpdate(status=BlastStatus.PRIMED))
await svc.fire_blast(blast.id, fire_authority_id="fm_001")
```

---

### 3. Blast Vibration Compliance Monitoring (NEW in v1.1)

Record PPV measurements from vibration monitors at sensitive receivers. The service automatically checks against configured limits and emits a `blast_vibration_breach` NATS event on breach.

```python
vibration = await svc.record_blast_vibration(
    blast_id=blast.id,
    sensor_id="SENSOR_RS_01",
    ppv_mmps=3.8,
    distance_m=450.0,
    receiver_type="residential",    # residential | industrial | infrastructure | heritage
    ppv_limit_mmps=5.0,             # from conf capability; default 5 mm/s residential
    recorded_by="blasting_tech_01",
)
print(f"Breach: {vibration['breach']}")  # False — within limit
```

---

### 4. Grade Control

```python
from capabilities.mining.pro.models import GradeBoundaryCreate, GradeControlMethod

# Create and approve a cut-off grade boundary
boundary = await svc.create_grade_boundary(
    GradeBoundaryCreate(
        tenant_id="site_alpha",
        mine_area="North Pit",
        period_start=datetime(2026, 6, 1),
        period_end=datetime(2026, 6, 30),
        method=GradeControlMethod.BLAST_HOLE_ASSAY,
        commodity="Au",
        cut_off_grade=0.5,
        grade_units="g/t",
        ore_boundary_coords=[
            {"easting": 412480.0, "northing": 8919990.0},
            {"easting": 412580.0, "northing": 8919990.0},
            {"easting": 412580.0, "northing": 8920050.0},
        ],
    ),
    created_by="geo_001",
)
await svc.approve_grade_boundary(boundary.id, approver_id="chief_geo_01")

# Look up active boundary at the current time
active = await svc.get_active_grade_boundary(mine_area="North Pit", commodity="Au")
```

---

### 5. Block Model Grade Reconciliation (NEW in v1.1)

Compare mined actuals to the geological block model. F-factor outside ±10% triggers a warning and NATS `reconciliation_variance_alert`.

```python
recon = await svc.reconcile_block_model(
    block_id="BLK_NP_4020_042",
    block_model_grade=1.95,     # g/t from resource model
    block_model_tonnes=17500.0,
    period="2026-06",
    section="North Pit",
)
print(f"F-factor: {recon['f_factor']}")        # e.g. 0.97 — within tolerance
print(f"Variance alert: {recon['variance_alert']}")  # False
```

**Interpretation**:
| Factor | Meaning |
|--------|---------|
| F-factor ~1.0 | Good reconciliation |
| F-factor < 0.90 | Metal losses; review ore loss / dilution |
| F-factor > 1.10 | Excess metal; check block model grade inflation |

---

### 6. Grade Control Sampling and Dilution

```python
# Record face samples
await svc.grade_control_sample(
    sample_id="GC-NP-4020-001",
    location={"easting": 412520.0, "northing": 8920010.0, "elevation": 4020.0},
    grade=1.75,
    classification="ore",
    element="Au",
    grade_unit="g/t",
    blast_block_id="BLK_NP_4020_042",
    sampled_by="geo_sampler_01",
)

# Calculate block dilution
dilution = await svc.dilution_calculation("BLK_NP_4020_042")
print(f"Dilution: {dilution['dilution_pct']}%")
print(f"Diluted grade: {dilution['diluted_grade']} g/t")
```

---

### 7. Ore Movement Tracking

```python
movement = await svc.ore_movement(
    from_location="North Pit Bench 4020",
    to_location="ROM Pad 1",
    tonnes=48.5,
    grade=1.82,
    truck_id="CAT_793_07",
    material_type="ore",
    grade_element="Au",
    grade_unit="g/t",
    recorded_by="weighbridge_01",
)
# Contained metal (oz equivalent) is auto-calculated
print(f"Contained metal: {movement['contained_metal']} kg Au")
```

---

### 8. Real-Time Truck Dispatch (NEW in v1.1)

Assign a truck to a destination with priority. Publishes to NATS subject `mining.pro.dispatch.{mine_area}` for consumption by onboard terminal units.

```python
dispatch = await svc.dispatch_truck(
    truck_id="CAT_793_07",
    destination="ROM Pad 1",
    load_tonnes=220.0,
    mine_area="North Pit",
    priority=2,          # 1 = highest priority
    assigned_by="dispatcher_01",
)
print(f"NATS subject: {dispatch['nats_subject']}")
# Raises ValueError if truck already has an active in_transit assignment
```

---

### 9. Stockpile Management

```python
from capabilities.mining.pro.models import StockpileCreate, StockpileType, StockpileMovementCreate, MaterialType

stockpile = await svc.create_stockpile(
    StockpileCreate(
        tenant_id="site_alpha",
        name="ROM Pad 1",
        stockpile_type=StockpileType.RUN_OF_MINE,
        mine_area="North Pit",
        capacity_tonnes=150000.0,
    ),
    created_by="ops_mgr_01",
)

# Add ore
await svc.record_stockpile_movement(
    StockpileMovementCreate(
        stockpile_id=stockpile.id,
        movement_type="add",
        tonnes=5000.0,
        material_type=MaterialType.ORE,
        grade_value=1.8,
        grade_units="g/t",
        source_area="North Pit Bench 4020",
        movement_at=datetime.utcnow(),
        operator_id="weighbridge_01",
    ),
    created_by="weighbridge_01",
)
```

---

### 10. Short-Interval Control (NEW in v1.1)

SIC reports capture 2–4 hour production windows and compare against the published schedule's hourly disaggregation. Critical variance (>15% below target) emits `sic.variance.critical` to NATS.

```python
sic = await svc.short_interval_report(
    section="North Pit",
    interval_start=datetime(2026, 6, 11, 6, 0),
    interval_end=datetime(2026, 6, 11, 10, 0),
    actual_tonnes=1850.0,
    actual_metres=12.0,
    supervisor_id="sup_001",
    comments="Drill rig relocation caused 30 min delay",
)
print(f"Variance: {sic['variance_pct']}%")
print(f"Critical: {sic['critical_variance']}")
```

---

### 11. Automated Shift Handover (NEW in v1.1)

Generate a structured handover package for the incoming supervisor. Assembles blast holds, safety holds, pending grade approvals, and stockpile levels in a single call.

```python
handover = await svc.generate_shift_handover(
    outgoing_shift_id=report.id,
    incoming_supervisor_id="sup_002",
)
print(f"Action items: {handover['action_items_count']}")
print(f"Open blast holds: {handover['open_blast_holds']}")
print(f"Stockpile snapshot: {handover['stockpile_snapshot']}")
# Published to NATS subject: mining.pro.handover
```

---

### 12. Equipment Availability Report (NEW in v1.1)

Compute Physical Availability (PA), Mechanical Availability (MA), and Utilisation (U) per SMRP definitions. Flags equipment below the 85% PA JORC threshold.

```python
avail = await svc.equipment_availability_report(
    equipment_id="CAT_793_07",
    period="2026-06",
)
print(f"PA: {avail['physical_availability_pct']}%")
print(f"MA: {avail['mechanical_availability_pct']}%")
print(f"U:  {avail['utilisation_pct']}%")
print(f"Meets JORC 85% PA target: {avail['pa_meets_target']}")
```

---

### 13. Delay Pareto Analysis (NEW in v1.1)

Identify the vital few delay categories driving 80% of lost production time, with automatic escalation routing to the responsible capability.

```python
pareto = await svc.delay_pareto_analysis(period="2026-06", section="North Pit")
for row in pareto["pareto"]:
    print(
        f"{row['category']:30s}  {row['delay_minutes']:7.1f} min  "
        f"{row['pct_share']:5.1f}%  cumulative {row['cumulative_pct']:5.1f}%  "
        f"→ {row['escalation_capability']}"
    )
```

Sample output:
```
equipment_breakdown            315.0 min   42.0%  cumulative  42.0%  → mining_eqp
blast_hold                     180.0 min   24.0%  cumulative  66.0%  → mining_pro
safety_hold                    105.0 min   14.0%  cumulative  80.0%  → mining_saf
```

---

### 14. Explosives Reconciliation (NEW in v1.1)

Compare magazine issues to blast plan consumption. Variances > 2 kg per explosive type are flagged as compliance breaches.

```python
recon = await svc.reconcile_explosives(
    period="2026-06",
    magazine_id="MAG_01",
    magazine_issues={
        "ANFO_kg": 14250.0,
        "booster_kg": 285.0,
        "detonators": 570.0,
    },
)
print(f"All compliant: {recon['all_compliant']}")
for exp_type, detail in recon["explosive_variances"].items():
    if detail["compliance_breach"]:
        print(f"BREACH: {exp_type} variance = {detail['variance_kg']} kg")
```

---

### 15. Production Reporting

```python
# Monthly production report
report = await svc.monthly_production_report(mine_id="mine_alpha", period="2026-06")
print(f"Blast plans executed: {report['blast_plans_executed']}")
print(f"Misfires: {report['blast_misfires']}")
print(f"Ore movements: {report['ore_movements_count']}")

# Target vs actual by section
tva = await svc.production_target_vs_actual(period="2026-06", section="North Pit")
print(f"Achievement: {tva['tonnes_achievement_pct']}%")

# Full analytics
analytics = await svc.production_analytics(period="2026-06")
print(f"Strip ratio: {analytics['strip_ratio']}")
print(f"Average grade: {analytics['average_ore_grade']} g/t")
```

---

## NATS Integration

All real-time events publish to NATS subjects. Connect a Bytewax pipeline to aggregate rolling KPIs:

```python
import nats

async def subscribe_production_events():
    nc = await nats.connect("nats://localhost:4222")
    # Truck dispatch
    await nc.subscribe("mining.pro.dispatch.*", cb=handle_dispatch)
    # SIC critical variance
    await nc.subscribe("sic.variance.critical", cb=handle_sic_critical)
    # Blast vibration breach
    await nc.subscribe("blast_vibration_breach", cb=handle_vibration_breach)
    # Shift handover
    await nc.subscribe("mining.pro.handover", cb=handle_handover)
```

---

## Permissions Reference

| Permission | Description |
|---|---|
| `mining_pro:view` | Read shift reports, blasts, stockpiles, schedules, analytics |
| `mining_pro:write` | Create/update shift reports, ore movements, SIC reports, dispatch |
| `mining_pro:blast_design` | Create/approve blast designs, record vibration, fire blasts |
| `mining_pro:grade_control` | Create/approve grade boundaries, reconciliation |
| `mining_pro:schedule` | Create/approve/publish production schedules |
| `mining_pro:dispatch` | Assign truck dispatches |
| `mining_pro:compliance` | Explosives reconciliation, compliance reporting |

---

## Error Reference

| Exception | Cause |
|---|---|
| `ValueError: Cannot create shift reports for future shifts` | shift_date in the future |
| `ValueError: Cannot modify an approved shift report` | Attempt to edit approved report |
| `ValueError: Invalid blast status transition: X -> Y` | Blast state machine violation |
| `PermissionError: Blast design must be approved before firing` | Missing design approval |
| `ValueError: Cannot reclaim Xt; only Yt available` | Stockpile reclaim exceeds inventory |
| `ValueError: Shift 'X' already reported` | Duplicate shift_id for tenant |
| `ValueError: Truck 'X' already has an active dispatch` | Duplicate truck assignment |
| `KeyError: Blast plan 'X' not found` | Invalid plan_id reference |
| `AssertionError: Cross-tenant access denied` | tenant_id mismatch |

---

## Composability Reference

```apg
use mining_pro;          // Mine production operations
use mining_saf;          // Safety monitoring (blast events feed here)
use mining_ore;          // Plant feed (stockpile movements feed here)
use mining_exp;          // Exploration (grade boundaries derived from here)
use mining_eqp;          // Equipment (delay records cross-reference here)
use schd;                // Scheduling (production schedules consumed here)
use moni;                // Monitoring (SIC alerts and dispatch status)
use ntfy;                // Notifications (blast authority, PPV breach, MCF alert)
```

---

## Further Reading

- `service.py` — Complete business logic (ProService class, 40+ async methods)
- `models.py` — Pydantic v2 data models and enums
- `api.py` — REST API endpoints (Flask-AppBuilder blueprints)
- `views.py` — Flask-AppBuilder views and schemas
- `README.md` — Quick reference and API route index
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 improvement specifications with competitor benchmarks
- `tests/test_service.py` — Service unit tests
- `tests/test_contract.py` — Capability contract tests
