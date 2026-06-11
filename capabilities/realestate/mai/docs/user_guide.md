# Facilities Maintenance — User Guide

**Capability ID**: `realestate_mai` | **Domain**: `realestate` | **Version**: `1.1.0`
**Copyright**: © 2025 Datacraft | www.datacraft.co.ke

---

## Overview

`realestate_mai` provides full CAFM-grade maintenance management for real estate portfolios: structured asset registers with lifecycle tracking, preventive maintenance (PPM) scheduling at 9 frequency tiers, corrective and emergency work orders with SLA deadline enforcement, contractor management with insurance and performance controls, statutory inspection and compliance certificate tracking, defect management, and portfolio-level analytics.

---

## Installation

```bash
pip install apg-realestate-mai
```

---

## Quick Start

```python
from capabilities.realestate.mai.service import MaiService
from capabilities.realestate.mai.models import (
    AssetCreate, AssetCategory, PpmScheduleCreate, WorkOrderCreate, WorkOrderType, Priority,
)
from datetime import date, timedelta
from decimal import Decimal

svc = MaiService(tenant_id="tenant-1", actor_id="facilities_manager")

# 1. Register an asset
asset = await svc.register_asset(AssetCreate(
    tenant_id="tenant-1",
    property_id="prop-1",
    asset_ref="HVAC-01",
    name="Roof AHU Unit 1",
    category=AssetCategory.hvac,
    install_date=date(2020, 3, 1),
    replacement_cost=Decimal("450000"),
    created_by="admin",
))

# 2. Schedule preventive maintenance
ppm = await svc.create_ppm_schedule(PpmScheduleCreate(
    tenant_id="tenant-1",
    asset_id=asset.id,
    property_id="prop-1",
    title="Quarterly HVAC service",
    frequency="quarterly",
    next_due=date.today() + timedelta(days=7),
    estimated_cost=Decimal("12000"),
    created_by="admin",
))

# 3. Raise a corrective work order
wo = await svc.raise_work_order(WorkOrderCreate(
    tenant_id="tenant-1",
    asset_id=asset.id,
    property_id="prop-1",
    work_order_type=WorkOrderType.corrective,
    priority=Priority.p2_high,
    title="Cooling failure — 2nd floor",
    description="AHU not reaching set-point. Temperature alarm active.",
    reported_by="operations",
    created_by="facilities_manager",
))
print(wo.ref, wo.sla_resolution_deadline)
```

---

## Core Workflows

### Asset Management

```python
# List all HVAC assets for a property
hvac_assets = await svc.list_assets("tenant-1", property_id="prop-1", category="hvac")

# Get end-of-life assets requiring capital planning
eol = await svc.get_end_of_life_assets("tenant-1")

# Update asset status after decommission
from capabilities.realestate.mai.models import AssetUpdate, AssetStatus, LifecyclePhase
await svc.update_asset(asset.id, "tenant-1", AssetUpdate(
    status=AssetStatus.decommissioned,
    lifecycle_phase=LifecyclePhase.decommissioned,
))
```

### Condition Scoring

Assign a numeric condition score (0–100) after any inspection or maintenance event. Scores below 40 automatically advance the lifecycle phase to `ageing`; below 20 to `end_of_life`.

```python
score = await svc.update_asset_condition_score(
    asset_id=asset.id,
    tenant_id="tenant-1",
    score=35,
    assessed_by="inspector-jane",
    notes="Bearings showing wear. Refrigerant low.",
)

# Get all assets needing attention
at_risk = await svc.get_assets_below_condition_threshold("tenant-1", threshold=40, property_id="prop-1")
```

### PPM Schedules

```python
# Complete a PPM — auto-calculates next due date
completed = await svc.complete_ppm(ppm.id, "tenant-1", completed_by="contractor-1")
print(completed.next_due)  # advanced by frequency

# Check overdue PPMs portfolio-wide
overdue = await svc.get_overdue_ppms("tenant-1")

# Run a full preventive maintenance cycle and auto-raise work orders
result = await svc.preventive_maintenance_run(
    period="2026-Q1",
    property_ids=["prop-1", "prop-2"],
    tenant_id="tenant-1",
    auto_raise_work_orders=True,
)
print(result["work_orders_raised"], "WOs raised")
```

### Work Orders

```python
# Assign a contractor
response = await svc.assign_contractor(
    work_order_id=wo.id,
    contractor_id="contractor-1",
    agreed_cost=Decimal("8500"),
    start_date=date.today() + timedelta(days=1),
    tenant_id="tenant-1",
    purchase_order_ref="PO-2026-0042",
)

# Field technician check-in / check-out (GPS-stamped)
await svc.checkin_work_order(wo.id, "tenant-1", "tech-007", latitude=-1.286, longitude=36.817)
# ... technician carries out the work ...
await svc.checkout_work_order(wo.id, "tenant-1", "tech-007", "Refrigerant topped up, unit operational.")

# Complete and sign off
await svc.complete_work_order(
    work_order_id=wo.id,
    actual_cost=Decimal("7800"),
    completion_date=date.today(),
    sign_off_by="facilities_manager",
    tenant_id="tenant-1",
)

# Close (requires verification_complete=True)
await svc.close_work_order(wo.id, "tenant-1", verified_by="property_director")
```

### SLA Monitoring

```python
# Dashboard — open WOs, breaches, overdue PPMs
dashboard = await svc.get_sla_dashboard("tenant-1")

# WOs approaching SLA deadline (>= 75% elapsed)
at_risk_wos = await svc.get_work_orders_near_sla_breach("tenant-1", warning_pct=75.0)
for wo_entry in at_risk_wos:
    print(f"{wo_entry['ref']}: {wo_entry['elapsed_pct']}% elapsed, {wo_entry['minutes_remaining']} min left")

# Full SLA compliance report by priority and contractor
compliance = await svc.service_level_compliance(period="2026-Q1", tenant_id="tenant-1")
```

### Contractor Management

```python
from capabilities.realestate.mai.models import MaintenanceContractorCreate
from datetime import date

contractor = await svc.register_contractor(MaintenanceContractorCreate(
    tenant_id="tenant-1",
    name="CoolAir Services Ltd",
    contractor_type="hvac",
    email="ops@coolair.co.ke",
    phone="+254700000000",
    insurance_expiry=date(2027, 6, 30),
    insurance_policy_ref="INS-2026-00321",
    specialisms=["hvac", "refrigeration"],
    created_by="admin",
))

# Rolling 90-day performance scorecard
scorecard = await svc.compute_contractor_scorecard(contractor.id, "tenant-1", rolling_days=90)
print(scorecard["first_time_fix_rate_pct"], "% FTFR")

# Full portfolio contractor ranking
league = await svc.get_contractor_league_table("tenant-1")
for rank_entry in league[:3]:
    print(rank_entry["rank"], rank_entry["name"], rank_entry["composite_score"])
```

### Statutory Compliance Certificates

```python
cert = await svc.register_compliance_certificate(
    tenant_id="tenant-1",
    property_id="prop-1",
    certificate_type="gas_safety",
    issuing_authority="Gas Safe Register",
    certificate_ref="GS-2026-88321",
    issue_date=date(2026, 1, 15),
    expiry_date=date(2027, 1, 14),  # auto-schedules renewal inspection 60 days before
    created_by="compliance_officer",
)

# Certificates expiring in the next 90 days
expiring = await svc.get_expiring_certificates("tenant-1", within_days=90)

# Full compliance snapshot for lettings due-diligence
status = await svc.get_property_compliance_status("tenant-1", "prop-1")
print(status["compliance_status"])  # "compliant" | "warning" | "non_compliant"
```

### Budget vs Actual

```python
# Set annual budget
await svc.set_maintenance_budget(
    tenant_id="tenant-1",
    property_id="prop-1",
    financial_year="2026",
    budget_amount=Decimal("2500000"),
    currency="KES",
    created_by="finance_director",
)

# Track spend during the year
variance = await svc.get_budget_vs_actual("tenant-1", "prop-1", "2026")
if variance["over_budget"]:
    print(f"Over budget by KES {abs(variance['remaining_budget']):,.0f}")
```

### Failure Pattern Detection

```python
# Identify assets with >= 3 corrective WOs in the last 30 days
patterns = await svc.detect_reactive_patterns("tenant-1", window_days=30, repeat_threshold=3)
for p in patterns:
    print(p["asset_name"], p["corrective_wo_count"], p["suggested_action"])
```

### Portfolio Benchmarking

```python
benchmark = await svc.benchmark_portfolio(
    tenant_id="tenant-1",
    property_ids=["prop-1", "prop-2", "prop-3"],
    period="2026-Q1",
    gross_areas_sqm={"prop-1": 4200.0, "prop-2": 2800.0, "prop-3": 6100.0},
)
for row in benchmark["ranking"]:
    print(row["rank"], row["property_id"], "composite:", row["composite_percentile"])
```

### Escalation Policies

```python
# Define escalation chain for P1 work orders
policy = await svc.create_escalation_policy(
    tenant_id="tenant-1",
    priority="p1_critical",
    levels=[
        {"level": 1, "delay_minutes": 15, "notified_roles": ["facilities_manager"]},
        {"level": 2, "delay_minutes": 30, "notified_roles": ["property_director"]},
        {"level": 3, "delay_minutes": 60, "notified_roles": ["ceo", "external_helpdesk"]},
    ],
    created_by="admin",
)

# Call this on a scheduler tick (e.g. every 5 minutes)
result = await svc.process_escalations("tenant-1")
print(result["escalations_triggered"], "escalations triggered")
```

### Sustainability Tracking

```python
record = await svc.sustainability_tracking(
    property_id="prop-1",
    energy_kwh=14500.0,
    water_m3=380.0,
    waste_kg=620.0,
    period="2026-04",
    tenant_id="tenant-1",
    recycling_rate_pct=42.0,
)
print(record["carbon_tonnes_co2e"], "tCO2e")
```

---

## Maintenance Analytics

```python
analytics = await svc.maintenance_analytics(period="2026-Q1", tenant_id="tenant-1")
# Keys: total_work_orders, sla_compliance_pct, ppm_completion_rate_pct,
#       eol_assets, open_defects, total_maintenance_cost, total_carbon_tonnes

cost_sqm = await svc.cost_per_sqm("prop-1", "2026", "tenant-1", gross_internal_area_sqm=4200)
print(f"KES {cost_sqm['cost_per_sqm']}/sqm")
```

---

## SLA Priority Matrix

| Priority | Response | Resolution | Typical Use |
|----------|----------|------------|-------------|
| P1 Critical | 1 h | 4 h | Life safety, complete service loss |
| P2 High | 4 h | 24 h | Major disruption, partial service loss |
| P3 Medium | 8 h | 72 h | Significant inconvenience |
| P4 Low | 24 h | 168 h | Minor defect, no immediate impact |
| P5 Planned | 72 h | 336 h | Scheduled/programmed work |

---

## PPM Frequency Reference

| Frequency Key | Interval |
|---------------|----------|
| `daily` | 1 day |
| `weekly` | 7 days |
| `fortnightly` | 14 days |
| `monthly` | 30 days |
| `quarterly` | 91 days |
| `semi_annual` | 182 days |
| `annual` | 365 days |
| `biennial` | 730 days |

---

## Business Rules

| Rule | Trigger | Effect |
|------|---------|--------|
| Decommissioned asset work order denied | Asset status = decommissioned | Deny WO creation |
| P1 requires immediate assignment | Priority = p1_critical, no contractor | Deny |
| Uninsured contractor denied | Insurance expired or absent | Deny assignment |
| SLA breach requires escalation | Breached, not escalated | Deny status update |
| Work order close requires verification | `verification_complete = False` | Deny close |
| Statutory inspection overdue triggers alert | Overdue + no alert sent | Deny completion |
| Condition score < 20 → end_of_life | Condition score update | Lifecycle phase escalated |

---

## Composability

```apg
use realestate_mai;
```

| Integration | Capability | Notes |
|-------------|------------|-------|
| Property master | `realestate_prm` | Asset `property_id` references |
| Ledger posting | `realestate_acc` | Completed WO costs post as maintenance charges |
| Contractor panel | `realestate_con` | Shared contractor registry |
| Notifications | `ntfy` | SLA breach, P1, statutory overdue, escalation alerts |
| Scheduler | `schd` | PPM forward generation, escalation tick |
| Workflows | `wflo` | Budget over-run approval, large WO authorisation |
| Message bus | `mqeb` | Domain events: `work_order_raised`, `ppm_completed`, etc. |

---

## Further Reading

- `service.py` — Business logic and all async methods
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 strategic capability enhancements
- `SPECIFICATION.md` — Full capability specification
