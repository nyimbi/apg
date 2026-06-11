# Cargo Management — User Guide

**Capability ID**: `transport_car` | **Domain**: `transport` | **Version**: `1.0.0`
**Copyright**: © 2025 Datacraft | **Author**: Nyimbi Odero

---

## Overview

The Cargo Management capability (`transport_car`) provides end-to-end cargo lifecycle management:
booking creation, manifest generation, dangerous-goods compliance, real-time tracking, predictive ETA,
yard/CFS management, LCL consolidation, transport-document generation, carbon footprint reporting,
customs pre-clearance, dispute resolution, insurance, revenue management, and analytics.

All business logic is in `service.py`. The service is tenant-scoped and fully async.

---

## Installation

```bash
pip install apg-transport-car
```

---

## Quick Start

```python
import asyncio
from apg_transport_car.service import CargoManagementService

svc = CargoManagementService(tenant_id="acme", actor_id="ops-user")

async def main():
    # 1. Create a booking
    booking = await svc.book_cargo(
        shipper_id="SHP-001",
        origin="Mombasa",
        destination="Nairobi",
        cargo_type="general",
        weight_kg=500.0,
        dimensions={"length_cm": 120, "width_cm": 100, "height_cm": 80},
        consignee_id="CON-002",
        incoterm="fob",
        packaging_type="pallets",
    )
    print(booking["booking_id"], booking["total_charge"])

asyncio.run(main())
```

---

## Core Workflows

### 1. Cargo Booking

```python
# Full booking with rate calculation (volumetric weight applied automatically)
booking = await svc.book_cargo(
    shipper_id="SHP-001",
    origin="Mombasa",
    destination="Nairobi",
    cargo_type="general",
    weight_kg=500.0,
    dimensions={"length_cm": 120, "width_cm": 100, "height_cm": 80},
)

# Rate inquiry — no booking created
rate = await svc.rate_inquiry("Mombasa", "Nairobi", "general", 500.0, 0.96)

# Amend a booking
amended = await svc.booking_amendment(
    booking["booking_id"],
    {"weight_kg": 550.0, "incoterm": "cif"},
)

# Bulk creation
bookings = await svc.bulk_create_bookings([
    {"shipper_id": "SHP-001", "origin": "Mombasa", "destination": "Kisumu",
     "cargo_type": "refrigerated", "weight_kg": 200.0,
     "dimensions": {"length_cm": 80, "width_cm": 60, "height_cm": 50}},
])
```

### 2. Cargo Manifest

```python
manifest = await svc.cargo_manifest(
    booking["booking_id"],
    customs_ref="CUS-KE-2026-0001",
)
```

### 3. Dangerous Goods

```python
# Check DG classification requirements
dg_check = await svc.dangerous_goods_check(
    booking["booking_id"],
    un_class="class_3_flammable_liquids",
    un_number="UN1203",
    packing_group="II",
)
print(dg_check["required_documents"])

# Declare DG on a booking
svc.declare_dangerous_goods(
    "DG-001", svc.tenant_id, booking["booking_id"],
    "class_3_flammable_liquids", "UN1203", "II",
    "emergency@acme.com", "imdg",
)
```

### 4. Cargo Tracking

```python
# Record a tracking event
svc.update_tracking(
    "EVT-001", svc.tenant_id, booking["booking_id"],
    "in_transit", "Nairobi ICD", "2026-06-11T10:00:00Z",
)

# Full tracking chain with milestones
track = await svc.track_cargo(booking["booking_id"])
print(track["current_status"], track["milestone_progress_pct"])

# Predictive ETA
eta = await svc.predict_eta(
    booking["booking_id"],
    carrier_avg_speed_kmh=80.0,
    distance_km=480.0,
)
print(f"P50 ETA: {eta['eta_p50']}  |  P90 ETA: {eta['eta_p90']}")
print(f"On-time probability: {eta['on_time_probability']:.0%}")
```

### 5. Customs Declaration and Pre-Clearance

```python
# Build a customs declaration with HS-code duty calculation
decl = await svc.customs_declaration(
    booking["booking_id"],
    value=15_000.0,
    hs_codes=["8471", "8473"],
    country_of_origin="CN",
    currency="USD",
)
print(f"Total duty estimate: {decl['total_estimated_duty']} USD")

# Submit to customs gateway
submission = await svc.submit_customs_pre_clearance(
    decl["declaration_ref"],
    customs_system="asycuda",
    notify_on_release=True,
)
print(f"Gateway ref: {submission['gateway_ref']}")
print(f"Clearance ETA: {submission['clearance_eta']}")
```

### 6. Transport Documents

```python
# Bill of Lading
bol = await svc.generate_transport_document(
    booking["booking_id"],
    doc_type="bol",
    issuer_name="Datacraft Logistics",
    signatory="Nyimbi Odero",
)

# Air Waybill
awb = await svc.generate_transport_document(booking["booking_id"], doc_type="awb")

# CMR consignment note
cmr = await svc.generate_transport_document(booking["booking_id"], doc_type="cmr")
```

### 7. LCL Consolidation

```python
# Consolidate two LCL bookings into a 20ft container
consol = await svc.consolidate_bookings(
    booking_ids=["CBK-A", "CBK-B"],
    container_type="container_20ft",
)
print(f"HBL: {consol['hbl_ref']}")
print(f"Fill rate: {consol['fill_rate_weight_pct']}% weight, {consol['fill_rate_volume_pct']}% volume")
print(f"Loadable: {consol['loadable']}")
if consol["segregation_warnings"]:
    print("WARNINGS:", consol["segregation_warnings"])
```

**Container limits**:
| Type | Max weight | Max volume |
|---|---|---|
| container_20ft | 28 000 kg | 25.0 CBM |
| container_40ft | 26 500 kg | 60.0 CBM |
| container_reefer | 26 000 kg | 58.0 CBM |

### 8. Yard / CFS Management

```python
# Assign to a yard bay
assignment = await svc.assign_yard_location(
    booking["booking_id"],
    yard_id="ICD-NRB-01",
    bay="B-12",
    stack="S-3",
    free_storage_days=5,
)

# Release from yard (computes dwell and storage charges automatically)
release = await svc.release_from_yard(
    assignment["assignment_id"],
    cargo_type="dry",
    currency="USD",
)
print(f"Dwell days: {release['actual_dwell_days']}")
print(f"Storage charge: {release['storage_charge']} USD")
```

### 9. Insurance and Claims

```python
# Attach cargo insurance
policy = await svc.cargo_insurance(
    booking["booking_id"],
    insured_value=20_000.0,
    policy_type="all_risk",
    insurer="AIG",
)
print(f"Premium: {policy['premium']} USD  ({policy['premium_rate_pct']}%)")

# File a loss claim
claim = await svc.cargo_loss_claim(
    booking["booking_id"],
    loss_description="Water damage to electronics",
    amount=5_000.0,
    evidence_refs=["PHOTO-001", "SURVEY-REP-2026-06"],
)

# Open a structured dispute
dispute = await svc.open_dispute(
    booking["booking_id"],
    dispute_type="damage",
    description="20% of consignment damaged on arrival",
    claimed_amount=3_000.0,
    evidence_refs=["PHOTO-001"],
)
print(f"Dispute ID: {dispute['dispute_id']}, covered: {dispute['covered_by_insurance']}")
```

### 10. Carbon Footprint

```python
footprint = await svc.calculate_carbon_footprint(
    booking["booking_id"],
    distance_km=480.0,
    mode="road",
)
print(f"Net CO₂e: {footprint['net_kg_co2e']} kg")
print(f"Offset cost: ${footprint['offset_cost_usd']}")
print(f"SBTi scope: {footprint['sbti_scope']}")
```

**Modal emission factors (g CO₂ / tonne-km)**:
| Mode | Factor |
|---|---|
| road | 62 |
| rail | 22 |
| sea | 8 |
| air | 602 |

### 11. Revenue Management

```python
# Record a freight charge
svc.record_revenue("REV-001", svc.tenant_id, booking["booking_id"],
                   "freight_charge", 850.0, "USD", "INV-2026-001")

# Route revenue analysis
revenue = await svc.revenue_management("Mombasa-Nairobi", "2026-06-11")
print(f"Yield per kg: {revenue['yield_per_kg']}")
print(f"Contribution margin: {revenue['contribution_margin']} USD ({revenue['cm_pct']}%)")
```

### 12. Analytics and Reporting

```python
# Aggregate KPIs for a period
kpis = await svc.cargo_analytics("2026-06")
print(f"Bookings: {kpis['total_bookings']}, Revenue: {kpis['total_revenue_usd']} USD")
print(f"Top route: {kpis['top_routes_by_volume'][0]['route']}")

# Performance KPIs
perf = await svc.performance_kpi()

# Cost analysis
costs = await svc.cost_analysis("2026-06")

# Full analytics dashboard
dash = await svc.analytics_dashboard()

# Export data
export = await svc.export_cargo_data("2026-06", format="csv")
print(export["download_ref"])

# Generate report
report = await svc.reporting_export("2026-06", report_type="detailed")
```

### 13. Operations

```python
# Log an exception
exc = await svc.exception_handling(
    booking["booking_id"], "delay", "Port congestion at Mombasa"
)

# Bulk operation
result = await svc.bulk_operation("confirm", ["CBK-001", "CBK-002", "CBK-003"])

# Send consignee notification
notif = await svc.customer_notification(
    booking["booking_id"],
    "Your cargo has cleared customs and is out for delivery.",
    channel="sms",
)

# Health check
health = await svc.health_check()
print(health["status"])
```

---

## Supported Values

### Cargo Types
`general`, `bulk`, `liquid`, `refrigerated`, `frozen`, `hazardous`, `oversized`,
`fragile`, `livestock`, `valuable`, `pharmaceutical`, `automotive`

### DG Classes
`class_1_explosives`, `class_2_gases`, `class_3_flammable_liquids`,
`class_4_flammable_solids`, `class_5_oxidizers`, `class_6_toxic`,
`class_7_radioactive`, `class_8_corrosives`, `class_9_miscellaneous`

### Incoterms
`exw`, `fca`, `cpt`, `cip`, `dat`, `dap`, `ddp`, `fas`, `fob`, `cfr`, `cif`

### Packaging Types
`pallet`, `crate`, `drum`, `ibc`, `flexibag`, `container_20ft`, `container_40ft`,
`container_reefer`, `loose`, `roll`, `coil`

### Revenue Types
`freight_charge`, `fuel_surcharge`, `hazmat_surcharge`, `oversize_surcharge`,
`storage_fee`, `customs_fee`, `insurance_fee`, `handling_fee`

### Tracking Events
`booked`, `collected`, `in_transit`, `customs_hold`, `out_for_delivery`,
`delivered`, `exception`, `returned`

### Dispute Types
`weight_discrepancy`, `damage`, `short_delivery`, `delay_penalty`, `billing_error`

### Customs Systems
`asycuda`, `tradenet`, `icegate`

### Transport Document Types
`bol` (Bill of Lading), `awb` (Air Waybill), `cmr` (CMR Consignment Note)

---

## Configuration

All keys are tenant-scoped. Override via the `conf` capability or environment variables
prefixed with `TRANSPORT_CAR_`:

| Key | Default | Description |
|-----|---------|-------------|
| `tenant_id` | `default` | Tenant identifier |
| `bookings.shipper_required` | `true` | Shipper is mandatory |
| `bookings.consignee_required` | `true` | Consignee is mandatory |
| `dangerous_goods.compliance_standards` | 8 standards | Enabled DG frameworks |
| `revenue.approval_required_above_threshold` | `true` | Large charge approval |
| `governance.require_tenant_context` | `true` | Tenant context enforced |
| `tracking.real_time_enabled` | `true` | Real-time events on |
| `tracking.geofencing_enabled` | `true` | Geofence breach alerts |

---

## Composability

| Capability | Integration Point |
|---|---|
| `transport_dis` | Dispatch assignment after booking confirmation |
| `transport_rou` | Route optimisation per booking leg |
| `transport_tra` | Real-time GPS/IoT telemetry ingestion |
| `transport_sch` | Load scheduling and yard time-window constraints |
| `comp` | DG certificates, customs document vault, arbitration records |
| `mqeb` | bytewax event streaming for all lifecycle events |
| `ntfy` | Customer and operator notifications |
| `audl` | Immutable audit trail for all write operations |

Reference in `.apg` source files:

```apg
use transport_car;
```

---

## Streaming Events (bytewax)

Stream: `apg.transport.cargo.lifecycle`

Events emitted on every state change:
- `cargo_booked`, `cargo_manifest_submitted`, `cargo_dg_declared`
- `cargo_tracking_updated`, `cargo_delivered`
- `cargo_revenue_recorded`, `cargo_compliance_checked`
- `cargo_agent_registered`
- `cargo_yard_assigned`, `cargo_yard_released`
- `transport_document_generated_bol`, `transport_document_generated_awb`, `transport_document_generated_cmr`
- `cargo_consolidation_created`, `carbon_footprint_calculated`
- `cargo_dispute_opened_*`, `customs_pre_clearance_submitted`
- `detention_demurrage_calculated`, `cargo_insurance_attached`, `cargo_loss_claim_submitted`

---

## Testing

```bash
# Run all tests
uv run pytest -vxs capabilities/transport/car/tests/

# Type check
uv run pyright capabilities/transport/car/service.py
```

---

## Further Reading

- `service.py` — Business logic implementation (all 39+ async methods)
- `models.py` — Data models
- `capability_contract.py` — Rule engine and supported values
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 planned enhancements roadmap
