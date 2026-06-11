# Delivery Management — User Guide

**Capability ID**: `transport_del` | **Domain**: `transport` | **Version**: `2.0.0`
**Author**: Nyimbi Odero | **Copyright**: © 2025 Datacraft

---

## Description

`transport_del` manages the complete last-mile delivery lifecycle: order intake, driver assignment, real-time ETA propagation, multi-modal proof-of-delivery (POD) capture, SLA tracking with penalty calculation, failed-delivery handling, return initiation, route optimisation, carbon footprint tracking, driver gamification, insurance claims, and webhook push to external systems.

---

## Installation

```bash
pip install apg-transport-del
```

---

## Quick Start

```python
import asyncio
from apg_transport_del.service import DeliveryManagementService

svc = DeliveryManagementService(tenant_id="acme", actor_id="ops-bot")

async def main():
    # 1. Create delivery from an order
    result = await svc.create_delivery_async(
        order_id="ORD-001",
        origin="Nairobi Depot",
        destination="14 Moi Ave, Nairobi",
        customer_phone="+254700000001",
        instructions="Leave at gate if no answer",
        delivery_type="express",
        sla_tier="gold",
    )
    delivery_id = result["delivery"]["id"]

    # 2. Assign a driver
    await svc.assign_driver(delivery_id, driver_id="DRV-42", vehicle_id="KBZ-123")

    # 3. Optimise route for the driver's full run
    route = await svc.optimise_route("DRV-42", [delivery_id], tenant_id="acme")

    # 4. Update live ETA as driver moves
    eta = await svc.update_realtime_eta(delivery_id, driver_lat=-1.2921, driver_lon=36.8219)

    # 5. Record POD on delivery
    pod = await svc.proof_of_delivery(
        delivery_id, signature="base64...", photo=None, gps="-1.2921,36.8219"
    )

    # 6. Customer rates the delivery
    rating = await svc.delivery_rating(delivery_id, score=5, comment="On time, friendly driver")

asyncio.run(main())
```

---

## Core Workflows

### 1. Delivery Creation

```python
# Synchronous (low-level)
svc.create_delivery(
    delivery_id="DLV-001", tenant_id="acme", delivery_type="standard",
    recipient_name="Alice Mwangi", delivery_address="14 Kenyatta Ave",
    time_window_start="2026-06-11T08:00:00Z",
    time_window_end="2026-06-11T12:00:00Z",
    sla_tier="silver",
)

# Async (recommended — attaches SLA + fires SMS)
result = await svc.create_delivery_async(
    order_id="ORD-9912",
    origin="Westlands Hub",
    destination="Karen, Nairobi",
    customer_phone="+254711000002",
    instructions="Ring twice",
)
```

### 2. Driver Assignment

```python
assignment = await svc.assign_driver(
    delivery_id="DLV-001",
    driver_id="DRV-7",
    vehicle_id="KCB-007A",
    estimated_pickup_at="2026-06-11T09:30:00Z",
)
# Raises ValueError if an active assignment already exists.
```

### 3. Proof of Delivery

```python
pod = await svc.proof_of_delivery(
    delivery_id="DLV-001",
    signature="base64-encoded-svg",
    photo=None,
    gps="-1.2864,36.8172",
    signatory_name="Bob Odhiambo",
)
# pod_type selected automatically: signature > photo > signature_and_photo
```

### 4. Failed Delivery

```python
failure = await svc.failed_delivery(
    delivery_id="DLV-001",
    reason="no_answer",
    next_action="reattempt",
    notes="Rang doorbell 3 times",
    notify_customer=True,
)
```

### 5. Reattempt and Rescheduling

```python
# Max 3 attempts enforced — raises ValueError on 4th
reattempt = await svc.reattempt_delivery(
    delivery_id="DLV-001",
    new_time_window_start="2026-06-12T10:00:00Z",
    new_time_window_end="2026-06-12T14:00:00Z",
)
```

### 6. SLA Tracking

```python
# Check current SLA status
status = await svc.delivery_sla_check("DLV-001")
# Returns: sla_tier, status ('on_track'|'at_risk'|'breached'|'met'),
#          penalty_exposure_usd, escalation_recommended

# Breach report for a period
report = await svc.sla_breach_report("2026-Q2")
```

### 7. Returns Management

```python
ret = await svc.returns_management(
    delivery_id="DLV-001",
    return_reason="damaged",
    rma_number="RMA-20260611",
    restocking_fee_pct=10.0,
)
```

### 8. Customer Ratings

```python
# Only valid after delivery status = 'delivered'
rating = await svc.delivery_rating(
    delivery_id="DLV-001",
    score=4,
    comment="Slightly late but polite driver",
)
# Returns driver_average_score across all rated deliveries
```

---

## Advanced Features

### Route Optimisation

```python
route = await svc.optimise_route(
    driver_id="DRV-7",
    delivery_ids=["DLV-001", "DLV-002", "DLV-003"],
    constraints={"avoid_highway": True},
)
# Returns optimised sequence with per-stop ETA offsets and total_km
```

In production, wire the inner loop to an OR-Tools VRP solver or a locally-hosted Ollama route-planner model. The stub uses a greedy nearest-neighbour at 5 km / 30 km/h.

### Real-Time ETA Updates

```python
eta = await svc.update_realtime_eta(
    delivery_id="DLV-001",
    driver_lat=-1.3000,
    driver_lon=36.8100,
    avg_speed_kmh=25.0,
)
# Automatically fires an SMS to the customer.
```

### Carbon Footprint

```python
co2 = await svc.compute_carbon_footprint(
    delivery_id="DLV-001",
    distance_km=14.3,
    vehicle_type="electric_van",
    load_kg=120.0,
)
# Returns carbon_kg, tier (low/medium/high), emission_factor_kg_per_km
# Factors follow DEFRA/GHG Protocol Scope 3 Category 4 defaults.
```

### Failure Risk Scoring

```python
risk = await svc.score_failed_delivery_risk("DLV-001")
# failure_probability: 0.0-1.0
# recommended_action: 'proceed' | 'pre_call' | 'confirm_before_dispatch'
```

Run this before dispatching to identify high-risk deliveries and trigger pre-delivery confirmation calls.

### Delivery Manifests (Multi-Parcel)

```python
# Group deliveries into one driver run
manifest = await svc.create_delivery_manifest(
    manifest_id="MFT-2026-001",
    driver_id="DRV-7",
    delivery_ids=["DLV-001", "DLV-002", "DLV-003"],
    vehicle_id="KCB-007A",
)

# Record batch POD for all deliveries at a shared location (e.g. building reception)
result = await svc.complete_manifest(
    manifest_id="MFT-2026-001",
    gps="-1.2921,36.8219",
    signatory_name="Concierge",
)
```

### Insurance Claims

```python
claim = await svc.file_insurance_claim(
    delivery_id="DLV-001",
    claim_type="damage",          # 'damage' | 'loss' | 'theft' | 'partial_loss'
    evidence_urls=["https://cdn.example.com/photos/pkg-damage-001.jpg"],
    declared_value_usd=250.0,
)
# Returns estimated_payout_usd = declared_value × payout_rate (0.80 for damage)
```

### Webhook Push

```python
webhook = await svc.register_webhook(
    webhook_id="WH-SHOPIFY-001",
    url="https://myshop.example.com/hooks/delivery",
    events=["delivery_completed", "delivery_failed", "pod_recorded"],
    secret="s3cr3t-hmac-key",
)
# Every matching event POSTs JSON signed with X-APG-Signature: sha256=<hmac>
```

### Driver Incentive Calculation

```python
incentive = await svc.compute_driver_incentive(
    driver_id="DRV-7",
    period="2026-06",
    base_incentive_usd=75.0,
)
# score: composite 0-1 from on_time_rate, pod_compliance, avg_rating, km_efficiency
# payout_usd = base_incentive_usd × score
```

---

## Analytics and Reporting

```python
# Last-mile KPIs
analytics = await svc.last_mile_analytics("2026-Q2", vehicle_type="van")

# Driver performance
driver_report = await svc.driver_performance_report("DRV-7", "2026-06")

# POD compliance audit
compliance = await svc.pod_compliance_check()

# Full cost breakdown
costs = await svc.cost_analysis("2026-06")

# Dashboard snapshot
dashboard = await svc.analytics_dashboard()

# Export records
export = await svc.export_delivery_data("2026-06", format="csv")
```

---

## Bulk Operations

```python
# Bulk create from order list
orders = [
    {"order_id": "ORD-A", "origin": "Depot", "destination": "Addr A",
     "customer_phone": "+254700000001", "instructions": ""},
    {"order_id": "ORD-B", "origin": "Depot", "destination": "Addr B",
     "customer_phone": "+254700000002", "instructions": ""},
]
results = await svc.bulk_create_deliveries(orders, sla_tier="gold")

# Apply operation to many deliveries
bulk = await svc.bulk_operation("flag_for_audit", ["DLV-001", "DLV-002"])
```

---

## Configuration Reference

All keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TRANSPORT_DEL_`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `failed_deliveries.max_attempts` | int | 3 | Hard limit on delivery attempts |
| `rescheduling.max_reschedule_count` | int | 3 | Max reschedules per delivery |
| `proof_of_delivery.geo_stamp_required` | bool | true | Mandatory geo-stamp on all POD |
| `sla.breach_alert_enabled` | bool | true | Alert supervisor on breach |
| `risk_scoring.failure_threshold` | float | 0.55 | Probability above which confirmation triggered |
| `carbon.default_vehicle_type` | str | van | Vehicle type for carbon estimates |
| `webhook.signing_algorithm` | str | hmac-sha256 | Algorithm for outbound webhook signatures |

---

## Error Handling

| Exception | Cause |
|-----------|-------|
| `KeyError` | Delivery / manifest / driver not found |
| `ValueError` | Invalid parameter, constraint violation (max attempts, score range, claim type) |
| `PermissionError` | Policy enforcement failure (cross-tenant write, POD falsification, missing RMA) |

---

## Interoperability

Reference in `.apg` source files:

```apg
use transport_del;
```

Composes with:
- `transport_dis` — Driver dispatch and capacity management
- `transport_tra` — GPS breadcrumb ingest and live map
- `transport_sch` — Time-window scheduling
- `transport_rou` — Network-level route planning
- `billing` — SLA penalty invoicing
- `ident` — Biometric KYC for high-value POD
- `esg` — GHG Scope 3 carbon disclosure

---

## Further Reading

- `service.py` — Full async service implementation
- `models.py` — Pydantic-free dataclass models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic v2 schemas
- `README.md` — Quick reference and method index
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 planned improvements
- `cap_spec.md` — Formal capability specification
