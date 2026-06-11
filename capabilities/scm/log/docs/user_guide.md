# Logistics & Transportation User Guide (scm_log)

## Overview

`scm_log` provides end-to-end logistics management for multi-tenant supply chains.
Core pillars: carrier integration, multi-modal shipment tracking, freight invoice
auditing, route optimisation, customs documentation, 3PL provider management,
carbon footprint reporting, SLA monitoring, shipment consolidation, carrier
performance scorecards, freight insurance, and proof-of-delivery.

---

## Key Use Cases

| Use Case | Primary Methods |
|----------|-----------------|
| Carrier onboarding | `create_carrier`, `update_carrier`, `rate_carrier` |
| Shipment lifecycle | `create_shipment`, `book_shipment`, `add_tracking_event`, `cancel_shipment` |
| Freight audit | `create_freight_audit`, `resolve_freight_audit`, `freight_cost_summary` |
| Route optimisation | `create_route`, `optimise_route` |
| Customs documentation | `create_customs_document`, `submit_customs_document` |
| 3PL management | `register_3pl_provider`, `assign_shipment_to_3pl` |
| Delivery exceptions | `raise_delivery_exception`, `resolve_delivery_exception` |
| Carbon tracking | `calculate_shipment_co2`, `emissions_report` |
| SLA monitoring | `check_sla_breaches`, `sla_performance_report` |
| Consolidation | `suggest_consolidation`, `create_consolidated_shipment` |
| Carrier scorecard | `generate_carrier_scorecard` |
| Insurance | `request_insurance_quote`, `bind_insurance_policy`, `file_insurance_claim` |
| Proof of delivery | `attach_pod`, `get_pod` |

---

## Quickstart

### 1. Register a carrier

```python
from capabilities.scm.log.service import LogisticsService

svc = LogisticsService(tenant_id="acme")

carrier = await svc.create_carrier(
    name="DHL Express",
    carrier_code="DHLX",
    carrier_type="air",
    country_of_origin="DE",
    services_offered=["express", "economy"],
    contact_email="ops@dhl.com",
)
```

### 2. Create and book a shipment

```python
shipment = await svc.create_shipment(
    carrier_id=carrier["id"],
    origin_address={"city": "Nairobi", "country": "KE"},
    destination_address={"city": "London", "country": "GB"},
    weight_kg=25.5,
    freight_mode="air",
    service_level="express",
    declared_value=3500.00,
    currency="USD",
)

booked = await svc.book_shipment(shipment["id"])
print(booked["tracking_number"])  # TRK...
```

### 3. Add tracking events

```python
await svc.add_tracking_event(
    shipment_id=booked["id"],
    event_type="pickup",
    location="Nairobi JKIA",
    description="Picked up from shipper",
)

await svc.add_tracking_event(
    shipment_id=booked["id"],
    event_type="in_transit",
    location="Dubai DXB Hub",
    description="Departed transit hub",
)
```

### 4. Carbon footprint

```python
co2 = await svc.calculate_shipment_co2(
    shipment_id=booked["id"],
    distance_km=6800,  # NBO → LHR great-circle ~6800 km
)
print(f"{co2['co2_kg']} kgCO2e  ({co2['framework']})")

report = await svc.emissions_report(date_from="2026-01-01Z", date_to="2026-06-30Z")
print(report["total_co2_tonnes"], "tCO2e  Scope 3")
```

### 5. SLA monitoring

```python
# Check for any shipments breached or within 4 hours of SLA
breaches = await svc.check_sla_breaches(at_risk_hours=4.0)
print(breaches["summary"])

# Performance report for a specific carrier over last 90 days
perf = await svc.sla_performance_report(carrier_id=carrier["id"], period_days=90)
```

### 6. Shipment consolidation

```python
# Identify consolidation opportunities across a batch of draft shipments
suggestion = await svc.suggest_consolidation(
    shipment_ids=[s1_id, s2_id, s3_id, s4_id],
    departure_window_hours=24.0,
)
print(suggestion["consolidation_proposals"])

# Execute a consolidation
master = await svc.create_consolidated_shipment(
    child_shipment_ids=[s1_id, s2_id],
    carrier_id=carrier["id"],
    freight_mode="sea",
    service_level="standard",
)
```

### 7. Carrier scorecard

```python
scorecard = await svc.generate_carrier_scorecard(
    carrier_id=carrier["id"],
    period_days=90,
)
print(f"On-time rate: {scorecard['on_time_rate_pct']}%")
print(f"Exception rate: {scorecard['exception_rate_pct']}%")
print(f"Audit dispute rate: {scorecard['audit_dispute_rate_pct']}%")
```

### 8. Insurance

```python
# Get a premium quote
quote = await svc.request_insurance_quote(
    shipment_id=booked["id"],
    coverage_type="all_risk",
)
print(f"Premium: {quote['premium']} {quote['currency']}")

# Bind the policy
policy = await svc.bind_insurance_policy(
    shipment_id=booked["id"],
    quote=quote,
    insured_by="jane.doe@acme.com",
)

# File a claim if something goes wrong
claim = await svc.file_insurance_claim(
    shipment_id=booked["id"],
    claim_type="damage",
    description="Goods arrived with visible water damage",
    claimed_amount=2800.00,
    filed_by="jane.doe@acme.com",
)
```

### 9. Proof of delivery

```python
pod = await svc.attach_pod(
    shipment_id=booked["id"],
    document_url="https://storage.acme.com/pods/shp-xyz.jpg",
    captured_by="driver-id-007",
    signature_hash="sha256:abcdef1234...",
    recipient_name="John Smith",
)

pods = await svc.get_pod(shipment_id=booked["id"])
```

### 10. Freight audit

```python
audit = await svc.create_freight_audit(
    shipment_id=booked["id"],
    carrier_id=carrier["id"],
    invoice_number="INV-2026-001",
    invoiced_amount=1250.00,
    expected_amount=1100.00,
)
print(f"Variance: {audit['variance']} {audit['currency']}")

resolved = await svc.resolve_freight_audit(
    audit_id=audit["id"],
    resolution="disputed",
    resolved_by="finance@acme.com",
    resolution_notes="Carrier charged accessorial not agreed in contract",
)
```

---

## Status Flows

```
Shipment:
  draft → booked → in_transit → delivered | exception | cancelled
                                      ↑
                                 attach_pod transitions in_transit → delivered

Freight Audit:
  pending → approved | disputed → resolved

Customs Document:
  draft → submitted → approved | rejected

Delivery Exception:
  open → resolved

Insurance Policy:
  active → expired | cancelled

Insurance Claim:
  filed → under_review → approved | rejected
```

---

## Carbon Emission Factors (GLEC Framework v3)

| Mode | Factor (kgCO2e / tonne-km) |
|------|---------------------------|
| Air | 0.602 |
| Sea | 0.016 |
| Road | 0.096 |
| Rail | 0.028 |
| Multimodal | 0.120 |

CO2 = `factor × weight_tonnes × distance_km`

---

## Insurance Rate Model

| Mode | Base Rate (% of declared value) |
|------|--------------------------------|
| Air | 0.50% |
| Sea | 0.80% |
| Road | 0.60% |
| Rail | 0.40% |
| Multimodal | 0.70% |

Named perils coverage applies a 0.65x multiplier vs all-risk.

---

## Multi-Tenant Usage

All methods accept an optional `tenant_id` parameter. When omitted, the service's
constructor `tenant_id` is used. Records are always scoped and isolated per tenant.

```python
svc_tenant_a = LogisticsService(tenant_id="tenant_a")
svc_tenant_b = LogisticsService(tenant_id="tenant_b")
# or use the tenant_id parameter on each call
await svc.create_shipment(..., tenant_id="tenant_b")
```

---

## Error Reference

| Exception | Cause |
|-----------|-------|
| `PermissionError: tenant_context_required` | No tenant_id provided anywhere |
| `KeyError: carrier '{id}' not found` | Carrier does not exist or belongs to another tenant |
| `KeyError: shipment '{id}' not found` | Shipment does not exist or belongs to another tenant |
| `ValueError: only draft shipments can be booked` | Attempted to book a non-draft shipment |
| `ValueError: cannot cancel a delivered shipment` | Attempted to cancel after delivery |
| `ValueError: shipment has no bound insurance policy` | Claim filed without a bound policy |
| `ValueError: coverage_type must be 'all_risk' or 'named_perils'` | Invalid insurance coverage type |
