# Logistics & Transportation (scm_log)

Carrier integration, shipment tracking, freight audit, route optimisation,
customs documentation, 3PL management, carbon footprint tracking, SLA monitoring,
shipment consolidation, carrier scorecards, freight insurance, and proof-of-delivery.

## API Endpoints

### Core

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/health | Health check |
| GET | /api/scm/log/describe | Capability contract |
| GET | /api/scm/log/audit-events | Tenant audit event log |

### Carriers

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/carriers | List carriers |
| POST | /api/scm/log/carriers | Create carrier |
| GET | /api/scm/log/carriers/{id} | Get carrier |
| PUT | /api/scm/log/carriers/{id} | Update carrier |
| DELETE | /api/scm/log/carriers/{id} | Soft-delete carrier |
| POST | /api/scm/log/carriers/{id}/rate | Rate carrier performance |
| GET | /api/scm/log/carriers/{id}/scorecard | Carrier performance scorecard |

### Shipments

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/shipments | List shipments |
| POST | /api/scm/log/shipments | Create shipment |
| POST | /api/scm/log/shipments/bulk | Bulk create shipments |
| GET | /api/scm/log/shipments/{id} | Get shipment |
| PUT | /api/scm/log/shipments/{id} | Update shipment |
| POST | /api/scm/log/shipments/{id}/book | Book shipment |
| POST | /api/scm/log/shipments/{id}/cancel | Cancel shipment |
| POST | /api/scm/log/shipments/consolidate/suggest | Suggest consolidation groups |
| POST | /api/scm/log/shipments/consolidate | Create consolidated shipment |

### Tracking

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/shipments/{id}/tracking | Get tracking events |
| POST | /api/scm/log/shipments/{id}/tracking | Add tracking event |
| GET | /api/scm/log/shipments/{id}/pod | Get proof-of-delivery records |
| POST | /api/scm/log/shipments/{id}/pod | Attach proof-of-delivery |

### Freight Audit

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/freight-audits | List freight audits |
| POST | /api/scm/log/freight-audits | Create freight audit |
| POST | /api/scm/log/freight-audits/{id}/resolve | Resolve audit |

### Routes

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/routes | List routes |
| POST | /api/scm/log/routes | Create route |
| POST | /api/scm/log/routes/{id}/optimise | Optimise route |

### Customs Documents

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/customs-documents | List customs documents |
| POST | /api/scm/log/customs-documents | Create customs document |
| POST | /api/scm/log/customs-documents/{id}/submit | Submit document |

### 3PL Providers

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/3pl-providers | List 3PL providers |
| POST | /api/scm/log/3pl-providers | Register 3PL provider |
| POST | /api/scm/log/shipments/{id}/assign-3pl | Assign shipment to 3PL |

### Delivery Exceptions

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/scm/log/shipments/{id}/exceptions | Raise delivery exception |
| POST | /api/scm/log/exceptions/{id}/resolve | Resolve exception |

### Carbon & Sustainability

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/shipments/{id}/co2 | Calculate shipment CO2e |
| GET | /api/scm/log/analytics/emissions | Emissions report (Scope 3) |

### SLA Monitoring

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/sla/breaches | Check SLA breaches and at-risk shipments |
| GET | /api/scm/log/sla/performance | SLA performance report by carrier |

### Insurance

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/scm/log/shipments/{id}/insurance/quote | Request insurance quote |
| POST | /api/scm/log/shipments/{id}/insurance/bind | Bind insurance policy |
| POST | /api/scm/log/shipments/{id}/insurance/claim | File insurance claim |

### Analytics

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/analytics/shipments | Shipment analytics |
| GET | /api/scm/log/analytics/freight-costs | Freight cost summary |

## Status Flows

```
Shipment:         draft → booked → in_transit → delivered | exception | cancelled
Freight Audit:    pending → approved | disputed → resolved
Customs Document: draft → submitted → approved | rejected
Delivery Exception: open → resolved
Insurance Policy: active → expired | cancelled
Insurance Claim:  filed → under_review → approved | rejected
```

## Supported Values

- **Freight modes**: air, sea, road, rail, multimodal
- **Service levels**: express, standard, economy
- **Carrier types**: air, sea, road, rail, multimodal, courier
- **Document types**: commercial_invoice, packing_list, bill_of_lading, certificate_of_origin, customs_declaration, airway_bill
- **Tracking events**: pickup, in_transit, customs_clearance, out_for_delivery, delivered, exception, returned

---

## World-Class Enhancements (v2.0)

1. **Carrier Webhook Integration** — normalise DHL/FedEx/UPS push payloads to canonical tracking events with HMAC verification
2. **VRP Route Optimisation** — OR-Tools solver with time windows, capacity constraints, CO2 estimate, and 6-hour plan cache
3. **Freight Rate Benchmarking** — p10/p50/p90 lane rates from configurable rate-card store (Freightos/Xeneta adapters)
4. **Shipment Consolidation Engine** — group by corridor + departure window; master bill of lading with parent-child relationships
5. **Carbon Footprint Tracking** — GLEC v3 emission factors per mode; Scope 3 Cat 4/9 aggregation; GHG Protocol CSV export
6. **Proof-of-Delivery Management** — object-storage-referenced POD with SHA-256 signature hash and pre-signed URL retrieval
7. **SLA Breach Detection** — proactive at-risk / breached scan with carrier SLA penalty calculation and event emission
8. **DG/HAZMAT Compliance** — IATA/IMDG classification, carrier capability check, DG manifest generation
9. **Multi-Currency Normalisation** — ECB/Open Exchange Rates FX conversion with `fx_rate_used` audit trail and 1-hour cache
10. **Carrier Scorecard & Tender** — on-time rate, CO2/tonne-km, cost/kg scoring; weighted tender ranking with audit trail
11. **Customs Tariff & Duty Estimation** — WCO HS code validation; CIF + tariff + VAT breakdown; trade agreement support
12. **Shipment Insurance Integration** — quote → bind → claim workflow keyed on declared value, route risk score, and commodity
13. **CloudEvents Bus** — async `_publish` with CloudEvents v1.0 JSON; Redis Streams / Bytewax / asyncio.Queue transports
14. **Predictive ETA Engine** — gradient-boosted model on historical transit data with weather and port congestion signals
15. **Distributed Tracing & Observability** — OpenTelemetry spans, W3C traceparent propagation, Prometheus metrics, Grafana dashboard

---

## New Methods

### `calculate_shipment_co2` — Scope 3 emissions per shipment

```python
svc = LogisticsService()

# book a road shipment first, then calculate emissions
co2 = await svc.calculate_shipment_co2(
    shipment_id="shp_abc123",
    distance_km=850.0,          # omit to use route distance or 1000 km fallback
    tenant_id="acme",
)
# co2["co2_kg"] => 8.16  (road: 0.096 kgCO2e/tonne-km, 100 kg, 850 km)
# co2["framework"] => "GLEC v3"
# co2["scope"]     => "scope_3"
```

### `check_sla_breaches` — proactive SLA monitoring

```python
report = await svc.check_sla_breaches(
    at_risk_hours=4.0,   # flag shipments with ETA within this window
    tenant_id="acme",
)
# report["breached"]  => list of overdue shipments with breach_hours
# report["at_risk"]   => shipments approaching SLA limit
# report["summary"]   => {"breached_count": 2, "at_risk_count": 5, "on_track_count": 41}
# emits "sla_breach_detected" / "sla_at_risk" CloudEvents per shipment
```

### `attach_pod` — dispute-proof delivery confirmation

```python
pod = await svc.attach_pod(
    shipment_id="shp_abc123",
    document_url="https://storage.example.com/pods/shp_abc123.jpg?X-Amz-Signature=...",
    captured_by="driver_007",
    mime_type="image/jpeg",
    signature_hash="e3b0c44298fc1c149afb...",  # SHA-256 of signature image
    recipient_name="Jane Doe",
    tenant_id="acme",
)
# pod["id"] persisted; shipment status auto-transitions in_transit → delivered
# emits "pod_attached" CloudEvent
```
