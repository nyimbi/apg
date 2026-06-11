# Pharmaceutical Distribution — User Guide

**Capability ID**: `pharma_dis` | **Domain**: `pharma` | **Version**: `1.1.0`
**Copyright**: © 2025 Datacraft | **Author**: Nyimbi Odero

---

## Description

Manages pharmaceutical distribution operations including cold chain monitoring, product serialisation and verification, wholesale distribution authorisations, product recalls, GDP compliance, and import/export shipment tracking. Enforces WDA requirements, temperature monitoring, serialisation verification, and recall timeline obligations at every distribution boundary.

Version 1.1 adds a fully async surface for high-throughput pipelines, ICH Q1A(R2) MKT calculation, IoT telemetry ingestion with anomaly detection, tiered recall propagation, GS1 aggregation hierarchy validation, WDA renewal workflows, GDP risk scoring, and supply chain integrity gates.

---

## Installation

```bash
pip install apg-pharma-dis
```

---

## Quick Start

```python
from capabilities.pharma.dis.service import PharmaceuticalDistributionService
from capabilities.pharma.dis.models import ShipmentCreate
from datetime import datetime, timedelta

svc = PharmaceuticalDistributionService(tenant_id="acme", actor_id="ops-user-1")

# 1. Create and dispatch a wholesale shipment
payload = ShipmentCreate(
    tenant_id="acme",
    shipment_number="SHP-2026-001",
    distribution_channel="wholesale",
    origin_site="SITE-MFG-01",
    destination_site="WH-LONDON",
    transport_mode="road",
    transport_condition="refrigerated",
    expected_delivery=datetime.utcnow() + timedelta(days=2),
    created_by="ops-user-1",
)
shipment = svc.create_shipment(payload)

# 2. Register WDA for the wholesaler and grant it
wda = svc.register_wda(
    tenant_id="acme",
    wda_number="WDA-GB-12345",
    market="GB",
    holder_name="ACME Wholesale Ltd",
    site_address="1 Distribution Way, London",
    scope=["human_use", "refrigerated"],
    issuing_authority="MHRA",
    created_by="regulatory-1",
)
wda = svc.grant_wda(wda.id, "acme",
    granted_date=datetime.utcnow(),
    expiry_date=datetime.utcnow() + timedelta(days=730),
)

# 3. Attach cold chain monitoring
cc = svc.create_cold_chain_record(
    tenant_id="acme",
    shipment_id=shipment.id,
    product_id="PROD-INSULN-001",
    cold_chain_classification="refrigerated",
    min_temp=2.0,
    max_temp=8.0,
    logger_device_id="LOGGER-XR42",
    qualification_reference="QUAL-2026-001",
    created_by="cold-chain-mgr",
)

# 4. Dispatch
dispatched = svc.dispatch_shipment(
    shipment.id, "acme",
    packing_list_reference="PL-001",
    coa_reference="COA-001",
    wda_reference="WDA-GB-12345",
)
```

---

## Core Workflows

### Shipment Lifecycle

```
planned → dispatched → in_transit → delivered
                     ↘ delayed ↗
                     ↘ exception
```

Key methods:

| Method | Description |
|--------|-------------|
| `create_shipment(payload)` | Create a planned shipment |
| `dispatch_shipment(id, tenant, pl_ref, coa_ref)` | Dispatch with documentation gate |
| `deliver_shipment(id, tenant, serialisation_verified)` | Record delivery with serial check |
| `get_shipment(id, tenant)` | Retrieve single shipment |
| `list_shipments(tenant, status=None)` | List with optional status filter |
| `track_shipment(id, tenant)` | Full tracking snapshot with cold chain status |

### Cold Chain Management

```python
# Monitor a temperature log (batch analysis)
result = svc.cold_chain_monitoring(
    shipment_id=shipment.id,
    temperature_log=[
        {"ts": "2026-06-01T08:00:00Z", "temp": 3.2},
        {"ts": "2026-06-01T09:00:00Z", "temp": 7.8},
        {"ts": "2026-06-01T10:00:00Z", "temp": 9.5},  # breach
    ],
    tenant_id="acme",
    min_acceptable=2.0,
    max_acceptable=8.0,
)
print(result["compliant"])       # False
print(result["breach_count"])    # 1
```

### IoT Telemetry Ingestion (async, v1.1)

```python
import asyncio

async def ingest():
    result = await svc.ingest_cold_chain_telemetry(
        shipment_id=shipment.id,
        tenant_id="acme",
        readings=[
            {"ts": "2026-06-01T08:00:00Z", "temp": 4.1, "humidity": 60},
            {"ts": "2026-06-01T08:05:00Z", "temp": 15.0, "humidity": 58},  # anomaly
        ],
        device_id="LOGGER-XR42",
        auto_excursion=True,
    )
    print(result["anomalies_detected"])   # 1
    print(result["excursion_raised"])     # True

asyncio.run(ingest())
```

### MKT Calculation (ICH Q1A(R2))

```python
import asyncio

async def calc():
    result = await svc.calculate_mkt(
        temperature_log=[{"ts": t, "temp": v} for t, v in zip(timestamps, readings)],
        tenant_id="acme",
        activation_energy_kj=83.14,
        reference_temp_celsius=25.0,
    )
    print(result["mkt_celsius"])         # e.g. 22.4
    print(result["zone_classification"]) # "II"
    print(result["compliant"])           # True

asyncio.run(calc())
```

MKT zones follow ICH Q1A(R2):
- Zone I: ≤21°C (temperate)
- Zone II: 21–25°C (subtropical/mediterranean)
- Zone III: 25–27°C (hot/dry)
- Zone IVa: 27–30°C (hot/humid)
- Zone IVb: >30°C (very hot/very humid)

### Serialisation and Verification (EU FMD / DSCSA)

```python
# Serialise a product unit
record = svc.serialise_product(
    tenant_id="acme",
    product_id="PROD-INSULN-001",
    serial_number="SN-00001234",
    batch_number="BATCH-2026-001",
    standard="eu_fmd",
    aggregation_level="unit",
    gtin="05901234123457",
    created_by="serialisation-sys",
)

# Point-of-receipt verification
result = svc.serialisation_verification(
    pack_id="PACK-001",
    serial_number="SN-00001234",
    gtin="05901234123457",
    tenant_id="acme",
    batch_number="BATCH-2026-001",
)
print(result["overall_verified"])   # True

# GS1 aggregation hierarchy validation (async, v1.1)
import asyncio
hierarchy = asyncio.run(
    svc.validate_aggregation_hierarchy(tenant_id="acme", sscc="00012345678901234562")
)
print(hierarchy["valid"])           # True
print(hierarchy["unit_count"])      # e.g. 60
```

### Bulk Serialisation (async, v1.1)

```python
import asyncio

async def bulk():
    result = await svc.bulk_serialise_products(
        tenant_id="acme",
        specs=[
            {"product_id": "PROD-X", "serial_number": f"SN-{i:06d}",
             "batch_number": "B001", "standard": "gs1", "aggregation_level": "unit"}
            for i in range(1000)
        ],
        created_by="serialisation-sys",
    )
    print(result["created_count"])   # 1000
    print(result["error_count"])     # 0

asyncio.run(bulk())
```

### Recall Management

```python
# Initiate a Class I recall
recall = svc.initiate_recall(
    tenant_id="acme",
    recall_number="RCL-2026-001",
    recall_class="class_i",
    product_id="PROD-INSULN-001",
    batch_numbers=["BATCH-2026-001", "BATCH-2026-002"],
    reason="Contamination identified in stability samples",
    recall_scope="distribution",
    created_by="qa-manager",
)

# Execute serial decommissioning
result = svc.product_recall(
    recall_id=recall.id,
    affected_serials=["SN-00001234", "SN-00001235"],
    tenant_id="acme",
    action="decommission",
)

# Propagate notifications through distribution network (async, v1.1)
import asyncio

async def notify():
    network = [
        {"entity_id": "WHOLESALER-01", "entity_type": "wholesaler", "contact": "ops@ws1.com", "tier": 1},
        {"entity_id": "PHARMACY-01",   "entity_type": "pharmacy",   "contact": "mgr@ph1.com", "tier": 2},
    ]
    result = await svc.propagate_recall_notification(
        recall_id=recall.id,
        tenant_id="acme",
        distribution_network=network,
        notification_channel="email",
        sent_by="regulatory-affairs",
    )
    print(result["coverage_pct"])   # 100.0

asyncio.run(notify())
```

### WDA Management

```python
# Check expiry alerts (fires at 90 days)
alerts = svc.check_wda_expiry("acme")

# Initiate renewal workflow (async, v1.1)
import asyncio
renewal = asyncio.run(svc.initiate_wda_renewal(
    wda_id=wda.id, tenant_id="acme", renewed_by="regulatory-1"
))
print(renewal["document_checklist"])   # 7 items: SMF, GDP cert, QP declaration...
print(renewal["days_to_expiry"])       # e.g. 62
```

### GDP Compliance

```python
# Record a deviation
dev = svc.record_gdp_deviation(
    tenant_id="acme",
    deviation_reference="DEV-2026-001",
    deviation_type="temperature_excursion",
    description="Cold room excursion during maintenance window",
    gdp_status="major",
    created_by="qa-officer",
)

# GDP inspection
inspection = svc.gdp_inspection(
    distributor_id="WHOLESALER-01",
    inspection_date=datetime.utcnow(),
    findings=[
        {"finding": "Batch records incomplete", "severity": "major"},
        {"finding": "Pest control log missing", "severity": "minor"},
    ],
    tenant_id="acme",
    inspector_id="inspector-1",
    inspection_type="routine",
)

# GDP Risk Score for a distributor (async, v1.1)
import asyncio
risk = asyncio.run(svc.gdp_risk_score(
    distributor_id="WHOLESALER-01", tenant_id="acme", lookback_days=365
))
print(risk["risk_score"])   # 0–100
print(risk["risk_band"])    # "low" | "medium" | "high" | "critical"
```

### Supply Chain Integrity Gate (async, v1.1)

Run before confirming a dispatch or accepting a shipment receipt:

```python
import asyncio
check = asyncio.run(svc.supply_chain_integrity_check(
    shipment_id=shipment.id, tenant_id="acme"
))
if not check["overall_pass"]:
    failed = [k for k, v in check["checks"].items() if not v]
    raise ValueError(f"Integrity check failed: {failed}")
```

The five gates checked:
1. No decommissioned serials in shipment batch
2. No active Class I/II recalls on product
3. No critical cold chain excursions
4. Valid WDA for wholesale channel
5. No open critical GDP deviations

### Returns Processing

```python
result = svc.returns_processing(
    return_id="RET-2026-001",
    quantity=5,
    reason="customer_refused",
    condition="saleable",
    tenant_id="acme",
    product_id="PROD-INSULN-001",
    batch_number="BATCH-2026-001",
    serial_numbers=["SN-00001234"],
    created_by="returns-dept",
)
print(result["disposition"])   # "restock"
```

Disposition rules:
- `saleable` → restock (serials kept active)
- `damaged` → destroy (serials decommissioned)
- `expired` → destroy (serials decommissioned)
- `unknown` → quarantine (pending quality review)

### Analytics and Reporting

```python
# Distribution KPIs
analytics = svc.distribution_analytics(period="2026-Q2", tenant_id="acme")
print(analytics["on_time_delivery_rate_pct"])   # e.g. 97.3
print(analytics["excursion_rate_pct"])           # e.g. 2.1

# Extended async regulatory report (v1.1)
import asyncio
report = asyncio.run(svc.async_regulatory_report(
    period="2026-Q2", jurisdiction="EU", tenant_id="acme",
    include_serialisation_summary=True,
))
print(report["class_i_recalls"])                 # 0
print(report["serialisation_verified_count"])    # e.g. 12400
```

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/pharma-dis/dashboard` | `pharma_dis:view` | Overview |
| `/pharma-dis/shipments` | `pharma_dis:shipments` | Operations |
| `/pharma-dis/shipments/<id>` | `pharma_dis:shipments` | Operations |
| `/pharma-dis/cold-chain` | `pharma_dis:cold_chain` | Cold Chain |
| `/pharma-dis/cold-chain/excursions` | `pharma_dis:cold_chain` | Cold Chain |
| `/pharma-dis/cold-chain/mkt` | `pharma_dis:cold_chain` | Cold Chain |
| `/pharma-dis/serialisation` | `pharma_dis:serialisation` | Traceability |
| `/pharma-dis/recalls` | `pharma_dis:recalls` | Recalls |
| `/pharma-dis/recalls/<id>` | `pharma_dis:recalls` | Recalls |
| `/pharma-dis/wda` | `pharma_dis:wda` | Authorisations |
| `/pharma-dis/distributors` | `pharma_dis:gdp` | GDP Compliance |
| `/pharma-dis/reports` | `pharma_dis:reports` | Reporting |

---

## Service Method Reference

### Sync Methods
| Method | Returns |
|--------|---------|
| `describe(tenant_id)` | Capability contract dict |
| `evaluate(context)` | Policy decision dict |
| `create_shipment(payload)` | `Shipment` |
| `dispatch_shipment(id, tenant, pl, coa, wda, by)` | `Shipment` |
| `deliver_shipment(id, tenant, ser_verified)` | `Shipment` |
| `get_shipment(id, tenant)` | `Shipment` |
| `list_shipments(tenant, status)` | `list[Shipment]` |
| `track_shipment(id, tenant)` | tracking dict |
| `create_cold_chain_record(...)` | `ColdChainRecord` |
| `report_excursion(...)` | `TemperatureExcursion` |
| `list_excursions(tenant, shipment_id)` | `list[TemperatureExcursion]` |
| `cold_chain_monitoring(id, log, tenant, ...)` | compliance dict |
| `serialise_product(...)` | `SerialisationRecord` |
| `verify_serialisation(tenant, serial)` | verification dict |
| `serialisation_verification(pack_id, serial, gtin, ...)` | full verification dict |
| `initiate_recall(...)` | `RecallRecord` |
| `complete_recall(id, tenant, ...)` | `RecallRecord` |
| `list_recalls(tenant, status)` | `list[RecallRecord]` |
| `product_recall(recall_id, serials, tenant, action)` | processing dict |
| `register_wda(...)` | `WholesaleDistributionAuthorisation` |
| `grant_wda(id, tenant, granted, expiry)` | `WholesaleDistributionAuthorisation` |
| `check_wda_expiry(tenant)` | list of expiry alert dicts |
| `list_wda(tenant)` | `list[WholesaleDistributionAuthorisation]` |
| `record_gdp_deviation(...)` | `GdpDeviationRecord` |
| `list_gdp_deviations(tenant)` | `list[GdpDeviationRecord]` |
| `gdp_inspection(distributor, date, findings, tenant, ...)` | inspection dict |
| `wholesale_order(wholesaler, products, quantities, tenant, ...)` | order dict |
| `authorised_distributor_check(distributor, tenant)` | authorisation dict |
| `returns_processing(return_id, qty, reason, condition, tenant, ...)` | return record dict |
| `distribution_analytics(period, tenant)` | KPI dict |
| `regulatory_reporting_distribution(period, jurisdiction, tenant)` | report dict |
| `dashboard_summary(tenant)` | dashboard dict |

### Async Methods (v1.1)
| Method | Returns |
|--------|---------|
| `async_create_shipment(payload)` | `Shipment` |
| `async_dispatch_shipment(...)` | `Shipment` |
| `async_deliver_shipment(...)` | `Shipment` |
| `calculate_mkt(temperature_log, tenant, Ea, ref_temp)` | MKT dict |
| `ingest_cold_chain_telemetry(id, tenant, readings, device, auto)` | telemetry dict |
| `propagate_recall_notification(recall_id, tenant, network, channel, by)` | dispatch dict |
| `validate_aggregation_hierarchy(tenant, sscc)` | hierarchy dict |
| `initiate_wda_renewal(wda_id, tenant, by, notes)` | renewal task dict |
| `gdp_risk_score(distributor, tenant, lookback_days)` | risk dict |
| `supply_chain_integrity_check(shipment_id, tenant)` | integrity dict |
| `async_regulatory_report(period, jurisdiction, tenant, ...)` | extended report dict |
| `bulk_serialise_products(tenant, specs, created_by)` | bulk result dict |

---

## Configuration

All keys are tenant-scoped. Set via the `conf` capability or env vars prefixed with `PHARMA_DIS_`.

| Key | Description | Default |
|-----|-------------|---------|
| `recalls.timeline_hours.class_i` | Class I recall notification deadline | 24 |
| `recalls.timeline_hours.class_ii` | Class II recall notification deadline | 72 |
| `cold_chain.temperature_monitoring_required` | Mandatory for cold-chain products | true |
| `wda.renewal_alert_days` | Days before WDA expiry for renewal alert | 90 |
| `mkt.default_activation_energy_kj` | Default Ea for MKT calculations | 83.14 |
| `serialisation.gtin_check_digit_validation` | Enforce GTIN Mod-10 check on ingest | true |
| `gdp.risk_score.critical_threshold` | GDP risk score triggering auto-suspension | 75 |

---

## Streaming Events

| Event | Trigger |
|-------|---------|
| `shipment_dispatched` | Shipment status → dispatched |
| `shipment_delivered` | Shipment status → delivered |
| `shipment_exception` | Shipment status → exception |
| `cold_chain_excursion_detected` | `report_excursion` called |
| `cold_chain_telemetry_ingested` | `ingest_cold_chain_telemetry` called |
| `temperature_breach_escalated` | Severity == critical |
| `mkt_calculated` | `calculate_mkt` called |
| `serialisation_created` | New serialisation record |
| `serialisation_verified` | Successful verification |
| `serialisation_violation_detected` | Failed verification or decommissioned |
| `bulk_serialisation_completed` | Bulk serialise batch done |
| `aggregation_hierarchy_validated` | GS1 hierarchy validated |
| `recall_initiated` | `initiate_recall` called |
| `recall_serials_processed` | `product_recall` called |
| `recall_notification_dispatched` | Per-entity in propagation |
| `recall_propagation_completed` | Full propagation done |
| `recall_completed` | `complete_recall` called |
| `gdp_deviation_recorded` | `record_gdp_deviation` called |
| `gdp_inspection_completed` | `gdp_inspection` called |
| `gdp_critical_finding_raised` | Critical finding in inspection |
| `gdp_risk_score_computed` | `gdp_risk_score` called |
| `wda_registered` | `register_wda` called |
| `wda_granted` | `grant_wda` called |
| `wda_expiring` | `check_wda_expiry` hit |
| `wda_renewal_initiated` | `initiate_wda_renewal` called |
| `supply_chain_integrity_checked` | `supply_chain_integrity_check` called |
| `return_processed` | `returns_processing` called |
| `distribution_analytics_generated` | `distribution_analytics` called |
| `regulatory_distribution_report_generated` | Regulatory report generated |

All events are appended to `_audit_events` with:
```json
{
  "tenant_id": "...",
  "event_type": "...",
  "reference_id": "...",
  "processor": "bytewax",
  "stream": "apg.pharma.dis.lifecycle"
}
```

---

## Interoperability

```apg
use pharma_dis;
```

Composition links:
- **pharma_mfg** → sends released batches for dispatch
- **pharma_rec** → receives recall and post-market surveillance events
- **pharma_qms** → receives GDP deviations for CAPA linkage
- **intel** → receives analytics events for demand forecasting
- **audl** → receives all audit events for GDP-compliant trail
- **ntfy** → receives WDA expiry and recall escalation notifications

---

## Further Reading

- `service.py` — Business logic implementation (sync + async methods)
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 prioritised engineering improvements
- `SPECIFICATION.md` — Full capability specification
- `cap_spec.md` — Machine-readable capability contract
