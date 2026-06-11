# Pharmaceutical Supply Chain — User Guide

**Capability ID**: `pharma_sup` | **Domain**: `pharma` | **Version**: `2.0.0`
**Copyright**: © 2025 Datacraft | **Author**: Nyimbi Odero

---

## Overview

`pharma_sup` manages the pharmaceutical supply chain end-to-end: API/excipient supplier qualification,
Contract Manufacturing Organisation (CMO) lifecycle, demand planning and S&OP, import licensing,
supply security monitoring, purchase order management, supply contracts, and — in v2 — GS1 serialisation,
cold chain temperature monitoring, GDP pre-shipment compliance, product recall management, supplier
performance scorecards, proactive shortage risk prediction, and regulatory intelligence feed ingestion.

---

## Installation

```bash
pip install apg-pharma-sup
```

---

## Quick Start

```python
import asyncio
from datetime import datetime, timedelta
from apg_pharma_sup import PharmaceuticalSupplyChainService
from apg_pharma_sup.models import SupplierCreate

svc = PharmaceuticalSupplyChainService(tenant_id="datacraft", actor_id="procurement_user")

# Create and qualify a supplier
payload = SupplierCreate(
    tenant_id="datacraft",
    supplier_code="SUP-001",
    name="Acme API Ltd",
    supplier_type="api_manufacturer",
    country="IN",
    created_by="procurement_user",
)
supplier = svc.create_supplier(payload)
qualified = svc.qualify_supplier(
    supplier.id, "datacraft",
    quality_agreement_reference="QA-2026-001",
    audit_date=datetime.utcnow(),
    approved_materials=["paracetamol_api"],
)

# Place an order
order = svc.place_order(
    tenant_id="datacraft",
    po_number="PO-2026-001",
    order_type="api_sourcing",
    supplier_id=qualified.id,
    product_id="paracetamol_api",
    quantity=500.0,
    unit_of_measure="kg",
    created_by="procurement_user",
    expected_delivery=datetime.utcnow() + timedelta(days=30),
    transport_condition="controlled_ambient",
)
```

---

## Core Workflows

### Supplier Qualification (Approved Supplier List)

```python
# 1. Create supplier
supplier = svc.create_supplier(SupplierCreate(
    tenant_id="t1", supplier_code="SUP-API-001", name="BioSynth GmbH",
    supplier_type="api_manufacturer", country="DE", created_by="qm_user"
))

# 2. Qualify and add to ASL
qualified = svc.qualify_supplier(
    supplier.id, "t1",
    quality_agreement_reference="QA-2026-042",
    audit_date=datetime.utcnow(),
    approved_materials=["ibuprofen_api", "ibuprofen_excipient"]
)

# 3. View Approved Supplier List
asl = svc.list_suppliers("t1", qualified_only=True)

# 4. Suspend if needed
svc.suspend_supplier(supplier.id, "t1", reason="GMP certificate expired")
```

### CMO Management

```python
cmo = svc.activate_cmo(
    tenant_id="t1",
    cmo_code="CMO-001",
    name="PharmaMake PLC",
    cmo_type="formulation",
    supplier_id=supplier.id,
    technical_agreement_reference="TA-2026-001",
    quality_agreement_reference="QA-CMO-2026-001",
    created_by="qm_user",
)

# Place a CMO manufacturing order
mfg_order = svc.cmo_order(
    cmo_id=cmo.id,
    product_id="ibuprofen_tablet_400mg",
    batch_size=100000.0,
    delivery_date=datetime.utcnow() + timedelta(days=45),
    tenant_id="t1",
    batch_count=3,
    packaging_spec="HDPE bottle 100-count",
    technical_agreement_ref="TA-2026-001",
)
```

### Demand Planning & S&OP

```python
forecast = svc.demand_planning(
    product_id="ibuprofen_tablet_400mg",
    forecast_periods=12,
    tenant_id="t1",
    method="statistical",
    base_demand=120000.0,
    growth_rate=0.02,
    seasonality_factors={"M01": 1.2, "M07": 0.85},
)

# Approve in S&OP gate
approved = svc.approve_sop(forecast.id, "t1")
```

### Import Licensing

```python
# Apply
lic = svc.import_licence_application(
    product_id="ibuprofen_api",
    country="KE",
    quantity=5000.0,
    tenant_id="t1",
    license_type="standard_import",
    issuing_authority="PPB Kenya",
)

# Grant
granted = svc.grant_import_license(
    lic.id, "t1",
    granted_date=datetime.utcnow(),
    expiry_date=datetime.utcnow() + timedelta(days=365),
)

# Check active license before import
active = svc.check_import_license_active("t1", "ibuprofen_api", "KE")

# Monitor expiry (90-day window)
alerts = svc.check_import_license_expiry("t1")
```

### Supply Security Monitoring

```python
svc.update_supply_security(
    tenant_id="t1",
    product_id="ibuprofen_api",
    supply_status="secure",
    risk_level="low",
    primary_supplier_id=supplier.id,
    created_by="supply_planner",
    dual_sourced=True,
    inventory_days=90.0,
)

# Monitor critical medicines
monitor = svc.security_of_supply_monitoring(
    critical_medicines=["ibuprofen_api", "paracetamol_api"],
    tenant_id="t1",
    inventory_threshold_days=30.0,
)
```

---

## v2 Extended Workflows

### GS1 Serialisation (FMD / DSCSA)

```python
# Commission serials for a batch
batch = await svc.serialise_batch(
    tenant_id="t1",
    product_id="ibuprofen_tablet_400mg",
    batch_number="BN-2026-0042",
    gtin="05901234123457",
    lot_number="L2026042",
    expiry_date=datetime(2028, 6, 30),
    quantity=50000,
    created_by="production_officer",
)

# Verify at point of dispense (FMD check)
result = await svc.verify_serial(
    tenant_id="t1",
    gtin="05901234123457",
    serial_number=batch["serial_numbers"][0],
    lot_number="L2026042",
    expiry_date=datetime(2028, 6, 30),
)
# result["verification_status"] == "verified"
```

### Cold Chain Monitoring

```python
# Record temperature readings from data logger
reading = await svc.record_temperature_reading(
    tenant_id="t1",
    shipment_id="SHIP-2026-001",
    logger_device_id="LOGGER-42",
    temperature_c=7.5,
    humidity_pct=55.0,
    recorded_at=datetime.utcnow(),
    setpoint_min_c=2.0,
    setpoint_max_c=8.0,
)

# After receipt, evaluate excursion impact against stability budget
impact = await svc.evaluate_excursion_impact(
    tenant_id="t1",
    shipment_id="SHIP-2026-001",
    product_stability_budget_hours=48.0,  # ICH Q1A stability data
)
if impact["quarantine_required"]:
    # Quarantine shipment, raise CAPA
    ...
```

### GDP Pre-Shipment Compliance Gate

```python
clearance = await svc.gdp_compliance_gate(
    tenant_id="t1",
    order_id=order.id,
    carrier_id="DHL-PHARMA",
    transport_mode="reefer_road",
    gdp_category="cold_chain_2_8",
    temperature_logger_commissioned=True,
    documents_present=[
        "commercial_invoice",
        "packing_list",
        "cmr_or_awb",
        "temperature_monitoring_plan",
    ],
)
if not clearance["cleared"]:
    raise ValueError(f"GDP gate failed: {clearance['violations']}")
# Proceed with shipment only if cleared
```

### Product Recall Management

```python
# Initiate Class II recall
recall = await svc.initiate_recall(
    tenant_id="t1",
    product_id="ibuprofen_tablet_400mg",
    lot_numbers=["L2026042", "L2026043"],
    recall_class="Class_II",
    reason="Dissolution specification failure — OOS at 45-min timepoint",
    initiated_by="qp_officer",
    regulatory_agency="EMA",
)

# Track recovery after field communications sent
updated_recall = await svc.track_recall_progress(
    tenant_id="t1",
    recall_id=recall["id"],
    units_distributed=48500,
    units_recovered=46300,
)
# updated_recall["recovery_rate_pct"] and effectiveness_status
```

### Supplier Performance Scorecard

```python
scorecard = await svc.calculate_supplier_scorecard(
    tenant_id="t1",
    supplier_id=supplier.id,
    evaluation_period_months=12,
)
# scorecard["weighted_score"], scorecard["rating"], scorecard["requalification_triggered"]
```

Score thresholds:
| Score | Rating | Action |
|-------|--------|--------|
| >= 90 | preferred | No action |
| 70–89 | acceptable | Continue monitoring |
| < 70 | at_risk | Automatic requalification workflow triggered |

### Shortage Risk Prediction

```python
risk = await svc.predict_shortage_risk(
    tenant_id="t1",
    product_id="ibuprofen_api",
    horizon_days=90,
)
# risk["risk_probability"], risk["action_tier"], risk["recommendation"]
```

Action tiers:
| Probability | Tier | Recommended Action |
|-------------|------|--------------------|
| >= 0.75 | critical_intervention | Emergency alternate source, notify authority |
| 0.50–0.74 | proactive_build | Build safety stock to 60+ days, qualify alternate |
| 0.25–0.49 | monitor | Increase monitoring cadence to weekly |
| < 0.25 | normal | No immediate action |

### Regulatory Intelligence Feed

```python
intel = await svc.ingest_regulatory_intelligence(
    tenant_id="t1",
    product_portfolio=["ibuprofen_api", "paracetamol_api", "amoxicillin_api"],
    source="ema",  # or "fda" / "who"
)
# intel["alerts_created"], intel["alerts"]
# Supply security records auto-updated for matched products
```

---

## Dashboard & Analytics

```python
# Real-time dashboard
dash = svc.dashboard_summary("t1")

# Supply chain KPIs for a period
kpis = svc.supply_analytics("2026-Q1", "t1")

# Regulatory supply report for a jurisdiction
report = svc.regulatory_supply_reporting("2026-Q1", "KE", "t1")
```

---

## Shortage Management

```python
shortage = svc.shortage_management(
    drug_id="ibuprofen_api",
    shortage_type="api_shortage",
    mitigation="Activate contingency supplier CON-SUP-002 and build inventory to 120 days",
    tenant_id="t1",
    estimated_duration_days=60,
    regulatory_notification_required=True,
    contingency_supplier_id="CON-SUP-002",
)
```

Supported shortage types: `manufacturing_delay`, `api_shortage`, `demand_surge`,
`regulatory_hold`, `logistics`, `force_majeure`.

---

## Supply Risk Assessment

```python
assessment = svc.supply_risk_assessment(
    product_id="ibuprofen_api",
    supply_chain_map={
        "dual_sourced": False,
        "nodes": [
            {"id": "api-supplier-01", "type": "api_manufacturer",
             "risk_score": 7, "criticality": 9, "mitigation": "Qualify alternate"},
            {"id": "excipient-01", "type": "excipient_supplier",
             "risk_score": 3, "criticality": 4, "mitigation": "Dual source in place"},
        ],
    },
    tenant_id="t1",
    assessment_method="fmea",
)
```

---

## Service Method Reference

### Synchronous Methods

| Method | Description |
|--------|-------------|
| `create_supplier()` | Create supplier record |
| `qualify_supplier()` | Qualify supplier, add to ASL |
| `suspend_supplier()` | Suspend supplier from ASL |
| `get_supplier()` | Fetch single supplier |
| `list_suppliers()` | List suppliers (optional qualified_only filter) |
| `activate_cmo()` | Activate Contract Manufacturing Organisation |
| `list_cmos()` | List CMOs |
| `cmo_order()` | Place CMO manufacturing order |
| `create_forecast()` | Create demand forecast |
| `approve_sop()` | Mark forecast S&OP approved |
| `demand_planning()` | Generate demand plan with growth/seasonality |
| `list_forecasts()` | List demand forecasts |
| `apply_import_license()` | Apply for import license |
| `import_licence_application()` | Apply for import license (convenience form) |
| `grant_import_license()` | Mark license granted |
| `check_import_license_expiry()` | 90-day expiry alerts |
| `check_import_license_active()` | Active license check for product+region |
| `list_import_licenses()` | List all import licenses |
| `update_supply_security()` | Update product supply security record |
| `list_supply_security()` | List supply security records |
| `security_of_supply_monitoring()` | Monitor critical medicines |
| `shortage_management()` | Declare and manage a drug shortage |
| `supply_risk_assessment()` | FMEA-based supply chain risk assessment |
| `place_order()` | Place ASL-gated purchase order |
| `receive_order()` | Receive order with CoA |
| `list_orders()` | List purchase orders |
| `api_sourcing()` | Place API-specific sourcing order |
| `create_contract()` | Create supply contract |
| `approve_contract()` | Approve supply contract |
| `check_contract_expiry()` | 60-day contract expiry alerts |
| `list_contracts()` | List supply contracts |
| `customs_clearance()` | Initiate customs clearance |
| `supply_analytics()` | Generate KPI analytics |
| `regulatory_supply_reporting()` | Regulatory jurisdiction supply report |
| `dashboard_summary()` | Tenant-level dashboard |

### Async Methods

| Method | Description |
|--------|-------------|
| `serialise_batch()` | GS1-EPCIS unit serialisation for a batch |
| `verify_serial()` | FMD/DSCSA serial verification at dispense |
| `record_temperature_reading()` | Cold chain temperature data point |
| `evaluate_excursion_impact()` | MKT excursion stability budget evaluation |
| `gdp_compliance_gate()` | Pre-shipment GDP compliance check |
| `initiate_recall()` | Class I/II/III product recall initiation |
| `track_recall_progress()` | Recall effectiveness reconciliation |
| `calculate_supplier_scorecard()` | Weighted KPI scorecard with re-qualification trigger |
| `predict_shortage_risk()` | 90-day forward shortage risk probability |
| `ingest_regulatory_intelligence()` | EMA/FDA/WHO regulatory feed ingestion |
| `export_records()` | Export tenant records (json/csv) |
| `health_check()` | Service health check |
| `compliance_report()` | GxP compliance attestation |
| `bulk_create_records()` | Bulk record import |
| `analytics_summary()` | Analytics summary |
| `ml_supply_chain_risk()` | AI-powered supply chain risk (Ollama) |

---

## Configuration Keys

All keys are tenant-scoped and set via the `conf` capability or `PHARMA_SUP_*` environment variables.

| Key | Description | Default |
|-----|-------------|---------|
| `suppliers.audit_cycle_months` | Supplier re-audit frequency | 24 |
| `import_licensing.renewal_alert_days` | Days before license expiry for alert | 90 |
| `contracts.renewal_alert_days` | Days before contract expiry for alert | 60 |
| `supply_security.dual_sourcing_threshold` | Risk level requiring dual sourcing | high |
| `cold_chain.default_setpoint_min_c` | Default cold chain minimum temperature | 2.0 |
| `cold_chain.default_setpoint_max_c` | Default cold chain maximum temperature | 8.0 |
| `shortage_prediction.horizon_days` | Forward shortage prediction window | 90 |
| `supplier_scorecard.requalification_threshold` | Score below which requalification fires | 70 |

---

## Composability

`pharma_sup` integrates with other APG capabilities:

```apg
use pharma_sup;
```

| Integration | Description |
|-------------|-------------|
| `pharma_mfg` | Qualified suppliers feed material receipt; CMO orders link to batch genealogy |
| `pharma_dis` | Demand forecasts drive inventory planning; import licenses gate import shipments |
| `pharma_rec` | Supply security data informs risk management plans |
| `pharma_qms` | Recall management integrates with deviation and CAPA workflows |

---

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints (Flask-AppBuilder blueprints)
- `views.py` — Flask-AppBuilder view models
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 architectural improvement specifications
- `cap_spec.md` — Capability specification
- `SPECIFICATION.md` — Detailed functional specification
