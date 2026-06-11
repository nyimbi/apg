# Fuel Management — User Guide

**Capability ID**: `transport_fue` | **Version**: `1.1.0`
**Copyright**: © 2025 Datacraft | **Author**: Nyimbi Odero

---

## 1. Introduction

The Fuel Management capability (`transport_fue`) provides end-to-end control of fleet fuel operations. It covers procurement, per-vehicle transaction recording, fuel card lifecycle management, storage tank monitoring, carbon footprint reporting, driver eco-scoring, and supplier contract evaluation — all behind a tenant-scoped policy engine.

This guide covers day-to-day operator workflows. For API details see `service.py` and `api.py`.

---

## 2. Installation

```bash
pip install apg-transport-fue
```

Or install from source within the APG monorepo:

```bash
cd capabilities/transport/fue
pip install -e .
```

---

## 3. Service Initialisation

```python
from capabilities.transport.fue.service import FuelManagementService

svc = FuelManagementService(
    tenant_id="acme_fleet",
    actor_id="ops_manager_01",
)
```

All methods that accept `tenant_id` as a keyword argument default to the value set at construction time, so you only need to pass it explicitly when operating across multiple tenants.

---

## 4. Core Workflows

### 4.1 Recording a Fuel Fill

```python
import asyncio

result = asyncio.run(svc.record_fuel_fill(
    vehicle_id="TRK-001",
    litres=180.5,
    unit_price=1.32,
    station="Shell Mombasa Road",
    odometer=42850.0,
    driver_id="DRV-007",
    fuel_type="diesel",
))
print(result["total_cost_usd"], result["over_tank_flag"])
```

`record_fuel_fill` automatically:
- Checks quantity against vehicle-class max fill (HGV: 800 L, car: 80 L, default: 200 L)
- Raises a `fraud_flags` entry if the fill exceeds the class limit
- Records the transaction and emits `fuel_transaction_recorded`

### 4.2 Fuel Card Transaction

```python
result = asyncio.run(svc.fuel_card_transaction(
    card_id="CARD-101",
    vehicle_id="TRK-001",
    amount=210.00,
    merchant="Total Energies Nakuru",
    fuel_type="diesel",
))
```

The method validates card `active` status, infers litres from a pump price estimate, and records the transaction against the card.

### 4.3 Enforcing Card Spend Limits

```python
# Will raise PermissionError if daily or monthly limit would be exceeded
auth = asyncio.run(svc.enforce_card_limits(
    card_id="CARD-101",
    amount_usd=150.0,
    merchant="Total Energies Nakuru",
))
print(auth["daily_remaining_usd"], auth["monthly_remaining_usd"])
```

Call `enforce_card_limits` **before** `fuel_card_transaction` in any authorisation flow. Default limits are 500 USD/day, 5 000 USD/month — override by setting `daily_limit_usd` and `monthly_limit_usd` on the `FuelCard` instance.

---

## 5. Procurement Management

### 5.1 Spot/Bulk Procurement

```python
# Automatic volume discount: <5000L=0%, 5000-20000L=2%, >20000L=4%
order = asyncio.run(svc.bulk_fuel_procurement(
    litres=25_000,
    supplier="TotalEnergies KE",
    delivery_date="2026-06-20",
    fuel_type="diesel",
    unit_price=1.30,
    currency="USD",
))
print(order["net_cost"], order["discount_pct"])  # 31200.0, 4.0
```

### 5.2 Supplier Contract Evaluation

Compare a long-term contract against continuing on spot prices:

```python
analysis = asyncio.run(svc.evaluate_supplier_contract(
    supplier_id="TotalEnergies KE",
    annual_volume_l=500_000,
    spot_price=1.35,
    contract_price=1.26,
    take_or_pay_pct=80.0,
    escalation_pct_pa=2.5,
    contract_years=3,
))
print(analysis["projected_savings"], analysis["contract_recommended"])
```

The NPV model accounts for: annual spot price escalation (5% pa default), contract price escalation (`escalation_pct_pa`), and take-or-pay penalty on shortfall volumes.

### 5.3 Supplier Performance Review

```python
perf = asyncio.run(svc.supplier_performance(
    supplier_id="TotalEnergies KE",
    period="2026-Q1",
))
print(perf["price_consistency"], perf["avg_price_per_litre"])
```

---

## 6. Storage Tank Management

### 6.1 Register a Tank

```python
svc.register_storage_tank(
    tank_id="TANK-A1",
    tenant_id="acme_fleet",
    storage_type="underground",
    location="Nairobi Depot",
    capacity_litres=50_000,
    fuel_type="diesel",
    last_calibrated="2026-01-15",
)
```

### 6.2 Stock Level Check

```python
stock = asyncio.run(svc.fuel_stock_level("Nairobi Depot"))
for s in stock["stock_by_fuel_type"]:
    print(s["fuel_type"], s["fill_pct"], s["low_stock_warning"])
```

### 6.3 Demand Forecast and Reorder

```python
# 30-day demand forecast using exponential smoothing
forecast = asyncio.run(svc.forecast_fuel_demand("Nairobi Depot", horizon_days=30))
for f in forecast["forecasts"]:
    if f["reorder_urgent"]:
        print(f"URGENT: {f['tank_id']} has {f['days_to_empty']} days remaining")

# Threshold-based reorder alert (instant check)
alerts = asyncio.run(svc.tank_reorder_alert("Nairobi Depot", reorder_threshold_pct=25.0))
```

### 6.4 Update Tank Level After Dispensing

```python
await svc.update_tank_level("TANK-A1", new_level_litres=38_400)
```

---

## 7. Analytics and Reporting

### 7.1 Fleet-Wide KPIs

```python
kpis = asyncio.run(svc.fuel_analytics(period="2026-05"))
print(kpis["total_litres"], kpis["avg_price_per_litre"], kpis["fraud_flags_total"])
```

### 7.2 Vehicle Efficiency Report

```python
eff = asyncio.run(svc.fuel_efficiency_report(
    vehicle_id="TRK-001",
    period="2026-05",
))
print(eff["km_per_litre"], eff["below_baseline"])
```

### 7.3 Fleet Efficiency Benchmark

Compares every vehicle in the fleet against the fleet median and industry baseline (8 km/L):

```python
bench = asyncio.run(svc.fleet_efficiency_benchmark(period="2026-05"))
for v in bench["vehicles"][:3]:
    print(v["rank"], v["vehicle_id"], v["km_per_litre"], v["action"])
```

Vehicles more than 15% below fleet median receive `action = "inspect_fuel_system"`.

### 7.4 Fuel Price Feed (Market Benchmarking)

```python
feed = asyncio.run(svc.fuel_price_feed(region="nairobi", fuel_type="diesel"))
print(feed["tenant_vwap_usd_per_l"], feed["delta_usd_per_l"], feed["savings_opportunity_usd"])
```

### 7.5 Monthly Budget Tracking

```python
budget = asyncio.run(svc.monthly_fuel_budget(
    vehicle_id="TRK-001",
    month="2026-05",
    budget_usd=800.0,
))
print(budget["over_budget"], budget["budget_utilisation_pct"])
```

### 7.6 Fleet-Wide Budget Variance

```python
variance = asyncio.run(svc.fuel_budget_variance(
    period="2026-05",
    budget_amount=50_000.0,
))
```

---

## 8. Driver Analytics

### 8.1 Driver Fuel Ranking

```python
ranking = asyncio.run(svc.driver_fuel_ranking(period="2026-05"))
print(ranking["most_efficient_driver"])
for r in ranking["rankings"][:5]:
    print(r["rank"], r["driver_id"], r["total_litres"])
```

### 8.2 Driver Eco Score

Composite eco-driving index (0–100) weighted across km/L vs fleet median (40 pts), fill quantity consistency (30 pts), and fraud-free record (30 pts):

```python
score = asyncio.run(svc.driver_eco_score(driver_id="DRV-007", period="2026-05"))
print(score["eco_score"], score["eco_compliant"])
# breakdown shows contribution of each factor
print(score["breakdown"])
```

A score >= 75 qualifies the driver as eco-compliant. Scores below 50 trigger a recommended driver coaching intervention.

### 8.3 MPG Trend (Monthly)

```python
trend = asyncio.run(svc.mpg_trend(vehicle_id="TRK-001", periods=6))
print(trend["trend_direction"])  # "improving" | "declining" | "stable"
for m in trend["monthly_data"]:
    print(m["month"], m["km_per_litre"])
```

---

## 9. Carbon and Sustainability

### 9.1 Per-Vehicle Carbon Footprint

```python
cf = asyncio.run(svc.carbon_footprint(
    vehicle_id="TRK-001",
    period="2026-05",
    standard="ghg_protocol",
))
print(cf["total_co2_tonnes"], cf["total_co2_kg"])
```

Uses IPCC AR6 emission factors per fuel type (diesel: 2.68 kg CO2/L, HVO: 0.45 kg CO2/L, etc.).

### 9.2 Fleet Carbon Report

```python
fleet_cf = asyncio.run(svc.fleet_carbon_report(period="2026-05"))
for v in fleet_cf["by_vehicle"][:3]:
    print(v["vehicle_id"], v["co2_kg"])
```

### 9.3 Carbon Offset Cost

```python
offset = asyncio.run(svc.carbon_offset_report(period="2026-05"))
print(offset["estimated_offset_cost_usd"])  # at ~$15/tonne CO2
```

### 9.4 Net-Zero Pathway Modelling

Model the emission trajectory to net zero given EV adoption rate and biofuel blending:

```python
pathway = asyncio.run(svc.net_zero_pathway(
    target_year=2035,
    ev_adoption_pct_pa=12.0,
    biofuel_blend_pct=7.0,
))
print(pathway["total_reduction_pct"], pathway["net_zero_achievable"])
for yr in pathway["trajectory"]:
    print(yr["year"], yr["projected_co2_tonnes"], yr["ev_fleet_pct"])
```

---

## 10. Fraud Detection

### 10.1 Batch Fraud Scan

```python
txns_to_scan = [
    {"vehicle_id": "TRK-001", "quantity_litres": 250, "transaction_at": "2026-05-10T08:00", "speed_kmh": 0},
    {"vehicle_id": "TRK-001", "quantity_litres": 80,  "transaction_at": "2026-05-10T08:30", "speed_kmh": 0},
    {"vehicle_id": "TRK-002", "quantity_litres": 100, "transaction_at": "2026-05-11T14:00", "speed_kmh": 72},
]
flags = asyncio.run(svc.fuel_fraud_detection(txns_to_scan))
print(flags["flags_raised"], flags["high_severity_flags"])
for f in flags["flags"]:
    print(f["rule"], f["severity"])
```

Heuristic rules: over-tank fill, fill-while-moving (speed > 5 km/h), duplicate fill within the same minute.

---

## 11. Route Fuel Planning

Compute the cheapest set of refuel stops along a multi-waypoint route while maintaining a 10% fuel reserve:

```python
waypoints = [
    {"lat": -1.286, "lon": 36.820},   # Nairobi
    {"lat": -0.512, "lon": 37.265, "station_id": "ST-Thika",   "price": 1.28},
    {"lat":  0.518, "lon": 35.270, "station_id": "ST-Nakuru",  "price": 1.31},
    {"lat":  0.091, "lon": 34.768, "station_id": "ST-Kisumu",  "price": 1.33},
    {"lat": -0.102, "lon": 34.761},   # Destination
]
plan = asyncio.run(svc.plan_fuel_stops(
    route_waypoints=waypoints,
    vehicle_range_km=600,
    current_level_l=120,
    tank_capacity_l=200,
    consumption_l_per_100km=28,
))
print(plan["total_fuel_cost_usd"], plan["stop_count"])
for stop in plan["stops"]:
    print(stop["station_id"], stop["fill_litres"], stop["cost_usd"])
```

---

## 12. Audit Trail Integrity

### 12.1 Verify Audit Chain

All service operations emit audit events. Call `verify_audit_chain` to confirm tamper-evidence:

```python
result = asyncio.run(svc.verify_audit_chain())
print(result["verified"], result["events_checked"], result["chain_tip_hash"])
```

Verifying a slice:

```python
result = asyncio.run(svc.verify_audit_chain(from_index=0, to_index=50))
```

Chain hashes are stamped onto events lazily on first verification. Subsequent calls validate against stored hashes.

---

## 13. Data Export and Integration

### 13.1 Export Fuel Data

```python
export = asyncio.run(svc.export_fuel_data(period="2026-05", format="csv"))
print(export["download_ref"], export["record_count"])
```

### 13.2 External Integration

Push transaction records to a fleet card or telematics provider:

```python
result = asyncio.run(svc.integration_external(
    provider="WEX",
    payload={"records": transactions_list},
))
print(result["integration_ref"], result["status"])
```

---

## 14. Health Check

```python
health = asyncio.run(svc.health_check())
print(health["status"])  # "healthy"
```

---

## 15. Configuration Reference

All keys are tenant-scoped and can be overridden via the `conf` capability or environment variables prefixed `TRANSPORT_FUE_`:

| Variable | Description | Default |
|----------|-------------|---------|
| `TRANSPORT_FUE_PHANTOM_FILL_DETECTION` | Enable phantom fill guard | `true` |
| `TRANSPORT_FUE_FRAUD_DETECTION` | Enable batch fraud heuristics | `true` |
| `TRANSPORT_FUE_RECONCILIATION_FREQUENCY` | Card reconciliation cadence | `daily` |
| `TRANSPORT_FUE_SCOPE1_REPORTING` | Enable Scope 1 CO2 reporting | `true` |
| `TRANSPORT_FUE_CARD_DAILY_LIMIT_USD` | Default card daily limit | `500.0` |
| `TRANSPORT_FUE_CARD_MONTHLY_LIMIT_USD` | Default card monthly limit | `5000.0` |
| `TRANSPORT_FUE_RESERVE_PCT` | Minimum tank reserve % | `10.0` |

---

## 16. Composability

`transport_fue` is designed to compose with other APG capabilities:

```apg
# Example: procurement triggers approval workflow
use transport_fue;
use wflo;

on transport_fue.fuel_procurement_recorded
  -> wflo.start_approval(ref=$procurement_id, policy="fuel_spend_approval");
```

Key integration points:
- `transport_fle`: vehicle and driver identity resolution
- `comp`: enterprise sustainability reporting (carbon records)
- `ntfy`: fraud alerts, low-stock warnings, card limit events
- `audl`: tamper-evident audit log persistence
- `mqeb`: bytewax / Kafka event streaming

---

## 17. Further Reading

- `service.py` — Complete implementation of all async methods
- `models.py` — Dataclass models for all entities
- `api.py` — REST API endpoints and request/response schemas
- `views.py` — Flask-AppBuilder view helpers
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 prioritised enhancement proposals
- `SPECIFICATION.md` — Formal capability specification
