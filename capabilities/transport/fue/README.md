# Fuel Management

**Capability ID**: `transport_fue` | **Domain**: `transport` | **Version**: `1.1.0`
**Copyright**: © 2025 Datacraft | **Author**: Nyimbi Odero

## Overview

The Fuel Management capability covers the full lifecycle of fleet fuel operations: procurement (bulk, contract, spot), per-vehicle transaction recording with odometer capture, multi-provider fuel card management and reconciliation, storage tank monitoring with demand forecasting, carbon footprint calculation (Scope 1/2/3), eco-driving analytics, fraud detection, and supplier contract modelling.

Built-in phantom fill detection, fill-while-moving guards, and ML-ready fraud flagging protect against the 12–25% of fleet fuel spend typically lost to misuse.

## Capability ID

`transport_fue`

## Provides

- `fuel_procurement_workflow`: Bulk, contract, and spot procurement management with volume discount tiers and NPV-based contract evaluation
- `fuel_consumption_tracking_workflow`: Per-vehicle/driver transaction recording with odometer, efficiency trending, and fleet benchmark comparison
- `bunker_management_workflow`: Marine and aviation bunker supply management
- `fuel_card_reconciliation_workflow`: Multi-provider card reconciliation with real-time limit enforcement and fraud detection
- `carbon_footprint_reporting_workflow`: Scope 1/2/3 emission calculation, net-zero pathway modelling, and offset cost estimation
- `fuel_analytics_workflow`: Fleet-wide KPIs, driver eco-scoring, price benchmarking, and demand forecasting

## Requires

- `auth`, `audl`, `mten`, `conf`: Core platform services
- `ntfy`: Low-stock, fraud, and limit-exceeded alerts
- `wflo`: Procurement approval workflow
- `moni`: Tank level and anomaly monitoring
- `comp`: Carbon reporting compliance
- `mqeb`: Event streaming (bytewax JetStream)
- `schd`: Scheduled reconciliation and reorder jobs

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| `fuel_cards.reconciliation_frequency` | Reconciliation cadence | `daily` |
| `fuel_cards.fraud_detection_enabled` | Rule-based fraud detection | `true` |
| `governance.phantom_fill_detection` | Phantom fill guard | `true` |
| `carbon.scope1_reporting` | Scope 1 CO2 | `true` |
| `procurement.bulk_discount_tiers` | Volume discount thresholds | `5000/20000 L` |
| `analytics.eco_score_threshold` | Eco-compliant score floor | `75` |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| `/transport-fuel/transactions` | GET | Transaction log | `transport_fue:transactions` |
| `/transport-fuel/cards` | GET | Fuel cards | `transport_fue:cards` |
| `/transport-fuel/cards/reconciliation` | GET | Card reconciliation | `transport_fue:cards` |
| `/transport-fuel/carbon` | GET | Carbon footprint | `transport_fue:carbon` |
| `/transport-fuel/storage` | GET | Tank levels | `transport_fue:storage` |
| `/transport-fuel/analytics` | GET | Fleet KPIs | `transport_fue:analytics` |
| `/transport-fuel/efficiency` | GET | Fleet efficiency benchmark | `transport_fue:analytics` |
| `/transport-fuel/forecast` | GET | Demand forecast | `transport_fue:storage` |

## Key Service Methods

### Core CRUD

| Method | Description |
|--------|-------------|
| `create_procurement(...)` | Record a procurement order |
| `record_transaction(...)` | Record a fuel transaction with odometer |
| `register_fuel_card(...)` | Register a card with a provider |
| `reconcile_fuel_card(...)` | Reconcile card statement vs actuals |
| `record_carbon_emission(...)` | Log a carbon emission record |
| `register_storage_tank(...)` | Register a depot storage tank |
| `register_fuel_agent(...)` | Register an AI agent |

### Async Operations

| Method | Description |
|--------|-------------|
| `record_fuel_fill(...)` | Record a station fill with auto fraud pre-check |
| `fuel_card_transaction(...)` | Process a card transaction at a merchant |
| `monthly_fuel_budget(...)` | Budget vs actual for a vehicle/month |
| `fuel_efficiency_report(...)` | km/L and L/100km for a vehicle/period |
| `bulk_fuel_procurement(...)` | Bulk order with volume discount calculation |
| `fuel_stock_level(...)` | Current stock per fuel type at a depot |
| `mpg_trend(...)` | km/L trend across last N calendar months |
| `carbon_footprint(...)` | Vehicle CO2 for a period (IPCC AR6 factors) |
| `fuel_fraud_detection(...)` | Batch fraud heuristic scan |
| `fuel_analytics(...)` | Fleet-wide KPIs for a period |
| `supplier_performance(...)` | Supplier delivery and price consistency |
| `fuel_budget_variance(...)` | Fleet spend vs budget |
| `tank_reorder_alert(...)` | Reorder alerts by threshold |
| `driver_fuel_ranking(...)` | Drivers ranked by consumption |
| `fleet_carbon_report(...)` | Fleet-wide carbon aggregation |
| `export_fuel_data(...)` | Export transaction metadata |
| `health_check()` | Service health status |
| `deactivate_fuel_card(...)` | Deactivate card with reason |
| `fuel_price_benchmark(...)` | Avg price vs market benchmark |
| `update_tank_level(...)` | Update tank fill level |
| `carbon_offset_report(...)` | Offset cost to neutralise fleet emissions |
| `fuel_price_feed(...)` | Live market VWAP comparison by region |
| `driver_eco_score(...)` | Composite eco-driving index (0–100) |
| `evaluate_supplier_contract(...)` | NPV: contract vs spot procurement |
| `net_zero_pathway(...)` | Annual CO2 trajectory to net-zero |
| `plan_fuel_stops(...)` | Optimal route refuel stop planning |
| `enforce_card_limits(...)` | Real-time daily/monthly limit enforcement |
| `forecast_fuel_demand(...)` | Exponential-smoothing depot demand forecast |
| `verify_audit_chain(...)` | Merkle-style audit integrity verification |
| `fleet_efficiency_benchmark(...)` | Fleet median km/L ranking with actions |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| `phantom_fill_detection` | Fill qty > vehicle class max | deny + flag |
| `fuel_theft_alert_enabled` | Theft pattern detected | deny |
| `transaction_quantity_positive` | Quantity <= 0 | deny |
| `transaction_odometer_required` | Odometer absent | deny |
| `cross_tenant_fuel_denied` | Cross-tenant write | deny |
| `card_inactive_blocked` | Card `active=False` | deny |
| `card_limit_exceeded` | Daily or monthly limit breach | deny + audit |

## Data Models

- `FuelProcurement`: procurement_type, supplier_id, fuel_type, quantity_litres, unit_price, currency
- `FuelTransaction`: transaction_type, vehicle_id, driver_id, fuel_type, quantity_litres, odometer_km, card_id
- `FuelCard`: provider, card_number_masked, vehicle_id, driver_id, active, pin_set
- `FuelCardReconciliation`: card_id, period, expected_total, actual_total, discrepancy
- `CarbonEmissionRecord`: vehicle_id, standard, fuel_type, quantity_litres, co2_kg
- `FuelStorageTank`: storage_type, location, capacity_litres, current_level_litres

## Streaming Events

- `fuel_procurement_recorded`, `fuel_transaction_recorded`, `fuel_card_reconciled`
- `carbon_emission_calculated`, `fuel_storage_updated`, `efficiency_alert_raised`
- `fuel_theft_detected`, `card_limit_exceeded_daily`, `card_limit_exceeded_monthly`
- `tank_reorder_alert_raised`, `fuel_data_exported`, `external_integration_sent`

## Emission Factors (IPCC AR6)

| Fuel | kg CO2/litre |
|------|-------------|
| Diesel | 2.68 |
| Petrol | 2.31 |
| LPG | 1.51 |
| CNG | 2.02 |
| HVO | 0.45 |
| Biodiesel | 0.67 |

## Edge Cases Handled

- Phantom fill detection fires independently of theft pattern detection
- Zero-quantity transactions are blocked at rule engine level
- Card reconciliation discrepancy is automatically calculated as `actual - expected`
- Odometer reading is mandatory for per-kilometre efficiency calculation
- Bunker supply uses the same transaction model with `marine_fuel` fuel type
- Multi-currency transactions are preserved as-is; normalisation is opt-in
- Audit chain hashes are stamped lazily on first `verify_audit_chain` call

## Composability

Depends on `transport_fle` for vehicle and driver identity. Carbon emission records feed into enterprise sustainability reporting via `comp`. Reconciliation anomalies trigger alerts through `ntfy` and are audited via `audl`. Demand forecasts integrate with `wflo` to auto-draft procurement orders.

```apg
use transport_fue;
```

## Further Reading

- `service.py` — Business logic and all async methods
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 prioritised enhancement proposals
- `docs/user_guide.md` — Step-by-step operator guide

---

## World-Class Enhancements (v2.0)

- **I1.** Fuel Management — World-Class Improvements
- **I2.** Real-Time Fuel Price Feed Integration
- **I3.** ML-Driven Anomaly Detection for Fraud
- **I4.** Telematics Integration for Phantom Fill Prevention
- **I5.** Predictive Reorder with Demand Forecasting
- **I6.** Fuel Card Lifecycle Automation
- **I7.** Multi-Currency Normalised Reporting
- **I8.** Driver Behaviour Scoring (Eco-Driving Index)
- **I9.** Bulk Contract Procurement Negotiation Support
- **I10.** Scope 1/2/3 Emissions Attribution and Net-Zero Pathway
- **I11.** Event-Sourced Audit Trail with Merkle Integrity
- **I12.** Vendor-Agnostic Fleet Card API Gateway
- **I13.** Geospatial Station Network and Route Fuel Planning
- **I14.** Regulatory Compliance Engine (ADR, OIML, REACH)
- **I15.** Zero-Trust Card PIN Management with HSM Integration

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
