# Fuel Management

## Overview
The Fuel Management capability covers fuel procurement, transaction recording with odometer capture, fuel card management and reconciliation, bunker management, carbon footprint calculation across GHG Protocol and ISO 14064 standards, and storage tank monitoring. Built-in phantom fill and theft pattern detection protect against fraud.

## Capability ID
`transport_fue`

## Provides
- fuel_procurement_workflow: Bulk, contract, and spot fuel procurement management
- fuel_consumption_tracking_workflow: Per-vehicle/driver transaction recording with odometer
- bunker_management_workflow: Marine and aviation bunker supply management
- fuel_card_reconciliation_workflow: Multi-provider card reconciliation with fraud detection
- carbon_footprint_reporting_workflow: Scope 1 carbon emission calculation and reporting

## Requires
- auth, audl, mten, conf: Core platform services
- ntfy: Low stock and fraud alerts
- wflo: Procurement approval workflow
- moni: Tank level and anomaly monitoring
- comp: Carbon reporting compliance
- mqeb: Event streaming
- schd: Scheduled reconciliation

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| fuel_cards.reconciliation_frequency | Reconciliation cadence | daily |
| fuel_cards.fraud_detection_enabled | Fraud detection | true |
| governance.phantom_fill_detection | Phantom fill guard | true |
| carbon.scope1_reporting | Scope 1 CO2 | true |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /transport-fuel/transactions | GET | Transaction log | transport_fue:transactions |
| /transport-fuel/cards | GET | Fuel cards | transport_fue:cards |
| /transport-fuel/cards/reconciliation | GET | Card reconciliation | transport_fue:cards |
| /transport-fuel/carbon | GET | Carbon footprint | transport_fue:carbon |
| /transport-fuel/storage | GET | Tank levels | transport_fue:storage |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| phantom_fill_detection | Phantom fill pattern | deny |
| fuel_theft_alert_enabled | Theft pattern detected | deny |
| transaction_quantity_positive | Quantity <= 0 | deny |
| transaction_odometer_required | Odometer absent | deny |
| cross_tenant_fuel_denied | Cross-tenant write | deny |

## Data Models
- FuelProcurement: id, procurement_type, supplier_id, fuel_type, quantity_litres, unit_price, currency
- FuelTransaction: id, transaction_type, vehicle_id, driver_id, fuel_type, quantity_litres, odometer_km
- FuelCard: id, provider, card_number_masked, vehicle_id, driver_id, active
- FuelCardReconciliation: id, card_id, period, expected_total, actual_total, discrepancy
- CarbonEmissionRecord: id, vehicle_id, standard, fuel_type, quantity_litres, co2_kg
- FuelStorageTank: id, storage_type, location, capacity_litres, current_level_litres

## Streaming Events
- fuel_procurement_recorded, fuel_transaction_recorded, fuel_card_reconciled
- carbon_emission_calculated, fuel_storage_updated, efficiency_alert_raised, fuel_theft_detected

## Edge Cases Handled
- Phantom fill detection fires independently of theft pattern detection
- Zero-quantity transactions are blocked at rule engine level
- Card reconciliation discrepancy is automatically calculated as actual - expected
- Odometer reading is mandatory for per-kilometre efficiency calculation
- Bunker supply uses the same transaction model with marine_fuel fuel type

## Composability Notes
Depends on `transport_fle` for vehicle and driver identity. Carbon emission records feed into enterprise sustainability reporting via `comp`. Reconciliation anomalies trigger alerts through `ntfy` and are audited via `audl`.
