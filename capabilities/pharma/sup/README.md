# Pharmaceutical Supply Chain

## Overview
Manages the pharmaceutical supply chain from active ingredient sourcing through CMO management, demand planning, import licensing, supply security monitoring, purchase order management, and supply contract lifecycle. Enforces approved supplier list requirements, quality agreement obligations, import license verification, and dual sourcing requirements for high-risk products.

## Capability ID
`pharma_sup`

## Provides
- active_ingredient_sourcing_workflow: API and excipient supplier qualification and ASL management
- cmo_management_workflow: CMO activation with technical/quality/manufacturing agreement enforcement
- demand_planning_workflow: Statistical and consensus forecast generation with S&OP approval
- import_licensing_workflow: Import permit application, grant, and 90-day renewal alert
- supply_security_monitoring_workflow: Risk-tiered supply status monitoring with shortage reporting
- supplier_qualification_workflow: Full qualification lifecycle from unqualified to ASL inclusion
- purchase_order_workflow: ASL-gated PO placement with CoA receipt enforcement
- supply_contract_workflow: Contract approval, version control, and 60-day renewal alert
- approved_supplier_list_workflow: ASL maintenance with qualification status enforcement
- supply_risk_workflow: Dual sourcing requirement enforcement for high-risk products

## Requires
| Capability | Reason |
|------------|--------|
| auth | Role-based access for procurement and supply chain |
| audl | Supplier qualification audit trail |
| mten | Company-level supply chain data isolation |
| conf | Qualification cycle and alert threshold configuration |
| ntfy | License expiry and supply risk notifications |
| wflo | Contract and qualification approval workflow |
| comp | GDP and import compliance enforcement |
| moni | Supply security real-time monitoring |
| schd | License renewal and audit scheduling |
| mqeb | Event streaming for supply chain events |

## Configuration
| Key | Description | Default |
|-----|-------------|---------|
| suppliers.audit_cycle_months | Supplier reaudit frequency | 24 |
| import_licensing.renewal_alert_days | Days before license expiry for alert | 90 |
| contracts.renewal_alert_days | Days before contract expiry for alert | 60 |
| supply_security.dual_sourcing_threshold | Risk level requiring dual sourcing | high |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /pharma-sup/api/v1/suppliers | POST | Create supplier | pharma_sup:suppliers |
| /pharma-sup/api/v1/suppliers/<id>/qualify | POST | Qualify supplier | pharma_sup:suppliers |
| /pharma-sup/api/v1/asl | GET | Approved Supplier List | pharma_sup:asl |
| /pharma-sup/api/v1/cmo | POST | Activate CMO | pharma_sup:cmo |
| /pharma-sup/api/v1/orders | POST | Place purchase order | pharma_sup:orders |
| /pharma-sup/api/v1/import-licenses | POST | Apply for import license | pharma_sup:import |
| /pharma-sup/api/v1/import-licenses/expiry-alerts | GET | License expiry alerts | pharma_sup:import |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| approved_supplier_list_required | Order placed for non-ASL supplier | Deny — qualify supplier for ASL |
| quality_agreement_required | Supplier activated without quality agreement | Deny — sign quality agreement |
| cmo_technical_agreement_required | CMO activated without technical agreement | Deny — sign technical agreement |
| import_license_required | Import attempted without active license | Deny — obtain import license |
| high_risk_dual_sourcing_required | High-risk product without dual source | Deny — identify alternate supplier |
| shortage_reporting_required | Shortage status set without regulatory notification | Deny — notify authority |
| order_coa_required | Order received without Certificate of Analysis | Deny — obtain CoA |

## Data Models
- Supplier: supplier_code, supplier_type, qualification_status, quality_agreement_reference, on_approved_supplier_list
- CmoRecord: cmo_code, cmo_type, technical_agreement_reference, quality_agreement_reference, manufacturing_agreement_reference
- DemandForecast: forecast_number, method, period, forecast_horizon_months, forecasted_demand, safety_stock, sop_approved
- ImportLicense: license_number, license_type, region, product_ids, authority_reference, expiry_date
- SupplySecurityRecord: product_id, supply_status, risk_level, dual_sourced, inventory_days, contingency_plan_reference
- PurchaseOrder: po_number, order_type, supplier_id, quantity, status, coa_reference
- SupplyContract: contract_number, contract_type, version, approved, expiry_date

## Streaming Events
- supplier_qualified, supplier_suspended, supplier_audit_completed
- cmo_activated, cmo_agreement_signed
- demand_forecast_updated, sop_completed
- order_placed, order_received
- import_license_granted, import_license_expiring, import_license_expired
- supply_shortage_detected, supply_risk_escalated
- contract_approved, contract_expiring

## Edge Cases Handled
- A suspended supplier's materials already in quarantine are not automatically rejected; they require manual disposition
- Import license validity is checked per product_id and region pair, not per license number alone
- Shortage reporting to regulatory authorities is required even for products under voluntary allocation
- Dual sourcing requirement triggers when risk_level is updated to high, not only at order placement
- Contract renewal alert fires at 60 days; a second alert fires at 14 days if renewal has not been initiated

## Composability Notes
Qualifies API suppliers for `pharma_mfg` material receipt. CMO records link to `pharma_mfg` batch genealogy. Demand forecasts feed `pharma_dis` inventory planning. Import licenses gate `pharma_dis` import shipments. Supply security data informs `pharma_rec` risk management plans.
