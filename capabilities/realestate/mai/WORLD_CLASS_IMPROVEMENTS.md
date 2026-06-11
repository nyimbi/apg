# Facilities Maintenance (realestate_mai) — World-Class Improvements

**Capability**: `realestate_mai` | **Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Predictive Maintenance via Condition Scoring

Current PPM logic is purely calendar-driven. Integrate an asset condition score (0–100) updated after each inspection and work order completion. When score drops below a configurable threshold, automatically escalate PPM frequency and pre-raise a predictive work order. Eliminates surprise failures on aging assets, especially HVAC and lifts.

**Methods**: `update_asset_condition_score()`, `get_assets_below_condition_threshold()`

---

## 2. Real-Time SLA Countdown with Warning Tiers

SLA deadlines are computed at creation but never re-evaluated until a breach occurs. Add a tiered warning system: 75%, 90%, and 100% of SLA elapsed each emit distinct events. Contractors receive automated notifications at each tier, enabling proactive escalation before breach rather than reactive escalation after.

**Methods**: `evaluate_sla_warning_tier()`, `get_work_orders_near_sla_breach()`

---

## 3. Contractor Performance Scorecards

The `MaintenanceContractorResponse` has `first_time_fix_rate` and `average_response_hours` but they are never populated. Implement rolling 30/90/365-day computation from completed work orders. Surface contractor league tables to inform assignment decisions and contract renewal. Block assignment of contractors whose 30-day first-time fix rate falls below configurable floor.

**Methods**: `compute_contractor_scorecard()`, `get_contractor_league_table()`

---

## 4. Budget Forecasting and Variance Tracking

Maintenance budgets are currently invisible to the service. Add a `MaintenanceBudget` model keyed by `(tenant_id, property_id, financial_year)`. Track committed cost (assigned WOs), actual spend (completed WOs), and remaining budget. Trigger approval-gate workflow when a single WO would exceed remaining budget.

**Methods**: `set_maintenance_budget()`, `get_budget_vs_actual()`, `check_budget_headroom()`

---

## 5. Mobile-Optimised Work Order Check-In / Check-Out

Field technicians need a lightweight protocol: check-in (GPS-stamped, starts actual timer), upload photos, check-out (ends timer, triggers next status). Model `WorkOrderCheckin` and `WorkOrderCheckout` with geolocation evidence. Elapsed time becomes the authoritative `actual_duration` for SLA measurement and invoicing reconciliation.

**Methods**: `checkin_work_order()`, `checkout_work_order()`

---

## 6. Warranty Claims Management

Assets carry `warranty_expiry` but there is no workflow for warranty claims. Add `WarrantyClaim` model linking an asset defect to the OEM/supplier, claim reference, submission date, and resolution. When a corrective work order is raised on an in-warranty asset, auto-prompt a claim. Reduces unnecessary contractor spend on manufacturer-liable faults.

**Methods**: `raise_warranty_claim()`, `resolve_warranty_claim()`, `list_open_warranty_claims()`

---

## 7. Spare Parts and Materials Inventory

Work orders frequently stall on `pending_parts` with no visibility into parts availability. Add a lightweight parts catalogue (`SparePart`) and `PartReservation` linked to work orders. Stock levels, re-order points, and lead times per part per property. Eliminates manual stocktaking spreadsheets and reduces WO cycle time.

**Methods**: `reserve_parts_for_work_order()`, `consume_parts_on_completion()`, `get_low_stock_parts()`

---

## 8. Statutory Compliance Certificate Register

Statutory inspections (gas safety, electrical, fire) produce certificates with expiry dates. Currently findings are free-form JSON. Model `StatutoryComplianceCertificate` with certificate type, issuing authority, issue date, expiry, and certificate reference. Auto-raise future inspection WOs 60 days before expiry. Enables one-click compliance status per property for lettings due-diligence.

**Methods**: `register_compliance_certificate()`, `get_expiring_certificates()`, `get_property_compliance_status()`

---

## 9. Multi-Site Portfolio Benchmarking

`cost_per_sqm()` works per-property but there is no cross-portfolio view. Add `benchmark_portfolio()` that ranks all properties by cost/sqm, ppm completion rate, open defect density, and SLA breach rate. Output percentile positions per metric. Facilities directors can identify outlier properties requiring capital intervention.

**Methods**: `benchmark_portfolio()`, `get_portfolio_maintenance_heatmap()`

---

## 10. Escalation Workflow Engine

P1 work orders require immediate assignment but there is no automated escalation chain when assignment does not happen within the SLA response window. Implement an `EscalationPolicy` model (levels, delay minutes per level, notified roles) and a `process_escalations()` method that advances unresolved WOs through levels on each scheduler tick. Reduces P1 mean-time-to-respond (MTTR) without manual management intervention.

**Methods**: `create_escalation_policy()`, `process_escalations()`, `get_escalation_history()`

---

## 11. Digital Twin Asset Import (BIM/IFC Integration)

Asset registration is manual. Support bulk import from IFC/COBie spreadsheet exports, mapping BIM object GUIDs to `asset_ref`. On re-import, detect new, modified, and decommissioned assets and reconcile. Removes the data-entry bottleneck that keeps asset registers stale in the first year after handover.

**Methods**: `import_assets_from_cobie()`, `reconcile_asset_import()`

---

## 12. Reactive Maintenance Pattern Detection

Repeated corrective work orders on the same asset within a rolling window signal an underlying fault. Detect when `>= N` corrective WOs are raised on one asset within `D` days and auto-raise a defect of severity `major` linking all contributing WOs. Surfaces hidden asset degradation before it escalates to P1.

**Methods**: `detect_reactive_patterns()`, `get_repeat_failure_assets()`

---

## 13. Tenant/Occupier-Reported Issue Portal

Occupiers need a frictionless channel to report issues without raising a full work order. Add `OccupierReport` (description, location, photo_ids, contact_email) that feeds a triage queue. A facilities manager reviews and either dismisses, raises a defect, or promotes to a work order. Captures demand not currently visible to CAFM.

**Methods**: `submit_occupier_report()`, `triage_occupier_report()`, `list_pending_occupier_reports()`

---

## 14. Net Zero Carbon Pathway Tracking

`sustainability_tracking()` records energy and carbon per period but there is no trajectory model. Add a `CarbonTarget` model (`baseline_year`, `target_year`, `target_reduction_pct`) and `compute_carbon_trajectory()` that projects required annual reductions and compares actuals to the glide path. Supports ESG reporting and green lease obligations.

**Methods**: `set_carbon_target()`, `compute_carbon_trajectory()`, `get_carbon_performance_vs_target()`

---

## 15. Invoice Reconciliation and Payment Authorisation

Work orders track `agreed_cost` and `actual_cost` but there is no invoice model. Add `ContractorInvoice` linked to one or more completed work orders, with line items, submitted amount, approved amount, and payment status. Implement three-way matching (PO ref, completion sign-off, invoice amount within tolerance). Auto-approve within tolerance; route over-run invoices through configurable approval workflow.

**Methods**: `submit_contractor_invoice()`, `approve_invoice()`, `get_outstanding_invoices()`
