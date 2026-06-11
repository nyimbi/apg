# Vehicle Maintenance — World-Class Improvement Plan

**Capability**: `transport_mai` | **Version target**: 2.0.0  
**Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Technician Skill Matching & Workload Balancing

**Problem**: Jobs are assigned to a single static `technician_id` with no validation that the technician's certified skills match the maintenance type (e.g., assigning an electricals specialist to a gearbox rebuild).

**Improvement**: Add a `TechnicianProfile` model carrying skill certifications and current workload. Extend `create_job` to query available technicians, score them by skill match and current open-job count, and auto-assign the best candidate. Surface a `get_technician_workload()` method for dispatch planning.

**Impact**: Reduces rework rates, improves first-time-fix rates, enables predictive capacity planning.

---

## 2. Real Odometer-Linked Service Due Dates

**Problem**: `predictive_maintenance_alert` uses a hardcoded `last_km = 0.0` stub, making interval calculations meaningless for production use.

**Improvement**: Introduce `record_odometer_reading(vehicle_id, km, recorded_at)` that stores time-stamped readings. `predictive_maintenance_alert` uses the most recent reading to compute km-since-last-service accurately. Derive projected service due date by extrapolating average daily km from reading history.

**Impact**: Eliminates the biggest correctness hole in predictive maintenance; enables accurate calendar scheduling.

---

## 3. Breakdown Event Pipeline with SLA Tracking

**Problem**: Breakdowns are treated the same as routine defects. There is no concept of response SLA (e.g., roadside attendance within 2 hours), no escalation timer, and no post-incident root-cause capture.

**Improvement**: Add `log_breakdown_event(vehicle_id, location, breakdown_type, sla_minutes)` that creates a high-priority job, starts an SLA clock, and attaches a `BreakdownSLA` record. A companion `check_sla_breaches()` method scans for jobs that exceeded their SLA target and returns breach metadata.

**Impact**: Reduces mean-time-to-repair, enables SLA-based vendor contract enforcement.

---

## 4. Parts Reorder Automation with Minimum Stock Rules

**Problem**: `parts_inventory_check` only inspects existing orders; there is no minimum-stock threshold or auto-reorder trigger.

**Improvement**: Introduce `set_parts_reorder_threshold(part_number, min_qty, reorder_qty, supplier_id)` and `trigger_reorder_if_low()` which scans thresholds, computes net available stock, and issues reorder `PartsOrder` records for any part falling below its minimum. Emit a streaming event per reorder.

**Impact**: Eliminates stockout-driven downtime; closes the inventory management loop.

---

## 5. Digital Twin Vehicle State Machine

**Problem**: Vehicle maintenance state (in_service, in_workshop, awaiting_parts, grounded, decommissioned) is implicit and scattered across job and defect records.

**Improvement**: Add a `VehicleState` enum and `update_vehicle_state(vehicle_id, new_state, reason)` that transitions state with guard conditions (e.g., cannot move from `grounded` to `in_service` without a passed roadworthiness check). Store state history for audit.

**Impact**: Single source of truth for vehicle availability; prevents dispatch of grounded vehicles.

---

## 6. Warranty Claim Auto-Filing Workflow

**Problem**: `record_warranty` is purely a data-entry call with no tie-in to repair jobs or claim submission.

**Improvement**: Add `file_warranty_claim(warranty_id, job_id, defect_description, evidence_refs)` that validates the warranty is active, attaches it to the job, constructs a structured claim payload, and routes it through the `wflo` workflow adapter. Track claim status (submitted, acknowledged, approved, rejected, paid).

**Impact**: Reduces manual claim administration overhead; enables recovery of warranty costs.

---

## 7. Maintenance Cost Ledger with Actual Parts Pricing

**Problem**: `cost_per_km` uses a stub unit cost of $15 for all parts, rendering cost analytics unreliable.

**Improvement**: Add `record_parts_unit_cost(part_number, unit_cost_usd, effective_date, supplier_id)` that builds a price catalogue. `complete_work_order` and `cost_per_km` resolve actual costs from the catalogue with effective-date lookups. Include a `cost_variance_report(vehicle_id, period)` comparing estimated vs actual cost.

**Impact**: Accurate P&L attribution per vehicle; enables budget forecasting and supplier negotiation.

---

## 8. Multi-Vehicle Bulk Inspection Campaigns

**Problem**: Inspections are created one at a time; there is no concept of a fleet-wide inspection campaign (e.g., annual MOT batch) with progress tracking.

**Improvement**: Add `create_inspection_campaign(vehicle_ids, inspection_type, due_by, inspector_id)` returning a campaign record with per-vehicle status. `get_campaign_progress(campaign_id)` returns completion %, outstanding vehicles, and pass/fail breakdown.

**Impact**: Reduces compliance risk from missed periodic inspections; enables fleet-wide compliance dashboards.

---

## 9. Predictive Failure Model Integration

**Problem**: `predictive_maintenance` uses a trivial `jobs_done * 0.03` proxy for fault probability — no telemetry, no component-level reasoning.

**Improvement**: Expose `ingest_telematics_event(vehicle_id, sensor_id, value, unit, recorded_at)` to accumulate sensor data (engine temperature, oil pressure, brake pad wear sensors). `predictive_maintenance` aggregates sensor anomalies using configurable thresholds to produce per-component risk scores. Emit a streaming alert when risk exceeds a threshold.

**Impact**: Moves from reactive to condition-based maintenance; reduces unplanned breakdowns.

---

## 10. Labour Time Tracking with Technician Clock-In/Clock-Out

**Problem**: Labour hours are entered as a single number on job close. There is no way to track multiple technicians working on the same job or verify actual vs booked time.

**Improvement**: Add `clock_in(job_id, technician_id)` / `clock_out(job_id, technician_id)` that create `LabourEntry` records. `close_job` aggregates all entries to compute actual hours. `labour_utilisation_report(technician_id, period)` computes billable vs total hours per technician.

**Impact**: Accurate labour cost capture; enables technician productivity and overage analysis.

---

## 11. Supplier Performance Scorecard

**Problem**: Parts orders are placed against `supplier_id` with no tracking of delivery lead time, fill rate, or defect rate.

**Improvement**: Add `record_parts_receipt(order_id, received_qty, received_at, quality_ok)` which updates `PartsOrder.received_at` and records quality outcome. `supplier_scorecard(supplier_id, period)` computes on-time delivery %, fill rate %, and defect rate % from receipts.

**Impact**: Data-driven supplier selection; supports contract renewal and tender processes.

---

## 12. Compliance Calendar with Automated Reminders

**Problem**: Roadworthiness and inspection due dates are stored but no system proactively surfaces upcoming deadlines or pushes reminders.

**Improvement**: Add `get_compliance_calendar(tenant_id, days_ahead)` that aggregates all due dates (MOT renewals, periodic inspections, scheduled services) into a unified timeline sorted by urgency. Integrate with the `ntfy` adapter to send reminders at configurable lead times (7, 14, 30 days).

**Impact**: Reduces compliance failures from missed renewals; enables proactive fleet management.

---

## 13. Job Dependency Graph for Complex Repairs

**Problem**: Related jobs (e.g., inspect brakes before ordering pads before fitting pads) have no dependency model. Jobs can be started or closed in any order.

**Improvement**: Add `link_jobs(parent_job_id, child_job_id, dependency_type)` (types: blocks, requires_parts_from, preceded_by). `get_job_dependency_graph(job_id)` returns the DAG. Status transitions on a child job validate that all blocking parents are completed.

**Impact**: Prevents technicians starting work before pre-requisites are met; models real workshop workflow.

---

## 14. Fleet-Wide TCO (Total Cost of Ownership) Report

**Problem**: Cost analytics are per-vehicle only. There is no fleet-level aggregation of maintenance TCO that can be used for vehicle lifecycle decisions (keep vs replace).

**Improvement**: Add `fleet_tco_report(tenant_id, period)` that aggregates labour cost, parts cost, downtime hours, and roadworthiness renewal fees per vehicle and fleet-total. Include a `replacement_candidates(threshold_cost_usd)` helper that flags vehicles whose maintenance cost exceeds the threshold.

**Impact**: Supports data-driven fleet replacement decisions; reduces total fleet operating cost.

---

## 15. Defect Resolution Workflow with Root Cause Classification

**Problem**: `log_defect` sets `resolved=False` with no structured resolution path. Defects are never marked resolved in the current code, so `roadworthiness_check` will always find blocking issues on vehicles with historical defects.

**Improvement**: Add `resolve_defect(defect_id, resolution_notes, root_cause_category, resolved_by)` that sets `resolved=True`, records root cause (driver_abuse, wear_and_tear, manufacturing_defect, accident_damage, environmental), and links to the closing job. `defect_recurrence_report(vehicle_id)` detects repeat root causes suggesting systematic issues.

**Impact**: Closes the defect-to-resolution loop; enables root cause analysis and recurring-fault detection.
