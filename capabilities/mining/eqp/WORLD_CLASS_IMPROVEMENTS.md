# World-Class Improvements — Equipment & Plant Management (mining_eqp)

**Capability**: `mining_eqp` | **Domain**: `mining` | **Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Predictive Failure Engine via Remaining Useful Life (RUL) Estimation

Current condition monitoring detects anomalies reactively. Replace the static threshold table with an
exponentially-weighted moving average (EWMA) per sensor, per asset. Each recording updates the EWMA
and computes a Z-score drift signal. When drift exceeds 2σ for three consecutive readings, generate a
`rul_alert` with an estimated days-to-failure derived from historical failure onset patterns stored in
`_breakdown_logs`. This shifts the posture from reactive to genuinely predictive — cutting unplanned
downtime by 20–35 % in comparable fleet deployments.

---

## 2. Shift-Based Availability Accounting (shift_hours vs calendar_hours)

`equipment_availability()` uses a hard-coded `30 * 24` calendar. Production mines run 2 × 12-hour or
3 × 8-hour shift patterns. Introduce a `ShiftPattern` enum and `shift_schedule` per asset. All KPI
calculations (PA, MA, utilisation) should reference scheduled operating hours, not calendar hours.
This aligns with IOGP / VDMA availability definitions and produces reportable metrics that hold up
under external audit.

---

## 3. MTBF / MTTR Trending with Statistical Confidence Intervals

`maintenance_kpi_report()` emits point-in-time averages. Add `mtbf_trend()` and `mttr_trend()` that
compute rolling 3-month and 6-month MTBF/MTTR time series per asset class, together with 95 %
bootstrap confidence intervals. Fleet managers can then distinguish genuine improvement from random
variation — preventing premature PM interval extension that causes infant-mortality breakdowns.

---

## 4. Automated PM Escalation via Rules Engine

Planned maintenance events that pass their `due_date` without being completed currently sit silently
in `scheduled` state. Add an async `escalate_overdue_pm()` background method that: (a) identifies
overdue PMs, (b) downgrades equipment availability to `standby`, (c) creates a `HIGH` priority work
order automatically, and (d) emits a `pm_overdue_escalated` event. Integrate with the `ntfy`
capability for shift supervisor notification.

---

## 5. Digital Twin State Synchronisation

Each `EquipmentResponse` is a static snapshot. Add an `EquipmentTwinState` Pydantic model capturing
live sensor telemetry: GPS coordinates, engine RPM, payload weight, tyre pressure, coolant
temperature, and cumulative idling hours. Expose `update_twin_state()` and `get_twin_state()` methods
wired to the `mqeb` event bus. This enables real-time map visualisation and forms the data backbone
for downstream ML models.

---

## 6. Tyre Life Cycle Management Module

Tyres are the single largest consumable cost in open-pit mining (25–40 % of maintenance budget), yet
the current implementation has only a note about tyre tracking via component records. Add a dedicated
`TyreRecord` model with fields: `position_code` (e.g., `FL`, `RR`), `make`, `compound`,
`hours_at_fit`, `pressure_history`, `rim_id`, `removal_reason`. Implement `fit_tyre()`,
`remove_tyre()`, `tyre_rotation()`, and `tyre_life_report()` methods with automatic TKPH (tonne-km
per hour) calculation to predict wear-out date.

---

## 7. Ground Engaging Tools (GET) Cost-per-Tonne Tracking

Bucket teeth, adapters, and lips are high-frequency replacements on excavators and wheel loaders.
Extend `major_component_tracking()` with `get_cost_per_tonne()` that correlates GET replacement
events with the `mining_pro` production records for the same machine and period, computing an
actual cost-per-tonne moved. Surface in `equipment_analytics()` as a benchmarking KPI against
fleet average and OEM target.

---

## 8. Operator Performance Profiling

Operator ID appears in dispatch, pre-start checks, and fuel dockets but is never aggregated. Add
`operator_performance_profile()` that computes per-operator: breakdown rate attributed to operator
damage, fuel over-consumption events, pre-start check pass rate, and average dispatch-to-park
duration. Surface high-risk operators for targeted training before an incident reaches `mining_saf`.

---

## 9. Spare Parts Inventory Integration

Work orders reference `SparePart` line items with `unit_cost`, but stock levels are never validated.
Add `check_parts_availability()` that calls the `invt` inventory capability to verify each part line
against on-site bin quantity before a WO moves to `IN_PROGRESS`. If parts are on back-order, set WO
status to `AWAITING_PARTS` and emit a `purchase_order_trigger` event to the procurement system. This
eliminates technician idle time caused by parts unavailability — a top-3 MTTR driver.

---

## 10. Economic Life Optimisation via Life Cycle Cost Analysis (LCCA)

`replacement_recommendation()` uses a hard-coded 60 % repair-cost-ratio rule. Replace with a full
Net Present Value (NPV) LCCA model: discount future ownership costs at site-specific WACC, model
resale value decay as a depreciation curve fitted to historical fleet disposal data, and compare
against the NPV of owning the asset for an additional N years. The output is a ranked replacement
queue with payback period — directly usable by CFO and fleet planning.

---

## 11. Regulatory Compliance Matrix and Certificate Tracking

Mining equipment requires statutory inspections (DoM, DMR, OSHA, MSHA depending on jurisdiction),
insurance certificates, and operator fitness declarations. Add a `ComplianceCertificate` model with
`certificate_type`, `issued_at`, `expires_at`, `issuing_authority`, `document_ref`. Implement
`list_expiring_certificates()` (configurable look-ahead window), auto-block dispatch for expired
certificates, and generate a `compliance_matrix_report()` showing the full certificate status grid
across the fleet. Prevents costly regulatory shutdowns.

---

## 12. Fuel Anti-Fraud Detection

Fuel theft and docket inflation are endemic in remote mining operations. Enhance `record_fuel_docket()`
with multi-signal fraud detection: (a) cross-reference quantity_litres against tank capacity for the
equipment class, (b) check fuelling interval plausibility against engine hours delta, (c) flag
duplicate docket numbers within rolling 7-day window, (d) compare cost_per_litre against the
site-configured fuel price with configurable tolerance. Generate `fuel_fraud_alert` events routed to
the security and finance capabilities.

---

## 13. Integration with Mine Planning Schedule (Rostering)

Equipment dispatch currently operates in isolation. Add `sync_dispatch_schedule()` that consumes a
`ShiftRoster` from the `schd` capability, pre-assigns equipment to activities (blast loading,
hauling, stripping), and produces a `DispatchPlan` object. Pre-shift inspections and operator
assignments are auto-populated for the coming shift, reducing the 30-minute manual dispatch board
preparation to zero.

---

## 14. Event-Sourced Audit Trail with Replay

All state mutations (dispatch, fault creation, WO completion) occur directly on in-memory dicts with
no event history. Introduce an `EqpEvent` base model and an append-only `_event_log` deque. Every
mutation emits an event. Add `replay_from_events()` to reconstruct state at any point in time.
This enables: full audit trail for regulatory inspections, time-travel debugging of fleet state, and
eventual migration to an event-sourced PostgreSQL backend without a Big Bang rewrite.

---

## 15. Multi-Site / Hub-and-Spoke Fleet Sharing

Assets are tenant-scoped but single-site. Large mining groups move equipment between pits and
processing plants. Add a `TransferOrder` model with `source_site`, `destination_site`, `handover_at`,
`transport_contractor`, and `condition_survey_id`. Implement `initiate_transfer()`,
`confirm_receipt()`, and `list_inter_site_transfers()`. During transfer, equipment lifecycle_status
moves to `IN_TRANSIT`; it is blocked from dispatch and maintenance scheduling on both sites until
receipt is confirmed. Supports group-level fleet optimisation: right asset, right place, right time.
