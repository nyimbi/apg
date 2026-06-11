# Pharmaceutical Manufacturing — World-Class Improvements

**Capability**: `pharma_mfg` | **Path**: `capabilities/pharma/mfg`

---

## 1. Async-First Service Layer

All public methods are currently synchronous. Blocking I/O in a FastAPI or Starlette host starves the event loop. Every method that touches the store, emits audit events, or calls external adapters should be `async def`. The existing `export_records`, `health_check`, and friends are already async but the core batch/deviation/equipment methods are not — this inconsistency is a maintenance hazard.

**Action**: Convert all public methods to `async def`. The in-memory stores are thread-safe dictionaries; wrapping them requires no `asyncio.Lock` unless concurrent writes become a concern.

---

## 2. Process Order / Work-In-Progress (WIP) Scheduling

There is no concept of scheduling batch manufacturing orders against constrained production lines and equipment calendars. A scheduler would let planners see line utilisation, detect conflicts, and generate Gantt-style capacity views — standard in any MES (Manufacturing Execution System).

**Action**: Add `async schedule_batch_production(batch_id, line_id, start_dt, duration_hours, tenant_id)` and a `get_production_schedule(tenant_id, date_range)` method that returns ordered work items with conflict detection.

---

## 3. Electronic Batch Record (EBR) Step Execution Engine

The current EBR is a status flag on a `BatchRecord`. A real EBR drives operators through numbered steps (weighing, granulation, compression, coating, packaging), each with its own entry/verification/review gate and e-signature. Without step-level granularity, the EBR does not satisfy 21 CFR Part 11 or EU GMP Annex 11.

**Action**: Add `async execute_ebr_step(batch_id, step_number, operator_id, data, tenant_id)` and `async verify_ebr_step(batch_id, step_number, reviewer_id, tenant_id)` with a dual-entry verification model.

---

## 4. Cleaning Validation with Residue Limit Calculation

Line clearance exists but cleaning *validation* — calculating Maximum Allowable Carryover (MACO), visual inspection limits, and analytical method verification — is absent. Sharing equipment between products requires documented cleaning validation per EU GMP Annex 15.

**Action**: Add `async calculate_maco(donor_product_id, recipient_product_id, equipment_surface_area, tenant_id)` using toxicological threshold data and `async record_cleaning_validation_run(protocol_id, run_number, results, tenant_id)`.

---

## 5. Environmental Monitoring Integration

Pharmaceutical cleanrooms require continuous temperature, humidity, and particulate monitoring. The `ProductionLine` model has `environmental_monitoring_active: bool` but no data ingestion, limit checking, or out-of-limit (OOL) alerting. A contamination event that bypasses EBR review is a critical GMP failure.

**Action**: Add `async record_environmental_sample(line_id, sample_point, parameter, value, unit, limit_low, limit_high, sampled_by, tenant_id)` and link OOL results to automatic deviation raising.

---

## 6. CAPA (Corrective and Preventive Action) Lifecycle

Deviations can carry a `capa_reference` string but there is no CAPA record, effectiveness check, or recurrence tracking. Regulators (FDA, EMA) expect a closed-loop CAPA system with root cause categorisation, action assignments, due dates, and effectiveness reviews.

**Action**: Add `async open_capa(deviation_id, root_cause_category, actions, due_date, owner_id, tenant_id)` and `async close_capa(capa_id, effectiveness_evidence, tenant_id)` with overdue CAPA detection in `gmp_compliance_check`.

---

## 7. Material Genealogy / Batch Traceability Graph

There is no link from `RawMaterial` lots used in a batch back to finished product lots. Recall management requires a complete forward/backward trace: raw lot → batch → sub-batches → finished pack → dispatch. Without it, a material-driven recall means recalling every batch that *could* have used that lot.

**Action**: Add `async link_material_to_batch(material_id, batch_id, quantity_dispensed, dispense_reference, tenant_id)` and `async trace_batch_genealogy(batch_id, tenant_id)` returning a full DAG of inputs and outputs.

---

## 8. Statistical Process Control (SPC) Charts

Yield variance is checked per batch but there is no trend detection across batches. SPC — Shewhart control charts (X-bar, R, CUSUM) — detects process drift before it causes an out-of-spec event. ICH Q10 expects continuous process verification (CPV) for established products.

**Action**: Add `async get_spc_data(product_id, parameter, tenant_id, n_batches=30)` returning control limits, individual values, and Western Electric rule violations. Feed this into `batch_analytics`.

---

## 9. Serialisation / Track-and-Trace Support

Serialisation of saleable units (2D DataMatrix on cartons and pallets) is mandated by the EU Falsified Medicines Directive and the US DSCSA. Finished goods must be assigned unique serial numbers, aggregated into hierarchical packs, and reported to national verification systems.

**Action**: Add `async assign_serial_numbers(batch_id, quantity, pack_level, tenant_id)` and `async commission_pack(serial_number, batch_id, expiry_date, tenant_id)` generating GS1-compliant identifiers.

---

## 10. Change Control Management

Process changes (formulation, equipment, site, supplier) must pass through a documented change control process before implementation. Currently there is no change request, impact assessment, or implementation sign-off record, leaving validation lifecycle gaps.

**Action**: Add `async raise_change_request(description, change_type, affected_products, impacted_systems, raised_by, tenant_id)` and `async approve_change_request(change_id, approver_id, conditions, tenant_id)`.

---

## 11. Out-of-Specification (OOS) / Out-of-Trend (OOT) Investigation Workflow

In-process checks can be flagged `out_of_spec` but there is no structured OOS investigation compliant with FDA 2006 OOS guidance: Phase I (laboratory investigation) → Phase II (manufacturing investigation) → disposition decision. OOT (trending toward spec limits) is not handled at all.

**Action**: Add `async open_oos_investigation(check_id, phase, assignee_id, tenant_id)` progressing through phases with data entry at each stage, and `async detect_oot(batch_id, parameter, historical_window, tenant_id)` using rolling statistics.

---

## 12. Calibration Management

Equipment has `next_calibration_due` but no calibration *record*, no calibration standard reference, and no blocked-use enforcement when the equipment is overdue. Overdue calibration is an immediate GMP finding.

**Action**: Add `async record_calibration(equipment_id, standard_reference, result, calibrated_by, next_due, tenant_id)` and integrate calibration status into `use_equipment` to block usage when expired, with audit trail.

---

## 13. Batch Disposition Workflow (Quarantine → Released / Rejected)

Batch status transitions happen directly via `release_batch` / `reject_batch` with no intermediate disposition review stage. A real QP review involves a physical document pack check, open deviation review, and formal disposition decision before release or rejection.

**Action**: Add `async submit_for_disposition(batch_id, qp_id, document_pack_reference, tenant_id)` creating a disposition record, then `async complete_disposition(disposition_id, decision, conditions, tenant_id)` with document reference linking.

---

## 14. Multi-Site / Contract Manufacturing Organisation (CMO) Support

The service is keyed on `tenant_id` but has no concept of sites, buildings, or rooms within a tenant. Multi-site manufacturers and CMOs need to partition data by site while sharing vendor and product master data at the tenant level.

**Action**: Add `site_id` as a first-class field on `BatchRecord`, `Equipment`, and `ProductionLine`. Add `async register_site(site_code, name, gmp_license_number, tenant_id)` and site-scoped list/filter parameters on all list methods.

---

## 15. Predictive Maintenance Scheduling

Equipment maintenance is tracked via `next_maintenance_due` date but this is purely calendar-based. Integrating usage counters (cycle count, operating hours) and connecting to the AI deviation detection backbone enables predictive maintenance — scheduling intervention before failure, not after.

**Action**: Add `async record_equipment_usage(equipment_id, usage_hours, cycle_count, tenant_id)` accumulating utilisation metrics, and `async predict_maintenance_window(equipment_id, tenant_id)` calling the OLLAMA ML adapter to predict remaining useful life and flag maintenance urgency.
