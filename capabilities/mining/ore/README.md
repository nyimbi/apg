# Ore Processing & Metallurgy

## Overview
Manages ore processing plant operations including plant feed tracking, process circuit status monitoring, reagent inventory management, metallurgical mass balance preparation and approval, product quality assurance, ore reconciliation, and process deviation alert management. Enforces metallurgical integrity constraints including recovery bounds [0, 100%], cyanide code compliance, approval gating before balance publication, and off-specification product dispatch controls.

## Capability ID
`mining_ore`

## Provides
| Service | Description |
|---|---|
| plant_feed_tracking | Per-period dry tonnes, grade, moisture, and feed source recording |
| metallurgical_balance_workflow | Mass balance submission, approval, and publication with recovery validation |
| reagent_management | Reagent usage recording, inventory deduction, and low-stock alerting |
| recovery_optimisation_tracking | Historical recovery trend data and deviation alert management |
| product_quality_management | Lot-level quality records with specification check and dispatch approval |
| process_circuit_monitoring | Real-time circuit status snapshots (throughput, power, downtime reason) |
| ore_reconciliation_workflow | Reconciliation of feed, concentrate, and tailings streams |
| deviation_alert_management | Raise, acknowledge, and resolve process deviation alerts |
| assay_database_management | Sample point assay records linked to circuit streams |
| process_kpi_reporting | Average recovery, throughput, open deviations |

## Requires
| Capability | Reason |
|---|---|
| auth | User authentication |
| audl | Audit trail for balance approvals and product dispatch |
| mten | Multi-tenancy isolation |
| conf | Runtime configuration |
| ntfy | Low reagent stock alerts and off-spec product notifications |
| wflo | Met balance approval and reconciliation workflows |
| moni | Real-time process variable monitoring |
| mqeb | Event streaming for process dashboards |

## Configuration
| Key | Default | Description |
|---|---|---|
| plant_feed.feed_grade_required | true | Grade value mandatory for all feed records |
| reagents.cyanide_code_compliance_required | true | ICMC cyanide code compliance check on cyanide usage |
| reagents.inventory_tracking_required | true | Inventory balance maintained per reagent type |
| metallurgical_balance.approval_required | true | Approval required before publication |
| product_quality.specification_check_required | true | Spec check result mandatory on product records |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /api/mining-ore/plant-feed | GET/POST | List/record plant feed | mining_ore:view/write |
| /api/mining-ore/plant-feed/:id | GET | Get feed record | mining_ore:view |
| /api/mining-ore/circuits/status | POST | Update circuit status | mining_ore:write |
| /api/mining-ore/circuits/current | GET | Current circuit statuses | mining_ore:view |
| /api/mining-ore/reagents/inventory | GET | Reagent inventory levels | mining_ore:view |
| /api/mining-ore/reagents/stock | POST | Add reagent stock | mining_ore:write |
| /api/mining-ore/reagents/usage | GET/POST | List/record usage | mining_ore:view/write |
| /api/mining-ore/met-balance | GET/POST | List/submit balances | mining_ore:met_balance |
| /api/mining-ore/met-balance/:id | GET | Get balance | mining_ore:met_balance |
| /api/mining-ore/met-balance/:id/approve | POST | Approve balance | mining_ore:met_balance |
| /api/mining-ore/met-balance/:id/publish | POST | Publish balance | mining_ore:met_balance |
| /api/mining-ore/product-quality | GET/POST | List/record quality | mining_ore:view/write |
| /api/mining-ore/product-quality/:id/approve-dispatch | POST | Approve dispatch | mining_ore:write |
| /api/mining-ore/deviations | GET/POST | List/raise alerts | mining_ore:view/write |
| /api/mining-ore/deviations/:id/acknowledge | POST | Acknowledge alert | mining_ore:write |
| /api/mining-ore/deviations/:id/resolve | POST | Resolve alert | mining_ore:write |
| /api/mining-ore/kpis | GET | Process KPI summary | mining_ore:view |
| /api/mining-ore/grind/optimise | POST | Grind P80 setpoint advisory | mining_ore:write |
| /api/mining-ore/met-balance/:id/close | POST | Two-product mass balance closure | mining_ore:met_balance |
| /api/mining-ore/cil-loading | GET/POST | CIL carbon loading profiles | mining_ore:view/write |
| /api/mining-ore/ore-type | POST | Ore type classification (XRF) | mining_ore:write |
| /api/mining-ore/ore-hardness | GET/POST | Bond Work Index records | mining_ore:view/write |
| /api/mining-ore/water-balance | GET/POST | Site water balance | mining_ore:view/write |
| /api/mining-ore/reagents/spc | POST | SPC control chart for reagent dosage | mining_ore:view |
| /api/mining-ore/thickener | GET/POST | Tailings thickener performance | mining_ore:view/write |
| /api/mining-ore/nsr | POST | Net Smelter Return calculation | mining_ore:view |
| /api/mining-ore/reports/shift | POST | Shift metallurgical report | mining_ore:view |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| negative_recovery_denied | Recovery < 0% | DENY |
| recovery_over_100_denied | Recovery > 100% | DENY |
| cyanide_code_compliance | Non-compliant cyanide use | DENY |
| met_balance_approval_required | Publish without approval | DENY |
| off_spec_dispatch_denied | Dispatch off-spec product without approval | DENY |
| reconciliation_approval_required | Finalise without approval | DENY |
| delete_approved_balance_denied | Delete approved balance | DENY — supersede instead |
| feed_grade_required | Missing feed grade | DENY |
| reagent_dosage_required | Missing dosage rate | DENY |
| cross_tenant_read_denied | Cross-tenant access | DENY |
| mass_balance_closure_threshold | Closure error > tolerance_pct (default 3%) | WARN — block publication if configured |
| cil_overload_threshold | Loaded carbon > 8000 g/t in any tank | WARN |
| water_permit_exceedance | Recycled water parameter > permit limit | WARN + compliance flag |
| nsr_negative | NSR < 0 USD/t at current spot prices | WARN |
| grind_deviation_critical | P80 deviation > 10% of target | WARN |

## Data Models
| Model | Key Fields |
|---|---|
| PlantFeedCreate/Response | feed_source, dry_tonnes, feed_grade, moisture_pct, period_start/end |
| MetallurgicalBalanceCreate/Response | balance_type, feed/concentrate/tailings_stream, calculated_recovery_pct, status, published |
| ReagentUsageCreate/Response | reagent_type, quantity_kg, dosage_rate_g_t, circuit_id, total_cost |
| ProductQualityCreate/Response | product_type, lot_number, commodity_grade, meets_specification, dispatched |
| DeviationAlertCreate/Response | deviation_type, alert_level, actual_value, target_value, variance_pct, resolved |
| GrindOptimisationResult | current_p80_um, target_p80_um, deviation_pct, recommended_mill_speed_pct, estimated_specific_energy_kwh_t |
| MassBalanceClosureResult | mass_closure_error_pct, metal_closure_error_pct, recovery_assay_pct, closure_ok |
| CILLoadingRecord | tank_profiles, average_loading_g_t, overloaded_tanks, loading_gradient_ok, total_gold_locked_kg |
| OreTypeClassification | ore_domain, expected_recovery_min/max_pct, recommended_processing_route, recommended_reagent_suite |
| OreHardnessRecord | bwi_kwh_t, abrasion_index, hardness_class, relative_throughput_factor |
| WaterBalanceRecord | recycle_rate_pct, water_intensity_m3_t, permit_compliant, compliance_exceedances |
| SPCReagentChart | mean/ucl/lcl dosage, western_electric_violations, dosage_recovery_correlation, recommendation |
| ThickenerPerformanceRecord | underflow_solids_pct, overflow_turbidity_ntu, unit_area_loading_t_m2_d |
| NSRCalculation | gross_value_usd_t, total_tc_rc_usd_t, nsr_usd_per_t_concentrate, nsr_total_usd |
| ShiftMetReport | total_feed_tonnes, open_deviations, critical_deviations, recovery_alert_threshold_pct |

## Streaming Events
- `plant_feed_recorded`
- `circuit_status_changed`
- `grind_p80_deviation_detected` / `grind_setpoint_adjusted`
- `reagent_usage_recorded` / `reagent_reorder_triggered` / `reagent_spc_violation`
- `metallurgical_balance_submitted` / `metallurgical_balance_approved` / `mass_balance_closure_failed`
- `product_quality_recorded` / `off_spec_product_flagged`
- `recovery_deviation_detected`
- `reconciliation_finalised`
- `cil_carbon_overloaded` / `cil_gradient_inverted`
- `ore_type_classified`
- `water_permit_exceedance`
- `nsr_negative_alert`
- `shift_report_generated`

## Edge Cases Handled
- Recovery validated at both service layer (ValueError) and rule engine; 0% and 100% are valid bounds
- Cyanide usage triggers a dedicated audit log entry regardless of quantity
- Reagent inventory cannot go below zero; deduction is clamped to zero
- Low-stock threshold (500 kg) triggers a warning log but does not block usage
- Off-spec product dispatch requires explicit approval; `meets_specification=False` alone blocks dispatch
- Met balance deletion blocked if status is APPROVED; supersede workflow required
- `close_metallurgical_balance()` rejects balances with mass or metal closure error > `tolerance_pct`; safe default is 3%
- `grind_optimisation_cycle()` clamps recommended mill speed to [60%, 85%] of critical speed regardless of control output
- `spc_reagent_control()` requires minimum 5 observations; raises `AssertionError` on fewer
- `record_cil_loading()` verifies loading gradient is monotonically decreasing (tank 1 highest); inverted gradient triggers warning
- `compute_nsr()` emits a warning when NSR is negative — does not block recording
- `classify_ore_type()` defaults to `refractory` (most conservative route) when arsenic > 800 ppm or sulphur > 2%
- `record_water_balance()` performs parameter-by-parameter permit compliance; each exceedance is individually logged

## Composability Notes
- Plant feed records derived from `mining_pro` stockpile movements and ore tracking
- Feed grade characterisation informed by `mining_exp` geological and assay data
- Ore type classification links to `mining_exp` block model ore domain attributes
- Bond Work Index records feed `mining_pro` throughput scheduling and liner replacement planning
- Off-spec product events feed `mining_saf` non-conformance tracking
- Reagent inventory feeds procurement via `scm` integration
- Water balance records feed `mining_env` environmental permit compliance reporting
- NSR calculations integrate with `fin` (finance) revenue forecasting and hedging modules
- Shift met reports are distributed via `ntfy` to shift supervisors and metallurgists
- Reconciliation data feeds ESG Scope 1/2 emissions calculations in `mining_env`
- Reconciliation data feeds ESG Scope 1/2 emissions calculations in `mining_env`

---

## World-Class Enhancements (v2.0)

- **I1.** Ore Processing (mining_ore) — World-Class Improvement Catalogue
- **I2.** Real-time Grind Circuit Optimisation (SAG/Ball Mill)
- **I3.** Automated Metallurgical Balance Closure Verification
- **I4.** Online Assay Integration via OPC-UA / MQTT
- **I5.** Multi-Element Grade Control Blending Optimiser
- **I6.** Predictive Reagent Dosage via Statistical Process Control
- **I7.** Concentrate Filter Cake Quality Tracking
- **I8.** Tailings Thickener Performance Management
- **I9.** Carbon-in-Leach (CIL) Loading Profile Management
- **I10.** Elution and Electrowinning Efficiency Tracking
- **I11.** Ore Hardness and Bond Work Index (BWI) Tracking
- **I12.** Water Balance and Recycled Water Quality Tracking
- **I13.** Automated Shift Metallurgical Report Generation
- **I14.** Ore Type Classification and Geometallurgical Mapping
- **I15.** Locked Cycle Flotation Test Results Repository

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
