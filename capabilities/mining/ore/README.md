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

## Data Models
| Model | Key Fields |
|---|---|
| PlantFeedCreate/Response | feed_source, dry_tonnes, feed_grade, moisture_pct, period_start/end |
| MetallurgicalBalanceCreate/Response | balance_type, feed/concentrate/tailings_stream, calculated_recovery_pct, status, published |
| ReagentUsageCreate/Response | reagent_type, quantity_kg, dosage_rate_g_t, circuit_id, total_cost |
| ProductQualityCreate/Response | product_type, lot_number, commodity_grade, meets_specification, dispatched |
| DeviationAlertCreate/Response | deviation_type, alert_level, actual_value, target_value, variance_pct, resolved |

## Streaming Events
- `plant_feed_recorded`
- `circuit_status_changed`
- `reagent_usage_recorded` / `reagent_reorder_triggered`
- `metallurgical_balance_submitted` / `metallurgical_balance_approved`
- `product_quality_recorded` / `off_spec_product_flagged`
- `recovery_deviation_detected`
- `reconciliation_finalised`

## Edge Cases Handled
- Recovery validated at both service layer (ValueError) and rule engine; 0% and 100% are valid bounds
- Cyanide usage triggers a dedicated audit log entry regardless of quantity
- Reagent inventory cannot go below zero; deduction is clamped to zero
- Low-stock threshold (500 kg) triggers a warning log but does not block usage
- Off-spec product dispatch requires explicit approval; `meets_specification=False` alone blocks dispatch
- Met balance deletion blocked if status is APPROVED; supersede workflow required

## Composability Notes
- Plant feed records derived from `mining_pro` stockpile movements and ore tracking
- Feed grade characterisation informed by `mining_exp` geological and assay data
- Off-spec product events feed `mining_saf` non-conformance tracking
- Reagent inventory feeds procurement via `scm` integration
- Reconciliation data feeds ESG Scope 1/2 emissions calculations in `mining_env`
