# Mine Production Operations

## Overview
Manages daily mine production operations including shift reporting, ore and waste movement tracking, blast design and firing authorisation, grade control boundary management, stockpile inventory, and production scheduling. Enforces a strict blast status state machine, requires fire authority before detonation, and gates grade boundary changes behind approval workflows to prevent unauthorised ore/waste misclassification.

New in v1.1: real-time truck dispatch, blast vibration compliance monitoring, block model grade reconciliation, Short-Interval Control (SIC) reporting, automated shift handover packages, equipment availability (PA/MA/U per SMRP), explosives consumption reconciliation, and delay Pareto analysis.

## Capability ID
`mining_pro`

## Provides
| Service | Description |
|---|---|
| shift_report_workflow | Draft → submitted → approved shift report lifecycle |
| production_ledger_management | Per-shift ore/waste tonnage records with material type tracking |
| blast_design_workflow | Blast design creation, hole data entry, and design approval |
| blast_firing_authorization | Fire authority gated blast execution with post-blast inspection |
| blast_vibration_compliance | PPV measurement recording with automatic limit breach detection |
| ore_tracking_management | Multi-method ore movement tracking (weighbridge, belt scale, survey) |
| grade_control_workflow | Cut-off grade boundary creation, approval, and active lookup |
| block_model_reconciliation | F-factor, C-factor, E-factor reconciliation against geological block model |
| production_scheduling | Weekly/monthly/LOM schedule creation, approval, and publication |
| short_interval_control | 2–4 hour SIC reporting with variance alerts via NATS |
| stockpile_inventory_management | Add/reclaim movements with current tonnage tracking |
| truck_dispatch | Real-time truck assignment with priority queuing |
| delay_recording | Categorised production delay capture against shift reports |
| delay_pareto_analysis | Ranked Pareto of delay categories with capability escalation mapping |
| equipment_availability | PA/MA/U per SMRP definitions, JORC 85% PA compliance check |
| explosives_reconciliation | Magazine issue vs blast plan variance with 2 kg compliance threshold |
| shift_handover_automation | Structured handover package with blast holds, stockpile snapshot, and safety items |
| production_kpi_reporting | Aggregate ore/waste tonnes, strip ratio, delay minutes across approved shifts |

## Requires
| Capability | Reason |
|---|---|
| auth | User authentication |
| audl | Audit trail for shift sign-off and blast fire authority |
| mten | Multi-tenancy isolation |
| conf | Runtime configuration including PPV limits and SIC thresholds |
| ntfy | Blast fire authority notifications, shift approval alerts, MCF and PPV breach alerts |
| wflo | Shift report and schedule approval workflows |
| moni | Real-time production rate monitoring |
| schd | Production schedule integration |
| mqeb | NATS event streaming for dispatch, SIC, vibration, and handover events |
| mining_eqp | Equipment availability and fault data cross-reference |
| mining_saf | Safety hold status and hazard incident feeds |
| mining_env | Environmental compliance for dust and water discharge tracking |

## Configuration
| Key | Default | Description |
|---|---|---|
| shifts.supervisor_sign_off_required | true | Supervisor ID mandatory on shift submission |
| blasting.design_approval_required | true | Design must be approved before charging |
| blasting.fire_authority_required | true | Fire authority ID required to fire blast |
| blasting.post_blast_inspection_required | true | Post-blast inspection before clearing |
| blasting.ppv_limit_residential_mmps | 5.0 | PPV limit at residential receivers (mm/s) |
| blasting.ppv_limit_industrial_mmps | 25.0 | PPV limit at industrial receivers (mm/s) |
| grade_control.ore_waste_boundary_approval_required | true | Boundary approval before use |
| scheduling.plan_vs_actual_tracking | true | Enables plan/actual variance reporting |
| sic.critical_variance_threshold_pct | 15.0 | SIC negative variance % that triggers critical alert |
| reconciliation.f_factor_tolerance | 0.10 | Block model F-factor tolerance before alert (±10%) |
| explosives.reconciliation_tolerance_kg | 2.0 | Per-type explosives variance tolerance before compliance flag |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /api/mining-pro/shifts | GET | List shift reports | mining_pro:view |
| /api/mining-pro/shifts | POST | Create shift report | mining_pro:write |
| /api/mining-pro/shifts/:id | GET/PUT | Get/update shift | mining_pro:view/write |
| /api/mining-pro/shifts/:id/submit | POST | Submit for approval | mining_pro:write |
| /api/mining-pro/shifts/:id/approve | POST | Approve shift report | mining_pro:write |
| /api/mining-pro/shifts/:id/handover | POST | Generate handover package | mining_pro:write |
| /api/mining-pro/blasts | GET/POST | List/create blasts | mining_pro:view/blast_design |
| /api/mining-pro/blasts/:id | GET/PUT | Get/update blast | mining_pro:view |
| /api/mining-pro/blasts/:id/approve-design | POST | Approve blast design | mining_pro:blast_design |
| /api/mining-pro/blasts/:id/fire | POST | Fire a blast | mining_pro:blast_design |
| /api/mining-pro/blasts/:id/vibration | POST | Record PPV measurement | mining_pro:blast_design |
| /api/mining-pro/grade-boundaries | POST | Create grade boundary | mining_pro:grade_control |
| /api/mining-pro/grade-boundaries/:id/approve | POST | Approve boundary | mining_pro:grade_control |
| /api/mining-pro/reconciliation/block/:id | POST | Block model reconciliation | mining_pro:grade_control |
| /api/mining-pro/stockpiles | GET/POST | List/create stockpiles | mining_pro:view/write |
| /api/mining-pro/stockpiles/movements | POST | Record movement | mining_pro:write |
| /api/mining-pro/schedules | POST | Create schedule | mining_pro:schedule |
| /api/mining-pro/schedules/:id/publish | POST | Publish schedule | mining_pro:schedule |
| /api/mining-pro/sic | POST | Short-interval control report | mining_pro:write |
| /api/mining-pro/dispatch | POST | Dispatch truck | mining_pro:dispatch |
| /api/mining-pro/analytics/delays/pareto | GET | Delay Pareto analysis | mining_pro:view |
| /api/mining-pro/analytics/equipment/:id/availability | GET | Equipment PA/MA/U report | mining_pro:view |
| /api/mining-pro/explosives/reconciliation | POST | Explosives reconciliation | mining_pro:compliance |
| /api/mining-pro/summary | GET | Production KPI summary | mining_pro:view |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| blast_design_approval_required | Charging without approval | DENY |
| blast_fire_authority_required | Firing without authority | DENY |
| post_blast_inspection_required | Clearing without inspection | DENY |
| grade_boundary_bypass_denied | Bypass without authority | DENY |
| ore_waste_boundary_approval_required | Unapproved boundary in use | DENY |
| negative_tonnes_denied | Negative tonnage entry | DENY |
| future_shift_report_denied | Shift date in future | DENY |
| ore_tracking_method_required | Missing tracking method | DENY |
| schedule_approval_required | Publishing unapproved schedule | DENY |
| cross_tenant_read_denied | Cross-tenant access | DENY |
| ppv_limit_breach_notified | PPV exceeds configured limit | ALERT (NATS + ntfy) |
| f_factor_variance_alerted | Block model F-factor deviation > ±10% | ALERT (NATS) |
| sic_critical_variance_alerted | SIC gap > 15% below target | ALERT (NATS) |
| explosives_variance_flagged | Per-type variance > 2 kg | COMPLIANCE FLAG |
| duplicate_truck_dispatch_denied | Truck already in_transit | DENY |

## Data Models
| Model | Key Fields |
|---|---|
| ShiftReportCreate/Response | shift_type, shift_date, supervisor_id, activities[], delays[], total_ore_tonnes, status |
| BlastCreate/Response | blast_name, blast_type, mine_area, holes[], status (state machine), fire_authority_id |
| GradeBoundaryCreate/Response | mine_area, commodity, cut_off_grade, method, ore_boundary_coords, approved |
| StockpileCreate/Response | stockpile_type, mine_area, current_tonnes, capacity_tonnes |
| ProductionScheduleCreate/Response | schedule_type, period, planned_ore/waste_tonnes, approved, published |

## NATS Streaming Events
| Subject | Trigger |
|---|---|
| `mining.pro.dispatch.{mine_area}` | Truck dispatch assignment |
| `mining.pro.handover` | Shift handover package generated |
| `blast_vibration_breach` | PPV measurement exceeds limit |
| `sic.variance.critical` | SIC cumulative gap exceeds 15% |
| `shift_report_submitted` / `shift_report_approved` | Shift workflow transitions |
| `production_tonnes_recorded` | Ore movement recorded |
| `blast_designed` / `blast_fired` / `blast_cleared` | Blast state machine transitions |
| `grade_boundary_updated` | Grade control boundary change |
| `stockpile_movement_recorded` | Stockpile add/reclaim |
| `production_schedule_published` | Schedule approved and live |
| `delay_recorded` | Production delay captured |
| `reconciliation_variance_alert` | Block model F-factor breach |

## Edge Cases Handled
- Blast state machine strictly enforced; skipping states (e.g. planned → fired) raises ValueError
- Stockpile reclaim exceeding current inventory raises ValueError before mutation
- Approved shift reports cannot be modified; returns 400
- Future-dated shift reports rejected
- Grade boundary active lookup returns most recently approved record covering current datetime
- Negative or zero ore tonnes rejected at model validation layer
- Truck dispatch rejected if truck already has an active in_transit record
- Misfire automatically sets safety_hold=True; cannot be overridden without explicit clearance
- Explosives reconciliation compares all explosive types in both magazine issues and blast plans
- SIC targets derived from published schedule; no schedule → target is None, not zero

## Composability Notes
- Blast firing events feed `mining_saf` hazard and incident monitoring
- Ore tonnage data feeds `mining_ore` plant feed records via stockpile movements
- Grade boundaries derived from `mining_exp` resource estimation cutoff grades
- Shift delay records feed `mining_eqp` availability KPI calculations
- Production schedules consumed by `schd` for resource planning
- Truck dispatch integrates with `mining_eqp` fleet status for availability checks
- Equipment availability report cross-references `mining_eqp` for maintenance categories
- SIC variance alerts consumed by supervisor dashboards via `moni`
- Explosives reconciliation feeds compliance reporting to `audl`

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Real-Time Truck Dispatch Optimisation** [Operational Intelligence]
- **I2. Blast Vibration Compliance Monitoring** [Safety & Regulatory]
- **I3. Block Model Grade Reconciliation** [Ore Value Chain]
- **I4. Equipment Utilisation and Availability Dashboarding** [Asset Management]
- **I5. NATS-Based Real-Time Production Event Streaming** [Integration / Streaming Architecture]
- **I6. Automated Mine Call Factor (MCF) Calculation** [Metallurgical Accounting]
- **I7. Short-Interval Control (SIC) Feedback Loop** [Production Optimisation]
- **I8. Geofenced Face Status Management** [Spatial Operations]
- **I9. Explosives Consumption Reconciliation** [Compliance & Cost Control]
- **I10. Automated Delay Pareto and Root-Cause Classification** [Continuous Improvement]
- **I11. Integrated Production Forecast (Monte Carlo)** [Planning Intelligence]
- **I12. Multi-Level Schedule Lock and Freeze Protocol** [Change Management / Governance]
- **I13. Automated Shift Handover Package** [Operational Continuity]
- **I14. Environmental Compliance: Dust and Water Discharge Tracking** [ESG / Regulatory]
- **I15. AI-Assisted Blast Design Optimisation** [Advanced Analytics / AI]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
