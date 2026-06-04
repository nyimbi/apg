# Mine Production Operations

## Overview
Manages daily mine production operations including shift reporting, ore and waste movement tracking, blast design and firing authorisation, grade control boundary management, stockpile inventory, and production scheduling. Enforces a strict blast status state machine, requires fire authority before detonation, and gates grade boundary changes behind approval workflows to prevent unauthorised ore/waste misclassification.

## Capability ID
`mining_pro`

## Provides
| Service | Description |
|---|---|
| shift_report_workflow | Draft → submitted → approved shift report lifecycle |
| production_ledger_management | Per-shift ore/waste tonnage records with material type tracking |
| blast_design_workflow | Blast design creation, hole data entry, and design approval |
| blast_firing_authorization | Fire authority gated blast execution with post-blast inspection |
| ore_tracking_management | Multi-method ore movement tracking (weighbridge, belt scale, survey) |
| grade_control_workflow | Cut-off grade boundary creation, approval, and active lookup |
| production_scheduling | Weekly/monthly/LOM schedule creation, approval, and publication |
| stockpile_inventory_management | Add/reclaim movements with current tonnage tracking |
| delay_recording | Categorised production delay capture against shift reports |
| production_kpi_reporting | Aggregate ore/waste tonnes, strip ratio, delay minutes across approved shifts |

## Requires
| Capability | Reason |
|---|---|
| auth | User authentication |
| audl | Audit trail for shift sign-off and blast fire authority |
| mten | Multi-tenancy isolation |
| conf | Runtime configuration |
| ntfy | Blast fire authority notifications and shift approval alerts |
| wflo | Shift report and schedule approval workflows |
| moni | Real-time production rate monitoring |
| schd | Production schedule integration |
| mqeb | Event streaming |

## Configuration
| Key | Default | Description |
|---|---|---|
| shifts.supervisor_sign_off_required | true | Supervisor ID mandatory on shift submission |
| blasting.design_approval_required | true | Design must be approved before charging |
| blasting.fire_authority_required | true | Fire authority ID required to fire blast |
| blasting.post_blast_inspection_required | true | Post-blast inspection before clearing |
| grade_control.ore_waste_boundary_approval_required | true | Boundary approval before use |
| scheduling.plan_vs_actual_tracking | true | Enables plan/actual variance reporting |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /api/mining-pro/shifts | GET | List shift reports | mining_pro:view |
| /api/mining-pro/shifts | POST | Create shift report | mining_pro:write |
| /api/mining-pro/shifts/:id | GET/PUT | Get/update shift | mining_pro:view/write |
| /api/mining-pro/shifts/:id/submit | POST | Submit for approval | mining_pro:write |
| /api/mining-pro/shifts/:id/approve | POST | Approve shift report | mining_pro:write |
| /api/mining-pro/blasts | GET/POST | List/create blasts | mining_pro:view/blast_design |
| /api/mining-pro/blasts/:id | GET/PUT | Get/update blast | mining_pro:view |
| /api/mining-pro/blasts/:id/approve-design | POST | Approve blast design | mining_pro:blast_design |
| /api/mining-pro/blasts/:id/fire | POST | Fire a blast | mining_pro:blast_design |
| /api/mining-pro/grade-boundaries | POST | Create grade boundary | mining_pro:grade_control |
| /api/mining-pro/grade-boundaries/:id/approve | POST | Approve boundary | mining_pro:grade_control |
| /api/mining-pro/stockpiles | GET/POST | List/create stockpiles | mining_pro:view/write |
| /api/mining-pro/stockpiles/movements | POST | Record movement | mining_pro:write |
| /api/mining-pro/schedules | POST | Create schedule | mining_pro:schedule |
| /api/mining-pro/schedules/:id/publish | POST | Publish schedule | mining_pro:schedule |
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

## Data Models
| Model | Key Fields |
|---|---|
| ShiftReportCreate/Response | shift_type, shift_date, supervisor_id, activities[], delays[], total_ore_tonnes, status |
| BlastCreate/Response | blast_name, blast_type, mine_area, holes[], status (state machine), fire_authority_id |
| GradeBoundaryCreate/Response | mine_area, commodity, cut_off_grade, method, ore_boundary_coords, approved |
| StockpileCreate/Response | stockpile_type, mine_area, current_tonnes, capacity_tonnes |
| ProductionScheduleCreate/Response | schedule_type, period, planned_ore/waste_tonnes, approved, published |

## Streaming Events
- `shift_report_submitted` / `shift_report_approved`
- `production_tonnes_recorded`
- `blast_designed` / `blast_fired` / `blast_cleared`
- `grade_boundary_updated`
- `stockpile_movement_recorded`
- `production_schedule_published`
- `delay_recorded`

## Edge Cases Handled
- Blast state machine strictly enforced; skipping states (e.g. planned → fired) raises ValueError
- Stockpile reclaim exceeding current inventory raises ValueError before mutation
- Approved shift reports cannot be modified; returns 400
- Future-dated shift reports rejected
- Grade boundary active lookup returns most recently approved record covering current datetime
- Negative or zero ore tonnes rejected at model validation layer

## Composability Notes
- Blast firing events feed `mining_saf` hazard and incident monitoring
- Ore tonnage data feeds `mining_ore` plant feed records via stockpile movements
- Grade boundaries derived from `mining_exp` resource estimation cutoff grades
- Shift delay records feed `mining_eqp` availability KPI calculations
- Production schedules consumed by `schd` for resource planning
