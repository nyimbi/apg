# Generation Management

## Overview
Generation Management provides end-to-end lifecycle management of power generation assets including thermal, hydro, and renewable plants. It covers plant registration, economic dispatch scheduling, outage management with approval workflows, KPI calculation (availability, capacity factor, heat rate), capacity planning, and fuel stock monitoring with low-supply alerting.

## Capability ID
`energy_gen`

## Provides
| Service | Description |
|---|---|
| `plant_registry` | Register and manage generation plants by type, fuel, and status |
| `dispatch_scheduling` | Create and approve dispatch schedules with mode and MW validation |
| `outage_management` | Schedule, approve, start and complete plant outages |
| `capacity_planning` | Multi-year capacity plans with reserve margin tracking |
| `generation_kpis` | Calculate and store KPIs: availability, capacity factor, heat rate, EAF |
| `fuel_management` | Track fuel stock levels and emit low-supply alerts |
| `performance_reporting` | Aggregate generation performance by plant and period |
| `dispatch_optimization` | Agent-driven dispatch optimization within physical constraints |

## Requires
| Capability | Reason |
|---|---|
| `auth` | User authentication and permission checks |
| `audl` | Immutable audit trail for all write operations |
| `mten` | Multi-tenant context isolation |
| `conf` | Capability configuration management |
| `ntfy` | Low fuel and outage notifications |
| `wflo` | Outage and dispatch approval workflows |
| `moni` | Real-time plant monitoring integration |
| `schd` | Scheduled KPI calculations and dispatch windows |
| `mqeb` | Bytewax event streaming for dispatch batches |
| `comp` | Environmental compliance and emissions reporting |

## Configuration
| Key | Type | Default | Description |
|---|---|---|---|
| `plants.capacity_unit` | string | `mw` | Unit for capacity values |
| `outages.minimum_notice_hours` | int | 24 | Minimum outage lead time |
| `kpis.auto_calculate` | bool | true | Auto-calculate KPIs after readings |
| `capacity.planning_horizon_years` | int | 10 | Maximum planning horizon |
| `fuel.alert_threshold_days` | float | 7.0 | Days of supply below which alert fires |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| `/energy-gen/api/v1/dashboard` | GET | Dashboard summary | `energy_gen:view` |
| `/energy-gen/api/v1/plants` | GET | List plants | `energy_gen:plants` |
| `/energy-gen/api/v1/plants` | POST | Register plant | `energy_gen:plants` |
| `/energy-gen/api/v1/plants/<id>` | GET | Plant detail | `energy_gen:plants` |
| `/energy-gen/api/v1/plants/<id>/status` | PUT | Update plant status | `energy_gen:plants` |
| `/energy-gen/api/v1/schedules` | GET | List dispatch schedules | `energy_gen:dispatch` |
| `/energy-gen/api/v1/schedules` | POST | Create dispatch schedule | `energy_gen:dispatch` |
| `/energy-gen/api/v1/schedules/<id>/approve` | PUT | Approve schedule | `energy_gen:dispatch` |
| `/energy-gen/api/v1/outages` | GET | List outages | `energy_gen:outages` |
| `/energy-gen/api/v1/outages` | POST | Schedule outage | `energy_gen:outages` |
| `/energy-gen/api/v1/outages/<id>/approve` | PUT | Approve outage | `energy_gen:outages` |
| `/energy-gen/api/v1/kpis` | GET | List KPIs | `energy_gen:kpis` |
| `/energy-gen/api/v1/kpis` | POST | Record KPI | `energy_gen:kpis` |
| `/energy-gen/api/v1/fuel` | POST | Update fuel stock | `energy_gen:fuel` |
| `/energy-gen/api/v1/agents` | POST | Register agent | `energy_gen:admin` |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| `tenant_context_required` | tenant_context_present=False | deny |
| `write_requires_policy` | operation_type=write AND policy_attached=False | deny |
| `plant_type_supported` | plant_type not in supported list | deny |
| `plant_capacity_positive` | capacity_mw <= 0 | deny |
| `dispatch_mw_within_capacity` | scheduled_mw > available_mw | deny |
| `dispatch_schedule_approval_required` | activating without approval | deny |
| `outage_overlap_check` | another active outage on same plant | deny |
| `outage_notice_period` | lead time < minimum_notice_hours | deny |
| `fuel_stock_non_negative` | quantity < 0 | deny |
| `cross_tenant_denied` | cross_tenant_access=True | deny |
| `plant_decommission_requires_approval` | decommission without approval | deny |
| `privileged_gen_agent_requires_human_approval` | agent dispatch action without human approval | deny |

## Data Models
| Model | Key Fields |
|---|---|
| `GenPlant` | id, tenant_id, name, plant_type, fuel_type, capacity_mw, status, owner_id, commissioning_date |
| `DispatchSchedule` | id, plant_id, dispatch_mode, scheduled_mw, start_time, end_time, status, approved_by |
| `PlantOutage` | id, plant_id, outage_type, status, planned_start, planned_end, derated_mw |
| `GenerationKPI` | id, plant_id, kpi_type, period, value, unit, calculated_at |
| `CapacityPlan` | id, plan_name, horizon_years, total_existing_mw, peak_demand_mw, reserve_margin_pct |
| `FuelStock` | id, plant_id, fuel_type, quantity, days_of_supply, is_low |
| `GenAgent` | id, name, runtime, role, scope |

## Streaming Events
- `plant_registered` — new plant added to registry
- `plant_status_changed` — operational status transition
- `dispatch_schedule_created` / `dispatch_schedule_approved`
- `outage_scheduled` / `outage_started` / `outage_completed`
- `kpi_calculated` — KPI value recorded
- `fuel_alert_triggered` — stock below threshold
- `gen_agent_registered`

## Edge Cases Handled
- Dispatch MW capped at `available_mw` (capacity minus derating), not raw capacity
- Outage overlap detection prevents double-booking the same plant
- Decommissioned plants block further status transitions
- Capacity plan horizon validated to 1-20 years
- Fuel stock type validated against plant's declared fuel type
- Agent dispatch actions on live grid require human approval in contract

## Composability Notes
- Pairs with `energy_dis` for coordinated generation-dispatch-load balancing
- Pairs with `energy_grd` for real-time dispatch instruction from grid state estimation
- Feeds `energy_bil` with metered generation data for settlement billing
- Pairs with `energy_ren` for hybrid plant portfolios (conventional + renewable)
- KPI data feeds `intel` analytics and `moni` monitoring dashboards
