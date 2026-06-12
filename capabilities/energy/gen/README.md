# Generation Management

## Overview
Generation Management provides end-to-end lifecycle management of power generation assets including thermal, hydro, and renewable plants. It covers plant registration, economic dispatch scheduling, outage management with approval workflows, KPI calculation (availability, capacity factor, heat rate), capacity planning, fuel stock monitoring, real-time SCADA ingestion, emissions tracking, predictive maintenance, and battery storage optimisation.

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
| `economic_dispatch` | LP-based merit-order dispatch minimising total generation cost |
| `unit_commitment` | Mixed-integer day-ahead commitment scheduling with ramp constraints |
| `emissions_tracking` | Per-MWh CO₂/NOₓ/SOₓ accounting with carbon intensity reporting |
| `predictive_maintenance` | Equipment health scoring from vibration, temperature, and run-hour data |
| `ancillary_services` | Spinning reserve, regulation, and reactive support market participation |
| `vre_forecasting` | Probabilistic solar/wind generation forecasts (P10/P50/P90) |
| `battery_storage` | BESS SoC tracking, charge/discharge enforcement, and arbitrage scheduling |
| `demand_response` | DR resource activation in merit order with compliance tracking |
| `grid_code_compliance` | Automated verification against EPRA, NERC CIP-014, and GB STC standards |
| `scada_ingest` | Bulk telemetry ingestion from RTU/PMU with quality-code filtering |

## Requires
| Capability | Reason |
|---|---|
| `auth` | User authentication and permission checks |
| `audl` | Immutable audit trail for all write operations |
| `mten` | Multi-tenant context isolation |
| `conf` | Capability configuration management |
| `ntfy` | Low fuel, outage, and demand-response notifications |
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

## Quick Start

```python
from capabilities.energy.gen.service import GenerationManagementService

svc = GenerationManagementService(tenant_id="ke-power", actor_id="ops-user")

# Register a CCGT plant
svc.register_plant(
    plant_id="olkaria-ccgt-1",
    tenant_id="ke-power",
    name="Olkaria CCGT Unit 1",
    plant_type="combined_cycle_gas",
    fuel_type="natural_gas",
    capacity_mw=140.0,
    owner_id="kplc",
    commissioning_date="2020-01-15",
    location_reference="Olkaria, Naivasha",
)

# Issue a dispatch instruction
await svc.dispatch_instruction(
    plant_id="olkaria-ccgt-1",
    dispatch_mw=120.0,
    start_time="2026-06-12T08:00:00Z",
    duration=4.0,
    instruction_type="economic",
    issued_by="ems-system",
)

# Record actual generation
await svc.actual_generation(
    plant_id="olkaria-ccgt-1",
    timestamp="2026-06-12T08:30:00Z",
    mw_generated=118.5,
    frequency_hz=50.02,
    power_factor=0.97,
)
```

## New Methods

### `generation_schedule` — Hourly dispatch schedules with MW validation
```python
schedule = await svc.generation_schedule(
    plant_id="olkaria-ccgt-1",
    period="2026-06",
    mw_schedule=[{"hour": h, "mw": 120.0, "duration_h": 1} for h in range(24)],
    schedule_type="day_ahead",
    approved_by="grid-operator",
)
# Returns: total_scheduled_mwh, average_scheduled_mw, status="approved"
```

### `heat_rate` — Thermal efficiency benchmarking
```python
hr = await svc.heat_rate(
    plant_id="olkaria-ccgt-1",
    period="2026-06",
    heat_input_gj=18_500.0,
    gross_generation_mwh=2_850.0,
    net_generation_mwh=2_780.0,
)
# Returns: gross_heat_rate_gj_mwh=6.49, thermal_efficiency_pct=55.5
```

### `generation_analytics` — Portfolio-level dashboard for a period
```python
analytics = await svc.generation_analytics(period="2026-06")
# Returns: total_generation_mwh, frequency_alerts, average_capacity_factor_pct,
#          average_heat_rate_gj_mwh, outage count
```

### `export_generation_data` — Bulk data export for settlement / reporting
```python
export = await svc.export_generation_data(period="2026-06", format="csv")
# format: "json" | "csv"  — returns record_count + content/records
```

### `generation_compliance_report` — Regulatory submission package
```python
report = await svc.generation_compliance_report(period="2026-06", standard="EPRA")
# Aggregates plant count, outage count, total MWh, capacity factor.
# Audit event: generation_compliance_report_generated
```

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
- `dispatch_instruction_issued`
- `outage_scheduled` / `outage_started` / `outage_completed`
- `kpi_calculated` — KPI value recorded
- `fuel_alert_triggered` — stock below threshold
- `frequency_deviation_detected` — freq outside ±0.5 Hz
- `generation_schedule_created`
- `heat_rate_calculated` / `capacity_factor_calculated`
- `regulatory_report_generated`
- `gen_agent_registered`

## Edge Cases Handled
- Dispatch MW capped at `available_mw` (capacity minus derating), not raw capacity
- Outage overlap detection prevents double-booking the same plant
- Decommissioned plants block further status transitions
- Capacity plan horizon validated to 1-20 years
- Fuel stock type validated against plant's declared fuel type
- Agent dispatch actions on live grid require human approval in contract
- Frequency deviations > ±0.5 Hz emit audit events and set `frequency_alert=True`
- Generation schedule intervals exceeding plant capacity are rejected before persisting

## World-Class Enhancements (v2.0)

These 15 improvements close the gap between the v1.0 in-memory prototype and production grid software (ERCOT, National Grid ESO, Kenya Grid Code).

| # | Enhancement | Category | Method |
|---|---|---|---|
| 1 | **Economic Dispatch Optimisation** | Dispatch Intelligence | `economic_dispatch_optimise(demand_mw, period)` — LP merit-order dispatch via `scipy`/`PuLP`, minimises generation cost, returns dispatch stack + system lambda |
| 2 | **Ramp Rate Enforcement** | Grid Physics | `check_ramp_feasibility(plant_id, target_mw, target_time)` — validates thermal ramp time before accepting dispatch instructions |
| 3 | **Unit Commitment Scheduling** | Dispatch Intelligence | `unit_commitment(demand_forecast, horizon_hours)` — mixed-integer day-ahead commitment with startup/no-load costs and min up/down time |
| 4 | **Real-Time Frequency Response** | Grid Stability | `frequency_response_event(measured_hz, nadir_hz, rocof_hz_per_s, timestamp)` — classifies severity (NERC/IEC 60255-181), triggers governor dispatch to frequency-responsive units |
| 5 | **Emissions Tracking** | Environmental Compliance | `record_emissions(plant_id, period, mwh_generated, fuel_consumed_gj)` + `carbon_intensity_report(period)` — IPCC emission factors, scope 1 CO₂e, EU ETS / Kenya Carbon Market |
| 6 | **Predictive Maintenance Scoring** | Asset Health | `compute_health_score(plant_id, run_hours, starts_count, vibration_rms_mm_s, bearing_temp_c, ...)` — composite 0-100 health index, auto-creates maintenance outage below score 40 |
| 7 | **Ancillary Services Management** | Market Services | `commit_ancillary_service(plant_id, service_type, mw, duration_h, market_price)` + `ancillary_revenue_summary(period)` — spinning/non-spinning reserve, regulation, reactive support |
| 8 | **VRE Generation Forecasting** | Renewable Integration | `generate_vre_forecast(plant_id, weather_data, horizon_hours)` — Ineichen-Perez clear-sky + Weibull power curve, returns hourly P10/P50/P90; `forecast_accuracy_report` computes MAE/RMSE |
| 9 | **Constraint Management & Redispatch** | Network Security | `check_network_constraints(dispatch_plan)` — DC load flow (PTDF matrix), greedy redispatch heuristic for N-1 security, returns feasible solution + shadow prices |
| 10 | **Battery Storage Integration** | Energy Storage | `bess_dispatch(plant_id, action, mw, duration_h)` + `bess_arbitrage_schedule(plant_id, price_forecast)` — SoC bounds, round-trip efficiency, DP arbitrage optimisation |
| 11 | **Demand Response Integration** | Demand-Side Management | `activate_demand_response(event_id, target_mw_reduction, duration_h, trigger_reason)` — merit-order DR dispatch, compliance tracking, FERC Order 745 settlement |
| 12 | **Fuel Procurement Optimisation** | Fuel Management | `optimise_fuel_procurement(plant_id, horizon_months, demand_forecast_gj)` — LP across suppliers with take-or-pay constraints, safety stock floors, cost-per-MWh |
| 13 | **Lifecycle Cost Analysis** | Asset Management | `lifecycle_cost_analysis(plant_id, discount_rate_pct, remaining_life_years, ...)` — NPV/IRR, retire/refurbish/replace recommendation, ±20% fuel price sensitivity |
| 14 | **SCADA Telemetry Ingest** | Data Integration | `ingest_scada_telemetry(telemetry_batch)` — IEC 60870-5-104/DNP3 quality codes, tag routing to entity methods, 30-second batch flush, latency metrics |
| 15 | **Grid Code Compliance Verification** | Regulatory Compliance | `verify_grid_code_compliance(plant_id, period)` — auto-evaluates EPRA / NERC CIP-014 / GB STC requirements, generates non-conformance reports (NCRs) |

## Composability Notes
- Pairs with `energy_dis` for coordinated generation-dispatch-load balancing
- Pairs with `energy_grd` for real-time dispatch instruction from grid state estimation
- Feeds `energy_bil` with metered generation data for settlement billing
- Pairs with `energy_ren` for hybrid plant portfolios (conventional + renewable)
- KPI data feeds `intel` analytics and `moni` monitoring dashboards
- DR activation emits notifications via `ntfy` capability
- Emissions records feed `comp` capability for regulatory submission
