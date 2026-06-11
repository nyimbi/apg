# Generation Management — World-Class Improvements

**Capability**: `energy_gen` | **Domain**: `energy` | **Version**: 1.0.0 → 2.0.0

Fifteen improvements distilled from operational experience at OSIsoft (now AVEVA), GE Digital, Siemens Spectrum Power, AVEVA System Platform, and modern grid operators (ERCOT, National Grid ESO, Kenya Power). Each improvement closes a specific gap between the current in-memory prototype and production grid software.

---

## 1. Economic Dispatch Optimisation (Linear Programming)

**Category**: Dispatch Intelligence

**Justification**: The current service issues a single-plant dispatch instruction with no portfolio optimisation. Real ISOs (ERCOT, National Grid ESO) run security-constrained economic dispatch (SCED) every 5 minutes. Without LP-based merit-order dispatch, cheaper plants are left idle while expensive ones run, directly increasing generation cost.

**Implementation**: Add `async def economic_dispatch_optimise(self, demand_mw, period, ...)` that uses `scipy.optimize.linprog` or `PuLP` to minimise total generation cost subject to: (a) sum of dispatched MW equals demand, (b) each plant MW within [min_stable_load, available_mw], (c) ramp rate constraints per 5-min interval. Returns dispatch stack sorted by marginal cost with shadow price for the system lambda (marginal cost of energy).

**Competitor Reference**: GE Digital APEX SCED engine; ERCOT DAM solver; ENTSO-E merit-order dispatch.

---

## 2. Ramp Rate Enforcement

**Category**: Grid Physics

**Justification**: Thermal plants cannot instantaneously change output. A 100 MW coal unit may ramp at 2 MW/min. Ignoring ramp constraints during dispatch creates physically unrealisable schedules that operators must manually correct, causing voltage deviations and frequency excursions.

**Implementation**: Store `ramp_rate_mw_per_min` on `GenPlant`. In `dispatch_instruction`, compute the time required to reach `dispatch_mw` from current output: `ramp_time_min = abs(target_mw - current_mw) / ramp_rate`. Reject instructions whose `start_time` is sooner than ramp time allows. Add `async def check_ramp_feasibility(plant_id, target_mw, target_time)` returning feasibility flag and earliest achievable time.

**Competitor Reference**: Siemens Spectrum Power OPF ramp constraint module; ABB Network Manager AGC.

---

## 3. Unit Commitment Scheduling (Mixed-Integer)

**Category**: Dispatch Intelligence

**Justification**: Day-ahead scheduling must decide which units to commit (start up / shut down) hour by hour considering minimum up/down times, start-up costs, and no-load costs. Running only economic dispatch without commitment leads to sub-optimal commitment decisions and excessive start-stop cycling that degrades turbine life.

**Implementation**: Add `async def unit_commitment(self, demand_forecast: list[float], horizon_hours: int)` using `mip` (Python-MIP) or CBC via `PuLP`. Decision variables: binary commitment state per plant-hour, continuous generation MW. Objective: minimise sum(startup_cost + no_load_cost + marginal_cost × MW). Constraints: demand balance, ramp, min up/down time. Returns hourly commitment schedule with per-unit economics.

**Competitor Reference**: PLEXOS unit commitment; PowerWorld Simulator; GAMS TIMES model.

---

## 4. Real-Time Frequency Response Monitoring

**Category**: Grid Stability

**Justification**: The current `actual_generation` method flags a frequency alert at ±0.5 Hz but takes no corrective action. NERC BAL-003 requires frequency response obligation tracking. Under-frequency load shedding (UFLS) and governor response must be coordinated automatically within 10–30 seconds of a frequency event.

**Implementation**: Add `async def frequency_response_event(self, measured_hz, nadir_hz, rocof_hz_per_s, timestamp)`. Classify event severity (normal / alert / emergency / UFLS) using thresholds from IEC 60255-181. Identify committed plants with governor response enabled, compute expected MW contribution per plant from droop setting (R%), and emit priority dispatch instructions to frequency-responsive units. Log event with ROCOF and nadir for post-event analysis.

**Competitor Reference**: National Grid ESO Mandatory Frequency Response; ERCOT UFLS programme; ABB Relion 670 series governor control.

---

## 5. Emissions Tracking and Carbon Intensity Reporting

**Category**: Environmental Compliance

**Justification**: Carbon pricing schemes (EU ETS, Kenya Carbon Market) require per-MWh CO₂, NOₓ, and SOₓ accounting. Regulators (EPRA, NERC CIP) increasingly mandate real-time emissions dashboards. The current service has no emissions model.

**Implementation**: Add `async def record_emissions(self, plant_id, period, mwh_generated, fuel_consumed_gj)`. Compute CO₂ equivalent using IPCC emission factors per fuel type (natural gas: 56.1 kg CO₂/GJ; coal: 94.6 kg CO₂/GJ; HFO: 77.4 kg CO₂/GJ). Store `EmissionRecord` with scope 1 CO₂e tonnes, carbon intensity (kg CO₂/MWh), and remaining allowance against allocated quota. Add `async def carbon_intensity_report(period)` aggregating fleet-level intensity for regulatory submission.

**Competitor Reference**: AVEVA Emissions Monitoring; OSIsoft PI AF emissions calculations; EPA eReporting.

---

## 6. Predictive Maintenance Scoring

**Category**: Asset Health

**Justification**: Unplanned forced outages cost utilities $50–200k/day in replacement power costs. GE APM and IBM Maximo use equipment health scores derived from vibration, temperature, and run-hour data to schedule maintenance before failure. The current service only records outages after they occur.

**Implementation**: Add `async def compute_health_score(self, plant_id, run_hours, starts_count, vibration_rms_mm_s, bearing_temp_c, last_major_overhaul_date)`. Score each indicator against OEM threshold bands (green/amber/red). Aggregate into a composite health index (0–100) using weighted sum. When score drops below 40, automatically create a recommended maintenance outage. Store score in `_health_score_records` with trend over last 12 readings.

**Competitor Reference**: GE APM (Asset Performance Management); Bentley APM; Emerson Plantweb Optics.

---

## 7. Ancillary Services Management (Reserves, Regulation)

**Category**: Market Services

**Justification**: Beyond energy, plants can provide frequency regulation (AGC), spinning reserve, non-spinning reserve, and reactive power support. These ancillary services often command premium prices. The current model treats all dispatch as energy-only, leaving ancillary revenue on the table.

**Implementation**: Add `AncillaryService` model with fields: `service_type` (spinning_reserve | non_spinning | regulation_up | regulation_down | reactive_support), `mw_committed`, `response_time_s`, `price_mw_hr`. Add `async def commit_ancillary_service(plant_id, service_type, mw, duration_h, market_price)`. Validate that committed ancillary MW + energy dispatch MW ≤ available_mw. Add `async def ancillary_revenue_summary(period)` returning revenue by service type across the fleet.

**Competitor Reference**: ERCOT Ancillary Services market; PJM Regulation market; National Grid BM ancillary procurement.

---

## 8. Generation Forecasting (Solar/Wind Intermittency)

**Category**: Renewable Integration

**Justification**: Solar and wind output is weather-dependent. Without probabilistic forecasts the system cannot pre-position conventional reserves to cover renewable variability. ENTSO-E mandates 72-hour generation forecasts for variable renewable energy (VRE) sources.

**Implementation**: Add `async def generate_vre_forecast(self, plant_id, weather_data: list[dict], horizon_hours: int)`. For solar: apply clear-sky irradiance model (Ineichen-Perez) × panel efficiency × derating factor. For wind: use Weibull-calibrated power curve (cut-in, rated, cut-out speeds). Return hourly P10/P50/P90 generation bands. Persist forecast in `_vre_forecasts` store. Add `async def forecast_accuracy_report(plant_id, period)` computing MAE and RMSE against actuals.

**Competitor Reference**: Meteologica VRE forecasting; SolarEdge forecast API; Vestas wind power forecasting.

---

## 9. Constraint Management and Redispatch

**Category**: Network Security

**Justification**: Physical transmission constraints can prevent economic dispatch solutions from being deliverable. When a line is congested, the operator must redispatch: increase generation on one side of the constraint (inc) and decrease it on the other (dec). Without constraint management, the cheapest dispatch may violate N-1 security criteria.

**Implementation**: Add `NetworkConstraint` model with `from_bus`, `to_bus`, `max_mw`, `current_flow_mw`. Add `async def check_network_constraints(self, dispatch_plan: dict[str, float])` that applies a DC load flow (PTDF matrix) to verify all constraint margins. Where violations are detected, implement a greedy redispatch heuristic: relieve the most binding constraint by swapping MW between the cheapest inc and dec plant pair on opposite sides. Return the feasible redispatch solution with constraint shadow prices.

**Competitor Reference**: Siemens Spectrum Power Congestion Management; PowerFactory DIgSILENT; PSS/E GE.

---

## 10. Battery Storage Integration and Charge/Dispatch Optimisation

**Category**: Energy Storage

**Justification**: Battery energy storage systems (BESS) are increasingly co-located with generation. A BESS charges at low-price/high-generation periods and discharges during peak demand. The current plant model has no state-of-charge (SoC) tracking or charge/discharge cycle management.

**Implementation**: Extend `GenPlant` with optional BESS fields: `battery_capacity_mwh`, `max_charge_rate_mw`, `max_discharge_rate_mw`, `round_trip_efficiency`, `soc_mwh`. Add `async def bess_dispatch(self, plant_id, action: str, mw: float, duration_h: float)` where action ∈ {charge, discharge, idle}. Enforce SoC bounds [0, battery_capacity_mwh] accounting for round-trip efficiency losses. Add `async def bess_arbitrage_schedule(plant_id, price_forecast: list[float])` that uses dynamic programming to find the charge/discharge sequence maximising revenue.

**Competitor Reference**: Tesla Autobidder; Fluence Mosaic; Wartsila GEMS energy management.

---

## 11. Demand Response Integration

**Category**: Demand-Side Management

**Justification**: Demand response (DR) resources can substitute for spinning reserve, reducing need to hold expensive fast-start peakers. FERC Order 745 mandates that DR resources receive LMP (locational marginal price) equal to generators. The current service models only supply-side resources.

**Implementation**: Add `DemandResponseResource` model with `facility_id`, `enrolled_mw`, `response_time_min`, `price_ceiling`, `notification_method`. Add `async def activate_demand_response(self, event_id, target_mw_reduction, duration_h, trigger_reason)`. Dispatch DR resources in merit order (cheapest first) until target reduction is met. Track activation history, compliance (actual vs committed reduction), and settlement credit. Emit notification via `ntfy` capability.

**Competitor Reference**: EnerNOC (now Enel X) DR platform; AutoGrid Flex; Oracle Utilities DR.

---

## 12. Multi-Period Fuel Procurement Optimisation

**Category**: Fuel Management

**Justification**: The current fuel management tracks stock levels reactively. Procurement decisions (when to order, how much, from which supplier) should be optimised across multiple periods to minimise total fuel cost while maintaining minimum stock buffers and respecting contract take-or-pay obligations.

**Implementation**: Add `FuelContract` model with `supplier_id`, `fuel_type`, `min_take_gj`, `max_take_gj`, `price_gj`, `delivery_lead_days`, `contract_start`, `contract_end`. Add `async def optimise_fuel_procurement(self, plant_id, horizon_months: int, demand_forecast_gj: list[float])`. Use LP to minimise total procurement cost across suppliers subject to: stock balance equation, min/max contract take, storage capacity bounds, and safety stock floor. Returns month-by-month order quantities per supplier with total cost and cost-per-MWh.

**Competitor Reference**: SAP Energy Management; Ventyx Fuel Management; EnergySys commodity management.

---

## 13. Lifecycle Cost Analysis and Retirement Planning

**Category**: Asset Management

**Justification**: Power plants have 30–60 year operational lifespans. Investment decisions (refurbishment vs retirement) require discounted cash flow (DCF) analysis comparing continued operation costs against replacement alternatives. The current service only records commissioning date with no lifecycle economics.

**Implementation**: Add `async def lifecycle_cost_analysis(self, plant_id, discount_rate_pct: float, remaining_life_years: int, annual_opex: float, capex_refurbishment: float)`. Compute NPV of continued operation including O&M escalation, fuel costs, carbon costs, and salvage value. Compare against NPV of replacement with a new-build alternative. Return IRR, payback period, and retire/refurbish/replace recommendation with sensitivity analysis across ±20% fuel price scenarios.

**Competitor Reference**: PLEXOS transmission/generation planning; ABB AbilityTM; Ventyx e.Forecast.

---

## 14. Real-Time SCADA Data Ingestion Pipeline

**Category**: Data Integration

**Justification**: Production generation management systems receive telemetry from RTUs and PMUs via ICCP/IEC 60870-5-104 or DNP3. The current service accepts only manually-posted generation readings with no time-series ingest pipeline, making real-time monitoring impossible.

**Implementation**: Add `async def ingest_scada_telemetry(self, telemetry_batch: list[dict])` accepting bulk telemetry records with fields: `plant_id`, `timestamp`, `tag_name`, `value`, `quality`. Validate quality codes (0 = good, 64 = questionable, 128 = bad) and reject or flag bad-quality readings. Route each tag to the appropriate entity (MW output → `actual_generation`, fuel flow → `fuel_consumption`, vibration → `compute_health_score`). Buffer in `_scada_buffer` and flush to persistent store in 30-second batches. Emit `scada_telemetry_received` audit event with ingestion latency metric.

**Competitor Reference**: OSIsoft PI (now AVEVA PI System); GE Historian; Honeywell PHD.

---

## 15. Automated Compliance Verification Against Grid Code

**Category**: Regulatory Compliance

**Justification**: Grid codes (Kenya Grid Code, GB Grid Code, NERC Reliability Standards) define performance obligations: frequency response deadband, reactive capability curve, protection relay settings, fault ride-through. Manual compliance checking is error-prone and creates regulatory exposure. Automated verification closes the gap between operational data and compliance obligations.

**Implementation**: Add `GridCodeRequirement` model with `standard_id`, `requirement_id`, `description`, `measurement_tag`, `threshold_value`, `comparison_operator`, `enforcement_action`. Add `async def verify_grid_code_compliance(self, plant_id, period)`. Pull relevant measurements from `_generation_records` and `_heat_rate_records`. Evaluate each applicable requirement against thresholds. Return `ComplianceVerificationReport` with pass/fail per requirement, evidence references, and auto-generated non-conformance reports (NCRs) for any failures. Supports EPRA, NERC CIP-014, and GB STC formats.

**Competitor Reference**: NERC e-Tag compliance; ENTSO-E TYNDP compliance toolkit; Siemens PSG regulatory reporting.
