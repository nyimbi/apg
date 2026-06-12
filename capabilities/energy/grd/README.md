# Grid Operations

## Overview

Grid Operations (`energy_grd`) provides the real-time operational intelligence layer for power system management. It covers WLS/LAV/EKF state estimation with convergence tracking, N-1/N-2 contingency analysis with automatic system status classification, voltage control via tap changers/SVCs/STATCOMs, frequency monitoring with PMU integration and automatic control action dispatch, market interval settlement with imbalance calculation, a full grid alarm management system with severity-gated acknowledgement, ancillary services procurement, islanding detection, black-start restoration planning, reactive power dispatch, and EMS function execution in real-time and study modes.

## Capability ID

`energy_grd`

## Provides

| Service | Description |
|---|---|
| `real_time_state_estimation` | Run and record WLS/LAV/EKF state estimation with convergence status |
| `contingency_analysis` | N-1/N-2 contingency analysis with violation detection and system status |
| `voltage_control` | Record and approve voltage control actions across multiple methods |
| `frequency_control` | Record frequency response actions, configure UFLS thresholds, PMU integration |
| `market_settlement` | Settle market dispatch intervals with imbalance and price calculation |
| `grid_alarm_management` | Raise, acknowledge and clear alarms with severity-gated workflow |
| `ems_function_management` | Execute and record EMS functions in real-time or study mode |
| `islanding_detection` | Multi-indicator passive islanding detection with automatic alarm |
| `black_start_planning` | Black-start restoration sequence planning and approval |
| `ancillary_services_procurement` | Frequency regulation, spinning reserve, voltage support procurement |
| `reactive_power_dispatch` | Reactive power schedule dispatch to generators and compensators |
| `grid_analytics` | SE convergence rates, frequency events, contingency violations, alarm stats |
| `grid_operational_reporting` | Operational summaries, alarm histories, settlement reports |

## Requires

| Capability | Reason |
|---|---|
| `auth` | Operator authentication and control room access |
| `audl` | Immutable audit trail for all control actions |
| `mten` | Multi-tenant grid data isolation |
| `conf` | State estimator and market configuration |
| `ntfy` | Critical alarm notifications to control room |
| `wflo` | Voltage and frequency control approval workflows |
| `moni` | Real-time grid telemetry monitoring |
| `comp` | Market settlement regulatory compliance |
| `mqeb` | Event streaming for alarm and SE lifecycle |
| `schd` | Scheduled SE runs and contingency analysis windows |

## Configuration

| Key | Type | Default | Description |
|---|---|---|---|
| `state_estimation.run_interval_seconds` | int | 30 | SE execution interval |
| `state_estimation.convergence_threshold` | float | 1e-4 | Residual threshold for convergence |
| `voltage_control.target_pu` | float | 1.0 | Nominal voltage setpoint |
| `voltage_control.tolerance_pu` | float | 0.05 | Voltage control tolerance band |
| `frequency_control.nominal_hz` | float | 50.0 | System nominal frequency |
| `frequency_control.ufls_threshold_hz` | float | 49.0 | UFLS trigger frequency |
| `contingency.n_1_mandatory` | bool | true | N-1 analysis cannot be skipped |
| `alarms.critical_requires_acknowledgement` | bool | true | Critical alarms must be acked before clearing |

## Quick Start

```python
from capabilities.energy.grd.service import GridOperationsService

svc = GridOperationsService(tenant_id="ke_tso", actor_id="operator_1")

# State estimation from SCADA snapshot
result = await svc.state_estimation(
    timestamp="2026-06-01T06:00:00Z",
    sensor_readings={"bus_1_voltage_pu": 1.02, "line_12_mw": 45.3, "bus_2_voltage_pu": 0.98},
    grid_area="national",
    estimator_type="WLS",
)
# {"converged": True, "residual": 0.000023, "voltage_violations": 0, ...}

# N-1 contingency analysis (requires a converged SE run)
contingency = await svc.contingency_analysis(n_minus_1=True)
# {"system_status": "alert", "violations": [...], "contingencies_analysed": 5}

# Frequency monitoring with alert and auto-dispatch
freq = await svc.frequency_monitoring(
    timestamp="2026-06-01T06:00:05Z",
    hz=49.3,
    source="PMU",
    rocof_hz_s=-0.7,
)
# {"alert": True, "under_frequency_alert": True, "high_rocof_alert": True}
```

## API Routes

| Path | Method | Description | Permission |
|---|---|---|---|
| `/energy-grd/api/v1/dashboard` | GET | Real-time grid dashboard | `energy_grd:view` |
| `/energy-grd/api/v1/state-estimation` | POST | Record SE run result | `energy_grd:state_estimation` |
| `/energy-grd/api/v1/contingency` | POST | Record contingency result | `energy_grd:contingency` |
| `/energy-grd/api/v1/contingency/<id>` | GET | Contingency detail | `energy_grd:contingency` |
| `/energy-grd/api/v1/voltage-control` | POST | Record voltage control action | `energy_grd:voltage_control` |
| `/energy-grd/api/v1/frequency-control` | POST | Record frequency control action | `energy_grd:frequency_control` |
| `/energy-grd/api/v1/frequency-control/ufls` | PUT | Configure UFLS threshold | `energy_grd:frequency_control` |
| `/energy-grd/api/v1/market-settlement` | POST | Settle market interval | `energy_grd:market_settlement` |
| `/energy-grd/api/v1/market-settlement/<id>/finalize` | PUT | Finalize settlement | `energy_grd:market_settlement` |
| `/energy-grd/api/v1/alarms` | POST | Raise alarm | `energy_grd:alarms` |
| `/energy-grd/api/v1/alarms/<id>/acknowledge` | PUT | Acknowledge alarm | `energy_grd:alarms` |
| `/energy-grd/api/v1/alarms/<id>/clear` | PUT | Clear alarm | `energy_grd:alarms` |
| `/energy-grd/api/v1/ems` | POST | Execute EMS function | `energy_grd:ems` |
| `/energy-grd/api/v1/agents` | POST | Register agent | `energy_grd:admin` |

## Business Rules

| Rule | Condition | Effect |
|---|---|---|
| `tenant_context_required` | tenant_context_present=False | deny |
| `se_network_model_required` | network_model_present=False | deny |
| `contingency_base_case_required` | base_case_converged=False | deny |
| `n1_contingency_mandatory` | skip_n1_contingency AND bypass not allowed | deny |
| `voltage_control_approval_required` | control without approval | deny |
| `frequency_ufls_threshold_valid` | threshold outside 47.0–49.5 Hz | deny |
| `market_metered_data_required` | metered_data_present=False | deny |
| `market_bid_offer_required` | bid_offer_present=False | deny |
| `critical_alarm_acknowledgement_required` | clear critical alarm without acknowledgement | deny |
| `emergency_control_requires_acknowledgement` | agent action during emergency, alarm not acked | deny |
| `cross_tenant_denied` | cross_tenant_access=True | deny |
| `privileged_grd_agent_requires_human_approval` | agent control action without human approval | deny |

## Data Models

| Model | Key Fields |
|---|---|
| `StateEstimationRun` | id, estimator_type, grid_area, converged, iterations, residual, voltage_violations |
| `ContingencyCase` | id, contingency_type, contingency_name, system_status, violations, max_overload_pct |
| `VoltageControlAction` | id, control_method, element_id, target_voltage_pu, achieved_voltage_pu, approved_by |
| `FrequencyControlAction` | id, control_method, trigger_frequency_hz, response_mw |
| `MarketSettlementInterval` | id, market_product, metered_mwh, scheduled_mwh, imbalance_mwh, settlement_amount, status |
| `GridAlarm` | id, alarm_category, severity, element_id, acknowledged, cleared_at, status |
| `EmsFunctionExecution` | id, ems_function, mode, status, result_summary, triggered_by |

## New Methods

### `state_estimation` — async WLS/LAV/EKF from raw SCADA snapshot

```python
result = await svc.state_estimation(
    timestamp="2026-06-01T06:00:00Z",
    sensor_readings={"bus_1_voltage_pu": 1.02, "line_12_mw": 45.3},
    grid_area="north_zone",
    estimator_type="WLS",  # WLS | LAV | WLAV | EKF
)
# Returns: converged, residual, voltage_violations, iterations, sensor_count
```

### `frequency_monitoring` — PMU-aware frequency recording with auto-dispatch

```python
rec = await svc.frequency_monitoring(
    timestamp="2026-06-01T12:00:00Z",
    hz=49.2,
    source="PMU",        # or "SCADA"
    area_id="south_area",
    rocof_hz_s=-0.8,     # rate of change of frequency; triggers high_rocof_alert if |rocof| > 0.5
)
# Under-frequency or high RoCoF auto-dispatches a FrequencyControlAction
# If OLLAMA_BASE_URL is set, appends ml_threat_class + ml_threat_confidence
```

### `islanding_detection` — multi-indicator passive detection

```python
event = await svc.islanding_detection(
    area_id="feeder_42",
    indicators={
        "voltage_delta_pu": 0.09,
        "frequency_delta_hz": 0.25,
        "rocof_hz_s": 0.6,
        "vector_shift_deg": 13.0,
    },
)
# islanding_detected=True if ≥2 indicators breach thresholds
# Auto-raises severity=emergency alarm when detected
```

### `ancillary_services_procurement` — reserve and regulation market clearing

```python
proc = await svc.ancillary_services_procurement(
    service_type="spinning_reserve",   # frequency_regulation | spinning_reserve | black_start | demand_response | ...
    period="2026-06",
    quantity=150.0,                    # MW required
    accepted_bids=[
        {"provider_id": "GEN_01", "mw": 80.0, "price": 12.5},
        {"provider_id": "GEN_03", "mw": 75.0, "price": 13.0},
    ],
    currency="KES",
)
# Returns: procured_mw, clearing_price, total_cost, procurement_status (awarded | under_procured)
```

### `grid_analytics` — period-level operational KPIs

```python
kpis = await svc.grid_analytics(period="2026-06")
# Returns: se_convergence_rate_pct, frequency_alerts, contingency_violations,
#          active_alarms, critical_alarms, voltage_control_actions, islanding_events
```

## Streaming Events

- `state_estimation_completed`
- `contingency_violation_detected` / `contingency_cleared`
- `voltage_control_action_taken`
- `frequency_control_action_taken` / `frequency_alert`
- `market_settlement_preliminary` / `market_settlement_final` / `market_settlement_completed`
- `grid_alarm_raised` / `grid_alarm_acknowledged` / `grid_alarm_cleared`
- `ems_function_executed`
- `islanding_detected`
- `black_start_plan_recorded`
- `ancillary_services_procured`
- `reactive_power_dispatched`

## World-Class Enhancements (v2.0)

Planned improvements aligned with TSO-grade operational standards (ENTSO-E, AEMO, PJM, NERC):

1. **Digital Twin with Real-Time Network Topology** — IEC 61970 CIM XML diff ingestion into sparse Ybus; `apply_topology_change()` with SE cache invalidation. Eliminates stale-topology SE errors after switching events.

2. **Bad Data Detection and Measurement Rejection** — Post-WLS largest normalised residual (LNR) test (`|r_N| > 3.0`); iterative bad-data suppression. IEEE 1159 compliant. Rejected measurements stored on SE run record.

3. **Optimal Power Flow (OPF) with Economic Dispatch** — DC-OPF linear program minimising total generation cost subject to Kirchhoff and thermal constraints. Returns locational marginal prices (LMPs) as shadow prices. `run_opf()` replaces rule-based redispatch. Industry standard for CAISO/ERCOT/NEM.

4. **Phasor Measurement Unit (PMU) Data Integration** — `ingest_pmu_frame()` at 30–120 fps with GPS timestamps; Prony analysis for oscillation mode detection; ring-buffer storage per PMU. Enables sub-cycle fault detection (NERC PRC-002).

5. **Automatic Generation Control (AGC) with Area Control Error** — `compute_ace()` and `agc_dispatch()` driving ACE to zero every 2–4 seconds. NERC BAL-001 compliance. CPS1/CPS2 flags returned per dispatch cycle.

6. **Demand Response Orchestration** — `dispatch_demand_response()` distributes MW reduction targets across enrolled participants by priority. `confirm_dr_response()` records performance against baseline. FERC Order 745 LMP compensation.

7. **Transmission Constraint Management (Security-Constrained Dispatch)** — `identify_binding_constraints()` and `compute_relief_actions()` using DC sensitivity shift factors. NERC TPL-001 compliant preventive dispatch.

8. **Wide-Area Monitoring, Protection, and Control (WAMPAC)** — `wampac_assessment()` computes voltage stability index (VSI) and inter-area angle divergence from PMU snapshots. Pre-emptive voltage support dispatch at VSI < 0.15. NERC PRC-026.

9. **Cybersecurity Anomaly Detection for SCADA Streams** — `scada_anomaly_check()` Z-score (`|z| > 4.0`) with rolling 24-hour baseline per sensor; correlated anomaly detection across physically related sensors. NERC CIP-007. Integrates with `intel` threat bus.

10. **Dynamic Line Rating (DLR)** — `compute_dynamic_rating()` via IEEE 738 heat balance using real-time weather (temp, wind, solar). Unlocks 10–40% stranded thermal capacity vs static rating. CIGRE TB 601.

11. **Energy Storage Integration and Dispatch Optimisation** — `dispatch_storage()` with SOC constraint validation; `optimize_storage_schedule()` rolling-horizon LP maximising arbitrage revenue minus degradation. Cycle count tracking for warranty. IEC 62933.

12. **Real-Time Congestion Revenue Management** — `compute_congestion_revenue()` allocates congestion rent to FTR holders proportional to hedged MW. FERC Order 681 FTR settlement. Feeds `energy_bil`.

13. **Adaptive Protection Coordination with Auto-Recloser Logic** — `compute_fault_levels()` from Zbus inverse; `recommend_protection_settings()` per relay zone; `trigger_autorecloser()` with 0.5s/5s/60s sequence. DG-aware fault-level recalculation. NERC PRC-025.

14. **Forecasting-Driven Preventive Dispatch (Day-Ahead Security)** — `day_ahead_security_assessment()` runs AC power flow over forecast horizon; `compute_preventive_dispatch()` solves SCUC LP for minimum-cost preventive rescheduling. NERC TPL-001 above 100 kV.

15. **Multi-Area Interchange Scheduling and Tie-Line Control** — `schedule_interchange()` with ATC check and e-tag reference; `monitor_tie_line()` feeds ACE; `resolve_interchange_dispute()` produces inter-TSO audit trail. NERC BAL-002 ±10 MW interchange compliance.

## Edge Cases Handled

- Contingency analysis blocked if base case has not converged
- N-1 analysis mandatory — bypass explicitly denied by rule engine
- Critical alarms must be acknowledged before clearing (safety sequencing)
- UFLS threshold validated to physically meaningful range (47–49.5 Hz)
- Settlement imbalance and amount calculated server-side from metered/scheduled inputs
- System status auto-classified (normal/alert/emergency) from contingency violations
- State estimation failures recorded with `failed_convergence` status for diagnostics
- Islanding auto-raises `severity=emergency` alarm and audits event
- Frequency RoCoF breach triggers inertial response dispatch independent of under-frequency state

## Composability Notes

- Receives dispatch schedules from `energy_gen` for unit commitment integration
- Coordinates with `energy_dis` for distribution-level topology during contingency
- Demand reduction signals from `energy_met` DR events reduce load in frequency control
- Market settlement data feeds `energy_bil` for wholesale billing
- Generation forecasts from `energy_ren` feed unit commitment in EMS scheduling
- Alarm stream feeds `intel` threat detection and `moni` operational dashboards
- Cybersecurity anomaly alarms (`alarm_category="cybersecurity"`) publish to `intel` threat bus
