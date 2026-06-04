# Grid Operations

## Overview
Grid Operations provides the real-time operational intelligence layer for power system management. It covers state estimation with convergence tracking, N-1/N-2 contingency analysis with automatic system status classification, voltage control via multiple methods (tap changers, SVCs, STATCOMs), frequency control including AGC and UFLS, market interval settlement with imbalance calculation, a full grid alarm management system with severity-gated acknowledgement, and EMS function execution in real-time and study modes.

## Capability ID
`energy_grd`

## Provides
| Service | Description |
|---|---|
| `real_time_state_estimation` | Run and record state estimation with convergence status |
| `contingency_analysis` | N-1/N-2 contingency analysis with violation detection and system status |
| `voltage_control` | Record and approve voltage control actions across multiple methods |
| `frequency_control` | Record frequency response actions and configure UFLS thresholds |
| `market_settlement` | Settle market dispatch intervals with imbalance and price calculation |
| `grid_alarm_management` | Raise, acknowledge and clear alarms with severity-gated workflow |
| `ems_function_management` | Execute and record EMS functions in real-time or study mode |
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
| `frequency_ufls_threshold_valid` | threshold outside 47.0-49.5 Hz | deny |
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

## Streaming Events
- `state_estimation_completed`
- `contingency_violation_detected` / `contingency_cleared`
- `voltage_control_action_taken`
- `frequency_control_action_taken`
- `market_settlement_preliminary` / `market_settlement_final`
- `grid_alarm_raised` / `grid_alarm_acknowledged` / `grid_alarm_cleared`
- `ems_function_executed`

## Edge Cases Handled
- Contingency analysis blocked if base case has not converged
- N-1 analysis mandatory — bypass explicitly denied by rule engine
- Critical alarms must be acknowledged before clearing (safety sequencing)
- UFLS threshold validated to physically meaningful range (47-49.5 Hz)
- Settlement imbalance and amount calculated server-side from metered/scheduled inputs
- System status auto-classified (normal/alert/emergency) from contingency violations
- State estimation failures recorded with failed_convergence status for diagnostics

## Composability Notes
- Receives dispatch schedules from `energy_gen` for unit commitment integration
- Coordinates with `energy_dis` for distribution-level topology during contingency
- Demand reduction signals from `energy_met` DR events reduce load in frequency control
- Market settlement data feeds `energy_bil` for wholesale billing
- Generation forecasts from `energy_ren` feed unit commitment in EMS scheduling
- Alarm stream feeds `intel` threat detection and `moni` operational dashboards
