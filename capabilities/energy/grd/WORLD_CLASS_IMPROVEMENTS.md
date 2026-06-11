# Grid Management (energy_grd) — World-Class Improvements

## 1. Digital Twin with Real-Time Network Topology
**Category**: Core Architecture
**Justification**: Modern TSOs (ENTSO-E, PJM, AEMO) maintain a live CIM-compliant digital twin that tracks every switching state change within seconds. The current service uses static `network_model_ref` strings with no topology awareness. Without a live twin, state estimation runs on stale topology, producing incorrect power flows after any switching event.
**Implementation**: Ingest IEC 61970 CIM XML diffs via `topology_update()` into a sparse adjacency matrix. Re-triangulate Ybus on each diff using sparse LU factorisation (scipy.sparse). Expose `async def apply_topology_change(switch_id, state, reason, approved_by)` that snapshots the prior model, applies the delta, and invalidates the SE cache.
**Competitor**: OSIsoft PI AF topology model; GE PSCAD real-time digital twin.

---

## 2. Bad Data Detection and Measurement Rejection
**Category**: State Estimation Quality
**Justification**: WLS state estimation silently absorbs SCADA telemetry errors, producing biased voltage and angle estimates that cascade into wrong contingency decisions. IEEE 1159-compliant grids require largest normalised residual (LNR) tests. Without it, a single stuck transmitter reading invalidates the entire SE solution.
**Implementation**: After each WLS solve, compute normalised residuals `r_N[i] = r[i] / sqrt(S[i,i])` where S is the covariance of measurement residuals. Flag measurements with `|r_N| > 3.0` as suspicious. Implement iterative bad-data suppression: remove flagged measurements, re-solve, repeat until clean. Store removed measurements in `se_run.rejected_measurements`.
**Competitor**: Siemens PSS/E bad-data processing; Schneider ENMAC measurement validation.

---

## 3. Optimal Power Flow (OPF) with Economic Dispatch
**Category**: Market Operations
**Justification**: Load balancing today is rule-based redispatch. DC-OPF minimises total generation cost subject to Kirchhoff constraints and line thermal limits, which is the industry standard (CAISO, ERCOT, NEM). Without OPF, dispatch decisions are suboptimal by 3–8% — material at scale. AC-OPF also handles reactive power pricing.
**Implementation**: Formulate DC-OPF as a linear program: `min c^T * p` s.t. `B * theta = p - d`, `|f_ij| <= f_max`, `p_min <= p <= p_max`. Solve via scipy.optimize.linprog or CVXPY. Return locational marginal prices (LMPs) as shadow prices of nodal balance constraints. `async def run_opf(period, generation_bids, load_forecast, network_model_ref)` returning optimal dispatch + LMPs.
**Competitor**: ABB Network Manager OPF; GE EMS economic dispatch engine.

---

## 4. Phasor Measurement Unit (PMU) Data Integration
**Category**: Real-Time Monitoring
**Justification**: Traditional SCADA updates at 2–4 second scan rates. PMUs report synchrophasor data at 30–120 frames/second with GPS-timestamped angles, enabling sub-cycle fault detection, oscillation monitoring, and dynamic state estimation. NERC PRC-002 mandates synchrophasor data retention. Without PMU integration the service cannot detect inter-area oscillations.
**Implementation**: Add `async def ingest_pmu_frame(pmu_id, timestamp_us, voltage_phasor, current_phasor, frequency_hz, rocof)` storing into a time-series ring buffer (circular deque, 10,000 frames per PMU). Compute Prony analysis for oscillation mode detection. Flag frames with GPS quality indicator `< 1` as bad. Feed PMU frequency into `frequency_monitoring()` with `source="PMU"`.
**Competitor**: SEL Real-Time Automation Controller; GE Grid Solutions PDC Manager.

---

## 5. Automatic Generation Control (AGC) with Area Control Error
**Category**: Frequency Regulation
**Justification**: ACE = (actual_interchange - scheduled_interchange) + 10B * (actual_freq - nominal_freq). AGC drives ACE toward zero by raising/lowering unit set-points every 2–4 seconds. This is a NERC BAL-001 compliance obligation. The current service has no ACE computation — frequency control is purely reactive, not regulating. Without AGC, the system accumulates frequency error causing CPS1/CPS2 violations.
**Implementation**: `async def compute_ace(actual_interchange_mw, scheduled_interchange_mw, actual_freq_hz, b_coefficient)` → ACE value + CPS1/CPS2 compliance flags. `async def agc_dispatch(ace, participating_units, droop_settings)` → raises/lowers each unit proportional to droop and ACE share. Store AGC signals in `_agc_signals` store. Audit every dispatch signal.
**Competitor**: ABB MicroSCADA Pro AGC; Siemens SPECTRUM Power AGC.

---

## 6. Demand Response Orchestration
**Category**: Load Management
**Justification**: Demand response is the fastest-response load balancing resource (sub-second for smart inverters, 10-minute for curtailable loads). FERC Order 745 requires demand response compensation at LMP. Without DR orchestration the grid cannot shed load intelligently during emergencies — it falls back to uncontrolled UFLS which causes over-shedding and quality-of-supply breaches.
**Implementation**: `async def dispatch_demand_response(event_id, target_reduction_mw, dr_type, participants, start_time, duration_min)` — distributes reduction targets across enrolled participants by priority. `async def confirm_dr_response(event_id, participant_id, actual_reduction_mw)` — records performance against baseline. Calculate performance score and feed to ancillary services settlement.
**Competitor**: Oracle Utilities DR Management; AutoGrid DROMS.

---

## 7. Transmission Constraint Management with Security-Constrained Dispatch
**Category**: Network Security
**Justification**: Real-time constraint management (RTCM) identifies binding transmission constraints and computes relief actions before they become contingency violations. NERC TPL-001 requires security-constrained economic dispatch. The current contingency analysis only detects violations post-facto — it does not compute preventive dispatch changes.
**Implementation**: `async def identify_binding_constraints(loading_threshold_pct)` → returns lines loaded above threshold with available transfer capability (ATC). `async def compute_relief_actions(binding_constraint_id, relief_type)` → runs DC-sensitivity-based redispatch, computing shift factors and relief MW per unit. Store in `_constraint_records`.
**Competitor**: GE PowerOn Fusion constraint management; Ventyx Monarch RTCM.

---

## 8. Wide-Area Monitoring, Protection, and Control (WAMPAC)
**Category**: Protection Coordination
**Justification**: Traditional zone-based protection misses inter-area instability modes that develop over tens of seconds. WAMPAC uses synchrophasor angles from multiple substations to detect voltage collapse trajectories, out-of-step conditions, and cascade-risk scenarios 30–60 seconds before relay action. NERC PRC-026 mandates load-responsive protection review.
**Implementation**: `async def wampac_assessment(pmu_snapshot)` — computes voltage stability index (VSI) at each load bus: `VSI = 4 * |V_s|^2 * |Z_L| / (|Z_s + Z_L|)^2`. VSI < 0.15 triggers pre-emptive voltage support dispatch. Detect angle divergence between areas with threshold 30°. Return stability margin, risk level, and recommended corrective actions.
**Competitor**: AREVA e-terra WAMS; EPRI CVSA tool.

---

## 9. Cybersecurity Anomaly Detection for SCADA Streams
**Category**: Security
**Justification**: ICS/SCADA systems are primary targets for grid attacks (Ukraine 2015/2016, Colonial Pipeline). False data injection (FDI) attacks manipulate sensor readings to produce wrong SE solutions without triggering traditional bad-data tests. NERC CIP-007 requires monitoring. Without behavioural anomaly detection, an attacker can steer dispatch toward unsafe operating points.
**Implementation**: `async def scada_anomaly_check(sensor_id, reading, expected_range, historical_mean, historical_std)` — computes Z-score; if `|z| > 4.0` flag as anomaly. Maintain rolling 24-hour baseline per sensor. Detect correlated anomalies across related sensors (e.g. both ends of a line move in the same direction simultaneously — impossible physically). Raise `alarm_category="cybersecurity"` on confirmed anomaly. Integrate with `intel` capability threat bus.
**Competitor**: Claroty OT anomaly detection; Dragos Platform for ICS.

---

## 10. Dynamic Line Rating (DLR)
**Category**: Asset Optimisation
**Justification**: Static thermal ratings (STR) are set at worst-case ambient (40°C, low wind). DLR uses actual weather data (temperature, wind speed, solar radiation) to compute real-time ampacity — typically 10–40% higher than STR in favourable conditions. This unlocks stranded transmission capacity. CIGRE TB 601 defines the standard. Without DLR, operators curtail renewable generation unnecessarily.
**Implementation**: `async def compute_dynamic_rating(line_id, conductor_temp_c, ambient_temp_c, wind_speed_ms, wind_angle_deg, solar_radiation_wm2)` using IEEE 738 heat balance: `I = sqrt((q_c + q_r - q_s) / R_ac)` where q_c = convective cooling, q_r = radiative cooling, q_s = solar heating. Update line rating in network model. Return ampacity, derating_pct vs static rating, and confidence interval.
**Competitor**: LineVision DLR sensors; Ampacimon real-time thermal rating.

---

## 11. Energy Storage Integration and Dispatch Optimisation
**Category**: Flexibility Resources
**Justification**: Battery energy storage systems (BESS) are now multi-GW resources in major grids. BESS provides synthetic inertia, fast frequency response (<200ms), and arbitrage. Without BESS dispatch optimisation the service treats storage as passive load, missing 15–25% reduction in operating reserve costs. IEC 62933 governs grid-scale storage integration.
**Implementation**: `async def dispatch_storage(asset_id, target_mw, duration_h, mode, soc_pct, soc_min_pct, soc_max_pct)` — validates SOC constraints, computes feasible dispatch range, records charge/discharge cycle. `async def optimize_storage_schedule(period, price_forecast, frequency_reserve_requirement, assets)` — rolling horizon LP to maximise arbitrage revenue minus degradation cost. Track cycle count for warranty monitoring.
**Competitor**: Tesla Autobidder; Fluence Mosaic BESS optimisation.

---

## 12. Real-Time Congestion Revenue Management
**Category**: Market Clearing
**Justification**: Transmission congestion produces LMP differentials between buses — the congestion rent. TSOs collect this as congestion revenue and redistribute via FTR/CRR auctions. Without congestion revenue accounting the market settlement is incomplete: participants holding FTRs are not compensated, and the operator cannot reconcile settlement. FERC Order 681 mandates FTR settlement.
**Implementation**: `async def compute_congestion_revenue(period, lmps, scheduled_flows, ftr_holdings)` — computes congestion component per interface: `CR = sum(f_ij * (LMP_j - LMP_i))`. Allocates revenue to FTR holders proportional to their hedged MW. Records congestion rent residue. Feeds `energy_bil` for settlement reconciliation.
**Competitor**: Ventyx Energy Settlements; ICE's Endur congestion management module.

---

## 13. Adaptive Protection Coordination with Auto-Recloser Logic
**Category**: Protection Systems
**Justification**: Fixed protection settings cause unnecessary permanent outages when a recloser could restore supply after a transient fault. Adaptive protection adjusts relay thresholds based on current network topology and fault level — critical when distributed generation (DG) changes fault current magnitude. NERC PRC-025 requires load-responsive settings. Without adaptive protection, DG interconnection causes protection blinding.
**Implementation**: `async def compute_fault_levels(bus_id, network_state_ref)` — builds Zbus from Ybus inverse, computes three-phase and single-line-to-ground fault MVA. `async def recommend_protection_settings(relay_id, fault_level_mva, ct_ratio, coordination_margin_s)` — derives pickup current, time-dial setting, and reaches for each relay zone. `async def trigger_autorecloser(line_id, fault_type, attempt_number)` — implements 0.5s/5s/60s reclose sequence, raises permanent fault alarm on third failure.
**Competitor**: SEL ARC adaptive protection; GE Multilin UR series relay coordination.

---

## 14. Forecasting-Driven Preventive Dispatch (Day-Ahead Security)
**Category**: Planning Operations
**Justification**: Real-time security monitoring reacts to present state. Day-ahead preventive security analysis uses load and generation forecasts to identify potential violations 4–24 hours ahead, allowing low-cost preventive redispatch rather than expensive emergency corrective actions. NERC TPL-001 mandates this for facilities above 100kV. Without it, operators face avoidable N-1 violations during peak demand transitions.
**Implementation**: `async def day_ahead_security_assessment(forecast_period, load_profile, generation_schedule, outage_schedule)` — runs AC power flow for each forecast hour using predicted network state. Identifies hours with N-1 violations. `async def compute_preventive_dispatch(violation_hour, violation_element, available_units)` — solves security-constrained unit commitment (SCUC) LP to find minimum-cost preventive rescheduling. Returns dispatch adjustments with cost delta.
**Competitor**: PowerWorld Simulator SCOPF; Plexos grid simulation.

---

## 15. Multi-Area Interchange Scheduling and Tie-Line Control
**Category**: Interconnection Operations
**Justification**: Interconnected grids schedule MW transfers across tie-lines via e-Tags (NERC TAG-001) in North America and ENTSO-E scheduling in Europe. Actual interchange must match scheduled interchange within ±10 MW (NERC BAL-002) or face financial penalties. Without interchange scheduling the AGC ACE calculation is wrong, and the service cannot participate in multi-area energy markets.
**Implementation**: `async def schedule_interchange(from_area, to_area, period, mw_schedule, e_tag_ref, approved_by)` — validates curtailment priority, checks ATC, records e-tag reference. `async def monitor_tie_line(tie_line_id, actual_mw, scheduled_mw)` — computes instantaneous interchange deviation, feeds ACE calculation. `async def resolve_interchange_dispute(tag_id, actual_mwh, scheduled_mwh, reason)` — raises settlement adjustment flag and audit trail for inter-TSO reconciliation.
**Competitor**: OATI webTag interchange scheduling; Eterra ICCP tie-line telemetry.
