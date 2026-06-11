# Mine Production Operations — World-Class Improvements

Fifteen targeted improvements to elevate `mining_pro` from functional to industry-leading, benchmarked against Maptek Evolution, Deswik.CAD, Micromine Pitram, and Wenco FMS.

---

### I1. Real-Time Truck Dispatch Optimisation
**Category**: Operational Intelligence
**Justification**: Static shift reporting captures production 8–12 hours late. Real-time truck assignment against live face status reduces haul cycle times by 8–15% (Wenco FMS benchmark), directly translating to cost per tonne improvement.
**Implementation**: Add `async def dispatch_truck(truck_id, destination, load_tonnes, priority)` backed by a priority-queue store (`_dispatch_queue`). NATS subject `mining.pro.dispatch.{mine_area}` streams assignments to onboard terminal units. Re-optimise on face completion events from `ore_movement`.
**Competitor**: Wenco FMS, Modular DISPATCH

---

### I2. Blast Vibration Compliance Monitoring
**Category**: Safety & Regulatory
**Justification**: Regulators in most jurisdictions require proof that peak particle velocity (PPV) stays below 5–25 mm/s at sensitive receivers. Manual log review misses violations. Automated envelope checking eliminates prosecution risk and prevents community relations damage.
**Implementation**: Add `async def record_blast_vibration(blast_id, sensor_id, ppv_mmps, distance_m, receiver_type)` with automatic breach detection against configurable limits stored in the `conf` capability. Emit `blast_vibration_breach` event on NATS when PPV exceeds threshold.
**Competitor**: Instantel BLASTWARE, ShotPlus-I

---

### I3. Block Model Grade Reconciliation
**Category**: Ore Value Chain
**Justification**: The gap between geological block model grade and mill feed grade (reconciliation factor) is the single most important metric for detecting ore loss and misclassification. Industry target is ±5%. Currently no reconciliation loop exists.
**Implementation**: Add `async def reconcile_block_model(block_id, block_model_grade, block_model_tonnes, period)` that computes F-factor (mine call factor), C-factor (concentration factor), and E-factor (extraction factor). Stores results in `_reconciliation_records` and publishes `reconciliation_variance_alert` on NATS when F-factor deviates >10%.
**Competitor**: Datamine Studio RM, Micromine Origin

---

### I4. Equipment Utilisation and Availability Dashboarding
**Category**: Asset Management
**Justification**: JORC/NI 43-101 compliant resource estimates require demonstrated plant availability >85%. Currently delay_hours are captured per shift but not aggregated per equipment ID. Wenco FMS reports show 12–18% productivity gains from real-time availability feedback loops.
**Implementation**: Add `async def equipment_availability_report(equipment_id, period)` that aggregates delay records by equipment, computes Physical Availability (PA), Mechanical Availability (MA), and Utilisation (U) using SMRP definitions. Feed from `_shifts` delay records cross-referenced with `mining_eqp`.
**Competitor**: Wenco FMS, Micromine Pitram, Modular Mining

---

### I5. NATS-Based Real-Time Production Event Streaming
**Category**: Integration / Streaming Architecture
**Justification**: Current state is in-memory; no downstream consumers can react sub-second to production events. Bytewax+NATS enables sub-100ms latency delivery of production events to process control, finance, and safety systems — matching Newmont's real-time production mesh architecture.
**Implementation**: Add `async def publish_production_event(event_type, payload, mine_area)` using `nats.aio` client with subjects `mining.pro.{event_type}.{mine_area}`. Integrate into `ore_movement`, `shift_report`, `blast_result`, and `grade_control_sample` methods as post-write side effects. Bytewax pipeline consumes for rolling KPI aggregation.
**Competitor**: Kafka-based OSIsoft PI, Aveva MES

---

### I6. Automated Mine Call Factor (MCF) Calculation
**Category**: Metallurgical Accounting
**Justification**: MCF is the ratio of metal reported by the mill to metal estimated from mining. Values below 0.85 trigger immediate geological review at tier-1 operations (Barrick, AngloGold). Currently no MCF tracking exists in `mining_pro`.
**Implementation**: Add `async def calculate_mine_call_factor(period, section)` pulling ore movements (mined metal) vs `mining_ore` mill feed receipts via the composition engine. Stores results, flags MCF < 0.90 as `mcf_alert` in NATS, integrates with `ntfy` for SMS/email to Mine Manager.
**Competitor**: Surpac, Micromine Origin, GEOVIA GEMS

---

### I7. Short-Interval Control (SIC) Feedback Loop
**Category**: Production Optimisation
**Justification**: SIC — comparing plan vs actual every 2–4 hours rather than end-of-shift — recovers 5–10% lost production by enabling real-time supervisor intervention. Tier-1 mines (Codelco, BHP) embed SIC in operational SOPs.
**Implementation**: Add `async def short_interval_report(section, interval_start, interval_end, actual_tonnes, actual_metres, supervisor_id)` with automatic variance calculation against the hourly disaggregation of the published weekly schedule. NATS publishes `sic.variance.critical` when cumulative gap exceeds 15%.
**Competitor**: Deswik.CAD SIC module, Pitram SIC

---

### I8. Geofenced Face Status Management
**Category**: Spatial Operations
**Justification**: Active face status (available, blasting, restricted, maintenance) must be spatially tracked to prevent equipment collision and unauthorised access to live firing areas. Pen-and-paper face boards are replaced in tier-1 mines with GIS-integrated face management.
**Implementation**: Add `async def update_face_status(face_id, status, polygon_wkt, reason, updated_by)` maintaining a spatial store `_face_status`. Status transitions emit NATS events consumed by dispatch (I1) to exclude restricted faces from truck assignments.
**Competitor**: Micromine Pitram face management, Wenco FMS geofence module

---

### I9. Explosives Consumption Reconciliation
**Category**: Compliance & Cost Control
**Justification**: Explosives magazines are regulated under national explosives acts. Variance between magazine issues and blast plans is a criminal compliance matter. Currently no reconciliation loop between `blast_plan.explosives_qty` and magazine records.
**Implementation**: Add `async def reconcile_explosives(period, magazine_id)` that compares total explosives issued from magazine records against sum of `blast_plan.total_explosive_kg` for the period. Returns variance per explosive type and flags any discrepancy >2 kg to compliance officer via `ntfy`.
**Competitor**: MAXAM CIDAN, Orica BlastIQ, Dyno Nobel SHOTPlus

---

### I10. Automated Delay Pareto and Root-Cause Classification
**Category**: Continuous Improvement
**Justification**: Pareto analysis of delay categories drives >60% of all production improvement initiatives at mature operations. Currently delays are stored but never ranked or classified for root-cause escalation. World-class operations use automated delay Pareto for weekly management reviews.
**Implementation**: Add `async def delay_pareto_analysis(period, section)` aggregating delay records by category, computing cumulative % share, and classifying top delays using a rule engine (equipment breakdown → `mining_eqp`, blasting hold → `mining_pro`, safety hold → `mining_saf`). Returns ranked Pareto dict with escalation recommendations.
**Competitor**: Pitram production analysis, Modular Mining DelayEntry

---

### I11. Integrated Production Forecast (Monte Carlo)
**Category**: Planning Intelligence
**Justification**: Deterministic schedules ignore grade variability and equipment availability uncertainty. Monte Carlo simulation (10k iterations) on grade distributions and equipment MTBF gives probabilistic production forecasts — enabling finance to set defensible quarterly guidance ranges rather than point estimates.
**Implementation**: Add `async def monte_carlo_production_forecast(section, horizon_months, n_simulations, grade_cv, availability_cv)` using NumPy-based simulation on historical grade distributions from grade control samples and historical availability from delay records. Returns P10/P50/P90 tonnage and grade envelopes.
**Competitor**: Maptek Evolution, Deswik.Sched probabilistic module

---

### I12. Multi-Level Schedule Lock and Freeze Protocol
**Category**: Change Management / Governance
**Justification**: Uncontrolled schedule changes after approval cause budget overruns at most operations. Tier-1 operators enforce a T-48h schedule lock (no changes within 48 hours of shift start) with a formal change control process for exceptions.
**Implementation**: Add `async def request_schedule_change(schedule_id, change_type, justification, requested_by)` and `async def approve_schedule_change(change_id, approver_id)` enforcing T-48h lock rule. Changes within the lock window require sign-off from both Mine Manager and Planning Manager (two-key approval). Stores change log in `_schedule_changes`.
**Competitor**: Deswik.CAD schedule management, XPAC scheduling system

---

### I13. Automated Shift Handover Package
**Category**: Operational Continuity
**Justification**: Poor shift handovers cause 20–30 min of lost production at each changeover (Nelson & Winter 2019 mining productivity study). Automated handover packages — pre-populated from shift data — reduce handover time to under 5 minutes and eliminate missed-information incidents.
**Implementation**: Add `async def generate_shift_handover(outgoing_shift_id, incoming_supervisor_id)` that assembles a structured handover document from the closing shift report, open blast holds, pending grade control decisions, stockpile levels, and active equipment faults from `mining_eqp`. Published to NATS for display on supervisor terminals.
**Competitor**: Pitram shift handover, SAP PM shift notes

---

### I14. Environmental Compliance: Dust and Water Discharge Tracking
**Category**: ESG / Regulatory
**Justification**: Environmental incidents at mine sites carry financial penalties (up to $50k/day in AU jurisdiction) and social licence risk. Water management (discharge volumes, quality) and dust suppression (water cart usage vs wind trigger) are increasingly mandated by mining licences.
**Implementation**: Add `async def record_dust_suppression_event(location, water_cart_id, volume_litres, wind_speed_kph, triggered_by)` and `async def record_water_discharge(point_id, volume_m3, quality_params, discharge_at)` with automatic flagging when discharge exceeds consent limits. Feeds `mining_env` capability.
**Competitor**: Intelex EHS, Cority environmental module, Hummingbird ESG

---

### I15. AI-Assisted Blast Design Optimisation
**Category**: Advanced Analytics / AI
**Justification**: Optimal powder factor, burden, and spacing calculation using historical fragmentation outcomes (coarse/medium/fine) and rock mass classification data reduces re-handling costs by 15–25% (Orica BlastIQ whitepaper 2023). Manual design relies solely on engineer experience.
**Implementation**: Add `async def suggest_blast_parameters(location_id, rock_mass_rating, desired_fragmentation, tonnage_target)` that queries historical `_blast_plans` and `_blast_results` for the same location, fits a regression model (scikit-learn LinearRegression or a simple lookup table fallback), and returns recommended burden_m, spacing_m, powder_factor, and expected fragmentation with confidence intervals. Uses locally-hosted Ollama model for natural-language design justification summary.
**Competitor**: Orica BlastIQ AI, MAXAM BlastGeo ML, Dyno Nobel SHOTPlus AI
