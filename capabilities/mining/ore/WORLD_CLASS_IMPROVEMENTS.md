# Ore Processing (mining_ore) — World-Class Improvement Catalogue

**Capability**: `mining_ore` | **Domain**: Mining Metallurgy | **Date**: 2026-06-11

---

## 1. Real-time Grind Circuit Optimisation (SAG/Ball Mill)

**Problem**: Grind P80 is set manually based on periodic lab assays, not real-time feedback.

**Improvement**: Implement `grind_optimisation_cycle()` that ingests PSA (particle size analyser) readings every 5 minutes, computes deviation from target P80, and emits PID setpoint adjustments for mill speed and water addition. Target P80 is commodity- and ore-type-specific.

**Value**: 1–3% recovery improvement from tighter grind control; reduces over-grinding energy cost (~15 kWh/t saving potential).

---

## 2. Automated Metallurgical Balance Closure Verification

**Problem**: `balance_closure_pct` is hardcoded to 100.0 — the mass balance is never actually closed.

**Improvement**: Implement `close_metallurgical_balance()` that iterates feed/concentrate/tailings stream assays, applies the two-product formula, and computes distribution coefficients per element. Rejects balances where closure error exceeds configurable threshold (default ±3%).

**Value**: Eliminates silent mass balance errors that distort recovery reporting; critical for royalty/off-take compliance.

---

## 3. Online Assay Integration via OPC-UA / MQTT

**Problem**: Assay data is entered manually — latency of hours to days.

**Improvement**: `ingest_online_assay()` subscribes to an OPC-UA or MQTT broker, maps tag addresses to sample points, and writes assay records in real time. Implements dead-band filtering to suppress noise.

**Value**: Shift reporting cycle from daily to hourly; early deviation detection reduces metallurgical losses.

---

## 4. Multi-Element Grade Control Blending Optimiser

**Problem**: `source_blend` in plant feeds is recorded after the fact; there is no forward-looking blend optimiser.

**Improvement**: `optimise_feed_blend()` takes stockpile inventory (tonnages + assay matrices per element) and downstream product specifications, then solves a linear programme (scipy.optimize or PuLP) to minimise penalty elements while meeting target head grade and throughput.

**Value**: Maximises recovered metal value; reduces smelter penalties from deleterious elements (As, Bi, Sb).

---

## 5. Predictive Reagent Dosage via Statistical Process Control

**Problem**: Reagent dosages are fixed setpoints; no SPC feedback loop.

**Improvement**: `spc_reagent_control()` computes Shewhart control charts (X-bar, R) on dosage rate versus recovery. Flags Western Electric rule violations and auto-generates dosage change recommendations. Tracks CUSUM for sustained drift.

**Value**: 5–10% reagent cost saving; faster response to ore type transitions.

---

## 6. Concentrate Filter Cake Quality Tracking

**Problem**: Product quality is recorded at lot dispatch only; no in-circuit quality trending.

**Improvement**: `record_filter_cake_quality()` captures moisture, throughput, and cake thickness measurements per filter press cycle. Tracks moisture exceedances against dewatering KPIs, triggers corrective action alerts.

**Value**: Prevents moisture penalties at the smelter (typically USD 0.5–2/dmt per 1% excess moisture).

---

## 7. Tailings Thickener Performance Management

**Problem**: Tailings density and overflow clarity are not tracked.

**Improvement**: `record_thickener_performance()` logs underflow density (% solids), overflow turbidity (NTU), and flocculant dosage per thickener. Computes unit area loading (t/m²·d) and flags underperformance vs design.

**Value**: Directly linked to water recovery rate; 1% improvement in water recovery at a 5 Mtpa plant saves ~50,000 m³/year.

---

## 8. Carbon-in-Leach (CIL) Loading Profile Management

**Problem**: No tracking of carbon loading profiles across CIL tanks.

**Improvement**: `record_cil_loading()` captures loaded carbon grade (g Au/t C) and carbon inventory per tank. Models carbon activity decay curve over leach time. Triggers advance/transfer recommendations when loading gradient is sub-optimal.

**Value**: 0.5–1% recovery uplift from optimised carbon management; reduces gold lock-up in circuit.

---

## 9. Elution and Electrowinning Efficiency Tracking

**Problem**: Elution strip efficiency and electrowinning cathode performance are not modelled.

**Improvement**: `record_elution_strip()` captures strip solution grade, elution efficiency (%), acid wash frequency, and cathode harvest weight. Computes stripping rate constant and flags declining performance.

**Value**: Early detection of carbon fouling or cathode scaling — prevents >1% recovery loss per event.

---

## 10. Ore Hardness and Bond Work Index (BWI) Tracking

**Problem**: Mill throughput varies with ore hardness, but there is no hardness index record.

**Improvement**: `record_ore_hardness()` stores Bond Work Index (kWh/t), Abrasion Index (Ai), and JK Drop Weight test parameters per ore type and source block. Links to feed blend to predict expected throughput vs nameplate capacity.

**Value**: Enables throughput forecasting accuracy within ±5%; improves maintenance scheduling for liner replacement.

---

## 11. Water Balance and Recycled Water Quality Tracking

**Problem**: Water is not tracked as a process input/output; environmental compliance is blind.

**Improvement**: `record_water_balance()` logs fresh water intake, process water recycled (m³), tailings dam return flow, and TSS/pH/conductivity of recycled water. Computes water intensity (m³/t) and flags quality exceedances against permit limits.

**Value**: Regulatory compliance (Environmental Permit); water cost savings; preparation for carbon water-intensity reporting.

---

## 12. Automated Shift Metallurgical Report Generation

**Problem**: Metallurgical reports are manually compiled (monthly only); no shift-level visibility.

**Improvement**: `generate_shift_met_report()` aggregates feed, recovery, reagent, and deviation data per 8-hour shift. Auto-distributes to shift supervisor via notification channel (ntfy). Flags shifts where recovery is below 2σ historical mean.

**Value**: Reduces metallurgist response time from 24h to <1h; drives immediate corrective action.

---

## 13. Ore Type Classification and Geometallurgical Mapping

**Problem**: All ore is treated identically; no geometallurgical domain classification.

**Improvement**: `classify_ore_type()` accepts XRF data (key element ratios) and assigns ore type from a configurable classification matrix (e.g. oxide/transition/primary/refractory). Links ore type to expected recovery, reagent suite, and grind target. Tags plant feed records with geometallurgical domain.

**Value**: 2–5% recovery improvement from ore-type-specific processing parameters; foundation for geometet forecasting.

---

## 14. Locked Cycle Flotation Test Results Repository

**Problem**: Locked cycle test results are stored in spreadsheets outside the system.

**Improvement**: `record_locked_cycle_test()` stores flotation test inputs (grind, pH, collector dosage, air flow, temperature), per-stage mass and grade pulls, and final concentrate/tailing assays. Links to ore type classification. Provides a queryable process mineralogy database.

**Value**: Enables reagent scheme optimisation based on empirical data rather than intuition; reduces plant trials needed.

---

## 15. Real-time Revenue and NSR (Net Smelter Return) Tracking

**Problem**: Revenue is not computed in real time; operators have no economic context for their process decisions.

**Improvement**: `compute_nsr()` takes current spot prices (fetched from configured price feed), concentrate grade, transport/treatment/refining charges (TC/RC), and smelter penalty schedule, and computes NSR (USD/t) in real time. Displays on circuit dashboard. Triggers economic optimisation flags when NSR drops below breakeven.

**Value**: Directly aligns metallurgical decisions with economic outcomes; typical NSR improvement from grade-recovery optimisation is 3–8 USD/t concentrate.

---

*© 2025 Datacraft | Author: Nyimbi Odero | www.datacraft.co.ke*
