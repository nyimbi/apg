# World-Class Improvements: Exploration Data Management (mining_exp)

## Summary

This document enumerates 15 high-impact improvements that would elevate `mining_exp` from a functional capability to a world-class exploration data management system. Improvements are ordered by estimated ROI and implementation complexity.

---

## 1. 3D Downhole Survey Desurveying Engine

**Current gap**: Drillholes store azimuth and dip at the collar only. Real holes deviate with depth. Without a desurveying engine, all downhole interval coordinates are wrong.

**Improvement**: Implement minimum-curvature desurveying — the industry standard (SPWLA) — that computes XYZ coordinates for every survey station and propagates them to all assay/geology intervals. Expose `desurvey_hole(hole_id)` returning a `list[SurveyStation]` with easting/northing/elevation at each depth.

**Impact**: Enables true 3D spatial queries, mine planning input, and correct resource block model coordinates.

---

## 2. Variogram Modelling and Geostatistical Input Preparation

**Current gap**: Resource estimation uses black-box method strings. No capability validates that the underlying geostatistics are sound.

**Improvement**: Add `compute_variogram(hole_ids, element, lag_distance, max_lag)` using experimental variogram calculation (method-of-moments). Return sill, range, nugget, and model type (spherical/exponential/Gaussian). Flag estimates where the variogram range is less than the drill spacing — a common misuse.

**Impact**: Directly addresses JORC Table 1 Section 5 (estimation and modelling techniques) requirements. Prevents overconfident resource classifications.

---

## 3. Composite Interval Calculator (Bench / Block Compositing)

**Current gap**: Assay results are stored as raw sample intervals. Resource estimation requires composited intervals aligned to bench height or regularised length.

**Improvement**: Add `composite_assay_intervals(hole_id, element, composite_length_m, method)` supporting bench (fixed-length), best-fit, and geological boundary compositing. Output preserves weighted-average grades and flags split intervals at lithology boundaries.

**Impact**: Standard pre-processing step before any resource estimate; eliminates manual spreadsheet compositing that is a major source of error.

---

## 4. QAQC Performance Dashboard with Statistical Control Limits

**Current gap**: `flag_qaqc_result` applies a flag but does not run statistical tests. There is no systematic way to detect lab bias or precision drift.

**Improvement**: Add `compute_qaqc_performance(batch_id)` that computes:
- Blank contamination rate (fails if any blank > 5× detection limit)
- CRM accuracy: z-score vs. certified value, ±2σ control limits
- Duplicate precision: half-absolute-relative-difference (HARD) plot, Thompson-Howarth precision
- Check assay bias via Pitard's protocol

Return a structured pass/fail report with control charts data.

**Impact**: Meets JORC Table 1 Section 11 (sample quality) requirements. Early detection of lab issues before data is used in resource estimates.

---

## 5. Bulk Density Assignment and Tonnes Calculation Audit

**Current gap**: Resource estimates store tonnage as a user-supplied float with no audit trail back to bulk density measurements.

**Improvement**: Add `record_bulk_density_measurement(hole_id, from_m, to_m, method, value_t_m3)` and `compute_resource_tonnes(estimate_id)` that validates contained tonnes against block volumes × measured bulk density distributions. Flag estimates where bulk density data coverage is < 1 measurement per 1000t.

**Impact**: Bulk density is one of the most commonly contested parameters in resource estimate audits. Traceability from BD measurements to final tonnes is a JORC requirement.

---

## 6. Automated Domaining by Geology Code

**Current gap**: Resource estimates treat the entire deposit as a single population. In practice, different lithological domains have different grade distributions.

**Improvement**: Add `define_resource_domain(name, lithology_codes, commodity, cut_off_grade)` and `assign_intervals_to_domains(hole_ids)`. Compute basic population statistics per domain (mean, CoV, log-normal fit). Flag intervals that switch domains mid-interval.

**Impact**: Domain-based estimation is the industry expectation for JORC Measured/Indicated resources. Eliminates grade smearing across geological boundaries.

---

## 7. Drill Programme Optimisation (Target to Discovery Ratio)

**Current gap**: No capability exists to evaluate whether the drilling programme is efficiently targeting anomalies.

**Improvement**: Add `drill_programme_efficiency(licence_id)` computing:
- Target-to-hole ratio (targets identified / holes drilled)
- Discovery rate (holes with significant intercepts / holes drilled)
- Cost-per-metre-of-significant-intercept
- Directional statistics on hole azimuth vs. target strike

**Impact**: Allows exploration managers to objectively assess programme effectiveness and redirect budget to highest-value targets.

---

## 8. Geochemical Multi-Element Anomaly Scoring

**Current gap**: Assay results are stored per-element with no cross-element analysis. Pathfinder element associations are critical in early-stage exploration.

**Improvement**: Add `score_geochemical_anomaly(hole_ids, elements, method)` supporting:
- Threshold methods (95th percentile, median ± 2MAD)
- Principal component analysis on element suite
- Iogas-style RPCA background separation
Return per-sample anomaly scores and top anomalous intervals ranked by composite score.

**Impact**: Directly improves the signal-to-noise ratio in early-stage target ranking. Reduces cost of follow-up drilling.

---

## 9. Reconciliation: Exploration vs. Mining Production

**Current gap**: Exploration estimates are stored but never compared to actual production figures from `mining_pro`.

**Improvement**: Add `reconcile_resource_estimate(estimate_id, production_period)` that accepts tonnes milled and head grade from the mining production capability and computes:
- Global reconciliation factor F1 (block model vs. mill)
- Geometric mean error per classification category
- Slope of regression line (grade prediction accuracy)

**Impact**: Grade reconciliation is the single most important feedback loop in resource geology. Poor reconciliation invalidates future estimates if unaddressed.

---

## 10. Competent Person Credential Registry and Expiry Tracking

**Current gap**: `competent_person_id` is stored as a free-text string with no validation against a credential registry.

**Improvement**: Add `register_competent_person(cp_id, full_name, professional_body, membership_number, commodity_specialisations, credential_expiry)` and enforce that any resource estimate or compliance report references a non-expired CP. Alert 90/30/7 days before expiry via `ntfy`.

**Impact**: JORC, NI 43-101, and SAMREC all require that the CP holds current membership of a recognised professional body. Automatic expiry tracking prevents inadvertent non-compliance.

---

## 11. Continuous Sampling Interval Gap Detection

**Current gap**: Geology and assay intervals are logged independently. Unsampled zones (gaps between intervals) are not flagged.

**Improvement**: Add `detect_sampling_gaps(hole_id, expected_to_depth_m)` that walks all logged intervals sorted by `from_m` and identifies:
- Gaps > 0.1 m between consecutive intervals
- Total unsampled depth as a percentage of total hole depth
- Deepest assayed interval vs. actual hole depth

**Impact**: Unsampled intervals must be disclosed under JORC Table 1 Section 9. Automated gap detection catches missing logs before submission.

---

## 12. Spatial Constraint Polygon Enforcement (Licence Boundary Check)

**Current gap**: Drillholes are registered with easting/northing coordinates but are never checked against the licence polygon boundary.

**Improvement**: Add `check_collar_within_licence(hole_id, licence_id)` using point-in-polygon (ray casting algorithm). Extend to `validate_programme_against_licence(licence_id)` that batch-checks all holes and returns those outside the permitted area.

**Impact**: Drilling outside the licence boundary is a regulatory breach. Automated spatial validation is a hard safety requirement before submission of any exploration report.

---

## 13. Drill Core Photography and Sample Tray Linkage

**Current gap**: Core logging has no linkage to physical sample photographs, which are mandatory under most modern reporting standards.

**Improvement**: Add `link_core_tray_photo(core_log_id, photo_uri, tray_number, from_depth_m, to_depth_m)` and `list_core_photos_for_hole(hole_id)`. Support bulk-import from a URI manifest. Enforce that resource estimates referencing a hole must have ≥ 80% photo coverage.

**Impact**: Core photography is an audit requirement. Digitising tray-photo linkage reduces physical archive dependency and accelerates remote CP review.

---

## 14. Automated NI 43-101 Technical Report Section Generator

**Current gap**: `create_compliance_report` creates a generic record with no structured content. A full NI 43-101 report requires 25 prescribed sections.

**Improvement**: Add `generate_ni43101_report_outline(licence_id, estimate_ids)` that auto-populates all 25 NI 43-101 sections from stored data:
- Item 4: Property description from licence registry
- Item 11: Sample preparation from assay method records
- Item 14: Data verification from QAQC performance report
- Item 18–19: Resource and reserve estimates with classification tables

Output a structured dict that maps directly to the prescribed section headings.

**Impact**: Reduces the time to produce a compliant technical report from weeks to hours. Eliminates transcription errors in tonnage and grade tables.

---

## 15. Exploration Budget Tracking and Cost-per-Ounce Equivalent Reporting

**Current gap**: No financial context is attached to exploration activities. Management cannot assess exploration efficiency in cost terms.

**Improvement**: Add `record_exploration_expenditure(licence_id, period, category, amount_usd, currency, notes)` and `compute_cost_per_resource_unit(licence_id, period, commodity)` that divides total expenditure by new resource ounces (or tonnes) discovered in the period. Support categories: drilling, geophysics, assaying, overhead, permitting.

**Impact**: Cost-per-ounce-equivalent is the primary KPI used by exploration investors to compare programmes globally. Without it, the capability has no financial intelligence layer.

---

## Implementation Priority Matrix

| # | Improvement | Effort | Compliance Value | Operational Value | Priority |
|---|---|---|---|---|---|
| 1 | 3D Desurveying | High | High | Critical | P0 |
| 3 | Composite Calculator | Medium | High | Critical | P0 |
| 4 | QAQC Statistics | Medium | High | High | P0 |
| 11 | Gap Detection | Low | High | High | P0 |
| 12 | Spatial Constraint | Medium | Critical | Critical | P0 |
| 10 | CP Registry | Low | Critical | Medium | P1 |
| 5 | Bulk Density | Medium | High | High | P1 |
| 6 | Domaining | High | High | High | P1 |
| 14 | NI 43-101 Generator | High | Critical | High | P1 |
| 2 | Variogram Modelling | Very High | Medium | High | P2 |
| 7 | Programme Efficiency | Low | Low | High | P2 |
| 8 | Geochemical Scoring | High | Medium | High | P2 |
| 9 | Reconciliation | Medium | Medium | Critical | P2 |
| 13 | Core Photography | Medium | Medium | Medium | P3 |
| 15 | Budget Tracking | Medium | Low | High | P3 |
