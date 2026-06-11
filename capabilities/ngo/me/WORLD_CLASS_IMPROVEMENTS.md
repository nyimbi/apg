# M&E (ngo_me) — World-Class Improvements

Fifteen targeted improvements that close the gap between the current baseline and best-in-class M&E platforms (DevResults, Apricot, DHIS2, Vera Solutions Amp Impact).

---

### I1. Theory of Change Linkage
**Category**: Feature
**Justification**: Donors (USAID, FCDO, Gates Foundation) mandate that every indicator traces back to a published Theory of Change node. Without it, indicator results can't be attributed to programme logic — the single biggest reason evaluations are contested. DevResults and Apricot both enforce this chain at indicator creation.
**Implementation**: Extend indicators with `toc_node_id` and `causal_pathway` fields; add `link_indicator_to_toc()` and `validate_indicator_chain()` methods that traverse parent→child ToC relationships and return a completeness score.
**Competitive reference**: DevResults — mandatory ToC linkage on all output/outcome indicators.

---

### I2. SMART Indicator Validator
**Category**: AI/ML
**Justification**: Field teams routinely submit poorly worded indicators (not specific, no unit, ambiguous baseline). SMART validation at entry time cuts the rework cycle from weeks to seconds and improves inter-rater reliability across programme staff.
**Implementation**: Add `validate_indicator_smart()` that scores Specific/Measurable/Achievable/Relevant/Time-bound dimensions via rule-based heuristics (unit present, target numeric, date parseable, name ≥ 5 tokens) returning a scored dict and a list of remediation hints.
**Competitive reference**: Vera Solutions Amp Impact — built-in SMART checklist on indicator setup.

---

### I3. Disaggregation-Aware Aggregation Engine
**Category**: Feature
**Justification**: Funders require sex-, age-, and geography-disaggregated reporting. The current `collect_data()` stores raw disaggregation blobs but never aggregates them, so programme officers must do it manually in spreadsheets — reintroducing the errors M&E systems exist to prevent.
**Implementation**: Add `aggregate_disaggregated_data()` that groups data-collection records by a named dimension (e.g., `sex`, `age_band`, `location`), sums or averages values per bucket, and returns a pivot-style dict with totals and sub-totals.
**Competitive reference**: DHIS2 — category option combinations as a first-class aggregation primitive.

---

### I4. Milestone Tracking and Deadline Alerts
**Category**: Feature
**Justification**: Quarterly targets are the earliest warning signal for underperformance. Without milestone checkpoints between baseline and final target, deviation is only visible at report time — too late to course-correct within the grant period.
**Implementation**: Add `create_milestone()`, `list_milestones()`, and `check_overdue_milestones()` methods; each milestone stores expected value, due date, and actual value once reported; `check_overdue_milestones()` returns all milestones past due with no actual value recorded.
**Competitive reference**: Apricot — sub-target milestone scheduling per indicator.

---

### I5. Donor-Ready Report Export (OECD DAC Markers)
**Category**: Compliance
**Justification**: USAID, EU, and UN agencies require submissions tagged with OECD DAC policy markers (gender equality, environment, governance). Manual tagging is error-prone and adds 2–4 hours per report; automated tagging cuts compliance overhead by 80%.
**Implementation**: Add `tag_report_dac_markers()` that attaches DAC code, significance level (principal/significant/not-targeted), and justification text to a progress report; add `export_report_oecd()` that serialises the report plus markers into a structured dict matching the OECD CRS++ schema.
**Competitive reference**: IATI Standard / UN-OCHA Financial Tracking Service.

---

### I6. Cost-Efficiency Analysis (Cost per Beneficiary)
**Category**: Feature
**Justification**: The shift from "did we deliver outputs?" to "at what cost?" is the dominant trend in impact investing and donor accountability (2024–2026). No OSS NGO M&E platform natively computes cost-per-output or cost-per-outcome; this becomes a differentiated selling point.
**Implementation**: Add `record_expenditure()` (amount: Decimal, period, category) and `compute_cost_efficiency()` that divides cumulative expenditure by achievement on selected output indicators, returning cost-per-unit and a breakdown by cost category. All monetary values use Decimal throughout.
**Competitive reference**: GlobalGiving analytics dashboard; Social Value International SROI methodology.

---

### I7. Beneficiary Registry Integration
**Category**: Integration
**Justification**: Deduplication of beneficiary counts is the most common data-quality failure flagged in NGO audits. Integrating with the APG beneficiary registry (`ngo_beneficiaries`) enables unique-count validation at data collection time.
**Implementation**: Add `link_beneficiary_registry()` and `get_unique_beneficiary_count()` that accept a programme_id and period, call the beneficiary registry adapter, and cross-check the collected "number of beneficiaries" indicator against the registry's deduplicated count, returning a reconciliation report.
**Competitive reference**: Salesforce.org NPSP + Apricot beneficiary deduplication.

---

### I8. Adaptive Management Triggers
**Category**: AI/ML
**Justification**: The adaptive management paradigm (USAID CLA, DFID MARO) requires that M&E systems surface decision points proactively, not reactively. When an indicator drops below a threshold, the system should automatically flag a review event.
**Implementation**: Add `set_adaptive_trigger()` (threshold_pct, action_description) per indicator and `evaluate_adaptive_triggers()` that scans all active indicators, compares achievement_pct against thresholds, and returns a list of triggered alerts with recommended actions and evidence links.
**Competitive reference**: USAID CLA Toolkit — learning trigger workflows.

---

### I9. Survey / Instrument Builder
**Category**: Feature
**Justification**: Paper-based and ODK-based data collection tools don't feed M&E systems in real time. A native instrument builder reduces the round-trip from field collection to dashboard from days to minutes and eliminates manual transcription errors.
**Implementation**: Add `create_survey_instrument()` (linked to indicator_ids, question schemas, field_staff_ids) and `ingest_survey_responses()` that maps survey fields to indicator data-collection records via a configurable field→indicator mapping, creating verified data collections automatically.
**Competitive reference**: KoboToolbox + DHIS2 data-entry forms.

---

### I10. Inter-Rater Reliability Scoring
**Category**: Feature
**Justification**: When multiple field officers collect the same indicator independently (gold-standard for quality assurance), the system must detect agreement and flag outliers — a requirement in USAID PEPFAR reporting and GAVI immunisation programmes.
**Implementation**: Add `compute_inter_rater_reliability()` that groups unverified data-collection records by (indicator_id, period), computes Cohen's kappa for categorical and CV% for continuous indicators, and returns reliability scores with records flagged as outliers (>2σ).
**Competitive reference**: PEPFAR DATIM — site-level data triangulation reports.

---

### I11. Log Frame / Results Framework Builder
**Category**: Feature
**Justification**: Every FCDO and EU programme contract requires a logical framework (logframe). Currently programme officers build these in Excel then re-enter data into M&E tools — double-handling. A native logframe builder eliminates this gap.
**Implementation**: Add `create_logframe()` with hierarchical goal→purpose→output→activity structure; `add_logframe_row()` attaches indicator_ids, means of verification, and assumptions per row; `export_logframe()` serialises to a flat list suitable for spreadsheet/PDF generation.
**Competitive reference**: Devex / MDF logframe builder; FCDO Smart Rules.

---

### I12. Real-Time GIS Location Tagging
**Category**: Feature
**Justification**: Geographic coverage reporting (ward, sub-county, GPS coordinates) is mandatory in health, WASH, and food-security programmes. Location-less data cannot be aggregated to administrative boundaries for donor dashboards or government reporting.
**Implementation**: Add `tag_data_collection_location()` (collection_id, latitude: Decimal, longitude: Decimal, admin_level_1, admin_level_2, admin_level_3) and `get_geographic_coverage()` that returns unique admin units reached per indicator, with a bbox-based spatial summary.
**Competitive reference**: DHIS2 GIS module; Ushahidi geospatial layers.

---

### I13. Automated Variance Explanation Engine
**Category**: AI/ML
**Justification**: Funders expect narrative explanations when actuals deviate >20% from targets. Generating these manually consumes 30–40% of programme officer time. Rule-based variance explanation is the first step toward LLM-assisted narrative generation.
**Implementation**: Add `generate_variance_explanation()` that compares current_value against target_value at a given date, categorises deviation as on_track/minor_variance/major_variance/critical, and produces a structured explanation dict including deviation_pct, contributing_factors (derived from linked challenges in progress reports), and a suggested_narrative template string.
**Competitive reference**: Vera Solutions Amp — automated traffic-light narrative generation.

---

### I14. Programme Portfolio Benchmarking
**Category**: Analytics
**Justification**: Multi-programme NGOs (IRC, Save the Children, Oxfam) need to compare indicator achievement rates across programmes to allocate resources and identify best practices. Cross-programme benchmarking is absent from all OSS M&E tools.
**Implementation**: Add `portfolio_benchmark()` that accepts a list of programme_ids and returns a ranked comparison table of avg_achievement_pct, on_track count, off_track count, and evaluation ratings per programme, plus a percentile rank for each.
**Competitive reference**: Devex Impact Analytics; GlobalGiving portfolio reporting.

---

### I15. Evidence Repository with Version Control
**Category**: Feature
**Justification**: Findings from evaluations and learning cycles lose value when stored as unstructured text. A versioned, tagged evidence repository enables institutional memory, supports systematic reviews, and satisfies FCDO's "leaving behind evidence" requirement for long-term programmes.
**Implementation**: Add `store_evidence()` (source_type: evaluation|learning_cycle|survey, content, tags: list[str], programme_ids: list[str]) with a semver-style `version` counter; `search_evidence()` filters by tag intersection and programme_id; `get_evidence_history()` returns all versions of an evidence record.
**Competitive reference**: 3ie Evidence Gap Map; USAID Development Experience Clearinghouse (DEC).
