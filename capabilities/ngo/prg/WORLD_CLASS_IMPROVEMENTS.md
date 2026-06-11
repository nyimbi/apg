# World-Class Improvements — Programme & Project Monitoring (ngo_prg)

Fifteen improvements that bring ngo_prg to enterprise M&E platform standard.

---

### I1. Theory of Change (ToC) Pathway Management
**Category**: Feature
**Justification**: Every evidence-based funder (USAID, FCDO, GIZ) requires explicit ToC chains — Impact → Outcome → Output → Activity causal maps. Without this, programmes cannot demonstrate attribution, blocking funding renewals.
**Implementation**: Add a `create_theory_of_change()` method that stores nodes and directed causal edges; `trace_impact_pathway()` walks edges to produce a human-readable impact narrative and validates that all logframe outputs are covered.
**Competitive reference**: Devex Impact, Amp Impact (Salesforce), TolaData

---

### I2. Indicator Baseline & Target Tracking with SMART Validation
**Category**: Compliance
**Justification**: OECD DAC evaluation criteria require baseline, mid-line, and end-line values against verified indicators. SMART validation (Specific, Measurable, Achievable, Relevant, Time-bound) flags indicators that will fail donor audits before submission.
**Implementation**: Add `create_indicator()` storing baseline/target as `Decimal`, collection frequency, and responsible data source; `validate_smart_indicator()` applies heuristic checks returning a structured findings list with fix hints.
**Competitive reference**: LogAlto, Amp Impact, DHIS2

---

### I3. Budget Variance & Burn-Rate Analytics
**Category**: Performance
**Justification**: Budget overruns are the #1 cause of programme suspension; real-time burn-rate against planned spend gives PMs early warning weeks before month-end finance reports. Competitors like Salesforce NPSP lack per-activity burn analytics.
**Implementation**: Add `budget_variance_report()` that computes planned-vs-actual spend, monthly burn rate as `Decimal`, months-to-zero, and a risk colour (green/amber/red) per activity and at programme rollup.
**Competitive reference**: Unit4 ERP (GAIN), SAP Grants Management, Frappe ERPNext

---

### I4. Milestone Critical Path Analysis
**Category**: Feature
**Justification**: Donors ask "which activities are on the critical path?" — without it, project managers protect all tasks equally and miss the few that will delay the programme end date. This is standard in Primavera P6 / MS Project but absent from NGO M&E tools.
**Implementation**: Add `compute_critical_path()` that topologically sorts activities by dependency edges, identifies float, flags zero-float activities as critical, and returns earliest/latest start dates.
**Competitive reference**: MS Project, Primavera P6, TeamGantt

---

### I5. Adaptive Management Trigger Rules
**Category**: AI/ML
**Justification**: USAID Collaborating, Learning and Adapting (CLA) framework requires documented programme adaptations triggered by evidence. Automated rule-based triggers (e.g., "if achievement_pct < 60% at mid-point, alert PM") close the learn-adapt loop without requiring dashboard vigilance.
**Implementation**: Add `register_adaptive_trigger()` storing condition expressions (field, operator, threshold, window_days) and `evaluate_adaptive_triggers()` that runs all rules against current outputs, emitting `adaptive_management_alert` events with recommended actions.
**Competitive reference**: USAID CLA Toolkit integration, TolaData, ActivityInfo

---

### I6. Beneficiary Disaggregation & Reach Tracking
**Category**: Compliance
**Justification**: All major donors mandate beneficiary counting disaggregated by sex, age, disability status, and geographic location (IASC standards). Tools that omit disaggregation force double-entry into separate spreadsheets, causing reporting errors.
**Implementation**: Add `record_beneficiary_reach()` storing counts keyed by disaggregation dimensions (sex, age_band, disability, location_level); `beneficiary_summary()` pivots dimensions into a cross-tab report suitable for donor templates.
**Competitive reference**: DHIS2, ODK Central, KoboToolbox reporting

---

### I7. Evidence Document Attachment Registry
**Category**: Feature
**Justification**: Auditors routinely disqualify outputs that lack source documentation. Maintaining an evidence registry linked to specific outputs and field data records closes the documentation chain and enables one-click audit packs.
**Implementation**: Add `attach_evidence()` linking a document reference (URI, hash, mime-type) to any output or field data record; `generate_evidence_pack()` returns a manifest of all evidence items for a programme, grouped by output.
**Competitive reference**: Salesforce Files + Amp Impact, ContractPodAi audit trail

---

### I8. Work Plan Auto-Generation from Logframe
**Category**: Feature
**Justification**: M&E officers waste 2-4 hours manually translating approved logframes into activity work plans. Auto-generating a draft work plan from logframe outputs and activities eliminates this rework and ensures consistency between planning documents.
**Implementation**: Add `generate_work_plan_from_logframe()` that reads logframe outputs, creates placeholder activities with proportional budget splits, stores them in a draft state, and returns a structured work plan ready for PM review.
**Competitive reference**: Devex Impact, LogAlto, Amp Impact

---

### I9. Donor Report Template Engine
**Category**: Feature
**Justification**: Programme staff spend up to 40% of their time on donor reporting. A template engine pre-mapped to USAID ADS 201, FCDO logframe, and UN OCHA formats generates 80% of report content from live programme data, reducing reporting burden dramatically.
**Implementation**: Add `generate_donor_report()` accepting a `template_id` (usaid_fy, fcdo_annual, un_ocha_sitrep) and programme snapshot; renders a structured dict aligned to the donor's required sections and flags data gaps.
**Competitive reference**: Amp Impact Salesforce templates, DevResults, IATI Publisher

---

### I10. IATI Standard XML Export
**Category**: Integration
**Justification**: Any NGO receiving FCDO, USAID, or EU funds is required to publish to the International Aid Transparency Initiative (IATI) registry. Generating valid IATI XML from programme data removes a compliance bottleneck that currently requires a dedicated data officer.
**Implementation**: Add `export_iati_activity()` that maps programme/logframe/budget/output data to IATI 2.03 schema elements, validates against mandatory fields, and returns well-formed XML bytes.
**Competitive reference**: IATI Publisher, Aid:Stream, Aidstream.org

---

### I11. Geospatial Activity Heatmap Data
**Category**: Feature
**Justification**: Donors and clusters require geographic coverage maps to avoid duplication and identify gaps. Programmes that geocode activities at ward/sub-county level can demonstrate reach without expensive GIS specialist time.
**Implementation**: Add `record_activity_location()` storing GeoJSON point/polygon and admin boundary codes (ISO 3166-2 / GADM); `get_geographic_coverage()` returns a FeatureCollection of all programme activities for direct Leaflet/MapLibre consumption.
**Competitive reference**: Reliefweb, ActivityInfo, OCHA HDX

---

### I12. Risk Register with Escalation Scoring
**Category**: Compliance
**Justification**: Fiduciary risk frameworks (FCDO MER, USAID ADS 303) require maintained risk registers. Scoring probability × impact and auto-escalating high-risk items to programme managers reduces risk of unsupported audit findings.
**Implementation**: Add `log_risk()` storing risk description, category, probability (1-5), impact (1-5), mitigation, and owner; `risk_heatmap()` returns risk matrix data sorted by composite score with threshold-triggered escalation flags.
**Competitive reference**: Riskonnect, LogAlto risk module, Amp Impact Risk

---

### I13. Offline-First Field Data Queue
**Category**: UX
**Justification**: Field officers in low-connectivity areas (Northern Kenya, DRC) cannot submit observation data in real time. Queuing submissions locally and bulk-syncing when connectivity returns prevents data loss and reduces resubmission errors.
**Implementation**: Add `queue_field_data_offline()` that accepts a batch of pre-timestamped records with a device_id, deduplicates on (device_id, collection_date, location) composite key, and bulk-inserts verified-false records returning a sync manifest.
**Competitive reference**: KoboToolbox offline, ODK Collect, CommCare

---

### I14. Learning Review & After-Action Record
**Category**: Feature
**Justification**: USAID CLA and FCDO MOPAN assessments score organisations on whether lessons learned are documented and fed back into programme design. A structured after-action record closes the accountability loop that most tools ignore.
**Implementation**: Add `create_learning_review()` linking to a programme/activity, storing what worked, what did not, root causes, and recommended changes; `list_learning_reviews()` with filter by sector enables cross-programme knowledge extraction.
**Competitive reference**: TolaData learning module, Amp Impact Lessons Learned, Devex Impact

---

### I15. Programme Health Score (Composite KPI)
**Category**: AI/ML
**Justification**: Portfolio managers reviewing 20+ programmes cannot read full progress reports for each. A single composite score (budget burn vs plan, activity completion, output achievement, risk level, data quality) surfaces lagging programmes in seconds.
**Implementation**: Add `compute_programme_health_score()` that weights five sub-scores (budget_adherence 25%, activity_completion 25%, output_achievement 25%, risk_level 15%, data_quality 10%) into a 0-100 index with colour band and breakdown explanation.
**Competitive reference**: Amp Impact Health Score, Salesforce CRM Analytics NGO pack, Devex Impact dashboard
