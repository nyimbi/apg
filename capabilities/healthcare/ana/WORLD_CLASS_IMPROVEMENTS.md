# Clinical Analytics — World-Class Improvement Plan

**Capability**: `healthcare_ana` | **Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Federated Cohort Computation with Privacy-Preserving Aggregation

Current cohort membership is held entirely in-process. Replace with a federated query model where each data partition (EMR, lab, pharmacy) computes local cohort statistics using differential privacy (ε-DP noise injection). Only aggregate summaries leave each partition. Eliminates PHI centralisation risk while enabling cross-silo population studies. Compose with `healthcare_emr` and `healthcare_lab` via the APG event bus.

## 2. Streaming Real-Time Metric Ingestion via CDC

Metric records are written synchronously via `record_metric`. Introduce a Change Data Capture (CDC) pipeline (PostgreSQL logical replication → Kafka/Redpanda) so that every clinical event (discharge, lab result, order) automatically materialises quality metrics. Latency drops from hours to seconds. The `mqeb` dependency already declares this intent; wire it fully.

## 3. Causal Inference for Intervention Effectiveness

`clinical_pathway_effectiveness` currently returns zeros ("no pathway enrolment data"). Replace with a Doubly Robust estimator (IPW + outcome regression) trained on historical cohort data. Proper causal estimates separate selection bias from true treatment effect, enabling defensible ROI claims for care programmes.

## 4. Adaptive Readmission Model with Concept Drift Detection

`predictive_readmission_score` uses a static heuristic. Add Population Stability Index (PSI) monitoring triggered weekly by `schd`. When PSI > 0.2, automatically queue a retraining job. Drift detection closes the model-governance loop without human intervention and keeps AUC above the 0.70 deployment threshold.

## 5. Multi-Level Benchmarking with Percentile Ranking

`benchmark_comparison` returns a hard-coded percentile of 75 or 40. Replace with a bootstrapped empirical CDF built from peer-group metric observations stored per tenant. Return exact percentile rank, IQR, and z-score. Makes benchmark reports defensible in regulatory submissions.

## 6. FHIR R4 Bidirectional Synchronisation

All entity types map to FHIR R4 resources (Patient, Observation, MeasureReport, RiskAssessment). Implement an async FHIR gateway that ingests from external EHR systems and exports reports as FHIR Bundle/MeasureReport. Removes manual data-entry burden and enables CMS interoperability rule compliance.

## 7. Explainable AI (XAI) Prediction Outputs

`generate_prediction` returns a probability score with no feature-level attribution. Add SHAP value computation (using `shap` library or a lightweight TreeSHAP port) so every prediction includes top-5 feature contributions with direction and magnitude. Clinicians can act on explanations; black-box scores are ignored.

## 8. Automated HEDIS Measure Calculation

Quality indicators require manual numerator/denominator entry. Implement a rules engine that computes HEDIS measures (e.g., CBP, HbA1c Control, Breast Cancer Screening) directly from structured cohort data. A `HEDISMeasureEngine` class consumes CohortResponse + MetricRecordResponse and emits QualityIndicatorResponse automatically, removing a major source of data-entry error.

## 9. Longitudinal Patient Timeline Reconstruction

No existing method lets analysts trace a patient's clinical journey across episodes. Add `patient_longitudinal_timeline(patient_id, start, end)` that assembles ordered clinical events from cohort memberships, care gaps, metric records, and prediction scores into a single timeline object. Enables root-cause analysis of adverse outcomes.

## 10. Anomaly Detection for Sentinel Events

`disease_surveillance` uses a simple threshold (> 50 incidents). Replace with an online Seasonal-Trend Decomposition using LOESS (STL) that models expected incidence rates from historical baselines and flags deviations beyond 3σ. Reduces both false positives (routine seasonality) and false negatives (slow-burn outbreaks).

## 11. Population Stratification by Social Determinants of Health (SDOH)

`population_health_report` stratifies only by clinical segment and ICD-10 codes. Add SDOH strata (deprivation index, rurality, housing instability) sourced from census linkage. SDOH-adjusted rates uncover health equity gaps invisible in raw clinical data and are increasingly required by CMS value-based programmes.

## 12. Automated Root-Cause Analysis for Below-Target QIs

When a QualityIndicator is recorded as `below_target`, trigger an async root-cause analysis workflow that correlates the indicator with recent changes in care gap rates, staffing levels, and equipment utilisation. Emit a structured `RCAReport` with ranked hypotheses. Closes the PDCA loop without analyst intervention.

## 13. Encrypted Audit Trail with Tamper Evidence

`_audit` writes plain dicts to a list. Replace with an append-only Merkle-hash chain stored in PostgreSQL (each event stores `SHA-256(prev_hash || event_json)`). Tamper evidence is verifiable without a separate blockchain, satisfies HIPAA audit control requirements, and survives service restarts.

## 14. Multi-Horizon LOS Prediction Ensemble

`predictive_los` uses a single regression heuristic. Build a stacked ensemble combining: (a) DRG-specific historical median from billing data, (b) XGBoost trained on age/comorbidities/procedure mix, (c) LSTM sequence model over prior admission LOS. Produce calibrated prediction intervals at 50%, 80%, and 95% confidence. Enables bed-management optimisation 24 hours before discharge.

## 15. Composable Analytics DSL for Ad-Hoc Queries

Power users currently call individual service methods. Implement a minimal declarative DSL (e.g., `SELECT cohort WHERE segment='chronic' MEASURE readmission_rate COMPARE national PERIOD Q1-2026`) that compiles to async service method chains. The DSL is parsed by a PEG grammar, validated against the capability contract, and returns a structured `AnalyticsQuery` result. Democratises analytics access for clinical informaticists without Python skills.
