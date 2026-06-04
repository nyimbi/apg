# Clinical Analytics

## Overview
Provides population health analytics, clinical outcomes measurement, readmission prediction, quality indicator tracking, and care gap identification for healthcare tenants. Supports cohort management, predictive model deployment, and structured report generation aligned with CMS Star, Joint Commission, and peer-group benchmarks.

## Capability ID
`healthcare_ana`

## Provides
- population_health_analytics: Segmented population health analysis across chronic, geriatric, maternal, and other clinical populations
- clinical_outcomes_measurement: Record and trend clinical metric types including mortality rate, readmission rate, LOS, and complication rate
- readmission_prediction: Deploy and run ML prediction models (AUC >= 0.70) for 30-day readmission risk
- quality_indicator_tracking: Record numerator/denominator quality indicators and compare against benchmarks
- cohort_management: Create, activate, and archive patient cohorts defined by ICD-10 codes and criteria
- clinical_benchmarking: Compare metrics against national, regional, CMS Star, and peer-group benchmarks
- analytics_report_generation: Generate analytics reports in PDF, Excel, CSV, JSON, HL7 FHIR, and CDA formats
- care_gap_identification: Identify and track clinical care gaps per patient with severity and evidence references
- predictive_model_management: Register, version, and retrain clinical prediction models with approval workflow

## Requires
- auth: Authentication and authorization for PHI access
- audl: Audit trail for all analytics operations
- mten: Multi-tenant isolation to prevent cross-tenant data leakage
- conf: Configuration management for tenant-specific settings
- ntfy: Alerts for care gaps and quality indicator thresholds
- nlpc: Natural language search over cohort criteria and reports
- moni: Operational monitoring for model drift and data pipeline health
- mqeb: Event emission for downstream consumers (EMR, dashboards)
- schd: Scheduled report generation and model retraining jobs

## Configuration

| Key | Type | Description |
|-----|------|-------------|
| tenant_id | string | Tenant identifier |
| prediction.min_auc | float | Minimum AUC required for model deployment (default: 0.70) |
| prediction.retraining_days | int | Days before model retraining is flagged overdue (default: 90) |
| governance.phi_de_identification_required | bool | Require PHI de-identification on data export |
| reporting.supported_formats | list | Allowed report output formats |

## API Routes

| Method | Path | Description | Permission |
|--------|------|-------------|------------|
| GET | /api/healthcare/ana/contract | Capability contract | healthcare_ana:view |
| GET | /api/healthcare/ana/cohorts | List cohorts | healthcare_ana:cohorts |
| POST | /api/healthcare/ana/cohorts | Create cohort | healthcare_ana:cohorts |
| GET | /api/healthcare/ana/cohorts/<id> | Cohort detail | healthcare_ana:cohorts |
| PUT | /api/healthcare/ana/cohorts/<id> | Update cohort | healthcare_ana:cohorts |
| DELETE | /api/healthcare/ana/cohorts/<id> | Delete cohort | healthcare_ana:cohorts |
| GET | /api/healthcare/ana/metrics | List metrics | healthcare_ana:metrics |
| POST | /api/healthcare/ana/metrics | Record metric | healthcare_ana:metrics |
| GET | /api/healthcare/ana/models | List prediction models | healthcare_ana:predictions |
| POST | /api/healthcare/ana/models | Deploy model | healthcare_ana:predictions |
| POST | /api/healthcare/ana/models/<id>/predict | Run prediction | healthcare_ana:predictions |
| GET | /api/healthcare/ana/quality-indicators | List QIs | healthcare_ana:quality |
| POST | /api/healthcare/ana/quality-indicators | Record QI | healthcare_ana:quality |
| GET | /api/healthcare/ana/care-gaps | List care gaps | healthcare_ana:care_gaps |
| POST | /api/healthcare/ana/care-gaps | Identify care gap | healthcare_ana:care_gaps |
| POST | /api/healthcare/ana/care-gaps/<id>/resolve | Resolve care gap | healthcare_ana:care_gaps |
| GET | /api/healthcare/ana/reports | List reports | healthcare_ana:reports |
| POST | /api/healthcare/ana/reports | Generate report | healthcare_ana:reports |
| GET | /api/healthcare/ana/reports/<id> | Report detail | healthcare_ana:reports |
| GET | /api/healthcare/ana/dashboard | Dashboard summary | healthcare_ana:view |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | tenant_context_present=False | deny |
| write_requires_policy | operation_type=write, policy_attached=False | deny |
| phi_export_requires_deidentification | operation=export_data, phi_deidentified=False | deny |
| model_deployment_requires_approval | operation=deploy_model, approval_present=False | deny |
| prediction_auc_threshold | operation=deploy_model, auc_above_threshold=False | deny |
| cross_tenant_data_denied | cross_tenant_access=True | deny |
| cohort_delete_requires_no_active_analyses | operation=delete_cohort, active_analyses_exist=True | deny |
| model_retraining_overdue | operation=generate_prediction, model_retraining_overdue=True | warn |

## Data Models
- CohortCreate/Response: patient cohort with segment, ICD-10 criteria, status, patient_count
- MetricRecordCreate/Response: metric type, value, unit, period, benchmark comparison
- PredictionModelCreate/Response: model type, AUC score, target outcome, deployment status
- QualityIndicatorCreate/Response: indicator code, numerator/denominator, benchmark, performance_status
- CareGapCreate/Response: patient_id, gap_type, severity, evidence_reference, resolution timestamp
- AnalyticsReportCreate/Response: report_type, format, cohort_ids, metric_types, download_url

## Streaming Events
- cohort_created, cohort_updated
- metric_recorded
- prediction_generated, model_deployed
- benchmark_updated
- care_gap_identified
- report_generated
- quality_indicator_updated

## Edge Cases Handled
- Cross-tenant data access is hard-denied regardless of analyst permissions
- PHI de-identification is enforced at the rule layer before any export operation
- Model AUC below 0.70 blocks deployment; a warning is issued when retraining is overdue (>90 days)
- Deleting a cohort with active metric records is denied until records are archived or deleted
- Cohort criteria and ICD-10 codes are stored as structured JSON for downstream querying
- Care gaps track resolution timestamps for SLA reporting

## Composability Notes
Composes naturally with `healthcare_emr` (patient records for cohort membership), `healthcare_lab` (lab result metrics), `healthcare_pha` (medication adherence metrics), and `healthcare_cli` (care plan adherence). Quality indicators feed directly into `healthcare_reg` for regulatory submissions.
