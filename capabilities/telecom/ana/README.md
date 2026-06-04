# Telecom Analytics

## Overview
Provides network performance analytics, churn prediction, ARPU analysis, usage pattern analytics, and revenue assurance for telecom operators. Integrates with ML model management to surface predictive insights, anomaly detection, and customer segmentation across all network layers.

## Capability ID
`telecom_ana`

## Provides
- analytics_pipeline: End-to-end analytics run orchestration
- churn_prediction_workflow: ML-driven subscriber churn scoring
- arpu_analysis_workflow: Average revenue per user trend analysis
- usage_pattern_workflow: Subscriber usage profiling and segmentation
- revenue_assurance_workflow: Revenue leak detection and reconciliation
- network_performance_analytics: Per-layer KPI aggregation and trending
- customer_segmentation_workflow: Rule-based and ML segment definition
- anomaly_detection_workflow: Statistical and ML anomaly flagging
- model_management_workflow: Model registration, versioning, validation
- analytics_reporting_workflow: Multi-format scheduled report generation
- analytics_agent_workflow: Automated agent-driven analytics tasks

## Requires
| Capability | Reason |
|------------|--------|
| auth | User authentication and permission checks |
| audl | Audit trail for all write operations |
| mten | Multi-tenancy context enforcement |
| conf | Runtime configuration management |
| ntfy | Breach and anomaly notifications |
| nlpc | NLP for search and text classification |
| moni | Operational monitoring |
| mqeb | Event stream via bytewax |
| schd | Scheduled report and batch job triggers |

## Configuration
| Key | Description |
|-----|-------------|
| analysis.supported_analysis_types | Valid analysis types (10 types) |
| churn.supported_risk_levels | low / medium / high / critical |
| revenue.supported_categories | 8 revenue stream categories |
| models.supported_model_types | regression, classification, clustering, etc. |
| governance.cross_tenant_data_denied | Blocks cross-tenant data access |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /telecom-ana/dashboard | GET | Analytics dashboard | telecom_ana:view |
| /telecom-ana/analysis | GET/POST | Analysis run management | telecom_ana:analysis |
| /telecom-ana/churn | GET/POST | Churn predictions | telecom_ana:churn |
| /telecom-ana/revenue | GET/POST | Revenue assurance events | telecom_ana:revenue |
| /telecom-ana/anomalies | GET/POST | Detected anomalies | telecom_ana:anomalies |
| /telecom-ana/models | GET/POST | Model registry | telecom_ana:models |
| /telecom-ana/reports | GET/POST | Analytics reports | telecom_ana:reports |
| /telecom-ana/agents | GET/POST | Agent workbench | telecom_ana:admin |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | tenant_context_present=False | deny |
| analysis_type_supported | unsupported type | deny |
| churn_model_required | model not registered | deny |
| confidence_score_invalid | score outside [0,1] | deny |
| unapproved_model_deployment_denied | agent deploys without approval | deny |
| cross_tenant_data_denied | agent accesses cross-tenant data | deny |
| ana_batch_requires_bytewax | batch not using bytewax | deny |

## Data Models
- AnaAnalysisRun: id, tenant_id, analysis_type, owner_id, time_granularity, evidence_reference
- AnaMetric: id, tenant_id, metric_type, metric_name, value, baseline_value, aggregation_type
- AnaChurnPrediction: id, tenant_id, customer_id, risk_level, confidence_score, model_id
- AnaRevenueEvent: id, tenant_id, category, amount, currency, period
- AnaSegment: id, tenant_id, segment_name, criteria, customer_count
- AnaNetworkAnalytics: id, tenant_id, network_layer, metric_name, value, threshold
- AnaAnomaly: id, tenant_id, anomaly_type, confidence_score, description, evidence_reference
- AnaModel: id, tenant_id, model_type, model_name, version, validation_reference
- AnaReport: id, tenant_id, report_format, analysis_id, approval_reference
- AnaAgent: id, tenant_id, name, runtime, role, scope

## Streaming Events
- analysis_run_recorded, metric_recorded, churn_prediction_recorded
- revenue_assurance_event_recorded, segment_recorded, network_analytics_recorded
- anomaly_detected, model_registered, report_generated, ana_agent_registered

## Edge Cases Handled
- Churn prediction blocked if referenced model not registered in same tenant
- Confidence scores outside [0,1] rejected at service layer before storage
- Revenue categories enforced; "data" ≠ "data_services" — exact match required
- Model deployment by agents requires human approval even when confidence is high
- Cross-tenant analytics queries denied at rule engine level regardless of agent identity

## Composability Notes
Consumes data from telecom_net (performance) and telecom_cus (churn signals). Feeds telecom_per (KPI baselines) and telecom_bil (revenue assurance). Agents composed with nlpc for NL query interfaces and ragn for report augmentation.
