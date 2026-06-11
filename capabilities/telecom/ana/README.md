# Telecom Analytics

## Overview
Provides network performance analytics, churn prediction, ARPU analysis, usage pattern analytics, and revenue assurance for telecom operators. Integrates with ML model management to surface predictive insights, anomaly detection, and customer segmentation across all network layers.

## Capability ID
`telecom_ana`

## Provides
- analytics_pipeline: End-to-end analytics run orchestration
- churn_prediction_workflow: ML-driven subscriber churn scoring
- arpu_analysis_workflow: Average revenue per user trend analysis and price-elasticity modelling
- usage_pattern_workflow: Subscriber usage profiling and segmentation
- revenue_assurance_workflow: Revenue leak detection, CDR reconciliation, and billing alignment
- network_performance_analytics: Per-layer KPI aggregation, trending, and spectrum efficiency
- customer_segmentation_workflow: Rule-based and ML segment definition
- anomaly_detection_workflow: Statistical and ML anomaly flagging
- model_management_workflow: Model registration, versioning, validation, and drift detection
- analytics_reporting_workflow: Multi-format scheduled report generation
- analytics_agent_workflow: Automated agent-driven analytics tasks
- 5g_slice_sla_workflow: Per-slice (eMBB/URLLC/mMTC) SLA compliance tracking
- capacity_planning_workflow: Predictive hotspot detection and demand forecasting
- rca_workflow: Automated KPI degradation root-cause analysis
- journey_analytics_workflow: Markov-chain subscriber lifecycle modelling
- dag_orchestration_workflow: Composable analytics DAG execution with lineage

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
| telecom_net | Network performance data feed |
| telecom_cus | Subscriber churn signals |
| telecom_bil | Revenue assurance reconciliation |
| telecom_per | KPI baseline exchange |

## Configuration
| Key | Description |
|-----|-------------|
| analysis.supported_analysis_types | Valid analysis types (10 types) |
| churn.supported_risk_levels | low / medium / high / critical |
| revenue.supported_categories | 8 revenue stream categories |
| models.supported_model_types | regression, classification, clustering, etc. |
| governance.cross_tenant_data_denied | Blocks cross-tenant data access |
| drift.psi_threshold | PSI threshold for model drift detection (default 0.25) |
| capacity.utilisation_alert_pct | PRB utilisation % that triggers hotspot alert (default 80) |

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
| /telecom-ana/spectrum | GET | Spectrum efficiency by cell | telecom_ana:network |
| /telecom-ana/slices | GET | 5G slice SLA compliance | telecom_ana:network |
| /telecom-ana/hotspots | GET | Capacity hotspot forecast | telecom_ana:network |
| /telecom-ana/rca | GET | KPI root-cause analysis | telecom_ana:analysis |
| /telecom-ana/reconcile | POST | Revenue CDR reconciliation | telecom_ana:revenue |
| /telecom-ana/drift | GET | Model drift PSI check | telecom_ana:models |
| /telecom-ana/journey | GET | Subscriber journey Markov analysis | telecom_ana:churn |
| /telecom-ana/elasticity | GET | ARPU price elasticity estimate | telecom_ana:revenue |
| /telecom-ana/dag | GET | Analytics DAG execution status | telecom_ana:admin |

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
| slice_type_constrained | slice_type not in {embb,urllc,mmtc} | ValueError |
| drift_requires_registered_model | model_id not in registry | ValueError |

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

## Key Service Methods

### Core Write Operations
| Method | Description |
|--------|-------------|
| `record_analysis_run()` | Register an analytics pipeline run |
| `record_metric()` | Ingest a KPI or derived metric value |
| `record_churn_prediction()` | Store an ML churn risk prediction |
| `record_revenue_event()` | Record a revenue assurance event |
| `record_segment()` | Define a customer segment |
| `record_network_analytics()` | Ingest network layer KPI data point |
| `record_anomaly()` | Flag a detected anomaly |
| `register_model()` | Register a predictive model in the tenant registry |
| `generate_report()` | Produce a formatted analytics report |
| `register_agent()` | Register an analytics automation agent |

### Analytics Queries (async)
| Method | Description |
|--------|-------------|
| `network_traffic_analytics()` | Per-layer throughput and breach rate analysis |
| `subscriber_analytics()` | Subscriber base statistics and churn risk distribution |
| `revenue_analytics()` | ARPU, total revenue, and revenue-by-category breakdown |
| `churn_prediction()` | Real-time churn risk with what-if feature overrides |
| `roaming_analytics()` | Roaming revenue, data, and partner network stats |
| `handset_analytics()` | Device fleet distribution and 5G capability rate |
| `five_g_adoption_analytics()` | 5G subscriber adoption and data uplift vs 4G |
| `data_consumption_trends()` | MoM growth, peak hours, and top app breakdown |
| `network_investment_roi()` | Capex ROI, payback months, and net benefit |
| `competitive_analytics()` | Multi-dimension competitive positioning score |
| `anomaly_detection()` | Z-score anomaly detection on metric time series |
| `churn_risk_scoring()` | Batch churn probability scores for customer list |
| `subscriber_segmentation()` | Segment distribution by ARPU, data, or churn risk |
| `network_performance_analytics()` | Cross-layer availability and performance aggregation |
| `forecast_demand()` | Trend-extrapolation demand forecast over a horizon |
| `analytics_compliance_check()` | Model and report governance compliance verification |
| `bulk_ingest_metrics()` | High-throughput batch metric ingestion |
| `export_analytics_report()` | JSON or CSV report export |
| `health_check()` | Service liveness and resource count |

### World-Class Methods (async)
| Method | Description |
|--------|-------------|
| `arpu_elasticity()` | Log-log price elasticity with bootstrapped CI for a segment |
| `spectrum_efficiency_analytics()` | Bits-per-Hz per cell; flags cells needing radio tuning |
| `model_drift_check()` | PSI-based concept drift detection with retraining recommendation |
| `subscriber_journey_analytics()` | Markov-chain lifecycle model with top intervention recommendation |
| `revenue_reconciliation()` | CDR-vs-billing total reconciliation with leakage candidate list |
| `kpi_root_cause_analysis()` | MI-ranked root-cause candidates for degraded KPIs |
| `predictive_capacity_hotspots()` | Forward-looking cell site utilisation hotspot ranking |
| `analytics_dag_status()` | DAG pipeline lineage and node-level execution status |
| `slice_sla_analytics()` | 5G slice (eMBB/URLLC/mMTC) P99 latency and SLA compliance |

## Streaming Events
- analysis_run_recorded, metric_recorded, churn_prediction_recorded
- revenue_assurance_event_recorded, segment_recorded, network_analytics_recorded
- anomaly_detected, model_registered, report_generated, ana_agent_registered
- model_drift_detected, revenue_reconciliation_mismatch, revenue_reconciliation_ok
- kpi_rca_run, predictive_capacity_hotspots_run, slice_sla_analytics_run
- arpu_elasticity_run, spectrum_efficiency_analytics_run, subscriber_journey_analytics_run

## Edge Cases Handled
- Churn prediction blocked if referenced model not registered in same tenant
- Confidence scores outside [0,1] rejected at service layer before storage
- Revenue categories enforced; exact match required
- Model deployment by agents requires human approval even when confidence is high
- Cross-tenant analytics queries denied at rule engine level regardless of agent identity
- `arpu_elasticity()` returns a no-data sentinel dict when no revenue events exist rather than raising
- `spectrum_efficiency_analytics()` clips bps/Hz calculation to prevent division-by-zero at zero PRB utilisation
- `model_drift_check()` falls back to uniform baseline distribution when validation data not stored
- `revenue_reconciliation()` treats gap within ±0.5% as reconciled to absorb rounding differences
- `slice_sla_analytics()` enforces slice_type enum and returns a no-records sentinel gracefully
- `kpi_root_cause_analysis()` returns undetermined top_cause when no correlated metrics deviate

## Composability Notes
Consumes data from telecom_net (performance) and telecom_cus (churn signals). Feeds telecom_per (KPI baselines) and telecom_bil (revenue assurance). Agents composed with nlpc for NL query interfaces and ragn for report augmentation. The `analytics_dag_status()` method surfaces lineage metadata consumed by the schd and audl capabilities.
