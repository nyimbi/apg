# Performance Management

## Overview
Telecom network performance management covering KPI monitoring across all network layers, SLA compliance tracking with breach alerting, capacity utilisation forecasting, trend analysis with ML-based predictions, configurable threshold management, benchmark gap analysis, and scheduled performance reporting.

## Capability ID
`telecom_per`

## Provides
- kpi_monitoring_workflow: Per-layer KPI recording and status tracking
- sla_compliance_workflow: SLA measurement with breach detection and notification
- capacity_utilisation_workflow: Utilisation state and congestion alerting
- trend_reporting_workflow: Historical trend analysis and degradation detection
- performance_reporting_workflow: Scheduled multi-period report generation
- threshold_management_workflow: Approval-gated threshold configuration
- benchmark_analysis_workflow: Gap analysis vs internal/regulatory/industry targets
- per_agent_workflow: Performance monitoring automation agents

## Requires
| Capability | Reason |
|------------|--------|
| auth | Authentication |
| audl | Threshold change audit trail |
| mten | Tenant isolation |
| conf | Configuration |
| ntfy | SLA breach and KPI threshold notifications |
| moni | Operational monitoring integration |
| mqeb | Event streaming |
| schd | Scheduled report delivery |
| nlpc | NL query interface for KPI search |

## Configuration
| Key | Description |
|-----|-------------|
| kpis.collection_interval_seconds | 5-minute collection default |
| kpis.retention_days | 365-day KPI retention |
| capacity.utilisation_threshold_pct | Alert at 80% utilisation |
| capacity.forecast_horizon_days | 90-day capacity forecast |
| sla_compliance.grace_period_minutes | 60-minute breach grace period |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /telecom-per/kpis | GET/POST | KPI console | telecom_per:kpis |
| /telecom-per/sla | GET/POST | SLA compliance | telecom_per:sla |
| /telecom-per/capacity | GET/POST | Capacity management | telecom_per:capacity |
| /telecom-per/trends | GET/POST | Trend analysis | telecom_per:trends |
| /telecom-per/thresholds | GET/POST | Threshold management | telecom_per:thresholds |
| /telecom-per/benchmarks | GET/POST | Benchmark analysis | telecom_per:benchmarks |
| /telecom-per/reports | GET/POST | Performance reports | telecom_per:reports |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| kpi_category_not_supported | unknown category | deny |
| kpi_baseline_required | no baseline value | deny |
| sla_breach_notification_required | breach without notification | deny |
| threshold_change_approval_required | no approval reference | deny |
| report_approval_required | no approval reference | deny |
| cross_tenant_data_denied | cross-tenant agent scope | deny |
| unapproved_threshold_change_denied | agent changes thresholds | deny |

## Data Models
- PerKpi: id, tenant_id, kpi_category, kpi_name, value, baseline_value, unit, status, network_layer
- PerSlaCompliance: id, tenant_id, sla_type, customer_id, target_value, actual_value, status, period
- PerCapacityRecord: id, tenant_id, resource_reference, capacity_state, utilisation_pct, forecast_horizon_days
- PerTrend: id, tenant_id, kpi_id, trend_direction, lookback_days, forecast_value
- PerThreshold: id, tenant_id, kpi_name, network_layer, warning_value, critical_value, action, approval_reference
- PerBenchmark: id, tenant_id, benchmark_type, kpi_name, benchmark_value, current_value, gap_pct
- PerReport: id, tenant_id, report_period, format, approval_reference
- PerAgent: id, tenant_id, name, runtime, role, scope

## Streaming Events
- kpi_threshold_breached, sla_breach_detected, capacity_congestion_alert
- trend_degradation_detected, report_generated, forecast_computed, benchmark_gap_detected
- threshold_changed, per_agent_registered

## Edge Cases Handled
- SLA breach direction is parameter-dependent: latency/loss use > target as breach, throughput uses < target
- SLA breach notification is a required field — breach without notification flag denied
- Benchmark gap_pct computed as (benchmark - current) / benchmark × 100, handles negative gaps
- Capacity states congested and overloaded both trigger audit events
- Trend direction=degrading fires audit event immediately, not on report schedule

## Composability Notes
Consumes KPI raw data from telecom_net (performance records) and telecom_ana (analytics pipeline). Feeds SLA breach data to telecom_bil (SLA credit calculations). Capacity forecasts feed telecom_pro (resource reservation planning). Threshold configuration integrates with telecom_qos for QoS tuning.
