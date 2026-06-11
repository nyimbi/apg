# Performance Management

## Overview
Telecom network performance management covering KPI monitoring across all network layers,
SLA compliance tracking with breach alerting, capacity utilisation forecasting, trend analysis,
configurable threshold management, benchmark gap analysis, anomaly detection, subscriber impact
scoring, and scheduled performance reporting.

## Capability ID
`telecom_per`

## Provides
- kpi_monitoring_workflow: Per-layer KPI recording, status tracking, and anomaly detection
- sla_compliance_workflow: SLA measurement with breach detection, penalty calculation, and notification
- capacity_utilisation_workflow: Utilisation state, congestion alerting, and heatmap generation
- trend_reporting_workflow: Historical trend analysis and degradation detection
- performance_reporting_workflow: Scheduled multi-period and compliance report generation
- threshold_management_workflow: Approval-gated threshold configuration with adaptive suggestions
- benchmark_analysis_workflow: Gap analysis vs internal/regulatory/industry targets
- per_agent_workflow: Performance monitoring automation agents
- subscriber_impact_workflow: Degradation event impact scoring and P1–P4 prioritisation
- intelligence_feed_workflow: Cross-capability performance intelligence publishing

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
| /telecom-per/alerts | GET/POST | Alert management | telecom_per:alerts |
| /telecom-per/intelligence | POST | Intelligence feed | telecom_per:intelligence |

## Core Service Methods

### Recording
- `record_kpi(kpi_id, tenant_id, kpi_category, kpi_name, value, baseline_value, unit, network_layer, recorded_at)`
- `update_kpi_status(kpi_id, tenant_id, new_status)`
- `record_sla_compliance(compliance_id, tenant_id, sla_type, customer_id, target_value, actual_value, period)`
- `record_capacity(record_id, tenant_id, resource_reference, capacity_state, utilisation_pct, forecast_horizon_days, recorded_at)`
- `record_trend(trend_id, tenant_id, kpi_id, trend_direction, lookback_days, forecast_value, recorded_at)`
- `set_threshold(threshold_id, tenant_id, kpi_name, network_layer, warning_value, critical_value, action, approval_reference, set_by)`
- `record_benchmark(benchmark_id, tenant_id, benchmark_type, kpi_name, benchmark_value, current_value, recorded_at)`
- `generate_report(report_id, tenant_id, report_period, fmt, approval_reference, generated_by, generated_at)`
- `register_agent(agent_id, tenant_id, name, runtime, role, scope)`

### Analytics (async)
- `kpi_collection(network_id, period, kpi_types)` — aggregate KPIs per type with p95 stats
- `sla_compliance_check(customer_id, service_id, period)` — breach rate and penalty trigger
- `network_quality_score(region, period)` — composite 0–100 quality score
- `call_drop_analysis(period, cell_ids)` — per-cell drop rates vs 2% SLA threshold
- `data_throughput_analytics(period, segment)` — DL/UL mean/median/p95
- `capacity_utilisation(network_element_id, period)` — mean, peak, trend, overload risk
- `benchmarking(competitor_data, period)` — gap analysis with position classification
- `performance_trending(kpi, periods)` — linear regression slope and direction
- `automated_kpi_report(period, audience)` — executive/technical/regulatory reports
- `performance_alert(kpi, threshold, current_value)` — deduplicated alerting
- `call_drop_analytics(period)` — drop counts by cell and reason
- `throughput_analytics(period)` — mean/p50/p95 DL and UL
- `raise_performance_alert(kpi_id, alert_type, severity, value, threshold)`
- `acknowledge_alert(alert_id, acknowledged_by)`
- `close_alert(alert_id, resolution)`
- `record_nps(customer_id, score, comment)` — NPS survey recording
- `nps_analytics(period)` — NPS score = %promoters − %detractors
- `bulk_import_kpis(kpi_rows)` — batch KPI ingestion
- `export_kpis(format)` — JSON or CSV export
- `performance_compliance_report(period)` — compliance rate and breach summary
- `health_check()` — service health status

### World-Class Enhancements (async)
- `detect_kpi_anomalies(kpi_name, lookback_days, z_threshold)` — z-score + IQR fencing
- `correlate_degradation(kpi_ids, correlation_threshold)` — Pearson correlation matrix
- `compute_sla_penalty(compliance_id, sla_tier, credit_rate_per_pct)` — tiered penalty credits
- `suggest_threshold_updates(lookback_days, warning_percentile, critical_percentile)` — adaptive thresholds
- `end_to_end_service_quality(service_id, period)` — 0–1000 MOS-like E2E score
- `subscriber_impact_score(event_id, affected_cells, affected_subscriber_count, degradation_duration_minutes)` — P1–P4 triage
- `capacity_heatmap(region, granularity, period)` — time×resource utilisation matrix
- `generate_compliance_evidence(regulator, standard, period)` — SHA-256 evidence package
- `publish_performance_intelligence(period)` — cross-capability intelligence feed

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
- threshold_changed, per_agent_registered, kpi_anomalies_detected
- degradation_correlation_detected, sla_penalty_computed, threshold_suggestions_generated
- e2e_service_quality_computed, subscriber_impact_scored, capacity_heatmap_hotspots_detected
- compliance_evidence_generated, performance_intelligence_published

## Edge Cases Handled
- SLA breach direction is parameter-dependent: latency/loss use > target as breach, throughput uses < target
- SLA breach notification is a required field — breach without notification flag denied
- Benchmark gap_pct computed as (benchmark - current) / benchmark × 100, handles negative gaps
- Capacity states congested and overloaded both trigger audit events
- Trend direction=degrading fires audit event immediately, not on report schedule
- Anomaly detection applies IQR clipping before z-score computation to avoid outlier inflation
- Alert deduplication uses last-10 scan; `correlate_degradation` provides upstream suppression data
- Subscriber impact scoring < 1,000 subscriber-minutes still logged (P4) for audit trail completeness

## Composability Notes
Consumes KPI raw data from telecom_net (performance records) and telecom_ana (analytics pipeline).
Feeds SLA breach data to telecom_bil (SLA credit calculations via compute_sla_penalty).
Capacity forecasts feed telecom_pro (resource reservation planning).
Threshold configuration integrates with telecom_qos for QoS tuning.
Performance intelligence feed publishes to apg.intel.per_feed for intel, telecom_ana, and telecom_bil.

## Copyright
© 2025 Datacraft — www.datacraft.co.ke
