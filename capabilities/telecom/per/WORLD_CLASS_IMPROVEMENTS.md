# World-Class Improvements — telecom_per (Performance Management)

## 1. Predictive Capacity Exhaustion with ML Regression

Replace the simple first/second-half trend heuristic in `capacity_utilisation` with a proper
linear regression that produces a **days-to-exhaustion** estimate. Feed training data from
the full `PerCapacityRecord` history, fit slope/intercept in-service, and surface a
`days_to_100pct` field. Operators can pre-provision before congestion rather than react.

## 2. Anomaly Detection via Z-Score / IQR Fencing

Add `async detect_kpi_anomalies(kpi_name, lookback_days, tenant_id)` that computes a rolling
mean and standard deviation over the stored KPI history, then flags any measurement beyond
±2σ as an anomaly. IQR fencing handles skewed distributions (latency, packet loss). The method
emits an audit event per anomaly and returns a structured list — eliminating the manual
threshold-hunting that currently drives most critical KPI alerts.

## 3. Root-Cause Correlation Engine

Add `async correlate_degradation(kpi_ids, window_minutes, tenant_id)` that performs pairwise
Pearson correlation between KPI time-series within a sliding window. Correlations above 0.85
are flagged as likely causal pairs. Output feeds the alert suppression logic (improvement #4)
to avoid storm notifications when a single upstream fault fans out across dozens of KPIs.

## 4. Alert Storm Suppression and Intelligent Grouping

The current `performance_alert` deduplication window only checks the last 10 alerts. Replace
with a time-bucketed (5-minute TTL) suppression map keyed on `(kpi, network_element, severity)`.
Add `async group_alerts(tenant_id, window_minutes)` that clusters open alerts by root-cause
correlation into a single incident record. This reduces operator fatigue and shortens MTTR.

## 5. SLA Penalty Auto-Calculator with Tiered Credits

Extend `sla_compliance_check` to compute penalty credits per the ITU-T G.826 and ETSI TS 102 250
frameworks. Add `async compute_sla_penalty(compliance_id, sla_tier, tenant_id)` that takes the
breach duration/magnitude and outputs a `credit_amount` in currency units, serialised into the
`telecom_bil` event stream. Eliminates manual credit note generation for high-volume breach periods.

## 6. Real-Time KPI Streaming Sink (Bytewax Integration)

Add `async stream_kpi_batch(records, stream_topic, tenant_id)` that publishes KPI measurements
to the Bytewax event stream `apg.telecom.per.kpi.raw` in Avro-compatible JSON. The streaming
path bypasses the in-memory dict stores, targeting sub-second latency for NOC dashboards. Include
back-pressure handling: if the queue depth exceeds 10,000 events, log a warning and apply
exponential back-off before the next publish.

## 7. Multi-Dimensional Capacity Heatmap Data API

Add `async capacity_heatmap(region, granularity, period, tenant_id)` that aggregates utilisation
percentages across resource references into a time×resource matrix suitable for rendering as a
heatmap. Granularity values: `hourly | daily | weekly`. Returns a compact `{"cells": [[t, r, util]]}` 
structure consumable by any frontend charting library without further transformation.

## 8. Automated Threshold Tuning via Adaptive Feedback

Add `async suggest_threshold_updates(tenant_id, lookback_days)` that examines the breach history,
computes the empirical 95th and 99th percentile of each KPI, and computes a recommended
`warning_value` and `critical_value` pair. Recommendations are returned as a diff against current
thresholds — human approval still required to commit (preserving the existing approval gate).

## 9. End-to-End Service Quality Score (E2E-SQS)

Add `async end_to_end_service_quality(service_id, period, tenant_id)` that combines radio KPIs
(RSRP, SINR, PRB utilisation) with core KPIs (latency, packet loss) and transport KPIs (jitter,
BER) into a single dimensionless quality score on a 0–1000 MOS-like scale. Score buckets:
Excellent ≥ 800, Good 600–799, Fair 400–599, Poor < 400. Integrates with `telecom_qos` for
joint optimisation.

## 10. Geo-Spatial Coverage Gap Analysis

Add `async coverage_gap_analysis(bounding_box, technology, tenant_id)` that ingests signal
strength KPIs tagged with latitude/longitude, identifies areas with RSRP below −110 dBm, and
returns a GeoJSON FeatureCollection of gap polygons. Enables targeted radio network planning
without exporting raw data to an external GIS tool.

## 11. Subscriber-Impact Scoring for Degradation Events

Add `async subscriber_impact_score(event_id, tenant_id)` that cross-references a KPI degradation
event with the active subscriber count on the affected cell, computes affected subscriber-minutes,
and assigns a business-impact tier (P1–P4). P1 events (>100,000 affected subscriber-minutes)
auto-create a high-priority incident in the ITSM adapter.

## 12. Regulatory Compliance Evidence Package Generator

Extend `performance_compliance_report` into a full evidence package: add
`async generate_compliance_evidence(regulator, standard, period, tenant_id)` that bundles KPI
summaries, SLA compliance records, breach notifications, threshold change approvals, and audit
trails into a structured JSON manifest signed with a SHA-256 hash. Targets BEREC, CA (Kenya),
and GSMA compliance frameworks. Eliminates the manual collation step before regulatory submissions.

## 13. Capacity Reservation and Forecasting API for telecom_pro

Add `async forecast_capacity_need(resource_id, horizon_days, tenant_id)` that extracts the
utilisation time-series, fits an exponential smoothing model (Holt-Winters), and returns a
per-day forecast vector with 80% and 95% confidence intervals. The output is published to the
`apg.telecom.pro.capacity_reservation` event topic so `telecom_pro` can auto-reserve bandwidth
before the congestion threshold is breached.

## 14. Peer-Group Benchmarking with Statistical Significance Testing

Extend `benchmarking` to include a two-sample t-test between own KPI distribution and the
supplied competitor benchmark, returning a p-value and effect size (Cohen's d). Surface
`statistically_significant: bool` alongside the existing `position` field. This prevents
operators from acting on benchmarking noise — an improvement over the current mean-only gap
percentage which can mislead when sample sizes are small.

## 15. Cross-Capability Performance Intelligence Feed

Add `async publish_performance_intelligence(period, tenant_id)` that consolidates the top
degrading KPIs, SLA breach hotspots, capacity risk nodes, and NPS detractor drivers into a
single structured intelligence payload published to the `apg.intel.per_feed` event topic.
Downstream capabilities (`intel`, `telecom_ana`, `telecom_bil`) consume this feed to drive
proactive optimisation, billing adjustments, and customer retention actions — completing the
composability loop described in the capability spec.
