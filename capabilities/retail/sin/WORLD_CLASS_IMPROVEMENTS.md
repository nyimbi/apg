# Store Intelligence — World-Class Improvement Roadmap

**Capability**: `retail_sin` | **Domain**: `retail`
**Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Real-Time Anomaly Detection on Foot Traffic Streams

Current traffic recording is batch-only with no anomaly detection. Add an async streaming pipeline that continuously evaluates incoming traffic counts against rolling z-score baselines per zone. Anomalies (sudden crowd surges, unexpected voids, egress spikes) are flagged as `SinTrafficAnomalyEvent` and routed to the `ntfy` capability. This enables proactive loss-prevention response within seconds rather than after-the-fact reporting.

**Impact**: Reduces shrinkage response latency from hours to seconds.

---

## 2. AI-Assisted Planogram Deviation Classification

The current audit records a human-assigned `compliance_status` string. Replace with a structured deviation classifier: each `deviation_detail` record should carry `deviation_type` (wrong_product, wrong_facing_count, wrong_position, price_tag_missing), `severity_score` (0–100), and an `ai_confidence` float from the vision model. The compliance score should be computed as a weighted sum rather than a status lookup table. This gives buyers actionable root-cause data instead of a single pass/fail score.

**Impact**: Converts audit from binary flag to continuous quality signal feeding category management.

---

## 3. Zone Dwell-Time Cohort Segmentation

`dwell_avg_seconds` is a scalar mean — it discards the distribution. Add `record_dwell_cohort()` that accepts a histogram of dwell buckets (0–30s, 30–120s, 2–5min, 5min+) per zone per interval. Downstream, `heat_map_analytics` should surface the cohort breakdown. Merchandisers can distinguish "browse and leave" zones from "engage and consider" zones, enabling planogram and fixture decisions grounded in actual shopper behaviour.

**Impact**: Unlocks evidence-based fixture design decisions.

---

## 4. Loss Prevention Incident Lifecycle

No loss-prevention (LP) data model exists. Add `SinLossPreventionIncidentCreate/Response` with fields: `incident_type` (shoplifting, staff_theft, admin_error, damage), `zone_id`, `sku`, `estimated_value_loss`, `sensor_ids_involved`, `investigation_status`, and `resolution`. Add `report_lp_incident()`, `escalate_lp_incident()`, and `close_lp_incident()` service methods. Wire OOS alerts with zero sensor activity as an automatic LP suspicion trigger.

**Impact**: Closes the gap between shelf-availability and LP disciplines — currently completely absent.

---

## 5. Sensor Network Health Scoring

`sensor_heartbeat()` marks sensors online/offline but provides no network-level health view. Add `sensor_network_health()` that computes: percentage of sensors online, mean heartbeat age, zones with no coverage, and an overall health score (0–100). Integrate with the `moni` capability to emit a `sensor_network_degraded` event when coverage drops below a configurable threshold. This is prerequisite data quality infrastructure — bad sensors produce misleading analytics.

**Impact**: Makes data quality visible and actionable before it corrupts KPIs.

---

## 6. Peer-Group Benchmarking Engine

`store_ranking()` ranks by absolute KPI value within the tenant's own stores. This is insufficient for multi-format or multi-region chains. Add `benchmark_peer_group()` that: (a) selects a peer group by `store_format` + `region` + `sqm_band`, (b) computes percentile rank per KPI, and (c) surfaces the gap-to-median and gap-to-top-quartile. Enforce the existing `benchmark_min_peer_stores=5` business rule at this layer. This turns KPI reporting from vanity metrics into actionable competitive positioning.

**Impact**: Enables regional managers to set evidence-based improvement targets.

---

## 7. Shopper Journey Attribution Graph

Conversion events are point-in-time and unlinked. Add a `stitch_shopper_journey()` method that groups `SinConversionEvent` records by `session_id` and constructs a directed path graph: entry_zone → browse_zones (ordered by `occurred_at`) → transaction_zone. Compute path frequency and drop-off rates at each transition. This reveals which zone sequences lead to purchase and which zones are journey dead-ends.

**Impact**: Gives visual merchandising teams a data-driven basis for traffic flow redesign.

---

## 8. Dynamic Reorder Point Calculation

`trigger_replenishment()` is a boolean flag with no intelligence. Replace with `calculate_reorder_point()` that, given historical sales velocity (from KPI snapshots), lead time (from config), and desired service level, computes the statistical reorder point using the newsvendor model: `ROP = μ_lead × d + Z_α × σ_d × √lead_time`. Surface this as a `reorder_point` field on `SinShelfAlertResponse`. This transforms reactive OOS alerts into proactive replenishment scheduling.

**Impact**: Reduces stockout frequency while avoiding overstock carrying costs.

---

## 9. Multi-Store Promotional Lift Analysis

No link between promotions and traffic/conversion exists. Add `analyse_promo_lift()` that accepts a `promo_id`, `promo_start`, `promo_end`, and list of `store_ids`. It computes traffic and conversion rate deltas vs. a matched pre-period baseline and a holdout group (stores not running the promo). Returns `lift_pct`, `statistical_significance` (p-value from t-test on daily observations), and confidence interval. Composable with `retail_prm`.

**Impact**: Closes the measurement loop between promotions and in-store behaviour.

---

## 10. Temporal KPI Trend Detection

KPI snapshots are stored but no trend detection exists. Add `detect_kpi_trends()` that applies a linear regression over the last N periods per KPI metric and returns `slope`, `r_squared`, `trend_direction` (improving/stable/degrading), and a `weeks_to_breach_threshold` estimate. Flag stores where a KPI is on a degradation trajectory before it crosses the alert threshold. This shifts store operations from reactive to predictive.

**Impact**: Enables area managers to intervene before stores reach critical KPI levels.

---

## 11. Occupancy Capacity Compliance Tracking

Peak occupancy is recorded but not validated against fire-code capacity limits. Add a `max_capacity` field to `SinStoreCreate`, and add `check_occupancy_compliance()` that checks every traffic count record against `max_capacity * 0.85` (safety margin). Emit a `capacity_limit_approaching` event when the rolling 5-minute occupancy exceeds 80% of legal limit. Mandatory in post-COVID retail environments and increasingly required by insurers.

**Impact**: Reduces regulatory and insurance liability; directly composable with ntfy.

---

## 12. Heatmap Temporal Diff Comparison

Heatmaps are static snapshots. Add `compute_heatmap_diff()` that takes two `heatmap_id` values and returns a signed intensity delta grid, highlighting zones that gained or lost foot traffic between periods (e.g., before vs. after a fixture relocation). Normalise by total store traffic to isolate layout effects from overall volume changes. This is the primary measurement tool for testing store layout hypotheses.

**Impact**: Makes store layout experiments measurable and falsifiable.

---

## 13. Staff Schedule Demand Forecasting

`staff_productivity()` uses hardcoded `staff_count=8` and a fixed 8h×22d assumption — these constants are wrong for every real store. Replace with `forecast_staffing_demand()` that uses trailing 4-week traffic patterns (by hour and day-of-week) to forecast expected foot traffic per period, then applies a configurable `traffic_to_staff_ratio` to produce a staffing demand curve. Output: recommended headcount per shift slot for the next 2 weeks.

**Impact**: Directly reduces payroll cost while maintaining service levels.

---

## 14. Multi-Sensor Fusion for Entry Counting

A single sensor per zone produces noisy counts. Add `fuse_sensor_counts()` that accepts concurrent counts from multiple sensors covering the same zone, applies a Kalman-filter-style weighted average (weights = inverse variance estimated from historical sensor noise), and emits a fused count with a `confidence_interval` field. This is particularly important for high-traffic entrances where undercounting/overcounting materially affects conversion rate calculations.

**Impact**: Improves KPI accuracy at the data collection layer; downstream metrics inherit the improvement.

---

## 15. Privacy-Preserving Data Export with Differential Privacy

`export_records()` currently returns raw data. For cross-retailer benchmarking or external analytics pipelines, add a `privacy_budget` parameter implementing the Laplace mechanism: noise ~ Laplace(sensitivity / epsilon) is added to all count and revenue fields before export. Include a `privacy_audit_log` entry recording the epsilon used and the sensitivity bounds applied. This enables participation in industry benchmarking consortia without exposing competitive data.

**Impact**: Unlocks external benchmarking partnerships that are currently blocked by data-sharing concerns.
