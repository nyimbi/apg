# World-Class Improvements: Intelligence Dashboard

**Capability**: `intel_dashboard` | **Domain**: `intel`

---

## 1. Real-Time Streaming Threat Feed (Bytewax/Kafka Integration)

Replace the polling-based `intelligence_feed` with a push-based stream consumer. Attach a Bytewax dataflow to a Kafka topic (`apg.intel.dashboard.metrics`) so analysts receive confidence-ranked alerts within 500 ms of ingest. This eliminates the latency gap between raw collection and analyst visibility.

## 2. Confidence Score Time-Series Tracking

Store metric confidence scores with timestamps in a ring-buffer (tenant-scoped). Expose `confidence_trend(metric_id, window="7d")` so analysts can distinguish a degrading source from a stable high-confidence signal. Prevents alert fatigue from stale high-confidence metrics.

## 3. Cross-Domain Correlation Engine

Add `cross_domain_correlate(domains: list[str])` that applies Jaccard similarity across metric references in distinct domains. Surfacing latent connections between, e.g., cyber and physical threat domains multiplies analyst leverage and catches composite threats that per-domain views miss.

## 4. Automated Classification Downgrade/Upgrade Workflow

Implement `reclassify_dashboard(dashboard_id, new_classification, approver_id, justification)` with full audit trail and dual-approval policy gate. Current classification is immutable after creation, forcing cloning workarounds that fragment audit lineage.

## 5. Analyst Collaboration Threads (Persistent Notes)

Extend `collaboration_note` with thread semantics: notes reference a `parent_note_id`, enabling nested discussion. Back by a `DashboardNote` model stored in the in-memory or PostgreSQL store. Essential for shift-handover intelligence continuity.

## 6. ML-Powered Anomaly Detection on Metric Confidence

Add `detect_confidence_anomalies(threshold_sigma=2.0)` that computes per-source confidence Z-scores and flags sources whose recent scores deviate beyond `threshold_sigma`. Hooks into OLLAMA for a brief natural-language explanation of each anomaly.

## 7. Dashboard Health Score (Composite KPI)

Implement `dashboard_health_score(dashboard_id)` returning a 0–100 composite score weighted across: source reliability index, widget coverage, open review age, and classification appropriateness. Gives managers a single KPI instead of requiring them to synthesise multiple sub-reports.

## 8. Role-Based View Rendering Pipeline

Add `render_view_for_role(dashboard_id, role)` that filters widget and metric content to only what `role` is authorised to see per the classification hierarchy. Currently all views expose the same data irrespective of the requesting role, creating classification-bleed risk.

## 9. Scheduled Executive Briefing Delivery

Add `schedule_briefing(classification, cron_expr, recipient_ids, channel)` that registers a cron job (via APG CronCreate) to auto-deliver `management_briefing_pack` to listed recipients. Removes manual pull friction for C-suite consumers who need push delivery.

## 10. Dashboard Version History (Snapshot/Rollback)

Implement `snapshot_dashboard(dashboard_id)` and `rollback_dashboard(dashboard_id, snapshot_id)` using an append-only snapshot store. Analysts frequently need to restore a prior widget layout after accidental reconfiguration; current architecture has no rollback path.

## 11. Geo-Spatial Threat Heatmap Feed

Add `geospatial_threat_feed(bounding_box: dict)` that returns metrics whose references encode geo-coordinates (ISO 6709 or WGS-84) within the bounding box. Feeds map widgets in the UI layer with structured GeoJSON so front-end rendering requires no server-side parsing logic.

## 12. Federated Multi-Tenant Aggregation

Add `federated_summary(peer_tenant_ids: list[str])` that aggregates threat levels and gap counts across authorised peer tenants (with explicit cross-tenant authority record). Enables command-level situational awareness without merging tenant stores.

## 13. Data Provenance Graph (Full Lineage)

Extend `link_diagram` into a full provenance graph: trace evidence_reference chains from widget → metric → source → authority. Export as a PROV-DM-compatible JSON-LD document. Supports audit under intelligence oversight frameworks (e.g., ICD 503).

## 14. Predictive Metric Staleness Alerts

Add `stale_metric_detector(max_age_hours=24)` that compares metric `evidence_reference` timestamps against the current clock. Metrics with no update within `max_age_hours` are flagged as `stale` in the intelligence feed, preventing analysts from acting on outdated signals.

## 15. Fine-Grained Permission Delegation (Scoped Tokens)

Replace the coarse `policy_attached: bool` guard with a scoped delegation token model: `issue_delegation_token(grantor_id, grantee_id, scope, ttl_seconds)` and `revoke_delegation_token(token_id)`. Tokens are validated in `_enforce`, enabling time-bounded, least-privilege access without modifying base authority records.
