# World-Class Improvements: Intel Monitoring Capability

**Capability**: `intel_monitoring` | **Domain**: `intel` | **Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Adaptive Baseline Engine

**Current state**: Thresholds are static scalars updated via `threshold_adapt`. There is no concept of a learned normal baseline per watch.

**Improvement**: Maintain per-watch rolling statistics (mean, stddev, p95, p99) computed from event confidence scores and event volumes over configurable windows. Expose `async update_watch_baseline(watch_id, window)` that recomputes the baseline and persists it. Alert generation then compares incoming confidence against `mean + k*stddev` rather than a fixed threshold. This eliminates tuning toil and reduces false-positive drift.

---

## 2. Multi-Tenant Watch Namespace Isolation

**Current state**: Watches are keyed `(tenant_id, watch_id)`, but `start_monitor` constructs watch IDs as `watch_{target_type}_{target_id}`, making ID collisions possible across tenants with identical targets.

**Improvement**: Prefix auto-generated IDs with a tenant-derived shard token (hash of tenant_id mod 64). Add a `watch_exists(watch_id)` async query that resolves within the caller's tenant namespace. Apply the same convention to auto-generated incident, correlation, and suppression IDs.

---

## 3. Streaming Event Ingestion Pipeline

**Current state**: `monitor_alert` ingests one event at a time. `validate_batch` validates but does not actually process.

**Improvement**: Add `async stream_events(event_stream: AsyncIterator[dict])` that consumes an async generator and calls `record_event` for each item, yielding `(index, result_or_error)`. Integrate with Bytewax via a compatible `EventSource` adapter in `domain/adapters.py`. This decouples the HTTP ingestion layer from batch processing semantics.

---

## 4. Keyword Watch Versioning

**Current state**: `watch_expression` is set once. Updating a watch requires deleting and recreating it.

**Improvement**: Add `async update_watch_expression(watch_id, new_expression, change_reason)` that appends a version record to a `watch_history` dict, preserving the previous expression and the analyst who changed it. This enables audit-grade keyword list management and rollback.

---

## 5. Alert Deduplication Registry

**Current state**: `event_fingerprint` is stored but never used to deduplicate inbound alerts.

**Improvement**: Maintain a per-watch fingerprint cache (bounded LRU, capacity configurable). `monitor_alert` checks the cache before writing; duplicate fingerprints within a configurable TTL window return a `{deduplicated: true}` response without writing. Report deduplication rate via `monitor_health_check`.

---

## 6. Structured Suppression with Reinstatement

**Current state**: `alert_suppress` records a suppression dict but never enforces it — `monitor_alert` ignores suppression state.

**Improvement**: Store suppressions in `self._suppressions: dict[str, dict]` keyed by `monitor_id`. `monitor_alert` checks suppression expiry before ingesting. Add `async unsuppress_monitor(monitor_id)` for early reinstatement. `monitor_health_check` reports active suppression count.

---

## 7. Signal Enrichment Pipeline

**Current state**: Signals are pure analytical records; no enrichment runs against external intelligence feeds.

**Improvement**: Add `async enrich_signal(signal_id, enrichment_sources: list[str])` that calls registered enrichment adapters (e.g., geo-IP, WHOIS, MITRE ATT&CK technique mapping) in parallel via `asyncio.gather`. Enrichment results are stored on a per-signal `enrichments` dict accessible via `get_signal_enrichments(signal_id)`.

---

## 8. Cross-Tenant Federated Watch Sharing

**Current state**: Watches are strictly tenant-scoped with no sharing mechanism.

**Improvement**: Add `async publish_watch_template(watch_id, visibility: str)` and `async import_watch_template(template_id, target_tenant_id)`. Published templates are stored in a `self._watch_templates` registry. Importing clones the expression and policy linkage into the target tenant, requiring explicit authority validation. This enables org-wide threat intelligence propagation.

---

## 9. Watchlist-Driven Entity Monitoring

**Current state**: Keyword watches operate on free-text expressions. There is no first-class concept of a monitored entity (person, org, IP, domain).

**Improvement**: Add `async add_to_watchlist(entity_type, entity_id, keywords, risk_tier)` and `async remove_from_watchlist(entity_id)`. Watchlist entries map to one or more underlying `MonitoringWatch` records. A `watchlist_report()` aggregates hit counts and last-seen timestamps per entity, enabling entity-centric rather than expression-centric monitoring.

---

## 10. Incident Playbook Integration

**Current state**: Incidents are recorded but there is no structured response workflow beyond referral and dissemination.

**Improvement**: Add `async attach_playbook(incident_id, playbook_id, playbook_steps: list[str])` and `async advance_playbook_step(incident_id, step_index, outcome)`. Playbook state is tracked per incident, with each step requiring an outcome (completed/skipped/blocked). `incident_timeline` includes playbook progress. This turns the monitoring capability into an end-to-end response orchestrator.

---

## 11. Confidence Score Calibration

**Current state**: Confidence scores are stored as raw analyst-supplied floats with no calibration or validity tracking.

**Improvement**: Add `async calibrate_confidence(signal_ids: list[str], ground_truth: list[bool])` that computes Brier score and ECE (Expected Calibration Error) against analyst-labelled outcomes. Returns calibration curve data. Expose `async recalibrate_signal(signal_id, calibration_factor)` to apply isotonic-regression-derived adjustments. Improves ROI of downstream ML triage.

---

## 12. Retention Policy Enforcement

**Current state**: `retention_class` is stored on watches but no purge logic exists.

**Improvement**: Add `async enforce_retention(dry_run: bool = True)` that identifies events and signals older than their watch's retention class TTL (e.g., `ephemeral`=7d, `standard`=90d, `long_term`=365d). Returns counts and optionally purges records. Include audit trail entries for every purge. This is mandatory for GDPR/POPIA compliance.

---

## 13. Real-Time Severity Heatmap

**Current state**: `monitor_analytics` returns aggregated severity distribution but no time-series data.

**Improvement**: Add `async severity_heatmap(granularity: str = "1h", periods: int = 24)` that bins events/signals by UTC hour bucket and severity. Returns a matrix `[period][severity] -> count` suitable for direct frontend consumption. Enables rapid visual identification of attack windows without full analytics pipeline.

---

## 14. Composite Health Score

**Current state**: `monitor_health_check` returns raw counts; operators must mentally synthesize health from multiple figures.

**Improvement**: Add a `health_score: float` field (0–100) computed from a weighted formula: stale_watch_penalty, signal_to_event_ratio normalised against target range, SLA breach rate, and false-positive rate. Thresholds map score to `health_status: str` ("healthy" / "degraded" / "critical"). Dashboard can render a single RAG indicator.

---

## 15. Exportable Audit Ledger with Tamper Evidence

**Current state**: `self.audit_events` is a plain list with no integrity protection.

**Improvement**: Add `async seal_audit_ledger(period_end: str)` that hashes the ordered audit events over a period using SHA-256 chaining (each entry includes the hash of the previous), returns a `ledger_root` hash, and stores the sealed record in `self._sealed_ledgers`. `async verify_audit_ledger(ledger_root)` re-derives the hash chain and returns a `{valid: bool, entry_count: int}` result. Required for regulatory admissibility of monitoring records.
