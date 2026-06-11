# Open Source Intelligence — World-Class Improvements

**Capability**: `intel_osint` | **Domain**: `intel` | **Author**: Nyimbi Odero  
**Date**: 2026-06-11 | **Copyright**: © 2025 Datacraft

---

## 1. Dark Web Crawl Pipeline

**Current gap**: `TaskType.DARK_WEB_CRAWL` exists as an enum value but there is no service method to record, validate, or process dark web crawl results.

**Improvement**: Add `async def dark_web_crawl(...)` that accepts Tor-circuit metadata, onion URL, content hash, and a mandatory high-risk approval reference. Enforces `RiskTier.CRITICAL` governance rules and emits `osint_dark_web_content_captured` event. Integrates with the existing `WebContentResponse` model augmented with `is_onion=True` flag.

---

## 2. Keyword-Driven Alert Subscription

**Current gap**: Social media monitoring and web scraping collect data but there is no mechanism for tenants to subscribe to keyword-driven alerts across all collection channels simultaneously.

**Improvement**: Add `async def subscribe_keyword_alert(...)` supporting multi-channel keyword subscriptions (web, social, news, paste sites, dark web) with per-channel rate limits, TLP markings, and auto-expiry. Returns a subscription record that feeds into a rule engine triggering `osint_keyword_alert_fired` events.

---

## 3. Cross-Source Pivot Search

**Current gap**: Domain records, IP intelligence, social profiles, and entities are stored in separate collections with no unified search across them.

**Improvement**: Add `async def pivot_search(query: str, pivot_type: str, ...)` that searches across all entity stores simultaneously using `asyncio.gather()`, ranks results by confidence score weighted by source credibility baseline, and returns a unified `PivotSearchResult` with provenance trail.

---

## 4. Intelligence Requirement Lifecycle (Async)

**Current gap**: `register_requirement` is synchronous-only. Requirements lack status tracking, priority escalation, and linkage to downstream processed intel.

**Improvement**: Add full async `create_requirement`, `update_requirement`, `close_requirement`, `list_requirements` methods backed by `IntelRequirementCreate/Response` Pydantic models. Wire requirement IDs to `ProcessedIntelligenceCreate.requirement_id` FK validation.

---

## 5. Confidence Decay Engine

**Current gap**: Intelligence items never age. A `confidence_score=0.95` item from 18 months ago is treated identically to one from yesterday.

**Improvement**: Add `async def apply_confidence_decay(decay_model: str = "exponential", half_life_days: int = 90)` that recomputes confidence scores for all raw and processed intel based on `captured_at` / `created_at` timestamps. Configurable decay models: `exponential`, `linear`, `step`. Emits `osint_confidence_decayed` events per updated item.

---

## 6. Bulk Ingestion with Backpressure

**Current gap**: `ingest_raw_intel` processes one item at a time. Large collection tasks (thousands of items) make synchronous sequential calls impractical.

**Improvement**: Add `async def bulk_ingest_raw_intel(payloads: list[RawIntelligenceCreate], max_concurrency: int = 50)` using an `asyncio.Semaphore` to bound parallelism, collecting per-item results and errors without aborting the batch. Returns `BulkIngestResult` with `succeeded`, `failed`, `duplicate_skipped` counts.

---

## 7. Entity Merge (Human-Confirmed)

**Current gap**: `duplicate_deduplication` identifies merge candidates but does not actually merge entity records in the service store. It only returns a report.

**Improvement**: Add `async def merge_entities(primary_id: str, secondary_ids: list[str], analyst_id: str, evidence_reference: str)` that consolidates aliases, relationships, source intel IDs, and tags into the primary entity, soft-deletes secondaries, and emits `osint_entities_merged` events. Requires explicit analyst sign-off to prevent silent data loss.

---

## 8. Threat Actor Profiling

**Current gap**: No higher-order abstraction groups entities, relationships, TTPs, and historical activity into a coherent threat actor profile.

**Improvement**: Add `async def build_threat_actor_profile(entity_id: str, lookback_days: int = 180)` that aggregates all relationships, associated IP intel, domain records, social profiles, and processed intel linked to the entity. Returns `ThreatActorProfile` with timeline, geographic footprint, communication channels, and composite risk score.

---

## 9. News Feed Parser & Dedup Pipeline

**Current gap**: RSS/news ingestion is modelled (`SourceType.RSS_FEED`, `SourceType.NEWS`) but has no dedicated parsing path. Raw HTML arrives in `WebContent` without structured article fields.

**Improvement**: Add `async def parse_news_feed(source_id: str, feed_url: str, max_items: int = 100)` that normalises RSS/Atom feed entries into `RawIntelligenceCreate` payloads with `fingerprint` computed from the article GUID + publication date. Returns parse summary with duplicate-skip count.

---

## 10. STIX 2.1 Export

**Current gap**: Intelligence outputs are tenant-private dicts. There is no standard interchange format for sharing with external SOC/CTI platforms.

**Improvement**: Add `async def export_stix_bundle(processed_intel_ids: list[str], include_entities: bool = True)` that serialises selected `ProcessedIntelligenceResponse` items, associated `OSEntityResponse` records, and `EntityRelationshipResponse` edges into a STIX 2.1 Bundle JSON document. Classification and TLP markings map to STIX `MarkingDefinition` objects.

---

## 11. Source Credibility Auto-Calibration

**Current gap**: `credibility_baseline` is set manually at registration and never updated automatically based on observed accuracy.

**Improvement**: Add `async def recalibrate_source_credibility(source_id: str)` that computes a new credibility score from the historical ratio of `TriageDecision.RELEVANT` vs `TriageDecision.IRRELEVANT` for raw intel from that source, weighted by analyst review outcomes. Updates `credibility_baseline` and creates a `CredibilityScoreResponse` audit record.

---

## 12. Graph Centrality Analytics

**Current gap**: `relationship_mapping()` calls `calculate_entity_centrality` from domain calculations but the result is not surfaced in the `EntityNetworkReport`.

**Improvement**: Add `async def entity_centrality_report(min_connections: int = 2)` that computes degree, betweenness, and eigenvector centrality for all entities, returns sorted `EntityCentralityReport` with top-N high-value nodes, and flags entities above configurable centrality thresholds as `watchlist_candidates`.

---

## 13. Temporal Pattern Detection

**Current gap**: Collection tasks and raw intel items have timestamps but no service-level analysis of temporal patterns (activity bursts, dormancy windows, coordinated campaigns).

**Improvement**: Add `async def detect_temporal_patterns(entity_id: str | None = None, window_hours: int = 24, min_burst_size: int = 5)` that buckets events into time windows, identifies burst periods, cross-correlates entity activity, and returns a `TemporalPatternReport` with detected anomalies and statistical confidence.

---

## 14. Collection Task Retry with Exponential Back-off

**Current gap**: `fail_task` marks a task failed and stops. There is no automated retry mechanism, so transient network failures permanently kill collection tasks.

**Improvement**: Add `async def retry_task(task_id: str, max_retries: int = 3, backoff_base_seconds: float = 2.0)` that resets status to `PENDING`, increments a retry counter, computes `scheduled_at = now + backoff_base ** retry_count`, and emits `osint_task_retry_scheduled`. The `CollectionTaskResponse` model gains `retry_count: int = 0` and `max_retries: int = 3` fields.

---

## 15. Watchlist Management

**Current gap**: No service-level watchlist concept exists. Analysts cannot formally flag entities, IPs, or domains for elevated monitoring without creating ad-hoc tags.

**Improvement**: Add `async def add_to_watchlist(reference_id: str, reference_type: str, reason: str, alert_threshold: float = 0.7, analyst_id: str, evidence_reference: str)` and `async def remove_from_watchlist(watchlist_id: str, analyst_id: str)` backed by a `WatchlistEntry` model. Watchlist entries are evaluated on every new raw intel ingest; matches trigger `osint_watchlist_hit` events and auto-escalate triage to `TriageDecision.ESCALATED`.
