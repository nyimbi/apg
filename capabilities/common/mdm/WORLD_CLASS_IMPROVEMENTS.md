# MDM World-Class Improvements

15 targeted improvements for APG Master Data Management capability.

---

## 1. Probabilistic Deduplication via Blocking + Fellegi-Sunter

**Current state**: Simple `SequenceMatcher` on name + business key, 100-entity scan limit.

**Improvement**: Implement LSH (MinHash / SimHash) blocking to reduce comparison space by 10–100x, then apply Fellegi-Sunter scoring across configurable field comparators (Jaro-Winkler for names, Soundex/Metaphone phonetic for person names, exact match for keys, token-set ratio for addresses). Store match weight distributions per entity type. This gives calibrated m/u probabilities and a principled threshold, eliminating tuning by hand.

---

## 2. Async Batch Quality Assessment Pipeline

**Current state**: Quality is assessed synchronously one entity at a time; full scan is not supported.

**Improvement**: Add `bulk_assess_quality_async` that fans out quality coroutines with a semaphore-bounded concurrency pool (e.g., 32 concurrent coroutines), accumulates results, and streams progress events. Returns per-entity outcomes plus aggregate statistics. Enables scheduled nightly quality sweeps without blocking the event loop.

---

## 3. Golden Record Attribute Survivorship Engine

**Current state**: Survivorship policy is stored as a label (`most_recent`, `most_trusted`) but attribute-level resolution is not implemented.

**Improvement**: Implement `resolve_golden_attributes` that applies per-field survivorship strategies: `most_recent` (highest `updated_at`), `most_trusted` (source system trust ranking), `most_complete` (fewest nulls), `majority_vote` (mode across sources), `concatenate` (union for multi-value fields). Survivorship rules registered via `survivorship_rule()` override the record-level policy for specific fields.

---

## 4. Deterministic Entity Resolution Graph

**Current state**: Golden records hold a flat `source_entity_ids` list; no relationship graph exists.

**Improvement**: Add an in-memory entity resolution graph (adjacency list keyed by `golden_record_id`) that tracks edges with edge type (`member`, `suspect_duplicate`, `split_from`), confidence, and timestamp. Expose `entity_graph()` returning a networkx-compatible adjacency dict. Enables transitive cluster analysis and visualization without a graph database dependency.

---

## 5. Data Stewardship SLA Tracker

**Current state**: Pending review records exist but no SLA monitoring is implemented.

**Improvement**: Add `stewardship_sla_report` that computes age (hours since `created_at`) for every record in `pending_review` or `review_required` status, classifies items as within-SLA / warning / breached based on configurable thresholds, and returns a ranked list by urgency. Feeds stewardship-queue dashboard panels and alert integrations.

---

## 6. Incremental Lineage Graph with Impact Analysis

**Current state**: `data_lineage()` returns cross-references for a single entity with no transitive traversal.

**Improvement**: Add `lineage_impact_analysis` that traverses the golden-record graph and cross-reference chains transitively (BFS, configurable max-depth), returns the full upstream-source and downstream-consumer subgraph, and annotates nodes with entity type and status. Critical for change-impact assessment before entity retirement or attribute rename.

---

## 7. Composite Business-Key Normalization

**Current state**: Business key is stored as-is; no normalization or collision detection.

**Improvement**: Add `normalize_business_key` that applies entity-type-specific normalization (uppercase trim for product SKUs, E.164 for phone-keyed entities, domain-stripped lowercase for email-keyed entities, standardized country-code prefix for tax IDs) and `detect_key_collision` that checks the normalized form against existing entities before registration. Prevents silent duplicates from case/format variants.

---

## 8. Configurable Quality Threshold Enforcement

**Current state**: Quality gates in `publish_entity` check presence of a quality assessment but do not enforce minimum score per channel or entity type.

**Improvement**: Add `configure_quality_gate` that stores per-(entity_type, channel) minimum scores and per-dimension floor values. `publish_entity` consults the gate configuration and returns a structured `gate_failures` list when thresholds are not met, rather than a boolean pass/fail. Enables different quality bars for internal vs. external-facing channels.

---

## 9. Audit Event Replay and Projection

**Current state**: Audit events are appended to a list but cannot be replayed to reconstruct entity state at a point in time.

**Improvement**: Add `project_entity_state_at` that replays audit events in chronological order up to a given timestamp, reconstructing the entity's attribute snapshot. Enables point-in-time recovery, forensic investigation, and regulatory lookback without separate version tables.

---

## 10. Cross-Tenant Entity Federation

**Current state**: All operations are strictly tenant-scoped with no cross-tenant visibility.

**Improvement**: Add `federated_entity_search` for authorized cross-tenant queries that returns attribute-masked results (only non-restricted fields), with per-tenant consent records audited. Enables enterprise hub-and-spoke MDM where a central tenant governs shared reference data (country codes, product categories) consumed by child tenants.

---

## 11. Attribute Change Propagation to Downstream Golden Records

**Current state**: Updating a source entity does not propagate changes to its parent golden record.

**Improvement**: Add `propagate_attribute_changes` that, on source entity update, re-runs survivorship resolution for the parent golden record's affected fields and emits a `golden_record.attribute_updated` audit event. Keeps golden records continuously fresh without requiring full re-merge.

---

## 12. Consent and Purpose Limitation Enforcement

**Current state**: Data classification (`restricted`, `confidential`) gates registration but no purpose-limitation checks exist at access time.

**Improvement**: Add `check_access_purpose` that evaluates whether an accessor's declared purpose (e.g., `marketing`, `fraud_detection`, `regulatory_audit`) is permitted for the entity's classification and owner-defined purpose list. Returns an allow/deny/require_review decision integrated into the existing capability-contract pattern. Enables GDPR Article 5(1)(b) enforcement at the data layer.

---

## 13. Streaming Quality Degradation Alerts

**Current state**: Quality is assessed on demand; no monitoring of quality drift over time.

**Improvement**: Add `quality_trend_analysis` that computes per-dimension score deltas across the last N assessments for an entity or entity type cohort, flags entities where any dimension has degraded by more than a configurable threshold in the rolling window, and returns a prioritized alert list. Feeds operational alerting and SLA dashboards.

---

## 14. Hierarchical Entity Relationships

**Current state**: Entities are flat; no parent-child or part-of relationships exist.

**Improvement**: Add `register_entity_relationship` that records typed relationships (`parent_of`, `part_of`, `affiliated_with`, `supersedes`) between entities with evidence and actor, and `get_entity_hierarchy` that returns the relationship tree rooted at a given entity. Enables product hierarchy, organizational reporting structure, and account-contact relationships within the MDM layer.

---

## 15. Probabilistic Completeness Profiling

**Current state**: Completeness is binary per required field; no profile of the attribute population across the full entity set.

**Improvement**: Add `profile_entity_completeness` that scans all entities of a given type and returns per-attribute population rates (% non-null, % non-empty string, % conforming to type), a recommended `required_fields` list based on high-population attributes (> configurable threshold), and a list of sparse attributes (< threshold) that may indicate data model drift. Drives quality rule tuning and data model governance.
