# World-Class Improvements — Metadata Management (meta)

Author: Nyimbi Odero
Copyright: © 2025 Datacraft

---

## Improvement 1: Async-First Service Layer

**Current state:** `MetaService` methods are synchronous; bulk operations block the event loop.
**Target:** Async variants for every mutation method with `asyncio.gather`-safe fan-out, allowing
concurrent asset registration, batch classification, and lineage ingestion without thread overhead.

**Implementation sketch:**
- Add `async def async_register_asset(...)` wrappers that off-load validation to a thread pool
  only when regex/rule evaluation is CPU-bound.
- Add `async def batch_register_assets(assets: list[dict]) -> list[MetaAssetRecord]` using
  `asyncio.gather` internally.
- Retains the sync API surface unchanged so existing generated apps are not broken.

---

## Improvement 2: Schema Registry with Version History

**Current state:** Assets store a flat `metadata` dict; schema evolution is untracked.
**Target:** A first-class `MetaSchemaVersion` record per asset with Avro/JSON Schema/Protobuf
payload, compatibility mode (BACKWARD / FORWARD / FULL), and fingerprint-based deduplication.

**Implementation sketch:**
- `register_schema_version(asset_id, schema_json, format, compatibility_mode)` stores a versioned
  record and blocks incompatible promotions unless override evidence is present.
- `get_schema_evolution(asset_id)` returns the full version history with diffs and compatibility
  verdicts so downstream consumers can detect breaking changes before they materialise.

---

## Improvement 3: Column-Level Lineage Capture

**Current state:** Lineage is asset-to-asset only.
**Target:** Column-level lineage edges with transformation SQL/expression snippets, enabling
field-level impact analysis and GDPR right-to-erasure scope calculations.

**Implementation sketch:**
- `MetaColumnLineageRecord` with `source_column`, `target_column`, `transformation_expr`, and
  `lineage_type` fields.
- `capture_column_lineage(...)` validates that parent asset lineage exists before accepting column
  edges, enforcing structural integrity.
- `get_column_impact(asset_id, column_name)` traverses the column graph downstream.

---

## Improvement 4: Pluggable Policy Engine

**Current state:** `evaluate_capability_rules` is a fixed function; adding new guardrails requires
editing `capability_contract.py`.
**Target:** A rule registry where policies are registered as Python callables or YAML declarations,
evaluated in priority order, short-circuiting on first `deny`.

**Implementation sketch:**
- `PolicyRegistry.register(rule_id, fn, priority)` with a `@policy_rule` decorator.
- Policies receive a typed `PolicyContext` dataclass rather than a raw dict, eliminating string-key
  typos.
- Dry-run mode returns matched rules and projected decision without side effects, enabling
  pre-flight checks from UI wizards.

---

## Improvement 5: Event-Sourced Audit Log

**Current state:** `audit_events` is an in-memory list; events are append-only but ephemeral.
**Target:** A CloudEvents-compatible append-only audit log persisted to PostgreSQL via insert-only
table, with projection support for point-in-time state reconstruction.

**Implementation sketch:**
- Audit events carry `specversion`, `type`, `source`, `subject`, `datacontenttype`, and `data`
  following the CloudEvents 1.0 spec.
- A `replay_state(asset_id, at: datetime)` method rebuilds the asset record from the audit log,
  enabling historical views and regulatory time-travel queries.

---

## Improvement 6: Data Contract Validation

**Current state:** No first-class concept of inter-team data contracts.
**Target:** `MetaDataContractRecord` capturing SLO terms (freshness SLA, row-count bounds, schema
commitments) with automated violation detection on each quality assessment.

**Implementation sketch:**
- `register_data_contract(producer, consumer, asset_id, slo_terms)` creates a versioned contract.
- `evaluate_contract(asset_id)` compares the latest quality assessment and schema version against
  contracted SLOs and produces a `ContractViolationEvent` when breached.
- Violations are surfaced as `pending_review` records and trigger the existing audit pipeline.

---

## Improvement 7: Metadata Quality Dimensions Engine

**Current state:** `assess_quality` accepts a free-form `dimensions` dict; no standard dimension
definitions or scoring algebra.
**Target:** A typed `QualityDimensions` model (completeness, accuracy, consistency, validity,
uniqueness, timeliness, lineage_coverage, classification_coverage) with weighted composite scoring
and per-dimension trend tracking.

**Implementation sketch:**
- `QualityDimensions` Pydantic model with `AfterValidator` enforcing 0–100 range per dimension.
- `compute_composite_score(dimensions, weights)` returns a reproducible composite; default weights
  match ISO/IEC 25012.
- `trend_quality(asset_id, window_days)` returns per-dimension moving averages so degradation is
  caught before certification gates fail.

---

## Improvement 8: Semantic Business Glossary with Term Relationships

**Current state:** Glossary terms are flat records with asset links.
**Target:** A term graph supporting `IS_A`, `RELATED_TO`, `SYNONYM_OF`, `ABBREVIATION_OF`, and
`DEPRECATED_BY` edges, enabling semantic search expansion and cross-domain term reconciliation.

**Implementation sketch:**
- `link_glossary_terms(source_term_id, target_term_id, relationship_type)` validates that both
  terms belong to the same tenant before creating the edge.
- `expand_term(term_id)` returns the transitive closure of synonyms and related terms, powering
  the search engine's semantic expansion.

---

## Improvement 9: Sensitivity-Aware Data Masking Profiles

**Current state:** Sensitivity labels are stored but not linked to masking strategies.
**Target:** `MetaMaskingProfileRecord` attaching masking strategies (NULLIFY, TOKENIZE, HASH, MASK,
GENERALIZE, SUPPRESS) to classification labels, propagated automatically to downstream copies
detected via lineage.

**Implementation sketch:**
- `register_masking_profile(classification_label, strategy, parameters)` stores the profile and
  links it to the classification record.
- `get_masking_profile_for_asset(asset_id)` resolves the effective masking strategy by walking the
  classification chain.

---

## Improvement 10: Federated Catalog Search with Cross-Tenant Discovery

**Current state:** Search is single-tenant; federated catalog access across tenant boundaries is
not modelled.
**Target:** Federated search queries multiple tenant catalogs simultaneously, returning a merged
result set with provenance metadata, subject to inter-tenant sharing agreements.

**Implementation sketch:**
- `MetaSharingAgreementRecord` capturing which tenant pairs may share which asset types at what
  sensitivity ceiling.
- `federated_search(query, requesting_tenant, target_tenants)` issues parallel searches and filters
  results through the sharing agreement registry before returning the union.

---

## Improvement 11: Automated Freshness Monitoring

**Current state:** `age_days` is set at registration and never updated; freshness checks are
manual.
**Target:** A background freshness monitor that re-evaluates asset age against SLA thresholds,
downgrades quality scores for stale assets, and emits `freshness.violated` audit events.

**Implementation sketch:**
- `register_freshness_sla(asset_id, max_age_hours, severity)` stores the SLA.
- `async def run_freshness_sweep(tenant_id)` iterates assets, computes actual age from the last
  known data timestamp, and fires audit events for violations.
- Violated assets are automatically moved to `pending_review` status until evidence of refresh is
  recorded.

---

## Improvement 12: Observability Middleware with OpenTelemetry Traces

**Current state:** Logging is done via print statements; no distributed tracing.
**Target:** OpenTelemetry spans wrapping every public method in both `MetaService` and
`APGMetadataService`, exporting to OTLP-compatible backends (Jaeger, Grafana Tempo).

**Implementation sketch:**
- A `@traced_operation(span_name)` decorator that creates a span, records tenant_id and asset_id
  as attributes, and sets error status on exceptions.
- `MetricsRecorder` tracking request latency histograms, error counters, and asset operation
  counters as OTLP gauge/counter metrics.
- Zero overhead when `OTEL_EXPORTER_OTLP_ENDPOINT` is not set (no-op tracer).

---

## Improvement 13: Replication-Aware Duplicate Detection

**Current state:** Assets are keyed by `tenant_id:asset_id`; duplicate registration raises no
warning if the `asset_id` differs but the `business_key` matches.
**Target:** Duplicate detection on `(tenant_id, business_key, source_system)` with configurable
merge strategies (FAIL, MERGE_LATEST, DEDUPLICATE_LINEAGE).

**Implementation sketch:**
- Pre-registration check queries the business key index; on collision, applies the configured
  merge strategy and records a `duplicate_detected` audit event.
- `MERGE_LATEST` updates mutable fields on the existing record and links the new registration as a
  provenance event without creating a separate asset record.

---

## Improvement 14: Compliance Report Generator

**Current state:** Compliance evidence lives in individual records; no aggregated report surface.
**Target:** `generate_compliance_report(tenant_id, regulation, as_of)` produces a structured
report mapping GDPR Articles 13–30, HIPAA § 164.514, or PCI DSS Requirement 3 controls to asset
evidence, certification status, and lineage coverage.

**Implementation sketch:**
- A `ComplianceMapper` registry keyed by regulation code, each mapping control IDs to evidence
  predicates evaluated against asset, classification, lineage, and quality records.
- Reports are returned as structured dicts and optionally persisted as `MetaComplianceReportRecord`
  for audit trail purposes.

---

## Improvement 15: LLM-Assisted Metadata Enrichment

**Current state:** AI classification is limited to column-level sensitivity labelling.
**Target:** `enrich_asset_metadata(asset_id, enrichment_type)` calls a locally hosted Ollama model
to generate asset descriptions, suggest business glossary links, propose tags, and draft data
quality rules from schema and sample data.

**Implementation sketch:**
- Enrichment is gated behind `contribution_disclosed=True` and a `catalog_agent` registration so
  every LLM-generated suggestion carries provenance, confidence, and a human review gate before
  the enrichment is applied to the asset record.
- Supports enrichment types: `description`, `tags`, `glossary_links`, `quality_rules`,
  `sensitivity_hints`.
- Ollama endpoint is configured via `OLLAMA_BASE_URL` environment variable; enrichment is skipped
  gracefully when unavailable.
