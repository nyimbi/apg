# APG Metadata Management Capability

`common/meta` provides the metadata catalog and governance layer for APG
applications. It lets generated applications register metadata assets, schedule
approved discovery, classify sensitive assets, capture lineage, assess metadata
quality, certify governed assets, manage glossary terms, evaluate publication
and retirement decisions, and preserve audit evidence.

The capability has two runtime surfaces:

- `MetaService`: a dependency-light control plane for generated applications,
  tests, local composition, UI models, and guardrail decisions.
- `APGMetadataService`: the production runtime that orchestrates database,
  discovery, AI classification, lineage, search, and APG integration adapters.

Generated APG applications should use `MetaService` first. Production adapters
can attach richer discovery, classification, lineage, search, persistence, and
Bytewax event streams behind the same contract.

## What It Provides

- Tenant-scoped asset catalog for databases, schemas, tables, columns, files,
  APIs, streams, reports, dashboards, models, pipelines, and glossary terms.
- Approved discovery scheduling for databases, files, APIs, streams, ML
  systems, and external catalogs.
- Dependency-light metadata fixture discovery for Oracle, SQL Server, Redis,
  and BigQuery so generated applications can compose catalog screens and
  governance flows before live vendor drivers are installed.
- Classification evidence with confidence, sensitivity labels, and steward
  review.
- Lineage capture between registered source and target assets.
- Metadata quality assessment across completeness, freshness, accuracy,
  lineage, classification, and usage.
- Certification gates for governed assets.
- Business glossary ownership and asset links.
- Publication and retirement guardrails.
- First-class catalog-agent registration for Codex, Claude Code, opencode, Pi,
  and future APG-compatible runtimes.
- Catalog-agent guardrails for supported roles, declared scope, owner, purpose,
  machine-contribution disclosure, and human approval for privileged roles.
- Durable review evidence for review-required assets, discovery jobs,
  classifications, lineage edges, quality assessments, certifications, glossary
  terms, privileged catalog agents, denied lifecycle batches, and audit events.
- Bytewax lifecycle batch validation for asset, discovery, classification,
  lineage, quality, certification, glossary, and catalog-agent streams.
- Generated-application UI routes, view models, theme tokens, and adapter
  metadata.
- Async-first batch registration with `asyncio.gather`-safe fan-out (v2.0).
- Schema registry with version history, compatibility modes, and fingerprint
  deduplication (v2.0).
- Column-level lineage capture for GDPR right-to-erasure scope calculations
  (v2.0).
- Pluggable policy engine with priority-ordered rule evaluation (v2.0).
- CloudEvents-compatible append-only audit log with point-in-time replay (v2.0).
- Data contract validation with SLO terms and automated violation detection
  (v2.0).
- Typed quality dimensions engine with ISO/IEC 25012-aligned composite scoring
  (v2.0).
- Semantic business glossary with term relationship graph (v2.0).
- Sensitivity-aware masking profiles propagated via lineage (v2.0).
- Federated catalog search across tenant boundaries subject to sharing
  agreements (v2.0).
- Automated freshness monitoring with SLA-driven audit events (v2.0).
- OpenTelemetry observability middleware with zero overhead when OTLP endpoint
  is absent (v2.0).
- Duplicate detection on `(tenant_id, business_key, source_system)` with
  configurable merge strategies (v2.0).
- Compliance report generator for GDPR, HIPAA, and PCI DSS controls (v2.0).
- LLM-assisted metadata enrichment via locally hosted Ollama models (v2.0).

## Core Lifecycle

1. Register metadata assets with tenant, type, business key, source system,
   owner, steward, sensitivity, tags, and metadata.
2. Schedule discovery only with approved connectors and reviewed schedules.
3. Classify sensitive assets and route low-confidence results to stewardship.
4. Capture lineage between registered assets.
5. Assess metadata quality.
6. Request certification after quality and lineage evidence are present.
7. Publish assets only when owner, quality, classification, and steward gates
   pass.
8. Register catalog agents that can contribute to metadata governance and
   publish-gate workflows.
9. Validate lifecycle batches through Bytewax before publishing operational
   evidence.
10. Manage glossary terms with accountable owners.
11. Retire assets only after impact-analysis evidence exists.
12. Preserve audit events for every lifecycle decision.

## Durable Review Evidence

META preserves policy evidence directly on lifecycle records so generated
applications can build stewardship queues without replaying transient
exceptions. Reviewable records include assets, discovery jobs, classifications,
lineage edges, quality assessments, certifications, glossary terms, catalog
agents, lifecycle batches, and audit events.

Each governed record exposes:

- `policy_decision`
- `matched_rules`
- `review_reasons`
- `review_evidence`

Privileged catalog agents without human approval are registered as
`pending_review` records when their runtime, role, owner, scope, purpose, and
contribution disclosure are otherwise valid. Invalid runtimes, unsupported
roles, missing owner/scope/purpose, and missing contribution disclosure remain
blocking denials.

Denied non-Bytewax lifecycle batches are persisted as `denied` records before
`PermissionError` is raised, giving operators durable evidence for remediation.

## Quick Start

```python
from capabilities.common.meta.service import MetaService

service = MetaService()

asset = service.register_asset(
    tenant_id="tenant-a",
    asset_id="warehouse.customers",
    asset_type="table",
    name="customers",
    business_key="warehouse.public.customers",
    source_system="warehouse",
    owner="data-owner",
    steward="data-steward",
    sensitivity="restricted",
)

service.classify_asset(
    tenant_id="tenant-a",
    asset_id=asset.asset_id,
    label="pii",
    confidence=0.96,
    classification_complete=True,
    steward_review_recorded=True,
)

service.assess_quality(
    tenant_id="tenant-a",
    asset_id=asset.asset_id,
    score=91.0,
    dimensions={"completeness": 95.0, "freshness": 90.0},
    assessor="quality-engine",
)

published = service.publish_asset(
    tenant_id="tenant-a",
    asset_id=asset.asset_id,
)

assert published.status == "published"
```

Register a governed catalog-agent contributor:

```python
agent = service.register_catalog_agent(
    tenant_id="tenant-a",
    agent_id="classification-reviewer",
    name="Classification Reviewer",
    runtime="codex",
    role="classification_reviewer",
    scope="restricted metadata classification",
    owner="metadata-office",
    purpose="review sensitive classification evidence",
    human_approval_required=True,
)

batch = service.validate_meta_lifecycle_batch(
    tenant_id="tenant-a",
    event_stream="bytewax",
    mutation_count=8,
)

assert agent.runtime == "codex"
assert batch.status == "accepted"
```

## Core API

| Method | Returns | Purpose |
|---|---|---|
| `register_asset(...)` | `MetaAssetRecord` | Register a new metadata asset with guardrail evaluation |
| `schedule_discovery(...)` | `MetaDiscoveryJobRecord` | Schedule approved connector discovery |
| `record_discovery_result(...)` | `MetaDiscoveryJobRecord` | Record discovered asset IDs against a job |
| `classify_asset(...)` | `MetaClassificationRecord` | Classify sensitivity with confidence and steward routing |
| `review_classification(...)` | `MetaClassificationRecord` | Record steward review decision with notes |
| `capture_lineage(...)` | `MetaLineageRecord` | Capture asset-to-asset lineage edge |
| `assess_quality(...)` | `MetaQualityRecord` | Record quality score and dimension breakdown |
| `request_certification(...)` | `MetaCertificationRecord` | Gate certification on lineage, quality, and freshness |
| `publish_asset(...)` | `MetaAssetRecord` | Evaluate publication guardrails |
| `register_glossary_term(...)` | `MetaGlossaryTermRecord` | Register business term with owner |
| `retire_asset(...)` | `MetaAssetRecord` | Gate retirement on impact analysis |
| `register_catalog_agent(...)` | `MetaCatalogAgentRecord` | Register AI/agent contributor with role guardrails |
| `validate_meta_lifecycle_batch(...)` | `MetaLifecycleBatchRecord` | Enforce Bytewax-only lifecycle mutations |
| `list_records(...)` | `list[dict]` | Enumerate records by type for a tenant |
| `list_pending_reviews(...)` | `list[dict]` | Return all records awaiting steward review |
| `dashboard_summary(...)` | `dict` | Aggregate counts for dashboard rendering |
| `describe(...)` | `dict` | Return capability contract for a tenant |
| `create_record(...)` | `dict` | Low-level keyed record creation |

## World-Class Enhancements (v2.0)

These 15 improvements have been specified and are being implemented incrementally.
Each section links the design rationale to the relevant extension point in `service.py`.

1. **Async-First Service Layer** — `async def batch_register_assets(assets)` using
   `asyncio.gather` fan-out; CPU-bound rule evaluation off-loaded to thread pool.
   Sync API surface unchanged.

2. **Schema Registry with Version History** — `register_schema_version(asset_id, schema_json,
   format, compatibility_mode)` stores Avro/JSON Schema/Protobuf payloads with fingerprint
   deduplication. `get_schema_evolution(asset_id)` returns full version history with diffs and
   compatibility verdicts.

3. **Column-Level Lineage Capture** — `MetaColumnLineageRecord` with `source_column`,
   `target_column`, `transformation_expr`, and `lineage_type`. `capture_column_lineage(...)`
   enforces parent asset lineage before accepting column edges. `get_column_impact(asset_id,
   column_name)` traverses downstream column graph; essential for GDPR erasure scope
   calculations.

4. **Pluggable Policy Engine** — `PolicyRegistry.register(rule_id, fn, priority)` with
   `@policy_rule` decorator. Policies receive a typed `PolicyContext` dataclass. Dry-run mode
   returns matched rules and projected decision without side effects.

5. **Event-Sourced Audit Log** — CloudEvents 1.0-compatible append-only log persisted to
   PostgreSQL insert-only table. `replay_state(asset_id, at: datetime)` rebuilds asset record
   from audit log for regulatory time-travel queries.

6. **Data Contract Validation** — `MetaDataContractRecord` captures SLO terms (freshness SLA,
   row-count bounds, schema commitments). `evaluate_contract(asset_id)` detects violations on
   each quality assessment and emits `ContractViolationEvent`.

7. **Metadata Quality Dimensions Engine** — Typed `QualityDimensions` Pydantic model
   (completeness, accuracy, consistency, validity, uniqueness, timeliness, lineage_coverage,
   classification_coverage) with ISO/IEC 25012-aligned default weights. `trend_quality(asset_id,
   window_days)` returns per-dimension moving averages.

8. **Semantic Business Glossary with Term Relationships** — `link_glossary_terms(source_term_id,
   target_term_id, relationship_type)` supporting `IS_A`, `RELATED_TO`, `SYNONYM_OF`,
   `ABBREVIATION_OF`, and `DEPRECATED_BY` edges. `expand_term(term_id)` returns transitive
   synonym closure for search expansion.

9. **Sensitivity-Aware Data Masking Profiles** — `MetaMaskingProfileRecord` attaching strategies
   (NULLIFY, TOKENIZE, HASH, MASK, GENERALIZE, SUPPRESS) to classification labels.
   `get_masking_profile_for_asset(asset_id)` resolves effective strategy by walking the
   classification chain; propagates automatically to downstream lineage copies.

10. **Federated Catalog Search** — `MetaSharingAgreementRecord` governs inter-tenant sharing.
    `federated_search(query, requesting_tenant, target_tenants)` issues parallel searches and
    filters results through the sharing agreement registry.

11. **Automated Freshness Monitoring** — `register_freshness_sla(asset_id, max_age_hours,
    severity)` stores SLA. `async run_freshness_sweep(tenant_id)` downgrades quality scores for
    stale assets and emits `freshness.violated` audit events, moving violated assets to
    `pending_review`.

12. **OpenTelemetry Observability Middleware** — `@traced_operation(span_name)` decorator creates
    spans with tenant_id and asset_id attributes, exports to OTLP-compatible backends
    (Jaeger, Grafana Tempo). `MetricsRecorder` tracks latency histograms and error counters.
    Zero overhead when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset.

13. **Replication-Aware Duplicate Detection** — Pre-registration check on `(tenant_id,
    business_key, source_system)`. Configurable merge strategies: FAIL, MERGE_LATEST,
    DEDUPLICATE_LINEAGE. `MERGE_LATEST` updates mutable fields and links the new registration
    as a provenance event.

14. **Compliance Report Generator** — `generate_compliance_report(tenant_id, regulation, as_of)`
    maps GDPR Articles 13-30, HIPAA § 164.514, and PCI DSS Requirement 3 controls to asset
    evidence, certification status, and lineage coverage. `ComplianceMapper` registry is
    extensible by regulation code.

15. **LLM-Assisted Metadata Enrichment** — `enrich_asset_metadata(asset_id, enrichment_type)`
    calls a locally hosted Ollama model (configured via `OLLAMA_BASE_URL`) to generate
    descriptions, suggest glossary links, propose tags, and draft quality rules. Every
    LLM-generated suggestion requires `contribution_disclosed=True`, carries a `catalog_agent`
    provenance record, and passes a human review gate before being applied.

## New Methods

### Async batch registration

```python
import asyncio
from capabilities.common.meta.service import MetaService

service = MetaService()

# Register multiple assets concurrently — fan-out is asyncio.gather-safe
assets = await service.batch_register_assets([
    {"tenant_id": "tenant-a", "asset_id": "db.orders", "asset_type": "table",
     "name": "orders", "business_key": "warehouse.public.orders",
     "source_system": "warehouse", "owner": "ops-team"},
    {"tenant_id": "tenant-a", "asset_id": "db.customers", "asset_type": "table",
     "name": "customers", "business_key": "warehouse.public.customers",
     "source_system": "warehouse", "owner": "ops-team"},
])
```

### Schema version registration and evolution

```python
# Register a schema version with backward compatibility enforcement
version = service.register_schema_version(
    asset_id="db.orders",
    schema_json='{"type":"record","fields":[{"name":"id","type":"int"}]}',
    format="avro",
    compatibility_mode="BACKWARD",
)

# Retrieve full version history with diffs and compatibility verdicts
history = service.get_schema_evolution(asset_id="db.orders")
# [{"version": 1, "fingerprint": "...", "compatibility": "BACKWARD", "diff": [...]}]
```

### Column-level lineage capture

```python
# Capture a column-level derivation with transformation evidence
col_edge = service.capture_column_lineage(
    tenant_id="tenant-a",
    source_asset_id="db.orders",
    source_column="customer_email",
    target_asset_id="mart.order_summary",
    target_column="masked_email",
    transformation_expr="SHA256(customer_email)",
    lineage_type="derived",
)

# Downstream impact analysis at column level (e.g., GDPR erasure scope)
impact = service.get_column_impact(
    asset_id="db.orders",
    column_name="customer_email",
)
```

### Compliance report generation

```python
# Generate a GDPR compliance report as of a specific date
report = service.generate_compliance_report(
    tenant_id="tenant-a",
    regulation="GDPR",
    as_of="2026-01-01T00:00:00Z",
)
# report["controls"]["Article_30"]["status"] == "compliant"
# report["controls"]["Article_17"]["evidence"]["lineage_coverage"] == 0.94
```

### LLM-assisted metadata enrichment

```python
# Enrich asset description using locally hosted Ollama (no data leaves the cluster)
suggestion = service.enrich_asset_metadata(
    asset_id="db.customers",
    enrichment_type="description",  # or "tags", "glossary_links", "quality_rules", "sensitivity_hints"
)
# suggestion.status == "pending_review"   — human gate required before application
# suggestion.confidence == 0.87
# suggestion.provenance["catalog_agent_id"] == "ollama-enrichment-agent"
```

## Generated UI Surfaces

`capability_contract.py` and `view_models.py` expose:

- Dashboard
- Asset catalog
- Discovery console
- Lineage viewer
- Classification review
- Quality console
- Certification queue
- Business glossary
- Impact analysis
- Search
- Audit timeline
- Adapter health
- Catalog-agent roster
- Lifecycle batch monitor
- Settings

The packet does not require a particular web framework. Generated APG targets
can render these models in their own UI shells.

## Database Connector Fixtures

Generated applications can use metadata-backed database connectors when live
database drivers are unavailable. The Oracle, SQL Server, Redis, and BigQuery
connectors accept `additional_params["offline_catalog"]` assets and expose the
same `test_connection()`, `discover_assets()`, `get_asset_schema()`, and
`sample_asset_data()` methods as live connectors.

```python
from capabilities.common.meta.connectors.base_connector import ConnectorConfig
from capabilities.common.meta.connectors.database_connectors import BigQueryConnector

connector = BigQueryConnector(ConnectorConfig(
    connection_string="bigquery://offline",
    database="analytics",
    additional_params={
        "offline_catalog": [{
            "name": "customers",
            "schema": "mart",
            "columns": [
                {"name": "customer_id", "data_type": "INT64", "primary_key": True},
                {"name": "email", "data_type": "STRING", "sample_values": ["owner@example.com"]},
            ],
            "sample_data": [{"customer_id": 1, "email": "owner@example.com"}],
        }]
    },
))

assets = await connector.discover_assets()
schema = await connector.get_asset_schema("mart.customers")
samples = await connector.sample_asset_data("mart.customers", 1)
```

This fixture mode is intentionally explicit. Production adapters can replace it
with live `oracledb`, `aioodbc`, Redis, or Google BigQuery clients later
without changing generated-app catalog workflows.

## Guardrail Summary

META evaluates deterministic rules before lifecycle decisions. Key guardrails:

- Tenant context is required.
- Asset type must be supported.
- Business key and source system are required for registration.
- Published assets require owners and quality evidence.
- Restricted assets require classification and stewards.
- Certification requires lineage and quality above threshold.
- Low-confidence classifications require steward review.
- Classification review decisions require notes.
- Discovery requires approved connectors and current schedule review.
- Lineage requires registered source and target assets.
- Excessive lineage depth requires review.
- Glossary terms require owners.
- Asset retirement requires impact analysis.
- Stale assets require freshness review before certification.
- Catalog-agent runtime and role must be supported.
- Catalog-agent scope, owner, purpose, and machine-contribution disclosure are
  required.
- Privileged catalog-agent roles require human approval evidence or pending
  review.
- Lifecycle batch processing must use Bytewax.
- Data contracts are evaluated on every quality assessment; violations emit
  audit events and move assets to `pending_review`.
- LLM-enrichment suggestions require `contribution_disclosed=True` and a
  human approval gate before being applied to the asset record.

## Adapter Boundaries

This packet defines the executable control plane. Production adapters may supply:

- Durable metadata store persistence.
- Discovery connector execution.
- AI or rules-based classification.
- Lineage graph persistence and traversal.
- Search index maintenance.
- Bytewax lifecycle streams for metadata and catalog-agent events.
- APG audit, auth, MDM, ETL, connector, monitoring, and notification
  integration.
- OTLP trace/metric export (when `OTEL_EXPORTER_OTLP_ENDPOINT` is set).
- Ollama LLM enrichment (when `OLLAMA_BASE_URL` is set).

Adapters must not bypass `capability_contract.py` decisions.

The META packet intentionally does not embed SDK clients for Codex, Claude
Code, opencode, Pi, or future agent providers. Those runtimes connect through
adapters that preserve the APG contract, guardrail decisions, audit events, and
human-approval requirements.

## Local Proof

Focused proof for this package:

```bash
./.venv/bin/python -m py_compile \
  capabilities/common/meta/__init__.py \
  capabilities/common/meta/capability_contract.py \
  capabilities/common/meta/service.py \
  capabilities/common/meta/api.py \
  capabilities/common/meta/view_models.py \
  capabilities/common/meta/app.py \
  capabilities/common/meta/test_capability_contract.py \
  capabilities/common/meta/tests/test_package_contract.py

./.venv/bin/pytest -q \
  capabilities/common/meta/test_capability_contract.py \
  capabilities/common/meta/tests/test_package_contract.py
```
