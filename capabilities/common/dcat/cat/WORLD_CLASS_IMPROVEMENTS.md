# Data Catalog — World-Class Improvements

15 prioritised improvements to elevate `dcat_cat` from a functional registry to a
production-grade data intelligence platform on par with DataHub, Collibra, and Alation.

---

### I1. Column-Level Lineage with Field Mapping
**Category**: Lineage depth
**Justification**: Table-level lineage is table-stakes. Column-level lineage lets engineers
trace exactly which source column flows into a report field — reducing root-cause
investigation from hours to minutes. LinkedIn's DataHub considers this a P0 capability.
**Implementation**: Extend `LineageEdge` with a `field_mappings: list[FieldMap]` structure
where `FieldMap` captures `{source_column, target_column, transform_expression}`. Walk the
graph at column granularity in `get_lineage_upstream`/`get_lineage_downstream`.
**Competitor**: DataHub (column-level lineage graph), dbt (column-level via `compiled_sql`)

---

### I2. Data Quality Score Embedding
**Category**: Quality governance
**Justification**: A catalog without quality signals is a phone book. Embedding freshness,
completeness, and validity scores directly on dataset records lets consumers make informed
decisions at discovery time, not post-ingestion. Alation calls this the "trust score."
**Implementation**: `async def record_quality_score(tenant_id, dataset_id, dimension, score,
measured_at, job_id)` persists quality metrics; `async def get_quality_profile(...)` returns
aggregated scores. Surface worst/best quality per domain in `catalog_statistics`.
**Competitor**: Alation (trust scores), Monte Carlo (data observability), Great Expectations

---

### I3. NATS-Backed Real-Time Catalog Events
**Category**: Event streaming
**Justification**: Synchronous in-process audit is not observable by external consumers.
Publishing catalog mutation events to NATS JetStream enables downstream capabilities
(intel, compliance, lineage trackers) to react in real time — decoupling them from
the catalog service entirely.
**Implementation**: Inject an optional `NatsClient`; in `_emit`, publish to subject
`dcat.cat.events.{tenant_id}.{event_type}` with CloudEvents envelope. Fallback to in-memory
list when NATS is absent.
**Competitor**: DataHub (Kafka-backed metadata events), Atlan (webhooks)

---

### I4. Semantic Search with Embeddings
**Category**: Discovery
**Justification**: Keyword search misses synonyms, typos, and intent. Vector similarity search
over dataset names + descriptions lets a data analyst type "revenue by geography" and surface
`fin_regional_sales_cube` — impossible with substring matching.
**Implementation**: `async def embed_dataset(dataset_id)` calls local Ollama
(`nomic-embed-text`) to generate embeddings stored in `self._embeddings: dict[str, list[float]]`.
`async def semantic_search(tenant_id, query, top_k)` computes cosine similarity in-process.
**Competitor**: Atlan (AI-powered search), Secoda (semantic layer)

---

### I5. Data Contract Enforcement
**Category**: Governance / SLOs
**Justification**: Data contracts formalise the SLA between producers and consumers. Without
them, schema changes silently break downstream pipelines. Encoding contracts in the catalog
makes the registry the single source of truth for pipeline SLOs.
**Implementation**: `async def create_data_contract(tenant_id, dataset_id, schema_sla,
freshness_sla, quality_sla, owner)` stores contracts; `async def validate_contract(...)` checks
current dataset state against declared SLOs and returns pass/fail with violation detail.
**Competitor**: Soda Data Contracts, PayPal Data Contract Standard, dbt contracts

---

### I6. Automated PII Detection and Classification
**Category**: Privacy / compliance
**Justification**: Manual classification is slow and error-prone. Auto-detecting PII columns
from schema field names and types surfaces compliance risk at registration time — before data
flows into the wrong place.
**Implementation**: `async def scan_pii_fields(tenant_id, dataset_id)` runs regex + keyword
matching over schema column names (`email`, `phone`, `ssn`, `dob`, `ip_address`, etc.) and
upgrades classification to `pii` when patterns match. Returns a `PIIScanResult` with flagged
fields and confidence scores.
**Competitor**: AWS Glue DataBrew (PII detection), BigQuery Data Policy, Privacera

---

### I7. Faceted Catalog Discovery API
**Category**: Discovery / UX
**Justification**: Filter menus on data portals are built on faceted counts. A single endpoint
returning per-facet counts (domain × classification × format × source_system) lets UIs render
discovery sidebars without N+1 queries.
**Implementation**: `async def get_catalog_facets(tenant_id, active_filters)` returns
`{domain: {payments: 42}, classification: {pii: 18}, format: {parquet: 91}, ...}` computed
in a single pass over in-memory datasets.
**Competitor**: Collibra (faceted search), DataHub (faceted browse)

---

### I8. Popularity and Usage Tracking
**Category**: Discovery intelligence
**Justification**: Datasets accessed by many consumers are implicitly more trustworthy.
Recording query events and surfacing "hot" datasets guides new analysts to well-understood
data and flags stale datasets no one is using.
**Implementation**: `async def record_dataset_access(tenant_id, dataset_id, accessor, access_type)`
increments access counters; `async def get_popular_datasets(tenant_id, limit, since_days)` returns
ranked list by access count. Integrate access signals into `catalog_statistics`.
**Competitor**: Alation (popularity score), DataHub (usage statistics), Atlan (access metrics)

---

### I9. Multi-Tenant Catalog Federation
**Category**: Multi-tenancy / enterprise
**Justification**: Enterprise data meshes span multiple BUs each with their own catalog instance.
Federation allows a root tenant to query across child tenants with proper access control —
enabling cross-domain lineage without merging tenants into a single namespace.
**Implementation**: `async def federate_search(root_tenant_id, query, child_tenant_ids)` fans
out `search_datasets` coroutines via `asyncio.gather` and merges results with a `source_tenant`
annotation. Guard with explicit federation allowlist per tenant.
**Competitor**: Collibra (federated catalog), DataHub (multi-cluster), Alation (connected
data sources)

---

### I10. Dataset Deprecation Workflow
**Category**: Lifecycle management
**Justification**: Deleting a dataset breaks consumers. A deprecation workflow — with a notice
period, migration hint, and successor dataset pointer — gives consumers time to migrate and
generates actionable alerts for owners.
**Implementation**: `async def deprecate_dataset(tenant_id, dataset_id, reason, successor_id,
deprecation_date)` sets `status="deprecated"` and records the deprecation plan; `async def
list_deprecated_datasets(tenant_id)` surfaces all deprecations with days-until-removal.
**Competitor**: dbt (deprecation blocks), DataHub (deprecation workflow), Atlan (sunset policy)

---

### I11. Schema Diff and Breaking Change Detection
**Category**: Schema governance
**Justification**: Schema changes that drop or rename columns break downstream consumers
silently. Computing a structured diff between consecutive schema versions and classifying
changes as `COMPATIBLE` | `WARNING` | `BREAKING` stops bad schema changes before they
reach production.
**Implementation**: `async def compute_schema_diff(tenant_id, dataset_id, from_version,
to_version)` compares field sets and types; classifies dropped fields and type changes as
`BREAKING`, new optional fields as `COMPATIBLE`. Emits `schema_breaking_change` audit event.
**Competitor**: Confluent Schema Registry (compatibility levels), dbt (schema tests), Buf

---

### I12. Glossary Term Linkage to Dataset Columns
**Category**: Semantic layer
**Justification**: Linking a glossary term to specific columns (not just datasets) enables
true semantic discovery: "find all columns that represent `net_revenue`" returns precisely
the columns, not the datasets. This is the foundation of a business semantic layer.
**Implementation**: `async def link_term_to_column(tenant_id, term_id, dataset_id, column_name)`
stores column-level term bindings; `async def find_columns_by_term(tenant_id, term_id)` returns
`[{dataset_id, dataset_name, column_name}]`. Expose in `atlas_get_entity` as `businessTerms`.
**Competitor**: Collibra (term-to-attribute linkage), DataHub (glossary node), Alation (table
field mapping)

---

### I13. RBAC-Aware Catalog Visibility
**Category**: Security / access control
**Justification**: A catalog that exposes PII dataset names to all users is itself a compliance
risk. Row-level visibility filtering based on caller roles (read from JWT claims or a
capability-local policy store) ensures `restricted` and `pii` datasets are invisible to
unauthorised viewers.
**Implementation**: `async def list_datasets_for_user(tenant_id, user_id, roles)` applies a
policy filter — datasets with `classification in (restricted, pii)` are stripped unless the
caller holds `data:restricted:read`. Policy rules stored in `self._policies`.
**Competitor**: Privacera, Immuta, Collibra (data access governance)

---

### I14. Catalog Health Scoring and Metadata Completeness
**Category**: Data governance maturity
**Justification**: An incomplete catalog entry (no description, no owner, no classification)
provides false confidence. A completeness score per dataset — and an aggregate maturity score
per domain — creates accountability and drives governance adoption.
**Implementation**: `async def score_dataset_completeness(tenant_id, dataset_id)` checks
presence of description, schema, tags, owner, classification, lineage edges, glossary links.
Returns `{score: 0.85, missing: ["glossary_link", "lineage_upstream"]}`. Aggregate into
`catalog_statistics` as `governance_health_score`.
**Competitor**: Collibra (governance maturity), Atlan (completeness score), DataHub (metadata
completeness)

---

### I15. OpenMetadata / DCAT-AP Export
**Category**: Interoperability / open standards
**Justification**: Vendor lock-in at the metadata layer is expensive. Exporting catalog contents
as DCAT-AP (W3C standard) or OpenMetadata schema enables plug-and-play integration with
government portals, data mesh platforms, and any DCAT-compatible consumer.
**Implementation**: `async def export_dcat_ap(tenant_id)` serialises datasets as
`dcat:Dataset` JSON-LD with `dcterms:title`, `dcterms:description`, `dcat:distribution`,
`dcterms:modified`. `async def export_open_metadata(tenant_id)` maps to OpenMetadata
`Table` entity schema v1.3.
**Competitor**: CKAN (DCAT-AP native), OpenMetadata (open standard entity model), Apache
Atlas (HCatalog / HMS export)
