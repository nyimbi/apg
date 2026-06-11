# Data Catalog (dcat_cat)

Dataset registry, data lineage graph, metadata tagging, glossary, Apache Atlas-compatible
API, ownership tracking, data quality scoring, PII auto-detection, schema diff, deprecation
workflow, governance health scoring, catalog facets, glossary-column linkage, federated
search, and W3C DCAT-AP export.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/dcat/cat/health | Service health |
| GET | /api/dcat/cat/datasets | List datasets |
| POST | /api/dcat/cat/datasets | Register dataset |
| POST | /api/dcat/cat/datasets/bulk | Bulk register datasets |
| GET | /api/dcat/cat/datasets/{id} | Get dataset |
| PUT | /api/dcat/cat/datasets/{id} | Update dataset |
| DELETE | /api/dcat/cat/datasets/{id} | Soft-delete dataset |
| GET | /api/dcat/cat/datasets/search?q= | Full-text search |
| GET | /api/dcat/cat/datasets/{id}/schema-history | Schema version history |
| GET | /api/dcat/cat/datasets/{id}/schema-diff?from=1&to=2 | Schema diff (COMPATIBLE/WARNING/BREAKING) |
| POST | /api/dcat/cat/lineage | Add lineage edge |
| GET | /api/dcat/cat/lineage | List lineage edges |
| GET | /api/dcat/cat/lineage/{id}/upstream | Upstream lineage walk |
| GET | /api/dcat/cat/lineage/{id}/downstream | Downstream lineage walk |
| DELETE | /api/dcat/cat/lineage/{id} | Remove lineage edge |
| POST | /api/dcat/cat/glossary | Create glossary term |
| GET | /api/dcat/cat/glossary | List glossary terms |
| GET | /api/dcat/cat/glossary/{id} | Get glossary term |
| PUT | /api/dcat/cat/glossary/{id} | Update glossary term |
| DELETE | /api/dcat/cat/glossary/{id} | Delete glossary term |
| GET | /api/dcat/cat/glossary/search?q= | Search glossary |
| POST | /api/dcat/cat/glossary/{id}/link-column | Link term to dataset column |
| GET | /api/dcat/cat/glossary/{id}/columns | Find all columns for a term |
| POST | /api/dcat/cat/tags | Create tag |
| GET | /api/dcat/cat/tags | List tags |
| POST | /api/dcat/cat/datasets/{id}/tags | Apply tag to dataset |
| DELETE | /api/dcat/cat/datasets/{id}/tags/{tag} | Remove tag from dataset |
| GET | /api/dcat/cat/datasets/{id}/quality | Quality profile (trust score) |
| POST | /api/dcat/cat/datasets/{id}/quality | Record quality score |
| GET | /api/dcat/cat/datasets/{id}/pii-scan | Scan schema columns for PII |
| POST | /api/dcat/cat/datasets/{id}/access | Record dataset access event |
| GET | /api/dcat/cat/datasets/popular | Popular datasets by access count |
| POST | /api/dcat/cat/datasets/{id}/deprecate | Deprecate a dataset |
| GET | /api/dcat/cat/datasets/deprecated | List all deprecated datasets |
| GET | /api/dcat/cat/datasets/{id}/completeness | Metadata completeness score |
| GET | /api/dcat/cat/statistics | Catalog statistics |
| GET | /api/dcat/cat/governance-health | Aggregate governance health score |
| GET | /api/dcat/cat/facets | Catalog discovery facets |
| GET | /api/dcat/cat/audit | Audit trail |
| GET | /api/dcat/cat/impact/{id} | Impact analysis (downstream) |
| GET | /api/dcat/cat/export | Full catalog export (JSON) |
| GET | /api/dcat/cat/export/dcat-ap | W3C DCAT-AP JSON-LD export |
| GET | /api/dcat/cat/atlas/entity/{id} | Atlas-compatible entity |
| POST | /api/dcat/cat/atlas/search | Atlas-compatible search |
| POST | /api/dcat/cat/atlas/lineage | Atlas batch lineage creation |
| POST | /api/dcat/cat/search/federated | Cross-tenant federated search |
| GET | /api/dcat/cat/ownership/{dataset_id} | Ownership history |

## New Capabilities (v1.1)

### Data Quality Trust Scores
Record per-dimension quality scores (completeness, freshness, validity, uniqueness,
accuracy) and compute an aggregate `trust_score` per dataset. Surfaced at discovery
time so consumers can assess data reliability before querying.

### PII Auto-Detection
`scan_pii_fields` inspects schema column names against 24 regex patterns covering email,
phone, SSN, GPS coordinates, salary, and more. Automatically upgrades `classification`
to `pii` when patterns match — never downgrades.

### Schema Diff and Breaking-Change Detection
`compute_schema_diff` compares any two schema versions and classifies each change as
`COMPATIBLE` (new column added), `WARNING` (type changed), or `BREAKING` (column removed).
Emits a `schema_breaking_change` audit event for compliance routing.

### Dataset Deprecation Workflow
`deprecate_dataset` records a structured deprecation with reason, optional successor
dataset pointer, and removal date. `list_deprecated_datasets` returns days-until-removal
so consumers can plan migrations.

### Governance Health Scoring
`score_dataset_completeness` scores each dataset across 10 dimensions (description,
schema, tags, lineage, quality coverage, etc.). `get_governance_health` fans these out
concurrently and returns per-domain averages plus a ranked list of low-quality datasets.

### Faceted Discovery
`get_catalog_facets` returns per-value counts across domain, classification, format,
source_system, status, and owner in a single O(n) pass — ready for sidebar rendering
with active-filter narrowing.

### Glossary-Column Linkage
`link_term_to_column` binds a business term to a specific schema column, not just the
dataset. `find_columns_by_term` supports column-level semantic queries: "show me all
columns that represent `net_revenue`."

### Popularity and Usage Tracking
`record_dataset_access` logs access events. `get_popular_datasets` ranks by access
frequency within a configurable trailing window (default 30 days).

### Federated Multi-Tenant Search
`federate_search` fans out `search_datasets` across multiple tenant namespaces
concurrently via `asyncio.gather`, merging results with a `source_tenant` annotation.
Designed for data mesh topologies.

### W3C DCAT-AP Export
`export_dcat_ap` serialises the catalog as DCAT-AP JSON-LD with `dcterms` and `dcat`
namespaces — compatible with EU open data portals, CKAN, and any DCAT-consuming
governance tool.

## Classifications

| Value | Meaning |
|-------|---------|
| `public` | Shareable externally |
| `internal` | Internal use only |
| `confidential` | Restricted circulation |
| `restricted` | Need-to-know basis |
| `pii` | Personally identifiable information |

## Supported Formats

`csv`, `parquet`, `avro`, `json`, `orc`, `delta`, `iceberg`, `unknown`

## Atlas Compatibility

`/api/dcat/cat/atlas/entity/{id}` returns an Atlas-compatible `hive_table` entity
representation. `/api/dcat/cat/atlas/lineage` accepts `{process_qualified_name, inputs,
outputs}` in Atlas batch-lineage format.

## Event Streaming

Catalog mutation events are emitted to `_audit_events` (in-process) and optionally
published to NATS JetStream subject `dcat.cat.events.{tenant_id}.{event_type}` when
a `NatsClient` is injected. Downstream capabilities (intel, compliance, lineage trackers)
subscribe independently — no polling required.
