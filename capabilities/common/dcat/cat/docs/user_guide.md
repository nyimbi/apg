# Data Catalog User Guide

## Overview

The Data Catalog capability (`dcat_cat`) provides a centralised registry for all datasets
across the platform. It tracks dataset metadata, lineage relationships, schema evolution,
business glossary terms, tags, ownership, quality scores, and access popularity — with an
Apache Atlas-compatible API surface and W3C DCAT-AP export.

---

## Use Cases

- Register datasets from any source system (databases, data lakes, APIs, files)
- Track column-level data lineage from ingestion through transformation to consumption
- Detect PII-sensitive columns automatically at registration time
- Record per-dimension quality scores and surface a trust score to data consumers
- Detect schema breaking changes before they reach downstream pipelines
- Manage dataset deprecation with successor pointers and removal timelines
- Compute governance health scores per domain to drive metadata completeness campaigns
- Assign business glossary terms to specific dataset columns for semantic discovery
- Track dataset access popularity to surface the most-used data assets
- Export catalog contents as W3C DCAT-AP JSON-LD for open data portals

---

## Quick Start

### Register a Dataset

```http
POST /api/dcat/cat/datasets
Content-Type: application/json

{
  "tenant_id": "acme",
  "name": "transactions_raw",
  "owner": "data-eng@acme.com",
  "source_system": "postgres-prod",
  "description": "Raw transaction events from the payments service",
  "classification": "confidential",
  "domain": "payments",
  "format": "parquet",
  "tags": ["payments", "raw"],
  "schema": {
    "transaction_id": "uuid",
    "amount": "numeric",
    "email": "text",
    "created_at": "timestamp"
  }
}
```

Response includes `id` — use it in all subsequent calls.

---

## Data Quality

### Record a Quality Score

```http
POST /api/dcat/cat/datasets/{id}/quality
Content-Type: application/json

{
  "tenant_id": "acme",
  "dimension": "completeness",
  "score": 0.94,
  "job_id": "dq-run-20260611",
  "details": {"null_rate": 0.06, "rows_checked": 1000000}
}
```

Supported dimensions: `completeness`, `freshness`, `validity`, `uniqueness`, `accuracy`.
Score must be in `[0.0, 1.0]`.

### Retrieve Trust Score

```http
GET /api/dcat/cat/datasets/{id}/quality?tenant_id=acme
```

Returns the latest score per dimension and an aggregate `trust_score` (arithmetic mean).

```json
{
  "dataset_id": "ds-abc123",
  "trust_score": 0.91,
  "dimensions": {
    "completeness": {"score": 0.94, "measured_at": "2026-06-11T10:00:00Z"},
    "freshness":    {"score": 0.88, "measured_at": "2026-06-11T10:00:00Z"}
  },
  "measured_count": 6
}
```

---

## PII Detection

### Scan for PII Columns

```http
GET /api/dcat/cat/datasets/{id}/pii-scan?tenant_id=acme
```

Scans all schema column names against 24 regex patterns (email, phone, SSN, GPS, salary,
etc.). If PII columns are detected the dataset `classification` is automatically upgraded
to `pii`.

```json
{
  "pii_detected": true,
  "flagged_fields": [
    {"column": "email", "matched_patterns": ["\\bemail\\b"], "confidence": 0.9}
  ],
  "classification_upgraded": true,
  "scanned_columns": 4
}
```

---

## Lineage

### Add a Lineage Edge

```http
POST /api/dcat/cat/lineage
Content-Type: application/json

{
  "tenant_id": "acme",
  "source_dataset_id": "ds-abc123",
  "target_dataset_id": "ds-def456",
  "transformation": "daily_aggregation_job",
  "job_name": "bytewax-daily-agg"
}
```

### Walk Upstream / Downstream

```http
GET /api/dcat/cat/lineage/{dataset_id}/upstream?tenant_id=acme&depth=5
GET /api/dcat/cat/lineage/{dataset_id}/downstream?tenant_id=acme&depth=5
```

### Impact Analysis

```http
GET /api/dcat/cat/impact/{dataset_id}?tenant_id=acme
```

Returns all downstream datasets that will be affected by changes to `dataset_id`. Use
this before schema changes or dataset retirement.

---

## Schema Management

### View Schema History

```http
GET /api/dcat/cat/datasets/{id}/schema-history?tenant_id=acme
```

Returns all recorded schema versions in order. A new version is appended every time
`schema` is included in an `update_dataset` call.

### Compute Schema Diff

```http
GET /api/dcat/cat/datasets/{id}/schema-diff?tenant_id=acme&from=1&to=3
```

Classifies changes between two schema versions:

| Severity | Meaning |
|----------|---------|
| `COMPATIBLE` | New optional column added — no downstream breakage |
| `WARNING` | Column type changed — consumers may need updates |
| `BREAKING` | Column removed — downstream pipelines will fail |

A `schema_breaking_change` audit event is emitted automatically when BREAKING changes
are detected.

---

## Deprecation Workflow

### Deprecate a Dataset

```http
POST /api/dcat/cat/datasets/{id}/deprecate
Content-Type: application/json

{
  "tenant_id": "acme",
  "reason": "Replaced by transactions_v2 with improved schema",
  "successor_id": "ds-xyz789",
  "deprecation_date": "2026-09-01"
}
```

Sets `status=deprecated` on the dataset. The `successor_id` pointer guides consumers
to the replacement.

### List Deprecated Datasets

```http
GET /api/dcat/cat/datasets/deprecated?tenant_id=acme
```

Returns all deprecations with `days_until_removal` for migration planning.

---

## Governance Health

### Dataset Completeness Score

```http
GET /api/dcat/cat/datasets/{id}/completeness?tenant_id=acme
```

Scores against 10 dimensions: description, schema, tags, owner, classification,
location_uri, format, domain, lineage edge presence, quality score presence.

```json
{
  "score": 0.8,
  "present": ["description", "schema", "tags", "owner", "classification", "domain"],
  "missing": ["location_uri"],
  "has_lineage": true,
  "has_quality": true
}
```

### Aggregate Governance Health

```http
GET /api/dcat/cat/governance-health?tenant_id=acme
```

Fans out completeness scoring across all active datasets concurrently. Returns:
- `health_score` — overall mean completeness
- `avg_by_domain` — per-domain breakdown
- `low_quality_datasets` — datasets scoring below 0.5, sorted ascending

---

## Catalog Discovery

### Faceted Browse

```http
GET /api/dcat/cat/facets?tenant_id=acme
GET /api/dcat/cat/facets?tenant_id=acme&domain=payments
```

Returns per-value counts across domain, classification, format, source_system, status,
and owner in a single pass. Active filters narrow dependent facets — ready for sidebar
rendering in the catalog UI.

### Full-Text Search

```http
GET /api/dcat/cat/datasets/search?tenant_id=acme&q=transactions
```

Matches against name, description, tags, and source_system.

### Popular Datasets

```http
GET /api/dcat/cat/datasets/popular?tenant_id=acme&limit=10&since_days=30
```

Returns datasets ranked by access count in the trailing window. Use
`POST /api/dcat/cat/datasets/{id}/access` to log access events.

---

## Federated Search

For data mesh topologies with multiple domain tenants:

```http
POST /api/dcat/cat/search/federated
Content-Type: application/json

{
  "root_tenant_id": "platform",
  "query": "customer",
  "child_tenant_ids": ["payments", "marketing", "logistics"]
}
```

Fans out search concurrently across all tenants and merges results with a
`source_tenant` annotation. Errors from individual tenants are captured separately
without blocking results from healthy tenants.

---

## Glossary

### Create a Term

```http
POST /api/dcat/cat/glossary
Content-Type: application/json

{
  "tenant_id": "acme",
  "term": "Net Revenue",
  "definition": "Gross revenue minus refunds, discounts, and chargebacks",
  "domain": "finance",
  "synonyms": ["net_rev", "revenue_net"],
  "related_terms": ["gross_revenue", "chargeback"]
}
```

### Link a Term to a Column

```http
POST /api/dcat/cat/glossary/{term_id}/link-column
Content-Type: application/json

{
  "tenant_id": "acme",
  "dataset_id": "ds-abc123",
  "column_name": "net_revenue_usd"
}
```

### Find All Columns for a Term

```http
GET /api/dcat/cat/glossary/{term_id}/columns?tenant_id=acme
```

Returns `[{dataset_id, dataset_name, column_name}]` — enabling column-level semantic
discovery across the entire catalog.

---

## Export

### Full Catalog Export (JSON)

```http
GET /api/dcat/cat/export?tenant_id=acme
```

Returns all active datasets, glossary terms, and lineage edges as a JSON document.

### W3C DCAT-AP Export (JSON-LD)

```http
GET /api/dcat/cat/export/dcat-ap?tenant_id=acme
```

Returns a DCAT-AP compliant JSON-LD document with `@context` for `dcat:` and
`dcterms:` namespaces. Compatible with CKAN, EU open data portals, and any
DCAT-consuming governance platform.

---

## Atlas Compatibility

`GET /api/dcat/cat/atlas/entity/{id}` returns an Atlas v2-compatible `hive_table` entity.
`POST /api/dcat/cat/atlas/lineage` accepts `{process_qualified_name, inputs, outputs}` for
batch lineage creation matching the Atlas REST API contract.

---

## Audit Trail

```http
GET /api/dcat/cat/audit?tenant_id=acme
```

Returns all catalog mutation events for the tenant. Event types include:
`dataset_created`, `dataset_updated`, `dataset_deleted`, `dataset_deprecated`,
`lineage_edge_added`, `lineage_edge_deleted`, `tag_created`, `dataset_tagged`,
`dataset_untagged`, `glossary_term_created`, `glossary_term_updated`,
`glossary_term_deleted`, `term_linked_to_column`, `ownership_assigned`,
`quality_score_recorded`, `classification_upgraded_to_pii`,
`schema_breaking_change`.

---

## Event Streaming (NATS)

Catalog mutation events are optionally published to NATS JetStream when a `NatsClient`
is injected into the service. The subject pattern is:

```
dcat.cat.events.{tenant_id}.{event_type}
```

Downstream APG capabilities (intel alerts, compliance, lineage tracker) subscribe to
this subject independently. No polling required. The APG streaming platform uses
bytewax + NATS — subscribe with a bytewax source connected to the NATS JetStream
consumer.

---

## Classifications

| Value | Meaning | Auto-detected |
|-------|---------|---------------|
| `public` | Shareable externally | No |
| `internal` | Internal use only | No |
| `confidential` | Restricted circulation | No |
| `restricted` | Need-to-know basis | No |
| `pii` | Personally identifiable information | Yes — via `scan_pii_fields` |
