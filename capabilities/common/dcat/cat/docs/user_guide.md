# Data Catalog User Guide

## Overview

The Data Catalog capability (`dcat_cat`) provides a centralised registry for all datasets across the platform. It tracks dataset metadata, lineage relationships, business glossary terms, tags, and ownership — with an Apache Atlas-compatible API surface.

## Use Cases

- Register datasets from any source system (databases, data lakes, APIs, files)
- Track data lineage from ingestion through transformation to consumption
- Assign business glossary terms to datasets for semantic clarity
- Tag datasets with custom labels for discovery and governance
- Track dataset ownership across teams
- Perform impact analysis before changing upstream datasets
- Export catalog metadata for external governance tools

## API Reference

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
  "tags": ["payments", "raw"]
}
```

### Add Lineage

```http
POST /api/dcat/cat/lineage
Content-Type: application/json

{
  "tenant_id": "acme",
  "source_dataset_id": "ds-abc123",
  "target_dataset_id": "ds-def456",
  "transformation": "daily_aggregation_job",
  "job_name": "spark-daily-agg"
}
```

### Search Catalog

```http
GET /api/dcat/cat/datasets/search?tenant_id=acme&q=transactions
```

### Impact Analysis

```http
GET /api/dcat/cat/impact/ds-abc123?tenant_id=acme
```

Returns all downstream datasets impacted by changes to `ds-abc123`.

## Classifications

- `public` — shareable externally
- `internal` — internal use only
- `confidential` — restricted circulation
- `restricted` — need-to-know basis
- `pii` — personally identifiable information

## Atlas Compatibility

The `/api/dcat/cat/atlas/entity/{id}` endpoint returns Atlas-compatible `hive_table` entity representation compatible with Apache Atlas REST API v2.
