# Search Engine

**Capability ID**: `srch` | **Domain**: `common` | **Version**: `1.0.0`

## Description

SRCH is the APG capability for governed enterprise search across tenant-scoped indices, documents, facets, keyword retrieval, semantic retrieval, hybrid retrieval, and query analytics. It lets generated applications create indices,

## Installation

```bash
pip install apg-common-srch
```

## Provides

- `enterprise_search`
- `semantic_retrieval`
- `search_agent_composition`

## Requires

- `etlp`
- `meta`
- `nlpc`
- `aicr`
- `conf`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/srch/dashboard` | `srch:view` | Overview |
| `/srch/search` | `srch:query` | Search |
| `/srch/indices` | `srch:manage_indices` | Indexes |
| `/srch/documents` | `srch:index` | Indexes |
| `/srch/bulk` | `srch:index` | Indexes |
| `/srch/facets` | `srch:view` | Search |
| `/srch/analytics` | `srch:view` | Operations |
| `/srch/ranking` | `srch:govern` | Operations |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_index()`
- `mark_embedding_index_ready()`
- `index_document()`
- `bulk_index_documents()`
- `query()`
- `facets()`
- `create_record()`
- `list_records()`

_(See `service.py` for complete API.)_

## Interoperability

`srch` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use srch;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `SRCH_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
