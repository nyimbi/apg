# Knowledge Graph

**Capability ID**: `kngr` | **Domain**: `common` | **Version**: `1.0.0`

## Description

KNGR provides APG's executable knowledge-graph capability: tenant-scoped source registration, entity resolution, evidence-backed relationship linking, semantic enrichment, bounded reasoning paths, curation, publication, first-class

## Installation

```bash
pip install apg-common-kngr
```

## Provides

- `knowledge_graph`
- `semantic_context`
- `knowledge_agent_composition`

## Requires

- `grph`
- `nlpc`
- `meta`
- `srch`
- `onto`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/kngr/dashboard` | `kngr:view` | Overview |
| `/kngr/sources` | `kngr:source` | Knowledge |
| `/kngr/entities` | `kngr:query` | Knowledge |
| `/kngr/relationships` | `kngr:query` | Knowledge |
| `/kngr/enrichment` | `kngr:enrich` | Knowledge |
| `/kngr/reasoning` | `kngr:reason` | Reasoning |
| `/kngr/context` | `kngr:query` | Context |
| `/kngr/curation` | `kngr:curate` | Curation |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_source()`
- `resolve_entity()`
- `link_relationship()`
- `enrich_entity()`
- `build_reasoning_path()`
- `curate_entity()`
- `publish_graph()`
- `context_neighborhood()`

_(See `service.py` for complete API.)_

## Interoperability

`kngr` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use kngr;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `KNGR_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
