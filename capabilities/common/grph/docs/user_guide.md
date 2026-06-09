# Graph Data Management

**Capability ID**: `grph` | **Domain**: `common` | **Version**: `1.0.0`

## Description

GRPH provides the APG graph foundation: tenant-scoped schemas, nodes, edges, lineage graphs, relationship governance, bounded traversal, graph quality inspection, first-class graph-agent composition, Bytewax lifecycle batch

## Installation

```bash
pip install apg-common-grph
```

## Provides

- `graph_data_management`
- `relationship_intelligence`
- `graph_agent_composition`

## Requires

- `mdm`
- `meta`
- `etlp`
- `srch`
- `aicr`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/grph/dashboard` | `grph:view` | Overview |
| `/grph/explorer` | `grph:query` | Graph |
| `/grph/schemas` | `grph:manage_schema` | Schema |
| `/grph/nodes` | `grph:write` | Graph |
| `/grph/edges` | `grph:write` | Graph |
| `/grph/traversal` | `grph:query` | Graph |
| `/grph/lineage` | `grph:view` | Lineage |
| `/grph/impact` | `grph:query` | Lineage |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_schema()`
- `create_node()`
- `create_edge()`
- `traverse()`
- `lineage_path()`
- `impact_analysis()`
- `neighborhood()`
- `quality_report()`

_(See `service.py` for complete API.)_

## Interoperability

`grph` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use grph;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GRPH_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
