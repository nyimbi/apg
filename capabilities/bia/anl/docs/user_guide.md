# Analytics Engine

**Capability ID**: `bia_anl` | **Domain**: `bia` | **Version**: `1.0.0`

## Description

The Analytics Engine (bia_anl) provides the core analytical computation runtime for the BIA domain. It delivers ad-hoc SQL query execution, OLAP cube management, metric definition and calculation, multi-datasource connectivity, result caching, query scheduling, and governed analytical data access — all scoped to a tenant.

## Installation

```bash
pip install apg-bia-anl
```

## Provides

- `ad_hoc_query_execution`
- `olap_cube_management`
- `metric_definition_registry`
- `analytical_data_access`
- `query_result_cache`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `schd`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/bia/anl/dashboard` | `bia_anl:view` | Overview |
| `/bia/anl/query-builder` | `bia_anl:query` | Querying |
| `/bia/anl/saved-queries` | `bia_anl:query` | Querying |
| `/bia/anl/saved-queries/<id>` | `bia_anl:query` | Querying |
| `/bia/anl/cubes` | `bia_anl:cubes` | OLAP |
| `/bia/anl/cubes/<id>` | `bia_anl:cubes` | OLAP |
| `/bia/anl/metrics` | `bia_anl:metrics` | Metrics |
| `/bia/anl/metrics/<id>` | `bia_anl:metrics` | Metrics |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_datasource()`
- `test_datasource()`
- `list_datasources()`
- `get_datasource()`
- `delete_datasource()`
- `save_query()`
- `get_query()`
- `list_queries()`

_(See `service.py` for complete API.)_

## Interoperability

`bia_anl` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use bia_anl;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `BIA_ANL_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
