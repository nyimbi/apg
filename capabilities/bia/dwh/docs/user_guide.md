# Data Warehouse

**Capability ID**: `bia_dwh` | **Domain**: `bia` | **Version**: `1.0.0`

## Description

The Data Warehouse capability (bia_dwh) provides dimensional modelling with star/snowflake/data-vault schema management, table registration, ETL job orchestration with multiple load strategies (full refresh, SCD type 1/2/3, incremental, merge), partition management, data quality rule enforcement with quarantine, and full lineage tracking.

## Installation

```bash
pip install apg-bia-dwh
```

## Provides

- `dimensional_schema_management`
- `star_snowflake_schema_design`
- `etl_orchestration`
- `data_partitioning`
- `data_quality_enforcement`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `schd`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/bia/dwh/dashboard` | `bia_dwh:view` | Overview |
| `/bia/dwh/schemas` | `bia_dwh:schemas` | Schema |
| `/bia/dwh/schemas/<id>` | `bia_dwh:schemas` | Schema |
| `/bia/dwh/tables` | `bia_dwh:tables` | Tables |
| `/bia/dwh/tables/<id>` | `bia_dwh:tables` | Tables |
| `/bia/dwh/etl` | `bia_dwh:etl` | ETL |
| `/bia/dwh/etl/<id>` | `bia_dwh:etl` | ETL |
| `/bia/dwh/quality` | `bia_dwh:quality` | Quality |

## Key Service Methods

- `describe()`
- `create_schema()`
- `get_schema()`
- `list_schemas()`
- `update_schema()`
- `delete_schema()`
- `register_table()`
- `get_table()`
- `list_tables()`
- `update_table()`

_(See `service.py` for complete API.)_

## Interoperability

`bia_dwh` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use bia_dwh;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `BIA_DWH_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
