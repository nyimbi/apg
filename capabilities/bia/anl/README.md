# Analytics Engine

## Overview
The Analytics Engine (bia_anl) provides the core analytical computation runtime for the BIA domain. It delivers ad-hoc SQL query execution, OLAP cube management, metric definition and calculation, multi-datasource connectivity, result caching, query scheduling, and governed analytical data access — all scoped to a tenant.

## Capability ID
`bia_anl`

## Provides
- ad_hoc_query_execution: Execute parameterised SQL queries against registered datasources
- olap_cube_management: Create, refresh, archive, and query multidimensional OLAP cubes
- metric_definition_registry: Define, version, and govern calculated business metrics
- analytical_data_access: Governed access layer with access-level enforcement
- query_result_cache: Session, hourly, daily, and weekly result caching
- datasource_connectivity: Register and test connections to 9 datasource types
- saved_query_library: Store, share, and version reusable queries
- query_scheduling: Schedule recurring query execution via the schd capability
- result_export: Export results to JSON, CSV, Parquet, Arrow, XLSX, HTML

## Requires
| Capability | Reason |
|------------|--------|
| auth | User identity and permission checks |
| audl | Audit trail for all query executions |
| mten | Tenant context enforcement |
| conf | Runtime configuration management |
| schd | Scheduled query execution |
| mqeb | Streaming query lifecycle events |
| moni | Operational monitoring of query performance |
| nlpc | Natural-language query parsing (future) |

## Configuration
| Option | Default | Description |
|--------|---------|-------------|
| max_rows_per_query | 100,000 | Hard row limit per query |
| timeout_seconds | 300 | Query execution timeout |
| default_cache_policy | session | Cache lifetime for results |
| require_approval_for_public | true | Public queries require approval |
| credentials_vault_required | true | Datasource credentials must be in vault |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /api/bia/anl/queries | GET | List saved queries | bia_anl:query |
| /api/bia/anl/queries | POST | Save a new query | bia_anl:query |
| /api/bia/anl/queries/<id> | GET | Get query detail | bia_anl:query |
| /api/bia/anl/queries/<id> | PUT | Update query | bia_anl:query |
| /api/bia/anl/queries/<id> | DELETE | Delete query | bia_anl:query |
| /api/bia/anl/queries/<id>/execute | POST | Execute query | bia_anl:query |
| /api/bia/anl/cubes | GET | List OLAP cubes | bia_anl:cubes |
| /api/bia/anl/cubes | POST | Create cube | bia_anl:cubes |
| /api/bia/anl/cubes/<id> | GET | Get cube detail | bia_anl:cubes |
| /api/bia/anl/cubes/<id>/refresh | POST | Refresh cube | bia_anl:cubes |
| /api/bia/anl/metrics | GET | List metrics | bia_anl:metrics |
| /api/bia/anl/metrics | POST | Define metric | bia_anl:metrics |
| /api/bia/anl/datasources | GET | List datasources | bia_anl:admin |
| /api/bia/anl/datasources | POST | Register datasource | bia_anl:admin |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context | deny |
| cross_tenant_query_denied | Cross-tenant access | deny |
| query_timeout_enforced | Timeout exceeded | deny |
| max_rows_enforced | Row limit exceeded | deny |
| public_access_requires_approval | access_level=public, not approved | deny |
| datasource_credentials_vault_required | Credentials not in vault | deny |
| stale_cube_read_allowed_with_warning | Cube state=stale | allow with metadata |

## Data Models
- DatasourceResponse: id, tenant_id, name, datasource_type, connection_config, credentials_vault_ref, owner_id
- QueryResponse: id, tenant_id, name, query_type, sql_text, datasource_id, access_level, cache_policy, owner_id
- CubeResponse: id, tenant_id, name, dimensions, measures, grain_sql, state, owner_id, last_refreshed_at
- MetricResponse: id, tenant_id, name, metric_type, formula, cube_id, unit, owner_id
- QueryResultResponse: query_id, columns, rows, row_count, execution_time_ms, cached

## Streaming Events
- query_executed, query_saved, query_scheduled
- cube_created, cube_refreshed, cube_archived
- metric_defined, metric_updated
- datasource_registered, datasource_tested
- result_exported

## Edge Cases Handled
- Stale cube reads return data with staleness metadata attached rather than blocking
- Archived cubes reject refresh requests with explicit restore instruction
- Shared queries (team access) can only be deleted by the owner
- Row and timeout limits are enforced per-query regardless of datasource type
- Credentials are never stored inline — vault reference is mandatory

## Composability Notes
- Feeds results into dsh (Dashboard Management) for widget data binding
- Metrics feed into rpt (Report Builder) for parameterised reports
- Cube refresh can be orchestrated by wflo (Workflow) with approval gates
- nlpc can translate natural-language questions into SQL for query_builder
- moni tracks query latency and cube staleness for SLA alerting
