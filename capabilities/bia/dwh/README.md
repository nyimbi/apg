# Data Warehouse

## Overview
The Data Warehouse capability (bia_dwh) provides dimensional modelling with star/snowflake/data-vault schema management, table registration, ETL job orchestration with multiple load strategies (full refresh, SCD type 1/2/3, incremental, merge), partition management, data quality rule enforcement with quarantine, and full lineage tracking.

## Capability ID
`bia_dwh`

## Provides
- dimensional_schema_management: Star, snowflake, galaxy, flat, data-vault schema lifecycle
- star_snowflake_schema_design: Fact and dimension table registration with grain definition
- etl_orchestration: ETL jobs with 7 load strategies and parallel execution control
- data_partitioning: Range, list, hash, composite partition strategies
- data_quality_enforcement: 8 rule types with quarantine-on-failure
- lineage_tracking: Source-to-target lineage recording per ETL run
- storage_tier_management: Hot/warm/cold/archive tier assignment
- warehouse_catalogue: Searchable table and schema catalogue

## Requires
| Capability | Reason |
|------------|--------|
| auth | User identity and permission checks |
| audl | Audit trail for schema and ETL changes |
| mten | Tenant context enforcement |
| conf | Runtime configuration management |
| schd | ETL job scheduling |
| mqeb | Streaming warehouse lifecycle events |
| moni | ETL run monitoring and alerting |
| comp | Regulatory compliance for data governance |

## Configuration
| Option | Default | Description |
|--------|---------|-------------|
| max_parallel_etl_jobs | 10 | Concurrent ETL job limit |
| quarantine_on_failure | true | Failed quality checks send rows to quarantine |
| lineage_tracking_required | true | All tables must have lineage reference |
| auto_tiering_enabled | true | Storage tier managed automatically |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /api/bia/dwh/schemas | GET | List schemas | bia_dwh:schemas |
| /api/bia/dwh/schemas | POST | Create schema | bia_dwh:schemas |
| /api/bia/dwh/schemas/<id> | GET/PUT/DELETE | Schema CRUD | bia_dwh:schemas |
| /api/bia/dwh/tables | GET | List tables | bia_dwh:tables |
| /api/bia/dwh/tables | POST | Register table | bia_dwh:tables |
| /api/bia/dwh/tables/<id> | GET/PUT/DELETE | Table CRUD | bia_dwh:tables |
| /api/bia/dwh/etl | GET | List ETL jobs | bia_dwh:etl |
| /api/bia/dwh/etl | POST | Create ETL job | bia_dwh:etl |
| /api/bia/dwh/etl/<id>/run | POST | Run ETL job | bia_dwh:etl |
| /api/bia/dwh/quality | GET/POST | Quality rules | bia_dwh:quality |
| /api/bia/dwh/lineage | GET/POST | Lineage records | bia_dwh:lineage |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context | deny |
| cross_tenant_access_denied | Cross-tenant access | deny |
| lineage_tracking_required | No lineage_ref on table | deny |
| etl_parallel_limit_enforced | Too many concurrent jobs | deny |
| quarantine_on_quality_failure | Quality check failed | deny (quarantine) |
| scd2_requires_surrogate_key | SCD type 2 + no surrogate key | deny |
| drop_table_requires_no_dependents | Table has dependents | deny |
| cold_tier_requires_archival_policy | Archive tier + no policy | deny |

## Data Models
- SchemaResponse: id, tenant_id, name, schema_type, grain, owner_id, table_count
- TableResponse: id, schema_id, name, table_type, columns, partition_strategy, storage_tier, lineage_ref
- ETLJobResponse: id, name, source_ref, target_table_id, load_strategy, state, last_run_at, last_run_rows
- QualityRuleResponse: id, table_id, rule_type, column, config, last_checked_at, last_result
- LineageRecord: id, source_table_id, target_table_id, etl_job_id, transformation_description

## Streaming Events
- schema_created, schema_updated, table_registered, table_updated
- etl_job_started, etl_job_completed, etl_job_failed
- quality_rule_violated, lineage_recorded, partition_created, storage_tier_changed

## Edge Cases Handled
- SCD type 2 jobs require a surrogate key on the target dimension — rejected at job creation
- Tables with downstream dependents cannot be dropped without removing dependents first
- Archive storage tier requires an explicit archival policy reference
- ETL parallel execution is capped per tenant to prevent resource exhaustion
- Lineage tracking is mandatory for all registered tables — sandbox tables use self-referential lineage

## Composability Notes
- bia_anl registers DWH tables as datasources for ad-hoc query execution
- bia_pda reads warehouse tables as training datasets for ML models
- bia_rpt uses warehouse views as report datasources
- comp can attach data retention and residency policies to tables
- schd drives scheduled ETL job execution with retry logic

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Column-Level Encryption with Key Rotation** [Security & Compliance]
- **I2. Materialized View Lifecycle Management** [Query Performance]
- **I3. Cost Attribution and Slot Budgeting per Tenant** [FinOps / Multi-tenancy]
- **I4. Automated Data Vault 2.0 Hub/Link/Satellite Generation** [Schema Design Automation]
- **I5. Real-Time CDC (Change Data Capture) Pipeline Management** [Data Ingestion]
- **I6. Adaptive Partitioning with Automatic Partition Pruning Statistics** [Query Optimisation]
- **I7. SCD Type 4 and Type 6 (Hybrid) Support** [Dimensional Modelling]
- **I8. Query Cost Estimation Before Execution** [Query Governance]
- **I9. Semantic Layer with Metric Definitions** [Business Intelligence]
- **I10. Automated Index Recommendation and Management** [Query Optimisation]
- **I11. Data Freshness SLA Monitoring with Breach Alerting** [Observability]
- **I12. Cross-Table Row-Level Security Policies** [Security & Governance]
- **I13. Time-Travel and Point-in-Time Query Support** [Auditability & Recovery]
- **I14. Automated Slowly Changing Dimension (SCD) Backfill Engine** [Historical Data Management]
- **I15. Intelligent ETL Job Dependency Graph with Topological Scheduling** [ETL Orchestration]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
