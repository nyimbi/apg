# Data Warehouse (bia_dwh) — World-Class Improvement Proposals

### I1. Column-Level Encryption with Key Rotation
**Category**: Security & Compliance
**Justification**: Enterprise DWH platforms (Snowflake, BigQuery) encrypt specific columns (PII, financial) with customer-managed keys. Storing plaintext in warehouse tables violates GDPR/HIPAA by default; per-column AES-256-GCM encryption with automatic key rotation reduces blast radius of a breach from full-table to individual column sets.
**Implementation**: Add `encrypt_column(tenant_id, table_id, column, key_ref)` that records an `EncryptionPolicy` per column. Intercept `load_fact` / `load_dimension` to apply envelope encryption using a KMS-backed DEK before storage. `decrypt_column` for authorised reads; `rotate_column_key` performs re-encryption in-place with audit trail.
**Competitor**: Snowflake Tri-Secret Secure, BigQuery CMEK column encryption, Databricks Delta Lake column masking.

---

### I2. Materialized View Lifecycle Management
**Category**: Query Performance
**Justification**: Ad-hoc analytical queries against billion-row fact tables are 10–100x slower than against pre-aggregated materialised views. dbt, Redshift MVs, and Snowflake Dynamic Tables make this a first-class primitive. Exposing MV lifecycle (create, refresh, expire, cascade) removes the manual SQL scaffolding that currently blocks BI analysts.
**Implementation**: `create_materialized_view(tenant_id, name, source_sql, refresh_schedule, incremental_key)` stores the definition and emits a `mv_refresh_requested` event. `refresh_materialized_view` recomputes and records `last_refresh_at`, `rows_produced`, and `refresh_duration_ms`. `expire_materialized_view` marks stale and alerts downstream consumers via lineage.
**Competitor**: dbt materialised models, Redshift Materialized Views, Snowflake Dynamic Tables.

---

### I3. Cost Attribution and Slot Budgeting per Tenant
**Category**: FinOps / Multi-tenancy
**Justification**: Without per-tenant compute cost attribution, shared DWH clusters turn into "noisy neighbour" environments where one tenant's large backfill consumes 80 % of cluster slots. BigQuery slot reservations and Redshift WLM queues prove this pays for itself in reduced infrastructure bills.
**Implementation**: `set_slot_budget(tenant_id, daily_slot_hours, burst_multiplier)` persists a budget record. `get_cost_attribution(tenant_id, period)` aggregates ETL run `duration_ms` × `slot_cost_rate` per job and returns `total_cost_usd: Decimal`, `top_consumers`, and `budget_utilisation_pct`. Warn at 80 %, deny new runs at 100 % with a budget override mechanism.
**Competitor**: BigQuery Slot Reservations, Redshift WLM, Databricks DBUs per workspace.

---

### I4. Automated Data Vault 2.0 Hub/Link/Satellite Generation
**Category**: Schema Design Automation
**Justification**: Data Vault 2.0 is the dominant pattern for audit-friendly, insert-only enterprise warehouses, but the schema boilerplate (hub tables, link tables, satellite tables, hash keys) is tedious and error-prone by hand. Tools like WhereScape and Vaultspeed charge six figures for this automation.
**Implementation**: `generate_data_vault_schema(tenant_id, source_entities, business_keys)` takes a list of source entity descriptions and emits a complete DV2 schema: hub tables with `hk_<entity>` SHA-256 hash keys, link tables for each M:N relationship, satellites with `load_date` / `load_end_date` / `record_source`. Validates hub key uniqueness and satellite-to-hub referential integrity.
**Competitor**: WhereScape RED, Vaultspeed, dbt-vault (AutomateDV), Datavault Builder.

---

### I5. Real-Time CDC (Change Data Capture) Pipeline Management
**Category**: Data Ingestion
**Justification**: Batch ETL introduces latency measured in hours. CDC via log-shipping (Debezium, PostgreSQL WAL) reduces data freshness to sub-second. Managing CDC pipeline lifecycle — source connector config, offset tracking, schema change propagation — in the DWH layer eliminates the need for a separate Kafka Connect cluster management tool.
**Implementation**: `register_cdc_source(tenant_id, source_dsn, tables, connector_type)` stores connector config and spawns a background task. `get_cdc_lag(tenant_id, source_id)` returns `lag_seconds`, `events_per_second`, and `last_offset`. `pause_cdc_source` / `resume_cdc_source` with graceful offset commit. Schema changes detected via CDC emit `cdc_schema_drift_detected` events for automated `schema_evolution`.
**Competitor**: Debezium + Kafka Connect, AWS DMS, Fivetran CDC, Airbyte CDC.

---

### I6. Adaptive Partitioning with Automatic Partition Pruning Statistics
**Category**: Query Optimisation
**Justification**: Manually managing time-based partitions as data grows leads to either too-fine granularity (millions of micro-partitions that slow metadata operations) or too-coarse (full-table scans). Snowflake micro-partitioning and BigQuery auto-clustering adapt partition boundaries to actual data distribution, delivering 10x query speedups without DBA intervention.
**Implementation**: `analyse_partition_skew(tenant_id, table_name)` samples row counts per partition and returns a skew coefficient. `auto_repartition(tenant_id, table_name, target_partition_size_mb)` computes optimal boundary values and emits DDL to merge/split partitions. Tracks `partition_pruning_effectiveness` — the ratio of partitions scanned vs. total — per ETL run.
**Competitor**: Snowflake Automatic Clustering, BigQuery Partition Pruning, Delta Lake Z-Order clustering.

---

### I7. SCD Type 4 and Type 6 (Hybrid) Support
**Category**: Dimensional Modelling
**Justification**: SCD Type 1 overwrites history; Type 2 explodes row counts. Type 4 (mini-dimension) and Type 6 (hybrid 1+2+3) are the correct solution for high-cardinality attributes that change frequently (e.g., customer loyalty tier, risk score). No open-source DWH framework natively orchestrates Type 4/6 loads; implementing this captures a gap in every competitor.
**Implementation**: `load_dimension` gains `scd_type: 4 | 6`. Type 4 extracts rapidly-changing columns into a `dim_<entity>_mini` satellite and replaces the foreign key on the main dimension with a surrogate FK. Type 6 maintains `current_<attr>` (Type 1 overwrite), `original_<attr>` (Type 3 original), and effective-date rows (Type 2 history). Returns `main_rows_updated`, `mini_rows_inserted`, `history_rows_created`.
**Competitor**: Kimball Group methodology; implemented natively by WhereScape, TimeXtender; absent in dbt by default.

---

### I8. Query Cost Estimation Before Execution
**Category**: Query Governance
**Justification**: A misconfigured BI report can trigger a full-scan of a 10 TB fact table, costing hundreds of dollars on cloud DWHs. Google BigQuery's "bytes billed estimate" before execution and Snowflake's `EXPLAIN` with cost estimates prevent budget shock. Exposing this as a service method lets the API layer gate expensive queries before they run.
**Implementation**: `estimate_query_cost(tenant_id, sql, explain_mode)` parses the query AST (via `sqlglot`), resolves table references to registered tables with their `size_bytes` statistics, and returns `bytes_scanned_estimate`, `slot_seconds_estimate`, `cost_usd_estimate: Decimal`, and a list of `optimisation_hints` (missing indices, full-scan warnings). Blocks execution if estimate exceeds tenant slot budget.
**Competitor**: BigQuery Job.statistics.totalBytesProcessed, Snowflake EXPLAIN, Redshift EXPLAIN ANALYZE.

---

### I9. Semantic Layer with Metric Definitions
**Category**: Business Intelligence
**Justification**: The "metric proliferation" problem — 47 definitions of "Monthly Active Users" across 12 dashboards — is the primary source of business distrust in data teams. dbt Semantic Layer, Cube.dev, and Looker's LookML solve this by making metrics first-class warehouse citizens, stored once and consumed everywhere.
**Implementation**: `define_metric(tenant_id, name, expression_sql, grain, dimensions, owner_id)` stores a metric definition referencing registered tables and columns. `compute_metric(tenant_id, metric_name, filters, time_grain)` generates and caches the aggregate SQL. `list_metrics` returns the full semantic catalogue. Metrics carry lineage refs back to source tables; changes to source tables emit `metric_stale` events.
**Competitor**: dbt Semantic Layer (MetricFlow), Cube.dev, Looker LookML, Atscale Semantic Layer.

---

### I10. Automated Index Recommendation and Management
**Category**: Query Optimisation
**Justification**: Database indices are the highest-ROI performance lever in relational DWH (Redshift sort/dist keys, PostgreSQL partial indices, Vertica projections), yet most warehouses under-index because the right columns require query workload analysis to identify. Automated index advisors like SQL Server DTA reduce query latency by 40–70 % with zero DBA involvement.
**Implementation**: `analyse_index_opportunities(tenant_id, table_name, sample_period)` reads the slow-query log from `query_performance_report`, extracts predicate and join columns, and returns `IndexRecommendation` records scored by estimated benefit. `apply_index_recommendation(tenant_id, recommendation_id)` emits DDL. `drop_unused_index(tenant_id, index_id, unused_since)` removes indices not used in N days.
**Competitor**: SQL Server Database Tuning Advisor, Redshift Advisor, pganalyze Index Advisor, EverSQL.

---

### I11. Data Freshness SLA Monitoring with Breach Alerting
**Category**: Observability
**Justification**: Stale data in dashboards destroys business trust. Modern data observability platforms (Monte Carlo, Acceldata, Datafold) track "data freshness" — the time since the last successful load — and alert before the SLA window closes. This is absent in most open-source DWH frameworks despite being the #1 data reliability complaint.
**Implementation**: `set_freshness_sla(tenant_id, table_name, max_age_minutes, alert_threshold_pct)` persists the SLA. A background coroutine (`_check_freshness_slas`) runs every minute, computes `current_age_minutes` from the latest ETL run `completed_at`, and emits `freshness_sla_breach` events when `current_age > max_age`. `get_freshness_status(tenant_id)` returns all tables with `status: fresh | stale | breached` and `age_minutes`.
**Competitor**: Monte Carlo Data Freshness, Datafold, Elementary, dbt source freshness.

---

### I12. Cross-Table Row-Level Security Policies
**Category**: Security & Governance
**Justification**: Multi-tenant warehouses need row-level security (RLS) so that `SELECT * FROM fact_sales` returns only rows belonging to the authenticated user's region/department. PostgreSQL RLS, Snowflake Row Access Policies, and BigQuery Row-Level Security are all DWH-native features. Without this, the only option is per-tenant physical table isolation, which multiplies storage costs 10x.
**Implementation**: `create_rls_policy(tenant_id, table_name, policy_sql, roles)` stores a policy that injects a `WHERE` clause based on `current_user` / session variables. `evaluate_rls(tenant_id, table_name, actor_id)` returns the effective filter clause. `list_rls_policies(tenant_id)` shows all active policies. All `load_fact` / `load_dimension` operations validate that the actor's effective policy allows write access to the target rows.
**Competitor**: PostgreSQL Row Level Security, Snowflake Row Access Policies, BigQuery Row-Level Security, Databricks Unity Catalog fine-grained access.

---

### I13. Time-Travel and Point-in-Time Query Support
**Category**: Auditability & Recovery
**Justification**: Regulatory auditors and ML feature stores require the ability to reconstruct warehouse state at any past timestamp — "what did the customer record look like on 2024-01-15?". Delta Lake time-travel, Snowflake `AT(TIMESTAMP => ...)` syntax, and Iceberg snapshots are industry-standard. Without this, answering audit questions requires restoring backups, which takes hours.
**Implementation**: `enable_time_travel(tenant_id, table_name, retention_days)` activates version-snapshot recording on each write. `query_at_timestamp(tenant_id, table_name, as_of: datetime, filters)` reconstructs the table state by replaying insert/update/delete audit events up to `as_of`. `list_snapshots(tenant_id, table_name)` returns available restore points. `restore_snapshot(tenant_id, table_name, snapshot_id)` rolls back the current state.
**Competitor**: Delta Lake time-travel, Apache Iceberg snapshots, Snowflake Time Travel, Databricks Delta time-travel.

---

### I14. Automated Slowly Changing Dimension (SCD) Backfill Engine
**Category**: Historical Data Management
**Justification**: When SCD Type 2 tracking is enabled after data already exists, the historical record is lost — every row appears as if it was always in its current state. No DWH tool handles retroactive SCD Type 2 backfilling from source system audit logs, forcing manual one-off migrations that routinely introduce data corruption.
**Implementation**: `backfill_scd2_history(tenant_id, table_name, source_audit_log, natural_key_columns, effective_date_column)` ingests a source system changelog, reconstructs the complete version history, generates `effective_from` / `effective_to` date ranges, and bulk-inserts historical rows. Returns `versions_created`, `rows_corrected`, and a `consistency_check` confirming no version gaps. Dry-run mode shows the change plan before applying.
**Competitor**: WhereScape Backfill, Fivetran History Mode, Airbyte History Mode; addressed in dbt only via custom macros.

---

### I15. Intelligent ETL Job Dependency Graph with Topological Scheduling
**Category**: ETL Orchestration
**Justification**: ETL jobs have implicit dependencies (fact table loads must wait for all dimension loads). Running jobs without respecting the dependency graph produces referential integrity violations and incorrect metrics. Apache Airflow DAGs, dbt's `ref()` model graph, and Dagster's asset graph all solve this, but require external schedulers. Embedding dependency-aware scheduling in the DWH capability eliminates the orchestration layer entirely.
**Implementation**: `add_etl_dependency(tenant_id, job_id, depends_on_job_id, dependency_type)` records a directed dependency edge. `resolve_execution_order(tenant_id)` runs Kahn's topological sort on the job dependency graph and returns a `ExecutionPlan` with `tiers` (jobs within a tier can run in parallel), `critical_path_duration_ms`, and `detected_cycles`. `run_pipeline(tenant_id, root_job_id, fail_fast)` executes the full downstream DAG tier by tier, short-circuiting on failure if `fail_fast=True`.
**Competitor**: Apache Airflow DAGs, dbt model dependencies (ref()), Dagster asset lineage, Prefect task dependencies.
