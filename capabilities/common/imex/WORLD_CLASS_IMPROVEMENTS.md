# IMEX World-Class Improvements

**Capability**: Import/Export (imex)
**Author**: Nyimbi Odero — Datacraft © 2026
**Date**: 2026-06-11

---

## 1. Adaptive Batch Sizing

Current batch sizes are configured statically. Real throughput depends on network latency, row width, and target ingestion rate. An adaptive controller should measure rolling throughput over a sliding window, then double or halve the batch size to converge on the empirical optimum. This eliminates manual tuning for each connector and prevents OOM spikes on wide rows.

**Method**: `adaptive_batch_size_controller(job_id, window_size, min_batch, max_batch)`

---

## 2. Dead-Letter Queue (DLQ) Routing

Failed records currently accumulate in an in-memory error list and are lost when the process exits. A DLQ router writes rejected records to a dedicated quarantine store (a separate DB table or object-storage prefix) with the originating job ID, execution ID, failure reason, and raw payload. Downstream remediation workflows can replay DLQ entries selectively.

**Method**: `route_to_dlq(job_id, execution_id, failed_records, reason)`

---

## 3. Column-Level Data Masking

Exports to external or lower-trust environments should mask or tokenise sensitive fields (PII, financial) before the payload leaves the platform. A masking plan maps column names to masking strategies (redact, hash, partial, fake) and is applied as a late-stage transformation step so upstream quality rules still operate on real data.

**Method**: `apply_data_masking(data, masking_plan)`

---

## 4. Checksum-Gated Idempotent Replay

Re-running an import that has already succeeded should be a no-op, not a duplicate insert. Content-addressable checksums (SHA-256 of the sorted, serialised payload) stored per execution allow the engine to detect and skip identical payloads. The method exposes the checksum registry and returns a replay decision with a diff summary.

**Method**: `check_and_replay_idempotent(job_id, payload_checksum)`

---

## 5. Multi-Format Parallel Export

A single logical export job should be able to emit multiple physical formats simultaneously (e.g. Parquet for analytics, CSV for legacy systems, JSON for APIs) using asyncio task groups. This halves the clock time for multi-consumer export patterns and removes orchestration overhead from callers.

**Method**: `export_multi_format(job_id, formats, output_base_path, execution_config)`

---

## 6. Schema Evolution Diff

When the detected schema of a recurring import differs from the schema captured during job creation, the system should compute a structured diff (added fields, removed fields, type changes) and surface it before execution. Callers can gate on drift thresholds to avoid silently loading mismatched data.

**Method**: `diff_schema_evolution(job_id, current_schema)`

---

## 7. Streaming Backpressure Control

High-throughput imports can overwhelm the target writer. A backpressure controller wraps the async generator pipeline with a semaphore that limits the number of in-flight batches. When the target is slow the semaphore blocks the reader, keeping memory bounded without dropping records.

**Method**: `stream_with_backpressure(source_config, target_config, max_inflight_batches)`

---

## 8. Job Dependency Graph Execution

Complex migration pipelines involve jobs that must run in sequence or fan-out in parallel. A dependency graph executor accepts a DAG of job IDs with edge semantics (sequential, parallel, conditional on prior success/failure) and orchestrates execution while respecting declared dependencies.

**Method**: `execute_job_dag(dag_config, created_by)`

---

## 9. Data Lineage Capture

Every transformation applied to every field should be recorded as a lineage event that can be replayed to reconstruct the provenance of a target record. Lineage events include the job ID, execution ID, source field, applied transformation, and resulting value hash. This enables compliance-grade audit trails beyond simple row counts.

**Method**: `capture_data_lineage(job_id, execution_id, record, transformation_log)`

---

## 10. Anomaly Detection on Incoming Data

Statistical outliers in imported data (e.g. a numeric column with a sudden shift in mean or a date column with future timestamps) should be flagged before the records reach the target. An anomaly detector computes z-scores and IQR bounds on a rolling baseline and surfaces flagged records with their anomaly scores.

**Method**: `detect_data_anomalies(data_sample, baseline_stats, sensitivity)`

---

## 11. Tenant-Scoped Rate Limiting

Without per-tenant quotas, a single runaway job can starve other tenants sharing the same database connection pool. A rate limiter enforces configurable records-per-second and concurrent-job ceilings per tenant, returning a `RateLimitResult` that callers use to decide whether to queue or reject new submissions.

**Method**: `enforce_rate_limit(tenant_id, requested_records_per_second, requested_workers)`

---

## 12. Webhook Notification on Job Events

Operators need real-time signals when jobs complete, fail, or breach quality thresholds. A webhook dispatcher fires signed HTTP POST payloads to registered endpoints on configurable events. The payload includes the job state snapshot and a HMAC-SHA256 signature for endpoint verification.

**Method**: `dispatch_webhook_notification(job_id, event_type, webhook_url, secret)`

---

## 13. Parquet Columnar Pushdown

When exporting from a database to Parquet, fetching only the projected columns and pushing WHERE predicates to the database layer can reduce I/O by an order of magnitude. A pushdown planner translates a schema mapping and optional filter expression into an optimised SQL SELECT and returns a Parquet write plan.

**Method**: `plan_parquet_pushdown(job_id, projected_columns, filter_expression)`

---

## 14. Cross-Tenant Data Sharing with Policy Gate

Data pipelines sometimes need to share anonymised or aggregated records across tenant boundaries. A policy gate checks whether a cross-tenant share is permitted under the data classification and consent rules registered for both tenants before returning a signed share token that downstream writers validate.

**Method**: `authorize_cross_tenant_share(source_tenant_id, target_tenant_id, data_classification, share_purpose)`

---

## 15. Execution Cost Estimation

Before submitting a large job to the scheduler, operators want to know the compute and I/O cost. A cost estimator models CPU-seconds, read bytes, write bytes, and estimated cloud-egress cost from job configuration and historical performance data, returning a cost envelope that budget gates can evaluate.

**Method**: `estimate_execution_cost(job_id, cost_model)`
