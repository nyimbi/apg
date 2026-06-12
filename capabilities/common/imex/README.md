# APG IMEX - Import/Export (v2.0)

IMEX is the APG capability for governed import, export, and migration
workflows. It gives generated applications a dependency-light runtime for
building transfer jobs while preserving integration points for ETLP, CONN,
AUTH, AUDL, MONI, KEYM, ENCR, and Bytewax.
IMEX treats AI and automation agents as governed transfer participants, so
tools such as Codex, Claude Code, OpenCode, Pi, and future runtimes compose
through policy-controlled adapters instead of untracked transfer scripts.

## What It Provides

- Tenant-scoped transfer endpoints bound to CONN-managed connections.
- Schema mapping profiles with source profiling, mapping, and quality gate
  references.
- Import, export, and migration jobs with owner, checksum, format, data
  classification, and environment metadata.
- Preview validation before execution.
- Transfer runs with checkpoint, monitoring, quality, audit, replay, and
  completion state.
- Artifact publication with checksum and retention metadata.
- Review queues for destination approvals, quality reviews, capacity reviews,
  purge reviews, and owner transfer.
- First-class transfer-agent composition with supported runtimes, role
  guardrails, bounded scope, accountable owner, purpose, contribution
  disclosure, and human approval for privileged transfer roles.
- Bytewax lifecycle batch validation for endpoint, mapping, job, run, artifact,
  review, and transfer-agent mutation streams.
- Durable review evidence for generated-app governance queues.
- UI model functions for dashboards, job design, mappings, transfer monitor,
  validation, import/export workbenches, approvals, artifacts, audit, and
  settings, plus transfer-agent roster and lifecycle-batch monitor surfaces.
- **v2.0**: Adaptive batch sizing, dead-letter queue routing, column-level
  masking, idempotent replay, multi-format parallel export, schema evolution
  diffing, streaming backpressure, DAG job orchestration, data lineage capture,
  statistical anomaly detection, tenant rate limiting, webhook notifications,
  Parquet columnar pushdown, cross-tenant policy-gated sharing, and execution
  cost estimation.

## Main Files

- `SPECIFICATION.md` - functional contract and lifecycle definition.
- `PLAN.md` - implementation plan for this packet.
- `capability_contract.py` - executable configuration, rule, UI, adapter, and
  theme contract.
- `service.py` - production-grade `ImportExportService` implementation.
- `imex_runtime.py` - dependency-light runtime service for generated apps.
- `view_models.py` - screen-ready generated-app UI models.
- `models.py` - Pydantic v2 domain models.
- `app.py` - dynamic package semantic model and self-test.
- `test_capability_contract.py` and `tests/test_package_contract.py` - focused
  package proof.

## Core API

| Method | Description |
|--------|-------------|
| `create_job(job_config, created_by)` | Create and validate a new import/export job |
| `execute_job(job_id, execution_config)` | Execute job with real-time monitoring |
| `get_job_metrics(job_id)` | Real-time `ProcessingMetrics` for active or completed job |
| `detect_schema_automatically(source_config)` | Auto-detect schema from file, DB, or API source |
| `suggest_field_mappings(source_schema, target_schema)` | AI-powered field mapping suggestions with confidence scores |
| `validate_data_quality(job_id, data_sample)` | Multi-dimensional quality assessment → `DataQualityReport` |
| `create_workflow(workflow_config, created_by)` | Register a multi-step IMEX workflow |
| `execute_workflow(workflow)` | Execute a workflow and return its execution ID |
| `format_detect_auto(source_config)` | Auto-detect file format by path and content sampling |
| `large_file_stream(source_config, batch_size)` | Plan chunked streaming for large-file ingestion |
| `progress_track(job_id)` | Real-time progress dict for active or recent job |
| `partial_failure_handle(job_id, error_strategy)` | Configure `skip_and_log`, `halt`, `retry`, or `quarantine` |
| `schema_validate_import(source_config, schema)` | Validate source conforms to expected schema before import |
| `transform_preview_imex(data_sample, steps)` | Dry-run transformation steps on a sample |
| `rollback_import(job_id, rollback_reason)` | Mark an import as rolled back |
| `export_incremental(job_id, since, execution_config)` | Incremental export from a watermark timestamp |
| `import_schedule(job_id, cron_expression, scheduled_by)` | Schedule a recurring import via cron |
| `imex_analytics(tenant_id, period)` | Job counts, throughput, quality scores, error rates |
| `list_jobs(tenant_id, status_filter)` | List jobs for a tenant with optional status filter |
| `cancel_job(job_id, cancelled_by)` | Cancel a queued or running job |
| `retry_job(job_id, execution_config)` | Retry a failed or cancelled job |
| `clone_job(job_id, new_name, created_by)` | Clone an existing job configuration as a new draft |
| `pause_job(job_id, paused_by)` / `resume_job(job_id, resumed_by)` | Pause/resume active jobs |
| `export_job_config(job_id)` / `import_job_config(config, created_by)` | Portable job config backup and restore |
| `add_validation_rule(job_id, rule)` | Append a validation rule to an existing job |
| `add_transformation_step(job_id, step)` | Append a transformation step to an existing job |
| `get_execution_history(job_id, limit)` | Execution history for a job |
| `bulk_create_jobs(job_configs, created_by)` | Batch-create multiple jobs from a config list |
| `estimate_job_duration(job_id)` | Estimated runtime based on historical throughput |
| `optimize_job_performance(job_id)` | Performance recommendations (chunk size, workers) |
| `health_check()` | Service health with database, cache, and component status |

## World-Class Enhancements (v2.0)

Fifteen production-grade improvements implemented in `service.py`:

1. **Adaptive Batch Sizing** — rolling-window throughput controller that doubles or halves batch size to converge on the empirical optimum; eliminates manual tuning and prevents OOM on wide rows.
   Method: `adaptive_batch_size_controller(job_id, window_size, min_batch, max_batch)`

2. **Dead-Letter Queue (DLQ) Routing** — writes rejected records to a dedicated quarantine store (DB table or object-storage prefix) with job ID, execution ID, failure reason, and raw payload; enables selective replay.
   Method: `route_to_dlq(job_id, execution_id, failed_records, reason)`

3. **Column-Level Data Masking** — masking plan maps column names to strategies (redact, hash, partial, fake) applied as a late-stage transformation so upstream quality rules still operate on real data.
   Method: `apply_data_masking(data, masking_plan)`

4. **Checksum-Gated Idempotent Replay** — SHA-256 content-addressable checksums stored per execution detect identical payloads and skip re-import; returns a replay decision with a diff summary.
   Method: `check_and_replay_idempotent(job_id, payload_checksum)`

5. **Multi-Format Parallel Export** — single logical export emits multiple physical formats simultaneously (Parquet, CSV, JSON) using `asyncio` task groups; halves clock time for multi-consumer patterns.
   Method: `export_multi_format(job_id, formats, output_base_path, execution_config)`

6. **Schema Evolution Diff** — computes a structured diff (added fields, removed fields, type changes) when detected schema diverges from the schema captured at job creation; supports drift-threshold gating.
   Method: `diff_schema_evolution(job_id, current_schema)`

7. **Streaming Backpressure Control** — semaphore limits in-flight batches in the async generator pipeline; blocks the reader when the target is slow, keeping memory bounded without dropping records.
   Method: `stream_with_backpressure(source_config, target_config, max_inflight_batches)`

8. **Job Dependency Graph Execution** — DAG executor accepts job IDs with edge semantics (sequential, parallel, conditional) and orchestrates execution respecting declared dependencies.
   Method: `execute_job_dag(dag_config, created_by)`

9. **Data Lineage Capture** — records every field-level transformation as a lineage event (job ID, execution ID, source field, transformation applied, resulting value hash) for compliance-grade audit trails.
   Method: `capture_data_lineage(job_id, execution_id, record, transformation_log)`

10. **Anomaly Detection on Incoming Data** — z-score and IQR-bound analysis on a rolling baseline flags statistical outliers (mean shifts, future timestamps) before records reach the target.
    Method: `detect_data_anomalies(data_sample, baseline_stats, sensitivity)`

11. **Tenant-Scoped Rate Limiting** — per-tenant records-per-second and concurrent-job ceilings prevent runaway jobs from starving other tenants sharing the connection pool.
    Method: `enforce_rate_limit(tenant_id, requested_records_per_second, requested_workers)`

12. **Webhook Notification on Job Events** — signed HTTP POST dispatched to registered endpoints on configurable events (complete, fail, quality breach); HMAC-SHA256 signature for endpoint verification.
    Method: `dispatch_webhook_notification(job_id, event_type, webhook_url, secret)`

13. **Parquet Columnar Pushdown** — translates a schema mapping and optional filter expression into an optimised SQL SELECT, reducing I/O by an order of magnitude for database-to-Parquet exports.
    Method: `plan_parquet_pushdown(job_id, projected_columns, filter_expression)`

14. **Cross-Tenant Data Sharing with Policy Gate** — checks data classification and consent rules for both tenants before returning a signed share token that downstream writers validate.
    Method: `authorize_cross_tenant_share(source_tenant_id, target_tenant_id, data_classification, share_purpose)`

15. **Execution Cost Estimation** — models CPU-seconds, read bytes, write bytes, and cloud-egress cost from job config and historical performance; returns a cost envelope for budget gates.
    Method: `estimate_execution_cost(job_id, cost_model)`

## New Methods — Usage Examples

### Adaptive Batch Sizing

```python
# Let the controller tune batch size to maximise throughput during execution.
controller_result = await service.adaptive_batch_size_controller(
    job_id="crm-migration",
    window_size=10,       # rolling sample of last 10 batches
    min_batch=500,
    max_batch=50_000,
)
# Returns: {"recommended_batch_size": 12500, "throughput_trend": "increasing", ...}
```

### Dead-Letter Queue Routing

```python
# Quarantine records that failed validation rather than losing them.
dlq_result = await service.route_to_dlq(
    job_id="crm-migration",
    execution_id="exec-001",
    failed_records=[{"id": 42, "email": "bad@@example"}],
    reason="email_format_invalid",
)
# Returns: {"quarantine_id": "...", "record_count": 1, "replay_url": "..."}
```

### Schema Evolution Diff

```python
# Gate on schema drift before loading into the warehouse.
current_schema = {"fields": [
    {"name": "customer_id", "type": "integer"},
    {"name": "email", "type": "varchar"},
    {"name": "tier", "type": "varchar"},   # new field — was not in job schema
]}
diff = await service.diff_schema_evolution("crm-migration", current_schema)
if diff["added_fields"] or diff["removed_fields"] or diff["type_changes"]:
    raise RuntimeError(f"Schema drift detected: {diff}")
```

### Multi-Format Parallel Export

```python
# Emit Parquet for the data lake and CSV for the legacy ETL in one pass.
result = await service.export_multi_format(
    job_id="crm-migration",
    formats=["parquet", "csv"],
    output_base_path="/exports/crm/2026-06-12",
    execution_config={"started_by": "pipeline-bot"},
)
# Both files written concurrently; result contains per-format metrics.
```

### Execution Cost Estimation

```python
# Evaluate budget impact before submitting a large job to the scheduler.
cost = await service.estimate_execution_cost(
    job_id="crm-migration",
    cost_model={
        "cpu_cost_per_second": 0.00004,
        "egress_cost_per_gb": 0.09,
        "storage_cost_per_gb": 0.023,
    },
)
# Returns: {"estimated_cpu_seconds": 420, "estimated_egress_gb": 1.2,
#            "estimated_total_cost_usd": 0.13, "confidence": "medium"}
```

## Generated-App Usage

```python
from capabilities.common.imex import ImexService

service = ImexService()
service.register_endpoint(
	"source-crm",
	"tenant-a",
	"CRM Export",
	"connection",
	"conn://crm",
	"data",
)
service.register_endpoint(
	"warehouse",
	"tenant-a",
	"Warehouse",
	"connection",
	"conn://warehouse",
	"data",
)
service.create_mapping_profile(
	"crm-map",
	"tenant-a",
	"CRM Mapping",
	"profiles/crm.json",
	"mappings/crm_to_wh.json",
	"quality/crm",
)
service.create_job(
	"crm-migration",
	"tenant-a",
	"CRM Migration",
	"migration",
	"source-crm",
	"warehouse",
	"parquet",
	"data",
	"production",
	"crm-map",
	"sha256:abc",
	etlp_plan_ref="etlp://crm-migration",
)
service.validate_preview("tenant-a", "crm-migration", quality_score=0.99)
run = service.execute_job(
	"tenant-a",
	"crm-migration",
	"run-001",
	record_count=50000,
	approval_recorded=True,
)
service.complete_run("tenant-a", run["id"], records_processed=50000, quality_score=0.99)
agent = service.register_transfer_agent(
	"migration-agent",
	"tenant-a",
	"Migration Agent",
	"codex",
	"migration_reviewer",
	"crm migration review",
	"integration-office",
	"review migration transfer evidence",
	human_approval_required=True,
)
batch = service.validate_imex_lifecycle_batch("tenant-a", "bytewax", 4)
assert agent["status"] == "active"
assert batch["status"] == "accepted"
```

## Review Evidence

Every generated-app lifecycle record carries `policy_decision`,
`matched_rules`, `review_reasons`, and `review_evidence` fields so generated
transfer consoles can render why an endpoint, mapping, job, run, artifact,
review, transfer-agent registration, or lifecycle batch is allowed, denied, or
awaiting review. `list_pending_reviews()` returns the composed queue across
all entity types. Denied non-Bytewax lifecycle batches are stored with
`status="denied"` and `required_processor="bytewax"` before the guardrail
raises `PermissionError`.

## Guardrails

IMEX blocks missing tenant context, unsupported formats, missing endpoints,
missing mappings, missing checksums, missing preview validation, unapproved
production transfers, unencrypted sensitive exports, unmonitored large
transfers, missing checkpoints, invalid records without quarantine, replay
without idempotency, artifact publication without retention, and destructive
purge without review.
It also blocks unsupported transfer-agent runtimes, unsupported agent roles,
missing agent scope, missing owner, missing purpose, missing contribution
disclosure, and non-Bytewax lifecycle batches. Privileged transfer-agent roles
require human approval evidence before mutation.

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/imex/__init__.py capabilities/common/imex/capability_contract.py capabilities/common/imex/imex_runtime.py capabilities/common/imex/api.py capabilities/common/imex/view_models.py capabilities/common/imex/app.py capabilities/common/imex/test_capability_contract.py capabilities/common/imex/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/imex/test_capability_contract.py capabilities/common/imex/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/imex --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/imex --strict --json
./.venv/bin/apg capabilities publish-plan capabilities/common/imex --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/common/imex --json
```

---

© 2025 Datacraft — Nyimbi Odero | www.datacraft.co.ke
