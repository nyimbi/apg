# ETLP - ETL/ELT Processing

ETLP is APG's tenant-scoped data pipeline capability. It gives generated APG
applications a composable control plane for pipeline design, datasource
registration, field mapping, execution, quality gates, lineage emission,
publishing, monitoring, and operational guardrails.

The capability exposes an executable contract, FastAPI runtime controller,
async pipeline service (45+ methods), Pydantic data models, field-mapping
support, view request/response models, package evidence, and focused regression
tests.

## What ETLP Provides

- Pipeline registration, update, deletion, execution, monitoring, and
  cancellation.
- Datasource and transformation management for batch, streaming, CDC, micro-batch,
  and ELT workloads.
- Field mapping helpers for source-to-target schema mapping with suggestion support.
- Quality-rule definitions and configurable quality-gate enforcement before output
  publication.
- Schema evolution tracking with backward-compatibility enforcement.
- Watermark management for incremental and CDC-based pipeline runs.
- Change Data Capture (CDC) event recording with LSN tracking.
- Partition strategy definitions with configurable retention.
- SLA monitors with failure-rate thresholds and alert routing.
- Compliance posture checks (SOX and extensible frameworks).
- Bulk pipeline creation and parallel pipeline execution.
- CSV and JSON export for all collections.
- Dashboard summary and health check endpoints.
- Analytics aggregation: failure rates, quality scores, lineage coverage.
- Lineage and audit integration points for META, AUDL, MONI, MQEB, AUTH, and MDM.
- Deterministic policy rules for tenant context, owner assignment, production
  approval, quality gates, lineage emission, and cost review.
- First-class pipeline-agent registration for Codex, Claude Code, opencode, Pi,
  and future APG-compatible runtimes.
- Pipeline-agent guardrails: supported roles, declared scope, owner, purpose,
  machine-contribution disclosure, and human approval for privileged roles.
- Durable review evidence for pipelines, datasources, mappings, executions,
  quality results, schedules, publish reviews, replay requests, privileged
  pipeline agents, denied lifecycle batches, and audit events.
- Bytewax lifecycle batch validation for pipeline, datasource, mapping,
  execution, quality, publish, replay, and pipeline-agent streams.
- UI routes for dashboard, workbench, designer, field mapper, executions,
  quality, datasources, pipeline-agent roster, lifecycle-batch monitor, and
  settings.
- Theme tokens and component contracts for generated application shells.

## Main Files

| File | Purpose |
|------|---------|
| `capability_contract.py` | ETLP configuration, rule engine, UI manifest, and theme contract |
| `models.py` | Pipeline, execution, datasource, transformation, quality-rule, schedule, and metric models |
| `service.py` | Async pipeline orchestration service — 45+ methods |
| `api.py` | FastAPI controller and route registration |
| `field_mapper.py` | Field schema, mapping configuration, suggestion, and transformation helpers |
| `views.py` | API request and response models |
| `app.py` | Publishable package entrypoint and semantic evidence |
| `SPECIFICATION.md` | Functional contract for the coherent ETLP packet |
| `PLAN.md` | Build sequence for remaining lifecycle work |

## World-Class Enhancements (v2.0)

The following improvements are planned or in-progress for ETLP. Items here are
not claims of implemented behavior unless also present in contract, service,
tests, and package evidence.

### Near-Term Packet Improvements

1. **Durable persistence adapter** — lifecycle records persisted beyond in-memory store.
2. **Datasource approval workflow** — integrated with AUTH and AUDL.
3. **Bytewax flow adapter** — adapter for streaming executions.
4. **Metadata and lineage adapter** — integration with META capability.
5. **Quality profiling adapter** — configurable quality dimensions.
6. **Rendered generated-app screens** — backed by `view_models.py`.
7. **Operational consoles** — schedule, retry, replay, and backfill management UIs.
8. **Secret-store adapter checks** — enforced on datasource definitions.

### Runtime Improvements

9. **Connector registry plugin interface** — pluggable connector ecosystem.
10. **Execution checkpoint and resume protocol** — fault-tolerant long-running pipelines.
11. **Dead-letter and quarantine handling** — routed through MQEB.
12. **Execution health metrics** — streamed through MONI.
13. **Cost-estimation adapter** — pre-execution pipeline plan costing.
14. **Backpressure and concurrency control** — runtime flow management.

### AI-Assisted Improvements

15. **Mapping suggestions** — from schema and sample data profiles.
16. **Pipeline plan linting** — automated pre-execution plan review.
17. **Failure classification and remediation** — AI-assisted root-cause suggestions.
18. **Quality-rule recommendations** — generated from data profiles.
19. **Cost and resource optimization recommendations** — AI-assisted tuning.

AI-assisted behavior remains behind adapters and does not bypass deterministic
ETLP guardrails.

## New Methods

### Schema Evolution

Detect and enforce backward compatibility on schema changes:

```python
from capabilities.common.etlp.service import ETLPService

svc = ETLPService(actor_id="pipeline-owner", tenant_id="tenant-data")
await svc.pipeline_design("tenant-data", "orders-pipeline", "Orders ETL", "elt", "data-team")

evolution = await svc.schema_evolution(
    tenant_id="tenant-data",
    evolution_id="evo-001",
    pipeline_id="orders-pipeline",
    old_schema={"fields": {"id": "int", "amount": "float"}},
    new_schema={"fields": {"id": "int", "amount": "float", "currency": "str"}},
    migration_strategy="backward_compatible",
)
# evolution["status"] == "applied", evolution["fields_added"] == ["currency"]
```

### CDC Capture

Record Change Data Capture events with LSN tracking:

```python
cdc = await svc.cdc_capture(
    tenant_id="tenant-data",
    capture_id="cdc-001",
    pipeline_id="orders-pipeline",
    source_id="pg-primary",
    table_name="orders",
    operation="update",
    row_data={"id": 42, "amount": 99.99},
    lsn="0/1A2B3C",
)
# cdc["operation"] == "update"
```

### SLA Monitor + SLA Check

Register SLA thresholds and evaluate compliance:

```python
await svc.sla_monitor(
    tenant_id="tenant-data",
    sla_id="sla-orders",
    pipeline_id="orders-pipeline",
    max_duration_minutes=30,
    max_failure_rate_percent=5.0,
    alert_recipient="ops-team",
)

result = await svc.sla_check("tenant-data", "orders-pipeline")
# result["compliant"] == True, result["violations"] == []
```

### Bulk Pipeline Operations

Create and run multiple pipelines concurrently:

```python
results = await svc.bulk_create_pipelines("tenant-data", [
    {"pipeline_id": "pipe-a", "name": "Pipeline A", "mode": "batch", "owner": "team-a"},
    {"pipeline_id": "pipe-b", "name": "Pipeline B", "mode": "elt",   "owner": "team-b"},
])

run_results = await svc.bulk_run_pipelines(
    "tenant-data", ["pipe-a", "pipe-b"], "development", "ci-system"
)
```

### Compliance Check

Evaluate pipeline compliance posture across the tenant:

```python
report = await svc.compliance_check("tenant-data", framework="SOX")
# report["passed"] — True if no lineage gaps or quality failures
# report["lineage_coverage_percent"] — % of active pipelines with lineage records
# report["issues"] — list of named violations
```

## ETLPService Method Reference

| # | Method | Description |
|---|--------|-------------|
| 1 | `pipeline_design` | Register a new pipeline |
| 2 | `source_connect` | Register a source datasource |
| 3 | `target_connect` | Register a target datasource |
| 4 | `transform_rule` | Define a transformation rule |
| 5 | `run_pipeline` | Execute a pipeline |
| 6 | `schedule_pipeline` | Schedule recurring execution |
| 7 | `monitor_pipeline` | Fetch execution status and quality metrics |
| 8 | `data_quality_gate` | Assess quality and record gate pass/fail |
| 9 | `schema_evolution` | Record and validate schema change events |
| 10 | `partition_strategy` | Define partitioning and retention policy |
| 11 | `watermark_management` | Set incremental run watermarks |
| 12 | `cdc_capture` | Record CDC events with LSN |
| 13 | `lineage_track` | Record execution data lineage |
| 14 | `sla_monitor` | Register SLA thresholds |
| 15 | `etl_analytics` | Aggregate tenant-level ETL analytics |
| 16 | `pipeline_validate` | Validate pipeline configuration pre-execution |
| 17 | `execution_complete` | Mark execution complete with final metrics |
| 18 | `pipeline_pause` | Prevent new executions |
| 19 | `pipeline_resume` | Re-enable a paused pipeline |
| 20 | `pipeline_retire` | Retire a pipeline permanently |
| 21 | `register_mapping` | Register field-level source-to-target mapping |
| 22 | `publish_output` | Publish output after quality gate |
| 23 | `retry_execution` | Retry a failed execution (max retries enforced) |
| 24 | `cancel_execution` | Cancel a running or queued execution |
| 25 | `replay_execution` | Replay a past execution within a time window |
| 26 | `register_pipeline_agent` | Register a pipeline automation agent |
| 27 | `bulk_create_pipelines` | Parallel bulk pipeline creation |
| 28 | `bulk_run_pipelines` | Parallel bulk pipeline execution |
| 29 | `compliance_check` | Evaluate SOX/framework compliance posture |
| 30 | `dashboard_summary` | Tenant-level operational dashboard |
| 31 | `health_check` | Service liveness probe |
| 32 | `export_csv` | Export any collection to CSV |
| 33 | `export_json` | Export any collection to JSON |
| 34–44 | `list_*` | Per-collection listing helpers (pipelines, sources, targets, mappings, executions, schedules, quality, lineage, cdc, agents, audit) |
| 45 | `sla_check` | Evaluate SLA compliance for a pipeline |

## Using the Capability Contract

```python
from capabilities.common.etlp import register_capability
from capabilities.common.etlp.capability_contract import evaluate_capability_rules

registration = register_capability()

decision = evaluate_capability_rules({
    "tenant_context_present": True,
    "operation": "publish_output",
    "quality_gate_passed": False,
})

assert decision["decision"] == "deny"
```

## Using the Service

```python
from capabilities.common.etlp.service import ETLPService

svc = ETLPService(actor_id="pipeline-owner", tenant_id="tenant-data")

pipeline = await svc.pipeline_design(
    tenant_id="tenant-data",
    pipeline_id="customer-sync",
    name="Customer sync",
    mode="elt",
    owner="pipeline-owner",
)

agent = await svc.register_pipeline_agent(
    tenant_id="tenant-data",
    agent_id="publish-reviewer",
    name="Publish Reviewer",
    runtime="codex",
    role="publish_gate_reviewer",
    scope="customer publish readiness",
    owner="pipeline-office",
    purpose="review transformed output before publication",
    human_approval_required=True,
)

assert pipeline["status"] == "draft"
assert agent["runtime"] == "codex"
```

`ETLPLifecycleService` is a backward-compatible alias for `ETLPService`.

## Durable Review Evidence

ETLP preserves policy evidence directly on lifecycle records so operators can
compose review queues without replaying transient exceptions. Reviewable records
include pipelines, datasources, mappings, executions, quality results, schedules,
publish reviews, replay requests, pipeline agents, lifecycle batches, and audit
events.

Each governed record exposes:

- `policy_decision`
- `matched_rules`
- `review_reasons`
- `review_evidence`

Privileged pipeline agents without human approval are registered as
`pending_review` when runtime, role, owner, scope, purpose, and contribution
disclosure are otherwise valid. Invalid runtimes, unsupported roles, missing
owner/scope/purpose, and missing contribution disclosure remain blocking denials.

Denied non-Bytewax lifecycle batches are persisted as `denied` records before
`PermissionError` is raised, giving operators durable remediation evidence.

## Guardrails

ETLP guardrails protect:

- Tenant isolation for every pipeline and execution operation.
- Owner assignment before pipeline execution.
- Explicit production approval before production runs.
- Quality-gate evidence before publishing pipeline output.
- Lineage emission for transformations.
- Cost review for high-estimate executions.
- Datasource approval, secret handling, retry policy, schedule review,
  backfill, replay, and destructive-delete controls.
- Pipeline-agent runtime, role, scope, owner, purpose, contribution disclosure,
  and privileged-role human approval or pending review.
- Bytewax lifecycle-batch validation.

## Adapter Boundaries

ETLP does not hardcode external engines into the control plane. Durable
execution engines, connector registries, Bytewax stream flows, metadata stores,
lineage emitters, quality profilers, AI optimizers, and observability sinks are
configured as adapters that receive guardrail decisions from the capability.

The ETLP packet does not embed SDK clients for Codex, Claude Code, opencode, Pi,
or future agent providers. Those runtimes connect through adapters that preserve
the APG contract, guardrail decisions, audit events, and human-approval
requirements.

## Focused Verification

```bash
./.venv/bin/python -m py_compile \
  capabilities/common/etlp/__init__.py \
  capabilities/common/etlp/capability_contract.py \
  capabilities/common/etlp/models.py \
  capabilities/common/etlp/service.py \
  capabilities/common/etlp/api.py \
  capabilities/common/etlp/field_mapper.py \
  capabilities/common/etlp/views.py \
  capabilities/common/etlp/app.py

./.venv/bin/pytest -q \
  capabilities/common/etlp/test_capability_contract.py \
  capabilities/common/etlp/tests/test_package_contract.py

./.venv/bin/apg capabilities implementation-audit --root capabilities/common/etlp --json
./.venv/bin/apg capabilities publish-plan capabilities/common/etlp --json
```

Full repository tests, live connector execution, Bytewax runtime flows, rendered
UI checks, and performance benchmarks are deferred to later verification passes.
