# LOGT Logging and Tracing Capability

LOGT provides APG applications with a tenant-scoped observability runtime:
structured log ingestion, distributed trace roots, span recording, diagnostic
search, approved diagnostic exports, retention policy, audit evidence,
observability agents, UI metadata, theme tokens, and Bytewax-backed lifecycle
events.

The package stays dependency-light. Production collectors, search indexes,
monitoring backends, compliance exporters, audit stores, and Bytewax workers
are represented as APG adapters in the executable contract and are bound by the
host application.

## What It Provides

- Structured log ingestion with tenant, pipeline, service, severity, trace,
  span, attribute, redaction, and privacy metadata.
- Distributed trace and span records with trace context, service ownership,
  duration validation, slow-span posture, and service-map summaries.
- Pipeline lifecycle with accountable owners, schema references, sampling
  policy, retention policy, and Bytewax stream enforcement.
- Diagnostic search with requester identity, large-query review, result
  counting, and audit evidence.
- Approved diagnostic exports for incident, compliance, and review bundles.
- First-class AI observability agents with runtime, role, scope,
  registration, and contribution-disclosure guardrails.
- UI route, API, view-model, theme, semantic-model, package-manifest, and
  release-report evidence.

## Main Files

- `SPECIFICATION.md` defines the normative capability behavior.
- `PLAN.md` records the implementation packet plan.
- `capability_contract.py` is the executable source of configuration, rules,
  routes, theme, adapters, provides/requires, and Bytewax stream metadata.
- `models.py` defines tenant-scoped pipelines, logs, traces, spans, queries,
  exports, retention policies, audit events, and agents.
- `observability_runtime.py` contains deterministic IDs, redaction, severity,
  query, span, service-map, and matching helpers.
- `service.py` implements the runtime facade.
- `api.py` exposes package-safe helper functions.
- `views.py` exposes UI view models.
- `test_capability_contract.py` proves lifecycle behavior and generated
  evidence.

## Basic Usage

```python
from capabilities.common.logt import LogtService

service = LogtService()
service.create_retention_policy(
    policy_id="retention-main",
    tenant_id="tenant-demo",
    name="Main diagnostics retention",
    log_retention_days=30,
)
service.create_pipeline(
    pipeline_id="pipeline-main",
    tenant_id="tenant-demo",
    name="Main diagnostics pipeline",
    owner="sre-team",
    schema_ref="schema://logs/v1",
    event_bus_ref="bytewax://diagnostics",
    sampling_policy="head-based-10pct",
    retention_policy_id="retention-main",
)
service.ingest_log(
    log_id="log-1",
    tenant_id="tenant-demo",
    pipeline_id="pipeline-main",
    service_name="orders-api",
    severity="info",
    message="order created",
)
```

## AI Observability Agents

Register AI agents before they assist with diagnostic operations:

```python
agent = service.register_logt_agent(
    tenant_id="tenant-demo",
    name="Incident reviewer",
    runtime="codex",
    role="incident_reviewer",
    scope="Review slow spans and error logs before incident export",
)
```

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`.
Supported roles cover pipeline, log, trace, incident, privacy, and retention
review.

## Composition

LOGT composes with:

- `moni` for monitoring integration and operational dashboards.
- `conf` for tenant configuration and policy.
- `audl` for durable audit evidence.
- `srch` for persistent diagnostic search indexes.
- `anom` for anomaly detection over spans, traces, and logs.
- `comp` for compliance export and retention attestations.

Batch diagnostic mutation and ingestion pipelines must use the `bytewax`
event-stream adapter.

## Verification

Focused verification for this packet:

```bash
./.venv/bin/python -m py_compile capabilities/common/logt/__init__.py capabilities/common/logt/capability_contract.py capabilities/common/logt/models.py capabilities/common/logt/observability_runtime.py capabilities/common/logt/service.py capabilities/common/logt/api.py capabilities/common/logt/views.py capabilities/common/logt/app.py capabilities/common/logt/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/logt/test_capability_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/logt --json
./.venv/bin/apg capabilities publish-plan capabilities/common/logt --json
```

Live collectors, search engines, monitoring backends, durable audit stores,
rendered UI, and Bytewax workers are integration concerns outside the package
proof.
