# Logging and Tracing Capability Specification

- **Capability Name**: Logging and Tracing
- **Capability ID**: `logt`
- **Category**: common
- **Version**: 1.0.0

## Purpose

LOGT is APG's package-backed observability runtime for tenant-scoped structured
logs, distributed traces, spans, diagnostic queries, exports, retention
policies, redaction, audit events, UI route metadata, theme metadata, rule
evaluation, semantic-model publication, and publish-plan evidence.

The package is dependency-light and deterministic. Live OpenTelemetry
collectors, event buses, object stores, search indexes, monitoring systems,
alerting backends, and external audit stores remain adapter boundaries until a
future slice wires and verifies them directly.

## Provided Services

- `structured_logging`
- `distributed_tracing`
- `trace_correlation`
- `log_search`
- `diagnostic_retention`
- `logt_operations`

## Required Services

- `tenant_context`
- `moni` for monitoring integration
- `mqeb` for event-bus ingestion adapters
- `conf` for configuration policy
- optional `audl`, `srch`, `anom`, and `comp` adapters for audit, search,
  anomaly, and compliance integration

## Runtime Surfaces

| File | Runtime responsibility |
| --- | --- |
| `models.py` | Pipeline, log, trace, span, query, export, retention, and audit dataclasses |
| `observability_runtime.py` | Deterministic IDs, severity normalization, redaction, span posture, query posture, service-map, and log matching helpers |
| `service.py` | Tenant-aware pipeline management, log ingestion, trace ingestion, span recording, search, export, retention, summaries, service maps, and guardrails |
| `api.py` | Dependency-light API helper functions over the service |
| `views.py` | Dashboard, log search, trace explorer, pipeline manager, retention center, and analytics view models |
| `app.py` | Publishable APG package entrypoint and semantic-model evidence |

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Runtime operations require tenant context.
Diagnostic behavior is governed by the `ingestion`, `tracing`, `privacy`,
`governance`, `ui`, and `theme` configuration sections.

## Rules

LOGT evaluates the deterministic rules from the capability contract:

- `tenant_context_required`
- `pipeline_requires_owner`
- `trace_context_required`
- `sensitive_log_requires_redaction`
- `log_export_requires_approval`
- `large_query_requires_review`

The service enforces these rules directly. Missing tenant context, missing
pipeline owner, missing trace context, unredacted sensitive logs, unapproved
exports, and unreviewed large diagnostic queries are blocked or require review.

## UI And Theme

The package exposes eight APG Python UI routes:

- `/logt/dashboard`
- `/logt/logs`
- `/logt/traces`
- `/logt/spans`
- `/logt/pipelines`
- `/logt/retention`
- `/logt/analytics`
- `/logt/settings`

View helpers expose diagnostic summaries, pipelines, retention policies, logs,
traces, spans, service maps, queries, exports, audit events, slow spans, error
logs, rules, and theme metadata. The package uses the
`logt_observability_console` theme contract with trace-waterfall, log-table,
pipeline-graph, and retention-panel component tokens.

## Adapter Boundaries

This package intentionally does not open network connections or require a live
observability backend. Production deployments should attach adapters for:

- OpenTelemetry collector ingestion;
- MQEB/event-bus delivery;
- object-store export bundles;
- search index persistence;
- monitoring and alert routing;
- audit-log persistence;
- anomaly detection;
- compliance retention attestations.

The in-process service remains the executable APG behavior used by generated
apps, tests, publish-plan checks, and local capacity slices.

## Focused Verification

Use battery-conscious verification for this package:

```bash
./.venv/bin/python -m py_compile capabilities/common/logt/__init__.py capabilities/common/logt/models.py capabilities/common/logt/observability_runtime.py capabilities/common/logt/service.py capabilities/common/logt/api.py capabilities/common/logt/views.py capabilities/common/logt/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/logt/test_capability_contract.py capabilities/common/logt/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/logt --json
./.venv/bin/apg capabilities publish-plan capabilities/common/logt --json
```
