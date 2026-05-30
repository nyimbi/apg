# LOGT Logging and Tracing Specification

## Purpose

LOGT is APG's common logging and tracing capability. It lets generated and
composed applications create diagnostic pipelines, ingest structured logs,
capture distributed traces, record spans, search diagnostics, approve exports,
retain diagnostic evidence, and operate observability workflows through APG UI
and API surfaces.

The capability is designed for executable applications first. It provides a
dependency-light runtime and explicit adapter boundaries so production systems
can connect real collectors, search engines, monitoring backends, compliance
exporters, audit stores, and Bytewax workers later.

## Capability Identity

- Capability id: `logt`
- Display name: `Logging and Tracing`
- Category: `common`
- Runtime target: `python`
- Primary service: `service.LogtService`
- UI prefix: `/logt`
- API prefix: `/logt/api/v1`
- Event stream processor: `bytewax`

## Provided Services

- `structured_logging`
- `distributed_tracing`
- `trace_correlation`
- `log_search`
- `diagnostic_retention`
- `diagnostic_exports`
- `logt_agents`

## Required Capabilities

- `moni` for monitoring integration.
- `conf` for tenant configuration and diagnostic policy.
- `audl` for durable audit evidence.

Optional adapters include `srch`, `anom`, and `comp`.

## Domain Model

`IngestionPipeline`

- Tenant-local pipeline id, name, accountable owner, schema reference, Bytewax
  event stream, sampling policy, retention policy, lifecycle status, and
  creation time.

`LogEvent`

- Structured log with tenant, pipeline, service, severity, redacted message,
  attributes, trace id, span id, sensitive-content flag, redaction flag, and
  timestamp.

`TraceRecord`

- Distributed trace root with tenant, pipeline, trace id, root service,
  operation, trace context, sampling policy, lifecycle status, and start time.

`SpanRecord`

- Trace span with tenant, trace id, span id, optional parent span, service,
  operation, duration, status, attributes, and timestamp.

`DiagnosticQuery`

- Audited diagnostic query with tenant, query text, requester, time window,
  review status, result count, and completion status.

`DiagnosticExport`

- Approved diagnostic export bundle with tenant, type, requester, approval
  reference, item ids, status, and creation time.

`RetentionPolicy`

- Tenant retention and privacy policy for logs, spans, redaction, and export
  approval.

`LogtAuditEvent`

- Governance record for diagnostic lifecycle actions.

`LogtAgent`

- Registered AI observability agent with tenant, runtime, role, explicit scope,
  registration status, contribution disclosure, and activity state.

## Rule Engine

The deterministic rule engine must enforce:

- tenant context on every diagnostic operation;
- accountable pipeline owner;
- pipeline schema reference;
- Bytewax event stream for ingestion pipelines;
- pipeline sampling policy;
- trace context;
- trace identifier;
- service identity on spans;
- non-negative span duration;
- redaction for sensitive logs;
- service identity on logs;
- requester identity on diagnostic queries;
- review for large diagnostic queries;
- approval for diagnostic exports;
- approval reference for diagnostic exports;
- registered AI observability agent;
- supported AI-agent runtime;
- supported AI-agent role;
- explicit AI-agent scope;
- AI contribution disclosure;
- audit evidence for lifecycle state changes;
- Bytewax event stream for batch diagnostic mutation.

## UI Contract

The capability exposes these APG Python UI routes:

- `/logt/dashboard`
- `/logt/logs`
- `/logt/traces`
- `/logt/spans`
- `/logt/pipelines`
- `/logt/retention`
- `/logt/agents`
- `/logt/analytics`
- `/logt/audit`
- `/logt/settings`

View models must expose diagnostic summaries, pipelines, retention policies,
logs, traces, spans, queries, exports, service maps, slow spans, error logs,
observability agents, rules, audit events, theme data, and Bytewax stream
metadata.

## Theme

The default theme is `logt_observability_console`. Theme components cover trace
waterfalls, structured log tables, pipeline graphs, retention panels, agent
panels, and audit timelines.

## Event Stream

Lifecycle telemetry is described by:

- processor: `bytewax`
- topic: `apg.logt.lifecycle`
- state: pipelines, logs, traces, spans, queries, exports, retention policies,
  LOGT agents, audit events
- events: pipeline created, log ingested, trace ingested, span recorded, query
  executed, export created, agent registered
- guardrail: `batch_diagnostic_mutation_requires_bytewax`

The package records stream metadata and guardrails. Live Bytewax topology is
an application deployment concern.

## Acceptance Criteria

- The package has local README, specification, plan, contract, runtime, view,
  API, test, package-manifest, semantic-model, and release-report artifacts.
- The contract exposes provides/requires, configuration schema, rules, routes,
  theme, and Bytewax stream metadata.
- The service supports pipeline creation, retention policy, log ingestion,
  trace ingestion, span recording, diagnostic search, approved export,
  AI-agent registration, audit events, tenant-local IDs, and Bytewax batch
  mutation validation.
- Focused tests prove the main lifecycle, guardrails, tenant isolation,
  generated evidence, and docs.
- Compile, focused pytest, implementation-audit, publish-plan, marker scan,
  and diff checks pass before commit.
