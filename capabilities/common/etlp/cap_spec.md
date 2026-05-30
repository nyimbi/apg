# ETLP Capability Package Scope

ETLP is APG's common capability for governed ETL/ELT pipeline composition. It
provides a production-oriented async runtime and a dependency-light
generated-application lifecycle surface.

## Implemented Packet

The current packet provides:

- Tenant-scoped capability contract with configuration, schema, deterministic
  rule engine, UI manifest, and theme.
- Dependency-light `ETLPLifecycleService` for generated APG applications.
- Lifecycle records for pipelines, datasources, mappings, executions, quality
  results, publish reviews, replay requests, and audit events.
- Guardrails for tenant context, owner assignment, supported modes,
  datasource approval, datasource type, embedded secrets, mapping schema
  validation, registered mapping endpoints, production approval, idempotency,
  publish quality, publish approval, lineage emission, cost review, retry
  review, replay reason, replay window, and destructive retirement review.
- FastAPI runtime controller with explicit adapter-boundary responses for
  persistence-backed operations that are not available without runtime storage.
- View models for generated application screens.
- Dynamic package entrypoint that derives semantic evidence from the current
  capability contract.

## Runtime Boundaries

The production `ETLPService` remains the async orchestration runtime for APG
deployments. It expects APG service injection for auth, audit, metadata,
notification, collaboration, connector registry, transformation, quality,
execution monitoring, and optimization services.

The dependency-light `ETLPLifecycleService` is intentionally in-memory and
side-effect free. It exists so generated applications can compose ETLP
workflows, evaluate guardrails, and render UI state without a live database,
connector registry, stream runtime, or orchestration engine.

## Adapter Requirements

Adapters must honor ETLP decisions before performing side effects:

- Execution engine.
- Connector registry.
- Metadata catalog and lineage emitter.
- Quality engine.
- Secret store.
- Bytewax event stream.
- Audit and monitoring sinks.

No adapter should execute, publish, replay, retry, backfill, or destructively
retire a pipeline when the lifecycle service returns `deny` or
`require_review`.

## UI Surfaces

ETLP exposes screens for:

- Dashboard.
- Pipeline workbench.
- Pipeline designer.
- Field mapper.
- Execution monitor.
- Quality console.
- Datasource manager.
- Schedule console.
- Publish review.
- Lineage.
- Replay/backfill.
- Adapter health.
- Audit timeline.
- Settings.

## Verification Scope

Focused packet checks cover contract shape, rule evaluation, lifecycle
guardrails, view-model composition, dynamic package evidence, import hygiene,
implementation audit, publish plan, stale-marker search, and whitespace checks.

The following are intentionally deferred:

- Full repository tests.
- Live database persistence.
- Physical connector execution.
- Bytewax flow execution.
- Rendered browser UI.
- Performance benchmarks.
