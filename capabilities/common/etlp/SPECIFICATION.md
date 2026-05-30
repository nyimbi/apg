# ETLP Capability Specification

## Purpose

ETLP provides APG with a first-class data pipeline capability. It must allow a
person or generated APG application to define, govern, execute, observe, and
compose ETL/ELT pipelines without bypassing tenant, security, quality, lineage,
and audit controls.

ETLP is not only a production pipeline runtime. It is also a generated
application building block with a deterministic lifecycle surface, rule engine,
UI manifest, theme contract, semantic evidence, and adapter boundaries.

## Capability Scope

ETLP owns:

- Pipeline definitions and lifecycle state.
- Datasource and target registration.
- Transformation and field-mapping definitions.
- Execution requests and execution state.
- Schedule, backfill, retry, and replay requests.
- Quality rules, quality assessments, and publish gates.
- Lineage requirements for transformed output.
- Cost, resource, concurrency, and production approval review gates.
- UI view models for pipeline design and operations.
- API/service helpers for generated APG applications.

ETLP integrates with:

- `meta` for dataset, pipeline, field, output, and lineage registration.
- `mdm` for governed master-data inputs and output references.
- `mqeb` for pipeline events and dead-letter/replay coordination.
- `moni` for metrics, health, alerts, and execution telemetry.
- `auth` for permissions and actor context.
- `audl` for immutable audit trails.
- `conf` for tenant configuration.
- `cach` for pipeline metadata and planning caches.

## Functional Requirements

### Pipeline Lifecycle

ETLP must support:

- Register pipeline.
- Validate pipeline dependencies and datasource references.
- Update pipeline metadata, steps, mappings, schedule, and target.
- Submit approval for production execution.
- Execute pipeline in batch, ELT, streaming, or hybrid mode.
- Pause, resume, cancel, retry, replay, and backfill executions.
- Publish pipeline output only after required quality and lineage evidence.
- Retire pipeline only after impact analysis and no running executions.

### Datasource Lifecycle

ETLP must support:

- Register source and target systems.
- Mark connectors as approved or pending review.
- Reference secrets through adapter configuration instead of storing secret
  material in pipeline records.
- Test connection health through runtime adapters.
- Track datasource ownership and tenant scope.

### Transformation Lifecycle

ETLP must support:

- Field mapping from source schema to target schema.
- Transformation steps with type, input references, output references, and
  deterministic configuration.
- Validation of source and target schema compatibility.
- Lineage event requirements for every transformation.
- Masking, filtering, enrichment, aggregation, and custom function adapters.

### Execution Lifecycle

ETLP must support:

- Execution request creation with actor, environment, trigger, estimated cost,
  approval state, and selected mode.
- Execution state transitions: queued, running, paused, completed, failed,
  cancelled, retrying, quarantined, and published.
- Log, metric, and checkpoint capture.
- Quarantine of failed records when quality gates fail.
- Idempotency controls for retry and replay.

### Quality and Publishing

ETLP must support:

- Quality-rule registration.
- Quality assessment for execution outputs.
- Minimum score thresholds for publish.
- Blocking publish when quality gate evidence is missing or failing.
- Recording matched guardrail rules and required actions.

### Rule Engine

The rule engine must be deterministic and side-effect free. It must evaluate
context and return:

- `decision`: `allow`, `deny`, or `require_review`.
- `matched_rules`: ordered rule names.
- `actions`: required actions and reasons.
- `context`: evaluated context.

Minimum guardrails:

- Tenant context required.
- Owner required before execution.
- Production approval required.
- Datasource approval required.
- Quality gate required before publish.
- Lineage required for transformations.
- Cost review required above configured threshold.
- Schedule review required for production schedules.
- Backfill and replay require reason and bounded time range.
- Retry count cannot exceed configured limit.
- Destructive delete requires no running executions and impact review.
- Secrets cannot be embedded in datasource or pipeline records.

### UI and Theming

ETLP must expose generated-application surfaces for:

- Dashboard.
- Pipeline workbench.
- Pipeline designer.
- Field mapper.
- Execution monitor.
- Quality console.
- Datasource manager.
- Schedule and backfill console.
- Lineage and publish review.
- Adapter health.
- Audit timeline.
- Settings.

The UI manifest must include route names, paths, components, nav groups,
permissions, shell target, view module, and theme requirement. The theme must
include tokens and component descriptors for pipeline status, field mapping,
execution timeline, quality gates, lineage, publish review, and adapter health.

## Non-Goals for This Packet

- Building every physical connector.
- Running a live distributed execution engine.
- Implementing Bytewax runtime flows inside the dependency-light control plane.
- Implementing AI optimization internals.
- Rendering browser UI.
- Replacing the existing production-oriented `ETLPService`.

These remain adapter-backed until the corresponding runtime packets are built.

## Acceptance Criteria

- Root `README.md`, `SPECIFICATION.md`, and `PLAN.md` exist and describe the
  practical capability.
- Stale overclaiming package language is removed from primary package docs.
- The package can be imported without runtime controller setup errors.
- The contract exposes expanded lifecycle configuration, rules, UI routes, and
  theme components.
- Generated applications can call dependency-light helpers for pipeline,
  datasource, mapping, execution, quality, publish, retry, replay, and retire
  workflows.
- Focused ETLP tests cover positive and negative guardrail paths.
- `app.py`, `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` reflect current contract-derived evidence.
- Focused compile, package tests, implementation audit, publish plan, stale
  marker search, and diff checks pass.
