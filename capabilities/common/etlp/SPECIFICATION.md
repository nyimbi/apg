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
- First-class pipeline-agent registration and governance.
- Bytewax lifecycle batch validation for pipeline mutation streams.

ETLP integrates with:

- `meta` for dataset, pipeline, field, output, and lineage registration.
- `mdm` for governed master-data inputs and output references.
- `mqeb` for pipeline events and dead-letter/replay coordination.
- `moni` for metrics, health, alerts, and execution telemetry.
- `auth` for permissions and actor context.
- `audl` for immutable audit trails.
- `conf` for tenant configuration.
- `cach` for pipeline metadata and planning caches.
- AI/pipeline-agent runtimes such as Codex, Claude Code, opencode, Pi, and
  future adapters through first-class but provider-neutral composition.

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

### Pipeline-Agent Composition

ETLP must support first-class AI/pipeline-agent contributors:

- Register pipeline agents per tenant.
- Require supported runtime, supported role, declared scope, owner, purpose,
  and machine-contribution disclosure.
- Require human approval for privileged roles that can influence datasource,
  execution, quality, publish, or replay decisions.
- Persist pipeline-agent records for UI display and audit evidence.
- Preserve otherwise valid privileged pipeline agents without approval as
  `pending_review` records with policy decision, matched rules, review reasons,
  and review evidence.
- Surface pipeline-agent registration failures as matched guardrails.

### Bytewax Lifecycle Batches

ETLP must support lifecycle-batch validation:

- Require lifecycle processors to declare the `bytewax` event stream.
- Track batch status, mutation count, and matched guardrails for UI/runtime
  evidence.
- Persist denied non-Bytewax lifecycle batches as `denied` records before
  raising `PermissionError`.
- Keep event brokers out of the core lifecycle contract. Brokers may exist
  behind adapters, but Bytewax is the required lifecycle processing engine for
  this packet.

### Rule Engine

The rule engine must be deterministic and side-effect free. It must evaluate
context and return:

- `decision`: `allow`, `deny`, or `require_review`.
- `matched_rules`: ordered rule names.
- `actions`: required actions and reasons.
- `context`: evaluated context.

### Durable Review Evidence

ETLP must persist durable review evidence on generated-application lifecycle
records and audit events:

- Persist `policy_decision`, `matched_rules`, `review_reasons`, and
  `review_evidence`.
- Expose pending-review queues for pipelines, datasources, mappings,
  executions, quality results, schedules, publish reviews, replay requests,
  pipeline agents, and lifecycle batches.
- Expose review evidence through registration metadata, semantic model output,
  API helpers, view models, settings, release evidence, and package tests.

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
- Pipeline-agent runtime and role must be supported.
- Pipeline-agent scope, owner, purpose, and machine-contribution disclosure are
  required.
- Privileged pipeline-agent roles require human approval evidence or pending
  review.
- ETLP lifecycle batches must use Bytewax.

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
- Pipeline-agent roster.
- Lifecycle batch monitor.
- Settings.

The UI manifest must include route names, paths, components, nav groups,
permissions, shell target, view module, and theme requirement. The theme must
include tokens and component descriptors for pipeline status, field mapping,
execution timeline, quality gates, lineage, publish review, and adapter health.

## Non-Goals for This Packet

- Building every physical connector.
- Embedding Codex, Claude Code, opencode, Pi, or any other agent runtime
  client directly in ETLP.
- Running a live distributed execution engine.
- Implementing Bytewax runtime flows inside the dependency-light control plane.
- Implementing AI optimization internals.
- Treating any broker as the core lifecycle processor.
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
- Generated applications can register pipeline agents and validate Bytewax
  lifecycle batches through dependency-light helpers.
- Generated applications can compose pending-review queues from durable review
  evidence instead of transient exceptions.
- Focused ETLP tests cover positive and negative guardrail paths.
- `app.py`, `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` reflect current contract-derived evidence.
- Focused compile, package tests, implementation audit, publish plan, stale
  marker search, and diff checks pass.
