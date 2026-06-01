# NLPC Specification

## Purpose

NLPC provides APG with a first-class, composable text intelligence capability.
It is designed for generated applications that need governed document ingestion,
NLP processing, annotation, model linkage, semantic search, language coverage,
tenant lexicons, and auditable policy enforcement.
It also treats AI agents as first-class text-governance actors so Codex,
Claude Code, OpenCode, Pi, and later provider runtimes can participate through
tenant-scoped, accountable, policy-checked composition rather than direct SDK
coupling.

## Scope

This packet establishes the executable baseline for NLPC:

- Contract-driven configuration, schema, adapters, deterministic rules, UI
  routes, and visual theme tokens.
- A dependency-light runtime service for generated applications.
- UI view models that can be composed into APG screens.
- First-class NLP-agent composition and Bytewax lifecycle-batch validation for
  generated-app and agent-authored text-intelligence mutations.
- Package evidence that can be published and self-tested without importing the
  legacy heavy service stack.
- Focused tests for the contract, lifecycle, guardrails, language coverage, and
  package evidence.

The legacy production modules remain available for richer deployments. The
generated-app runtime is intentionally smaller so APG can build executable
applications quickly and replace internals behind stable contracts later.

## Actors

- Application user: submits documents and reviews processing outputs.
- Language operations team: owns pipelines, lexicons, and annotation projects.
- Model governance team: registers and releases NLP models through MLCM.
- Platform operator: configures adapters, Bytewax event streaming, audit, auth,
  metrics, and generated app deployment.
- NLP agent owner: registers provider-neutral agents with scope, owner,
  purpose, contribution disclosure, and privileged-role review evidence.

## Functional Requirements

### Document Lifecycle

- Ingest document content by tenant.
- Capture document id, source reference, language, content hash, character
  count, status, and metadata.
- Auto-detect language when requested.
- Enforce tenant context, content presence, document size, and language
  evidence.

### Processing Lifecycle

- Run one or more enabled tasks against an ingested document.
- Store a processing run with tasks, language, confidence, result map, and
  status.
- Support sentiment analysis, entity recognition, PII detection, summarization,
  semantic search, translation, classification, topic modeling, keyword
  extraction, and governed text generation.
- Enforce task enablement, language support, PII redaction policy, generation
  safety policy, generation model policy, search index binding, translation
  language pair evidence, length budgets, and review requirements.
- Preserve processing decisions, matched rules, and review reasons on each run
  so low-confidence or budget-incomplete processing can be routed to human
  review without rerunning NLP tasks.
- Denial guardrails must stop processing before task execution; review-required
  guardrails must create `pending_review` run evidence for generated-app
  review queues.

### Pipeline Lifecycle

- Register a pipeline with id, tenant, name, owner, model reference, version,
  tasks, status, and metadata.
- Require owner, registered model evidence, version metadata, and enabled tasks.

### Model Lifecycle

- Register an NLP model with MLCM linkage, owner, policy reference, evaluation
  state, approval state, and metadata.
- Release a model only when evaluation and approval evidence are present.

### Annotation Lifecycle

- Create annotation projects with guidelines, task, and consensus threshold.
- Submit annotations against tenant-scoped documents.
- Require guidelines for projects and adjudication evidence for low consensus.

### Lexicon Lifecycle

- Register tenant lexicons with language, owner, and terms.
- Require language metadata and reject unsupported language codes.

### NLP Agent Lifecycle

- Register NLP agents with id, tenant, name, runtime, role, scope, owner,
  purpose, contribution disclosure, human-approval flag, and status.
- Support `codex`, `claude_code`, `opencode`, and `pi` runtime codes through
  AICR adapter boundaries.
- Support document review, language review, PII review, generation safety,
  annotation review, pipeline review, model release review, semantic search
  review, and language steward roles.
- Mark privileged agents without human approval as `pending_review` while
  denying unsupported runtime, unsupported role, missing scope, missing owner,
  missing purpose, or undisclosed machine contribution.

### Lifecycle Batch Governance

- Validate NLPC lifecycle mutation batches before accepting generated-app or
  agent-authored changes.
- Require Bytewax as the lifecycle processor and reject non-Bytewax streams.
- Support document, processing, pipeline, annotation, model, lexicon, language
  registry, and NLP-agent batch operations.
- Retain accepted and denied batch evidence for dashboard and governance views.

### Language Coverage

- Maintain the existing broad language registry.
- Preserve at least 40 African language codes in contract configuration and
  runtime UI evidence.

### UI and Theming

- Expose routes for dashboard, processing, documents, pipelines, batches,
  annotations, review, models, languages, lexicons, search, agents, lifecycle,
  governance, audit, and settings.
- Provide view models for each route.
- Publish theme tokens and component hints for generated applications.

### Adapters

- Use Bytewax for event streaming.
- Expose adapter keys for generated app runtime, production runtime, HTTP API,
  AICR, MLCM, CONF, AUTH, AUDL, MONI, and SRCH.
- Keep external AI-agent runtimes behind AICR provider-neutral adapters.

## Non-Goals

- Full live model inference.
- Live Bytewax stream execution.
- Browser-rendered Flask-AppBuilder validation.
- Persistent database migrations.
- Provider-specific LLM calls.
- Live external AI-agent runtime execution.
- Load, latency, drift, and throughput benchmarking.

These are later integration and hardening tasks once the executable baseline is
stable.

## Acceptance Criteria

- `get_capability_contract()` exposes at least 38 deterministic rules, at least
  16 UI routes, first-class agent metadata, Bytewax lifecycle metadata,
  runtime adapter evidence, and at
  least 40 African language codes.
- `NlpcService` executes document, processing, pipeline, model, annotation,
  lexicon, NLP-agent, lifecycle-batch, list, dashboard, and compatibility
  flows.
- Guardrail tests prove denied cases fail before state is accepted and
  review-required processing cases become pending review evidence.
- `app.self_test()` passes and fails if route, rule, Bytewax, or runtime
  evidence becomes stale.
- Package JSON evidence can be regenerated from `app.semantic_model()` and
  `app.component_manifest()`.
