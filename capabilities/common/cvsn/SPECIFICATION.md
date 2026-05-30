# CVSN Specification

## Purpose

CVSN provides APG with first-class, composable visual intelligence. It supports
generated applications that need governed image, document, and video ingestion;
vision task execution; quality and safety workflows; visual model lifecycle;
semantic UI composition; and auditable policy enforcement.

## Scope

This packet establishes the executable integration baseline for CVSN:

- Contract-driven configuration, schema, adapters, deterministic rules, UI
  routes, and visual theme tokens.
- A dependency-light runtime service for generated applications.
- UI view models that can be composed into APG screens.
- Package evidence that can be published and self-tested without importing the
  legacy heavy vision service stack.
- Focused tests for the contract, lifecycle, guardrails, view models, and
  package evidence.

The existing production modules remain available for richer deployments. The
generated-app runtime provides a smaller executable baseline so APG can build
working applications quickly and replace internals behind stable contracts
later.

## Actors

- Application user: submits visual assets and reviews processing results.
- Vision operations team: owns pipelines, inspection plans, task policies, and
  review workflows.
- Model governance team: registers and releases vision models through MLCM.
- Platform operator: configures adapters, Bytewax event streaming, object
  storage, audit, auth, metrics, and generated app deployment.

## Functional Requirements

### Asset Lifecycle

- Ingest tenant-scoped visual assets.
- Capture asset id, kind, MIME type, file size, source reference, content hash,
  status, and metadata.
- Enforce tenant context, source evidence, supported MIME type, size limits, and
  hash evidence.

### Processing Lifecycle

- Run enabled vision tasks against ingested assets.
- Store processing jobs with task, operator, confidence, results, and status.
- Support OCR, object detection, image classification, quality inspection,
  factory safety, video analytics, visual similarity, barcode/QR, facial
  analysis, and content moderation.
- Evaluate policy before task execution.
- Enforce task enablement, operator evidence, asset-kind constraints, quality
  plan evidence, defect taxonomy, safety alerting, biometric consent,
  anonymization, retention limits, moderation policy, review requirements, batch
  limits, and video sampling policy.

### Model Lifecycle

- Register a vision model with MLCM linkage, owner, type, version, model-card
  reference, evaluation state, approval state, and metadata.
- Release a model only when evaluation and approval evidence are present.

### Pipeline Lifecycle

- Register a pipeline with id, tenant, name, owner, model reference, version,
  tasks, and status.
- Require owner, model reference, version metadata, and enabled tasks.

### UI and Theming

- Expose routes for dashboard, assets, documents, images, video, quality,
  safety, similarity search, review, models, governance, audit, and settings.
- Provide route-specific view models.
- Publish compact industrial theme tokens and component hints for generated
  applications.

### Adapters

- Use Bytewax for event streaming.
- Expose adapter keys for generated app runtime, production runtime, HTTP API,
  AICR, MLCM, CONF, AUTH, AUDL, MONI, object storage, and SRCH.

## Non-Goals

- Live OpenCV, YOLO, OCR, or transformer inference.
- Live Bytewax stream execution.
- Browser-rendered Flask-AppBuilder validation.
- Persistent database migrations.
- Load, latency, accuracy, drift, and throughput benchmarking.

These are later integration and hardening tasks once the executable baseline is
stable.

## Acceptance Criteria

- `get_capability_contract()` exposes at least 30 deterministic rules, at least
  12 UI routes, Bytewax adapter evidence, runtime adapter evidence, and theme
  component metadata.
- `CvsnService` executes asset, job, pipeline, model, release, list, dashboard,
  and APG record compatibility flows.
- Guardrail tests prove denied or review-required cases fail before processing
  work is accepted.
- `app.self_test()` passes and fails if route, rule, Bytewax, or runtime
  evidence becomes stale.
- Package JSON evidence can be regenerated from `app.semantic_model()` and
  `app.component_manifest()`.
