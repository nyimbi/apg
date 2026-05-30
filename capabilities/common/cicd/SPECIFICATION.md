# CICD Capability Specification

## Purpose

CICD defines the APG continuous integration and delivery capability. It gives
generated applications an executable, governed lifecycle for defining
pipelines, running builds, publishing artifacts, recording quality gates,
promoting releases, registering AI delivery agents, composing CI/CD UIs, and
auditing every meaningful delivery transition.

## Scope

In scope:

- Tenant-aware in-memory service state for package checks and generated apps.
- Pipeline, build, artifact, quality gate, promotion, delivery-agent, and audit
  models.
- Deterministic trace IDs and artifact digests.
- Deterministic rule evaluation for CI/CD guardrails.
- First-class AI delivery-agent registration with runtime, role, scope,
  disclosure, and policy reference.
- Bytewax stream contract metadata for lifecycle and batch mutation events.
- Framework-neutral API helpers and UI view models.
- Theme tokens and component metadata for generated APG applications.
- Package evidence through `app.py`, `semantic_model.json`,
  `package_manifest.json`, and `release_report.json`.

Out of scope for this dependency-light package:

- Live Git provider operations.
- Build-runner execution.
- Container or artifact registry writes.
- Production scanner execution.
- Deployment platform calls.
- Persistent database storage.
- Rendered browser UI.

Those behaviors must attach through explicit adapters so local APG tooling stays
safe, deterministic, and side-effect free.

## Functional Requirements

### Pipelines

- Create tenant-scoped pipelines with ID, name, owner, source reference, worker
  pool, stages, secret scope, cache policy, quality gate, and parallelism.
- Deny missing owner, source policy, worker pool, stages, secret scope, cache
  policy, or quality gate policy.
- Route high-parallelism pipelines to review unless capacity review evidence is
  recorded.
- Approve pending-review pipelines.
- Change pipeline state only with reason and audit evidence.
- Store duplicate pipeline IDs safely across tenants.

### Builds

- Run builds only for active tenant-local pipelines.
- Require secret-scope evidence and log/trace capture.
- Generate deterministic trace IDs.
- Store commit reference, triggering actor, trace ID, status, secret scope, and
  cache policy.

### Artifacts

- Publish artifacts only for tenant-local builds.
- Store name, version, digest, signature state, and status.
- Generate deterministic artifact digests.

### Quality Gates

- Record quality gates only for tenant-local artifacts.
- Require security scan evidence.
- Store test result, scan result, artifact signature state, approval evidence,
  findings, and pass/fail status.

### Promotions

- Promote artifacts only when the artifact and quality gate belong to the same
  tenant and the gate belongs to the artifact.
- Require signed artifact, passing quality gate, approval evidence, environment
  policy, and separation of duties.
- Store source environment, target environment, requesting actor, quality gate,
  approval state, and promotion status.

### AI Delivery Agents

- Register AI delivery agents as first-class CICD records.
- Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.
- Supported roles: `pipeline_designer`, `build_operator`,
  `security_reviewer`, `release_manager`, `incident_reviewer`.
- Require registration flag, supported runtime, supported role, explicit scope,
  and contribution disclosure.
- Isolate agent registrations by tenant even when agent IDs are reused.

### UI And Theme

CICD must expose route metadata for:

- `dashboard`
- `pipelines`
- `builds`
- `artifacts`
- `gates`
- `promotions`
- `agents`
- `audit`
- `analytics`
- `settings`

CICD must expose view-model functions for these operational surfaces and
publish the `cicd_pipeline_ops` theme with pipeline, build, artifact, quality
gate, agent, and audit component metadata.

### Streaming

CICD must declare Bytewax as the lifecycle stream processor. The stream
contract must include pipeline, build, artifact, gate, promotion, delivery
agent, and audit state families. Batch pipeline mutation must be denied unless
the event stream is `bytewax`.

## Rule Engine Requirements

The deterministic rules must cover:

- tenant context;
- pipeline owner, source policy, worker pool, stages, secret scope, cache
  policy, quality gate, and high-parallelism review;
- build secret scope and trace capture;
- quality gate security scan;
- artifact signature;
- promotion quality gate, approval, environment policy, and separation of
  duties;
- AI delivery-agent registration, runtime, role, scope, and disclosure;
- state-change reason and audit evidence;
- cross-tenant access denial;
- Bytewax event stream requirement for batch mutation.

The rule evaluator must support equality plus numeric `_lt`, `_lte`, `_gt`,
`_gte`, and inequality `_ne` conditions.

## Non-Functional Requirements

- Importing the package must not require live adapters.
- Service operations must remain tenant-scoped.
- Generated package evidence must stay synchronized with the contract.
- API and view-model functions must return plain Python dictionaries/lists.
- Focused tests must cover the main lifecycle, guardrail failures, AI agents,
  tenant-safe duplicate IDs, Bytewax metadata, and generated evidence.
- Documentation must explain use, architecture, boundaries, and verification.

## Acceptance Criteria

- `README.md`, `SPECIFICATION.md`, `PLAN.md`, and `cap_spec.md` describe the
  same executable packet.
- `register_capability()` exposes dependencies, optional adapters,
  permissions, endpoints, UI metadata, theme, and Bytewax stream contract.
- Focused CICD tests pass.
- `app.self_test()` passes.
- `semantic_model.json` exposes CICD routes, rules, configuration, theme, and
  Bytewax stream metadata.
- Implementation audit and publish-plan pass for CICD.
- Stale-marker search finds no unsupported overclaims, unfinished markers, or
  unsupported stream-provider references in CICD.

