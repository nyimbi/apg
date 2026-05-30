# DTWN Capability Specification

## Purpose

DTWN defines the APG digital-twin capability. It gives generated applications
an executable, governed lifecycle for creating twins, registering simulation
models, ingesting authenticated telemetry, mapping topology, running approved
simulations, recording prediction review state, registering AI twin agents,
composing digital-twin UIs, and auditing every meaningful twin transition.

## Scope

In scope:

- Tenant-aware in-memory service state for package checks and generated apps.
- Digital-twin, simulation-model, telemetry-sample, topology-link,
  simulation-run, prediction, twin-agent, and audit models.
- Deterministic state versions, simulation outputs, recommendation outputs,
  and audit digests.
- Deterministic rule evaluation for digital-twin guardrails.
- First-class AI twin-agent registration with runtime, role, scope,
  disclosure, and policy reference.
- Bytewax stream contract metadata for lifecycle and batch mutation events.
- Framework-neutral API helpers and UI view models.
- Theme tokens and component metadata for generated APG applications.
- Package evidence through `app.py`, `semantic_model.json`,
  `package_manifest.json`, and `release_report.json`.

Out of scope for this dependency-light package:

- Live IoT broker ingestion.
- Live geospatial service calls.
- Computer-vision pipeline execution.
- Machine-controller operations.
- External simulator execution.
- Time-series database persistence.
- External prediction service execution.
- Rendered browser UI.

Those behaviors must attach through explicit adapters so local APG tooling stays
safe, deterministic, and side-effect free.

## Functional Requirements

### Twins

- Create tenant-scoped twins with ID, asset identity, name, owner, type,
  location, initial state, topology references, state version, and status.
- Deny missing tenant context, missing owner, and missing asset identity.
- Store duplicate twin IDs safely across tenants.
- Change twin status only with reason and audit evidence.

### Simulation Models

- Register tenant-scoped simulation models with name, version, owner, type,
  calibration evidence, approval metadata, confidence, and status.
- Deny missing calibration evidence.
- Deny confidence below the configured threshold.
- Mark models as approved only when approval metadata is present.

### Telemetry

- Ingest telemetry only for tenant-local twins.
- Require authenticated source identity and at least one measurement.
- Fuse measurements into twin state without mutating caller input.
- Advance deterministic state versions on every accepted telemetry sample.

### Topology

- Link only tenant-local twins.
- Store topology relationship metadata.
- Update source and target twin topology references.

### Simulations

- Run simulations only against tenant-local twins and models.
- Require approved models.
- Require approval evidence for production simulations.
- Store deterministic outputs with state digest, normalized load, risk score,
  and recommendation.

### Predictions

- Record predictions only for tenant-local twins and models.
- Route high-risk predictions to review unless review evidence is recorded.
- Allow review completion with reviewer evidence.

### AI Twin Agents

- Register AI twin agents as first-class DTWN records.
- Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.
- Supported roles: `twin_designer`, `telemetry_reviewer`,
  `simulation_operator`, `prediction_reviewer`, `incident_reviewer`.
- Require registration flag, supported runtime, supported role, explicit scope,
  and contribution disclosure.
- Isolate agent registrations by tenant even when agent IDs are reused.

### UI And Theme

DTWN must expose route metadata for:

- `dashboard`
- `twins`
- `models`
- `telemetry`
- `simulations`
- `predictions`
- `topology`
- `agents`
- `audit`
- `analytics`
- `settings`

DTWN must expose view-model functions for these operational surfaces and
publish the `dtwn_digital_twin_ops` theme with twin, topology, simulation,
telemetry, agent, and audit component metadata.

### Streaming

DTWN must declare Bytewax as the lifecycle stream processor. The stream
contract must include twin, model, telemetry, topology, simulation, prediction,
twin-agent, and audit state families. Batch twin mutation must be denied unless
the event stream is `bytewax`.

## Rule Engine Requirements

The deterministic rules must cover:

- tenant context;
- twin owner and asset identity;
- simulation model calibration and confidence;
- approved simulation model requirement;
- telemetry source authentication and measurements;
- production simulation approval;
- high-risk prediction review;
- AI twin-agent registration, runtime, role, scope, and disclosure;
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
  tenant-safe duplicate IDs, Bytewax metadata, topology, prediction review, and
  generated evidence.
- Documentation must explain use, architecture, boundaries, and verification.

## Acceptance Criteria

- `README.md`, `SPECIFICATION.md`, `PLAN.md`, and `cap_spec.md` describe the
  same executable packet.
- `register_capability()` exposes dependencies, optional adapters,
  permissions, endpoints, UI metadata, theme, and Bytewax stream contract.
- Focused DTWN tests pass.
- `app.self_test()` passes.
- `semantic_model.json` exposes DTWN routes, rules, configuration, theme, and
  Bytewax stream metadata.
- Implementation audit and publish-plan pass for DTWN.
- Stale-marker search finds no unsupported overclaims, unfinished markers, or
  unsupported stream-provider references in DTWN.
