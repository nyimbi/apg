# SCRP Capability Specification

## Purpose

SCRP defines the APG data-harvesting capability. It gives generated
applications a governed way to register sources, define extractors, schedule
jobs, execute harvest lifecycles, retain result metadata, hand results to data
pipelines, compose UI screens, register AI harvest agents, and enforce
deterministic guardrails before live external harvesting adapters are attached.

## Scope

In scope:

- Tenant-aware in-memory service state for executable package checks.
- Source, extractor, job, run, result, handoff, agent, and audit models.
- Deterministic rule evaluation for operational and governance constraints.
- First-class AI harvest-agent registration with runtime, role, scope,
  disclosure, and policy reference.
- Bytewax stream contract metadata for lifecycle and batch mutation events.
- Framework-neutral API helpers and UI view models.
- Theme tokens and component metadata for generated APG applications.
- Package evidence through `app.py`, `semantic_model.json`,
  `package_manifest.json`, and `release_report.json`.

Out of scope for this dependency-light package:

- Live website crawling, browser automation, or remote API calls.
- Credential retrieval from a production vault.
- Production DLP scanning, scheduler execution, or ETL submission.
- Persistent database storage.
- Rendered browser UI.

Those behaviors belong behind explicit adapters so the capability remains
publishable, testable, and safe in local APG tooling.

## Functional Requirements

### Source Registry

- Register a source for one tenant.
- Normalize source type to `api`, `website`, `database`, `file`, or `feed`.
- Require endpoint, owner, terms evidence, credential vault reference,
  robots/terms policy, and positive rate limit.
- Support PII flags, PII policy evidence, sensitive-source review, and tags.
- Emit `source_registered` audit events.

### Extractor Profiles

- Register extractor profiles for one tenant.
- Normalize extractor type to `html`, `json`, `csv`, `xml`, `text`, or `api`.
- Require owner and schema.
- Support output mapping and incremental cursor field.
- Emit `extractor_created` audit events.

### Harvest Jobs

- Create tenant-local jobs that bind source, extractor, schedule policy, mode,
  pipeline target, owner, and enabled state.
- Normalize mode to `full`, `incremental`, or `sample`.
- Require downstream pipeline target.
- Allow guarded enabled/disabled state changes with reason and audit evidence.
- Emit `harvest_job_created` and `harvest_job_state_changed` audit events.

### Harvest Runs

- Start runs only for enabled jobs in the same tenant.
- Re-evaluate source terms, PII policy, sensitive-source review, and schedule
  policy before execution.
- Complete runs with non-negative record, error, and DLP violation counts.
- Require DLP scan evidence before completing PII-bearing runs.
- Derive run status from extracted record count, errors, and DLP outcome.
- Create result batches and pipeline handoff records for successful runs.
- Emit `harvest_run_started` and `harvest_run_completed` audit events.

### AI Harvest Agents

- Register AI harvest agents as first-class SCRP records.
- Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.
- Supported roles: `source_reviewer`, `extractor_designer`,
  `compliance_reviewer`, `run_operator`, `pipeline_operator`.
- Require registration flag, supported runtime, supported role, explicit scope,
  and contribution disclosure.
- Store policy reference and status.
- Isolate agent registrations by tenant even when agent IDs are reused.
- Emit `harvest_agent_registered` audit events.

### UI And Theme

SCRP must expose these route names:

- `dashboard`
- `sources`
- `jobs`
- `extractors`
- `pipelines`
- `compliance`
- `results`
- `agents`
- `audit`
- `analytics`
- `settings`

SCRP must expose view-model functions for the same operational surfaces and
publish the `scrp_harvest_ops` theme with source, job, extractor, compliance,
agent, and audit component metadata.

### Streaming

SCRP must declare Bytewax as the lifecycle stream processor. The stream
contract must include source, extractor, job, run, result, handoff, harvest
agent, and audit state families. Batch harvest mutation must be denied unless
the event stream is `bytewax`.

## Rule Engine Requirements

The deterministic rules must cover:

- tenant context;
- source owner;
- source terms evidence;
- credential vault reference;
- robots/terms policy;
- positive rate limit;
- PII handling policy;
- sensitive-source review;
- extractor schema;
- pipeline handoff target;
- schedule policy;
- DLP scan evidence;
- AI harvest-agent registration, runtime, role, scope, and disclosure;
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
- Focused tests must cover the main success lifecycle and guardrail failures.
- Documentation must explain use, architecture, boundaries, and verification.

## Acceptance Criteria

- `README.md`, `SPECIFICATION.md`, `PLAN.md`, and `cap_spec.md` describe the
  same executable packet.
- `register_capability()` exposes dependencies, optional adapters,
  permissions, endpoints, UI metadata, theme, and Bytewax stream contract.
- Focused SCRP tests pass.
- `app.self_test()` passes.
- `semantic_model.json` exposes SCRP routes, rules, configuration, theme, and
  Bytewax stream metadata.
- Implementation audit and publish-plan pass for SCRP.
- Stale-marker search finds no unsupported overclaims, unfinished markers, or
  unsupported stream-provider references in SCRP.
