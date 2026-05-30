# DEPL Capability Specification

## Purpose

DEPL defines the APG deployment-management capability. It gives generated
applications an executable, governed lifecycle for defining environments,
creating releases, attaching rollback plans, recording health gates, planning
rollouts, executing deployments, registering AI deployment agents, composing
deployment UIs, and auditing every meaningful deployment transition.

## Scope

In scope:

- Tenant-aware in-memory service state for package checks and generated apps.
- Environment, release, rollback-plan, health-gate, deployment-plan,
  deployment-run, rollback-event, deployment-agent, and audit models.
- Deterministic deployment fingerprints and audit payload hashes.
- Deterministic rule evaluation for deployment guardrails.
- First-class AI deployment-agent registration with runtime, role, scope,
  disclosure, and policy reference.
- Bytewax stream contract metadata for lifecycle and batch mutation events.
- Framework-neutral API helpers and UI view models.
- Theme tokens and component metadata for generated APG applications.
- Package evidence through `app.py`, `semantic_model.json`,
  `package_manifest.json`, and `release_report.json`.

Out of scope for this dependency-light package:

- Live cloud-provider deployment calls.
- Kubernetes or container-orchestrator operations.
- Registry writes or artifact downloads.
- Ticketing-system mutation.
- Notification dispatch.
- Persistent database storage.
- Rendered browser UI.

Those behaviors must attach through explicit adapters so local APG tooling stays
safe, deterministic, and side-effect free.

## Functional Requirements

### Environments

- Register tenant-scoped environments with ID, name, tier, owner, policy,
  approvers, and active status.
- Deny missing tenant context, missing owner, missing policy, and production
  environments without approvers.
- Store duplicate environment IDs safely across tenants.

### Releases

- Create tenant-scoped release manifests with version, owner, manifest payload,
  artifact digest, artifact signature, change ticket, creator, and status.
- Deny missing owner, manifest, artifact digest, artifact signature, or change
  ticket.
- Store duplicate release IDs safely across tenants.

### Rollback Plans

- Attach rollback plans only to tenant-local releases.
- Require owner, steps, and tested evidence.
- Store duplicate rollback-plan IDs safely across tenants.

### Health Gates

- Record health gates only for tenant-local releases.
- Require at least one check, health report reference, and log-trace link.
- Mark health gates as passed only when every check passes and evidence exists.

### Deployment Plans

- Create deployment plans only when release, environment, rollback plan, and
  health gate all belong to the same tenant.
- Support `rolling`, `blue_green`, and `canary` strategies.
- Deny unsupported strategies, missing change ticket, failed health gates,
  missing rollback plan, and production plans without approval.
- Route large canary plans to review unless review evidence is recorded.
- Allow reviewed plans to be approved.
- Change plan state only with reason and audit evidence.

### Deployment Runs

- Execute only approved deployment plans.
- Require log-trace evidence and health report references.
- Generate deterministic deployment fingerprints.
- Store duplicate run IDs safely across tenants.

### Rollback Events

- Execute rollback only against tenant-local deployment runs.
- Require rollback reason.
- Move the deployment run and deployment plan into rollback state.

### AI Deployment Agents

- Register AI deployment agents as first-class DEPL records.
- Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.
- Supported roles: `release_planner`, `rollout_operator`,
  `health_reviewer`, `rollback_coordinator`, `incident_reviewer`.
- Require registration flag, supported runtime, supported role, explicit scope,
  and contribution disclosure.
- Isolate agent registrations by tenant even when agent IDs are reused.

### UI And Theme

DEPL must expose route metadata for:

- `dashboard`
- `releases`
- `deployments`
- `rollouts`
- `health`
- `rollback`
- `agents`
- `evidence`
- `audit`
- `analytics`
- `settings`

DEPL must expose view-model functions for these operational surfaces and
publish the `depl_release_ops` theme with release, rollout, health, rollback,
agent, and audit component metadata.

### Streaming

DEPL must declare Bytewax as the lifecycle stream processor. The stream
contract must include environment, release, rollback-plan, health-gate,
deployment-plan, deployment-run, rollback-event, deployment-agent, and audit
state families. Batch deployment mutation must be denied unless the event
stream is `bytewax`.

## Rule Engine Requirements

The deterministic rules must cover:

- tenant context;
- release owner, manifest, artifact signature, and change-ticket evidence;
- health-check evidence, health gate, production approval, rollback plan,
  canary review, and trace capture;
- AI deployment-agent registration, runtime, role, scope, and disclosure;
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
- Focused DEPL tests pass.
- `app.self_test()` passes.
- `semantic_model.json` exposes DEPL routes, rules, configuration, theme, and
  Bytewax stream metadata.
- Implementation audit and publish-plan pass for DEPL.
- Stale-marker search finds no unsupported overclaims, unfinished markers, or
  unsupported stream-provider references in DEPL.
