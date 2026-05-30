# SBOX Sandbox/Testing Environment Specification

## Purpose

SBOX is APG's common sandbox and testing capability. It lets generated and
composed applications create isolated environments, manage templates, attach
controlled datasets, run tests and experiments, monitor results, register AI
sandbox agents, and govern sandbox lifecycle work through APG UI and API
surfaces.

The capability is designed for executable applications first. It provides a
dependency-light runtime and explicit adapter boundaries so production systems
can connect real container runtimes, environment managers, data masking
services, network policy engines, secret vaults, audit stores, logging systems,
and Bytewax workers later.

## Capability Identity

- Capability id: `sbox`
- Display name: `Sandbox/Testing Environment`
- Category: `common`
- Runtime target: `python`
- Primary service: `service.SboxService`
- UI prefix: `/sbox`
- API prefix: `/sbox/api/v1`
- Event stream processor: `bytewax`

## Provided Services

- `sandbox_registry`
- `isolation_profiles`
- `test_runs`
- `synthetic_datasets`
- `safety_policy`
- `sbox_agents`

## Required Capabilities

- `plgn` for plugin test policies and extension validation.
- `secu` for isolation, network, data, and secret controls.
- `envm` for environment template posture.
- `audl` for durable audit evidence.

Optional adapters include `cicd`, `depl`, `logt`, and `agnt`.

## Domain Model

`IsolationProfile`

- Tenant-local profile id, name, isolation level, network policy posture,
  secret redaction, data masking, outbound network approval, approver, and
  creation time.

`SandboxTemplate`

- Reusable sandbox blueprint with tenant, runtime, owner, default TTL,
  plugin-test policy posture, tags, and creation time.

`SandboxDataset`

- Dataset record with tenant, type, owner, lineage, retention, production
  review posture, masking state, and creation time.

`SandboxEnvironment`

- Tenant-local sandbox with template, isolation profile, owner, TTL, datasets,
  state, lifecycle review, secret access, outbound network request, risk score,
  and timestamps.

`SandboxRun`

- Test or experiment run with tenant, sandbox id, run type, requester, status,
  requested/passed/failed/blocked counts, timestamps, and logs.

`SboxAuditEvent`

- Governance record for sandbox lifecycle actions.

`SboxAgent`

- Registered AI sandbox agent with tenant, runtime, role, explicit scope,
  registration status, contribution disclosure, and activity state.

## Rule Engine

The deterministic rule engine must enforce:

- tenant context on every sandbox operation;
- sandbox owner identity;
- sandbox template;
- isolation profile;
- positive TTL;
- secret redaction when secret access is requested;
- outbound-network approval;
- lifecycle review for long-lived sandboxes;
- dataset owner identity;
- dataset lineage;
- dataset retention;
- production sample dataset review;
- masking for sensitive datasets;
- run requester identity;
- positive test count;
- plugin-test policy for plugin runs;
- Bytewax event stream for run lifecycle events;
- registered AI sandbox agent;
- supported AI-agent runtime;
- supported AI-agent role;
- explicit AI-agent scope;
- AI contribution disclosure;
- audit evidence for lifecycle state changes;
- Bytewax event stream for batch sandbox mutation.

## UI Contract

The capability exposes these APG Python UI routes:

- `/sbox/dashboard`
- `/sbox/sandboxes`
- `/sbox/templates`
- `/sbox/datasets`
- `/sbox/runs`
- `/sbox/agents`
- `/sbox/policies`
- `/sbox/audit`
- `/sbox/logs`
- `/sbox/settings`

View models must expose sandbox summaries, templates, isolation profiles,
datasets, runs, sandbox agents, policy rules, audit events, theme data, and
Bytewax stream metadata.

## Theme

The default theme is `sbox_safe_testing`. Theme components cover sandbox cards,
run monitors, dataset managers, policy centers, agent panels, and audit
timelines.

## Event Stream

Lifecycle telemetry is described by:

- processor: `bytewax`
- topic: `apg.sbox.lifecycle`
- state: isolation profiles, templates, datasets, sandboxes, runs, SBOX
  agents, audit events
- events: isolation profile created, template created, dataset registered,
  sandbox created, run started, run completed, agent registered
- guardrail: `batch_sandbox_mutation_requires_bytewax`

The package records stream metadata and guardrails. Live Bytewax topology is
an application deployment concern.

## Acceptance Criteria

- The package has local README, specification, plan, contract, runtime, view,
  API, test, package-manifest, semantic-model, and release-report artifacts.
- The contract exposes provides/requires, configuration schema, rules, routes,
  theme, and Bytewax stream metadata.
- The service supports isolation profiles, templates, datasets, sandboxes,
  runs, AI-agent registration, audit events, tenant-local IDs, and Bytewax
  batch mutation validation.
- Focused tests prove the main lifecycle, guardrails, tenant isolation,
  generated evidence, and docs.
- Compile, focused pytest, implementation-audit, publish-plan, marker scan,
  and diff checks pass before commit.
