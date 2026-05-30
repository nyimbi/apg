# QUAN Quantum Computing Specification

## Purpose

QUAN is APG's common quantum-computing capability. It lets generated and
composed applications register quantum backends, attach quota policies, manage
circuits, submit quantum jobs, capture measurement results, create experiments,
and govern cryptographic transition work through APG UI and API surfaces.

The capability is designed for executable applications first. It provides a
dependency-light runtime and explicit adapter boundaries so production systems
can connect real quantum providers, credential vaults, encryption services,
audit stores, monitoring systems, experiment repositories, compliance systems,
and Bytewax workers later.

## Capability Identity

- Capability id: `quan`
- Display name: `Quantum Computing`
- Category: `common`
- Runtime target: `python`
- Primary service: `service.QuanService`
- UI prefix: `/quan`
- API prefix: `/quan/api/v1`
- Event stream processor: `bytewax`

## Provided Services

- `quantum_backend_registry`
- `circuit_management`
- `quantum_job_orchestration`
- `result_analysis`
- `post_quantum_governance`
- `quan_agents`

## Required Capabilities

- `aicr` for AI-assisted experiment analysis and agent orchestration.
- `encr` for sensitive-input encryption policy.
- `keym` for provider credential references.
- `audl` for durable audit evidence.

Optional adapters include `mlcm`, `pred`, `comp`, `moni`, and `logt`.

## Domain Model

`QuantumBackend`

- Tenant-local backend id, name, provider, backend type, qubit capacity,
  approval state, credential reference, quota-policy posture, simulator
  fallback, lifecycle status, metadata, and timestamps.

`QuantumQuotaPolicy`

- Backend quota record with max shots per job, max jobs per day, cost limit,
  retry policy, and creation time.

`QuantumCircuit`

- Circuit record with owner, version, qubit requirement, gates, sensitive input
  posture, encryption state, experiment metadata, lifecycle status, and
  timestamps.

`QuantumJob`

- Job record with backend, circuit, submitter, shot count, estimated cost,
  review state, retry posture, lifecycle status, and timestamps.

`QuantumResult`

- Result record with measurement counts, confidence, analysis summary,
  retention period, and creation time.

`QuantumExperiment`

- Experiment record with circuit, jobs, hypothesis, post-quantum review
  posture, lifecycle status, and creation time.

`QuanAuditEvent`

- Governance record for quantum lifecycle actions.

`QuanAgent`

- Registered AI quantum agent with tenant, runtime, role, explicit scope,
  registration status, contribution disclosure, and activity state.

## Rule Engine

The deterministic rule engine must enforce:

- tenant context on every quantum operation;
- backend approval;
- credential references for external providers;
- positive backend qubit capacity;
- circuit owner identity;
- circuit version;
- positive circuit qubit requirement;
- at least one circuit gate;
- encryption for sensitive circuit inputs;
- experiment metadata;
- quota policy before job submission;
- submitter identity;
- retry policy;
- positive shot count;
- review for large jobs;
- Bytewax event stream for job lifecycle events;
- experiment hypothesis;
- post-quantum review for cryptographic transition experiments;
- registered AI quantum agent;
- supported AI-agent runtime;
- supported AI-agent role;
- explicit AI-agent scope;
- AI contribution disclosure;
- audit evidence for lifecycle state changes;
- Bytewax event stream for batch quantum mutation.

## UI Contract

The capability exposes these APG Python UI routes:

- `/quan/dashboard`
- `/quan/backends`
- `/quan/circuits`
- `/quan/jobs`
- `/quan/experiments`
- `/quan/results`
- `/quan/agents`
- `/quan/audit`
- `/quan/governance`
- `/quan/settings`

View models must expose backend summaries, quota policies, circuits, jobs,
results, experiments, quantum agents, rules, audit events, theme data, and
Bytewax stream metadata.

## Theme

The default theme is `quan_quantum_lab`. Theme components cover backend cards,
circuit libraries, job queues, result viewers, agent panels, and audit
timelines.

## Event Stream

Lifecycle telemetry is described by:

- processor: `bytewax`
- topic: `apg.quan.lifecycle`
- state: backends, circuits, quota policies, jobs, results, experiments, QUAN
  agents, audit events
- events: backend registered, quota policy attached, circuit created, job
  submitted, result recorded, experiment created, agent registered
- guardrail: `batch_quantum_mutation_requires_bytewax`

The package records stream metadata and guardrails. Live Bytewax topology is
an application deployment concern.

## Acceptance Criteria

- The package has local README, specification, plan, contract, runtime, view,
  API, test, package-manifest, semantic-model, and release-report artifacts.
- The contract exposes provides/requires, configuration schema, rules, routes,
  theme, and Bytewax stream metadata.
- The service supports backends, quota policies, circuits, jobs, results,
  experiments, AI-agent registration, audit events, tenant-local IDs, and
  Bytewax batch mutation validation.
- Focused tests prove the main lifecycle, guardrails, tenant isolation,
  generated evidence, and docs.
- Compile, focused pytest, implementation-audit, publish-plan, marker scan,
  and diff checks pass before commit.
