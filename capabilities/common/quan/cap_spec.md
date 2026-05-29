# Quantum Computing Capability Specification

- **Capability Name**: Quantum Computing
- **Capability ID**: `quan`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`quan` gives APG applications a local, deterministic quantum-lab runtime for
composing quantum backends, circuit definitions, quota policies, job
submission, result capture, experiment workbenches, and governance evidence.

The package is intentionally provider-adapter oriented. It can execute and test
tenant-scoped quantum workflows without live quantum hardware, while preserving
clear seams for cloud quantum providers, KeyM credential storage, Encr input
protection, AICR analysis, billing, durable audit export, and post-quantum
review systems.

## Provided Services

- `quantum_backend_registry`
- `circuit_management`
- `quantum_job_orchestration`
- `result_analysis`
- `post_quantum_governance`

## Required Services

- `aicr`
- `encr`
- `keym`

Optional composition partners include `mlcm`, `pred`, `comp`, and `logt`.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`.

The executable configuration covers:

- approved backend registration;
- KeyM-managed provider credential references for non-local providers;
- circuit ownership, versioning, sensitive input encryption, and experiment
  metadata;
- job quota policies, shot limits, retry policy, cost limits, and result
  retention;
- tenant context, quantum-job audit, post-quantum review, UI enablement, and
  theme selection.

## Rules

- `tenant_context_required`
- `backend_requires_approval`
- `circuit_requires_owner`
- `sensitive_input_requires_encryption`
- `job_requires_quota`
- `large_job_requires_review`

`QuanService.evaluate()` delegates to the deterministic rule engine and the
service layer enforces additional local package guardrails for credential
references, quota values, circuit metadata, retry policy, qubit capacity, cost
limits, and post-quantum experiment review.

## Runtime Surfaces

- `models.py` defines tenant-scoped backends, circuits, quota policies, jobs,
  results, experiments, and audit events.
- `quantum_runtime.py` provides deterministic helpers for stable IDs, provider
  normalization, backend type normalization, retry policy validation, job cost
  estimation, deterministic measurement counts, result confidence, and qubit
  capacity checks.
- `service.py` owns the in-process lifecycle for backend registration, quota
  attachment, circuit creation, job submission, job completion, experiment
  creation, compatibility records, list/query helpers, dashboard summaries, and
  audit events.
- `api.py` exposes dependency-light helpers over the service for generated APG
  applications and package tests.
- `views.py` returns route-aware view models for the dashboard, backend
  registry, circuit library, job queue, experiment workbench, result viewer,
  and governance surfaces.

## Executable Lifecycle

1. Register an approved backend for a tenant.
2. Attach a quota policy with shot, job, cost, and retry limits.
3. Create an owned, versioned circuit with gates and experiment metadata.
4. Submit a job against a compatible backend and circuit.
5. Complete the job to record deterministic measurement counts and confidence.
6. Group completed work into an experiment.
7. Inspect dashboard, queue, registry, result, and governance view models.

Negative paths block missing tenant context, unapproved backends, non-local
provider credentials without a KeyM reference, ownerless circuits, sensitive
inputs without encryption, circuits without metadata, jobs without quota,
large jobs without review, quota/cost/qubit overflow, missing retry policy, and
post-quantum experiments without review.

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `quan_quantum_lab` APG theme contract.

## Adapter Boundaries

The current package does not call live quantum providers, perform real
cryptographic encryption, read KeyM secrets, invoke AICR analysis, charge
billing systems, export durable audit logs, or complete external
post-quantum assessments. Those systems should be connected through explicit
adapters after the local deterministic lifecycle remains green.

## Focused Verification

```bash
./.venv/bin/pytest -q capabilities/common/quan/test_capability_contract.py capabilities/common/quan/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/quan --json
./.venv/bin/apg capabilities publish-plan capabilities/common/quan --json
```
