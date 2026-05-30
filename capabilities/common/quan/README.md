# QUAN Quantum Computing Capability

QUAN gives APG applications a tenant-scoped quantum lab runtime: backend
registry, provider credentials posture, quota policies, circuit library, job
submission, deterministic result capture, experiment workbench, quantum agents,
UI metadata, theme tokens, audit evidence, and Bytewax-backed lifecycle events.

The package stays dependency-light. Production quantum providers, provider
credential vaults, encryption systems, cost controls, monitoring systems,
audit sinks, experiment stores, and Bytewax workers are represented as APG
adapters in the executable contract and are bound by the host application.

## What It Provides

- Quantum backend registry with provider, backend type, qubit capacity,
  approval state, credential references, simulator fallback, and metadata.
- Quota policies for shots per job, jobs per day, cost limit, and retry policy.
- Circuit library with owners, versions, qubit requirements, gates, sensitive
  input encryption, and experiment metadata.
- Job queue with submitter identity, quota checks, shot limits, cost
  estimation, retry posture, review gates, and Bytewax stream validation.
- Result capture with deterministic measurement counts, confidence, retention,
  and summaries for generated-application proof.
- Experiment workbench with post-quantum review guardrails.
- First-class AI quantum agents with runtime, role, scope, registration, and
  contribution-disclosure guardrails.
- UI route, API, view-model, theme, semantic-model, package-manifest, and
  release-report evidence.

## Main Files

- `SPECIFICATION.md` defines the normative capability behavior.
- `PLAN.md` records the implementation packet plan.
- `capability_contract.py` is the executable source of configuration, rules,
  routes, theme, adapters, provides/requires, and Bytewax stream metadata.
- `models.py` defines tenant-scoped backends, circuits, quotas, jobs, results,
  experiments, audit events, and agents.
- `quantum_runtime.py` contains deterministic IDs, provider normalization,
  retry policy normalization, cost estimation, measurement generation, and
  result summaries.
- `service.py` implements the runtime facade.
- `api.py` exposes package-safe helper functions.
- `views.py` exposes UI view models.
- `test_capability_contract.py` proves lifecycle behavior and generated
  evidence.

## Basic Usage

```python
from capabilities.common.quan import QuanService

service = QuanService()
backend = service.register_backend(
    backend_id="local-sim",
    tenant_id="tenant-demo",
    name="Local simulator",
    provider="local",
    backend_type="simulator",
    qubit_count=8,
    approved=True,
)
service.attach_quota_policy(
    policy_id="quota-local",
    tenant_id="tenant-demo",
    backend_id=backend["id"],
    max_shots_per_job=4096,
    max_jobs_per_day=20,
    cost_limit=100.0,
)
circuit = service.create_circuit(
    circuit_id="bell-v1",
    tenant_id="tenant-demo",
    name="Bell pair",
    owner="research-owner",
    version="1.0.0",
    qubits_required=2,
    gates=["h", "cx", "measure"],
    experiment_metadata={"purpose": "entanglement validation"},
)
job = service.submit_job(
    job_id="job-001",
    tenant_id="tenant-demo",
    backend_id=backend["id"],
    circuit_id=circuit["id"],
    submitted_by="researcher",
    shot_count=1024,
)
result = service.complete_job("result-001", "tenant-demo", job["id"])
```

## AI Quantum Agents

Register AI agents before they assist with quantum governance:

```python
agent = service.register_quan_agent(
    tenant_id="tenant-demo",
    name="Job reviewer",
    runtime="codex",
    role="job_reviewer",
    scope="Review quota, retry, cost, shot-count, and Bytewax stream gates",
)
```

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`.
Supported roles cover backend, circuit, job, result, cost, and post-quantum
review.

## Composition

QUAN composes with:

- `aicr` for AI-assisted experiment analysis and agent orchestration.
- `encr` for encryption policy on sensitive inputs.
- `keym` for provider credential references.
- `audl` for durable audit evidence.
- `moni` and `logt` for operational telemetry and diagnostics.
- `comp` for regulated experiment and cryptographic-transition review.

Batch quantum mutation and quantum job lifecycle events must use the `bytewax`
event-stream adapter.

## Verification

Focused verification for this packet:

```bash
./.venv/bin/python -m py_compile capabilities/common/quan/__init__.py capabilities/common/quan/capability_contract.py capabilities/common/quan/models.py capabilities/common/quan/quantum_runtime.py capabilities/common/quan/service.py capabilities/common/quan/api.py capabilities/common/quan/views.py capabilities/common/quan/app.py capabilities/common/quan/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/quan/test_capability_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/quan --json
./.venv/bin/apg capabilities publish-plan capabilities/common/quan --json
```

Live quantum provider execution, durable experiment stores, hardware access,
credential vault calls, rendered UI, and Bytewax workers are integration
concerns outside the package proof.
