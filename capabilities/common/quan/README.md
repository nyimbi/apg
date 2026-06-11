# QUAN Quantum Computing Capability

QUAN gives APG applications a tenant-scoped quantum lab runtime: backend
registry, provider credentials posture, quota policies, circuit library, job
submission, deterministic result capture, experiment workbench, quantum agents,
noise modelling, fidelity monitoring, Grover search, async execution, and
Decimal-precision cost accounting.

The package stays dependency-light. Production quantum providers, provider
credential vaults, encryption systems, cost controls, monitoring systems,
audit sinks, experiment stores, and Bytewax workers are represented as APG
adapters in the executable contract and are bound by the host application.

## What It Provides

- Quantum backend registry with provider, backend type, qubit capacity,
  approval state, credential references, simulator fallback, and metadata.
- Quota policies for shots per job, jobs per day, cost limit, and retry policy.
- Circuit library with owners, versions, qubit requirements, gates, sensitive
  input encryption, experiment metadata, and structural complexity metrics.
- Job queue with submitter identity, quota checks, shot limits,
  Decimal-precision cost estimation, retry posture, review gates, and
  Bytewax stream validation.
- Result capture with deterministic measurement counts, confidence, retention,
  and summaries for generated-application proof.
- Quantum error mitigation: zero-noise extrapolation, probabilistic error
  cancellation, Clifford data regression, symmetry verification.
- Variational Quantum Eigensolver (VQE) with configurable ansatz and optimiser.
- Quantum Approximate Optimisation Algorithm (QAOA) for max-cut, graph
  colouring, portfolio optimisation, TSP, and vertex cover.
- Quantum Key Distribution (QKD) simulation: BB84, E91, B92, SARG04.
- Post-quantum encryption: Kyber, Dilithium, Falcon, SPHINCS+, NTRU.
- Quantum simulation of Ising, Hubbard, transverse-field, Heisenberg, and
  Bose-Hubbard systems via Trotter decomposition.
- Grover's search oracle with optimal iteration count and quadratic speedup
  ratio vs. classical brute force.
- Noise model registry: depolarising, thermal relaxation, readout error,
  crosstalk — apply to results for NISQ device fidelity estimation.
- Fidelity snapshot recording and drift detection with EMA-based alerting
  and automatic FIDELITY_DRIFT_ALERT audit events.
- Circuit complexity metrics: gate counts by type, two-qubit fraction,
  depth estimate, T-gate count, Meyer-Wallach entanglement proxy.
- Decimal-precision cost accounting via `quantum_cost_estimate_decimal`.
- Async execution: `async_submit_quantum_job`, `async_batch_submit_jobs`,
  `async_vqe_solve`, `async_qaoa_solve`, `async_quantum_simulation`,
  `async_quantum_analytics`.
- Quantum-inspired random number generation with QRNG audit trail.
- Circuit optimisation (depth/gate-count reduction at levels 0–3).
- Backend status with queue depth, availability, and calibration age.
- First-class AI quantum agents with runtime, role, scope, registration,
  and contribution-disclosure guardrails.
- UI route, API, view-model, theme, semantic-model, package-manifest, and
  release-report evidence.

## Main Files

- `SPECIFICATION.md` defines the normative capability behavior.
- `PLAN.md` records the implementation packet plan.
- `WORLD_CLASS_IMPROVEMENTS.md` documents 15 prioritised enhancement paths.
- `capability_contract.py` is the executable source of configuration, rules,
  routes, theme, adapters, provides/requires, and Bytewax stream metadata.
- `models.py` defines tenant-scoped backends, circuits, quotas, jobs, results,
  experiments, audit events, and agents.
- `quantum_runtime.py` contains deterministic IDs, provider normalization,
  retry policy normalization, cost estimation, measurement generation, and
  result summaries.
- `service.py` implements the runtime facade with 50+ methods.
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

## Async Batch Execution

```python
import asyncio
from capabilities.common.quan import QuanService

service = QuanService()
# ... register backend + circuits as above ...

jobs = [
    {"circuit_definition": {"name": "ghz", "qubits": 3, "gates": ["h", "cx", "cx"]},
     "backend": "local-sim", "shots": 512},
    {"circuit_definition": {"name": "bell", "qubits": 2, "gates": ["h", "cx"]},
     "backend": "local-sim", "shots": 1024},
]
results = asyncio.run(
    service.async_batch_submit_jobs(jobs, tenant_id="tenant-demo", submitted_by="researcher")
)
```

## Grover's Search

```python
result = service.grover_search(
    oracle_spec={"function_description": "3-SAT with 6 clauses"},
    n_qubits=8,
    marked_items=4,
    tenant_id="tenant-demo",
    shots=2048,
)
print(result["optimal_iterations"])    # 12
print(result["quantum_speedup_ratio"]) # 10.67x vs brute force
```

## Noise Modelling

```python
noise_model = service.noise_model_register(
    model_id="ibm-nairobi-approx",
    tenant_id="tenant-demo",
    model_type="depolarising",
    params={"gate_error_rate": 0.001, "two_qubit_error_rate": 0.01},
)
noisy = service.noise_model_apply(
    result_id="result-001",
    noise_model_id=noise_model["noise_model_id"],
    tenant_id="tenant-demo",
)
```

## Fidelity Drift Detection

```python
# Record calibration snapshots periodically
service.fidelity_snapshot_record(
    backend_id="qpu-01", tenant_id="tenant-demo",
    gate_fidelity=0.998, readout_fidelity=0.995,
    t1_us=120.0, t2_us=80.0,
)
# ... record more snapshots over time ...

alert = service.fidelity_drift_detect(
    backend_id="qpu-01", tenant_id="tenant-demo", drift_threshold=0.02,
)
if alert["drift_detected"]:
    print(alert["recommendation"])  # "halt_jobs_and_recalibrate"
```

## Decimal-Precision Cost Estimation

```python
cost = service.quantum_cost_estimate_decimal(
    tenant_id="tenant-demo",
    backend_id="local-sim",
    circuit_id="bell-v1",
    shot_count=1_000_000,
)
print(cost["estimated_cost"])  # "0.100000"  (string, 6 dp, no float error)
```

## Circuit Metrics

```python
metrics = service.circuit_metrics(circuit_id="bell-v1", tenant_id="tenant-demo")
print(metrics["two_qubit_fraction"])   # 0.5
print(metrics["circuit_depth_estimate"])  # 2
print(metrics["complexity_tier"])     # "high"
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

Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.
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

```bash
./.venv/bin/python -m py_compile capabilities/common/quan/__init__.py capabilities/common/quan/capability_contract.py capabilities/common/quan/models.py capabilities/common/quan/quantum_runtime.py capabilities/common/quan/service.py capabilities/common/quan/api.py capabilities/common/quan/views.py capabilities/common/quan/app.py capabilities/common/quan/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/quan/test_capability_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/quan --json
./.venv/bin/apg capabilities publish-plan capabilities/common/quan --json
```

Live quantum provider execution, durable experiment stores, hardware access,
credential vault calls, rendered UI, and Bytewax workers are integration
concerns outside the package proof.
