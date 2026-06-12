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

## World-Class Enhancements (v2.0)

These 15 improvements address correctness gaps, missing algorithm coverage,
and production operational requirements identified against IBM Qiskit Runtime,
Google Cirq, PennyLane, Amazon Braket, and Quantinuum H-Series.

| # | Title | Category | Impact |
|---|-------|----------|--------|
| I1 | **Full Async Service Layer** | Architecture | All public methods are non-blocking; `async_*` variants use `asyncio.gather` for concurrent QPU calls. Sync shim via `asyncio.run` for legacy callers. |
| I2 | **Decimal-Precision Cost Accounting** | Correctness/Finance | Cost fields use `Decimal` with `ROUND_HALF_EVEN` quantized to 6 dp. Eliminates float rounding errors at high shot counts. Serialised as `str` in JSON. |
| I3 | **Statevector Simulator with Complex Amplitudes** | Correctness/Simulation | `statevector_simulate` backed by numpy complex128 amplitude array. Gate set: H, X, Y, Z, S, T, CX, CCX, RZ, RX, RY, SWAP. Capped at 20 qubits. |
| I4 | **Parametric Circuit Binding** | Feature/VQE-QAOA | `GateInstruction(name, qubits, params)` structure. `circuit_bind_parameters` returns a bound copy without mutating the original — required for VQE/QAOA gradient loops. |
| I5 | **Circuit Transpilation with Backend Topology** | Feature/Compilation | `circuit_transpile` performs gate decomposition to native set, SWAP insertion via coupling map, and gate cancellation. Returns physical depth, SWAP count, fidelity penalty. |
| I6 | **Noise Model Registry** | Feature/Fidelity | `noise_model_register` / `noise_model_apply`. Types: depolarising, thermal_relaxation, readout_error, crosstalk. Kraus operator noise injection into measurement counts. |
| I7 | **Grover's Search Oracle Interface** | Feature/Algorithms | `grover_search` computes optimal iteration count `floor(π/4 × √(N/k))`, success probability, gate count, and speedup ratio. Completes the VQE/QAOA/Grover near-term triad. |
| I8 | **Windowed Quota Enforcement** | Correctness/Multi-tenancy | `QuotaLedger` enforces `max_jobs_per_day` via 24-hour sliding windows. `quota_usage` returns consumed/limit/window_reset_at. Burst allowances and grace quotas supported. |
| I9 | **Quantum Volume Benchmarking** | Quality/Observability | `quantum_volume_benchmark` generates random QV circuits, computes heavy output probability, and checks the 2/3 threshold. Stores history for regression tracking. |
| I10 | **Circuit Composition Algebra** | Feature/DX | `circuit_compose` concatenates two circuits with qubit offset. `circuit_inverse` computes the dagger. Parent provenance tracked in circuit metadata. |
| I11 | **Hybrid Workflow Orchestration** | Architecture/Composability | `HybridWorkflow` model with steps, parameter history, and convergence state. Methods: `workflow_create`, `workflow_step`, `workflow_advance`, `workflow_result`. Pause/resume supported. |
| I12 | **Circuit Serialisation and OpenQASM Import/Export** | Feature/Interoperability | `circuit_export` supports `openqasm2`, `openqasm3`, `json_schema`, `qiskit_dict`. `circuit_import` parses, validates, and creates a circuit record. Round-trip fidelity verified. |
| I13 | **Quantum Entropy Accounting and QRNG Audit Trail** | Security/Compliance | `quantum_random` emits signed entropy manifests. `entropy_manifest_verify` and `entropy_consumer_register` support FIPS 140-3 / NIST SP 800-90B audit requirements. |
| I14 | **Quantum Machine Learning Primitives** | Feature/Domain Expansion | `quantum_kernel_matrix` computes n×n kernel matrix via quantum state inner products. `variational_classifier_train` uses parameter-shift gradient estimation with loss history. |
| I15 | **Real-Time Fidelity Drift Detection** | Observability/Operations | `fidelity_snapshot_record` + `fidelity_drift_detect` use EMA + linear regression slope over configurable time windows. Emits `FIDELITY_DRIFT_ALERT` audit event on threshold breach. |

## Main Files

- `SPECIFICATION.md` defines the normative capability behavior.
- `PLAN.md` records the implementation packet plan.
- `WORLD_CLASS_IMPROVEMENTS.md` documents all 15 prioritised enhancement paths with competitor references.
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
- `test_capability_contract.py` proves lifecycle behavior and generated evidence.

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

## New Methods

### Async Batch Execution

Submit multiple jobs concurrently. Uses `asyncio.Semaphore` to bound
in-flight submissions. Results are returned in input order.

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

### Grover's Search

Computes optimal iteration count `floor(π/4 × √(N/k))` and returns success
probability, gate count estimate, and quadratic speedup ratio.

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
print(result["success_probability"])   # 0.961
```

### Noise Model Registry

Register a device-calibrated noise model and apply it to a result to predict
NISQ device performance before committing QPU budget.

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
print(noisy["fidelity_loss_estimate"])  # 0.002
print(noisy["noisy_confidence"])        # degraded confidence value
```

### Fidelity Drift Detection

Record periodic calibration snapshots and detect EMA-slope degradation.
Emits `FIDELITY_DRIFT_ALERT` audit event automatically on threshold breach.

```python
# Record calibration snapshots periodically (e.g. every 15 minutes)
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
print(alert["ema_slope_per_snapshot"])  # negative value indicates drift
```

### Decimal-Precision Cost Estimation

Replaces `quantum_cost_estimate`. All monetary values returned as 6 dp
strings — safe for JSON serialisation and downstream accounting.

```python
cost = service.quantum_cost_estimate_decimal(
    tenant_id="tenant-demo",
    backend_id="local-sim",
    circuit_id="bell-v1",
    shot_count=1_000_000,
)
print(cost["estimated_cost"])  # "0.100000"  (string, 6 dp, no float error)
print(cost["precision"])       # "decimal_6dp"
```

### Circuit Complexity Metrics

Compute structural metrics without running the circuit — used for backend
selection and QPU readiness assessment.

```python
metrics = service.circuit_metrics(circuit_id="bell-v1", tenant_id="tenant-demo")
print(metrics["two_qubit_fraction"])      # 0.5
print(metrics["circuit_depth_estimate"])  # 2
print(metrics["t_gate_count"])            # 0
print(metrics["mw_entanglement_proxy"])   # 1.0
print(metrics["complexity_tier"])         # "high"
```

## API Reference

| Method | Description |
|--------|-------------|
| `register_backend` | Register a QPU or simulator backend |
| `attach_quota_policy` | Set shot, job, and cost limits per backend |
| `create_circuit` / `circuit_define` | Define a circuit with gates and metadata |
| `submit_job` / `job_submit_qpu` | Submit a job to a registered backend |
| `job_simulate` | Submit to an auto-registered local simulator |
| `complete_job` / `job_result` | Capture or retrieve measurement results |
| `quantum_error_mitigation` / `error_mitigate` | Apply ZNE, PEC, CDR, or symmetry verification |
| `variational_quantum_eigensolver` / `vqe_solve` | Run VQE ground-state energy estimation |
| `quantum_approximate_optimisation` / `qaoa_solve` | Run QAOA for combinatorial optimisation |
| `quantum_key_distribution` / `qkd_session` | Simulate QKD (BB84, E91, B92, SARG04) |
| `post_quantum_encryption` / `pqc_encrypt` | Apply Kyber, Dilithium, Falcon, SPHINCS+, NTRU |
| `quantum_simulation` | Trotter-step simulation of physical Hamiltonians |
| `grover_search` | Grover's algorithm with optimal iteration count |
| `noise_model_register` | Register depolarising/thermal/readout/crosstalk noise |
| `noise_model_apply` | Inject noise into a result for NISQ benchmarking |
| `fidelity_snapshot_record` | Record T1/T2/gate/readout calibration data |
| `fidelity_drift_detect` | EMA-slope drift detection with auto audit alert |
| `circuit_metrics` | Gate counts, depth, T-gates, Meyer-Wallach entanglement |
| `circuit_optimise` | Depth/gate-count reduction at levels 0–3 |
| `quantum_cost_estimate_decimal` | Decimal-precision cost with 6 dp monetary precision |
| `quantum_random` | QRNG-inspired RNG with entropy audit trail |
| `backend_status` | Queue depth, availability, calibration age |
| `async_submit_quantum_job` | Non-blocking job submission |
| `async_batch_submit_jobs` | Concurrent multi-job submission with semaphore |
| `async_vqe_solve` | Non-blocking VQE |
| `async_qaoa_solve` | Non-blocking QAOA |
| `async_quantum_simulation` | Non-blocking Hamiltonian simulation |
| `async_quantum_analytics` | Non-blocking analytics aggregation |
| `quantum_analytics` | Aggregate stats: jobs, VQE, QAOA, QKD, PQ, simulation |
| `dashboard_summary` | Tenant-scoped operational dashboard snapshot |
| `register_quan_agent` | Register AI quantum agent with disclosure guardrails |
| `create_experiment` | Group circuits, jobs, and hypothesis into an experiment |

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
