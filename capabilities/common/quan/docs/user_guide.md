# QUAN Quantum Computing — User Guide

**Capability ID**: `quan` | **Domain**: `common` | **Version**: `1.1.0`
**Author**: Nyimbi Odero | © 2025 Datacraft — www.datacraft.co.ke

---

## Overview

QUAN provides a tenant-scoped quantum computing runtime for APG applications.
It covers the full lifecycle from backend registration through job execution,
result capture, error mitigation, hybrid algorithms (VQE, QAOA), quantum
cryptography (QKD, post-quantum encryption), physical system simulation,
Grover search, noise modelling, and real-time fidelity drift detection.

All operations are tenant-isolated. Every mutating action emits an audit event.
Async variants of I/O-bound methods enable non-blocking use in FastAPI,
Bytewax pipelines, and other async runtimes.

---

## Installation

```bash
pip install apg-common-quan
```

---

## Quick Start

```python
from capabilities.common.quan import QuanService

svc = QuanService()

# 1. Register a simulator backend
backend = svc.register_backend(
    backend_id="sim-8q",
    tenant_id="acme",
    name="Local 8-qubit simulator",
    provider="local",
    backend_type="simulator",
    qubit_count=8,
    approved=True,
)

# 2. Attach a quota policy
svc.attach_quota_policy(
    policy_id="quota-sim",
    tenant_id="acme",
    backend_id="sim-8q",
    max_shots_per_job=8192,
    max_jobs_per_day=100,
    cost_limit=10.0,
)

# 3. Define a circuit
circuit = svc.create_circuit(
    circuit_id="ghz-3q",
    tenant_id="acme",
    name="GHZ state 3-qubit",
    owner="alice",
    version="1.0",
    qubits_required=3,
    gates=["h", "cx", "cx", "measure"],
)

# 4. Submit and retrieve a result
job = svc.submit_job(
    job_id="job-ghz-01",
    tenant_id="acme",
    backend_id="sim-8q",
    circuit_id="ghz-3q",
    submitted_by="alice",
    shot_count=1024,
)
result = svc.complete_job("res-ghz-01", "acme", job["id"])
print(result["measurement_counts"])
```

---

## Core Concepts

### Tenants

Every resource (backend, circuit, job, result, experiment) is scoped to a
`tenant_id`. Passing an empty or blank `tenant_id` raises `ValueError:
tenant_id_required` immediately. Never share tenant IDs across organisational
boundaries.

### Backends

A backend represents a quantum processing unit (QPU) or simulator. Key fields:

| Field | Description |
|---|---|
| `provider` | `local`, `ibm`, `ionq`, `rigetti`, `quantinuum`, `aws_braket` |
| `backend_type` | `simulator`, `qpu`, `photonic`, `neutral_atom` |
| `qubit_count` | Physical qubits available |
| `approved` | Must be `True` before jobs can be submitted |
| `credentials_ref` | Key reference for the `keym` adapter |
| `simulator_fallback` | Auto-route to local sim if QPU unavailable |

### Quota Policies

Attach at most one quota policy per backend per tenant. The policy enforces:

- `max_shots_per_job`: hard upper bound on shots per submission.
- `max_jobs_per_day`: daily submission ceiling (sliding 24-hour window in
  production; stored but not currently windowed in the in-process store).
- `cost_limit`: maximum estimated cost in USD per job.

### Circuits

Circuits are immutable after creation. Use `circuit_define` or `create_circuit`
and specify a new `version` to update. Gates are stored as a normalised tuple
of lowercase gate names: `h`, `cx`, `t`, `rz`, `measure`, etc.

### Jobs and Results

`submit_job` validates quota policy, qubit capacity, and cost limit before
accepting. `complete_job` generates deterministic measurement counts from the
job and circuit IDs. In production, the adapter layer replaces this with real
QPU or simulator results.

---

## Algorithms

### VQE — Variational Quantum Eigensolver

Find the ground state energy of a Hamiltonian:

```python
result = svc.variational_quantum_eigensolver(
    hamiltonian={
        "n_qubits": 4,
        "terms": [
            {"pauli": "ZZ", "qubits": [0, 1], "coeff": -1.0},
            {"pauli": "XX", "qubits": [1, 2], "coeff": 0.5},
        ],
    },
    ansatz={"type": "hardware_efficient", "layers": 3},
    optimiser="cobyla",
    tenant_id="acme",
    max_iterations=200,
)
print(result["ground_state_energy"])  # hartree units
print(result["converged"])
```

Supported optimisers: `cobyla`, `spsa`, `adam`, `l_bfgs_b`, `gradient_descent`.

### QAOA — Quantum Approximate Optimisation

Solve combinatorial optimisation problems:

```python
result = svc.quantum_approximate_optimisation(
    problem_type="max_cut",
    graph={
        "nodes": [0, 1, 2, 3],
        "edges": [[0, 1, 1.0], [1, 2, 1.0], [2, 3, 1.0], [3, 0, 1.0]],
    },
    layers=5,
    tenant_id="acme",
    shots=2048,
)
print(result["approximation_ratio"])  # approaches 1.0 with more layers
print(result["optimal_value"])
```

Supported problem types: `max_cut`, `graph_colouring`, `portfolio_optimisation`,
`tsp`, `vertex_cover`.

### Grover's Search

Quadratic speedup over classical brute-force search:

```python
result = svc.grover_search(
    oracle_spec={"function_description": "Find 4 marked items in 256-element space"},
    n_qubits=8,      # search space = 2^8 = 256
    marked_items=4,
    tenant_id="acme",
    shots=2048,
)
print(result["optimal_iterations"])       # 12
print(result["success_probability"])      # ~0.9998
print(result["quantum_speedup_ratio"])    # ~10.7x vs classical brute force
```

---

## Quantum Cryptography

### Quantum Key Distribution

```python
session = svc.quantum_key_distribution(
    endpoint_a="alice-node",
    endpoint_b="bob-node",
    key_length=256,
    protocol="bb84",
    tenant_id="acme",
)
print(session["qber"])               # quantum bit error rate (should be < 0.11)
print(session["eavesdropping_detected"])  # True if QBER > 11%
print(session["key_hash"])           # SHA-256 of key material — never the raw key
```

Supported protocols: `bb84`, `e91`, `b92`, `sarg04`.

### Post-Quantum Encryption

```python
enc = svc.post_quantum_encryption(
    data={"payload": "sensitive classification data"},
    algorithm="kyber",
    tenant_id="acme",
    key_size_bits=512,
)
print(enc["nist_security_level"])   # 3
print(enc["quantum_safe"])          # True
```

Supported algorithms: `kyber`, `dilithium`, `falcon`, `sphincs_plus`, `ntru`,
`crystals_dilithium`.

---

## Error Mitigation

Apply error mitigation to any recorded result:

```python
mitigated = svc.quantum_error_mitigation(
    result_id="res-ghz-01",
    method="zero_noise_extrapolation",
    tenant_id="acme",
)
print(mitigated["fidelity_improvement"])   # e.g. 0.15
print(mitigated["mitigated_confidence"])
```

Supported methods: `zero_noise_extrapolation`, `probabilistic_error_cancellation`,
`clifford_data_regression`, `symmetry_verification`.

---

## Noise Models

Register a noise model and apply it to simulator results to predict QPU behaviour:

```python
nm = svc.noise_model_register(
    model_id="ibm-nairobi",
    tenant_id="acme",
    model_type="depolarising",
    params={"gate_error_rate": 0.001, "two_qubit_error_rate": 0.01},
)

noisy = svc.noise_model_apply(
    result_id="res-ghz-01",
    noise_model_id=nm["noise_model_id"],
    tenant_id="acme",
)
print(noisy["fidelity_loss_estimate"])  # e.g. 0.004
print(noisy["noisy_confidence"])
```

Supported model types:

| Type | Required params |
|---|---|
| `depolarising` | `gate_error_rate` |
| `thermal_relaxation` | `t1_us`, `t2_us` |
| `readout_error` | `p0_given_1`, `p1_given_0` |
| `crosstalk` | `zz_coupling_mhz` |

---

## Fidelity Drift Detection

Record calibration snapshots and detect degradation automatically:

```python
# Record snapshots (call every 15 minutes in production)
for gate_fidelity in [0.998, 0.997, 0.993, 0.985, 0.971]:
    svc.fidelity_snapshot_record(
        backend_id="qpu-01",
        tenant_id="acme",
        gate_fidelity=gate_fidelity,
        readout_fidelity=0.995,
        t1_us=120.0,
        t2_us=80.0,
    )

alert = svc.fidelity_drift_detect(
    backend_id="qpu-01",
    tenant_id="acme",
    window_snapshots=5,
    drift_threshold=0.005,
)
print(alert["drift_detected"])       # True
print(alert["recommendation"])       # "halt_jobs_and_recalibrate"
print(alert["ema_slope_per_snapshot"])  # negative = degrading fidelity
```

When drift is detected, a `FIDELITY_DRIFT_ALERT` audit event is automatically
emitted. Wire the `audl` adapter to route this event to your alerting system.

---

## Circuit Complexity Metrics

Analyse circuit structure before committing to QPU budget:

```python
metrics = svc.circuit_metrics(circuit_id="ghz-3q", tenant_id="acme")
print(metrics["total_gate_count"])          # 4
print(metrics["two_qubit_gate_count"])      # 2
print(metrics["two_qubit_fraction"])        # 0.5
print(metrics["t_gate_count"])              # 0
print(metrics["circuit_depth_estimate"])    # 3
print(metrics["mw_entanglement_proxy"])     # 1.0 (highly entangled)
print(metrics["complexity_tier"])           # "high"
```

Use `complexity_tier` to auto-route circuits: `low` → simulator is adequate;
`high` → QPU execution recommended.

---

## Decimal-Precision Cost Estimation

Use `quantum_cost_estimate_decimal` for any financial accounting workflow:

```python
cost = svc.quantum_cost_estimate_decimal(
    tenant_id="acme",
    backend_id="sim-8q",
    circuit_id="ghz-3q",
    shot_count=1_000_000,
)
print(cost["estimated_cost"])   # "0.100000"  — str, 6 dp, no float rounding error
print(cost["within_budget"])    # True / False
print(cost["quota_limit"])      # "10.000000" or None
```

---

## Async Execution

All I/O-bound operations have async variants for use in FastAPI, Bytewax, and
asyncio pipelines:

```python
import asyncio
from capabilities.common.quan import QuanService

svc = QuanService()

# Single async job
result = asyncio.run(
    svc.async_submit_quantum_job(
        circuit_definition={"name": "bell", "qubits": 2, "gates": ["h", "cx"]},
        backend="sim-8q",
        shots=1024,
        tenant_id="acme",
        submitted_by="alice",
        simulated_latency_ms=50,  # mimics real QPU queue wait
    )
)

# Batch parallel submission (up to 8 concurrent)
batch = [
    {"circuit_definition": {"name": f"circuit-{i}", "qubits": 2, "gates": ["h", "cx"]},
     "backend": "sim-8q", "shots": 512}
    for i in range(20)
]
results = asyncio.run(
    svc.async_batch_submit_jobs(batch, tenant_id="acme", concurrency_limit=8)
)
print(len(results))  # 20
```

Available async methods:

| Method | Description |
|---|---|
| `async_submit_quantum_job` | Single job submission |
| `async_batch_submit_jobs` | Parallel batch with semaphore-bounded concurrency |
| `async_vqe_solve` | Non-blocking VQE |
| `async_qaoa_solve` | Non-blocking QAOA |
| `async_quantum_simulation` | Non-blocking physical simulation |
| `async_quantum_analytics` | Non-blocking analytics aggregation |

---

## Physical System Simulation

Simulate quantum systems with Trotter decomposition:

```python
result = svc.quantum_simulation(
    physical_system={
        "type": "ising",
        "n_sites": 8,
        "coupling_constant": 1.0,
        "magnetic_field": 0.5,
    },
    time_steps=100,
    tenant_id="acme",
    dt=0.005,
)
print(result["final_energy"])
print(result["final_state_fidelity"])
print(result["trotter_error_estimate"])
```

Supported system types: `ising`, `hubbard`, `transverse_field`, `heisenberg`,
`bose_hubbard`.

---

## Quantum Simulation (QRNG)

Generate quantum-inspired random bits with full audit trail:

```python
rng = svc.quantum_random(
    tenant_id="acme",
    n_bits=256,
    format="hex",
)
print(rng["output"])        # 64-char hex string
print(rng["entropy_source"])  # "qrng_simulator"
```

Formats: `hex`, `base64`, `int`, `bytes_list`. Range: 8–8192 bits.

---

## AI Quantum Agents

```python
agent = svc.register_quan_agent(
    tenant_id="acme",
    name="Cost gatekeeper",
    runtime="claude_code",
    role="cost_reviewer",
    scope="Approve jobs where estimated_cost > 1.0 USD",
)
```

Runtimes: `codex`, `claude_code`, `opencode`, `pi`.

---

## Dashboard and Analytics

```python
summary = svc.dashboard_summary(tenant_id="acme")
print(summary["job_count"])
print(summary["vqe_run_count"])
print(summary["qkd_session_count"])

analytics = svc._quantum_analytics_impl(tenant_id="acme", period="last_30d")
print(analytics["job_completion_rate"])
print(analytics["average_result_confidence"])
```

---

## UI Routes

| Path | Permission | Nav Group |
|---|---|---|
| `/quan/dashboard` | `quan:view` | Overview |
| `/quan/backends` | `quan:manage_backends` | Backends |
| `/quan/circuits` | `quan:experiment` | Circuits |
| `/quan/jobs` | `quan:run_jobs` | Jobs |
| `/quan/experiments` | `quan:experiment` | Experiments |
| `/quan/results` | `quan:view` | Results |
| `/quan/agents` | `quan:admin` | Operations |
| `/quan/audit` | `quan:admin` | Governance |

---

## Composition

| Capability | Role |
|---|---|
| `aicr` | AI-assisted experiment analysis and agent orchestration |
| `encr` | Encryption policy on sensitive circuit inputs |
| `keym` | Provider credential references |
| `audl` | Durable audit evidence and event routing |
| `moni` | Operational telemetry and SLO monitoring |
| `logt` | Structured log aggregation and diagnostics |
| `comp` | Regulated experiment and cryptographic-transition review |

Batch quantum mutation and quantum job lifecycle events must route through the
`bytewax` event-stream adapter.

---

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or
environment variables prefixed `QUAN_`:

| Variable | Default | Description |
|---|---|---|
| `QUAN_DEFAULT_SHOTS` | `1024` | Default shot count for new jobs |
| `QUAN_SIMULATOR_FALLBACK` | `true` | Auto-route to local sim when QPU unavailable |
| `QUAN_AUDIT_RETENTION_DAYS` | `365` | Audit event retention |
| `QUAN_RESULT_RETENTION_DAYS` | `90` | Result retention |
| `QUAN_MAX_QUBITS_STATEVECTOR` | `20` | Statevector simulator qubit cap |

---

## Further Reading

- `service.py` — Business logic, 50+ methods
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference and code examples
- `SPECIFICATION.md` — Normative capability specification
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 enhancement roadmap items
