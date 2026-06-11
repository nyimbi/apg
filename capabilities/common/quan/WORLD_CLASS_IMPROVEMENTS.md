# QUAN Capability — World-Class Improvements

**Capability**: Quantum Computing (`quan`)
**Author**: Nyimbi Odero
**Date**: 2026-06-11
© 2025 Datacraft — www.datacraft.co.ke

---

## 1. Full Async Service Layer

**Problem**: Every public method is synchronous. Real quantum backends (IBM Q, IonQ, Rigetti) expose async REST APIs. Blocking I/O in a synchronous facade forces callers to use thread pools, introduces latency, and prevents horizontal scaling under concurrent workloads.

**Improvement**: Provide `async` variants of all I/O-bound methods (submit_job, complete_job, quantum_simulation, VQE, QAOA, QKD). Use `asyncio.sleep` to simulate backend latency in tests; real adapters await HTTP clients. Expose a parallel batch runner that executes N jobs concurrently via `asyncio.gather`.

---

## 2. Statevector Simulator with Proper Complex Amplitudes

**Problem**: Measurement counts are generated deterministically from a hash — not from an actual quantum state. This makes the simulator useless for validating correctness of algorithms such as Grover search, QPE, or Shor.

**Improvement**: Implement a minimal numpy-backed statevector simulator that tracks a complex `2^n` amplitude array. Support H, X, Y, Z, S, T, CX, CCX, RZ, RX, RY, SWAP gates with matrix representations. Expose `statevector()` and `density_matrix()` read-outs. Cap at 20 qubits with a clear error message beyond that.

---

## 3. Noise Model Integration

**Problem**: The simulator produces ideal (noiseless) results. NISQ devices exhibit gate errors, decoherence (T1/T2), and readout assignment errors. Without a noise model, benchmarks are meaningless against real hardware.

**Improvement**: Add a `NoiseModel` dataclass parameterised by per-gate depolarising probability, T1/T2 relaxation times, and readout confusion matrix. Integrate it into the statevector simulator via Kraus operator application after each gate. Support `IBM_nairobi`, `IBM_perth`, and `custom` presets.

---

## 4. Parametric Circuit Binding

**Problem**: Circuits are stored as flat gate lists with no parameter support. VQE and QAOA require circuits with free angular parameters (θ) that are bound at each optimiser iteration, not at circuit-definition time.

**Improvement**: Extend `QuantumCircuit` to support `GateInstruction(name, qubits, params)` objects with symbolic or numeric parameters. Add `circuit_bind_parameters(circuit_id, param_values)` that returns a bound copy without mutating the original. Track parameter provenance for reproducibility.

---

## 5. Circuit Transpilation Pipeline

**Problem**: Circuits are submitted using an abstract gate set regardless of the target backend's native gate set and qubit connectivity. IBM backends restrict to `{CX, ID, RZ, SX, X}` and a linear connectivity. Transpilation is a prerequisite for real execution.

**Improvement**: Implement a multi-pass transpiler: (a) gate decomposition to backend native set, (b) qubit routing via SWAP insertion based on a coupling map, (c) gate cancellation, (d) commutation analysis. Accept `CouplingMap` configs per backend. Expose `transpile_circuit(circuit_id, backend_id)` returning the physical circuit depth and SWAP overhead.

---

## 6. Entanglement and Complexity Metrics

**Problem**: There is no way to assess circuit complexity before running it. Circuit depth, T-gate count, and entanglement structure are critical for estimating QPU execution time and noise sensitivity.

**Improvement**: Add `circuit_metrics(circuit_id)` returning: gate count by type, two-qubit gate fraction, circuit depth (critical path length), T-gate count (magic state overhead), and Meyer-Wallach global entanglement measure estimate. These metrics guide backend selection and optimisation decisions.

---

## 7. Real QAOA Parameter Optimisation Loop

**Problem**: The current QAOA implementation returns a synthetic approximation ratio without actually optimising the variational parameters γ and β. The solution bitstring is always the zero string.

**Improvement**: Implement a two-level optimisation loop: (a) compute the cost-function expectation value from the statevector for given γ/β, (b) call a gradient-free optimiser (Nelder-Mead via scipy or a pure-Python COBYLA) to minimise it. Return the best-found bitstring, exact cut value, and per-iteration expectation value trace.

---

## 8. Tensor Network Contraction Backend

**Problem**: The statevector simulator has exponential memory in qubit count. Many practical circuits (shallow, MPS-structured, low entanglement) can be simulated with polynomial cost via tensor networks.

**Improvement**: Add a `TensorNetworkBackend` that represents circuit state as a matrix product state (MPS) with configurable bond dimension χ. Automatically select this backend for circuits exceeding 20 qubits. Expose `backend_type=mps` with a `bond_dimension` configuration field.

---

## 9. Quantum Volume Benchmarking

**Problem**: QPU quality metrics are inconsistently reported. Quantum Volume (QV) is IBM's standard figure of merit and is universally understood, yet the capability exposes no standardised benchmarking.

**Improvement**: Implement `quantum_volume_benchmark(backend_id, n_qubits)` that constructs random QV circuits, runs them on the simulator (or real backend adapter), computes heavy output probability, and reports whether the QV threshold is achieved. Store benchmark history for regression tracking.

---

## 10. Structured Event Sourcing and Replay

**Problem**: `_audit_events` is a flat in-memory dict. There is no ordering guarantee, no event replay, and no way to reconstruct state from events alone. The system cannot be debugged from audit logs.

**Improvement**: Replace the audit store with an append-only `EventLog` that assigns monotonic sequence numbers and wall-clock timestamps. Expose `replay_tenant_state(tenant_id, up_to_seq)` that reconstructs a `QuantumComputingService` snapshot from event history alone. Emit events to the Bytewax stream adapter synchronously via a pluggable sink.

---

## 11. Multi-Tenant Quota Enforcement with Windowed Rate Limits

**Problem**: Quota policies store per-backend limits but there is no actual enforcement of `max_jobs_per_day` — the counter is never checked or incremented. A tenant can exceed their daily limit without any error.

**Improvement**: Add a `QuotaLedger` that tracks per-tenant per-backend job counts in sliding 24-hour windows. Increment atomically on job submission and check before accepting. Expose `quota_usage(tenant_id, backend_id)` returning consumed/limit/window_reset_at. Support bursting allowances and grace quotas.

---

## 12. Hybrid Classical-Quantum Workflow Orchestration

**Problem**: VQE and QAOA are monolithic single-call operations. Real hybrid algorithms alternate between quantum circuit execution (QPU) and classical optimisation steps, each step potentially running on different backends or requiring human review checkpoints.

**Improvement**: Introduce a `HybridWorkflow` model with steps, current step, parameter history, and convergence state. Add `workflow_create`, `workflow_step`, `workflow_advance`, and `workflow_result` methods. Each step records the classical parameters used, the quantum result obtained, and the updated objective value. Support pause/resume and per-step audit gates.

---

## 13. Quantum Machine Learning Primitives

**Problem**: The capability covers optimisation (QAOA) and chemistry (VQE) but omits quantum machine learning, one of the fastest-growing near-term applications. QML primitives (quantum kernel estimation, variational classifiers, data re-uploading) are absent.

**Improvement**: Add `quantum_kernel_matrix(feature_map, data_points)` that computes the `n×n` kernel matrix via inner products of quantum feature states. Add `variational_classifier_train(circuit_template, training_data, labels)` using parameter-shift gradient estimation. Expose these as first-class service methods with telemetry.

---

## 14. Circuit Serialisation and Import/Export

**Problem**: Circuits are opaque Python dicts with no portable serialisation. Teams cannot share circuits across tenants, import from OpenQASM 2/3, or export to Qiskit/Cirq format for cross-platform validation.

**Improvement**: Implement `circuit_export(circuit_id, format)` supporting `openqasm2`, `openqasm3`, `json_schema`, and `qiskit_dict`. Implement `circuit_import(serialised, format, owner, tenant_id)` that parses and validates inbound circuit definitions. Round-trip fidelity tests verify import→export→import identity.

---

## 15. Observability and Distributed Tracing Integration

**Problem**: Service calls emit audit events but there are no latency histograms, error-rate counters, or distributed trace spans. Operating the system in production is blind — there is no way to diagnose slow backends, identify hot circuits, or correlate quantum job latency with upstream SLO violations.

**Improvement**: Instrument every public method with OpenTelemetry spans (`quan.submit_job`, `quan.vqe_solve`, etc.) carrying tenant_id, backend_id, and circuit_id as span attributes. Emit Prometheus-compatible counters and histograms for job submission rate, queue depth per backend, result confidence distribution, and error mitigation frequency. Wire into the `moni` and `logt` composition adapters.
