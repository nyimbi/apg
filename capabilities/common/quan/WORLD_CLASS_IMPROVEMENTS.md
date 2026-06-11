# QUAN — World-Class Improvements

**Capability**: Quantum Computing (`quan`) | **Domain**: `common`
**Author**: Nyimbi Odero | © 2025 Datacraft — www.datacraft.co.ke

---

### I1. Full Async Service Layer

**Category**: Architecture
**Justification**: Every public method blocks the event loop. IBM Q, IonQ, and Rigetti all expose async REST APIs; synchronous wrappers force thread-pool overhead and prevent 10x horizontal scaling under concurrent job workloads.
**Implementation**: Convert all public methods to `async def`. Replace `_require_*` helpers with async variants. Use `asyncio.gather` in analytics aggregation. Ship an `AsyncQuantumComputingService` alias; provide a thin sync shim via `asyncio.run` for legacy callers.
**Competitor**: IBM `qiskit-ibm-runtime>=0.20` is fully async; Google Cirq uses `asyncio` natively for cloud execution paths.

---

### I2. Decimal-Precision Cost Accounting

**Category**: Correctness / Finance
**Justification**: `float` arithmetic on `cost_limit` and `estimated_cost` silently accumulates rounding error — $0.000001/shot × 10M shots produces a $10 budget error that can breach monthly limits. `Decimal` with `ROUND_HALF_EVEN` is the only correct approach for money fields.
**Implementation**: Replace all cost `float` fields in `models.py`, `service.py`, and `quantum_runtime.py` with `Decimal`. Quantize to `Decimal("0.000001")` at all write boundaries. Serialise as `str` in `to_dict()` to preserve precision across JSON round-trips.
**Competitor**: AWS Braket pricing SDK uses `Decimal` internally; Rigetti QCS bills at 6 decimal-place precision.

---

### I3. Statevector Simulator with Complex Amplitudes

**Category**: Correctness / Simulation
**Justification**: Measurement counts derived from a SHA-256 hash produce no valid quantum state. Grover, QPE, and Shor correctness verification is impossible without an actual `2^n` complex amplitude array. A real simulator makes the VQE/QAOA result physically meaningful rather than synthetic.
**Implementation**: Add `statevector_simulate(circuit_id, tenant_id)` backed by a numpy complex128 amplitude array. Gate set: H, X, Y, Z, S, T, CX, CCX, RZ, RX, RY, SWAP. Expose `statevector()` and `density_matrix()` readouts. Cap at 20 qubits; raise `QubitCapacityError` beyond that.
**Competitor**: Qiskit Aer `StatevectorSimulator`; Cirq `Simulator`; PennyLane `default.qubit`.

---

### I4. Parametric Circuit Binding

**Category**: Feature / VQE-QAOA
**Justification**: VQE and QAOA require circuits with free angular parameters θ bound at each optimiser iteration. Storing flat gate string lists prevents parameter re-use, makes gradient computation impossible, and locks callers into recreating circuits on every call — a 100x overhead in typical 500-iteration VQE runs.
**Implementation**: Extend `QuantumCircuit` with a `GateInstruction(name, qubits, params)` structure. Add `circuit_bind_parameters(circuit_id, param_values, tenant_id)` returning a bound copy without mutating the original. Track symbolic parameter names and provenance for reproducibility audits.
**Competitor**: Qiskit `ParameterVector`; PennyLane `qml.templates`; Cirq `Sweep`.

---

### I5. Circuit Transpilation with Backend Topology

**Category**: Feature / Compilation
**Justification**: Abstract gate sets fail silently on real QPUs that restrict to native gate sets (`{CX, ID, RZ, SX, X}` on IBM) and linear/heavy-hex connectivity. Without transpilation, QPU execution is impossible and depth estimates are fiction.
**Implementation**: Add `circuit_transpile(circuit_id, backend_id, optimisation_level, tenant_id)`. Perform three passes: (a) gate decomposition to backend native set, (b) SWAP insertion via coupling map, (c) gate cancellation. Accept `coupling_map` in backend metadata. Return physical circuit depth, SWAP count, and fidelity penalty estimate.
**Competitor**: Qiskit `transpile()` with `coupling_map`; Google Cirq `RouteCQC`; Quantinuum `pytket` compiler.

---

### I6. Noise Model Registry

**Category**: Feature / Fidelity
**Justification**: Ideal simulator results are misleading for QPU readiness assessment. Configurable noise models (depolarising, thermal relaxation, readout error) allow researchers to predict QPU performance from simulator runs, reducing wasted QPU budget by 40–60%.
**Implementation**: Add `noise_model_register(model_id, tenant_id, model_type, params)` and `noise_model_apply(result_id, noise_model_id, tenant_id)`. Types: `depolarising`, `thermal_relaxation`, `readout_error`, `crosstalk`. Apply Kraus operator noise injection to measurement counts proportional to error parameters. Store in `_noise_models`.
**Competitor**: Qiskit Aer `NoiseModel`; Cirq `ConstantQubitNoiseModel`; Amazon Braket local simulator noise.

---

### I7. Grover's Search Oracle Interface

**Category**: Feature / Algorithms
**Justification**: Grover provides quadratic search speedup (O(√N) vs O(N)) and is among the most practically applicable quantum algorithms. Omitting it leaves the "big three" near-term algorithms (VQE, QAOA, Grover) incomplete and prevents use in database search, SAT solving, and collision-finding applications.
**Implementation**: Add `grover_search(oracle_spec, n_qubits, marked_items, tenant_id, shots)`. Compute optimal iteration count `floor(π/4 × √(N/k))`, success probability, and synthetic measurement distribution peaked on marked states. Estimate gate count as `O(n × iterations)`.
**Competitor**: Qiskit Algorithms `Grover` class; PennyLane `qml.GroverOperator`; Cirq Grover example.

---

### I8. Windowed Quota Enforcement

**Category**: Correctness / Multi-tenancy
**Justification**: `max_jobs_per_day` is stored in quota policies but never enforced — the counter is never checked or incremented. Any tenant can exceed daily limits without restriction, which breaks SLA guarantees and enables runaway billing. This is a correctness bug, not a feature gap.
**Implementation**: Add a `QuotaLedger` tracking per-tenant, per-backend job submission timestamps in sliding 24-hour windows. Enforce before `submit_job` completes. Add `quota_usage(tenant_id, backend_id)` returning `{consumed, limit, window_reset_at}`. Support burst allowances and grace quotas.
**Competitor**: IBM Quantum Network fair-share scheduling; IonQ cloud queue priority tiers.

---

### I9. Quantum Volume Benchmarking

**Category**: Quality / Observability
**Justification**: QPU quality metrics are inconsistently reported without a standard figure of merit. Quantum Volume is the universal industry benchmark (IBM, Quantinuum, IonQ all publish QV); without it, backend comparison across providers is subjective and untrustworthy.
**Implementation**: Add `quantum_volume_benchmark(backend_id, n_qubits, trials, tenant_id)`. Generate random QV circuits (square random unitaries), compute heavy output probability from statevector simulation, and report whether the 2/3 heavy output threshold is achieved. Store benchmark history in `_benchmarks` for regression tracking.
**Competitor**: IBM Quantum `QuantumVolumeFitter`; Quantinuum H-Series QV reports; IonQ #AQ metric (closely related).

---

### I10. Quantum Circuit Composition Algebra

**Category**: Feature / Developer Experience
**Justification**: Complex quantum programs are built from reusable sub-circuit modules (state preparation, oracle, diffuser, ansatz). Without a composition algebra, gate-list manual concatenation with qubit register offset tracking is error-prone at scale and takes 5x longer to write than it should.
**Implementation**: Add `circuit_compose(base_circuit_id, subcircuit_id, qubit_offset, tenant_id, composed_id)` producing a new circuit whose gates are the concatenation of both, with qubit count `= max(base_qubits, subcircuit_qubits + offset)`. Add `circuit_inverse(circuit_id, tenant_id)` computing the dagger. Track parent provenance in circuit metadata.
**Competitor**: Qiskit `QuantumCircuit.compose()` and `.inverse()`; Cirq `Circuit.concat_ragged()`; PennyLane adjoint transforms.

---

### I11. Hybrid Workflow Orchestration

**Category**: Architecture / Composability
**Justification**: VQE and QAOA are implemented as monolithic single calls. Real hybrid algorithms alternate quantum circuit execution with classical optimisation steps, each potentially requiring human review, different backends, or pause/resume across sessions — impossible with the current design.
**Implementation**: Add `HybridWorkflow` model with steps, current step, parameter history, and convergence state. Methods: `workflow_create`, `workflow_step`, `workflow_advance`, `workflow_result`. Each step records classical parameters, quantum result, and updated objective value. Support pause/resume and per-step audit gates.
**Competitor**: Qiskit Runtime `Sampler`/`Estimator` primitives in session mode; PennyLane `qml.qnode` + classical optimizer loop.

---

### I12. Circuit Serialisation and OpenQASM Import/Export

**Category**: Feature / Interoperability
**Justification**: Circuits stored as opaque Python dicts cannot be shared across tenants, imported from Qiskit/Cirq ecosystems, or validated against external tools. OpenQASM 2/3 is the universal quantum assembly language; lack of support creates a hard ecosystem lock-in.
**Implementation**: Add `circuit_export(circuit_id, format, tenant_id)` supporting `openqasm2`, `openqasm3`, `json_schema`, `qiskit_dict`. Add `circuit_import(serialised, format, owner, tenant_id)` that parses, validates, and creates a new circuit record. Round-trip fidelity tests verify import→export→import gate identity.
**Competitor**: Qiskit `QuantumCircuit.qasm()` and `from_qasm_str()`; Cirq `cirq.to_json`; tket `pytket.qasm`.

---

### I13. Quantum Entropy Accounting and QRNG Audit Trail

**Category**: Security / Compliance
**Justification**: Regulated industries (finance, defence, healthcare) require a verifiable entropy audit trail for all cryptographic RNG. The current `quantum_random` method has no linkage between QRNG output and downstream cryptographic use, which fails FIPS 140-3 and NIST SP 800-90B requirements.
**Implementation**: Extend `quantum_random` to emit a signed entropy manifest (seed hash, extraction method, consumer registration). Add `entropy_manifest_verify(request_id, tenant_id)` and `entropy_consumer_register(consumer_id, purpose, tenant_id)`. All QRNG calls log to a dedicated `_entropy_ledger` with configurable retention policy.
**Competitor**: ID Quantique Quantis QRNG provides NIST-certified entropy attestation; Cambridge Quantum IronBridge API includes entropy certificates.

---

### I14. Quantum Machine Learning Primitives

**Category**: Feature / Domain Expansion
**Justification**: The capability covers chemistry (VQE) and combinatorics (QAOA) but omits quantum ML — one of the fastest-growing near-term application areas. Quantum kernel estimation and variational classifiers have demonstrated practical advantage on structured datasets and are production-ready on current NISQ hardware.
**Implementation**: Add `quantum_kernel_matrix(feature_map, data_points, tenant_id)` computing the n×n kernel matrix via quantum state inner products. Add `variational_classifier_train(circuit_template, training_data, labels, tenant_id)` using parameter-shift gradient estimation. Expose training loss history and prediction accuracy.
**Competitor**: PennyLane `qml.kernels`; Qiskit Machine Learning `FidelityQuantumKernel`; TensorFlow Quantum `tfq.layers.PQC`.

---

### I15. Real-Time Fidelity Drift Detection

**Category**: Observability / Operations
**Justification**: QPU fidelity degrades continuously due to environmental noise and decoherence. Without real-time calibration drift alerting, users run experiments on degraded hardware — wasting 20–30% of QPU budget and producing statistically invalid results. Drift detection is mandatory for production quantum operations.
**Implementation**: Add `fidelity_snapshot_record(backend_id, gate_fidelity, readout_fidelity, t1_us, t2_us, tenant_id)` storing time-series fidelity data. Add `fidelity_drift_detect(backend_id, tenant_id, window_h)` computing exponential moving average of gate fidelity, flagging `drift_detected` when slope exceeds configurable threshold. Emit `FIDELITY_DRIFT_ALERT` audit event automatically.
**Competitor**: IBM Quantum backend calibration API; IonQ system performance dashboard; Quantinuum H-Series calibration reporting.
