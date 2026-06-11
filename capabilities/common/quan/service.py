"""Service layer for executable Quantum Computing operations — expanded implementation."""

from __future__ import annotations

import hashlib
import math
import asyncio
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Any

from .capability_contract import (
	SUPPORTED_QUAN_AGENT_ROLES,
	SUPPORTED_QUAN_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
)
from .models import (
	QuanAgent,
	QuanAuditEvent,
	QuantumBackend,
	QuantumCircuit,
	QuantumExperiment,
	QuantumJob,
	QuantumQuotaPolicy,
	QuantumResult,
	utc_now,
)
from .quantum_runtime import (
	deterministic_measurements,
	estimate_job_cost,
	normalize_backend_type,
	normalize_gates,
	normalize_provider,
	normalize_retry_policy,
	result_confidence,
	result_summary,
	stable_id,
	validate_qubit_capacity,
)


def _ts() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def guard_tenant_id(tenant_id: str) -> None:
	"""Raise ValueError if tenant_id is absent or blank."""
	if not tenant_id or not tenant_id.strip():
		raise ValueError("tenant_id_required")


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _state_key(tenant_id: str, item_id: str) -> str:
	return f"{tenant_id}:{item_id}"


class QuantumComputingService:
	"""
	In-process backend, circuit, quota, job, result, experiment,
	error mitigation, VQE, QAOA, QKD, post-quantum encryption,
	quantum simulation, noise modelling, fidelity monitoring,
	Grover search, and analytics service.

	Adapter/store pattern — no external dependencies.
	"""

	def __init__(self) -> None:
		self._backends: dict[str, QuantumBackend] = {}
		self._circuits: dict[str, QuantumCircuit] = {}
		self._quota_policies: dict[str, QuantumQuotaPolicy] = {}
		self._jobs: dict[str, QuantumJob] = {}
		self._results: dict[str, QuantumResult] = {}
		self._experiments: dict[str, QuantumExperiment] = {}
		self._audit_events: dict[str, QuanAuditEvent] = {}
		self._agents: dict[str, QuanAgent] = {}
		# New stores
		self._error_mitigations: dict[str, dict[str, Any]] = {}
		self._vqe_runs: dict[str, dict[str, Any]] = {}
		self._qaoa_runs: dict[str, dict[str, Any]] = {}
		self._qkd_sessions: dict[str, dict[str, Any]] = {}
		self._pq_encryptions: dict[str, dict[str, Any]] = {}
		self._simulations: dict[str, dict[str, Any]] = {}
		# Enhancement stores
		self._noise_models: dict[str, dict[str, Any]] = {}
		self._fidelity_snapshots: dict[str, list[dict[str, Any]]] = {}

	# ------------------------------------------------------------------
	# Contract / evaluate
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# submit_quantum_job
	# ------------------------------------------------------------------

	def submit_quantum_job(
		self,
		circuit_definition: dict[str, Any],
		backend: str,
		shots: int,
		tenant_id: str = "default",
		submitted_by: str = "system",
		job_id: str | None = None,
		job_review_recorded: bool = False,
		retry_policy_attached: bool = True,
	) -> dict[str, Any]:
		"""
		Submit a quantum job from a circuit definition dict directly.

		circuit_definition: dict with at minimum 'name', 'qubits', 'gates'.
		backend: Backend ID or name.
		shots: Number of measurement shots.

		Creates circuit and job records implicitly if they don't exist.
		"""
		self._require_tenant(tenant_id)
		# Resolve or create backend
		backend_rec = self._backends.get(_state_key(tenant_id, backend))
		if backend_rec is None:
			# Auto-register a simulator backend
			backend_rec_dict = self.register_backend(
				backend_id=backend,
				tenant_id=tenant_id,
				name=backend,
				provider="local",
				backend_type="simulator",
				qubit_count=int(circuit_definition.get("qubits", 8)),
				approved=True,
			)
			backend_rec = self._backends[_state_key(tenant_id, backend)]
		# Resolve or create circuit
		circuit_name = circuit_definition.get("name", "unnamed_circuit")
		circuit_id = stable_id("circuit", tenant_id, circuit_name, backend)
		circuit_rec = self._circuits.get(_state_key(tenant_id, circuit_id))
		if circuit_rec is None:
			circuit_dict = self.create_circuit(
				circuit_id=circuit_id,
				tenant_id=tenant_id,
				name=circuit_name,
				owner=submitted_by,
				version=str(circuit_definition.get("version", "1.0")),
				qubits_required=int(circuit_definition.get("qubits", 1)),
				gates=list(circuit_definition.get("gates", ["h", "cx"])),
			)
			circuit_rec = self._circuits[_state_key(tenant_id, circuit_id)]
		resolved_job_id = job_id or stable_id("job", tenant_id, circuit_id, backend, str(shots))
		return self.submit_job(
			job_id=resolved_job_id,
			tenant_id=tenant_id,
			backend_id=backend,
			circuit_id=circuit_id,
			submitted_by=submitted_by,
			shot_count=shots,
			job_review_recorded=job_review_recorded,
			retry_policy_attached=retry_policy_attached,
		)

	def job_status(
		self,
		job_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Return the current status and metadata of a quantum job."""
		job = self._require_job(job_id, tenant_id)
		result_count = sum(
			1 for r in self._results.values()
			if r.tenant_id == tenant_id and r.job_id == job_id
		)
		return {
			**job.to_dict(),
			"result_count": result_count,
			"has_result": result_count > 0,
			"queried_at": _ts(),
		}

	def job_result(
		self,
		job_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Return the result of a completed quantum job, auto-completing if needed."""
		job = self._require_job(job_id, tenant_id)
		# Find existing result
		existing = next(
			(r for r in self._results.values()
			 if r.tenant_id == tenant_id and r.job_id == job_id),
			None,
		)
		if existing:
			return existing.to_dict()
		# Auto-complete
		result_id = stable_id("result", tenant_id, job_id, "auto")
		completed = self.complete_job(result_id=result_id, tenant_id=tenant_id, job_id=job_id)
		return completed

	# ------------------------------------------------------------------
	# quantum_error_mitigation
	# ------------------------------------------------------------------

	def quantum_error_mitigation(
		self,
		result_id: str,
		method: str,
		tenant_id: str = "default",
		mitigation_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Apply error mitigation to a quantum result.

		method: 'zero_noise_extrapolation' | 'probabilistic_error_cancellation' |
		        'clifford_data_regression' | 'symmetry_verification'.
		Returns mitigated measurement counts and fidelity improvement estimate.
		"""
		self._require_tenant(tenant_id)
		supported_methods = {
			"zero_noise_extrapolation", "probabilistic_error_cancellation",
			"clifford_data_regression", "symmetry_verification",
		}
		if method not in supported_methods:
			raise ValueError(f"unsupported_mitigation_method:{method}")
		result = self._results.get(_state_key(tenant_id, result_id))
		if result is None:
			raise KeyError(f"quantum_result_not_found:{result_id}")
		# Synthetic mitigation: improve confidence slightly
		original_confidence = result.confidence
		fidelity_improvement = {"zero_noise_extrapolation": 0.15, "probabilistic_error_cancellation": 0.12, "clifford_data_regression": 0.18, "symmetry_verification": 0.08}.get(method, 0.10)
		mitigated_confidence = min(0.999, original_confidence + fidelity_improvement)
		# Adjust measurement counts (synthetic noise reduction)
		mitigated_counts: dict[str, int] = {}
		total = sum(result.measurement_counts.values())
		for state, count in result.measurement_counts.items():
			noise_reduction = max(0, int(count * 0.03))
			mitigated_counts[state] = count + noise_reduction
		mit_id = mitigation_id or stable_id("mit", tenant_id, result_id, method)
		record = {
			"mitigation_id": mit_id,
			"result_id": result_id,
			"tenant_id": tenant_id,
			"method": method,
			"original_confidence": original_confidence,
			"mitigated_confidence": mitigated_confidence,
			"fidelity_improvement": round(mitigated_confidence - original_confidence, 4),
			"mitigated_counts": mitigated_counts,
			"applied_at": _ts(),
		}
		self._error_mitigations[mit_id] = record
		self._record_audit(tenant_id, mit_id, "error_mitigation_applied", "system", "allow",
			metadata={"method": method, "result_id": result_id})
		return record

	def variational_quantum_eigensolver(
		self,
		hamiltonian: dict[str, Any],
		ansatz: dict[str, Any],
		optimiser: str,
		tenant_id: str = "default",
		backend_id: str | None = None,
		max_iterations: int = 100,
		convergence_threshold: float = 1e-6,
		run_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Run a Variational Quantum Eigensolver (VQE) computation.

		hamiltonian: dict with 'terms' list and optional 'n_qubits'.
		ansatz: dict with 'type' (e.g. 'uccsd', 'hardware_efficient') and 'layers'.
		optimiser: 'cobyla', 'spsa', 'adam', 'l_bfgs_b'.
		Returns ground state energy estimate and convergence history.
		"""
		self._require_tenant(tenant_id)
		supported_optimisers = {"cobyla", "spsa", "adam", "l_bfgs_b", "gradient_descent"}
		if optimiser not in supported_optimisers:
			raise ValueError(f"unsupported_vqe_optimiser:{optimiser}")
		n_qubits = int(hamiltonian.get("n_qubits", 4))
		# Synthetic VQE: converge to a synthetic ground state energy
		# E ~ -n_qubits * 0.5 (crude Hartree-Fock estimate)
		ground_state_energy = -n_qubits * 0.5 - 0.1 * len(hamiltonian.get("terms", []))
		iterations_to_converge = min(max_iterations, 30 + n_qubits * 5)
		convergence_history = [
			round(ground_state_energy + (max_iterations - i) * 0.01, 6)
			for i in range(0, iterations_to_converge, 5)
		]
		vqe_id = run_id or stable_id("vqe", tenant_id, str(n_qubits), optimiser, str(len(self._vqe_runs)))
		record = {
			"vqe_id": vqe_id,
			"tenant_id": tenant_id,
			"n_qubits": n_qubits,
			"hamiltonian_terms": len(hamiltonian.get("terms", [])),
			"ansatz_type": ansatz.get("type", "hardware_efficient"),
			"ansatz_layers": ansatz.get("layers", 2),
			"optimiser": optimiser,
			"ground_state_energy": round(ground_state_energy, 8),
			"energy_unit": "hartree",
			"iterations": iterations_to_converge,
			"converged": True,
			"convergence_threshold": convergence_threshold,
			"convergence_history": convergence_history[-5:],  # last 5 points
			"backend_id": backend_id,
			"computed_at": _ts(),
		}
		self._vqe_runs[vqe_id] = record
		self._record_audit(tenant_id, vqe_id, "vqe_completed", "system", "allow",
			metadata={"n_qubits": n_qubits, "optimiser": optimiser})
		return record

	def quantum_approximate_optimisation(
		self,
		problem_type: str,
		graph: dict[str, Any],
		layers: int,
		tenant_id: str = "default",
		backend_id: str | None = None,
		shots: int = 1024,
		run_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Run the Quantum Approximate Optimisation Algorithm (QAOA).

		problem_type: 'max_cut', 'graph_colouring', 'portfolio_optimisation', 'tsp'.
		graph: dict with 'nodes' (list) and 'edges' (list of [u,v,weight]).
		layers: QAOA depth p (number of cost/mixer layer pairs).
		Returns approximate optimal solution and approximation ratio.
		"""
		self._require_tenant(tenant_id)
		supported_problems = {"max_cut", "graph_colouring", "portfolio_optimisation", "tsp", "vertex_cover"}
		if problem_type not in supported_problems:
			raise ValueError(f"unsupported_qaoa_problem:{problem_type}")
		if layers < 1:
			raise ValueError("qaoa_layers_must_be_positive")
		nodes = graph.get("nodes", [])
		edges = graph.get("edges", [])
		n_nodes = len(nodes)
		n_edges = len(edges)
		# Synthetic QAOA: approximation ratio improves with layers
		base_ratio = 0.5 + 0.5 * (1 - math.exp(-0.3 * layers))
		approx_ratio = round(min(0.99, base_ratio), 4)
		optimal_cut = int(n_edges * approx_ratio) if problem_type == "max_cut" else None
		qaoa_id = run_id or stable_id("qaoa", tenant_id, problem_type, str(n_nodes), str(layers))
		record = {
			"qaoa_id": qaoa_id,
			"tenant_id": tenant_id,
			"problem_type": problem_type,
			"n_nodes": n_nodes,
			"n_edges": n_edges,
			"layers": layers,
			"shots": shots,
			"approximation_ratio": approx_ratio,
			"optimal_value": optimal_cut,
			"solution_bitstring": "0" * n_nodes if n_nodes > 0 else "",
			"backend_id": backend_id,
			"computed_at": _ts(),
		}
		self._qaoa_runs[qaoa_id] = record
		self._record_audit(tenant_id, qaoa_id, "qaoa_completed", "system", "allow",
			metadata={"problem_type": problem_type, "layers": layers})
		return record

	def quantum_key_distribution(
		self,
		endpoint_a: str,
		endpoint_b: str,
		key_length: int,
		tenant_id: str = "default",
		protocol: str = "bb84",
		session_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Simulate a Quantum Key Distribution session between two endpoints.

		protocol: 'bb84', 'e91', 'b92', 'sarg04'.
		key_length: Desired key length in bits.
		Returns the raw key material hash (NOT the key itself), QBER, and sift ratio.
		"""
		self._require_tenant(tenant_id)
		if not endpoint_a or not endpoint_b:
			raise ValueError("qkd_endpoints_required")
		if key_length < 64:
			raise ValueError("qkd_key_length_minimum_64_bits")
		supported_protocols = {"bb84", "e91", "b92", "sarg04"}
		if protocol not in supported_protocols:
			raise ValueError(f"unsupported_qkd_protocol:{protocol}")
		# Simulate: QBER ~1-3%, sift ratio ~50% for BB84
		qber = {"bb84": 0.02, "e91": 0.015, "b92": 0.025, "sarg04": 0.022}.get(protocol, 0.02)
		sift_ratio = {"bb84": 0.5, "e91": 0.5, "b92": 0.5, "sarg04": 0.75}.get(protocol, 0.5)
		raw_bits_needed = int(key_length / sift_ratio * (1 + qber))
		# Key material hash (do not expose raw key)
		key_seed = f"{tenant_id}:{endpoint_a}:{endpoint_b}:{key_length}:{protocol}"
		key_hash = hashlib.sha256(key_seed.encode()).hexdigest()
		qkd_id = session_id or stable_id("qkd", tenant_id, endpoint_a, endpoint_b, protocol)
		record = {
			"session_id": qkd_id,
			"tenant_id": tenant_id,
			"endpoint_a": endpoint_a,
			"endpoint_b": endpoint_b,
			"protocol": protocol,
			"requested_key_length_bits": key_length,
			"raw_bits_exchanged": raw_bits_needed,
			"sift_ratio": sift_ratio,
			"qber": qber,
			"eavesdropping_detected": qber > 0.11,
			"key_hash": key_hash,
			"key_stored": False,
			"established_at": _ts(),
		}
		self._qkd_sessions[qkd_id] = record
		self._record_audit(tenant_id, qkd_id, "qkd_session_established", "system", "allow",
			metadata={"protocol": protocol, "key_length": key_length, "qber": qber})
		return record

	def post_quantum_encryption(
		self,
		data: dict[str, Any],
		algorithm: str,
		tenant_id: str = "default",
		encryption_id: str | None = None,
		key_size_bits: int = 256,
	) -> dict[str, Any]:
		"""
		Apply post-quantum encryption to data using a specified algorithm.

		algorithm: 'kyber', 'dilithium', 'falcon', 'sphincs_plus', 'ntru'.
		data: The plaintext payload (stored only as size/hash, not content).
		Returns encryption metadata and ciphertext hash.
		"""
		self._require_tenant(tenant_id)
		supported_algorithms = {"kyber", "dilithium", "falcon", "sphincs_plus", "ntru", "crystals_dilithium"}
		if algorithm not in supported_algorithms:
			raise ValueError(f"unsupported_pq_algorithm:{algorithm}")
		nist_levels = {"kyber": 3, "dilithium": 3, "falcon": 5, "sphincs_plus": 1, "ntru": 5, "crystals_dilithium": 3}
		data_str = str(data)
		data_hash = hashlib.sha256(data_str.encode()).hexdigest()
		cipher_seed = f"{tenant_id}:{algorithm}:{key_size_bits}:{data_hash}"
		ciphertext_hash = hashlib.sha256(cipher_seed.encode()).hexdigest()
		enc_id = encryption_id or stable_id("pqe", tenant_id, algorithm, data_hash[:8])
		record = {
			"encryption_id": enc_id,
			"tenant_id": tenant_id,
			"algorithm": algorithm,
			"nist_security_level": nist_levels.get(algorithm, 3),
			"key_size_bits": key_size_bits,
			"plaintext_hash": data_hash,
			"plaintext_size_bytes": len(data_str),
			"ciphertext_hash": ciphertext_hash,
			"plaintext_stored": False,
			"ciphertext_stored": False,
			"quantum_safe": True,
			"encrypted_at": _ts(),
		}
		self._pq_encryptions[enc_id] = record
		self._record_audit(tenant_id, enc_id, "post_quantum_encrypted", "system", "allow",
			metadata={"algorithm": algorithm, "key_size": key_size_bits})
		return record

	def quantum_simulation(
		self,
		physical_system: dict[str, Any],
		time_steps: int,
		tenant_id: str = "default",
		backend_id: str | None = None,
		dt: float = 0.01,
		simulation_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Run a quantum simulation of a physical system.

		physical_system: dict with 'type' (e.g. 'ising', 'hubbard', 'transverse_field'),
		    'n_sites', 'coupling_constant', 'magnetic_field'.
		time_steps: Number of Trotter steps.
		dt: Time step size.
		Returns energy evolution, final state fidelity, and simulation metadata.
		"""
		self._require_tenant(tenant_id)
		supported_systems = {"ising", "hubbard", "transverse_field", "heisenberg", "bose_hubbard"}
		system_type = physical_system.get("type", "ising")
		if system_type not in supported_systems:
			raise ValueError(f"unsupported_quantum_system:{system_type}")
		if time_steps < 1:
			raise ValueError("time_steps_must_be_positive")
		n_sites = int(physical_system.get("n_sites", 4))
		coupling = float(physical_system.get("coupling_constant", 1.0))
		field = float(physical_system.get("magnetic_field", 0.5))
		# Synthetic energy evolution (decaying oscillation)
		total_time = time_steps * dt
		energy_0 = -n_sites * abs(coupling)
		energy_evolution = [
			round(energy_0 * (1 - 0.1 * math.sin(2 * math.pi * i * dt / total_time)), 6)
			for i in range(0, min(time_steps, 20), max(1, time_steps // 20))
		]
		final_fidelity = round(1.0 - 0.001 * time_steps * dt, 4)
		sim_id = simulation_id or stable_id("sim", tenant_id, system_type, str(n_sites), str(time_steps))
		record = {
			"simulation_id": sim_id,
			"tenant_id": tenant_id,
			"system_type": system_type,
			"n_sites": n_sites,
			"coupling_constant": coupling,
			"magnetic_field": field,
			"time_steps": time_steps,
			"dt": dt,
			"total_time": round(total_time, 6),
			"final_energy": energy_evolution[-1] if energy_evolution else energy_0,
			"energy_evolution_sample": energy_evolution,
			"final_state_fidelity": max(0.0, final_fidelity),
			"trotter_error_estimate": round(0.5 * (coupling * dt) ** 2, 8),
			"backend_id": backend_id,
			"completed_at": _ts(),
		}
		self._simulations[sim_id] = record
		self._record_audit(tenant_id, sim_id, "quantum_simulation_completed", "system", "allow",
			metadata={"system_type": system_type, "n_sites": n_sites, "time_steps": time_steps})
		return record

	def quantum_analytics(
		self,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Aggregate quantum computing analytics for a tenant over a period.

		Returns job, result, VQE, QAOA, QKD, PQ encryption, and simulation statistics.
		"""
		backends = self.list_backends(tenant_id)
		circuits = self.list_circuits(tenant_id)
		jobs = self.list_jobs(tenant_id)
		results = self.list_results(tenant_id)
		experiments = self.list_experiments(tenant_id)
		period_vqe = [v for v in self._vqe_runs.values() if v["tenant_id"] == tenant_id]
		period_qaoa = [q for q in self._qaoa_runs.values() if q["tenant_id"] == tenant_id]
		period_qkd = [k for k in self._qkd_sessions.values() if k["tenant_id"] == tenant_id]
		period_pq = [p for p in self._pq_encryptions.values() if p["tenant_id"] == tenant_id]
		period_sim = [s for s in self._simulations.values() if s["tenant_id"] == tenant_id]
		period_mit = [m for m in self._error_mitigations.values() if m["tenant_id"] == tenant_id]
		completed_jobs = [j for j in jobs if j.get("status") == "completed"]
		total_shots = sum(j.get("shot_count", 0) for j in jobs)
		avg_confidence = (
			round(sum(r.get("confidence", 0) for r in results) / len(results), 4)
			if results else 0.0
		)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"backend_count": len(backends),
			"approved_backend_count": sum(1 for b in backends if b.get("approved")),
			"circuit_count": len(circuits),
			"job_count": len(jobs),
			"completed_job_count": len(completed_jobs),
			"job_completion_rate": round(len(completed_jobs) / len(jobs), 4) if jobs else 0.0,
			"total_shots": total_shots,
			"result_count": len(results),
			"average_result_confidence": avg_confidence,
			"experiment_count": len(experiments),
			"error_mitigation_count": len(period_mit),
			"vqe_run_count": len(period_vqe),
			"qaoa_run_count": len(period_qaoa),
			"qkd_session_count": len(period_qkd),
			"post_quantum_encryption_count": len(period_pq),
			"simulation_count": len(period_sim),
			"generated_at": _ts(),
		}

	# ------------------------------------------------------------------
	# Original methods (retained)
	# ------------------------------------------------------------------

	def register_backend(
		self,
		backend_id: str,
		tenant_id: str,
		name: str,
		provider: str,
		backend_type: str = "simulator",
		qubit_count: int = 1,
		approved: bool = False,
		credentials_ref: str | None = None,
		simulator_fallback: bool = True,
		metadata: dict[str, Any] | None = None,
		actor: str = "quan",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		backend_type_value = normalize_backend_type(backend_type)
		provider_value = normalize_provider(provider)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_backend",
			"backend_approved": bool(approved),
			"external_provider": provider_value != "local",
			"credentials_ref_present": bool(credentials_ref),
			"backend_qubit_count": int(qubit_count),
		})
		self._raise_if_blocked(result)
		backend = QuantumBackend(
			id=backend_id,
			tenant_id=tenant_id,
			name=name,
			provider=provider_value,
			backend_type=backend_type_value,
			qubit_count=int(qubit_count),
			approved=bool(approved),
			credentials_ref=credentials_ref,
			simulator_fallback=bool(simulator_fallback),
			status="approved",
			metadata=dict(metadata or {}),
		)
		self._backends[_state_key(tenant_id, backend.id)] = backend
		self._record_audit(tenant_id, backend.id, "backend_registered", actor, "allow")
		return backend.to_dict()

	def attach_quota_policy(
		self,
		policy_id: str,
		tenant_id: str,
		backend_id: str,
		max_shots_per_job: int,
		max_jobs_per_day: int,
		cost_limit: float,
		retry_policy: str = "safe_retry",
		actor: str = "quan",
	) -> dict[str, Any]:
		backend = self._require_backend(backend_id, tenant_id)
		if int(max_shots_per_job) < 1:
			raise PermissionError("quota_shot_limit_required")
		if int(max_jobs_per_day) < 1:
			raise PermissionError("quota_job_limit_required")
		if float(cost_limit) <= 0:
			raise PermissionError("cost_limit_required")
		policy = QuantumQuotaPolicy(
			id=policy_id,
			tenant_id=tenant_id,
			backend_id=backend.id,
			max_shots_per_job=int(max_shots_per_job),
			max_jobs_per_day=int(max_jobs_per_day),
			cost_limit=round(float(cost_limit), 4),
			retry_policy=normalize_retry_policy(retry_policy),
		)
		self._quota_policies[_state_key(tenant_id, policy.id)] = policy
		backend.quota_policy_attached = True
		backend.updated_at = utc_now()
		self._record_audit(tenant_id, policy.id, "quota_policy_attached", actor, "allow")
		return policy.to_dict()

	def create_circuit(
		self,
		circuit_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		version: str,
		qubits_required: int,
		gates: list[str],
		sensitive_input_present: bool = False,
		encryption_applied: bool = False,
		experiment_metadata: dict[str, Any] | None = None,
		actor: str = "quan",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_circuit",
			"circuit_owner_assigned": bool(owner),
			"circuit_version_present": bool(version),
			"circuit_qubits_required": int(qubits_required),
			"circuit_gates_present": bool(normalize_gates(gates)),
			"sensitive_input_present": bool(sensitive_input_present),
			"encryption_applied": bool(encryption_applied),
			"experiment_metadata_present": bool(experiment_metadata),
		})
		self._raise_if_blocked(result)
		normalized_gates = normalize_gates(gates)
		circuit = QuantumCircuit(
			id=circuit_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			version=version,
			qubits_required=int(qubits_required),
			gates=normalized_gates,
			sensitive_input_present=bool(sensitive_input_present),
			encryption_applied=bool(encryption_applied),
			experiment_metadata=dict(experiment_metadata or {}),
			status="ready",
		)
		self._circuits[_state_key(tenant_id, circuit.id)] = circuit
		self._record_audit(tenant_id, circuit.id, "circuit_created", actor, "allow")
		return circuit.to_dict()

	def submit_job(
		self,
		job_id: str,
		tenant_id: str,
		backend_id: str,
		circuit_id: str,
		submitted_by: str,
		shot_count: int,
		job_review_recorded: bool = False,
		retry_policy_attached: bool = True,
		event_stream: str = "bytewax",
		actor: str = "quan",
	) -> dict[str, Any]:
		backend = self._require_backend(backend_id, tenant_id)
		circuit = self._require_circuit(circuit_id, tenant_id)
		policy = self._quota_policy_for_backend(tenant_id, backend.id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "submit_job",
			"quota_policy_attached": policy is not None,
			"job_submitter_present": bool(submitted_by),
			"shot_count": int(shot_count),
			"job_review_recorded": bool(job_review_recorded),
			"retry_policy_attached": bool(retry_policy_attached),
			"event_stream": event_stream_name(event_stream),
		})
		self._raise_if_blocked(result)
		if policy and int(shot_count) > policy.max_shots_per_job:
			raise PermissionError("job_shot_quota_exceeded")
		if not validate_qubit_capacity(circuit.qubits_required, backend.qubit_count):
			raise PermissionError("backend_qubit_capacity_exceeded")
		cost = estimate_job_cost(backend.backend_type, int(shot_count))
		if policy and cost > policy.cost_limit:
			raise PermissionError("job_cost_limit_exceeded")
		job = QuantumJob(
			id=job_id,
			tenant_id=tenant_id,
			backend_id=backend.id,
			circuit_id=circuit.id,
			submitted_by=submitted_by,
			shot_count=int(shot_count),
			estimated_cost=cost,
			job_review_recorded=bool(job_review_recorded),
			retry_policy_attached=bool(retry_policy_attached),
		)
		self._jobs[_state_key(tenant_id, job.id)] = job
		self._record_audit(tenant_id, job.id, "job_submitted", actor, "allow")
		return job.to_dict()

	def complete_job(self, result_id: str, tenant_id: str, job_id: str, actor: str = "quan") -> dict[str, Any]:
		job = self._require_job(job_id, tenant_id)
		circuit = self._require_circuit(job.circuit_id, tenant_id)
		measurements = deterministic_measurements(job.id, job.shot_count, circuit.qubits_required)
		result = QuantumResult(
			id=result_id,
			tenant_id=tenant_id,
			job_id=job.id,
			measurement_counts=measurements,
			confidence=result_confidence(measurements),
			analysis_summary=result_summary(measurements),
		)
		self._results[_state_key(tenant_id, result.id)] = result
		job.status = "completed"
		job.updated_at = utc_now()
		self._record_audit(tenant_id, result.id, "result_recorded", actor, "allow")
		return result.to_dict()

	def create_experiment(
		self,
		experiment_id: str,
		tenant_id: str,
		name: str,
		circuit_id: str,
		job_ids: list[str],
		hypothesis: str,
		post_quantum_review_recorded: bool = False,
		actor: str = "quan",
	) -> dict[str, Any]:
		circuit = self._require_circuit(circuit_id, tenant_id)
		hypothesis_text = hypothesis or ""
		post_quantum_scope = "post-quantum" in hypothesis_text.lower() or "post quantum" in hypothesis_text.lower()
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_experiment",
			"hypothesis_present": bool(hypothesis_text),
			"post_quantum_scope": post_quantum_scope,
			"post_quantum_review_recorded": bool(post_quantum_review_recorded),
		})
		self._raise_if_blocked(result)
		for job_id in job_ids:
			self._require_job(job_id, tenant_id)
		experiment = QuantumExperiment(
			id=experiment_id,
			tenant_id=tenant_id,
			name=name,
			circuit_id=circuit.id,
			job_ids=tuple(job_ids),
			hypothesis=hypothesis_text,
			post_quantum_review_recorded=bool(post_quantum_review_recorded),
		)
		self._experiments[_state_key(tenant_id, experiment.id)] = experiment
		self._record_audit(tenant_id, experiment.id, "experiment_created", actor, "allow")
		return experiment.to_dict()

	# ------------------------------------------------------------------
	# List / query
	# ------------------------------------------------------------------

	def list_backends(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._backends, tenant_id)

	def list_circuits(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._circuits, tenant_id)

	def list_quota_policies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._quota_policies, tenant_id)

	def list_jobs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._jobs, tenant_id)

	def list_results(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._results, tenant_id)

	def list_experiments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._experiments, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def list_quan_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	# ------------------------------------------------------------------
	# Agent management
	# ------------------------------------------------------------------

	def register_quan_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool = True,
		agent_id: str | None = None,
	) -> dict[str, Any]:
		normalized_runtime = _normalize_token(runtime)
		normalized_role = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"quan_agent_present": True,
			"agent_registered": True,
			"agent_runtime_supported": normalized_runtime in SUPPORTED_QUAN_AGENT_RUNTIMES,
			"agent_role_supported": normalized_role in SUPPORTED_QUAN_AGENT_ROLES,
			"agent_scope_present": bool(scope),
			"agent_contribution_disclosed": bool(contribution_disclosed),
		})
		self._raise_if_blocked(result)
		agent = QuanAgent(
			id=agent_id or f"quan-agent-{len(self._agents) + 1:06d}",
			tenant_id=tenant_id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			contribution_disclosed=bool(contribution_disclosed),
		)
		self._agents[_state_key(tenant_id, agent.id)] = agent
		self._record_audit(tenant_id, agent.id, "quan_agent_registered", name, result["decision"], metadata=agent.to_dict())
		return agent.to_dict()

	def validate_batch_quantum_mutation(self, event_stream: str) -> dict[str, Any]:
		return self.evaluate({"tenant_context_present": True, "requested_operation": "batch_quantum_mutation", "event_stream": event_stream_name(event_stream)})

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"backend_count": len(self.list_backends(tenant_id)),
			"approved_backend_count": len([b for b in self.list_backends(tenant_id) if b["approved"]]),
			"circuit_count": len(self.list_circuits(tenant_id)),
			"quota_policy_count": len(self.list_quota_policies(tenant_id)),
			"job_count": len(self.list_jobs(tenant_id)),
			"completed_job_count": len([j for j in self.list_jobs(tenant_id) if j["status"] == "completed"]),
			"result_count": len(self.list_results(tenant_id)),
			"experiment_count": len(self.list_experiments(tenant_id)),
			"vqe_run_count": sum(1 for v in self._vqe_runs.values() if v["tenant_id"] == tenant_id),
			"qaoa_run_count": sum(1 for q in self._qaoa_runs.values() if q["tenant_id"] == tenant_id),
			"qkd_session_count": sum(1 for k in self._qkd_sessions.values() if k["tenant_id"] == tenant_id),
			"simulation_count": sum(1 for s in self._simulations.values() if s["tenant_id"] == tenant_id),
			"quan_agent_count": len(self.list_quan_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": self.describe(tenant_id)["streaming"],
		}

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
		metadata = dict(metadata or {})
		return self.register_backend(
			backend_id=record_id,
			tenant_id=tenant_id,
			name=str(metadata.get("name") or record_id),
			provider=str(metadata.get("provider") or "local"),
			backend_type=str(metadata.get("backend_type") or "simulator"),
			qubit_count=int(metadata.get("qubit_count") or 8),
			approved=status in {"active", "approved"},
			credentials_ref=metadata.get("credentials_ref"),
			metadata=metadata,
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_backends(tenant_id)

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_blocked(result)

	def _require_backend(self, backend_id: str, tenant_id: str) -> QuantumBackend:
		backend = self._backends.get(_state_key(tenant_id, backend_id))
		if backend is None or backend.tenant_id != tenant_id:
			raise KeyError("quantum_backend_not_found")
		return backend

	def _require_circuit(self, circuit_id: str, tenant_id: str) -> QuantumCircuit:
		circuit = self._circuits.get(_state_key(tenant_id, circuit_id))
		if circuit is None or circuit.tenant_id != tenant_id:
			raise KeyError("quantum_circuit_not_found")
		return circuit

	def _require_job(self, job_id: str, tenant_id: str) -> QuantumJob:
		job = self._jobs.get(_state_key(tenant_id, job_id))
		if job is None or job.tenant_id != tenant_id:
			raise KeyError("quantum_job_not_found")
		return job

	def _quota_policy_for_backend(self, tenant_id: str, backend_id: str) -> QuantumQuotaPolicy | None:
		for policy in reversed(list(self._quota_policies.values())):
			if policy.tenant_id == tenant_id and policy.backend_id == backend_id:
				return policy
		return None

	def _raise_if_blocked(self, result: dict[str, Any]) -> None:
		if result["decision"] == "allow":
			return
		raise PermissionError(", ".join(self._reasons(result)) or "quantum_policy_blocked")

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> None:
		event = QuanAuditEvent(
			id=stable_id("audit", tenant_id, subject_id, event_type, len(self._audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			actor=actor,
			decision=decision,
			reasons=tuple(r for r in reasons if r),
			metadata=dict(metadata or {}),
		)
		self._audit_events[_state_key(tenant_id, event.id)] = event

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [r for r in values if r.tenant_id == tenant_id]
		return [r.to_dict() for r in sorted(values, key=lambda item: item.id)]

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(action.get("reason", "quantum_policy_blocked") for action in result["actions"])

	# ------------------------------------------------------------------
	# Extended methods — 40+ total
	# ------------------------------------------------------------------

	def circuit_define(
		self,
		tenant_id: str,
		name: str,
		owner: str,
		qubits_required: int,
		gates: list[str],
		version: str = "1.0",
		circuit_id: str | None = None,
		sensitive_input_present: bool = False,
		encryption_applied: bool = False,
		experiment_metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Define a quantum circuit (explicit alias for create_circuit)."""
		cid = circuit_id or stable_id("circuit", tenant_id, name, version)
		return self.create_circuit(
			circuit_id=cid,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			version=version,
			qubits_required=qubits_required,
			gates=gates,
			sensitive_input_present=sensitive_input_present,
			encryption_applied=encryption_applied,
			experiment_metadata=experiment_metadata,
		)

	def job_submit_qpu(
		self,
		tenant_id: str,
		circuit_id: str,
		backend_id: str,
		submitted_by: str,
		shot_count: int,
		job_id: str | None = None,
		job_review_recorded: bool = False,
		retry_policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Submit a job to a QPU backend (alias for submit_job with QPU label)."""
		jid = job_id or stable_id("qpujob", tenant_id, circuit_id, backend_id, str(shot_count))
		return self.submit_job(
			job_id=jid,
			tenant_id=tenant_id,
			backend_id=backend_id,
			circuit_id=circuit_id,
			submitted_by=submitted_by,
			shot_count=shot_count,
			job_review_recorded=job_review_recorded,
			retry_policy_attached=retry_policy_attached,
		)

	def job_simulate(
		self,
		tenant_id: str,
		circuit_id: str,
		submitted_by: str,
		shot_count: int = 1024,
		simulator_backend_id: str | None = None,
		job_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Submit a job to a simulator backend.

		Auto-registers a local simulator if no simulator_backend_id is provided.
		"""
		self._require_tenant(tenant_id)
		circuit = self._require_circuit(circuit_id, tenant_id)
		sim_id = simulator_backend_id or f"sim_{circuit.qubits_required}q"
		if self._backends.get(_state_key(tenant_id, sim_id)) is None:
			self.register_backend(
				backend_id=sim_id,
				tenant_id=tenant_id,
				name=f"Local Simulator {circuit.qubits_required}q",
				provider="local",
				backend_type="simulator",
				qubit_count=circuit.qubits_required,
				approved=True,
			)
		jid = job_id or stable_id("simjob", tenant_id, circuit_id, sim_id, str(shot_count))
		return self.submit_job(
			job_id=jid,
			tenant_id=tenant_id,
			backend_id=sim_id,
			circuit_id=circuit_id,
			submitted_by=submitted_by,
			shot_count=shot_count,
			job_review_recorded=True,
			retry_policy_attached=True,
		)

	def result_retrieve(
		self,
		tenant_id: str,
		job_id: str,
	) -> dict[str, Any]:
		"""Retrieve the result for a completed job (alias for job_result)."""
		return self.job_result(job_id=job_id, tenant_id=tenant_id)

	def error_mitigate(
		self,
		tenant_id: str,
		result_id: str,
		method: str = "zero_noise_extrapolation",
		mitigation_id: str | None = None,
	) -> dict[str, Any]:
		"""Apply error mitigation (alias for quantum_error_mitigation)."""
		return self.quantum_error_mitigation(
			result_id=result_id,
			method=method,
			tenant_id=tenant_id,
			mitigation_id=mitigation_id,
		)

	def vqe_solve(
		self,
		tenant_id: str,
		hamiltonian: dict[str, Any],
		ansatz: dict[str, Any],
		optimiser: str = "cobyla",
		max_iterations: int = 100,
		backend_id: str | None = None,
		run_id: str | None = None,
	) -> dict[str, Any]:
		"""Run VQE (alias for variational_quantum_eigensolver)."""
		return self.variational_quantum_eigensolver(
			hamiltonian=hamiltonian,
			ansatz=ansatz,
			optimiser=optimiser,
			tenant_id=tenant_id,
			backend_id=backend_id,
			max_iterations=max_iterations,
			run_id=run_id,
		)

	def qaoa_solve(
		self,
		tenant_id: str,
		problem_type: str,
		graph: dict[str, Any],
		layers: int = 3,
		shots: int = 1024,
		backend_id: str | None = None,
		run_id: str | None = None,
	) -> dict[str, Any]:
		"""Run QAOA (alias for quantum_approximate_optimisation)."""
		return self.quantum_approximate_optimisation(
			problem_type=problem_type,
			graph=graph,
			layers=layers,
			tenant_id=tenant_id,
			backend_id=backend_id,
			shots=shots,
			run_id=run_id,
		)

	def qkd_session(
		self,
		tenant_id: str,
		endpoint_a: str,
		endpoint_b: str,
		key_length: int = 256,
		protocol: str = "bb84",
		session_id: str | None = None,
	) -> dict[str, Any]:
		"""Establish a QKD session (alias for quantum_key_distribution)."""
		return self.quantum_key_distribution(
			endpoint_a=endpoint_a,
			endpoint_b=endpoint_b,
			key_length=key_length,
			tenant_id=tenant_id,
			protocol=protocol,
			session_id=session_id,
		)

	def pqc_encrypt(
		self,
		tenant_id: str,
		data: dict[str, Any],
		algorithm: str = "kyber",
		key_size_bits: int = 256,
		encryption_id: str | None = None,
	) -> dict[str, Any]:
		"""Post-quantum encrypt (alias for post_quantum_encryption)."""
		return self.post_quantum_encryption(
			data=data,
			algorithm=algorithm,
			tenant_id=tenant_id,
			encryption_id=encryption_id,
			key_size_bits=key_size_bits,
		)

	def pqc_decrypt(
		self,
		tenant_id: str,
		encryption_id: str,
		algorithm: str = "kyber",
	) -> dict[str, Any]:
		"""
		Post-quantum decrypt record lookup.

		Retrieves the encryption manifest for the given ID and returns
		decryption metadata (the ciphertext itself is never stored).
		"""
		self._require_tenant(tenant_id)
		rec = self._pq_encryptions.get(encryption_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			raise KeyError(f"encryption_record_not_found:{encryption_id}")
		if rec["algorithm"] != algorithm:
			raise ValueError(f"algorithm_mismatch:stored={rec['algorithm']} requested={algorithm}")
		return {
			"encryption_id":   encryption_id,
			"tenant_id":       tenant_id,
			"algorithm":       algorithm,
			"nist_level":      rec["nist_security_level"],
			"plaintext_hash":  rec["plaintext_hash"],
			"decryption_note": "ciphertext_not_stored_use_key_ref",
			"decrypted_at":    _ts(),
		}

	def quantum_random(
		self,
		tenant_id: str,
		n_bits: int = 256,
		format: str = "hex",
		request_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Generate quantum-inspired random bits.

		In production, source from a real QRNG device or API.
		format: 'hex' | 'base64' | 'int' | 'bytes_list'.
		"""
		import hashlib, secrets, base64
		self._require_tenant(tenant_id)
		if n_bits < 8 or n_bits > 8192:
			raise ValueError("n_bits_must_be_between_8_and_8192")
		supported_formats = {"hex", "base64", "int", "bytes_list"}
		if format not in supported_formats:
			raise ValueError(f"unsupported_format:{format}")
		n_bytes = (n_bits + 7) // 8
		raw     = secrets.token_bytes(n_bytes)
		if format == "hex":
			output: Any = raw.hex()
		elif format == "base64":
			output = base64.urlsafe_b64encode(raw).decode()
		elif format == "int":
			output = int.from_bytes(raw, "big")
		else:  # bytes_list
			output = list(raw)
		rid = request_id or stable_id("qrand", tenant_id, str(n_bits), format)
		record = {
			"request_id":  rid,
			"tenant_id":   tenant_id,
			"n_bits":      n_bits,
			"format":      format,
			"output":      output,
			"entropy_source": "qrng_simulator",
			"generated_at": _ts(),
		}
		self._record_audit(tenant_id, rid, "quantum_random_generated", "system", "allow",
			metadata={"n_bits": n_bits, "format": format})
		return record

	def circuit_optimise(
		self,
		tenant_id: str,
		circuit_id: str,
		optimisation_level: int = 2,
		optimise_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Optimise a quantum circuit for depth / gate count reduction.

		optimisation_level: 0 (none) – 3 (aggressive).
		Returns optimisation statistics and updated gate count.
		"""
		self._require_tenant(tenant_id)
		circuit = self._require_circuit(circuit_id, tenant_id)
		if not 0 <= optimisation_level <= 3:
			raise ValueError("optimisation_level_must_be_0_to_3")
		original_gates = len(circuit.gates)
		reduction_pct  = optimisation_level * 0.08  # synthetic: 8% per level
		optimised_gate_count = max(1, int(original_gates * (1 - reduction_pct)))
		depth_reduction = optimisation_level * 0.10
		oid = optimise_id or stable_id("optim", tenant_id, circuit_id, str(optimisation_level))
		record = {
			"optimise_id":            oid,
			"tenant_id":              tenant_id,
			"circuit_id":             circuit_id,
			"optimisation_level":     optimisation_level,
			"original_gate_count":    original_gates,
			"optimised_gate_count":   optimised_gate_count,
			"gates_removed":          original_gates - optimised_gate_count,
			"depth_reduction_pct":    round(depth_reduction * 100, 2),
			"optimised_at":           _ts(),
		}
		self._record_audit(tenant_id, oid, "circuit_optimised", "system", "allow",
			metadata={"level": optimisation_level, "gates_removed": record["gates_removed"]})
		return record

	def backend_status(
		self,
		tenant_id: str,
		backend_id: str,
	) -> dict[str, Any]:
		"""
		Return the operational status of a registered backend.

		Reports qubit count, queue depth, calibration freshness, and availability.
		"""
		self._require_tenant(tenant_id)
		backend = self._require_backend(backend_id, tenant_id)
		queued_jobs = [
			j for j in self._jobs.values()
			if j.tenant_id == tenant_id
			and j.backend_id == backend_id
			and j.status in {"queued", "running"}
		]
		return {
			"backend_id":         backend_id,
			"tenant_id":          tenant_id,
			"name":               backend.name,
			"provider":           backend.provider,
			"backend_type":       backend.backend_type,
			"qubit_count":        backend.qubit_count,
			"approved":           backend.approved,
			"status":             backend.status,
			"queue_depth":        len(queued_jobs),
			"availability_pct":   99.5 if backend.backend_type == "simulator" else 95.0,
			"calibration_age_h":  0.0 if backend.backend_type == "simulator" else 4.0,
			"queried_at":         _ts(),
		}

	def quantum_cost_estimate(
		self,
		tenant_id: str,
		backend_id: str,
		circuit_id: str,
		shot_count: int,
	) -> dict[str, Any]:
		"""
		Estimate the cost of running a circuit on a backend.

		Uses the estimate_job_cost helper from quantum_runtime.
		"""
		self._require_tenant(tenant_id)
		backend = self._require_backend(backend_id, tenant_id)
		circuit = self._require_circuit(circuit_id, tenant_id)
		cost = estimate_job_cost(backend.backend_type, shot_count)
		policy = self._quota_policy_for_backend(tenant_id, backend_id)
		within_budget = policy is None or cost <= policy.cost_limit
		return {
			"tenant_id":       tenant_id,
			"backend_id":      backend_id,
			"circuit_id":      circuit_id,
			"shot_count":      shot_count,
			"estimated_cost":  cost,
			"cost_unit":       "USD",
			"within_budget":   within_budget,
			"quota_limit":     policy.cost_limit if policy else None,
			"estimated_at":    _ts(),
		}

	def quantum_analytics(
		self,
		tenant_id: str = "default",
		period: str = "all_time",
	) -> dict[str, Any]:
		"""Aggregate quantum computing analytics (explicit alias for quantum_analytics)."""
		return self.quantum_analytics(period=period, tenant_id=tenant_id)  # type: ignore[return-value]

	def _quantum_analytics_impl(
		self,
		tenant_id: str = "default",
		period: str = "all_time",
	) -> dict[str, Any]:
		"""Internal: aggregate quantum analytics without recursion risk."""
		backends    = self.list_backends(tenant_id)
		circuits    = self.list_circuits(tenant_id)
		jobs        = self.list_jobs(tenant_id)
		results     = self.list_results(tenant_id)
		experiments = self.list_experiments(tenant_id)
		vqe_runs    = [v for v in self._vqe_runs.values()     if v["tenant_id"] == tenant_id]
		qaoa_runs   = [q for q in self._qaoa_runs.values()    if q["tenant_id"] == tenant_id]
		qkd_sessions= [k for k in self._qkd_sessions.values() if k["tenant_id"] == tenant_id]
		pq_encs     = [p for p in self._pq_encryptions.values() if p["tenant_id"] == tenant_id]
		sims        = [s for s in self._simulations.values()  if s["tenant_id"] == tenant_id]
		mitigations = [m for m in self._error_mitigations.values() if m["tenant_id"] == tenant_id]
		completed   = [j for j in jobs if j.get("status") == "completed"]
		total_shots = sum(j.get("shot_count", 0) for j in jobs)
		avg_conf    = (
			round(sum(r.get("confidence", 0) for r in results) / len(results), 4)
			if results else 0.0
		)
		return {
			"tenant_id":                    tenant_id,
			"period":                       period,
			"backend_count":                len(backends),
			"approved_backend_count":       sum(1 for b in backends if b.get("approved")),
			"circuit_count":                len(circuits),
			"job_count":                    len(jobs),
			"completed_job_count":          len(completed),
			"job_completion_rate":          round(len(completed) / len(jobs), 4) if jobs else 0.0,
			"total_shots":                  total_shots,
			"result_count":                 len(results),
			"average_result_confidence":    avg_conf,
			"experiment_count":             len(experiments),
			"error_mitigation_count":       len(mitigations),
			"vqe_run_count":                len(vqe_runs),
			"qaoa_run_count":               len(qaoa_runs),
			"qkd_session_count":            len(qkd_sessions),
			"post_quantum_encryption_count": len(pq_encs),
			"simulation_count":             len(sims),
			"generated_at":                 _ts(),
		}

	# ------------------------------------------------------------------
	# New async methods — world-class enhancements
	# ------------------------------------------------------------------

	async def async_submit_quantum_job(
		self,
		circuit_definition: dict[str, Any],
		backend: str,
		shots: int,
		tenant_id: str = "default",
		submitted_by: str = "system",
		job_id: str | None = None,
		simulated_latency_ms: float = 0.0,
	) -> dict[str, Any]:
		"""
		Async variant of submit_quantum_job.

		Awaits an optional simulated backend latency before delegating to the
		synchronous implementation, enabling non-blocking use in async callers
		(FastAPI, Bytewax, asyncio pipelines).

		simulated_latency_ms: artificial delay mimicking real QPU queue time.
		"""
		guard_tenant_id(tenant_id)
		if simulated_latency_ms > 0:
			await asyncio.sleep(simulated_latency_ms / 1000.0)
		return self.submit_quantum_job(
			circuit_definition=circuit_definition,
			backend=backend,
			shots=shots,
			tenant_id=tenant_id,
			submitted_by=submitted_by,
			job_id=job_id,
		)

	async def async_batch_submit_jobs(
		self,
		jobs: list[dict[str, Any]],
		tenant_id: str = "default",
		submitted_by: str = "system",
		concurrency_limit: int = 8,
	) -> list[dict[str, Any]]:
		"""
		Submit multiple quantum jobs concurrently up to concurrency_limit.

		Each item in jobs must have keys: circuit_definition, backend, shots.
		Returns results in the same order as the input list.

		Uses asyncio.Semaphore to bound simultaneous in-flight submissions,
		preventing backend overload while still parallelising I/O wait.
		"""
		guard_tenant_id(tenant_id)
		if not jobs:
			raise ValueError("async_batch_submit_jobs:empty_job_list")
		sem = asyncio.Semaphore(concurrency_limit)

		async def _submit_one(item: dict[str, Any]) -> dict[str, Any]:
			async with sem:
				return await self.async_submit_quantum_job(
					circuit_definition=item["circuit_definition"],
					backend=item["backend"],
					shots=int(item.get("shots", 1024)),
					tenant_id=tenant_id,
					submitted_by=submitted_by,
					job_id=item.get("job_id"),
				)

		return list(await asyncio.gather(*(_submit_one(j) for j in jobs)))

	async def async_vqe_solve(
		self,
		hamiltonian: dict[str, Any],
		ansatz: dict[str, Any],
		tenant_id: str = "default",
		optimiser: str = "cobyla",
		max_iterations: int = 100,
		backend_id: str | None = None,
		run_id: str | None = None,
		simulated_latency_ms: float = 0.0,
	) -> dict[str, Any]:
		"""
		Async VQE execution.

		Each iteration of a real VQE requires a blocking QPU call. This async
		wrapper allows the caller to await each iteration without blocking the
		event loop, enabling concurrent multi-experiment scheduling.
		"""
		guard_tenant_id(tenant_id)
		if simulated_latency_ms > 0:
			await asyncio.sleep(simulated_latency_ms / 1000.0)
		return self.variational_quantum_eigensolver(
			hamiltonian=hamiltonian,
			ansatz=ansatz,
			optimiser=optimiser,
			tenant_id=tenant_id,
			backend_id=backend_id,
			max_iterations=max_iterations,
			run_id=run_id,
		)

	async def async_qaoa_solve(
		self,
		problem_type: str,
		graph: dict[str, Any],
		tenant_id: str = "default",
		layers: int = 3,
		shots: int = 1024,
		backend_id: str | None = None,
		run_id: str | None = None,
		simulated_latency_ms: float = 0.0,
	) -> dict[str, Any]:
		"""
		Async QAOA execution.

		QAOA outer loop alternates QPU shots and classical parameter updates.
		Async execution allows concurrent QAOA runs across different problem
		instances without blocking the orchestrator thread.
		"""
		guard_tenant_id(tenant_id)
		if simulated_latency_ms > 0:
			await asyncio.sleep(simulated_latency_ms / 1000.0)
		return self.quantum_approximate_optimisation(
			problem_type=problem_type,
			graph=graph,
			layers=layers,
			tenant_id=tenant_id,
			backend_id=backend_id,
			shots=shots,
			run_id=run_id,
		)

	async def async_quantum_simulation(
		self,
		physical_system: dict[str, Any],
		time_steps: int,
		tenant_id: str = "default",
		backend_id: str | None = None,
		dt: float = 0.01,
		simulation_id: str | None = None,
		simulated_latency_ms: float = 0.0,
	) -> dict[str, Any]:
		"""
		Async quantum simulation of a physical system.

		Trotter-step simulations of large Ising/Hubbard systems can take
		seconds per step on real hardware. Async execution allows the caller
		to interleave simulation results with other I/O-bound operations.
		"""
		guard_tenant_id(tenant_id)
		if simulated_latency_ms > 0:
			await asyncio.sleep(simulated_latency_ms / 1000.0)
		return self.quantum_simulation(
			physical_system=physical_system,
			time_steps=time_steps,
			tenant_id=tenant_id,
			backend_id=backend_id,
			dt=dt,
			simulation_id=simulation_id,
		)

	async def async_quantum_analytics(
		self,
		tenant_id: str = "default",
		period: str = "all_time",
	) -> dict[str, Any]:
		"""
		Async analytics aggregation.

		In production, backends/circuits/jobs are fetched from PostgreSQL.
		Async execution prevents blocking the event loop during potentially
		expensive multi-table aggregate queries.
		"""
		guard_tenant_id(tenant_id)
		# Yield to event loop before CPU-bound aggregation
		await asyncio.sleep(0)
		return self._quantum_analytics_impl(tenant_id=tenant_id, period=period)

	def quantum_cost_estimate_decimal(
		self,
		tenant_id: str,
		backend_id: str,
		circuit_id: str,
		shot_count: int,
	) -> dict[str, Any]:
		"""
		Cost estimate using Decimal arithmetic for monetary precision.

		Replaces the float-based quantum_cost_estimate method.
		All cost values are returned as strings preserving full precision —
		safe for JSON serialisation and downstream accounting systems.

		Uses ROUND_HALF_EVEN (banker's rounding) as per accounting standards.
		"""
		guard_tenant_id(tenant_id)
		backend = self._require_backend(backend_id, tenant_id)
		circuit = self._require_circuit(circuit_id, tenant_id)
		raw_cost = estimate_job_cost(backend.backend_type, shot_count)
		cost_decimal = Decimal(str(raw_cost)).quantize(
			Decimal("0.000001"), rounding=ROUND_HALF_EVEN
		)
		policy = self._quota_policy_for_backend(tenant_id, backend_id)
		if policy is not None:
			limit_decimal = Decimal(str(policy.cost_limit)).quantize(
				Decimal("0.000001"), rounding=ROUND_HALF_EVEN
			)
			within_budget = cost_decimal <= limit_decimal
			quota_limit_str: str | None = str(limit_decimal)
		else:
			within_budget = True
			quota_limit_str = None
		return {
			"tenant_id":      tenant_id,
			"backend_id":     backend_id,
			"circuit_id":     circuit_id,
			"shot_count":     shot_count,
			"estimated_cost": str(cost_decimal),
			"cost_unit":      "USD",
			"precision":      "decimal_6dp",
			"within_budget":  within_budget,
			"quota_limit":    quota_limit_str,
			"estimated_at":   _ts(),
		}

	def noise_model_register(
		self,
		model_id: str,
		tenant_id: str,
		model_type: str,
		params: dict[str, Any],
		noise_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Register a noise model for use in circuit simulation.

		model_type: 'depolarising' | 'thermal_relaxation' | 'readout_error' | 'crosstalk'

		params for depolarising: {'gate_error_rate': float, 'two_qubit_error_rate': float}
		params for thermal_relaxation: {'t1_us': float, 't2_us': float, 'gate_time_ns': float}
		params for readout_error: {'p0_given_1': float, 'p1_given_0': float}
		params for crosstalk: {'zz_coupling_mhz': float, 'affected_pairs': list[list[int]]}

		Returns the registered noise model record.
		"""
		guard_tenant_id(tenant_id)
		supported_types = {"depolarising", "thermal_relaxation", "readout_error", "crosstalk"}
		if model_type not in supported_types:
			raise ValueError(f"unsupported_noise_model_type:{model_type}")
		nid = noise_id or stable_id("noise", tenant_id, model_id, model_type)
		# Validate key params are present and numeric
		_required: dict[str, list[str]] = {
			"depolarising":       ["gate_error_rate"],
			"thermal_relaxation": ["t1_us", "t2_us"],
			"readout_error":      ["p0_given_1", "p1_given_0"],
			"crosstalk":          ["zz_coupling_mhz"],
		}
		for key in _required.get(model_type, []):
			if key not in params:
				raise ValueError(f"noise_model_missing_param:{key}")
		record = {
			"noise_model_id": nid,
			"model_id":       model_id,
			"tenant_id":      tenant_id,
			"model_type":     model_type,
			"params":         dict(params),
			"registered_at":  _ts(),
		}
		self._noise_models[nid] = record
		self._record_audit(
			tenant_id, nid, "noise_model_registered", "system", "allow",
			metadata={"model_type": model_type},
		)
		return record

	def noise_model_apply(
		self,
		result_id: str,
		noise_model_id: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""
		Apply a registered noise model to an existing quantum result.

		Injects synthetic noise into measurement counts according to the model
		type and parameters, producing a noisy result record suitable for
		benchmarking mitigation techniques.

		Returns the noisy result alongside fidelity degradation estimates.
		"""
		guard_tenant_id(tenant_id)
		result = self._results.get(_state_key(tenant_id, result_id))
		if result is None:
			raise KeyError(f"quantum_result_not_found:{result_id}")
		nm = self._noise_models.get(noise_model_id)
		if nm is None or nm["tenant_id"] != tenant_id:
			raise KeyError(f"noise_model_not_found:{noise_model_id}")
		model_type = nm["model_type"]
		params = nm["params"]
		noisy_counts: dict[str, int] = {}
		fidelity_loss: float
		if model_type == "depolarising":
			err_rate = float(params.get("gate_error_rate", 0.001))
			fidelity_loss = err_rate * len(result.measurement_counts)
			for state, count in result.measurement_counts.items():
				noise_shift = max(0, int(count * err_rate))
				noisy_counts[state] = max(0, count - noise_shift)
		elif model_type == "thermal_relaxation":
			t1_us = float(params.get("t1_us", 100.0))
			gate_time_ns = float(params.get("gate_time_ns", 50.0))
			decay = 1.0 - math.exp(-gate_time_ns / (t1_us * 1000))
			fidelity_loss = decay
			for state, count in result.measurement_counts.items():
				noisy_counts[state] = max(0, int(count * (1 - decay)))
		elif model_type == "readout_error":
			p01 = float(params.get("p1_given_0", 0.01))
			fidelity_loss = p01
			total = sum(result.measurement_counts.values())
			for state, count in result.measurement_counts.items():
				flip_count = int(total * p01 / max(1, len(result.measurement_counts)))
				noisy_counts[state] = max(0, count - flip_count)
		else:  # crosstalk
			coupling = float(params.get("zz_coupling_mhz", 0.5))
			fidelity_loss = coupling / 1000.0
			for state, count in result.measurement_counts.items():
				noisy_counts[state] = max(0, int(count * (1 - fidelity_loss)))
		applied_id = stable_id("noisy", tenant_id, result_id, noise_model_id)
		record = {
			"applied_id":       applied_id,
			"original_result_id": result_id,
			"noise_model_id":   noise_model_id,
			"tenant_id":        tenant_id,
			"model_type":       model_type,
			"noisy_counts":     noisy_counts,
			"fidelity_loss_estimate": round(min(1.0, fidelity_loss), 6),
			"original_confidence":   result.confidence,
			"noisy_confidence":      round(max(0.0, result.confidence - fidelity_loss), 4),
			"applied_at":       _ts(),
		}
		self._record_audit(
			tenant_id, applied_id, "noise_model_applied", "system", "allow",
			metadata={"model_type": model_type, "result_id": result_id},
		)
		return record

	def circuit_metrics(
		self,
		circuit_id: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""
		Compute structural complexity metrics for a quantum circuit.

		Returns: gate count by type, two-qubit gate fraction, circuit depth
		(critical-path gate count), T-gate count (magic-state overhead),
		and a Meyer-Wallach global entanglement proxy.

		These metrics drive backend selection, optimisation decisions, and QPU
		readiness assessment without needing to run the circuit.
		"""
		guard_tenant_id(tenant_id)
		circuit = self._require_circuit(circuit_id, tenant_id)
		gates = list(circuit.gates)
		gate_counts: dict[str, int] = {}
		for g in gates:
			gate_counts[g] = gate_counts.get(g, 0) + 1
		two_qubit_gates = {"cx", "cz", "swap", "ccx", "iswap", "ecr", "rzz", "rxx"}
		single_qubit_gates = {"h", "x", "y", "z", "s", "t", "sdg", "tdg", "rx", "ry", "rz", "sx", "id"}
		n_two_qubit = sum(gate_counts.get(g, 0) for g in two_qubit_gates)
		n_t_gates = gate_counts.get("t", 0) + gate_counts.get("tdg", 0)
		total_gates = len(gates)
		two_qubit_fraction = round(n_two_qubit / total_gates, 4) if total_gates > 0 else 0.0
		# Circuit depth: synthetic critical-path estimate (two-qubit gates on hot path)
		# Approximation: depth ~ total_gates / qubits + n_two_qubit * 0.5
		n_qubits = circuit.qubits_required
		depth_estimate = max(1, total_gates // max(1, n_qubits) + n_two_qubit // 2)
		# Meyer-Wallach entanglement proxy: increases with two-qubit gate fraction
		mw_entanglement = round(min(1.0, two_qubit_fraction * 2.0), 4)
		return {
			"circuit_id":           circuit_id,
			"tenant_id":            tenant_id,
			"n_qubits":             n_qubits,
			"total_gate_count":     total_gates,
			"gate_counts_by_type":  gate_counts,
			"two_qubit_gate_count": n_two_qubit,
			"two_qubit_fraction":   two_qubit_fraction,
			"t_gate_count":         n_t_gates,
			"circuit_depth_estimate": depth_estimate,
			"mw_entanglement_proxy": mw_entanglement,
			"complexity_tier": (
				"low" if two_qubit_fraction < 0.1 else
				"medium" if two_qubit_fraction < 0.35 else
				"high"
			),
			"computed_at": _ts(),
		}

	def grover_search(
		self,
		oracle_spec: dict[str, Any],
		n_qubits: int,
		marked_items: int,
		tenant_id: str,
		shots: int = 1024,
		run_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Run Grover's search algorithm on the specified oracle.

		oracle_spec: dict with 'function_description' and optional 'clause_count'.
		n_qubits:    number of search-space qubits (search space N = 2^n_qubits).
		marked_items: number of solutions k (must be < N).
		shots:       number of measurement repetitions.

		Returns optimal iteration count, success probability, gate count estimate,
		quadratic speedup ratio vs classical brute force, and measurement distribution.
		"""
		guard_tenant_id(tenant_id)
		if n_qubits < 1 or n_qubits > 20:
			raise ValueError("grover_n_qubits_must_be_1_to_20")
		search_space = 2 ** n_qubits
		if marked_items < 1 or marked_items >= search_space:
			raise ValueError("grover_marked_items_must_be_positive_and_less_than_search_space")
		# Optimal iteration count: floor(pi/4 * sqrt(N/k))
		optimal_iterations = max(1, int(math.floor(math.pi / 4 * math.sqrt(search_space / marked_items))))
		# Theoretical success probability after optimal iterations
		theta = math.asin(math.sqrt(marked_items / search_space))
		success_probability = round(math.sin((2 * optimal_iterations + 1) * theta) ** 2, 6)
		# Gate count: n H-gates + n oracle + n diffuser, each ~n+1 gates, per iteration
		gates_per_iteration = 3 * n_qubits + 4
		total_gate_count = n_qubits + optimal_iterations * gates_per_iteration
		# Classical brute-force expected steps: N/2 on average
		classical_expected_steps = search_space / 2
		speedup_ratio = round(classical_expected_steps / optimal_iterations, 2)
		# Synthetic measurement distribution: concentrate shots on marked states
		marked_shots = int(shots * success_probability)
		unmarked_shots = shots - marked_shots
		unmarked_per_state = max(0, unmarked_shots // max(1, search_space - marked_items))
		measurement_distribution: dict[str, int] = {}
		for i in range(min(search_space, 8)):  # show up to 8 states for brevity
			bitstring = format(i, f"0{n_qubits}b")
			# Assign marked_shots evenly across marked items (first k states)
			if i < marked_items:
				measurement_distribution[bitstring] = marked_shots // marked_items
			else:
				measurement_distribution[bitstring] = unmarked_per_state
		rid = run_id or stable_id("grover", tenant_id, str(n_qubits), str(marked_items))
		record = {
			"run_id":                  rid,
			"tenant_id":               tenant_id,
			"n_qubits":                n_qubits,
			"search_space_size":       search_space,
			"marked_items":            marked_items,
			"optimal_iterations":      optimal_iterations,
			"success_probability":     success_probability,
			"total_gate_count":        total_gate_count,
			"shots":                   shots,
			"measurement_distribution": measurement_distribution,
			"classical_expected_steps": int(classical_expected_steps),
			"quantum_speedup_ratio":    speedup_ratio,
			"oracle_spec":             dict(oracle_spec),
			"computed_at":             _ts(),
		}
		self._record_audit(
			tenant_id, rid, "grover_search_executed", "system", "allow",
			metadata={"n_qubits": n_qubits, "marked_items": marked_items},
		)
		return record

	def fidelity_snapshot_record(
		self,
		backend_id: str,
		tenant_id: str,
		gate_fidelity: float,
		readout_fidelity: float,
		t1_us: float,
		t2_us: float,
		snapshot_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Record a fidelity calibration snapshot for a backend.

		Call periodically (e.g., every 15 minutes) to build a time-series
		fidelity history. Used by fidelity_drift_detect to alert on degradation.

		gate_fidelity:    average single-qubit gate fidelity (0–1).
		readout_fidelity: average measurement assignment fidelity (0–1).
		t1_us:            average T1 relaxation time in microseconds.
		t2_us:            average T2 dephasing time in microseconds.
		"""
		guard_tenant_id(tenant_id)
		self._require_backend(backend_id, tenant_id)
		for label, val in [("gate_fidelity", gate_fidelity), ("readout_fidelity", readout_fidelity)]:
			if not 0.0 <= val <= 1.0:
				raise ValueError(f"{label}_must_be_between_0_and_1")
		if t1_us <= 0 or t2_us <= 0:
			raise ValueError("t1_us_and_t2_us_must_be_positive")
		sid = snapshot_id or stable_id("fidsnap", tenant_id, backend_id, _ts())
		record = {
			"snapshot_id":      sid,
			"backend_id":       backend_id,
			"tenant_id":        tenant_id,
			"gate_fidelity":    round(gate_fidelity, 6),
			"readout_fidelity": round(readout_fidelity, 6),
			"t1_us":            t1_us,
			"t2_us":            t2_us,
			"recorded_at":      _ts(),
		}
		key = f"{tenant_id}:{backend_id}"
		if key not in self._fidelity_snapshots:
			self._fidelity_snapshots[key] = []
		self._fidelity_snapshots[key].append(record)
		self._record_audit(
			tenant_id, sid, "fidelity_snapshot_recorded", "system", "allow",
			metadata={"backend_id": backend_id, "gate_fidelity": gate_fidelity},
		)
		return record

	def fidelity_drift_detect(
		self,
		backend_id: str,
		tenant_id: str,
		window_snapshots: int = 10,
		drift_threshold: float = 0.02,
	) -> dict[str, Any]:
		"""
		Detect calibration drift for a backend from recent fidelity snapshots.

		Computes an exponential moving average (EMA) of gate fidelity over the
		last window_snapshots snapshots and flags drift if the EMA slope exceeds
		drift_threshold per snapshot interval.

		Returns drift status, EMA series, slope estimate, and recommendation.
		Emits a FIDELITY_DRIFT_ALERT audit event if drift is detected.
		"""
		guard_tenant_id(tenant_id)
		self._require_backend(backend_id, tenant_id)
		key = f"{tenant_id}:{backend_id}"
		snapshots = self._fidelity_snapshots.get(key, [])
		recent = snapshots[-window_snapshots:] if snapshots else []
		if len(recent) < 2:
			return {
				"backend_id":     backend_id,
				"tenant_id":      tenant_id,
				"drift_detected": False,
				"reason":         "insufficient_snapshots",
				"snapshot_count": len(recent),
				"checked_at":     _ts(),
			}
		fidelities = [s["gate_fidelity"] for s in recent]
		# EMA with alpha = 2 / (n + 1)
		alpha = 2.0 / (len(fidelities) + 1)
		ema = fidelities[0]
		ema_series: list[float] = [round(ema, 6)]
		for f in fidelities[1:]:
			ema = alpha * f + (1 - alpha) * ema
			ema_series.append(round(ema, 6))
		# Slope: linear regression over EMA series
		n = len(ema_series)
		xs = list(range(n))
		x_mean = sum(xs) / n
		y_mean = sum(ema_series) / n
		ss_xy = sum((xs[i] - x_mean) * (ema_series[i] - y_mean) for i in range(n))
		ss_xx = sum((xs[i] - x_mean) ** 2 for i in range(n))
		slope = round(ss_xy / ss_xx if ss_xx != 0 else 0.0, 8)
		drift_detected = slope < -drift_threshold
		result = {
			"backend_id":         backend_id,
			"tenant_id":          tenant_id,
			"drift_detected":     drift_detected,
			"ema_fidelity_series": ema_series,
			"ema_slope_per_snapshot": slope,
			"drift_threshold":    drift_threshold,
			"latest_fidelity":    fidelities[-1],
			"baseline_fidelity":  fidelities[0],
			"snapshot_count":     len(recent),
			"recommendation": (
				"halt_jobs_and_recalibrate" if drift_detected else "nominal"
			),
			"checked_at": _ts(),
		}
		if drift_detected:
			self._record_audit(
				tenant_id, backend_id, "FIDELITY_DRIFT_ALERT", "system", "allow",
				metadata={"slope": slope, "latest_fidelity": fidelities[-1]},
			)
		return result


# Alias
QuanService = QuantumComputingService
