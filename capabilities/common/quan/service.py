"""Service layer for executable Quantum Computing operations."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
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


class QuanService:
	"""In-process backend, circuit, quota, job, result, and experiment service."""

	def __init__(self) -> None:
		self._backends: dict[str, QuantumBackend] = {}
		self._circuits: dict[str, QuantumCircuit] = {}
		self._quota_policies: dict[str, QuantumQuotaPolicy] = {}
		self._jobs: dict[str, QuantumJob] = {}
		self._results: dict[str, QuantumResult] = {}
		self._experiments: dict[str, QuantumExperiment] = {}
		self._audit_events: dict[str, QuanAuditEvent] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

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
		})
		self._raise_if_blocked(result)
		if provider_value != "local" and not credentials_ref:
			raise PermissionError("provider_credentials_required")
		if int(qubit_count) < 1:
			raise PermissionError("backend_qubit_capacity_required")
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
		self._backends[backend.id] = backend
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
		self._quota_policies[policy.id] = policy
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
			"sensitive_input_present": bool(sensitive_input_present),
			"encryption_applied": bool(encryption_applied),
		})
		self._raise_if_blocked(result)
		if not version:
			raise PermissionError("circuit_version_required")
		if int(qubits_required) < 1:
			raise PermissionError("circuit_qubit_requirement_required")
		normalized_gates = normalize_gates(gates)
		if not normalized_gates:
			raise PermissionError("circuit_gates_required")
		if not experiment_metadata:
			raise PermissionError("experiment_metadata_required")
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
			experiment_metadata=dict(experiment_metadata),
			status="ready",
		)
		self._circuits[circuit.id] = circuit
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
		actor: str = "quan",
	) -> dict[str, Any]:
		backend = self._require_backend(backend_id, tenant_id)
		circuit = self._require_circuit(circuit_id, tenant_id)
		policy = self._quota_policy_for_backend(tenant_id, backend.id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "submit_job",
			"quota_policy_attached": policy is not None,
			"shot_count": int(shot_count),
			"job_review_recorded": bool(job_review_recorded),
		})
		self._raise_if_blocked(result)
		if not submitted_by:
			raise PermissionError("job_submitter_required")
		if not retry_policy_attached:
			raise PermissionError("retry_policy_required")
		if int(shot_count) < 1:
			raise PermissionError("job_shot_count_required")
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
		self._jobs[job.id] = job
		self._record_audit(tenant_id, job.id, "job_submitted", actor, "allow")
		return job.to_dict()

	def complete_job(
		self,
		result_id: str,
		tenant_id: str,
		job_id: str,
		actor: str = "quan",
	) -> dict[str, Any]:
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
		self._results[result.id] = result
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
		if not hypothesis:
			raise PermissionError("experiment_hypothesis_required")
		if "post-quantum" in hypothesis.lower() and not post_quantum_review_recorded:
			raise PermissionError("post_quantum_review_required")
		for job_id in job_ids:
			self._require_job(job_id, tenant_id)
		experiment = QuantumExperiment(
			id=experiment_id,
			tenant_id=tenant_id,
			name=name,
			circuit_id=circuit.id,
			job_ids=tuple(job_ids),
			hypothesis=hypothesis,
			post_quantum_review_recorded=bool(post_quantum_review_recorded),
		)
		self._experiments[experiment.id] = experiment
		self._record_audit(tenant_id, experiment.id, "experiment_created", actor, "allow")
		return experiment.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
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

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"backend_count": len(self.list_backends(tenant_id)),
			"approved_backend_count": sum(1 for item in self._backends.values() if item.tenant_id == tenant_id and item.approved),
			"circuit_count": len(self.list_circuits(tenant_id)),
			"quota_policy_count": len(self.list_quota_policies(tenant_id)),
			"job_count": len(self.list_jobs(tenant_id)),
			"completed_job_count": sum(1 for item in self._jobs.values() if item.tenant_id == tenant_id and item.status == "completed"),
			"result_count": len(self.list_results(tenant_id)),
			"experiment_count": len(self.list_experiments(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_blocked(result)

	def _require_backend(self, backend_id: str, tenant_id: str) -> QuantumBackend:
		backend = self._backends.get(backend_id)
		if backend is None or backend.tenant_id != tenant_id:
			raise KeyError("quantum_backend_not_found")
		return backend

	def _require_circuit(self, circuit_id: str, tenant_id: str) -> QuantumCircuit:
		circuit = self._circuits.get(circuit_id)
		if circuit is None or circuit.tenant_id != tenant_id:
			raise KeyError("quantum_circuit_not_found")
		return circuit

	def _require_job(self, job_id: str, tenant_id: str) -> QuantumJob:
		job = self._jobs.get(job_id)
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
	) -> None:
		event = QuanAuditEvent(
			id=stable_id("audit", tenant_id, subject_id, event_type, len(self._audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			actor=actor,
			decision=decision,
			reasons=reasons,
		)
		self._audit_events[event.id] = event

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(values, key=lambda item: item.id)]

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(action.get("reason", "quantum_policy_blocked") for action in result["actions"])
