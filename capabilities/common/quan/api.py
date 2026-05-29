"""API helpers for the Quantum Computing capability."""

from __future__ import annotations

from typing import Any

from .service import QuanService


SERVICE = QuanService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"backend_count": summary["backend_count"],
		"circuit_count": summary["circuit_count"],
		"job_count": summary["job_count"],
	}


def register_backend(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_backend(
		backend_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		provider=str(payload["provider"]),
		backend_type=str(payload.get("backend_type") or "simulator"),
		qubit_count=int(payload.get("qubit_count") or 1),
		approved=bool(payload.get("approved", False)),
		credentials_ref=payload.get("credentials_ref"),
		simulator_fallback=bool(payload.get("simulator_fallback", True)),
		metadata=dict(payload.get("metadata") or {}),
		actor=str(payload.get("actor") or "quan"),
	)


def attach_quota_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.attach_quota_policy(
		policy_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		backend_id=str(payload["backend_id"]),
		max_shots_per_job=int(payload["max_shots_per_job"]),
		max_jobs_per_day=int(payload["max_jobs_per_day"]),
		cost_limit=float(payload["cost_limit"]),
		retry_policy=str(payload.get("retry_policy") or "safe_retry"),
		actor=str(payload.get("actor") or "quan"),
	)


def create_circuit(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_circuit(
		circuit_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload["owner"]),
		version=str(payload["version"]),
		qubits_required=int(payload["qubits_required"]),
		gates=list(payload.get("gates") or ()),
		sensitive_input_present=bool(payload.get("sensitive_input_present", False)),
		encryption_applied=bool(payload.get("encryption_applied", False)),
		experiment_metadata=dict(payload.get("experiment_metadata") or {}),
		actor=str(payload.get("actor") or "quan"),
	)


def submit_job(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.submit_job(
		job_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		backend_id=str(payload["backend_id"]),
		circuit_id=str(payload["circuit_id"]),
		submitted_by=str(payload["submitted_by"]),
		shot_count=int(payload["shot_count"]),
		job_review_recorded=bool(payload.get("job_review_recorded", False)),
		retry_policy_attached=bool(payload.get("retry_policy_attached", True)),
		actor=str(payload.get("actor") or "quan"),
	)


def complete_job(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.complete_job(
		result_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		job_id=str(payload["job_id"]),
		actor=str(payload.get("actor") or "quan"),
	)


def create_experiment(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_experiment(
		experiment_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		circuit_id=str(payload["circuit_id"]),
		job_ids=list(payload.get("job_ids") or ()),
		hypothesis=str(payload["hypothesis"]),
		post_quantum_review_recorded=bool(payload.get("post_quantum_review_recorded", False)),
		actor=str(payload.get("actor") or "quan"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
