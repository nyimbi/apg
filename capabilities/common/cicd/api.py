"""API helpers for APG Continuous Integration and Delivery."""

from __future__ import annotations

from typing import Any

from .service import CicdService


SERVICE = CicdService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"streaming": contract["streaming"],
		**SERVICE.pipeline_summary(tenant_id),
	}


def create_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_pipeline(
		pipeline_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload.get("owner") or ""),
		source_ref=str(payload.get("source_ref") or ""),
		worker_pool=str(payload.get("worker_pool") or ""),
		stages=[str(item) for item in payload.get("stages", [])],
		secret_scope=str(payload.get("secret_scope") or ""),
		cache_policy=str(payload.get("cache_policy") or ""),
		quality_gate=str(payload.get("quality_gate") or ""),
		parallel_job_count=int(payload.get("parallel_job_count", 1)),
		capacity_review_recorded=bool(payload.get("capacity_review_recorded", True)),
	)


def approve_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_pipeline(
		pipeline_id=str(payload["id"]),
		reviewer=str(payload.get("reviewer") or "reviewer"),
		tenant_id=str(payload["tenant_id"]) if payload.get("tenant_id") else None,
	)


def run_build(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.run_build(
		build_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		pipeline_id=str(payload["pipeline_id"]),
		commit_ref=str(payload.get("commit_ref") or ""),
		triggered_by=str(payload.get("triggered_by") or "system"),
		secret_scope_attached=bool(payload.get("secret_scope_attached", True)),
		log_trace_captured=bool(payload.get("log_trace_captured", True)),
	)


def publish_artifact(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_artifact(
		artifact_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		build_id=str(payload["build_id"]),
		name=str(payload.get("name") or payload["id"]),
		version=str(payload.get("version") or "0.0.0"),
		signed=bool(payload.get("signed", False)),
	)


def record_quality_gate(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_quality_gate(
		gate_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		artifact_id=str(payload["artifact_id"]),
		tests_passed=bool(payload.get("tests_passed", False)),
		security_scan_passed=bool(payload.get("security_scan_passed", False)),
		approval_recorded=bool(payload.get("approval_recorded", False)),
	)


def promote_artifact(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.promote_artifact(
		promotion_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		artifact_id=str(payload["artifact_id"]),
		quality_gate_id=str(payload["quality_gate_id"]),
		source_environment=str(payload.get("source_environment") or "build"),
		target_environment=str(payload.get("target_environment") or "staging"),
		requested_by=str(payload.get("requested_by") or "release-manager"),
		approval_recorded=bool(payload.get("approval_recorded", False)),
		approver=str(payload.get("approver") or "") or None,
		environment_policy_attached=bool(payload.get("environment_policy_attached", True)),
	)


def register_delivery_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_delivery_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		agent_id=str(payload["id"]),
		name=str(payload.get("name") or payload["id"]),
		runtime=str(payload.get("runtime") or ""),
		role=str(payload.get("role") or ""),
		scope=str(payload.get("scope") or ""),
		contribution_disclosed=bool(payload.get("contribution_disclosed", False)),
		policy_ref=str(payload.get("policy_ref") or ""),
		registered=bool(payload.get("registered", True)),
	)


def change_pipeline_state(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.change_pipeline_state(
		tenant_id=str(payload.get("tenant_id") or "default"),
		pipeline_id=str(payload["id"]),
		status=str(payload.get("status") or "paused"),
		reason=str(payload.get("reason") or ""),
		audit_recorded=bool(payload.get("audit_recorded", True)),
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


def list_pipelines(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_pipelines(tenant_id)


def list_builds(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_builds(tenant_id)


def list_artifacts(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_artifacts(tenant_id)


def list_gates(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_gates(tenant_id)


def list_promotions(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_promotions(tenant_id)


def list_delivery_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_delivery_agents(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)
