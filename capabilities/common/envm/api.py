"""API helpers for APG Environment Management."""

from __future__ import annotations

from typing import Any

from .service import EnvmService


SERVICE = EnvmService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**summary,
	}


def register_environment(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_environment(
		environment_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		stage=str(payload["stage"]),
		region=str(payload["region"]),
		owner=str(payload["owner"]),
		configuration_source=str(payload["configuration_source"]),
		rbac_policy=str(payload["rbac_policy"]),
		secret_scope_policy=str(payload["secret_scope_policy"]),
		approval_recorded=bool(payload.get("approval_recorded", True)),
		status=str(payload.get("status") or "active"),
	)


def create_promotion_path(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_promotion_path(
		path_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		source_environment_id=str(payload["source_environment_id"]),
		target_environment_id=str(payload["target_environment_id"]),
		deployment_link=str(payload["deployment_link"]),
		rollback_environment_id=str(payload["rollback_environment_id"]),
		approval_recorded=bool(payload.get("approval_recorded", False)),
		promotion_path_attached=bool(payload.get("promotion_path_attached", True)),
	)


def run_promotion(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.run_promotion(
		run_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		promotion_path_id=str(payload["promotion_path_id"]),
		requested_by=str(payload["requested_by"]),
		artifact_ref=str(payload["artifact_ref"]),
		approval_recorded=bool(payload.get("approval_recorded", False)),
	)


def record_drift(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_drift(
		report_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		environment_id=str(payload["environment_id"]),
		declared_version=str(payload["declared_version"]),
		observed_version=str(payload["observed_version"]),
		changed_items=int(payload.get("changed_items") or 0),
		total_items=int(payload.get("total_items") or 0),
		drift_review_recorded=bool(payload.get("drift_review_recorded", False)),
		remediation_action=str(payload.get("remediation_action") or ""),
	)


def register_secret_scope(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_secret_scope(
		scope_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		environment_id=str(payload["environment_id"]),
		name=str(payload.get("name") or payload["id"]),
		policy_ref=str(payload["policy_ref"]),
		secret_refs=tuple(payload.get("secret_refs") or ()),
		access_roles=tuple(payload.get("access_roles") or ()),
		secret_policy_attached=bool(payload.get("secret_policy_attached", True)),
	)


def register_envm_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_envm_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload["scope"]),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		agent_id=payload.get("id"),
	)


def validate_batch_environment_mutation(event_stream: str) -> dict[str, Any]:
	return SERVICE.validate_batch_environment_mutation(event_stream)


def list_environments(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_environments(tenant_id)


def list_promotion_paths(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_promotion_paths(tenant_id)


def list_promotion_runs(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_promotion_runs(tenant_id)


def list_drift_reports(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_drift_reports(tenant_id)


def list_secret_scopes(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_secret_scopes(tenant_id)


def list_envm_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_envm_agents(tenant_id)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)
