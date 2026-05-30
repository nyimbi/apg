"""API helpers for the APG Sandbox/Testing Environment capability."""

from __future__ import annotations

from typing import Any

from .service import SboxService


SERVICE = SboxService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"theme": contract["theme"]["name"],
		**SERVICE.dashboard_summary(tenant_id),
	}


def create_isolation_profile(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_isolation_profile(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		level=str(payload.get("level") or "strict"),
		approved_by=payload.get("approved_by"),
		outbound_network_allowed=bool(payload.get("outbound_network_allowed", False)),
		network_approval_recorded=bool(payload.get("network_approval_recorded", False)),
		secret_redaction_enabled=bool(payload.get("secret_redaction_enabled", True)),
		data_masking_enabled=bool(payload.get("data_masking_enabled", True)),
	)


def create_template(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_template(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload.get("runtime") or "python"),
		owner=str(payload["owner"]),
		default_ttl_hours=int(payload.get("default_ttl_hours", 24)),
		plugin_test_policy_required=bool(payload.get("plugin_test_policy_required", True)),
		tags=list(payload.get("tags") or []),
	)


def register_dataset(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_dataset(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		dataset_type=str(payload.get("dataset_type") or "synthetic"),
		owner=str(payload["owner"]),
		lineage=str(payload["lineage"]),
		retention_days=int(payload.get("retention_days", 30)),
		production_review_recorded=bool(payload.get("production_review_recorded", False)),
		masked=bool(payload.get("masked", True)),
	)


def create_sandbox(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_sandbox(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		template_id=str(payload["template_id"]),
		isolation_profile_id=str(payload["isolation_profile_id"]),
		owner=str(payload["owner"]),
		ttl_hours=int(payload["ttl_hours"]) if "ttl_hours" in payload else None,
		dataset_ids=list(payload.get("dataset_ids") or []),
		lifecycle_review_recorded=bool(payload.get("lifecycle_review_recorded", False)),
		secret_access_requested=bool(payload.get("secret_access_requested", False)),
		outbound_network_requested=bool(payload.get("outbound_network_requested", False)),
	)


def start_run(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.start_run(
		tenant_id=str(payload.get("tenant_id") or "default"),
		sandbox_id=str(payload["sandbox_id"]),
		run_type=str(payload.get("run_type") or "integration"),
		requested_by=str(payload["requested_by"]),
		tests_requested=int(payload.get("tests_requested", 1)),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def complete_run(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.complete_run(
		tenant_id=str(payload.get("tenant_id") or "default"),
		run_id=str(payload["run_id"]),
		tests_passed=int(payload.get("tests_passed", 0)),
		tests_failed=int(payload.get("tests_failed", 0)),
		tests_blocked=int(payload.get("tests_blocked", 0)),
		logs=list(payload.get("logs") or []),
	)


def expire_sandbox(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.expire_sandbox(
		tenant_id=str(payload.get("tenant_id") or "default"),
		sandbox_id=str(payload["sandbox_id"]),
		actor=str(payload.get("actor") or "system"),
	)


def register_sbox_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_sbox_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload["scope"]),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		agent_id=str(payload["id"]) if payload.get("id") else None,
	)


def validate_batch_sandbox_mutation(event_stream: str) -> dict[str, Any]:
	return SERVICE.validate_batch_sandbox_mutation(event_stream)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def list_sandboxes(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_sandboxes(tenant_id)


def list_runs(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_runs(tenant_id)


def list_sbox_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_sbox_agents(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
