"""API helpers for the Deployment Management capability."""

from __future__ import annotations

from typing import Any

from .service import DeplService


SERVICE = DeplService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"environment_count": summary["environment_count"],
		"release_count": summary["release_count"],
		"deployed_run_count": summary["deployed_run_count"],
		"pending_review_count": summary["pending_review_count"],
		"deployment_agent_count": summary["deployment_agent_count"],
		"audit_event_count": summary["audit_event_count"],
		"governance_posture": summary["governance_posture"],
	}


def register_environment(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_environment(
		environment_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		tier=str(payload.get("tier") or "nonproduction"),
		owner=str(payload.get("owner") or ""),
		policy=str(payload.get("policy") or ""),
		approvers=[str(item) for item in payload.get("approvers", [])],
	)


def create_release(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_release(
		release_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		version=str(payload["version"]),
		owner=str(payload.get("owner") or ""),
		manifest=dict(payload.get("manifest") or {}),
		artifact_digest=str(payload.get("artifact_digest") or ""),
		artifact_signature=str(payload.get("artifact_signature") or ""),
		change_ticket=str(payload.get("change_ticket") or ""),
		created_by=str(payload["created_by"]),
	)


def attach_rollback_plan(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.attach_rollback_plan(
		rollback_plan_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		release_id=str(payload["release_id"]),
		owner=str(payload.get("owner") or ""),
		steps=[str(item) for item in payload.get("steps", [])],
		tested=bool(payload.get("tested", False)),
	)


def record_health_gate(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_health_gate(
		health_gate_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		release_id=str(payload["release_id"]),
		checks={str(key): bool(value) for key, value in dict(payload.get("checks") or {}).items()},
		report_reference=str(payload.get("report_reference") or ""),
		log_trace_link=str(payload.get("log_trace_link") or ""),
		recorded_by=str(payload["recorded_by"]),
	)


def create_deployment_plan(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_deployment_plan(
		plan_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		release_id=str(payload["release_id"]),
		environment_id=str(payload["environment_id"]),
		strategy=str(payload.get("strategy") or "rolling"),
		requested_by=str(payload["requested_by"]),
		approval_recorded=bool(payload.get("approval_recorded", False)),
		rollback_plan_id=str(payload["rollback_plan_id"]),
		health_gate_id=str(payload["health_gate_id"]),
		change_ticket=str(payload.get("change_ticket") or ""),
		canary_percent=int(payload.get("canary_percent") or 0),
		canary_review_recorded=bool(payload.get("canary_review_recorded", True)),
	)


def approve_deployment_plan(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_deployment_plan(
		plan_id=str(payload["plan_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
	)


def execute_deployment(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.execute_deployment(
		run_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		plan_id=str(payload["plan_id"]),
		actor=str(payload["actor"]),
		log_trace_link=str(payload.get("log_trace_link") or ""),
		health_report_reference=str(payload.get("health_report_reference") or ""),
	)


def execute_rollback(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.execute_rollback(
		rollback_event_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		run_id=str(payload["run_id"]),
		actor=str(payload["actor"]),
		reason=str(payload.get("reason") or ""),
	)


def register_deployment_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_deployment_agent(
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


def change_deployment_plan_state(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.change_deployment_plan_state(
		tenant_id=str(payload.get("tenant_id") or "default"),
		plan_id=str(payload["plan_id"]),
		status=str(payload["status"]),
		reason=str(payload.get("reason") or ""),
		actor=str(payload.get("actor") or "deployment-operator"),
		audit_recorded=bool(payload.get("audit_recorded", True)),
	)


def validate_batch_deployment_mutation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_batch_deployment_mutation(
		tenant_id=str(payload.get("tenant_id") or "default"),
		event_stream=str(payload.get("event_stream") or ""),
		actor=str(payload.get("actor") or "deployment-operator"),
	)


def deployment_state(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"summary": SERVICE.dashboard_summary(tenant_id),
		"environments": SERVICE.list_environments(tenant_id),
		"releases": SERVICE.list_releases(tenant_id),
		"rollback_plans": SERVICE.list_rollback_plans(tenant_id),
		"health_gates": SERVICE.list_health_gates(tenant_id),
		"deployment_plans": SERVICE.list_deployment_plans(tenant_id),
		"deployment_runs": SERVICE.list_deployment_runs(tenant_id),
		"rollback_events": SERVICE.list_rollback_events(tenant_id),
		"deployment_agents": SERVICE.list_deployment_agents(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
	}
