"""UI metadata helpers for APG Continuous Integration and Delivery."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import CicdService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: CicdService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CicdService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.pipeline_summary(tenant_id),
		"pipelines": service.list_pipelines(tenant_id),
		"builds": service.list_builds(tenant_id),
		"artifacts": service.list_artifacts(tenant_id),
		"gates": service.list_gates(tenant_id),
		"promotions": service.list_promotions(tenant_id),
		"delivery_agents": service.list_delivery_agents(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def pipeline_console_model(
	service: CicdService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CicdService()
	pipelines = service.list_pipelines(tenant_id)
	return {
		"tenant_id": tenant_id,
		"pipelines": pipelines,
		"active": [item for item in pipelines if item["status"] == "active"],
		"pending_review": [item for item in pipelines if item["status"] == "pending_review"],
		"actions": ["create_pipeline", "approve_pipeline", "change_pipeline_state"],
	}


def build_monitor_model(
	service: CicdService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CicdService()
	return {
		"tenant_id": tenant_id,
		"builds": service.list_builds(tenant_id),
		"artifacts": service.list_artifacts(tenant_id),
		"actions": ["run_build", "publish_artifact"],
	}


def artifact_registry_model(
	service: CicdService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CicdService()
	return {
		"tenant_id": tenant_id,
		"artifacts": service.list_artifacts(tenant_id),
		"gates": service.list_gates(tenant_id),
		"guardrails": ["artifact_requires_signature", "promotion_requires_quality_gate"],
	}


def quality_gate_model(
	service: CicdService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CicdService()
	return {
		"tenant_id": tenant_id,
		"gates": service.list_gates(tenant_id),
		"guardrails": ["gate_requires_security_scan", "promotion_requires_quality_gate", "promotion_requires_approval"],
		"actions": ["record_quality_gate"],
	}


def promotion_console_model(
	service: CicdService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CicdService()
	gates = service.list_gates(tenant_id)
	return {
		"tenant_id": tenant_id,
		"gates": gates,
		"failed_gates": [item for item in gates if item["status"] == "failed"],
		"promotions": service.list_promotions(tenant_id),
		"actions": ["promote_artifact"],
	}


def delivery_agents_model(
	service: CicdService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CicdService()
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": service.list_delivery_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["delivery_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["delivery_agents"]["allowed_roles"],
		"guardrails": [
			"delivery_agent_requires_registration",
			"delivery_agent_runtime_supported",
			"delivery_agent_role_supported",
			"delivery_agent_requires_scope",
			"delivery_agent_requires_disclosure",
		],
		"actions": ["register_delivery_agent"],
	}


def audit_trail_model(
	service: CicdService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CicdService()
	return {
		"tenant_id": tenant_id,
		"events": service.list_audit_events(tenant_id),
		"guardrails": ["cicd_state_change_requires_reason", "cicd_state_change_requires_audit", "cross_tenant_pipeline_access_denied"],
	}


def analytics_model(
	service: CicdService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CicdService()
	summary = service.pipeline_summary(tenant_id)
	return {
		"tenant_id": tenant_id,
		"summary": summary,
		"quality": {
			"build_pass_rate": summary["passed_build_count"] / summary["build_count"] if summary["build_count"] else 0,
			"promotion_rate": summary["promotion_count"] / summary["artifact_count"] if summary["artifact_count"] else 0,
		},
		"streaming": get_capability_contract(tenant_id)["streaming"],
	}


def settings_model(
	service: CicdService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CicdService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
		"permissions": ["cicd:view", "cicd:manage_pipelines", "cicd:run_builds", "cicd:promote", "cicd:audit", "cicd:admin"],
	}
