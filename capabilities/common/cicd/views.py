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
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
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
	}
