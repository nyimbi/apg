"""UI metadata helpers for APG Backup and Restore."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import BkupService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: BkupService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or BkupService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.continuity_summary(tenant_id),
		"plans": service.list_plans(tenant_id),
		"snapshots": service.list_snapshots(tenant_id),
		"restores": service.list_restores(tenant_id),
		"reports": service.list_reports(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"restore_review_queue": [
			item for item in service.list_restores(tenant_id)
			if item["status"] == "pending_review"
		],
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def plan_manager_model(
	service: BkupService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or BkupService()
	return {
		"tenant_id": tenant_id,
		"plans": service.list_plans(tenant_id),
		"reports": service.list_reports(tenant_id),
	}


def restore_console_model(
	service: BkupService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or BkupService()
	restores = service.list_restores(tenant_id)
	return {
		"tenant_id": tenant_id,
		"snapshots": service.list_snapshots(tenant_id),
		"restores": restores,
		"pending_review": [item for item in restores if item["status"] == "pending_review"],
		"completed": [item for item in restores if item["status"] == "completed"],
	}
