"""UI metadata helpers for the Shutdown and Lifecycle Control capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import ShdnService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: ShdnService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ShdnService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def service_console_model(
	service: ShdnService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ShdnService()
	return {
		"route": "/shdn/services",
		"tenant_id": tenant_id,
		"targets": service.list_targets(tenant_id),
		"state_filters": ["running", "draining", "quiesced", "snapshot_ready", "stopped", "recovered", "failed"],
		"target_types": ["service", "worker", "database", "queue", "tenant_app", "integration"],
	}


def plan_builder_model(
	service: ShdnService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ShdnService()
	return {
		"route": "/shdn/plans",
		"tenant_id": tenant_id,
		"plans": service.list_plans(tenant_id),
		"approval_required": True,
		"required_fields": ["owner", "target_ids", "reason", "rollback_plan_ref", "restart_sequence", "maintenance_window_ref"],
	}


def execution_monitor_model(
	service: ShdnService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ShdnService()
	return {
		"route": "/shdn/executions",
		"tenant_id": tenant_id,
		"drains": service.list_drains(tenant_id),
		"executions": service.list_executions(tenant_id),
		"statuses": ["pending", "draining", "quiesced", "completed", "blocked"],
	}


def approvals_model(
	service: ShdnService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ShdnService()
	return {
		"route": "/shdn/approvals",
		"tenant_id": tenant_id,
		"plans": [
			plan
			for plan in service.list_plans(tenant_id)
			if plan["status"] in {"approved", "scheduled", "blocked"}
		],
		"force_shutdown_review_required": True,
	}


def recovery_center_model(
	service: ShdnService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ShdnService()
	return {
		"route": "/shdn/recovery",
		"tenant_id": tenant_id,
		"snapshots": service.list_snapshots(tenant_id),
		"recoveries": service.list_recoveries(tenant_id),
		"required_evidence": ["backup_snapshot", "restore_test", "post_shutdown_health_check", "incident_link"],
	}


def audit_model(
	service: ShdnService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ShdnService()
	return {
		"route": "/shdn/audit",
		"tenant_id": tenant_id,
		"events": service.list_audit_events(tenant_id),
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/shdn/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"theme": contract["theme"],
	}
