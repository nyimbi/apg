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
	service = _service_or_default(service)
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
		"restore_approvals": service.list_restore_approvals(tenant_id),
		"retention_dispositions": service.list_retention_dispositions(tenant_id),
		"reports": service.list_reports(tenant_id),
		"backup_agents": service.list_backup_agents(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"restore_review_queue": [
			item for item in service.list_restores(tenant_id)
			if item["status"] == "pending_review"
		],
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def plan_manager_model(
	service: BkupService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	return {
		"tenant_id": tenant_id,
		"plans": service.list_plans(tenant_id),
		"reports": service.list_reports(tenant_id),
	}


def restore_console_model(
	service: BkupService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	restores = service.list_restores(tenant_id)
	return {
		"tenant_id": tenant_id,
		"snapshots": service.list_snapshots(tenant_id),
		"restores": restores,
		"restore_approvals": service.list_restore_approvals(tenant_id),
		"pending_review": [item for item in restores if item["status"] == "pending_review"],
		"completed": [item for item in restores if item["status"] == "completed"],
	}


def restore_approval_model(
	service: BkupService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	approvals = service.list_restore_approvals(tenant_id)
	return {
		"tenant_id": tenant_id,
		"approvals": approvals,
		"pending_approvals": [item for item in approvals if item["status"] == "pending"],
		"decided_approvals": [item for item in approvals if item["status"] != "pending"],
		"required_decision_fields": ["reviewer", "decision", "notes"],
		"guardrails": ["independent_reviewer", "matching_snapshot_target", "reviewer_notes_required"],
	}


def retention_disposition_model(
	service: BkupService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	dispositions = service.list_retention_dispositions(tenant_id)
	return {
		"tenant_id": tenant_id,
		"snapshots": service.list_snapshots(tenant_id),
		"dispositions": dispositions,
		"pending_dispositions": [item for item in dispositions if item["status"] == "pending"],
		"decided_dispositions": [item for item in dispositions if item["status"] != "pending"],
		"required_decision_fields": ["reviewer", "decision", "notes"],
		"guardrails": ["legal_hold_blocks_disposition", "independent_reviewer", "reviewer_notes_required"],
	}


def audit_model(
	service: BkupService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"summary": service.continuity_summary(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def backup_agent_model(
	service: BkupService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": service.list_backup_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["backup_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["backup_agents"]["allowed_roles"],
		"required_fields": ["name", "runtime", "role", "scope", "contribution_disclosed"],
		"actions": ["register", "scope", "review_contribution", "deactivate"],
	}


def analytics_model(
	service: BkupService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	summary = service.continuity_summary(tenant_id)
	return {
		"tenant_id": tenant_id,
		"summary": summary,
		"restore_completion_rate": _safe_ratio(summary["completed_restore_count"], summary["restore_count"]),
		"available_snapshot_rate": _safe_ratio(summary["available_snapshot_count"], summary["snapshot_count"]),
		"continuity_review_rate": _safe_ratio(summary["review_required_report_count"], summary["continuity_report_count"]),
		"agent_coverage": _safe_ratio(summary["backup_agent_count"], max(summary["plan_count"], 1)),
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def _service_or_default(service: BkupService | None) -> BkupService:
	if service is not None:
		return service
	try:
		from .api import SERVICE

		return SERVICE
	except ImportError:  # pragma: no cover - standalone package loading path
		return BkupService()


def _safe_ratio(numerator: int, denominator: int) -> float:
	return round(numerator / denominator, 4) if denominator else 0.0
