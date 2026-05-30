"""UI metadata and view models for APG Anomaly Detection."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import AnomService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.signal_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"sources": service.list_sources(tenant_id),
		"baselines": service.list_baselines(tenant_id),
		"signals": service.list_signals(tenant_id),
		"investigations": service.list_investigations(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def signal_board_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	return {
		"signals": service.list_signals(tenant_id),
		"severity_columns": ["critical", "high", "medium", "normal"],
		"actions": ["open_investigation", "mark_false_positive", "close_signal"],
	}


def source_registry_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"sources": service.list_sources(tenant_id),
		"allowed_kinds": contract["configuration"]["sources"]["allowed_kinds"],
		"route": "/anom/sources",
	}


def baseline_console_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	return {
		"sources": service.list_sources(tenant_id),
		"baselines": service.list_baselines(tenant_id),
		"sensitivity_options": ["low", "medium", "high"],
		"required_fields": ["source_id", "metric", "values"],
	}


def detection_workbench_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	return {
		"tenant_id": tenant_id,
		"sources": service.list_sources(tenant_id),
		"baselines": service.list_baselines(tenant_id),
		"recent_signals": service.list_signals(tenant_id),
		"required_fields": ["source_id", "baseline_id", "metric", "value"],
		"route": "/anom/detector",
	}


def investigation_queue_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	return {
		"investigations": service.list_investigations(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"status_groups": ["open", "triage", "mitigating", "closed"],
		"required_fields": ["signal_id", "owner"],
		"closure_required_fields": ["resolution", "closed_by", "resolution_evidence"],
	}


def alert_queue_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"signals": [signal for signal in service.list_signals(tenant_id) if signal["severity"] in {"critical", "high"}],
		"notification_adapter": contract["configuration"]["adapters"]["notification"],
		"route": "/anom/alerts",
	}


def rule_manager_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	return {
		"tenant_id": tenant_id,
		"rules": service.describe(tenant_id)["rule_engine"]["rules"],
		"route": "/anom/rules",
	}


def feedback_review_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	summary = service.signal_summary(tenant_id)
	return {
		"feedback": service.list_feedback(tenant_id),
		"false_positive_rate": summary["false_positive_rate"],
		"requires_tuning_review": summary["false_positive_rate"] > 0.2,
		"labels": ["true_positive", "false_positive", "expected_change"],
	}


def quality_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	summary = service.signal_summary(tenant_id)
	return {
		"tenant_id": tenant_id,
		"summary": summary,
		"feedback": service.list_feedback(tenant_id),
		"tuning_required": summary["false_positive_rate"] > 0.2,
		"route": "/anom/quality",
	}


def audit_timeline_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	return {
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"route": "/anom/audit",
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"theme": contract["theme"],
		"route": "/anom/settings",
	}
