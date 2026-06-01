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
	sources = service.list_sources(tenant_id)
	baselines = service.list_baselines(tenant_id)
	signals = service.list_signals(tenant_id)
	feedback = service.list_feedback(tenant_id)
	agents = service.list_anomaly_agents(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.signal_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"sources": sources,
		"baselines": baselines,
		"signals": signals,
		"investigations": service.list_investigations(tenant_id),
		"feedback": feedback,
		"anomaly_agents": agents,
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"pending_reviews": {
			"sources": _pending_review(sources),
			"baselines": _pending_review(baselines),
			"signals": _pending_review(signals),
			"feedback": _pending_review(feedback),
			"agents": _pending_review(agents),
		},
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def signal_board_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	signals = service.list_signals(tenant_id)
	return {
		"signals": signals,
		"pending_review": _pending_review(signals),
		"severity_columns": ["critical", "high", "medium", "normal"],
		"actions": ["open_investigation", "mark_false_positive", "close_signal"],
	}


def source_registry_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	contract = service.describe(tenant_id)
	sources = service.list_sources(tenant_id)
	return {
		"tenant_id": tenant_id,
		"sources": sources,
		"pending_review": _pending_review(sources),
		"allowed_kinds": contract["configuration"]["sources"]["allowed_kinds"],
		"route": "/anom/sources",
	}


def baseline_console_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	baselines = service.list_baselines(tenant_id)
	return {
		"sources": service.list_sources(tenant_id),
		"baselines": baselines,
		"pending_review": _pending_review(baselines),
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
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rules": contract["rule_engine"]["rules"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"route": "/anom/rules",
	}


def feedback_review_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	feedback = service.list_feedback(tenant_id)
	summary = service.signal_summary(tenant_id)
	return {
		"feedback": feedback,
		"pending_review": _pending_review(feedback),
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
	feedback = service.list_feedback(tenant_id)
	return {
		"tenant_id": tenant_id,
		"summary": summary,
		"feedback": feedback,
		"pending_feedback_review": _pending_review(feedback),
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


def anomaly_agent_roster_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	contract = service.describe(tenant_id)
	agents = service.list_anomaly_agents(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/anom/agents",
		"agents": agents,
		"pending_review": [item for item in agents if item["status"] == "pending_review"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
	}


def lifecycle_batch_model(
	service: AnomService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AnomService()
	contract = service.describe(tenant_id)
	batches = service.list_lifecycle_batches(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/anom/lifecycle",
		"batches": batches,
		"denied": [item for item in batches if item["status"] == "denied"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"topics": contract["streaming"]["topics"],
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
		"route": "/anom/settings",
	}


def _pending_review(records: list[dict[str, object]]) -> list[dict[str, object]]:
	return [item for item in records if item.get("status") == "pending_review"]
