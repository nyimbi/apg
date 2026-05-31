"""UI metadata helpers for the Help and Knowledge Base capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import HelpService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: HelpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or HelpService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"help_agents": service.list_help_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"routes": capability_routes(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def help_center_model(
	service: HelpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or HelpService()
	return {
		"tenant_id": tenant_id,
		"articles": [
			article for article in service.list_articles(tenant_id)
			if article["status"] == "published"
		],
		"routes": capability_routes(tenant_id),
	}


def source_registry_model(
	service: HelpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or HelpService()
	contract = service.describe(tenant_id)
	return {
		"route": "/help/sources",
		"tenant_id": tenant_id,
		"sources": service.list_sources(tenant_id),
		"approval_required": contract["configuration"]["sources"]["source_approval_required"],
		"theme": contract["theme"]["components"]["source_registry"],
	}


def article_editor_model(
	service: HelpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or HelpService()
	return {
		"tenant_id": tenant_id,
		"drafts": [
			article for article in service.list_articles(tenant_id)
			if article["status"] in {"draft", "review"}
		],
		"visibility_options": ["public", "internal", "restricted"],
		"supported_locales": service.describe(tenant_id)["configuration"]["content"]["supported_locales"],
	}


def answer_console_model(
	service: HelpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or HelpService()
	return {
		"tenant_id": tenant_id,
		"answers": service.list_answers(tenant_id),
		"requires_citations": True,
		"minimum_confidence": service.describe(tenant_id)["configuration"]["answers"]["minimum_answer_confidence"],
	}


def localization_workbench_model(
	service: HelpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or HelpService()
	contract = service.describe(tenant_id)
	return {
		"route": "/help/localization",
		"tenant_id": tenant_id,
		"localizations": service.list_localizations(tenant_id),
		"supported_locales": contract["configuration"]["localization"]["supported_locales"],
		"theme": contract["theme"]["components"]["localization_workbench"],
	}


def curation_queue_model(
	service: HelpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or HelpService()
	return {
		"tenant_id": tenant_id,
		"curation_items": service.list_curation_items(tenant_id),
		"feedback": [
			item for item in service.list_feedback(tenant_id)
			if item["requires_review"]
		],
	}


def help_agent_roster_model(
	service: HelpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or HelpService()
	contract = service.describe(tenant_id)
	agents = service.list_help_agents(tenant_id)
	return {
		"route": "/help/agents",
		"tenant_id": tenant_id,
		"agents": agents,
		"active": [agent for agent in agents if agent["status"] == "active"],
		"pending_review": [agent for agent in agents if agent["status"] == "pending_review"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"actions": ["register_help_agent", "record_human_help_agent_approval"],
		"theme_component": "help_agent_roster",
	}


def lifecycle_batch_model(
	service: HelpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or HelpService()
	contract = service.describe(tenant_id)
	batches = service.list_lifecycle_batches(tenant_id)
	return {
		"route": "/help/lifecycle",
		"tenant_id": tenant_id,
		"lifecycle_stream": contract["streaming"]["lifecycle_stream"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"batches": batches,
		"accepted": [batch for batch in batches if batch["status"] == "accepted"],
		"denied": [batch for batch in batches if batch["status"] == "denied"],
		"actions": ["validate_lifecycle_batch", "inspect_bytewax_lifecycle"],
		"theme_component": "bytewax_lifecycle_panel",
	}


def audit_model(
	service: HelpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or HelpService()
	contract = service.describe(tenant_id)
	return {
		"route": "/help/audit",
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"event_stream": contract["configuration"]["observability"]["event_stream"],
		"theme": contract["theme"]["components"]["audit_timeline"],
	}


def support_analytics_model(
	service: HelpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or HelpService()
	return {
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"feedback": service.list_feedback(tenant_id),
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/help/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}
