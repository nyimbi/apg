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
