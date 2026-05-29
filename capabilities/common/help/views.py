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
