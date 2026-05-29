"""UI metadata helpers for the Internationalization capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import I18nService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: I18nService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or I18nService()
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


def locale_console_model(
	service: I18nService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or I18nService()
	return {
		"tenant_id": tenant_id,
		"locales": service.list_locales(tenant_id),
		"default_locale": service.describe(tenant_id)["configuration"]["locales"]["default_locale"],
	}


def translation_workbench_model(
	service: I18nService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or I18nService()
	return {
		"tenant_id": tenant_id,
		"translations": service.list_translations(tenant_id),
		"glossary_terms": service.list_glossary_terms(tenant_id),
		"requires_machine_translation_review": True,
	}


def glossary_manager_model(
	service: I18nService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or I18nService()
	return {
		"tenant_id": tenant_id,
		"glossary_terms": service.list_glossary_terms(tenant_id),
	}


def coverage_dashboard_model(
	service: I18nService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or I18nService()
	return {
		"tenant_id": tenant_id,
		"coverage_reports": service.list_coverage_reports(tenant_id),
		"minimum_coverage_percent": service.describe(tenant_id)["configuration"]["translations"]["minimum_coverage_percent"],
	}


def publish_queue_model(
	service: I18nService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or I18nService()
	return {
		"tenant_id": tenant_id,
		"publish_batches": service.list_publish_batches(tenant_id),
		"approval_required": True,
	}
