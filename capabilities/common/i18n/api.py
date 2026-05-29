"""API helpers for the Internationalization capability."""

from __future__ import annotations

from typing import Any

from .service import I18nService


SERVICE = I18nService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**SERVICE.dashboard_summary(tenant_id),
	}


def create_locale(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_locale(
		locale_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		locale_code=str(payload["locale_code"]),
		display_name=str(payload["display_name"]),
		owner_id=str(payload["owner_id"]),
		fallback_locale=payload.get("fallback_locale"),
		regional_format=dict(payload.get("regional_format") or {}),
		timezone=str(payload.get("timezone") or "UTC"),
	)


def add_glossary_term(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.add_glossary_term(
		term_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		source_term=str(payload["source_term"]),
		localized_terms=dict(payload.get("localized_terms") or {}),
		description=str(payload.get("description") or ""),
		owner_id=str(payload.get("owner_id") or ""),
	)


def upsert_translation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.upsert_translation(
		translation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		key=str(payload["key"]),
		locale_code=str(payload["locale_code"]),
		source_text=str(payload["source_text"]),
		translated_text=str(payload["translated_text"]),
		machine_translation_used=bool(payload.get("machine_translation_used", False)),
		translation_review_recorded=bool(payload.get("translation_review_recorded", True)),
		reviewer_id=payload.get("reviewer_id"),
		restricted=bool(payload.get("restricted", False)),
		rbac_filter_applied=bool(payload.get("rbac_filter_applied", True)),
	)


def publish_translations(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_translations(
		batch_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		locale_code=str(payload["locale_code"]),
		translation_ids=list(payload.get("translation_ids") or []),
		approver_id=str(payload["approver_id"]),
		approval_recorded=bool(payload.get("approval_recorded", True)),
		coverage_review_recorded=bool(payload.get("coverage_review_recorded", True)),
	)


def resolve_text(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.resolve_text(
		tenant_id=str(payload.get("tenant_id") or "default"),
		key=str(payload["key"]),
		locale_code=str(payload["locale_code"]),
		default_locale=payload.get("default_locale"),
	)


def coverage_report(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.coverage_report(
		report_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		locale_code=str(payload["locale_code"]),
		required_keys=list(payload.get("required_keys") or []),
		coverage_review_recorded=bool(payload.get("coverage_review_recorded", True)),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def dashboard_summary(tenant_id: str | None = None) -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
