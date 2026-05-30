"""API helpers for the Help and Knowledge Base capability."""

from __future__ import annotations

from typing import Any

from .service import HelpService


SERVICE = HelpService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**summary,
	}


def register_source(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_source(
		source_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		title=str(payload["title"]),
		uri=str(payload["uri"]),
		owner_id=str(payload["owner_id"]),
		visibility=str(payload.get("visibility") or "internal"),
	)


def approve_source(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_source(
		source_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		approver_id=str(payload["approver_id"]),
	)


def create_article(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_article(
		article_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		title=str(payload["title"]),
		body=str(payload["body"]),
		owner_id=str(payload["owner_id"]),
		topics=list(payload.get("topics") or []),
		locale=str(payload.get("locale") or "en"),
		visibility=str(payload.get("visibility") or "internal"),
		source_ids=list(payload.get("source_ids") or []),
		source_approval_recorded=bool(payload.get("source_approval_recorded", True)),
	)


def publish_article(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_article(
		article_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		approver_id=str(payload["approver_id"]),
		publication_approved=bool(payload.get("publication_approved", True)),
		rbac_filter_applied=bool(payload.get("rbac_filter_applied", True)),
		freshness_review_recorded=bool(payload.get("freshness_review_recorded", True)),
		article_age_days=int(payload.get("article_age_days") or 0),
	)


def search_articles(payload: dict[str, Any]) -> list[dict[str, Any]]:
	return SERVICE.search_articles(
		tenant_id=str(payload.get("tenant_id") or "default"),
		query=str(payload["query"]),
		locale=payload.get("locale"),
		rbac_filter_applied=bool(payload.get("rbac_filter_applied", True)),
		include_restricted=bool(payload.get("include_restricted", False)),
		limit=int(payload.get("limit") or 5),
	)


def generate_answer(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.generate_answer(
		answer_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		query=str(payload["query"]),
		locale=payload.get("locale"),
		rbac_filter_applied=bool(payload.get("rbac_filter_applied", True)),
		include_restricted=bool(payload.get("include_restricted", False)),
		minimum_confidence=payload.get("minimum_confidence"),
	)


def record_feedback(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_feedback(
		feedback_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		user_id=str(payload["user_id"]),
		rating=int(payload["rating"]),
		comment=str(payload.get("comment") or ""),
		article_id=payload.get("article_id"),
		answer_id=payload.get("answer_id"),
		requires_review=payload.get("requires_review"),
	)


def localize_article(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.localize_article(
		localization_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		article_id=str(payload["article_id"]),
		locale=str(payload["locale"]),
		title=str(payload["title"]),
		body=str(payload["body"]),
		translator_id=str(payload["translator_id"]),
		source_locale=str(payload.get("source_locale") or "en"),
		fallback_locale=str(payload.get("fallback_locale") or "en"),
	)


def close_curation_item(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.close_curation_item(
		curation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer_id=str(payload["reviewer_id"]),
		evidence=[str(item) for item in payload.get("evidence", [])],
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


def help_state(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"sources": SERVICE.list_sources(tenant_id),
		"articles": SERVICE.list_articles(tenant_id),
		"answers": SERVICE.list_answers(tenant_id),
		"feedback": SERVICE.list_feedback(tenant_id),
		"localizations": SERVICE.list_localizations(tenant_id),
		"curation": SERVICE.list_curation_items(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}


def dashboard_summary(tenant_id: str | None = None) -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
