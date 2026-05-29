"""API helpers for the Website Builder capability."""

from __future__ import annotations

from typing import Any

from .service import WsblService


SERVICE = WsblService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"site_count": summary["site_count"],
		"page_count": summary["page_count"],
		"component_count": summary["component_count"],
		"publish_request_count": summary["publish_request_count"],
	}


def create_site(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_site(
		site_key=str(payload["site_key"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner_id=str(payload["owner_id"]),
		primary_domain=payload.get("primary_domain"),
		locale=str(payload.get("locale") or "en"),
		public_site=bool(payload.get("public_site", True)),
		privacy_banner_required=bool(payload.get("privacy_banner_required", True)),
		domain_validated=bool(payload.get("domain_validated", False)),
		metadata=dict(payload.get("metadata") or {}),
	)


def validate_domain(domain_id: str, actor_id: str) -> dict[str, Any]:
	return SERVICE.validate_domain(domain_id, actor_id)


def create_component(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_component(
		component_key=str(payload["component_key"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		component_type=str(payload.get("component_type") or "section"),
		custom=bool(payload.get("custom", False)),
		reviewed=bool(payload.get("reviewed", False)),
		reviewed_by=payload.get("reviewed_by"),
		policy_id=payload.get("policy_id"),
		metadata=dict(payload.get("metadata") or {}),
	)


def review_component(component_id: str, reviewer_id: str, policy_id: str | None = None) -> dict[str, Any]:
	return SERVICE.review_component(component_id, reviewer_id, policy_id)


def create_page(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_page(
		site_id=str(payload["site_id"]),
		slug=str(payload["slug"]),
		title=str(payload["title"]),
		tenant_id=payload.get("tenant_id"),
		metadata=dict(payload.get("metadata") or {}),
	)


def add_page_section(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.add_page_section(
		page_id=str(payload["page_id"]),
		component_id=str(payload["component_id"]),
		content=dict(payload.get("content") or {}),
		position=payload.get("position"),
		actor_id=str(payload.get("actor_id") or "system"),
	)


def create_publish_request(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_publish_request(
		site_id=str(payload["site_id"]),
		requested_by=str(payload["requested_by"]),
		environment=str(payload.get("environment") or "production"),
		approval_recorded=bool(payload.get("approval_recorded", False)),
		accessibility_passed=bool(payload.get("accessibility_passed", False)),
		consent_policy_attached=bool(payload.get("consent_policy_attached", False)),
	)


def publish_site(publish_request_id: str, actor_id: str) -> dict[str, Any]:
	return SERVICE.publish_site(publish_request_id, actor_id)


def rollback_site(site_id: str, version: int, actor_id: str) -> dict[str, Any]:
	return SERVICE.rollback_site(site_id, version, actor_id)


def list_website_builder(tenant_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
	return {
		"sites": SERVICE.list_sites(tenant_id),
		"domains": SERVICE.list_domains(tenant_id),
		"pages": SERVICE.list_pages(tenant_id),
		"components": SERVICE.list_components(tenant_id),
		"publish_requests": SERVICE.list_publish_requests(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
	}


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)
