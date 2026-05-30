"""API helpers for the UI/UX Theming and Branding capability."""

from __future__ import annotations

from typing import Any

from .service import ThemService


SERVICE = ThemService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"theme_count": summary["theme_count"],
		"published_theme_count": summary["published_theme_count"],
		"review_required_theme_count": summary["review_required_theme_count"],
		"approved_asset_count": summary["approved_asset_count"],
		"them_agent_count": summary["them_agent_count"],
		"audit_event_count": summary["audit_event_count"],
		"streaming": summary["streaming"],
	}


def create_theme(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_theme(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner=str(payload.get("owner") or ""),
		brand_name=str(payload.get("brand_name") or ""),
		guidelines_ref=str(payload.get("guidelines_ref") or ""),
		fallback_theme_id=payload.get("fallback_theme_id"),
	)


def update_tokens(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.update_tokens(
		tenant_id=str(payload.get("tenant_id") or "default"),
		theme_id=str(payload["theme_id"]),
		group=str(payload.get("group") or "component"),
		tokens=dict(payload.get("tokens") or {}),
		updated_by=str(payload.get("updated_by") or ""),
		contrast_validated=bool(payload.get("contrast_validated", False)),
		reviewer=payload.get("reviewer"),
	)


def add_brand_asset(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.add_brand_asset(
		tenant_id=str(payload.get("tenant_id") or "default"),
		theme_id=str(payload["theme_id"]),
		asset_name=str(payload["asset_name"]),
		asset_type=str(payload.get("asset_type") or "image"),
		license_ref=str(payload.get("license_ref") or ""),
		approved_by=str(payload.get("approved_by") or ""),
	)


def create_preview(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_preview(
		tenant_id=str(payload.get("tenant_id") or "default"),
		theme_id=str(payload["theme_id"]),
		surface=str(payload.get("surface") or "app_shell"),
		viewport=str(payload.get("viewport") or "desktop"),
		preview_ref=str(payload.get("preview_ref") or ""),
		contrast_passed=bool(payload.get("contrast_passed", False)),
		created_by=str(payload.get("created_by") or ""),
	)


def publish_theme(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_theme(
		tenant_id=str(payload.get("tenant_id") or "default"),
		theme_id=str(payload["theme_id"]),
		published_by=str(payload.get("published_by") or ""),
		approval_ref=str(payload.get("approval_ref") or ""),
		target_tenant_count=int(payload.get("target_tenant_count", 1)),
		rollout_review_recorded=bool(payload.get("rollout_review_recorded", False)),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def register_them_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_them_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload.get("runtime") or ""),
		role=str(payload.get("role") or ""),
		scope=str(payload.get("scope") or ""),
		owner=str(payload.get("owner") or "platform"),
		human_approval_required=bool(payload.get("human_approval_required", True)),
	)


def validate_agent_theme_action(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_agent_theme_action(
		tenant_id=str(payload.get("tenant_id") or "default"),
		agent_id=str(payload["agent_id"]),
		action=str(payload.get("action") or "review"),
		privileged_scope=bool(payload.get("privileged_scope", False)),
		human_approval_ref=payload.get("human_approval_ref"),
	)


def validate_batch_theme_rollout(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_batch_theme_rollout(
		tenant_id=str(payload.get("tenant_id") or "default"),
		target_tenant_count=int(payload.get("target_tenant_count", 1)),
		event_stream=str(payload.get("event_stream") or "bytewax"),
		rollout_review_recorded=bool(payload.get("rollout_review_recorded", False)),
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


def list_theme_system(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"themes": SERVICE.list_themes(tenant_id),
		"tokens": SERVICE.list_tokens(tenant_id),
		"assets": SERVICE.list_assets(tenant_id),
		"previews": SERVICE.list_previews(tenant_id),
		"publications": SERVICE.list_publications(tenant_id),
		"them_agents": SERVICE.list_them_agents(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}
