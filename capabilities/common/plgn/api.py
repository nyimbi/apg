"""Dependency-light API helpers for Plugin/Extension Framework."""

from __future__ import annotations

from typing import Any

from .service import PlgnService


SERVICE = PlgnService()


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


def register_plugin(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_plugin(
		plugin_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload.get("owner") or ""),
		version=str(payload.get("version") or "0.1.0"),
		publisher=str(payload.get("publisher") or "tenant"),
		release_channel=str(payload.get("release_channel") or "stable"),
		permissions=list(payload.get("permissions") or []),
		dependencies=list(payload.get("dependencies") or []),
		external_plugin=bool(payload.get("external_plugin", False)),
		signature_verified=bool(payload.get("signature_verified", True)),
		manifest_schema_valid=bool(payload.get("manifest_schema_valid", True)),
		dependency_validation_passed=bool(payload.get("dependency_validation_passed", True)),
		supply_chain_scan_passed=bool(payload.get("supply_chain_scan_passed", True)),
		external_review_recorded=bool(payload.get("external_review_recorded", False)),
		permission_review_recorded=bool(payload.get("permission_review_recorded", False)),
		metadata=dict(payload.get("metadata") or {}),
	)


def review_permissions(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.review_permissions(
		review_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		plugin_id=str(payload["plugin_id"]),
		reviewer=str(payload.get("reviewer") or ""),
		approved_scopes=list(payload.get("approved_scopes") or []),
		denied_scopes=list(payload.get("denied_scopes") or []),
		secret_access_allowed=bool(payload.get("secret_access_allowed", False)),
		notes=str(payload.get("notes") or ""),
	)


def attach_sandbox_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.attach_sandbox_policy(
		policy_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		plugin_id=str(payload["plugin_id"]),
		policy_name=str(payload.get("policy_name") or payload["id"]),
		network_access=str(payload.get("network_access") or "deny"),
		filesystem_access=str(payload.get("filesystem_access") or "read_only"),
		secret_access=str(payload.get("secret_access") or "deny"),
		tool_allowlist=list(payload.get("tool_allowlist") or []),
	)


def publish_listing(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_listing(
		listing_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		plugin_id=str(payload["plugin_id"]),
		title=str(payload.get("title") or payload["plugin_id"]),
		publisher_verified=bool(payload.get("publisher_verified", True)),
		curated=bool(payload.get("curated", True)),
		install_policy=str(payload.get("install_policy") or "tenant_allowed"),
	)


def create_release(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_release(
		release_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		plugin_id=str(payload["plugin_id"]),
		version=str(payload.get("version") or "0.1.0"),
		channel=str(payload.get("channel") or "stable"),
		signature_ref=str(payload.get("signature_ref") or ""),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def install_plugin(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.install_plugin(
		installation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		plugin_id=str(payload["plugin_id"]),
		installed_by=str(payload.get("installed_by") or "tenant-admin"),
	)


def enable_plugin(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.enable_plugin(
		installation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		actor=str(payload.get("actor") or "tenant-admin"),
	)


def register_plgn_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_plgn_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload["scope"]),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		agent_id=str(payload["id"]) if payload.get("id") else None,
	)


def validate_batch_plugin_mutation(event_stream: str) -> dict[str, Any]:
	return SERVICE.validate_batch_plugin_mutation(event_stream)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def list_plgn_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_plgn_agents(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
