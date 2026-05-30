"""API helpers for the No-Code/Low-Code Builder capability."""

from __future__ import annotations

from typing import Any

from .service import NcodService


SERVICE = NcodService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"app_count": summary["app_count"],
		"published_app_count": summary["published_app_count"],
		"release_count": summary["release_count"],
		"deployment_count": summary["deployment_count"],
		"builder_agent_count": summary["builder_agent_count"],
	}


def create_app(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_app(
		app_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner=str(payload.get("owner") or ""),
		description=str(payload.get("description") or ""),
		theme=str(payload.get("theme") or "ncod_app_builder"),
		rbac_policy_ref=str(payload.get("rbac_policy_ref") or ""),
		data_residency_policy_ref=str(payload.get("data_residency_policy_ref") or ""),
		accessibility_checked=bool(payload.get("accessibility_checked")),
		metadata=dict(payload.get("metadata") or {}),
	)


def add_page(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.add_page(
		page_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		app_id=str(payload["app_id"]),
		name=str(payload["name"]),
		route=str(payload["route"]),
		layout=str(payload.get("layout") or "responsive_grid"),
		metadata=dict(payload.get("metadata") or {}),
	)


def add_component(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.add_component(
		component_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		page_id=str(payload["page_id"]),
		component_type=str(payload.get("component_type") or "text"),
		name=str(payload["name"]),
		props=dict(payload.get("props") or {}),
		bindings=dict(payload.get("bindings") or {}),
		accessibility_label=str(payload.get("accessibility_label") or ""),
		order=int(payload.get("order") or 0),
	)


def bind_data_source(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.bind_data_source(
		binding_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		app_id=str(payload["app_id"]),
		name=str(payload["name"]),
		source_type=str(payload.get("source_type") or "entity"),
		source_ref=str(payload["source_ref"]),
		schema=dict(payload.get("schema") or {}),
		policy_ref=str(payload.get("policy_ref") or ""),
	)


def define_data_model(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.define_data_model(
		model_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		app_id=str(payload["app_id"]),
		name=str(payload["name"]),
		fields=list(payload.get("fields") or []),
		policy_ref=str(payload.get("policy_ref") or ""),
		metadata=dict(payload.get("metadata") or {}),
	)


def attach_workflow(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.attach_workflow(
		binding_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		app_id=str(payload["app_id"]),
		trigger=str(payload["trigger"]),
		workflow_ref=str(payload["workflow_ref"]),
		policy_ref=str(payload.get("policy_ref") or ""),
		enabled=bool(payload.get("enabled", True)),
		metadata=dict(payload.get("metadata") or {}),
	)


def create_theme_variant(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_theme_variant(
		theme_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		app_id=str(payload["app_id"]),
		name=str(payload["name"]),
		tokens=dict(payload.get("tokens") or {}),
		policy_ref=str(payload.get("policy_ref") or ""),
		approved=bool(payload.get("approved")),
	)


def add_script_extension(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.add_script_extension(
		extension_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		app_id=str(payload["app_id"]),
		name=str(payload["name"]),
		hook=str(payload["hook"]),
		script_ref=str(payload["script_ref"]),
		policy_ref=str(payload.get("policy_ref") or ""),
	)


def add_connector_binding(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.add_connector_binding(
		binding_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		app_id=str(payload["app_id"]),
		name=str(payload["name"]),
		connector_ref=str(payload["connector_ref"]),
		policy_ref=str(payload.get("policy_ref") or ""),
		scopes=list(payload.get("scopes") or []),
	)


def register_builder_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_builder_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		app_id=str(payload["app_id"]),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload.get("scope") or ""),
		contribution_disclosed=bool(payload.get("contribution_disclosed")),
		policy_ref=str(payload.get("policy_ref") or ""),
		registered=bool(payload.get("registered", True)),
	)


def validate_app(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_app(
		validation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		app_id=str(payload["app_id"]),
	)


def deploy_release(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.deploy_release(
		deployment_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		release_id=str(payload["release_id"]),
		target_runtime=str(payload.get("target_runtime") or "python"),
		target_ref=str(payload.get("target_ref") or ""),
		approval_recorded=bool(payload.get("approval_recorded")),
		rollback_plan_ref=str(payload.get("rollback_plan_ref") or ""),
		approval_ref=str(payload.get("approval_ref") or ""),
	)


def change_app_state(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.change_app_state(
		tenant_id=str(payload.get("tenant_id") or "default"),
		app_id=str(payload["app_id"]),
		status=str(payload["status"]),
		reason=str(payload.get("reason") or ""),
		audit_recorded=bool(payload.get("audit_recorded", True)),
	)


def publish_app(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_app(
		release_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		app_id=str(payload["app_id"]),
		target_environment=str(payload.get("target_environment") or "staging"),
		approval_recorded=bool(payload.get("approval_recorded")),
		approval_ref=str(payload.get("approval_ref") or ""),
		change_review_recorded=bool(payload.get("change_review_recorded")),
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


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
