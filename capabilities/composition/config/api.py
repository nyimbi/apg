"""API helpers for the Central Configuration Management capability."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.requests import Request

from .service import CompositionConfigService

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _clean_text(value: Any) -> Optional[str]:
	"""Return a non-empty stripped string or None."""
	if value is None:
		return None
	text = str(value).strip()
	return text or None


async def verify_api_key(
	request: Request,
	api_key: Optional[str] = Security(api_key_header),
) -> Dict[str, Any]:
	"""Verify API key and resolve user/tenant context from headers, query, or environment."""
	if not api_key:
		raise HTTPException(status_code=401, detail="API key required")

	headers = getattr(request, "headers", {})
	query_params = getattr(request, "query_params", {})

	def _hget(*keys: str) -> Optional[str]:
		for k in keys:
			v = _clean_text(headers.get(k))
			if v:
				return v
		return None

	def _qget(*keys: str) -> Optional[str]:
		for k in keys:
			v = _clean_text(query_params.get(k))
			if v:
				return v
		return None

	user_id = (
		_hget("X-APG-User-ID", "X-User-ID")
		or _qget("user_id", "user")
		or os.getenv("APG_API_KEY_USER_ID", os.getenv("APG_DEFAULT_USER_ID", "system"))
	)
	tenant_id = (
		_hget("X-APG-Tenant-ID", "X-Tenant-ID")
		or _qget("tenant_id", "tenant")
		or os.getenv("APG_API_KEY_TENANT_ID", os.getenv("APG_DEFAULT_TENANT_ID", "default"))
	)

	return {"user_id": user_id, "tenant_id": tenant_id, "api_key": api_key}


# ==================== Dependency Injection


SERVICE = CompositionConfigService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"record_count": len(SERVICE.list_records(tenant_id)),
		"namespace_count": summary["namespace_count"],
		"configuration_count": summary["configuration_count"],
		"deployment_count": summary["deployment_count"],
		"config_agent_count": summary["config_agent_count"],
		"audit_event_count": summary["audit_event_count"],
		"streaming": summary["streaming"],
	}


def register_namespace(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_namespace(
		namespace_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		environment=str(payload.get("environment") or "development"),
		owner_id=str(payload["owner_id"]),
		path_prefix=str(payload.get("path_prefix") or "/default"),
		capability_id=str(payload.get("capability_id") or "composition_config"),
		metadata=dict(payload.get("metadata") or {}),
	)


def create_configuration(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_configuration(
		config_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		namespace_id=str(payload["namespace_id"]),
		key_path=str(payload["key_path"]),
		value=dict(payload.get("value") or {}),
		owner_id=str(payload["owner_id"]),
		restricted=bool(payload.get("restricted", False)),
		secret=bool(payload.get("secret", False)),
		schema=payload.get("schema"),
		secret_reference=payload.get("secret_reference"),
		policy_attached=bool(payload.get("policy_attached", True)),
		metadata=dict(payload.get("metadata") or {}),
	)


def validate_configuration(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_configuration(
		configuration_id=str(payload["configuration_id"]),
		actor_id=str(payload["actor_id"]),
		evidence=str(payload["evidence"]),
	)


def activate_configuration(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.activate_configuration(
		configuration_id=str(payload["configuration_id"]),
		actor_id=str(payload["actor_id"]),
		validation_evidence=payload.get("validation_evidence"),
	)


def deploy_configuration(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.deploy_configuration(
		deployment_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		configuration_id=str(payload["configuration_id"]),
		environment=str(payload.get("environment") or "development"),
		impact_level=str(payload.get("impact_level") or "standard"),
		actor_id=str(payload["actor_id"]),
		approved_by=payload.get("approved_by"),
		canary_evidence=payload.get("canary_evidence"),
		event_stream=str(payload.get("event_stream") or "bytewax"),
		metadata=dict(payload.get("metadata") or {}),
	)


def rollback_configuration(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.rollback_configuration(
		deployment_id=str(payload["deployment_id"]),
		actor_id=str(payload["actor_id"]),
		reason=str(payload.get("reason") or ""),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def create_template(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_template(
		template_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner_id=str(payload["owner_id"]),
		values=dict(payload.get("values") or {}),
		variable_schema=dict(payload.get("variable_schema") or {}),
		shared=bool(payload.get("shared", False)),
		reviewed_by=payload.get("reviewed_by"),
		metadata=dict(payload.get("metadata") or {}),
	)


def register_config_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_config_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		instructions=str(payload.get("instructions") or ""),
		metadata=dict(payload.get("metadata") or {}),
	)


def validate_agent_config_action(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_agent_config_action(
		tenant_id=str(payload.get("tenant_id") or "default"),
		agent_id=str(payload["agent_id"]),
		action=str(payload.get("action") or "review"),
		privileged_scope=bool(payload.get("privileged_scope", False)),
		human_approval_recorded=bool(payload.get("human_approval_recorded", False)),
	)


def validate_batch_configuration_change(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_batch_configuration_change(
		tenant_id=str(payload.get("tenant_id") or "default"),
		change_count=int(payload.get("change_count") or 0),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
		policy_attached=bool(payload.get("policy_attached", True)),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def capability_listing(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"namespaces": SERVICE.list_namespaces(tenant_id),
		"configurations": SERVICE.list_configurations(tenant_id),
		"deployments": SERVICE.list_deployments(tenant_id),
		"templates": SERVICE.list_templates(tenant_id),
		"drift": SERVICE.list_drift_records(tenant_id),
		"agents": SERVICE.list_config_agents(tenant_id),
		"audit_events": SERVICE.audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}
