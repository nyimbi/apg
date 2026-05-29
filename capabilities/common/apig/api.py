"""API helpers for the APG Intelligent Gateway capability."""

from __future__ import annotations

from typing import Any

from .gateway_runtime import ApigService


SERVICE = ApigService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**SERVICE.gateway_summary(tenant_id),
	}


def register_upstream(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_upstream(
		upstream_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		base_url=str(payload["base_url"]),
		owner=str(payload["owner"]),
		health=str(payload.get("health") or "healthy"),
		labels=dict(payload.get("labels") or {}),
	)


def request_route(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_route(
		route_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		path=str(payload["path"]),
		methods=[str(method) for method in payload.get("methods", ["GET"])],
		upstream_id=str(payload["upstream_id"]),
		owner=str(payload["owner"]),
		route_exposure=str(payload.get("route_exposure") or "internal"),
		auth_policy_attached=_payload_bool(payload, "auth_policy_attached", True),
		threat_policy_attached=_payload_bool(payload, "threat_policy_attached", True),
		requested_rps_limit=int(payload.get("requested_rps_limit") or 1000),
		wasm_filter_attached=_payload_bool(payload, "wasm_filter_attached", False),
		filter_signature_verified=_payload_bool(payload, "filter_signature_verified", True),
		justification=str(payload.get("justification") or ""),
	)


def decide_quota_review(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_quota_review(
		review_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload.get("notes") or ""),
	)


def activate_route(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.activate_route(
		route_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
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


def list_upstreams(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_upstreams(tenant_id)


def list_routes(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_routes(tenant_id)


def list_quota_reviews(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_quota_reviews(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)


def _payload_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
	value = payload.get(key, default)
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return bool(value)
