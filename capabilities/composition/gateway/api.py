"""API helpers for the API Service Mesh capability."""

from __future__ import annotations

from typing import Any

from .service import CompositionGatewayService


SERVICE = CompositionGatewayService()


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
		"service_count": summary["service_count"],
		"mesh_route_count": summary["route_count"],
		"policy_count": summary["policy_count"],
		"gateway_agent_count": summary["gateway_agent_count"],
		"audit_event_count": summary["audit_event_count"],
		"streaming": summary["streaming"],
	}


def register_service(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_service(
		service_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner_id=str(payload["owner_id"]),
		endpoints=list(payload.get("endpoints") or []),
		health_check_path=str(payload.get("health_check_path") or ""),
		capability_id=str(payload.get("capability_id") or "composition_gateway"),
		public_service=bool(payload.get("public_service", False)),
		metadata=dict(payload.get("metadata") or {}),
	)


def create_route(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_route(
		route_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		service_id=str(payload["service_id"]),
		path=str(payload["path"]),
		methods=list(payload.get("methods") or ["GET"]),
		public_route=bool(payload.get("public_route", False)),
		policy_id=payload.get("policy_id"),
		approved_by=payload.get("approved_by"),
		tls_enabled=bool(payload.get("tls_enabled", False)),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def attach_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.attach_policy(
		policy_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		service_id=str(payload["service_id"]),
		rate_limit_configured=bool(payload.get("rate_limit_configured", False)),
		circuit_breaker_configured=bool(payload.get("circuit_breaker_configured", False)),
		owner_id=str(payload["owner_id"]),
	)


def shift_traffic(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.shift_traffic(
		shift_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		route_id=str(payload["route_id"]),
		weights=dict(payload.get("weights") or {}),
		actor_id=str(payload["actor_id"]),
		canary_shift=bool(payload.get("canary_shift", False)),
		canary_evidence=payload.get("canary_evidence"),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def register_certificate(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_certificate(
		certificate_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		domain=str(payload["domain"]),
		owner_id=str(payload["owner_id"]),
		secret_reference=str(payload.get("secret_reference") or ""),
		expires_at=str(payload["expires_at"]),
	)


def register_gateway_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_gateway_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		instructions=str(payload.get("instructions") or ""),
	)


def validate_agent_gateway_action(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_agent_gateway_action(
		tenant_id=str(payload.get("tenant_id") or "default"),
		agent_id=str(payload["agent_id"]),
		action=str(payload.get("action") or "review"),
		privileged_scope=bool(payload.get("privileged_scope", False)),
		human_approval_recorded=bool(payload.get("human_approval_recorded", False)),
	)


def validate_batch_route_change(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_batch_route_change(
		tenant_id=str(payload.get("tenant_id") or "default"),
		route_count=int(payload.get("route_count") or 0),
		event_stream=str(payload.get("event_stream") or "bytewax"),
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


def capability_listing(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"services": SERVICE.list_services(tenant_id),
		"routes": SERVICE.list_routes(tenant_id),
		"policies": SERVICE.list_policies(tenant_id),
		"certificates": SERVICE.list_certificates(tenant_id),
		"traffic_shifts": SERVICE.list_traffic_shifts(tenant_id),
		"agents": SERVICE.list_gateway_agents(tenant_id),
		"audit_events": SERVICE.audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}
