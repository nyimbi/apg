"""API helpers for the Access Control Integration Hub capability."""

from __future__ import annotations

from typing import Any

from .service import CompositionAccessService


SERVICE = CompositionAccessService()


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
		"provider_count": summary["provider_count"],
		"resource_count": summary["resource_count"],
		"policy_count": summary["policy_count"],
		"grant_count": summary["grant_count"],
		"access_agent_count": summary["access_agent_count"],
		"audit_event_count": summary["audit_event_count"],
		"streaming": summary["streaming"],
	}


def register_provider(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_provider(
		provider_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		provider_type=str(payload.get("provider_type") or "oidc"),
		owner_id=str(payload["owner_id"]),
		external=bool(payload.get("external", True)),
		metadata_validated=bool(payload.get("metadata_validated", False)),
		secret_reference=payload.get("secret_reference"),
		test_evidence=payload.get("test_evidence"),
		metadata=dict(payload.get("metadata") or {}),
	)


def activate_provider(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.activate_provider(
		provider_id=str(payload["provider_id"]),
		actor_id=str(payload["actor_id"]),
		metadata_validated=bool(payload.get("metadata_validated", True)),
		secret_reference=payload.get("secret_reference"),
		test_evidence=payload.get("test_evidence"),
	)


def register_resource(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_resource(
		resource_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		display_name=str(payload.get("display_name") or payload["id"]),
		owner_id=str(payload["owner_id"]),
		scopes=list(payload.get("scopes") or ["read"]),
		capability_id=str(payload.get("capability_id") or "composition_access"),
		sensitive=bool(payload.get("sensitive", False)),
		metadata=dict(payload.get("metadata") or {}),
	)


def create_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_policy(
		policy_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		resource_id=str(payload["resource_id"]),
		owner_id=str(payload["owner_id"]),
		effect=str(payload.get("effect") or "allow"),
		conditions=dict(payload.get("conditions") or {}),
		risk_level=str(payload.get("risk_level") or "standard"),
		metadata=dict(payload.get("metadata") or {}),
	)


def activate_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.activate_policy(
		policy_id=str(payload["policy_id"]),
		actor_id=str(payload["actor_id"]),
		simulation_evidence=payload.get("simulation_evidence"),
		reviewed_by=payload.get("reviewed_by"),
	)


def create_grant(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_grant(
		grant_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		resource_id=str(payload["resource_id"]),
		scopes=list(payload.get("scopes") or ["read"]),
		requested_by=str(payload["requested_by"]),
		justification=str(payload.get("justification") or ""),
		privileged=bool(payload.get("privileged", False)),
		approved_by=payload.get("approved_by"),
		expires_at=payload.get("expires_at"),
		metadata=dict(payload.get("metadata") or {}),
	)


def evaluate_session(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.evaluate_session(
		session_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		provider_id=str(payload["provider_id"]),
		risk_score=int(payload.get("risk_score") or 0),
		step_up_completed=bool(payload.get("step_up_completed", False)),
		metadata=dict(payload.get("metadata") or {}),
	)


def record_decision(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_decision(
		decision_key=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		resource_id=str(payload["resource_id"]),
		action=str(payload.get("action") or "read"),
		decision=str(payload.get("decision") or "allow"),
		reason=str(payload.get("reason") or "policy_match"),
		policy_ids=list(payload.get("policy_ids") or []),
		event_stream=str(payload.get("event_stream") or "bytewax"),
		metadata=dict(payload.get("metadata") or {}),
	)


def register_access_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_access_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		instructions=str(payload.get("instructions") or ""),
		metadata=dict(payload.get("metadata") or {}),
	)


def validate_agent_access_action(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_agent_access_action(
		tenant_id=str(payload.get("tenant_id") or "default"),
		agent_id=str(payload["agent_id"]),
		action=str(payload.get("action") or "review"),
		privileged_scope=bool(payload.get("privileged_scope", False)),
		human_approval_recorded=bool(payload.get("human_approval_recorded", False)),
	)


def validate_batch_grant(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_batch_grant(
		tenant_id=str(payload.get("tenant_id") or "default"),
		grant_count=int(payload.get("grant_count") or 0),
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
		"providers": SERVICE.list_providers(tenant_id),
		"resources": SERVICE.list_resources(tenant_id),
		"policies": SERVICE.list_policies(tenant_id),
		"grants": SERVICE.list_grants(tenant_id),
		"sessions": SERVICE.list_sessions(tenant_id),
		"decisions": SERVICE.list_decisions(tenant_id),
		"agents": SERVICE.list_access_agents(tenant_id),
		"audit_events": SERVICE.audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}
