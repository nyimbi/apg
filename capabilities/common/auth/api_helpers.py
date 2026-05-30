"""Dependency-light AUTH API helpers for generated APG applications."""

from __future__ import annotations

from typing import Any

from .service import AuthService


SERVICE = AuthService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"streaming": contract["streaming"],
		**SERVICE.dashboard_summary(tenant_id),
	}


def register_identity(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_identity(
		user_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		email=str(payload["email"]),
		display_name=str(payload["display_name"]),
		status=str(payload.get("status") or "active"),
		tenant_memberships=[str(item) for item in payload.get("tenant_memberships", [])],
		mfa_enabled=_payload_bool(payload, "mfa_enabled", False),
		behavioral_trust_score=float(payload.get("behavioral_trust_score", 1.0)),
		biometric_enrolled=_payload_bool(payload, "biometric_enrolled", False),
		quantum_key_registered=_payload_bool(payload, "quantum_key_registered", False),
		privacy_budget=float(payload.get("privacy_budget", 1.0)),
		metadata=dict(payload.get("metadata") or {}),
	)


def define_role(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.define_role(
		role_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		permissions=[str(item) for item in payload["permissions"]],
		tier=str(payload.get("tier") or "standard"),
		approval_recorded=_payload_bool(payload, "approval_recorded", False),
	)


def request_role_assignment_approval(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_role_assignment_approval(
		approval_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		user_id=str(payload["user_id"]),
		role_id=str(payload["role_id"]),
		requested_by=str(payload["requested_by"]),
		justification=str(payload["justification"]),
	)


def decide_role_assignment_approval(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_role_assignment_approval(
		approval_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload["notes"]),
	)


def request_privacy_budget_approval(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_privacy_budget_approval(
		approval_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		user_id=str(payload["user_id"]),
		query_type=str(payload["query_type"]),
		epsilon_cost=float(payload["epsilon_cost"]),
		requested_by=str(payload["requested_by"]),
		justification=str(payload["justification"]),
	)


def decide_privacy_budget_approval(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_privacy_budget_approval(
		approval_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload["notes"]),
	)


def assign_role(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.assign_role(
		assignment_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		user_id=str(payload["user_id"]),
		role_id=str(payload["role_id"]),
		assigned_by=str(payload["assigned_by"]),
		approval_recorded=_payload_bool(payload, "approval_recorded", False),
		approval_id=str(payload["approval_id"]) if payload.get("approval_id") else None,
	)


def start_session(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.start_session(
		session_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		user_id=str(payload["user_id"]),
		device_id=str(payload["device_id"]),
		auth_source=str(payload.get("auth_source") or "local"),
		issuer_trusted=_payload_bool(payload, "issuer_trusted", True),
		mfa_verified=_payload_bool(payload, "mfa_verified", False),
		risk_level=str(payload.get("risk_level") or "low"),
		step_up_completed=_payload_bool(payload, "step_up_completed", False),
	)


def evaluate_access(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.evaluate_access(
		decision_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		user_id=str(payload["user_id"]),
		permission=str(payload["permission"]),
		session_id=str(payload["session_id"]) if payload.get("session_id") else None,
		requested_permission_tier=str(payload["requested_permission_tier"]) if payload.get("requested_permission_tier") else None,
		mfa_verified=_optional_bool(payload, "mfa_verified"),
		step_up_completed=_optional_bool(payload, "step_up_completed"),
		risk_level=str(payload["risk_level"]) if payload.get("risk_level") else None,
	)


def run_privacy_query(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.run_privacy_query(
		query_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		user_id=str(payload["user_id"]),
		query_type=str(payload["query_type"]),
		epsilon_cost=float(payload["epsilon_cost"]),
		approval_recorded=_payload_bool(payload, "approval_recorded", False),
		approval_id=str(payload["approval_id"]) if payload.get("approval_id") else None,
	)


def revoke_session(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.revoke_session(
		session_id=str(payload["id"]),
		actor=str(payload["actor"]),
		tenant_id=str(payload["tenant_id"]) if payload.get("tenant_id") else None,
	)


def register_security_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_security_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		runtime=str(payload.get("runtime") or "codex"),
		role=str(payload.get("role") or "identity_reviewer"),
		scope=str(payload.get("scope") or ""),
		registered=_payload_bool(payload, "registered", True),
		contribution_disclosed=_payload_bool(payload, "contribution_disclosed", True),
		policy_ref=str(payload["policy_ref"]) if payload.get("policy_ref") else None,
		status=str(payload.get("status") or "active"),
	)


def validate_batch_auth_mutation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_batch_auth_mutation(
		tenant_id=str(payload.get("tenant_id") or "default"),
		event_stream=str(payload.get("event_stream") or "bytewax"),
		mutation_count=int(payload.get("mutation_count") or 0),
	)


def list_identities(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_identities(tenant_id)


def list_roles(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_roles(tenant_id)


def list_role_assignment_approvals(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_role_assignment_approvals(tenant_id)


def list_privacy_budget_approvals(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_privacy_budget_approvals(tenant_id)


def list_role_assignments(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_role_assignments(tenant_id)


def list_sessions(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_sessions(tenant_id)


def list_access_decisions(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_access_decisions(tenant_id)


def list_privacy_queries(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_privacy_queries(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)


def list_security_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_security_agents(tenant_id)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def _payload_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
	value = payload.get(key, default)
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return bool(value)


def _optional_bool(payload: dict[str, Any], key: str) -> bool | None:
	if key not in payload:
		return None
	return _payload_bool(payload, key, False)
