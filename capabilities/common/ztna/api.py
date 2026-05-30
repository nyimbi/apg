"""API helpers for the Zero Trust Network Access capability."""

from __future__ import annotations

from typing import Any

from .service import ZtnaService


SERVICE = ZtnaService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"identity_count": summary["identity_count"],
		"device_count": summary["device_count"],
		"resource_count": summary["resource_count"],
		"access_request_count": summary["access_request_count"],
		"active_session_count": summary["active_session_count"],
	}


def register_identity(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_identity(
		identity_key=str(payload["identity_key"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		display_name=str(payload["display_name"]),
		verified=bool(payload.get("verified", False)),
		privileged=bool(payload.get("privileged", False)),
		mfa_completed=bool(payload.get("mfa_completed", False)),
		federated_provider=payload.get("federated_provider"),
		metadata=dict(payload.get("metadata") or {}),
	)


def verify_identity(identity_id: str, actor_id: str, mfa_completed: bool | None = None) -> dict[str, Any]:
	return SERVICE.verify_identity(identity_id, actor_id, mfa_completed)


def register_device(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_device(
		device_key=str(payload["device_key"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		identity_id=str(payload["identity_id"]),
		name=str(payload["name"]),
		trust_score=float(payload.get("trust_score", 0.0)),
		posture_present=bool(payload.get("posture_present", True)),
		managed=bool(payload.get("managed", False)),
		attested=bool(payload.get("attested", False)),
		compliant=bool(payload.get("compliant", True)),
		metadata=dict(payload.get("metadata") or {}),
	)


def update_device_posture(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.update_device_posture(
		device_id=str(payload["device_id"]),
		trust_score=float(payload.get("trust_score", 0.0)),
		posture_present=bool(payload.get("posture_present", True)),
		compliant=bool(payload.get("compliant", True)),
		attested=payload.get("attested"),
		actor_id=str(payload.get("actor_id") or "system"),
	)


def register_resource(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_resource(
		resource_key=str(payload["resource_key"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		access_level=str(payload.get("access_level") or "standard"),
		sensitive=bool(payload.get("sensitive", False)),
		policy_attached=bool(payload.get("policy_attached", False)),
		policy_id=payload.get("policy_id"),
		network_segment=str(payload.get("network_segment") or "default"),
		metadata=dict(payload.get("metadata") or {}),
	)


def attach_resource_policy(resource_id: str, policy_id: str, actor_id: str) -> dict[str, Any]:
	return SERVICE.attach_resource_policy(resource_id, policy_id, actor_id)


def request_access(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_access(
		identity_id=str(payload["identity_id"]),
		device_id=str(payload["device_id"]),
		resource_id=str(payload["resource_id"]),
		requested_by=str(payload["requested_by"]),
		mfa_completed=payload.get("mfa_completed"),
		access_review_recorded=bool(payload.get("access_review_recorded", False)),
		just_in_time_approval_present=bool(payload.get("just_in_time_approval_present", False)),
		least_privilege_scope_present=bool(payload.get("least_privilege_scope_present", True)),
		explicit_access_decision_present=bool(payload.get("explicit_access_decision_present", True)),
		access_risk_score=payload.get("access_risk_score"),
	)


def approve_access_request(request_id: str, reviewer_id: str) -> dict[str, Any]:
	return SERVICE.approve_access_request(request_id, reviewer_id)


def start_session(request_id: str, actor_id: str) -> dict[str, Any]:
	return SERVICE.start_session(request_id, actor_id)


def reevaluate_session(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.reevaluate_session(
		session_id=str(payload["session_id"]),
		risk_score=float(payload.get("risk_score", 0.0)),
		identity_verified=bool(payload.get("identity_verified", True)),
		device_posture_present=bool(payload.get("device_posture_present", True)),
		access_review_recorded=bool(payload.get("access_review_recorded", False)),
		actor_id=str(payload.get("actor_id") or "system"),
	)


def close_session(session_id: str, actor_id: str) -> dict[str, Any]:
	return SERVICE.close_session(session_id, actor_id)


def list_zero_trust_access(tenant_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
	return {
		"identities": SERVICE.list_identities(tenant_id),
		"devices": SERVICE.list_devices(tenant_id),
		"resources": SERVICE.list_resources(tenant_id),
		"access_requests": SERVICE.list_access_requests(tenant_id),
		"sessions": SERVICE.list_sessions(tenant_id),
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
