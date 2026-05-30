"""API helpers for the User Management capability."""

from __future__ import annotations

from typing import Any

from .service import UsrmService


SERVICE = UsrmService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"user_count": summary["user_count"],
		"active_user_count": summary["active_user_count"],
		"privileged_user_count": summary["privileged_user_count"],
		"access_review_count": summary["access_review_count"],
		"deprovision_count": summary["deprovision_count"],
		"usrm_agent_count": summary["usrm_agent_count"],
		"audit_event_count": summary["audit_event_count"],
		"streaming": summary["streaming"],
	}


def create_user(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_user(
		tenant_id=str(payload.get("tenant_id") or "default"),
		identity=str(payload["identity"]),
		display_name=str(payload.get("display_name") or payload["identity"]),
		email=str(payload["email"]),
		owner=str(payload.get("owner") or ""),
		profile_validated=bool(payload.get("profile_validated", True)),
		privileged_user=bool(payload.get("privileged_user", False)),
		mfa_enabled=bool(payload.get("mfa_enabled", False)),
		manager_id=payload.get("manager_id"),
	)


def update_profile(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.update_profile(
		tenant_id=str(payload.get("tenant_id") or "default"),
		user_id=str(payload["user_id"]),
		attributes=dict(payload.get("attributes") or {}),
		privacy_preferences=dict(payload.get("privacy_preferences") or {}),
		consent_notice_ref=str(payload.get("consent_notice_ref") or ""),
		updated_by=str(payload.get("updated_by") or ""),
		privacy_sync_recorded=bool(payload.get("privacy_sync_recorded", True)),
	)


def invite_user(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.invite_user(
		tenant_id=str(payload.get("tenant_id") or "default"),
		user_id=str(payload["user_id"]),
		channel=str(payload.get("channel") or "email"),
		consent_notice_ref=str(payload.get("consent_notice_ref") or ""),
		invited_by=str(payload.get("invited_by") or ""),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def assign_role(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.assign_role(
		tenant_id=str(payload.get("tenant_id") or "default"),
		user_id=str(payload["user_id"]),
		role=str(payload["role"]),
		scope=str(payload.get("scope") or "tenant"),
		privileged=bool(payload.get("privileged", False)),
		mfa_enabled=bool(payload.get("mfa_enabled", False)),
		approved_by=str(payload.get("approved_by") or ""),
	)


def record_access_review(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_access_review(
		tenant_id=str(payload.get("tenant_id") or "default"),
		user_id=str(payload["user_id"]),
		reviewer=str(payload.get("reviewer") or ""),
		decision=str(payload.get("decision") or "defer"),
		findings=list(payload.get("findings") or []),
	)


def deprovision_user(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.deprovision_user(
		tenant_id=str(payload.get("tenant_id") or "default"),
		user_id=str(payload["user_id"]),
		actor=str(payload.get("actor") or ""),
		access_revoked=bool(payload.get("access_revoked", False)),
		evidence_ref=str(payload.get("evidence_ref") or ""),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def bulk_suspend_users(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.bulk_suspend_users(
		tenant_id=str(payload.get("tenant_id") or "default"),
		user_ids=[str(item) for item in list(payload.get("user_ids") or [])],
		actor=str(payload.get("actor") or ""),
		bulk_review_recorded=bool(payload.get("bulk_review_recorded", False)),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def register_usrm_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_usrm_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload.get("runtime") or ""),
		role=str(payload.get("role") or ""),
		scope=str(payload.get("scope") or ""),
		owner=str(payload.get("owner") or "platform"),
		human_approval_required=bool(payload.get("human_approval_required", True)),
	)


def validate_agent_user_action(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_agent_user_action(
		tenant_id=str(payload.get("tenant_id") or "default"),
		agent_id=str(payload["agent_id"]),
		action=str(payload.get("action") or "review"),
		privileged_scope=bool(payload.get("privileged_scope", False)),
		human_approval_ref=payload.get("human_approval_ref"),
	)


def validate_batch_user_lifecycle(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_batch_user_lifecycle(
		tenant_id=str(payload.get("tenant_id") or "default"),
		affected_user_count=int(payload.get("affected_user_count", 0)),
		event_stream=str(payload.get("event_stream") or "bytewax"),
		bulk_review_recorded=bool(payload.get("bulk_review_recorded", False)),
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


def list_user_management(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"users": SERVICE.list_users(tenant_id),
		"profiles": SERVICE.list_profiles(tenant_id),
		"invitations": SERVICE.list_invitations(tenant_id),
		"role_assignments": SERVICE.list_role_assignments(tenant_id),
		"access_reviews": SERVICE.list_access_reviews(tenant_id),
		"deprovisions": SERVICE.list_deprovisions(tenant_id),
		"bulk_actions": SERVICE.list_bulk_actions(tenant_id),
		"usrm_agents": SERVICE.list_usrm_agents(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}
