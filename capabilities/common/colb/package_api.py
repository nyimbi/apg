"""Dependency-light API helpers for the COLB generated package."""

from __future__ import annotations

from typing import Any

from .collaboration_runtime import CollaborationRuntime


RUNTIME = CollaborationRuntime()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = RUNTIME.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**RUNTIME.dashboard_summary(tenant_id),
	}


def create_workspace(payload: dict[str, Any]) -> dict[str, Any]:
	return RUNTIME.create_workspace(
		tenant_id=str(payload.get("tenant_id") or "default"),
		workspace_id=str(payload["workspace_id"]),
		name=str(payload.get("name") or payload["workspace_id"]),
		owner=str(payload.get("owner") or ""),
		participants=[str(item) for item in payload.get("participants", [])],
		retention_policy=str(payload.get("retention_policy") or ""),
		external_participants=[str(item) for item in payload.get("external_participants", [])],
		external_policy_attached=bool(payload.get("external_policy_attached", True)),
		external_access_expiry_present=bool(payload.get("external_access_expiry_present", True)),
		membership_review_recorded=bool(payload.get("membership_review_recorded", True)),
	)


def approve_workspace(payload: dict[str, Any]) -> dict[str, Any]:
	return RUNTIME.approve_workspace(str(payload.get("tenant_id") or "default"), str(payload["workspace_id"]), str(payload.get("reviewer") or "reviewer"))


def start_session(payload: dict[str, Any]) -> dict[str, Any]:
	return RUNTIME.start_session(
		tenant_id=str(payload.get("tenant_id") or "default"),
		session_id=str(payload["session_id"]),
		workspace_id=str(payload["workspace_id"]),
		owner=str(payload.get("owner") or ""),
		protocol=str(payload.get("protocol") or "websocket"),
		secure_transport=bool(payload.get("secure_transport", True)),
		protocol_healthy=bool(payload.get("protocol_healthy", True)),
		recording_requested=bool(payload.get("recording_requested", False)),
		recording_retention_policy_attached=bool(payload.get("recording_retention_policy_attached", True)),
		event_bus_present=bool(payload.get("event_bus_present", True)),
	)


def join_session(payload: dict[str, Any]) -> dict[str, Any]:
	return RUNTIME.join_session(str(payload.get("tenant_id") or "default"), str(payload["session_id"]), str(payload["participant_id"]))


def share_artifact(payload: dict[str, Any]) -> dict[str, Any]:
	return RUNTIME.share_artifact(
		tenant_id=str(payload.get("tenant_id") or "default"),
		artifact_id=str(payload["artifact_id"]),
		workspace_id=str(payload["workspace_id"]),
		name=str(payload.get("name") or payload["artifact_id"]),
		owner=str(payload.get("owner") or ""),
		artifact_type=str(payload.get("artifact_type") or "document"),
		artifact_policy_attached=bool(payload.get("artifact_policy_attached", True)),
		version_history_enabled=bool(payload.get("version_history_enabled", True)),
		external_share=bool(payload.get("external_share", False)),
		dlp_check_completed=bool(payload.get("dlp_check_completed", True)),
	)


def add_annotation(payload: dict[str, Any]) -> dict[str, Any]:
	return RUNTIME.add_annotation(str(payload.get("tenant_id") or "default"), str(payload["annotation_id"]), str(payload["artifact_id"]), str(payload.get("author") or ""), str(payload.get("body") or ""))


def record_decision(payload: dict[str, Any]) -> dict[str, Any]:
	return RUNTIME.record_decision(str(payload.get("tenant_id") or "default"), str(payload["decision_id"]), str(payload["annotation_id"]), str(payload.get("owner") or ""), str(payload.get("decision") or ""), [str(item) for item in payload.get("evidence", [])])


def update_presence(payload: dict[str, Any]) -> dict[str, Any]:
	return RUNTIME.update_presence(str(payload.get("tenant_id") or "default"), str(payload["session_id"]), str(payload["participant_id"]), str(payload.get("status") or "online"), dict(payload.get("cursor") or {}))


def collaboration_state(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"summary": RUNTIME.dashboard_summary(tenant_id),
		"workspaces": RUNTIME.list_workspaces(tenant_id),
		"sessions": RUNTIME.list_sessions(tenant_id),
		"artifacts": RUNTIME.list_artifacts(tenant_id),
		"annotations": RUNTIME.list_annotations(tenant_id),
		"decisions": RUNTIME.list_decisions(tenant_id),
		"presence": RUNTIME.list_presence(tenant_id),
		"audit_events": RUNTIME.list_audit_events(tenant_id),
	}
