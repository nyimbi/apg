"""API helpers for the APG Pose Estimation capability."""

from __future__ import annotations

from typing import Any

from .service import PoseService


SERVICE = PoseService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"model_count": summary["model_count"],
		"session_count": summary["session_count"],
		"estimate_count": summary["estimate_count"],
		"agent_count": summary["agent_count"],
	}


def register_model(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_model(
		model_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		model_type=str(payload.get("model_type") or "rtmpose"),
		owner=str(payload.get("owner") or ""),
		policy_ref=str(payload.get("policy_ref") or ""),
		minimum_keypoint_confidence=float(payload.get("minimum_keypoint_confidence", 0.72)),
		edge_ready=bool(payload.get("edge_ready")),
	)


def start_session(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.start_session(
		session_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload.get("owner") or ""),
		source_ref=str(payload.get("source_ref") or ""),
		model_id=str(payload["model_id"]),
		subject_consent_recorded=bool(payload.get("subject_consent_recorded")),
		secure_stream=bool(payload.get("secure_stream")),
		realtime_stream=bool(payload.get("realtime_stream")),
		sensitive_use=bool(payload.get("sensitive_use")),
		approval_ref=str(payload.get("approval_ref") or ""),
		max_persons=int(payload.get("max_persons") or 1),
		metadata=dict(payload.get("metadata") or {}),
	)


def record_frame(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_frame(
		frame_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		session_id=str(payload["session_id"]),
		frame_number=int(payload.get("frame_number") or 0),
		occurred_at=str(payload.get("occurred_at") or ""),
		source_ref=str(payload.get("source_ref") or ""),
		width=int(payload.get("width") or 0),
		height=int(payload.get("height") or 0),
	)


def estimate_pose(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.estimate_pose(
		estimate_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		session_id=str(payload["session_id"]),
		frame_id=str(payload["frame_id"]),
		model_id=str(payload["model_id"]),
		keypoints=list(payload.get("keypoints") or []),
		person_count=int(payload.get("person_count") or 1),
		quality_score=float(payload["quality_score"]) if "quality_score" in payload else None,
		quality_review_recorded=bool(payload.get("quality_review_recorded")),
	)


def analyze_pose(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.analyze_pose(
		analysis_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		estimation_id=str(payload["estimation_id"]),
		analysis_type=str(payload.get("analysis_type") or "biomechanical"),
		medical_grade=bool(payload.get("medical_grade")),
		reviewer=str(payload.get("reviewer") or ""),
		metrics=dict(payload.get("metrics") or {}),
	)


def reconstruct_3d(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.reconstruct_3d(
		reconstruction_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		estimation_id=str(payload["estimation_id"]),
		camera_calibration_ref=str(payload.get("camera_calibration_ref") or ""),
	)


def register_pose_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_pose_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload.get("scope") or ""),
		contribution_disclosed=bool(payload.get("contribution_disclosed")),
		policy_ref=str(payload.get("policy_ref") or ""),
		registered=bool(payload.get("registered", True)),
	)


def change_session_state(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.change_session_state(
		tenant_id=str(payload.get("tenant_id") or "default"),
		session_id=str(payload["session_id"]),
		status=str(payload["status"]),
		reason=str(payload.get("reason") or ""),
		audit_recorded=bool(payload.get("audit_recorded", True)),
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
