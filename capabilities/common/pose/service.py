"""Service layer for executable APG pose-estimation operations."""

from __future__ import annotations

from hashlib import sha256
from statistics import mean
from typing import Any

from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, evaluate_capability_rules, get_capability_contract
from .models import (
	PoseAgentRecord,
	PoseAnalysisRecord,
	PoseAuditEvent,
	PoseEstimateRecord,
	PoseFrameRecord,
	PoseModelRecord,
	PoseReconstructionRecord,
	PoseSessionRecord,
	utc_now_iso,
)


SUPPORTED_MODEL_TYPES = {"movenet", "rtmpose", "vitpose", "swin_pose", "edge_pose"}
SESSION_STATUSES = {"active", "paused", "completed", "retired"}


class PoseService:
	"""In-process pose service enforcing tenant, consent, quality, and audit guardrails."""

	def __init__(self) -> None:
		self._models: dict[str, PoseModelRecord] = {}
		self._sessions: dict[str, PoseSessionRecord] = {}
		self._frames: dict[str, PoseFrameRecord] = {}
		self._estimates: dict[str, PoseEstimateRecord] = {}
		self._analyses: dict[str, PoseAnalysisRecord] = {}
		self._reconstructions: dict[str, PoseReconstructionRecord] = {}
		self._agents: dict[str, PoseAgentRecord] = {}
		self._audit_events: dict[str, PoseAuditEvent] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_model(
		self,
		model_id: str,
		tenant_id: str,
		name: str,
		model_type: str,
		owner: str,
		policy_ref: str,
		minimum_keypoint_confidence: float = 0.72,
		edge_ready: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_model",
			"model_owner_present": bool(owner.strip()),
			"model_policy_attached": bool(policy_ref.strip()),
		})
		self._raise_if_blocked(result)
		normalized_type = _normalize_model_type(model_type)
		model = PoseModelRecord(
			id=model_id,
			tenant_id=tenant_id,
			name=name or model_id,
			model_type=normalized_type,
			owner=owner,
			policy_ref=policy_ref,
			minimum_keypoint_confidence=round(float(minimum_keypoint_confidence), 4),
			edge_ready=edge_ready,
		)
		self._models[model.id] = model
		self._audit(tenant_id, "pose_model_registered", model.id, f"Registered pose model {model.name}")
		return model.to_dict()

	def start_session(
		self,
		session_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		source_ref: str,
		model_id: str,
		subject_consent_recorded: bool,
		secure_stream: bool,
		realtime_stream: bool = False,
		sensitive_use: bool = False,
		approval_ref: str = "",
		max_persons: int = 1,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		model = self._require_model(model_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "start_tracking",
			"session_owner_assigned": bool(owner.strip()),
			"source_reference_present": bool(source_ref.strip()),
			"subject_consent_recorded": bool(subject_consent_recorded),
			"realtime_stream": bool(realtime_stream),
			"secure_stream": bool(secure_stream),
			"sensitive_use": bool(sensitive_use),
			"approval_recorded": bool(approval_ref.strip()),
		})
		self._raise_if_blocked(result)
		if int(max_persons) < 1:
			raise PermissionError("max_persons_required")
		session = PoseSessionRecord(
			id=session_id,
			tenant_id=tenant_id,
			name=name or session_id,
			owner=owner,
			source_ref=source_ref,
			model_id=model.id,
			subject_consent_recorded=subject_consent_recorded,
			secure_stream=secure_stream,
			realtime_stream=realtime_stream,
			sensitive_use=sensitive_use,
			approval_ref=approval_ref,
			max_persons=int(max_persons),
			metadata=dict(metadata or {}),
		)
		self._sessions[session.id] = session
		self._audit(tenant_id, "pose_session_started", session.id, f"Started pose session {session.name}")
		return session.to_dict()

	def record_frame(
		self,
		frame_id: str,
		tenant_id: str,
		session_id: str,
		frame_number: int,
		occurred_at: str,
		source_ref: str,
		width: int,
		height: int,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		session = self._require_session(session_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "record_frame",
			"frame_timestamp_present": bool(occurred_at.strip()),
		})
		self._raise_if_blocked(result)
		if int(frame_number) < 0:
			raise PermissionError("frame_number_required")
		if int(width) <= 0 or int(height) <= 0:
			raise PermissionError("frame_dimensions_required")
		frame = PoseFrameRecord(
			id=frame_id,
			tenant_id=tenant_id,
			session_id=session.id,
			frame_number=int(frame_number),
			occurred_at=occurred_at,
			source_ref=source_ref,
			width=int(width),
			height=int(height),
		)
		self._frames[frame.id] = frame
		self._audit(tenant_id, "pose_frame_recorded", frame.id, f"Recorded frame {frame.frame_number}")
		return frame.to_dict()

	def estimate_pose(
		self,
		estimate_id: str,
		tenant_id: str,
		session_id: str,
		frame_id: str,
		model_id: str,
		keypoints: list[dict[str, Any]],
		person_count: int = 1,
		quality_score: float | None = None,
		quality_review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		session = self._require_session(session_id, tenant_id)
		frame = self._require_frame(frame_id, tenant_id)
		model = self._require_model(model_id, tenant_id)
		if frame.session_id != session.id:
			raise PermissionError("frame_session_mismatch")
		normalized_keypoints = [_normalize_keypoint(item) for item in keypoints]
		confidence = round(mean([item["confidence"] for item in normalized_keypoints]), 4) if normalized_keypoints else 0.0
		quality = round(float(quality_score if quality_score is not None else confidence), 4)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "estimate_pose",
			"keypoint_count": len(normalized_keypoints),
			"person_count": int(person_count),
			"pose_quality_score": quality,
			"quality_review_recorded": bool(quality_review_recorded),
		})
		if result["decision"] != "allow":
			self._raise_if_blocked(result)
		if confidence < model.minimum_keypoint_confidence and not quality_review_recorded:
			raise PermissionError("keypoint_confidence_review_required")
		estimate = PoseEstimateRecord(
			id=estimate_id,
			tenant_id=tenant_id,
			session_id=session.id,
			frame_id=frame.id,
			model_id=model.id,
			keypoints=normalized_keypoints,
			person_count=int(person_count),
			quality_score=quality,
			confidence=confidence,
			quality_review_recorded=quality_review_recorded,
		)
		self._estimates[estimate.id] = estimate
		self._audit(tenant_id, "pose_estimated", estimate.id, f"Estimated pose with {len(normalized_keypoints)} keypoints")
		return estimate.to_dict()

	def analyze_pose(
		self,
		analysis_id: str,
		tenant_id: str,
		estimation_id: str,
		analysis_type: str,
		medical_grade: bool = False,
		reviewer: str = "",
		metrics: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		estimate = self._require_estimate(estimation_id, tenant_id)
		session = self._require_session(estimate.session_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "analyze_pose",
			"subject_consent_recorded": session.subject_consent_recorded,
			"medical_grade": bool(medical_grade),
			"medical_review_recorded": bool(reviewer.strip()),
		})
		self._raise_if_blocked(result)
		analysis = PoseAnalysisRecord(
			id=analysis_id,
			tenant_id=tenant_id,
			estimation_id=estimate.id,
			analysis_type=analysis_type or "biomechanical",
			metrics=dict(metrics or _basic_metrics(estimate.keypoints)),
			medical_grade=medical_grade,
			reviewer=reviewer,
		)
		self._analyses[analysis.id] = analysis
		self._audit(tenant_id, "pose_analysis_completed", analysis.id, f"Completed {analysis.analysis_type} analysis")
		return analysis.to_dict()

	def reconstruct_3d(
		self,
		reconstruction_id: str,
		tenant_id: str,
		estimation_id: str,
		camera_calibration_ref: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		estimate = self._require_estimate(estimation_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "reconstruct_3d",
			"camera_calibration_present": bool(camera_calibration_ref.strip()),
		})
		self._raise_if_blocked(result)
		keypoints_3d = [
			{**item, "z": round((index + 1) / max(1, len(estimate.keypoints)), 4)}
			for index, item in enumerate(estimate.keypoints)
		]
		reconstruction = PoseReconstructionRecord(
			id=reconstruction_id,
			tenant_id=tenant_id,
			estimation_id=estimate.id,
			camera_calibration_ref=camera_calibration_ref,
			keypoints_3d=keypoints_3d,
		)
		self._reconstructions[reconstruction.id] = reconstruction
		self._audit(tenant_id, "pose_3d_reconstructed", reconstruction.id, "Reconstructed 3D pose")
		return reconstruction.to_dict()

	def register_pose_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool,
		policy_ref: str = "",
		registered: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		normalized_runtime = _normalize_agent_runtime(runtime)
		normalized_role = _normalize_agent_role(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"pose_agent_present": True,
			"agent_registered": bool(registered),
			"agent_runtime_supported": bool(normalized_runtime),
			"agent_scope_present": bool(scope.strip()),
			"agent_contribution_disclosed": bool(contribution_disclosed),
		})
		self._raise_if_blocked(result)
		agent = PoseAgentRecord(
			id=agent_id,
			tenant_id=tenant_id,
			name=name or agent_id,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			registered=registered,
			contribution_disclosed=contribution_disclosed,
			policy_ref=policy_ref,
		)
		self._agents[agent.id] = agent
		self._audit(tenant_id, "pose_agent_registered", agent.id, f"Registered pose agent {agent.name}")
		return agent.to_dict()

	def change_session_state(
		self,
		tenant_id: str,
		session_id: str,
		status: str,
		reason: str,
		audit_recorded: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		session = self._require_session(session_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"state_change_requested": True,
			"state_change_reason_present": bool(reason.strip()),
			"audit_event_recorded": bool(audit_recorded),
		})
		self._raise_if_blocked(result)
		normalized_status = status.strip().lower()
		if normalized_status not in SESSION_STATUSES:
			raise PermissionError("unsupported_session_status")
		session.status = normalized_status
		session.updated_at = utc_now_iso()
		self._audit(tenant_id, "pose_session_state_changed", session.id, reason)
		return session.to_dict()

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
		metadata = dict(metadata or {})
		model = self.register_model(
			model_id=f"{record_id}-model",
			tenant_id=tenant_id,
			name=str(metadata.get("model_name") or "Compatibility Pose Model"),
			model_type=str(metadata.get("model_type") or "rtmpose"),
			owner=str(metadata.get("owner") or "pose"),
			policy_ref=str(metadata.get("policy_ref") or "pose-policy:compatibility"),
		)
		session = self.start_session(
			session_id=record_id,
			tenant_id=tenant_id,
			name=str(metadata.get("name") or record_id),
			owner=str(metadata.get("owner") or "pose"),
			source_ref=str(metadata.get("source_ref") or "source:compatibility"),
			model_id=model["id"],
			subject_consent_recorded=bool(metadata.get("subject_consent_recorded", True)),
			secure_stream=bool(metadata.get("secure_stream", True)),
		)
		if status != "active":
			self._sessions[record_id].status = status
		return session

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_sessions(tenant_id)

	def list_models(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._models, tenant_id)

	def list_sessions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._sessions, tenant_id)

	def list_frames(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._frames, tenant_id)

	def list_estimates(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._estimates, tenant_id)

	def list_analyses(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._analyses, tenant_id)

	def list_reconstructions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._reconstructions, tenant_id)

	def list_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		return {
			"tenant_id": tenant_id,
			"model_count": len(self.list_models(tenant_id)),
			"session_count": len(self.list_sessions(tenant_id)),
			"active_session_count": sum(1 for item in self._sessions.values() if item.tenant_id == tenant_id and item.status == "active"),
			"frame_count": len(self.list_frames(tenant_id)),
			"estimate_count": len(self.list_estimates(tenant_id)),
			"analysis_count": len(self.list_analyses(tenant_id)),
			"reconstruction_count": len(self.list_reconstructions(tenant_id)),
			"agent_count": len(self.list_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _require_tenant(self, tenant_id: str) -> None:
		self._raise_if_blocked(self.evaluate({"tenant_context_present": bool(tenant_id)}))

	def _require_model(self, model_id: str, tenant_id: str) -> PoseModelRecord:
		model = self._models.get(model_id)
		if model is None or model.tenant_id != tenant_id:
			raise LookupError("pose_model_not_found")
		return model

	def _require_session(self, session_id: str, tenant_id: str) -> PoseSessionRecord:
		session = self._sessions.get(session_id)
		if session is None or session.tenant_id != tenant_id:
			raise LookupError("pose_session_not_found")
		return session

	def _require_frame(self, frame_id: str, tenant_id: str) -> PoseFrameRecord:
		frame = self._frames.get(frame_id)
		if frame is None or frame.tenant_id != tenant_id:
			raise LookupError("pose_frame_not_found")
		return frame

	def _require_estimate(self, estimate_id: str, tenant_id: str) -> PoseEstimateRecord:
		estimate = self._estimates.get(estimate_id)
		if estimate is None or estimate.tenant_id != tenant_id:
			raise LookupError("pose_estimate_not_found")
		return estimate

	def _raise_if_blocked(self, result: dict[str, Any]) -> None:
		if result["decision"] != "allow":
			raise PermissionError(", ".join(self._reasons(result)) or "pose_policy_blocked")

	def _audit(self, tenant_id: str, event_type: str, subject_id: str, message: str, severity: str = "info", metadata: dict[str, Any] | None = None) -> None:
		event = PoseAuditEvent(
			id=_stable_id("poseaudit", tenant_id, event_type, subject_id, len(self._audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			severity=severity,
			metadata=dict(metadata or {}),
		)
		self._audit_events[event.id] = event

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(values, key=lambda item: item.id)]

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(action.get("reason", "pose_policy_blocked") for action in result.get("actions", []))


PoseEstimationService = PoseService


def _stable_id(prefix: str, *parts: object) -> str:
	seed = "|".join(str(part) for part in parts if part is not None)
	return f"{prefix}_{sha256(seed.encode('utf-8')).hexdigest()[:12]}"


def _normalize_model_type(model_type: str) -> str:
	value = (model_type or "rtmpose").strip().lower()
	if value not in SUPPORTED_MODEL_TYPES:
		raise PermissionError("unsupported_pose_model_type")
	return value


def _normalize_agent_runtime(runtime: str) -> str:
	value = (runtime or "").strip().lower()
	if value not in SUPPORTED_AGENT_RUNTIMES:
		raise PermissionError("pose_agent_runtime_not_supported")
	return value


def _normalize_agent_role(role: str) -> str:
	value = (role or "").strip().lower()
	if value not in SUPPORTED_AGENT_ROLES:
		raise PermissionError("unsupported_pose_agent_role")
	return value


def _normalize_keypoint(keypoint: dict[str, Any]) -> dict[str, Any]:
	name = str(keypoint.get("name") or keypoint.get("type") or "").strip().lower()
	if not name:
		raise PermissionError("keypoint_name_required")
	confidence = round(float(keypoint.get("confidence", 0.0)), 4)
	if not 0 <= confidence <= 1:
		raise PermissionError("keypoint_confidence_invalid")
	return {
		"name": name,
		"x": round(float(keypoint.get("x", 0.0)), 4),
		"y": round(float(keypoint.get("y", 0.0)), 4),
		"confidence": confidence,
		"visibility": round(float(keypoint.get("visibility", 1.0)), 4),
	}


def _basic_metrics(keypoints: list[dict[str, Any]]) -> dict[str, Any]:
	confidences = [float(item["confidence"]) for item in keypoints]
	return {
		"keypoint_count": len(keypoints),
		"average_confidence": round(mean(confidences), 4) if confidences else 0.0,
		"low_confidence_keypoints": sum(1 for confidence in confidences if confidence < 0.72),
	}
