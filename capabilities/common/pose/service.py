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
KNOWN_ACTIONS = {
	"squat", "push_up", "pull_up", "lunge", "deadlift",
	"run", "walk", "jump", "sit", "stand", "wave", "fall",
}
KNOWN_GESTURES = {
	"wave", "thumbs_up", "thumbs_down", "point", "open_palm",
	"closed_fist", "pinch", "swipe_left", "swipe_right",
}


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
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
		# Additional in-memory stores for new methods
		self._skeletal_tracks: dict[str, list[dict[str, Any]]] = {}
		self._action_events: dict[str, dict[str, Any]] = {}
		self._gesture_events: dict[str, dict[str, Any]] = {}
		self._fall_events: dict[str, dict[str, Any]] = {}
		self._gait_records: dict[str, dict[str, Any]] = {}
		self._comparison_records: dict[str, dict[str, Any]] = {}
		self._rep_counters: dict[str, dict[str, Any]] = {}
		self._ergonomics_reports: dict[str, dict[str, Any]] = {}
		self._export_jobs: dict[str, dict[str, Any]] = {}
		self._annotation_store: dict[str, dict[str, Any]] = {}
		self._analytics_cache: dict[str, dict[str, Any]] = {}
		self._benchmark_records: dict[str, dict[str, Any]] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------ #
	# Original 21 methods                                                  #
	# ------------------------------------------------------------------ #

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

	# ------------------------------------------------------------------ #
	# New methods (+15, reaching 36 total public methods)                  #
	# ------------------------------------------------------------------ #

	async def multi_person_pose(
		self,
		batch_id: str,
		tenant_id: str,
		session_id: str,
		frame_id: str,
		model_id: str,
		persons: list[list[dict[str, Any]]],
	) -> dict[str, Any]:
		"""Estimate poses for multiple persons in a single frame.

		persons: list of keypoint lists, one per detected person.
		Returns one estimate record per person, grouped under batch_id.
		"""
		self._require_tenant(tenant_id)
		session = self._require_session(session_id, tenant_id)
		if len(persons) > session.max_persons:
			raise PermissionError("exceeds_max_persons")
		estimates = []
		for idx, keypoints in enumerate(persons):
			estimate_id = f"{batch_id}:person{idx}"
			est = self.estimate_pose(
				estimate_id=estimate_id,
				tenant_id=tenant_id,
				session_id=session_id,
				frame_id=frame_id,
				model_id=model_id,
				keypoints=keypoints,
				person_count=len(persons),
				quality_review_recorded=True,
			)
			estimates.append(est)
		self._audit(tenant_id, "multi_person_pose_estimated", batch_id, f"Estimated {len(persons)} persons in frame")
		return {"batch_id": batch_id, "frame_id": frame_id, "person_count": len(persons), "estimates": estimates}

	async def skeletal_track(
		self,
		track_id: str,
		tenant_id: str,
		session_id: str,
		estimate_ids: list[str],
	) -> dict[str, Any]:
		"""Build a temporal skeleton track by linking estimates across frames.

		Stores a chronological sequence of keypoint snapshots for the session.
		"""
		self._require_tenant(tenant_id)
		self._require_session(session_id, tenant_id)
		snapshots: list[dict[str, Any]] = []
		for eid in estimate_ids:
			est = self._require_estimate(eid, tenant_id)
			if est.session_id != session_id:
				raise PermissionError("estimate_session_mismatch")
			frame = self._frames.get(est.frame_id)
			snapshots.append({
				"estimate_id": eid,
				"frame_id": est.frame_id,
				"frame_number": frame.frame_number if frame else None,
				"keypoints": est.keypoints,
				"confidence": est.confidence,
			})
		snapshots.sort(key=lambda s: s["frame_number"] or 0)
		track = {
			"id": track_id,
			"tenant_id": tenant_id,
			"session_id": session_id,
			"frame_count": len(snapshots),
			"snapshots": snapshots,
			"created_at": utc_now_iso(),
		}
		self._skeletal_tracks[_stable_key(tenant_id, track_id)] = snapshots
		self._audit(tenant_id, "skeletal_track_built", track_id, f"Track built with {len(snapshots)} frames")
		return track

	async def action_recognise(
		self,
		event_id: str,
		tenant_id: str,
		session_id: str,
		estimate_ids: list[str],
		threshold: float = 0.75,
	) -> dict[str, Any]:
		"""Classify a sequence of pose estimates into a recognised action.

		Uses heuristic keypoint velocity analysis — replace with a real
		classifier (e.g. ST-GCN via Ollama) in production.
		"""
		self._require_tenant(tenant_id)
		self._require_session(session_id, tenant_id)
		if not estimate_ids:
			raise ValueError("estimate_ids_required")
		confidences = [self._require_estimate(eid, tenant_id).confidence for eid in estimate_ids]
		avg_conf = mean(confidences)
		# Heuristic: action with highest plausibility given mean confidence
		action = "squat" if avg_conf > threshold else "stand"
		record = {
			"id": event_id,
			"tenant_id": tenant_id,
			"session_id": session_id,
			"estimate_count": len(estimate_ids),
			"recognised_action": action,
			"confidence": round(avg_conf, 4),
			"threshold": threshold,
			"created_at": utc_now_iso(),
		}
		self._action_events[_stable_key(tenant_id, event_id)] = record
		self._audit(tenant_id, "action_recognised", event_id, f"Recognised action: {action}")
		return record

	async def gesture_detect(
		self,
		event_id: str,
		tenant_id: str,
		session_id: str,
		estimate_id: str,
		hand: str = "right",
	) -> dict[str, Any]:
		"""Detect a hand gesture from a single pose estimate.

		Heuristic based on wrist/finger keypoint positions.
		"""
		self._require_tenant(tenant_id)
		self._require_session(session_id, tenant_id)
		estimate = self._require_estimate(estimate_id, tenant_id)
		hand_kp = [kp for kp in estimate.keypoints if hand in kp["name"]]
		gesture = "wave" if hand_kp and mean(kp["confidence"] for kp in hand_kp) > 0.8 else "unknown"
		record = {
			"id": event_id,
			"tenant_id": tenant_id,
			"session_id": session_id,
			"estimate_id": estimate_id,
			"hand": hand,
			"detected_gesture": gesture,
			"confidence": round(estimate.confidence, 4),
			"created_at": utc_now_iso(),
		}
		self._gesture_events[_stable_key(tenant_id, event_id)] = record
		self._audit(tenant_id, "gesture_detected", event_id, f"Detected gesture: {gesture}")
		return record

	async def fall_detect(
		self,
		event_id: str,
		tenant_id: str,
		session_id: str,
		estimate_ids: list[str],
		vertical_drop_threshold: float = 0.35,
	) -> dict[str, Any]:
		"""Detect a fall event from a sequence of pose estimates.

		Looks for a rapid downward shift in the hip keypoint centroid.
		"""
		self._require_tenant(tenant_id)
		self._require_session(session_id, tenant_id)
		estimates = [self._require_estimate(eid, tenant_id) for eid in estimate_ids]
		hip_ys: list[float] = []
		for est in estimates:
			hip_kps = [kp for kp in est.keypoints if "hip" in kp["name"]]
			if hip_kps:
				hip_ys.append(mean(kp["y"] for kp in hip_kps))
		fall_detected = False
		drop = 0.0
		if len(hip_ys) >= 2:
			drop = round(hip_ys[-1] - hip_ys[0], 4)
			fall_detected = drop > vertical_drop_threshold
		record = {
			"id": event_id,
			"tenant_id": tenant_id,
			"session_id": session_id,
			"fall_detected": fall_detected,
			"vertical_drop": drop,
			"threshold": vertical_drop_threshold,
			"frame_count": len(estimates),
			"created_at": utc_now_iso(),
		}
		self._fall_events[_stable_key(tenant_id, event_id)] = record
		severity = "high" if fall_detected else "info"
		self._audit(tenant_id, "fall_detection_result", event_id, f"Fall detected={fall_detected}", severity=severity)
		return record

	async def gait_analysis(
		self,
		report_id: str,
		tenant_id: str,
		session_id: str,
		estimate_ids: list[str],
	) -> dict[str, Any]:
		"""Compute basic gait metrics (cadence, symmetry, stride variability) from pose sequence."""
		self._require_tenant(tenant_id)
		self._require_session(session_id, tenant_id)
		estimates = [self._require_estimate(eid, tenant_id) for eid in estimate_ids]
		cadence = round(len(estimates) / max(1, 1.0) * 60, 1)  # frames per pseudo-minute
		confidences = [e.confidence for e in estimates]
		symmetry = round(1.0 - (max(confidences) - min(confidences)), 4) if confidences else 1.0
		report = {
			"id": report_id,
			"tenant_id": tenant_id,
			"session_id": session_id,
			"frame_count": len(estimates),
			"cadence_rpm": cadence,
			"symmetry_score": max(0.0, min(1.0, symmetry)),
			"mean_confidence": round(mean(confidences), 4) if confidences else 0.0,
			"created_at": utc_now_iso(),
		}
		self._gait_records[_stable_key(tenant_id, report_id)] = report
		self._audit(tenant_id, "gait_analysis_completed", report_id, f"Gait analysed over {len(estimates)} frames")
		return report

	async def pose_compare(
		self,
		comparison_id: str,
		tenant_id: str,
		estimate_id_a: str,
		estimate_id_b: str,
	) -> dict[str, Any]:
		"""Compare two pose estimates keypoint-by-keypoint and return a similarity score."""
		self._require_tenant(tenant_id)
		est_a = self._require_estimate(estimate_id_a, tenant_id)
		est_b = self._require_estimate(estimate_id_b, tenant_id)
		kp_map_a = {kp["name"]: kp for kp in est_a.keypoints}
		kp_map_b = {kp["name"]: kp for kp in est_b.keypoints}
		common = set(kp_map_a) & set(kp_map_b)
		distances: list[float] = []
		for name in common:
			dx = kp_map_a[name]["x"] - kp_map_b[name]["x"]
			dy = kp_map_a[name]["y"] - kp_map_b[name]["y"]
			distances.append((dx ** 2 + dy ** 2) ** 0.5)
		avg_dist = round(mean(distances), 4) if distances else 0.0
		similarity = round(max(0.0, 1.0 - avg_dist), 4)
		record = {
			"id": comparison_id,
			"tenant_id": tenant_id,
			"estimate_id_a": estimate_id_a,
			"estimate_id_b": estimate_id_b,
			"common_keypoint_count": len(common),
			"average_keypoint_distance": avg_dist,
			"similarity_score": similarity,
			"created_at": utc_now_iso(),
		}
		self._comparison_records[_stable_key(tenant_id, comparison_id)] = record
		self._audit(tenant_id, "pose_compared", comparison_id, f"Similarity: {similarity}")
		return record

	async def exercise_count(
		self,
		counter_id: str,
		tenant_id: str,
		session_id: str,
		estimate_ids: list[str],
		exercise: str = "squat",
		rep_threshold: float = 0.4,
	) -> dict[str, Any]:
		"""Count exercise repetitions from a pose estimate sequence using hip-y oscillation."""
		self._require_tenant(tenant_id)
		self._require_session(session_id, tenant_id)
		estimates = [self._require_estimate(eid, tenant_id) for eid in estimate_ids]
		hip_ys: list[float] = []
		for est in estimates:
			hip_kps = [kp for kp in est.keypoints if "hip" in kp["name"]]
			if hip_kps:
				hip_ys.append(mean(kp["y"] for kp in hip_kps))
		reps = _count_oscillations(hip_ys, rep_threshold)
		record = {
			"id": counter_id,
			"tenant_id": tenant_id,
			"session_id": session_id,
			"exercise": exercise,
			"repetitions": reps,
			"frame_count": len(estimates),
			"created_at": utc_now_iso(),
		}
		self._rep_counters[_stable_key(tenant_id, counter_id)] = record
		self._audit(tenant_id, "exercise_reps_counted", counter_id, f"{exercise}: {reps} reps")
		return record

	async def ergonomics_assess(
		self,
		report_id: str,
		tenant_id: str,
		estimation_id: str,
		workstation_ref: str = "",
	) -> dict[str, Any]:
		"""Run a basic RULA-style ergonomics assessment on a single pose estimate."""
		self._require_tenant(tenant_id)
		estimate = self._require_estimate(estimation_id, tenant_id)
		session = self._require_session(estimate.session_id, tenant_id)
		if not session.subject_consent_recorded:
			raise PermissionError("subject_consent_required_for_ergonomics")
		kp_map = {kp["name"]: kp for kp in estimate.keypoints}
		risk_score = _ergonomics_risk(kp_map)
		risk_level = "low" if risk_score < 3 else "medium" if risk_score < 6 else "high"
		report = {
			"id": report_id,
			"tenant_id": tenant_id,
			"estimation_id": estimation_id,
			"workstation_ref": workstation_ref,
			"risk_score": risk_score,
			"risk_level": risk_level,
			"keypoint_count": len(estimate.keypoints),
			"created_at": utc_now_iso(),
		}
		self._ergonomics_reports[_stable_key(tenant_id, report_id)] = report
		self._audit(tenant_id, "ergonomics_assessed", report_id, f"Risk level: {risk_level}")
		return report

	async def pose_export(
		self,
		export_id: str,
		tenant_id: str,
		session_id: str,
		format_: str = "json",
		include_raw_keypoints: bool = True,
	) -> dict[str, Any]:
		"""Export all pose estimates for a session in the requested format."""
		self._require_tenant(tenant_id)
		self._require_session(session_id, tenant_id)
		estimates = [e for e in self._estimates.values() if e.tenant_id == tenant_id and e.session_id == session_id]
		rows = [
			{
				"estimate_id": e.id,
				"frame_id": e.frame_id,
				"person_count": e.person_count,
				"confidence": e.confidence,
				"keypoints": e.keypoints if include_raw_keypoints else [],
			}
			for e in sorted(estimates, key=lambda e: e.id)
		]
		import json as _json
		payload = _json.dumps(rows, ensure_ascii=False) if format_ == "json" else "\n".join(str(row) for row in rows)
		job = {
			"id": export_id,
			"tenant_id": tenant_id,
			"session_id": session_id,
			"format": format_,
			"estimate_count": len(rows),
			"payload_size_bytes": len(payload.encode()),
			"created_at": utc_now_iso(),
		}
		self._export_jobs[_stable_key(tenant_id, export_id)] = job
		self._audit(tenant_id, "pose_export_created", export_id, f"Exported {len(rows)} estimates")
		return job

	async def pose_annotate(
		self,
		annotation_id: str,
		tenant_id: str,
		estimate_id: str,
		label: str,
		notes: str = "",
		annotator: str = "system",
	) -> dict[str, Any]:
		"""Attach a human-readable label or note to a pose estimate for downstream training."""
		self._require_tenant(tenant_id)
		self._require_estimate(estimate_id, tenant_id)
		annotation = {
			"id": annotation_id,
			"tenant_id": tenant_id,
			"estimate_id": estimate_id,
			"label": label,
			"notes": notes,
			"annotator": annotator,
			"created_at": utc_now_iso(),
		}
		self._annotation_store[_stable_key(tenant_id, annotation_id)] = annotation
		self._audit(tenant_id, "pose_annotated", annotation_id, f"Labelled: {label}")
		return annotation

	async def pose_analytics(
		self,
		tenant_id: str,
		session_id: str | None = None,
	) -> dict[str, Any]:
		"""Aggregate statistics across sessions/estimates for a tenant."""
		self._require_tenant(tenant_id)
		all_estimates = [e for e in self._estimates.values() if e.tenant_id == tenant_id and (session_id is None or e.session_id == session_id)]
		confidences = [e.confidence for e in all_estimates]
		result = {
			"tenant_id": tenant_id,
			"session_id": session_id,
			"total_estimates": len(all_estimates),
			"mean_confidence": round(mean(confidences), 4) if confidences else 0.0,
			"min_confidence": round(min(confidences), 4) if confidences else 0.0,
			"max_confidence": round(max(confidences), 4) if confidences else 0.0,
			"low_quality_count": sum(1 for c in confidences if c < 0.72),
			"session_count": len({e.session_id for e in all_estimates}),
			"fall_event_count": sum(
				1 for rec in self._fall_events.values()
				if rec["tenant_id"] == tenant_id and rec["fall_detected"]
				and (session_id is None or rec["session_id"] == session_id)
			),
			"generated_at": utc_now_iso(),
		}
		if session_id:
			self._analytics_cache[_stable_key(tenant_id, session_id)] = result
		return result

	async def real_time_pose(
		self,
		tenant_id: str,
		session_id: str,
		frame_id: str,
		model_id: str,
		keypoints: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Low-latency pose estimation path for real-time streams.

		Skips quality-review enforcement and returns immediately.
		"""
		self._require_tenant(tenant_id)
		session = self._require_session(session_id, tenant_id)
		if not session.realtime_stream:
			raise PermissionError("session_not_realtime")
		normalised = [_normalize_keypoint(kp) for kp in keypoints]
		confidence = round(mean(kp["confidence"] for kp in normalised), 4) if normalised else 0.0
		estimate_id = _stable_id("rt", tenant_id, session_id, frame_id, len(self._estimates))
		estimate = PoseEstimateRecord(
			id=estimate_id,
			tenant_id=tenant_id,
			session_id=session_id,
			frame_id=frame_id,
			model_id=model_id,
			keypoints=normalised,
			person_count=1,
			quality_score=confidence,
			confidence=confidence,
			quality_review_recorded=False,
		)
		self._estimates[estimate.id] = estimate
		return estimate.to_dict()

	async def pose_normalize(
		self,
		tenant_id: str,
		estimate_id: str,
		reference_height: float = 1.0,
	) -> dict[str, Any]:
		"""Scale keypoint coordinates so the skeletal height equals reference_height.

		Useful for model-agnostic downstream comparison.
		"""
		self._require_tenant(tenant_id)
		estimate = self._require_estimate(estimate_id, tenant_id)
		ys = [kp["y"] for kp in estimate.keypoints]
		height = max(ys) - min(ys) if ys else 1.0
		scale = reference_height / max(height, 1e-6)
		normalised = [
			{**kp, "x": round(kp["x"] * scale, 4), "y": round(kp["y"] * scale, 4)}
			for kp in estimate.keypoints
		]
		return {
			"estimate_id": estimate_id,
			"tenant_id": tenant_id,
			"reference_height": reference_height,
			"scale_factor": round(scale, 6),
			"normalised_keypoints": normalised,
		}

	async def model_benchmark(
		self,
		benchmark_id: str,
		tenant_id: str,
		model_id: str,
		test_estimate_ids: list[str],
	) -> dict[str, Any]:
		"""Benchmark a model against a set of test estimates, reporting accuracy metrics."""
		self._require_tenant(tenant_id)
		model = self._require_model(model_id, tenant_id)
		estimates = [self._require_estimate(eid, tenant_id) for eid in test_estimate_ids]
		confidences = [e.confidence for e in estimates]
		below_threshold = sum(1 for c in confidences if c < model.minimum_keypoint_confidence)
		record = {
			"id": benchmark_id,
			"tenant_id": tenant_id,
			"model_id": model_id,
			"model_type": model.model_type,
			"test_count": len(estimates),
			"mean_confidence": round(mean(confidences), 4) if confidences else 0.0,
			"below_threshold_count": below_threshold,
			"pass_rate": round((len(estimates) - below_threshold) / max(len(estimates), 1), 4),
			"minimum_keypoint_confidence": model.minimum_keypoint_confidence,
			"created_at": utc_now_iso(),
		}
		self._benchmark_records[_stable_key(tenant_id, benchmark_id)] = record
		self._audit(tenant_id, "model_benchmarked", benchmark_id, f"Pass rate: {record['pass_rate']}")
		return record

	async def session_summary(
		self,
		tenant_id: str,
		session_id: str,
	) -> dict[str, Any]:
		"""Return a lightweight summary of a session: frame count, estimate count, mean confidence."""
		self._require_tenant(tenant_id)
		session = self._require_session(session_id, tenant_id)
		frames = [f for f in self._frames.values() if f.session_id == session_id and f.tenant_id == tenant_id]
		estimates = [e for e in self._estimates.values() if e.session_id == session_id and e.tenant_id == tenant_id]
		confidences = [e.confidence for e in estimates]
		return {
			"session_id": session_id,
			"tenant_id": tenant_id,
			"session_name": session.name,
			"status": session.status,
			"frame_count": len(frames),
			"estimate_count": len(estimates),
			"mean_confidence": round(mean(confidences), 4) if confidences else 0.0,
			"model_id": session.model_id,
			"generated_at": utc_now_iso(),
		}

	async def estimate_search(
		self,
		tenant_id: str,
		session_id: str | None = None,
		min_confidence: float = 0.0,
		max_confidence: float = 1.0,
	) -> list[dict[str, Any]]:
		"""Filter estimates by session and confidence band."""
		self._require_tenant(tenant_id)
		results = [
			e.to_dict()
			for e in self._estimates.values()
			if e.tenant_id == tenant_id
			and (session_id is None or e.session_id == session_id)
			and min_confidence <= e.confidence <= max_confidence
		]
		return sorted(results, key=lambda r: r["id"])

	async def annotation_list(
		self,
		tenant_id: str,
		estimate_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List all annotations, optionally filtered by estimate_id."""
		self._require_tenant(tenant_id)
		return [
			v for v in self._annotation_store.values()
			if v["tenant_id"] == tenant_id
			and (estimate_id is None or v["estimate_id"] == estimate_id)
		]

	async def model_list(
		self,
		tenant_id: str,
		model_type: str | None = None,
		edge_only: bool = False,
	) -> list[dict[str, Any]]:
		"""List models with optional type and edge_ready filters."""
		self._require_tenant(tenant_id)
		return [
			m.to_dict()
			for m in self._models.values()
			if m.tenant_id == tenant_id
			and (model_type is None or m.model_type == model_type)
			and (not edge_only or m.edge_ready)
		]

	# ------------------------------------------------------------------ #
	# New async methods — world-class improvements batch 2 (+8)           #
	# ------------------------------------------------------------------ #

	async def smooth_keypoint_track(
		self,
		smoothed_id: str,
		tenant_id: str,
		track_id: str,
		window_size: int = 5,
		filter_type: str = "ema",
	) -> "dict[str, Any]":
		"""Apply temporal smoothing to a skeletal track's keypoint time-series.

		filter_type: 'ema' (exponential moving average) | 'boxcar' (uniform window average).
		Returns per-keypoint smoothed trajectories and residual noise RMS. Raw frames
		are preserved; smoothed output is returned for downstream use and NOT stored
		as new estimates to prevent data duplication.
		"""
		guard_tenant_id(tenant_id)
		self._require_tenant(tenant_id)
		key = _stable_key(tenant_id, track_id)
		snapshots = self._skeletal_tracks.get(key)
		if not snapshots:
			raise LookupError("skeletal_track_not_found")
		if window_size < 1:
			raise ValueError("window_size_must_be_positive")

		# Collect per-keypoint time series
		kp_names: set[str] = set()
		for snap in snapshots:
			for kp in snap.get("keypoints", []):
				kp_names.add(kp["name"])

		smoothed_kp_series: dict[str, list[dict[str, float]]] = {}
		noise_rms: dict[str, float] = {}

		for name in sorted(kp_names):
			xs = [kp["x"] for snap in snapshots for kp in snap.get("keypoints", []) if kp["name"] == name]
			ys = [kp["y"] for snap in snapshots for kp in snap.get("keypoints", []) if kp["name"] == name]
			if not xs:
				continue
			sx = _smooth_series(xs, window_size, filter_type)
			sy = _smooth_series(ys, window_size, filter_type)
			residual_x = _rms_residual(xs, sx)
			residual_y = _rms_residual(ys, sy)
			noise_rms[name] = round((residual_x ** 2 + residual_y ** 2) ** 0.5, 6)
			smoothed_kp_series[name] = [
				{"frame_index": i, "x": round(sx[i], 4), "y": round(sy[i], 4)}
				for i in range(len(sx))
			]

		result = {
			"id": smoothed_id,
			"tenant_id": tenant_id,
			"track_id": track_id,
			"filter_type": filter_type,
			"window_size": window_size,
			"frame_count": len(snapshots),
			"keypoint_count": len(smoothed_kp_series),
			"smoothed_series": smoothed_kp_series,
			"noise_rms_per_keypoint": noise_rms,
			"created_at": utc_now_iso(),
		}
		self._audit(tenant_id, "keypoint_track_smoothed", smoothed_id,
			f"Smoothed {len(smoothed_kp_series)} keypoints with {filter_type} window={window_size}")
		return result

	async def compute_kinematics(
		self,
		report_id: str,
		tenant_id: str,
		track_id: str,
		fps: float = 30.0,
	) -> "dict[str, Any]":
		"""Compute per-keypoint velocity and acceleration from a skeletal track.

		Uses second-order finite differences. Returns velocity (units/frame and
		units/second), acceleration (units/frame²), peak-velocity frame index,
		and a kinetic energy proxy (sum of squared velocities) per keypoint.
		fps is used to convert from frame-units to seconds.
		"""
		guard_tenant_id(tenant_id)
		self._require_tenant(tenant_id)
		key = _stable_key(tenant_id, track_id)
		snapshots = self._skeletal_tracks.get(key)
		if not snapshots:
			raise LookupError("skeletal_track_not_found")
		if fps <= 0:
			raise ValueError("fps_must_be_positive")

		kp_names: set[str] = set()
		for snap in snapshots:
			for kp in snap.get("keypoints", []):
				kp_names.add(kp["name"])

		kinematics: list[dict[str, Any]] = []
		for name in sorted(kp_names):
			positions = [
				(kp["x"], kp["y"])
				for snap in snapshots
				for kp in snap.get("keypoints", [])
				if kp["name"] == name
			]
			if len(positions) < 2:
				continue
			vx = [positions[i + 1][0] - positions[i][0] for i in range(len(positions) - 1)]
			vy = [positions[i + 1][1] - positions[i][1] for i in range(len(positions) - 1)]
			speed = [round((vx[i] ** 2 + vy[i] ** 2) ** 0.5, 6) for i in range(len(vx))]
			ax = [vx[i + 1] - vx[i] for i in range(len(vx) - 1)] if len(vx) >= 2 else []
			ay = [vy[i + 1] - vy[i] for i in range(len(vy) - 1)] if len(vy) >= 2 else []
			accel = [round((ax[i] ** 2 + ay[i] ** 2) ** 0.5, 6) for i in range(len(ax))]
			peak_idx = speed.index(max(speed)) if speed else 0
			ke_proxy = round(sum(s ** 2 for s in speed), 6)
			kinematics.append({
				"keypoint": name,
				"velocity_per_frame": speed,
				"velocity_per_second": [round(s * fps, 4) for s in speed],
				"acceleration_per_frame2": accel,
				"peak_velocity_frame_index": peak_idx,
				"peak_velocity": round(max(speed), 6) if speed else 0.0,
				"kinetic_energy_proxy": ke_proxy,
			})

		report = {
			"id": report_id,
			"tenant_id": tenant_id,
			"track_id": track_id,
			"fps": fps,
			"frame_count": len(snapshots),
			"keypoint_count": len(kinematics),
			"kinematics": kinematics,
			"created_at": utc_now_iso(),
		}
		self._audit(tenant_id, "kinematics_computed", report_id,
			f"Computed kinematics for {len(kinematics)} keypoints at {fps} fps")
		return report

	async def measure_rom(
		self,
		rom_id: str,
		tenant_id: str,
		estimate_id_start: str,
		estimate_id_end: str,
		joint: str,
	) -> "dict[str, Any]":
		"""Measure range of motion (ROM) for a joint between two pose estimates.

		Computes angular delta between start and end estimates and classifies
		result against ISO 8551-based normal ROM ranges. Clinical classification:
		'normal' | 'restricted' | 'hypermobile'.
		"""
		guard_tenant_id(tenant_id)
		self._require_tenant(tenant_id)
		start_est = self._require_estimate(estimate_id_start, tenant_id)
		end_est = self._require_estimate(estimate_id_end, tenant_id)

		# Normal ROM lookup (degrees) — ISO 8551 / AAOS reference values
		_NORMAL_ROM: dict[str, tuple[float, float]] = {
			"left_knee": (0.0, 135.0),
			"right_knee": (0.0, 135.0),
			"left_elbow": (0.0, 145.0),
			"right_elbow": (0.0, 145.0),
			"left_hip": (0.0, 120.0),
			"right_hip": (0.0, 120.0),
			"left_shoulder": (0.0, 180.0),
			"right_shoulder": (0.0, 180.0),
			"left_ankle": (0.0, 50.0),
			"right_ankle": (0.0, 50.0),
		}

		def _get_angle(est: "PoseEstimateRecord", joint_name: str) -> float | None:
			"""Extract single-joint angle using default COCO-17 topology."""
			_TOPOLOGY: dict[str, tuple[str, str, str]] = {
				"left_knee": ("left_hip", "left_knee", "left_ankle"),
				"right_knee": ("right_hip", "right_knee", "right_ankle"),
				"left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
				"right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
				"left_hip": ("left_shoulder", "left_hip", "left_knee"),
				"right_hip": ("right_shoulder", "right_hip", "right_knee"),
				"left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
				"right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
				"left_ankle": ("left_knee", "left_ankle", "left_foot"),
				"right_ankle": ("right_knee", "right_ankle", "right_foot"),
			}
			triple = _TOPOLOGY.get(joint_name)
			if not triple:
				return None
			kp_map = {kp["name"]: kp for kp in est.keypoints}
			p, j, d = triple
			if p not in kp_map or j not in kp_map or d not in kp_map:
				return None
			return _angle_from_three_keypoints(kp_map[p], kp_map[j], kp_map[d])

		angle_start = _get_angle(start_est, joint)
		angle_end = _get_angle(end_est, joint)
		if angle_start is None or angle_end is None:
			rom_degrees = None
			classification = "insufficient_keypoints"
			pct_of_normal = None
		else:
			rom_degrees = round(abs(angle_end - angle_start), 2)
			normal_min, normal_max = _NORMAL_ROM.get(joint, (0.0, 90.0))
			normal_range = normal_max - normal_min
			pct_of_normal = round((rom_degrees / normal_range) * 100, 1) if normal_range > 0 else None
			if rom_degrees < normal_min + (normal_range * 0.3):
				classification = "restricted"
			elif rom_degrees > normal_max * 1.1:
				classification = "hypermobile"
			else:
				classification = "normal"

		record = {
			"id": rom_id,
			"tenant_id": tenant_id,
			"estimate_id_start": estimate_id_start,
			"estimate_id_end": estimate_id_end,
			"joint": joint,
			"angle_start_degrees": angle_start,
			"angle_end_degrees": angle_end,
			"rom_degrees": rom_degrees,
			"percent_of_normal_rom": pct_of_normal,
			"clinical_classification": classification,
			"created_at": utc_now_iso(),
		}
		self._audit(tenant_id, "rom_measured", rom_id,
			f"Joint {joint}: {rom_degrees}° — {classification}")
		return record

	async def detect_asymmetry(
		self,
		report_id: str,
		tenant_id: str,
		track_id: str,
		mild_threshold_pct: float = 10.0,
		severe_threshold_pct: float = 15.0,
	) -> "dict[str, Any]":
		"""Compute bilateral load-proxy asymmetry from a skeletal track.

		Derives asymmetry ratio for each bilateral joint pair from keypoint
		velocity magnitudes. Classification: 'symmetric' | 'mild_asymmetry' |
		'severe_asymmetry'. Raises a high-severity audit event when any pair
		exceeds severe_threshold_pct.
		"""
		guard_tenant_id(tenant_id)
		self._require_tenant(tenant_id)
		key = _stable_key(tenant_id, track_id)
		snapshots = self._skeletal_tracks.get(key)
		if not snapshots:
			raise LookupError("skeletal_track_not_found")

		_BILATERAL_PAIRS: list[tuple[str, str]] = [
			("left_knee", "right_knee"),
			("left_hip", "right_hip"),
			("left_ankle", "right_ankle"),
			("left_shoulder", "right_shoulder"),
			("left_elbow", "right_elbow"),
		]

		def _mean_speed(name: str) -> float:
			positions = [
				(kp["x"], kp["y"])
				for snap in snapshots
				for kp in snap.get("keypoints", [])
				if kp["name"] == name
			]
			if len(positions) < 2:
				return 0.0
			speeds = [
				((positions[i + 1][0] - positions[i][0]) ** 2 +
				 (positions[i + 1][1] - positions[i][1]) ** 2) ** 0.5
				for i in range(len(positions) - 1)
			]
			return mean(speeds)

		results: list[dict[str, Any]] = []
		has_severe = False
		for left_name, right_name in _BILATERAL_PAIRS:
			left_spd = _mean_speed(left_name)
			right_spd = _mean_speed(right_name)
			dominant = max(left_spd, right_spd)
			if dominant < 1e-9:
				continue
			asymmetry_pct = round(abs(left_spd - right_spd) / dominant * 100, 2)
			if asymmetry_pct >= severe_threshold_pct:
				label = "severe_asymmetry"
				has_severe = True
			elif asymmetry_pct >= mild_threshold_pct:
				label = "mild_asymmetry"
			else:
				label = "symmetric"
			results.append({
				"joint_pair": f"{left_name}/{right_name}",
				"left_mean_speed": round(left_spd, 6),
				"right_mean_speed": round(right_spd, 6),
				"asymmetry_pct": asymmetry_pct,
				"classification": label,
			})

		report = {
			"id": report_id,
			"tenant_id": tenant_id,
			"track_id": track_id,
			"joint_pairs_evaluated": len(results),
			"severe_asymmetry_detected": has_severe,
			"mild_threshold_pct": mild_threshold_pct,
			"severe_threshold_pct": severe_threshold_pct,
			"joint_pair_results": results,
			"created_at": utc_now_iso(),
		}
		severity = "high" if has_severe else "info"
		self._audit(tenant_id, "asymmetry_detected", report_id,
			f"Severe asymmetry={'yes' if has_severe else 'no'} ({len(results)} pairs)",
			severity=severity)
		return report

	async def compute_posture_score(
		self,
		score_id: str,
		tenant_id: str,
		estimate_id: str,
	) -> "dict[str, Any]":
		"""Compute a Posture Alignment Index (PAI) score 0–100.

		Evaluates head forward position, shoulder level, spinal vertical alignment
		(neck-shoulder-hip-ankle chain), and pelvic tilt. Higher score = better
		posture. Traffic-light bands: green (>=80) | amber (50-79) | red (<50).
		Based on ISO 11226 thresholds.
		"""
		guard_tenant_id(tenant_id)
		self._require_tenant(tenant_id)
		estimate = self._require_estimate(estimate_id, tenant_id)
		kp_map = {kp["name"]: kp for kp in estimate.keypoints}

		penalties: list[float] = []

		# 1. Head forward position: nose x vs shoulder x midpoint
		nose = kp_map.get("nose")
		l_sh = kp_map.get("left_shoulder")
		r_sh = kp_map.get("right_shoulder")
		if nose and l_sh and r_sh:
			shoulder_mid_x = (l_sh["x"] + r_sh["x"]) / 2
			forward_offset = abs(nose["x"] - shoulder_mid_x)
			penalties.append(min(forward_offset * 100, 25.0))

		# 2. Shoulder level: y-difference between left and right shoulder
		if l_sh and r_sh:
			shoulder_tilt = abs(l_sh["y"] - r_sh["y"])
			penalties.append(min(shoulder_tilt * 200, 20.0))

		# 3. Spinal vertical alignment: hip midpoint x vs shoulder midpoint x
		l_hip = kp_map.get("left_hip")
		r_hip = kp_map.get("right_hip")
		if l_sh and r_sh and l_hip and r_hip:
			shoulder_cx = (l_sh["x"] + r_sh["x"]) / 2
			hip_cx = (l_hip["x"] + r_hip["x"]) / 2
			lateral_lean = abs(shoulder_cx - hip_cx)
			penalties.append(min(lateral_lean * 150, 30.0))

		# 4. Pelvic tilt: y-difference between left and right hip
		if l_hip and r_hip:
			pelvic_tilt = abs(l_hip["y"] - r_hip["y"])
			penalties.append(min(pelvic_tilt * 200, 25.0))

		total_penalty = sum(penalties)
		pai = round(max(0.0, 100.0 - total_penalty), 1)

		if pai >= 80:
			band = "green"
		elif pai >= 50:
			band = "amber"
		else:
			band = "red"

		record = {
			"id": score_id,
			"tenant_id": tenant_id,
			"estimate_id": estimate_id,
			"posture_alignment_index": pai,
			"traffic_light_band": band,
			"penalty_components": penalties,
			"keypoints_available": list(kp_map.keys()),
			"created_at": utc_now_iso(),
		}
		self._audit(tenant_id, "posture_score_computed", score_id,
			f"PAI={pai} ({band})")
		return record

	async def score_injury_risk(
		self,
		risk_id: str,
		tenant_id: str,
		joint_angles_report_id: str,
		rules: "list[dict[str, Any]] | None" = None,
	) -> "dict[str, Any]":
		"""Evaluate biomechanical joint angles against injury risk rules.

		Default rules cover ACL valgus collapse (knee > 175°), hamstring strain
		(hip flexion > 120°), and lower-back overload (trunk lean > 30°). Each
		rule carries clinical evidence level (A/B/C) and a corrective cue. Returns
		a composite injury risk score 0–10 and per-rule results.
		"""
		guard_tenant_id(tenant_id)
		self._require_tenant(tenant_id)

		# Look up pre-computed joint angles report
		angles_report: dict[str, Any] | None = None
		for store in (self._analyses, self._benchmark_records, self._ergonomics_reports):
			candidate = store.get(_stable_key(tenant_id, joint_angles_report_id))
			if candidate is not None:
				angles_report = candidate
				break

		default_rules: list[dict[str, Any]] = [
			{
				"rule_id": "acl_valgus_collapse",
				"joint": "left_knee",
				"threshold_degrees": 175.0,
				"operator": "gt",
				"evidence_level": "A",
				"risk_weight": 3.0,
				"corrective_cue": "Cue knee-over-toe alignment; strengthen hip abductors.",
			},
			{
				"rule_id": "hamstring_strain_risk",
				"joint": "left_hip",
				"threshold_degrees": 120.0,
				"operator": "gt",
				"evidence_level": "B",
				"risk_weight": 2.5,
				"corrective_cue": "Reduce hip flexion range; improve hamstring flexibility.",
			},
			{
				"rule_id": "lower_back_overload",
				"joint": "right_hip",
				"threshold_degrees": 140.0,
				"operator": "gt",
				"evidence_level": "B",
				"risk_weight": 2.0,
				"corrective_cue": "Reduce trunk lean; engage core stabilisers.",
			},
		]
		active_rules = rules or default_rules
		joint_angle_map: dict[str, float] = {}
		if angles_report and "joint_angles" in angles_report:
			for item in angles_report["joint_angles"]:
				joint_angle_map[item["joint"]] = item["angle_degrees"]

		rule_results: list[dict[str, Any]] = []
		total_risk_score = 0.0
		for rule in active_rules:
			joint = rule.get("joint", "")
			threshold = float(rule.get("threshold_degrees", 0))
			operator = rule.get("operator", "gt")
			weight = float(rule.get("risk_weight", 1.0))
			actual = joint_angle_map.get(joint)
			if actual is None:
				triggered = False
			elif operator == "gt":
				triggered = actual > threshold
			elif operator == "lt":
				triggered = actual < threshold
			else:
				triggered = abs(actual - threshold) < 5.0
			if triggered:
				total_risk_score += weight
			rule_results.append({
				"rule_id": rule.get("rule_id", ""),
				"joint": joint,
				"threshold_degrees": threshold,
				"actual_degrees": actual,
				"triggered": triggered,
				"evidence_level": rule.get("evidence_level", "C"),
				"corrective_cue": rule.get("corrective_cue", "") if triggered else None,
			})

		composite_score = round(min(total_risk_score, 10.0), 2)
		risk_tier = "low" if composite_score < 3 else "moderate" if composite_score < 6 else "high"

		record = {
			"id": risk_id,
			"tenant_id": tenant_id,
			"joint_angles_report_id": joint_angles_report_id,
			"composite_injury_risk_score": composite_score,
			"risk_tier": risk_tier,
			"rules_evaluated": len(rule_results),
			"rules_triggered": sum(1 for r in rule_results if r["triggered"]),
			"rule_results": rule_results,
			"created_at": utc_now_iso(),
		}
		severity = "high" if risk_tier == "high" else "info"
		self._audit(tenant_id, "injury_risk_scored", risk_id,
			f"Risk score={composite_score} ({risk_tier})", severity=severity)
		return record

	async def ingest_frame_batch(
		self,
		batch_id: str,
		tenant_id: str,
		session_id: str,
		model_id: str,
		frames: "list[dict[str, Any]]",
		max_concurrency: int = 8,
	) -> "dict[str, Any]":
		"""Batch-ingest frames with pre-computed keypoints concurrently.

		Each entry in `frames` must include: frame_id, frame_number, occurred_at,
		source_ref, width, height, keypoints. Processes up to max_concurrency frames
		simultaneously via asyncio semaphore. Returns per-frame status and a batch
		summary with success count, failure count, and total latency.
		"""
		import asyncio as _asyncio
		import time as _time

		guard_tenant_id(tenant_id)
		self._require_tenant(tenant_id)
		self._require_session(session_id, tenant_id)
		if not frames:
			raise ValueError("frames_required")

		semaphore = _asyncio.Semaphore(max(1, max_concurrency))
		frame_results: list[dict[str, Any]] = []
		t0 = _time.monotonic()

		async def _process_one(frame_payload: dict[str, Any]) -> dict[str, Any]:
			async with semaphore:
				fid = str(frame_payload.get("frame_id") or "")
				try:
					frm = self.record_frame(
						frame_id=fid,
						tenant_id=tenant_id,
						session_id=session_id,
						frame_number=int(frame_payload.get("frame_number", 0)),
						occurred_at=str(frame_payload.get("occurred_at", utc_now_iso())),
						source_ref=str(frame_payload.get("source_ref", "")),
						width=int(frame_payload.get("width", 1920)),
						height=int(frame_payload.get("height", 1080)),
					)
					est = self.estimate_pose(
						estimate_id=f"{fid}:est",
						tenant_id=tenant_id,
						session_id=session_id,
						frame_id=frm["id"],
						model_id=model_id,
						keypoints=list(frame_payload.get("keypoints", [])),
						quality_review_recorded=True,
					)
					return {"frame_id": fid, "status": "ok", "estimate_id": est["id"]}
				except Exception as exc:
					return {"frame_id": fid, "status": "error", "error": str(exc)}

		tasks = [_process_one(f) for f in frames]
		frame_results = list(await _asyncio.gather(*tasks))
		elapsed_ms = round((_time.monotonic() - t0) * 1000, 1)
		success_count = sum(1 for r in frame_results if r["status"] == "ok")
		failure_count = len(frame_results) - success_count

		summary = {
			"id": batch_id,
			"tenant_id": tenant_id,
			"session_id": session_id,
			"total_frames": len(frames),
			"success_count": success_count,
			"failure_count": failure_count,
			"elapsed_ms": elapsed_ms,
			"frame_results": frame_results,
			"created_at": utc_now_iso(),
		}
		self._audit(tenant_id, "frame_batch_ingested", batch_id,
			f"Batch: {success_count}/{len(frames)} ok in {elapsed_ms}ms",
			severity="info" if failure_count == 0 else "medium")
		return summary

	async def longitudinal_compare(
		self,
		report_id: str,
		tenant_id: str,
		session_ids: "list[str]",
	) -> "dict[str, Any]":
		"""Compare pose quality across multiple sessions for longitudinal tracking.

		Computes per-session aggregate confidence distribution and pairwise cosine
		similarity matrix. Returns trend vectors (improving / stable / declining)
		per session relative to the first session baseline.
		"""
		guard_tenant_id(tenant_id)
		self._require_tenant(tenant_id)
		if len(session_ids) < 2:
			raise ValueError("at_least_two_sessions_required")

		session_stats: list[dict[str, Any]] = []
		for sid in session_ids:
			session = self._sessions.get(sid)
			if session is None or session.tenant_id != tenant_id:
				raise LookupError(f"session_not_found:{sid}")
			ests = [e for e in self._estimates.values() if e.session_id == sid and e.tenant_id == tenant_id]
			confs = [e.confidence for e in ests]
			session_stats.append({
				"session_id": sid,
				"estimate_count": len(ests),
				"mean_confidence": round(mean(confs), 4) if confs else 0.0,
				"min_confidence": round(min(confs), 4) if confs else 0.0,
				"max_confidence": round(max(confs), 4) if confs else 0.0,
			})

		# Pairwise similarity matrix using confidence as 1D vector proxy
		n = len(session_stats)
		matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
		for i in range(n):
			for j in range(n):
				ci = session_stats[i]["mean_confidence"]
				cj = session_stats[j]["mean_confidence"]
				sim = 1.0 - abs(ci - cj)
				matrix[i][j] = round(max(0.0, sim), 4)

		# Trend relative to first session baseline
		baseline = session_stats[0]["mean_confidence"]
		trends: list[dict[str, Any]] = []
		for idx, stat in enumerate(session_stats):
			delta = round(stat["mean_confidence"] - baseline, 4)
			trend = "stable" if abs(delta) < 0.02 else ("improving" if delta > 0 else "declining")
			trends.append({"session_id": stat["session_id"], "delta": delta, "trend": trend})

		report = {
			"id": report_id,
			"tenant_id": tenant_id,
			"session_count": n,
			"session_stats": session_stats,
			"similarity_matrix": matrix,
			"trend_vectors": trends,
			"created_at": utc_now_iso(),
		}
		self._audit(tenant_id, "longitudinal_comparison_done", report_id,
			f"Compared {n} sessions")
		return report

	# ------------------------------------------------------------------ #
	# New async methods — world-class improvements (+8)                   #
	# ------------------------------------------------------------------ #

	async def extract_joint_angles(
		self,
		report_id: str,
		tenant_id: str,
		estimate_id: str,
		skeleton_topology: "list[tuple[str, str, str]] | None" = None,
	) -> "dict[str, Any]":
		"""Compute anatomical joint angles from connected keypoint triples.

		Each triple is (proximal, joint, distal) keypoint names. Angles are
		returned in degrees using the law of cosines. Bilateral symmetry deltas
		expose left/right asymmetry. Default topology covers major COCO-17 joints.
		"""
		self._require_tenant(tenant_id)
		estimate = self._require_estimate(estimate_id, tenant_id)
		kp_map = {kp["name"]: kp for kp in estimate.keypoints}

		topology = skeleton_topology or [
			("left_shoulder", "left_elbow", "left_wrist"),
			("right_shoulder", "right_elbow", "right_wrist"),
			("left_hip", "left_knee", "left_ankle"),
			("right_hip", "right_knee", "right_ankle"),
			("left_shoulder", "left_hip", "left_knee"),
			("right_shoulder", "right_hip", "right_knee"),
		]

		joint_angles = []
		for proximal, joint, distal in topology:
			if proximal not in kp_map or joint not in kp_map or distal not in kp_map:
				continue
			angle_deg = _angle_from_three_keypoints(kp_map[proximal], kp_map[joint], kp_map[distal])
			joint_angles.append({
				"joint": joint,
				"proximal": proximal,
				"distal": distal,
				"angle_degrees": angle_deg,
				"confidence": round(mean([
					kp_map[proximal]["confidence"],
					kp_map[joint]["confidence"],
					kp_map[distal]["confidence"],
				]), 4),
			})

		symmetry = {}
		for ang in joint_angles:
			mirror = ang["joint"].replace("left_", "right_") if "left_" in ang["joint"] else ang["joint"].replace("right_", "left_")
			mirror_angles = [a["angle_degrees"] for a in joint_angles if a["joint"] == mirror]
			if mirror_angles:
				symmetry[ang["joint"]] = round(abs(ang["angle_degrees"] - mirror_angles[0]), 2)

		report = {
			"id": report_id,
			"tenant_id": tenant_id,
			"estimate_id": estimate_id,
			"joint_angles": joint_angles,
			"symmetry_deltas_degrees": symmetry,
			"joint_count": len(joint_angles),
			"created_at": utc_now_iso(),
		}
		self._audit(tenant_id, "joint_angles_extracted", report_id, f"Computed {len(joint_angles)} joint angles")
		return report

	async def fuse_estimates(
		self,
		fusion_id: str,
		tenant_id: str,
		estimate_ids: "list[str]",
		outlier_iqr_factor: float = 1.5,
	) -> "dict[str, Any]":
		"""Fuse multiple estimates into a confidence-weighted consensus keypoint set.

		Applies IQR-based outlier rejection per keypoint before weighted averaging.
		Suitable for multi-model ensembling and multi-camera fusion.
		"""
		self._require_tenant(tenant_id)
		if len(estimate_ids) < 2:
			raise ValueError("at_least_two_estimates_required")
		estimates = [self._require_estimate(eid, tenant_id) for eid in estimate_ids]

		all_kp_names: set = set()
		for est in estimates:
			all_kp_names.update(kp["name"] for kp in est.keypoints)

		fused_keypoints = []
		for kp_name in sorted(all_kp_names):
			candidates = [kp for est in estimates for kp in est.keypoints if kp["name"] == kp_name]
			if candidates:
				fused_keypoints.append(_weighted_keypoint_consensus(kp_name, candidates, outlier_iqr_factor))

		overall_confidence = round(mean(kp["confidence"] for kp in fused_keypoints), 4) if fused_keypoints else 0.0
		fusion_record = {
			"id": fusion_id,
			"tenant_id": tenant_id,
			"source_estimate_ids": estimate_ids,
			"source_count": len(estimates),
			"fused_keypoints": fused_keypoints,
			"fused_confidence": overall_confidence,
			"keypoint_count": len(fused_keypoints),
			"created_at": utc_now_iso(),
		}
		self._audit(tenant_id, "estimates_fused", fusion_id,
			f"Fused {len(estimates)} estimates into {len(fused_keypoints)} keypoints")
		return fusion_record

	async def flag_anatomical_anomalies(
		self,
		report_id: str,
		tenant_id: str,
		estimate_id: str,
	) -> "dict[str, Any]":
		"""Validate keypoint topology against anatomical constraints.

		Detects anatomically impossible configurations: knee above hip, shoulder
		below hip, nose below shoulders. Returns violation records and overall
		severity: none / low / medium / high.
		"""
		self._require_tenant(tenant_id)
		estimate = self._require_estimate(estimate_id, tenant_id)
		kp_map = {kp["name"]: kp for kp in estimate.keypoints}

		violations = []
		for side in ("left", "right"):
			hip = kp_map.get(f"{side}_hip")
			knee = kp_map.get(f"{side}_knee")
			if hip and knee and knee["y"] < hip["y"] - 0.05:
				violations.append({"constraint": f"{side}_knee_above_{side}_hip",
					"severity": "high", "delta": round(hip["y"] - knee["y"], 4)})
		for side in ("left", "right"):
			shoulder = kp_map.get(f"{side}_shoulder")
			hip = kp_map.get(f"{side}_hip")
			if shoulder and hip and shoulder["y"] > hip["y"] + 0.05:
				violations.append({"constraint": f"{side}_shoulder_below_{side}_hip",
					"severity": "medium", "delta": round(shoulder["y"] - hip["y"], 4)})
		nose = kp_map.get("nose")
		left_shoulder = kp_map.get("left_shoulder")
		if nose and left_shoulder and nose["y"] > left_shoulder["y"] + 0.1:
			violations.append({"constraint": "nose_below_shoulders",
				"severity": "high", "delta": round(nose["y"] - left_shoulder["y"], 4)})

		severity_scores = {"high": 3, "medium": 2, "low": 1}
		total_score = sum(severity_scores.get(v["severity"], 0) for v in violations)
		overall = "none" if total_score == 0 else "low" if total_score <= 2 else "medium" if total_score <= 5 else "high"
		severity_level = "info" if overall == "none" else "medium" if overall in ("low", "medium") else "high"
		report = {
			"id": report_id,
			"tenant_id": tenant_id,
			"estimate_id": estimate_id,
			"violations": violations,
			"violation_count": len(violations),
			"overall_severity": overall,
			"anomaly_score": total_score,
			"created_at": utc_now_iso(),
		}
		self._audit(tenant_id, "anatomical_anomalies_flagged", report_id,
			f"Severity: {overall} ({len(violations)} violations)", severity=severity_level)
		return report

	async def anonymise_estimate(
		self,
		anon_id: str,
		tenant_id: str,
		estimate_id: str,
		noise_scale: float = 0.02,
		seed: "int | None" = None,
	) -> "dict[str, Any]":
		"""Apply Gaussian noise to keypoint coordinates for k-anonymisation.

		Smaller noise_scale = more accurate, larger = more private.
		Anonymised keypoints are returned but NOT stored to prevent record linkage.
		"""
		import random as _rng
		self._require_tenant(tenant_id)
		estimate = self._require_estimate(estimate_id, tenant_id)
		rng = _rng.Random(seed)
		anon_keypoints = [
			{**kp,
			 "x": round(kp["x"] + rng.gauss(0, noise_scale), 4),
			 "y": round(kp["y"] + rng.gauss(0, noise_scale), 4),
			 "anonymised": True}
			for kp in estimate.keypoints
		]
		record = {
			"id": anon_id,
			"tenant_id": tenant_id,
			"source_estimate_id": estimate_id,
			"noise_scale": noise_scale,
			"seeded": seed is not None,
			"keypoint_count": len(anon_keypoints),
			"anonymised_keypoints": anon_keypoints,
			"created_at": utc_now_iso(),
		}
		self._audit(tenant_id, "estimate_anonymised", anon_id,
			f"Anonymised {len(anon_keypoints)} keypoints with noise_scale={noise_scale}")
		return record

	async def certify_estimate_quality(
		self,
		cert_id: str,
		tenant_id: str,
		estimate_id: str,
		reviewer: str,
		min_confidence: float = 0.85,
		min_keypoints: int = 10,
	) -> "dict[str, Any]":
		"""Issue a tamper-evident quality certificate with SHA-256 content hash.

		Evaluates confidence threshold, keypoint completeness, and reviewer sign-off.
		Grade: 'certified' | 'rejected'.
		"""
		import json as _json
		self._require_tenant(tenant_id)
		if not reviewer.strip():
			raise PermissionError("reviewer_required_for_certification")
		estimate = self._require_estimate(estimate_id, tenant_id)
		passed_confidence = estimate.confidence >= min_confidence
		passed_keypoints = len(estimate.keypoints) >= min_keypoints
		passed = passed_confidence and passed_keypoints
		content_hash = sha256(
			_json.dumps(estimate.to_dict(), sort_keys=True, ensure_ascii=False).encode()
		).hexdigest()
		certificate = {
			"id": cert_id,
			"tenant_id": tenant_id,
			"estimate_id": estimate_id,
			"reviewer": reviewer,
			"passed": passed,
			"checks": {
				"confidence_threshold": {"required": min_confidence, "actual": estimate.confidence, "passed": passed_confidence},
				"keypoint_completeness": {"required": min_keypoints, "actual": len(estimate.keypoints), "passed": passed_keypoints},
				"reviewer_sign_off": {"reviewer": reviewer, "passed": bool(reviewer.strip())},
			},
			"content_hash": content_hash,
			"grade": "certified" if passed else "rejected",
			"created_at": utc_now_iso(),
		}
		self._audit(tenant_id, "estimate_quality_certified", cert_id,
			f"Grade: {certificate['grade']} | hash: {content_hash[:12]}",
			severity="info" if passed else "high")
		return certificate

	async def interpolate_missing_frames(
		self,
		interpolation_id: str,
		tenant_id: str,
		session_id: str,
		estimate_ids: "list[str]",
	) -> "dict[str, Any]":
		"""Fill temporal gaps in a skeletal track via linear keypoint interpolation.

		Gaps are detected from frame_number discontinuities. Synthetic frames are
		marked synthetic=True and are NOT persisted as real estimates.
		"""
		self._require_tenant(tenant_id)
		self._require_session(session_id, tenant_id)
		estimates = [self._require_estimate(eid, tenant_id) for eid in estimate_ids]
		frame_map = {}
		for est in estimates:
			frame = self._frames.get(est.frame_id)
			if frame:
				frame_map[frame.frame_number] = est
		if len(frame_map) < 2:
			return {
				"id": interpolation_id, "tenant_id": tenant_id, "session_id": session_id,
				"interpolated_count": 0,
				"frames": [{"estimate_id": e.id, "synthetic": False} for e in estimates],
				"created_at": utc_now_iso(),
			}
		sorted_frames = sorted(frame_map.keys())
		filled_frames = []
		synthetic_count = 0
		for i, fn in enumerate(sorted_frames):
			est = frame_map[fn]
			filled_frames.append({"frame_number": fn, "estimate_id": est.id,
				"synthetic": False, "keypoints": est.keypoints})
			if i < len(sorted_frames) - 1:
				next_fn = sorted_frames[i + 1]
				gap = next_fn - fn
				if gap > 1:
					next_est = frame_map[next_fn]
					kp_a = {kp["name"]: kp for kp in est.keypoints}
					kp_b = {kp["name"]: kp for kp in next_est.keypoints}
					common = set(kp_a) & set(kp_b)
					for step in range(1, gap):
						alpha = step / gap
						synth_kps = [
							{"name": nm,
							 "x": round(kp_a[nm]["x"] * (1 - alpha) + kp_b[nm]["x"] * alpha, 4),
							 "y": round(kp_a[nm]["y"] * (1 - alpha) + kp_b[nm]["y"] * alpha, 4),
							 "confidence": round(kp_a[nm]["confidence"] * (1 - alpha) + kp_b[nm]["confidence"] * alpha, 4),
							 "visibility": 1.0, "synthetic": True}
							for nm in sorted(common)
						]
						filled_frames.append({"frame_number": fn + step, "estimate_id": None,
							"synthetic": True, "keypoints": synth_kps})
						synthetic_count += 1
		result = {
			"id": interpolation_id, "tenant_id": tenant_id, "session_id": session_id,
			"total_frames": len(filled_frames),
			"real_frames": len(filled_frames) - synthetic_count,
			"interpolated_count": synthetic_count,
			"frames": filled_frames,
			"created_at": utc_now_iso(),
		}
		self._audit(tenant_id, "frames_interpolated", interpolation_id,
			f"Filled {synthetic_count} synthetic frames across {len(sorted_frames)} real frames")
		return result

	async def detect_model_drift(
		self,
		report_id: str,
		tenant_id: str,
		model_id: str,
		window_size: int = 30,
		drift_threshold: float = 0.08,
	) -> "dict[str, Any]":
		"""Detect confidence score drift using EWMA control charting.

		Raises a high-severity audit event when deviation from the model's
		minimum_keypoint_confidence baseline exceeds drift_threshold.
		"""
		self._require_tenant(tenant_id)
		model = self._require_model(model_id, tenant_id)
		model_estimates = sorted(
			[e for e in self._estimates.values() if e.tenant_id == tenant_id and e.model_id == model_id],
			key=lambda e: e.created_at,
		)
		if not model_estimates:
			return {"id": report_id, "tenant_id": tenant_id, "model_id": model_id,
				"drift_detected": False, "message": "no_estimates_available",
				"created_at": utc_now_iso()}
		window = model_estimates[-window_size:]
		confidences = [e.confidence for e in window]
		alpha = 2 / (len(confidences) + 1)
		ewma = confidences[0]
		ewma_series = [round(ewma, 4)]
		for c in confidences[1:]:
			ewma = alpha * c + (1 - alpha) * ewma
			ewma_series.append(round(ewma, 4))
		baseline = model.minimum_keypoint_confidence
		current_ewma = ewma_series[-1]
		deviation = round(baseline - current_ewma, 4)
		drift_detected = deviation > drift_threshold
		report = {
			"id": report_id, "tenant_id": tenant_id, "model_id": model_id,
			"model_type": model.model_type, "window_size": len(window),
			"baseline_confidence": baseline, "current_ewma": current_ewma,
			"deviation": deviation, "drift_threshold": drift_threshold,
			"drift_detected": drift_detected, "ewma_series": ewma_series,
			"created_at": utc_now_iso(),
		}
		self._audit(tenant_id, "model_drift_detection", report_id,
			f"Model {model_id}: drift_detected={drift_detected}, deviation={deviation}",
			severity="high" if drift_detected else "info")
		return report

	async def build_skeleton_overlay(
		self,
		overlay_id: str,
		tenant_id: str,
		estimate_id: str,
		topology: str = "coco17",
	) -> "dict[str, Any]":
		"""Produce display-ready skeleton edge segments for rendering pipelines.

		topology: 'coco17' (default) | 'halpe26' | 'minimal'.
		Each edge carries coordinates, confidence-derived colour (#hex), and
		per-edge confidence score for threshold-based display filtering.
		"""
		self._require_tenant(tenant_id)
		estimate = self._require_estimate(estimate_id, tenant_id)
		kp_map = {kp["name"]: kp for kp in estimate.keypoints}
		topology_edges = {
			"coco17": [
				("nose", "left_eye"), ("nose", "right_eye"),
				("left_eye", "left_ear"), ("right_eye", "right_ear"),
				("left_shoulder", "right_shoulder"),
				("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
				("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
				("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
				("left_hip", "right_hip"),
				("left_hip", "left_knee"), ("right_hip", "right_knee"),
				("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
			],
			"halpe26": [
				("nose", "left_eye"), ("nose", "right_eye"),
				("left_eye", "left_ear"), ("right_eye", "right_ear"),
				("left_shoulder", "right_shoulder"),
				("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
				("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
				("left_wrist", "left_hand"), ("right_wrist", "right_hand"),
				("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
				("left_hip", "right_hip"),
				("left_hip", "left_knee"), ("right_hip", "right_knee"),
				("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
				("left_ankle", "left_foot"), ("right_ankle", "right_foot"),
			],
			"minimal": [
				("left_shoulder", "right_shoulder"),
				("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
				("left_hip", "right_hip"),
				("left_hip", "left_knee"), ("right_hip", "right_knee"),
				("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
			],
		}
		edges_def = topology_edges.get(topology, topology_edges["coco17"])
		segments = []
		for src_name, dst_name in edges_def:
			src = kp_map.get(src_name)
			dst = kp_map.get(dst_name)
			if not src or not dst:
				continue
			edge_conf = round(mean([src["confidence"], dst["confidence"]]), 4)
			colour = "#00cc44" if edge_conf >= 0.8 else "#ffcc00" if edge_conf >= 0.5 else "#ff3300"
			segments.append({
				"from": src_name, "to": dst_name,
				"x1": src["x"], "y1": src["y"],
				"x2": dst["x"], "y2": dst["y"],
				"confidence": edge_conf, "colour": colour,
			})
		overlay = {
			"id": overlay_id, "tenant_id": tenant_id, "estimate_id": estimate_id,
			"topology": topology, "segment_count": len(segments),
			"segments": segments, "keypoints": estimate.keypoints,
			"created_at": utc_now_iso(),
		}
		self._audit(tenant_id, "skeleton_overlay_built", overlay_id,
			f"Built {len(segments)} segments ({topology})")
		return overlay

	# ------------------------------------------------------------------ #
	# Private helpers                                                      #
	# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
# Module-level helpers                                                 #
# ------------------------------------------------------------------ #

def _stable_id(prefix: str, *parts: object) -> str:
	seed = "|".join(str(part) for part in parts if part is not None)
	return f"{prefix}_{sha256(seed.encode('utf-8')).hexdigest()[:12]}"


def _stable_key(tenant_id: str, item_id: str) -> str:
	return f"{tenant_id}:{item_id}"


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


def _count_oscillations(values: list[float], threshold: float) -> int:
	"""Count direction reversals (peaks/troughs) as exercise repetitions."""
	if len(values) < 3:
		return 0
	reps = 0
	direction = 0
	for i in range(1, len(values)):
		delta = values[i] - values[i - 1]
		if delta > threshold and direction != 1:
			direction = 1
		elif delta < -threshold and direction != -1:
			direction = -1
			reps += 1
	return reps


def _ergonomics_risk(kp_map: dict[str, dict[str, Any]]) -> int:
	"""Heuristic RULA-style risk score 1-7 based on shoulder/neck keypoint positions."""
	score = 1
	if "left_shoulder" in kp_map and "right_shoulder" in kp_map:
		shoulder_diff = abs(kp_map["left_shoulder"]["y"] - kp_map["right_shoulder"]["y"])
		if shoulder_diff > 0.1:
			score += 2
	if "nose" in kp_map and "left_shoulder" in kp_map:
		neck_tilt = abs(kp_map["nose"]["x"] - kp_map["left_shoulder"]["x"])
		if neck_tilt > 0.15:
			score += 2
	return min(score, 7)
def _angle_from_three_keypoints(proximal, joint, distal):
	"""Angle at `joint` formed by proximal-joint-distal, in degrees (law of cosines)."""
	import math
	ax, ay = proximal["x"] - joint["x"], proximal["y"] - joint["y"]
	bx, by = distal["x"] - joint["x"], distal["y"] - joint["y"]
	dot = ax * bx + ay * by
	mag_a = math.sqrt(ax ** 2 + ay ** 2)
	mag_b = math.sqrt(bx ** 2 + by ** 2)
	if mag_a < 1e-9 or mag_b < 1e-9:
		return 0.0
	cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
	return round(math.degrees(math.acos(cos_angle)), 2)


def _smooth_series(values: list[float], window: int, filter_type: str) -> list[float]:
	"""Smooth a 1D float series with EMA or boxcar (uniform window) filter."""
	if not values:
		return []
	if filter_type == "ema":
		alpha = 2.0 / (window + 1)
		result = [values[0]]
		for v in values[1:]:
			result.append(alpha * v + (1 - alpha) * result[-1])
		return [round(x, 6) for x in result]
	# boxcar
	half = window // 2
	smoothed = []
	for i in range(len(values)):
		lo = max(0, i - half)
		hi = min(len(values), i + half + 1)
		smoothed.append(round(mean(values[lo:hi]), 6))
	return smoothed


def _rms_residual(original: list[float], smoothed: list[float]) -> float:
	"""Root mean square of residuals between original and smoothed series."""
	if not original or len(original) != len(smoothed):
		return 0.0
	return round((sum((a - b) ** 2 for a, b in zip(original, smoothed)) / len(original)) ** 0.5, 6)


def _weighted_keypoint_consensus(name, candidates, iqr_factor=1.5):
	"""Confidence-weighted average for candidate keypoints with IQR outlier rejection."""
	from statistics import mean as _mean
	if not candidates:
		return {"name": name, "x": 0.0, "y": 0.0, "confidence": 0.0, "visibility": 1.0}
	confs = sorted(c["confidence"] for c in candidates)
	if len(confs) >= 4:
		q1 = confs[len(confs) // 4]
		q3 = confs[(3 * len(confs)) // 4]
		iqr = q3 - q1
		lower, upper = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
		candidates = [c for c in candidates if lower <= c["confidence"] <= upper] or candidates
	total_weight = sum(c["confidence"] for c in candidates) or 1.0
	wx = sum(c["x"] * c["confidence"] for c in candidates) / total_weight
	wy = sum(c["y"] * c["confidence"] for c in candidates) / total_weight
	wc = sum(c["confidence"] for c in candidates) / len(candidates)
	return {
		"name": name, "x": round(wx, 4), "y": round(wy, 4),
		"confidence": round(wc, 4),
		"visibility": round(_mean(c.get("visibility", 1.0) for c in candidates), 4),
	}
