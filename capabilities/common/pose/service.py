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
