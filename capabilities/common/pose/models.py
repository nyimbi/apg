"""Dependency-light domain models for the APG Pose Estimation capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class PoseModelRecord:
	id: str
	tenant_id: str
	name: str
	model_type: str
	owner: str
	policy_ref: str
	minimum_keypoint_confidence: float = 0.72
	edge_ready: bool = False
	status: str = "registered"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"model_type": self.model_type,
			"owner": self.owner,
			"policy_ref": self.policy_ref,
			"minimum_keypoint_confidence": self.minimum_keypoint_confidence,
			"edge_ready": self.edge_ready,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass
class PoseSessionRecord:
	id: str
	tenant_id: str
	name: str
	owner: str
	source_ref: str
	model_id: str
	subject_consent_recorded: bool
	secure_stream: bool
	realtime_stream: bool = False
	sensitive_use: bool = False
	approval_ref: str = ""
	max_persons: int = 1
	status: str = "active"
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)
	updated_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"source_ref": self.source_ref,
			"model_id": self.model_id,
			"subject_consent_recorded": self.subject_consent_recorded,
			"secure_stream": self.secure_stream,
			"realtime_stream": self.realtime_stream,
			"sensitive_use": self.sensitive_use,
			"approval_ref": self.approval_ref,
			"max_persons": self.max_persons,
			"status": self.status,
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
			"updated_at": self.updated_at,
		}


@dataclass
class PoseFrameRecord:
	id: str
	tenant_id: str
	session_id: str
	frame_number: int
	occurred_at: str
	source_ref: str
	width: int
	height: int
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"session_id": self.session_id,
			"frame_number": self.frame_number,
			"occurred_at": self.occurred_at,
			"source_ref": self.source_ref,
			"width": self.width,
			"height": self.height,
			"created_at": self.created_at,
		}


@dataclass
class PoseEstimateRecord:
	id: str
	tenant_id: str
	session_id: str
	frame_id: str
	model_id: str
	keypoints: list[dict[str, Any]]
	person_count: int
	quality_score: float
	confidence: float
	quality_review_recorded: bool = False
	status: str = "estimated"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"session_id": self.session_id,
			"frame_id": self.frame_id,
			"model_id": self.model_id,
			"keypoints": [dict(item) for item in self.keypoints],
			"person_count": self.person_count,
			"quality_score": self.quality_score,
			"confidence": self.confidence,
			"quality_review_recorded": self.quality_review_recorded,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass
class PoseAnalysisRecord:
	id: str
	tenant_id: str
	estimation_id: str
	analysis_type: str
	metrics: dict[str, Any]
	medical_grade: bool = False
	reviewer: str = ""
	status: str = "completed"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"estimation_id": self.estimation_id,
			"analysis_type": self.analysis_type,
			"metrics": dict(self.metrics),
			"medical_grade": self.medical_grade,
			"reviewer": self.reviewer,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass
class PoseReconstructionRecord:
	id: str
	tenant_id: str
	estimation_id: str
	camera_calibration_ref: str
	keypoints_3d: list[dict[str, Any]]
	status: str = "reconstructed"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"estimation_id": self.estimation_id,
			"camera_calibration_ref": self.camera_calibration_ref,
			"keypoints_3d": [dict(item) for item in self.keypoints_3d],
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass
class PoseAgentRecord:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = False
	policy_ref: str = ""
	status: str = "active"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope": self.scope,
			"registered": self.registered,
			"contribution_disclosed": self.contribution_disclosed,
			"policy_ref": self.policy_ref,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass
class PoseAuditEvent:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	severity: str = "info"
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"message": self.message,
			"severity": self.severity,
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
		}


# Compatibility alias for older package callers that import PoseRecord.
PoseRecord = PoseSessionRecord
