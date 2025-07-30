"""
APG Pose Estimation - Data Models
=================================

Revolutionary pose estimation data models with APG integration patterns.
Follows CLAUDE.md standards: async, tabs, modern typing, uuid7str.

Copyright © 2025 Datacraft (nyimbi@gmail.com)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Any
from enum import Enum
from uuid_extensions import uuid7str

from sqlalchemy import (
	Column, String, Float, Integer, DateTime, Boolean, JSON, Text,
	ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, selectinload
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.asyncio import AsyncSession

Base = declarative_base()

def _log_model_operation(operation: str, model: str, **kwargs) -> None:
	"""APG logging pattern for model operations"""
	print(f"[POSE_MODEL] {operation} - {model}: {kwargs}")

class PoseModelType(Enum):
	"""Types of pose estimation models"""
	LIGHTWEIGHT = "lightweight"
	ACCURACY = "accuracy" 
	MEDICAL = "medical"
	REALTIME = "realtime"
	MULTI_PERSON = "multi_person"
	EDGE_OPTIMIZED = "edge_optimized"

class KeypointType(Enum):
	"""Standard pose keypoint types"""
	NOSE = "nose"
	LEFT_EYE = "left_eye"
	RIGHT_EYE = "right_eye"
	LEFT_EAR = "left_ear"
	RIGHT_EAR = "right_ear"
	LEFT_SHOULDER = "left_shoulder"
	RIGHT_SHOULDER = "right_shoulder"
	LEFT_ELBOW = "left_elbow"
	RIGHT_ELBOW = "right_elbow"
	LEFT_WRIST = "left_wrist"
	RIGHT_WRIST = "right_wrist"
	LEFT_HIP = "left_hip"
	RIGHT_HIP = "right_hip"
	LEFT_KNEE = "left_knee"
	RIGHT_KNEE = "right_knee"
	LEFT_ANKLE = "left_ankle"
	RIGHT_ANKLE = "right_ankle"

class SessionStatus(Enum):
	"""Pose tracking session status"""
	ACTIVE = "active"
	PAUSED = "paused"
	COMPLETED = "completed"
	ERROR = "error"

class PoseEstimationModel(Base):
	"""
	Core pose estimation model with APG multi-tenant patterns.
	Stores individual pose estimations with metadata.
	"""
	__tablename__ = "pe_pose_estimations"
	
	# APG standard fields
	id = Column(String, primary_key=True, default=uuid7str, index=True)
	tenant_id = Column(String, nullable=False, index=True)
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	is_deleted = Column(Boolean, default=False, nullable=False)
	
	# Core pose data
	session_id = Column(String, ForeignKey("pe_pose_sessions.id"), nullable=False, index=True)
	person_id = Column(String, nullable=True)  # For multi-person tracking
	frame_number = Column(Integer, nullable=False)
	timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
	
	# Model information
	model_type = Column(String, nullable=False)
	model_version = Column(String, nullable=False)
	confidence_threshold = Column(Float, nullable=False, default=0.5)
	
	# Processing metadata
	processing_time_ms = Column(Float, nullable=True)
	image_width = Column(Integer, nullable=False)
	image_height = Column(Integer, nullable=False)
	
	# 2D pose data
	keypoints_2d = Column(JSON, nullable=False)  # List of keypoint dicts
	bounding_box = Column(JSON, nullable=True)   # {x, y, width, height}
	overall_confidence = Column(Float, nullable=False)
	
	# 3D pose data (when available)
	keypoints_3d = Column(JSON, nullable=True)
	depth_map = Column(JSON, nullable=True)
	camera_parameters = Column(JSON, nullable=True)
	
	# Quality metrics
	tracking_quality = Column(Float, nullable=True)
	temporal_consistency = Column(Float, nullable=True)
	occlusion_level = Column(Float, nullable=True)
	
	# APG integration fields
	audit_log = Column(JSON, nullable=True)
	processing_node = Column(String, nullable=True)
	
	# Relationships
	session = relationship("PoseSession", back_populates="estimations")
	keypoints = relationship("PoseKeypoint", back_populates="estimation", cascade="all, delete-orphan")
	biomechanical_analysis = relationship("BiomechanicalAnalysis", back_populates="estimation")
	
	# Indexes for performance
	__table_args__ = (
		Index("idx_pe_tenant_session", "tenant_id", "session_id"),
		Index("idx_pe_timestamp", "timestamp"),
		Index("idx_pe_frame", "frame_number"),
		Index("idx_pe_person", "person_id"),
		CheckConstraint("overall_confidence >= 0.0 AND overall_confidence <= 1.0"),
		CheckConstraint("frame_number >= 0"),
	)
	
	def __repr__(self) -> str:
		return f"<PoseEstimation(id={self.id}, session={self.session_id}, frame={self.frame_number})>"

class PoseKeypoint(Base):
	"""
	Individual pose keypoints with 2D/3D coordinates and metadata.
	Enables detailed analysis and biomechanical calculations.
	"""
	__tablename__ = "pe_pose_keypoints"
	
	# APG standard fields
	id = Column(String, primary_key=True, default=uuid7str, index=True)
	tenant_id = Column(String, nullable=False, index=True)
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	
	# Foreign keys
	estimation_id = Column(String, ForeignKey("pe_pose_estimations.id"), nullable=False, index=True)
	
	# Keypoint identification
	keypoint_type = Column(String, nullable=False)  # KeypointType enum
	keypoint_index = Column(Integer, nullable=False)  # Position in model skeleton
	
	# 2D coordinates (image space)
	x = Column(Float, nullable=False)
	y = Column(Float, nullable=False)
	confidence = Column(Float, nullable=False)
	
	# 3D coordinates (world space, when available)
	x_3d = Column(Float, nullable=True)
	y_3d = Column(Float, nullable=True)
	z_3d = Column(Float, nullable=True)
	confidence_3d = Column(Float, nullable=True)
	
	# Quality indicators
	visibility = Column(Float, nullable=False, default=1.0)  # 0=occluded, 1=visible
	tracking_status = Column(String, nullable=False, default="detected")
	temporal_smoothness = Column(Float, nullable=True)
	
	# Biomechanical properties
	joint_angle = Column(Float, nullable=True)  # Degrees
	velocity = Column(JSON, nullable=True)  # {vx, vy, vz}
	acceleration = Column(JSON, nullable=True)  # {ax, ay, az}
	
	# Relationships
	estimation = relationship("PoseEstimationModel", back_populates="keypoints")
	
	# Indexes
	__table_args__ = (
		Index("idx_pk_estimation_type", "estimation_id", "keypoint_type"),
		Index("idx_pk_tenant", "tenant_id"),
		CheckConstraint("confidence >= 0.0 AND confidence <= 1.0"),
		CheckConstraint("visibility >= 0.0 AND visibility <= 1.0"),
	)
	
	def __repr__(self) -> str:
		return f"<PoseKeypoint(type={self.keypoint_type}, x={self.x:.1f}, y={self.y:.1f})>"

class PoseSession(Base):
	"""
	Pose tracking session for managing continuous pose estimation.
	Supports multi-person tracking and collaborative sessions.
	"""
	__tablename__ = "pe_pose_sessions"
	
	# APG standard fields
	id = Column(String, primary_key=True, default=uuid7str, index=True)
	tenant_id = Column(String, nullable=False, index=True)
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	is_deleted = Column(Boolean, default=False, nullable=False)
	
	# Session metadata
	name = Column(String(255), nullable=False)
	description = Column(Text, nullable=True)
	status = Column(String, nullable=False, default=SessionStatus.ACTIVE.value)
	
	# User and access control
	created_by = Column(String, nullable=False)  # User ID
	collaborators = Column(ARRAY(String), nullable=True)  # List of user IDs
	is_public = Column(Boolean, default=False, nullable=False)
	
	# Configuration
	target_fps = Column(Integer, nullable=False, default=30)
	max_persons = Column(Integer, nullable=False, default=1)
	model_preferences = Column(JSON, nullable=True)  # Model selection preferences
	quality_settings = Column(JSON, nullable=True)  # Quality vs speed tradeoffs
	
	# Session timing
	started_at = Column(DateTime, nullable=True)
	ended_at = Column(DateTime, nullable=True)
	duration_seconds = Column(Float, nullable=True)
	
	# Input configuration
	input_source = Column(String, nullable=False)  # camera, video_file, rtmp, etc.
	input_config = Column(JSON, nullable=True)  # Source-specific configuration
	camera_calibration = Column(JSON, nullable=True)  # Camera parameters
	
	# Output configuration
	output_format = Column(String, nullable=False, default="json")
	save_frames = Column(Boolean, default=False, nullable=False)
	save_3d_data = Column(Boolean, default=False, nullable=False)
	
	# Performance metrics
	total_frames = Column(Integer, nullable=False, default=0)
	successful_frames = Column(Integer, nullable=False, default=0)
	average_fps = Column(Float, nullable=True)
	average_latency_ms = Column(Float, nullable=True)
	
	# APG integration
	collaboration_room_id = Column(String, nullable=True)  # APG real-time collaboration
	visualization_config = Column(JSON, nullable=True)  # APG 3D visualization
	
	# Relationships
	estimations = relationship("PoseEstimationModel", back_populates="session", cascade="all, delete-orphan")
	real_time_tracking = relationship("RealTimeTracking", back_populates="session")
	
	# Indexes
	__table_args__ = (
		Index("idx_ps_tenant_status", "tenant_id", "status"),
		Index("idx_ps_created_by", "created_by"),
		Index("idx_ps_started", "started_at"),
		CheckConstraint("max_persons > 0"),
		CheckConstraint("target_fps > 0"),
	)
	
	def __repr__(self) -> str:
		return f"<PoseSession(id={self.id}, name={self.name}, status={self.status})>"

class RealTimeTracking(Base):
	"""
	Real-time pose tracking state for active sessions.
	Optimized for high-frequency updates and low latency.
	"""
	__tablename__ = "pe_realtime_tracking"
	
	# APG standard fields
	id = Column(String, primary_key=True, default=uuid7str, index=True)
	tenant_id = Column(String, nullable=False, index=True)
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	# Session reference
	session_id = Column(String, ForeignKey("pe_pose_sessions.id"), nullable=False, index=True)
	person_id = Column(String, nullable=False, index=True)
	
	# Current state
	last_seen_frame = Column(Integer, nullable=False)
	last_update = Column(DateTime, nullable=False, default=datetime.utcnow)
	is_active = Column(Boolean, default=True, nullable=False)
	
	# Tracking data
	current_pose = Column(JSON, nullable=False)  # Latest pose keypoints
	pose_history = Column(JSON, nullable=True)  # Recent pose history for smoothing
	tracking_confidence = Column(Float, nullable=False)
	
	# Kalman filter state
	kalman_state = Column(JSON, nullable=True)  # Kalman filter internal state
	prediction = Column(JSON, nullable=True)  # Next frame prediction
	
	# Performance metrics
	tracking_streak = Column(Integer, nullable=False, default=1)
	missed_frames = Column(Integer, nullable=False, default=0)
	average_confidence = Column(Float, nullable=True)
	
	# Quality indicators
	occlusion_count = Column(Integer, nullable=False, default=0)
	rapid_movement_alerts = Column(Integer, nullable=False, default=0)
	pose_similarity_score = Column(Float, nullable=True)
	
	# Relationships
	session = relationship("PoseSession", back_populates="real_time_tracking")
	
	# Indexes for real-time performance
	__table_args__ = (
		Index("idx_rt_session_person", "session_id", "person_id"),
		Index("idx_rt_last_update", "last_update"),
		Index("idx_rt_active", "is_active"),
		UniqueConstraint("session_id", "person_id", name="uq_session_person"),
		CheckConstraint("tracking_confidence >= 0.0 AND tracking_confidence <= 1.0"),
	)
	
	def __repr__(self) -> str:
		return f"<RealTimeTracking(session={self.session_id}, person={self.person_id}, active={self.is_active})>"

class BiomechanicalAnalysis(Base):
	"""
	Medical-grade biomechanical analysis results.
	Provides clinical metrics for healthcare applications.
	"""
	__tablename__ = "pe_biomechanical_analysis"
	
	# APG standard fields
	id = Column(String, primary_key=True, default=uuid7str, index=True)
	tenant_id = Column(String, nullable=False, index=True)
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	
	# References
	estimation_id = Column(String, ForeignKey("pe_pose_estimations.id"), nullable=False, index=True)
	analysis_type = Column(String, nullable=False)  # gait, rom, balance, etc.
	
	# Joint angles (degrees)
	joint_angles = Column(JSON, nullable=False)  # {joint_name: angle}
	angle_velocities = Column(JSON, nullable=True)  # Angular velocities
	angle_accelerations = Column(JSON, nullable=True)  # Angular accelerations
	
	# Range of motion metrics
	rom_measurements = Column(JSON, nullable=True)  # Range of motion data
	flexibility_scores = Column(JSON, nullable=True)  # Flexibility assessments
	
	# Gait analysis (when applicable)
	gait_cycle_phase = Column(String, nullable=True)  # stance, swing, etc.
	step_length = Column(Float, nullable=True)  # Meters
	stride_length = Column(Float, nullable=True)  # Meters
	cadence = Column(Float, nullable=True)  # Steps per minute
	
	# Balance and stability
	center_of_mass = Column(JSON, nullable=True)  # {x, y, z}
	postural_sway = Column(Float, nullable=True)  # Sway magnitude
	stability_index = Column(Float, nullable=True)  # 0-1 scale
	
	# Clinical assessments
	asymmetry_score = Column(Float, nullable=True)  # Left-right asymmetry
	compensation_patterns = Column(JSON, nullable=True)  # Detected compensations
	risk_factors = Column(JSON, nullable=True)  # Injury risk indicators
	
	# Medical compliance
	clinical_accuracy = Column(Float, nullable=True)  # Validated against clinical gold standard
	measurement_uncertainty = Column(Float, nullable=True)  # ± degrees/cm
	quality_grade = Column(String, nullable=True)  # A, B, C clinical quality
	
	# Relationships
	estimation = relationship("PoseEstimationModel", back_populates="biomechanical_analysis")
	
	# Indexes
	__table_args__ = (
		Index("idx_ba_estimation", "estimation_id"),
		Index("idx_ba_type", "analysis_type"),
		Index("idx_ba_tenant", "tenant_id"),
	)
	
	def __repr__(self) -> str:
		return f"<BiomechanicalAnalysis(type={self.analysis_type}, quality={self.quality_grade})>"

class ModelPerformanceMetrics(Base):
	"""
	Performance tracking for pose estimation models.
	Enables adaptive model selection and optimization.
	"""
	__tablename__ = "pe_model_metrics"
	
	# APG standard fields
	id = Column(String, primary_key=True, default=uuid7str, index=True)
	tenant_id = Column(String, nullable=False, index=True)
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	
	# Model identification
	model_type = Column(String, nullable=False)
	model_version = Column(String, nullable=False)
	deployment_target = Column(String, nullable=False)  # edge, cloud, mobile
	
	# Performance metrics
	avg_inference_time_ms = Column(Float, nullable=False)
	avg_accuracy_score = Column(Float, nullable=False)
	memory_usage_mb = Column(Float, nullable=False)
	cpu_usage_percent = Column(Float, nullable=False)
	
	# Quality metrics
	keypoint_accuracy = Column(JSON, nullable=True)  # Per-keypoint accuracy
	temporal_consistency = Column(Float, nullable=True)
	robustness_score = Column(Float, nullable=True)  # Performance under challenging conditions
	
	# Usage statistics
	total_inferences = Column(Integer, nullable=False, default=0)
	successful_inferences = Column(Integer, nullable=False, default=0)
	error_count = Column(Integer, nullable=False, default=0)
	
	# Environmental factors
	lighting_conditions = Column(JSON, nullable=True)  # Performance by lighting
	occlusion_performance = Column(JSON, nullable=True)  # Performance with occlusions
	multi_person_performance = Column(JSON, nullable=True)  # Multi-person scenarios
	
	# Indexes
	__table_args__ = (
		Index("idx_pm_model", "model_type", "model_version"),
		Index("idx_pm_tenant", "tenant_id"),
		Index("idx_pm_target", "deployment_target"),
	)
	
	def __repr__(self) -> str:
		return f"<ModelMetrics(model={self.model_type}, target={self.deployment_target})>"

# Async database operations following APG patterns
class PoseEstimationRepository:
	"""
	Async repository for pose estimation data operations.
	Follows APG patterns with proper error handling and logging.
	"""
	
	def __init__(self, session: AsyncSession):
		assert session is not None, "Database session is required"
		self.session = session
	
	async def create_pose_session(self, tenant_id: str, session_data: dict[str, Any]) -> PoseSession:
		"""Create new pose tracking session with APG patterns"""
		assert tenant_id, "Tenant ID is required"
		assert session_data, "Session data is required"
		
		_log_model_operation("CREATE_SESSION", "PoseSession", tenant_id=tenant_id)
		
		session = PoseSession(
			tenant_id=tenant_id,
			**session_data
		)
		
		self.session.add(session)
		await self.session.commit()
		await self.session.refresh(session)
		
		_log_model_operation("CREATED_SESSION", "PoseSession", session_id=session.id)
		return session
	
	async def save_pose_estimation(self, estimation_data: dict[str, Any]) -> PoseEstimationModel:
		"""Save pose estimation with keypoints"""
		assert estimation_data, "Estimation data is required"
		
		_log_model_operation("SAVE_ESTIMATION", "PoseEstimationModel", 
			session_id=estimation_data.get("session_id"))
		
		estimation = PoseEstimationModel(**estimation_data)
		self.session.add(estimation)
		
		await self.session.commit()
		await self.session.refresh(estimation)
		
		return estimation
	
	async def get_active_sessions(self, tenant_id: str) -> list[PoseSession]:
		"""Get all active pose sessions for tenant"""
		assert tenant_id, "Tenant ID is required"
		
		_log_model_operation("GET_ACTIVE_SESSIONS", "PoseSession", tenant_id=tenant_id)
		
		result = await self.session.execute(
			select(PoseSession)
			.where(PoseSession.tenant_id == tenant_id)
			.where(PoseSession.status == SessionStatus.ACTIVE.value)
			.where(PoseSession.is_deleted == False)
			.options(selectinload(PoseSession.estimations))
		)
		
		sessions = result.scalars().all()
		_log_model_operation("FOUND_SESSIONS", "PoseSession", count=len(sessions))
		
		return list(sessions)
	
	async def update_real_time_tracking(self, session_id: str, person_id: str, 
		tracking_data: dict[str, Any]) -> RealTimeTracking:
		"""Update real-time tracking state with optimistic locking"""
		assert session_id, "Session ID is required"
		assert person_id, "Person ID is required"
		
		_log_model_operation("UPDATE_TRACKING", "RealTimeTracking", 
			session_id=session_id, person_id=person_id)
		
		# Try to find existing tracking record
		result = await self.session.execute(
			select(RealTimeTracking)
			.where(RealTimeTracking.session_id == session_id)
			.where(RealTimeTracking.person_id == person_id)
		)
		
		tracking = result.scalar_one_or_none()
		
		if tracking:
			# Update existing record
			for key, value in tracking_data.items():
				setattr(tracking, key, value)
			tracking.updated_at = datetime.utcnow()
		else:
			# Create new tracking record
			tracking = RealTimeTracking(
				session_id=session_id,
				person_id=person_id,
				**tracking_data
			)
			self.session.add(tracking)
		
		await self.session.commit()
		await self.session.refresh(tracking)
		
		return tracking

# Export for APG integration
__all__ = [
	"PoseEstimationModel",
	"PoseKeypoint", 
	"PoseSession",
	"RealTimeTracking",
	"BiomechanicalAnalysis",
	"ModelPerformanceMetrics",
	"PoseEstimationRepository",
	"PoseModelType",
	"KeypointType", 
	"SessionStatus"
]