"""
APG Pose Estimation - Views and Data Models
==========================================

Pydantic v2 models for pose estimation API with APG integration patterns.
Follows CLAUDE.md standards: ConfigDict, AfterValidator, modern typing.

Copyright © 2025 Datacraft (nyimbi@gmail.com)
"""

from datetime import datetime
from typing import Optional, Any, Annotated
from enum import Enum
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict, field_validator, AfterValidator
import numpy as np

def _log_validation_operation(operation: str, model: str, **kwargs) -> None:
	"""APG logging pattern for validation operations"""
	print(f"[POSE_VIEWS] {operation} - {model}: {kwargs}")

# Validation functions following APG patterns
def validate_confidence_score(v: float) -> float:
	"""Validate confidence score is between 0 and 1"""
	if not 0.0 <= v <= 1.0:
		raise ValueError(f"Confidence score must be between 0 and 1, got {v}")
	return v

def validate_positive_number(v: float) -> float:
	"""Validate number is positive"""
	if v < 0:
		raise ValueError(f"Value must be positive, got {v}")
	return v

def validate_keypoint_coordinates(v: float) -> float:
	"""Validate keypoint coordinates are reasonable"""
	if not -10000 <= v <= 10000:
		raise ValueError(f"Coordinate value out of reasonable range: {v}")
	return v

def validate_frame_number(v: int) -> int:
	"""Validate frame number is non-negative"""
	if v < 0:
		raise ValueError(f"Frame number must be non-negative, got {v}")
	return v

def validate_person_count(v: int) -> int:
	"""Validate person count is positive"""
	if v < 0:
		raise ValueError(f"Person count must be non-negative, got {v}")
	return v

# Enums for type safety
class PoseModelTypeEnum(str, Enum):
	"""Pose estimation model types"""
	LIGHTWEIGHT = "lightweight"
	ACCURACY = "accuracy"
	MEDICAL = "medical"
	REALTIME = "realtime"
	MULTI_PERSON = "multi_person"
	EDGE_OPTIMIZED = "edge_optimized"

class KeypointTypeEnum(str, Enum):
	"""Standard COCO pose keypoint types"""
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

class SessionStatusEnum(str, Enum):
	"""Pose tracking session status"""
	ACTIVE = "active"
	PAUSED = "paused"
	COMPLETED = "completed"
	ERROR = "error"

class QualityGradeEnum(str, Enum):
	"""Clinical quality grades"""
	A = "A"  # Medical grade (>95% accuracy)
	B = "B"  # High quality (>85% accuracy)  
	C = "C"  # Good quality (>70% accuracy)
	D = "D"  # Fair quality (>50% accuracy)
	F = "F"  # Poor quality (<50% accuracy)

# Core data models following APG Pydantic v2 patterns
class PoseKeypointRequest(BaseModel):
	"""Individual pose keypoint for API requests"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True
	)
	
	type: KeypointTypeEnum = Field(
		...,
		description="Type of pose keypoint (COCO standard)"
	)
	
	x: Annotated[float, AfterValidator(validate_keypoint_coordinates)] = Field(
		...,
		description="X coordinate in image space",
		ge=-10000,
		le=10000
	)
	
	y: Annotated[float, AfterValidator(validate_keypoint_coordinates)] = Field(
		...,
		description="Y coordinate in image space", 
		ge=-10000,
		le=10000
	)
	
	confidence: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		...,
		description="Detection confidence score (0-1)",
		ge=0.0,
		le=1.0
	)
	
	visibility: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		default=1.0,
		description="Visibility score (0=occluded, 1=visible)",
		ge=0.0,
		le=1.0
	)
	
	# Optional 3D coordinates
	x_3d: Optional[float] = Field(
		default=None,
		description="3D X coordinate in world space"
	)
	
	y_3d: Optional[float] = Field(
		default=None,
		description="3D Y coordinate in world space"
	)
	
	z_3d: Optional[float] = Field(
		default=None,
		description="3D Z coordinate (depth) in world space"
	)

class PoseKeypointResponse(PoseKeypointRequest):
	"""Enhanced keypoint response with processing metadata"""
	
	# Processing quality metrics
	tracking_status: str = Field(
		default="detected",
		description="Tracking status (detected, interpolated, lost)"
	)
	
	temporal_smoothness: Optional[float] = Field(
		default=None,
		description="Temporal consistency score (0-1)"
	)
	
	joint_angle: Optional[float] = Field(
		default=None,
		description="Joint angle in degrees (if applicable)"
	)
	
	velocity: Optional[dict[str, float]] = Field(
		default=None,
		description="Velocity vector {vx, vy, vz}"
	)

class BoundingBoxRequest(BaseModel):
	"""Bounding box for person detection"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	x: Annotated[float, AfterValidator(validate_positive_number)] = Field(
		...,
		description="Top-left X coordinate",
		ge=0
	)
	
	y: Annotated[float, AfterValidator(validate_positive_number)] = Field(
		...,
		description="Top-left Y coordinate", 
		ge=0
	)
	
	width: Annotated[float, AfterValidator(validate_positive_number)] = Field(
		...,
		description="Bounding box width",
		gt=0
	)
	
	height: Annotated[float, AfterValidator(validate_positive_number)] = Field(
		...,
		description="Bounding box height",
		gt=0
	)
	
	confidence: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		default=1.0,
		description="Detection confidence",
		ge=0.0,
		le=1.0
	)

class PoseEstimationRequest(BaseModel):
	"""Request for pose estimation API"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True
	)
	
	# Session identification
	session_id: str = Field(
		...,
		description="Pose tracking session ID",
		min_length=1,
		max_length=255
	)
	
	frame_number: Annotated[int, AfterValidator(validate_frame_number)] = Field(
		...,
		description="Sequential frame number",
		ge=0
	)
	
	# Image data (base64 encoded or URL)
	image_data: Optional[str] = Field(
		default=None,
		description="Base64 encoded image data"
	)
	
	image_url: Optional[str] = Field(
		default=None,
		description="URL to image for processing"
	)
	
	# Processing configuration
	model_preference: Optional[PoseModelTypeEnum] = Field(
		default=None,
		description="Preferred model type for estimation"
	)
	
	confidence_threshold: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		default=0.5,
		description="Minimum confidence threshold",
		ge=0.0,
		le=1.0
	)
	
	max_persons: Annotated[int, AfterValidator(validate_person_count)] = Field(
		default=1,
		description="Maximum number of persons to detect",
		ge=1,
		le=50
	)
	
	# Quality vs speed tradeoffs
	accuracy_priority: bool = Field(
		default=False,
		description="Prioritize accuracy over speed"
	)
	
	enable_3d_reconstruction: bool = Field(
		default=False,
		description="Enable 3D pose reconstruction"
	)
	
	enable_temporal_smoothing: bool = Field(
		default=True,
		description="Apply temporal consistency smoothing"
	)
	
	# Medical/clinical requirements
	medical_grade: bool = Field(
		default=False,
		description="Require medical-grade accuracy"
	)
	
	@field_validator('image_data', 'image_url')
	@classmethod
	def validate_image_source(cls, v, info):
		"""Ensure either image_data or image_url is provided"""
		if info.field_name == 'image_data' and v is None:
			# Check if image_url is provided in the model data
			if not info.data.get('image_url'):
				raise ValueError("Either image_data or image_url must be provided")
		return v

class PoseEstimationResponse(BaseModel):
	"""Response from pose estimation API"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Request identification
	session_id: str = Field(..., description="Source session ID")
	frame_number: int = Field(..., description="Processed frame number")
	estimation_id: str = Field(
		default_factory=uuid7str,
		description="Unique estimation ID"
	)
	
	# Processing results
	success: bool = Field(..., description="Whether estimation succeeded")
	error_message: Optional[str] = Field(
		default=None,
		description="Error message if processing failed"
	)
	
	# Pose data
	keypoints_2d: list[PoseKeypointResponse] = Field(
		default_factory=list,
		description="2D pose keypoints"
	)
	
	keypoints_3d: Optional[list[PoseKeypointResponse]] = Field(
		default=None,
		description="3D pose keypoints (if enabled)"
	)
	
	person_count: Annotated[int, AfterValidator(validate_person_count)] = Field(
		default=0,
		description="Number of persons detected"
	)
	
	bounding_boxes: Optional[list[BoundingBoxRequest]] = Field(
		default=None,
		description="Person bounding boxes"
	)
	
	# Quality metrics
	overall_confidence: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		default=0.0,
		description="Overall pose confidence",
		ge=0.0,
		le=1.0
	)
	
	tracking_quality: Optional[float] = Field(
		default=None,
		description="Tracking quality score"
	)
	
	temporal_consistency: Optional[float] = Field(
		default=None,
		description="Temporal consistency score"
	)
	
	occlusion_level: Optional[float] = Field(
		default=None,
		description="Level of occlusion detected"
	)
	
	# Processing metadata
	model_used: str = Field(..., description="Model used for estimation")
	model_version: str = Field(default="1.0.0", description="Model version")
	processing_time_ms: Annotated[float, AfterValidator(validate_positive_number)] = Field(
		...,
		description="Processing time in milliseconds"
	)
	
	timestamp: datetime = Field(
		default_factory=datetime.utcnow,
		description="Processing timestamp"
	)
	
	# APG integration metadata
	tenant_id: str = Field(default="default", description="APG tenant ID")
	processing_node: Optional[str] = Field(
		default=None,
		description="Processing node identifier"
	)

class RealTimeTrackingRequest(BaseModel):
	"""Request for real-time pose tracking"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True
	)
	
	session_id: str = Field(
		...,
		description="Tracking session ID",
		min_length=1
	)
	
	person_id: str = Field(
		...,
		description="Person identifier for tracking",
		min_length=1
	)
	
	# Tracking configuration
	prediction_horizon: int = Field(
		default=3,
		description="Number of frames to predict ahead",
		ge=1,
		le=10
	)
	
	smoothing_factor: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		default=0.7,
		description="Temporal smoothing factor",
		ge=0.0,
		le=1.0
	)
	
	enable_kalman_filter: bool = Field(
		default=True,
		description="Enable Kalman filtering for tracking"
	)
	
	lost_track_threshold: int = Field(
		default=5,
		description="Frames before considering track lost",
		ge=1,
		le=30
	)

class RealTimeTrackingResponse(BaseModel):
	"""Response for real-time tracking status"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	session_id: str = Field(..., description="Tracking session ID")
	person_id: str = Field(..., description="Person identifier")
	
	# Current tracking state
	is_active: bool = Field(..., description="Whether tracking is active")
	last_seen_frame: int = Field(..., description="Last frame with detection")
	tracking_streak: int = Field(..., description="Consecutive successful frames")
	missed_frames: int = Field(..., description="Recent missed detections")
	
	# Current pose
	current_pose: list[PoseKeypointResponse] = Field(
		default_factory=list,
		description="Current pose keypoints"
	)
	
	predicted_pose: Optional[list[PoseKeypointResponse]] = Field(
		default=None,
		description="Predicted next frame pose"
	)
	
	# Quality metrics
	tracking_confidence: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		...,
		description="Tracking confidence",
		ge=0.0,
		le=1.0
	)
	
	average_confidence: Optional[float] = Field(
		default=None,
		description="Average confidence over tracking period"
	)
	
	pose_similarity_score: Optional[float] = Field(
		default=None,
		description="Pose similarity with previous frames"
	)
	
	# Performance metrics
	last_update: datetime = Field(
		default_factory=datetime.utcnow,
		description="Last tracking update timestamp"
	)
	
	update_frequency_hz: Optional[float] = Field(
		default=None,
		description="Tracking update frequency"
	)

class BiomechanicalAnalysisRequest(BaseModel):
	"""Request for biomechanical analysis"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True
	)
	
	estimation_id: str = Field(
		...,
		description="Pose estimation ID to analyze",
		min_length=1
	)
	
	analysis_type: str = Field(
		default="comprehensive",
		description="Type of biomechanical analysis"
	)
	
	# Analysis configuration
	include_joint_angles: bool = Field(
		default=True,
		description="Calculate joint angles"
	)
	
	include_gait_analysis: bool = Field(
		default=False,
		description="Perform gait analysis (requires motion)"
	)
	
	include_balance_metrics: bool = Field(
		default=True,
		description="Calculate balance and stability metrics"
	)
	
	clinical_accuracy_required: bool = Field(
		default=False,
		description="Require clinical-grade accuracy"
	)
	
	# Reference standards
	patient_height_cm: Optional[float] = Field(
		default=None,
		description="Patient height for normalization",
		gt=50,
		lt=250
	)
	
	patient_age: Optional[int] = Field(
		default=None,
		description="Patient age for reference norms",
		ge=0,
		le=120
	)

class JointAngleData(BaseModel):
	"""Joint angle measurement data"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	joint_name: str = Field(..., description="Joint identifier")
	angle_degrees: float = Field(..., description="Joint angle in degrees")
	angle_velocity: Optional[float] = Field(
		default=None,
		description="Angular velocity (degrees/second)"
	)
	
	measurement_uncertainty: Optional[float] = Field(
		default=None,
		description="Measurement uncertainty (±degrees)"
	)
	
	normal_range_min: Optional[float] = Field(
		default=None,
		description="Normal range minimum"
	)
	
	normal_range_max: Optional[float] = Field(
		default=None,
		description="Normal range maximum"
	)
	
	clinical_significance: Optional[str] = Field(
		default=None,
		description="Clinical interpretation"
	)

class GaitMetrics(BaseModel):
	"""Gait analysis metrics"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Temporal parameters
	cadence_steps_per_min: Optional[float] = Field(
		default=None,
		description="Walking cadence (steps per minute)"
	)
	
	step_length_m: Optional[float] = Field(
		default=None,
		description="Average step length (meters)"
	)
	
	stride_length_m: Optional[float] = Field(
		default=None,
		description="Average stride length (meters)"
	)
	
	# Gait cycle phases
	stance_phase_percent: Optional[float] = Field(
		default=None,
		description="Stance phase percentage of gait cycle"
	)
	
	swing_phase_percent: Optional[float] = Field(
		default=None,
		description="Swing phase percentage of gait cycle"
	)
	
	# Symmetry measures
	left_right_symmetry: Optional[float] = Field(
		default=None,
		description="Left-right symmetry score (0-1)"
	)
	
	variability_score: Optional[float] = Field(
		default=None,
		description="Gait variability score"
	)

class BalanceMetrics(BaseModel):
	"""Balance and postural stability metrics"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	center_of_mass: dict[str, float] = Field(
		...,
		description="Center of mass coordinates {x, y, z}"
	)
	
	postural_sway_mm: Optional[float] = Field(
		default=None,
		description="Postural sway magnitude (millimeters)"
	)
	
	stability_index: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		default=0.5,
		description="Overall stability index (0-1)",
		ge=0.0,
		le=1.0
	)
	
	weight_distribution: Optional[dict[str, float]] = Field(
		default=None,
		description="Weight distribution {left, right}"
	)
	
	balance_confidence: Optional[float] = Field(
		default=None,
		description="Balance assessment confidence"
	)

class BiomechanicalAnalysisResponse(BaseModel):
	"""Response from biomechanical analysis"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Request identification
	estimation_id: str = Field(..., description="Source estimation ID")
	analysis_id: str = Field(
		default_factory=uuid7str,
		description="Unique analysis ID"
	)
	
	# Analysis results
	success: bool = Field(..., description="Whether analysis succeeded")
	error_message: Optional[str] = Field(
		default=None,
		description="Error message if analysis failed"
	)
	
	# Biomechanical data
	joint_angles: list[JointAngleData] = Field(
		default_factory=list,
		description="Joint angle measurements"
	)
	
	gait_metrics: Optional[GaitMetrics] = Field(
		default=None,
		description="Gait analysis results"
	)
	
	balance_metrics: Optional[BalanceMetrics] = Field(
		default=None,
		description="Balance and stability metrics"
	)
	
	# Clinical assessment
	clinical_accuracy: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		default=0.0,
		description="Clinical accuracy score (0-1)",
		ge=0.0,
		le=1.0
	)
	
	quality_grade: QualityGradeEnum = Field(
		default=QualityGradeEnum.C,
		description="Clinical quality grade"
	)
	
	measurement_uncertainty: Optional[float] = Field(
		default=None,
		description="Overall measurement uncertainty"
	)
	
	# Risk assessment
	asymmetry_score: Optional[float] = Field(
		default=None,
		description="Bilateral asymmetry score"
	)
	
	compensation_patterns: Optional[list[str]] = Field(
		default=None,
		description="Detected movement compensation patterns"
	)
	
	risk_factors: Optional[list[str]] = Field(
		default=None,
		description="Identified injury risk factors"
	)
	
	# Processing metadata
	analysis_timestamp: datetime = Field(
		default_factory=datetime.utcnow,
		description="Analysis completion timestamp"
	)
	
	processing_time_ms: Annotated[float, AfterValidator(validate_positive_number)] = Field(
		...,
		description="Analysis processing time"
	)

class PoseSessionCreateRequest(BaseModel):
	"""Request to create new pose tracking session"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True
	)
	
	name: str = Field(
		...,
		description="Session name",
		min_length=1,
		max_length=255
	)
	
	description: Optional[str] = Field(
		default=None,
		description="Session description",
		max_length=1000
	)
	
	# Configuration
	target_fps: int = Field(
		default=30,
		description="Target frames per second",
		ge=1,
		le=120
	)
	
	max_persons: Annotated[int, AfterValidator(validate_person_count)] = Field(
		default=1,
		description="Maximum persons to track",
		ge=1,
		le=50
	)
	
	# Input configuration
	input_source: str = Field(
		default="camera",
		description="Input source type (camera, video_file, rtmp)"
	)
	
	input_config: Optional[dict[str, Any]] = Field(
		default=None,
		description="Source-specific configuration"
	)
	
	# Model preferences
	model_preferences: Optional[dict[str, Any]] = Field(
		default=None,
		description="Model selection preferences"
	)
	
	quality_settings: Optional[dict[str, Any]] = Field(
		default=None,
		description="Quality vs speed tradeoffs"
	)
	
	# Collaboration settings
	is_public: bool = Field(
		default=False,
		description="Whether session is publicly accessible"
	)
	
	collaborators: Optional[list[str]] = Field(
		default=None,
		description="List of collaborator user IDs"
	)
	
	# Output configuration
	save_frames: bool = Field(
		default=False,
		description="Save processed frames"
	)
	
	save_3d_data: bool = Field(
		default=False,
		description="Save 3D pose data"
	)

class PoseSessionResponse(BaseModel):
	"""Response for pose session operations"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Session identification
	session_id: str = Field(..., description="Unique session ID")
	tenant_id: str = Field(..., description="APG tenant ID")
	
	# Session metadata
	name: str = Field(..., description="Session name")
	description: Optional[str] = Field(default=None, description="Session description")
	status: SessionStatusEnum = Field(..., description="Current session status")
	
	# Timing information
	created_at: datetime = Field(..., description="Session creation timestamp")
	started_at: Optional[datetime] = Field(default=None, description="Session start time")
	ended_at: Optional[datetime] = Field(default=None, description="Session end time")
	duration_seconds: Optional[float] = Field(default=None, description="Session duration")
	
	# Configuration
	target_fps: int = Field(..., description="Target frame rate")
	max_persons: int = Field(..., description="Maximum persons")
	
	# Performance metrics
	total_frames: int = Field(default=0, description="Total frames processed")
	successful_frames: int = Field(default=0, description="Successfully processed frames")
	average_fps: Optional[float] = Field(default=None, description="Actual average FPS")
	average_latency_ms: Optional[float] = Field(default=None, description="Average processing latency")
	
	# Access control
	created_by: str = Field(..., description="Session creator user ID")
	is_public: bool = Field(..., description="Public accessibility")
	collaborators: Optional[list[str]] = Field(default=None, description="Collaborator list")

class ModelPerformanceResponse(BaseModel):
	"""Model performance metrics response"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	model_name: str = Field(..., description="Model identifier")
	model_type: PoseModelTypeEnum = Field(..., description="Model type")
	model_version: str = Field(..., description="Model version")
	
	# Performance metrics
	avg_inference_time_ms: Annotated[float, AfterValidator(validate_positive_number)] = Field(
		...,
		description="Average inference time"
	)
	
	avg_accuracy_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		...,
		description="Average accuracy score"
	)
	
	memory_usage_mb: Annotated[float, AfterValidator(validate_positive_number)] = Field(
		...,
		description="Memory usage in MB"
	)
	
	# Usage statistics
	total_inferences: int = Field(..., description="Total inferences performed")
	success_rate: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		...,
		description="Success rate"
	)
	
	# Environmental performance
	performance_by_scenario: Optional[dict[str, float]] = Field(
		default=None,
		description="Performance metrics by scenario"
	)

# Export for APG integration
__all__ = [
	# Enums
	"PoseModelTypeEnum",
	"KeypointTypeEnum", 
	"SessionStatusEnum",
	"QualityGradeEnum",
	
	# Request models
	"PoseKeypointRequest",
	"BoundingBoxRequest",
	"PoseEstimationRequest",
	"RealTimeTrackingRequest",
	"BiomechanicalAnalysisRequest",
	"PoseSessionCreateRequest",
	
	# Response models
	"PoseKeypointResponse",
	"PoseEstimationResponse",
	"RealTimeTrackingResponse", 
	"BiomechanicalAnalysisResponse",
	"PoseSessionResponse",
	"ModelPerformanceResponse",
	
	# Data structures
	"JointAngleData",
	"GaitMetrics",
	"BalanceMetrics",
	
	# Validation functions
	"validate_confidence_score",
	"validate_positive_number",
	"validate_keypoint_coordinates",
	"validate_frame_number",
	"validate_person_count"
]