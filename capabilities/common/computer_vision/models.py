"""
Computer Vision & Visual Intelligence - Data Models

APG-compatible Pydantic v2 data models for computer vision capability providing
enterprise-grade visual content processing, analysis, and intelligence with
comprehensive validation, multi-tenant support, and audit compliance.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Annotated
from uuid import UUID

from pydantic import (
	BaseModel, Field, ConfigDict, AfterValidator, field_validator,
	computed_field, model_validator
)
from uuid_extensions import uuid7str


# Validation Functions
def _validate_confidence_score(v: float) -> float:
	"""Validate confidence scores are between 0.0 and 1.0"""
	if not (0.0 <= v <= 1.0):
		raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {v}")
	return v


def _validate_bounding_box(v: Dict[str, float]) -> Dict[str, float]:
	"""Validate bounding box coordinates"""
	required_keys = {'x', 'y', 'width', 'height'}
	if not all(key in v for key in required_keys):
		raise ValueError(f"Bounding box must contain keys: {required_keys}")
	
	for key, value in v.items():
		if not isinstance(value, (int, float)) or value < 0:
			raise ValueError(f"Bounding box {key} must be non-negative number")
	
	return v


def _validate_image_dimensions(v: Dict[str, int]) -> Dict[str, int]:
	"""Validate image dimensions are positive integers"""
	required_keys = {'width', 'height'}
	if not all(key in v for key in required_keys):
		raise ValueError(f"Dimensions must contain keys: {required_keys}")
	
	for key, value in v.items():
		if not isinstance(value, int) or value <= 0:
			raise ValueError(f"Image {key} must be positive integer")
	
	return v


def _validate_file_path(v: str) -> str:
	"""Validate file path format and security"""
	if not v or len(v.strip()) == 0:
		raise ValueError("File path cannot be empty")
	
	# Basic security checks
	dangerous_patterns = ['../', '..\\', '/etc/', '/root/', 'C:\\Windows\\']
	for pattern in dangerous_patterns:
		if pattern in v:
			raise ValueError(f"File path contains dangerous pattern: {pattern}")
	
	return v.strip()


def _validate_processing_parameters(v: Dict[str, Any]) -> Dict[str, Any]:
	"""Validate processing parameters structure"""
	if not isinstance(v, dict):
		raise ValueError("Processing parameters must be a dictionary")
	
	# Validate common parameter types
	for key, value in v.items():
		if not isinstance(key, str):
			raise ValueError("Parameter keys must be strings")
		
		# Allow basic JSON-serializable types
		if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
			raise ValueError(f"Parameter {key} has unsupported type: {type(value)}")
	
	return v


# Enums
class ProcessingStatus(str, Enum):
	"""Processing job status enumeration"""
	PENDING = "pending"
	QUEUED = "queued"
	PROCESSING = "processing"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	RETRY = "retry"


class ProcessingType(str, Enum):
	"""Type of computer vision processing"""
	OCR = "ocr"
	OBJECT_DETECTION = "object_detection"
	IMAGE_CLASSIFICATION = "image_classification"
	FACIAL_RECOGNITION = "facial_recognition"
	QUALITY_CONTROL = "quality_control"
	VIDEO_ANALYSIS = "video_analysis"
	DOCUMENT_ANALYSIS = "document_analysis"
	SIMILARITY_SEARCH = "similarity_search"


class ContentType(str, Enum):
	"""Supported content types for processing"""
	IMAGE = "image"
	VIDEO = "video"
	DOCUMENT = "document"
	PDF = "pdf"
	CAMERA_STREAM = "camera_stream"


class QualityControlType(str, Enum):
	"""Quality control inspection types"""
	DEFECT_DETECTION = "defect_detection"
	SURFACE_INSPECTION = "surface_inspection"
	DIMENSIONAL_ANALYSIS = "dimensional_analysis"
	ASSEMBLY_VERIFICATION = "assembly_verification"
	PACKAGING_INSPECTION = "packaging_inspection"
	COMPLIANCE_CHECK = "compliance_check"


class FacialFeature(str, Enum):
	"""Facial recognition feature types"""
	IDENTITY = "identity"
	EMOTION = "emotion"
	AGE = "age"
	GENDER = "gender"
	DEMOGRAPHICS = "demographics"
	LIVENESS = "liveness"


class AnalysisLevel(str, Enum):
	"""Analysis detail level"""
	BASIC = "basic"
	STANDARD = "standard"
	DETAILED = "detailed"
	COMPREHENSIVE = "comprehensive"


# Base Models
class CVBaseModel(BaseModel):
	"""Base model for all computer vision models with APG compliance"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		validate_default=True,
		frozen=False
	)
	
	id: str = Field(default_factory=uuid7str, description="Unique identifier using UUID7")
	tenant_id: str = Field(..., description="Multi-tenant isolation identifier")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: str = Field(..., description="User ID who created this record")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CVProcessingJob(CVBaseModel):
	"""
	Computer Vision Processing Job Model
	
	Manages async processing jobs for all types of computer vision operations
	with comprehensive status tracking, error handling, and audit trails.
	"""
	
	# Core job information
	job_name: str = Field(..., min_length=1, max_length=255, description="Human-readable job name")
	processing_type: ProcessingType = Field(..., description="Type of computer vision processing")
	content_type: ContentType = Field(..., description="Type of content being processed")
	status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Current job status")
	
	# Input/Output
	input_file_path: Annotated[str, AfterValidator(_validate_file_path)] = Field(
		..., description="Path to input file or content"
	)
	output_file_path: Optional[Annotated[str, AfterValidator(_validate_file_path)]] = Field(
		None, description="Path to output file or results"
	)
	
	# Processing configuration
	processing_parameters: Annotated[Dict[str, Any], AfterValidator(_validate_processing_parameters)] = Field(
		default_factory=dict, description="Processing configuration parameters"
	)
	priority: int = Field(default=5, ge=1, le=10, description="Job priority (1=highest, 10=lowest)")
	
	# Timing and performance
	started_at: Optional[datetime] = Field(None, description="Processing start time")
	completed_at: Optional[datetime] = Field(None, description="Processing completion time")
	estimated_duration: Optional[int] = Field(None, ge=0, description="Estimated duration in seconds")
	actual_duration: Optional[int] = Field(None, ge=0, description="Actual duration in seconds")
	
	# Results and errors
	results: Dict[str, Any] = Field(default_factory=dict, description="Processing results")
	error_message: Optional[str] = Field(None, max_length=1000, description="Error message if failed")
	error_details: Dict[str, Any] = Field(default_factory=dict, description="Detailed error information")
	
	# Retry handling
	retry_count: int = Field(default=0, ge=0, le=5, description="Number of retry attempts")
	max_retries: int = Field(default=3, ge=0, le=5, description="Maximum retry attempts")
	
	# Progress tracking
	progress_percentage: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.0, description="Processing progress as percentage"
	)
	progress_message: Optional[str] = Field(None, max_length=255, description="Current progress message")
	
	@computed_field
	@property
	def is_completed(self) -> bool:
		"""Check if job is in completed state"""
		return self.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]
	
	@computed_field
	@property
	def duration_seconds(self) -> Optional[int]:
		"""Calculate actual duration if job is completed"""
		if self.started_at and self.completed_at:
			return int((self.completed_at - self.started_at).total_seconds())
		return None
	
	@model_validator(mode='after')
	def _validate_job_consistency(self):
		"""Validate job state consistency"""
		# Validate completion times
		if self.completed_at and not self.started_at:
			raise ValueError("Job cannot have completion time without start time")
		
		if self.started_at and self.completed_at and self.completed_at < self.started_at:
			raise ValueError("Completion time cannot be before start time")
		
		# Validate status consistency
		if self.status == ProcessingStatus.COMPLETED and not self.completed_at:
			raise ValueError("Completed jobs must have completion time")
		
		if self.status == ProcessingStatus.FAILED and not self.error_message:
			raise ValueError("Failed jobs must have error message")
		
		return self


class CVImageProcessing(CVBaseModel):
	"""
	Image Processing Model
	
	Stores results from image processing operations including object detection,
	classification, OCR, and analysis with comprehensive metadata and validation.
	"""
	
	# Source information
	job_id: str = Field(..., description="Associated processing job ID")
	original_filename: str = Field(..., min_length=1, max_length=255, description="Original filename")
	file_path: Annotated[str, AfterValidator(_validate_file_path)] = Field(
		..., description="File storage path"
	)
	file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
	file_hash: str = Field(..., min_length=32, max_length=128, description="File content hash (SHA-256)")
	
	# Image properties
	image_dimensions: Annotated[Dict[str, int], AfterValidator(_validate_image_dimensions)] = Field(
		..., description="Image width and height in pixels"
	)
	image_format: str = Field(..., max_length=10, description="Image format (JPEG, PNG, etc.)")
	color_mode: str = Field(..., max_length=20, description="Color mode (RGB, RGBA, Grayscale)")
	
	# Processing results
	processing_type: ProcessingType = Field(..., description="Type of processing performed")
	confidence_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="Overall confidence in processing results"
	)
	processing_duration_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
	
	# Analysis results
	detected_objects: List[Dict[str, Any]] = Field(
		default_factory=list, description="Detected objects with bounding boxes"
	)
	classification_results: List[Dict[str, Any]] = Field(
		default_factory=list, description="Image classification results"
	)
	extracted_text: Optional[str] = Field(None, description="OCR extracted text")
	quality_metrics: Dict[str, Any] = Field(
		default_factory=dict, description="Image quality assessment metrics"
	)
	
	# Technical metadata
	exif_data: Dict[str, Any] = Field(default_factory=dict, description="EXIF metadata from image")
	processing_model: str = Field(..., max_length=100, description="AI model used for processing")
	model_version: str = Field(..., max_length=50, description="AI model version")
	
	@field_validator('detected_objects')
	@classmethod
	def _validate_detected_objects(cls, v):
		"""Validate detected objects structure"""
		for obj in v:
			if not isinstance(obj, dict):
				raise ValueError("Each detected object must be a dictionary")
			
			required_fields = {'class_name', 'confidence', 'bounding_box'}
			if not all(field in obj for field in required_fields):
				raise ValueError(f"Detected object must contain: {required_fields}")
			
			# Validate bounding box
			_validate_bounding_box(obj['bounding_box'])
			
			# Validate confidence
			_validate_confidence_score(obj['confidence'])
		
		return v
	
	@computed_field
	@property
	def aspect_ratio(self) -> float:
		"""Calculate image aspect ratio"""
		return self.image_dimensions['width'] / self.image_dimensions['height']
	
	@computed_field
	@property
	def megapixels(self) -> float:
		"""Calculate image size in megapixels"""
		return (self.image_dimensions['width'] * self.image_dimensions['height']) / 1_000_000


class CVDocumentAnalysis(CVBaseModel):
	"""
	Document Analysis Model
	
	Stores results from document processing including OCR, form field extraction,
	layout analysis, and content understanding with validation and structure.
	"""
	
	# Source information
	job_id: str = Field(..., description="Associated processing job ID")
	document_filename: str = Field(..., min_length=1, max_length=255, description="Document filename")
	document_type: str = Field(..., max_length=50, description="Document type (PDF, image, scan)")
	page_count: int = Field(..., ge=1, description="Number of pages in document")
	
	# OCR results
	extracted_text: str = Field(..., description="Full extracted text content")
	text_confidence: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="Overall OCR confidence score"
	)
	language_detected: str = Field(..., max_length=10, description="Detected document language")
	
	# Structure analysis
	page_results: List[Dict[str, Any]] = Field(
		default_factory=list, description="Per-page analysis results"
	)
	form_fields: List[Dict[str, Any]] = Field(
		default_factory=list, description="Extracted form fields and values"
	)
	tables: List[Dict[str, Any]] = Field(
		default_factory=list, description="Extracted table data"
	)
	
	# Content analysis
	document_classification: str = Field(..., max_length=100, description="Document category/type")
	key_entities: List[Dict[str, Any]] = Field(
		default_factory=list, description="Named entities and important information"
	)
	sentiment_analysis: Dict[str, Any] = Field(
		default_factory=dict, description="Document sentiment analysis results"
	)
	
	# Technical details
	processing_duration_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
	ocr_engine: str = Field(..., max_length=50, description="OCR engine used")
	ocr_engine_version: str = Field(..., max_length=20, description="OCR engine version")
	
	@field_validator('form_fields')
	@classmethod
	def _validate_form_fields(cls, v):
		"""Validate form fields structure"""
		for field in v:
			if not isinstance(field, dict):
				raise ValueError("Each form field must be a dictionary")
			
			required_fields = {'field_name', 'field_value', 'confidence', 'field_type'}
			if not all(f in field for f in required_fields):
				raise ValueError(f"Form field must contain: {required_fields}")
			
			# Validate confidence
			_validate_confidence_score(field['confidence'])
		
		return v
	
	@computed_field
	@property
	def word_count(self) -> int:
		"""Calculate word count in extracted text"""
		return len(self.extracted_text.split()) if self.extracted_text else 0
	
	@computed_field
	@property
	def avg_confidence(self) -> float:
		"""Calculate average confidence across all fields"""
		if not self.form_fields:
			return self.text_confidence
		
		total_confidence = sum(field['confidence'] for field in self.form_fields)
		return total_confidence / len(self.form_fields)


class CVObjectDetection(CVBaseModel):
	"""
	Object Detection Model
	
	Stores object detection results with bounding boxes, classifications,
	tracking information, and spatial relationships between detected objects.
	"""
	
	# Source information
	job_id: str = Field(..., description="Associated processing job ID")
	image_id: str = Field(..., description="Associated image processing record ID")
	detection_model: str = Field(..., max_length=100, description="Object detection model used")
	model_version: str = Field(..., max_length=50, description="Model version")
	
	# Detection results
	detected_objects: List[Dict[str, Any]] = Field(..., description="List of detected objects")
	total_objects: int = Field(..., ge=0, description="Total number of objects detected")
	detection_confidence: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="Overall detection confidence"
	)
	
	# Performance metrics
	inference_time_ms: int = Field(..., ge=0, description="Model inference time in milliseconds")
	preprocessing_time_ms: int = Field(..., ge=0, description="Image preprocessing time")
	postprocessing_time_ms: int = Field(..., ge=0, description="Results postprocessing time")
	
	# Analysis metadata
	image_resolution: Annotated[Dict[str, int], AfterValidator(_validate_image_dimensions)] = Field(
		..., description="Image resolution used for detection"
	)
	detection_threshold: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.5, description="Minimum confidence threshold for detections"
	)
	nms_threshold: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.4, description="Non-maximum suppression threshold"
	)
	
	# Spatial analysis
	object_relationships: List[Dict[str, Any]] = Field(
		default_factory=list, description="Spatial relationships between objects"
	)
	scene_analysis: Dict[str, Any] = Field(
		default_factory=dict, description="Overall scene understanding"
	)
	
	@field_validator('detected_objects')
	@classmethod
	def _validate_detected_objects(cls, v):
		"""Validate detected objects with comprehensive checks"""
		for i, obj in enumerate(v):
			if not isinstance(obj, dict):
				raise ValueError(f"Object {i} must be a dictionary")
			
			required_fields = {
				'object_id', 'class_name', 'confidence', 'bounding_box',
				'class_id', 'area_pixels'
			}
			if not all(field in obj for field in required_fields):
				raise ValueError(f"Object {i} must contain: {required_fields}")
			
			# Validate specific fields
			_validate_bounding_box(obj['bounding_box'])
			_validate_confidence_score(obj['confidence'])
			
			if not isinstance(obj['class_id'], int) or obj['class_id'] < 0:
				raise ValueError(f"Object {i} class_id must be non-negative integer")
			
			if not isinstance(obj['area_pixels'], (int, float)) or obj['area_pixels'] <= 0:
				raise ValueError(f"Object {i} area_pixels must be positive number")
		
		return v
	
	@computed_field
	@property
	def total_processing_time_ms(self) -> int:
		"""Calculate total processing time"""
		return self.inference_time_ms + self.preprocessing_time_ms + self.postprocessing_time_ms
	
	@computed_field
	@property
	def objects_by_class(self) -> Dict[str, int]:
		"""Count objects by class name"""
		class_counts = {}
		for obj in self.detected_objects:
			class_name = obj['class_name']
			class_counts[class_name] = class_counts.get(class_name, 0) + 1
		return class_counts


class CVFacialRecognition(CVBaseModel):
	"""
	Facial Recognition Model
	
	Stores facial recognition results including identity verification, emotion analysis,
	demographic estimation, and biometric features with privacy controls and compliance.
	"""
	
	# Source information
	job_id: str = Field(..., description="Associated processing job ID")
	image_id: str = Field(..., description="Associated image processing record ID")
	face_detection_model: str = Field(..., max_length=100, description="Face detection model used")
	recognition_model: str = Field(..., max_length=100, description="Face recognition model used")
	
	# Face detection results
	faces_detected: List[Dict[str, Any]] = Field(..., description="Detected faces with metadata")
	total_faces: int = Field(..., ge=0, description="Total number of faces detected")
	
	# Recognition features
	features_extracted: List[FacialFeature] = Field(
		default_factory=list, description="Types of facial features analyzed"
	)
	
	# Privacy and compliance
	anonymized: bool = Field(default=True, description="Whether biometric data is anonymized")
	consent_recorded: bool = Field(default=False, description="Whether user consent was recorded")
	retention_period_days: int = Field(default=30, ge=1, le=2555, description="Data retention period")
	
	# Processing performance
	detection_time_ms: int = Field(..., ge=0, description="Face detection time")
	recognition_time_ms: int = Field(..., ge=0, description="Face recognition time")
	analysis_time_ms: int = Field(..., ge=0, description="Feature analysis time")
	
	# Quality metrics
	image_quality_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="Input image quality for face analysis"
	)
	detection_confidence: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="Face detection confidence"
	)
	
	@field_validator('faces_detected')
	@classmethod
	def _validate_faces_detected(cls, v):
		"""Validate faces detected structure with privacy considerations"""
		for i, face in enumerate(v):
			if not isinstance(face, dict):
				raise ValueError(f"Face {i} must be a dictionary")
			
			required_fields = {
				'face_id', 'bounding_box', 'confidence', 'landmarks_detected'
			}
			if not all(field in face for field in required_fields):
				raise ValueError(f"Face {i} must contain: {required_fields}")
			
			# Validate bounding box and confidence
			_validate_bounding_box(face['bounding_box'])
			_validate_confidence_score(face['confidence'])
			
			# Validate biometric data handling
			if 'biometric_template' in face and not isinstance(face['biometric_template'], str):
				raise ValueError(f"Face {i} biometric_template must be encrypted string")
		
		return v
	
	@computed_field
	@property
	def total_processing_time_ms(self) -> int:
		"""Calculate total processing time"""
		return self.detection_time_ms + self.recognition_time_ms + self.analysis_time_ms
	
	@computed_field
	@property
	def data_retention_expires_at(self) -> datetime:
		"""Calculate when biometric data should be deleted"""
		return self.created_at + timedelta(days=self.retention_period_days)


class CVQualityControl(CVBaseModel):
	"""
	Quality Control Model
	
	Stores manufacturing quality control inspection results including defect detection,
	dimensional analysis, compliance verification, and production line integration.
	"""
	
	# Source information
	job_id: str = Field(..., description="Associated processing job ID")
	inspection_type: QualityControlType = Field(..., description="Type of quality inspection")
	product_identifier: str = Field(..., max_length=100, description="Product or batch identifier")
	inspection_station: str = Field(..., max_length=50, description="Inspection station or line")
	
	# Inspection results
	pass_fail_status: str = Field(..., regex="^(PASS|FAIL|WARNING|REVIEW)$", description="Overall inspection result")
	overall_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="Overall quality score (0-1)"
	)
	
	# Defect analysis
	defects_detected: List[Dict[str, Any]] = Field(
		default_factory=list, description="Detected defects with classifications"
	)
	defect_count: int = Field(..., ge=0, description="Total number of defects found")
	severity_distribution: Dict[str, int] = Field(
		default_factory=dict, description="Count of defects by severity level"
	)
	
	# Measurements and tolerances
	dimensional_measurements: List[Dict[str, Any]] = Field(
		default_factory=list, description="Dimensional measurements and tolerances"
	)
	specification_compliance: Dict[str, Any] = Field(
		default_factory=dict, description="Compliance with product specifications"
	)
	
	# Production context
	production_line: str = Field(..., max_length=50, description="Production line identifier")
	shift_identifier: str = Field(..., max_length=20, description="Production shift")
	operator_id: Optional[str] = Field(None, max_length=50, description="Operator identifier")
	
	# Processing details
	inspection_duration_ms: int = Field(..., ge=0, description="Inspection processing time")
	ai_model_used: str = Field(..., max_length=100, description="AI model for defect detection")
	model_confidence: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="AI model confidence in results"
	)
	
	# Compliance and traceability
	regulatory_standards: List[str] = Field(
		default_factory=list, description="Applicable regulatory standards"
	)
	inspection_certificate_id: Optional[str] = Field(
		None, max_length=100, description="Quality certificate identifier"
	)
	
	@field_validator('defects_detected')
	@classmethod
	def _validate_defects_detected(cls, v):
		"""Validate defect detection results"""
		for i, defect in enumerate(v):
			if not isinstance(defect, dict):
				raise ValueError(f"Defect {i} must be a dictionary")
			
			required_fields = {
				'defect_id', 'defect_type', 'severity', 'confidence',
				'location', 'description'
			}
			if not all(field in defect for field in required_fields):
				raise ValueError(f"Defect {i} must contain: {required_fields}")
			
			# Validate severity levels
			valid_severities = {'CRITICAL', 'MAJOR', 'MINOR', 'COSMETIC'}
			if defect['severity'] not in valid_severities:
				raise ValueError(f"Defect {i} severity must be one of: {valid_severities}")
			
			# Validate confidence
			_validate_confidence_score(defect['confidence'])
		
		return v
	
	@computed_field
	@property
	def critical_defects_count(self) -> int:
		"""Count critical defects that require immediate attention"""
		return len([d for d in self.defects_detected if d.get('severity') == 'CRITICAL'])
	
	@computed_field
	@property
	def inspection_passed(self) -> bool:
		"""Determine if inspection passed based on status"""
		return self.pass_fail_status == 'PASS'


class CVModel(CVBaseModel):
	"""
	Computer Vision Model Registry
	
	Tracks AI models used for computer vision processing including versions,
	performance metrics, training data, and deployment configurations.
	"""
	
	# Model identification
	model_name: str = Field(..., min_length=1, max_length=100, description="Human-readable model name")
	model_type: ProcessingType = Field(..., description="Type of processing this model performs")
	model_version: str = Field(..., max_length=50, description="Model version identifier")
	model_framework: str = Field(..., max_length=50, description="ML framework (PyTorch, TensorFlow, etc.)")
	
	# Model metadata
	model_description: str = Field(..., max_length=1000, description="Detailed model description")
	model_architecture: str = Field(..., max_length=100, description="Model architecture (YOLO, ResNet, etc.)")
	input_requirements: Dict[str, Any] = Field(..., description="Input format and requirements")
	output_format: Dict[str, Any] = Field(..., description="Output format specification")
	
	# Performance metrics
	accuracy_score: Optional[Annotated[float, AfterValidator(_validate_confidence_score)]] = Field(
		None, description="Model accuracy on test set"
	)
	precision_score: Optional[Annotated[float, AfterValidator(_validate_confidence_score)]] = Field(
		None, description="Model precision score"
	)
	recall_score: Optional[Annotated[float, AfterValidator(_validate_confidence_score)]] = Field(
		None, description="Model recall score"
	)
	f1_score: Optional[Annotated[float, AfterValidator(_validate_confidence_score)]] = Field(
		None, description="Model F1 score"
	)
	
	# Training information
	training_dataset_size: Optional[int] = Field(None, ge=0, description="Size of training dataset")
	training_duration_hours: Optional[float] = Field(None, ge=0, description="Training time in hours")
	training_completed_at: Optional[datetime] = Field(None, description="Training completion timestamp")
	
	# Deployment configuration
	model_file_path: Annotated[str, AfterValidator(_validate_file_path)] = Field(
		..., description="Path to model file"
	)
	model_file_size_mb: float = Field(..., ge=0, description="Model file size in MB")
	deployment_status: str = Field(
		default="ACTIVE", regex="^(ACTIVE|INACTIVE|DEPRECATED|TESTING)$",
		description="Model deployment status"
	)
	
	# Resource requirements
	memory_requirements_mb: int = Field(..., ge=0, description="Memory requirements in MB")
	gpu_required: bool = Field(default=False, description="Whether GPU is required")
	inference_time_ms: int = Field(..., ge=0, description="Average inference time in milliseconds")
	
	# Usage statistics
	total_inferences: int = Field(default=0, ge=0, description="Total number of inferences performed")
	last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")
	usage_statistics: Dict[str, Any] = Field(
		default_factory=dict, description="Detailed usage statistics"
	)
	
	@computed_field
	@property
	def model_identifier(self) -> str:
		"""Create unique model identifier"""
		return f"{self.model_name}:{self.model_version}"
	
	@computed_field
	@property
	def is_active(self) -> bool:
		"""Check if model is active and ready for use"""
		return self.deployment_status == "ACTIVE"


class CVAnalyticsReport(CVBaseModel):
	"""
	Computer Vision Analytics Report
	
	Generates business intelligence reports from computer vision processing results
	with KPIs, trends, performance metrics, and actionable insights.
	"""
	
	# Report metadata
	report_name: str = Field(..., min_length=1, max_length=200, description="Report name")
	report_type: str = Field(..., max_length=50, description="Type of analytics report")
	report_period_start: datetime = Field(..., description="Report period start date")
	report_period_end: datetime = Field(..., description="Report period end date")
	
	# Data scope
	processing_types_included: List[ProcessingType] = Field(..., description="Processing types in report")
	total_jobs_analyzed: int = Field(..., ge=0, description="Total jobs included in analysis")
	data_quality_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="Quality score of analyzed data"
	)
	
	# Key performance indicators
	kpis: Dict[str, Any] = Field(..., description="Key performance indicators")
	trends: Dict[str, Any] = Field(..., description="Trend analysis results")
	comparisons: Dict[str, Any] = Field(default_factory=dict, description="Period-over-period comparisons")
	
	# Processing statistics
	total_images_processed: int = Field(..., ge=0, description="Total images processed")
	total_documents_processed: int = Field(..., ge=0, description="Total documents processed")
	total_videos_processed: int = Field(..., ge=0, description="Total videos processed")
	average_processing_time_ms: float = Field(..., ge=0, description="Average processing time")
	
	# Quality metrics
	average_confidence_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="Average confidence across all processing"
	)
	success_rate: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="Processing success rate"
	)
	error_rate: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="Processing error rate"
	)
	
	# Business insights
	insights: List[str] = Field(default_factory=list, description="Generated business insights")
	recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
	cost_analysis: Dict[str, Any] = Field(default_factory=dict, description="Cost analysis and optimization")
	
	# Report generation
	generated_by: str = Field(..., description="User or system that generated report")
	generation_duration_ms: int = Field(..., ge=0, description="Report generation time")
	report_file_path: Optional[Annotated[str, AfterValidator(_validate_file_path)]] = Field(
		None, description="Path to generated report file"
	)
	
	@computed_field
	@property
	def report_period_days(self) -> int:
		"""Calculate report period duration in days"""
		return (self.report_period_end - self.report_period_start).days
	
	@computed_field
	@property
	def processing_volume_per_day(self) -> float:
		"""Calculate average processing volume per day"""
		if self.report_period_days == 0:
			return float(self.total_jobs_analyzed)
		return self.total_jobs_analyzed / self.report_period_days
	
	@model_validator(mode='after')
	def _validate_report_consistency(self):
		"""Validate report data consistency"""
		if self.report_period_end <= self.report_period_start:
			raise ValueError("Report period end must be after start")
		
		if self.success_rate + self.error_rate > 1.01:  # Allow small floating point errors
			raise ValueError("Success rate + error rate cannot exceed 1.0")
		
		return self


# Export all models
__all__ = [
	# Enums
	'ProcessingStatus', 'ProcessingType', 'ContentType', 'QualityControlType',
	'FacialFeature', 'AnalysisLevel',
	
	# Models
	'CVBaseModel', 'CVProcessingJob', 'CVImageProcessing', 'CVDocumentAnalysis',
	'CVObjectDetection', 'CVFacialRecognition', 'CVQualityControl',
	'CVModel', 'CVAnalyticsReport',
	
	# Validation functions (for testing)
	'_validate_confidence_score', '_validate_bounding_box', '_validate_image_dimensions',
	'_validate_file_path', '_validate_processing_parameters'
]