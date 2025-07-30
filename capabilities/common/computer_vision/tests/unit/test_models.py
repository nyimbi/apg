"""
Computer Vision & Visual Intelligence - Unit Tests for Models

Unit tests for Pydantic data models including validation, serialization,
and business logic methods.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from ...models import (
	CVProcessingJob, CVImageProcessing, CVDocumentAnalysis,
	CVObjectDetection, CVFacialRecognition, CVQualityControl,
	CVModel, CVAnalyticsReport, ProcessingStatus, ProcessingType,
	ContentType, QualityControlType, FacialFeature, AnalysisLevel,
	_validate_confidence_score, _validate_bounding_box, _validate_image_dimensions
)


class TestCVProcessingJob:
	"""Test CVProcessingJob model"""
	
	def test_creation_with_defaults(self):
		"""Test job creation with default values"""
		job = CVProcessingJob(
			job_name="Test Job",
			processing_type=ProcessingType.OCR,
			content_type=ContentType.IMAGE,
			input_file_path="/test/image.jpg",
			tenant_id="tenant_123",
			created_by="user_456"
		)
		
		assert job.job_name == "Test Job"
		assert job.processing_type == ProcessingType.OCR
		assert job.status == ProcessingStatus.PENDING
		assert job.progress_percentage == 0.0
		assert job.retry_count == 0
		assert job.max_retries == 3
		assert isinstance(job.created_at, datetime)
		assert job.is_completed == False
		assert job.duration_seconds is None
	
	def test_validation_errors(self):
		"""Test model validation errors"""
		# Test invalid priority
		with pytest.raises(ValueError):
			CVProcessingJob(
				job_name="Test",
				processing_type=ProcessingType.OCR,
				content_type=ContentType.IMAGE,
				input_file_path="/test.jpg",
				tenant_id="tenant",
				created_by="user",
				priority=15  # Invalid priority
			)
		
		# Test empty job name
		with pytest.raises(ValueError):
			CVProcessingJob(
				job_name="",
				processing_type=ProcessingType.OCR,
				content_type=ContentType.IMAGE,
				input_file_path="/test.jpg",
				tenant_id="tenant",
				created_by="user"
			)
	
	def test_completion_logic(self):
		"""Test job completion state logic"""
		job = CVProcessingJob(
			job_name="Test",
			processing_type=ProcessingType.OCR,
			content_type=ContentType.IMAGE,
			input_file_path="/test.jpg",
			tenant_id="tenant",
			created_by="user"
		)
		
		# Test in-progress job
		job.status = ProcessingStatus.PROCESSING
		job.started_at = datetime.utcnow()
		assert job.is_completed == False
		
		# Test completed job
		job.status = ProcessingStatus.COMPLETED
		job.completed_at = datetime.utcnow()
		assert job.is_completed == True
		assert job.duration_seconds > 0


class TestCVImageProcessing:
	"""Test CVImageProcessing model"""
	
	def test_creation_and_calculations(self):
		"""Test image processing model creation and calculations"""
		image_data = {
			"job_id": "job_123",
			"original_filename": "test.jpg",
			"file_path": "/storage/test.jpg",
			"file_size_bytes": 1024000,
			"file_hash": "a" * 64,
			"image_dimensions": {"width": 800, "height": 600},
			"image_format": "JPEG",
			"color_mode": "RGB",
			"processing_type": ProcessingType.OBJECT_DETECTION,
			"confidence_score": 0.85,
			"processing_duration_ms": 150,
			"processing_model": "yolov8n",
			"model_version": "8.0.0",
			"tenant_id": "tenant",
			"created_by": "user"
		}
		
		image_proc = CVImageProcessing(**image_data)
		
		assert image_proc.aspect_ratio == 800 / 600
		assert image_proc.megapixels == 0.48
		assert image_proc.confidence_score == 0.85


class TestCVObjectDetection:
	"""Test CVObjectDetection model"""
	
	def test_detection_with_objects(self):
		"""Test object detection with detected objects"""
		detection_data = {
			"job_id": "job_123",
			"image_id": "img_456",
			"detection_model": "yolov8n",
			"model_version": "8.0.0",
			"detected_objects": [
				{
					"object_id": "obj_1",
					"class_name": "person",
					"class_id": 0,
					"confidence": 0.9,
					"bounding_box": {"x": 100, "y": 200, "width": 150, "height": 200},
					"area_pixels": 30000
				},
				{
					"object_id": "obj_2",
					"class_name": "car",
					"class_id": 2,
					"confidence": 0.75,
					"bounding_box": {"x": 300, "y": 400, "width": 200, "height": 100},
					"area_pixels": 20000
				}
			],
			"total_objects": 2,
			"detection_confidence": 0.825,
			"inference_time_ms": 45,
			"preprocessing_time_ms": 10,
			"postprocessing_time_ms": 5,
			"image_resolution": {"width": 640, "height": 480},
			"tenant_id": "tenant",
			"created_by": "user"
		}
		
		detection = CVObjectDetection(**detection_data)
		
		assert detection.total_processing_time_ms == 60
		assert detection.objects_by_class == {"person": 1, "car": 1}
		assert len(detection.detected_objects) == 2


class TestCVFacialRecognition:
	"""Test CVFacialRecognition model"""
	
	def test_privacy_controls(self):
		"""Test facial recognition privacy controls"""
		facial_data = {
			"job_id": "job_123",
			"image_id": "img_456",
			"face_detection_model": "mtcnn",
			"recognition_model": "facenet",
			"faces_detected": [
				{
					"face_id": "face_1",
					"bounding_box": {"x": 100, "y": 150, "width": 80, "height": 100},
					"confidence": 0.95,
					"landmarks_detected": True
				}
			],
			"total_faces": 1,
			"features_extracted": [FacialFeature.EMOTION, FacialFeature.AGE],
			"anonymized": True,
			"consent_recorded": True,
			"retention_period_days": 30,
			"detection_time_ms": 50,
			"recognition_time_ms": 100,
			"analysis_time_ms": 25,
			"image_quality_score": 0.8,
			"detection_confidence": 0.95,
			"tenant_id": "tenant",
			"created_by": "user"
		}
		
		facial = CVFacialRecognition(**facial_data)
		
		assert facial.anonymized == True
		assert facial.consent_recorded == True
		assert facial.total_processing_time_ms == 175
		assert isinstance(facial.data_retention_expires_at, datetime)


class TestCVQualityControl:
	"""Test CVQualityControl model"""
	
	def test_quality_control_with_defects(self):
		"""Test quality control model with defects"""
		qc_data = {
			"job_id": "job_123",
			"inspection_type": QualityControlType.DEFECT_DETECTION,
			"product_identifier": "PROD-001",
			"inspection_station": "QC-STATION-A",
			"pass_fail_status": "PASS",
			"overall_score": 0.92,
			"defects_detected": [
				{
					"defect_id": "def_1",
					"defect_type": "scratch",
					"severity": "MINOR",
					"confidence": 0.8,
					"location": {"x": 150, "y": 200, "width": 5, "height": 2},
					"description": "Minor surface scratch"
				}
			],
			"defect_count": 1,
			"production_line": "LINE-A",
			"shift_identifier": "SHIFT-1",
			"inspection_duration_ms": 500,
			"ai_model_used": "defect_detector_v1",
			"model_confidence": 0.85,
			"tenant_id": "tenant",
			"created_by": "user"
		}
		
		qc = CVQualityControl(**qc_data)
		
		assert qc.critical_defects_count == 0
		assert qc.inspection_passed == True
		assert qc.defect_count == 1


class TestValidationFunctions:
	"""Test custom validation functions"""
	
	def test_confidence_score_validation(self):
		"""Test confidence score validation"""
		# Valid scores
		assert _validate_confidence_score(0.5) == 0.5
		assert _validate_confidence_score(0.0) == 0.0
		assert _validate_confidence_score(1.0) == 1.0
		
		# Invalid scores
		with pytest.raises(ValueError):
			_validate_confidence_score(-0.1)
		with pytest.raises(ValueError):
			_validate_confidence_score(1.1)
	
	def test_bounding_box_validation(self):
		"""Test bounding box validation"""
		# Valid bounding box
		valid_bbox = {"x": 10, "y": 20, "width": 100, "height": 150}
		assert _validate_bounding_box(valid_bbox) == valid_bbox
		
		# Invalid bounding boxes
		with pytest.raises(ValueError):
			_validate_bounding_box({"x": 10, "y": 20})  # Missing width/height
		
		with pytest.raises(ValueError):
			_validate_bounding_box({"x": -10, "y": 20, "width": 100, "height": 150})  # Negative x
	
	def test_image_dimensions_validation(self):
		"""Test image dimensions validation"""
		# Valid dimensions
		valid_dims = {"width": 800, "height": 600}
		assert _validate_image_dimensions(valid_dims) == valid_dims
		
		# Invalid dimensions
		with pytest.raises(ValueError):
			_validate_image_dimensions({"width": 0, "height": 600})  # Zero width


class TestModelSerialization:
	"""Test model serialization and deserialization"""
	
	def test_job_json_serialization(self):
		"""Test job JSON serialization"""
		job = CVProcessingJob(
			job_name="Serialization Test",
			processing_type=ProcessingType.OCR,
			content_type=ContentType.DOCUMENT,
			input_file_path="/test/doc.pdf",
			tenant_id="tenant_123",
			created_by="user_456"
		)
		
		# Test serialization
		json_data = job.model_dump()
		assert json_data["job_name"] == "Serialization Test"
		assert json_data["processing_type"] == "OCR"
		
		# Test deserialization
		job_copy = CVProcessingJob(**json_data)
		assert job_copy.job_name == job.job_name
		assert job_copy.processing_type == job.processing_type
	
	def test_complex_model_serialization(self):
		"""Test complex model with nested data serialization"""
		detection_data = {
			"job_id": "job_123",
			"image_id": "img_456",
			"detection_model": "yolov8n",
			"model_version": "8.0.0",
			"detected_objects": [
				{
					"object_id": "obj_1",
					"class_name": "person",
					"class_id": 0,
					"confidence": 0.9,
					"bounding_box": {"x": 100, "y": 200, "width": 150, "height": 200},
					"area_pixels": 30000
				}
			],
			"total_objects": 1,
			"detection_confidence": 0.9,
			"inference_time_ms": 45,
			"preprocessing_time_ms": 10,
			"postprocessing_time_ms": 5,
			"image_resolution": {"width": 640, "height": 480},
			"tenant_id": "tenant",
			"created_by": "user"
		}
		
		detection = CVObjectDetection(**detection_data)
		
		# Test serialization preserves nested objects
		json_data = detection.model_dump()
		assert len(json_data["detected_objects"]) == 1
		assert json_data["detected_objects"][0]["class_name"] == "person"
		
		# Test deserialization recreates nested objects
		detection_copy = CVObjectDetection(**json_data)
		assert len(detection_copy.detected_objects) == 1
		assert detection_copy.detected_objects[0]["class_name"] == "person"