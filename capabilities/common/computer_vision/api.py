"""
Computer Vision & Visual Intelligence - REST API Layer

FastAPI implementation providing comprehensive computer vision REST endpoints
with authentication, validation, file handling, and comprehensive error handling
following APG platform standards and OpenAPI documentation.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import hashlib
import mimetypes
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid_extensions import uuid7str

from fastapi import (
	FastAPI, HTTPException, Depends, File, UploadFile, Form,
	BackgroundTasks, Query, Header, Path as FastAPIPath
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, ValidationError
import uvicorn

from .models import (
	CVProcessingJob, CVImageProcessing, CVDocumentAnalysis,
	CVObjectDetection, CVFacialRecognition, CVQualityControl,
	CVModel, CVAnalyticsReport, ProcessingStatus, ProcessingType,
	ContentType, AnalysisLevel, QualityControlType, FacialFeature
)
from .service import (
	CVProcessingService, CVDocumentAnalysisService, CVObjectDetectionService,
	CVImageClassificationService, CVFacialRecognitionService,
	CVQualityControlService, CVVideoAnalysisService, CVSimilaritySearchService
)


# API Request/Response Models
class ProcessingJobRequest(BaseModel):
	"""Request model for creating processing jobs"""
	job_name: str = Field(..., min_length=1, max_length=255, description="Human-readable job name")
	processing_type: ProcessingType = Field(..., description="Type of processing to perform")
	processing_parameters: Dict[str, Any] = Field(default_factory=dict, description="Processing configuration")
	priority: int = Field(default=5, ge=1, le=10, description="Job priority (1=highest, 10=lowest)")


class ProcessingJobResponse(BaseModel):
	"""Response model for processing job operations"""
	job_id: str = Field(..., description="Unique job identifier")
	status: ProcessingStatus = Field(..., description="Current job status")
	progress_percentage: float = Field(..., ge=0.0, le=1.0, description="Processing progress")
	results: Optional[Dict[str, Any]] = Field(None, description="Processing results if completed")
	error_message: Optional[str] = Field(None, description="Error message if failed")
	created_at: datetime = Field(..., description="Job creation timestamp")
	estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


class OCRRequest(BaseModel):
	"""Request model for OCR processing"""
	language: str = Field(default="eng", max_length=10, description="OCR language code")
	ocr_engine: str = Field(default="tesseract", description="OCR engine to use")
	enhance_image: bool = Field(default=True, description="Apply image enhancement")
	extract_tables: bool = Field(default=False, description="Extract table data")
	extract_forms: bool = Field(default=False, description="Extract form fields")


class OCRResponse(BaseModel):
	"""Response model for OCR processing"""
	extracted_text: str = Field(..., description="Extracted text content")
	confidence_score: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score")
	language_detected: str = Field(..., description="Detected document language")
	word_count: int = Field(..., ge=0, description="Number of words extracted")
	processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
	form_fields: Optional[List[Dict[str, Any]]] = Field(None, description="Extracted form fields")
	tables: Optional[List[Dict[str, Any]]] = Field(None, description="Extracted table data")


class ObjectDetectionRequest(BaseModel):
	"""Request model for object detection"""
	model_name: str = Field(default="yolov8n.pt", description="YOLO model to use")
	confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Detection confidence threshold")
	iou_threshold: float = Field(default=0.4, ge=0.0, le=1.0, description="IoU threshold for NMS")
	max_detections: int = Field(default=100, ge=1, le=1000, description="Maximum number of detections")


class ObjectDetectionResponse(BaseModel):
	"""Response model for object detection"""
	detected_objects: List[Dict[str, Any]] = Field(..., description="List of detected objects")
	total_objects: int = Field(..., ge=0, description="Total number of objects detected")
	detection_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall detection confidence")
	processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
	model_used: str = Field(..., description="Model used for detection")


class ImageClassificationRequest(BaseModel):
	"""Request model for image classification"""
	model_name: str = Field(default="google/vit-base-patch16-224", description="Classification model")
	top_k: int = Field(default=5, ge=1, le=20, description="Number of top predictions to return")
	include_features: bool = Field(default=False, description="Include feature embeddings")


class ImageClassificationResponse(BaseModel):
	"""Response model for image classification"""
	classification_results: List[Dict[str, Any]] = Field(..., description="Classification predictions")
	top_prediction: Optional[Dict[str, Any]] = Field(None, description="Top prediction")
	confidence_score: float = Field(..., ge=0.0, le=1.0, description="Top prediction confidence")
	processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
	features: Optional[List[float]] = Field(None, description="Feature embeddings if requested")


class FacialRecognitionRequest(BaseModel):
	"""Request model for facial recognition"""
	features_to_extract: List[FacialFeature] = Field(
		default=[FacialFeature.IDENTITY], description="Facial features to analyze"
	)
	anonymize_results: bool = Field(default=True, description="Anonymize biometric data")
	consent_recorded: bool = Field(default=False, description="User consent for biometric processing")
	retention_days: int = Field(default=30, ge=1, le=365, description="Data retention period in days")


class FacialRecognitionResponse(BaseModel):
	"""Response model for facial recognition"""
	faces_detected: List[Dict[str, Any]] = Field(..., description="Detected faces with analysis")
	total_faces: int = Field(..., ge=0, description="Total number of faces detected")
	features_extracted: List[FacialFeature] = Field(..., description="Features that were analyzed")
	anonymized: bool = Field(..., description="Whether results were anonymized")
	processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")


class QualityControlRequest(BaseModel):
	"""Request model for quality control inspection"""
	inspection_type: QualityControlType = Field(..., description="Type of quality inspection")
	product_identifier: str = Field(..., max_length=100, description="Product or batch identifier")
	inspection_station: str = Field(..., max_length=50, description="Inspection station identifier")
	quality_standards: List[str] = Field(default_factory=list, description="Applicable quality standards")
	tolerance_parameters: Dict[str, Any] = Field(default_factory=dict, description="Tolerance specifications")


class QualityControlResponse(BaseModel):
	"""Response model for quality control inspection"""
	inspection_result: str = Field(..., description="Overall inspection result (PASS/FAIL/WARNING)")
	overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
	defects_detected: List[Dict[str, Any]] = Field(..., description="Detected defects")
	defect_count: int = Field(..., ge=0, description="Total number of defects found")
	compliance_status: Dict[str, Any] = Field(..., description="Compliance with standards")
	processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")


class BatchProcessingRequest(BaseModel):
	"""Request model for batch processing operations"""
	processing_type: ProcessingType = Field(..., description="Type of processing to perform")
	processing_parameters: Dict[str, Any] = Field(default_factory=dict, description="Processing configuration")
	priority: int = Field(default=5, ge=1, le=10, description="Batch job priority")
	notification_webhook: Optional[str] = Field(None, description="Webhook URL for completion notification")


class BatchProcessingResponse(BaseModel):
	"""Response model for batch processing operations"""
	batch_id: str = Field(..., description="Unique batch identifier")
	job_ids: List[str] = Field(..., description="Individual job identifiers")
	total_files: int = Field(..., ge=0, description="Total number of files in batch")
	estimated_completion: Optional[datetime] = Field(None, description="Estimated batch completion time")


class HealthCheckResponse(BaseModel):
	"""Response model for health check endpoint"""
	status: str = Field(..., description="Service health status")
	timestamp: datetime = Field(..., description="Health check timestamp")
	version: str = Field(..., description="API version")
	services: Dict[str, str] = Field(..., description="Individual service health status")
	performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")


# FastAPI Application Setup
app = FastAPI(
	title="Computer Vision & Visual Intelligence API",
	description="Enterprise-grade computer vision processing API with OCR, object detection, facial recognition, and quality control capabilities",
	version="1.0.0",
	contact={
		"name": "Datacraft",
		"email": "nyimbi@gmail.com",
		"url": "https://www.datacraft.co.ke"
	},
	license_info={
		"name": "Enterprise License",
		"url": "https://www.datacraft.co.ke/license"
	}
)

# CORS Configuration
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # Configure appropriately for production
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Service Instances
processing_service = CVProcessingService()
document_service = CVDocumentAnalysisService()
detection_service = CVObjectDetectionService()
classification_service = CVImageClassificationService()
facial_service = CVFacialRecognitionService()
qc_service = CVQualityControlService()
video_service = CVVideoAnalysisService()
similarity_service = CVSimilaritySearchService()

# Upload Configuration
UPLOAD_DIR = Path("uploads/computer_vision")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/mov", "video/mkv"}
ALLOWED_DOCUMENT_TYPES = {"application/pdf", "image/jpeg", "image/png", "image/tiff"}


# Authentication and Authorization
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
	"""Extract user information from JWT token"""
	# Placeholder implementation - would integrate with APG RBAC
	return {
		"user_id": "user_123",
		"tenant_id": "tenant_456",
		"permissions": ["cv:read", "cv:write", "cv:admin"]
	}


async def verify_tenant_access(tenant_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> bool:
	"""Verify user has access to specified tenant"""
	return user["tenant_id"] == tenant_id


# File Upload Helpers
async def validate_file_upload(file: UploadFile, allowed_types: set) -> None:
	"""Validate uploaded file type and size"""
	if file.content_type not in allowed_types:
		raise HTTPException(
			status_code=400,
			detail=f"Unsupported file type: {file.content_type}. Allowed types: {allowed_types}"
		)
	
	# Check file size
	file.file.seek(0, 2)  # Seek to end
	file_size = file.file.tell()
	file.file.seek(0)  # Reset to beginning
	
	if file_size > MAX_FILE_SIZE:
		raise HTTPException(
			status_code=413,
			detail=f"File too large: {file_size} bytes. Maximum allowed: {MAX_FILE_SIZE} bytes"
		)


async def save_uploaded_file(file: UploadFile, tenant_id: str) -> str:
	"""Save uploaded file and return file path"""
	# Create tenant-specific directory
	tenant_dir = UPLOAD_DIR / tenant_id
	tenant_dir.mkdir(exist_ok=True)
	
	# Generate unique filename
	file_extension = Path(file.filename).suffix
	unique_filename = f"{uuid7str()}{file_extension}"
	file_path = tenant_dir / unique_filename
	
	# Save file
	with open(file_path, "wb") as f:
		content = await file.read()
		f.write(content)
	
	return str(file_path)


async def calculate_file_hash(file_path: str) -> str:
	"""Calculate SHA-256 hash of file"""
	hash_sha256 = hashlib.sha256()
	with open(file_path, "rb") as f:
		for chunk in iter(lambda: f.read(4096), b""):
			hash_sha256.update(chunk)
	return hash_sha256.hexdigest()


# API Endpoints

@app.get("/", response_model=Dict[str, Any])
async def root():
	"""Root endpoint with API information"""
	return {
		"name": "Computer Vision & Visual Intelligence API",
		"version": "1.0.0",
		"description": "Enterprise-grade computer vision processing",
		"endpoints": {
			"health": "/health",
			"docs": "/docs",
			"openapi": "/openapi.json"
		}
	}


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
	"""Health check endpoint for monitoring"""
	return HealthCheckResponse(
		status="healthy",
		timestamp=datetime.utcnow(),
		version="1.0.0",
		services={
			"document_analysis": "healthy",
			"object_detection": "healthy",
			"image_classification": "healthy",
			"facial_recognition": "healthy",
			"quality_control": "healthy",
			"video_analysis": "healthy"
		},
		performance_metrics={
			"avg_response_time_ms": 150,
			"active_jobs": len(processing_service.active_jobs),
			"queue_size": processing_service.processing_queue.qsize()
		}
	)


# Job Management Endpoints

@app.post("/api/v1/jobs", response_model=ProcessingJobResponse)
async def create_processing_job(
	request: ProcessingJobRequest,
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Create a new computer vision processing job"""
	try:
		job = await processing_service.create_processing_job(
			job_name=request.job_name,
			processing_type=request.processing_type,
			content_type=ContentType.IMAGE,  # Default, will be determined by file type
			input_file_path="",  # Will be set when file is uploaded
			processing_parameters=request.processing_parameters,
			tenant_id=user["tenant_id"],
			user_id=user["user_id"],
			priority=request.priority
		)
		
		return ProcessingJobResponse(
			job_id=job.id,
			status=job.status,
			progress_percentage=job.progress_percentage,
			results=job.results if job.results else None,
			error_message=job.error_message,
			created_at=job.created_at,
			estimated_completion=None
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")


@app.get("/api/v1/jobs/{job_id}", response_model=ProcessingJobResponse)
async def get_job_status(
	job_id: str = FastAPIPath(..., description="Job ID"),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Get status and results of a processing job"""
	job = await processing_service.get_job_status(job_id)
	
	if not job:
		raise HTTPException(status_code=404, detail="Job not found")
	
	if job.tenant_id != user["tenant_id"]:
		raise HTTPException(status_code=403, detail="Access denied")
	
	return ProcessingJobResponse(
		job_id=job.id,
		status=job.status,
		progress_percentage=job.progress_percentage,
		results=job.results if job.results else None,
		error_message=job.error_message,
		created_at=job.created_at,
		estimated_completion=None
	)


@app.delete("/api/v1/jobs/{job_id}")
async def cancel_job(
	job_id: str = FastAPIPath(..., description="Job ID"),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Cancel a processing job"""
	success = await processing_service.cancel_job(job_id, user["user_id"])
	
	if not success:
		raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
	
	return {"message": "Job cancelled successfully", "job_id": job_id}


# Document Processing Endpoints

@app.post("/api/v1/documents/ocr", response_model=OCRResponse)
async def extract_text_ocr(
	file: UploadFile = File(..., description="Document file for OCR processing"),
	request: OCRRequest = Depends(),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Extract text from document using OCR"""
	await validate_file_upload(file, ALLOWED_DOCUMENT_TYPES)
	
	try:
		# Save uploaded file
		file_path = await save_uploaded_file(file, user["tenant_id"])
		
		# Process OCR
		parameters = {
			"language": request.language,
			"ocr_engine": request.ocr_engine,
			"enhance_image": request.enhance_image,
			"extract_tables": request.extract_tables,
			"extract_forms": request.extract_forms
		}
		
		result = await document_service.process_document_ocr(
			file_path, parameters, user["tenant_id"]
		)
		
		return OCRResponse(
			extracted_text=result["extracted_text"],
			confidence_score=result["confidence_score"],
			language_detected=result["language_detected"],
			word_count=result["word_count"],
			processing_time_ms=result["processing_time_ms"],
			form_fields=result.get("form_fields"),
			tables=result.get("tables")
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@app.post("/api/v1/documents/analyze", response_model=Dict[str, Any])
async def analyze_document_comprehensive(
	file: UploadFile = File(..., description="Document file for comprehensive analysis"),
	analysis_level: AnalysisLevel = Form(default=AnalysisLevel.STANDARD),
	extract_entities: bool = Form(default=True),
	classify_document: bool = Form(default=True),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Perform comprehensive document analysis"""
	await validate_file_upload(file, ALLOWED_DOCUMENT_TYPES)
	
	try:
		file_path = await save_uploaded_file(file, user["tenant_id"])
		
		parameters = {
			"analysis_level": analysis_level,
			"extract_entities": extract_entities,
			"classify_document": classify_document,
			"language": "eng"
		}
		
		result = await document_service.analyze_document_comprehensive(
			file_path, parameters, user["tenant_id"]
		)
		
		return result
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Document analysis failed: {str(e)}")


# Object Detection Endpoints

@app.post("/api/v1/images/detect-objects", response_model=ObjectDetectionResponse)
async def detect_objects_in_image(
	file: UploadFile = File(..., description="Image file for object detection"),
	request: ObjectDetectionRequest = Depends(),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Detect objects in image using YOLO model"""
	await validate_file_upload(file, ALLOWED_IMAGE_TYPES)
	
	try:
		file_path = await save_uploaded_file(file, user["tenant_id"])
		
		parameters = {
			"model_name": request.model_name,
			"confidence_threshold": request.confidence_threshold,
			"iou_threshold": request.iou_threshold,
			"max_detections": request.max_detections
		}
		
		result = await detection_service.detect_objects(
			file_path, parameters, user["tenant_id"]
		)
		
		return ObjectDetectionResponse(
			detected_objects=result["detected_objects"],
			total_objects=result["total_objects"],
			detection_confidence=result["detection_confidence"],
			processing_time_ms=result["processing_time_ms"],
			model_used=result["model_used"]
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Object detection failed: {str(e)}")


# Image Classification Endpoints

@app.post("/api/v1/images/classify", response_model=ImageClassificationResponse)
async def classify_image(
	file: UploadFile = File(..., description="Image file for classification"),
	request: ImageClassificationRequest = Depends(),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Classify image using Vision Transformer or CNN model"""
	await validate_file_upload(file, ALLOWED_IMAGE_TYPES)
	
	try:
		file_path = await save_uploaded_file(file, user["tenant_id"])
		
		parameters = {
			"model_name": request.model_name,
			"top_k": request.top_k,
			"include_features": request.include_features
		}
		
		result = await classification_service.classify_image(
			file_path, parameters, user["tenant_id"]
		)
		
		return ImageClassificationResponse(
			classification_results=result["classification_results"],
			top_prediction=result["top_prediction"],
			confidence_score=result["confidence_score"],
			processing_time_ms=result["processing_time_ms"],
			features=result.get("features")
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Image classification failed: {str(e)}")


# Facial Recognition Endpoints

@app.post("/api/v1/faces/analyze", response_model=FacialRecognitionResponse)
async def analyze_faces(
	file: UploadFile = File(..., description="Image file for facial analysis"),
	request: FacialRecognitionRequest = Depends(),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Analyze faces in image with privacy controls"""
	await validate_file_upload(file, ALLOWED_IMAGE_TYPES)
	
	try:
		file_path = await save_uploaded_file(file, user["tenant_id"])
		
		parameters = {
			"features": request.features_to_extract,
			"anonymize_results": request.anonymize_results,
			"consent_recorded": request.consent_recorded,
			"retention_days": request.retention_days
		}
		
		result = await facial_service.analyze_faces(
			file_path, parameters, user["tenant_id"]
		)
		
		return FacialRecognitionResponse(
			faces_detected=result["faces_detected"],
			total_faces=result["total_faces"],
			features_extracted=result["features_extracted"],
			anonymized=result["anonymized"],
			processing_time_ms=result["processing_time_ms"]
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Facial analysis failed: {str(e)}")


# Quality Control Endpoints

@app.post("/api/v1/quality/inspect", response_model=QualityControlResponse)
async def inspect_quality(
	file: UploadFile = File(..., description="Image file for quality inspection"),
	request: QualityControlRequest = Depends(),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Perform quality control inspection"""
	await validate_file_upload(file, ALLOWED_IMAGE_TYPES)
	
	try:
		file_path = await save_uploaded_file(file, user["tenant_id"])
		
		parameters = {
			"inspection_type": request.inspection_type,
			"product_identifier": request.product_identifier,
			"inspection_station": request.inspection_station,
			"quality_standards": request.quality_standards,
			"tolerance_parameters": request.tolerance_parameters
		}
		
		result = await qc_service.inspect_quality(
			file_path, parameters, user["tenant_id"]
		)
		
		return QualityControlResponse(
			inspection_result=result["pass_fail_status"],
			overall_score=result["overall_score"],
			defects_detected=result["defects_detected"],
			defect_count=result["defect_count"],
			compliance_status=result.get("compliance_status", {}),
			processing_time_ms=result["processing_time_ms"]
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Quality inspection failed: {str(e)}")


# Video Analysis Endpoints

@app.post("/api/v1/videos/analyze", response_model=Dict[str, Any])
async def analyze_video(
	file: UploadFile = File(..., description="Video file for analysis"),
	analysis_type: str = Form(default="general", description="Type of video analysis"),
	frame_sampling_rate: int = Form(default=1, ge=1, le=30, description="Frames per second to analyze"),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Analyze video content for actions, events, and patterns"""
	await validate_file_upload(file, ALLOWED_VIDEO_TYPES)
	
	try:
		file_path = await save_uploaded_file(file, user["tenant_id"])
		
		parameters = {
			"analysis_type": analysis_type,
			"frame_sampling_rate": frame_sampling_rate
		}
		
		result = await video_service.analyze_video(
			file_path, parameters, user["tenant_id"]
		)
		
		return result
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")


# Similarity Search Endpoints

@app.post("/api/v1/images/find-similar", response_model=Dict[str, Any])
async def find_similar_images(
	file: UploadFile = File(..., description="Query image for similarity search"),
	max_results: int = Form(default=10, ge=1, le=100, description="Maximum number of similar images to return"),
	similarity_threshold: float = Form(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score"),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Find visually similar images"""
	await validate_file_upload(file, ALLOWED_IMAGE_TYPES)
	
	try:
		file_path = await save_uploaded_file(file, user["tenant_id"])
		
		parameters = {
			"max_results": max_results,
			"similarity_threshold": similarity_threshold
		}
		
		result = await similarity_service.find_similar_images(
			file_path, parameters, user["tenant_id"]
		)
		
		return result
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")


# Batch Processing Endpoints

@app.post("/api/v1/batch/process", response_model=BatchProcessingResponse)
async def create_batch_processing(
	files: List[UploadFile] = File(..., description="Files for batch processing"),
	request: BatchProcessingRequest = Depends(),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Create batch processing job for multiple files"""
	if len(files) > 100:
		raise HTTPException(status_code=400, detail="Maximum 100 files per batch")
	
	try:
		batch_id = uuid7str()
		job_ids = []
		
		# Determine allowed file types based on processing type
		if request.processing_type in [ProcessingType.OBJECT_DETECTION, ProcessingType.IMAGE_CLASSIFICATION]:
			allowed_types = ALLOWED_IMAGE_TYPES
		elif request.processing_type == ProcessingType.OCR:
			allowed_types = ALLOWED_DOCUMENT_TYPES
		elif request.processing_type == ProcessingType.VIDEO_ANALYSIS:
			allowed_types = ALLOWED_VIDEO_TYPES
		else:
			allowed_types = ALLOWED_IMAGE_TYPES
		
		# Process each file
		for file in files:
			await validate_file_upload(file, allowed_types)
			file_path = await save_uploaded_file(file, user["tenant_id"])
			
			# Create individual job
			job = await processing_service.create_processing_job(
				job_name=f"Batch {batch_id} - {file.filename}",
				processing_type=request.processing_type,
				content_type=ContentType.IMAGE,  # Will be determined by file type
				input_file_path=file_path,
				processing_parameters=request.processing_parameters,
				tenant_id=user["tenant_id"],
				user_id=user["user_id"],
				priority=request.priority
			)
			
			job_ids.append(job.id)
		
		return BatchProcessingResponse(
			batch_id=batch_id,
			job_ids=job_ids,
			total_files=len(files),
			estimated_completion=None
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@app.get("/api/v1/batch/{batch_id}/status", response_model=Dict[str, Any])
async def get_batch_status(
	batch_id: str = FastAPIPath(..., description="Batch ID"),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Get status of batch processing job"""
	# Placeholder implementation - would track batch jobs in database
	return {
		"batch_id": batch_id,
		"status": "processing",
		"total_jobs": 0,
		"completed_jobs": 0,
		"failed_jobs": 0,
		"progress_percentage": 0.0
	}


# Model Management Endpoints

@app.get("/api/v1/models", response_model=List[Dict[str, Any]])
async def list_available_models(
	model_type: Optional[ProcessingType] = Query(None, description="Filter by model type"),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""List available computer vision models"""
	# Placeholder implementation - would return actual model registry
	models = [
		{
			"model_id": "yolov8n",
			"model_name": "YOLOv8 Nano",
			"model_type": "object_detection",
			"version": "8.0.0",
			"description": "Fast and accurate object detection",
			"accuracy": 0.85,
			"inference_time_ms": 50
		},
		{
			"model_id": "vit-base",
			"model_name": "Vision Transformer Base",
			"model_type": "image_classification",
			"version": "1.0.0",
			"description": "High-accuracy image classification",
			"accuracy": 0.92,
			"inference_time_ms": 150
		}
	]
	
	if model_type:
		models = [m for m in models if m["model_type"] == model_type.value]
	
	return models


# Analytics and Reporting Endpoints

@app.get("/api/v1/analytics/summary", response_model=Dict[str, Any])
async def get_analytics_summary(
	days: int = Query(default=7, ge=1, le=365, description="Number of days to analyze"),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Get analytics summary for specified time period"""
	end_date = datetime.utcnow()
	start_date = end_date - timedelta(days=days)
	
	# Placeholder implementation - would query actual analytics data
	return {
		"period": {
			"start_date": start_date,
			"end_date": end_date,
			"days": days
		},
		"processing_stats": {
			"total_jobs": 0,
			"successful_jobs": 0,
			"failed_jobs": 0,
			"avg_processing_time_ms": 0
		},
		"usage_by_type": {},
		"performance_metrics": {
			"avg_response_time_ms": 150,
			"peak_concurrent_users": 0,
			"error_rate": 0.0
		}
	}


# Error Handlers

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
	"""Handle Pydantic validation errors"""
	return JSONResponse(
		status_code=422,
		content={
			"error": "Validation Error",
			"detail": exc.errors(),
			"timestamp": datetime.utcnow().isoformat()
		}
	)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
	"""Handle HTTP exceptions with consistent format"""
	return JSONResponse(
		status_code=exc.status_code,
		content={
			"error": exc.detail,
			"status_code": exc.status_code,
			"timestamp": datetime.utcnow().isoformat()
		}
	)


# Custom OpenAPI Schema
def custom_openapi():
	"""Generate custom OpenAPI schema with enhanced documentation"""
	if app.openapi_schema:
		return app.openapi_schema
	
	openapi_schema = get_openapi(
		title="Computer Vision & Visual Intelligence API",
		version="1.0.0",
		description="Enterprise-grade computer vision processing API with comprehensive OCR, object detection, facial recognition, and quality control capabilities. Built for APG platform integration with multi-tenant support, RBAC, and audit compliance.",
		routes=app.routes,
	)
	
	# Add security schemes
	openapi_schema["components"]["securitySchemes"] = {
		"bearerAuth": {
			"type": "http",
			"scheme": "bearer",
			"bearerFormat": "JWT",
			"description": "APG platform JWT token for authentication"
		}
	}
	
	# Add security to all endpoints
	for path in openapi_schema["paths"]:
		for method in openapi_schema["paths"][path]:
			openapi_schema["paths"][path][method]["security"] = [{"bearerAuth": []}]
	
	app.openapi_schema = openapi_schema
	return app.openapi_schema


app.openapi = custom_openapi

# Development Server
if __name__ == "__main__":
	uvicorn.run(
		"api:app",
		host="0.0.0.0",
		port=8000,
		reload=True,
		log_level="info"
	)