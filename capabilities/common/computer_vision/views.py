"""
Computer Vision & Visual Intelligence - View Models & UI

Combined Pydantic v2 view models for data serialization and Flask-AppBuilder
dashboard views for comprehensive computer vision management interface with
specialized workspaces for different processing types.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid_extensions import uuid7str

from flask import flash, redirect, request, url_for, jsonify, render_template_string
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.mixins import AuditMixin
from flask_appbuilder.security.decorators import protect
from pydantic import BaseModel, Field, ConfigDict
from wtforms import Form, StringField, SelectField, TextAreaField, FileField, IntegerField, FloatField
from wtforms.validators import DataRequired, Length, NumberRange

from .models import (
	CVProcessingJob, CVImageProcessing, CVDocumentAnalysis,
	CVObjectDetection, CVFacialRecognition, CVQualityControl,
	CVModel, CVAnalyticsReport, ProcessingStatus, ProcessingType,
	ContentType, QualityControlType, FacialFeature, AnalysisLevel
)
from .service import (
	CVProcessingService, CVDocumentAnalysisService, CVObjectDetectionService,
	CVImageClassificationService, CVFacialRecognitionService,
	CVQualityControlService, CVVideoAnalysisService, CVSimilaritySearchService
)


# ===== PYDANTIC V2 VIEW MODELS =====

class CVBaseViewModel(BaseModel):
	"""Base view model for all computer vision UI data"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		validate_default=True
	)


class ProcessingJobViewModel(CVBaseViewModel):
	"""View model for processing job display"""
	job_id: str = Field(..., description="Unique job identifier")
	job_name: str = Field(..., description="Human-readable job name")
	processing_type: ProcessingType = Field(..., description="Type of processing")
	status: ProcessingStatus = Field(..., description="Current job status")
	progress_percentage: float = Field(..., ge=0.0, le=1.0, description="Processing progress")
	created_at: datetime = Field(..., description="Creation timestamp")
	started_at: Optional[datetime] = Field(None, description="Processing start time")
	completed_at: Optional[datetime] = Field(None, description="Completion time")
	duration_seconds: Optional[int] = Field(None, description="Processing duration")
	error_message: Optional[str] = Field(None, description="Error message if failed")
	results_summary: Optional[str] = Field(None, description="Brief results summary")
	
	@classmethod
	def from_job(cls, job: CVProcessingJob) -> 'ProcessingJobViewModel':
		"""Create view model from processing job"""
		results_summary = None
		if job.results:
			if job.processing_type == ProcessingType.OCR:
				word_count = job.results.get('word_count', 0)
				results_summary = f"Extracted {word_count} words"
			elif job.processing_type == ProcessingType.OBJECT_DETECTION:
				object_count = job.results.get('total_objects', 0)
				results_summary = f"Detected {object_count} objects"
			elif job.processing_type == ProcessingType.FACIAL_RECOGNITION:
				face_count = job.results.get('total_faces', 0)
				results_summary = f"Analyzed {face_count} faces"
			elif job.processing_type == ProcessingType.QUALITY_CONTROL:
				status = job.results.get('pass_fail_status', 'Unknown')
				results_summary = f"Quality check: {status}"
		
		return cls(
			job_id=job.id,
			job_name=job.job_name,
			processing_type=job.processing_type,
			status=job.status,
			progress_percentage=job.progress_percentage,
			created_at=job.created_at,
			started_at=job.started_at,
			completed_at=job.completed_at,
			duration_seconds=job.duration_seconds,
			error_message=job.error_message,
			results_summary=results_summary
		)


class DashboardStatsViewModel(CVBaseViewModel):
	"""View model for dashboard statistics"""
	total_jobs_today: int = Field(..., ge=0, description="Total jobs processed today")
	successful_jobs_today: int = Field(..., ge=0, description="Successful jobs today")
	failed_jobs_today: int = Field(..., ge=0, description="Failed jobs today")
	active_jobs: int = Field(..., ge=0, description="Currently active jobs")
	avg_processing_time_ms: float = Field(..., ge=0, description="Average processing time")
	
	processing_by_type: Dict[str, int] = Field(default_factory=dict, description="Jobs by processing type")
	recent_jobs: List[ProcessingJobViewModel] = Field(default_factory=list, description="Recent jobs")
	performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
	
	success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate percentage")
	queue_length: int = Field(..., ge=0, description="Current queue length")


class OCRResultViewModel(CVBaseViewModel):
	"""View model for OCR results display"""
	extracted_text: str = Field(..., description="Extracted text content")
	confidence_score: float = Field(..., ge=0.0, le=1.0, description="OCR confidence")
	language_detected: str = Field(..., description="Detected language")
	word_count: int = Field(..., ge=0, description="Number of words")
	character_count: int = Field(..., ge=0, description="Number of characters")
	processing_time_ms: int = Field(..., ge=0, description="Processing time")
	
	form_fields: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted form fields")
	tables: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted tables")
	key_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Named entities")
	
	has_forms: bool = Field(default=False, description="Document contains forms")
	has_tables: bool = Field(default=False, description="Document contains tables")
	
	@classmethod
	def from_document_analysis(cls, analysis: Dict[str, Any]) -> 'OCRResultViewModel':
		"""Create view model from document analysis results"""
		return cls(
			extracted_text=analysis.get('extracted_text', ''),
			confidence_score=analysis.get('confidence_score', 0.0),
			language_detected=analysis.get('language_detected', 'unknown'),
			word_count=analysis.get('word_count', 0),
			character_count=len(analysis.get('extracted_text', '')),
			processing_time_ms=analysis.get('processing_time_ms', 0),
			form_fields=analysis.get('form_fields', []),
			tables=analysis.get('tables', []),
			key_entities=analysis.get('key_entities', []),
			has_forms=len(analysis.get('form_fields', [])) > 0,
			has_tables=len(analysis.get('tables', [])) > 0
		)


class ObjectDetectionResultViewModel(CVBaseViewModel):
	"""View model for object detection results"""
	detected_objects: List[Dict[str, Any]] = Field(..., description="Detected objects")
	total_objects: int = Field(..., ge=0, description="Total objects detected")
	detection_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
	processing_time_ms: int = Field(..., ge=0, description="Processing time")
	model_used: str = Field(..., description="Detection model used")
	
	objects_by_class: Dict[str, int] = Field(default_factory=dict, description="Object counts by class")
	highest_confidence_object: Optional[Dict[str, Any]] = Field(None, description="Most confident detection")
	
	@classmethod
	def from_detection_results(cls, results: Dict[str, Any]) -> 'ObjectDetectionResultViewModel':
		"""Create view model from object detection results"""
		objects = results.get('detected_objects', [])
		
		# Count objects by class
		objects_by_class = {}
		highest_confidence = None
		max_confidence = 0.0
		
		for obj in objects:
			class_name = obj.get('class_name', 'unknown')
			objects_by_class[class_name] = objects_by_class.get(class_name, 0) + 1
			
			if obj.get('confidence', 0) > max_confidence:
				max_confidence = obj.get('confidence', 0)
				highest_confidence = obj
		
		return cls(
			detected_objects=objects,
			total_objects=results.get('total_objects', 0),
			detection_confidence=results.get('detection_confidence', 0.0),
			processing_time_ms=results.get('processing_time_ms', 0),
			model_used=results.get('model_used', 'unknown'),
			objects_by_class=objects_by_class,
			highest_confidence_object=highest_confidence
		)


class QualityControlResultViewModel(CVBaseViewModel):
	"""View model for quality control results"""
	inspection_result: str = Field(..., description="Overall inspection result")
	overall_score: float = Field(..., ge=0.0, le=1.0, description="Quality score")
	defects_detected: List[Dict[str, Any]] = Field(..., description="Detected defects")
	defect_count: int = Field(..., ge=0, description="Total defects")
	processing_time_ms: int = Field(..., ge=0, description="Processing time")
	
	critical_defects: int = Field(default=0, description="Number of critical defects")
	major_defects: int = Field(default=0, description="Number of major defects")
	minor_defects: int = Field(default=0, description="Number of minor defects")
	
	passed_inspection: bool = Field(default=False, description="Whether inspection passed")
	
	@classmethod
	def from_qc_results(cls, results: Dict[str, Any]) -> 'QualityControlResultViewModel':
		"""Create view model from quality control results"""
		defects = results.get('defects_detected', [])
		
		critical_count = len([d for d in defects if d.get('severity') == 'CRITICAL'])
		major_count = len([d for d in defects if d.get('severity') == 'MAJOR'])
		minor_count = len([d for d in defects if d.get('severity') == 'MINOR'])
		
		return cls(
			inspection_result=results.get('pass_fail_status', 'UNKNOWN'),
			overall_score=results.get('overall_score', 0.0),
			defects_detected=defects,
			defect_count=results.get('defect_count', 0),
			processing_time_ms=results.get('processing_time_ms', 0),
			critical_defects=critical_count,
			major_defects=major_count,
			minor_defects=minor_count,
			passed_inspection=results.get('pass_fail_status') == 'PASS'
		)


# ===== FLASK-APPBUILDER DASHBOARD VIEWS =====

class ComputerVisionBaseView(BaseView):
	"""Base view for all computer vision interfaces"""
	
	def __init__(self):
		super().__init__()
		self.processing_service = CVProcessingService()
		self.document_service = CVDocumentAnalysisService()
		self.detection_service = CVObjectDetectionService()
		self.classification_service = CVImageClassificationService()
		self.facial_service = CVFacialRecognitionService()
		self.qc_service = CVQualityControlService()
		self.video_service = CVVideoAnalysisService()
		self.similarity_service = CVSimilaritySearchService()
	
	def get_current_user_info(self) -> Dict[str, Any]:
		"""Get current user information for multi-tenant access"""
		# Placeholder - would integrate with APG RBAC
		return {
			"user_id": "user_123",
			"tenant_id": "tenant_456",
			"permissions": ["cv:read", "cv:write", "cv:admin"]
		}


class ComputerVisionDashboardView(ComputerVisionBaseView):
	"""Main computer vision dashboard with overview and statistics"""
	
	route_base = "/computer_vision"
	default_view = "dashboard"
	
	@expose("/")
	@expose("/dashboard")
	@has_access
	def dashboard(self):
		"""Main dashboard with overview statistics and recent activity"""
		user_info = self.get_current_user_info()
		
		# Get dashboard statistics
		stats = self._get_dashboard_stats(user_info["tenant_id"])
		
		dashboard_template = """
		<div class="row">
			<div class="col-md-12">
				<h2>Computer Vision & Visual Intelligence</h2>
				<p class="lead">Enterprise-grade visual content processing and analysis</p>
			</div>
		</div>
		
		<div class="row">
			<div class="col-md-3">
				<div class="panel panel-primary">
					<div class="panel-heading">
						<h3 class="panel-title">Jobs Today</h3>
					</div>
					<div class="panel-body">
						<h2>{{ stats.total_jobs_today }}</h2>
						<p>{{ stats.successful_jobs_today }} successful, {{ stats.failed_jobs_today }} failed</p>
					</div>
				</div>
			</div>
			<div class="col-md-3">
				<div class="panel panel-info">
					<div class="panel-heading">
						<h3 class="panel-title">Active Jobs</h3>
					</div>
					<div class="panel-body">
						<h2>{{ stats.active_jobs }}</h2>
						<p>Queue length: {{ stats.queue_length }}</p>
					</div>
				</div>
			</div>
			<div class="col-md-3">
				<div class="panel panel-success">
					<div class="panel-heading">
						<h3 class="panel-title">Success Rate</h3>
					</div>
					<div class="panel-body">
						<h2>{{ (stats.success_rate * 100)|round(1) }}%</h2>
						<p>Last 24 hours</p>
					</div>
				</div>
			</div>
			<div class="col-md-3">
				<div class="panel panel-warning">
					<div class="panel-heading">
						<h3 class="panel-title">Avg Processing</h3>
					</div>
					<div class="panel-body">
						<h2>{{ (stats.avg_processing_time_ms)|round(0)|int }}ms</h2>
						<p>Average response time</p>
					</div>
				</div>
			</div>
		</div>
		
		<div class="row">
			<div class="col-md-6">
				<div class="panel panel-default">
					<div class="panel-heading">
						<h3 class="panel-title">Processing by Type</h3>
					</div>
					<div class="panel-body">
						<canvas id="processingChart" width="400" height="200"></canvas>
					</div>
				</div>
			</div>
			<div class="col-md-6">
				<div class="panel panel-default">
					<div class="panel-heading">
						<h3 class="panel-title">Recent Jobs</h3>
					</div>
					<div class="panel-body">
						<div class="table-responsive">
							<table class="table table-striped">
								<thead>
									<tr>
										<th>Job Name</th>
										<th>Type</th>
										<th>Status</th>
										<th>Time</th>
									</tr>
								</thead>
								<tbody>
									{% for job in stats.recent_jobs %}
									<tr>
										<td>{{ job.job_name }}</td>
										<td>{{ job.processing_type.value }}</td>
										<td>
											<span class="label label-{% if job.status.value == 'completed' %}success{% elif job.status.value == 'failed' %}danger{% else %}warning{% endif %}">
												{{ job.status.value }}
											</span>
										</td>
										<td>{{ job.created_at.strftime('%H:%M') }}</td>
									</tr>
									{% endfor %}
								</tbody>
							</table>
						</div>
					</div>
				</div>
			</div>
		</div>
		
		<div class="row">
			<div class="col-md-12">
				<div class="panel panel-default">
					<div class="panel-heading">
						<h3 class="panel-title">Quick Actions</h3>
					</div>
					<div class="panel-body">
						<div class="btn-group" role="group">
							<a href="{{ url_for('ComputerVisionDocumentView.workspace') }}" class="btn btn-primary">
								<i class="fa fa-file-text"></i> Document Processing
							</a>
							<a href="{{ url_for('ComputerVisionImageView.workspace') }}" class="btn btn-info">
								<i class="fa fa-image"></i> Image Analysis
							</a>
							<a href="{{ url_for('ComputerVisionQualityView.workspace') }}" class="btn btn-success">
								<i class="fa fa-check-circle"></i> Quality Control
							</a>
							<a href="{{ url_for('ComputerVisionVideoView.workspace') }}" class="btn btn-warning">
								<i class="fa fa-video-camera"></i> Video Analysis
							</a>
							<a href="{{ url_for('ComputerVisionModelView.workspace') }}" class="btn btn-default">
								<i class="fa fa-cogs"></i> Model Management
							</a>
						</div>
					</div>
				</div>
			</div>
		</div>
		
		<script>
		// Chart.js for processing type visualization
		var ctx = document.getElementById('processingChart').getContext('2d');
		var processingChart = new Chart(ctx, {
			type: 'doughnut',
			data: {
				labels: {{ stats.processing_by_type.keys()|list|tojson }},
				datasets: [{
					data: {{ stats.processing_by_type.values()|list|tojson }},
					backgroundColor: [
						'#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'
					]
				}]
			},
			options: {
				responsive: true,
				legend: {
					position: 'bottom'
				}
			}
		});
		</script>
		"""
		
		return render_template_string(dashboard_template, stats=stats)
	
	def _get_dashboard_stats(self, tenant_id: str) -> DashboardStatsViewModel:
		"""Get dashboard statistics for display"""
		# Placeholder implementation - would query actual data
		today = datetime.utcnow().date()
		
		return DashboardStatsViewModel(
			total_jobs_today=25,
			successful_jobs_today=22,
			failed_jobs_today=3,
			active_jobs=len(self.processing_service.active_jobs),
			avg_processing_time_ms=1250.0,
			processing_by_type={
				"OCR": 10,
				"Object Detection": 8,
				"Image Classification": 5,
				"Quality Control": 2
			},
			recent_jobs=[],
			performance_metrics={
				"throughput": 100.0,
				"accuracy": 0.95
			},
			success_rate=0.88,
			queue_length=self.processing_service.processing_queue.qsize()
		)


class ComputerVisionDocumentView(ComputerVisionBaseView):
	"""Document processing workspace with OCR and analysis tools"""
	
	route_base = "/computer_vision/documents"
	default_view = "workspace"
	
	@expose("/")
	@expose("/workspace")
	@has_access
	def workspace(self):
		"""Document processing workspace"""
		document_workspace_template = """
		<div class="row">
			<div class="col-md-12">
				<h2>Document Processing & OCR</h2>
				<p>Extract text, analyze forms, and process documents with enterprise-grade accuracy</p>
			</div>
		</div>
		
		<div class="row">
			<div class="col-md-8">
				<div class="panel panel-primary">
					<div class="panel-heading">
						<h3 class="panel-title">Upload Document</h3>
					</div>
					<div class="panel-body">
						<form id="documentForm" enctype="multipart/form-data">
							<div class="form-group">
								<label for="documentFile">Select Document</label>
								<input type="file" class="form-control" id="documentFile" name="file" 
								       accept=".pdf,.jpg,.jpeg,.png,.tiff" required>
								<small class="help-block">Supported formats: PDF, JPEG, PNG, TIFF</small>
							</div>
							
							<div class="form-group">
								<label for="ocrLanguage">Language</label>
								<select class="form-control" id="ocrLanguage" name="language">
									<option value="eng">English</option>
									<option value="spa">Spanish</option>
									<option value="fra">French</option>
									<option value="deu">German</option>
									<option value="chi_sim">Chinese (Simplified)</option>
								</select>
							</div>
							
							<div class="form-group">
								<label for="ocrEngine">OCR Engine</label>
								<select class="form-control" id="ocrEngine" name="ocr_engine">
									<option value="tesseract">Tesseract (Default)</option>
									<option value="easyocr">EasyOCR</option>
									<option value="paddleocr">PaddleOCR</option>
								</select>
							</div>
							
							<div class="checkbox">
								<label>
									<input type="checkbox" id="enhanceImage" name="enhance_image" checked>
									Apply image enhancement
								</label>
							</div>
							
							<div class="checkbox">
								<label>
									<input type="checkbox" id="extractTables" name="extract_tables">
									Extract table data
								</label>
							</div>
							
							<div class="checkbox">
								<label>
									<input type="checkbox" id="extractForms" name="extract_forms">
									Extract form fields
								</label>
							</div>
							
							<button type="submit" class="btn btn-primary">
								<i class="fa fa-upload"></i> Process Document
							</button>
						</form>
					</div>
				</div>
			</div>
			
			<div class="col-md-4">
				<div class="panel panel-info">
					<div class="panel-heading">
						<h3 class="panel-title">Processing Tips</h3>
					</div>
					<div class="panel-body">
						<ul>
							<li>Use high-resolution images for better OCR accuracy</li>
							<li>Ensure good contrast between text and background</li>
							<li>Enable image enhancement for scanned documents</li>
							<li>Select the correct language for optimal results</li>
							<li>Table extraction works best with clear table borders</li>
						</ul>
					</div>
				</div>
			</div>
		</div>
		
		<div class="row" id="resultsSection" style="display: none;">
			<div class="col-md-12">
				<div class="panel panel-success">
					<div class="panel-heading">
						<h3 class="panel-title">Processing Results</h3>
					</div>
					<div class="panel-body">
						<div id="processingResults"></div>
					</div>
				</div>
			</div>
		</div>
		
		<script>
		document.getElementById('documentForm').addEventListener('submit', function(e) {
			e.preventDefault();
			
			var formData = new FormData(this);
			var submitBtn = this.querySelector('button[type="submit"]');
			
			submitBtn.disabled = true;
			submitBtn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Processing...';
			
			fetch('/api/v1/documents/ocr', {
				method: 'POST',
				body: formData,
				headers: {
					'Authorization': 'Bearer ' + localStorage.getItem('token')
				}
			})
			.then(response => response.json())
			.then(data => {
				displayResults(data);
				document.getElementById('resultsSection').style.display = 'block';
			})
			.catch(error => {
				alert('Error processing document: ' + error.message);
			})
			.finally(() => {
				submitBtn.disabled = false;
				submitBtn.innerHTML = '<i class="fa fa-upload"></i> Process Document';
			});
		});
		
		function displayResults(data) {
			var resultsHtml = `
				<div class="row">
					<div class="col-md-6">
						<h4>Extracted Text</h4>
						<div class="well" style="max-height: 400px; overflow-y: auto;">
							<pre>${data.extracted_text}</pre>
						</div>
					</div>
					<div class="col-md-6">
						<h4>Analysis Summary</h4>
						<table class="table table-striped">
							<tr><th>Confidence Score</th><td>${(data.confidence_score * 100).toFixed(1)}%</td></tr>
							<tr><th>Language Detected</th><td>${data.language_detected}</td></tr>
							<tr><th>Word Count</th><td>${data.word_count}</td></tr>
							<tr><th>Processing Time</th><td>${data.processing_time_ms}ms</td></tr>
						</table>
					</div>
				</div>
			`;
			
			if (data.form_fields && data.form_fields.length > 0) {
				resultsHtml += `
					<div class="row">
						<div class="col-md-12">
							<h4>Form Fields</h4>
							<div class="table-responsive">
								<table class="table table-striped">
									<thead>
										<tr><th>Field Name</th><th>Value</th><th>Confidence</th></tr>
									</thead>
									<tbody>
										${data.form_fields.map(field => 
											`<tr><td>${field.field_name}</td><td>${field.field_value}</td><td>${(field.confidence * 100).toFixed(1)}%</td></tr>`
										).join('')}
									</tbody>
								</table>
							</div>
						</div>
					</div>
				`;
			}
			
			document.getElementById('processingResults').innerHTML = resultsHtml;
		}
		</script>
		"""
		
		return render_template_string(document_workspace_template)


class ComputerVisionImageView(ComputerVisionBaseView):
	"""Image analysis workspace with object detection and classification"""
	
	route_base = "/computer_vision/images"
	default_view = "workspace"
	
	@expose("/")
	@expose("/workspace")
	@has_access
	def workspace(self):
		"""Image analysis workspace"""
		image_workspace_template = """
		<div class="row">
			<div class="col-md-12">
				<h2>Image Analysis & Classification</h2>
				<p>Detect objects, classify images, and analyze visual content with AI-powered precision</p>
			</div>
		</div>
		
		<div class="row">
			<div class="col-md-6">
				<div class="panel panel-primary">
					<div class="panel-heading">
						<h3 class="panel-title">Object Detection</h3>
					</div>
					<div class="panel-body">
						<form id="detectionForm" enctype="multipart/form-data">
							<div class="form-group">
								<label for="detectionFile">Select Image</label>
								<input type="file" class="form-control" id="detectionFile" name="file" 
								       accept=".jpg,.jpeg,.png,.bmp,.webp" required>
							</div>
							
							<div class="form-group">
								<label for="detectionModel">Detection Model</label>
								<select class="form-control" id="detectionModel" name="model_name">
									<option value="yolov8n.pt">YOLOv8 Nano (Fast)</option>
									<option value="yolov8s.pt">YOLOv8 Small</option>
									<option value="yolov8m.pt">YOLOv8 Medium</option>
									<option value="yolov8l.pt">YOLOv8 Large (Accurate)</option>
								</select>
							</div>
							
							<div class="form-group">
								<label for="confidenceThreshold">Confidence Threshold</label>
								<input type="range" class="form-control" id="confidenceThreshold" 
								       name="confidence_threshold" min="0.1" max="1.0" step="0.1" value="0.5">
								<small class="help-block">Current: <span id="confidenceValue">0.5</span></small>
							</div>
							
							<button type="submit" class="btn btn-primary">
								<i class="fa fa-search"></i> Detect Objects
							</button>
						</form>
					</div>
				</div>
			</div>
			
			<div class="col-md-6">
				<div class="panel panel-info">
					<div class="panel-heading">
						<h3 class="panel-title">Image Classification</h3>
					</div>
					<div class="panel-body">
						<form id="classificationForm" enctype="multipart/form-data">
							<div class="form-group">
								<label for="classificationFile">Select Image</label>
								<input type="file" class="form-control" id="classificationFile" name="file" 
								       accept=".jpg,.jpeg,.png,.bmp,.webp" required>
							</div>
							
							<div class="form-group">
								<label for="classificationModel">Classification Model</label>
								<select class="form-control" id="classificationModel" name="model_name">
									<option value="google/vit-base-patch16-224">Vision Transformer (Recommended)</option>
									<option value="microsoft/resnet-50">ResNet-50</option>
									<option value="google/efficientnet-b0">EfficientNet-B0</option>
								</select>
							</div>
							
							<div class="form-group">
								<label for="topK">Top Predictions</label>
								<select class="form-control" id="topK" name="top_k">
									<option value="3">Top 3</option>
									<option value="5" selected>Top 5</option>
									<option value="10">Top 10</option>
								</select>
							</div>
							
							<button type="submit" class="btn btn-info">
								<i class="fa fa-tags"></i> Classify Image
							</button>
						</form>
					</div>
				</div>
			</div>
		</div>
		
		<div class="row" id="imageResultsSection" style="display: none;">
			<div class="col-md-12">
				<div class="panel panel-success">
					<div class="panel-heading">
						<h3 class="panel-title">Analysis Results</h3>
					</div>
					<div class="panel-body">
						<div id="imageResults"></div>
					</div>
				</div>
			</div>
		</div>
		
		<script>
		// Update confidence threshold display
		document.getElementById('confidenceThreshold').addEventListener('input', function() {
			document.getElementById('confidenceValue').textContent = this.value;
		});
		
		// Object detection form
		document.getElementById('detectionForm').addEventListener('submit', function(e) {
			e.preventDefault();
			processImageRequest(this, '/api/v1/images/detect-objects', 'detection');
		});
		
		// Classification form
		document.getElementById('classificationForm').addEventListener('submit', function(e) {
			e.preventDefault();
			processImageRequest(this, '/api/v1/images/classify', 'classification');
		});
		
		function processImageRequest(form, endpoint, type) {
			var formData = new FormData(form);
			var submitBtn = form.querySelector('button[type="submit"]');
			
			submitBtn.disabled = true;
			submitBtn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Processing...';
			
			fetch(endpoint, {
				method: 'POST',
				body: formData,
				headers: {
					'Authorization': 'Bearer ' + localStorage.getItem('token')
				}
			})
			.then(response => response.json())
			.then(data => {
				displayImageResults(data, type);
				document.getElementById('imageResultsSection').style.display = 'block';
			})
			.catch(error => {
				alert('Error processing image: ' + error.message);
			})
			.finally(() => {
				submitBtn.disabled = false;
				var icon = type === 'detection' ? 'search' : 'tags';
				var text = type === 'detection' ? 'Detect Objects' : 'Classify Image';
				submitBtn.innerHTML = `<i class="fa fa-${icon}"></i> ${text}`;
			});
		}
		
		function displayImageResults(data, type) {
			var resultsHtml = '';
			
			if (type === 'detection') {
				resultsHtml = `
					<h4>Object Detection Results</h4>
					<div class="row">
						<div class="col-md-6">
							<table class="table table-striped">
								<tr><th>Total Objects</th><td>${data.total_objects}</td></tr>
								<tr><th>Detection Confidence</th><td>${(data.detection_confidence * 100).toFixed(1)}%</td></tr>
								<tr><th>Processing Time</th><td>${data.processing_time_ms}ms</td></tr>
								<tr><th>Model Used</th><td>${data.model_used}</td></tr>
							</table>
						</div>
						<div class="col-md-6">
							<h5>Detected Objects</h5>
							<div class="table-responsive" style="max-height: 300px; overflow-y: auto;">
								<table class="table table-sm">
									<thead>
										<tr><th>Object</th><th>Confidence</th><th>Size</th></tr>
									</thead>
									<tbody>
										${data.detected_objects.map(obj => 
											`<tr><td>${obj.class_name}</td><td>${(obj.confidence * 100).toFixed(1)}%</td><td>${Math.round(obj.area_pixels)} px²</td></tr>`
										).join('')}
									</tbody>
								</table>
							</div>
						</div>
					</div>
				`;
			} else if (type === 'classification') {
				resultsHtml = `
					<h4>Image Classification Results</h4>
					<div class="row">
						<div class="col-md-6">
							<table class="table table-striped">
								<tr><th>Top Prediction</th><td>${data.top_prediction ? data.top_prediction.class_name : 'None'}</td></tr>
								<tr><th>Confidence</th><td>${(data.confidence_score * 100).toFixed(1)}%</td></tr>
								<tr><th>Processing Time</th><td>${data.processing_time_ms}ms</td></tr>
							</table>
						</div>
						<div class="col-md-6">
							<h5>All Predictions</h5>
							<div class="table-responsive">
								<table class="table table-sm">
									<thead>
										<tr><th>Rank</th><th>Class</th><th>Confidence</th></tr>
									</thead>
									<tbody>
										${data.classification_results.map((pred, index) => 
											`<tr><td>${index + 1}</td><td>${pred.class_name}</td><td>${(pred.confidence * 100).toFixed(1)}%</td></tr>`
										).join('')}
									</tbody>
								</table>
							</div>
						</div>
					</div>
				`;
			}
			
			document.getElementById('imageResults').innerHTML = resultsHtml;
		}
		</script>
		"""
		
		return render_template_string(image_workspace_template)


class ComputerVisionQualityView(ComputerVisionBaseView):
	"""Quality control workspace for manufacturing inspection"""
	
	route_base = "/computer_vision/quality"
	default_view = "workspace"
	
	@expose("/")
	@expose("/workspace")
	@has_access
	def workspace(self):
		"""Quality control inspection workspace"""
		quality_workspace_template = """
		<div class="row">
			<div class="col-md-12">
				<h2>Quality Control & Inspection</h2>
				<p>Automated quality inspection with defect detection and compliance verification</p>
			</div>
		</div>
		
		<div class="row">
			<div class="col-md-8">
				<div class="panel panel-primary">
					<div class="panel-heading">
						<h3 class="panel-title">Quality Inspection</h3>
					</div>
					<div class="panel-body">
						<form id="qualityForm" enctype="multipart/form-data">
							<div class="form-group">
								<label for="qualityFile">Upload Product Image</label>
								<input type="file" class="form-control" id="qualityFile" name="file" 
								       accept=".jpg,.jpeg,.png,.bmp,.tiff" required>
							</div>
							
							<div class="form-group">
								<label for="inspectionType">Inspection Type</label>
								<select class="form-control" id="inspectionType" name="inspection_type" required>
									<option value="defect_detection">Defect Detection</option>
									<option value="surface_inspection">Surface Inspection</option>
									<option value="dimensional_analysis">Dimensional Analysis</option>
									<option value="assembly_verification">Assembly Verification</option>
									<option value="packaging_inspection">Packaging Inspection</option>
								</select>
							</div>
							
							<div class="form-group">
								<label for="productId">Product Identifier</label>
								<input type="text" class="form-control" id="productId" name="product_identifier" 
								       placeholder="e.g., PROD-001, BATCH-123" required>
							</div>
							
							<div class="form-group">
								<label for="inspectionStation">Inspection Station</label>
								<input type="text" class="form-control" id="inspectionStation" name="inspection_station" 
								       placeholder="e.g., STATION-A1" required>
							</div>
							
							<div class="form-group">
								<label for="qualityStandards">Quality Standards</label>
								<select multiple class="form-control" id="qualityStandards" name="quality_standards">
									<option value="ISO-9001">ISO 9001</option>
									<option value="ISO-14001">ISO 14001</option>
									<option value="FDA-GMP">FDA GMP</option>
									<option value="CE-MARKING">CE Marking</option>
									<option value="CUSTOM">Custom Standards</option>
								</select>
								<small class="help-block">Hold Ctrl/Cmd to select multiple standards</small>
							</div>
							
							<button type="submit" class="btn btn-primary">
								<i class="fa fa-check-circle"></i> Start Inspection
							</button>
						</form>
					</div>
				</div>
			</div>
			
			<div class="col-md-4">
				<div class="panel panel-info">
					<div class="panel-heading">
						<h3 class="panel-title">Inspection Guidelines</h3>
					</div>
					<div class="panel-body">
						<h5>Defect Detection</h5>
						<ul>
							<li>Ensure good lighting conditions</li>
							<li>Use high-resolution images (>2MP)</li>
							<li>Center the product in the frame</li>
							<li>Avoid shadows and reflections</li>
						</ul>
						
						<h5>Surface Inspection</h5>
						<ul>
							<li>Clean the surface before imaging</li>
							<li>Use consistent background color</li>
							<li>Maintain consistent distance</li>
						</ul>
					</div>
				</div>
			</div>
		</div>
		
		<div class="row" id="qualityResultsSection" style="display: none;">
			<div class="col-md-12">
				<div class="panel panel-default">
					<div class="panel-heading">
						<h3 class="panel-title">Inspection Results</h3>
					</div>
					<div class="panel-body">
						<div id="qualityResults"></div>
					</div>
				</div>
			</div>
		</div>
		
		<script>
		document.getElementById('qualityForm').addEventListener('submit', function(e) {
			e.preventDefault();
			
			var formData = new FormData(this);
			var submitBtn = this.querySelector('button[type="submit"]');
			
			// Handle multiple select for quality standards
			var standards = Array.from(document.getElementById('qualityStandards').selectedOptions)
			                    .map(option => option.value);
			formData.delete('quality_standards');
			standards.forEach(standard => formData.append('quality_standards', standard));
			
			submitBtn.disabled = true;
			submitBtn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Inspecting...';
			
			fetch('/api/v1/quality/inspect', {
				method: 'POST',
				body: formData,
				headers: {
					'Authorization': 'Bearer ' + localStorage.getItem('token')
				}
			})
			.then(response => response.json())
			.then(data => {
				displayQualityResults(data);
				document.getElementById('qualityResultsSection').style.display = 'block';
			})
			.catch(error => {
				alert('Error during quality inspection: ' + error.message);
			})
			.finally(() => {
				submitBtn.disabled = false;
				submitBtn.innerHTML = '<i class="fa fa-check-circle"></i> Start Inspection';
			});
		});
		
		function displayQualityResults(data) {
			var statusClass = data.inspection_result === 'PASS' ? 'success' : 
			                 data.inspection_result === 'FAIL' ? 'danger' : 'warning';
			
			var resultsHtml = `
				<div class="row">
					<div class="col-md-12">
						<div class="alert alert-${statusClass}">
							<h4><i class="fa fa-${data.inspection_result === 'PASS' ? 'check' : 'times'}"></i> 
							    Inspection ${data.inspection_result}</h4>
							<p>Overall Quality Score: <strong>${(data.overall_score * 100).toFixed(1)}%</strong></p>
						</div>
					</div>
				</div>
				
				<div class="row">
					<div class="col-md-6">
						<h4>Inspection Summary</h4>
						<table class="table table-striped">
							<tr><th>Result</th><td><span class="label label-${statusClass}">${data.inspection_result}</span></td></tr>
							<tr><th>Quality Score</th><td>${(data.overall_score * 100).toFixed(1)}%</td></tr>
							<tr><th>Defects Found</th><td>${data.defect_count}</td></tr>
							<tr><th>Processing Time</th><td>${data.processing_time_ms}ms</td></tr>
						</table>
					</div>
					<div class="col-md-6">
						<h4>Compliance Status</h4>
						<div id="complianceStatus">
							${Object.keys(data.compliance_status || {}).length > 0 ? 
								Object.entries(data.compliance_status).map(([standard, status]) => 
									`<span class="label label-${status === 'compliant' ? 'success' : 'danger'}">${standard}: ${status}</span> `
								).join('') : 
								'<p class="text-muted">No compliance standards specified</p>'
							}
						</div>
					</div>
				</div>
			`;
			
			if (data.defects_detected && data.defects_detected.length > 0) {
				resultsHtml += `
					<div class="row">
						<div class="col-md-12">
							<h4>Detected Defects</h4>
							<div class="table-responsive">
								<table class="table table-striped">
									<thead>
										<tr>
											<th>Defect Type</th>
											<th>Severity</th>
											<th>Confidence</th>
											<th>Location</th>
											<th>Description</th>
										</tr>
									</thead>
									<tbody>
										${data.defects_detected.map(defect => {
											var severityClass = defect.severity === 'CRITICAL' ? 'danger' :
											                   defect.severity === 'MAJOR' ? 'warning' : 'info';
											return `
												<tr>
													<td>${defect.defect_type}</td>
													<td><span class="label label-${severityClass}">${defect.severity}</span></td>
													<td>${(defect.confidence * 100).toFixed(1)}%</td>
													<td>x:${Math.round(defect.location.x)}, y:${Math.round(defect.location.y)}</td>
													<td>${defect.description}</td>
												</tr>
											`;
										}).join('')}
									</tbody>
								</table>
							</div>
						</div>
					</div>
				`;
			}
			
			document.getElementById('qualityResults').innerHTML = resultsHtml;
		}
		</script>
		"""
		
		return render_template_string(quality_workspace_template)


class ComputerVisionVideoView(ComputerVisionBaseView):
	"""Video analysis workspace for action recognition and event detection"""
	
	route_base = "/computer_vision/video"
	default_view = "workspace"
	
	@expose("/")
	@expose("/workspace")
	@has_access
	def workspace(self):
		"""Video analysis workspace"""
		video_workspace_template = """
		<div class="row">
			<div class="col-md-12">
				<h2>Video Analysis & Processing</h2>
				<p>Analyze video content for actions, events, and temporal patterns</p>
			</div>
		</div>
		
		<div class="row">
			<div class="col-md-8">
				<div class="panel panel-primary">
					<div class="panel-heading">
						<h3 class="panel-title">Video Upload & Analysis</h3>
					</div>
					<div class="panel-body">
						<form id="videoForm" enctype="multipart/form-data">
							<div class="form-group">
								<label for="videoFile">Select Video File</label>
								<input type="file" class="form-control" id="videoFile" name="file" 
								       accept=".mp4,.avi,.mov,.mkv,.webm" required>
								<small class="help-block">Supported formats: MP4, AVI, MOV, MKV, WebM (Max: 100MB)</small>
							</div>
							
							<div class="form-group">
								<label for="analysisType">Analysis Type</label>
								<select class="form-control" id="analysisType" name="analysis_type">
									<option value="general">General Analysis</option>
									<option value="action_recognition">Action Recognition</option>
									<option value="object_tracking">Object Tracking</option>
									<option value="scene_analysis">Scene Analysis</option>
									<option value="motion_detection">Motion Detection</option>
								</select>
							</div>
							
							<div class="form-group">
								<label for="frameSampling">Frame Sampling Rate (FPS)</label>
								<input type="number" class="form-control" id="frameSampling" name="frame_sampling_rate" 
								       min="1" max="30" value="1">
								<small class="help-block">Higher values provide more detail but increase processing time</small>
							</div>
							
							<div class="checkbox">
								<label>
									<input type="checkbox" id="detectObjects" name="detect_objects" checked>
									Enable object detection in frames
								</label>
							</div>
							
							<div class="checkbox">
								<label>
									<input type="checkbox" id="extractKeyframes" name="extract_keyframes">
									Extract key frames for detailed analysis
								</label>
							</div>
							
							<button type="submit" class="btn btn-primary">
								<i class="fa fa-play"></i> Analyze Video
							</button>
						</form>
					</div>
				</div>
			</div>
			
			<div class="col-md-4">
				<div class="panel panel-info">
					<div class="panel-heading">
						<h3 class="panel-title">Analysis Types</h3>
					</div>
					<div class="panel-body">
						<h5>General Analysis</h5>
						<p>Overall video statistics, scene changes, and basic content analysis.</p>
						
						<h5>Action Recognition</h5>
						<p>Identify human actions and activities throughout the video.</p>
						
						<h5>Object Tracking</h5>
						<p>Track objects across frames and analyze movement patterns.</p>
						
						<h5>Scene Analysis</h5>
						<p>Detect scene transitions and classify different environments.</p>
						
						<h5>Motion Detection</h5>
						<p>Analyze motion patterns and detect significant movement events.</p>
					</div>
				</div>
			</div>
		</div>
		
		<div class="row" id="videoResultsSection" style="display: none;">
			<div class="col-md-12">
				<div class="panel panel-success">
					<div class="panel-heading">
						<h3 class="panel-title">Video Analysis Results</h3>
					</div>
					<div class="panel-body">
						<div id="videoResults"></div>
					</div>
				</div>
			</div>
		</div>
		
		<script>
		document.getElementById('videoForm').addEventListener('submit', function(e) {
			e.preventDefault();
			
			var formData = new FormData(this);
			var submitBtn = this.querySelector('button[type="submit"]');
			var fileInput = document.getElementById('videoFile');
			
			// Check file size (100MB limit)
			if (fileInput.files[0] && fileInput.files[0].size > 100 * 1024 * 1024) {
				alert('File size must be less than 100MB');
				return;
			}
			
			submitBtn.disabled = true;
			submitBtn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Processing Video...';
			
			fetch('/api/v1/videos/analyze', {
				method: 'POST',
				body: formData,
				headers: {
					'Authorization': 'Bearer ' + localStorage.getItem('token')
				}
			})
			.then(response => response.json())
			.then(data => {
				displayVideoResults(data);
				document.getElementById('videoResultsSection').style.display = 'block';
			})
			.catch(error => {
				alert('Error analyzing video: ' + error.message);
			})
			.finally(() => {
				submitBtn.disabled = false;
				submitBtn.innerHTML = '<i class="fa fa-play"></i> Analyze Video';
			});
		});
		
		function displayVideoResults(data) {
			var props = data.video_properties || {};
			var analysis = data.analysis_results || {};
			
			var resultsHtml = `
				<div class="row">
					<div class="col-md-6">
						<h4>Video Properties</h4>
						<table class="table table-striped">
							<tr><th>Duration</th><td>${props.duration_seconds ? props.duration_seconds.toFixed(1) + 's' : 'Unknown'}</td></tr>
							<tr><th>Frame Count</th><td>${props.frame_count || 'Unknown'}</td></tr>
							<tr><th>Frame Rate</th><td>${props.fps ? props.fps.toFixed(1) + ' FPS' : 'Unknown'}</td></tr>
							<tr><th>Resolution</th><td>${props.width && props.height ? props.width + 'x' + props.height : 'Unknown'}</td></tr>
							<tr><th>Processing Time</th><td>${data.processing_time_ms}ms</td></tr>
						</table>
					</div>
					<div class="col-md-6">
						<h4>Analysis Summary</h4>
						<table class="table table-striped">
							<tr><th>Actions Detected</th><td>${analysis.actions_detected ? analysis.actions_detected.length : 0}</td></tr>
							<tr><th>Events</th><td>${analysis.events ? analysis.events.length : 0}</td></tr>
							<tr><th>Scene Changes</th><td>${analysis.scene_changes ? analysis.scene_changes.length : 0}</td></tr>
							<tr><th>Motion Analysis</th><td>${analysis.motion_analysis ? 'Available' : 'Not available'}</td></tr>
						</table>
					</div>
				</div>
			`;
			
			// Add detailed results sections as they become available
			if (analysis.actions_detected && analysis.actions_detected.length > 0) {
				resultsHtml += `
					<div class="row">
						<div class="col-md-12">
							<h4>Detected Actions</h4>
							<div class="table-responsive">
								<table class="table table-striped">
									<thead>
										<tr><th>Time</th><th>Action</th><th>Confidence</th><th>Duration</th></tr>
									</thead>
									<tbody>
										${analysis.actions_detected.map(action => 
											`<tr><td>${action.timestamp}s</td><td>${action.action_name}</td><td>${(action.confidence * 100).toFixed(1)}%</td><td>${action.duration}s</td></tr>`
										).join('')}
									</tbody>
								</table>
							</div>
						</div>
					</div>
				`;
			}
			
			document.getElementById('videoResults').innerHTML = resultsHtml;
		}
		</script>
		"""
		
		return render_template_string(video_workspace_template)


class ComputerVisionModelView(ComputerVisionBaseView):
	"""Model management workspace for AI model deployment and monitoring"""
	
	route_base = "/computer_vision/models"
	default_view = "workspace"
	
	@expose("/")
	@expose("/workspace")
	@has_access
	def workspace(self):
		"""Model management workspace"""
		model_workspace_template = """
		<div class="row">
			<div class="col-md-12">
				<h2>AI Model Management</h2>
				<p>Deploy, monitor, and manage computer vision models across the platform</p>
			</div>
		</div>
		
		<div class="row">
			<div class="col-md-12">
				<div class="panel panel-primary">
					<div class="panel-heading">
						<h3 class="panel-title">Available Models</h3>
					</div>
					<div class="panel-body">
						<div class="table-responsive">
							<table class="table table-striped" id="modelsTable">
								<thead>
									<tr>
										<th>Model Name</th>
										<th>Type</th>
										<th>Version</th>
										<th>Status</th>
										<th>Accuracy</th>
										<th>Inference Time</th>
										<th>Last Used</th>
										<th>Actions</th>
									</tr>
								</thead>
								<tbody>
									<tr>
										<td>YOLOv8 Nano</td>
										<td>Object Detection</td>
										<td>8.0.0</td>
										<td><span class="label label-success">Active</span></td>
										<td>85.2%</td>
										<td>45ms</td>
										<td>2 hours ago</td>
										<td>
											<button class="btn btn-sm btn-info">Monitor</button>
											<button class="btn btn-sm btn-warning">Configure</button>
										</td>
									</tr>
									<tr>
										<td>Vision Transformer</td>
										<td>Image Classification</td>
										<td>1.0.0</td>
										<td><span class="label label-success">Active</span></td>
										<td>92.1%</td>
										<td>150ms</td>
										<td>1 hour ago</td>
										<td>
											<button class="btn btn-sm btn-info">Monitor</button>
											<button class="btn btn-sm btn-warning">Configure</button>
										</td>
									</tr>
									<tr>
										<td>Tesseract OCR</td>
										<td>Text Recognition</td>
										<td>5.3.0</td>
										<td><span class="label label-success">Active</span></td>
										<td>94.7%</td>
										<td>800ms</td>
										<td>30 minutes ago</td>
										<td>
											<button class="btn btn-sm btn-info">Monitor</button>
											<button class="btn btn-sm btn-warning">Configure</button>
										</td>
									</tr>
									<tr>
										<td>FaceNet</td>
										<td>Facial Recognition</td>
										<td>2.1.0</td>
										<td><span class="label label-warning">Testing</span></td>
										<td>96.8%</td>
										<td>200ms</td>
										<td>Never</td>
										<td>
											<button class="btn btn-sm btn-success">Activate</button>
											<button class="btn btn-sm btn-danger">Remove</button>
										</td>
									</tr>
								</tbody>
							</table>
						</div>
					</div>
				</div>
			</div>
		</div>
		
		<div class="row">
			<div class="col-md-6">
				<div class="panel panel-info">
					<div class="panel-heading">
						<h3 class="panel-title">Model Performance</h3>
					</div>
					<div class="panel-body">
						<canvas id="performanceChart" width="400" height="200"></canvas>
					</div>
				</div>
			</div>
			<div class="col-md-6">
				<div class="panel panel-default">
					<div class="panel-heading">
						<h3 class="panel-title">Usage Statistics</h3>
					</div>
					<div class="panel-body">
						<canvas id="usageChart" width="400" height="200"></canvas>
					</div>
				</div>
			</div>
		</div>
		
		<div class="row">
			<div class="col-md-12">
				<div class="panel panel-default">
					<div class="panel-heading">
						<h3 class="panel-title">Model Deployment</h3>
					</div>
					<div class="panel-body">
						<form id="deploymentForm">
							<div class="row">
								<div class="col-md-4">
									<div class="form-group">
										<label for="modelFile">Upload Model</label>
										<input type="file" class="form-control" id="modelFile" name="model_file" 
										       accept=".pt,.onnx,.pb,.tflite">
									</div>
								</div>
								<div class="col-md-4">
									<div class="form-group">
										<label for="modelType">Model Type</label>
										<select class="form-control" id="modelType" name="model_type">
											<option value="object_detection">Object Detection</option>
											<option value="image_classification">Image Classification</option>
											<option value="facial_recognition">Facial Recognition</option>
											<option value="ocr">Text Recognition</option>
										</select>
									</div>
								</div>
								<div class="col-md-4">
									<div class="form-group">
										<label for="modelName">Model Name</label>
										<input type="text" class="form-control" id="modelName" name="model_name" 
										       placeholder="Enter model name" required>
									</div>
								</div>
							</div>
							
							<div class="row">
								<div class="col-md-12">
									<button type="submit" class="btn btn-primary">
										<i class="fa fa-cloud-upload"></i> Deploy Model
									</button>
								</div>
							</div>
						</form>
					</div>
				</div>
			</div>
		</div>
		
		<script>
		// Performance chart
		var perfCtx = document.getElementById('performanceChart').getContext('2d');
		var performanceChart = new Chart(perfCtx, {
			type: 'line',
			data: {
				labels: ['6h ago', '5h ago', '4h ago', '3h ago', '2h ago', '1h ago', 'Now'],
				datasets: [{
					label: 'Inference Time (ms)',
					data: [180, 165, 150, 145, 140, 135, 130],
					borderColor: '#36A2EB',
					tension: 0.1
				}, {
					label: 'Accuracy (%)',
					data: [89, 90, 91, 92, 93, 92, 93],
					borderColor: '#4BC0C0',
					tension: 0.1
				}]
			},
			options: {
				responsive: true,
				scales: {
					y: {
						beginAtZero: true
					}
				}
			}
		});
		
		// Usage chart
		var usageCtx = document.getElementById('usageChart').getContext('2d');
		var usageChart = new Chart(usageCtx, {
			type: 'bar',
			data: {
				labels: ['Object Detection', 'Classification', 'OCR', 'Face Recognition'],
				datasets: [{
					label: 'Requests Today',
					data: [150, 89, 67, 23],
					backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
				}]
			},
			options: {
				responsive: true,
				scales: {
					y: {
						beginAtZero: true
					}
				}
			}
		});
		
		// Deployment form
		document.getElementById('deploymentForm').addEventListener('submit', function(e) {
			e.preventDefault();
			alert('Model deployment functionality will be implemented in the full version.');
		});
		</script>
		"""
		
		return render_template_string(model_workspace_template)


# Export all view classes
__all__ = [
	# View Models
	'CVBaseViewModel', 'ProcessingJobViewModel', 'DashboardStatsViewModel',
	'OCRResultViewModel', 'ObjectDetectionResultViewModel', 'QualityControlResultViewModel',
	
	# Flask-AppBuilder Views
	'ComputerVisionBaseView', 'ComputerVisionDashboardView',
	'ComputerVisionDocumentView', 'ComputerVisionImageView',
	'ComputerVisionQualityView', 'ComputerVisionVideoView',
	'ComputerVisionModelView'
]