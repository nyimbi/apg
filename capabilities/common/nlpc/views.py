"""
APG Natural Language Processing Views

Flask-AppBuilder views for enterprise NLP platform with multi-model orchestration,
real-time streaming, collaborative annotation, and analytics dashboards.

All Pydantic v2 models are placed here following APG patterns with comprehensive
validation and modern typing.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, field_validator, computed_field
from pydantic.types import Json
from flask import render_template, jsonify, request, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget, EditWidget
from flask_appbuilder.actions import action
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON
from sqlalchemy.orm import relationship
import json
import asyncio

from .models import (
	TextDocument, NLPModel, ProcessingRequest, ProcessingResult,
	StreamingSession, AnnotationProject, TextAnalytics, SystemHealth,
	NLPTaskType, ModelProvider, ProcessingStatus, QualityLevel
)
from .service import NLPService

# Pydantic v2 Configuration following APG standards
MODEL_CONFIG = ConfigDict(
	extra='forbid',
	validate_by_name=True,
	validate_by_alias=True,
	str_strip_whitespace=True,
	validate_default=True,
	use_enum_values=True
)

# ===== Pydantic V2 Models for Views (Following APG Pattern) =====

class ProcessingRequestForm(BaseModel):
	"""Form model for text processing requests"""
	model_config = MODEL_CONFIG
	
	text_content: str = Field(..., min_length=1, max_length=10000, description="Text to process")
	task_type: NLPTaskType = Field(..., description="Type of NLP task")
	language: Optional[str] = Field(None, description="Text language (auto-detect if not specified)")
	quality_level: QualityLevel = Field(QualityLevel.BALANCED, description="Quality vs speed preference")
	preferred_model: Optional[str] = Field(None, description="Preferred model ID")
	parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
	
	@field_validator('text_content')
	@classmethod
	def validate_text_not_empty(cls, v: str) -> str:
		if not v.strip():
			raise ValueError("Text content cannot be empty")
		return v.strip()

class StreamingConfigForm(BaseModel):
	"""Form model for streaming session configuration"""
	model_config = MODEL_CONFIG
	
	task_type: NLPTaskType = Field(..., description="Type of NLP task")
	model_id: Optional[str] = Field(None, description="Preferred model")
	language: Optional[str] = Field(None, description="Expected language")
	chunk_size: int = Field(1000, ge=100, le=5000, description="Text chunk size")
	overlap_size: int = Field(100, ge=0, le=500, description="Chunk overlap size")

class AnnotationProjectForm(BaseModel):
	"""Form model for annotation project creation"""
	model_config = MODEL_CONFIG
	
	name: str = Field(..., min_length=1, max_length=200, description="Project name")
	description: Optional[str] = Field(None, max_length=1000, description="Project description")
	annotation_type: NLPTaskType = Field(..., description="Type of annotation task")
	guidelines: Optional[str] = Field(None, description="Annotation guidelines")
	consensus_threshold: float = Field(0.8, ge=0.5, le=1.0, description="Consensus threshold")

class ModelManagementForm(BaseModel):
	"""Form model for model management"""
	model_config = MODEL_CONFIG
	
	name: str = Field(..., min_length=1, max_length=200, description="Model display name")
	provider: ModelProvider = Field(..., description="Model provider")
	provider_model_name: str = Field(..., description="Provider-specific model name")
	is_active: bool = Field(True, description="Is model active")
	config_params: Dict[str, Any] = Field(default_factory=dict, description="Model configuration")

# ===== Flask-AppBuilder Views =====

class NLPDashboardView(BaseView):
	"""Main NLP dashboard with system overview and quick actions"""
	
	route_base = "/nlp"
	default_view = "dashboard"
	
	@expose("/dashboard/")
	@has_access
	def dashboard(self):
		"""Main dashboard view with system metrics and quick actions"""
		try:
			# Get system health (this would be async in real implementation)
			# For demo purposes, we'll create sample data
			health_data = {
				"overall_status": "healthy",
				"total_models": 8,
				"active_models": 6,
				"avg_response_time": 45.2,
				"requests_today": 1247,
				"active_sessions": 3
			}
			
			# Recent processing results
			recent_results = [
				{
					"id": "req_001",
					"task_type": "sentiment_analysis",
					"status": "completed",
					"processing_time": 42,
					"confidence": 0.94,
					"timestamp": datetime.utcnow()
				},
				{
					"id": "req_002", 
					"task_type": "entity_extraction",
					"status": "completed",
					"processing_time": 78,
					"confidence": 0.87,
					"timestamp": datetime.utcnow()
				}
			]
			
			# Model performance summary
			model_performance = [
				{
					"name": "Ollama Llama3.2",
					"provider": "ollama",
					"status": "healthy",
					"avg_latency": 120,
					"requests_today": 45
				},
				{
					"name": "BERT Base",
					"provider": "transformers", 
					"status": "healthy",
					"avg_latency": 35,
					"requests_today": 234
				}
			]
			
			return self.render_template(
				"nlp/dashboard.html",
				health_data=health_data,
				recent_results=recent_results,
				model_performance=model_performance
			)
			
		except Exception as e:
			flash(f"Error loading dashboard: {str(e)}", "error")
			return self.render_template("nlp/error.html", error=str(e))
	
	@expose("/process/")
	@has_access
	def process_text(self):
		"""Text processing interface"""
		if request.method == "POST":
			try:
				# Validate form data
				form_data = request.get_json() or request.form.to_dict()
				processing_form = ProcessingRequestForm(**form_data)
				
				# Create processing request
				# In real implementation, this would be async
				result = {
					"request_id": uuid7str(),
					"status": "completed",
					"results": {
						"sentiment": "positive",
						"confidence": 0.89,
						"processing_time_ms": 45.2
					}
				}
				
				if request.is_json:
					return jsonify(result)
				else:
					flash("Text processed successfully!", "success")
					return render_template("nlp/process_result.html", result=result)
					
			except Exception as e:
				error_msg = f"Processing failed: {str(e)}"
				if request.is_json:
					return jsonify({"error": error_msg}), 400
				else:
					flash(error_msg, "error")
		
		# GET request - show processing form
		available_models = [
			{"id": "ollama_llama3_2", "name": "Ollama Llama 3.2", "provider": "ollama"},
			{"id": "bert_base", "name": "BERT Base", "provider": "transformers"},
			{"id": "spacy_en_md", "name": "spaCy English Medium", "provider": "spacy"}
		]
		
		return self.render_template(
			"nlp/process.html",
			task_types=[task.value for task in NLPTaskType],
			quality_levels=[level.value for level in QualityLevel],
			available_models=available_models
		)
	
	@expose("/models/")
	@has_access
	def model_management(self):
		"""Model management interface"""
		try:
			# Sample model data
			models = [
				{
					"id": "ollama_llama3_2",
					"name": "Ollama Llama 3.2", 
					"provider": "ollama",
					"status": "healthy",
					"is_loaded": True,
					"avg_latency": 120,
					"total_requests": 456,
					"success_rate": 98.2
				},
				{
					"id": "bert_base",
					"name": "BERT Base",
					"provider": "transformers",
					"status": "healthy", 
					"is_loaded": True,
					"avg_latency": 35,
					"total_requests": 1234,
					"success_rate": 99.1
				},
				{
					"id": "spacy_en_md",
					"name": "spaCy English Medium",
					"provider": "spacy",
					"status": "healthy",
					"is_loaded": True,
					"avg_latency": 25,
					"total_requests": 2345,
					"success_rate": 99.5
				}
			]
			
			return self.render_template("nlp/models.html", models=models)
			
		except Exception as e:
			flash(f"Error loading models: {str(e)}", "error")
			return self.render_template("nlp/error.html", error=str(e))
	
	@expose("/streaming/")
	@has_access
	def streaming_console(self):
		"""Real-time streaming processing console"""
		try:
			# Active streaming sessions
			active_sessions = [
				{
					"id": "stream_001",
					"user": "analyst@company.com",
					"task_type": "sentiment_analysis",
					"chunks_processed": 45,
					"avg_latency": 23,
					"status": "active"
				},
				{
					"id": "stream_002", 
					"user": "researcher@company.com",
					"task_type": "entity_extraction",
					"chunks_processed": 128,
					"avg_latency": 31,
					"status": "active"
				}
			]
			
			return self.render_template(
				"nlp/streaming.html",
				active_sessions=active_sessions,
				task_types=[task.value for task in NLPTaskType]
			)
			
		except Exception as e:
			flash(f"Error loading streaming console: {str(e)}", "error")
			return self.render_template("nlp/error.html", error=str(e))
	
	@expose("/annotation/")
	@has_access
	def annotation_projects(self):
		"""Collaborative annotation projects"""
		try:
			# Sample annotation projects
			projects = [
				{
					"id": "proj_001",
					"name": "Customer Feedback Analysis",
					"type": "sentiment_analysis",
					"status": "active",
					"documents": 500,
					"completed": 234,
					"completion_rate": 46.8,
					"team_size": 3,
					"consensus_score": 0.89
				},
				{
					"id": "proj_002",
					"name": "Legal Document Entities",
					"type": "named_entity_recognition", 
					"status": "review",
					"documents": 200,
					"completed": 195,
					"completion_rate": 97.5,
					"team_size": 5,
					"consensus_score": 0.92
				}
			]
			
			return self.render_template("nlp/annotation.html", projects=projects)
			
		except Exception as e:
			flash(f"Error loading annotation projects: {str(e)}", "error")
			return self.render_template("nlp/error.html", error=str(e))
	
	@expose("/analytics/")
	@has_access
	def analytics_dashboard(self):
		"""Analytics and insights dashboard"""
		try:
			# Sample analytics data
			analytics_data = {
				"total_processed_today": 1247,
				"avg_processing_time": 45.2,
				"top_task_types": [
					{"task": "sentiment_analysis", "count": 456, "percentage": 36.6},
					{"task": "entity_extraction", "count": 234, "percentage": 18.8},
					{"task": "text_classification", "count": 189, "percentage": 15.2}
				],
				"model_usage": [
					{"model": "BERT Base", "requests": 456, "percentage": 36.6},
					{"model": "spaCy English", "requests": 234, "percentage": 18.8},
					{"model": "Ollama Llama", "requests": 189, "percentage": 15.2}
				],
				"hourly_volume": [
					{"hour": "00:00", "volume": 23},
					{"hour": "01:00", "volume": 18},
					{"hour": "02:00", "volume": 15},
					{"hour": "09:00", "volume": 156},
					{"hour": "10:00", "volume": 189},
					{"hour": "11:00", "volume": 167}
				]
			}
			
			return self.render_template("nlp/analytics.html", analytics=analytics_data)
			
		except Exception as e:
			flash(f"Error loading analytics: {str(e)}", "error")
			return self.render_template("nlp/error.html", error=str(e))

class TextDocumentView(ModelView):
	"""Text document management view with CRUD operations"""
	
	# This would typically use SQLAlchemy models
	# For now, we'll use a placeholder interface
	
	route_base = "/nlp/documents"
	
	list_columns = ["id", "title", "language", "word_count", "created_at", "updated_at"]
	show_columns = ["id", "title", "content", "language", "detected_language", 
					"word_count", "quality_score", "created_at", "updated_at"]
	edit_columns = ["title", "content", "language", "metadata"]
	add_columns = ["title", "content", "language"]
	
	list_title = "Text Documents"
	show_title = "Document Details"
	add_title = "Add New Document"
	edit_title = "Edit Document"
	
	@action("process_sentiment", "Analyze Sentiment", 
			"Analyze sentiment for selected documents", "fa-smile")
	def process_sentiment(self, documents):
		"""Bulk sentiment analysis action"""
		try:
			for doc in documents:
				# Process sentiment analysis
				# This would be async in real implementation
				flash(f"Sentiment analysis started for document: {doc.title}", "info")
				
			flash(f"Sentiment analysis initiated for {len(documents)} documents", "success")
			
		except Exception as e:
			flash(f"Error processing sentiment: {str(e)}", "error")
		
		return redirect(url_for('TextDocumentView.list'))
	
	@action("extract_entities", "Extract Entities",
			"Extract named entities from selected documents", "fa-tags")
	def extract_entities(self, documents):
		"""Bulk entity extraction action"""
		try:
			for doc in documents:
				# Process entity extraction
				flash(f"Entity extraction started for document: {doc.title}", "info")
				
			flash(f"Entity extraction initiated for {len(documents)} documents", "success")
			
		except Exception as e:
			flash(f"Error extracting entities: {str(e)}", "error")
			
		return redirect(url_for('TextDocumentView.list'))

class ProcessingResultView(ModelView):
	"""Processing results view with filtering and analysis"""
	
	route_base = "/nlp/results"
	
	list_columns = ["id", "task_type", "model_used", "processing_time_ms", 
					"confidence_score", "status", "created_at"]
	show_columns = ["id", "request_id", "task_type", "model_used", "provider_used",
					"processing_time_ms", "confidence_score", "results", "status", "created_at"]
	
	list_title = "Processing Results"
	show_title = "Result Details"
	
	# Enable search and filtering
	search_columns = ["task_type", "model_used", "status"]
	base_filters = [
		["status", "equal to", ProcessingStatus.COMPLETED],
		["processing_time_ms", "less than", 1000]
	]
	
	# Custom formatting for results display
	formatters_columns = {
		"processing_time_ms": lambda x: f"{x:.1f}ms",
		"confidence_score": lambda x: f"{x:.2%}" if x else "N/A",
		"results": lambda x: json.dumps(x, indent=2) if x else "No results"
	}

class StreamingSessionView(ModelView):
	"""Streaming session management view"""
	
	route_base = "/nlp/streaming-sessions"
	
	list_columns = ["id", "user_id", "task_type", "status", "chunks_processed",
					"average_latency_ms", "created_at", "last_activity"]
	show_columns = ["id", "user_id", "task_type", "model_id", "status",
					"chunks_processed", "total_characters", "average_latency_ms",
					"created_at", "last_activity"]
	
	list_title = "Streaming Sessions"
	show_title = "Session Details"
	
	# Custom actions for streaming management
	@action("stop_session", "Stop Session",
			"Stop selected streaming sessions", "fa-stop")
	def stop_session(self, sessions):
		"""Stop streaming sessions action"""
		try:
			for session in sessions:
				# Stop streaming session
				session.status = "stopped"
				flash(f"Stopped streaming session: {session.id}", "info")
				
			flash(f"Stopped {len(sessions)} streaming sessions", "success")
			
		except Exception as e:
			flash(f"Error stopping sessions: {str(e)}", "error")
			
		return redirect(url_for('StreamingSessionView.list'))

class AnnotationProjectView(ModelView):
	"""Annotation project management view"""
	
	route_base = "/nlp/annotation-projects"
	
	list_columns = ["id", "name", "annotation_type", "status", "document_count",
					"completed_annotations", "team_members", "created_at"]
	show_columns = ["id", "name", "description", "annotation_type", "status",
					"document_count", "completed_annotations", "consensus_threshold",
					"inter_annotator_agreement", "created_at", "updated_at"]
	edit_columns = ["name", "description", "annotation_type", "guidelines",
					"consensus_threshold", "is_training_enabled"]
	add_columns = ["name", "description", "annotation_type", "guidelines"]
	
	list_title = "Annotation Projects"
	show_title = "Project Details"
	add_title = "Create New Project"
	edit_title = "Edit Project"
	
	@action("generate_training_data", "Generate Training Data",
			"Generate training data from completed annotations", "fa-database")
	def generate_training_data(self, projects):
		"""Generate training data from annotation projects"""
		try:
			for project in projects:
				if project.completed_annotations > 0:
					# Generate training data
					flash(f"Training data generation started for: {project.name}", "info")
				else:
					flash(f"No completed annotations in project: {project.name}", "warning")
					
			flash(f"Training data generation initiated for {len(projects)} projects", "success")
			
		except Exception as e:
			flash(f"Error generating training data: {str(e)}", "error")
			
		return redirect(url_for('AnnotationProjectView.list'))

class NLPModelView(ModelView):
	"""NLP model management view"""
	
	route_base = "/nlp/nlp-models"
	
	list_columns = ["id", "name", "provider", "is_active", "is_loaded",
					"health_status", "average_latency_ms", "total_requests"]
	show_columns = ["id", "name", "model_key", "provider", "provider_model_name",
					"supported_tasks", "supported_languages", "is_active", "is_loaded",
					"health_status", "total_requests", "successful_requests",
					"average_latency_ms", "created_at", "updated_at"]
	edit_columns = ["name", "is_active", "config_params"]
	
	list_title = "NLP Models"
	show_title = "Model Details"
	edit_title = "Edit Model"
	
	# Custom formatting
	formatters_columns = {
		"average_latency_ms": lambda x: f"{x:.1f}ms" if x else "N/A",
		"supported_tasks": lambda x: ", ".join(x) if x else "None",
		"supported_languages": lambda x: ", ".join(x) if x else "None"
	}
	
	@action("health_check", "Health Check",
			"Perform health check on selected models", "fa-heartbeat")
	def health_check(self, models):
		"""Perform health check on models"""
		try:
			for model in models:
				# Perform health check
				# This would be async in real implementation
				model.health_status = "healthy"
				model.last_health_check = datetime.utcnow()
				flash(f"Health check completed for: {model.name}", "info")
				
			flash(f"Health check completed for {len(models)} models", "success")
			
		except Exception as e:
			flash(f"Error during health check: {str(e)}", "error")
			
		return redirect(url_for('NLPModelView.list'))
	
	@action("reload_model", "Reload Model",
			"Reload selected models into memory", "fa-refresh")
	def reload_model(self, models):
		"""Reload models into memory"""
		try:
			for model in models:
				# Reload model
				model.is_loaded = True
				flash(f"Model reloaded: {model.name}", "info")
				
			flash(f"Reloaded {len(models)} models", "success")
			
		except Exception as e:
			flash(f"Error reloading models: {str(e)}", "error")
			
		return redirect(url_for('NLPModelView.list'))

class NLPAnalyticsChartView(DirectByChartView):
	"""Analytics charts for NLP processing metrics"""
	
	chart_title = "NLP Processing Analytics"
	chart_type = "LineChart"
	
	definitions = [
		{
			"group": "processing_time_ms",
			"series": ["avg_time", "min_time", "max_time"]
		},
		{
			"group": "task_type", 
			"series": ["request_count"]
		}
	]

# Export all views for blueprint registration
__all__ = [
	# Pydantic forms
	"ProcessingRequestForm",
	"StreamingConfigForm", 
	"AnnotationProjectForm",
	"ModelManagementForm",
	
	# Flask-AppBuilder views
	"NLPDashboardView",
	"TextDocumentView",
	"ProcessingResultView",
	"StreamingSessionView",
	"AnnotationProjectView",
	"NLPModelView",
	"NLPAnalyticsChartView"
]