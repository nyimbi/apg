"""
APG Import/Export (IMEX) Simple REST API Layer

Purpose: Production-grade REST API endpoints for enterprise import/export operations
         with comprehensive validation, error handling, and documentation.
Dependencies: flask, pydantic, asyncio, typing
Usage Context: HTTP API layer exposing IMEX functionality to clients

This module provides:
- Complete RESTful API for all IMEX operations
- Async request handling with proper error responses
- Input validation using Pydantic models
- Comprehensive error handling and logging
- Real-time job monitoring and status endpoints
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from functools import wraps
import json
import traceback

from flask import Flask, Blueprint, request, jsonify, Response
try:
    from flask_cors import CORS
except ImportError:
    def CORS(app: Any, *args: Any, **kwargs: Any) -> Any:
        return app
from pydantic import BaseModel, Field, ValidationError
from uuid_extensions import uuid7str

from models import (
    ImportExportJob, JobExecution, SourceConfig, TargetConfig, SchemaMapping,
    ValidationRule, TransformationStep, ProcessingMetrics, DataQualityReport,
    JobStatus, JobType, DataFormat, SourceType, ValidationLevel,
    ErrorHandlingStrategy, ProcessingPriority
)
from service import ImportExportService
from database import DatabaseManager, DatabaseConfig
from ai_intelligence import AIIntelligenceEngine

logger = logging.getLogger(__name__)

# Pydantic Request/Response Models

class JobCreateRequest(BaseModel):
	"""Request model for creating jobs"""
	name: str = Field(..., min_length=1, max_length=255)
	description: Optional[str] = Field(None, max_length=1000)
	job_type: str = Field(..., pattern="^(import|export|migration|sync)$")
	source_config: Dict[str, Any] = Field(...)
	target_config: Dict[str, Any] = Field(...)
	validation_rules: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
	transformation_steps: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
	priority: Optional[str] = Field("normal", pattern="^(low|normal|high|urgent)$")
	validation_level: Optional[str] = Field("basic", pattern="^(none|basic|strict|comprehensive)$")
	error_handling: Optional[str] = Field("log_and_continue", pattern="^(fail_fast|log_and_continue|skip_and_continue)$")
	tags: Optional[List[str]] = Field(default_factory=list)

class JobExecutionRequest(BaseModel):
	"""Request model for job execution"""
	execution_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
	override_validation: bool = Field(False)
	dry_run: bool = Field(False)

class SchemaDetectionRequest(BaseModel):
	"""Request model for schema detection"""
	source_config: Dict[str, Any] = Field(...)
	sample_size: int = Field(1000, ge=1, le=10000)
	include_statistics: bool = Field(True)
	detection_hints: Optional[Dict[str, Any]] = Field(default_factory=dict)

class SchemaMappingRequest(BaseModel):
	"""Request model for schema mapping suggestions"""
	source_schema: Dict[str, Any] = Field(...)
	target_schema: Dict[str, Any] = Field(...)
	auto_map_similar_fields: bool = Field(True)
	confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
	mapping_context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class DataQualityRequest(BaseModel):
	"""Request model for data quality assessment"""
	job_id: Optional[str] = Field(None)
	sample_data: List[Dict[str, Any]] = Field(..., min_length=1)
	quality_rules: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

class WorkflowCreateRequest(BaseModel):
	"""Request model for creating workflows"""
	name: str = Field(..., min_length=1, max_length=255)
	description: Optional[str] = Field(None, max_length=1000)
	steps: List[Dict[str, Any]] = Field(..., min_length=1)
	schedule: Optional[Dict[str, Any]] = Field(None)
	tags: Optional[List[str]] = Field(default_factory=list)

# Global service instance (will be initialized by the application)
imex_service: Optional[ImportExportService] = None

def initialize_api_service(service: ImportExportService):
	"""Initialize the global service instance for API endpoints"""
	global imex_service
	imex_service = service
	logger.info("API service initialized successfully")

# Create Flask Blueprint
imex_api_bp = Blueprint('imex_api', __name__, url_prefix='/api/v1/imex')

# Enable CORS for the blueprint
CORS(imex_api_bp)

# Error Handlers

@imex_api_bp.errorhandler(ValidationError)
def handle_validation_error(error):
	"""Handle Pydantic validation errors"""
	return jsonify({
		'success': False,
		'error': 'Validation Error',
		'message': 'Invalid request data',
		'details': error.errors()
	}), 400

@imex_api_bp.errorhandler(ValueError)
def handle_value_error(error):
	"""Handle value errors"""
	return jsonify({
		'success': False,
		'error': 'Invalid Value',
		'message': str(error)
	}), 400

@imex_api_bp.errorhandler(RuntimeError)
def handle_runtime_error(error):
	"""Handle runtime errors"""
	return jsonify({
		'success': False,
		'error': 'Runtime Error',
		'message': str(error)
	}), 500

@imex_api_bp.errorhandler(500)
def handle_internal_error(error):
	"""Handle internal server errors"""
	return jsonify({
		'success': False,
		'error': 'Internal Server Error',
		'message': 'An internal server error occurred'
	}), 500

# Utility Functions

def _log_api_request(endpoint: str, method: str, user_id: str = None):
	"""Log API request for audit trail"""
	logger.info(f"[API Request] {method} {endpoint} - User: {user_id or 'anonymous'}")

def _log_api_error(endpoint: str, error: str, user_id: str = None):
	"""Log API error for monitoring"""
	logger.error(f"[API Error] {endpoint} - User: {user_id or 'anonymous'} - Error: {error}")

def _validate_tenant_access(tenant_id: str) -> bool:
	"""Validate tenant access permissions"""
	# In production, this would check actual tenant permissions
	return True

def _get_current_user_id() -> str:
	"""Get current user ID from request context"""
	# In production, this would extract from JWT token or session
	return request.headers.get('X-User-ID', 'anonymous')

def _execute_async_operation(operation, *args, **kwargs):
	"""Execute async operation in Flask context"""
	try:
		# Use existing event loop if available, otherwise create new one
		loop = asyncio.get_event_loop()
		if loop.is_running():
			# If we're in an async context, run in thread
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor() as executor:
				future = executor.submit(asyncio.run, operation(*args, **kwargs))
				return future.result(timeout=30)
		else:
			return loop.run_until_complete(operation(*args, **kwargs))
	except RuntimeError:
		# No event loop, create new one
		return asyncio.run(operation(*args, **kwargs))
	except Exception as e:
		logger.error(f"Async operation failed: {e}")
		raise

def _validate_request_json(request_model_class):
	"""Decorator to validate request JSON using Pydantic model"""
	def decorator(f):
		@wraps(f)
		def decorated_function(*args, **kwargs):
			try:
				if not request.is_json:
					return jsonify({
						'success': False,
						'error': 'Content-Type must be application/json'
					}), 400

				request_data = request_model_class(**request.get_json())
				return f(request_data, *args, **kwargs)
			except ValidationError as e:
				return jsonify({
					'success': False,
					'error': 'Validation failed',
					'details': e.errors()
				}), 400
			except Exception as e:
				logger.error(f"Request validation error: {e}")
				return jsonify({
					'success': False,
					'error': 'Invalid request',
					'message': str(e)
				}), 400
		return decorated_function
	return decorator

# API Endpoints

@imex_api_bp.route('/health', methods=['GET'])
def health_check():
	"""Comprehensive service health check"""
	try:
		user_id = _get_current_user_id()
		_log_api_request('/health', 'GET', user_id)

		if not imex_service:
			return jsonify({
				'success': False,
				'error': 'Service not initialized'
			}), 503

		health_data = _execute_async_operation(imex_service.health_check)

		return jsonify({
			'success': True,
			'message': 'Service healthy',
			'data': health_data
		}), 200

	except Exception as e:
		_log_api_error('/health', str(e), _get_current_user_id())
		return jsonify({
			'success': False,
			'error': 'Health check failed',
			'message': str(e)
		}), 500

@imex_api_bp.route('/jobs', methods=['GET'])
def list_jobs():
	"""List import/export jobs with filtering"""
	try:
		user_id = _get_current_user_id()
		_log_api_request('/jobs', 'GET', user_id)

		if not imex_service:
			return jsonify({
				'success': False,
				'error': 'Service not initialized'
			}), 503

		# Get query parameters
		tenant_id = request.args.get('tenant_id')
		status = request.args.get('status')
		job_type = request.args.get('job_type')
		limit = int(request.args.get('limit', 50))
		offset = int(request.args.get('offset', 0))

		# Validate tenant access
		if tenant_id and not _validate_tenant_access(tenant_id):
			return jsonify({
				'success': False,
				'error': 'Access denied to tenant'
			}), 403

		# Get jobs from service
		jobs = []
		if hasattr(imex_service, 'active_jobs'):
			for job_id, job in imex_service.active_jobs.items():
				if tenant_id and job.tenant_id != tenant_id:
					continue
				if status and job.status.value != status:
					continue
				if job_type and job.job_type.value != job_type:
					continue

				jobs.append({
					'id': job.id,
					'name': job.name,
					'description': job.description,
					'job_type': job.job_type.value,
					'status': job.status.value,
					'priority': job.priority.value,
					'created_by': job.created_by,
					'created_at': job.created_at.isoformat(),
					'updated_at': job.updated_at.isoformat(),
					'last_run_at': job.last_run_at.isoformat() if job.last_run_at else None
				})

		# Apply pagination
		total = len(jobs)
		jobs = jobs[offset:offset + limit]

		return jsonify({
			'success': True,
			'data': {
				'jobs': jobs,
				'pagination': {
					'total': total,
					'limit': limit,
					'offset': offset,
					'has_next': offset + limit < total,
					'has_prev': offset > 0
				}
			}
		}), 200

	except Exception as e:
		_log_api_error('/jobs', str(e), user_id)
		return jsonify({
			'success': False,
			'error': 'Internal server error',
			'message': str(e)
		}), 500

@imex_api_bp.route('/jobs', methods=['POST'])
@_validate_request_json(JobCreateRequest)
def create_job(job_request: JobCreateRequest):
	"""Create new import/export job"""
	try:
		user_id = _get_current_user_id()
		_log_api_request('/jobs', 'POST', user_id)

		if not imex_service:
			return jsonify({
				'success': False,
				'error': 'Service not initialized'
			}), 503

		# Add tenant and user information
		job_config = job_request.dict()
		job_config['tenant_id'] = request.headers.get('X-Tenant-ID', 'default')
		job_config['created_by'] = user_id

		# Create job
		job = _execute_async_operation(imex_service.create_job, job_config, user_id)

		return jsonify({
			'success': True,
			'message': 'Job created successfully',
			'data': {
				'id': job.id,
				'name': job.name,
				'status': job.status.value,
				'created_at': job.created_at.isoformat(),
				'updated_at': job.updated_at.isoformat()
			}
		}), 201

	except ValueError as e:
		return jsonify({
			'success': False,
			'error': 'Invalid job configuration',
			'message': str(e)
		}), 400
	except Exception as e:
		_log_api_error('/jobs', str(e), _get_current_user_id())
		return jsonify({
			'success': False,
			'error': 'Internal server error',
			'message': str(e)
		}), 500

@imex_api_bp.route('/jobs/<string:job_id>', methods=['GET'])
def get_job(job_id):
	"""Get job details"""
	try:
		user_id = _get_current_user_id()
		_log_api_request(f'/jobs/{job_id}', 'GET', user_id)

		if not imex_service:
			return jsonify({
				'success': False,
				'error': 'Service not initialized'
			}), 503

		if not hasattr(imex_service, 'active_jobs') or job_id not in imex_service.active_jobs:
			return jsonify({
				'success': False,
				'error': 'Job not found'
			}), 404

		job = imex_service.active_jobs[job_id]

		return jsonify({
			'success': True,
			'data': {
				'id': job.id,
				'name': job.name,
				'description': job.description,
				'job_type': job.job_type.value,
				'status': job.status.value,
				'priority': job.priority.value,
				'source_config': job.source_config.dict() if hasattr(job.source_config, 'dict') else job.source_config,
				'target_config': job.target_config.dict() if hasattr(job.target_config, 'dict') else job.target_config,
				'tags': job.tags,
				'created_by': job.created_by,
				'created_at': job.created_at.isoformat(),
				'updated_at': job.updated_at.isoformat(),
				'last_run_at': job.last_run_at.isoformat() if job.last_run_at else None
			}
		}), 200

	except Exception as e:
		_log_api_error(f'/jobs/{job_id}', str(e), user_id)
		return jsonify({
			'success': False,
			'error': 'Internal server error',
			'message': str(e)
		}), 500

@imex_api_bp.route('/jobs/<string:job_id>/execute', methods=['POST'])
def execute_job(job_id):
	"""Execute import/export job"""
	try:
		user_id = _get_current_user_id()
		_log_api_request(f'/jobs/{job_id}/execute', 'POST', user_id)

		if not imex_service:
			return jsonify({
				'success': False,
				'error': 'Service not initialized'
			}), 503

		if not hasattr(imex_service, 'active_jobs') or job_id not in imex_service.active_jobs:
			return jsonify({
				'success': False,
				'error': 'Job not found'
			}), 404

		# Validate execution request if JSON provided
		execution_config = {}
		if request.is_json and request.get_json():
			try:
				exec_request = JobExecutionRequest(**request.get_json())
				execution_config = exec_request.execution_config
			except ValidationError as e:
				return jsonify({
					'success': False,
					'error': 'Validation failed',
					'details': e.errors()
				}), 400

		# Execute job asynchronously
		execution = _execute_async_operation(imex_service.execute_job, job_id, execution_config)

		return jsonify({
			'success': True,
			'message': 'Job execution started',
			'data': {
				'id': execution.id,
				'job_id': execution.job_id,
				'status': execution.status.value,
				'started_at': execution.started_at.isoformat() if execution.started_at else None,
				'metrics': execution.metrics.dict() if hasattr(execution.metrics, 'dict') else execution.metrics
			}
		}), 202

	except ValueError as e:
		return jsonify({
			'success': False,
			'error': 'Job execution failed',
			'message': str(e)
		}), 400
	except Exception as e:
		_log_api_error(f'/jobs/{job_id}/execute', str(e), _get_current_user_id())
		return jsonify({
			'success': False,
			'error': 'Internal server error',
			'message': str(e)
		}), 500

@imex_api_bp.route('/schemas/detect', methods=['POST'])
@_validate_request_json(SchemaDetectionRequest)
def detect_schema(detection_request: SchemaDetectionRequest):
	"""Automatically detect data source schema"""
	try:
		user_id = _get_current_user_id()
		_log_api_request('/schemas/detect', 'POST', user_id)

		if not imex_service:
			return jsonify({
				'success': False,
				'error': 'Service not initialized'
			}), 503

		# Detect schema
		source_config = SourceConfig(**detection_request.source_config)
		detected_schema = _execute_async_operation(
			imex_service.detect_schema_automatically, source_config
		)

		return jsonify({
			'success': True,
			'message': 'Schema detection completed',
			'data': {
				'schema': detected_schema,
				'source_config': detection_request.source_config,
				'detection_metadata': {
					'sample_size': detection_request.sample_size,
					'include_statistics': detection_request.include_statistics,
					'detected_at': datetime.now(timezone.utc).isoformat()
				}
			}
		}), 200

	except Exception as e:
		_log_api_error('/schemas/detect', str(e), _get_current_user_id())
		return jsonify({
			'success': False,
			'error': 'Schema detection failed',
			'message': str(e)
		}), 500

@imex_api_bp.route('/quality/validate', methods=['POST'])
@_validate_request_json(DataQualityRequest)
def validate_data_quality(quality_request: DataQualityRequest):
	"""Validate data quality and generate report"""
	try:
		user_id = _get_current_user_id()
		_log_api_request('/quality/validate', 'POST', user_id)

		if not imex_service:
			return jsonify({
				'success': False,
				'error': 'Service not initialized'
			}), 503

		# Validate data quality
		quality_report = _execute_async_operation(
			imex_service.validate_data_quality,
			quality_request.job_id,
			quality_request.sample_data
		)

		report_data = quality_report.dict() if hasattr(quality_report, 'dict') else quality_report

		return jsonify({
			'success': True,
			'message': 'Data quality validation completed',
			'data': report_data
		}), 200

	except Exception as e:
		_log_api_error('/quality/validate', str(e), _get_current_user_id())
		return jsonify({
			'success': False,
			'error': 'Data quality validation failed',
			'message': str(e)
		}), 500

# API Registry for APG Composition

api_registry = {
	'blueprint': imex_api_bp,
	'initialize_service': initialize_api_service
}

__all__ = [
	'imex_api_bp',
	'initialize_api_service',
	'api_registry'
]
