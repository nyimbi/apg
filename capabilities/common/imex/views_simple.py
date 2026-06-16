"""
APG Import/Export (IMEX) Simplified Flask Views

Purpose: Production-grade web interface for enterprise import/export operations
         compatible with Pydantic models and our current service architecture.
Dependencies: flask, pydantic, wtforms
Usage Context: Web UI layer for IMEX capability management

This module provides:
- Simple Flask views compatible with Pydantic models
- Real-time job monitoring dashboard
- AI-powered schema detection interface
- Data quality assessment UI
- Job creation and management forms
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json
import asyncio

from flask import Flask, Blueprint, request, jsonify, render_template, flash, redirect, url_for
from pydantic import BaseModel, Field, ValidationError
from wtforms import Form, StringField, SelectField, TextAreaField, IntegerField, BooleanField
from wtforms.validators import DataRequired, Length, Optional as WTFOptional
from wtforms.widgets import TextArea

from models import JobType, DataFormat, SourceType, ValidationLevel, ErrorHandlingStrategy, ProcessingPriority
from service import ImportExportService

logger = logging.getLogger(__name__)

# Global service instance
imex_service: Optional[ImportExportService] = None

def set_imex_service(service: ImportExportService):
    """Set the global IMEX service instance"""
    global imex_service
    imex_service = service
    logger.info("IMEX service set for simplified views")

# Pydantic Request Models

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

class SchemaDetectionRequest(BaseModel):
    """Request model for schema detection"""
    source_config: Dict[str, Any] = Field(...)
    sample_size: int = Field(1000, ge=1, le=10000)
    include_statistics: bool = Field(True)

class DataQualityRequest(BaseModel):
    """Request model for data quality assessment"""
    sample_data: List[Dict[str, Any]] = Field(..., min_length=1)
    job_id: Optional[str] = Field(None)

# WTForms for HTML Forms

class JobCreateForm(Form):
    """Form for creating import/export jobs"""
    name = StringField('Job Name',
                      validators=[DataRequired(), Length(min=1, max=255)],
                      render_kw={'placeholder': 'Enter descriptive job name', 'class': 'form-control'})

    description = TextAreaField('Description',
                              validators=[WTFOptional(), Length(max=1000)],
                              render_kw={'placeholder': 'Optional job description', 'rows': 3, 'class': 'form-control'})

    job_type = SelectField('Job Type',
                          choices=[('import', 'Import'), ('export', 'Export'),
                                 ('migration', 'Migration'), ('sync', 'Sync')],
                          validators=[DataRequired()],
                          render_kw={'class': 'form-control'})

    priority = SelectField('Priority',
                          choices=[('low', 'Low'), ('normal', 'Normal'),
                                 ('high', 'High'), ('urgent', 'Urgent')],
                          default='normal',
                          validators=[DataRequired()],
                          render_kw={'class': 'form-control'})

    validation_level = SelectField('Validation Level',
                                 choices=[('none', 'None'), ('basic', 'Basic'),
                                        ('strict', 'Strict'), ('comprehensive', 'Comprehensive')],
                                 default='basic',
                                 validators=[DataRequired()],
                                 render_kw={'class': 'form-control'})

    error_handling = SelectField('Error Handling',
                               choices=[('fail_fast', 'Fail Fast'),
                                      ('log_and_continue', 'Log and Continue'),
                                      ('skip_and_continue', 'Skip and Continue')],
                               default='log_and_continue',
                               validators=[DataRequired()],
                               render_kw={'class': 'form-control'})

    source_config = TextAreaField('Source Configuration (JSON)',
                                 validators=[DataRequired()],
                                 render_kw={'placeholder': '{"source_type": "file", "format": "csv", "file_path": "/path/to/file.csv"}',
                                          'rows': 6, 'class': 'form-control json-editor'})

    target_config = TextAreaField('Target Configuration (JSON)',
                                 validators=[DataRequired()],
                                 render_kw={'placeholder': '{"target_type": "database", "format": "csv"}',
                                          'rows': 6, 'class': 'form-control json-editor'})

    tags = StringField('Tags (comma-separated)',
                      validators=[WTFOptional()],
                      render_kw={'placeholder': 'production, migration, customer-data', 'class': 'form-control'})

class SchemaDetectionForm(Form):
    """Form for schema detection"""
    source_type = SelectField('Source Type',
                             choices=[('file', 'File'), ('database', 'Database'),
                                    ('api', 'API'), ('stream', 'Stream')],
                             validators=[DataRequired()],
                             render_kw={'class': 'form-control'})

    format = SelectField('Data Format',
                        choices=[('csv', 'CSV'), ('json', 'JSON'), ('xml', 'XML'),
                               ('parquet', 'Parquet'), ('excel', 'Excel')],
                        validators=[DataRequired()],
                        render_kw={'class': 'form-control'})

    file_path = StringField('File Path',
                           validators=[WTFOptional(), Length(max=500)],
                           render_kw={'placeholder': '/path/to/data/file.csv', 'class': 'form-control'})

    sample_size = IntegerField('Sample Size',
                              default=1000,
                              validators=[WTFOptional()],
                              render_kw={'min': 1, 'max': 10000, 'class': 'form-control'})

    include_statistics = BooleanField('Include Statistics', default=True)
    has_header = BooleanField('Has Header Row', default=True)

# Create Blueprint
imex_views_bp = Blueprint('imex_views', __name__, url_prefix='/imex',
                         template_folder='templates')

# Utility Functions

def _execute_async_operation(operation, *args, **kwargs):
    """Execute async operation in Flask context"""
    try:
        return asyncio.run(operation(*args, **kwargs))
    except Exception as e:
        logger.error(f"Async operation failed: {e}")
        raise

def _parse_json_field(field_data: str) -> Any:
    """Parse JSON field data with error handling"""
    if not field_data:
        return {}
    try:
        return json.loads(field_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")

# View Routes

@imex_views_bp.route('/')
@imex_views_bp.route('/dashboard')
def dashboard():
    """Main IMEX dashboard"""
    try:
        # Get dashboard data
        dashboard_data = {
            'total_jobs': 0,
            'active_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'recent_jobs': [],
            'system_status': 'unknown'
        }

        if imex_service:
            try:
                health_data = _execute_async_operation(imex_service.health_check)
                perf_metrics = health_data.get('performance_metrics', {})

                dashboard_data.update({
                    'total_jobs': perf_metrics.get('jobs_created', 0),
                    'active_jobs': health_data.get('active_jobs', 0),
                    'completed_jobs': perf_metrics.get('jobs_completed', 0),
                    'failed_jobs': perf_metrics.get('jobs_failed', 0),
                    'system_status': health_data.get('status', 'unknown'),
                    'recent_jobs': _get_recent_jobs()
                })
            except Exception as e:
                logger.error(f"Error getting dashboard data: {e}")
                flash(f"Dashboard data unavailable: {str(e)}", "warning")
        else:
            flash("IMEX service not available", "error")

        return render_template('imex/dashboard.html',
                             dashboard_data=dashboard_data,
                             title="IMEX Dashboard")

    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return render_template('imex/error.html', error=str(e), title="Dashboard Error")

@imex_views_bp.route('/jobs')
def jobs_list():
    """List all jobs"""
    try:
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        status_filter = request.args.get('status', '')
        type_filter = request.args.get('job_type', '')

        # Get jobs data
        jobs_data = {'jobs': [], 'total': 0, 'pagination': {}}

        if imex_service and hasattr(imex_service, 'active_jobs'):
            try:
                jobs = []
                for job_id, job in imex_service.active_jobs.items():
                    if status_filter and str(job.status).lower() != status_filter.lower():
                        continue
                    if type_filter and str(job.job_type).lower() != type_filter.lower():
                        continue

                    jobs.append({
                        'id': job.id,
                        'name': job.name,
                        'description': job.description or '',
                        'job_type': str(job.job_type),
                        'status': str(job.status),
                        'priority': str(job.priority),
                        'created_by': job.created_by,
                        'created_at': job.created_at.strftime('%Y-%m-%d %H:%M:%S') if job.created_at else 'Unknown',
                        'tags': getattr(job, 'tags', [])
                    })

                # Pagination
                total = len(jobs)
                start = (page - 1) * per_page
                end = start + per_page
                paginated_jobs = jobs[start:end]

                jobs_data = {
                    'jobs': paginated_jobs,
                    'total': total,
                    'pagination': {
                        'page': page,
                        'per_page': per_page,
                        'total_pages': (total + per_page - 1) // per_page,
                        'has_prev': page > 1,
                        'has_next': end < total
                    }
                }
            except Exception as e:
                logger.error(f"Error getting jobs data: {e}")
                flash(f"Error loading jobs: {str(e)}", "error")

        return render_template('imex/jobs_list.html',
                             jobs_data=jobs_data,
                             status_filter=status_filter,
                             type_filter=type_filter,
                             title="IMEX Jobs")

    except Exception as e:
        logger.error(f"Jobs list error: {e}")
        return render_template('imex/error.html', error=str(e), title="Jobs List Error")

@imex_views_bp.route('/jobs/create', methods=['GET', 'POST'])
def create_job():
    """Create new job"""
    form = JobCreateForm(request.form)

    if request.method == 'POST' and form.validate():
        try:
            if not imex_service:
                flash("IMEX service not available", "error")
                return render_template('imex/job_create.html', form=form, title="Create Job")

            # Parse form data
            job_config = {
                'name': form.name.data,
                'description': form.description.data,
                'job_type': form.job_type.data,
                'priority': form.priority.data,
                'validation_level': form.validation_level.data,
                'error_handling': form.error_handling.data,
                'source_config': _parse_json_field(form.source_config.data),
                'target_config': _parse_json_field(form.target_config.data),
                'tags': [tag.strip() for tag in form.tags.data.split(',') if tag.strip()] if form.tags.data else [],
                'tenant_id': 'default',
                'created_by': 'web-user'
            }

            # Create job
            job = _execute_async_operation(imex_service.create_job, job_config, 'web-user')

            flash(f"Job '{job.name}' created successfully!", "success")
            return redirect(url_for('imex_views.job_detail', job_id=job.id))

        except ValueError as e:
            flash(f"Validation error: {str(e)}", "error")
        except json.JSONDecodeError as e:
            flash(f"Invalid JSON configuration: {str(e)}", "error")
        except Exception as e:
            logger.error(f"Job creation error: {e}")
            flash(f"Job creation failed: {str(e)}", "error")

    return render_template('imex/job_create.html', form=form, title="Create Import/Export Job")

@imex_views_bp.route('/jobs/<job_id>')
def job_detail(job_id):
    """View job details"""
    try:
        if not imex_service:
            flash("IMEX service not available", "error")
            return redirect(url_for('imex_views.dashboard'))

        # Get job data
        job_data = None
        if hasattr(imex_service, 'active_jobs') and job_id in imex_service.active_jobs:
            job = imex_service.active_jobs[job_id]

            job_data = {
                'id': job.id,
                'name': job.name,
                'description': job.description or '',
                'job_type': str(job.job_type),
                'status': str(job.status),
                'priority': str(job.priority),
                'source_config': job.source_config,
                'target_config': job.target_config,
                'validation_level': str(getattr(job, 'validation_level', 'basic')),
                'error_handling': str(getattr(job, 'error_handling', 'log_and_continue')),
                'tags': getattr(job, 'tags', []),
                'created_by': job.created_by,
                'created_at': job.created_at.strftime('%Y-%m-%d %H:%M:%S') if job.created_at else 'Unknown',
                'updated_at': job.updated_at.strftime('%Y-%m-%d %H:%M:%S') if job.updated_at else 'Unknown',
                'last_run_at': job.last_run_at.strftime('%Y-%m-%d %H:%M:%S') if job.last_run_at else None
            }

        if not job_data:
            flash("Job not found", "error")
            return redirect(url_for('imex_views.jobs_list'))

        return render_template('imex/job_detail.html',
                             job_data=job_data,
                             title=f"Job: {job_data['name']}")

    except Exception as e:
        logger.error(f"Job detail error: {e}")
        flash(f"Error loading job: {str(e)}", "error")
        return redirect(url_for('imex_views.jobs_list'))

@imex_views_bp.route('/jobs/<job_id>/execute', methods=['POST'])
def execute_job(job_id):
    """Execute a job"""
    try:
        if not imex_service:
            flash("IMEX service not available", "error")
            return redirect(url_for('imex_views.job_detail', job_id=job_id))

        # Execute job
        execution = _execute_async_operation(imex_service.execute_job, job_id, {})

        flash(f"Job execution started (ID: {execution.id})", "success")

    except ValueError as e:
        flash(f"Execution error: {str(e)}", "error")
    except Exception as e:
        logger.error(f"Job execution error: {e}")
        flash(f"Execution failed: {str(e)}", "error")

    return redirect(url_for('imex_views.job_detail', job_id=job_id))

@imex_views_bp.route('/schema/detect', methods=['GET', 'POST'])
def schema_detection():
    """Schema detection interface"""
    form = SchemaDetectionForm(request.form)
    schema_result = None

    if request.method == 'POST' and form.validate():
        try:
            if not imex_service:
                flash("IMEX service not available", "error")
                return render_template('imex/schema_detect.html', form=form, title="Schema Detection")

            # Build source config
            from models import SourceConfig, SourceType, DataFormat

            source_config = SourceConfig(
                source_type=SourceType(form.source_type.data),
                format=DataFormat(form.format.data),
                file_path=form.file_path.data if form.file_path.data else None,
                has_header=form.has_header.data,
                encoding='utf-8'
            )

            # Detect schema
            schema_result = _execute_async_operation(
                imex_service.detect_schema_automatically, source_config
            )

            flash("Schema detection completed successfully!", "success")

        except Exception as e:
            logger.error(f"Schema detection error: {e}")
            flash(f"Schema detection failed: {str(e)}", "error")

    return render_template('imex/schema_detect.html',
                         form=form,
                         schema_result=schema_result,
                         title="AI Schema Detection")

@imex_views_bp.route('/api/jobs/<job_id>/metrics')
def job_metrics_api(job_id):
    """API endpoint for real-time job metrics"""
    try:
        if not imex_service:
            return jsonify({'error': 'Service not available'}), 503

        metrics = _execute_async_operation(imex_service.get_job_metrics, job_id)
        return jsonify(metrics.dict() if hasattr(metrics, 'dict') else metrics)

    except Exception as e:
        logger.error(f"Metrics API error: {e}")
        return jsonify({'error': str(e)}), 500

# Helper Functions

def _get_recent_jobs(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent jobs list"""
    try:
        if not imex_service or not hasattr(imex_service, 'active_jobs'):
            return []

        jobs = []
        for job_id, job in list(imex_service.active_jobs.items())[:limit]:
            jobs.append({
                'id': job.id,
                'name': job.name,
                'job_type': str(job.job_type),
                'status': str(job.status),
                'created_at': job.created_at.strftime('%Y-%m-%d %H:%M') if job.created_at else 'Unknown',
                'priority': str(job.priority)
            })
        return jobs
    except Exception as e:
        logger.error(f"Recent jobs error: {e}")
        return []

# Error Handlers

@imex_views_bp.errorhandler(404)
def not_found_error(error):
    return render_template('imex/error.html',
                         error="Page not found",
                         title="404 Not Found"), 404

@imex_views_bp.errorhandler(500)
def internal_error(error):
    return render_template('imex/error.html',
                         error="Internal server error",
                         title="500 Internal Error"), 500

# View Registry for APG Integration

views_registry = {
    'blueprint': imex_views_bp,
    'set_service': set_imex_service,
    'forms': {
        'JobCreateForm': JobCreateForm,
        'SchemaDetectionForm': SchemaDetectionForm
    },
    'models': {
        'JobCreateRequest': JobCreateRequest,
        'SchemaDetectionRequest': SchemaDetectionRequest,
        'DataQualityRequest': DataQualityRequest
    }
}

__all__ = [
    'imex_views_bp',
    'set_imex_service',
    'JobCreateRequest',
    'SchemaDetectionRequest',
    'DataQualityRequest',
    'JobCreateForm',
    'SchemaDetectionForm',
    'views_registry'
]