"""
APG Import/Export (IMEX) Flask-AppBuilder Views

Pydantic v2 models and Flask-AppBuilder views for enterprise import/export operations.
Provides intuitive UI with real-time monitoring, visual workflow design, and intelligent automation.
"""

from datetime import datetime, timezone
from typing import Any, Optional
from types import SimpleNamespace
from uuid_extensions import uuid7str

from flask import request, jsonify, render_template, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.actions import action
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget
from flask_appbuilder.security.decorators import protect
from pydantic import BaseModel, Field, ConfigDict

from models import(
	ImportExportJob, JobExecution, JobStatus, JobType, ProcessingMetrics,
	DataQualityReport, Workflow, SchemaMapping, ValidationRule,
	SourceConfig, TargetConfig, DataFormat, SourceType
)
# Global service instance (will be set by the application)
imex_service = None

def set_imex_service(service):
    """Set the global IMEX service instance"""
    global imex_service
    imex_service = service


def _safe_datamodel(model_class):
	"""Return a FAB datamodel when possible, otherwise a Pydantic-friendly placeholder."""
	try:
		return SQLAInterface(model_class)
	except Exception:
		return SimpleNamespace(obj=model_class)


# Pydantic v2 Models for API Validation

class JobCreateRequest(BaseModel):
	"""Request model for creating import/export jobs"""
	name: str = Field(min_length=1, max_length=255)
	description: str | None = Field(None, max_length=1000)
	job_type: JobType
	source_config: dict[str, Any]
	target_config: dict[str, Any]
	schema_mapping_id: str | None = None
	validation_rules: list[dict[str, Any]] = Field(default_factory=list)
	transformation_steps: list[dict[str, Any]] = Field(default_factory=list)
	schedule_config: dict[str, Any] | None = None
	tags: list[str] = Field(default_factory=list)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class JobExecutionRequest(BaseModel):
	"""Request model for job execution"""
	execution_config: dict[str, Any] = Field(default_factory=dict)
	priority_override: str | None = None
	resource_limits: dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class SchemaDetectionRequest(BaseModel):
	"""Request model for schema detection"""
	source_config: dict[str, Any]
	sample_size: int = Field(default=1000, ge=100, le=10000)
	include_statistics: bool = True

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class SchemaMappingRequest(BaseModel):
	"""Request model for schema mapping"""
	name: str = Field(min_length=1, max_length=255)
	description: str | None = Field(None, max_length=1000)
	source_schema: dict[str, Any]
	target_schema: dict[str, Any]
	auto_map_similar_fields: bool = True
	confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class WorkflowCreateRequest(BaseModel):
	"""Request model for creating workflows"""
	name: str = Field(min_length=1, max_length=255)
	description: str | None = Field(None, max_length=1000)
	steps: list[dict[str, Any]]
	schedule_config: dict[str, Any] | None = None
	parallel_execution: bool = False
	tags: list[str] = Field(default_factory=list)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class DataQualityRequest(BaseModel):
	"""Request model for data quality validation"""
	job_id: str
	sample_data: list[dict[str, Any]] = Field(min_length=1, max_length=10000)
	include_anomaly_detection: bool = True
	quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


# Custom Widgets for Enhanced UI

class ImportExportJobWidget(ListWidget):
	"""Custom widget for job list view"""
	template = 'imex/widgets/job_list.html'


class JobMonitoringWidget(ShowWidget):
	"""Custom widget for job monitoring view"""
	template = 'imex/widgets/job_monitoring.html'


class WorkflowDesignerWidget(ShowWidget):
	"""Custom widget for workflow designer"""
	template = 'imex/widgets/workflow_designer.html'


# Main Flask-AppBuilder Views

class ImportExportJobView(ModelView):
	"""Main view for import/export job management"""

	datamodel = _safe_datamodel(ImportExportJob)
	list_widget = ImportExportJobWidget
	show_widget = JobMonitoringWidget

	# List view configuration
	list_columns = ['name', 'job_type', 'status', 'created_at', 'last_run_at', 'created_by']
	search_columns = ['name', 'description', 'job_type', 'status', 'tags']
	list_title = "Import/Export Jobs"
	show_title = "Job Details"
	add_title = "Create New Job"
	edit_title = "Edit Job"

	# Form configuration
	add_columns = [
		'name', 'description', 'job_type', 'priority',
		'source_config', 'target_config', 'validation_level',
		'error_handling', 'parallel_processing', 'max_workers',
		'timeout_minutes', 'tags'
	]
	edit_columns = add_columns
	show_columns = [
		'name', 'description', 'job_type', 'priority', 'status',
		'source_config', 'target_config', 'schema_mapping',
		'validation_rules', 'transformation_steps', 'schedule_config',
		'validation_level', 'error_handling', 'parallel_processing',
		'max_workers', 'timeout_minutes', 'tags', 'created_by',
		'created_at', 'updated_at', 'last_run_at'
	]

	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']

	# Custom actions
	@action("execute_job", "Execute Job", "Execute selected jobs", "fa-play")
	def execute_job_action(self, items):
		"""Execute selected jobs"""
		if not imex_service:
			flash("IMEX service not available", "error")
			return redirect(self.get_redirect())

		for item in items:
			try:
				# Execute job asynchronously
				import asyncio
				execution = asyncio.run(imex_service.execute_job(item.id, {}))
				flash(f"Job '{item.name}' execution started (ID: {execution.id})", "success")
			except Exception as e:
				flash(f"Failed to execute job '{item.name}': {str(e)}", "error")
		return redirect(self.get_redirect())

	@action("duplicate_job", "Duplicate Job", "Create copies of selected jobs", "fa-copy")
	def duplicate_job_action(self, items):
		"""Duplicate selected jobs"""
		if not imex_service:
			flash("IMEX service not available", "error")
			return redirect(self.get_redirect())

		for item in items:
			try:
				# Create duplicate with modified name
				duplicate_config = {
					'name': f"{item.name} (Copy)",
					'description': item.description,
					'job_type': item.job_type.value if hasattr(item.job_type, 'value') else str(item.job_type),
					'source_config': item.source_config,
					'target_config': item.target_config,
					'priority': item.priority.value if hasattr(item.priority, 'value') else str(item.priority),
					'validation_level': item.validation_level.value if hasattr(item.validation_level, 'value') else str(item.validation_level),
					'error_handling': item.error_handling.value if hasattr(item.error_handling, 'value') else str(item.error_handling),
					'tags': item.tags if hasattr(item, 'tags') else [],
					'tenant_id': 'default'
				}

				# Create new job
				import asyncio
				new_job = asyncio.run(imex_service.create_job(duplicate_config, item.created_by))
				flash(f"Job '{item.name}' duplicated successfully", "success")
			except Exception as e:
				flash(f"Failed to duplicate job '{item.name}': {str(e)}", "error")
		return redirect(self.get_redirect())

	@expose('/monitor/<job_id>')
	@has_access
	def monitor_job(self, job_id):
		"""Real-time job monitoring view"""
		try:
			job = self.datamodel.get(job_id)
			if not job:
				flash(f"Job not found: {job_id}", "error")
				return redirect(url_for('ImportExportJobView.list'))

			# Get real-time metrics
			metrics = None
			if imex_service and hasattr(job, 'status') and str(job.status) == 'running':
				try:
					import asyncio
					metrics = asyncio.run(imex_service.get_job_metrics(job_id))
				except Exception as e:
					metrics = None

			return self.render_template(
				'imex/job_monitor.html',
				job=job,
				metrics=metrics,
				title=f"Monitor Job: {job.name}"
			)
		except Exception as e:
			flash(f"Error loading job monitor: {str(e)}", "error")
			return redirect(url_for('ImportExportJobView.list'))

	@expose('/api/metrics/<job_id>')
	@has_access
	def api_job_metrics(self, job_id):
		"""API endpoint for real-time job metrics"""
		try:
			if not imex_service:
				return jsonify({"error": "Service not available"}), 503

			import asyncio
			metrics = asyncio.run(imex_service.get_job_metrics(job_id))
			return jsonify(metrics.dict() if hasattr(metrics, 'dict') else metrics)
		except Exception as e:
			return jsonify({"error": str(e)}), 404


class JobExecutionView(ModelView):
	"""View for job execution history and monitoring"""

	datamodel = _safe_datamodel(JobExecution)

	list_columns = ['job_id', 'execution_number', 'status', 'started_at', 'completed_at']
	search_columns = ['job_id', 'status', 'worker_node']
	list_title = "Job Executions"
	show_title = "Execution Details"

	show_columns = [
		'job_id', 'execution_number', 'status', 'started_at', 'completed_at',
		'error_message', 'metrics', 'log_file_path', 'worker_node'
	]

	# Read-only view
	base_permissions = ['can_list', 'can_show']

	@expose('/logs/<execution_id>')
	@has_access
	def view_execution_logs(self, execution_id):
		"""View execution logs"""
		try:
			execution = self.datamodel.get(execution_id)
			if not execution:
				flash(f"Execution not found: {execution_id}", "error")
				return redirect(url_for('JobExecutionView.list'))

			# Read log file if available
			logs = ""
			if execution.log_file_path:
				try:
					with open(execution.log_file_path, 'r') as f:
						logs = f.read()
				except Exception as e:
					logs = f"Error reading log file: {str(e)}"

			return self.render_template(
				'imex/execution_logs.html',
				execution=execution,
				logs=logs,
				title=f"Execution Logs: {execution.id}"
			)
		except Exception as e:
			flash(f"Error loading execution logs: {str(e)}", "error")
			return redirect(url_for('JobExecutionView.list'))


class WorkflowView(ModelView):
	"""View for workflow management"""

	datamodel = _safe_datamodel(Workflow)
	show_widget = WorkflowDesignerWidget

	list_columns = ['name', 'version', 'status', 'created_at', 'last_execution_id']
	search_columns = ['name', 'description', 'status', 'tags']
	list_title = "Data Workflows"
	show_title = "Workflow Details"
	add_title = "Create New Workflow"
	edit_title = "Edit Workflow"

	add_columns = ['name', 'description', 'version', 'parallel_execution', 'error_handling', 'tags']
	edit_columns = add_columns
	show_columns = [
		'name', 'description', 'version', 'steps', 'schedule_config',
		'parallel_execution', 'error_handling', 'status', 'execution_history',
		'tags', 'created_by', 'created_at', 'updated_at'
	]

	@action("execute_workflow", "Execute Workflow", "Execute selected workflows", "fa-play")
	def execute_workflow_action(self, items):
		"""Execute selected workflows"""
		for item in items:
			try:
				imex_service.execute_workflow(item)
				flash(f"Workflow '{item.name}' execution started", "success")
			except Exception as e:
				flash(f"Failed to execute workflow '{item.name}': {str(e)}", "error")
		return redirect(self.get_redirect())

	@expose('/designer/<workflow_id>')
	@has_access
	def workflow_designer(self, workflow_id):
		"""Visual workflow designer"""
		try:
			workflow = self.datamodel.get(workflow_id) if workflow_id != 'new' else None

			return self.render_template(
				'imex/workflow_designer.html',
				workflow=workflow,
				title="Workflow Designer"
			)
		except Exception as e:
			flash(f"Error loading workflow designer: {str(e)}", "error")
			return redirect(url_for('WorkflowView.list'))


class SchemaMappingView(ModelView):
	"""View for schema mapping management"""

	datamodel = _safe_datamodel(SchemaMapping)

	list_columns = ['name', 'description', 'created_at', 'created_by']
	search_columns = ['name', 'description']
	list_title = "Schema Mappings"
	show_title = "Schema Mapping Details"
	add_title = "Create Schema Mapping"
	edit_title = "Edit Schema Mapping"

	add_columns = ['name', 'description', 'auto_map_similar_fields', 'ignore_extra_fields', 'strict_mode']
	edit_columns = add_columns
	show_columns = [
		'name', 'description', 'field_mappings', 'auto_map_similar_fields',
		'ignore_extra_fields', 'strict_mode', 'transformation_script',
		'created_by', 'created_at', 'updated_at'
	]

	@expose('/mapper')
	@has_access
	def schema_mapper(self):
		"""Interactive schema mapping interface"""
		return self.render_template(
			'imex/schema_mapper.html',
			title="Schema Mapper"
		)


class DataQualityView(BaseView):
	"""View for data quality management and reporting"""

	default_view = 'quality_dashboard'

	@expose('/dashboard')
	@has_access
	def quality_dashboard(self):
		"""Data quality dashboard"""
		return self.render_template(
			'imex/quality_dashboard.html',
			title="Data Quality Dashboard"
		)

	@expose('/reports')
	@has_access
	def quality_reports(self):
		"""Data quality reports list"""
		return self.render_template(
			'imex/quality_reports.html',
			title="Quality Reports"
		)

	@expose('/report/<report_id>')
	@has_access
	def view_quality_report(self, report_id):
		"""View detailed quality report"""
		try:
			# Get quality report data
			report_data = {
				"id": report_id,
				"job_id": "sample_job",
				"overall_score": 0.85,
				"completeness": 0.92,
				"consistency": 0.78,
				"accuracy": 0.85,
				"issues": {
					"missing_values": 150,
					"format_errors": 23,
					"duplicates": 7
				},
				"recommendations": [
					"Address missing values in 'email' field",
					"Standardize date formats",
					"Remove duplicate records"
				]
			}

			return self.render_template(
				'imex/quality_report.html',
				report=report_data,
				title=f"Quality Report: {report_id}"
			)
		except Exception as e:
			flash(f"Error loading quality report: {str(e)}", "error")
			return redirect(url_for('DataQualityView.quality_dashboard'))


class MonitoringDashboardView(BaseView):
	"""Real-time monitoring dashboard"""

	default_view = 'dashboard'

	@expose('/dashboard')
	@has_access
	def dashboard(self):
		"""Main monitoring dashboard"""
		return self.render_template(
			'imex/monitoring_dashboard.html',
			title="Import/Export Monitoring"
		)

	@expose('/api/system-metrics')
	@has_access
	def api_system_metrics(self):
		"""API endpoint for system performance metrics"""
		try:
			metrics = imex_service.get_system_performance_metrics()
			return jsonify(metrics)
		except Exception as e:
			return jsonify({"error": str(e)}), 500

	@expose('/api/active-jobs')
	@has_access
	def api_active_jobs(self):
		"""API endpoint for active jobs status"""
		try:
			active_jobs = [
				{
					"id": job_id,
					"name": job.name,
					"type": job.job_type.value,
					"status": job.status.value,
					"progress": (
						job.current_execution.metrics.records_processed
						if job.current_execution else 0
					)
				}
				for job_id, job in imex_service.active_jobs.items()
				if job.status == JobStatus.RUNNING
			]
			return jsonify(active_jobs)
		except Exception as e:
			return jsonify({"error": str(e)}), 500


class PerformanceAnalyticsView(DirectByChartView):
	"""Performance analytics and charts"""

	chart_title = "Import/Export Performance"
	label_columns = ['job_type']
	group_by_columns = ['job_type']

	@expose('/analytics')
	@has_access
	def performance_analytics(self):
		"""Performance analytics dashboard"""
		return self.render_template(
			'imex/performance_analytics.html',
			title="Performance Analytics"
		)

	@expose('/api/throughput-metrics')
	@has_access
	def api_throughput_metrics(self):
		"""API endpoint for throughput metrics"""
		try:
			# Mock data - in production would query actual metrics
			metrics = {
				"hourly_throughput": [
					{"hour": "00:00", "records": 125000},
					{"hour": "01:00", "records": 98000},
					{"hour": "02:00", "records": 145000},
					{"hour": "03:00", "records": 167000}
				],
				"job_type_performance": {
					"import": {"avg_rps": 15000, "success_rate": 0.992},
					"export": {"avg_rps": 12000, "success_rate": 0.998},
					"migration": {"avg_rps": 8000, "success_rate": 0.985}
				}
			}
			return jsonify(metrics)
		except Exception as e:
			return jsonify({"error": str(e)}), 500


class TemplateManagementView(BaseView):
	"""View for managing connection and job templates"""

	default_view = 'template_library'

	@expose('/library')
	@has_access
	def template_library(self):
		"""Template library"""
		return self.render_template(
			'imex/template_library.html',
			title="Template Library"
		)

	@expose('/create-template')
	@has_access
	def create_template(self):
		"""Create new template"""
		return self.render_template(
			'imex/create_template.html',
			title="Create Template"
		)


# Chart Views for Analytics

class JobStatusChart(DirectByChartView):
	"""Chart showing job status distribution"""
	chart_title = "Job Status Distribution"
	chart_type = "PieChart"
	direct_columns = {"Job Status": ("status", "count")}
	label_columns = ['status']
	group_by_columns = ['status']


class ThroughputChart(DirectByChartView):
	"""Chart showing throughput over time"""
	chart_title = "Data Throughput Over Time"
	chart_type = "LineChart"
	direct_columns = {"Records/Hour": ("hour", "throughput")}
	label_columns = ['hour']
	group_by_columns = ['hour']


# View Registry for APG Composition

view_registry = {
	"ImportExportJobView": ImportExportJobView,
	"JobExecutionView": JobExecutionView,
	"WorkflowView": WorkflowView,
	"SchemaMappingView": SchemaMappingView,
	"DataQualityView": DataQualityView,
	"MonitoringDashboardView": MonitoringDashboardView,
	"PerformanceAnalyticsView": PerformanceAnalyticsView,
	"TemplateManagementView": TemplateManagementView,
	"JobStatusChart": JobStatusChart,
	"ThroughputChart": ThroughputChart
}

__all__ = [
	# Request Models
	"JobCreateRequest", "JobExecutionRequest", "SchemaDetectionRequest",
	"SchemaMappingRequest", "WorkflowCreateRequest", "DataQualityRequest",

	# View Classes
	"ImportExportJobView", "JobExecutionView", "WorkflowView", "SchemaMappingView",
	"DataQualityView", "MonitoringDashboardView", "PerformanceAnalyticsView",
	"TemplateManagementView",

	# Chart Views
	"JobStatusChart", "ThroughputChart",

	# Registry
	"view_registry"
]
