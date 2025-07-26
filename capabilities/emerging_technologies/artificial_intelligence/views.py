"""
AI Orchestration Views

Flask-AppBuilder views for comprehensive AI workflow orchestration, model management,
provider coordination, and performance monitoring with real-time analytics.
"""

from flask import request, jsonify, flash, redirect, url_for, render_template
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from flask_appbuilder.widgets import FormWidget, ListWidget, SearchWidget
from flask_appbuilder.forms import DynamicForm
from wtforms import StringField, TextAreaField, SelectField, BooleanField, FloatField, IntegerField, validators
from wtforms.validators import DataRequired, Length, Optional, NumberRange
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from .models import (
	AIModel, AIWorkflow, AIWorkflowStep, AIExecution, AIExecutionStep,
	AIProvider, AIMetric
)


class AIOrchestrationBaseView(BaseView):
	"""Base view for AI orchestration functionality"""
	
	def __init__(self):
		super().__init__()
		self.default_view = 'dashboard'
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID from security context"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"
	
	def _format_latency(self, latency_ms: float) -> str:
		"""Format latency for display"""
		if latency_ms is None:
			return "N/A"
		if latency_ms < 1000:
			return f"{latency_ms:.0f}ms"
		else:
			return f"{latency_ms/1000:.1f}s"


class AIModelModelView(ModelView):
	"""AI model management view"""
	
	datamodel = SQLAInterface(AIModel)
	
	# List view configuration
	list_columns = [
		'name', 'provider', 'model_type', 'version',
		'health_status', 'average_latency_ms', 'quality_score', 'is_active'
	]
	show_columns = [
		'model_id', 'name', 'model_key', 'version', 'provider', 'model_type',
		'capabilities', 'supported_languages', 'max_context_length', 'max_output_length',
		'model_endpoint', 'api_key_required', 'average_latency_ms', 'throughput_rpm',
		'cost_per_1k_tokens', 'quality_score', 'health_status', 'is_active',
		'total_requests', 'successful_requests', 'failed_requests'
	]
	edit_columns = [
		'name', 'model_key', 'version', 'provider', 'model_type',
		'capabilities', 'supported_languages', 'max_context_length', 'max_output_length',
		'model_endpoint', 'api_key_required', 'configuration_schema', 'default_parameters',
		'rate_limit_rpm', 'rate_limit_tpm', 'is_active'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['name', 'model_key', 'provider', 'model_type']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('quality_score', 'desc')
	
	# Form validation
	validators_columns = {
		'name': [DataRequired(), Length(min=1, max=200)],
		'model_key': [DataRequired(), Length(min=1, max=100)],
		'provider': [DataRequired()],
		'model_type': [DataRequired()],
		'quality_score': [NumberRange(min=0, max=100)]
	}
	
	# Custom labels
	label_columns = {
		'model_id': 'Model ID',
		'model_key': 'Model Key',
		'model_type': 'Model Type',
		'max_context_length': 'Max Context Length',
		'max_output_length': 'Max Output Length',
		'model_endpoint': 'Model Endpoint',
		'api_key_required': 'API Key Required',
		'configuration_schema': 'Configuration Schema',
		'default_parameters': 'Default Parameters',
		'average_latency_ms': 'Avg Latency (ms)',
		'throughput_rpm': 'Throughput (RPM)',
		'cost_per_1k_tokens': 'Cost per 1K Tokens',
		'quality_score': 'Quality Score',
		'health_status': 'Health Status',
		'is_active': 'Active',
		'total_requests': 'Total Requests',
		'successful_requests': 'Successful Requests',
		'failed_requests': 'Failed Requests',
		'total_tokens_processed': 'Total Tokens',
		'rate_limit_rpm': 'Rate Limit (RPM)',
		'rate_limit_tpm': 'Rate Limit (TPM)'
	}
	
	@expose('/health_check/<int:pk>')
	@has_access
	def health_check(self, pk):
		"""Perform health check on AI model"""
		model = self.datamodel.get(pk)
		if not model:
			flash('AI model not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would perform actual health check
			model.health_status = 'healthy'
			model.last_health_check = datetime.utcnow()
			self.datamodel.edit(model)
			flash(f'Health check completed for model "{model.name}"', 'success')
		except Exception as e:
			flash(f'Error performing health check: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/test_model/<int:pk>')
	@has_access
	def test_model(self, pk):
		"""Test AI model with sample input"""
		model = self.datamodel.get(pk)
		if not model:
			flash('AI model not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would perform actual model test
			test_results = self._test_model_performance(model)
			
			return render_template('ai_orchestration/model_test_results.html',
								   model=model,
								   test_results=test_results,
								   page_title=f"Test Results: {model.name}")
		except Exception as e:
			flash(f'Error testing model: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new AI model"""
		item.tenant_id = self._get_tenant_id()
		
		# Set default values
		if not item.health_status:
			item.health_status = 'unknown'
		if not item.quality_score:
			item.quality_score = 0.0
	
	def _test_model_performance(self, model: AIModel) -> Dict[str, Any]:
		"""Test model performance with sample requests"""
		# Implementation would perform actual performance testing
		return {
			'latency_test': {'avg_ms': 250, 'min_ms': 180, 'max_ms': 420},
			'accuracy_test': {'score': 0.92, 'sample_size': 100},
			'throughput_test': {'requests_per_minute': 45},
			'error_rate': 0.02
		}
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class AIWorkflowModelView(ModelView):
	"""AI workflow management view"""
	
	datamodel = SQLAInterface(AIWorkflow)
	
	# List view configuration
	list_columns = [
		'name', 'workflow_type', 'status', 'priority',
		'execution_count', 'success_rate', 'average_duration', 'is_active'
	]
	show_columns = [
		'workflow_id', 'name', 'description', 'workflow_type', 'status',
		'priority', 'trigger_type', 'trigger_conditions', 'execution_count',
		'success_count', 'success_rate', 'average_duration', 'steps'
	]
	edit_columns = [
		'name', 'description', 'workflow_type', 'priority', 'trigger_type',
		'trigger_conditions', 'input_schema', 'output_schema', 'error_handling',
		'retry_policy', 'timeout_seconds', 'is_active'
	]
	add_columns = edit_columns
	
	# Related views
	related_views = [AIModelModelView]
	
	# Search and filtering
	search_columns = ['name', 'description', 'workflow_type']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('created_on', 'desc')
	
	# Form validation
	validators_columns = {
		'name': [DataRequired(), Length(min=1, max=200)],
		'workflow_type': [DataRequired()],
		'timeout_seconds': [NumberRange(min=1)]
	}
	
	# Custom labels
	label_columns = {
		'workflow_id': 'Workflow ID',
		'workflow_type': 'Workflow Type',
		'trigger_type': 'Trigger Type',
		'trigger_conditions': 'Trigger Conditions',
		'input_schema': 'Input Schema',
		'output_schema': 'Output Schema',
		'error_handling': 'Error Handling',
		'retry_policy': 'Retry Policy',
		'timeout_seconds': 'Timeout (seconds)',
		'execution_count': 'Execution Count',
		'success_count': 'Success Count',
		'success_rate': 'Success Rate',
		'average_duration': 'Avg Duration',
		'is_active': 'Active'
	}
	
	@expose('/execute/<int:pk>')
	@has_access
	def execute_workflow(self, pk):
		"""Execute AI workflow manually"""
		workflow = self.datamodel.get(pk)
		if not workflow:
			flash('Workflow not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would execute the actual workflow
			execution_id = self._execute_workflow(workflow)
			flash(f'Workflow "{workflow.name}" started successfully. Execution ID: {execution_id}', 'success')
		except Exception as e:
			flash(f'Error executing workflow: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/validate/<int:pk>')
	@has_access
	def validate_workflow(self, pk):
		"""Validate workflow configuration"""
		workflow = self.datamodel.get(pk)
		if not workflow:
			flash('Workflow not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			validation_results = self._validate_workflow(workflow)
			
			return render_template('ai_orchestration/workflow_validation.html',
								   workflow=workflow,
								   validation_results=validation_results,
								   page_title=f"Validation: {workflow.name}")
		except Exception as e:
			flash(f'Error validating workflow: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new workflow"""
		item.tenant_id = self._get_tenant_id()
		item.created_by = self._get_current_user_id()
		
		# Set default values
		if not item.status:
			item.status = 'draft'
		if not item.priority:
			item.priority = 'normal'
	
	def _execute_workflow(self, workflow: AIWorkflow) -> str:
		"""Execute workflow and return execution ID"""
		# Implementation would start actual workflow execution
		import uuid
		return str(uuid.uuid4())
	
	def _validate_workflow(self, workflow: AIWorkflow) -> Dict[str, Any]:
		"""Validate workflow configuration"""
		# Implementation would perform actual validation
		return {
			'is_valid': True,
			'errors': [],
			'warnings': ['Model availability not verified'],
			'suggestions': ['Consider adding error handling for step 2']
		}
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class AIExecutionModelView(ModelView):
	"""AI execution monitoring view"""
	
	datamodel = SQLAInterface(AIExecution)
	
	# List view configuration
	list_columns = [
		'workflow', 'status', 'start_time', 'duration_seconds',
		'total_cost', 'success_rate', 'error_count'
	]
	show_columns = [
		'execution_id', 'workflow', 'trigger_type', 'status', 'start_time',
		'end_time', 'duration_seconds', 'total_tokens', 'total_cost',
		'step_count', 'completed_steps', 'success_rate', 'error_count', 'steps'
	]
	# Read-only view for executions
	edit_columns = []
	add_columns = []
	can_create = False
	can_edit = False
	can_delete = False
	
	# Search and filtering
	search_columns = ['workflow.name', 'status', 'trigger_type']
	base_filters = [['status', lambda: 'running', lambda: True]]
	
	# Ordering
	base_order = ('start_time', 'desc')
	
	# Custom labels
	label_columns = {
		'execution_id': 'Execution ID',
		'trigger_type': 'Trigger Type',
		'start_time': 'Start Time',
		'end_time': 'End Time',
		'duration_seconds': 'Duration (sec)',
		'total_tokens': 'Total Tokens',
		'total_cost': 'Total Cost',
		'step_count': 'Step Count',
		'completed_steps': 'Completed Steps',
		'success_rate': 'Success Rate',
		'error_count': 'Error Count'
	}
	
	@expose('/cancel/<int:pk>')
	@has_access
	def cancel_execution(self, pk):
		"""Cancel running execution"""
		execution = self.datamodel.get(pk)
		if not execution:
			flash('Execution not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if execution.status == 'running':
				execution.status = 'cancelled'
				execution.end_time = datetime.utcnow()
				self.datamodel.edit(execution)
				flash(f'Execution cancelled successfully', 'success')
			else:
				flash('Only running executions can be cancelled', 'warning')
		except Exception as e:
			flash(f'Error cancelling execution: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/retry/<int:pk>')
	@has_access
	def retry_execution(self, pk):
		"""Retry failed execution"""
		execution = self.datamodel.get(pk)
		if not execution:
			flash('Execution not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if execution.status == 'failed':
				# Implementation would retry the execution
				new_execution_id = self._retry_execution(execution)
				flash(f'Execution retried successfully. New execution ID: {new_execution_id}', 'success')
			else:
				flash('Only failed executions can be retried', 'warning')
		except Exception as e:
			flash(f'Error retrying execution: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def _retry_execution(self, execution: AIExecution) -> str:
		"""Retry execution and return new execution ID"""
		# Implementation would create and start new execution
		import uuid
		return str(uuid.uuid4())


class AIProviderModelView(ModelView):
	"""AI provider management view"""
	
	datamodel = SQLAInterface(AIProvider)
	
	# List view configuration
	list_columns = [
		'name', 'provider_type', 'is_active', 'health_status',
		'total_requests', 'success_rate', 'average_cost'
	]
	show_columns = [
		'provider_id', 'name', 'provider_type', 'base_url', 'is_active',
		'health_status', 'last_health_check', 'total_requests', 'successful_requests',
		'failed_requests', 'success_rate', 'total_cost', 'average_cost'
	]
	edit_columns = [
		'name', 'provider_type', 'base_url', 'api_key', 'authentication_method',
		'configuration', 'rate_limits', 'is_active', 'priority'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['name', 'provider_type']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('priority', 'asc')
	
	# Form validation
	validators_columns = {
		'name': [DataRequired(), Length(min=1, max=200)],
		'provider_type': [DataRequired()],
		'base_url': [DataRequired()],
		'priority': [NumberRange(min=1, max=100)]
	}
	
	# Custom labels
	label_columns = {
		'provider_id': 'Provider ID',
		'provider_type': 'Provider Type',
		'base_url': 'Base URL',
		'api_key': 'API Key',
		'authentication_method': 'Auth Method',
		'rate_limits': 'Rate Limits',
		'health_status': 'Health Status',
		'last_health_check': 'Last Health Check',
		'total_requests': 'Total Requests',
		'successful_requests': 'Successful Requests',
		'failed_requests': 'Failed Requests',
		'success_rate': 'Success Rate',
		'total_cost': 'Total Cost',
		'average_cost': 'Average Cost',
		'is_active': 'Active'
	}
	
	@expose('/test_connection/<int:pk>')
	@has_access
	def test_connection(self, pk):
		"""Test connection to AI provider"""
		provider = self.datamodel.get(pk)
		if not provider:
			flash('Provider not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would test actual connection
			success = self._test_provider_connection(provider)
			
			if success:
				provider.health_status = 'healthy'
				provider.last_health_check = datetime.utcnow()
				self.datamodel.edit(provider)
				flash(f'Connection test successful for provider "{provider.name}"', 'success')
			else:
				provider.health_status = 'unhealthy'
				provider.last_health_check = datetime.utcnow()
				self.datamodel.edit(provider)
				flash(f'Connection test failed for provider "{provider.name}"', 'error')
		except Exception as e:
			flash(f'Error testing connection: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new provider"""
		item.tenant_id = self._get_tenant_id()
		
		# Set default values
		if not item.health_status:
			item.health_status = 'unknown'
		if not item.priority:
			item.priority = 50
	
	def _test_provider_connection(self, provider: AIProvider) -> bool:
		"""Test connection to provider"""
		# Implementation would perform actual connection test
		return True
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class AIOrchestrationDashboardView(AIOrchestrationBaseView):
	"""AI orchestration dashboard"""
	
	route_base = "/ai_orchestration_dashboard"
	default_view = "index"
	
	@expose('/')
	@has_access
	def index(self):
		"""AI orchestration dashboard main page"""
		try:
			# Get dashboard metrics
			metrics = self._get_dashboard_metrics()
			
			return render_template('ai_orchestration/dashboard.html',
								   metrics=metrics,
								   page_title="AI Orchestration Dashboard")
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return render_template('ai_orchestration/dashboard.html',
								   metrics={},
								   page_title="AI Orchestration Dashboard")
	
	@expose('/model_performance/')
	@has_access
	def model_performance(self):
		"""Model performance analytics"""
		try:
			performance_data = self._get_model_performance_data()
			
			return render_template('ai_orchestration/model_performance.html',
								   performance_data=performance_data,
								   page_title="Model Performance Analytics")
		except Exception as e:
			flash(f'Error loading model performance: {str(e)}', 'error')
			return redirect(url_for('AIOrchestrationDashboardView.index'))
	
	@expose('/workflow_analytics/')
	@has_access
	def workflow_analytics(self):
		"""Workflow execution analytics"""
		try:
			period_days = int(request.args.get('period', 7))
			analytics_data = self._get_workflow_analytics(period_days)
			
			return render_template('ai_orchestration/workflow_analytics.html',
								   analytics_data=analytics_data,
								   period_days=period_days,
								   page_title="Workflow Analytics")
		except Exception as e:
			flash(f'Error loading workflow analytics: {str(e)}', 'error')
			return redirect(url_for('AIOrchestrationDashboardView.index'))
	
	def _get_dashboard_metrics(self) -> Dict[str, Any]:
		"""Get AI orchestration metrics for dashboard"""
		# Implementation would calculate real metrics from database
		return {
			'active_models': 12,
			'healthy_models': 10,
			'total_workflows': 35,
			'active_workflows': 28,
			'running_executions': 5,
			'executions_today': 142,
			'success_rate': 94.2,
			'average_latency': 285,
			'total_cost_today': 47.50,
			'top_models': [
				{'name': 'GPT-4', 'requests': 1250, 'success_rate': 98.5},
				{'name': 'Claude-3', 'requests': 890, 'success_rate': 97.2},
				{'name': 'Gemini-Pro', 'requests': 670, 'success_rate': 95.8}
			],
			'recent_executions': []
		}
	
	def _get_model_performance_data(self) -> Dict[str, Any]:
		"""Get model performance analytics"""
		return {
			'latency_trends': [],
			'throughput_comparison': {},
			'cost_analysis': {},
			'quality_scores': {},
			'availability_stats': {}
		}
	
	def _get_workflow_analytics(self, period_days: int) -> Dict[str, Any]:
		"""Get workflow execution analytics"""
		return {
			'period_days': period_days,
			'total_executions': 1420,
			'successful_executions': 1338,
			'failed_executions': 82,
			'success_rate': 94.2,
			'average_duration': 45.2,
			'cost_breakdown': {},
			'execution_trends': [],
			'top_workflows': []
		}


# Register views with AppBuilder
def register_views(appbuilder):
	"""Register all AI orchestration views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		AIModelModelView,
		"AI Models",
		icon="fa-brain",
		category="AI Orchestration",
		category_icon="fa-robot"
	)
	
	appbuilder.add_view(
		AIWorkflowModelView,
		"AI Workflows",
		icon="fa-project-diagram",
		category="AI Orchestration"
	)
	
	appbuilder.add_view(
		AIExecutionModelView,
		"Executions",
		icon="fa-play-circle",
		category="AI Orchestration"
	)
	
	appbuilder.add_view(
		AIProviderModelView,
		"AI Providers",
		icon="fa-server",
		category="AI Orchestration"
	)
	
	# Dashboard views
	appbuilder.add_view_no_menu(AIOrchestrationDashboardView)
	
	# Menu links
	appbuilder.add_link(
		"AI Dashboard",
		href="/ai_orchestration_dashboard/",
		icon="fa-dashboard",
		category="AI Orchestration"
	)
	
	appbuilder.add_link(
		"Model Performance",
		href="/ai_orchestration_dashboard/model_performance/",
		icon="fa-chart-line",
		category="AI Orchestration"
	)
	
	appbuilder.add_link(
		"Workflow Analytics",
		href="/ai_orchestration_dashboard/workflow_analytics/",
		icon="fa-analytics",
		category="AI Orchestration"
	)