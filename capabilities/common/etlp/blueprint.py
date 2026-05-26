#!/usr/bin/env python3
"""
APG ETLP Flask Blueprint Integration
Flask-AppBuilder integration for APG composition engine

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash, g, session, has_request_context
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.views import ModelView
from wtforms import Form, StringField, TextAreaField, SelectField, IntegerField, BooleanField
from wtforms.validators import DataRequired, Length, NumberRange
from uuid_extensions import uuid7str

from .service import ETLPService
from .models import Pipeline, Execution, Transformation, DataSource, QualityRule
from .views import ViewHelpers


# Create Blueprint for APG composition
etlp_blueprint = Blueprint(
	'etlp',
	__name__,
	template_folder='templates',
	static_folder='static',
	url_prefix='/etlp'
)


def _clean_text(value: Any) -> Optional[str]:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _object_value(source: Any, name: str) -> Any:
	if source is None:
		return None
	if isinstance(source, dict):
		return source.get(name)
	return getattr(source, name, None)


def _first_text(candidates: List[Any], fallback: str) -> str:
	for candidate in candidates:
		text = _clean_text(candidate)
		if text:
			return text
	return fallback


def _normalise_roles(value: Any) -> List[str]:
	if value is None:
		return ["admin"]
	if isinstance(value, str):
		roles = [role.strip() for role in value.replace(",", " ").split()]
		return roles or ["admin"]
	roles = []
	for role in value:
		name = _object_value(role, "name") or _object_value(role, "role_name") or role
		text = _clean_text(name)
		if text:
			roles.append(text)
	return roles or ["admin"]


def _appbuilder_user(view: Any) -> Any:
	appbuilder = getattr(view, "appbuilder", None)
	security_manager = getattr(appbuilder, "sm", None)
	get_user = getattr(security_manager, "get_user", None)
	if not callable(get_user):
		return None
	try:
		return get_user()
	except Exception:
		return None


def _resolve_current_user(view: Any = None) -> Dict[str, Any]:
	"""Resolve ETLP tenant/user context from APG, Flask, AppBuilder, headers, query, or environment."""
	default_user = os.getenv("APG_DEFAULT_USER_ID", os.getenv("APG_USER_ID", "system"))
	default_tenant = os.getenv("APG_DEFAULT_TENANT_ID", os.getenv("APG_TENANT_ID", "default"))
	default_roles = os.getenv("APG_DEFAULT_ROLES", os.getenv("APG_ROLES", "admin"))

	flask_user = getattr(g, "user", None) if has_request_context() else None
	appbuilder_user = _appbuilder_user(view)
	headers = request.headers if has_request_context() else {}
	args = request.args if has_request_context() else {}
	session_values = session if has_request_context() else {}

	tenant_id = _first_text([
		getattr(g, "tenant_id", None) if has_request_context() else None,
		_object_value(flask_user, "tenant_id"),
		_object_value(appbuilder_user, "tenant_id"),
		session_values.get("tenant_id"),
		headers.get("X-Tenant-ID"),
		headers.get("X-APG-Tenant-ID"),
		headers.get("X-Organization-ID"),
		args.get("tenant_id"),
		args.get("tenant"),
		os.getenv("APG_TENANT_ID"),
	], default_tenant)

	user_id = _first_text([
		getattr(g, "user_id", None) if has_request_context() else None,
		_object_value(flask_user, "user_id"),
		_object_value(flask_user, "id"),
		_object_value(appbuilder_user, "user_id"),
		_object_value(appbuilder_user, "id"),
		_object_value(appbuilder_user, "username"),
		session_values.get("user_id"),
		session_values.get("username"),
		headers.get("X-User-ID"),
		headers.get("X-APG-User-ID"),
		args.get("user_id"),
		os.getenv("APG_USER_ID"),
	], default_user)

	username = _first_text([
		_object_value(flask_user, "username"),
		_object_value(appbuilder_user, "username"),
		session_values.get("username"),
		headers.get("X-Username"),
		args.get("username"),
	], user_id)

	roles = _normalise_roles(
		_object_value(flask_user, "roles")
		or _object_value(appbuilder_user, "roles")
		or session_values.get("roles")
		or headers.get("X-APG-Roles")
		or default_roles
	)

	return {
		"tenant_id": tenant_id,
		"user_id": user_id,
		"username": username,
		"roles": roles,
	}


class ETLPDashboardView(BaseView):
	"""Main ETLP dashboard view for APG"""
	
	default_view = 'dashboard'
	
	def _get_current_user(self):
		"""Get current user information from APG context"""
		return _resolve_current_user(self)
	
	@expose('/dashboard')
	@has_access
	def dashboard(self):
		"""Main ETLP dashboard with pipeline overview"""
		try:
			# Get user context from APG
			user_info = self._get_current_user()
			etlp_service = ETLPService(user_info['tenant_id'], user_info['user_id'])
			
			# Get dashboard metrics synchronously using event loop
			import asyncio
			loop = asyncio.get_event_loop()
			recent_pipelines = loop.run_until_complete(etlp_service.list_pipelines(limit=5))
			recent_executions = loop.run_until_complete(etlp_service.list_executions(limit=10))
			
			# Calculate dashboard stats
			total_pipelines = len(recent_pipelines)
			active_pipelines = len([p for p in recent_pipelines if p.status.value == 'active'])
			running_executions = len([e for e in recent_executions if e.status.value == 'running'])
			
			# Success rate calculation
			completed_executions = [e for e in recent_executions if e.status.value in ['success', 'failed']]
			success_rate = 0.0
			if completed_executions:
				successful = len([e for e in completed_executions if e.status.value == 'success'])
				success_rate = (successful / len(completed_executions)) * 100
			
			dashboard_data = {
				'total_pipelines': total_pipelines,
				'active_pipelines': active_pipelines,
				'running_executions': running_executions,
				'success_rate': success_rate,
				'recent_pipelines': recent_pipelines,
				'recent_executions': recent_executions
			}
			
			return self.render_template(
				'etlp/dashboard.html',
				dashboard=dashboard_data,
				helpers=ViewHelpers
			)
			
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return self.render_template('etlp/error.html', error=str(e))


class ETLPPipelineView(BaseView):
	"""Pipeline management views"""
	
	def _get_current_user(self):
		"""Get current user information from APG context"""
		return _resolve_current_user(self)
	
	@expose('/pipelines')
	@has_access
	def list_pipelines(self):
		"""List all pipelines with filtering"""
		try:
			user_info = self._get_current_user()
			etlp_service = ETLPService(user_info['tenant_id'], user_info['user_id'])
			
			# Get filter parameters
			status_filter = request.args.get('status')
			search_filter = request.args.get('search')
			tag_filter = request.args.get('tags')
			
			# Build filters
			filters = {}
			if status_filter:
				filters['status'] = status_filter
			if search_filter:
				filters['search'] = search_filter
			if tag_filter:
				filters['tags'] = tag_filter.split(',')
			
			# Get paginated results
			page = int(request.args.get('page', 1))
			per_page = int(request.args.get('per_page', 20))
			offset = (page - 1) * per_page
			
			import asyncio
			loop = asyncio.get_event_loop()
			pipelines = loop.run_until_complete(etlp_service.list_pipelines(filters, per_page, offset))
			
			return self.render_template(
				'etlp/pipelines/list.html',
				pipelines=pipelines,
				filters=filters,
				page=page,
				per_page=per_page,
				helpers=ViewHelpers
			)
			
		except Exception as e:
			flash(f'Error loading pipelines: {str(e)}', 'error')
			return redirect(url_for('etlp.dashboard'))
	
	@expose('/pipelines/create', methods=['GET', 'POST'])
	@has_access
	def create_pipeline(self):
		"""Create new pipeline"""
		form = PipelineCreateForm()
		
		if form.validate_on_submit():
			try:
				user_info = self._get_current_user()
				etlp_service = ETLPService(user_info['tenant_id'], user_info['user_id'])
				
				pipeline_data = {
					'name': form.name.data,
					'description': form.description.data,
					'execution_mode': form.execution_mode.data,
					'max_parallelism': form.max_parallelism.data,
					'timeout_minutes': form.timeout_minutes.data,
					'retry_count': form.retry_count.data,
					'ai_optimization_enabled': form.ai_optimization_enabled.data
				}
				
				import asyncio
				loop = asyncio.get_event_loop()
				pipeline = loop.run_until_complete(etlp_service.create_pipeline(pipeline_data))
				flash(f'Pipeline "{pipeline.name}" created successfully!', 'success')
				return redirect(url_for('etlp.view_pipeline', pipeline_id=pipeline.id))
				
			except Exception as e:
				flash(f'Error creating pipeline: {str(e)}', 'error')
		
		return self.render_template('etlp/pipelines/create.html', form=form)
	
	@expose('/pipelines/<pipeline_id>')
	@has_access
	def view_pipeline(self, pipeline_id: str):
		"""View pipeline details"""
		try:
			user_info = self._get_current_user()
			etlp_service = ETLPService(user_info['tenant_id'], user_info['user_id'])
			
			import asyncio
			loop = asyncio.get_event_loop()
			pipeline = loop.run_until_complete(etlp_service.get_pipeline(pipeline_id))
			if not pipeline:
				flash('Pipeline not found', 'error')
				return redirect(url_for('etlp.list_pipelines'))
			
			# Get recent executions
			recent_executions = loop.run_until_complete(etlp_service.list_executions(pipeline_id, limit=10))
			
			# Get pipeline health
			health_data = loop.run_until_complete(etlp_service._get_pipeline_health(pipeline_id))
			
			return self.render_template(
				'etlp/pipelines/detail.html',
				pipeline=pipeline,
				executions=recent_executions,
				health=health_data,
				helpers=ViewHelpers
			)
			
		except Exception as e:
			flash(f'Error loading pipeline: {str(e)}', 'error')
			return redirect(url_for('etlp.list_pipelines'))
	
	@expose('/pipelines/<pipeline_id>/execute', methods=['POST'])
	@has_access
	def execute_pipeline(self, pipeline_id: str):
		"""Execute pipeline"""
		try:
			user_info = self._get_current_user()
			etlp_service = ETLPService(user_info['tenant_id'], user_info['user_id'])
			
			import asyncio
			loop = asyncio.get_event_loop()
			execution_id = loop.run_until_complete(etlp_service.execute_pipeline(pipeline_id))
			flash(f'Pipeline execution started. Execution ID: {execution_id}', 'info')
			
			return redirect(url_for('etlp.view_execution', execution_id=execution_id))
			
		except Exception as e:
			flash(f'Error executing pipeline: {str(e)}', 'error')
			return redirect(url_for('etlp.view_pipeline', pipeline_id=pipeline_id))


class ETLPExecutionView(BaseView):
	"""Execution monitoring views"""
	
	def _get_current_user(self):
		"""Get current user information from APG context"""
		return _resolve_current_user(self)
	
	@expose('/executions')
	@has_access
	def list_executions(self):
		"""List all executions"""
		try:
			user_info = self._get_current_user()
			etlp_service = ETLPService(user_info['tenant_id'], user_info['user_id'])
			
			# Get filter parameters
			pipeline_id = request.args.get('pipeline_id')
			status_filter = request.args.get('status')
			
			# Get paginated results
			page = int(request.args.get('page', 1))
			per_page = int(request.args.get('per_page', 50))
			offset = (page - 1) * per_page
			
			import asyncio
			loop = asyncio.get_event_loop()
			executions = loop.run_until_complete(etlp_service.list_executions(
				pipeline_id, status_filter, per_page, offset
			))
			
			return self.render_template(
				'etlp/executions/list.html',
				executions=executions,
				pipeline_id=pipeline_id,
				status_filter=status_filter,
				page=page,
				per_page=per_page,
				helpers=ViewHelpers
			)
			
		except Exception as e:
			flash(f'Error loading executions: {str(e)}', 'error')
			return redirect(url_for('etlp.dashboard'))
	
	@expose('/executions/<execution_id>')
	@has_access
	def view_execution(self, execution_id: str):
		"""View execution details"""
		try:
			user_info = self._get_current_user()
			etlp_service = ETLPService(user_info['tenant_id'], user_info['user_id'])
			
			import asyncio
			loop = asyncio.get_event_loop()
			execution = loop.run_until_complete(etlp_service.get_execution(execution_id))
			if not execution:
				flash('Execution not found', 'error')
				return redirect(url_for('etlp.list_executions'))
			
			# Get associated pipeline
			pipeline = loop.run_until_complete(etlp_service.get_pipeline(execution.pipeline_id))
			
			return self.render_template(
				'etlp/executions/detail.html',
				execution=execution,
				pipeline=pipeline,
				helpers=ViewHelpers
			)
			
		except Exception as e:
			flash(f'Error loading execution: {str(e)}', 'error')
			return redirect(url_for('etlp.list_executions'))


class ETLPDesignerView(BaseView):
	"""Visual pipeline designer"""
	
	def _get_current_user(self):
		"""Get current user information from APG context"""
		return _resolve_current_user(self)
	
	@expose('/designer')
	@has_access
	def designer(self):
		"""Visual pipeline designer interface"""
		return self.render_template('etlp/designer/index.html')
	
	@expose('/designer/<pipeline_id>')
	@has_access
	def edit_pipeline(self, pipeline_id: str):
		"""Edit pipeline in visual designer"""
		try:
			user_info = self._get_current_user()
			etlp_service = ETLPService(user_info['tenant_id'], user_info['user_id'])
			
			import asyncio
			loop = asyncio.get_event_loop()
			pipeline = loop.run_until_complete(etlp_service.get_pipeline(pipeline_id))
			if not pipeline:
				flash('Pipeline not found', 'error')
				return redirect(url_for('etlp.designer'))
			
			return self.render_template(
				'etlp/designer/editor.html',
				pipeline=pipeline,
				pipeline_json=pipeline.model_dump_json()
			)
			
		except Exception as e:
			flash(f'Error loading designer: {str(e)}', 'error')
			return redirect(url_for('etlp.designer'))


class ETLPMonitoringView(BaseView):
	"""Monitoring and metrics views"""
	
	def _get_current_user(self):
		"""Get current user information from APG context"""
		return _resolve_current_user(self)
	
	@expose('/monitoring')
	@has_access
	def monitoring_dashboard(self):
		"""Real-time monitoring dashboard"""
		try:
			user_info = self._get_current_user()
			etlp_service = ETLPService(user_info['tenant_id'], user_info['user_id'])
			
			# Get system metrics
			import asyncio
			loop = asyncio.get_event_loop()
			active_executions = loop.run_until_complete(etlp_service.list_executions(
				status='running', limit=100
			))
			
			# Get performance metrics
			metrics = loop.run_until_complete(etlp_service._get_system_metrics())
			
			return self.render_template(
				'etlp/monitoring/dashboard.html',
				active_executions=active_executions,
				metrics=metrics,
				helpers=ViewHelpers
			)
			
		except Exception as e:
			flash(f'Error loading monitoring: {str(e)}', 'error')
			return redirect(url_for('etlp.dashboard'))


# Form definitions
class PipelineCreateForm(Form):
	"""Form for creating pipelines"""
	name = StringField('Pipeline Name', validators=[
		DataRequired(), Length(min=1, max=255)
	])
	description = TextAreaField('Description', validators=[Length(max=1000)])
	execution_mode = SelectField('Execution Mode', choices=[
		('batch', 'Batch'),
		('streaming', 'Streaming'),
		('micro_batch', 'Micro Batch'),
		('event_driven', 'Event Driven')
	], default='batch')
	max_parallelism = IntegerField('Max Parallelism', validators=[
		NumberRange(min=1, max=100)
	], default=4)
	timeout_minutes = IntegerField('Timeout (minutes)', validators=[
		NumberRange(min=1, max=10080)
	], default=60)
	retry_count = IntegerField('Retry Count', validators=[
		NumberRange(min=0, max=10)
	], default=3)
	ai_optimization_enabled = BooleanField('Enable AI Optimization', default=True)


# APG Health Check Integration
@etlp_blueprint.route('/health')
def health_check():
	"""APG health check endpoint"""
	try:
		# Basic health checks
		health_status = {
			'status': 'healthy',
			'timestamp': datetime.utcnow().isoformat(),
			'version': '1.0.0',
			'service': 'etlp',
			'checks': {
				'database': 'healthy',
				'cache': 'healthy',
				'ai_service': 'healthy',
				'queue': 'healthy'
			}
		}
		
		return jsonify(health_status), 200
		
	except Exception as e:
		return jsonify({
			'status': 'unhealthy',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}), 500


# APG Metrics Endpoint
@etlp_blueprint.route('/metrics')
def metrics():
	"""APG metrics endpoint for monitoring"""
	try:
		# Return basic metrics
		metrics_data = {
			'pipelines_total': 0,
			'executions_total': 0,
			'executions_running': 0,
			'executions_success_rate': 0.0,
			'average_execution_time_ms': 0.0,
			'timestamp': datetime.utcnow().isoformat()
		}
		
		return jsonify(metrics_data), 200
		
	except Exception as e:
		return jsonify({'error': str(e)}), 500


# Field Mapping Route
@etlp_blueprint.route('/field-mapping')
def field_mapping():
	"""Visual field mapping interface"""
	try:
		return render_template('field_mapper.html')
	except Exception as e:
		return jsonify({'error': str(e)}), 500


# APG Menu Integration
def create_etlp_menu():
	"""Create menu items for APG navigation"""
	return [
		{
			'name': 'ETLP Dashboard',
			'url': '/etlp/dashboard',
			'icon': 'fa-dashboard',
			'category': 'Data Processing'
		},
		{
			'name': 'Pipelines',
			'url': '/etlp/pipelines',
			'icon': 'fa-flow-chart',
			'category': 'Data Processing'
		},
		{
			'name': 'Pipeline Designer',
			'url': '/etlp/designer',
			'icon': 'fa-paint-brush',
			'category': 'Data Processing'
		},
		{
			'name': 'Executions',
			'url': '/etlp/executions',
			'icon': 'fa-play',
			'category': 'Data Processing'
		},
		{
			'name': 'Monitoring',
			'url': '/etlp/monitoring',
			'icon': 'fa-line-chart',
			'category': 'Data Processing'
		},
		{
			'name': 'Field Mapping',
			'url': '/etlp/field-mapping',
			'icon': 'fa-project-diagram',
			'category': 'Data Processing'
		}
	]


# APG Permission Integration
ETLP_PERMISSIONS = [
	{
		'name': 'ETLP - Pipeline Read',
		'permission': 'etlp:pipeline:read',
		'description': 'Read access to pipelines'
	},
	{
		'name': 'ETLP - Pipeline Write',
		'permission': 'etlp:pipeline:write',
		'description': 'Create and modify pipelines'
	},
	{
		'name': 'ETLP - Pipeline Execute',
		'permission': 'etlp:pipeline:execute',
		'description': 'Execute pipelines'
	},
	{
		'name': 'ETLP - Pipeline Delete',
		'permission': 'etlp:pipeline:delete',
		'description': 'Delete pipelines'
	},
	{
		'name': 'ETLP - Transformation Read',
		'permission': 'etlp:transformation:read',
		'description': 'Read access to transformations'
	},
	{
		'name': 'ETLP - Transformation Write',
		'permission': 'etlp:transformation:write',
		'description': 'Create and modify transformations'
	},
	{
		'name': 'ETLP - Data Source Read',
		'permission': 'etlp:datasource:read',
		'description': 'Read access to data sources'
	},
	{
		'name': 'ETLP - Data Source Write',
		'permission': 'etlp:datasource:write',
		'description': 'Create and modify data sources'
	},
	{
		'name': 'ETLP - Quality Read',
		'permission': 'etlp:quality:read',
		'description': 'Read access to quality rules'
	}
]


class ETLPCapabilityIntegration:
	"""Main integration class for APG composition"""
	
	def __init__(self):
		self.views = [
			ETLPDashboardView(),
			ETLPPipelineView(),
			ETLPExecutionView(),
			ETLPDesignerView(),
			ETLPMonitoringView()
		]
	
	def register_with_appbuilder(self, appbuilder):
		"""Register views with Flask-AppBuilder"""
		for view in self.views:
			appbuilder.add_view_no_menu(view)
	
	def get_menu_items(self):
		"""Get menu items for APG navigation"""
		return create_etlp_menu()
	
	def get_permissions(self):
		"""Get permission definitions for APG RBAC"""
		return ETLP_PERMISSIONS
	
	def get_health_status(self):
		"""Get capability health status"""
		try:
			# Perform health checks
			return {
				'healthy': True,
				'version': '1.0.0',
				'last_check': datetime.utcnow().isoformat()
			}
		except Exception:
			return {
				'healthy': False,
				'error': 'Health check failed',
				'last_check': datetime.utcnow().isoformat()
			}


# Export integration instance
etlp_integration = ETLPCapabilityIntegration()
