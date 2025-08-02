"""
APG Workflow Orchestration Flask-AppBuilder Blueprint

APG-integrated Flask-AppBuilder blueprint with comprehensive workflow management views,
composition engine registration, menu integration, and APG platform compatibility.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import logging
import json

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.models.decorators import renders
from flask_appbuilder.widgets import ListWidget, ShowWidget, EditWidget
from flask_appbuilder.forms import DynamicForm
from flask_appbuilder.fieldwidgets import BS3TextAreaFieldWidget, BS3TextFieldWidget, Select2Widget
from flask_appbuilder.validators import Unique
from wtforms import StringField, TextAreaField, SelectField, IntegerField, BooleanField
from wtforms.validators import DataRequired, Length, Optional as OptionalValidator
from wtforms.widgets import TextArea
import redis.asyncio as redis

from .models import Workflow, WorkflowStatus, Priority, TaskType
from .database import CRWorkflow, CRWorkflowInstance, CRTaskExecution, DatabaseManager
from .service import WorkflowOrchestrationService
from .management import WorkflowManager, WorkflowValidationLevel
from .api import create_api_app

logger = logging.getLogger(__name__)

# Custom Widgets
class WorkflowDefinitionWidget(BS3TextAreaFieldWidget):
	"""Custom widget for workflow definition JSON editing."""
	
	def __call__(self, field, **kwargs):
		kwargs.setdefault('rows', 20)
		kwargs.setdefault('class', 'form-control workflow-definition-editor')
		kwargs.setdefault('data-mode', 'json')
		return super().__call__(field, **kwargs)

class TagsWidget(BS3TextFieldWidget):
	"""Custom widget for tags input."""
	
	def __call__(self, field, **kwargs):
		kwargs.setdefault('class', 'form-control tags-input')
		kwargs.setdefault('data-role', 'tagsinput')
		return super().__call__(field, **kwargs)

# Custom Fields
class WorkflowDefinitionField(TextAreaField):
	"""Custom field for workflow definition with JSON validation."""
	
	widget = WorkflowDefinitionWidget()
	
	def process_formdata(self, valuelist):
		if valuelist:
			try:
				# Validate JSON format
				json.loads(valuelist[0])
				self.data = valuelist[0]
			except json.JSONDecodeError as e:
				raise ValueError(f"Invalid JSON format: {e}")
		else:
			self.data = None

# APG Integration Components
class APGWorkflowForm(DynamicForm):
	"""APG-integrated workflow form with enhanced validation."""
	
	name = StringField(
		'Workflow Name',
		validators=[DataRequired(), Length(min=1, max=200)],
		widget=BS3TextFieldWidget(),
		description="Unique name for the workflow"
	)
	
	description = TextAreaField(
		'Description',
		validators=[OptionalValidator(), Length(max=1000)],
		widget=BS3TextAreaFieldWidget(),
		description="Detailed description of the workflow purpose"
	)
	
	definition = WorkflowDefinitionField(
		'Workflow Definition',
		validators=[DataRequired()],
		description="JSON definition of workflow tasks and structure"
	)
	
	priority = SelectField(
		'Priority',
		choices=[
			(Priority.LOW.value, 'Low'),
			(Priority.MEDIUM.value, 'Medium'),
			(Priority.HIGH.value, 'High'),
			(Priority.CRITICAL.value, 'Critical')
		],
		default=Priority.MEDIUM.value,
		widget=Select2Widget(),
		description="Workflow execution priority"
	)
	
	validation_level = SelectField(
		'Validation Level',
		choices=[
			(WorkflowValidationLevel.BASIC.value, 'Basic'),
			(WorkflowValidationLevel.STANDARD.value, 'Standard'),
			(WorkflowValidationLevel.STRICT.value, 'Strict'),
			(WorkflowValidationLevel.ENTERPRISE.value, 'Enterprise')
		],
		default=WorkflowValidationLevel.STANDARD.value,
		widget=Select2Widget(),
		description="Level of validation to apply"
	)
	
	tags = StringField(
		'Tags',
		validators=[OptionalValidator()],
		widget=TagsWidget(),
		description="Comma-separated tags for categorization"
	)
	
	sla_hours = IntegerField(
		'SLA Hours',
		validators=[OptionalValidator()],
		description="Service Level Agreement in hours"
	)
	
	auto_start = BooleanField(
		'Auto Start',
		default=False,
		description="Automatically start workflow when created"
	)

# APG Model Views
class APGWorkflowModelView(ModelView):
	"""APG-integrated workflow model view with enhanced functionality."""
	
	datamodel = SQLAInterface(CRWorkflow)
	
	# List view configuration
	list_columns = ['name', 'status', 'priority', 'created_by', 'created_at', 'updated_at']
	list_title = "APG Workflow Orchestration"
	
	# Show view configuration
	show_columns = [
		'name', 'description', 'status', 'priority', 'definition',
		'tags', 'sla_hours', 'tenant_id', 'created_by', 'created_at',
		'updated_by', 'updated_at', 'metadata'
	]
	show_title = "Workflow Details"
	
	# Edit view configuration
	edit_columns = [
		'name', 'description', 'definition', 'priority', 
		'validation_level', 'tags', 'sla_hours'
	]
	edit_title = "Edit Workflow"
	edit_form = APGWorkflowForm
	
	# Add view configuration
	add_columns = [
		'name', 'description', 'definition', 'priority',
		'validation_level', 'tags', 'sla_hours', 'auto_start'
	]
	add_title = "Create New Workflow"
	add_form = APGWorkflowForm
	
	# Search configuration
	search_columns = ['name', 'description', 'tags', 'created_by']
	search_exclude_columns = ['definition', 'metadata']
	
	# Filters
	base_filters = [['tenant_id', lambda: get_current_tenant_id(), '=']]
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	
	# Custom formatters
	formatters_columns = {
		'definition': lambda x: f"<pre class='workflow-json'>{json.dumps(json.loads(x) if isinstance(x, str) else x, indent=2)}</pre>",
		'tags': lambda x: ', '.join(json.loads(x) if isinstance(x, str) else x or []),
		'status': lambda x: f"<span class='label label-{get_status_class(x)}'>{x}</span>",
		'priority': lambda x: f"<span class='badge badge-{get_priority_class(x)}'>{x}</span>"
	}
	
	# Custom actions
	@expose('/execute/<workflow_id>')
	@has_access
	def execute_workflow(self, workflow_id):
		"""Execute a workflow."""
		try:
			# Get workflow service
			workflow_service = get_workflow_service()
			
			# Execute workflow
			instance = asyncio.run(workflow_service.execute_workflow(
				workflow_id=workflow_id,
				input_data={},
				user_id=get_current_user_id()
			))
			
			flash(f"Workflow execution started. Instance ID: {instance.id}", "success")
			return redirect(url_for('APGWorkflowInstanceModelView.show', pk=instance.id))
			
		except Exception as e:
			flash(f"Failed to execute workflow: {str(e)}", "error")
			return redirect(url_for('APGWorkflowModelView.list'))
	
	@expose('/clone/<workflow_id>')
	@has_access
	def clone_workflow(self, workflow_id):
		"""Clone a workflow."""
		try:
			# Get workflow manager
			workflow_manager = get_workflow_manager()
			
			# Get original workflow
			original = self.datamodel.get(workflow_id)
			if not original:
				flash("Workflow not found", "error")
				return redirect(url_for('APGWorkflowModelView.list'))
			
			# Clone workflow
			new_name = f"{original.name} (Clone)"
			cloned_workflow = asyncio.run(workflow_manager.clone_workflow(
				workflow_id, new_name, get_current_user_id()
			))
			
			flash(f"Workflow cloned successfully: {new_name}", "success")
			return redirect(url_for('APGWorkflowModelView.show', pk=cloned_workflow.id))
			
		except Exception as e:
			flash(f"Failed to clone workflow: {str(e)}", "error")
			return redirect(url_for('APGWorkflowModelView.list'))
	
	@expose('/export/<workflow_id>')
	@has_access
	def export_workflow(self, workflow_id):
		"""Export workflow definition."""
		try:
			workflow = self.datamodel.get(workflow_id)
			if not workflow:
				flash("Workflow not found", "error")
				return redirect(url_for('APGWorkflowModelView.list'))
			
			# Prepare export data
			export_data = {
				"name": workflow.name,
				"description": workflow.description,
				"definition": json.loads(workflow.definition) if isinstance(workflow.definition, str) else workflow.definition,
				"priority": workflow.priority,
				"tags": json.loads(workflow.tags) if isinstance(workflow.tags, str) else workflow.tags or [],
				"sla_hours": workflow.sla_hours,
				"exported_at": datetime.now(timezone.utc).isoformat(),
				"exported_by": get_current_user_id()
			}
			
			# Return JSON response for download
			response = jsonify(export_data)
			response.headers['Content-Disposition'] = f'attachment; filename={workflow.name}_export.json'
			response.headers['Content-Type'] = 'application/json'
			return response
			
		except Exception as e:
			flash(f"Failed to export workflow: {str(e)}", "error")
			return redirect(url_for('APGWorkflowModelView.list'))

class APGWorkflowInstanceModelView(ModelView):
	"""APG workflow instance view for execution monitoring."""
	
	datamodel = SQLAInterface(CRWorkflowInstance)
	
	# List view configuration
	list_columns = ['workflow_id', 'status', 'started_at', 'completed_at', 'started_by', 'progress_percentage']
	list_title = "Workflow Executions"
	
	# Show view configuration
	show_columns = [
		'workflow_id', 'status', 'started_at', 'completed_at', 
		'started_by', 'progress_percentage', 'input_data', 'output_data',
		'error_details', 'tenant_id'
	]
	show_title = "Execution Details"
	
	# No editing for instances
	base_permissions = ['can_list', 'can_show']
	
	# Filters
	base_filters = [['tenant_id', lambda: get_current_tenant_id(), '=']]
	
	# Custom formatters
	formatters_columns = {
		'status': lambda x: f"<span class='label label-{get_status_class(x)}'>{x}</span>",
		'progress_percentage': lambda x: f"<div class='progress'><div class='progress-bar' style='width: {x}%'>{x}%</div></div>",
		'input_data': lambda x: f"<pre class='json-data'>{json.dumps(json.loads(x) if isinstance(x, str) else x, indent=2)}</pre>",
		'output_data': lambda x: f"<pre class='json-data'>{json.dumps(json.loads(x) if isinstance(x, str) else x, indent=2)}</pre>" if x else "N/A"
	}
	
	# Custom actions
	@expose('/pause/<instance_id>')
	@has_access
	def pause_instance(self, instance_id):
		"""Pause workflow instance."""
		try:
			workflow_service = get_workflow_service()
			success = asyncio.run(workflow_service.pause_workflow_instance(
				instance_id, get_current_user_id()
			))
			
			if success:
				flash("Workflow instance paused successfully", "success")
			else:
				flash("Failed to pause workflow instance", "warning")
				
		except Exception as e:
			flash(f"Error pausing workflow instance: {str(e)}", "error")
		
		return redirect(url_for('APGWorkflowInstanceModelView.show', pk=instance_id))
	
	@expose('/resume/<instance_id>')
	@has_access
	def resume_instance(self, instance_id):
		"""Resume workflow instance."""
		try:
			workflow_service = get_workflow_service()
			success = asyncio.run(workflow_service.resume_workflow_instance(
				instance_id, get_current_user_id()
			))
			
			if success:
				flash("Workflow instance resumed successfully", "success")
			else:
				flash("Failed to resume workflow instance", "warning")
				
		except Exception as e:
			flash(f"Error resuming workflow instance: {str(e)}", "error")
		
		return redirect(url_for('APGWorkflowInstanceModelView.show', pk=instance_id))
	
	@expose('/stop/<instance_id>')
	@has_access
	def stop_instance(self, instance_id):
		"""Stop workflow instance."""
		try:
			workflow_service = get_workflow_service()
			success = asyncio.run(workflow_service.stop_workflow_instance(
				instance_id, get_current_user_id(), "Stopped by user"
			))
			
			if success:
				flash("Workflow instance stopped successfully", "success")
			else:
				flash("Failed to stop workflow instance", "warning")
				
		except Exception as e:
			flash(f"Error stopping workflow instance: {str(e)}", "error")
		
		return redirect(url_for('APGWorkflowInstanceModelView.show', pk=instance_id))

class APGWorkflowDashboardView(BaseView):
	"""APG workflow orchestration dashboard view."""
	
	route_base = "/workflow-dashboard"
	
	@expose('/')
	@has_access
	def dashboard(self):
		"""Main workflow dashboard."""
		try:
			# Get dashboard data
			workflow_manager = get_workflow_manager()
			
			# Get workflow statistics
			total_workflows = asyncio.run(get_workflow_count())
			active_instances = asyncio.run(get_active_instance_count())
			recent_executions = asyncio.run(get_recent_executions(limit=10))
			
			dashboard_data = {
				'total_workflows': total_workflows,
				'active_instances': active_instances,
				'recent_executions': recent_executions,
				'status_distribution': asyncio.run(get_status_distribution()),
				'performance_metrics': asyncio.run(get_performance_metrics())
			}
			
			return self.render_template(
				'workflow_dashboard.html',
				data=dashboard_data,
				title="Workflow Orchestration Dashboard"
			)
			
		except Exception as e:
			flash(f"Error loading dashboard: {str(e)}", "error")
			return self.render_template('error.html', error=str(e))
	
	@expose('/designer')
	@has_access
	def workflow_designer(self):
		"""Visual workflow designer interface."""
		return self.render_template(
			'workflow_designer.html',
			title="Workflow Designer"
		)
	
	@expose('/templates')
	@has_access
	def workflow_templates(self):
		"""Workflow templates gallery."""
		try:
			# Get available templates
			templates = asyncio.run(get_workflow_templates())
			
			return self.render_template(
				'workflow_templates.html',
				templates=templates,
				title="Workflow Templates"
			)
			
		except Exception as e:
			flash(f"Error loading templates: {str(e)}", "error")
			return self.render_template('error.html', error=str(e))
	
	@expose('/monitoring')
	@has_access
	def workflow_monitoring(self):
		"""Real-time workflow monitoring."""
		return self.render_template(
			'workflow_monitoring.html',
			title="Workflow Monitoring"
		)

# APG Composition Engine Integration
class APGWorkflowOrchestrationBlueprint:
	"""APG composition engine integrated blueprint."""
	
	def __init__(
		self,
		database_manager: DatabaseManager,
		redis_client: redis.Redis,
		tenant_id: str = "default"
	):
		self.database_manager = database_manager
		self.redis_client = redis_client
		self.tenant_id = tenant_id
		
		# Initialize services
		self.workflow_service = WorkflowOrchestrationService(database_manager, redis_client, tenant_id)
		self.workflow_manager = WorkflowManager(database_manager, redis_client, tenant_id)
		
		# Create blueprint
		self.blueprint = self._create_blueprint()
		
		logger.info(f"Initialized APG Workflow Orchestration Blueprint for tenant {tenant_id}")
	
	def _create_blueprint(self) -> Blueprint:
		"""Create Flask blueprint with APG integration."""
		
		blueprint = Blueprint(
			'workflow_orchestration',
			__name__,
			template_folder='templates',
			static_folder='static',
			url_prefix='/workflow-orchestration'
		)
		
		# Register API routes
		api_app = create_api_app(self.database_manager, self.redis_client, self.tenant_id)
		
		# Mount API as sub-application
		@blueprint.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
		def api_proxy(path):
			"""Proxy requests to FastAPI application."""
			# This would integrate the FastAPI app with Flask blueprint
			# Implementation would depend on the specific APG architecture
			pass
		
		return blueprint
	
	def register_with_appbuilder(self, appbuilder):
		"""Register views with Flask-AppBuilder."""
		
		# Register model views
		appbuilder.add_view(
			APGWorkflowModelView,
			"Workflows",
			icon="fa-sitemap",
			category="Workflow Orchestration",
			category_icon="fa-cogs"
		)
		
		appbuilder.add_view(
			APGWorkflowInstanceModelView,
			"Executions",
			icon="fa-play",
			category="Workflow Orchestration"
		)
		
		# Register dashboard view
		appbuilder.add_view(
			APGWorkflowDashboardView,
			"Dashboard",
			icon="fa-dashboard",
			category="Workflow Orchestration"
		)
		
		# Register menu items
		appbuilder.add_link(
			"Workflow Designer",
			href="/workflow-dashboard/designer",
			icon="fa-pencil-square-o",
			category="Workflow Orchestration"
		)
		
		appbuilder.add_link(
			"Templates",
			href="/workflow-dashboard/templates",
			icon="fa-file-code-o",
			category="Workflow Orchestration"
		)
		
		appbuilder.add_link(
			"Monitoring",
			href="/workflow-dashboard/monitoring",
			icon="fa-line-chart",
			category="Workflow Orchestration"
		)
		
		# Register separator
		appbuilder.add_separator("Workflow Orchestration")
		
		# Register API documentation
		appbuilder.add_link(
			"API Documentation",
			href="/workflow-orchestration/api/docs",
			icon="fa-book",
			category="Workflow Orchestration"
		)
	
	def get_blueprint(self) -> Blueprint:
		"""Get the configured blueprint."""
		return self.blueprint
	
	def get_capability_info(self) -> Dict[str, Any]:
		"""Get APG capability information for composition engine."""
		return {
			"name": "workflow_orchestration",
			"version": "1.0.0",
			"description": "Enterprise workflow orchestration and automation platform",
			"capabilities_provided": [
				"workflow_creation",
				"workflow_execution", 
				"task_scheduling",
				"state_management",
				"version_control",
				"deployment_automation",
				"monitoring_analytics"
			],
			"capabilities_required": [
				"auth_rbac",
				"audit_compliance", 
				"notification_engine",
				"document_management"
			],
			"endpoints": {
				"health": "/workflow-orchestration/health",
				"api": "/workflow-orchestration/api",
				"dashboard": "/workflow-dashboard",
				"designer": "/workflow-dashboard/designer"
			},
			"menu_items": [
				{
					"name": "Workflow Orchestration",
					"icon": "fa-cogs",
					"category": True,
					"items": [
						{"name": "Dashboard", "href": "/workflow-dashboard", "icon": "fa-dashboard"},
						{"name": "Workflows", "href": "/workflow-orchestration/workflows", "icon": "fa-sitemap"},
						{"name": "Executions", "href": "/workflow-orchestration/instances", "icon": "fa-play"},
						{"name": "Designer", "href": "/workflow-dashboard/designer", "icon": "fa-pencil-square-o"},
						{"name": "Templates", "href": "/workflow-dashboard/templates", "icon": "fa-file-code-o"},
						{"name": "Monitoring", "href": "/workflow-dashboard/monitoring", "icon": "fa-line-chart"}
					]
				}
			],
			"permissions": [
				"workflows.read",
				"workflows.write", 
				"workflows.execute",
				"workflows.delete",
				"workflows.admin"
			],
			"database_tables": [
				"cr_workflows",
				"cr_workflow_instances",
				"cr_task_executions",
				"cr_workflow_versions",
				"cr_deployment_environments",
				"cr_deployment_plans",
				"cr_deployment_executions"
			]
		}

# Utility Functions
def get_current_tenant_id() -> str:
	"""Get current tenant ID from session/context."""
	try:
		# Try to get tenant ID from Flask session/request context
		from flask import session, request, g
		
		# Check session first
		if 'tenant_id' in session:
			return session['tenant_id']
		
		# Check request headers for API calls
		if hasattr(request, 'headers') and 'X-Tenant-ID' in request.headers:
			return request.headers['X-Tenant-ID']
		
		# Check Flask global context
		if hasattr(g, 'tenant_id'):
			return g.tenant_id
		
		# Check for APG context if available
		try:
			from apg.core.context import get_current_context
			context = get_current_context()
			if context and hasattr(context, 'tenant_id'):
				return context.tenant_id
		except ImportError:
			pass  # APG context not available
		
		# Fallback to environment variable
		import os
		tenant_id = os.getenv('APG_TENANT_ID', 'default_tenant')
		return tenant_id
		
	except Exception as e:
		logger.warning(f"Failed to get tenant ID from context: {e}")
		return "default_tenant"

def get_current_user_id() -> str:
	"""Get current user ID from session/context."""
	try:
		# Try to get user ID from Flask session/request context
		from flask import session, request, g
		from flask_login import current_user
		
		# Check Flask-Login current user
		if hasattr(current_user, 'id') and current_user.is_authenticated:
			return str(current_user.id)
		
		# Check session
		if 'user_id' in session:
			return session['user_id']
		
		# Check request headers for API calls
		if hasattr(request, 'headers'):
			if 'X-User-ID' in request.headers:
				return request.headers['X-User-ID']
			
			# Check Authorization header for JWT token
			auth_header = request.headers.get('Authorization', '')
			if auth_header.startswith('Bearer '):
				try:
					import jwt
					token = auth_header.split(' ')[1]
					# Decode without verification for user ID (verification should be done in middleware)
					payload = jwt.decode(token, options={"verify_signature": False})
					if 'user_id' in payload:
						return str(payload['user_id'])
					elif 'sub' in payload:
						return str(payload['sub'])
				except Exception:
					pass  # JWT decode failed
		
		# Check Flask global context
		if hasattr(g, 'user_id'):
			return g.user_id
		
		# Check for APG authentication context
		try:
			from apg.core.auth import get_current_user
			user = get_current_user()
			if user and hasattr(user, 'id'):
				return str(user.id)
		except ImportError:
			pass  # APG auth not available
		
		# Check for Flask-AppBuilder security context
		try:
			from flask_appbuilder.security.decorators import current_user as fab_user
			if fab_user and hasattr(fab_user, 'id'):
				return str(fab_user.id)
		except ImportError:
			pass  # Flask-AppBuilder not available
		
		# Fallback to environment variable or system user
		import os
		user_id = os.getenv('APG_USER_ID', 'system_user')
		return user_id
		
	except Exception as e:
		logger.warning(f"Failed to get user ID from context: {e}")
		return "system_user"

def get_workflow_service() -> WorkflowOrchestrationService:
	"""Get workflow service instance."""
	try:
		# Try to get service from Flask application context
		from flask import current_app, g
		
		# Check Flask global context first
		if hasattr(g, 'workflow_service'):
			return g.workflow_service
		
		# Check Flask application context
		if hasattr(current_app, 'extensions') and 'workflow_orchestration' in current_app.extensions:
			service = current_app.extensions['workflow_orchestration'].workflow_service
			if service:
				return service
		
		# Check for APG service registry
		try:
			from apg.core.services import get_service
			service = get_service('workflow_orchestration')
			if service:
				return service
		except ImportError:
			pass  # APG service registry not available
		
		# Try to create service from configuration
		try:
			from .service import WorkflowOrchestrationService
			from .database import DatabaseManager
			import redis.asyncio as redis
			import os
			
			# Get configuration from environment or app config
			database_url = os.getenv('DATABASE_URL', 'postgresql://localhost/workflow_orchestration')
			redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
			tenant_id = get_current_tenant_id()
			
			# Create database manager
			database_manager = DatabaseManager(database_url)
			
			# Create Redis client
			redis_client = redis.from_url(redis_url)
			
			# Create service instance
			service = WorkflowOrchestrationService(database_manager, redis_client, tenant_id)
			
			# Cache in Flask global context
			g.workflow_service = service
			
			return service
			
		except Exception as service_error:
			logger.error(f"Failed to create workflow service: {service_error}")
			return None
		
	except Exception as e:
		logger.error(f"Failed to get workflow service from context: {e}")
		return None

def get_workflow_manager() -> WorkflowManager:
	"""Get workflow manager instance."""
	try:
		# Try to get manager from Flask application context
		from flask import current_app, g
		
		# Check Flask global context first
		if hasattr(g, 'workflow_manager'):
			return g.workflow_manager
		
		# Check Flask application context
		if hasattr(current_app, 'extensions') and 'workflow_orchestration' in current_app.extensions:
			manager = current_app.extensions['workflow_orchestration'].workflow_manager
			if manager:
				return manager
		
		# Check for APG service registry
		try:
			from apg.core.services import get_service
			manager = get_service('workflow_manager')
			if manager:
				return manager
		except ImportError:
			pass  # APG service registry not available
		
		# Try to create manager from configuration
		try:
			from .management import WorkflowManager
			from .database import DatabaseManager
			import redis.asyncio as redis
			import os
			
			# Get configuration from environment or app config
			database_url = os.getenv('DATABASE_URL', 'postgresql://localhost/workflow_orchestration')
			redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
			tenant_id = get_current_tenant_id()
			
			# Create database manager
			database_manager = DatabaseManager(database_url)
			
			# Create Redis client
			redis_client = redis.from_url(redis_url)
			
			# Create manager instance
			manager = WorkflowManager(database_manager, redis_client, tenant_id)
			
			# Cache in Flask global context
			g.workflow_manager = manager
			
			return manager
			
		except Exception as manager_error:
			logger.error(f"Failed to create workflow manager: {manager_error}")
			return None
		
	except Exception as e:
		logger.error(f"Failed to get workflow manager from context: {e}")
		return None

def get_status_class(status: str) -> str:
	"""Get CSS class for status display."""
	status_classes = {
		'draft': 'default',
		'active': 'success',
		'paused': 'warning',
		'completed': 'success',
		'failed': 'danger',
		'cancelled': 'default'
	}
	return status_classes.get(status.lower(), 'default')

def get_priority_class(priority: str) -> str:
	"""Get CSS class for priority display."""
	priority_classes = {
		'low': 'default',
		'medium': 'info',
		'high': 'warning',
		'critical': 'danger'
	}
	return priority_classes.get(priority.lower(), 'default')

# Async helper functions
async def get_workflow_count() -> int:
	"""Get total workflow count from database."""
	try:
		from .models import WorkflowDefinition
		from .database import get_async_session
		from sqlalchemy import func, select
		
		async with get_async_session() as session:
			result = await session.execute(
				select(func.count(WorkflowDefinition.id))
			)
			count = result.scalar()
			return count or 0
	except Exception:
		# Fallback for development/testing
		return 0

async def get_active_instance_count() -> int:
	"""Get active workflow instance count from database."""
	try:
		from .models import WorkflowInstance
		from .database import get_async_session
		from sqlalchemy import func, select
		
		async with get_async_session() as session:
			result = await session.execute(
				select(func.count(WorkflowInstance.id))
				.where(WorkflowInstance.status.in_(['running', 'paused', 'waiting']))
			)
			count = result.scalar()
			return count or 0
	except Exception:
		# Fallback for development/testing
		return 0

async def get_recent_executions(limit: int = 10) -> List[Dict[str, Any]]:
	"""Get recent workflow executions from database."""
	try:
		from .models import WorkflowExecution
		from .database import get_async_session
		from sqlalchemy import select, desc
		
		async with get_async_session() as session:
			result = await session.execute(
				select(WorkflowExecution)
				.order_by(desc(WorkflowExecution.created_at))
				.limit(limit)
			)
			executions = result.scalars().all()
			
			return [
				{
					"id": str(execution.id),
					"workflow_id": str(execution.workflow_id),
					"status": execution.status,
					"started_at": execution.started_at.isoformat() if execution.started_at else None,
					"completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
					"duration": (execution.completed_at - execution.started_at).total_seconds() if execution.completed_at and execution.started_at else None,
					"success_rate": execution.success_rate or 0.0
				}
				for execution in executions
			]
	except Exception:
		# Fallback for development/testing
		return []

async def get_status_distribution() -> Dict[str, int]:
	"""Get workflow status distribution from database."""
	try:
		from .models import WorkflowInstance
		from .database import get_async_session
		from sqlalchemy import func, select
		
		async with get_async_session() as session:
			result = await session.execute(
				select(
					WorkflowInstance.status,
					func.count(WorkflowInstance.id).label('count')
				).group_by(WorkflowInstance.status)
			)
			
			distribution = {}
			for status, count in result.fetchall():
				distribution[status] = count
			
			return distribution
	except Exception:
		# Fallback for development/testing
		return {}

async def get_performance_metrics() -> Dict[str, Any]:
	"""Get performance metrics from database."""
	try:
		from .models import WorkflowExecution, WorkflowInstance
		from .database import get_async_session
		from sqlalchemy import func, select
		from datetime import datetime, timedelta
		
		async with get_async_session() as session:
			# Get metrics for last 30 days
			thirty_days_ago = datetime.utcnow() - timedelta(days=30)
			
			# Average execution time
			avg_duration_result = await session.execute(
				select(func.avg(
					func.extract('epoch', WorkflowExecution.completed_at - WorkflowExecution.started_at)
				)).where(
					WorkflowExecution.completed_at.isnot(None),
					WorkflowExecution.started_at >= thirty_days_ago
				)
			)
			avg_duration = avg_duration_result.scalar() or 0
			
			# Success rate
			success_result = await session.execute(
				select(
					func.count().filter(WorkflowExecution.status == 'completed').label('successful'),
					func.count().label('total')
				).where(WorkflowExecution.created_at >= thirty_days_ago)
			)
			success_data = success_result.fetchone()
			success_rate = (success_data.successful / success_data.total * 100) if success_data.total > 0 else 0
			
			# Throughput (executions per day)
			throughput_result = await session.execute(
				select(func.count(WorkflowExecution.id))
				.where(WorkflowExecution.created_at >= thirty_days_ago)
			)
			total_executions = throughput_result.scalar() or 0
			throughput = total_executions / 30.0
			
			return {
				"average_duration_seconds": round(avg_duration, 2),
				"success_rate_percent": round(success_rate, 2),
				"daily_throughput": round(throughput, 2),
				"total_executions_30d": total_executions,
				"period": "last_30_days"
			}
	except Exception:
		# Fallback for development/testing
		return {
			"average_duration_seconds": 0.0,
			"success_rate_percent": 0.0,
			"daily_throughput": 0.0,
			"total_executions_30d": 0,
			"period": "last_30_days"
		}

async def get_workflow_templates() -> List[Dict[str, Any]]:
	"""Get available workflow templates from database and library."""
	try:
		from .models import WorkflowTemplate as WorkflowTemplateModel
		from .database import get_async_session
		from sqlalchemy import select
		
		async with get_async_session() as session:
			result = await session.execute(
				select(WorkflowTemplateModel)
				.where(WorkflowTemplateModel.is_active == True)
				.order_by(WorkflowTemplateModel.name)
			)
			templates = result.scalars().all()
			
			return [
				{
					"id": str(template.id),
					"name": template.name,
					"description": template.description,
					"category": template.category,
					"tags": template.tags or [],
					"version": template.version,
					"complexity_score": template.complexity_score,
					"estimated_duration": template.estimated_duration,
					"is_featured": template.is_featured,
					"created_at": template.created_at.isoformat() if template.created_at else None,
					"updated_at": template.updated_at.isoformat() if template.updated_at else None
				}
				for template in templates
			]
	except Exception:
		# Fallback - return some basic templates
		return [
			{
				"id": "basic_approval_001",
				"name": "Basic Approval Workflow",
				"description": "Simple approval workflow template",
				"category": "business_process",
				"tags": ["approval", "basic"],
				"version": "1.0.0",
				"complexity_score": 3.0,
				"estimated_duration": 3600,
				"is_featured": True,
				"created_at": datetime.utcnow().isoformat(),
				"updated_at": datetime.utcnow().isoformat()
			}
		]

# Factory function
def create_apg_blueprint(
	database_manager: DatabaseManager,
	redis_client: redis.Redis,
	tenant_id: str = "default"
) -> APGWorkflowOrchestrationBlueprint:
	"""Factory function to create APG workflow orchestration blueprint."""
	return APGWorkflowOrchestrationBlueprint(database_manager, redis_client, tenant_id)

# Export blueprint components
__all__ = [
	"APGWorkflowOrchestrationBlueprint",
	"APGWorkflowModelView",
	"APGWorkflowInstanceModelView", 
	"APGWorkflowDashboardView",
	"create_apg_blueprint"
]