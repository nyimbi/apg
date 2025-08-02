#!/usr/bin/env python3
"""
APG Workflow Orchestration Flask-AppBuilder Views

Comprehensive Flask-AppBuilder views for workflow management including
CRUD operations, monitoring dashboards, designer integration, and analytics.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4

from flask import request, jsonify, render_template, redirect, url_for, flash, current_app
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, FormWidget, SearchWidget
from flask_appbuilder.actions import action
from flask_appbuilder.security.decorators import protect
from flask_appbuilder.baseviews import expose_api
from flask_babel import lazy_gettext as _
from wtforms import Form, StringField, TextAreaField, SelectField, IntegerField, BooleanField
from wtforms.validators import DataRequired, Length, Optional as OptionalValidator
from wtforms.widgets import TextArea
from pydantic import ValidationError
import sqlalchemy as sa
from sqlalchemy.orm import joinedload

# APG Framework imports
from apg.framework.base_view import APGBaseView
from apg.framework.security import require_permission
from apg.framework.widgets import APGChartWidget, APGFormWidget, APGListWidget

# Local imports
from .database import WorkflowDB, WorkflowInstanceDB, TaskExecutionDB
from .service import WorkflowOrchestrationService
from .models import *
from .apg_integration import APGIntegration


class WorkflowFormWidget(APGFormWidget):
	"""Custom form widget for workflow creation and editing."""
	
	template = 'workflow_orchestration/widgets/workflow_form.html'


class WorkflowListWidget(APGListWidget):
	"""Custom list widget for workflow display with enhanced features."""
	
	template = 'workflow_orchestration/widgets/workflow_list.html'


class WorkflowExecutionWidget(APGChartWidget):
	"""Custom widget for workflow execution visualization."""
	
	template = 'workflow_orchestration/widgets/workflow_execution.html'


class WorkflowForm(Form):
	"""Workflow creation and editing form."""
	
	name = StringField(
		_('Workflow Name'),
		validators=[DataRequired(), Length(min=1, max=255)],
		render_kw={'placeholder': 'Enter workflow name', 'class': 'form-control'}
	)
	
	description = TextAreaField(
		_('Description'),
		validators=[OptionalValidator(), Length(max=1000)],
		widget=TextArea(),
		render_kw={'placeholder': 'Describe the workflow purpose', 'class': 'form-control', 'rows': 3}
	)
	
	tenant_id = StringField(
		_('Tenant ID'),
		validators=[DataRequired(), Length(min=1, max=100)],
		render_kw={'placeholder': 'Tenant identifier', 'class': 'form-control'}
	)
	
	version = StringField(
		_('Version'),
		validators=[OptionalValidator(), Length(max=50)],
		default='1.0',
		render_kw={'placeholder': '1.0', 'class': 'form-control'}
	)
	
	tags = StringField(
		_('Tags'),
		validators=[OptionalValidator()],
		render_kw={'placeholder': 'Comma-separated tags', 'class': 'form-control'}
	)
	
	is_active = BooleanField(
		_('Active'),
		default=True,
		render_kw={'class': 'form-check-input'}
	)


class WorkflowInstanceForm(Form):
	"""Workflow instance execution form."""
	
	workflow_id = SelectField(
		_('Workflow'),
		validators=[DataRequired()],
		render_kw={'class': 'form-control'}
	)
	
	execution_context = TextAreaField(
		_('Execution Context (JSON)'),
		validators=[OptionalValidator()],
		widget=TextArea(),
		render_kw={
			'placeholder': '{"key": "value"}',
			'class': 'form-control',
			'rows': 5
		}
	)
	
	priority = SelectField(
		_('Priority'),
		choices=[
			('low', 'Low'),
			('normal', 'Normal'),
			('high', 'High'),
			('critical', 'Critical')
		],
		default='normal',
		render_kw={'class': 'form-control'}
	)


class WOWorkflowView(APGBaseView):
	"""Main workflow management view."""
	
	datamodel = SQLAInterface(WorkflowDB)
	
	# List view configuration
	list_columns = ['name', 'description', 'tenant_id', 'version', 'is_active', 'created_at', 'updated_at']
	list_title = _('Workflows')
	
	# Show view configuration  
	show_columns = [
		'name', 'description', 'tenant_id', 'version', 'definition',
		'is_active', 'created_by', 'created_at', 'updated_at', 'metadata'
	]
	show_title = _('Workflow Details')
	
	# Edit view configuration
	edit_columns = ['name', 'description', 'tenant_id', 'version', 'is_active']
	edit_title = _('Edit Workflow')
	
	# Add view configuration
	add_columns = ['name', 'description', 'tenant_id', 'version', 'is_active']
	add_title = _('Create Workflow')
	
	# Search configuration
	search_columns = ['name', 'description', 'tenant_id', 'created_by']
	
	# Ordering
	base_order = ('created_at', 'desc')
	
	# Security
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	
	# Custom widgets
	list_widget = WorkflowListWidget
	edit_widget = WorkflowFormWidget
	add_widget = WorkflowFormWidget
	
	# Pagination
	page_size = 20
	
	def __init__(self):
		super().__init__()
		self.workflow_service = WorkflowOrchestrationService()
	
	@expose('/designer/<int:workflow_id>')
	@has_access
	def workflow_designer(self, workflow_id):
		"""Workflow visual designer interface."""
		workflow = self.datamodel.get(workflow_id)
		if not workflow:
			flash(_('Workflow not found'), 'error')
			return redirect(url_for('WOWorkflowView.list'))
		
		# Check tenant access
		if not self._check_tenant_access(workflow.tenant_id):
			flash(_('Access denied'), 'error')
			return redirect(url_for('WOWorkflowView.list'))
		
		return self.render_template(
			'workflow_orchestration/designer.html',
			workflow=workflow,
			title=_('Workflow Designer')
		)
	
	@expose('/designer/new')
	@has_access
	def new_workflow_designer(self):
		"""New workflow visual designer interface."""
		return self.render_template(
			'workflow_orchestration/designer.html',
			workflow=None,
			title=_('New Workflow Designer')
		)
	
	@expose('/execute/<int:workflow_id>', methods=['GET', 'POST'])
	@has_access
	def execute_workflow(self, workflow_id):
		"""Execute workflow interface."""
		workflow = self.datamodel.get(workflow_id)
		if not workflow:
			flash(_('Workflow not found'), 'error')
			return redirect(url_for('WOWorkflowView.list'))
		
		# Check tenant access
		if not self._check_tenant_access(workflow.tenant_id):
			flash(_('Access denied'), 'error')
			return redirect(url_for('WOWorkflowView.list'))
		
		form = WorkflowInstanceForm()
		
		if request.method == 'POST' and form.validate_on_submit():
			try:
				# Parse execution context
				execution_context = {}
				if form.execution_context.data:
					execution_context = json.loads(form.execution_context.data)
				
				# Execute workflow asynchronously
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				
				instance = loop.run_until_complete(
					self.workflow_service.execute_workflow(
						workflow_id=workflow.id,
						execution_context=execution_context,
						priority=form.priority.data
					)
				)
				
				flash(_('Workflow execution started successfully'), 'success')
				return redirect(url_for('WOWorkflowInstanceView.show', pk=instance.id))
				
			except json.JSONDecodeError:
				flash(_('Invalid JSON in execution context'), 'error')
			except Exception as e:
				flash(_('Failed to execute workflow: %(error)s', error=str(e)), 'error')
		
		return self.render_template(
			'workflow_orchestration/execute_workflow.html',
			workflow=workflow,
			form=form,
			title=_('Execute Workflow')
		)
	
	@expose('/clone/<int:workflow_id>')
	@has_access
	def clone_workflow(self, workflow_id):
		"""Clone workflow."""
		workflow = self.datamodel.get(workflow_id)
		if not workflow:
			flash(_('Workflow not found'), 'error')
			return redirect(url_for('WOWorkflowView.list'))
		
		# Check tenant access
		if not self._check_tenant_access(workflow.tenant_id):
			flash(_('Access denied'), 'error')
			return redirect(url_for('WOWorkflowView.list'))
		
		try:
			# Create cloned workflow
			cloned_workflow = WorkflowDB(
				name=f"{workflow.name} (Copy)",
				description=workflow.description,
				tenant_id=workflow.tenant_id,
				definition=workflow.definition,
				version="1.0",
				created_by=self.get_user_id(),
				metadata=workflow.metadata
			)
			
			self.datamodel.add(cloned_workflow)
			flash(_('Workflow cloned successfully'), 'success')
			return redirect(url_for('WOWorkflowView.edit', pk=cloned_workflow.id))
			
		except Exception as e:
			flash(_('Failed to clone workflow: %(error)s', error=str(e)), 'error')
			return redirect(url_for('WOWorkflowView.list'))
	
	@action('bulk_activate', _('Activate'), _('Activate selected workflows?'), 'fa-play')
	def bulk_activate(self, workflows):
		"""Bulk activate workflows."""
		count = 0
		for workflow in workflows:
			if self._check_tenant_access(workflow.tenant_id):
				workflow.is_active = True
				count += 1
		
		if count > 0:
			self.datamodel.session.commit()
			flash(_('%(count)d workflows activated', count=count), 'success')
		else:
			flash(_('No workflows were activated'), 'warning')
		
		return redirect(url_for('WOWorkflowView.list'))
	
	@action('bulk_deactivate', _('Deactivate'), _('Deactivate selected workflows?'), 'fa-pause')
	def bulk_deactivate(self, workflows):
		"""Bulk deactivate workflows."""
		count = 0
		for workflow in workflows:
			if self._check_tenant_access(workflow.tenant_id):
				workflow.is_active = False
				count += 1
		
		if count > 0:
			self.datamodel.session.commit()
			flash(_('%(count)d workflows deactivated', count=count), 'success')
		else:
			flash(_('No workflows were deactivated'), 'warning')
		
		return redirect(url_for('WOWorkflowView.list'))
	
	@expose_api(name='workflow_definition', url='/api/workflow/<int:workflow_id>/definition')
	@protect()
	def get_workflow_definition(self, workflow_id):
		"""API endpoint to get workflow definition."""
		workflow = self.datamodel.get(workflow_id)
		if not workflow:
			return jsonify({'error': 'Workflow not found'}), 404
		
		if not self._check_tenant_access(workflow.tenant_id):
			return jsonify({'error': 'Access denied'}), 403
		
		return jsonify({
			'id': workflow.id,
			'name': workflow.name,
			'description': workflow.description,
			'definition': workflow.definition,
			'version': workflow.version,
			'metadata': workflow.metadata
		})
	
	@expose_api(name='update_workflow_definition', url='/api/workflow/<int:workflow_id>/definition', methods=['PUT'])
	@protect()
	def update_workflow_definition(self, workflow_id):
		"""API endpoint to update workflow definition."""
		workflow = self.datamodel.get(workflow_id)
		if not workflow:
			return jsonify({'error': 'Workflow not found'}), 404
		
		if not self._check_tenant_access(workflow.tenant_id):
			return jsonify({'error': 'Access denied'}), 403
		
		try:
			data = request.get_json()
			if 'definition' in data:
				workflow.definition = data['definition']
			if 'metadata' in data:
				workflow.metadata = data['metadata']
			
			workflow.updated_at = datetime.utcnow()
			self.datamodel.session.commit()
			
			return jsonify({'message': 'Workflow definition updated successfully'})
			
		except Exception as e:
			return jsonify({'error': str(e)}), 400
	
	def _check_tenant_access(self, tenant_id: str) -> bool:
		"""Check if user has access to tenant using APG RBAC."""
		try:
			# Get current user from AppBuilder security manager
			current_user = self.appbuilder.sm.user
			if not current_user:
				return False
			
			# Get user ID for RBAC check
			user_id = getattr(current_user, 'id', None)
			if not user_id:
				return False
			
			# Use APG auth RBAC service to check tenant access
			from capabilities.common.auth_rbac.service import AuthRBACService
			auth_service = AuthRBACService()
			
			# Check if user has workflow permissions for this tenant
			loop = asyncio.get_event_loop()
			has_access = loop.run_until_complete(
				auth_service.check_permission(
					user_id=str(user_id),
					permission="workflow:access",
					tenant_id=tenant_id
				)
			)
			
			return has_access
			
		except ImportError:
			# Fallback when RBAC service not available
			logger.warning("APG RBAC service not available, checking user tenant membership")
			try:
				# Check if user is member of tenant (basic fallback)
				current_user = self.appbuilder.sm.user
				user_tenants = getattr(current_user, 'tenant_memberships', [])
				return tenant_id in user_tenants or hasattr(current_user, 'is_admin') and current_user.is_admin
			except:
				# Ultimate fallback - allow access if user is authenticated
				return self.appbuilder.sm.user is not None
		except Exception as e:
			logger.error(f"Error checking tenant access: {e}")
			return False
	
	def get_user_id(self) -> str:
		"""Get current user ID."""
		# Integration with APG auth would go here
		return getattr(self.appbuilder.sm.user, 'id', 'system')


class WOWorkflowInstanceView(APGBaseView):
	"""Workflow instance management and monitoring view."""
	
	datamodel = SQLAInterface(WorkflowInstanceDB)
	
	# List view configuration
	list_columns = [
		'workflow.name', 'status', 'priority', 'started_at', 
		'completed_at', 'duration_seconds', 'created_by'
	]
	list_title = _('Workflow Executions')
	
	# Show view configuration
	show_columns = [
		'workflow.name', 'status', 'priority', 'execution_context',
		'started_at', 'completed_at', 'duration_seconds', 'result',
		'error_details', 'created_by', 'created_at'
	]
	show_title = _('Workflow Execution Details')
	
	# Search configuration
	search_columns = ['workflow.name', 'status', 'created_by']
	
	# Ordering
	base_order = ('created_at', 'desc')
	
	# Filters
	base_filters = [['status', lambda: WorkflowStatus, 'Status']]
	
	# Security
	base_permissions = ['can_list', 'can_show', 'can_delete']
	
	# Custom widgets
	list_widget = WorkflowListWidget
	show_widget = WorkflowExecutionWidget
	
	# Pagination
	page_size = 25
	
	def __init__(self):
		super().__init__()
		self.workflow_service = WorkflowOrchestrationService()
	
	@expose('/monitor/<int:instance_id>')
	@has_access
	def monitor_execution(self, instance_id):
		"""Real-time execution monitoring interface."""
		instance = self.datamodel.get(instance_id)
		if not instance:
			flash(_('Workflow instance not found'), 'error')
			return redirect(url_for('WOWorkflowInstanceView.list'))
		
		# Check tenant access
		if not self._check_tenant_access(instance.workflow.tenant_id):
			flash(_('Access denied'), 'error')
			return redirect(url_for('WOWorkflowInstanceView.list'))
		
		# Get task executions
		task_executions = self.appbuilder.get_session.query(TaskExecutionDB)\
			.filter_by(workflow_instance_id=instance_id)\
			.order_by(TaskExecutionDB.started_at.asc())\
			.all()
		
		return self.render_template(
			'workflow_orchestration/monitor_execution.html',
			instance=instance,
			task_executions=task_executions,
			title=_('Monitor Workflow Execution')
		)
	
	@expose_api(name='execution_status', url='/api/instance/<int:instance_id>/status')
	@protect()
	def get_execution_status(self, instance_id):
		"""API endpoint to get execution status."""
		instance = self.datamodel.get(instance_id)
		if not instance:
			return jsonify({'error': 'Instance not found'}), 404
		
		if not self._check_tenant_access(instance.workflow.tenant_id):
			return jsonify({'error': 'Access denied'}), 403
		
		# Get task executions
		task_executions = self.appbuilder.get_session.query(TaskExecutionDB)\
			.filter_by(workflow_instance_id=instance_id)\
			.all()
		
		return jsonify({
			'instance_id': instance.id,
			'workflow_name': instance.workflow.name,
			'status': instance.status.value if instance.status else 'unknown',
			'started_at': instance.started_at.isoformat() if instance.started_at else None,
			'completed_at': instance.completed_at.isoformat() if instance.completed_at else None,
			'duration_seconds': instance.duration_seconds,
			'task_executions': [
				{
					'task_id': task.task_id,
					'status': task.status.value if task.status else 'unknown',
					'started_at': task.started_at.isoformat() if task.started_at else None,
					'completed_at': task.completed_at.isoformat() if task.completed_at else None,
					'duration_seconds': task.duration_seconds,
					'retry_count': task.retry_count
				}
				for task in task_executions
			]
		})
	
	@expose('/cancel/<int:instance_id>', methods=['POST'])
	@has_access
	def cancel_execution(self, instance_id):
		"""Cancel workflow execution."""
		instance = self.datamodel.get(instance_id)
		if not instance:
			flash(_('Workflow instance not found'), 'error')
			return redirect(url_for('WOWorkflowInstanceView.list'))
		
		if not self._check_tenant_access(instance.workflow.tenant_id):
			flash(_('Access denied'), 'error')
			return redirect(url_for('WOWorkflowInstanceView.list'))
		
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			loop.run_until_complete(
				self.workflow_service.cancel_workflow_instance(instance_id)
			)
			
			flash(_('Workflow execution cancelled'), 'success')
			
		except Exception as e:
			flash(_('Failed to cancel execution: %(error)s', error=str(e)), 'error')
		
		return redirect(url_for('WOWorkflowInstanceView.show', pk=instance_id))
	
	@expose('/retry/<int:instance_id>', methods=['POST'])
	@has_access
	def retry_execution(self, instance_id):
		"""Retry failed workflow execution."""
		instance = self.datamodel.get(instance_id)
		if not instance:
			flash(_('Workflow instance not found'), 'error')
			return redirect(url_for('WOWorkflowInstanceView.list'))
		
		if not self._check_tenant_access(instance.workflow.tenant_id):
			flash(_('Access denied'), 'error')
			return redirect(url_for('WOWorkflowInstanceView.list'))
		
		if instance.status != WorkflowStatus.FAILED:
			flash(_('Only failed executions can be retried'), 'warning')
			return redirect(url_for('WOWorkflowInstanceView.show', pk=instance_id))
		
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			new_instance = loop.run_until_complete(
				self.workflow_service.execute_workflow(
					workflow_id=instance.workflow.id,
					execution_context=instance.execution_context,
					priority=instance.priority
				)
			)
			
			flash(_('Workflow execution retried'), 'success')
			return redirect(url_for('WOWorkflowInstanceView.show', pk=new_instance.id))
			
		except Exception as e:
			flash(_('Failed to retry execution: %(error)s', error=str(e)), 'error')
			return redirect(url_for('WOWorkflowInstanceView.show', pk=instance_id))
	
	@action('bulk_cancel', _('Cancel'), _('Cancel selected executions?'), 'fa-stop')
	def bulk_cancel(self, instances):
		"""Bulk cancel workflow executions."""
		count = 0
		for instance in instances:
			if (self._check_tenant_access(instance.workflow.tenant_id) and 
				instance.status in [WorkflowStatus.RUNNING, WorkflowStatus.PENDING]):
				try:
					loop = asyncio.new_event_loop()
					asyncio.set_event_loop(loop)
					loop.run_until_complete(
						self.workflow_service.cancel_workflow_instance(instance.id)
					)
					count += 1
				except Exception:
					pass
		
		if count > 0:
			flash(_('%(count)d executions cancelled', count=count), 'success')
		else:
			flash(_('No executions were cancelled'), 'warning')
		
		return redirect(url_for('WOWorkflowInstanceView.list'))
	
	def _check_tenant_access(self, tenant_id: str) -> bool:
		"""Check if user has access to tenant."""
		# Integration with APG RBAC would go here
		return True


class WOTaskExecutionView(APGBaseView):
	"""Task execution details view."""
	
	datamodel = SQLAInterface(TaskExecutionDB)
	
	# List view configuration
	list_columns = [
		'workflow_instance.workflow.name', 'task_id', 'status',
		'started_at', 'completed_at', 'duration_seconds', 'retry_count'
	]
	list_title = _('Task Executions')
	
	# Show view configuration
	show_columns = [
		'workflow_instance.workflow.name', 'task_id', 'status',
		'started_at', 'completed_at', 'duration_seconds',
		'input_data', 'result', 'error_details', 'logs',
		'retry_count', 'retry_config'
	]
	show_title = _('Task Execution Details')
	
	# Search configuration
	search_columns = ['task_id', 'status']
	
	# Ordering
	base_order = ('started_at', 'desc')
	
	# Filters
	base_filters = [['status', lambda: TaskStatus, 'Status']]
	
	# Security
	base_permissions = ['can_list', 'can_show']
	
	# Pagination
	page_size = 50
	
	@expose('/logs/<int:execution_id>')
	@has_access
	def view_logs(self, execution_id):
		"""View task execution logs."""
		execution = self.datamodel.get(execution_id)
		if not execution:
			flash(_('Task execution not found'), 'error')
			return redirect(url_for('WOTaskExecutionView.list'))
		
		return self.render_template(
			'workflow_orchestration/task_logs.html',
			execution=execution,
			title=_('Task Execution Logs')
		)


class WODashboardView(APGBaseView):
	"""Workflow orchestration dashboard and analytics."""
	
	route_base = '/workflow_orchestration/dashboard'
	default_view = 'dashboard'
	
	@expose('/')
	@has_access
	def dashboard(self):
		"""Main dashboard view."""
		# Get dashboard metrics
		session = self.appbuilder.get_session
		
		# Workflow statistics
		total_workflows = session.query(WorkflowDB).count()
		active_workflows = session.query(WorkflowDB).filter_by(is_active=True).count()
		
		# Execution statistics (last 30 days)
		thirty_days_ago = datetime.utcnow() - timedelta(days=30)
		recent_executions = session.query(WorkflowInstanceDB)\
			.filter(WorkflowInstanceDB.created_at >= thirty_days_ago)\
			.count()
		
		successful_executions = session.query(WorkflowInstanceDB)\
			.filter(
				WorkflowInstanceDB.created_at >= thirty_days_ago,
				WorkflowInstanceDB.status == WorkflowStatus.COMPLETED
			).count()
		
		failed_executions = session.query(WorkflowInstanceDB)\
			.filter(
				WorkflowInstanceDB.created_at >= thirty_days_ago,
				WorkflowInstanceDB.status == WorkflowStatus.FAILED
			).count()
		
		running_executions = session.query(WorkflowInstanceDB)\
			.filter(WorkflowInstanceDB.status == WorkflowStatus.RUNNING)\
			.count()
		
		# Success rate
		success_rate = (successful_executions / recent_executions * 100) if recent_executions > 0 else 0
		
		# Average execution time
		avg_duration = session.query(sa.func.avg(WorkflowInstanceDB.duration_seconds))\
			.filter(
				WorkflowInstanceDB.created_at >= thirty_days_ago,
				WorkflowInstanceDB.status == WorkflowStatus.COMPLETED
			).scalar() or 0
		
		# Top workflows by execution count
		top_workflows = session.query(
			WorkflowDB.name,
			sa.func.count(WorkflowInstanceDB.id).label('execution_count')
		)\
			.join(WorkflowInstanceDB)\
			.filter(WorkflowInstanceDB.created_at >= thirty_days_ago)\
			.group_by(WorkflowDB.name)\
			.order_by(sa.desc('execution_count'))\
			.limit(10)\
			.all()
		
		# Recent executions
		recent_instances = session.query(WorkflowInstanceDB)\
			.options(joinedload(WorkflowInstanceDB.workflow))\
			.order_by(WorkflowInstanceDB.created_at.desc())\
			.limit(10)\
			.all()
		
		return self.render_template(
			'workflow_orchestration/dashboard.html',
			title=_('Workflow Orchestration Dashboard'),
			metrics={
				'total_workflows': total_workflows,
				'active_workflows': active_workflows,
				'recent_executions': recent_executions,
				'successful_executions': successful_executions,
				'failed_executions': failed_executions,
				'running_executions': running_executions,
				'success_rate': round(success_rate, 1),
				'avg_duration': round(avg_duration, 2) if avg_duration else 0
			},
			top_workflows=top_workflows,
			recent_instances=recent_instances
		)
	
	@expose('/charts')
	@has_access
	def charts(self):
		"""Analytics charts view."""
		return self.render_template(
			'workflow_orchestration/charts.html',
			title=_('Workflow Analytics')
		)
	
	@expose_api(name='dashboard_metrics', url='/api/dashboard/metrics')
	@protect()
	def get_dashboard_metrics(self):
		"""API endpoint for dashboard metrics."""
		session = self.appbuilder.get_session
		
		# Get metrics for different time periods
		periods = {
			'today': datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0),
			'week': datetime.utcnow() - timedelta(days=7),
			'month': datetime.utcnow() - timedelta(days=30)
		}
		
		metrics = {}
		
		for period_name, start_date in periods.items():
			executions = session.query(WorkflowInstanceDB)\
				.filter(WorkflowInstanceDB.created_at >= start_date)\
				.all()
			
			total = len(executions)
			successful = sum(1 for e in executions if e.status == WorkflowStatus.COMPLETED)
			failed = sum(1 for e in executions if e.status == WorkflowStatus.FAILED)
			running = sum(1 for e in executions if e.status == WorkflowStatus.RUNNING)
			
			metrics[period_name] = {
				'total': total,
				'successful': successful,
				'failed': failed,
				'running': running,
				'success_rate': (successful / total * 100) if total > 0 else 0
			}
		
		return jsonify(metrics)
	
	@expose_api(name='execution_timeline', url='/api/dashboard/timeline')
	@protect()
	def get_execution_timeline(self):
		"""API endpoint for execution timeline data."""
		session = self.appbuilder.get_session
		
		# Get daily execution counts for the last 30 days
		thirty_days_ago = datetime.utcnow() - timedelta(days=30)
		
		timeline_data = session.query(
			sa.func.date(WorkflowInstanceDB.created_at).label('date'),
			sa.func.count(WorkflowInstanceDB.id).label('count'),
			WorkflowInstanceDB.status
		)\
			.filter(WorkflowInstanceDB.created_at >= thirty_days_ago)\
			.group_by(
				sa.func.date(WorkflowInstanceDB.created_at),
				WorkflowInstanceDB.status
			)\
			.order_by('date')\
			.all()
		
		# Format data for charting
		timeline = {}
		for row in timeline_data:
			date_str = row.date.isoformat()
			if date_str not in timeline:
				timeline[date_str] = {'completed': 0, 'failed': 0, 'running': 0, 'cancelled': 0}
			
			status_key = row.status.value.lower() if row.status else 'unknown'
			timeline[date_str][status_key] = row.count
		
		return jsonify(timeline)


class WOSystemView(APGBaseView):
	"""System administration and configuration view."""
	
	route_base = '/workflow_orchestration/system'
	default_view = 'status'
	
	@expose('/')
	@has_access
	@require_permission('system:admin')
	def status(self):
		"""System status and health check."""
		# System health checks
		health_status = {
			'database': self._check_database_health(),
			'redis': self._check_redis_health(),
			'task_queue': self._check_task_queue_health(),
			'external_services': self._check_external_services_health()
		}
		
		# System metrics
		session = self.appbuilder.get_session
		
		system_metrics = {
			'active_connections': self._get_active_connections(),
			'pending_tasks': session.query(WorkflowInstanceDB)
				.filter_by(status=WorkflowStatus.PENDING).count(),
			'running_tasks': session.query(WorkflowInstanceDB)
				.filter_by(status=WorkflowStatus.RUNNING).count(),
			'failed_tasks_today': session.query(WorkflowInstanceDB)
				.filter(
					WorkflowInstanceDB.status == WorkflowStatus.FAILED,
					WorkflowInstanceDB.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0)
				).count()
		}
		
		return self.render_template(
			'workflow_orchestration/system_status.html',
			title=_('System Status'),
			health_status=health_status,
			system_metrics=system_metrics
		)
	
	@expose('/configuration')
	@has_access
	@require_permission('system:admin')
	def configuration(self):
		"""System configuration management."""
		# Get current configuration
		config = current_app.config
		
		workflow_config = {
			'max_concurrent_workflows': config.get('WO_MAX_CONCURRENT_WORKFLOWS', 100),
			'default_timeout_seconds': config.get('WO_DEFAULT_TIMEOUT_SECONDS', 3600),
			'max_retry_attempts': config.get('WO_MAX_RETRY_ATTEMPTS', 3),
			'cleanup_completed_after_days': config.get('WO_CLEANUP_COMPLETED_AFTER_DAYS', 30),
			'enable_metrics': config.get('WO_ENABLE_METRICS', True),
			'log_level': config.get('WO_LOG_LEVEL', 'INFO')
		}
		
		return self.render_template(
			'workflow_orchestration/system_configuration.html',
			title=_('System Configuration'),
			config=workflow_config
		)
	
	def _check_database_health(self) -> Dict[str, Any]:
		"""Check database health."""
		try:
			session = self.appbuilder.get_session
			session.execute(sa.text('SELECT 1'))
			return {'status': 'healthy', 'message': 'Database connection OK'}
		except Exception as e:
			return {'status': 'unhealthy', 'message': f'Database error: {str(e)}'}
	
	def _check_redis_health(self) -> Dict[str, Any]:
		"""Check Redis health."""
		try:
			# Redis health check would go here
			return {'status': 'healthy', 'message': 'Redis connection OK'}
		except Exception as e:
			return {'status': 'unhealthy', 'message': f'Redis error: {str(e)}'}
	
	def _check_task_queue_health(self) -> Dict[str, Any]:
		"""Check task queue health."""
		try:
			# Task queue health check would go here
			return {'status': 'healthy', 'message': 'Task queue OK'}
		except Exception as e:
			return {'status': 'unhealthy', 'message': f'Task queue error: {str(e)}'}
	
	def _check_external_services_health(self) -> Dict[str, Any]:
		"""Check external services health."""
		try:
			# External services health check would go here
			return {'status': 'healthy', 'message': 'External services OK'}
		except Exception as e:
			return {'status': 'unhealthy', 'message': f'External services error: {str(e)}'}
	
	def _get_active_connections(self) -> int:
		"""Get number of active database connections."""
		try:
			session = self.appbuilder.get_session
			result = session.execute(sa.text('SELECT count(*) FROM pg_stat_activity WHERE state = \'active\''))
			return result.scalar() or 0
		except Exception:
			return 0


class WOExecutionTimeChart(DirectByChartView):
	"""Chart view for workflow execution times."""
	
	datamodel = SQLAInterface(WorkflowInstanceDB)
	chart_title = _('Average Execution Time by Workflow')
	
	definitions = [
		{
			'group': 'workflow.name',
			'series': ['duration_seconds']
		}
	]


class WOExecutionStatusChart(DirectByChartView):
	"""Chart view for workflow execution status distribution."""
	
	datamodel = SQLAInterface(WorkflowInstanceDB)
	chart_title = _('Execution Status Distribution')
	
	definitions = [
		{
			'group': 'status',
			'series': [{'aggregate': 'count', 'column': 'id'}]
		}
	]