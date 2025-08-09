"""
APG Configuration Management Blueprint - Flask-AppBuilder Web Interface
Provides comprehensive web UI for configuration management with GitOps workflows.
"""

from flask import request, jsonify, render_template, flash, redirect, url_for
from flask_appbuilder import ModelView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import ChartView, TimeChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget, EditWidget
from flask_appbuilder.forms import DynamicForm
from wtforms import StringField, SelectField, TextAreaField, BooleanField, IntegerField
from wtforms.validators import DataRequired, Length, Optional
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from ..models import (
	CMResource, 
	ConfigurationDSL, 
	ResourceType, 
	CloudProvider,
	CMCollaboration,
	CMDeployment,
	CMGitOpsRepository,
	CMGitOpsManifest
)
from ..service import get_config_manager


class ConfigurationForm(DynamicForm):
	"""Dynamic form for configuration creation and editing"""
	
	name = StringField(
		'Configuration Name',
		validators=[DataRequired(), Length(min=3, max=100)],
		description="Unique name for this configuration"
	)
	
	resource_type = SelectField(
		'Resource Type',
		choices=[
			('container', 'Container/Microservice'),
			('virtual_machine', 'Virtual Machine'),
			('database', 'Database Service'),
			('storage', 'Storage System'),
			('network', 'Network Component'),
			('security', 'Security Service'),
			('monitoring', 'Monitoring/Observability')
		],
		validators=[DataRequired()],
		description="Type of infrastructure resource"
	)
	
	cloud_provider = SelectField(
		'Cloud Provider',
		choices=[
			('aws', 'Amazon Web Services'),
			('azure', 'Microsoft Azure'),
			('gcp', 'Google Cloud Platform'),
			('multi_cloud', 'Multi-Cloud Deployment'),
			('on_premise', 'On-Premise Infrastructure')
		],
		validators=[DataRequired()],
		description="Target cloud provider or deployment type"
	)
	
	configuration_spec = TextAreaField(
		'Configuration Specification',
		validators=[DataRequired()],
		description="JSON specification for the resource configuration",
		render_kw={
			'rows': 20,
			'placeholder': '{\n  "kind": "WebApplication",\n  "spec": {\n    "resources": {\n      "cpu": "2",\n      "memory": "4Gi"\n    },\n    "replicas": 3\n  }\n}'
		}
	)
	
	description = TextAreaField(
		'Description',
		validators=[Optional(), Length(max=500)],
		description="Optional description of this configuration"
	)
	
	security_level = SelectField(
		'Security Level',
		choices=[
			('public', 'Public'),
			('internal', 'Internal'),
			('confidential', 'Confidential'),
			('restricted', 'Restricted')
		],
		default='internal',
		validators=[DataRequired()],
		description="Security classification for this configuration"
	)


class GitOpsRepositoryForm(DynamicForm):
	"""Form for GitOps repository configuration"""
	
	name = StringField(
		'Repository Name',
		validators=[DataRequired(), Length(min=3, max=100)],
		description="Name for this GitOps repository"
	)
	
	url = StringField(
		'Repository URL',
		validators=[DataRequired()],
		description="Git repository URL (e.g., https://github.com/org/repo.git)"
	)
	
	branch = StringField(
		'Default Branch',
		default='main',
		validators=[DataRequired()],
		description="Default branch for GitOps operations"
	)
	
	sync_enabled = BooleanField(
		'Auto Sync Enabled',
		default=True,
		description="Enable automatic synchronization with repository"
	)
	
	sync_interval = IntegerField(
		'Sync Interval (seconds)',
		default=300,
		validators=[Optional()],
		description="Interval between automatic syncs"
	)


class ConfigurationView(ModelView):
	"""Main view for configuration management"""
	
	datamodel = SQLAInterface(CMResource)
	
	# List view configuration
	list_columns = ['name', 'resource_type', 'cloud_provider', 'state', 'created_at', 'health_score']
	search_columns = ['name', 'resource_type', 'cloud_provider', 'description']
	show_columns = ['name', 'resource_type', 'cloud_provider', 'configuration', 'state', 
					'health_score', 'created_at', 'last_modified', 'description']
	edit_columns = ['name', 'resource_type', 'cloud_provider', 'configuration', 'description']
	add_columns = ['name', 'resource_type', 'cloud_provider', 'configuration', 'description']
	
	# Formatters for better display
	formatters_columns = {
		'configuration': lambda x: json.dumps(x, indent=2) if isinstance(x, dict) else str(x),
		'health_score': lambda x: f"{x:.1f}%" if x else "N/A",
		'state': lambda x: f"<span class='label label-{ConfigurationView._state_label_class(x)}'>{x}</span>"
	}
	
	# Custom widgets
	list_widget = ListWidget
	show_widget = ShowWidget
	edit_widget = EditWidget
	
	@staticmethod
	def _state_label_class(state):
		"""Get CSS class for state labels"""
		state_classes = {
			'active': 'success',
			'pending': 'warning',
			'failed': 'danger',
			'deploying': 'info',
			'rollback': 'warning'
		}
		return state_classes.get(state, 'default')
	
	@expose('/create_from_natural_language/')
	@has_access
	def create_from_natural_language(self):
		"""Create configuration from natural language description"""
		if request.method == 'POST':
			nl_request = request.form.get('natural_language_request', '')
			context = {
				'environment': request.form.get('environment', 'production'),
				'team': request.form.get('team', 'platform'),
				'compliance_level': request.form.get('compliance_level', 'standard')
			}
			
			try:
				# Process natural language request
				config_manager = get_config_manager()
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				
				result = loop.run_until_complete(
					config_manager.process_natural_language_request(nl_request, context)
				)
				
				if result.get('success'):
					# Create configuration from AI result
					config_data = {
						'name': result.get('suggested_name', 'ai-generated-config'),
						'type': result.get('resource_type', 'container'),
						'cloud_provider': result.get('cloud_provider', 'aws'),
						'configuration': result.get('generated_configuration'),
						'description': f"AI-generated from: {nl_request}",
						'created_by': 'ai-assistant',
						'security_level': context.get('compliance_level', 'internal')
					}
					
					resource_id = loop.run_until_complete(
						config_manager.create_configuration(config_data)
					)
					
					flash(f'Configuration created successfully from natural language! ID: {resource_id}', 'success')
					return redirect(url_for('ConfigurationView.show', pk=resource_id))
				else:
					flash('Failed to process natural language request', 'danger')
					
			except Exception as e:
				flash(f'Error processing request: {str(e)}', 'danger')
			finally:
				loop.close()
		
		return self.render_template(
			'appbuilder/general/model/create_from_nl.html',
			title="Create Configuration from Natural Language"
		)
	
	@expose('/optimize/<pk>')
	@has_access
	def optimize_configuration(self, pk):
		"""Optimize existing configuration using AI"""
		try:
			config_manager = get_config_manager()
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			optimization_result = loop.run_until_complete(
				config_manager.optimize_configuration(pk)
			)
			
			if optimization_result.get('success'):
				flash('Configuration optimization completed!', 'success')
				return jsonify({
					'success': True,
					'optimizations': optimization_result.get('optimizations', []),
					'health_score_improvement': optimization_result.get('health_score_improvement', 0)
				})
			else:
				flash('Optimization failed', 'danger')
				return jsonify({'success': False, 'error': 'Optimization failed'})
				
		except Exception as e:
			flash(f'Optimization error: {str(e)}', 'danger')
			return jsonify({'success': False, 'error': str(e)})
		finally:
			if 'loop' in locals():
				loop.close()
	
	@expose('/deploy/<pk>')
	@has_access
	def deploy_configuration(self, pk):
		"""Deploy configuration using GitOps workflow"""
		try:
			config_manager = get_config_manager()
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			# Get available repositories for deployment
			repositories = loop.run_until_complete(
				config_manager.get_gitops_repositories()
			)
			
			if request.method == 'POST':
				repo_id = request.form.get('repository_id')
				environment = request.form.get('environment', 'production')
				namespace = request.form.get('namespace', 'default')
				
				deployment_result = loop.run_until_complete(
					config_manager.deploy_configuration_gitops(
						resource_id=pk,
						repository_id=repo_id,
						environment=environment,
						namespace=namespace
					)
				)
				
				if deployment_result.get('success'):
					flash('Configuration deployment initiated!', 'success')
					return redirect(url_for('ConfigurationView.show', pk=pk))
				else:
					flash('Deployment failed', 'danger')
			
			return self.render_template(
				'appbuilder/general/model/deploy_config.html',
				title="Deploy Configuration",
				repositories=repositories,
				config_id=pk
			)
			
		except Exception as e:
			flash(f'Deployment error: {str(e)}', 'danger')
			return redirect(url_for('ConfigurationView.show', pk=pk))
		finally:
			if 'loop' in locals():
				loop.close()


class GitOpsRepositoryView(ModelView):
	"""View for GitOps repository management"""
	
	datamodel = SQLAInterface(CMGitOpsRepository)
	
	list_columns = ['name', 'url', 'branch', 'sync_enabled', 'last_sync_at']
	search_columns = ['name', 'url', 'branch']
	show_columns = ['name', 'url', 'branch', 'sync_enabled', 'auto_sync_interval', 
					'last_sync_at', 'created_at']
	edit_columns = ['name', 'url', 'branch', 'sync_enabled', 'auto_sync_interval']
	add_columns = ['name', 'url', 'branch', 'sync_enabled', 'auto_sync_interval']
	
	@expose('/sync/<pk>')
	@has_access
	def sync_repository(self, pk):
		"""Manually trigger repository sync"""
		try:
			config_manager = get_config_manager()
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			sync_result = loop.run_until_complete(
				config_manager.sync_gitops_repository(pk)
			)
			
			if sync_result.get('success'):
				flash('Repository sync completed successfully!', 'success')
			else:
				flash('Repository sync failed', 'danger')
				
		except Exception as e:
			flash(f'Sync error: {str(e)}', 'danger')
		finally:
			if 'loop' in locals():
				loop.close()
		
		return redirect(url_for('GitOpsRepositoryView.show', pk=pk))


class GitOpsManifestView(ModelView):
	"""View for GitOps manifest management"""
	
	datamodel = SQLAInterface(CMGitOpsManifest)
	
	list_columns = ['resource_name', 'repository_name', 'environment', 'namespace', 'created_at']
	search_columns = ['resource_name', 'repository_name', 'environment', 'namespace']
	show_columns = ['resource_name', 'repository_name', 'environment', 'namespace',
					'file_path', 'content', 'created_at']
	
	formatters_columns = {
		'content': lambda x: f"<pre>{json.dumps(x, indent=2)}</pre>" if isinstance(x, dict) else str(x)
	}


class DeploymentView(ModelView):
	"""View for deployment management and monitoring"""
	
	datamodel = SQLAInterface(CMDeployment)
	
	list_columns = ['resource_name', 'environment', 'strategy', 'status', 'started_at', 'progress_percentage']
	search_columns = ['resource_name', 'environment', 'strategy', 'status']
	show_columns = ['resource_name', 'environment', 'strategy', 'status', 'started_at',
					'completed_at', 'progress_percentage', 'health_checks', 'rollback_triggered']
	
	formatters_columns = {
		'progress_percentage': lambda x: f"{x:.1f}%" if x else "0%",
		'status': lambda x: f"<span class='label label-{DeploymentView._status_label_class(x)}'>{x}</span>",
		'health_checks': lambda x: json.dumps(x, indent=2) if isinstance(x, dict) else str(x)
	}
	
	@staticmethod
	def _status_label_class(status):
		"""Get CSS class for status labels"""
		status_classes = {
			'success': 'success',
			'running': 'info',
			'failed': 'danger',
			'pending': 'warning',
			'cancelled': 'default'
		}
		return status_classes.get(status, 'default')
	
	@expose('/rollback/<pk>')
	@has_access
	def rollback_deployment(self, pk):
		"""Trigger deployment rollback"""
		try:
			config_manager = get_config_manager()
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			rollback_result = loop.run_until_complete(
				config_manager.trigger_deployment_rollback(
					deployment_id=pk,
					reason="Manual rollback triggered from web interface"
				)
			)
			
			if rollback_result.get('success'):
				flash('Deployment rollback initiated!', 'success')
			else:
				flash('Rollback failed', 'danger')
				
		except Exception as e:
			flash(f'Rollback error: {str(e)}', 'danger')
		finally:
			if 'loop' in locals():
				loop.close()
		
		return redirect(url_for('DeploymentView.show', pk=pk))


class CollaborationView(ModelView):
	"""View for collaboration session management"""
	
	datamodel = SQLAInterface(CMCollaboration)
	
	list_columns = ['resource_name', 'owner_id', 'session_name', 'status', 'created_at', 'participant_count']
	search_columns = ['resource_name', 'owner_id', 'session_name']
	show_columns = ['resource_name', 'owner_id', 'session_name', 'status', 'created_at',
					'participant_count', 'user_permissions', 'change_log']
	
	formatters_columns = {
		'user_permissions': lambda x: json.dumps(x, indent=2) if isinstance(x, dict) else str(x),
		'change_log': lambda x: json.dumps(x[-5:], indent=2) if isinstance(x, list) and len(x) > 0 else "No changes"
	}


class SystemMetricsChartView(ChartView):
	"""Chart view for system metrics and performance"""
	
	chart_title = 'APG Configuration Management Metrics'
	label_columns = {'total_configurations': 'Total Configurations',
					 'autonomous_remediations': 'AI Remediations',
					 'deployment_success_rate': 'Deployment Success Rate'}
	
	@expose('/chart/')
	@has_access
	def chart(self):
		"""Display system metrics charts"""
		try:
			config_manager = get_config_manager()
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			metrics = loop.run_until_complete(
				config_manager.get_revolutionary_metrics()
			)
			
			chart_data = {
				'data': [
					{
						'label': 'Configurations',
						'value': metrics.get('system_metrics', {}).get('total_configurations', 0)
					},
					{
						'label': 'AI Remediations',
						'value': metrics.get('system_metrics', {}).get('autonomous_remediations', 0)
					},
					{
						'label': 'Deployments',
						'value': metrics.get('gitops_metrics', {}).get('deployment_success_rate', 0) * 1000
					}
				]
			}
			
			return self.render_template(
				'appbuilder/general/charts/chart.html',
				title=self.chart_title,
				chart_data=json.dumps(chart_data)
			)
			
		except Exception as e:
			flash(f'Metrics error: {str(e)}', 'danger')
			return self.render_template(
				'appbuilder/general/charts/chart.html',
				title=self.chart_title,
				chart_data=json.dumps({'data': []})
			)
		finally:
			if 'loop' in locals():
				loop.close()


class DashboardView(ModelView):
	"""Main dashboard view with system overview"""
	
	@expose('/dashboard/')
	@has_access
	def dashboard(self):
		"""Display main dashboard"""
		try:
			config_manager = get_config_manager()
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			# Get comprehensive system status
			metrics = loop.run_until_complete(config_manager.get_revolutionary_metrics())
			gitops_status = loop.run_until_complete(config_manager.get_gitops_status())
			
			dashboard_data = {
				'total_configurations': metrics.get('system_metrics', {}).get('total_configurations', 0),
				'autonomous_remediations': metrics.get('system_metrics', {}).get('autonomous_remediations', 0),
				'deployment_success_rate': metrics.get('gitops_metrics', {}).get('deployment_success_rate', 0),
				'gitops_repositories': gitops_status.get('repositories', 0),
				'active_deployments': gitops_status.get('active_deployments', 0),
				'system_health': metrics.get('performance_indicators', {}).get('autonomous_operations_percentage', 0)
			}
			
			return self.render_template(
				'dashboard.html',
				title='APG Configuration Management Dashboard',
				dashboard_data=dashboard_data
			)
			
		except Exception as e:
			flash(f'Dashboard error: {str(e)}', 'danger')
			return self.render_template(
				'dashboard.html',
				title='APG Configuration Management Dashboard',
				dashboard_data={}
			)
		finally:
			if 'loop' in locals():
				loop.close()


# Custom templates for enhanced UI
CUSTOM_TEMPLATES = {
	'create_from_nl.html': '''
	{% extends "appbuilder/base.html" %}
	
	{% block content %}
	<div class="container-fluid">
		<div class="row">
			<div class="col-md-12">
				<div class="panel panel-default">
					<div class="panel-heading">
						<h3 class="panel-title">Create Configuration from Natural Language</h3>
					</div>
					<div class="panel-body">
						<form method="POST">
							<div class="form-group">
								<label for="natural_language_request">Describe your infrastructure needs:</label>
								<textarea id="natural_language_request" name="natural_language_request" 
										  class="form-control" rows="4" 
										  placeholder="Example: Create a highly available web service with auto-scaling running nginx in AWS with 4GB memory"></textarea>
							</div>
							<div class="row">
								<div class="col-md-4">
									<div class="form-group">
										<label for="environment">Environment:</label>
										<select id="environment" name="environment" class="form-control">
											<option value="production">Production</option>
											<option value="staging">Staging</option>
											<option value="development">Development</option>
										</select>
									</div>
								</div>
								<div class="col-md-4">
									<div class="form-group">
										<label for="team">Team:</label>
										<select id="team" name="team" class="form-control">
											<option value="platform">Platform</option>
											<option value="application">Application</option>
											<option value="security">Security</option>
											<option value="data">Data</option>
										</select>
									</div>
								</div>
								<div class="col-md-4">
									<div class="form-group">
										<label for="compliance_level">Compliance Level:</label>
										<select id="compliance_level" name="compliance_level" class="form-control">
											<option value="standard">Standard</option>
											<option value="high">High</option>
											<option value="critical">Critical</option>
										</select>
									</div>
								</div>
							</div>
							<button type="submit" class="btn btn-primary">
								<i class="fa fa-magic"></i> Generate Configuration
							</button>
							<a href="{{ url_for('ConfigurationView.list') }}" class="btn btn-default">
								<i class="fa fa-arrow-left"></i> Back to Configurations
							</a>
						</form>
					</div>
				</div>
			</div>
		</div>
	</div>
	{% endblock %}
	''',
	
	'dashboard.html': '''
	{% extends "appbuilder/base.html" %}
	
	{% block content %}
	<div class="container-fluid">
		<div class="row">
			<div class="col-md-3 col-sm-6 col-xs-12">
				<div class="info-box">
					<span class="info-box-icon bg-aqua"><i class="fa fa-cogs"></i></span>
					<div class="info-box-content">
						<span class="info-box-text">Configurations</span>
						<span class="info-box-number">{{ dashboard_data.total_configurations or 0 }}</span>
					</div>
				</div>
			</div>
			
			<div class="col-md-3 col-sm-6 col-xs-12">
				<div class="info-box">
					<span class="info-box-icon bg-green"><i class="fa fa-rocket"></i></span>
					<div class="info-box-content">
						<span class="info-box-text">Deployments</span>
						<span class="info-box-number">{{ dashboard_data.active_deployments or 0 }}</span>
					</div>
				</div>
			</div>
			
			<div class="col-md-3 col-sm-6 col-xs-12">
				<div class="info-box">
					<span class="info-box-icon bg-yellow"><i class="fa fa-magic"></i></span>
					<div class="info-box-content">
						<span class="info-box-text">AI Remediations</span>
						<span class="info-box-number">{{ dashboard_data.autonomous_remediations or 0 }}</span>
					</div>
				</div>
			</div>
			
			<div class="col-md-3 col-sm-6 col-xs-12">
				<div class="info-box">
					<span class="info-box-icon bg-red"><i class="fa fa-heartbeat"></i></span>
					<div class="info-box-content">
						<span class="info-box-text">System Health</span>
						<span class="info-box-number">{{ "%.1f"|format(dashboard_data.system_health or 0) }}%</span>
					</div>
				</div>
			</div>
		</div>
		
		<div class="row">
			<div class="col-md-6">
				<div class="box box-primary">
					<div class="box-header with-border">
						<h3 class="box-title">GitOps Status</h3>
					</div>
					<div class="box-body">
						<p><strong>Repositories:</strong> {{ dashboard_data.gitops_repositories or 0 }}</p>
						<p><strong>Success Rate:</strong> {{ "%.1f"|format((dashboard_data.deployment_success_rate or 0) * 100) }}%</p>
						<p><strong>Active Deployments:</strong> {{ dashboard_data.active_deployments or 0 }}</p>
					</div>
				</div>
			</div>
			
			<div class="col-md-6">
				<div class="box box-success">
					<div class="box-header with-border">
						<h3 class="box-title">Revolutionary Capabilities</h3>
					</div>
					<div class="box-body">
						<p>✅ AI-Native Intelligence</p>
						<p>✅ Universal Cloud Abstraction</p>
						<p>✅ Zero-Trust Security</p>
						<p>✅ GitOps Excellence</p>
						<p>✅ Real-Time Collaboration</p>
					</div>
				</div>
			</div>
		</div>
	</div>
	{% endblock %}
	'''
}


# Export views for registration with Flask-AppBuilder
__all__ = [
	'ConfigurationView',
	'GitOpsRepositoryView', 
	'GitOpsManifestView',
	'DeploymentView',
	'CollaborationView',
	'SystemMetricsChartView',
	'DashboardView',
	'ConfigurationForm',
	'GitOpsRepositoryForm',
	'CUSTOM_TEMPLATES'
]