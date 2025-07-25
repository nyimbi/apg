"""
Distributed Computing Views

Flask-AppBuilder views for distributed computing cluster management,
job scheduling, resource monitoring, and workflow orchestration.
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
	DCCluster, DCNode, DCJob, DCJobExecution, DCWorkflow
)


class DistributedComputingBaseView(BaseView):
	"""Base view for distributed computing functionality"""
	
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
	
	def _format_duration(self, minutes: float) -> str:
		"""Format duration for display"""
		if minutes is None:
			return "N/A"
		
		if minutes < 60:
			return f"{minutes:.1f} min"
		elif minutes < 1440:  # 24 hours
			hours = minutes / 60
			return f"{hours:.1f} hrs"
		else:
			days = minutes / 1440
			return f"{days:.1f} days"
	
	def _format_cost(self, cost: float) -> str:
		"""Format cost for display"""
		if cost is None:
			return "N/A"
		return f"${cost:.2f}"


class DCClusterModelView(ModelView):
	"""Distributed computing cluster management view"""
	
	datamodel = SQLAInterface(DCCluster)
	
	# List view configuration
	list_columns = [
		'cluster_name', 'cluster_type', 'region', 'status',
		'health_status', 'total_cpu_cores', 'total_memory_gb', 'auto_scaling_enabled'
	]
	show_columns = [
		'cluster_id', 'cluster_name', 'description', 'cluster_type', 'cluster_version',
		'region', 'availability_zone', 'status', 'health_status', 'last_health_check',
		'total_cpu_cores', 'total_memory_gb', 'total_storage_gb', 'total_gpu_count',
		'used_cpu_cores', 'used_memory_gb', 'used_storage_gb', 'used_gpu_count',
		'auto_scaling_enabled', 'min_nodes', 'max_nodes', 'scale_up_threshold',
		'scale_down_threshold', 'hourly_cost', 'monthly_budget', 'access_endpoint'
	]
	edit_columns = [
		'cluster_name', 'description', 'cluster_type', 'cluster_version',
		'region', 'availability_zone', 'configuration', 'environment_variables',
		'auto_scaling_enabled', 'min_nodes', 'max_nodes', 'scale_up_threshold',
		'scale_down_threshold', 'hourly_cost', 'monthly_budget', 'cost_alerts_enabled'
	]
	add_columns = [
		'cluster_name', 'description', 'cluster_type', 'region',
		'auto_scaling_enabled', 'min_nodes', 'max_nodes'
	]
	
	# Search and filtering
	search_columns = ['cluster_name', 'cluster_type', 'region']
	base_filters = [['status', lambda: 'active', lambda: True]]
	
	# Ordering
	base_order = ('cluster_name', 'asc')
	
	# Form validation
	validators_columns = {
		'cluster_name': [DataRequired(), Length(min=3, max=200)],
		'cluster_type': [DataRequired()],
		'min_nodes': [NumberRange(min=1)],
		'max_nodes': [NumberRange(min=1)],
		'scale_up_threshold': [NumberRange(min=0.1, max=1.0)],
		'scale_down_threshold': [NumberRange(min=0.1, max=1.0)],
		'hourly_cost': [NumberRange(min=0)]
	}
	
	# Custom labels
	label_columns = {
		'cluster_id': 'Cluster ID',
		'cluster_name': 'Cluster Name',
		'cluster_type': 'Cluster Type',
		'cluster_version': 'Version',
		'availability_zone': 'Availability Zone',
		'health_status': 'Health Status',
		'last_health_check': 'Last Health Check',
		'total_cpu_cores': 'Total CPU Cores',
		'total_memory_gb': 'Total Memory (GB)',
		'total_storage_gb': 'Total Storage (GB)',
		'total_gpu_count': 'Total GPUs',
		'used_cpu_cores': 'Used CPU Cores',
		'used_memory_gb': 'Used Memory (GB)',
		'used_storage_gb': 'Used Storage (GB)',
		'used_gpu_count': 'Used GPUs',
		'auto_scaling_enabled': 'Auto Scaling',
		'min_nodes': 'Min Nodes',
		'max_nodes': 'Max Nodes',
		'scale_up_threshold': 'Scale Up Threshold',
		'scale_down_threshold': 'Scale Down Threshold',
		'hourly_cost': 'Hourly Cost',
		'monthly_budget': 'Monthly Budget',
		'cost_alerts_enabled': 'Cost Alerts',
		'access_endpoint': 'Access Endpoint',
		'authentication_method': 'Auth Method',
		'configuration': 'Configuration',
		'environment_variables': 'Environment Variables'
	}
	
	@expose('/cluster_dashboard/<int:pk>')
	@has_access
	def cluster_dashboard(self, pk):
		"""Cluster monitoring dashboard"""
		cluster = self.datamodel.get(pk)
		if not cluster:
			flash('Cluster not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			dashboard_data = self._get_cluster_dashboard_data(cluster)
			
			return render_template('distributed_computing/cluster_dashboard.html',
								   cluster=cluster,
								   dashboard_data=dashboard_data,
								   page_title=f"Cluster Dashboard: {cluster.cluster_name}")
		except Exception as e:
			flash(f'Error loading cluster dashboard: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/scale_cluster/<int:pk>')
	@has_access
	def scale_cluster(self, pk):
		"""Scale cluster up or down"""
		cluster = self.datamodel.get(pk)
		if not cluster:
			flash('Cluster not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			action = request.args.get('action', 'auto')
			scaling_needed = cluster.needs_scaling()
			
			if action == 'auto':
				if scaling_needed != 'none':
					# Implementation would trigger actual scaling
					flash(f'Cluster scaling {scaling_needed} initiated', 'success')
				else:
					flash('No scaling needed', 'info')
			elif action in ['up', 'down']:
				# Manual scaling
				flash(f'Manual scaling {action} initiated', 'success')
			
		except Exception as e:
			flash(f'Error scaling cluster: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/health_check/<int:pk>')
	@has_access
	def health_check(self, pk):
		"""Perform cluster health check"""
		cluster = self.datamodel.get(pk)
		if not cluster:
			flash('Cluster not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Simulate health check
			cluster.last_health_check = datetime.utcnow()
			cluster.health_status = 'healthy'  # Would be determined by actual check
			self.datamodel.edit(cluster)
			flash('Health check completed successfully', 'success')
		except Exception as e:
			flash(f'Error performing health check: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new cluster"""
		item.tenant_id = self._get_tenant_id()
		
		# Set default values
		if not item.status:
			item.status = 'active'
		if not item.health_status:
			item.health_status = 'healthy'
		if not item.authentication_method:
			item.authentication_method = 'token'
	
	def _get_cluster_dashboard_data(self, cluster: DCCluster) -> Dict[str, Any]:
		"""Get dashboard data for cluster"""
		utilization = cluster.calculate_utilization()
		
		return {
			'utilization': utilization,
			'node_count': len(cluster.nodes),
			'active_jobs': len([job for job in cluster.jobs if job.status == 'running']),
			'queued_jobs': len([job for job in cluster.jobs if job.status == 'queued']),
			'scaling_recommendation': cluster.needs_scaling(),
			'recent_activity': [],
			'performance_metrics': {
				'avg_job_completion_time': 0.0,
				'job_success_rate': 95.5,
				'resource_efficiency': 87.2
			}
		}
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class DCNodeModelView(ModelView):
	"""Distributed computing node management view"""
	
	datamodel = SQLAInterface(DCNode)
	
	# List view configuration
	list_columns = [
		'node_name', 'cluster', 'node_type', 'status', 'health_status',
		'cpu_cores', 'memory_gb', 'current_job_count', 'cpu_usage_percent'
	]
	show_columns = [
		'node_id', 'node_name', 'cluster', 'hostname', 'ip_address',
		'node_type', 'instance_type', 'availability_zone', 'cpu_cores',
		'cpu_architecture', 'memory_gb', 'storage_gb', 'gpu_count', 'gpu_type',
		'status', 'health_status', 'last_heartbeat', 'uptime_hours',
		'current_job_count', 'max_concurrent_jobs', 'total_jobs_completed',
		'cpu_usage_percent', 'memory_usage_gb', 'gpu_usage_percent',
		'average_cpu_usage', 'job_success_rate', 'hourly_cost'
	]
	edit_columns = [
		'node_name', 'node_type', 'max_concurrent_jobs', 'status',
		'environment_labels'
	]
	add_columns = [
		'node_name', 'hostname', 'node_type', 'cpu_cores', 'memory_gb',
		'storage_gb', 'gpu_count', 'gpu_type'
	]
	
	# Search and filtering
	search_columns = ['node_name', 'hostname', 'ip_address', 'node_type']
	base_filters = [['status', lambda: 'active', lambda: True]]
	
	# Ordering
	base_order = ('node_name', 'asc')
	
	# Form validation
	validators_columns = {
		'node_name': [DataRequired(), Length(min=3, max=200)],
		'cpu_cores': [DataRequired(), NumberRange(min=1)],
		'memory_gb': [DataRequired(), NumberRange(min=1)],
		'storage_gb': [DataRequired(), NumberRange(min=1)],
		'max_concurrent_jobs': [NumberRange(min=1)]
	}
	
	# Custom labels
	label_columns = {
		'node_id': 'Node ID',
		'node_name': 'Node Name',
		'ip_address': 'IP Address',
		'external_ip': 'External IP',
		'node_type': 'Node Type',
		'instance_type': 'Instance Type',
		'availability_zone': 'Availability Zone',
		'cpu_cores': 'CPU Cores',
		'cpu_architecture': 'CPU Architecture',
		'memory_gb': 'Memory (GB)',
		'storage_gb': 'Storage (GB)',
		'gpu_count': 'GPU Count',
		'gpu_type': 'GPU Type',
		'network_bandwidth_gbps': 'Network Bandwidth (Gbps)',
		'health_status': 'Health Status',
		'last_heartbeat': 'Last Heartbeat',
		'uptime_hours': 'Uptime (Hours)',
		'operating_system': 'Operating System',
		'container_runtime': 'Container Runtime',
		'kubernetes_version': 'Kubernetes Version',
		'assigned_jobs': 'Assigned Jobs',
		'max_concurrent_jobs': 'Max Concurrent Jobs',
		'current_job_count': 'Current Jobs',
		'total_jobs_completed': 'Total Jobs Completed',
		'cpu_usage_percent': 'CPU Usage (%)',
		'memory_usage_gb': 'Memory Usage (GB)',
		'gpu_usage_percent': 'GPU Usage (%)',
		'average_cpu_usage': 'Average CPU Usage',
		'job_success_rate': 'Job Success Rate (%)',
		'average_job_duration': 'Average Job Duration',
		'hourly_cost': 'Hourly Cost',
		'environment_labels': 'Environment Labels'
	}
	
	@expose('/node_metrics/<int:pk>')
	@has_access
	def node_metrics(self, pk):
		"""View detailed node metrics"""
		node = self.datamodel.get(pk)
		if not node:
			flash('Node not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			metrics_data = self._get_node_metrics(node)
			
			return render_template('distributed_computing/node_metrics.html',
								   node=node,
								   metrics_data=metrics_data,
								   page_title=f"Node Metrics: {node.node_name}")
		except Exception as e:
			flash(f'Error loading node metrics: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/drain_node/<int:pk>')
	@has_access
	def drain_node(self, pk):
		"""Drain node for maintenance"""
		node = self.datamodel.get(pk)
		if not node:
			flash('Node not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			node.status = 'draining'
			self.datamodel.edit(node)
			flash(f'Node {node.node_name} is being drained', 'success')
		except Exception as e:
			flash(f'Error draining node: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/restart_node/<int:pk>')
	@has_access
	def restart_node(self, pk):
		"""Restart node"""
		node = self.datamodel.get(pk)
		if not node:
			flash('Node not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would trigger actual node restart
			flash(f'Restart initiated for node {node.node_name}', 'success')
		except Exception as e:
			flash(f'Error restarting node: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new node"""
		item.tenant_id = self._get_tenant_id()
		
		# Set default values
		if not item.status:
			item.status = 'active'
		if not item.health_status:
			item.health_status = 'healthy'
		if not item.cpu_architecture:
			item.cpu_architecture = 'x86_64'
		if not item.max_concurrent_jobs:
			item.max_concurrent_jobs = 4
	
	def _get_node_metrics(self, node: DCNode) -> Dict[str, Any]:
		"""Get detailed metrics for node"""
		available_resources = node.get_available_resources()
		
		return {
			'current_utilization': {
				'cpu': node.cpu_usage_percent,
				'memory': (node.memory_usage_gb / node.memory_gb * 100) if node.memory_gb > 0 else 0,
				'storage': (node.storage_usage_gb / node.storage_gb * 100) if node.storage_gb > 0 else 0,
				'gpu': node.gpu_usage_percent
			},
			'available_resources': available_resources,
			'job_statistics': {
				'current_jobs': node.current_job_count,
				'completed_jobs': node.total_jobs_completed,
				'success_rate': node.job_success_rate,
				'average_duration': node.average_job_duration
			},
			'performance_trends': [],
			'health_indicators': {
				'last_heartbeat': node.last_heartbeat,
				'uptime_hours': node.uptime_hours,
				'health_status': node.health_status
			}
		}
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class DCJobModelView(ModelView):
	"""Distributed computing job management view"""
	
	datamodel = SQLAInterface(DCJob)
	
	# List view configuration
	list_columns = [
		'job_name', 'cluster', 'job_type', 'status', 'priority',
		'submitted_at', 'progress_percentage', 'required_cpu_cores', 'required_memory_gb'
	]
	show_columns = [
		'job_id', 'job_name', 'description', 'cluster', 'job_type', 'simulation_type',
		'command', 'arguments', 'status', 'progress_percentage', 'priority',
		'required_cpu_cores', 'required_memory_gb', 'required_storage_gb',
		'required_gpu_count', 'max_execution_time_minutes', 'retry_count', 'max_retries',
		'submitted_at', 'queued_at', 'started_at', 'completed_at',
		'execution_time_minutes', 'queue_time_minutes', 'submitted_by'
	]
	edit_columns = [
		'job_name', 'description', 'command', 'arguments', 'environment_variables',
		'required_cpu_cores', 'required_memory_gb', 'required_storage_gb',
		'required_gpu_count', 'max_execution_time_minutes', 'max_retries',
		'priority', 'tags'
	]
	add_columns = [
		'job_name', 'description', 'job_type', 'command', 'arguments',
		'required_cpu_cores', 'required_memory_gb', 'required_storage_gb',
		'priority'
	]
	
	# Search and filtering
	search_columns = ['job_name', 'job_type', 'status', 'submitted_by']
	base_filters = [['status', lambda: 'running', lambda: True]]
	
	# Ordering
	base_order = ('submitted_at', 'desc')
	
	# Form validation
	validators_columns = {
		'job_name': [DataRequired(), Length(min=3, max=200)],
		'command': [DataRequired()],
		'required_cpu_cores': [DataRequired(), NumberRange(min=1)],
		'required_memory_gb': [DataRequired(), NumberRange(min=0.1)],
		'required_storage_gb': [DataRequired(), NumberRange(min=0.1)],
		'priority': [NumberRange(min=1, max=10)],
		'max_execution_time_minutes': [NumberRange(min=1)]
	}
	
	# Custom labels
	label_columns = {
		'job_id': 'Job ID',
		'job_name': 'Job Name',
		'job_type': 'Job Type',
		'simulation_type': 'Simulation Type',
		'environment_variables': 'Environment Variables',
		'working_directory': 'Working Directory',
		'required_cpu_cores': 'Required CPU Cores',
		'required_memory_gb': 'Required Memory (GB)',
		'required_storage_gb': 'Required Storage (GB)',
		'required_gpu_count': 'Required GPUs',
		'required_gpu_type': 'Required GPU Type',
		'max_execution_time_minutes': 'Max Execution Time (min)',
		'retry_count': 'Retry Count',
		'max_retries': 'Max Retries',
		'parallelizable': 'Parallelizable',
		'max_parallel_tasks': 'Max Parallel Tasks',
		'depends_on_jobs': 'Depends On Jobs',
		'blocks_jobs': 'Blocks Jobs',
		'input_files': 'Input Files',
		'output_files': 'Output Files',
		'input_data_size_gb': 'Input Data Size (GB)',
		'output_data_size_gb': 'Output Data Size (GB)',
		'progress_percentage': 'Progress (%)',
		'error_message': 'Error Message',
		'submitted_at': 'Submitted At',
		'queued_at': 'Queued At',
		'started_at': 'Started At',
		'completed_at': 'Completed At',
		'execution_time_minutes': 'Execution Time (min)',
		'queue_time_minutes': 'Queue Time (min)',
		'cpu_usage_average': 'Average CPU Usage',
		'memory_usage_peak_gb': 'Peak Memory Usage (GB)',
		'gpu_usage_average': 'Average GPU Usage',
		'estimated_cost': 'Estimated Cost',
		'actual_cost': 'Actual Cost',
		'submitted_by': 'Submitted By',
		'project_id': 'Project ID',
		'billing_account': 'Billing Account',
		'custom_metadata': 'Metadata'
	}
	
	@expose('/job_details/<int:pk>')
	@has_access
	def job_details(self, pk):
		"""View detailed job information"""
		job = self.datamodel.get(pk)
		if not job:
			flash('Job not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			job_details = self._get_job_details(job)
			
			return render_template('distributed_computing/job_details.html',
								   job=job,
								   job_details=job_details,
								   page_title=f"Job Details: {job.job_name}")
		except Exception as e:
			flash(f'Error loading job details: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/cancel_job/<int:pk>')
	@has_access
	def cancel_job(self, pk):
		"""Cancel running job"""
		job = self.datamodel.get(pk)
		if not job:
			flash('Job not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if job.status in ['pending', 'queued', 'running']:
				job.status = 'cancelled'
				self.datamodel.edit(job)
				flash(f'Job {job.job_name} cancelled', 'success')
			else:
				flash('Job cannot be cancelled in current state', 'warning')
		except Exception as e:
			flash(f'Error cancelling job: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/retry_job/<int:pk>')
	@has_access
	def retry_job(self, pk):
		"""Retry failed job"""
		job = self.datamodel.get(pk)
		if not job:
			flash('Job not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if job.status == 'failed' and job.retry_count < job.max_retries:
				job.status = 'pending'
				job.retry_count += 1
				job.error_message = None
				self.datamodel.edit(job)
				flash(f'Job {job.job_name} queued for retry', 'success')
			else:
				flash('Job cannot be retried', 'warning')
		except Exception as e:
			flash(f'Error retrying job: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new job"""
		item.tenant_id = self._get_tenant_id()
		item.submitted_by = self._get_current_user_id()
		item.submitted_at = datetime.utcnow()
		
		# Set default values
		if not item.status:
			item.status = 'pending'
		if not item.priority:
			item.priority = 5
		if not item.max_execution_time_minutes:
			item.max_execution_time_minutes = 60
		if not item.max_retries:
			item.max_retries = 3
	
	def _get_job_details(self, job: DCJob) -> Dict[str, Any]:
		"""Get detailed information for job"""
		return {
			'resource_requirements': job.get_resource_requirements(),
			'execution_history': [
				{
					'attempt': execution.execution_attempt,
					'node': execution.node.node_name if execution.node else 'N/A',
					'status': execution.status,
					'duration': execution.duration_seconds,
					'cpu_usage': execution.average_cpu_usage,
					'memory_usage': execution.average_memory_usage_gb
				}
				for execution in job.executions
			],
			'dependencies': {
				'depends_on': job.depends_on_jobs,
				'blocks': job.blocks_jobs
			},
			'performance_metrics': {
				'execution_time': job.execution_time_minutes,
				'queue_time': job.queue_time_minutes,
				'cpu_efficiency': 0.0,  # Would be calculated from executions
				'cost_efficiency': 0.0
			}
		}
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class DCJobExecutionModelView(ModelView):
	"""Job execution monitoring view"""
	
	datamodel = SQLAInterface(DCJobExecution)
	
	# List view configuration
	list_columns = [
		'job', 'node', 'status', 'execution_attempt',
		'assigned_at', 'duration_seconds', 'peak_cpu_usage', 'peak_memory_usage_gb'
	]
	show_columns = [
		'execution_id', 'job', 'node', 'execution_attempt', 'container_id',
		'status', 'progress_percentage', 'exit_code', 'error_message',
		'assigned_at', 'started_at', 'completed_at', 'duration_seconds',
		'peak_cpu_usage', 'average_cpu_usage', 'peak_memory_usage_gb',
		'average_memory_usage_gb', 'peak_gpu_usage', 'average_gpu_usage',
		'cpu_efficiency', 'memory_efficiency', 'gpu_efficiency', 'total_cost'
	]
	# Read-only view for executions
	edit_columns = []
	add_columns = []
	can_create = False
	can_edit = False
	can_delete = False
	
	# Search and filtering
	search_columns = ['job.job_name', 'node.node_name', 'status']
	base_filters = [['status', lambda: 'completed', lambda: True]]
	
	# Ordering
	base_order = ('assigned_at', 'desc')
	
	# Custom labels
	label_columns = {
		'execution_id': 'Execution ID',
		'execution_attempt': 'Attempt #',
		'container_id': 'Container ID',
		'process_id': 'Process ID',
		'progress_percentage': 'Progress (%)',
		'exit_code': 'Exit Code',
		'error_message': 'Error Message',
		'assigned_at': 'Assigned At',
		'started_at': 'Started At',
		'completed_at': 'Completed At',
		'duration_seconds': 'Duration (seconds)',
		'cpu_usage_samples': 'CPU Usage Samples',
		'memory_usage_samples': 'Memory Usage Samples',
		'gpu_usage_samples': 'GPU Usage Samples',
		'network_io_bytes': 'Network I/O (bytes)',
		'disk_io_bytes': 'Disk I/O (bytes)',
		'peak_cpu_usage': 'Peak CPU Usage (%)',
		'average_cpu_usage': 'Average CPU Usage (%)',
		'peak_memory_usage_gb': 'Peak Memory (GB)',
		'average_memory_usage_gb': 'Average Memory (GB)',
		'peak_gpu_usage': 'Peak GPU Usage (%)',
		'average_gpu_usage': 'Average GPU Usage (%)',
		'stdout_log': 'Standard Output',
		'stderr_log': 'Standard Error',
		'log_file_path': 'Log File Path',
		'output_file_paths': 'Output Files',
		'cpu_efficiency': 'CPU Efficiency',
		'memory_efficiency': 'Memory Efficiency',
		'gpu_efficiency': 'GPU Efficiency',
		'compute_cost': 'Compute Cost',
		'storage_cost': 'Storage Cost',
		'network_cost': 'Network Cost',
		'total_cost': 'Total Cost'
	}
	
	@expose('/execution_logs/<int:pk>')
	@has_access
	def execution_logs(self, pk):
		"""View execution logs"""
		execution = self.datamodel.get(pk)
		if not execution:
			flash('Execution not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			logs_data = self._get_execution_logs(execution)
			
			return render_template('distributed_computing/execution_logs.html',
								   execution=execution,
								   logs_data=logs_data,
								   page_title=f"Execution Logs: {execution.job.job_name}")
		except Exception as e:
			flash(f'Error loading execution logs: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/resource_usage/<int:pk>')
	@has_access
	def resource_usage(self, pk):
		"""View resource usage timeline"""
		execution = self.datamodel.get(pk)
		if not execution:
			flash('Execution not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			usage_data = self._get_resource_usage_data(execution)
			
			return render_template('distributed_computing/resource_usage.html',
								   execution=execution,
								   usage_data=usage_data,
								   page_title=f"Resource Usage: {execution.job.job_name}")
		except Exception as e:
			flash(f'Error loading resource usage: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def _get_execution_logs(self, execution: DCJobExecution) -> Dict[str, Any]:
		"""Get execution logs and output"""
		return {
			'stdout': execution.stdout_log or "No standard output available",
			'stderr': execution.stderr_log or "No standard error output",
			'log_file': execution.log_file_path,
			'output_files': execution.output_file_paths,
			'container_id': execution.container_id,
			'exit_code': execution.exit_code
		}
	
	def _get_resource_usage_data(self, execution: DCJobExecution) -> Dict[str, Any]:
		"""Get resource usage timeline data"""
		return {
			'cpu_samples': execution.cpu_usage_samples,
			'memory_samples': execution.memory_usage_samples,
			'gpu_samples': execution.gpu_usage_samples,
			'peak_usage': {
				'cpu': execution.peak_cpu_usage,
				'memory': execution.peak_memory_usage_gb,
				'gpu': execution.peak_gpu_usage
			},
			'efficiency_metrics': {
				'cpu': execution.cpu_efficiency,
				'memory': execution.memory_efficiency,
				'gpu': execution.gpu_efficiency
			}
		}


class DistributedComputingDashboardView(DistributedComputingBaseView):
	"""Distributed computing dashboard"""
	
	route_base = "/distributed_computing_dashboard"
	default_view = "index"
	
	@expose('/')
	@has_access
	def index(self):
		"""Distributed computing dashboard main page"""
		try:
			# Get dashboard metrics
			metrics = self._get_dashboard_metrics()
			
			return render_template('distributed_computing/dashboard.html',
								   metrics=metrics,
								   page_title="Distributed Computing Dashboard")
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return render_template('distributed_computing/dashboard.html',
								   metrics={},
								   page_title="Distributed Computing Dashboard")
	
	@expose('/cluster_overview/')
	@has_access
	def cluster_overview(self):
		"""Cluster overview page"""
		try:
			overview_data = self._get_cluster_overview_data()
			
			return render_template('distributed_computing/cluster_overview.html',
								   overview_data=overview_data,
								   page_title="Cluster Overview")
		except Exception as e:
			flash(f'Error loading cluster overview: {str(e)}', 'error')
			return redirect(url_for('DistributedComputingDashboardView.index'))
	
	@expose('/job_queue/')
	@has_access
	def job_queue(self):
		"""Job queue management page"""
		try:
			queue_data = self._get_job_queue_data()
			
			return render_template('distributed_computing/job_queue.html',
								   queue_data=queue_data,
								   page_title="Job Queue")
		except Exception as e:
			flash(f'Error loading job queue: {str(e)}', 'error')
			return redirect(url_for('DistributedComputingDashboardView.index'))
	
	@expose('/performance_analytics/')
	@has_access
	def performance_analytics(self):
		"""Performance analytics dashboard"""
		try:
			period_days = int(request.args.get('period', 7))
			analytics_data = self._get_performance_analytics(period_days)
			
			return render_template('distributed_computing/performance_analytics.html',
								   analytics_data=analytics_data,
								   period_days=period_days,
								   page_title="Performance Analytics")
		except Exception as e:
			flash(f'Error loading performance analytics: {str(e)}', 'error')
			return redirect(url_for('DistributedComputingDashboardView.index'))
	
	def _get_dashboard_metrics(self) -> Dict[str, Any]:
		"""Get distributed computing dashboard metrics"""
		# Implementation would calculate real metrics from database
		return {
			'total_clusters': 5,
			'active_clusters': 4,
			'total_nodes': 125,
			'healthy_nodes': 118,
			'total_jobs': 2456,
			'running_jobs': 23,
			'queued_jobs': 8,
			'completed_jobs_today': 89,
			'failed_jobs_today': 3,
			'total_cpu_cores': 2400,
			'used_cpu_cores': 1680,
			'total_memory_gb': 9600,
			'used_memory_gb': 6720,
			'total_gpus': 480,
			'used_gpus': 156,
			'average_job_duration': 42.5,
			'job_success_rate': 96.8,
			'cluster_utilization': {
				'cpu': 70.0,
				'memory': 70.0,
				'gpu': 32.5
			},
			'cost_metrics': {
				'hourly_cost': 245.60,
				'daily_cost': 5894.40,
				'monthly_budget': 180000.0,
				'cost_efficiency': 87.2
			}
		}
	
	def _get_cluster_overview_data(self) -> Dict[str, Any]:
		"""Get cluster overview data"""
		return {
			'clusters': [
				{
					'name': 'Production Cluster',
					'nodes': 45,
					'status': 'healthy',
					'utilization': {'cpu': 78, 'memory': 65, 'gpu': 45},
					'cost_per_hour': 89.50
				},
				{
					'name': 'Development Cluster',
					'nodes': 15,
					'status': 'healthy',
					'utilization': {'cpu': 45, 'memory': 38, 'gpu': 12},
					'cost_per_hour': 28.75
				},
				{
					'name': 'Research Cluster',
					'nodes': 35,
					'status': 'healthy',
					'utilization': {'cpu': 89, 'memory': 82, 'gpu': 76},
					'cost_per_hour': 127.35
				}
			],
			'node_types': {
				'cpu_only': {'count': 65, 'utilization': 72},
				'gpu_enabled': {'count': 35, 'utilization': 58},
				'high_memory': {'count': 20, 'utilization': 43},
				'storage_optimized': {'count': 5, 'utilization': 89}
			},
			'scaling_recommendations': [
				{'cluster': 'Research Cluster', 'action': 'scale_up', 'reason': 'High utilization'},
				{'cluster': 'Development Cluster', 'action': 'scale_down', 'reason': 'Low utilization'}
			]
		}
	
	def _get_job_queue_data(self) -> Dict[str, Any]:
		"""Get job queue data"""
		return {
			'queue_statistics': {
				'pending_jobs': 15,
				'queued_jobs': 8,
				'running_jobs': 23,
				'average_queue_time': 3.2,
				'average_execution_time': 42.5
			},
			'priority_distribution': {
				'high': 5,
				'medium': 18,
				'low': 23
			},
			'job_types': {
				'simulation': 28,
				'analysis': 12,
				'training': 4,
				'inference': 2
			},
			'recent_jobs': [
				{
					'name': 'Fluid Dynamics Simulation #1247',
					'status': 'running',
					'progress': 68,
					'node': 'gpu-node-15',
					'runtime': 45
				},
				{
					'name': 'Molecular Analysis #892',
					'status': 'queued',
					'priority': 8,
					'queue_time': 2.1,
					'estimated_start': 5
				}
			]
		}
	
	def _get_performance_analytics(self, period_days: int) -> Dict[str, Any]:
		"""Get performance analytics data"""
		return {
			'period_days': period_days,
			'job_completion_trends': [
				{'date': '2024-01-15', 'completed': 89, 'failed': 3},
				{'date': '2024-01-16', 'completed': 95, 'failed': 2},
				{'date': '2024-01-17', 'completed': 78, 'failed': 5}
			],
			'resource_utilization_trends': {
				'cpu': [65, 72, 78, 68, 75, 82, 70],
				'memory': [58, 63, 70, 65, 68, 75, 70],
				'gpu': [28, 35, 42, 38, 45, 52, 32]
			},
			'performance_metrics': {
				'average_job_duration': 42.5,
				'job_success_rate': 96.8,
				'queue_efficiency': 94.2,
				'resource_efficiency': 87.1,
				'cost_per_job': 12.45
			},
			'bottleneck_analysis': {
				'cpu_bound_jobs': 45,
				'memory_bound_jobs': 23,
				'io_bound_jobs': 18,
				'gpu_bound_jobs': 14
			}
		}


# Register views with AppBuilder
def register_views(appbuilder):
	"""Register all distributed computing views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		DCClusterModelView,
		"Clusters",
		icon="fa-server",
		category="Distributed Computing",
		category_icon="fa-network-wired"
	)
	
	appbuilder.add_view(
		DCNodeModelView,
		"Compute Nodes",
		icon="fa-microchip",
		category="Distributed Computing"
	)
	
	appbuilder.add_view(
		DCJobModelView,
		"Jobs",
		icon="fa-tasks",
		category="Distributed Computing"
	)
	
	appbuilder.add_view(
		DCJobExecutionModelView,
		"Job Executions",
		icon="fa-play-circle",
		category="Distributed Computing"
	)
	
	# Dashboard views
	appbuilder.add_view_no_menu(DistributedComputingDashboardView)
	
	# Menu links
	appbuilder.add_link(
		"Computing Dashboard",
		href="/distributed_computing_dashboard/",
		icon="fa-dashboard",
		category="Distributed Computing"
	)
	
	appbuilder.add_link(
		"Cluster Overview",
		href="/distributed_computing_dashboard/cluster_overview/",
		icon="fa-eye",
		category="Distributed Computing"
	)
	
	appbuilder.add_link(
		"Job Queue",
		href="/distributed_computing_dashboard/job_queue/",
		icon="fa-list",
		category="Distributed Computing"
	)
	
	appbuilder.add_link(
		"Performance Analytics",
		href="/distributed_computing_dashboard/performance_analytics/",
		icon="fa-chart-line",
		category="Distributed Computing"
	)