"""
Flask-AppBuilder blueprint for Edge Computing management interface

This module provides a comprehensive web interface for managing edge computing
resources, monitoring performance, and orchestrating distributed workloads.
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.widgets import ListWidget, ShowWidget, FormWidget
from flask_appbuilder.baseviews import expose
from flask_appbuilder.security.decorators import has_access
from flask_appbuilder.fields import QuerySelectField
from wtforms import Form, StringField, FloatField, IntegerField, SelectField, TextAreaField
from wtforms.validators import DataRequired, NumberRange
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from flask_appbuilder import Model

from ..capabilities.edge_computing import (
	EdgeComputingCluster,
	EdgeComputingNode,
	EdgeTask,
	EdgeNodeType,
	EdgeTaskPriority,
	EdgeEnabledDigitalTwin
)

Base = declarative_base()

# Database Models for Edge Computing
class EdgeNodeModel(Model):
	"""Database model for edge computing nodes"""
	
	__tablename__ = 'edge_nodes'
	
	id = Column(String(36), primary_key=True)
	name = Column(String(200), nullable=False)
	node_type = Column(String(50), nullable=False)
	location_data = Column(JSON)
	capacity_data = Column(JSON)
	current_load_data = Column(JSON)
	status = Column(String(50), default='inactive')
	last_heartbeat = Column(DateTime)
	network_latency_ms = Column(Float, default=0.0)
	reliability_score = Column(Float, default=1.0)
	energy_efficiency = Column(Float, default=1.0)
	specialized_capabilities = Column(JSON)
	connected_devices = Column(JSON)
	metadata = Column(JSON)
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	def __repr__(self):
		return f'<EdgeNode {self.name}>'

class EdgeTaskModel(Model):
	"""Database model for edge computing tasks"""
	
	__tablename__ = 'edge_tasks'
	
	id = Column(String(36), primary_key=True)
	twin_id = Column(String(36), nullable=False)
	task_type = Column(String(100), nullable=False)
	priority = Column(String(50), nullable=False)
	requirements_data = Column(JSON)
	payload_data = Column(JSON)
	created_at = Column(DateTime, default=datetime.utcnow)
	deadline = Column(DateTime)
	assigned_node = Column(String(36))
	status = Column(String(50), default='pending')
	result_data = Column(JSON)
	execution_time_ms = Column(Float)
	metadata = Column(JSON)
	
	def __repr__(self):
		return f'<EdgeTask {self.id} - {self.task_type}>'

class EdgeClusterMetricsModel(Model):
	"""Database model for edge cluster performance metrics"""
	
	__tablename__ = 'edge_cluster_metrics'
	
	id = Column(Integer, primary_key=True)
	cluster_id = Column(String(36), nullable=False)
	timestamp = Column(DateTime, default=datetime.utcnow)
	total_nodes = Column(Integer, default=0)
	active_nodes = Column(Integer, default=0)
	total_tasks_processed = Column(Integer, default=0)
	avg_task_latency_ms = Column(Float, default=0.0)
	successful_tasks = Column(Integer, default=0)
	failed_tasks = Column(Integer, default=0)
	network_utilization = Column(Float, default=0.0)
	metrics_data = Column(JSON)
	
	def __repr__(self):
		return f'<EdgeMetrics {self.cluster_id} at {self.timestamp}>'

# Forms for Edge Computing management
class EdgeNodeForm(Form):
	"""Form for creating/editing edge computing nodes"""
	
	name = StringField('Node Name', validators=[DataRequired()])
	node_type = SelectField('Node Type', choices=[
		('gateway', 'Gateway'),
		('compute', 'Compute'),
		('storage', 'Storage'),
		('hybrid', 'Hybrid'),
		('iot_bridge', 'IoT Bridge')
	], validators=[DataRequired()])
	
	# Location fields
	latitude = FloatField('Latitude', validators=[DataRequired(), NumberRange(-90, 90)])
	longitude = FloatField('Longitude', validators=[DataRequired(), NumberRange(-180, 180)])
	altitude = FloatField('Altitude (m)', default=0.0)
	
	# Capacity fields
	cpu_cores = IntegerField('CPU Cores', validators=[DataRequired(), NumberRange(1, 128)])
	memory_gb = FloatField('Memory (GB)', validators=[DataRequired(), NumberRange(0.1, 1024)])
	storage_gb = FloatField('Storage (GB)', validators=[DataRequired(), NumberRange(1, 10000)])
	network_mbps = FloatField('Network (Mbps)', validators=[DataRequired(), NumberRange(1, 100000)])
	gpu_cores = IntegerField('GPU Cores', default=0, validators=[NumberRange(0, 64)])
	
	# Performance characteristics
	network_latency_ms = FloatField('Network Latency (ms)', default=1.0, validators=[NumberRange(0.1, 1000)])
	reliability_score = FloatField('Reliability Score', default=1.0, validators=[NumberRange(0.0, 1.0)])
	energy_efficiency = FloatField('Energy Efficiency', default=1.0, validators=[NumberRange(0.1, 2.0)])
	
	specialized_capabilities = TextAreaField('Specialized Capabilities (JSON)')

class EdgeTaskForm(Form):
	"""Form for creating edge computing tasks"""
	
	twin_id = StringField('Digital Twin ID', validators=[DataRequired()])
	task_type = SelectField('Task Type', choices=[
		('sensor_data_processing', 'Sensor Data Processing'),
		('predictive_analysis', 'Predictive Analysis'),
		('real_time_control', 'Real-time Control'),
		('stream_analytics', 'Stream Analytics'),
		('ml_inference', 'ML Inference'),
		('data_aggregation', 'Data Aggregation')
	], validators=[DataRequired()])
	
	priority = SelectField('Priority', choices=[
		('critical', 'Critical (<1ms)'),
		('high', 'High (<5ms)'),
		('normal', 'Normal (<10ms)'),
		('low', 'Low (<50ms)')
	], validators=[DataRequired()])
	
	# Resource requirements
	cpu_cores = FloatField('CPU Cores', validators=[DataRequired(), NumberRange(0.1, 32)])
	memory_mb = FloatField('Memory (MB)', validators=[DataRequired(), NumberRange(1, 32768)])
	max_latency_ms = FloatField('Max Latency (ms)', validators=[DataRequired(), NumberRange(0.1, 10000)])
	gpu_required = SelectField('GPU Required', choices=[('false', 'No'), ('true', 'Yes')])
	
	payload_data = TextAreaField('Task Payload (JSON)')

# Model Views for Flask-AppBuilder
class EdgeNodeModelView(ModelView):
	"""Model view for edge computing nodes"""
	
	datamodel = SQLAInterface(EdgeNodeModel)
	
	list_columns = ['name', 'node_type', 'status', 'network_latency_ms', 'reliability_score', 'last_heartbeat']
	show_columns = ['name', 'node_type', 'location_data', 'capacity_data', 'current_load_data', 
					'status', 'network_latency_ms', 'reliability_score', 'energy_efficiency',
					'specialized_capabilities', 'connected_devices', 'last_heartbeat', 'created_at']
	add_columns = ['name', 'node_type', 'location_data', 'capacity_data', 'network_latency_ms',
				   'reliability_score', 'energy_efficiency', 'specialized_capabilities']
	edit_columns = add_columns
	
	base_order = ('name', 'asc')
	
	list_title = "Edge Computing Nodes"
	show_title = "Edge Node Details"
	add_title = "Add Edge Node"
	edit_title = "Edit Edge Node"

class EdgeTaskModelView(ModelView):
	"""Model view for edge computing tasks"""
	
	datamodel = SQLAInterface(EdgeTaskModel)
	
	list_columns = ['twin_id', 'task_type', 'priority', 'status', 'assigned_node', 'execution_time_ms', 'created_at']
	show_columns = ['twin_id', 'task_type', 'priority', 'requirements_data', 'payload_data',
					'status', 'assigned_node', 'result_data', 'execution_time_ms', 'created_at', 'deadline']
	add_columns = ['twin_id', 'task_type', 'priority', 'requirements_data', 'payload_data', 'deadline']
	edit_columns = ['status', 'assigned_node', 'result_data']
	
	base_order = ('created_at', 'desc')
	
	list_title = "Edge Computing Tasks"
	show_title = "Edge Task Details"
	add_title = "Submit Edge Task"
	edit_title = "Edit Edge Task"

class EdgeClusterMetricsModelView(ModelView):
	"""Model view for edge cluster metrics"""
	
	datamodel = SQLAInterface(EdgeClusterMetricsModel)
	
	list_columns = ['cluster_id', 'timestamp', 'total_nodes', 'active_nodes', 'total_tasks_processed', 
					'avg_task_latency_ms', 'successful_tasks', 'failed_tasks']
	show_columns = ['cluster_id', 'timestamp', 'total_nodes', 'active_nodes', 'total_tasks_processed',
					'avg_task_latency_ms', 'successful_tasks', 'failed_tasks', 'network_utilization', 'metrics_data']
	
	base_order = ('timestamp', 'desc')
	base_permissions = ['can_list', 'can_show']
	
	list_title = "Edge Cluster Metrics"
	show_title = "Cluster Metrics Details"

# Custom Views for Edge Computing Management
class EdgeComputingDashboardView(BaseView):
	"""Dashboard view for edge computing overview"""
	
	route_base = "/edgecomputing"
	
	@expose('/dashboard/')
	@has_access
	def dashboard(self):
		"""Main edge computing dashboard"""
		
		# Get cluster status
		cluster_status = self._get_cluster_status()
		
		# Get recent metrics
		recent_metrics = self._get_recent_metrics()
		
		# Get active tasks
		active_tasks = self._get_active_tasks()
		
		return self.render_template(
			'edge_computing/dashboard.html',
			cluster_status=cluster_status,
			recent_metrics=recent_metrics,
			active_tasks=active_tasks
		)
	
	@expose('/nodes/')
	@has_access
	def nodes(self):
		"""Edge nodes management page"""
		
		# Get all nodes with current status
		nodes = self._get_all_nodes()
		
		return self.render_template(
			'edge_computing/nodes.html',
			nodes=nodes
		)
	
	@expose('/nodes/add/', methods=['GET', 'POST'])
	@has_access
	def add_node(self):
		"""Add new edge node"""
		
		form = EdgeNodeForm(request.form)
		
		if request.method == 'POST' and form.validate():
			# Create new edge node
			node_data = {
				'name': form.name.data,
				'node_type': form.node_type.data,
				'location_data': {
					'lat': form.latitude.data,
					'lng': form.longitude.data,
					'altitude': form.altitude.data
				},
				'capacity_data': {
					'cpu_cores': form.cpu_cores.data,
					'memory_gb': form.memory_gb.data,
					'storage_gb': form.storage_gb.data,
					'network_mbps': form.network_mbps.data,
					'gpu_cores': form.gpu_cores.data
				},
				'network_latency_ms': form.network_latency_ms.data,
				'reliability_score': form.reliability_score.data,
				'energy_efficiency': form.energy_efficiency.data
			}
			
			# Parse specialized capabilities if provided
			if form.specialized_capabilities.data:
				try:
					node_data['specialized_capabilities'] = json.loads(form.specialized_capabilities.data)
				except json.JSONDecodeError:
					node_data['specialized_capabilities'] = []
			
			if self._create_edge_node(node_data):
				flash('Edge node created successfully', 'success')
				return redirect(url_for('EdgeComputingDashboardView.nodes'))
			else:
				flash('Failed to create edge node', 'error')
		
		return self.render_template(
			'edge_computing/add_node.html',
			form=form
		)
	
	@expose('/tasks/')
	@has_access
	def tasks(self):
		"""Edge tasks management page"""
		
		# Get all tasks with current status
		tasks = self._get_all_tasks()
		
		return self.render_template(
			'edge_computing/tasks.html',
			tasks=tasks
		)
	
	@expose('/tasks/submit/', methods=['GET', 'POST'])
	@has_access
	def submit_task(self):
		"""Submit new edge task"""
		
		form = EdgeTaskForm(request.form)
		
		if request.method == 'POST' and form.validate():
			# Create new edge task
			task_data = {
				'twin_id': form.twin_id.data,
				'task_type': form.task_type.data,
				'priority': form.priority.data,
				'requirements_data': {
					'cpu_cores': form.cpu_cores.data,
					'memory_mb': form.memory_mb.data,
					'max_latency_ms': form.max_latency_ms.data,
					'gpu_required': form.gpu_required.data == 'true'
				}
			}
			
			# Parse payload data if provided
			if form.payload_data.data:
				try:
					task_data['payload_data'] = json.loads(form.payload_data.data)
				except json.JSONDecodeError:
					task_data['payload_data'] = {}
			
			if self._submit_edge_task(task_data):
				flash('Edge task submitted successfully', 'success')
				return redirect(url_for('EdgeComputingDashboardView.tasks'))
			else:
				flash('Failed to submit edge task', 'error')
		
		return self.render_template(
			'edge_computing/submit_task.html',
			form=form
		)
	
	@expose('/performance/')
	@has_access
	def performance(self):
		"""Performance monitoring and analytics"""
		
		# Get performance metrics
		performance_data = self._get_performance_metrics()
		
		return self.render_template(
			'edge_computing/performance.html',
			performance_data=performance_data
		)
	
	@expose('/api/cluster/status')
	@has_access
	def api_cluster_status(self):
		"""API endpoint for cluster status"""
		
		status = self._get_cluster_status()
		return jsonify(status)
	
	@expose('/api/nodes/<node_id>/metrics')
	@has_access
	def api_node_metrics(self, node_id):
		"""API endpoint for node-specific metrics"""
		
		metrics = self._get_node_metrics(node_id)
		return jsonify(metrics)
	
	@expose('/api/tasks/<task_id>/status')
	@has_access
	def api_task_status(self, task_id):
		"""API endpoint for task status"""
		
		status = self._get_task_status(task_id)
		return jsonify(status)
	
	# Helper methods
	def _get_cluster_status(self) -> Dict[str, Any]:
		"""Get current cluster status"""
		# In a real implementation, this would connect to the actual edge cluster
		return {
			"nodes": {
				"total": 5,
				"active": 4,
				"inactive": 1
			},
			"tasks": {
				"total": 127,
				"pending": 3,
				"executing": 8,
				"completed": 115,
				"failed": 1
			},
			"performance": {
				"avg_latency_ms": 4.2,
				"throughput_tasks_per_sec": 45.7,
				"success_rate": 99.2
			},
			"timestamp": datetime.utcnow().isoformat()
		}
	
	def _get_recent_metrics(self) -> List[Dict[str, Any]]:
		"""Get recent performance metrics"""
		return [
			{
				"timestamp": "2024-07-24T05:00:00Z",
				"avg_latency_ms": 4.1,
				"throughput": 48.2,
				"active_nodes": 4
			},
			{
				"timestamp": "2024-07-24T04:55:00Z",
				"avg_latency_ms": 4.3,
				"throughput": 46.8,
				"active_nodes": 4
			}
		]
	
	def _get_active_tasks(self) -> List[Dict[str, Any]]:
		"""Get currently active tasks"""
		return [
			{
				"id": "task_001",
				"twin_id": "twin_factory_01",
				"task_type": "sensor_data_processing",
				"priority": "high",
				"status": "executing",
				"assigned_node": "edge_node_01"
			}
		]
	
	def _get_all_nodes(self) -> List[Dict[str, Any]]:
		"""Get all edge nodes"""
		return [
			{
				"id": "node_001",
				"name": "Gateway-01",
				"node_type": "gateway",
				"status": "active",
				"load": 65.2,
				"latency_ms": 2.1
			}
		]
	
	def _get_all_tasks(self) -> List[Dict[str, Any]]:
		"""Get all edge tasks"""
		return [
			{
				"id": "task_001",
				"twin_id": "twin_factory_01",
				"task_type": "sensor_data_processing",
				"priority": "high",
				"status": "completed",
				"execution_time_ms": 3.4
			}
		]
	
	def _create_edge_node(self, node_data: Dict[str, Any]) -> bool:
		"""Create a new edge node"""
		try:
			# In a real implementation, this would create the actual edge node
			# and add it to the cluster
			return True
		except Exception as e:
			logger.error(f"Failed to create edge node: {e}")
			return False
	
	def _submit_edge_task(self, task_data: Dict[str, Any]) -> bool:
		"""Submit a new edge task"""
		try:
			# In a real implementation, this would submit the task to the cluster
			return True
		except Exception as e:
			logger.error(f"Failed to submit edge task: {e}")
			return False
	
	def _get_performance_metrics(self) -> Dict[str, Any]:
		"""Get comprehensive performance metrics"""
		return {
			"cluster_performance": {
				"avg_latency_ms": 4.2,
				"throughput_tasks_per_sec": 45.7,
				"success_rate": 99.2,
				"total_tasks_processed": 127
			},
			"node_performance": [
				{
					"node_id": "node_001",
					"cpu_utilization": 65.2,
					"memory_utilization": 43.8,
					"network_utilization": 23.4,
					"tasks_processed": 42
				}
			]
		}
	
	def _get_node_metrics(self, node_id: str) -> Dict[str, Any]:
		"""Get metrics for specific node"""
		return {
			"node_id": node_id,
			"cpu_utilization": 65.2,
			"memory_utilization": 43.8,
			"network_utilization": 23.4,
			"tasks_processed": 42,
			"avg_latency_ms": 3.8,
			"uptime_hours": 72.5
		}
	
	def _get_task_status(self, task_id: str) -> Dict[str, Any]:
		"""Get status of specific task"""
		return {
			"task_id": task_id,
			"status": "completed",
			"assigned_node": "node_001",
			"execution_time_ms": 3.4,
			"result": {"processed": True}
		}

# Template functions for edge computing
def init_edge_computing_blueprint(appbuilder):
	"""Initialize edge computing blueprint with Flask-AppBuilder"""
	
	# Add model views
	appbuilder.add_view(
		EdgeNodeModelView,
		"Edge Nodes",
		icon="fa-server",
		category="Edge Computing"
	)
	
	appbuilder.add_view(
		EdgeTaskModelView,
		"Edge Tasks",
		icon="fa-tasks",
		category="Edge Computing"
	)
	
	appbuilder.add_view(
		EdgeClusterMetricsModelView,
		"Cluster Metrics",
		icon="fa-chart-line",
		category="Edge Computing"
	)
	
	# Add dashboard view
	appbuilder.add_view(
		EdgeComputingDashboardView,
		"Edge Dashboard",
		icon="fa-tachometer-alt",
		category="Edge Computing"
	)
	
	# Add separator for edge computing section
	appbuilder.add_separator("Edge Computing")

# Utility functions
def create_edge_computing_tables(db):
	"""Create database tables for edge computing"""
	EdgeNodeModel.__table__.create(db.engine, checkfirst=True)
	EdgeTaskModel.__table__.create(db.engine, checkfirst=True)
	EdgeClusterMetricsModel.__table__.create(db.engine, checkfirst=True)

def _log_edge_operation(operation: str, details: str) -> str:
	"""Log edge computing operations"""
	return f"[EDGE-WEB] {operation}: {details}"