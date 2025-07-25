"""
Distributed Computing Models

Database models for distributed computing framework, cluster management,
job scheduling, resource allocation, and performance monitoring.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


class DCCluster(Model, AuditMixin, BaseMixin):
	"""
	Distributed computing cluster configuration and management.
	
	Represents a cluster of compute nodes for distributed processing
	with auto-scaling, resource management, and monitoring capabilities.
	"""
	__tablename__ = 'dc_cluster'
	
	# Identity
	cluster_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Cluster Information
	cluster_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	cluster_type = Column(String(50), nullable=False, index=True)  # kubernetes, docker_swarm, slurm
	cluster_version = Column(String(20), nullable=True)
	
	# Configuration
	configuration = Column(JSON, default=dict)  # Cluster configuration
	environment_variables = Column(JSON, default=dict)  # Environment settings
	security_settings = Column(JSON, default=dict)  # Security configuration
	
	# Location and Network
	region = Column(String(50), nullable=True)
	availability_zone = Column(String(50), nullable=True)
	network_config = Column(JSON, default=dict)  # Network settings
	storage_config = Column(JSON, default=dict)  # Storage configuration
	
	# Status and Health
	status = Column(String(20), default='active', index=True)  # active, inactive, maintenance, error
	health_status = Column(String(20), default='healthy', index=True)  # healthy, degraded, unhealthy
	last_health_check = Column(DateTime, nullable=True)
	
	# Capacity and Resources
	total_cpu_cores = Column(Integer, default=0)
	total_memory_gb = Column(Float, default=0.0)
	total_storage_gb = Column(Float, default=0.0)
	total_gpu_count = Column(Integer, default=0)
	
	# Current Usage
	used_cpu_cores = Column(Integer, default=0)
	used_memory_gb = Column(Float, default=0.0)
	used_storage_gb = Column(Float, default=0.0)
	used_gpu_count = Column(Integer, default=0)
	
	# Scaling Configuration
	auto_scaling_enabled = Column(Boolean, default=False)
	min_nodes = Column(Integer, default=1)
	max_nodes = Column(Integer, default=10)
	scale_up_threshold = Column(Float, default=0.8)  # CPU threshold for scaling up
	scale_down_threshold = Column(Float, default=0.3)  # CPU threshold for scaling down
	
	# Cost and Billing
	hourly_cost = Column(Float, default=0.0)
	monthly_budget = Column(Float, nullable=True)
	cost_alerts_enabled = Column(Boolean, default=True)
	
	# Access and Authentication
	access_endpoint = Column(String(500), nullable=True)
	authentication_method = Column(String(50), default='token')  # token, certificate, password
	access_credentials = Column(JSON, default=dict)  # Encrypted credentials
	
	# Relationships
	nodes = relationship("DCNode", back_populates="cluster")
	jobs = relationship("DCJob", back_populates="cluster")
	
	def __repr__(self):
		return f"<DCCluster {self.cluster_name}>"
	
	def calculate_utilization(self) -> Dict[str, float]:
		"""Calculate resource utilization percentages"""
		cpu_util = (self.used_cpu_cores / self.total_cpu_cores * 100) if self.total_cpu_cores > 0 else 0
		memory_util = (self.used_memory_gb / self.total_memory_gb * 100) if self.total_memory_gb > 0 else 0
		storage_util = (self.used_storage_gb / self.total_storage_gb * 100) if self.total_storage_gb > 0 else 0
		gpu_util = (self.used_gpu_count / self.total_gpu_count * 100) if self.total_gpu_count > 0 else 0
		
		return {
			'cpu': cpu_util,
			'memory': memory_util,
			'storage': storage_util,
			'gpu': gpu_util
		}
	
	def needs_scaling(self) -> str:
		"""Determine if cluster needs scaling"""
		if not self.auto_scaling_enabled:
			return 'none'
		
		utilization = self.calculate_utilization()
		cpu_util = utilization['cpu'] / 100
		
		if cpu_util >= self.scale_up_threshold:
			return 'up'
		elif cpu_util <= self.scale_down_threshold:
			return 'down'
		
		return 'none'


class DCNode(Model, AuditMixin, BaseMixin):
	"""
	Individual compute node in distributed cluster.
	
	Represents a single compute node with resource specifications,
	status monitoring, and job assignment capabilities.
	"""
	__tablename__ = 'dc_node'
	
	# Identity
	node_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	cluster_id = Column(String(36), ForeignKey('dc_cluster.cluster_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Node Information
	node_name = Column(String(200), nullable=False, index=True)
	hostname = Column(String(200), nullable=True, index=True)
	ip_address = Column(String(45), nullable=True, index=True)
	external_ip = Column(String(45), nullable=True)
	
	# Node Classification
	node_type = Column(String(50), nullable=False, index=True)  # cpu_only, gpu_enabled, high_memory, etc.
	instance_type = Column(String(50), nullable=True)  # AWS/Azure instance type
	availability_zone = Column(String(50), nullable=True)
	
	# Hardware Specifications
	cpu_cores = Column(Integer, nullable=False)
	cpu_architecture = Column(String(20), default='x86_64')  # x86_64, arm64
	memory_gb = Column(Float, nullable=False)
	storage_gb = Column(Float, nullable=False)
	gpu_count = Column(Integer, default=0)
	gpu_type = Column(String(50), nullable=True)  # V100, A100, RTX3090, etc.
	network_bandwidth_gbps = Column(Float, nullable=True)
	
	# Current Resource Usage
	cpu_usage_percent = Column(Float, default=0.0)
	memory_usage_gb = Column(Float, default=0.0)
	storage_usage_gb = Column(Float, default=0.0)
	gpu_usage_percent = Column(Float, default=0.0)
	network_usage_mbps = Column(Float, default=0.0)
	
	# Status and Health
	status = Column(String(20), default='active', index=True)  # active, inactive, maintenance, draining
	health_status = Column(String(20), default='healthy', index=True)  # healthy, degraded, unhealthy
	last_heartbeat = Column(DateTime, nullable=True, index=True)
	last_health_check = Column(DateTime, nullable=True)
	uptime_hours = Column(Float, default=0.0)
	
	# Software and Environment
	operating_system = Column(String(50), nullable=True)
	container_runtime = Column(String(50), nullable=True)  # docker, containerd, cri-o
	kubernetes_version = Column(String(20), nullable=True)
	installed_packages = Column(JSON, default=list)  # List of installed software
	environment_labels = Column(JSON, default=dict)  # Node labels and tags
	
	# Job Assignment
	assigned_jobs = Column(JSON, default=list)  # Currently assigned job IDs
	max_concurrent_jobs = Column(Integer, default=4)
	current_job_count = Column(Integer, default=0)
	total_jobs_completed = Column(Integer, default=0)
	
	# Performance Metrics
	average_cpu_usage = Column(Float, default=0.0)
	average_memory_usage = Column(Float, default=0.0)
	job_success_rate = Column(Float, default=100.0)
	average_job_duration = Column(Float, default=0.0)  # Average job duration in minutes
	
	# Cost and Billing
	hourly_cost = Column(Float, default=0.0)
	total_cost = Column(Float, default=0.0)
	cost_per_job = Column(Float, nullable=True)
	
	# Relationships
	cluster = relationship("DCCluster", back_populates="nodes")
	job_executions = relationship("DCJobExecution", back_populates="node")
	
	def __repr__(self):
		return f"<DCNode {self.node_name} ({self.node_type})>"
	
	def is_available_for_job(self) -> bool:
		"""Check if node is available for new job assignment"""
		if self.status != 'active' or self.health_status != 'healthy':
			return False
		
		if self.current_job_count >= self.max_concurrent_jobs:
			return False
		
		# Check if last heartbeat is recent (within 5 minutes)
		if self.last_heartbeat:
			heartbeat_age = datetime.utcnow() - self.last_heartbeat
			if heartbeat_age > timedelta(minutes=5):
				return False
		
		return True
	
	def get_available_resources(self) -> Dict[str, float]:
		"""Get available resources on the node"""
		return {
			'cpu_cores': self.cpu_cores - (self.cpu_cores * self.cpu_usage_percent / 100),
			'memory_gb': self.memory_gb - self.memory_usage_gb,
			'storage_gb': self.storage_gb - self.storage_usage_gb,
			'gpu_count': self.gpu_count - (self.gpu_count * self.gpu_usage_percent / 100)
		}
	
	def assign_job(self, job_id: str) -> bool:
		"""Assign a job to this node"""
		if not self.is_available_for_job():
			return False
		
		if job_id not in self.assigned_jobs:
			self.assigned_jobs.append(job_id)
			self.current_job_count += 1
		
		return True
	
	def release_job(self, job_id: str):
		"""Release a job from this node"""
		if job_id in self.assigned_jobs:
			self.assigned_jobs.remove(job_id)
			self.current_job_count = max(0, self.current_job_count - 1)
			self.total_jobs_completed += 1


class DCJob(Model, AuditMixin, BaseMixin):
	"""
	Distributed computing job definition and management.
	
	Represents a computational job that can be distributed across
	multiple nodes with dependencies, resource requirements, and monitoring.
	"""
	__tablename__ = 'dc_job'
	
	# Identity
	job_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	cluster_id = Column(String(36), ForeignKey('dc_cluster.cluster_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Job Information
	job_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	job_type = Column(String(50), nullable=False, index=True)  # simulation, analysis, training, inference
	simulation_type = Column(String(50), nullable=True)  # physics, fluid_dynamics, molecular, etc.
	
	# Job Definition
	command = Column(Text, nullable=False)  # Command to execute
	arguments = Column(JSON, default=list)  # Command arguments
	environment_variables = Column(JSON, default=dict)  # Environment variables
	working_directory = Column(String(500), nullable=True)
	
	# Resource Requirements
	required_cpu_cores = Column(Integer, default=1)
	required_memory_gb = Column(Float, default=1.0)
	required_storage_gb = Column(Float, default=1.0)
	required_gpu_count = Column(Integer, default=0)
	required_gpu_type = Column(String(50), nullable=True)
	
	# Execution Configuration
	max_execution_time_minutes = Column(Integer, default=60)
	retry_count = Column(Integer, default=0)
	max_retries = Column(Integer, default=3)
	priority = Column(Integer, default=5)  # 1-10, higher is more priority
	parallelizable = Column(Boolean, default=False)
	max_parallel_tasks = Column(Integer, default=1)
	
	# Dependencies
	depends_on_jobs = Column(JSON, default=list)  # List of job IDs this job depends on
	blocks_jobs = Column(JSON, default=list)  # List of job IDs that depend on this job
	
	# Input/Output
	input_files = Column(JSON, default=list)  # List of input file paths
	output_files = Column(JSON, default=list)  # List of expected output file paths
	input_data_size_gb = Column(Float, default=0.0)
	output_data_size_gb = Column(Float, nullable=True)
	
	# Status and Progress
	status = Column(String(20), default='pending', index=True)  # pending, queued, running, completed, failed, cancelled
	progress_percentage = Column(Float, default=0.0)
	error_message = Column(Text, nullable=True)
	
	# Timing
	submitted_at = Column(DateTime, nullable=False, index=True)
	queued_at = Column(DateTime, nullable=True)
	started_at = Column(DateTime, nullable=True)
	completed_at = Column(DateTime, nullable=True)
	
	# Performance Metrics
	execution_time_minutes = Column(Float, nullable=True)
	queue_time_minutes = Column(Float, nullable=True)
	cpu_usage_average = Column(Float, nullable=True)
	memory_usage_peak_gb = Column(Float, nullable=True)
	gpu_usage_average = Column(Float, nullable=True)
	
	# Cost and Billing
	estimated_cost = Column(Float, nullable=True)
	actual_cost = Column(Float, nullable=True)
	cost_breakdown = Column(JSON, default=dict)  # Detailed cost breakdown
	
	# User and Attribution
	submitted_by = Column(String(36), nullable=False, index=True)
	project_id = Column(String(36), nullable=True, index=True)
	billing_account = Column(String(36), nullable=True)
	
	# Metadata
	tags = Column(JSON, default=list)  # Job tags for organization
	custom_metadata = Column(JSON, default=dict)  # Custom metadata
	
	# Relationships
	cluster = relationship("DCCluster", back_populates="jobs")
	executions = relationship("DCJobExecution", back_populates="job")
	
	def __repr__(self):
		return f"<DCJob {self.job_name} ({self.status})>"
	
	def calculate_queue_time(self) -> Optional[float]:
		"""Calculate time spent in queue (minutes)"""
		if self.queued_at and self.started_at:
			queue_time = self.started_at - self.queued_at
			self.queue_time_minutes = queue_time.total_seconds() / 60
			return self.queue_time_minutes
		return None
	
	def calculate_execution_time(self) -> Optional[float]:
		"""Calculate total execution time (minutes)"""
		if self.started_at and self.completed_at:
			execution_time = self.completed_at - self.started_at
			self.execution_time_minutes = execution_time.total_seconds() / 60
			return self.execution_time_minutes
		return None
	
	def is_ready_to_run(self) -> bool:
		"""Check if job is ready to run (all dependencies completed)"""
		if not self.depends_on_jobs:
			return True
		
		# Implementation would check status of dependent jobs
		# For now, return True as a placeholder
		return True
	
	def get_resource_requirements(self) -> Dict[str, Any]:
		"""Get comprehensive resource requirements"""
		return {
			'cpu_cores': self.required_cpu_cores,
			'memory_gb': self.required_memory_gb,
			'storage_gb': self.required_storage_gb,
			'gpu_count': self.required_gpu_count,
			'gpu_type': self.required_gpu_type,
			'estimated_duration_minutes': self.max_execution_time_minutes
		}


class DCJobExecution(Model, AuditMixin, BaseMixin):
	"""
	Individual job execution instance on a specific node.
	
	Tracks the execution of a job on a specific node with detailed
	performance metrics, resource usage, and execution history.
	"""
	__tablename__ = 'dc_job_execution'
	
	# Identity
	execution_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	job_id = Column(String(36), ForeignKey('dc_job.job_id'), nullable=False, index=True)
	node_id = Column(String(36), ForeignKey('dc_node.node_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Execution Details
	execution_attempt = Column(Integer, default=1)  # Which retry attempt this is
	container_id = Column(String(100), nullable=True)  # Container/process ID
	process_id = Column(Integer, nullable=True)  # Operating system process ID
	
	# Status and Progress
	status = Column(String(20), default='pending', index=True)  # pending, running, completed, failed, killed
	progress_percentage = Column(Float, default=0.0)
	exit_code = Column(Integer, nullable=True)
	error_message = Column(Text, nullable=True)
	
	# Timing
	assigned_at = Column(DateTime, nullable=False, index=True)
	started_at = Column(DateTime, nullable=True)
	completed_at = Column(DateTime, nullable=True)
	duration_seconds = Column(Float, nullable=True)
	
	# Resource Usage Tracking
	cpu_usage_samples = Column(JSON, default=list)  # CPU usage over time
	memory_usage_samples = Column(JSON, default=list)  # Memory usage over time
	gpu_usage_samples = Column(JSON, default=list)  # GPU usage over time
	network_io_bytes = Column(Float, default=0.0)
	disk_io_bytes = Column(Float, default=0.0)
	
	# Performance Metrics
	peak_cpu_usage = Column(Float, default=0.0)
	average_cpu_usage = Column(Float, default=0.0)
	peak_memory_usage_gb = Column(Float, default=0.0)
	average_memory_usage_gb = Column(Float, default=0.0)
	peak_gpu_usage = Column(Float, default=0.0)
	average_gpu_usage = Column(Float, default=0.0)
	
	# Output and Logs
	stdout_log = Column(Text, nullable=True)  # Standard output
	stderr_log = Column(Text, nullable=True)  # Standard error
	log_file_path = Column(String(500), nullable=True)  # Path to detailed log file
	output_file_paths = Column(JSON, default=list)  # Paths to output files
	
	# Resource Efficiency
	cpu_efficiency = Column(Float, nullable=True)  # Actual vs requested CPU
	memory_efficiency = Column(Float, nullable=True)  # Actual vs requested memory
	gpu_efficiency = Column(Float, nullable=True)  # Actual vs requested GPU
	
	# Cost Information
	compute_cost = Column(Float, nullable=True)  # Cost for compute resources
	storage_cost = Column(Float, nullable=True)  # Cost for storage usage
	network_cost = Column(Float, nullable=True)  # Cost for network transfer
	total_cost = Column(Float, nullable=True)  # Total execution cost
	
	# Relationships
	job = relationship("DCJob", back_populates="executions")
	node = relationship("DCNode", back_populates="job_executions")
	
	def __repr__(self):
		return f"<DCJobExecution {self.job.job_name} on {self.node.node_name}>"
	
	def calculate_duration(self) -> Optional[float]:
		"""Calculate execution duration in seconds"""
		if self.started_at and self.completed_at:
			duration = self.completed_at - self.started_at
			self.duration_seconds = duration.total_seconds()
			return self.duration_seconds
		return None
	
	def calculate_efficiency_metrics(self):
		"""Calculate resource efficiency metrics"""
		job_requirements = self.job.get_resource_requirements()
		
		# CPU efficiency
		if job_requirements['cpu_cores'] > 0 and self.average_cpu_usage > 0:
			self.cpu_efficiency = (self.average_cpu_usage / 100) / job_requirements['cpu_cores']
		
		# Memory efficiency
		if job_requirements['memory_gb'] > 0 and self.average_memory_usage_gb > 0:
			self.memory_efficiency = self.average_memory_usage_gb / job_requirements['memory_gb']
		
		# GPU efficiency
		if job_requirements['gpu_count'] > 0 and self.average_gpu_usage > 0:
			self.gpu_efficiency = (self.average_gpu_usage / 100) / job_requirements['gpu_count']
	
	def add_resource_sample(self, cpu_percent: float, memory_gb: float, gpu_percent: float = 0.0):
		"""Add a resource usage sample"""
		timestamp = datetime.utcnow().isoformat()
		
		self.cpu_usage_samples.append({'timestamp': timestamp, 'value': cpu_percent})
		self.memory_usage_samples.append({'timestamp': timestamp, 'value': memory_gb})
		if gpu_percent > 0:
			self.gpu_usage_samples.append({'timestamp': timestamp, 'value': gpu_percent})
		
		# Update peak and average values
		self.peak_cpu_usage = max(self.peak_cpu_usage, cpu_percent)
		self.peak_memory_usage_gb = max(self.peak_memory_usage_gb, memory_gb)
		self.peak_gpu_usage = max(self.peak_gpu_usage, gpu_percent)
		
		# Calculate running averages
		if self.cpu_usage_samples:
			self.average_cpu_usage = sum(s['value'] for s in self.cpu_usage_samples) / len(self.cpu_usage_samples)
		if self.memory_usage_samples:
			self.average_memory_usage_gb = sum(s['value'] for s in self.memory_usage_samples) / len(self.memory_usage_samples)
		if self.gpu_usage_samples:
			self.average_gpu_usage = sum(s['value'] for s in self.gpu_usage_samples) / len(self.gpu_usage_samples)


class DCWorkflow(Model, AuditMixin, BaseMixin):
	"""
	Distributed computing workflow with multiple interconnected jobs.
	
	Represents a complex workflow consisting of multiple jobs with
	dependencies, data flow, and coordinated execution across the cluster.
	"""
	__tablename__ = 'dc_workflow'
	
	# Identity
	workflow_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Workflow Information
	workflow_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	workflow_type = Column(String(50), nullable=False, index=True)  # pipeline, dag, parallel, sequential
	version = Column(String(20), default='1.0.0')
	
	# Workflow Definition
	job_definitions = Column(JSON, default=list)  # List of job definitions
	job_dependencies = Column(JSON, default=dict)  # Job dependency graph
	data_flow = Column(JSON, default=dict)  # Data flow between jobs
	workflow_config = Column(JSON, default=dict)  # Workflow configuration
	
	# Status and Progress
	status = Column(String(20), default='draft', index=True)  # draft, ready, running, completed, failed, cancelled
	current_stage = Column(String(100), nullable=True)  # Current execution stage
	completed_jobs = Column(Integer, default=0)
	total_jobs = Column(Integer, default=0)
	progress_percentage = Column(Float, default=0.0)
	
	# Timing
	submitted_at = Column(DateTime, nullable=True)
	started_at = Column(DateTime, nullable=True)
	completed_at = Column(DateTime, nullable=True)
	estimated_completion = Column(DateTime, nullable=True)
	
	# Resource Planning
	total_cpu_hours = Column(Float, nullable=True)  # Estimated total CPU hours
	total_memory_gb_hours = Column(Float, nullable=True)  # Estimated memory usage
	total_gpu_hours = Column(Float, nullable=True)  # Estimated GPU hours
	estimated_cost = Column(Float, nullable=True)
	actual_cost = Column(Float, nullable=True)
	
	# User and Attribution
	created_by = Column(String(36), nullable=False, index=True)
	project_id = Column(String(36), nullable=True, index=True)
	billing_account = Column(String(36), nullable=True)
	
	# Metadata
	tags = Column(JSON, default=list)  # Workflow tags
	custom_metadata = Column(JSON, default=dict)  # Custom metadata
	
	def __repr__(self):
		return f"<DCWorkflow {self.workflow_name} ({self.status})>"
	
	def calculate_progress(self) -> float:
		"""Calculate workflow progress percentage"""
		if self.total_jobs == 0:
			return 0.0
		
		self.progress_percentage = (self.completed_jobs / self.total_jobs) * 100
		return self.progress_percentage
	
	def estimate_completion_time(self) -> Optional[datetime]:
		"""Estimate workflow completion time"""
		if not self.started_at or self.progress_percentage == 0:
			return None
		
		elapsed_time = datetime.utcnow() - self.started_at
		total_estimated_time = elapsed_time * (100 / self.progress_percentage)
		self.estimated_completion = self.started_at + total_estimated_time
		
		return self.estimated_completion
	
	def get_next_ready_jobs(self) -> List[str]:
		"""Get list of jobs that are ready to run"""
		ready_jobs = []
		
		# Implementation would analyze job dependencies and return
		# jobs whose dependencies have been completed
		# For now, return empty list as placeholder
		
		return ready_jobs