"""
Edge Computing Integration for Real-Time Digital Twin Processing

This module provides edge computing capabilities for digital twins with sub-10ms latency
requirements, distributed processing, and intelligent workload management.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pydantic import BaseModel, Field, ConfigDict, validator
import logging
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import socket
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

class EdgeNodeType(str, Enum):
	"""Types of edge computing nodes"""
	GATEWAY = "gateway"
	COMPUTE = "compute"
	STORAGE = "storage"
	HYBRID = "hybrid"
	IOT_BRIDGE = "iot_bridge"

class EdgeTaskPriority(str, Enum):
	"""Priority levels for edge tasks"""
	CRITICAL = "critical"		# <1ms latency requirement
	HIGH = "high"				# <5ms latency requirement
	NORMAL = "normal"			# <10ms latency requirement
	LOW = "low"					# <50ms latency requirement

class EdgeResourceType(str, Enum):
	"""Types of edge computing resources"""
	CPU = "cpu"
	MEMORY = "memory"
	STORAGE = "storage"
	NETWORK = "network"
	GPU = "gpu"
	ACCELERATOR = "accelerator"

@dataclass
class EdgeNodeCapacity:
	"""Resource capacity for an edge node"""
	cpu_cores: int
	memory_gb: float
	storage_gb: float
	network_mbps: float
	gpu_cores: int = 0
	accelerator_units: int = 0
	specialized_compute: Dict[str, int] = field(default_factory=dict)

@dataclass
class EdgeTaskRequirements:
	"""Resource requirements for an edge task"""
	cpu_cores: float
	memory_mb: float
	storage_mb: float
	network_mbps: float
	max_latency_ms: float
	gpu_required: bool = False
	accelerator_required: bool = False
	specialized_compute: Dict[str, float] = field(default_factory=dict)

class EdgeComputingNode(BaseModel):
	"""Represents an edge computing node in the network"""
	
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	id: str = Field(default_factory=lambda: str(uuid.uuid4()))
	name: str
	node_type: EdgeNodeType
	location: Dict[str, float]  # lat, lng, altitude
	capacity: Dict[str, Any]
	current_load: Dict[str, float] = Field(default_factory=dict)
	status: str = "active"
	last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
	network_latency_ms: float = 0.0
	reliability_score: float = 1.0
	energy_efficiency: float = 1.0
	specialized_capabilities: List[str] = Field(default_factory=list)
	connected_devices: List[str] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)

class EdgeTask(BaseModel):
	"""Represents a task to be executed on the edge network"""
	
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	id: str = Field(default_factory=lambda: str(uuid.uuid4()))
	twin_id: str
	task_type: str
	priority: EdgeTaskPriority
	requirements: Dict[str, Any]
	payload: Dict[str, Any]
	created_at: datetime = Field(default_factory=datetime.utcnow)
	deadline: Optional[datetime] = None
	assigned_node: Optional[str] = None
	status: str = "pending"
	result: Optional[Dict[str, Any]] = None
	execution_time_ms: Optional[float] = None
	metadata: Dict[str, Any] = Field(default_factory=dict)

class EdgeStreamProcessor:
	"""High-performance stream processor for real-time data"""
	
	def __init__(self, buffer_size: int = 10000):
		self.buffer_size = buffer_size
		self.processing_functions: Dict[str, Callable] = {}
		self.stream_buffers: Dict[str, List] = {}
		self.processing_stats: Dict[str, Dict] = {}
		self._running = False
		self._executor = ThreadPoolExecutor(max_workers=8)
	
	def register_processor(self, stream_type: str, processor_func: Callable):
		"""Register a processing function for a stream type"""
		self.processing_functions[stream_type] = processor_func
		self.stream_buffers[stream_type] = []
		self.processing_stats[stream_type] = {
			'processed': 0,
			'errors': 0,
			'avg_latency_ms': 0.0,
			'last_processed': None
		}
	
	async def process_stream_data(self, stream_type: str, data: Any) -> Dict[str, Any]:
		"""Process incoming stream data with sub-ms latency"""
		start_time = time.perf_counter()
		
		try:
			if stream_type not in self.processing_functions:
				raise ValueError(f"No processor registered for stream type: {stream_type}")
			
			# Add to buffer for batch processing if needed
			buffer = self.stream_buffers[stream_type]
			buffer.append(data)
			
			# Keep buffer size manageable
			if len(buffer) > self.buffer_size:
				buffer.pop(0)
			
			# Process the data
			processor = self.processing_functions[stream_type]
			if asyncio.iscoroutinefunction(processor):
				result = await processor(data)
			else:
				# Run sync function in executor to avoid blocking
				result = await asyncio.get_event_loop().run_in_executor(
					self._executor, processor, data
				)
			
			# Update statistics
			processing_time = (time.perf_counter() - start_time) * 1000
			stats = self.processing_stats[stream_type]
			stats['processed'] += 1
			stats['avg_latency_ms'] = (
				(stats['avg_latency_ms'] * (stats['processed'] - 1) + processing_time) / 
				stats['processed']
			)
			stats['last_processed'] = datetime.utcnow()
			
			return {
				'result': result,
				'processing_time_ms': processing_time,
				'timestamp': datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			self.processing_stats[stream_type]['errors'] += 1
			logger.error(f"Stream processing error for {stream_type}: {e}")
			return {
				'error': str(e),
				'processing_time_ms': (time.perf_counter() - start_time) * 1000,
				'timestamp': datetime.utcnow().isoformat()
			}

class EdgeComputingCluster:
	"""Manages a cluster of edge computing nodes with intelligent workload distribution"""
	
	def __init__(self):
		self.nodes: Dict[str, EdgeComputingNode] = {}
		self.tasks: Dict[str, EdgeTask] = {}
		self.task_queue: asyncio.Queue = asyncio.Queue()
		self.stream_processor = EdgeStreamProcessor()
		self.network_topology: Dict[str, List[str]] = {}  # node_id -> connected_nodes
		self.load_balancer = EdgeLoadBalancer()
		self._running = False
		self._task_executor = ThreadPoolExecutor(max_workers=16)
		
		# Performance monitoring
		self.performance_metrics: Dict[str, Dict] = {
			'cluster': {
				'total_tasks_processed': 0,
				'avg_task_latency_ms': 0.0,
				'successful_tasks': 0,
				'failed_tasks': 0,
				'nodes_active': 0,
				'network_utilization': 0.0
			}
		}
	
	async def add_node(self, node: EdgeComputingNode) -> bool:
		"""Add a new edge node to the cluster"""
		try:
			# Validate node configuration
			if not self._validate_node_config(node):
				return False
			
			# Initialize node metrics
			self.performance_metrics[node.id] = {
				'tasks_processed': 0,
				'avg_latency_ms': 0.0,
				'cpu_utilization': 0.0,
				'memory_utilization': 0.0,
				'network_utilization': 0.0,
				'uptime_hours': 0.0,
				'last_heartbeat': datetime.utcnow()
			}
			
			# Add to cluster
			self.nodes[node.id] = node
			self.network_topology[node.id] = []
			
			logger.info(f"Added edge node {node.id} ({node.name}) to cluster")
			return True
			
		except Exception as e:
			logger.error(f"Failed to add edge node: {e}")
			return False
	
	def _validate_node_config(self, node: EdgeComputingNode) -> bool:
		"""Validate edge node configuration"""
		required_capacity_keys = ['cpu_cores', 'memory_gb', 'storage_gb', 'network_mbps']
		
		if not all(key in node.capacity for key in required_capacity_keys):
			logger.error(f"Node {node.id} missing required capacity information")
			return False
		
		if any(node.capacity[key] <= 0 for key in required_capacity_keys):
			logger.error(f"Node {node.id} has invalid capacity values")
			return False
		
		return True
	
	async def submit_task(self, task: EdgeTask) -> str:
		"""Submit a task for execution on the edge network"""
		try:
			# Validate task requirements
			if not self._validate_task_requirements(task):
				raise ValueError("Invalid task requirements")
			
			# Add to task registry
			self.tasks[task.id] = task
			
			# Queue for scheduling
			await self.task_queue.put(task)
			
			logger.info(f"Submitted edge task {task.id} for twin {task.twin_id}")
			return task.id
			
		except Exception as e:
			logger.error(f"Failed to submit edge task: {e}")
			raise
	
	def _validate_task_requirements(self, task: EdgeTask) -> bool:
		"""Validate task resource requirements"""
		required_keys = ['cpu_cores', 'memory_mb', 'max_latency_ms']
		
		if not all(key in task.requirements for key in required_keys):
			logger.error(f"Task {task.id} missing required resource specifications")
			return False
		
		if task.requirements['max_latency_ms'] <= 0:
			logger.error(f"Task {task.id} has invalid latency requirement")
			return False
		
		return True
	
	async def schedule_tasks(self):
		"""Intelligent task scheduling with latency optimization"""
		while self._running:
			try:
				# Get next task from queue (with timeout to avoid blocking)
				try:
					task = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)
				except asyncio.TimeoutError:
					continue
				
				# Find optimal node for task execution
				optimal_node = await self._find_optimal_node(task)
				
				if optimal_node:
					# Assign task to node
					task.assigned_node = optimal_node.id
					task.status = "assigned"
					
					# Execute task asynchronously
					asyncio.create_task(self._execute_task(task, optimal_node))
				else:
					# No suitable node available, re-queue with delay
					task.status = "queued"
					await asyncio.sleep(0.001)  # 1ms delay
					await self.task_queue.put(task)
				
			except Exception as e:
				logger.error(f"Task scheduling error: {e}")
				await asyncio.sleep(0.01)
	
	async def _find_optimal_node(self, task: EdgeTask) -> Optional[EdgeComputingNode]:
		"""Find the optimal edge node for task execution using multi-criteria optimization"""
		if not self.nodes:
			return None
		
		best_node = None
		best_score = -1.0
		
		for node in self.nodes.values():
			if node.status != "active":
				continue
			
			# Check if node can meet resource requirements
			if not self._can_node_handle_task(node, task):
				continue
			
			# Calculate optimization score
			score = self._calculate_node_score(node, task)
			
			if score > best_score:
				best_score = score
				best_node = node
		
		return best_node
	
	def _can_node_handle_task(self, node: EdgeComputingNode, task: EdgeTask) -> bool:
		"""Check if node can handle the task requirements"""
		reqs = task.requirements
		capacity = node.capacity
		current_load = node.current_load
		
		# Check CPU availability
		available_cpu = capacity['cpu_cores'] - current_load.get('cpu_cores', 0)
		if available_cpu < reqs.get('cpu_cores', 0):
			return False
		
		# Check memory availability
		available_memory = (capacity['memory_gb'] * 1024) - current_load.get('memory_mb', 0)
		if available_memory < reqs.get('memory_mb', 0):
			return False
		
		# Check latency requirement
		if node.network_latency_ms > reqs.get('max_latency_ms', float('inf')):
			return False
		
		# Check specialized compute requirements
		if reqs.get('gpu_required', False) and capacity.get('gpu_cores', 0) == 0:
			return False
		
		return True
	
	def _calculate_node_score(self, node: EdgeComputingNode, task: EdgeTask) -> float:
		"""Calculate optimization score for node selection"""
		# Factors: latency (50%), resource availability (30%), reliability (20%)
		
		# Latency score (lower is better)
		max_acceptable_latency = task.requirements.get('max_latency_ms', 100)
		latency_score = max(0, 1.0 - (node.network_latency_ms / max_acceptable_latency))
		
		# Resource availability score
		cpu_util = node.current_load.get('cpu_utilization', 0) / 100.0
		memory_util = node.current_load.get('memory_utilization', 0) / 100.0
		resource_score = 1.0 - ((cpu_util + memory_util) / 2.0)
		
		# Reliability score
		reliability_score = node.reliability_score
		
		# Energy efficiency bonus
		efficiency_bonus = node.energy_efficiency * 0.1
		
		# Weighted final score
		total_score = (
			latency_score * 0.5 + 
			resource_score * 0.3 + 
			reliability_score * 0.2 + 
			efficiency_bonus
		)
		
		return total_score
	
	async def _execute_task(self, task: EdgeTask, node: EdgeComputingNode):
		"""Execute a task on the assigned edge node"""
		start_time = time.perf_counter()
		
		try:
			task.status = "executing"
			
			# Simulate task execution based on type
			result = await self._simulate_task_execution(task, node)
			
			# Calculate execution time
			execution_time_ms = (time.perf_counter() - start_time) * 1000
			
			# Update task with results
			task.result = result
			task.execution_time_ms = execution_time_ms
			task.status = "completed"
			
			# Update node load (simulate resource release)
			await self._update_node_load(node, task, release=True)
			
			# Update performance metrics
			self._update_performance_metrics(task, node, execution_time_ms, success=True)
			
			logger.info(f"Task {task.id} completed on node {node.id} in {execution_time_ms:.2f}ms")
			
		except Exception as e:
			task.status = "failed"
			task.result = {"error": str(e)}
			execution_time_ms = (time.perf_counter() - start_time) * 1000
			task.execution_time_ms = execution_time_ms
			
			self._update_performance_metrics(task, node, execution_time_ms, success=False)
			logger.error(f"Task {task.id} failed on node {node.id}: {e}")
	
	async def _simulate_task_execution(self, task: EdgeTask, node: EdgeComputingNode) -> Dict[str, Any]:
		"""Simulate task execution with realistic processing"""
		task_type = task.task_type
		payload = task.payload
		
		# Update node load (simulate resource allocation)
		await self._update_node_load(node, task, release=False)
		
		if task_type == "sensor_data_processing":
			# Process sensor data with minimal latency
			await asyncio.sleep(0.001)  # 1ms processing time
			return {
				"processed_readings": len(payload.get('readings', [])),
				"anomalies_detected": 0,
				"processing_node": node.id,
				"timestamp": datetime.utcnow().isoformat()
			}
		
		elif task_type == "predictive_analysis":
			# Run predictive model
			await asyncio.sleep(0.005)  # 5ms processing time
			return {
				"prediction": "normal_operation",
				"confidence": 0.95,
				"model_version": "v2.1",
				"processing_node": node.id,
				"timestamp": datetime.utcnow().isoformat()
			}
		
		elif task_type == "real_time_control":
			# Critical control task
			await asyncio.sleep(0.0005)  # 0.5ms processing time
			return {
				"control_signal": payload.get('target_value', 0),
				"adjustment": 0.1,
				"safety_check": "passed",
				"processing_node": node.id,
				"timestamp": datetime.utcnow().isoformat()
			}
		
		elif task_type == "stream_analytics":
			# Stream processing
			stream_data = payload.get('stream_data', [])
			result = await self.stream_processor.process_stream_data(
				payload.get('stream_type', 'default'), 
				stream_data
			)
			return result
		
		else:
			# Generic task processing
			await asyncio.sleep(0.01)  # 10ms default processing time
			return {
				"status": "processed",
				"task_type": task_type,
				"processing_node": node.id,
				"timestamp": datetime.utcnow().isoformat()
			}
	
	async def _update_node_load(self, node: EdgeComputingNode, task: EdgeTask, release: bool = False):
		"""Update node resource load based on task allocation/release"""
		reqs = task.requirements
		
		multiplier = -1 if release else 1
		
		# Update CPU load
		current_cpu = node.current_load.get('cpu_cores', 0)
		node.current_load['cpu_cores'] = max(0, current_cpu + (reqs.get('cpu_cores', 0) * multiplier))
		
		# Update memory load
		current_memory = node.current_load.get('memory_mb', 0)
		node.current_load['memory_mb'] = max(0, current_memory + (reqs.get('memory_mb', 0) * multiplier))
		
		# Calculate utilization percentages
		node.current_load['cpu_utilization'] = (
			node.current_load['cpu_cores'] / node.capacity['cpu_cores']
		) * 100
		
		node.current_load['memory_utilization'] = (
			node.current_load['memory_mb'] / (node.capacity['memory_gb'] * 1024)
		) * 100
	
	def _update_performance_metrics(self, task: EdgeTask, node: EdgeComputingNode, 
									execution_time_ms: float, success: bool):
		"""Update performance metrics for cluster and node"""
		# Update cluster metrics
		cluster_metrics = self.performance_metrics['cluster']
		cluster_metrics['total_tasks_processed'] += 1
		
		if success:
			cluster_metrics['successful_tasks'] += 1
		else:
			cluster_metrics['failed_tasks'] += 1
		
		# Update average latency
		total_tasks = cluster_metrics['total_tasks_processed']
		current_avg = cluster_metrics['avg_task_latency_ms']
		cluster_metrics['avg_task_latency_ms'] = (
			(current_avg * (total_tasks - 1) + execution_time_ms) / total_tasks
		)
		
		# Update node metrics
		if node.id in self.performance_metrics:
			node_metrics = self.performance_metrics[node.id]
			node_metrics['tasks_processed'] += 1
			
			# Update node average latency
			node_total = node_metrics['tasks_processed']
			node_current_avg = node_metrics['avg_latency_ms']
			node_metrics['avg_latency_ms'] = (
				(node_current_avg * (node_total - 1) + execution_time_ms) / node_total
			)
			
			# Update resource utilization
			node_metrics['cpu_utilization'] = node.current_load.get('cpu_utilization', 0)
			node_metrics['memory_utilization'] = node.current_load.get('memory_utilization', 0)
			node_metrics['last_heartbeat'] = datetime.utcnow()
	
	async def start_cluster(self):
		"""Start the edge computing cluster"""
		self._running = True
		
		# Start task scheduler
		scheduler_task = asyncio.create_task(self.schedule_tasks())
		
		# Start heartbeat monitoring
		heartbeat_task = asyncio.create_task(self._monitor_node_heartbeats())
		
		logger.info("Edge computing cluster started")
		
		return scheduler_task, heartbeat_task
	
	async def stop_cluster(self):
		"""Stop the edge computing cluster"""
		self._running = False
		self._task_executor.shutdown(wait=True)
		logger.info("Edge computing cluster stopped")
	
	async def _monitor_node_heartbeats(self):
		"""Monitor node health with heartbeat checking"""
		while self._running:
			try:
				current_time = datetime.utcnow()
				inactive_nodes = []
				
				for node_id, node in self.nodes.items():
					# Check if node hasn't sent heartbeat in last 30 seconds
					time_since_heartbeat = current_time - node.last_heartbeat
					if time_since_heartbeat > timedelta(seconds=30):
						if node.status == "active":
							node.status = "inactive"
							inactive_nodes.append(node_id)
							logger.warning(f"Node {node_id} marked as inactive")
				
				# Update cluster active node count
				active_nodes = sum(1 for node in self.nodes.values() if node.status == "active")
				self.performance_metrics['cluster']['nodes_active'] = active_nodes
				
				await asyncio.sleep(10)  # Check every 10 seconds
				
			except Exception as e:
				logger.error(f"Heartbeat monitoring error: {e}")
				await asyncio.sleep(10)
	
	def get_cluster_status(self) -> Dict[str, Any]:
		"""Get comprehensive cluster status"""
		return {
			"nodes": {
				"total": len(self.nodes),
				"active": sum(1 for n in self.nodes.values() if n.status == "active"),
				"inactive": sum(1 for n in self.nodes.values() if n.status == "inactive")
			},
			"tasks": {
				"total": len(self.tasks),
				"pending": sum(1 for t in self.tasks.values() if t.status == "pending"),
				"executing": sum(1 for t in self.tasks.values() if t.status == "executing"),
				"completed": sum(1 for t in self.tasks.values() if t.status == "completed"),
				"failed": sum(1 for t in self.tasks.values() if t.status == "failed")
			},
			"performance": self.performance_metrics,
			"queue_size": self.task_queue.qsize(),
			"timestamp": datetime.utcnow().isoformat()
		}

class EdgeLoadBalancer:
	"""Advanced load balancer for edge computing workloads"""
	
	def __init__(self):
		self.load_history: Dict[str, List[float]] = {}
		self.prediction_weights = [0.5, 0.3, 0.2]  # Recent, medium, older history
	
	def predict_node_load(self, node_id: str, current_load: float) -> float:
		"""Predict future node load based on historical patterns"""
		if node_id not in self.load_history:
			self.load_history[node_id] = []
		
		# Add current load to history
		history = self.load_history[node_id]
		history.append(current_load)
		
		# Keep only last 100 measurements
		if len(history) > 100:
			history.pop(0)
		
		# Predict based on weighted average of recent measurements
		if len(history) < 3:
			return current_load
		
		recent_loads = history[-3:]
		predicted_load = sum(
			load * weight for load, weight in zip(recent_loads, self.prediction_weights)
		)
		
		return predicted_load
	
	def calculate_load_distribution_score(self, nodes: List[EdgeComputingNode]) -> float:
		"""Calculate how well load is distributed across nodes"""
		if not nodes:
			return 0.0
		
		loads = [node.current_load.get('cpu_utilization', 0) for node in nodes]
		
		# Calculate coefficient of variation (lower is better distribution)
		if not loads or all(load == 0 for load in loads):
			return 1.0
		
		mean_load = sum(loads) / len(loads)
		variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
		std_dev = variance ** 0.5
		
		if mean_load == 0:
			return 1.0
		
		cv = std_dev / mean_load
		
		# Convert to score (1.0 = perfect distribution, 0.0 = worst)
		return max(0.0, min(1.0, 1.0 - cv))

# Integration with existing digital twin system
class EdgeEnabledDigitalTwin:
	"""Digital twin with integrated edge computing capabilities"""
	
	def __init__(self, twin_id: str):
		self.twin_id = twin_id
		self.edge_cluster = EdgeComputingCluster()
		self.real_time_processors: Dict[str, Callable] = {}
		self.edge_tasks: List[str] = []
	
	async def register_real_time_processor(self, data_type: str, processor: Callable):
		"""Register a real-time processor for specific data types"""
		self.real_time_processors[data_type] = processor
		self.edge_cluster.stream_processor.register_processor(data_type, processor)
	
	async def process_real_time_data(self, data_type: str, data: Any, 
									 max_latency_ms: float = 10.0) -> Dict[str, Any]:
		"""Process data with real-time edge computing"""
		
		# Create edge task for real-time processing
		task = EdgeTask(
			twin_id=self.twin_id,
			task_type="stream_analytics",
			priority=EdgeTaskPriority.HIGH if max_latency_ms <= 5 else EdgeTaskPriority.NORMAL,
			requirements={
				"cpu_cores": 0.5,
				"memory_mb": 128,
				"max_latency_ms": max_latency_ms
			},
			payload={
				"stream_type": data_type,
				"stream_data": data
			}
		)
		
		# Submit task and wait for completion
		task_id = await self.edge_cluster.submit_task(task)
		self.edge_tasks.append(task_id)
		
		# Wait for task completion with timeout
		timeout = max_latency_ms / 1000.0  # Convert to seconds
		start_time = time.time()
		
		while time.time() - start_time < timeout:
			if task.status == "completed":
				return task.result
			elif task.status == "failed":
				raise RuntimeError(f"Edge processing failed: {task.result}")
			
			await asyncio.sleep(0.001)  # 1ms polling interval
		
		raise TimeoutError(f"Edge processing exceeded {max_latency_ms}ms deadline")
	
	async def deploy_to_edge(self, edge_nodes: List[EdgeComputingNode]):
		"""Deploy this digital twin to edge computing nodes"""
		for node in edge_nodes:
			await self.edge_cluster.add_node(node)
		
		# Start the edge cluster
		await self.edge_cluster.start_cluster()
		
		logger.info(f"Deployed digital twin {self.twin_id} to {len(edge_nodes)} edge nodes")
	
	def get_edge_performance_metrics(self) -> Dict[str, Any]:
		"""Get comprehensive edge computing performance metrics"""
		cluster_status = self.edge_cluster.get_cluster_status()
		
		return {
			"twin_id": self.twin_id,
			"edge_cluster": cluster_status,
			"real_time_processors": list(self.real_time_processors.keys()),
			"total_edge_tasks": len(self.edge_tasks),
			"timestamp": datetime.utcnow().isoformat()
		}

def _log_edge_task_submission(twin_id: str, task_type: str, priority: str) -> str:
	"""Log edge task submission for monitoring"""
	return f"[EDGE] Twin {twin_id}: Submitted {task_type} task (Priority: {priority})"

def _log_edge_node_status(node_id: str, status: str, load: float) -> str:
	"""Log edge node status for monitoring"""
	return f"[EDGE] Node {node_id}: Status={status}, Load={load:.1f}%"

def _log_cluster_performance(total_tasks: int, avg_latency: float, active_nodes: int) -> str:
	"""Log cluster performance metrics"""
	return f"[EDGE] Cluster: {total_tasks} tasks, {avg_latency:.2f}ms avg latency, {active_nodes} active nodes"

# Example usage and testing
async def create_sample_edge_cluster():
	"""Create a sample edge computing cluster for testing"""
	cluster = EdgeComputingCluster()
	
	# Create sample edge nodes
	gateway_node = EdgeComputingNode(
		name="Gateway-001",
		node_type=EdgeNodeType.GATEWAY,
		location={"lat": 37.7749, "lng": -122.4194, "altitude": 50},
		capacity={
			"cpu_cores": 8,
			"memory_gb": 16,
			"storage_gb": 500,
			"network_mbps": 1000,
			"gpu_cores": 0
		},
		network_latency_ms=2.0,
		specialized_capabilities=["sensor_aggregation", "protocol_translation"]
	)
	
	compute_node = EdgeComputingNode(
		name="Compute-001",
		node_type=EdgeNodeType.COMPUTE,
		location={"lat": 37.7849, "lng": -122.4094, "altitude": 100},
		capacity={
			"cpu_cores": 16,
			"memory_gb": 32,
			"storage_gb": 1000,
			"network_mbps": 2000,
			"gpu_cores": 4
		},
		network_latency_ms=5.0,
		specialized_capabilities=["ml_inference", "stream_processing"]
	)
	
	# Add nodes to cluster
	await cluster.add_node(gateway_node)
	await cluster.add_node(compute_node)
	
	return cluster

if __name__ == "__main__":
	# Test the edge computing system
	async def test_edge_computing():
		cluster = await create_sample_edge_cluster()
		
		# Start cluster
		scheduler_task, heartbeat_task = await cluster.start_cluster()
		
		# Create test tasks
		tasks = []
		for i in range(5):
			task = EdgeTask(
				twin_id=f"twin_{i}",
				task_type="sensor_data_processing",
				priority=EdgeTaskPriority.HIGH,
				requirements={
					"cpu_cores": 1.0,
					"memory_mb": 256,
					"max_latency_ms": 5.0
				},
				payload={
					"readings": [{"sensor_id": f"sensor_{j}", "value": j * 10} for j in range(10)]
				}
			)
			tasks.append(task)
		
		# Submit tasks
		task_ids = []
		for task in tasks:
			task_id = await cluster.submit_task(task)
			task_ids.append(task_id)
		
		# Wait for completion
		await asyncio.sleep(1)
		
		# Check results
		print("Edge Computing Test Results:")
		status = cluster.get_cluster_status()
		print(json.dumps(status, indent=2, default=str))
		
		# Stop cluster
		await cluster.stop_cluster()
	
	# Run test
	asyncio.run(test_edge_computing())