"""
Comprehensive tests for edge computing capabilities

This test suite verifies the functionality of edge computing integration,
including node management, task scheduling, and performance optimization.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

from capabilities.edge_computing import (
	EdgeComputingCluster,
	EdgeComputingNode,
	EdgeTask,
	EdgeNodeType,
	EdgeTaskPriority,
	EdgeStreamProcessor,
	EdgeEnabledDigitalTwin,
	EdgeLoadBalancer
)

class TestEdgeComputingNode:
	"""Test edge computing node functionality"""
	
	def test_node_creation(self):
		"""Test edge node creation and validation"""
		node = EdgeComputingNode(
			name="Test-Node-01",
			node_type=EdgeNodeType.COMPUTE,
			location={"lat": 37.7749, "lng": -122.4194, "altitude": 100},
			capacity={
				"cpu_cores": 8,
				"memory_gb": 16,
				"storage_gb": 500,
				"network_mbps": 1000
			},
			network_latency_ms=5.0,
			reliability_score=0.95
		)
		
		assert node.name == "Test-Node-01"
		assert node.node_type == EdgeNodeType.COMPUTE
		assert node.location["lat"] == 37.7749
		assert node.capacity["cpu_cores"] == 8
		assert node.network_latency_ms == 5.0
		assert node.reliability_score == 0.95
		assert node.status == "active"
	
	def test_node_capacity_validation(self):
		"""Test node capacity validation"""
		# Valid node should work
		valid_node = EdgeComputingNode(
			name="Valid-Node",
			node_type=EdgeNodeType.GATEWAY,
			location={"lat": 0, "lng": 0, "altitude": 0},
			capacity={
				"cpu_cores": 4,
				"memory_gb": 8,
				"storage_gb": 100,
				"network_mbps": 500
			}
		)
		assert valid_node.capacity["cpu_cores"] == 4
	
	def test_specialized_capabilities(self):
		"""Test node specialized capabilities"""
		node = EdgeComputingNode(
			name="Specialized-Node",
			node_type=EdgeNodeType.HYBRID,
			location={"lat": 0, "lng": 0, "altitude": 0},
			capacity={
				"cpu_cores": 16,
				"memory_gb": 32,
				"storage_gb": 1000,
				"network_mbps": 2000,
				"gpu_cores": 8
			},
			specialized_capabilities=["ml_inference", "gpu_compute", "video_processing"]
		)
		
		assert "ml_inference" in node.specialized_capabilities
		assert "gpu_compute" in node.specialized_capabilities
		assert len(node.specialized_capabilities) == 3

class TestEdgeTask:
	"""Test edge computing task functionality"""
	
	def test_task_creation(self):
		"""Test edge task creation"""
		task = EdgeTask(
			twin_id="twin_test_01",
			task_type="sensor_data_processing",
			priority=EdgeTaskPriority.HIGH,
			requirements={
				"cpu_cores": 2.0,
				"memory_mb": 512,
				"max_latency_ms": 5.0
			},
			payload={
				"sensor_readings": [1, 2, 3, 4, 5]
			}
		)
		
		assert task.twin_id == "twin_test_01"
		assert task.task_type == "sensor_data_processing"
		assert task.priority == EdgeTaskPriority.HIGH
		assert task.requirements["cpu_cores"] == 2.0
		assert task.status == "pending"
		assert len(task.payload["sensor_readings"]) == 5
	
	def test_task_requirements_validation(self):
		"""Test task requirements validation"""
		task = EdgeTask(
			twin_id="twin_test_02",
			task_type="predictive_analysis",
			priority=EdgeTaskPriority.NORMAL,
			requirements={
				"cpu_cores": 1.5,
				"memory_mb": 256,
				"max_latency_ms": 10.0,
				"gpu_required": True
			},
			payload={"model_input": [1.0, 2.0, 3.0]}
		)
		
		assert task.requirements["gpu_required"] is True
		assert task.requirements["max_latency_ms"] == 10.0
	
	def test_task_priority_levels(self):
		"""Test different task priority levels"""
		critical_task = EdgeTask(
			twin_id="twin_critical",
			task_type="real_time_control",
			priority=EdgeTaskPriority.CRITICAL,
			requirements={
				"cpu_cores": 0.5,
				"memory_mb": 64,
				"max_latency_ms": 1.0
			},
			payload={"control_signal": 42}
		)
		
		low_task = EdgeTask(
			twin_id="twin_low",
			task_type="data_aggregation",
			priority=EdgeTaskPriority.LOW,
			requirements={
				"cpu_cores": 0.25,
				"memory_mb": 128,
				"max_latency_ms": 50.0
			},
			payload={"batch_data": []}
		)
		
		assert critical_task.priority == EdgeTaskPriority.CRITICAL
		assert low_task.priority == EdgeTaskPriority.LOW
		assert critical_task.requirements["max_latency_ms"] < low_task.requirements["max_latency_ms"]

class TestEdgeStreamProcessor:
	"""Test edge stream processing functionality"""
	
	@pytest.fixture
	def stream_processor(self):
		"""Create stream processor for testing"""
		return EdgeStreamProcessor(buffer_size=1000)
	
	def test_processor_registration(self, stream_processor):
		"""Test stream processor registration"""
		def sample_processor(data):
			return {"processed": len(data) if isinstance(data, list) else 1}
		
		stream_processor.register_processor("test_stream", sample_processor)
		
		assert "test_stream" in stream_processor.processing_functions
		assert "test_stream" in stream_processor.stream_buffers
		assert "test_stream" in stream_processor.processing_stats
	
	@pytest.mark.asyncio
	async def test_stream_data_processing(self, stream_processor):
		"""Test stream data processing"""
		def sample_processor(data):
			return {"count": len(data), "sum": sum(data)}
		
		stream_processor.register_processor("numeric_stream", sample_processor)
		
		# Process test data
		test_data = [1, 2, 3, 4, 5]
		result = await stream_processor.process_stream_data("numeric_stream", test_data)
		
		assert "result" in result
		assert result["result"]["count"] == 5
		assert result["result"]["sum"] == 15
		assert "processing_time_ms" in result
		assert result["processing_time_ms"] >= 0
	
	@pytest.mark.asyncio
	async def test_async_processor(self, stream_processor):
		"""Test asynchronous stream processor"""
		async def async_processor(data):
			await asyncio.sleep(0.001)  # Simulate async processing
			return {"async_processed": True, "data_length": len(data) if hasattr(data, '__len__') else 0}
		
		stream_processor.register_processor("async_stream", async_processor)
		
		result = await stream_processor.process_stream_data("async_stream", "test_data")
		
		assert result["result"]["async_processed"] is True
		assert "processing_time_ms" in result
	
	@pytest.mark.asyncio
	async def test_processing_error_handling(self, stream_processor):
		"""Test error handling in stream processing"""
		def error_processor(data):
			raise ValueError("Simulated processing error")
		
		stream_processor.register_processor("error_stream", error_processor)
		
		result = await stream_processor.process_stream_data("error_stream", "test_data")
		
		assert "error" in result
		assert "Simulated processing error" in result["error"]
		assert "processing_time_ms" in result
	
	@pytest.mark.asyncio
	async def test_processing_statistics(self, stream_processor):
		"""Test processing statistics tracking"""
		def stats_processor(data):
			return {"processed": True}
		
		stream_processor.register_processor("stats_stream", stats_processor)
		
		# Process multiple items
		for i in range(5):
			await stream_processor.process_stream_data("stats_stream", f"data_{i}")
		
		stats = stream_processor.processing_stats["stats_stream"]
		assert stats["processed"] == 5
		assert stats["errors"] == 0
		assert stats["avg_latency_ms"] > 0
		assert stats["last_processed"] is not None

class TestEdgeComputingCluster:
	"""Test edge computing cluster functionality"""
	
	@pytest.fixture
	async def cluster(self):
		"""Create edge computing cluster for testing"""
		cluster = EdgeComputingCluster()
		yield cluster
		await cluster.stop_cluster()
	
	@pytest.fixture
	def sample_node(self):
		"""Create sample edge node for testing"""
		return EdgeComputingNode(
			name="Test-Cluster-Node",
			node_type=EdgeNodeType.COMPUTE,
			location={"lat": 37.7749, "lng": -122.4194, "altitude": 50},
			capacity={
				"cpu_cores": 8,
				"memory_gb": 16,
				"storage_gb": 500,
				"network_mbps": 1000
			},
			network_latency_ms=3.0,
			reliability_score=0.98
		)
	
	@pytest.fixture
	def sample_task(self):
		"""Create sample edge task for testing"""
		return EdgeTask(
			twin_id="twin_cluster_test",
			task_type="sensor_data_processing",
			priority=EdgeTaskPriority.HIGH,
			requirements={
				"cpu_cores": 1.0,
				"memory_mb": 256,
				"max_latency_ms": 5.0
			},
			payload={
				"readings": [{"sensor_id": "temp_01", "value": 23.5}]
			}
		)
	
	@pytest.mark.asyncio
	async def test_cluster_node_management(self, cluster, sample_node):
		"""Test adding and managing nodes in cluster"""
		# Add node to cluster
		success = await cluster.add_node(sample_node)
		assert success is True
		assert sample_node.id in cluster.nodes
		
		# Check node metrics were initialized
		assert sample_node.id in cluster.performance_metrics
		assert cluster.performance_metrics[sample_node.id]["tasks_processed"] == 0
	
	@pytest.mark.asyncio
	async def test_cluster_task_submission(self, cluster, sample_node, sample_task):
		"""Test task submission to cluster"""
		# Add node first
		await cluster.add_node(sample_node)
		
		# Submit task
		task_id = await cluster.submit_task(sample_task)
		assert task_id == sample_task.id
		assert sample_task.id in cluster.tasks
	
	@pytest.mark.asyncio
	async def test_cluster_task_scheduling(self, cluster, sample_node, sample_task):
		"""Test task scheduling and execution"""
		# Add node and submit task
		await cluster.add_node(sample_node)
		await cluster.submit_task(sample_task)
		
		# Start cluster to begin scheduling
		cluster._running = True
		scheduler_task = asyncio.create_task(cluster.schedule_tasks())
		
		# Give some time for scheduling
		await asyncio.sleep(0.1)
		
		# Stop scheduling
		cluster._running = False
		scheduler_task.cancel()
		
		# Check task was processed
		task = cluster.tasks[sample_task.id]
		assert task.status in ["assigned", "executing", "completed"]
	
	@pytest.mark.asyncio
	async def test_node_selection_algorithm(self, cluster):
		"""Test optimal node selection algorithm"""
		# Create nodes with different characteristics
		fast_node = EdgeComputingNode(
			name="Fast-Node",
			node_type=EdgeNodeType.COMPUTE,
			location={"lat": 0, "lng": 0, "altitude": 0},
			capacity={"cpu_cores": 16, "memory_gb": 32, "storage_gb": 1000, "network_mbps": 2000},
			network_latency_ms=1.0,
			reliability_score=0.99
		)
		
		slow_node = EdgeComputingNode(
			name="Slow-Node",
			node_type=EdgeNodeType.GATEWAY,
			location={"lat": 0, "lng": 0, "altitude": 0},
			capacity={"cpu_cores": 4, "memory_gb": 8, "storage_gb": 200, "network_mbps": 500},
			network_latency_ms=10.0,
			reliability_score=0.85
		)
		
		await cluster.add_node(fast_node)
		await cluster.add_node(slow_node)
		
		# Create high-priority task requiring low latency
		urgent_task = EdgeTask(
			twin_id="twin_urgent",
			task_type="real_time_control",
			priority=EdgeTaskPriority.CRITICAL,
			requirements={
				"cpu_cores": 2.0,
				"memory_mb": 512,
				"max_latency_ms": 2.0
			},
			payload={"control_value": 100}
		)
		
		# Find optimal node
		optimal_node = await cluster._find_optimal_node(urgent_task)
		
		# Should select the fast node due to latency requirement
		assert optimal_node is not None
		assert optimal_node.id == fast_node.id
	
	@pytest.mark.asyncio
	async def test_resource_capacity_checking(self, cluster):
		"""Test resource capacity validation"""
		# Create node with limited resources
		limited_node = EdgeComputingNode(
			name="Limited-Node",
			node_type=EdgeNodeType.COMPUTE,
			location={"lat": 0, "lng": 0, "altitude": 0},
			capacity={"cpu_cores": 2, "memory_gb": 4, "storage_gb": 100, "network_mbps": 100}
		)
		
		await cluster.add_node(limited_node)
		
		# Create task requiring more resources than available
		heavy_task = EdgeTask(
			twin_id="twin_heavy",
			task_type="ml_inference",
			priority=EdgeTaskPriority.NORMAL,
			requirements={
				"cpu_cores": 4.0,  # More than node capacity
				"memory_mb": 2048,
				"max_latency_ms": 100.0
			},
			payload={"model_data": "large_model"}
		)
		
		# Should not be able to handle this task
		can_handle = cluster._can_node_handle_task(limited_node, heavy_task)
		assert can_handle is False
	
	@pytest.mark.asyncio
	async def test_load_balancing(self, cluster):
		"""Test load balancing across multiple nodes"""
		# Create multiple nodes
		nodes = []
		for i in range(3):
			node = EdgeComputingNode(
				name=f"LoadBalance-Node-{i}",
				node_type=EdgeNodeType.COMPUTE,
				location={"lat": 0, "lng": 0, "altitude": 0},
				capacity={"cpu_cores": 8, "memory_gb": 16, "storage_gb": 500, "network_mbps": 1000},
				network_latency_ms=5.0
			)
			nodes.append(node)
			await cluster.add_node(node)
		
		# Submit multiple tasks
		tasks = []
		for i in range(6):
			task = EdgeTask(
				twin_id=f"twin_lb_{i}",
				task_type="data_processing",
				priority=EdgeTaskPriority.NORMAL,
				requirements={
					"cpu_cores": 1.0,
					"memory_mb": 256,
					"max_latency_ms": 10.0
				},
				payload={"data": f"batch_{i}"}
			)
			tasks.append(task)
			await cluster.submit_task(task)
		
		# Start cluster briefly to process tasks
		cluster._running = True
		scheduler_task = asyncio.create_task(cluster.schedule_tasks())
		await asyncio.sleep(0.2)
		cluster._running = False
		scheduler_task.cancel()
		
		# Check that tasks were distributed across nodes
		assigned_nodes = set()
		for task in tasks:
			if cluster.tasks[task.id].assigned_node:
				assigned_nodes.add(cluster.tasks[task.id].assigned_node)
		
		# Should have used multiple nodes for load balancing
		assert len(assigned_nodes) > 1
	
	def test_cluster_status_reporting(self, cluster):
		"""Test cluster status reporting"""
		status = cluster.get_cluster_status()
		
		assert "nodes" in status
		assert "tasks" in status
		assert "performance" in status
		assert "queue_size" in status
		assert "timestamp" in status
		
		assert status["nodes"]["total"] == 0  # No nodes added yet
		assert status["tasks"]["total"] == 0  # No tasks submitted yet

class TestEdgeLoadBalancer:
	"""Test edge load balancer functionality"""
	
	@pytest.fixture
	def load_balancer(self):
		"""Create load balancer for testing"""
		return EdgeLoadBalancer()
	
	def test_load_prediction(self, load_balancer):
		"""Test load prediction based on history"""
		node_id = "test_node_lb"
		
		# Add historical load data
		loads = [50.0, 55.0, 60.0]
		for load in loads:
			predicted = load_balancer.predict_node_load(node_id, load)
			# First few predictions should equal current load
		
		# After enough history, prediction should be weighted average
		history = load_balancer.load_history[node_id]
		assert len(history) == 3
		assert history == loads
	
	def test_load_distribution_scoring(self, load_balancer):
		"""Test load distribution scoring"""
		# Create nodes with different load levels
		balanced_nodes = []
		for i in range(3):
			node = EdgeComputingNode(
				name=f"Balanced-{i}",
				node_type=EdgeNodeType.COMPUTE,
				location={"lat": 0, "lng": 0, "altitude": 0},
				capacity={"cpu_cores": 8, "memory_gb": 16, "storage_gb": 500, "network_mbps": 1000}
			)
			# Set similar load levels for good distribution
			node.current_load = {"cpu_utilization": 50.0 + i * 5}
			balanced_nodes.append(node)
		
		unbalanced_nodes = []
		for i, load in enumerate([10.0, 90.0]):
			node = EdgeComputingNode(
				name=f"Unbalanced-{i}",
				node_type=EdgeNodeType.COMPUTE,
				location={"lat": 0, "lng": 0, "altitude": 0},
				capacity={"cpu_cores": 8, "memory_gb": 16, "storage_gb": 500, "network_mbps": 1000}
			)
			node.current_load = {"cpu_utilization": load}
			unbalanced_nodes.append(node)
		
		balanced_score = load_balancer.calculate_load_distribution_score(balanced_nodes)
		unbalanced_score = load_balancer.calculate_load_distribution_score(unbalanced_nodes)
		
		# Balanced distribution should have higher score
		assert balanced_score > unbalanced_score

class TestEdgeEnabledDigitalTwin:
	"""Test edge-enabled digital twin functionality"""
	
	@pytest.fixture
	async def edge_twin(self):
		"""Create edge-enabled digital twin for testing"""
		twin = EdgeEnabledDigitalTwin("twin_edge_test")
		yield twin
		await twin.edge_cluster.stop_cluster()
	
	@pytest.fixture
	def edge_node(self):
		"""Create edge node for twin testing"""
		return EdgeComputingNode(
			name="Twin-Edge-Node",
			node_type=EdgeNodeType.HYBRID,
			location={"lat": 37.7749, "lng": -122.4194, "altitude": 75},
			capacity={
				"cpu_cores": 12,
				"memory_gb": 24,
				"storage_gb": 750,
				"network_mbps": 1500,
				"gpu_cores": 2
			},
			network_latency_ms=2.5,
			specialized_capabilities=["real_time_processing", "ml_inference"]
		)
	
	@pytest.mark.asyncio
	async def test_real_time_processor_registration(self, edge_twin):
		"""Test registering real-time processors"""
		def temperature_processor(data):
			return {"avg_temp": sum(data) / len(data) if data else 0}
		
		await edge_twin.register_real_time_processor("temperature", temperature_processor)
		
		assert "temperature" in edge_twin.real_time_processors
		assert "temperature" in edge_twin.edge_cluster.stream_processor.processing_functions
	
	@pytest.mark.asyncio
	async def test_real_time_data_processing(self, edge_twin, edge_node):
		"""Test real-time data processing through edge network"""
		# Deploy twin to edge
		await edge_twin.deploy_to_edge([edge_node])
		
		# Register processor
		def sensor_processor(data):
			return {"processed_readings": len(data), "anomalies": 0}
		
		await edge_twin.register_real_time_processor("sensor_data", sensor_processor)
		
		# Process real-time data
		sensor_readings = [
			{"sensor_id": "temp_01", "value": 23.5},
			{"sensor_id": "temp_02", "value": 24.1},
			{"sensor_id": "temp_03", "value": 22.8}
		]
		
		result = await edge_twin.process_real_time_data("sensor_data", sensor_readings, max_latency_ms=10.0)
		
		assert "result" in result
		assert result["result"]["processed_readings"] == 3
	
	@pytest.mark.asyncio
	async def test_edge_deployment(self, edge_twin, edge_node):
		"""Test deploying digital twin to edge nodes"""
		# Deploy to edge
		await edge_twin.deploy_to_edge([edge_node])
		
		# Check node was added to cluster
		assert edge_node.id in edge_twin.edge_cluster.nodes
		
		# Check cluster is running
		assert edge_twin.edge_cluster._running is True
	
	@pytest.mark.asyncio
	async def test_performance_metrics_collection(self, edge_twin, edge_node):
		"""Test edge performance metrics collection"""
		# Deploy and process some data
		await edge_twin.deploy_to_edge([edge_node])
		
		# Get performance metrics
		metrics = edge_twin.get_edge_performance_metrics()
		
		assert "twin_id" in metrics
		assert metrics["twin_id"] == "twin_edge_test"
		assert "edge_cluster" in metrics
		assert "real_time_processors" in metrics
		assert "total_edge_tasks" in metrics
	
	@pytest.mark.asyncio
	async def test_latency_timeout_handling(self, edge_twin, edge_node):
		"""Test handling of latency timeout scenarios"""
		# Create slow node with high latency
		slow_node = EdgeComputingNode(
			name="Slow-Edge-Node",
			node_type=EdgeNodeType.GATEWAY,
			location={"lat": 0, "lng": 0, "altitude": 0},
			capacity={"cpu_cores": 2, "memory_gb": 4, "storage_gb": 100, "network_mbps": 100},
			network_latency_ms=100.0  # Very high latency
		)
		
		await edge_twin.deploy_to_edge([slow_node])
		
		def slow_processor(data):
			time.sleep(0.1)  # Simulate slow processing
			return {"processed": True}
		
		await edge_twin.register_real_time_processor("slow_stream", slow_processor)
		
		# Try to process with very strict latency requirement
		with pytest.raises(TimeoutError):
			await edge_twin.process_real_time_data("slow_stream", [1, 2, 3], max_latency_ms=1.0)

class TestEdgeComputingIntegration:
	"""Integration tests for complete edge computing system"""
	
	@pytest.mark.asyncio
	async def test_end_to_end_workflow(self):
		"""Test complete end-to-end edge computing workflow"""
		# Create edge cluster
		cluster = EdgeComputingCluster()
		
		# Create and add multiple nodes with different capabilities
		gateway_node = EdgeComputingNode(
			name="Gateway-Integration",
			node_type=EdgeNodeType.GATEWAY,
			location={"lat": 37.7749, "lng": -122.4194, "altitude": 50},
			capacity={"cpu_cores": 4, "memory_gb": 8, "storage_gb": 200, "network_mbps": 500},
			network_latency_ms=1.0,
			specialized_capabilities=["data_aggregation", "protocol_translation"]
		)
		
		compute_node = EdgeComputingNode(
			name="Compute-Integration",
			node_type=EdgeNodeType.COMPUTE,
			location={"lat": 37.7849, "lng": -122.4094, "altitude": 100},
			capacity={"cpu_cores": 16, "memory_gb": 32, "storage_gb": 1000, "network_mbps": 2000, "gpu_cores": 4},
			network_latency_ms=3.0,
			specialized_capabilities=["ml_inference", "gpu_compute"]
		)
		
		await cluster.add_node(gateway_node)
		await cluster.add_node(compute_node)
		
		# Start cluster
		cluster._running = True
		scheduler_task = asyncio.create_task(cluster.schedule_tasks())
		
		# Submit different types of tasks
		tasks = [
			EdgeTask(
				twin_id="factory_twin_01",
				task_type="sensor_data_processing",
				priority=EdgeTaskPriority.HIGH,
				requirements={"cpu_cores": 0.5, "memory_mb": 128, "max_latency_ms": 2.0},
				payload={"readings": [{"temp": 23.5}, {"pressure": 1013.25}]}
			),
			EdgeTask(
				twin_id="factory_twin_01",
				task_type="predictive_analysis",
				priority=EdgeTaskPriority.NORMAL,
				requirements={"cpu_cores": 2.0, "memory_mb": 512, "max_latency_ms": 10.0, "gpu_required": True},
				payload={"model_input": [1.0, 2.0, 3.0, 4.0]}
			),
			EdgeTask(
				twin_id="factory_twin_02",
				task_type="real_time_control",
				priority=EdgeTaskPriority.CRITICAL,
				requirements={"cpu_cores": 1.0, "memory_mb": 256, "max_latency_ms": 1.0},
				payload={"control_signal": {"valve_position": 75}}
			)
		]
		
		# Submit all tasks
		task_ids = []
		for task in tasks:
			task_id = await cluster.submit_task(task)
			task_ids.append(task_id)
		
		# Wait for processing
		await asyncio.sleep(0.5)
		
		# Stop cluster
		cluster._running = False
		scheduler_task.cancel()
		
		# Verify results
		completed_tasks = 0
		for task_id in task_ids:
			task = cluster.tasks[task_id]
			if task.status == "completed":
				completed_tasks += 1
				assert task.result is not None
				assert task.execution_time_ms is not None
				assert task.assigned_node is not None
		
		# Should have completed most/all tasks
		assert completed_tasks >= len(tasks) - 1
		
		# Check performance metrics
		status = cluster.get_cluster_status()
		assert status["tasks"]["completed"] >= completed_tasks
		assert status["performance"]["cluster"]["total_tasks_processed"] >= completed_tasks
		
		# Cleanup
		await cluster.stop_cluster()
	
	@pytest.mark.asyncio
	async def test_fault_tolerance(self):
		"""Test fault tolerance and node failure handling"""
		cluster = EdgeComputingCluster()
		
		# Add reliable and unreliable nodes
		reliable_node = EdgeComputingNode(
			name="Reliable-Node",
			node_type=EdgeNodeType.COMPUTE,
			location={"lat": 0, "lng": 0, "altitude": 0},
			capacity={"cpu_cores": 8, "memory_gb": 16, "storage_gb": 500, "network_mbps": 1000},
			reliability_score=0.99
		)
		
		unreliable_node = EdgeComputingNode(
			name="Unreliable-Node",
			node_type=EdgeNodeType.COMPUTE,
			location={"lat": 0, "lng": 0, "altitude": 0},
			capacity={"cpu_cores": 8, "memory_gb": 16, "storage_gb": 500, "network_mbps": 1000},
			reliability_score=0.70
		)
		
		await cluster.add_node(reliable_node)
		await cluster.add_node(unreliable_node)
		
		# Simulate node failure by setting outdated heartbeat
		unreliable_node.last_heartbeat = datetime.utcnow() - timedelta(minutes=2)
		
		# Submit critical task
		critical_task = EdgeTask(
			twin_id="critical_twin",
			task_type="real_time_control",
			priority=EdgeTaskPriority.CRITICAL,
			requirements={"cpu_cores": 1.0, "memory_mb": 256, "max_latency_ms": 2.0},
			payload={"emergency_stop": True}
		)
		
		await cluster.submit_task(critical_task)
		
		# Start monitoring and scheduling
		cluster._running = True
		heartbeat_task = asyncio.create_task(cluster._monitor_node_heartbeats())
		scheduler_task = asyncio.create_task(cluster.schedule_tasks())
		
		# Wait for processing
		await asyncio.sleep(0.2)
		
		# Stop cluster
		cluster._running = False
		heartbeat_task.cancel()
		scheduler_task.cancel()
		
		# Verify unreliable node was marked inactive
		assert unreliable_node.status == "inactive"
		
		# Critical task should still be processed by reliable node
		task = cluster.tasks[critical_task.id]
		if task.assigned_node:
			assigned_node = cluster.nodes[task.assigned_node]
			assert assigned_node.reliability_score >= 0.99
		
		await cluster.stop_cluster()
	
	@pytest.mark.asyncio
	async def test_performance_under_load(self):
		"""Test system performance under high load"""
		cluster = EdgeComputingCluster()
		
		# Add multiple high-performance nodes
		nodes = []
		for i in range(4):
			node = EdgeComputingNode(
				name=f"HighPerf-Node-{i}",
				node_type=EdgeNodeType.COMPUTE,
				location={"lat": 0, "lng": 0, "altitude": 0},
				capacity={"cpu_cores": 16, "memory_gb": 32, "storage_gb": 1000, "network_mbps": 2000},
				network_latency_ms=1.0,
				reliability_score=0.98
			)
			nodes.append(node)
			await cluster.add_node(node)
		
		# Start cluster
		cluster._running = True
		scheduler_task = asyncio.create_task(cluster.schedule_tasks())
		
		# Submit high volume of tasks
		num_tasks = 50
		start_time = time.time()
		
		task_ids = []
		for i in range(num_tasks):
			task = EdgeTask(
				twin_id=f"load_twin_{i % 10}",
				task_type="sensor_data_processing",
				priority=EdgeTaskPriority.HIGH,
				requirements={"cpu_cores": 0.5, "memory_mb": 128, "max_latency_ms": 5.0},
				payload={"batch_id": i, "data": list(range(100))}
			)
			task_id = await cluster.submit_task(task)
			task_ids.append(task_id)
		
		# Wait for processing
		await asyncio.sleep(1.0)
		submission_time = time.time() - start_time
		
		# Stop cluster
		cluster._running = False
		scheduler_task.cancel()
		
		# Analyze performance
		completed_tasks = sum(1 for tid in task_ids if cluster.tasks[tid].status == "completed")
		throughput = completed_tasks / submission_time
		
		status = cluster.get_cluster_status()
		avg_latency = status["performance"]["cluster"]["avg_task_latency_ms"]
		
		# Performance assertions
		assert completed_tasks >= num_tasks * 0.8  # At least 80% completion
		assert throughput >= 10  # At least 10 tasks per second
		assert avg_latency <= 50  # Average latency under 50ms
		
		print(f"Performance Test Results:")
		print(f"  Completed: {completed_tasks}/{num_tasks} tasks")
		print(f"  Throughput: {throughput:.1f} tasks/sec")
		print(f"  Avg Latency: {avg_latency:.2f}ms")
		
		await cluster.stop_cluster()

if __name__ == "__main__":
	pytest.main([__file__, "-v"])