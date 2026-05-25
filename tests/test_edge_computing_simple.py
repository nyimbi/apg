"""
Simple test for edge computing capabilities
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from capabilities.edge_computing import (
	EdgeComputingCluster,
	EdgeComputingNode,
	EdgeTask,
	EdgeNodeType,
	EdgeTaskPriority,
	EdgeEnabledDigitalTwin
)

async def test_basic_edge_computing():
	"""Test basic edge computing functionality"""
	print("Testing Edge Computing System...")
	
	# Create cluster
	cluster = EdgeComputingCluster()
	
	# Create edge nodes
	gateway_node = EdgeComputingNode(
		name="Gateway-Test",
		node_type=EdgeNodeType.GATEWAY,
		location={"lat": 37.7749, "lng": -122.4194, "altitude": 50},
		capacity={
			"cpu_cores": 8,
			"memory_gb": 16,
			"storage_gb": 500,
			"network_mbps": 1000
		},
		network_latency_ms=2.0
	)
	
	compute_node = EdgeComputingNode(
		name="Compute-Test",
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
	success1 = await cluster.add_node(gateway_node)
	success2 = await cluster.add_node(compute_node)
	
	print(f"✓ Added gateway node: {success1}")
	print(f"✓ Added compute node: {success2}")
	
	# Start cluster
	cluster._running = True
	scheduler_task = asyncio.create_task(cluster.schedule_tasks())
	
	# Create test tasks
	tasks = []
	for i in range(3):
		task = EdgeTask(
			twin_id=f"twin_test_{i}",
			task_type="sensor_data_processing",
			priority=EdgeTaskPriority.HIGH,
			requirements={
				"cpu_cores": 1.0,
				"memory_mb": 256,
				"max_latency_ms": 10.0
			},
			payload={
				"readings": [{"sensor_id": f"sensor_{j}", "value": j * 10} for j in range(5)]
			}
		)
		tasks.append(task)
	
	# Submit tasks
	task_ids = []
	for task in tasks:
		task_id = await cluster.submit_task(task)
		task_ids.append(task_id)
		print(f"✓ Submitted task {task.id}")
	
	# Wait for processing
	print("⏱ Processing tasks...")
	await asyncio.sleep(0.5)
	
	# Check results
	completed = 0
	for task_id in task_ids:
		task = cluster.tasks[task_id]
		if task.status == "completed":
			completed += 1
			print(f"✓ Task {task_id} completed in {task.execution_time_ms:.2f}ms")
		else:
			print(f"⚠ Task {task_id} status: {task.status}")
	
	# Stop cluster
	cluster._running = False
	scheduler_task.cancel()
	
	# Get cluster status
	status = cluster.get_cluster_status()
	print(f"\n📊 Final Status:")
	print(f"  Nodes: {status['nodes']['active']}/{status['nodes']['total']} active")
	print(f"  Tasks: {status['tasks']['completed']}/{status['tasks']['total']} completed")
	print(f"  Avg Latency: {status['performance']['cluster']['avg_task_latency_ms']:.2f}ms")
	
	await cluster.stop_cluster()
	
	print(f"✅ Test completed: {completed}/{len(tasks)} tasks successful")
	return completed == len(tasks)

async def test_edge_enabled_digital_twin():
	"""Test edge-enabled digital twin"""
	print("\nTesting Edge-Enabled Digital Twin...")
	
	# Create digital twin
	twin = EdgeEnabledDigitalTwin("twin_edge_test")
	
	# Create edge node
	edge_node = EdgeComputingNode(
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
		network_latency_ms=2.5
	)
	
	# Deploy twin to edge
	await twin.deploy_to_edge([edge_node])
	print("✓ Deployed digital twin to edge node")
	
	# Register real-time processor
	def temperature_processor(data):
		if isinstance(data, list):
			temps = [reading.get('temperature', 0) for reading in data if isinstance(reading, dict)]
			if temps:
				return {"avg_temperature": sum(temps) / len(temps), "readings_count": len(temps)}
		return {"avg_temperature": 0, "readings_count": 0}
	
	await twin.register_real_time_processor("temperature", temperature_processor)
	print("✓ Registered temperature processor")
	
	# Process real-time data
	sensor_data = [
		{"sensor_id": "temp_01", "temperature": 23.5},
		{"sensor_id": "temp_02", "temperature": 24.1},
		{"sensor_id": "temp_03", "temperature": 22.8}
	]
	
	try:
		result = await twin.process_real_time_data("temperature", sensor_data, max_latency_ms=50.0)
		print(f"✓ Processed real-time data: {result['result']}")
		
		# Get performance metrics
		metrics = twin.get_edge_performance_metrics()
		print(f"✓ Performance metrics collected for twin {metrics['twin_id']}")
		
		await twin.edge_cluster.stop_cluster()
		print("✅ Digital twin edge test completed successfully")
		return True
		
	except Exception as e:
		print(f"❌ Digital twin edge test failed: {e}")
		await twin.edge_cluster.stop_cluster()
		return False

async def main():
	"""Run all edge computing tests"""
	print("🚀 Starting Edge Computing Tests\n")
	
	# Test basic cluster functionality
	basic_success = await test_basic_edge_computing()
	
	# Test digital twin integration
	twin_success = await test_edge_enabled_digital_twin()
	
	print(f"\n🏁 Test Summary:")
	print(f"  Basic Edge Computing: {'✅ PASS' if basic_success else '❌ FAIL'}")
	print(f"  Digital Twin Integration: {'✅ PASS' if twin_success else '❌ FAIL'}")
	
	if basic_success and twin_success:
		print("\n🎉 All edge computing tests passed!")
		return True
	else:
		print("\n⚠️ Some edge computing tests failed")
		return False

if __name__ == "__main__":
	success = asyncio.run(main())
	exit(0 if success else 1)