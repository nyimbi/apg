#!/usr/bin/env python3
"""
APG Workflow Orchestration Performance Tests

Comprehensive performance testing including load testing, stress testing,
throughput measurement, memory usage analysis, and scalability validation.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import pytest
import uuid
import time
import psutil
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import AsyncMock, Mock, patch
from concurrent.futures import ThreadPoolExecutor
import threading
import gc
import sys

# APG Core imports
from ..service import WorkflowOrchestrationService
from ..engine import WorkflowExecutionEngine
from ..database import DatabaseManager
from ..models import *

# Test utilities
from .conftest import TestHelpers


class PerformanceMetrics:
	"""Helper class for collecting and analyzing performance metrics."""
	
	def __init__(self):
		self.start_time = None
		self.end_time = None
		self.memory_usage = []
		self.cpu_usage = []
		self.execution_times = []
		self.throughput_data = []
	
	def start_monitoring(self):
		"""Start performance monitoring."""
		self.start_time = time.time()
		self.memory_usage = []
		self.cpu_usage = []
		self.execution_times = []
	
	def record_execution(self, execution_time: float, memory_mb: float = None, cpu_percent: float = None):
		"""Record a single execution's performance."""
		self.execution_times.append(execution_time)
		if memory_mb is not None:
			self.memory_usage.append(memory_mb)
		if cpu_percent is not None:
			self.cpu_usage.append(cpu_percent)
	
	def stop_monitoring(self):
		"""Stop performance monitoring."""
		self.end_time = time.time()
	
	def get_stats(self) -> Dict[str, Any]:
		"""Get performance statistics."""
		if not self.execution_times:
			return {"error": "No execution data recorded"}
		
		return {
			"total_duration": self.end_time - self.start_time if self.end_time else None,
			"execution_count": len(self.execution_times),
			"execution_times": {
				"mean": statistics.mean(self.execution_times),
				"median": statistics.median(self.execution_times),
				"min": min(self.execution_times),
				"max": max(self.execution_times),
				"stdev": statistics.stdev(self.execution_times) if len(self.execution_times) > 1 else 0
			},
			"memory_usage": {
				"mean": statistics.mean(self.memory_usage) if self.memory_usage else 0,
				"max": max(self.memory_usage) if self.memory_usage else 0,
				"min": min(self.memory_usage) if self.memory_usage else 0
			} if self.memory_usage else None,
			"cpu_usage": {
				"mean": statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
				"max": max(self.cpu_usage) if self.cpu_usage else 0
			} if self.cpu_usage else None,
			"throughput": len(self.execution_times) / (self.end_time - self.start_time) if self.end_time else 0
		}


class TestWorkflowPerformance:
	"""Test workflow execution performance under various conditions."""
	
	@pytest.mark.performance
	async def test_single_workflow_performance_baseline(self, workflow_service):
		"""Establish performance baseline for single workflow execution."""
		# Simple workflow for baseline measurement
		workflow_data = {
			"name": "Performance Baseline Workflow",
			"description": "Simple workflow for performance baseline",
			"tenant_id": "perf_test",
			"tasks": [
				{
					"id": "baseline_task",
					"name": "Baseline Task",
					"task_type": "script",
					"config": {"script": "return {'baseline': True, 'timestamp': time.time()}"}
				}
			]
		}
		
		workflow = await workflow_service.create_workflow(workflow_data, user_id="perf_user")
		
		# Measure single execution performance
		metrics = PerformanceMetrics()
		metrics.start_monitoring()
		
		execution_start = time.time()
		instance = await workflow_service.execute_workflow(workflow.id)
		
		# Wait for completion
		while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
			await asyncio.sleep(0.01)
			instance = await workflow_service.get_workflow_instance(instance.id)
		
		execution_end = time.time()
		execution_time = execution_end - execution_start
		
		# Record metrics
		process = psutil.Process()
		memory_mb = process.memory_info().rss / 1024 / 1024
		cpu_percent = process.cpu_percent()
		
		metrics.record_execution(execution_time, memory_mb, cpu_percent)
		metrics.stop_monitoring()
		
		stats = metrics.get_stats()
		
		# Baseline performance assertions
		assert instance.status == WorkflowStatus.COMPLETED
		assert stats["execution_times"]["mean"] < 1.0  # Should complete in under 1 second
		assert stats["memory_usage"]["mean"] < 500  # Should use less than 500MB
		
		print(f"Baseline Performance: {stats['execution_times']['mean']:.3f}s, {stats['memory_usage']['mean']:.1f}MB")
	
	@pytest.mark.performance
	async def test_concurrent_workflow_performance(self, workflow_service):
		"""Test performance under concurrent workflow execution."""
		# Create workflow template
		workflow_data = {
			"name": "Concurrent Performance Test",
			"description": "Workflow for concurrent performance testing",
			"tenant_id": "perf_test",
			"tasks": [
				{
					"id": "concurrent_task",
					"name": "Concurrent Task",
					"task_type": "script",
					"config": {
						"script": """
import time
import random
# Simulate variable processing time
time.sleep(random.uniform(0.05, 0.15))
return {'task_id': context.get('task_id', 'unknown'), 'completed': True}
"""
					}
				}
			]
		}
		
		workflow = await workflow_service.create_workflow(workflow_data, user_id="perf_user")
		
		# Test different concurrency levels
		concurrency_levels = [1, 5, 10, 20]
		results = {}
		
		for concurrency in concurrency_levels:
			metrics = PerformanceMetrics()
			metrics.start_monitoring()
			
			# Execute workflows concurrently
			start_time = time.time()
			tasks = []
			for i in range(concurrency):
				task = asyncio.create_task(
					workflow_service.execute_workflow(
						workflow.id,
						execution_context={"task_id": f"concurrent_{i}"}
					)
				)
				tasks.append(task)
			
			# Wait for all to start
			instances = await asyncio.gather(*tasks)
			
			# Wait for all to complete
			completion_tasks = []
			for instance in instances:
				async def wait_completion(inst_id):
					while True:
						inst = await workflow_service.get_workflow_instance(inst_id) 
						if inst.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
							return inst
						await asyncio.sleep(0.01)
				completion_tasks.append(wait_completion(instance.id))
			
			completed_instances = await asyncio.gather(*completion_tasks)
			end_time = time.time()
			
			total_time = end_time - start_time
			successful = sum(1 for inst in completed_instances if inst.status == WorkflowStatus.COMPLETED)
			
			# Record metrics
			process = psutil.Process()
			memory_mb = process.memory_info().rss / 1024 / 1024
			cpu_percent = process.cpu_percent()
			
			metrics.record_execution(total_time, memory_mb, cpu_percent)
			metrics.stop_monitoring()
			
			results[concurrency] = {
				"total_time": total_time,
				"successful_executions": successful,
				"throughput": successful / total_time,
				"memory_mb": memory_mb,
				"cpu_percent": cpu_percent
			}
			
			# Performance assertions
			assert successful == concurrency  # All should complete successfully
			assert total_time < 5.0  # Should complete within reasonable time
		
		# Analyze scalability
		throughputs = [results[c]["throughput"] for c in concurrency_levels]
		
		# Throughput should generally increase with concurrency (up to a point)
		assert throughputs[1] > throughputs[0]  # 5 concurrent should be faster than 1
		
		print("Concurrent Performance Results:")
		for concurrency in concurrency_levels:
			r = results[concurrency]
			print(f"  {concurrency} concurrent: {r['throughput']:.2f} workflows/sec, {r['memory_mb']:.1f}MB")
	
	@pytest.mark.performance
	async def test_large_workflow_performance(self, workflow_service):
		"""Test performance with workflows containing many tasks."""
		# Create workflow with varying numbers of tasks
		task_counts = [10, 25, 50, 100]
		results = {}
		
		for task_count in task_counts:
			# Generate tasks
			tasks = []
			for i in range(task_count):
				task = {
					"id": f"task_{i}",
					"name": f"Task {i}",
					"task_type": "script",
					"config": {"script": f"return {{'task_index': {i}, 'completed': True}}"}
				}
				# Create dependency chain for sequential execution
				if i > 0:
					task["depends_on"] = [f"task_{i-1}"]
				tasks.append(task)
			
			workflow_data = {
				"name": f"Large Workflow ({task_count} tasks)",
				"description": f"Performance test workflow with {task_count} tasks",
				"tenant_id": "perf_test",
				"tasks": tasks
			}
			
			workflow = await workflow_service.create_workflow(workflow_data, user_id="perf_user")
			
			# Measure execution performance
			start_time = time.time()
			process = psutil.Process()
			start_memory = process.memory_info().rss / 1024 / 1024
			
			instance = await workflow_service.execute_workflow(workflow.id)
			
			# Wait for completion
			while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
				await asyncio.sleep(0.1)
				instance = await workflow_service.get_workflow_instance(instance.id)
			
			end_time = time.time()
			end_memory = process.memory_info().rss / 1024 / 1024
			
			execution_time = end_time - start_time
			memory_increase = end_memory - start_memory
			
			results[task_count] = {
				"execution_time": execution_time,
				"memory_increase": memory_increase,
				"successful": instance.status == WorkflowStatus.COMPLETED,
				"task_executions": len(instance.task_executions),
				"throughput": task_count / execution_time if execution_time > 0 else 0
			}
			
			# Performance assertions
			assert instance.status == WorkflowStatus.COMPLETED
			assert len(instance.task_executions) == task_count
			assert execution_time < task_count * 0.5  # Should be faster than 0.5s per task
		
		# Analyze scalability with task count
		print("Large Workflow Performance Results:")
		for task_count in task_counts:
			r = results[task_count]
			print(f"  {task_count} tasks: {r['execution_time']:.2f}s, {r['throughput']:.1f} tasks/sec, +{r['memory_increase']:.1f}MB")
		
		# Execution time should scale reasonably with task count
		assert results[50]["execution_time"] < results[10]["execution_time"] * 10  # Should not scale linearly
	
	@pytest.mark.performance
	async def test_memory_usage_patterns(self, workflow_service):
		"""Test memory usage patterns during workflow execution."""
		# Create workflow that processes data
		workflow_data = {
			"name": "Memory Usage Test Workflow",
			"description": "Workflow for testing memory usage patterns",
			"tenant_id": "perf_test",
			"tasks": [
				{
					"id": "memory_task",
					"name": "Memory Usage Task",
					"task_type": "script",
					"config": {
						"script": """
# Create and process data to use memory
data_size = context.get('data_size', 1000)
data = [{'id': i, 'value': i * 2, 'text': f'item_{i}'} for i in range(data_size)]
processed = [item for item in data if item['value'] % 2 == 0]
return {'processed_count': len(processed), 'original_count': len(data)}
"""
					}
				}
			]
		}
		
		workflow = await workflow_service.create_workflow(workflow_data, user_id="perf_user")
		
		# Test different data sizes
		data_sizes = [100, 1000, 5000, 10000]
		memory_results = {}
		
		for data_size in data_sizes:
			# Force garbage collection before test
			gc.collect()
			
			process = psutil.Process()
			initial_memory = process.memory_info().rss / 1024 / 1024
			
			# Execute workflow with specific data size
			instance = await workflow_service.execute_workflow(
				workflow.id,
				execution_context={"data_size": data_size}
			)
			
			# Monitor memory during execution
			peak_memory = initial_memory
			while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
				current_memory = process.memory_info().rss / 1024 / 1024
				peak_memory = max(peak_memory, current_memory)
				await asyncio.sleep(0.05)
				instance = await workflow_service.get_workflow_instance(instance.id)
			
			final_memory = process.memory_info().rss / 1024 / 1024
			
			memory_results[data_size] = {
				"initial_memory": initial_memory,
				"peak_memory": peak_memory,
				"final_memory": final_memory,
				"memory_increase": peak_memory - initial_memory,
				"memory_retained": final_memory - initial_memory,
				"successful": instance.status == WorkflowStatus.COMPLETED
			}
			
			# Performance assertions
			assert instance.status == WorkflowStatus.COMPLETED
			assert memory_results[data_size]["memory_increase"] > 0  # Should use some memory
		
		# Analyze memory usage patterns
		print("Memory Usage Pattern Results:")
		for data_size in data_sizes:
			r = memory_results[data_size]
			print(f"  {data_size} items: +{r['memory_increase']:.1f}MB peak, {r['memory_retained']:.1f}MB retained")
		
		# Memory usage should scale with data size but not excessively
		largest_increase = memory_results[10000]["memory_increase"]
		smallest_increase = memory_results[100]["memory_increase"]
		scaling_factor = largest_increase / smallest_increase if smallest_increase > 0 else 0
		
		# Should scale somewhat with data size but not linearly due to efficiency optimizations
		assert scaling_factor < 200  # Should not increase memory 200x for 100x data
	
	@pytest.mark.performance
	@pytest.mark.slow
	async def test_sustained_load_performance(self, workflow_service):
		"""Test performance under sustained load over time."""
		# Create simple workflow for sustained testing
		workflow_data = {
			"name": "Sustained Load Test Workflow",
			"description": "Workflow for sustained load testing",
			"tenant_id": "perf_test",
			"tasks": [
				{
					"id": "load_task",
					"name": "Load Test Task",
					"task_type": "script",
					"config": {
						"script": """
import time
import random
# Small amount of work with random variation
time.sleep(random.uniform(0.01, 0.05))
return {'load_test': True, 'timestamp': time.time()}
"""
					}
				}
			]
		}
		
		workflow = await workflow_service.create_workflow(workflow_data, user_id="perf_user")
		
		# Run sustained load test
		test_duration = 30  # 30 seconds
		target_rps = 5  # 5 requests per second
		
		metrics = PerformanceMetrics()
		metrics.start_monitoring()
		
		start_time = time.time()
		execution_count = 0
		successful_count = 0
		
		while time.time() - start_time < test_duration:
			batch_start = time.time()
			
			# Execute batch of workflows
			batch_size = min(target_rps, 10)  # Limit batch size
			batch_tasks = []
			
			for _ in range(batch_size):
				task = asyncio.create_task(workflow_service.execute_workflow(workflow.id))
				batch_tasks.append(task)
			
			# Wait for batch to start
			instances = await asyncio.gather(*batch_tasks)
			execution_count += len(instances)
			
			# Check some completed instances (don't wait for all to avoid blocking)
			for instance in instances[:2]:  # Check first 2
				try:
					# Quick check for completion
					for _ in range(20):  # Max 1 second wait
						current_instance = await workflow_service.get_workflow_instance(instance.id)
						if current_instance.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
							if current_instance.status == WorkflowStatus.COMPLETED:
								successful_count += 1
							break
						await asyncio.sleep(0.05)
				except Exception:
					pass  # Continue with load test even if individual checks fail
			
			# Control rate
			batch_time = time.time() - batch_start
			sleep_time = max(0, 1.0 - batch_time)  # Target 1 second per batch
			await asyncio.sleep(sleep_time)
		
		metrics.stop_monitoring()
		
		total_time = time.time() - start_time
		actual_rps = execution_count / total_time
		success_rate = successful_count / max(execution_count, 1)
		
		print(f"Sustained Load Results:")
		print(f"  Duration: {total_time:.1f}s")
		print(f"  Executions: {execution_count}")
		print(f"  Target RPS: {target_rps}, Actual RPS: {actual_rps:.2f}")
		print(f"  Success Rate: {success_rate:.2f}")
		
		# Performance assertions for sustained load
		assert execution_count > test_duration * 2  # Should have reasonable throughput
		assert actual_rps >= target_rps * 0.7  # Should achieve at least 70% of target RPS
		assert success_rate >= 0.8  # At least 80% success rate under load


class TestDatabasePerformance:
	"""Test database performance and query optimization."""
	
	@pytest.mark.performance
	async def test_database_query_performance(self, database_manager):
		"""Test database query performance under load."""
		# Create test data
		test_workflows = []
		for i in range(100):
			workflow_data = {
				"name": f"DB Test Workflow {i}",
				"description": f"Workflow {i} for database performance testing",
				"tenant_id": "db_perf_test",
				"definition": {"tasks": [{"id": f"task_{i}", "name": f"Task {i}"}]},
				"created_by": "db_test_user",
				"version": "1.0"
			}
			test_workflows.append(workflow_data)
		
		# Measure bulk insert performance
		start_time = time.time()
		
		async with database_manager.get_session() as session:
			db_workflows = []
			for workflow_data in test_workflows:
				db_workflow = WorkflowDB(
					name=workflow_data["name"],
					description=workflow_data["description"],
					tenant_id=workflow_data["tenant_id"],
					definition=workflow_data["definition"],
					created_by=workflow_data["created_by"],
					version=workflow_data["version"]
				)
				db_workflows.append(db_workflow)
			
			session.add_all(db_workflows)
			await session.commit()
		
		insert_time = time.time() - start_time
		
		# Measure query performance
		query_times = []
		
		# Test various query patterns
		queries = [
			# Simple queries
			lambda s: s.query(WorkflowDB).filter(WorkflowDB.tenant_id == "db_perf_test"),
			lambda s: s.query(WorkflowDB).filter(WorkflowDB.created_by == "db_test_user"),
			lambda s: s.query(WorkflowDB).filter(WorkflowDB.name.like("DB Test Workflow%")),
			
			# Count queries
			lambda s: s.query(WorkflowDB).filter(WorkflowDB.tenant_id == "db_perf_test").count(),
			
			# Ordering queries
			lambda s: s.query(WorkflowDB).filter(WorkflowDB.tenant_id == "db_perf_test").order_by(WorkflowDB.created_at.desc()),
		]
		
		for query_func in queries:
			start_time = time.time()
			
			async with database_manager.get_session() as session:
				if callable(query_func):
					# Handle both query objects and count() calls
					try:
						result = query_func(session)
						if hasattr(result, 'all'):
							await result.all()
						elif hasattr(result, 'scalar'):
							await result.scalar()
					except Exception:
						# For count queries and similar
						pass
			
			query_time = time.time() - start_time
			query_times.append(query_time)
		
		# Performance assertions
		assert insert_time < 5.0  # Bulk insert should complete within 5 seconds
		assert max(query_times) < 1.0  # Individual queries should complete within 1 second
		assert statistics.mean(query_times) < 0.5  # Average query time should be reasonable
		
		print(f"Database Performance:")
		print(f"  Bulk insert (100 records): {insert_time:.3f}s")
		print(f"  Query times: avg={statistics.mean(query_times):.3f}s, max={max(query_times):.3f}s")
	
	@pytest.mark.performance
	async def test_concurrent_database_access(self, database_manager):
		"""Test database performance under concurrent access."""
		# Test concurrent reads and writes
		concurrent_operations = 20
		
		async def concurrent_workflow_creation(index):
			workflow_data = {
				"name": f"Concurrent Workflow {index}",
				"description": f"Concurrent test workflow {index}",
				"tenant_id": f"concurrent_tenant_{index % 5}",  # 5 different tenants
				"definition": {"tasks": [{"id": f"task_{index}"}]},
				"created_by": f"user_{index}",
				"version": "1.0"
			}
			
			start_time = time.time()
			
			async with database_manager.get_session() as session:
				db_workflow = WorkflowDB(**workflow_data)
				session.add(db_workflow)
				await session.commit()
				
				# Immediately read it back
				result = await session.get(WorkflowDB, db_workflow.id)
				assert result is not None
			
			return time.time() - start_time
		
		# Execute concurrent operations
		start_time = time.time()
		
		operation_times = await asyncio.gather(*[
			concurrent_workflow_creation(i) for i in range(concurrent_operations)
		])
		
		total_time = time.time() - start_time
		
		# Performance analysis
		avg_operation_time = statistics.mean(operation_times)
		max_operation_time = max(operation_times)
		throughput = concurrent_operations / total_time
		
		print(f"Concurrent Database Access:")
		print(f"  {concurrent_operations} operations in {total_time:.3f}s")
		print(f"  Throughput: {throughput:.2f} ops/sec")
		print(f"  Avg operation time: {avg_operation_time:.3f}s")
		print(f"  Max operation time: {max_operation_time:.3f}s")
		
		# Performance assertions
		assert total_time < 10.0  # Should complete within reasonable time
		assert avg_operation_time < 1.0  # Average operation should be fast
		assert throughput >= 2.0  # Should achieve reasonable throughput


class TestScalabilityAndLimits:
	"""Test system scalability and identify performance limits."""
	
	@pytest.mark.performance
	@pytest.mark.slow
	async def test_workflow_scalability_limits(self, workflow_service):
		"""Test workflow system scalability limits."""
		# Create simple workflow template
		workflow_data = {
			"name": "Scalability Test Workflow",
			"description": "Workflow for scalability testing",
			"tenant_id": "scalability_test",
			"tasks": [
				{
					"id": "scalability_task",
					"name": "Scalability Task",
					"task_type": "script",
					"config": {"script": "return {'scale_test': True}"}
				}
			]
		}
		
		workflow = await workflow_service.create_workflow(workflow_data, user_id="scale_user")
		
		# Test increasing loads to find limits
		load_levels = [10, 25, 50, 100, 200]
		scalability_results = {}
		
		for load_level in load_levels:
			print(f"Testing scalability at {load_level} concurrent workflows...")
			
			start_time = time.time()
			process = psutil.Process()
			initial_memory = process.memory_info().rss / 1024 / 1024
			
			try:
				# Create concurrent executions
				tasks = []
				for i in range(load_level):
					task = asyncio.create_task(workflow_service.execute_workflow(workflow.id))
					tasks.append(task)
				
				# Wait for all to start
				instances = await asyncio.gather(*tasks, return_exceptions=True)
				successful_starts = sum(1 for inst in instances if not isinstance(inst, Exception))
				
				# Wait for completions (with timeout)
				completion_tasks = []
				for instance in instances:
					if not isinstance(instance, Exception):
						async def wait_with_timeout(inst_id, timeout=30):
							try:
								end_time = time.time() + timeout
								while time.time() < end_time:
									inst = await workflow_service.get_workflow_instance(inst_id)
									if inst.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
										return inst
									await asyncio.sleep(0.1)
								return None  # Timeout
							except Exception:
								return None
						
						completion_tasks.append(wait_with_timeout(instance.id))
				
				completed_instances = await asyncio.gather(*completion_tasks)
				successful_completions = sum(1 for inst in completed_instances if inst and inst.status == WorkflowStatus.COMPLETED)
				
			except Exception as e:
				successful_starts = 0
				successful_completions = 0
				print(f"Exception at load level {load_level}: {e}")
			
			end_time = time.time()
			final_memory = process.memory_info().rss / 1024 / 1024
			
			total_time = end_time - start_time
			memory_increase = final_memory - initial_memory
			
			scalability_results[load_level] = {
				"successful_starts": successful_starts,
				"successful_completions": successful_completions,
				"total_time": total_time,
				"memory_increase": memory_increase,
				"start_success_rate": successful_starts / load_level,
				"completion_success_rate": successful_completions / load_level if load_level > 0 else 0,
				"throughput": successful_completions / total_time if total_time > 0 else 0
			}
			
			# Break if success rate drops too low
			if scalability_results[load_level]["completion_success_rate"] < 0.5:
				print(f"Success rate dropped below 50% at {load_level} concurrent workflows")
				break
		
		# Analyze scalability results
		print("\nScalability Test Results:")
		for load_level, results in scalability_results.items():
			print(f"  {load_level} concurrent: {results['completion_success_rate']:.1%} success, "
				  f"{results['throughput']:.2f} workflows/sec, +{results['memory_increase']:.1f}MB")
		
		# Find maximum sustainable load
		max_sustainable_load = 0
		for load_level, results in scalability_results.items():
			if results["completion_success_rate"] >= 0.9:  # 90% success rate
				max_sustainable_load = load_level
		
		print(f"\nMaximum sustainable load: {max_sustainable_load} concurrent workflows")
		
		# Scalability assertions
		assert max_sustainable_load >= 10  # Should handle at least 10 concurrent workflows
		assert scalability_results[10]["completion_success_rate"] >= 0.9  # Should handle 10 with 90% success
	
	@pytest.mark.performance
	async def test_resource_usage_optimization(self, workflow_service):
		"""Test resource usage optimization and efficiency."""
		# Create workflow that can test resource optimization
		workflow_data = {
			"name": "Resource Optimization Test",
			"description": "Workflow for testing resource optimization",
			"tenant_id": "resource_test",
			"tasks": [
				{
					"id": "resource_task",
					"name": "Resource Usage Task",
					"task_type": "script",
					"config": {
						"script": """
import gc
# Force garbage collection
gc.collect()
return {'resource_optimized': True, 'gc_collected': True}
"""
					}
				}
			]
		}
		
		workflow = await workflow_service.create_workflow(workflow_data, user_id="resource_user")
		
		# Test resource usage over multiple executions
		execution_count = 20
		memory_measurements = []
		
		process = psutil.Process()
		initial_memory = process.memory_info().rss / 1024 / 1024
		
		for i in range(execution_count):
			# Execute workflow
			instance = await workflow_service.execute_workflow(workflow.id)
			
			# Wait for completion
			while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
				await asyncio.sleep(0.01)
				instance = await workflow_service.get_workflow_instance(instance.id)
			
			# Measure memory after each execution
			current_memory = process.memory_info().rss / 1024 / 1024
			memory_measurements.append(current_memory)
			
			assert instance.status == WorkflowStatus.COMPLETED
		
		final_memory = process.memory_info().rss / 1024 / 1024
		
		# Analyze memory usage patterns
		memory_growth = final_memory - initial_memory
		max_memory = max(memory_measurements)
		min_memory = min(memory_measurements)
		memory_variance = max_memory - min_memory
		
		print(f"Resource Usage Optimization:")
		print(f"  Initial memory: {initial_memory:.1f}MB")
		print(f"  Final memory: {final_memory:.1f}MB")
		print(f"  Memory growth: {memory_growth:.1f}MB over {execution_count} executions")
		print(f"  Memory variance: {memory_variance:.1f}MB")
		
		# Resource optimization assertions
		assert memory_growth < 50  # Should not grow more than 50MB over 20 executions
		assert memory_variance < 100  # Memory variance should be controlled
		
		# Memory usage should be relatively stable (not growing unboundedly)
		if execution_count >= 10:
			first_half_avg = statistics.mean(memory_measurements[:execution_count//2])
			second_half_avg = statistics.mean(memory_measurements[execution_count//2:])
			growth_rate = (second_half_avg - first_half_avg) / first_half_avg
			
			assert abs(growth_rate) < 0.2  # Memory usage should not grow more than 20% between halves


class TestPerformanceRegression:
	"""Test for performance regressions and establish benchmarks."""
	
	@pytest.mark.performance
	async def test_performance_benchmarks(self, workflow_service):
		"""Establish and verify performance benchmarks."""
		# Define benchmark workflows
		benchmark_workflows = {
			"simple": {
				"name": "Simple Benchmark",
				"tasks": [{"id": "simple", "name": "Simple", "task_type": "script", "config": {"script": "return {'simple': True}"}}],
				"expected_time": 0.5
			},
			"sequential": {
				"name": "Sequential Benchmark", 
				"tasks": [
					{"id": "seq1", "name": "Seq 1", "task_type": "script", "config": {"script": "return {'seq': 1}"}},
					{"id": "seq2", "name": "Seq 2", "task_type": "script", "config": {"script": "return {'seq': 2}"}, "depends_on": ["seq1"]},
					{"id": "seq3", "name": "Seq 3", "task_type": "script", "config": {"script": "return {'seq': 3}"}, "depends_on": ["seq2"]}
				],
				"expected_time": 1.0
			},
			"parallel": {
				"name": "Parallel Benchmark",
				"tasks": [
					{"id": "start", "name": "Start", "task_type": "script", "config": {"script": "return {'start': True}"}},
					{"id": "par1", "name": "Par 1", "task_type": "script", "config": {"script": "return {'par': 1}"}, "depends_on": ["start"]},
					{"id": "par2", "name": "Par 2", "task_type": "script", "config": {"script": "return {'par': 2}"}, "depends_on": ["start"]},
					{"id": "par3", "name": "Par 3", "task_type": "script", "config": {"script": "return {'par': 3}"}, "depends_on": ["start"]},
					{"id": "end", "name": "End", "task_type": "script", "config": {"script": "return {'end': True}"}, "depends_on": ["par1", "par2", "par3"]}
				],
				"expected_time": 1.5
			}
		}
		
		benchmark_results = {}
		
		for benchmark_name, benchmark_config in benchmark_workflows.items():
			workflow_data = {
				"name": benchmark_config["name"],
				"description": f"Benchmark workflow: {benchmark_name}",
				"tenant_id": "benchmark_test",
				"tasks": benchmark_config["tasks"]
			}
			
			workflow = await workflow_service.create_workflow(workflow_data, user_id="benchmark_user")
			
			# Run benchmark multiple times for accuracy
			execution_times = []
			
			for _ in range(5):  # 5 runs for statistical accuracy
				start_time = time.time()
				
				instance = await workflow_service.execute_workflow(workflow.id)
				
				# Wait for completion
				while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
					await asyncio.sleep(0.01)
					instance = await workflow_service.get_workflow_instance(instance.id)
				
				end_time = time.time()
				execution_time = end_time - start_time
				
				assert instance.status == WorkflowStatus.COMPLETED
				execution_times.append(execution_time)
			
			# Calculate benchmark statistics
			avg_time = statistics.mean(execution_times)
			median_time = statistics.median(execution_times)
			stdev_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
			
			benchmark_results[benchmark_name] = {
				"avg_time": avg_time,
				"median_time": median_time,
				"stdev_time": stdev_time,
				"expected_time": benchmark_config["expected_time"],
				"performance_ratio": avg_time / benchmark_config["expected_time"]
			}
			
			# Performance regression check
			assert avg_time <= benchmark_config["expected_time"] * 1.5  # Allow 50% variance
			
			print(f"Benchmark '{benchmark_name}': {avg_time:.3f}s avg (expected: {benchmark_config['expected_time']}s)")
		
		# Overall benchmark assessment
		overall_performance = statistics.mean([r["performance_ratio"] for r in benchmark_results.values()])
		
		print(f"\nOverall Performance Ratio: {overall_performance:.2f} (1.0 = meets expectations)")
		
		# Overall performance should be reasonable
		assert overall_performance <= 2.0  # Should not be more than 2x slower than expected
		
		return benchmark_results