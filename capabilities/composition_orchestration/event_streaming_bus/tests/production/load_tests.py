"""
APG Event Streaming Bus - Production Load Tests

Comprehensive load testing for production validation and performance benchmarking.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import time
import json
import logging
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import websockets
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import redis
import psutil
import numpy as np
from uuid_extensions import uuid7str

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
	"""Configuration for load tests."""
	api_url: str = "http://localhost:8080"
	ws_url: str = "ws://localhost:8080"
	kafka_servers: str = "localhost:9092"
	redis_url: str = "redis://localhost:6379/0"
	
	# Test parameters
	concurrent_users: int = 100
	test_duration_seconds: int = 300  # 5 minutes
	ramp_up_seconds: int = 60  # 1 minute
	events_per_second: int = 1000
	batch_size: int = 50
	
	# Performance thresholds
	max_response_time_ms: int = 100
	max_p95_response_time_ms: int = 200
	max_error_rate_percent: float = 1.0
	min_throughput_eps: int = 800  # events per second

@dataclass
class TestResult:
	"""Individual test result."""
	test_name: str
	start_time: datetime
	end_time: datetime
	duration_ms: float
	success: bool
	error_message: Optional[str] = None
	response_size: int = 0
	status_code: Optional[int] = None

@dataclass
class LoadTestReport:
	"""Comprehensive load test report."""
	test_name: str
	config: LoadTestConfig
	start_time: datetime
	end_time: datetime
	total_requests: int
	successful_requests: int
	failed_requests: int
	error_rate_percent: float
	
	# Response time statistics
	avg_response_time_ms: float
	min_response_time_ms: float
	max_response_time_ms: float
	p50_response_time_ms: float
	p95_response_time_ms: float
	p99_response_time_ms: float
	
	# Throughput statistics
	requests_per_second: float
	events_per_second: float
	bytes_per_second: float
	
	# Resource usage
	peak_cpu_percent: float
	peak_memory_mb: float
	peak_network_mbps: float
	
	# Detailed results
	results: List[TestResult]
	errors: Dict[str, int]
	
	def passed_thresholds(self, config: LoadTestConfig) -> bool:
		"""Check if test passed performance thresholds."""
		return (
			self.error_rate_percent <= config.max_error_rate_percent and
			self.p95_response_time_ms <= config.max_p95_response_time_ms and
			self.events_per_second >= config.min_throughput_eps
		)

class LoadTestExecutor:
	"""Execute comprehensive load tests."""
	
	def __init__(self, config: LoadTestConfig):
		self.config = config
		self.session: Optional[aiohttp.ClientSession] = None
		self.kafka_producer: Optional[KafkaProducer] = None
		self.redis_client: Optional[redis.Redis] = None
		self.results: List[TestResult] = []
		self.errors: Dict[str, int] = {}
		
		# Performance monitoring
		self.cpu_samples: List[float] = []
		self.memory_samples: List[float] = []
		self.network_samples: List[float] = []
	
	async def setup(self):
		"""Setup test environment."""
		logger.info("Setting up load test environment...")
		
		# HTTP session
		timeout = aiohttp.ClientTimeout(total=30)
		self.session = aiohttp.ClientSession(timeout=timeout)
		
		# Kafka producer
		self.kafka_producer = KafkaProducer(
			bootstrap_servers=self.config.kafka_servers.split(','),
			value_serializer=lambda x: json.dumps(x).encode('utf-8'),
			acks='all',
			retries=3,
			batch_size=16384,
			linger_ms=10,
			compression_type='snappy'
		)
		
		# Redis client
		self.redis_client = redis.from_url(self.config.redis_url)
		
		# Test connectivity
		await self._test_connectivity()
		
		logger.info("Load test environment setup completed")
	
	async def teardown(self):
		"""Cleanup test environment."""
		logger.info("Cleaning up load test environment...")
		
		if self.session:
			await self.session.close()
		
		if self.kafka_producer:
			self.kafka_producer.close()
		
		if self.redis_client:
			self.redis_client.close()
		
		logger.info("Load test environment cleanup completed")
	
	async def _test_connectivity(self):
		"""Test connectivity to all services."""
		# Test API
		async with self.session.get(f"{self.config.api_url}/health") as response:
			if response.status != 200:
				raise Exception(f"API health check failed: {response.status}")
		
		# Test Kafka
		try:
			self.kafka_producer.send('test-topic', {'test': 'connectivity'}).get(timeout=10)
		except KafkaError as e:
			raise Exception(f"Kafka connectivity test failed: {e}")
		
		# Test Redis
		if not self.redis_client.ping():
			raise Exception("Redis connectivity test failed")
	
	def _record_result(self, result: TestResult):
		"""Record individual test result."""
		self.results.append(result)
		
		if not result.success and result.error_message:
			error_key = result.error_message[:100]  # Truncate long errors
			self.errors[error_key] = self.errors.get(error_key, 0) + 1
	
	def _monitor_resources(self):
		"""Monitor system resources during test."""
		try:
			# CPU usage
			cpu_percent = psutil.cpu_percent(interval=1)
			self.cpu_samples.append(cpu_percent)
			
			# Memory usage
			memory = psutil.virtual_memory()
			memory_mb = memory.used / (1024 * 1024)
			self.memory_samples.append(memory_mb)
			
			# Network usage (simplified)
			net_io = psutil.net_io_counters()
			if hasattr(self, '_prev_net_bytes'):
				bytes_sent = net_io.bytes_sent - self._prev_net_bytes
				network_mbps = (bytes_sent * 8) / (1024 * 1024)  # Convert to Mbps
				self.network_samples.append(network_mbps)
			self._prev_net_bytes = net_io.bytes_sent
			
		except Exception as e:
			logger.warning(f"Resource monitoring error: {e}")
	
	async def _api_load_test(self) -> List[TestResult]:
		"""Execute API load test."""
		logger.info("Starting API load test...")
		
		results = []
		tasks = []
		
		# Create test data
		test_events = [
			{
				"event_type": "user.created",
				"source_capability": "user_management",
				"aggregate_id": f"user_{i}",
				"aggregate_type": "User",
				"payload": {
					"user_name": f"test_user_{i}",
					"email": f"user_{i}@test.com",
					"created_at": datetime.now(timezone.utc).isoformat()
				}
			}
			for i in range(self.config.concurrent_users * 10)
		]
		
		async def api_request(event_data: Dict[str, Any]) -> TestResult:
			"""Execute single API request."""
			start_time = datetime.now(timezone.utc)
			
			try:
				start_ms = time.time() * 1000
				
				async with self.session.post(
					f"{self.config.api_url}/api/v1/events",
					json=event_data,
					headers={"Content-Type": "application/json"}
				) as response:
					response_text = await response.text()
					end_ms = time.time() * 1000
					
					return TestResult(
						test_name="api_publish_event",
						start_time=start_time,
						end_time=datetime.now(timezone.utc),
						duration_ms=end_ms - start_ms,
						success=response.status == 200,
						error_message=None if response.status == 200 else f"HTTP {response.status}: {response_text}",
						response_size=len(response_text),
						status_code=response.status
					)
			
			except Exception as e:
				return TestResult(
					test_name="api_publish_event",
					start_time=start_time,
					end_time=datetime.now(timezone.utc),
					duration_ms=0,
					success=False,
					error_message=str(e)
				)
		
		# Execute concurrent requests
		semaphore = asyncio.Semaphore(self.config.concurrent_users)
		
		async def rate_limited_request(event_data):
			async with semaphore:
				return await api_request(event_data)
		
		# Create tasks with rate limiting
		for event_data in test_events:
			task = asyncio.create_task(rate_limited_request(event_data))
			tasks.append(task)
			
			# Rate limiting
			if len(tasks) % self.config.batch_size == 0:
				await asyncio.sleep(0.1)  # Small delay between batches
		
		# Wait for all requests to complete
		for task in asyncio.as_completed(tasks):
			result = await task
			results.append(result)
			self._record_result(result)
		
		logger.info(f"API load test completed: {len(results)} requests")
		return results
	
	async def _websocket_load_test(self) -> List[TestResult]:
		"""Execute WebSocket load test."""
		logger.info("Starting WebSocket load test...")
		
		results = []
		
		async def websocket_session():
			"""Single WebSocket session."""
			start_time = datetime.now(timezone.utc)
			
			try:
				start_ms = time.time() * 1000
				
				ws_url = f"{self.config.ws_url.replace('http', 'ws')}/ws/events/test_stream"
				
				async with websockets.connect(ws_url) as websocket:
					# Send test message
					test_message = {
						"type": "subscribe",
						"stream": "test_stream",
						"consumer_group": "load_test_group"
					}
					
					await websocket.send(json.dumps(test_message))
					
					# Wait for response
					response = await asyncio.wait_for(websocket.recv(), timeout=10)
					end_ms = time.time() * 1000
					
					return TestResult(
						test_name="websocket_connection",
						start_time=start_time,
						end_time=datetime.now(timezone.utc),
						duration_ms=end_ms - start_ms,
						success=True,
						response_size=len(response)
					)
			
			except Exception as e:
				return TestResult(
					test_name="websocket_connection",
					start_time=start_time,
					end_time=datetime.now(timezone.utc),
					duration_ms=0,
					success=False,
					error_message=str(e)
				)
		
		# Execute concurrent WebSocket connections
		tasks = [websocket_session() for _ in range(min(50, self.config.concurrent_users))]
		
		for task in asyncio.as_completed(tasks):
			result = await task
			results.append(result)
			self._record_result(result)
		
		logger.info(f"WebSocket load test completed: {len(results)} connections")
		return results
	
	async def _kafka_load_test(self) -> List[TestResult]:
		"""Execute Kafka load test."""
		logger.info("Starting Kafka load test...")
		
		results = []
		
		# Producer test
		for i in range(self.config.events_per_second):
			start_time = datetime.now(timezone.utc)
			start_ms = time.time() * 1000
			
			try:
				event_data = {
					"event_id": uuid7str(),
					"event_type": "load_test.event",
					"payload": {"index": i, "timestamp": start_time.isoformat()},
					"stream_id": "load_test_stream"
				}
				
				# Send to Kafka
				future = self.kafka_producer.send('apg-events', event_data)
				record_metadata = future.get(timeout=10)
				
				end_ms = time.time() * 1000
				
				result = TestResult(
					test_name="kafka_publish",
					start_time=start_time,
					end_time=datetime.now(timezone.utc),
					duration_ms=end_ms - start_ms,
					success=True,
					response_size=len(json.dumps(event_data))
				)
				
			except Exception as e:
				result = TestResult(
					test_name="kafka_publish",
					start_time=start_time,
					end_time=datetime.now(timezone.utc),
					duration_ms=0,
					success=False,
					error_message=str(e)
				)
			
			results.append(result)
			self._record_result(result)
			
			# Rate limiting
			if i % 100 == 0:
				await asyncio.sleep(0.1)
		
		logger.info(f"Kafka load test completed: {len(results)} events published")
		return results
	
	async def _database_load_test(self) -> List[TestResult]:
		"""Execute database load test through API."""
		logger.info("Starting database load test...")
		
		results = []
		
		# Query performance test
		queries = [
			"/api/v1/events?limit=100",
			"/api/v1/streams",
			"/api/v1/subscriptions",
			"/api/v1/schemas",
			"/api/v1/metrics/streams",
			"/api/v1/status"
		]
		
		for _ in range(self.config.concurrent_users):
			for query in queries:
				start_time = datetime.now(timezone.utc)
				start_ms = time.time() * 1000
				
				try:
					async with self.session.get(f"{self.config.api_url}{query}") as response:
						response_text = await response.text()
						end_ms = time.time() * 1000
						
						result = TestResult(
							test_name="database_query",
							start_time=start_time,
							end_time=datetime.now(timezone.utc),
							duration_ms=end_ms - start_ms,
							success=response.status == 200,
							error_message=None if response.status == 200 else f"HTTP {response.status}",
							response_size=len(response_text),
							status_code=response.status
						)
				
				except Exception as e:
					result = TestResult(
						test_name="database_query",
						start_time=start_time,
						end_time=datetime.now(timezone.utc),
						duration_ms=0,
						success=False,
						error_message=str(e)
					)
				
				results.append(result)
				self._record_result(result)
		
		logger.info(f"Database load test completed: {len(results)} queries")
		return results
	
	def _calculate_statistics(self, results: List[TestResult]) -> Dict[str, Any]:
		"""Calculate performance statistics."""
		if not results:
			return {}
		
		# Response times
		response_times = [r.duration_ms for r in results if r.success]
		
		if not response_times:
			return {"error": "No successful requests"}
		
		# Calculate percentiles
		response_times_sorted = sorted(response_times)
		
		return {
			"total_requests": len(results),
			"successful_requests": len([r for r in results if r.success]),
			"failed_requests": len([r for r in results if not r.success]),
			"avg_response_time_ms": statistics.mean(response_times),
			"min_response_time_ms": min(response_times),
			"max_response_time_ms": max(response_times),
			"p50_response_time_ms": np.percentile(response_times, 50),
			"p95_response_time_ms": np.percentile(response_times, 95),
			"p99_response_time_ms": np.percentile(response_times, 99),
			"response_size_bytes": sum(r.response_size for r in results if r.success),
		}
	
	async def execute_load_test(self) -> LoadTestReport:
		"""Execute comprehensive load test."""
		logger.info("Starting comprehensive load test...")
		
		test_start = datetime.now(timezone.utc)
		
		# Resource monitoring task
		monitor_task = asyncio.create_task(self._monitor_resources_continuously())
		
		try:
			# Execute all test types
			api_results = await self._api_load_test()
			ws_results = await self._websocket_load_test()
			kafka_results = await self._kafka_load_test()
			db_results = await self._database_load_test()
			
			# Combine all results
			all_results = api_results + ws_results + kafka_results + db_results
			
		finally:
			# Stop monitoring
			monitor_task.cancel()
			try:
				await monitor_task
			except asyncio.CancelledError:
				pass
		
		test_end = datetime.now(timezone.utc)
		test_duration = (test_end - test_start).total_seconds()
		
		# Calculate statistics
		stats = self._calculate_statistics(all_results)
		
		# Create report
		report = LoadTestReport(
			test_name="comprehensive_load_test",
			config=self.config,
			start_time=test_start,
			end_time=test_end,
			total_requests=stats.get("total_requests", 0),
			successful_requests=stats.get("successful_requests", 0),
			failed_requests=stats.get("failed_requests", 0),
			error_rate_percent=(stats.get("failed_requests", 0) / max(stats.get("total_requests", 1), 1)) * 100,
			avg_response_time_ms=stats.get("avg_response_time_ms", 0),
			min_response_time_ms=stats.get("min_response_time_ms", 0),
			max_response_time_ms=stats.get("max_response_time_ms", 0),
			p50_response_time_ms=stats.get("p50_response_time_ms", 0),
			p95_response_time_ms=stats.get("p95_response_time_ms", 0),
			p99_response_time_ms=stats.get("p99_response_time_ms", 0),
			requests_per_second=stats.get("total_requests", 0) / test_duration,
			events_per_second=len(kafka_results) / test_duration,
			bytes_per_second=stats.get("response_size_bytes", 0) / test_duration,
			peak_cpu_percent=max(self.cpu_samples) if self.cpu_samples else 0,
			peak_memory_mb=max(self.memory_samples) if self.memory_samples else 0,
			peak_network_mbps=max(self.network_samples) if self.network_samples else 0,
			results=all_results,
			errors=self.errors
		)
		
		logger.info("Comprehensive load test completed")
		return report
	
	async def _monitor_resources_continuously(self):
		"""Continuously monitor resources during test."""
		while True:
			self._monitor_resources()
			await asyncio.sleep(1)

async def run_production_load_tests():
	"""Run production load tests."""
	config = LoadTestConfig(
		concurrent_users=100,
		test_duration_seconds=300,
		events_per_second=1000,
		max_response_time_ms=100,
		max_p95_response_time_ms=200,
		max_error_rate_percent=1.0,
		min_throughput_eps=800
	)
	
	executor = LoadTestExecutor(config)
	
	try:
		await executor.setup()
		report = await executor.execute_load_test()
		
		# Save report
		report_data = asdict(report)
		report_data['results'] = [asdict(r) for r in report.results]
		
		with open(f"load_test_report_{int(time.time())}.json", "w") as f:
			json.dump(report_data, f, indent=2, default=str)
		
		# Print summary
		print("\n" + "="*50)
		print("LOAD TEST RESULTS SUMMARY")
		print("="*50)
		print(f"Test Duration: {(report.end_time - report.start_time).total_seconds():.2f} seconds")
		print(f"Total Requests: {report.total_requests}")
		print(f"Successful: {report.successful_requests}")
		print(f"Failed: {report.failed_requests}")
		print(f"Error Rate: {report.error_rate_percent:.2f}%")
		print(f"Avg Response Time: {report.avg_response_time_ms:.2f}ms")
		print(f"P95 Response Time: {report.p95_response_time_ms:.2f}ms")
		print(f"P99 Response Time: {report.p99_response_time_ms:.2f}ms")
		print(f"Requests/Second: {report.requests_per_second:.2f}")
		print(f"Events/Second: {report.events_per_second:.2f}")
		print(f"Peak CPU: {report.peak_cpu_percent:.2f}%")
		print(f"Peak Memory: {report.peak_memory_mb:.2f}MB")
		
		# Performance validation
		if report.passed_thresholds(config):
			print("\n✅ ALL PERFORMANCE THRESHOLDS PASSED")
		else:
			print("\n❌ PERFORMANCE THRESHOLDS FAILED")
			if report.error_rate_percent > config.max_error_rate_percent:
				print(f"   - Error rate too high: {report.error_rate_percent:.2f}% > {config.max_error_rate_percent}%")
			if report.p95_response_time_ms > config.max_p95_response_time_ms:
				print(f"   - P95 response time too high: {report.p95_response_time_ms:.2f}ms > {config.max_p95_response_time_ms}ms")
			if report.events_per_second < config.min_throughput_eps:
				print(f"   - Throughput too low: {report.events_per_second:.2f} < {config.min_throughput_eps} events/second")
		
		print("="*50)
		
		return report
		
	finally:
		await executor.teardown()

if __name__ == "__main__":
	asyncio.run(run_production_load_tests())