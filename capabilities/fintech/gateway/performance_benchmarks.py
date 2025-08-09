"""
Performance Benchmarking Suite for APG Payment Gateway

Comprehensive performance testing to validate production readiness
and ensure the gateway can handle enterprise-scale transaction volumes.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Dict, Any, Optional
import aiohttp
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
	"""Results from a benchmark test"""
	test_name: str
	total_requests: int
	successful_requests: int
	failed_requests: int
	average_response_time: float
	min_response_time: float
	max_response_time: float
	p95_response_time: float
	p99_response_time: float
	requests_per_second: float
	total_duration: float
	error_rate: float
	throughput_mbps: float
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'test_name': self.test_name,
			'total_requests': self.total_requests,
			'successful_requests': self.successful_requests,
			'failed_requests': self.failed_requests,
			'average_response_time_ms': round(self.average_response_time * 1000, 2),
			'min_response_time_ms': round(self.min_response_time * 1000, 2),
			'max_response_time_ms': round(self.max_response_time * 1000, 2),
			'p95_response_time_ms': round(self.p95_response_time * 1000, 2),
			'p99_response_time_ms': round(self.p99_response_time * 1000, 2),
			'requests_per_second': round(self.requests_per_second, 2),
			'total_duration_seconds': round(self.total_duration, 2),
			'error_rate_percent': round(self.error_rate * 100, 2),
			'throughput_mbps': round(self.throughput_mbps, 2)
		}


class PaymentGatewayBenchmark:
	"""Performance benchmark suite for APG Payment Gateway"""
	
	def __init__(self, base_url: str = "http://localhost:8080", auth_token: Optional[str] = None):
		self.base_url = base_url.rstrip('/')
		self.auth_token = auth_token
		self.session: Optional[aiohttp.ClientSession] = None
		
	async def __aenter__(self):
		connector = aiohttp.TCPConnector(
			limit=1000,  # Connection pool limit
			limit_per_host=100,
			ttl_dns_cache=300,
			use_dns_cache=True,
			keepalive_timeout=30
		)
		
		timeout = aiohttp.ClientTimeout(total=60, connect=10)
		
		headers = {}
		if self.auth_token:
			headers['Authorization'] = f'Bearer {self.auth_token}'
		
		self.session = aiohttp.ClientSession(
			connector=connector,
			timeout=timeout,
			headers=headers
		)
		return self
	
	async def __aexit__(self, exc_type, exc_val, exc_tb):
		if self.session:
			await self.session.close()
	
	async def _make_request(self, method: str, endpoint: str, payload: Optional[Dict] = None) -> Dict[str, Any]:
		"""Make a request and measure response time"""
		url = f"{self.base_url}{endpoint}"
		start_time = time.time()
		
		try:
			async with self.session.request(method, url, json=payload) as response:
				response_time = time.time() - start_time
				response_data = await response.json() if response.content_type == 'application/json' else {}
				
				return {
					'success': response.status < 400,
					'status_code': response.status,
					'response_time': response_time,
					'response_size': len(await response.read()),
					'data': response_data
				}
		except Exception as e:
			response_time = time.time() - start_time
			return {
				'success': False,
				'status_code': 0,
				'response_time': response_time,
				'response_size': 0,
				'error': str(e)
			}
	
	async def _run_concurrent_requests(
		self, 
		method: str, 
		endpoint: str, 
		payload_generator,
		num_requests: int, 
		concurrency: int
	) -> List[Dict[str, Any]]:
		"""Run concurrent requests with specified concurrency level"""
		semaphore = asyncio.Semaphore(concurrency)
		
		async def bounded_request(request_id: int):
			async with semaphore:
				payload = payload_generator(request_id) if callable(payload_generator) else payload_generator
				return await self._make_request(method, endpoint, payload)
		
		tasks = [bounded_request(i) for i in range(num_requests)]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		# Filter out exceptions
		valid_results = [r for r in results if isinstance(r, dict)]
		return valid_results
	
	def _calculate_benchmark_result(self, test_name: str, results: List[Dict[str, Any]], total_duration: float) -> BenchmarkResult:
		"""Calculate benchmark statistics from results"""
		total_requests = len(results)
		successful_requests = sum(1 for r in results if r['success'])
		failed_requests = total_requests - successful_requests
		
		response_times = [r['response_time'] for r in results]
		response_sizes = [r['response_size'] for r in results]
		
		if not response_times:
			response_times = [0]
		
		# Calculate statistics
		avg_response_time = statistics.mean(response_times)
		min_response_time = min(response_times)
		max_response_time = max(response_times)
		
		# Percentiles
		sorted_times = sorted(response_times)
		p95_response_time = sorted_times[int(0.95 * len(sorted_times))] if sorted_times else 0
		p99_response_time = sorted_times[int(0.99 * len(sorted_times))] if sorted_times else 0
		
		# Throughput calculations
		requests_per_second = total_requests / total_duration if total_duration > 0 else 0
		error_rate = failed_requests / total_requests if total_requests > 0 else 0
		
		# Calculate throughput in Mbps
		total_bytes = sum(response_sizes)
		throughput_mbps = (total_bytes * 8) / (total_duration * 1_000_000) if total_duration > 0 else 0
		
		return BenchmarkResult(
			test_name=test_name,
			total_requests=total_requests,
			successful_requests=successful_requests,
			failed_requests=failed_requests,
			average_response_time=avg_response_time,
			min_response_time=min_response_time,
			max_response_time=max_response_time,
			p95_response_time=p95_response_time,
			p99_response_time=p99_response_time,
			requests_per_second=requests_per_second,
			total_duration=total_duration,
			error_rate=error_rate,
			throughput_mbps=throughput_mbps
		)
	
	async def benchmark_health_check(self, num_requests: int = 1000, concurrency: int = 50) -> BenchmarkResult:
		"""Benchmark health check endpoint"""
		logger.info(f"Running health check benchmark: {num_requests} requests, concurrency: {concurrency}")
		
		start_time = time.time()
		results = await self._run_concurrent_requests(
			'GET', '/health', None, num_requests, concurrency
		)
		total_duration = time.time() - start_time
		
		return self._calculate_benchmark_result('Health Check', results, total_duration)
	
	async def benchmark_payment_processing(self, num_requests: int = 500, concurrency: int = 25) -> BenchmarkResult:
		"""Benchmark payment processing endpoint"""
		logger.info(f"Running payment processing benchmark: {num_requests} requests, concurrency: {concurrency}")
		
		def generate_payment_payload(request_id: int) -> Dict[str, Any]:
			return {
				"transaction": {
					"id": f"bench_txn_{request_id}_{int(time.time())}",
					"amount": str(Decimal("100.00")),
					"currency": "USD",
					"description": f"Benchmark payment {request_id}",
					"customer_email": f"bench{request_id}@example.com",
					"customer_name": f"Benchmark Customer {request_id}"
				},
				"payment_method": {
					"method_type": "card",
					"metadata": {
						"card_number": "4111111111111111",
						"exp_month": "12",
						"exp_year": "2025",
						"cvc": "123",
						"cardholder_name": f"Test Customer {request_id}"
					}
				},
				"provider": "stripe"
			}
		
		start_time = time.time()
		results = await self._run_concurrent_requests(
			'POST', '/api/v1/payments', generate_payment_payload, num_requests, concurrency
		)
		total_duration = time.time() - start_time
		
		return self._calculate_benchmark_result('Payment Processing', results, total_duration)
	
	async def benchmark_payment_verification(self, num_requests: int = 1000, concurrency: int = 50) -> BenchmarkResult:
		"""Benchmark payment verification endpoint"""
		logger.info(f"Running payment verification benchmark: {num_requests} requests, concurrency: {concurrency}")
		
		def generate_verification_payload(request_id: int) -> Dict[str, Any]:
			return {
				"transaction_id": f"pi_test_{request_id}",
				"provider": "stripe"
			}
		
		start_time = time.time()
		results = await self._run_concurrent_requests(
			'POST', '/api/v1/payments/verify', generate_verification_payload, num_requests, concurrency
		)
		total_duration = time.time() - start_time
		
		return self._calculate_benchmark_result('Payment Verification', results, total_duration)
	
	async def benchmark_webhook_processing(self, num_requests: int = 2000, concurrency: int = 100) -> BenchmarkResult:
		"""Benchmark webhook processing endpoint"""
		logger.info(f"Running webhook processing benchmark: {num_requests} requests, concurrency: {concurrency}")
		
		def generate_webhook_payload(request_id: int) -> Dict[str, Any]:
			return {
				"type": "payment_intent.succeeded",
				"data": {
					"object": {
						"id": f"pi_webhook_{request_id}",
						"status": "succeeded",
						"amount": 10000,
						"currency": "usd",
						"metadata": {"request_id": str(request_id)}
					}
				}
			}
		
		start_time = time.time()
		results = await self._run_concurrent_requests(
			'POST', '/webhooks/stripe', generate_webhook_payload, num_requests, concurrency
		)
		total_duration = time.time() - start_time
		
		return self._calculate_benchmark_result('Webhook Processing', results, total_duration)
	
	async def benchmark_provider_health_checks(self, num_requests: int = 500, concurrency: int = 25) -> BenchmarkResult:
		"""Benchmark provider health check endpoint"""
		logger.info(f"Running provider health checks benchmark: {num_requests} requests, concurrency: {concurrency}")
		
		start_time = time.time()
		results = await self._run_concurrent_requests(
			'GET', '/api/v1/providers/health', None, num_requests, concurrency
		)
		total_duration = time.time() - start_time
		
		return self._calculate_benchmark_result('Provider Health Checks', results, total_duration)
	
	async def benchmark_metrics_endpoint(self, num_requests: int = 1000, concurrency: int = 50) -> BenchmarkResult:
		"""Benchmark metrics endpoint"""
		logger.info(f"Running metrics endpoint benchmark: {num_requests} requests, concurrency: {concurrency}")
		
		start_time = time.time()
		results = await self._run_concurrent_requests(
			'GET', '/metrics', None, num_requests, concurrency
		)
		total_duration = time.time() - start_time
		
		return self._calculate_benchmark_result('Metrics Endpoint', results, total_duration)
	
	async def run_stress_test(self, duration_seconds: int = 300, concurrency: int = 100) -> Dict[str, Any]:
		"""Run stress test for specified duration"""
		logger.info(f"Running stress test: {duration_seconds} seconds, concurrency: {concurrency}")
		
		end_time = time.time() + duration_seconds
		results = []
		request_count = 0
		
		async def stress_request():
			nonlocal request_count
			request_count += 1
			payload = {
				"transaction": {
					"id": f"stress_txn_{request_count}_{int(time.time())}",
					"amount": "50.00",
					"currency": "USD",
					"description": f"Stress test payment {request_count}"
				},
				"payment_method": {
					"method_type": "card",
					"metadata": {"card_number": "4111111111111111"}
				},
				"provider": "stripe"
			}
			return await self._make_request('POST', '/api/v1/payments', payload)
		
		semaphore = asyncio.Semaphore(concurrency)
		
		async def bounded_stress_request():
			async with semaphore:
				return await stress_request()
		
		start_time = time.time()
		
		while time.time() < end_time:
			batch_size = min(100, int((end_time - time.time()) * concurrency / 10))
			if batch_size <= 0:
				break
			
			tasks = [bounded_stress_request() for _ in range(batch_size)]
			batch_results = await asyncio.gather(*tasks, return_exceptions=True)
			
			valid_results = [r for r in batch_results if isinstance(r, dict)]
			results.extend(valid_results)
			
			# Small delay to prevent overwhelming
			await asyncio.sleep(0.1)
		
		total_duration = time.time() - start_time
		benchmark_result = self._calculate_benchmark_result('Stress Test', results, total_duration)
		
		return {
			'benchmark_result': benchmark_result,
			'duration_seconds': total_duration,
			'target_concurrency': concurrency,
			'actual_requests': len(results)
		}
	
	async def run_load_ramp_test(self, max_concurrency: int = 200, ramp_duration: int = 600) -> List[Dict[str, Any]]:
		"""Run load ramp test gradually increasing concurrency"""
		logger.info(f"Running load ramp test: max concurrency {max_concurrency}, duration {ramp_duration}s")
		
		results = []
		step_duration = 60  # 1 minute per step
		steps = ramp_duration // step_duration
		concurrency_step = max_concurrency // steps
		
		for step in range(1, steps + 1):
			current_concurrency = step * concurrency_step
			logger.info(f"Ramp step {step}/{steps}: concurrency {current_concurrency}")
			
			# Run benchmark for this concurrency level
			step_results = await self._run_concurrent_requests(
				'GET', '/health', None, current_concurrency * 10, current_concurrency
			)
			
			step_result = self._calculate_benchmark_result(
				f'Ramp Step {step}', step_results, step_duration
			)
			
			results.append({
				'step': step,
				'concurrency': current_concurrency,
				'benchmark_result': step_result
			})
			
			# Brief pause between steps
			await asyncio.sleep(5)
		
		return results


async def run_comprehensive_benchmark_suite():
	"""Run the complete benchmark suite"""
	logger.info("Starting comprehensive benchmark suite for APG Payment Gateway")
	
	async with PaymentGatewayBenchmark() as benchmark:
		results = {}
		
		# Basic performance benchmarks
		logger.info("Running basic performance benchmarks...")
		results['health_check'] = await benchmark.benchmark_health_check()
		results['payment_processing'] = await benchmark.benchmark_payment_processing()
		results['payment_verification'] = await benchmark.benchmark_payment_verification()
		results['webhook_processing'] = await benchmark.benchmark_webhook_processing()
		results['provider_health'] = await benchmark.benchmark_provider_health_checks()
		results['metrics_endpoint'] = await benchmark.benchmark_metrics_endpoint()
		
		# Stress testing
		logger.info("Running stress test...")
		results['stress_test'] = await benchmark.run_stress_test(duration_seconds=180, concurrency=50)
		
		# Load ramp testing
		logger.info("Running load ramp test...")
		results['load_ramp'] = await benchmark.run_load_ramp_test(max_concurrency=100, ramp_duration=300)
		
		return results


def print_benchmark_results(results: Dict[str, Any]):
	"""Print formatted benchmark results"""
	print("\n" + "="*80)
	print("APG PAYMENT GATEWAY - PERFORMANCE BENCHMARK RESULTS")
	print("="*80)
	
	# Basic benchmarks
	basic_tests = ['health_check', 'payment_processing', 'payment_verification', 
				   'webhook_processing', 'provider_health', 'metrics_endpoint']
	
	for test_name in basic_tests:
		if test_name in results:
			result = results[test_name]
			print(f"\n{result.test_name.upper()}")
			print("-" * 50)
			print(f"Total Requests: {result.total_requests:,}")
			print(f"Successful: {result.successful_requests:,} ({100 - result.error_rate * 100:.1f}%)")
			print(f"Failed: {result.failed_requests:,} ({result.error_rate * 100:.1f}%)")
			print(f"Requests/sec: {result.requests_per_second:.1f}")
			print(f"Avg Response: {result.average_response_time * 1000:.1f}ms")
			print(f"P95 Response: {result.p95_response_time * 1000:.1f}ms")
			print(f"P99 Response: {result.p99_response_time * 1000:.1f}ms")
			print(f"Throughput: {result.throughput_mbps:.2f} Mbps")
	
	# Stress test results
	if 'stress_test' in results:
		stress_result = results['stress_test']['benchmark_result']
		print(f"\nSTRESS TEST")
		print("-" * 50)
		print(f"Duration: {results['stress_test']['duration_seconds']:.1f}s")
		print(f"Total Requests: {stress_result.total_requests:,}")
		print(f"Requests/sec: {stress_result.requests_per_second:.1f}")
		print(f"Success Rate: {100 - stress_result.error_rate * 100:.1f}%")
		print(f"P95 Response: {stress_result.p95_response_time * 1000:.1f}ms")
	
	# Performance summary
	print(f"\nPERFORMANCE SUMMARY")
	print("-" * 50)
	
	# Calculate overall metrics
	health_check = results.get('health_check')
	payment_processing = results.get('payment_processing')
	webhook_processing = results.get('webhook_processing')
	
	if health_check:
		print(f"Health Check RPS: {health_check.requests_per_second:.0f}")
	if payment_processing:
		print(f"Payment Processing RPS: {payment_processing.requests_per_second:.0f}")
	if webhook_processing:
		print(f"Webhook Processing RPS: {webhook_processing.requests_per_second:.0f}")
	
	# Performance rating
	min_rps = 100  # Minimum acceptable RPS
	good_rps = 500  # Good performance RPS
	excellent_rps = 1000  # Excellent performance RPS
	
	avg_rps = statistics.mean([
		r.requests_per_second for r in [health_check, payment_processing, webhook_processing] 
		if r is not None
	])
	
	if avg_rps >= excellent_rps:
		rating = "EXCELLENT"
	elif avg_rps >= good_rps:
		rating = "GOOD"
	elif avg_rps >= min_rps:
		rating = "ACCEPTABLE"
	else:
		rating = "NEEDS IMPROVEMENT"
	
	print(f"\nOVERALL PERFORMANCE RATING: {rating}")
	print(f"Average RPS across tests: {avg_rps:.0f}")
	
	print("\n" + "="*80)


async def main():
	"""Main benchmark execution"""
	try:
		results = await run_comprehensive_benchmark_suite()
		print_benchmark_results(results)
		
		# Save results to file
		import json
		from datetime import datetime
		
		# Convert results to JSON-serializable format
		json_results = {}
		for key, value in results.items():
			if key == 'load_ramp':
				json_results[key] = [
					{
						'step': item['step'],
						'concurrency': item['concurrency'],
						'benchmark_result': item['benchmark_result'].to_dict()
					}
					for item in value
				]
			elif key == 'stress_test':
				json_results[key] = {
					'benchmark_result': value['benchmark_result'].to_dict(),
					'duration_seconds': value['duration_seconds'],
					'target_concurrency': value['target_concurrency'],
					'actual_requests': value['actual_requests']
				}
			else:
				json_results[key] = value.to_dict()
		
		# Add metadata
		json_results['metadata'] = {
			'timestamp': datetime.now().isoformat(),
			'version': '1.0.0',
			'test_suite': 'APG Payment Gateway Benchmark'
		}
		
		with open(f'benchmark_results_{int(time.time())}.json', 'w') as f:
			json.dump(json_results, f, indent=2)
		
		logger.info("Benchmark results saved to file")
		
	except Exception as e:
		logger.error(f"Benchmark failed: {e}")
		raise


if __name__ == "__main__":
	asyncio.run(main())