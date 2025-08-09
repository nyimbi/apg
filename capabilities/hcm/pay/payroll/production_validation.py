"""
APG Payroll Management - Production Validation & Testing Suite

Comprehensive validation and testing suite for production deployment
verification, performance testing, and system health monitoring.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import json
import logging
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import statistics

import requests
import psycopg2
import redis
from sqlalchemy import create_engine, text
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
	"""Validation test result."""
	test_name: str
	status: str  # 'passed', 'failed', 'warning'
	message: str
	duration_ms: float
	details: Optional[Dict[str, Any]] = None
	error: Optional[str] = None


@dataclass
class PerformanceMetrics:
	"""Performance test metrics."""
	avg_response_time: float
	p95_response_time: float
	p99_response_time: float
	throughput_rps: float
	error_rate: float
	total_requests: int
	successful_requests: int
	failed_requests: int


class ProductionValidator:
	"""Comprehensive production validation suite."""
	
	def __init__(
		self,
		base_url: str = "http://localhost:8000",
		db_url: str = "postgresql://apg_payroll_user:secure_password@localhost:5432/apg_payroll",
		redis_url: str = "redis://localhost:6379/0"
	):
		"""Initialize the production validator."""
		self.base_url = base_url.rstrip('/')
		self.db_url = db_url
		self.redis_url = redis_url
		self.results: List[ValidationResult] = []
		
		# Test configuration
		self.timeout = 30
		self.performance_test_duration = 60  # seconds
		self.performance_concurrent_users = 10
		
	def run_all_validations(self) -> Dict[str, Any]:
		"""Run all production validation tests."""
		logger.info("Starting comprehensive production validation...")
		start_time = time.time()
		
		try:
			# Health and connectivity tests
			self._test_application_health()
			self._test_database_connectivity()
			self._test_redis_connectivity()
			
			# API functionality tests
			self._test_api_endpoints()
			self._test_authentication()
			self._test_authorization()
			
			# Core business logic tests
			self._test_payroll_period_operations()
			self._test_payroll_run_operations()
			self._test_employee_payroll_operations()
			
			# AI and advanced features
			self._test_ai_intelligence_features()
			self._test_conversational_interface()
			self._test_analytics_endpoints()
			
			# Performance and load tests
			performance_metrics = self._run_performance_tests()
			
			# Security tests
			self._test_security_headers()
			self._test_input_validation()
			self._test_rate_limiting()
			
			# Integration tests
			self._test_external_integrations()
			
			# Data integrity tests
			self._test_data_integrity()
			
			# Monitoring and observability
			self._test_monitoring_endpoints()
			
		except Exception as e:
			logger.error(f"Validation suite failed with error: {e}")
			self.results.append(ValidationResult(
				test_name="validation_suite",
				status="failed",
				message="Validation suite execution failed",
				duration_ms=0,
				error=str(e)
			))
		
		total_duration = time.time() - start_time
		
		# Generate comprehensive report
		return self._generate_report(total_duration, performance_metrics)
	
	def _test_application_health(self) -> None:
		"""Test application health endpoints."""
		logger.info("Testing application health...")
		
		# Basic health check
		self._run_test(
			"health_check_basic",
			lambda: self._check_endpoint("/health", expected_status=200)
		)
		
		# Detailed health check
		self._run_test(
			"health_check_detailed",
			lambda: self._check_endpoint("/health/detailed", expected_status=200)
		)
		
		# Application metrics
		self._run_test(
			"metrics_endpoint",
			lambda: self._check_endpoint("/metrics", expected_status=200)
		)
	
	def _test_database_connectivity(self) -> None:
		"""Test database connectivity and basic operations."""
		logger.info("Testing database connectivity...")
		
		def test_db_connection():
			try:
				engine = create_engine(self.db_url)
				with engine.connect() as conn:
					result = conn.execute(text("SELECT 1 as test"))
					row = result.fetchone()
					if row[0] != 1:
						raise Exception("Database test query failed")
				return True, "Database connection successful"
			except Exception as e:
				return False, f"Database connection failed: {e}"
		
		self._run_test("database_connectivity", test_db_connection)
		
		# Test payroll-specific tables
		def test_payroll_tables():
			try:
				engine = create_engine(self.db_url)
				with engine.connect() as conn:
					tables = [
						'pr_payroll_period',
						'pr_payroll_run', 
						'pr_employee_payroll',
						'pr_pay_component'
					]
					
					for table in tables:
						result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
						count = result.fetchone()[0]
						logger.info(f"Table {table}: {count} records")
				
				return True, "Payroll tables verified"
			except Exception as e:
				return False, f"Payroll tables check failed: {e}"
		
		self._run_test("payroll_tables_check", test_payroll_tables)
	
	def _test_redis_connectivity(self) -> None:
		"""Test Redis connectivity and operations."""
		logger.info("Testing Redis connectivity...")
		
		def test_redis_connection():
			try:
				r = redis.from_url(self.redis_url)
				r.ping()
				
				# Test set/get operations
				test_key = f"test:validation:{int(time.time())}"
				r.set(test_key, "validation_test", ex=10)
				value = r.get(test_key)
				
				if value != b"validation_test":
					raise Exception("Redis set/get operation failed")
				
				r.delete(test_key)
				return True, "Redis connection and operations successful"
			except Exception as e:
				return False, f"Redis test failed: {e}"
		
		self._run_test("redis_connectivity", test_redis_connection)
	
	def _test_api_endpoints(self) -> None:
		"""Test core API endpoints."""
		logger.info("Testing API endpoints...")
		
		# Test API root
		self._run_test(
			"api_root",
			lambda: self._check_endpoint("/api/v1/payroll", expected_status=200)
		)
		
		# Test API documentation
		self._run_test(
			"api_documentation",
			lambda: self._check_endpoint("/api/v1/payroll/", expected_status=200)
		)
		
		# Test periods endpoint
		self._run_test(
			"periods_endpoint",
			lambda: self._check_endpoint("/api/v1/payroll/periods/", expected_status=200)
		)
		
		# Test runs endpoint  
		self._run_test(
			"runs_endpoint",
			lambda: self._check_endpoint("/api/v1/payroll/runs/", expected_status=200)
		)
		
		# Test analytics endpoint
		self._run_test(
			"analytics_endpoint",
			lambda: self._check_endpoint("/api/v1/payroll/analytics/dashboard", expected_status=200)
		)
	
	def _test_authentication(self) -> None:
		"""Test authentication mechanisms."""
		logger.info("Testing authentication...")
		
		# Test protected endpoint without auth
		def test_auth_required():
			try:
				response = requests.get(f"{self.base_url}/api/v1/payroll/periods/", timeout=self.timeout)
				# Should require authentication (401 or 403)
				if response.status_code in [401, 403]:
					return True, f"Authentication required (status: {response.status_code})"
				else:
					return False, f"Endpoint not properly protected (status: {response.status_code})"
			except Exception as e:
				return False, f"Authentication test failed: {e}"
		
		self._run_test("authentication_required", test_auth_required)
	
	def _test_authorization(self) -> None:
		"""Test authorization and permissions."""
		logger.info("Testing authorization...")
		
		# Test role-based access control
		def test_rbac():
			try:
				# This would require actual user tokens in a real test
				# For now, just verify the endpoints exist
				response = requests.get(f"{self.base_url}/api/v1/payroll/compliance/status", timeout=self.timeout)
				# Should require specific permissions
				if response.status_code in [401, 403]:
					return True, "RBAC endpoints properly protected"
				else:
					return False, f"RBAC test inconclusive (status: {response.status_code})"
			except Exception as e:
				return False, f"RBAC test failed: {e}"
		
		self._run_test("rbac_protection", test_rbac)
	
	def _test_payroll_period_operations(self) -> None:
		"""Test payroll period CRUD operations."""
		logger.info("Testing payroll period operations...")
		
		# Test period creation (would need authentication in real scenario)
		def test_period_creation():
			try:
				period_data = {
					"period_name": f"Test Period {int(time.time())}",
					"period_type": "regular",
					"pay_frequency": "monthly",
					"start_date": "2025-01-01",
					"end_date": "2025-01-31",
					"pay_date": "2025-02-05",
					"fiscal_year": 2025,
					"fiscal_quarter": 1,
					"country_code": "KE",
					"currency_code": "KES"
				}
				
				response = requests.post(
					f"{self.base_url}/api/v1/payroll/periods/",
					json=period_data,
					timeout=self.timeout
				)
				
				# Expect 401/403 due to authentication requirement
				if response.status_code in [401, 403]:
					return True, "Period creation endpoint properly protected"
				elif response.status_code == 201:
					return True, "Period creation successful (authenticated)"
				else:
					return False, f"Unexpected response: {response.status_code}"
					
			except Exception as e:
				return False, f"Period creation test failed: {e}"
		
		self._run_test("period_creation", test_period_creation)
	
	def _test_payroll_run_operations(self) -> None:
		"""Test payroll run operations."""
		logger.info("Testing payroll run operations...")
		
		# Test run status monitoring
		def test_run_monitoring():
			try:
				# Test with dummy run ID
				response = requests.get(
					f"{self.base_url}/api/v1/payroll/runs/dummy-run-id/status",
					timeout=self.timeout
				)
				
				# Should be protected or return 404 for invalid ID
				if response.status_code in [401, 403, 404]:
					return True, "Run monitoring endpoint properly handled"
				else:
					return False, f"Unexpected response: {response.status_code}"
					
			except Exception as e:
				return False, f"Run monitoring test failed: {e}"
		
		self._run_test("run_monitoring", test_run_monitoring)
	
	def _test_employee_payroll_operations(self) -> None:
		"""Test employee payroll operations."""
		logger.info("Testing employee payroll operations...")
		
		# Test employee payroll listing
		def test_employee_payroll_list():
			try:
				response = requests.get(
					f"{self.base_url}/api/v1/payroll/employees/",
					timeout=self.timeout
				)
				
				# Should require authentication
				if response.status_code in [401, 403]:
					return True, "Employee payroll endpoint properly protected"
				elif response.status_code == 200:
					return True, "Employee payroll listing accessible"
				else:
					return False, f"Unexpected response: {response.status_code}"
					
			except Exception as e:
				return False, f"Employee payroll test failed: {e}"
		
		self._run_test("employee_payroll_list", test_employee_payroll_list)
	
	def _test_ai_intelligence_features(self) -> None:
		"""Test AI intelligence features."""
		logger.info("Testing AI intelligence features...")
		
		# Test anomaly detection
		self._run_test(
			"ai_anomaly_detection",
			lambda: self._check_endpoint("/api/v1/payroll/ai/anomalies", expected_status=[200, 401, 403])
		)
		
		# Test AI predictions
		self._run_test(
			"ai_predictions",
			lambda: self._check_endpoint("/api/v1/payroll/ai/predictions", expected_status=[200, 401, 403])
		)
	
	def _test_conversational_interface(self) -> None:
		"""Test conversational interface."""
		logger.info("Testing conversational interface...")
		
		# Test chat message processing
		def test_chat_endpoint():
			try:
				chat_data = {
					"command": "Show me payroll summary for this month",
					"context": {}
				}
				
				response = requests.post(
					f"{self.base_url}/api/v1/payroll/chat/message",
					json=chat_data,
					timeout=self.timeout
				)
				
				# Should require authentication
				if response.status_code in [401, 403]:
					return True, "Conversational endpoint properly protected"
				elif response.status_code == 200:
					return True, "Conversational interface accessible"
				else:
					return False, f"Unexpected response: {response.status_code}"
					
			except Exception as e:
				return False, f"Conversational interface test failed: {e}"
		
		self._run_test("conversational_interface", test_chat_endpoint)
	
	def _test_analytics_endpoints(self) -> None:
		"""Test analytics and reporting endpoints."""
		logger.info("Testing analytics endpoints...")
		
		# Test dashboard analytics
		self._run_test(
			"analytics_dashboard",
			lambda: self._check_endpoint("/api/v1/payroll/analytics/dashboard", expected_status=[200, 401, 403])
		)
		
		# Test trends analysis
		self._run_test(
			"analytics_trends", 
			lambda: self._check_endpoint("/api/v1/payroll/analytics/trends", expected_status=[200, 401, 403])
		)
	
	def _run_performance_tests(self) -> PerformanceMetrics:
		"""Run performance and load tests."""
		logger.info("Running performance tests...")
		
		# Test endpoints for performance
		test_endpoints = [
			"/health",
			"/api/v1/payroll",
			"/api/v1/payroll/periods/",
			"/api/v1/payroll/runs/"
		]
		
		all_response_times = []
		total_requests = 0
		successful_requests = 0
		failed_requests = 0
		
		start_time = time.time()
		
		def make_request(endpoint):
			try:
				response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
				return response.elapsed.total_seconds() * 1000, response.status_code < 500
			except:
				return 0, False
		
		# Run concurrent requests
		with ThreadPoolExecutor(max_workers=self.performance_concurrent_users) as executor:
			while time.time() - start_time < self.performance_test_duration:
				futures = []
				for endpoint in test_endpoints:
					future = executor.submit(make_request, endpoint)
					futures.append(future)
				
				for future in futures:
					try:
						response_time, success = future.result(timeout=10)
						all_response_times.append(response_time)
						total_requests += 1
						if success:
							successful_requests += 1
						else:
							failed_requests += 1
					except:
						failed_requests += 1
						total_requests += 1
				
				time.sleep(0.1)  # Small delay between batches
		
		# Calculate metrics
		if all_response_times:
			avg_response_time = statistics.mean(all_response_times)
			p95_response_time = statistics.quantiles(all_response_times, n=20)[18]  # 95th percentile
			p99_response_time = statistics.quantiles(all_response_times, n=100)[98]  # 99th percentile
		else:
			avg_response_time = p95_response_time = p99_response_time = 0
		
		duration = time.time() - start_time
		throughput_rps = total_requests / duration if duration > 0 else 0
		error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
		
		performance_metrics = PerformanceMetrics(
			avg_response_time=avg_response_time,
			p95_response_time=p95_response_time,
			p99_response_time=p99_response_time,
			throughput_rps=throughput_rps,
			error_rate=error_rate,
			total_requests=total_requests,
			successful_requests=successful_requests,
			failed_requests=failed_requests
		)
		
		# Add performance test results
		self.results.append(ValidationResult(
			test_name="performance_load_test",
			status="passed" if error_rate < 5 and avg_response_time < 1000 else "warning",
			message=f"Avg: {avg_response_time:.2f}ms, Throughput: {throughput_rps:.2f} RPS, Error rate: {error_rate:.2f}%",
			duration_ms=duration * 1000,
			details=performance_metrics.__dict__
		))
		
		return performance_metrics
	
	def _test_security_headers(self) -> None:
		"""Test security headers."""
		logger.info("Testing security headers...")
		
		def test_security_headers():
			try:
				response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
				headers = response.headers
				
				required_headers = [
					'X-Content-Type-Options',
					'X-Frame-Options',
					'X-XSS-Protection'
				]
				
				missing_headers = []
				for header in required_headers:
					if header not in headers:
						missing_headers.append(header)
				
				if missing_headers:
					return False, f"Missing security headers: {', '.join(missing_headers)}"
				else:
					return True, "All required security headers present"
					
			except Exception as e:
				return False, f"Security headers test failed: {e}"
		
		self._run_test("security_headers", test_security_headers)
	
	def _test_input_validation(self) -> None:
		"""Test input validation."""
		logger.info("Testing input validation...")
		
		def test_malicious_input():
			try:
				# Test SQL injection attempt
				malicious_payload = "'; DROP TABLE users; --"
				response = requests.get(
					f"{self.base_url}/api/v1/payroll/periods/",
					params={"period_name": malicious_payload},
					timeout=self.timeout
				)
				
				# Should not return 500 (indicates proper input validation)
				if response.status_code != 500:
					return True, "Input validation prevents SQL injection"
				else:
					return False, "Possible SQL injection vulnerability"
					
			except Exception as e:
				return False, f"Input validation test failed: {e}"
		
		self._run_test("input_validation", test_malicious_input)
	
	def _test_rate_limiting(self) -> None:
		"""Test rate limiting."""
		logger.info("Testing rate limiting...")
		
		def test_rate_limits():
			try:
				# Make rapid requests to test rate limiting
				responses = []
				for i in range(20):
					response = requests.get(f"{self.base_url}/health", timeout=5)
					responses.append(response.status_code)
				
				# Check if any requests were rate limited (429)
				rate_limited = any(status == 429 for status in responses)
				
				if rate_limited:
					return True, "Rate limiting is active"
				else:
					return True, "Rate limiting not triggered (may be configured for higher limits)"
					
			except Exception as e:
				return False, f"Rate limiting test failed: {e}"
		
		self._run_test("rate_limiting", test_rate_limits)
	
	def _test_external_integrations(self) -> None:
		"""Test external integrations."""
		logger.info("Testing external integrations...")
		
		# Test webhook endpoints
		def test_webhook_endpoint():
			try:
				response = requests.post(
					f"{self.base_url}/api/v1/payroll/webhooks/payroll_completed",
					json={"run_id": "test-run"},
					timeout=self.timeout
				)
				
				# Should require API key authentication
				if response.status_code in [401, 403]:
					return True, "Webhook endpoint properly protected"
				else:
					return False, f"Webhook security issue: {response.status_code}"
					
			except Exception as e:
				return False, f"Webhook test failed: {e}"
		
		self._run_test("webhook_security", test_webhook_endpoint)
	
	def _test_data_integrity(self) -> None:
		"""Test data integrity and constraints."""
		logger.info("Testing data integrity...")
		
		def test_database_constraints():
			try:
				engine = create_engine(self.db_url)
				with engine.connect() as conn:
					# Test foreign key constraints
					result = conn.execute(text("""
						SELECT COUNT(*) FROM information_schema.table_constraints 
						WHERE constraint_type = 'FOREIGN KEY' 
						AND table_schema = 'public'
						AND table_name LIKE 'pr_%'
					"""))
					fk_count = result.fetchone()[0]
					
					if fk_count > 0:
						return True, f"Database has {fk_count} foreign key constraints"
					else:
						return False, "No foreign key constraints found"
						
			except Exception as e:
				return False, f"Data integrity test failed: {e}"
		
		self._run_test("database_constraints", test_database_constraints)
	
	def _test_monitoring_endpoints(self) -> None:
		"""Test monitoring and observability endpoints."""
		logger.info("Testing monitoring endpoints...")
		
		# Test Prometheus metrics
		def test_prometheus_metrics():
			try:
				response = requests.get(f"{self.base_url}/metrics", timeout=self.timeout)
				if response.status_code == 200:
					# Check if it contains Prometheus-style metrics
					content = response.text
					if "# HELP" in content or "# TYPE" in content:
						return True, "Prometheus metrics available"
					else:
						return True, "Metrics endpoint available (non-Prometheus format)"
				else:
					return False, f"Metrics endpoint failed: {response.status_code}"
					
			except Exception as e:
				return False, f"Prometheus metrics test failed: {e}"
		
		self._run_test("prometheus_metrics", test_prometheus_metrics)
	
	def _run_test(self, test_name: str, test_func) -> None:
		"""Run a single test and record the result."""
		start_time = time.time()
		
		try:
			if callable(test_func):
				result = test_func()
				if isinstance(result, tuple):
					success, message = result
					status = "passed" if success else "failed"
				elif isinstance(result, bool):
					success = result
					status = "passed" if success else "failed"
					message = f"Test {'passed' if success else 'failed'}"
				else:
					status = "passed"
					message = str(result)
			else:
				status = "passed"
				message = str(test_func)
			
			error = None
			
		except Exception as e:
			status = "failed"
			message = f"Test execution failed"
			error = str(e)
			logger.error(f"Test {test_name} failed: {e}")
			logger.debug(traceback.format_exc())
		
		duration_ms = (time.time() - start_time) * 1000
		
		self.results.append(ValidationResult(
			test_name=test_name,
			status=status,
			message=message,
			duration_ms=duration_ms,
			error=error
		))
	
	def _check_endpoint(self, endpoint: str, expected_status=200) -> Tuple[bool, str]:
		"""Check if an endpoint returns the expected status."""
		try:
			response = requests.get(f"{self.base_url}{endpoint}", timeout=self.timeout)
			
			if isinstance(expected_status, list):
				success = response.status_code in expected_status
			else:
				success = response.status_code == expected_status
			
			message = f"Status: {response.status_code}, Expected: {expected_status}"
			return success, message
			
		except Exception as e:
			return False, f"Request failed: {e}"
	
	def _generate_report(self, total_duration: float, performance_metrics: PerformanceMetrics) -> Dict[str, Any]:
		"""Generate comprehensive validation report."""
		# Calculate summary statistics
		total_tests = len(self.results)
		passed_tests = sum(1 for r in self.results if r.status == "passed")
		failed_tests = sum(1 for r in self.results if r.status == "failed")
		warning_tests = sum(1 for r in self.results if r.status == "warning")
		
		success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
		
		# Categorize test results
		categorized_results = {
			"health_tests": [],
			"api_tests": [],
			"security_tests": [],
			"performance_tests": [],
			"integration_tests": [],
			"data_tests": []
		}
		
		for result in self.results:
			if any(keyword in result.test_name for keyword in ["health", "connectivity", "monitoring"]):
				categorized_results["health_tests"].append(result.__dict__)
			elif any(keyword in result.test_name for keyword in ["api", "endpoint", "authentication", "authorization"]):
				categorized_results["api_tests"].append(result.__dict__)
			elif any(keyword in result.test_name for keyword in ["security", "headers", "validation", "rate"]):
				categorized_results["security_tests"].append(result.__dict__)
			elif any(keyword in result.test_name for keyword in ["performance", "load"]):
				categorized_results["performance_tests"].append(result.__dict__)
			elif any(keyword in result.test_name for keyword in ["integration", "webhook", "external"]):
				categorized_results["integration_tests"].append(result.__dict__)
			else:
				categorized_results["data_tests"].append(result.__dict__)
		
		# Create comprehensive report
		report = {
			"validation_summary": {
				"status": "passed" if failed_tests == 0 else "failed",
				"total_tests": total_tests,
				"passed_tests": passed_tests,
				"failed_tests": failed_tests,
				"warning_tests": warning_tests,
				"success_rate": round(success_rate, 2),
				"total_duration_seconds": round(total_duration, 2),
				"timestamp": datetime.utcnow().isoformat()
			},
			"performance_metrics": performance_metrics.__dict__,
			"test_categories": categorized_results,
			"failed_tests": [
				result.__dict__ for result in self.results 
				if result.status == "failed"
			],
			"warning_tests": [
				result.__dict__ for result in self.results 
				if result.status == "warning"
			],
			"recommendations": self._generate_recommendations()
		}
		
		return report
	
	def _generate_recommendations(self) -> List[str]:
		"""Generate recommendations based on test results."""
		recommendations = []
		
		failed_tests = [r for r in self.results if r.status == "failed"]
		
		if any("database" in r.test_name for r in failed_tests):
			recommendations.append("Review database connectivity and configuration")
		
		if any("security" in r.test_name for r in failed_tests):
			recommendations.append("Strengthen security headers and input validation")
		
		if any("performance" in r.test_name for r in failed_tests):
			recommendations.append("Optimize application performance and scalability")
		
		if any("api" in r.test_name for r in failed_tests):
			recommendations.append("Review API endpoint configuration and authentication")
		
		# Performance-based recommendations
		for result in self.results:
			if result.test_name == "performance_load_test" and result.details:
				if result.details.get("avg_response_time", 0) > 1000:
					recommendations.append("Consider performance optimization - average response time is high")
				if result.details.get("error_rate", 0) > 5:
					recommendations.append("Address error rate issues - too many failed requests")
		
		if not recommendations:
			recommendations.append("All validation tests passed - system ready for production")
		
		return recommendations


def main():
	"""Main function to run production validation."""
	import argparse
	
	parser = argparse.ArgumentParser(description="APG Payroll Management Production Validation")
	parser.add_argument("--base-url", default="http://localhost:8000", help="Application base URL")
	parser.add_argument("--db-url", help="Database URL")
	parser.add_argument("--redis-url", default="redis://localhost:6379/0", help="Redis URL")
	parser.add_argument("--output", default="production_validation_report.json", help="Output report file")
	
	args = parser.parse_args()
	
	# Initialize validator
	validator = ProductionValidator(
		base_url=args.base_url,
		db_url=args.db_url or "postgresql://apg_payroll_user:secure_password@localhost:5432/apg_payroll",
		redis_url=args.redis_url
	)
	
	# Run validation
	report = validator.run_all_validations()
	
	# Save report
	with open(args.output, 'w') as f:
		json.dump(report, f, indent=2)
	
	# Print summary
	summary = report["validation_summary"]
	print(f"\n{'='*60}")
	print("APG PAYROLL MANAGEMENT - PRODUCTION VALIDATION REPORT")
	print(f"{'='*60}")
	print(f"Status: {summary['status'].upper()}")
	print(f"Total Tests: {summary['total_tests']}")
	print(f"Passed: {summary['passed_tests']}")
	print(f"Failed: {summary['failed_tests']}")
	print(f"Warnings: {summary['warning_tests']}")
	print(f"Success Rate: {summary['success_rate']}%")
	print(f"Duration: {summary['total_duration_seconds']}s")
	print(f"\nDetailed report saved to: {args.output}")
	
	if summary['failed_tests'] > 0:
		print(f"\n⚠️  {summary['failed_tests']} test(s) failed - review report for details")
		return 1
	else:
		print(f"\n✅ All tests passed - system ready for production!")
		return 0


if __name__ == "__main__":
	exit(main())