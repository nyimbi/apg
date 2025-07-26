"""
APG Accounts Receivable - Performance Load Tests
Load testing and performance validation for AR capability

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
import time
import statistics
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from uuid_extensions import uuid7str

from ..models import (
	ARCustomer, ARInvoice, ARPayment, ARCollectionActivity,
	ARCustomerType, ARCustomerStatus, ARInvoiceStatus, ARPaymentStatus
)
from ..service import (
	ARCustomerService, ARInvoiceService, ARCollectionsService,
	ARCashApplicationService, ARAnalyticsService
)
from ..ai_credit_scoring import APGCreditScoringService, CreditScoringResult
from ..ai_collections_optimization import APGCollectionsAIService
from ..ai_cashflow_forecasting import APGCashFlowForecastingService


class PerformanceMetrics:
	"""Track performance metrics during load testing."""
	
	def __init__(self):
		self.response_times: List[float] = []
		self.error_count = 0
		self.success_count = 0
		self.start_time = None
		self.end_time = None
		self.lock = threading.Lock()
	
	def record_response(self, response_time: float, success: bool = True):
		"""Record a response time and success/failure."""
		with self.lock:
			self.response_times.append(response_time)
			if success:
				self.success_count += 1
			else:
				self.error_count += 1
	
	def start_timer(self):
		"""Start timing the test."""
		self.start_time = time.time()
	
	def stop_timer(self):
		"""Stop timing the test."""
		self.end_time = time.time()
	
	@property
	def total_requests(self) -> int:
		"""Total number of requests processed."""
		return self.success_count + self.error_count
	
	@property
	def success_rate(self) -> float:
		"""Success rate percentage."""
		if self.total_requests == 0:
			return 0.0
		return (self.success_count / self.total_requests) * 100
	
	@property
	def requests_per_second(self) -> float:
		"""Requests per second throughput."""
		if not self.start_time or not self.end_time:
			return 0.0
		duration = self.end_time - self.start_time
		return self.total_requests / duration if duration > 0 else 0.0
	
	@property
	def avg_response_time(self) -> float:
		"""Average response time in milliseconds."""
		if not self.response_times:
			return 0.0
		return statistics.mean(self.response_times) * 1000
	
	@property
	def p95_response_time(self) -> float:
		"""95th percentile response time in milliseconds."""
		if not self.response_times:
			return 0.0
		sorted_times = sorted(self.response_times)
		index = int(0.95 * len(sorted_times))
		return sorted_times[index] * 1000
	
	@property
	def p99_response_time(self) -> float:
		"""99th percentile response time in milliseconds."""
		if not self.response_times:
			return 0.0
		sorted_times = sorted(self.response_times)
		index = int(0.99 * len(sorted_times))
		return sorted_times[index] * 1000


@pytest.mark.performance
class TestCustomerServicePerformance:
	"""Performance tests for customer service operations."""
	
	@pytest.fixture
	def customer_service(self):
		"""Create customer service for testing."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		return ARCustomerService(tenant_id, user_id)
	
	async def test_concurrent_customer_creation(self, customer_service):
		"""Test concurrent customer creation performance."""
		metrics = PerformanceMetrics()
		concurrent_requests = 50
		
		async def create_customer_task(index: int):
			"""Create a single customer and measure performance."""
			start_time = time.time()
			try:
				customer_data = {
					'customer_code': f'PERF{index:05d}',
					'legal_name': f'Performance Test Customer {index}',
					'customer_type': ARCustomerType.CORPORATION,
					'status': ARCustomerStatus.ACTIVE,
					'credit_limit': Decimal('50000.00'),
					'payment_terms_days': 30
				}
				
				# Mock service dependencies
				with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock):
					with patch('uuid_extensions.uuid7str', return_value=f'customer-{index}'):
						customer = await customer_service.create_customer(customer_data)
				
				response_time = time.time() - start_time
				metrics.record_response(response_time, True)
				return customer
				
			except Exception as e:
				response_time = time.time() - start_time
				metrics.record_response(response_time, False)
				raise e
		
		# Execute concurrent requests
		metrics.start_timer()
		
		tasks = [create_customer_task(i) for i in range(concurrent_requests)]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		metrics.stop_timer()
		
		# Validate performance requirements
		successful_results = [r for r in results if not isinstance(r, Exception)]
		
		assert len(successful_results) == concurrent_requests, f"Expected {concurrent_requests} successful creations"
		assert metrics.success_rate >= 99.0, f"Success rate {metrics.success_rate:.1f}% below 99%"
		assert metrics.avg_response_time <= 200, f"Average response time {metrics.avg_response_time:.1f}ms exceeds 200ms"
		assert metrics.p95_response_time <= 500, f"P95 response time {metrics.p95_response_time:.1f}ms exceeds 500ms"
		assert metrics.requests_per_second >= 100, f"Throughput {metrics.requests_per_second:.1f} RPS below 100"
		
		print(f"Customer Creation Performance:")
		print(f"  Success Rate: {metrics.success_rate:.1f}%")
		print(f"  Avg Response Time: {metrics.avg_response_time:.1f}ms")
		print(f"  P95 Response Time: {metrics.p95_response_time:.1f}ms")
		print(f"  P99 Response Time: {metrics.p99_response_time:.1f}ms")
		print(f"  Throughput: {metrics.requests_per_second:.1f} RPS")
	
	async def test_customer_query_performance(self, customer_service):
		"""Test customer query performance under load."""
		metrics = PerformanceMetrics()
		concurrent_requests = 100
		
		# Mock existing customers
		mock_customers = []
		for i in range(20):
			customer = ARCustomer(
				id=f'query-customer-{i}',
				tenant_id=customer_service.tenant_id,
				customer_code=f'QUERY{i:03d}',
				legal_name=f'Query Test Customer {i}',
				customer_type=ARCustomerType.CORPORATION,
				status=ARCustomerStatus.ACTIVE,
				credit_limit=Decimal('75000.00'),
				payment_terms_days=30,
				created_by=customer_service.user_id,
				updated_by=customer_service.user_id
			)
			mock_customers.append(customer)
		
		async def query_customers_task():
			"""Query customers and measure performance."""
			start_time = time.time()
			try:
				# Mock service dependencies
				with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock):
					with patch.object(customer_service, '_get_customers_filtered_from_db', new_callable=AsyncMock) as mock_query:
						mock_query.return_value = mock_customers[:10]  # Return subset
						
						customers = await customer_service.get_customers_filtered(
							customer_type=ARCustomerType.CORPORATION,
							status=ARCustomerStatus.ACTIVE,
							page=1,
							per_page=10
						)
				
				response_time = time.time() - start_time
				metrics.record_response(response_time, True)
				return customers
				
			except Exception as e:
				response_time = time.time() - start_time
				metrics.record_response(response_time, False)
				raise e
		
		# Execute concurrent queries
		metrics.start_timer()
		
		tasks = [query_customers_task() for _ in range(concurrent_requests)]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		metrics.stop_timer()
		
		# Validate performance requirements
		successful_results = [r for r in results if not isinstance(r, Exception)]
		
		assert len(successful_results) == concurrent_requests
		assert metrics.success_rate >= 99.5, f"Query success rate {metrics.success_rate:.1f}% below 99.5%"
		assert metrics.avg_response_time <= 100, f"Query avg response time {metrics.avg_response_time:.1f}ms exceeds 100ms"
		assert metrics.p95_response_time <= 200, f"Query P95 response time {metrics.p95_response_time:.1f}ms exceeds 200ms"
		assert metrics.requests_per_second >= 200, f"Query throughput {metrics.requests_per_second:.1f} RPS below 200"
		
		print(f"Customer Query Performance:")
		print(f"  Success Rate: {metrics.success_rate:.1f}%")
		print(f"  Avg Response Time: {metrics.avg_response_time:.1f}ms")
		print(f"  P95 Response Time: {metrics.p95_response_time:.1f}ms")
		print(f"  Throughput: {metrics.requests_per_second:.1f} RPS")


@pytest.mark.performance
class TestInvoiceServicePerformance:
	"""Performance tests for invoice service operations."""
	
	@pytest.fixture
	def invoice_service(self):
		"""Create invoice service for testing."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		return ARInvoiceService(tenant_id, user_id)
	
	async def test_bulk_invoice_creation_performance(self, invoice_service):
		"""Test bulk invoice creation performance."""
		metrics = PerformanceMetrics()
		batch_size = 1000
		
		# Create customer IDs for invoices
		customer_ids = [uuid7str() for _ in range(100)]
		
		async def create_invoice_batch():
			"""Create a batch of invoices."""
			start_time = time.time()
			try:
				invoices = []
				for i in range(batch_size):
					customer_id = customer_ids[i % len(customer_ids)]
					invoice_data = {
						'customer_id': customer_id,
						'invoice_number': f'BULK-{i:06d}',
						'invoice_date': date.today(),
						'due_date': date.today() + timedelta(days=30),
						'total_amount': Decimal('5000.00'),
						'description': f'Bulk performance test invoice {i}'
					}
					
					# Mock service dependencies
					with patch.object(invoice_service, '_validate_permissions', new_callable=AsyncMock):
						with patch.object(invoice_service, '_validate_customer_exists', new_callable=AsyncMock):
							with patch('uuid_extensions.uuid7str', return_value=f'invoice-{i}'):
								invoice = await invoice_service.create_invoice(invoice_data)
								invoices.append(invoice)
				
				response_time = time.time() - start_time
				metrics.record_response(response_time, True)
				return invoices
				
			except Exception as e:
				response_time = time.time() - start_time
				metrics.record_response(response_time, False)
				raise e
		
		# Execute batch creation
		metrics.start_timer()
		
		result = await create_invoice_batch()
		
		metrics.stop_timer()
		
		# Validate performance requirements
		assert len(result) == batch_size
		assert metrics.success_rate == 100.0
		
		# Performance targets: Process 1000 invoices in under 5 seconds
		total_time = metrics.end_time - metrics.start_time
		invoices_per_second = batch_size / total_time
		
		assert total_time <= 5.0, f"Bulk creation took {total_time:.2f}s, exceeds 5s limit"
		assert invoices_per_second >= 200, f"Bulk throughput {invoices_per_second:.1f} invoices/sec below 200"
		
		print(f"Bulk Invoice Creation Performance:")
		print(f"  Total Time: {total_time:.2f}s")
		print(f"  Invoices/Second: {invoices_per_second:.1f}")
		print(f"  Avg per Invoice: {(total_time * 1000 / batch_size):.2f}ms")


@pytest.mark.performance
class TestAIServicePerformance:
	"""Performance tests for AI-powered services."""
	
	@pytest.fixture
	def credit_ai_service(self):
		"""Create credit AI service for testing."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		return APGCreditScoringService(tenant_id, user_id)
	
	@pytest.fixture
	def collections_ai_service(self):
		"""Create collections AI service for testing."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		return APGCollectionsAIService(tenant_id, user_id)
	
	async def test_batch_credit_assessment_performance(self, credit_ai_service):
		"""Test batch credit assessment performance."""
		metrics = PerformanceMetrics()
		batch_size = 100
		
		# Create customer IDs for assessment
		customer_ids = [uuid7str() for _ in range(batch_size)]
		
		# Mock assessment results
		mock_results = []
		for i, customer_id in enumerate(customer_ids):
			result = CreditScoringResult(
				customer_id=customer_id,
				assessment_date=date.today(),
				credit_score=600 + (i % 200),  # Varying scores
				risk_level='MEDIUM',
				confidence_score=0.80 + (i % 20) * 0.01,
				feature_importance={'payment_history': 0.4},
				explanations=[f'AI assessment for customer {i}']
			)
			mock_results.append(result)
		
		async def batch_assessment_task():
			"""Perform batch credit assessment."""
			start_time = time.time()
			try:
				# Mock individual assessments
				with patch.object(credit_ai_service, 'assess_customer_credit', new_callable=AsyncMock) as mock_assess:
					mock_assess.side_effect = mock_results
					
					with patch.object(credit_ai_service, '_get_customer_from_db', new_callable=AsyncMock) as mock_get:
						# Mock customer objects
						mock_customers = []
						for customer_id in customer_ids:
							customer = Mock()
							customer.id = customer_id
							mock_customers.append(customer)
						mock_get.side_effect = mock_customers
						
						results = await credit_ai_service.batch_assess_customers_credit(customer_ids)
				
				response_time = time.time() - start_time
				metrics.record_response(response_time, True)
				return results
				
			except Exception as e:
				response_time = time.time() - start_time
				metrics.record_response(response_time, False)
				raise e
		
		# Execute batch assessment
		metrics.start_timer()
		
		results = await batch_assessment_task()
		
		metrics.stop_timer()
		
		# Validate performance requirements
		assert len(results) == batch_size
		assert metrics.success_rate == 100.0
		
		# Performance targets: Process 100 assessments in under 10 seconds
		total_time = metrics.end_time - metrics.start_time
		assessments_per_second = batch_size / total_time
		
		assert total_time <= 10.0, f"Batch assessment took {total_time:.2f}s, exceeds 10s limit"
		assert assessments_per_second >= 10, f"Assessment throughput {assessments_per_second:.1f}/sec below 10"
		
		print(f"Batch Credit Assessment Performance:")
		print(f"  Total Time: {total_time:.2f}s")
		print(f"  Assessments/Second: {assessments_per_second:.1f}")
		print(f"  Avg per Assessment: {(total_time * 1000 / batch_size):.2f}ms")
	
	async def test_concurrent_ai_operations(self, credit_ai_service, collections_ai_service):
		"""Test concurrent AI operations performance."""
		metrics = PerformanceMetrics()
		concurrent_requests = 20
		
		async def mixed_ai_operations_task(index: int):
			"""Perform mixed AI operations."""
			start_time = time.time()
			try:
				customer_id = uuid7str()
				
				# Mock customer
				mock_customer = Mock()
				mock_customer.id = customer_id
				
				# Mock credit assessment
				mock_credit_result = CreditScoringResult(
					customer_id=customer_id,
					assessment_date=date.today(),
					credit_score=700 + (index % 100),
					risk_level='MEDIUM',
					confidence_score=0.85,
					feature_importance={'payment_history': 0.4},
					explanations=['Concurrent test assessment']
				)
				
				# Mock collections optimization
				from ..ai_collections_optimization import CustomerCollectionProfile, CollectionStrategyRecommendation, CollectionStrategyType, CollectionChannelType
				
				collection_profile = CustomerCollectionProfile(
					customer_id=customer_id,
					overdue_amount=Decimal('5000.00'),
					days_overdue=10,
					payment_history_score=0.75,
					customer_segment='CORPORATION'
				)
				
				mock_strategy = CollectionStrategyRecommendation(
					customer_id=customer_id,
					recommended_strategy=CollectionStrategyType.EMAIL_REMINDER,
					contact_method=CollectionChannelType.EMAIL,
					success_probability=0.70,
					estimated_resolution_days=8,
					priority_level='MEDIUM'
				)
				
				# Execute both AI operations concurrently
				with patch.object(credit_ai_service, 'assess_customer_credit', new_callable=AsyncMock) as mock_credit:
					mock_credit.return_value = mock_credit_result
					
					with patch.object(collections_ai_service, 'optimize_collection_strategy', new_callable=AsyncMock) as mock_collections:
						mock_collections.return_value = mock_strategy
						
						# Run both operations
						credit_task = credit_ai_service.assess_customer_credit(mock_customer)
						collections_task = collections_ai_service.optimize_collection_strategy(collection_profile)
						
						credit_result, strategy_result = await asyncio.gather(credit_task, collections_task)
				
				response_time = time.time() - start_time
				metrics.record_response(response_time, True)
				return credit_result, strategy_result
				
			except Exception as e:
				response_time = time.time() - start_time
				metrics.record_response(response_time, False)
				raise e
		
		# Execute concurrent AI operations
		metrics.start_timer()
		
		tasks = [mixed_ai_operations_task(i) for i in range(concurrent_requests)]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		metrics.stop_timer()
		
		# Validate performance requirements
		successful_results = [r for r in results if not isinstance(r, Exception)]
		
		assert len(successful_results) == concurrent_requests
		assert metrics.success_rate >= 95.0, f"AI operations success rate {metrics.success_rate:.1f}% below 95%"
		assert metrics.avg_response_time <= 1000, f"AI avg response time {metrics.avg_response_time:.1f}ms exceeds 1000ms"
		assert metrics.requests_per_second >= 10, f"AI throughput {metrics.requests_per_second:.1f} RPS below 10"
		
		print(f"Concurrent AI Operations Performance:")
		print(f"  Success Rate: {metrics.success_rate:.1f}%")
		print(f"  Avg Response Time: {metrics.avg_response_time:.1f}ms")
		print(f"  P95 Response Time: {metrics.p95_response_time:.1f}ms")
		print(f"  Throughput: {metrics.requests_per_second:.1f} RPS")


@pytest.mark.performance
class TestEndToEndPerformance:
	"""End-to-end performance tests for complete workflows."""
	
	async def test_complete_ar_workflow_performance(self):
		"""Test complete AR workflow performance under load."""
		metrics = PerformanceMetrics()
		concurrent_workflows = 10
		
		async def complete_workflow_task(workflow_id: int):
			"""Execute complete AR workflow."""
			start_time = time.time()
			try:
				tenant_id = uuid7str()
				user_id = uuid7str()
				
				# Initialize services
				customer_service = ARCustomerService(tenant_id, user_id)
				invoice_service = ARInvoiceService(tenant_id, user_id)
				payment_service = ARCashApplicationService(tenant_id, user_id)
				credit_ai_service = APGCreditScoringService(tenant_id, user_id)
				
				# Mock all service dependencies
				with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock), \
					 patch.object(invoice_service, '_validate_permissions', new_callable=AsyncMock), \
					 patch.object(payment_service, '_validate_permissions', new_callable=AsyncMock):
					
					# 1. Create Customer
					customer_data = {
						'customer_code': f'WF{workflow_id:05d}',
						'legal_name': f'Workflow Customer {workflow_id}',
						'customer_type': ARCustomerType.CORPORATION,
						'status': ARCustomerStatus.ACTIVE,
						'credit_limit': Decimal('100000.00'),
						'payment_terms_days': 30
					}
					
					with patch('uuid_extensions.uuid7str', side_effect=[
						f'customer-{workflow_id}', f'invoice-{workflow_id}', f'payment-{workflow_id}'
					]):
						customer = await customer_service.create_customer(customer_data)
						
						# 2. AI Credit Assessment
						mock_credit_result = CreditScoringResult(
							customer_id=customer.id,
							assessment_date=date.today(),
							credit_score=720,
							risk_level='MEDIUM',
							confidence_score=0.85,
							feature_importance={'payment_history': 0.4},
							explanations=['Workflow assessment']
						)
						
						with patch.object(credit_ai_service, 'assess_customer_credit', new_callable=AsyncMock) as mock_credit:
							mock_credit.return_value = mock_credit_result
							credit_result = await credit_ai_service.assess_customer_credit(customer)
						
						# 3. Create Invoice
						invoice_data = {
							'customer_id': customer.id,
							'invoice_number': f'WF-INV-{workflow_id:05d}',
							'invoice_date': date.today(),
							'due_date': date.today() + timedelta(days=30),
							'total_amount': Decimal('25000.00'),
							'description': f'Workflow invoice {workflow_id}'
						}
						
						with patch.object(invoice_service, '_validate_customer_exists', new_callable=AsyncMock):
							invoice = await invoice_service.create_invoice(invoice_data)
						
						# 4. Process Payment
						payment_data = {
							'customer_id': customer.id,
							'payment_reference': f'WF-PAY-{workflow_id:05d}',
							'payment_date': date.today(),
							'payment_amount': Decimal('15000.00'),
							'payment_method': 'WIRE_TRANSFER'
						}
						
						with patch.object(payment_service, '_validate_customer_exists', new_callable=AsyncMock):
							payment = await payment_service.create_payment(payment_data)
				
				response_time = time.time() - start_time
				metrics.record_response(response_time, True)
				return {
					'customer': customer,
					'credit_result': credit_result,
					'invoice': invoice,
					'payment': payment
				}
				
			except Exception as e:
				response_time = time.time() - start_time
				metrics.record_response(response_time, False)
				raise e
		
		# Execute concurrent workflows
		metrics.start_timer()
		
		tasks = [complete_workflow_task(i) for i in range(concurrent_workflows)]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		metrics.stop_timer()
		
		# Validate performance requirements
		successful_results = [r for r in results if not isinstance(r, Exception)]
		
		assert len(successful_results) == concurrent_workflows
		assert metrics.success_rate >= 95.0, f"Workflow success rate {metrics.success_rate:.1f}% below 95%"
		assert metrics.avg_response_time <= 2000, f"Workflow avg time {metrics.avg_response_time:.1f}ms exceeds 2000ms"
		assert metrics.requests_per_second >= 5, f"Workflow throughput {metrics.requests_per_second:.1f} RPS below 5"
		
		print(f"Complete AR Workflow Performance:")
		print(f"  Success Rate: {metrics.success_rate:.1f}%")
		print(f"  Avg Workflow Time: {metrics.avg_response_time:.1f}ms")
		print(f"  P95 Workflow Time: {metrics.p95_response_time:.1f}ms")
		print(f"  Workflow Throughput: {metrics.requests_per_second:.1f} RPS")


# Performance test runner
if __name__ == "__main__":
	# Run performance tests with detailed output
	pytest.main([__file__, "-v", "-s", "-m", "performance"])