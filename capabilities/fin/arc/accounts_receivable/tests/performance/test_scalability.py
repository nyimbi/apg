"""
APG Accounts Receivable - Scalability Tests
Scalability validation for increasing load patterns

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
from typing import List, Dict, Any, Tuple
import math

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


class ScalabilityMetrics:
	"""Track scalability metrics across different load levels."""
	
	def __init__(self):
		self.load_tests: List[Dict[str, Any]] = []
	
	def record_load_test(self, load_level: int, response_times: List[float], success_count: int, error_count: int, duration: float):
		"""Record results from a load test."""
		total_requests = success_count + error_count
		
		self.load_tests.append({
			'load_level': load_level,
			'total_requests': total_requests,
			'success_count': success_count,
			'error_count': error_count,
			'success_rate': (success_count / total_requests * 100) if total_requests > 0 else 0,
			'duration': duration,
			'throughput': total_requests / duration if duration > 0 else 0,
			'avg_response_time': statistics.mean(response_times) * 1000 if response_times else 0,
			'p95_response_time': self._percentile(response_times, 0.95) * 1000 if response_times else 0,
			'p99_response_time': self._percentile(response_times, 0.99) * 1000 if response_times else 0
		})
	
	def _percentile(self, data: List[float], percentile: float) -> float:
		"""Calculate percentile from sorted data."""
		if not data:
			return 0.0
		sorted_data = sorted(data)
		index = int(percentile * len(sorted_data))
		return sorted_data[index] if index < len(sorted_data) else sorted_data[-1]
	
	def analyze_scalability(self) -> Dict[str, Any]:
		"""Analyze scalability characteristics."""
		if len(self.load_tests) < 2:
			return {'error': 'Need at least 2 load tests for scalability analysis'}
		
		# Calculate scalability factors
		baseline = self.load_tests[0]
		max_load = self.load_tests[-1]
		
		load_multiplier = max_load['load_level'] / baseline['load_level']
		throughput_multiplier = max_load['throughput'] / baseline['throughput'] if baseline['throughput'] > 0 else 0
		response_time_multiplier = max_load['avg_response_time'] / baseline['avg_response_time'] if baseline['avg_response_time'] > 0 else 0
		
		# Linear scalability would have throughput_multiplier == load_multiplier
		scalability_efficiency = throughput_multiplier / load_multiplier if load_multiplier > 0 else 0
		
		# Performance degradation (higher is worse)
		performance_degradation = response_time_multiplier / load_multiplier if load_multiplier > 0 else 0
		
		return {
			'load_multiplier': load_multiplier,
			'throughput_multiplier': throughput_multiplier,
			'response_time_multiplier': response_time_multiplier,
			'scalability_efficiency': scalability_efficiency,
			'performance_degradation': performance_degradation,
			'load_tests': self.load_tests
		}


@pytest.mark.performance
class TestLinearScalability:
	"""Test linear scalability characteristics."""
	
	async def test_customer_service_scalability(self):
		"""Test customer service scalability across increasing load."""
		scalability = ScalabilityMetrics()
		
		# Test with increasing load levels
		load_levels = [10, 25, 50, 100, 200]
		
		for load_level in load_levels:
			tenant_id = uuid7str()
			user_id = uuid7str()
			customer_service = ARCustomerService(tenant_id, user_id)
			
			response_times = []
			success_count = 0
			error_count = 0
			
			async def create_customer_task(index: int):
				"""Create customer and track performance."""
				start_time = time.time()
				try:
					customer_data = {
						'customer_code': f'SCALE{load_level:03d}{index:05d}',
						'legal_name': f'Scalability Test Customer {load_level}-{index}',
						'customer_type': ARCustomerType.CORPORATION,
						'status': ARCustomerStatus.ACTIVE,
						'credit_limit': Decimal('75000.00'),
						'payment_terms_days': 30
					}
					
					with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock):
						with patch('uuid_extensions.uuid7str', return_value=f'scale-customer-{load_level}-{index}'):
							customer = await customer_service.create_customer(customer_data)
					
					response_time = time.time() - start_time
					response_times.append(response_time)
					return True
					
				except Exception:
					response_time = time.time() - start_time
					response_times.append(response_time)
					return False
			
			# Execute load test
			start_time = time.time()
			
			tasks = [create_customer_task(i) for i in range(load_level)]
			results = await asyncio.gather(*tasks, return_exceptions=True)
			
			duration = time.time() - start_time
			
			# Count successes and errors
			for result in results:
				if result is True:
					success_count += 1
				else:
					error_count += 1
			
			# Record load test results
			scalability.record_load_test(load_level, response_times, success_count, error_count, duration)
			
			print(f"Load Level {load_level:3d}: {success_count:3d} success, {error_count:2d} errors, "
				  f"{duration:.2f}s, {load_level/duration:.1f} RPS")
		
		# Analyze scalability
		analysis = scalability.analyze_scalability()
		
		# Validate scalability requirements
		assert analysis['scalability_efficiency'] >= 0.7, f"Scalability efficiency {analysis['scalability_efficiency']:.2f} below 70%"
		assert analysis['performance_degradation'] <= 2.0, f"Performance degradation {analysis['performance_degradation']:.2f} exceeds 2x"
		
		# Success rate should remain high across all load levels
		for test in analysis['load_tests']:
			assert test['success_rate'] >= 95.0, f"Success rate {test['success_rate']:.1f}% at load {test['load_level']} below 95%"
		
		print(f"Customer Service Scalability Analysis:")
		print(f"  Load Multiplier: {analysis['load_multiplier']:.1f}x")
		print(f"  Throughput Multiplier: {analysis['throughput_multiplier']:.1f}x")
		print(f"  Scalability Efficiency: {analysis['scalability_efficiency']:.1%}")
		print(f"  Performance Degradation: {analysis['performance_degradation']:.2f}x")
	
	async def test_mixed_operations_scalability(self):
		"""Test scalability with mixed operation types."""
		scalability = ScalabilityMetrics()
		
		# Test with increasing load levels for mixed operations
		load_levels = [20, 50, 100, 150]
		
		for load_level in load_levels:
			tenant_id = uuid7str()
			user_id = uuid7str()
			
			# Initialize multiple services
			customer_service = ARCustomerService(tenant_id, user_id)
			invoice_service = ARInvoiceService(tenant_id, user_id)
			payment_service = ARCashApplicationService(tenant_id, user_id)
			
			response_times = []
			success_count = 0
			error_count = 0
			
			async def mixed_operations_task(index: int):
				"""Perform mixed AR operations."""
				start_time = time.time()
				try:
					operation_type = index % 3  # Rotate between operation types
					
					# Mock service dependencies
					with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock), \
						 patch.object(invoice_service, '_validate_permissions', new_callable=AsyncMock), \
						 patch.object(payment_service, '_validate_permissions', new_callable=AsyncMock):
						
						if operation_type == 0:
							# Customer creation (33% of operations)
							customer_data = {
								'customer_code': f'MIX{load_level:03d}{index:05d}',
								'legal_name': f'Mixed Ops Customer {load_level}-{index}',
								'customer_type': ARCustomerType.INDIVIDUAL,
								'status': ARCustomerStatus.ACTIVE,
								'credit_limit': Decimal('50000.00'),
								'payment_terms_days': 30
							}
							
							with patch('uuid_extensions.uuid7str', return_value=f'mix-customer-{load_level}-{index}'):
								await customer_service.create_customer(customer_data)
						
						elif operation_type == 1:
							# Invoice creation (33% of operations)
							invoice_data = {
								'customer_id': uuid7str(),
								'invoice_number': f'MIX-INV-{load_level:03d}{index:05d}',
								'invoice_date': date.today(),
								'due_date': date.today() + timedelta(days=30),
								'total_amount': Decimal('12000.00'),
								'description': f'Mixed ops invoice {load_level}-{index}'
							}
							
							with patch.object(invoice_service, '_validate_customer_exists', new_callable=AsyncMock):
								with patch('uuid_extensions.uuid7str', return_value=f'mix-invoice-{load_level}-{index}'):
									await invoice_service.create_invoice(invoice_data)
						
						else:
							# Payment creation (33% of operations)
							payment_data = {
								'customer_id': uuid7str(),
								'payment_reference': f'MIX-PAY-{load_level:03d}{index:05d}',
								'payment_date': date.today(),
								'payment_amount': Decimal('8000.00'),
								'payment_method': 'CREDIT_CARD'
							}
							
							with patch.object(payment_service, '_validate_customer_exists', new_callable=AsyncMock):
								with patch('uuid_extensions.uuid7str', return_value=f'mix-payment-{load_level}-{index}'):
									await payment_service.create_payment(payment_data)
					
					response_time = time.time() - start_time
					response_times.append(response_time)
					return True
					
				except Exception:
					response_time = time.time() - start_time
					response_times.append(response_time)
					return False
			
			# Execute mixed operations load test
			start_time = time.time()
			
			tasks = [mixed_operations_task(i) for i in range(load_level)]
			results = await asyncio.gather(*tasks, return_exceptions=True)
			
			duration = time.time() - start_time
			
			# Count successes and errors
			for result in results:
				if result is True:
					success_count += 1
				else:
					error_count += 1
			
			# Record load test results
			scalability.record_load_test(load_level, response_times, success_count, error_count, duration)
			
			print(f"Mixed Ops Load {load_level:3d}: {success_count:3d} success, {error_count:2d} errors, "
				  f"{duration:.2f}s, {load_level/duration:.1f} RPS")
		
		# Analyze scalability
		analysis = scalability.analyze_scalability()
		
		# Validate mixed operations scalability
		assert analysis['scalability_efficiency'] >= 0.6, f"Mixed ops scalability efficiency {analysis['scalability_efficiency']:.2f} below 60%"
		assert analysis['performance_degradation'] <= 3.0, f"Mixed ops performance degradation {analysis['performance_degradation']:.2f} exceeds 3x"
		
		print(f"Mixed Operations Scalability Analysis:")
		print(f"  Scalability Efficiency: {analysis['scalability_efficiency']:.1%}")
		print(f"  Performance Degradation: {analysis['performance_degradation']:.2f}x")


@pytest.mark.performance
class TestAIScalability:
	"""Test AI service scalability under increasing load."""
	
	async def test_ai_credit_scoring_scalability(self):
		"""Test AI credit scoring scalability."""
		scalability = ScalabilityMetrics()
		
		# Test AI operations with increasing batch sizes
		batch_sizes = [5, 15, 30, 50]
		
		for batch_size in batch_sizes:
			tenant_id = uuid7str()
			user_id = uuid7str()
			credit_ai_service = APGCreditScoringService(tenant_id, user_id)
			
			response_times = []
			success_count = 0
			error_count = 0
			
			async def ai_assessment_task():
				"""Perform AI credit assessment batch."""
				start_time = time.time()
				try:
					# Create customer IDs for batch assessment
					customer_ids = [uuid7str() for _ in range(batch_size)]
					
					# Mock assessment results
					mock_results = []
					for i, customer_id in enumerate(customer_ids):
						result = CreditScoringResult(
							customer_id=customer_id,
							assessment_date=date.today(),
							credit_score=650 + (i % 150),
							risk_level='MEDIUM',
							confidence_score=0.80 + (i % 20) * 0.01,
							feature_importance={'payment_history': 0.4},
							explanations=[f'AI scalability assessment {i}']
						)
						mock_results.append(result)
					
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
					response_times.append(response_time)
					return len(results) == batch_size
					
				except Exception:
					response_time = time.time() - start_time
					response_times.append(response_time)
					return False
			
			# Execute AI scalability test
			start_time = time.time()
			
			# Run single batch assessment (testing batch size scalability)
			result = await ai_assessment_task()
			
			duration = time.time() - start_time
			
			if result:
				success_count = 1
			else:
				error_count = 1
			
			# Record results (treating each batch as one "request")
			scalability.record_load_test(batch_size, response_times, success_count, error_count, duration)
			
			assessments_per_second = batch_size / duration if duration > 0 else 0
			
			print(f"AI Batch Size {batch_size:2d}: {success_count} success, {error_count} errors, "
				  f"{duration:.2f}s, {assessments_per_second:.1f} assessments/sec")
		
		# Analyze AI scalability
		analysis = scalability.analyze_scalability()
		
		# AI operations may not scale linearly due to computational complexity
		assert analysis['scalability_efficiency'] >= 0.5, f"AI scalability efficiency {analysis['scalability_efficiency']:.2f} below 50%"
		assert analysis['performance_degradation'] <= 4.0, f"AI performance degradation {analysis['performance_degradation']:.2f} exceeds 4x"
		
		# Response times should remain reasonable even at large batch sizes
		for test in analysis['load_tests']:
			assert test['avg_response_time'] <= 5000, f"AI avg response time {test['avg_response_time']:.0f}ms at batch {test['load_level']} exceeds 5s"
		
		print(f"AI Credit Scoring Scalability Analysis:")
		print(f"  Scalability Efficiency: {analysis['scalability_efficiency']:.1%}")
		print(f"  Performance Degradation: {analysis['performance_degradation']:.2f}x")


@pytest.mark.performance
class TestDataVolumeScalability:
	"""Test scalability with increasing data volumes."""
	
	async def test_query_performance_with_large_datasets(self):
		"""Test query performance as dataset size increases."""
		scalability = ScalabilityMetrics()
		
		# Test with increasing dataset sizes
		dataset_sizes = [1000, 5000, 10000, 20000]
		
		for dataset_size in dataset_sizes:
			tenant_id = uuid7str()
			user_id = uuid7str()
			customer_service = ARCustomerService(tenant_id, user_id)
			
			# Create mock dataset
			mock_customers = []
			for i in range(dataset_size):
				customer = ARCustomer(
					id=f'dataset-customer-{i}',
					tenant_id=tenant_id,
					customer_code=f'DS{i:06d}',
					legal_name=f'Dataset Customer {i}',
					customer_type=ARCustomerType.CORPORATION if i % 2 == 0 else ARCustomerType.INDIVIDUAL,
					status=ARCustomerStatus.ACTIVE if i % 3 != 0 else ARCustomerStatus.INACTIVE,
					credit_limit=Decimal('50000.00') + (i % 100000),
					total_outstanding=Decimal('10000.00') + (i % 50000),
					payment_terms_days=30,
					created_by=user_id,
					updated_by=user_id
				)
				mock_customers.append(customer)
			
			response_times = []
			success_count = 0
			error_count = 0
			
			async def query_large_dataset_task():
				"""Query operations on large dataset."""
				start_time = time.time()
				try:
					# Mock service dependencies
					with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock):
						with patch.object(customer_service, '_get_customers_filtered_from_db', new_callable=AsyncMock) as mock_query:
							# Simulate filtering large dataset
							filtered_customers = [
								c for c in mock_customers 
								if c.customer_type == ARCustomerType.CORPORATION and 
								   c.status == ARCustomerStatus.ACTIVE and
								   c.total_outstanding > Decimal('25000.00')
							]
							mock_query.return_value = filtered_customers[:100]  # Return paginated results
							
							results = await customer_service.get_customers_filtered(
								customer_type=ARCustomerType.CORPORATION,
								status=ARCustomerStatus.ACTIVE,
								min_outstanding=Decimal('25000.00'),
								page=1,
								per_page=100
							)
					
					response_time = time.time() - start_time
					response_times.append(response_time)
					return len(results) > 0
					
				except Exception:
					response_time = time.time() - start_time
					response_times.append(response_time)
					return False
			
			# Execute query on large dataset
			start_time = time.time()
			
			# Run multiple concurrent queries to test scalability
			concurrent_queries = 10
			tasks = [query_large_dataset_task() for _ in range(concurrent_queries)]
			results = await asyncio.gather(*tasks, return_exceptions=True)
			
			duration = time.time() - start_time
			
			# Count successes and errors
			for result in results:
				if result is True:
					success_count += 1
				else:
					error_count += 1
			
			# Record results (using dataset size as "load level")
			scalability.record_load_test(dataset_size, response_times, success_count, error_count, duration)
			
			queries_per_second = concurrent_queries / duration if duration > 0 else 0
			
			print(f"Dataset Size {dataset_size:5d}: {success_count:2d} success, {error_count} errors, "
				  f"{duration:.2f}s, {queries_per_second:.1f} queries/sec")
		
		# Analyze data volume scalability
		analysis = scalability.analyze_scalability()
		
		# Query performance should degrade gracefully with dataset size
		assert analysis['performance_degradation'] <= 3.0, f"Query performance degradation {analysis['performance_degradation']:.2f} exceeds 3x"
		
		# All queries should succeed regardless of dataset size
		for test in analysis['load_tests']:
			assert test['success_rate'] == 100.0, f"Query success rate {test['success_rate']:.1f}% at dataset {test['load_level']} below 100%"
			assert test['avg_response_time'] <= 1000, f"Query avg response time {test['avg_response_time']:.0f}ms at dataset {test['load_level']} exceeds 1s"
		
		print(f"Data Volume Scalability Analysis:")
		print(f"  Performance Degradation: {analysis['performance_degradation']:.2f}x")
		print(f"  Largest Dataset Query Time: {analysis['load_tests'][-1]['avg_response_time']:.0f}ms")


# Scalability test runner
if __name__ == "__main__":
	pytest.main([__file__, "-v", "-s", "-m", "performance"])