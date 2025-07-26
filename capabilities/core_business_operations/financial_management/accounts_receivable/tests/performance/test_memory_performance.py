"""
APG Accounts Receivable - Memory Performance Tests
Memory usage and optimization validation for AR capability

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
import gc
import psutil
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any
import tracemalloc

from uuid_extensions import uuid7str

from ..models import (
	ARCustomer, ARInvoice, ARPayment, ARCollectionActivity,
	ARCustomerType, ARCustomerStatus, ARInvoiceStatus, ARPaymentStatus
)
from ..service import (
	ARCustomerService, ARInvoiceService, ARCollectionsService,
	ARCashApplicationService, ARAnalyticsService
)


class MemoryProfiler:
	"""Profile memory usage during testing."""
	
	def __init__(self):
		self.start_memory = None
		self.peak_memory = None
		self.end_memory = None
		self.memory_snapshots = []
		self.tracemalloc_started = False
	
	def start_profiling(self):
		"""Start memory profiling."""
		# Force garbage collection before starting
		gc.collect()
		
		# Start tracemalloc for detailed tracking
		tracemalloc.start()
		self.tracemalloc_started = True
		
		# Record initial memory usage
		process = psutil.Process(os.getpid())
		self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
		self.peak_memory = self.start_memory
	
	def record_snapshot(self, label: str = ""):
		"""Record a memory snapshot."""
		if not self.tracemalloc_started:
			return
		
		process = psutil.Process(os.getpid())
		current_memory = process.memory_info().rss / 1024 / 1024  # MB
		
		snapshot = tracemalloc.take_snapshot()
		top_stats = snapshot.statistics('lineno')
		
		# Update peak memory
		if current_memory > self.peak_memory:
			self.peak_memory = current_memory
		
		self.memory_snapshots.append({
			'label': label,
			'memory_mb': current_memory,
			'top_allocations': top_stats[:5]  # Top 5 memory allocations
		})
	
	def stop_profiling(self):
		"""Stop memory profiling and return results."""
		if not self.tracemalloc_started:
			return {}
		
		# Record final memory usage
		process = psutil.Process(os.getpid())
		self.end_memory = process.memory_info().rss / 1024 / 1024  # MB
		
		# Stop tracemalloc
		tracemalloc.stop()
		self.tracemalloc_started = False
		
		# Force garbage collection
		gc.collect()
		
		return {
			'start_memory_mb': self.start_memory,
			'peak_memory_mb': self.peak_memory,
			'end_memory_mb': self.end_memory,
			'memory_increase_mb': self.end_memory - self.start_memory,
			'peak_increase_mb': self.peak_memory - self.start_memory,
			'snapshots': self.memory_snapshots
		}


@pytest.mark.performance
class TestMemoryUsageOptimization:
	"""Test memory usage and optimization patterns."""
	
	async def test_large_dataset_memory_efficiency(self):
		"""Test memory efficiency with large datasets."""
		profiler = MemoryProfiler()
		profiler.start_profiling()
		
		tenant_id = uuid7str()
		user_id = uuid7str()
		customer_service = ARCustomerService(tenant_id, user_id)
		
		# Create large number of customers in memory
		large_dataset_size = 10000
		customers = []
		
		profiler.record_snapshot("Before customer creation")
		
		# Mock service dependencies
		with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock):
			for i in range(large_dataset_size):
				customer_data = {
					'customer_code': f'MEM{i:06d}',
					'legal_name': f'Memory Test Customer {i}',
					'customer_type': ARCustomerType.CORPORATION,
					'status': ARCustomerStatus.ACTIVE,
					'credit_limit': Decimal('50000.00'),
					'payment_terms_days': 30,
					'contact_email': f'customer{i}@memtest.com',
					'total_outstanding': Decimal('5000.00') * (i % 10),  # Varying amounts
					'overdue_amount': Decimal('1000.00') * (i % 5)
				}
				
				with patch('uuid_extensions.uuid7str', return_value=f'mem-customer-{i}'):
					customer = await customer_service.create_customer(customer_data)
					customers.append(customer)
				
				# Record memory usage every 1000 customers
				if (i + 1) % 1000 == 0:
					profiler.record_snapshot(f"After {i+1} customers")
		
		profiler.record_snapshot("After all customers created")
		
		# Test memory efficiency of bulk operations
		filtered_customers = [c for c in customers if c.total_outstanding > Decimal('25000.00')]
		profiler.record_snapshot("After filtering customers")
		
		# Test memory cleanup
		del customers
		del filtered_customers
		gc.collect()
		profiler.record_snapshot("After cleanup")
		
		# Get profiling results
		results = profiler.stop_profiling()
		
		# Validate memory efficiency requirements
		assert results['peak_increase_mb'] <= 500, f"Peak memory increase {results['peak_increase_mb']:.1f}MB exceeds 500MB"
		assert results['memory_increase_mb'] <= 100, f"Final memory increase {results['memory_increase_mb']:.1f}MB exceeds 100MB"
		
		# Calculate memory per customer
		memory_per_customer = (results['peak_increase_mb'] * 1024 * 1024) / large_dataset_size  # bytes
		assert memory_per_customer <= 1024, f"Memory per customer {memory_per_customer:.0f} bytes exceeds 1KB"
		
		print(f"Large Dataset Memory Performance:")
		print(f"  Dataset Size: {large_dataset_size:,} customers")
		print(f"  Peak Memory Increase: {results['peak_increase_mb']:.1f}MB")
		print(f"  Final Memory Increase: {results['memory_increase_mb']:.1f}MB")
		print(f"  Memory per Customer: {memory_per_customer:.0f} bytes")
	
	async def test_concurrent_operations_memory_stability(self):
		"""Test memory stability under concurrent operations."""
		profiler = MemoryProfiler()
		profiler.start_profiling()
		
		tenant_id = uuid7str()
		user_id = uuid7str()
		
		# Initialize multiple services
		customer_service = ARCustomerService(tenant_id, user_id)
		invoice_service = ARInvoiceService(tenant_id, user_id)
		payment_service = ARCashApplicationService(tenant_id, user_id)
		
		profiler.record_snapshot("Services initialized")
		
		async def concurrent_operation_task(task_id: int):
			"""Perform concurrent operations that might cause memory leaks."""
			operations_per_task = 100
			
			# Mock service dependencies
			with patch.object(customer_service, '_validate_permissions', new_callable=AsyncMock), \
				 patch.object(invoice_service, '_validate_permissions', new_callable=AsyncMock), \
				 patch.object(payment_service, '_validate_permissions', new_callable=AsyncMock):
				
				for i in range(operations_per_task):
					# Create customer
					customer_data = {
						'customer_code': f'CONC{task_id:02d}{i:03d}',
						'legal_name': f'Concurrent Customer {task_id}-{i}',
						'customer_type': ARCustomerType.INDIVIDUAL,
						'status': ARCustomerStatus.ACTIVE,
						'credit_limit': Decimal('25000.00'),
						'payment_terms_days': 30
					}
					
					with patch('uuid_extensions.uuid7str', side_effect=[
						f'conc-customer-{task_id}-{i}',
						f'conc-invoice-{task_id}-{i}',
						f'conc-payment-{task_id}-{i}'
					]):
						customer = await customer_service.create_customer(customer_data)
						
						# Create invoice
						invoice_data = {
							'customer_id': customer.id,
							'invoice_number': f'CONC-INV-{task_id:02d}{i:03d}',
							'invoice_date': date.today(),
							'due_date': date.today() + timedelta(days=30),
							'total_amount': Decimal('10000.00'),
							'description': f'Concurrent test invoice {task_id}-{i}'
						}
						
						with patch.object(invoice_service, '_validate_customer_exists', new_callable=AsyncMock):
							invoice = await invoice_service.create_invoice(invoice_data)
						
						# Create payment
						payment_data = {
							'customer_id': customer.id,
							'payment_reference': f'CONC-PAY-{task_id:02d}{i:03d}',
							'payment_date': date.today(),
							'payment_amount': Decimal('6000.00'),
							'payment_method': 'CREDIT_CARD'
						}
						
						with patch.object(payment_service, '_validate_customer_exists', new_callable=AsyncMock):
							payment = await payment_service.create_payment(payment_data)
						
						# Force cleanup of local variables
						del customer, invoice, payment
		
		# Run concurrent tasks
		concurrent_tasks = 20
		tasks = [concurrent_operation_task(i) for i in range(concurrent_tasks)]
		
		profiler.record_snapshot("Before concurrent operations")
		
		await asyncio.gather(*tasks)
		
		profiler.record_snapshot("After concurrent operations")
		
		# Force garbage collection
		gc.collect()
		profiler.record_snapshot("After garbage collection")
		
		# Get profiling results
		results = profiler.stop_profiling()
		
		# Validate memory stability requirements
		total_operations = concurrent_tasks * 100 * 3  # 3 operations per iteration
		
		assert results['peak_increase_mb'] <= 200, f"Peak memory increase {results['peak_increase_mb']:.1f}MB exceeds 200MB"
		assert results['memory_increase_mb'] <= 50, f"Final memory increase {results['memory_increase_mb']:.1f}MB exceeds 50MB"
		
		# Calculate memory per operation
		memory_per_operation = (results['peak_increase_mb'] * 1024 * 1024) / total_operations  # bytes
		assert memory_per_operation <= 512, f"Memory per operation {memory_per_operation:.0f} bytes exceeds 512 bytes"
		
		print(f"Concurrent Operations Memory Performance:")
		print(f"  Total Operations: {total_operations:,}")
		print(f"  Peak Memory Increase: {results['peak_increase_mb']:.1f}MB")
		print(f"  Final Memory Increase: {results['memory_increase_mb']:.1f}MB")
		print(f"  Memory per Operation: {memory_per_operation:.0f} bytes")
	
	async def test_long_running_process_memory_stability(self):
		"""Test memory stability over long-running processes."""
		profiler = MemoryProfiler()
		profiler.start_profiling()
		
		tenant_id = uuid7str()
		user_id = uuid7str()
		analytics_service = ARAnalyticsService(tenant_id, user_id)
		
		# Simulate long-running analytics process
		iterations = 1000
		batch_size = 50
		
		profiler.record_snapshot("Process started")
		
		# Mock service dependencies
		with patch.object(analytics_service, '_validate_permissions', new_callable=AsyncMock):
			for iteration in range(iterations):
				# Simulate periodic analytics calculations
				mock_metrics = {
					'total_ar_balance': Decimal('450000.00') + (iteration * 100),
					'overdue_amount': Decimal('125000.00') + (iteration * 50),
					'current_month_sales': Decimal('85000.00'),
					'total_customers': 156 + iteration,
					'active_customers': 142 + iteration,
					'collection_effectiveness_index': 0.85,
					'ai_assessments_today': 8 + (iteration % 10)
				}
				
				with patch.object(analytics_service, '_calculate_dashboard_metrics', new_callable=AsyncMock) as mock_calc:
					mock_calc.return_value = mock_metrics
					dashboard_metrics = await analytics_service.get_ar_dashboard_metrics()
				
				# Simulate data processing
				processed_data = {
					'iteration': iteration,
					'metrics': dashboard_metrics,
					'calculated_ratios': {
						'overdue_ratio': float(dashboard_metrics['overdue_amount'] / dashboard_metrics['total_ar_balance']),
						'customer_growth': dashboard_metrics['total_customers'] - 156
					}
				}
				
				# Periodic memory snapshots
				if (iteration + 1) % 200 == 0:
					profiler.record_snapshot(f"Iteration {iteration + 1}")
					
					# Force periodic cleanup
					if (iteration + 1) % 400 == 0:
						gc.collect()
						profiler.record_snapshot(f"Iteration {iteration + 1} (after GC)")
				
				# Clear processed data to avoid accumulation
				del processed_data, dashboard_metrics
		
		profiler.record_snapshot("Process completed")
		
		# Get profiling results
		results = profiler.stop_profiling()
		
		# Validate long-running stability requirements
		assert results['peak_increase_mb'] <= 100, f"Peak memory increase {results['peak_increase_mb']:.1f}MB exceeds 100MB"
		assert results['memory_increase_mb'] <= 25, f"Final memory increase {results['memory_increase_mb']:.1f}MB exceeds 25MB"
		
		# Calculate memory stability
		memory_growth_rate = results['memory_increase_mb'] / iterations  # MB per iteration
		assert memory_growth_rate <= 0.01, f"Memory growth rate {memory_growth_rate:.4f}MB/iteration too high"
		
		print(f"Long-Running Process Memory Performance:")
		print(f"  Iterations: {iterations:,}")
		print(f"  Peak Memory Increase: {results['peak_increase_mb']:.1f}MB")
		print(f"  Final Memory Increase: {results['memory_increase_mb']:.1f}MB")
		print(f"  Memory Growth Rate: {memory_growth_rate:.4f}MB/iteration")


@pytest.mark.performance
class TestObjectLifecycleOptimization:
	"""Test object lifecycle and garbage collection optimization."""
	
	async def test_object_creation_cleanup_efficiency(self):
		"""Test efficiency of object creation and cleanup cycles."""
		profiler = MemoryProfiler()
		profiler.start_profiling()
		
		# Test multiple creation/cleanup cycles
		cycles = 10
		objects_per_cycle = 1000
		
		profiler.record_snapshot("Starting cycles")
		
		for cycle in range(cycles):
			# Create objects
			customers = []
			invoices = []
			
			tenant_id = uuid7str()
			user_id = uuid7str()
			
			for i in range(objects_per_cycle):
				# Create customer object
				customer = ARCustomer(
					id=uuid7str(),
					tenant_id=tenant_id,
					customer_code=f'CYCLE{cycle:02d}{i:04d}',
					legal_name=f'Cycle {cycle} Customer {i}',
					customer_type=ARCustomerType.CORPORATION,
					status=ARCustomerStatus.ACTIVE,
					credit_limit=Decimal('50000.00'),
					payment_terms_days=30,
					total_outstanding=Decimal('10000.00'),
					created_by=user_id,
					updated_by=user_id
				)
				customers.append(customer)
				
				# Create invoice object
				invoice = ARInvoice(
					id=uuid7str(),
					tenant_id=tenant_id,
					customer_id=customer.id,
					invoice_number=f'CYCLE-INV-{cycle:02d}{i:04d}',
					invoice_date=date.today(),
					due_date=date.today() + timedelta(days=30),
					total_amount=Decimal('15000.00'),
					outstanding_amount=Decimal('15000.00'),
					status=ARInvoiceStatus.DRAFT,
					created_by=user_id,
					updated_by=user_id
				)
				invoices.append(invoice)
			
			profiler.record_snapshot(f"Cycle {cycle + 1} objects created")
			
			# Process objects (simulate business logic)
			processed_customers = [c for c in customers if c.total_outstanding > Decimal('5000.00')]
			processed_invoices = [i for i in invoices if i.total_amount > Decimal('10000.00')]
			
			# Clear objects
			del customers, invoices, processed_customers, processed_invoices
			
			# Force garbage collection
			gc.collect()
			
			profiler.record_snapshot(f"Cycle {cycle + 1} completed (GC)")
		
		# Get profiling results
		results = profiler.stop_profiling()
		
		# Validate object lifecycle efficiency
		total_objects = cycles * objects_per_cycle * 2  # 2 types of objects
		
		assert results['peak_increase_mb'] <= 150, f"Peak memory increase {results['peak_increase_mb']:.1f}MB exceeds 150MB"
		assert results['memory_increase_mb'] <= 10, f"Final memory increase {results['memory_increase_mb']:.1f}MB exceeds 10MB"
		
		# Memory should return close to baseline after cycles
		cleanup_efficiency = 1 - (results['memory_increase_mb'] / results['peak_increase_mb'])
		assert cleanup_efficiency >= 0.9, f"Cleanup efficiency {cleanup_efficiency:.2f} below 90%"
		
		print(f"Object Lifecycle Optimization Performance:")
		print(f"  Cycles: {cycles}")
		print(f"  Objects per Cycle: {objects_per_cycle:,}")
		print(f"  Total Objects Processed: {total_objects:,}")
		print(f"  Peak Memory Increase: {results['peak_increase_mb']:.1f}MB")
		print(f"  Final Memory Increase: {results['memory_increase_mb']:.1f}MB")
		print(f"  Cleanup Efficiency: {cleanup_efficiency:.1%}")


# Memory performance test runner
if __name__ == "__main__":
	# Check if psutil is available
	try:
		import psutil
		pytest.main([__file__, "-v", "-s", "-m", "performance"])
	except ImportError:
		print("psutil not available. Install with: pip install psutil")
		exit(1)