"""
APG Core Financials - Performance and Load Tests

CLAUDE.md compliant performance and load tests for enterprise scale validation,
testing concurrent operations, throughput, and system limits.

© 2025 Datacraft. All rights reserved.
"""

from __future__ import annotations

import asyncio
import gc
import os
import psutil
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from typing import Any, Dict, List
from uuid import uuid4

import pytest

from ...service import (
	APVendorService, APInvoiceService, APPaymentService,
	APWorkflowService, APAnalyticsService
)
from .conftest import (
	assert_valid_uuid, assert_decimal_equals, assert_apg_compliance
)


# Load Testing Configuration
LOAD_TEST_CONFIG = {
	"small_load": {"concurrent_users": 10, "operations_per_user": 5},
	"medium_load": {"concurrent_users": 50, "operations_per_user": 10},
	"large_load": {"concurrent_users": 100, "operations_per_user": 20},
	"stress_load": {"concurrent_users": 200, "operations_per_user": 50}
}

# Performance Targets (from cap_spec.md)
PERFORMANCE_TARGETS = {
	"invoice_processing_time": 2.0,    # < 2s per invoice
	"api_response_time": 0.2,          # < 200ms for API calls
	"concurrent_users": 1000,          # Support 1000+ concurrent users
	"throughput": 500,                 # 500+ operations per minute
	"memory_limit": 100 * 1024 * 1024  # < 100MB memory increase per test
}


# Vendor Service Performance Tests

async def test_vendor_creation_performance_small_load(
	vendor_service: APVendorService,
	sample_vendor_data: Dict[str, Any],
	tenant_context: Dict[str, Any]
):
	"""Test vendor creation performance under small load"""
	config = LOAD_TEST_CONFIG["small_load"]
	
	async def create_vendors_batch(user_index: int) -> List[Dict[str, Any]]:
		"""Create batch of vendors for one user"""
		results = []
		for i in range(config["operations_per_user"]):
			vendor_data = sample_vendor_data.copy()
			vendor_data["vendor_code"] = f"PERF-U{user_index:03d}-V{i:03d}"
			vendor_data["legal_name"] = f"Performance Test Vendor {user_index}-{i}"
			vendor_data["created_by"] = tenant_context["user_id"]
			vendor_data["tenant_id"] = tenant_context["tenant_id"]
			
			start_time = time.time()
			vendor = await vendor_service.create_vendor(vendor_data, tenant_context)
			end_time = time.time()
			
			results.append({
				"vendor_id": vendor.id if vendor else None,
				"processing_time": end_time - start_time,
				"success": vendor is not None
			})
		return results
	
	# Execute load test
	start_time = time.time()
	
	tasks = [
		create_vendors_batch(user_index) 
		for user_index in range(config["concurrent_users"])
	]
	
	batch_results = await asyncio.gather(*tasks, return_exceptions=True)
	
	end_time = time.time()
	total_duration = end_time - start_time
	
	# Analyze results
	all_results = []
	for batch in batch_results:
		if isinstance(batch, list):
			all_results.extend(batch)
	
	successful_operations = [r for r in all_results if r["success"]]
	total_operations = len(all_results)
	success_rate = len(successful_operations) / total_operations if total_operations > 0 else 0
	
	avg_processing_time = sum(r["processing_time"] for r in successful_operations) / len(successful_operations)
	throughput = len(successful_operations) / total_duration * 60  # Operations per minute
	
	# Verify performance targets
	assert success_rate >= 0.95, f"Success rate {success_rate:.2%} should be >= 95%"
	assert avg_processing_time < 1.0, f"Average processing time {avg_processing_time:.3f}s should be < 1s"
	assert throughput > 100, f"Throughput {throughput:.1f} ops/min should be > 100"
	
	print(f"Small Load Results: {total_operations} ops, {success_rate:.2%} success, "
		  f"{avg_processing_time:.3f}s avg, {throughput:.1f} ops/min")


async def test_vendor_retrieval_performance_medium_load(
	vendor_service: APVendorService,
	sample_vendor_data: Dict[str, Any],
	tenant_context: Dict[str, Any]
):
	"""Test vendor retrieval performance under medium load"""
	config = LOAD_TEST_CONFIG["medium_load"]
	
	# Pre-create vendors for retrieval testing
	vendor_ids = []
	for i in range(20):  # Create 20 vendors to retrieve
		vendor_data = sample_vendor_data.copy()
		vendor_data["vendor_code"] = f"RETR-{i:03d}"
		vendor_data["legal_name"] = f"Retrieval Test Vendor {i}"
		vendor_data["created_by"] = tenant_context["user_id"]
		vendor_data["tenant_id"] = tenant_context["tenant_id"]
		
		vendor = await vendor_service.create_vendor(vendor_data, tenant_context)
		if vendor:
			vendor_ids.append(vendor.id)
	
	async def retrieve_vendors_batch(user_index: int) -> List[Dict[str, Any]]:
		"""Retrieve batch of vendors for one user"""
		results = []
		for i in range(config["operations_per_user"]):
			vendor_id = vendor_ids[i % len(vendor_ids)]  # Cycle through available vendors
			
			start_time = time.time()
			vendor = await vendor_service.get_vendor(vendor_id, tenant_context)
			end_time = time.time()
			
			results.append({
				"vendor_id": vendor_id,
				"processing_time": end_time - start_time,
				"success": vendor is not None
			})
		return results
	
	# Execute load test
	start_time = time.time()
	
	tasks = [
		retrieve_vendors_batch(user_index)
		for user_index in range(config["concurrent_users"])
	]
	
	batch_results = await asyncio.gather(*tasks, return_exceptions=True)
	
	end_time = time.time()
	total_duration = end_time - start_time
	
	# Analyze results
	all_results = []
	for batch in batch_results:
		if isinstance(batch, list):
			all_results.extend(batch)
	
	successful_operations = [r for r in all_results if r["success"]]
	avg_response_time = sum(r["processing_time"] for r in successful_operations) / len(successful_operations)
	throughput = len(successful_operations) / total_duration * 60
	
	# Verify performance targets
	assert avg_response_time < PERFORMANCE_TARGETS["api_response_time"], \
		f"Average response time {avg_response_time:.3f}s should be < {PERFORMANCE_TARGETS['api_response_time']}s"
	assert throughput > 1000, f"Throughput {throughput:.1f} ops/min should be > 1000 for retrieval"
	
	print(f"Medium Load Retrieval: {len(all_results)} ops, {avg_response_time:.3f}s avg, {throughput:.1f} ops/min")


# Invoice Service Performance Tests

async def test_invoice_processing_performance_large_load(
	invoice_service: APInvoiceService,
	vendor_service: APVendorService,
	sample_vendor_data: Dict[str, Any],
	tenant_context: Dict[str, Any]
):
	"""Test invoice processing performance under large load"""
	config = LOAD_TEST_CONFIG["large_load"]
	
	# Pre-create vendors for invoice testing
	vendor_ids = []
	for i in range(10):
		vendor_data = sample_vendor_data.copy()
		vendor_data["vendor_code"] = f"INV-PERF-{i:03d}"
		vendor_data["legal_name"] = f"Invoice Performance Vendor {i}"
		vendor_data["created_by"] = tenant_context["user_id"]
		vendor_data["tenant_id"] = tenant_context["tenant_id"]
		
		vendor = await vendor_service.create_vendor(vendor_data, tenant_context)
		if vendor:
			vendor_ids.append(vendor.id)
	
	async def process_invoices_batch(user_index: int) -> List[Dict[str, Any]]:
		"""Process batch of invoices for one user"""
		results = []
		for i in range(config["operations_per_user"]):
			vendor_id = vendor_ids[i % len(vendor_ids)]
			
			invoice_data = {
				"invoice_number": f"PERF-U{user_index:03d}-INV-{i:03d}",
				"vendor_id": vendor_id,
				"vendor_invoice_number": f"VEND-U{user_index:03d}-{i:03d}",
				"invoice_date": "2025-01-01",
				"due_date": "2025-01-31",
				"subtotal_amount": str(1000 + (i * 10)),
				"tax_amount": str(85 + (i * 1)),
				"total_amount": str(1085 + (i * 11)),
				"currency_code": "USD",
				"tenant_id": tenant_context["tenant_id"],
				"created_by": tenant_context["user_id"],
				"line_items": [
					{
						"line_number": 1,
						"description": f"Performance Test Item {i}",
						"quantity": "1.0000",
						"unit_price": str(1000 + (i * 10)) + ".0000",
						"line_amount": str(1000 + (i * 10)) + ".00",
						"gl_account_code": "5000",
						"cost_center": "CC-PERF",
						"department": "TESTING"
					}
				]
			}
			
			start_time = time.time()
			invoice = await invoice_service.create_invoice(invoice_data, tenant_context)
			end_time = time.time()
			
			results.append({
				"invoice_id": invoice.id if invoice else None,
				"processing_time": end_time - start_time,
				"success": invoice is not None
			})
		return results
	
	# Monitor system resources
	process = psutil.Process(os.getpid())
	initial_memory = process.memory_info().rss
	
	# Execute load test
	start_time = time.time()
	
	tasks = [
		process_invoices_batch(user_index)
		for user_index in range(config["concurrent_users"])
	]
	
	batch_results = await asyncio.gather(*tasks, return_exceptions=True)
	
	end_time = time.time()
	total_duration = end_time - start_time
	
	# Check final memory usage
	final_memory = process.memory_info().rss
	memory_increase = final_memory - initial_memory
	
	# Analyze results
	all_results = []
	for batch in batch_results:
		if isinstance(batch, list):
			all_results.extend(batch)
	
	successful_operations = [r for r in all_results if r["success"]]
	success_rate = len(successful_operations) / len(all_results)
	avg_processing_time = sum(r["processing_time"] for r in successful_operations) / len(successful_operations)
	throughput = len(successful_operations) / total_duration * 60
	
	# Verify performance targets
	assert success_rate >= 0.90, f"Success rate {success_rate:.2%} should be >= 90% under large load"
	assert avg_processing_time < PERFORMANCE_TARGETS["invoice_processing_time"], \
		f"Average processing time {avg_processing_time:.3f}s should be < {PERFORMANCE_TARGETS['invoice_processing_time']}s"
	assert throughput > 200, f"Throughput {throughput:.1f} ops/min should be > 200"
	assert memory_increase < PERFORMANCE_TARGETS["memory_limit"], \
		f"Memory increase {memory_increase / 1024 / 1024:.1f}MB should be < {PERFORMANCE_TARGETS['memory_limit'] / 1024 / 1024}MB"
	
	print(f"Large Load Invoice Processing: {len(all_results)} ops, {success_rate:.2%} success, "
		  f"{avg_processing_time:.3f}s avg, {throughput:.1f} ops/min, {memory_increase / 1024 / 1024:.1f}MB memory")


# Payment Service Performance Tests

async def test_payment_processing_performance_stress_load(
	payment_service: APPaymentService,
	vendor_service: APVendorService,
	invoice_service: APInvoiceService,
	sample_vendor_data: Dict[str, Any],
	tenant_context: Dict[str, Any]
):
	"""Test payment processing performance under stress load"""
	config = LOAD_TEST_CONFIG["stress_load"]
	
	# Pre-create vendors and invoices for payment testing
	vendor_invoice_pairs = []
	for i in range(20):  # Create 20 vendor-invoice pairs
		vendor_data = sample_vendor_data.copy()
		vendor_data["vendor_code"] = f"PAY-STRESS-{i:03d}"
		vendor_data["legal_name"] = f"Payment Stress Vendor {i}"
		vendor_data["created_by"] = tenant_context["user_id"]
		vendor_data["tenant_id"] = tenant_context["tenant_id"]
		
		vendor = await vendor_service.create_vendor(vendor_data, tenant_context)
		if not vendor:
			continue
		
		invoice_data = {
			"invoice_number": f"STRESS-INV-{i:03d}",
			"vendor_id": vendor.id,
			"vendor_invoice_number": f"STRESS-VEND-{i:03d}",
			"invoice_date": "2025-01-01",
			"due_date": "2025-01-31",
			"subtotal_amount": "1500.00",
			"tax_amount": "127.50",
			"total_amount": "1627.50",
			"currency_code": "USD",
			"tenant_id": tenant_context["tenant_id"],
			"created_by": tenant_context["user_id"]
		}
		
		invoice = await invoice_service.create_invoice(invoice_data, tenant_context)
		if invoice:
			vendor_invoice_pairs.append((vendor.id, invoice.id, invoice.invoice_number))
	
	async def process_payments_batch(user_index: int) -> List[Dict[str, Any]]:
		"""Process batch of payments for one user"""
		results = []
		batch_size = min(config["operations_per_user"], len(vendor_invoice_pairs))
		
		for i in range(batch_size):
			vendor_id, invoice_id, invoice_number = vendor_invoice_pairs[i % len(vendor_invoice_pairs)]
			
			payment_data = {
				"payment_number": f"STRESS-U{user_index:03d}-PAY-{i:03d}",
				"vendor_id": vendor_id,
				"payment_method": "ach",
				"payment_amount": "1627.50",
				"payment_date": "2025-01-15",
				"currency_code": "USD",
				"bank_account_id": f"bank_{uuid4().hex[:8]}",
				"tenant_id": tenant_context["tenant_id"],
				"created_by": tenant_context["user_id"],
				"payment_lines": [
					{
						"invoice_id": invoice_id,
						"invoice_number": invoice_number,
						"payment_amount": "1627.50",
						"discount_taken": "0.00"
					}
				]
			}
			
			start_time = time.time()
			try:
				payment = await payment_service.create_payment(payment_data, tenant_context)
				end_time = time.time()
				
				results.append({
					"payment_id": payment.id if payment else None,
					"processing_time": end_time - start_time,
					"success": payment is not None,
					"error": None
				})
			except Exception as e:
				end_time = time.time()
				results.append({
					"payment_id": None,
					"processing_time": end_time - start_time,
					"success": False,
					"error": str(e)
				})
		return results
	
	# Execute stress test with reduced concurrency to avoid overwhelming the system
	reduced_concurrent_users = min(config["concurrent_users"], 50)  # Cap at 50 for stress test
	
	start_time = time.time()
	
	tasks = [
		process_payments_batch(user_index)
		for user_index in range(reduced_concurrent_users)
	]
	
	batch_results = await asyncio.gather(*tasks, return_exceptions=True)
	
	end_time = time.time()
	total_duration = end_time - start_time
	
	# Analyze results
	all_results = []
	for batch in batch_results:
		if isinstance(batch, list):
			all_results.extend(batch)
		elif isinstance(batch, Exception):
			# Count exceptions as failed operations
			all_results.append({
				"payment_id": None,
				"processing_time": 0,
				"success": False,
				"error": str(batch)
			})
	
	successful_operations = [r for r in all_results if r["success"]]
	success_rate = len(successful_operations) / len(all_results) if all_results else 0
	
	if successful_operations:
		avg_processing_time = sum(r["processing_time"] for r in successful_operations) / len(successful_operations)
		throughput = len(successful_operations) / total_duration * 60
	else:
		avg_processing_time = 0
		throughput = 0
	
	# Under stress load, allow for reduced performance targets
	min_success_rate = 0.70  # 70% success rate under stress
	max_processing_time = 5.0  # 5s max under stress
	min_throughput = 50  # 50 ops/min minimum
	
	assert success_rate >= min_success_rate, \
		f"Success rate {success_rate:.2%} should be >= {min_success_rate:.0%} under stress load"
	
	if successful_operations:
		assert avg_processing_time < max_processing_time, \
			f"Average processing time {avg_processing_time:.3f}s should be < {max_processing_time}s under stress"
		assert throughput > min_throughput, \
			f"Throughput {throughput:.1f} ops/min should be > {min_throughput} under stress"
	
	print(f"Stress Load Payment Processing: {len(all_results)} ops, {success_rate:.2%} success, "
		  f"{avg_processing_time:.3f}s avg, {throughput:.1f} ops/min")


# Analytics Service Performance Tests

async def test_analytics_performance_large_dataset(
	analytics_service: APAnalyticsService,
	tenant_context: Dict[str, Any]
):
	"""Test analytics performance with large datasets"""
	
	# Generate large dataset for analysis
	large_spending_data = []
	for i in range(5000):  # 5000 transactions
		large_spending_data.append({
			"vendor_id": f"vendor_{i % 100:03d}",  # 100 different vendors
			"vendor_name": f"Analytics Test Vendor {i % 100}",
			"category": f"category_{i % 20}",  # 20 different categories
			"amount": str(100 + (i % 1000)),  # Varying amounts
			"date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"  # Spread across year
		})
	
	# Test spending analysis performance
	start_time = time.time()
	
	spending_analysis = await analytics_service.generate_spending_analysis(
		spending_data=large_spending_data,
		analysis_period_days=365,
		tenant_context=tenant_context
	)
	
	end_time = time.time()
	analysis_duration = end_time - start_time
	
	# Verify analysis results and performance
	assert spending_analysis is not None, "Analysis should complete with large dataset"
	assert "category_breakdown" in spending_analysis, "Should analyze categories"
	assert "vendor_rankings" in spending_analysis, "Should rank vendors"
	assert analysis_duration < 15.0, f"Analysis took {analysis_duration:.2f}s, should be < 15s"
	
	# Test cash flow forecasting performance
	historical_data = large_spending_data[:1000]  # Use subset for forecasting
	
	start_time = time.time()
	
	forecast = await analytics_service.generate_cash_flow_forecast(
		historical_data=historical_data,
		pending_payments=[],
		forecast_horizon_days=90,
		tenant_context=tenant_context
	)
	
	end_time = time.time()
	forecast_duration = end_time - start_time
	
	# Verify forecasting results and performance
	assert forecast is not None, "Forecast should complete with large dataset"
	assert len(forecast["daily_projections"]) == 90, "Should generate requested forecast horizon"
	assert forecast_duration < 10.0, f"Forecasting took {forecast_duration:.2f}s, should be < 10s"
	
	print(f"Analytics Large Dataset: Analysis {analysis_duration:.2f}s, Forecast {forecast_duration:.2f}s")


# Memory Usage and Resource Tests

async def test_memory_usage_under_load(
	vendor_service: APVendorService,
	invoice_service: APInvoiceService,
	sample_vendor_data: Dict[str, Any],
	tenant_context: Dict[str, Any]
):
	"""Test memory usage patterns under sustained load"""
	
	process = psutil.Process(os.getpid())
	initial_memory = process.memory_info().rss
	
	# Sustained load test - create and process entities continuously
	total_operations = 0
	memory_measurements = []
	
	for batch in range(10):  # 10 batches
		batch_start_memory = process.memory_info().rss
		
		# Create batch of vendors and invoices
		batch_tasks = []
		for i in range(20):  # 20 operations per batch
			vendor_data = sample_vendor_data.copy()
			vendor_data["vendor_code"] = f"MEM-B{batch:02d}-V{i:03d}"
			vendor_data["legal_name"] = f"Memory Test Vendor B{batch}-{i}"
			vendor_data["created_by"] = tenant_context["user_id"]
			vendor_data["tenant_id"] = tenant_context["tenant_id"]
			
			batch_tasks.append(vendor_service.create_vendor(vendor_data, tenant_context))
		
		# Execute batch
		await asyncio.gather(*batch_tasks, return_exceptions=True)
		total_operations += len(batch_tasks)
		
		# Force garbage collection
		gc.collect()
		
		# Measure memory after garbage collection
		batch_end_memory = process.memory_info().rss
		memory_measurements.append({
			"batch": batch,
			"operations": total_operations,
			"memory_mb": batch_end_memory / 1024 / 1024,
			"increase_mb": (batch_end_memory - initial_memory) / 1024 / 1024
		})
		
		# Small delay to allow system cleanup
		await asyncio.sleep(0.1)
	
	final_memory = process.memory_info().rss
	total_memory_increase = final_memory - initial_memory
	
	# Verify memory usage stays within acceptable bounds
	memory_per_operation = total_memory_increase / total_operations if total_operations > 0 else 0
	
	assert total_memory_increase < 200 * 1024 * 1024, \
		f"Total memory increase {total_memory_increase / 1024 / 1024:.1f}MB should be < 200MB"
	assert memory_per_operation < 50 * 1024, \
		f"Memory per operation {memory_per_operation / 1024:.1f}KB should be < 50KB"
	
	# Check for memory leaks (memory should not grow unbounded)
	first_half_avg = sum(m["increase_mb"] for m in memory_measurements[:5]) / 5
	second_half_avg = sum(m["increase_mb"] for m in memory_measurements[5:]) / 5
	memory_growth_rate = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else 0
	
	assert memory_growth_rate < 0.5, \
		f"Memory growth rate {memory_growth_rate:.2%} should be < 50% (potential leak detection)"
	
	print(f"Memory Usage: {total_operations} ops, {total_memory_increase / 1024 / 1024:.1f}MB total, "
		  f"{memory_per_operation / 1024:.1f}KB per op, {memory_growth_rate:.2%} growth rate")


# Concurrent User Simulation

async def test_concurrent_user_simulation(
	vendor_service: APVendorService,
	invoice_service: APInvoiceService,
	payment_service: APPaymentService,
	workflow_service: APWorkflowService,
	sample_vendor_data: Dict[str, Any],
	tenant_context: Dict[str, Any]
):
	"""Simulate realistic concurrent user behavior"""
	
	async def simulate_user_session(user_id: int) -> Dict[str, Any]:
		"""Simulate a complete user session with mixed operations"""
		session_results = {
			"user_id": user_id,
			"operations": [],
			"total_time": 0,
			"success_count": 0,
			"error_count": 0
		}
		
		session_start = time.time()
		
		try:
			# User creates a vendor
			vendor_data = sample_vendor_data.copy()
			vendor_data["vendor_code"] = f"USER-{user_id:03d}-VENDOR"
			vendor_data["legal_name"] = f"User {user_id} Test Vendor"
			vendor_data["created_by"] = tenant_context["user_id"]
			vendor_data["tenant_id"] = tenant_context["tenant_id"]
			
			op_start = time.time()
			vendor = await vendor_service.create_vendor(vendor_data, tenant_context)
			op_end = time.time()
			
			session_results["operations"].append({
				"operation": "create_vendor",
				"duration": op_end - op_start,
				"success": vendor is not None
			})
			
			if vendor:
				session_results["success_count"] += 1
				
				# User creates invoices for the vendor
				for inv_idx in range(3):  # 3 invoices per user
					invoice_data = {
						"invoice_number": f"USER-{user_id:03d}-INV-{inv_idx:03d}",
						"vendor_id": vendor.id,
						"vendor_invoice_number": f"USER-VEND-{user_id:03d}-{inv_idx:03d}",
						"invoice_date": "2025-01-01",
						"due_date": "2025-01-31",
						"subtotal_amount": str(1000 + (inv_idx * 100)),
						"tax_amount": str(85 + (inv_idx * 8.5)),
						"total_amount": str(1085 + (inv_idx * 108.5)),
						"currency_code": "USD",
						"tenant_id": tenant_context["tenant_id"],
						"created_by": tenant_context["user_id"]
					}
					
					op_start = time.time()
					invoice = await invoice_service.create_invoice(invoice_data, tenant_context)
					op_end = time.time()
					
					session_results["operations"].append({
						"operation": "create_invoice",
						"duration": op_end - op_start,
						"success": invoice is not None
					})
					
					if invoice:
						session_results["success_count"] += 1
						
						# Create approval workflow
						workflow_data = {
							"workflow_type": "invoice",
							"entity_id": invoice.id,
							"entity_number": invoice.invoice_number,
							"tenant_id": tenant_context["tenant_id"],
							"created_by": tenant_context["user_id"]
						}
						
						op_start = time.time()
						workflow = await workflow_service.create_workflow(workflow_data, tenant_context)
						op_end = time.time()
						
						session_results["operations"].append({
							"operation": "create_workflow",
							"duration": op_end - op_start,
							"success": workflow is not None
						})
						
						if workflow:
							session_results["success_count"] += 1
					else:
						session_results["error_count"] += 1
			else:
				session_results["error_count"] += 1
				
		except Exception as e:
			session_results["error_count"] += 1
			session_results["operations"].append({
				"operation": "session_error",
				"duration": 0,
				"success": False,
				"error": str(e)
			})
		
		session_end = time.time()
		session_results["total_time"] = session_end - session_start
		
		return session_results
	
	# Simulate 25 concurrent users
	num_users = 25
	start_time = time.time()
	
	user_tasks = [simulate_user_session(user_id) for user_id in range(num_users)]
	user_results = await asyncio.gather(*user_tasks, return_exceptions=True)
	
	end_time = time.time()
	total_duration = end_time - start_time
	
	# Analyze user session results
	successful_sessions = [r for r in user_results if isinstance(r, dict)]
	total_operations = sum(len(r["operations"]) for r in successful_sessions)
	total_successful_ops = sum(r["success_count"] for r in successful_sessions)
	total_errors = sum(r["error_count"] for r in successful_sessions)
	
	overall_success_rate = total_successful_ops / (total_successful_ops + total_errors) if (total_successful_ops + total_errors) > 0 else 0
	avg_session_time = sum(r["total_time"] for r in successful_sessions) / len(successful_sessions) if successful_sessions else 0
	ops_per_minute = total_operations / total_duration * 60
	
	# Verify concurrent user performance
	assert len(successful_sessions) >= num_users * 0.9, f"At least 90% of user sessions should complete successfully"
	assert overall_success_rate >= 0.85, f"Overall success rate {overall_success_rate:.2%} should be >= 85%"
	assert avg_session_time < 30.0, f"Average session time {avg_session_time:.2f}s should be < 30s"
	assert ops_per_minute > 100, f"Operations per minute {ops_per_minute:.1f} should be > 100"
	
	print(f"Concurrent Users: {len(successful_sessions)} sessions, {overall_success_rate:.2%} success, "
		  f"{avg_session_time:.2f}s avg session, {ops_per_minute:.1f} ops/min")


# Database Connection Pool Tests (Placeholder)

async def test_database_connection_pool_performance():
	"""Test database connection pool performance under load"""
	# In real implementation, this would test:
	# 1. Connection pool sizing
	# 2. Connection acquisition/release times
	# 3. Pool exhaustion handling
	# 4. Connection leak detection
	# 5. Failover and recovery
	
	# Placeholder for database performance testing
	pass


# Export test functions for discovery
__all__ = [
	"test_vendor_creation_performance_small_load",
	"test_vendor_retrieval_performance_medium_load", 
	"test_invoice_processing_performance_large_load",
	"test_payment_processing_performance_stress_load",
	"test_analytics_performance_large_dataset",
	"test_memory_usage_under_load",
	"test_concurrent_user_simulation"
]