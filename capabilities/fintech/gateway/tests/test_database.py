"""
Database Service Tests

Comprehensive tests for database operations with real data.

© 2025 Datacraft. All rights reserved.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from uuid_extensions import uuid7str

from ..models import PaymentTransaction, PaymentMethod, PaymentStatus, PaymentMethodType


class TestDatabaseService:
	"""Test database service operations"""
	
	async def test_database_initialization(self, temp_database):
		"""Test database service initialization"""
		assert temp_database is not None
		assert temp_database._engine is not None
		
		# Test health check
		health_result = await temp_database.health_check()
		assert health_result["status"] == "healthy"
	
	async def test_create_payment_transaction(self, temp_database, sample_transaction):
		"""Test creating payment transaction in database"""
		# Create transaction
		created_transaction = await temp_database.create_payment_transaction(sample_transaction)
		
		assert created_transaction.id == sample_transaction.id
		assert created_transaction.amount == sample_transaction.amount
		assert created_transaction.currency == sample_transaction.currency
		assert created_transaction.status == PaymentStatus.PENDING
		
		# Verify it's stored in database
		retrieved_transaction = await temp_database.get_payment_transaction(sample_transaction.id)
		assert retrieved_transaction is not None
		assert retrieved_transaction.id == sample_transaction.id
	
	async def test_update_transaction_status(self, temp_database, sample_transaction):
		"""Test updating transaction status"""
		# Create transaction
		await temp_database.create_payment_transaction(sample_transaction)
		
		# Update status
		updated_transaction = await temp_database.update_transaction_status(
			sample_transaction.id, 
			PaymentStatus.COMPLETED,
			processor_transaction_id="mpesa_12345"
		)
		
		assert updated_transaction.status == PaymentStatus.COMPLETED
		assert updated_transaction.processor_transaction_id == "mpesa_12345"
		assert updated_transaction.updated_at is not None
	
	async def test_create_payment_method(self, temp_database, sample_mpesa_payment_method):
		"""Test creating payment method in database"""
		# Create payment method
		created_method = await temp_database.create_payment_method(sample_mpesa_payment_method)
		
		assert created_method.id == sample_mpesa_payment_method.id
		assert created_method.type == PaymentMethodType.MPESA
		assert created_method.details["phone_number"] == "+254712345678"
		
		# Verify retrieval
		retrieved_method = await temp_database.get_payment_method(sample_mpesa_payment_method.id)
		assert retrieved_method is not None
		assert retrieved_method.id == sample_mpesa_payment_method.id
	
	async def test_get_transactions_by_merchant(self, temp_database, sample_transaction):
		"""Test retrieving transactions by merchant"""
		# Create multiple transactions for different merchants
		transaction1 = sample_transaction
		
		transaction2 = PaymentTransaction(
			id=uuid7str(),
			amount=5000,
			currency="USD",
			payment_method_type=PaymentMethodType.CREDIT_CARD,
			merchant_id="different_merchant",
			customer_id="customer_789",
			description="Different merchant transaction",
			status=PaymentStatus.COMPLETED
		)
		
		await temp_database.create_payment_transaction(transaction1)
		await temp_database.create_payment_transaction(transaction2)
		
		# Get transactions for specific merchant
		merchant_transactions = await temp_database.get_transactions_by_merchant(
			"test_merchant_123", limit=10
		)
		
		assert len(merchant_transactions) == 1
		assert merchant_transactions[0].merchant_id == "test_merchant_123"
	
	async def test_get_transactions_by_customer(self, temp_database, sample_transaction):
		"""Test retrieving transactions by customer"""
		# Create transaction
		await temp_database.create_payment_transaction(sample_transaction)
		
		# Get transactions for customer
		customer_transactions = await temp_database.get_transactions_by_customer(
			"test_customer_456", limit=10
		)
		
		assert len(customer_transactions) == 1
		assert customer_transactions[0].customer_id == "test_customer_456"
	
	async def test_merchant_analytics(self, temp_database, sample_transaction):
		"""Test merchant analytics calculation"""
		# Create multiple transactions
		transactions = []
		for i in range(5):
			transaction = PaymentTransaction(
				id=uuid7str(),
				amount=1000 * (i + 1),  # 10.00, 20.00, 30.00, 40.00, 50.00
				currency="KES",
				payment_method_type=PaymentMethodType.MPESA,
				merchant_id="test_merchant_123",
				customer_id=f"customer_{i}",
				description=f"Test transaction {i}",
				status=PaymentStatus.COMPLETED if i % 2 == 0 else PaymentStatus.FAILED
			)
			transactions.append(transaction)
			await temp_database.create_payment_transaction(transaction)
		
		# Get analytics
		analytics = await temp_database.get_merchant_analytics("test_merchant_123")
		
		assert analytics["merchant_id"] == "test_merchant_123"
		assert analytics["total_transactions"] == 5
		assert analytics["successful_transactions"] == 3  # Even indexed ones
		assert analytics["failed_transactions"] == 2     # Odd indexed ones
		assert analytics["total_amount"] == 15000       # Sum of all amounts
	
	async def test_transaction_analytics(self, temp_database):
		"""Test transaction analytics with date filtering"""
		# Create transactions with different dates
		now = datetime.now(timezone.utc)
		
		# Recent transaction
		recent_transaction = PaymentTransaction(
			id=uuid7str(),
			amount=5000,
			currency="KES", 
			payment_method_type=PaymentMethodType.MPESA,
			merchant_id="merchant_1",
			customer_id="customer_1",
			description="Recent transaction",
			status=PaymentStatus.COMPLETED,
			created_at=now
		)
		
		await temp_database.create_payment_transaction(recent_transaction)
		
		# Get analytics for today
		today_str = now.strftime('%Y-%m-%d')
		analytics = await temp_database.get_transaction_analytics(today_str, today_str)
		
		assert analytics["date_range"]["start"] == today_str
		assert analytics["date_range"]["end"] == today_str
		assert analytics["total_transactions"] >= 1
		assert analytics["total_amount"] >= 5000
	
	async def test_concurrent_transactions(self, temp_database):
		"""Test handling concurrent transaction creation"""
		# Create multiple transactions concurrently
		tasks = []
		for i in range(10):
			transaction = PaymentTransaction(
				id=uuid7str(),
				amount=1000,
				currency="KES",
				payment_method_type=PaymentMethodType.MPESA,
				merchant_id=f"merchant_{i}",
				customer_id=f"customer_{i}",
				description=f"Concurrent transaction {i}",
				status=PaymentStatus.PENDING
			)
			tasks.append(temp_database.create_payment_transaction(transaction))
		
		# Execute all tasks concurrently
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		# Verify all transactions were created successfully
		successful_creates = [r for r in results if not isinstance(r, Exception)]
		assert len(successful_creates) == 10
	
	async def test_system_metrics(self, temp_database, sample_transaction):
		"""Test system metrics collection"""
		# Create some test data
		await temp_database.create_payment_transaction(sample_transaction)
		
		# Get system metrics
		metrics = await temp_database.get_system_metrics()
		
		assert "database_status" in metrics
		assert "connection_pool" in metrics
		assert "query_performance" in metrics
		assert metrics["database_status"] == "healthy"