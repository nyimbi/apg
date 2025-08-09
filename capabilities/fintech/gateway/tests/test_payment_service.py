"""
Payment Service Tests

Comprehensive tests for payment processing service with real data.

© 2025 Datacraft. All rights reserved.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from uuid_extensions import uuid7str

from ..models import PaymentTransaction, PaymentMethod, PaymentStatus, PaymentMethodType
from ..payment_processor import PaymentResult


class TestPaymentGatewayService:
	"""Test payment gateway service operations"""
	
	async def test_service_initialization(self, payment_service):
		"""Test payment service initialization"""
		assert payment_service is not None
		assert payment_service._initialized is True
		assert payment_service._database_service is not None
		assert len(payment_service._processors) > 0
	
	async def test_processor_loading(self, payment_service):
		"""Test payment processor loading"""
		# Verify all expected processors are loaded
		expected_processors = ["mpesa", "stripe", "paypal", "adyen"]
		
		for processor_name in expected_processors:
			assert processor_name in payment_service._processors
			processor = payment_service._processors[processor_name]
			assert processor is not None
			assert hasattr(processor, 'process_payment')
	
	async def test_mpesa_payment_processing(self, payment_service, sample_transaction, sample_mpesa_payment_method):
		"""Test MPESA payment processing with real data"""
		# Set test mode to avoid actual API calls
		additional_data = {"test_mode": True}
		
		# Process payment
		result = await payment_service.process_payment(
			sample_transaction,
			sample_mpesa_payment_method, 
			additional_data
		)
		
		assert isinstance(result, PaymentResult)
		# In test mode, we expect either success or pending status
		assert result.status in [PaymentStatus.COMPLETED, PaymentStatus.PENDING, PaymentStatus.PROCESSING]
		
		# Verify transaction was stored in database
		stored_transaction = await payment_service._database_service.get_payment_transaction(
			sample_transaction.id
		)
		assert stored_transaction is not None
		assert stored_transaction.id == sample_transaction.id
	
	async def test_card_payment_processing(self, payment_service, sample_card_payment_method):
		"""Test credit card payment processing"""
		# Create card transaction
		card_transaction = PaymentTransaction(
			id=uuid7str(),
			amount=5000,  # $50.00
			currency="USD",
			payment_method_type=PaymentMethodType.CREDIT_CARD,
			merchant_id="test_merchant_card",
			customer_id="test_customer_card",
			description="Test card payment",
			status=PaymentStatus.PENDING
		)
		
		# Process payment in test mode
		result = await payment_service.process_payment(
			card_transaction,
			sample_card_payment_method,
			{"test_mode": True}
		)
		
		assert isinstance(result, PaymentResult)
		# Card payments typically complete immediately or require 3DS
		assert result.status in [
			PaymentStatus.COMPLETED,
			PaymentStatus.PENDING, 
			PaymentStatus.AUTHORIZED
		]
	
	async def test_payment_capture(self, payment_service, sample_transaction, sample_card_payment_method):
		"""Test payment capture functionality"""
		# First create an authorized payment
		sample_transaction.status = PaymentStatus.AUTHORIZED
		await payment_service._database_service.create_payment_transaction(sample_transaction)
		
		# Capture the payment
		capture_result = await payment_service.capture_payment(
			sample_transaction.id,
			amount=sample_transaction.amount
		)
		
		assert isinstance(capture_result, PaymentResult)
		# Capture should succeed or be processing
		assert capture_result.status in [PaymentStatus.CAPTURED, PaymentStatus.PROCESSING]
	
	async def test_payment_refund(self, payment_service, sample_transaction):
		"""Test payment refund functionality"""
		# First create a completed payment
		sample_transaction.status = PaymentStatus.COMPLETED
		await payment_service._database_service.create_payment_transaction(sample_transaction)
		
		# Refund the payment
		refund_result = await payment_service.refund_payment(
			sample_transaction.id,
			amount=5000,  # Partial refund
			reason="Customer request"
		)
		
		assert isinstance(refund_result, PaymentResult)
		# Refund should succeed or be processing
		assert refund_result.status in [PaymentStatus.REFUNDED, PaymentStatus.PROCESSING]
	
	async def test_concurrent_payment_processing(self, payment_service):
		"""Test concurrent payment processing"""
		# Create multiple payment tasks
		tasks = []
		for i in range(5):
			transaction = PaymentTransaction(
				id=uuid7str(),
				amount=1000 * (i + 1),
				currency="KES",
				payment_method_type=PaymentMethodType.MPESA,
				merchant_id=f"merchant_{i}",
				customer_id=f"customer_{i}",
				description=f"Concurrent payment {i}",
				status=PaymentStatus.PENDING
			)
			
			payment_method = PaymentMethod(
				id=uuid7str(),
				type=PaymentMethodType.MPESA,
				details={"phone_number": f"+25471234567{i}"},
				customer_id=f"customer_{i}"
			)
			
			tasks.append(payment_service.process_payment(
				transaction, payment_method, {"test_mode": True}
			))
		
		# Execute all payments concurrently
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		# Verify all payments were processed
		successful_payments = [r for r in results if isinstance(r, PaymentResult)]
		assert len(successful_payments) == 5
	
	async def test_payment_status_tracking(self, payment_service, sample_transaction, sample_mpesa_payment_method):
		"""Test payment status tracking through lifecycle"""
		# Process initial payment
		result = await payment_service.process_payment(
			sample_transaction,
			sample_mpesa_payment_method,
			{"test_mode": True}
		)
		
		# Check initial status
		stored_transaction = await payment_service._database_service.get_payment_transaction(
			sample_transaction.id
		)
		assert stored_transaction.status in [
			PaymentStatus.PENDING,
			PaymentStatus.PROCESSING,
			PaymentStatus.COMPLETED
		]
		
		# Simulate status update (e.g., from webhook)
		await payment_service.update_payment_status(
			sample_transaction.id,
			PaymentStatus.COMPLETED,
			processor_transaction_id="mpesa_test_12345"
		)
		
		# Verify status update
		updated_transaction = await payment_service._database_service.get_payment_transaction(
			sample_transaction.id
		)
		assert updated_transaction.status == PaymentStatus.COMPLETED
		assert updated_transaction.processor_transaction_id == "mpesa_test_12345"
	
	async def test_fraud_analysis(self, payment_service, sample_transaction, sample_mpesa_payment_method):
		"""Test fraud analysis during payment processing"""
		# Add suspicious data
		additional_data = {
			"test_mode": True,
			"risk_data": {
				"ip_address": "1.2.3.4",
				"user_agent": "Test Agent",
				"velocity_check": True
			}
		}
		
		# Process payment with fraud analysis
		result = await payment_service.process_payment(
			sample_transaction,
			sample_mpesa_payment_method,
			additional_data
		)
		
		# Verify fraud analysis was performed
		assert isinstance(result, PaymentResult)
		# In test mode, should complete normally
		assert result.status in [PaymentStatus.COMPLETED, PaymentStatus.PENDING]
		
		# Check if fraud analysis was stored
		fraud_analyses = await payment_service._database_service.get_fraud_analyses_by_transaction(
			sample_transaction.id
		)
		assert len(fraud_analyses) >= 1
	
	async def test_webhook_processing(self, payment_service):
		"""Test webhook processing for payment updates"""
		# Create test webhook payload (MPESA format)
		webhook_payload = {
			"Body": {
				"stkCallback": {
					"MerchantRequestID": "test_merchant_req_123",
					"CheckoutRequestID": "test_checkout_req_456", 
					"ResultCode": 0,
					"ResultDesc": "The service request is processed successfully.",
					"CallbackMetadata": {
						"Item": [
							{"Name": "Amount", "Value": 100.00},
							{"Name": "MpesaReceiptNumber", "Value": "LGR7OWQX0R"},
							{"Name": "TransactionDate", "Value": 20250130143000},
							{"Name": "PhoneNumber", "Value": "254712345678"}
						]
					}
				}
			}
		}
		
		# Process webhook
		result = await payment_service.process_webhook(
			"mpesa",
			webhook_payload,
			{"test_mode": True}
		)
		
		assert result["processed"] is True
		assert "transaction_id" in result or "merchant_request_id" in result
	
	async def test_service_health_check(self, payment_service):
		"""Test service health check"""
		health_result = await payment_service.health_check()
		
		assert health_result["status"] in ["healthy", "degraded"]
		assert "processors" in health_result
		assert "database" in health_result
		
		# Check processor health
		for processor_name, processor_health in health_result["processors"].items():
			assert processor_name in ["mpesa", "stripe", "paypal", "adyen"]
			assert "status" in processor_health
	
	async def test_service_metrics(self, payment_service, sample_transaction, sample_mpesa_payment_method):
		"""Test service metrics collection"""
		# Process a payment to generate metrics
		await payment_service.process_payment(
			sample_transaction,
			sample_mpesa_payment_method,
			{"test_mode": True}
		)
		
		# Get service metrics
		metrics = await payment_service.get_metrics()
		
		assert "total_transactions" in metrics
		assert "successful_transactions" in metrics
		assert "failed_transactions" in metrics
		assert "processor_performance" in metrics
		assert metrics["total_transactions"] >= 1
	
	async def test_multi_currency_support(self, payment_service):
		"""Test multi-currency payment processing"""
		currencies = ["USD", "EUR", "GBP", "KES"]
		
		for currency in currencies:
			transaction = PaymentTransaction(
				id=uuid7str(),
				amount=10000,  # Amount in cents/minor units
				currency=currency,
				payment_method_type=PaymentMethodType.CREDIT_CARD,
				merchant_id="multi_currency_merchant",
				customer_id="multi_currency_customer",
				description=f"Test {currency} payment",
				status=PaymentStatus.PENDING
			)
			
			payment_method = PaymentMethod(
				id=uuid7str(),
				type=PaymentMethodType.CREDIT_CARD,
				details={"last4": "4242", "brand": "visa"},
				customer_id="multi_currency_customer",
				token="pm_test_card"
			)
			
			# Process payment
			result = await payment_service.process_payment(
				transaction,
				payment_method,
				{"test_mode": True}
			)
			
			assert isinstance(result, PaymentResult)
			# Should handle all major currencies
			assert result.status in [
				PaymentStatus.COMPLETED,
				PaymentStatus.PENDING,
				PaymentStatus.PROCESSING
			]
	
	async def test_error_handling(self, payment_service):
		"""Test error handling in payment processing"""
		# Test with invalid transaction data
		invalid_transaction = PaymentTransaction(
			id="invalid_id",  # Invalid UUID format
			amount=-1000,     # Negative amount
			currency="INVALID",
			payment_method_type=PaymentMethodType.MPESA,
			merchant_id="",   # Empty merchant ID
			customer_id=None,
			description="Invalid transaction test",
			status=PaymentStatus.PENDING
		)
		
		invalid_payment_method = PaymentMethod(
			id=uuid7str(),
			type=PaymentMethodType.MPESA,  
			details={},  # Missing required details
			customer_id=None
		)
		
		# Process invalid payment
		result = await payment_service.process_payment(
			invalid_transaction,
			invalid_payment_method,
			{"test_mode": True}
		)
		
		# Should handle errors gracefully
		assert isinstance(result, PaymentResult)
		assert result.success is False
		assert result.error_message is not None
		assert result.status == PaymentStatus.FAILED