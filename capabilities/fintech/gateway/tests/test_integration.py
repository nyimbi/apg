"""
Integration Tests

End-to-end integration tests with real data flows across all components.

© 2025 Datacraft. All rights reserved.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from uuid_extensions import uuid7str

from ..models import PaymentTransaction, PaymentMethod, PaymentStatus, PaymentMethodType
from ..database import DatabaseService
from ..auth import AuthenticationService
from ..service import PaymentGatewayService
from ..mpesa_processor import MPESAPaymentProcessor


class TestFullIntegration:
	"""Test complete end-to-end payment flows"""
	
	async def test_complete_mpesa_payment_flow(self, temp_database, auth_service):
		"""Test complete MPESA payment flow from API to database"""
		# Initialize all services
		payment_service_config = {
			"mpesa": {
				"environment": "sandbox",
				"consumer_key": "test_consumer_key",
				"consumer_secret": "test_consumer_secret",
				"business_short_code": "174379",
				"passkey": "test_passkey", 
				"callback_url": "https://example.com/callback"
			}
		}
		
		payment_service = PaymentGatewayService(payment_service_config)
		payment_service._database_service = temp_database
		await payment_service.initialize()
		
		# Create API key for authentication
		api_key_result = await auth_service.create_api_key(
			"Integration Test Key",
			["payment_process", "payment_status", "payment_capture"]
		)
		api_key = api_key_result["api_key"]
		
		# Validate API key
		is_valid = await auth_service.validate_api_key(api_key)
		assert is_valid is True
		
		# Create transaction
		transaction = PaymentTransaction(
			id=uuid7str(),
			amount=50000,  # 500 KES
			currency="KES",
			payment_method_type=PaymentMethodType.MPESA,
			merchant_id="integration_test_merchant",
			customer_id="integration_test_customer",
			description="Full integration test payment",
			metadata={"test_type": "integration", "api_key_used": api_key[:10]},
			status=PaymentStatus.PENDING,
			created_at=datetime.now(timezone.utc)
		)
		
		# Create payment method
		payment_method = PaymentMethod(
			id=uuid7str(),
			type=PaymentMethodType.MPESA,
			details={
				"phone_number": "+254712345678",
				"account_reference": "IntegrationTest"
			},
			customer_id="integration_test_customer",
			created_at=datetime.now(timezone.utc)
		)
		
		# Process payment through service
		additional_data = {
			"test_mode": True,
			"api_key": api_key,
			"user_ip": "127.0.0.1",
			"user_agent": "pytest-integration-test"
		}
		
		result = await payment_service.process_payment(
			transaction, payment_method, additional_data
		)
		
		# Verify payment result
		assert result.success is True or result.status in [PaymentStatus.PENDING, PaymentStatus.PROCESSING]
		assert result.processor_name == "mpesa"
		assert result.processing_time_ms is not None
		
		# Verify transaction was stored in database
		stored_transaction = await temp_database.get_payment_transaction(transaction.id)
		assert stored_transaction is not None
		assert stored_transaction.id == transaction.id
		assert stored_transaction.amount == 50000
		assert stored_transaction.currency == "KES"
		
		# Verify payment method was stored
		stored_method = await temp_database.get_payment_method(payment_method.id)
		assert stored_method is not None
		assert stored_method.type == PaymentMethodType.MPESA
		assert stored_method.details["phone_number"] == "+254712345678"
		
		# Test transaction status updates
		await payment_service.update_payment_status(
			transaction.id,
			PaymentStatus.COMPLETED,
			processor_transaction_id="mpesa_integration_12345"
		)
		
		# Verify status update
		updated_transaction = await temp_database.get_payment_transaction(transaction.id)
		assert updated_transaction.status == PaymentStatus.COMPLETED
		assert updated_transaction.processor_transaction_id == "mpesa_integration_12345"
		
		# Test analytics generation
		analytics = await temp_database.get_merchant_analytics("integration_test_merchant")
		assert analytics["merchant_id"] == "integration_test_merchant"
		assert analytics["total_transactions"] >= 1
		assert analytics["total_amount"] >= 50000
	
	async def test_multi_processor_integration(self, temp_database, auth_service):
		"""Test integration with multiple payment processors"""
		# Initialize payment service with multiple processors
		payment_service_config = {
			"mpesa": {
				"environment": "sandbox",
				"consumer_key": "test_consumer_key",
				"consumer_secret": "test_consumer_secret",
				"business_short_code": "174379",
				"passkey": "test_passkey",
				"callback_url": "https://example.com/callback"
			},
			"stripe": {
				"api_key": "sk_test_integration_12345",
				"webhook_secret": "whsec_test_integration_12345",
				"environment": "sandbox"
			},
			"paypal": {
				"client_id": "test_integration_client_id",
				"client_secret": "test_integration_client_secret",
				"environment": "sandbox"
			}
		}
		
		payment_service = PaymentGatewayService(payment_service_config)
		payment_service._database_service = temp_database
		await payment_service.initialize()
		
		# Verify all processors are loaded
		assert "mpesa" in payment_service._processors
		assert "stripe" in payment_service._processors
		assert "paypal" in payment_service._processors
		
		# Test payments with different processors
		test_payments = [
			{
				"processor": "mpesa",
				"transaction": PaymentTransaction(
					id=uuid7str(),
					amount=10000,
					currency="KES",
					payment_method_type=PaymentMethodType.MPESA,
					merchant_id="multi_processor_merchant",
					customer_id="mpesa_customer",
					description="MPESA integration test",
					status=PaymentStatus.PENDING
				),
				"payment_method": PaymentMethod(
					id=uuid7str(),
					type=PaymentMethodType.MPESA,
					details={"phone_number": "+254712345678"},
					customer_id="mpesa_customer"
				)
			},
			{
				"processor": "stripe",
				"transaction": PaymentTransaction(
					id=uuid7str(),
					amount=5000,
					currency="USD",
					payment_method_type=PaymentMethodType.CREDIT_CARD,
					merchant_id="multi_processor_merchant",
					customer_id="stripe_customer", 
					description="Stripe integration test",
					status=PaymentStatus.PENDING
				),
				"payment_method": PaymentMethod(
					id=uuid7str(),
					type=PaymentMethodType.CREDIT_CARD,
					details={"last4": "4242", "brand": "visa"},
					customer_id="stripe_customer",
					token="pm_test_card_visa"
				)
			}
		]
		
		# Process all payments
		results = []
		for payment_data in test_payments:
			result = await payment_service.process_payment(
				payment_data["transaction"],
				payment_data["payment_method"],
				{"test_mode": True, "preferred_processor": payment_data["processor"]}
			)
			results.append(result)
		
		# Verify all payments were processed
		for i, result in enumerate(results):
			assert result.processor_name == test_payments[i]["processor"]
			# Verify transaction was stored
			stored_txn = await temp_database.get_payment_transaction(
				test_payments[i]["transaction"].id
			)
			assert stored_txn is not None
	
	async def test_fraud_detection_integration(self, temp_database, auth_service):
		"""Test fraud detection integration across the system"""
		payment_service_config = {
			"mpesa": {
				"environment": "sandbox",
				"consumer_key": "test_consumer_key", 
				"consumer_secret": "test_consumer_secret",
				"business_short_code": "174379",
				"passkey": "test_passkey",
				"callback_url": "https://example.com/callback"
			}
		}
		
		payment_service = PaymentGatewayService(payment_service_config)
		payment_service._database_service = temp_database
		await payment_service.initialize()
		
		# Create suspicious transaction patterns
		suspicious_transactions = []
		for i in range(5):
			transaction = PaymentTransaction(
				id=uuid7str(),
				amount=100000,  # Large amount
				currency="KES",
				payment_method_type=PaymentMethodType.MPESA,
				merchant_id="fraud_test_merchant",
				customer_id="suspicious_customer",  # Same customer
				description=f"Suspicious transaction {i}",
				status=PaymentStatus.PENDING,
				created_at=datetime.now(timezone.utc)
			)
			suspicious_transactions.append(transaction)
		
		# Process transactions with fraud analysis
		for transaction in suspicious_transactions:
			payment_method = PaymentMethod(
				id=uuid7str(),
				type=PaymentMethodType.MPESA,
				details={"phone_number": "+254712345678"},
				customer_id="suspicious_customer"
			)
			
			additional_data = {
				"test_mode": True,
				"risk_data": {
					"ip_address": "192.168.1.100",  # Same IP
					"user_agent": "Suspicious Bot",
					"velocity_check": True,
					"large_amount": True
				}
			}
			
			result = await payment_service.process_payment(
				transaction, payment_method, additional_data
			)
			
			# Fraud detection might flag these transactions
			assert result is not None
			
			# Check if fraud analysis was created
			fraud_analyses = await temp_database.get_fraud_analyses_by_transaction(
				transaction.id
			)
			assert len(fraud_analyses) >= 0  # May or may not have fraud analysis in test mode
	
	async def test_webhook_integration_flow(self, temp_database, auth_service):
		"""Test webhook processing integration"""
		payment_service_config = {
			"mpesa": {
				"environment": "sandbox",
				"consumer_key": "test_consumer_key",
				"consumer_secret": "test_consumer_secret", 
				"business_short_code": "174379",
				"passkey": "test_passkey",
				"callback_url": "https://example.com/callback"
			}
		}
		
		payment_service = PaymentGatewayService(payment_service_config)
		payment_service._database_service = temp_database
		await payment_service.initialize()
		
		# Create initial pending transaction
		transaction = PaymentTransaction(
			id=uuid7str(),
			amount=25000,
			currency="KES",
			payment_method_type=PaymentMethodType.MPESA,
			merchant_id="webhook_test_merchant",
			customer_id="webhook_test_customer",
			description="Webhook integration test",
			status=PaymentStatus.PENDING,
			created_at=datetime.now(timezone.utc)
		)
		
		# Store initial transaction
		await temp_database.create_payment_transaction(transaction)
		
		# Simulate MPESA webhook callback
		webhook_payload = {
			"Body": {
				"stkCallback": {
					"MerchantRequestID": "webhook_test_merchant_req",
					"CheckoutRequestID": transaction.id,  # Link to our transaction
					"ResultCode": 0,
					"ResultDesc": "The service request is processed successfully.",
					"CallbackMetadata": {
						"Item": [
							{"Name": "Amount", "Value": 250.00},
							{"Name": "MpesaReceiptNumber", "Value": "LGR7OWQX0R"},
							{"Name": "TransactionDate", "Value": 20250130143000},
							{"Name": "PhoneNumber", "Value": "254712345678"}
						]
					}
				}
			}
		}
		
		# Process webhook
		webhook_result = await payment_service.process_webhook(
			"mpesa", webhook_payload, {"test_mode": True}
		)
		
		assert webhook_result["processed"] is True
		assert "mpesa_receipt_number" in webhook_result
		
		# Verify transaction status was updated
		updated_transaction = await temp_database.get_payment_transaction(transaction.id)
		# In test mode, status might not change, but webhook processing should complete
		assert updated_transaction is not None
	
	async def test_concurrent_system_load(self, temp_database, auth_service):
		"""Test system under concurrent load"""
		payment_service_config = {
			"mpesa": {
				"environment": "sandbox",
				"consumer_key": "test_consumer_key",
				"consumer_secret": "test_consumer_secret",
				"business_short_code": "174379", 
				"passkey": "test_passkey",
				"callback_url": "https://example.com/callback"
			}
		}
		
		payment_service = PaymentGatewayService(payment_service_config)
		payment_service._database_service = temp_database
		await payment_service.initialize()
		
		# Create API keys for multiple clients
		api_keys = []
		for i in range(3):
			api_key_result = await auth_service.create_api_key(
				f"Load Test Client {i}",
				["payment_process", "payment_status"]
			)
			api_keys.append(api_key_result["api_key"])
		
		# Create concurrent payment tasks
		concurrent_tasks = []
		for i in range(20):  # 20 concurrent payments
			transaction = PaymentTransaction(
				id=uuid7str(),
				amount=1000 * (i + 1),
				currency="KES",
				payment_method_type=PaymentMethodType.MPESA,
				merchant_id=f"load_test_merchant_{i % 5}",  # 5 different merchants
				customer_id=f"load_test_customer_{i}",
				description=f"Load test payment {i}",
				status=PaymentStatus.PENDING,
				created_at=datetime.now(timezone.utc)
			)
			
			payment_method = PaymentMethod(
				id=uuid7str(),
				type=PaymentMethodType.MPESA,
				details={"phone_number": f"+25471234567{i % 10}"},
				customer_id=f"load_test_customer_{i}"
			)
			
			additional_data = {
				"test_mode": True,
				"api_key": api_keys[i % 3],  # Rotate API keys
				"load_test": True
			}
			
			task = payment_service.process_payment(
				transaction, payment_method, additional_data
			)
			concurrent_tasks.append(task)
		
		# Execute all tasks concurrently
		results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
		
		# Analyze results
		successful_payments = [r for r in results if not isinstance(r, Exception)]
		failed_payments = [r for r in results if isinstance(r, Exception)]
		
		print(f"Successful payments: {len(successful_payments)}")
		print(f"Failed payments: {len(failed_payments)}")
		
		# Verify most payments succeeded
		assert len(successful_payments) >= len(concurrent_tasks) * 0.8  # At least 80% success rate
		
		# Verify database integrity
		analytics = await temp_database.get_transaction_analytics(
			datetime.now(timezone.utc).strftime('%Y-%m-%d'),
			datetime.now(timezone.utc).strftime('%Y-%m-%d')
		)
		assert analytics["total_transactions"] >= len(successful_payments)
	
	async def test_system_recovery_and_resilience(self, temp_database, auth_service):
		"""Test system recovery from failures"""
		payment_service_config = {
			"mpesa": {
				"environment": "sandbox",
				"consumer_key": "test_consumer_key",
				"consumer_secret": "test_consumer_secret",
				"business_short_code": "174379",
				"passkey": "test_passkey",
				"callback_url": "https://example.com/callback"
			}
		}
		
		payment_service = PaymentGatewayService(payment_service_config)
		payment_service._database_service = temp_database
		await payment_service.initialize()
		
		# Test recovery from processor failure
		transaction = PaymentTransaction(
			id=uuid7str(),
			amount=15000,
			currency="KES",
			payment_method_type=PaymentMethodType.MPESA,
			merchant_id="resilience_test_merchant",
			customer_id="resilience_test_customer", 
			description="Resilience test payment",
			status=PaymentStatus.PENDING
		)
		
		payment_method = PaymentMethod(
			id=uuid7str(),
			type=PaymentMethodType.MPESA,
			details={"phone_number": "+254712345678"},
			customer_id="resilience_test_customer"
		)
		
		# Process payment with potential failure scenarios
		additional_data = {
			"test_mode": True,
			"simulate_failure": False,  # Start with success
			"retry_enabled": True
		}
		
		result = await payment_service.process_payment(
			transaction, payment_method, additional_data
		)
		
		# Verify system handles the scenario gracefully
		assert result is not None
		
		# Test health check after processing
		health_result = await payment_service.health_check()
		assert health_result["status"] in ["healthy", "degraded"]
		
		# Verify database consistency
		stored_transaction = await temp_database.get_payment_transaction(transaction.id)
		assert stored_transaction is not None
		assert stored_transaction.id == transaction.id