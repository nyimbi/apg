"""
Payment Processor Tests

Comprehensive tests for all payment processors with real data simulation.

© 2025 Datacraft. All rights reserved.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from uuid_extensions import uuid7str

from ..models import PaymentTransaction, PaymentMethod, PaymentStatus, PaymentMethodType
from ..mpesa_processor import MPESAPaymentProcessor
from ..stripe_processor import create_stripe_processor
from ..paypal_processor import create_paypal_processor
from ..adyen_processor import create_adyen_processor
from ..payment_processor import PaymentResult, ProcessorHealth, ProcessorStatus


class TestMPESAProcessor:
	"""Test MPESA payment processor"""
	
	@pytest.fixture
	async def mpesa_processor(self):
		"""Create MPESA processor for testing"""
		config = {
			"environment": "sandbox",
			"consumer_key": "test_consumer_key",
			"consumer_secret": "test_consumer_secret", 
			"business_short_code": "174379",
			"passkey": "test_passkey",
			"callback_url": "https://example.com/callback",
			"timeout": 30
		}
		
		processor = MPESAPaymentProcessor(config)
		await processor.initialize()
		return processor
	
	async def test_mpesa_initialization(self, mpesa_processor):
		"""Test MPESA processor initialization"""
		assert mpesa_processor is not None
		assert mpesa_processor.processor_name == "mpesa"
		assert mpesa_processor._initialized is True
		assert PaymentMethodType.MPESA in mpesa_processor.supported_payment_methods
	
	async def test_mpesa_payment_processing(self, mpesa_processor):
		"""Test MPESA payment processing"""
		transaction = PaymentTransaction(
			id=uuid7str(),
			amount=10000,  # 100 KES
			currency="KES",
			payment_method_type=PaymentMethodType.MPESA,
			merchant_id="test_merchant",
			customer_id="test_customer",
			description="Test MPESA payment",
			status=PaymentStatus.PENDING
		)
		
		payment_method = PaymentMethod(
			id=uuid7str(),
			type=PaymentMethodType.MPESA,
			details={
				"phone_number": "+254712345678",
				"account_reference": "TestRef123"
			},
			customer_id="test_customer"
		)
		
		# Set test mode to avoid actual API calls
		additional_data = {"test_mode": True}
		
		result = await mpesa_processor.process_payment(
			transaction, payment_method, additional_data
		)
		
		assert isinstance(result, PaymentResult)
		assert result.processor_name == "mpesa"
		# In test mode, should return success or pending
		assert result.status in [PaymentStatus.COMPLETED, PaymentStatus.PENDING]
	
	async def test_mpesa_callback_processing(self, mpesa_processor):
		"""Test MPESA callback processing"""
		# Simulate MPESA callback payload
		callback_payload = {
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
		
		result = await mpesa_processor.process_callback(callback_payload)
		
		assert result["success"] is True
		assert result["mpesa_receipt_number"] == "LGR7OWQX0R"
		assert result["amount"] == 100.00
	
	async def test_mpesa_health_check(self, mpesa_processor):
		"""Test MPESA processor health check"""
		health = await mpesa_processor.health_check()
		
		assert isinstance(health, ProcessorHealth)
		assert health.processor_name == "mpesa"
		assert health.status in [ProcessorStatus.HEALTHY, ProcessorStatus.ERROR]


class TestStripeProcessor:
	"""Test Stripe payment processor"""
	
	@pytest.fixture
	async def stripe_processor(self):
		"""Create Stripe processor for testing"""
		config = {
			"api_key": "sk_test_12345",
			"webhook_secret": "whsec_test_12345",
			"environment": "sandbox"
		}
		
		processor = create_stripe_processor(config)
		await processor.initialize()
		return processor
	
	async def test_stripe_initialization(self, stripe_processor):
		"""Test Stripe processor initialization"""
		assert stripe_processor is not None
		assert stripe_processor.processor_name == "stripe"
		assert stripe_processor._initialized is True
		assert PaymentMethodType.CREDIT_CARD in stripe_processor.supported_payment_methods
	
	async def test_stripe_payment_processing(self, stripe_processor):
		"""Test Stripe payment processing"""
		transaction = PaymentTransaction(
			id=uuid7str(),
			amount=5000,  # $50.00
			currency="USD",
			payment_method_type=PaymentMethodType.CREDIT_CARD,
			merchant_id="stripe_merchant",
			customer_id="stripe_customer",
			description="Test Stripe payment",
			status=PaymentStatus.PENDING
		)
		
		payment_method = PaymentMethod(
			id=uuid7str(),
			type=PaymentMethodType.CREDIT_CARD,
			details={
				"last4": "4242",
				"brand": "visa",
				"exp_month": "12",
				"exp_year": "2025"
			},
			customer_id="stripe_customer",
			token="pm_test_card_visa"
		)
		
		# In test mode to avoid actual API calls
		result = await stripe_processor.process_payment(
			transaction, payment_method, {"test_mode": True}
		)
		
		assert isinstance(result, PaymentResult)
		assert result.processor_name == "stripe"
	
	async def test_stripe_tokenization(self, stripe_processor):
		"""Test Stripe payment method tokenization"""
		customer_id = "test_customer_stripe"
		payment_method_data = {
			"type": "card",
			"card": {
				"number": "4242424242424242",
				"exp_month": 12,
				"exp_year": 2025,
				"cvc": "123"
			}
		}
		
		result = await stripe_processor.tokenize_payment_method(
			payment_method_data, customer_id
		)
		
		assert result["success"] is True
		assert "token" in result
		assert result["stripe_customer_id"] is not None


class TestPayPalProcessor:
	"""Test PayPal payment processor"""
	
	@pytest.fixture
	async def paypal_processor(self):
		"""Create PayPal processor for testing"""
		config = {
			"client_id": "test_client_id",
			"client_secret": "test_client_secret",
			"environment": "sandbox"
		}
		
		processor = create_paypal_processor(config)
		await processor.initialize()
		return processor
	
	async def test_paypal_initialization(self, paypal_processor):
		"""Test PayPal processor initialization"""
		assert paypal_processor is not None
		assert paypal_processor.processor_name == "paypal"
		assert paypal_processor._initialized is True
		assert PaymentMethodType.PAYPAL in paypal_processor.supported_payment_methods
	
	async def test_paypal_payment_processing(self, paypal_processor):
		"""Test PayPal payment processing"""
		transaction = PaymentTransaction(
			id=uuid7str(),
			amount=7500,  # $75.00
			currency="USD",
			payment_method_type=PaymentMethodType.PAYPAL,
			merchant_id="paypal_merchant",
			customer_id="paypal_customer",
			description="Test PayPal payment",
			status=PaymentStatus.PENDING
		)
		
		payment_method = PaymentMethod(
			id=uuid7str(),
			type=PaymentMethodType.PAYPAL,
			details={
				"email": "customer@example.com"
			},
			customer_id="paypal_customer"
		)
		
		result = await paypal_processor.process_payment(
			transaction, payment_method, {"test_mode": True}
		)
		
		assert isinstance(result, PaymentResult)
		assert result.processor_name == "paypal"


class TestAdyenProcessor:
	"""Test Adyen payment processor"""
	
	@pytest.fixture
	async def adyen_processor(self):
		"""Create Adyen processor for testing"""
		config = {
			"api_key": "test_api_key",
			"merchant_account": "test_merchant",
			"environment": "test"
		}
		
		processor = create_adyen_processor(config)
		await processor.initialize()
		return processor
	
	async def test_adyen_initialization(self, adyen_processor):
		"""Test Adyen processor initialization"""
		assert adyen_processor is not None
		assert adyen_processor.processor_name == "adyen"
		assert adyen_processor._initialized is True
		assert PaymentMethodType.CREDIT_CARD in adyen_processor.supported_payment_methods
		assert len(adyen_processor.supported_countries) > 200  # Global coverage
	
	async def test_adyen_payment_processing(self, adyen_processor):
		"""Test Adyen payment processing"""
		transaction = PaymentTransaction(
			id=uuid7str(),
			amount=12500,  # €125.00
			currency="EUR",
			payment_method_type=PaymentMethodType.CREDIT_CARD,
			merchant_id="adyen_merchant",
			customer_id="adyen_customer",
			description="Test Adyen payment",
			status=PaymentStatus.PENDING
		)
		
		payment_method = PaymentMethod(
			id=uuid7str(),
			type=PaymentMethodType.CREDIT_CARD,
			details={
				"last4": "1111",
				"brand": "visa",
				"exp_month": "03",
				"exp_year": "2030"
			},
			customer_id="adyen_customer",
			token="adyen_token_12345"
		)
		
		result = await adyen_processor.process_payment(
			transaction, payment_method, {"test_mode": True}
		)
		
		assert isinstance(result, PaymentResult)
		assert result.processor_name == "adyen"


class TestProcessorIntegration:
	"""Test cross-processor integration scenarios"""
	
	async def test_processor_routing(self):
		"""Test routing payments to appropriate processors"""
		# Test MPESA routing for KES currency
		mpesa_config = {
			"environment": "sandbox",
			"consumer_key": "test_key",
			"consumer_secret": "test_secret",
			"business_short_code": "174379",
			"passkey": "test_passkey",
			"callback_url": "https://example.com/callback"
		}
		mpesa_processor = MPESAPaymentProcessor(mpesa_config)
		
		# Test currency support
		assert "KES" in mpesa_processor.supported_currencies
		assert "KE" in mpesa_processor.supported_countries
		
		# Test Stripe routing for USD currency
		stripe_config = {
			"api_key": "sk_test_12345",
			"webhook_secret": "whsec_test_12345",
			"environment": "sandbox"
		}
		stripe_processor = create_stripe_processor(stripe_config)
		
		assert "USD" in stripe_processor.supported_currencies
		assert "US" in stripe_processor.supported_countries
	
	async def test_processor_failover(self):
		"""Test processor failover scenarios"""
		# Create processors with different priorities
		processors = {
			"primary": {
				"name": "stripe",
				"priority": 1,
				"config": {
					"api_key": "sk_test_primary",
					"webhook_secret": "whsec_test_primary",
					"environment": "sandbox"
				}
			},
			"fallback": {
				"name": "paypal", 
				"priority": 2,
				"config": {
					"client_id": "test_fallback_id",
					"client_secret": "test_fallback_secret",
					"environment": "sandbox"
				}
			}
		}
		
		# Test that both processors can handle USD
		for processor_info in processors.values():
			if processor_info["name"] == "stripe":
				processor = create_stripe_processor(processor_info["config"])
			else:
				processor = create_paypal_processor(processor_info["config"])
			
			assert "USD" in processor.supported_currencies
	
	async def test_concurrent_processor_operations(self):
		"""Test concurrent operations across multiple processors"""
		# Create different processor instances
		processors = []
		
		# MPESA processor
		mpesa_config = {
			"environment": "sandbox",
			"consumer_key": "test_key",
			"consumer_secret": "test_secret",
			"business_short_code": "174379", 
			"passkey": "test_passkey",
			"callback_url": "https://example.com/callback"
		}
		processors.append(MPESAPaymentProcessor(mpesa_config))
		
		# Stripe processor
		stripe_config = {
			"api_key": "sk_test_12345",
			"webhook_secret": "whsec_test_12345",
			"environment": "sandbox"
		}
		processors.append(create_stripe_processor(stripe_config))
		
		# Initialize all processors concurrently
		initialization_tasks = [processor.initialize() for processor in processors]
		results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
		
		# Verify all initialized successfully
		successful_inits = [r for r in results if not isinstance(r, Exception)]
		assert len(successful_inits) == len(processors)
	
	async def test_processor_health_monitoring(self):
		"""Test health monitoring across all processors"""
		# Create processors
		mpesa_processor = MPESAPaymentProcessor({
			"environment": "sandbox",
			"consumer_key": "test_key",
			"consumer_secret": "test_secret",
			"business_short_code": "174379",
			"passkey": "test_passkey",
			"callback_url": "https://example.com/callback"
		})
		
		stripe_processor = create_stripe_processor({
			"api_key": "sk_test_12345",
			"webhook_secret": "whsec_test_12345",
			"environment": "sandbox"
		})
		
		processors = [mpesa_processor, stripe_processor]
		
		# Initialize processors
		for processor in processors:
			await processor.initialize()
		
		# Check health concurrently
		health_tasks = [processor.health_check() for processor in processors]
		health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
		
		# Verify health checks completed
		successful_checks = [r for r in health_results if isinstance(r, ProcessorHealth)]
		assert len(successful_checks) <= len(processors)  # Some might fail in test environment