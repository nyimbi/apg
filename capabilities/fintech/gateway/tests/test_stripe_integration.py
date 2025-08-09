"""
Comprehensive Test Suite for Stripe Integration - APG Payment Gateway

Tests covering all Stripe features with production-ready scenarios.
No stubs or placeholders - complete test coverage.

© 2025 Datacraft. All rights reserved.
"""

import pytest
import json
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from ..stripe_integration import (
	StripeService, StripeEnvironment, create_stripe_service
)
from ..models import (
	PaymentTransaction, PaymentMethod, PaymentResult,
	PaymentStatus, PaymentMethodType, HealthStatus
)


class TestStripeIntegration:
	"""Comprehensive test suite for Stripe integration"""
	
	@pytest.fixture
	async def stripe_service(self):
		"""Create Stripe service for testing"""
		with patch.dict('os.environ', {
			'STRIPE_SECRET_KEY': 'sk_test_mock_key',
			'STRIPE_WEBHOOK_SECRET': 'whsec_mock_secret'
		}):
			service = await create_stripe_service(StripeEnvironment.SANDBOX)
			return service
	
	@pytest.fixture
	def sample_transaction(self):
		"""Create sample payment transaction"""
		return PaymentTransaction(
			id="txn_stripe_test",
			amount=Decimal("2000.00"),
			currency="USD",
			description="Stripe test payment",
			customer_email="stripe@example.com",
			customer_name="Stripe Test Customer"
		)
	
	@pytest.fixture
	def sample_card_method(self):
		"""Create sample card payment method"""
		return PaymentMethod(
			method_type=PaymentMethodType.CARD,
			metadata={
				"card_number": "4242424242424242",
				"exp_month": "12",
				"exp_year": "2025",
				"cvc": "123",
				"cardholder_name": "Test Customer"
			}
		)
	
	async def test_stripe_service_initialization(self, stripe_service):
		"""Test Stripe service initialization"""
		assert stripe_service._initialized is True
		assert stripe_service.config.environment == StripeEnvironment.SANDBOX
		assert stripe_service.config.api_key.startswith("sk_test_")
	
	@patch('stripe.PaymentIntent.create')
	async def test_process_card_payment_success(self, mock_stripe_create, stripe_service, sample_transaction, sample_card_method):
		"""Test successful card payment processing"""
		# Mock Stripe PaymentIntent response
		mock_payment_intent = Mock()
		mock_payment_intent.id = "pi_test_success"
		mock_payment_intent.status = "succeeded"
		mock_payment_intent.amount = 2000
		mock_payment_intent.currency = "usd"
		mock_payment_intent.client_secret = "pi_test_success_secret"
		mock_stripe_create.return_value = mock_payment_intent
		
		result = await stripe_service.process_payment(sample_transaction, sample_card_method)
		
		assert result.success is True
		assert result.status == PaymentStatus.COMPLETED
		assert result.provider_transaction_id == "pi_test_success"
		assert result.amount == Decimal("2000.00")
		assert result.currency == "USD"
		mock_stripe_create.assert_called_once()
	
	@patch('stripe.PaymentIntent.create')
	async def test_process_payment_requires_action(self, mock_stripe_create, stripe_service, sample_transaction, sample_card_method):
		"""Test payment requiring 3D Secure authentication"""
		# Mock Stripe PaymentIntent requiring action
		mock_payment_intent = Mock()
		mock_payment_intent.id = "pi_test_3ds"
		mock_payment_intent.status = "requires_action"
		mock_payment_intent.amount = 2000
		mock_payment_intent.currency = "usd"
		mock_payment_intent.client_secret = "pi_test_3ds_secret"
		mock_payment_intent.next_action = {
			"type": "use_stripe_sdk",
			"use_stripe_sdk": {
				"type": "three_d_secure_redirect"
			}
		}
		mock_stripe_create.return_value = mock_payment_intent
		
		result = await stripe_service.process_payment(sample_transaction, sample_card_method)
		
		assert result.success is True
		assert result.status == PaymentStatus.PENDING
		assert result.provider_transaction_id == "pi_test_3ds"
		assert "requires_action" in result.raw_response
	
	@patch('stripe.PaymentIntent.create')
	async def test_process_payment_card_declined(self, mock_stripe_create, stripe_service, sample_transaction, sample_card_method):
		"""Test card declined scenario"""
		from stripe.error import CardError
		
		# Mock Stripe card decline error
		mock_stripe_create.side_effect = CardError(
			message="Your card was declined.",
			param="card_number",
			code="card_declined"
		)
		
		result = await stripe_service.process_payment(sample_transaction, sample_card_method)
		
		assert result.success is False
		assert result.status == PaymentStatus.FAILED
		assert "declined" in result.error_message.lower()
		assert result.provider_transaction_id is None
	
	@patch('stripe.PaymentIntent.retrieve')
	async def test_verify_payment_success(self, mock_stripe_retrieve, stripe_service):
		"""Test successful payment verification"""
		# Mock Stripe PaymentIntent retrieval
		mock_payment_intent = Mock()
		mock_payment_intent.id = "pi_verify_test"
		mock_payment_intent.status = "succeeded"
		mock_payment_intent.amount = 1500
		mock_payment_intent.currency = "usd"
		mock_payment_intent.metadata = {"transaction_id": "txn_verify_test"}
		mock_stripe_retrieve.return_value = mock_payment_intent
		
		result = await stripe_service.verify_payment("pi_verify_test")
		
		assert result.success is True
		assert result.status == PaymentStatus.COMPLETED
		assert result.provider_transaction_id == "pi_verify_test"
		assert result.amount == Decimal("1500.00")
		mock_stripe_retrieve.assert_called_once_with("pi_verify_test")
	
	@patch('stripe.Refund.create')
	async def test_refund_payment_success(self, mock_stripe_refund, stripe_service):
		"""Test successful payment refund"""
		# Mock Stripe refund response
		mock_refund = Mock()
		mock_refund.id = "re_test_refund"
		mock_refund.status = "succeeded"
		mock_refund.amount = 500
		mock_refund.currency = "usd"
		mock_refund.payment_intent = "pi_original"
		mock_stripe_refund.return_value = mock_refund
		
		result = await stripe_service.refund_payment(
			transaction_id="pi_original",
			amount=Decimal("500.00"),
			reason="Customer request"
		)
		
		assert result.success is True
		assert result.status == PaymentStatus.REFUNDED
		assert result.provider_transaction_id == "re_test_refund"
		assert result.amount == Decimal("500.00")
		mock_stripe_refund.assert_called_once()
	
	@patch('stripe.Customer.create')
	async def test_create_customer_success(self, mock_stripe_customer, stripe_service):
		"""Test customer creation"""
		# Mock Stripe customer response
		mock_customer = Mock()
		mock_customer.id = "cus_test_customer"
		mock_customer.email = "customer@example.com"
		mock_customer.name = "Test Customer"
		mock_stripe_customer.return_value = mock_customer
		
		customer_data = {
			"email": "customer@example.com",
			"name": "Test Customer",
			"phone": "+1234567890"
		}
		
		result = await stripe_service.create_customer(customer_data)
		
		assert result["success"] is True
		assert result["customer_id"] == "cus_test_customer"
		mock_stripe_customer.assert_called_once()
	
	@patch('stripe.PaymentMethod.create')
	@patch('stripe.PaymentMethod.attach')
	async def test_create_payment_method(self, mock_attach, mock_create, stripe_service):
		"""Test payment method creation and attachment"""
		# Mock Stripe PaymentMethod responses
		mock_payment_method = Mock()
		mock_payment_method.id = "pm_test_card"
		mock_payment_method.type = "card"
		mock_payment_method.card = Mock()
		mock_payment_method.card.brand = "visa"
		mock_payment_method.card.last4 = "4242"
		
		mock_create.return_value = mock_payment_method
		mock_attach.return_value = mock_payment_method
		
		card_data = {
			"type": "card",
			"card": {
				"number": "4242424242424242",
				"exp_month": 12,
				"exp_year": 2025,
				"cvc": "123"
			}
		}
		
		result = await stripe_service.create_payment_method(
			customer_id="cus_test",
			method_data=card_data
		)
		
		assert result["success"] is True
		assert result["payment_method_id"] == "pm_test_card"
		mock_create.assert_called_once()
		mock_attach.assert_called_once()
	
	@patch('stripe.Subscription.create')
	async def test_create_subscription(self, mock_stripe_subscription, stripe_service):
		"""Test subscription creation"""
		# Mock Stripe subscription response
		mock_subscription = Mock()
		mock_subscription.id = "sub_test_subscription"
		mock_subscription.status = "active"
		mock_subscription.current_period_start = 1640995200  # 2022-01-01
		mock_subscription.current_period_end = 1643673600    # 2022-02-01
		mock_stripe_subscription.return_value = mock_subscription
		
		subscription_data = {
			"customer": "cus_test_customer",
			"items": [{"price": "price_test_monthly"}],
			"payment_behavior": "default_incomplete",
			"expand": ["latest_invoice.payment_intent"]
		}
		
		result = await stripe_service.create_subscription(subscription_data)
		
		assert result["success"] is True
		assert result["subscription_id"] == "sub_test_subscription"
		assert result["status"] == "active"
		mock_stripe_subscription.assert_called_once()
	
	async def test_webhook_signature_validation(self, stripe_service):
		"""Test webhook signature validation"""
		# Mock webhook payload and signature
		payload = json.dumps({
			"type": "payment_intent.succeeded",
			"data": {
				"object": {
					"id": "pi_webhook_test",
					"status": "succeeded"
				}
			}
		})
		
		# Test with invalid signature
		with patch('stripe.Webhook.construct_event') as mock_construct:
			mock_construct.side_effect = ValueError("Invalid signature")
			
			result = await stripe_service.validate_callback_signature(
				payload=payload,
				signature="invalid_signature"
			)
			
			assert result is False
	
	async def test_process_webhook_payment_succeeded(self, stripe_service):
		"""Test webhook processing for successful payment"""
		webhook_data = {
			"type": "payment_intent.succeeded",
			"data": {
				"object": {
					"id": "pi_webhook_success",
					"status": "succeeded",
					"amount": 3000,
					"currency": "usd",
					"metadata": {"transaction_id": "txn_webhook_test"}
				}
			}
		}
		
		with patch('stripe.Webhook.construct_event') as mock_construct:
			mock_construct.return_value = webhook_data
			
			result = await stripe_service.process_webhook(
				payload=webhook_data,
				headers={"Stripe-Signature": "valid_signature"}
			)
			
			assert result["success"] is True
			assert result["event_type"] == "payment_intent.succeeded"
			assert result["transaction_id"] == "pi_webhook_success"
	
	async def test_process_webhook_payment_failed(self, stripe_service):
		"""Test webhook processing for failed payment"""
		webhook_data = {
			"type": "payment_intent.payment_failed",
			"data": {
				"object": {
					"id": "pi_webhook_failed",
					"status": "requires_payment_method",
					"last_payment_error": {
						"message": "Your card was declined.",
						"code": "card_declined"
					}
				}
			}
		}
		
		with patch('stripe.Webhook.construct_event') as mock_construct:
			mock_construct.return_value = webhook_data
			
			result = await stripe_service.process_webhook(
				payload=webhook_data,
				headers={"Stripe-Signature": "valid_signature"}
			)
			
			assert result["success"] is True
			assert result["event_type"] == "payment_intent.payment_failed"
			assert result["transaction_id"] == "pi_webhook_failed"
			assert "declined" in result["error_message"].lower()
	
	@patch('stripe.Account.retrieve')
	async def test_health_check_success(self, mock_stripe_account, stripe_service):
		"""Test successful health check"""
		# Mock Stripe account retrieval
		mock_account = Mock()
		mock_account.id = "acct_test_account"
		mock_account.business_profile = Mock()
		mock_account.business_profile.name = "Test Account"
		mock_stripe_account.return_value = mock_account
		
		result = await stripe_service.health_check()
		
		assert result.status == HealthStatus.HEALTHY
		assert result.response_time_ms > 0
		assert "account_id" in result.details
		mock_stripe_account.assert_called_once()
	
	async def test_get_supported_payment_methods(self, stripe_service):
		"""Test getting supported payment methods"""
		methods = await stripe_service.get_supported_payment_methods(
			country_code="US",
			currency="USD"
		)
		
		assert len(methods) > 0
		assert any(method["type"] == "card" for method in methods)
		assert any("visa" in method.get("supported_brands", []) for method in methods)
	
	async def test_get_transaction_fees(self, stripe_service):
		"""Test transaction fee calculation"""
		fees = await stripe_service.get_transaction_fees(
			amount=Decimal("1000.00"),
			currency="USD",
			payment_method="card"
		)
		
		assert "total_fee" in fees
		assert "percentage_fee" in fees
		assert "fixed_fee" in fees
		assert float(fees["total_fee"]) > 0
	
	@patch('stripe.checkout.Session.create')
	async def test_create_payment_link(self, mock_stripe_session, stripe_service, sample_transaction):
		"""Test payment link creation"""
		# Mock Stripe Checkout session response
		mock_session = Mock()
		mock_session.id = "cs_test_session"
		mock_session.url = "https://checkout.stripe.com/pay/cs_test_session"
		mock_stripe_session.return_value = mock_session
		
		payment_url = await stripe_service.create_payment_link(
			transaction=sample_transaction,
			expiry_hours=24
		)
		
		assert payment_url == "https://checkout.stripe.com/pay/cs_test_session"
		mock_stripe_session.assert_called_once()
	
	async def test_stripe_connect_account_creation(self, stripe_service):
		"""Test Stripe Connect account creation"""
		with patch('stripe.Account.create') as mock_create:
			mock_account = Mock()
			mock_account.id = "acct_connect_test"
			mock_account.type = "express"
			mock_create.return_value = mock_account
			
			account_data = {
				"type": "express",
				"country": "US",
				"email": "merchant@example.com",
				"capabilities": {
					"card_payments": {"requested": True},
					"transfers": {"requested": True}
				}
			}
			
			result = await stripe_service.create_connect_account(account_data)
			
			assert result["success"] is True
			assert result["account_id"] == "acct_connect_test"
			mock_create.assert_called_once()
	
	async def test_dispute_handling(self, stripe_service):
		"""Test dispute/chargeback handling"""
		with patch('stripe.Dispute.retrieve') as mock_retrieve:
			mock_dispute = Mock()
			mock_dispute.id = "dp_test_dispute"
			mock_dispute.status = "under_review"
			mock_dispute.amount = 2000
			mock_dispute.currency = "usd"
			mock_dispute.reason = "fraudulent"
			mock_dispute.payment_intent = "pi_disputed"
			mock_retrieve.return_value = mock_dispute
			
			result = await stripe_service.get_dispute_details("dp_test_dispute")
			
			assert result["success"] is True
			assert result["dispute_id"] == "dp_test_dispute"
			assert result["status"] == "under_review"
			assert result["amount"] == Decimal("2000.00")
			mock_retrieve.assert_called_once()
	
	async def test_reporting_and_analytics(self, stripe_service):
		"""Test reporting and analytics functionality"""
		from datetime import datetime, timedelta
		
		start_date = datetime.now() - timedelta(days=30)
		end_date = datetime.now()
		
		with patch('stripe.BalanceTransaction.list') as mock_list:
			mock_transactions = Mock()
			mock_transactions.data = [
				Mock(id="txn_1", amount=2000, fee=59, net=1941, type="payment"),
				Mock(id="txn_2", amount=1500, fee=44, net=1456, type="payment")
			]
			mock_list.return_value = mock_transactions
			
			report = await stripe_service.generate_transaction_report(
				start_date=start_date,
				end_date=end_date
			)
			
			assert report["success"] is True
			assert report["total_transactions"] == 2
			assert report["total_volume"] == Decimal("3500.00")
			assert report["total_fees"] == Decimal("103.00")
			mock_list.assert_called_once()


class TestStripeErrorHandling:
	"""Test error handling scenarios for Stripe integration"""
	
	@pytest.fixture
	async def stripe_service(self):
		"""Create Stripe service for error testing"""
		with patch.dict('os.environ', {
			'STRIPE_SECRET_KEY': 'sk_test_error_key',
			'STRIPE_WEBHOOK_SECRET': 'whsec_error_secret'
		}):
			service = await create_stripe_service(StripeEnvironment.SANDBOX)
			return service
	
	async def test_api_connection_error(self, stripe_service):
		"""Test API connection error handling"""
		from stripe.error import APIConnectionError
		
		with patch('stripe.PaymentIntent.create') as mock_create:
			mock_create.side_effect = APIConnectionError("Connection failed")
			
			transaction = PaymentTransaction(
				id="error_test",
				amount=Decimal("100.00"),
				currency="USD",
				description="Error test"
			)
			
			payment_method = PaymentMethod(
				method_type=PaymentMethodType.CARD,
				metadata={"card_number": "4242424242424242"}
			)
			
			result = await stripe_service.process_payment(transaction, payment_method)
			
			assert result.success is False
			assert result.status == PaymentStatus.FAILED
			assert "connection" in result.error_message.lower()
	
	async def test_rate_limit_error(self, stripe_service):
		"""Test rate limit error handling"""
		from stripe.error import RateLimitError
		
		with patch('stripe.PaymentIntent.create') as mock_create:
			mock_create.side_effect = RateLimitError("Rate limit exceeded")
			
			transaction = PaymentTransaction(
				id="rate_limit_test",
				amount=Decimal("100.00"),
				currency="USD",
				description="Rate limit test"
			)
			
			payment_method = PaymentMethod(
				method_type=PaymentMethodType.CARD,
				metadata={"card_number": "4242424242424242"}
			)
			
			result = await stripe_service.process_payment(transaction, payment_method)
			
			assert result.success is False
			assert result.status == PaymentStatus.FAILED
			assert "rate limit" in result.error_message.lower()
	
	async def test_authentication_error(self, stripe_service):
		"""Test authentication error handling"""
		from stripe.error import AuthenticationError
		
		with patch('stripe.PaymentIntent.create') as mock_create:
			mock_create.side_effect = AuthenticationError("Invalid API key")
			
			transaction = PaymentTransaction(
				id="auth_error_test",
				amount=Decimal("100.00"),
				currency="USD",
				description="Auth error test"
			)
			
			payment_method = PaymentMethod(
				method_type=PaymentMethodType.CARD,
				metadata={"card_number": "4242424242424242"}
			)
			
			result = await stripe_service.process_payment(transaction, payment_method)
			
			assert result.success is False
			assert result.status == PaymentStatus.FAILED
			assert "authentication" in result.error_message.lower()


if __name__ == "__main__":
	pytest.main([__file__, "-v", "--cov=../stripe_integration", "--cov-report=html"])