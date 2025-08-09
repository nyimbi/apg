"""
Comprehensive Test Suite for Adyen Integration - APG Payment Gateway

Tests covering all Adyen features including cards, digital wallets, and marketplace payments.
Production-ready tests with complete coverage.

© 2025 Datacraft. All rights reserved.
"""

import pytest
import json
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from ..adyen_integration import (
	AdyenService, AdyenEnvironment, create_adyen_service
)
from ..models import (
	PaymentTransaction, PaymentMethod, PaymentResult,
	PaymentStatus, PaymentMethodType, HealthStatus
)


class TestAdyenIntegration:
	"""Comprehensive test suite for Adyen integration"""
	
	@pytest.fixture
	async def adyen_service(self):
		"""Create Adyen service for testing"""
		with patch.dict('os.environ', {
			'ADYEN_API_KEY': 'AQE1hmfuXNWTK0Qc+iSS4...',
			'ADYEN_MERCHANT_ACCOUNT': 'TestMerchant',
			'ADYEN_CLIENT_KEY': 'test_client_key',
			'ADYEN_HMAC_KEY': 'mock_hmac_key'
		}):
			service = await create_adyen_service(AdyenEnvironment.TEST)
			return service
	
	@pytest.fixture
	def sample_transaction(self):
		"""Create sample payment transaction"""
		return PaymentTransaction(
			id="txn_adyen_test",
			amount=Decimal("2500.00"),
			currency="EUR",
			description="Adyen test payment",
			customer_email="adyen@example.com",
			customer_name="Adyen Test Customer"
		)
	
	@pytest.fixture
	def sample_card_method(self):
		"""Create sample card payment method"""
		return PaymentMethod(
			method_type=PaymentMethodType.CARD,
			metadata={
				"card_number": "4111111111111111",
				"exp_month": "03",
				"exp_year": "2026",
				"cvc": "737",
				"cardholder_name": "Test Customer"
			}
		)
	
	@pytest.fixture
	def sample_sepa_method(self):
		"""Create sample SEPA payment method"""
		return PaymentMethod(
			method_type=PaymentMethodType.BANK_TRANSFER,
			metadata={
				"iban": "NL13TEST0123456789",
				"account_holder_name": "Test Customer"
			}
		)
	
	async def test_adyen_service_initialization(self, adyen_service):
		"""Test Adyen service initialization"""
		assert adyen_service._initialized is True
		assert adyen_service.config.environment == AdyenEnvironment.TEST
		assert adyen_service.config.api_key.startswith("AQE1")
	
	@patch('Adyen.checkout.Checkout.payments')
	async def test_process_card_payment_success(self, mock_adyen_payments, adyen_service, sample_transaction, sample_card_method):
		"""Test successful card payment processing"""
		# Mock Adyen payments response
		mock_adyen_payments.return_value = {
			"resultCode": "Authorised",
			"pspReference": "ADY_123456789",
			"amount": {"value": 2500, "currency": "EUR"},
			"merchantReference": "txn_adyen_test",
			"paymentMethod": {"type": "scheme", "brand": "visa"}
		}
		
		result = await adyen_service.process_payment(sample_transaction, sample_card_method)
		
		assert result.success is True
		assert result.status == PaymentStatus.COMPLETED
		assert result.provider_transaction_id == "ADY_123456789"
		assert result.amount == Decimal("2500.00")
		assert result.currency == "EUR"
	
	@patch('Adyen.checkout.Checkout.payments')
	async def test_process_payment_requires_action(self, mock_adyen_payments, adyen_service, sample_transaction, sample_card_method):
		"""Test payment requiring 3D Secure authentication"""
		# Mock Adyen 3DS response
		mock_adyen_payments.return_value = {
			"resultCode": "RedirectShopper",
			"pspReference": "ADY_3DS_123",
			"amount": {"value": 2500, "currency": "EUR"},
			"merchantReference": "txn_adyen_test",
			"action": {
				"type": "redirect",
				"url": "https://test.adyen.com/hpp/3d/validate.shtml",
				"method": "POST",
				"paymentData": "Ab02b4c0!BQABAgB..."
			}
		}
		
		result = await adyen_service.process_payment(sample_transaction, sample_card_method)
		
		assert result.success is True
		assert result.status == PaymentStatus.PENDING
		assert result.provider_transaction_id == "ADY_3DS_123"
		assert result.payment_url is not None
		assert "redirect" in result.raw_response
	
	@patch('Adyen.checkout.Checkout.payments')
	async def test_process_sepa_payment(self, mock_adyen_payments, adyen_service, sample_transaction, sample_sepa_method):
		"""Test SEPA direct debit payment"""
		# Mock Adyen SEPA response
		mock_adyen_payments.return_value = {
			"resultCode": "Received",
			"pspReference": "ADY_SEPA_123",
			"amount": {"value": 2500, "currency": "EUR"},
			"merchantReference": "txn_adyen_test",
			"paymentMethod": {"type": "sepadirectdebit"}
		}
		
		result = await adyen_service.process_payment(sample_transaction, sample_sepa_method)
		
		assert result.success is True
		assert result.status == PaymentStatus.PENDING
		assert result.provider_transaction_id == "ADY_SEPA_123"
	
	@patch('Adyen.checkout.Checkout.payments')
	async def test_process_payment_declined(self, mock_adyen_payments, adyen_service, sample_transaction, sample_card_method):
		"""Test declined payment"""
		# Mock Adyen decline response
		mock_adyen_payments.return_value = {
			"resultCode": "Refused",
			"pspReference": "ADY_DECLINED_123",
			"refusalReason": "Insufficient funds",
			"refusalReasonCode": "5"
		}
		
		result = await adyen_service.process_payment(sample_transaction, sample_card_method)
		
		assert result.success is False
		assert result.status == PaymentStatus.FAILED
		assert "insufficient funds" in result.error_message.lower()
	
	@patch('Adyen.management.Management.payment_details')
	async def test_verify_payment_success(self, mock_adyen_details, adyen_service):
		"""Test successful payment verification"""
		# Mock Adyen payment details response
		mock_adyen_details.return_value = {
			"pspReference": "ADY_VERIFY_123",
			"status": "Settled",
			"amount": {"value": 1500, "currency": "EUR"},
			"merchantReference": "txn_verify_test"
		}
		
		result = await adyen_service.verify_payment("ADY_VERIFY_123")
		
		assert result.success is True
		assert result.status == PaymentStatus.COMPLETED
		assert result.provider_transaction_id == "ADY_VERIFY_123"
		assert result.amount == Decimal("1500.00")
	
	@patch('Adyen.checkout.Checkout.payments_reversals')
	async def test_refund_payment_success(self, mock_adyen_refund, adyen_service):
		"""Test successful payment refund"""
		# Mock Adyen refund response
		mock_adyen_refund.return_value = {
			"status": "received",
			"pspReference": "ADY_REFUND_123",
			"response": "[refund-received]"
		}
		
		result = await adyen_service.refund_payment(
			transaction_id="ADY_ORIGINAL_123",
			amount=Decimal("500.00"),
			reason="Customer request"
		)
		
		assert result.success is True
		assert result.status == PaymentStatus.REFUNDED
		assert result.provider_transaction_id == "ADY_REFUND_123"
		assert result.amount == Decimal("500.00")
	
	async def test_webhook_signature_validation(self, adyen_service):
		"""Test webhook signature validation"""
		payload = json.dumps({
			"live": "false",
			"notificationItems": [{
				"NotificationRequestItem": {
					"eventCode": "AUTHORISATION",
					"success": "true",
					"pspReference": "ADY_WEBHOOK_123",
					"merchantReference": "txn_webhook_test",
					"amount": {"value": 2000, "currency": "EUR"}
				}
			}]
		})
		
		# Test with invalid signature
		result = await adyen_service.validate_callback_signature(
			payload=payload,
			signature="invalid_signature"
		)
		assert result is False
	
	async def test_process_webhook_authorisation(self, adyen_service):
		"""Test webhook processing for authorization"""
		webhook_data = {
			"live": "false",
			"notificationItems": [{
				"NotificationRequestItem": {
					"eventCode": "AUTHORISATION",
					"success": "true",
					"pspReference": "ADY_WEBHOOK_AUTH",
					"merchantReference": "txn_webhook_test",
					"amount": {"value": 3000, "currency": "EUR"},
					"paymentMethod": "visa"
				}
			}]
		}
		
		result = await adyen_service.process_webhook(webhook_data)
		
		assert result["success"] is True
		assert result["event_type"] == "AUTHORISATION"
		assert result["transaction_id"] == "ADY_WEBHOOK_AUTH"
		assert result["status"] == "completed"
	
	async def test_process_webhook_refund(self, adyen_service):
		"""Test webhook processing for refund"""
		webhook_data = {
			"live": "false",
			"notificationItems": [{
				"NotificationRequestItem": {
					"eventCode": "REFUND",
					"success": "true",
					"pspReference": "ADY_WEBHOOK_REFUND",
					"originalReference": "ADY_ORIGINAL_123",
					"amount": {"value": 1000, "currency": "EUR"}
				}
			}]
		}
		
		result = await adyen_service.process_webhook(webhook_data)
		
		assert result["success"] is True
		assert result["event_type"] == "REFUND"
		assert result["transaction_id"] == "ADY_WEBHOOK_REFUND"
		assert result["status"] == "refunded"
	
	@patch('Adyen.management.Management.me')
	async def test_health_check_success(self, mock_adyen_me, adyen_service):
		"""Test successful health check"""
		# Mock Adyen account info response
		mock_adyen_me.return_value = {
			"id": "TestMerchant",
			"name": "Test Merchant Account",
			"status": "Active"
		}
		
		result = await adyen_service.health_check()
		
		assert result.status == HealthStatus.HEALTHY
		assert result.response_time_ms > 0
		assert "account_id" in result.details
	
	async def test_get_supported_payment_methods_europe(self, adyen_service):
		"""Test getting supported payment methods for Europe"""
		methods = await adyen_service.get_supported_payment_methods(
			country_code="NL",
			currency="EUR"
		)
		
		assert len(methods) > 0
		assert any(method["type"] == "scheme" for method in methods)
		assert any(method["type"] == "sepadirectdebit" for method in methods)
		assert any(method["type"] == "ideal" for method in methods)
	
	async def test_get_supported_payment_methods_africa(self, adyen_service):
		"""Test getting supported payment methods for Africa"""
		methods = await adyen_service.get_supported_payment_methods(
			country_code="KE",
			currency="KES"
		)
		
		assert len(methods) > 0
		assert any(method["type"] == "scheme" for method in methods)
		assert any(method["type"] == "mobilepay" for method in methods)
	
	async def test_get_transaction_fees(self, adyen_service):
		"""Test transaction fee calculation"""
		fees = await adyen_service.get_transaction_fees(
			amount=Decimal("1000.00"),
			currency="EUR",
			payment_method="scheme"
		)
		
		assert "total_fee" in fees
		assert "interchange_fee" in fees
		assert "scheme_fee" in fees
		assert "processing_fee" in fees
		assert float(fees["total_fee"]) > 0
	
	@patch('Adyen.checkout.Checkout.payment_links')
	async def test_create_payment_link(self, mock_adyen_link, adyen_service, sample_transaction):
		"""Test payment link creation"""
		# Mock Adyen payment link response
		mock_adyen_link.return_value = {
			"id": "PL_ADYEN_TEST_123",
			"url": "https://test.adyen.link/PL_ADYEN_TEST_123",
			"status": "active",
			"expiresAt": "2025-12-31T23:59:59Z"
		}
		
		payment_url = await adyen_service.create_payment_link(
			transaction=sample_transaction,
			expiry_hours=24
		)
		
		assert payment_url == "https://test.adyen.link/PL_ADYEN_TEST_123"
	
	async def test_marketplace_payment_processing(self, adyen_service):
		"""Test marketplace payment with splits"""
		transaction = PaymentTransaction(
			id="marketplace_test",
			amount=Decimal("10000.00"),
			currency="EUR",
			description="Marketplace payment"
		)
		
		payment_method = PaymentMethod(
			method_type=PaymentMethodType.CARD,
			metadata={"card_number": "4111111111111111"}
		)
		
		# Mock marketplace payment with splits
		with patch('Adyen.checkout.Checkout.payments') as mock_payments:
			mock_payments.return_value = {
				"resultCode": "Authorised",
				"pspReference": "ADY_MARKETPLACE_123",
				"amount": {"value": 10000, "currency": "EUR"},
				"splits": [
					{"account": "platform_account", "amount": {"value": 1000}},
					{"account": "merchant_account", "amount": {"value": 9000}}
				]
			}
			
			result = await adyen_service.process_marketplace_payment(
				transaction=transaction,
				payment_method=payment_method,
				splits=[
					{"account": "platform_account", "amount": Decimal("1000.00")},
					{"account": "merchant_account", "amount": Decimal("9000.00")}
				]
			)
			
			assert result.success is True
			assert result.status == PaymentStatus.COMPLETED
			assert "splits" in result.raw_response
	
	async def test_recurring_payment_setup(self, adyen_service):
		"""Test recurring payment setup"""
		transaction = PaymentTransaction(
			id="recurring_setup",
			amount=Decimal("0.00"),  # Zero amount for setup
			currency="EUR",
			description="Recurring setup"
		)
		
		payment_method = PaymentMethod(
			method_type=PaymentMethodType.CARD,
			metadata={"card_number": "4111111111111111"}
		)
		
		with patch('Adyen.checkout.Checkout.payments') as mock_payments:
			mock_payments.return_value = {
				"resultCode": "Authorised",
				"pspReference": "ADY_RECURRING_SETUP",
				"recurringDetailReference": "REC_123456789"
			}
			
			result = await adyen_service.setup_recurring_payment(
				transaction=transaction,
				payment_method=payment_method,
				customer_id="customer_123"
			)
			
			assert result["success"] is True
			assert result["recurring_reference"] == "REC_123456789"
	
	async def test_digital_wallet_payment(self, adyen_service):
		"""Test digital wallet payment (Apple Pay/Google Pay)"""
		transaction = PaymentTransaction(
			id="wallet_test",
			amount=Decimal("5000.00"),
			currency="USD",
			description="Apple Pay payment"
		)
		
		payment_method = PaymentMethod(
			method_type=PaymentMethodType.DIGITAL_WALLET,
			metadata={
				"type": "applepay",
				"applepay.token": "mock_apple_pay_token"
			}
		)
		
		with patch('Adyen.checkout.Checkout.payments') as mock_payments:
			mock_payments.return_value = {
				"resultCode": "Authorised",
				"pspReference": "ADY_APPLEPAY_123",
				"amount": {"value": 5000, "currency": "USD"},
				"paymentMethod": {"type": "applepay"}
			}
			
			result = await adyen_service.process_payment(transaction, payment_method)
			
			assert result.success is True
			assert result.status == PaymentStatus.COMPLETED
			assert result.provider_transaction_id == "ADY_APPLEPAY_123"


class TestAdyenErrorHandling:
	"""Test error handling scenarios for Adyen integration"""
	
	@pytest.fixture
	async def adyen_service(self):
		"""Create Adyen service for error testing"""
		with patch.dict('os.environ', {
			'ADYEN_API_KEY': 'AQE1error_key',
			'ADYEN_MERCHANT_ACCOUNT': 'ErrorMerchant'
		}):
			service = await create_adyen_service(AdyenEnvironment.TEST)
			return service
	
	async def test_api_connection_error(self, adyen_service):
		"""Test API connection error handling"""
		from Adyen.exceptions import AdyenError
		
		with patch('Adyen.checkout.Checkout.payments') as mock_payments:
			mock_payments.side_effect = AdyenError("Connection failed")
			
			transaction = PaymentTransaction(
				id="error_test",
				amount=Decimal("100.00"),
				currency="EUR",
				description="Error test"
			)
			
			payment_method = PaymentMethod(
				method_type=PaymentMethodType.CARD,
				metadata={"card_number": "4111111111111111"}
			)
			
			result = await adyen_service.process_payment(transaction, payment_method)
			
			assert result.success is False
			assert result.status == PaymentStatus.FAILED
			assert "connection" in result.error_message.lower()
	
	async def test_invalid_merchant_account(self, adyen_service):
		"""Test invalid merchant account error"""
		with patch('Adyen.checkout.Checkout.payments') as mock_payments:
			mock_payments.return_value = {
				"status": 422,
				"errorCode": "702",
				"message": "Invalid Merchant Account",
				"errorType": "validation"
			}
			
			transaction = PaymentTransaction(
				id="invalid_merchant_test",
				amount=Decimal("100.00"),
				currency="EUR",
				description="Invalid merchant test"
			)
			
			payment_method = PaymentMethod(
				method_type=PaymentMethodType.CARD,
				metadata={"card_number": "4111111111111111"}
			)
			
			result = await adyen_service.process_payment(transaction, payment_method)
			
			assert result.success is False
			assert result.status == PaymentStatus.FAILED
			assert "merchant account" in result.error_message.lower()
	
	async def test_blocked_card_error(self, adyen_service):
		"""Test blocked card error handling"""
		with patch('Adyen.checkout.Checkout.payments') as mock_payments:
			mock_payments.return_value = {
				"resultCode": "Refused",
				"pspReference": "ADY_BLOCKED_123",
				"refusalReason": "BLOCKED CARD",
				"refusalReasonCode": "7"
			}
			
			transaction = PaymentTransaction(
				id="blocked_card_test",
				amount=Decimal("100.00"),
				currency="EUR",
				description="Blocked card test"
			)
			
			payment_method = PaymentMethod(
				method_type=PaymentMethodType.CARD,
				metadata={"card_number": "4000000000000002"}  # Blocked test card
			)
			
			result = await adyen_service.process_payment(transaction, payment_method)
			
			assert result.success is False
			assert result.status == PaymentStatus.FAILED
			assert "blocked" in result.error_message.lower()


if __name__ == "__main__":
	pytest.main([__file__, "-v", "--cov=../adyen_integration", "--cov-report=html"])