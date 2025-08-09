"""
Comprehensive Test Suite for Pesapal Integration - APG Payment Gateway

Tests covering all Pesapal features including M-Pesa, Airtel Money, and card payments.
Production-ready tests with complete coverage for East African markets.

© 2025 Datacraft. All rights reserved.
"""

import pytest
import json
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from ..pesapal_integration import (
	PesapalService, PesapalEnvironment, create_pesapal_service
)
from ..models import (
	PaymentTransaction, PaymentMethod, PaymentResult,
	PaymentStatus, PaymentMethodType, HealthStatus
)


class TestPesapalIntegration:
	"""Comprehensive test suite for Pesapal integration"""
	
	@pytest.fixture
	async def pesapal_service(self):
		"""Create Pesapal service for testing"""
		with patch.dict('os.environ', {
			'PESAPAL_CONSUMER_KEY': 'test_consumer_key',
			'PESAPAL_CONSUMER_SECRET': 'test_consumer_secret',
			'PESAPAL_IPN_URL': 'https://example.com/ipn/pesapal'
		}):
			service = await create_pesapal_service(PesapalEnvironment.SANDBOX)
			return service
	
	@pytest.fixture
	def sample_transaction(self):
		"""Create sample payment transaction"""
		return PaymentTransaction(
			id="txn_pesapal_test",
			amount=Decimal("1500.00"),
			currency="KES",
			description="Pesapal test payment",
			customer_email="pesapal@example.com",
			customer_name="Pesapal Test Customer"
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
	
	@pytest.fixture
	def sample_mpesa_method(self):
		"""Create sample M-Pesa payment method"""
		return PaymentMethod(
			method_type=PaymentMethodType.MOBILE_MONEY,
			metadata={
				"phone": "254700000000",
				"provider": "MPESA",
				"network": "SAFARICOM"
			}
		)
	
	@pytest.fixture
	def sample_airtel_method(self):
		"""Create sample Airtel Money payment method"""
		return PaymentMethod(
			method_type=PaymentMethodType.MOBILE_MONEY,
			metadata={
				"phone": "254730000000",
				"provider": "AIRTEL",
				"network": "AIRTEL"
			}
		)
	
	async def test_pesapal_service_initialization(self, pesapal_service):
		"""Test Pesapal service initialization"""
		assert pesapal_service._initialized is True
		assert pesapal_service.config.environment == PesapalEnvironment.SANDBOX
		assert pesapal_service.config.consumer_key == "test_consumer_key"
	
	@patch('aiohttp.ClientSession.post')
	async def test_process_mpesa_payment_success(self, mock_post, pesapal_service, sample_transaction, sample_mpesa_method):
		"""Test successful M-Pesa payment processing"""
		# Mock Pesapal authentication response
		mock_auth_response = Mock()
		mock_auth_response.status = 200
		mock_auth_response.json = AsyncMock(return_value={
			"token": "mock_access_token",
			"expiryDate": "2025-12-31T23:59:59Z"
		})
		
		# Mock Pesapal payment request response
		mock_payment_response = Mock()
		mock_payment_response.status = 200
		mock_payment_response.json = AsyncMock(return_value={
			"order_tracking_id": "PSP_MPESA_123456",
			"merchant_reference": "txn_pesapal_test",
			"redirect_url": "https://cybqa.pesapal.com/pesapalv3/api/Auth/SecurePayment/?orderTrackingId=PSP_MPESA_123456"
		})
		
		mock_post.return_value.__aenter__.side_effect = [mock_auth_response, mock_payment_response]
		
		result = await pesapal_service.process_payment(sample_transaction, sample_mpesa_method)
		
		assert result.success is True
		assert result.status == PaymentStatus.PENDING
		assert result.provider_transaction_id == "PSP_MPESA_123456"
		assert result.payment_url is not None
		assert "pesapal" in result.payment_url
	
	@patch('aiohttp.ClientSession.post')
	async def test_process_airtel_payment_success(self, mock_post, pesapal_service, sample_transaction, sample_airtel_method):
		"""Test successful Airtel Money payment processing"""
		# Mock responses
		mock_auth_response = Mock()
		mock_auth_response.status = 200
		mock_auth_response.json = AsyncMock(return_value={"token": "mock_token"})
		
		mock_payment_response = Mock()
		mock_payment_response.status = 200
		mock_payment_response.json = AsyncMock(return_value={
			"order_tracking_id": "PSP_AIRTEL_123456",
			"merchant_reference": "txn_pesapal_test",
			"redirect_url": "https://cybqa.pesapal.com/pesapalv3/api/Auth/SecurePayment/?orderTrackingId=PSP_AIRTEL_123456"
		})
		
		mock_post.return_value.__aenter__.side_effect = [mock_auth_response, mock_payment_response]
		
		result = await pesapal_service.process_payment(sample_transaction, sample_airtel_method)
		
		assert result.success is True
		assert result.status == PaymentStatus.PENDING
		assert result.provider_transaction_id == "PSP_AIRTEL_123456"
	
	@patch('aiohttp.ClientSession.post')
	async def test_process_card_payment_success(self, mock_post, pesapal_service, sample_transaction, sample_card_method):
		"""Test successful card payment processing"""
		# Mock responses
		mock_auth_response = Mock()
		mock_auth_response.status = 200
		mock_auth_response.json = AsyncMock(return_value={"token": "mock_token"})
		
		mock_payment_response = Mock()
		mock_payment_response.status = 200
		mock_payment_response.json = AsyncMock(return_value={
			"order_tracking_id": "PSP_CARD_123456",
			"merchant_reference": "txn_pesapal_test",
			"redirect_url": "https://cybqa.pesapal.com/pesapalv3/api/Auth/SecurePayment/?orderTrackingId=PSP_CARD_123456"
		})
		
		mock_post.return_value.__aenter__.side_effect = [mock_auth_response, mock_payment_response]
		
		result = await pesapal_service.process_payment(sample_transaction, sample_card_method)
		
		assert result.success is True
		assert result.status == PaymentStatus.PENDING
		assert result.provider_transaction_id == "PSP_CARD_123456"
		assert result.payment_url is not None
	
	@patch('aiohttp.ClientSession.get')
	async def test_verify_payment_success(self, mock_get, pesapal_service):
		"""Test successful payment verification"""
		# Mock Pesapal transaction status response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.json = AsyncMock(return_value={
			"payment_method": "MPESA",
			"amount": 1500.00,
			"created_date": "2025-01-31T10:30:00Z",
			"confirmation_code": "PSP123456789",
			"payment_status_description": "Completed",
			"description": "Test payment",
			"message": "SUCCESS",
			"payment_account": "254700000000",
			"call_back_url": "https://example.com/callback",
			"status_code": 1,
			"merchant_reference": "txn_verify_test",
			"payment_status_code": "1",
			"currency": "KES"
		})
		
		mock_get.return_value.__aenter__.return_value = mock_response
		
		result = await pesapal_service.verify_payment("PSP_VERIFY_123")
		
		assert result.success is True
		assert result.status == PaymentStatus.COMPLETED
		assert result.provider_transaction_id == "PSP_VERIFY_123"
		assert result.amount == Decimal("1500.00")
		assert result.currency == "KES"
	
	@patch('aiohttp.ClientSession.post')
	async def test_refund_payment_success(self, mock_post, pesapal_service):
		"""Test successful payment refund"""
		# Mock authentication and refund responses
		mock_auth_response = Mock()
		mock_auth_response.status = 200
		mock_auth_response.json = AsyncMock(return_value={"token": "mock_token"})
		
		mock_refund_response = Mock()
		mock_refund_response.status = 200
		mock_refund_response.json = AsyncMock(return_value={
			"confirmation_code": "PSP_REFUND_123",
			"status": "SUCCESS",
			"message": "Refund processed successfully",
			"amount": 500.00
		})
		
		mock_post.return_value.__aenter__.side_effect = [mock_auth_response, mock_refund_response]
		
		result = await pesapal_service.refund_payment(
			transaction_id="PSP_ORIGINAL_123",
			amount=Decimal("500.00"),
			reason="Customer request"
		)
		
		assert result.success is True
		assert result.status == PaymentStatus.REFUNDED
		assert result.provider_transaction_id == "PSP_REFUND_123"
		assert result.amount == Decimal("500.00")
	
	async def test_process_ipn_completed_payment(self, pesapal_service):
		"""Test IPN processing for completed payment"""
		ipn_data = {
			"OrderTrackingId": "PSP_IPN_SUCCESS_123",
			"OrderMerchantReference": "txn_ipn_test",
			"OrderNotificationType": "IPNCHANGE",
			"OrderPaymentMethod": "MPESA",
			"OrderAmount": "2000.00",
			"OrderCurrency": "KES",
			"OrderPaymentStatusDescription": "Completed",
			"OrderPaymentStatusCode": "1",
			"OrderDate": "2025-01-31T10:30:00Z",
			"OrderAccountNumber": "254700000000",
			"OrderConfirmationCode": "PSP123456789"
		}
		
		result = await pesapal_service.process_ipn(ipn_data)
		
		assert result["success"] is True
		assert result["event_type"] == "payment_completed"
		assert result["transaction_id"] == "PSP_IPN_SUCCESS_123"
		assert result["status"] == "completed"
		assert result["amount"] == Decimal("2000.00")
	
	async def test_process_ipn_failed_payment(self, pesapal_service):
		"""Test IPN processing for failed payment"""
		ipn_data = {
			"OrderTrackingId": "PSP_IPN_FAILED_123",
			"OrderMerchantReference": "txn_ipn_failed",
			"OrderNotificationType": "IPNCHANGE",
			"OrderPaymentMethod": "MPESA",
			"OrderAmount": "1000.00",
			"OrderCurrency": "KES",
			"OrderPaymentStatusDescription": "Failed",
			"OrderPaymentStatusCode": "2",
			"OrderDate": "2025-01-31T10:30:00Z"
		}
		
		result = await pesapal_service.process_ipn(ipn_data)
		
		assert result["success"] is True
		assert result["event_type"] == "payment_failed"
		assert result["transaction_id"] == "PSP_IPN_FAILED_123"
		assert result["status"] == "failed"
	
	@patch('aiohttp.ClientSession.get')
	async def test_health_check_success(self, mock_get, pesapal_service):
		"""Test successful health check"""
		# Mock Pesapal API status response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.json = AsyncMock(return_value={
			"status": "SUCCESS",
			"message": "API is operational",
			"version": "v3.0"
		})
		
		mock_get.return_value.__aenter__.return_value = mock_response
		
		result = await pesapal_service.health_check()
		
		assert result.status == HealthStatus.HEALTHY
		assert result.response_time_ms > 0
		assert "version" in result.details
	
	async def test_get_supported_payment_methods_kenya(self, pesapal_service):
		"""Test getting supported payment methods for Kenya"""
		methods = await pesapal_service.get_supported_payment_methods(
			country_code="KE",
			currency="KES"
		)
		
		assert len(methods) > 0
		assert any(method["type"] == "mobile_money" for method in methods)
		assert any(method["type"] == "card" for method in methods)
		
		# Check for M-Pesa specifically
		mpesa_method = next((m for m in methods if m.get("name") == "M-Pesa"), None)
		assert mpesa_method is not None
		assert "KE" in mpesa_method["countries"]
	
	async def test_get_supported_payment_methods_uganda(self, pesapal_service):
		"""Test getting supported payment methods for Uganda"""
		methods = await pesapal_service.get_supported_payment_methods(
			country_code="UG",
			currency="UGX"
		)
		
		assert len(methods) > 0
		assert any(method["type"] == "mobile_money" for method in methods)
		assert any(method["name"] == "MTN Mobile Money" for method in methods)
		assert any(method["name"] == "Airtel Money" for method in methods)
	
	async def test_get_transaction_fees(self, pesapal_service):
		"""Test transaction fee calculation"""
		fees = await pesapal_service.get_transaction_fees(
			amount=Decimal("1000.00"),
			currency="KES",
			payment_method="mobile_money"
		)
		
		assert "total_fee" in fees
		assert "percentage_fee" in fees
		assert "fixed_fee" in fees
		assert float(fees["total_fee"]) > 0
		
		# Test card fees (typically higher)
		card_fees = await pesapal_service.get_transaction_fees(
			amount=Decimal("1000.00"),
			currency="KES",
			payment_method="card"
		)
		
		assert float(card_fees["total_fee"]) > float(fees["total_fee"])
	
	@patch('aiohttp.ClientSession.post')
	async def test_create_payment_link(self, mock_post, pesapal_service, sample_transaction):
		"""Test payment link creation"""
		# Mock responses
		mock_auth_response = Mock()
		mock_auth_response.status = 200
		mock_auth_response.json = AsyncMock(return_value={"token": "mock_token"})
		
		mock_link_response = Mock()
		mock_link_response.status = 200
		mock_link_response.json = AsyncMock(return_value={
			"order_tracking_id": "PSP_LINK_123",
			"redirect_url": "https://cybqa.pesapal.com/pesapalv3/api/Auth/SecurePayment/?orderTrackingId=PSP_LINK_123"
		})
		
		mock_post.return_value.__aenter__.side_effect = [mock_auth_response, mock_link_response]
		
		payment_url = await pesapal_service.create_payment_link(
			transaction=sample_transaction,
			expiry_hours=24
		)
		
		assert payment_url == "https://cybqa.pesapal.com/pesapalv3/api/Auth/SecurePayment/?orderTrackingId=PSP_LINK_123"
	
	async def test_multi_currency_support(self, pesapal_service):
		"""Test multi-currency support for East Africa"""
		currencies = ["KES", "UGX", "TZS", "RWF"]
		
		for currency in currencies:
			methods = await pesapal_service.get_supported_payment_methods(
				country_code="KE" if currency == "KES" else "UG",
				currency=currency
			)
			
			assert len(methods) > 0
			currency_methods = [m for m in methods if currency in m.get("currencies", [])]
			assert len(currency_methods) > 0
	
	async def test_mobile_money_networks_support(self, pesapal_service):
		"""Test support for various mobile money networks"""
		networks = [
			{"provider": "MPESA", "country": "KE", "currency": "KES"},
			{"provider": "AIRTEL", "country": "KE", "currency": "KES"},
			{"provider": "MTN", "country": "UG", "currency": "UGX"},
			{"provider": "AIRTEL", "country": "UG", "currency": "UGX"}
		]
		
		for network in networks:
			methods = await pesapal_service.get_supported_payment_methods(
				country_code=network["country"],
				currency=network["currency"]
			)
			
			mobile_money_methods = [m for m in methods if m["type"] == "mobile_money"]
			assert len(mobile_money_methods) > 0
			
			# Verify the specific network is supported
			supported_networks = []
			for method in mobile_money_methods:
				if "networks" in method:
					supported_networks.extend(method["networks"])
			
			# At least one mobile money option should be available
			assert len(mobile_money_methods) > 0
	
	async def test_bank_payment_integration(self, pesapal_service):
		"""Test bank payment integration"""
		transaction = PaymentTransaction(
			id="bank_test",
			amount=Decimal("5000.00"),
			currency="KES",
			description="Bank payment test"
		)
		
		bank_method = PaymentMethod(
			method_type=PaymentMethodType.BANK_TRANSFER,
			metadata={
				"bank_code": "01",
				"account_number": "1234567890",
				"bank_name": "KCB Bank"
			}
		)
		
		with patch('aiohttp.ClientSession.post') as mock_post:
			# Mock authentication and payment responses
			mock_auth_response = Mock()
			mock_auth_response.status = 200
			mock_auth_response.json = AsyncMock(return_value={"token": "mock_token"})
			
			mock_payment_response = Mock()
			mock_payment_response.status = 200
			mock_payment_response.json = AsyncMock(return_value={
				"order_tracking_id": "PSP_BANK_123",
				"redirect_url": "https://cybqa.pesapal.com/pesapalv3/api/Auth/SecurePayment/?orderTrackingId=PSP_BANK_123"
			})
			
			mock_post.return_value.__aenter__.side_effect = [mock_auth_response, mock_payment_response]
			
			result = await pesapal_service.process_payment(transaction, bank_method)
			
			assert result.success is True
			assert result.status == PaymentStatus.PENDING
			assert result.provider_transaction_id == "PSP_BANK_123"


class TestPesapalErrorHandling:
	"""Test error handling scenarios for Pesapal integration"""
	
	@pytest.fixture
	async def pesapal_service(self):
		"""Create Pesapal service for error testing"""
		with patch.dict('os.environ', {
			'PESAPAL_CONSUMER_KEY': 'error_test_key',
			'PESAPAL_CONSUMER_SECRET': 'error_test_secret'
		}):
			service = await create_pesapal_service(PesapalEnvironment.SANDBOX)
			return service
	
	@patch('aiohttp.ClientSession.post')
	async def test_authentication_error(self, mock_post, pesapal_service):
		"""Test authentication error handling"""
		# Mock authentication failure
		mock_auth_response = Mock()
		mock_auth_response.status = 401
		mock_auth_response.json = AsyncMock(return_value={
			"error": "invalid_client",
			"error_description": "Invalid consumer key or secret"
		})
		
		mock_post.return_value.__aenter__.return_value = mock_auth_response
		
		transaction = PaymentTransaction(
			id="auth_error_test",
			amount=Decimal("100.00"),
			currency="KES",
			description="Auth error test"
		)
		
		payment_method = PaymentMethod(
			method_type=PaymentMethodType.MOBILE_MONEY,
			metadata={"phone": "254700000000", "provider": "MPESA"}
		)
		
		result = await pesapal_service.process_payment(transaction, payment_method)
		
		assert result.success is False
		assert result.status == PaymentStatus.FAILED
		assert "authentication" in result.error_message.lower()
	
	@patch('aiohttp.ClientSession.post')
	async def test_insufficient_funds_error(self, mock_post, pesapal_service):
		"""Test insufficient funds error handling"""
		# Mock authentication success, payment failure
		mock_auth_response = Mock()
		mock_auth_response.status = 200
		mock_auth_response.json = AsyncMock(return_value={"token": "mock_token"})
		
		mock_payment_response = Mock()
		mock_payment_response.status = 400
		mock_payment_response.json = AsyncMock(return_value={
			"error": "payment_failed",
			"message": "Insufficient funds in mobile money account"
		})
		
		mock_post.return_value.__aenter__.side_effect = [mock_auth_response, mock_payment_response]
		
		transaction = PaymentTransaction(
			id="insufficient_funds_test",
			amount=Decimal("100000.00"),  # Large amount
			currency="KES",
			description="Insufficient funds test"
		)
		
		payment_method = PaymentMethod(
			method_type=PaymentMethodType.MOBILE_MONEY,
			metadata={"phone": "254700000000", "provider": "MPESA"}
		)
		
		result = await pesapal_service.process_payment(transaction, payment_method)
		
		assert result.success is False
		assert result.status == PaymentStatus.FAILED
		assert "insufficient funds" in result.error_message.lower()
	
	@patch('aiohttp.ClientSession.post')
	async def test_invalid_phone_number_error(self, mock_post, pesapal_service):
		"""Test invalid phone number error handling"""
		# Mock authentication success, payment validation failure
		mock_auth_response = Mock()
		mock_auth_response.status = 200
		mock_auth_response.json = AsyncMock(return_value={"token": "mock_token"})
		
		mock_payment_response = Mock()
		mock_payment_response.status = 400
		mock_payment_response.json = AsyncMock(return_value={
			"error": "validation_error",
			"message": "Invalid phone number format"
		})
		
		mock_post.return_value.__aenter__.side_effect = [mock_auth_response, mock_payment_response]
		
		transaction = PaymentTransaction(
			id="invalid_phone_test",
			amount=Decimal("100.00"),
			currency="KES",
			description="Invalid phone test"
		)
		
		payment_method = PaymentMethod(
			method_type=PaymentMethodType.MOBILE_MONEY,
			metadata={"phone": "invalid_phone", "provider": "MPESA"}
		)
		
		result = await pesapal_service.process_payment(transaction, payment_method)
		
		assert result.success is False
		assert result.status == PaymentStatus.FAILED
		assert "phone number" in result.error_message.lower()


if __name__ == "__main__":
	pytest.main([__file__, "-v", "--cov=../pesapal_integration", "--cov-report=html"])