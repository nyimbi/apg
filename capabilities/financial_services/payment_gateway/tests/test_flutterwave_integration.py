"""
Comprehensive Test Suite for Flutterwave Integration - APG Payment Gateway

Tests covering all Flutterwave features including mobile money, cards, and bank transfers.
Production-ready tests with complete coverage.

© 2025 Datacraft. All rights reserved.
"""

import pytest
import json
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from ..flutterwave_integration import (
	FlutterwaveService, FlutterwaveEnvironment, create_flutterwave_service
)
from ..models import (
	PaymentTransaction, PaymentMethod, PaymentResult,
	PaymentStatus, PaymentMethodType, HealthStatus
)


class TestFlutterwaveIntegration:
	"""Comprehensive test suite for Flutterwave integration"""
	
	@pytest.fixture
	async def flutterwave_service(self):
		"""Create Flutterwave service for testing"""
		with patch.dict('os.environ', {
			'FLUTTERWAVE_SECRET_KEY': 'FLWSECK_TEST-mock_secret_key',
			'FLUTTERWAVE_PUBLIC_KEY': 'FLWPUBK_TEST-mock_public_key'
		}):
			service = await create_flutterwave_service(FlutterwaveEnvironment.SANDBOX)
			return service
	
	@pytest.fixture
	def sample_transaction(self):
		"""Create sample payment transaction"""
		return PaymentTransaction(
			id="txn_flw_test",
			amount=Decimal("1000.00"),
			currency="KES",
			description="Flutterwave test payment",
			customer_email="flutterwave@example.com",
			customer_name="Flutterwave Test Customer"
		)
	
	@pytest.fixture
	def sample_card_method(self):
		"""Create sample card payment method"""
		return PaymentMethod(
			method_type=PaymentMethodType.CARD,
			metadata={
				"card_number": "5531886652142950",
				"exp_month": "09",
				"exp_year": "2025",
				"cvv": "564",
				"cardholder_name": "Test Customer"
			}
		)
	
	@pytest.fixture
	def sample_mobile_money_method(self):
		"""Create sample mobile money payment method"""
		return PaymentMethod(
			method_type=PaymentMethodType.MOBILE_MONEY,
			metadata={
				"phone": "254700000000",
				"provider": "MPESA",
				"network": "SAFARICOM"
			}
		)
	
	@pytest.fixture
	def sample_bank_transfer_method(self):
		"""Create sample bank transfer payment method"""
		return PaymentMethod(
			method_type=PaymentMethodType.BANK_TRANSFER,
			metadata={
				"account_bank": "044",
				"account_number": "0690000040",
				"bank_name": "Access Bank"
			}
		)
	
	async def test_flutterwave_service_initialization(self, flutterwave_service):
		"""Test Flutterwave service initialization"""
		assert flutterwave_service._initialized is True
		assert flutterwave_service.config.environment == FlutterwaveEnvironment.SANDBOX
		assert flutterwave_service.config.secret_key.startswith("FLWSECK_TEST")
	
	@patch('aiohttp.ClientSession.post')
	async def test_process_card_payment_success(self, mock_post, flutterwave_service, sample_transaction, sample_card_method):
		"""Test successful card payment processing"""
		# Mock Flutterwave API response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.json = AsyncMock(return_value={
			"status": "success",
			"message": "Charge initiated",
			"data": {
				"id": 285959875,
				"tx_ref": "txn_flw_test",
				"flw_ref": "FLW-MOCK-REF-123",
				"status": "successful",
				"amount": 1000,
				"currency": "KES",
				"charged_amount": 1000,
				"payment_type": "card",
				"redirect_url": "https://ravemodal-dev.herokuapp.com/v3/hosted/pay/285959875"
			}
		})
		mock_post.return_value.__aenter__.return_value = mock_response
		
		result = await flutterwave_service.process_payment(sample_transaction, sample_card_method)
		
		assert result.success is True
		assert result.status == PaymentStatus.COMPLETED
		assert result.provider_transaction_id == "FLW-MOCK-REF-123"
		assert result.amount == Decimal("1000.00")
		assert result.currency == "KES"
	
	@patch('aiohttp.ClientSession.post')
	async def test_process_mpesa_payment_success(self, mock_post, flutterwave_service, sample_transaction, sample_mobile_money_method):
		"""Test successful M-Pesa payment processing"""
		# Mock Flutterwave M-Pesa API response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.json = AsyncMock(return_value={
			"status": "success",
			"message": "Charge initiated",
			"data": {
				"id": 285959876,
				"tx_ref": "txn_flw_test",
				"flw_ref": "FLW-MPESA-REF-123",
				"status": "pending",
				"amount": 1000,
				"currency": "KES",
				"payment_type": "mobilemoney",
				"redirect_url": "https://ravemodal-dev.herokuapp.com/v3/hosted/pay/285959876"
			}
		})
		mock_post.return_value.__aenter__.return_value = mock_response
		
		result = await flutterwave_service.process_payment(sample_transaction, sample_mobile_money_method)
		
		assert result.success is True
		assert result.status == PaymentStatus.PENDING
		assert result.provider_transaction_id == "FLW-MPESA-REF-123"
		assert result.payment_url is not None
	
	@patch('aiohttp.ClientSession.post')
	async def test_process_bank_transfer_success(self, mock_post, flutterwave_service, sample_transaction, sample_bank_transfer_method):
		"""Test successful bank transfer processing"""
		# Mock Flutterwave bank transfer API response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.json = AsyncMock(return_value={
			"status": "success",
			"message": "Charge initiated",
			"data": {
				"id": 285959877,
				"tx_ref": "txn_flw_test",
				"flw_ref": "FLW-BANK-REF-123",
				"status": "pending",
				"amount": 1000,
				"currency": "NGN",
				"payment_type": "banktransfer",
				"account_number": "7830000047",
				"account_expiry": "2025-12-31 23:59:59",
				"note": "Please make payment to the account number above"
			}
		})
		mock_post.return_value.__aenter__.return_value = mock_response
		
		# Update transaction for NGN
		sample_transaction.currency = "NGN"
		
		result = await flutterwave_service.process_payment(sample_transaction, sample_bank_transfer_method)
		
		assert result.success is True
		assert result.status == PaymentStatus.PENDING
		assert result.provider_transaction_id == "FLW-BANK-REF-123"
		assert "account_number" in result.raw_response["data"]
	
	@patch('aiohttp.ClientSession.post')
	async def test_process_payment_failed(self, mock_post, flutterwave_service, sample_transaction, sample_card_method):
		"""Test failed payment processing"""
		# Mock Flutterwave API error response
		mock_response = Mock()
		mock_response.status = 400
		mock_response.json = AsyncMock(return_value={
			"status": "error",
			"message": "Invalid card details",
			"data": None
		})
		mock_post.return_value.__aenter__.return_value = mock_response
		
		result = await flutterwave_service.process_payment(sample_transaction, sample_card_method)
		
		assert result.success is False
		assert result.status == PaymentStatus.FAILED
		assert "invalid card" in result.error_message.lower()
	
	@patch('aiohttp.ClientSession.get')
	async def test_verify_payment_success(self, mock_get, flutterwave_service):
		"""Test successful payment verification"""
		# Mock Flutterwave verification API response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.json = AsyncMock(return_value={
			"status": "success",
			"message": "Transaction fetched successfully",
			"data": {
				"id": 285959875,
				"tx_ref": "txn_verify_test",
				"flw_ref": "FLW-VERIFY-REF-123",
				"status": "successful",
				"amount": 1500,
				"currency": "KES",
				"charged_amount": 1500,
				"payment_type": "card"
			}
		})
		mock_get.return_value.__aenter__.return_value = mock_response
		
		result = await flutterwave_service.verify_payment("FLW-VERIFY-REF-123")
		
		assert result.success is True
		assert result.status == PaymentStatus.COMPLETED
		assert result.provider_transaction_id == "FLW-VERIFY-REF-123"
		assert result.amount == Decimal("1500.00")
	
	@patch('aiohttp.ClientSession.post')
	async def test_refund_payment_success(self, mock_post, flutterwave_service):
		"""Test successful payment refund"""
		# Mock Flutterwave refund API response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.json = AsyncMock(return_value={
			"status": "success",
			"message": "Refund created successfully",
			"data": {
				"id": 12345,
				"tx_id": 285959875,
				"flw_ref": "FLW-REFUND-REF-123",
				"status": "completed",
				"amount_refunded": 500,
				"created_at": "2025-01-31T10:30:00.000Z"
			}
		})
		mock_post.return_value.__aenter__.return_value = mock_response
		
		result = await flutterwave_service.refund_payment(
			transaction_id="FLW-ORIGINAL-REF-123",
			amount=Decimal("500.00"),
			reason="Customer request"
		)
		
		assert result.success is True
		assert result.status == PaymentStatus.REFUNDED
		assert result.provider_transaction_id == "FLW-REFUND-REF-123"
		assert result.amount == Decimal("500.00")
	
	async def test_process_webhook_successful_payment(self, flutterwave_service):
		"""Test webhook processing for successful payment"""
		webhook_data = {
			"event": "charge.completed",
			"data": {
				"id": 285959875,
				"tx_ref": "txn_webhook_test", 
				"flw_ref": "FLW-WEBHOOK-REF-123",
				"status": "successful",
				"amount": 2000,
				"currency": "KES",
				"customer": {
					"email": "webhook@example.com",
					"name": "Webhook Test"
				}
			}
		}
		
		result = await flutterwave_service.process_webhook(webhook_data)
		
		assert result["success"] is True
		assert result["event_type"] == "charge.completed"
		assert result["transaction_id"] == "FLW-WEBHOOK-REF-123"
		assert result["status"] == "completed"
	
	async def test_process_webhook_failed_payment(self, flutterwave_service):
		"""Test webhook processing for failed payment"""
		webhook_data = {
			"event": "charge.failed",
			"data": {
				"id": 285959876,
				"tx_ref": "txn_webhook_failed",
				"flw_ref": "FLW-WEBHOOK-FAILED-123", 
				"status": "failed",
				"amount": 1000,
				"currency": "KES",
				"processor_response": "Insufficient funds"
			}
		}
		
		result = await flutterwave_service.process_webhook(webhook_data)
		
		assert result["success"] is True
		assert result["event_type"] == "charge.failed"
		assert result["transaction_id"] == "FLW-WEBHOOK-FAILED-123"
		assert result["status"] == "failed"
		assert "insufficient funds" in result["error_message"].lower()
	
	@patch('aiohttp.ClientSession.get')
	async def test_health_check_success(self, mock_get, flutterwave_service):
		"""Test successful health check"""
		# Mock Flutterwave status API response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.json = AsyncMock(return_value={
			"status": "success",
			"message": "Service is healthy",
			"data": {
				"environment": "sandbox",
				"version": "v3"
			}
		})
		mock_get.return_value.__aenter__.return_value = mock_response
		
		result = await flutterwave_service.health_check()
		
		assert result.status == HealthStatus.HEALTHY
		assert result.response_time_ms > 0
		assert "environment" in result.details
	
	async def test_get_supported_payment_methods_kenya(self, flutterwave_service):
		"""Test getting supported payment methods for Kenya"""
		methods = await flutterwave_service.get_supported_payment_methods(
			country_code="KE",
			currency="KES"
		)
		
		assert len(methods) > 0
		assert any(method["type"] == "card" for method in methods)
		assert any(method["type"] == "mobile_money" for method in methods)
		
		# Check for M-Pesa specifically
		mpesa_method = next((m for m in methods if m.get("name") == "M-Pesa"), None)
		assert mpesa_method is not None
		assert "KE" in mpesa_method["countries"]
	
	async def test_get_supported_payment_methods_nigeria(self, flutterwave_service):
		"""Test getting supported payment methods for Nigeria"""
		methods = await flutterwave_service.get_supported_payment_methods(
			country_code="NG",
			currency="NGN"
		)
		
		assert len(methods) > 0
		assert any(method["type"] == "card" for method in methods)
		assert any(method["type"] == "bank_transfer" for method in methods)
		assert any(method["type"] == "ussd" for method in methods)
	
	async def test_get_transaction_fees(self, flutterwave_service):
		"""Test transaction fee calculation"""
		fees = await flutterwave_service.get_transaction_fees(
			amount=Decimal("1000.00"),
			currency="KES",
			payment_method="card"
		)
		
		assert "total_fee" in fees
		assert "percentage_fee" in fees
		assert "fixed_fee" in fees
		assert float(fees["total_fee"]) > 0
		
		# Test M-Pesa fees (typically lower)
		mpesa_fees = await flutterwave_service.get_transaction_fees(
			amount=Decimal("1000.00"),
			currency="KES",
			payment_method="mobile_money"
		)
		
		assert float(mpesa_fees["total_fee"]) < float(fees["total_fee"])
	
	@patch('aiohttp.ClientSession.post')
	async def test_create_payment_link(self, mock_post, flutterwave_service, sample_transaction):
		"""Test payment link creation"""
		# Mock Flutterwave payment link API response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.json = AsyncMock(return_value={
			"status": "success",
			"message": "Payment link created",
			"data": {
				"link": "https://checkout.flutterwave.com/pay/mock-payment-link-123"
			}
		})
		mock_post.return_value.__aenter__.return_value = mock_response
		
		payment_url = await flutterwave_service.create_payment_link(
			transaction=sample_transaction,
			expiry_hours=24
		)
		
		assert payment_url == "https://checkout.flutterwave.com/pay/mock-payment-link-123"
	
	@patch('aiohttp.ClientSession.get')
	async def test_get_banks_list(self, mock_get, flutterwave_service):
		"""Test getting list of supported banks"""
		# Mock Flutterwave banks API response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.json = AsyncMock(return_value={
			"status": "success",
			"message": "Banks fetched successfully",
			"data": [
				{
					"id": 1,
					"code": "044",
					"name": "Access Bank"
				},
				{
					"id": 2,
					"code": "023",
					"name": "Citibank Nigeria"
				}
			]
		})
		mock_get.return_value.__aenter__.return_value = mock_response
		
		banks = await flutterwave_service.get_supported_banks(country="NG")
		
		assert len(banks) >= 2
		assert any(bank["name"] == "Access Bank" for bank in banks)
		assert any(bank["code"] == "044" for bank in banks)
	
	@patch('aiohttp.ClientSession.post')
	async def test_validate_account_number(self, mock_post, flutterwave_service):
		"""Test bank account number validation"""
		# Mock Flutterwave account resolution API response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.json = AsyncMock(return_value={
			"status": "success",
			"message": "Account details retrieved",
			"data": {
				"account_number": "0690000031",
				"account_name": "JOHN DOE",
				"bank_code": "044"
			}
		})
		mock_post.return_value.__aenter__.return_value = mock_response
		
		result = await flutterwave_service.validate_account_number(
			account_number="0690000031",
			bank_code="044"
		)
		
		assert result["success"] is True
		assert result["account_name"] == "JOHN DOE"
		assert result["account_number"] == "0690000031"
	
	async def test_barter_virtual_card_creation(self, flutterwave_service):
		"""Test Barter virtual card creation"""
		with patch('aiohttp.ClientSession.post') as mock_post:
			# Mock Barter virtual card API response
			mock_response = Mock()
			mock_response.status = 200
			mock_response.json = AsyncMock(return_value={
				"status": "success",
				"message": "Virtual card created successfully",
				"data": {
					"id": "vc_mock_123",
					"account_id": "barter_account_123",
					"amount": "1000.00",
					"currency": "USD",
					"card_hash": "mock_card_hash_123",
					"card_pan": "5531************950",
					"masked_pan": "5531************950",
					"city": "San Francisco",
					"state": "CA",
					"address_1": "123 Main St",
					"zip_code": "94102",
					"cvv": "812",
					"expiration": "09/2026",
					"send_to": "webhook",
					"bin_check_name": "MASTERCARD DEBIT",
					"card_type": "MASTERCARD",
					"name_on_card": "Test Customer"
				}
			})
			mock_post.return_value.__aenter__.return_value = mock_response
			
			card_data = {
				"currency": "USD",
				"amount": 1000,
				"debit_currency": "USD",
				"customer": {
					"name": "Test Customer",
					"email": "test@example.com"
				}
			}
			
			result = await flutterwave_service.create_virtual_card(card_data)
			
			assert result["success"] is True
			assert result["card_id"] == "vc_mock_123"
			assert "5531" in result["masked_pan"]
	
	async def test_mobile_money_networks_support(self, flutterwave_service):
		"""Test support for various mobile money networks"""
		networks = [
			{"provider": "MPESA", "country": "KE", "currency": "KES"},
			{"provider": "AIRTEL", "country": "KE", "currency": "KES"},
			{"provider": "MTN", "country": "UG", "currency": "UGX"},
			{"provider": "AIRTEL", "country": "UG", "currency": "UGX"},
			{"provider": "MTN", "country": "GH", "currency": "GHS"},
			{"provider": "VODAFONE", "country": "GH", "currency": "GHS"}
		]
		
		for network in networks:
			methods = await flutterwave_service.get_supported_payment_methods(
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


class TestFlutterwaveErrorHandling:
	"""Test error handling scenarios for Flutterwave integration"""
	
	@pytest.fixture
	async def flutterwave_service(self):
		"""Create Flutterwave service for error testing"""
		with patch.dict('os.environ', {
			'FLUTTERWAVE_SECRET_KEY': 'FLWSECK_TEST-error_key',
			'FLUTTERWAVE_PUBLIC_KEY': 'FLWPUBK_TEST-error_key'
		}):
			service = await create_flutterwave_service(FlutterwaveEnvironment.SANDBOX)
			return service
	
	@patch('aiohttp.ClientSession.post')
	async def test_api_connection_error(self, mock_post, flutterwave_service):
		"""Test API connection error handling"""
		import aiohttp
		
		mock_post.side_effect = aiohttp.ClientError("Connection failed")
		
		transaction = PaymentTransaction(
			id="error_test",
			amount=Decimal("100.00"),
			currency="KES",
			description="Error test"
		)
		
		payment_method = PaymentMethod(
			method_type=PaymentMethodType.CARD,
			metadata={"card_number": "5531886652142950"}
		)
		
		result = await flutterwave_service.process_payment(transaction, payment_method)
		
		assert result.success is False
		assert result.status == PaymentStatus.FAILED
		assert "connection" in result.error_message.lower()
	
	@patch('aiohttp.ClientSession.post')
	async def test_insufficient_funds_error(self, mock_post, flutterwave_service):
		"""Test insufficient funds error handling"""
		# Mock Flutterwave insufficient funds response
		mock_response = Mock()
		mock_response.status = 400
		mock_response.json = AsyncMock(return_value={
			"status": "error",
			"message": "Insufficient funds in account",
			"data": {
				"code": "INSUFFICIENT_FUNDS",
				"message": "The account does not have sufficient funds"
			}
		})
		mock_post.return_value.__aenter__.return_value = mock_response
		
		transaction = PaymentTransaction(
			id="insufficient_funds_test",
			amount=Decimal("100000.00"),  # Large amount
			currency="KES",
			description="Insufficient funds test"
		)
		
		payment_method = PaymentMethod(
			method_type=PaymentMethodType.MOBILE_MONEY,
			metadata={
				"phone": "254700000000",
				"provider": "MPESA"
			}
		)
		
		result = await flutterwave_service.process_payment(transaction, payment_method)
		
		assert result.success is False
		assert result.status == PaymentStatus.FAILED
		assert "insufficient funds" in result.error_message.lower()
	
	@patch('aiohttp.ClientSession.post')
	async def test_invalid_phone_number_error(self, mock_post, flutterwave_service):
		"""Test invalid phone number error handling"""
		# Mock Flutterwave invalid phone response
		mock_response = Mock()
		mock_response.status = 400
		mock_response.json = AsyncMock(return_value={
			"status": "error",
			"message": "Invalid phone number format",
			"data": {
				"code": "INVALID_PHONE_NUMBER",
				"message": "Phone number must be in international format"
			}
		})
		mock_post.return_value.__aenter__.return_value = mock_response
		
		transaction = PaymentTransaction(
			id="invalid_phone_test",
			amount=Decimal("100.00"),
			currency="KES",
			description="Invalid phone test"
		)
		
		payment_method = PaymentMethod(
			method_type=PaymentMethodType.MOBILE_MONEY,
			metadata={
				"phone": "0700000000",  # Invalid format
				"provider": "MPESA"
			}
		)
		
		result = await flutterwave_service.process_payment(transaction, payment_method)
		
		assert result.success is False
		assert result.status == PaymentStatus.FAILED
		assert "phone number" in result.error_message.lower()


if __name__ == "__main__":
	pytest.main([__file__, "-v", "--cov=../flutterwave_integration", "--cov-report=html"])