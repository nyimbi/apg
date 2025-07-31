"""
Comprehensive Test Suite for DPO Integration - APG Payment Gateway

Tests covering all DPO features including mobile money, cards, and bank transfers across Africa.
Production-ready tests with complete coverage for Pan-African markets.

© 2025 Datacraft. All rights reserved.
"""

import pytest
import json
import xml.etree.ElementTree as ET
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from ..dpo_integration import (
	DPOService, DPOEnvironment, create_dpo_service
)
from ..models import (
	PaymentTransaction, PaymentMethod, PaymentResult,
	PaymentStatus, PaymentMethodType, HealthStatus
)


class TestDPOIntegration:
	"""Comprehensive test suite for DPO integration"""
	
	@pytest.fixture
	async def dpo_service(self):
		"""Create DPO service for testing"""
		with patch.dict('os.environ', {
			'DPO_COMPANY_TOKEN_SANDBOX': 'test_company_token',
			'DPO_SERVICE_TYPE': 'test_service_type',
			'DPO_CALLBACK_URL': 'https://example.com/callback/dpo'
		}):
			service = await create_dpo_service(DPOEnvironment.SANDBOX)
			return service
	
	@pytest.fixture
	def sample_transaction(self):
		"""Create sample payment transaction"""
		return PaymentTransaction(
			id="txn_dpo_test",
			amount=Decimal("2000.00"),
			currency="KES",
			description="DPO test payment",
			customer_email="dpo@example.com",
			customer_name="DPO Test Customer"
		)
	
	@pytest.fixture
	def sample_card_method(self):
		"""Create sample card payment method"""
		return PaymentMethod(
			method_type=PaymentMethodType.CARD,
			metadata={
				"card_number": "5123450000000008",
				"exp_month": "12",
				"exp_year": "2025",
				"cvv": "100",
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
	def sample_bank_method(self):
		"""Create sample bank transfer payment method"""
		return PaymentMethod(
			method_type=PaymentMethodType.BANK_TRANSFER,
			metadata={
				"bank_code": "KCB",
				"account_number": "1234567890",
				"bank_name": "KCB Bank Kenya"
			}
		)
	
	async def test_dpo_service_initialization(self, dpo_service):
		"""Test DPO service initialization"""
		assert dpo_service._initialized is True
		assert dpo_service.config.environment == DPOEnvironment.SANDBOX
		assert dpo_service.config.company_token == "test_company_token"
	
	@patch('aiohttp.ClientSession.post')
	async def test_create_payment_token_success(self, mock_post, dpo_service, sample_transaction):
		"""Test successful payment token creation"""
		# Mock DPO createToken API response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.text = AsyncMock(return_value="""<?xml version="1.0" encoding="utf-8"?>
			<API3G>
				<Result>000</Result>
				<ResultExplanation>Transaction token created successfully</ResultExplanation>
				<TransToken>DPO_TOKEN_12345</TransToken>
				<TransRef>DPO_REF_12345</TransRef>
			</API3G>""")
		
		mock_post.return_value.__aenter__.return_value = mock_response
		
		token_result = await dpo_service.create_payment_token(sample_transaction)
		
		assert token_result["success"] is True
		assert token_result["token"] == "DPO_TOKEN_12345"
		assert token_result["reference"] == "DPO_REF_12345"
	
	@patch('aiohttp.ClientSession.post')
	async def test_process_card_payment_success(self, mock_post, dpo_service, sample_transaction, sample_card_method):
		"""Test successful card payment processing"""
		# Mock token creation and charge card responses
		mock_token_response = Mock()
		mock_token_response.status = 200
		mock_token_response.text = AsyncMock(return_value="""<?xml version="1.0" encoding="utf-8"?>
			<API3G>
				<Result>000</Result>
				<TransToken>DPO_TOKEN_CARD</TransToken>
				<TransRef>DPO_REF_CARD</TransRef>
			</API3G>""")
		
		mock_charge_response = Mock()
		mock_charge_response.status = 200
		mock_charge_response.text = AsyncMock(return_value="""<?xml version="1.0" encoding="utf-8"?>
			<API3G>
				<Result>000</Result>
				<ResultExplanation>Transaction successful</ResultExplanation>
				<TransToken>DPO_TOKEN_CARD</TransToken>
				<TransRef>DPO_REF_CARD</TransRef>
				<CCDapproval>123456</CCDapproval>
				<TransactionApproval>APPROVED</TransactionApproval>
			</API3G>""")
		
		mock_post.return_value.__aenter__.side_effect = [mock_token_response, mock_charge_response]
		
		result = await dpo_service.process_payment(sample_transaction, sample_card_method)
		
		assert result.success is True
		assert result.status == PaymentStatus.COMPLETED
		assert result.provider_transaction_id == "DPO_REF_CARD"
		assert result.amount == Decimal("2000.00")
		assert result.currency == "KES"
	
	@patch('aiohttp.ClientSession.post')
	async def test_process_mpesa_payment_success(self, mock_post, dpo_service, sample_transaction, sample_mpesa_method):
		"""Test successful M-Pesa payment processing"""
		# Mock token creation and mobile money responses
		mock_token_response = Mock()
		mock_token_response.status = 200
		mock_token_response.text = AsyncMock(return_value="""<?xml version="1.0" encoding="utf-8"?>
			<API3G>
				<Result>000</Result>
				<TransToken>DPO_TOKEN_MPESA</TransToken>
				<TransRef>DPO_REF_MPESA</TransRef>
			</API3G>""")
		
		mock_mobile_response = Mock()
		mock_mobile_response.status = 200
		mock_mobile_response.text = AsyncMock(return_value="""<?xml version="1.0" encoding="utf-8"?>
			<API3G>
				<Result>000</Result>
				<ResultExplanation>Mobile money payment initiated</ResultExplanation>
				<TransToken>DPO_TOKEN_MPESA</TransToken>
				<TransRef>DPO_REF_MPESA</TransRef>
				<MNOReference>MPESA_REF_123</MNOReference>
				<CustomerMessage>Enter PIN to confirm payment</CustomerMessage>
			</API3G>""")
		
		mock_post.return_value.__aenter__.side_effect = [mock_token_response, mock_mobile_response]
		
		result = await dpo_service.process_payment(sample_transaction, sample_mpesa_method)
		
		assert result.success is True
		assert result.status == PaymentStatus.PENDING
		assert result.provider_transaction_id == "DPO_REF_MPESA"
		assert "Enter PIN" in result.raw_response.get("CustomerMessage", "")
	
	@patch('aiohttp.ClientSession.post')
	async def test_process_bank_payment_success(self, mock_post, dpo_service, sample_transaction, sample_bank_method):
		"""Test successful bank payment processing"""
		# Mock token creation response
		mock_token_response = Mock()
		mock_token_response.status = 200
		mock_token_response.text = AsyncMock(return_value="""<?xml version="1.0" encoding="utf-8"?>
			<API3G>
				<Result>000</Result>
				<TransToken>DPO_TOKEN_BANK</TransToken>
				<TransRef>DPO_REF_BANK</TransRef>
			</API3G>""")
		
		mock_post.return_value.__aenter__.return_value = mock_token_response
		
		result = await dpo_service.process_payment(sample_transaction, sample_bank_method)
		
		assert result.success is True
		assert result.status == PaymentStatus.PENDING
		assert result.provider_transaction_id == "DPO_REF_BANK"
		assert result.payment_url is not None
		assert "dpo" in result.payment_url.lower()
	
	@patch('aiohttp.ClientSession.post')
	async def test_verify_payment_success(self, mock_post, dpo_service):
		"""Test successful payment verification"""
		# Mock DPO verifyToken API response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.text = AsyncMock(return_value="""<?xml version="1.0" encoding="utf-8"?>
			<API3G>
				<Result>000</Result>
				<ResultExplanation>Transaction Paid</ResultExplanation>
				<TransToken>DPO_VERIFY_TOKEN</TransToken>
				<TransRef>DPO_VERIFY_REF</TransRef>
				<TransactionApproval>APPROVED</TransactionApproval>
				<TransactionCurrency>KES</TransactionCurrency>
				<TransactionAmount>1500.00</TransactionAmount>
				<CustomerPhone>254700000000</CustomerPhone>
				<CustomerCountry>KE</CustomerCountry>
				<TransactionRollingReserveAmount>0.00</TransactionRollingReserveAmount>
				<TransactionSettlementDate>2025-01-31</TransactionSettlementDate>
			</API3G>""")
		
		mock_post.return_value.__aenter__.return_value = mock_response
		
		result = await dpo_service.verify_payment("DPO_VERIFY_TOKEN")
		
		assert result.success is True
		assert result.status == PaymentStatus.COMPLETED
		assert result.provider_transaction_id == "DPO_VERIFY_REF"
		assert result.amount == Decimal("1500.00")
		assert result.currency == "KES"
	
	@patch('aiohttp.ClientSession.post')
	async def test_refund_payment_success(self, mock_post, dpo_service):
		"""Test successful payment refund"""
		# Mock DPO refund API response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.text = AsyncMock(return_value="""<?xml version="1.0" encoding="utf-8"?>
			<API3G>
				<Result>000</Result>
				<ResultExplanation>Refund successful</ResultExplanation>
				<TransToken>DPO_REFUND_TOKEN</TransToken>
				<TransRef>DPO_REFUND_REF</TransRef>
				<RefundAmount>500.00</RefundAmount>
				<RefundDate>2025-01-31</RefundDate>
			</API3G>""")
		
		mock_post.return_value.__aenter__.return_value = mock_response
		
		result = await dpo_service.refund_payment(
			transaction_id="DPO_ORIGINAL_REF",
			amount=Decimal("500.00"),
			reason="Customer request"
		)
		
		assert result.success is True
		assert result.status == PaymentStatus.REFUNDED
		assert result.provider_transaction_id == "DPO_REFUND_REF"
		assert result.amount == Decimal("500.00")
	
	async def test_process_callback_successful_payment(self, dpo_service):
		"""Test callback processing for successful payment"""
		callback_data = {
			"TransactionToken": "DPO_CALLBACK_TOKEN",
			"TransRef": "DPO_CALLBACK_REF",
			"TransactionApproval": "APPROVED",
			"TransactionCurrency": "KES",
			"TransactionAmount": "2500.00",
			"CustomerPhone": "254700000000",
			"PaymentMethod": "MPESA"
		}
		
		result = await dpo_service.process_callback(callback_data)
		
		assert result["success"] is True
		assert result["event_type"] == "payment_completed"
		assert result["transaction_id"] == "DPO_CALLBACK_REF"
		assert result["status"] == "completed"
		assert result["amount"] == Decimal("2500.00")
	
	async def test_process_callback_failed_payment(self, dpo_service):
		"""Test callback processing for failed payment"""
		callback_data = {
			"TransactionToken": "DPO_CALLBACK_FAILED",
			"TransRef": "DPO_CALLBACK_FAILED_REF",
			"TransactionApproval": "DECLINED",
			"TransactionCurrency": "KES",
			"TransactionAmount": "1000.00",
			"ResultExplanation": "Insufficient funds"
		}
		
		result = await dpo_service.process_callback(callback_data)
		
		assert result["success"] is True
		assert result["event_type"] == "payment_failed"
		assert result["transaction_id"] == "DPO_CALLBACK_FAILED_REF"
		assert result["status"] == "failed"
		assert "insufficient funds" in result["error_message"].lower()
	
	@patch('aiohttp.ClientSession.get')
	async def test_health_check_success(self, mock_get, dpo_service):
		"""Test successful health check"""
		# Mock DPO API status response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.text = AsyncMock(return_value="""<?xml version="1.0" encoding="utf-8"?>
			<API3G>
				<Result>000</Result>
				<ResultExplanation>API is operational</ResultExplanation>
				<ServiceStatus>ACTIVE</ServiceStatus>
				<Version>3.0</Version>
			</API3G>""")
		
		mock_get.return_value.__aenter__.return_value = mock_response
		
		result = await dpo_service.health_check()
		
		assert result.status == HealthStatus.HEALTHY
		assert result.response_time_ms > 0
		assert "version" in result.details
	
	async def test_get_supported_payment_methods_kenya(self, dpo_service):
		"""Test getting supported payment methods for Kenya"""
		methods = await dpo_service.get_supported_payment_methods(
			country_code="KE",
			currency="KES"
		)
		
		assert len(methods) > 0
		assert any(method["type"] == "card" for method in methods)
		assert any(method["type"] == "mobile_money" for method in methods)
		assert any(method["type"] == "bank_transfer" for method in methods)
		
		# Check for M-Pesa specifically
		mpesa_method = next((m for m in methods if m.get("name") == "M-Pesa"), None)
		assert mpesa_method is not None
		assert "KE" in mpesa_method["countries"]
	
	async def test_get_supported_payment_methods_south_africa(self, dpo_service):
		"""Test getting supported payment methods for South Africa"""
		methods = await dpo_service.get_supported_payment_methods(
			country_code="ZA",
			currency="ZAR"
		)
		
		assert len(methods) > 0
		assert any(method["type"] == "card" for method in methods)
		assert any(method["type"] == "bank_transfer" for method in methods)
		assert any(method["type"] == "eft" for method in methods)
	
	async def test_get_transaction_fees(self, dpo_service):
		"""Test transaction fee calculation"""
		fees = await dpo_service.get_transaction_fees(
			amount=Decimal("1000.00"),
			currency="KES",
			payment_method="mobile_money"
		)
		
		assert "total_fee" in fees
		assert "percentage_fee" in fees
		assert "fixed_fee" in fees
		assert float(fees["total_fee"]) > 0
		
		# Test card fees (typically higher)
		card_fees = await dpo_service.get_transaction_fees(
			amount=Decimal("1000.00"),
			currency="KES",
			payment_method="card"
		)
		
		assert float(card_fees["total_fee"]) > float(fees["total_fee"])
	
	@patch('aiohttp.ClientSession.post')
	async def test_create_payment_link(self, mock_post, dpo_service, sample_transaction):
		"""Test payment link creation"""
		# Mock DPO createToken response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.text = AsyncMock(return_value="""<?xml version="1.0" encoding="utf-8"?>
			<API3G>
				<Result>000</Result>
				<TransToken>DPO_LINK_TOKEN</TransToken>
				<TransRef>DPO_LINK_REF</TransRef>
			</API3G>""")
		
		mock_post.return_value.__aenter__.return_value = mock_response
		
		payment_url = await dpo_service.create_payment_link(
			transaction=sample_transaction,
			expiry_hours=24
		)
		
		assert "secure.3gdirectpay.com" in payment_url
		assert "DPO_LINK_TOKEN" in payment_url
	
	async def test_multi_currency_support(self, dpo_service):
		"""Test multi-currency support across Africa"""
		currencies = ["KES", "UGX", "TZS", "ZAR", "NGN", "GHS"]
		
		for currency in currencies:
			methods = await dpo_service.get_supported_payment_methods(
				country_code="KE" if currency == "KES" else "ZA",
				currency=currency
			)
			
			assert len(methods) > 0
			currency_methods = [m for m in methods if currency in m.get("currencies", [])]
			assert len(currency_methods) > 0
	
	async def test_mobile_money_networks_support(self, dpo_service):
		"""Test support for various mobile money networks"""
		networks = [
			{"provider": "MPESA", "country": "KE", "currency": "KES"},
			{"provider": "AIRTEL", "country": "KE", "currency": "KES"},
			{"provider": "MTN", "country": "UG", "currency": "UGX"},
			{"provider": "TIGO", "country": "TZ", "currency": "TZS"},
			{"provider": "ORANGE", "country": "CI", "currency": "XOF"}
		]
		
		for network in networks:
			methods = await dpo_service.get_supported_payment_methods(
				country_code=network["country"],
				currency=network["currency"]
			)
			
			mobile_money_methods = [m for m in methods if m["type"] == "mobile_money"]
			assert len(mobile_money_methods) > 0
	
	async def test_bank_integration_multiple_countries(self, dpo_service):
		"""Test bank integration across multiple African countries"""
		countries = [
			{"code": "KE", "currency": "KES", "banks": ["KCB", "Equity", "Co-operative"]},
			{"code": "ZA", "currency": "ZAR", "banks": ["FNB", "Standard Bank", "ABSA"]},
			{"code": "NG", "currency": "NGN", "banks": ["GTBank", "Zenith", "First Bank"]}
		]
		
		for country in countries:
			methods = await dpo_service.get_supported_payment_methods(
				country_code=country["code"],
				currency=country["currency"]
			)
			
			bank_methods = [m for m in methods if m["type"] == "bank_transfer"]
			assert len(bank_methods) > 0
			
			# Check that some banks are supported
			supported_banks = []
			for method in bank_methods:
				if "supported_banks" in method:
					supported_banks.extend(method["supported_banks"])
			
			assert len(supported_banks) > 0
	
	@patch('aiohttp.ClientSession.post')
	async def test_fraud_detection_integration(self, mock_post, dpo_service):
		"""Test fraud detection integration"""
		# Test with high-risk transaction
		high_risk_transaction = PaymentTransaction(
			id="fraud_test",
			amount=Decimal("50000.00"),  # High amount
			currency="USD",
			description="High value test",
			customer_email="suspicious@example.com"
		)
		
		payment_method = PaymentMethod(
			method_type=PaymentMethodType.CARD,
			metadata={"card_number": "4000000000000002"}  # Declined test card
		)
		
		# Mock DPO fraud screening response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.text = AsyncMock(return_value="""<?xml version="1.0" encoding="utf-8"?>
			<API3G>
				<Result>904</Result>
				<ResultExplanation>Transaction flagged for review</ResultExplanation>
				<TransToken>DPO_FRAUD_TOKEN</TransToken>
				<FraudAlert>true</FraudAlert>
				<FraudScore>85</FraudScore>
			</API3G>""")
		
		mock_post.return_value.__aenter__.return_value = mock_response
		
		result = await dpo_service.process_payment(high_risk_transaction, payment_method)
		
		assert result.success is False
		assert result.status == PaymentStatus.FAILED
		assert "fraud" in result.error_message.lower() or "review" in result.error_message.lower()


class TestDPOErrorHandling:
	"""Test error handling scenarios for DPO integration"""
	
	@pytest.fixture
	async def dpo_service(self):
		"""Create DPO service for error testing"""
		with patch.dict('os.environ', {
			'DPO_COMPANY_TOKEN_SANDBOX': 'error_test_token',
			'DPO_SERVICE_TYPE': 'error_service'
		}):
			service = await create_dpo_service(DPOEnvironment.SANDBOX)
			return service
	
	@patch('aiohttp.ClientSession.post')
	async def test_invalid_company_token_error(self, mock_post, dpo_service):
		"""Test invalid company token error handling"""
		# Mock DPO authentication failure
		mock_response = Mock()
		mock_response.status = 200
		mock_response.text = AsyncMock(return_value="""<?xml version="1.0" encoding="utf-8"?>
			<API3G>
				<Result>901</Result>
				<ResultExplanation>Invalid company token</ResultExplanation>
			</API3G>""")
		
		mock_post.return_value.__aenter__.return_value = mock_response
		
		transaction = PaymentTransaction(
			id="auth_error_test",
			amount=Decimal("100.00"),
			currency="KES",
			description="Auth error test"
		)
		
		token_result = await dpo_service.create_payment_token(transaction)
		
		assert token_result["success"] is False
		assert "invalid company token" in token_result["error"].lower()
	
	@patch('aiohttp.ClientSession.post')
	async def test_insufficient_funds_error(self, mock_post, dpo_service):
		"""Test insufficient funds error handling"""
		# Mock token creation success, mobile money failure
		mock_token_response = Mock()
		mock_token_response.status = 200
		mock_token_response.text = AsyncMock(return_value="""<?xml version="1.0" encoding="utf-8"?>
			<API3G>
				<Result>000</Result>
				<TransToken>DPO_TOKEN_INSUFFICIENT</TransToken>
			</API3G>""")
		
		mock_mobile_response = Mock()
		mock_mobile_response.status = 200
		mock_mobile_response.text = AsyncMock(return_value="""<?xml version="1.0" encoding="utf-8"?>
			<API3G>
				<Result>801</Result>
				<ResultExplanation>Insufficient funds</ResultExplanation>
				<TransToken>DPO_TOKEN_INSUFFICIENT</TransToken>
			</API3G>""")
		
		mock_post.return_value.__aenter__.side_effect = [mock_token_response, mock_mobile_response]
		
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
		
		result = await dpo_service.process_payment(transaction, payment_method)
		
		assert result.success is False
		assert result.status == PaymentStatus.FAILED
		assert "insufficient funds" in result.error_message.lower()
	
	@patch('aiohttp.ClientSession.post')
	async def test_network_timeout_error(self, mock_post, dpo_service):
		"""Test network timeout error handling"""
		import asyncio
		
		mock_post.side_effect = asyncio.TimeoutError("Request timed out")
		
		transaction = PaymentTransaction(
			id="timeout_test",
			amount=Decimal("100.00"),
			currency="KES",
			description="Timeout test"
		)
		
		payment_method = PaymentMethod(
			method_type=PaymentMethodType.CARD,
			metadata={"card_number": "5123450000000008"}
		)
		
		result = await dpo_service.process_payment(transaction, payment_method)
		
		assert result.success is False
		assert result.status == PaymentStatus.FAILED
		assert "timeout" in result.error_message.lower() or "connection" in result.error_message.lower()


if __name__ == "__main__":
	pytest.main([__file__, "-v", "--cov=../dpo_integration", "--cov-report=html"])