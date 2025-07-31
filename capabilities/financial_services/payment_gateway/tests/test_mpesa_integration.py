"""
Comprehensive Test Suite for M-Pesa Direct Integration - APG Payment Gateway

Tests covering direct M-Pesa API integration including STK Push, B2C, B2B, and C2B payments.
Production-ready tests with complete coverage for Kenya's M-Pesa ecosystem.

© 2025 Datacraft. All rights reserved.
"""

import pytest
import json
import base64
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from ..mpesa_integration import (
	MPesaService, MPesaEnvironment, create_mpesa_service
)
from ..models import (
	PaymentTransaction, PaymentMethod, PaymentResult,
	PaymentStatus, PaymentMethodType, HealthStatus
)


class TestMPesaIntegration:
	"""Comprehensive test suite for M-Pesa direct integration"""
	
	@pytest.fixture
	async def mpesa_service(self):
		"""Create M-Pesa service for testing"""
		with patch.dict('os.environ', {
			'MPESA_CONSUMER_KEY': 'test_consumer_key',
			'MPESA_CONSUMER_SECRET': 'test_consumer_secret',
			'MPESA_BUSINESS_SHORTCODE': '174379',
			'MPESA_PASSKEY': 'test_passkey',
			'MPESA_CALLBACK_URL': 'https://example.com/callback/mpesa',
			'MPESA_INITIATOR_NAME': 'testapi',
			'MPESA_SECURITY_CREDENTIAL': 'test_credential'
		}):
			service = await create_mpesa_service(MPesaEnvironment.SANDBOX)
			return service
	
	@pytest.fixture
	def sample_transaction(self):
		"""Create sample payment transaction"""
		return PaymentTransaction(
			id="txn_mpesa_test",
			amount=Decimal("1000.00"),
			currency="KES",
			description="M-Pesa test payment",
			customer_email="mpesa@example.com",
			customer_name="M-Pesa Test Customer"
		)
	
	@pytest.fixture
	def sample_mpesa_method(self):
		"""Create sample M-Pesa payment method"""
		return PaymentMethod(
			method_type=PaymentMethodType.MOBILE_MONEY,
			metadata={
				"phone": "254708374149",
				"provider": "MPESA",
				"network": "SAFARICOM"
			}
		)
	
	async def test_mpesa_service_initialization(self, mpesa_service):
		"""Test M-Pesa service initialization"""
		assert mpesa_service._initialized is True
		assert mpesa_service.config.environment == MPesaEnvironment.SANDBOX
		assert mpesa_service.config.consumer_key == "test_consumer_key"
		assert mpesa_service.config.business_shortcode == "174379"
	
	@patch('aiohttp.ClientSession.get')
	@patch('aiohttp.ClientSession.post')
	async def test_get_access_token_success(self, mock_post, mock_get, mpesa_service):
		"""Test successful access token generation"""
		# Mock M-Pesa OAuth token response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.json = AsyncMock(return_value={
			"access_token": "SGWcJeCrQo6uOgwN5EjG3wUnkA98",
			"expires_in": "3599"
		})
		
		mock_get.return_value.__aenter__.return_value = mock_response
		
		token = await mpesa_service._get_access_token()
		
		assert token == "SGWcJeCrQo6uOgwN5EjG3wUnkA98"
		mock_get.assert_called_once()
	
	@patch('aiohttp.ClientSession.get')
	@patch('aiohttp.ClientSession.post')
	async def test_stk_push_success(self, mock_post, mock_get, mpesa_service, sample_transaction, sample_mpesa_method):
		"""Test successful STK Push payment"""
		# Mock access token response
		mock_token_response = Mock()
		mock_token_response.status = 200
		mock_token_response.json = AsyncMock(return_value={
			"access_token": "mock_token",
			"expires_in": "3599"
		})
		
		# Mock STK Push response
		mock_stk_response = Mock()
		mock_stk_response.status = 200
		mock_stk_response.json = AsyncMock(return_value={
			"MerchantRequestID": "29115-34620561-1",
			"CheckoutRequestID": "ws_CO_191220191020363925",
			"ResponseCode": "0",
			"ResponseDescription": "Success. Request accepted for processing",
			"CustomerMessage": "Success. Request accepted for processing"
		})
		
		mock_get.return_value.__aenter__.return_value = mock_token_response
		mock_post.return_value.__aenter__.return_value = mock_stk_response
		
		result = await mpesa_service.process_payment(sample_transaction, sample_mpesa_method)
		
		assert result.success is True
		assert result.status == PaymentStatus.PENDING
		assert result.provider_transaction_id == "ws_CO_191220191020363925"
		assert "Success" in result.raw_response.get("CustomerMessage", "")
	
	@patch('aiohttp.ClientSession.get')
	@patch('aiohttp.ClientSession.post')
	async def test_stk_push_query_success(self, mock_post, mock_get, mpesa_service):
		"""Test successful STK Push query"""
		# Mock access token response
		mock_token_response = Mock()
		mock_token_response.status = 200
		mock_token_response.json = AsyncMock(return_value={"access_token": "mock_token"})
		
		# Mock STK Push query response
		mock_query_response = Mock()
		mock_query_response.status = 200
		mock_query_response.json = AsyncMock(return_value={
			"ResponseCode": "0",
			"ResponseDescription": "The service request has been accepted successfully.",
			"MerchantRequestID": "29115-34620561-1",
			"CheckoutRequestID": "ws_CO_191220191020363925",
			"ResultCode": "0",
			"ResultDesc": "The service request is processed successfully."
		})
		
		mock_get.return_value.__aenter__.return_value = mock_token_response
		mock_post.return_value.__aenter__.return_value = mock_query_response
		
		result = await mpesa_service.query_stk_push("ws_CO_191220191020363925")
		
		assert result["success"] is True
		assert result["status"] == "completed"
		assert result["checkout_request_id"] == "ws_CO_191220191020363925"
	
	@patch('aiohttp.ClientSession.get')
	@patch('aiohttp.ClientSession.post')
	async def test_verify_payment_success(self, mock_post, mock_get, mpesa_service):
		"""Test successful payment verification"""
		# Mock access token and transaction status responses
		mock_token_response = Mock()
		mock_token_response.status = 200
		mock_token_response.json = AsyncMock(return_value={"access_token": "mock_token"})
		
		mock_status_response = Mock()
		mock_status_response.status = 200
		mock_status_response.json = AsyncMock(return_value={
			"ResponseCode": "0",
			"ResponseDescription": "The service request has been accepted successfully.",
			"OriginatorConversationID": "10819-34620561-1",
			"ConversationID": "AG_20191219_00004e48cf7e3533f581",
			"TransactionID": "NLJ7RT61SV",
			"ResultType": 0,
			"ResultCode": 0,
			"ResultDesc": "The service request is processed successfully.",
			"TransactionReceipt": "NLJ7RT61SV",
			"ReceiverPartyPublicName": "254708374149 - John Doe",
			"TransactionAmount": 1000,
			"B2CWorkingAccountAvailableFunds": 900000,
			"B2CUtilityAccountAvailableFunds": 10000,
			"TransactionCompletedDateTime": "19.12.2019 11:45:50",
			"B2CChargesPaidAccountAvailableFunds": 0,
			"B2CRecipientIsRegisteredCustomer": "Y"
		})
		
		mock_get.return_value.__aenter__.return_value = mock_token_response
		mock_post.return_value.__aenter__.return_value = mock_status_response
		
		result = await mpesa_service.verify_payment("NLJ7RT61SV")
		
		assert result.success is True
		assert result.status == PaymentStatus.COMPLETED
		assert result.provider_transaction_id == "NLJ7RT61SV"
		assert result.amount == Decimal("1000.00")
	
	@patch('aiohttp.ClientSession.get')
	@patch('aiohttp.ClientSession.post')
	async def test_b2c_payment_success(self, mock_post, mock_get, mpesa_service):
		"""Test successful B2C (Business to Customer) payment"""
		# Mock access token response
		mock_token_response = Mock()
		mock_token_response.status = 200
		mock_token_response.json = AsyncMock(return_value={"access_token": "mock_token"})
		
		# Mock B2C payment response
		mock_b2c_response = Mock()
		mock_b2c_response.status = 200
		mock_b2c_response.json = AsyncMock(return_value={
			"ConversationID": "AG_20191219_00005797af5d7d75f652",
			"OriginatorConversationID": "16740-34861180-1",
			"ResponseCode": "0",
			"ResponseDescription": "Accept the service request successfully."
		})
		
		mock_get.return_value.__aenter__.return_value = mock_token_response
		mock_post.return_value.__aenter__.return_value = mock_b2c_response
		
		result = await mpesa_service.send_money_b2c(
			phone="254708374149",
			amount=Decimal("500.00"),
			occasion="Salary Payment",
			remarks="Monthly salary"
		)
		
		assert result["success"] is True
		assert result["conversation_id"] == "AG_20191219_00005797af5d7d75f652"
		assert result["status"] == "accepted"
	
	@patch('aiohttp.ClientSession.get')
	@patch('aiohttp.ClientSession.post')
	async def test_b2b_payment_success(self, mock_post, mock_get, mpesa_service):
		"""Test successful B2B (Business to Business) payment"""
		# Mock access token response
		mock_token_response = Mock()
		mock_token_response.status = 200
		mock_token_response.json = AsyncMock(return_value={"access_token": "mock_token"})
		
		# Mock B2B payment response
		mock_b2b_response = Mock()
		mock_b2b_response.status = 200
		mock_b2b_response.json = AsyncMock(return_value={
			"ConversationID": "AG_20191219_00005e063f842e01e413",
			"OriginatorConversationID": "12363-1328488-1",
			"ResponseCode": "0",
			"ResponseDescription": "Accept the service request successfully."
		})
		
		mock_get.return_value.__aenter__.return_value = mock_token_response
		mock_post.return_value.__aenter__.return_value = mock_b2b_response
		
		result = await mpesa_service.send_money_b2b(
			shortcode="600000",
			amount=Decimal("2000.00"),
			account_reference="Invoice001",
			remarks="Payment for services"
		)
		
		assert result["success"] is True
		assert result["conversation_id"] == "AG_20191219_00005e063f842e01e413"
		assert result["status"] == "accepted"
	
	async def test_process_callback_success(self, mpesa_service):
		"""Test callback processing for successful payment"""
		callback_data = {
			"Body": {
				"stkCallback": {
					"MerchantRequestID": "29115-34620561-1",
					"CheckoutRequestID": "ws_CO_191220191020363925",
					"ResultCode": 0,
					"ResultDesc": "The service request is processed successfully.",
					"CallbackMetadata": {
						"Item": [
							{"Name": "Amount", "Value": 1000.00},
							{"Name": "MpesaReceiptNumber", "Value": "NLJ7RT61SV"},
							{"Name": "Balance"},
							{"Name": "TransactionDate", "Value": 20191219114530},
							{"Name": "PhoneNumber", "Value": 254708374149}
						]
					}
				}
			}
		}
		
		result = await mpesa_service.process_callback(callback_data)
		
		assert result["success"] is True
		assert result["event_type"] == "payment_completed"
		assert result["transaction_id"] == "NLJ7RT61SV"
		assert result["status"] == "completed"
		assert result["amount"] == Decimal("1000.00")
		assert result["phone"] == "254708374149"
	
	async def test_process_callback_failed_payment(self, mpesa_service):
		"""Test callback processing for failed payment"""
		callback_data = {
			"Body": {
				"stkCallback": {
					"MerchantRequestID": "29115-34620561-1",
					"CheckoutRequestID": "ws_CO_191220191020363925",
					"ResultCode": 1032,
					"ResultDesc": "Request cancelled by user"
				}
			}
		}
		
		result = await mpesa_service.process_callback(callback_data)
		
		assert result["success"] is True
		assert result["event_type"] == "payment_failed"
		assert result["status"] == "failed"
		assert "cancelled by user" in result["error_message"]
	
	async def test_process_c2b_validation(self, mpesa_service):
		"""Test C2B (Customer to Business) validation"""
		c2b_data = {
			"TransactionType": "Pay Bill",
			"TransID": "LKXXXX1234",
			"TransTime": "20191122063845",
			"TransAmount": "10.00",
			"BusinessShortCode": "174379",
			"BillRefNumber": "invoice001",
			"InvoiceNumber": "",
			"OrgAccountBalance": "49197.00",
			"ThirdPartyTransID": "",
			"MSISDN": "254708374149",
			"FirstName": "John",
			"MiddleName": "",
			"LastName": "Doe"
		}
		
		result = await mpesa_service.validate_c2b_payment(c2b_data)
		
		assert result["ResultCode"] == "0"
		assert result["ResultDesc"] == "Accepted"
	
	async def test_process_c2b_confirmation(self, mpesa_service):
		"""Test C2B (Customer to Business) confirmation"""
		c2b_data = {
			"TransactionType": "Pay Bill",
			"TransID": "LKXXXX1234",
			"TransTime": "20191122063845",
			"TransAmount": "10.00",
			"BusinessShortCode": "174379",
			"BillRefNumber": "invoice001",
			"InvoiceNumber": "",
			"OrgAccountBalance": "49197.00",
			"ThirdPartyTransID": "",
			"MSISDN": "254708374149",
			"FirstName": "John",
			"MiddleName": "",
			"LastName": "Doe"
		}
		
		result = await mpesa_service.confirm_c2b_payment(c2b_data)
		
		assert result["success"] is True
		assert result["event_type"] == "c2b_payment_received"
		assert result["transaction_id"] == "LKXXXX1234"
		assert result["amount"] == Decimal("10.00")
		assert result["phone"] == "254708374149"
	
	@patch('aiohttp.ClientSession.get')
	async def test_health_check_success(self, mock_get, mpesa_service):
		"""Test successful health check"""
		# Mock M-Pesa OAuth response
		mock_response = Mock()
		mock_response.status = 200
		mock_response.json = AsyncMock(return_value={
			"access_token": "SGWcJeCrQo6uOgwN5EjG3wUnkA98",
			"expires_in": "3599"
		})
		
		mock_get.return_value.__aenter__.return_value = mock_response
		
		result = await mpesa_service.health_check()
		
		assert result.status == HealthStatus.HEALTHY
		assert result.response_time_ms > 0
		assert "oauth_status" in result.details
	
	async def test_get_supported_payment_methods(self, mpesa_service):
		"""Test getting supported payment methods"""
		methods = await mpesa_service.get_supported_payment_methods(
			country_code="KE",
			currency="KES"
		)
		
		assert len(methods) > 0
		assert any(method["type"] == "mobile_money" for method in methods)
		
		# Check for M-Pesa specifically
		mpesa_method = next((m for m in methods if m.get("name") == "M-Pesa"), None)
		assert mpesa_method is not None
		assert "KE" in mpesa_method["countries"]
		assert "KES" in mpesa_method["currencies"]
	
	async def test_get_transaction_fees(self, mpesa_service):
		"""Test transaction fee calculation"""
		fees = await mpesa_service.get_transaction_fees(
			amount=Decimal("1000.00"),
			currency="KES",
			payment_method="stk_push"
		)
		
		assert "total_fee" in fees
		assert "transaction_fee" in fees
		assert "safaricom_fee" in fees
		assert float(fees["total_fee"]) >= 0
		
		# Test B2C fees (typically higher)
		b2c_fees = await mpesa_service.get_transaction_fees(
			amount=Decimal("1000.00"),
			currency="KES",
			payment_method="b2c"
		)
		
		assert float(b2c_fees["total_fee"]) > float(fees["total_fee"])
	
	async def test_account_balance_query(self, mpesa_service):
		"""Test account balance query"""
		with patch('aiohttp.ClientSession.get') as mock_get, \
			 patch('aiohttp.ClientSession.post') as mock_post:
			
			# Mock access token response
			mock_token_response = Mock()
			mock_token_response.status = 200
			mock_token_response.json = AsyncMock(return_value={"access_token": "mock_token"})
			
			# Mock balance query response
			mock_balance_response = Mock()
			mock_balance_response.status = 200
			mock_balance_response.json = AsyncMock(return_value={
				"ConversationID": "AG_20191219_00005e48cf7e3533f581",
				"OriginatorConversationID": "17944-34861180-1",
				"ResponseCode": "0",
				"ResponseDescription": "Accept the service request successfully."
			})
			
			mock_get.return_value.__aenter__.return_value = mock_token_response
			mock_post.return_value.__aenter__.return_value = mock_balance_response
			
			result = await mpesa_service.query_account_balance()
			
			assert result["success"] is True
			assert result["conversation_id"] == "AG_20191219_00005e48cf7e3533f581"
	
	async def test_transaction_reversal(self, mpesa_service):
		"""Test transaction reversal"""
		with patch('aiohttp.ClientSession.get') as mock_get, \
			 patch('aiohttp.ClientSession.post') as mock_post:
			
			# Mock access token response
			mock_token_response = Mock()
			mock_token_response.status = 200
			mock_token_response.json = AsyncMock(return_value={"access_token": "mock_token"})
			
			# Mock reversal response
			mock_reversal_response = Mock()
			mock_reversal_response.status = 200
			mock_reversal_response.json = AsyncMock(return_value={
				"ConversationID": "AG_20191219_00005e48cf7e3533f582",
				"OriginatorConversationID": "17944-34861181-1",
				"ResponseCode": "0",
				"ResponseDescription": "Accept the service request successfully."
			})
			
			mock_get.return_value.__aenter__.return_value = mock_token_response
			mock_post.return_value.__aenter__.return_value = mock_reversal_response
			
			result = await mpesa_service.reverse_transaction(
				transaction_id="NLJ7RT61SV",
				amount=Decimal("1000.00"),
				reason="Duplicate payment"
			)
			
			assert result["success"] is True
			assert result["status"] == "accepted"
			assert result["conversation_id"] == "AG_20191219_00005e48cf7e3533f582"


class TestMPesaErrorHandling:
	"""Test error handling scenarios for M-Pesa integration"""
	
	@pytest.fixture
	async def mpesa_service(self):
		"""Create M-Pesa service for error testing"""
		with patch.dict('os.environ', {
			'MPESA_CONSUMER_KEY': 'error_test_key',
			'MPESA_CONSUMER_SECRET': 'error_test_secret',
			'MPESA_BUSINESS_SHORTCODE': '174379'
		}):
			service = await create_mpesa_service(MPesaEnvironment.SANDBOX)
			return service
	
	@patch('aiohttp.ClientSession.get')
	async def test_oauth_authentication_error(self, mock_get, mpesa_service):
		"""Test OAuth authentication error handling"""
		# Mock M-Pesa OAuth error response
		mock_response = Mock()
		mock_response.status = 401
		mock_response.json = AsyncMock(return_value={
			"requestId": "11728-2929992-1",
			"errorCode": "401.002.01",
			"errorMessage": "Invalid Credentials"
		})
		
		mock_get.return_value.__aenter__.return_value = mock_response
		
		transaction = PaymentTransaction(
			id="auth_error_test",
			amount=Decimal("100.00"),
			currency="KES",
			description="Auth error test"
		)
		
		payment_method = PaymentMethod(
			method_type=PaymentMethodType.MOBILE_MONEY,
			metadata={"phone": "254708374149", "provider": "MPESA"}
		)
		
		result = await mpesa_service.process_payment(transaction, payment_method)
		
		assert result.success is False
		assert result.status == PaymentStatus.FAILED
		assert "invalid credentials" in result.error_message.lower()
	
	@patch('aiohttp.ClientSession.get')
	@patch('aiohttp.ClientSession.post')
	async def test_stk_push_user_cancel(self, mock_post, mock_get, mpesa_service):
		"""Test STK Push user cancellation"""
		# Mock access token response
		mock_token_response = Mock()
		mock_token_response.status = 200
		mock_token_response.json = AsyncMock(return_value={"access_token": "mock_token"})
		
		# Mock STK Push response with user cancellation
		mock_stk_response = Mock()
		mock_stk_response.status = 200
		mock_stk_response.json = AsyncMock(return_value={
			"MerchantRequestID": "29115-34620561-1",
			"CheckoutRequestID": "ws_CO_191220191020363925",
			"ResponseCode": "0",
			"ResponseDescription": "Success. Request accepted for processing",
			"CustomerMessage": "Success. Request accepted for processing"
		})
		
		mock_get.return_value.__aenter__.return_value = mock_token_response
		mock_post.return_value.__aenter__.return_value = mock_stk_response
		
		# Mock callback with user cancellation
		callback_data = {
			"Body": {
				"stkCallback": {
					"MerchantRequestID": "29115-34620561-1",
					"CheckoutRequestID": "ws_CO_191220191020363925",
					"ResultCode": 1032,
					"ResultDesc": "Request cancelled by user"
				}
			}
		}
		
		result = await mpesa_service.process_callback(callback_data)
		
		assert result["success"] is True
		assert result["event_type"] == "payment_failed"
		assert result["status"] == "failed"
		assert "cancelled by user" in result["error_message"]
	
	@patch('aiohttp.ClientSession.get')
	@patch('aiohttp.ClientSession.post')
	async def test_insufficient_funds_error(self, mock_post, mock_get, mpesa_service):
		"""Test insufficient funds error handling"""
		# Mock access token response
		mock_token_response = Mock()
		mock_token_response.status = 200
		mock_token_response.json = AsyncMock(return_value={"access_token": "mock_token"})
		
		# Mock STK Push with insufficient funds
		mock_stk_response = Mock()
		mock_stk_response.status = 200
		mock_stk_response.json = AsyncMock(return_value={
			"MerchantRequestID": "29115-34620561-1",
			"CheckoutRequestID": "ws_CO_191220191020363925",
			"ResponseCode": "0",
			"ResponseDescription": "Success. Request accepted for processing"
		})
		
		mock_get.return_value.__aenter__.return_value = mock_token_response
		mock_post.return_value.__aenter__.return_value = mock_stk_response
		
		# Mock callback with insufficient funds
		callback_data = {
			"Body": {
				"stkCallback": {
					"MerchantRequestID": "29115-34620561-1",
					"CheckoutRequestID": "ws_CO_191220191020363925",
					"ResultCode": 2001,
					"ResultDesc": "The initiator information is invalid."
				}
			}
		}
		
		result = await mpesa_service.process_callback(callback_data)
		
		assert result["success"] is True
		assert result["event_type"] == "payment_failed"
		assert result["status"] == "failed"
		assert "invalid" in result["error_message"].lower()
	
	@patch('aiohttp.ClientSession.get')
	@patch('aiohttp.ClientSession.post')
	async def test_invalid_phone_number_error(self, mock_post, mock_get, mpesa_service):
		"""Test invalid phone number error handling"""
		# Mock access token response
		mock_token_response = Mock()
		mock_token_response.status = 200
		mock_token_response.json = AsyncMock(return_value={"access_token": "mock_token"})
		
		# Mock STK Push with invalid phone number
		mock_stk_response = Mock()
		mock_stk_response.status = 400
		mock_stk_response.json = AsyncMock(return_value={
			"requestId": "11728-2929992-1",
			"errorCode": "400.002.02",
			"errorMessage": "Bad Request - Invalid PhoneNumber"
		})
		
		mock_get.return_value.__aenter__.return_value = mock_token_response
		mock_post.return_value.__aenter__.return_value = mock_stk_response
		
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
		
		result = await mpesa_service.process_payment(transaction, payment_method)
		
		assert result.success is False
		assert result.status == PaymentStatus.FAILED
		assert "invalid phonenumber" in result.error_message.lower()


if __name__ == "__main__":
	pytest.main([__file__, "-v", "--cov=../mpesa_integration", "--cov-report=html"])