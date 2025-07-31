"""
Test Configuration and Fixtures

Pytest configuration and shared fixtures for payment gateway tests.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import pytest
import tempfile
import os
from datetime import datetime, timezone
from uuid_extensions import uuid7str

from ..database import DatabaseService
from ..auth import AuthenticationService
from ..service import PaymentGatewayService
from ..models import PaymentTransaction, PaymentMethod, PaymentStatus, PaymentMethodType


@pytest.fixture(scope="session")
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


@pytest.fixture
async def temp_database():
	"""Create a temporary database for testing"""
	# Create temporary database file
	db_fd, db_path = tempfile.mkstemp(suffix='.db')
	os.close(db_fd)
	
	try:
		database_url = f"sqlite:///{db_path}"
		database_service = DatabaseService(database_url)
		await database_service.initialize()
		yield database_service
	finally:
		# Clean up
		if os.path.exists(db_path):
			os.unlink(db_path)


@pytest.fixture
async def auth_service():
	"""Create authentication service for testing"""
	auth = AuthenticationService()
	return auth


@pytest.fixture
async def payment_service(temp_database):
	"""Create payment service with test configuration"""
	config = {
		"mpesa": {
			"environment": "sandbox",
			"consumer_key": "test_consumer_key",
			"consumer_secret": "test_consumer_secret",
			"business_short_code": "174379",
			"passkey": "test_passkey",
			"callback_url": "https://example.com/callback"
		},
		"stripe": {
			"api_key": "sk_test_12345",
			"webhook_secret": "whsec_test_12345",
			"environment": "sandbox"
		},
		"paypal": {
			"client_id": "test_client_id",
			"client_secret": "test_client_secret",  
			"environment": "sandbox"
		},
		"adyen": {
			"api_key": "test_api_key",
			"merchant_account": "test_merchant",
			"environment": "test"
		}
	}
	
	service = PaymentGatewayService(config)
	service._database_service = temp_database
	await service.initialize()
	return service


@pytest.fixture
def sample_transaction():
	"""Create a sample payment transaction for testing"""
	return PaymentTransaction(
		id=uuid7str(),
		amount=10000,  # 100.00 in cents
		currency="KES",
		payment_method_type=PaymentMethodType.MPESA,
		merchant_id="test_merchant_123",
		customer_id="test_customer_456",
		description="Test payment transaction",
		metadata={"test": True, "source": "pytest"},
		status=PaymentStatus.PENDING,
		created_at=datetime.now(timezone.utc)
	)


@pytest.fixture
def sample_mpesa_payment_method():
	"""Create a sample MPESA payment method for testing"""
	return PaymentMethod(
		id=uuid7str(),
		type=PaymentMethodType.MPESA,
		details={
			"phone_number": "+254712345678",
			"account_reference": "TestRef123"
		},
		customer_id="test_customer_456",
		created_at=datetime.now(timezone.utc)
	)


@pytest.fixture
def sample_card_payment_method():
	"""Create a sample card payment method for testing"""
	return PaymentMethod(
		id=uuid7str(),
		type=PaymentMethodType.CREDIT_CARD,
		details={
			"last4": "4242",
			"brand": "visa",
			"exp_month": "12",
			"exp_year": "2025"
		},
		customer_id="test_customer_456",
		token="pm_test_12345",
		created_at=datetime.now(timezone.utc)
	)


@pytest.fixture
def test_merchant_data():
	"""Sample merchant data for testing"""
	return {
		"id": "test_merchant_123",
		"name": "Test Merchant Ltd",
		"email": "merchant@test.com",
		"country": "KE",
		"currency": "KES",
		"status": "active"
	}


@pytest.fixture
def test_api_key_data():
	"""Sample API key data for testing"""
	return {
		"name": "Test API Key",
		"permissions": [
			"payment_process",
			"payment_status", 
			"payment_capture",
			"payment_refund"
		]
	}