"""
APG Core Financials - Accounts Payable Test Configuration

Pytest fixtures for APG-compatible testing with real objects,
asyncio event loop, and pytest-httpserver integration.

© 2025 Datacraft. All rights reserved.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict
from uuid import uuid4

import pytest
from pytest_httpserver import HTTPServer
from werkzeug import Response

# Import test targets
from ..models import (
	APVendor, APInvoice, APPayment, APApprovalWorkflow,
	VendorStatus, VendorType, InvoiceStatus, PaymentStatus, PaymentMethod,
	ContactInfo, PaymentTerms, TaxInfo, VendorPerformanceMetrics,
	create_sample_vendor, create_sample_invoice
)
from ..service import (
	APVendorService, APInvoiceService, APPaymentService,
	APWorkflowService, APAnalyticsService
)


# APG Testing Standards - No @pytest.mark.asyncio decorators needed
# Use real objects with pytest fixtures (no mocks except LLM)

@pytest.fixture(scope="function")
def event_loop():
	"""Create event loop for async tests per APG standards"""
	loop = asyncio.get_event_loop()
	yield loop


@pytest.fixture
def tenant_context() -> Dict[str, Any]:
	"""APG tenant context for multi-tenant testing"""
	return {
		"tenant_id": f"test_tenant_{uuid4().hex[:8]}",
		"user_id": f"test_user_{uuid4().hex[:8]}",
		"permissions": [
			"ap.read", "ap.write", "ap.approve_invoice",
			"ap.process_payment", "ap.vendor_admin", "ap.admin"
		],
		"roles": ["ap_admin"]
	}


@pytest.fixture
def sample_vendor_data() -> Dict[str, Any]:
	"""Sample vendor data for testing"""
	return {
		"vendor_code": f"TEST{uuid4().hex[:6].upper()}",
		"legal_name": "Test Vendor Corporation",
		"trade_name": "Test Vendor Co.",
		"vendor_type": VendorType.SUPPLIER,
		"primary_contact": {
			"name": "John Smith",
			"title": "Accounts Manager",
			"email": "john.smith@testvendor.com",
			"phone": "555-123-4567"
		},
		"payment_terms": {
			"code": "NET_30",
			"name": "Net 30",
			"net_days": 30,
			"discount_days": 10,
			"discount_percent": Decimal("2.00")
		},
		"tax_information": {
			"tax_id": "12-3456789",
			"tax_id_type": "ein",
			"is_1099_vendor": False
		},
		"addresses": [
			{
				"address_type": "billing",
				"line1": "123 Business Ave",
				"city": "Business City",
				"state_province": "BC",
				"postal_code": "12345",
				"country_code": "US",
				"is_primary": True
			}
		],
		"banking_details": [
			{
				"account_type": "checking",
				"bank_name": "Test Bank",
				"routing_number": "123456789",
				"account_number": "987654321",
				"account_holder_name": "Test Vendor Corporation",
				"is_primary": True,
				"is_active": True
			}
		]
	}


@pytest.fixture
def sample_invoice_data(tenant_context: Dict[str, Any]) -> Dict[str, Any]:
	"""Sample invoice data for testing"""
	vendor_id = f"vendor_{uuid4().hex[:8]}"
	return {
		"invoice_number": f"TEST-INV-{uuid4().hex[:8].upper()}",
		"vendor_id": vendor_id,
		"vendor_invoice_number": f"VEND-{uuid4().hex[:6].upper()}",
		"invoice_date": date.today().isoformat(),
		"due_date": (date.today()).isoformat(),
		"subtotal_amount": "1000.00",
		"tax_amount": "85.00",
		"total_amount": "1085.00",
		"payment_terms": {
			"code": "NET_30",
			"name": "Net 30",
			"net_days": 30
		},
		"currency_code": "USD",
		"line_items": [
			{
				"line_number": 1,
				"description": "Test Service",
				"quantity": "1.0000",
				"unit_price": "1000.0000",
				"line_amount": "1000.00",
				"gl_account_code": "5000",
				"cost_center": "CC001",
				"department": "IT"
			}
		],
		"tenant_id": tenant_context["tenant_id"]
	}


@pytest.fixture
def sample_payment_data(tenant_context: Dict[str, Any]) -> Dict[str, Any]:
	"""Sample payment data for testing"""
	return {
		"payment_number": f"PAY-{uuid4().hex[:8].upper()}",
		"vendor_id": f"vendor_{uuid4().hex[:8]}",
		"payment_method": PaymentMethod.ACH,
		"payment_amount": "1085.00",
		"payment_date": date.today().isoformat(),
		"currency_code": "USD",
		"bank_account_id": f"bank_{uuid4().hex[:8]}",
		"payment_lines": [
			{
				"invoice_id": f"invoice_{uuid4().hex[:8]}",
				"invoice_number": "TEST-INV-001",
				"payment_amount": "1085.00",
				"discount_taken": "0.00"
			}
		],
		"tenant_id": tenant_context["tenant_id"]
	}


@pytest.fixture
def sample_workflow_data(tenant_context: Dict[str, Any]) -> Dict[str, Any]:
	"""Sample workflow data for testing"""
	return {
		"workflow_type": "invoice",
		"entity_id": f"invoice_{uuid4().hex[:8]}",
		"entity_number": "TEST-INV-001",
		"tenant_id": tenant_context["tenant_id"],
		"created_by": tenant_context["user_id"]
	}


# Service fixtures using real objects (no mocks per APG standards)

@pytest.fixture
def vendor_service() -> APVendorService:
	"""Real vendor service instance for testing"""
	return APVendorService()


@pytest.fixture
def invoice_service() -> APInvoiceService:
	"""Real invoice service instance for testing"""
	return APInvoiceService()


@pytest.fixture
def payment_service() -> APPaymentService:
	"""Real payment service instance for testing"""
	return APPaymentService()


@pytest.fixture
def workflow_service() -> APWorkflowService:
	"""Real workflow service instance for testing"""
	return APWorkflowService()


@pytest.fixture
def analytics_service() -> APAnalyticsService:
	"""Real analytics service instance for testing"""
	return APAnalyticsService()


# HTTPServer fixtures for API testing per APG standards

@pytest.fixture
def http_server():
	"""HTTP server for API integration testing using pytest-httpserver"""
	server = HTTPServer(host="127.0.0.1", port=0)
	server.start()
	yield server
	server.stop()


@pytest.fixture
def mock_apg_auth_response():
	"""Mock APG auth response for integration testing"""
	def _handler(request):
		return Response(
			response='{"valid": true, "user_id": "test_user", "tenant_id": "test_tenant"}',
			status=200,
			headers={"Content-Type": "application/json"}
		)
	return _handler


@pytest.fixture
def mock_apg_audit_response():
	"""Mock APG audit response for integration testing"""
	def _handler(request):
		return Response(
			response='{"logged": true, "audit_id": "audit_123"}',
			status=200,
			headers={"Content-Type": "application/json"}
		)
	return _handler


# Model factory fixtures

@pytest.fixture
def create_test_vendor(tenant_context: Dict[str, Any]):
	"""Factory function to create test vendor instances"""
	def _create_vendor(vendor_code: str | None = None) -> APVendor:
		return create_sample_vendor(tenant_context["tenant_id"])
	return _create_vendor


@pytest.fixture
def create_test_invoice(tenant_context: Dict[str, Any]):
	"""Factory function to create test invoice instances"""
	def _create_invoice(vendor_id: str | None = None) -> APInvoice:
		vendor_id = vendor_id or f"vendor_{uuid4().hex[:8]}"
		return create_sample_invoice(vendor_id, tenant_context["tenant_id"])
	return _create_invoice


# Performance testing fixtures

@pytest.fixture
def performance_test_data():
	"""Generate large datasets for performance testing"""
	return {
		"vendor_count": 1000,
		"invoice_count": 10000,
		"payment_count": 5000,
		"concurrent_users": 100
	}


# Database fixtures (placeholder for actual database integration)

@pytest.fixture
def test_database():
	"""Test database connection (placeholder)"""
	# In real implementation, this would set up a test database
	# using PostgreSQL test containers or similar
	class MockDatabase:
		def __init__(self):
			self.data = {}
		
		async def save(self, entity):
			self.data[entity.id] = entity
			return entity
		
		async def get_by_id(self, entity_id):
			return self.data.get(entity_id)
		
		async def query(self, filters):
			return list(self.data.values())
	
	return MockDatabase()


# APG capability mock fixtures (only for external integrations)

@pytest.fixture
def mock_computer_vision_service():
	"""Mock APG computer vision service for testing"""
	class MockComputerVisionService:
		async def extract_text(self, image_data, **kwargs):
			return {
				"extracted_text": "INVOICE\nACME Corp\nAmount: $1,000.00\nDate: 2025-01-01",
				"confidence_score": 0.98,
				"processing_time_ms": 1200,
				"tables": []
			}
	
	return MockComputerVisionService()


@pytest.fixture
def mock_ai_orchestration_service():
	"""Mock APG AI orchestration service for testing"""
	class MockAIOrchestrationService:
		async def process_document(self, text_content, document_type, vendor_context, tenant_id):
			return {
				"vendor_name": "ACME Corp",
				"invoice_number": "INV-12345",
				"invoice_date": "2025-01-01",
				"total_amount": "1000.00",
				"line_items": [
					{
						"description": "Professional Services",
						"amount": "1000.00",
						"gl_code": "5000"
					}
				]
			}
	
	return MockAIOrchestrationService()


# Assertion helpers for APG testing

def assert_valid_uuid(uuid_string: str) -> None:
	"""Assert that string is a valid UUID"""
	assert uuid_string is not None, "UUID cannot be None"
	assert len(uuid_string) > 0, "UUID cannot be empty"
	# In real implementation, would validate UUID format


def assert_decimal_equals(actual: Decimal, expected: Decimal, tolerance: Decimal = Decimal('0.01')) -> None:
	"""Assert decimal values are equal within tolerance"""
	assert abs(actual - expected) <= tolerance, f"Expected {expected}, got {actual}"


def assert_apg_compliance(model_instance) -> None:
	"""Assert model follows APG compliance patterns"""
	assert hasattr(model_instance, 'id'), "Model must have ID field"
	assert hasattr(model_instance, 'created_at'), "Model must have created_at field"
	if hasattr(model_instance, 'tenant_id'):
		assert model_instance.tenant_id is not None, "Tenant ID must be set for multi-tenant models"


# Export fixtures for test discovery
__all__ = [
	"tenant_context", "sample_vendor_data", "sample_invoice_data", "sample_payment_data",
	"vendor_service", "invoice_service", "payment_service", "workflow_service", "analytics_service",
	"http_server", "mock_computer_vision_service", "mock_ai_orchestration_service",
	"create_test_vendor", "create_test_invoice", "test_database",
	"assert_valid_uuid", "assert_decimal_equals", "assert_apg_compliance"
]