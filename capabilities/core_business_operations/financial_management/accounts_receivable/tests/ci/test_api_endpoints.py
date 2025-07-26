"""
APG Accounts Receivable - API Endpoints Tests
Unit tests for FastAPI routes with APG authentication

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any
import json

from uuid_extensions import uuid7str

# Import the FastAPI app and dependencies
from ..api_endpoints import app
from ..models import (
	ARCustomer, ARInvoice, ARPayment, ARCollectionActivity,
	ARCustomerType, ARCustomerStatus, ARInvoiceStatus, ARPaymentStatus
)
from ..service import (
	ARCustomerService, ARInvoiceService, ARCollectionsService,
	ARCashApplicationService, ARAnalyticsService
)


# Test client
client = TestClient(app)


class TestAPIAuthentication:
	"""Test API authentication and authorization."""
	
	def test_missing_authentication(self):
		"""Test API call without authentication."""
		response = client.get("/api/ar/customers")
		assert response.status_code == 401
		assert "authentication" in response.json()["detail"].lower()
	
	def test_invalid_token(self):
		"""Test API call with invalid token."""
		headers = {"Authorization": "Bearer invalid_token"}
		response = client.get("/api/ar/customers", headers=headers)
		assert response.status_code == 401
		assert "invalid" in response.json()["detail"].lower()
	
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_current_user')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_tenant_context')
	def test_valid_authentication(self, mock_tenant, mock_user):
		"""Test API call with valid authentication."""
		# Mock authentication
		mock_user.return_value = {
			'user_id': uuid7str(),
			'email': 'test@example.com',
			'permissions': ['ar:read', 'ar:write']
		}
		mock_tenant.return_value = {
			'tenant_id': uuid7str(),
			'tenant_name': 'Test Tenant'
		}
		
		# Mock service dependencies
		with patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_customer_service') as mock_service:
			mock_service.return_value = Mock()
			mock_service.return_value.get_customers_filtered = AsyncMock(return_value=[])
			
			headers = {"Authorization": "Bearer valid_token"}
			response = client.get("/api/ar/customers", headers=headers)
			assert response.status_code == 200


class TestCustomerEndpoints:
	"""Test customer management API endpoints."""
	
	@pytest.fixture
	def auth_headers(self):
		"""Create authentication headers for testing."""
		return {"Authorization": "Bearer test_token"}
	
	@pytest.fixture
	def sample_customer_data(self):
		"""Create sample customer data for testing."""
		return {
			'customer_code': 'TEST001',
			'legal_name': 'Test Customer Corp',
			'customer_type': 'CORPORATION',
			'status': 'ACTIVE',
			'credit_limit': 25000.00,
			'payment_terms_days': 30,
			'contact_email': 'billing@testcorp.com',
			'contact_phone': '+1-555-123-4567'
		}
	
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_current_user')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_tenant_context')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_customer_service')
	def test_create_customer_success(self, mock_service, mock_tenant, mock_user, auth_headers, sample_customer_data):
		"""Test successful customer creation."""
		# Mock authentication
		mock_user.return_value = {'user_id': uuid7str(), 'permissions': ['ar:write']}
		mock_tenant.return_value = {'tenant_id': uuid7str()}
		
		# Mock customer service
		mock_customer = ARCustomer(
			id=uuid7str(),
			tenant_id=mock_tenant.return_value['tenant_id'],
			customer_code=sample_customer_data['customer_code'],
			legal_name=sample_customer_data['legal_name'],
			customer_type=ARCustomerType.CORPORATION,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal(str(sample_customer_data['credit_limit'])),
			payment_terms_days=sample_customer_data['payment_terms_days'],
			contact_email=sample_customer_data['contact_email'],
			contact_phone=sample_customer_data['contact_phone'],
			created_by=mock_user.return_value['user_id'],
			updated_by=mock_user.return_value['user_id']
		)
		
		mock_service.return_value = Mock()
		mock_service.return_value.create_customer = AsyncMock(return_value=mock_customer)
		
		response = client.post(
			"/api/ar/customers",
			json=sample_customer_data,
			headers=auth_headers
		)
		
		assert response.status_code == 201
		data = response.json()
		assert data['data']['customer_code'] == sample_customer_data['customer_code']
		assert data['data']['legal_name'] == sample_customer_data['legal_name']
		assert data['data']['customer_type'] == 'CORPORATION'
	
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_current_user')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_tenant_context')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_customer_service')
	def test_get_customer_success(self, mock_service, mock_tenant, mock_user, auth_headers):
		"""Test successful customer retrieval."""
		customer_id = uuid7str()
		
		# Mock authentication
		mock_user.return_value = {'user_id': uuid7str(), 'permissions': ['ar:read']}
		mock_tenant.return_value = {'tenant_id': uuid7str()}
		
		# Mock customer service
		mock_customer = ARCustomer(
			id=customer_id,
			tenant_id=mock_tenant.return_value['tenant_id'],
			customer_code='EXISTING001',
			legal_name='Existing Customer',
			customer_type=ARCustomerType.INDIVIDUAL,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal('10000.00'),
			payment_terms_days=30,
			created_by=mock_user.return_value['user_id'],
			updated_by=mock_user.return_value['user_id']
		)
		
		mock_service.return_value = Mock()
		mock_service.return_value.get_customer = AsyncMock(return_value=mock_customer)
		
		response = client.get(f"/api/ar/customers/{customer_id}", headers=auth_headers)
		
		assert response.status_code == 200
		data = response.json()
		assert data['data']['id'] == customer_id
		assert data['data']['customer_code'] == 'EXISTING001'
	
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_current_user')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_tenant_context')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_customer_service')
	def test_get_customer_not_found(self, mock_service, mock_tenant, mock_user, auth_headers):
		"""Test customer retrieval when customer not found."""
		customer_id = uuid7str()
		
		# Mock authentication
		mock_user.return_value = {'user_id': uuid7str(), 'permissions': ['ar:read']}
		mock_tenant.return_value = {'tenant_id': uuid7str()}
		
		# Mock customer service - customer not found
		mock_service.return_value = Mock()
		mock_service.return_value.get_customer = AsyncMock(return_value=None)
		
		response = client.get(f"/api/ar/customers/{customer_id}", headers=auth_headers)
		
		assert response.status_code == 404
		assert "not found" in response.json()["detail"].lower()
	
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_current_user')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_tenant_context')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_customer_service')
	def test_list_customers_with_filters(self, mock_service, mock_tenant, mock_user, auth_headers):
		"""Test customer listing with filters."""
		# Mock authentication
		mock_user.return_value = {'user_id': uuid7str(), 'permissions': ['ar:read']}
		mock_tenant.return_value = {'tenant_id': uuid7str()}
		
		# Mock customer service
		mock_customers = [
			ARCustomer(
				id=uuid7str(),
				tenant_id=mock_tenant.return_value['tenant_id'],
				customer_code=f'CORP{i:03d}',
				legal_name=f'Corporation {i}',
				customer_type=ARCustomerType.CORPORATION,
				status=ARCustomerStatus.ACTIVE,
				credit_limit=Decimal('50000.00'),
				payment_terms_days=30,
				created_by=mock_user.return_value['user_id'],
				updated_by=mock_user.return_value['user_id']
			) for i in range(1, 4)
		]
		
		mock_service.return_value = Mock()
		mock_service.return_value.get_customers_filtered = AsyncMock(return_value=mock_customers)
		
		response = client.get(
			"/api/ar/customers?customer_type=CORPORATION&status=ACTIVE&page=1&per_page=10",
			headers=auth_headers
		)
		
		assert response.status_code == 200
		data = response.json()
		assert len(data['data']) == 3
		assert all(customer['customer_type'] == 'CORPORATION' for customer in data['data'])


class TestInvoiceEndpoints:
	"""Test invoice management API endpoints."""
	
	@pytest.fixture
	def auth_headers(self):
		"""Create authentication headers for testing."""
		return {"Authorization": "Bearer test_token"}
	
	@pytest.fixture
	def sample_invoice_data(self):
		"""Create sample invoice data for testing."""
		return {
			'customer_id': uuid7str(),
			'invoice_number': 'INV-TEST-001',
			'invoice_date': str(date.today()),
			'due_date': str(date.today() + timedelta(days=30)),
			'total_amount': 5000.00,
			'currency_code': 'USD',
			'description': 'Test invoice for services'
		}
	
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_current_user')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_tenant_context')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_invoice_service')
	def test_create_invoice_success(self, mock_service, mock_tenant, mock_user, auth_headers, sample_invoice_data):
		"""Test successful invoice creation."""
		# Mock authentication
		mock_user.return_value = {'user_id': uuid7str(), 'permissions': ['ar:write']}
		mock_tenant.return_value = {'tenant_id': uuid7str()}
		
		# Mock invoice service
		mock_invoice = ARInvoice(
			id=uuid7str(),
			tenant_id=mock_tenant.return_value['tenant_id'],
			customer_id=sample_invoice_data['customer_id'],
			invoice_number=sample_invoice_data['invoice_number'],
			invoice_date=date.fromisoformat(sample_invoice_data['invoice_date']),
			due_date=date.fromisoformat(sample_invoice_data['due_date']),
			total_amount=Decimal(str(sample_invoice_data['total_amount'])),
			outstanding_amount=Decimal(str(sample_invoice_data['total_amount'])),
			status=ARInvoiceStatus.DRAFT,
			currency_code=sample_invoice_data['currency_code'],
			description=sample_invoice_data['description'],
			created_by=mock_user.return_value['user_id'],
			updated_by=mock_user.return_value['user_id']
		)
		
		mock_service.return_value = Mock()
		mock_service.return_value.create_invoice = AsyncMock(return_value=mock_invoice)
		
		response = client.post(
			"/api/ar/invoices",
			json=sample_invoice_data,
			headers=auth_headers
		)
		
		assert response.status_code == 201
		data = response.json()
		assert data['data']['invoice_number'] == sample_invoice_data['invoice_number']
		assert data['data']['total_amount'] == sample_invoice_data['total_amount']
		assert data['data']['status'] == 'DRAFT'
	
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_current_user')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_tenant_context')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_invoice_service')
	def test_predict_payment_date(self, mock_service, mock_tenant, mock_user, auth_headers):
		"""Test AI payment date prediction."""
		invoice_id = uuid7str()
		
		# Mock authentication
		mock_user.return_value = {'user_id': uuid7str(), 'permissions': ['ar:read']}
		mock_tenant.return_value = {'tenant_id': uuid7str()}
		
		# Mock invoice service
		mock_prediction = {
			'predicted_payment_date': str(date.today() + timedelta(days=32)),
			'confidence_level': 0.85,
			'factors': [
				'Customer has good payment history',
				'Invoice amount within normal range',
				'No seasonal payment delays expected'
			]
		}
		
		mock_service.return_value = Mock()
		mock_service.return_value.predict_payment_date = AsyncMock(return_value=mock_prediction)
		
		response = client.post(f"/api/ar/invoices/{invoice_id}/predict-payment", headers=auth_headers)
		
		assert response.status_code == 200
		data = response.json()
		assert 'predicted_payment_date' in data['data']
		assert data['data']['confidence_level'] == 0.85
		assert len(data['data']['factors']) == 3


class TestCollectionsEndpoints:
	"""Test collections management API endpoints."""
	
	@pytest.fixture
	def auth_headers(self):
		"""Create authentication headers for testing."""
		return {"Authorization": "Bearer test_token"}
	
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_current_user')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_tenant_context')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_collections_service')
	def test_send_payment_reminder(self, mock_service, mock_tenant, mock_user, auth_headers):
		"""Test sending payment reminder."""
		invoice_id = uuid7str()
		
		# Mock authentication
		mock_user.return_value = {'user_id': uuid7str(), 'permissions': ['ar:write']}
		mock_tenant.return_value = {'tenant_id': uuid7str()}
		
		# Mock collections service
		mock_service.return_value = Mock()
		mock_service.return_value.send_payment_reminder = AsyncMock(return_value=True)
		
		response = client.post(f"/api/ar/collections/{invoice_id}/remind", headers=auth_headers)
		
		assert response.status_code == 200
		data = response.json()
		assert data['data']['success'] is True
		assert 'reminder sent' in data['data']['message'].lower()
	
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_current_user')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_tenant_context')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_collections_ai_service')
	def test_optimize_collections_strategy(self, mock_service, mock_tenant, mock_user, auth_headers):
		"""Test AI collections strategy optimization."""
		# Mock authentication
		mock_user.return_value = {'user_id': uuid7str(), 'permissions': ['ar:read', 'ar:ai']}
		mock_tenant.return_value = {'tenant_id': uuid7str()}
		
		# Mock optimization request
		optimization_request = {
			'optimization_scope': 'batch',
			'customer_ids': [uuid7str(), uuid7str()],
			'scenario_type': 'realistic',
			'include_ai_recommendations': True,
			'generate_campaign_plan': True
		}
		
		# Mock optimization response
		mock_optimization_result = {
			'total_customers': 2,
			'overall_success_probability': 0.72,
			'total_target_amount': Decimal('15000.00'),
			'estimated_collection_days': 14,
			'strategy_breakdown': {
				'EMAIL_REMINDER': 1,
				'PHONE_CALL': 1
			},
			'customer_strategies': [
				{
					'customer_id': optimization_request['customer_ids'][0],
					'customer_name': 'Test Customer 1',
					'customer_code': 'TEST001',
					'overdue_amount': Decimal('8000.00'),
					'recommended_strategy': 'EMAIL_REMINDER',
					'contact_method': 'EMAIL',
					'success_probability': 0.75,
					'priority': 'MEDIUM'
				}
			],
			'insights': ['Customers respond well to email communication'],
			'recommendations': ['Send reminders during business hours']
		}
		
		mock_service.return_value = Mock()
		mock_service.return_value.batch_optimize_strategies = AsyncMock(return_value=mock_optimization_result)
		
		response = client.post(
			"/api/ar/collections/optimize",
			json=optimization_request,
			headers=auth_headers
		)
		
		assert response.status_code == 200
		data = response.json()
		assert data['data']['total_customers'] == 2
		assert data['data']['overall_success_probability'] == 0.72
		assert len(data['data']['customer_strategies']) == 1


class TestAnalyticsEndpoints:
	"""Test analytics API endpoints."""
	
	@pytest.fixture
	def auth_headers(self):
		"""Create authentication headers for testing."""
		return {"Authorization": "Bearer test_token"}
	
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_current_user')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_tenant_context')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_analytics_service')
	def test_get_dashboard_metrics(self, mock_service, mock_tenant, mock_user, auth_headers):
		"""Test AR dashboard metrics retrieval."""
		# Mock authentication
		mock_user.return_value = {'user_id': uuid7str(), 'permissions': ['ar:read']}
		mock_tenant.return_value = {'tenant_id': uuid7str()}
		
		# Mock analytics service
		mock_metrics = {
			'total_ar_balance': Decimal('450000.00'),
			'overdue_amount': Decimal('125000.00'),
			'current_month_sales': Decimal('85000.00'),
			'current_month_collections': Decimal('92000.00'),
			'total_customers': 156,
			'active_customers': 142,
			'overdue_customers': 28,
			'average_days_to_pay': 31.5,
			'collection_effectiveness_index': 0.85,
			'days_sales_outstanding': 38.2,
			'ai_assessments_today': 8,
			'ai_collection_recommendations': 15
		}
		
		mock_service.return_value = Mock()
		mock_service.return_value.get_ar_dashboard_metrics = AsyncMock(return_value=mock_metrics)
		
		response = client.get("/api/ar/analytics/dashboard", headers=auth_headers)
		
		assert response.status_code == 200
		data = response.json()
		assert data['data']['total_customers'] == 156
		assert data['data']['collection_effectiveness_index'] == 0.85
		assert data['data']['ai_assessments_today'] == 8
	
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_current_user')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_tenant_context')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_cashflow_service')
	def test_generate_cashflow_forecast(self, mock_service, mock_tenant, mock_user, auth_headers):
		"""Test AI cash flow forecast generation."""
		# Mock authentication
		mock_user.return_value = {'user_id': uuid7str(), 'permissions': ['ar:read', 'ar:ai']}
		mock_tenant.return_value = {'tenant_id': uuid7str()}
		
		# Mock forecast request
		forecast_request = {
			'forecast_start_date': str(date.today()),
			'forecast_end_date': str(date.today() + timedelta(days=30)),
			'forecast_period': 'daily',
			'scenario_type': 'realistic',
			'include_seasonal_trends': True,
			'include_external_factors': True,
			'confidence_level': 0.95
		}
		
		# Mock forecast response
		mock_forecast_points = []
		for i in range(30):
			mock_forecast_points.append({
				'forecast_date': str(date.today() + timedelta(days=i)),
				'expected_collections': float(Decimal('2500.00')),
				'invoice_receipts': float(Decimal('1800.00')),
				'overdue_collections': float(Decimal('700.00')),
				'total_cash_flow': float(Decimal('5000.00')),
				'confidence_interval_lower': float(Decimal('4500.00')),
				'confidence_interval_upper': float(Decimal('5500.00'))
			})
		
		mock_forecast_result = {
			'forecast_id': uuid7str(),
			'tenant_id': mock_tenant.return_value['tenant_id'],
			'scenario_type': 'realistic',
			'forecast_points': mock_forecast_points,
			'overall_accuracy': 0.92,
			'model_confidence': 0.88,
			'seasonal_factors': ['Month-end payment patterns'],
			'risk_factors': ['Economic uncertainty'],
			'insights': ['Strong payment patterns expected']
		}
		
		mock_service.return_value = Mock()
		mock_service.return_value.generate_forecast = AsyncMock(return_value=mock_forecast_result)
		
		response = client.post(
			"/api/ar/analytics/cashflow-forecast",
			json=forecast_request,
			headers=auth_headers
		)
		
		assert response.status_code == 200
		data = response.json()
		assert len(data['data']['forecast_points']) == 30
		assert data['data']['overall_accuracy'] == 0.92
		assert len(data['data']['insights']) == 1


class TestErrorHandling:
	"""Test API error handling."""
	
	@pytest.fixture
	def auth_headers(self):
		"""Create authentication headers for testing."""
		return {"Authorization": "Bearer test_token"}
	
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_current_user')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_tenant_context')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_customer_service')
	def test_validation_error(self, mock_service, mock_tenant, mock_user, auth_headers):
		"""Test validation error handling."""
		# Mock authentication
		mock_user.return_value = {'user_id': uuid7str(), 'permissions': ['ar:write']}
		mock_tenant.return_value = {'tenant_id': uuid7str()}
		
		# Invalid customer data (missing required fields)
		invalid_data = {
			'legal_name': 'Test Customer',
			# Missing customer_code
		}
		
		response = client.post(
			"/api/ar/customers",
			json=invalid_data,
			headers=auth_headers
		)
		
		assert response.status_code == 422
		assert "validation error" in response.json()["detail"].lower()
	
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_current_user')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_tenant_context')
	@patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_customer_service')
	def test_service_error(self, mock_service, mock_tenant, mock_user, auth_headers):
		"""Test service layer error handling."""
		# Mock authentication
		mock_user.return_value = {'user_id': uuid7str(), 'permissions': ['ar:write']}
		mock_tenant.return_value = {'tenant_id': uuid7str()}
		
		# Mock service error
		mock_service.return_value = Mock()
		mock_service.return_value.create_customer = AsyncMock(
			side_effect=ValueError("Customer code already exists")
		)
		
		customer_data = {
			'customer_code': 'DUPLICATE001',
			'legal_name': 'Duplicate Customer'
		}
		
		response = client.post(
			"/api/ar/customers",
			json=customer_data,
			headers=auth_headers
		)
		
		assert response.status_code == 400
		assert "already exists" in response.json()["detail"]
	
	def test_permission_denied(self):
		"""Test permission denied error."""
		# Mock user without required permissions
		with patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_current_user') as mock_user:
			with patch('apg.capabilities.core_financials.accounts_receivable.api_endpoints.get_tenant_context') as mock_tenant:
				mock_user.return_value = {'user_id': uuid7str(), 'permissions': ['ar:read']}  # No write permission
				mock_tenant.return_value = {'tenant_id': uuid7str()}
				
				headers = {"Authorization": "Bearer test_token"}
				customer_data = {
					'customer_code': 'TEST001',
					'legal_name': 'Test Customer'
				}
				
				response = client.post(
					"/api/ar/customers",
					json=customer_data,
					headers=headers
				)
				
				assert response.status_code == 403
				assert "permission" in response.json()["detail"].lower()


# Run tests
if __name__ == "__main__":
	pytest.main([__file__, "-v"])