"""
APG Billing Service Tests

Comprehensive test suite for the billing service including subscription management,
usage tracking, invoice generation, and payment processing.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

from ..service import (
	BillingService, BillingError, SubscriptionError, UsageError, 
	InvoiceError, PaymentError, get_billing_service
)
from ..models import (
	BLCustomer, BLPlan, BLSubscription, BLUsage, BLInvoice, BLPayment,
	SubscriptionStatus, InvoiceStatus, PaymentStatus, BillingCurrency,
	CreateSubscriptionRequest, UsageSubmissionRequest, InvoiceGenerationRequest
)


class TestBillingService:
	"""Test cases for BillingService"""
	
	@pytest.fixture
	def service(self):
		"""Fixture providing a BillingService instance"""
		service = BillingService()
		# Setup test auth and audit services
		service._auth_service_available = True
		service._audit_service_available = True
		
		# Mock auth service for testing
		class MockAuthService:
			async def check_permission(self, user_id: str, permission: str) -> bool:
				# Grant permissions to test users
				if user_id in ["test-user-123", "admin-user"]:
					return True
				return False
			
			async def get_user(self, user_id: str) -> dict:
				return {"tenant_id": "test_tenant"}
		
		# Mock audit service for testing
		class MockAuditService:
			def __init__(self):
				self.logs = []
			
			async def log_action(self, user_id: str, action: str, resource_type: str, resource_id: str, details: dict):
				self.logs.append({
					"user_id": user_id,
					"action": action,
					"resource_type": resource_type,
					"resource_id": resource_id,
					"details": details,
					"timestamp": datetime.utcnow()
				})
		
		service.auth_service = MockAuthService()
		service.audit_service = MockAuditService()
		
		return service
	
	@pytest.fixture
	def sample_customer_data(self):
		"""Fixture providing sample customer data"""
		return {
			"name": "Test Customer",
			"email": "test@example.com",
			"company": "Test Company",
			"phone": "+1234567890",
			"currency": BillingCurrency.USD,
			"billing_address": {
				"street": "123 Test St",
				"city": "Test City",
				"state": "TS",
				"postal_code": "12345",
				"country": "US"
			},
			"payment_terms": 30
		}
	
	@pytest.fixture
	def sample_plan_data(self):
		"""Fixture providing sample plan data"""
		return {
			"name": "Test Plan",
			"description": "A test billing plan",
			"base_price": Decimal("29.99"),
			"currency": BillingCurrency.USD,
			"billing_period": "monthly",
			"features": ["feature1", "feature2"],
			"trial_period_days": 14
		}
	
	# Customer Management Tests
	
	async def test_create_customer_success(self, service, sample_customer_data):
		"""Test successful customer creation"""
		user_id = "test-user-123"
		
		customer = await service.create_customer(user_id, sample_customer_data)
		
		assert isinstance(customer, BLCustomer)
		assert customer.name == "Test Customer"
		assert customer.email == "test@example.com"
		assert customer.tenant_id == "test_tenant"
		assert customer.id in service.customers
		
		# Verify audit log
		assert len(service.audit_service.logs) == 1
		log = service.audit_service.logs[0]
		assert log["action"] == "create_customer"
		assert log["user_id"] == user_id
		assert log["resource_id"] == customer.id
	
	async def test_create_customer_permission_denied(self, service, sample_customer_data):
		"""Test customer creation with insufficient permissions"""
		user_id = "no-access-user"
		
		with pytest.raises(BillingError, match="not authorized"):
			await service.create_customer(user_id, sample_customer_data)
		
		# Verify no customer was created
		assert len(service.customers) == 0
	
	async def test_get_customer_success(self, service, sample_customer_data):
		"""Test successful customer retrieval"""
		user_id = "test-user-123"
		
		# Create customer first
		customer = await service.create_customer(user_id, sample_customer_data)
		
		# Retrieve customer
		retrieved_customer = await service.get_customer(user_id, customer.id)
		
		assert retrieved_customer is not None
		assert retrieved_customer.id == customer.id
		assert retrieved_customer.name == customer.name
	
	async def test_get_customer_not_found(self, service):
		"""Test customer retrieval for non-existent customer"""
		user_id = "test-user-123"
		non_existent_id = "non-existent-customer"
		
		result = await service.get_customer(user_id, non_existent_id)
		assert result is None
	
	async def test_list_customers_success(self, service, sample_customer_data):
		"""Test successful customer listing"""
		user_id = "test-user-123"
		
		# Create multiple customers
		customer1 = await service.create_customer(user_id, {
			**sample_customer_data,
			"name": "Customer 1",
			"email": "customer1@example.com"
		})
		customer2 = await service.create_customer(user_id, {
			**sample_customer_data,
			"name": "Customer 2",
			"email": "customer2@example.com"
		})
		
		# List customers
		customers = await service.list_customers(user_id)
		
		assert len(customers) == 2
		assert any(c.name == "Customer 1" for c in customers)
		assert any(c.name == "Customer 2" for c in customers)
	
	# Plan Management Tests
	
	async def test_create_plan_success(self, service, sample_plan_data):
		"""Test successful plan creation"""
		user_id = "test-user-123"
		
		plan = await service.create_plan(user_id, sample_plan_data)
		
		assert isinstance(plan, BLPlan)
		assert plan.name == "Test Plan"
		assert plan.base_price == Decimal("29.99")
		assert plan.tenant_id == "test_tenant"
		assert plan.id in service.plans
		
		# Verify audit log
		assert len(service.audit_service.logs) == 1
		log = service.audit_service.logs[0]
		assert log["action"] == "create_plan"
	
	async def test_get_plan_success(self, service, sample_plan_data):
		"""Test successful plan retrieval"""
		user_id = "test-user-123"
		
		# Create plan first
		plan = await service.create_plan(user_id, sample_plan_data)
		
		# Retrieve plan
		retrieved_plan = await service.get_plan(user_id, plan.id)
		
		assert retrieved_plan is not None
		assert retrieved_plan.id == plan.id
		assert retrieved_plan.name == plan.name
	
	# Subscription Management Tests
	
	async def test_create_subscription_success(self, service, sample_customer_data, sample_plan_data):
		"""Test successful subscription creation"""
		user_id = "test-user-123"
		
		# Create customer and plan
		customer = await service.create_customer(user_id, sample_customer_data)
		plan = await service.create_plan(user_id, sample_plan_data)
		
		# Create subscription
		request = CreateSubscriptionRequest(
			customer_id=customer.id,
			plan_id=plan.id,
			trial_period_days=14,
			metadata={"test": "data"}
		)
		
		subscription = await service.create_subscription(user_id, request)
		
		assert isinstance(subscription, BLSubscription)
		assert subscription.customer_id == customer.id
		assert subscription.plan_id == plan.id
		assert subscription.status == SubscriptionStatus.TRIAL
		assert subscription.trial_end is not None
		assert subscription.id in service.subscriptions
		
		# Verify audit log
		logs = [log for log in service.audit_service.logs if log["action"] == "create_subscription"]
		assert len(logs) == 1
	
	async def test_create_subscription_customer_not_found(self, service, sample_plan_data):
		"""Test subscription creation with non-existent customer"""
		user_id = "test-user-123"
		
		# Create plan only
		plan = await service.create_plan(user_id, sample_plan_data)
		
		request = CreateSubscriptionRequest(
			customer_id="non-existent-customer",
			plan_id=plan.id
		)
		
		with pytest.raises(SubscriptionError, match="Customer not found"):
			await service.create_subscription(user_id, request)
	
	async def test_create_subscription_plan_not_found(self, service, sample_customer_data):
		"""Test subscription creation with non-existent plan"""
		user_id = "test-user-123"
		
		# Create customer only
		customer = await service.create_customer(user_id, sample_customer_data)
		
		request = CreateSubscriptionRequest(
			customer_id=customer.id,
			plan_id="non-existent-plan"
		)
		
		with pytest.raises(SubscriptionError, match="Plan not found"):
			await service.create_subscription(user_id, request)
	
	async def test_cancel_subscription_success(self, service, sample_customer_data, sample_plan_data):
		"""Test successful subscription cancellation"""
		user_id = "test-user-123"
		
		# Create subscription
		customer = await service.create_customer(user_id, sample_customer_data)
		plan = await service.create_plan(user_id, sample_plan_data)
		request = CreateSubscriptionRequest(customer_id=customer.id, plan_id=plan.id)
		subscription = await service.create_subscription(user_id, request)
		
		# Cancel subscription
		cancelled_subscription = await service.cancel_subscription(
			user_id=user_id,
			subscription_id=subscription.id,
			cancel_at_period_end=True,
			reason="Test cancellation"
		)
		
		assert cancelled_subscription.cancel_at_period_end is True
		assert cancelled_subscription.cancellation_reason == "Test cancellation"
		assert cancelled_subscription.status != SubscriptionStatus.CANCELLED  # Not immediate
		
		# Test immediate cancellation
		immediate_cancelled = await service.cancel_subscription(
			user_id=user_id,
			subscription_id=subscription.id,
			cancel_at_period_end=False,
			reason="Immediate cancellation"
		)
		
		assert immediate_cancelled.status == SubscriptionStatus.CANCELLED
		assert immediate_cancelled.cancelled_at is not None
	
	# Usage Tracking Tests
	
	async def test_submit_usage_success(self, service, sample_customer_data, sample_plan_data):
		"""Test successful usage submission"""
		user_id = "test-user-123"
		
		# Create subscription
		customer = await service.create_customer(user_id, sample_customer_data)
		plan = await service.create_plan(user_id, sample_plan_data)
		request = CreateSubscriptionRequest(customer_id=customer.id, plan_id=plan.id)
		subscription = await service.create_subscription(user_id, request)
		
		# Submit usage
		usage_request = UsageSubmissionRequest(
			subscription_id=subscription.id,
			metric_name="api_calls",
			quantity=Decimal("1000"),
			metadata={"source": "test"}
		)
		
		usage = await service.submit_usage(user_id, usage_request)
		
		assert isinstance(usage, BLUsage)
		assert usage.subscription_id == subscription.id
		assert usage.metric_name == "api_calls"
		assert usage.quantity == Decimal("1000")
		assert usage in service.usage_records
		
		# Verify audit log
		logs = [log for log in service.audit_service.logs if log["action"] == "submit_usage"]
		assert len(logs) == 1
	
	async def test_submit_usage_subscription_not_found(self, service):
		"""Test usage submission with non-existent subscription"""
		user_id = "test-user-123"
		
		usage_request = UsageSubmissionRequest(
			subscription_id="non-existent-subscription",
			metric_name="api_calls",
			quantity=Decimal("1000")
		)
		
		with pytest.raises(UsageError, match="Subscription not found"):
			await service.submit_usage(user_id, usage_request)
	
	async def test_get_usage_summary(self, service, sample_customer_data, sample_plan_data):
		"""Test usage summary retrieval"""
		user_id = "test-user-123"
		
		# Create subscription
		customer = await service.create_customer(user_id, sample_customer_data)
		plan = await service.create_plan(user_id, sample_plan_data)
		request = CreateSubscriptionRequest(customer_id=customer.id, plan_id=plan.id)
		subscription = await service.create_subscription(user_id, request)
		
		# Submit multiple usage records
		for i in range(3):
			usage_request = UsageSubmissionRequest(
				subscription_id=subscription.id,
				metric_name="api_calls",
				quantity=Decimal("1000")
			)
			await service.submit_usage(user_id, usage_request)
		
		# Get usage summary
		summary = await service.get_usage_summary(user_id, subscription.id)
		
		assert summary["subscription_id"] == subscription.id
		assert "usage_by_metric" in summary
		assert "api_calls" in summary["usage_by_metric"]
		assert summary["usage_by_metric"]["api_calls"]["total_quantity"] == Decimal("3000")
		assert summary["usage_by_metric"]["api_calls"]["count"] == 3
	
	# Invoice Management Tests
	
	async def test_generate_invoice_success(self, service, sample_customer_data, sample_plan_data):
		"""Test successful invoice generation"""
		user_id = "test-user-123"
		
		# Create subscription
		customer = await service.create_customer(user_id, sample_customer_data)
		plan = await service.create_plan(user_id, sample_plan_data)
		request = CreateSubscriptionRequest(customer_id=customer.id, plan_id=plan.id)
		subscription = await service.create_subscription(user_id, request)
		
		# Submit some usage
		usage_request = UsageSubmissionRequest(
			subscription_id=subscription.id,
			metric_name="api_calls",
			quantity=Decimal("5000")
		)
		await service.submit_usage(user_id, usage_request)
		
		# Generate invoice
		invoice_request = InvoiceGenerationRequest(
			subscription_id=subscription.id,
			billing_period_start=subscription.current_period_start,
			billing_period_end=subscription.current_period_end,
			include_usage=True,
			auto_finalize=False
		)
		
		invoice = await service.generate_invoice(user_id, invoice_request)
		
		assert isinstance(invoice, BLInvoice)
		assert invoice.subscription_id == subscription.id
		assert invoice.customer_id == customer.id
		assert invoice.status == InvoiceStatus.DRAFT
		assert invoice.total > 0
		assert len(invoice.line_items) > 0
		assert invoice.id in service.invoices
		
		# Verify line items include subscription and usage charges
		line_item_types = [item["type"] for item in invoice.line_items]
		assert "subscription" in line_item_types
		assert "usage" in line_item_types
	
	async def test_generate_invoice_subscription_not_found(self, service):
		"""Test invoice generation with non-existent subscription"""
		user_id = "test-user-123"
		
		invoice_request = InvoiceGenerationRequest(
			subscription_id="non-existent-subscription",
			billing_period_start=datetime.utcnow(),
			billing_period_end=datetime.utcnow() + timedelta(days=30)
		)
		
		with pytest.raises(InvoiceError, match="Subscription not found"):
			await service.generate_invoice(user_id, invoice_request)
	
	# Payment Processing Tests
	
	async def test_process_payment_success(self, service, sample_customer_data, sample_plan_data):
		"""Test successful payment processing"""
		user_id = "test-user-123"
		
		# Create customer and invoice
		customer = await service.create_customer(user_id, sample_customer_data)
		plan = await service.create_plan(user_id, sample_plan_data)
		request = CreateSubscriptionRequest(customer_id=customer.id, plan_id=plan.id)
		subscription = await service.create_subscription(user_id, request)
		
		invoice_request = InvoiceGenerationRequest(
			subscription_id=subscription.id,
			billing_period_start=subscription.current_period_start,
			billing_period_end=subscription.current_period_end
		)
		invoice = await service.generate_invoice(user_id, invoice_request)
		
		# Process payment
		payment_data = {
			"customer_id": customer.id,
			"invoice_id": invoice.id,
			"amount": invoice.total,
			"currency": BillingCurrency.USD,
			"payment_method_type": "credit_card"
		}
		
		payment = await service.process_payment(user_id, payment_data)
		
		assert isinstance(payment, BLPayment)
		assert payment.customer_id == customer.id
		assert payment.invoice_id == invoice.id
		assert payment.amount == invoice.total
		assert payment.status in [PaymentStatus.PROCESSING, PaymentStatus.SUCCEEDED]
		assert payment.id in service.payments
	
	# Analytics Tests
	
	async def test_get_billing_analytics(self, service, sample_customer_data, sample_plan_data):
		"""Test billing analytics retrieval"""
		user_id = "test-user-123"
		
		# Create some test data
		customer = await service.create_customer(user_id, sample_customer_data)
		plan = await service.create_plan(user_id, sample_plan_data)
		request = CreateSubscriptionRequest(customer_id=customer.id, plan_id=plan.id)
		subscription = await service.create_subscription(user_id, request)
		
		# Generate and pay invoice
		invoice_request = InvoiceGenerationRequest(
			subscription_id=subscription.id,
			billing_period_start=subscription.current_period_start,
			billing_period_end=subscription.current_period_end
		)
		invoice = await service.generate_invoice(user_id, invoice_request)
		invoice.status = InvoiceStatus.PAID  # Simulate payment
		
		# Get analytics
		analytics = await service.get_billing_analytics(user_id)
		
		assert "period_start" in analytics
		assert "period_end" in analytics
		assert "metrics" in analytics
		assert analytics["metrics"]["total_customers"] == 1
		assert analytics["metrics"]["active_subscriptions"] == 1
		assert analytics["metrics"]["total_invoices"] == 1
	
	# Error Handling Tests
	
	async def test_permission_denied_errors(self, service, sample_customer_data):
		"""Test permission denied errors for unauthorized users"""
		unauthorized_user = "unauthorized-user"
		
		# Test customer creation
		with pytest.raises(BillingError, match="not authorized"):
			await service.create_customer(unauthorized_user, sample_customer_data)
		
		# Test customer listing
		with pytest.raises(BillingError, match="not authorized"):
			await service.list_customers(unauthorized_user)
		
		# Test analytics access
		with pytest.raises(BillingError, match="not authorized"):
			await service.get_billing_analytics(unauthorized_user)
	
	async def test_service_status(self, service):
		"""Test service status reporting"""
		status = await service.get_service_status()
		
		assert status["service"] == "BillingService"
		assert status["status"] == "healthy"
		assert "customers" in status
		assert "subscriptions" in status
		assert "invoices" in status
		assert "payments" in status
		assert "integrations" in status
		assert "timestamp" in status


class TestBillingServiceSingleton:
	"""Test billing service singleton pattern"""
	
	def test_get_billing_service_singleton(self):
		"""Test that get_billing_service returns same instance"""
		service1 = get_billing_service()
		service2 = get_billing_service()
		
		assert service1 is service2
		assert isinstance(service1, BillingService)
	
	def test_service_initialization(self):
		"""Test service initialization"""
		service = get_billing_service()
		
		assert hasattr(service, 'customers')
		assert hasattr(service, 'plans')
		assert hasattr(service, 'subscriptions')
		assert hasattr(service, 'usage_records')
		assert hasattr(service, 'invoices')
		assert hasattr(service, 'payments')
		
		# Check that collections are initialized as empty
		assert len(service.customers) == 0
		assert len(service.plans) == 0
		assert len(service.subscriptions) == 0
		assert len(service.usage_records) == 0
		assert len(service.invoices) == 0
		assert len(service.payments) == 0


class TestBillingIntegration:
	"""Test billing integration with other APG capabilities"""
	
	@pytest.fixture
	def integrated_service(self):
		"""Service with mock APG integrations"""
		service = BillingService()
		
		# Mock all integrations as available
		service._auth_service_available = True
		service._audit_service_available = True
		service._ai_orchestration_available = True
		
		# Mock services
		class MockAuthService:
			async def check_permission(self, user_id: str, permission: str) -> bool:
				return user_id == "test-user-123"
			
			async def get_user(self, user_id: str) -> dict:
				return {"tenant_id": "test_tenant"}
		
		class MockAuditService:
			def __init__(self):
				self.logs = []
			
			async def log_action(self, user_id: str, action: str, resource_type: str, resource_id: str, details: dict):
				self.logs.append({
					"user_id": user_id,
					"action": action,
					"resource_type": resource_type,
					"resource_id": resource_id,
					"details": details
				})
		
		class MockAIOrchestration:
			async def submit_task(self, task_definition: dict) -> str:
				return "task-123"
			
			async def get_task_status(self, task_id: str) -> dict:
				return {"status": "completed", "result": {"success": True}}
		
		service.auth_service = MockAuthService()
		service.audit_service = MockAuditService()
		service.ai_orchestration = MockAIOrchestration()
		
		return service
	
	async def test_auth_integration(self, integrated_service):
		"""Test auth service integration"""
		user_id = "test-user-123"
		
		# Test permission check
		has_permission = await integrated_service._check_billing_permissions(user_id, "create_customer")
		assert has_permission is True
		
		# Test tenant retrieval
		tenant_id = await integrated_service._get_user_tenant(user_id)
		assert tenant_id == "test_tenant"
	
	async def test_audit_integration(self, integrated_service):
		"""Test audit service integration"""
		user_id = "test-user-123"
		resource_id = "test-resource-123"
		
		# Test audit logging
		await integrated_service._audit_billing_action(
			action="test_action",
			user_id=user_id,
			resource_id=resource_id,
			details={"test": "data"}
		)
		
		# Verify audit log was created
		assert len(integrated_service.audit_service.logs) == 1
		log = integrated_service.audit_service.logs[0]
		assert log["action"] == "test_action"
		assert log["user_id"] == user_id
		assert log["resource_id"] == resource_id
		assert log["details"]["test"] == "data"
	
	async def test_service_degradation_without_integrations(self):
		"""Test service functions when APG integrations are unavailable"""
		service = BillingService()
		
		# All integrations unavailable
		service._auth_service_available = False
		service._audit_service_available = False
		
		# Auth should default to allow with warning
		has_permission = await service._check_billing_permissions("user", "create_customer")
		assert has_permission is True
		
		# Audit should not fail
		await service._audit_billing_action("test", "user", "resource", {})
		
		# Tenant should default
		tenant = await service._get_user_tenant("user")
		assert tenant == "default"