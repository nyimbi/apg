"""
APG Accounts Receivable - Models Tests
Unit tests for Pydantic data models with validation

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any

from uuid_extensions import uuid7str

from ..models import (
	ARCustomer, ARInvoice, ARPayment, ARCollectionActivity, ARCreditAssessment, ARDispute,
	ARCustomerType, ARCustomerStatus, ARInvoiceStatus, ARPaymentStatus, 
	ARCollectionPriority, ARCreditRiskLevel, ARDisputeStatus
)


class TestEnums:
	"""Test enum value definitions."""
	
	def test_customer_type_enum(self):
		"""Test customer type enum values."""
		assert ARCustomerType.INDIVIDUAL == "INDIVIDUAL"
		assert ARCustomerType.CORPORATION == "CORPORATION"
		assert ARCustomerType.PARTNERSHIP == "PARTNERSHIP"
		assert ARCustomerType.GOVERNMENT == "GOVERNMENT"
	
	def test_customer_status_enum(self):
		"""Test customer status enum values."""
		assert ARCustomerStatus.ACTIVE == "ACTIVE"
		assert ARCustomerStatus.INACTIVE == "INACTIVE"
		assert ARCustomerStatus.SUSPENDED == "SUSPENDED"
		assert ARCustomerStatus.CLOSED == "CLOSED"
	
	def test_invoice_status_enum(self):
		"""Test invoice status enum values."""
		assert ARInvoiceStatus.DRAFT == "DRAFT"
		assert ARInvoiceStatus.SENT == "SENT"
		assert ARInvoiceStatus.PAID == "PAID"
		assert ARInvoiceStatus.OVERDUE == "OVERDUE"
		assert ARInvoiceStatus.CANCELLED == "CANCELLED"
	
	def test_payment_status_enum(self):
		"""Test payment status enum values."""
		assert ARPaymentStatus.PENDING == "PENDING"
		assert ARPaymentStatus.PROCESSING == "PROCESSING"
		assert ARPaymentStatus.PROCESSED == "PROCESSED"
		assert ARPaymentStatus.FAILED == "FAILED"
		assert ARPaymentStatus.CANCELLED == "CANCELLED"


class TestARCustomer:
	"""Test AR customer model."""
	
	def test_customer_creation_valid_data(self):
		"""Test creating customer with valid data."""
		customer = ARCustomer(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_code="CUST001",
			legal_name="Test Customer Corp",
			display_name="Test Customer",
			customer_type=ARCustomerType.CORPORATION,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal('50000.00'),
			payment_terms_days=30,
			total_outstanding=Decimal('12500.00'),
			overdue_amount=Decimal('2500.00'),
			contact_email="billing@testcustomer.com",
			contact_phone="+1-555-123-4567",
			billing_address="123 Business St, City, ST 12345",
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		assert customer.customer_code == "CUST001"
		assert customer.legal_name == "Test Customer Corp"
		assert customer.customer_type == ARCustomerType.CORPORATION
		assert customer.status == ARCustomerStatus.ACTIVE
		assert customer.credit_limit == Decimal('50000.00')
		assert customer.payment_terms_days == 30
		assert customer.total_outstanding == Decimal('12500.00')
		assert customer.overdue_amount == Decimal('2500.00')
		assert customer.contact_email == "billing@testcustomer.com"
		assert customer.billing_address == "123 Business St, City, ST 12345"
	
	def test_customer_defaults(self):
		"""Test customer model default values."""
		customer = ARCustomer(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_code="CUST002",
			legal_name="Minimal Customer",
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		assert customer.customer_type == ARCustomerType.INDIVIDUAL
		assert customer.status == ARCustomerStatus.ACTIVE
		assert customer.credit_limit == Decimal('0.00')
		assert customer.payment_terms_days == 30
		assert customer.total_outstanding == Decimal('0.00')
		assert customer.overdue_amount == Decimal('0.00')
		assert customer.display_name is None
		assert customer.contact_email is None
		assert customer.contact_phone is None
		assert customer.billing_address is None
	
	def test_customer_validation_errors(self):
		"""Test customer validation errors."""
		
		# Test negative credit limit
		with pytest.raises(ValueError, match="Credit limit must be non-negative"):
			ARCustomer(
				id=uuid7str(),
				tenant_id=uuid7str(),
				customer_code="CUST003",
				legal_name="Invalid Customer",
				credit_limit=Decimal('-1000.00'),
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
		
		# Test negative payment terms
		with pytest.raises(ValueError, match="Payment terms must be positive"):
			ARCustomer(
				id=uuid7str(),
				tenant_id=uuid7str(),
				customer_code="CUST004",
				legal_name="Invalid Customer",
				payment_terms_days=-5,
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
		
		# Test negative amounts
		with pytest.raises(ValueError, match="Total outstanding must be non-negative"):
			ARCustomer(
				id=uuid7str(),
				tenant_id=uuid7str(),
				customer_code="CUST005",
				legal_name="Invalid Customer",
				total_outstanding=Decimal('-500.00'),
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
	
	def test_customer_email_validation(self):
		"""Test customer email validation."""
		
		# Valid email should work
		customer = ARCustomer(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_code="CUST006",
			legal_name="Valid Email Customer",
			contact_email="valid@example.com",
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		assert customer.contact_email == "valid@example.com"
		
		# Invalid email should raise error
		with pytest.raises(ValueError, match="Invalid email format"):
			ARCustomer(
				id=uuid7str(),
				tenant_id=uuid7str(),
				customer_code="CUST007",
				legal_name="Invalid Email Customer",
				contact_email="invalid-email",
				created_by=uuid7str(),
				updated_by=uuid7str()
			)


class TestARInvoice:
	"""Test AR invoice model."""
	
	def test_invoice_creation_valid_data(self):
		"""Test creating invoice with valid data."""
		invoice = ARInvoice(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_id=uuid7str(),
			invoice_number="INV-2025-001",
			invoice_date=date.today(),
			due_date=date.today() + timedelta(days=30),
			total_amount=Decimal('5000.00'),
			paid_amount=Decimal('2500.00'),
			outstanding_amount=Decimal('2500.00'),
			currency_code="USD",
			status=ARInvoiceStatus.SENT,
			payment_status="PARTIAL",
			description="Test invoice for services",
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		assert invoice.invoice_number == "INV-2025-001"
		assert invoice.invoice_date == date.today()
		assert invoice.due_date == date.today() + timedelta(days=30)
		assert invoice.total_amount == Decimal('5000.00')
		assert invoice.paid_amount == Decimal('2500.00')
		assert invoice.outstanding_amount == Decimal('2500.00')
		assert invoice.currency_code == "USD"
		assert invoice.status == ARInvoiceStatus.SENT
		assert invoice.payment_status == "PARTIAL"
		assert invoice.description == "Test invoice for services"
	
	def test_invoice_defaults(self):
		"""Test invoice model default values."""
		invoice = ARInvoice(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_id=uuid7str(),
			invoice_number="INV-2025-002",
			invoice_date=date.today(),
			due_date=date.today() + timedelta(days=30),
			total_amount=Decimal('1000.00'),
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		assert invoice.paid_amount == Decimal('0.00')
		assert invoice.outstanding_amount == Decimal('1000.00')
		assert invoice.currency_code == "USD"
		assert invoice.status == ARInvoiceStatus.DRAFT
		assert invoice.payment_status == "UNPAID"
		assert invoice.description is None
	
	def test_invoice_date_validation(self):
		"""Test invoice date validation."""
		
		# Due date before invoice date should raise error
		with pytest.raises(ValueError, match="Due date must be after invoice date"):
			ARInvoice(
				id=uuid7str(),
				tenant_id=uuid7str(),
				customer_id=uuid7str(),
				invoice_number="INV-2025-003",
				invoice_date=date.today(),
				due_date=date.today() - timedelta(days=1),
				total_amount=Decimal('1000.00'),
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
	
	def test_invoice_amount_validation(self):
		"""Test invoice amount validation."""
		
		# Negative total amount
		with pytest.raises(ValueError, match="Total amount must be positive"):
			ARInvoice(
				id=uuid7str(),
				tenant_id=uuid7str(),
				customer_id=uuid7str(),
				invoice_number="INV-2025-004",
				invoice_date=date.today(),
				due_date=date.today() + timedelta(days=30),
				total_amount=Decimal('-100.00'),
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
		
		# Paid amount exceeding total
		with pytest.raises(ValueError, match="Paid amount cannot exceed total amount"):
			ARInvoice(
				id=uuid7str(),
				tenant_id=uuid7str(),
				customer_id=uuid7str(),
				invoice_number="INV-2025-005",
				invoice_date=date.today(),
				due_date=date.today() + timedelta(days=30),
				total_amount=Decimal('1000.00'),
				paid_amount=Decimal('1500.00'),
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
	
	def test_invoice_outstanding_calculation(self):
		"""Test invoice outstanding amount calculation."""
		invoice = ARInvoice(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_id=uuid7str(),
			invoice_number="INV-2025-006",
			invoice_date=date.today(),
			due_date=date.today() + timedelta(days=30),
			total_amount=Decimal('5000.00'),
			paid_amount=Decimal('3000.00'),
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		# Outstanding amount should be auto-calculated if not provided
		assert invoice.outstanding_amount == Decimal('2000.00')


class TestARPayment:
	"""Test AR payment model."""
	
	def test_payment_creation_valid_data(self):
		"""Test creating payment with valid data."""
		payment = ARPayment(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_id=uuid7str(),
			payment_reference="PAY-2025-001",
			payment_date=date.today(),
			payment_amount=Decimal('2500.00'),
			payment_method="CREDIT_CARD",
			status=ARPaymentStatus.PROCESSED,
			currency_code="USD",
			bank_reference="TXN123456789",
			notes="Payment for invoice INV-2025-001",
			processed_at=datetime.utcnow(),
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		assert payment.payment_reference == "PAY-2025-001"
		assert payment.payment_date == date.today()
		assert payment.payment_amount == Decimal('2500.00')
		assert payment.payment_method == "CREDIT_CARD"
		assert payment.status == ARPaymentStatus.PROCESSED
		assert payment.currency_code == "USD"
		assert payment.bank_reference == "TXN123456789"
		assert payment.notes == "Payment for invoice INV-2025-001"
		assert payment.processed_at is not None
	
	def test_payment_defaults(self):
		"""Test payment model default values."""
		payment = ARPayment(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_id=uuid7str(),
			payment_reference="PAY-2025-002",
			payment_date=date.today(),
			payment_amount=Decimal('1000.00'),
			payment_method="CHECK",
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		assert payment.status == ARPaymentStatus.PENDING
		assert payment.currency_code == "USD"
		assert payment.bank_reference is None
		assert payment.notes is None
		assert payment.processed_at is None
	
	def test_payment_amount_validation(self):
		"""Test payment amount validation."""
		
		# Negative payment amount
		with pytest.raises(ValueError, match="Payment amount must be positive"):
			ARPayment(
				id=uuid7str(),
				tenant_id=uuid7str(),
				customer_id=uuid7str(),
				payment_reference="PAY-2025-003",
				payment_date=date.today(),
				payment_amount=Decimal('-100.00'),
				payment_method="CASH",
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
		
		# Zero payment amount
		with pytest.raises(ValueError, match="Payment amount must be positive"):
			ARPayment(
				id=uuid7str(),
				tenant_id=uuid7str(),
				customer_id=uuid7str(),
				payment_reference="PAY-2025-004",
				payment_date=date.today(),
				payment_amount=Decimal('0.00'),
				payment_method="CASH",
				created_by=uuid7str(),
				updated_by=uuid7str()
			)


class TestARCollectionActivity:
	"""Test AR collection activity model."""
	
	def test_collection_activity_creation_valid_data(self):
		"""Test creating collection activity with valid data."""
		activity = ARCollectionActivity(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_id=uuid7str(),
			activity_type="PHONE_CALL",
			activity_date=date.today(),
			priority=ARCollectionPriority.HIGH,
			contact_method="PHONE",
			outcome="PROMISED_PAYMENT",
			status="COMPLETED",
			notes="Customer promised payment by end of week",
			follow_up_date=date.today() + timedelta(days=7),
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		assert activity.activity_type == "PHONE_CALL"
		assert activity.activity_date == date.today()
		assert activity.priority == ARCollectionPriority.HIGH
		assert activity.contact_method == "PHONE"
		assert activity.outcome == "PROMISED_PAYMENT"
		assert activity.status == "COMPLETED"
		assert activity.notes == "Customer promised payment by end of week"
		assert activity.follow_up_date == date.today() + timedelta(days=7)
	
	def test_collection_activity_defaults(self):
		"""Test collection activity model default values."""
		activity = ARCollectionActivity(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_id=uuid7str(),
			activity_type="EMAIL",
			activity_date=date.today(),
			contact_method="EMAIL",
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		assert activity.priority == ARCollectionPriority.MEDIUM
		assert activity.outcome is None
		assert activity.status == "PENDING"
		assert activity.notes is None
		assert activity.follow_up_date is None


class TestARCreditAssessment:
	"""Test AR credit assessment model."""
	
	def test_credit_assessment_creation_valid_data(self):
		"""Test creating credit assessment with valid data."""
		assessment = ARCreditAssessment(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_id=uuid7str(),
			assessment_date=date.today(),
			credit_score=750,
			risk_level=ARCreditRiskLevel.LOW,
			recommended_credit_limit=Decimal('75000.00'),
			confidence_score=0.92,
			assessment_notes="Strong payment history, stable business",
			expires_at=datetime.utcnow() + timedelta(days=90),
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		assert assessment.assessment_date == date.today()
		assert assessment.credit_score == 750
		assert assessment.risk_level == ARCreditRiskLevel.LOW
		assert assessment.recommended_credit_limit == Decimal('75000.00')
		assert assessment.confidence_score == 0.92
		assert assessment.assessment_notes == "Strong payment history, stable business"
		assert assessment.expires_at is not None
	
	def test_credit_assessment_validation(self):
		"""Test credit assessment validation."""
		
		# Invalid credit score (too low)
		with pytest.raises(ValueError, match="Credit score must be between 300 and 850"):
			ARCreditAssessment(
				id=uuid7str(),
				tenant_id=uuid7str(),
				customer_id=uuid7str(),
				assessment_date=date.today(),
				credit_score=250,
				risk_level=ARCreditRiskLevel.HIGH,
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
		
		# Invalid credit score (too high)
		with pytest.raises(ValueError, match="Credit score must be between 300 and 850"):
			ARCreditAssessment(
				id=uuid7str(),
				tenant_id=uuid7str(),
				customer_id=uuid7str(),
				assessment_date=date.today(),
				credit_score=900,
				risk_level=ARCreditRiskLevel.LOW,
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
		
		# Invalid confidence score (too low)
		with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
			ARCreditAssessment(
				id=uuid7str(),
				tenant_id=uuid7str(),
				customer_id=uuid7str(),
				assessment_date=date.today(),
				credit_score=700,
				risk_level=ARCreditRiskLevel.MEDIUM,
				confidence_score=-0.1,
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
		
		# Invalid confidence score (too high)
		with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
			ARCreditAssessment(
				id=uuid7str(),
				tenant_id=uuid7str(),
				customer_id=uuid7str(),
				assessment_date=date.today(),
				credit_score=700,
				risk_level=ARCreditRiskLevel.MEDIUM,
				confidence_score=1.5,
				created_by=uuid7str(),
				updated_by=uuid7str()
			)


class TestARDispute:
	"""Test AR dispute model."""
	
	def test_dispute_creation_valid_data(self):
		"""Test creating dispute with valid data."""
		dispute = ARDispute(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_id=uuid7str(),
			invoice_id=uuid7str(),
			dispute_reference="DISP-2025-001",
			dispute_date=date.today(),
			dispute_amount=Decimal('1000.00'),
			dispute_reason="BILLING_ERROR",
			status=ARDisputeStatus.OPEN,
			description="Disputed charges for services not received",
			resolution_notes=None,
			resolved_date=None,
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		assert dispute.dispute_reference == "DISP-2025-001"
		assert dispute.dispute_date == date.today()
		assert dispute.dispute_amount == Decimal('1000.00')
		assert dispute.dispute_reason == "BILLING_ERROR"
		assert dispute.status == ARDisputeStatus.OPEN
		assert dispute.description == "Disputed charges for services not received"
		assert dispute.resolution_notes is None
		assert dispute.resolved_date is None
	
	def test_dispute_amount_validation(self):
		"""Test dispute amount validation."""
		
		# Negative dispute amount
		with pytest.raises(ValueError, match="Dispute amount must be positive"):
			ARDispute(
				id=uuid7str(),
				tenant_id=uuid7str(),
				customer_id=uuid7str(),
				invoice_id=uuid7str(),
				dispute_reference="DISP-2025-002",
				dispute_date=date.today(),
				dispute_amount=Decimal('-500.00'),
				dispute_reason="QUALITY_ISSUE",
				created_by=uuid7str(),
				updated_by=uuid7str()
			)


class TestModelIntegration:
	"""Test model integration scenarios."""
	
	def test_customer_invoice_relationship_data_consistency(self):
		"""Test data consistency between customer and invoice models."""
		
		# Create customer
		customer = ARCustomer(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_code="CUST100",
			legal_name="Integration Test Customer",
			customer_type=ARCustomerType.CORPORATION,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal('25000.00'),
			payment_terms_days=30,
			total_outstanding=Decimal('5000.00'),
			overdue_amount=Decimal('1000.00'),
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		# Create invoice for the customer
		invoice = ARInvoice(
			id=uuid7str(),
			tenant_id=customer.tenant_id,  # Same tenant
			customer_id=customer.id,  # Linked to customer
			invoice_number="INV-INTEG-001",
			invoice_date=date.today(),
			due_date=date.today() + timedelta(days=customer.payment_terms_days),
			total_amount=Decimal('2500.00'),
			paid_amount=Decimal('1500.00'),
			outstanding_amount=Decimal('1000.00'),
			currency_code="USD",
			status=ARInvoiceStatus.SENT,
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		# Verify relationships
		assert invoice.tenant_id == customer.tenant_id
		assert invoice.customer_id == customer.id
		assert invoice.due_date == customer.created_at.date() + timedelta(days=customer.payment_terms_days)
		assert invoice.outstanding_amount <= customer.total_outstanding
	
	def test_full_ar_workflow_models(self):
		"""Test complete AR workflow using all models."""
		
		# Create customer
		customer = ARCustomer(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_code="WORKFLOW001",
			legal_name="Workflow Test Corp",
			customer_type=ARCustomerType.CORPORATION,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal('50000.00'),
			payment_terms_days=30,
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		# Create credit assessment
		assessment = ARCreditAssessment(
			id=uuid7str(),
			tenant_id=customer.tenant_id,
			customer_id=customer.id,
			assessment_date=date.today(),
			credit_score=720,
			risk_level=ARCreditRiskLevel.MEDIUM,
			recommended_credit_limit=Decimal('45000.00'),
			confidence_score=0.87,
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		# Create invoice
		invoice = ARInvoice(
			id=uuid7str(),
			tenant_id=customer.tenant_id,
			customer_id=customer.id,
			invoice_number="WF-INV-001",
			invoice_date=date.today(),
			due_date=date.today() + timedelta(days=30),
			total_amount=Decimal('10000.00'),
			status=ARInvoiceStatus.SENT,
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		# Create payment
		payment = ARPayment(
			id=uuid7str(),
			tenant_id=customer.tenant_id,
			customer_id=customer.id,
			payment_reference="WF-PAY-001",
			payment_date=date.today(),
			payment_amount=Decimal('6000.00'),
			payment_method="WIRE_TRANSFER",
			status=ARPaymentStatus.PROCESSED,
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		# Create collection activity
		collection = ARCollectionActivity(
			id=uuid7str(),
			tenant_id=customer.tenant_id,
			customer_id=customer.id,
			activity_type="EMAIL_REMINDER",
			activity_date=date.today(),
			priority=ARCollectionPriority.MEDIUM,
			contact_method="EMAIL",
			outcome="SENT",
			status="COMPLETED",
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		# Verify all models are consistent
		assert all(model.tenant_id == customer.tenant_id for model in [assessment, invoice, payment, collection])
		assert all(model.customer_id == customer.id for model in [assessment, invoice, payment, collection])
		
		# Verify business logic
		remaining_balance = invoice.total_amount - payment.payment_amount
		assert remaining_balance == Decimal('4000.00')
		assert remaining_balance <= customer.credit_limit
		assert assessment.recommended_credit_limit <= customer.credit_limit


# Run tests
if __name__ == "__main__":
	pytest.main([__file__, "-v"])