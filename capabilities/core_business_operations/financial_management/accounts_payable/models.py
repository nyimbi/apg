"""
APG Core Financials - Accounts Payable Data Models

CLAUDE.md compliant data models with async Python, modern typing,
and APG platform integration for enterprise-grade AP operations.

© 2025 Datacraft. All rights reserved.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, validator, field_validator
from uuid_extensions import uuid7str


# Core Enumerations

class VendorStatus(str, Enum):
	"""Vendor status enumeration"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	SUSPENDED = "suspended"
	PENDING_APPROVAL = "pending_approval"
	BLOCKED = "blocked"


class VendorType(str, Enum):
	"""Vendor type classification"""
	SUPPLIER = "supplier"
	CONTRACTOR = "contractor" 
	UTILITY = "utility"
	LANDLORD = "landlord"
	EMPLOYEE = "employee"
	GOVERNMENT = "government"
	PROFESSIONAL = "professional"
	TECHNOLOGY = "technology"


class InvoiceStatus(str, Enum):
	"""Invoice processing status"""
	PENDING = "pending"
	IN_PROCESS = "in_process"
	APPROVED = "approved"
	REJECTED = "rejected"
	POSTED = "posted"
	PAID = "paid"
	CANCELLED = "cancelled"
	ON_HOLD = "on_hold"


class PaymentStatus(str, Enum):
	"""Payment processing status"""
	SCHEDULED = "scheduled"
	PROCESSING = "processing"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	RETURNED = "returned"


class PaymentMethod(str, Enum):
	"""Payment processing methods"""
	ACH = "ach"
	WIRE = "wire"
	CHECK = "check"
	VIRTUAL_CARD = "virtual_card"
	RTP = "rtp"
	FEDNOW = "fednow"
	CRYPTOCURRENCY = "cryptocurrency"


class ApprovalStatus(str, Enum):
	"""Approval workflow status"""
	PENDING = "pending"
	IN_PROGRESS = "in_progress"
	APPROVED = "approved"
	REJECTED = "rejected"
	ESCALATED = "escalated"
	DELEGATED = "delegated"


class MatchingStatus(str, Enum):
	"""Three-way matching status"""
	NOT_MATCHED = "not_matched"
	PARTIAL_MATCH = "partial_match"
	COMPLETE_MATCH = "complete_match"
	FAILED_MATCH = "failed_match"
	WAITING = "waiting"
	EXCEPTION = "exception"


# Base Configuration for all models
MODEL_CONFIG = ConfigDict(
	extra='forbid',
	validate_by_name=True,
	validate_by_alias=True,
	str_strip_whitespace=True,
	validate_default=True
)


# Core Data Models

@dataclass
class ContactInfo(BaseModel):
	"""Contact information structure"""
	name: str
	title: str | None = None
	email: str | None = None
	phone: str | None = None
	mobile: str | None = None
	fax: str | None = None
	
	model_config = MODEL_CONFIG


@dataclass 
class Address(BaseModel):
	"""Address information structure"""
	address_type: str  # billing, shipping, remit_to
	line1: str
	line2: str | None = None
	city: str
	state_province: str
	postal_code: str
	country_code: str
	is_primary: bool = False
	
	model_config = MODEL_CONFIG


@dataclass
class BankingInfo(BaseModel):
	"""Banking and payment information"""
	account_type: str  # checking, savings, wire
	bank_name: str
	routing_number: str
	account_number: str
	account_holder_name: str
	swift_code: str | None = None
	iban: str | None = None
	is_primary: bool = False
	is_active: bool = True
	
	model_config = MODEL_CONFIG


@dataclass
class TaxInfo(BaseModel):
	"""Tax identification information"""
	tax_id: str | None = None
	tax_id_type: str | None = None  # ein, ssn, vat, etc.
	tax_classification: str | None = None
	is_1099_vendor: bool = False
	backup_withholding: bool = False
	tax_exempt: bool = False
	tax_exempt_certificate: str | None = None
	
	model_config = MODEL_CONFIG


@dataclass
class PaymentTerms(BaseModel):
	"""Payment terms configuration"""
	code: str
	name: str
	net_days: int
	discount_days: int = 0
	discount_percent: Decimal = Field(default=Decimal('0.00'), max_digits=5, decimal_places=2)
	
	model_config = MODEL_CONFIG


@dataclass
class VendorPerformanceMetrics(BaseModel):
	"""Vendor performance tracking"""
	overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
	on_time_delivery_rate: float = Field(default=0.0, ge=0.0, le=100.0)
	quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
	cost_competitiveness: float = Field(default=0.0, ge=0.0, le=100.0)
	last_updated: datetime = Field(default_factory=datetime.utcnow)
	
	model_config = MODEL_CONFIG


@dataclass
class APVendor(BaseModel):
	"""Master vendor record with comprehensive data"""
	id: str = Field(default_factory=uuid7str)
	vendor_code: str
	legal_name: str
	trade_name: str | None = None
	vendor_type: VendorType
	status: VendorStatus = VendorStatus.PENDING_APPROVAL
	
	# Contact and address information
	primary_contact: ContactInfo
	addresses: List[Address] = Field(default_factory=list)
	
	# Financial information
	payment_terms: PaymentTerms
	banking_details: List[BankingInfo] = Field(default_factory=list)
	credit_limit: Decimal | None = Field(default=None, max_digits=15, decimal_places=2)
	
	# Tax and compliance
	tax_information: TaxInfo
	
	# Performance tracking
	performance_metrics: VendorPerformanceMetrics = Field(default_factory=VendorPerformanceMetrics)
	
	# APG multi-tenancy
	tenant_id: str
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	updated_by: str | None = None
	
	model_config = MODEL_CONFIG
	
	@field_validator('vendor_code')
	@classmethod
	def validate_vendor_code(cls, v: str) -> str:
		"""Validate vendor code format"""
		assert len(v) >= 3, "Vendor code must be at least 3 characters"
		assert len(v) <= 20, "Vendor code must be no more than 20 characters"
		assert v.isalnum() or all(c.isalnum() or c in '-_' for c in v), "Vendor code can only contain alphanumeric characters, hyphens, and underscores"
		return v.upper()
	
	def _log_vendor_update(self, field: str, old_value: Any, new_value: Any) -> None:
		"""Log vendor data changes for audit"""
		print(f"Vendor {self.id} field '{field}' changed from {old_value} to {new_value}")


@dataclass
class APInvoiceLine(BaseModel):
	"""Invoice line item with GL coding"""
	id: str = Field(default_factory=uuid7str)
	line_number: int
	description: str
	quantity: Decimal = Field(max_digits=15, decimal_places=4)
	unit_price: Decimal = Field(max_digits=15, decimal_places=4)
	line_amount: Decimal = Field(max_digits=15, decimal_places=2)
	
	# GL coding
	gl_account_code: str
	cost_center: str | None = None
	department: str | None = None
	project_code: str | None = None
	
	# Tax information
	tax_code: str | None = None
	tax_amount: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	
	# Three-way matching
	purchase_order_number: str | None = None
	po_line_number: int | None = None
	receipt_number: str | None = None
	
	model_config = MODEL_CONFIG
	
	@field_validator('line_amount')
	@classmethod
	def calculate_line_amount(cls, v: Decimal, info) -> Decimal:
		"""Auto-calculate line amount if not provided"""
		if 'quantity' in info.data and 'unit_price' in info.data:
			calculated = info.data['quantity'] * info.data['unit_price']
			if v != calculated:
				return calculated
		return v


@dataclass
class APInvoice(BaseModel):
	"""Vendor invoice header with processing status"""
	id: str = Field(default_factory=uuid7str)
	invoice_number: str
	vendor_id: str
	vendor_invoice_number: str
	
	# Dates
	invoice_date: date
	due_date: date
	received_date: date = Field(default_factory=date.today)
	
	# Amounts
	subtotal_amount: Decimal = Field(max_digits=15, decimal_places=2)
	tax_amount: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	total_amount: Decimal = Field(max_digits=15, decimal_places=2)
	
	# Currency
	currency_code: str = "USD"
	exchange_rate: Decimal = Field(default=Decimal('1.00'), max_digits=10, decimal_places=6)
	
	# Status and workflow
	status: InvoiceStatus = InvoiceStatus.PENDING
	approval_workflow_id: str | None = None
	
	# Processing information
	payment_terms: PaymentTerms
	line_items: List[APInvoiceLine] = Field(default_factory=list)
	
	# Three-way matching
	matching_status: MatchingStatus = MatchingStatus.NOT_MATCHED
	purchase_order_number: str | None = None
	
	# Document management
	document_id: str | None = None
	ocr_confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
	
	# APG multi-tenancy
	tenant_id: str
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	updated_by: str | None = None
	
	model_config = MODEL_CONFIG
	
	@field_validator('total_amount')
	@classmethod
	def validate_total_amount(cls, v: Decimal, info) -> Decimal:
		"""Validate total amount equals subtotal + tax"""
		if 'subtotal_amount' in info.data and 'tax_amount' in info.data:
			calculated_total = info.data['subtotal_amount'] + info.data['tax_amount']
			if abs(v - calculated_total) > Decimal('0.01'):
				raise ValueError(f"Total amount {v} does not match subtotal + tax {calculated_total}")
		return v
	
	def _log_invoice_status_change(self, old_status: InvoiceStatus, new_status: InvoiceStatus) -> None:
		"""Log invoice status changes for audit"""
		print(f"Invoice {self.invoice_number} status changed from {old_status} to {new_status}")


@dataclass
class APPaymentLine(BaseModel):
	"""Payment allocation to invoices"""
	id: str = Field(default_factory=uuid7str)
	invoice_id: str
	invoice_number: str
	payment_amount: Decimal = Field(max_digits=15, decimal_places=2)
	discount_taken: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	
	model_config = MODEL_CONFIG


@dataclass
class APPayment(BaseModel):
	"""Payment header with method and status"""
	id: str = Field(default_factory=uuid7str)
	payment_number: str
	vendor_id: str
	
	# Payment details
	payment_method: PaymentMethod
	payment_amount: Decimal = Field(max_digits=15, decimal_places=2)
	currency_code: str = "USD"
	exchange_rate: Decimal = Field(default=Decimal('1.00'), max_digits=10, decimal_places=6)
	
	# Dates
	payment_date: date
	scheduled_date: date | None = None
	cleared_date: date | None = None
	
	# Status
	status: PaymentStatus = PaymentStatus.SCHEDULED
	
	# Banking information
	bank_account_id: str | None = None
	check_number: str | None = None
	reference_number: str | None = None
	
	# Payment allocation
	payment_lines: List[APPaymentLine] = Field(default_factory=list)
	
	# APG multi-tenancy
	tenant_id: str
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	updated_by: str | None = None
	
	model_config = MODEL_CONFIG
	
	def _log_payment_processing(self, status_change: str) -> None:
		"""Log payment processing events"""
		print(f"Payment {self.payment_number} {status_change} - Amount: {self.payment_amount}")


@dataclass
class APApprovalStep(BaseModel):
	"""Individual approval step in workflow"""
	id: str = Field(default_factory=uuid7str)
	step_number: int
	approver_id: str
	approver_name: str
	status: ApprovalStatus = ApprovalStatus.PENDING
	
	# Approval details
	approval_date: datetime | None = None
	comments: str | None = None
	
	# Delegation
	delegated_to: str | None = None
	delegated_at: datetime | None = None
	
	model_config = MODEL_CONFIG


@dataclass
class APApprovalWorkflow(BaseModel):
	"""Approval workflow with routing and status"""
	id: str = Field(default_factory=uuid7str)
	workflow_type: str  # invoice, payment, expense
	entity_id: str  # invoice_id, payment_id, etc.
	entity_number: str  # human-readable reference
	
	# Workflow status
	status: ApprovalStatus = ApprovalStatus.PENDING
	priority: str = "normal"  # low, normal, high, urgent
	
	# Approval steps
	approval_steps: List[APApprovalStep] = Field(default_factory=list)
	current_step: int = 1
	
	# Dates
	initiated_at: datetime = Field(default_factory=datetime.utcnow)
	completed_at: datetime | None = None
	due_date: datetime | None = None
	
	# APG multi-tenancy
	tenant_id: str
	
	# Audit fields
	created_by: str
	
	model_config = MODEL_CONFIG
	
	def _log_workflow_progress(self, step: int, action: str) -> None:
		"""Log workflow progress for monitoring"""
		print(f"Workflow {self.id} step {step}: {action}")


@dataclass
class APExpenseLine(BaseModel):
	"""Expense report line item"""
	id: str = Field(default_factory=uuid7str)
	line_number: int
	expense_date: date
	description: str
	category_code: str
	amount: Decimal = Field(max_digits=15, decimal_places=2)
	
	# GL coding
	gl_account_code: str
	cost_center: str | None = None
	department: str | None = None
	project_code: str | None = None
	
	# Receipt information
	receipt_number: str | None = None
	merchant_name: str | None = None
	
	model_config = MODEL_CONFIG


@dataclass
class APExpenseReport(BaseModel):
	"""Employee expense report"""
	id: str = Field(default_factory=uuid7str)
	report_number: str
	employee_id: str
	employee_name: str
	
	# Report details
	report_title: str
	description: str | None = None
	total_amount: Decimal = Field(max_digits=15, decimal_places=2)
	
	# Status
	status: InvoiceStatus = InvoiceStatus.PENDING
	approval_workflow_id: str | None = None
	
	# Dates
	report_date: date = Field(default_factory=date.today)
	submitted_date: date | None = None
	
	# Line items
	expense_lines: List[APExpenseLine] = Field(default_factory=list)
	
	# APG multi-tenancy
	tenant_id: str
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	updated_by: str | None = None
	
	model_config = MODEL_CONFIG


@dataclass
class APTaxCode(BaseModel):
	"""Tax code configuration"""
	id: str = Field(default_factory=uuid7str)
	code: str
	name: str
	description: str | None = None
	rate: Decimal = Field(max_digits=5, decimal_places=4)
	gl_account_code: str | None = None
	is_active: bool = True
	
	# APG multi-tenancy
	tenant_id: str
	
	model_config = MODEL_CONFIG


@dataclass
class APAging(BaseModel):
	"""Accounts payable aging analysis"""
	id: str = Field(default_factory=uuid7str)
	vendor_id: str
	vendor_name: str
	
	# Aging buckets (in days)
	current_amount: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	past_due_1_30: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	past_due_31_60: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	past_due_61_90: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	past_due_over_90: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	total_outstanding: Decimal = Field(default=Decimal('0.00'), max_digits=15, decimal_places=2)
	
	# Analysis date
	as_of_date: date = Field(default_factory=date.today)
	
	# APG multi-tenancy
	tenant_id: str
	
	model_config = MODEL_CONFIG


@dataclass
class APAnalytics(BaseModel):
	"""Performance metrics and KPIs"""
	id: str = Field(default_factory=uuid7str)
	metric_type: str  # processing_time, accuracy, cost_per_invoice, etc.
	metric_value: Decimal = Field(max_digits=15, decimal_places=6)
	measurement_unit: str  # seconds, percentage, dollars, etc.
	
	# Time period
	period_start: date
	period_end: date
	
	# Dimensions
	vendor_id: str | None = None
	department: str | None = None
	category: str | None = None
	
	# APG multi-tenancy
	tenant_id: str
	
	# Audit fields
	calculated_at: datetime = Field(default_factory=datetime.utcnow)
	
	model_config = MODEL_CONFIG


# Utility Models

@dataclass
class InvoiceProcessingResult(BaseModel):
	"""Result of AI-powered invoice processing"""
	invoice_id: str
	extracted_data: Dict[str, Any]
	suggested_gl_codes: List[Dict[str, str]]
	confidence_score: float = Field(ge=0.0, le=1.0)
	processing_time_ms: int
	validation_errors: List[str] = Field(default_factory=list)
	
	model_config = MODEL_CONFIG


@dataclass
class CashFlowForecast(BaseModel):
	"""AI-powered cash flow forecast"""
	forecast_id: str = Field(default_factory=uuid7str)
	tenant_id: str
	forecast_horizon_days: int
	
	# Projections
	daily_projections: List[Dict[str, Any]] = Field(default_factory=list)
	confidence_intervals: Dict[str, float] = Field(default_factory=dict)
	risk_factors: List[str] = Field(default_factory=list)
	optimization_recommendations: List[str] = Field(default_factory=list)
	
	# Metadata
	generated_at: datetime = Field(default_factory=datetime.utcnow)
	model_version: str = "1.0"
	
	model_config = MODEL_CONFIG


# Runtime validation functions

async def validate_vendor_data(vendor_data: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
	"""Validate vendor data with business rules"""
	assert vendor_data is not None, "Vendor data must be provided"
	assert tenant_id is not None, "Tenant ID must be provided"
	
	errors = []
	warnings = []
	
	# Validate required fields
	required_fields = ['vendor_code', 'legal_name', 'vendor_type', 'primary_contact']
	for field in required_fields:
		if field not in vendor_data or not vendor_data[field]:
			errors.append(f"Required field '{field}' is missing")
	
	# Validate vendor code uniqueness (would typically check database)
	# This is a placeholder for actual database validation
	if 'vendor_code' in vendor_data:
		vendor_code = vendor_data['vendor_code']
		if len(vendor_code) < 3:
			errors.append("Vendor code must be at least 3 characters")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}


async def validate_invoice_data(invoice_data: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
	"""Validate invoice data with business rules"""
	assert invoice_data is not None, "Invoice data must be provided"
	assert tenant_id is not None, "Tenant ID must be provided"
	
	errors = []
	warnings = []
	
	# Validate required fields
	required_fields = ['vendor_id', 'invoice_number', 'invoice_date', 'total_amount']
	for field in required_fields:
		if field not in invoice_data or invoice_data[field] is None:
			errors.append(f"Required field '{field}' is missing")
	
	# Validate amounts
	if 'total_amount' in invoice_data:
		total = Decimal(str(invoice_data['total_amount']))
		if total <= 0:
			errors.append("Total amount must be greater than zero")
	
	# Validate dates
	if 'invoice_date' in invoice_data and 'due_date' in invoice_data:
		if invoice_data['due_date'] < invoice_data['invoice_date']:
			errors.append("Due date cannot be before invoice date")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}


async def _log_model_validation(model_type: str, model_id: str, result: str) -> None:
	"""Log model validation results for monitoring"""
	print(f"Model validation - {model_type} {model_id}: {result}")


# Model factory functions for testing

def create_sample_vendor(tenant_id: str) -> APVendor:
	"""Create sample vendor for testing"""
	assert tenant_id is not None, "Tenant ID must be provided"
	
	return APVendor(
		vendor_code="ACME001",
		legal_name="ACME Corporation",
		vendor_type=VendorType.SUPPLIER,
		primary_contact=ContactInfo(
			name="John Smith",
			email="john.smith@acme.com",
			phone="555-123-4567"
		),
		payment_terms=PaymentTerms(
			code="NET_30",
			name="Net 30",
			net_days=30
		),
		tax_information=TaxInfo(
			tax_id="12-3456789",
			tax_id_type="ein"
		),
		tenant_id=tenant_id,
		created_by="system"
	)


def create_sample_invoice(vendor_id: str, tenant_id: str) -> APInvoice:
	"""Create sample invoice for testing"""
	assert vendor_id is not None, "Vendor ID must be provided"
	assert tenant_id is not None, "Tenant ID must be provided"
	
	return APInvoice(
		invoice_number="INV-001",
		vendor_id=vendor_id,
		vendor_invoice_number="ACME-12345",
		invoice_date=date.today(),
		due_date=date.today(),
		subtotal_amount=Decimal('1000.00'),
		tax_amount=Decimal('85.00'),
		total_amount=Decimal('1085.00'),
		payment_terms=PaymentTerms(
			code="NET_30",
			name="Net 30",
			net_days=30
		),
		tenant_id=tenant_id,
		created_by="system"
	)


# Export all models and utilities
__all__ = [
	# Enums
	'VendorStatus', 'VendorType', 'InvoiceStatus', 'PaymentStatus', 
	'PaymentMethod', 'ApprovalStatus', 'MatchingStatus',
	
	# Core Models
	'APVendor', 'APInvoice', 'APInvoiceLine', 'APPayment', 'APPaymentLine',
	'APApprovalWorkflow', 'APApprovalStep', 'APExpenseReport', 'APExpenseLine',
	'APTaxCode', 'APAging', 'APAnalytics',
	
	# Supporting Models
	'ContactInfo', 'Address', 'BankingInfo', 'TaxInfo', 'PaymentTerms',
	'VendorPerformanceMetrics', 'InvoiceProcessingResult', 'CashFlowForecast',
	
	# Validation Functions
	'validate_vendor_data', 'validate_invoice_data',
	
	# Factory Functions
	'create_sample_vendor', 'create_sample_invoice'
]