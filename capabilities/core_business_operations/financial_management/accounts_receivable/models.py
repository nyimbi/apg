"""
APG Accounts Receivable - Data Models

Enterprise-grade data models for the APG Accounts Receivable capability.
Implements CLAUDE.md standards with async Python, modern typing, and APG integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic import ConfigDict
from pydantic.types import EmailStr, PositiveFloat, PositiveInt
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from uuid_extensions import uuid7str


# =============================================================================
# Configuration and Validation
# =============================================================================

def _log_validation_error(field_name: str, value: Any, error: str) -> str:
	"""Log validation errors with consistent formatting."""
	return f"Validation failed for {field_name}: {value} - {error}"


def validate_positive_amount(value: Union[float, Decimal]) -> Decimal:
	"""Validate that monetary amounts are positive."""
	if value <= 0:
		raise ValueError(_log_validation_error("amount", value, "must be positive"))
	return Decimal(str(value))


def validate_currency_code(value: str) -> str:
	"""Validate ISO 4217 currency codes."""
	if len(value) != 3 or not value.isupper() or not value.isalpha():
		raise ValueError(_log_validation_error("currency_code", value, "must be 3-letter ISO code"))
	return value


def validate_email_format(value: str) -> str:
	"""Validate email format with additional business rules."""
	if not value or "@" not in value:
		raise ValueError(_log_validation_error("email", value, "invalid format"))
	return value.lower().strip()


def validate_phone_number(value: str) -> str:
	"""Validate phone number format."""
	cleaned = "".join(c for c in value if c.isdigit() or c in "+()-. ")
	if len(cleaned) < 10:
		raise ValueError(_log_validation_error("phone", value, "too short"))
	return cleaned


def validate_non_empty_string(value: str) -> str:
	"""Validate non-empty strings."""
	if not value or not value.strip():
		raise ValueError(_log_validation_error("string", value, "cannot be empty"))
	return value.strip()


# Type aliases with validation
PositiveAmount = Annotated[Decimal, AfterValidator(validate_positive_amount)]
CurrencyCode = Annotated[str, AfterValidator(validate_currency_code)]
ValidatedEmail = Annotated[str, AfterValidator(validate_email_format)]
ValidatedPhone = Annotated[str, AfterValidator(validate_phone_number)]
NonEmptyString = Annotated[str, AfterValidator(validate_non_empty_string)]


# =============================================================================
# Enumerations
# =============================================================================

class ARCustomerStatus(str, Enum):
	"""Customer status enumeration."""
	ACTIVE = "active"
	INACTIVE = "inactive"
	SUSPENDED = "suspended"
	PENDING_APPROVAL = "pending_approval"
	CREDIT_HOLD = "credit_hold"
	BLACKLISTED = "blacklisted"


class ARCustomerType(str, Enum):
	"""Customer type enumeration."""
	INDIVIDUAL = "individual"
	BUSINESS = "business"
	GOVERNMENT = "government"
	NON_PROFIT = "non_profit"
	PARTNER = "partner"
	INTERNAL = "internal"


class ARInvoiceStatus(str, Enum):
	"""Invoice status enumeration."""
	DRAFT = "draft"
	PENDING = "pending"
	SENT = "sent"
	PARTIALLY_PAID = "partially_paid"
	PAID = "paid"
	OVERDUE = "overdue"
	DISPUTED = "disputed"
	CANCELLED = "cancelled"
	WRITTEN_OFF = "written_off"


class ARPaymentStatus(str, Enum):
	"""Payment status enumeration."""
	PENDING = "pending"
	PROCESSING = "processing"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	REFUNDED = "refunded"
	DISPUTED = "disputed"


class ARPaymentMethod(str, Enum):
	"""Payment method enumeration."""
	CASH = "cash"
	CHECK = "check"
	CREDIT_CARD = "credit_card"
	DEBIT_CARD = "debit_card"
	ACH = "ach"
	WIRE = "wire"
	BANK_TRANSFER = "bank_transfer"
	DIGITAL_WALLET = "digital_wallet"
	CRYPTOCURRENCY = "cryptocurrency"


class ARCollectionPriority(str, Enum):
	"""Collection priority enumeration."""
	LOW = "low"
	NORMAL = "normal"
	HIGH = "high"
	URGENT = "urgent"
	LEGAL = "legal"


class ARDisputeStatus(str, Enum):
	"""Dispute status enumeration."""
	OPEN = "open"
	INVESTIGATING = "investigating"
	PENDING_CUSTOMER = "pending_customer"
	PENDING_INTERNAL = "pending_internal"
	RESOLVED = "resolved"
	ESCALATED = "escalated"
	CLOSED = "closed"


class ARCreditRating(str, Enum):
	"""Credit rating enumeration."""
	EXCELLENT = "excellent"  # AAA
	VERY_GOOD = "very_good"  # AA
	GOOD = "good"  # A
	FAIR = "fair"  # BBB
	POOR = "poor"  # BB
	VERY_POOR = "very_poor"  # B
	DEFAULT_RISK = "default_risk"  # CCC and below


# =============================================================================
# Base Models
# =============================================================================

class APGBaseModel(BaseModel):
	"""Base model with APG multi-tenant patterns and common fields."""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		validate_default=True
	)
	
	# APG Integration Fields
	id: str = Field(default_factory=uuid7str, description="Unique identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# Audit Fields (APG audit_compliance integration)
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: str = Field(..., description="User who created the record")
	updated_by: str = Field(..., description="User who last updated the record")
	
	# Versioning for optimistic locking
	version: int = Field(default=1, description="Record version for optimistic locking")
	
	# Soft delete support
	is_deleted: bool = Field(default=False, description="Soft delete flag")
	deleted_at: Optional[datetime] = Field(default=None, description="Deletion timestamp")
	deleted_by: Optional[str] = Field(default=None, description="User who deleted the record")

	def _log_model_validation(self, action: str) -> str:
		"""Log model validation actions."""
		return f"Model {self.__class__.__name__} {action}: {self.id}"

	@root_validator
	def validate_apg_integration(cls, values: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate APG integration requirements."""
		assert values.get('tenant_id'), "tenant_id required for APG multi-tenancy"
		assert values.get('created_by'), "created_by required for audit compliance"
		return values


# =============================================================================
# Customer Management Models
# =============================================================================

class ARCustomerContact(BaseModel):
	"""Customer contact information."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	contact_type: str = Field(..., description="billing, shipping, primary, etc.")
	first_name: NonEmptyString = Field(..., max_length=100)
	last_name: NonEmptyString = Field(..., max_length=100)
	title: Optional[str] = Field(None, max_length=100)
	email: ValidatedEmail = Field(..., max_length=255)
	phone: ValidatedPhone = Field(..., max_length=50)
	mobile: Optional[ValidatedPhone] = Field(None, max_length=50)
	fax: Optional[str] = Field(None, max_length=50)
	is_primary: bool = Field(default=False)
	is_active: bool = Field(default=True)


class ARCustomerAddress(BaseModel):
	"""Customer address information."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	address_type: str = Field(..., description="billing, shipping, headquarters")
	line1: NonEmptyString = Field(..., max_length=255)
	line2: Optional[str] = Field(None, max_length=255)
	city: NonEmptyString = Field(..., max_length=100)
	state_province: NonEmptyString = Field(..., max_length=100)
	postal_code: NonEmptyString = Field(..., max_length=20)
	country_code: str = Field(..., min_length=2, max_length=3)
	is_primary: bool = Field(default=False)
	is_active: bool = Field(default=True)
	
	@validator('country_code')
	def validate_country_code(cls, v: str) -> str:
		"""Validate ISO country codes."""
		if len(v) not in [2, 3] or not v.isupper():
			raise ValueError(_log_validation_error("country_code", v, "must be ISO 2 or 3 letter code"))
		return v


class ARCustomerPaymentTerms(BaseModel):
	"""Customer payment terms configuration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	terms_code: NonEmptyString = Field(..., max_length=20)
	terms_name: NonEmptyString = Field(..., max_length=100)
	net_days: PositiveInt = Field(..., ge=1, le=365)
	discount_days: Optional[PositiveInt] = Field(None, ge=0, le=365)
	discount_percent: Optional[PositiveFloat] = Field(None, ge=0.0, le=100.0)
	late_fee_percent: Optional[PositiveFloat] = Field(None, ge=0.0, le=100.0)
	late_fee_grace_days: Optional[PositiveInt] = Field(None, ge=0, le=90)


class ARCustomerCreditInfo(BaseModel):
	"""Customer credit information and limits."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	credit_limit: PositiveAmount = Field(..., description="Credit limit amount")
	credit_rating: ARCreditRating = Field(...)
	risk_score: float = Field(..., ge=0.0, le=1.0, description="AI-calculated risk score")
	payment_behavior_score: float = Field(..., ge=0.0, le=1.0)
	days_sales_outstanding: Optional[float] = Field(None, ge=0.0)
	last_credit_review: Optional[date] = Field(None)
	next_credit_review: Optional[date] = Field(None)
	credit_analyst: Optional[str] = Field(None, description="User ID of credit analyst")
	external_credit_rating: Optional[str] = Field(None, max_length=10)
	external_rating_agency: Optional[str] = Field(None, max_length=100)


class ARCustomer(APGBaseModel):
	"""
	Comprehensive customer master data model for APG Accounts Receivable.
	
	Integrates with:
	- customer_relationship_management for customer data
	- auth_rbac for access control
	- audit_compliance for transaction logging
	"""
	
	# Basic Customer Information
	customer_code: NonEmptyString = Field(..., max_length=50, description="Unique customer code")
	legal_name: NonEmptyString = Field(..., max_length=255, description="Legal business name")
	trade_name: Optional[str] = Field(None, max_length=255, description="Trading name")
	customer_type: ARCustomerType = Field(...)
	status: ARCustomerStatus = Field(default=ARCustomerStatus.PENDING_APPROVAL)
	
	# Business Information
	industry_code: Optional[str] = Field(None, max_length=20, description="NAICS or SIC code")
	business_registration_number: Optional[str] = Field(None, max_length=100)
	tax_id: Optional[str] = Field(None, max_length=50, description="Tax identification number")
	vat_number: Optional[str] = Field(None, max_length=50)
	
	# Contact Information
	primary_contact: ARCustomerContact = Field(...)
	additional_contacts: List[ARCustomerContact] = Field(default_factory=list)
	addresses: List[ARCustomerAddress] = Field(default_factory=list)
	
	# Financial Configuration
	payment_terms: ARCustomerPaymentTerms = Field(...)
	credit_info: ARCustomerCreditInfo = Field(...)
	default_currency: CurrencyCode = Field(default="USD")
	
	# APG Integration Fields
	crm_customer_id: Optional[str] = Field(None, description="CRM system customer ID")
	document_folder_id: Optional[str] = Field(None, description="Document management folder")
	
	# Performance Metrics (calculated by ARAnalyticsService)
	total_outstanding_amount: Decimal = Field(default=Decimal('0.00'), ge=0)
	current_balance: Decimal = Field(default=Decimal('0.00'))
	overdue_amount: Decimal = Field(default=Decimal('0.00'), ge=0)
	last_payment_date: Optional[date] = Field(None)
	average_days_to_pay: Optional[float] = Field(None, ge=0.0)
	payment_history_months: Optional[int] = Field(None, ge=0, le=120)
	
	# Collection Information
	collection_priority: ARCollectionPriority = Field(default=ARCollectionPriority.NORMAL)
	last_collection_contact: Optional[date] = Field(None)
	collection_notes: Optional[str] = Field(None, max_length=2000)
	
	@validator('customer_code')
	def validate_customer_code(cls, v: str) -> str:
		"""Validate customer code format."""
		if not v.replace('-', '').replace('_', '').isalnum():
			raise ValueError(_log_validation_error("customer_code", v, "must be alphanumeric"))
		return v.upper()

	@root_validator
	def validate_customer_relationships(cls, values: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate customer data relationships."""
		# Ensure at least one address exists
		addresses = values.get('addresses', [])
		if not addresses:
			raise ValueError("At least one address is required")
		
		# Ensure only one primary address
		primary_addresses = [addr for addr in addresses if addr.is_primary]
		if len(primary_addresses) != 1:
			raise ValueError("Exactly one primary address is required")
		
		# Validate credit limit vs outstanding amount
		credit_info = values.get('credit_info')
		outstanding = values.get('total_outstanding_amount', Decimal('0.00'))
		if credit_info and outstanding > credit_info.credit_limit:
			# Log but don't fail - this is a business condition to monitor
			pass
		
		return values


# =============================================================================
# Invoice Management Models
# =============================================================================

class ARInvoiceLineItem(BaseModel):
	"""Invoice line item details."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	line_number: PositiveInt = Field(...)
	product_code: Optional[str] = Field(None, max_length=100)
	description: NonEmptyString = Field(..., max_length=500)
	quantity: Decimal = Field(..., gt=0, decimal_places=4)
	unit_price: Decimal = Field(..., ge=0, decimal_places=4)
	line_amount: Decimal = Field(..., ge=0, decimal_places=2)
	discount_amount: Decimal = Field(default=Decimal('0.00'), ge=0)
	tax_amount: Decimal = Field(default=Decimal('0.00'), ge=0)
	
	# GL Integration
	gl_account_code: Optional[str] = Field(None, max_length=50)
	cost_center: Optional[str] = Field(None, max_length=50)
	project_code: Optional[str] = Field(None, max_length=50)
	department: Optional[str] = Field(None, max_length=50)
	
	# Tax Information
	tax_code: Optional[str] = Field(None, max_length=20)
	tax_rate: Optional[Decimal] = Field(None, ge=0, le=1, decimal_places=4)
	is_tax_exempt: bool = Field(default=False)

	@validator('line_amount')
	def validate_line_amount(cls, v: Decimal, values: Dict[str, Any]) -> Decimal:
		"""Validate line amount calculation."""
		quantity = values.get('quantity', Decimal('0'))
		unit_price = values.get('unit_price', Decimal('0'))
		discount = values.get('discount_amount', Decimal('0'))
		
		expected_amount = (quantity * unit_price) - discount
		if abs(v - expected_amount) > Decimal('0.01'):  # Allow small rounding differences
			raise ValueError(_log_validation_error("line_amount", v, f"should be {expected_amount}"))
		
		return v


class ARInvoiceTax(BaseModel):
	"""Invoice tax calculation details."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	tax_code: NonEmptyString = Field(..., max_length=20)
	tax_name: NonEmptyString = Field(..., max_length=100)
	tax_rate: Decimal = Field(..., ge=0, le=1, decimal_places=4)
	tax_amount: PositiveAmount = Field(...)
	taxable_amount: PositiveAmount = Field(...)
	tax_jurisdiction: Optional[str] = Field(None, max_length=100)


class ARInvoice(APGBaseModel):
	"""
	Comprehensive invoice model for APG Accounts Receivable.
	
	Integrates with:
	- document_management for invoice documents
	- ai_orchestration for smart invoice processing
	- workflow_engine for approval workflows
	"""
	
	# Basic Invoice Information
	invoice_number: NonEmptyString = Field(..., max_length=100, description="Unique invoice number")
	customer_id: str = Field(..., description="Reference to ARCustomer")
	invoice_date: date = Field(...)
	due_date: date = Field(...)
	status: ARInvoiceStatus = Field(default=ARInvoiceStatus.DRAFT)
	
	# Reference Information
	purchase_order_number: Optional[str] = Field(None, max_length=100)
	sales_order_number: Optional[str] = Field(None, max_length=100)
	contract_number: Optional[str] = Field(None, max_length=100)
	external_reference: Optional[str] = Field(None, max_length=100)
	
	# Financial Information
	currency_code: CurrencyCode = Field(default="USD")
	exchange_rate: Decimal = Field(default=Decimal('1.00'), gt=0, decimal_places=6)
	subtotal_amount: PositiveAmount = Field(...)
	discount_amount: Decimal = Field(default=Decimal('0.00'), ge=0)
	tax_amount: Decimal = Field(default=Decimal('0.00'), ge=0)
	total_amount: PositiveAmount = Field(...)
	outstanding_amount: Decimal = Field(default=Decimal('0.00'), ge=0)
	
	# Line Items and Tax Details
	line_items: List[ARInvoiceLineItem] = Field(..., min_items=1)
	tax_details: List[ARInvoiceTax] = Field(default_factory=list)
	
	# Payment Terms
	payment_terms_code: NonEmptyString = Field(..., max_length=20)
	payment_terms_description: str = Field(..., max_length=200)
	early_payment_discount_percent: Optional[Decimal] = Field(None, ge=0, le=1)
	early_payment_discount_days: Optional[int] = Field(None, ge=0, le=365)
	
	# Collection and Aging
	aging_bucket: str = Field(default="current", description="current, 30, 60, 90, 90+")
	days_outstanding: int = Field(default=0, ge=0)
	collection_attempts: int = Field(default=0, ge=0)
	last_collection_date: Optional[date] = Field(None)
	collection_status: str = Field(default="none", max_length=50)
	
	# APG Integration Fields
	document_id: Optional[str] = Field(None, description="Document management system ID")
	workflow_id: Optional[str] = Field(None, description="Approval workflow ID")
	ai_processing_id: Optional[str] = Field(None, description="AI processing job ID")
	
	# AI/ML Insights
	fraud_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
	payment_prediction_score: float = Field(default=0.5, ge=0.0, le=1.0)
	collection_difficulty_score: float = Field(default=0.0, ge=0.0, le=1.0)
	ai_recommended_actions: List[str] = Field(default_factory=list)
	
	# Notes and Communication
	invoice_notes: Optional[str] = Field(None, max_length=2000)
	internal_notes: Optional[str] = Field(None, max_length=2000)
	customer_message: Optional[str] = Field(None, max_length=1000)

	@validator('due_date')
	def validate_due_date(cls, v: date, values: Dict[str, Any]) -> date:
		"""Validate due date is after invoice date."""
		invoice_date = values.get('invoice_date')
		if invoice_date and v < invoice_date:
			raise ValueError(_log_validation_error("due_date", v, "must be after invoice date"))
		return v

	@validator('total_amount')
	def validate_total_amount(cls, v: Decimal, values: Dict[str, Any]) -> Decimal:
		"""Validate total amount calculation."""
		subtotal = values.get('subtotal_amount', Decimal('0.00'))
		discount = values.get('discount_amount', Decimal('0.00'))
		tax = values.get('tax_amount', Decimal('0.00'))
		
		expected_total = subtotal - discount + tax
		if abs(v - expected_total) > Decimal('0.01'):
			raise ValueError(_log_validation_error("total_amount", v, f"should be {expected_total}"))
		
		return v

	@root_validator
	def validate_invoice_integrity(cls, values: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate invoice data integrity."""
		# Validate line items total
		line_items = values.get('line_items', [])
		if line_items:
			calculated_subtotal = sum(item.line_amount for item in line_items)
			declared_subtotal = values.get('subtotal_amount', Decimal('0.00'))
			if abs(calculated_subtotal - declared_subtotal) > Decimal('0.01'):
				raise ValueError(f"Line items total {calculated_subtotal} doesn't match subtotal {declared_subtotal}")
		
		# Validate outstanding amount doesn't exceed total
		outstanding = values.get('outstanding_amount', Decimal('0.00'))
		total = values.get('total_amount', Decimal('0.00'))
		if outstanding > total:
			raise ValueError("Outstanding amount cannot exceed total amount")
		
		return values


# =============================================================================
# Payment Management Models
# =============================================================================

class ARPaymentAllocation(BaseModel):
	"""Payment allocation to specific invoices."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	invoice_id: str = Field(..., description="Reference to ARInvoice")
	invoice_number: NonEmptyString = Field(..., max_length=100)
	allocated_amount: PositiveAmount = Field(...)
	discount_taken: Decimal = Field(default=Decimal('0.00'), ge=0)
	writeoff_amount: Decimal = Field(default=Decimal('0.00'), ge=0)
	allocation_date: date = Field(default_factory=date.today)
	allocation_notes: Optional[str] = Field(None, max_length=500)


class ARPayment(APGBaseModel):
	"""
	Comprehensive payment model for APG Accounts Receivable.
	
	Integrates with:
	- banking systems for payment processing
	- ai_orchestration for fraud detection
	- cash_management for cash application
	"""
	
	# Basic Payment Information
	payment_number: NonEmptyString = Field(..., max_length=100, description="Unique payment number")
	customer_id: str = Field(..., description="Reference to ARCustomer")
	payment_date: date = Field(...)
	received_date: date = Field(default_factory=date.today)
	status: ARPaymentStatus = Field(default=ARPaymentStatus.PENDING)
	
	# Payment Details
	payment_method: ARPaymentMethod = Field(...)
	payment_amount: PositiveAmount = Field(...)
	currency_code: CurrencyCode = Field(default="USD")
	exchange_rate: Decimal = Field(default=Decimal('1.00'), gt=0, decimal_places=6)
	
	# Bank/Payment Information
	bank_reference: Optional[str] = Field(None, max_length=100, description="Bank transaction reference")
	check_number: Optional[str] = Field(None, max_length=50)
	credit_card_last_four: Optional[str] = Field(None, max_length=4)
	payment_processor_id: Optional[str] = Field(None, max_length=100)
	
	# Cash Application
	allocations: List[ARPaymentAllocation] = Field(default_factory=list)
	unapplied_amount: Decimal = Field(default=Decimal('0.00'), ge=0)
	is_fully_applied: bool = Field(default=False)
	auto_applied: bool = Field(default=False, description="Applied by AI system")
	
	# APG Integration Fields
	banking_integration_id: Optional[str] = Field(None, description="Banking system integration ID")
	ai_matching_score: float = Field(default=0.0, ge=0.0, le=1.0, description="AI matching confidence")
	
	# Fraud Detection
	fraud_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
	fraud_check_status: str = Field(default="pending", max_length=50)
	fraud_alerts: List[str] = Field(default_factory=list)
	
	# Settlement Information
	settlement_date: Optional[date] = Field(None)
	settlement_reference: Optional[str] = Field(None, max_length=100)
	bank_fee_amount: Decimal = Field(default=Decimal('0.00'), ge=0)
	
	# Notes and Communication
	payment_memo: Optional[str] = Field(None, max_length=500)
	customer_notes: Optional[str] = Field(None, max_length=1000)
	internal_notes: Optional[str] = Field(None, max_length=2000)

	@validator('received_date')
	def validate_received_date(cls, v: date, values: Dict[str, Any]) -> date:
		"""Validate received date is not in the future."""
		if v > date.today():
			raise ValueError(_log_validation_error("received_date", v, "cannot be in the future"))
		return v

	@root_validator
	def validate_payment_integrity(cls, values: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate payment data integrity."""
		# Validate allocations don't exceed payment amount
		allocations = values.get('allocations', [])
		payment_amount = values.get('payment_amount', Decimal('0.00'))
		
		if allocations:
			total_allocated = sum(alloc.allocated_amount + alloc.discount_taken + alloc.writeoff_amount 
								for alloc in allocations)
			unapplied = values.get('unapplied_amount', Decimal('0.00'))
			
			if abs((total_allocated + unapplied) - payment_amount) > Decimal('0.01'):
				raise ValueError("Allocations + unapplied amount must equal payment amount")
		
		# Update application status
		unapplied = values.get('unapplied_amount', Decimal('0.00'))
		values['is_fully_applied'] = unapplied == Decimal('0.00')
		
		return values


# =============================================================================
# Collection Management Models
# =============================================================================

class ARCollectionActivity(APGBaseModel):
	"""
	Collection activity tracking for APG Accounts Receivable.
	
	Integrates with:
	- notification_engine for automated communications
	- workflow_engine for collection workflows
	- ai_orchestration for collection optimization
	"""
	
	# Basic Information
	customer_id: str = Field(..., description="Reference to ARCustomer")
	invoice_id: Optional[str] = Field(None, description="Specific invoice being collected")
	activity_type: str = Field(..., max_length=50, description="call, email, letter, legal")
	activity_date: date = Field(default_factory=date.today)
	
	# Activity Details
	subject: NonEmptyString = Field(..., max_length=200)
	description: str = Field(..., max_length=2000)
	outcome: Optional[str] = Field(None, max_length=500)
	next_action: Optional[str] = Field(None, max_length=500)
	next_action_date: Optional[date] = Field(None)
	
	# Contact Information
	contact_method: str = Field(..., max_length=50, description="phone, email, mail, in_person")
	contacted_person: Optional[str] = Field(None, max_length=200)
	contact_details: Optional[str] = Field(None, max_length=500)
	
	# Results and Follow-up
	promise_to_pay_date: Optional[date] = Field(None)
	promised_amount: Optional[PositiveAmount] = Field(None)
	payment_plan_proposed: bool = Field(default=False)
	dispute_raised: bool = Field(default=False)
	
	# Priority and Status
	priority: ARCollectionPriority = Field(default=ARCollectionPriority.NORMAL)
	status: str = Field(default="open", max_length=50)
	escalation_level: int = Field(default=0, ge=0, le=10)
	
	# APG Integration Fields
	notification_id: Optional[str] = Field(None, description="Notification engine message ID")
	workflow_step_id: Optional[str] = Field(None, description="Collection workflow step")
	ai_suggested: bool = Field(default=False, description="Suggested by AI system")
	ai_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
	
	# Performance Tracking
	cost_of_collection: Decimal = Field(default=Decimal('0.00'), ge=0)
	time_spent_minutes: int = Field(default=0, ge=0)
	effectiveness_score: Optional[float] = Field(None, ge=0.0, le=1.0)

	@validator('next_action_date')
	def validate_next_action_date(cls, v: Optional[date], values: Dict[str, Any]) -> Optional[date]:
		"""Validate next action date is in the future."""
		if v and v <= date.today():
			raise ValueError(_log_validation_error("next_action_date", v, "must be in the future"))
		return v


# =============================================================================
# Credit Assessment Models
# =============================================================================

class ARCreditAssessment(APGBaseModel):
	"""
	AI-powered credit assessment for customers.
	
	Integrates with:
	- federated_learning for credit scoring models
	- external credit bureaus
	- risk management systems
	"""
	
	# Assessment Information
	customer_id: str = Field(..., description="Reference to ARCustomer")
	assessment_date: date = Field(default_factory=date.today)
	assessment_type: str = Field(..., max_length=50, description="initial, periodic, trigger_based")
	assessment_reason: str = Field(..., max_length=200)
	
	# Credit Scoring Results
	credit_score: int = Field(..., ge=300, le=850, description="FICO-style credit score")
	risk_rating: ARCreditRating = Field(...)
	probability_of_default: float = Field(..., ge=0.0, le=1.0)
	recommended_credit_limit: PositiveAmount = Field(...)
	
	# AI Model Information
	model_version: str = Field(..., max_length=50, description="AI model version used")
	model_confidence: float = Field(..., ge=0.0, le=1.0)
	feature_importance: Dict[str, float] = Field(default_factory=dict)
	
	# External Credit Bureau Data
	external_scores: Dict[str, int] = Field(default_factory=dict, description="Scores from credit bureaus")
	external_reports: Dict[str, str] = Field(default_factory=dict, description="Credit bureau report IDs")
	
	# Financial Analysis
	annual_revenue: Optional[PositiveAmount] = Field(None)
	debt_to_income_ratio: Optional[float] = Field(None, ge=0.0, le=10.0)
	payment_history_score: float = Field(..., ge=0.0, le=1.0)
	industry_risk_factor: float = Field(default=1.0, ge=0.1, le=3.0)
	
	# Assessment Results
	approved_credit_limit: PositiveAmount = Field(...)
	payment_terms_approved: str = Field(..., max_length=100)
	special_conditions: List[str] = Field(default_factory=list)
	review_frequency_months: int = Field(default=12, ge=1, le=60)
	next_review_date: date = Field(...)
	
	# APG Integration Fields
	federated_learning_job_id: Optional[str] = Field(None)
	external_api_requests: List[str] = Field(default_factory=list)
	
	# Approval Workflow
	assessment_status: str = Field(default="draft", max_length=50)
	reviewed_by: Optional[str] = Field(None, description="Credit analyst user ID")
	approved_by: Optional[str] = Field(None, description="Manager user ID")
	approval_notes: Optional[str] = Field(None, max_length=1000)

	@validator('next_review_date')
	def validate_next_review_date(cls, v: date, values: Dict[str, Any]) -> date:
		"""Validate next review date is in the future."""
		assessment_date = values.get('assessment_date', date.today())
		if v <= assessment_date:
			raise ValueError(_log_validation_error("next_review_date", v, "must be after assessment date"))
		return v


# =============================================================================
# Dispute Management Models
# =============================================================================

class ARDispute(APGBaseModel):
	"""
	Comprehensive dispute management for APG Accounts Receivable.
	
	Integrates with:
	- document_management for dispute documentation
	- workflow_engine for dispute resolution process
	- customer_relationship_management for customer communication
	"""
	
	# Basic Dispute Information
	dispute_number: NonEmptyString = Field(..., max_length=100, description="Unique dispute number")
	customer_id: str = Field(..., description="Reference to ARCustomer")
	invoice_id: Optional[str] = Field(None, description="Disputed invoice")
	dispute_date: date = Field(default_factory=date.today)
	status: ARDisputeStatus = Field(default=ARDisputeStatus.OPEN)
	
	# Dispute Details
	dispute_type: str = Field(..., max_length=100, description="billing_error, quality, delivery, etc.")
	dispute_category: str = Field(..., max_length=100, description="amount, product, service, process")
	disputed_amount: PositiveAmount = Field(...)
	currency_code: CurrencyCode = Field(default="USD")
	
	# Description and Resolution
	customer_description: str = Field(..., max_length=2000, description="Customer's dispute description")
	internal_description: Optional[str] = Field(None, max_length=2000)
	resolution_description: Optional[str] = Field(None, max_length=2000)
	root_cause: Optional[str] = Field(None, max_length=500)
	
	# Timeline and Priority
	priority: str = Field(default="medium", max_length=20)
	expected_resolution_date: Optional[date] = Field(None)
	actual_resolution_date: Optional[date] = Field(None)
	escalation_level: int = Field(default=0, ge=0, le=5)
	
	# Financial Impact
	credit_amount: Decimal = Field(default=Decimal('0.00'), ge=0)
	refund_amount: Decimal = Field(default=Decimal('0.00'), ge=0)
	adjustment_amount: Decimal = Field(default=Decimal('0.00'))
	collection_hold: bool = Field(default=True, description="Hold collection activities")
	
	# Contact and Communication
	primary_contact_name: Optional[str] = Field(None, max_length=200)
	primary_contact_email: Optional[ValidatedEmail] = Field(None)
	primary_contact_phone: Optional[ValidatedPhone] = Field(None)
	customer_reference: Optional[str] = Field(None, max_length=100)
	
	# APG Integration Fields
	document_ids: List[str] = Field(default_factory=list, description="Supporting documents")
	workflow_id: Optional[str] = Field(None, description="Dispute resolution workflow")
	crm_case_id: Optional[str] = Field(None, description="CRM system case ID")
	
	# Assignment and Ownership
	assigned_to: Optional[str] = Field(None, description="User ID of assigned resolver")
	assigned_date: Optional[date] = Field(None)
	department: Optional[str] = Field(None, max_length=100)
	specialist_required: bool = Field(default=False)
	
	# Performance Tracking
	first_response_date: Optional[date] = Field(None)
	resolution_time_hours: Optional[int] = Field(None, ge=0)
	customer_satisfaction_score: Optional[int] = Field(None, ge=1, le=5)
	
	# Legal and Compliance
	legal_review_required: bool = Field(default=False)
	regulatory_implications: bool = Field(default=False)
	external_counsel_involved: bool = Field(default=False)
	compliance_notes: Optional[str] = Field(None, max_length=1000)

	@validator('actual_resolution_date')
	def validate_resolution_date(cls, v: Optional[date], values: Dict[str, Any]) -> Optional[date]:
		"""Validate resolution date is after dispute date."""
		if v:
			dispute_date = values.get('dispute_date', date.today())
			if v < dispute_date:
				raise ValueError(_log_validation_error("actual_resolution_date", v, "must be after dispute date"))
		return v

	@root_validator
	def validate_dispute_integrity(cls, values: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate dispute data integrity."""
		# If status is resolved, must have resolution date
		status = values.get('status')
		resolution_date = values.get('actual_resolution_date')
		
		if status == ARDisputeStatus.RESOLVED and not resolution_date:
			raise ValueError("Resolved disputes must have actual resolution date")
		
		# Validate financial amounts
		credit = values.get('credit_amount', Decimal('0.00'))
		refund = values.get('refund_amount', Decimal('0.00'))
		disputed = values.get('disputed_amount', Decimal('0.00'))
		
		if (credit + refund) > disputed:
			raise ValueError("Credit and refund amounts cannot exceed disputed amount")
		
		return values


# =============================================================================
# Cash Application Models
# =============================================================================

class ARCashApplication(APGBaseModel):
	"""
	AI-powered cash application for automated payment matching.
	
	Integrates with:
	- ai_orchestration for intelligent matching
	- banking systems for payment data
	- machine learning for pattern recognition
	"""
	
	# Basic Information
	application_number: NonEmptyString = Field(..., max_length=100)
	customer_id: str = Field(..., description="Reference to ARCustomer")
	payment_id: str = Field(..., description="Reference to ARPayment")
	application_date: date = Field(default_factory=date.today)
	
	# Matching Information
	matching_method: str = Field(..., max_length=50, description="ai_auto, manual, semi_auto")
	ai_matching_score: float = Field(default=0.0, ge=0.0, le=1.0)
	matching_rules_applied: List[str] = Field(default_factory=list)
	matching_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
	
	# Application Results
	matched_invoices: List[str] = Field(default_factory=list, description="Invoice IDs matched")
	total_applied_amount: Decimal = Field(default=Decimal('0.00'), ge=0)
	unmatched_amount: Decimal = Field(default=Decimal('0.00'), ge=0)
	application_status: str = Field(default="pending", max_length=50)
	
	# AI Model Information
	model_version: str = Field(..., max_length=50)
	feature_vector: Dict[str, float] = Field(default_factory=dict)
	alternative_matches: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Exception Handling
	exceptions: List[str] = Field(default_factory=list)
	requires_manual_review: bool = Field(default=False)
	review_reason: Optional[str] = Field(None, max_length=500)
	reviewed_by: Optional[str] = Field(None, description="User who reviewed")
	review_date: Optional[date] = Field(None)
	
	# Performance Metrics
	processing_time_seconds: Optional[float] = Field(None, ge=0.0)
	accuracy_score: Optional[float] = Field(None, ge=0.0, le=1.0)
	efficiency_gain: Optional[float] = Field(None, description="vs manual processing")
	
	# APG Integration Fields
	ai_job_id: Optional[str] = Field(None, description="AI orchestration job ID")
	banking_transaction_id: Optional[str] = Field(None)
	
	# Notes and Communication
	application_notes: Optional[str] = Field(None, max_length=1000)
	customer_notification_sent: bool = Field(default=False)
	internal_alerts: List[str] = Field(default_factory=list)

	@root_validator
	def validate_application_amounts(cls, values: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate cash application amounts."""
		applied = values.get('total_applied_amount', Decimal('0.00'))
		unmatched = values.get('unmatched_amount', Decimal('0.00'))
		
		# Note: payment_amount would come from the linked ARPayment
		# This validation would need to be done in the service layer
		# where we have access to the payment record
		
		if applied < 0 or unmatched < 0:
			raise ValueError("Applied and unmatched amounts must be non-negative")
		
		return values


# =============================================================================
# Model Registration and Export
# =============================================================================

# Export all models for use by other modules
__all__ = [
	# Base Models
	'APGBaseModel',
	
	# Customer Models
	'ARCustomer', 'ARCustomerContact', 'ARCustomerAddress', 
	'ARCustomerPaymentTerms', 'ARCustomerCreditInfo',
	
	# Invoice Models
	'ARInvoice', 'ARInvoiceLineItem', 'ARInvoiceTax',
	
	# Payment Models
	'ARPayment', 'ARPaymentAllocation',
	
	# Collection Models
	'ARCollectionActivity',
	
	# Credit Assessment Models
	'ARCreditAssessment',
	
	# Dispute Models
	'ARDispute',
	
	# Cash Application Models
	'ARCashApplication',
	
	# Enumerations
	'ARCustomerStatus', 'ARCustomerType', 'ARInvoiceStatus', 
	'ARPaymentStatus', 'ARPaymentMethod', 'ARCollectionPriority',
	'ARDisputeStatus', 'ARCreditRating',
	
	# Type Aliases
	'PositiveAmount', 'CurrencyCode', 'ValidatedEmail', 
	'ValidatedPhone', 'NonEmptyString'
]


def _log_model_summary() -> str:
	"""Log summary of registered models."""
	model_count = len([name for name in __all__ if not name.startswith('AR') or name.endswith('Status')])
	return f"APG Accounts Receivable models loaded: {model_count} models, {len(__all__)} total exports"