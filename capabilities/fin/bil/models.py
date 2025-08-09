"""
APG Billing Models

Comprehensive billing data models with BL prefix following APG standards.
Supports multi-tenant billing, subscriptions, usage tracking, and revenue optimization.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic.types import UUID4


class BillingCurrency(str, Enum):
	"""Supported billing currencies"""
	USD = "USD"
	EUR = "EUR"
	GBP = "GBP"
	KES = "KES"
	JPY = "JPY"
	CAD = "CAD"
	AUD = "AUD"


class SubscriptionStatus(str, Enum):
	"""Subscription status enumeration"""
	TRIAL = "trial"
	ACTIVE = "active"
	PAST_DUE = "past_due"
	CANCELLED = "cancelled"
	UNPAID = "unpaid"
	PAUSED = "paused"
	EXPIRED = "expired"


class InvoiceStatus(str, Enum):
	"""Invoice status enumeration"""
	DRAFT = "draft"
	PENDING = "pending"
	PAID = "paid"
	VOID = "void"
	UNCOLLECTIBLE = "uncollectible"
	OVERDUE = "overdue"


class PaymentStatus(str, Enum):
	"""Payment status enumeration"""
	PENDING = "pending"
	PROCESSING = "processing"
	SUCCEEDED = "succeeded"
	FAILED = "failed"
	CANCELLED = "cancelled"
	REFUNDED = "refunded"
	DISPUTED = "disputed"


class BillingPeriod(str, Enum):
	"""Billing period types"""
	DAILY = "daily"
	WEEKLY = "weekly"
	MONTHLY = "monthly"
	QUARTERLY = "quarterly"
	YEARLY = "yearly"
	USAGE_BASED = "usage_based"


class PricingModel(str, Enum):
	"""Pricing model types"""
	FLAT_RATE = "flat_rate"
	TIERED = "tiered"
	VOLUME = "volume"
	USAGE_BASED = "usage_based"
	FREEMIUM = "freemium"
	HYBRID = "hybrid"


class UsageAggregation(str, Enum):
	"""Usage aggregation methods"""
	SUM = "sum"
	MAX = "max"
	LAST_VALUE = "last_value"
	UNIQUE_COUNT = "unique_count"
	AVERAGE = "average"


class TaxType(str, Enum):
	"""Tax type enumeration"""
	VAT = "vat"
	GST = "gst"
	SALES_TAX = "sales_tax"
	EXCISE = "excise"
	CUSTOM = "custom"


# Core Models

class BLCustomer(BaseModel):
	"""Customer billing information model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique customer ID")
	tenant_id: str = Field(..., description="Tenant identifier for multi-tenancy")
	external_id: Optional[str] = Field(None, description="External system customer ID")
	
	# Customer Information
	name: str = Field(..., min_length=1, max_length=255, description="Customer name")
	email: str = Field(..., description="Customer email address")
	company: Optional[str] = Field(None, max_length=255, description="Company name")
	phone: Optional[str] = Field(None, max_length=50, description="Phone number")
	
	# Billing Details
	currency: BillingCurrency = Field(default=BillingCurrency.USD, description="Default billing currency")
	billing_address: Dict[str, Any] = Field(default_factory=dict, description="Billing address")
	tax_info: Dict[str, Any] = Field(default_factory=dict, description="Tax information")
	payment_terms: Optional[int] = Field(None, description="Payment terms in days")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	# Status
	active: bool = Field(default=True, description="Customer active status")
	credit_limit: Optional[Decimal] = Field(None, description="Customer credit limit")
	
	@field_validator('email')
	@classmethod
	def validate_email(cls, v: str) -> str:
		if '@' not in v:
			raise ValueError('Invalid email format')
		return v.lower()


class BLPlan(BaseModel):
	"""Billing plan definition model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique plan ID")
	tenant_id: str = Field(..., description="Tenant identifier")
	
	# Plan Details
	name: str = Field(..., min_length=1, max_length=255, description="Plan name")
	description: Optional[str] = Field(None, description="Plan description")
	external_id: Optional[str] = Field(None, description="External plan identifier")
	
	# Pricing
	pricing_model: PricingModel = Field(default=PricingModel.FLAT_RATE, description="Pricing model")
	base_price: Decimal = Field(default=Decimal('0'), description="Base subscription price")
	currency: BillingCurrency = Field(default=BillingCurrency.USD, description="Plan currency")
	billing_period: BillingPeriod = Field(default=BillingPeriod.MONTHLY, description="Billing period")
	
	# Usage-based pricing
	usage_charges: List[Dict[str, Any]] = Field(default_factory=list, description="Usage-based charges")
	included_usage: Dict[str, Decimal] = Field(default_factory=dict, description="Included usage quotas")
	
	# Features and Limits
	features: List[str] = Field(default_factory=list, description="Plan features")
	limits: Dict[str, Any] = Field(default_factory=dict, description="Plan limits")
	
	# Trial Settings
	trial_period_days: Optional[int] = Field(None, description="Trial period in days")
	trial_requires_payment: bool = Field(default=False, description="Trial requires payment method")
	
	# Plan Management
	active: bool = Field(default=True, description="Plan active status")
	version: int = Field(default=1, description="Plan version")
	archived: bool = Field(default=False, description="Plan archived status")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	@field_validator('base_price')
	@classmethod
	def validate_base_price(cls, v: Decimal) -> Decimal:
		if v < 0:
			raise ValueError('Base price cannot be negative')
		return v


class BLSubscription(BaseModel):
	"""Subscription management model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique subscription ID")
	tenant_id: str = Field(..., description="Tenant identifier")
	customer_id: str = Field(..., description="Customer ID")
	plan_id: str = Field(..., description="Billing plan ID")
	
	# Subscription Details
	external_id: Optional[str] = Field(None, description="External subscription ID")
	status: SubscriptionStatus = Field(default=SubscriptionStatus.TRIAL, description="Subscription status")
	
	# Billing Cycle
	current_period_start: datetime = Field(default_factory=datetime.utcnow, description="Current period start")
	current_period_end: datetime = Field(..., description="Current period end")
	billing_cycle_anchor: Optional[datetime] = Field(None, description="Billing cycle anchor date")
	
	# Trial Management
	trial_start: Optional[datetime] = Field(None, description="Trial start date")
	trial_end: Optional[datetime] = Field(None, description="Trial end date")
	
	# Pricing Overrides
	price_override: Optional[Decimal] = Field(None, description="Override price for this subscription")
	currency_override: Optional[BillingCurrency] = Field(None, description="Override currency")
	
	# Usage Tracking
	usage_reset_date: Optional[datetime] = Field(None, description="Usage reset date")
	included_usage_override: Dict[str, Decimal] = Field(default_factory=dict, description="Override included usage")
	
	# Discounts and Promotions
	applied_discounts: List[str] = Field(default_factory=list, description="Applied discount IDs")
	discount_amount: Decimal = Field(default=Decimal('0'), description="Total discount amount")
	
	# Lifecycle Management
	cancel_at_period_end: bool = Field(default=False, description="Cancel at period end")
	cancelled_at: Optional[datetime] = Field(None, description="Cancellation timestamp")
	cancellation_reason: Optional[str] = Field(None, description="Cancellation reason")
	
	# Pause/Resume
	paused_at: Optional[datetime] = Field(None, description="Pause timestamp")
	pause_reason: Optional[str] = Field(None, description="Pause reason")
	
	# Payment
	default_payment_method: Optional[str] = Field(None, description="Default payment method ID")
	collection_method: str = Field(default="charge_automatically", description="Collection method")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	# Analytics
	churn_score: Optional[float] = Field(None, description="Churn prediction score")
	lifetime_value: Optional[Decimal] = Field(None, description="Customer lifetime value")


class BLUsage(BaseModel):
	"""Usage tracking model for billing"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique usage record ID")
	tenant_id: str = Field(..., description="Tenant identifier")
	subscription_id: str = Field(..., description="Subscription ID")
	customer_id: str = Field(..., description="Customer ID")
	
	# Usage Details
	metric_name: str = Field(..., description="Usage metric name")
	quantity: Decimal = Field(..., description="Usage quantity")
	unit: str = Field(..., description="Usage unit")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Usage timestamp")
	
	# Billing Period
	billing_period_start: datetime = Field(..., description="Billing period start")
	billing_period_end: datetime = Field(..., description="Billing period end")
	
	# Pricing Context
	unit_price: Optional[Decimal] = Field(None, description="Unit price for this usage")
	total_amount: Optional[Decimal] = Field(None, description="Total amount for this usage")
	currency: BillingCurrency = Field(default=BillingCurrency.USD, description="Usage currency")
	
	# Aggregation
	aggregation_method: UsageAggregation = Field(default=UsageAggregation.SUM, description="Aggregation method")
	aggregation_key: Optional[str] = Field(None, description="Aggregation grouping key")
	
	# Context Information
	source_system: str = Field(..., description="Source system generating usage")
	source_id: Optional[str] = Field(None, description="Source record ID")
	resource_id: Optional[str] = Field(None, description="Resource identifier")
	
	# Processing Status
	processed: bool = Field(default=False, description="Usage processed for billing")
	processed_at: Optional[datetime] = Field(None, description="Processing timestamp")
	invoice_id: Optional[str] = Field(None, description="Associated invoice ID")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	
	@field_validator('quantity')
	@classmethod
	def validate_quantity(cls, v: Decimal) -> Decimal:
		if v < 0:
			raise ValueError('Usage quantity cannot be negative')
		return v


class BLInvoice(BaseModel):
	"""Invoice generation and management model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique invoice ID")
	tenant_id: str = Field(..., description="Tenant identifier")
	customer_id: str = Field(..., description="Customer ID")
	subscription_id: Optional[str] = Field(None, description="Subscription ID")
	
	# Invoice Details
	invoice_number: str = Field(..., description="Human-readable invoice number")
	external_id: Optional[str] = Field(None, description="External invoice ID")
	status: InvoiceStatus = Field(default=InvoiceStatus.DRAFT, description="Invoice status")
	
	# Amounts
	subtotal: Decimal = Field(default=Decimal('0'), description="Invoice subtotal")
	tax_amount: Decimal = Field(default=Decimal('0'), description="Tax amount")
	discount_amount: Decimal = Field(default=Decimal('0'), description="Discount amount")
	total: Decimal = Field(..., description="Invoice total")
	amount_paid: Decimal = Field(default=Decimal('0'), description="Amount paid")
	amount_due: Decimal = Field(default=Decimal('0'), description="Amount due")
	currency: BillingCurrency = Field(default=BillingCurrency.USD, description="Invoice currency")
	
	# Billing Period
	period_start: datetime = Field(..., description="Billing period start")
	period_end: datetime = Field(..., description="Billing period end")
	
	# Invoice Dates
	invoice_date: datetime = Field(default_factory=datetime.utcnow, description="Invoice date")
	due_date: datetime = Field(..., description="Payment due date")
	paid_at: Optional[datetime] = Field(None, description="Payment timestamp")
	
	# Line Items
	line_items: List[Dict[str, Any]] = Field(default_factory=list, description="Invoice line items")
	
	# Discounts and Taxes
	applied_discounts: List[Dict[str, Any]] = Field(default_factory=list, description="Applied discounts")
	tax_details: List[Dict[str, Any]] = Field(default_factory=list, description="Tax calculation details")
	
	# Payment Information
	payment_method: Optional[str] = Field(None, description="Payment method used")
	payment_intent_id: Optional[str] = Field(None, description="Payment intent ID")
	
	# Document Information
	pdf_url: Optional[str] = Field(None, description="PDF invoice URL")
	hosted_url: Optional[str] = Field(None, description="Hosted invoice URL")
	
	# Collection
	collection_method: str = Field(default="charge_automatically", description="Collection method")
	attempted_collections: int = Field(default=0, description="Collection attempt count")
	next_payment_attempt: Optional[datetime] = Field(None, description="Next payment attempt")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	@field_validator('total')
	@classmethod
	def validate_total(cls, v: Decimal) -> Decimal:
		if v < 0:
			raise ValueError('Invoice total cannot be negative')
		return v


class BLPayment(BaseModel):
	"""Payment processing and tracking model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique payment ID")
	tenant_id: str = Field(..., description="Tenant identifier")
	customer_id: str = Field(..., description="Customer ID")
	invoice_id: Optional[str] = Field(None, description="Associated invoice ID")
	
	# Payment Details
	external_id: Optional[str] = Field(None, description="External payment processor ID")
	payment_intent_id: Optional[str] = Field(None, description="Payment intent ID")
	status: PaymentStatus = Field(default=PaymentStatus.PENDING, description="Payment status")
	
	# Amount Information
	amount: Decimal = Field(..., description="Payment amount")
	currency: BillingCurrency = Field(default=BillingCurrency.USD, description="Payment currency")
	fee_amount: Optional[Decimal] = Field(None, description="Processing fee amount")
	net_amount: Optional[Decimal] = Field(None, description="Net amount after fees")
	
	# Payment Method
	payment_method_type: str = Field(..., description="Payment method type")
	payment_method_id: Optional[str] = Field(None, description="Payment method ID")
	payment_processor: str = Field(default="stripe", description="Payment processor")
	
	# Processing Information
	processed_at: Optional[datetime] = Field(None, description="Processing timestamp")
	settled_at: Optional[datetime] = Field(None, description="Settlement timestamp")
	failure_reason: Optional[str] = Field(None, description="Failure reason")
	failure_code: Optional[str] = Field(None, description="Failure code")
	
	# Refund Information
	refunded: bool = Field(default=False, description="Payment refunded")
	refund_amount: Decimal = Field(default=Decimal('0'), description="Refunded amount")
	refunded_at: Optional[datetime] = Field(None, description="Refund timestamp")
	
	# Dispute Information
	disputed: bool = Field(default=False, description="Payment disputed")
	dispute_reason: Optional[str] = Field(None, description="Dispute reason")
	disputed_at: Optional[datetime] = Field(None, description="Dispute timestamp")
	
	# Risk Assessment
	risk_score: Optional[float] = Field(None, description="Fraud risk score")
	risk_level: Optional[str] = Field(None, description="Risk level")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	@field_validator('amount')
	@classmethod
	def validate_amount(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError('Payment amount must be positive')
		return v


class BLPricingRule(BaseModel):
	"""Dynamic pricing rules model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique pricing rule ID")
	tenant_id: str = Field(..., description="Tenant identifier")
	
	# Rule Details
	name: str = Field(..., min_length=1, max_length=255, description="Rule name")
	description: Optional[str] = Field(None, description="Rule description")
	active: bool = Field(default=True, description="Rule active status")
	
	# Pricing Configuration
	metric_name: str = Field(..., description="Usage metric this rule applies to")
	pricing_tiers: List[Dict[str, Any]] = Field(default_factory=list, description="Pricing tier definitions")
	
	# Conditions
	conditions: Dict[str, Any] = Field(default_factory=dict, description="Rule application conditions")
	customer_segments: List[str] = Field(default_factory=list, description="Applicable customer segments")
	
	# Time-based Rules
	effective_date: Optional[datetime] = Field(None, description="Rule effective date")
	expiry_date: Optional[datetime] = Field(None, description="Rule expiry date")
	time_based_conditions: Dict[str, Any] = Field(default_factory=dict, description="Time-based conditions")
	
	# Priority and Conflicts
	priority: int = Field(default=100, description="Rule priority (lower = higher priority)")
	conflict_resolution: str = Field(default="highest_priority", description="Conflict resolution strategy")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class BLTax(BaseModel):
	"""Tax calculation and compliance model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique tax record ID")
	tenant_id: str = Field(..., description="Tenant identifier")
	invoice_id: str = Field(..., description="Associated invoice ID")
	
	# Tax Details
	tax_type: TaxType = Field(..., description="Type of tax")
	tax_name: str = Field(..., description="Tax name/description")
	tax_rate: Decimal = Field(..., description="Tax rate (as decimal)")
	tax_amount: Decimal = Field(..., description="Calculated tax amount")
	
	# Tax Jurisdiction
	country: str = Field(..., description="Tax country")
	state_province: Optional[str] = Field(None, description="State/province")
	city: Optional[str] = Field(None, description="City")
	postal_code: Optional[str] = Field(None, description="Postal code")
	
	# Tax Base
	taxable_amount: Decimal = Field(..., description="Amount subject to tax")
	exempt_amount: Decimal = Field(default=Decimal('0'), description="Tax-exempt amount")
	
	# Compliance
	tax_id: Optional[str] = Field(None, description="Tax registration ID")
	reverse_charge: bool = Field(default=False, description="Reverse charge applicable")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	
	@field_validator('tax_rate')
	@classmethod
	def validate_tax_rate(cls, v: Decimal) -> Decimal:
		if v < 0 or v > 1:
			raise ValueError('Tax rate must be between 0 and 1')
		return v


class BLDiscount(BaseModel):
	"""Discount and promotion management model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique discount ID")
	tenant_id: str = Field(..., description="Tenant identifier")
	
	# Discount Details
	name: str = Field(..., min_length=1, max_length=255, description="Discount name")
	code: Optional[str] = Field(None, description="Discount code")
	description: Optional[str] = Field(None, description="Discount description")
	
	# Discount Type
	discount_type: str = Field(..., description="Discount type (percentage, fixed, etc.)")
	discount_value: Decimal = Field(..., description="Discount value")
	currency: Optional[BillingCurrency] = Field(None, description="Currency for fixed discounts")
	
	# Validity
	valid_from: datetime = Field(default_factory=datetime.utcnow, description="Valid from date")
	valid_until: Optional[datetime] = Field(None, description="Valid until date")
	active: bool = Field(default=True, description="Discount active status")
	
	# Usage Limits
	max_uses: Optional[int] = Field(None, description="Maximum number of uses")
	max_uses_per_customer: Optional[int] = Field(None, description="Max uses per customer")
	current_uses: int = Field(default=0, description="Current usage count")
	
	# Applicability
	applicable_plans: List[str] = Field(default_factory=list, description="Applicable plan IDs")
	customer_segments: List[str] = Field(default_factory=list, description="Applicable customer segments")
	minimum_amount: Optional[Decimal] = Field(None, description="Minimum order amount")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class BLRevenue(BaseModel):
	"""Revenue recognition and tracking model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique revenue record ID")
	tenant_id: str = Field(..., description="Tenant identifier")
	subscription_id: Optional[str] = Field(None, description="Associated subscription ID")
	invoice_id: Optional[str] = Field(None, description="Associated invoice ID")
	
	# Revenue Details
	revenue_amount: Decimal = Field(..., description="Revenue amount")
	currency: BillingCurrency = Field(default=BillingCurrency.USD, description="Revenue currency")
	recognition_date: datetime = Field(..., description="Revenue recognition date")
	
	# Revenue Category
	revenue_type: str = Field(..., description="Revenue type (subscription, usage, one-time)")
	product_category: Optional[str] = Field(None, description="Product category")
	
	# Recognition Schedule
	deferred_revenue: Decimal = Field(default=Decimal('0'), description="Deferred revenue amount")
	recognized_revenue: Decimal = Field(default=Decimal('0'), description="Recognized revenue amount")
	recognition_schedule: List[Dict[str, Any]] = Field(default_factory=list, description="Revenue recognition schedule")
	
	# Performance Obligations
	performance_obligations: List[Dict[str, Any]] = Field(default_factory=list, description="Performance obligations")
	
	# Compliance
	accounting_period: str = Field(..., description="Accounting period")
	revenue_stream: str = Field(..., description="Revenue stream identifier")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


# Request/Response Models

class CreateSubscriptionRequest(BaseModel):
	"""Request model for creating subscriptions"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	customer_id: str = Field(..., description="Customer ID")
	plan_id: str = Field(..., description="Plan ID")
	trial_period_days: Optional[int] = Field(None, description="Trial period override")
	payment_method_id: Optional[str] = Field(None, description="Payment method ID")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class UsageSubmissionRequest(BaseModel):
	"""Request model for submitting usage data"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	subscription_id: str = Field(..., description="Subscription ID")
	metric_name: str = Field(..., description="Usage metric name")
	quantity: Decimal = Field(..., description="Usage quantity")
	timestamp: Optional[datetime] = Field(None, description="Usage timestamp")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class InvoiceGenerationRequest(BaseModel):
	"""Request model for generating invoices"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	subscription_id: str = Field(..., description="Subscription ID")
	billing_period_start: datetime = Field(..., description="Billing period start")
	billing_period_end: datetime = Field(..., description="Billing period end")
	include_usage: bool = Field(default=True, description="Include usage charges")
	auto_finalize: bool = Field(default=False, description="Auto-finalize invoice")


# Export all models
__all__ = [
	"BillingCurrency", "SubscriptionStatus", "InvoiceStatus", "PaymentStatus",
	"BillingPeriod", "PricingModel", "UsageAggregation", "TaxType",
	"BLCustomer", "BLPlan", "BLSubscription", "BLUsage", "BLInvoice",
	"BLPayment", "BLPricingRule", "BLTax", "BLDiscount", "BLRevenue",
	"CreateSubscriptionRequest", "UsageSubmissionRequest", "InvoiceGenerationRequest"
]