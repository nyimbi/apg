"""
APG Payment Gateway Data Models

Comprehensive data models following APG coding standards with async Python,
modern typing, and Pydantic v2 validation for payment processing.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, Any, List, Optional
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic import AfterValidator
from annotated_types import Annotated
from sqlalchemy import Column, String, Integer, DateTime, Text, Numeric, Boolean, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()

class PaymentStatus(str, Enum):
	"""Payment transaction status enumeration"""
	PENDING = "pending"
	PROCESSING = "processing"
	AUTHORIZED = "authorized"
	CAPTURED = "captured"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	REFUNDED = "refunded"
	PARTIALLY_REFUNDED = "partially_refunded"
	DISPUTED = "disputed"
	CHARGEBACK = "chargeback"

class PaymentMethodType(str, Enum):
	"""Payment method type enumeration"""
	CREDIT_CARD = "credit_card"
	DEBIT_CARD = "debit_card"
	BANK_TRANSFER = "bank_transfer"
	ACH = "ach"
	DIGITAL_WALLET = "digital_wallet"
	APPLE_PAY = "apple_pay"
	GOOGLE_PAY = "google_pay"
	PAYPAL = "paypal"
	CRYPTOCURRENCY = "cryptocurrency"
	BUY_NOW_PAY_LATER = "bnpl"
	BANK_REDIRECT = "bank_redirect"
	# African Mobile Money
	MPESA = "mpesa"
	AIRTEL_MONEY = "airtel_money"
	TIGO_PESA = "tigo_pesa"
	MOBILE_MONEY = "mobile_money"

class FraudRiskLevel(str, Enum):
	"""Fraud risk level enumeration"""
	VERY_LOW = "very_low"
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	VERY_HIGH = "very_high"
	BLOCKED = "blocked"

class MerchantStatus(str, Enum):
	"""Merchant account status enumeration"""
	PENDING = "pending"
	ACTIVE = "active"
	SUSPENDED = "suspended"
	CLOSED = "closed"
	UNDER_REVIEW = "under_review"

def validate_currency_code(currency: str) -> str:
	"""Validate ISO 4217 currency codes"""
	if len(currency) != 3 or not currency.isupper():
		raise ValueError("Currency must be 3-letter uppercase ISO 4217 code")
	return currency

def validate_amount(amount: int) -> int:
	"""Validate payment amount (in cents)"""
	if amount < 0:
		raise ValueError("Amount cannot be negative")
	if amount > 999999999999:  # $9.99 billion limit
		raise ValueError("Amount exceeds maximum allowed")
	return amount

# Type aliases with validation
CurrencyCode = Annotated[str, AfterValidator(validate_currency_code)]
PaymentAmount = Annotated[int, AfterValidator(validate_amount)]

class PaymentTransaction(BaseModel):
	"""
	Payment transaction model with comprehensive tracking and APG integration
	"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		json_encoders={
			datetime: lambda v: v.isoformat(),
			Decimal: lambda v: str(v)
		}
	)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Unique transaction ID")
	tenant_id: str = Field(..., description="APG tenant identifier")
	merchant_id: str = Field(..., description="Merchant account identifier")
	customer_id: str | None = Field(None, description="Customer identifier")
	
	# Transaction details
	amount: PaymentAmount = Field(..., description="Amount in smallest currency unit (cents)")
	currency: CurrencyCode = Field(default="USD", description="ISO 4217 currency code")
	description: str | None = Field(None, max_length=500, description="Transaction description")
	reference: str | None = Field(None, max_length=100, description="Merchant reference")
	
	# Payment method
	payment_method_id: str = Field(..., description="Payment method used")
	payment_method_type: PaymentMethodType = Field(..., description="Type of payment method")
	
	# Status and processing
	status: PaymentStatus = Field(default=PaymentStatus.PENDING, description="Transaction status")
	processor: str | None = Field(None, description="Payment processor used")
	processor_transaction_id: str | None = Field(None, description="Processor transaction ID")
	
	# Fraud and security
	fraud_score: float | None = Field(None, ge=0.0, le=1.0, description="AI fraud score (0-1)")
	fraud_risk_level: FraudRiskLevel | None = Field(None, description="Fraud risk assessment")
	
	# Financial details
	processing_fee: int | None = Field(None, description="Processing fee in cents")
	net_amount: int | None = Field(None, description="Net amount after fees")
	
	# Metadata and context
	metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	customer_ip: str | None = Field(None, description="Customer IP address")
	user_agent: str | None = Field(None, description="Customer user agent")
	
	# APG integration
	business_context: dict[str, Any] = Field(default_factory=dict, description="APG business context")
	workflow_triggers: list[str] = Field(default_factory=list, description="APG workflow triggers")
	
	# Timestamps
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	processed_at: datetime | None = Field(None, description="Processing completion time")
	
	# Audit trail
	created_by: str = Field(..., description="User who created transaction")
	updated_by: str | None = Field(None, description="User who last updated transaction")
	
	@field_validator('amount')
	@classmethod
	def validate_amount_positive(cls, v):
		if v <= 0:
			raise ValueError('Amount must be positive')
		return v
	
	def _log_transaction_created(self):
		"""Log transaction creation with APG patterns"""
		print(f"💳 Payment transaction created: {self.id}")
		print(f"   Amount: {self.amount} {self.currency}")
		print(f"   Merchant: {self.merchant_id}")
		print(f"   Status: {self.status}")

class PaymentMethod(BaseModel):
	"""
	Payment method model with tokenization and security features
	"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Payment method ID")
	tenant_id: str = Field(..., description="APG tenant identifier")
	customer_id: str = Field(..., description="Customer who owns this payment method")
	
	# Payment method details
	type: PaymentMethodType = Field(..., description="Payment method type")
	provider: str = Field(..., description="Payment provider (stripe, adyen, etc.)")
	token: str = Field(..., description="Tokenized payment method reference")
	
	# Card details (if applicable)
	card_brand: str | None = Field(None, description="Card brand (visa, mastercard, etc.)")
	card_last4: str | None = Field(None, max_length=4, description="Last 4 digits of card")
	card_exp_month: int | None = Field(None, ge=1, le=12, description="Card expiration month")
	card_exp_year: int | None = Field(None, ge=2024, le=2050, description="Card expiration year")
	card_country: str | None = Field(None, max_length=2, description="Card issuing country")
	
	# Bank details (if applicable)
	bank_name: str | None = Field(None, description="Bank name")
	account_type: str | None = Field(None, description="Account type (checking, savings)")
	routing_number: str | None = Field(None, description="Bank routing number")
	account_last4: str | None = Field(None, max_length=4, description="Last 4 digits of account")
	
	# Digital wallet details
	wallet_type: str | None = Field(None, description="Digital wallet type")
	wallet_email: str | None = Field(None, description="Wallet email address")
	
	# Verification and security
	is_verified: bool = Field(default=False, description="Payment method verified")
	verification_method: str | None = Field(None, description="How payment method was verified")
	
	# Metadata
	nickname: str | None = Field(None, max_length=100, description="Customer nickname for method")
	is_default: bool = Field(default=False, description="Default payment method for customer")
	metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	
	# Timestamps
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	last_used_at: datetime | None = Field(None, description="Last time method was used")
	
	def _log_payment_method_created(self):
		"""Log payment method creation"""
		print(f"💳 Payment method created: {self.id}")
		print(f"   Type: {self.type}")
		print(f"   Customer: {self.customer_id}")
		print(f"   Provider: {self.provider}")

class Merchant(BaseModel):
	"""
	Merchant account model with comprehensive business information
	"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Merchant account ID")
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# Business information
	business_name: str = Field(..., max_length=200, description="Legal business name")
	display_name: str = Field(..., max_length=100, description="Display name for customers")
	business_type: str = Field(..., description="Type of business")
	industry: str = Field(..., description="Industry category")
	
	# Contact information
	email: str = Field(..., description="Primary business email")
	phone: str | None = Field(None, description="Business phone number")
	website: str | None = Field(None, description="Business website URL")
	
	# Address information
	address_line1: str = Field(..., description="Street address")
	address_line2: str | None = Field(None, description="Additional address info")
	city: str = Field(..., description="City")
	state: str | None = Field(None, description="State/province")
	postal_code: str = Field(..., description="Postal/ZIP code")
	country: str = Field(..., max_length=2, description="ISO country code")
	
	# Processing configuration
	supported_currencies: list[str] = Field(default_factory=lambda: ["USD"], description="Supported currencies")
	processing_countries: list[str] = Field(default_factory=list, description="Countries where processing is enabled")
	
	# Financial settings
	settlement_schedule: str = Field(default="daily", description="Settlement schedule")
	settlement_currency: CurrencyCode = Field(default="USD", description="Settlement currency")
	processing_fees: dict[str, Any] = Field(default_factory=dict, description="Fee structure")
	
	# Risk and compliance
	risk_level: str = Field(default="medium", description="Merchant risk level")
	kyc_status: str = Field(default="pending", description="KYC verification status")
	compliance_documents: list[str] = Field(default_factory=list, description="Compliance document IDs")
	
	# Account status
	status: MerchantStatus = Field(default=MerchantStatus.PENDING, description="Account status")
	activation_date: datetime | None = Field(None, description="Account activation date")
	suspension_reason: str | None = Field(None, description="Reason for suspension")
	
	# APG integration
	apg_capabilities: list[str] = Field(default_factory=list, description="Enabled APG capabilities")
	business_workflows: dict[str, Any] = Field(default_factory=dict, description="Configured workflows")
	
	# Metadata
	metadata: dict[str, Any] = Field(default_factory=dict, description="Additional merchant data")
	
	# Timestamps
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	last_activity_at: datetime | None = Field(None, description="Last merchant activity")
	
	# Audit
	created_by: str = Field(..., description="User who created merchant account")
	updated_by: str | None = Field(None, description="User who last updated account")
	
	def _log_merchant_created(self):
		"""Log merchant creation"""
		print(f"🏢 Merchant created: {self.id}")
		print(f"   Business: {self.business_name}")
		print(f"   Industry: {self.industry}")
		print(f"   Status: {self.status}")

class FraudAnalysis(BaseModel):
	"""
	AI-powered fraud analysis model for transaction risk assessment
	"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Fraud analysis ID")
	transaction_id: str = Field(..., description="Associated transaction ID")
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# Fraud scoring
	overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall fraud score (0-1)")
	risk_level: FraudRiskLevel = Field(..., description="Assessed risk level")
	confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence (0-1)")
	
	# Individual risk factors
	device_risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Device risk score")
	location_risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Location risk score")
	behavioral_risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Behavioral risk score")
	transaction_risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Transaction risk score")
	
	# Risk factors
	risk_factors: list[str] = Field(default_factory=list, description="Identified risk factors")
	anomalies_detected: list[str] = Field(default_factory=list, description="Detected anomalies")
	
	# Device information
	device_fingerprint: str | None = Field(None, description="Device fingerprint hash")
	ip_address: str | None = Field(None, description="Transaction IP address")
	geolocation: dict[str, Any] = Field(default_factory=dict, description="IP geolocation data")
	
	# Machine learning data
	model_version: str = Field(..., description="Fraud detection model version")
	feature_vector: dict[str, float] = Field(default_factory=dict, description="ML feature vector")
	model_explanation: dict[str, Any] = Field(default_factory=dict, description="Model decision explanation")
	
	# Actions taken
	actions_taken: list[str] = Field(default_factory=list, description="Automated actions taken")
	requires_review: bool = Field(default=False, description="Requires human review")
	review_assigned_to: str | None = Field(None, description="Assigned reviewer")
	
	# Resolution
	final_decision: str | None = Field(None, description="Final fraud decision")
	decision_reason: str | None = Field(None, description="Reason for decision")
	false_positive: bool | None = Field(None, description="Marked as false positive")
	
	# Timestamps
	analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	reviewed_at: datetime | None = Field(None, description="Human review timestamp")
	resolved_at: datetime | None = Field(None, description="Resolution timestamp")
	
	def _log_fraud_analysis_created(self):
		"""Log fraud analysis creation"""
		print(f"🛡️  Fraud analysis created: {self.id}")
		print(f"   Score: {self.overall_score:.3f}")
		print(f"   Risk Level: {self.risk_level}")
		print(f"   Confidence: {self.confidence:.3f}")

class PaymentProcessor(BaseModel):
	"""
	Payment processor configuration and status model
	"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Processor configuration ID")
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# Processor information
	name: str = Field(..., description="Processor name (stripe, adyen, etc.)")
	display_name: str = Field(..., description="Human-readable processor name")
	version: str = Field(..., description="Processor API version")
	
	# Configuration
	is_enabled: bool = Field(default=True, description="Processor enabled for use")
	is_primary: bool = Field(default=False, description="Primary processor for tenant")
	priority: int = Field(default=100, description="Processor priority (lower = higher priority)")
	
	# Capabilities
	supported_payment_methods: list[PaymentMethodType] = Field(default_factory=list, description="Supported payment methods")
	supported_currencies: list[str] = Field(default_factory=list, description="Supported currencies")
	supported_countries: list[str] = Field(default_factory=list, description="Supported countries")
	
	# Routing rules
	routing_rules: dict[str, Any] = Field(default_factory=dict, description="Intelligent routing configuration")
	fallback_processors: list[str] = Field(default_factory=list, description="Fallback processor chain")
	
	# Performance metrics
	success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Current success rate")
	average_response_time: float = Field(default=0.0, description="Average response time in ms")
	uptime_percentage: float = Field(default=100.0, ge=0.0, le=100.0, description="Uptime percentage")
	
	# Configuration data
	api_credentials: dict[str, str] = Field(default_factory=dict, description="Encrypted API credentials")
	webhook_config: dict[str, Any] = Field(default_factory=dict, description="Webhook configuration")
	
	# Health status
	health_status: str = Field(default="healthy", description="Current health status")
	last_health_check: datetime | None = Field(None, description="Last health check time")
	health_check_interval: int = Field(default=300, description="Health check interval in seconds")
	
	# Timestamps
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	last_used_at: datetime | None = Field(None, description="Last time processor was used")
	
	def _log_processor_configured(self):
		"""Log processor configuration"""
		print(f"⚙️  Payment processor configured: {self.name}")
		print(f"   Enabled: {self.is_enabled}")
		print(f"   Priority: {self.priority}")
		print(f"   Methods: {len(self.supported_payment_methods)}")

# SQLAlchemy table definitions for database schema
class PaymentTransactionTable(Base):
	__tablename__ = 'pg_payment_transactions'
	
	id = Column(UUID(as_uuid=False), primary_key=True, default=uuid7str)
	tenant_id = Column(UUID(as_uuid=False), nullable=False, index=True)
	merchant_id = Column(UUID(as_uuid=False), nullable=False, index=True)
	customer_id = Column(UUID(as_uuid=False), nullable=True, index=True)
	
	amount = Column(Integer, nullable=False)
	currency = Column(String(3), nullable=False, default='USD')
	description = Column(String(500), nullable=True)
	reference = Column(String(100), nullable=True)
	
	payment_method_id = Column(UUID(as_uuid=False), nullable=False)
	payment_method_type = Column(String(50), nullable=False)
	
	status = Column(String(50), nullable=False, default='pending', index=True)
	processor = Column(String(50), nullable=True)
	processor_transaction_id = Column(String(200), nullable=True)
	
	fraud_score = Column(Numeric(5, 4), nullable=True)
	fraud_risk_level = Column(String(20), nullable=True)
	
	processing_fee = Column(Integer, nullable=True)
	net_amount = Column(Integer, nullable=True)
	
	metadata = Column(JSON, nullable=True)
	customer_ip = Column(String(45), nullable=True)
	user_agent = Column(Text, nullable=True)
	
	business_context = Column(JSON, nullable=True)
	workflow_triggers = Column(JSON, nullable=True)
	
	created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
	updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
	processed_at = Column(DateTime(timezone=True), nullable=True)
	
	created_by = Column(UUID(as_uuid=False), nullable=False)
	updated_by = Column(UUID(as_uuid=False), nullable=True)
	
	# Performance indexes
	__table_args__ = (
		Index('idx_pg_transactions_tenant_status', 'tenant_id', 'status'),
		Index('idx_pg_transactions_merchant_date', 'merchant_id', 'created_at'),
		Index('idx_pg_transactions_customer_date', 'customer_id', 'created_at'),
		Index('idx_pg_transactions_processor_status', 'processor', 'status'),
	)

# Additional SQLAlchemy table definitions

class PaymentMethodTable(Base):
	__tablename__ = 'pg_payment_methods'
	
	id = Column(UUID(as_uuid=False), primary_key=True, default=uuid7str)
	tenant_id = Column(UUID(as_uuid=False), nullable=False, index=True)
	customer_id = Column(UUID(as_uuid=False), nullable=False, index=True)
	
	type = Column(String(50), nullable=False)
	provider = Column(String(50), nullable=False)
	token = Column(String(500), nullable=False)
	
	card_brand = Column(String(20), nullable=True)
	card_last4 = Column(String(4), nullable=True)
	card_exp_month = Column(Integer, nullable=True)
	card_exp_year = Column(Integer, nullable=True)
	card_country = Column(String(2), nullable=True)
	
	bank_name = Column(String(100), nullable=True)
	account_type = Column(String(50), nullable=True)
	routing_number = Column(String(50), nullable=True)
	account_last4 = Column(String(4), nullable=True)
	
	wallet_type = Column(String(50), nullable=True)
	wallet_email = Column(String(255), nullable=True)
	
	is_verified = Column(Boolean, default=False)
	verification_method = Column(String(50), nullable=True)
	
	nickname = Column(String(100), nullable=True)
	is_default = Column(Boolean, default=False)
	metadata = Column(JSON, nullable=True)
	
	created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
	updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
	last_used_at = Column(DateTime(timezone=True), nullable=True)
	
	__table_args__ = (
		Index('idx_pg_payment_methods_customer_type', 'customer_id', 'type'),
		Index('idx_pg_payment_methods_provider', 'provider'),
	)

class MerchantTable(Base):
	__tablename__ = 'pg_merchants'
	
	id = Column(UUID(as_uuid=False), primary_key=True, default=uuid7str)
	tenant_id = Column(UUID(as_uuid=False), nullable=False, index=True)
	
	business_name = Column(String(200), nullable=False)
	display_name = Column(String(100), nullable=False)
	business_type = Column(String(50), nullable=False)
	industry = Column(String(50), nullable=False)
	
	email = Column(String(255), nullable=False, index=True)
	phone = Column(String(50), nullable=True)
	website = Column(String(500), nullable=True)
	
	address_line1 = Column(String(200), nullable=False)
	address_line2 = Column(String(200), nullable=True)
	city = Column(String(100), nullable=False)
	state = Column(String(100), nullable=True)
	postal_code = Column(String(20), nullable=False)
	country = Column(String(2), nullable=False)
	
	supported_currencies = Column(JSON, nullable=True)
	processing_countries = Column(JSON, nullable=True)
	
	settlement_schedule = Column(String(20), default='daily')
	settlement_currency = Column(String(3), default='USD')
	processing_fees = Column(JSON, nullable=True)
	
	risk_level = Column(String(20), default='medium')
	kyc_status = Column(String(20), default='pending')
	compliance_documents = Column(JSON, nullable=True)
	
	status = Column(String(20), nullable=False, default='pending', index=True)
	activation_date = Column(DateTime(timezone=True), nullable=True)
	suspension_reason = Column(String(500), nullable=True)
	
	apg_capabilities = Column(JSON, nullable=True)
	business_workflows = Column(JSON, nullable=True)
	metadata = Column(JSON, nullable=True)
	
	created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
	updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
	last_activity_at = Column(DateTime(timezone=True), nullable=True)
	
	created_by = Column(UUID(as_uuid=False), nullable=False)
	updated_by = Column(UUID(as_uuid=False), nullable=True)
	
	__table_args__ = (
		Index('idx_pg_merchants_status_industry', 'status', 'industry'),
		Index('idx_pg_merchants_email_unique', 'email', unique=True),
	)

class FraudAnalysisTable(Base):
	__tablename__ = 'pg_fraud_analysis'
	
	id = Column(UUID(as_uuid=False), primary_key=True, default=uuid7str)
	transaction_id = Column(UUID(as_uuid=False), nullable=False, index=True)
	tenant_id = Column(UUID(as_uuid=False), nullable=False, index=True)
	
	overall_score = Column(Numeric(5, 4), nullable=False)
	risk_level = Column(String(20), nullable=False, index=True)
	confidence = Column(Numeric(5, 4), nullable=False)
	
	device_risk_score = Column(Numeric(5, 4), default=0.0)
	location_risk_score = Column(Numeric(5, 4), default=0.0)
	behavioral_risk_score = Column(Numeric(5, 4), default=0.0)
	transaction_risk_score = Column(Numeric(5, 4), default=0.0)
	
	risk_factors = Column(JSON, nullable=True)
	anomalies_detected = Column(JSON, nullable=True)
	
	device_fingerprint = Column(String(500), nullable=True, index=True)
	ip_address = Column(String(45), nullable=True)
	geolocation = Column(JSON, nullable=True)
	
	model_version = Column(String(20), nullable=False)
	feature_vector = Column(JSON, nullable=True)
	model_explanation = Column(JSON, nullable=True)
	
	actions_taken = Column(JSON, nullable=True)
	requires_review = Column(Boolean, default=False, index=True)
	review_assigned_to = Column(UUID(as_uuid=False), nullable=True)
	
	final_decision = Column(String(50), nullable=True)
	decision_reason = Column(String(500), nullable=True)
	false_positive = Column(Boolean, nullable=True)
	
	analyzed_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
	reviewed_at = Column(DateTime(timezone=True), nullable=True)
	resolved_at = Column(DateTime(timezone=True), nullable=True)
	
	__table_args__ = (
		Index('idx_pg_fraud_transaction_score', 'transaction_id', 'overall_score'),
		Index('idx_pg_fraud_risk_level_date', 'risk_level', 'analyzed_at'),
	)

class PaymentProcessorTable(Base):
	__tablename__ = 'pg_payment_processors'
	
	id = Column(UUID(as_uuid=False), primary_key=True, default=uuid7str)
	tenant_id = Column(UUID(as_uuid=False), nullable=False, index=True)
	
	name = Column(String(50), nullable=False, index=True)
	display_name = Column(String(100), nullable=False)
	version = Column(String(20), nullable=False)
	
	is_enabled = Column(Boolean, default=True, index=True)
	is_primary = Column(Boolean, default=False)
	priority = Column(Integer, default=100)
	
	supported_payment_methods = Column(JSON, nullable=True)
	supported_currencies = Column(JSON, nullable=True)
	supported_countries = Column(JSON, nullable=True)
	
	routing_rules = Column(JSON, nullable=True)
	fallback_processors = Column(JSON, nullable=True)
	
	success_rate = Column(Numeric(5, 4), default=0.0)
	average_response_time = Column(Numeric(8, 2), default=0.0)
	uptime_percentage = Column(Numeric(5, 2), default=100.0)
	
	api_credentials = Column(Text, nullable=True)  # Encrypted
	webhook_config = Column(JSON, nullable=True)
	
	health_status = Column(String(20), default='healthy', index=True)
	last_health_check = Column(DateTime(timezone=True), nullable=True)
	health_check_interval = Column(Integer, default=300)
	
	created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
	last_used_at = Column(DateTime(timezone=True), nullable=True)
	
	__table_args__ = (
		Index('idx_pg_processors_enabled_priority', 'is_enabled', 'priority'),
		Index('idx_pg_processors_health_status', 'health_status'),
	)

async def create_database_schema(engine):
	"""Create database schema for payment gateway tables"""
	print("🗄️  Creating payment gateway database schema...")
	
	try:
		# Create all tables
		Base.metadata.create_all(bind=engine)
		
		# Create indexes for performance
		with engine.connect() as conn:
			# Additional performance indexes
			conn.execute("""
				CREATE INDEX IF NOT EXISTS idx_pg_transactions_fraud_score 
				ON pg_payment_transactions(fraud_score) 
				WHERE fraud_score > 0.5;
			""")
			
			conn.execute("""
				CREATE INDEX IF NOT EXISTS idx_pg_transactions_amount_currency 
				ON pg_payment_transactions(amount, currency);
			""")
			
			conn.execute("""
				CREATE INDEX IF NOT EXISTS idx_pg_payment_methods_customer_type 
				ON pg_payment_methods(customer_id, type);
			""")
			
			conn.commit()
		
		print("✅ Payment gateway database schema created successfully")
		return True
		
	except Exception as e:
		print(f"❌ Failed to create database schema: {e}")
		raise

def _log_models_loaded():
	"""Log successful model loading"""
	print("📋 APG Payment Gateway models loaded successfully")
	print("   - PaymentTransaction: Core transaction processing")
	print("   - PaymentMethod: Tokenized payment methods")
	print("   - Merchant: Comprehensive merchant management")
	print("   - FraudAnalysis: AI-powered fraud detection")
	print("   - PaymentProcessor: Multi-processor orchestration")

# Execute model loading log
_log_models_loaded()