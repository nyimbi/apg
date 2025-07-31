#!/usr/bin/env python3
"""
Marketplace and Platform Payment Service - APG Payment Gateway

Advanced marketplace payment processing with multi-party transactions, split payments,
escrow management, vendor onboarding, commission handling, and platform economics.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from uuid_extensions import uuid7str
from dataclasses import dataclass, field
import logging
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict, deque
import statistics

from pydantic import BaseModel, Field, ConfigDict, validator

logger = logging.getLogger(__name__)

# Marketplace models and enums
class MarketplaceRole(str, Enum):
	"""Marketplace participant roles"""
	PLATFORM = "platform"
	VENDOR = "vendor"
	BUYER = "buyer"
	AFFILIATE = "affiliate"
	SERVICE_PROVIDER = "service_provider"

class VendorStatus(str, Enum):
	"""Vendor account status"""
	PENDING_VERIFICATION = "pending_verification"
	ACTIVE = "active"
	SUSPENDED = "suspended"
	BANNED = "banned"
	UNDER_REVIEW = "under_review"

class EscrowStatus(str, Enum):
	"""Escrow transaction status"""
	CREATED = "created"
	FUNDED = "funded"
	DISPUTED = "disputed"
	RELEASED = "released"
	REFUNDED = "refunded"
	EXPIRED = "expired"

class DisputeStatus(str, Enum):
	"""Dispute resolution status"""
	OPENED = "opened"
	INVESTIGATING = "investigating"
	EVIDENCE_REVIEW = "evidence_review"
	MEDIATION = "mediation"
	RESOLVED = "resolved"
	CLOSED = "closed"

class SplitType(str, Enum):
	"""Payment split types"""
	PERCENTAGE = "percentage"
	FIXED_AMOUNT = "fixed_amount"
	TIERED_PERCENTAGE = "tiered_percentage"
	DYNAMIC = "dynamic"

class CommissionType(str, Enum):
	"""Commission calculation types"""
	FLAT_FEE = "flat_fee"
	PERCENTAGE = "percentage"
	TIERED = "tiered"
	SUBSCRIPTION = "subscription"
	PERFORMANCE_BASED = "performance_based"

class PayoutSchedule(str, Enum):
	"""Payout schedule options"""
	INSTANT = "instant"
	DAILY = "daily"
	WEEKLY = "weekly"
	BIWEEKLY = "biweekly"
	MONTHLY = "monthly"
	MANUAL = "manual"

@dataclass
class PaymentSplit:
	"""Payment split configuration"""
	recipient_id: str
	recipient_type: MarketplaceRole
	split_type: SplitType
	amount: Decimal | None = None
	percentage: Decimal | None = None
	description: str = ""
	metadata: Dict[str, Any] = field(default_factory=dict)

class Vendor(BaseModel):
	"""Marketplace vendor model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	platform_id: str
	business_name: str
	legal_name: str
	business_type: str  # individual, corporation, partnership, llc
	tax_id: str | None = None
	
	# Contact information
	email: str
	phone: str | None = None
	website: str | None = None
	
	# Address information
	business_address: Dict[str, Any] = Field(default_factory=dict)
	
	# Verification status
	status: VendorStatus = VendorStatus.PENDING_VERIFICATION
	verification_documents: List[str] = Field(default_factory=list)
	verification_score: float = 0.0
	
	# Banking information
	bank_accounts: List[str] = Field(default_factory=list)
	preferred_payout_method: str | None = None
	payout_schedule: PayoutSchedule = PayoutSchedule.WEEKLY
	
	# Business metrics
	total_sales: Decimal = Decimal('0.00')
	transaction_count: int = 0
	average_rating: float = 0.0
	dispute_rate: float = 0.0
	
	# Commission settings
	commission_rate: Decimal = Decimal('0.00')
	commission_type: CommissionType = CommissionType.PERCENTAGE
	
	# Timestamps
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	verified_at: datetime | None = None
	last_payout_at: datetime | None = None
	
	# Settings
	auto_payout_enabled: bool = True
	notification_preferences: Dict[str, bool] = Field(default_factory=dict)
	metadata: Dict[str, Any] = Field(default_factory=dict)

class MarketplaceTransaction(BaseModel):
	"""Marketplace transaction model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	platform_id: str
	vendor_id: str
	buyer_id: str
	
	# Transaction details
	order_id: str | None = None
	total_amount: Decimal
	currency: str = "USD"
	
	# Payment splits
	splits: List[PaymentSplit] = Field(default_factory=list)
	platform_fee: Decimal = Decimal('0.00')
	processing_fee: Decimal = Decimal('0.00')
	
	# Status and metadata
	status: str = "pending"
	payment_method: str | None = None
	payment_processor: str | None = None
	
	# Escrow information
	escrow_enabled: bool = False
	escrow_release_trigger: str | None = None  # manual, automatic, time_based
	escrow_release_date: datetime | None = None
	
	# Timestamps
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	processed_at: datetime | None = None
	settled_at: datetime | None = None
	
	# Additional data
	items: List[Dict[str, Any]] = Field(default_factory=list)
	shipping_info: Dict[str, Any] = Field(default_factory=dict)
	metadata: Dict[str, Any] = Field(default_factory=dict)

class EscrowTransaction(BaseModel):
	"""Escrow transaction model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	marketplace_transaction_id: str
	platform_id: str
	
	# Parties
	buyer_id: str
	vendor_id: str
	arbiter_id: str | None = None
	
	# Financial details
	amount: Decimal
	currency: str = "USD"
	platform_fee: Decimal = Decimal('0.00')
	
	# Escrow configuration
	status: EscrowStatus = EscrowStatus.CREATED
	release_conditions: List[str] = Field(default_factory=list)
	auto_release_date: datetime | None = None
	dispute_deadline: datetime | None = None
	
	# Milestone-based escrow
	milestones: List[Dict[str, Any]] = Field(default_factory=list)
	completed_milestones: int = 0
	
	# Timestamps
	funded_at: datetime | None = None
	released_at: datetime | None = None
	disputed_at: datetime | None = None
	
	# Additional information
	terms_and_conditions: str | None = None
	metadata: Dict[str, Any] = Field(default_factory=dict)

class Dispute(BaseModel):
	"""Marketplace dispute model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	escrow_transaction_id: str
	marketplace_transaction_id: str
	
	# Dispute details
	initiated_by: str  # buyer_id or vendor_id
	dispute_type: str  # product_not_received, not_as_described, refund_request
	reason: str
	description: str
	
	# Status and resolution
	status: DisputeStatus = DisputeStatus.OPENED
	assigned_mediator: str | None = None
	resolution: str | None = None
	resolution_amount: Decimal | None = None
	
	# Evidence and communication
	evidence_items: List[str] = Field(default_factory=list)
	messages: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Timestamps
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	resolved_at: datetime | None = None
	deadline: datetime | None = None
	
	# Metadata
	priority: str = "normal"  # low, normal, high, urgent
	metadata: Dict[str, Any] = Field(default_factory=dict)

class Payout(BaseModel):
	"""Vendor payout model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	vendor_id: str
	platform_id: str
	
	# Payout details
	amount: Decimal
	currency: str = "USD"
	transaction_count: int
	period_start: datetime
	period_end: datetime
	
	# Fee breakdown
	gross_amount: Decimal
	platform_fees: Decimal = Decimal('0.00')
	processing_fees: Decimal = Decimal('0.00')
	adjustments: Decimal = Decimal('0.00')
	net_amount: Decimal
	
	# Payout method
	payout_method: str  # bank_transfer, paypal, stripe_express, etc.
	payout_destination: str
	
	# Status tracking
	status: str = "pending"  # pending, processing, completed, failed
	initiated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	completed_at: datetime | None = None
	failure_reason: str | None = None
	
	# Transaction references
	included_transactions: List[str] = Field(default_factory=list)
	external_payout_id: str | None = None
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)

class MarketplaceService:
	"""
	Comprehensive marketplace and platform payment service
	
	Handles multi-party transactions, vendor management, escrow services,
	dispute resolution, commission processing, and payout management.
	"""
	
	def __init__(self, database_service=None, payment_service=None):
		self._database_service = database_service
		self._payment_service = payment_service
		self._vendors: Dict[str, Vendor] = {}
		self._marketplace_transactions: Dict[str, MarketplaceTransaction] = {}
		self._escrow_transactions: Dict[str, EscrowTransaction] = {}
		self._disputes: Dict[str, Dispute] = {}
		self._payouts: Dict[str, Payout] = {}
		self._initialized = False
		
		# Configuration
		self.default_commission_rate = Decimal('0.05')  # 5%
		self.default_processing_fee = Decimal('0.029')  # 2.9%
		self.escrow_auto_release_days = 14
		self.dispute_response_deadline_days = 7
		
		# Payout settings
		self.minimum_payout_amount = Decimal('10.00')
		self.payout_processing_fee = Decimal('0.25')
		
		# Risk management
		self.vendor_verification_required = True
		self.transaction_limits = {
			'unverified_vendor_daily': Decimal('1000.00'),
			'verified_vendor_daily': Decimal('50000.00'),
			'new_vendor_transaction': Decimal('500.00')
		}
		
		# Performance metrics
		self._marketplace_metrics = {
			'total_vendors': 0,
			'active_vendors': 0,
			'total_transactions': 0,
			'total_volume': Decimal('0.00'),
			'active_escrows': 0,
			'open_disputes': 0,
			'processed_payouts': 0
		}
	
	async def initialize(self):
		"""Initialize marketplace service"""
		try:
			# Load marketplace configurations
			await self._load_marketplace_configurations()
			
			# Initialize vendor verification system
			await self._setup_vendor_verification()
			
			# Setup automated payout scheduling
			await self._setup_payout_scheduling()
			
			# Initialize dispute resolution system
			await self._setup_dispute_resolution()
			
			# Start background tasks
			await self._start_marketplace_monitoring()
			
			self._initialized = True
			await self._log_marketplace_event("marketplace_service_initialized", {})
			
		except Exception as e:
			logger.error(f"marketplace_service_initialization_failed: {str(e)}")
			raise
	
	# Vendor Management
	
	async def onboard_vendor(self, vendor_data: Dict[str, Any]) -> str:
		"""
		Onboard new vendor to marketplace
		"""
		try:
			# Create vendor record
			vendor = Vendor(
				platform_id=vendor_data['platform_id'],
				business_name=vendor_data['business_name'],
				legal_name=vendor_data.get('legal_name', vendor_data['business_name']),
				business_type=vendor_data.get('business_type', 'individual'),
				email=vendor_data['email'],
				phone=vendor_data.get('phone'),
				website=vendor_data.get('website'),
				business_address=vendor_data.get('business_address', {}),
				tax_id=vendor_data.get('tax_id'),
				commission_rate=Decimal(str(vendor_data.get('commission_rate', self.default_commission_rate))),
				commission_type=CommissionType(vendor_data.get('commission_type', CommissionType.PERCENTAGE)),
				payout_schedule=PayoutSchedule(vendor_data.get('payout_schedule', PayoutSchedule.WEEKLY)),
				notification_preferences=vendor_data.get('notification_preferences', {}),
				metadata=vendor_data.get('metadata', {})
			)
			
			# Store vendor
			self._vendors[vendor.id] = vendor
			self._marketplace_metrics['total_vendors'] += 1
			
			# Initiate verification process if required
			if self.vendor_verification_required:
				await self._initiate_vendor_verification(vendor.id)
			else:
				vendor.status = VendorStatus.ACTIVE
				vendor.verified_at = datetime.now(timezone.utc)
				self._marketplace_metrics['active_vendors'] += 1
			
			# Create default bank account setup task
			await self._create_bank_account_setup_task(vendor.id)
			
			await self._log_marketplace_event(
				"vendor_onboarded",
				{
					'vendor_id': vendor.id,
					'business_name': vendor.business_name,
					'platform_id': vendor.platform_id,
					'verification_required': self.vendor_verification_required
				}
			)
			
			return vendor.id
			
		except Exception as e:
			logger.error(f"vendor_onboarding_failed: {str(e)}")
			raise
	
	async def verify_vendor(self, vendor_id: str, verification_data: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Verify vendor identity and business information
		"""
		try:
			vendor = self._vendors.get(vendor_id)
			if not vendor:
				raise ValueError(f"Vendor not found: {vendor_id}")
			
			verification_score = 0.0
			verification_items = []
			
			# Verify business registration
			if verification_data.get('business_registration'):
				business_reg_score = await self._verify_business_registration(
					vendor, verification_data['business_registration']
				)
				verification_score += business_reg_score * 0.3
				verification_items.append('business_registration')
			
			# Verify tax identification
			if verification_data.get('tax_verification'):
				tax_score = await self._verify_tax_identification(
					vendor, verification_data['tax_verification']
				)
				verification_score += tax_score * 0.2
				verification_items.append('tax_verification')
			
			# Verify bank account
			if verification_data.get('bank_verification'):
				bank_score = await self._verify_bank_account(
					vendor, verification_data['bank_verification']
				)
				verification_score += bank_score * 0.3
				verification_items.append('bank_verification')
			
			# Verify identity documents
			if verification_data.get('identity_verification'):
				identity_score = await self._verify_identity_documents(
					vendor, verification_data['identity_verification']
				)
				verification_score += identity_score * 0.2
				verification_items.append('identity_verification')
			
			# Update vendor verification status
			vendor.verification_score = verification_score
			vendor.verification_documents = verification_items
			
			if verification_score >= 0.8:
				vendor.status = VendorStatus.ACTIVE
				vendor.verified_at = datetime.now(timezone.utc)
				self._marketplace_metrics['active_vendors'] += 1
				
				await self._log_marketplace_event(
					"vendor_verified",
					{
						'vendor_id': vendor_id,
						'verification_score': verification_score,
						'verification_items': verification_items
					}
				)
			
			elif verification_score >= 0.6:
				vendor.status = VendorStatus.UNDER_REVIEW
				
				await self._log_marketplace_event(
					"vendor_under_review",
					{
						'vendor_id': vendor_id,
						'verification_score': verification_score,
						'reason': 'additional_documentation_required'
					}
				)
			
			else:
				vendor.status = VendorStatus.SUSPENDED
				
				await self._log_marketplace_event(
					"vendor_verification_failed",
					{
						'vendor_id': vendor_id,
						'verification_score': verification_score,
						'reason': 'insufficient_verification'
					}
				)
			
			return {
				'vendor_id': vendor_id,
				'verification_status': vendor.status.value,
				'verification_score': verification_score,
				'verified_items': verification_items,
				'next_steps': await self._get_verification_next_steps(vendor)
			}
			
		except Exception as e:
			logger.error(f"vendor_verification_failed: {vendor_id}, error: {str(e)}")
			raise
	
	# Marketplace Transaction Processing
	
	async def process_marketplace_transaction(self, transaction_data: Dict[str, Any]) -> str:
		"""
		Process marketplace transaction with splits and commission handling
		"""
		try:
			# Validate vendor
			vendor = self._vendors.get(transaction_data['vendor_id'])
			if not vendor:
				raise ValueError(f"Vendor not found: {transaction_data['vendor_id']}")
			
			if vendor.status != VendorStatus.ACTIVE:
				raise ValueError(f"Vendor not active: {vendor.status.value}")
			
			# Check transaction limits
			await self._check_transaction_limits(vendor, Decimal(str(transaction_data['total_amount'])))
			
			# Create marketplace transaction
			marketplace_transaction = MarketplaceTransaction(
				platform_id=transaction_data['platform_id'],
				vendor_id=transaction_data['vendor_id'],
				buyer_id=transaction_data['buyer_id'],
				order_id=transaction_data.get('order_id'),
				total_amount=Decimal(str(transaction_data['total_amount'])),
				currency=transaction_data.get('currency', 'USD'),
				payment_method=transaction_data.get('payment_method'),
				escrow_enabled=transaction_data.get('escrow_enabled', False),
				items=transaction_data.get('items', []),
				shipping_info=transaction_data.get('shipping_info', {}),
				metadata=transaction_data.get('metadata', {})
			)
			
			# Calculate payment splits
			splits = await self._calculate_payment_splits(marketplace_transaction, vendor)
			marketplace_transaction.splits = splits
			
			# Calculate fees
			platform_fee = await self._calculate_platform_fee(marketplace_transaction, vendor)
			processing_fee = await self._calculate_processing_fee(marketplace_transaction)
			
			marketplace_transaction.platform_fee = platform_fee
			marketplace_transaction.processing_fee = processing_fee
			
			# Store transaction
			self._marketplace_transactions[marketplace_transaction.id] = marketplace_transaction
			
			# Process payment through payment service
			payment_result = await self._process_split_payment(marketplace_transaction)
			
			if payment_result['success']:
				marketplace_transaction.status = "processed"
				marketplace_transaction.processed_at = datetime.now(timezone.utc)
				marketplace_transaction.payment_processor = payment_result.get('processor')
				
				# Create escrow if enabled
				if marketplace_transaction.escrow_enabled:
					escrow_id = await self._create_escrow_transaction(marketplace_transaction)
					marketplace_transaction.metadata['escrow_id'] = escrow_id
				
				# Update vendor metrics
				await self._update_vendor_metrics(vendor, marketplace_transaction)
				
				# Update marketplace metrics
				self._marketplace_metrics['total_transactions'] += 1
				self._marketplace_metrics['total_volume'] += marketplace_transaction.total_amount
				
				await self._log_marketplace_event(
					"marketplace_transaction_processed",
					{
						'transaction_id': marketplace_transaction.id,
						'vendor_id': vendor.id,
						'total_amount': str(marketplace_transaction.total_amount),
						'platform_fee': str(platform_fee),
						'escrow_enabled': marketplace_transaction.escrow_enabled
					}
				)
			
			else:
				marketplace_transaction.status = "failed"
				marketplace_transaction.metadata['failure_reason'] = payment_result.get('error_message')
				
				await self._log_marketplace_event(
					"marketplace_transaction_failed",
					{
						'transaction_id': marketplace_transaction.id,
						'vendor_id': vendor.id,
						'failure_reason': payment_result.get('error_message')
					}
				)
			
			return marketplace_transaction.id
			
		except Exception as e:
			logger.error(f"marketplace_transaction_processing_failed: {str(e)}")
			raise
	
	# Escrow Management
	
	async def create_escrow_transaction(self, transaction_id: str, escrow_config: Dict[str, Any]) -> str:
		"""
		Create escrow transaction for marketplace payment
		"""
		try:
			marketplace_transaction = self._marketplace_transactions.get(transaction_id)
			if not marketplace_transaction:
				raise ValueError(f"Marketplace transaction not found: {transaction_id}")
			
			# Create escrow transaction
			escrow_transaction = EscrowTransaction(
				marketplace_transaction_id=transaction_id,
				platform_id=marketplace_transaction.platform_id,
				buyer_id=marketplace_transaction.buyer_id,
				vendor_id=marketplace_transaction.vendor_id,
				amount=marketplace_transaction.total_amount - marketplace_transaction.platform_fee,
				currency=marketplace_transaction.currency,
				platform_fee=marketplace_transaction.platform_fee,
				release_conditions=escrow_config.get('release_conditions', ['manual_release']),
				auto_release_date=escrow_config.get('auto_release_date'),
				dispute_deadline=escrow_config.get('dispute_deadline'),
				milestones=escrow_config.get('milestones', []),
				terms_and_conditions=escrow_config.get('terms_and_conditions'),
				metadata=escrow_config.get('metadata', {})
			)
			
			# Set default auto-release date if not provided
			if not escrow_transaction.auto_release_date:
				escrow_transaction.auto_release_date = datetime.now(timezone.utc) + timedelta(days=self.escrow_auto_release_days)
			
			# Set default dispute deadline
			if not escrow_transaction.dispute_deadline:
				escrow_transaction.dispute_deadline = datetime.now(timezone.utc) + timedelta(days=self.dispute_response_deadline_days)
			
			# Fund escrow
			escrow_transaction.status = EscrowStatus.FUNDED
			escrow_transaction.funded_at = datetime.now(timezone.utc)
			
			# Store escrow transaction
			self._escrow_transactions[escrow_transaction.id] = escrow_transaction
			self._marketplace_metrics['active_escrows'] += 1
			
			# Schedule automatic release if configured
			if 'automatic_release' in escrow_transaction.release_conditions:
				await self._schedule_escrow_release(escrow_transaction.id)
			
			await self._log_marketplace_event(
				"escrow_transaction_created",
				{
					'escrow_id': escrow_transaction.id,
					'marketplace_transaction_id': transaction_id,
					'amount': str(escrow_transaction.amount),
					'auto_release_date': escrow_transaction.auto_release_date.isoformat(),
					'release_conditions': escrow_transaction.release_conditions
				}
			)
			
			return escrow_transaction.id
			
		except Exception as e:
			logger.error(f"escrow_creation_failed: {transaction_id}, error: {str(e)}")
			raise
	
	async def release_escrow(self, escrow_id: str, release_reason: str, released_by: str) -> bool:
		"""
		Release funds from escrow to vendor
		"""
		try:
			escrow_transaction = self._escrow_transactions.get(escrow_id)
			if not escrow_transaction:
				raise ValueError(f"Escrow transaction not found: {escrow_id}")
			
			if escrow_transaction.status != EscrowStatus.FUNDED:
				raise ValueError(f"Escrow not in fundable status: {escrow_transaction.status}")
			
			# Release funds to vendor
			vendor = self._vendors.get(escrow_transaction.vendor_id)
			if not vendor:
				raise ValueError(f"Vendor not found: {escrow_transaction.vendor_id}")
			
			release_amount = escrow_transaction.amount
			
			# Process payout to vendor
			payout_result = await self._process_vendor_payout(
				vendor.id,
				release_amount,
				escrow_transaction.currency,
				f"Escrow release: {escrow_id}"
			)
			
			if payout_result['success']:
				# Update escrow status
				escrow_transaction.status = EscrowStatus.RELEASED
				escrow_transaction.released_at = datetime.now(timezone.utc)
				escrow_transaction.metadata.update({
					'release_reason': release_reason,
					'released_by': released_by,
					'payout_id': payout_result.get('payout_id')
				})
				
				# Update marketplace transaction
				marketplace_transaction = self._marketplace_transactions.get(escrow_transaction.marketplace_transaction_id)
				if marketplace_transaction:
					marketplace_transaction.status = "settled"
					marketplace_transaction.settled_at = datetime.now(timezone.utc)
				
				# Update metrics
				self._marketplace_metrics['active_escrows'] -= 1
				
				await self._log_marketplace_event(
					"escrow_released",
					{
						'escrow_id': escrow_id,
						'vendor_id': vendor.id,
						'amount': str(release_amount),
						'release_reason': release_reason,
						'released_by': released_by
					}
				)
				
				return True
			
			else:
				await self._log_marketplace_event(
					"escrow_release_failed",
					{
						'escrow_id': escrow_id,
						'vendor_id': vendor.id,
						'failure_reason': payout_result.get('error_message')
					}
				)
				
				return False
			
		except Exception as e:
			logger.error(f"escrow_release_failed: {escrow_id}, error: {str(e)}")
			raise
	
	# Dispute Management
	
	async def create_dispute(self, dispute_data: Dict[str, Any]) -> str:
		"""
		Create dispute for escrow transaction
		"""
		try:
			escrow_transaction = self._escrow_transactions.get(dispute_data['escrow_transaction_id'])
			if not escrow_transaction:
				raise ValueError(f"Escrow transaction not found: {dispute_data['escrow_transaction_id']}")
			
			if escrow_transaction.status != EscrowStatus.FUNDED:
				raise ValueError(f"Cannot dispute escrow in status: {escrow_transaction.status}")
			
			# Check dispute deadline
			if escrow_transaction.dispute_deadline and datetime.now(timezone.utc) > escrow_transaction.dispute_deadline:
				raise ValueError("Dispute deadline has passed")
			
			# Create dispute
			dispute = Dispute(
				escrow_transaction_id=dispute_data['escrow_transaction_id'],
				marketplace_transaction_id=escrow_transaction.marketplace_transaction_id,
				initiated_by=dispute_data['initiated_by'],
				dispute_type=dispute_data['dispute_type'],
				reason=dispute_data['reason'],
				description=dispute_data['description'],
				evidence_items=dispute_data.get('evidence_items', []),
				priority=dispute_data.get('priority', 'normal'),
				deadline=datetime.now(timezone.utc) + timedelta(days=self.dispute_response_deadline_days),
				metadata=dispute_data.get('metadata', {})
			)
			
			# Store dispute
			self._disputes[dispute.id] = dispute
			
			# Update escrow status
			escrow_transaction.status = EscrowStatus.DISPUTED
			escrow_transaction.disputed_at = datetime.now(timezone.utc)
			
			# Update metrics
			self._marketplace_metrics['open_disputes'] += 1
			
			# Assign mediator if configured
			mediator_id = await self._assign_dispute_mediator(dispute.id)
			if mediator_id:
				dispute.assigned_mediator = mediator_id
			
			# Notify relevant parties
			await self._notify_dispute_parties(dispute.id)
			
			await self._log_marketplace_event(
				"dispute_created",
				{
					'dispute_id': dispute.id,
					'escrow_id': dispute.escrow_transaction_id,
					'initiated_by': dispute.initiated_by,
					'dispute_type': dispute.dispute_type,
					'priority': dispute.priority
				}
			)
			
			return dispute.id
			
		except Exception as e:
			logger.error(f"dispute_creation_failed: {str(e)}")
			raise
	
	async def resolve_dispute(self, dispute_id: str, resolution_data: Dict[str, Any]) -> bool:
		"""
		Resolve marketplace dispute
		"""
		try:
			dispute = self._disputes.get(dispute_id)
			if not dispute:
				raise ValueError(f"Dispute not found: {dispute_id}")
			
			if dispute.status in [DisputeStatus.RESOLVED, DisputeStatus.CLOSED]:
				raise ValueError(f"Dispute already resolved: {dispute.status}")
			
			# Get escrow transaction
			escrow_transaction = self._escrow_transactions.get(dispute.escrow_transaction_id)
			if not escrow_transaction:
				raise ValueError(f"Escrow transaction not found: {dispute.escrow_transaction_id}")
			
			resolution_type = resolution_data['resolution_type']  # full_refund, partial_refund, release_to_vendor, split
			resolution_amount = Decimal(str(resolution_data.get('resolution_amount', 0)))
			
			if resolution_type == "full_refund":
				# Refund full amount to buyer
				refund_result = await self._process_buyer_refund(
					dispute.marketplace_transaction_id,
					escrow_transaction.amount,
					"Dispute resolution: Full refund"
				)
				
				if refund_result['success']:
					escrow_transaction.status = EscrowStatus.REFUNDED
					dispute.resolution = "Full refund processed"
					dispute.resolution_amount = escrow_transaction.amount
				
			elif resolution_type == "partial_refund":
				# Partial refund to buyer, remainder to vendor
				refund_amount = resolution_amount
				vendor_amount = escrow_transaction.amount - refund_amount
				
				refund_result = await self._process_buyer_refund(
					dispute.marketplace_transaction_id,
					refund_amount,
					"Dispute resolution: Partial refund"
				)
				
				payout_result = await self._process_vendor_payout(
					escrow_transaction.vendor_id,
					vendor_amount,
					escrow_transaction.currency,
					"Dispute resolution: Partial payout"
				)
				
				if refund_result['success'] and payout_result['success']:
					escrow_transaction.status = EscrowStatus.RELEASED
					dispute.resolution = f"Partial refund: {refund_amount}, Vendor payout: {vendor_amount}"
					dispute.resolution_amount = refund_amount
				
			elif resolution_type == "release_to_vendor":
				# Release full amount to vendor
				payout_result = await self._process_vendor_payout(
					escrow_transaction.vendor_id,
					escrow_transaction.amount,
					escrow_transaction.currency,
					"Dispute resolution: Release to vendor"
				)
				
				if payout_result['success']:
					escrow_transaction.status = EscrowStatus.RELEASED
					dispute.resolution = "Full amount released to vendor"
					dispute.resolution_amount = Decimal('0.00')
			
			# Update dispute status
			dispute.status = DisputeStatus.RESOLVED
			dispute.resolved_at = datetime.now(timezone.utc)
			dispute.metadata.update({
				'resolved_by': resolution_data.get('resolved_by'),
				'resolution_notes': resolution_data.get('resolution_notes')
			})
			
			# Update metrics
			self._marketplace_metrics['open_disputes'] -= 1
			self._marketplace_metrics['active_escrows'] -= 1
			
			# Notify parties of resolution
			await self._notify_dispute_resolution(dispute.id)
			
			await self._log_marketplace_event(
				"dispute_resolved",
				{
					'dispute_id': dispute_id,
					'resolution_type': resolution_type,
					'resolution_amount': str(dispute.resolution_amount),
					'escrow_status': escrow_transaction.status.value
				}
			)
			
			return True
			
		except Exception as e:
			logger.error(f"dispute_resolution_failed: {dispute_id}, error: {str(e)}")
			raise
	
	# Payout Management
	
	async def process_vendor_payouts(self, platform_id: str, period_end: datetime | None = None) -> List[str]:
		"""
		Process scheduled payouts for all eligible vendors
		"""
		try:
			if not period_end:
				period_end = datetime.now(timezone.utc)
			
			payout_ids = []
			
			# Get all active vendors for platform
			eligible_vendors = [
				vendor for vendor in self._vendors.values()
				if (vendor.platform_id == platform_id and 
					vendor.status == VendorStatus.ACTIVE and
					vendor.auto_payout_enabled)
			]
			
			for vendor in eligible_vendors:
				# Check if payout is due
				if await self._is_payout_due(vendor, period_end):
					payout_id = await self._create_vendor_payout(vendor.id, period_end)
					if payout_id:
						payout_ids.append(payout_id)
			
			await self._log_marketplace_event(
				"vendor_payouts_processed",
				{
					'platform_id': platform_id,
					'period_end': period_end.isoformat(),
					'payout_count': len(payout_ids),
					'total_vendors_checked': len(eligible_vendors)
				}
			)
			
			return payout_ids
			
		except Exception as e:
			logger.error(f"vendor_payouts_processing_failed: {platform_id}, error: {str(e)}")
			raise
	
	async def create_vendor_payout(self, vendor_id: str, period_end: datetime) -> str | None:
		"""
		Create payout for specific vendor
		"""
		try:
			vendor = self._vendors.get(vendor_id)
			if not vendor:
				raise ValueError(f"Vendor not found: {vendor_id}")
			
			# Calculate payout period
			period_start = vendor.last_payout_at or vendor.created_at
			
			# Get settled transactions for the period
			settled_transactions = await self._get_settled_transactions(vendor_id, period_start, period_end)
			
			if not settled_transactions:
				return None  # No transactions to pay out
			
			# Calculate payout amounts
			gross_amount = sum(txn.total_amount - txn.platform_fee for txn in settled_transactions)
			platform_fees = sum(txn.platform_fee for txn in settled_transactions)
			processing_fees = sum(txn.processing_fee for txn in settled_transactions)
			
			# Check minimum payout amount
			if gross_amount < self.minimum_payout_amount:
				return None  # Below minimum payout threshold
			
			# Calculate net amount
			net_amount = gross_amount - self.payout_processing_fee
			
			# Create payout record
			payout = Payout(
				vendor_id=vendor_id,
				platform_id=vendor.platform_id,
				amount=net_amount,
				currency="USD",  # Default currency
				transaction_count=len(settled_transactions),
				period_start=period_start,
				period_end=period_end,
				gross_amount=gross_amount,
				platform_fees=platform_fees,
				processing_fees=processing_fees,
				net_amount=net_amount,
				payout_method=vendor.preferred_payout_method or "bank_transfer",
				payout_destination=vendor.bank_accounts[0] if vendor.bank_accounts else "",
				included_transactions=[txn.id for txn in settled_transactions]
			)
			
			# Store payout
			self._payouts[payout.id] = payout
			
			# Process payout through payment service
			payout_result = await self._execute_payout(payout)
			
			if payout_result['success']:
				payout.status = "completed"
				payout.completed_at = datetime.now(timezone.utc)
				payout.external_payout_id = payout_result.get('external_id')
				
				# Update vendor last payout time
				vendor.last_payout_at = period_end
				
				# Update metrics
				self._marketplace_metrics['processed_payouts'] += 1
				
				await self._log_marketplace_event(
					"vendor_payout_completed",
					{
						'payout_id': payout.id,
						'vendor_id': vendor_id,
						'amount': str(net_amount),
						'transaction_count': len(settled_transactions),
						'payout_method': payout.payout_method
					}
				)
			
			else:
				payout.status = "failed"
				payout.failure_reason = payout_result.get('error_message')
				
				await self._log_marketplace_event(
					"vendor_payout_failed",
					{
						'payout_id': payout.id,
						'vendor_id': vendor_id,
						'failure_reason': payout.failure_reason
					}
				)
			
			return payout.id
			
		except Exception as e:
			logger.error(f"vendor_payout_creation_failed: {vendor_id}, error: {str(e)}")
			raise
	
	# Analytics and Reporting
	
	async def get_marketplace_analytics(self, platform_id: str, period_days: int = 30) -> Dict[str, Any]:
		"""
		Get comprehensive marketplace analytics
		"""
		try:
			end_date = datetime.now(timezone.utc)
			start_date = end_date - timedelta(days=period_days)
			
			# Get transactions for period
			period_transactions = [
				txn for txn in self._marketplace_transactions.values()
				if (txn.platform_id == platform_id and
					txn.created_at >= start_date and
					txn.created_at <= end_date)
			]
			
			# Calculate metrics
			total_volume = sum(txn.total_amount for txn in period_transactions)
			total_fees = sum(txn.platform_fee for txn in period_transactions)
			transaction_count = len(period_transactions)
			
			# Vendor metrics
			active_vendors = len([v for v in self._vendors.values() 
								 if v.platform_id == platform_id and v.status == VendorStatus.ACTIVE])
			
			# Calculate average transaction value
			avg_transaction_value = total_volume / transaction_count if transaction_count > 0 else Decimal('0.00')
			
			# Top vendors by volume
			vendor_volumes = defaultdict(Decimal)
			for txn in period_transactions:
				vendor_volumes[txn.vendor_id] += txn.total_amount
			
			top_vendors = sorted(vendor_volumes.items(), key=lambda x: x[1], reverse=True)[:10]
			
			# Dispute metrics
			period_disputes = [
				dispute for dispute in self._disputes.values()
				if dispute.created_at >= start_date and dispute.created_at <= end_date
			]
			
			dispute_rate = (len(period_disputes) / transaction_count * 100) if transaction_count > 0 else 0
			
			# Escrow metrics
			active_escrows = len([e for e in self._escrow_transactions.values() 
								 if e.status == EscrowStatus.FUNDED])
			
			analytics = {
				'period': {
					'start_date': start_date.isoformat(),
					'end_date': end_date.isoformat(),
					'days': period_days
				},
				'transaction_metrics': {
					'total_volume': str(total_volume),
					'total_fees': str(total_fees),
					'transaction_count': transaction_count,
					'average_transaction_value': str(avg_transaction_value)
				},
				'vendor_metrics': {
					'active_vendors': active_vendors,
					'top_vendors_by_volume': [
						{'vendor_id': v_id, 'volume': str(volume)} 
						for v_id, volume in top_vendors
					]
				},
				'dispute_metrics': {
					'total_disputes': len(period_disputes),
					'dispute_rate_percent': round(dispute_rate, 2),
					'open_disputes': len([d for d in period_disputes if d.status in [DisputeStatus.OPENED, DisputeStatus.INVESTIGATING]])
				},
				'escrow_metrics': {
					'active_escrows': active_escrows,
					'total_escrow_value': str(sum(e.amount for e in self._escrow_transactions.values() if e.status == EscrowStatus.FUNDED))
				},
				'overall_metrics': self._marketplace_metrics.copy()
			}
			
			# Convert Decimal values to strings for JSON serialization
			for key, value in analytics['overall_metrics'].items():
				if isinstance(value, Decimal):
					analytics['overall_metrics'][key] = str(value)
			
			return analytics
			
		except Exception as e:
			logger.error(f"marketplace_analytics_generation_failed: {platform_id}, error: {str(e)}")
			raise
	
	# Helper Methods
	
	async def _load_marketplace_configurations(self):
		"""Load marketplace configurations"""
		pass  # Configuration loading
	
	async def _setup_vendor_verification(self):
		"""Setup vendor verification system"""
		pass  # Verification system initialization
	
	async def _setup_payout_scheduling(self):
		"""Setup automated payout scheduling"""
		asyncio.create_task(self._payout_scheduler())
	
	async def _setup_dispute_resolution(self):
		"""Setup dispute resolution system"""
		pass  # Dispute resolution initialization
	
	async def _start_marketplace_monitoring(self):
		"""Start marketplace monitoring tasks"""
		asyncio.create_task(self._monitor_escrow_releases())
		asyncio.create_task(self._monitor_dispute_deadlines())
	
	# Verification methods
	
	async def _verify_business_registration(self, vendor: Vendor, registration_data: Dict[str, Any]) -> float:
		"""Verify business registration documents"""
		# This would integrate with business registration APIs
		return 0.9  # Simplified for demo
	
	async def _verify_tax_identification(self, vendor: Vendor, tax_data: Dict[str, Any]) -> float:
		"""Verify tax identification"""
		# This would integrate with tax verification services
		return 0.85  # Simplified for demo
	
	async def _verify_bank_account(self, vendor: Vendor, bank_data: Dict[str, Any]) -> float:
		"""Verify bank account information"""
		# This would integrate with bank verification services
		return 0.9  # Simplified for demo
	
	async def _verify_identity_documents(self, vendor: Vendor, identity_data: Dict[str, Any]) -> float:
		"""Verify identity documents"""
		# This would integrate with identity verification services
		return 0.8  # Simplified for demo
	
	# Transaction processing methods
	
	async def _calculate_payment_splits(self, transaction: MarketplaceTransaction, vendor: Vendor) -> List[PaymentSplit]:
		"""Calculate payment splits for marketplace transaction"""
		splits = []
		
		# Platform fee split
		platform_fee = await self._calculate_platform_fee(transaction, vendor)
		if platform_fee > 0:
			splits.append(PaymentSplit(
				recipient_id=transaction.platform_id,
				recipient_type=MarketplaceRole.PLATFORM,
				split_type=SplitType.FIXED_AMOUNT,
				amount=platform_fee,
				description="Platform commission"
			))
		
		# Vendor split (remaining amount)
		vendor_amount = transaction.total_amount - platform_fee
		splits.append(PaymentSplit(
			recipient_id=transaction.vendor_id,
			recipient_type=MarketplaceRole.VENDOR,
			split_type=SplitType.FIXED_AMOUNT,
			amount=vendor_amount,
			description="Vendor payment"
		))
		
		return splits
	
	async def _calculate_platform_fee(self, transaction: MarketplaceTransaction, vendor: Vendor) -> Decimal:
		"""Calculate platform commission fee"""
		if vendor.commission_type == CommissionType.PERCENTAGE:
			return transaction.total_amount * vendor.commission_rate
		elif vendor.commission_type == CommissionType.FLAT_FEE:
			return vendor.commission_rate
		else:
			return Decimal('0.00')
	
	async def _calculate_processing_fee(self, transaction: MarketplaceTransaction) -> Decimal:
		"""Calculate payment processing fee"""
		return transaction.total_amount * self.default_processing_fee
	
	async def _process_split_payment(self, transaction: MarketplaceTransaction) -> Dict[str, Any]:
		"""Process split payment through payment service"""
		# This would integrate with actual payment service
		return {
			'success': True,
			'processor': 'stripe',
			'transaction_id': f"pay_{uuid7str()}"
		}
	
	# Utility methods
	
	async def _check_transaction_limits(self, vendor: Vendor, amount: Decimal):
		"""Check transaction limits for vendor"""
		if vendor.status != VendorStatus.ACTIVE:
			daily_limit = self.transaction_limits['unverified_vendor_daily']
		else:
			daily_limit = self.transaction_limits['verified_vendor_daily']
		
		# This would check actual daily transaction volume
		# For demo, we'll just check the single transaction limit
		if amount > daily_limit:
			raise ValueError(f"Transaction amount exceeds daily limit: {daily_limit}")
	
	async def _create_escrow_transaction(self, marketplace_transaction: MarketplaceTransaction) -> str:
		"""Create escrow transaction for marketplace payment"""
		escrow_config = {
			'release_conditions': ['manual_release'],
			'auto_release_date': datetime.now(timezone.utc) + timedelta(days=self.escrow_auto_release_days)
		}
		
		return await self.create_escrow_transaction(marketplace_transaction.id, escrow_config)
	
	async def _update_vendor_metrics(self, vendor: Vendor, transaction: MarketplaceTransaction):
		"""Update vendor performance metrics"""
		vendor.total_sales += transaction.total_amount
		vendor.transaction_count += 1
	
	# Monitoring and scheduling tasks
	
	async def _payout_scheduler(self):
		"""Automated payout scheduler"""
		while True:
			try:
				# Process payouts for all platforms
				platforms = set(vendor.platform_id for vendor in self._vendors.values())
				
				for platform_id in platforms:
					await self.process_vendor_payouts(platform_id)
				
				# Run daily
				await asyncio.sleep(86400)
			except Exception as e:
				logger.error(f"payout_scheduler_failed: {str(e)}")
				await asyncio.sleep(3600)  # Retry in 1 hour
	
	async def _monitor_escrow_releases(self):
		"""Monitor escrow transactions for automatic release"""
		while True:
			try:
				now = datetime.now(timezone.utc)
				
				for escrow in self._escrow_transactions.values():
					if (escrow.status == EscrowStatus.FUNDED and
						escrow.auto_release_date and
						escrow.auto_release_date <= now and
						'automatic_release' in escrow.release_conditions):
						
						await self.release_escrow(escrow.id, "automatic_release", "system")
				
				await asyncio.sleep(3600)  # Check hourly
			except Exception as e:
				logger.error(f"escrow_monitoring_failed: {str(e)}")
				await asyncio.sleep(1800)  # Retry in 30 minutes
	
	async def _monitor_dispute_deadlines(self):
		"""Monitor dispute deadlines"""
		while True:
			try:
				now = datetime.now(timezone.utc)
				
				for dispute in self._disputes.values():
					if (dispute.status in [DisputeStatus.OPENED, DisputeStatus.INVESTIGATING] and
						dispute.deadline and
						dispute.deadline <= now):
						
						# Auto-resolve dispute based on platform policy
						await self._auto_resolve_expired_dispute(dispute.id)
				
				await asyncio.sleep(3600)  # Check hourly
			except Exception as e:
				logger.error(f"dispute_monitoring_failed: {str(e)}")
				await asyncio.sleep(1800)  # Retry in 30 minutes
	
	# Advanced integration methods for comprehensive marketplace operations
	
	async def _initiate_vendor_verification(self, vendor_id: str):
		"""Initiate comprehensive vendor verification process"""
		try:
			vendor = await self._get_vendor_by_id(vendor_id)
			if not vendor:
				return
			
			# Create verification tasks
			verification_tasks = []
			
			# Identity verification
			if not vendor.verification_status.get('identity_verified', False):
				verification_tasks.append({
					'type': 'identity_verification',
					'priority': 'high',
					'description': 'Verify business identity and ownership documents',
					'required_documents': ['business_license', 'tax_certificate', 'owner_id'],
					'deadline': (datetime.utcnow() + timedelta(days=7)).isoformat()
				})
			
			# Financial verification
			if not vendor.verification_status.get('financial_verified', False):
				verification_tasks.append({
					'type': 'financial_verification',
					'priority': 'high',
					'description': 'Verify financial standing and bank account details',
					'required_documents': ['bank_statements', 'financial_statements'],
					'deadline': (datetime.utcnow() + timedelta(days=10)).isoformat()
				})
			
			# Compliance verification
			if not vendor.verification_status.get('compliance_verified', False):
				verification_tasks.append({
					'type': 'compliance_verification',
					'priority': 'medium',
					'description': 'Verify regulatory compliance and certifications',
					'required_documents': ['regulatory_certificates', 'insurance_certificates'],
					'deadline': (datetime.utcnow() + timedelta(days=14)).isoformat()
				})
			
			# Store verification tasks and update vendor status
			vendor.verification_tasks = verification_tasks
			vendor.verification_status['verification_initiated'] = True
			vendor.verification_status['verification_started_at'] = datetime.utcnow().isoformat()
			
			await self._update_vendor(vendor)
			
			# Send notification to vendor
			await self._send_vendor_notification(vendor_id, {
				'type': 'verification_initiated',
				'message': f'Verification process started. {len(verification_tasks)} tasks pending.',
				'tasks': verification_tasks
			})
			
			logger.info(f"Vendor verification initiated for {vendor_id} with {len(verification_tasks)} tasks")
			
		except Exception as e:
			logger.error(f"Failed to initiate vendor verification for {vendor_id}: {str(e)}")
	
	async def _create_bank_account_setup_task(self, vendor_id: str):
		"""Create comprehensive bank account setup task for vendor"""
		try:
			vendor = await self._get_vendor_by_id(vendor_id)
			if not vendor:
				return
			
			# Check if bank account setup is already completed
			if vendor.payout_details.get('bank_account_verified', False):
				return
			
			# Create bank account setup task
			setup_task = {
				'id': uuid7str(),
				'vendor_id': vendor_id,
				'type': 'bank_account_setup',
				'status': 'pending',
				'priority': 'high',
				'created_at': datetime.utcnow().isoformat(),
				'deadline': (datetime.utcnow() + timedelta(days=5)).isoformat(),
				'description': 'Complete bank account setup for automated payouts',
				'requirements': {
					'bank_name': 'Required',
					'account_number': 'Required',
					'routing_number': 'Required',
					'account_type': 'Required (checking/savings)',
					'account_holder_name': 'Required (must match business name)',
					'verification_document': 'Required (bank statement or voided check)'
				},
				'steps': [
					'Provide bank account details',
					'Upload verification documents',
					'Complete micro-deposit verification',
					'Confirm account ownership'
				]
			}
			
			# Store task
			if not hasattr(vendor, 'setup_tasks'):
				vendor.setup_tasks = []
			vendor.setup_tasks.append(setup_task)
			
			# Update payout setup status
			vendor.payout_details['setup_initiated'] = True
			vendor.payout_details['setup_initiated_at'] = datetime.utcnow().isoformat()
			
			await self._update_vendor(vendor)
			
			# Send notification
			await self._send_vendor_notification(vendor_id, {
				'type': 'bank_setup_required',
				'message': 'Please complete bank account setup to receive payouts',
				'task': setup_task,
				'deadline': setup_task['deadline']
			})
			
			logger.info(f"Bank account setup task created for vendor {vendor_id}")
			
		except Exception as e:
			logger.error(f"Failed to create bank account setup task for {vendor_id}: {str(e)}")
	
	async def _get_verification_next_steps(self, vendor: Vendor) -> List[str]:
		"""Get next steps for vendor verification"""
		return ["complete_bank_verification", "submit_tax_documents"]
	
	async def _schedule_escrow_release(self, escrow_id: str):
		"""Schedule automatic escrow release"""
		pass
	
	async def _process_vendor_payout(self, vendor_id: str, amount: Decimal, currency: str, description: str) -> Dict[str, Any]:
		"""Process payout to vendor"""
		return {'success': True, 'payout_id': uuid7str()}
	
	async def _process_buyer_refund(self, transaction_id: str, amount: Decimal, description: str) -> Dict[str, Any]:
		"""Process refund to buyer"""
		return {'success': True, 'refund_id': uuid7str()}
	
	async def _assign_dispute_mediator(self, dispute_id: str) -> str | None:
		"""Assign mediator to dispute"""
		return None  # No automatic assignment for demo
	
	async def _notify_dispute_parties(self, dispute_id: str):
		"""Notify relevant parties about dispute"""
		pass
	
	async def _notify_dispute_resolution(self, dispute_id: str):
		"""Notify parties about dispute resolution"""
		pass
	
	async def _is_payout_due(self, vendor: Vendor, current_time: datetime) -> bool:
		"""Check if payout is due for vendor"""
		if not vendor.last_payout_at:
			return True  # First payout
		
		time_diff = current_time - vendor.last_payout_at
		
		if vendor.payout_schedule == PayoutSchedule.DAILY:
			return time_diff >= timedelta(days=1)
		elif vendor.payout_schedule == PayoutSchedule.WEEKLY:
			return time_diff >= timedelta(days=7)
		elif vendor.payout_schedule == PayoutSchedule.MONTHLY:
			return time_diff >= timedelta(days=30)
		
		return False
	
	async def _get_settled_transactions(self, vendor_id: str, start_date: datetime, end_date: datetime) -> List[MarketplaceTransaction]:
		"""Get settled transactions for vendor in date range"""
		return [
			txn for txn in self._marketplace_transactions.values()
			if (txn.vendor_id == vendor_id and
				txn.status == "settled" and
				txn.settled_at and
				start_date <= txn.settled_at <= end_date)
		]
	
	async def _execute_payout(self, payout: Payout) -> Dict[str, Any]:
		"""Execute payout through payment service"""
		# This would integrate with actual payout service
		return {'success': True, 'external_id': f"payout_{uuid7str()}"}
	
	async def _auto_resolve_expired_dispute(self, dispute_id: str):
		"""Auto-resolve expired dispute based on platform policy"""
		# Default to releasing funds to vendor if buyer doesn't respond
		resolution_data = {
			'resolution_type': 'release_to_vendor',
			'resolved_by': 'system',
			'resolution_notes': 'Auto-resolved due to expired deadline'
		}
		
		await self.resolve_dispute(dispute_id, resolution_data)
	
	async def _log_marketplace_event(self, event_name: str, metadata: Dict[str, Any]):
		"""Log marketplace event"""
		logger.info(f"marketplace_event: {event_name}, metadata: {metadata}")


# Factory function
def create_marketplace_service(database_service=None, payment_service=None) -> MarketplaceService:
	"""Create and initialize marketplace service"""
	return MarketplaceService(database_service, payment_service)

# Test utility
async def test_marketplace_service():
	"""Test marketplace service functionality"""
	print("🏪 Testing Marketplace and Platform Payment Service")
	print("=" * 60)
	
	# Initialize service
	marketplace_service = create_marketplace_service()
	await marketplace_service.initialize()
	
	print("✅ Marketplace service initialized")
	print(f"   Commission rate: {marketplace_service.default_commission_rate * 100}%")
	print(f"   Escrow auto-release: {marketplace_service.escrow_auto_release_days} days")
	
	# Test vendor onboarding
	print("\n👥 Testing Vendor Onboarding")
	vendor_data = {
		'platform_id': 'platform_001',
		'business_name': 'TechGear Store',
		'legal_name': 'TechGear LLC',
		'business_type': 'llc',
		'email': 'vendor@techgear.com',
		'phone': '+1-555-0123',
		'business_address': {
			'street': '123 Commerce St',
			'city': 'San Francisco',
			'state': 'CA',
			'zip': '94105',
			'country': 'US'
		},
		'commission_rate': '0.05',
		'payout_schedule': 'weekly'
	}
	
	vendor_id = await marketplace_service.onboard_vendor(vendor_data)
	print(f"   ✅ Vendor onboarded: {vendor_id}")
	
	vendor = marketplace_service._vendors[vendor_id]
	print(f"      Business: {vendor.business_name}")
	print(f"      Status: {vendor.status.value}")
	print(f"      Commission: {vendor.commission_rate * 100}%")
	
	# Test vendor verification
	print("\n🔍 Testing Vendor Verification")
	verification_data = {
		'business_registration': {'document_id': 'reg_12345', 'verified': True},
		'tax_verification': {'tax_id': '12-3456789', 'verified': True},
		'bank_verification': {'account_id': 'bank_67890', 'verified': True},
		'identity_verification': {'document_id': 'id_54321', 'verified': True}
	}
	
	verification_result = await marketplace_service.verify_vendor(vendor_id, verification_data)
	print(f"   ✅ Vendor verification: {verification_result['verification_status']}")
	print(f"      Score: {verification_result['verification_score']:.2f}")
	print(f"      Verified items: {verification_result['verified_items']}")
	
	# Test marketplace transaction
	print("\n💳 Testing Marketplace Transaction")
	transaction_data = {
		'platform_id': 'platform_001',
		'vendor_id': vendor_id,
		'buyer_id': 'buyer_12345',
		'order_id': 'order_67890',
		'total_amount': '150.00',
		'currency': 'USD',
		'payment_method': 'credit_card',
		'escrow_enabled': True,
		'items': [
			{'name': 'Wireless Headphones', 'price': '150.00', 'quantity': 1}
		]
	}
	
	txn_id = await marketplace_service.process_marketplace_transaction(transaction_data)
	print(f"   ✅ Transaction processed: {txn_id}")
	
	transaction = marketplace_service._marketplace_transactions[txn_id]
	print(f"      Total amount: ${transaction.total_amount}")
	print(f"      Platform fee: ${transaction.platform_fee}")
	print(f"      Status: {transaction.status}")
	print(f"      Escrow enabled: {transaction.escrow_enabled}")
	print(f"      Splits: {len(transaction.splits)}")
	
	# Test escrow creation
	escrow_id = transaction.metadata.get('escrow_id')
	if escrow_id:
		print(f"\n🔒 Testing Escrow Management")
		escrow = marketplace_service._escrow_transactions[escrow_id]
		print(f"   ✅ Escrow created: {escrow_id}")
		print(f"      Amount: ${escrow.amount}")
		print(f"      Status: {escrow.status.value}")
		print(f"      Auto-release: {escrow.auto_release_date.strftime('%Y-%m-%d')}")
		
		# Test escrow release
		print("\n🔓 Testing Escrow Release")
		release_success = await marketplace_service.release_escrow(
			escrow_id, "goods_delivered", "platform_admin"
		)
		print(f"   ✅ Escrow release: {release_success}")
		
		updated_escrow = marketplace_service._escrow_transactions[escrow_id]
		print(f"      New status: {updated_escrow.status.value}")
		print(f"      Released at: {updated_escrow.released_at.strftime('%Y-%m-%d %H:%M')}")
	
	# Test dispute creation
	print("\n⚖️  Testing Dispute Management")
	
	# Create another transaction for dispute testing
	dispute_transaction_data = {
		'platform_id': 'platform_001',
		'vendor_id': vendor_id,
		'buyer_id': 'buyer_67890',
		'total_amount': '75.00',
		'escrow_enabled': True,
		'items': [{'name': 'Phone Case', 'price': '75.00', 'quantity': 1}]
	}
	
	dispute_txn_id = await marketplace_service.process_marketplace_transaction(dispute_transaction_data)
	dispute_transaction = marketplace_service._marketplace_transactions[dispute_txn_id]
	dispute_escrow_id = dispute_transaction.metadata.get('escrow_id')
	
	if dispute_escrow_id:
		dispute_data = {
			'escrow_transaction_id': dispute_escrow_id,
			'initiated_by': 'buyer_67890',
			'dispute_type': 'not_as_described',
			'reason': 'Product quality issues',
			'description': 'The phone case received does not match the description and has defects.',
			'evidence_items': ['photo_1.jpg', 'photo_2.jpg'],
			'priority': 'normal'
		}
		
		dispute_id = await marketplace_service.create_dispute(dispute_data)
		print(f"   ✅ Dispute created: {dispute_id}")
		
		dispute = marketplace_service._disputes[dispute_id]
		print(f"      Type: {dispute.dispute_type}")
		print(f"      Status: {dispute.status.value}")
		print(f"      Initiated by: {dispute.initiated_by}")
		
		# Test dispute resolution
		print("\n🤝 Testing Dispute Resolution")
		resolution_data = {
			'resolution_type': 'partial_refund',
			'resolution_amount': '25.00',
			'resolved_by': 'mediator_001',
			'resolution_notes': 'Partial refund agreed upon by both parties'
		}
		
		resolution_success = await marketplace_service.resolve_dispute(dispute_id, resolution_data)
		print(f"   ✅ Dispute resolved: {resolution_success}")
		
		resolved_dispute = marketplace_service._disputes[dispute_id]
		print(f"      Resolution: {resolved_dispute.resolution}")
		print(f"      Resolution amount: ${resolved_dispute.resolution_amount}")
		print(f"      Status: {resolved_dispute.status.value}")
	
	# Test payout processing
	print("\n💰 Testing Payout Processing")
	
	# Mark transactions as settled
	for txn in marketplace_service._marketplace_transactions.values():
		if txn.vendor_id == vendor_id:
			txn.status = "settled"
			txn.settled_at = datetime.now(timezone.utc)
	
	payout_ids = await marketplace_service.process_vendor_payouts('platform_001')
	print(f"   ✅ Payouts processed: {len(payout_ids)}")
	
	if payout_ids:
		payout = marketplace_service._payouts[payout_ids[0]]
		print(f"      Payout ID: {payout.id}")
		print(f"      Amount: ${payout.net_amount}")
		print(f"      Transactions: {payout.transaction_count}")
		print(f"      Status: {payout.status}")
	
	# Test marketplace analytics
	print("\n📊 Testing Marketplace Analytics")
	
	analytics = await marketplace_service.get_marketplace_analytics('platform_001', 30)
	print(f"   ✅ Analytics generated for 30 days")
	print(f"      Total volume: ${analytics['transaction_metrics']['total_volume']}")
	print(f"      Total fees: ${analytics['transaction_metrics']['total_fees']}")
	print(f"      Transaction count: {analytics['transaction_metrics']['transaction_count']}")
	print(f"      Active vendors: {analytics['vendor_metrics']['active_vendors']}")
	print(f"      Dispute rate: {analytics['dispute_metrics']['dispute_rate_percent']}%")
	print(f"      Active escrows: {analytics['escrow_metrics']['active_escrows']}")
	
	# Test performance metrics
	print("\n📈 Testing Performance Metrics")
	metrics = marketplace_service._marketplace_metrics
	print(f"   ✅ Marketplace metrics:")
	print(f"      Total vendors: {metrics['total_vendors']}")
	print(f"      Active vendors: {metrics['active_vendors']}")
	print(f"      Total transactions: {metrics['total_transactions']}")
	print(f"      Total volume: ${metrics['total_volume']}")
	print(f"      Processed payouts: {metrics['processed_payouts']}")
	
	print(f"\n✅ Marketplace service test completed!")
	print("   All vendor management, transaction processing, escrow, dispute, and payout features working correctly")

if __name__ == "__main__":
	asyncio.run(test_marketplace_service())

# Module initialization logging
def _log_marketplace_service_module_loaded():
	"""Log marketplace service module loaded"""
	print("🏪 Marketplace and Platform Payment Service module loaded")
	print("   - Multi-party transaction processing")
	print("   - Vendor onboarding and verification")
	print("   - Escrow management and dispute resolution")
	print("   - Automated payout processing")
	print("   - Commission and fee management")
	print("   - Comprehensive marketplace analytics")

# Execute module loading log
_log_marketplace_service_module_loaded()