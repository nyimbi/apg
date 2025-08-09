"""
Advanced Merchant Operations Service
Comprehensive merchant management with analytics, settlement, and fee optimization.

Copyright (c) 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, validator
from uuid_extensions import uuid7str

logger = logging.getLogger(__name__)


class MerchantTier(str, Enum):
	ENTERPRISE = "enterprise"
	PREMIUM = "premium"
	STANDARD = "standard"
	STARTER = "starter"
	TRIAL = "trial"


class SettlementFrequency(str, Enum):
	INSTANT = "instant"
	DAILY = "daily"
	WEEKLY = "weekly"
	MONTHLY = "monthly"
	CUSTOM = "custom"


class FeeStructure(str, Enum):
	FLAT_RATE = "flat_rate"
	TIERED = "tiered"
	INTERCHANGE_PLUS = "interchange_plus"
	BLENDED = "blended"
	CUSTOM = "custom"


class MerchantStatus(str, Enum):
	ACTIVE = "active"
	SUSPENDED = "suspended"
	PENDING_REVIEW = "pending_review"
	ONBOARDING = "onboarding"
	CLOSED = "closed"


class SettlementStatus(str, Enum):
	PENDING = "pending"
	PROCESSING = "processing"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"


class KYCStatus(str, Enum):
	NOT_STARTED = "not_started"
	IN_PROGRESS = "in_progress"
	PENDING_REVIEW = "pending_review"
	APPROVED = "approved"
	REJECTED = "rejected"
	EXPIRED = "expired"


class MerchantProfile(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	business_name: str
	legal_name: str
	business_type: str
	industry: str
	tax_id: str
	registration_number: Optional[str] = None
	tier: MerchantTier = MerchantTier.STARTER
	status: MerchantStatus = MerchantStatus.ONBOARDING
	kyc_status: KYCStatus = KYCStatus.NOT_STARTED
	contact_info: Dict[str, Any] = Field(default_factory=dict)
	business_address: Dict[str, Any] = Field(default_factory=dict)
	banking_info: Dict[str, Any] = Field(default_factory=dict)
	risk_profile: Dict[str, Any] = Field(default_factory=dict)
	processing_limits: Dict[str, Any] = Field(default_factory=dict)
	fee_structure: FeeStructure = FeeStructure.BLENDED
	settlement_frequency: SettlementFrequency = SettlementFrequency.DAILY
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	onboarded_at: Optional[datetime] = None


class MerchantAnalytics(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	merchant_id: str
	period_start: datetime
	period_end: datetime
	total_transactions: int = 0
	successful_transactions: int = 0
	failed_transactions: int = 0
	total_volume: Decimal = Field(default=Decimal("0.00"))
	average_transaction_size: Decimal = Field(default=Decimal("0.00"))
	success_rate: float = 0.0
	chargeback_count: int = 0
	chargeback_rate: float = 0.0
	refund_count: int = 0
	refund_rate: float = 0.0
	total_fees: Decimal = Field(default=Decimal("0.00"))
	net_revenue: Decimal = Field(default=Decimal("0.00"))
	top_payment_methods: Dict[str, int] = Field(default_factory=dict)
	geographic_distribution: Dict[str, int] = Field(default_factory=dict)
	peak_hours: Dict[int, int] = Field(default_factory=dict)
	customer_retention_rate: float = 0.0
	new_customer_acquisition: int = 0


class SettlementBatch(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	merchant_id: str
	batch_date: datetime = Field(default_factory=datetime.utcnow)
	settlement_date: datetime
	transaction_ids: List[str] = Field(default_factory=list)
	gross_amount: Decimal = Field(default=Decimal("0.00"))
	fee_amount: Decimal = Field(default=Decimal("0.00"))
	chargeback_amount: Decimal = Field(default=Decimal("0.00"))
	refund_amount: Decimal = Field(default=Decimal("0.00"))
	net_amount: Decimal = Field(default=Decimal("0.00"))
	currency: str = "USD"
	status: SettlementStatus = SettlementStatus.PENDING
	bank_account: str
	reference_number: Optional[str] = None
	processing_errors: List[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	processed_at: Optional[datetime] = None


class FeeOptimization(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	merchant_id: str
	current_fee_structure: Dict[str, Any] = Field(default_factory=dict)
	optimized_fee_structure: Dict[str, Any] = Field(default_factory=dict)
	projected_savings: Decimal = Field(default=Decimal("0.00"))
	savings_percentage: float = 0.0
	analysis_period: Dict[str, datetime] = Field(default_factory=dict)
	recommendations: List[str] = Field(default_factory=list)
	implementation_date: Optional[datetime] = None
	created_at: datetime = Field(default_factory=datetime.utcnow)


class SplitPaymentConfig(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	merchant_id: str
	config_name: str
	split_rules: List[Dict[str, Any]] = Field(default_factory=list)
	default_config: bool = False
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class MerchantDashboard(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	merchant_id: str
	dashboard_config: Dict[str, Any] = Field(default_factory=dict)
	widgets: List[Dict[str, Any]] = Field(default_factory=list)
	alerts: List[Dict[str, Any]] = Field(default_factory=list)
	last_updated: datetime = Field(default_factory=datetime.utcnow)


class AdvancedMerchantOperationsService:
	"""Advanced merchant operations with comprehensive analytics and automation."""
	
	def __init__(self):
		self._merchant_profiles: Dict[str, MerchantProfile] = {}
		self._merchant_analytics: Dict[str, List[MerchantAnalytics]] = {}
		self._settlement_batches: Dict[str, SettlementBatch] = {}
		self._fee_optimizations: Dict[str, List[FeeOptimization]] = {}
		self._split_payment_configs: Dict[str, List[SplitPaymentConfig]] = {}
		self._merchant_dashboards: Dict[str, MerchantDashboard] = {}
		
		# Processing queues
		self._settlement_queue: asyncio.Queue = asyncio.Queue()
		self._kyc_queue: asyncio.Queue = asyncio.Queue()
		
		# Fee optimization engine
		self._fee_optimization_rules: Dict[str, Any] = {}
		self._processor_fee_data: Dict[str, Dict[str, Any]] = {}
		
		# Real-time metrics
		self._real_time_metrics: Dict[str, Dict[str, Any]] = {}
		
		# Initialize services
		asyncio.create_task(self._initialize_merchant_operations())
		asyncio.create_task(self._start_settlement_processor())
		asyncio.create_task(self._start_kyc_processor())
	
	async def _initialize_merchant_operations(self) -> None:
		"""Initialize merchant operations system."""
		# Initialize fee optimization rules
		self._fee_optimization_rules = {
			"volume_tiers": {
				"tier_1": {"min_volume": 0, "max_volume": 10000, "discount": 0.0},
				"tier_2": {"min_volume": 10000, "max_volume": 50000, "discount": 0.05},
				"tier_3": {"min_volume": 50000, "max_volume": 250000, "discount": 0.10},
				"tier_4": {"min_volume": 250000, "max_volume": float('inf'), "discount": 0.15}
			},
			"industry_rates": {
				"retail": {"base_rate": 2.9, "min_rate": 2.5},
				"ecommerce": {"base_rate": 2.9, "min_rate": 2.6},
				"saas": {"base_rate": 2.7, "min_rate": 2.4},
				"healthcare": {"base_rate": 3.1, "min_rate": 2.8},
				"non_profit": {"base_rate": 2.2, "min_rate": 1.9}
			},
			"risk_adjustments": {
				"low_risk": -0.1,
				"medium_risk": 0.0,
				"high_risk": 0.3
			}
		}
		
		# Initialize processor fee data (mock data)
		self._processor_fee_data = {
			"stripe": {
				"card_present": {"percentage": 2.7, "fixed": 0.05},
				"card_not_present": {"percentage": 2.9, "fixed": 0.30},
				"digital_wallet": {"percentage": 2.9, "fixed": 0.30}
			},
			"adyen": {
				"card_present": {"percentage": 2.6, "fixed": 0.10},
				"card_not_present": {"percentage": 2.8, "fixed": 0.20},
				"digital_wallet": {"percentage": 2.7, "fixed": 0.15}
			},
			"square": {
				"card_present": {"percentage": 2.6, "fixed": 0.10},
				"card_not_present": {"percentage": 2.9, "fixed": 0.30},
				"digital_wallet": {"percentage": 2.7, "fixed": 0.20}
			}
		}
		
		logger.info("Advanced merchant operations initialized")
	
	async def create_merchant_profile(
		self,
		merchant_data: Dict[str, Any],
		auto_onboard: bool = False
	) -> str:
		"""Create a new merchant profile."""
		profile = MerchantProfile(
			business_name=merchant_data["business_name"],
			legal_name=merchant_data.get("legal_name", merchant_data["business_name"]),
			business_type=merchant_data["business_type"],
			industry=merchant_data["industry"],
			tax_id=merchant_data["tax_id"],
			registration_number=merchant_data.get("registration_number"),
			contact_info=merchant_data.get("contact_info", {}),
			business_address=merchant_data.get("business_address", {}),
			banking_info=merchant_data.get("banking_info", {})
		)
		
		# Determine initial tier based on business data
		profile.tier = await self._determine_merchant_tier(merchant_data)
		
		# Set processing limits based on tier
		profile.processing_limits = await self._get_tier_processing_limits(profile.tier)
		
		# Initialize risk profile
		profile.risk_profile = await self._assess_initial_risk(merchant_data)
		
		self._merchant_profiles[profile.id] = profile
		
		# Initialize analytics
		self._merchant_analytics[profile.id] = []
		
		# Initialize dashboard
		await self._create_default_dashboard(profile.id)
		
		# Start KYC process if auto-onboarding
		if auto_onboard:
			await self._initiate_kyc_process(profile.id)
		
		logger.info(f"Created merchant profile for {profile.business_name} (ID: {profile.id})")
		return profile.id
	
	async def _determine_merchant_tier(self, merchant_data: Dict[str, Any]) -> MerchantTier:
		"""Determine appropriate merchant tier based on business data."""
		# Analyze business characteristics
		business_type = merchant_data.get("business_type", "").lower()
		industry = merchant_data.get("industry", "").lower()
		projected_volume = merchant_data.get("projected_monthly_volume", 0)
		business_age = merchant_data.get("years_in_business", 0)
		
		# Enterprise tier criteria
		if (projected_volume > 1000000 or 
			business_age >= 10 or 
			business_type in ["corporation", "enterprise"] or
			industry in ["healthcare", "financial_services"]):
			return MerchantTier.ENTERPRISE
		
		# Premium tier criteria
		elif (projected_volume > 100000 or 
			  business_age >= 3 or
			  business_type in ["llc", "partnership"]):
			return MerchantTier.PREMIUM
		
		# Standard tier criteria
		elif projected_volume > 10000 or business_age >= 1:
			return MerchantTier.STANDARD
		
		# Default to starter
		else:
			return MerchantTier.STARTER
	
	async def _get_tier_processing_limits(self, tier: MerchantTier) -> Dict[str, Any]:
		"""Get processing limits based on merchant tier."""
		limits = {
			MerchantTier.ENTERPRISE: {
				"daily_limit": Decimal("1000000.00"),
				"monthly_limit": Decimal("30000000.00"),
				"single_transaction_limit": Decimal("100000.00"),
				"chargeback_threshold": 0.01
			},
			MerchantTier.PREMIUM: {
				"daily_limit": Decimal("100000.00"),
				"monthly_limit": Decimal("3000000.00"),
				"single_transaction_limit": Decimal("25000.00"),
				"chargeback_threshold": 0.005
			},
			MerchantTier.STANDARD: {
				"daily_limit": Decimal("25000.00"),
				"monthly_limit": Decimal("750000.00"),
				"single_transaction_limit": Decimal("5000.00"),
				"chargeback_threshold": 0.002
			},
			MerchantTier.STARTER: {
				"daily_limit": Decimal("5000.00"),
				"monthly_limit": Decimal("150000.00"),
				"single_transaction_limit": Decimal("1000.00"),
				"chargeback_threshold": 0.001
			},
			MerchantTier.TRIAL: {
				"daily_limit": Decimal("1000.00"),
				"monthly_limit": Decimal("10000.00"),
				"single_transaction_limit": Decimal("500.00"),
				"chargeback_threshold": 0.0005
			}
		}
		
		return limits.get(tier, limits[MerchantTier.STARTER])
	
	async def _assess_initial_risk(self, merchant_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Assess initial risk profile for merchant."""
		risk_factors = {
			"industry_risk": "medium",
			"business_age_risk": "medium",
			"volume_risk": "low",
			"geographic_risk": "low",
			"overall_score": 0.5
		}
		
		industry = merchant_data.get("industry", "").lower()
		high_risk_industries = ["adult", "gambling", "cryptocurrency", "travel", "debt_collection"]
		
		if industry in high_risk_industries:
			risk_factors["industry_risk"] = "high"
			risk_factors["overall_score"] += 0.3
		
		years_in_business = merchant_data.get("years_in_business", 0)
		if years_in_business < 1:
			risk_factors["business_age_risk"] = "high"
			risk_factors["overall_score"] += 0.2
		elif years_in_business >= 5:
			risk_factors["business_age_risk"] = "low"
			risk_factors["overall_score"] -= 0.1
		
		# Normalize score
		risk_factors["overall_score"] = max(0.0, min(1.0, risk_factors["overall_score"]))
		
		return risk_factors
	
	async def _create_default_dashboard(self, merchant_id: str) -> None:
		"""Create default dashboard configuration for merchant."""
		default_widgets = [
			{
				"id": "revenue_overview",
				"type": "chart",
				"title": "Revenue Overview",
				"position": {"x": 0, "y": 0, "width": 6, "height": 4},
				"config": {"chart_type": "line", "time_period": "30d"}
			},
			{
				"id": "transaction_volume",
				"type": "metric",
				"title": "Transaction Volume",
				"position": {"x": 6, "y": 0, "width": 3, "height": 2},
				"config": {"metric": "total_transactions", "comparison": "previous_period"}
			},
			{
				"id": "success_rate",
				"type": "metric",
				"title": "Success Rate",
				"position": {"x": 9, "y": 0, "width": 3, "height": 2},
				"config": {"metric": "success_rate", "format": "percentage"}
			},
			{
				"id": "top_payment_methods",
				"type": "chart",
				"title": "Top Payment Methods",
				"position": {"x": 6, "y": 2, "width": 6, "height": 3},
				"config": {"chart_type": "pie"}
			},
			{
				"id": "recent_transactions",
				"type": "table",
				"title": "Recent Transactions",
				"position": {"x": 0, "y": 4, "width": 12, "height": 4},
				"config": {"limit": 10, "columns": ["date", "amount", "status", "customer"]}
			}
		]
		
		dashboard = MerchantDashboard(
			merchant_id=merchant_id,
			dashboard_config={
				"theme": "light",
				"auto_refresh": 300,  # 5 minutes
				"timezone": "UTC"
			},
			widgets=default_widgets
		)
		
		self._merchant_dashboards[merchant_id] = dashboard
	
	async def _initiate_kyc_process(self, merchant_id: str) -> None:
		"""Initiate KYC verification process."""
		await self._kyc_queue.put({
			"merchant_id": merchant_id,
			"action": "start_verification",
			"timestamp": datetime.utcnow()
		})
	
	async def _start_kyc_processor(self) -> None:
		"""Start KYC processing background task."""
		while True:
			try:
				kyc_task = await asyncio.wait_for(self._kyc_queue.get(), timeout=5.0)
				await self._process_kyc_task(kyc_task)
			except asyncio.TimeoutError:
				continue
			except Exception as e:
				logger.error(f"Error in KYC processor: {str(e)}")
				await asyncio.sleep(1)
	
	async def _process_kyc_task(self, task: Dict[str, Any]) -> None:
		"""Process KYC verification task."""
		merchant_id = task["merchant_id"]
		profile = self._merchant_profiles.get(merchant_id)
		
		if not profile:
			logger.error(f"Merchant profile not found: {merchant_id}")
			return
		
		try:
			if task["action"] == "start_verification":
				profile.kyc_status = KYCStatus.IN_PROGRESS
				
				# Simulate KYC verification process
				await asyncio.sleep(2)  # Simulate processing time
				
				# Mock verification result (90% success rate)
				import random
				if random.random() < 0.9:
					profile.kyc_status = KYCStatus.APPROVED
					profile.status = MerchantStatus.ACTIVE
					profile.onboarded_at = datetime.utcnow()
					
					logger.info(f"KYC approved for merchant {merchant_id}")
				else:
					profile.kyc_status = KYCStatus.PENDING_REVIEW
					logger.info(f"KYC requires manual review for merchant {merchant_id}")
			
			profile.updated_at = datetime.utcnow()
			
		except Exception as e:
			profile.kyc_status = KYCStatus.REJECTED
			logger.error(f"KYC verification failed for merchant {merchant_id}: {str(e)}")
	
	async def calculate_merchant_analytics(
		self,
		merchant_id: str,
		period_start: datetime,
		period_end: datetime,
		transaction_data: Optional[List[Dict[str, Any]]] = None
	) -> MerchantAnalytics:
		"""Calculate comprehensive merchant analytics."""
		# Mock transaction data if not provided
		if transaction_data is None:
			transaction_data = await self._generate_mock_transaction_data(merchant_id, period_start, period_end)
		
		analytics = MerchantAnalytics(
			merchant_id=merchant_id,
			period_start=period_start,
			period_end=period_end
		)
		
		if not transaction_data:
			return analytics
		
		# Calculate basic metrics
		analytics.total_transactions = len(transaction_data)
		analytics.successful_transactions = sum(1 for tx in transaction_data if tx.get("status") == "completed")
		analytics.failed_transactions = analytics.total_transactions - analytics.successful_transactions
		analytics.success_rate = (analytics.successful_transactions / analytics.total_transactions) if analytics.total_transactions > 0 else 0
		
		# Calculate volume metrics
		successful_amounts = [Decimal(str(tx.get("amount", 0))) for tx in transaction_data if tx.get("status") == "completed"]
		analytics.total_volume = sum(successful_amounts)
		analytics.average_transaction_size = analytics.total_volume / len(successful_amounts) if successful_amounts else Decimal("0")
		
		# Calculate chargebacks and refunds
		analytics.chargeback_count = sum(1 for tx in transaction_data if tx.get("chargeback"))
		analytics.chargeback_rate = (analytics.chargeback_count / analytics.successful_transactions) if analytics.successful_transactions > 0 else 0
		
		analytics.refund_count = sum(1 for tx in transaction_data if tx.get("refunded"))
		analytics.refund_rate = (analytics.refund_count / analytics.successful_transactions) if analytics.successful_transactions > 0 else 0
		
		# Calculate fees and revenue
		analytics.total_fees = await self._calculate_total_fees(merchant_id, successful_amounts)
		analytics.net_revenue = analytics.total_volume - analytics.total_fees
		
		# Analyze payment methods
		payment_methods = {}
		for tx in transaction_data:
			if tx.get("status") == "completed":
				method = tx.get("payment_method", "unknown")
				payment_methods[method] = payment_methods.get(method, 0) + 1
		analytics.top_payment_methods = dict(sorted(payment_methods.items(), key=lambda x: x[1], reverse=True)[:5])
		
		# Analyze geographic distribution
		countries = {}
		for tx in transaction_data:
			if tx.get("status") == "completed":
				country = tx.get("country", "unknown")
				countries[country] = countries.get(country, 0) + 1
		analytics.geographic_distribution = dict(sorted(countries.items(), key=lambda x: x[1], reverse=True)[:10])
		
		# Analyze peak hours
		hours = {}
		for tx in transaction_data:
			if tx.get("status") == "completed" and tx.get("timestamp"):
				hour = tx["timestamp"].hour
				hours[hour] = hours.get(hour, 0) + 1
		analytics.peak_hours = dict(sorted(hours.items(), key=lambda x: x[1], reverse=True)[:5])
		
		# Calculate customer metrics (mock)
		unique_customers = set(tx.get("customer_id") for tx in transaction_data if tx.get("customer_id"))
		analytics.new_customer_acquisition = len(unique_customers)
		analytics.customer_retention_rate = 0.75  # Mock retention rate
		
		# Store analytics
		if merchant_id not in self._merchant_analytics:
			self._merchant_analytics[merchant_id] = []
		self._merchant_analytics[merchant_id].append(analytics)
		
		return analytics
	
	async def _generate_mock_transaction_data(
		self,
		merchant_id: str,
		period_start: datetime,
		period_end: datetime
	) -> List[Dict[str, Any]]:
		"""Generate mock transaction data for testing."""
		import random
		
		# Generate random number of transactions
		num_transactions = random.randint(100, 1000)
		transactions = []
		
		payment_methods = ["card", "digital_wallet", "bank_transfer", "crypto"]
		countries = ["US", "GB", "CA", "AU", "DE", "FR", "JP"]
		
		for i in range(num_transactions):
			# Random timestamp within period
			time_diff = period_end - period_start
			random_time = period_start + timedelta(seconds=random.randint(0, int(time_diff.total_seconds())))
			
			transaction = {
				"id": uuid7str(),
				"merchant_id": merchant_id,
				"customer_id": f"cust_{random.randint(1000, 9999)}",
				"amount": round(random.uniform(10, 1000), 2),
				"currency": "USD",
				"payment_method": random.choice(payment_methods),
				"country": random.choice(countries),
				"status": "completed" if random.random() < 0.95 else "failed",
				"timestamp": random_time,
				"chargeback": random.random() < 0.005,
				"refunded": random.random() < 0.02
			}
			
			transactions.append(transaction)
		
		return transactions
	
	async def _calculate_total_fees(self, merchant_id: str, amounts: List[Decimal]) -> Decimal:
		"""Calculate total fees for transactions."""
		profile = self._merchant_profiles.get(merchant_id)
		if not profile:
			return Decimal("0")
		
		# Mock fee calculation based on tier
		fee_rates = {
			MerchantTier.ENTERPRISE: Decimal("0.025"),  # 2.5%
			MerchantTier.PREMIUM: Decimal("0.027"),    # 2.7%
			MerchantTier.STANDARD: Decimal("0.029"),   # 2.9%
			MerchantTier.STARTER: Decimal("0.031"),    # 3.1%
			MerchantTier.TRIAL: Decimal("0.035")       # 3.5%
		}
		
		rate = fee_rates.get(profile.tier, Decimal("0.029"))
		total_fees = sum(amount * rate for amount in amounts)
		
		return total_fees
	
	async def optimize_merchant_fees(self, merchant_id: str) -> FeeOptimization:
		"""Analyze and optimize merchant fee structure."""
		profile = self._merchant_profiles.get(merchant_id)
		if not profile:
			raise ValueError(f"Merchant profile not found: {merchant_id}")
		
		# Get recent transaction data for analysis
		end_date = datetime.utcnow()
		start_date = end_date - timedelta(days=90)  # 90-day analysis
		
		transaction_data = await self._generate_mock_transaction_data(merchant_id, start_date, end_date)
		current_analytics = await self.calculate_merchant_analytics(merchant_id, start_date, end_date, transaction_data)
		
		# Current fee structure
		current_fees = {
			"structure_type": profile.fee_structure.value,
			"current_rate": await self._get_current_fee_rate(profile),
			"monthly_fees": float(current_analytics.total_fees)
		}
		
		# Analyze potential optimizations
		optimizations = await self._analyze_fee_optimizations(profile, current_analytics)
		
		# Calculate projected savings
		optimized_rate = optimizations["optimized_rate"]
		current_rate = current_fees["current_rate"]
		
		monthly_volume = float(current_analytics.total_volume)
		current_monthly_fees = monthly_volume * current_rate
		optimized_monthly_fees = monthly_volume * optimized_rate
		
		projected_savings = Decimal(str(current_monthly_fees - optimized_monthly_fees))
		savings_percentage = ((current_rate - optimized_rate) / current_rate * 100) if current_rate > 0 else 0
		
		fee_optimization = FeeOptimization(
			merchant_id=merchant_id,
			current_fee_structure=current_fees,
			optimized_fee_structure=optimizations,
			projected_savings=projected_savings,
			savings_percentage=savings_percentage,
			analysis_period={"start": start_date, "end": end_date},
			recommendations=await self._generate_fee_recommendations(profile, optimizations)
		)
		
		# Store optimization
		if merchant_id not in self._fee_optimizations:
			self._fee_optimizations[merchant_id] = []
		self._fee_optimizations[merchant_id].append(fee_optimization)
		
		logger.info(f"Generated fee optimization for merchant {merchant_id}: {savings_percentage:.1f}% savings")
		return fee_optimization
	
	async def _get_current_fee_rate(self, profile: MerchantProfile) -> float:
		"""Get current fee rate for merchant."""
		base_rates = {
			MerchantTier.ENTERPRISE: 0.025,
			MerchantTier.PREMIUM: 0.027,
			MerchantTier.STANDARD: 0.029,
			MerchantTier.STARTER: 0.031,
			MerchantTier.TRIAL: 0.035
		}
		
		return base_rates.get(profile.tier, 0.029)
	
	async def _analyze_fee_optimizations(
		self,
		profile: MerchantProfile,
		analytics: MerchantAnalytics
	) -> Dict[str, Any]:
		"""Analyze potential fee optimizations."""
		current_rate = await self._get_current_fee_rate(profile)
		
		# Volume-based optimization
		monthly_volume = float(analytics.total_volume)
		volume_discount = 0.0
		
		for tier_name, tier_info in self._fee_optimization_rules["volume_tiers"].items():
			if tier_info["min_volume"] <= monthly_volume <= tier_info["max_volume"]:
				volume_discount = tier_info["discount"]
				break
		
		# Industry-based optimization
		industry_info = self._fee_optimization_rules["industry_rates"].get(
			profile.industry.lower(), 
			{"base_rate": 0.029, "min_rate": 0.025}
		)
		
		# Risk-based adjustment
		risk_level = profile.risk_profile.get("overall_score", 0.5)
		if risk_level < 0.3:
			risk_adjustment = self._fee_optimization_rules["risk_adjustments"]["low_risk"]
		elif risk_level > 0.7:
			risk_adjustment = self._fee_optimization_rules["risk_adjustments"]["high_risk"]
		else:
			risk_adjustment = self._fee_optimization_rules["risk_adjustments"]["medium_risk"]
		
		# Calculate optimized rate
		base_rate = industry_info["min_rate"] / 100
		optimized_rate = base_rate * (1 - volume_discount) + (risk_adjustment / 100)
		optimized_rate = max(optimized_rate, industry_info["min_rate"] / 100)
		
		return {
			"structure_type": "optimized_tiered",
			"optimized_rate": optimized_rate,
			"volume_discount": volume_discount,
			"industry_rate": industry_info["min_rate"] / 100,
			"risk_adjustment": risk_adjustment,
			"processor_comparison": await self._compare_processor_rates(analytics)
		}
	
	async def _compare_processor_rates(self, analytics: MerchantAnalytics) -> Dict[str, Any]:
		"""Compare rates across different processors."""
		total_volume = float(analytics.total_volume)
		processor_costs = {}
		
		for processor, rates in self._processor_fee_data.items():
			# Simulate rate calculation based on payment mix
			avg_cost = 0
			for method, method_count in analytics.top_payment_methods.items():
				method_percentage = method_count / analytics.total_transactions
				
				if method == "card":
					rate_info = rates.get("card_not_present", rates["card_present"])
				elif method == "digital_wallet":
					rate_info = rates.get("digital_wallet", rates["card_not_present"])
				else:
					rate_info = rates["card_not_present"]
				
				method_cost = (rate_info["percentage"] / 100) * total_volume * method_percentage
				avg_cost += method_cost
			
			processor_costs[processor] = {
				"estimated_monthly_cost": avg_cost,
				"effective_rate": (avg_cost / total_volume) if total_volume > 0 else 0
			}
		
		return processor_costs
	
	async def _generate_fee_recommendations(
		self,
		profile: MerchantProfile,
		optimizations: Dict[str, Any]
	) -> List[str]:
		"""Generate fee optimization recommendations."""
		recommendations = []
		
		current_rate = await self._get_current_fee_rate(profile)
		optimized_rate = optimizations["optimized_rate"]
		
		if optimized_rate < current_rate:
			savings_pct = ((current_rate - optimized_rate) / current_rate) * 100
			recommendations.append(f"Switch to optimized tiered pricing for {savings_pct:.1f}% savings")
		
		if optimizations["volume_discount"] > 0:
			discount_pct = optimizations["volume_discount"] * 100
			recommendations.append(f"Qualify for {discount_pct:.1f}% volume discount based on transaction volume")
		
		# Processor recommendations
		processor_comparison = optimizations.get("processor_comparison", {})
		if processor_comparison:
			cheapest_processor = min(processor_comparison.items(), key=lambda x: x[1]["effective_rate"])
			recommendations.append(f"Consider switching to {cheapest_processor[0]} for optimal rates")
		
		if profile.tier != MerchantTier.ENTERPRISE:
			recommendations.append("Upgrade to higher tier for better rates and features")
		
		if len(recommendations) == 0:
			recommendations.append("Current fee structure is already optimized")
		
		return recommendations
	
	async def create_settlement_batch(
		self,
		merchant_id: str,
		transaction_ids: List[str],
		settlement_date: Optional[datetime] = None
	) -> str:
		"""Create settlement batch for merchant."""
		profile = self._merchant_profiles.get(merchant_id)
		if not profile:
			raise ValueError(f"Merchant profile not found: {merchant_id}")
		
		if settlement_date is None:
			settlement_date = datetime.utcnow() + timedelta(days=1)  # Next business day
		
		# Mock transaction data for settlement calculation
		mock_transactions = []
		total_amount = Decimal("0")
		
		for tx_id in transaction_ids:
			# Generate mock transaction data
			import random
			amount = Decimal(str(round(random.uniform(10, 1000), 2)))
			mock_transactions.append({
				"id": tx_id,
				"amount": amount,
				"fee": amount * Decimal("0.029"),  # 2.9% fee
				"status": "completed"
			})
			total_amount += amount
		
		# Calculate settlement amounts
		gross_amount = total_amount
		fee_amount = sum(tx["fee"] for tx in mock_transactions)
		
		# Mock adjustments
		chargeback_amount = gross_amount * Decimal("0.001")  # 0.1% chargeback reserve
		refund_amount = Decimal("0")  # No refunds in this batch
		
		net_amount = gross_amount - fee_amount - chargeback_amount - refund_amount
		
		# Get banking information
		bank_account = profile.banking_info.get("account_number", "****1234")
		
		settlement_batch = SettlementBatch(
			merchant_id=merchant_id,
			settlement_date=settlement_date,
			transaction_ids=transaction_ids,
			gross_amount=gross_amount,
			fee_amount=fee_amount,
			chargeback_amount=chargeback_amount,
			refund_amount=refund_amount,
			net_amount=net_amount,
			bank_account=bank_account
		)
		
		self._settlement_batches[settlement_batch.id] = settlement_batch
		
		# Queue for processing
		await self._settlement_queue.put(settlement_batch.id)
		
		logger.info(f"Created settlement batch {settlement_batch.id} for merchant {merchant_id}: {net_amount}")
		return settlement_batch.id
	
	async def _start_settlement_processor(self) -> None:
		"""Start settlement processing background task."""
		while True:
			try:
				batch_id = await asyncio.wait_for(self._settlement_queue.get(), timeout=10.0)
				await self._process_settlement_batch(batch_id)
			except asyncio.TimeoutError:
				continue
			except Exception as e:
				logger.error(f"Error in settlement processor: {str(e)}")
				await asyncio.sleep(1)
	
	async def _process_settlement_batch(self, batch_id: str) -> None:
		"""Process settlement batch."""
		batch = self._settlement_batches.get(batch_id)
		if not batch:
			logger.error(f"Settlement batch not found: {batch_id}")
			return
		
		try:
			batch.status = SettlementStatus.PROCESSING
			
			# Simulate bank transfer processing
			await asyncio.sleep(2)  # Simulate processing time
			
			# Mock settlement success (95% success rate)
			import random
			if random.random() < 0.95:
				batch.status = SettlementStatus.COMPLETED
				batch.reference_number = f"SET-{random.randint(100000, 999999)}"
				batch.processed_at = datetime.utcnow()
				
				logger.info(f"Settlement batch {batch_id} completed: {batch.net_amount}")
			else:
				batch.status = SettlementStatus.FAILED
				batch.processing_errors.append("Bank transfer failed - insufficient account details")
				
				logger.error(f"Settlement batch {batch_id} failed")
		
		except Exception as e:
			batch.status = SettlementStatus.FAILED
			batch.processing_errors.append(str(e))
			logger.error(f"Settlement batch {batch_id} processing error: {str(e)}")
	
	async def setup_split_payment_config(
		self,
		merchant_id: str,
		config_name: str,
		split_rules: List[Dict[str, Any]]
	) -> str:
		"""Set up split payment configuration for marketplace merchants."""
		config = SplitPaymentConfig(
			merchant_id=merchant_id,
			config_name=config_name,
			split_rules=split_rules
		)
		
		if merchant_id not in self._split_payment_configs:
			self._split_payment_configs[merchant_id] = []
		
		self._split_payment_configs[merchant_id].append(config)
		
		logger.info(f"Created split payment config '{config_name}' for merchant {merchant_id}")
		return config.id
	
	async def process_split_payment(
		self,
		merchant_id: str,
		transaction_amount: Decimal,
		config_id: Optional[str] = None
	) -> Dict[str, Any]:
		"""Process payment with split configuration."""
		# Find split config
		configs = self._split_payment_configs.get(merchant_id, [])
		
		if config_id:
			config = next((c for c in configs if c.id == config_id), None)
		else:
			config = next((c for c in configs if c.default_config), None)
		
		if not config:
			raise ValueError("No split payment configuration found")
		
		splits = []
		remaining_amount = transaction_amount
		
		for rule in config.split_rules:
			split_type = rule.get("type", "percentage")
			recipient = rule.get("recipient")
			
			if split_type == "percentage":
				percentage = Decimal(str(rule.get("percentage", 0))) / 100
				split_amount = transaction_amount * percentage
			elif split_type == "fixed":
				split_amount = Decimal(str(rule.get("amount", 0)))
			else:
				continue
			
			split_amount = min(split_amount, remaining_amount)
			
			splits.append({
				"recipient": recipient,
				"amount": float(split_amount),
				"type": split_type,
				"description": rule.get("description", "")
			})
			
			remaining_amount -= split_amount
		
		# Remainder goes to primary merchant
		if remaining_amount > 0:
			splits.append({
				"recipient": merchant_id,
				"amount": float(remaining_amount),
				"type": "remainder",
				"description": "Primary merchant share"
			})
		
		return {
			"config_id": config.id,
			"config_name": config.config_name,
			"total_amount": float(transaction_amount),
			"splits": splits,
			"split_count": len(splits)
		}
	
	async def get_merchant_dashboard_data(self, merchant_id: str) -> Dict[str, Any]:
		"""Get comprehensive dashboard data for merchant."""
		profile = self._merchant_profiles.get(merchant_id)
		if not profile:
			raise ValueError(f"Merchant profile not found: {merchant_id}")
		
		# Get recent analytics
		recent_analytics = None
		if merchant_id in self._merchant_analytics and self._merchant_analytics[merchant_id]:
			recent_analytics = self._merchant_analytics[merchant_id][-1]
		
		# Mock real-time metrics
		real_time_data = {
			"today_volume": float(Decimal("12547.83")),
			"today_transactions": 47,
			"success_rate_24h": 97.2,
			"avg_response_time": 245,  # milliseconds
			"active_disputes": 2,
			"pending_settlements": 1
		}
		
		# Get dashboard configuration
		dashboard = self._merchant_dashboards.get(merchant_id)
		
		# Get recent settlement batches
		recent_settlements = [
			batch for batch in self._settlement_batches.values()
			if batch.merchant_id == merchant_id
		][-5:]  # Last 5 settlements
		
		# Get fee optimization if available
		latest_fee_optimization = None
		if merchant_id in self._fee_optimizations and self._fee_optimizations[merchant_id]:
			latest_fee_optimization = self._fee_optimizations[merchant_id][-1]
		
		return {
			"merchant_profile": profile.dict(),
			"real_time_metrics": real_time_data,
			"recent_analytics": recent_analytics.dict() if recent_analytics else None,
			"dashboard_config": dashboard.dict() if dashboard else None,
			"recent_settlements": [batch.dict() for batch in recent_settlements],
			"fee_optimization": latest_fee_optimization.dict() if latest_fee_optimization else None,
			"alerts": await self._generate_merchant_alerts(merchant_id),
			"last_updated": datetime.utcnow().isoformat()
		}
	
	async def _generate_merchant_alerts(self, merchant_id: str) -> List[Dict[str, Any]]:
		"""Generate alerts for merchant."""
		alerts = []
		
		profile = self._merchant_profiles.get(merchant_id)
		if not profile:
			return alerts
		
		# KYC status alerts
		if profile.kyc_status == KYCStatus.PENDING_REVIEW:
			alerts.append({
				"type": "warning",
				"title": "KYC Review Required",
				"message": "Your account requires manual KYC review. Processing may be limited.",
				"action_required": True
			})
		
		# Settlement alerts
		pending_settlements = sum(1 for batch in self._settlement_batches.values() 
								 if batch.merchant_id == merchant_id and batch.status == SettlementStatus.PENDING)
		
		if pending_settlements > 0:
			alerts.append({
				"type": "info",
				"title": "Pending Settlements",
				"message": f"You have {pending_settlements} settlement batch(es) pending processing.",
				"action_required": False
			})
		
		# Fee optimization alert
		if merchant_id in self._fee_optimizations and self._fee_optimizations[merchant_id]:
			latest_optimization = self._fee_optimizations[merchant_id][-1]
			if latest_optimization.savings_percentage > 5:
				alerts.append({
					"type": "success",
					"title": "Fee Savings Available",
					"message": f"You could save {latest_optimization.savings_percentage:.1f}% on processing fees.",
					"action_required": True
				})
		
		return alerts
	
	async def get_merchant_operations_analytics(
		self,
		time_range: Optional[Dict[str, datetime]] = None
	) -> Dict[str, Any]:
		"""Get comprehensive merchant operations analytics."""
		start_time = (time_range or {}).get('start', datetime.utcnow() - timedelta(days=30))
		end_time = (time_range or {}).get('end', datetime.utcnow())
		
		# Merchant distribution by tier
		tier_distribution = {}
		for profile in self._merchant_profiles.values():
			tier = profile.tier.value
			tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
		
		# KYC status distribution
		kyc_distribution = {}
		for profile in self._merchant_profiles.values():
			status = profile.kyc_status.value
			kyc_distribution[status] = kyc_distribution.get(status, 0) + 1
		
		# Settlement analytics
		total_settlements = len(self._settlement_batches)
		completed_settlements = sum(1 for batch in self._settlement_batches.values() 
								   if batch.status == SettlementStatus.COMPLETED)
		
		total_settlement_volume = sum(batch.net_amount for batch in self._settlement_batches.values() 
									 if batch.status == SettlementStatus.COMPLETED)
		
		# Fee optimization analytics
		total_optimizations = sum(len(opts) for opts in self._fee_optimizations.values())
		total_projected_savings = sum(
			opt.projected_savings 
			for opts in self._fee_optimizations.values() 
			for opt in opts
		)
		
		# Merchant growth metrics
		new_merchants_period = sum(
			1 for profile in self._merchant_profiles.values()
			if start_time <= profile.created_at <= end_time
		)
		
		return {
			"time_range": {
				"start": start_time.isoformat(),
				"end": end_time.isoformat()
			},
			"merchant_analytics": {
				"total_merchants": len(self._merchant_profiles),
				"tier_distribution": tier_distribution,
				"kyc_distribution": kyc_distribution,
				"new_merchants_period": new_merchants_period,
				"active_merchants": sum(1 for p in self._merchant_profiles.values() if p.status == MerchantStatus.ACTIVE)
			},
			"settlement_analytics": {
				"total_settlement_batches": total_settlements,
				"completed_settlements": completed_settlements,
				"settlement_success_rate": (completed_settlements / total_settlements * 100) if total_settlements > 0 else 0,
				"total_settlement_volume": float(total_settlement_volume),
				"average_settlement_size": float(total_settlement_volume / completed_settlements) if completed_settlements > 0 else 0
			},
			"fee_optimization_analytics": {
				"total_optimizations_generated": total_optimizations,
				"total_projected_savings": float(total_projected_savings),
				"merchants_with_optimizations": len(self._fee_optimizations),
				"average_savings_percentage": sum(
					opt.savings_percentage 
					for opts in self._fee_optimizations.values() 
					for opt in opts
				) / total_optimizations if total_optimizations > 0 else 0
			},
			"operational_metrics": {
				"split_payment_configs": sum(len(configs) for configs in self._split_payment_configs.values()),
				"active_dashboards": len(self._merchant_dashboards),
				"merchant_onboarding_rate": (sum(1 for p in self._merchant_profiles.values() if p.onboarded_at) / len(self._merchant_profiles) * 100) if self._merchant_profiles else 0
			}
		}