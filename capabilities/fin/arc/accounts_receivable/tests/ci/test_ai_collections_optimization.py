"""
APG Accounts Receivable - AI Collections Optimization Tests
Unit tests for AI-powered collections strategy optimization

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List

from uuid_extensions import uuid7str

from ..ai_collections_optimization import (
	CollectionChannelType, CollectionStrategyType, CollectionOutcome,
	CustomerCollectionProfile, CollectionStrategyRecommendation,
	CollectionCampaignPlan, CollectionsOptimizationConfig,
	APGCollectionsAIService, create_collections_ai_service
)
from ..models import (
	ARCustomer, ARInvoice, ARPayment, ARCollectionActivity,
	ARCustomerStatus, ARCustomerType, ARInvoiceStatus, ARPaymentStatus,
	ARCollectionPriority
)


class TestCollectionEnums:
	"""Test collection-related enums and constants."""
	
	def test_collection_channel_types(self):
		"""Test collection channel type enum values."""
		assert CollectionChannelType.EMAIL == "email"
		assert CollectionChannelType.PHONE == "phone"
		assert CollectionChannelType.SMS == "sms"
		assert CollectionChannelType.LETTER == "letter"
		assert CollectionChannelType.IN_PERSON == "in_person"
		assert CollectionChannelType.AUTOMATED_SYSTEM == "automated_system"
		assert CollectionChannelType.LEGAL_NOTICE == "legal_notice"
	
	def test_collection_strategy_types(self):
		"""Test collection strategy type enum values."""
		assert CollectionStrategyType.SOFT_APPROACH == "soft_approach"
		assert CollectionStrategyType.STANDARD_DUNNING == "standard_dunning"
		assert CollectionStrategyType.AGGRESSIVE_COLLECTION == "aggressive_collection"
		assert CollectionStrategyType.LEGAL_ACTION == "legal_action"
		assert CollectionStrategyType.SETTLEMENT_NEGOTIATION == "settlement_negotiation"
		assert CollectionStrategyType.PAYMENT_PLAN == "payment_plan"
		assert CollectionStrategyType.WRITE_OFF == "write_off"
	
	def test_collection_outcomes(self):
		"""Test collection outcome enum values."""
		assert CollectionOutcome.FULL_PAYMENT == "full_payment"
		assert CollectionOutcome.PARTIAL_PAYMENT == "partial_payment"
		assert CollectionOutcome.PAYMENT_PROMISE == "payment_promise"
		assert CollectionOutcome.DISPUTE_RAISED == "dispute_raised"
		assert CollectionOutcome.NO_RESPONSE == "no_response"
		assert CollectionOutcome.CONTACT_UNAVAILABLE == "contact_unavailable"
		assert CollectionOutcome.REFUSED_TO_PAY == "refused_to_pay"
		assert CollectionOutcome.BANKRUPTCY_DECLARED == "bankruptcy_declared"


class TestCustomerCollectionProfile:
	"""Test customer collection profile model."""
	
	def test_profile_creation_valid_data(self):
		"""Test creating collection profile with valid data."""
		profile = CustomerCollectionProfile(
			customer_id=uuid7str(),
			tenant_id=uuid7str(),
			customer_type="corporation",
			total_outstanding=Decimal('15000.00'),
			overdue_amount=Decimal('8000.00'),
			days_overdue=45,
			total_invoices=20,
			paid_invoices=18,
			late_payments=2,
			payment_consistency_score=0.85,
			previous_collection_attempts=3,
			successful_collections=2,
			communication_responsiveness=0.7
		)
		
		assert profile.customer_type == "corporation"
		assert profile.total_outstanding == Decimal('15000.00')
		assert profile.overdue_amount == Decimal('8000.00')
		assert profile.days_overdue == 45
		assert profile.payment_consistency_score == 0.85
		assert profile.communication_responsiveness == 0.7
	
	def test_profile_validation_paid_invoices(self):
		"""Test validation that paid invoices cannot exceed total invoices."""
		with pytest.raises(ValueError, match="Paid invoices cannot exceed total invoices"):
			CustomerCollectionProfile(
				customer_id=uuid7str(),
				tenant_id=uuid7str(),
				customer_type="individual",
				total_outstanding=Decimal('5000.00'),
				overdue_amount=Decimal('2000.00'),
				days_overdue=30,
				total_invoices=10,
				paid_invoices=15  # Invalid: more than total
			)
	
	def test_profile_defaults(self):
		"""Test default values for optional fields."""
		profile = CustomerCollectionProfile(
			customer_id=uuid7str(),
			tenant_id=uuid7str(),
			customer_type="individual",
			total_outstanding=Decimal('1000.00'),
			overdue_amount=Decimal('500.00'),
			days_overdue=15
		)
		
		assert profile.total_invoices == 0
		assert profile.paid_invoices == 0
		assert profile.late_payments == 0
		assert profile.payment_consistency_score == 0.5
		assert profile.previous_collection_attempts == 0
		assert profile.successful_collections == 0
		assert profile.communication_responsiveness == 0.5
		assert profile.dispute_frequency == 0


class TestCollectionStrategyRecommendation:
	"""Test collection strategy recommendation model."""
	
	def test_recommendation_creation(self):
		"""Test creating strategy recommendation with valid data."""
		recommendation = CollectionStrategyRecommendation(
			customer_id=uuid7str(),
			strategy_type=CollectionStrategyType.STANDARD_DUNNING,
			recommended_channels=[CollectionChannelType.EMAIL, CollectionChannelType.PHONE],
			channel_sequence=[
				{'channel': 'email', 'delay_days': 0, 'attempts': 1},
				{'channel': 'phone', 'delay_days': 3, 'attempts': 2}
			],
			optimal_contact_times=['09:00', '14:00'],
			frequency_days=7,
			max_attempts=3,
			message_tone='professional',
			personalization_level='standard',
			urgency_level='medium',
			success_probability=0.75,
			expected_collection_amount=Decimal('7500.00'),
			estimated_collection_days=21,
			model_version='1.5.0',
			confidence_score=0.85,
			escalation_risk=0.25,
			customer_relationship_impact='moderate_impact',
			valid_until=date.today() + timedelta(days=30)
		)
		
		assert recommendation.strategy_type == CollectionStrategyType.STANDARD_DUNNING
		assert len(recommendation.recommended_channels) == 2
		assert CollectionChannelType.EMAIL in recommendation.recommended_channels
		assert CollectionChannelType.PHONE in recommendation.recommended_channels
		assert recommendation.success_probability == 0.75
		assert recommendation.expected_collection_amount == Decimal('7500.00')
		assert recommendation.confidence_score == 0.85
	
	def test_recommendation_validation_ranges(self):
		"""Test validation of numeric ranges in recommendation."""
		
		# Test frequency_days range
		with pytest.raises(ValueError):
			CollectionStrategyRecommendation(
				customer_id=uuid7str(),
				strategy_type=CollectionStrategyType.STANDARD_DUNNING,
				recommended_channels=[CollectionChannelType.EMAIL],
				channel_sequence=[],
				optimal_contact_times=['09:00'],
				frequency_days=0,  # Invalid: below minimum
				max_attempts=3,
				message_tone='professional',
				personalization_level='standard',
				urgency_level='medium',
				success_probability=0.75,
				expected_collection_amount=Decimal('1000.00'),
				estimated_collection_days=21,
				model_version='1.0.0',
				confidence_score=0.85,
				escalation_risk=0.25,
				customer_relationship_impact='moderate',
				valid_until=date.today() + timedelta(days=30)
			)
		
		# Test max_attempts range
		with pytest.raises(ValueError):
			CollectionStrategyRecommendation(
				customer_id=uuid7str(),
				strategy_type=CollectionStrategyType.STANDARD_DUNNING,
				recommended_channels=[CollectionChannelType.EMAIL],
				channel_sequence=[],
				optimal_contact_times=['09:00'],
				frequency_days=7,
				max_attempts=15,  # Invalid: above maximum
				message_tone='professional',
				personalization_level='standard',
				urgency_level='medium',
				success_probability=0.75,
				expected_collection_amount=Decimal('1000.00'),
				estimated_collection_days=21,
				model_version='1.0.0',
				confidence_score=0.85,
				escalation_risk=0.25,
				customer_relationship_impact='moderate',
				valid_until=date.today() + timedelta(days=30)
			)


class TestCollectionCampaignPlan:
	"""Test collection campaign plan model."""
	
	def test_campaign_plan_creation(self):
		"""Test creating campaign plan with valid data."""
		campaign = CollectionCampaignPlan(
			tenant_id=uuid7str(),
			campaign_name="Q1 2025 Collections",
			target_customers=[uuid7str() for _ in range(5)],
			campaign_duration_days=30,
			start_date=date.today(),
			end_date=date.today() + timedelta(days=30),
			strategy_distribution={
				CollectionStrategyType.SOFT_APPROACH: 2,
				CollectionStrategyType.STANDARD_DUNNING: 3
			},
			channel_allocation={
				CollectionChannelType.EMAIL: 5,
				CollectionChannelType.PHONE: 3
			},
			predicted_collection_rate=0.72,
			predicted_collection_amount=Decimal('125000.00'),
			estimated_cost=Decimal('5000.00'),
			roi_estimate=25.0,
			staff_hours_required=40.0,
			automated_activities_count=3,
			manual_activities_count=2,
			optimization_model_version='1.5.0',
			optimization_confidence=0.88
		)
		
		assert campaign.campaign_name == "Q1 2025 Collections"
		assert len(campaign.target_customers) == 5
		assert campaign.campaign_duration_days == 30
		assert campaign.predicted_collection_rate == 0.72
		assert campaign.predicted_collection_amount == Decimal('125000.00')
		assert campaign.roi_estimate == 25.0


class TestCollectionsOptimizationConfig:
	"""Test collections optimization configuration."""
	
	def test_config_creation_defaults(self):
		"""Test creating config with default values."""
		config = CollectionsOptimizationConfig(
			ai_orchestration_endpoint="https://ai.apg.company.com/v1"
		)
		
		assert config.ai_orchestration_endpoint == "https://ai.apg.company.com/v1"
		assert config.collections_model_name == "ar_collections_optimizer_v1"
		assert config.model_version == "1.5.0"
		assert config.success_threshold == 0.70
		assert config.confidence_threshold == 0.80
		assert config.max_collection_attempts == 5
		
		# Test default channel costs
		assert CollectionChannelType.EMAIL in config.channel_costs
		assert CollectionChannelType.PHONE in config.channel_costs
		assert config.channel_costs[CollectionChannelType.EMAIL] == Decimal('0.50')
		assert config.channel_costs[CollectionChannelType.PHONE] == Decimal('5.00')
		
		# Test default success rates
		assert CollectionChannelType.EMAIL in config.channel_success_rates
		assert config.channel_success_rates[CollectionChannelType.PHONE] == 0.65
	
	def test_config_custom_values(self):
		"""Test creating config with custom values."""
		custom_channel_costs = {
			CollectionChannelType.EMAIL: Decimal('1.00'),
			CollectionChannelType.SMS: Decimal('0.75')
		}
		
		config = CollectionsOptimizationConfig(
			ai_orchestration_endpoint="https://test.ai.com/v1",
			collections_model_name="test_model",
			model_version="2.0.0",
			success_threshold=0.80,
			channel_costs=custom_channel_costs
		)
		
		assert config.ai_orchestration_endpoint == "https://test.ai.com/v1"
		assert config.collections_model_name == "test_model"
		assert config.model_version == "2.0.0"
		assert config.success_threshold == 0.80
		assert config.channel_costs[CollectionChannelType.EMAIL] == Decimal('1.00')


class TestAPGCollectionsAIService:
	"""Test APG collections AI service functionality."""
	
	@pytest.fixture
	def sample_customer(self):
		"""Create sample customer for testing."""
		return ARCustomer(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_code="COLL001",
			legal_name="Collection Test Corp",
			customer_type=ARCustomerType.CORPORATION,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal('25000.00'),
			payment_terms_days=30,
			total_outstanding=Decimal('12000.00'),
			overdue_amount=Decimal('6000.00'),
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
	
	@pytest.fixture
	def overdue_invoices(self, sample_customer):
		"""Create sample overdue invoices for testing."""
		invoices = []
		for i in range(3):
			invoice = ARInvoice(
				id=uuid7str(),
				tenant_id=sample_customer.tenant_id,
				customer_id=sample_customer.id,
				invoice_number=f"OVD-{i+1:03d}",
				invoice_date=date.today() - timedelta(days=60+i*10),
				due_date=date.today() - timedelta(days=30+i*10),
				total_amount=Decimal('2000.00'),
				paid_amount=Decimal('0.00'),
				currency_code="USD",
				status=ARInvoiceStatus.OVERDUE,
				payment_status="overdue",
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
			invoices.append(invoice)
		return invoices
	
	@pytest.fixture
	def collection_history(self, sample_customer):
		"""Create sample collection history for testing."""
		activities = []
		for i in range(5):
			activity = ARCollectionActivity(
				id=uuid7str(),
				tenant_id=sample_customer.tenant_id,
				customer_id=sample_customer.id,
				activity_type="phone_call",
				activity_date=date.today() - timedelta(days=10+i*5),
				priority=ARCollectionPriority.NORMAL,
				contact_method="phone",
				notes=f"Collection attempt {i+1}",
				outcome="successful" if i < 3 else "no_response",
				status="completed",
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
			activities.append(activity)
		return activities
	
	@pytest.fixture
	def collections_config(self):
		"""Create collections optimization configuration for testing."""
		return CollectionsOptimizationConfig(
			ai_orchestration_endpoint="https://test.ai.com/v1",
			collections_model_name="test_collections_model",
			model_version="1.0.0",
			success_threshold=0.60,
			confidence_threshold=0.70
		)
	
	@pytest.fixture
	def collections_service(self, collections_config):
		"""Create collections AI service for testing."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		return APGCollectionsAIService(tenant_id, user_id, collections_config)
	
	async def test_extract_customer_collection_profile(self, collections_service, sample_customer,
													  overdue_invoices, collection_history):
		"""Test extracting customer profile for collections optimization."""
		
		profile = await collections_service._extract_customer_collection_profile(
			sample_customer, overdue_invoices, [], collection_history
		)
		
		assert isinstance(profile, CustomerCollectionProfile)
		assert profile.customer_id == sample_customer.id
		assert profile.tenant_id == sample_customer.tenant_id
		assert profile.customer_type == "corporation"
		assert profile.total_outstanding == Decimal('6000.00')  # Sum of overdue invoices
		assert profile.overdue_amount == Decimal('6000.00')
		assert profile.days_overdue == 50  # Max days overdue from sample data
		assert profile.total_invoices == 3
		assert profile.paid_invoices == 0  # All overdue in sample
		assert profile.late_payments == 3  # All overdue
		assert profile.previous_collection_attempts == 5
		assert profile.successful_collections == 3  # First 3 were successful
		assert profile.communication_responsiveness == 0.6  # 3/5 success rate
	
	async def test_optimize_collection_strategy(self, collections_service, sample_customer,
											   overdue_invoices, collection_history):
		"""Test AI-powered collection strategy optimization."""
		
		recommendation = await collections_service.optimize_collection_strategy(
			sample_customer, overdue_invoices, [], collection_history
		)
		
		assert isinstance(recommendation, CollectionStrategyRecommendation)
		assert recommendation.customer_id == sample_customer.id
		assert isinstance(recommendation.strategy_type, CollectionStrategyType)
		assert len(recommendation.recommended_channels) > 0
		assert 0 <= recommendation.success_probability <= 1
		assert recommendation.expected_collection_amount >= 0
		assert recommendation.estimated_collection_days > 0
		assert 0 <= recommendation.confidence_score <= 1
		assert 0 <= recommendation.escalation_risk <= 1
		assert len(recommendation.specific_actions) > 0
		assert recommendation.valid_until > date.today()
	
	async def test_optimize_strategy_different_customer_types(self, collections_service):
		"""Test strategy optimization for different customer profiles."""
		
		# High-risk customer
		high_risk_customer = ARCustomer(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_code="HIGHRISK001",
			legal_name="High Risk Corp",
			customer_type=ARCustomerType.CORPORATION,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal('10000.00'),
			payment_terms_days=30,
			total_outstanding=Decimal('15000.00'),
			overdue_amount=Decimal('12000.00'),
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
		
		# Create severely overdue invoices
		overdue_invoices = []
		for i in range(2):
			invoice = ARInvoice(
				id=uuid7str(),
				tenant_id=high_risk_customer.tenant_id,
				customer_id=high_risk_customer.id,
				invoice_number=f"SEVERE-{i+1:03d}",
				invoice_date=date.today() - timedelta(days=120+i*30),
				due_date=date.today() - timedelta(days=90+i*30),
				total_amount=Decimal('6000.00'),
				paid_amount=Decimal('0.00'),
				currency_code="USD",
				status=ARInvoiceStatus.OVERDUE,
				payment_status="overdue",
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
			overdue_invoices.append(invoice)
		
		recommendation = await collections_service.optimize_collection_strategy(
			high_risk_customer, overdue_invoices, [], []
		)
		
		# Should recommend aggressive strategy for severely overdue customer
		assert recommendation.strategy_type in [
			CollectionStrategyType.AGGRESSIVE_COLLECTION,
			CollectionStrategyType.LEGAL_ACTION
		]
		assert recommendation.escalation_risk > 0.5
		assert CollectionChannelType.LEGAL_NOTICE in recommendation.recommended_channels or \
			   CollectionChannelType.LETTER in recommendation.recommended_channels
	
	async def test_batch_optimize_strategies(self, collections_service):
		"""Test batch optimization of collection strategies."""
		
		# Create multiple customers
		customers = []
		for i in range(3):
			customer = ARCustomer(
				id=uuid7str(),
				tenant_id=uuid7str(),
				customer_code=f"BATCH{i+1:03d}",
				legal_name=f"Batch Customer {i+1}",
				customer_type=ARCustomerType.CORPORATION,
				status=ARCustomerStatus.ACTIVE,
				credit_limit=Decimal('15000.00'),
				payment_terms_days=30,
				total_outstanding=Decimal('8000.00'),
				overdue_amount=Decimal('4000.00'),
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
			customers.append(customer)
		
		recommendations = await collections_service.batch_optimize_strategies(customers)
		
		assert len(recommendations) == 3
		for recommendation in recommendations:
			assert isinstance(recommendation, CollectionStrategyRecommendation)
			assert recommendation.customer_id in [c.id for c in customers]
			assert 0 <= recommendation.success_probability <= 1
	
	def test_determine_optimal_strategy_soft_approach(self, collections_service):
		"""Test strategy determination for good payment history customers."""
		
		good_profile = CustomerCollectionProfile(
			customer_id=uuid7str(),
			tenant_id=uuid7str(),
			customer_type="corporation",
			total_outstanding=Decimal('5000.00'),
			overdue_amount=Decimal('2000.00'),
			days_overdue=20,  # Recently overdue
			payment_consistency_score=0.9,  # Excellent history
			communication_responsiveness=0.8  # Very responsive
		)
		
		strategy = collections_service._determine_optimal_strategy(good_profile)
		assert strategy == CollectionStrategyType.SOFT_APPROACH
	
	def test_determine_optimal_strategy_legal_action(self, collections_service):
		"""Test strategy determination for severely overdue customers."""
		
		severe_profile = CustomerCollectionProfile(
			customer_id=uuid7str(),
			tenant_id=uuid7str(),
			customer_type="corporation",
			total_outstanding=Decimal('50000.00'),
			overdue_amount=Decimal('25000.00'),
			days_overdue=120,  # Severely overdue
			payment_consistency_score=0.2,  # Poor history
			communication_responsiveness=0.1  # Non-responsive
		)
		
		strategy = collections_service._determine_optimal_strategy(severe_profile)
		assert strategy == CollectionStrategyType.LEGAL_ACTION
	
	def test_select_optimal_channels(self, collections_service):
		"""Test optimal channel selection based on customer profile."""
		
		# Responsive customer profile
		responsive_profile = CustomerCollectionProfile(
			customer_id=uuid7str(),
			tenant_id=uuid7str(),
			customer_type="corporation",
			total_outstanding=Decimal('15000.00'),
			overdue_amount=Decimal('8000.00'),
			days_overdue=45,
			communication_responsiveness=0.8  # Very responsive
		)
		
		channels = collections_service._select_optimal_channels(responsive_profile)
		
		assert CollectionChannelType.EMAIL in channels  # Always included
		assert CollectionChannelType.PHONE in channels  # For responsive customers
		assert CollectionChannelType.LETTER in channels  # For significant amounts
		assert len(channels) <= 3  # Limited to top 3 channels
	
	def test_calculate_success_probability(self, collections_service):
		"""Test success probability calculation logic."""
		
		# High success probability profile
		good_profile = CustomerCollectionProfile(
			customer_id=uuid7str(),
			tenant_id=uuid7str(),
			customer_type="corporation",
			total_outstanding=Decimal('5000.00'),
			overdue_amount=Decimal('2000.00'),
			days_overdue=30,
			payment_consistency_score=0.9,
			communication_responsiveness=0.8,
			previous_collection_attempts=2,
			successful_collections=2
		)
		
		strategy = CollectionStrategyType.SOFT_APPROACH
		channels = [CollectionChannelType.EMAIL, CollectionChannelType.PHONE]
		
		success_prob = collections_service._calculate_success_probability(
			good_profile, strategy, channels
		)
		
		assert 0.1 <= success_prob <= 0.95
		assert success_prob > 0.6  # Should be high for good profile
		
		# Low success probability profile
		poor_profile = CustomerCollectionProfile(
			customer_id=uuid7str(),
			tenant_id=uuid7str(),
			customer_type="individual",
			total_outstanding=Decimal('20000.00'),
			overdue_amount=Decimal('15000.00'),
			days_overdue=90,  # Severely overdue
			payment_consistency_score=0.2,  # Poor history
			communication_responsiveness=0.1,  # Non-responsive
			previous_collection_attempts=5,
			successful_collections=1
		)
		
		poor_success_prob = collections_service._calculate_success_probability(
			poor_profile, CollectionStrategyType.AGGRESSIVE_COLLECTION, channels
		)
		
		assert 0.1 <= poor_success_prob <= 0.95
		assert poor_success_prob < success_prob  # Should be lower for poor profile
	
	async def test_create_campaign_plan(self, collections_service):
		"""Test creating optimized collection campaign plan."""
		
		# Create customer profiles for campaign
		profiles = []
		for i in range(5):
			profile = CustomerCollectionProfile(
				customer_id=uuid7str(),
				tenant_id=uuid7str(),
				customer_type="corporation",
				total_outstanding=Decimal('10000.00'),
				overdue_amount=Decimal('5000.00'),
				days_overdue=30 + i*10,
				payment_consistency_score=0.8 - i*0.1,
				communication_responsiveness=0.7 - i*0.1
			)
			profiles.append(profile)
		
		campaign_parameters = {
			'name': 'Test Campaign',
			'duration_days': 45,
			'start_date': date.today()
		}
		
		campaign_plan = await collections_service.create_campaign_plan(
			profiles, campaign_parameters
		)
		
		assert isinstance(campaign_plan, CollectionCampaignPlan)
		assert campaign_plan.campaign_name == 'Test Campaign'
		assert len(campaign_plan.target_customers) == 5
		assert campaign_plan.campaign_duration_days == 45
		assert campaign_plan.predicted_collection_rate > 0
		assert campaign_plan.predicted_collection_amount > 0
		assert campaign_plan.estimated_cost > 0
		assert campaign_plan.roi_estimate > 0
		assert campaign_plan.staff_hours_required > 0
		assert len(campaign_plan.strategy_distribution) > 0
		assert len(campaign_plan.channel_allocation) > 0


class TestServiceFactory:
	"""Test collections AI service factory functions."""
	
	async def test_create_collections_ai_service_default_config(self):
		"""Test creating service with default configuration."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		
		service = await create_collections_ai_service(tenant_id, user_id)
		
		assert isinstance(service, APGCollectionsAIService)
		assert service.tenant_id == tenant_id
		assert service.user_id == user_id
		assert service.config.collections_model_name == "ar_collections_optimizer_v1"
		assert service.config.model_version == "1.5.0"
	
	async def test_create_collections_ai_service_custom_config(self):
		"""Test creating service with custom configuration."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		
		custom_config = CollectionsOptimizationConfig(
			ai_orchestration_endpoint="https://custom.ai.com/v1",
			collections_model_name="custom_model",
			model_version="2.0.0"
		)
		
		service = await create_collections_ai_service(tenant_id, user_id, custom_config)
		
		assert isinstance(service, APGCollectionsAIService)
		assert service.config.collections_model_name == "custom_model"
		assert service.config.model_version == "2.0.0"


class TestIntegrationScenarios:
	"""Test realistic integration scenarios."""
	
	@pytest.fixture
	def responsive_customer(self):
		"""Customer with good communication history."""
		return ARCustomer(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_code="RESPONSIVE001",
			legal_name="Responsive Corp",
			customer_type=ARCustomerType.CORPORATION,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal('30000.00'),
			payment_terms_days=30,
			total_outstanding=Decimal('8000.00'),
			overdue_amount=Decimal('3000.00'),
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
	
	@pytest.fixture
	def non_responsive_customer(self):
		"""Customer with poor communication history."""
		return ARCustomer(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_code="NONRESP001",
			legal_name="Non-Responsive LLC",
			customer_type=ARCustomerType.CORPORATION,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal('15000.00'),
			payment_terms_days=30,
			total_outstanding=Decimal('18000.00'),
			overdue_amount=Decimal('15000.00'),
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
	
	async def test_responsive_customer_optimization(self, responsive_customer):
		"""Test optimization for responsive customer."""
		service = await create_collections_ai_service(uuid7str(), uuid7str())
		
		# Create moderate overdue situation
		invoices = [
			ARInvoice(
				id=uuid7str(),
				tenant_id=responsive_customer.tenant_id,
				customer_id=responsive_customer.id,
				invoice_number="RESP-001",
				invoice_date=date.today() - timedelta(days=45),
				due_date=date.today() - timedelta(days=15),
				total_amount=Decimal('3000.00'),
				paid_amount=Decimal('0.00'),
				currency_code="USD",
				status=ARInvoiceStatus.OVERDUE,
				payment_status="overdue",
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
		]
		
		# Good collection history
		collection_history = [
			ARCollectionActivity(
				id=uuid7str(),
				tenant_id=responsive_customer.tenant_id,
				customer_id=responsive_customer.id,
				activity_type="email",
				activity_date=date.today() - timedelta(days=5),
				priority=ARCollectionPriority.NORMAL,
				contact_method="email",
				outcome="successful",
				status="completed",
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
		]
		
		recommendation = await service.optimize_collection_strategy(
			responsive_customer, invoices, [], collection_history
		)
		
		# Should recommend soft approach for responsive customer
		assert recommendation.strategy_type in [
			CollectionStrategyType.SOFT_APPROACH,
			CollectionStrategyType.STANDARD_DUNNING
		]
		assert recommendation.success_probability > 0.6
		assert recommendation.escalation_risk < 0.5
		assert 'friendly' in recommendation.message_tone or 'professional' in recommendation.message_tone
	
	async def test_non_responsive_customer_optimization(self, non_responsive_customer):
		"""Test optimization for non-responsive customer."""
		service = await create_collections_ai_service(uuid7str(), uuid7str())
		
		# Create severely overdue situation
		invoices = [
			ARInvoice(
				id=uuid7str(),
				tenant_id=non_responsive_customer.tenant_id,
				customer_id=non_responsive_customer.id,
				invoice_number="NONRESP-001",
				invoice_date=date.today() - timedelta(days=120),
				due_date=date.today() - timedelta(days=90),
				total_amount=Decimal('15000.00'),
				paid_amount=Decimal('0.00'),
				currency_code="USD",
				status=ARInvoiceStatus.OVERDUE,
				payment_status="overdue",
				created_by=uuid7str(),
				updated_by=uuid7str()
			)
		]
		
		# Poor collection history with no responses
		collection_history = [
			ARCollectionActivity(
				id=uuid7str(),
				tenant_id=non_responsive_customer.tenant_id,
				customer_id=non_responsive_customer.id,
				activity_type="phone_call",
				activity_date=date.today() - timedelta(days=i*7),
				priority=ARCollectionPriority.HIGH,
				contact_method="phone",
				outcome="no_response",
				status="completed",
				created_by=uuid7str(),
				updated_by=uuid7str()
			) for i in range(5)
		]
		
		recommendation = await service.optimize_collection_strategy(
			non_responsive_customer, invoices, [], collection_history
		)
		
		# Should recommend aggressive strategy for non-responsive customer
		assert recommendation.strategy_type in [
			CollectionStrategyType.AGGRESSIVE_COLLECTION,
			CollectionStrategyType.LEGAL_ACTION
		]
		assert recommendation.escalation_risk > 0.5
		assert CollectionChannelType.LEGAL_NOTICE in recommendation.recommended_channels or \
			   CollectionChannelType.LETTER in recommendation.recommended_channels
		assert 'urgent' in recommendation.message_tone or 'formal' in recommendation.message_tone


# Run tests
if __name__ == "__main__":
	pytest.main([__file__, "-v"])