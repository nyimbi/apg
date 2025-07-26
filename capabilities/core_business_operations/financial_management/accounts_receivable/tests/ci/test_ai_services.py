"""
APG Accounts Receivable - AI Services Tests
Unit tests for AI-powered service components with APG integration

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any

from uuid_extensions import uuid7str

from ..ai_credit_scoring import (
	APGCreditScoringService, CreditScoringFeatures, CreditScoringResult, CreditScoringConfig
)
from ..ai_collections_optimization import (
	APGCollectionsAIService, CustomerCollectionProfile, CollectionStrategyRecommendation,
	CollectionChannelType, CollectionStrategyType
)
from ..ai_cashflow_forecasting import (
	APGCashFlowForecastingService, CashFlowForecastInput, CashFlowDataPoint, CashFlowForecastSummary
)
from ..models import (
	ARCustomer, ARInvoice, ARPayment, ARCollectionActivity,
	ARCustomerType, ARCustomerStatus, ARInvoiceStatus, ARPaymentStatus
)


class TestAPGCreditScoringService:
	"""Test AI credit scoring service functionality."""
	
	@pytest.fixture
	def credit_service(self):
		"""Create credit scoring service for testing."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		return APGCreditScoringService(tenant_id, user_id)
	
	@pytest.fixture
	def sample_customer(self):
		"""Create sample customer for testing."""
		return ARCustomer(
			id=uuid7str(),
			tenant_id=uuid7str(),
			customer_code='CREDIT001',
			legal_name='Credit Test Customer',
			customer_type=ARCustomerType.CORPORATION,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal('25000.00'),
			total_outstanding=Decimal('8000.00'),
			overdue_amount=Decimal('1500.00'),
			payment_terms_days=30,
			created_by=uuid7str(),
			updated_by=uuid7str()
		)
	
	async def test_assess_customer_credit_success(self, credit_service, sample_customer):
		"""Test successful credit assessment."""
		
		# Mock APG federated learning service
		mock_fl_result = {
			'credit_score': 725,
			'risk_level': 'MEDIUM',
			'confidence_score': 0.87,
			'feature_importance': {
				'payment_history': 0.35,
				'credit_utilization': 0.25,
				'financial_stability': 0.20,
				'business_longevity': 0.20
			},
			'explanations': [
				'Good payment history with average 28-day payment cycles',
				'Credit utilization at 32% is within acceptable range',
				'Stable business operations with consistent revenue'
			]
		}
		
		with patch.object(credit_service, '_get_customer_features', new_callable=AsyncMock) as mock_features:
			mock_features.return_value = CreditScoringFeatures(
				customer_id=sample_customer.id,
				payment_history_score=0.75,
				credit_utilization_ratio=0.32,
				overdue_frequency=0.05,
				average_payment_days=28.5,
				total_outstanding_amount=Decimal('8000.00'),
				business_age_months=36,
				invoice_count=24,
				payment_count=22
			)
			
			with patch.object(credit_service, '_call_federated_learning', new_callable=AsyncMock) as mock_fl:
				mock_fl.return_value = mock_fl_result
				
				result = await credit_service.assess_customer_credit(sample_customer)
				
				assert result.customer_id == sample_customer.id
				assert result.credit_score == 725
				assert result.risk_level == 'MEDIUM'
				assert result.confidence_score == 0.87
				assert len(result.explanations) == 3
				assert 'payment_history' in result.feature_importance
				mock_fl.assert_called_once()
	
	async def test_batch_assess_customers_credit(self, credit_service):
		"""Test batch credit assessment."""
		customer_ids = [uuid7str() for _ in range(3)]
		
		# Mock individual assessments
		mock_results = []
		for i, customer_id in enumerate(customer_ids):
			mock_results.append(CreditScoringResult(
				customer_id=customer_id,
				assessment_date=date.today(),
				credit_score=700 + (i * 25),
				risk_level='MEDIUM' if i == 1 else 'LOW',
				confidence_score=0.85 + (i * 0.05),
				feature_importance={'payment_history': 0.4},
				explanations=[f'Assessment for customer {i+1}']
			))
		
		with patch.object(credit_service, 'assess_customer_credit', new_callable=AsyncMock) as mock_assess:
			mock_assess.side_effect = mock_results
			
			with patch.object(credit_service, '_get_customer_from_db', new_callable=AsyncMock) as mock_get:
				# Mock customer objects
				mock_customers = []
				for customer_id in customer_ids:
					customer = Mock()
					customer.id = customer_id
					mock_customers.append(customer)
				mock_get.side_effect = mock_customers
				
				results = await credit_service.batch_assess_customers_credit(customer_ids)
				
				assert len(results) == 3
				assert all(isinstance(r, CreditScoringResult) for r in results)
				assert results[0].credit_score == 700
				assert results[1].credit_score == 725
				assert results[2].credit_score == 750
				assert mock_assess.call_count == 3
	
	async def test_monitor_credit_risk_changes(self, credit_service):
		"""Test credit risk monitoring."""
		customer_id = uuid7str()
		
		# Mock historical assessment
		historical_assessment = CreditScoringResult(
			customer_id=customer_id,
			assessment_date=date.today() - timedelta(days=90),
			credit_score=680,
			risk_level='MEDIUM',
			confidence_score=0.82,
			feature_importance={'payment_history': 0.4},
			explanations=['Previous assessment']
		)
		
		# Mock current assessment
		current_assessment = CreditScoringResult(
			customer_id=customer_id,
			assessment_date=date.today(),
			credit_score=720,
			risk_level='LOW',
			confidence_score=0.88,
			feature_importance={'payment_history': 0.4},
			explanations=['Current assessment']
		)
		
		with patch.object(credit_service, '_get_latest_assessment', new_callable=AsyncMock) as mock_latest:
			mock_latest.return_value = historical_assessment
			
			with patch.object(credit_service, 'assess_customer_credit', new_callable=AsyncMock) as mock_assess:
				mock_assess.return_value = current_assessment
				
				with patch.object(credit_service, '_get_customer_from_db', new_callable=AsyncMock) as mock_get:
					mock_customer = Mock()
					mock_customer.id = customer_id
					mock_get.return_value = mock_customer
					
					risk_change = await credit_service.monitor_credit_risk_changes(customer_id)
					
					assert risk_change['customer_id'] == customer_id
					assert risk_change['score_change'] == 40  # 720 - 680
					assert risk_change['risk_level_change'] == 'IMPROVED'  # MEDIUM -> LOW
					assert risk_change['confidence_change'] == 0.06  # 0.88 - 0.82
					assert risk_change['monitoring_recommendation'] == 'STANDARD'  # Improved risk
	
	async def test_get_customer_credit_insights(self, credit_service):
		"""Test customer credit insights generation."""
		customer_id = uuid7str()
		
		# Mock assessment result
		mock_assessment = CreditScoringResult(
			customer_id=customer_id,
			assessment_date=date.today(),
			credit_score=750,
			risk_level='LOW',
			confidence_score=0.91,
			feature_importance={
				'payment_history': 0.40,
				'credit_utilization': 0.25,
				'financial_stability': 0.20,
				'business_longevity': 0.15
			},
			explanations=[
				'Excellent payment history',
				'Low credit utilization',
				'Strong financial position'
			]
		)
		
		with patch.object(credit_service, '_get_latest_assessment', new_callable=AsyncMock) as mock_latest:
			mock_latest.return_value = mock_assessment
			
			with patch.object(credit_service, '_calculate_credit_trends', new_callable=AsyncMock) as mock_trends:
				mock_trends.return_value = {
					'score_trend': 'IMPROVING',
					'trend_strength': 0.15,
					'months_trend': 6
				}
				
				insights = await credit_service.get_customer_credit_insights(customer_id)
				
				assert insights['customer_id'] == customer_id
				assert insights['current_score'] == 750
				assert insights['risk_level'] == 'LOW'
				assert insights['score_trend'] == 'IMPROVING'
				assert len(insights['key_strengths']) > 0
				assert len(insights['improvement_areas']) >= 0
				assert 'recommended_credit_limit' in insights
				assert 'monitoring_frequency' in insights


class TestAPGCollectionsAIService:
	"""Test AI collections optimization service functionality."""
	
	@pytest.fixture
	def collections_service(self):
		"""Create collections AI service for testing."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		return APGCollectionsAIService(tenant_id, user_id)
	
	@pytest.fixture
	def sample_customer_profile(self):
		"""Create sample customer profile for testing."""
		return CustomerCollectionProfile(
			customer_id=uuid7str(),
			overdue_amount=Decimal('5000.00'),
			days_overdue=15,
			payment_history_score=0.75,
			previous_collection_attempts=2,
			preferred_contact_method='EMAIL',
			response_rate_email=0.65,
			response_rate_phone=0.45,
			last_payment_date=date.today() - timedelta(days=45),
			customer_segment='SMALL_BUSINESS',
			risk_level='MEDIUM'
		)
	
	async def test_optimize_collection_strategy_success(self, collections_service, sample_customer_profile):
		"""Test collection strategy optimization."""
		
		# Mock AI orchestration response
		mock_ai_response = {
			'recommended_strategy': 'EMAIL_SEQUENCE',
			'contact_method': 'EMAIL',
			'success_probability': 0.72,
			'estimated_resolution_days': 12,
			'priority_level': 'MEDIUM',
			'strategy_explanation': 'Customer responds well to email communication',
			'alternative_strategies': [
				{
					'strategy': 'PHONE_FOLLOW_UP',
					'success_probability': 0.58,
					'reasoning': 'Lower email response rate suggests phone follow-up'
				}
			]
		}
		
		with patch.object(collections_service, '_call_ai_orchestration', new_callable=AsyncMock) as mock_ai:
			mock_ai.return_value = mock_ai_response
			
			recommendation = await collections_service.optimize_collection_strategy(sample_customer_profile)
			
			assert recommendation.customer_id == sample_customer_profile.customer_id
			assert recommendation.recommended_strategy == CollectionStrategyType.EMAIL_SEQUENCE
			assert recommendation.contact_method == CollectionChannelType.EMAIL
			assert recommendation.success_probability == 0.72
			assert recommendation.estimated_resolution_days == 12
			assert recommendation.priority_level == 'MEDIUM'
			assert len(recommendation.alternative_strategies) > 0
			mock_ai.assert_called_once()
	
	async def test_create_campaign_plan(self, collections_service):
		"""Test collection campaign plan creation."""
		customer_profiles = [
			CustomerCollectionProfile(
				customer_id=uuid7str(),
				overdue_amount=Decimal('3000.00'),
				days_overdue=10,
				payment_history_score=0.8,
				previous_collection_attempts=1,
				customer_segment='INDIVIDUAL'
			),
			CustomerCollectionProfile(
				customer_id=uuid7str(),
				overdue_amount=Decimal('8000.00'),
				days_overdue=25,
				payment_history_score=0.6,
				previous_collection_attempts=3,
				customer_segment='CORPORATION'
			)
		]
		
		# Mock strategy recommendations
		mock_recommendations = [
			CollectionStrategyRecommendation(
				customer_id=customer_profiles[0].customer_id,
				recommended_strategy=CollectionStrategyType.EMAIL_REMINDER,
				contact_method=CollectionChannelType.EMAIL,
				success_probability=0.75,
				estimated_resolution_days=8,
				priority_level='LOW'
			),
			CollectionStrategyRecommendation(
				customer_id=customer_profiles[1].customer_id,
				recommended_strategy=CollectionStrategyType.PHONE_CALL,
				contact_method=CollectionChannelType.PHONE,
				success_probability=0.65,
				estimated_resolution_days=15,
				priority_level='HIGH'
			)
		]
		
		with patch.object(collections_service, 'optimize_collection_strategy', new_callable=AsyncMock) as mock_optimize:
			mock_optimize.side_effect = mock_recommendations
			
			campaign_plan = await collections_service.create_campaign_plan(customer_profiles)
			
			assert campaign_plan['campaign_name'] == 'Automated Collections Campaign'
			assert len(campaign_plan['strategies']) == 2
			assert campaign_plan['total_target_amount'] == Decimal('11000.00')
			assert campaign_plan['estimated_success_rate'] == 0.70  # Average of 0.75 and 0.65
			assert len(campaign_plan['phases']) > 0
			assert mock_optimize.call_count == 2
	
	async def test_batch_optimize_strategies(self, collections_service):
		"""Test batch strategy optimization."""
		customer_profiles = [
			CustomerCollectionProfile(
				customer_id=uuid7str(),
				overdue_amount=Decimal('2000.00'),
				days_overdue=5,
				customer_segment='INDIVIDUAL'
			),
			CustomerCollectionProfile(
				customer_id=uuid7str(),
				overdue_amount=Decimal('6000.00'),
				days_overdue=20,
				customer_segment='CORPORATION'
			),
			CustomerCollectionProfile(
				customer_id=uuid7str(),
				overdue_amount=Decimal('4000.00'),
				days_overdue=12,
				customer_segment='SMALL_BUSINESS'
			)
		]
		
		# Mock individual optimizations
		mock_recommendations = []
		for i, profile in enumerate(customer_profiles):
			mock_recommendations.append(CollectionStrategyRecommendation(
				customer_id=profile.customer_id,
				recommended_strategy=CollectionStrategyType.EMAIL_REMINDER,
				contact_method=CollectionChannelType.EMAIL,
				success_probability=0.7 + (i * 0.05),
				estimated_resolution_days=10 + (i * 2),
				priority_level='MEDIUM'
			))
		
		with patch.object(collections_service, 'optimize_collection_strategy', new_callable=AsyncMock) as mock_optimize:
			mock_optimize.side_effect = mock_recommendations
			
			results = await collections_service.batch_optimize_strategies(customer_profiles)
			
			assert len(results) == 3
			assert all(isinstance(r, CollectionStrategyRecommendation) for r in results)
			assert results[0].success_probability == 0.70
			assert results[1].success_probability == 0.75
			assert results[2].success_probability == 0.80
			assert mock_optimize.call_count == 3


class TestAPGCashFlowForecastingService:
	"""Test AI cash flow forecasting service functionality."""
	
	@pytest.fixture
	def cashflow_service(self):
		"""Create cash flow forecasting service for testing."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		return APGCashFlowForecastingService(tenant_id, user_id)
	
	@pytest.fixture
	def sample_forecast_input(self):
		"""Create sample forecast input for testing."""
		return CashFlowForecastInput(
			tenant_id=uuid7str(),
			forecast_start_date=date.today(),
			forecast_end_date=date.today() + timedelta(days=30),
			include_seasonal_trends=True,
			include_external_factors=True,
			scenario_type='realistic',
			confidence_level=0.95
		)
	
	async def test_generate_forecast_success(self, cashflow_service, sample_forecast_input):
		"""Test successful cash flow forecast generation."""
		
		# Mock time series analytics response
		mock_forecast_points = []
		current_date = sample_forecast_input.forecast_start_date
		for i in range(30):
			mock_forecast_points.append({
				'forecast_date': current_date + timedelta(days=i),
				'expected_collections': float(Decimal('2500.00') + (i * 50)),
				'invoice_receipts': float(Decimal('1800.00') + (i * 30)),
				'overdue_collections': float(Decimal('700.00') + (i * 20)),
				'total_cash_flow': float(Decimal('5000.00') + (i * 100)),
				'confidence_interval_lower': float(Decimal('4500.00') + (i * 90)),
				'confidence_interval_upper': float(Decimal('5500.00') + (i * 110))
			})
		
		mock_ts_response = {
			'forecast_points': mock_forecast_points,
			'overall_accuracy': 0.92,
			'model_confidence': 0.88,
			'seasonal_factors': ['Month-end payment patterns', 'Holiday payment delays'],
			'risk_factors': ['Economic uncertainty', 'Industry volatility'],
			'insights': ['Strong payment patterns expected', 'Minimal seasonal impact']
		}
		
		with patch.object(cashflow_service, '_prepare_historical_data', new_callable=AsyncMock) as mock_prep:
			mock_prep.return_value = {'historical_payments': [], 'invoice_data': []}
			
			with patch.object(cashflow_service, '_call_time_series_analytics', new_callable=AsyncMock) as mock_ts:
				mock_ts.return_value = mock_ts_response
				
				forecast = await cashflow_service.generate_forecast(sample_forecast_input)
				
				assert forecast.forecast_id is not None
				assert forecast.tenant_id == sample_forecast_input.tenant_id
				assert len(forecast.forecast_points) == 30
				assert forecast.overall_accuracy == 0.92
				assert forecast.model_confidence == 0.88
				assert len(forecast.seasonal_factors) == 2
				assert len(forecast.risk_factors) == 2
				assert len(forecast.insights) == 1
				mock_ts.assert_called_once()
	
	async def test_compare_scenarios(self, cashflow_service, sample_forecast_input):
		"""Test scenario comparison functionality."""
		
		# Mock different scenario responses
		scenarios = ['optimistic', 'realistic', 'pessimistic']
		mock_scenario_data = {}
		
		for scenario in scenarios:
			base_amount = Decimal('3000.00') if scenario == 'optimistic' else \
						  Decimal('2500.00') if scenario == 'realistic' else \
						  Decimal('2000.00')
			
			mock_points = []
			for i in range(10):
				mock_points.append(CashFlowDataPoint(
					forecast_date=date.today() + timedelta(days=i),
					expected_collections=base_amount + (i * 50),
					invoice_receipts=base_amount * Decimal('0.7'),
					overdue_collections=base_amount * Decimal('0.3'),
					total_cash_flow=base_amount + (i * 100),
					confidence_interval_lower=base_amount * Decimal('0.9'),
					confidence_interval_upper=base_amount * Decimal('1.1')
				))
			
			mock_scenario_data[scenario] = mock_points
		
		with patch.object(cashflow_service, 'generate_forecast', new_callable=AsyncMock) as mock_forecast:
			# Mock different forecast results for each scenario
			def side_effect(forecast_input):
				scenario = forecast_input.scenario_type
				return CashFlowForecastSummary(
					forecast_id=uuid7str(),
					tenant_id=forecast_input.tenant_id,
					forecast_start_date=forecast_input.forecast_start_date,
					forecast_end_date=forecast_input.forecast_end_date,
					scenario_type=scenario,
					forecast_points=mock_scenario_data[scenario],
					overall_accuracy=0.90,
					model_confidence=0.85,
					seasonal_factors=[],
					risk_factors=[],
					insights=[]
				)
			
			mock_forecast.side_effect = side_effect
			
			comparison = await cashflow_service.compare_scenarios(sample_forecast_input)
			
			assert 'optimistic' in comparison
			assert 'realistic' in comparison
			assert 'pessimistic' in comparison
			assert len(comparison['optimistic'].forecast_points) == 10
			assert len(comparison['realistic'].forecast_points) == 10
			assert len(comparison['pessimistic'].forecast_points) == 10
			assert mock_forecast.call_count == 3
	
	async def test_get_forecast_accuracy_metrics(self, cashflow_service):
		"""Test forecast accuracy metrics calculation."""
		tenant_id = uuid7str()
		
		# Mock historical accuracy data
		mock_accuracy_data = {
			'overall_accuracy': 0.89,
			'accuracy_by_period': {
				'7_days': 0.95,
				'14_days': 0.92,
				'30_days': 0.89,
				'60_days': 0.84,
				'90_days': 0.78
			},
			'accuracy_trends': [
				{'period': '2024-12', 'accuracy': 0.87},
				{'period': '2025-01', 'accuracy': 0.91},
				{'period': '2025-02', 'accuracy': 0.89}
			],
			'model_performance': {
				'mean_absolute_error': 0.08,
				'root_mean_square_error': 0.12,
				'directional_accuracy': 0.85
			}
		}
		
		with patch.object(cashflow_service, '_calculate_historical_accuracy', new_callable=AsyncMock) as mock_calc:
			mock_calc.return_value = mock_accuracy_data
			
			metrics = await cashflow_service.get_forecast_accuracy_metrics(tenant_id)
			
			assert metrics['overall_accuracy'] == 0.89
			assert metrics['accuracy_by_period']['7_days'] == 0.95
			assert metrics['accuracy_by_period']['90_days'] == 0.78
			assert len(metrics['accuracy_trends']) == 3
			assert 'model_performance' in metrics
			assert metrics['model_performance']['directional_accuracy'] == 0.85
			mock_calc.assert_called_once_with(tenant_id)


class TestAIServiceIntegration:
	"""Test AI service integration scenarios."""
	
	async def test_complete_ai_workflow(self):
		"""Test complete AI workflow across all services."""
		tenant_id = uuid7str()
		user_id = uuid7str()
		
		# Initialize AI services
		credit_service = APGCreditScoringService(tenant_id, user_id)
		collections_service = APGCollectionsAIService(tenant_id, user_id)
		cashflow_service = APGCashFlowForecastingService(tenant_id, user_id)
		
		customer_id = uuid7str()
		
		# Mock customer data
		mock_customer = ARCustomer(
			id=customer_id,
			tenant_id=tenant_id,
			customer_code='AI001',
			legal_name='AI Integration Test',
			customer_type=ARCustomerType.CORPORATION,
			status=ARCustomerStatus.ACTIVE,
			credit_limit=Decimal('30000.00'),
			total_outstanding=Decimal('12000.00'),
			overdue_amount=Decimal('3000.00'),
			payment_terms_days=30,
			created_by=user_id,
			updated_by=user_id
		)
		
		# 1. Credit Assessment
		mock_credit_result = CreditScoringResult(
			customer_id=customer_id,
			assessment_date=date.today(),
			credit_score=680,
			risk_level='MEDIUM',
			confidence_score=0.83,
			feature_importance={'payment_history': 0.4},
			explanations=['Credit assessment complete']
		)
		
		# 2. Collections Optimization
		mock_collection_profile = CustomerCollectionProfile(
			customer_id=customer_id,
			overdue_amount=Decimal('3000.00'),
			days_overdue=15,
			payment_history_score=0.68,
			previous_collection_attempts=1,
			customer_segment='CORPORATION'
		)
		
		mock_collection_strategy = CollectionStrategyRecommendation(
			customer_id=customer_id,
			recommended_strategy=CollectionStrategyType.EMAIL_REMINDER,
			contact_method=CollectionChannelType.EMAIL,
			success_probability=0.72,
			estimated_resolution_days=10,
			priority_level='MEDIUM'
		)
		
		# 3. Cash Flow Forecast
		mock_forecast_input = CashFlowForecastInput(
			tenant_id=tenant_id,
			forecast_start_date=date.today(),
			forecast_end_date=date.today() + timedelta(days=30),
			scenario_type='realistic'
		)
		
		mock_forecast_result = CashFlowForecastSummary(
			forecast_id=uuid7str(),
			tenant_id=tenant_id,
			forecast_start_date=date.today(),
			forecast_end_date=date.today() + timedelta(days=30),
			scenario_type='realistic',
			forecast_points=[],
			overall_accuracy=0.91,
			model_confidence=0.87,
			seasonal_factors=[],
			risk_factors=[],
			insights=[]
		)
		
		# Mock all service calls
		with patch.object(credit_service, 'assess_customer_credit', new_callable=AsyncMock) as mock_credit:
			mock_credit.return_value = mock_credit_result
			
			with patch.object(collections_service, 'optimize_collection_strategy', new_callable=AsyncMock) as mock_collections:
				mock_collections.return_value = mock_collection_strategy
				
				with patch.object(cashflow_service, 'generate_forecast', new_callable=AsyncMock) as mock_forecast:
					mock_forecast.return_value = mock_forecast_result
					
					# Execute workflow
					credit_assessment = await credit_service.assess_customer_credit(mock_customer)
					collection_strategy = await collections_service.optimize_collection_strategy(mock_collection_profile)
					cash_forecast = await cashflow_service.generate_forecast(mock_forecast_input)
					
					# Verify integration consistency
					assert credit_assessment.customer_id == customer_id
					assert collection_strategy.customer_id == customer_id
					assert cash_forecast.tenant_id == tenant_id
					
					# Verify AI coordination
					assert credit_assessment.risk_level == 'MEDIUM'
					assert collection_strategy.priority_level == 'MEDIUM'
					assert cash_forecast.overall_accuracy > 0.8
					
					# Verify service calls
					mock_credit.assert_called_once()
					mock_collections.assert_called_once()
					mock_forecast.assert_called_once()


# Run tests
if __name__ == "__main__":
	pytest.main([__file__, "-v"])