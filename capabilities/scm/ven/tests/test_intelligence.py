"""
APG Vendor Management - Intelligence Service Tests
Comprehensive tests for AI-powered vendor intelligence functionality

Author: Nyimbi Odero (nyimbi@gmail.com)
Copyright: © 2025 Datacraft (www.datacraft.co.ke)
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4
from decimal import Decimal
from datetime import datetime, timedelta

from ..intelligence_service import VendorIntelligenceEngine
from ..models import VMIntelligence, BehaviorPattern, PredictiveInsight, OptimizationPlan


# ============================================================================
# INTELLIGENCE ENGINE INITIALIZATION TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.ai
@pytest.mark.asyncio
class TestIntelligenceEngineInitialization:
	"""Test intelligence engine initialization and configuration"""
	
	async def test_engine_creation(self, mock_db_context, test_tenant_id):
		"""Test intelligence engine creation"""
		
		engine = VendorIntelligenceEngine(test_tenant_id, mock_db_context)
		
		assert engine.tenant_id == test_tenant_id
		assert engine.db_context == mock_db_context
		assert engine.current_user_id is None
		assert engine.model_version == 'v1.0'
	
	async def test_engine_user_setting(self, intelligence_engine, test_user_id):
		"""Test setting current user on intelligence engine"""
		
		intelligence_engine.set_current_user(test_user_id)
		
		assert intelligence_engine.current_user_id == test_user_id
	
	async def test_engine_validation(self, mock_db_context):
		"""Test engine validation with invalid parameters"""
		
		# Invalid tenant ID
		with pytest.raises(ValueError):
			VendorIntelligenceEngine("invalid-uuid", mock_db_context)
		
		# None database context
		with pytest.raises(ValueError):
			VendorIntelligenceEngine(UUID('00000000-0000-0000-0000-000000000000'), None)


# ============================================================================
# BEHAVIOR PATTERN ANALYSIS TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.ai
@pytest.mark.asyncio
class TestBehaviorPatternAnalysis:
	"""Test vendor behavior pattern analysis functionality"""
	
	async def test_analyze_communication_patterns(self, intelligence_engine, sample_vendor):
		"""Test communication pattern analysis"""
		
		# Mock communication data
		mock_communications = [
			{
				'communication_type': 'email',
				'communication_date': datetime.now() - timedelta(days=1),
				'response_time_hours': 2.5,
				'sentiment_score': 0.8,
				'direction': 'inbound'
			},
			{
				'communication_type': 'email',
				'communication_date': datetime.now() - timedelta(days=3),
				'response_time_hours': 1.2,
				'sentiment_score': 0.9,
				'direction': 'inbound'
			}
		]
		
		intelligence_engine._fetch_all = AsyncMock(return_value=mock_communications)
		
		patterns = await intelligence_engine.analyze_vendor_behavior_patterns(sample_vendor.id)
		
		assert isinstance(patterns, list)
		
		# Should identify responsive communication pattern
		communication_patterns = [p for p in patterns if p.pattern_type == 'communication']
		assert len(communication_patterns) > 0
		
		responsive_pattern = next(
			(p for p in communication_patterns if 'responsive' in p.pattern_name.lower()), 
			None
		)
		assert responsive_pattern is not None
		assert responsive_pattern.confidence > 0.7
	
	async def test_analyze_performance_consistency(self, intelligence_engine, sample_vendor):
		"""Test performance consistency pattern analysis"""
		
		# Mock performance data with consistent scores
		mock_performance = [
			{
				'measurement_period': 'monthly',
				'overall_score': 85.0,
				'start_date': datetime.now() - timedelta(days=30),
				'quality_score': 88.0,
				'delivery_score': 82.0
			},
			{
				'measurement_period': 'monthly',
				'overall_score': 87.0,
				'start_date': datetime.now() - timedelta(days=60),
				'quality_score': 89.0,
				'delivery_score': 85.0
			},
			{
				'measurement_period': 'monthly',
				'overall_score': 86.0,
				'start_date': datetime.now() - timedelta(days=90),
				'quality_score': 87.0,
				'delivery_score': 84.0
			}
		]
		
		intelligence_engine._fetch_all = AsyncMock(return_value=mock_performance)
		
		patterns = await intelligence_engine.analyze_vendor_behavior_patterns(sample_vendor.id)
		
		# Should identify consistent performance pattern
		performance_patterns = [p for p in patterns if p.pattern_type == 'performance']
		assert len(performance_patterns) > 0
		
		consistency_pattern = next(
			(p for p in performance_patterns if 'consistent' in p.pattern_name.lower()),
			None
		)
		assert consistency_pattern is not None
		assert consistency_pattern.confidence > 0.6
	
	async def test_analyze_risk_behavior(self, intelligence_engine, sample_vendor):
		"""Test risk behavior pattern analysis"""
		
		# Mock risk data showing proactive risk management
		mock_risks = [
			{
				'risk_type': 'operational',
				'severity': 'medium',
				'mitigation_status': 'resolved',
				'identified_date': datetime.now() - timedelta(days=30),
				'resolved_date': datetime.now() - timedelta(days=20)
			},
			{
				'risk_type': 'financial',
				'severity': 'low',
				'mitigation_status': 'resolved',
				'identified_date': datetime.now() - timedelta(days=45),
				'resolved_date': datetime.now() - timedelta(days=35)
			}
		]
		
		intelligence_engine._fetch_all = AsyncMock(return_value=mock_risks)
		
		patterns = await intelligence_engine.analyze_vendor_behavior_patterns(sample_vendor.id)
		
		# Should identify proactive risk management pattern
		risk_patterns = [p for p in patterns if p.pattern_type == 'risk_management']
		assert len(risk_patterns) > 0
	
	async def test_pattern_confidence_calculation(self, intelligence_engine, sample_vendor):
		"""Test pattern confidence score calculation"""
		
		# Mock minimal data for low confidence
		intelligence_engine._fetch_all = AsyncMock(return_value=[])
		
		patterns = await intelligence_engine.analyze_vendor_behavior_patterns(sample_vendor.id)
		
		# With no data, patterns should have lower confidence
		for pattern in patterns:
			assert 0.0 <= pattern.confidence <= 1.0
	
	@pytest.mark.performance
	async def test_pattern_analysis_performance(self, intelligence_engine, sample_vendor):
		"""Test pattern analysis performance with large datasets"""
		
		# Mock large dataset
		large_communication_data = [
			{
				'communication_type': 'email',
				'communication_date': datetime.now() - timedelta(days=i),
				'response_time_hours': 2.0 + (i % 5),
				'sentiment_score': 0.7 + (i % 3) * 0.1
			}
			for i in range(1000)
		]
		
		intelligence_engine._fetch_all = AsyncMock(return_value=large_communication_data)
		
		import time
		start_time = time.time()
		
		patterns = await intelligence_engine.analyze_vendor_behavior_patterns(sample_vendor.id)
		
		end_time = time.time()
		execution_time = end_time - start_time
		
		# Should complete within reasonable time
		assert execution_time < 10.0  # 10 seconds threshold
		assert len(patterns) > 0


# ============================================================================
# PREDICTIVE INSIGHTS TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.ai
@pytest.mark.asyncio
class TestPredictiveInsights:
	"""Test predictive insights generation functionality"""
	
	async def test_generate_performance_forecast(self, intelligence_engine, sample_vendor):
		"""Test performance forecast generation"""
		
		# Mock historical performance data with upward trend
		mock_performance = [
			{'overall_score': 80.0, 'start_date': datetime.now() - timedelta(days=120)},
			{'overall_score': 82.0, 'start_date': datetime.now() - timedelta(days=90)},
			{'overall_score': 85.0, 'start_date': datetime.now() - timedelta(days=60)},
			{'overall_score': 87.0, 'start_date': datetime.now() - timedelta(days=30)}
		]
		
		intelligence_engine._fetch_all = AsyncMock(return_value=mock_performance)
		
		insights = await intelligence_engine.generate_predictive_insights(sample_vendor.id)
		
		assert isinstance(insights, list)
		
		# Should generate performance forecast
		performance_forecasts = [i for i in insights if i.insight_type == 'performance_forecast']
		assert len(performance_forecasts) > 0
		
		forecast = performance_forecasts[0]
		assert forecast.confidence > 0.5
		assert forecast.time_horizon > 0
		assert forecast.prediction in ['improvement', 'decline', 'stable']
	
	async def test_generate_risk_prediction(self, intelligence_engine, sample_vendor):
		"""Test risk prediction generation"""
		
		# Mock risk data with escalating pattern
		mock_risks = [
			{
				'risk_type': 'delivery',
				'severity': 'low',
				'identified_date': datetime.now() - timedelta(days=90),
				'overall_risk_score': 30.0
			},
			{
				'risk_type': 'delivery',
				'severity': 'medium',
				'identified_date': datetime.now() - timedelta(days=30),
				'overall_risk_score': 55.0
			}
		]
		
		intelligence_engine._fetch_all = AsyncMock(return_value=mock_risks)
		
		insights = await intelligence_engine.generate_predictive_insights(sample_vendor.id)
		
		# Should generate risk predictions
		risk_predictions = [i for i in insights if i.insight_type == 'risk_forecast']
		assert len(risk_predictions) > 0
		
		risk_forecast = risk_predictions[0]
		assert risk_forecast.confidence > 0.4
		assert 'risk' in risk_forecast.description.lower()
	
	async def test_generate_relationship_insights(self, intelligence_engine, sample_vendor):
		"""Test relationship health insights generation"""
		
		# Mock relationship data
		mock_communications = [
			{
				'sentiment_score': 0.9,
				'communication_date': datetime.now() - timedelta(days=10)
			},
			{
				'sentiment_score': 0.8,
				'communication_date': datetime.now() - timedelta(days=20)
			}
		]
		
		intelligence_engine._fetch_all = AsyncMock(return_value=mock_communications)
		
		insights = await intelligence_engine.generate_predictive_insights(sample_vendor.id)
		
		# Should generate relationship insights
		relationship_insights = [i for i in insights if i.insight_type == 'relationship_health']
		assert len(relationship_insights) > 0
	
	async def test_insight_confidence_weighting(self, intelligence_engine, sample_vendor):
		"""Test insight confidence weighting based on data quality"""
		
		# Test with high-quality data (recent, complete)
		high_quality_data = [
			{
				'overall_score': 85.0,
				'start_date': datetime.now() - timedelta(days=30),
				'data_completeness': 100.0
			}
		]
		
		intelligence_engine._fetch_all = AsyncMock(return_value=high_quality_data)
		
		insights = await intelligence_engine.generate_predictive_insights(sample_vendor.id)
		
		# Insights should have reasonable confidence with good data
		for insight in insights:
			assert insight.confidence >= 0.3  # Minimum threshold
	
	@pytest.mark.ai
	async def test_machine_learning_integration(self, intelligence_engine, mock_ai_models):
		"""Test integration with machine learning models"""
		
		# Mock AI model predictions
		intelligence_engine.ai_models = mock_ai_models
		
		vendor_id = str(uuid4())
		
		# Mock data for ML models
		intelligence_engine._fetch_all = AsyncMock(return_value=[
			{'overall_score': 85.0, 'start_date': datetime.now() - timedelta(days=30)}
		])
		
		insights = await intelligence_engine.generate_predictive_insights(vendor_id)
		
		# Should use ML models for predictions
		assert len(insights) > 0
		
		# Verify ML models were called
		mock_ai_models['performance_predictor'].predict.assert_called()


# ============================================================================
# OPTIMIZATION PLAN GENERATION TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.ai
@pytest.mark.asyncio
class TestOptimizationPlanGeneration:
	"""Test optimization plan generation functionality"""
	
	async def test_generate_performance_optimization(self, intelligence_engine, sample_vendor):
		"""Test performance optimization plan generation"""
		
		objectives = ['performance_improvement']
		
		# Mock vendor data with improvement opportunities
		mock_vendor_data = sample_vendor.model_dump()
		mock_vendor_data['performance_score'] = 75.0  # Room for improvement
		
		intelligence_engine._fetch_one = AsyncMock(return_value=mock_vendor_data)
		intelligence_engine._fetch_all = AsyncMock(return_value=[])
		
		optimization_plan = await intelligence_engine.generate_optimization_plan(
			sample_vendor.id, 
			objectives
		)
		
		assert isinstance(optimization_plan, OptimizationPlan)
		assert optimization_plan.vendor_id == sample_vendor.id
		assert 'performance_improvement' in optimization_plan.optimization_objectives
		assert len(optimization_plan.recommended_actions) > 0
		
		# Should include performance-focused recommendations
		performance_actions = [
			action for action in optimization_plan.recommended_actions
			if 'performance' in action.action_type.lower()
		]
		assert len(performance_actions) > 0
	
	async def test_generate_cost_optimization(self, intelligence_engine, sample_vendor):
		"""Test cost optimization plan generation"""
		
		objectives = ['cost_reduction']
		
		# Mock vendor with high costs
		mock_vendor_data = sample_vendor.model_dump()
		mock_performance_data = [
			{
				'cost_score': 60.0,  # Low cost efficiency
				'total_spend': 1000000.0,
				'average_order_value': 25000.0
			}
		]
		
		intelligence_engine._fetch_one = AsyncMock(return_value=mock_vendor_data)
		intelligence_engine._fetch_all = AsyncMock(return_value=mock_performance_data)
		
		optimization_plan = await intelligence_engine.generate_optimization_plan(
			sample_vendor.id,
			objectives
		)
		
		assert 'cost_reduction' in optimization_plan.optimization_objectives
		
		# Should include cost-focused recommendations
		cost_actions = [
			action for action in optimization_plan.recommended_actions
			if 'cost' in action.action_type.lower() or 'price' in action.action_type.lower()
		]
		assert len(cost_actions) > 0
	
	async def test_generate_risk_mitigation_plan(self, intelligence_engine, sample_vendor):
		"""Test risk mitigation optimization plan"""
		
		objectives = ['risk_mitigation']
		
		# Mock vendor with active risks
		mock_risks = [
			{
				'risk_type': 'delivery',
				'severity': 'high',
				'mitigation_status': 'pending',
				'overall_risk_score': 80.0
			},
			{
				'risk_type': 'financial',
				'severity': 'medium',
				'mitigation_status': 'pending',
				'overall_risk_score': 60.0
			}
		]
		
		intelligence_engine._fetch_one = AsyncMock(return_value=sample_vendor.model_dump())
		intelligence_engine._fetch_all = AsyncMock(return_value=mock_risks)
		
		optimization_plan = await intelligence_engine.generate_optimization_plan(
			sample_vendor.id,
			objectives
		)
		
		assert 'risk_mitigation' in optimization_plan.optimization_objectives
		
		# Should include risk mitigation recommendations
		risk_actions = [
			action for action in optimization_plan.recommended_actions
			if 'risk' in action.action_type.lower()
		]
		assert len(risk_actions) > 0
	
	async def test_multi_objective_optimization(self, intelligence_engine, sample_vendor):
		"""Test multi-objective optimization plan generation"""
		
		objectives = ['performance_improvement', 'cost_reduction', 'risk_mitigation']
		
		intelligence_engine._fetch_one = AsyncMock(return_value=sample_vendor.model_dump())
		intelligence_engine._fetch_all = AsyncMock(return_value=[])
		
		optimization_plan = await intelligence_engine.generate_optimization_plan(
			sample_vendor.id,
			objectives
		)
		
		assert len(optimization_plan.optimization_objectives) == 3
		assert len(optimization_plan.recommended_actions) > 0
		
		# Should balance different objectives
		action_types = [action.action_type for action in optimization_plan.recommended_actions]
		assert len(set(action_types)) > 1  # Multiple types of actions
	
	async def test_optimization_outcome_prediction(self, intelligence_engine, sample_vendor):
		"""Test optimization outcome prediction"""
		
		objectives = ['performance_improvement']
		
		intelligence_engine._fetch_one = AsyncMock(return_value=sample_vendor.model_dump())
		intelligence_engine._fetch_all = AsyncMock(return_value=[])
		
		optimization_plan = await intelligence_engine.generate_optimization_plan(
			sample_vendor.id,
			objectives
		)
		
		assert optimization_plan.predicted_outcomes is not None
		assert isinstance(optimization_plan.predicted_outcomes, dict)
		
		# Should predict improvement metrics
		if 'performance_improvement' in optimization_plan.predicted_outcomes:
			improvement = optimization_plan.predicted_outcomes['performance_improvement']
			assert isinstance(improvement, (int, float))
			assert improvement > 0  # Should predict positive improvement
	
	@pytest.mark.ai
	async def test_optimization_with_machine_learning(self, intelligence_engine, mock_ai_models):
		"""Test optimization using machine learning models"""
		
		intelligence_engine.ai_models = mock_ai_models
		
		vendor_id = str(uuid4())
		objectives = ['performance_improvement']
		
		intelligence_engine._fetch_one = AsyncMock(return_value={
			'id': vendor_id,
			'performance_score': 75.0
		})
		intelligence_engine._fetch_all = AsyncMock(return_value=[])
		
		optimization_plan = await intelligence_engine.generate_optimization_plan(
			vendor_id,
			objectives
		)
		
		# Should use ML models for optimization
		mock_ai_models['optimization_engine'].optimize.assert_called()
		
		# Should incorporate ML recommendations
		assert len(optimization_plan.recommended_actions) > 0


# ============================================================================
# INTELLIGENCE PERSISTENCE TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestIntelligencePersistence:
	"""Test intelligence data persistence and retrieval"""
	
	async def test_save_intelligence_record(self, intelligence_engine, sample_intelligence):
		"""Test saving intelligence record to database"""
		
		intelligence_engine._execute_query = AsyncMock()
		intelligence_engine._fetch_one = AsyncMock(return_value={'id': sample_intelligence.id})
		
		saved_intelligence = await intelligence_engine._save_intelligence_record(sample_intelligence)
		
		assert saved_intelligence.id == sample_intelligence.id
		intelligence_engine._execute_query.assert_called_once()
	
	async def test_get_latest_intelligence(self, intelligence_engine, sample_intelligence):
		"""Test retrieving latest intelligence record"""
		
		intelligence_data = sample_intelligence.model_dump()
		intelligence_engine._fetch_one = AsyncMock(return_value=intelligence_data)
		
		latest_intelligence = await intelligence_engine.get_latest_vendor_intelligence(
			sample_intelligence.vendor_id
		)
		
		assert latest_intelligence.id == sample_intelligence.id
		assert latest_intelligence.vendor_id == sample_intelligence.vendor_id
	
	async def test_intelligence_versioning(self, intelligence_engine, sample_vendor):
		"""Test intelligence record versioning"""
		
		# Mock multiple intelligence records
		old_intelligence = {
			'id': 'intel-old',
			'vendor_id': sample_vendor.id,
			'model_version': 'v0.9',
			'intelligence_date': datetime.now() - timedelta(days=30)
		}
		
		new_intelligence = {
			'id': 'intel-new',
			'vendor_id': sample_vendor.id,
			'model_version': 'v1.0',
			'intelligence_date': datetime.now()
		}
		
		intelligence_engine._fetch_one = AsyncMock(return_value=new_intelligence)
		
		latest = await intelligence_engine.get_latest_vendor_intelligence(sample_vendor.id)
		
		# Should return the most recent version
		assert latest.id == 'intel-new'
		assert latest.model_version == 'v1.0'


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestIntelligenceErrorHandling:
	"""Test intelligence service error handling"""
	
	async def test_invalid_vendor_id_handling(self, intelligence_engine):
		"""Test handling of invalid vendor IDs"""
		
		with pytest.raises(ValueError, match="Invalid vendor ID"):
			await intelligence_engine.analyze_vendor_behavior_patterns('invalid-uuid')
	
	async def test_missing_vendor_data_handling(self, intelligence_engine):
		"""Test handling of missing vendor data"""
		
		intelligence_engine._fetch_one = AsyncMock(return_value=None)
		
		with pytest.raises(ValueError, match="Vendor not found"):
			await intelligence_engine.generate_predictive_insights('non-existent-vendor')
	
	async def test_insufficient_data_handling(self, intelligence_engine, sample_vendor):
		"""Test handling of insufficient data for analysis"""
		
		# Mock empty data sets
		intelligence_engine._fetch_all = AsyncMock(return_value=[])
		intelligence_engine._fetch_one = AsyncMock(return_value=sample_vendor.model_dump())
		
		patterns = await intelligence_engine.analyze_vendor_behavior_patterns(sample_vendor.id)
		
		# Should return patterns with low confidence or default patterns
		assert isinstance(patterns, list)
		
		for pattern in patterns:
			assert 0.0 <= pattern.confidence <= 1.0
	
	async def test_database_error_handling(self, intelligence_engine, sample_vendor):
		"""Test handling of database errors"""
		
		intelligence_engine._fetch_all = AsyncMock(
			side_effect=Exception("Database connection failed")
		)
		
		with pytest.raises(Exception, match="Database connection failed"):
			await intelligence_engine.analyze_vendor_behavior_patterns(sample_vendor.id)
	
	@pytest.mark.ai
	async def test_ai_model_error_handling(self, intelligence_engine, mock_ai_models):
		"""Test handling of AI model errors"""
		
		# Mock AI model failure
		mock_ai_models['performance_predictor'].predict.side_effect = Exception("Model error")
		intelligence_engine.ai_models = mock_ai_models
		
		intelligence_engine._fetch_all = AsyncMock(return_value=[
			{'overall_score': 85.0}
		])
		
		# Should handle AI model errors gracefully
		insights = await intelligence_engine.generate_predictive_insights(str(uuid4()))
		
		# Should fall back to statistical methods
		assert isinstance(insights, list)