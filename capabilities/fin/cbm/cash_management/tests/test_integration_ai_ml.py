"""APG Cash Management - Integration Tests for AI/ML Components

Comprehensive integration tests for AI forecasting, ML models, and optimization.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import pytest
import pytest_asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from ..ai_forecasting import AIForecastingEngine
from ..advanced_ml_models import AdvancedMLModelManager, ModelType
from ..intelligent_optimization import (
	IntelligentCashFlowOptimizer, 
	OptimizationObjective, 
	OptimizationMethod,
	OptimizationConstraint,
	ConstraintType
)
from ..advanced_risk_analytics import AdvancedRiskAnalyticsEngine

# ============================================================================
# AI Forecasting Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.ml
class TestAIForecastingIntegration:
	"""Integration tests for AI forecasting engine."""
	
	async def test_end_to_end_forecasting_pipeline(
		self, 
		ai_forecasting, 
		sample_cash_flows, 
		time_series_generator
	):
		"""Test complete forecasting pipeline from data to predictions."""
		
		# Generate realistic time series data
		ts_data = time_series_generator(
			length=100, 
			trend=0.01, 
			seasonality=True, 
			noise_level=0.1
		)
		
		# Convert to cash flow format
		cash_flows = []
		for _, row in ts_data.iterrows():
			cash_flows.append({
				'date': row['date'],
				'amount': Decimal(str(round(row['value'] * 1000, 2))),
				'account_id': 'ACC001'
			})
		
		# Test forecast generation
		forecast_result = await ai_forecasting.generate_cash_flow_forecast(
			account_id='ACC001',
			forecast_horizon=30,
			confidence_level=0.95,
			historical_data=cash_flows
		)
		
		# Assertions
		assert forecast_result['success'] is True
		assert len(forecast_result['predictions']) == 30
		assert 'confidence_intervals' in forecast_result
		assert 'model_performance' in forecast_result
		
		# Test forecast quality
		predictions = forecast_result['predictions']
		assert all(isinstance(p, (int, float)) for p in predictions)
		assert not any(np.isnan(predictions))
		
		# Test confidence intervals
		ci_lower = forecast_result['confidence_intervals']['lower']
		ci_upper = forecast_result['confidence_intervals']['upper']
		
		assert len(ci_lower) == len(predictions)
		assert len(ci_upper) == len(predictions)
		assert all(l <= p <= u for l, p, u in zip(ci_lower, predictions, ci_upper))
	
	async def test_scenario_analysis_integration(self, ai_forecasting):
		"""Test scenario analysis with multiple economic conditions."""
		
		scenarios = {
			'base_case': {'growth_rate': 0.02, 'volatility': 0.15},
			'recession': {'growth_rate': -0.05, 'volatility': 0.25},
			'expansion': {'growth_rate': 0.08, 'volatility': 0.10}
		}
		
		# Mock historical data
		base_amount = 10000
		historical_data = [
			{
				'date': datetime.now() - timedelta(days=i),
				'amount': Decimal(str(base_amount + np.random.normal(0, 1000))),
				'account_id': 'ACC001'
			}
			for i in range(90, 0, -1)  # 90 days of history
		]
		
		scenario_results = await ai_forecasting.run_scenario_analysis(
			account_id='ACC001',
			scenarios=scenarios,
			forecast_horizon=30,
			historical_data=historical_data
		)
		
		# Assertions
		assert len(scenario_results) == 3
		assert all(scenario in scenario_results for scenario in scenarios.keys())
		
		# Test scenario differentiation
		base_forecast = scenario_results['base_case']['mean_forecast']
		recession_forecast = scenario_results['recession']['mean_forecast']
		expansion_forecast = scenario_results['expansion']['mean_forecast']
		
		# Recession should be lower than base case
		assert np.mean(recession_forecast) < np.mean(base_forecast)
		
		# Expansion should be higher than base case
		assert np.mean(expansion_forecast) > np.mean(base_forecast)
	
	async def test_model_performance_tracking(self, ai_forecasting):
		"""Test model performance tracking and validation."""
		
		# Generate synthetic data with known pattern
		dates = pd.date_range(start='2024-01-01', periods=120, freq='D')
		true_values = 1000 + 100 * np.sin(2 * np.pi * np.arange(120) / 30) + np.random.normal(0, 50, 120)
		
		historical_data = [
			{
				'date': date,
				'amount': Decimal(str(round(value, 2))),
				'account_id': 'ACC001'
			}
			for date, value in zip(dates[:90], true_values[:90])
		]
		
		# Generate forecast
		forecast_result = await ai_forecasting.generate_cash_flow_forecast(
			account_id='ACC001',
			forecast_horizon=30,
			historical_data=historical_data
		)
		
		# Test against known future values
		actual_values = true_values[90:120]
		predicted_values = forecast_result['predictions']
		
		# Calculate performance metrics
		mae = np.mean(np.abs(np.array(predicted_values) - actual_values))
		mape = np.mean(np.abs((np.array(predicted_values) - actual_values) / actual_values)) * 100
		
		# Performance should be reasonable for synthetic data
		assert mae < 200  # Mean absolute error should be reasonable
		assert mape < 20   # MAPE should be under 20%
		
		# Test model metadata
		assert 'model_used' in forecast_result['model_performance']
		assert 'training_metrics' in forecast_result['model_performance']

# ============================================================================
# ML Models Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.ml
@pytest.mark.slow
class TestMLModelsIntegration:
	"""Integration tests for advanced ML models."""
	
	async def test_model_training_pipeline(self, ml_manager, time_series_generator):
		"""Test complete model training pipeline."""
		
		# Generate training data
		training_data = time_series_generator(
			length=500,  # Larger dataset for training
			trend=0.005,
			seasonality=True,
			noise_level=0.08
		)
		
		# Add target column
		training_data['amount'] = training_data['value'] * 1000
		
		# Test model training
		performances = await ml_manager.train_all_models(
			training_data=training_data,
			target_column='amount',
			validation_split=0.2
		)
		
		# Assertions
		assert len(performances) > 5  # Should have multiple models
		assert all(perf.success for perf in performances.values())
		
		# Test model diversity
		model_types = [perf.model_type for perf in performances.values()]
		assert ModelType.XGBOOST in model_types
		assert ModelType.RANDOM_FOREST in model_types
		assert ModelType.ELASTIC_NET in model_types
		
		# Test performance metrics
		for perf in performances.values():
			assert 'r2' in perf.validation_metrics
			assert 'rmse' in perf.validation_metrics
			assert perf.validation_metrics['r2'] >= -1.0  # R² should be reasonable
			assert perf.training_time > 0
	
	async def test_ensemble_model_creation(self, ml_manager, time_series_generator):
		"""Test ensemble model creation and performance."""
		
		# Generate training data
		training_data = time_series_generator(length=300)
		training_data['amount'] = training_data['value'] * 1000 + np.random.normal(0, 100, 300)
		
		# Train models (mocked for speed)
		with patch.object(ml_manager, 'train_all_models') as mock_train:
			# Mock successful training results
			mock_performances = {
				'xgboost': AsyncMock(
					model_name='xgboost',
					validation_metrics={'r2': 0.85, 'rmse': 150},
					success=True
				),
				'random_forest': AsyncMock(
					model_name='random_forest',
					validation_metrics={'r2': 0.82, 'rmse': 160},
					success=True
				),
				'elastic_net': AsyncMock(
					model_name='elastic_net',
					validation_metrics={'r2': 0.78, 'rmse': 170},
					success=True
				)
			}
			mock_train.return_value = mock_performances
			
			performances = await ml_manager.train_all_models(training_data)
			
			# Test ensemble creation
			assert 'xgboost' in performances
			assert 'random_forest' in performances
			assert 'elastic_net' in performances
	
	async def test_model_prediction_with_uncertainty(self, ml_manager, time_series_generator):
		"""Test model predictions with uncertainty quantification."""
		
		# Mock trained models
		ml_manager.models = {
			'model1': AsyncMock(),
			'model2': AsyncMock(),
			'model3': AsyncMock()
		}
		
		# Mock model predictions
		ml_manager.models['model1'].predict = AsyncMock(return_value=np.array([1000, 1100, 1200]))
		ml_manager.models['model2'].predict = AsyncMock(return_value=np.array([1050, 1150, 1250]))
		ml_manager.models['model3'].predict = AsyncMock(return_value=np.array([980, 1080, 1180]))
		
		# Generate test data
		test_data = time_series_generator(length=30)
		
		# Mock feature engineering
		with patch.object(ml_manager.feature_engineer, 'engineer_features') as mock_features:
			mock_features.return_value = (test_data[['value']], pd.Series([1000] * 30))
			
			# Test prediction with uncertainty
			forecast_result = await ml_manager.predict_with_uncertainty(
				data=test_data,
				model_names=['model1', 'model2', 'model3'],
				return_intervals=True
			)
			
			# Assertions
			assert len(forecast_result.predictions) == 3
			assert len(forecast_result.confidence_intervals[0]) == 3
			assert len(forecast_result.confidence_intervals[1]) == 3
			assert forecast_result.epistemic_uncertainty is not None
			assert len(forecast_result.model_weights) == 3
	
	async def test_model_insights_generation(self, ml_manager):
		"""Test model insights and interpretation."""
		
		# Mock model performances
		ml_manager.model_performances = {
			'xgboost': AsyncMock(
				model_name='xgboost',
				validation_metrics={'r2': 0.88, 'rmse': 120},
				training_time=45.2,
				stability_score=0.95,
				feature_importance={'feature1': 0.4, 'feature2': 0.3, 'feature3': 0.3}
			),
			'random_forest': AsyncMock(
				model_name='random_forest',
				validation_metrics={'r2': 0.85, 'rmse': 130},
				training_time=32.1,
				stability_score=0.92,
				feature_importance={'feature1': 0.35, 'feature2': 0.35, 'feature3': 0.3}
			)
		}
		
		insights = await ml_manager.get_model_insights()
		
		# Assertions
		assert 'performance_summary' in insights
		assert 'best_models' in insights
		assert 'top_features' in insights
		assert 'model_count' in insights
		
		assert insights['model_count'] == 2
		assert insights['best_models']['accuracy'] == 'xgboost'
		assert insights['best_models']['speed'] == 'random_forest'

# ============================================================================
# Optimization Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.optimization
class TestOptimizationIntegration:
	"""Integration tests for intelligent optimization."""
	
	async def test_end_to_end_optimization(
		self, 
		optimization_engine, 
		sample_cash_accounts, 
		portfolio_data_generator
	):
		"""Test complete optimization pipeline."""
		
		# Generate portfolio data
		portfolio = portfolio_data_generator(num_accounts=5, total_value=1000000)
		
		# Define optimization objectives
		objectives = [OptimizationObjective.MAXIMIZE_YIELD, OptimizationObjective.MINIMIZE_RISK]
		
		# Define constraints
		constraints = [
			OptimizationConstraint(
				name="balance_conservation",
				constraint_type=ConstraintType.BALANCE_REQUIREMENT,
				target_value=1000000.0,
				is_hard_constraint=True
			),
			OptimizationConstraint(
				name="concentration_limit",
				constraint_type=ConstraintType.CONCENTRATION_LIMIT,
				upper_bound=0.4,
				is_hard_constraint=False
			)
		]
		
		# Run optimization
		result = await optimization_engine.optimize_cash_allocation(
			accounts=sample_cash_accounts,
			objectives=objectives,
			constraints=constraints,
			optimization_horizon=30,
			method=OptimizationMethod.MULTI_OBJECTIVE
		)
		
		# Assertions
		assert result.success is True
		assert result.objective_value >= 0
		assert len(result.optimal_solution) == len(sample_cash_accounts)
		assert result.confidence_score > 0.5
		
		# Test constraint satisfaction
		total_allocation = sum(result.optimal_solution.values())
		assert abs(total_allocation - 1000000.0) < 10000  # Within tolerance
		
		# Test recommendations
		assert len(result.recommendations) > 0
		assert all(isinstance(rec, str) for rec in result.recommendations)
	
	async def test_optimization_methods_comparison(
		self, 
		optimization_engine, 
		sample_cash_accounts
	):
		"""Test different optimization methods."""
		
		objectives = [OptimizationObjective.MAXIMIZE_YIELD]
		constraints = [
			OptimizationConstraint(
				name="balance_conservation",
				constraint_type=ConstraintType.BALANCE_REQUIREMENT,
				target_value=850000.0,  # Total from sample accounts
				is_hard_constraint=True
			)
		]
		
		methods = [
			OptimizationMethod.LINEAR_PROGRAMMING,
			OptimizationMethod.GENETIC_ALGORITHM,
			OptimizationMethod.DIFFERENTIAL_EVOLUTION
		]
		
		results = {}
		for method in methods:
			result = await optimization_engine.optimize_cash_allocation(
				accounts=sample_cash_accounts,
				objectives=objectives,
				constraints=constraints,
				method=method
			)
			results[method] = result
		
		# All methods should produce valid results
		for method, result in results.items():
			assert result.success is True, f"Method {method} failed"
			assert result.objective_value >= 0
			assert result.method_used == method
		
		# Results should be reasonably close
		objective_values = [r.objective_value for r in results.values()]
		max_diff = max(objective_values) - min(objective_values)
		avg_value = sum(objective_values) / len(objective_values)
		
		# Relative difference should be reasonable
		if avg_value > 0:
			relative_diff = max_diff / avg_value
			assert relative_diff < 0.5  # Within 50% of each other
	
	async def test_cash_allocation_recommendations(
		self, 
		optimization_engine, 
		sample_cash_accounts
	):
		"""Test cash allocation recommendation generation."""
		
		# Mock optimization result
		optimization_result = AsyncMock()
		optimization_result.success = True
		optimization_result.optimal_solution = {
			'ACC001': 120000.0,  # Increase from 100k
			'ACC002': 230000.0,  # Decrease from 250k
			'ACC003': 500000.0   # Same as current
		}
		
		# Generate recommendations
		recommendations = await optimization_engine.generate_cash_allocation_recommendations(
			accounts=sample_cash_accounts,
			optimization_result=optimization_result
		)
		
		# Assertions
		assert len(recommendations) == 3
		
		# Check recommendation types
		actions = [rec.recommended_action for rec in recommendations]
		assert 'transfer_in' in actions   # ACC001 should increase
		assert 'transfer_out' in actions  # ACC002 should decrease
		assert 'maintain' in actions      # ACC003 should stay same
		
		# Check amounts
		for rec in recommendations:
			assert rec.amount >= 0
			assert rec.priority >= 1
			assert rec.rationale is not None
			assert rec.expected_yield >= 0
			assert rec.risk_score >= 0

# ============================================================================
# Risk Analytics Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.risk
class TestRiskAnalyticsIntegration:
	"""Integration tests for risk analytics engine."""
	
	async def test_comprehensive_risk_calculation(
		self, 
		risk_analytics, 
		sample_returns_data, 
		portfolio_data_generator
	):
		"""Test comprehensive risk metrics calculation."""
		
		# Generate portfolio data
		portfolio = portfolio_data_generator(num_accounts=3, total_value=1000000)
		
		# Calculate risk metrics
		risk_metrics = await risk_analytics.calculate_comprehensive_risk_metrics(
			portfolio_data=portfolio,
			returns_data=sample_returns_data,
			confidence_levels=[0.95, 0.99],
			holding_periods=[1, 10]
		)
		
		# Assertions
		assert 'value_at_risk' in risk_metrics
		assert 'expected_shortfall' in risk_metrics
		assert 'descriptive_statistics' in risk_metrics
		assert 'performance_ratios' in risk_metrics
		assert 'liquidity_risk' in risk_metrics
		
		# Test VaR calculations
		var_data = risk_metrics['value_at_risk']
		assert 'var_95_1d' in var_data
		assert 'var_99_1d' in var_data
		assert 'var_95_10d' in var_data
		
		# Test VaR values are positive and reasonable
		for var_key, var_results in var_data.items():
			for method, result in var_results.items():
				if isinstance(result, dict) and 'value' in result:
					assert result['value'] >= 0
					assert result['value'] < 1.0  # Should be less than 100%
		
		# Test Expected Shortfall
		es_data = risk_metrics['expected_shortfall']
		for es_key, es_results in es_data.items():
			for method, result in es_results.items():
				if isinstance(result, dict) and 'value' in result:
					assert result['value'] >= 0
		
		# Test performance ratios
		perf_ratios = risk_metrics['performance_ratios']
		assert 'sharpe_ratio' in perf_ratios
		assert 'sortino_ratio' in perf_ratios
		assert 'max_drawdown' in perf_ratios
		assert isinstance(perf_ratios['annual_volatility'], float)
	
	async def test_stress_testing_suite(self, risk_analytics, portfolio_data_generator):
		"""Test comprehensive stress testing."""
		
		# Generate portfolio data
		portfolio = portfolio_data_generator(num_accounts=4, total_value=2000000)
		
		# Run stress tests
		stress_results = await risk_analytics.run_comprehensive_stress_tests(
			portfolio_data=portfolio
		)
		
		# Assertions
		assert 'historical_scenarios' in stress_results
		assert 'monte_carlo' in stress_results
		assert 'liquidity_stress' in stress_results
		assert 'summary' in stress_results
		
		# Test historical scenarios
		historical = stress_results['historical_scenarios']
		assert '2008_financial_crisis' in historical
		assert 'covid_pandemic' in historical
		
		for scenario_name, result in historical.items():
			assert result.loss_amount >= 0
			assert result.loss_percentage >= 0
			assert len(result.affected_accounts) >= 0
			assert result.recovery_time_days > 0
			assert len(result.mitigation_actions) > 0
		
		# Test Monte Carlo results
		mc_results = stress_results['monte_carlo']
		assert 0.95 in mc_results
		assert 0.99 in mc_results
		
		# Higher confidence level should show higher losses
		loss_95 = mc_results[0.95].loss_amount
		loss_99 = mc_results[0.99].loss_amount
		assert loss_99 >= loss_95
		
		# Test summary
		summary = stress_results['summary']
		assert 'worst_case_loss' in summary
		assert 'scenarios_tested' in summary
		assert summary['scenarios_tested'] > 0
	
	async def test_liquidity_risk_analysis(self, risk_analytics, portfolio_data_generator):
		"""Test liquidity risk analysis."""
		
		# Generate portfolio with mixed liquidity
		portfolio = {
			'ACC001': {'balance': 500000, 'type': 'checking', 'liquidity_score': 1.0},
			'ACC002': {'balance': 300000, 'type': 'savings', 'liquidity_score': 0.95},
			'ACC003': {'balance': 200000, 'type': 'investment', 'liquidity_score': 0.7}
		}
		
		# Analyze liquidity risk
		liquidity_risk = await risk_analytics.liquidity_analyzer.assess_liquidity_risk(
			portfolio_data=portfolio
		)
		
		# Assertions
		assert liquidity_risk.total_liquid_assets > 0
		assert liquidity_risk.liquidity_coverage_ratio >= 0
		assert liquidity_risk.net_stable_funding_ratio >= 0
		assert isinstance(liquidity_risk.funding_concentration, dict)
		assert isinstance(liquidity_risk.liquidity_buffers, dict)
		assert liquidity_risk.stress_test_survival_days > 0
		
		# Test funding concentration metrics
		concentration = liquidity_risk.funding_concentration
		if concentration:
			assert 'hhi' in concentration
			assert 'largest_source' in concentration
			assert concentration['hhi'] >= 0
			assert concentration['largest_source'] <= 1.0
	
	async def test_risk_dashboard_integration(self, risk_analytics):
		"""Test risk dashboard data aggregation."""
		
		# Mock some risk data
		with patch.object(risk_analytics.cache, 'get') as mock_cache_get:
			mock_risk_data = {
				'risk_metrics': {
					'value_at_risk': {'var_95_1d': {'historical': {'value': 0.05}}}
				},
				'calculation_timestamp': datetime.now().isoformat()
			}
			mock_stress_data = {
				'stress_test_results': {
					'summary': {'worst_case_loss': 50000, 'scenarios_tested': 5}
				}
			}
			
			mock_cache_get.side_effect = [mock_risk_data, mock_stress_data]
			
			# Get dashboard data
			dashboard_data = await risk_analytics.get_risk_dashboard_data()
			
			# Assertions
			assert 'risk_metrics' in dashboard_data
			assert 'stress_tests' in dashboard_data
			assert 'recent_alerts' in dashboard_data
			assert 'risk_summary' in dashboard_data
			
			risk_summary = dashboard_data['risk_summary']
			assert 'total_alerts' in risk_summary
			assert 'critical_alerts' in risk_summary
			assert 'risk_score' in risk_summary
			
			# Risk score should be reasonable
			assert 0 <= risk_summary['risk_score'] <= 100

# ============================================================================
# Cross-Component Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestCrossComponentIntegration:
	"""Integration tests across multiple AI/ML components."""
	
	async def test_forecasting_to_optimization_pipeline(
		self, 
		ai_forecasting, 
		optimization_engine,
		sample_cash_accounts,
		time_series_generator
	):
		"""Test pipeline from forecasting to optimization."""
		
		# Step 1: Generate forecasts
		ts_data = time_series_generator(length=90)
		cash_flows = [
			{
				'date': row['date'],
				'amount': Decimal(str(row['value'] * 1000)),
				'account_id': 'ACC001'
			}
			for _, row in ts_data.iterrows()
		]
		
		forecast_result = await ai_forecasting.generate_cash_flow_forecast(
			account_id='ACC001',
			forecast_horizon=30,
			historical_data=cash_flows
		)
		
		# Step 2: Use forecasts in optimization
		# Mock optimization data preparation
		with patch.object(
			optimization_engine, 
			'_generate_cash_flow_forecasts',
			return_value={'ACC001': forecast_result['predictions']}
		):
			
			objectives = [OptimizationObjective.MAXIMIZE_YIELD]
			constraints = [
				OptimizationConstraint(
					name="balance_conservation",
					constraint_type=ConstraintType.BALANCE_REQUIREMENT,
					target_value=850000.0
				)
			]
			
			optimization_result = await optimization_engine.optimize_cash_allocation(
				accounts=sample_cash_accounts,
				objectives=objectives,
				constraints=constraints
			)
		
		# Assertions
		assert forecast_result['success'] is True
		assert optimization_result.success is True
		
		# Optimization should use forecast insights
		assert len(optimization_result.optimal_solution) == len(sample_cash_accounts)
	
	async def test_ml_to_risk_analytics_pipeline(
		self, 
		ml_manager, 
		risk_analytics,
		sample_returns_data,
		time_series_generator
	):
		"""Test pipeline from ML predictions to risk analytics."""
		
		# Step 1: Generate ML predictions
		test_data = time_series_generator(length=30)
		
		# Mock ML prediction
		with patch.object(ml_manager, 'predict_with_uncertainty') as mock_predict:
			mock_forecast_result = AsyncMock()
			mock_forecast_result.predictions = sample_returns_data[:30]
			mock_forecast_result.epistemic_uncertainty = np.abs(sample_returns_data[:30]) * 0.1
			mock_predict.return_value = mock_forecast_result
			
			ml_predictions = await ml_manager.predict_with_uncertainty(test_data)
		
		# Step 2: Use predictions in risk analytics
		# Convert predictions to returns format
		predicted_returns = ml_predictions.predictions
		
		# Calculate risk metrics using predicted returns
		portfolio = {'ACC001': {'balance': 1000000, 'type': 'investment'}}
		
		risk_metrics = await risk_analytics.calculate_comprehensive_risk_metrics(
			portfolio_data=portfolio,
			returns_data=predicted_returns
		)
		
		# Assertions
		assert ml_predictions.predictions is not None
		assert len(ml_predictions.predictions) == 30
		assert 'value_at_risk' in risk_metrics
		
		# Risk metrics should incorporate ML uncertainty
		var_data = risk_metrics['value_at_risk']
		assert len(var_data) > 0
	
	async def test_optimization_to_risk_feedback_loop(
		self, 
		optimization_engine, 
		risk_analytics,
		sample_cash_accounts,
		portfolio_data_generator
	):
		"""Test feedback loop from optimization to risk validation."""
		
		# Step 1: Run optimization
		portfolio = portfolio_data_generator(num_accounts=3)
		
		objectives = [OptimizationObjective.MAXIMIZE_YIELD]
		constraints = []
		
		optimization_result = await optimization_engine.optimize_cash_allocation(
			accounts=sample_cash_accounts,
			objectives=objectives,
			constraints=constraints
		)
		
		# Step 2: Validate optimization with risk analytics
		# Create portfolio based on optimization results
		optimized_portfolio = {}
		for i, (account_id, balance) in enumerate(optimization_result.optimal_solution.items()):
			optimized_portfolio[account_id] = {
				'balance': balance,
				'type': sample_cash_accounts[i]['account_type'],
				'liquidity_score': 0.9
			}
		
		# Generate synthetic returns for risk analysis
		np.random.seed(42)
		synthetic_returns = np.random.normal(0.001, 0.02, 100)
		
		risk_metrics = await risk_analytics.calculate_comprehensive_risk_metrics(
			portfolio_data=optimized_portfolio,
			returns_data=synthetic_returns
		)
		
		# Step 3: Validate risk is within acceptable bounds
		var_95 = None
		var_data = risk_metrics.get('value_at_risk', {})
		if 'var_95_1d' in var_data and 'historical' in var_data['var_95_1d']:
			var_95 = var_data['var_95_1d']['historical'].get('value', 0)
		
		# Assertions
		assert optimization_result.success is True
		assert var_95 is not None
		assert var_95 >= 0
		
		# Risk should be reasonable for optimized portfolio
		if var_95 > 0:
			assert var_95 < 0.1  # VaR should be less than 10%

if __name__ == "__main__":
	pytest.main([__file__, "-v"])