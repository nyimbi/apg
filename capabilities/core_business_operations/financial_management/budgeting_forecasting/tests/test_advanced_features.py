"""
APG Budgeting & Forecasting - Advanced Features Tests

Focused tests for advanced features including real-time collaboration,
AI recommendations, ML forecasting, and automated monitoring.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
import json

from ..service import APGTenantContext, BFServiceConfig
from .. import create_budgeting_forecasting_capability
from ..realtime_collaboration import (
	CollaborationEventType, UserPresenceStatus, ConflictResolutionStrategy,
	create_realtime_collaboration_service
)
from ..ai_budget_recommendations import (
	RecommendationType, RecommendationPriority, ConfidenceLevel,
	create_ai_budget_recommendations_service
)
from ..ml_forecasting_engine import (
	ForecastAlgorithm, ForecastHorizon, ModelStatus,
	create_ml_forecasting_engine_service
)
from ..automated_monitoring import (
	AlertType, AlertSeverity, MonitoringFrequency,
	create_automated_budget_monitoring_service
)


# =============================================================================
# Test Fixtures for Advanced Features
# =============================================================================

@pytest.fixture
def advanced_tenant_context():
	"""Create tenant context for advanced features testing."""
	return APGTenantContext(
		tenant_id="advanced_test_tenant",
		user_id="advanced_test_user"
	)

@pytest.fixture
def advanced_config():
	"""Create advanced configuration."""
	return BFServiceConfig(
		database_url="postgresql://test:test@localhost:5432/test_apg_bf_advanced",
		cache_enabled=True,
		audit_enabled=True,
		ml_enabled=True,
		ai_recommendations_enabled=True,
		real_time_collaboration_enabled=True,
		automated_monitoring_enabled=True
	)

@pytest.fixture
async def collaboration_service(advanced_tenant_context, advanced_config):
	"""Create real-time collaboration service."""
	return create_realtime_collaboration_service(advanced_tenant_context, advanced_config)

@pytest.fixture
async def ai_recommendations_service(advanced_tenant_context, advanced_config):
	"""Create AI recommendations service."""
	return create_ai_budget_recommendations_service(advanced_tenant_context, advanced_config)

@pytest.fixture
async def ml_forecasting_service(advanced_tenant_context, advanced_config):
	"""Create ML forecasting service."""
	return create_ml_forecasting_engine_service(advanced_tenant_context, advanced_config)

@pytest.fixture
async def monitoring_service(advanced_tenant_context, advanced_config):
	"""Create automated monitoring service."""
	return create_automated_budget_monitoring_service(advanced_tenant_context, advanced_config)


# =============================================================================
# Real-Time Collaboration Advanced Tests
# =============================================================================

class TestAdvancedCollaboration:
	"""Test advanced real-time collaboration features."""

	async def test_complex_collaboration_scenario(self, collaboration_service):
		"""Test complex multi-user collaboration scenario."""
		
		# Create collaboration session
		session_config = {
			"session_name": "Complex Collaboration Test",
			"budget_id": "budget_complex_001",
			"max_participants": 5,
			"session_type": "budget_editing",
			"advanced_features": {
				"conflict_resolution": ConflictResolutionStrategy.LAST_WRITER_WINS.value,
				"change_tracking": True,
				"comment_threading": True,
				"live_cursors": True
			}
		}
		
		response = await collaboration_service.create_collaboration_session(session_config)
		assert response.success
		session_id = response.data["session_id"]
		
		# Simulate multiple users joining
		users = [
			{"user_id": "user_001", "name": "Alice", "role": "editor"},
			{"user_id": "user_002", "name": "Bob", "role": "reviewer"},
			{"user_id": "user_003", "name": "Carol", "role": "approver"}
		]
		
		join_responses = []
		for user in users:
			join_config = {
				"user_id": user["user_id"],
				"user_name": user["name"],
				"role": user["role"],
				"permissions": ["edit", "comment"] if user["role"] == "editor" else ["comment"]
			}
			
			join_response = await collaboration_service.join_collaboration_session(session_id, join_config)
			assert join_response.success
			join_responses.append(join_response)
		
		# Simulate concurrent editing events
		edit_events = [
			{
				"event_type": CollaborationEventType.BUDGET_LINE_EDIT.value,
				"user_id": "user_001",
				"target_id": "bf_line_001",
				"changes": {"amount": 85000.00, "previous_amount": 80000.00},
				"timestamp": datetime.utcnow()
			},
			{
				"event_type": CollaborationEventType.COMMENT_ADDED.value,
				"user_id": "user_002",
				"target_id": "bf_line_001",
				"content": "This increase looks reasonable given market conditions",
				"timestamp": datetime.utcnow()
			}
		]
		
		for event in edit_events:
			event_response = await collaboration_service.send_collaboration_event(session_id, event)
			assert event_response.success

	async def test_conflict_resolution_strategies(self, collaboration_service):
		"""Test different conflict resolution strategies."""
		
		# Test optimistic locking strategy
		session_config_optimistic = {
			"session_name": "Optimistic Conflict Test",
			"budget_id": "budget_conflict_001",
			"conflict_resolution": ConflictResolutionStrategy.OPTIMISTIC_LOCKING.value
		}
		
		optimistic_response = await collaboration_service.create_collaboration_session(session_config_optimistic)
		assert optimistic_response.success
		
		# Test manual resolution strategy
		session_config_manual = {
			"session_name": "Manual Conflict Test",
			"budget_id": "budget_conflict_002",
			"conflict_resolution": ConflictResolutionStrategy.MANUAL_RESOLUTION.value
		}
		
		manual_response = await collaboration_service.create_collaboration_session(session_config_manual)
		assert manual_response.success

	async def test_change_request_workflow(self, collaboration_service):
		"""Test change request and approval workflow."""
		
		# Create session with change request capability
		session_config = {
			"session_name": "Change Request Test",
			"budget_id": "budget_cr_001",
			"change_request_enabled": True
		}
		
		session_response = await collaboration_service.create_collaboration_session(session_config)
		session_id = session_response.data["session_id"]
		
		# Create change request
		change_request = {
			"request_title": "Increase Marketing Budget",
			"request_description": "Requesting 20% increase for Q2 campaign",
			"requested_changes": [
				{
					"target_line": "bf_line_marketing_001",
					"change_type": "amount_update",
					"current_value": 100000.00,
					"proposed_value": 120000.00,
					"justification": "Expanded market opportunity"
				}
			],
			"priority": "high"
		}
		
		cr_response = await collaboration_service.create_change_request(session_id, change_request)
		assert cr_response.success
		assert "request_id" in cr_response.data


# =============================================================================
# AI Recommendations Advanced Tests
# =============================================================================

class TestAdvancedAIRecommendations:
	"""Test advanced AI recommendation features."""

	async def test_industry_benchmark_integration(self, ai_recommendations_service):
		"""Test AI recommendations with industry benchmark integration."""
		
		context_config = {
			"budget_id": "budget_ai_001",
			"analysis_period": "last_24_months",
			"industry": "SaaS Technology",
			"company_size": "mid_market",
			"business_model": "subscription",
			"growth_stage": "expansion",
			"strategic_goals": [
				"aggressive_growth",
				"market_expansion",
				"operational_efficiency"
			],
			"risk_tolerance": "medium_high",
			"include_external_benchmarks": True,
			"benchmark_sources": [
				"industry_reports",
				"peer_companies",
				"market_research"
			]
		}
		
		response = await ai_recommendations_service.generate_budget_recommendations(context_config)
		assert response.success
		assert "bundle_id" in response.data
		
		recommendations = response.data["recommendations"]
		assert len(recommendations) > 0
		
		# Verify recommendations include benchmark data
		for rec in recommendations:
			assert "benchmarks" in rec
			assert "supporting_metrics" in rec
			assert rec["confidence_level"] in [level.value for level in ConfidenceLevel]

	async def test_contextual_recommendation_generation(self, ai_recommendations_service):
		"""Test context-aware recommendation generation."""
		
		# Test recommendations for different business contexts
		contexts = [
			{
				"context_name": "startup_context",
				"business_stage": "startup",
				"funding_stage": "series_a",
				"focus": ["user_acquisition", "product_development"]
			},
			{
				"context_name": "enterprise_context", 
				"business_stage": "enterprise",
				"focus": ["cost_optimization", "compliance", "risk_management"]
			},
			{
				"context_name": "growth_context",
				"business_stage": "scale_up",
				"focus": ["market_expansion", "team_scaling", "infrastructure"]
			}
		]
		
		for context in contexts:
			context_config = {
				"budget_id": f"budget_{context['context_name']}",
				"analysis_period": "last_12_months",
				"business_context": context,
				"strategic_goals": context["focus"]
			}
			
			response = await ai_recommendations_service.generate_budget_recommendations(context_config)
			assert response.success
			
			# Verify recommendations are contextually appropriate
			recommendations = response.data["recommendations"]
			assert len(recommendations) > 0
			
			# Check that recommendations align with business stage
			for rec in recommendations:
				assert rec["category"] in [
					"revenue", "expenses", "capital_expenditure", 
					"operational_efficiency", "strategic_investment"
				]

	async def test_recommendation_learning_and_improvement(self, ai_recommendations_service):
		"""Test recommendation learning from feedback."""
		
		# Generate initial recommendations
		context_config = {
			"budget_id": "budget_learning_001",
			"analysis_period": "last_12_months",
			"industry": "Technology"
		}
		
		initial_response = await ai_recommendations_service.generate_budget_recommendations(context_config)
		assert initial_response.success
		
		recommendations = initial_response.data["recommendations"]
		recommendation_id = recommendations[0]["recommendation_id"]
		
		# Implement recommendation
		implementation_config = {
			"implementation_plan": "gradual",
			"implementation_timeline": "6_months",
			"monitoring_enabled": True
		}
		
		implement_response = await ai_recommendations_service.implement_recommendation(
			recommendation_id, implementation_config
		)
		assert implement_response.success
		
		# Simulate performance tracking over time
		performance_response = await ai_recommendations_service.track_recommendation_performance(recommendation_id)
		assert performance_response.success
		
		# Verify performance data includes learning metrics
		performance_data = performance_response.data
		assert "accuracy" in performance_data["performance_summary"]
		assert "lessons_learned" in performance_data
		assert "future_improvements" in performance_data

	async def test_custom_recommendation_templates(self, ai_recommendations_service):
		"""Test custom recommendation template creation and usage."""
		
		# Create custom template
		template_config = {
			"template_name": "Seasonal Budget Adjustment",
			"recommendation_type": RecommendationType.SEASONAL_ADJUSTMENT.value,
			"title_template": "Adjust {category} budget for {season} season",
			"description_template": "Based on historical patterns, recommend {adjustment_type} {category} budget by {percentage}% for {season}",
			"rationale_template": "Historical analysis shows {percentage}% {trend} in {category} spending during {season}",
			"trigger_conditions": [
				{
					"condition_type": "seasonal_variance",
					"threshold": 0.15,
					"metric": "category_spending"
				}
			],
			"impact_calculation": "base_amount * seasonal_factor * adjustment_percentage",
			"confidence_factors": [
				"historical_data_quality",
				"seasonal_pattern_consistency",
				"external_market_factors"
			]
		}
		
		template_response = await ai_recommendations_service.create_custom_recommendation_template(template_config)
		assert template_response.success
		assert "template_id" in template_response.data
		
		# Test template validation
		template_data = template_response.data
		assert template_data["template_name"] == "Seasonal Budget Adjustment"
		assert template_data["recommendation_type"] == RecommendationType.SEASONAL_ADJUSTMENT.value


# =============================================================================
# ML Forecasting Advanced Tests  
# =============================================================================

class TestAdvancedMLForecasting:
	"""Test advanced ML forecasting features."""

	async def test_multi_algorithm_model_ensemble(self, ml_forecasting_service):
		"""Test ensemble of multiple forecasting algorithms."""
		
		# Create individual models with different algorithms
		algorithms = [
			ForecastAlgorithm.RANDOM_FOREST,
			ForecastAlgorithm.GRADIENT_BOOSTING,
			ForecastAlgorithm.LSTM_NEURAL_NETWORK
		]
		
		model_ids = []
		for i, algorithm in enumerate(algorithms):
			model_config = {
				"model_name": f"Ensemble Model {i+1} - {algorithm.value}",
				"algorithm": algorithm.value,
				"target_variable": "budget_amount",
				"horizon": ForecastHorizon.MEDIUM_TERM.value,
				"frequency": "monthly",
				"training_window": 24,
				"features": [
					{
						"feature_name": "historical_budget",
						"feature_type": "historical_values",
						"source_column": "budget_amount",
						"lag_periods": [1, 3, 6, 12]
					},
					{
						"feature_name": "seasonal_indicators",
						"feature_type": "categorical",
						"source_column": "month",
						"encoding": "one_hot"
					}
				],
				"hyperparameters": {
					"validation_split": 0.2,
					"early_stopping": True,
					"max_iterations": 1000
				}
			}
			
			model_response = await ml_forecasting_service.create_forecasting_model(model_config)
			assert model_response.success
			model_ids.append(model_response.data["model_id"])
		
		# Create ensemble model
		ensemble_config = {
			"ensemble_name": "Multi-Algorithm Budget Forecast Ensemble",
			"base_models": model_ids,
			"ensemble_method": "stacked_generalization",
			"meta_learner": "linear_regression",
			"weights": "auto",  # Automatically determine weights
			"cross_validation_folds": 5,
			"performance_metrics": ["mae", "rmse", "mape"],
			"meta_features": [
				"model_confidence",
				"prediction_variance",
				"feature_importance"
			]
		}
		
		ensemble_response = await ml_forecasting_service.create_model_ensemble(ensemble_config)
		assert ensemble_response.success
		assert "ensemble_id" in ensemble_response.data
		
		# Train ensemble
		training_config = {
			"ensemble_training": True,
			"base_model_training": True,
			"meta_learner_training": True,
			"validation_strategy": "time_series_split"
		}
		
		ensemble_id = ensemble_response.data["ensemble_id"]
		training_response = await ml_forecasting_service.train_forecasting_model(ensemble_id, training_config)
		assert training_response.success

	async def test_advanced_feature_engineering(self, ml_forecasting_service):
		"""Test advanced feature engineering capabilities."""
		
		model_config = {
			"model_name": "Advanced Feature Engineering Model",
			"algorithm": ForecastAlgorithm.GRADIENT_BOOSTING.value,
			"target_variable": "budget_amount",
			"horizon": ForecastHorizon.LONG_TERM.value,
			"frequency": "monthly",
			"training_window": 36,
			"advanced_features": {
				"lag_features": {
					"target_lags": [1, 3, 6, 12, 24],
					"rolling_statistics": {
						"windows": [3, 6, 12],
						"statistics": ["mean", "std", "min", "max", "median"]
					}
				},
				"seasonal_features": {
					"seasonal_decomposition": True,
					"fourier_terms": 4,
					"holiday_indicators": True,
					"business_calendar": True
				},
				"external_features": {
					"economic_indicators": [
						"gdp_growth",
						"inflation_rate",
						"unemployment_rate"
					],
					"industry_metrics": [
						"industry_growth",
						"market_volatility"
					]
				},
				"interaction_features": {
					"polynomial_degree": 2,
					"feature_crosses": [
						["seasonal", "economic"],
						["lag_features", "external"]
					]
				}
			},
			"feature_selection": {
				"method": "recursive_feature_elimination",
				"max_features": 50,
				"importance_threshold": 0.01
			}
		}
		
		response = await ml_forecasting_service.create_forecasting_model(model_config)
		assert response.success
		assert "model_id" in response.data
		
		# Verify advanced features are properly configured
		model_data = response.data
		assert "advanced_features" in model_data
		assert model_data["feature_count"] > 20  # Should have many engineered features

	async def test_scenario_based_forecasting(self, ml_forecasting_service):
		"""Test scenario-based forecasting with multiple assumptions."""
		
		model_id = "trained_model_001"  # Assume pre-trained model
		
		# Define multiple forecast scenarios
		scenarios = [
			{
				"scenario_name": "Optimistic Growth",
				"description": "Aggressive market expansion scenario",
				"assumptions": {
					"revenue_growth_rate": 0.25,
					"market_expansion": 0.15,
					"customer_acquisition_rate": 0.30,
					"pricing_adjustment": 0.05,
					"operational_efficiency": 0.10
				},
				"external_factors": {
					"economic_outlook": "positive",
					"market_conditions": "favorable",
					"competitive_pressure": "low"
				}
			},
			{
				"scenario_name": "Conservative Plan",
				"description": "Risk-averse conservative scenario",
				"assumptions": {
					"revenue_growth_rate": 0.08,
					"market_expansion": 0.03,
					"customer_acquisition_rate": 0.12,
					"pricing_adjustment": 0.02,
					"operational_efficiency": 0.05
				},
				"external_factors": {
					"economic_outlook": "stable",
					"market_conditions": "neutral",
					"competitive_pressure": "medium"
				}
			},
			{
				"scenario_name": "Pessimistic Outlook",
				"description": "Economic downturn scenario",
				"assumptions": {
					"revenue_growth_rate": -0.05,
					"market_expansion": -0.02,
					"customer_acquisition_rate": 0.05,
					"pricing_adjustment": -0.03,
					"operational_efficiency": 0.15  # Cost cutting
				},
				"external_factors": {
					"economic_outlook": "negative",
					"market_conditions": "challenging",
					"competitive_pressure": "high"
				}
			}
		]
		
		scenario_results = []
		for scenario in scenarios:
			forecast_config = {
				"scenario_name": scenario["scenario_name"],
				"start_date": "2025-04-01",
				"end_date": "2025-12-31",
				"assumptions": scenario["assumptions"],
				"external_factors": scenario["external_factors"],
				"confidence_intervals": [0.80, 0.90, 0.95],
				"monte_carlo_simulations": 1000
			}
			
			forecast_response = await ml_forecasting_service.generate_forecast(model_id, forecast_config)
			assert forecast_response.success
			scenario_results.append(forecast_response.data)
		
		# Verify all scenarios generated successfully
		assert len(scenario_results) == 3
		
		# Verify scenario differences
		optimistic_total = scenario_results[0]["total_forecast"]
		conservative_total = scenario_results[1]["total_forecast"]
		pessimistic_total = scenario_results[2]["total_forecast"]
		
		assert optimistic_total > conservative_total > pessimistic_total


# =============================================================================
# Automated Monitoring Advanced Tests
# =============================================================================

class TestAdvancedAutomatedMonitoring:
	"""Test advanced automated monitoring features."""

	async def test_intelligent_alerting_system(self, monitoring_service):
		"""Test intelligent alerting with adaptive thresholds."""
		
		# Create intelligent monitoring rule
		rule_config = {
			"rule_name": "Intelligent Budget Variance Monitor",
			"alert_type": AlertType.INTELLIGENT_THRESHOLD.value,
			"description": "AI-powered variance monitoring with adaptive thresholds",
			"scope": "budget_category",
			"target_entities": ["category_marketing", "category_sales", "category_operations"],
			"intelligent_features": {
				"adaptive_thresholds": True,
				"seasonal_adjustment": True,
				"trend_analysis": True,
				"anomaly_detection": True,
				"context_awareness": True
			},
			"baseline_metrics": {
				"variance_threshold": 0.10,  # 10% baseline
				"confidence_level": 0.95,
				"historical_window": "12_months",
				"minimum_data_points": 24
			},
			"alert_conditions": {
				"statistical_significance": True,
				"consecutive_violations": 2,
				"magnitude_threshold": "dynamic",
				"pattern_recognition": True
			},
			"escalation_rules": {
				"time_based": {
					"warning_timeout": "2_hours",
					"critical_timeout": "30_minutes"
				},
				"magnitude_based": {
					"severe_multiplier": 2.0,
					"critical_multiplier": 3.0
				}
			},
			"notification_intelligence": {
				"smart_batching": True,
				"priority_routing": True,
				"context_enrichment": True,
				"action_recommendations": True
			}
		}
		
		rule_response = await monitoring_service.create_monitoring_rule(rule_config)
		assert rule_response.success
		assert "rule_id" in rule_response.data
		
		# Verify intelligent features are configured
		rule_data = rule_response.data
		assert rule_data["intelligent_features"]["adaptive_thresholds"] == True
		assert rule_data["alert_conditions"]["pattern_recognition"] == True

	async def test_predictive_alerting(self, monitoring_service):
		"""Test predictive alerting based on trend analysis."""
		
		predictive_rule_config = {
			"rule_name": "Predictive Budget Overrun Alert",
			"alert_type": AlertType.PREDICTIVE_TREND.value,
			"description": "Predict budget overruns before they occur",
			"prediction_horizon": "3_months",
			"prediction_confidence": 0.85,
			"predictive_features": {
				"spending_velocity": True,
				"seasonal_patterns": True,
				"commitment_pipeline": True,
				"historical_overruns": True,
				"external_indicators": True
			},
			"trigger_conditions": {
				"overrun_probability": 0.70,
				"magnitude_threshold": 0.05,  # 5% overrun prediction
				"time_to_overrun": "60_days"
			},
			"early_warning_stages": [
				{
					"stage": "early_indicator",
					"probability_threshold": 0.30,
					"notification_level": "info"
				},
				{
					"stage": "warning_signal",
					"probability_threshold": 0.50,
					"notification_level": "warning"
				},
				{
					"stage": "high_risk",
					"probability_threshold": 0.70,
					"notification_level": "critical"
				}
			]
		}
		
		predictive_response = await monitoring_service.create_monitoring_rule(predictive_rule_config)
		assert predictive_response.success
		
		# Test predictive analysis
		analysis_config = {
			"analysis_type": "predictive_trend",
			"target_budget": "budget_predictive_001",
			"analysis_horizon": "6_months",
			"include_scenarios": True
		}
		
		prediction_response = await monitoring_service.perform_predictive_analysis(analysis_config)
		assert prediction_response.success
		assert "predictions" in prediction_response.data
		assert "risk_assessment" in prediction_response.data

	async def test_comprehensive_anomaly_detection(self, monitoring_service):
		"""Test comprehensive anomaly detection system."""
		
		anomaly_config = {
			"detection_name": "Comprehensive Budget Anomaly Detection",
			"detection_scope": "multi_dimensional",
			"detection_methods": [
				{
					"method": "statistical_outliers",
					"parameters": {
						"z_score_threshold": 3.0,
						"iqr_multiplier": 1.5,
						"seasonal_decomposition": True
					}
				},
				{
					"method": "isolation_forest",
					"parameters": {
						"contamination": 0.1,
						"n_estimators": 100,
						"max_features": 1.0
					}
				},
				{
					"method": "local_outlier_factor",
					"parameters": {
						"n_neighbors": 20,
						"contamination": 0.1
					}
				},
				{
					"method": "one_class_svm",
					"parameters": {
						"nu": 0.1,
						"kernel": "rbf",
						"gamma": "scale"
					}
				}
			],
			"feature_dimensions": [
				"spending_amount",
				"spending_velocity",
				"transaction_frequency",
				"vendor_diversity",
				"category_distribution",
				"time_patterns"
			],
			"anomaly_types": [
				"point_anomalies",
				"contextual_anomalies", 
				"collective_anomalies",
				"trend_anomalies"
			],
			"sensitivity_levels": {
				"high_sensitivity": {
					"threshold": 0.05,
					"use_cases": ["fraud_detection", "compliance_monitoring"]
				},
				"medium_sensitivity": {
					"threshold": 0.10,
					"use_cases": ["budget_variance", "operational_efficiency"]
				},
				"low_sensitivity": {
					"threshold": 0.20,
					"use_cases": ["trend_monitoring", "planning_insights"]
				}
			}
		}
		
		anomaly_response = await monitoring_service.perform_anomaly_detection(anomaly_config)
		assert anomaly_response.success
		assert "anomalies_detected" in anomaly_response.data
		assert "confidence_scores" in anomaly_response.data
		assert "detection_methods_used" in anomaly_response.data


# =============================================================================
# Cross-Service Integration Tests
# =============================================================================

class TestCrossServiceIntegration:
	"""Test integration between advanced services."""

	async def test_ai_recommendations_with_ml_forecasting(self, ai_recommendations_service, ml_forecasting_service):
		"""Test AI recommendations leveraging ML forecasting insights."""
		
		# Create and train forecasting model
		model_config = {
			"model_name": "Integrated Forecast Model",
			"algorithm": ForecastAlgorithm.RANDOM_FOREST.value,
			"target_variable": "budget_amount",
			"horizon": ForecastHorizon.MEDIUM_TERM.value
		}
		
		model_response = await ml_forecasting_service.create_forecasting_model(model_config)
		model_id = model_response.data["model_id"]
		
		# Generate forecast
		forecast_config = {
			"scenario_name": "Base Case Forecast",
			"start_date": "2025-04-01",
			"end_date": "2025-12-31"
		}
		
		forecast_response = await ml_forecasting_service.generate_forecast(model_id, forecast_config)
		assert forecast_response.success
		
		# Generate AI recommendations using forecast insights
		context_config = {
			"budget_id": "budget_integrated_001",
			"analysis_period": "last_12_months",
			"include_ml_forecasts": True,
			"forecast_insights": {
				"model_id": model_id,
				"forecast_scenario": forecast_response.data["scenario_id"],
				"forecast_accuracy": forecast_response.data["confidence_score"]
			},
			"recommendation_focus": "forecast_informed_optimization"
		}
		
		recommendations_response = await ai_recommendations_service.generate_budget_recommendations(context_config)
		assert recommendations_response.success
		
		# Verify recommendations incorporate forecast insights
		recommendations = recommendations_response.data["recommendations"]
		forecast_informed_recs = [
			rec for rec in recommendations 
			if "forecast_insights" in rec.get("supporting_metrics", {})
		]
		assert len(forecast_informed_recs) > 0

	async def test_monitoring_triggered_ai_recommendations(self, monitoring_service, ai_recommendations_service):
		"""Test AI recommendations triggered by monitoring alerts."""
		
		# Create monitoring rule that triggers AI recommendations
		rule_config = {
			"rule_name": "AI-Enhanced Variance Monitor",
			"alert_type": AlertType.VARIANCE_THRESHOLD.value,
			"threshold_value": 5000.00,
			"automated_actions": {
				"trigger_ai_recommendations": True,
				"recommendation_context": {
					"focus_area": "variance_correction",
					"urgency": "high",
					"scope": "affected_categories"
				}
			}
		}
		
		rule_response = await monitoring_service.create_monitoring_rule(rule_config)
		assert rule_response.success
		
		# Simulate alert trigger and AI recommendation generation
		alert_simulation = {
			"rule_id": rule_response.data["rule_id"],
			"triggered_entity": "budget_monitoring_001",
			"variance_amount": 7500.00,
			"variance_category": "Marketing"
		}
		
		# This would normally be triggered automatically by the monitoring system
		triggered_recommendations = {
			"budget_id": "budget_monitoring_001",
			"trigger_context": "variance_alert",
			"focus_areas": ["Marketing"],
			"urgency": "high"
		}
		
		recommendations_response = await ai_recommendations_service.generate_budget_recommendations(triggered_recommendations)
		assert recommendations_response.success
		
		# Verify recommendations are focused on variance correction
		recommendations = recommendations_response.data["recommendations"]
		variance_recs = [
			rec for rec in recommendations 
			if rec["recommendation_type"] == RecommendationType.VARIANCE_CORRECTION.value
		]
		assert len(variance_recs) > 0


if __name__ == "__main__":
	"""Run advanced features tests."""
	pytest.main([__file__, "-v", "--tb=short"])