"""
APG Customer Relationship Management - Predictive Analytics Migration

Database migration to create predictive analytics tables and supporting 
structures for machine learning models, forecasting, churn prediction, and AI insights.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from .base_migration import BaseMigration, MigrationDirection


logger = logging.getLogger(__name__)


class PredictiveAnalyticsMigration(BaseMigration):
	"""Migration for predictive analytics functionality"""
	
	def _get_migration_id(self) -> str:
		return "019_predictive_analytics"
	
	def _get_version(self) -> str:
		return "019"
	
	def _get_description(self) -> str:
		return "Predictive analytics with ML models, forecasting, and AI insights"
	
	def _get_dependencies(self) -> list[str]:
		return ["018_reporting_engine"]

	async def _execute_up_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the up migration"""
		try:
			logger.info("🔄 Creating predictive analytics tables...")
			
			# Create model type enum
			await conn.execute("""
				CREATE TYPE model_type AS ENUM (
					'regression',
					'classification',
					'clustering',
					'time_series',
					'neural_network',
					'ensemble'
				);
			""")
			
			# Create algorithm enum
			await conn.execute("""
				CREATE TYPE ml_algorithm AS ENUM (
					'linear_regression',
					'logistic_regression',
					'random_forest',
					'gradient_boosting',
					'svm',
					'neural_network',
					'k_means',
					'decision_tree',
					'naive_bayes',
					'xgboost',
					'lstm',
					'arima'
				);
			""")
			
			# Create prediction type enum
			await conn.execute("""
				CREATE TYPE prediction_type AS ENUM (
					'single',
					'batch',
					'streaming',
					'scheduled'
				);
			""")
			
			# Create forecast type enum
			await conn.execute("""
				CREATE TYPE forecast_type AS ENUM (
					'revenue',
					'deals',
					'pipeline',
					'leads',
					'activities',
					'conversions',
					'churn'
				);
			""")
			
			# Create churn risk level enum
			await conn.execute("""
				CREATE TYPE churn_risk_level AS ENUM (
					'low',
					'medium',
					'high',
					'critical'
				);
			""")
			
			# Create prediction models table
			await conn.execute("""
				CREATE TABLE crm_prediction_models (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					name VARCHAR(255) NOT NULL,
					description TEXT,
					model_type model_type NOT NULL,
					algorithm ml_algorithm NOT NULL,
					target_variable VARCHAR(100) NOT NULL,
					feature_columns JSONB NOT NULL DEFAULT '[]',
					data_sources JSONB NOT NULL DEFAULT '[]',
					training_data_query TEXT NOT NULL,
					hyperparameters JSONB DEFAULT '{}',
					accuracy_score DECIMAL(5,4),
					precision_score DECIMAL(5,4),
					recall_score DECIMAL(5,4),
					f1_score DECIMAL(5,4),
					training_samples INTEGER,
					validation_samples INTEGER,
					cross_validation_scores JSONB DEFAULT '[]',
					feature_importance JSONB DEFAULT '{}',
					confusion_matrix JSONB DEFAULT '{}',
					last_trained_at TIMESTAMP WITH TIME ZONE,
					model_path VARCHAR(500),
					model_version VARCHAR(20) DEFAULT '1.0',
					is_active BOOLEAN DEFAULT true,
					auto_retrain BOOLEAN DEFAULT false,
					retrain_frequency_days INTEGER DEFAULT 30,
					performance_threshold DECIMAL(5,4) DEFAULT 0.80,
					drift_detection_enabled BOOLEAN DEFAULT true,
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					updated_by VARCHAR(36)
				);
			""")
			
			# Create prediction requests table
			await conn.execute("""
				CREATE TABLE crm_prediction_requests (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					model_id VARCHAR(36) NOT NULL REFERENCES crm_prediction_models(id) ON DELETE CASCADE,
					prediction_type prediction_type NOT NULL,
					request_data JSONB NOT NULL,
					batch_size INTEGER DEFAULT 1,
					confidence_threshold DECIMAL(3,2) DEFAULT 0.70,
					include_explanations BOOLEAN DEFAULT true,
					include_feature_importance BOOLEAN DEFAULT false,
					status VARCHAR(20) DEFAULT 'pending',
					started_at TIMESTAMP WITH TIME ZONE,
					completed_at TIMESTAMP WITH TIME ZONE,
					execution_time_ms DECIMAL(10,2),
					error_message TEXT,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL
				);
			""")
			
			# Create prediction results table
			await conn.execute("""
				CREATE TABLE crm_prediction_results (
					id VARCHAR(36) PRIMARY KEY,
					request_id VARCHAR(36) NOT NULL REFERENCES crm_prediction_requests(id) ON DELETE CASCADE,
					model_id VARCHAR(36) NOT NULL REFERENCES crm_prediction_models(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					predictions JSONB NOT NULL,
					confidence_scores JSONB NOT NULL DEFAULT '[]',
					probability_distributions JSONB DEFAULT '[]',
					feature_importance JSONB DEFAULT '{}',
					model_explanations JSONB DEFAULT '[]',
					prediction_intervals JSONB DEFAULT '{}',
					outlier_scores JSONB DEFAULT '[]',
					execution_time_ms DECIMAL(10,2) NOT NULL,
					model_version VARCHAR(20),
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create forecasting insights table
			await conn.execute("""
				CREATE TABLE crm_forecasting_insights (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					forecast_type forecast_type NOT NULL,
					period_type VARCHAR(20) NOT NULL,
					forecast_period VARCHAR(50) NOT NULL,
					predicted_value DECIMAL(15,2) NOT NULL,
					confidence_interval_lower DECIMAL(15,2) NOT NULL,
					confidence_interval_upper DECIMAL(15,2) NOT NULL,
					accuracy_score DECIMAL(5,4) NOT NULL,
					mean_absolute_error DECIMAL(15,2),
					root_mean_square_error DECIMAL(15,2),
					trend_direction VARCHAR(20) NOT NULL,
					trend_strength DECIMAL(3,2),
					seasonality_detected BOOLEAN DEFAULT false,
					seasonal_patterns JSONB DEFAULT '{}',
					key_drivers JSONB DEFAULT '[]',
					external_factors JSONB DEFAULT '[]',
					recommendations JSONB DEFAULT '[]',
					forecast_horizon_days INTEGER,
					data_quality_score DECIMAL(3,2),
					model_confidence DECIMAL(3,2),
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					expires_at TIMESTAMP WITH TIME ZONE,
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create churn predictions table
			await conn.execute("""
				CREATE TABLE crm_churn_predictions (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					entity_type VARCHAR(20) NOT NULL,
					entity_id VARCHAR(36) NOT NULL,
					churn_probability DECIMAL(5,4) NOT NULL,
					churn_risk_level churn_risk_level NOT NULL,
					churn_score INTEGER,
					key_risk_factors JSONB DEFAULT '[]',
					protective_factors JSONB DEFAULT '[]',
					retention_recommendations JSONB DEFAULT '[]',
					predicted_churn_date DATE,
					confidence_level DECIMAL(3,2),
					model_confidence DECIMAL(3,2) NOT NULL,
					historical_behavior JSONB DEFAULT '{}',
					engagement_metrics JSONB DEFAULT '{}',
					financial_metrics JSONB DEFAULT '{}',
					support_interaction_score DECIMAL(3,2),
					product_usage_score DECIMAL(3,2),
					satisfaction_score DECIMAL(3,2),
					lifetime_value DECIMAL(15,2),
					intervention_priority INTEGER DEFAULT 1,
					last_contact_date DATE,
					next_recommended_action VARCHAR(255),
					action_urgency VARCHAR(20) DEFAULT 'medium',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create lead scoring insights table
			await conn.execute("""
				CREATE TABLE crm_lead_scoring_insights (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					lead_id VARCHAR(36) NOT NULL,
					predicted_score DECIMAL(5,2) NOT NULL,
					conversion_probability DECIMAL(5,4) NOT NULL,
					quality_grade VARCHAR(2),
					optimal_contact_time JSONB DEFAULT '{}',
					recommended_actions JSONB DEFAULT '[]',
					scoring_factors JSONB DEFAULT '{}',
					demographic_score DECIMAL(5,2),
					behavioral_score DECIMAL(5,2),
					engagement_score DECIMAL(5,2),
					firmographic_score DECIMAL(5,2),
					competitive_analysis JSONB DEFAULT '{}',
					market_timing_score DECIMAL(5,2),
					budget_probability DECIMAL(5,4),
					authority_level DECIMAL(3,2),
					need_urgency DECIMAL(3,2),
					timeline_prediction VARCHAR(50),
					expected_deal_size DECIMAL(15,2),
					win_probability DECIMAL(5,4),
					competitor_risk DECIMAL(3,2),
					nurturing_track VARCHAR(100),
					sales_readiness BOOLEAN DEFAULT false,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create market segments table
			await conn.execute("""
				CREATE TABLE crm_market_segments (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					segment_name VARCHAR(255) NOT NULL,
					segment_description TEXT,
					segment_code VARCHAR(50),
					customer_count INTEGER DEFAULT 0,
					avg_lifetime_value DECIMAL(15,2),
					total_segment_value DECIMAL(15,2),
					key_characteristics JSONB DEFAULT '{}',
					demographic_profile JSONB DEFAULT '{}',
					behavior_patterns JSONB DEFAULT '{}',
					purchase_patterns JSONB DEFAULT '{}',
					communication_preferences JSONB DEFAULT '{}',
					marketing_recommendations JSONB DEFAULT '[]',
					product_affinity JSONB DEFAULT '{}',
					price_sensitivity DECIMAL(3,2),
					churn_risk DECIMAL(3,2),
					growth_potential DECIMAL(3,2),
					profitability_score DECIMAL(5,2),
					acquisition_cost DECIMAL(15,2),
					retention_rate DECIMAL(5,4),
					engagement_level VARCHAR(20),
					preferred_channels JSONB DEFAULT '[]',
					seasonal_trends JSONB DEFAULT '{}',
					competitive_threats JSONB DEFAULT '[]',
					segment_health_score DECIMAL(5,2),
					is_active BOOLEAN DEFAULT true,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create AI insights table
			await conn.execute("""
				CREATE TABLE crm_ai_insights (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					insight_category VARCHAR(50) NOT NULL,
					insight_type VARCHAR(50) NOT NULL,
					title VARCHAR(255) NOT NULL,
					description TEXT NOT NULL,
					detailed_analysis TEXT,
					confidence_score DECIMAL(3,2) NOT NULL,
					impact_level VARCHAR(20) NOT NULL,
					urgency_level VARCHAR(20) DEFAULT 'medium',
					supporting_data JSONB DEFAULT '{}',
					visualization_data JSONB DEFAULT '{}',
					affected_entities JSONB DEFAULT '[]',
					recommended_actions JSONB DEFAULT '[]',
					potential_value DECIMAL(15,2),
					implementation_effort VARCHAR(20),
					timeframe_days INTEGER,
					success_probability DECIMAL(3,2),
					risk_assessment JSONB DEFAULT '{}',
					dependencies JSONB DEFAULT '[]',
					is_active BOOLEAN DEFAULT true,
					is_acknowledged BOOLEAN DEFAULT false,
					acknowledged_by VARCHAR(36),
					acknowledged_at TIMESTAMP WITH TIME ZONE,
					resolution_status VARCHAR(20) DEFAULT 'pending',
					resolution_notes TEXT,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					expires_at TIMESTAMP WITH TIME ZONE,
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create model performance tracking table
			await conn.execute("""
				CREATE TABLE crm_model_performance (
					id VARCHAR(36) PRIMARY KEY,
					model_id VARCHAR(36) NOT NULL REFERENCES crm_prediction_models(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					evaluation_date DATE NOT NULL,
					accuracy_score DECIMAL(5,4),
					precision_score DECIMAL(5,4),
					recall_score DECIMAL(5,4),
					f1_score DECIMAL(5,4),
					auc_score DECIMAL(5,4),
					mean_absolute_error DECIMAL(15,4),
					root_mean_square_error DECIMAL(15,4),
					prediction_count INTEGER DEFAULT 0,
					correct_predictions INTEGER DEFAULT 0,
					false_positives INTEGER DEFAULT 0,
					false_negatives INTEGER DEFAULT 0,
					average_confidence DECIMAL(3,2),
					model_drift_score DECIMAL(3,2),
					data_quality_score DECIMAL(3,2),
					training_data_size INTEGER,
					prediction_latency_ms DECIMAL(10,2),
					resource_usage JSONB DEFAULT '{}',
					performance_trends JSONB DEFAULT '{}',
					alerts_triggered JSONB DEFAULT '[]',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					UNIQUE(model_id, evaluation_date)
				);
			""")
			
			# Create analytics experiments table
			await conn.execute("""
				CREATE TABLE crm_analytics_experiments (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					experiment_name VARCHAR(255) NOT NULL,
					experiment_type VARCHAR(50) NOT NULL,
					description TEXT,
					hypothesis TEXT,
					success_criteria JSONB DEFAULT '{}',
					control_group_size INTEGER,
					treatment_group_size INTEGER,
					start_date DATE NOT NULL,
					end_date DATE,
					status VARCHAR(20) DEFAULT 'planning',
					configuration JSONB DEFAULT '{}',
					baseline_metrics JSONB DEFAULT '{}',
					current_metrics JSONB DEFAULT '{}',
					statistical_significance DECIMAL(5,4),
					p_value DECIMAL(10,8),
					effect_size DECIMAL(5,4),
					confidence_interval JSONB DEFAULT '{}',
					results_summary TEXT,
					conclusions TEXT,
					recommendations JSONB DEFAULT '[]',
					business_impact DECIMAL(15,2),
					created_by VARCHAR(36) NOT NULL,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create indexes for performance
			await conn.execute("CREATE INDEX idx_prediction_models_tenant ON crm_prediction_models(tenant_id);")
			await conn.execute("CREATE INDEX idx_prediction_models_type ON crm_prediction_models(model_type);")
			await conn.execute("CREATE INDEX idx_prediction_models_algorithm ON crm_prediction_models(algorithm);")
			await conn.execute("CREATE INDEX idx_prediction_models_active ON crm_prediction_models(is_active);")
			await conn.execute("CREATE INDEX idx_prediction_models_last_trained ON crm_prediction_models(last_trained_at);")
			
			await conn.execute("CREATE INDEX idx_prediction_requests_model ON crm_prediction_requests(model_id);")
			await conn.execute("CREATE INDEX idx_prediction_requests_tenant ON crm_prediction_requests(tenant_id);")
			await conn.execute("CREATE INDEX idx_prediction_requests_status ON crm_prediction_requests(status);")
			await conn.execute("CREATE INDEX idx_prediction_requests_created ON crm_prediction_requests(created_at);")
			
			await conn.execute("CREATE INDEX idx_prediction_results_request ON crm_prediction_results(request_id);")
			await conn.execute("CREATE INDEX idx_prediction_results_model ON crm_prediction_results(model_id);")
			await conn.execute("CREATE INDEX idx_prediction_results_tenant ON crm_prediction_results(tenant_id);")
			await conn.execute("CREATE INDEX idx_prediction_results_created ON crm_prediction_results(created_at);")
			
			await conn.execute("CREATE INDEX idx_forecasting_insights_tenant ON crm_forecasting_insights(tenant_id);")
			await conn.execute("CREATE INDEX idx_forecasting_insights_type ON crm_forecasting_insights(forecast_type);")
			await conn.execute("CREATE INDEX idx_forecasting_insights_period ON crm_forecasting_insights(forecast_period);")
			await conn.execute("CREATE INDEX idx_forecasting_insights_created ON crm_forecasting_insights(created_at);")
			await conn.execute("CREATE INDEX idx_forecasting_insights_expires ON crm_forecasting_insights(expires_at);")
			
			await conn.execute("CREATE INDEX idx_churn_predictions_tenant ON crm_churn_predictions(tenant_id);")
			await conn.execute("CREATE INDEX idx_churn_predictions_entity ON crm_churn_predictions(entity_type, entity_id);")
			await conn.execute("CREATE INDEX idx_churn_predictions_risk ON crm_churn_predictions(churn_risk_level);")
			await conn.execute("CREATE INDEX idx_churn_predictions_probability ON crm_churn_predictions(churn_probability);")
			await conn.execute("CREATE INDEX idx_churn_predictions_date ON crm_churn_predictions(predicted_churn_date);")
			
			await conn.execute("CREATE INDEX idx_lead_scoring_insights_tenant ON crm_lead_scoring_insights(tenant_id);")
			await conn.execute("CREATE INDEX idx_lead_scoring_insights_lead ON crm_lead_scoring_insights(lead_id);")
			await conn.execute("CREATE INDEX idx_lead_scoring_insights_score ON crm_lead_scoring_insights(predicted_score);")
			await conn.execute("CREATE INDEX idx_lead_scoring_insights_probability ON crm_lead_scoring_insights(conversion_probability);")
			await conn.execute("CREATE INDEX idx_lead_scoring_insights_created ON crm_lead_scoring_insights(created_at);")
			
			await conn.execute("CREATE INDEX idx_market_segments_tenant ON crm_market_segments(tenant_id);")
			await conn.execute("CREATE INDEX idx_market_segments_name ON crm_market_segments(segment_name);")
			await conn.execute("CREATE INDEX idx_market_segments_active ON crm_market_segments(is_active);")
			await conn.execute("CREATE INDEX idx_market_segments_health ON crm_market_segments(segment_health_score);")
			
			await conn.execute("CREATE INDEX idx_ai_insights_tenant ON crm_ai_insights(tenant_id);")
			await conn.execute("CREATE INDEX idx_ai_insights_category ON crm_ai_insights(insight_category);")
			await conn.execute("CREATE INDEX idx_ai_insights_type ON crm_ai_insights(insight_type);")
			await conn.execute("CREATE INDEX idx_ai_insights_active ON crm_ai_insights(is_active);")
			await conn.execute("CREATE INDEX idx_ai_insights_acknowledged ON crm_ai_insights(is_acknowledged);")
			await conn.execute("CREATE INDEX idx_ai_insights_impact ON crm_ai_insights(impact_level);")
			await conn.execute("CREATE INDEX idx_ai_insights_expires ON crm_ai_insights(expires_at);")
			
			await conn.execute("CREATE INDEX idx_model_performance_model ON crm_model_performance(model_id);")
			await conn.execute("CREATE INDEX idx_model_performance_date ON crm_model_performance(evaluation_date);")
			await conn.execute("CREATE INDEX idx_model_performance_tenant ON crm_model_performance(tenant_id);")
			
			await conn.execute("CREATE INDEX idx_analytics_experiments_tenant ON crm_analytics_experiments(tenant_id);")
			await conn.execute("CREATE INDEX idx_analytics_experiments_type ON crm_analytics_experiments(experiment_type);")
			await conn.execute("CREATE INDEX idx_analytics_experiments_status ON crm_analytics_experiments(status);")
			await conn.execute("CREATE INDEX idx_analytics_experiments_dates ON crm_analytics_experiments(start_date, end_date);")
			
			# Insert default AI insight categories
			await conn.execute("""
				INSERT INTO crm_ai_insights (
					id, tenant_id, insight_category, insight_type, title, description,
					confidence_score, impact_level, supporting_data, recommended_actions,
					potential_value, is_active, created_at
				) VALUES 
				(
					'ai_insight_sample_01',
					'system',
					'sales_optimization',
					'opportunity_risk',
					'High-Value Opportunities at Risk',
					'Several high-value opportunities show signs of potential loss based on engagement patterns and competitive indicators.',
					0.87,
					'high',
					'{"at_risk_count": 5, "total_value": 450000, "average_days_stalled": 14}',
					'["Schedule urgent stakeholder meetings", "Provide competitive analysis", "Offer limited-time incentives"]',
					125000.00,
					true,
					NOW()
				),
				(
					'ai_insight_sample_02',
					'system',
					'lead_generation',
					'source_optimization',
					'Underperforming Lead Sources Identified',
					'Analysis reveals that certain lead sources have significantly lower conversion rates and should be optimized or discontinued.',
					0.92,
					'medium',
					'{"low_performing_sources": ["trade_shows", "cold_email"], "conversion_rates": {"trade_shows": 0.02, "cold_email": 0.01}}',
					'["Reallocate budget from underperforming channels", "Optimize landing pages", "Improve lead qualification"]',
					75000.00,
					true,
					NOW()
				),
				(
					'ai_insight_sample_03',
					'system',
					'customer_success',
					'churn_prevention',
					'Customer Churn Risk Spike Detected',
					'Predictive models indicate a 23% increase in churn risk among enterprise customers due to reduced product usage.',
					0.94,
					'critical',
					'{"at_risk_customers": 12, "predicted_revenue_loss": 280000, "primary_risk_factor": "decreased_usage"}',
					'["Launch proactive customer success campaigns", "Provide additional training", "Offer product usage consulting"]',
					200000.00,
					true,
					NOW()
				)
			""")
			
			logger.info("✅ Predictive analytics tables created successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to create predictive analytics tables: {str(e)}")
			raise
	
	async def _execute_down_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the down migration"""
		try:
			logger.info("🔄 Dropping predictive analytics tables...")
			
			# Drop tables in reverse order
			await conn.execute("DROP TABLE IF EXISTS crm_analytics_experiments CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_model_performance CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_ai_insights CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_market_segments CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_lead_scoring_insights CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_churn_predictions CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_forecasting_insights CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_prediction_results CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_prediction_requests CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_prediction_models CASCADE;")
			
			# Drop enums
			await conn.execute("DROP TYPE IF EXISTS churn_risk_level CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS forecast_type CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS prediction_type CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS ml_algorithm CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS model_type CASCADE;")
			
			logger.info("✅ Predictive analytics tables dropped successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to drop predictive analytics tables: {str(e)}")
			raise