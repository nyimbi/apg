"""
APG Customer Relationship Management - Performance Benchmarking Migration

Database migration to create performance benchmarking tables and supporting 
structures for KPI tracking, team comparisons, goal tracking, and performance reports.

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


class PerformanceBenchmarkingMigration(BaseMigration):
	"""Migration for performance benchmarking functionality"""
	
	def _get_migration_id(self) -> str:
		return "020_performance_benchmarking"
	
	def _get_version(self) -> str:
		return "020"
	
	def _get_description(self) -> str:
		return "Performance benchmarking with KPI tracking and team comparisons"
	
	def _get_dependencies(self) -> list[str]:
		return ["019_predictive_analytics"]

	async def _execute_up_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the up migration"""
		try:
			logger.info("🔄 Creating performance benchmarking tables...")
			
			# Create benchmark type enum
			await conn.execute("""
				CREATE TYPE benchmark_type AS ENUM (
					'individual',
					'team',
					'department',
					'organization',
					'industry'
				);
			""")
			
			# Create performance rating enum
			await conn.execute("""
				CREATE TYPE performance_rating AS ENUM (
					'poor',
					'fair',
					'good',
					'excellent'
				);
			""")
			
			# Create trend direction enum
			await conn.execute("""
				CREATE TYPE trend_direction AS ENUM (
					'improving',
					'declining',
					'stable'
				);
			""")
			
			# Create goal type enum
			await conn.execute("""
				CREATE TYPE goal_type AS ENUM (
					'revenue',
					'deals',
					'activities',
					'conversion',
					'retention',
					'satisfaction',
					'productivity',
					'quality',
					'efficiency'
				);
			""")
			
			# Create performance benchmarks table
			await conn.execute("""
				CREATE TABLE crm_performance_benchmarks (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					name VARCHAR(255) NOT NULL,
					description TEXT,
					benchmark_type benchmark_type NOT NULL,
					metric_name VARCHAR(100) NOT NULL,
					measurement_unit VARCHAR(50) NOT NULL,
					benchmark_value DECIMAL(15,4) NOT NULL,
					target_value DECIMAL(15,4),
					threshold_ranges JSONB DEFAULT '{}',
					period_type VARCHAR(20) DEFAULT 'monthly',
					industry_category VARCHAR(100),
					data_source VARCHAR(100) NOT NULL,
					calculation_method VARCHAR(100) NOT NULL,
					weighting_factor DECIMAL(3,2) DEFAULT 1.00,
					aggregation_method VARCHAR(50) DEFAULT 'average',
					is_active BOOLEAN DEFAULT true,
					is_public BOOLEAN DEFAULT false,
					refresh_frequency VARCHAR(20) DEFAULT 'daily',
					data_retention_days INTEGER DEFAULT 365,
					alert_thresholds JSONB DEFAULT '{}',
					notification_rules JSONB DEFAULT '[]',
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					updated_by VARCHAR(36)
				);
			""")
			
			# Create performance metrics table
			await conn.execute("""
				CREATE TABLE crm_performance_metrics (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					benchmark_id VARCHAR(36) NOT NULL REFERENCES crm_performance_benchmarks(id) ON DELETE CASCADE,
					entity_type VARCHAR(20) NOT NULL,
					entity_id VARCHAR(36) NOT NULL,
					entity_name VARCHAR(255) NOT NULL,
					measurement_period VARCHAR(50) NOT NULL,
					period_start DATE NOT NULL,
					period_end DATE NOT NULL,
					actual_value DECIMAL(15,4) NOT NULL,
					benchmark_value DECIMAL(15,4) NOT NULL,
					target_value DECIMAL(15,4),
					variance_amount DECIMAL(15,4) NOT NULL,
					variance_percentage DECIMAL(8,4) NOT NULL,
					performance_rating performance_rating NOT NULL,
					performance_score DECIMAL(5,2),
					trend_direction trend_direction DEFAULT 'stable',
					confidence_level DECIMAL(3,2) DEFAULT 0.95,
					data_quality_score DECIMAL(3,2) DEFAULT 1.00,
					outlier_flag BOOLEAN DEFAULT false,
					supporting_data JSONB DEFAULT '{}',
					calculation_details JSONB DEFAULT '{}',
					external_factors JSONB DEFAULT '[]',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					INDEX (tenant_id, entity_type, entity_id, period_start),
					INDEX (benchmark_id, period_start),
					UNIQUE(benchmark_id, entity_id, measurement_period)
				);
			""")
			
			# Create performance comparisons table
			await conn.execute("""
				CREATE TABLE crm_performance_comparisons (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					comparison_name VARCHAR(255) NOT NULL,
					comparison_type VARCHAR(50) NOT NULL,
					entities JSONB NOT NULL DEFAULT '[]',
					metrics JSONB NOT NULL DEFAULT '[]',
					period_start DATE NOT NULL,
					period_end DATE NOT NULL,
					rankings JSONB DEFAULT '[]',
					statistical_analysis JSONB DEFAULT '{}',
					correlation_analysis JSONB DEFAULT '{}',
					insights JSONB DEFAULT '[]',
					recommendations JSONB DEFAULT '[]',
					significance_tests JSONB DEFAULT '{}',
					sample_size INTEGER,
					confidence_interval JSONB DEFAULT '{}',
					is_published BOOLEAN DEFAULT false,
					access_level VARCHAR(20) DEFAULT 'private',
					expiry_date DATE,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL
				);
			""")
			
			# Create goal tracking table
			await conn.execute("""
				CREATE TABLE crm_goal_tracking (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					goal_name VARCHAR(255) NOT NULL,
					description TEXT,
					goal_type goal_type NOT NULL,
					target_value DECIMAL(15,4) NOT NULL,
					current_value DECIMAL(15,4) DEFAULT 0.00,
					progress_percentage DECIMAL(5,2) DEFAULT 0.00,
					entity_type VARCHAR(20) NOT NULL,
					entity_id VARCHAR(36) NOT NULL,
					entity_name VARCHAR(255) NOT NULL,
					start_date DATE NOT NULL,
					end_date DATE NOT NULL,
					milestone_dates JSONB DEFAULT '[]',
					milestone_progress JSONB DEFAULT '[]',
					milestone_achievements JSONB DEFAULT '[]',
					is_active BOOLEAN DEFAULT true,
					priority_level VARCHAR(20) DEFAULT 'medium',
					tracking_frequency VARCHAR(20) DEFAULT 'daily',
					reminder_frequency VARCHAR(20) DEFAULT 'weekly',
					auto_update_enabled BOOLEAN DEFAULT true,
					reward_structure JSONB DEFAULT '{}',
					penalty_structure JSONB DEFAULT '{}',
					stakeholders JSONB DEFAULT '[]',
					reporting_schedule JSONB DEFAULT '{}',
					last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL
				);
			""")
			
			# Create performance reports table
			await conn.execute("""
				CREATE TABLE crm_performance_reports (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					report_name VARCHAR(255) NOT NULL,
					report_type VARCHAR(50) NOT NULL,
					entity_id VARCHAR(36) NOT NULL,
					entity_name VARCHAR(255) NOT NULL,
					reporting_period VARCHAR(100) NOT NULL,
					period_start DATE NOT NULL,
					period_end DATE NOT NULL,
					overall_score DECIMAL(5,2) NOT NULL,
					performance_grade VARCHAR(2) NOT NULL,
					key_metrics JSONB DEFAULT '[]',
					strengths JSONB DEFAULT '[]',
					improvement_areas JSONB DEFAULT '[]',
					recommendations JSONB DEFAULT '[]',
					goal_achievements JSONB DEFAULT '[]',
					trend_analysis JSONB DEFAULT '{}',
					peer_comparison JSONB DEFAULT '{}',
					industry_comparison JSONB DEFAULT '{}',
					coaching_suggestions JSONB DEFAULT '[]',
					training_recommendations JSONB DEFAULT '[]',
					next_review_date DATE,
					report_status VARCHAR(20) DEFAULT 'draft',
					approval_status VARCHAR(20) DEFAULT 'pending',
					approved_by VARCHAR(36),
					approved_at TIMESTAMP WITH TIME ZONE,
					shared_with JSONB DEFAULT '[]',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create performance alerts table
			await conn.execute("""
				CREATE TABLE crm_performance_alerts (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					alert_type VARCHAR(50) NOT NULL,
					severity_level VARCHAR(20) NOT NULL,
					title VARCHAR(255) NOT NULL,
					description TEXT NOT NULL,
					entity_type VARCHAR(20) NOT NULL,
					entity_id VARCHAR(36) NOT NULL,
					entity_name VARCHAR(255) NOT NULL,
					metric_name VARCHAR(100),
					threshold_value DECIMAL(15,4),
					actual_value DECIMAL(15,4),
					variance_percentage DECIMAL(8,4),
					alert_conditions JSONB DEFAULT '{}',
					triggered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					acknowledged BOOLEAN DEFAULT false,
					acknowledged_by VARCHAR(36),
					acknowledged_at TIMESTAMP WITH TIME ZONE,
					resolved BOOLEAN DEFAULT false,
					resolved_by VARCHAR(36),
					resolved_at TIMESTAMP WITH TIME ZONE,
					resolution_notes TEXT,
					escalation_level INTEGER DEFAULT 0,
					next_escalation_at TIMESTAMP WITH TIME ZONE,
					notification_sent BOOLEAN DEFAULT false,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create benchmark categories table
			await conn.execute("""
				CREATE TABLE crm_benchmark_categories (
					id VARCHAR(36) PRIMARY KEY,
					name VARCHAR(100) NOT NULL,
					description TEXT,
					category_type VARCHAR(50) NOT NULL,
					parent_category_id VARCHAR(36) REFERENCES crm_benchmark_categories(id),
					display_order INTEGER DEFAULT 0,
					icon VARCHAR(100),
					color VARCHAR(7) DEFAULT '#007bff',
					is_active BOOLEAN DEFAULT true,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					UNIQUE(name, category_type)
				);
			""")
			
			# Create performance coaching table
			await conn.execute("""
				CREATE TABLE crm_performance_coaching (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					coach_id VARCHAR(36) NOT NULL,
					coachee_id VARCHAR(36) NOT NULL,
					coaching_session_name VARCHAR(255) NOT NULL,
					session_type VARCHAR(50) NOT NULL,
					session_date DATE NOT NULL,
					duration_minutes INTEGER,
					performance_focus_areas JSONB DEFAULT '[]',
					goals_discussed JSONB DEFAULT '[]',
					action_items JSONB DEFAULT '[]',
					progress_notes TEXT,
					next_session_date DATE,
					session_rating INTEGER CHECK (session_rating >= 1 AND session_rating <= 5),
					coach_notes TEXT,
					coachee_feedback TEXT,
					coaching_materials JSONB DEFAULT '[]',
					follow_up_required BOOLEAN DEFAULT false,
					status VARCHAR(20) DEFAULT 'scheduled',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create indexes for performance
			await conn.execute("CREATE INDEX idx_performance_benchmarks_tenant ON crm_performance_benchmarks(tenant_id);")
			await conn.execute("CREATE INDEX idx_performance_benchmarks_type ON crm_performance_benchmarks(benchmark_type);")
			await conn.execute("CREATE INDEX idx_performance_benchmarks_metric ON crm_performance_benchmarks(metric_name);")
			await conn.execute("CREATE INDEX idx_performance_benchmarks_active ON crm_performance_benchmarks(is_active);")
			await conn.execute("CREATE INDEX idx_performance_benchmarks_industry ON crm_performance_benchmarks(industry_category);")
			
			await conn.execute("CREATE INDEX idx_performance_metrics_tenant ON crm_performance_metrics(tenant_id);")
			await conn.execute("CREATE INDEX idx_performance_metrics_benchmark ON crm_performance_metrics(benchmark_id);")
			await conn.execute("CREATE INDEX idx_performance_metrics_entity ON crm_performance_metrics(entity_type, entity_id);")
			await conn.execute("CREATE INDEX idx_performance_metrics_period ON crm_performance_metrics(period_start, period_end);")
			await conn.execute("CREATE INDEX idx_performance_metrics_rating ON crm_performance_metrics(performance_rating);")
			await conn.execute("CREATE INDEX idx_performance_metrics_trend ON crm_performance_metrics(trend_direction);")
			
			await conn.execute("CREATE INDEX idx_performance_comparisons_tenant ON crm_performance_comparisons(tenant_id);")
			await conn.execute("CREATE INDEX idx_performance_comparisons_type ON crm_performance_comparisons(comparison_type);")
			await conn.execute("CREATE INDEX idx_performance_comparisons_period ON crm_performance_comparisons(period_start, period_end);")
			await conn.execute("CREATE INDEX idx_performance_comparisons_published ON crm_performance_comparisons(is_published);")
			
			await conn.execute("CREATE INDEX idx_goal_tracking_tenant ON crm_goal_tracking(tenant_id);")
			await conn.execute("CREATE INDEX idx_goal_tracking_entity ON crm_goal_tracking(entity_type, entity_id);")
			await conn.execute("CREATE INDEX idx_goal_tracking_type ON crm_goal_tracking(goal_type);")
			await conn.execute("CREATE INDEX idx_goal_tracking_active ON crm_goal_tracking(is_active);")
			await conn.execute("CREATE INDEX idx_goal_tracking_dates ON crm_goal_tracking(start_date, end_date);")
			await conn.execute("CREATE INDEX idx_goal_tracking_progress ON crm_goal_tracking(progress_percentage);")
			
			await conn.execute("CREATE INDEX idx_performance_reports_tenant ON crm_performance_reports(tenant_id);")
			await conn.execute("CREATE INDEX idx_performance_reports_entity ON crm_performance_reports(entity_id);")
			await conn.execute("CREATE INDEX idx_performance_reports_type ON crm_performance_reports(report_type);")
			await conn.execute("CREATE INDEX idx_performance_reports_period ON crm_performance_reports(period_start, period_end);")
			await conn.execute("CREATE INDEX idx_performance_reports_grade ON crm_performance_reports(performance_grade);")
			await conn.execute("CREATE INDEX idx_performance_reports_status ON crm_performance_reports(report_status);")
			
			await conn.execute("CREATE INDEX idx_performance_alerts_tenant ON crm_performance_alerts(tenant_id);")
			await conn.execute("CREATE INDEX idx_performance_alerts_entity ON crm_performance_alerts(entity_type, entity_id);")
			await conn.execute("CREATE INDEX idx_performance_alerts_type ON crm_performance_alerts(alert_type);")
			await conn.execute("CREATE INDEX idx_performance_alerts_severity ON crm_performance_alerts(severity_level);")
			await conn.execute("CREATE INDEX idx_performance_alerts_acknowledged ON crm_performance_alerts(acknowledged);")
			await conn.execute("CREATE INDEX idx_performance_alerts_resolved ON crm_performance_alerts(resolved);")
			await conn.execute("CREATE INDEX idx_performance_alerts_triggered ON crm_performance_alerts(triggered_at);")
			
			await conn.execute("CREATE INDEX idx_benchmark_categories_type ON crm_benchmark_categories(category_type);")
			await conn.execute("CREATE INDEX idx_benchmark_categories_parent ON crm_benchmark_categories(parent_category_id);")
			await conn.execute("CREATE INDEX idx_benchmark_categories_active ON crm_benchmark_categories(is_active);")
			
			await conn.execute("CREATE INDEX idx_performance_coaching_tenant ON crm_performance_coaching(tenant_id);")
			await conn.execute("CREATE INDEX idx_performance_coaching_coach ON crm_performance_coaching(coach_id);")
			await conn.execute("CREATE INDEX idx_performance_coaching_coachee ON crm_performance_coaching(coachee_id);")
			await conn.execute("CREATE INDEX idx_performance_coaching_date ON crm_performance_coaching(session_date);")
			await conn.execute("CREATE INDEX idx_performance_coaching_status ON crm_performance_coaching(status);")
			
			# Insert default benchmark categories
			await conn.execute("""
				INSERT INTO crm_benchmark_categories (id, name, description, category_type, display_order, icon, color) VALUES 
				('cat_sales_performance', 'Sales Performance', 'Sales-related performance metrics', 'sales', 1, 'trending-up', '#28a745'),
				('cat_lead_generation', 'Lead Generation', 'Lead generation and qualification metrics', 'marketing', 2, 'users', '#17a2b8'),
				('cat_customer_success', 'Customer Success', 'Customer satisfaction and retention metrics', 'customer_success', 3, 'heart', '#fd7e14'),
				('cat_activity_metrics', 'Activity Metrics', 'Daily activity and productivity metrics', 'productivity', 4, 'activity', '#6f42c1'),
				('cat_financial_metrics', 'Financial Metrics', 'Revenue and financial performance indicators', 'financial', 5, 'dollar-sign', '#20c997'),
				('cat_quality_metrics', 'Quality Metrics', 'Service and delivery quality measurements', 'quality', 6, 'award', '#e83e8c'),
				('cat_efficiency_metrics', 'Efficiency Metrics', 'Process efficiency and optimization metrics', 'efficiency', 7, 'zap', '#ffc107')
			""")
			
			# Insert sample performance benchmarks
			await conn.execute("""
				INSERT INTO crm_performance_benchmarks (
					id, tenant_id, name, description, benchmark_type, metric_name,
					measurement_unit, benchmark_value, target_value, threshold_ranges,
					period_type, data_source, calculation_method, created_by
				) VALUES 
				(
					'bench_monthly_revenue',
					'system',
					'Monthly Revenue Target',
					'Monthly revenue generation benchmark for sales representatives',
					'individual',
					'monthly_revenue',
					'USD',
					50000.00,
					75000.00,
					'{"poor": 25000, "fair": 40000, "good": 60000, "excellent": 80000}',
					'monthly',
					'opportunities',
					'sum_amount',
					'system'
				),
				(
					'bench_lead_conversion',
					'system',
					'Lead Conversion Rate',
					'Lead to opportunity conversion rate benchmark',
					'individual',
					'lead_conversion_rate',
					'percentage',
					15.00,
					20.00,
					'{"poor": 5, "fair": 10, "good": 18, "excellent": 25}',
					'monthly',
					'leads',
					'conversion_rate',
					'system'
				),
				(
					'bench_activity_count',
					'system',
					'Monthly Activity Count',
					'Number of sales activities completed per month',
					'individual',
					'monthly_activities',
					'count',
					100.00,
					150.00,
					'{"poor": 50, "fair": 80, "good": 120, "excellent": 180}',
					'monthly',
					'activities',
					'count',
					'system'
				),
				(
					'bench_deal_closure_time',
					'system',
					'Average Deal Closure Time',
					'Average time to close deals from opportunity creation',
					'individual',
					'avg_deal_closure_days',
					'days',
					45.00,
					30.00,
					'{"excellent": 20, "good": 35, "fair": 50, "poor": 70}',
					'monthly',
					'opportunities',
					'average_days',
					'system'
				)
			""")
			
			logger.info("✅ Performance benchmarking tables created successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to create performance benchmarking tables: {str(e)}")
			raise
	
	async def _execute_down_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the down migration"""
		try:
			logger.info("🔄 Dropping performance benchmarking tables...")
			
			# Drop tables in reverse order
			await conn.execute("DROP TABLE IF EXISTS crm_performance_coaching CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_benchmark_categories CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_performance_alerts CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_performance_reports CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_goal_tracking CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_performance_comparisons CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_performance_metrics CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_performance_benchmarks CASCADE;")
			
			# Drop enums
			await conn.execute("DROP TYPE IF EXISTS goal_type CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS trend_direction CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS performance_rating CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS benchmark_type CASCADE;")
			
			logger.info("✅ Performance benchmarking tables dropped successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to drop performance benchmarking tables: {str(e)}")
			raise