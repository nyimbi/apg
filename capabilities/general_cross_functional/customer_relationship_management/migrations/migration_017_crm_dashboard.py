"""
APG Customer Relationship Management - CRM Dashboard Migration

Database migration to create comprehensive dashboard tables and supporting 
structures for real-time analytics, AI-powered insights, and interactive visualizations.

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


class CRMDashboardMigration(BaseMigration):
	"""Migration for CRM dashboard functionality"""
	
	def _get_migration_id(self) -> str:
		return "017_crm_dashboard"
	
	def _get_version(self) -> str:
		return "017"
	
	def _get_description(self) -> str:
		return "CRM dashboards with real-time analytics and AI-powered insights"
	
	def _get_dependencies(self) -> list[str]:
		return ["016_lead_nurturing"]

	async def _execute_up_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the up migration"""
		try:
			logger.info("🔄 Creating CRM dashboard tables...")
			
			# Create dashboard type enum
			await conn.execute("""
				CREATE TYPE dashboard_type AS ENUM (
					'executive',
					'sales_manager',
					'sales_rep',
					'marketing',
					'customer_success',
					'operations',
					'custom'
				);
			""")
			
			# Create widget type enum
			await conn.execute("""
				CREATE TYPE widget_type AS ENUM (
					'kpi_card',
					'line_chart',
					'bar_chart',
					'pie_chart',
					'funnel_chart',
					'heatmap',
					'table',
					'leaderboard',
					'activity_feed',
					'pipeline_view',
					'forecast_chart',
					'geographic_map',
					'gauge_chart',
					'waterfall_chart',
					'trend_indicator'
				);
			""")
			
			# Create metric type enum
			await conn.execute("""
				CREATE TYPE metric_type AS ENUM (
					'count',
					'sum',
					'average',
					'percentage',
					'ratio',
					'growth_rate',
					'conversion_rate',
					'velocity',
					'forecast',
					'trend'
				);
			""")
			
			# Create time range enum
			await conn.execute("""
				CREATE TYPE time_range AS ENUM (
					'today',
					'yesterday',
					'this_week',
					'last_week',
					'this_month',
					'last_month',
					'this_quarter',
					'last_quarter',
					'this_year',
					'last_year',
					'last_7_days',
					'last_30_days',
					'last_90_days',
					'last_12_months',
					'custom'
				);
			""")
			
			# Create data source enum
			await conn.execute("""
				CREATE TYPE data_source AS ENUM (
					'contacts',
					'leads',
					'opportunities',
					'accounts',
					'activities',
					'emails',
					'calls',
					'meetings',
					'campaigns',
					'revenue',
					'forecasts',
					'pipeline',
					'assignments',
					'nurturing',
					'approvals',
					'custom_query'
				);
			""")
			
			# Create dashboards table
			await conn.execute("""
				CREATE TABLE crm_dashboards (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					name VARCHAR(255) NOT NULL,
					description TEXT,
					dashboard_type dashboard_type NOT NULL,
					grid_columns INTEGER DEFAULT 12,
					grid_rows INTEGER DEFAULT 20,
					widgets JSONB NOT NULL DEFAULT '[]',
					owner_id VARCHAR(36) NOT NULL,
					is_public BOOLEAN DEFAULT false,
					shared_with JSONB DEFAULT '[]',
					access_level VARCHAR(20) DEFAULT 'view',
					theme VARCHAR(20) DEFAULT 'light',
					background_color VARCHAR(7) DEFAULT '#ffffff',
					auto_refresh BOOLEAN DEFAULT true,
					refresh_interval INTEGER DEFAULT 300,
					filters_enabled BOOLEAN DEFAULT true,
					export_enabled BOOLEAN DEFAULT true,
					drill_down_enabled BOOLEAN DEFAULT true,
					real_time_updates BOOLEAN DEFAULT false,
					cache_enabled BOOLEAN DEFAULT true,
					cache_ttl INTEGER DEFAULT 300,
					lazy_loading BOOLEAN DEFAULT true,
					metadata JSONB NOT NULL DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					updated_by VARCHAR(36)
				);
			""")
			
			# Create dashboard widgets table (separate table for better querying)
			await conn.execute("""
				CREATE TABLE crm_dashboard_widgets (
					id VARCHAR(36) PRIMARY KEY,
					dashboard_id VARCHAR(36) NOT NULL REFERENCES crm_dashboards(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					name VARCHAR(255) NOT NULL,
					description TEXT,
					widget_type widget_type NOT NULL,
					position_x INTEGER DEFAULT 0,
					position_y INTEGER DEFAULT 0,
					width INTEGER DEFAULT 2,
					height INTEGER DEFAULT 2,
					data_source data_source NOT NULL,
					metric_type metric_type NOT NULL,
					time_range time_range DEFAULT 'last_30_days',
					filters JSONB DEFAULT '{}',
					group_by VARCHAR(100),
					sort_by VARCHAR(100),
					sort_order VARCHAR(4) DEFAULT 'desc',
					limit_results INTEGER,
					title VARCHAR(255),
					color_scheme VARCHAR(20) DEFAULT 'blue',
					show_legend BOOLEAN DEFAULT true,
					show_grid BOOLEAN DEFAULT true,
					show_values BOOLEAN DEFAULT true,
					number_format VARCHAR(20) DEFAULT 'auto',
					date_format VARCHAR(20) DEFAULT 'MMM DD',
					decimal_places INTEGER DEFAULT 2,
					is_clickable BOOLEAN DEFAULT false,
					drill_down_url VARCHAR(500),
					refresh_interval INTEGER DEFAULT 300,
					ai_insights_enabled BOOLEAN DEFAULT false,
					predictive_analytics BOOLEAN DEFAULT false,
					alert_thresholds JSONB DEFAULT '{}',
					benchmark_comparisons JSONB DEFAULT '[]',
					custom_query TEXT,
					custom_javascript TEXT,
					custom_css TEXT,
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create dashboard data cache table
			await conn.execute("""
				CREATE TABLE crm_dashboard_cache (
					id VARCHAR(36) PRIMARY KEY,
					dashboard_id VARCHAR(36) NOT NULL REFERENCES crm_dashboards(id) ON DELETE CASCADE,
					widget_id VARCHAR(36) NOT NULL REFERENCES crm_dashboard_widgets(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					cache_key VARCHAR(500) NOT NULL,
					data JSONB NOT NULL,
					labels JSONB DEFAULT '[]',
					datasets JSONB DEFAULT '[]',
					total_records INTEGER DEFAULT 0,
					filtered_records INTEGER DEFAULT 0,
					time_range_start TIMESTAMP WITH TIME ZONE,
					time_range_end TIMESTAMP WITH TIME ZONE,
					query_time_ms DECIMAL(10,2) DEFAULT 0.00,
					cache_hit BOOLEAN DEFAULT false,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
					UNIQUE(cache_key)
				);
			""")
			
			# Create dashboard insights table
			await conn.execute("""
				CREATE TABLE crm_dashboard_insights (
					id VARCHAR(36) PRIMARY KEY,
					dashboard_id VARCHAR(36) NOT NULL REFERENCES crm_dashboards(id) ON DELETE CASCADE,
					widget_id VARCHAR(36) REFERENCES crm_dashboard_widgets(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					insight_type VARCHAR(50) NOT NULL,
					title VARCHAR(255) NOT NULL,
					description TEXT NOT NULL,
					confidence_score DECIMAL(3,2) NOT NULL,
					supporting_data JSONB DEFAULT '{}',
					visualization_data JSONB DEFAULT '{}',
					impact_level VARCHAR(20) NOT NULL,
					recommended_actions JSONB DEFAULT '[]',
					potential_value DECIMAL(15,2),
					is_active BOOLEAN DEFAULT true,
					is_acknowledged BOOLEAN DEFAULT false,
					acknowledged_by VARCHAR(36),
					acknowledged_at TIMESTAMP WITH TIME ZONE,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					expires_at TIMESTAMP WITH TIME ZONE
				);
			""")
			
			# Create dashboard access log table
			await conn.execute("""
				CREATE TABLE crm_dashboard_access_log (
					id VARCHAR(36) PRIMARY KEY,
					dashboard_id VARCHAR(36) NOT NULL REFERENCES crm_dashboards(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					user_id VARCHAR(36) NOT NULL,
					user_name VARCHAR(255),
					access_type VARCHAR(20) NOT NULL,
					ip_address INET,
					user_agent TEXT,
					session_duration_seconds INTEGER,
					widgets_viewed JSONB DEFAULT '[]',
					filters_applied JSONB DEFAULT '{}',
					exports_generated INTEGER DEFAULT 0,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create dashboard performance metrics table
			await conn.execute("""
				CREATE TABLE crm_dashboard_performance (
					id VARCHAR(36) PRIMARY KEY,
					dashboard_id VARCHAR(36) NOT NULL REFERENCES crm_dashboards(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					date_tracked DATE NOT NULL,
					total_views INTEGER DEFAULT 0,
					unique_viewers INTEGER DEFAULT 0,
					avg_session_duration_seconds DECIMAL(10,2) DEFAULT 0.00,
					avg_load_time_ms DECIMAL(10,2) DEFAULT 0.00,
					cache_hit_rate DECIMAL(5,2) DEFAULT 0.00,
					error_rate DECIMAL(5,2) DEFAULT 0.00,
					widget_performance JSONB DEFAULT '{}',
					user_engagement JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					UNIQUE(dashboard_id, date_tracked)
				);
			""")
			
			# Create dashboard templates table
			await conn.execute("""
				CREATE TABLE crm_dashboard_templates (
					id VARCHAR(36) PRIMARY KEY,
					name VARCHAR(255) NOT NULL,
					description TEXT,
					dashboard_type dashboard_type NOT NULL,
					template_data JSONB NOT NULL,
					preview_image VARCHAR(500),
					is_public BOOLEAN DEFAULT true,
					usage_count INTEGER DEFAULT 0,
					rating DECIMAL(3,2) DEFAULT 0.00,
					created_by VARCHAR(36) NOT NULL,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create indexes for performance
			await conn.execute("CREATE INDEX idx_dashboards_tenant ON crm_dashboards(tenant_id);")
			await conn.execute("CREATE INDEX idx_dashboards_type ON crm_dashboards(dashboard_type);")
			await conn.execute("CREATE INDEX idx_dashboards_owner ON crm_dashboards(owner_id);")
			await conn.execute("CREATE INDEX idx_dashboards_public ON crm_dashboards(is_public);")
			
			await conn.execute("CREATE INDEX idx_dashboard_widgets_dashboard ON crm_dashboard_widgets(dashboard_id);")
			await conn.execute("CREATE INDEX idx_dashboard_widgets_tenant ON crm_dashboard_widgets(tenant_id);")
			await conn.execute("CREATE INDEX idx_dashboard_widgets_type ON crm_dashboard_widgets(widget_type);")
			await conn.execute("CREATE INDEX idx_dashboard_widgets_source ON crm_dashboard_widgets(data_source);")
			
			await conn.execute("CREATE INDEX idx_dashboard_cache_dashboard ON crm_dashboard_cache(dashboard_id);")
			await conn.execute("CREATE INDEX idx_dashboard_cache_widget ON crm_dashboard_cache(widget_id);")
			await conn.execute("CREATE INDEX idx_dashboard_cache_expires ON crm_dashboard_cache(expires_at);")
			await conn.execute("CREATE INDEX idx_dashboard_cache_tenant ON crm_dashboard_cache(tenant_id);")
			
			await conn.execute("CREATE INDEX idx_dashboard_insights_dashboard ON crm_dashboard_insights(dashboard_id);")
			await conn.execute("CREATE INDEX idx_dashboard_insights_widget ON crm_dashboard_insights(widget_id);")
			await conn.execute("CREATE INDEX idx_dashboard_insights_tenant ON crm_dashboard_insights(tenant_id);")
			await conn.execute("CREATE INDEX idx_dashboard_insights_active ON crm_dashboard_insights(is_active);")
			await conn.execute("CREATE INDEX idx_dashboard_insights_type ON crm_dashboard_insights(insight_type);")
			
			await conn.execute("CREATE INDEX idx_dashboard_access_dashboard ON crm_dashboard_access_log(dashboard_id);")
			await conn.execute("CREATE INDEX idx_dashboard_access_user ON crm_dashboard_access_log(user_id);")
			await conn.execute("CREATE INDEX idx_dashboard_access_date ON crm_dashboard_access_log(created_at);")
			await conn.execute("CREATE INDEX idx_dashboard_access_tenant ON crm_dashboard_access_log(tenant_id);")
			
			await conn.execute("CREATE INDEX idx_dashboard_performance_dashboard ON crm_dashboard_performance(dashboard_id);")
			await conn.execute("CREATE INDEX idx_dashboard_performance_date ON crm_dashboard_performance(date_tracked);")
			await conn.execute("CREATE INDEX idx_dashboard_performance_tenant ON crm_dashboard_performance(tenant_id);")
			
			await conn.execute("CREATE INDEX idx_dashboard_templates_type ON crm_dashboard_templates(dashboard_type);")
			await conn.execute("CREATE INDEX idx_dashboard_templates_public ON crm_dashboard_templates(is_public);")
			await conn.execute("CREATE INDEX idx_dashboard_templates_usage ON crm_dashboard_templates(usage_count);")
			
			# Insert default dashboard templates
			await conn.execute("""
				INSERT INTO crm_dashboard_templates (
					id, name, description, dashboard_type, template_data, is_public, created_by
				) VALUES 
				(
					'tpl_executive_01',
					'Executive Overview',
					'High-level metrics for executive leadership',
					'executive',
					'{"widgets": [{"name": "Total Revenue", "widget_type": "kpi_card", "data_source": "revenue", "metric_type": "sum", "position_x": 0, "position_y": 0, "width": 3, "height": 2}, {"name": "Active Opportunities", "widget_type": "kpi_card", "data_source": "opportunities", "metric_type": "count", "position_x": 3, "position_y": 0, "width": 3, "height": 2}, {"name": "Sales Pipeline", "widget_type": "funnel_chart", "data_source": "pipeline", "metric_type": "count", "position_x": 0, "position_y": 2, "width": 6, "height": 4}]}',
					true,
					'system'
				),
				(
					'tpl_sales_manager_01',
					'Sales Manager Dashboard',
					'Team performance and pipeline management',
					'sales_manager',
					'{"widgets": [{"name": "Team Performance", "widget_type": "bar_chart", "data_source": "opportunities", "metric_type": "sum", "group_by": "assigned_to", "position_x": 0, "position_y": 0, "width": 6, "height": 3}, {"name": "Pipeline by Stage", "widget_type": "pipeline_view", "data_source": "pipeline", "metric_type": "count", "position_x": 6, "position_y": 0, "width": 6, "height": 3}]}',
					true,
					'system'
				),
				(
					'tpl_sales_rep_01',
					'Sales Rep Dashboard',
					'Individual performance and activity tracking',
					'sales_rep',
					'{"widgets": [{"name": "My Opportunities", "widget_type": "table", "data_source": "opportunities", "metric_type": "count", "position_x": 0, "position_y": 0, "width": 8, "height": 4}, {"name": "Activity Summary", "widget_type": "pie_chart", "data_source": "activities", "metric_type": "count", "group_by": "activity_type", "position_x": 8, "position_y": 0, "width": 4, "height": 4}]}',
					true,
					'system'
				)
			""")
			
			logger.info("✅ CRM dashboard tables created successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to create CRM dashboard tables: {str(e)}")
			raise
	
	async def _execute_down_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the down migration"""
		try:
			logger.info("🔄 Dropping CRM dashboard tables...")
			
			# Drop tables in reverse order
			await conn.execute("DROP TABLE IF EXISTS crm_dashboard_templates CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_dashboard_performance CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_dashboard_access_log CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_dashboard_insights CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_dashboard_cache CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_dashboard_widgets CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_dashboards CASCADE;")
			
			# Drop enums
			await conn.execute("DROP TYPE IF EXISTS data_source CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS time_range CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS metric_type CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS widget_type CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS dashboard_type CASCADE;")
			
			logger.info("✅ CRM dashboard tables dropped successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to drop CRM dashboard tables: {str(e)}")
			raise