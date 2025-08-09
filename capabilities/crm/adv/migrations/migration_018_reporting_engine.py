"""
APG Customer Relationship Management - Reporting Engine Migration

Database migration to create advanced reporting engine tables and supporting 
structures for custom reports, scheduled executions, and multi-format exports.

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


class ReportingEngineMigration(BaseMigration):
	"""Migration for reporting engine functionality"""
	
	def _get_migration_id(self) -> str:
		return "018_reporting_engine"
	
	def _get_version(self) -> str:
		return "018"
	
	def _get_description(self) -> str:
		return "Advanced reporting engine with custom reports and scheduled executions"
	
	def _get_dependencies(self) -> list[str]:
		return ["017_crm_dashboard"]

	async def _execute_up_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the up migration"""
		try:
			logger.info("🔄 Creating reporting engine tables...")
			
			# Create report type enum
			await conn.execute("""
				CREATE TYPE report_type AS ENUM (
					'tabular',
					'summary',
					'dashboard',
					'chart',
					'crosstab',
					'matrix',
					'subreport',
					'custom'
				);
			""")
			
			# Create report status enum
			await conn.execute("""
				CREATE TYPE report_status AS ENUM (
					'draft',
					'active',
					'scheduled',
					'running',
					'completed',
					'failed',
					'archived'
				);
			""")
			
			# Create export format enum
			await conn.execute("""
				CREATE TYPE export_format AS ENUM (
					'pdf',
					'excel',
					'csv',
					'json',
					'html',
					'png',
					'svg'
				);
			""")
			
			# Create schedule frequency enum
			await conn.execute("""
				CREATE TYPE schedule_frequency AS ENUM (
					'daily',
					'weekly',
					'monthly',
					'quarterly',
					'yearly',
					'custom'
				);
			""")
			
			# Create data aggregation enum
			await conn.execute("""
				CREATE TYPE data_aggregation AS ENUM (
					'sum',
					'count',
					'average',
					'median',
					'min',
					'max',
					'distinct_count',
					'standard_deviation',
					'variance',
					'percentile'
				);
			""")
			
			# Create chart type enum
			await conn.execute("""
				CREATE TYPE chart_type AS ENUM (
					'line',
					'bar',
					'column',
					'pie',
					'donut',
					'area',
					'scatter',
					'bubble',
					'heatmap',
					'treemap',
					'waterfall',
					'gauge',
					'funnel',
					'radar',
					'sankey'
				);
			""")
			
			# Create reports table
			await conn.execute("""
				CREATE TABLE crm_reports (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					name VARCHAR(255) NOT NULL,
					description TEXT,
					report_type report_type NOT NULL,
					status report_status DEFAULT 'draft',
					data_sources JSONB NOT NULL DEFAULT '[]',
					joins JSONB DEFAULT '[]',
					base_query TEXT,
					fields JSONB NOT NULL DEFAULT '[]',
					filters JSONB DEFAULT '[]',
					parameters JSONB DEFAULT '[]',
					visualizations JSONB DEFAULT '[]',
					page_size VARCHAR(10) DEFAULT 'A4',
					orientation VARCHAR(10) DEFAULT 'portrait',
					margins JSONB DEFAULT '{"top": 20, "right": 20, "bottom": 20, "left": 20}',
					header_text TEXT,
					footer_text TEXT,
					logo_url VARCHAR(500),
					row_limit INTEGER DEFAULT 10000,
					timeout_seconds INTEGER DEFAULT 300,
					cache_enabled BOOLEAN DEFAULT true,
					cache_ttl INTEGER DEFAULT 3600,
					owner_id VARCHAR(36) NOT NULL,
					is_public BOOLEAN DEFAULT false,
					shared_with JSONB DEFAULT '[]',
					access_permissions JSONB DEFAULT '{}',
					last_run_at TIMESTAMP WITH TIME ZONE,
					last_run_by VARCHAR(36),
					run_count INTEGER DEFAULT 0,
					avg_execution_time_ms DECIMAL(10,2) DEFAULT 0.00,
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					updated_by VARCHAR(36)
				);
			""")
			
			# Create report executions table
			await conn.execute("""
				CREATE TABLE crm_report_executions (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					report_id VARCHAR(36) NOT NULL REFERENCES crm_reports(id) ON DELETE CASCADE,
					executed_by VARCHAR(36) NOT NULL,
					execution_type VARCHAR(20) NOT NULL,
					parameters JSONB DEFAULT '{}',
					filters JSONB DEFAULT '{}',
					status VARCHAR(20) DEFAULT 'pending',
					started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					completed_at TIMESTAMP WITH TIME ZONE,
					execution_time_ms DECIMAL(10,2),
					row_count INTEGER,
					data_size_bytes INTEGER,
					export_format export_format,
					export_url VARCHAR(500),
					error_message TEXT,
					error_details JSONB,
					query_time_ms DECIMAL(10,2),
					rendering_time_ms DECIMAL(10,2),
					export_time_ms DECIMAL(10,2),
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create report schedules table
			await conn.execute("""
				CREATE TABLE crm_report_schedules (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					report_id VARCHAR(36) NOT NULL REFERENCES crm_reports(id) ON DELETE CASCADE,
					name VARCHAR(255) NOT NULL,
					frequency schedule_frequency NOT NULL,
					cron_expression VARCHAR(100),
					start_date DATE NOT NULL,
					end_date DATE,
					next_run_at TIMESTAMP WITH TIME ZONE,
					export_formats JSONB NOT NULL DEFAULT '[]',
					email_recipients JSONB DEFAULT '[]',
					email_subject VARCHAR(255),
					email_body TEXT,
					save_to_storage BOOLEAN DEFAULT true,
					storage_path VARCHAR(500),
					webhook_url VARCHAR(500),
					is_active BOOLEAN DEFAULT true,
					last_run_at TIMESTAMP WITH TIME ZONE,
					last_run_status VARCHAR(20),
					run_count INTEGER DEFAULT 0,
					success_count INTEGER DEFAULT 0,
					retry_count INTEGER DEFAULT 3,
					retry_delay_minutes INTEGER DEFAULT 15,
					on_failure_action VARCHAR(20) DEFAULT 'email',
					metadata JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL
				);
			""")
			
			# Create report templates table
			await conn.execute("""
				CREATE TABLE crm_report_templates (
					id VARCHAR(36) PRIMARY KEY,
					name VARCHAR(255) NOT NULL,
					description TEXT,
					category VARCHAR(100),
					report_type report_type NOT NULL,
					template_data JSONB NOT NULL,
					preview_image VARCHAR(500),
					is_public BOOLEAN DEFAULT true,
					usage_count INTEGER DEFAULT 0,
					rating DECIMAL(3,2) DEFAULT 0.00,
					tags JSONB DEFAULT '[]',
					created_by VARCHAR(36) NOT NULL,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create report data cache table
			await conn.execute("""
				CREATE TABLE crm_report_cache (
					id VARCHAR(36) PRIMARY KEY,
					report_id VARCHAR(36) NOT NULL REFERENCES crm_reports(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					cache_key VARCHAR(500) NOT NULL,
					parameters_hash VARCHAR(64) NOT NULL,
					data JSONB NOT NULL,
					row_count INTEGER DEFAULT 0,
					data_size_bytes INTEGER DEFAULT 0,
					query_time_ms DECIMAL(10,2) DEFAULT 0.00,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
					access_count INTEGER DEFAULT 0,
					last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					UNIQUE(cache_key, parameters_hash)
				);
			""")
			
			# Create report access log table
			await conn.execute("""
				CREATE TABLE crm_report_access_log (
					id VARCHAR(36) PRIMARY KEY,
					report_id VARCHAR(36) NOT NULL REFERENCES crm_reports(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					user_id VARCHAR(36) NOT NULL,
					user_name VARCHAR(255),
					access_type VARCHAR(20) NOT NULL,
					execution_id VARCHAR(36) REFERENCES crm_report_executions(id),
					ip_address INET,
					user_agent TEXT,
					parameters JSONB DEFAULT '{}',
					execution_time_ms DECIMAL(10,2),
					export_format export_format,
					success BOOLEAN DEFAULT true,
					error_message TEXT,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
				);
			""")
			
			# Create report performance metrics table
			await conn.execute("""
				CREATE TABLE crm_report_performance (
					id VARCHAR(36) PRIMARY KEY,
					report_id VARCHAR(36) NOT NULL REFERENCES crm_reports(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					date_tracked DATE NOT NULL,
					total_executions INTEGER DEFAULT 0,
					successful_executions INTEGER DEFAULT 0,
					failed_executions INTEGER DEFAULT 0,
					avg_execution_time_ms DECIMAL(10,2) DEFAULT 0.00,
					avg_query_time_ms DECIMAL(10,2) DEFAULT 0.00,
					avg_row_count INTEGER DEFAULT 0,
					cache_hit_rate DECIMAL(5,2) DEFAULT 0.00,
					unique_users INTEGER DEFAULT 0,
					total_exports INTEGER DEFAULT 0,
					export_breakdown JSONB DEFAULT '{}',
					error_breakdown JSONB DEFAULT '{}',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					UNIQUE(report_id, date_tracked)
				);
			""")
			
			# Create indexes for performance
			await conn.execute("CREATE INDEX idx_reports_tenant ON crm_reports(tenant_id);")
			await conn.execute("CREATE INDEX idx_reports_type ON crm_reports(report_type);")
			await conn.execute("CREATE INDEX idx_reports_status ON crm_reports(status);")
			await conn.execute("CREATE INDEX idx_reports_owner ON crm_reports(owner_id);")
			await conn.execute("CREATE INDEX idx_reports_public ON crm_reports(is_public);")
			await conn.execute("CREATE INDEX idx_reports_last_run ON crm_reports(last_run_at);")
			
			await conn.execute("CREATE INDEX idx_report_executions_report ON crm_report_executions(report_id);")
			await conn.execute("CREATE INDEX idx_report_executions_tenant ON crm_report_executions(tenant_id);")
			await conn.execute("CREATE INDEX idx_report_executions_user ON crm_report_executions(executed_by);")
			await conn.execute("CREATE INDEX idx_report_executions_status ON crm_report_executions(status);")
			await conn.execute("CREATE INDEX idx_report_executions_date ON crm_report_executions(started_at);")
			
			await conn.execute("CREATE INDEX idx_report_schedules_report ON crm_report_schedules(report_id);")
			await conn.execute("CREATE INDEX idx_report_schedules_tenant ON crm_report_schedules(tenant_id);")
			await conn.execute("CREATE INDEX idx_report_schedules_active ON crm_report_schedules(is_active);")
			await conn.execute("CREATE INDEX idx_report_schedules_next_run ON crm_report_schedules(next_run_at);")
			
			await conn.execute("CREATE INDEX idx_report_templates_type ON crm_report_templates(report_type);")
			await conn.execute("CREATE INDEX idx_report_templates_category ON crm_report_templates(category);")
			await conn.execute("CREATE INDEX idx_report_templates_public ON crm_report_templates(is_public);")
			await conn.execute("CREATE INDEX idx_report_templates_usage ON crm_report_templates(usage_count);")
			
			await conn.execute("CREATE INDEX idx_report_cache_report ON crm_report_cache(report_id);")
			await conn.execute("CREATE INDEX idx_report_cache_tenant ON crm_report_cache(tenant_id);")
			await conn.execute("CREATE INDEX idx_report_cache_expires ON crm_report_cache(expires_at);")
			await conn.execute("CREATE INDEX idx_report_cache_accessed ON crm_report_cache(last_accessed_at);")
			
			await conn.execute("CREATE INDEX idx_report_access_report ON crm_report_access_log(report_id);")
			await conn.execute("CREATE INDEX idx_report_access_user ON crm_report_access_log(user_id);")
			await conn.execute("CREATE INDEX idx_report_access_date ON crm_report_access_log(created_at);")
			await conn.execute("CREATE INDEX idx_report_access_tenant ON crm_report_access_log(tenant_id);")
			
			await conn.execute("CREATE INDEX idx_report_performance_report ON crm_report_performance(report_id);")
			await conn.execute("CREATE INDEX idx_report_performance_date ON crm_report_performance(date_tracked);")
			await conn.execute("CREATE INDEX idx_report_performance_tenant ON crm_report_performance(tenant_id);")
			
			# Insert default report templates
			await conn.execute("""
				INSERT INTO crm_report_templates (
					id, name, description, category, report_type, template_data, is_public, created_by
				) VALUES 
				(
					'rpt_tpl_sales_summary',
					'Sales Summary Report',
					'Comprehensive sales performance summary with key metrics',
					'Sales',
					'summary',
					'{"data_sources": ["crm_opportunities"], "fields": [{"name": "total_revenue", "display_name": "Total Revenue", "data_type": "number", "source_table": "crm_opportunities", "source_column": "amount", "aggregation": "sum"}, {"name": "opportunity_count", "display_name": "Total Opportunities", "data_type": "number", "source_table": "crm_opportunities", "source_column": "id", "aggregation": "count"}], "visualizations": [{"name": "Revenue Trend", "chart_type": "line", "x_axis_field": "created_date", "y_axis_fields": ["amount"]}]}',
					true,
					'system'
				),
				(
					'rpt_tpl_lead_analysis',
					'Lead Analysis Report',
					'Detailed lead performance and conversion analysis',
					'Marketing',
					'tabular',
					'{"data_sources": ["crm_leads"], "fields": [{"name": "lead_source", "display_name": "Lead Source", "data_type": "string", "source_table": "crm_leads", "source_column": "source", "group_by": true}, {"name": "lead_count", "display_name": "Lead Count", "data_type": "number", "source_table": "crm_leads", "source_column": "id", "aggregation": "count"}, {"name": "conversion_rate", "display_name": "Conversion Rate", "data_type": "number", "source_table": "crm_leads", "source_column": "converted", "aggregation": "average"}]}',
					true,
					'system'
				),
				(
					'rpt_tpl_pipeline_analysis',
					'Sales Pipeline Analysis',
					'Pipeline stage analysis with forecasting',
					'Sales',
					'chart',
					'{"data_sources": ["crm_opportunities"], "fields": [{"name": "stage", "display_name": "Pipeline Stage", "data_type": "string", "source_table": "crm_opportunities", "source_column": "stage", "group_by": true}, {"name": "opportunity_count", "display_name": "Opportunity Count", "data_type": "number", "source_table": "crm_opportunities", "source_column": "id", "aggregation": "count"}, {"name": "stage_value", "display_name": "Stage Value", "data_type": "number", "source_table": "crm_opportunities", "source_column": "amount", "aggregation": "sum"}], "visualizations": [{"name": "Pipeline Funnel", "chart_type": "funnel", "series_field": "stage", "value_field": "stage_value"}]}',
					true,
					'system'
				),
				(
					'rpt_tpl_activity_report',
					'Activity Performance Report',
					'User activity tracking and performance metrics',
					'Operations',
					'tabular',
					'{"data_sources": ["crm_activities"], "fields": [{"name": "user_name", "display_name": "User", "data_type": "string", "source_table": "crm_activities", "source_column": "assigned_to_name", "group_by": true}, {"name": "activity_count", "display_name": "Total Activities", "data_type": "number", "source_table": "crm_activities", "source_column": "id", "aggregation": "count"}, {"name": "completed_activities", "display_name": "Completed Activities", "data_type": "number", "source_table": "crm_activities", "source_column": "status", "aggregation": "count"}]}',
					true,
					'system'
				)
			""")
			
			logger.info("✅ Reporting engine tables created successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to create reporting engine tables: {str(e)}")
			raise
	
	async def _execute_down_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the down migration"""
		try:
			logger.info("🔄 Dropping reporting engine tables...")
			
			# Drop tables in reverse order
			await conn.execute("DROP TABLE IF EXISTS crm_report_performance CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_report_access_log CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_report_cache CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_report_templates CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_report_schedules CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_report_executions CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_reports CASCADE;")
			
			# Drop enums
			await conn.execute("DROP TYPE IF EXISTS chart_type CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS data_aggregation CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS schedule_frequency CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS export_format CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS report_status CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS report_type CASCADE;")
			
			logger.info("✅ Reporting engine tables dropped successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to drop reporting engine tables: {str(e)}")
			raise