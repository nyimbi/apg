"""
APG Customer Relationship Management - Email Integration Migration

Database migration to create email integration tables and supporting structures
for advanced email management, tracking, and analytics capabilities.

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


class EmailIntegrationMigration(BaseMigration):
	"""Migration for email integration functionality"""
	
	def _get_migration_id(self) -> str:
		return "012_email_integration"
	
	def _get_version(self) -> str:
		return "012"
	
	def _get_description(self) -> str:
		return "Create email integration tables and supporting structures"
	
	def _get_dependencies(self) -> list:
		return ["001_initial_schema"]
	
	def _is_reversible(self) -> bool:
		return True
	
	async def up(self, connection) -> None:
		"""Apply the migration"""
		try:
			logger.info("🔄 Creating email integration structures...")
			
			# Create email provider enum
			await connection.execute("""
				CREATE TYPE crm_email_provider AS ENUM (
					'smtp', 'gmail', 'outlook', 'exchange', 'sendgrid', 'mailgun', 'aws_ses', 'custom'
				)
			""")
			
			# Create email type enum
			await connection.execute("""
				CREATE TYPE crm_email_type AS ENUM (
					'outbound', 'inbound', 'template', 'campaign', 'automated', 'reply', 'forward'
				)
			""")
			
			# Create email status enum
			await connection.execute("""
				CREATE TYPE crm_email_status AS ENUM (
					'draft', 'queued', 'sending', 'sent', 'delivered', 'opened', 'clicked', 
					'replied', 'bounced', 'failed', 'spam', 'unsubscribed'
				)
			""")
			
			# Create email priority enum
			await connection.execute("""
				CREATE TYPE crm_email_priority AS ENUM (
					'low', 'normal', 'high', 'urgent'
				)
			""")
			
			# Create email templates table
			await connection.execute("""
				CREATE TABLE crm_email_templates (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					
					-- Template information
					name TEXT NOT NULL,
					description TEXT,
					category TEXT DEFAULT 'general',
					tags TEXT[] DEFAULT '{}',
					
					-- Template content
					subject TEXT NOT NULL,
					html_content TEXT,
					text_content TEXT,
					
					-- Template settings
					is_active BOOLEAN DEFAULT true,
					is_default BOOLEAN DEFAULT false,
					
					-- Personalization
					merge_fields TEXT[] DEFAULT '{}',
					required_fields TEXT[] DEFAULT '{}',
					dynamic_content JSONB DEFAULT '{}',
					
					-- Tracking settings
					enable_tracking BOOLEAN DEFAULT true,
					track_opens BOOLEAN DEFAULT true,
					track_clicks BOOLEAN DEFAULT true,
					track_downloads BOOLEAN DEFAULT false,
					
					-- Usage statistics
					usage_count INTEGER DEFAULT 0,
					success_rate DECIMAL(5,2) DEFAULT 0,
					average_open_rate DECIMAL(5,2) DEFAULT 0,
					average_click_rate DECIMAL(5,2) DEFAULT 0,
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					version INTEGER DEFAULT 1,
					
					-- Constraints
					CONSTRAINT check_template_name_length CHECK (char_length(name) >= 1 AND char_length(name) <= 200),
					CONSTRAINT check_template_description_length CHECK (char_length(description) <= 1000),
					CONSTRAINT check_template_category_length CHECK (char_length(category) <= 50),
					CONSTRAINT check_template_subject_length CHECK (char_length(subject) >= 1),
					CONSTRAINT check_usage_count_positive CHECK (usage_count >= 0),
					CONSTRAINT check_success_rate_range CHECK (success_rate >= 0 AND success_rate <= 100),
					CONSTRAINT check_open_rate_range CHECK (average_open_rate >= 0 AND average_open_rate <= 100),
					CONSTRAINT check_click_rate_range CHECK (average_click_rate >= 0 AND average_click_rate <= 100)
				)
			""")
			
			# Create email messages table
			await connection.execute("""
				CREATE TABLE crm_email_messages (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					
					-- Message identification
					message_id TEXT,
					thread_id TEXT,
					conversation_id TEXT,
					
					-- Email details
					email_type crm_email_type NOT NULL,
					subject TEXT NOT NULL,
					from_email TEXT NOT NULL,
					from_name TEXT,
					to_emails TEXT[] NOT NULL,
					cc_emails TEXT[] DEFAULT '{}',
					bcc_emails TEXT[] DEFAULT '{}',
					reply_to TEXT,
					
					-- Content
					html_content TEXT,
					text_content TEXT,
					attachments JSONB DEFAULT '[]',
					
					-- Status and tracking
					status crm_email_status DEFAULT 'draft',
					priority crm_email_priority DEFAULT 'normal',
					scheduled_at TIMESTAMP WITH TIME ZONE,
					sent_at TIMESTAMP WITH TIME ZONE,
					delivered_at TIMESTAMP WITH TIME ZONE,
					
					-- CRM relationships
					contact_id TEXT,
					account_id TEXT,
					lead_id TEXT,
					opportunity_id TEXT,
					campaign_id TEXT,
					
					-- Template and automation
					template_id TEXT,
					workflow_id TEXT,
					is_automated BOOLEAN DEFAULT false,
					
					-- Tracking data
					tracking_enabled BOOLEAN DEFAULT true,
					tracking_pixel_url TEXT,
					click_tracking_urls JSONB DEFAULT '{}',
					
					-- Metrics
					open_count INTEGER DEFAULT 0,
					click_count INTEGER DEFAULT 0,
					reply_count INTEGER DEFAULT 0,
					forward_count INTEGER DEFAULT 0,
					
					-- Email provider details
					provider crm_email_provider DEFAULT 'smtp',
					provider_message_id TEXT,
					provider_metadata JSONB DEFAULT '{}',
					
					-- Error handling
					send_attempts INTEGER DEFAULT 0,
					last_error TEXT,
					error_details JSONB DEFAULT '{}',
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					
					-- Foreign key constraints
					FOREIGN KEY (template_id) REFERENCES crm_email_templates(id) ON DELETE SET NULL,
					FOREIGN KEY (contact_id) REFERENCES crm_contacts(id) ON DELETE SET NULL,
					FOREIGN KEY (account_id) REFERENCES crm_accounts(id) ON DELETE SET NULL,
					FOREIGN KEY (lead_id) REFERENCES crm_leads(id) ON DELETE SET NULL,
					FOREIGN KEY (opportunity_id) REFERENCES crm_opportunities(id) ON DELETE SET NULL,
					
					-- Constraints
					CONSTRAINT check_subject_length CHECK (char_length(subject) >= 1),
					CONSTRAINT check_from_email_format CHECK (from_email ~ '^[^@]+@[^@]+\.[^@]+$'),
					CONSTRAINT check_to_emails_not_empty CHECK (array_length(to_emails, 1) > 0),
					CONSTRAINT check_engagement_counters_positive CHECK (
						open_count >= 0 AND 
						click_count >= 0 AND 
						reply_count >= 0 AND 
						forward_count >= 0
					),
					CONSTRAINT check_send_attempts_positive CHECK (send_attempts >= 0),
					CONSTRAINT check_scheduled_sent_order CHECK (
						scheduled_at IS NULL OR sent_at IS NULL OR sent_at >= scheduled_at
					),
					CONSTRAINT check_sent_delivered_order CHECK (
						sent_at IS NULL OR delivered_at IS NULL OR delivered_at >= sent_at
					)
				)
			""")
			
			# Create email tracking table
			await connection.execute("""
				CREATE TABLE crm_email_tracking (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					email_id TEXT NOT NULL,
					
					-- Event details
					event_type TEXT NOT NULL,
					event_data JSONB DEFAULT '{}',
					
					-- User information
					user_agent TEXT,
					ip_address TEXT,
					location_data JSONB DEFAULT '{}',
					device_info JSONB DEFAULT '{}',
					
					-- URL and link tracking
					clicked_url TEXT,
					link_text TEXT,
					link_position INTEGER,
					
					-- Timing
					event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					time_since_sent INTEGER,
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Foreign key constraints
					FOREIGN KEY (email_id) REFERENCES crm_email_messages(id) ON DELETE CASCADE,
					
					-- Constraints
					CONSTRAINT check_event_type_length CHECK (char_length(event_type) >= 1),
					CONSTRAINT check_link_position_positive CHECK (link_position IS NULL OR link_position > 0),
					CONSTRAINT check_time_since_sent_positive CHECK (time_since_sent IS NULL OR time_since_sent >= 0)
				)
			""")
			
			# Create email configurations table
			await connection.execute("""
				CREATE TABLE crm_email_configurations (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					
					-- Configuration details
					name TEXT NOT NULL,
					provider crm_email_provider NOT NULL,
					is_default BOOLEAN DEFAULT false,
					is_active BOOLEAN DEFAULT true,
					
					-- SMTP/Provider settings
					smtp_host TEXT,
					smtp_port INTEGER,
					smtp_username TEXT,
					smtp_password TEXT,
					use_tls BOOLEAN DEFAULT true,
					use_ssl BOOLEAN DEFAULT false,
					
					-- API settings for cloud providers
					api_key TEXT,
					api_secret TEXT,
					api_endpoint TEXT,
					
					-- Default settings
					default_from_email TEXT,
					default_from_name TEXT,
					default_reply_to TEXT,
					
					-- Limits and quotas
					daily_send_limit INTEGER,
					hourly_send_limit INTEGER,
					current_daily_count INTEGER DEFAULT 0,
					current_hourly_count INTEGER DEFAULT 0,
					
					-- Features
					supports_html BOOLEAN DEFAULT true,
					supports_attachments BOOLEAN DEFAULT true,
					supports_tracking BOOLEAN DEFAULT true,
					supports_templates BOOLEAN DEFAULT true,
					
					-- Configuration data
					configuration JSONB DEFAULT '{}',
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					
					-- Constraints
					CONSTRAINT check_config_name_length CHECK (char_length(name) >= 1 AND char_length(name) <= 100),
					CONSTRAINT check_smtp_port_range CHECK (smtp_port IS NULL OR (smtp_port > 0 AND smtp_port <= 65535)),
					CONSTRAINT check_send_limits_positive CHECK (
						(daily_send_limit IS NULL OR daily_send_limit > 0) AND
						(hourly_send_limit IS NULL OR hourly_send_limit > 0)
					),
					CONSTRAINT check_current_counts_positive CHECK (
						current_daily_count >= 0 AND current_hourly_count >= 0
					),
					CONSTRAINT check_default_from_email_format CHECK (
						default_from_email IS NULL OR default_from_email ~ '^[^@]+@[^@]+\.[^@]+$'
					)
				)
			""")
			
			# Create email analytics table
			await connection.execute("""
				CREATE TABLE crm_email_analytics (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					
					-- Time period
					period_start TIMESTAMP WITH TIME ZONE NOT NULL,
					period_end TIMESTAMP WITH TIME ZONE NOT NULL,
					period_type TEXT NOT NULL, -- 'hour', 'day', 'week', 'month'
					
					-- Overall metrics
					total_emails_sent INTEGER DEFAULT 0,
					total_emails_delivered INTEGER DEFAULT 0,
					total_emails_opened INTEGER DEFAULT 0,
					total_emails_clicked INTEGER DEFAULT 0,
					total_emails_replied INTEGER DEFAULT 0,
					total_emails_bounced INTEGER DEFAULT 0,
					total_emails_unsubscribed INTEGER DEFAULT 0,
					
					-- Rates
					delivery_rate DECIMAL(5,2) DEFAULT 0,
					open_rate DECIMAL(5,2) DEFAULT 0,
					click_rate DECIMAL(5,2) DEFAULT 0,
					reply_rate DECIMAL(5,2) DEFAULT 0,
					bounce_rate DECIMAL(5,2) DEFAULT 0,
					unsubscribe_rate DECIMAL(5,2) DEFAULT 0,
					
					-- Engagement metrics
					unique_opens INTEGER DEFAULT 0,
					unique_clicks INTEGER DEFAULT 0,
					average_time_to_open DECIMAL(10,2) DEFAULT 0,
					average_time_to_click DECIMAL(10,2) DEFAULT 0,
					
					-- Segmentation data
					segment_performance JSONB DEFAULT '{}',
					geographic_performance JSONB DEFAULT '{}',
					device_performance JSONB DEFAULT '{}',
					
					-- Content analysis
					top_subjects JSONB DEFAULT '[]',
					top_links JSONB DEFAULT '[]',
					template_performance JSONB DEFAULT '{}',
					
					-- Temporal patterns
					hourly_engagement JSONB DEFAULT '{}',
					daily_engagement JSONB DEFAULT '{}',
					
					-- Trends
					engagement_trend DECIMAL(5,2) DEFAULT 0,
					delivery_trend DECIMAL(5,2) DEFAULT 0,
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					
					-- Constraints
					CONSTRAINT check_period_valid CHECK (period_end > period_start),
					CONSTRAINT check_period_type CHECK (period_type IN ('hour', 'day', 'week', 'month')),
					CONSTRAINT check_email_counters_positive CHECK (
						total_emails_sent >= 0 AND 
						total_emails_delivered >= 0 AND 
						total_emails_opened >= 0 AND 
						total_emails_clicked >= 0 AND 
						total_emails_replied >= 0 AND 
						total_emails_bounced >= 0 AND
						total_emails_unsubscribed >= 0
					),
					CONSTRAINT check_rates_range CHECK (
						delivery_rate >= 0 AND delivery_rate <= 100 AND
						open_rate >= 0 AND open_rate <= 100 AND
						click_rate >= 0 AND click_rate <= 100 AND
						reply_rate >= 0 AND reply_rate <= 100 AND
						bounce_rate >= 0 AND bounce_rate <= 100 AND
						unsubscribe_rate >= 0 AND unsubscribe_rate <= 100
					),
					CONSTRAINT check_engagement_metrics_positive CHECK (
						unique_opens >= 0 AND 
						unique_clicks >= 0 AND 
						average_time_to_open >= 0 AND 
						average_time_to_click >= 0
					)
				)
			""")
			
			# Create indexes for email templates table
			await connection.execute("""
				CREATE INDEX idx_crm_email_templates_tenant 
				ON crm_email_templates(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_templates_category 
				ON crm_email_templates(category)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_templates_active 
				ON crm_email_templates(is_active)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_templates_default 
				ON crm_email_templates(is_default)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_templates_created_at 
				ON crm_email_templates(created_at)
			""")
			
			# Create GIN indexes for templates arrays and JSONB
			await connection.execute("""
				CREATE INDEX idx_crm_email_templates_tags 
				ON crm_email_templates USING GIN (tags)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_templates_merge_fields 
				ON crm_email_templates USING GIN (merge_fields)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_templates_metadata 
				ON crm_email_templates USING GIN (metadata)
			""")
			
			# Create composite indexes for templates
			await connection.execute("""
				CREATE INDEX idx_crm_email_templates_tenant_category 
				ON crm_email_templates(tenant_id, category)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_templates_tenant_active 
				ON crm_email_templates(tenant_id, is_active)
			""")
			
			# Ensure only one default template per category per tenant
			await connection.execute("""
				CREATE UNIQUE INDEX idx_crm_email_templates_unique_default 
				ON crm_email_templates(tenant_id, category)
				WHERE is_default = true
			""")
			
			# Create indexes for email messages table
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_tenant 
				ON crm_email_messages(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_message_id 
				ON crm_email_messages(message_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_thread_id 
				ON crm_email_messages(thread_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_conversation_id 
				ON crm_email_messages(conversation_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_email_type 
				ON crm_email_messages(email_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_status 
				ON crm_email_messages(status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_priority 
				ON crm_email_messages(priority)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_from_email 
				ON crm_email_messages(from_email)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_scheduled_at 
				ON crm_email_messages(scheduled_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_sent_at 
				ON crm_email_messages(sent_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_created_at 
				ON crm_email_messages(created_at)
			""")
			
			# Create indexes for CRM relationships
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_contact_id 
				ON crm_email_messages(contact_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_account_id 
				ON crm_email_messages(account_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_lead_id 
				ON crm_email_messages(lead_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_opportunity_id 
				ON crm_email_messages(opportunity_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_campaign_id 
				ON crm_email_messages(campaign_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_template_id 
				ON crm_email_messages(template_id)
			""")
			
			# Create GIN indexes for emails arrays and JSONB
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_to_emails 
				ON crm_email_messages USING GIN (to_emails)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_cc_emails 
				ON crm_email_messages USING GIN (cc_emails)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_click_tracking_urls 
				ON crm_email_messages USING GIN (click_tracking_urls)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_provider_metadata 
				ON crm_email_messages USING GIN (provider_metadata)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_metadata 
				ON crm_email_messages USING GIN (metadata)
			""")
			
			# Create composite indexes for messages
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_tenant_status 
				ON crm_email_messages(tenant_id, status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_tenant_sent_at 
				ON crm_email_messages(tenant_id, sent_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_messages_contact_sent_at 
				ON crm_email_messages(contact_id, sent_at)
			""")
			
			# Create indexes for email tracking table
			await connection.execute("""
				CREATE INDEX idx_crm_email_tracking_tenant 
				ON crm_email_tracking(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_tracking_email_id 
				ON crm_email_tracking(email_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_tracking_event_type 
				ON crm_email_tracking(event_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_tracking_event_timestamp 
				ON crm_email_tracking(event_timestamp)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_tracking_clicked_url 
				ON crm_email_tracking(clicked_url)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_tracking_ip_address 
				ON crm_email_tracking(ip_address)
			""")
			
			# Create GIN indexes for tracking JSONB fields
			await connection.execute("""
				CREATE INDEX idx_crm_email_tracking_event_data 
				ON crm_email_tracking USING GIN (event_data)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_tracking_location_data 
				ON crm_email_tracking USING GIN (location_data)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_tracking_device_info 
				ON crm_email_tracking USING GIN (device_info)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_tracking_metadata 
				ON crm_email_tracking USING GIN (metadata)
			""")
			
			# Create composite indexes for tracking
			await connection.execute("""
				CREATE INDEX idx_crm_email_tracking_tenant_event_type 
				ON crm_email_tracking(tenant_id, event_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_tracking_email_event_timestamp 
				ON crm_email_tracking(email_id, event_timestamp)
			""")
			
			# Create indexes for email configurations table
			await connection.execute("""
				CREATE INDEX idx_crm_email_configurations_tenant 
				ON crm_email_configurations(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_configurations_provider 
				ON crm_email_configurations(provider)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_configurations_active 
				ON crm_email_configurations(is_active)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_configurations_default 
				ON crm_email_configurations(is_default)
			""")
			
			# Create composite indexes for configurations
			await connection.execute("""
				CREATE INDEX idx_crm_email_configurations_tenant_active 
				ON crm_email_configurations(tenant_id, is_active)
			""")
			
			# Ensure only one default configuration per tenant
			await connection.execute("""
				CREATE UNIQUE INDEX idx_crm_email_configurations_unique_default 
				ON crm_email_configurations(tenant_id)
				WHERE is_default = true
			""")
			
			# Create indexes for email analytics table
			await connection.execute("""
				CREATE INDEX idx_crm_email_analytics_tenant 
				ON crm_email_analytics(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_analytics_period 
				ON crm_email_analytics(period_start, period_end)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_analytics_period_type 
				ON crm_email_analytics(period_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_email_analytics_created_at 
				ON crm_email_analytics(created_at)
			""")
			
			# Create composite indexes for analytics
			await connection.execute("""
				CREATE INDEX idx_crm_email_analytics_tenant_period 
				ON crm_email_analytics(tenant_id, period_start)
			""")
			
			# Create triggers for updating updated_at timestamps
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_email_templates_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_email_templates_updated_at
					BEFORE UPDATE ON crm_email_templates
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_email_templates_updated_at()
			""")
			
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_email_messages_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_email_messages_updated_at
					BEFORE UPDATE ON crm_email_messages
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_email_messages_updated_at()
			""")
			
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_email_configurations_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_email_configurations_updated_at
					BEFORE UPDATE ON crm_email_configurations
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_email_configurations_updated_at()
			""")
			
			# Create view for email performance summary
			await connection.execute("""
				CREATE VIEW crm_email_performance AS
				SELECT 
					em.tenant_id,
					DATE_TRUNC('day', em.sent_at) as send_date,
					
					-- Overall metrics
					COUNT(*) as total_sent,
					COUNT(*) FILTER (WHERE em.status = 'delivered') as delivered,
					COUNT(*) FILTER (WHERE em.open_count > 0) as opened,
					COUNT(*) FILTER (WHERE em.click_count > 0) as clicked,
					COUNT(*) FILTER (WHERE em.reply_count > 0) as replied,
					COUNT(*) FILTER (WHERE em.status = 'bounced') as bounced,
					
					-- Rates
					CASE 
						WHEN COUNT(*) > 0 THEN
							(COUNT(*) FILTER (WHERE em.status = 'delivered')::DECIMAL / COUNT(*)) * 100
						ELSE 0
					END as delivery_rate,
					
					CASE 
						WHEN COUNT(*) FILTER (WHERE em.status = 'delivered') > 0 THEN
							(COUNT(*) FILTER (WHERE em.open_count > 0)::DECIMAL / COUNT(*) FILTER (WHERE em.status = 'delivered')) * 100
						ELSE 0
					END as open_rate,
					
					CASE 
						WHEN COUNT(*) FILTER (WHERE em.status = 'delivered') > 0 THEN
							(COUNT(*) FILTER (WHERE em.click_count > 0)::DECIMAL / COUNT(*) FILTER (WHERE em.status = 'delivered')) * 100
						ELSE 0
					END as click_rate,
					
					-- Engagement metrics
					SUM(em.open_count) as total_opens,
					SUM(em.click_count) as total_clicks,
					AVG(em.open_count) as avg_opens_per_email,
					AVG(em.click_count) as avg_clicks_per_email
					
				FROM crm_email_messages em
				WHERE em.sent_at IS NOT NULL
				AND em.sent_at >= NOW() - INTERVAL '90 days'
				GROUP BY em.tenant_id, DATE_TRUNC('day', em.sent_at)
				ORDER BY send_date DESC
			""")
			
			# Create view for template performance
			await connection.execute("""
				CREATE VIEW crm_template_performance AS
				SELECT 
					et.id as template_id,
					et.tenant_id,
					et.name as template_name,
					et.category,
					
					-- Usage metrics
					COUNT(em.id) as emails_sent,
					COUNT(em.id) FILTER (WHERE em.open_count > 0) as emails_opened,
					COUNT(em.id) FILTER (WHERE em.click_count > 0) as emails_clicked,
					COUNT(em.id) FILTER (WHERE em.reply_count > 0) as emails_replied,
					
					-- Performance rates
					CASE 
						WHEN COUNT(em.id) > 0 THEN
							(COUNT(em.id) FILTER (WHERE em.open_count > 0)::DECIMAL / COUNT(em.id)) * 100
						ELSE 0
					END as open_rate,
					
					CASE 
						WHEN COUNT(em.id) > 0 THEN
							(COUNT(em.id) FILTER (WHERE em.click_count > 0)::DECIMAL / COUNT(em.id)) * 100
						ELSE 0
					END as click_rate,
					
					CASE 
						WHEN COUNT(em.id) > 0 THEN
							(COUNT(em.id) FILTER (WHERE em.reply_count > 0)::DECIMAL / COUNT(em.id)) * 100
						ELSE 0
					END as reply_rate,
					
					-- Recent activity
					MAX(em.sent_at) as last_used_at,
					COUNT(em.id) FILTER (WHERE em.sent_at >= NOW() - INTERVAL '30 days') as recent_usage
					
				FROM crm_email_templates et
				LEFT JOIN crm_email_messages em ON et.id = em.template_id
					AND em.sent_at IS NOT NULL
				GROUP BY et.id, et.tenant_id, et.name, et.category
			""")
			
			# Create function for calculating email engagement scores
			await connection.execute("""
				CREATE OR REPLACE FUNCTION calculate_email_engagement_score(
					tenant_filter TEXT,
					days_back INTEGER DEFAULT 30
				)
				RETURNS TABLE(
					period_date DATE,
					engagement_score DECIMAL,
					total_sent INTEGER,
					total_engaged INTEGER
				) AS $$
				BEGIN
					RETURN QUERY
					SELECT 
						DATE_TRUNC('day', em.sent_at)::DATE as period_date,
						CASE 
							WHEN COUNT(*) > 0 THEN
								((COUNT(*) FILTER (WHERE em.open_count > 0) * 1.0 +
								  COUNT(*) FILTER (WHERE em.click_count > 0) * 2.0 +
								  COUNT(*) FILTER (WHERE em.reply_count > 0) * 3.0) / 
								 COUNT(*)) * 100
							ELSE 0.0
						END as engagement_score,
						COUNT(*)::INTEGER as total_sent,
						(COUNT(*) FILTER (WHERE em.open_count > 0 OR em.click_count > 0 OR em.reply_count > 0))::INTEGER as total_engaged
					FROM crm_email_messages em
					WHERE em.tenant_id = tenant_filter
					AND em.sent_at >= NOW() - (days_back || ' days')::INTERVAL
					AND em.sent_at IS NOT NULL
					GROUP BY DATE_TRUNC('day', em.sent_at)
					ORDER BY period_date DESC;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			# Create function for identifying top email performers
			await connection.execute("""
				CREATE OR REPLACE FUNCTION identify_top_email_performers(
					tenant_filter TEXT,
					days_back INTEGER DEFAULT 30,
					limit_results INTEGER DEFAULT 10
				)
				RETURNS TABLE(
					email_id TEXT,
					subject TEXT,
					template_name TEXT,
					sent_at TIMESTAMP WITH TIME ZONE,
					total_engagement INTEGER,
					engagement_score DECIMAL
				) AS $$
				BEGIN
					RETURN QUERY
					SELECT 
						em.id as email_id,
						em.subject,
						COALESCE(et.name, 'No Template') as template_name,
						em.sent_at,
						(em.open_count + em.click_count + em.reply_count)::INTEGER as total_engagement,
						((em.open_count * 1.0 + em.click_count * 2.0 + em.reply_count * 3.0))::DECIMAL as engagement_score
					FROM crm_email_messages em
					LEFT JOIN crm_email_templates et ON em.template_id = et.id
					WHERE em.tenant_id = tenant_filter
					AND em.sent_at >= NOW() - (days_back || ' days')::INTERVAL
					AND em.sent_at IS NOT NULL
					ORDER BY engagement_score DESC, total_engagement DESC
					LIMIT limit_results;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			logger.info("✅ Email integration structures created successfully")
			
		except Exception as e:
			logger.error(f"Failed to create email integration structures: {str(e)}", exc_info=True)
			raise
	
	async def down(self, connection) -> None:
		"""Rollback the migration"""
		try:
			logger.info("🔄 Rolling back email integration migration...")
			
			# Drop functions
			await connection.execute("DROP FUNCTION IF EXISTS identify_top_email_performers CASCADE")
			await connection.execute("DROP FUNCTION IF EXISTS calculate_email_engagement_score CASCADE")
			
			# Drop views
			await connection.execute("DROP VIEW IF EXISTS crm_template_performance CASCADE")
			await connection.execute("DROP VIEW IF EXISTS crm_email_performance CASCADE")
			
			# Drop triggers and functions
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_email_configurations_updated_at ON crm_email_configurations")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_email_configurations_updated_at CASCADE")
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_email_messages_updated_at ON crm_email_messages")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_email_messages_updated_at CASCADE")
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_email_templates_updated_at ON crm_email_templates")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_email_templates_updated_at CASCADE")
			
			# Drop tables (will cascade to indexes)
			await connection.execute("DROP TABLE IF EXISTS crm_email_analytics CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_email_configurations CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_email_tracking CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_email_messages CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_email_templates CASCADE")
			
			# Drop enums
			await connection.execute("DROP TYPE IF EXISTS crm_email_priority CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_email_status CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_email_type CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_email_provider CASCADE")
			
			logger.info("✅ Email integration migration rolled back successfully")
			
		except Exception as e:
			logger.error(f"Failed to rollback email integration migration: {str(e)}", exc_info=True)
			raise
	
	async def validate(self, connection) -> bool:
		"""Validate the migration was applied correctly"""
		try:
			# Check if main tables exist
			tables_exist = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.tables 
				WHERE table_name IN (
					'crm_email_templates', 'crm_email_messages', 
					'crm_email_tracking', 'crm_email_configurations', 
					'crm_email_analytics'
				)
			""")
			
			if tables_exist != 5:
				return False
			
			# Check if enums exist
			enum_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_type 
				WHERE typname IN (
					'crm_email_provider', 'crm_email_type',
					'crm_email_status', 'crm_email_priority'
				)
			""")
			
			if enum_count != 4:
				return False
			
			# Check if views exist
			view_count = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.views 
				WHERE table_name IN ('crm_email_performance', 'crm_template_performance')
			""")
			
			if view_count != 2:
				return False
			
			# Check if functions exist
			function_count = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.routines 
				WHERE routine_name IN (
					'calculate_email_engagement_score',
					'identify_top_email_performers',
					'update_crm_email_templates_updated_at',
					'update_crm_email_messages_updated_at',
					'update_crm_email_configurations_updated_at'
				)
			""")
			
			if function_count < 5:
				return False
			
			# Check key indexes
			index_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_indexes 
				WHERE tablename IN ('crm_email_templates', 'crm_email_messages', 'crm_email_tracking')
				AND indexname IN (
					'idx_crm_email_templates_tenant',
					'idx_crm_email_messages_tenant',
					'idx_crm_email_tracking_tenant'
				)
			""")
			
			if index_count < 3:
				return False
			
			return True
			
		except Exception as e:
			logger.error(f"Migration validation failed: {str(e)}", exc_info=True)
			return False