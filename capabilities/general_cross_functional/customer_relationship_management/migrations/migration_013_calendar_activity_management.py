"""
APG Customer Relationship Management - Calendar and Activity Management Migration

Database migration to create calendar and activity management tables and supporting 
structures for comprehensive scheduling, meeting coordination, and activity tracking.

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


class CalendarActivityManagementMigration(BaseMigration):
	"""Migration for calendar and activity management functionality"""
	
	def _get_migration_id(self) -> str:
		return "013_calendar_activity_management"
	
	def _get_version(self) -> str:
		return "013"
	
	def _get_description(self) -> str:
		return "Create calendar and activity management tables and supporting structures"
	
	def _get_dependencies(self) -> list:
		return ["001_initial_schema"]
	
	def _is_reversible(self) -> bool:
		return True
	
	async def up(self, connection) -> None:
		"""Apply the migration"""
		try:
			logger.info("🔄 Creating calendar and activity management structures...")
			
			# Create calendar provider enum
			await connection.execute("""
				CREATE TYPE crm_calendar_provider AS ENUM (
					'internal', 'google', 'outlook', 'exchange', 'caldav', 'ical', 'custom'
				)
			""")
			
			# Create event type enum
			await connection.execute("""
				CREATE TYPE crm_event_type AS ENUM (
					'meeting', 'call', 'demo', 'presentation', 'training', 'follow_up',
					'deadline', 'reminder', 'task', 'appointment', 'conference', 'webinar', 'custom'
				)
			""")
			
			# Create event status enum
			await connection.execute("""
				CREATE TYPE crm_event_status AS ENUM (
					'scheduled', 'confirmed', 'tentative', 'cancelled', 'completed', 
					'rescheduled', 'no_show', 'in_progress'
				)
			""")
			
			# Create event priority enum
			await connection.execute("""
				CREATE TYPE crm_event_priority AS ENUM (
					'low', 'normal', 'high', 'urgent'
				)
			""")
			
			# Create attendee status enum
			await connection.execute("""
				CREATE TYPE crm_attendee_status AS ENUM (
					'invited', 'accepted', 'declined', 'tentative', 'no_response'
				)
			""")
			
			# Create recurrence frequency enum
			await connection.execute("""
				CREATE TYPE crm_recurrence_frequency AS ENUM (
					'daily', 'weekly', 'monthly', 'yearly', 'custom'
				)
			""")
			
			# Create activity type enum
			await connection.execute("""
				CREATE TYPE crm_activity_type AS ENUM (
					'email', 'call', 'meeting', 'task', 'note', 'document', 'proposal',
					'contract', 'demo', 'follow_up', 'social_media', 'marketing', 'support', 'custom'
				)
			""")
			
			# Create activity status enum
			await connection.execute("""
				CREATE TYPE crm_activity_status AS ENUM (
					'planned', 'in_progress', 'completed', 'cancelled', 'overdue', 'deferred'
				)
			""")
			
			# Create calendar events table
			await connection.execute("""
				CREATE TABLE crm_calendar_events (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					
					-- Event identification
					external_id TEXT,
					calendar_provider crm_calendar_provider DEFAULT 'internal',
					provider_metadata JSONB DEFAULT '{}',
					
					-- Event details
					title TEXT NOT NULL,
					description TEXT,
					location TEXT,
					virtual_meeting_url TEXT,
					
					-- Event type and status
					event_type crm_event_type DEFAULT 'meeting',
					status crm_event_status DEFAULT 'scheduled',
					priority crm_event_priority DEFAULT 'normal',
					
					-- Timing
					start_time TIMESTAMP WITH TIME ZONE NOT NULL,
					end_time TIMESTAMP WITH TIME ZONE NOT NULL,
					timezone TEXT DEFAULT 'UTC',
					all_day BOOLEAN DEFAULT false,
					
					-- CRM relationships
					contact_id TEXT,
					account_id TEXT,
					lead_id TEXT,
					opportunity_id TEXT,
					campaign_id TEXT,
					
					-- Organization
					organizer_email TEXT NOT NULL,
					organizer_name TEXT,
					
					-- Recurrence
					is_recurring BOOLEAN DEFAULT false,
					parent_event_id TEXT,
					
					-- Notifications and reminders
					reminder_minutes INTEGER[] DEFAULT '{}',
					notification_sent BOOLEAN DEFAULT false,
					
					-- Activity tracking
					actual_start_time TIMESTAMP WITH TIME ZONE,
					actual_end_time TIMESTAMP WITH TIME ZONE,
					actual_duration_minutes INTEGER,
					
					-- Meeting notes and outcomes
					meeting_notes TEXT,
					meeting_outcomes TEXT[] DEFAULT '{}',
					action_items JSONB DEFAULT '[]',
					
					-- Integration
					sync_with_external BOOLEAN DEFAULT true,
					last_synced_at TIMESTAMP WITH TIME ZONE,
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					version INTEGER DEFAULT 1,
					
					-- Foreign key constraints
					FOREIGN KEY (contact_id) REFERENCES crm_contacts(id) ON DELETE SET NULL,
					FOREIGN KEY (account_id) REFERENCES crm_accounts(id) ON DELETE SET NULL,
					FOREIGN KEY (lead_id) REFERENCES crm_leads(id) ON DELETE SET NULL,
					FOREIGN KEY (opportunity_id) REFERENCES crm_opportunities(id) ON DELETE SET NULL,
					FOREIGN KEY (parent_event_id) REFERENCES crm_calendar_events(id) ON DELETE SET NULL,
					
					-- Constraints
					CONSTRAINT check_event_title_length CHECK (char_length(title) >= 1 AND char_length(title) <= 500),
					CONSTRAINT check_event_description_length CHECK (char_length(description) <= 5000),
					CONSTRAINT check_event_location_length CHECK (char_length(location) <= 500),
					CONSTRAINT check_end_after_start CHECK (end_time > start_time),
					CONSTRAINT check_organizer_email_format CHECK (organizer_email ~ '^[^@]+@[^@]+\.[^@]+$'),
					CONSTRAINT check_actual_duration_positive CHECK (actual_duration_minutes IS NULL OR actual_duration_minutes > 0),
					CONSTRAINT check_virtual_meeting_url_format CHECK (
						virtual_meeting_url IS NULL OR 
						virtual_meeting_url ~ '^https?://'
					)
				)
			""")
			
			# Create calendar event attendees table
			await connection.execute("""
				CREATE TABLE crm_calendar_event_attendees (
					id TEXT PRIMARY KEY,
					event_id TEXT NOT NULL,
					
					-- Attendee details
					email TEXT NOT NULL,
					name TEXT,
					role TEXT DEFAULT 'attendee',
					
					-- Response status
					status crm_attendee_status DEFAULT 'invited',
					response_time TIMESTAMP WITH TIME ZONE,
					
					-- CRM relationships
					contact_id TEXT,
					account_id TEXT,
					lead_id TEXT,
					
					-- Settings
					is_organizer BOOLEAN DEFAULT false,
					is_required BOOLEAN DEFAULT true,
					send_notifications BOOLEAN DEFAULT true,
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Foreign key constraints
					FOREIGN KEY (event_id) REFERENCES crm_calendar_events(id) ON DELETE CASCADE,
					FOREIGN KEY (contact_id) REFERENCES crm_contacts(id) ON DELETE SET NULL,
					FOREIGN KEY (account_id) REFERENCES crm_accounts(id) ON DELETE SET NULL,
					FOREIGN KEY (lead_id) REFERENCES crm_leads(id) ON DELETE SET NULL,
					
					-- Constraints
					CONSTRAINT check_attendee_email_format CHECK (email ~ '^[^@]+@[^@]+\.[^@]+$'),
					CONSTRAINT check_attendee_name_length CHECK (char_length(name) <= 200),
					CONSTRAINT check_attendee_role_length CHECK (char_length(role) <= 50)
				)
			""")
			
			# Create calendar event recurrence table
			await connection.execute("""
				CREATE TABLE crm_calendar_event_recurrence (
					id TEXT PRIMARY KEY,
					event_id TEXT NOT NULL,
					
					-- Recurrence pattern
					frequency crm_recurrence_frequency NOT NULL,
					interval INTEGER DEFAULT 1,
					
					-- Recurrence rules
					days_of_week INTEGER[] DEFAULT '{}',
					days_of_month INTEGER[] DEFAULT '{}',
					months_of_year INTEGER[] DEFAULT '{}',
					
					-- Recurrence limits
					end_date DATE,
					occurrence_count INTEGER,
					
					-- Exception dates
					exception_dates DATE[] DEFAULT '{}',
					
					-- Custom rule
					custom_rule TEXT,
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Foreign key constraints
					FOREIGN KEY (event_id) REFERENCES crm_calendar_events(id) ON DELETE CASCADE,
					
					-- Constraints
					CONSTRAINT check_recurrence_interval_positive CHECK (interval > 0),
					CONSTRAINT check_occurrence_count_positive CHECK (occurrence_count IS NULL OR occurrence_count > 0),
					CONSTRAINT check_days_of_week_valid CHECK (
						days_of_week IS NULL OR 
						(array_length(days_of_week, 1) IS NULL OR 
						 NOT EXISTS (SELECT 1 FROM unnest(days_of_week) AS dow WHERE dow < 0 OR dow > 6))
					),
					CONSTRAINT check_days_of_month_valid CHECK (
						days_of_month IS NULL OR 
						(array_length(days_of_month, 1) IS NULL OR 
						 NOT EXISTS (SELECT 1 FROM unnest(days_of_month) AS dom WHERE dom < 1 OR dom > 31))
					),
					CONSTRAINT check_months_of_year_valid CHECK (
						months_of_year IS NULL OR 
						(array_length(months_of_year, 1) IS NULL OR 
						 NOT EXISTS (SELECT 1 FROM unnest(months_of_year) AS moy WHERE moy < 1 OR moy > 12))
					)
				)
			""")
			
			# Create activities table
			await connection.execute("""
				CREATE TABLE crm_activities (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					
					-- Activity details
					title TEXT NOT NULL,
					description TEXT,
					activity_type crm_activity_type NOT NULL,
					status crm_activity_status DEFAULT 'planned',
					priority crm_event_priority DEFAULT 'normal',
					
					-- Timing
					due_date TIMESTAMP WITH TIME ZONE,
					completed_at TIMESTAMP WITH TIME ZONE,
					estimated_duration_minutes INTEGER,
					actual_duration_minutes INTEGER,
					
					-- CRM relationships
					contact_id TEXT,
					account_id TEXT,
					lead_id TEXT,
					opportunity_id TEXT,
					campaign_id TEXT,
					
					-- Assignment
					assigned_to TEXT NOT NULL,
					assigned_by TEXT,
					team_id TEXT,
					
					-- Related records
					parent_activity_id TEXT,
					related_event_id TEXT,
					related_email_id TEXT,
					
					-- Results and outcomes
					outcome TEXT,
					notes TEXT,
					attachments JSONB DEFAULT '[]',
					
					-- Automation
					is_automated BOOLEAN DEFAULT false,
					workflow_id TEXT,
					
					-- Tracking
					reminder_sent BOOLEAN DEFAULT false,
					overdue_notifications_sent INTEGER DEFAULT 0,
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by TEXT NOT NULL,
					updated_by TEXT NOT NULL,
					version INTEGER DEFAULT 1,
					
					-- Foreign key constraints
					FOREIGN KEY (contact_id) REFERENCES crm_contacts(id) ON DELETE SET NULL,
					FOREIGN KEY (account_id) REFERENCES crm_accounts(id) ON DELETE SET NULL,
					FOREIGN KEY (lead_id) REFERENCES crm_leads(id) ON DELETE SET NULL,
					FOREIGN KEY (opportunity_id) REFERENCES crm_opportunities(id) ON DELETE SET NULL,
					FOREIGN KEY (parent_activity_id) REFERENCES crm_activities(id) ON DELETE SET NULL,
					FOREIGN KEY (related_event_id) REFERENCES crm_calendar_events(id) ON DELETE SET NULL,
					
					-- Constraints
					CONSTRAINT check_activity_title_length CHECK (char_length(title) >= 1 AND char_length(title) <= 500),
					CONSTRAINT check_activity_description_length CHECK (char_length(description) <= 5000),
					CONSTRAINT check_activity_outcome_length CHECK (char_length(outcome) <= 1000),
					CONSTRAINT check_estimated_duration_positive CHECK (estimated_duration_minutes IS NULL OR estimated_duration_minutes > 0),
					CONSTRAINT check_actual_duration_positive CHECK (actual_duration_minutes IS NULL OR actual_duration_minutes > 0),
					CONSTRAINT check_overdue_notifications_positive CHECK (overdue_notifications_sent >= 0),
					CONSTRAINT check_completed_at_after_creation CHECK (
						completed_at IS NULL OR completed_at >= created_at
					)
				)
			""")
			
			# Create activity templates table
			await connection.execute("""
				CREATE TABLE crm_activity_templates (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					
					-- Template details
					name TEXT NOT NULL,
					description TEXT,
					category TEXT DEFAULT 'general',
					
					-- Default activity settings
					activity_type crm_activity_type NOT NULL,
					priority crm_event_priority DEFAULT 'normal',
					estimated_duration_minutes INTEGER,
					
					-- Template content
					title_template TEXT NOT NULL,
					description_template TEXT,
					
					-- Default assignments
					default_assigned_to TEXT,
					default_team_id TEXT,
					
					-- Automation settings
					auto_due_date_days INTEGER,
					auto_reminders INTEGER[] DEFAULT '{}',
					
					-- Usage tracking
					usage_count INTEGER DEFAULT 0,
					is_active BOOLEAN DEFAULT true,
					
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
					CONSTRAINT check_template_title_length CHECK (char_length(title_template) >= 1),
					CONSTRAINT check_template_estimated_duration_positive CHECK (estimated_duration_minutes IS NULL OR estimated_duration_minutes > 0),
					CONSTRAINT check_template_auto_due_date_positive CHECK (auto_due_date_days IS NULL OR auto_due_date_days > 0),
					CONSTRAINT check_template_usage_count_positive CHECK (usage_count >= 0)
				)
			""")
			
			# Create calendar analytics table
			await connection.execute("""
				CREATE TABLE crm_calendar_analytics (
					id TEXT PRIMARY KEY,
					tenant_id TEXT NOT NULL,
					
					-- Analysis period
					period_start TIMESTAMP WITH TIME ZONE NOT NULL,
					period_end TIMESTAMP WITH TIME ZONE NOT NULL,
					period_type TEXT NOT NULL, -- 'hour', 'day', 'week', 'month'
					
					-- Event metrics
					total_events INTEGER DEFAULT 0,
					completed_events INTEGER DEFAULT 0,
					cancelled_events INTEGER DEFAULT 0,
					no_show_events INTEGER DEFAULT 0,
					
					-- Meeting metrics
					total_meetings INTEGER DEFAULT 0,
					average_meeting_duration DECIMAL(8,2) DEFAULT 0,
					total_meeting_time INTEGER DEFAULT 0,
					meeting_attendance_rate DECIMAL(5,2) DEFAULT 0,
					
					-- Activity metrics
					total_activities INTEGER DEFAULT 0,
					completed_activities INTEGER DEFAULT 0,
					overdue_activities INTEGER DEFAULT 0,
					average_completion_time DECIMAL(8,2) DEFAULT 0,
					
					-- Productivity metrics
					events_per_day DECIMAL(8,2) DEFAULT 0,
					activities_per_day DECIMAL(8,2) DEFAULT 0,
					completion_rate DECIMAL(5,2) DEFAULT 0,
					on_time_rate DECIMAL(5,2) DEFAULT 0,
					
					-- Time distribution
					busiest_hours JSONB DEFAULT '{}',
					busiest_days JSONB DEFAULT '{}',
					peak_productivity_hours JSONB DEFAULT '[]',
					
					-- User performance
					top_performers JSONB DEFAULT '[]',
					user_metrics JSONB DEFAULT '{}',
					
					-- CRM impact
					events_with_outcomes INTEGER DEFAULT 0,
					activities_driving_revenue INTEGER DEFAULT 0,
					average_deal_size_impact DECIMAL(15,2) DEFAULT 0,
					
					-- Trends
					productivity_trend DECIMAL(5,2) DEFAULT 0,
					completion_trend DECIMAL(5,2) DEFAULT 0,
					engagement_trend DECIMAL(5,2) DEFAULT 0,
					
					-- Metadata
					metadata JSONB DEFAULT '{}',
					
					-- Audit fields
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					
					-- Constraints
					CONSTRAINT check_calendar_period_valid CHECK (period_end > period_start),
					CONSTRAINT check_calendar_period_type CHECK (period_type IN ('hour', 'day', 'week', 'month')),
					CONSTRAINT check_calendar_event_counters_positive CHECK (
						total_events >= 0 AND 
						completed_events >= 0 AND 
						cancelled_events >= 0 AND 
						no_show_events >= 0 AND
						total_meetings >= 0
					),
					CONSTRAINT check_calendar_activity_counters_positive CHECK (
						total_activities >= 0 AND 
						completed_activities >= 0 AND 
						overdue_activities >= 0
					),
					CONSTRAINT check_calendar_rates_range CHECK (
						meeting_attendance_rate >= 0 AND meeting_attendance_rate <= 100 AND
						completion_rate >= 0 AND completion_rate <= 100 AND
						on_time_rate >= 0 AND on_time_rate <= 100
					),
					CONSTRAINT check_calendar_metrics_positive CHECK (
						average_meeting_duration >= 0 AND 
						total_meeting_time >= 0 AND 
						average_completion_time >= 0 AND
						events_per_day >= 0 AND 
						activities_per_day >= 0 AND
						events_with_outcomes >= 0 AND 
						activities_driving_revenue >= 0 AND
						average_deal_size_impact >= 0
					)
				)
			""")
			
			# Create indexes for calendar events table
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_tenant 
				ON crm_calendar_events(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_external_id 
				ON crm_calendar_events(external_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_provider 
				ON crm_calendar_events(calendar_provider)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_event_type 
				ON crm_calendar_events(event_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_status 
				ON crm_calendar_events(status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_priority 
				ON crm_calendar_events(priority)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_start_time 
				ON crm_calendar_events(start_time)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_end_time 
				ON crm_calendar_events(end_time)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_organizer_email 
				ON crm_calendar_events(organizer_email)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_is_recurring 
				ON crm_calendar_events(is_recurring)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_parent_event_id 
				ON crm_calendar_events(parent_event_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_created_at 
				ON crm_calendar_events(created_at)
			""")
			
			# Create indexes for CRM relationships
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_contact_id 
				ON crm_calendar_events(contact_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_account_id 
				ON crm_calendar_events(account_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_lead_id 
				ON crm_calendar_events(lead_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_opportunity_id 
				ON crm_calendar_events(opportunity_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_campaign_id 
				ON crm_calendar_events(campaign_id)
			""")
			
			# Create GIN indexes for JSONB and array fields
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_provider_metadata 
				ON crm_calendar_events USING GIN (provider_metadata)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_reminder_minutes 
				ON crm_calendar_events USING GIN (reminder_minutes)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_meeting_outcomes 
				ON crm_calendar_events USING GIN (meeting_outcomes)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_action_items 
				ON crm_calendar_events USING GIN (action_items)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_metadata 
				ON crm_calendar_events USING GIN (metadata)
			""")
			
			# Create composite indexes for events
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_tenant_time_range 
				ON crm_calendar_events(tenant_id, start_time, end_time)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_tenant_status 
				ON crm_calendar_events(tenant_id, status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_events_tenant_organizer 
				ON crm_calendar_events(tenant_id, organizer_email)
			""")
			
			# Create indexes for attendees table
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_event_attendees_event_id 
				ON crm_calendar_event_attendees(event_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_event_attendees_email 
				ON crm_calendar_event_attendees(email)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_event_attendees_status 
				ON crm_calendar_event_attendees(status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_event_attendees_contact_id 
				ON crm_calendar_event_attendees(contact_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_event_attendees_is_organizer 
				ON crm_calendar_event_attendees(is_organizer)
			""")
			
			# Create indexes for recurrence table  
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_event_recurrence_event_id 
				ON crm_calendar_event_recurrence(event_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_event_recurrence_frequency 
				ON crm_calendar_event_recurrence(frequency)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_event_recurrence_end_date 
				ON crm_calendar_event_recurrence(end_date)
			""")
			
			# Create indexes for activities table
			await connection.execute("""
				CREATE INDEX idx_crm_activities_tenant 
				ON crm_activities(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_activity_type 
				ON crm_activities(activity_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_status 
				ON crm_activities(status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_priority 
				ON crm_activities(priority)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_due_date 
				ON crm_activities(due_date)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_completed_at 
				ON crm_activities(completed_at)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_assigned_to 
				ON crm_activities(assigned_to)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_assigned_by 
				ON crm_activities(assigned_by)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_team_id 
				ON crm_activities(team_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_parent_activity_id 
				ON crm_activities(parent_activity_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_related_event_id 
				ON crm_activities(related_event_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_created_at 
				ON crm_activities(created_at)
			""")
			
			# Create indexes for CRM relationships in activities
			await connection.execute("""
				CREATE INDEX idx_crm_activities_contact_id 
				ON crm_activities(contact_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_account_id 
				ON crm_activities(account_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_lead_id 
				ON crm_activities(lead_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_opportunity_id 
				ON crm_activities(opportunity_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_campaign_id 
				ON crm_activities(campaign_id)
			""")
			
			# Create GIN indexes for activities JSONB fields
			await connection.execute("""
				CREATE INDEX idx_crm_activities_attachments 
				ON crm_activities USING GIN (attachments)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_metadata 
				ON crm_activities USING GIN (metadata)
			""")
			
			# Create composite indexes for activities
			await connection.execute("""
				CREATE INDEX idx_crm_activities_tenant_status 
				ON crm_activities(tenant_id, status)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_tenant_assigned_to 
				ON crm_activities(tenant_id, assigned_to)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_tenant_due_date 
				ON crm_activities(tenant_id, due_date)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activities_assigned_to_due_date 
				ON crm_activities(assigned_to, due_date)
			""")
			
			# Create special index for overdue activities
			await connection.execute("""
				CREATE INDEX idx_crm_activities_overdue 
				ON crm_activities(tenant_id, due_date)
				WHERE status NOT IN ('completed', 'cancelled') AND due_date < NOW()
			""")
			
			# Create indexes for activity templates table
			await connection.execute("""
				CREATE INDEX idx_crm_activity_templates_tenant 
				ON crm_activity_templates(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activity_templates_category 
				ON crm_activity_templates(category)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activity_templates_activity_type 
				ON crm_activity_templates(activity_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activity_templates_is_active 
				ON crm_activity_templates(is_active)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activity_templates_usage_count 
				ON crm_activity_templates(usage_count DESC)
			""")
			
			# Create composite indexes for templates
			await connection.execute("""
				CREATE INDEX idx_crm_activity_templates_tenant_active 
				ON crm_activity_templates(tenant_id, is_active)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_activity_templates_tenant_category 
				ON crm_activity_templates(tenant_id, category)
			""")
			
			# Create indexes for calendar analytics table
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_analytics_tenant 
				ON crm_calendar_analytics(tenant_id)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_analytics_period 
				ON crm_calendar_analytics(period_start, period_end)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_analytics_period_type 
				ON crm_calendar_analytics(period_type)
			""")
			
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_analytics_created_at 
				ON crm_calendar_analytics(created_at)
			""")
			
			# Create composite indexes for analytics
			await connection.execute("""
				CREATE INDEX idx_crm_calendar_analytics_tenant_period 
				ON crm_calendar_analytics(tenant_id, period_start)
			""")
			
			# Create triggers for updating updated_at timestamps
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_calendar_events_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_calendar_events_updated_at
					BEFORE UPDATE ON crm_calendar_events
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_calendar_events_updated_at()
			""")
			
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_activities_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_activities_updated_at
					BEFORE UPDATE ON crm_activities
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_activities_updated_at()
			""")
			
			await connection.execute("""
				CREATE OR REPLACE FUNCTION update_crm_activity_templates_updated_at()
				RETURNS TRIGGER AS $$
				BEGIN
					NEW.updated_at = NOW();
					RETURN NEW;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			await connection.execute("""
				CREATE TRIGGER trigger_crm_activity_templates_updated_at
					BEFORE UPDATE ON crm_activity_templates
					FOR EACH ROW
					EXECUTE FUNCTION update_crm_activity_templates_updated_at()
			""")
			
			# Create view for calendar dashboard
			await connection.execute("""
				CREATE VIEW crm_calendar_dashboard AS
				SELECT 
					ce.tenant_id,
					DATE_TRUNC('day', ce.start_time) as event_date,
					
					-- Event counts by status
					COUNT(*) as total_events,
					COUNT(*) FILTER (WHERE ce.status = 'scheduled') as scheduled_events,
					COUNT(*) FILTER (WHERE ce.status = 'confirmed') as confirmed_events,
					COUNT(*) FILTER (WHERE ce.status = 'completed') as completed_events,
					COUNT(*) FILTER (WHERE ce.status = 'cancelled') as cancelled_events,
					
					-- Event types
					COUNT(*) FILTER (WHERE ce.event_type = 'meeting') as meetings,
					COUNT(*) FILTER (WHERE ce.event_type = 'call') as calls,
					COUNT(*) FILTER (WHERE ce.event_type = 'demo') as demos,
					
					-- Time metrics
					AVG(EXTRACT(EPOCH FROM (ce.end_time - ce.start_time))/60) as avg_duration_minutes,
					SUM(EXTRACT(EPOCH FROM (ce.end_time - ce.start_time))/60) as total_duration_minutes,
					
					-- Attendee metrics
					AVG(attendee_counts.attendee_count) as avg_attendees_per_event,
					
					-- CRM relationships
					COUNT(*) FILTER (WHERE ce.contact_id IS NOT NULL) as events_with_contacts,
					COUNT(*) FILTER (WHERE ce.opportunity_id IS NOT NULL) as events_with_opportunities
					
				FROM crm_calendar_events ce
				LEFT JOIN (
					SELECT 
						event_id, 
						COUNT(*) as attendee_count
					FROM crm_calendar_event_attendees
					GROUP BY event_id
				) attendee_counts ON ce.id = attendee_counts.event_id
				WHERE ce.start_time >= NOW() - INTERVAL '90 days'
				GROUP BY ce.tenant_id, DATE_TRUNC('day', ce.start_time)
				ORDER BY event_date DESC
			""")
			
			# Create view for activity dashboard
			await connection.execute("""
				CREATE VIEW crm_activity_dashboard AS
				SELECT 
					a.tenant_id,
					a.assigned_to,
					DATE_TRUNC('day', COALESCE(a.due_date, a.created_at)) as activity_date,
					
					-- Activity counts by status
					COUNT(*) as total_activities,
					COUNT(*) FILTER (WHERE a.status = 'planned') as planned_activities,
					COUNT(*) FILTER (WHERE a.status = 'in_progress') as in_progress_activities,
					COUNT(*) FILTER (WHERE a.status = 'completed') as completed_activities,
					COUNT(*) FILTER (WHERE a.status = 'overdue') as overdue_activities,
					
					-- Activity types
					COUNT(*) FILTER (WHERE a.activity_type = 'email') as email_activities,
					COUNT(*) FILTER (WHERE a.activity_type = 'call') as call_activities,
					COUNT(*) FILTER (WHERE a.activity_type = 'meeting') as meeting_activities,
					COUNT(*) FILTER (WHERE a.activity_type = 'task') as task_activities,
					COUNT(*) FILTER (WHERE a.activity_type = 'follow_up') as follow_up_activities,
					
					-- Performance metrics
					AVG(CASE 
						WHEN a.completed_at IS NOT NULL AND a.created_at IS NOT NULL THEN
							EXTRACT(EPOCH FROM (a.completed_at - a.created_at))/3600
						ELSE NULL
					END) as avg_completion_hours,
					
					CASE 
						WHEN COUNT(*) > 0 THEN
							(COUNT(*) FILTER (WHERE a.status = 'completed')::DECIMAL / COUNT(*)) * 100
						ELSE 0
					END as completion_rate,
					
					-- CRM relationships
					COUNT(*) FILTER (WHERE a.contact_id IS NOT NULL) as activities_with_contacts,
					COUNT(*) FILTER (WHERE a.opportunity_id IS NOT NULL) as activities_with_opportunities
					
				FROM crm_activities a
				WHERE a.created_at >= NOW() - INTERVAL '90 days'
				GROUP BY a.tenant_id, a.assigned_to, DATE_TRUNC('day', COALESCE(a.due_date, a.created_at))
				ORDER BY activity_date DESC
			""")
			
			# Create function for calculating productivity scores
			await connection.execute("""
				CREATE OR REPLACE FUNCTION calculate_user_productivity_score(
					tenant_filter TEXT,
					user_filter TEXT,
					days_back INTEGER DEFAULT 30
				)
				RETURNS TABLE(
					user_id TEXT,
					productivity_score DECIMAL,
					events_completed INTEGER,
					activities_completed INTEGER,
					avg_meeting_duration DECIMAL,
					completion_rate DECIMAL
				) AS $$
				BEGIN
					RETURN QUERY
					SELECT 
						user_filter as user_id,
						-- Weighted productivity score
						(
							(COALESCE(event_metrics.completed_events, 0) * 2.0) +
							(COALESCE(activity_metrics.completed_activities, 0) * 1.5) +
							(COALESCE(activity_metrics.completion_rate, 0) * 0.5)
						) as productivity_score,
						COALESCE(event_metrics.completed_events, 0)::INTEGER as events_completed,
						COALESCE(activity_metrics.completed_activities, 0)::INTEGER as activities_completed,
						COALESCE(event_metrics.avg_duration, 0) as avg_meeting_duration,
						COALESCE(activity_metrics.completion_rate, 0) as completion_rate
						
					FROM (
						SELECT 
							COUNT(*) FILTER (WHERE status = 'completed') as completed_events,
							AVG(EXTRACT(EPOCH FROM (end_time - start_time))/60) as avg_duration
						FROM crm_calendar_events
						WHERE tenant_id = tenant_filter
						AND organizer_email LIKE '%' || user_filter || '%'
						AND start_time >= NOW() - (days_back || ' days')::INTERVAL
					) event_metrics
					CROSS JOIN (
						SELECT 
							COUNT(*) FILTER (WHERE status = 'completed') as completed_activities,
							CASE 
								WHEN COUNT(*) > 0 THEN
									(COUNT(*) FILTER (WHERE status = 'completed')::DECIMAL / COUNT(*)) * 100
								ELSE 0
							END as completion_rate
						FROM crm_activities
						WHERE tenant_id = tenant_filter
						AND assigned_to = user_filter
						AND created_at >= NOW() - (days_back || ' days')::INTERVAL
					) activity_metrics;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			# Create function for identifying scheduling conflicts
			await connection.execute("""
				CREATE OR REPLACE FUNCTION identify_scheduling_conflicts(
					tenant_filter TEXT,
					user_email TEXT,
					start_time TIMESTAMP WITH TIME ZONE,
					end_time TIMESTAMP WITH TIME ZONE
				)
				RETURNS TABLE(
					event_id TEXT,
					event_title TEXT,
					event_start TIMESTAMP WITH TIME ZONE,
					event_end TIMESTAMP WITH TIME ZONE,
					conflict_type TEXT
				) AS $$
				BEGIN
					RETURN QUERY
					SELECT 
						ce.id as event_id,
						ce.title as event_title,
						ce.start_time as event_start,
						ce.end_time as event_end,
						CASE 
							WHEN ce.start_time <= start_time AND ce.end_time >= end_time THEN 'complete_overlap'
							WHEN ce.start_time >= start_time AND ce.end_time <= end_time THEN 'contained'
							WHEN ce.start_time < end_time AND ce.end_time > start_time THEN 'partial_overlap'
							ELSE 'adjacent'
						END as conflict_type
					FROM crm_calendar_events ce
					WHERE ce.tenant_id = tenant_filter
					AND ce.organizer_email = user_email
					AND ce.status NOT IN ('cancelled', 'completed')
					AND (
						(ce.start_time <= start_time AND ce.end_time > start_time) OR
						(ce.start_time < end_time AND ce.end_time >= end_time) OR
						(ce.start_time >= start_time AND ce.end_time <= end_time)
					)
					ORDER BY ce.start_time;
				END;
				$$ LANGUAGE plpgsql
			""")
			
			logger.info("✅ Calendar and activity management structures created successfully")
			
		except Exception as e:
			logger.error(f"Failed to create calendar and activity management structures: {str(e)}", exc_info=True)
			raise
	
	async def down(self, connection) -> None:
		"""Rollback the migration"""
		try:
			logger.info("🔄 Rolling back calendar and activity management migration...")
			
			# Drop functions
			await connection.execute("DROP FUNCTION IF EXISTS identify_scheduling_conflicts CASCADE")
			await connection.execute("DROP FUNCTION IF EXISTS calculate_user_productivity_score CASCADE")
			
			# Drop views
			await connection.execute("DROP VIEW IF EXISTS crm_activity_dashboard CASCADE")
			await connection.execute("DROP VIEW IF EXISTS crm_calendar_dashboard CASCADE")
			
			# Drop triggers and functions
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_activity_templates_updated_at ON crm_activity_templates")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_activity_templates_updated_at CASCADE")
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_activities_updated_at ON crm_activities")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_activities_updated_at CASCADE")
			await connection.execute("DROP TRIGGER IF EXISTS trigger_crm_calendar_events_updated_at ON crm_calendar_events")
			await connection.execute("DROP FUNCTION IF EXISTS update_crm_calendar_events_updated_at CASCADE")
			
			# Drop tables (will cascade to indexes)
			await connection.execute("DROP TABLE IF EXISTS crm_calendar_analytics CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_activity_templates CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_activities CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_calendar_event_recurrence CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_calendar_event_attendees CASCADE")
			await connection.execute("DROP TABLE IF EXISTS crm_calendar_events CASCADE")
			
			# Drop enums
			await connection.execute("DROP TYPE IF EXISTS crm_activity_status CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_activity_type CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_recurrence_frequency CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_attendee_status CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_event_priority CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_event_status CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_event_type CASCADE")
			await connection.execute("DROP TYPE IF EXISTS crm_calendar_provider CASCADE")
			
			logger.info("✅ Calendar and activity management migration rolled back successfully")
			
		except Exception as e:
			logger.error(f"Failed to rollback calendar and activity management migration: {str(e)}", exc_info=True)
			raise
	
	async def validate(self, connection) -> bool:
		"""Validate the migration was applied correctly"""
		try:
			# Check if main tables exist
			tables_exist = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.tables 
				WHERE table_name IN (
					'crm_calendar_events', 'crm_calendar_event_attendees', 
					'crm_calendar_event_recurrence', 'crm_activities', 
					'crm_activity_templates', 'crm_calendar_analytics'
				)
			""")
			
			if tables_exist != 6:
				return False
			
			# Check if enums exist
			enum_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_type 
				WHERE typname IN (
					'crm_calendar_provider', 'crm_event_type', 'crm_event_status', 
					'crm_event_priority', 'crm_attendee_status', 'crm_recurrence_frequency',
					'crm_activity_type', 'crm_activity_status'
				)
			""")
			
			if enum_count != 8:
				return False
			
			# Check if views exist
			view_count = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.views 
				WHERE table_name IN ('crm_calendar_dashboard', 'crm_activity_dashboard')
			""")
			
			if view_count != 2:
				return False
			
			# Check if functions exist
			function_count = await connection.fetchval("""
				SELECT COUNT(*) FROM information_schema.routines 
				WHERE routine_name IN (
					'calculate_user_productivity_score',
					'identify_scheduling_conflicts',
					'update_crm_calendar_events_updated_at',
					'update_crm_activities_updated_at',
					'update_crm_activity_templates_updated_at'
				)
			""")
			
			if function_count < 5:
				return False
			
			# Check key indexes
			index_count = await connection.fetchval("""
				SELECT COUNT(*) FROM pg_indexes 
				WHERE tablename IN ('crm_calendar_events', 'crm_activities', 'crm_activity_templates')
				AND indexname IN (
					'idx_crm_calendar_events_tenant',
					'idx_crm_activities_tenant',
					'idx_crm_activity_templates_tenant'
				)
			""")
			
			if index_count < 3:
				return False
			
			return True
			
		except Exception as e:
			logger.error(f"Migration validation failed: {str(e)}", exc_info=True)
			return False