"""
APG Customer Relationship Management - Advanced Features Migration

Advanced CRM features migration adding AI insights tracking, relationship mapping,
communication history, and enhanced analytics capabilities to the CRM schema.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncpg
from .base_migration import BaseMigration, table_exists, column_exists


class AdvancedFeaturesMigration(BaseMigration):
	"""Advanced features migration for enhanced CRM capabilities"""
	
	def _get_migration_id(self) -> str:
		return "002_advanced_features"
	
	def _get_version(self) -> str:
		return "002"
	
	def _get_description(self) -> str:
		return "Add advanced CRM features: AI insights, relationships, communication history"
	
	def _get_dependencies(self) -> list:
		return ["001_initial_schema"]
	
	def _is_reversible(self) -> bool:
		return True
	
	async def up(self, connection: asyncpg.Connection) -> None:
		"""Add advanced CRM features"""
		
		# Add new enum types for advanced features
		await connection.execute("""
			CREATE TYPE insight_type AS ENUM (
				'lead_scoring', 'opportunity_prediction', 'customer_segmentation',
				'churn_prediction', 'next_best_action', 'sentiment_analysis',
				'engagement_optimization', 'price_optimization'
			)
		""")
		
		await connection.execute("""
			CREATE TYPE communication_channel AS ENUM (
				'email', 'phone', 'sms', 'social_media', 'chat', 'video_call',
				'meeting', 'mail', 'fax', 'other'
			)
		""")
		
		await connection.execute("""
			CREATE TYPE communication_direction AS ENUM (
				'inbound', 'outbound'
			)
		""")
		
		await connection.execute("""
			CREATE TYPE relationship_type AS ENUM (
				'reports_to', 'manages', 'colleague', 'partner', 'vendor',
				'customer', 'influencer', 'decision_maker', 'champion', 'other'
			)
		""")
		
		# 1. AI Insights tracking table
		await connection.execute("""
			CREATE TABLE crm_ai_insights (
				id VARCHAR(26) PRIMARY KEY,
				tenant_id VARCHAR(100) NOT NULL,
				insight_type insight_type NOT NULL,
				entity_type VARCHAR(50) NOT NULL,
				entity_id VARCHAR(26) NOT NULL,
				confidence_score DECIMAL(5,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 100),
				insights_data JSONB NOT NULL DEFAULT '{}',
				recommendations JSONB DEFAULT '[]',
				generated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				expires_at TIMESTAMP WITH TIME ZONE,
				model_version VARCHAR(50),
				metadata JSONB DEFAULT '{}',
				created_by VARCHAR(100) NOT NULL DEFAULT 'system',
				created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				
				-- Constraints
				CONSTRAINT crm_ai_insights_entity_types CHECK (
					entity_type IN ('contact', 'account', 'lead', 'opportunity', 'campaign')
				),
				CONSTRAINT crm_ai_insights_tenant_isolation UNIQUE (id, tenant_id)
			)
		""")
		
		# 2. Communication History table
		await connection.execute("""
			CREATE TABLE crm_communications (
				id VARCHAR(26) PRIMARY KEY,
				tenant_id VARCHAR(100) NOT NULL,
				communication_type VARCHAR(50) NOT NULL,
				channel communication_channel NOT NULL,
				direction communication_direction NOT NULL,
				subject VARCHAR(500),
				content TEXT,
				from_address VARCHAR(255),
				to_addresses TEXT[],
				cc_addresses TEXT[],
				bcc_addresses TEXT[],
				sent_at TIMESTAMP WITH TIME ZONE,
				received_at TIMESTAMP WITH TIME ZONE,
				read_at TIMESTAMP WITH TIME ZONE,
				replied_at TIMESTAMP WITH TIME ZONE,
				status VARCHAR(50) DEFAULT 'sent',
				external_id VARCHAR(255),
				thread_id VARCHAR(255),
				attachments JSONB DEFAULT '[]',
				metadata JSONB DEFAULT '{}',
				sentiment_score DECIMAL(5,2) CHECK (sentiment_score >= -100 AND sentiment_score <= 100),
				contact_id VARCHAR(26),
				account_id VARCHAR(26),
				opportunity_id VARCHAR(26),
				lead_id VARCHAR(26),
				activity_id VARCHAR(26),
				campaign_id VARCHAR(26),
				created_by VARCHAR(100) NOT NULL,
				created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				
				-- Constraints
				CONSTRAINT crm_communications_addresses_check CHECK (
					from_address IS NOT NULL OR array_length(to_addresses, 1) > 0
				),
				CONSTRAINT crm_communications_tenant_isolation UNIQUE (id, tenant_id),
				CONSTRAINT crm_communications_contact_fk FOREIGN KEY (contact_id) REFERENCES crm_contacts(id) ON DELETE CASCADE,
				CONSTRAINT crm_communications_account_fk FOREIGN KEY (account_id) REFERENCES crm_accounts(id) ON DELETE CASCADE,
				CONSTRAINT crm_communications_opportunity_fk FOREIGN KEY (opportunity_id) REFERENCES crm_opportunities(id) ON DELETE CASCADE,
				CONSTRAINT crm_communications_lead_fk FOREIGN KEY (lead_id) REFERENCES crm_leads(id) ON DELETE CASCADE,
				CONSTRAINT crm_communications_activity_fk FOREIGN KEY (activity_id) REFERENCES crm_activities(id) ON DELETE CASCADE,
				CONSTRAINT crm_communications_campaign_fk FOREIGN KEY (campaign_id) REFERENCES crm_campaigns(id) ON DELETE CASCADE
			)
		""")
		
		# 3. Contact Relationships table
		await connection.execute("""
			CREATE TABLE crm_contact_relationships (
				id VARCHAR(26) PRIMARY KEY,
				tenant_id VARCHAR(100) NOT NULL,
				from_contact_id VARCHAR(26) NOT NULL,
				to_contact_id VARCHAR(26) NOT NULL,
				relationship_type relationship_type NOT NULL,
				strength INTEGER DEFAULT 5 CHECK (strength >= 1 AND strength <= 10),
				notes TEXT,
				is_active BOOLEAN NOT NULL DEFAULT TRUE,
				metadata JSONB DEFAULT '{}',
				created_by VARCHAR(100) NOT NULL,
				updated_by VARCHAR(100) NOT NULL,
				created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				
				-- Constraints
				CONSTRAINT crm_contact_relationships_different_contacts CHECK (from_contact_id != to_contact_id),
				CONSTRAINT crm_contact_relationships_unique UNIQUE (from_contact_id, to_contact_id, relationship_type),
				CONSTRAINT crm_contact_relationships_tenant_isolation UNIQUE (id, tenant_id),
				CONSTRAINT crm_contact_relationships_from_fk FOREIGN KEY (from_contact_id) REFERENCES crm_contacts(id) ON DELETE CASCADE,
				CONSTRAINT crm_contact_relationships_to_fk FOREIGN KEY (to_contact_id) REFERENCES crm_contacts(id) ON DELETE CASCADE
			)
		""")
		
		# 4. Sales Pipeline Stages configuration table
		await connection.execute("""
			CREATE TABLE crm_pipeline_stages (
				id VARCHAR(26) PRIMARY KEY,
				tenant_id VARCHAR(100) NOT NULL,
				stage_name VARCHAR(100) NOT NULL,
				stage_order INTEGER NOT NULL,
				probability_percentage DECIMAL(5,2) NOT NULL DEFAULT 0 CHECK (probability_percentage >= 0 AND probability_percentage <= 100),
				is_closed_won BOOLEAN NOT NULL DEFAULT FALSE,
				is_closed_lost BOOLEAN NOT NULL DEFAULT FALSE,
				color_code VARCHAR(7),
				description TEXT,
				is_active BOOLEAN NOT NULL DEFAULT TRUE,
				created_by VARCHAR(100) NOT NULL,
				updated_by VARCHAR(100) NOT NULL,
				created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				
				-- Constraints
				CONSTRAINT crm_pipeline_stages_name_check CHECK (LENGTH(TRIM(stage_name)) > 0),
				CONSTRAINT crm_pipeline_stages_order_positive CHECK (stage_order > 0),
				CONSTRAINT crm_pipeline_stages_one_closed_won CHECK (
					NOT (is_closed_won = TRUE AND is_closed_lost = TRUE)
				),
				CONSTRAINT crm_pipeline_stages_tenant_order UNIQUE (tenant_id, stage_order),
				CONSTRAINT crm_pipeline_stages_tenant_name UNIQUE (tenant_id, stage_name),
				CONSTRAINT crm_pipeline_stages_tenant_isolation UNIQUE (id, tenant_id)
			)
		""")
		
		# 5. Campaign Members table (many-to-many between campaigns and contacts/leads)
		await connection.execute("""
			CREATE TABLE crm_campaign_members (
				id VARCHAR(26) PRIMARY KEY,
				tenant_id VARCHAR(100) NOT NULL,
				campaign_id VARCHAR(26) NOT NULL,
				contact_id VARCHAR(26),
				lead_id VARCHAR(26),
				status VARCHAR(50) NOT NULL DEFAULT 'sent',
				response_status VARCHAR(50),
				first_responded_date TIMESTAMP WITH TIME ZONE,
				sent_date TIMESTAMP WITH TIME ZONE,
				bounced_date TIMESTAMP WITH TIME ZONE,
				opened_date TIMESTAMP WITH TIME ZONE,
				clicked_date TIMESTAMP WITH TIME ZONE,
				unsubscribed_date TIMESTAMP WITH TIME ZONE,
				notes TEXT,
				custom_fields JSONB DEFAULT '{}',
				created_by VARCHAR(100) NOT NULL,
				created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				
				-- Constraints
				CONSTRAINT crm_campaign_members_target_check CHECK (
					(contact_id IS NOT NULL AND lead_id IS NULL) OR 
					(contact_id IS NULL AND lead_id IS NOT NULL)
				),
				CONSTRAINT crm_campaign_members_tenant_isolation UNIQUE (id, tenant_id),
				CONSTRAINT crm_campaign_members_campaign_fk FOREIGN KEY (campaign_id) REFERENCES crm_campaigns(id) ON DELETE CASCADE,
				CONSTRAINT crm_campaign_members_contact_fk FOREIGN KEY (contact_id) REFERENCES crm_contacts(id) ON DELETE CASCADE,
				CONSTRAINT crm_campaign_members_lead_fk FOREIGN KEY (lead_id) REFERENCES crm_leads(id) ON DELETE CASCADE
			)
		""")
		
		# 6. Add new columns to existing tables for enhanced functionality
		
		# Add AI-related columns to contacts
		await connection.execute("""
			ALTER TABLE crm_contacts ADD COLUMN 
			ai_engagement_score DECIMAL(5,2) CHECK (ai_engagement_score >= 0 AND ai_engagement_score <= 100)
		""")
		
		await connection.execute("""
			ALTER TABLE crm_contacts ADD COLUMN 
			last_engagement_date TIMESTAMP WITH TIME ZONE
		""")
		
		await connection.execute("""
			ALTER TABLE crm_contacts ADD COLUMN 
			communication_preferences JSONB DEFAULT '{}'
		""")
		
		# Add relationship tracking to accounts
		await connection.execute("""
			ALTER TABLE crm_accounts ADD COLUMN 
			key_contacts JSONB DEFAULT '[]'
		""")
		
		await connection.execute("""
			ALTER TABLE crm_accounts ADD COLUMN 
			last_interaction_date TIMESTAMP WITH TIME ZONE
		""")
		
		# Add AI predictions to opportunities
		await connection.execute("""
			ALTER TABLE crm_opportunities ADD COLUMN 
			ai_close_date_prediction DATE
		""")
		
		await connection.execute("""
			ALTER TABLE crm_opportunities ADD COLUMN 
			ai_amount_prediction DECIMAL(15,2)
		""")
		
		await connection.execute("""
			ALTER TABLE crm_opportunities ADD COLUMN 
			competitive_threats JSONB DEFAULT '[]'
		""")
		
		# Add engagement tracking to activities
		await connection.execute("""
			ALTER TABLE crm_activities ADD COLUMN 
			engagement_score DECIMAL(5,2) CHECK (engagement_score >= 0 AND engagement_score <= 100)
		""")
		
		await connection.execute("""
			ALTER TABLE crm_activities ADD COLUMN 
			outcome_summary TEXT
		""")
		
		# Add performance metrics to campaigns
		await connection.execute("""
			ALTER TABLE crm_campaigns ADD COLUMN 
			leads_generated INTEGER DEFAULT 0 CHECK (leads_generated >= 0)
		""")
		
		await connection.execute("""
			ALTER TABLE crm_campaigns ADD COLUMN 
			opportunities_created INTEGER DEFAULT 0 CHECK (opportunities_created >= 0)
		""")
		
		await connection.execute("""
			ALTER TABLE crm_campaigns ADD COLUMN 
			revenue_attributed DECIMAL(15,2) DEFAULT 0 CHECK (revenue_attributed >= 0)
		""")
		
		# Create indexes for the new tables and columns
		
		# AI Insights indexes
		await connection.execute("CREATE INDEX idx_crm_ai_insights_tenant_id ON crm_ai_insights(tenant_id)")
		await connection.execute("CREATE INDEX idx_crm_ai_insights_entity ON crm_ai_insights(entity_type, entity_id)")
		await connection.execute("CREATE INDEX idx_crm_ai_insights_type ON crm_ai_insights(insight_type)")
		await connection.execute("CREATE INDEX idx_crm_ai_insights_generated_at ON crm_ai_insights(generated_at)")
		await connection.execute("CREATE INDEX idx_crm_ai_insights_expires_at ON crm_ai_insights(expires_at) WHERE expires_at IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_ai_insights_confidence ON crm_ai_insights(confidence_score)")
		
		# Communications indexes
		await connection.execute("CREATE INDEX idx_crm_communications_tenant_id ON crm_communications(tenant_id)")
		await connection.execute("CREATE INDEX idx_crm_communications_channel ON crm_communications(channel)")
		await connection.execute("CREATE INDEX idx_crm_communications_direction ON crm_communications(direction)")
		await connection.execute("CREATE INDEX idx_crm_communications_sent_at ON crm_communications(sent_at) WHERE sent_at IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_communications_contact_id ON crm_communications(contact_id) WHERE contact_id IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_communications_account_id ON crm_communications(account_id) WHERE account_id IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_communications_opportunity_id ON crm_communications(opportunity_id) WHERE opportunity_id IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_communications_thread_id ON crm_communications(thread_id) WHERE thread_id IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_communications_status ON crm_communications(status)")
		
		# Relationships indexes
		await connection.execute("CREATE INDEX idx_crm_contact_relationships_tenant_id ON crm_contact_relationships(tenant_id)")
		await connection.execute("CREATE INDEX idx_crm_contact_relationships_from_contact ON crm_contact_relationships(from_contact_id)")
		await connection.execute("CREATE INDEX idx_crm_contact_relationships_to_contact ON crm_contact_relationships(to_contact_id)")
		await connection.execute("CREATE INDEX idx_crm_contact_relationships_type ON crm_contact_relationships(relationship_type)")
		await connection.execute("CREATE INDEX idx_crm_contact_relationships_strength ON crm_contact_relationships(strength)")
		await connection.execute("CREATE INDEX idx_crm_contact_relationships_active ON crm_contact_relationships(is_active)")
		
		# Pipeline stages indexes
		await connection.execute("CREATE INDEX idx_crm_pipeline_stages_tenant_id ON crm_pipeline_stages(tenant_id)")
		await connection.execute("CREATE INDEX idx_crm_pipeline_stages_order ON crm_pipeline_stages(stage_order)")
		await connection.execute("CREATE INDEX idx_crm_pipeline_stages_active ON crm_pipeline_stages(is_active)")
		
		# Campaign members indexes
		await connection.execute("CREATE INDEX idx_crm_campaign_members_tenant_id ON crm_campaign_members(tenant_id)")
		await connection.execute("CREATE INDEX idx_crm_campaign_members_campaign_id ON crm_campaign_members(campaign_id)")
		await connection.execute("CREATE INDEX idx_crm_campaign_members_contact_id ON crm_campaign_members(contact_id) WHERE contact_id IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_campaign_members_lead_id ON crm_campaign_members(lead_id) WHERE lead_id IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_campaign_members_status ON crm_campaign_members(status)")
		await connection.execute("CREATE INDEX idx_crm_campaign_members_sent_date ON crm_campaign_members(sent_date) WHERE sent_date IS NOT NULL")
		
		# New column indexes
		await connection.execute("CREATE INDEX idx_crm_contacts_ai_engagement_score ON crm_contacts(ai_engagement_score) WHERE ai_engagement_score IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_contacts_last_engagement ON crm_contacts(last_engagement_date) WHERE last_engagement_date IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_accounts_last_interaction ON crm_accounts(last_interaction_date) WHERE last_interaction_date IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_opportunities_ai_close_date ON crm_opportunities(ai_close_date_prediction) WHERE ai_close_date_prediction IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_activities_engagement_score ON crm_activities(engagement_score) WHERE engagement_score IS NOT NULL")
		
		# Full-text search for communications
		await connection.execute("""
			CREATE INDEX idx_crm_communications_fulltext ON crm_communications 
			USING GIN(to_tsvector('english', 
				COALESCE(subject, '') || ' ' || 
				COALESCE(content, '')
			))
		""")
		
		# Insert default pipeline stages
		await connection.execute("""
			INSERT INTO crm_pipeline_stages (id, tenant_id, stage_name, stage_order, probability_percentage, is_closed_won, is_closed_lost, color_code, description, created_by, updated_by)
			VALUES 
			('default_stage_1', 'default', 'Prospecting', 1, 10, FALSE, FALSE, '#E3F2FD', 'Initial contact and research phase', 'system', 'system'),
			('default_stage_2', 'default', 'Qualification', 2, 25, FALSE, FALSE, '#BBDEFB', 'Qualifying leads and understanding needs', 'system', 'system'),
			('default_stage_3', 'default', 'Needs Analysis', 3, 40, FALSE, FALSE, '#90CAF9', 'Detailed needs assessment and solution design', 'system', 'system'),
			('default_stage_4', 'default', 'Value Proposition', 4, 60, FALSE, FALSE, '#64B5F6', 'Presenting value proposition and benefits', 'system', 'system'),
			('default_stage_5', 'default', 'Proposal', 5, 75, FALSE, FALSE, '#42A5F5', 'Formal proposal and negotiation phase', 'system', 'system'),
			('default_stage_6', 'default', 'Negotiation', 6, 90, FALSE, FALSE, '#2196F3', 'Final negotiations and contract terms', 'system', 'system'),
			('default_stage_7', 'default', 'Closed Won', 7, 100, TRUE, FALSE, '#4CAF50', 'Successfully closed deal', 'system', 'system'),
			('default_stage_8', 'default', 'Closed Lost', 8, 0, FALSE, TRUE, '#F44336', 'Lost opportunity', 'system', 'system')
		""")
	
	async def down(self, connection: asyncpg.Connection) -> None:
		"""Remove advanced CRM features"""
		
		# Remove new columns from existing tables
		await connection.execute("ALTER TABLE crm_contacts DROP COLUMN IF EXISTS ai_engagement_score")
		await connection.execute("ALTER TABLE crm_contacts DROP COLUMN IF EXISTS last_engagement_date")
		await connection.execute("ALTER TABLE crm_contacts DROP COLUMN IF EXISTS communication_preferences")
		
		await connection.execute("ALTER TABLE crm_accounts DROP COLUMN IF EXISTS key_contacts")
		await connection.execute("ALTER TABLE crm_accounts DROP COLUMN IF EXISTS last_interaction_date")
		
		await connection.execute("ALTER TABLE crm_opportunities DROP COLUMN IF EXISTS ai_close_date_prediction")
		await connection.execute("ALTER TABLE crm_opportunities DROP COLUMN IF EXISTS ai_amount_prediction")
		await connection.execute("ALTER TABLE crm_opportunities DROP COLUMN IF EXISTS competitive_threats")
		
		await connection.execute("ALTER TABLE crm_activities DROP COLUMN IF EXISTS engagement_score")
		await connection.execute("ALTER TABLE crm_activities DROP COLUMN IF EXISTS outcome_summary")
		
		await connection.execute("ALTER TABLE crm_campaigns DROP COLUMN IF EXISTS leads_generated")
		await connection.execute("ALTER TABLE crm_campaigns DROP COLUMN IF EXISTS opportunities_created")
		await connection.execute("ALTER TABLE crm_campaigns DROP COLUMN IF EXISTS revenue_attributed")
		
		# Drop new tables
		await connection.execute("DROP TABLE IF EXISTS crm_campaign_members CASCADE")
		await connection.execute("DROP TABLE IF EXISTS crm_pipeline_stages CASCADE")
		await connection.execute("DROP TABLE IF EXISTS crm_contact_relationships CASCADE")
		await connection.execute("DROP TABLE IF EXISTS crm_communications CASCADE")
		await connection.execute("DROP TABLE IF EXISTS crm_ai_insights CASCADE")
		
		# Drop new enum types
		await connection.execute("DROP TYPE IF EXISTS relationship_type CASCADE")
		await connection.execute("DROP TYPE IF EXISTS communication_direction CASCADE")
		await connection.execute("DROP TYPE IF EXISTS communication_channel CASCADE")
		await connection.execute("DROP TYPE IF EXISTS insight_type CASCADE")
	
	async def _validate_preconditions(self, connection: asyncpg.Connection, direction) -> None:
		"""Validate preconditions"""
		if direction.value == "up":
			# Ensure core tables exist
			required_tables = ["crm_contacts", "crm_accounts", "crm_leads", "crm_opportunities", "crm_activities", "crm_campaigns"]
			for table_name in required_tables:
				if not await table_exists(connection, table_name):
					raise Exception(f"Required table {table_name} does not exist")
	
	async def _validate_postconditions(self, connection: asyncpg.Connection, direction) -> None:
		"""Validate postconditions"""
		if direction.value == "up":
			# Ensure new tables were created
			new_tables = ["crm_ai_insights", "crm_communications", "crm_contact_relationships", "crm_pipeline_stages", "crm_campaign_members"]
			for table_name in new_tables:
				if not await table_exists(connection, table_name):
					raise Exception(f"Failed to create table {table_name}")
			
			# Ensure new columns were added
			new_columns = [
				("crm_contacts", "ai_engagement_score"),
				("crm_accounts", "key_contacts"),
				("crm_opportunities", "ai_close_date_prediction"),
				("crm_activities", "engagement_score"),
				("crm_campaigns", "leads_generated")
			]
			
			for table_name, column_name in new_columns:
				if not await column_exists(connection, table_name, column_name):
					raise Exception(f"Failed to add column {column_name} to table {table_name}")
	
	async def validate_schema_state(self, connection: asyncpg.Connection) -> bool:
		"""Validate schema state for advanced features"""
		try:
			# Check new tables exist
			new_tables = ["crm_ai_insights", "crm_communications", "crm_contact_relationships", "crm_pipeline_stages", "crm_campaign_members"]
			for table_name in new_tables:
				if not await table_exists(connection, table_name):
					return False
			
			# Check new columns exist
			if not await column_exists(connection, "crm_contacts", "ai_engagement_score"):
				return False
			
			if not await column_exists(connection, "crm_opportunities", "ai_close_date_prediction"):
				return False
			
			# Check default pipeline stages exist
			stage_count = await connection.fetchval("""
				SELECT COUNT(*) FROM crm_pipeline_stages WHERE tenant_id = 'default'
			""")
			
			if stage_count < 8:  # Should have 8 default stages
				return False
			
			return True
			
		except Exception:
			return False