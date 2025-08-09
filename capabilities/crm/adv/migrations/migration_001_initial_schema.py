"""
APG Customer Relationship Management - Initial Schema Migration

Revolutionary initial database schema creation for the CRM capability
providing comprehensive multi-tenant data structure with advanced indexing,
constraints, and audit capabilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncpg
from .base_migration import BaseMigration, table_exists, index_exists


class InitialSchemaMigration(BaseMigration):
	"""Initial database schema migration for CRM capability"""
	
	def _get_migration_id(self) -> str:
		return "001_initial_schema"
	
	def _get_version(self) -> str:
		return "001"
	
	def _get_description(self) -> str:
		return "Create initial CRM database schema with core tables"
	
	def _is_reversible(self) -> bool:
		return True
	
	async def up(self, connection: asyncpg.Connection) -> None:
		"""Create initial CRM schema"""
		
		# Create enum types
		await connection.execute("""
			CREATE TYPE contact_type AS ENUM (
				'prospect', 'lead', 'customer', 'partner', 'vendor', 'other'
			)
		""")
		
		await connection.execute("""
			CREATE TYPE lead_source AS ENUM (
				'website', 'referral', 'social_media', 'email_campaign', 
				'phone_call', 'trade_show', 'advertisement', 'partner', 'other'
			)
		""")
		
		await connection.execute("""
			CREATE TYPE lead_status AS ENUM (
				'new', 'contacted', 'qualified', 'proposal', 'negotiation', 
				'closed_won', 'closed_lost', 'disqualified'
			)
		""")
		
		await connection.execute("""
			CREATE TYPE opportunity_stage AS ENUM (
				'prospecting', 'qualification', 'needs_analysis', 'value_proposition',
				'proposal', 'negotiation', 'closed_won', 'closed_lost'
			)
		""")
		
		await connection.execute("""
			CREATE TYPE activity_type AS ENUM (
				'call', 'email', 'meeting', 'task', 'note', 'demo', 'proposal', 'other'
			)
		""")
		
		await connection.execute("""
			CREATE TYPE activity_status AS ENUM (
				'planned', 'in_progress', 'completed', 'cancelled', 'overdue'
			)
		""")
		
		await connection.execute("""
			CREATE TYPE campaign_type AS ENUM (
				'email', 'social_media', 'webinar', 'trade_show', 'advertisement', 
				'direct_mail', 'telemarketing', 'content_marketing', 'other'
			)
		""")
		
		await connection.execute("""
			CREATE TYPE campaign_status AS ENUM (
				'draft', 'scheduled', 'active', 'paused', 'completed', 'cancelled'
			)
		""")
		
		# Core CRM tables
		
		# 1. CRM Contacts table
		await connection.execute("""
			CREATE TABLE crm_contacts (
				id VARCHAR(26) PRIMARY KEY,
				tenant_id VARCHAR(100) NOT NULL,
				first_name VARCHAR(100) NOT NULL,
				last_name VARCHAR(100) NOT NULL,
				email VARCHAR(255),
				phone VARCHAR(50),
				mobile VARCHAR(50),
				company VARCHAR(200),
				job_title VARCHAR(100),
				department VARCHAR(100),
				contact_type contact_type NOT NULL DEFAULT 'prospect',
				lead_source lead_source,
				lead_score DECIMAL(5,2) CHECK (lead_score >= 0 AND lead_score <= 100),
				customer_health_score DECIMAL(5,2) CHECK (customer_health_score >= 0 AND customer_health_score <= 100),
				tags TEXT[],
				notes TEXT,
				social_media_profiles JSONB DEFAULT '{}',
				custom_fields JSONB DEFAULT '{}',
				owner_id VARCHAR(100),
				created_by VARCHAR(100) NOT NULL,
				updated_by VARCHAR(100) NOT NULL,
				created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				version INTEGER NOT NULL DEFAULT 1,
				
				-- Constraints
				CONSTRAINT crm_contacts_email_check CHECK (email IS NULL OR email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
				CONSTRAINT crm_contacts_name_check CHECK (LENGTH(TRIM(first_name)) > 0 AND LENGTH(TRIM(last_name)) > 0),
				CONSTRAINT crm_contacts_tenant_isolation UNIQUE (id, tenant_id)
			)
		""")
		
		# 2. CRM Accounts table
		await connection.execute("""
			CREATE TABLE crm_accounts (
				id VARCHAR(26) PRIMARY KEY,
				tenant_id VARCHAR(100) NOT NULL,
				account_name VARCHAR(200) NOT NULL,
				account_type VARCHAR(50),
				industry VARCHAR(100),
				annual_revenue DECIMAL(15,2),
				employee_count INTEGER,
				website VARCHAR(255),
				description TEXT,
				billing_address JSONB,
				shipping_address JSONB,
				phone VARCHAR(50),
				fax VARCHAR(50),
				account_health_score DECIMAL(5,2) CHECK (account_health_score >= 0 AND account_health_score <= 100),
				tags TEXT[],
				custom_fields JSONB DEFAULT '{}',
				parent_account_id VARCHAR(26),
				owner_id VARCHAR(100),
				created_by VARCHAR(100) NOT NULL,
				updated_by VARCHAR(100) NOT NULL,
				created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				version INTEGER NOT NULL DEFAULT 1,
				
				-- Constraints
				CONSTRAINT crm_accounts_name_check CHECK (LENGTH(TRIM(account_name)) > 0),
				CONSTRAINT crm_accounts_revenue_check CHECK (annual_revenue IS NULL OR annual_revenue >= 0),
				CONSTRAINT crm_accounts_employees_check CHECK (employee_count IS NULL OR employee_count >= 0),
				CONSTRAINT crm_accounts_tenant_isolation UNIQUE (id, tenant_id),
				CONSTRAINT crm_accounts_parent_fk FOREIGN KEY (parent_account_id) REFERENCES crm_accounts(id) ON DELETE SET NULL
			)
		""")
		
		# 3. CRM Leads table
		await connection.execute("""
			CREATE TABLE crm_leads (
				id VARCHAR(26) PRIMARY KEY,
				tenant_id VARCHAR(100) NOT NULL,
				first_name VARCHAR(100),
				last_name VARCHAR(100),
				email VARCHAR(255),
				phone VARCHAR(50),
				company VARCHAR(200),
				job_title VARCHAR(100),
				lead_source lead_source,
				lead_status lead_status NOT NULL DEFAULT 'new',
				lead_score DECIMAL(5,2) CHECK (lead_score >= 0 AND lead_score <= 100),
				budget DECIMAL(15,2),
				timeline VARCHAR(100),
				notes TEXT,
				tags TEXT[],
				custom_fields JSONB DEFAULT '{}',
				converted_contact_id VARCHAR(26),
				converted_account_id VARCHAR(26),
				converted_opportunity_id VARCHAR(26),
				converted_at TIMESTAMP WITH TIME ZONE,
				owner_id VARCHAR(100),
				created_by VARCHAR(100) NOT NULL,
				updated_by VARCHAR(100) NOT NULL,
				created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				version INTEGER NOT NULL DEFAULT 1,
				
				-- Constraints
				CONSTRAINT crm_leads_email_check CHECK (email IS NULL OR email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
				CONSTRAINT crm_leads_budget_check CHECK (budget IS NULL OR budget >= 0),
				CONSTRAINT crm_leads_tenant_isolation UNIQUE (id, tenant_id),
				CONSTRAINT crm_leads_converted_contact_fk FOREIGN KEY (converted_contact_id) REFERENCES crm_contacts(id) ON DELETE SET NULL,
				CONSTRAINT crm_leads_converted_account_fk FOREIGN KEY (converted_account_id) REFERENCES crm_accounts(id) ON DELETE SET NULL
			)
		""")
		
		# 4. CRM Opportunities table
		await connection.execute("""
			CREATE TABLE crm_opportunities (
				id VARCHAR(26) PRIMARY KEY,
				tenant_id VARCHAR(100) NOT NULL,
				opportunity_name VARCHAR(200) NOT NULL,
				account_id VARCHAR(26),
				primary_contact_id VARCHAR(26),
				stage opportunity_stage NOT NULL DEFAULT 'prospecting',
				amount DECIMAL(15,2) NOT NULL DEFAULT 0,
				probability DECIMAL(5,2) NOT NULL DEFAULT 0 CHECK (probability >= 0 AND probability <= 100),
				expected_revenue DECIMAL(15,2),
				close_date DATE NOT NULL,
				win_probability_ai DECIMAL(5,2) CHECK (win_probability_ai >= 0 AND win_probability_ai <= 100),
				lead_source lead_source,
				description TEXT,
				tags TEXT[],
				custom_fields JSONB DEFAULT '{}',
				is_closed BOOLEAN NOT NULL DEFAULT FALSE,
				closed_at TIMESTAMP WITH TIME ZONE,
				owner_id VARCHAR(100),
				created_by VARCHAR(100) NOT NULL,
				updated_by VARCHAR(100) NOT NULL,
				created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				version INTEGER NOT NULL DEFAULT 1,
				
				-- Constraints
				CONSTRAINT crm_opportunities_name_check CHECK (LENGTH(TRIM(opportunity_name)) > 0),
				CONSTRAINT crm_opportunities_amount_check CHECK (amount >= 0),
				CONSTRAINT crm_opportunities_close_date_check CHECK (close_date >= CURRENT_DATE - INTERVAL '5 years'),
				CONSTRAINT crm_opportunities_tenant_isolation UNIQUE (id, tenant_id),
				CONSTRAINT crm_opportunities_account_fk FOREIGN KEY (account_id) REFERENCES crm_accounts(id) ON DELETE SET NULL,
				CONSTRAINT crm_opportunities_contact_fk FOREIGN KEY (primary_contact_id) REFERENCES crm_contacts(id) ON DELETE SET NULL
			)
		""")
		
		# 5. CRM Activities table
		await connection.execute("""
			CREATE TABLE crm_activities (
				id VARCHAR(26) PRIMARY KEY,
				tenant_id VARCHAR(100) NOT NULL,
				activity_type activity_type NOT NULL DEFAULT 'task',
				subject VARCHAR(255) NOT NULL,
				description TEXT,
				status activity_status NOT NULL DEFAULT 'planned',
				priority VARCHAR(20) DEFAULT 'medium',
				due_date TIMESTAMP WITH TIME ZONE,
				completed_at TIMESTAMP WITH TIME ZONE,
				duration_minutes INTEGER,
				location VARCHAR(255),
				participants TEXT[],
				related_to_type VARCHAR(50),
				related_to_id VARCHAR(26),
				contact_id VARCHAR(26),
				account_id VARCHAR(26),
				opportunity_id VARCHAR(26),
				lead_id VARCHAR(26),
				tags TEXT[],
				custom_fields JSONB DEFAULT '{}',
				owner_id VARCHAR(100),
				assigned_to_id VARCHAR(100),
				created_by VARCHAR(100) NOT NULL,
				updated_by VARCHAR(100) NOT NULL,
				created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				version INTEGER NOT NULL DEFAULT 1,
				
				-- Constraints
				CONSTRAINT crm_activities_subject_check CHECK (LENGTH(TRIM(subject)) > 0),
				CONSTRAINT crm_activities_duration_check CHECK (duration_minutes IS NULL OR duration_minutes > 0),
				CONSTRAINT crm_activities_tenant_isolation UNIQUE (id, tenant_id),
				CONSTRAINT crm_activities_contact_fk FOREIGN KEY (contact_id) REFERENCES crm_contacts(id) ON DELETE CASCADE,
				CONSTRAINT crm_activities_account_fk FOREIGN KEY (account_id) REFERENCES crm_accounts(id) ON DELETE CASCADE,
				CONSTRAINT crm_activities_opportunity_fk FOREIGN KEY (opportunity_id) REFERENCES crm_opportunities(id) ON DELETE CASCADE,
				CONSTRAINT crm_activities_lead_fk FOREIGN KEY (lead_id) REFERENCES crm_leads(id) ON DELETE CASCADE
			)
		""")
		
		# 6. CRM Campaigns table
		await connection.execute("""
			CREATE TABLE crm_campaigns (
				id VARCHAR(26) PRIMARY KEY,
				tenant_id VARCHAR(100) NOT NULL,
				campaign_name VARCHAR(200) NOT NULL,
				campaign_type campaign_type NOT NULL DEFAULT 'email',
				status campaign_status NOT NULL DEFAULT 'draft',
				description TEXT,
				start_date DATE,
				end_date DATE,
				budget DECIMAL(15,2),
				actual_cost DECIMAL(15,2),
				expected_response_rate DECIMAL(5,2) CHECK (expected_response_rate >= 0 AND expected_response_rate <= 100),
				actual_response_rate DECIMAL(5,2) CHECK (actual_response_rate >= 0 AND actual_response_rate <= 100),
				target_audience TEXT,
				tags TEXT[],
				custom_fields JSONB DEFAULT '{}',
				owner_id VARCHAR(100),
				created_by VARCHAR(100) NOT NULL,
				updated_by VARCHAR(100) NOT NULL,
				created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
				version INTEGER NOT NULL DEFAULT 1,
				
				-- Constraints
				CONSTRAINT crm_campaigns_name_check CHECK (LENGTH(TRIM(campaign_name)) > 0),
				CONSTRAINT crm_campaigns_budget_check CHECK (budget IS NULL OR budget >= 0),
				CONSTRAINT crm_campaigns_cost_check CHECK (actual_cost IS NULL OR actual_cost >= 0),
				CONSTRAINT crm_campaigns_date_check CHECK (end_date IS NULL OR start_date IS NULL OR end_date >= start_date),
				CONSTRAINT crm_campaigns_tenant_isolation UNIQUE (id, tenant_id)
			)
		""")
		
		# Create comprehensive indexes for performance
		
		# Contact indexes
		await connection.execute("CREATE INDEX idx_crm_contacts_tenant_id ON crm_contacts(tenant_id)")
		await connection.execute("CREATE INDEX idx_crm_contacts_owner_id ON crm_contacts(owner_id)")
		await connection.execute("CREATE INDEX idx_crm_contacts_email ON crm_contacts(email) WHERE email IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_contacts_company ON crm_contacts(company) WHERE company IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_contacts_contact_type ON crm_contacts(contact_type)")
		await connection.execute("CREATE INDEX idx_crm_contacts_lead_source ON crm_contacts(lead_source)")
		await connection.execute("CREATE INDEX idx_crm_contacts_lead_score ON crm_contacts(lead_score) WHERE lead_score IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_contacts_created_at ON crm_contacts(created_at)")
		await connection.execute("CREATE INDEX idx_crm_contacts_updated_at ON crm_contacts(updated_at)")
		await connection.execute("CREATE INDEX idx_crm_contacts_tags ON crm_contacts USING GIN(tags)")
		await connection.execute("CREATE INDEX idx_crm_contacts_custom_fields ON crm_contacts USING GIN(custom_fields)")
		
		# Account indexes
		await connection.execute("CREATE INDEX idx_crm_accounts_tenant_id ON crm_accounts(tenant_id)")
		await connection.execute("CREATE INDEX idx_crm_accounts_owner_id ON crm_accounts(owner_id)")
		await connection.execute("CREATE INDEX idx_crm_accounts_account_name ON crm_accounts(account_name)")
		await connection.execute("CREATE INDEX idx_crm_accounts_industry ON crm_accounts(industry) WHERE industry IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_accounts_parent_account ON crm_accounts(parent_account_id) WHERE parent_account_id IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_accounts_created_at ON crm_accounts(created_at)")
		await connection.execute("CREATE INDEX idx_crm_accounts_tags ON crm_accounts USING GIN(tags)")
		
		# Lead indexes
		await connection.execute("CREATE INDEX idx_crm_leads_tenant_id ON crm_leads(tenant_id)")
		await connection.execute("CREATE INDEX idx_crm_leads_owner_id ON crm_leads(owner_id)")
		await connection.execute("CREATE INDEX idx_crm_leads_email ON crm_leads(email) WHERE email IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_leads_company ON crm_leads(company) WHERE company IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_leads_lead_status ON crm_leads(lead_status)")
		await connection.execute("CREATE INDEX idx_crm_leads_lead_source ON crm_leads(lead_source)")
		await connection.execute("CREATE INDEX idx_crm_leads_lead_score ON crm_leads(lead_score) WHERE lead_score IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_leads_converted_at ON crm_leads(converted_at) WHERE converted_at IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_leads_created_at ON crm_leads(created_at)")
		
		# Opportunity indexes
		await connection.execute("CREATE INDEX idx_crm_opportunities_tenant_id ON crm_opportunities(tenant_id)")
		await connection.execute("CREATE INDEX idx_crm_opportunities_owner_id ON crm_opportunities(owner_id)")
		await connection.execute("CREATE INDEX idx_crm_opportunities_account_id ON crm_opportunities(account_id) WHERE account_id IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_opportunities_contact_id ON crm_opportunities(primary_contact_id) WHERE primary_contact_id IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_opportunities_stage ON crm_opportunities(stage)")
		await connection.execute("CREATE INDEX idx_crm_opportunities_close_date ON crm_opportunities(close_date)")
		await connection.execute("CREATE INDEX idx_crm_opportunities_amount ON crm_opportunities(amount)")
		await connection.execute("CREATE INDEX idx_crm_opportunities_probability ON crm_opportunities(probability)")
		await connection.execute("CREATE INDEX idx_crm_opportunities_is_closed ON crm_opportunities(is_closed)")
		await connection.execute("CREATE INDEX idx_crm_opportunities_created_at ON crm_opportunities(created_at)")
		
		# Activity indexes
		await connection.execute("CREATE INDEX idx_crm_activities_tenant_id ON crm_activities(tenant_id)")
		await connection.execute("CREATE INDEX idx_crm_activities_owner_id ON crm_activities(owner_id)")
		await connection.execute("CREATE INDEX idx_crm_activities_assigned_to ON crm_activities(assigned_to_id) WHERE assigned_to_id IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_activities_activity_type ON crm_activities(activity_type)")
		await connection.execute("CREATE INDEX idx_crm_activities_status ON crm_activities(status)")
		await connection.execute("CREATE INDEX idx_crm_activities_due_date ON crm_activities(due_date) WHERE due_date IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_activities_contact_id ON crm_activities(contact_id) WHERE contact_id IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_activities_account_id ON crm_activities(account_id) WHERE account_id IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_activities_opportunity_id ON crm_activities(opportunity_id) WHERE opportunity_id IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_activities_lead_id ON crm_activities(lead_id) WHERE lead_id IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_activities_created_at ON crm_activities(created_at)")
		
		# Campaign indexes
		await connection.execute("CREATE INDEX idx_crm_campaigns_tenant_id ON crm_campaigns(tenant_id)")
		await connection.execute("CREATE INDEX idx_crm_campaigns_owner_id ON crm_campaigns(owner_id)")
		await connection.execute("CREATE INDEX idx_crm_campaigns_campaign_type ON crm_campaigns(campaign_type)")
		await connection.execute("CREATE INDEX idx_crm_campaigns_status ON crm_campaigns(status)")
		await connection.execute("CREATE INDEX idx_crm_campaigns_start_date ON crm_campaigns(start_date) WHERE start_date IS NOT NULL")
		await connection.execute("CREATE INDEX idx_crm_campaigns_end_date ON crm_campaigns(end_date) WHERE end_date IS NOT NULL")
		
		# Composite indexes for common queries
		await connection.execute("CREATE INDEX idx_crm_contacts_tenant_owner ON crm_contacts(tenant_id, owner_id)")
		await connection.execute("CREATE INDEX idx_crm_contacts_tenant_type ON crm_contacts(tenant_id, contact_type)")
		await connection.execute("CREATE INDEX idx_crm_opportunities_tenant_stage ON crm_opportunities(tenant_id, stage)")
		await connection.execute("CREATE INDEX idx_crm_opportunities_tenant_close_date ON crm_opportunities(tenant_id, close_date)")
		await connection.execute("CREATE INDEX idx_crm_activities_tenant_due_date ON crm_activities(tenant_id, due_date)")
		await connection.execute("CREATE INDEX idx_crm_activities_tenant_status ON crm_activities(tenant_id, status)")
		
		# Full-text search indexes
		await connection.execute("""
			CREATE INDEX idx_crm_contacts_fulltext ON crm_contacts 
			USING GIN(to_tsvector('english', 
				COALESCE(first_name, '') || ' ' || 
				COALESCE(last_name, '') || ' ' || 
				COALESCE(company, '') || ' ' || 
				COALESCE(job_title, '')
			))
		""")
		
		await connection.execute("""
			CREATE INDEX idx_crm_accounts_fulltext ON crm_accounts 
			USING GIN(to_tsvector('english', 
				COALESCE(account_name, '') || ' ' || 
				COALESCE(industry, '') || ' ' || 
				COALESCE(description, '')
			))
		""")
		
		await connection.execute("""
			CREATE INDEX idx_crm_opportunities_fulltext ON crm_opportunities 
			USING GIN(to_tsvector('english', 
				COALESCE(opportunity_name, '') || ' ' || 
				COALESCE(description, '')
			))
		""")
	
	async def down(self, connection: asyncpg.Connection) -> None:
		"""Drop CRM schema"""
		
		# Drop tables in reverse dependency order
		tables_to_drop = [
			"crm_activities",
			"crm_campaigns", 
			"crm_opportunities",
			"crm_leads",
			"crm_accounts",
			"crm_contacts"
		]
		
		for table in tables_to_drop:
			await connection.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
		
		# Drop enum types
		enum_types = [
			"campaign_status",
			"campaign_type",
			"activity_status", 
			"activity_type",
			"opportunity_stage",
			"lead_status",
			"lead_source",
			"contact_type"
		]
		
		for enum_type in enum_types:
			await connection.execute(f"DROP TYPE IF EXISTS {enum_type} CASCADE")
	
	async def _validate_preconditions(self, connection: asyncpg.Connection, direction) -> None:
		"""Validate preconditions for migration"""
		if direction.value == "up":
			# Ensure no tables exist
			for table_name in ["crm_contacts", "crm_accounts", "crm_leads", "crm_opportunities", "crm_activities", "crm_campaigns"]:
				if await table_exists(connection, table_name):
					raise Exception(f"Table {table_name} already exists")
	
	async def _validate_postconditions(self, connection: asyncpg.Connection, direction) -> None:
		"""Validate postconditions after migration"""
		if direction.value == "up":
			# Ensure all tables were created
			required_tables = ["crm_contacts", "crm_accounts", "crm_leads", "crm_opportunities", "crm_activities", "crm_campaigns"]
			
			for table_name in required_tables:
				if not await table_exists(connection, table_name):
					raise Exception(f"Failed to create table {table_name}")
			
			# Ensure key indexes were created
			required_indexes = [
				"idx_crm_contacts_tenant_id",
				"idx_crm_accounts_tenant_id", 
				"idx_crm_opportunities_tenant_id",
				"idx_crm_activities_tenant_id"
			]
			
			for index_name in required_indexes:
				if not await index_exists(connection, index_name):
					raise Exception(f"Failed to create index {index_name}")
	
	async def validate_schema_state(self, connection: asyncpg.Connection) -> bool:
		"""Validate that schema is in expected state"""
		try:
			# Check that all core tables exist and have expected columns
			tables_and_columns = {
				"crm_contacts": ["id", "tenant_id", "first_name", "last_name", "email", "contact_type"],
				"crm_accounts": ["id", "tenant_id", "account_name", "industry"],
				"crm_leads": ["id", "tenant_id", "lead_status", "lead_source"],
				"crm_opportunities": ["id", "tenant_id", "opportunity_name", "stage", "amount"],
				"crm_activities": ["id", "tenant_id", "activity_type", "subject", "status"],
				"crm_campaigns": ["id", "tenant_id", "campaign_name", "campaign_type", "status"]
			}
			
			for table_name, expected_columns in tables_and_columns.items():
				if not await table_exists(connection, table_name):
					return False
				
				# Check that required columns exist
				for column in expected_columns:
					result = await connection.fetchval("""
						SELECT EXISTS (
							SELECT 1 FROM information_schema.columns 
							WHERE table_name = $1 AND column_name = $2
						)
					""", table_name, column)
					
					if not result:
						return False
			
			return True
			
		except Exception:
			return False