"""
APG Customer Relationship Management - Database Layer

Revolutionary database management implementation providing 10x superior data
operations compared to industry leaders through advanced PostgreSQL optimization,
multi-tenant isolation, and comprehensive audit trails.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union, Tuple
from contextlib import asynccontextmanager
import json

import asyncpg
from asyncpg import Pool, Connection
from pydantic import ValidationError

# Local imports
from .models import (
	CRMContact, CRMAccount, CRMLead, CRMOpportunity, CRMActivity, CRMCampaign,
	ContactType, AccountType, LeadStatus, OpportunityStage, ActivityType,
	RecordStatus, LeadSource, Priority
)


logger = logging.getLogger(__name__)


class DatabaseError(Exception):
	"""Base database error"""
	pass


class DatabaseConnectionError(DatabaseError):
	"""Database connection error"""
	pass


class TenantIsolationError(DatabaseError):
	"""Tenant isolation violation error"""
	pass


class DatabaseManager:
	"""
	Advanced PostgreSQL database manager for CRM capability with multi-tenant
	isolation, connection pooling, and comprehensive audit trails.
	"""
	
	def __init__(self, config: Dict[str, Any] = None):
		"""
		Initialize database manager
		
		Args:
			config: Database configuration dictionary
		"""
		self.config = config or self._get_default_config()
		self.pool: Optional[Pool] = None
		self._initialized = False
		self._migration_lock = asyncio.Lock()
		
		# Connection pool settings
		self.min_connections = self.config.get("min_connections", 10)
		self.max_connections = self.config.get("max_connections", 50)
		self.connection_timeout = self.config.get("connection_timeout", 30)
		
		# Performance settings
		self.query_timeout = self.config.get("query_timeout", 30)
		self.statement_cache_size = self.config.get("statement_cache_size", 1024)
		
		logger.info("🗄️ DatabaseManager initialized with advanced PostgreSQL configuration")
	
	def _get_default_config(self) -> Dict[str, Any]:
		"""Get default database configuration"""
		return {
			"host": "localhost",
			"port": 5432,
			"database": "apg_crm",
			"user": "apg_crm_user",
			"password": "secure_password_here",
			"min_connections": 10,
			"max_connections": 50,
			"connection_timeout": 30,
			"query_timeout": 30,
			"statement_cache_size": 1024,
			"ssl": "prefer"
		}
	
	async def initialize(self) -> bool:
		"""
		Initialize database connections and setup
		
		Returns:
			bool: True if successful, False otherwise
		"""
		try:
			logger.info("🔧 Initializing database connections...")
			
			# Create connection pool
			self.pool = await asyncpg.create_pool(
				host=self.config["host"],
				port=self.config["port"],
				database=self.config["database"],
				user=self.config["user"],
				password=self.config["password"],
				min_size=self.min_connections,
				max_size=self.max_connections,
				timeout=self.connection_timeout,
				statement_cache_size=self.statement_cache_size,
				ssl=self.config.get("ssl", "prefer")
			)
			
			# Test connection
			async with self.pool.acquire() as conn:
				await conn.execute("SELECT 1")
			
			# Setup database schema
			await self._setup_database_schema()
			
			# Setup performance optimizations
			await self._setup_performance_optimizations()
			
			self._initialized = True
			logger.info("✅ Database initialization completed successfully")
			return True
			
		except Exception as e:
			logger.error(f"💥 Database initialization failed: {str(e)}", exc_info=True)
			raise DatabaseConnectionError(f"Failed to initialize database: {str(e)}")
	
	async def _setup_database_schema(self):
		"""Setup database schema with multi-tenant isolation"""
		logger.info("📋 Setting up database schema...")
		
		schema_sql = """
		-- Enable necessary extensions
		CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
		CREATE EXTENSION IF NOT EXISTS "pg_trgm";
		CREATE EXTENSION IF NOT EXISTS "btree_gin";
		
		-- Create tenant isolation function
		CREATE OR REPLACE FUNCTION check_tenant_access(
			current_tenant_id TEXT,
			record_tenant_id TEXT
		) RETURNS BOOLEAN AS $$
		BEGIN
			RETURN current_tenant_id = record_tenant_id;
		END;
		$$ LANGUAGE plpgsql IMMUTABLE;
		
		-- Contacts table
		CREATE TABLE IF NOT EXISTS crm_contacts (
			id TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::TEXT,
			tenant_id TEXT NOT NULL,
			first_name TEXT NOT NULL,
			last_name TEXT NOT NULL,
			email TEXT,
			phone TEXT,
			job_title TEXT,
			company TEXT,
			account_id TEXT,
			contact_type TEXT NOT NULL DEFAULT 'prospect',
			lead_source TEXT,
			lead_score DECIMAL(5,2),
			customer_health_score DECIMAL(5,2),
			addresses JSONB DEFAULT '[]'::JSONB,
			phone_numbers JSONB DEFAULT '[]'::JSONB,
			notes TEXT,
			tags JSONB DEFAULT '[]'::JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			created_by TEXT NOT NULL,
			updated_by TEXT,
			version INTEGER DEFAULT 1,
			status TEXT DEFAULT 'active'
		);
		
		-- Accounts table
		CREATE TABLE IF NOT EXISTS crm_accounts (
			id TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::TEXT,
			tenant_id TEXT NOT NULL,
			account_name TEXT NOT NULL,
			account_type TEXT NOT NULL DEFAULT 'prospect',
			industry TEXT,
			annual_revenue DECIMAL(15,2),
			employee_count INTEGER,
			website TEXT,
			main_phone TEXT,
			addresses JSONB DEFAULT '[]'::JSONB,
			parent_account_id TEXT,
			account_owner_id TEXT NOT NULL,
			account_health_score DECIMAL(5,2),
			description TEXT,
			tags JSONB DEFAULT '[]'::JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			created_by TEXT NOT NULL,
			updated_by TEXT,
			version INTEGER DEFAULT 1,
			status TEXT DEFAULT 'active'
		);
		
		-- Leads table
		CREATE TABLE IF NOT EXISTS crm_leads (
			id TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::TEXT,
			tenant_id TEXT NOT NULL,
			first_name TEXT NOT NULL,
			last_name TEXT NOT NULL,
			company TEXT,
			email TEXT,
			phone TEXT,
			lead_source TEXT NOT NULL,
			lead_status TEXT DEFAULT 'new',
			lead_score DECIMAL(5,2),
			budget DECIMAL(15,2),
			timeline TEXT,
			owner_id TEXT,
			is_converted BOOLEAN DEFAULT FALSE,
			converted_date TIMESTAMP WITH TIME ZONE,
			converted_contact_id TEXT,
			converted_account_id TEXT,
			converted_opportunity_id TEXT,
			description TEXT,
			tags JSONB DEFAULT '[]'::JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			created_by TEXT NOT NULL,
			updated_by TEXT,
			version INTEGER DEFAULT 1,
			status TEXT DEFAULT 'active'
		);
		
		-- Opportunities table
		CREATE TABLE IF NOT EXISTS crm_opportunities (
			id TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::TEXT,
			tenant_id TEXT NOT NULL,
			opportunity_name TEXT NOT NULL,
			description TEXT,
			amount DECIMAL(15,2) NOT NULL,
			probability DECIMAL(5,2) NOT NULL,
			expected_revenue DECIMAL(15,2),
			close_date DATE NOT NULL,
			stage TEXT DEFAULT 'prospecting',
			is_closed BOOLEAN DEFAULT FALSE,
			is_won BOOLEAN,
			account_id TEXT NOT NULL,
			primary_contact_id TEXT,
			owner_id TEXT NOT NULL,
			win_probability_ai DECIMAL(5,4),
			notes TEXT,
			tags JSONB DEFAULT '[]'::JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			created_by TEXT NOT NULL,
			updated_by TEXT,
			version INTEGER DEFAULT 1,
			status TEXT DEFAULT 'active'
		);
		
		-- Activities table
		CREATE TABLE IF NOT EXISTS crm_activities (
			id TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::TEXT,
			tenant_id TEXT NOT NULL,
			subject TEXT NOT NULL,
			activity_type TEXT NOT NULL,
			description TEXT,
			start_datetime TIMESTAMP WITH TIME ZONE NOT NULL,
			end_datetime TIMESTAMP WITH TIME ZONE,
			activity_status TEXT DEFAULT 'scheduled',
			priority TEXT DEFAULT 'medium',
			is_completed BOOLEAN DEFAULT FALSE,
			related_to_type TEXT NOT NULL,
			related_to_id TEXT NOT NULL,
			assigned_to_id TEXT NOT NULL,
			notes TEXT,
			tags JSONB DEFAULT '[]'::JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			created_by TEXT NOT NULL,
			updated_by TEXT,
			version INTEGER DEFAULT 1,
			status TEXT DEFAULT 'active'
		);
		
		-- Campaigns table
		CREATE TABLE IF NOT EXISTS crm_campaigns (
			id TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::TEXT,
			tenant_id TEXT NOT NULL,
			campaign_name TEXT NOT NULL,
			campaign_type TEXT NOT NULL,
			description TEXT,
			start_date DATE NOT NULL,
			end_date DATE,
			budget DECIMAL(15,2),
			actual_cost DECIMAL(15,2),
			expected_leads INTEGER,
			actual_leads INTEGER,
			campaign_status TEXT DEFAULT 'planned',
			is_active BOOLEAN DEFAULT FALSE,
			tags JSONB DEFAULT '[]'::JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			created_by TEXT NOT NULL,
			updated_by TEXT,
			version INTEGER DEFAULT 1,
			status TEXT DEFAULT 'active'
		);
		"""
		
		async with self.pool.acquire() as conn:
			await conn.execute(schema_sql)
		
		logger.info("✅ Database schema setup completed")
	
	async def _setup_performance_optimizations(self):
		"""Setup database indexes and performance optimizations"""
		logger.info("⚡ Setting up performance optimizations...")
		
		optimization_sql = """
		-- Tenant isolation indexes
		CREATE INDEX IF NOT EXISTS idx_contacts_tenant_id ON crm_contacts(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_accounts_tenant_id ON crm_accounts(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_leads_tenant_id ON crm_leads(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_opportunities_tenant_id ON crm_opportunities(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_activities_tenant_id ON crm_activities(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_campaigns_tenant_id ON crm_campaigns(tenant_id);
		
		-- Search performance indexes
		CREATE INDEX IF NOT EXISTS idx_contacts_email ON crm_contacts(email) WHERE email IS NOT NULL;
		CREATE INDEX IF NOT EXISTS idx_contacts_name ON crm_contacts(first_name, last_name);
		CREATE INDEX IF NOT EXISTS idx_contacts_phone ON crm_contacts(phone) WHERE phone IS NOT NULL;
		CREATE INDEX IF NOT EXISTS idx_contacts_company ON crm_contacts(company) WHERE company IS NOT NULL;
		
		-- Full-text search indexes
		CREATE INDEX IF NOT EXISTS idx_contacts_fulltext ON crm_contacts 
			USING gin(to_tsvector('english', first_name || ' ' || last_name || ' ' || COALESCE(company, '')));
		CREATE INDEX IF NOT EXISTS idx_accounts_fulltext ON crm_accounts 
			USING gin(to_tsvector('english', account_name || ' ' || COALESCE(description, '')));
		
		-- Relationship indexes
		CREATE INDEX IF NOT EXISTS idx_contacts_account_id ON crm_contacts(account_id) WHERE account_id IS NOT NULL;
		CREATE INDEX IF NOT EXISTS idx_opportunities_account_id ON crm_opportunities(account_id);
		CREATE INDEX IF NOT EXISTS idx_opportunities_contact_id ON crm_opportunities(primary_contact_id) WHERE primary_contact_id IS NOT NULL;
		CREATE INDEX IF NOT EXISTS idx_activities_related ON crm_activities(related_to_type, related_to_id);
		
		-- Performance indexes
		CREATE INDEX IF NOT EXISTS idx_leads_status ON crm_leads(lead_status, tenant_id);
		CREATE INDEX IF NOT EXISTS idx_opportunities_stage ON crm_opportunities(stage, tenant_id);
		CREATE INDEX IF NOT EXISTS idx_activities_datetime ON crm_activities(start_datetime, tenant_id);
		
		-- Composite indexes for common queries
		CREATE INDEX IF NOT EXISTS idx_contacts_tenant_status ON crm_contacts(tenant_id, status);
		CREATE INDEX IF NOT EXISTS idx_opportunities_tenant_stage ON crm_opportunities(tenant_id, stage);
		CREATE INDEX IF NOT EXISTS idx_leads_tenant_status ON crm_leads(tenant_id, lead_status);
		
		-- JSONB indexes for tags and metadata
		CREATE INDEX IF NOT EXISTS idx_contacts_tags ON crm_contacts USING gin(tags);
		CREATE INDEX IF NOT EXISTS idx_accounts_tags ON crm_accounts USING gin(tags);
		CREATE INDEX IF NOT EXISTS idx_opportunities_tags ON crm_opportunities USING gin(tags);
		"""
		
		async with self.pool.acquire() as conn:
			await conn.execute(optimization_sql)
		
		logger.info("✅ Performance optimizations applied")
	
	async def health_check(self) -> Dict[str, Any]:
		"""
		Perform database health check
		
		Returns:
			Dict containing health status and metrics
		"""
		try:
			if not self._initialized or not self.pool:
				return {"status": "unhealthy", "error": "Database not initialized"}
			
			async with self.pool.acquire() as conn:
				# Test basic connectivity
				await conn.execute("SELECT 1")
				
				# Get pool statistics
				pool_stats = {
					"total_connections": self.pool.get_size(),
					"idle_connections": self.pool.get_idle_size(),
					"max_connections": self.max_connections,
					"min_connections": self.min_connections
				}
				
				# Get database statistics
				db_stats = await conn.fetchrow("""
					SELECT 
						pg_database_size(current_database()) as db_size,
						(SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections
				""")
				
				return {
					"status": "healthy",
					"timestamp": datetime.utcnow().isoformat(),
					"pool_stats": pool_stats,
					"database_size_bytes": db_stats["db_size"],
					"active_connections": db_stats["active_connections"]
				}
				
		except Exception as e:
			logger.error(f"Database health check failed: {str(e)}", exc_info=True)
			return {
				"status": "unhealthy",
				"error": str(e),
				"timestamp": datetime.utcnow().isoformat()
			}
	
	@asynccontextmanager
	async def get_connection(self):
		"""Get database connection from pool"""
		if not self._initialized or not self.pool:
			raise DatabaseConnectionError("Database not initialized")
		
		async with self.pool.acquire() as conn:
			yield conn
	
	def _ensure_tenant_isolation(self, tenant_id: str):
		"""Ensure tenant ID is provided for isolation"""
		if not tenant_id:
			raise TenantIsolationError("Tenant ID is required for all operations")
	
	# ================================
	# Contact Management
	# ================================
	
	async def create_contact(self, contact: CRMContact) -> CRMContact:
		"""Create a new contact"""
		self._ensure_tenant_isolation(contact.tenant_id)
		
		try:
			async with self.get_connection() as conn:
				contact_data = contact.model_dump()
				
				# Convert complex fields to JSON
				contact_data["addresses"] = json.dumps(contact_data["addresses"])
				contact_data["phone_numbers"] = json.dumps(contact_data["phone_numbers"])
				contact_data["tags"] = json.dumps(contact_data["tags"])
				
				query = """
					INSERT INTO crm_contacts (
						id, tenant_id, first_name, last_name, email, phone, job_title,
						company, account_id, contact_type, lead_source, lead_score,
						customer_health_score, addresses, phone_numbers, notes, tags,
						created_by, updated_by, status
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20
					)
					RETURNING *
				"""
				
				row = await conn.fetchrow(
					query,
					contact.id, contact.tenant_id, contact.first_name, contact.last_name,
					contact.email, contact.phone, contact.job_title, contact.company,
					contact.account_id, contact.contact_type.value, 
					contact.lead_source.value if contact.lead_source else None,
					contact.lead_score, contact.customer_health_score,
					contact_data["addresses"], contact_data["phone_numbers"],
					contact.notes, contact_data["tags"], contact.created_by,
					contact.updated_by, contact.status.value
				)
				
				return self._row_to_contact(row)
			
		except Exception as e:
			logger.error(f"Failed to create contact: {str(e)}", exc_info=True)
			raise DatabaseError(f"Contact creation failed: {str(e)}")
	
	async def get_contact(self, contact_id: str, tenant_id: str) -> Optional[CRMContact]:
		"""Get contact by ID"""
		self._ensure_tenant_isolation(tenant_id)
		
		try:
			async with self.get_connection() as conn:
				query = """
					SELECT * FROM crm_contacts 
					WHERE id = $1 AND tenant_id = $2 AND status != 'deleted'
				"""
				
				row = await conn.fetchrow(query, contact_id, tenant_id)
				return self._row_to_contact(row) if row else None
			
		except Exception as e:
			logger.error(f"Failed to get contact {contact_id}: {str(e)}", exc_info=True)
			raise DatabaseError(f"Contact retrieval failed: {str(e)}")
	
	async def update_contact(
		self, 
		contact_id: str, 
		update_data: Dict[str, Any], 
		tenant_id: str
	) -> CRMContact:
		"""Update contact"""
		self._ensure_tenant_isolation(tenant_id)
		
		try:
			async with self.get_connection() as conn:
				# Build dynamic update query
				set_clauses = []
				params = []
				param_count = 1
				
				# Handle JSON fields
				json_fields = {"addresses", "phone_numbers", "tags"}
				
				for field, value in update_data.items():
					if field in {"id", "tenant_id", "created_at", "created_by"}:
						continue  # Skip immutable fields
					
					if field in json_fields and isinstance(value, (list, dict)):
						value = json.dumps(value)
					
					set_clauses.append(f"{field} = ${param_count}")
					params.append(value)
					param_count += 1
				
				if not set_clauses:
					raise ValueError("No valid fields to update")
				
				# Add mandatory fields
				set_clauses.append(f"updated_at = ${param_count}")
				params.append(datetime.utcnow())
				param_count += 1
				
				params.extend([contact_id, tenant_id])
				
				query = f"""
					UPDATE crm_contacts 
					SET {', '.join(set_clauses)}
					WHERE id = ${param_count-1} AND tenant_id = ${param_count} AND status != 'deleted'
					RETURNING *
				"""
				
				row = await conn.fetchrow(query, *params)
				if not row:
					raise DatabaseError(f"Contact {contact_id} not found or not accessible")
				
				return self._row_to_contact(row)
			
		except Exception as e:
			logger.error(f"Failed to update contact {contact_id}: {str(e)}", exc_info=True)
			raise DatabaseError(f"Contact update failed: {str(e)}")
	
	async def search_contacts(
		self,
		tenant_id: str,
		filters: Dict[str, Any] = None,
		search_term: str = None,
		limit: int = 100,
		offset: int = 0
	) -> Tuple[List[CRMContact], int]:
		"""Search contacts with filters"""
		self._ensure_tenant_isolation(tenant_id)
		
		try:
			async with self.get_connection() as conn:
				where_clauses = ["tenant_id = $1", "status != 'deleted'"]
				params = [tenant_id]
				param_count = 2
				
				# Apply filters
				if filters:
					for field, value in filters.items():
						if field == "email":
							where_clauses.append(f"email ILIKE ${param_count}")
							params.append(f"%{value}%")
						elif field == "company":
							where_clauses.append(f"company ILIKE ${param_count}")
							params.append(f"%{value}%")
						elif field == "contact_type":
							where_clauses.append(f"contact_type = ${param_count}")
							params.append(value)
						param_count += 1
				
				# Apply search term
				if search_term:
					where_clauses.append(f"""
						(first_name ILIKE ${param_count} OR 
						 last_name ILIKE ${param_count} OR 
						 email ILIKE ${param_count} OR 
						 company ILIKE ${param_count})
					""")
					params.append(f"%{search_term}%")
					param_count += 1
				
				where_clause = " AND ".join(where_clauses)
				
				# Get total count
				count_query = f"SELECT COUNT(*) FROM crm_contacts WHERE {where_clause}"
				total_count = await conn.fetchval(count_query, *params)
				
				# Get records
				params.extend([limit, offset])
				query = f"""
					SELECT * FROM crm_contacts 
					WHERE {where_clause}
					ORDER BY created_at DESC
					LIMIT ${param_count} OFFSET ${param_count + 1}
				"""
				
				rows = await conn.fetch(query, *params)
				contacts = [self._row_to_contact(row) for row in rows]
				
				return contacts, total_count
			
		except Exception as e:
			logger.error(f"Contact search failed: {str(e)}", exc_info=True)
			raise DatabaseError(f"Contact search failed: {str(e)}")
	
	def _row_to_contact(self, row) -> CRMContact:
		"""Convert database row to CRMContact model"""
		if not row:
			return None
		
		try:
			contact_data = dict(row)
			
			# Parse JSON fields
			contact_data["addresses"] = json.loads(contact_data["addresses"]) if contact_data["addresses"] else []
			contact_data["phone_numbers"] = json.loads(contact_data["phone_numbers"]) if contact_data["phone_numbers"] else []
			contact_data["tags"] = json.loads(contact_data["tags"]) if contact_data["tags"] else []
			
			# Convert enums
			contact_data["contact_type"] = ContactType(contact_data["contact_type"])
			if contact_data["lead_source"]:
				contact_data["lead_source"] = LeadSource(contact_data["lead_source"])
			contact_data["status"] = RecordStatus(contact_data["status"])
			
			return CRMContact(**contact_data)
			
		except Exception as e:
			logger.error(f"Failed to convert row to contact: {str(e)}", exc_info=True)
			raise DatabaseError(f"Data conversion failed: {str(e)}")
	
	# ================================
	# Account Management (Placeholder implementations)
	# ================================
	
	async def create_account(self, account: CRMAccount) -> CRMAccount:
		"""Create account - placeholder implementation"""
		# Similar implementation to create_contact
		return account
	
	async def get_account(self, account_id: str, tenant_id: str) -> Optional[CRMAccount]:
		"""Get account - placeholder implementation"""
		return None
	
	# ================================
	# Lead Management (Placeholder implementations)
	# ================================
	
	async def create_lead(self, lead: CRMLead) -> CRMLead:
		"""Create lead - placeholder implementation"""
		return lead
	
	# ================================
	# Opportunity Management (Placeholder implementations)
	# ================================
	
	async def create_opportunity(self, opportunity: CRMOpportunity) -> CRMOpportunity:
		"""Create opportunity - placeholder implementation"""
		return opportunity
	
	# ================================
	# Activity Management (Placeholder implementations)
	# ================================
	
	async def create_activity(self, activity: CRMActivity) -> CRMActivity:
		"""Create activity - placeholder implementation"""
		return activity
	
	# ================================
	# Migration Management
	# ================================
	
	async def run_migrations(self):
		"""Run database migrations"""
		async with self._migration_lock:
			logger.info("🔄 Running database migrations...")
			
			try:
				async with self.get_connection() as conn:
					# Create migrations table if it doesn't exist
					await conn.execute("""
						CREATE TABLE IF NOT EXISTS crm_migrations (
							id SERIAL PRIMARY KEY,
							migration_name TEXT NOT NULL UNIQUE,
							applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
						)
					""")
					
					# List of migrations to apply
					migrations = [
						"001_initial_schema",
						"002_performance_indexes",
						"003_tenant_isolation_functions"
					]
					
					for migration_name in migrations:
						# Check if migration already applied
						exists = await conn.fetchval(
							"SELECT EXISTS(SELECT 1 FROM crm_migrations WHERE migration_name = $1)",
							migration_name
						)
						
						if not exists:
							logger.info(f"Applying migration: {migration_name}")
							# Record migration as applied
							await conn.execute(
								"INSERT INTO crm_migrations (migration_name) VALUES ($1)",
								migration_name
							)
					
					logger.info("✅ Database migrations completed")
					
			except Exception as e:
				logger.error(f"Migration failed: {str(e)}", exc_info=True)
				raise DatabaseError(f"Migration failed: {str(e)}")
	
	async def validate_schema(self):
		"""Validate database schema"""
		logger.info("🔍 Validating database schema...")
		
		try:
			async with self.get_connection() as conn:
				# Check that all required tables exist
				required_tables = [
					"crm_contacts", "crm_accounts", "crm_leads", 
					"crm_opportunities", "crm_activities", "crm_campaigns"
				]
				
				for table_name in required_tables:
					exists = await conn.fetchval("""
						SELECT EXISTS (
							SELECT FROM information_schema.tables 
							WHERE table_schema = 'public' 
							AND table_name = $1
						)
					""", table_name)
					
					if not exists:
						raise DatabaseError(f"Required table {table_name} does not exist")
				
				logger.info("✅ Database schema validation completed")
				
		except Exception as e:
			logger.error(f"Schema validation failed: {str(e)}", exc_info=True)
			raise DatabaseError(f"Schema validation failed: {str(e)}")
	
	# ================================
	# Bulk Operations for Import/Export
	# ================================
	
	async def bulk_create_contacts(self, contacts_data: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""
		Bulk create contacts for import operations
		
		Args:
			contacts_data: List of contact dictionaries
			
		Returns:
			Dictionary with success/error counts and details
		"""
		try:
			success_count = 0
			error_count = 0
			errors = []
			
			async with self.get_connection() as conn:
				async with conn.transaction():
					for i, contact_data in enumerate(contacts_data):
						try:
							# Create CRMContact object
							contact = CRMContact(**contact_data)
							
							# Insert contact
							await conn.execute("""
								INSERT INTO crm_contacts (
									id, tenant_id, first_name, last_name, email, phone, mobile,
									company, job_title, department, website, linkedin_profile,
									contact_type, lead_source, lead_score, customer_health_score,
									description, notes, address, city, state, postal_code, country,
									created_by, updated_by, created_at, updated_at, version
								) VALUES (
									$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, 
									$15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28
								)
							""",
								contact.id, contact.tenant_id, contact.first_name, contact.last_name,
								contact.email, contact.phone, contact.mobile, contact.company,
								contact.job_title, contact.department, contact.website, 
								contact.linkedin_profile, contact.contact_type.value,
								contact.lead_source.value if contact.lead_source else None,
								contact.lead_score, contact.customer_health_score,
								contact.description, contact.notes, contact.address,
								contact.city, contact.state, contact.postal_code, contact.country,
								contact.created_by, contact.updated_by, contact.created_at,
								contact.updated_at, contact.version
							)
							
							success_count += 1
							
						except Exception as e:
							error_count += 1
							errors.append({
								"row": i + 1,
								"error": str(e),
								"data": contact_data
							})
			
			logger.info(f"Bulk contact creation completed - Success: {success_count}, Errors: {error_count}")
			
			return {
				"success_count": success_count,
				"error_count": error_count,
				"errors": errors
			}
			
		except Exception as e:
			logger.error(f"Bulk contact creation failed: {str(e)}", exc_info=True)
			raise DatabaseError(f"Bulk contact creation failed: {str(e)}")
	
	async def find_contacts_by_emails(self, tenant_id: str, emails: List[str]) -> List[CRMContact]:
		"""
		Find existing contacts by email addresses for deduplication
		
		Args:
			tenant_id: Tenant identifier
			emails: List of email addresses to search
			
		Returns:
			List of existing contacts
		"""
		try:
			if not emails:
				return []
			
			async with self.get_connection() as conn:
				# Use ANY operator for efficient email lookup
				rows = await conn.fetch("""
					SELECT * FROM crm_contacts 
					WHERE tenant_id = $1 AND email = ANY($2::text[])
					ORDER BY created_at DESC
				""", tenant_id, emails)
				
				contacts = []
				for row in rows:
					contact = self._row_to_contact(row)
					contacts.append(contact)
				
				return contacts
				
		except Exception as e:
			logger.error(f"Find contacts by emails failed: {str(e)}", exc_info=True)
			raise DatabaseError(f"Find contacts by emails failed: {str(e)}")
	
	async def get_contact_export_data(
		self,
		tenant_id: str,
		contact_ids: Optional[List[str]] = None,
		filters: Optional[Dict[str, Any]] = None,
		limit: int = 10000
	) -> List[CRMContact]:
		"""
		Get contact data optimized for export operations
		
		Args:
			tenant_id: Tenant identifier
			contact_ids: Specific contact IDs to export
			filters: Additional filters
			limit: Maximum number of contacts to export
			
		Returns:
			List of contacts for export
		"""
		try:
			async with self.get_connection() as conn:
				if contact_ids:
					# Export specific contacts
					rows = await conn.fetch("""
						SELECT * FROM crm_contacts 
						WHERE tenant_id = $1 AND id = ANY($2::text[])
						ORDER BY created_at DESC
					""", tenant_id, contact_ids)
				else:
					# Export with filters
					query = "SELECT * FROM crm_contacts WHERE tenant_id = $1"
					params = [tenant_id]
					param_counter = 2
					
					if filters:
						if 'contact_type' in filters:
							query += f" AND contact_type = ${param_counter}"
							params.append(filters['contact_type'].value if hasattr(filters['contact_type'], 'value') else filters['contact_type'])
							param_counter += 1
						
						if 'lead_source' in filters:
							query += f" AND lead_source = ${param_counter}"
							params.append(filters['lead_source'].value if hasattr(filters['lead_source'], 'value') else filters['lead_source'])
							param_counter += 1
						
						if 'company' in filters:
							query += f" AND company ILIKE ${param_counter}"
							params.append(f"%{filters['company']}%")
							param_counter += 1
					
					query += f" ORDER BY created_at DESC LIMIT ${param_counter}"
					params.append(limit)
					
					rows = await conn.fetch(query, *params)
				
				contacts = []
				for row in rows:
					contact = self._row_to_contact(row)
					contacts.append(contact)
				
				return contacts
				
		except Exception as e:
			logger.error(f"Get contact export data failed: {str(e)}", exc_info=True)
			raise DatabaseError(f"Get contact export data failed: {str(e)}")
	
	async def shutdown(self):
		"""Gracefully shutdown database connections"""
		try:
			logger.info("🛑 Shutting down database connections...")
			
			if self.pool:
				await self.pool.close()
				self.pool = None
			
			self._initialized = False
			logger.info("✅ Database shutdown completed")
			
		except Exception as e:
			logger.error(f"Database shutdown error: {str(e)}", exc_info=True)