"""
APG Customer Relationship Management - Advanced Authentication Migration

Database migration to create advanced authentication and authorization tables
for enterprise-grade security with MFA, RBAC, session management, and audit logging.

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


class AdvancedAuthenticationMigration(BaseMigration):
	"""Migration for advanced authentication and authorization system"""
	
	def _get_migration_id(self) -> str:
		return "026_advanced_authentication"
	
	def _get_version(self) -> str:
		return "026"
	
	def _get_description(self) -> str:
		return "Advanced authentication with MFA, RBAC, session management, and audit logging"
	
	def _get_dependencies(self) -> list[str]:
		return ["025_api_versioning"]

	async def _execute_up_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the up migration"""
		try:
			logger.info("🔐 Creating advanced authentication and authorization tables...")
			
			# Create authentication method enum
			await conn.execute("""
				CREATE TYPE authentication_method AS ENUM (
					'password',
					'mfa_totp',
					'mfa_sms',
					'mfa_email',
					'biometric',
					'oauth2',
					'saml',
					'ldap',
					'api_key',
					'jwt_token'
				);
			""")
			
			# Create user role enum
			await conn.execute("""
				CREATE TYPE user_role AS ENUM (
					'super_admin',
					'tenant_admin',
					'sales_manager',
					'sales_rep',
					'marketing_manager',
					'marketing_user',
					'support_manager',
					'support_agent',
					'analyst',
					'viewer',
					'api_user',
					'integration_user'
				);
			""")
			
			# Create permission scope enum
			await conn.execute("""
				CREATE TYPE permission_scope AS ENUM (
					'global',
					'tenant',
					'department',
					'team',
					'personal'
				);
			""")
			
			# Create resource type enum
			await conn.execute("""
				CREATE TYPE resource_type AS ENUM (
					'contact',
					'account',
					'lead',
					'opportunity',
					'activity',
					'campaign',
					'report',
					'dashboard',
					'integration',
					'system_config',
					'user_management'
				);
			""")
			
			# Create permission action enum
			await conn.execute("""
				CREATE TYPE permission_action AS ENUM (
					'create',
					'read',
					'update',
					'delete',
					'export',
					'import',
					'share',
					'approve',
					'configure',
					'administer'
				);
			""")
			
			# Create session status enum
			await conn.execute("""
				CREATE TYPE session_status AS ENUM (
					'active',
					'expired',
					'revoked',
					'suspicious',
					'locked'
				);
			""")
			
			# Create auth event type enum
			await conn.execute("""
				CREATE TYPE auth_event_type AS ENUM (
					'login_success',
					'login_failed',
					'logout',
					'password_change',
					'mfa_setup',
					'mfa_success',
					'mfa_failed',
					'permission_denied',
					'session_expired',
					'suspicious_activity',
					'account_locked',
					'account_unlocked'
				);
			""")
			
			# Create user accounts table
			await conn.execute("""
				CREATE TABLE crm_user_accounts (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					username VARCHAR(255) NOT NULL,
					email VARCHAR(255) NOT NULL,
					first_name VARCHAR(255) NOT NULL,
					last_name VARCHAR(255) NOT NULL,
					phone_number VARCHAR(50),
					password_hash TEXT NOT NULL,
					salt VARCHAR(255) NOT NULL,
					is_active BOOLEAN DEFAULT true,
					is_verified BOOLEAN DEFAULT false,
					is_locked BOOLEAN DEFAULT false,
					failed_login_attempts INTEGER DEFAULT 0,
					last_login_at TIMESTAMP WITH TIME ZONE,
					last_password_change TIMESTAMP WITH TIME ZONE,
					password_expires_at TIMESTAMP WITH TIME ZONE,
					mfa_enabled BOOLEAN DEFAULT false,
					mfa_secret TEXT,
					backup_codes JSONB DEFAULT '[]',
					roles JSONB DEFAULT '[]',
					permissions JSONB DEFAULT '{}',
					profile_data JSONB DEFAULT '{}',
					security_preferences JSONB DEFAULT '{}',
					login_history JSONB DEFAULT '[]',
					device_tokens JSONB DEFAULT '[]',
					api_keys JSONB DEFAULT '[]',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					metadata JSONB DEFAULT '{}',
					UNIQUE(tenant_id, username),
					UNIQUE(tenant_id, email)
				);
			""")
			
			# Create permissions table
			await conn.execute("""
				CREATE TABLE crm_permissions (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					permission_name VARCHAR(255) NOT NULL,
					resource_type resource_type NOT NULL,
					action permission_action NOT NULL,
					scope permission_scope DEFAULT 'tenant',
					conditions JSONB DEFAULT '{}',
					is_system_permission BOOLEAN DEFAULT false,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					metadata JSONB DEFAULT '{}',
					UNIQUE(tenant_id, permission_name)
				);
			""")
			
			# Create roles table
			await conn.execute("""
				CREATE TABLE crm_user_roles (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					role_name VARCHAR(255) NOT NULL,
					role_type user_role NOT NULL,
					description TEXT,
					permissions JSONB DEFAULT '[]',
					is_system_role BOOLEAN DEFAULT false,
					is_active BOOLEAN DEFAULT true,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					metadata JSONB DEFAULT '{}',
					UNIQUE(tenant_id, role_name)
				);
			""")
			
			# Create role permissions junction table
			await conn.execute("""
				CREATE TABLE crm_role_permissions (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					role_id VARCHAR(36) NOT NULL REFERENCES crm_user_roles(id) ON DELETE CASCADE,
					permission_id VARCHAR(36) NOT NULL REFERENCES crm_permissions(id) ON DELETE CASCADE,
					granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					granted_by VARCHAR(36) NOT NULL,
					conditions JSONB DEFAULT '{}',
					is_active BOOLEAN DEFAULT true,
					expires_at TIMESTAMP WITH TIME ZONE,
					metadata JSONB DEFAULT '{}',
					UNIQUE(tenant_id, role_id, permission_id)
				);
			""")
			
			# Create user sessions table
			await conn.execute("""
				CREATE TABLE crm_user_sessions (
					id VARCHAR(36) PRIMARY KEY,
					session_token VARCHAR(255) NOT NULL UNIQUE,
					user_id VARCHAR(36) NOT NULL REFERENCES crm_user_accounts(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					device_info JSONB DEFAULT '{}',
					ip_address INET NOT NULL,
					user_agent TEXT,
					location_data JSONB DEFAULT '{}',
					status session_status DEFAULT 'active',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
					last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					activity_count INTEGER DEFAULT 0,
					security_score DECIMAL(5,2) DEFAULT 100.00,
					risk_factors JSONB DEFAULT '[]',
					metadata JSONB DEFAULT '{}'
				);
			""")
			
			# Create authentication events table
			await conn.execute("""
				CREATE TABLE crm_auth_events (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					user_id VARCHAR(36),
					session_id VARCHAR(36),
					event_type auth_event_type NOT NULL,
					authentication_method authentication_method NOT NULL,
					ip_address INET NOT NULL,
					user_agent TEXT,
					device_fingerprint VARCHAR(255),
					location_data JSONB DEFAULT '{}',
					success BOOLEAN NOT NULL,
					error_code VARCHAR(50),
					error_message TEXT,
					risk_score DECIMAL(5,2) DEFAULT 0.00,
					additional_data JSONB DEFAULT '{}',
					timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					metadata JSONB DEFAULT '{}'
				);
			""")
			
			# Create MFA challenges table
			await conn.execute("""
				CREATE TABLE crm_mfa_challenges (
					id VARCHAR(36) PRIMARY KEY,
					user_id VARCHAR(36) NOT NULL REFERENCES crm_user_accounts(id) ON DELETE CASCADE,
					tenant_id VARCHAR(100) NOT NULL,
					challenge_type authentication_method NOT NULL,
					challenge_data JSONB DEFAULT '{}',
					is_verified BOOLEAN DEFAULT false,
					attempts INTEGER DEFAULT 0,
					max_attempts INTEGER DEFAULT 3,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
					verified_at TIMESTAMP WITH TIME ZONE,
					metadata JSONB DEFAULT '{}'
				);
			""")
			
			# Create user role assignments table
			await conn.execute("""
				CREATE TABLE crm_user_role_assignments (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					user_id VARCHAR(36) NOT NULL REFERENCES crm_user_accounts(id) ON DELETE CASCADE,
					role_id VARCHAR(36) NOT NULL REFERENCES crm_user_roles(id) ON DELETE CASCADE,
					assigned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					assigned_by VARCHAR(36) NOT NULL,
					expires_at TIMESTAMP WITH TIME ZONE,
					is_active BOOLEAN DEFAULT true,
					assignment_reason TEXT,
					metadata JSONB DEFAULT '{}',
					UNIQUE(tenant_id, user_id, role_id)
				);
			""")
			
			# Create API keys table
			await conn.execute("""
				CREATE TABLE crm_api_keys (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					user_id VARCHAR(36) NOT NULL REFERENCES crm_user_accounts(id) ON DELETE CASCADE,
					key_name VARCHAR(255) NOT NULL,
					key_hash VARCHAR(255) NOT NULL UNIQUE,
					key_prefix VARCHAR(20) NOT NULL,
					permissions JSONB DEFAULT '[]',
					rate_limit_config JSONB DEFAULT '{}',
					ip_restrictions JSONB DEFAULT '[]',
					is_active BOOLEAN DEFAULT true,
					last_used_at TIMESTAMP WITH TIME ZONE,
					usage_count BIGINT DEFAULT 0,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					expires_at TIMESTAMP WITH TIME ZONE,
					created_by VARCHAR(36) NOT NULL,
					metadata JSONB DEFAULT '{}'
				);
			""")
			
			# Create password reset tokens table
			await conn.execute("""
				CREATE TABLE crm_password_reset_tokens (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					user_id VARCHAR(36) NOT NULL REFERENCES crm_user_accounts(id) ON DELETE CASCADE,
					token_hash VARCHAR(255) NOT NULL UNIQUE,
					is_used BOOLEAN DEFAULT false,
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
					used_at TIMESTAMP WITH TIME ZONE,
					ip_address INET,
					user_agent TEXT,
					metadata JSONB DEFAULT '{}'
				);
			""")
			
			# Create security policies table
			await conn.execute("""
				CREATE TABLE crm_security_policies (
					id VARCHAR(36) PRIMARY KEY,
					tenant_id VARCHAR(100) NOT NULL,
					policy_name VARCHAR(255) NOT NULL,
					policy_type VARCHAR(100) NOT NULL,
					policy_config JSONB NOT NULL DEFAULT '{}',
					is_active BOOLEAN DEFAULT true,
					applies_to JSONB DEFAULT '[]',
					enforcement_level VARCHAR(50) DEFAULT 'strict',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
					created_by VARCHAR(36) NOT NULL,
					metadata JSONB DEFAULT '{}',
					UNIQUE(tenant_id, policy_name)
				);
			""")
			
			# Create comprehensive indexes for performance
			await conn.execute("CREATE INDEX idx_user_accounts_tenant ON crm_user_accounts(tenant_id);")
			await conn.execute("CREATE INDEX idx_user_accounts_username ON crm_user_accounts(tenant_id, username);")
			await conn.execute("CREATE INDEX idx_user_accounts_email ON crm_user_accounts(tenant_id, email);")
			await conn.execute("CREATE INDEX idx_user_accounts_active ON crm_user_accounts(is_active);")
			await conn.execute("CREATE INDEX idx_user_accounts_locked ON crm_user_accounts(is_locked);")
			await conn.execute("CREATE INDEX idx_user_accounts_mfa ON crm_user_accounts(mfa_enabled);")
			await conn.execute("CREATE INDEX idx_user_accounts_last_login ON crm_user_accounts(last_login_at);")
			await conn.execute("CREATE INDEX idx_user_accounts_roles ON crm_user_accounts USING GIN(roles);")
			
			await conn.execute("CREATE INDEX idx_permissions_tenant ON crm_permissions(tenant_id);")
			await conn.execute("CREATE INDEX idx_permissions_resource ON crm_permissions(resource_type);")
			await conn.execute("CREATE INDEX idx_permissions_action ON crm_permissions(action);")
			await conn.execute("CREATE INDEX idx_permissions_scope ON crm_permissions(scope);")
			await conn.execute("CREATE INDEX idx_permissions_system ON crm_permissions(is_system_permission);")
			
			await conn.execute("CREATE INDEX idx_user_roles_tenant ON crm_user_roles(tenant_id);")
			await conn.execute("CREATE INDEX idx_user_roles_type ON crm_user_roles(role_type);")
			await conn.execute("CREATE INDEX idx_user_roles_active ON crm_user_roles(is_active);")
			await conn.execute("CREATE INDEX idx_user_roles_system ON crm_user_roles(is_system_role);")
			
			await conn.execute("CREATE INDEX idx_role_permissions_tenant ON crm_role_permissions(tenant_id);")
			await conn.execute("CREATE INDEX idx_role_permissions_role ON crm_role_permissions(role_id);")
			await conn.execute("CREATE INDEX idx_role_permissions_permission ON crm_role_permissions(permission_id);")
			await conn.execute("CREATE INDEX idx_role_permissions_active ON crm_role_permissions(is_active);")
			await conn.execute("CREATE INDEX idx_role_permissions_expires ON crm_role_permissions(expires_at);")
			
			await conn.execute("CREATE INDEX idx_user_sessions_token ON crm_user_sessions(session_token);")
			await conn.execute("CREATE INDEX idx_user_sessions_user ON crm_user_sessions(user_id);")
			await conn.execute("CREATE INDEX idx_user_sessions_tenant ON crm_user_sessions(tenant_id);")
			await conn.execute("CREATE INDEX idx_user_sessions_status ON crm_user_sessions(status);")
			await conn.execute("CREATE INDEX idx_user_sessions_expires ON crm_user_sessions(expires_at);")
			await conn.execute("CREATE INDEX idx_user_sessions_activity ON crm_user_sessions(last_activity_at);")
			await conn.execute("CREATE INDEX idx_user_sessions_ip ON crm_user_sessions(ip_address);")
			await conn.execute("CREATE INDEX idx_user_sessions_score ON crm_user_sessions(security_score);")
			
			await conn.execute("CREATE INDEX idx_auth_events_tenant ON crm_auth_events(tenant_id);")
			await conn.execute("CREATE INDEX idx_auth_events_user ON crm_auth_events(user_id);")
			await conn.execute("CREATE INDEX idx_auth_events_session ON crm_auth_events(session_id);")
			await conn.execute("CREATE INDEX idx_auth_events_type ON crm_auth_events(event_type);")
			await conn.execute("CREATE INDEX idx_auth_events_method ON crm_auth_events(authentication_method);")
			await conn.execute("CREATE INDEX idx_auth_events_success ON crm_auth_events(success);")
			await conn.execute("CREATE INDEX idx_auth_events_timestamp ON crm_auth_events(timestamp);")
			await conn.execute("CREATE INDEX idx_auth_events_ip ON crm_auth_events(ip_address);")
			await conn.execute("CREATE INDEX idx_auth_events_risk ON crm_auth_events(risk_score);")
			
			await conn.execute("CREATE INDEX idx_mfa_challenges_user ON crm_mfa_challenges(user_id);")
			await conn.execute("CREATE INDEX idx_mfa_challenges_tenant ON crm_mfa_challenges(tenant_id);")
			await conn.execute("CREATE INDEX idx_mfa_challenges_type ON crm_mfa_challenges(challenge_type);")
			await conn.execute("CREATE INDEX idx_mfa_challenges_verified ON crm_mfa_challenges(is_verified);")
			await conn.execute("CREATE INDEX idx_mfa_challenges_expires ON crm_mfa_challenges(expires_at);")
			
			await conn.execute("CREATE INDEX idx_role_assignments_tenant ON crm_user_role_assignments(tenant_id);")
			await conn.execute("CREATE INDEX idx_role_assignments_user ON crm_user_role_assignments(user_id);")
			await conn.execute("CREATE INDEX idx_role_assignments_role ON crm_user_role_assignments(role_id);")
			await conn.execute("CREATE INDEX idx_role_assignments_active ON crm_user_role_assignments(is_active);")
			await conn.execute("CREATE INDEX idx_role_assignments_expires ON crm_user_role_assignments(expires_at);")
			
			await conn.execute("CREATE INDEX idx_api_keys_tenant ON crm_api_keys(tenant_id);")
			await conn.execute("CREATE INDEX idx_api_keys_user ON crm_api_keys(user_id);")
			await conn.execute("CREATE INDEX idx_api_keys_hash ON crm_api_keys(key_hash);")
			await conn.execute("CREATE INDEX idx_api_keys_prefix ON crm_api_keys(key_prefix);")
			await conn.execute("CREATE INDEX idx_api_keys_active ON crm_api_keys(is_active);")
			await conn.execute("CREATE INDEX idx_api_keys_expires ON crm_api_keys(expires_at);")
			await conn.execute("CREATE INDEX idx_api_keys_last_used ON crm_api_keys(last_used_at);")
			
			await conn.execute("CREATE INDEX idx_password_reset_tenant ON crm_password_reset_tokens(tenant_id);")
			await conn.execute("CREATE INDEX idx_password_reset_user ON crm_password_reset_tokens(user_id);")
			await conn.execute("CREATE INDEX idx_password_reset_token ON crm_password_reset_tokens(token_hash);")
			await conn.execute("CREATE INDEX idx_password_reset_used ON crm_password_reset_tokens(is_used);")
			await conn.execute("CREATE INDEX idx_password_reset_expires ON crm_password_reset_tokens(expires_at);")
			
			await conn.execute("CREATE INDEX idx_security_policies_tenant ON crm_security_policies(tenant_id);")
			await conn.execute("CREATE INDEX idx_security_policies_type ON crm_security_policies(policy_type);")
			await conn.execute("CREATE INDEX idx_security_policies_active ON crm_security_policies(is_active);")
			
			# Insert default system permissions
			await conn.execute("""
				INSERT INTO crm_permissions (
					id, tenant_id, permission_name, resource_type, action, scope,
					is_system_permission, created_by
				) VALUES 
				('perm_contact_admin', 'system', 'contact_administration', 'contact', 'administer', 'tenant', true, 'system'),
				('perm_account_admin', 'system', 'account_administration', 'account', 'administer', 'tenant', true, 'system'),
				('perm_lead_admin', 'system', 'lead_administration', 'lead', 'administer', 'tenant', true, 'system'),
				('perm_opportunity_admin', 'system', 'opportunity_administration', 'opportunity', 'administer', 'tenant', true, 'system'),
				('perm_system_admin', 'system', 'system_administration', 'system_config', 'administer', 'global', true, 'system'),
				('perm_user_mgmt', 'system', 'user_management', 'user_management', 'administer', 'tenant', true, 'system'),
				('perm_report_admin', 'system', 'report_administration', 'report', 'administer', 'tenant', true, 'system'),
				('perm_dashboard_admin', 'system', 'dashboard_administration', 'dashboard', 'administer', 'tenant', true, 'system'),
				('perm_integration_admin', 'system', 'integration_administration', 'integration', 'administer', 'tenant', true, 'system'),
				('perm_contact_read', 'system', 'contact_read', 'contact', 'read', 'tenant', true, 'system'),
				('perm_contact_create', 'system', 'contact_create', 'contact', 'create', 'tenant', true, 'system'),
				('perm_contact_update', 'system', 'contact_update', 'contact', 'update', 'tenant', true, 'system'),
				('perm_contact_delete', 'system', 'contact_delete', 'contact', 'delete', 'tenant', true, 'system'),
				('perm_account_read', 'system', 'account_read', 'account', 'read', 'tenant', true, 'system'),
				('perm_account_create', 'system', 'account_create', 'account', 'create', 'tenant', true, 'system'),
				('perm_account_update', 'system', 'account_update', 'account', 'update', 'tenant', true, 'system'),
				('perm_lead_read', 'system', 'lead_read', 'lead', 'read', 'tenant', true, 'system'),
				('perm_lead_create', 'system', 'lead_create', 'lead', 'create', 'tenant', true, 'system'),
				('perm_opportunity_read', 'system', 'opportunity_read', 'opportunity', 'read', 'tenant', true, 'system'),
				('perm_opportunity_create', 'system', 'opportunity_create', 'opportunity', 'create', 'tenant', true, 'system')
			""")
			
			# Insert default system roles
			await conn.execute("""
				INSERT INTO crm_user_roles (
					id, tenant_id, role_name, role_type, description,
					is_system_role, created_by
				) VALUES 
				('role_super_admin', 'system', 'Super Administrator', 'super_admin', 'Full system access across all tenants', true, 'system'),
				('role_tenant_admin', 'system', 'Tenant Administrator', 'tenant_admin', 'Full access within tenant', true, 'system'),
				('role_sales_manager', 'system', 'Sales Manager', 'sales_manager', 'Sales team management and reporting', true, 'system'),
				('role_sales_rep', 'system', 'Sales Representative', 'sales_rep', 'Sales activities and customer management', true, 'system'),
				('role_marketing_manager', 'system', 'Marketing Manager', 'marketing_manager', 'Marketing campaign and lead management', true, 'system'),
				('role_marketing_user', 'system', 'Marketing User', 'marketing_user', 'Marketing activities and content creation', true, 'system'),
				('role_support_manager', 'system', 'Support Manager', 'support_manager', 'Customer support team management', true, 'system'),
				('role_support_agent', 'system', 'Support Agent', 'support_agent', 'Customer support and case management', true, 'system'),
				('role_analyst', 'system', 'Data Analyst', 'analyst', 'Data analysis and reporting', true, 'system'),
				('role_viewer', 'system', 'Viewer', 'viewer', 'Read-only access to permitted resources', true, 'system'),
				('role_api_user', 'system', 'API User', 'api_user', 'Programmatic access via API', true, 'system'),
				('role_integration_user', 'system', 'Integration User', 'integration_user', 'Third-party integration access', true, 'system')
			""")
			
			# Create default role-permission assignments for system roles
			role_permission_mappings = [
				# Super Admin - All permissions
				('role_super_admin', ['perm_contact_admin', 'perm_account_admin', 'perm_lead_admin', 'perm_opportunity_admin', 'perm_system_admin', 'perm_user_mgmt', 'perm_report_admin', 'perm_dashboard_admin', 'perm_integration_admin']),
				# Tenant Admin - All tenant permissions
				('role_tenant_admin', ['perm_contact_admin', 'perm_account_admin', 'perm_lead_admin', 'perm_opportunity_admin', 'perm_user_mgmt', 'perm_report_admin', 'perm_dashboard_admin', 'perm_integration_admin']),
				# Sales Manager - Sales-related permissions
				('role_sales_manager', ['perm_contact_admin', 'perm_account_admin', 'perm_lead_admin', 'perm_opportunity_admin', 'perm_report_admin']),
				# Sales Rep - Basic sales permissions
				('role_sales_rep', ['perm_contact_read', 'perm_contact_create', 'perm_contact_update', 'perm_account_read', 'perm_account_create', 'perm_account_update', 'perm_lead_read', 'perm_lead_create', 'perm_opportunity_read', 'perm_opportunity_create']),
				# Marketing Manager - Marketing permissions
				('role_marketing_manager', ['perm_contact_admin', 'perm_lead_admin', 'perm_report_admin']),
				# Marketing User - Basic marketing permissions
				('role_marketing_user', ['perm_contact_read', 'perm_contact_create', 'perm_lead_read', 'perm_lead_create']),
				# Viewer - Read-only permissions
				('role_viewer', ['perm_contact_read', 'perm_account_read', 'perm_lead_read', 'perm_opportunity_read'])
			]
			
			for role_id, permission_ids in role_permission_mappings:
				for perm_id in permission_ids:
					await conn.execute("""
						INSERT INTO crm_role_permissions (
							id, tenant_id, role_id, permission_id, granted_by
						) VALUES ($1, 'system', $2, $3, 'system')
					""", f"rp_{role_id}_{perm_id}", role_id, perm_id)
			
			# Insert default security policies
			await conn.execute("""
				INSERT INTO crm_security_policies (
					id, tenant_id, policy_name, policy_type, policy_config, created_by
				) VALUES 
				('policy_password_strength', 'system', 'Password Strength Policy', 'password_policy', 
				 '{"min_length": 8, "require_uppercase": true, "require_lowercase": true, "require_digits": true, "require_special": true, "max_age_days": 90}', 'system'),
				('policy_session_timeout', 'system', 'Session Timeout Policy', 'session_policy',
				 '{"idle_timeout_minutes": 30, "absolute_timeout_hours": 24, "max_concurrent_sessions": 5}', 'system'),
				('policy_mfa_requirement', 'system', 'MFA Requirement Policy', 'mfa_policy',
				 '{"required_for_admin": true, "required_for_api": false, "risk_based_mfa": true, "risk_threshold": 50}', 'system'),
				('policy_rate_limiting', 'system', 'Rate Limiting Policy', 'rate_limit_policy',
				 '{"login_attempts_per_minute": 5, "api_requests_per_minute": 1000, "lockout_duration_minutes": 15}', 'system')
			""")
			
			# Create sample system admin user (with secure default password that should be changed)
			await conn.execute("""
				INSERT INTO crm_user_accounts (
					id, tenant_id, username, email, first_name, last_name,
					password_hash, salt, roles, is_active, is_verified, created_by
				) VALUES (
					'user_system_admin',
					'system',
					'admin',
					'admin@datacraft.co.ke',
					'System',
					'Administrator',
					'$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LowrLdJvNQgUvUOHG',
					'$2b$12$LQv3c1yqBWVHxkd0LHAkCO',
					'["super_admin"]',
					true,
					true,
					'system'
				)
			""")
			
			logger.info("✅ Advanced authentication and authorization tables created successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to create authentication tables: {str(e)}")
			raise
	
	async def _execute_down_migration(self, conn, config: Dict[str, Any]) -> None:
		"""Execute the down migration"""
		try:
			logger.info("🔄 Dropping advanced authentication and authorization tables...")
			
			# Drop tables in reverse order of dependencies
			await conn.execute("DROP TABLE IF EXISTS crm_security_policies CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_password_reset_tokens CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_api_keys CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_user_role_assignments CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_mfa_challenges CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_auth_events CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_user_sessions CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_role_permissions CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_user_roles CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_permissions CASCADE;")
			await conn.execute("DROP TABLE IF EXISTS crm_user_accounts CASCADE;")
			
			# Drop enums
			await conn.execute("DROP TYPE IF EXISTS auth_event_type CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS session_status CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS permission_action CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS resource_type CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS permission_scope CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS user_role CASCADE;")
			await conn.execute("DROP TYPE IF EXISTS authentication_method CASCADE;")
			
			logger.info("✅ Advanced authentication and authorization tables dropped successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to drop authentication tables: {str(e)}")
			raise