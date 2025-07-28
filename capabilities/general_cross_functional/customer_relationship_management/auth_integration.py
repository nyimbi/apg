"""
APG Customer Relationship Management - Authentication Integration

Revolutionary authentication and authorization integration providing seamless
APG auth_rbac integration with advanced role-based access control, 
tenant isolation, and comprehensive security enforcement.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

# APG Core imports (these would be actual APG framework imports)
from apg.core.auth import APGAuthProvider, UserContext, Permission, Role
from apg.core.rbac import RBACManager, AccessPolicy


logger = logging.getLogger(__name__)


class CRMPermission(str, Enum):
	"""CRM-specific permissions"""
	# Contact permissions
	CONTACT_CREATE = "crm.contact.create"
	CONTACT_READ = "crm.contact.read"
	CONTACT_UPDATE = "crm.contact.update"
	CONTACT_DELETE = "crm.contact.delete"
	CONTACT_MERGE = "crm.contact.merge"
	CONTACT_EXPORT = "crm.contact.export"
	CONTACT_IMPORT = "crm.contact.import"
	
	# Account permissions
	ACCOUNT_CREATE = "crm.account.create"
	ACCOUNT_READ = "crm.account.read"
	ACCOUNT_UPDATE = "crm.account.update"
	ACCOUNT_DELETE = "crm.account.delete"
	ACCOUNT_MANAGE_HIERARCHY = "crm.account.manage_hierarchy"
	
	# Lead permissions
	LEAD_CREATE = "crm.lead.create"
	LEAD_READ = "crm.lead.read"
	LEAD_UPDATE = "crm.lead.update"
	LEAD_DELETE = "crm.lead.delete"
	LEAD_CONVERT = "crm.lead.convert"
	LEAD_ASSIGN = "crm.lead.assign"
	LEAD_BULK_OPERATIONS = "crm.lead.bulk_operations"
	
	# Opportunity permissions
	OPPORTUNITY_CREATE = "crm.opportunity.create"
	OPPORTUNITY_READ = "crm.opportunity.read"
	OPPORTUNITY_UPDATE = "crm.opportunity.update"
	OPPORTUNITY_DELETE = "crm.opportunity.delete"
	OPPORTUNITY_MANAGE_STAGES = "crm.opportunity.manage_stages"
	OPPORTUNITY_FORECAST = "crm.opportunity.forecast"
	
	# Activity permissions
	ACTIVITY_CREATE = "crm.activity.create"
	ACTIVITY_READ = "crm.activity.read"
	ACTIVITY_UPDATE = "crm.activity.update"
	ACTIVITY_DELETE = "crm.activity.delete"
	ACTIVITY_ASSIGN = "crm.activity.assign"
	
	# Campaign permissions
	CAMPAIGN_CREATE = "crm.campaign.create"
	CAMPAIGN_READ = "crm.campaign.read"
	CAMPAIGN_UPDATE = "crm.campaign.update"
	CAMPAIGN_DELETE = "crm.campaign.delete"
	CAMPAIGN_EXECUTE = "crm.campaign.execute"
	
	# Analytics permissions
	ANALYTICS_VIEW_REPORTS = "crm.analytics.view_reports"
	ANALYTICS_CREATE_REPORTS = "crm.analytics.create_reports"
	ANALYTICS_EXPORT_DATA = "crm.analytics.export_data"
	ANALYTICS_VIEW_DASHBOARDS = "crm.analytics.view_dashboards"
	
	# Administrative permissions
	ADMIN_MANAGE_USERS = "crm.admin.manage_users"
	ADMIN_MANAGE_ROLES = "crm.admin.manage_roles"
	ADMIN_MANAGE_SETTINGS = "crm.admin.manage_settings"
	ADMIN_VIEW_AUDIT_LOGS = "crm.admin.view_audit_logs"
	ADMIN_MANAGE_INTEGRATIONS = "crm.admin.manage_integrations"


class CRMRole(str, Enum):
	"""CRM-specific roles"""
	SALES_REP = "crm.sales_rep"
	SALES_MANAGER = "crm.sales_manager"
	MARKETING_USER = "crm.marketing_user"
	MARKETING_MANAGER = "crm.marketing_manager"
	CUSTOMER_SERVICE_REP = "crm.service_rep"
	CUSTOMER_SERVICE_MANAGER = "crm.service_manager"
	CRM_ADMIN = "crm.admin"
	CRM_READONLY = "crm.readonly"
	DATA_ANALYST = "crm.analyst"


@dataclass
class CRMUserContext:
	"""CRM-specific user context"""
	user_id: str
	username: str
	email: str
	tenant_id: str
	roles: List[CRMRole]
	permissions: Set[CRMPermission]
	territories: List[str]
	department_id: Optional[str] = None
	manager_id: Optional[str] = None
	is_active: bool = True
	session_id: Optional[str] = None
	login_time: Optional[datetime] = None
	last_activity: Optional[datetime] = None


class CRMAuthProvider:
	"""
	CRM authentication provider integrating with APG auth_rbac system
	"""
	
	def __init__(self, apg_auth: Optional[APGAuthProvider] = None):
		"""
		Initialize CRM auth provider
		
		Args:
			apg_auth: APG authentication provider instance
		"""
		self.apg_auth = apg_auth or APGAuthProvider()
		self.rbac_manager = RBACManager()
		
		# JWT configuration
		self.jwt_secret = os.getenv("CRM_JWT_SECRET", "crm_jwt_secret_key_change_in_production")
		self.jwt_algorithm = "HS256"
		self.token_expiry = timedelta(hours=8)
		
		# Session management
		self.active_sessions: Dict[str, CRMUserContext] = {}
		self.session_timeout = timedelta(hours=24)
		
		# Role-permission mappings
		self.role_permissions = self._initialize_role_permissions()
		
		self._initialized = False
		
		logger.info("🔐 CRM Auth Provider initialized")
	
	async def initialize(self):
		"""Initialize auth provider"""
		try:
			logger.info("🔧 Initializing CRM auth provider...")
			
			# Initialize APG auth integration
			await self.apg_auth.initialize()
			
			# Setup CRM-specific roles and permissions
			await self._setup_crm_roles()
			
			# Initialize RBAC policies
			await self._setup_rbac_policies()
			
			self._initialized = True
			logger.info("✅ CRM auth provider initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize CRM auth provider: {str(e)}", exc_info=True)
			raise
	
	def _initialize_role_permissions(self) -> Dict[CRMRole, Set[CRMPermission]]:
		"""Initialize role-permission mappings"""
		return {
			CRMRole.SALES_REP: {
				CRMPermission.CONTACT_CREATE, CRMPermission.CONTACT_READ, 
				CRMPermission.CONTACT_UPDATE, CRMPermission.LEAD_CREATE,
				CRMPermission.LEAD_READ, CRMPermission.LEAD_UPDATE,
				CRMPermission.LEAD_CONVERT, CRMPermission.OPPORTUNITY_CREATE,
				CRMPermission.OPPORTUNITY_READ, CRMPermission.OPPORTUNITY_UPDATE,
				CRMPermission.ACTIVITY_CREATE, CRMPermission.ACTIVITY_READ,
				CRMPermission.ACTIVITY_UPDATE, CRMPermission.ANALYTICS_VIEW_DASHBOARDS
			},
			
			CRMRole.SALES_MANAGER: {
				# All sales rep permissions plus management permissions
				CRMPermission.CONTACT_CREATE, CRMPermission.CONTACT_READ,
				CRMPermission.CONTACT_UPDATE, CRMPermission.CONTACT_DELETE,
				CRMPermission.CONTACT_MERGE, CRMPermission.LEAD_CREATE,
				CRMPermission.LEAD_READ, CRMPermission.LEAD_UPDATE,
				CRMPermission.LEAD_DELETE, CRMPermission.LEAD_CONVERT,
				CRMPermission.LEAD_ASSIGN, CRMPermission.LEAD_BULK_OPERATIONS,
				CRMPermission.OPPORTUNITY_CREATE, CRMPermission.OPPORTUNITY_READ,
				CRMPermission.OPPORTUNITY_UPDATE, CRMPermission.OPPORTUNITY_DELETE,
				CRMPermission.OPPORTUNITY_MANAGE_STAGES, CRMPermission.OPPORTUNITY_FORECAST,
				CRMPermission.ACTIVITY_CREATE, CRMPermission.ACTIVITY_READ,
				CRMPermission.ACTIVITY_UPDATE, CRMPermission.ACTIVITY_DELETE,
				CRMPermission.ACTIVITY_ASSIGN, CRMPermission.ANALYTICS_VIEW_REPORTS,
				CRMPermission.ANALYTICS_VIEW_DASHBOARDS, CRMPermission.ANALYTICS_EXPORT_DATA
			},
			
			CRMRole.MARKETING_USER: {
				CRMPermission.CONTACT_READ, CRMPermission.LEAD_CREATE,
				CRMPermission.LEAD_READ, CRMPermission.LEAD_UPDATE,
				CRMPermission.CAMPAIGN_CREATE, CRMPermission.CAMPAIGN_READ,
				CRMPermission.CAMPAIGN_UPDATE, CRMPermission.CAMPAIGN_EXECUTE,
				CRMPermission.ANALYTICS_VIEW_DASHBOARDS
			},
			
			CRMRole.MARKETING_MANAGER: {
				# All marketing user permissions plus management permissions
				CRMPermission.CONTACT_READ, CRMPermission.CONTACT_EXPORT,
				CRMPermission.CONTACT_IMPORT, CRMPermission.LEAD_CREATE,
				CRMPermission.LEAD_READ, CRMPermission.LEAD_UPDATE,
				CRMPermission.LEAD_DELETE, CRMPermission.LEAD_ASSIGN,
				CRMPermission.LEAD_BULK_OPERATIONS, CRMPermission.CAMPAIGN_CREATE,
				CRMPermission.CAMPAIGN_READ, CRMPermission.CAMPAIGN_UPDATE,
				CRMPermission.CAMPAIGN_DELETE, CRMPermission.CAMPAIGN_EXECUTE,
				CRMPermission.ANALYTICS_VIEW_REPORTS, CRMPermission.ANALYTICS_CREATE_REPORTS,
				CRMPermission.ANALYTICS_VIEW_DASHBOARDS, CRMPermission.ANALYTICS_EXPORT_DATA
			},
			
			CRMRole.CRM_ADMIN: {
				# All permissions
				*list(CRMPermission)
			},
			
			CRMRole.CRM_READONLY: {
				CRMPermission.CONTACT_READ, CRMPermission.ACCOUNT_READ,
				CRMPermission.LEAD_READ, CRMPermission.OPPORTUNITY_READ,
				CRMPermission.ACTIVITY_READ, CRMPermission.CAMPAIGN_READ,
				CRMPermission.ANALYTICS_VIEW_REPORTS, CRMPermission.ANALYTICS_VIEW_DASHBOARDS
			},
			
			CRMRole.DATA_ANALYST: {
				CRMPermission.CONTACT_READ, CRMPermission.ACCOUNT_READ,
				CRMPermission.LEAD_READ, CRMPermission.OPPORTUNITY_READ,
				CRMPermission.ACTIVITY_READ, CRMPermission.CAMPAIGN_READ,
				CRMPermission.ANALYTICS_VIEW_REPORTS, CRMPermission.ANALYTICS_CREATE_REPORTS,
				CRMPermission.ANALYTICS_VIEW_DASHBOARDS, CRMPermission.ANALYTICS_EXPORT_DATA
			}
		}
	
	async def _setup_crm_roles(self):
		"""Setup CRM-specific roles in APG RBAC"""
		for role in CRMRole:
			permissions = self.role_permissions.get(role, set())
			
			await self.rbac_manager.create_role(
				role_name=role.value,
				description=f"CRM {role.value.replace('crm.', '').replace('_', ' ').title()}",
				permissions=[p.value for p in permissions]
			)
	
	async def _setup_rbac_policies(self):
		"""Setup RBAC access policies"""
		# Data access policies
		policies = [
			AccessPolicy(
				name="contact_owner_access",
				description="Users can access contacts they own",
				resource_type="contact",
				conditions={"owner_id": "${user.user_id}"}
			),
			AccessPolicy(
				name="territory_based_access",
				description="Users can access records in their territories",
				resource_type="*",
				conditions={"territory": "${user.territories}"}
			),
			AccessPolicy(
				name="department_access",
				description="Users can access department records",
				resource_type="*",
				conditions={"department_id": "${user.department_id}"}
			),
			AccessPolicy(
				name="tenant_isolation",
				description="Users can only access their tenant's data",
				resource_type="*",
				conditions={"tenant_id": "${user.tenant_id}"}
			)
		]
		
		for policy in policies:
			await self.rbac_manager.create_policy(policy)
	
	async def authenticate_user(
		self, 
		username: str, 
		password: str,
		tenant_id: str
	) -> Optional[CRMUserContext]:
		"""
		Authenticate user credentials
		
		Args:
			username: Username or email
			password: User password
			tenant_id: Tenant identifier
			
		Returns:
			CRM user context if authentication successful
		"""
		try:
			# Authenticate with APG auth system
			apg_user = await self.apg_auth.authenticate(
				username=username,
				password=password,
				tenant_id=tenant_id
			)
			
			if not apg_user:
				return None
			
			# Create CRM user context
			crm_user = await self._create_crm_user_context(apg_user)
			
			# Create session
			session_id = self._generate_session_id()
			crm_user.session_id = session_id
			crm_user.login_time = datetime.utcnow()
			crm_user.last_activity = datetime.utcnow()
			
			# Store session
			self.active_sessions[session_id] = crm_user
			
			logger.info(f"🔑 User authenticated: {username} (tenant: {tenant_id})")
			return crm_user
			
		except Exception as e:
			logger.error(f"Authentication failed for {username}: {str(e)}")
			return None
	
	async def validate_token(self, token: str) -> Optional[CRMUserContext]:
		"""
		Validate JWT token and return user context
		
		Args:
			token: JWT token
			
		Returns:
			CRM user context if token is valid
		"""
		try:
			# Decode JWT token
			payload = jwt.decode(
				token, 
				self.jwt_secret, 
				algorithms=[self.jwt_algorithm]
			)
			
			user_id = payload.get("user_id")
			session_id = payload.get("session_id")
			tenant_id = payload.get("tenant_id")
			
			if not all([user_id, session_id, tenant_id]):
				return None
			
			# Check if session exists and is valid
			if session_id not in self.active_sessions:
				return None
			
			user_context = self.active_sessions[session_id]
			
			# Check session timeout
			if (datetime.utcnow() - user_context.last_activity) > self.session_timeout:
				await self.invalidate_session(session_id)
				return None
			
			# Update last activity
			user_context.last_activity = datetime.utcnow()
			
			return user_context
			
		except jwt.ExpiredSignatureError:
			logger.warning("Token expired")
			return None
		except jwt.InvalidTokenError:
			logger.warning("Invalid token")
			return None
		except Exception as e:
			logger.error(f"Token validation error: {str(e)}")
			return None
	
	async def create_token(self, user_context: CRMUserContext) -> str:
		"""
		Create JWT token for user
		
		Args:
			user_context: CRM user context
			
		Returns:
			JWT token string
		"""
		try:
			now = datetime.utcnow()
			payload = {
				"user_id": user_context.user_id,
				"username": user_context.username,
				"tenant_id": user_context.tenant_id,
				"session_id": user_context.session_id,
				"roles": [role.value for role in user_context.roles],
				"iat": now,
				"exp": now + self.token_expiry
			}
			
			token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
			return token
			
		except Exception as e:
			logger.error(f"Token creation error: {str(e)}")
			raise
	
	async def check_permission(
		self, 
		user_context: CRMUserContext, 
		permission: CRMPermission,
		resource_data: Optional[Dict[str, Any]] = None
	) -> bool:
		"""
		Check if user has specific permission
		
		Args:
			user_context: CRM user context
			permission: Permission to check
			resource_data: Optional resource data for contextual checks
			
		Returns:
			True if user has permission
		"""
		try:
			# Check if user has the permission
			if permission not in user_context.permissions:
				return False
			
			# Apply RBAC policies if resource data provided
			if resource_data:
				return await self._apply_rbac_policies(
					user_context, 
					permission, 
					resource_data
				)
			
			return True
			
		except Exception as e:
			logger.error(f"Permission check error: {str(e)}")
			return False
	
	async def _apply_rbac_policies(
		self,
		user_context: CRMUserContext,
		permission: CRMPermission,
		resource_data: Dict[str, Any]
	) -> bool:
		"""Apply RBAC policies for access control"""
		try:
			# Tenant isolation check
			if resource_data.get("tenant_id") != user_context.tenant_id:
				return False
			
			# Owner-based access check
			if "owner_id" in resource_data:
				if resource_data["owner_id"] == user_context.user_id:
					return True
			
			# Territory-based access check
			if "territory" in resource_data and user_context.territories:
				if resource_data["territory"] in user_context.territories:
					return True
			
			# Department-based access check
			if "department_id" in resource_data and user_context.department_id:
				if resource_data["department_id"] == user_context.department_id:
					return True
			
			# Manager access check
			if "assigned_to_id" in resource_data and user_context.manager_id:
				# Managers can access records of their subordinates
				# This would require looking up reporting hierarchy
				pass
			
			# Admin users have access to everything in their tenant
			if CRMRole.CRM_ADMIN in user_context.roles:
				return True
			
			return False
			
		except Exception as e:
			logger.error(f"RBAC policy application error: {str(e)}")
			return False
	
	async def _create_crm_user_context(self, apg_user: UserContext) -> CRMUserContext:
		"""Create CRM user context from APG user"""
		# Get user's CRM roles
		crm_roles = await self._get_user_crm_roles(apg_user.user_id)
		
		# Calculate permissions from roles
		permissions = set()
		for role in crm_roles:
			role_permissions = self.role_permissions.get(role, set())
			permissions.update(role_permissions)
		
		# Get user territories and department
		territories = await self._get_user_territories(apg_user.user_id)
		department_id = await self._get_user_department(apg_user.user_id)
		manager_id = await self._get_user_manager(apg_user.user_id)
		
		return CRMUserContext(
			user_id=apg_user.user_id,
			username=apg_user.username,
			email=apg_user.email,
			tenant_id=apg_user.tenant_id,
			roles=crm_roles,
			permissions=permissions,
			territories=territories,
			department_id=department_id,
			manager_id=manager_id,
			is_active=apg_user.is_active
		)
	
	async def _get_user_crm_roles(self, user_id: str) -> List[CRMRole]:
		"""Get user's CRM roles"""
		# This would query the user's role assignments
		# For now, return default role
		return [CRMRole.SALES_REP]
	
	async def _get_user_territories(self, user_id: str) -> List[str]:
		"""Get user's assigned territories"""
		# This would query user territory assignments
		return ["default_territory"]
	
	async def _get_user_department(self, user_id: str) -> Optional[str]:
		"""Get user's department"""
		# This would query user department
		return "sales"
	
	async def _get_user_manager(self, user_id: str) -> Optional[str]:
		"""Get user's manager"""
		# This would query reporting hierarchy
		return None
	
	def _generate_session_id(self) -> str:
		"""Generate unique session ID"""
		import uuid
		return str(uuid.uuid4())
	
	async def invalidate_session(self, session_id: str):
		"""Invalidate user session"""
		if session_id in self.active_sessions:
			user = self.active_sessions[session_id]
			del self.active_sessions[session_id]
			logger.info(f"🔓 Session invalidated for user: {user.username}")
	
	async def refresh_user_permissions(self, user_id: str):
		"""Refresh user permissions (after role changes)"""
		# Find active sessions for user
		for session_id, user_context in self.active_sessions.items():
			if user_context.user_id == user_id:
				# Refresh roles and permissions
				apg_user = await self.apg_auth.get_user(user_id)
				if apg_user:
					updated_context = await self._create_crm_user_context(apg_user)
					updated_context.session_id = session_id
					updated_context.login_time = user_context.login_time
					updated_context.last_activity = user_context.last_activity
					self.active_sessions[session_id] = updated_context
	
	async def cleanup_expired_sessions(self):
		"""Cleanup expired sessions"""
		now = datetime.utcnow()
		expired_sessions = []
		
		for session_id, user_context in self.active_sessions.items():
			if (now - user_context.last_activity) > self.session_timeout:
				expired_sessions.append(session_id)
		
		for session_id in expired_sessions:
			await self.invalidate_session(session_id)
		
		if expired_sessions:
			logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
	
	def get_active_session_count(self) -> int:
		"""Get count of active sessions"""
		return len(self.active_sessions)
	
	async def health_check(self) -> Dict[str, Any]:
		"""Health check for auth provider"""
		return {
			"status": "healthy" if self._initialized else "unhealthy",
			"active_sessions": len(self.active_sessions),
			"apg_auth_status": await self.apg_auth.health_check(),
			"timestamp": datetime.utcnow().isoformat()
		}
	
	async def shutdown(self):
		"""Shutdown auth provider"""
		try:
			logger.info("🛑 Shutting down CRM auth provider...")
			
			# Clear all sessions
			self.active_sessions.clear()
			
			# Shutdown APG auth
			if self.apg_auth:
				await self.apg_auth.shutdown()
			
			self._initialized = False
			logger.info("✅ CRM auth provider shutdown completed")
			
		except Exception as e:
			logger.error(f"Error during auth provider shutdown: {str(e)}", exc_info=True)


# Utility decorators for permission checking

def require_permission(permission: CRMPermission):
	"""Decorator to require specific permission"""
	def decorator(func):
		async def wrapper(*args, **kwargs):
			# This would be used with FastAPI dependency injection
			# to check permissions before executing endpoint
			return await func(*args, **kwargs)
		return wrapper
	return decorator


def require_role(role: CRMRole):
	"""Decorator to require specific role"""
	def decorator(func):
		async def wrapper(*args, **kwargs):
			# This would be used with FastAPI dependency injection
			# to check roles before executing endpoint
			return await func(*args, **kwargs)
		return wrapper
	return decorator


# Export classes and functions
__all__ = [
	"CRMAuthProvider",
	"CRMUserContext",
	"CRMPermission",
	"CRMRole",
	"require_permission",
	"require_role"
]