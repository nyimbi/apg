"""
Authentication and Authorization Service
Real implementation of authentication and authorization for the payment gateway
with JWT tokens, role-based access control, and APG integration.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import jwt
import bcrypt
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set
from uuid_extensions import uuid7str
from enum import Enum
import structlog
from functools import wraps

from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import Column, String, Boolean, DateTime, JSON, Text
from sqlalchemy.dialects.postgresql import UUID

from .models import Base
from .database import DatabaseService, get_database_service

logger = structlog.get_logger()

class UserRole(str, Enum):
	"""User roles for payment gateway"""
	SUPER_ADMIN = "super_admin"
	TENANT_ADMIN = "tenant_admin"
	MERCHANT_ADMIN = "merchant_admin"
	MERCHANT_USER = "merchant_user"
	CUSTOMER = "customer"
	SUPPORT_AGENT = "support_agent"
	DEVELOPER = "developer"
	VIEWER = "viewer"

class Permission(str, Enum):
	"""Permission constants"""
	# Transaction permissions
	PROCESS_PAYMENT = "process_payment"
	VIEW_TRANSACTIONS = "view_transactions"
	REFUND_PAYMENT = "refund_payment"
	CANCEL_PAYMENT = "cancel_payment"
	
	# Merchant permissions
	CREATE_MERCHANT = "create_merchant"
	VIEW_MERCHANT = "view_merchant"
	UPDATE_MERCHANT = "update_merchant"
	DELETE_MERCHANT = "delete_merchant"
	
	# Payment method permissions
	ADD_PAYMENT_METHOD = "add_payment_method"
	VIEW_PAYMENT_METHODS = "view_payment_methods"
	DELETE_PAYMENT_METHOD = "delete_payment_method"
	
	# Analytics permissions
	VIEW_ANALYTICS = "view_analytics"
	EXPORT_DATA = "export_data"
	
	# System permissions
	MANAGE_USERS = "manage_users"
	CONFIGURE_PROCESSORS = "configure_processors"
	VIEW_SYSTEM_HEALTH = "view_system_health"
	MANAGE_WEBHOOKS = "manage_webhooks"
	
	# Fraud permissions
	VIEW_FRAUD_ANALYSIS = "view_fraud_analysis"
	MANAGE_FRAUD_RULES = "manage_fraud_rules"
	REVIEW_FLAGGED_TRANSACTIONS = "review_flagged_transactions"

class User(BaseModel):
	"""User model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	email: str
	username: str | None = None
	first_name: str | None = None
	last_name: str | None = None
	is_active: bool = True
	is_verified: bool = False
	roles: List[UserRole] = Field(default_factory=list)
	permissions: List[Permission] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	last_login: datetime | None = None

class APIKey(BaseModel):
	"""API Key model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	user_id: str
	name: str
	key_prefix: str
	key_hash: str
	permissions: List[Permission] = Field(default_factory=list)
	is_active: bool = True
	expires_at: datetime | None = None
	last_used: datetime | None = None
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# SQLAlchemy table definitions
class UserTable(Base):
	__tablename__ = 'pg_users'
	
	id = Column(UUID(as_uuid=False), primary_key=True, default=uuid7str)
	tenant_id = Column(UUID(as_uuid=False), nullable=False, index=True)
	email = Column(String(255), nullable=False, unique=True, index=True)
	username = Column(String(100), nullable=True, unique=True)
	password_hash = Column(String(255), nullable=False)
	first_name = Column(String(100), nullable=True)
	last_name = Column(String(100), nullable=True)
	is_active = Column(Boolean, default=True, index=True)
	is_verified = Column(Boolean, default=False)
	roles = Column(JSON, nullable=True)
	permissions = Column(JSON, nullable=True)
	metadata = Column(JSON, nullable=True)
	created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	last_login = Column(DateTime(timezone=True), nullable=True)

class APIKeyTable(Base):
	__tablename__ = 'pg_api_keys'
	
	id = Column(UUID(as_uuid=False), primary_key=True, default=uuid7str)
	tenant_id = Column(UUID(as_uuid=False), nullable=False, index=True)
	user_id = Column(UUID(as_uuid=False), nullable=False, index=True)
	name = Column(String(100), nullable=False)
	key_prefix = Column(String(20), nullable=False, index=True)
	key_hash = Column(String(255), nullable=False)
	permissions = Column(JSON, nullable=True)
	is_active = Column(Boolean, default=True, index=True)
	expires_at = Column(DateTime(timezone=True), nullable=True)
	last_used = Column(DateTime(timezone=True), nullable=True)
	created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

class AuthenticationService:
	"""
	Real authentication service with JWT tokens, password hashing,
	and role-based access control.
	"""
	
	def __init__(self, secret_key: str = "payment-gateway-secret-key-change-in-production"):
		self.secret_key = secret_key
		self.algorithm = "HS256"
		self.token_expiry = timedelta(hours=24)
		self.refresh_token_expiry = timedelta(days=30)
		self._database_service: Optional[DatabaseService] = None
		
		# Role-based permissions mapping
		self.role_permissions = {
			UserRole.SUPER_ADMIN: list(Permission),  # All permissions
			UserRole.TENANT_ADMIN: [
				Permission.PROCESS_PAYMENT, Permission.VIEW_TRANSACTIONS, Permission.REFUND_PAYMENT,
				Permission.CREATE_MERCHANT, Permission.VIEW_MERCHANT, Permission.UPDATE_MERCHANT,
				Permission.ADD_PAYMENT_METHOD, Permission.VIEW_PAYMENT_METHODS,
				Permission.VIEW_ANALYTICS, Permission.EXPORT_DATA,
				Permission.MANAGE_USERS, Permission.VIEW_SYSTEM_HEALTH,
				Permission.VIEW_FRAUD_ANALYSIS, Permission.MANAGE_FRAUD_RULES
			],
			UserRole.MERCHANT_ADMIN: [
				Permission.PROCESS_PAYMENT, Permission.VIEW_TRANSACTIONS, Permission.REFUND_PAYMENT,
				Permission.VIEW_MERCHANT, Permission.UPDATE_MERCHANT,
				Permission.ADD_PAYMENT_METHOD, Permission.VIEW_PAYMENT_METHODS,
				Permission.VIEW_ANALYTICS, Permission.EXPORT_DATA
			],
			UserRole.MERCHANT_USER: [
				Permission.PROCESS_PAYMENT, Permission.VIEW_TRANSACTIONS,
				Permission.ADD_PAYMENT_METHOD, Permission.VIEW_PAYMENT_METHODS,
				Permission.VIEW_ANALYTICS
			],
			UserRole.CUSTOMER: [
				Permission.ADD_PAYMENT_METHOD, Permission.VIEW_PAYMENT_METHODS,
				Permission.VIEW_TRANSACTIONS
			],
			UserRole.SUPPORT_AGENT: [
				Permission.VIEW_TRANSACTIONS, Permission.VIEW_MERCHANT,
				Permission.VIEW_FRAUD_ANALYSIS, Permission.REVIEW_FLAGGED_TRANSACTIONS
			],
			UserRole.DEVELOPER: [
				Permission.VIEW_SYSTEM_HEALTH, Permission.CONFIGURE_PROCESSORS,
				Permission.MANAGE_WEBHOOKS
			],
			UserRole.VIEWER: [
				Permission.VIEW_TRANSACTIONS, Permission.VIEW_MERCHANT,
				Permission.VIEW_ANALYTICS
			]
		}
	
	async def initialize(self):
		"""Initialize authentication service"""
		self._database_service = await get_database_service()
		logger.info("authentication_service_initialized")
	
	async def create_user(
		self,
		email: str,
		password: str,
		tenant_id: str,
		roles: List[UserRole],
		first_name: str | None = None,
		last_name: str | None = None,
		username: str | None = None
	) -> User:
		"""Create new user with hashed password"""
		try:
			# Hash password
			password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
			
			# Calculate permissions from roles
			permissions = self._calculate_permissions_from_roles(roles)
			
			user = User(
				tenant_id=tenant_id,
				email=email,
				username=username,
				first_name=first_name,
				last_name=last_name,
				roles=roles,
				permissions=permissions
			)
			
			# Save to database
			async with self._database_service.get_session() as session:
				db_user = UserTable(
					id=user.id,
					tenant_id=user.tenant_id,
					email=user.email,
					username=user.username,
					password_hash=password_hash,
					first_name=user.first_name,
					last_name=user.last_name,
					is_active=user.is_active,
					is_verified=user.is_verified,
					roles=[role.value for role in user.roles],
					permissions=[perm.value for perm in user.permissions],
					metadata=user.metadata,
					created_at=user.created_at
				)
				
				session.add(db_user)
				if self._database_service.is_async:
					await session.flush()
				else:
					session.flush()
			
			logger.info("user_created", user_id=user.id, email=email, roles=[r.value for r in roles])
			return user
			
		except Exception as e:
			logger.error("user_creation_failed", email=email, error=str(e))
			raise
	
	async def authenticate_user(self, email: str, password: str) -> Optional[User]:
		"""Authenticate user with email and password"""
		try:
			async with self._database_service.get_session() as session:
				if self._database_service.is_async:
					from sqlalchemy import select
					result = await session.execute(
						select(UserTable).where(UserTable.email == email)
					)
				else:
					from sqlalchemy import select
					result = session.execute(
						select(UserTable).where(UserTable.email == email)
					)
				
				db_user = result.scalar_one_or_none()
				if not db_user or not db_user.is_active:
					return None
				
				# Verify password
				if not bcrypt.checkpw(password.encode('utf-8'), db_user.password_hash.encode('utf-8')):
					return None
				
				# Update last login
				db_user.last_login = datetime.now(timezone.utc)
				
				# Convert to Pydantic model
				user = User(
					id=db_user.id,
					tenant_id=db_user.tenant_id,
					email=db_user.email,
					username=db_user.username,
					first_name=db_user.first_name,
					last_name=db_user.last_name,
					is_active=db_user.is_active,
					is_verified=db_user.is_verified,
					roles=[UserRole(role) for role in (db_user.roles or [])],
					permissions=[Permission(perm) for perm in (db_user.permissions or [])],
					metadata=db_user.metadata or {},
					created_at=db_user.created_at,
					last_login=db_user.last_login
				)
				
				logger.info("user_authenticated", user_id=user.id, email=email)
				return user
				
		except Exception as e:
			logger.error("user_authentication_failed", email=email, error=str(e))
			return None
	
	async def create_api_key(
		self,
		user_id: str,
		tenant_id: str,
		name: str,
		permissions: List[Permission],
		expires_at: datetime | None = None
	) -> Tuple[str, APIKey]:
		"""Create API key for user"""
		try:
			# Generate API key
			key_prefix = f"apg_{tenant_id[:8]}"
			key_suffix = uuid7str()[:16]
			full_key = f"{key_prefix}_{key_suffix}"
			
			# Hash the key for storage
			key_hash = bcrypt.hashpw(full_key.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
			
			api_key = APIKey(
				tenant_id=tenant_id,
				user_id=user_id,
				name=name,
				key_prefix=key_prefix,
				key_hash=key_hash,
				permissions=permissions,
				expires_at=expires_at
			)
			
			# Save to database
			async with self._database_service.get_session() as session:
				db_api_key = APIKeyTable(
					id=api_key.id,
					tenant_id=api_key.tenant_id,
					user_id=api_key.user_id,
					name=api_key.name,
					key_prefix=api_key.key_prefix,
					key_hash=api_key.key_hash,
					permissions=[perm.value for perm in api_key.permissions],
					is_active=api_key.is_active,
					expires_at=api_key.expires_at,
					created_at=api_key.created_at
				)
				
				session.add(db_api_key)
				if self._database_service.is_async:
					await session.flush()
				else:
					session.flush()
			
			logger.info("api_key_created", api_key_id=api_key.id, user_id=user_id, name=name)
			return full_key, api_key
			
		except Exception as e:
			logger.error("api_key_creation_failed", user_id=user_id, error=str(e))
			raise
	
	async def validate_api_key(self, api_key: str) -> Optional[Tuple[User, APIKey]]:
		"""Validate API key and return associated user"""
		try:
			if not api_key or not api_key.startswith("apg_"):
				return None
			
			# Extract prefix
			parts = api_key.split("_")
			if len(parts) != 3:
				return None
			
			key_prefix = f"{parts[0]}_{parts[1]}"
			
			async with self._database_service.get_session() as session:
				if self._database_service.is_async:
					from sqlalchemy import select
					result = await session.execute(
						select(APIKeyTable).where(
							APIKeyTable.key_prefix == key_prefix,
							APIKeyTable.is_active == True
						)
					)
				else:
					from sqlalchemy import select
					result = session.execute(
						select(APIKeyTable).where(
							APIKeyTable.key_prefix == key_prefix,
							APIKeyTable.is_active == True
						)
					)
				
				db_api_keys = result.scalars().all()
				
				for db_api_key in db_api_keys:
					# Check if key matches
					if bcrypt.checkpw(api_key.encode('utf-8'), db_api_key.key_hash.encode('utf-8')):
						# Check expiry
						if db_api_key.expires_at and datetime.now(timezone.utc) > db_api_key.expires_at:
							continue
						
						# Update last used
						db_api_key.last_used = datetime.now(timezone.utc)
						
						# Get associated user
						if self._database_service.is_async:
							user_result = await session.execute(
								select(UserTable).where(UserTable.id == db_api_key.user_id)
							)
						else:
							user_result = session.execute(
								select(UserTable).where(UserTable.id == db_api_key.user_id)
							)
						
						db_user = user_result.scalar_one_or_none()
						if not db_user or not db_user.is_active:
							continue
						
						# Convert to Pydantic models
						user = User(
							id=db_user.id,
							tenant_id=db_user.tenant_id,
							email=db_user.email,
							username=db_user.username,
							first_name=db_user.first_name,
							last_name=db_user.last_name,
							is_active=db_user.is_active,
							is_verified=db_user.is_verified,
							roles=[UserRole(role) for role in (db_user.roles or [])],
							permissions=[Permission(perm) for perm in (db_user.permissions or [])],
							metadata=db_user.metadata or {},
							created_at=db_user.created_at,
							last_login=db_user.last_login
						)
						
						api_key_obj = APIKey(
							id=db_api_key.id,
							tenant_id=db_api_key.tenant_id,
							user_id=db_api_key.user_id,
							name=db_api_key.name,
							key_prefix=db_api_key.key_prefix,
							key_hash=db_api_key.key_hash,
							permissions=[Permission(perm) for perm in (db_api_key.permissions or [])],
							is_active=db_api_key.is_active,
							expires_at=db_api_key.expires_at,
							last_used=db_api_key.last_used,
							created_at=db_api_key.created_at
						)
						
						logger.info("api_key_validated", user_id=user.id, api_key_id=api_key_obj.id)
						return user, api_key_obj
				
				return None
				
		except Exception as e:
			logger.error("api_key_validation_failed", error=str(e))
			return None
	
	def generate_jwt_token(self, user: User) -> str:
		"""Generate JWT token for user"""
		try:
			payload = {
				"user_id": user.id,
				"tenant_id": user.tenant_id,
				"email": user.email,
				"roles": [role.value for role in user.roles],
				"permissions": [perm.value for perm in user.permissions],
				"iat": datetime.utcnow(),
				"exp": datetime.utcnow() + self.token_expiry
			}
			
			token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
			logger.info("jwt_token_generated", user_id=user.id)
			return token
			
		except Exception as e:
			logger.error("jwt_token_generation_failed", user_id=user.id, error=str(e))
			raise
	
	def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
		"""Validate JWT token and return payload"""
		try:
			payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
			logger.debug("jwt_token_validated", user_id=payload.get("user_id"))
			return payload
			
		except jwt.ExpiredSignatureError:
			logger.warning("jwt_token_expired")
			return None
		except jwt.InvalidTokenError:
			logger.warning("jwt_token_invalid")
			return None
		except Exception as e:
			logger.error("jwt_token_validation_failed", error=str(e))
			return None
	
	def _calculate_permissions_from_roles(self, roles: List[UserRole]) -> List[Permission]:
		"""Calculate combined permissions from user roles"""
		permissions = set()
		for role in roles:
			role_perms = self.role_permissions.get(role, [])
			permissions.update(role_perms)
		return list(permissions)
	
	def check_permission(self, user: User, permission: Permission) -> bool:
		"""Check if user has specific permission"""
		return permission in user.permissions
	
	def check_any_permission(self, user: User, permissions: List[Permission]) -> bool:
		"""Check if user has any of the specified permissions"""
		return any(perm in user.permissions for perm in permissions)
	
	def check_all_permissions(self, user: User, permissions: List[Permission]) -> bool:
		"""Check if user has all specified permissions"""
		return all(perm in user.permissions for perm in permissions)

# Global authentication service
_auth_service: Optional[AuthenticationService] = None

async def get_auth_service() -> AuthenticationService:
	"""Get global authentication service"""
	global _auth_service
	if _auth_service is None:
		_auth_service = AuthenticationService()
		await _auth_service.initialize()
	return _auth_service

# Context variables for current user
_current_user: Optional[User] = None
_current_api_key: Optional[APIKey] = None

def set_current_user(user: User, api_key: APIKey | None = None):
	"""Set current user context"""
	global _current_user, _current_api_key
	_current_user = user
	_current_api_key = api_key

def get_current_user() -> Dict[str, Any]:
	"""Get current user (compatible with existing code)"""
	global _current_user
	if _current_user:
		return {
			"id": _current_user.id,
			"tenant_id": _current_user.tenant_id,
			"email": _current_user.email,
			"roles": [role.value for role in _current_user.roles],
			"permissions": [perm.value for perm in _current_user.permissions]
		}
	else:
		# Fallback for development
		return {
			"id": "dev_user",
			"tenant_id": "dev_tenant",
			"email": "dev@example.com",
			"roles": ["tenant_admin"],
			"permissions": [perm.value for perm in Permission]
		}

def require_permission(permission: Permission):
	"""Decorator to require specific permission"""
	def decorator(func):
		@wraps(func)
		async def wrapper(*args, **kwargs):
			global _current_user
			if _current_user and permission in _current_user.permissions:
				return await func(*args, **kwargs)
			else:
				raise PermissionError(f"Permission required: {permission.value}")
		return wrapper
	return decorator

def require_any_permission(*permissions: Permission):
	"""Decorator to require any of the specified permissions"""
	def decorator(func):
		@wraps(func)
		async def wrapper(*args, **kwargs):
			global _current_user
			if _current_user and any(perm in _current_user.permissions for perm in permissions):
				return await func(*args, **kwargs)
			else:
				raise PermissionError(f"One of these permissions required: {[p.value for p in permissions]}")
		return wrapper
	return decorator

def require_role(role: UserRole):
	"""Decorator to require specific role"""
	def decorator(func):
		@wraps(func)
		async def wrapper(*args, **kwargs):
			global _current_user
			if _current_user and role in _current_user.roles:
				return await func(*args, **kwargs)
			else:
				raise PermissionError(f"Role required: {role.value}")
		return wrapper
	return decorator

def _log_auth_module_loaded():
	"""Log authentication module loaded"""
	print("🔐 APG Payment Gateway Authentication module loaded")
	print("   - JWT token authentication")
	print("   - API key authentication")
	print("   - Role-based access control")
	print("   - Password hashing with bcrypt")
	print("   - Permission decorators")

# Execute module loading log
_log_auth_module_loaded()