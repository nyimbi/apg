"""
APG Customer Relationship Management - Advanced Authentication & Authorization

Enterprise-grade authentication and authorization system with multi-factor authentication,
role-based access control, session management, and advanced security features.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import bcrypt
import hashlib
import jwt
import secrets
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import re
import ipaddress
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, field_validator, AfterValidator
from annotated_types import Annotated

import asyncpg
import redis.asyncio as redis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import pyotp
import qrcode
from io import BytesIO
import base64


logger = logging.getLogger(__name__)


# ================================
# Enums and Constants
# ================================

class AuthenticationMethod(Enum):
	PASSWORD = "password"
	MFA_TOTP = "mfa_totp"
	MFA_SMS = "mfa_sms"
	MFA_EMAIL = "mfa_email"
	BIOMETRIC = "biometric"
	OAUTH2 = "oauth2"
	SAML = "saml"
	LDAP = "ldap"
	API_KEY = "api_key"
	JWT_TOKEN = "jwt_token"


class UserRole(Enum):
	SUPER_ADMIN = "super_admin"
	TENANT_ADMIN = "tenant_admin"
	SALES_MANAGER = "sales_manager"
	SALES_REP = "sales_rep"
	MARKETING_MANAGER = "marketing_manager"
	MARKETING_USER = "marketing_user"
	SUPPORT_MANAGER = "support_manager"
	SUPPORT_AGENT = "support_agent"
	ANALYST = "analyst"
	VIEWER = "viewer"
	API_USER = "api_user"
	INTEGRATION_USER = "integration_user"


class PermissionScope(Enum):
	GLOBAL = "global"
	TENANT = "tenant"
	DEPARTMENT = "department"
	TEAM = "team"
	PERSONAL = "personal"


class ResourceType(Enum):
	CONTACT = "contact"
	ACCOUNT = "account"
	LEAD = "lead"
	OPPORTUNITY = "opportunity"
	ACTIVITY = "activity"
	CAMPAIGN = "campaign"
	REPORT = "report"
	DASHBOARD = "dashboard"
	INTEGRATION = "integration"
	SYSTEM_CONFIG = "system_config"
	USER_MANAGEMENT = "user_management"


class PermissionAction(Enum):
	CREATE = "create"
	READ = "read"
	UPDATE = "update"
	DELETE = "delete"
	EXPORT = "export"
	IMPORT = "import"
	SHARE = "share"
	APPROVE = "approve"
	CONFIGURE = "configure"
	ADMINISTER = "administer"


class SessionStatus(Enum):
	ACTIVE = "active"
	EXPIRED = "expired"
	REVOKED = "revoked"
	SUSPICIOUS = "suspicious"
	LOCKED = "locked"


class AuthEventType(Enum):
	LOGIN_SUCCESS = "login_success"
	LOGIN_FAILED = "login_failed"
	LOGOUT = "logout"
	PASSWORD_CHANGE = "password_change"
	MFA_SETUP = "mfa_setup"
	MFA_SUCCESS = "mfa_success"
	MFA_FAILED = "mfa_failed"
	PERMISSION_DENIED = "permission_denied"
	SESSION_EXPIRED = "session_expired"
	SUSPICIOUS_ACTIVITY = "suspicious_activity"
	ACCOUNT_LOCKED = "account_locked"
	ACCOUNT_UNLOCKED = "account_unlocked"


# ================================
# Pydantic Models
# ================================

class UserAccount(BaseModel):
	"""User account model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	username: str
	email: str
	first_name: str
	last_name: str
	phone_number: Optional[str] = None
	password_hash: str
	salt: str
	is_active: bool = True
	is_verified: bool = False
	is_locked: bool = False
	failed_login_attempts: int = 0
	last_login_at: Optional[datetime] = None
	last_password_change: Optional[datetime] = None
	password_expires_at: Optional[datetime] = None
	mfa_enabled: bool = False
	mfa_secret: Optional[str] = None
	backup_codes: List[str] = Field(default_factory=list)
	roles: List[UserRole] = Field(default_factory=list)
	permissions: Dict[str, Any] = Field(default_factory=dict)
	profile_data: Dict[str, Any] = Field(default_factory=dict)
	security_preferences: Dict[str, Any] = Field(default_factory=dict)
	login_history: List[Dict[str, Any]] = Field(default_factory=list)
	device_tokens: List[str] = Field(default_factory=list)
	api_keys: List[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	@field_validator('email')
	@classmethod
	def validate_email(cls, v):
		email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
		if not email_pattern.match(v):
			raise ValueError('Invalid email format')
		return v.lower()
	
	@field_validator('username')
	@classmethod
	def validate_username(cls, v):
		if len(v) < 3 or len(v) > 50:
			raise ValueError('Username must be between 3 and 50 characters')
		if not re.match(r'^[a-zA-Z0-9_.-]+$', v):
			raise ValueError('Username can only contain alphanumeric characters, dots, dashes, and underscores')
		return v.lower()


class Permission(BaseModel):
	"""Permission model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	permission_name: str
	resource_type: ResourceType
	action: PermissionAction
	scope: PermissionScope
	conditions: Dict[str, Any] = Field(default_factory=dict)
	is_system_permission: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	metadata: Dict[str, Any] = Field(default_factory=dict)


class Role(BaseModel):
	"""Role model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	role_name: str
	role_type: UserRole
	description: str
	permissions: List[str] = Field(default_factory=list)
	is_system_role: bool = False
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	metadata: Dict[str, Any] = Field(default_factory=dict)


class UserSession(BaseModel):
	"""User session model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	session_token: str
	user_id: str
	tenant_id: str
	device_info: Dict[str, Any] = Field(default_factory=dict)
	ip_address: str
	user_agent: str
	location_data: Dict[str, Any] = Field(default_factory=dict)
	status: SessionStatus = SessionStatus.ACTIVE
	created_at: datetime = Field(default_factory=datetime.utcnow)
	expires_at: datetime
	last_activity_at: datetime = Field(default_factory=datetime.utcnow)
	activity_count: int = 0
	security_score: Decimal = Decimal('100.0')
	risk_factors: List[str] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)


class AuthenticationEvent(BaseModel):
	"""Authentication event model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	user_id: Optional[str] = None
	session_id: Optional[str] = None
	event_type: AuthEventType
	authentication_method: AuthenticationMethod
	ip_address: str
	user_agent: str
	device_fingerprint: Optional[str] = None
	location_data: Dict[str, Any] = Field(default_factory=dict)
	success: bool
	error_code: Optional[str] = None
	error_message: Optional[str] = None
	risk_score: Decimal = Decimal('0.0')
	additional_data: Dict[str, Any] = Field(default_factory=dict)
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	metadata: Dict[str, Any] = Field(default_factory=dict)


class MFAChallenge(BaseModel):
	"""Multi-factor authentication challenge"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	user_id: str
	tenant_id: str
	challenge_type: AuthenticationMethod
	challenge_data: Dict[str, Any] = Field(default_factory=dict)
	is_verified: bool = False
	attempts: int = 0
	max_attempts: int = 3
	created_at: datetime = Field(default_factory=datetime.utcnow)
	expires_at: datetime
	verified_at: Optional[datetime] = None
	metadata: Dict[str, Any] = Field(default_factory=dict)


# ================================
# Advanced Authentication Manager
# ================================

class AdvancedAuthenticationManager:
	"""Advanced authentication and authorization manager"""
	
	def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis = None):
		self.db_pool = db_pool
		self.redis_client = redis_client
		self.jwt_secret = self._generate_secure_key()
		self.encryption_key = self._generate_encryption_key()
		self.fernet = Fernet(self.encryption_key)
		self._initialized = False
	
	def _generate_secure_key(self) -> str:
		"""Generate a secure key for JWT signing"""
		return secrets.token_urlsafe(64)
	
	def _generate_encryption_key(self) -> bytes:
		"""Generate encryption key for sensitive data"""
		password = secrets.token_bytes(32)
		salt = secrets.token_bytes(16)
		kdf = PBKDF2HMAC(
			algorithm=hashes.SHA256(),
			length=32,
			salt=salt,
			iterations=100000,
		)
		key = base64.urlsafe_b64encode(kdf.derive(password))
		return key
	
	async def initialize(self):
		"""Initialize the authentication manager"""
		try:
			if self._initialized:
				return
			
			logger.info("🔐 Initializing Advanced Authentication Manager...")
			
			# Validate database connection
			async with self.db_pool.acquire() as conn:
				await conn.execute("SELECT 1")
			
			# Initialize Redis if available
			if self.redis_client:
				await self.redis_client.ping()
				logger.info("📦 Redis connection established for session caching")
			
			# Set up system roles and permissions
			await self._initialize_system_roles_and_permissions()
			
			self._initialized = True
			logger.info("✅ Advanced Authentication Manager initialized successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to initialize authentication manager: {str(e)}")
			raise
	
	async def _initialize_system_roles_and_permissions(self):
		"""Initialize system roles and permissions"""
		try:
			async with self.db_pool.acquire() as conn:
				# Check if system roles already exist
				existing_roles = await conn.fetch("""
					SELECT role_name FROM crm_user_roles 
					WHERE is_system_role = true
				""")
				
				if existing_roles:
					logger.info("🔧 System roles and permissions already initialized")
					return
				
				# Create system permissions
				system_permissions = [
					{
						'permission_name': 'contact_full_access',
						'resource_type': ResourceType.CONTACT.value,
						'action': PermissionAction.ADMINISTER.value,
						'scope': PermissionScope.TENANT.value
					},
					{
						'permission_name': 'account_full_access',
						'resource_type': ResourceType.ACCOUNT.value,
						'action': PermissionAction.ADMINISTER.value,
						'scope': PermissionScope.TENANT.value
					},
					{
						'permission_name': 'system_administration',
						'resource_type': ResourceType.SYSTEM_CONFIG.value,
						'action': PermissionAction.ADMINISTER.value,
						'scope': PermissionScope.GLOBAL.value
					}
				]
				
				for perm in system_permissions:
					await conn.execute("""
						INSERT INTO crm_permissions (
							id, tenant_id, permission_name, resource_type, 
							action, scope, is_system_permission, created_by
						) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
						ON CONFLICT (tenant_id, permission_name) DO NOTHING
					""", 
					uuid7str(), 'system', perm['permission_name'], 
					perm['resource_type'], perm['action'], perm['scope'], 
					True, 'system')
				
				logger.info("🔧 System roles and permissions initialized")
				
		except Exception as e:
			logger.error(f"Failed to initialize system roles: {str(e)}")
			raise
	
	def _hash_password(self, password: str, salt: str = None) -> Tuple[str, str]:
		"""Hash password with salt"""
		if not salt:
			salt = bcrypt.gensalt().decode('utf-8')
		
		password_hash = bcrypt.hashpw(
			password.encode('utf-8'), 
			salt.encode('utf-8')
		).decode('utf-8')
		
		return password_hash, salt
	
	def _verify_password(self, password: str, hashed_password: str) -> bool:
		"""Verify password against hash"""
		try:
			return bcrypt.checkpw(
				password.encode('utf-8'), 
				hashed_password.encode('utf-8')
			)
		except Exception:
			return False
	
	def _generate_mfa_secret(self) -> str:
		"""Generate MFA secret key"""
		return pyotp.random_base32()
	
	def _generate_qr_code(self, secret: str, user_email: str) -> str:
		"""Generate QR code for MFA setup"""
		try:
			totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
				name=user_email,
				issuer_name="APG CRM"
			)
			
			qr = qrcode.QRCode(version=1, box_size=10, border=5)
			qr.add_data(totp_uri)
			qr.make(fit=True)
			
			img = qr.make_image(fill_color="black", back_color="white")
			buffer = BytesIO()
			img.save(buffer, format='PNG')
			buffer.seek(0)
			
			return base64.b64encode(buffer.getvalue()).decode()
			
		except Exception as e:
			logger.error(f"Error generating QR code: {str(e)}")
			return ""
	
	def _verify_totp_code(self, secret: str, code: str) -> bool:
		"""Verify TOTP code"""
		try:
			totp = pyotp.TOTP(secret)
			return totp.verify(code, valid_window=1)
		except Exception:
			return False
	
	def _generate_jwt_token(
		self, 
		user_id: str, 
		tenant_id: str, 
		session_id: str,
		expires_in: int = 3600
	) -> str:
		"""Generate JWT token"""
		payload = {
			'user_id': user_id,
			'tenant_id': tenant_id,
			'session_id': session_id,
			'iat': datetime.utcnow(),
			'exp': datetime.utcnow() + timedelta(seconds=expires_in)
		}
		
		return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
	
	def _verify_jwt_token(self, token: str) -> Dict[str, Any]:
		"""Verify JWT token"""
		try:
			payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
			return payload
		except jwt.ExpiredSignatureError:
			raise Exception("Token has expired")
		except jwt.InvalidTokenError:
			raise Exception("Invalid token")
	
	def _calculate_risk_score(
		self, 
		ip_address: str, 
		device_info: Dict[str, Any],
		login_history: List[Dict[str, Any]]
	) -> Decimal:
		"""Calculate security risk score"""
		risk_score = Decimal('0.0')
		
		# Check for unusual IP address
		if self._is_unusual_ip(ip_address, login_history):
			risk_score += Decimal('30.0')
		
		# Check for new device
		if self._is_new_device(device_info, login_history):
			risk_score += Decimal('20.0')
		
		# Check for suspicious patterns
		if self._has_suspicious_patterns(login_history):
			risk_score += Decimal('40.0')
		
		# Check for rapid login attempts
		if self._has_rapid_attempts(login_history):
			risk_score += Decimal('25.0')
		
		return min(risk_score, Decimal('100.0'))
	
	def _is_unusual_ip(self, ip_address: str, login_history: List[Dict[str, Any]]) -> bool:
		"""Check if IP address is unusual"""
		recent_ips = [
			item.get('ip_address') for item in login_history[-10:] 
			if item.get('success', False)
		]
		return ip_address not in recent_ips
	
	def _is_new_device(self, device_info: Dict[str, Any], login_history: List[Dict[str, Any]]) -> bool:
		"""Check if device is new"""
		device_fingerprint = device_info.get('fingerprint')
		if not device_fingerprint:
			return True
		
		recent_devices = [
			item.get('device_info', {}).get('fingerprint') 
			for item in login_history[-20:]
			if item.get('success', False)
		]
		return device_fingerprint not in recent_devices
	
	def _has_suspicious_patterns(self, login_history: List[Dict[str, Any]]) -> bool:
		"""Check for suspicious login patterns"""
		if len(login_history) < 3:
			return False
		
		# Check for multiple failed attempts followed by success
		recent_events = login_history[-5:]
		failed_count = sum(1 for event in recent_events if not event.get('success', True))
		
		return failed_count >= 3
	
	def _has_rapid_attempts(self, login_history: List[Dict[str, Any]]) -> bool:
		"""Check for rapid login attempts"""
		if len(login_history) < 3:
			return False
		
		recent_events = login_history[-3:]
		time_diff = (
			recent_events[-1]['timestamp'] - recent_events[0]['timestamp']
		).total_seconds()
		
		return time_diff < 60  # 3 attempts in less than 1 minute
	
	async def create_user_account(
		self,
		tenant_id: str,
		username: str,
		email: str,
		password: str,
		first_name: str,
		last_name: str,
		roles: List[UserRole] = None,
		created_by: str = None,
		**kwargs
	) -> UserAccount:
		"""Create a new user account"""
		try:
			# Validate password strength
			self._validate_password_strength(password)
			
			# Hash password
			password_hash, salt = self._hash_password(password)
			
			# Create user account
			user_data = {
				'id': uuid7str(),
				'tenant_id': tenant_id,
				'username': username,
				'email': email,
				'first_name': first_name,
				'last_name': last_name,
				'password_hash': password_hash,
				'salt': salt,
				'roles': [role.value for role in (roles or [])],
				'created_by': created_by or 'system',
				**kwargs
			}
			
			user_account = UserAccount(**user_data)
			
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_user_accounts (
						id, tenant_id, username, email, first_name, last_name,
						password_hash, salt, roles, is_active, created_by, metadata
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
				""", 
				user_account.id, user_account.tenant_id, user_account.username,
				user_account.email, user_account.first_name, user_account.last_name,
				user_account.password_hash, user_account.salt,
				json.dumps(user_account.roles), user_account.is_active,
				user_account.created_by, json.dumps(user_account.metadata))
			
			logger.info(f"👤 User account created: {username} ({email})")
			return user_account
			
		except Exception as e:
			logger.error(f"Error creating user account: {str(e)}")
			raise
	
	def _validate_password_strength(self, password: str):
		"""Validate password strength"""
		if len(password) < 8:
			raise ValueError("Password must be at least 8 characters long")
		
		if not re.search(r'[A-Z]', password):
			raise ValueError("Password must contain at least one uppercase letter")
		
		if not re.search(r'[a-z]', password):
			raise ValueError("Password must contain at least one lowercase letter")
		
		if not re.search(r'\d', password):
			raise ValueError("Password must contain at least one digit")
		
		if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
			raise ValueError("Password must contain at least one special character")
	
	async def authenticate_user(
		self,
		tenant_id: str,
		username_or_email: str,
		password: str,
		ip_address: str,
		user_agent: str,
		device_info: Dict[str, Any] = None
	) -> Dict[str, Any]:
		"""Authenticate user with password"""
		try:
			async with self.db_pool.acquire() as conn:
				# Get user account
				user_row = await conn.fetchrow("""
					SELECT * FROM crm_user_accounts 
					WHERE tenant_id = $1 AND (username = $2 OR email = $2)
					AND is_active = true
				""", tenant_id, username_or_email.lower())
				
				if not user_row:
					await self._log_auth_event(
						tenant_id, None, AuthEventType.LOGIN_FAILED,
						AuthenticationMethod.PASSWORD, ip_address, user_agent,
						success=False, error_message="User not found"
					)
					raise ValueError("Invalid credentials")
				
				user_data = dict(user_row)
				user_data['roles'] = json.loads(user_data.get('roles', '[]'))
				user_data['login_history'] = json.loads(user_data.get('login_history', '[]'))
				user_data['metadata'] = json.loads(user_data.get('metadata', '{}'))
				
				user_account = UserAccount(**user_data)
				
				# Check if account is locked
				if user_account.is_locked:
					await self._log_auth_event(
						tenant_id, user_account.id, AuthEventType.LOGIN_FAILED,
						AuthenticationMethod.PASSWORD, ip_address, user_agent,
						success=False, error_message="Account locked"
					)
					raise ValueError("Account is locked")
				
				# Verify password
				if not self._verify_password(password, user_account.password_hash):
					# Increment failed attempts
					await self._increment_failed_attempts(user_account.id)
					
					await self._log_auth_event(
						tenant_id, user_account.id, AuthEventType.LOGIN_FAILED,
						AuthenticationMethod.PASSWORD, ip_address, user_agent,
						success=False, error_message="Invalid password"
					)
					raise ValueError("Invalid credentials")
				
				# Calculate risk score
				risk_score = self._calculate_risk_score(
					ip_address, device_info or {}, user_account.login_history
				)
				
				# Check if MFA is required
				mfa_required = user_account.mfa_enabled or risk_score > Decimal('50.0')
				
				if mfa_required and not user_account.mfa_enabled:
					# Force MFA setup for high-risk login
					return {
						'status': 'mfa_setup_required',
						'user_id': user_account.id,
						'message': 'Multi-factor authentication setup required'
					}
				
				if mfa_required:
					# Create MFA challenge
					challenge = await self._create_mfa_challenge(
						user_account.id, tenant_id, AuthenticationMethod.MFA_TOTP
					)
					
					return {
						'status': 'mfa_required',
						'challenge_id': challenge.id,
						'user_id': user_account.id,
						'risk_score': float(risk_score)
					}
				
				# Create session
				session = await self._create_user_session(
					user_account.id, tenant_id, ip_address, user_agent,
					device_info or {}, risk_score
				)
				
				# Generate JWT token
				jwt_token = self._generate_jwt_token(
					user_account.id, tenant_id, session.id
				)
				
				# Update login history
				await self._update_login_history(user_account.id, ip_address, device_info)
				
				# Reset failed attempts
				await self._reset_failed_attempts(user_account.id)
				
				await self._log_auth_event(
					tenant_id, user_account.id, AuthEventType.LOGIN_SUCCESS,
					AuthenticationMethod.PASSWORD, ip_address, user_agent,
					success=True, session_id=session.id, risk_score=risk_score
				)
				
				return {
					'status': 'success',
					'user_id': user_account.id,
					'session_id': session.id,
					'jwt_token': jwt_token,
					'expires_at': session.expires_at.isoformat(),
					'user_info': {
						'username': user_account.username,
						'email': user_account.email,
						'first_name': user_account.first_name,
						'last_name': user_account.last_name,
						'roles': user_account.roles
					}
				}
				
		except Exception as e:
			logger.error(f"Authentication error: {str(e)}")
			raise
	
	async def _increment_failed_attempts(self, user_id: str):
		"""Increment failed login attempts"""
		async with self.db_pool.acquire() as conn:
			result = await conn.fetchrow("""
				UPDATE crm_user_accounts 
				SET failed_login_attempts = failed_login_attempts + 1,
					is_locked = CASE WHEN failed_login_attempts >= 4 THEN true ELSE is_locked END,
					updated_at = NOW()
				WHERE id = $1
				RETURNING failed_login_attempts, is_locked
			""", user_id)
			
			if result and result['is_locked']:
				logger.warning(f"🔒 User account locked due to failed attempts: {user_id}")
	
	async def _reset_failed_attempts(self, user_id: str):
		"""Reset failed login attempts"""
		async with self.db_pool.acquire() as conn:
			await conn.execute("""
				UPDATE crm_user_accounts 
				SET failed_login_attempts = 0, updated_at = NOW()
				WHERE id = $1
			""", user_id)
	
	async def _create_mfa_challenge(
		self, 
		user_id: str, 
		tenant_id: str, 
		challenge_type: AuthenticationMethod
	) -> MFAChallenge:
		"""Create MFA challenge"""
		challenge_data = {
			'id': uuid7str(),
			'user_id': user_id,
			'tenant_id': tenant_id,
			'challenge_type': challenge_type,
			'expires_at': datetime.utcnow() + timedelta(minutes=10)
		}
		
		challenge = MFAChallenge(**challenge_data)
		
		async with self.db_pool.acquire() as conn:
			await conn.execute("""
				INSERT INTO crm_mfa_challenges (
					id, user_id, tenant_id, challenge_type, expires_at, metadata
				) VALUES ($1, $2, $3, $4, $5, $6)
			""", 
			challenge.id, challenge.user_id, challenge.tenant_id,
			challenge.challenge_type.value, challenge.expires_at,
			json.dumps(challenge.metadata))
		
		return challenge
	
	async def _create_user_session(
		self,
		user_id: str,
		tenant_id: str,
		ip_address: str,
		user_agent: str,
		device_info: Dict[str, Any],
		risk_score: Decimal
	) -> UserSession:
		"""Create user session"""
		session_data = {
			'id': uuid7str(),
			'session_token': secrets.token_urlsafe(64),
			'user_id': user_id,
			'tenant_id': tenant_id,
			'device_info': device_info,
			'ip_address': ip_address,
			'user_agent': user_agent,
			'expires_at': datetime.utcnow() + timedelta(hours=24),
			'security_score': risk_score
		}
		
		session = UserSession(**session_data)
		
		async with self.db_pool.acquire() as conn:
			await conn.execute("""
				INSERT INTO crm_user_sessions (
					id, session_token, user_id, tenant_id, device_info,
					ip_address, user_agent, status, expires_at, security_score, metadata
				) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
			""", 
			session.id, session.session_token, session.user_id,
			session.tenant_id, json.dumps(session.device_info),
			session.ip_address, session.user_agent, session.status.value,
			session.expires_at, session.security_score, json.dumps(session.metadata))
		
		# Cache session in Redis if available
		if self.redis_client:
			await self.redis_client.setex(
				f"session:{session.session_token}",
				timedelta(hours=24),
				json.dumps({
					'user_id': user_id,
					'tenant_id': tenant_id,
					'expires_at': session.expires_at.isoformat()
				})
			)
		
		return session
	
	async def _update_login_history(
		self, 
		user_id: str, 
		ip_address: str, 
		device_info: Dict[str, Any]
	):
		"""Update user login history"""
		async with self.db_pool.acquire() as conn:
			# Get current login history
			history_row = await conn.fetchrow("""
				SELECT login_history FROM crm_user_accounts WHERE id = $1
			""", user_id)
			
			current_history = json.loads(history_row['login_history'] or '[]')
			
			# Add new login event
			new_event = {
				'timestamp': datetime.utcnow().isoformat(),
				'ip_address': ip_address,
				'device_info': device_info,
				'success': True
			}
			
			current_history.append(new_event)
			
			# Keep only last 50 events
			current_history = current_history[-50:]
			
			await conn.execute("""
				UPDATE crm_user_accounts 
				SET login_history = $1, last_login_at = NOW(), updated_at = NOW()
				WHERE id = $2
			""", json.dumps(current_history), user_id)
	
	async def _log_auth_event(
		self,
		tenant_id: str,
		user_id: Optional[str],
		event_type: AuthEventType,
		auth_method: AuthenticationMethod,
		ip_address: str,
		user_agent: str,
		success: bool,
		session_id: Optional[str] = None,
		risk_score: Optional[Decimal] = None,
		error_message: Optional[str] = None
	):
		"""Log authentication event"""
		try:
			event_data = {
				'id': uuid7str(),
				'tenant_id': tenant_id,
				'user_id': user_id,
				'session_id': session_id,
				'event_type': event_type,
				'authentication_method': auth_method,
				'ip_address': ip_address,
				'user_agent': user_agent,
				'success': success,
				'risk_score': risk_score or Decimal('0.0'),
				'error_message': error_message
			}
			
			event = AuthenticationEvent(**event_data)
			
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_auth_events (
						id, tenant_id, user_id, session_id, event_type,
						authentication_method, ip_address, user_agent,
						success, risk_score, error_message, metadata
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
				""", 
				event.id, event.tenant_id, event.user_id, event.session_id,
				event.event_type.value, event.authentication_method.value,
				event.ip_address, event.user_agent, event.success,
				event.risk_score, event.error_message, json.dumps(event.metadata))
			
		except Exception as e:
			logger.error(f"Error logging auth event: {str(e)}")
	
	async def verify_session(self, session_token: str) -> Dict[str, Any]:
		"""Verify user session"""
		try:
			# Check Redis cache first
			if self.redis_client:
				cached_session = await self.redis_client.get(f"session:{session_token}")
				if cached_session:
					session_data = json.loads(cached_session)
					expires_at = datetime.fromisoformat(session_data['expires_at'])
					
					if expires_at > datetime.utcnow():
						return {
							'valid': True,
							'user_id': session_data['user_id'],
							'tenant_id': session_data['tenant_id']
						}
			
			# Check database
			async with self.db_pool.acquire() as conn:
				session_row = await conn.fetchrow("""
					SELECT s.*, u.username, u.email, u.roles
					FROM crm_user_sessions s
					JOIN crm_user_accounts u ON s.user_id = u.id
					WHERE s.session_token = $1 
					AND s.status = 'active'
					AND s.expires_at > NOW()
				""", session_token)
				
				if not session_row:
					return {'valid': False, 'reason': 'Session not found or expired'}
				
				# Update last activity
				await conn.execute("""
					UPDATE crm_user_sessions 
					SET last_activity_at = NOW(), activity_count = activity_count + 1
					WHERE session_token = $1
				""", session_token)
				
				return {
					'valid': True,
					'user_id': session_row['user_id'],
					'tenant_id': session_row['tenant_id'],
					'username': session_row['username'],
					'email': session_row['email'],
					'roles': json.loads(session_row['roles'] or '[]')
				}
				
		except Exception as e:
			logger.error(f"Session verification error: {str(e)}")
			return {'valid': False, 'reason': 'Verification error'}
	
	async def setup_mfa(self, user_id: str, tenant_id: str) -> Dict[str, Any]:
		"""Setup multi-factor authentication"""
		try:
			# Generate MFA secret
			mfa_secret = self._generate_mfa_secret()
			
			async with self.db_pool.acquire() as conn:
				# Get user email for QR code
				user_row = await conn.fetchrow("""
					SELECT email FROM crm_user_accounts WHERE id = $1
				""", user_id)
				
				if not user_row:
					raise ValueError("User not found")
				
				# Generate QR code
				qr_code = self._generate_qr_code(mfa_secret, user_row['email'])
				
				# Generate backup codes
				backup_codes = [secrets.token_hex(8) for _ in range(10)]
				encrypted_backup_codes = [
					self.fernet.encrypt(code.encode()).decode() 
					for code in backup_codes
				]
				
				# Update user account with MFA secret
				await conn.execute("""
					UPDATE crm_user_accounts 
					SET mfa_secret = $1, backup_codes = $2, updated_at = NOW()
					WHERE id = $3
				""", 
				self.fernet.encrypt(mfa_secret.encode()).decode(),
				json.dumps(encrypted_backup_codes), user_id)
				
				await self._log_auth_event(
					tenant_id, user_id, AuthEventType.MFA_SETUP,
					AuthenticationMethod.MFA_TOTP, "system", "system",
					success=True
				)
				
				return {
					'secret': mfa_secret,
					'qr_code': qr_code,
					'backup_codes': backup_codes
				}
				
		except Exception as e:
			logger.error(f"MFA setup error: {str(e)}")
			raise
	
	async def verify_mfa(
		self, 
		user_id: str, 
		tenant_id: str, 
		code: str,
		challenge_id: str = None
	) -> bool:
		"""Verify MFA code"""
		try:
			async with self.db_pool.acquire() as conn:
				user_row = await conn.fetchrow("""
					SELECT mfa_secret, backup_codes FROM crm_user_accounts 
					WHERE id = $1
				""", user_id)
				
				if not user_row or not user_row['mfa_secret']:
					return False
				
				# Decrypt MFA secret
				encrypted_secret = user_row['mfa_secret']
				mfa_secret = self.fernet.decrypt(encrypted_secret.encode()).decode()
				
				# Verify TOTP code
				if self._verify_totp_code(mfa_secret, code):
					await self._log_auth_event(
						tenant_id, user_id, AuthEventType.MFA_SUCCESS,
						AuthenticationMethod.MFA_TOTP, "system", "system",
						success=True
					)
					
					# Mark challenge as verified if provided
					if challenge_id:
						await conn.execute("""
							UPDATE crm_mfa_challenges 
							SET is_verified = true, verified_at = NOW()
							WHERE id = $1
						""", challenge_id)
					
					return True
				
				# Check backup codes
				backup_codes = json.loads(user_row['backup_codes'] or '[]')
				for encrypted_code in backup_codes:
					backup_code = self.fernet.decrypt(encrypted_code.encode()).decode()
					if backup_code == code:
						# Remove used backup code
						backup_codes.remove(encrypted_code)
						await conn.execute("""
							UPDATE crm_user_accounts 
							SET backup_codes = $1, updated_at = NOW()
							WHERE id = $2
						""", json.dumps(backup_codes), user_id)
						
						await self._log_auth_event(
							tenant_id, user_id, AuthEventType.MFA_SUCCESS,
							AuthenticationMethod.MFA_TOTP, "system", "system",
							success=True
						)
						return True
				
				await self._log_auth_event(
					tenant_id, user_id, AuthEventType.MFA_FAILED,
					AuthenticationMethod.MFA_TOTP, "system", "system",
					success=False, error_message="Invalid MFA code"
				)
				return False
				
		except Exception as e:
			logger.error(f"MFA verification error: {str(e)}")
			return False
	
	async def check_permission(
		self, 
		user_id: str, 
		tenant_id: str,
		resource_type: ResourceType,
		action: PermissionAction,
		resource_id: str = None
	) -> bool:
		"""Check if user has permission for action on resource"""
		try:
			async with self.db_pool.acquire() as conn:
				# Get user roles and permissions
				user_row = await conn.fetchrow("""
					SELECT roles, permissions FROM crm_user_accounts 
					WHERE id = $1 AND tenant_id = $2
				""", user_id, tenant_id)
				
				if not user_row:
					return False
				
				user_roles = json.loads(user_row['roles'] or '[]')
				user_permissions = json.loads(user_row['permissions'] or '{}')
				
				# Check direct permissions
				permission_key = f"{resource_type.value}:{action.value}"
				if permission_key in user_permissions:
					return user_permissions[permission_key]
				
				# Check role-based permissions
				role_permissions = await conn.fetch("""
					SELECT p.resource_type, p.action, p.scope, p.conditions
					FROM crm_permissions p
					JOIN crm_roles r ON r.id = ANY(p.role_ids)
					WHERE r.role_type = ANY($1) AND r.tenant_id = $2 AND r.is_active = true
				""", user_roles, tenant_id)
				
				for perm in role_permissions:
					if (perm['resource_type'] == resource_type.value and 
						perm['action'] == action.value):
						
						# Check scope and conditions
						if self._check_permission_conditions(
							perm, user_id, tenant_id, resource_id
						):
							return True
				
				return False
				
		except Exception as e:
			logger.error(f"Permission check error: {str(e)}")
			return False
	
	def _check_permission_conditions(
		self, 
		permission: Dict[str, Any],
		user_id: str,
		tenant_id: str,
		resource_id: str = None
	) -> bool:
		"""Check permission conditions"""
		conditions = json.loads(permission.get('conditions', '{}'))
		
		# Check scope conditions
		scope = permission['scope']
		if scope == PermissionScope.PERSONAL.value and resource_id:
			# Check if user owns the resource
			return self._check_resource_ownership(user_id, resource_id)
		
		# Add more condition checks as needed
		return True
	
	def _check_resource_ownership(self, user_id: str, resource_id: str) -> bool:
		"""Check if user owns the resource"""
		# This would need to be implemented based on your resource ownership model
		# For now, return True as a placeholder
		return True
	
	async def logout_user(self, session_token: str) -> bool:
		"""Logout user and invalidate session"""
		try:
			async with self.db_pool.acquire() as conn:
				# Update session status
				result = await conn.fetchrow("""
					UPDATE crm_user_sessions 
					SET status = 'revoked', updated_at = NOW()
					WHERE session_token = $1
					RETURNING user_id, tenant_id
				""", session_token)
				
				if result:
					# Remove from Redis cache
					if self.redis_client:
						await self.redis_client.delete(f"session:{session_token}")
					
					await self._log_auth_event(
						result['tenant_id'], result['user_id'], AuthEventType.LOGOUT,
						AuthenticationMethod.JWT_TOKEN, "system", "system",
						success=True
					)
					return True
				
				return False
				
		except Exception as e:
			logger.error(f"Logout error: {str(e)}")
			return False
	
	async def get_user_sessions(self, user_id: str, tenant_id: str) -> List[Dict[str, Any]]:
		"""Get active user sessions"""
		try:
			async with self.db_pool.acquire() as conn:
				sessions = await conn.fetch("""
					SELECT id, ip_address, user_agent, device_info, created_at,
						   last_activity_at, security_score, status
					FROM crm_user_sessions 
					WHERE user_id = $1 AND tenant_id = $2 
					AND status = 'active' AND expires_at > NOW()
					ORDER BY last_activity_at DESC
				""", user_id, tenant_id)
				
				return [
					{
						'id': session['id'],
						'ip_address': session['ip_address'],
						'user_agent': session['user_agent'],
						'device_info': json.loads(session['device_info'] or '{}'),
						'created_at': session['created_at'].isoformat(),
						'last_activity_at': session['last_activity_at'].isoformat(),
						'security_score': float(session['security_score']),
						'status': session['status']
					}
					for session in sessions
				]
				
		except Exception as e:
			logger.error(f"Error getting user sessions: {str(e)}")
			return []
	
	async def revoke_session(self, session_id: str, user_id: str, tenant_id: str) -> bool:
		"""Revoke a specific user session"""
		try:
			async with self.db_pool.acquire() as conn:
				result = await conn.fetchrow("""
					UPDATE crm_user_sessions 
					SET status = 'revoked', updated_at = NOW()
					WHERE id = $1 AND user_id = $2 AND tenant_id = $3
					RETURNING session_token
				""", session_id, user_id, tenant_id)
				
				if result:
					# Remove from Redis cache
					if self.redis_client:
						await self.redis_client.delete(f"session:{result['session_token']}")
					
					return True
				
				return False
				
		except Exception as e:
			logger.error(f"Session revocation error: {str(e)}")
			return False
	
	async def change_password(
		self, 
		user_id: str, 
		tenant_id: str,
		current_password: str,
		new_password: str
	) -> bool:
		"""Change user password"""
		try:
			async with self.db_pool.acquire() as conn:
				# Verify current password
				user_row = await conn.fetchrow("""
					SELECT password_hash FROM crm_user_accounts 
					WHERE id = $1 AND tenant_id = $2
				""", user_id, tenant_id)
				
				if not user_row:
					return False
				
				if not self._verify_password(current_password, user_row['password_hash']):
					return False
				
				# Validate new password
				self._validate_password_strength(new_password)
				
				# Hash new password
				new_password_hash, salt = self._hash_password(new_password)
				
				# Update password
				await conn.execute("""
					UPDATE crm_user_accounts 
					SET password_hash = $1, salt = $2, 
						last_password_change = NOW(), updated_at = NOW()
					WHERE id = $3
				""", new_password_hash, salt, user_id)
				
				await self._log_auth_event(
					tenant_id, user_id, AuthEventType.PASSWORD_CHANGE,
					AuthenticationMethod.PASSWORD, "system", "system",
					success=True
				)
				
				return True
				
		except Exception as e:
			logger.error(f"Password change error: {str(e)}")
			return False
	
	async def health_check(self) -> Dict[str, Any]:
		"""Perform health check"""
		try:
			health_status = {
				'status': 'healthy',
				'timestamp': datetime.utcnow().isoformat(),
				'components': {}
			}
			
			# Check database connection
			try:
				async with self.db_pool.acquire() as conn:
					await conn.execute("SELECT 1")
				health_status['components']['database'] = 'healthy'
			except Exception as e:
				health_status['components']['database'] = f'unhealthy: {str(e)}'
				health_status['status'] = 'degraded'
			
			# Check Redis connection
			if self.redis_client:
				try:
					await self.redis_client.ping()
					health_status['components']['redis'] = 'healthy'
				except Exception as e:
					health_status['components']['redis'] = f'unhealthy: {str(e)}'
					health_status['status'] = 'degraded'
			else:
				health_status['components']['redis'] = 'not_configured'
			
			return health_status
			
		except Exception as e:
			return {
				'status': 'unhealthy',
				'error': str(e),
				'timestamp': datetime.utcnow().isoformat()
			}