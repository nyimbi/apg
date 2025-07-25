"""
Authentication & RBAC Service

Core authentication, authorization, and role-based access control services
with session management, multi-factor authentication, and ABAC integration.
"""

import asyncio
import logging
import hashlib
import secrets
import pyotp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from .models import ARUser, ARRole, ARPermission, ARUserSession, ARLoginAttempt
from .abac_service import ABACService, AccessRequestContext, AuthorizationDecision
from .exceptions import (
	AuthenticationError, InvalidCredentialsError, AccountLockedError,
	MFARequiredError, SessionExpiredError, PermissionDeniedError
)

logger = logging.getLogger(__name__)


@dataclass
class LoginRequest:
	"""Login request data structure"""
	email: str
	password: str
	mfa_token: Optional[str] = None
	remember_me: bool = False
	device_info: Dict[str, Any] = None
	ip_address: str = None
	user_agent: str = None
	
	def __post_init__(self):
		if self.device_info is None:
			self.device_info = {}


@dataclass
class LoginResult:
	"""Login result with session and user information"""
	success: bool
	user_id: str
	session_id: str
	access_token: str
	refresh_token: str
	expires_at: datetime
	requires_mfa: bool = False
	mfa_methods: List[str] = None
	
	def __post_init__(self):
		if self.mfa_methods is None:
			self.mfa_methods = []


class AuthenticationService:
	"""
	Core authentication service with multi-factor authentication,
	session management, and security monitoring.
	"""
	
	def __init__(self, db_session: Session, config: Dict[str, Any] = None):
		self.db = db_session
		self.config = config or {}
		self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
		
		# Security configuration
		self.max_failed_attempts = self.config.get('max_failed_attempts', 5)
		self.lockout_duration = self.config.get('lockout_duration_minutes', 30)
		self.session_timeout = self.config.get('session_timeout_minutes', 60)
		self.max_sessions = self.config.get('max_concurrent_sessions', 5)
		
		# Password policy
		self.password_policy = self.config.get('password_policy', {
			'min_length': 8,
			'require_uppercase': True,
			'require_lowercase': True,
			'require_numbers': True,
			'require_symbols': False,
			'history_count': 12
		})
		
		# Background tasks
		self._start_background_tasks()
	
	async def authenticate_user(self, login_request: LoginRequest, tenant_id: str) -> LoginResult:
		"""
		Authenticate user with credentials and optional MFA.
		
		Args:
			login_request: Login credentials and device information
			tenant_id: Tenant context
			
		Returns:
			LoginResult with session information or MFA requirement
		"""
		try:
			# Log login attempt
			self._log_login_attempt(
				login_request.email, tenant_id, 'password',
				login_request.ip_address, login_request.user_agent
			)
			
			# Get user
			user = self.db.query(ARUser).filter(
				and_(
					ARUser.email == login_request.email.lower(),
					ARUser.tenant_id == tenant_id
				)
			).first()
			
			if not user:
				self._log_login_attempt(
					login_request.email, tenant_id, 'password',
					login_request.ip_address, login_request.user_agent, 
					success=False, failure_reason='user_not_found'
				)
				raise InvalidCredentialsError("Invalid email or password")
			
			# Check account status
			if not user.is_active:
				raise AccountLockedError("Account is disabled")
			
			if user.is_account_locked():
				raise AccountLockedError(
					"Account is locked due to failed login attempts",
					details={'locked_until': user.account_locked_until.isoformat()}
				)
			
			# Verify password
			if not user.check_password(login_request.password):
				user.record_failed_login()
				self.db.commit()
				
				self._log_login_attempt(
					login_request.email, tenant_id, 'password',
					login_request.ip_address, login_request.user_agent,
					success=False, failure_reason='invalid_password', user_id=user.user_id
				)
				raise InvalidCredentialsError("Invalid email or password")
			
			# Check MFA requirement
			if user.mfa_enabled or user.require_mfa:
				if not login_request.mfa_token:
					return LoginResult(
						success=False,
						user_id=user.user_id,
						session_id="",
						access_token="",
						refresh_token="",
						expires_at=datetime.utcnow(),
						requires_mfa=True,
						mfa_methods=['totp', 'backup_code']
					)
				
				# Verify MFA token
				if not user.verify_mfa_token(login_request.mfa_token):
					self._log_login_attempt(
						login_request.email, tenant_id, 'mfa',
						login_request.ip_address, login_request.user_agent,
						success=False, failure_reason='invalid_mfa', user_id=user.user_id
					)
					raise MFARequiredError("Invalid MFA token")
			
			# Create session
			session = await self._create_user_session(
				user, login_request, 'password' if not login_request.mfa_token else 'mfa'
			)
			
			# Update user login info
			user.last_login_at = datetime.utcnow()
			user.last_login_ip = login_request.ip_address
			user.failed_login_attempts = 0
			user.account_locked_until = None
			
			self.db.commit()
			
			# Log successful login
			self._log_login_attempt(
				login_request.email, tenant_id, 'mfa' if login_request.mfa_token else 'password',
				login_request.ip_address, login_request.user_agent,
				success=True, user_id=user.user_id
			)
			
			self.logger.info(f"User {user.user_id} authenticated successfully")
			
			return LoginResult(
				success=True,
				user_id=user.user_id,
				session_id=session.session_id,
				access_token=session.jwt_token_id,
				refresh_token=session.refresh_token_id,
				expires_at=session.expires_at
			)
			
		except (InvalidCredentialsError, AccountLockedError, MFARequiredError):
			raise
		except Exception as e:
			self.logger.error(f"Authentication failed for {login_request.email}: {str(e)}")
			raise AuthenticationError(f"Authentication failed: {str(e)}")
	
	async def validate_session(self, session_id: str, extend_session: bool = True) -> Optional[ARUser]:
		"""
		Validate user session and optionally extend expiration.
		
		Args:
			session_id: Session identifier
			extend_session: Whether to extend session expiration
			
		Returns:
			User object if session is valid, None otherwise
		"""
		try:
			session = self.db.query(ARUserSession).filter(
				ARUserSession.session_id == session_id
			).first()
			
			if not session or not session.is_active():
				return None
			
			# Get user
			user = session.user
			if not user or not user.is_active:
				return None
			
			# Update session activity
			if extend_session:
				session.update_activity()
				session.extend_session(self.session_timeout)
				self.db.commit()
			
			return user
			
		except Exception as e:
			self.logger.error(f"Session validation failed for {session_id}: {str(e)}")
			return None
	
	async def logout_user(self, session_id: str, reason: str = 'user') -> bool:
		"""
		Logout user by terminating session.
		
		Args:
			session_id: Session to terminate
			reason: Logout reason
			
		Returns:
			True if logout successful
		"""
		try:
			session = self.db.query(ARUserSession).filter(
				ARUserSession.session_id == session_id
			).first()
			
			if session:
				session.terminate_session(reason)
				self.db.commit()
				
				self.logger.info(f"User {session.user_id} logged out, reason: {reason}")
				return True
			
			return False
			
		except Exception as e:
			self.logger.error(f"Logout failed for session {session_id}: {str(e)}")
			return False
	
	async def setup_mfa(self, user_id: str) -> Dict[str, Any]:
		"""
		Set up multi-factor authentication for user.
		
		Args:
			user_id: User identifier
			
		Returns:
			MFA setup information including QR code URL
		"""
		try:
			user = self.db.query(ARUser).filter(ARUser.user_id == user_id).first()
			if not user:
				raise AuthenticationError("User not found")
			
			# Generate TOTP secret and QR code
			qr_url = user.setup_mfa()
			
			self.db.commit()
			
			return {
				'qr_code_url': qr_url,
				'backup_codes': user.mfa_backup_codes,
				'secret': user.mfa_secret  # For manual entry
			}
			
		except Exception as e:
			self.logger.error(f"MFA setup failed for user {user_id}: {str(e)}")
			raise AuthenticationError(f"MFA setup failed: {str(e)}")
	
	async def change_password(self, user_id: str, current_password: str, 
							 new_password: str) -> bool:
		"""
		Change user password with validation.
		
		Args:
			user_id: User identifier
			current_password: Current password for verification
			new_password: New password
			
		Returns:
			True if password changed successfully
		"""
		try:
			user = self.db.query(ARUser).filter(ARUser.user_id == user_id).first()
			if not user:
				raise AuthenticationError("User not found")
			
			# Verify current password
			if not user.check_password(current_password):
				raise InvalidCredentialsError("Current password is incorrect")
			
			# Validate new password
			self._validate_password(new_password, user)
			
			# Check password history
			if user.is_password_in_history(new_password):
				raise AuthenticationError("Password was used recently")
			
			# Set new password
			user.set_password(new_password)
			user.require_password_change = False
			
			self.db.commit()
			
			self.logger.info(f"Password changed for user {user_id}")
			return True
			
		except Exception as e:
			self.logger.error(f"Password change failed for user {user_id}: {str(e)}")
			raise
	
	async def _create_user_session(self, user: ARUser, login_request: LoginRequest, 
								  login_method: str) -> ARUserSession:
		"""Create new user session."""
		
		# Check maximum concurrent sessions
		active_sessions = self.db.query(ARUserSession).filter(
			and_(
				ARUserSession.user_id == user.user_id,
				ARUserSession.logout_at.is_(None),
				ARUserSession.expires_at > datetime.utcnow()
			)
		).count()
		
		if active_sessions >= user.max_concurrent_sessions:
			# Terminate oldest session
			oldest_session = self.db.query(ARUserSession).filter(
				and_(
					ARUserSession.user_id == user.user_id,
					ARUserSession.logout_at.is_(None),
					ARUserSession.expires_at > datetime.utcnow()
				)
			).order_by(ARUserSession.last_activity_at.asc()).first()
			
			if oldest_session:
				oldest_session.terminate_session('max_sessions_exceeded')
		
		# Create new session
		session = ARUserSession(
			session_id=secrets.token_urlsafe(32),
			user_id=user.user_id,
			tenant_id=user.tenant_id,
			jwt_token_id=secrets.token_urlsafe(32),
			refresh_token_id=secrets.token_urlsafe(32),
			device_fingerprint=self._generate_device_fingerprint(login_request.device_info),
			user_agent=login_request.user_agent,
			ip_address=login_request.ip_address,
			login_method=login_method,
			expires_at=datetime.utcnow() + timedelta(minutes=self.session_timeout)
		)
		
		# Extract device information
		if login_request.device_info:
			session.device_type = login_request.device_info.get('device_type', 'unknown')
			session.browser_name = login_request.device_info.get('browser', 'unknown')
			session.os_name = login_request.device_info.get('os', 'unknown')
		
		# Calculate anomaly score
		session.calculate_anomaly_score()
		
		self.db.add(session)
		return session
	
	def _validate_password(self, password: str, user: ARUser = None) -> None:
		"""Validate password against policy."""
		policy = self.password_policy
		
		if len(password) < policy.get('min_length', 8):
			raise AuthenticationError(f"Password must be at least {policy['min_length']} characters")
		
		if policy.get('require_uppercase', True) and not any(c.isupper() for c in password):
			raise AuthenticationError("Password must contain at least one uppercase letter")
		
		if policy.get('require_lowercase', True) and not any(c.islower() for c in password):
			raise AuthenticationError("Password must contain at least one lowercase letter")
		
		if policy.get('require_numbers', True) and not any(c.isdigit() for c in password):
			raise AuthenticationError("Password must contain at least one number")
		
		if policy.get('require_symbols', False):
			symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?"
			if not any(c in symbols for c in password):
				raise AuthenticationError("Password must contain at least one symbol")
	
	def _generate_device_fingerprint(self, device_info: Dict[str, Any]) -> str:
		"""Generate device fingerprint from device information."""
		fingerprint_data = {
			'user_agent': device_info.get('user_agent', ''),
			'screen_resolution': device_info.get('screen_resolution', ''),
			'timezone': device_info.get('timezone', ''),
			'language': device_info.get('language', ''),
			'platform': device_info.get('platform', '')
		}
		
		fingerprint_string = json.dumps(fingerprint_data, sort_keys=True)
		return hashlib.sha256(fingerprint_string.encode()).hexdigest()
	
	def _log_login_attempt(self, email: str, tenant_id: str, method: str,
						  ip_address: str, user_agent: str, success: bool = True,
						  failure_reason: str = None, user_id: str = None) -> None:
		"""Log login attempt for security monitoring."""
		try:
			attempt = ARLoginAttempt(
				user_id=user_id,
				tenant_id=tenant_id,
				email=email.lower(),
				attempt_method=method,
				success=success,
				failure_reason=failure_reason,
				ip_address=ip_address,
				user_agent=user_agent
			)
			
			self.db.add(attempt)
			self.db.commit()
			
		except Exception as e:
			self.logger.error(f"Failed to log login attempt: {str(e)}")
	
	def _start_background_tasks(self):
		"""Start background maintenance tasks."""
		asyncio.create_task(self._cleanup_expired_sessions())
		asyncio.create_task(self._monitor_suspicious_activity())
		self.logger.info("Started authentication background tasks")
	
	async def _cleanup_expired_sessions(self):
		"""Clean up expired sessions periodically."""
		while True:
			try:
				# Clean up expired sessions every 5 minutes
				await asyncio.sleep(300)
				
				expired_sessions = self.db.query(ARUserSession).filter(
					and_(
						ARUserSession.expires_at < datetime.utcnow(),
						ARUserSession.logout_at.is_(None)
					)
				).all()
				
				for session in expired_sessions:
					session.terminate_session('expired')
				
				self.db.commit()
				
				if expired_sessions:
					self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
					
			except Exception as e:
				self.logger.error(f"Session cleanup error: {str(e)}")
	
	async def _monitor_suspicious_activity(self):
		"""Monitor for suspicious authentication activity."""
		while True:
			try:
				# Monitor suspicious activity every minute
				await asyncio.sleep(60)
				
				# Check for brute force attempts
				recent_failures = self.db.query(ARLoginAttempt).filter(
					and_(
						ARLoginAttempt.success == False,
						ARLoginAttempt.created_on > datetime.utcnow() - timedelta(minutes=5)
					)
				).all()
				
				# Group by IP address
				ip_failures = {}
				for attempt in recent_failures:
					ip_failures[attempt.ip_address] = ip_failures.get(attempt.ip_address, 0) + 1
				
				# Flag suspicious IPs
				for ip_address, count in ip_failures.items():
					if count >= 10:  # More than 10 failures in 5 minutes
						self.logger.warning(f"Suspicious activity detected from IP {ip_address}: {count} failed attempts")
						# Could implement IP blocking here
				
			except Exception as e:
				self.logger.error(f"Suspicious activity monitoring error: {str(e)}")


class AuthorizationService:
	"""
	Authorization service with RBAC and ABAC integration.
	"""
	
	def __init__(self, db_session: Session, abac_service: ABACService = None):
		self.db = db_session
		self.abac_service = abac_service or ABACService(db_session)
		self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
	
	async def check_permission(self, user_id: str, resource_type: str, 
							  resource_id: str, action: str, tenant_id: str,
							  context: Dict[str, Any] = None) -> bool:
		"""
		Check if user has permission to perform action on resource.
		
		Args:
			user_id: User identifier
			resource_type: Type of resource
			resource_id: Specific resource identifier
			action: Action to perform
			tenant_id: Tenant context
			context: Additional context for ABAC
			
		Returns:
			True if access is permitted
		"""
		try:
			# Use ABAC service for comprehensive authorization
			decision = self.abac_service.authorize(
				subject_id=user_id,
				resource_type=resource_type,
				resource_id=resource_id,
				action=action,
				tenant_id=tenant_id,
				context=context
			)
			
			return decision.is_permitted()
			
		except Exception as e:
			self.logger.error(f"Permission check failed: {str(e)}")
			return False  # Secure default - deny on error
	
	async def get_user_permissions(self, user_id: str, tenant_id: str) -> List[Dict[str, Any]]:
		"""Get all effective permissions for a user."""
		try:
			user = self.db.query(ARUser).filter(
				and_(
					ARUser.user_id == user_id,
					ARUser.tenant_id == tenant_id
				)
			).first()
			
			if not user:
				return []
			
			return user.get_effective_permissions()
			
		except Exception as e:
			self.logger.error(f"Failed to get user permissions: {str(e)}")
			return []


# Capability composition functions
def get_auth_service(db_session: Session, config: Dict[str, Any] = None) -> AuthenticationService:
	"""Get authentication service instance."""
	return AuthenticationService(db_session, config)


def get_authorization_service(db_session: Session, abac_service: ABACService = None) -> AuthorizationService:
	"""Get authorization service instance."""
	return AuthorizationService(db_session, abac_service)