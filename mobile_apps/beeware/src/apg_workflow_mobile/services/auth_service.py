"""
Authentication Service for APG Workflow Mobile

Handles user authentication, token management, and biometric authentication.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
import keyring
from pathlib import Path

from ..models.user import User, AuthToken, LoginCredentials, AuthResponse, UserRole
from ..models.api_response import APIResponse
from ..services.api_service import APIService
from ..services.biometric_service import BiometricService
from ..utils.constants import APP_NAME, KEYRING_SERVICE_NAME
from ..utils.exceptions import AuthenticationException, ValidationException
from ..utils.security import encrypt_data, decrypt_data, generate_device_id


class AuthService:
	"""Authentication service for user login/logout and token management"""
	
	def __init__(self, api_service: APIService, app=None):
		self.api_service = api_service
		self.app = app
		self.logger = logging.getLogger(__name__)
		
		# Current authentication state
		self.current_user: Optional[User] = None
		self.current_token: Optional[AuthToken] = None
		self.is_authenticated = False
		
		# Biometric service
		self.biometric_service: Optional[BiometricService] = None
		
		# Device identification
		self.device_id = generate_device_id()
		
		self.logger.info("Auth Service initialized")
	
	async def initialize_biometric_service(self):
		"""Initialize biometric service"""
		try:
			self.biometric_service = BiometricService()
			await self.biometric_service.initialize()
			self.logger.info("Biometric service initialized")
		except Exception as e:
			self.logger.warning(f"Failed to initialize biometric service: {e}")
	
	async def login(self, credentials: LoginCredentials) -> AuthResponse:
		"""Authenticate user with username/password"""
		try:
			self.logger.info(f"Attempting login for user: {credentials.username}")
			
			# Validate credentials
			self._validate_login_credentials(credentials)
			
			# Prepare login request
			login_data = {
				"username": credentials.username,
				"password": credentials.password,
				"tenant_id": credentials.tenant_id,
				"device_id": self.device_id,
				"remember_me": credentials.remember_me
			}
			
			# Make login request
			response = await self.api_service.post("/auth/login", login_data)
			
			if not response.success:
				raise AuthenticationException(
					response.message or "Login failed"
				)
			
			# Parse authentication response
			auth_response = self._parse_auth_response(response.data)
			
			# Store authentication state
			await self._store_auth_state(auth_response)
			
			self.logger.info("Login successful")
			return auth_response
			
		except ValidationException:
			raise
		except AuthenticationException:
			raise
		except Exception as e:
			self.logger.error(f"Login error: {e}")
			raise AuthenticationException(f"Login failed: {e}")
	
	async def login_with_biometric(self) -> AuthResponse:
		"""Authenticate user with biometric data"""
		try:
			if not self.biometric_service:
				raise AuthenticationException("Biometric authentication not available")
			
			self.logger.info("Attempting biometric login")
			
			# Get stored biometric template
			stored_template = await self._get_stored_biometric_template()
			if not stored_template:
				raise AuthenticationException("No biometric template found")
			
			# Authenticate with biometric
			biometric_result = await self.biometric_service.authenticate()
			if not biometric_result.success:
				raise AuthenticationException(
					biometric_result.error or "Biometric authentication failed"
				)
			
			# Prepare biometric login request
			login_data = {
				"device_id": self.device_id,
				"biometric_signature": biometric_result.signature,
				"biometric_template": stored_template
			}
			
			# Make biometric login request
			response = await self.api_service.post("/auth/biometric-login", login_data)
			
			if not response.success:
				raise AuthenticationException(
					response.message or "Biometric login failed"
				)
			
			# Parse authentication response
			auth_response = self._parse_auth_response(response.data)
			
			# Store authentication state
			await self._store_auth_state(auth_response)
			
			self.logger.info("Biometric login successful")
			return auth_response
			
		except AuthenticationException:
			raise
		except Exception as e:
			self.logger.error(f"Biometric login error: {e}")
			raise AuthenticationException(f"Biometric login failed: {e}")
	
	async def logout(self):
		"""Logout current user"""
		try:
			self.logger.info("Logging out user")
			
			if self.current_token:
				# Notify server of logout
				try:
					await self.api_service.post("/auth/logout", {
						"token": self.current_token.access_token
					})
				except Exception as e:
					self.logger.warning(f"Server logout notification failed: {e}")
			
			# Clear authentication state
			await self._clear_auth_state()
			
			self.logger.info("Logout completed")
			
		except Exception as e:
			self.logger.error(f"Logout error: {e}")
			# Always clear local state even if server logout fails
			await self._clear_auth_state()
	
	async def refresh_token(self) -> bool:
		"""Refresh access token using refresh token"""
		try:
			if not self.current_token or not self.current_token.refresh_token:
				return False
			
			self.logger.info("Refreshing access token")
			
			refresh_data = {
				"refresh_token": self.current_token.refresh_token,
				"device_id": self.device_id
			}
			
			response = await self.api_service.post("/auth/refresh", refresh_data)
			
			if not response.success:
				self.logger.error("Token refresh failed")
				await self._clear_auth_state()
				return False
			
			# Update token
			token_data = response.data
			self.current_token = AuthToken(
				access_token=token_data["access_token"],
				refresh_token=token_data.get("refresh_token", self.current_token.refresh_token),
				expires_in=token_data.get("expires_in", 3600),
				token_type=token_data.get("token_type", "Bearer")
			)
			
			# Update API service token
			self.api_service.set_auth_token(
				self.current_token.access_token,
				self.current_token.refresh_token,
				self.current_token.expires_at
			)
			
			# Store updated token
			await self._store_auth_token(self.current_token)
			
			self.logger.info("Token refreshed successfully")
			return True
			
		except Exception as e:
			self.logger.error(f"Token refresh error: {e}")
			await self._clear_auth_state()
			return False
	
	async def check_stored_auth(self) -> bool:
		"""Check if there's stored authentication data"""
		try:
			# Try to load stored authentication state
			user_data = await self._load_stored_user()
			token_data = await self._load_stored_token()
			
			if not user_data or not token_data:
				return False
			
			# Recreate user and token objects
			self.current_user = User.from_dict(user_data)
			self.current_token = AuthToken(**token_data)
			
			# Check if token is expired
			if self.current_token.is_expired:
				# Try to refresh
				if not await self.refresh_token():
					return False
			
			# Set authentication state
			self.is_authenticated = True
			
			# Update API service
			self.api_service.set_auth_token(
				self.current_token.access_token,
				self.current_token.refresh_token,
				self.current_token.expires_at
			)
			
			self.logger.info("Restored authentication from storage")
			return True
			
		except Exception as e:
			self.logger.error(f"Error checking stored auth: {e}")
			await self._clear_auth_state()
			return False
	
	async def update_user_profile(self, updates: Dict[str, Any]) -> User:
		"""Update user profile"""
		try:
			if not self.is_authenticated or not self.current_user:
				raise AuthenticationException("User not authenticated")
			
			self.logger.info("Updating user profile")
			
			response = await self.api_service.put(
				f"/users/{self.current_user.id}",
				updates
			)
			
			if not response.success:
				raise Exception(response.message or "Profile update failed")
			
			# Update current user
			updated_user_data = response.data
			self.current_user = User.from_dict(updated_user_data)
			
			# Store updated user
			await self._store_user_data(self.current_user)
			
			self.logger.info("User profile updated successfully")
			return self.current_user
			
		except AuthenticationException:
			raise
		except Exception as e:
			self.logger.error(f"Profile update error: {e}")
			raise Exception(f"Failed to update profile: {e}")
	
	async def setup_biometric_auth(self) -> bool:
		"""Set up biometric authentication for current user"""
		try:
			if not self.is_authenticated or not self.current_user:
				raise AuthenticationException("User not authenticated")
			
			if not self.biometric_service:
				await self.initialize_biometric_service()
			
			if not self.biometric_service:
				raise Exception("Biometric service not available")
			
			self.logger.info("Setting up biometric authentication")
			
			# Enroll biometric
			enrollment_result = await self.biometric_service.enroll_user(
				self.current_user.id
			)
			
			if not enrollment_result.success:
				raise Exception(
					enrollment_result.error or "Biometric enrollment failed"
				)
			
			# Send template to server
			biometric_data = {
				"user_id": self.current_user.id,
				"biometric_template": enrollment_result.template,
				"biometric_method": enrollment_result.method,
				"device_id": self.device_id
			}
			
			response = await self.api_service.post("/auth/setup-biometric", biometric_data)
			
			if not response.success:
				raise Exception(response.message or "Biometric setup failed")
			
			# Store template locally
			await self._store_biometric_template(enrollment_result.template)
			
			# Update user biometric config
			self.current_user.enable_biometric(enrollment_result.method)
			await self._store_user_data(self.current_user)
			
			self.logger.info("Biometric authentication setup successful")
			return True
			
		except AuthenticationException:
			raise
		except Exception as e:
			self.logger.error(f"Biometric setup error: {e}")
			raise Exception(f"Failed to setup biometric authentication: {e}")
	
	async def disable_biometric_auth(self) -> bool:
		"""Disable biometric authentication"""
		try:
			if not self.is_authenticated or not self.current_user:
				raise AuthenticationException("User not authenticated")
			
			self.logger.info("Disabling biometric authentication")
			
			# Notify server
			response = await self.api_service.post("/auth/disable-biometric", {
				"user_id": self.current_user.id,
				"device_id": self.device_id
			})
			
			if not response.success:
				self.logger.warning("Server biometric disable failed")
			
			# Clear local biometric data
			await self._clear_biometric_template()
			
			# Update user config
			self.current_user.disable_biometric()
			await self._store_user_data(self.current_user)
			
			self.logger.info("Biometric authentication disabled")
			return True
			
		except Exception as e:
			self.logger.error(f"Biometric disable error: {e}")
			return False
	
	def _validate_login_credentials(self, credentials: LoginCredentials):
		"""Validate login credentials"""
		if not credentials.username or len(credentials.username.strip()) < 3:
			raise ValidationException("Username must be at least 3 characters")
		
		if not credentials.password or len(credentials.password) < 6:
			raise ValidationException("Password must be at least 6 characters")
		
		if not credentials.tenant_id or len(credentials.tenant_id.strip()) == 0:
			raise ValidationException("Tenant ID is required")
	
	def _parse_auth_response(self, data: Dict[str, Any]) -> AuthResponse:
		"""Parse authentication response data"""
		try:
			user_data = data["user"]
			token_data = data["token"]
			
			user = User.from_dict(user_data)
			token = AuthToken(
				access_token=token_data["access_token"],
				refresh_token=token_data["refresh_token"],
				expires_in=token_data.get("expires_in", 3600),
				token_type=token_data.get("token_type", "Bearer")
			)
			
			permissions = data.get("permissions", [])
			message = data.get("message")
			
			return AuthResponse(
				user=user,
				token=token,
				permissions=permissions,
				message=message
			)
			
		except KeyError as e:
			raise AuthenticationException(f"Invalid authentication response: missing {e}")
		except Exception as e:
			raise AuthenticationException(f"Failed to parse authentication response: {e}")
	
	async def _store_auth_state(self, auth_response: AuthResponse):
		"""Store authentication state"""
		self.current_user = auth_response.user
		self.current_token = auth_response.token
		self.is_authenticated = True
		
		# Update API service
		self.api_service.set_auth_token(
			self.current_token.access_token,
			self.current_token.refresh_token,
			self.current_token.expires_at
		)
		
		# Store to secure storage
		await self._store_user_data(self.current_user)
		await self._store_auth_token(self.current_token)
		
		# Update app state if available
		if self.app and hasattr(self.app, 'app_state'):
			self.app.app_state.set_current_user(self.current_user)
			self.app.app_state.set_authenticated(True)
	
	async def _clear_auth_state(self):
		"""Clear authentication state"""
		self.current_user = None
		self.current_token = None
		self.is_authenticated = False
		
		# Clear API service
		self.api_service.clear_auth_token()
		
		# Clear secure storage
		await self._clear_stored_user()
		await self._clear_stored_token()
		
		# Update app state if available
		if self.app and hasattr(self.app, 'app_state'):
			self.app.app_state.clear_current_user()
			self.app.app_state.set_authenticated(False)
	
	async def _store_user_data(self, user: User):
		"""Store user data securely"""
		try:
			user_json = json.dumps(user.to_dict())
			encrypted_data = encrypt_data(user_json, self.device_id)
			keyring.set_password(KEYRING_SERVICE_NAME, "user_data", encrypted_data)
		except Exception as e:
			self.logger.error(f"Failed to store user data: {e}")
	
	async def _load_stored_user(self) -> Optional[Dict[str, Any]]:
		"""Load stored user data"""
		try:
			encrypted_data = keyring.get_password(KEYRING_SERVICE_NAME, "user_data")
			if not encrypted_data:
				return None
			
			user_json = decrypt_data(encrypted_data, self.device_id)
			return json.loads(user_json)
		except Exception as e:
			self.logger.error(f"Failed to load user data: {e}")
			return None
	
	async def _clear_stored_user(self):
		"""Clear stored user data"""
		try:
			keyring.delete_password(KEYRING_SERVICE_NAME, "user_data")
		except Exception as e:
			self.logger.debug(f"No user data to clear: {e}")
	
	async def _store_auth_token(self, token: AuthToken):
		"""Store authentication token securely"""
		try:
			token_data = {
				"access_token": token.access_token,
				"refresh_token": token.refresh_token,
				"expires_at": token.expires_at,
				"token_type": token.token_type
			}
			token_json = json.dumps(token_data)
			encrypted_data = encrypt_data(token_json, self.device_id)
			keyring.set_password(KEYRING_SERVICE_NAME, "auth_token", encrypted_data)
		except Exception as e:
			self.logger.error(f"Failed to store auth token: {e}")
	
	async def _load_stored_token(self) -> Optional[Dict[str, Any]]:
		"""Load stored authentication token"""
		try:
			encrypted_data = keyring.get_password(KEYRING_SERVICE_NAME, "auth_token")
			if not encrypted_data:
				return None
			
			token_json = decrypt_data(encrypted_data, self.device_id)
			return json.loads(token_json)
		except Exception as e:
			self.logger.error(f"Failed to load auth token: {e}")
			return None
	
	async def _clear_stored_token(self):
		"""Clear stored authentication token"""
		try:
			keyring.delete_password(KEYRING_SERVICE_NAME, "auth_token")
		except Exception as e:
			self.logger.debug(f"No auth token to clear: {e}")
	
	async def _store_biometric_template(self, template: str):
		"""Store biometric template securely"""
		try:
			encrypted_template = encrypt_data(template, self.device_id)
			keyring.set_password(KEYRING_SERVICE_NAME, "biometric_template", encrypted_template)
		except Exception as e:
			self.logger.error(f"Failed to store biometric template: {e}")
	
	async def _get_stored_biometric_template(self) -> Optional[str]:
		"""Get stored biometric template"""
		try:
			encrypted_template = keyring.get_password(KEYRING_SERVICE_NAME, "biometric_template")
			if not encrypted_template:
				return None
			
			return decrypt_data(encrypted_template, self.device_id)
		except Exception as e:
			self.logger.error(f"Failed to get biometric template: {e}")
			return None
	
	async def _clear_biometric_template(self):
		"""Clear stored biometric template"""
		try:
			keyring.delete_password(KEYRING_SERVICE_NAME, "biometric_template")
		except Exception as e:
			self.logger.debug(f"No biometric template to clear: {e}")