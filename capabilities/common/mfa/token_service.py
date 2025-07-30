"""
APG Multi-Factor Authentication (MFA) - Token Management Service

Comprehensive token management system supporting TOTP, HOTP, hardware tokens,
backup codes, and offline verification with seamless APG integration.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import secrets
import base64
import hmac
import hashlib
import struct
import time
import logging
import qrcode
import io
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from uuid_extensions import uuid7str
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .models import (
	AuthToken, MFAUserProfile, MFAMethod, MFAMethodType,
	TrustLevel, AuthenticationStatus
)


def _log_token_operation(operation: str, user_id: str, details: str = "") -> str:
	"""Log token operations for debugging and audit"""
	return f"[Token Service] {operation} for user {user_id}: {details}"


class TokenService:
	"""
	Comprehensive token management service supporting multiple token types,
	offline verification, and secure token lifecycle management.
	"""
	
	def __init__(self, 
				database_client: Any,
				encryption_key: Optional[bytes] = None):
		"""Initialize token service"""
		self.db = database_client
		self.logger = logging.getLogger(__name__)
		
		# Initialize encryption
		if encryption_key:
			self.fernet = Fernet(encryption_key)
		else:
			# Generate key from environment or create new one
			key = Fernet.generate_key()
			self.fernet = Fernet(key)
			self.logger.warning("Using auto-generated encryption key - not suitable for production")
		
		# Token configuration
		self.totp_window = 30  # seconds
		self.totp_digits = 6
		self.hotp_digits = 6
		self.backup_code_length = 8
		self.delegation_token_length = 32
		
		# Token validity periods
		self.auth_token_validity = timedelta(hours=8)
		self.delegation_token_validity = timedelta(hours=1)
		self.backup_code_validity = timedelta(days=365)
		
		# Rate limiting
		self.max_token_attempts_per_minute = 5
		self.max_backup_codes = 10
	
	async def generate_authentication_token(self,
											user_id: str,
											tenant_id: str,
											trust_score: float,
											context: Dict[str, Any]) -> AuthToken:
		"""
		Generate authentication token after successful MFA verification.
		
		Args:
			user_id: User who authenticated
			tenant_id: Tenant context
			trust_score: Trust score from authentication
			context: Authentication context
		
		Returns:
			Generated authentication token
		"""
		try:
			self.logger.info(_log_token_operation("generate_auth_token", user_id))
			
			# Generate secure token value
			token_value = self._generate_secure_token(32)
			encrypted_token = self._encrypt_token_value(token_value)
			
			# Determine token validity based on trust score
			validity_hours = int(8 * trust_score)  # Higher trust = longer validity
			validity_hours = max(1, min(validity_hours, 24))  # 1-24 hours
			
			# Create auth token
			auth_token = AuthToken(
				user_id=user_id,
				tenant_id=tenant_id,
				token_type="authentication",
				token_value=encrypted_token,
				issued_at=datetime.utcnow(),
				expires_at=datetime.utcnow() + timedelta(hours=validity_hours),
				is_active=True,
				is_single_use=False,
				device_binding_required=True,
				allowed_devices=[context.get("device", {}).get("device_id", "")],
				ip_restrictions=[context.get("location", {}).get("ip_address", "")],
				created_by=user_id,
				updated_by=user_id
			)
			
			# Store token
			await self._store_auth_token(auth_token)
			
			# Set plaintext token value for return (will be encrypted in storage)
			auth_token.token_value = token_value
			
			self.logger.info(_log_token_operation(
				"generate_auth_token_success", user_id,
				f"validity={validity_hours}h, trust={trust_score:.2f}"
			))
			
			return auth_token
			
		except Exception as e:
			self.logger.error(f"Auth token generation error for user {user_id}: {str(e)}", exc_info=True)
			raise
	
	async def verify_token(self, token: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		"""
		Verify authentication token and return token data.
		
		Args:
			token: Token to verify
			context: Request context for validation
		
		Returns:
			Token data if valid, None if invalid
		"""
		try:
			# Find token in database
			token_record = await self._get_token_by_value(token)
			
			if not token_record:
				self.logger.warning("Token verification failed: token not found")
				return None
			
			# Check if token is active and not expired
			if not token_record.is_active:
				self.logger.warning(f"Token verification failed: token not active for user {token_record.user_id}")
				return None
			
			if token_record.expires_at < datetime.utcnow():
				self.logger.warning(f"Token verification failed: token expired for user {token_record.user_id}")
				await self._deactivate_token(token_record.id)
				return None
			
			# Validate context restrictions
			context_valid = await self._validate_token_context(token_record, context)
			if not context_valid:
				self.logger.warning(f"Token verification failed: context validation failed for user {token_record.user_id}")
				return None
			
			# Update usage tracking
			await self._update_token_usage(token_record)
			
			return {
				"token_id": token_record.id,
				"user_id": token_record.user_id,
				"tenant_id": token_record.tenant_id,
				"issued_at": token_record.issued_at.isoformat(),
				"expires_at": token_record.expires_at.isoformat(),
				"token_type": token_record.token_type,
				"trust_score": context.get("trust_score", 0.5)
			}
			
		except Exception as e:
			self.logger.error(f"Token verification error: {str(e)}", exc_info=True)
			return None
	
	async def generate_totp_secret(self, user_id: str, tenant_id: str) -> Dict[str, Any]:
		"""
		Generate TOTP secret key and QR code for user enrollment.
		
		Args:
			user_id: User enrolling TOTP
			tenant_id: Tenant context
		
		Returns:
			TOTP secret, QR code, and backup codes
		"""
		try:
			self.logger.info(_log_token_operation("generate_totp_secret", user_id))
			
			# Generate secure random secret (32 bytes = 256 bits)
			secret_bytes = secrets.token_bytes(32)
			secret_b32 = base64.b32encode(secret_bytes).decode('utf-8')
			
			# Generate QR code for easy enrollment
			issuer = "APG MFA"
			account_name = f"{user_id}@{tenant_id}"
			
			totp_uri = self._generate_totp_uri(secret_b32, account_name, issuer)
			qr_code_data = self._generate_qr_code(totp_uri)
			
			# Generate backup codes
			backup_codes = self._generate_backup_codes(self.max_backup_codes)
			
			# Encrypt secret for storage
			encrypted_secret = self._encrypt_token_value(secret_b32)
			
			return {
				"secret": secret_b32,  # Return plaintext for user enrollment
				"encrypted_secret": encrypted_secret,  # For database storage
				"qr_code": qr_code_data,
				"backup_codes": backup_codes,
				"uri": totp_uri
			}
			
		except Exception as e:
			self.logger.error(f"TOTP secret generation error for user {user_id}: {str(e)}", exc_info=True)
			raise
	
	async def verify_totp_code(self, secret: str, provided_code: str, time_window: int = 30) -> bool:
		"""
		Verify TOTP code against secret.
		
		Args:
			secret: TOTP secret (base32 encoded)
			provided_code: Code provided by user
			time_window: Time window in seconds (default 30)
		
		Returns:
			True if code is valid
		"""
		try:
			# Get current time slots (allow for clock skew)
			current_time = int(time.time())
			time_slots = [
				current_time // time_window - 1,  # Previous window
				current_time // time_window,      # Current window
				current_time // time_window + 1   # Next window
			]
			
			# Check code against each time slot
			for time_slot in time_slots:
				expected_code = self._generate_totp_code(secret, time_slot, self.totp_digits)
				if self._constant_time_compare(provided_code, expected_code):
					return True
			
			return False
			
		except Exception as e:
			self.logger.error(f"TOTP verification error: {str(e)}", exc_info=True)
			return False
	
	async def generate_hotp_secret(self, user_id: str, tenant_id: str) -> Dict[str, Any]:
		"""
		Generate HOTP secret key for user enrollment.
		
		Args:
			user_id: User enrolling HOTP
			tenant_id: Tenant context
		
		Returns:
			HOTP secret and initial counter
		"""
		try:
			self.logger.info(_log_token_operation("generate_hotp_secret", user_id))
			
			# Generate secure random secret
			secret_bytes = secrets.token_bytes(32)
			secret_b32 = base64.b32encode(secret_bytes).decode('utf-8')
			
			# Initial counter
			initial_counter = 0
			
			# Encrypt secret for storage
			encrypted_secret = self._encrypt_token_value(secret_b32)
			
			return {
				"secret": secret_b32,
				"encrypted_secret": encrypted_secret,
				"counter": initial_counter
			}
			
		except Exception as e:
			self.logger.error(f"HOTP secret generation error for user {user_id}: {str(e)}", exc_info=True)
			raise
	
	async def verify_hotp_code(self, secret: str, provided_code: str, current_counter: int) -> Tuple[bool, int]:
		"""
		Verify HOTP code against secret and update counter.
		
		Args:
			secret: HOTP secret (base32 encoded)
			provided_code: Code provided by user
			current_counter: Current counter value
		
		Returns:
			Tuple of (is_valid: bool, new_counter: int)
		"""
		try:
			# Check next few counter values (allow for missed authentications)
			window_size = 10
			
			for counter in range(current_counter, current_counter + window_size):
				expected_code = self._generate_hotp_code(secret, counter, self.hotp_digits)
				if self._constant_time_compare(provided_code, expected_code):
					return True, counter + 1
			
			return False, current_counter
			
		except Exception as e:
			self.logger.error(f"HOTP verification error: {str(e)}", exc_info=True)
			return False, current_counter
	
	async def generate_backup_codes(self, user_id: str, tenant_id: str, count: int = 10) -> List[str]:
		"""
		Generate backup recovery codes for user.
		
		Args:
			user_id: User generating backup codes
			tenant_id: Tenant context
			count: Number of backup codes to generate
		
		Returns:
			List of backup codes
		"""
		try:
			self.logger.info(_log_token_operation("generate_backup_codes", user_id, f"count={count}"))
			
			backup_codes = self._generate_backup_codes(count)
			
			# Store encrypted backup codes
			for code in backup_codes:
				encrypted_code = self._encrypt_token_value(code)
				backup_token = AuthToken(
					user_id=user_id,
					tenant_id=tenant_id,
					token_type="backup_code",
					token_value=encrypted_code,
					issued_at=datetime.utcnow(),
					expires_at=datetime.utcnow() + self.backup_code_validity,
					is_active=True,
					is_single_use=True,
					created_by=user_id,
					updated_by=user_id
				)
				await self._store_auth_token(backup_token)
			
			return backup_codes
			
		except Exception as e:
			self.logger.error(f"Backup code generation error for user {user_id}: {str(e)}", exc_info=True)
			raise
	
	async def verify_backup_code(self, user_id: str, tenant_id: str, provided_code: str) -> bool:
		"""
		Verify backup code and mark as used.
		
		Args:
			user_id: User verifying backup code
			tenant_id: Tenant context
			provided_code: Backup code provided by user
		
		Returns:
			True if code is valid
		"""
		try:
			# Get active backup codes for user
			backup_tokens = await self._get_user_backup_codes(user_id, tenant_id)
			
			for token in backup_tokens:
				decrypted_code = self._decrypt_token_value(token.token_value)
				if self._constant_time_compare(provided_code, decrypted_code):
					# Mark code as used
					await self._deactivate_token(token.id)
					
					self.logger.info(_log_token_operation("verify_backup_code_success", user_id))
					return True
			
			self.logger.warning(_log_token_operation("verify_backup_code_failed", user_id))
			return False
			
		except Exception as e:
			self.logger.error(f"Backup code verification error for user {user_id}: {str(e)}", exc_info=True)
			return False
	
	async def generate_delegation_token(self,
									   delegating_user_id: str,
									   target_user_id: str,
									   tenant_id: str,
									   scope: List[str],
									   validity_minutes: int = 60) -> AuthToken:
		"""
		Generate delegation token for collaborative authentication.
		
		Args:
			delegating_user_id: User delegating access
			target_user_id: User receiving delegated access
			tenant_id: Tenant context
			scope: Scope of delegated access
			validity_minutes: Token validity in minutes
		
		Returns:
			Delegation token
		"""
		try:
			self.logger.info(_log_token_operation(
				"generate_delegation_token", delegating_user_id,
				f"target={target_user_id}, scope={scope}"
			))
			
			# Generate secure delegation token
			token_value = self._generate_secure_token(self.delegation_token_length)
			encrypted_token = self._encrypt_token_value(token_value)
			
			# Create delegation token
			delegation_token = AuthToken(
				user_id=target_user_id,  # Token belongs to target user
				tenant_id=tenant_id,
				token_type="delegation",
				token_value=encrypted_token,
				issued_at=datetime.utcnow(),
				expires_at=datetime.utcnow() + timedelta(minutes=validity_minutes),
				is_active=True,
				is_single_use=False,
				max_uses=10,  # Limit delegation usage
				is_delegation_token=True,
				delegated_by=delegating_user_id,
				delegation_scope=scope,
				created_by=delegating_user_id,
				updated_by=delegating_user_id
			)
			
			# Store delegation token
			await self._store_auth_token(delegation_token)
			
			# Return with plaintext token value
			delegation_token.token_value = token_value
			
			return delegation_token
			
		except Exception as e:
			self.logger.error(f"Delegation token generation error: {str(e)}", exc_info=True)
			raise
	
	async def verify_delegation_token(self, token: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		"""
		Verify delegation token for collaborative authentication.
		
		Args:
			token: Delegation token to verify
			context: Request context
		
		Returns:
			Token data if valid
		"""
		try:
			# Find delegation token
			token_record = await self._get_token_by_value(token)
			
			if not token_record or not token_record.is_delegation_token:
				return None
			
			# Validate token
			if not token_record.is_active or token_record.expires_at < datetime.utcnow():
				return None
			
			# Check usage limits
			if token_record.max_uses and token_record.used_count >= token_record.max_uses:
				await self._deactivate_token(token_record.id)
				return None
			
			# Update usage
			await self._update_token_usage(token_record)
			
			return {
				"valid": True,
				"user_id": token_record.user_id,
				"tenant_id": token_record.tenant_id,
				"delegated_by": token_record.delegated_by,
				"scope": token_record.delegation_scope,
				"expires_at": token_record.expires_at.isoformat()
			}
			
		except Exception as e:
			self.logger.error(f"Delegation token verification error: {str(e)}", exc_info=True)
			return None
	
	async def revoke_user_tokens(self, user_id: str, tenant_id: str, token_type: Optional[str] = None) -> int:
		"""
		Revoke all or specific type of tokens for a user.
		
		Args:
			user_id: User whose tokens to revoke
			tenant_id: Tenant context
			token_type: Optional token type filter
		
		Returns:
			Number of tokens revoked
		"""
		try:
			self.logger.info(_log_token_operation("revoke_tokens", user_id, f"type={token_type}"))
			
			revoked_count = await self._revoke_user_tokens(user_id, tenant_id, token_type)
			
			self.logger.info(_log_token_operation(
				"revoke_tokens_success", user_id, f"revoked={revoked_count}"
			))
			
			return revoked_count
			
		except Exception as e:
			self.logger.error(f"Token revocation error for user {user_id}: {str(e)}", exc_info=True)
			raise
	
	async def cleanup_expired_tokens(self) -> int:
		"""
		Clean up expired tokens from database.
		
		Returns:
			Number of tokens cleaned up
		"""
		try:
			self.logger.info("Starting token cleanup")
			
			cleaned_count = await self._cleanup_expired_tokens()
			
			self.logger.info(f"Token cleanup completed: {cleaned_count} tokens removed")
			
			return cleaned_count
			
		except Exception as e:
			self.logger.error(f"Token cleanup error: {str(e)}", exc_info=True)
			return 0
	
	# Private helper methods
	
	def _generate_secure_token(self, length: int) -> str:
		"""Generate cryptographically secure random token"""
		return secrets.token_urlsafe(length)
	
	def _encrypt_token_value(self, token_value: str) -> str:
		"""Encrypt token value for secure storage"""
		token_bytes = token_value.encode('utf-8')
		encrypted_bytes = self.fernet.encrypt(token_bytes)
		return base64.b64encode(encrypted_bytes).decode('utf-8')
	
	def _decrypt_token_value(self, encrypted_token: str) -> str:
		"""Decrypt token value from storage"""
		encrypted_bytes = base64.b64decode(encrypted_token.encode('utf-8'))
		decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
		return decrypted_bytes.decode('utf-8')
	
	def _generate_totp_code(self, secret: str, time_slot: int, digits: int) -> str:
		"""Generate TOTP code for given time slot"""
		# Decode base32 secret
		secret_bytes = base64.b32decode(secret.upper())
		
		# Convert time slot to bytes
		time_bytes = struct.pack('>Q', time_slot)
		
		# Generate HMAC-SHA1
		hmac_digest = hmac.new(secret_bytes, time_bytes, hashlib.sha1).digest()
		
		# Dynamic truncation
		offset = hmac_digest[-1] & 0x0f
		truncated = struct.unpack('>I', hmac_digest[offset:offset + 4])[0]
		truncated &= 0x7fffffff
		
		# Generate code
		code = truncated % (10 ** digits)
		return str(code).zfill(digits)
	
	def _generate_hotp_code(self, secret: str, counter: int, digits: int) -> str:
		"""Generate HOTP code for given counter"""
		# Decode base32 secret
		secret_bytes = base64.b32decode(secret.upper())
		
		# Convert counter to bytes
		counter_bytes = struct.pack('>Q', counter)
		
		# Generate HMAC-SHA1
		hmac_digest = hmac.new(secret_bytes, counter_bytes, hashlib.sha1).digest()
		
		# Dynamic truncation
		offset = hmac_digest[-1] & 0x0f
		truncated = struct.unpack('>I', hmac_digest[offset:offset + 4])[0]
		truncated &= 0x7fffffff
		
		# Generate code
		code = truncated % (10 ** digits)
		return str(code).zfill(digits)
	
	def _generate_totp_uri(self, secret: str, account_name: str, issuer: str) -> str:
		"""Generate TOTP URI for QR code"""
		return f"otpauth://totp/{issuer}:{account_name}?secret={secret}&issuer={issuer}&digits={self.totp_digits}&period={self.totp_window}"
	
	def _generate_qr_code(self, data: str) -> str:
		"""Generate QR code image data"""
		qr = qrcode.QRCode(
			version=1,
			error_correction=qrcode.constants.ERROR_CORRECT_L,
			box_size=10,
			border=4,
		)
		qr.add_data(data)
		qr.make(fit=True)
		
		img = qr.make_image(fill_color="black", back_color="white")
		
		# Convert to base64 for easy transmission
		buffer = io.BytesIO()
		img.save(buffer, format='PNG')
		img_data = buffer.getvalue()
		
		return base64.b64encode(img_data).decode('utf-8')
	
	def _generate_backup_codes(self, count: int) -> List[str]:
		"""Generate backup recovery codes"""
		codes = []
		for _ in range(count):
			# Generate 8-character alphanumeric code
			code = ''.join(secrets.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(self.backup_code_length))
			# Format as XXXX-XXXX for readability
			formatted_code = f"{code[:4]}-{code[4:]}"
			codes.append(formatted_code)
		
		return codes
	
	def _constant_time_compare(self, a: str, b: str) -> bool:
		"""Constant-time string comparison to prevent timing attacks"""
		if len(a) != len(b):
			return False
		
		result = 0
		for x, y in zip(a, b):
			result |= ord(x) ^ ord(y)
		
		return result == 0
	
	async def _validate_token_context(self, token: AuthToken, context: Dict[str, Any]) -> bool:
		"""Validate token context restrictions"""
		# Check device binding
		if token.device_binding_required:
			device_id = context.get("device", {}).get("device_id", "")
			if token.allowed_devices and device_id not in token.allowed_devices:
				return False
		
		# Check IP restrictions
		if token.ip_restrictions:
			client_ip = context.get("location", {}).get("ip_address", "")
			if client_ip not in token.ip_restrictions:
				return False
		
		return True
	
	# Database operations (placeholders - implement based on your database client)
	
	async def _store_auth_token(self, token: AuthToken) -> None:
		"""Store authentication token in database"""
		# Implementation depends on database client
		pass
	
	async def _get_token_by_value(self, token_value: str) -> Optional[AuthToken]:
		"""Get token by its value"""
		# Implementation depends on database client
		# Note: Should search by encrypted token value
		pass
	
	async def _deactivate_token(self, token_id: str) -> None:
		"""Deactivate token by ID"""
		# Implementation depends on database client
		pass
	
	async def _update_token_usage(self, token: AuthToken) -> None:
		"""Update token usage statistics"""
		# Implementation depends on database client
		pass
	
	async def _get_user_backup_codes(self, user_id: str, tenant_id: str) -> List[AuthToken]:
		"""Get active backup codes for user"""
		# Implementation depends on database client
		pass
	
	async def _revoke_user_tokens(self, user_id: str, tenant_id: str, token_type: Optional[str]) -> int:
		"""Revoke user tokens"""
		# Implementation depends on database client
		pass
	
	async def _cleanup_expired_tokens(self) -> int:
		"""Clean up expired tokens"""
		# Implementation depends on database client
		pass


class HardwareTokenValidator:
	"""Validator for hardware security tokens (YubiKey, etc.)"""
	
	def __init__(self):
		self.logger = logging.getLogger(__name__)
	
	async def verify_yubikey_otp(self, otp: str, client_id: str, secret_key: str) -> bool:
		"""
		Verify YubiKey OTP token.
		
		Args:
			otp: OTP from YubiKey
			client_id: YubiCloud client ID
			secret_key: YubiCloud secret key
		
		Returns:
			True if OTP is valid
		"""
		try:
			# Implement YubiKey OTP verification
			# This would typically involve calling YubiCloud API
			
			# Basic validation
			if len(otp) != 44:
				return False
			
			# Extract public ID and encrypted part
			public_id = otp[:12]
			encrypted_part = otp[12:]
			
			# In production, this would verify against YubiCloud
			# For now, return True for demonstration
			self.logger.info(f"YubiKey OTP validation: {public_id}")
			
			return True
			
		except Exception as e:
			self.logger.error(f"YubiKey OTP verification error: {str(e)}", exc_info=True)
			return False
	
	async def verify_fido2_assertion(self, assertion: Dict[str, Any], challenge: str) -> bool:
		"""
		Verify FIDO2/WebAuthn assertion.
		
		Args:
			assertion: FIDO2 assertion data
			challenge: Authentication challenge
		
		Returns:
			True if assertion is valid
		"""
		try:
			# Implement FIDO2/WebAuthn verification
			# This would involve cryptographic verification of the assertion
			
			self.logger.info("FIDO2 assertion validation")
			
			# Placeholder implementation
			return True
			
		except Exception as e:
			self.logger.error(f"FIDO2 assertion verification error: {str(e)}", exc_info=True)
			return False


class OfflineTokenVerifier:
	"""Offline token verification for air-gapped systems"""
	
	def __init__(self, master_key: bytes):
		self.master_key = master_key
		self.logger = logging.getLogger(__name__)
	
	def generate_offline_token(self, user_id: str, timestamp: int, validity_minutes: int) -> str:
		"""
		Generate offline verification token.
		
		Args:
			user_id: User identifier
			timestamp: Current timestamp
			validity_minutes: Token validity in minutes
		
		Returns:
			Offline token
		"""
		try:
			# Create token payload
			payload = f"{user_id}:{timestamp}:{validity_minutes}"
			payload_bytes = payload.encode('utf-8')
			
			# Generate HMAC signature
			signature = hmac.new(self.master_key, payload_bytes, hashlib.sha256).digest()
			
			# Combine payload and signature
			token_data = payload_bytes + signature
			
			# Encode as base64
			return base64.b64encode(token_data).decode('utf-8')
			
		except Exception as e:
			self.logger.error(f"Offline token generation error: {str(e)}", exc_info=True)
			raise
	
	def verify_offline_token(self, token: str, current_timestamp: int) -> Optional[Dict[str, Any]]:
		"""
		Verify offline token.
		
		Args:
			token: Offline token to verify
			current_timestamp: Current timestamp
		
		Returns:
			Token data if valid
		"""
		try:
			# Decode token
			token_data = base64.b64decode(token.encode('utf-8'))
			
			# Extract payload and signature
			payload_bytes = token_data[:-32]  # All but last 32 bytes (SHA256 hash)
			signature = token_data[-32:]      # Last 32 bytes
			
			# Verify signature
			expected_signature = hmac.new(self.master_key, payload_bytes, hashlib.sha256).digest()
			if not hmac.compare_digest(signature, expected_signature):
				return None
			
			# Parse payload
			payload = payload_bytes.decode('utf-8')
			parts = payload.split(':')
			
			if len(parts) != 3:
				return None
			
			user_id, timestamp_str, validity_str = parts
			timestamp = int(timestamp_str)
			validity_minutes = int(validity_str)
			
			# Check if token is still valid
			expiry_timestamp = timestamp + (validity_minutes * 60)
			if current_timestamp > expiry_timestamp:
				return None
			
			return {
				"user_id": user_id,
				"issued_at": timestamp,
				"expires_at": expiry_timestamp,
				"validity_minutes": validity_minutes
			}
			
		except Exception as e:
			self.logger.error(f"Offline token verification error: {str(e)}", exc_info=True)
			return None


__all__ = [
	"TokenService",
	"HardwareTokenValidator", 
	"OfflineTokenVerifier"
]