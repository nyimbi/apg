"""
Security utilities and cryptographic functions

© 2025 Datacraft. All rights reserved.
"""

import hashlib
import hmac
import secrets
import base64
import uuid
import platform
from typing import Optional, Union, Dict, Any
from datetime import datetime, timedelta
import json

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

from .constants import ENCRYPTION_KEY_LENGTH


class SecurityError(Exception):
	"""Base exception for security-related errors"""
	pass


class EncryptionError(SecurityError):
	"""Exception for encryption/decryption errors"""
	pass


class SignatureError(SecurityError):
	"""Exception for signature verification errors"""
	pass


def generate_random_bytes(length: int = 32) -> bytes:
	"""Generate cryptographically secure random bytes"""
	return secrets.token_bytes(length)


def generate_random_string(length: int = 32, url_safe: bool = True) -> str:
	"""Generate cryptographically secure random string"""
	if url_safe:
		return secrets.token_urlsafe(length)
	else:
		return secrets.token_hex(length)


def generate_device_id() -> str:
	"""Generate unique device identifier"""
	# Combine system information for device fingerprinting
	system_info = [
		platform.system(),
		platform.machine(),
		platform.processor(),
		str(uuid.getnode()),  # MAC address
	]
	
	# Create deterministic but unique device ID
	combined = "".join(system_info).encode('utf-8')
	device_hash = hashlib.sha256(combined).hexdigest()
	
	# Format as UUID-like string
	return f"device-{device_hash[:32]}"


def derive_key_from_password(password: str, salt: Optional[bytes] = None) -> tuple[bytes, bytes]:
	"""Derive encryption key from password using PBKDF2"""
	if salt is None:
		salt = generate_random_bytes(16)
	
	kdf = PBKDF2HMAC(
		algorithm=hashes.SHA256(),
		length=ENCRYPTION_KEY_LENGTH,
		salt=salt,
		iterations=100000,
		backend=default_backend()
	)
	
	key = kdf.derive(password.encode('utf-8'))
	return key, salt


def generate_encryption_key() -> bytes:
	"""Generate new encryption key"""
	return Fernet.generate_key()


def encrypt_data(data: Union[str, bytes], key: Union[str, bytes]) -> str:
	"""Encrypt data using Fernet symmetric encryption"""
	try:
		# Ensure key is bytes
		if isinstance(key, str):
			# If key is a string, derive a proper key from it
			derived_key, _ = derive_key_from_password(key)
			fernet_key = base64.urlsafe_b64encode(derived_key)
		else:
			# If key is bytes, ensure it's properly formatted for Fernet
			if len(key) == ENCRYPTION_KEY_LENGTH:
				fernet_key = base64.urlsafe_b64encode(key)
			else:
				fernet_key = key
		
		# Ensure data is bytes
		if isinstance(data, str):
			data = data.encode('utf-8')
		
		# Create Fernet cipher
		cipher = Fernet(fernet_key)
		
		# Encrypt and return base64 encoded string
		encrypted_data = cipher.encrypt(data)
		return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
		
	except Exception as e:
		raise EncryptionError(f"Failed to encrypt data: {e}")


def decrypt_data(encrypted_data: str, key: Union[str, bytes]) -> str:
	"""Decrypt data using Fernet symmetric encryption"""
	try:
		# Ensure key is bytes
		if isinstance(key, str):
			# If key is a string, derive a proper key from it
			derived_key, _ = derive_key_from_password(key)
			fernet_key = base64.urlsafe_b64encode(derived_key)
		else:
			# If key is bytes, ensure it's properly formatted for Fernet
			if len(key) == ENCRYPTION_KEY_LENGTH:
				fernet_key = base64.urlsafe_b64encode(key)
			else:
				fernet_key = key
		
		# Decode the encrypted data
		encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
		
		# Create Fernet cipher
		cipher = Fernet(fernet_key)
		
		# Decrypt and return string
		decrypted_data = cipher.decrypt(encrypted_bytes)
		return decrypted_data.decode('utf-8')
		
	except Exception as e:
		raise EncryptionError(f"Failed to decrypt data: {e}")


def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
	"""Hash password using PBKDF2 with SHA-256"""
	if salt is None:
		salt = generate_random_bytes(16)
	
	# Use PBKDF2 for password hashing
	pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
	
	# Return hash and salt as base64 strings
	return (
		base64.urlsafe_b64encode(pwdhash).decode('utf-8'),
		base64.urlsafe_b64encode(salt).decode('utf-8')
	)


def verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
	"""Verify password against stored hash and salt"""
	try:
		salt = base64.urlsafe_b64decode(stored_salt.encode('utf-8'))
		expected_hash = base64.urlsafe_b64decode(stored_hash.encode('utf-8'))
		
		# Hash the provided password with the stored salt
		pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
		
		# Use constant-time comparison to prevent timing attacks
		return hmac.compare_digest(pwdhash, expected_hash)
		
	except Exception:
		return False


def generate_signature(data: Union[str, bytes], secret_key: str) -> str:
	"""Generate HMAC signature for data"""
	if isinstance(data, str):
		data = data.encode('utf-8')
	
	signature = hmac.new(
		secret_key.encode('utf-8'),
		data,
		hashlib.sha256
	).digest()
	
	return base64.urlsafe_b64encode(signature).decode('utf-8')


def verify_signature(data: Union[str, bytes], signature: str, secret_key: str) -> bool:
	"""Verify HMAC signature for data"""
	try:
		expected_signature = generate_signature(data, secret_key)
		return hmac.compare_digest(signature, expected_signature)
	except Exception:
		return False


def generate_rsa_keypair(key_size: int = 2048) -> tuple[bytes, bytes]:
	"""Generate RSA public/private key pair"""
	private_key = rsa.generate_private_key(
		public_exponent=65537,
		key_size=key_size,
		backend=default_backend()
	)
	
	public_key = private_key.public_key()
	
	# Serialize keys to PEM format
	private_pem = private_key.private_bytes(
		encoding=serialization.Encoding.PEM,
		format=serialization.PrivateFormat.PKCS8,
		encryption_algorithm=serialization.NoEncryption()
	)
	
	public_pem = public_key.public_bytes(
		encoding=serialization.Encoding.PEM,
		format=serialization.PublicFormat.SubjectPublicKeyInfo
	)
	
	return private_pem, public_pem


def encrypt_with_rsa_public_key(data: Union[str, bytes], public_key_pem: bytes) -> str:
	"""Encrypt data with RSA public key"""
	try:
		if isinstance(data, str):
			data = data.encode('utf-8')
		
		# Load public key
		public_key = serialization.load_pem_public_key(
			public_key_pem,
			backend=default_backend()
		)
		
		# Encrypt data
		encrypted_data = public_key.encrypt(
			data,
			padding.OAEP(
				mgf=padding.MGF1(algorithm=hashes.SHA256()),
				algorithm=hashes.SHA256(),
				label=None
			)
		)
		
		return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
		
	except Exception as e:
		raise EncryptionError(f"Failed to encrypt with RSA public key: {e}")


def decrypt_with_rsa_private_key(encrypted_data: str, private_key_pem: bytes) -> str:
	"""Decrypt data with RSA private key"""
	try:
		# Decode encrypted data
		encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
		
		# Load private key
		private_key = serialization.load_pem_private_key(
			private_key_pem,
			password=None,
			backend=default_backend()
		)
		
		# Decrypt data
		decrypted_data = private_key.decrypt(
			encrypted_bytes,
			padding.OAEP(
				mgf=padding.MGF1(algorithm=hashes.SHA256()),
				algorithm=hashes.SHA256(),
				label=None
			)
		)
		
		return decrypted_data.decode('utf-8')
		
	except Exception as e:
		raise EncryptionError(f"Failed to decrypt with RSA private key: {e}")


def sign_with_rsa_private_key(data: Union[str, bytes], private_key_pem: bytes) -> str:
	"""Sign data with RSA private key"""
	try:
		if isinstance(data, str):
			data = data.encode('utf-8')
		
		# Load private key
		private_key = serialization.load_pem_private_key(
			private_key_pem,
			password=None,
			backend=default_backend()
		)
		
		# Sign data
		signature = private_key.sign(
			data,
			padding.PSS(
				mgf=padding.MGF1(hashes.SHA256()),
				salt_length=padding.PSS.MAX_LENGTH
			),
			hashes.SHA256()
		)
		
		return base64.urlsafe_b64encode(signature).decode('utf-8')
		
	except Exception as e:
		raise SignatureError(f"Failed to sign with RSA private key: {e}")


def verify_rsa_signature(data: Union[str, bytes], signature: str, public_key_pem: bytes) -> bool:
	"""Verify RSA signature with public key"""
	try:
		if isinstance(data, str):
			data = data.encode('utf-8')
		
		# Decode signature
		signature_bytes = base64.urlsafe_b64decode(signature.encode('utf-8'))
		
		# Load public key
		public_key = serialization.load_pem_public_key(
			public_key_pem,
			backend=default_backend()
		)
		
		# Verify signature
		public_key.verify(
			signature_bytes,
			data,
			padding.PSS(
				mgf=padding.MGF1(hashes.SHA256()),
				salt_length=padding.PSS.MAX_LENGTH
			),
			hashes.SHA256()
		)
		
		return True
		
	except Exception:
		return False


def secure_hash(data: Union[str, bytes], algorithm: str = 'sha256') -> str:
	"""Generate secure hash of data"""
	if isinstance(data, str):
		data = data.encode('utf-8')
	
	if algorithm == 'sha256':
		hash_obj = hashlib.sha256(data)
	elif algorithm == 'sha512':
		hash_obj = hashlib.sha512(data)
	elif algorithm == 'sha1':
		hash_obj = hashlib.sha1(data)
	else:
		raise ValueError(f"Unsupported hash algorithm: {algorithm}")
	
	return hash_obj.hexdigest()


def generate_jwt_secret() -> str:
	"""Generate secure JWT secret key"""
	return generate_random_string(64)


def generate_csrf_token() -> str:
	"""Generate CSRF token"""
	return generate_random_string(32)


def secure_compare(a: str, b: str) -> bool:
	"""Constant-time string comparison to prevent timing attacks"""
	return hmac.compare_digest(a, b)


class SecureStorage:
	"""Secure storage utility for sensitive data"""
	
	def __init__(self, master_key: Optional[str] = None):
		if master_key:
			self.key, _ = derive_key_from_password(master_key)
		else:
			self.key = generate_encryption_key()
	
	def store(self, key: str, value: Any) -> str:
		"""Store value securely"""
		# Serialize value to JSON
		json_data = json.dumps(value, default=str)
		
		# Encrypt the data
		encrypted_data = encrypt_data(json_data, self.key)
		
		return encrypted_data
	
	def retrieve(self, key: str, encrypted_data: str) -> Any:
		"""Retrieve value securely"""
		try:
			# Decrypt the data
			json_data = decrypt_data(encrypted_data, self.key)
			
			# Deserialize from JSON
			return json.loads(json_data)
			
		except Exception as e:
			raise SecurityError(f"Failed to retrieve secure data: {e}")
	
	def update_key(self, new_master_key: str):
		"""Update the master key"""
		self.key, _ = derive_key_from_password(new_master_key)


class TokenManager:
	"""Utility for managing secure tokens"""
	
	def __init__(self, secret_key: Optional[str] = None):
		self.secret_key = secret_key or generate_random_string(64)
	
	def generate_token(self, payload: Dict[str, Any], expiry_hours: int = 24) -> str:
		"""Generate signed token with expiry"""
		# Add expiry timestamp
		expiry = datetime.utcnow() + timedelta(hours=expiry_hours)
		payload['exp'] = expiry.timestamp()
		payload['iat'] = datetime.utcnow().timestamp()
		
		# Serialize payload
		payload_json = json.dumps(payload, sort_keys=True, default=str)
		
		# Generate signature
		signature = generate_signature(payload_json, self.secret_key)
		
		# Combine payload and signature
		token_data = {
			'payload': base64.urlsafe_b64encode(payload_json.encode('utf-8')).decode('utf-8'),
			'signature': signature
		}
		
		return base64.urlsafe_b64encode(json.dumps(token_data).encode('utf-8')).decode('utf-8')
	
	def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
		"""Verify and decode token"""
		try:
			# Decode token
			token_data = json.loads(base64.urlsafe_b64decode(token.encode('utf-8')).decode('utf-8'))
			
			# Extract payload and signature
			payload_b64 = token_data['payload']
			signature = token_data['signature']
			
			# Decode payload
			payload_json = base64.urlsafe_b64decode(payload_b64.encode('utf-8')).decode('utf-8')
			
			# Verify signature
			if not verify_signature(payload_json, signature, self.secret_key):
				return None
			
			# Parse payload
			payload = json.loads(payload_json)
			
			# Check expiry
			if 'exp' in payload:
				if datetime.utcnow().timestamp() > payload['exp']:
					return None  # Token expired
			
			return payload
			
		except Exception:
			return None
	
	def is_token_expired(self, token: str) -> bool:
		"""Check if token is expired"""
		payload = self.verify_token(token)
		return payload is None


def sanitize_filename(filename: str) -> str:
	"""Sanitize filename to prevent path traversal attacks"""
	import re
	import os.path
	
	# Remove any path components
	filename = os.path.basename(filename)
	
	# Remove or replace dangerous characters
	filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
	
	# Remove leading/trailing whitespace and dots
	filename = filename.strip(' .')
	
	# Ensure filename is not empty
	if not filename:
		filename = "file"
	
	# Limit filename length
	if len(filename) > 255:
		name, ext = os.path.splitext(filename)
		filename = name[:255-len(ext)] + ext
	
	return filename


def validate_input(data: str, max_length: int = 1000, allow_html: bool = False) -> str:
	"""Validate and sanitize user input"""
	if not isinstance(data, str):
		raise ValueError("Input must be a string")
	
	# Check length
	if len(data) > max_length:
		raise ValueError(f"Input too long (max {max_length} characters)")
	
	# Remove null bytes
	data = data.replace('\x00', '')
	
	# HTML sanitization if not allowed
	if not allow_html:
		import html
		data = html.escape(data)
	
	return data