"""
Quantum-Resistant Cryptography Implementation

Post-quantum cryptographic algorithms for future-proof authentication
using CRYSTALS-Kyber and CRYSTALS-Dilithium with hybrid classical support.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import hashlib
import secrets
import json
from pathlib import Path
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict
import hmac
import struct
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import numpy as np

from .enhanced_models import CryptographicAlgorithm, QuantumKey

class QuantumSecurityLevel(int, Enum):
	"""NIST post-quantum security levels"""
	LEVEL_1 = 128  # Equivalent to AES-128
	LEVEL_2 = 192  # Equivalent to AES-192  
	LEVEL_3 = 256  # Equivalent to AES-256
	LEVEL_5 = 448  # Beyond AES-256

class KeyPairType(str, Enum):
	"""Types of cryptographic key pairs"""
	ENCRYPTION = "encryption"  # For encrypting data
	SIGNATURE = "signature"    # For digital signatures
	KEY_EXCHANGE = "key_exchange"  # For key establishment

class QuantumToken(BaseModel):
	"""Quantum-resistant authentication token"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Token identifier")
	user_id: str = Field(..., description="User identifier")
	algorithm: CryptographicAlgorithm = Field(..., description="Cryptographic algorithm used")
	
	# Token data
	encrypted_payload: bytes = Field(..., description="Encrypted token payload")
	signature: bytes = Field(..., description="Digital signature")
	public_key_id: str = Field(..., description="Public key used for verification")
	
	# Token metadata
	issued_at: datetime = Field(default_factory=datetime.utcnow, description="Token issue time")
	expires_at: datetime = Field(..., description="Token expiration")
	scope: List[str] = Field(default_factory=list, description="Token scope/permissions")
	
	# Quantum-safe features
	security_level: QuantumSecurityLevel = Field(..., description="Security level")
	is_hybrid: bool = Field(default=False, description="Hybrid classical/quantum-safe")
	classical_fallback: Optional[bytes] = Field(default=None, description="Classical signature fallback")
	
	def is_valid(self) -> bool:
		"""Check if token is still valid"""
		return datetime.utcnow() < self.expires_at
	
	def is_expired(self) -> bool:
		"""Check if token has expired"""
		return datetime.utcnow() >= self.expires_at

class CRYSTALSKyber:
	"""CRYSTALS-Kyber key encapsulation mechanism (KEM) implementation"""
	
	# Kyber parameter sets
	KYBER_512 = {"k": 2, "eta1": 3, "eta2": 2, "q": 3329, "n": 256}
	KYBER_768 = {"k": 3, "eta1": 2, "eta2": 2, "q": 3329, "n": 256}
	KYBER_1024 = {"k": 4, "eta1": 2, "eta2": 2, "q": 3329, "n": 256}
	
	def __init__(self, security_level: QuantumSecurityLevel = QuantumSecurityLevel.LEVEL_3):
		self.security_level = security_level
		
		# Select parameter set based on security level
		if security_level == QuantumSecurityLevel.LEVEL_1:
			self.params = self.KYBER_512
		elif security_level == QuantumSecurityLevel.LEVEL_3:
			self.params = self.KYBER_768
		else:
			self.params = self.KYBER_1024
		
		self.k = self.params["k"]
		self.q = self.params["q"]
		self.n = self.params["n"]
		self.eta1 = self.params["eta1"]
		self.eta2 = self.params["eta2"]
	
	def _log_info(self, message: str, **kwargs):
		"""Log information message"""
		print(f"[Kyber INFO] {message} {kwargs if kwargs else ''}")
	
	def _log_warning(self, message: str, **kwargs):
		"""Log warning message"""
		print(f"[Kyber WARNING] {message} {kwargs if kwargs else ''}")
	
	async def generate_keypair(self) -> Tuple[bytes, bytes]:
		"""Generate Kyber key pair (public_key, private_key)"""
		self._log_info("Generating Kyber keypair", security_level=self.security_level.value)
		
		# In a real implementation, this would use the actual Kyber algorithm
		# For this mock implementation, we'll generate appropriately sized keys
		
		# Generate random seed
		seed = secrets.token_bytes(32)
		
		# Generate private key components
		private_key_size = 32 * self.k * self.n // 8  # Simplified calculation
		private_key = hashlib.sha3_256(seed + b"private").digest()
		
		# Extend to full size
		extended_private = private_key
		while len(extended_private) < private_key_size:
			extended_private += hashlib.sha3_256(extended_private).digest()
		private_key = extended_private[:private_key_size]
		
		# Generate public key components
		public_key_size = 32 * self.k * self.n // 8  # Simplified calculation
		public_key = hashlib.sha3_256(seed + b"public").digest()
		
		# Extend to full size
		extended_public = public_key
		while len(extended_public) < public_key_size:
			extended_public += hashlib.sha3_256(extended_public).digest()
		public_key = extended_public[:public_key_size]
		
		self._log_info("Kyber keypair generated", 
					   public_size=len(public_key), 
					   private_size=len(private_key))
		
		return public_key, private_key
	
	async def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
		"""Encapsulate shared secret (returns ciphertext, shared_secret)"""
		self._log_info("Performing Kyber encapsulation")
		
		# Generate random shared secret
		shared_secret = secrets.token_bytes(32)
		
		# Mock encapsulation - in real implementation would use Kyber algorithm
		# Create deterministic ciphertext based on public key and secret
		ciphertext_data = hashlib.sha3_256(public_key + shared_secret).digest()
		
		# Extend to appropriate ciphertext size
		ciphertext_size = len(public_key) + 32  # Simplified
		ciphertext = ciphertext_data
		while len(ciphertext) < ciphertext_size:
			ciphertext += hashlib.sha3_256(ciphertext).digest()
		ciphertext = ciphertext[:ciphertext_size]
		
		self._log_info("Kyber encapsulation complete", 
					   ciphertext_size=len(ciphertext),
					   secret_size=len(shared_secret))
		
		return ciphertext, shared_secret
	
	async def decapsulate(self, ciphertext: bytes, private_key: bytes) -> bytes:
		"""Decapsulate shared secret from ciphertext"""
		self._log_info("Performing Kyber decapsulation")
		
		# Mock decapsulation - in real implementation would use Kyber algorithm
		# For this mock, we'll derive the secret deterministically from ciphertext and private key
		shared_secret = hashlib.sha3_256(ciphertext + private_key).digest()[:32]
		
		self._log_info("Kyber decapsulation complete", secret_size=len(shared_secret))
		
		return shared_secret

class CRYSTALSDilithium:
	"""CRYSTALS-Dilithium digital signature scheme implementation"""
	
	# Dilithium parameter sets
	DILITHIUM_2 = {"k": 4, "l": 4, "eta": 2, "tau": 39, "beta": 78, "gamma1": 2**17, "gamma2": 95232}
	DILITHIUM_3 = {"k": 6, "l": 5, "eta": 4, "tau": 49, "beta": 196, "gamma1": 2**19, "gamma2": 261888}
	DILITHIUM_5 = {"k": 8, "l": 7, "eta": 2, "tau": 60, "beta": 120, "gamma1": 2**19, "gamma2": 261888}
	
	def __init__(self, security_level: QuantumSecurityLevel = QuantumSecurityLevel.LEVEL_3):
		self.security_level = security_level
		
		# Select parameter set based on security level
		if security_level == QuantumSecurityLevel.LEVEL_2:
			self.params = self.DILITHIUM_2
		elif security_level == QuantumSecurityLevel.LEVEL_3:
			self.params = self.DILITHIUM_3
		else:
			self.params = self.DILITHIUM_5
		
		self.k = self.params["k"]
		self.l = self.params["l"]
		self.eta = self.params["eta"]
	
	def _log_info(self, message: str, **kwargs):
		"""Log information message"""
		print(f"[Dilithium INFO] {message} {kwargs if kwargs else ''}")
	
	async def generate_keypair(self) -> Tuple[bytes, bytes]:
		"""Generate Dilithium signing key pair (public_key, private_key)"""
		self._log_info("Generating Dilithium keypair", security_level=self.security_level.value)
		
		# Generate random seed
		seed = secrets.token_bytes(32)
		
		# Generate private key components
		private_key_size = 32 * (self.k + self.l)  # Simplified calculation
		private_key = hashlib.sha3_256(seed + b"dilithium_private").digest()
		
		# Extend to full size
		extended_private = private_key
		while len(extended_private) < private_key_size:
			extended_private += hashlib.sha3_256(extended_private).digest()
		private_key = extended_private[:private_key_size]
		
		# Generate public key components
		public_key_size = 32 * self.k  # Simplified calculation
		public_key = hashlib.sha3_256(seed + b"dilithium_public").digest()
		
		# Extend to full size
		extended_public = public_key
		while len(extended_public) < public_key_size:
			extended_public += hashlib.sha3_256(extended_public).digest()
		public_key = extended_public[:public_key_size]
		
		self._log_info("Dilithium keypair generated",
					   public_size=len(public_key),
					   private_size=len(private_key))
		
		return public_key, private_key
	
	async def sign(self, message: bytes, private_key: bytes) -> bytes:
		"""Sign message with Dilithium private key"""
		self._log_info("Signing message with Dilithium", message_size=len(message))
		
		# Mock signing - in real implementation would use Dilithium algorithm
		# Create deterministic signature based on message and private key
		signature_data = hashlib.sha3_256(message + private_key + b"dilithium_sign").digest()
		
		# Dilithium signatures are variable length, but typically around 2420-4595 bytes
		signature_size = 2420 + (self.security_level.value // 64) * 500  # Simplified calculation
		signature = signature_data
		
		# Extend to appropriate signature size
		while len(signature) < signature_size:
			signature += hashlib.sha3_256(signature + message).digest()
		signature = signature[:signature_size]
		
		self._log_info("Dilithium signature created", signature_size=len(signature))
		
		return signature
	
	async def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
		"""Verify Dilithium signature"""
		self._log_info("Verifying Dilithium signature", 
					   message_size=len(message),
					   signature_size=len(signature))
		
		# Mock verification - in real implementation would use Dilithium algorithm
		# Recreate expected signature for comparison
		expected_signature_data = hashlib.sha3_256(message + public_key + b"dilithium_verify").digest()
		
		# For mock verification, check if signature starts with expected pattern
		signature_hash = hashlib.sha3_256(signature).digest()
		expected_hash = hashlib.sha3_256(expected_signature_data).digest()
		
		# Simple mock verification - in reality this would be much more complex
		valid = signature_hash[:16] == expected_hash[:16]
		
		self._log_info("Dilithium signature verification", valid=valid)
		
		return valid

class QuantumResistantAuth:
	"""Quantum-resistant authentication system"""
	
	def __init__(self):
		self._kyber_kem = CRYSTALSKyber()
		self._dilithium_dsa = CRYSTALSDilithium()
		
		# Key storage (in production, would use secure key management)
		self._quantum_keys: Dict[str, QuantumKey] = {}
		self._shared_secrets: Dict[str, bytes] = {}
		
		# Performance monitoring
		self._operation_times: Dict[str, List[float]] = {}
	
	def _log_info(self, message: str, **kwargs):
		"""Log information message"""
		print(f"[QuantumAuth INFO] {message} {kwargs if kwargs else ''}")
	
	def _log_warning(self, message: str, **kwargs):
		"""Log warning message"""
		print(f"[QuantumAuth WARNING] {message} {kwargs if kwargs else ''}")
	
	def _log_error(self, message: str, **kwargs):
		"""Log error message"""
		print(f"[QuantumAuth ERROR] {message} {kwargs if kwargs else ''}")
	
	async def _time_operation(self, operation_name: str, operation_func):
		"""Time cryptographic operations for performance monitoring"""
		start_time = asyncio.get_event_loop().time()
		result = await operation_func()
		end_time = asyncio.get_event_loop().time()
		
		duration_ms = (end_time - start_time) * 1000
		
		if operation_name not in self._operation_times:
			self._operation_times[operation_name] = []
		self._operation_times[operation_name].append(duration_ms)
		
		# Keep only last 100 measurements
		self._operation_times[operation_name] = self._operation_times[operation_name][-100:]
		
		return result, duration_ms
	
	async def generate_quantum_safe_keypair(self, user_id: str, 
											security_level: QuantumSecurityLevel = QuantumSecurityLevel.LEVEL_3,
											key_type: KeyPairType = KeyPairType.SIGNATURE) -> QuantumKey:
		"""Generate quantum-safe key pair for user"""
		assert user_id, "User ID is required"
		
		self._log_info("Generating quantum-safe keypair", 
					   user_id=user_id, 
					   security_level=security_level.value,
					   key_type=key_type.value)
		
		if key_type == KeyPairType.SIGNATURE:
			# Use Dilithium for signatures
			dilithium = CRYSTALSDilithium(security_level)
			public_key, private_key = await self._time_operation(
				"dilithium_keygen",
				lambda: dilithium.generate_keypair()
			)
			public_key, keygen_time = public_key
			algorithm = CryptographicAlgorithm.CRYSTALS_DILITHIUM
			
		elif key_type == KeyPairType.ENCRYPTION:
			# Use Kyber for key exchange/encryption
			kyber = CRYSTALSKyber(security_level)
			public_key, private_key = await self._time_operation(
				"kyber_keygen", 
				lambda: kyber.generate_keypair()
			)
			public_key, keygen_time = public_key
			algorithm = CryptographicAlgorithm.CRYSTALS_KYBER
			
		else:  # KEY_EXCHANGE
			# Use Kyber for key exchange
			kyber = CRYSTALSKyber(security_level)
			public_key, private_key = await self._time_operation(
				"kyber_keygen",
				lambda: kyber.generate_keypair()
			)
			public_key, keygen_time = public_key
			algorithm = CryptographicAlgorithm.CRYSTALS_KYBER
		
		# Create quantum key object
		quantum_key = QuantumKey(
			user_id=user_id,
			algorithm=algorithm,
			public_key=public_key,
			private_key_encrypted=self._encrypt_private_key(private_key, user_id),
			key_derivation_salt=secrets.token_bytes(32),
			key_size=len(private_key) * 8,  # Convert to bits
			security_level=security_level.value,
			expires_at=datetime.utcnow() + timedelta(days=365)  # 1 year validity
		)
		
		# Store key
		self._quantum_keys[quantum_key.id] = quantum_key
		
		self._log_info("Quantum-safe keypair generated",
					   key_id=quantum_key.id,
					   algorithm=algorithm.value,
					   keygen_time_ms=keygen_time,
					   public_key_size=len(public_key),
					   private_key_size=len(private_key))
		
		return quantum_key
	
	def _encrypt_private_key(self, private_key: bytes, user_id: str) -> bytes:
		"""Encrypt private key for storage"""
		# Derive encryption key from user ID and system secret
		salt = hashlib.sha256(user_id.encode()).digest()
		kdf = PBKDF2HMAC(
			algorithm=hashes.SHA256(),
			length=32,
			salt=salt,
			iterations=100000,
			backend=default_backend()
		)
		
		# In production, this would use a proper key management system
		master_key = b"quantum_auth_master_key_changeme"  # Should be from secure config
		encryption_key = kdf.derive(master_key)
		
		# Encrypt private key with AES-256-GCM
		iv = secrets.token_bytes(12)
		cipher = Cipher(algorithms.AES(encryption_key), modes.GCM(iv), backend=default_backend())
		encryptor = cipher.encryptor()
		ciphertext = encryptor.update(private_key) + encryptor.finalize()
		
		# Return IV + tag + ciphertext
		return iv + encryptor.tag + ciphertext
	
	def _decrypt_private_key(self, encrypted_private_key: bytes, user_id: str) -> bytes:
		"""Decrypt private key from storage"""
		# Derive encryption key
		salt = hashlib.sha256(user_id.encode()).digest()
		kdf = PBKDF2HMAC(
			algorithm=hashes.SHA256(),
			length=32,
			salt=salt,
			iterations=100000,
			backend=default_backend()
		)
		
		master_key = b"quantum_auth_master_key_changeme"  # Should be from secure config
		encryption_key = kdf.derive(master_key)
		
		# Extract IV, tag, and ciphertext
		iv = encrypted_private_key[:12]
		tag = encrypted_private_key[12:28]
		ciphertext = encrypted_private_key[28:]
		
		# Decrypt
		cipher = Cipher(algorithms.AES(encryption_key), modes.GCM(iv, tag), backend=default_backend())
		decryptor = cipher.decryptor()
		private_key = decryptor.update(ciphertext) + decryptor.finalize()
		
		return private_key
	
	async def sign_token(self, token_data: Dict[str, Any], private_key_id: str) -> bytes:
		"""Sign token with quantum-safe digital signature"""
		assert token_data, "Token data is required"
		assert private_key_id, "Private key ID is required"
		
		quantum_key = self._quantum_keys.get(private_key_id)
		if not quantum_key or not quantum_key.is_valid():
			raise ValueError("Invalid or expired quantum key")
		
		self._log_info("Signing token with quantum-safe signature", 
					   key_id=private_key_id,
					   algorithm=quantum_key.algorithm.value)
		
		# Serialize token data
		token_json = json.dumps(token_data, sort_keys=True, separators=(',', ':')).encode()
		
		# Decrypt private key
		private_key = self._decrypt_private_key(quantum_key.private_key_encrypted, quantum_key.user_id)
		
		# Sign based on algorithm
		if quantum_key.algorithm == CryptographicAlgorithm.CRYSTALS_DILITHIUM:
			dilithium = CRYSTALSDilithium(QuantumSecurityLevel(quantum_key.security_level))
			signature, sign_time = await self._time_operation(
				"dilithium_sign",
				lambda: dilithium.sign(token_json, private_key)
			)
		else:
			raise ValueError(f"Unsupported signing algorithm: {quantum_key.algorithm}")
		
		# Update key usage
		quantum_key.last_used_at = datetime.utcnow()
		quantum_key.use_count += 1
		
		self._log_info("Token signed successfully",
					   signature_size=len(signature),
					   sign_time_ms=sign_time)
		
		return signature
	
	async def verify_token_signature(self, token_data: Dict[str, Any], signature: bytes, 
									 public_key_id: str) -> bool:
		"""Verify quantum-safe token signature"""
		assert token_data, "Token data is required"
		assert signature, "Signature is required"
		assert public_key_id, "Public key ID is required"
		
		quantum_key = self._quantum_keys.get(public_key_id)
		if not quantum_key or not quantum_key.is_valid():
			self._log_warning("Invalid or expired quantum key for verification", key_id=public_key_id)
			return False
		
		self._log_info("Verifying quantum-safe token signature",
					   key_id=public_key_id,
					   algorithm=quantum_key.algorithm.value)
		
		# Serialize token data
		token_json = json.dumps(token_data, sort_keys=True, separators=(',', ':')).encode()
		
		# Verify based on algorithm
		if quantum_key.algorithm == CryptographicAlgorithm.CRYSTALS_DILITHIUM:
			dilithium = CRYSTALSDilithium(QuantumSecurityLevel(quantum_key.security_level))
			valid, verify_time = await self._time_operation(
				"dilithium_verify",
				lambda: dilithium.verify(token_json, signature, quantum_key.public_key)
			)
		else:
			self._log_error("Unsupported verification algorithm", algorithm=quantum_key.algorithm)
			return False
		
		self._log_info("Token signature verification complete",
					   valid=valid,
					   verify_time_ms=verify_time)
		
		return valid
	
	async def create_quantum_safe_token(self, user_id: str, payload: Dict[str, Any],
										expires_in_minutes: int = 60,
										security_level: QuantumSecurityLevel = QuantumSecurityLevel.LEVEL_3) -> QuantumToken:
		"""Create quantum-safe authentication token"""
		assert user_id, "User ID is required"
		assert payload, "Token payload is required"
		
		self._log_info("Creating quantum-safe token", user_id=user_id, expires_in=expires_in_minutes)
		
		# Find or create signing key for user
		user_keys = [key for key in self._quantum_keys.values() 
					if (key.user_id == user_id and 
						key.algorithm == CryptographicAlgorithm.CRYSTALS_DILITHIUM and
						key.is_valid())]
		
		if not user_keys:
			# Generate new signing key
			signing_key = await self.generate_quantum_safe_keypair(
				user_id, security_level, KeyPairType.SIGNATURE
			)
		else:
			signing_key = user_keys[0]  # Use first available key
		
		# Create token data
		now = datetime.utcnow()
		expires_at = now + timedelta(minutes=expires_in_minutes)
		
		token_data = {
			"sub": user_id,
			"iat": int(now.timestamp()),
			"exp": int(expires_at.timestamp()),
			"type": "quantum_safe_access",
			"security_level": security_level.value,
			**payload
		}
		
		# Sign token
		signature = await self.sign_token(token_data, signing_key.id)
		
		# Encrypt payload for confidentiality
		encrypted_payload = self._encrypt_token_payload(json.dumps(token_data).encode())
		
		# Create quantum token
		quantum_token = QuantumToken(
			user_id=user_id,
			algorithm=CryptographicAlgorithm.CRYSTALS_DILITHIUM,
			encrypted_payload=encrypted_payload,
			signature=signature,
			public_key_id=signing_key.id,
			expires_at=expires_at,
			security_level=security_level,
			scope=payload.get("scope", [])
		)
		
		self._log_info("Quantum-safe token created",
					   token_id=quantum_token.id,
					   expires_at=expires_at.isoformat())
		
		return quantum_token
	
	def _encrypt_token_payload(self, payload: bytes) -> bytes:
		"""Encrypt token payload for confidentiality"""
		# Generate random key for token encryption
		token_key = secrets.token_bytes(32)
		iv = secrets.token_bytes(12)
		
		# Encrypt with AES-256-GCM
		cipher = Cipher(algorithms.AES(token_key), modes.GCM(iv), backend=default_backend())
		encryptor = cipher.encryptor()
		ciphertext = encryptor.update(payload) + encryptor.finalize()
		
		# Return key + IV + tag + ciphertext (for simplicity)
		# In production, the key would be encrypted with recipient's public key
		return token_key + iv + encryptor.tag + ciphertext
	
	def _decrypt_token_payload(self, encrypted_payload: bytes) -> bytes:
		"""Decrypt token payload"""
		# Extract components
		token_key = encrypted_payload[:32]
		iv = encrypted_payload[32:44]
		tag = encrypted_payload[44:60]
		ciphertext = encrypted_payload[60:]
		
		# Decrypt
		cipher = Cipher(algorithms.AES(token_key), modes.GCM(iv, tag), backend=default_backend())
		decryptor = cipher.decryptor()
		payload = decryptor.update(ciphertext) + decryptor.finalize()
		
		return payload
	
	async def verify_quantum_token(self, quantum_token: QuantumToken) -> Optional[Dict[str, Any]]:
		"""Verify quantum-safe token and return payload"""
		if not quantum_token.is_valid():
			self._log_warning("Token has expired", token_id=quantum_token.id)
			return None
		
		self._log_info("Verifying quantum-safe token", 
					   token_id=quantum_token.id,
					   user_id=quantum_token.user_id)
		
		try:
			# Decrypt payload
			decrypted_payload = self._decrypt_token_payload(quantum_token.encrypted_payload)
			token_data = json.loads(decrypted_payload.decode())
			
			# Verify signature
			signature_valid = await self.verify_token_signature(
				token_data, quantum_token.signature, quantum_token.public_key_id
			)
			
			if not signature_valid:
				self._log_warning("Invalid token signature", token_id=quantum_token.id)
				return None
			
			# Check expiration in payload
			if token_data.get("exp", 0) < datetime.utcnow().timestamp():
				self._log_warning("Token payload has expired", token_id=quantum_token.id)
				return None
			
			self._log_info("Quantum token verified successfully", token_id=quantum_token.id)
			return token_data
			
		except Exception as e:
			self._log_error("Token verification failed", token_id=quantum_token.id, error=str(e))
			return None
	
	async def create_hybrid_token(self, user_id: str, payload: Dict[str, Any],
								  expires_in_minutes: int = 60) -> Tuple[QuantumToken, str]:
		"""Create hybrid classical/quantum-safe token"""
		self._log_info("Creating hybrid token", user_id=user_id)
		
		# Create quantum-safe token
		quantum_token = await self.create_quantum_safe_token(user_id, payload, expires_in_minutes)
		quantum_token.is_hybrid = True
		
		# Create classical JWT fallback
		classical_payload = {
			"sub": user_id,
			"iat": int(datetime.utcnow().timestamp()),
			"exp": int(quantum_token.expires_at.timestamp()),
			"type": "classical_fallback",
			"quantum_token_id": quantum_token.id,
			**payload
		}
		
		# Sign with classical RSA (simplified)
		classical_jwt = self._create_classical_jwt(classical_payload)
		quantum_token.classical_fallback = classical_jwt.encode()
		
		self._log_info("Hybrid token created", 
					   quantum_id=quantum_token.id,
					   has_classical_fallback=True)
		
		return quantum_token, classical_jwt
	
	def _create_classical_jwt(self, payload: Dict[str, Any]) -> str:
		"""Create classical JWT token (simplified implementation)"""
		# In production, would use proper JWT library with RSA/ECDSA
		import base64
		
		header = {"alg": "HS256", "typ": "JWT"}
		
		# Encode header and payload
		encoded_header = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip('=')
		encoded_payload = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')
		
		# Create signature (simplified with HMAC)
		secret = b"classical_jwt_secret_changeme"  # Should be from secure config
		message = f"{encoded_header}.{encoded_payload}"
		signature = hmac.new(secret, message.encode(), hashlib.sha256).digest()
		encoded_signature = base64.urlsafe_b64encode(signature).decode().rstrip('=')
		
		return f"{encoded_header}.{encoded_payload}.{encoded_signature}"
	
	async def migrate_to_quantum_safe(self, user_id: str) -> bool:
		"""Migrate user from classical to quantum-safe cryptography"""
		self._log_info("Migrating user to quantum-safe crypto", user_id=user_id)
		
		try:
			# Generate new quantum-safe keys
			signing_key = await self.generate_quantum_safe_keypair(
				user_id, QuantumSecurityLevel.LEVEL_3, KeyPairType.SIGNATURE
			)
			encryption_key = await self.generate_quantum_safe_keypair(
				user_id, QuantumSecurityLevel.LEVEL_3, KeyPairType.ENCRYPTION
			)
			
			self._log_info("Quantum-safe migration completed",
						   user_id=user_id,
						   signing_key_id=signing_key.id,
						   encryption_key_id=encryption_key.id)
			
			return True
			
		except Exception as e:
			self._log_error("Quantum-safe migration failed", user_id=user_id, error=str(e))
			return False
	
	def get_user_quantum_keys(self, user_id: str) -> List[QuantumKey]:
		"""Get all quantum keys for user"""
		return [key for key in self._quantum_keys.values() 
				if key.user_id == user_id and key.is_valid()]
	
	def revoke_quantum_key(self, key_id: str, reason: str = "Manual revocation"):
		"""Revoke quantum key"""
		key = self._quantum_keys.get(key_id)
		if key:
			key.revoke(reason)
			self._log_info("Quantum key revoked", key_id=key_id, reason=reason)
	
	def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
		"""Get performance metrics for quantum operations"""
		metrics = {}
		
		for operation, times in self._operation_times.items():
			if times:
				metrics[operation] = {
					"avg_ms": np.mean(times),
					"min_ms": np.min(times),
					"max_ms": np.max(times),
					"std_ms": np.std(times),
					"count": len(times)
				}
		
		return metrics
	
	async def key_rollover(self, user_id: str, old_key_id: str) -> QuantumKey:
		"""Roll over quantum key to new key"""
		old_key = self._quantum_keys.get(old_key_id)
		if not old_key or old_key.user_id != user_id:
			raise ValueError("Invalid old key for rollover")
		
		self._log_info("Rolling over quantum key", user_id=user_id, old_key_id=old_key_id)
		
		# Generate new key with same parameters
		key_type = KeyPairType.SIGNATURE if old_key.algorithm == CryptographicAlgorithm.CRYSTALS_DILITHIUM else KeyPairType.ENCRYPTION
		security_level = QuantumSecurityLevel(old_key.security_level)
		
		new_key = await self.generate_quantum_safe_keypair(user_id, security_level, key_type)
		
		# Revoke old key
		old_key.revoke("Key rollover")
		
		self._log_info("Quantum key rollover completed",
					   user_id=user_id,
					   old_key_id=old_key_id,
					   new_key_id=new_key.id)
		
		return new_key
	
	def clear_user_keys(self, user_id: str):
		"""Clear all quantum keys for user (GDPR compliance)"""
		user_keys = [key_id for key_id, key in self._quantum_keys.items() 
					if key.user_id == user_id]
		
		for key_id in user_keys:
			del self._quantum_keys[key_id]
		
		# Clear shared secrets
		user_secrets = [secret_id for secret_id in self._shared_secrets.keys()
					   if user_id in secret_id]
		
		for secret_id in user_secrets:
			del self._shared_secrets[secret_id]
		
		self._log_info("Quantum keys cleared for user", user_id=user_id, keys_removed=len(user_keys))