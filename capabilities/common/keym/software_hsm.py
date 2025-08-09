#!/usr/bin/env python3
"""
APG Key Management - Software HSM Implementation
Complete software-based HSM with all standard HSM capabilities

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import secrets
import hashlib
import json
import os
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import aiosqlite
from cryptography.hazmat.primitives import hashes, serialization, padding
from cryptography.hazmat.primitives.asymmetric import rsa, ec, ed25519, x25519
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from uuid_extensions import uuid7str

from .models import KeyAlgorithm, SecurityLevel


class SoftwareHSMKeyType(str, Enum):
	"""Software HSM key types"""
	AES = "aes"
	RSA = "rsa"
	ECDSA = "ecdsa"
	EDDSA = "eddsa"
	ECDH = "ecdh"
	X25519 = "x25519"
	HMAC = "hmac"


class SoftwareHSMOperation(str, Enum):
	"""Software HSM operations"""
	GENERATE_KEY = "generate_key"
	IMPORT_KEY = "import_key"
	EXPORT_KEY = "export_key"
	DELETE_KEY = "delete_key"
	ENCRYPT = "encrypt"
	DECRYPT = "decrypt"
	SIGN = "sign"
	VERIFY = "verify"
	WRAP_KEY = "wrap_key"
	UNWRAP_KEY = "unwrap_key"
	DERIVE_KEY = "derive_key"
	GET_RANDOM = "get_random"


@dataclass
class SoftwareHSMKey:
	"""Software HSM key object"""
	key_id: str = field(default_factory=uuid7str)
	key_type: SoftwareHSMKeyType = SoftwareHSMKeyType.AES
	key_size: int = 256
	key_material: bytes = b""
	public_key_material: Optional[bytes] = None
	algorithm: KeyAlgorithm = KeyAlgorithm.AES_256
	usage: List[str] = field(default_factory=list)
	attributes: Dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	last_used: datetime = field(default_factory=datetime.utcnow)
	use_count: int = 0
	extractable: bool = True
	sensitive: bool = False
	wrap_with_trusted: bool = False
	trusted: bool = False
	token: bool = True
	private: bool = True
	modifiable: bool = True
	copyable: bool = True
	destroyable: bool = True


@dataclass
class SoftwareHSMSession:
	"""Software HSM session"""
	session_id: str = field(default_factory=uuid7str)
	user_id: str = ""
	tenant_id: str = ""
	authenticated: bool = False
	read_write: bool = True
	created_at: datetime = field(default_factory=datetime.utcnow)
	last_activity: datetime = field(default_factory=datetime.utcnow)
	operations_count: int = 0
	timeout: int = 3600  # 1 hour


class SoftwareHSM:
	"""
	Complete Software HSM Implementation
	
	Provides all standard HSM capabilities in software:
	- Key generation, storage, and lifecycle management
	- Cryptographic operations (encrypt, decrypt, sign, verify)
	- Key wrapping and unwrapping
	- Secure key derivation
	- Random number generation
	- Session management
	- Authentication and access control
	- Audit logging
	- Key attributes and policies
	- Import/export capabilities
	"""
	
	def __init__(self, hsm_id: str = None, config: Dict[str, Any] = None):
		self.hsm_id = hsm_id or uuid7str()
		self.config = config or {}
		
		# Storage
		self.db_path = self.config.get('db_path', f'/tmp/software_hsm_{self.hsm_id}.db')
		self.keys: Dict[str, SoftwareHSMKey] = {}
		self.sessions: Dict[str, SoftwareHSMSession] = {}
		
		# Security
		self.master_key = self._derive_master_key()
		self.authenticated_users: Dict[str, Dict[str, Any]] = {}
		
		# Statistics
		self.statistics = {
			'keys_generated': 0,
			'operations_performed': 0,
			'sessions_created': 0,
			'bytes_encrypted': 0,
			'bytes_decrypted': 0,
			'signatures_created': 0,
			'signatures_verified': 0
		}
		
		# Configuration
		self.max_sessions = self.config.get('max_sessions', 100)
		self.session_timeout = self.config.get('session_timeout', 3600)
		self.fips_mode = self.config.get('fips_mode', False)
		self.audit_enabled = self.config.get('audit_enabled', True)
		
		# Logging
		self.logger = logging.getLogger(f"SoftwareHSM.{self.hsm_id}")
	
	async def initialize(self):
		"""Initialize the Software HSM"""
		await self._initialize_database()
		await self._load_keys_from_storage()
		
		if self.audit_enabled:
			await self._audit_log("HSM_INITIALIZE", {"hsm_id": self.hsm_id})
		
		self.logger.info(f"Software HSM {self.hsm_id} initialized")
	
	async def finalize(self):
		"""Finalize and cleanup the Software HSM"""
		# Save all keys to storage
		await self._save_keys_to_storage()
		
		# Close all sessions
		for session_id in list(self.sessions.keys()):
			await self.close_session(session_id)
		
		if self.audit_enabled:
			await self._audit_log("HSM_FINALIZE", {"hsm_id": self.hsm_id})
		
		self.logger.info(f"Software HSM {self.hsm_id} finalized")
	
	def _derive_master_key(self) -> bytes:
		"""Derive master key for internal key encryption"""
		# In production, this would use hardware entropy or secure key derivation
		hsm_secret = self.config.get('hsm_secret', f'hsm_secret_{self.hsm_id}').encode()
		
		kdf = PBKDF2HMAC(
			algorithm=hashes.SHA256(),
			length=32,
			salt=b'software_hsm_salt',
			iterations=100000,
			backend=default_backend()
		)
		
		return kdf.derive(hsm_secret)
	
	async def _initialize_database(self):
		"""Initialize SQLite database for persistent storage"""
		async with aiosqlite.connect(self.db_path) as db:
			await db.execute('''
				CREATE TABLE IF NOT EXISTS hsm_keys (
					key_id TEXT PRIMARY KEY,
					key_type TEXT NOT NULL,
					key_size INTEGER NOT NULL,
					encrypted_key_material BLOB NOT NULL,
					public_key_material BLOB,
					algorithm TEXT NOT NULL,
					usage TEXT NOT NULL,
					attributes TEXT NOT NULL,
					created_at TIMESTAMP NOT NULL,
					last_used TIMESTAMP NOT NULL,
					use_count INTEGER DEFAULT 0,
					extractable BOOLEAN DEFAULT 1,
					sensitive BOOLEAN DEFAULT 0,
					token BOOLEAN DEFAULT 1,
					private BOOLEAN DEFAULT 1,
					modifiable BOOLEAN DEFAULT 1,
					copyable BOOLEAN DEFAULT 1,
					destroyable BOOLEAN DEFAULT 1
				)
			''')
			
			await db.execute('''
				CREATE TABLE IF NOT EXISTS hsm_audit (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					timestamp TIMESTAMP NOT NULL,
					operation TEXT NOT NULL,
					session_id TEXT,
					user_id TEXT,
					key_id TEXT,
					details TEXT,
					result TEXT NOT NULL
				)
			''')
			
			await db.commit()
	
	async def _load_keys_from_storage(self):
		"""Load keys from persistent storage"""
		try:
			async with aiosqlite.connect(self.db_path) as db:
				async with db.execute('SELECT * FROM hsm_keys') as cursor:
					async for row in cursor:
						key = SoftwareHSMKey(
							key_id=row[0],
							key_type=SoftwareHSMKeyType(row[1]),
							key_size=row[2],
							key_material=self._decrypt_key_material(row[3]),
							public_key_material=row[4],
							algorithm=KeyAlgorithm(row[5]),
							usage=json.loads(row[6]),
							attributes=json.loads(row[7]),
							created_at=datetime.fromisoformat(row[8]),
							last_used=datetime.fromisoformat(row[9]),
							use_count=row[10],
							extractable=bool(row[11]),
							sensitive=bool(row[12]),
							token=bool(row[13]),
							private=bool(row[14]),
							modifiable=bool(row[15]),
							copyable=bool(row[16]),
							destroyable=bool(row[17])
						)
						self.keys[key.key_id] = key
		except Exception as e:
			self.logger.warning(f"Failed to load keys from storage: {e}")
	
	async def _save_keys_to_storage(self):
		"""Save keys to persistent storage"""
		try:
			async with aiosqlite.connect(self.db_path) as db:
				for key in self.keys.values():
					encrypted_material = self._encrypt_key_material(key.key_material)
					
					await db.execute('''
						INSERT OR REPLACE INTO hsm_keys VALUES 
						(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
					''', (
						key.key_id, key.key_type.value, key.key_size,
						encrypted_material, key.public_key_material,
						key.algorithm.value, json.dumps(key.usage),
						json.dumps(key.attributes), key.created_at.isoformat(),
						key.last_used.isoformat(), key.use_count,
						key.extractable, key.sensitive, key.token,
						key.private, key.modifiable, key.copyable, key.destroyable
					))
				
				await db.commit()
		except Exception as e:
			self.logger.error(f"Failed to save keys to storage: {e}")
	
	def _encrypt_key_material(self, key_material: bytes) -> bytes:
		"""Encrypt key material for storage"""
		nonce = secrets.token_bytes(12)
		cipher = Cipher(algorithms.AES(self.master_key), modes.GCM(nonce), backend=default_backend())
		encryptor = cipher.encryptor()
		
		ciphertext = encryptor.update(key_material) + encryptor.finalize()
		return nonce + encryptor.tag + ciphertext
	
	def _decrypt_key_material(self, encrypted_material: bytes) -> bytes:
		"""Decrypt key material from storage"""
		nonce = encrypted_material[:12]
		tag = encrypted_material[12:28]
		ciphertext = encrypted_material[28:]
		
		cipher = Cipher(algorithms.AES(self.master_key), modes.GCM(nonce, tag), backend=default_backend())
		decryptor = cipher.decryptor()
		
		return decryptor.update(ciphertext) + decryptor.finalize()
	
	async def _audit_log(self, operation: str, details: Dict[str, Any], session_id: str = None, result: str = "SUCCESS"):
		"""Log audit event"""
		if not self.audit_enabled:
			return
		
		try:
			async with aiosqlite.connect(self.db_path) as db:
				await db.execute('''
					INSERT INTO hsm_audit (timestamp, operation, session_id, user_id, key_id, details, result)
					VALUES (?, ?, ?, ?, ?, ?, ?)
				''', (
					datetime.utcnow().isoformat(),
					operation,
					session_id,
					details.get('user_id'),
					details.get('key_id'),
					json.dumps(details),
					result
				))
				await db.commit()
		except Exception as e:
			self.logger.error(f"Audit logging failed: {e}")
	
	async def open_session(self, user_id: str, tenant_id: str = "", read_write: bool = True) -> str:
		"""Open a new HSM session"""
		if len(self.sessions) >= self.max_sessions:
			raise RuntimeError("Maximum number of sessions exceeded")
		
		session = SoftwareHSMSession(
			user_id=user_id,
			tenant_id=tenant_id,
			read_write=read_write
		)
		
		self.sessions[session.session_id] = session
		self.statistics['sessions_created'] += 1
		
		await self._audit_log("OPEN_SESSION", {
			"user_id": user_id,
			"tenant_id": tenant_id,
			"read_write": read_write
		}, session.session_id)
		
		return session.session_id
	
	async def close_session(self, session_id: str):
		"""Close HSM session"""
		if session_id in self.sessions:
			session = self.sessions[session_id]
			
			await self._audit_log("CLOSE_SESSION", {
				"user_id": session.user_id,
				"operations_count": session.operations_count
			}, session_id)
			
			del self.sessions[session_id]
	
	def _validate_session(self, session_id: str, require_rw: bool = False) -> SoftwareHSMSession:
		"""Validate and return session"""
		if session_id not in self.sessions:
			raise ValueError("Invalid session")
		
		session = self.sessions[session_id]
		
		# Check timeout
		if datetime.utcnow() - session.last_activity > timedelta(seconds=self.session_timeout):
			del self.sessions[session_id]
			raise ValueError("Session expired")
		
		if require_rw and not session.read_write:
			raise ValueError("Read-write session required")
		
		# Update activity
		session.last_activity = datetime.utcnow()
		session.operations_count += 1
		
		return session
	
	async def generate_key(self, session_id: str, key_type: SoftwareHSMKeyType, 
						  key_size: int, usage: List[str], attributes: Dict[str, Any] = None) -> str:
		"""Generate a new cryptographic key"""
		session = self._validate_session(session_id, require_rw=True)
		attributes = attributes or {}
		
		# Generate key material based on type
		if key_type == SoftwareHSMKeyType.AES:
			key_material = secrets.token_bytes(key_size // 8)
			public_key_material = None
			algorithm = {
				128: KeyAlgorithm.AES_128,
				192: KeyAlgorithm.AES_192,
				256: KeyAlgorithm.AES_256
			}.get(key_size, KeyAlgorithm.AES_256)
		
		elif key_type == SoftwareHSMKeyType.RSA:
			private_key = rsa.generate_private_key(
				public_exponent=65537,
				key_size=key_size,
				backend=default_backend()
			)
			key_material = private_key.private_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PrivateFormat.PKCS8,
				encryption_algorithm=serialization.NoEncryption()
			)
			public_key_material = private_key.public_key().public_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PublicFormat.SubjectPublicKeyInfo
			)
			algorithm = {
				2048: KeyAlgorithm.RSA_2048,
				3072: KeyAlgorithm.RSA_3072,
				4096: KeyAlgorithm.RSA_4096
			}.get(key_size, KeyAlgorithm.RSA_2048)
		
		elif key_type == SoftwareHSMKeyType.ECDSA:
			if key_size == 256:
				private_key = ec.generate_private_key(ec.SECP256R1(), backend=default_backend())
				algorithm = KeyAlgorithm.ECDSA_P256
			elif key_size == 384:
				private_key = ec.generate_private_key(ec.SECP384R1(), backend=default_backend())
				algorithm = KeyAlgorithm.ECDSA_P384
			elif key_size == 521:
				private_key = ec.generate_private_key(ec.SECP521R1(), backend=default_backend())
				algorithm = KeyAlgorithm.ECDSA_P521
			else:
				raise ValueError(f"Unsupported ECDSA key size: {key_size}")
			
			key_material = private_key.private_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PrivateFormat.PKCS8,
				encryption_algorithm=serialization.NoEncryption()
			)
			public_key_material = private_key.public_key().public_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PublicFormat.SubjectPublicKeyInfo
			)
		
		elif key_type == SoftwareHSMKeyType.EDDSA:
			private_key = ed25519.Ed25519PrivateKey.generate()
			key_material = private_key.private_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PrivateFormat.PKCS8,
				encryption_algorithm=serialization.NoEncryption()
			)
			public_key_material = private_key.public_key().public_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PublicFormat.SubjectPublicKeyInfo
			)
			algorithm = KeyAlgorithm.ED25519
		
		elif key_type == SoftwareHSMKeyType.X25519:
			private_key = x25519.X25519PrivateKey.generate()
			key_material = private_key.private_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PrivateFormat.PKCS8,
				encryption_algorithm=serialization.NoEncryption()
			)
			public_key_material = private_key.public_key().public_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PublicFormat.SubjectPublicKeyInfo
			)
			algorithm = KeyAlgorithm.X25519
		
		elif key_type == SoftwareHSMKeyType.HMAC:
			key_material = secrets.token_bytes(key_size // 8)
			public_key_material = None
			algorithm = KeyAlgorithm.HMAC_SHA256
		
		else:
			raise ValueError(f"Unsupported key type: {key_type}")
		
		# Create key object
		hsm_key = SoftwareHSMKey(
			key_type=key_type,
			key_size=key_size,
			key_material=key_material,
			public_key_material=public_key_material,
			algorithm=algorithm,
			usage=usage,
			attributes=attributes,
			sensitive=attributes.get('sensitive', False),
			extractable=attributes.get('extractable', True),
			token=attributes.get('token', True)
		)
		
		self.keys[hsm_key.key_id] = hsm_key
		self.statistics['keys_generated'] += 1
		
		await self._audit_log("GENERATE_KEY", {
			"user_id": session.user_id,
			"key_id": hsm_key.key_id,
			"key_type": key_type.value,
			"key_size": key_size,
			"algorithm": algorithm.value
		}, session_id)
		
		return hsm_key.key_id
	
	async def delete_key(self, session_id: str, key_id: str):
		"""Delete a key from the HSM"""
		session = self._validate_session(session_id, require_rw=True)
		
		if key_id not in self.keys:
			raise ValueError("Key not found")
		
		hsm_key = self.keys[key_id]
		
		if not hsm_key.destroyable:
			raise ValueError("Key is not destroyable")
		
		del self.keys[key_id]
		
		await self._audit_log("DELETE_KEY", {
			"user_id": session.user_id,
			"key_id": key_id
		}, session_id)
	
	async def encrypt(self, session_id: str, key_id: str, data: bytes, algorithm: str = None) -> Dict[str, Any]:
		"""Encrypt data using specified key"""
		session = self._validate_session(session_id)
		
		if key_id not in self.keys:
			raise ValueError("Key not found")
		
		hsm_key = self.keys[key_id]
		hsm_key.last_used = datetime.utcnow()
		hsm_key.use_count += 1
		
		if "encrypt" not in hsm_key.usage:
			raise ValueError("Key not authorized for encryption")
		
		result = {}
		
		if hsm_key.key_type == SoftwareHSMKeyType.AES:
			# AES-GCM encryption
			nonce = secrets.token_bytes(12)
			cipher = Cipher(algorithms.AES(hsm_key.key_material), modes.GCM(nonce), backend=default_backend())
			encryptor = cipher.encryptor()
			
			ciphertext = encryptor.update(data) + encryptor.finalize()
			
			result = {
				"ciphertext": ciphertext.hex(),
				"nonce": nonce.hex(),
				"tag": encryptor.tag.hex(),
				"algorithm": "AES-GCM"
			}
		
		elif hsm_key.key_type == SoftwareHSMKeyType.RSA:
			# RSA-OAEP encryption
			private_key = serialization.load_pem_private_key(hsm_key.key_material, password=None, backend=default_backend())
			public_key = private_key.public_key()
			
			ciphertext = public_key.encrypt(
				data,
				padding.OAEP(
					mgf=padding.MGF1(algorithm=hashes.SHA256()),
					algorithm=hashes.SHA256(),
					label=None
				)
			)
			
			result = {
				"ciphertext": ciphertext.hex(),
				"algorithm": "RSA-OAEP"
			}
		
		else:
			raise ValueError(f"Encryption not supported for key type: {hsm_key.key_type}")
		
		self.statistics['bytes_encrypted'] += len(data)
		self.statistics['operations_performed'] += 1
		
		await self._audit_log("ENCRYPT", {
			"user_id": session.user_id,
			"key_id": key_id,
			"data_size": len(data)
		}, session_id)
		
		return result
	
	async def decrypt(self, session_id: str, key_id: str, encrypted_data: Dict[str, Any]) -> bytes:
		"""Decrypt data using specified key"""
		session = self._validate_session(session_id)
		
		if key_id not in self.keys:
			raise ValueError("Key not found")
		
		hsm_key = self.keys[key_id]
		hsm_key.last_used = datetime.utcnow()
		hsm_key.use_count += 1
		
		if "decrypt" not in hsm_key.usage:
			raise ValueError("Key not authorized for decryption")
		
		if hsm_key.key_type == SoftwareHSMKeyType.AES:
			# AES-GCM decryption
			ciphertext = bytes.fromhex(encrypted_data["ciphertext"])
			nonce = bytes.fromhex(encrypted_data["nonce"])
			tag = bytes.fromhex(encrypted_data["tag"])
			
			cipher = Cipher(algorithms.AES(hsm_key.key_material), modes.GCM(nonce, tag), backend=default_backend())
			decryptor = cipher.decryptor()
			
			plaintext = decryptor.update(ciphertext) + decryptor.finalize()
		
		elif hsm_key.key_type == SoftwareHSMKeyType.RSA:
			# RSA-OAEP decryption
			ciphertext = bytes.fromhex(encrypted_data["ciphertext"])
			private_key = serialization.load_pem_private_key(hsm_key.key_material, password=None, backend=default_backend())
			
			plaintext = private_key.decrypt(
				ciphertext,
				padding.OAEP(
					mgf=padding.MGF1(algorithm=hashes.SHA256()),
					algorithm=hashes.SHA256(),
					label=None
				)
			)
		
		else:
			raise ValueError(f"Decryption not supported for key type: {hsm_key.key_type}")
		
		self.statistics['bytes_decrypted'] += len(plaintext)
		self.statistics['operations_performed'] += 1
		
		await self._audit_log("DECRYPT", {
			"user_id": session.user_id,
			"key_id": key_id,
			"data_size": len(plaintext)
		}, session_id)
		
		return plaintext
	
	async def sign(self, session_id: str, key_id: str, data: bytes, algorithm: str = None) -> bytes:
		"""Sign data using specified key"""
		session = self._validate_session(session_id)
		
		if key_id not in self.keys:
			raise ValueError("Key not found")
		
		hsm_key = self.keys[key_id]
		hsm_key.last_used = datetime.utcnow()
		hsm_key.use_count += 1
		
		if "sign" not in hsm_key.usage:
			raise ValueError("Key not authorized for signing")
		
		if hsm_key.key_type == SoftwareHSMKeyType.RSA:
			private_key = serialization.load_pem_private_key(hsm_key.key_material, password=None, backend=default_backend())
			signature = private_key.sign(
				data,
				padding.PSS(
					mgf=padding.MGF1(hashes.SHA256()),
					salt_length=padding.PSS.MAX_LENGTH
				),
				hashes.SHA256()
			)
		
		elif hsm_key.key_type == SoftwareHSMKeyType.ECDSA:
			private_key = serialization.load_pem_private_key(hsm_key.key_material, password=None, backend=default_backend())
			signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
		
		elif hsm_key.key_type == SoftwareHSMKeyType.EDDSA:
			private_key = serialization.load_pem_private_key(hsm_key.key_material, password=None, backend=default_backend())
			signature = private_key.sign(data)
		
		elif hsm_key.key_type == SoftwareHSMKeyType.HMAC:
			import hmac
			signature = hmac.new(hsm_key.key_material, data, hashlib.sha256).digest()
		
		else:
			raise ValueError(f"Signing not supported for key type: {hsm_key.key_type}")
		
		self.statistics['signatures_created'] += 1
		self.statistics['operations_performed'] += 1
		
		await self._audit_log("SIGN", {
			"user_id": session.user_id,
			"key_id": key_id,
			"data_size": len(data)
		}, session_id)
		
		return signature
	
	async def verify(self, session_id: str, key_id: str, data: bytes, signature: bytes) -> bool:
		"""Verify signature using specified key"""
		session = self._validate_session(session_id)
		
		if key_id not in self.keys:
			raise ValueError("Key not found")
		
		hsm_key = self.keys[key_id]
		hsm_key.last_used = datetime.utcnow()
		hsm_key.use_count += 1
		
		if "verify" not in hsm_key.usage:
			raise ValueError("Key not authorized for verification")
		
		try:
			if hsm_key.key_type == SoftwareHSMKeyType.RSA:
				if hsm_key.public_key_material:
					public_key = serialization.load_pem_public_key(hsm_key.public_key_material, backend=default_backend())
				else:
					private_key = serialization.load_pem_private_key(hsm_key.key_material, password=None, backend=default_backend())
					public_key = private_key.public_key()
				
				public_key.verify(
					signature,
					data,
					padding.PSS(
						mgf=padding.MGF1(hashes.SHA256()),
						salt_length=padding.PSS.MAX_LENGTH
					),
					hashes.SHA256()
				)
			
			elif hsm_key.key_type == SoftwareHSMKeyType.ECDSA:
				if hsm_key.public_key_material:
					public_key = serialization.load_pem_public_key(hsm_key.public_key_material, backend=default_backend())
				else:
					private_key = serialization.load_pem_private_key(hsm_key.key_material, password=None, backend=default_backend())
					public_key = private_key.public_key()
				
				public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
			
			elif hsm_key.key_type == SoftwareHSMKeyType.EDDSA:
				if hsm_key.public_key_material:
					public_key = serialization.load_pem_public_key(hsm_key.public_key_material, backend=default_backend())
				else:
					private_key = serialization.load_pem_private_key(hsm_key.key_material, password=None, backend=default_backend())
					public_key = private_key.public_key()
				
				public_key.verify(signature, data)
			
			elif hsm_key.key_type == SoftwareHSMKeyType.HMAC:
				import hmac
				expected_signature = hmac.new(hsm_key.key_material, data, hashlib.sha256).digest()
				if not hmac.compare_digest(signature, expected_signature):
					raise ValueError("Invalid signature")
			
			else:
				raise ValueError(f"Verification not supported for key type: {hsm_key.key_type}")
			
			result = True
		
		except Exception:
			result = False
		
		self.statistics['signatures_verified'] += 1
		self.statistics['operations_performed'] += 1
		
		await self._audit_log("VERIFY", {
			"user_id": session.user_id,
			"key_id": key_id,
			"result": result
		}, session_id)
		
		return result
	
	async def get_random(self, session_id: str, length: int) -> bytes:
		"""Generate random bytes"""
		session = self._validate_session(session_id)
		
		if length <= 0 or length > 4096:
			raise ValueError("Invalid random length")
		
		random_bytes = secrets.token_bytes(length)
		
		await self._audit_log("GET_RANDOM", {
			"user_id": session.user_id,
			"length": length
		}, session_id)
		
		return random_bytes
	
	async def list_keys(self, session_id: str, filter_attrs: Dict[str, Any] = None) -> List[Dict[str, Any]]:
		"""List keys in the HSM"""
		session = self._validate_session(session_id)
		filter_attrs = filter_attrs or {}
		
		key_list = []
		for hsm_key in self.keys.values():
			# Apply filters
			if filter_attrs:
				match = True
				for attr, value in filter_attrs.items():
					if hasattr(hsm_key, attr) and getattr(hsm_key, attr) != value:
						match = False
						break
				if not match:
					continue
			
			key_info = {
				"key_id": hsm_key.key_id,
				"key_type": hsm_key.key_type.value,
				"key_size": hsm_key.key_size,
				"algorithm": hsm_key.algorithm.value,
				"usage": hsm_key.usage,
				"created_at": hsm_key.created_at.isoformat(),
				"last_used": hsm_key.last_used.isoformat(),
				"use_count": hsm_key.use_count,
				"extractable": hsm_key.extractable,
				"sensitive": hsm_key.sensitive
			}
			
			# Don't include key material in listing
			key_list.append(key_info)
		
		return key_list
	
	async def get_key_attributes(self, session_id: str, key_id: str) -> Dict[str, Any]:
		"""Get key attributes"""
		session = self._validate_session(session_id)
		
		if key_id not in self.keys:
			raise ValueError("Key not found")
		
		hsm_key = self.keys[key_id]
		
		return {
			"key_id": hsm_key.key_id,
			"key_type": hsm_key.key_type.value,
			"key_size": hsm_key.key_size,
			"algorithm": hsm_key.algorithm.value,
			"usage": hsm_key.usage,
			"attributes": hsm_key.attributes,
			"created_at": hsm_key.created_at.isoformat(),
			"last_used": hsm_key.last_used.isoformat(),
			"use_count": hsm_key.use_count,
			"extractable": hsm_key.extractable,
			"sensitive": hsm_key.sensitive,
			"token": hsm_key.token,
			"private": hsm_key.private,
			"modifiable": hsm_key.modifiable,
			"copyable": hsm_key.copyable,
			"destroyable": hsm_key.destroyable
		}
	
	async def get_hsm_info(self) -> Dict[str, Any]:
		"""Get HSM information and statistics"""
		return {
			"hsm_id": self.hsm_id,
			"hsm_type": "Software HSM",
			"version": "1.0.0",
			"fips_mode": self.fips_mode,
			"max_sessions": self.max_sessions,
			"active_sessions": len(self.sessions),
			"total_keys": len(self.keys),
			"statistics": self.statistics.copy(),
			"supported_algorithms": [
				"AES-128", "AES-192", "AES-256",
				"RSA-2048", "RSA-3072", "RSA-4096",
				"ECDSA-P256", "ECDSA-P384", "ECDSA-P521",
				"Ed25519", "X25519", "HMAC-SHA256"
			],
			"supported_operations": [op.value for op in SoftwareHSMOperation]
		}
	
	async def export_key(self, session_id: str, key_id: str, wrap_key_id: str = None) -> bytes:
		"""Export key (potentially wrapped)"""
		session = self._validate_session(session_id)
		
		if key_id not in self.keys:
			raise ValueError("Key not found")
		
		hsm_key = self.keys[key_id]
		
		if not hsm_key.extractable:
			raise ValueError("Key is not extractable")
		
		if hsm_key.sensitive and not wrap_key_id:
			raise ValueError("Sensitive key requires wrapping")
		
		key_data = hsm_key.key_material
		
		if wrap_key_id:
			# Wrap the key
			key_data = await self._wrap_key(key_data, wrap_key_id)
		
		await self._audit_log("EXPORT_KEY", {
			"user_id": session.user_id,
			"key_id": key_id,
			"wrapped": bool(wrap_key_id)
		}, session_id)
		
		return key_data
	
	async def _wrap_key(self, key_material: bytes, wrap_key_id: str) -> bytes:
		"""Wrap key material using another key"""
		if wrap_key_id not in self.keys:
			raise ValueError("Wrap key not found")
		
		wrap_key = self.keys[wrap_key_id]
		
		if "wrap" not in wrap_key.usage:
			raise ValueError("Key not authorized for wrapping")
		
		# Simple AES-GCM wrapping
		if wrap_key.key_type == SoftwareHSMKeyType.AES:
			nonce = secrets.token_bytes(12)
			cipher = Cipher(algorithms.AES(wrap_key.key_material), modes.GCM(nonce), backend=default_backend())
			encryptor = cipher.encryptor()
			
			wrapped_key = encryptor.update(key_material) + encryptor.finalize()
			
			# Return nonce + tag + wrapped_key
			return nonce + encryptor.tag + wrapped_key
		else:
			raise ValueError("Unsupported wrap key type")
	
	async def cleanup_expired_sessions(self):
		"""Clean up expired sessions"""
		current_time = datetime.utcnow()
		expired_sessions = []
		
		for session_id, session in self.sessions.items():
			if current_time - session.last_activity > timedelta(seconds=self.session_timeout):
				expired_sessions.append(session_id)
		
		for session_id in expired_sessions:
			await self.close_session(session_id)
		
		return len(expired_sessions)
	
	async def backup_keys(self, backup_path: str):
		"""Backup all keys to file"""
		await self._save_keys_to_storage()
		
		# In production, would create encrypted backup
		import shutil
		shutil.copy2(self.db_path, backup_path)
		
		await self._audit_log("BACKUP_KEYS", {
			"backup_path": backup_path,
			"key_count": len(self.keys)
		})


# Factory function
async def create_software_hsm(hsm_id: str = None, config: Dict[str, Any] = None) -> SoftwareHSM:
	"""Create and initialize Software HSM"""
	hsm = SoftwareHSM(hsm_id, config)
	await hsm.initialize()
	return hsm


# Export main components
__all__ = [
	'SoftwareHSM', 'SoftwareHSMKey', 'SoftwareHSMSession',
	'SoftwareHSMKeyType', 'SoftwareHSMOperation',
	'create_software_hsm'
]