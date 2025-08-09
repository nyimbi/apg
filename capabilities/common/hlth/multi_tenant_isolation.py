#!/usr/bin/env python3
"""
APG System Health Management (HLTH) - Multi-Tenant Isolation
Advanced multi-tenant isolation and security for enterprise deployments

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import hashlib
import hmac
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict

from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str

from .models import HealthMetric, HealthAlert, SystemComponent
from .enterprise_features import TenantTier, TenantConfiguration


class IsolationLevel(Enum):
	"""Multi-tenant isolation levels"""
	SHARED = "shared"              # Shared infrastructure, logical separation
	DEDICATED = "dedicated"        # Dedicated infrastructure per tenant
	HYBRID = "hybrid"             # Mix of shared and dedicated resources
	SOVEREIGN = "sovereign"       # Complete sovereignty and control


class DataClassification(Enum):
	"""Data classification levels"""
	PUBLIC = "public"
	INTERNAL = "internal"
	CONFIDENTIAL = "confidential"
	RESTRICTED = "restricted"


@dataclass
class TenantIsolationPolicy:
	"""Tenant isolation policy configuration"""
	tenant_id: str
	isolation_level: IsolationLevel
	data_classification: DataClassification
	network_isolation: bool = True
	compute_isolation: bool = False
	storage_isolation: bool = True
	encryption_at_rest: bool = True
	encryption_in_transit: bool = True
	access_logging: bool = True
	data_residency_requirements: List[str] = field(default_factory=list)
	allowed_integrations: List[str] = field(default_factory=list)
	custom_security_policies: Dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TenantResource:
	"""Isolated tenant resource representation"""
	resource_id: str
	tenant_id: str
	resource_type: str
	namespace: str
	isolation_tags: List[str]
	access_policies: Dict[str, Any]
	encryption_keys: Dict[str, str]
	created_at: datetime = field(default_factory=datetime.utcnow)
	metadata: Dict[str, Any] = field(default_factory=dict)


class TenantIsolationManager:
	"""Advanced multi-tenant isolation manager"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self.tenant_policies: Dict[str, TenantIsolationPolicy] = {}
		self.tenant_resources: Dict[str, List[TenantResource]] = defaultdict(list)
		self.tenant_namespaces: Dict[str, str] = {}
		self.encryption_keys: Dict[str, Dict[str, str]] = defaultdict(dict)
		self.access_tokens: Dict[str, Dict[str, str]] = defaultdict(dict)
		self.tenant_boundaries: Dict[str, Set[str]] = defaultdict(set)
		
		# Cross-tenant access audit log
		self.cross_tenant_access_log: List[Dict[str, Any]] = []
		
		# Initialize encryption service
		self._init_encryption_service()
	
	def _init_encryption_service(self):
		"""Initialize comprehensive encryption service for tenant data"""
		try:
			# Initialize with secure key management
			self.master_key = self.config.get('master_key')
			if not self.master_key or self.master_key == 'dev-master-key-not-for-production':
				# Generate a secure master key if none provided
				self.master_key = self._generate_secure_master_key()
				print("[HLTH-ISO] Generated secure master key for encryption service")
			
			# Initialize encryption algorithms and configurations
			self.encryption_algorithms = {
				'AES256': {
					'algorithm': 'AES',
					'key_size': 256,
					'mode': 'GCM',
					'iv_size': 16
				},
				'ChaCha20': {
					'algorithm': 'ChaCha20-Poly1305',
					'key_size': 256,
					'nonce_size': 12
				}
			}
			
			# Set default algorithm
			self.default_algorithm = self.config.get('encryption_algorithm', 'AES256')
			
			# Initialize key derivation parameters
			self.kdf_iterations = 100000  # PBKDF2 iterations
			self.salt_size = 32
			
		except Exception as e:
			print(f"[HLTH-ISO] Error initializing encryption service: {str(e)}")
			raise RuntimeError(f"Failed to initialize encryption service: {str(e)}")
	
	async def create_tenant_isolation(self, tenant_id: str, 
									  tenant_config: TenantConfiguration,
									  isolation_config: Dict[str, Any]) -> TenantIsolationPolicy:
		"""Create comprehensive tenant isolation"""
		try:
			# Determine isolation level based on tenant tier
			isolation_level = self._determine_isolation_level(tenant_config.tier)
			
			# Create isolation policy
			isolation_policy = TenantIsolationPolicy(
				tenant_id=tenant_id,
				isolation_level=isolation_level,
				data_classification=DataClassification(
					isolation_config.get('data_classification', 'internal')
				),
				network_isolation=isolation_config.get('network_isolation', True),
				compute_isolation=isolation_config.get('compute_isolation', 
													   isolation_level in [IsolationLevel.DEDICATED, IsolationLevel.SOVEREIGN]),
				storage_isolation=isolation_config.get('storage_isolation', True),
				encryption_at_rest=isolation_config.get('encryption_at_rest', True),
				encryption_in_transit=isolation_config.get('encryption_in_transit', True),
				access_logging=isolation_config.get('access_logging', True),
				data_residency_requirements=isolation_config.get('data_residency_requirements', []),
				allowed_integrations=isolation_config.get('allowed_integrations', []),
				custom_security_policies=isolation_config.get('custom_security_policies', {})
			)
			
			# Store isolation policy
			self.tenant_policies[tenant_id] = isolation_policy
			
			# Create tenant namespace
			await self._create_tenant_namespace(tenant_id, isolation_policy)
			
			# Generate tenant encryption keys
			await self._generate_tenant_encryption_keys(tenant_id, isolation_policy)
			
			# Setup network isolation if required
			if isolation_policy.network_isolation:
				await self._setup_network_isolation(tenant_id, isolation_policy)
			
			# Setup storage isolation if required
			if isolation_policy.storage_isolation:
				await self._setup_storage_isolation(tenant_id, isolation_policy)
			
			# Setup access controls
			await self._setup_tenant_access_controls(tenant_id, isolation_policy)
			
			return isolation_policy
			
		except Exception as e:
			raise RuntimeError(f"Failed to create tenant isolation: {str(e)}")
	
	def _determine_isolation_level(self, tenant_tier: TenantTier) -> IsolationLevel:
		"""Determine isolation level based on tenant tier"""
		tier_isolation_map = {
			TenantTier.BASIC: IsolationLevel.SHARED,
			TenantTier.PROFESSIONAL: IsolationLevel.SHARED,
			TenantTier.ENTERPRISE: IsolationLevel.HYBRID,
			TenantTier.ENTERPRISE_PLUS: IsolationLevel.DEDICATED
		}
		return tier_isolation_map.get(tenant_tier, IsolationLevel.SHARED)
	
	async def _create_tenant_namespace(self, tenant_id: str, 
									   policy: TenantIsolationPolicy) -> None:
		"""Create isolated namespace for tenant resources"""
		# Generate unique namespace
		namespace = f"hlth-{tenant_id[:8]}-{policy.isolation_level.value}"
		self.tenant_namespaces[tenant_id] = namespace
		
		# In production, this would create actual Kubernetes namespaces or similar
		print(f"Created namespace '{namespace}' for tenant {tenant_id}")
	
	async def _generate_tenant_encryption_keys(self, tenant_id: str,
											   policy: TenantIsolationPolicy) -> None:
		"""Generate tenant-specific encryption keys"""
		if policy.encryption_at_rest or policy.encryption_in_transit:
			# Generate tenant-specific encryption keys
			tenant_seed = f"{tenant_id}-{self.master_key}-{datetime.utcnow().isoformat()}"
			
			# Data encryption key
			data_key = hashlib.pbkdf2_hmac('sha256', 
										   tenant_seed.encode(), 
										   tenant_id.encode(), 
										   100000)
			
			# API access key
			api_key = hmac.new(self.master_key.encode(), 
							   f"api-{tenant_id}".encode(), 
							   hashlib.sha256).hexdigest()
			
			# Store keys securely (in production, use proper KMS)
			self.encryption_keys[tenant_id] = {
				'data_key': data_key.hex(),
				'api_key': api_key,
				'created_at': datetime.utcnow().isoformat()
			}
	
	async def _setup_network_isolation(self, tenant_id: str,
									   policy: TenantIsolationPolicy) -> None:
		"""Setup network-level isolation for tenant"""
		if policy.isolation_level in [IsolationLevel.DEDICATED, IsolationLevel.SOVEREIGN]:
			# Create dedicated VPC/network for tenant
			network_config = {
				'tenant_id': tenant_id,
				'network_type': 'dedicated',
				'vpc_cidr': f"10.{hash(tenant_id) % 255}.0.0/16",
				'subnets': [
					{'type': 'public', 'cidr': f"10.{hash(tenant_id) % 255}.1.0/24"},
					{'type': 'private', 'cidr': f"10.{hash(tenant_id) % 255}.2.0/24"}
				],
				'security_groups': {
					'health_api': {'ports': [443, 80], 'protocol': 'tcp'},
					'health_internal': {'ports': [8080, 9090], 'protocol': 'tcp'}
				}
			}
		else:
			# Shared network with tenant-specific security groups
			network_config = {
				'tenant_id': tenant_id,
				'network_type': 'shared',
				'security_group': f"hlth-sg-{tenant_id[:8]}",
				'network_policies': [
					f"deny-cross-tenant-{tenant_id}",
					f"allow-internal-{tenant_id}"
				]
			}
		
		# Store network configuration
		self.tenant_resources[tenant_id].append(
			TenantResource(
				resource_id=f"network-{tenant_id}",
				tenant_id=tenant_id,
				resource_type='network',
				namespace=self.tenant_namespaces[tenant_id],
				isolation_tags=['network', 'security'],
				access_policies={'network_config': network_config},
				encryption_keys={}
			)
		)
	
	async def _setup_storage_isolation(self, tenant_id: str,
									   policy: TenantIsolationPolicy) -> None:
		"""Setup storage-level isolation for tenant"""
		storage_config = {
			'tenant_id': tenant_id,
			'encryption_at_rest': policy.encryption_at_rest,
			'encryption_key_id': self.encryption_keys[tenant_id].get('data_key'),
			'data_classification': policy.data_classification.value,
			'retention_policies': {},
			'access_patterns': ['tenant_only']
		}
		
		if policy.isolation_level in [IsolationLevel.DEDICATED, IsolationLevel.SOVEREIGN]:
			storage_config.update({
				'storage_type': 'dedicated',
				'dedicated_volumes': [
					f"hlth-metrics-{tenant_id}",
					f"hlth-alerts-{tenant_id}",
					f"hlth-reports-{tenant_id}"
				]
			})
		else:
			storage_config.update({
				'storage_type': 'shared',
				'tenant_prefix': f"hlth/{tenant_id}/",
				'isolation_method': 'path_based'
			})
		
		# Add data residency requirements if specified
		if policy.data_residency_requirements:
			storage_config['data_residency'] = {
				'allowed_regions': policy.data_residency_requirements,
				'enforcement': 'strict'
			}
		
		# Store storage configuration
		self.tenant_resources[tenant_id].append(
			TenantResource(
				resource_id=f"storage-{tenant_id}",
				tenant_id=tenant_id,
				resource_type='storage',
				namespace=self.tenant_namespaces[tenant_id],
				isolation_tags=['storage', 'encryption'],
				access_policies={'storage_config': storage_config},
				encryption_keys=self.encryption_keys[tenant_id]
			)
		)
	
	async def _setup_tenant_access_controls(self, tenant_id: str,
											policy: TenantIsolationPolicy) -> None:
		"""Setup comprehensive access controls for tenant"""
		access_config = {
			'tenant_id': tenant_id,
			'access_logging': policy.access_logging,
			'authentication': {
				'methods': ['api_key', 'jwt', 'oauth2'],
				'mfa_required': policy.isolation_level in [IsolationLevel.DEDICATED, IsolationLevel.SOVEREIGN],
				'session_timeout': 3600  # 1 hour
			},
			'authorization': {
				'model': 'rbac',  # Role-Based Access Control
				'tenant_isolation': True,
				'cross_tenant_access': False
			},
			'api_rate_limiting': {
				'enabled': True,
				'requests_per_minute': 1000,
				'burst_limit': 1500
			}
		}
		
		# Generate tenant-specific access token
		access_token = self._generate_tenant_access_token(tenant_id, policy)
		self.access_tokens[tenant_id]['primary'] = access_token
		
		# Store access configuration
		self.tenant_resources[tenant_id].append(
			TenantResource(
				resource_id=f"access-{tenant_id}",
				tenant_id=tenant_id,
				resource_type='access_control',
				namespace=self.tenant_namespaces[tenant_id],
				isolation_tags=['access', 'security'],
				access_policies={'access_config': access_config},
				encryption_keys={'access_token': access_token}
			)
		)
	
	def _generate_tenant_access_token(self, tenant_id: str, 
									  policy: TenantIsolationPolicy) -> str:
		"""Generate secure tenant access token"""
		token_payload = {
			'tenant_id': tenant_id,
			'isolation_level': policy.isolation_level.value,
			'permissions': ['health.read', 'health.write', 'health.admin'],
			'issued_at': datetime.utcnow().timestamp(),
			'expires_at': (datetime.utcnow() + timedelta(days=365)).timestamp()
		}
		
		# In production, use proper JWT library
		token_string = json.dumps(token_payload)
		signature = hmac.new(
			self.encryption_keys[tenant_id]['api_key'].encode(),
			token_string.encode(),
			hashlib.sha256
		).hexdigest()
		
		return f"{token_string}.{signature}"
	
	async def enforce_tenant_boundaries(self, requesting_tenant_id: str,
									   target_tenant_id: str,
									   resource_type: str,
									   operation: str) -> Dict[str, Any]:
		"""Enforce tenant boundary isolation"""
		try:
			# Same tenant access is always allowed
			if requesting_tenant_id == target_tenant_id:
				return {'allowed': True, 'reason': 'Same tenant access'}
			
			# Get tenant isolation policies
			requesting_policy = self.tenant_policies.get(requesting_tenant_id)
			target_policy = self.tenant_policies.get(target_tenant_id)
			
			if not requesting_policy or not target_policy:
				return {'allowed': False, 'reason': 'Tenant policy not found'}
			
			# Check if cross-tenant access is explicitly allowed
			allowed_integrations = requesting_policy.allowed_integrations
			if target_tenant_id not in allowed_integrations:
				# Log cross-tenant access attempt
				await self._log_cross_tenant_access_attempt(
					requesting_tenant_id, target_tenant_id, resource_type, operation, False
				)
				
				return {
					'allowed': False,
					'reason': 'Cross-tenant access not permitted',
					'requesting_tenant': requesting_tenant_id,
					'target_tenant': target_tenant_id,
					'resource_type': resource_type,
					'operation': operation
				}
			
			# Additional checks for sovereign isolation
			if target_policy.isolation_level == IsolationLevel.SOVEREIGN:
				return {
					'allowed': False,
					'reason': 'Target tenant has sovereign isolation',
					'target_isolation_level': IsolationLevel.SOVEREIGN.value
				}
			
			# Check data classification compatibility
			if not self._check_data_classification_compatibility(
				requesting_policy.data_classification,
				target_policy.data_classification
			):
				return {
					'allowed': False,
					'reason': 'Data classification incompatible',
					'requesting_classification': requesting_policy.data_classification.value,
					'target_classification': target_policy.data_classification.value
				}
			
			# Log successful cross-tenant access
			await self._log_cross_tenant_access_attempt(
				requesting_tenant_id, target_tenant_id, resource_type, operation, True
			)
			
			return {
				'allowed': True,
				'reason': 'Cross-tenant access permitted',
				'conditions': [
					'Access logged for audit',
					'Rate limiting applied',
					'Encryption required'
				]
			}
			
		except Exception as e:
			return {
				'allowed': False,
				'reason': f'Boundary check failed: {str(e)}',
				'timestamp': datetime.utcnow().isoformat()
			}
	
	def _check_data_classification_compatibility(self, 
												 requesting_class: DataClassification,
												 target_class: DataClassification) -> bool:
		"""Check if data classifications are compatible for cross-tenant access"""
		# Classification hierarchy: PUBLIC < INTERNAL < CONFIDENTIAL < RESTRICTED
		classification_levels = {
			DataClassification.PUBLIC: 0,
			DataClassification.INTERNAL: 1,
			DataClassification.CONFIDENTIAL: 2,
			DataClassification.RESTRICTED: 3
		}
		
		requesting_level = classification_levels[requesting_class]
		target_level = classification_levels[target_class]
		
		# Can only access data at same or lower classification level
		return requesting_level >= target_level
	
	async def _log_cross_tenant_access_attempt(self, requesting_tenant_id: str,
											   target_tenant_id: str,
											   resource_type: str,
											   operation: str,
											   allowed: bool) -> None:
		"""Log cross-tenant access attempts for security audit"""
		access_log = {
			'timestamp': datetime.utcnow().isoformat(),
			'requesting_tenant_id': requesting_tenant_id,
			'target_tenant_id': target_tenant_id,
			'resource_type': resource_type,
			'operation': operation,
			'allowed': allowed,
			'security_event': True,
			'audit_required': True
		}
		
		self.cross_tenant_access_log.append(access_log)
		
		# In production, this would integrate with SIEM/security monitoring
		print(f"Cross-tenant access: {requesting_tenant_id} -> {target_tenant_id} "
			  f"({resource_type}.{operation}): {'ALLOWED' if allowed else 'DENIED'}")
	
	async def get_tenant_isolation_status(self, tenant_id: str) -> Dict[str, Any]:
		"""Get comprehensive tenant isolation status"""
		try:
			policy = self.tenant_policies.get(tenant_id)
			if not policy:
				return {'error': f'Tenant {tenant_id} not found'}
			
			resources = self.tenant_resources.get(tenant_id, [])
			namespace = self.tenant_namespaces.get(tenant_id, 'unknown')
			encryption_keys = self.encryption_keys.get(tenant_id, {})
			
			return {
				'tenant_id': tenant_id,
				'isolation_policy': {
					'isolation_level': policy.isolation_level.value,
					'data_classification': policy.data_classification.value,
					'network_isolation': policy.network_isolation,
					'compute_isolation': policy.compute_isolation,
					'storage_isolation': policy.storage_isolation,
					'encryption_at_rest': policy.encryption_at_rest,
					'encryption_in_transit': policy.encryption_in_transit
				},
				'namespace': namespace,
				'resources_count': len(resources),
				'resources': [
					{
						'resource_id': r.resource_id,
						'resource_type': r.resource_type,
						'isolation_tags': r.isolation_tags,
						'created_at': r.created_at.isoformat()
					} for r in resources
				],
				'encryption_status': {
					'keys_generated': len(encryption_keys) > 0,
					'key_types': list(encryption_keys.keys()) if encryption_keys else []
				},
				'security_boundaries': {
					'cross_tenant_access_attempts': len([
						log for log in self.cross_tenant_access_log 
						if log['requesting_tenant_id'] == tenant_id or log['target_tenant_id'] == tenant_id
					]),
					'allowed_integrations': len(policy.allowed_integrations)
				},
				'compliance_status': {
					'data_residency_enforced': len(policy.data_residency_requirements) > 0,
					'access_logging_enabled': policy.access_logging,
					'custom_policies_count': len(policy.custom_security_policies)
				},
				'last_updated': policy.updated_at.isoformat()
			}
			
		except Exception as e:
			return {
				'error': f'Failed to get tenant isolation status: {str(e)}',
				'tenant_id': tenant_id,
				'timestamp': datetime.utcnow().isoformat()
			}
	
	async def encrypt_tenant_data(self, tenant_id: str, data: str, 
								  algorithm: str = None) -> Dict[str, str]:
		"""Encrypt data using tenant-specific keys with proper cryptographic methods"""
		try:
			encryption_keys = self.encryption_keys.get(tenant_id, {})
			if not encryption_keys or 'data_key' not in encryption_keys:
				raise ValueError(f"No encryption key found for tenant {tenant_id}")
			
			algorithm = algorithm or self.default_algorithm
			if algorithm not in self.encryption_algorithms:
				raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
			
			# Get tenant's data encryption key
			data_key = encryption_keys['data_key']
			
			# Encrypt using AES-256-GCM (production-ready encryption)
			if algorithm == 'AES256':
				return await self._encrypt_aes_gcm(data, data_key, tenant_id)
			elif algorithm == 'ChaCha20':
				return await self._encrypt_chacha20(data, data_key, tenant_id)
			else:
				raise ValueError(f"Encryption algorithm {algorithm} not implemented")
			
		except Exception as e:
			raise RuntimeError(f"Failed to encrypt data for tenant {tenant_id}: {str(e)}")
	
	async def decrypt_tenant_data(self, tenant_id: str, 
								  encrypted_data: Dict[str, str]) -> str:
		"""Decrypt data using tenant-specific keys with proper cryptographic methods"""
		try:
			encryption_keys = self.encryption_keys.get(tenant_id, {})
			if not encryption_keys or 'data_key' not in encryption_keys:
				raise ValueError(f"No encryption key found for tenant {tenant_id}")
			
			# Validate encrypted data format
			if not isinstance(encrypted_data, dict) or 'algorithm' not in encrypted_data:
				raise ValueError("Invalid encrypted data format")
			
			algorithm = encrypted_data['algorithm']
			if algorithm not in self.encryption_algorithms:
				raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
			
			# Get tenant's data encryption key
			data_key = encryption_keys['data_key']
			
			# Decrypt using the appropriate algorithm
			if algorithm == 'AES256':
				return await self._decrypt_aes_gcm(encrypted_data, data_key, tenant_id)
			elif algorithm == 'ChaCha20':
				return await self._decrypt_chacha20(encrypted_data, data_key, tenant_id)
			else:
				raise ValueError(f"Decryption algorithm {algorithm} not implemented")
			
		except Exception as e:
			raise RuntimeError(f"Failed to decrypt data for tenant {tenant_id}: {str(e)}")
	
	def _generate_secure_master_key(self) -> str:
		"""Generate a cryptographically secure master key"""
		try:
			import secrets
			# Generate a 256-bit (32-byte) secure random key
			key_bytes = secrets.token_bytes(32)
			return key_bytes.hex()
		except ImportError:
			# Fallback to os.urandom if secrets module not available
			import os
			key_bytes = os.urandom(32)
			return key_bytes.hex()
	
	async def _encrypt_aes_gcm(self, data: str, key: str, tenant_id: str) -> Dict[str, str]:
		"""Encrypt data using AES-256-GCM (production-ready encryption)"""
		try:
			# Try to use cryptography library for proper AES-GCM encryption
			try:
				from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
				from cryptography.hazmat.backends import default_backend
				import os
				
				# Derive encryption key from the stored key
				key_bytes = bytes.fromhex(key)[:32]  # Use first 32 bytes for AES-256
				
				# Generate random IV (96-bit for GCM)
				iv = os.urandom(12)
				
				# Create cipher
				cipher = Cipher(
					algorithms.AES(key_bytes),
					modes.GCM(iv),
					backend=default_backend()
				)
				encryptor = cipher.encryptor()
				
				# Encrypt the data
				data_bytes = data.encode('utf-8')
				ciphertext = encryptor.update(data_bytes) + encryptor.finalize()
				
				# Return encrypted data with metadata
				return {
					'algorithm': 'AES256',
					'ciphertext': ciphertext.hex(),
					'iv': iv.hex(),
					'tag': encryptor.tag.hex(),
					'tenant_id': tenant_id,
					'encrypted_at': datetime.utcnow().isoformat()
				}
				
			except ImportError:
				# Fallback to HMAC-based encryption if cryptography library not available
				return self._fallback_encrypt_hmac(data, key, tenant_id, 'AES256')
				
		except Exception as e:
			raise RuntimeError(f"AES-GCM encryption failed: {str(e)}")
	
	async def _decrypt_aes_gcm(self, encrypted_data: Dict[str, str], 
							   key: str, tenant_id: str) -> str:
		"""Decrypt data using AES-256-GCM (production-ready decryption)"""
		try:
			# Try to use cryptography library for proper AES-GCM decryption
			try:
				from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
				from cryptography.hazmat.backends import default_backend
				
				# Extract components
				ciphertext = bytes.fromhex(encrypted_data['ciphertext'])
				iv = bytes.fromhex(encrypted_data['iv'])
				tag = bytes.fromhex(encrypted_data['tag'])
				
				# Derive decryption key
				key_bytes = bytes.fromhex(key)[:32]  # Use first 32 bytes for AES-256
				
				# Create cipher
				cipher = Cipher(
					algorithms.AES(key_bytes),
					modes.GCM(iv, tag),
					backend=default_backend()
				)
				decryptor = cipher.decryptor()
				
				# Decrypt the data
				plaintext = decryptor.update(ciphertext) + decryptor.finalize()
				
				return plaintext.decode('utf-8')
				
			except ImportError:
				# Fallback to HMAC-based decryption if cryptography library not available
				return self._fallback_decrypt_hmac(encrypted_data, key, tenant_id)
				
		except Exception as e:
			raise RuntimeError(f"AES-GCM decryption failed: {str(e)}")
	
	async def _encrypt_chacha20(self, data: str, key: str, tenant_id: str) -> Dict[str, str]:
		"""Encrypt data using ChaCha20-Poly1305 (alternative encryption)"""
		try:
			# Try to use cryptography library for proper ChaCha20-Poly1305 encryption
			try:
				from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
				import os
				
				# Derive encryption key from the stored key
				key_bytes = bytes.fromhex(key)[:32]  # Use first 32 bytes for ChaCha20
				
				# Generate random nonce (96-bit for ChaCha20Poly1305)
				nonce = os.urandom(12)
				
				# Create cipher
				cipher = ChaCha20Poly1305(key_bytes)
				
				# Encrypt the data
				data_bytes = data.encode('utf-8')
				ciphertext = cipher.encrypt(nonce, data_bytes, None)
				
				# Return encrypted data with metadata
				return {
					'algorithm': 'ChaCha20',
					'ciphertext': ciphertext.hex(),
					'nonce': nonce.hex(),
					'tenant_id': tenant_id,
					'encrypted_at': datetime.utcnow().isoformat()
				}
				
			except ImportError:
				# Fallback to HMAC-based encryption if cryptography library not available
				return self._fallback_encrypt_hmac(data, key, tenant_id, 'ChaCha20')
				
		except Exception as e:
			raise RuntimeError(f"ChaCha20-Poly1305 encryption failed: {str(e)}")
	
	async def _decrypt_chacha20(self, encrypted_data: Dict[str, str], 
								key: str, tenant_id: str) -> str:
		"""Decrypt data using ChaCha20-Poly1305 (alternative decryption)"""
		try:
			# Try to use cryptography library for proper ChaCha20-Poly1305 decryption
			try:
				from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
				
				# Extract components
				ciphertext = bytes.fromhex(encrypted_data['ciphertext'])
				nonce = bytes.fromhex(encrypted_data['nonce'])
				
				# Derive decryption key
				key_bytes = bytes.fromhex(key)[:32]  # Use first 32 bytes for ChaCha20
				
				# Create cipher
				cipher = ChaCha20Poly1305(key_bytes)
				
				# Decrypt the data
				plaintext = cipher.decrypt(nonce, ciphertext, None)
				
				return plaintext.decode('utf-8')
				
			except ImportError:
				# Fallback to HMAC-based decryption if cryptography library not available
				return self._fallback_decrypt_hmac(encrypted_data, key, tenant_id)
				
		except Exception as e:
			raise RuntimeError(f"ChaCha20-Poly1305 decryption failed: {str(e)}")
	
	def _fallback_encrypt_hmac(self, data: str, key: str, tenant_id: str, 
							   algorithm: str) -> Dict[str, str]:
		"""Fallback HMAC-based encryption when proper crypto libraries unavailable"""
		try:
			import os
			
			# Generate salt for key derivation
			salt = os.urandom(16)
			
			# Derive key using PBKDF2
			derived_key = hashlib.pbkdf2_hmac('sha256', key.encode(), salt, 100000)
			
			# Simple XOR encryption with HMAC authentication
			data_bytes = data.encode('utf-8')
			key_stream = hashlib.pbkdf2_hmac('sha256', derived_key, b'encrypt', len(data_bytes))
			
			# XOR encryption
			ciphertext = bytes(a ^ b for a, b in zip(data_bytes, key_stream))
			
			# Generate HMAC for authentication
			auth_key = hashlib.pbkdf2_hmac('sha256', derived_key, b'auth', 32)
			tag = hmac.new(auth_key, salt + ciphertext, hashlib.sha256).hexdigest()
			
			print(f"[HLTH-ISO] Using fallback encryption for tenant {tenant_id} (crypto library not available)")
			
			return {
				'algorithm': algorithm,
				'ciphertext': ciphertext.hex(),
				'salt': salt.hex(),
				'tag': tag,
				'tenant_id': tenant_id,
				'encrypted_at': datetime.utcnow().isoformat(),
				'fallback': True
			}
			
		except Exception as e:
			raise RuntimeError(f"Fallback encryption failed: {str(e)}")
	
	def _fallback_decrypt_hmac(self, encrypted_data: Dict[str, str], 
							   key: str, tenant_id: str) -> str:
		"""Fallback HMAC-based decryption when proper crypto libraries unavailable"""
		try:
			# Extract components
			ciphertext = bytes.fromhex(encrypted_data['ciphertext'])
			salt = bytes.fromhex(encrypted_data['salt'])
			tag = encrypted_data['tag']
			
			# Derive key using PBKDF2
			derived_key = hashlib.pbkdf2_hmac('sha256', key.encode(), salt, 100000)
			
			# Verify HMAC authentication
			auth_key = hashlib.pbkdf2_hmac('sha256', derived_key, b'auth', 32)
			expected_tag = hmac.new(auth_key, salt + ciphertext, hashlib.sha256).hexdigest()
			
			if not hmac.compare_digest(tag, expected_tag):
				raise ValueError("Authentication tag verification failed")
			
			# Generate key stream for decryption
			key_stream = hashlib.pbkdf2_hmac('sha256', derived_key, b'encrypt', len(ciphertext))
			
			# XOR decryption
			plaintext_bytes = bytes(a ^ b for a, b in zip(ciphertext, key_stream))
			
			return plaintext_bytes.decode('utf-8')
			
		except Exception as e:
			raise RuntimeError(f"Fallback decryption failed: {str(e)}")


# Export classes
__all__ = [
	'IsolationLevel',
	'DataClassification',
	'TenantIsolationPolicy',
	'TenantResource',
	'TenantIsolationManager'
]