"""
APG Document Security Framework

Multi-layered security framework integrated with APG auth_rbac and audit_compliance
for comprehensive document protection, access control, and compliance monitoring.

Author: Nyimbi Odero <nyimbi@gmail.com>
Copyright: © 2025 Datacraft
"""

import asyncio
import hashlib
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

from .apg_context import APGContext
from .config import APGDocumentConfig

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
	"""Document security classification levels"""
	PUBLIC = "public"
	INTERNAL = "internal"
	CONFIDENTIAL = "confidential"
	RESTRICTED = "restricted"
	TOP_SECRET = "top_secret"


class AccessAction(str, Enum):
	"""Document access actions"""
	READ = "document:read"
	WRITE = "document:write"
	DELETE = "document:delete"
	SHARE = "document:share"
	ADMIN = "document:admin"
	DOWNLOAD = "document:download"
	PRINT = "document:print"
	COPY = "document:copy"


@dataclass
class SecurityPolicy:
	"""Document security policy definition"""
	classification: SecurityLevel
	required_permissions: List[str]
	encryption_required: bool
	audit_level: str  # basic, detailed, comprehensive
	retention_days: int
	access_restrictions: Dict[str, Any]
	compliance_requirements: List[str]


@dataclass
class AccessContext:
	"""Access request context for authorization"""
	user_id: str
	tenant_id: str
	document_id: str
	action: AccessAction
	ip_address: Optional[str] = None
	user_agent: Optional[str] = None
	device_id: Optional[str] = None
	location: Optional[str] = None
	additional_context: Dict[str, Any] = None
	
	def __post_init__(self):
		if self.additional_context is None:
			self.additional_context = {}


@dataclass
class SecurityDecision:
	"""Authorization decision result"""
	allowed: bool
	reason: str
	security_level: SecurityLevel
	required_actions: List[str] = None
	warnings: List[str] = None
	audit_required: bool = True
	
	def __post_init__(self):
		if self.required_actions is None:
			self.required_actions = []
		if self.warnings is None:
			self.warnings = []


class DocumentEncryption:
	"""Document encryption service using APG security patterns"""
	
	def __init__(self, config: APGDocumentConfig):
		self.config = config
		self._encryption_key = self._derive_encryption_key()
		self._cipher = Fernet(self._encryption_key)
		
		self._log_encryption_initialized()
	
	def _log_encryption_initialized(self) -> None:
		"""Log encryption service initialization"""
		logger.info("Document encryption service initialized")
		logger.info(f"Encryption enabled: {self.config.encryption_enabled}")
	
	def _derive_encryption_key(self) -> bytes:
		"""Derive encryption key from configuration"""
		# In production, use a proper key management system
		password = os.getenv("APG_DOCUMENT_ENCRYPTION_KEY", "default-key-change-in-production").encode()
		salt = os.getenv("APG_DOCUMENT_ENCRYPTION_SALT", "default-salt").encode()
		
		kdf = PBKDF2HMAC(
			algorithm=hashes.SHA256(),
			length=32,
			salt=salt,
			iterations=100000,
		)
		key = base64.urlsafe_b64encode(kdf.derive(password))
		return key
	
	async def encrypt_content(self, content: str, security_level: SecurityLevel) -> str:
		"""Encrypt document content based on security level"""
		assert content is not None, "Content cannot be None"
		assert isinstance(security_level, SecurityLevel), "security_level must be SecurityLevel enum"
		
		if not self.config.encryption_enabled:
			return content
		
		# Only encrypt sensitive content
		if security_level in [SecurityLevel.PUBLIC, SecurityLevel.INTERNAL]:
			return content
		
		try:
			encrypted_bytes = self._cipher.encrypt(content.encode())
			encrypted_content = base64.urlsafe_b64encode(encrypted_bytes).decode()
			
			logger.debug(f"Encrypted content for security level: {security_level}")
			return encrypted_content
			
		except Exception as e:
			logger.error(f"Content encryption failed: {e}")
			raise
	
	async def decrypt_content(self, encrypted_content: str, security_level: SecurityLevel) -> str:
		"""Decrypt document content"""
		assert encrypted_content is not None, "Encrypted content cannot be None"
		
		if not self.config.encryption_enabled:
			return encrypted_content
		
		# Only decrypt if it was encrypted
		if security_level in [SecurityLevel.PUBLIC, SecurityLevel.INTERNAL]:
			return encrypted_content
		
		try:
			encrypted_bytes = base64.urlsafe_b64decode(encrypted_content.encode())
			decrypted_bytes = self._cipher.decrypt(encrypted_bytes)
			decrypted_content = decrypted_bytes.decode()
			
			logger.debug(f"Decrypted content for security level: {security_level}")
			return decrypted_content
			
		except Exception as e:
			logger.error(f"Content decryption failed: {e}")
			raise
	
	async def generate_content_hash(self, content: str) -> str:
		"""Generate content hash for integrity verification"""
		assert content is not None, "Content cannot be None"
		
		content_bytes = content.encode()
		hash_obj = hashlib.sha256(content_bytes)
		content_hash = hash_obj.hexdigest()
		
		return content_hash
	
	async def verify_content_integrity(self, content: str, expected_hash: str) -> bool:
		"""Verify content integrity using hash"""
		assert content is not None, "Content cannot be None"
		assert expected_hash, "Expected hash is required"
		
		actual_hash = await self.generate_content_hash(content)
		return actual_hash == expected_hash


class DocumentSecurityManager:
	"""
	Comprehensive document security manager integrated with APG capabilities.
	
	Provides authorization, audit logging, encryption, and compliance monitoring
	for document operations within the APG ecosystem.
	"""
	
	def __init__(self, apg_context: APGContext, config: APGDocumentConfig):
		assert apg_context, "APG context is required"
		assert config, "Configuration is required"
		
		self.apg_context = apg_context
		self.config = config
		self.encryption = DocumentEncryption(config)
		
		# Security policies by classification level
		self._security_policies = self._initialize_security_policies()
		
		self._log_security_manager_initialized()
	
	def _log_security_manager_initialized(self) -> None:
		"""Log security manager initialization"""
		logger.info(f"Document Security Manager initialized for tenant: {self.apg_context.tenant_id}")
		logger.info(f"Security policies loaded: {len(self._security_policies)}")
		logger.info(f"Audit compliance enabled: {self.config.audit_all_operations}")
	
	def _initialize_security_policies(self) -> Dict[SecurityLevel, SecurityPolicy]:
		"""Initialize security policies for different classification levels"""
		return {
			SecurityLevel.PUBLIC: SecurityPolicy(
				classification=SecurityLevel.PUBLIC,
				required_permissions=["document:read"],
				encryption_required=False,
				audit_level="basic",
				retention_days=365,
				access_restrictions={},
				compliance_requirements=[]
			),
			SecurityLevel.INTERNAL: SecurityPolicy(
				classification=SecurityLevel.INTERNAL,
				required_permissions=["document:read", "document:write"],
				encryption_required=False,
				audit_level="detailed",
				retention_days=1095,  # 3 years
				access_restrictions={"require_authentication": True},
				compliance_requirements=["internal_access_control"]
			),
			SecurityLevel.CONFIDENTIAL: SecurityPolicy(
				classification=SecurityLevel.CONFIDENTIAL,
				required_permissions=["document:read", "document:write", "confidential:access"],
				encryption_required=True,
				audit_level="comprehensive",
				retention_days=2555,  # 7 years
				access_restrictions={
					"require_authentication": True,
					"require_mfa": False,
					"ip_restrictions": []
				},
				compliance_requirements=["data_protection", "audit_trail"]
			),
			SecurityLevel.RESTRICTED: SecurityPolicy(
				classification=SecurityLevel.RESTRICTED,
				required_permissions=["document:read", "document:write", "restricted:access"],
				encryption_required=True,
				audit_level="comprehensive",
				retention_days=3650,  # 10 years
				access_restrictions={
					"require_authentication": True,
					"require_mfa": True,
					"ip_restrictions": [],
					"time_restrictions": []
				},
				compliance_requirements=["data_protection", "audit_trail", "restricted_access"]
			),
			SecurityLevel.TOP_SECRET: SecurityPolicy(
				classification=SecurityLevel.TOP_SECRET,
				required_permissions=["document:read", "document:write", "top_secret:access", "document:admin"],
				encryption_required=True,
				audit_level="comprehensive",
				retention_days=7300,  # 20 years
				access_restrictions={
					"require_authentication": True,
					"require_mfa": True,
					"require_biometric": True,
					"ip_restrictions": [],
					"time_restrictions": [],
					"location_restrictions": []
				},
				compliance_requirements=["data_protection", "audit_trail", "top_secret_clearance", "full_monitoring"]
			)
		}
	
	async def authorize_document_access(self, access_context: AccessContext, 
									   document_classification: SecurityLevel) -> SecurityDecision:
		"""
		Authorize document access through APG auth_rbac with comprehensive security evaluation.
		
		Args:
			access_context: Access request context
			document_classification: Document security classification
			
		Returns:
			SecurityDecision with authorization result and required actions
		"""
		assert access_context, "Access context is required"
		assert document_classification, "Document classification is required"
		
		self._log_authorization_start(access_context, document_classification)
		
		try:
			# Get security policy for classification level
			policy = self._security_policies.get(document_classification)
			if not policy:
				return SecurityDecision(
					allowed=False,
					reason=f"Unknown security classification: {document_classification}",
					security_level=document_classification
				)
			
			# Check user permissions through APG auth_rbac
			auth_service = self.apg_context.get_capability("auth_rbac")
			if not auth_service:
				return SecurityDecision(
					allowed=False,
					reason="Authentication service unavailable",
					security_level=document_classification
				)
			
			# Evaluate access through APG RBAC
			access_decision = await auth_service.evaluate_access(
				subject=access_context.user_id,
				resource=access_context.document_id,
				action=access_context.action.value,
				context={
					"document_classification": document_classification.value,
					"tenant_id": access_context.tenant_id,
					"ip_address": access_context.ip_address,
					"user_agent": access_context.user_agent,
					"device_id": access_context.device_id,
					"location": access_context.location,
					**access_context.additional_context
				}
			)
			
			if not access_decision.allowed:
				decision = SecurityDecision(
					allowed=False,
					reason=access_decision.reason or "Access denied by authorization service",
					security_level=document_classification
				)
			else:
				# Additional security checks based on policy
				decision = await self._evaluate_additional_security_requirements(
					access_context, policy, access_decision
				)
			
			# Log authorization attempt through APG audit_compliance
			await self._log_authorization_attempt(access_context, document_classification, decision)
			
			self._log_authorization_complete(access_context, decision)
			return decision
			
		except Exception as e:
			logger.error(f"Authorization error: {e}")
			decision = SecurityDecision(
				allowed=False,
				reason=f"Authorization failed: {str(e)}",
				security_level=document_classification
			)
			await self._log_authorization_attempt(access_context, document_classification, decision)
			return decision
	
	async def _evaluate_additional_security_requirements(self, access_context: AccessContext,
														policy: SecurityPolicy, base_decision: Any) -> SecurityDecision:
		"""Evaluate additional security requirements beyond basic RBAC"""
		warnings = []
		required_actions = []
		
		# Check access restrictions
		restrictions = policy.access_restrictions
		
		# IP restrictions
		if restrictions.get("ip_restrictions") and access_context.ip_address:
			allowed_ips = restrictions["ip_restrictions"]
			if allowed_ips and access_context.ip_address not in allowed_ips:
				return SecurityDecision(
					allowed=False,
					reason=f"Access denied: IP address {access_context.ip_address} not in allowed list",
					security_level=policy.classification
				)
		
		# MFA requirements
		if restrictions.get("require_mfa", False):
			# In production, check if user has completed MFA
			required_actions.append("mfa_verification")
			warnings.append("Multi-factor authentication may be required")
		
		# Biometric requirements
		if restrictions.get("require_biometric", False):
			required_actions.append("biometric_verification")
			warnings.append("Biometric verification may be required")
		
		# Time-based restrictions
		if restrictions.get("time_restrictions"):
			current_hour = datetime.utcnow().hour
			allowed_hours = restrictions["time_restrictions"]
			if allowed_hours and current_hour not in allowed_hours:
				return SecurityDecision(
					allowed=False,
					reason=f"Access denied: Current time {current_hour}:00 not in allowed hours {allowed_hours}",
					security_level=policy.classification
				)
		
		return SecurityDecision(
			allowed=True,
			reason="Access granted with security requirements",
			security_level=policy.classification,
			required_actions=required_actions,
			warnings=warnings,
			audit_required=policy.audit_level in ["detailed", "comprehensive"]
		)
	
	async def _log_authorization_attempt(self, access_context: AccessContext, 
										classification: SecurityLevel, decision: SecurityDecision) -> None:
		"""Log authorization attempt through APG audit_compliance"""
		audit_service = self.apg_context.get_capability("audit_compliance")
		if not audit_service:
			logger.warning("Audit service unavailable - authorization not logged")
			return
		
		try:
			await audit_service.log_access_attempt(
				user_id=access_context.user_id,
				resource_type="document",
				resource_id=access_context.document_id,
				action=access_context.action.value,
				result=decision.allowed,
				context={
					"security_classification": classification.value,
					"tenant_id": access_context.tenant_id,
					"ip_address": access_context.ip_address,
					"user_agent": access_context.user_agent,
					"device_id": access_context.device_id,
					"decision_reason": decision.reason,
					"required_actions": decision.required_actions,
					"warnings": decision.warnings
				}
			)
		except Exception as e:
			logger.error(f"Failed to log authorization attempt: {e}")
	
	async def log_document_operation(self, operation: str, document_id: str, user_id: str,
									metadata: Dict[str, Any] = None) -> None:
		"""Log document operation through APG audit_compliance"""
		if metadata is None:
			metadata = {}
		
		audit_service = self.apg_context.get_capability("audit_compliance")
		if not audit_service:
			logger.warning(f"Audit service unavailable - operation {operation} not logged")
			return
		
		try:
			await audit_service.log_event(
				event_type=f"document_{operation}",
				resource_id=document_id,
				user_id=user_id,
				metadata={
					"tenant_id": self.apg_context.tenant_id,
					"timestamp": datetime.utcnow().isoformat(),
					"operation": operation,
					**metadata
				}
			)
		except Exception as e:
			logger.error(f"Failed to log document operation {operation}: {e}")
	
	async def encrypt_document_content(self, content: str, classification: SecurityLevel) -> str:
		"""Encrypt document content based on classification level"""
		return await self.encryption.encrypt_content(content, classification)
	
	async def decrypt_document_content(self, encrypted_content: str, classification: SecurityLevel) -> str:
		"""Decrypt document content based on classification level"""
		return await self.encryption.decrypt_content(encrypted_content, classification)
	
	async def generate_content_hash(self, content: str) -> str:
		"""Generate content hash for integrity verification"""
		return await self.encryption.generate_content_hash(content)
	
	async def verify_content_integrity(self, content: str, expected_hash: str) -> bool:
		"""Verify content integrity using hash"""
		return await self.encryption.verify_content_integrity(content, expected_hash)
	
	def get_security_policy(self, classification: SecurityLevel) -> Optional[SecurityPolicy]:
		"""Get security policy for classification level"""
		return self._security_policies.get(classification)
	
	async def validate_document_classification(self, content: str, proposed_classification: SecurityLevel) -> SecurityLevel:
		"""Validate and potentially adjust document classification based on content"""
		# Simple content analysis for classification validation
		content_lower = content.lower()
		
		# Check for sensitive keywords
		top_secret_keywords = ["top secret", "classified", "confidential", "restricted"]
		confidential_keywords = ["internal use", "confidential", "proprietary", "sensitive"]
		
		if any(keyword in content_lower for keyword in top_secret_keywords):
			recommended_classification = SecurityLevel.TOP_SECRET
		elif any(keyword in content_lower for keyword in confidential_keywords):
			recommended_classification = SecurityLevel.CONFIDENTIAL
		else:
			recommended_classification = proposed_classification
		
		# Return the higher of proposed or recommended classification
		classification_levels = {
			SecurityLevel.PUBLIC: 0,
			SecurityLevel.INTERNAL: 1,
			SecurityLevel.CONFIDENTIAL: 2,
			SecurityLevel.RESTRICTED: 3,
			SecurityLevel.TOP_SECRET: 4
		}
		
		proposed_level = classification_levels.get(proposed_classification, 0)
		recommended_level = classification_levels.get(recommended_classification, 0)
		
		if recommended_level > proposed_level:
			logger.warning(f"Document classification upgraded from {proposed_classification} to {recommended_classification} based on content analysis")
			return recommended_classification
		
		return proposed_classification
	
	def _log_authorization_start(self, access_context: AccessContext, classification: SecurityLevel) -> None:
		"""Log authorization process start"""
		logger.debug(f"Starting authorization for user {access_context.user_id} on document {access_context.document_id}")
		logger.debug(f"Action: {access_context.action}, Classification: {classification}")
	
	def _log_authorization_complete(self, access_context: AccessContext, decision: SecurityDecision) -> None:
		"""Log authorization process completion"""
		result = "ALLOWED" if decision.allowed else "DENIED"
		logger.debug(f"Authorization {result} for user {access_context.user_id}: {decision.reason}")


async def create_security_manager(apg_context: APGContext, config: APGDocumentConfig) -> DocumentSecurityManager:
	"""Create and initialize document security manager"""
	security_manager = DocumentSecurityManager(apg_context, config)
	logger.info("Document security manager created successfully")
	return security_manager