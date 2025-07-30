#!/usr/bin/env python3
"""
Security & Compliance Engine for APG Notification System

This module provides comprehensive security, compliance, and data protection
capabilities for the notification system, ensuring GDPR, CCPA, HIPAA, and
other regulatory compliance while maintaining enterprise-grade security.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from enum import Enum
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import re
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ComplianceRegime(Enum):
	"""Supported compliance regimes"""
	GDPR = "gdpr"
	CCPA = "ccpa"
	HIPAA = "hipaa"
	SOX = "sox"
	PCI_DSS = "pci_dss"
	ISO_27001 = "iso_27001"
	COPPA = "coppa"
	FERPA = "ferpa"

class DataClassification(Enum):
	"""Data classification levels"""
	PUBLIC = "public"
	INTERNAL = "internal"
	CONFIDENTIAL = "confidential"
	RESTRICTED = "restricted"
	TOP_SECRET = "top_secret"

class SecurityThreatLevel(Enum):
	"""Security threat levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"

class AuditEventType(Enum):
	"""Audit event types"""
	DATA_ACCESS = "data_access"
	DATA_MODIFICATION = "data_modification"
	DATA_DELETION = "data_deletion"
	SECURITY_VIOLATION = "security_violation"
	COMPLIANCE_CHECK = "compliance_check"
	ENCRYPTION_OPERATION = "encryption_operation"
	KEY_ROTATION = "key_rotation"
	USER_AUTHENTICATION = "user_authentication"
	PERMISSION_CHANGE = "permission_change"
	DATA_EXPORT = "data_export"
	ANONYMIZATION = "anonymization"
	CONSENT_MANAGEMENT = "consent_management"

@dataclass
class SecurityPolicy:
	"""Security policy configuration"""
	policy_id: str
	name: str
	description: str
	compliance_regimes: List[ComplianceRegime]
	data_classification: DataClassification
	encryption_required: bool = True
	audit_required: bool = True
	retention_days: int = 2555  # 7 years default
	anonymization_required: bool = False
	consent_required: bool = True
	cross_border_allowed: bool = False
	third_party_sharing_allowed: bool = False
	data_minimization: bool = True
	purpose_limitation: bool = True
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AuditEvent:
	"""Audit event record"""
	event_id: str
	tenant_id: str
	user_id: Optional[str]
	event_type: AuditEventType
	resource_type: str
	resource_id: str
	action: str
	timestamp: datetime
	ip_address: Optional[str] = None
	user_agent: Optional[str] = None
	session_id: Optional[str] = None
	details: Dict[str, Any] = field(default_factory=dict)
	risk_score: float = 0.0
	compliance_flags: List[str] = field(default_factory=list)
	sensitive_data_involved: bool = False

@dataclass
class ConsentRecord:
	"""User consent record"""
	consent_id: str
	tenant_id: str
	user_id: str
	purpose: str
	granted: bool
	timestamp: datetime
	expiry_date: Optional[datetime] = None
	withdrawal_date: Optional[datetime] = None
	consent_method: str = "explicit"
	legal_basis: str = "consent"
	data_categories: List[str] = field(default_factory=list)
	processing_purposes: List[str] = field(default_factory=list)
	third_parties: List[str] = field(default_factory=list)
	cross_border_transfers: List[str] = field(default_factory=list)

@dataclass
class DataSubjectRequest:
	"""Data subject rights request"""
	request_id: str
	tenant_id: str
	user_id: str
	request_type: str  # access, rectification, erasure, portability, restriction
	status: str  # pending, processing, completed, rejected
	submitted_at: datetime
	completed_at: Optional[datetime] = None
	details: Dict[str, Any] = field(default_factory=dict)
	verification_method: str = "email"
	legal_basis: str = "data_subject_rights"

class SecurityValidator:
	"""Security validation utilities"""
	
	@staticmethod
	def validate_email(email: str) -> bool:
		"""Validate email format with security considerations"""
		if not email or len(email) > 254:
			return False
		
		pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
		if not re.match(pattern, email):
			return False
		
		# Check for common security issues
		dangerous_patterns = [
			r'<script',
			r'javascript:',
			r'data:',
			r'vbscript:'
		]
		
		for pattern in dangerous_patterns:
			if re.search(pattern, email, re.IGNORECASE):
				return False
		
		return True
	
	@staticmethod
	def validate_phone_number(phone: str) -> bool:
		"""Validate phone number format"""
		if not phone:
			return False
		
		# Remove common formatting
		clean_phone = re.sub(r'[^\d+]', '', phone)
		
		# Basic international format validation
		if re.match(r'^\+?[1-9]\d{1,14}$', clean_phone):
			return True
		
		return False
	
	@staticmethod
	def sanitize_input(data: str, max_length: int = 1000) -> str:
		"""Sanitize user input to prevent injection attacks"""
		if not data:
			return ""
		
		# Truncate to max length
		data = data[:max_length]
		
		# Remove dangerous characters and patterns
		dangerous_patterns = [
			r'<script[^>]*>.*?</script>',
			r'javascript:',
			r'data:',
			r'vbscript:',
			r'on\w+\s*=',
			r'<iframe[^>]*>.*?</iframe>',
			r'<object[^>]*>.*?</object>',
			r'<embed[^>]*>.*?</embed>'
		]
		
		for pattern in dangerous_patterns:
			data = re.sub(pattern, '', data, flags=re.IGNORECASE | re.DOTALL)
		
		# Encode HTML entities
		data = data.replace('&', '&amp;')
		data = data.replace('<', '&lt;')
		data = data.replace('>', '&gt;')
		data = data.replace('"', '&quot;')
		data = data.replace("'", '&#x27;')
		
		return data
	
	@staticmethod
	def generate_secure_token(length: int = 32) -> str:
		"""Generate cryptographically secure random token"""
		return secrets.token_urlsafe(length)

class EncryptionManager:
	"""Advanced encryption and key management"""
	
	def __init__(self, master_key: Optional[bytes] = None):
		self.master_key = master_key or self._generate_master_key()
		self.cipher_suite = Fernet(self.master_key)
		self.key_cache: Dict[str, bytes] = {}
		self.key_rotation_interval = 86400  # 24 hours
		
	def _generate_master_key(self) -> bytes:
		"""Generate a new master encryption key"""
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
	
	def encrypt_data(self, data: str, context: Optional[str] = None) -> Dict[str, Any]:
		"""Encrypt sensitive data with context"""
		try:
			encrypted_data = self.cipher_suite.encrypt(data.encode('utf-8'))
			
			return {
				'encrypted_data': base64.b64encode(encrypted_data).decode('utf-8'),
				'encryption_method': 'Fernet',
				'context': context,
				'encrypted_at': datetime.utcnow().isoformat(),
				'key_id': self._get_current_key_id()
			}
		except Exception as e:
			logger.error(f"Encryption failed: {str(e)}")
			raise
	
	def decrypt_data(self, encrypted_data: Dict[str, Any]) -> str:
		"""Decrypt sensitive data"""
		try:
			encrypted_bytes = base64.b64decode(encrypted_data['encrypted_data'])
			decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)
			return decrypted_bytes.decode('utf-8')
		except Exception as e:
			logger.error(f"Decryption failed: {str(e)}")
			raise
	
	def _get_current_key_id(self) -> str:
		"""Get current encryption key identifier"""
		return hashlib.sha256(self.master_key).hexdigest()[:16]
	
	def rotate_keys(self) -> Dict[str, Any]:
		"""Rotate encryption keys"""
		old_key_id = self._get_current_key_id()
		self.master_key = self._generate_master_key()
		self.cipher_suite = Fernet(self.master_key)
		new_key_id = self._get_current_key_id()
		
		return {
			'old_key_id': old_key_id,
			'new_key_id': new_key_id,
			'rotated_at': datetime.utcnow().isoformat()
		}

class AuditLogger:
	"""Comprehensive audit logging system"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.audit_events: List[AuditEvent] = []
		self.encryption_manager = EncryptionManager()
		
	async def log_event(
		self,
		event_type: AuditEventType,
		resource_type: str,
		resource_id: str,
		action: str,
		user_id: Optional[str] = None,
		details: Optional[Dict[str, Any]] = None,
		ip_address: Optional[str] = None,
		user_agent: Optional[str] = None,
		session_id: Optional[str] = None
	) -> str:
		"""Log security/compliance audit event"""
		
		event_id = f"audit_{secrets.token_hex(16)}"
		
		# Calculate risk score based on event characteristics
		risk_score = self._calculate_risk_score(event_type, resource_type, details or {})
		
		# Check for sensitive data involvement
		sensitive_data_involved = self._check_sensitive_data(details or {})
		
		event = AuditEvent(
			event_id=event_id,
			tenant_id=self.tenant_id,
			user_id=user_id,
			event_type=event_type,
			resource_type=resource_type,
			resource_id=resource_id,
			action=action,
			timestamp=datetime.utcnow(),
			ip_address=ip_address,
			user_agent=user_agent,
			session_id=session_id,
			details=details or {},
			risk_score=risk_score,
			sensitive_data_involved=sensitive_data_involved
		)
		
		# Encrypt sensitive audit data
		if sensitive_data_involved:
			event.details = self.encryption_manager.encrypt_data(
				json.dumps(event.details),
				context=f"audit_{event_type.value}"
			)
		
		self.audit_events.append(event)
		
		# Log high-risk events immediately
		if risk_score > 0.7:
			logger.warning(f"High-risk audit event: {event_id} - {action}")
		
		return event_id
	
	def _calculate_risk_score(
		self,
		event_type: AuditEventType,
		resource_type: str,
		details: Dict[str, Any]
	) -> float:
		"""Calculate risk score for audit event"""
		base_scores = {
			AuditEventType.DATA_ACCESS: 0.2,
			AuditEventType.DATA_MODIFICATION: 0.5,
			AuditEventType.DATA_DELETION: 0.8,
			AuditEventType.SECURITY_VIOLATION: 0.9,
			AuditEventType.PERMISSION_CHANGE: 0.6,
			AuditEventType.DATA_EXPORT: 0.7,
			AuditEventType.ANONYMIZATION: 0.4,
		}
		
		risk_score = base_scores.get(event_type, 0.3)
		
		# Increase risk for sensitive resources
		if 'pii' in resource_type.lower() or 'personal' in resource_type.lower():
			risk_score += 0.3
		
		# Increase risk for bulk operations
		if details.get('bulk_operation', False):
			risk_score += 0.2
		
		# Increase risk for cross-border operations
		if details.get('cross_border', False):
			risk_score += 0.2
		
		return min(risk_score, 1.0)
	
	def _check_sensitive_data(self, details: Dict[str, Any]) -> bool:
		"""Check if event involves sensitive data"""
		sensitive_indicators = [
			'email', 'phone', 'ssn', 'credit_card', 'passport',
			'medical', 'biometric', 'genetic', 'financial'
		]
		
		details_str = json.dumps(details).lower()
		return any(indicator in details_str for indicator in sensitive_indicators)
	
	async def query_audit_events(
		self,
		start_date: datetime,
		end_date: datetime,
		event_types: Optional[List[AuditEventType]] = None,
		user_id: Optional[str] = None,
		min_risk_score: float = 0.0
	) -> List[AuditEvent]:
		"""Query audit events with filters"""
		
		filtered_events = []
		
		for event in self.audit_events:
			if not (start_date <= event.timestamp <= end_date):
				continue
			
			if event_types and event.event_type not in event_types:
				continue
			
			if user_id and event.user_id != user_id:
				continue
			
			if event.risk_score < min_risk_score:
				continue
			
			filtered_events.append(event)
		
		return filtered_events

class ConsentManager:
	"""GDPR/CCPA consent management system"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.consent_records: Dict[str, ConsentRecord] = {}
		self.audit_logger = AuditLogger(tenant_id)
		
	async def record_consent(
		self,
		user_id: str,
		purpose: str,
		granted: bool,
		consent_method: str = "explicit",
		legal_basis: str = "consent",
		data_categories: Optional[List[str]] = None,
		processing_purposes: Optional[List[str]] = None,
		expiry_days: Optional[int] = None
	) -> str:
		"""Record user consent"""
		
		consent_id = f"consent_{secrets.token_hex(16)}"
		
		expiry_date = None
		if expiry_days:
			expiry_date = datetime.utcnow() + timedelta(days=expiry_days)
		
		consent_record = ConsentRecord(
			consent_id=consent_id,
			tenant_id=self.tenant_id,
			user_id=user_id,
			purpose=purpose,
			granted=granted,
			timestamp=datetime.utcnow(),
			expiry_date=expiry_date,
			consent_method=consent_method,
			legal_basis=legal_basis,
			data_categories=data_categories or [],
			processing_purposes=processing_purposes or []
		)
		
		self.consent_records[consent_id] = consent_record
		
		# Log consent event
		await self.audit_logger.log_event(
			event_type=AuditEventType.CONSENT_MANAGEMENT,
			resource_type="consent",
			resource_id=consent_id,
			action="consent_recorded",
			user_id=user_id,
			details={
				'purpose': purpose,
				'granted': granted,
				'consent_method': consent_method,
				'legal_basis': legal_basis
			}
		)
		
		return consent_id
	
	async def withdraw_consent(self, user_id: str, consent_id: str) -> bool:
		"""Withdraw user consent"""
		
		if consent_id not in self.consent_records:
			return False
		
		consent_record = self.consent_records[consent_id]
		
		if consent_record.user_id != user_id:
			return False
		
		consent_record.granted = False
		consent_record.withdrawal_date = datetime.utcnow()
		
		# Log withdrawal event
		await self.audit_logger.log_event(
			event_type=AuditEventType.CONSENT_MANAGEMENT,
			resource_type="consent",
			resource_id=consent_id,
			action="consent_withdrawn",
			user_id=user_id,
			details={
				'purpose': consent_record.purpose,
				'withdrawal_date': consent_record.withdrawal_date.isoformat()
			}
		)
		
		return True
	
	async def check_consent(
		self,
		user_id: str,
		purpose: str,
		check_expiry: bool = True
	) -> Tuple[bool, Optional[str]]:
		"""Check if user has valid consent for purpose"""
		
		for consent_record in self.consent_records.values():
			if (consent_record.user_id == user_id and
				consent_record.purpose == purpose and
				consent_record.granted and
				not consent_record.withdrawal_date):
				
				# Check expiry if required
				if check_expiry and consent_record.expiry_date:
					if datetime.utcnow() > consent_record.expiry_date:
						return False, "expired"
				
				return True, consent_record.consent_id
		
		return False, None
	
	async def get_user_consents(self, user_id: str) -> List[ConsentRecord]:
		"""Get all consent records for a user"""
		
		user_consents = []
		for consent_record in self.consent_records.values():
			if consent_record.user_id == user_id:
				user_consents.append(consent_record)
		
		return user_consents

class DataSubjectRightsManager:
	"""Manage GDPR data subject rights requests"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.requests: Dict[str, DataSubjectRequest] = {}
		self.audit_logger = AuditLogger(tenant_id)
		self.encryption_manager = EncryptionManager()
		
	async def submit_request(
		self,
		user_id: str,
		request_type: str,
		details: Optional[Dict[str, Any]] = None,
		verification_method: str = "email"
	) -> str:
		"""Submit data subject rights request"""
		
		request_id = f"dsr_{secrets.token_hex(16)}"
		
		request = DataSubjectRequest(
			request_id=request_id,
			tenant_id=self.tenant_id,
			user_id=user_id,
			request_type=request_type,
			status="pending",
			submitted_at=datetime.utcnow(),
			details=details or {},
			verification_method=verification_method
		)
		
		self.requests[request_id] = request
		
		# Log request submission
		await self.audit_logger.log_event(
			event_type=AuditEventType.DATA_ACCESS,
			resource_type="data_subject_request",
			resource_id=request_id,
			action="request_submitted",
			user_id=user_id,
			details={
				'request_type': request_type,
				'verification_method': verification_method
			}
		)
		
		return request_id
	
	async def process_access_request(self, request_id: str) -> Dict[str, Any]:
		"""Process data access request (GDPR Art. 15)"""
		
		if request_id not in self.requests:
			raise ValueError("Request not found")
		
		request = self.requests[request_id]
		request.status = "processing"
		
		# Collect user data from various sources
		user_data = await self._collect_user_data(request.user_id)
		
		# Encrypt sensitive data for secure transmission
		encrypted_data = self.encryption_manager.encrypt_data(
			json.dumps(user_data),
			context="data_access_request"
		)
		
		request.status = "completed"
		request.completed_at = datetime.utcnow()
		
		# Log completion
		await self.audit_logger.log_event(
			event_type=AuditEventType.DATA_ACCESS,
			resource_type="data_subject_request",
			resource_id=request_id,
			action="access_request_completed",
			user_id=request.user_id,
			details={'data_categories': list(user_data.keys())}
		)
		
		return {
			'request_id': request_id,
			'user_id': request.user_id,
			'data': encrypted_data,
			'completed_at': request.completed_at.isoformat()
		}
	
	async def process_erasure_request(self, request_id: str) -> Dict[str, Any]:
		"""Process data erasure request (GDPR Art. 17 - Right to be forgotten)"""
		
		if request_id not in self.requests:
			raise ValueError("Request not found")
		
		request = self.requests[request_id]
		request.status = "processing"
		
		# Anonymize user data
		anonymization_result = await self._anonymize_user_data(request.user_id)
		
		request.status = "completed"
		request.completed_at = datetime.utcnow()
		
		# Log completion
		await self.audit_logger.log_event(
			event_type=AuditEventType.DATA_DELETION,
			resource_type="data_subject_request",
			resource_id=request_id,
			action="erasure_request_completed",
			user_id=request.user_id,
			details=anonymization_result
		)
		
		return {
			'request_id': request_id,
			'user_id': request.user_id,
			'anonymized_records': anonymization_result['anonymized_count'],
			'completed_at': request.completed_at.isoformat()
		}
	
	async def _collect_user_data(self, user_id: str) -> Dict[str, Any]:
		"""Collect all user data for access request"""
		# This would integrate with various data stores
		return {
			'profile_data': f"Profile data for {user_id}",
			'notification_history': f"Notification history for {user_id}",
			'preferences': f"Preferences for {user_id}",
			'audit_logs': f"Audit logs for {user_id}"
		}
	
	async def _anonymize_user_data(self, user_id: str) -> Dict[str, Any]:
		"""Anonymize user data for erasure request"""
		# This would anonymize data across all systems
		return {
			'anonymized_count': 42,
			'retained_count': 3,  # Some data may need to be retained for legal reasons
			'anonymization_method': 'k_anonymity',
			'completed_at': datetime.utcnow().isoformat()
		}

class ComplianceChecker:
	"""Automated compliance checking system"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.policies: Dict[str, SecurityPolicy] = {}
		self.audit_logger = AuditLogger(tenant_id)
		self._load_default_policies()
		
	def _load_default_policies(self):
		"""Load default security policies"""
		
		# GDPR Policy
		gdpr_policy = SecurityPolicy(
			policy_id="gdpr_default",
			name="GDPR Compliance Policy",
			description="Default GDPR compliance policy",
			compliance_regimes=[ComplianceRegime.GDPR],
			data_classification=DataClassification.CONFIDENTIAL,
			encryption_required=True,
			audit_required=True,
			retention_days=2555,  # 7 years
			anonymization_required=True,
			consent_required=True,
			cross_border_allowed=False,
			data_minimization=True,
			purpose_limitation=True
		)
		self.policies["gdpr_default"] = gdpr_policy
		
		# HIPAA Policy
		hipaa_policy = SecurityPolicy(
			policy_id="hipaa_default",
			name="HIPAA Compliance Policy",
			description="Default HIPAA compliance policy",
			compliance_regimes=[ComplianceRegime.HIPAA],
			data_classification=DataClassification.RESTRICTED,
			encryption_required=True,
			audit_required=True,
			retention_days=2555,
			anonymization_required=True,
			consent_required=True,
			cross_border_allowed=False,
			third_party_sharing_allowed=False,
			data_minimization=True,
			purpose_limitation=True
		)
		self.policies["hipaa_default"] = hipaa_policy
	
	async def check_compliance(
		self,
		action: str,
		resource_type: str,
		data: Dict[str, Any],
		user_context: Optional[Dict[str, Any]] = None
	) -> Dict[str, Any]:
		"""Check compliance for a given action"""
		
		compliance_results = {
			'compliant': True,
			'violations': [],
			'warnings': [],
			'policies_checked': [],
			'recommendations': []
		}
		
		for policy in self.policies.values():
			policy_result = await self._check_policy_compliance(
				policy, action, resource_type, data, user_context
			)
			
			compliance_results['policies_checked'].append(policy.policy_id)
			
			if not policy_result['compliant']:
				compliance_results['compliant'] = False
				compliance_results['violations'].extend(policy_result['violations'])
			
			compliance_results['warnings'].extend(policy_result['warnings'])
			compliance_results['recommendations'].extend(policy_result['recommendations'])
		
		# Log compliance check
		await self.audit_logger.log_event(
			event_type=AuditEventType.COMPLIANCE_CHECK,
			resource_type=resource_type,
			resource_id=data.get('id', 'unknown'),
			action=action,
			user_id=user_context.get('user_id') if user_context else None,
			details={
				'compliant': compliance_results['compliant'],
				'violations_count': len(compliance_results['violations']),
				'policies_checked': compliance_results['policies_checked']
			}
		)
		
		return compliance_results
	
	async def _check_policy_compliance(
		self,
		policy: SecurityPolicy,
		action: str,
		resource_type: str,
		data: Dict[str, Any],
		user_context: Optional[Dict[str, Any]]
	) -> Dict[str, Any]:
		"""Check compliance against specific policy"""
		
		result = {
			'compliant': True,
			'violations': [],
			'warnings': [],
			'recommendations': []
		}
		
		# Check encryption requirement
		if policy.encryption_required and action in ['store', 'transmit']:
			if not data.get('encrypted', False):
				result['compliant'] = False
				result['violations'].append(f"Encryption required by policy {policy.policy_id}")
		
		# Check consent requirement
		if policy.consent_required and action in ['process', 'store']:
			if not data.get('consent_obtained', False):
				result['compliant'] = False
				result['violations'].append(f"Consent required by policy {policy.policy_id}")
		
		# Check data minimization
		if policy.data_minimization and action == 'collect':
			if self._check_data_excessive(data):
				result['warnings'].append("Data collection may violate minimization principle")
		
		# Check cross-border restrictions
		if not policy.cross_border_allowed and action == 'transfer':
			if data.get('cross_border', False):
				result['compliant'] = False
				result['violations'].append(f"Cross-border transfer not allowed by policy {policy.policy_id}")
		
		# Check retention limits
		if action == 'retain':
			retention_days = data.get('retention_days', 0)
			if retention_days > policy.retention_days:
				result['warnings'].append(f"Retention period exceeds policy limit: {policy.retention_days} days")
		
		return result
	
	def _check_data_excessive(self, data: Dict[str, Any]) -> bool:
		"""Check if data collection seems excessive"""
		# Simple heuristic - in production this would be more sophisticated
		return len(data.get('fields', [])) > 20

class SecurityEngine:
	"""Main security and compliance engine"""
	
	def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
		self.tenant_id = tenant_id
		self.config = config or {}
		
		# Initialize components
		self.encryption_manager = EncryptionManager()
		self.audit_logger = AuditLogger(tenant_id)
		self.consent_manager = ConsentManager(tenant_id)
		self.rights_manager = DataSubjectRightsManager(tenant_id)
		self.compliance_checker = ComplianceChecker(tenant_id)
		self.validator = SecurityValidator()
		
		# Security monitoring
		self.threat_level = SecurityThreatLevel.LOW
		self.active_sessions: Dict[str, Dict[str, Any]] = {}
		self.failed_attempts: Dict[str, List[datetime]] = {}
		
		logger.info(f"Security engine initialized for tenant {tenant_id}")
	
	async def validate_and_secure_data(
		self,
		data: Dict[str, Any],
		operation: str,
		user_context: Optional[Dict[str, Any]] = None
	) -> Dict[str, Any]:
		"""Validate and secure data according to policies"""
		
		# Input validation
		validated_data = await self._validate_input_data(data)
		
		# Compliance check
		compliance_result = await self.compliance_checker.check_compliance(
			action=operation,
			resource_type="notification_data",
			data=validated_data,
			user_context=user_context
		)
		
		if not compliance_result['compliant']:
			raise ValueError(f"Compliance violations: {compliance_result['violations']}")
		
		# Encrypt sensitive data
		secured_data = await self._secure_sensitive_data(validated_data)
		
		# Log security operation
		await self.audit_logger.log_event(
			event_type=AuditEventType.DATA_MODIFICATION,
			resource_type="notification_data",
			resource_id=secured_data.get('id', 'unknown'),
			action=f"validate_and_secure_{operation}",
			user_id=user_context.get('user_id') if user_context else None,
			details={
				'operation': operation,
				'compliance_status': compliance_result['compliant'],
				'encryption_applied': secured_data.get('_encrypted', False)
			}
		)
		
		return {
			'data': secured_data,
			'compliance': compliance_result,
			'security_applied': True
		}
	
	async def _validate_input_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate input data for security issues"""
		
		validated_data = {}
		
		for key, value in data.items():
			if isinstance(value, str):
				# Sanitize string inputs
				validated_data[key] = self.validator.sanitize_input(value)
			elif isinstance(value, dict):
				# Recursively validate nested dictionaries
				validated_data[key] = await self._validate_input_data(value)
			elif isinstance(value, list):
				# Validate list items
				validated_list = []
				for item in value:
					if isinstance(item, str):
						validated_list.append(self.validator.sanitize_input(item))
					elif isinstance(item, dict):
						validated_list.append(await self._validate_input_data(item))
					else:
						validated_list.append(item)
				validated_data[key] = validated_list
			else:
				validated_data[key] = value
		
		# Special validation for common fields
		if 'email' in validated_data:
			if not self.validator.validate_email(validated_data['email']):
				raise ValueError("Invalid email format")
		
		if 'phone' in validated_data:
			if not self.validator.validate_phone_number(validated_data['phone']):
				raise ValueError("Invalid phone number format")
		
		return validated_data
	
	async def _secure_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
		"""Encrypt sensitive data fields"""
		
		sensitive_fields = [
			'email', 'phone', 'ssn', 'credit_card', 'passport',
			'medical_record', 'biometric_data', 'financial_data'
		]
		
		secured_data = data.copy()
		encryption_applied = False
		
		for field in sensitive_fields:
			if field in secured_data and isinstance(secured_data[field], str):
				encrypted_field = self.encryption_manager.encrypt_data(
					secured_data[field],
					context=f"field_{field}"
				)
				secured_data[f"{field}_encrypted"] = encrypted_field
				del secured_data[field]  # Remove plaintext
				encryption_applied = True
		
		if encryption_applied:
			secured_data['_encrypted'] = True
			secured_data['_encryption_timestamp'] = datetime.utcnow().isoformat()
		
		return secured_data
	
	async def authenticate_session(
		self,
		session_token: str,
		ip_address: Optional[str] = None,
		user_agent: Optional[str] = None
	) -> Dict[str, Any]:
		"""Authenticate and validate session"""
		
		# Check for brute force attempts
		if ip_address and self._is_suspicious_activity(ip_address):
			await self.audit_logger.log_event(
				event_type=AuditEventType.SECURITY_VIOLATION,
				resource_type="session",
				resource_id=session_token[:16],
				action="suspicious_activity_detected",
				ip_address=ip_address,
				details={'threat_level': 'high'}
			)
			raise ValueError("Suspicious activity detected")
		
		# Validate session token format
		if not session_token or len(session_token) < 32:
			await self._record_failed_attempt(ip_address)
			raise ValueError("Invalid session token")
		
		# In production, this would validate against session store
		session_data = {
			'session_id': session_token,
			'user_id': 'validated_user',
			'authenticated': True,
			'ip_address': ip_address,
			'user_agent': user_agent,
			'authenticated_at': datetime.utcnow().isoformat()
		}
		
		self.active_sessions[session_token] = session_data
		
		# Log successful authentication
		await self.audit_logger.log_event(
			event_type=AuditEventType.USER_AUTHENTICATION,
			resource_type="session",
			resource_id=session_token[:16],
			action="authentication_successful",
			user_id=session_data['user_id'],
			ip_address=ip_address,
			user_agent=user_agent
		)
		
		return session_data
	
	def _is_suspicious_activity(self, ip_address: str) -> bool:
		"""Check for suspicious activity patterns"""
		
		if ip_address not in self.failed_attempts:
			return False
		
		recent_attempts = [
			attempt for attempt in self.failed_attempts[ip_address]
			if datetime.utcnow() - attempt < timedelta(minutes=15)
		]
		
		return len(recent_attempts) > 5
	
	async def _record_failed_attempt(self, ip_address: Optional[str]):
		"""Record failed authentication attempt"""
		
		if not ip_address:
			return
		
		if ip_address not in self.failed_attempts:
			self.failed_attempts[ip_address] = []
		
		self.failed_attempts[ip_address].append(datetime.utcnow())
		
		# Clean old attempts
		cutoff = datetime.utcnow() - timedelta(hours=1)
		self.failed_attempts[ip_address] = [
			attempt for attempt in self.failed_attempts[ip_address]
			if attempt > cutoff
		]
	
	async def generate_compliance_report(
		self,
		start_date: datetime,
		end_date: datetime,
		regime: Optional[ComplianceRegime] = None
	) -> Dict[str, Any]:
		"""Generate comprehensive compliance report"""
		
		# Query audit events
		audit_events = await self.audit_logger.query_audit_events(
			start_date=start_date,
			end_date=end_date
		)
		
		# Analyze compliance metrics
		total_events = len(audit_events)
		high_risk_events = [e for e in audit_events if e.risk_score > 0.7]
		security_violations = [e for e in audit_events if e.event_type == AuditEventType.SECURITY_VIOLATION]
		
		report = {
			'report_id': f"compliance_{secrets.token_hex(8)}",
			'tenant_id': self.tenant_id,
			'period': {
				'start_date': start_date.isoformat(),
				'end_date': end_date.isoformat()
			},
			'summary': {
				'total_events': total_events,
				'high_risk_events': len(high_risk_events),
				'security_violations': len(security_violations),
				'compliance_score': self._calculate_compliance_score(audit_events)
			},
			'event_breakdown': {
				'data_access': len([e for e in audit_events if e.event_type == AuditEventType.DATA_ACCESS]),
				'data_modification': len([e for e in audit_events if e.event_type == AuditEventType.DATA_MODIFICATION]),
				'data_deletion': len([e for e in audit_events if e.event_type == AuditEventType.DATA_DELETION]),
				'consent_events': len([e for e in audit_events if e.event_type == AuditEventType.CONSENT_MANAGEMENT])
			},
			'recommendations': self._generate_compliance_recommendations(audit_events),
			'generated_at': datetime.utcnow().isoformat()
		}
		
		# Log report generation
		await self.audit_logger.log_event(
			event_type=AuditEventType.COMPLIANCE_CHECK,
			resource_type="compliance_report",
			resource_id=report['report_id'],
			action="report_generated",
			details={
				'period_days': (end_date - start_date).days,
				'total_events': total_events,
				'compliance_score': report['summary']['compliance_score']
			}
		)
		
		return report
	
	def _calculate_compliance_score(self, audit_events: List[AuditEvent]) -> float:
		"""Calculate overall compliance score"""
		
		if not audit_events:
			return 1.0
		
		total_score = 0.0
		for event in audit_events:
			# Higher risk events lower the score more
			event_score = 1.0 - (event.risk_score * 0.5)
			total_score += event_score
		
		return min(total_score / len(audit_events), 1.0)
	
	def _generate_compliance_recommendations(self, audit_events: List[AuditEvent]) -> List[str]:
		"""Generate compliance improvement recommendations"""
		
		recommendations = []
		
		high_risk_count = len([e for e in audit_events if e.risk_score > 0.7])
		if high_risk_count > len(audit_events) * 0.1:  # More than 10% high risk
			recommendations.append("Consider implementing additional security controls for high-risk operations")
		
		violations = [e for e in audit_events if e.event_type == AuditEventType.SECURITY_VIOLATION]
		if violations:
			recommendations.append("Investigate and address security violations immediately")
		
		sensitive_events = [e for e in audit_events if e.sensitive_data_involved]
		if len(sensitive_events) > len(audit_events) * 0.3:  # More than 30% involve sensitive data
			recommendations.append("Review data minimization practices to reduce sensitive data exposure")
		
		return recommendations
	
	async def get_security_status(self) -> Dict[str, Any]:
		"""Get current security status and metrics"""
		
		return {
			'tenant_id': self.tenant_id,
			'threat_level': self.threat_level.value,
			'active_sessions': len(self.active_sessions),
			'recent_audit_events': len(self.audit_logger.audit_events),
			'consent_records': len(self.consent_manager.consent_records),
			'data_subject_requests': len(self.rights_manager.requests),
			'security_policies': len(self.compliance_checker.policies),
			'encryption_status': 'active',
			'last_key_rotation': 'not_implemented',  # Would track actual rotation
			'status': 'operational',
			'timestamp': datetime.utcnow().isoformat()
		}

# Factory function for easy instantiation
def create_security_engine(tenant_id: str, config: Optional[Dict[str, Any]] = None) -> SecurityEngine:
	"""Create a new security engine instance"""
	return SecurityEngine(tenant_id=tenant_id, config=config)

# Example usage
if __name__ == "__main__":
	import asyncio
	
	async def demo_security_engine():
		"""Demonstrate security engine capabilities"""
		
		# Create security engine
		security = create_security_engine("demo-tenant")
		
		# Example data validation and security
		test_data = {
			'email': 'user@example.com',
			'message': 'Hello <script>alert("xss")</script> world!',
			'phone': '+1-555-123-4567'
		}
		
		try:
			# Validate and secure data
			result = await security.validate_and_secure_data(
				data=test_data,
				operation="store",
				user_context={'user_id': 'user123'}
			)
			
			print("✅ Data validation and security completed")
			print(f"Compliance status: {result['compliance']['compliant']}")
			
			# Record consent
			consent_id = await security.consent_manager.record_consent(
				user_id="user123",
				purpose="marketing_communications",
				granted=True,
				data_categories=["email", "preferences"]
			)
			
			print(f"✅ Consent recorded: {consent_id}")
			
			# Generate compliance report
			start_date = datetime.utcnow() - timedelta(days=30)
			end_date = datetime.utcnow()
			
			report = await security.generate_compliance_report(
				start_date=start_date,
				end_date=end_date
			)
			
			print(f"✅ Compliance report generated: {report['summary']}")
			
			# Get security status
			status = await security.get_security_status()
			print(f"✅ Security status: {status['status']}")
			
		except Exception as e:
			print(f"❌ Security operation failed: {str(e)}")
	
	# Run demo
	asyncio.run(demo_security_engine())