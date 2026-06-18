"""
APG NLP Enterprise Security and Compliance Engine

Comprehensive security, privacy, and compliance framework for enterprise NLP operations.
Integrates with APG's auth_rbac system for advanced access control and audit trails.

Features:
- GDPR and data privacy compliance
- Advanced audit logging and compliance reporting
- Role-based access control for NLP operations
- Data retention and deletion policies
- Security monitoring and threat detection
- Compliance dashboard and reporting
"""

import asyncio
import json
import logging
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from contextlib import asynccontextmanager
from uuid_extensions import uuid7str
import re

from .models import (
	TextDocument, ProcessingRequest, ProcessingResult, NLPTaskType,
	LanguageCode, ProcessingStatus, AnnotationProject, TextAnnotation
)

# Configure logging
logger = logging.getLogger(__name__)

class ComplianceFramework(str, Enum):
	"""Supported compliance frameworks"""
	GDPR = "gdpr"
	CCPA = "ccpa"
	HIPAA = "hipaa"
	SOC2 = "soc2"
	ISO27001 = "iso27001"
	PCI_DSS = "pci_dss"
	CUSTOM = "custom"

class DataClassification(str, Enum):
	"""Data sensitivity classification levels"""
	PUBLIC = "public"
	INTERNAL = "internal"
	CONFIDENTIAL = "confidential"
	RESTRICTED = "restricted"
	TOP_SECRET = "top_secret"

class SecurityEvent(str, Enum):
	"""Security event types for monitoring"""
	ACCESS_GRANTED = "access_granted"
	ACCESS_DENIED = "access_denied"
	DATA_ACCESSED = "data_accessed"
	DATA_MODIFIED = "data_modified"
	DATA_DELETED = "data_deleted"
	POLICY_VIOLATION = "policy_violation"
	SUSPICIOUS_ACTIVITY = "suspicious_activity"
	AUTHENTICATION_FAILURE = "authentication_failure"
	PRIVILEGE_ESCALATION = "privilege_escalation"
	DATA_EXPORT = "data_export"

class RetentionAction(str, Enum):
	"""Data retention actions"""
	RETAIN = "retain"
	ARCHIVE = "archive"
	DELETE = "delete"
	ANONYMIZE = "anonymize"
	PSEUDONYMIZE = "pseudonymize"

@dataclass
class SecurityContext:
	"""Security context for NLP operations"""
	user_id: str
	tenant_id: str
	session_id: str = field(default_factory=uuid7str)
	ip_address: Optional[str] = None
	user_agent: Optional[str] = None
	authentication_method: Optional[str] = None
	roles: List[str] = field(default_factory=list)
	permissions: Set[str] = field(default_factory=set)
	security_clearance: Optional[DataClassification] = None
	request_timestamp: datetime = field(default_factory=datetime.utcnow)
	geographic_location: Optional[Dict[str, str]] = None
	risk_score: float = 0.0

@dataclass
class CompliancePolicy:
	"""Compliance policy definition"""
	policy_id: str = field(default_factory=uuid7str)
	tenant_id: str = ""
	name: str = ""
	framework: ComplianceFramework = ComplianceFramework.GDPR
	description: str = ""
	
	# Policy rules
	data_classification_rules: Dict[str, Any] = field(default_factory=dict)
	retention_rules: Dict[str, Any] = field(default_factory=dict)
	access_control_rules: Dict[str, Any] = field(default_factory=dict)
	processing_restrictions: Dict[str, Any] = field(default_factory=dict)
	
	# Policy status
	is_active: bool = True
	version: str = "1.0.0"
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)
	created_by: str = ""

@dataclass
class AuditEvent:
	"""Comprehensive audit event for compliance tracking"""
	event_id: str = field(default_factory=uuid7str)
	tenant_id: str = ""
	event_type: SecurityEvent = SecurityEvent.DATA_ACCESSED
	event_category: str = "nlp_processing"
	
	# Event context
	user_id: str = ""
	session_id: str = ""
	resource_type: str = ""
	resource_id: str = ""
	action_performed: str = ""
	
	# Security context
	ip_address: Optional[str] = None
	user_agent: Optional[str] = None
	authentication_method: Optional[str] = None
	geographic_location: Optional[Dict[str, str]] = None
	
	# Event details
	success: bool = True
	error_message: Optional[str] = None
	risk_score: float = 0.0
	sensitive_data_accessed: bool = False
	data_classification: Optional[DataClassification] = None
	
	# Compliance tracking
	compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
	policy_violations: List[str] = field(default_factory=list)
	consent_status: Optional[str] = None
	
	# Technical details
	request_data: Dict[str, Any] = field(default_factory=dict)
	response_data: Dict[str, Any] = field(default_factory=dict)
	processing_time_ms: float = 0.0
	
	# APG audit fields
	timestamp: datetime = field(default_factory=datetime.utcnow)
	correlation_id: Optional[str] = None

@dataclass
class DataRetentionRecord:
	"""Data retention tracking record"""
	record_id: str = field(default_factory=uuid7str)
	tenant_id: str = ""
	resource_type: str = ""
	resource_id: str = ""
	
	# Retention policy
	retention_policy_id: str = ""
	data_classification: DataClassification = DataClassification.INTERNAL
	retention_period_days: int = 365
	
	# Retention status
	created_at: datetime = field(default_factory=datetime.utcnow)
	retention_expires_at: datetime = field(default_factory=datetime.utcnow)
	last_accessed_at: Optional[datetime] = None
	deletion_scheduled_at: Optional[datetime] = None
	deletion_completed_at: Optional[datetime] = None
	
	# Retention actions
	scheduled_action: RetentionAction = RetentionAction.DELETE
	action_completed: bool = False
	action_metadata: Dict[str, Any] = field(default_factory=dict)

class SecurityComplianceEngine:
	"""Enterprise security and compliance engine for APG NLP"""
	
	def __init__(self, tenant_id: str, config: Dict[str, Any] = None):
		assert tenant_id, "Tenant ID is required for security compliance engine"
		
		self.tenant_id = tenant_id
		self.config = config or {}
		
		# Security state
		self.compliance_policies: Dict[str, CompliancePolicy] = {}
		self.audit_events: deque = deque(maxlen=10000)  # Keep last 10K events in memory
		self.security_contexts: Dict[str, SecurityContext] = {}
		self.retention_records: Dict[str, DataRetentionRecord] = {}
		
		# Security monitoring
		self.threat_patterns: Dict[str, Dict[str, Any]] = {}
		self.risk_thresholds: Dict[str, float] = {}
		self.active_alerts: Dict[str, Dict[str, Any]] = {}
		
		# Data classification cache
		self.classification_cache: Dict[str, DataClassification] = {}
		self.consent_records: Dict[str, Dict[str, Any]] = {}
		
		self._setup_security_config()
		self._initialize_compliance_frameworks()
		self._log_engine_initialized()
	
	def _setup_security_config(self) -> None:
		"""Setup security and compliance configuration"""
		self.audit_retention_days = self.config.get("audit_retention_days", 2555)  # 7 years
		self.real_time_monitoring = self.config.get("real_time_monitoring", True)
		self.risk_scoring_enabled = self.config.get("risk_scoring_enabled", True)
		self.automatic_classification = self.config.get("automatic_classification", True)
		self.threat_detection_enabled = self.config.get("threat_detection_enabled", True)
		
		# Security thresholds
		self.max_failed_attempts = self.config.get("max_failed_attempts", 5)
		self.suspicious_activity_threshold = self.config.get("suspicious_activity_threshold", 0.7)
		self.high_risk_threshold = self.config.get("high_risk_threshold", 0.8)
		self.data_export_approval_required = self.config.get("data_export_approval_required", True)
	
	def _initialize_compliance_frameworks(self) -> None:
		"""Initialize compliance framework policies"""
		# GDPR default policy
		gdpr_policy = CompliancePolicy(
			tenant_id=self.tenant_id,
			name="GDPR Data Protection Policy",
			framework=ComplianceFramework.GDPR,
			description="EU General Data Protection Regulation compliance policy",
			data_classification_rules={
				"personal_data_categories": [
					"names", "email_addresses", "phone_numbers", "ip_addresses",
					"biometric_data", "health_records", "financial_data"
				],
				"special_categories": [
					"racial_origin", "political_opinions", "religious_beliefs",
					"health_data", "sexual_orientation", "genetic_data"
				],
				"auto_classify": True,
				"require_consent": True
			},
			retention_rules={
				"default_retention_days": 1095,  # 3 years
				"personal_data_retention_days": 730,  # 2 years
				"special_category_retention_days": 365,  # 1 year
				"automatic_deletion": True,
				"right_to_erasure": True
			},
			access_control_rules={
				"data_minimization": True,
				"purpose_limitation": True,
				"consent_required": True,
				"legitimate_interest_assessment": True
			},
			created_by="system"
		)
		
		self.compliance_policies[gdpr_policy.policy_id] = gdpr_policy
		
		# Initialize threat detection patterns
		self._initialize_threat_patterns()
	
	def _initialize_threat_patterns(self) -> None:
		"""Initialize threat detection patterns"""
		self.threat_patterns = {
			"unusual_access_pattern": {
				"description": "Unusual data access patterns",
				"indicators": ["high_volume_access", "off_hours_access", "geographic_anomaly"],
				"risk_score": 0.7,
				"response_actions": ["log_alert", "require_additional_auth"]
			},
			"data_exfiltration": {
				"description": "Potential data exfiltration attempt",
				"indicators": ["bulk_export", "sensitive_data_access", "unusual_download_volume"],
				"risk_score": 0.9,
				"response_actions": ["block_action", "immediate_alert", "admin_notification"]
			},
			"privilege_escalation": {
				"description": "Privilege escalation attempt",
				"indicators": ["permission_boundary_test", "unauthorized_resource_access"],
				"risk_score": 0.8,
				"response_actions": ["log_alert", "security_review", "access_restriction"]
			},
			"policy_violation": {
				"description": "Compliance policy violation",
				"indicators": ["retention_policy_breach", "unauthorized_processing"],
				"risk_score": 0.6,
				"response_actions": ["log_violation", "compliance_review"]
			}
		}
	
	def _log_engine_initialized(self) -> None:
		"""Log security engine initialization"""
		logger.info(f"Security compliance engine initialized for tenant: {self.tenant_id}")
	
	async def create_security_context(self, user_id: str, request_data: Dict[str, Any]) -> SecurityContext:
		"""Create security context for user request"""
		context = SecurityContext(
			user_id=user_id,
			tenant_id=self.tenant_id,
			ip_address=request_data.get("ip_address"),
			user_agent=request_data.get("user_agent"),
			authentication_method=request_data.get("auth_method", "unknown"),
			geographic_location=request_data.get("geo_location")
		)
		
		# Get user roles and permissions from APG auth_rbac
		context.roles = await self._get_user_roles(user_id)
		context.permissions = await self._get_user_permissions(user_id)
		context.security_clearance = await self._get_security_clearance(user_id)
		
		# Calculate risk score
		context.risk_score = await self._calculate_risk_score(context, request_data)
		
		# Store context
		self.security_contexts[context.session_id] = context
		
		self._log_context_created(context.session_id, user_id, context.risk_score)
		
		return context
	
	async def _get_user_roles(self, user_id: str) -> List[str]:
		"""Get user roles from APG auth_rbac"""
		# Integration with APG's auth_rbac capability
		# This would typically call the auth_rbac service
		default_roles = ["nlp_user"]
		return default_roles
	
	async def _get_user_permissions(self, user_id: str) -> Set[str]:
		"""Get user permissions from APG auth_rbac"""
		# Integration with APG's auth_rbac capability
		default_permissions = {
			"nlp:read", "nlp:process", "document:read"
		}
		return default_permissions
	
	async def _get_security_clearance(self, user_id: str) -> Optional[DataClassification]:
		"""Get user security clearance level"""
		# This would integrate with organizational directory/LDAP
		return DataClassification.INTERNAL
	
	async def _calculate_risk_score(self, context: SecurityContext, request_data: Dict[str, Any]) -> float:
		"""Calculate risk score for security context"""
		if not self.risk_scoring_enabled:
			return 0.0
		
		risk_factors = []
		
		# Geographic risk
		if context.geographic_location:
			risk_countries = self.config.get("high_risk_countries", [])
			if context.geographic_location.get("country") in risk_countries:
				risk_factors.append(0.3)
		
		# Time-based risk
		current_hour = datetime.utcnow().hour
		if current_hour < 6 or current_hour > 22:  # Off-hours access
			risk_factors.append(0.2)
		
		# Authentication method risk
		auth_method = context.authentication_method
		if auth_method == "password_only":
			risk_factors.append(0.4)
		elif auth_method == "mfa":
			risk_factors.append(-0.2)  # Reduce risk
		
		# Request pattern analysis
		recent_requests = await self._get_recent_user_requests(context.user_id)
		if len(recent_requests) > 100:  # High volume
			risk_factors.append(0.3)
		
		# Calculate final risk score
		base_risk = 0.1
		total_risk = base_risk + sum(risk_factors)
		return min(max(total_risk, 0.0), 1.0)
	
	async def _get_recent_user_requests(self, user_id: str) -> List[Dict[str, Any]]:
		"""Get recent requests for user activity analysis"""
		recent_events = [
			event for event in self.audit_events
			if event.user_id == user_id and 
			event.timestamp > datetime.utcnow() - timedelta(hours=1)
		]
		return recent_events
	
	def _log_context_created(self, session_id: str, user_id: str, risk_score: float) -> None:
		"""Log security context creation"""
		logger.info(f"Security context created: {session_id} (user: {user_id}, risk: {risk_score:.3f})")
	
	async def authorize_operation(self, context: SecurityContext, operation: str, 
								 resource_type: str, resource_id: str = None) -> bool:
		"""Authorize NLP operation based on security context and policies"""
		
		# Check basic permissions
		required_permission = f"{resource_type}:{operation}"
		if required_permission not in context.permissions:
			await self._log_security_event(
				context, SecurityEvent.ACCESS_DENIED,
				resource_type, resource_id, operation,
				success=False, error_message="Insufficient permissions"
			)
			return False
		
		# Check data classification access
		if resource_id:
			data_classification = await self._get_resource_classification(resource_type, resource_id)
			if not await self._check_classification_access(context, data_classification):
				await self._log_security_event(
					context, SecurityEvent.ACCESS_DENIED,
					resource_type, resource_id, operation,
					success=False, error_message="Insufficient security clearance"
				)
				return False
		
		# Risk-based access control
		if context.risk_score > self.high_risk_threshold:
			if not await self._additional_verification_required(context, operation):
				await self._log_security_event(
					context, SecurityEvent.ACCESS_DENIED,
					resource_type, resource_id, operation,
					success=False, error_message="High risk score - additional verification required"
				)
				return False
		
		# Check compliance policies
		policy_violations = await self._check_compliance_policies(context, operation, resource_type)
		if policy_violations:
			await self._log_security_event(
				context, SecurityEvent.POLICY_VIOLATION,
				resource_type, resource_id, operation,
				success=False, policy_violations=policy_violations
			)
			return False
		
		# Log successful authorization
		await self._log_security_event(
			context, SecurityEvent.ACCESS_GRANTED,
			resource_type, resource_id, operation,
			success=True
		)
		
		return True
	
	async def _get_resource_classification(self, resource_type: str, resource_id: str) -> DataClassification:
		"""Get data classification for resource"""
		cache_key = f"{resource_type}:{resource_id}"
		
		if cache_key in self.classification_cache:
			return self.classification_cache[cache_key]
		
		# Auto-classify based on content if enabled
		if self.automatic_classification:
			classification = await self._auto_classify_resource(resource_type, resource_id)
		else:
			classification = DataClassification.INTERNAL  # Default
		
		# Cache result
		self.classification_cache[cache_key] = classification
		return classification
	
	async def _auto_classify_resource(self, resource_type: str, resource_id: str) -> DataClassification:
		"""Automatically classify resource based on content analysis"""
		
		if resource_type == "document":
			# Analyze document content for sensitive information
			content_analysis = await self._analyze_document_sensitivity(resource_id)
			
			if content_analysis.get("contains_pii", False):
				return DataClassification.CONFIDENTIAL
			elif content_analysis.get("contains_financial", False):
				return DataClassification.RESTRICTED
			elif content_analysis.get("contains_health", False):
				return DataClassification.RESTRICTED
			else:
				return DataClassification.INTERNAL
		
		elif resource_type == "annotation_project":
			# Classification based on annotation type and content
			return DataClassification.CONFIDENTIAL
		
		else:
			return DataClassification.INTERNAL
	
	async def _analyze_document_sensitivity(self, document_id: str) -> Dict[str, bool]:
		"""Analyze document for sensitive information patterns"""
		# This would integrate with the document content
		# For now, return mock analysis
		sensitivity_analysis = {
			"contains_pii": False,
			"contains_financial": False,
			"contains_health": False,
			"contains_biometric": False
		}
		
		# Pattern matching for sensitive data would go here
		# Example patterns:
		patterns = {
			"email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
			"phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
			"ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
			"credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
		}
		
		# This would analyze actual document content
		# sensitivity_analysis["contains_pii"] = bool(re.search(patterns["email"], document_content))
		
		return sensitivity_analysis
	
	async def _check_classification_access(self, context: SecurityContext, 
										  classification: DataClassification) -> bool:
		"""Check if user has access to data classification level"""
		if not context.security_clearance:
			return classification == DataClassification.PUBLIC
		
		# Define access hierarchy
		clearance_levels = {
			DataClassification.PUBLIC: 0,
			DataClassification.INTERNAL: 1,
			DataClassification.CONFIDENTIAL: 2,
			DataClassification.RESTRICTED: 3,
			DataClassification.TOP_SECRET: 4
		}
		
		user_level = clearance_levels.get(context.security_clearance, 0)
		required_level = clearance_levels.get(classification, 0)
		
		return user_level >= required_level
	
	async def _additional_verification_required(self, context: SecurityContext, operation: str) -> bool:
		"""Check if additional verification is required for high-risk operations"""
		# High-risk operations that require additional verification
		high_risk_operations = ["export", "bulk_download", "delete", "modify_permissions"]
		
		if operation in high_risk_operations:
			# Would integrate with MFA or additional authentication
			return False  # Assume additional verification not provided
		
		return True
	
	async def _check_compliance_policies(self, context: SecurityContext, operation: str, 
										resource_type: str) -> List[str]:
		"""Check operation against compliance policies"""
		violations = []
		
		for policy in self.compliance_policies.values():
			if not policy.is_active:
				continue
			
			# Check processing restrictions
			restrictions = policy.processing_restrictions
			if restrictions.get("require_consent", False):
				if not await self._has_valid_consent(context.user_id, resource_type):
					violations.append(f"Policy {policy.name}: Consent required")
			
			# Check purpose limitation
			if restrictions.get("purpose_limitation", False):
				if not await self._check_purpose_limitation(operation, resource_type):
					violations.append(f"Policy {policy.name}: Purpose limitation violation")
			
			# Check data minimization
			if restrictions.get("data_minimization", False):
				if not await self._check_data_minimization(operation, resource_type):
					violations.append(f"Policy {policy.name}: Data minimization violation")
		
		return violations
	
	async def _has_valid_consent(self, user_id: str, resource_type: str) -> bool:
		"""Check if user has valid consent for data processing"""
		consent_key = f"{user_id}:{resource_type}"
		consent_record = self.consent_records.get(consent_key)
		
		if not consent_record:
			return False
		
		# Check if consent is still valid
		expiry = consent_record.get("expires_at")
		if expiry and datetime.fromisoformat(expiry) < datetime.utcnow():
			return False
		
		return consent_record.get("granted", False)
	
	async def _check_purpose_limitation(self, operation: str, resource_type: str) -> bool:
		"""Check if operation aligns with stated processing purposes"""
		# This would check against stated purposes for data collection
		# For now, return True (no violation)
		return True
	
	async def _check_data_minimization(self, operation: str, resource_type: str) -> bool:
		"""Check if operation follows data minimization principles"""
		# This would ensure only necessary data is being processed
		# For now, return True (no violation)
		return True
	
	async def _log_security_event(self, context: SecurityContext, event_type: SecurityEvent,
								 resource_type: str, resource_id: Optional[str], action: str,
								 success: bool = True, error_message: Optional[str] = None,
								 policy_violations: List[str] = None) -> None:
		"""Log security event for audit trail"""
		
		event = AuditEvent(
			tenant_id=self.tenant_id,
			event_type=event_type,
			user_id=context.user_id,
			session_id=context.session_id,
			resource_type=resource_type,
			resource_id=resource_id or "",
			action_performed=action,
			ip_address=context.ip_address,
			user_agent=context.user_agent,
			authentication_method=context.authentication_method,
			geographic_location=context.geographic_location,
			success=success,
			error_message=error_message,
			risk_score=context.risk_score,
			policy_violations=policy_violations or []
		)
		
		# Determine if sensitive data was accessed
		if resource_id:
			classification = await self._get_resource_classification(resource_type, resource_id)
			event.sensitive_data_accessed = classification in [
				DataClassification.CONFIDENTIAL,
				DataClassification.RESTRICTED,
				DataClassification.TOP_SECRET
			]
			event.data_classification = classification
		
		# Set applicable compliance frameworks
		event.compliance_frameworks = [policy.framework for policy in self.compliance_policies.values()]
		
		# Store audit event
		self.audit_events.append(event)
		
		# Trigger real-time monitoring if enabled
		if self.real_time_monitoring:
			await self._process_security_event(event)
		
		self._log_audit_event_created(event.event_id, event_type, success)
	
	async def _process_security_event(self, event: AuditEvent) -> None:
		"""Process security event for real-time monitoring and threat detection"""
		
		if not self.threat_detection_enabled:
			return
		
		# Analyze event for threat patterns
		threats_detected = await self._analyze_threat_patterns(event)
		
		for threat in threats_detected:
			await self._handle_threat_detection(event, threat)
		
		# Update risk scores and monitoring metrics
		await self._update_security_metrics(event)
	
	async def _analyze_threat_patterns(self, event: AuditEvent) -> List[Dict[str, Any]]:
		"""Analyze event against known threat patterns"""
		detected_threats = []
		
		for pattern_name, pattern_config in self.threat_patterns.items():
			indicators = pattern_config["indicators"]
			matches = 0
			
			# Check for pattern indicators
			if "high_volume_access" in indicators:
				recent_events = await self._get_recent_user_requests(event.user_id)
				if len(recent_events) > 50:  # High volume threshold
					matches += 1
			
			if "off_hours_access" in indicators:
				current_hour = event.timestamp.hour
				if current_hour < 6 or current_hour > 22:
					matches += 1
			
			if "geographic_anomaly" in indicators:
				if await self._is_geographic_anomaly(event):
					matches += 1
			
			if "bulk_export" in indicators:
				if "export" in event.action_performed and event.sensitive_data_accessed:
					matches += 1
			
			if "unauthorized_resource_access" in indicators:
				if not event.success and event.event_type == SecurityEvent.ACCESS_DENIED:
					matches += 1
			
			# Threat detected if majority of indicators match
			if matches >= len(indicators) * 0.6:  # 60% threshold
				detected_threats.append({
					"pattern_name": pattern_name,
					"pattern_config": pattern_config,
					"matches": matches,
					"indicators_matched": matches
				})
		
		return detected_threats
	
	async def _is_geographic_anomaly(self, event: AuditEvent) -> bool:
		"""Check if event represents geographic anomaly"""
		if not event.geographic_location:
			return False
		
		# Get user's typical locations
		recent_events = await self._get_recent_user_requests(event.user_id)
		typical_countries = set()
		
		for recent_event in recent_events:
			if hasattr(recent_event, 'geographic_location') and recent_event.geographic_location:
				typical_countries.add(recent_event.geographic_location.get("country"))
		
		current_country = event.geographic_location.get("country")
		return current_country not in typical_countries
	
	async def _handle_threat_detection(self, event: AuditEvent, threat: Dict[str, Any]) -> None:
		"""Handle detected threat based on response actions"""
		pattern_config = threat["pattern_config"]
		response_actions = pattern_config["response_actions"]
		
		alert_data = {
			"alert_id": uuid7str(),
			"threat_type": threat["pattern_name"],
			"event_id": event.event_id,
			"user_id": event.user_id,
			"risk_score": pattern_config["risk_score"],
			"indicators_matched": threat["indicators_matched"],
			"timestamp": datetime.utcnow(),
			"status": "active"
		}
		
		# Execute response actions
		for action in response_actions:
			if action == "log_alert":
				await self._create_security_alert(alert_data)
			elif action == "immediate_alert":
				await self._send_immediate_alert(alert_data)
			elif action == "admin_notification":
				await self._notify_administrators(alert_data)
			elif action == "block_action":
				await self._block_user_action(event.user_id, event.session_id)
			elif action == "require_additional_auth":
				await self._require_additional_authentication(event.user_id)
		
		self._log_threat_detected(threat["pattern_name"], event.user_id)
	
	async def _create_security_alert(self, alert_data: Dict[str, Any]) -> None:
		"""Create security alert for monitoring"""
		self.active_alerts[alert_data["alert_id"]] = alert_data
		logger.warning(f"Security alert created: {alert_data['threat_type']} for user {alert_data['user_id']}")
	
	async def _send_immediate_alert(self, alert_data: Dict[str, Any]) -> None:
		"""Send immediate alert to security team"""
		# Integration with notification system
		logger.critical(f"IMMEDIATE SECURITY ALERT: {alert_data['threat_type']} - Risk Score: {alert_data['risk_score']}")
	
	async def _notify_administrators(self, alert_data: Dict[str, Any]) -> None:
		"""Notify system administrators of security event"""
		# Integration with admin notification system
		logger.error(f"Administrator notification: Security threat detected - {alert_data['threat_type']}")
	
	async def _block_user_action(self, user_id: str, session_id: str) -> None:
		"""Block user action for security reasons"""
		# This would integrate with session management to block further actions
		logger.warning(f"User action blocked for security: {user_id} (session: {session_id})")
	
	async def _require_additional_authentication(self, user_id: str) -> None:
		"""Require additional authentication for user"""
		# Integration with authentication system
		logger.info(f"Additional authentication required for user: {user_id}")
	
	async def _update_security_metrics(self, event: AuditEvent) -> None:
		"""Update security monitoring metrics"""
		# Update threat detection metrics
		# This would typically integrate with monitoring systems
		pass
	
	def _log_threat_detected(self, threat_type: str, user_id: str) -> None:
		"""Log threat detection"""
		logger.warning(f"Security threat detected: {threat_type} for user {user_id}")
	
	def _log_audit_event_created(self, event_id: str, event_type: SecurityEvent, success: bool) -> None:
		"""Log audit event creation"""
		logger.info(f"Audit event created: {event_id} ({event_type}, success: {success})")
	
	async def create_retention_policy(self, resource_type: str, resource_id: str,
									 classification: DataClassification,
									 retention_days: Optional[int] = None) -> DataRetentionRecord:
		"""Create data retention record for resource"""
		
		# Determine retention period based on classification and compliance policies
		if retention_days is None:
			retention_days = await self._get_default_retention_days(classification)
		
		retention_record = DataRetentionRecord(
			tenant_id=self.tenant_id,
			resource_type=resource_type,
			resource_id=resource_id,
			data_classification=classification,
			retention_period_days=retention_days,
			retention_expires_at=datetime.utcnow() + timedelta(days=retention_days)
		)
		
		# Store retention record
		self.retention_records[retention_record.record_id] = retention_record
		
		# Schedule automatic deletion if policy allows
		await self._schedule_retention_action(retention_record)
		
		self._log_retention_policy_created(retention_record.record_id, resource_type, retention_days)
		
		return retention_record
	
	async def _get_default_retention_days(self, classification: DataClassification) -> int:
		"""Get default retention period for data classification"""
		retention_defaults = {
			DataClassification.PUBLIC: 1095,        # 3 years
			DataClassification.INTERNAL: 730,       # 2 years
			DataClassification.CONFIDENTIAL: 365,   # 1 year
			DataClassification.RESTRICTED: 180,     # 6 months
			DataClassification.TOP_SECRET: 90       # 3 months
		}
		
		return retention_defaults.get(classification, 365)  # Default 1 year
	
	async def _schedule_retention_action(self, retention_record: DataRetentionRecord) -> None:
		"""Schedule retention action for data"""
		# This would integrate with a job scheduler
		scheduled_time = retention_record.retention_expires_at
		retention_record.deletion_scheduled_at = scheduled_time
		
		logger.info(f"Retention action scheduled for {scheduled_time}: {retention_record.record_id}")
	
	def _log_retention_policy_created(self, record_id: str, resource_type: str, retention_days: int) -> None:
		"""Log retention policy creation"""
		logger.info(f"Retention policy created: {record_id} ({resource_type}, {retention_days} days)")
	
	async def process_data_deletion_request(self, user_id: str, resource_type: str, 
										   resource_id: str, reason: str = "user_request") -> bool:
		"""Process data deletion request (Right to Erasure / Right to be Forgotten)"""
		
		# Find retention record
		retention_record = None
		for record in self.retention_records.values():
			if (record.resource_type == resource_type and 
				record.resource_id == resource_id):
				retention_record = record
				break
		
		if not retention_record:
			logger.warning(f"No retention record found for {resource_type}:{resource_id}")
			return False
		
		# Check if deletion is allowed
		if not await self._can_delete_data(retention_record, reason):
			return False
		
		# Execute deletion
		success = await self._execute_data_deletion(retention_record)
		
		if success:
			retention_record.deletion_completed_at = datetime.utcnow()
			retention_record.action_completed = True
			retention_record.action_metadata = {
				"deletion_reason": reason,
				"requested_by": user_id,
				"completion_method": "immediate_deletion"
			}
			
			# Log deletion for audit trail
			await self._log_data_deletion(retention_record, user_id, reason)
		
		return success
	
	async def _can_delete_data(self, retention_record: DataRetentionRecord, reason: str) -> bool:
		"""Check if data can be deleted based on policies and legal requirements"""
		
		# Check legal hold requirements
		if await self._has_legal_hold(retention_record):
			return False
		
		# Check regulatory requirements
		if await self._has_regulatory_hold(retention_record):
			return False
		
		# Check business requirements
		if await self._has_business_hold(retention_record):
			return False
		
		return True
	
	async def _has_legal_hold(self, retention_record: DataRetentionRecord) -> bool:
		"""Check if data is under legal hold"""
		# This would integrate with legal hold management system
		return False
	
	async def _has_regulatory_hold(self, retention_record: DataRetentionRecord) -> bool:
		"""Check if data is under regulatory hold"""
		# Check regulatory retention requirements
		return False
	
	async def _has_business_hold(self, retention_record: DataRetentionRecord) -> bool:
		"""Check if data is under business hold"""
		# Check business retention requirements
		return False
	
	async def _execute_data_deletion(self, retention_record: DataRetentionRecord) -> bool:
		"""Execute data deletion from all systems"""
		try:
			# Delete from primary storage
			await self._delete_from_primary_storage(retention_record)
			
			# Delete from backups (if policy requires)
			await self._delete_from_backups(retention_record)
			
			# Delete from caches
			await self._delete_from_caches(retention_record)
			
			# Delete from audit logs (if allowed by policy)
			await self._clean_audit_logs(retention_record)
			
			return True
		
		except Exception as e:
			logger.error(f"Data deletion failed for {retention_record.record_id}: {str(e)}")
			return False
	
	async def _delete_from_primary_storage(self, retention_record: DataRetentionRecord) -> None:
		"""Delete data from primary storage"""
		# This would integrate with the main database/storage systems
		pass
	
	async def _delete_from_backups(self, retention_record: DataRetentionRecord) -> None:
		"""Delete data from backup systems"""
		# This would integrate with backup management systems
		pass
	
	async def _delete_from_caches(self, retention_record: DataRetentionRecord) -> None:
		"""Delete data from cache systems"""
		# Clear from classification cache
		cache_keys_to_remove = [
			key for key in self.classification_cache.keys()
			if retention_record.resource_id in key
		]
		
		for key in cache_keys_to_remove:
			del self.classification_cache[key]
	
	async def _clean_audit_logs(self, retention_record: DataRetentionRecord) -> None:
		"""Clean related audit log entries if policy allows"""
		# Some regulations require audit log retention even after data deletion
		# This would implement policy-based audit log cleaning
		pass
	
	async def _log_data_deletion(self, retention_record: DataRetentionRecord, 
								user_id: str, reason: str) -> None:
		"""Log data deletion for compliance audit trail"""
		deletion_event = AuditEvent(
			tenant_id=self.tenant_id,
			event_type=SecurityEvent.DATA_DELETED,
			user_id=user_id,
			resource_type=retention_record.resource_type,
			resource_id=retention_record.resource_id,
			action_performed="data_deletion",
			success=True,
			request_data={
				"deletion_reason": reason,
				"retention_record_id": retention_record.record_id,
				"data_classification": retention_record.data_classification.value
			}
		)
		
		self.audit_events.append(deletion_event)
		
		logger.info(f"Data deletion completed and logged: {retention_record.resource_id} (reason: {reason})")
	
	def get_compliance_dashboard(self) -> Dict[str, Any]:
		"""Get comprehensive compliance dashboard data"""
		
		# Calculate compliance metrics
		total_events = len(self.audit_events)
		failed_events = len([e for e in self.audit_events if not e.success])
		policy_violations = len([e for e in self.audit_events if e.policy_violations])
		
		# Security metrics
		high_risk_events = len([e for e in self.audit_events if e.risk_score > 0.7])
		active_alerts_count = len(self.active_alerts)
		
		# Data management metrics
		retention_records_count = len(self.retention_records)
		expired_records = len([
			r for r in self.retention_records.values()
			if r.retention_expires_at < datetime.utcnow() and not r.action_completed
		])
		
		# User activity metrics
		unique_users = len(set(e.user_id for e in self.audit_events))
		sensitive_data_accesses = len([e for e in self.audit_events if e.sensitive_data_accessed])
		
		return {
			"compliance_summary": {
				"total_policies": len(self.compliance_policies),
				"active_policies": len([p for p in self.compliance_policies.values() if p.is_active]),
				"compliance_frameworks": list(set(p.framework.value for p in self.compliance_policies.values()))
			},
			"security_metrics": {
				"total_events": total_events,
				"failed_events": failed_events,
				"success_rate": ((total_events - failed_events) / max(total_events, 1)) * 100,
				"policy_violations": policy_violations,
				"high_risk_events": high_risk_events,
				"active_alerts": active_alerts_count,
				"threat_patterns_configured": len(self.threat_patterns)
			},
			"data_management": {
				"retention_records": retention_records_count,
				"expired_records_pending": expired_records,
				"data_classifications": {
					cls.value: len([r for r in self.retention_records.values() if r.data_classification == cls])
					for cls in DataClassification
				}
			},
			"user_activity": {
				"unique_users": unique_users,
				"sensitive_data_accesses": sensitive_data_accesses,
				"average_risk_score": sum(e.risk_score for e in self.audit_events) / max(total_events, 1)
			},
			"recent_activity": {
				"last_24_hours": len([
					e for e in self.audit_events
					if e.timestamp > datetime.utcnow() - timedelta(days=1)
				]),
				"last_7_days": len([
					e for e in self.audit_events
					if e.timestamp > datetime.utcnow() - timedelta(days=7)
				])
			},
			"dashboard_generated": datetime.utcnow().isoformat()
		}
	
	def get_audit_report(self, start_date: datetime, end_date: datetime, 
						filters: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Generate comprehensive audit report for compliance"""
		filters = filters or {}
		
		# Filter events by date range
		filtered_events = [
			event for event in self.audit_events
			if start_date <= event.timestamp <= end_date
		]
		
		# Apply additional filters
		if "user_id" in filters:
			filtered_events = [e for e in filtered_events if e.user_id == filters["user_id"]]
		
		if "event_type" in filters:
			filtered_events = [e for e in filtered_events if e.event_type == filters["event_type"]]
		
		if "resource_type" in filters:
			filtered_events = [e for e in filtered_events if e.resource_type == filters["resource_type"]]
		
		# Generate report
		return {
			"report_metadata": {
				"start_date": start_date.isoformat(),
				"end_date": end_date.isoformat(),
				"filters_applied": filters,
				"total_events": len(filtered_events),
				"report_generated": datetime.utcnow().isoformat()
			},
			"security_summary": {
				"successful_events": len([e for e in filtered_events if e.success]),
				"failed_events": len([e for e in filtered_events if not e.success]),
				"policy_violations": len([e for e in filtered_events if e.policy_violations]),
				"high_risk_events": len([e for e in filtered_events if e.risk_score > 0.7])
			},
			"event_breakdown": {
				"by_type": {
					event_type.value: len([e for e in filtered_events if e.event_type == event_type])
					for event_type in SecurityEvent
				},
				"by_user": self._get_user_activity_breakdown(filtered_events),
				"by_resource_type": self._get_resource_breakdown(filtered_events)
			},
			"compliance_analysis": {
				"gdpr_events": len([e for e in filtered_events if ComplianceFramework.GDPR in e.compliance_frameworks]),
				"sensitive_data_events": len([e for e in filtered_events if e.sensitive_data_accessed]),
				"consent_related_events": len([e for e in filtered_events if e.consent_status])
			},
			"detailed_events": [
				{
					"event_id": event.event_id,
					"timestamp": event.timestamp.isoformat(),
					"event_type": event.event_type.value,
					"user_id": event.user_id,
					"resource_type": event.resource_type,
					"action": event.action_performed,
					"success": event.success,
					"risk_score": event.risk_score,
					"policy_violations": event.policy_violations
				}
				for event in filtered_events
			]
		}
	
	def _get_user_activity_breakdown(self, events: List[AuditEvent]) -> Dict[str, int]:
		"""Get user activity breakdown from events"""
		user_counts = defaultdict(int)
		for event in events:
			user_counts[event.user_id] += 1
		return dict(user_counts)
	
	def _get_resource_breakdown(self, events: List[AuditEvent]) -> Dict[str, int]:
		"""Get resource type breakdown from events"""
		resource_counts = defaultdict(int)
		for event in events:
			resource_counts[event.resource_type] += 1
		return dict(resource_counts)
	
	async def cleanup(self) -> None:
		"""Cleanup security compliance engine resources"""
		# Clear sensitive data from memory
		self.security_contexts.clear()
		self.classification_cache.clear()
		self.consent_records.clear()
		self.active_alerts.clear()
		
		# Archive audit events if needed
		if len(self.audit_events) > 0:
			await self._archive_audit_events()
		
		self.audit_events.clear()
		
		logger.info(f"Security compliance engine cleanup completed for tenant: {self.tenant_id}")
	
	async def _archive_audit_events(self) -> None:
		"""Archive audit events for long-term compliance storage"""
		# This would integrate with long-term audit storage systems
		logger.info(f"Archived {len(self.audit_events)} audit events for compliance")

# Export main classes
__all__ = [
	"SecurityComplianceEngine", "SecurityContext", "CompliancePolicy", 
	"AuditEvent", "DataRetentionRecord", "ComplianceFramework",
	"DataClassification", "SecurityEvent", "RetentionAction"
]