#!/usr/bin/env python3
"""
APG Message Queue Event Bus (MQEB) - Compliance and Data Governance
Automated compliance management and data governance for enterprise messaging

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import re
import json
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import uuid

from .models import MQMessage, TopicConfiguration
from .service import MQEBService


class ComplianceFramework(str, Enum):
	"""Supported compliance frameworks"""
	GDPR = "gdpr"
	HIPAA = "hipaa" 
	PCI_DSS = "pci_dss"
	SOX = "sox"
	ISO_27001 = "iso_27001"
	CCPA = "ccpa"
	NIST = "nist"


class DataClassification(str, Enum):
	"""Data classification levels"""
	PUBLIC = "public"
	INTERNAL = "internal"
	CONFIDENTIAL = "confidential"
	RESTRICTED = "restricted"
	TOP_SECRET = "top_secret"


class PIIType(str, Enum):
	"""Types of personally identifiable information"""
	EMAIL = "email"
	PHONE = "phone"
	SSN = "ssn"
	CREDIT_CARD = "credit_card"
	BANK_ACCOUNT = "bank_account"
	IP_ADDRESS = "ip_address"
	BIOMETRIC = "biometric"
	LOCATION = "location"
	NAME = "name"
	ADDRESS = "address"


class RetentionAction(str, Enum):
	"""Actions for data retention policy"""
	DELETE = "delete"
	ARCHIVE = "archive"
	ANONYMIZE = "anonymize"
	NOTIFY = "notify"


@dataclass
class PIIDetectionResult:
	"""Result of PII detection in message"""
	message_id: str
	pii_types: List[PIIType]
	confidence_scores: Dict[PIIType, float]
	detected_patterns: Dict[PIIType, List[str]]
	risk_level: str
	recommendations: List[str]


@dataclass
class ComplianceRule:
	"""Compliance rule definition"""
	rule_id: str
	framework: ComplianceFramework
	name: str
	description: str
	category: str
	severity: str  # low, medium, high, critical
	conditions: Dict[str, Any]
	actions: List[str]
	enabled: bool = True


@dataclass
class DataRetentionPolicy:
	"""Data retention policy configuration"""
	policy_id: str
	name: str
	description: str
	tenant_id: str
	topic_patterns: List[str]
	retention_period_days: int
	retention_action: RetentionAction
	compliance_frameworks: List[ComplianceFramework]
	metadata_retention_days: Optional[int] = None
	encryption_required: bool = True
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComplianceViolation:
	"""Compliance violation record"""
	violation_id: str
	message_id: str
	tenant_id: str
	framework: ComplianceFramework
	rule_id: str
	severity: str
	description: str
	detected_at: datetime
	remediation_actions: List[str]
	resolved: bool = False
	resolved_at: Optional[datetime] = None


@dataclass
class AuditLogEntry:
	"""Immutable audit log entry"""
	log_id: str
	timestamp: datetime
	tenant_id: str
	user_id: Optional[str]
	action: str
	resource_type: str
	resource_id: str
	details: Dict[str, Any]
	compliance_frameworks: List[ComplianceFramework]
	integrity_hash: str = field(init=False)
	
	def __post_init__(self):
		"""Calculate integrity hash for tamper detection"""
		content = f"{self.timestamp}{self.tenant_id}{self.action}{self.resource_id}{json.dumps(self.details, sort_keys=True)}"
		self.integrity_hash = hashlib.sha256(content.encode()).hexdigest()


class PIIDetectionEngine:
	"""Advanced PII detection using pattern matching and ML"""
	
	def __init__(self):
		self.pii_patterns = self._initialize_pii_patterns()
		self.ml_models = {}  # Placeholder for ML models
		self.detection_history = deque(maxlen=10000)
		
		self.logger = logging.getLogger('mqeb.pii_detection')
	
	def _initialize_pii_patterns(self) -> Dict[PIIType, List[re.Pattern]]:
		"""Initialize PII detection patterns"""
		return {
			PIIType.EMAIL: [
				re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE)
			],
			PIIType.PHONE: [
				re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
				re.compile(r'\b\d{3}-\d{3}-\d{4}\b'),
				re.compile(r'\b\(\d{3}\)\s?\d{3}-\d{4}\b')
			],
			PIIType.SSN: [
				re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
				re.compile(r'\b\d{9}\b')
			],
			PIIType.CREDIT_CARD: [
				re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b')
			],
			PIIType.IP_ADDRESS: [
				re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
			],
			PIIType.ADDRESS: [
				re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b', re.IGNORECASE)
			]
		}
	
	async def detect_pii(self, message: MQMessage) -> PIIDetectionResult:
		"""Detect PII in message content"""
		try:
			# Decode message content
			try:
				content = message.payload.decode('utf-8', errors='ignore')
			except:
				content = str(message.payload)
			
			detected_pii = {}
			confidence_scores = {}
			detected_patterns = {}
			
			# Pattern-based detection
			for pii_type, patterns in self.pii_patterns.items():
				matches = []
				for pattern in patterns:
					pattern_matches = pattern.findall(content)
					matches.extend(pattern_matches)
				
				if matches:
					detected_pii[pii_type] = True
					confidence_scores[pii_type] = min(1.0, len(matches) * 0.3)
					detected_patterns[pii_type] = matches[:5]  # Limit to first 5 matches
				else:
					confidence_scores[pii_type] = 0.0
			
			# Additional contextual analysis
			content_lower = content.lower()
			
			# Name detection (simplified)
			if any(word in content_lower for word in ['firstname', 'lastname', 'fullname', 'name']):
				detected_pii[PIIType.NAME] = True
				confidence_scores[PIIType.NAME] = 0.7
			
			# Location detection
			location_indicators = ['latitude', 'longitude', 'address', 'location', 'gps']
			if any(indicator in content_lower for indicator in location_indicators):
				detected_pii[PIIType.LOCATION] = True
				confidence_scores[PIIType.LOCATION] = 0.6
			
			# Calculate overall risk level
			detected_types = list(detected_pii.keys())
			risk_level = self._calculate_risk_level(detected_types, confidence_scores)
			
			# Generate recommendations
			recommendations = self._generate_pii_recommendations(detected_types, message)
			
			result = PIIDetectionResult(
				message_id=message.id,
				pii_types=detected_types,
				confidence_scores=confidence_scores,
				detected_patterns=detected_patterns,
				risk_level=risk_level,
				recommendations=recommendations
			)
			
			# Record detection for analytics
			self.detection_history.append({
				'timestamp': datetime.utcnow(),
				'message_id': message.id,
				'pii_types': detected_types,
				'risk_level': risk_level
			})
			
			return result
			
		except Exception as e:
			self.logger.error(f"PII detection failed for message {message.id}: {e}")
			return PIIDetectionResult(
				message_id=message.id,
				pii_types=[],
				confidence_scores={},
				detected_patterns={},
				risk_level='unknown',
				recommendations=['pii_detection_failed']
			)
	
	def _calculate_risk_level(self, pii_types: List[PIIType], confidence_scores: Dict[PIIType, float]) -> str:
		"""Calculate overall PII risk level"""
		if not pii_types:
			return 'low'
		
		# High-risk PII types
		high_risk_types = {PIIType.SSN, PIIType.CREDIT_CARD, PIIType.BIOMETRIC}
		medium_risk_types = {PIIType.EMAIL, PIIType.PHONE, PIIType.ADDRESS}
		
		has_high_risk = any(pii_type in high_risk_types for pii_type in pii_types)
		has_medium_risk = any(pii_type in medium_risk_types for pii_type in pii_types)
		
		# Calculate weighted score
		total_score = sum(confidence_scores.get(pii_type, 0) for pii_type in pii_types)
		avg_confidence = total_score / len(pii_types) if pii_types else 0
		
		if has_high_risk or (len(pii_types) > 3 and avg_confidence > 0.8):
			return 'critical'
		elif has_medium_risk or (len(pii_types) > 2 and avg_confidence > 0.6):
			return 'high'
		elif len(pii_types) > 1 or avg_confidence > 0.7:
			return 'medium'
		else:
			return 'low'
	
	def _generate_pii_recommendations(self, pii_types: List[PIIType], message: MQMessage) -> List[str]:
		"""Generate recommendations for PII handling"""
		recommendations = []
		
		if not pii_types:
			return recommendations
		
		# General recommendations
		recommendations.append('enable_encryption')
		recommendations.append('restrict_access')
		recommendations.append('enable_audit_logging')
		
		# Specific recommendations by PII type
		if PIIType.CREDIT_CARD in pii_types:
			recommendations.extend(['pci_compliance_required', 'tokenize_card_data'])
		
		if PIIType.SSN in pii_types:
			recommendations.extend(['redact_ssn', 'extra_access_controls'])
		
		if any(pii in pii_types for pii in [PIIType.EMAIL, PIIType.NAME, PIIType.ADDRESS]):
			recommendations.extend(['gdpr_compliance_check', 'data_subject_rights'])
		
		if PIIType.LOCATION in pii_types:
			recommendations.extend(['location_data_consent', 'anonymize_location'])
		
		# Message-specific recommendations
		if not message.encrypted:
			recommendations.append('immediate_encryption_required')
		
		return recommendations


class ComplianceRuleEngine:
	"""Engine for evaluating compliance rules"""
	
	def __init__(self):
		self.rules: Dict[str, ComplianceRule] = {}
		self.violations: List[ComplianceViolation] = []
		self.rule_cache = {}
		
		self._initialize_default_rules()
		self.logger = logging.getLogger('mqeb.compliance_rules')
	
	def _initialize_default_rules(self):
		"""Initialize default compliance rules"""
		
		# GDPR Rules
		self.rules['gdpr_encryption'] = ComplianceRule(
			rule_id='gdpr_encryption',
			framework=ComplianceFramework.GDPR,
			name='Personal Data Encryption',
			description='Personal data must be encrypted in transit and at rest',
			category='data_protection',
			severity='high',
			conditions={
				'has_pii': True,
				'encryption_required': True
			},
			actions=['encrypt_message', 'audit_access']
		)
		
		self.rules['gdpr_retention'] = ComplianceRule(
			rule_id='gdpr_retention',
			framework=ComplianceFramework.GDPR,
			name='Data Retention Limits',
			description='Personal data must not be retained longer than necessary',
			category='data_minimization',
			severity='medium',
			conditions={
				'has_pii': True,
				'retention_period_exceeded': True
			},
			actions=['schedule_deletion', 'notify_data_controller']
		)
		
		# HIPAA Rules
		self.rules['hipaa_access_control'] = ComplianceRule(
			rule_id='hipaa_access_control',
			framework=ComplianceFramework.HIPAA,
			name='Health Information Access Control',
			description='Access to PHI must be restricted to authorized individuals',
			category='access_control',
			severity='critical',
			conditions={
				'contains_phi': True,
				'unauthorized_access': True
			},
			actions=['deny_access', 'audit_access_attempt', 'notify_security']
		)
		
		# PCI DSS Rules
		self.rules['pci_cardholder_data'] = ComplianceRule(
			rule_id='pci_cardholder_data',
			framework=ComplianceFramework.PCI_DSS,
			name='Cardholder Data Protection',
			description='Cardholder data must be protected per PCI DSS requirements',
			category='data_protection',
			severity='critical',
			conditions={
				'has_credit_card_data': True
			},
			actions=['encrypt_message', 'mask_card_data', 'restrict_access']
		)
		
		# SOX Rules
		self.rules['sox_financial_controls'] = ComplianceRule(
			rule_id='sox_financial_controls',
			framework=ComplianceFramework.SOX,
			name='Financial Data Controls',
			description='Financial data must have proper controls and audit trails',
			category='financial_reporting',
			severity='high',
			conditions={
				'contains_financial_data': True
			},
			actions=['enable_audit_trail', 'segregate_duties', 'approve_changes']
		)
	
	async def evaluate_message_compliance(self, message: MQMessage, 
										 pii_result: PIIDetectionResult,
										 context: Dict[str, Any]) -> List[ComplianceViolation]:
		"""Evaluate message against compliance rules"""
		violations = []
		
		# Prepare evaluation context
		eval_context = {
			'message': message,
			'pii_result': pii_result,
			'has_pii': len(pii_result.pii_types) > 0,
			'has_credit_card_data': PIIType.CREDIT_CARD in pii_result.pii_types,
			'contains_phi': self._detect_phi(message),
			'contains_financial_data': self._detect_financial_data(message),
			'encryption_required': not message.encrypted,
			'unauthorized_access': context.get('unauthorized_access', False),
			'retention_period_exceeded': await self._check_retention_period(message),
			**context
		}
		
		# Evaluate each rule
		for rule_id, rule in self.rules.items():
			if not rule.enabled:
				continue
			
			try:
				if await self._evaluate_rule_conditions(rule, eval_context):
					violation = ComplianceViolation(
						violation_id=f"viol_{secrets.token_hex(8)}",
						message_id=message.id,
						tenant_id=message.tenant_id,
						framework=rule.framework,
						rule_id=rule.rule_id,
						severity=rule.severity,
						description=f"{rule.name}: {rule.description}",
						detected_at=datetime.utcnow(),
						remediation_actions=rule.actions
					)
					violations.append(violation)
					
			except Exception as e:
				self.logger.error(f"Error evaluating rule {rule_id}: {e}")
		
		# Store violations
		self.violations.extend(violations)
		
		if violations:
			self.logger.warning(f"Found {len(violations)} compliance violations for message {message.id}")
		
		return violations
	
	async def _evaluate_rule_conditions(self, rule: ComplianceRule, context: Dict[str, Any]) -> bool:
		"""Evaluate if rule conditions are met"""
		for condition_key, expected_value in rule.conditions.items():
			actual_value = context.get(condition_key, False)
			
			if isinstance(expected_value, bool):
				if actual_value != expected_value:
					return False
			elif isinstance(expected_value, (int, float)):
				if actual_value < expected_value:
					return False
			elif isinstance(expected_value, str):
				if str(actual_value) != expected_value:
					return False
		
		return True
	
	def _detect_phi(self, message: MQMessage) -> bool:
		"""Detect protected health information (simplified)"""
		try:
			content = message.payload.decode('utf-8', errors='ignore').lower()
			phi_indicators = [
				'medical', 'health', 'patient', 'diagnosis', 'treatment',
				'medication', 'hospital', 'doctor', 'physician', 'healthcare'
			]
			return any(indicator in content for indicator in phi_indicators)
		except:
			return False
	
	def _detect_financial_data(self, message: MQMessage) -> bool:
		"""Detect financial data (simplified)"""
		try:
			content = message.payload.decode('utf-8', errors='ignore').lower()
			financial_indicators = [
				'payment', 'transaction', 'invoice', 'billing', 'revenue',
				'expense', 'financial', 'accounting', 'budget', 'cost'
			]
			return any(indicator in content for indicator in financial_indicators)
		except:
			return False
	
	async def _check_retention_period(self, message: MQMessage) -> bool:
		"""Check if message retention period is exceeded (simplified)"""
		# Simple check: messages older than 1 year
		age_days = (datetime.utcnow() - message.timestamp).days
		return age_days > 365


class DataGovernanceEngine:
	"""Data governance and lifecycle management"""
	
	def __init__(self):
		self.retention_policies: Dict[str, DataRetentionPolicy] = {}
		self.data_classifications: Dict[str, DataClassification] = {}
		self.audit_logs: List[AuditLogEntry] = []
		self.governance_metrics = defaultdict(int)
		
		self.logger = logging.getLogger('mqeb.data_governance')
	
	async def classify_message_data(self, message: MQMessage, pii_result: PIIDetectionResult) -> DataClassification:
		"""Automatically classify message data"""
		try:
			# Classification based on PII content
			if pii_result.risk_level == 'critical':
				classification = DataClassification.TOP_SECRET
			elif pii_result.risk_level == 'high':
				classification = DataClassification.RESTRICTED
			elif pii_result.risk_level == 'medium':
				classification = DataClassification.CONFIDENTIAL
			elif len(pii_result.pii_types) > 0:
				classification = DataClassification.INTERNAL
			else:
				classification = DataClassification.PUBLIC
			
			# Topic-based classification refinement
			topic_lower = message.topic.lower()
			if any(keyword in topic_lower for keyword in ['public', 'marketing', 'announcement']):
				classification = DataClassification.PUBLIC
			elif any(keyword in topic_lower for keyword in ['internal', 'employee']):
				classification = max(classification, DataClassification.INTERNAL)
			elif any(keyword in topic_lower for keyword in ['confidential', 'private']):
				classification = max(classification, DataClassification.CONFIDENTIAL)
			elif any(keyword in topic_lower for keyword in ['secret', 'classified']):
				classification = DataClassification.TOP_SECRET
			
			# Store classification
			self.data_classifications[message.id] = classification
			self.governance_metrics['messages_classified'] += 1
			
			self.logger.debug(f"Message {message.id} classified as {classification.value}")
			return classification
			
		except Exception as e:
			self.logger.error(f"Data classification failed for message {message.id}: {e}")
			return DataClassification.INTERNAL  # Default to internal
	
	async def apply_retention_policy(self, message: MQMessage, policy_id: str) -> bool:
		"""Apply data retention policy to message"""
		try:
			policy = self.retention_policies.get(policy_id)
			if not policy:
				self.logger.error(f"Retention policy {policy_id} not found")
				return False
			
			# Check if message matches policy criteria
			if not self._message_matches_policy(message, policy):
				return True  # No action needed
			
			# Calculate retention deadline
			retention_deadline = message.timestamp + timedelta(days=policy.retention_period_days)
			current_time = datetime.utcnow()
			
			if current_time >= retention_deadline:
				# Execute retention action
				success = await self._execute_retention_action(message, policy)
				
				if success:
					# Log retention action
					await self._log_audit_event(
						action=f"retention_{policy.retention_action.value}",
						resource_type="message",
						resource_id=message.id,
						tenant_id=message.tenant_id,
						details={
							'policy_id': policy_id,
							'retention_period_days': policy.retention_period_days,
							'retention_action': policy.retention_action.value,
							'message_age_days': (current_time - message.timestamp).days
						},
						compliance_frameworks=policy.compliance_frameworks
					)
					
					self.governance_metrics['retention_actions_executed'] += 1
				
				return success
			
			return True  # Not yet time for retention action
			
		except Exception as e:
			self.logger.error(f"Failed to apply retention policy {policy_id} to message {message.id}: {e}")
			return False
	
	def _message_matches_policy(self, message: MQMessage, policy: DataRetentionPolicy) -> bool:
		"""Check if message matches retention policy criteria"""
		# Check tenant
		if message.tenant_id != policy.tenant_id:
			return False
		
		# Check topic patterns
		import fnmatch
		topic_matches = any(
			fnmatch.fnmatch(message.topic, pattern) 
			for pattern in policy.topic_patterns
		)
		
		return topic_matches
	
	async def _execute_retention_action(self, message: MQMessage, policy: DataRetentionPolicy) -> bool:
		"""Execute retention action on message"""
		try:
			if policy.retention_action == RetentionAction.DELETE:
				# Mark message for deletion
				# In a real system, this would trigger actual deletion
				message.headers['retention_status'] = 'deleted'
				message.headers['deleted_at'] = datetime.utcnow().isoformat()
				self.logger.info(f"Message {message.id} marked for deletion per policy {policy.policy_id}")
				
			elif policy.retention_action == RetentionAction.ARCHIVE:
				# Move message to archive storage
				message.headers['retention_status'] = 'archived'
				message.headers['archived_at'] = datetime.utcnow().isoformat()
				self.logger.info(f"Message {message.id} archived per policy {policy.policy_id}")
				
			elif policy.retention_action == RetentionAction.ANONYMIZE:
				# Anonymize PII in message
				await self._anonymize_message(message)
				message.headers['retention_status'] = 'anonymized'
				message.headers['anonymized_at'] = datetime.utcnow().isoformat()
				self.logger.info(f"Message {message.id} anonymized per policy {policy.policy_id}")
				
			elif policy.retention_action == RetentionAction.NOTIFY:
				# Send notification about retention
				self.logger.info(f"Retention notification sent for message {message.id}")
			
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to execute retention action {policy.retention_action} on message {message.id}: {e}")
			return False
	
	async def _anonymize_message(self, message: MQMessage):
		"""Anonymize PII in message payload"""
		try:
			content = message.payload.decode('utf-8', errors='ignore')
			
			# Simple anonymization patterns
			anonymization_patterns = [
				(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]'),
				(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]'),
				(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[CARD_REDACTED]'),
				(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', '[PHONE_REDACTED]')
			]
			
			for pattern, replacement in anonymization_patterns:
				content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
			
			message.payload = content.encode('utf-8')
			
		except Exception as e:
			self.logger.error(f"Failed to anonymize message {message.id}: {e}")
	
	async def _log_audit_event(self, action: str, resource_type: str, resource_id: str,
							  tenant_id: str, details: Dict[str, Any],
							  compliance_frameworks: List[ComplianceFramework],
							  user_id: Optional[str] = None):
		"""Log audit event with integrity protection"""
		try:
			audit_entry = AuditLogEntry(
				log_id=f"audit_{uuid.uuid4()}",
				timestamp=datetime.utcnow(),
				tenant_id=tenant_id,
				user_id=user_id,
				action=action,
				resource_type=resource_type,
				resource_id=resource_id,
				details=details,
				compliance_frameworks=compliance_frameworks
			)
			
			self.audit_logs.append(audit_entry)
			self.governance_metrics['audit_events_logged'] += 1
			
		except Exception as e:
			self.logger.error(f"Failed to log audit event: {e}")
	
	async def create_retention_policy(self, policy: DataRetentionPolicy) -> str:
		"""Create new data retention policy"""
		try:
			self.retention_policies[policy.policy_id] = policy
			
			await self._log_audit_event(
				action="create_retention_policy",
				resource_type="retention_policy",
				resource_id=policy.policy_id,
				tenant_id=policy.tenant_id,
				details={
					'policy_name': policy.name,
					'retention_period_days': policy.retention_period_days,
					'retention_action': policy.retention_action.value,
					'topic_patterns': policy.topic_patterns
				},
				compliance_frameworks=policy.compliance_frameworks
			)
			
			self.logger.info(f"Created retention policy {policy.policy_id}")
			return policy.policy_id
			
		except Exception as e:
			self.logger.error(f"Failed to create retention policy: {e}")
			raise
	
	async def get_compliance_report(self, tenant_id: str, frameworks: List[ComplianceFramework],
								   start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Generate comprehensive compliance report"""
		try:
			# Filter audit logs for period and frameworks
			relevant_logs = [
				log for log in self.audit_logs
				if (log.tenant_id == tenant_id and
					start_date <= log.timestamp <= end_date and
					any(framework in log.compliance_frameworks for framework in frameworks))
			]
			
			report = {
				'tenant_id': tenant_id,
				'frameworks': [f.value for f in frameworks],
				'report_period': {
					'start': start_date.isoformat(),
					'end': end_date.isoformat()
				},
				'generated_at': datetime.utcnow().isoformat(),
				'summary': {
					'total_audit_events': len(relevant_logs),
					'governance_actions': len([log for log in relevant_logs if log.action.startswith('retention_')]),
					'policy_violations': len([log for log in relevant_logs if log.action == 'compliance_violation']),
					'data_classifications': self.governance_metrics.get('messages_classified', 0),
					'retention_actions': self.governance_metrics.get('retention_actions_executed', 0)
				},
				'audit_events': [
					{
						'timestamp': log.timestamp.isoformat(),
						'action': log.action,
						'resource_type': log.resource_type,
						'resource_id': log.resource_id,
						'details': log.details
					}
					for log in relevant_logs[-100:]  # Last 100 events
				],
				'retention_policies': [
					{
						'policy_id': policy.policy_id,
						'name': policy.name,
						'retention_period_days': policy.retention_period_days,
						'retention_action': policy.retention_action.value,
						'topic_patterns': policy.topic_patterns
					}
					for policy in self.retention_policies.values()
					if policy.tenant_id == tenant_id and
					any(framework in policy.compliance_frameworks for framework in frameworks)
				]
			}
			
			return report
			
		except Exception as e:
			self.logger.error(f"Failed to generate compliance report: {e}")
			raise


class ComplianceGovernanceEngine:
	"""Main compliance and governance orchestration engine"""
	
	def __init__(self, mqeb_service: MQEBService):
		self.service = mqeb_service
		self.pii_detector = PIIDetectionEngine()
		self.rule_engine = ComplianceRuleEngine()
		self.governance = DataGovernanceEngine()
		
		# Configuration
		self.enabled = True
		self.auto_classification = True
		self.auto_retention = True
		
		# Background tasks
		self._background_tasks: Set[asyncio.Task] = set()
		
		self.logger = logging.getLogger('mqeb.compliance_governance')
	
	async def initialize(self) -> None:
		"""Initialize compliance governance engine"""
		self.logger.info("Initializing compliance governance engine...")
		
		# Start background tasks
		await self._start_background_tasks()
		
		self.logger.info("Compliance governance engine initialized")
	
	async def shutdown(self) -> None:
		"""Shutdown compliance governance engine"""
		self.enabled = False
		
		# Cancel background tasks
		for task in self._background_tasks:
			task.cancel()
		
		await asyncio.gather(*self._background_tasks, return_exceptions=True)
		self.logger.info("Compliance governance engine shut down")
	
	async def process_message_compliance(self, message: MQMessage, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Process message for compliance and governance"""
		try:
			result = {
				'compliant': True,
				'violations': [],
				'pii_detected': False,
				'data_classification': DataClassification.PUBLIC.value,
				'governance_actions': []
			}
			
			if not self.enabled:
				return result
			
			# PII Detection
			pii_result = await self.pii_detector.detect_pii(message)
			result['pii_detected'] = len(pii_result.pii_types) > 0
			result['pii_details'] = {
				'types': [pii.value for pii in pii_result.pii_types],
				'risk_level': pii_result.risk_level,
				'recommendations': pii_result.recommendations
			}
			
			# Data Classification
			if self.auto_classification:
				classification = await self.governance.classify_message_data(message, pii_result)
				result['data_classification'] = classification.value
			
			# Compliance Rule Evaluation
			violations = await self.rule_engine.evaluate_message_compliance(
				message, pii_result, context
			)
			
			if violations:
				result['compliant'] = False
				result['violations'] = [
					{
						'framework': v.framework.value,
						'rule_id': v.rule_id,
						'severity': v.severity,
						'description': v.description,
						'remediation_actions': v.remediation_actions
					}
					for v in violations
				]
			
			# Data Governance Actions
			if self.auto_retention:
				governance_actions = await self._apply_governance_policies(message, context)
				result['governance_actions'] = governance_actions
			
			return result
			
		except Exception as e:
			self.logger.error(f"Compliance processing failed for message {message.id}: {e}")
			return {
				'compliant': False,
				'error': str(e),
				'violations': [],
				'pii_detected': False,
				'data_classification': DataClassification.INTERNAL.value,
				'governance_actions': []
			}
	
	async def _apply_governance_policies(self, message: MQMessage, context: Dict[str, Any]) -> List[str]:
		"""Apply data governance policies"""
		actions = []
		
		try:
			# Apply retention policies
			for policy_id in self.governance.retention_policies:
				success = await self.governance.apply_retention_policy(message, policy_id)
				if success:
					actions.append(f"retention_policy_applied:{policy_id}")
			
		except Exception as e:
			self.logger.error(f"Failed to apply governance policies: {e}")
		
		return actions
	
	async def _start_background_tasks(self) -> None:
		"""Start background compliance and governance tasks"""
		
		# Compliance monitoring task
		task = asyncio.create_task(self._compliance_monitoring_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Data retention task
		task = asyncio.create_task(self._data_retention_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Audit log integrity task
		task = asyncio.create_task(self._audit_integrity_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
	
	async def _compliance_monitoring_loop(self) -> None:
		"""Background compliance monitoring"""
		while self.enabled:
			try:
				await asyncio.sleep(3600)  # Check every hour
				
				# Monitor compliance violations
				critical_violations = [
					v for v in self.rule_engine.violations
					if v.severity == 'critical' and not v.resolved
				]
				
				if critical_violations:
					self.logger.critical(f"Found {len(critical_violations)} unresolved critical violations")
				
				# Cleanup old violations
				cutoff_date = datetime.utcnow() - timedelta(days=90)
				self.rule_engine.violations = [
					v for v in self.rule_engine.violations
					if v.detected_at > cutoff_date
				]
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Compliance monitoring error: {e}")
	
	async def _data_retention_loop(self) -> None:
		"""Background data retention processing"""
		while self.enabled:
			try:
				await asyncio.sleep(86400)  # Check daily
				
				# Process retention policies for all messages
				# In a real system, this would query the message database
				self.logger.info("Daily data retention processing completed")
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Data retention processing error: {e}")
	
	async def _audit_integrity_loop(self) -> None:
		"""Background audit log integrity verification"""
		while self.enabled:
			try:
				await asyncio.sleep(3600)  # Check every hour
				
				# Verify audit log integrity
				tampered_logs = []
				for log in self.governance.audit_logs:
					# Recalculate integrity hash
					content = f"{log.timestamp}{log.tenant_id}{log.action}{log.resource_id}{json.dumps(log.details, sort_keys=True)}"
					expected_hash = hashlib.sha256(content.encode()).hexdigest()
					
					if log.integrity_hash != expected_hash:
						tampered_logs.append(log.log_id)
				
				if tampered_logs:
					self.logger.critical(f"Detected {len(tampered_logs)} tampered audit logs: {tampered_logs}")
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Audit integrity check error: {e}")
	
	async def get_compliance_status(self) -> Dict[str, Any]:
		"""Get current compliance status"""
		return {
			'enabled': self.enabled,
			'total_violations': len(self.rule_engine.violations),
			'critical_violations': len([v for v in self.rule_engine.violations if v.severity == 'critical']),
			'pii_detections': len(self.pii_detector.detection_history),
			'data_classifications': self.governance.governance_metrics.get('messages_classified', 0),
			'retention_policies': len(self.governance.retention_policies),
			'audit_logs': len(self.governance.audit_logs),
			'governance_metrics': dict(self.governance.governance_metrics)
		}


# Factory function
async def create_compliance_governance_engine(mqeb_service: MQEBService) -> ComplianceGovernanceEngine:
	"""Create and initialize compliance governance engine"""
	engine = ComplianceGovernanceEngine(mqeb_service)
	await engine.initialize()
	return engine


# Export components
__all__ = [
	'ComplianceGovernanceEngine', 'PIIDetectionEngine', 'ComplianceRuleEngine', 'DataGovernanceEngine',
	'ComplianceFramework', 'DataClassification', 'PIIType', 'RetentionAction',
	'PIIDetectionResult', 'ComplianceRule', 'DataRetentionPolicy', 'ComplianceViolation', 'AuditLogEntry',
	'create_compliance_governance_engine'
]