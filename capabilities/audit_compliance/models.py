"""
Audit & Compliance Management Models

Database models for comprehensive audit logging, compliance monitoring,
and regulatory reporting with tamper-proof storage and integrity verification.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import hashlib
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


def uuid7str():
	"""Generate UUID7 string for consistent ID generation"""
	from uuid_extensions import uuid7
	return str(uuid7())


class ACAuditLog(Model, AuditMixin, BaseMixin):
	"""
	Comprehensive audit event logging with tamper-proof storage.
	
	Records all system activities with detailed context information,
	cryptographic integrity verification, and compliance categorization.
	"""
	__tablename__ = 'ac_audit_log'
	
	# Identity
	log_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Event Classification
	event_type = Column(String(50), nullable=False, index=True)  # login, data_access, data_change, system_event
	event_category = Column(String(50), nullable=False, index=True)  # security, data, system, api, compliance
	event_source = Column(String(100), nullable=False)  # capability or service name
	severity = Column(String(20), default='info', index=True)  # debug, info, warn, error, critical
	
	# Actor Information
	user_id = Column(String(36), nullable=True, index=True)
	session_id = Column(String(128), nullable=True, index=True)
	impersonated_by = Column(String(36), nullable=True)  # For admin impersonation
	service_account = Column(String(100), nullable=True)  # For system actions
	
	# Action Details
	action = Column(String(100), nullable=False, index=True)
	resource_type = Column(String(100), nullable=True, index=True)
	resource_id = Column(String(200), nullable=True, index=True)
	resource_name = Column(String(500), nullable=True)
	
	# Context Information
	ip_address = Column(String(45), nullable=True, index=True)
	user_agent = Column(Text, nullable=True)
	request_id = Column(String(64), nullable=True, index=True)
	correlation_id = Column(String(64), nullable=True, index=True)
	
	# Data Changes (for data modification events)
	old_values = Column(JSON, nullable=True)  # Before state
	new_values = Column(JSON, nullable=True)  # After state
	changed_fields = Column(JSON, default=list)  # List of changed field names
	
	# Event Metadata
	event_data = Column(JSON, default=dict)  # Additional event-specific data
	tags = Column(JSON, default=list)  # Event tags for categorization
	business_context = Column(JSON, default=dict)  # Business process context
	
	# Performance Metrics
	processing_time_ms = Column(Float, nullable=True)
	response_size_bytes = Column(Integer, nullable=True)
	database_queries = Column(Integer, nullable=True)
	
	# Compliance Flags
	pii_accessed = Column(Boolean, default=False, index=True)
	sensitive_data = Column(Boolean, default=False, index=True)
	compliance_relevant = Column(Boolean, default=True, index=True)
	retention_class = Column(String(20), default='standard', index=True)  # standard, extended, permanent
	
	# Geographic and Legal Context
	jurisdiction = Column(String(10), nullable=True)  # ISO country code
	data_classification = Column(String(50), nullable=True)  # public, internal, confidential, restricted
	legal_hold = Column(Boolean, default=False, index=True)
	
	# Integrity and Verification
	event_hash = Column(String(64), nullable=False, index=True)  # SHA-256 hash for tamper detection
	signature = Column(String(1024), nullable=True)  # Digital signature
	merkle_root = Column(String(64), nullable=True)  # For blockchain anchoring
	
	# Relationships
	compliance_violations = relationship("ACComplianceViolation", back_populates="audit_log", cascade="all, delete-orphan")
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		# Calculate event hash on creation
		self.event_hash = self._calculate_event_hash()
	
	def __repr__(self):
		return f"<ACAuditLog {self.event_type}:{self.action} by {self.user_id or 'system'}>"
	
	def _calculate_event_hash(self) -> str:
		"""Calculate SHA-256 hash of event data for tamper detection"""
		hash_data = {
			'event_type': self.event_type,
			'event_category': self.event_category,
			'action': self.action,
			'user_id': self.user_id,
			'resource_type': self.resource_type,
			'resource_id': self.resource_id,
			'timestamp': self.created_on.isoformat() if self.created_on else None,
			'old_values': self.old_values,
			'new_values': self.new_values
		}
		
		# Create deterministic JSON string
		hash_string = json.dumps(hash_data, sort_keys=True, default=str)
		return hashlib.sha256(hash_string.encode()).hexdigest()
	
	def verify_integrity(self) -> bool:
		"""Verify event integrity by recalculating hash"""
		calculated_hash = self._calculate_event_hash()
		return calculated_hash == self.event_hash
	
	def is_compliance_relevant(self) -> bool:
		"""Check if event is relevant for compliance monitoring"""
		return (self.compliance_relevant or 
				self.pii_accessed or 
				self.sensitive_data or
				self.event_category in ['security', 'compliance', 'data'])
	
	def get_data_changes_summary(self) -> Dict[str, Any]:
		"""Get summary of data changes for this event"""
		if not self.old_values or not self.new_values:
			return {}
		
		changes = {}
		for field in self.changed_fields:
			changes[field] = {
				'old': self.old_values.get(field),
				'new': self.new_values.get(field)
			}
		
		return changes
	
	def add_compliance_violation(self, rule_id: str, severity: str, description: str) -> None:
		"""Add compliance violation associated with this audit event"""
		violation = ACComplianceViolation(
			audit_log_id=self.log_id,
			rule_id=rule_id,
			severity=severity,
			description=description,
			tenant_id=self.tenant_id
		)
		self.compliance_violations.append(violation)


class ACComplianceRule(Model, AuditMixin, BaseMixin):
	"""
	Configurable compliance rules and policies.
	
	Defines compliance monitoring rules with conditions, actions,
	and effectiveness tracking for automated compliance enforcement.
	"""
	__tablename__ = 'ac_compliance_rule'
	
	rule_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Rule Definition
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	rule_type = Column(String(50), nullable=False, index=True)  # access_control, data_retention, privacy, etc.
	compliance_framework = Column(String(50), nullable=False, index=True)  # GDPR, HIPAA, SOX, PCI_DSS, etc.
	
	# Rule Logic
	conditions = Column(JSON, nullable=False)  # Rule conditions in structured format
	actions = Column(JSON, default=list)  # Actions to take when rule is triggered
	severity = Column(String(20), default='medium', index=True)  # low, medium, high, critical
	
	# Rule Configuration
	is_active = Column(Boolean, default=True, index=True)
	auto_remediate = Column(Boolean, default=False)
	notification_enabled = Column(Boolean, default=True)
	
	# Scope and Targeting
	applicable_events = Column(JSON, default=list)  # Event types this rule applies to
	excluded_users = Column(JSON, default=list)  # Users exempt from this rule
	applicable_resources = Column(JSON, default=list)  # Resource types covered
	
	# Effectiveness Tracking
	triggered_count = Column(Integer, default=0)
	last_triggered = Column(DateTime, nullable=True, index=True)
	false_positive_count = Column(Integer, default=0)
	effectiveness_score = Column(Float, default=0.0)
	
	# Performance Metrics
	average_evaluation_time = Column(Float, default=0.0)  # milliseconds
	total_evaluations = Column(Integer, default=0)
	
	# Relationships
	violations = relationship("ACComplianceViolation", back_populates="rule", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<ACComplianceRule {self.name} ({self.compliance_framework})>"
	
	def evaluate_conditions(self, audit_event: ACAuditLog) -> bool:
		"""
		Evaluate rule conditions against an audit event.
		
		Args:
			audit_event: The audit event to evaluate
			
		Returns:
			True if conditions are met, False otherwise
		"""
		if not self.is_active:
			return False
		
		# Check if event type is applicable
		if self.applicable_events and audit_event.event_type not in self.applicable_events:
			return False
		
		# Check if user is excluded
		if audit_event.user_id in self.excluded_users:
			return False
		
		# Check if resource type is applicable
		if (self.applicable_resources and 
			audit_event.resource_type and 
			audit_event.resource_type not in self.applicable_resources):
			return False
		
		# Evaluate custom conditions
		return self._evaluate_custom_conditions(audit_event)
	
	def _evaluate_custom_conditions(self, audit_event: ACAuditLog) -> bool:
		"""Evaluate custom rule conditions"""
		try:
			# Simple condition evaluation - in production, use a proper rules engine
			for condition_group in self.conditions.get('groups', []):
				group_result = True
				
				for condition in condition_group.get('conditions', []):
					field = condition.get('field')
					operator = condition.get('operator')
					value = condition.get('value')
					
					# Get field value from audit event
					event_value = getattr(audit_event, field, None)
					if event_value is None and field in (audit_event.event_data or {}):
						event_value = audit_event.event_data[field]
					
					# Evaluate condition
					condition_result = self._evaluate_condition(event_value, operator, value)
					
					if condition_group.get('operator', 'AND') == 'AND':
						group_result = group_result and condition_result
					else:  # OR
						group_result = group_result or condition_result
				
				if group_result:  # If any group matches, rule is triggered
					return True
			
			return False
			
		except Exception as e:
			# Log evaluation error and return False for safety
			return False
	
	def _evaluate_condition(self, event_value: Any, operator: str, expected_value: Any) -> bool:
		"""Evaluate individual condition"""
		if operator == 'equals':
			return event_value == expected_value
		elif operator == 'not_equals':
			return event_value != expected_value
		elif operator == 'contains':
			return expected_value in str(event_value) if event_value else False
		elif operator == 'not_contains':
			return expected_value not in str(event_value) if event_value else True
		elif operator == 'greater_than':
			return float(event_value) > float(expected_value) if event_value else False
		elif operator == 'less_than':
			return float(event_value) < float(expected_value) if event_value else False
		elif operator == 'in_list':
			return event_value in expected_value if isinstance(expected_value, list) else False
		elif operator == 'not_in_list':
			return event_value not in expected_value if isinstance(expected_value, list) else True
		elif operator == 'is_null':
			return event_value is None
		elif operator == 'is_not_null':
			return event_value is not None
		
		return False
	
	def record_trigger(self, audit_event: ACAuditLog) -> None:
		"""Record that this rule was triggered"""
		self.triggered_count += 1
		self.last_triggered = datetime.utcnow()
		self.total_evaluations += 1
	
	def calculate_effectiveness(self) -> None:
		"""Calculate rule effectiveness score"""
		if self.total_evaluations > 0:
			accuracy = 1.0 - (self.false_positive_count / self.total_evaluations)
			relevance = min(1.0, self.triggered_count / max(1, self.total_evaluations / 10))
			self.effectiveness_score = (accuracy * 0.7 + relevance * 0.3) * 100


class ACComplianceViolation(Model, AuditMixin, BaseMixin):
	"""
	Compliance rule violations with remediation tracking.
	
	Records instances where compliance rules are violated,
	tracks remediation efforts, and maintains violation history.
	"""
	__tablename__ = 'ac_compliance_violation'
	
	violation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Violation Reference
	audit_log_id = Column(String(36), ForeignKey('ac_audit_log.log_id'), nullable=False, index=True)
	rule_id = Column(String(36), ForeignKey('ac_compliance_rule.rule_id'), nullable=False, index=True)
	
	# Violation Details
	severity = Column(String(20), nullable=False, index=True)  # low, medium, high, critical
	description = Column(Text, nullable=False)
	violation_type = Column(String(50), nullable=True, index=True)
	
	# Status and Resolution
	status = Column(String(20), default='open', index=True)  # open, investigating, resolved, false_positive
	assigned_to = Column(String(36), nullable=True)  # User ID of assigned investigator
	resolution_notes = Column(Text, nullable=True)
	resolved_at = Column(DateTime, nullable=True, index=True)
	resolved_by = Column(String(36), nullable=True)
	
	# Risk Assessment
	risk_score = Column(Float, default=0.0)
	business_impact = Column(String(20), nullable=True)  # low, medium, high, critical
	regulatory_impact = Column(String(20), nullable=True)  # none, low, medium, high
	
	# Notification and Escalation
	notifications_sent = Column(JSON, default=list)
	escalated_at = Column(DateTime, nullable=True)
	escalation_level = Column(Integer, default=0)
	
	# Remediation Tracking
	remediation_actions = Column(JSON, default=list)
	remediation_deadline = Column(DateTime, nullable=True)
	remediation_cost = Column(Float, nullable=True)
	
	# Relationships
	audit_log = relationship("ACAuditLog", back_populates="compliance_violations")
	rule = relationship("ACComplianceRule", back_populates="violations")
	
	def __repr__(self):
		return f"<ACComplianceViolation {self.violation_id} ({self.severity})>"
	
	def is_overdue(self) -> bool:
		"""Check if violation remediation is overdue"""
		return (self.remediation_deadline is not None and 
				self.status not in ['resolved', 'false_positive'] and
				datetime.utcnow() > self.remediation_deadline)
	
	def calculate_risk_score(self) -> None:
		"""Calculate violation risk score based on multiple factors"""
		base_score = {
			'low': 25,
			'medium': 50, 
			'high': 75,
			'critical': 100
		}.get(self.severity, 50)
		
		# Adjust for business impact
		business_multiplier = {
			'low': 0.8,
			'medium': 1.0,
			'high': 1.3,
			'critical': 1.5
		}.get(self.business_impact, 1.0)
		
		# Adjust for regulatory impact
		regulatory_multiplier = {
			'none': 0.9,
			'low': 1.0,
			'medium': 1.2,
			'high': 1.4
		}.get(self.regulatory_impact, 1.0)
		
		# Adjust for time since detection
		if self.created_on:
			days_open = (datetime.utcnow() - self.created_on).days
			time_multiplier = min(1.5, 1.0 + (days_open / 30))  # Increase over time
		else:
			time_multiplier = 1.0
		
		self.risk_score = min(100.0, base_score * business_multiplier * regulatory_multiplier * time_multiplier)
	
	def add_remediation_action(self, action_type: str, description: str, assigned_to: str = None) -> None:
		"""Add remediation action to violation"""
		action = {
			'id': uuid7str(),
			'type': action_type,
			'description': description,
			'assigned_to': assigned_to,
			'created_at': datetime.utcnow().isoformat(),
			'status': 'pending'
		}
		
		if self.remediation_actions is None:
			self.remediation_actions = []
		
		self.remediation_actions.append(action)
	
	def escalate(self, escalation_reason: str = None) -> None:
		"""Escalate violation to higher level"""
		self.escalation_level += 1
		if self.escalated_at is None:
			self.escalated_at = datetime.utcnow()
		
		# Add escalation to remediation actions
		self.add_remediation_action(
			'escalation',
			f"Escalated to level {self.escalation_level}. Reason: {escalation_reason or 'Automatic escalation'}"
		)


class ACDataRetentionPolicy(Model, AuditMixin, BaseMixin):
	"""
	Data retention policies for automated lifecycle management.
	
	Defines retention rules for different data types with automated
	deletion, archival, and compliance requirements.
	"""
	__tablename__ = 'ac_data_retention_policy'
	
	policy_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Policy Definition
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	policy_type = Column(String(50), nullable=False, index=True)  # audit_logs, user_data, financial_records
	
	# Data Classification
	data_types = Column(JSON, nullable=False)  # Types of data covered
	data_sources = Column(JSON, default=list)  # Source systems/tables
	compliance_frameworks = Column(JSON, default=list)  # Applicable compliance frameworks
	
	# Retention Rules
	retention_period_days = Column(Integer, nullable=False)
	retention_trigger = Column(String(50), default='creation_date')  # creation_date, last_access, custom
	archival_period_days = Column(Integer, nullable=True)  # Archive before deletion
	
	# Deletion Configuration
	deletion_method = Column(String(50), default='soft_delete')  # soft_delete, hard_delete, anonymize
	secure_deletion = Column(Boolean, default=True)
	deletion_verification = Column(Boolean, default=True)
	
	# Policy Status
	is_active = Column(Boolean, default=True, index=True)
	auto_execute = Column(Boolean, default=False)
	require_approval = Column(Boolean, default=True)
	
	# Legal and Compliance
	legal_hold_exempt = Column(Boolean, default=False)
	gdpr_compliant = Column(Boolean, default=True)
	regulatory_requirements = Column(JSON, default=list)
	
	# Execution Tracking
	last_executed = Column(DateTime, nullable=True, index=True)
	next_execution = Column(DateTime, nullable=True, index=True)
	execution_frequency = Column(String(20), default='monthly')  # daily, weekly, monthly
	
	# Performance Metrics
	total_records_processed = Column(Integer, default=0)
	total_records_deleted = Column(Integer, default=0)
	total_records_archived = Column(Integer, default=0)
	execution_count = Column(Integer, default=0)
	
	def __repr__(self):
		return f"<ACDataRetentionPolicy {self.name} ({self.retention_period_days} days)>"
	
	def should_execute(self) -> bool:
		"""Check if policy should be executed based on schedule"""
		if not self.is_active:
			return False
		
		if self.next_execution is None:
			return True
		
		return datetime.utcnow() >= self.next_execution
	
	def calculate_next_execution(self) -> None:
		"""Calculate next execution time based on frequency"""
		now = datetime.utcnow()
		
		if self.execution_frequency == 'daily':
			self.next_execution = now + timedelta(days=1)
		elif self.execution_frequency == 'weekly':
			self.next_execution = now + timedelta(weeks=1)
		elif self.execution_frequency == 'monthly':
			self.next_execution = now + timedelta(days=30)
		else:
			self.next_execution = now + timedelta(days=1)  # Default to daily
	
	def get_records_for_retention(self) -> List[str]:
		"""Get list of record IDs eligible for retention processing"""
		# This would be implemented with specific queries based on data_sources
		# and retention criteria. Placeholder implementation.
		return []
	
	def is_exempt_from_deletion(self, record_data: Dict[str, Any]) -> bool:
		"""Check if record is exempt from deletion due to legal hold or other reasons"""
		if self.legal_hold_exempt and record_data.get('legal_hold', False):
			return True
		
		# Check for other exemption criteria
		return False


class ACComplianceReport(Model, AuditMixin, BaseMixin):
	"""
	Generated compliance reports with metrics and findings.
	
	Stores compliance assessment reports with detailed findings,
	metrics, and recommendations for regulatory compliance.
	"""
	__tablename__ = 'ac_compliance_report'
	
	report_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Report Identity
	name = Column(String(200), nullable=False)
	report_type = Column(String(50), nullable=False, index=True)  # audit_summary, compliance_assessment, violation_report
	compliance_framework = Column(String(50), nullable=False, index=True)
	
	# Report Scope
	reporting_period_start = Column(DateTime, nullable=False, index=True)
	reporting_period_end = Column(DateTime, nullable=False, index=True)
	scope_description = Column(Text, nullable=True)
	included_systems = Column(JSON, default=list)
	
	# Report Status
	status = Column(String(20), default='generating', index=True)  # generating, completed, failed, archived
	generated_by = Column(String(36), nullable=True)
	generated_at = Column(DateTime, nullable=True, index=True)
	
	# Report Content
	executive_summary = Column(Text, nullable=True)
	findings = Column(JSON, default=list)
	recommendations = Column(JSON, default=list)
	metrics = Column(JSON, default=dict)
	
	# Compliance Scores
	overall_compliance_score = Column(Float, default=0.0)
	control_scores = Column(JSON, default=dict)  # Scores by control area
	trend_analysis = Column(JSON, default=dict)
	
	# Report Metadata
	report_format = Column(String(20), default='json')  # json, pdf, html, csv
	file_path = Column(String(500), nullable=True)
	file_size_bytes = Column(Integer, nullable=True)
	
	# Distribution
	recipients = Column(JSON, default=list)
	distribution_status = Column(JSON, default=dict)
	confidentiality_level = Column(String(20), default='internal')  # public, internal, confidential, restricted
	
	def __repr__(self):
		return f"<ACComplianceReport {self.name} ({self.compliance_framework})>"
	
	def calculate_compliance_score(self, audit_events: List[ACAuditLog] = None) -> None:
		"""Calculate overall compliance score based on violations and adherence"""
		# Placeholder implementation - would calculate based on:
		# - Number of violations vs total events
		# - Severity of violations
		# - Remediation status
		# - Control effectiveness
		
		if audit_events:
			total_events = len(audit_events)
			compliance_events = sum(1 for event in audit_events if event.is_compliance_relevant())
			violation_events = sum(1 for event in audit_events if event.compliance_violations)
			
			if compliance_events > 0:
				base_score = ((compliance_events - violation_events) / compliance_events) * 100
				self.overall_compliance_score = max(0.0, min(100.0, base_score))
			else:
				self.overall_compliance_score = 100.0
	
	def add_finding(self, finding_type: str, severity: str, description: str, 
					affected_systems: List[str] = None, remediation: str = None) -> None:
		"""Add finding to report"""
		finding = {
			'id': uuid7str(),
			'type': finding_type,
			'severity': severity,
			'description': description,
			'affected_systems': affected_systems or [],
			'remediation': remediation,
			'identified_at': datetime.utcnow().isoformat()
		}
		
		if self.findings is None:
			self.findings = []
		
		self.findings.append(finding)
	
	def add_recommendation(self, category: str, priority: str, description: str, 
						   estimated_effort: str = None, timeline: str = None) -> None:
		"""Add recommendation to report"""
		recommendation = {
			'id': uuid7str(),
			'category': category,
			'priority': priority,
			'description': description,
			'estimated_effort': estimated_effort,
			'timeline': timeline,
			'created_at': datetime.utcnow().isoformat()
		}
		
		if self.recommendations is None:
			self.recommendations = []
		
		self.recommendations.append(recommendation)
	
	def finalize_report(self) -> None:
		"""Finalize report generation"""
		self.status = 'completed'
		self.generated_at = datetime.utcnow()
		
		# Calculate final metrics
		if self.findings:
			severity_counts = {}
			for finding in self.findings:
				severity = finding.get('severity', 'unknown')
				severity_counts[severity] = severity_counts.get(severity, 0) + 1
			
			self.metrics['findings_by_severity'] = severity_counts
			self.metrics['total_findings'] = len(self.findings)
		
		if self.recommendations:
			self.metrics['total_recommendations'] = len(self.recommendations)


class ACSystemConfiguration(Model, AuditMixin, BaseMixin):
	"""
	System configuration for audit and compliance settings.
	
	Stores tenant-specific configuration for audit logging,
	compliance monitoring, and reporting preferences.
	"""
	__tablename__ = 'ac_system_configuration'
	
	config_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, unique=True, index=True)
	
	# Audit Configuration
	audit_enabled = Column(Boolean, default=True)
	audit_level = Column(String(20), default='standard')  # minimal, standard, comprehensive, verbose
	audit_retention_days = Column(Integer, default=2555)  # 7 years default
	
	# Compliance Monitoring
	compliance_monitoring_enabled = Column(Boolean, default=True)
	active_frameworks = Column(JSON, default=list)  # List of active compliance frameworks
	violation_notification_enabled = Column(Boolean, default=True)
	
	# Real-time Monitoring
	real_time_alerts = Column(Boolean, default=True)
	alert_thresholds = Column(JSON, default=dict)
	notification_channels = Column(JSON, default=list)
	
	# Data Retention
	auto_retention_enabled = Column(Boolean, default=False)
	retention_approval_required = Column(Boolean, default=True)
	secure_deletion_enabled = Column(Boolean, default=True)
	
	# Reporting Configuration
	auto_reporting_enabled = Column(Boolean, default=False)
	report_frequency = Column(String(20), default='monthly')
	report_recipients = Column(JSON, default=list)
	report_formats = Column(JSON, default=['pdf', 'json'])
	
	# Integration Settings
	blockchain_anchoring = Column(Boolean, default=False)
	external_siem_integration = Column(Boolean, default=False)
	siem_endpoint = Column(String(500), nullable=True)
	
	# Performance Settings
	batch_processing_enabled = Column(Boolean, default=True)
	max_batch_size = Column(Integer, default=1000)
	processing_threads = Column(Integer, default=4)
	
	def __repr__(self):
		return f"<ACSystemConfiguration for tenant {self.tenant_id}>"
	
	def get_audit_level_config(self) -> Dict[str, Any]:
		"""Get configuration details for current audit level"""
		audit_configs = {
			'minimal': {
				'log_data_changes': False,
				'log_access_events': True,
				'log_system_events': False,
				'include_request_data': False,
				'include_response_data': False
			},
			'standard': {
				'log_data_changes': True,
				'log_access_events': True,
				'log_system_events': True,
				'include_request_data': True,
				'include_response_data': False
			},
			'comprehensive': {
				'log_data_changes': True,
				'log_access_events': True,
				'log_system_events': True,
				'include_request_data': True,
				'include_response_data': True
			},
			'verbose': {
				'log_data_changes': True,
				'log_access_events': True,
				'log_system_events': True,
				'include_request_data': True,
				'include_response_data': True,
				'log_debug_events': True,
				'include_stack_traces': True
			}
		}
		
		return audit_configs.get(self.audit_level, audit_configs['standard'])
	
	def should_log_event(self, event_type: str, event_category: str) -> bool:
		"""Determine if event should be logged based on configuration"""
		if not self.audit_enabled:
			return False
		
		config = self.get_audit_level_config()
		
		if event_category == 'data' and not config.get('log_data_changes', False):
			return False
		
		if event_category == 'system' and not config.get('log_system_events', False):
			return False
		
		return True