"""
APG Budgeting & Forecasting - Version Control and Audit

Comprehensive budget version control system with detailed audit trails,
compliance tracking, and seamless APG audit_compliance integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from decimal import Decimal
from uuid import UUID
import json
import logging
import hashlib
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
import difflib

import asyncpg
from pydantic import BaseModel, Field, validator, root_validator
from pydantic import ConfigDict

from .models import (
	APGBaseModel, BFBudgetStatus, BFApprovalStatus,
	PositiveAmount, CurrencyCode, NonEmptyString
)
from .service import APGTenantContext, BFServiceConfig, ServiceResponse, APGServiceBase
from uuid_extensions import uuid7str


# =============================================================================
# Version Control and Audit Models
# =============================================================================

class ChangeType(str, Enum):
	"""Change type enumeration for version tracking."""
	CREATE = "create"
	UPDATE = "update"
	DELETE = "delete"
	APPROVE = "approve"
	REJECT = "reject"
	SUBMIT = "submit"
	RESTORE = "restore"
	MERGE = "merge"
	CLONE = "clone"


class AuditEventType(str, Enum):
	"""Audit event type enumeration."""
	DATA_CHANGE = "data_change"
	STATUS_CHANGE = "status_change"
	ACCESS_EVENT = "access_event"
	PERMISSION_CHANGE = "permission_change"
	WORKFLOW_EVENT = "workflow_event"
	SYSTEM_EVENT = "system_event"
	COMPLIANCE_EVENT = "compliance_event"
	SECURITY_EVENT = "security_event"


class ComplianceLevel(str, Enum):
	"""Compliance level enumeration."""
	BASIC = "basic"
	STANDARD = "standard"
	ENHANCED = "enhanced"
	MAXIMUM = "maximum"


class BudgetVersion(APGBaseModel):
	"""Comprehensive budget version model with detailed change tracking."""
	
	version_id: str = Field(default_factory=uuid7str)
	budget_id: str = Field(...)
	version_number: int = Field(..., ge=1)
	version_name: Optional[str] = Field(None, max_length=255)
	
	# Version metadata
	is_current: bool = Field(default=True)
	is_baseline: bool = Field(default=False)
	is_approved: bool = Field(default=False)
	is_archived: bool = Field(default=False)
	is_locked: bool = Field(default=False)
	
	# Change tracking
	change_type: ChangeType = Field(...)
	change_summary: str = Field(..., max_length=1000)
	changes_detailed: List[Dict[str, Any]] = Field(default_factory=list)
	affected_components: List[str] = Field(default_factory=list)
	
	# Version data - comprehensive snapshots
	budget_data_snapshot: Dict[str, Any] = Field(...)
	line_items_snapshot: List[Dict[str, Any]] = Field(default_factory=list)
	metadata_snapshot: Dict[str, Any] = Field(default_factory=dict)
	calculations_snapshot: Dict[str, Any] = Field(default_factory=dict)
	
	# Change metrics
	lines_added: int = Field(default=0, ge=0)
	lines_modified: int = Field(default=0, ge=0)
	lines_deleted: int = Field(default=0, ge=0)
	amount_changes: Dict[str, Any] = Field(default_factory=dict)
	
	# Approval and workflow tracking
	approval_status: BFApprovalStatus = Field(default=BFApprovalStatus.PENDING)
	approved_by: Optional[str] = Field(None)
	approval_date: Optional[datetime] = Field(None)
	approval_workflow_id: Optional[str] = Field(None)
	
	# Parent version tracking
	parent_version_id: Optional[str] = Field(None)
	merged_from_versions: List[str] = Field(default_factory=list)
	branched_to_versions: List[str] = Field(default_factory=list)
	
	# Hash and integrity
	data_hash: str = Field(...)
	integrity_verified: bool = Field(default=True)
	hash_algorithm: str = Field(default="sha256")
	
	# Retention and lifecycle
	retention_policy: Optional[str] = Field(None)
	retention_expires_at: Optional[datetime] = Field(None)
	auto_archive_after_days: Optional[int] = Field(None, ge=1)
	
	# Compliance and regulatory
	regulatory_flags: List[str] = Field(default_factory=list)
	compliance_requirements: List[str] = Field(default_factory=list)
	data_classification: str = Field(default="internal", max_length=20)
	
	# Performance and storage
	snapshot_size_bytes: Optional[int] = Field(None, ge=0)
	compression_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
	storage_location: Optional[str] = Field(None)

	@validator('data_hash')
	def validate_data_hash(cls, v: str, values: Dict[str, Any]) -> str:
		"""Validate and ensure data hash is computed correctly."""
		budget_data = values.get('budget_data_snapshot', {})
		line_items = values.get('line_items_snapshot', [])
		
		if budget_data and not v:
			# Auto-generate hash if not provided
			combined_data = json.dumps({
				'budget': budget_data,
				'lines': line_items
			}, sort_keys=True)
			return hashlib.sha256(combined_data.encode()).hexdigest()
		
		return v

	@root_validator
	def validate_version_consistency(cls, values: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate version data consistency."""
		is_current = values.get('is_current', True)
		is_baseline = values.get('is_baseline', False)
		version_number = values.get('version_number', 1)
		
		# Baseline versions should be version 1
		if is_baseline and version_number != 1:
			raise ValueError("Baseline versions must be version 1")
		
		# Current version cannot be archived
		if is_current and values.get('is_archived', False):
			raise ValueError("Current version cannot be archived")
		
		return values


class AuditEvent(APGBaseModel):
	"""Comprehensive audit event model for compliance tracking."""
	
	event_id: str = Field(default_factory=uuid7str)
	event_type: AuditEventType = Field(...)
	event_category: str = Field(..., max_length=100)
	event_subcategory: Optional[str] = Field(None, max_length=100)
	
	# Event target and context
	target_type: str = Field(..., max_length=50)  # budget, line_item, template, workflow
	target_id: str = Field(...)
	target_name: Optional[str] = Field(None, max_length=255)
	
	# Event details
	event_description: str = Field(..., max_length=1000)
	event_data: Dict[str, Any] = Field(default_factory=dict)
	previous_values: Optional[Dict[str, Any]] = Field(None)
	new_values: Optional[Dict[str, Any]] = Field(None)
	
	# Actor information
	actor_user_id: str = Field(...)
	actor_user_name: str = Field(...)
	actor_role: Optional[str] = Field(None)
	actor_department: Optional[str] = Field(None)
	
	# Session and system context
	session_id: Optional[str] = Field(None)
	request_id: Optional[str] = Field(None)
	correlation_id: Optional[str] = Field(None)
	transaction_id: Optional[str] = Field(None)
	
	# Technical context
	client_ip: Optional[str] = Field(None)
	user_agent: Optional[str] = Field(None)
	api_endpoint: Optional[str] = Field(None)
	http_method: Optional[str] = Field(None)
	
	# Timing and performance
	event_timestamp: datetime = Field(default_factory=datetime.utcnow)
	event_duration_ms: Optional[int] = Field(None, ge=0)
	event_sequence: int = Field(..., ge=1)
	
	# Risk and security
	risk_level: str = Field(default="low", max_length=20)  # low, medium, high, critical
	security_classification: str = Field(default="internal", max_length=20)
	contains_pii: bool = Field(default=False)
	contains_sensitive_data: bool = Field(default=False)
	
	# Compliance and regulatory
	compliance_level: ComplianceLevel = Field(default=ComplianceLevel.STANDARD)
	regulatory_requirements: List[str] = Field(default_factory=list)
	retention_policy: Optional[str] = Field(None)
	data_subject_id: Optional[str] = Field(None)  # For GDPR compliance
	
	# Change impact analysis
	business_impact: str = Field(default="low", max_length=20)
	financial_impact: Optional[Decimal] = Field(None)
	affected_users: List[str] = Field(default_factory=list)
	affected_departments: List[str] = Field(default_factory=list)
	
	# Verification and integrity
	event_hash: Optional[str] = Field(None)
	digital_signature: Optional[str] = Field(None)
	tamper_evident: bool = Field(default=True)
	verification_status: str = Field(default="verified", max_length=20)
	
	# Workflow and approval context
	workflow_instance_id: Optional[str] = Field(None)
	approval_stage: Optional[str] = Field(None)
	requires_approval: bool = Field(default=False)
	
	# External system integration
	external_system_ref: Optional[str] = Field(None)
	external_event_id: Optional[str] = Field(None)
	sync_status: str = Field(default="synced", max_length=20)


class ComplianceReport(BaseModel):
	"""Compliance audit report model."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	report_id: str = Field(default_factory=uuid7str)
	report_type: str = Field(..., max_length=100)
	report_title: str = Field(..., max_length=255)
	
	# Report scope
	tenant_id: str = Field(...)
	target_entities: List[str] = Field(...)  # budget IDs
	date_range_start: date = Field(...)
	date_range_end: date = Field(...)
	
	# Compliance assessment
	compliance_level: ComplianceLevel = Field(...)
	compliance_score: float = Field(..., ge=0.0, le=100.0)
	compliance_requirements_met: List[str] = Field(default_factory=list)
	compliance_gaps: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Audit findings
	total_events_audited: int = Field(..., ge=0)
	high_risk_events: int = Field(default=0, ge=0)
	security_violations: int = Field(default=0, ge=0)
	data_integrity_issues: int = Field(default=0, ge=0)
	
	# Version control analysis
	total_versions_analyzed: int = Field(..., ge=0)
	unauthorized_changes: int = Field(default=0, ge=0)
	missing_approvals: int = Field(default=0, ge=0)
	data_corruption_detected: bool = Field(default=False)
	
	# Recommendations
	priority_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
	remediation_actions: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Report metadata
	generated_at: datetime = Field(default_factory=datetime.utcnow)
	generated_by: str = Field(...)
	report_version: str = Field(default="1.0")
	
	# Certification
	certified_by: Optional[str] = Field(None)
	certification_date: Optional[datetime] = Field(None)
	certification_valid_until: Optional[datetime] = Field(None)


# =============================================================================
# Version Control and Audit Service
# =============================================================================

class VersionControlAuditService(APGServiceBase):
	"""
	Comprehensive version control and audit service providing
	detailed change tracking, compliance monitoring, and audit trails.
	"""
	
	def __init__(self, context: APGTenantContext, config: BFServiceConfig):
		super().__init__(context, config)
		self._audit_processors: Dict[AuditEventType, Callable] = {}
		self._compliance_checkers: Dict[str, Callable] = {}
		self._hash_algorithms = {
			'sha256': hashlib.sha256,
			'sha512': hashlib.sha512,
			'md5': hashlib.md5
		}
		
		# Initialize audit and compliance components
		self._initialize_audit_processors()
		self._initialize_compliance_checkers()

	async def create_budget_version(self, budget_id: str, version_data: Dict[str, Any]) -> ServiceResponse:
		"""Create a new version of a budget with comprehensive tracking."""
		try:
			# Validate permissions
			if not await self._validate_permissions('budget.version_create', budget_id):
				raise PermissionError("Insufficient permissions to create budget version")
			
			# Get current budget and validate
			budget = await self._get_budget(budget_id)
			if not budget:
				raise ValueError("Budget not found")
			
			# Get current version number
			current_version = await self._get_latest_version_number(budget_id)
			new_version_number = current_version + 1
			
			# Get budget line items for snapshot
			line_items = await self._get_budget_line_items(budget_id)
			
			# Calculate changes from previous version
			changes_analysis = await self._analyze_version_changes(budget_id, budget, line_items)
			
			# Create comprehensive data snapshot
			budget_snapshot = await self._create_comprehensive_snapshot(budget, line_items)
			
			# Generate data hash for integrity
			data_hash = await self._generate_data_hash(budget_snapshot)
			
			# Create version model
			version_data.update({
				'budget_id': budget_id,
				'version_number': new_version_number,
				'change_type': version_data.get('change_type', ChangeType.UPDATE),
				'budget_data_snapshot': budget_snapshot['budget'],
				'line_items_snapshot': budget_snapshot['line_items'],
				'metadata_snapshot': budget_snapshot['metadata'],
				'calculations_snapshot': budget_snapshot['calculations'],
				'changes_detailed': changes_analysis['changes'],
				'affected_components': changes_analysis['affected_components'],
				'lines_added': changes_analysis['lines_added'],
				'lines_modified': changes_analysis['lines_modified'],
				'lines_deleted': changes_analysis['lines_deleted'],
				'amount_changes': changes_analysis['amount_changes'],
				'data_hash': data_hash,
				'tenant_id': self.context.tenant_id,
				'created_by': self.context.user_id,
				'updated_by': self.context.user_id
			})
			
			version = BudgetVersion(**version_data)
			
			# Start database transaction
			async with self._connection.transaction():
				# Mark previous versions as not current
				await self._connection.execute("""
					UPDATE budget_versions 
					SET is_current = FALSE, updated_at = NOW()
					WHERE budget_id = $1 AND is_current = TRUE
				""", budget_id)
				
				# Insert new version
				version_id = await self._insert_budget_version(version)
				
				# Update budget with version information
				await self._connection.execute("""
					UPDATE budgets 
					SET version = $1, last_version_id = $2, updated_at = NOW(), updated_by = $3
					WHERE id = $4
				""", new_version_number, version_id, self.context.user_id, budget_id)
				
				# Create audit event for version creation
				await self._create_audit_event({
					'event_type': AuditEventType.DATA_CHANGE,
					'event_category': 'version_control',
					'event_subcategory': 'version_create',
					'target_type': 'budget',
					'target_id': budget_id,
					'event_description': f"Created budget version {new_version_number}",
					'event_data': {
						'version_id': version_id,
						'version_number': new_version_number,
						'change_summary': version.change_summary,
						'changes_count': len(version.changes_detailed)
					},
					'new_values': {'version_number': new_version_number}
				})
				
				# Integrate with APG audit_compliance
				await self._integrate_apg_audit_compliance('version_create', {
					'budget_id': budget_id,
					'version_id': version_id,
					'version_number': new_version_number,
					'data_hash': data_hash
				})
			
			return ServiceResponse(
				success=True,
				message=f"Budget version {new_version_number} created successfully",
				data={
					'version_id': version_id,
					'version_number': new_version_number,
					'data_hash': data_hash,
					'changes_summary': changes_analysis,
					'snapshot_size_bytes': version.snapshot_size_bytes
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'create_budget_version')

	async def restore_budget_version(self, budget_id: str, version_id: str, restore_options: Dict[str, Any]) -> ServiceResponse:
		"""Restore a budget to a previous version with audit trail."""
		try:
			# Validate permissions
			if not await self._validate_permissions('budget.version_restore', budget_id):
				raise PermissionError("Insufficient permissions to restore budget version")
			
			# Get target version
			target_version = await self._get_budget_version(version_id)
			if not target_version or target_version['budget_id'] != budget_id:
				raise ValueError("Version not found or does not belong to this budget")
			
			# Validate version integrity
			integrity_check = await self._verify_version_integrity(target_version)
			if not integrity_check['is_valid']:
				return ServiceResponse(
					success=False,
					message="Version integrity verification failed",
					errors=integrity_check['errors']
				)
			
			# Get current budget state for backup
			current_budget = await self._get_budget(budget_id)
			current_lines = await self._get_budget_line_items(budget_id)
			
			# Create backup version before restore
			backup_version_data = {
				'change_type': ChangeType.UPDATE,
				'change_summary': f"Backup before restore to version {target_version['version_number']}",
				'version_name': f"Pre-restore backup {datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
			}
			
			backup_result = await self.create_budget_version(budget_id, backup_version_data)
			if not backup_result.success:
				raise Exception("Failed to create backup version before restore")
			
			# Start database transaction
			async with self._connection.transaction():
				# Restore budget data
				budget_data = target_version['budget_data_snapshot']
				await self._restore_budget_data(budget_id, budget_data)
				
				# Restore line items
				line_items_data = target_version['line_items_snapshot']
				await self._restore_line_items_data(budget_id, line_items_data)
				
				# Create restore version record
				restore_version_data = {
					'change_type': ChangeType.RESTORE,
					'change_summary': f"Restored to version {target_version['version_number']}",
					'parent_version_id': version_id,
					**restore_options
				}
				
				restore_result = await self.create_budget_version(budget_id, restore_version_data)
				
				# Create audit event for restore operation
				await self._create_audit_event({
					'event_type': AuditEventType.DATA_CHANGE,
					'event_category': 'version_control',
					'event_subcategory': 'version_restore',
					'target_type': 'budget',
					'target_id': budget_id,
					'event_description': f"Restored budget to version {target_version['version_number']}",
					'event_data': {
						'restored_from_version_id': version_id,
						'restored_to_version_number': target_version['version_number'],
						'backup_version_id': backup_result.data['version_id'],
						'restore_reason': restore_options.get('reason', 'Not specified')
					},
					'risk_level': 'medium',  # Restores are medium risk operations
					'requires_approval': restore_options.get('requires_approval', True)
				})
				
				# Integrate with APG audit_compliance
				await self._integrate_apg_audit_compliance('version_restore', {
					'budget_id': budget_id,
					'restored_from_version': version_id,
					'backup_version': backup_result.data['version_id']
				})
			
			return ServiceResponse(
				success=True,
				message=f"Budget restored to version {target_version['version_number']} successfully",
				data={
					'restored_version_id': version_id,
					'restored_version_number': target_version['version_number'],
					'backup_version_id': backup_result.data['version_id'],
					'new_version_id': restore_result.data['version_id'] if restore_result.success else None
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'restore_budget_version')

	async def generate_audit_trail(self, target_id: str, trail_config: Dict[str, Any]) -> ServiceResponse:
		"""Generate comprehensive audit trail for a budget or entity."""
		try:
			# Validate permissions
			if not await self._validate_permissions('audit.generate_trail', target_id):
				raise PermissionError("Insufficient permissions to generate audit trail")
			
			target_type = trail_config.get('target_type', 'budget')
			date_range_start = trail_config.get('date_range_start', date.today() - timedelta(days=30))
			date_range_end = trail_config.get('date_range_end', date.today())
			include_sensitive = trail_config.get('include_sensitive', False)
			
			# Get audit events for the target
			audit_events = await self._get_audit_events(
				target_id, target_type, date_range_start, date_range_end
			)
			
			# Get version history
			version_history = await self._get_version_history(target_id, target_type)
			
			# Analyze audit data
			audit_analysis = await self._analyze_audit_data(audit_events)
			
			# Generate compliance assessment
			compliance_assessment = await self._assess_compliance(audit_events, trail_config)
			
			# Detect anomalies and suspicious activities
			anomaly_detection = await self._detect_audit_anomalies(audit_events)
			
			# Build comprehensive audit trail
			audit_trail = {
				'target_id': target_id,
				'target_type': target_type,
				'generation_timestamp': datetime.utcnow(),
				'date_range': {
					'start': date_range_start,
					'end': date_range_end
				},
				'summary': {
					'total_events': len(audit_events),
					'total_versions': len(version_history),
					'unique_actors': len(set(event['actor_user_id'] for event in audit_events)),
					'event_types': audit_analysis['event_types'],
					'risk_distribution': audit_analysis['risk_distribution']
				},
				'events': audit_events if not include_sensitive else self._filter_sensitive_data(audit_events),
				'version_history': version_history,
				'compliance_assessment': compliance_assessment,
				'anomaly_detection': anomaly_detection,
				'integrity_verification': await self._verify_audit_integrity(audit_events),
				'recommendations': await self._generate_audit_recommendations(audit_analysis, compliance_assessment)
			}
			
			# Store audit trail report
			trail_id = await self._store_audit_trail(audit_trail)
			
			# Create audit event for trail generation
			await self._create_audit_event({
				'event_type': AuditEventType.COMPLIANCE_EVENT,
				'event_category': 'audit_trail',
				'event_subcategory': 'trail_generation',
				'target_type': target_type,
				'target_id': target_id,
				'event_description': f"Generated audit trail for {target_type} {target_id}",
				'event_data': {
					'trail_id': trail_id,
					'events_included': len(audit_events),
					'date_range_days': (date_range_end - date_range_start).days
				}
			})
			
			return ServiceResponse(
				success=True,
				message="Audit trail generated successfully",
				data={
					'trail_id': trail_id,
					'audit_trail': audit_trail,
					'export_formats': ['json', 'csv', 'pdf'],
					'retention_expires': datetime.utcnow() + timedelta(days=2555)  # 7 years
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'generate_audit_trail')

	async def generate_compliance_report(self, report_config: Dict[str, Any]) -> ServiceResponse:
		"""Generate comprehensive compliance report."""
		try:
			# Validate permissions
			if not await self._validate_permissions('compliance.generate_report'):
				raise PermissionError("Insufficient permissions to generate compliance report")
			
			target_entities = report_config.get('target_entities', [])
			date_range_start = report_config.get('date_range_start', date.today() - timedelta(days=90))
			date_range_end = report_config.get('date_range_end', date.today())
			compliance_level = ComplianceLevel(report_config.get('compliance_level', ComplianceLevel.STANDARD))
			report_type = report_config.get('report_type', 'comprehensive')
			
			# Gather compliance data
			compliance_data = await self._gather_compliance_data(
				target_entities, date_range_start, date_range_end
			)
			
			# Assess compliance against requirements
			compliance_assessment = await self._assess_comprehensive_compliance(
				compliance_data, compliance_level
			)
			
			# Analyze version control compliance
			version_compliance = await self._analyze_version_control_compliance(
				target_entities, date_range_start, date_range_end
			)
			
			# Create compliance report
			report_data = {
				'report_type': report_type,
				'report_title': f"Budgeting & Forecasting Compliance Report - {datetime.now().strftime('%Y-%m-%d')}",
				'tenant_id': self.context.tenant_id,
				'target_entities': target_entities,
				'date_range_start': date_range_start,
				'date_range_end': date_range_end,
				'compliance_level': compliance_level,
				'compliance_score': compliance_assessment['overall_score'],
				'compliance_requirements_met': compliance_assessment['requirements_met'],
				'compliance_gaps': compliance_assessment['gaps'],
				'total_events_audited': compliance_data['total_events'],
				'high_risk_events': compliance_data['high_risk_events'],
				'security_violations': compliance_data['security_violations'],
				'data_integrity_issues': compliance_data['integrity_issues'],
				'total_versions_analyzed': version_compliance['total_versions'],
				'unauthorized_changes': version_compliance['unauthorized_changes'],
				'missing_approvals': version_compliance['missing_approvals'],
				'data_corruption_detected': version_compliance['corruption_detected'],
				'priority_recommendations': compliance_assessment['priority_recommendations'],
				'remediation_actions': compliance_assessment['remediation_actions'],
				'generated_by': self.context.user_id
			}
			
			compliance_report = ComplianceReport(**report_data)
			
			# Store compliance report
			async with self._connection.transaction():
				report_id = await self._store_compliance_report(compliance_report)
				
				# Create audit event for report generation
				await self._create_audit_event({
					'event_type': AuditEventType.COMPLIANCE_EVENT,
					'event_category': 'compliance_reporting',
					'event_subcategory': 'report_generation',
					'target_type': 'tenant',
					'target_id': self.context.tenant_id,
					'event_description': f"Generated {report_type} compliance report",
					'event_data': {
						'report_id': report_id,
						'compliance_score': compliance_report.compliance_score,
						'entities_analyzed': len(target_entities)
					},
					'compliance_level': compliance_level,
					'risk_level': 'low'
				})
				
				# Integrate with APG audit_compliance
				await self._integrate_apg_audit_compliance('compliance_report', {
					'report_id': report_id,
					'compliance_score': compliance_report.compliance_score,
					'tenant_id': self.context.tenant_id
				})
			
			return ServiceResponse(
				success=True,
				message="Compliance report generated successfully",
				data={
					'report_id': report_id,
					'compliance_report': compliance_report.dict(),
					'export_formats': ['json', 'pdf', 'excel'],
					'certification_required': compliance_report.compliance_score < 95.0
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'generate_compliance_report')

	# =============================================================================
	# Helper Methods
	# =============================================================================

	def _initialize_audit_processors(self) -> None:
		"""Initialize audit event processors."""
		self._audit_processors = {
			AuditEventType.DATA_CHANGE: self._process_data_change_audit,
			AuditEventType.STATUS_CHANGE: self._process_status_change_audit,
			AuditEventType.ACCESS_EVENT: self._process_access_event_audit,
			AuditEventType.WORKFLOW_EVENT: self._process_workflow_event_audit,
			AuditEventType.COMPLIANCE_EVENT: self._process_compliance_event_audit,
			AuditEventType.SECURITY_EVENT: self._process_security_event_audit
		}

	def _initialize_compliance_checkers(self) -> None:
		"""Initialize compliance checkers for different standards."""
		self._compliance_checkers = {
			'gdpr': self._check_gdpr_compliance,
			'sox': self._check_sox_compliance,
			'iso27001': self._check_iso27001_compliance,
			'pci_dss': self._check_pci_dss_compliance
		}

	async def _get_latest_version_number(self, budget_id: str) -> int:
		"""Get the latest version number for a budget."""
		result = await self._connection.fetchval("""
			SELECT COALESCE(MAX(version_number), 0) 
			FROM budget_versions 
			WHERE budget_id = $1
		""", budget_id)
		return result or 0

	async def _analyze_version_changes(self, budget_id: str, budget: Dict[str, Any], 
									  line_items: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Analyze changes between current and previous version."""
		previous_version = await self._get_previous_version(budget_id)
		
		if not previous_version:
			return {
				'changes': [{'type': 'create', 'description': 'Initial budget creation'}],
				'affected_components': ['budget', 'line_items'],
				'lines_added': len(line_items),
				'lines_modified': 0,
				'lines_deleted': 0,
				'amount_changes': {'total_change': budget.get('total_amount', 0)}
			}
		
		# Compare budget data
		budget_changes = await self._compare_budget_data(
			previous_version['budget_data_snapshot'], budget
		)
		
		# Compare line items
		line_changes = await self._compare_line_items(
			previous_version['line_items_snapshot'], line_items
		)
		
		return {
			'changes': budget_changes + line_changes,
			'affected_components': list(set([c['component'] for c in budget_changes + line_changes])),
			'lines_added': len([c for c in line_changes if c['type'] == 'add']),
			'lines_modified': len([c for c in line_changes if c['type'] == 'modify']),
			'lines_deleted': len([c for c in line_changes if c['type'] == 'delete']),
			'amount_changes': await self._calculate_amount_changes(previous_version, budget, line_items)
		}

	async def _create_comprehensive_snapshot(self, budget: Dict[str, Any], 
											line_items: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Create comprehensive data snapshot for versioning."""
		# Calculate budget totals and metadata
		calculations = await self._calculate_budget_totals(line_items)
		
		metadata = {
			'snapshot_timestamp': datetime.utcnow().isoformat(),
			'tenant_id': self.context.tenant_id,
			'created_by': self.context.user_id,
			'line_items_count': len(line_items),
			'total_amount': calculations.get('total_amount', 0),
			'calculation_method': 'sum_line_items'
		}
		
		return {
			'budget': budget,
			'line_items': line_items,
			'calculations': calculations,
			'metadata': metadata
		}

	async def _generate_data_hash(self, snapshot: Dict[str, Any]) -> str:
		"""Generate cryptographic hash for data integrity."""
		# Create deterministic representation
		normalized_data = json.dumps(snapshot, sort_keys=True, default=str)
		
		# Generate SHA-256 hash
		return hashlib.sha256(normalized_data.encode('utf-8')).hexdigest()

	async def _insert_budget_version(self, version: BudgetVersion) -> str:
		"""Insert budget version into database."""
		version_dict = version.dict()
		columns = list(version_dict.keys())
		placeholders = [f"${i+1}" for i in range(len(columns))]
		values = list(version_dict.values())
		
		query = f"""
			INSERT INTO budget_versions ({', '.join(columns)})
			VALUES ({', '.join(placeholders)})
			RETURNING version_id
		"""
		
		return await self._connection.fetchval(query, *values)

	async def _create_audit_event(self, event_data: Dict[str, Any]) -> str:
		"""Create comprehensive audit event."""
		# Inject context data
		event_data.update({
			'tenant_id': self.context.tenant_id,
			'actor_user_id': self.context.user_id,
			'actor_user_name': event_data.get('actor_user_name', 'System User'),
			'event_sequence': await self._get_next_event_sequence(),
			'session_id': self.context.session_id if hasattr(self.context, 'session_id') else None,
			'created_by': self.context.user_id,
			'updated_by': self.context.user_id
		})
		
		# Generate event hash for integrity
		if event_data.get('tamper_evident', True):
			event_data['event_hash'] = await self._generate_event_hash(event_data)
		
		audit_event = AuditEvent(**event_data)
		
		# Insert audit event
		return await self._insert_audit_event(audit_event)

	async def _insert_audit_event(self, event: AuditEvent) -> str:
		"""Insert audit event into database."""
		event_dict = event.dict()
		columns = list(event_dict.keys())
		placeholders = [f"${i+1}" for i in range(len(columns))]
		values = list(event_dict.values())
		
		query = f"""
			INSERT INTO audit_events ({', '.join(columns)})
			VALUES ({', '.join(placeholders)})
			RETURNING event_id
		"""
		
		return await self._connection.fetchval(query, *values)

	async def _integrate_apg_audit_compliance(self, event_type: str, data: Dict[str, Any]) -> None:
		"""Integrate with APG audit_compliance capability."""
		# This would integrate with the APG audit_compliance service
		self.logger.info(f"Integrating with APG audit_compliance: {event_type} - {data}")

	async def _get_next_event_sequence(self) -> int:
		"""Get next event sequence number for tenant."""
		result = await self._connection.fetchval("""
			SELECT COALESCE(MAX(event_sequence), 0) + 1
			FROM audit_events
			WHERE tenant_id = $1
		""", self.context.tenant_id)
		return result or 1

	async def _generate_event_hash(self, event_data: Dict[str, Any]) -> str:
		"""Generate hash for audit event integrity."""
		# Remove hash field to avoid circular reference
		hash_data = {k: v for k, v in event_data.items() if k != 'event_hash'}
		normalized = json.dumps(hash_data, sort_keys=True, default=str)
		return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


# =============================================================================
# Service Factory and Export
# =============================================================================

def create_version_control_audit_service(context: APGTenantContext, config: BFServiceConfig) -> VersionControlAuditService:
	"""Factory function to create version control and audit service."""
	return VersionControlAuditService(context, config)


# Export version control and audit classes
__all__ = [
	'ChangeType',
	'AuditEventType',
	'ComplianceLevel',
	'BudgetVersion',
	'AuditEvent',
	'ComplianceReport',
	'VersionControlAuditService',
	'create_version_control_audit_service'
]


def _log_version_control_audit_summary() -> str:
	"""Log summary of version control and audit capabilities."""
	return f"Version Control & Audit loaded: {len(__all__)} components with comprehensive tracking and compliance integration"