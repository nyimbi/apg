"""
APG Incident Response Management - Pydantic Models

Enterprise incident response models with automated workflows,
forensic investigation tracking, and comprehensive case management.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic import ConfigDict
from uuid_extensions import uuid7str


class IncidentSeverity(str, Enum):
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"


class IncidentStatus(str, Enum):
	NEW = "new"
	ACKNOWLEDGED = "acknowledged"
	INVESTIGATING = "investigating"
	CONTAINING = "containing"
	ERADICATING = "eradicating"
	RECOVERING = "recovering"
	RESOLVED = "resolved"
	CLOSED = "closed"


class IncidentCategory(str, Enum):
	MALWARE = "malware"
	PHISHING = "phishing"
	DATA_BREACH = "data_breach"
	UNAUTHORIZED_ACCESS = "unauthorized_access"
	DENIAL_OF_SERVICE = "denial_of_service"
	INSIDER_THREAT = "insider_threat"
	RANSOMWARE = "ransomware"
	SUPPLY_CHAIN = "supply_chain"
	COMPLIANCE_VIOLATION = "compliance_violation"
	SYSTEM_COMPROMISE = "system_compromise"


class EvidenceType(str, Enum):
	DIGITAL_FORENSICS = "digital_forensics"
	NETWORK_CAPTURE = "network_capture"
	LOG_FILES = "log_files"
	MEMORY_DUMP = "memory_dump"
	DISK_IMAGE = "disk_image"
	EMAIL_HEADERS = "email_headers"
	MALWARE_SAMPLE = "malware_sample"
	SCREENSHOTS = "screenshots"
	DOCUMENTATION = "documentation"


class ActionType(str, Enum):
	CONTAINMENT = "containment"
	ERADICATION = "eradication"
	RECOVERY = "recovery"
	INVESTIGATION = "investigation"
	COMMUNICATION = "communication"
	DOCUMENTATION = "documentation"
	FORENSICS = "forensics"


class SecurityIncident(BaseModel):
	"""Comprehensive security incident definition and tracking"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	# Incident identification
	incident_number: str = Field(description="Unique incident number")
	title: str = Field(description="Incident title")
	description: str = Field(description="Detailed incident description")
	
	# Classification
	category: IncidentCategory
	subcategory: Optional[str] = None
	severity: IncidentSeverity
	priority: IncidentSeverity = IncidentSeverity.MEDIUM
	
	# Status and lifecycle
	status: IncidentStatus = IncidentStatus.NEW
	incident_declared: datetime = Field(default_factory=datetime.utcnow)
	last_updated: datetime = Field(default_factory=datetime.utcnow)
	
	# Discovery and reporting
	discovered_by: str = Field(description="Who discovered the incident")
	discovery_method: str = Field(description="How the incident was discovered")
	reported_by: str = Field(description="Who reported the incident")
	reported_at: datetime = Field(default_factory=datetime.utcnow)
	
	# Timeline tracking
	detection_time: Optional[datetime] = None
	acknowledgment_time: Optional[datetime] = None
	containment_time: Optional[datetime] = None
	eradication_time: Optional[datetime] = None
	recovery_time: Optional[datetime] = None
	resolution_time: Optional[datetime] = None
	
	# Response team assignments
	incident_commander: Optional[str] = None
	lead_investigator: Optional[str] = None
	response_team: List[str] = Field(default_factory=list)
	external_parties: List[str] = Field(default_factory=list)
	
	# Affected systems and scope
	affected_systems: List[str] = Field(default_factory=list)
	affected_users: List[str] = Field(default_factory=list)
	affected_services: List[str] = Field(default_factory=list)
	business_impact: str = Field(description="Business impact assessment")
	
	# Threat intelligence context
	threat_actors: List[str] = Field(default_factory=list)
	attack_vectors: List[str] = Field(default_factory=list)
	attack_techniques: List[str] = Field(default_factory=list, description="MITRE ATT&CK techniques")
	indicators_of_compromise: List[str] = Field(default_factory=list)
	
	# Damage assessment
	data_compromised: bool = False
	data_exfiltrated: bool = False
	systems_compromised: int = 0
	estimated_financial_impact: Optional[Decimal] = None
	
	# Containment and mitigation
	containment_actions: List[str] = Field(default_factory=list)
	mitigation_actions: List[str] = Field(default_factory=list)
	recovery_actions: List[str] = Field(default_factory=list)
	
	# Communication and notifications
	stakeholders_notified: List[str] = Field(default_factory=list)
	regulatory_notifications: List[str] = Field(default_factory=list)
	public_disclosure: bool = False
	media_involvement: bool = False
	
	# Legal and compliance
	legal_review_required: bool = False
	regulatory_implications: List[str] = Field(default_factory=list)
	law_enforcement_involved: bool = False
	legal_hold_applied: bool = False
	
	# Quality metrics
	response_effectiveness: Optional[Decimal] = Field(None, ge=0, le=100)
	lessons_learned: List[str] = Field(default_factory=list)
	improvement_recommendations: List[str] = Field(default_factory=list)
	
	# External references
	related_incidents: List[str] = Field(default_factory=list)
	reference_tickets: List[str] = Field(default_factory=list)
	external_case_numbers: List[str] = Field(default_factory=list)
	
	# Closure information
	root_cause: Optional[str] = None
	resolution_summary: Optional[str] = None
	closed_by: Optional[str] = None
	closed_at: Optional[datetime] = None
	
	created_by: str = Field(description="Incident creator")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class IncidentAction(BaseModel):
	"""Incident response action tracking"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	incident_id: str = Field(description="Associated incident")
	
	action_title: str = Field(description="Action title")
	action_description: str = Field(description="Detailed action description")
	action_type: ActionType
	
	# Action execution
	assigned_to: str = Field(description="Action assignee")
	assigned_by: str = Field(description="Who assigned the action")
	assigned_at: datetime = Field(default_factory=datetime.utcnow)
	
	# Priority and timing
	priority: IncidentSeverity = IncidentSeverity.MEDIUM
	due_date: Optional[datetime] = None
	estimated_duration: Optional[timedelta] = None
	
	# Status tracking
	action_status: str = Field(default="pending")
	progress_percentage: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Execution details
	start_time: Optional[datetime] = None
	completion_time: Optional[datetime] = None
	actual_duration: Optional[timedelta] = None
	
	# Results and outcomes
	action_results: Dict[str, Any] = Field(default_factory=dict)
	success: Optional[bool] = None
	complications: List[str] = Field(default_factory=list)
	
	# Dependencies and blockers
	depends_on: List[str] = Field(default_factory=list)
	blocks: List[str] = Field(default_factory=list)
	prerequisites: List[str] = Field(default_factory=list)
	
	# Evidence and documentation
	evidence_collected: List[str] = Field(default_factory=list)
	documentation_links: List[str] = Field(default_factory=list)
	supporting_files: List[str] = Field(default_factory=list)
	
	# Communication
	updates: List[Dict[str, Any]] = Field(default_factory=list)
	escalation_required: bool = False
	escalated_to: Optional[str] = None
	escalated_at: Optional[datetime] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class IncidentEvidence(BaseModel):
	"""Digital evidence collection and preservation for incidents"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	incident_id: str = Field(description="Associated incident")
	
	evidence_name: str = Field(description="Evidence identifier")
	evidence_type: EvidenceType
	evidence_description: str = Field(description="Evidence description")
	
	# Source information
	source_system: str = Field(description="Source system")
	source_location: str = Field(description="Source location/path")
	collection_method: str = Field(description="Collection method")
	
	# File metadata
	file_name: Optional[str] = None
	file_size: Optional[int] = None
	file_hash_md5: Optional[str] = None
	file_hash_sha1: Optional[str] = None
	file_hash_sha256: Optional[str] = None
	
	# Collection details
	collected_by: str = Field(description="Evidence collector")
	collection_timestamp: datetime = Field(default_factory=datetime.utcnow)
	collection_tools: List[str] = Field(default_factory=list)
	
	# Chain of custody
	custody_log: List[Dict[str, Any]] = Field(default_factory=list)
	current_custodian: str = Field(description="Current custodian")
	storage_location: str = Field(description="Storage location")
	
	# Integrity verification
	integrity_verified: bool = False
	verification_method: Optional[str] = None
	verification_timestamp: Optional[datetime] = None
	verification_results: Dict[str, Any] = Field(default_factory=dict)
	
	# Analysis status
	analysis_requested: bool = False
	analysis_assigned_to: Optional[str] = None
	analysis_started: Optional[datetime] = None
	analysis_completed: Optional[datetime] = None
	analysis_results: Dict[str, Any] = Field(default_factory=dict)
	
	# Legal considerations
	legal_hold: bool = False
	admissibility_reviewed: bool = False
	privilege_considerations: List[str] = Field(default_factory=list)
	
	# Evidence relationships
	related_evidence: List[str] = Field(default_factory=list)
	parent_evidence: Optional[str] = None
	derived_evidence: List[str] = Field(default_factory=list)
	
	# Retention and disposal
	retention_period: Optional[timedelta] = None
	disposal_date: Optional[datetime] = None
	disposal_method: Optional[str] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class IncidentCommunication(BaseModel):
	"""Incident communication and notification tracking"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	incident_id: str = Field(description="Associated incident")
	
	communication_type: str = Field(description="Type of communication")
	communication_method: str = Field(description="Communication method")
	communication_subject: str = Field(description="Communication subject")
	
	# Recipients
	internal_recipients: List[str] = Field(default_factory=list)
	external_recipients: List[str] = Field(default_factory=list)
	regulatory_recipients: List[str] = Field(default_factory=list)
	
	# Content
	message_content: str = Field(description="Message content")
	attachments: List[str] = Field(default_factory=list)
	
	# Timing and urgency
	urgency_level: IncidentSeverity = IncidentSeverity.MEDIUM
	scheduled_time: Optional[datetime] = None
	sent_time: Optional[datetime] = None
	
	# Delivery tracking
	delivery_status: str = Field(default="pending")
	delivery_confirmations: List[Dict[str, Any]] = Field(default_factory=list)
	read_receipts: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Response tracking
	responses_received: List[Dict[str, Any]] = Field(default_factory=list)
	acknowledgments: List[str] = Field(default_factory=list)
	
	# Template and automation
	template_used: Optional[str] = None
	automated_communication: bool = False
	trigger_conditions: Dict[str, Any] = Field(default_factory=dict)
	
	# Compliance and legal
	regulatory_requirement: bool = False
	legal_review_required: bool = False
	approved_by: Optional[str] = None
	approved_at: Optional[datetime] = None
	
	sent_by: str = Field(description="Communication sender")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class IncidentTimeline(BaseModel):
	"""Incident timeline and chronological event tracking"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	incident_id: str = Field(description="Associated incident")
	
	event_timestamp: datetime = Field(description="When the event occurred")
	event_type: str = Field(description="Type of timeline event")
	event_description: str = Field(description="Event description")
	
	# Event details
	event_source: str = Field(description="Source of the event information")
	confidence_level: Decimal = Field(default=Decimal('100.0'), ge=0, le=100)
	
	# System and user context
	affected_systems: List[str] = Field(default_factory=list)
	involved_users: List[str] = Field(default_factory=list)
	involved_processes: List[str] = Field(default_factory=list)
	
	# Technical details
	log_entries: List[Dict[str, Any]] = Field(default_factory=list)
	network_connections: List[Dict[str, Any]] = Field(default_factory=list)
	file_operations: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Evidence correlation
	supporting_evidence: List[str] = Field(default_factory=list)
	contradicting_evidence: List[str] = Field(default_factory=list)
	
	# Analysis and interpretation
	analysis_notes: str = Field(default="")
	investigator_comments: List[Dict[str, Any]] = Field(default_factory=list)
	significance_rating: Decimal = Field(default=Decimal('50.0'), ge=0, le=100)
	
	# Timeline relationships
	preceded_by: List[str] = Field(default_factory=list)
	followed_by: List[str] = Field(default_factory=list)
	concurrent_events: List[str] = Field(default_factory=list)
	
	# Verification status
	verified: bool = False
	verified_by: Optional[str] = None
	verification_method: Optional[str] = None
	verification_timestamp: Optional[datetime] = None
	
	created_by: str = Field(description="Timeline entry creator")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class IncidentMetrics(BaseModel):
	"""Incident response metrics and performance indicators"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	metric_period_start: datetime
	metric_period_end: datetime
	
	# Incident volume
	total_incidents: int = 0
	new_incidents: int = 0
	resolved_incidents: int = 0
	open_incidents: int = 0
	
	# Severity distribution
	critical_incidents: int = 0
	high_incidents: int = 0
	medium_incidents: int = 0
	low_incidents: int = 0
	
	# Category distribution
	incidents_by_category: Dict[str, int] = Field(default_factory=dict)
	malware_incidents: int = 0
	phishing_incidents: int = 0
	data_breach_incidents: int = 0
	
	# Response time metrics
	mean_time_to_acknowledge: Optional[timedelta] = None
	mean_time_to_containment: Optional[timedelta] = None
	mean_time_to_resolution: Optional[timedelta] = None
	median_response_time: Optional[timedelta] = None
	
	# SLA compliance
	sla_breaches: int = 0
	critical_sla_compliance: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	overall_sla_compliance: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Team performance
	active_responders: int = 0
	average_incidents_per_responder: Decimal = Field(default=Decimal('0.0'))
	responder_workload_distribution: Dict[str, int] = Field(default_factory=dict)
	
	# Quality metrics
	false_positive_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	escalation_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	repeat_incident_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Communication metrics
	stakeholder_notifications: int = 0
	regulatory_notifications: int = 0
	average_notification_time: Optional[timedelta] = None
	
	# Business impact
	total_downtime: timedelta = Field(default=timedelta())
	estimated_financial_impact: Decimal = Field(default=Decimal('0.0'))
	affected_users_total: int = 0
	affected_systems_total: int = 0
	
	# Process effectiveness
	containment_success_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	eradication_success_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	recovery_success_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Trend analysis
	incident_trend: str = Field(default="stable")  # increasing, decreasing, stable
	severity_trend: str = Field(default="stable")
	response_time_trend: str = Field(default="stable")
	
	# Continuous improvement
	lessons_learned_documented: int = 0
	process_improvements_implemented: int = 0
	training_sessions_conducted: int = 0
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class IncidentPlaybook(BaseModel):
	"""Incident response playbook and procedure definition"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	playbook_name: str = Field(description="Playbook name")
	playbook_description: str = Field(description="Playbook description")
	playbook_version: str = Field(description="Playbook version")
	
	# Applicability
	incident_categories: List[IncidentCategory] = Field(default_factory=list)
	severity_levels: List[IncidentSeverity] = Field(default_factory=list)
	trigger_conditions: Dict[str, Any] = Field(default_factory=dict)
	
	# Playbook structure
	phases: List[Dict[str, Any]] = Field(default_factory=list)
	procedures: List[Dict[str, Any]] = Field(default_factory=list)
	checklists: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Response team roles
	required_roles: List[str] = Field(default_factory=list)
	role_assignments: Dict[str, str] = Field(default_factory=dict)
	escalation_matrix: Dict[str, Any] = Field(default_factory=dict)
	
	# Timeline and SLAs
	target_response_times: Dict[str, timedelta] = Field(default_factory=dict)
	phase_durations: Dict[str, timedelta] = Field(default_factory=dict)
	
	# Communication templates
	notification_templates: Dict[str, str] = Field(default_factory=dict)
	status_update_templates: Dict[str, str] = Field(default_factory=dict)
	
	# Tools and resources
	required_tools: List[str] = Field(default_factory=list)
	reference_materials: List[str] = Field(default_factory=list)
	contact_lists: Dict[str, List[str]] = Field(default_factory=dict)
	
	# Quality and validation
	testing_schedule: Optional[str] = None
	last_tested: Optional[datetime] = None
	test_results: Dict[str, Any] = Field(default_factory=dict)
	
	# Usage tracking
	usage_count: int = 0
	success_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	average_execution_time: Optional[timedelta] = None
	
	# Approval and governance
	approved_by: str = Field(description="Playbook approver")
	approval_date: datetime = Field(default_factory=datetime.utcnow)
	review_frequency: str = Field(default="annual")
	next_review_date: Optional[datetime] = None
	
	# Versioning and change control
	change_log: List[Dict[str, Any]] = Field(default_factory=list)
	previous_versions: List[str] = Field(default_factory=list)
	
	is_active: bool = True
	created_by: str = Field(description="Playbook creator")
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class PostIncidentReview(BaseModel):
	"""Post-incident review and lessons learned"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	incident_id: str = Field(description="Associated incident")
	
	review_name: str = Field(description="Review session name")
	review_date: datetime = Field(description="Review session date")
	
	# Participants
	facilitator: str = Field(description="Review facilitator")
	participants: List[str] = Field(default_factory=list)
	stakeholders: List[str] = Field(default_factory=list)
	
	# Review scope
	review_objectives: List[str] = Field(default_factory=list)
	areas_reviewed: List[str] = Field(default_factory=list)
	
	# Timeline analysis
	response_timeline_accuracy: Decimal = Field(ge=0, le=100)
	timeline_gaps_identified: List[str] = Field(default_factory=list)
	
	# Effectiveness assessment
	detection_effectiveness: Decimal = Field(ge=0, le=100)
	response_effectiveness: Decimal = Field(ge=0, le=100)
	communication_effectiveness: Decimal = Field(ge=0, le=100)
	containment_effectiveness: Decimal = Field(ge=0, le=100)
	
	# What worked well
	successes: List[str] = Field(default_factory=list)
	effective_procedures: List[str] = Field(default_factory=list)
	good_decisions: List[str] = Field(default_factory=list)
	
	# Areas for improvement
	failures: List[str] = Field(default_factory=list)
	gaps_identified: List[str] = Field(default_factory=list)
	process_breakdowns: List[str] = Field(default_factory=list)
	
	# Lessons learned
	key_lessons: List[str] = Field(default_factory=list)
	technical_lessons: List[str] = Field(default_factory=list)
	process_lessons: List[str] = Field(default_factory=list)
	communication_lessons: List[str] = Field(default_factory=list)
	
	# Recommendations
	immediate_actions: List[Dict[str, Any]] = Field(default_factory=list)
	process_improvements: List[Dict[str, Any]] = Field(default_factory=list)
	training_recommendations: List[str] = Field(default_factory=list)
	tool_improvements: List[str] = Field(default_factory=list)
	
	# Follow-up tracking
	action_items: List[Dict[str, Any]] = Field(default_factory=list)
	responsible_parties: Dict[str, str] = Field(default_factory=dict)
	target_completion_dates: Dict[str, datetime] = Field(default_factory=dict)
	
	# Documentation
	review_notes: str = Field(default="")
	supporting_documents: List[str] = Field(default_factory=list)
	recordings: List[str] = Field(default_factory=list)
	
	# Review quality
	completeness_score: Decimal = Field(ge=0, le=100)
	stakeholder_satisfaction: Decimal = Field(ge=0, le=100)
	
	# Status and tracking
	review_status: str = Field(default="completed")
	follow_up_required: bool = False
	next_review_date: Optional[datetime] = None
	
	created_by: str = Field(description="Review creator")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None