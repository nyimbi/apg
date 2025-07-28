"""
APG Threat Hunting Platform - Pydantic Models

Enterprise threat hunting models with advanced query capabilities, 
hypothesis-driven investigations, and collaborative hunting workflows.

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


class HuntType(str, Enum):
	HYPOTHESIS_DRIVEN = "hypothesis_driven"
	INTELLIGENCE_DRIVEN = "intelligence_driven"
	ANOMALY_DRIVEN = "anomaly_driven"
	IOC_HUNTING = "ioc_hunting"
	BEHAVIORAL_HUNTING = "behavioral_hunting"
	PROACTIVE_HUNTING = "proactive_hunting"


class HuntStatus(str, Enum):
	PLANNING = "planning"
	ACTIVE = "active"
	INVESTIGATING = "investigating"
	VALIDATING = "validating"
	CONCLUDED = "concluded"
	SUSPENDED = "suspended"
	CANCELLED = "cancelled"


class HuntPriority(str, Enum):
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"


class QueryLanguage(str, Enum):
	KQL = "kql"  # Kusto Query Language
	SPL = "spl"  # Splunk Processing Language
	SQL = "sql"
	YARA = "yara"
	SIGMA = "sigma"
	ELASTIC = "elastic"
	CUSTOM = "custom"


class HuntOutcome(str, Enum):
	TRUE_POSITIVE = "true_positive"
	FALSE_POSITIVE = "false_positive"
	BENIGN_POSITIVE = "benign_positive"
	INCONCLUSIVE = "inconclusive"
	REQUIRES_ESCALATION = "requires_escalation"


class EvidenceType(str, Enum):
	LOG_ENTRY = "log_entry"
	NETWORK_TRAFFIC = "network_traffic"
	FILE_ARTIFACT = "file_artifact"
	REGISTRY_KEY = "registry_key"
	PROCESS_EXECUTION = "process_execution"
	MEMORY_DUMP = "memory_dump"
	DISK_IMAGE = "disk_image"
	SCREENSHOT = "screenshot"


class ThreatHunt(BaseModel):
	"""Comprehensive threat hunting campaign definition"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	name: str = Field(description="Hunt campaign name")
	description: str = Field(description="Hunt description and objectives")
	hunt_type: HuntType
	priority: HuntPriority = HuntPriority.MEDIUM
	
	# Hunt hypothesis and objectives
	hypothesis: str = Field(description="Hunt hypothesis statement")
	objectives: List[str] = Field(default_factory=list)
	success_criteria: List[str] = Field(default_factory=list)
	
	# Threat intelligence context
	threat_actors: List[str] = Field(default_factory=list)
	attack_techniques: List[str] = Field(default_factory=list, description="MITRE ATT&CK techniques")
	indicators_of_compromise: List[str] = Field(default_factory=list)
	
	# Hunt scope and timeline
	hunt_scope: Dict[str, Any] = Field(default_factory=dict)
	data_sources: List[str] = Field(default_factory=list)
	time_range_start: datetime = Field(description="Hunt time range start")
	time_range_end: datetime = Field(description="Hunt time range end")
	
	# Hunt team and assignments
	lead_hunter: str = Field(description="Lead hunter responsible")
	hunt_team: List[str] = Field(default_factory=list)
	assigned_analysts: List[str] = Field(default_factory=list)
	
	# Hunt workflow
	hunt_phases: List[Dict[str, Any]] = Field(default_factory=list)
	current_phase: Optional[str] = None
	
	# Status and progress
	status: HuntStatus = HuntStatus.PLANNING
	progress_percentage: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Results and findings
	findings_count: int = 0
	true_positives: int = 0
	false_positives: int = 0
	investigation_required: int = 0
	
	# Hunt metrics
	queries_executed: int = 0
	data_volume_processed: Decimal = Field(default=Decimal('0.0'))
	hunt_duration: Optional[timedelta] = None
	
	# Collaboration and documentation
	hunt_notes: List[Dict[str, Any]] = Field(default_factory=list)
	lessons_learned: List[str] = Field(default_factory=list)
	recommendations: List[str] = Field(default_factory=list)
	
	# Compliance and approval
	requires_approval: bool = False
	approved_by: Optional[str] = None
	approved_at: Optional[datetime] = None
	
	is_active: bool = True
	created_by: str = Field(description="Hunt creator")
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class HuntQuery(BaseModel):
	"""Hunt query specification and execution tracking"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	hunt_id: str = Field(description="Associated hunt campaign")
	
	query_name: str = Field(description="Query name or identifier")
	query_description: str = Field(description="Query purpose and logic")
	
	# Query specification
	query_language: QueryLanguage
	query_text: str = Field(description="Query code/text")
	query_parameters: Dict[str, Any] = Field(default_factory=dict)
	
	# Data source targeting
	target_data_sources: List[str] = Field(default_factory=list)
	target_indexes: List[str] = Field(default_factory=list)
	
	# Execution configuration
	time_range_start: datetime
	time_range_end: datetime
	max_results: Optional[int] = None
	timeout_seconds: int = Field(default=300)
	
	# Query execution tracking
	execution_count: int = 0
	last_executed: Optional[datetime] = None
	average_execution_time: Optional[Decimal] = None
	
	# Results summary
	total_results: int = 0
	unique_results: int = 0
	result_hash: Optional[str] = None
	
	# Query optimization
	query_plan: Optional[Dict[str, Any]] = None
	performance_metrics: Dict[str, Any] = Field(default_factory=dict)
	optimization_suggestions: List[str] = Field(default_factory=list)
	
	# Validation and quality
	is_validated: bool = False
	validation_results: Dict[str, Any] = Field(default_factory=dict)
	false_positive_rate: Optional[Decimal] = None
	
	# Query metadata
	query_tags: List[str] = Field(default_factory=list)
	mitre_techniques: List[str] = Field(default_factory=list)
	
	created_by: str = Field(description="Query author")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class HuntFinding(BaseModel):
	"""Hunt finding and investigation tracking"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	hunt_id: str = Field(description="Associated hunt campaign")
	query_id: Optional[str] = None
	
	finding_title: str = Field(description="Finding summary title")
	finding_description: str = Field(description="Detailed finding description")
	
	# Finding classification
	finding_type: str = Field(description="Type of finding")
	severity: HuntPriority = HuntPriority.MEDIUM
	confidence_score: Decimal = Field(ge=0, le=100)
	
	# Threat context
	related_threats: List[str] = Field(default_factory=list)
	attack_techniques: List[str] = Field(default_factory=list)
	indicators_found: List[str] = Field(default_factory=list)
	
	# Finding data
	raw_data: Dict[str, Any] = Field(default_factory=dict)
	processed_data: Dict[str, Any] = Field(default_factory=dict)
	data_sources: List[str] = Field(default_factory=list)
	
	# Timeline and context
	first_observed: datetime = Field(description="First occurrence time")
	last_observed: datetime = Field(description="Last occurrence time")
	event_count: int = Field(default=1)
	
	# Investigation status
	investigation_status: str = Field(default="new")
	assigned_investigator: Optional[str] = None
	investigation_priority: HuntPriority = HuntPriority.MEDIUM
	
	# Analysis results
	analysis_notes: List[Dict[str, Any]] = Field(default_factory=list)
	investigation_timeline: List[Dict[str, Any]] = Field(default_factory=list)
	related_findings: List[str] = Field(default_factory=list)
	
	# Final disposition
	outcome: Optional[HuntOutcome] = None
	outcome_reason: Optional[str] = None
	outcome_confidence: Optional[Decimal] = None
	
	# Response actions
	response_actions: List[str] = Field(default_factory=list)
	containment_actions: List[str] = Field(default_factory=list)
	remediation_actions: List[str] = Field(default_factory=list)
	
	# Evidence collection
	evidence_collected: List[str] = Field(default_factory=list)
	evidence_preservation: Dict[str, Any] = Field(default_factory=dict)
	
	# Quality metrics
	false_positive_likelihood: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	business_impact_score: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Escalation tracking
	escalated: bool = False
	escalation_reason: Optional[str] = None
	escalated_to: Optional[str] = None
	escalated_at: Optional[datetime] = None
	
	created_by: str = Field(description="Finding discoverer")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class HuntEvidence(BaseModel):
	"""Digital evidence collection and preservation"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	hunt_id: str = Field(description="Associated hunt campaign")
	finding_id: Optional[str] = None
	
	evidence_name: str = Field(description="Evidence identifier")
	evidence_type: EvidenceType
	evidence_description: str = Field(description="Evidence description")
	
	# Evidence source
	source_system: str = Field(description="System where evidence was found")
	source_location: str = Field(description="Specific location/path")
	collection_method: str = Field(description="How evidence was collected")
	
	# Evidence metadata
	file_hash: Optional[str] = None
	file_size: Optional[int] = None
	file_type: Optional[str] = None
	creation_time: Optional[datetime] = None
	modification_time: Optional[datetime] = None
	
	# Collection details
	collected_by: str = Field(description="Evidence collector")
	collection_timestamp: datetime = Field(default_factory=datetime.utcnow)
	collection_tools: List[str] = Field(default_factory=list)
	
	# Chain of custody
	custody_chain: List[Dict[str, Any]] = Field(default_factory=list)
	access_log: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Storage and preservation
	storage_location: str = Field(description="Evidence storage location")
	backup_locations: List[str] = Field(default_factory=list)
	retention_period: Optional[timedelta] = None
	
	# Integrity verification
	integrity_hashes: Dict[str, str] = Field(default_factory=dict)
	digital_signature: Optional[str] = None
	verification_status: str = Field(default="pending")
	
	# Analysis results
	analysis_performed: List[str] = Field(default_factory=list)
	analysis_results: Dict[str, Any] = Field(default_factory=dict)
	malware_analysis: Optional[Dict[str, Any]] = None
	
	# Legal and compliance
	legal_hold: bool = False
	compliance_requirements: List[str] = Field(default_factory=list)
	admissibility_status: str = Field(default="pending")
	
	# Evidence relationships
	related_evidence: List[str] = Field(default_factory=list)
	parent_evidence: Optional[str] = None
	child_evidence: List[str] = Field(default_factory=list)
	
	is_sealed: bool = False
	sealed_by: Optional[str] = None
	sealed_at: Optional[datetime] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class HuntWorkflow(BaseModel):
	"""Hunt workflow and process automation"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	hunt_id: str = Field(description="Associated hunt campaign")
	
	workflow_name: str = Field(description="Workflow name")
	workflow_type: str = Field(description="Type of workflow")
	workflow_description: str = Field(description="Workflow purpose")
	
	# Workflow definition
	workflow_steps: List[Dict[str, Any]] = Field(default_factory=list)
	step_dependencies: Dict[str, List[str]] = Field(default_factory=dict)
	
	# Execution tracking
	current_step: Optional[str] = None
	completed_steps: List[str] = Field(default_factory=list)
	failed_steps: List[str] = Field(default_factory=list)
	
	# Workflow status
	workflow_status: str = Field(default="pending")
	progress_percentage: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Automation configuration
	automated_steps: List[str] = Field(default_factory=list)
	manual_approval_required: List[str] = Field(default_factory=list)
	notification_triggers: Dict[str, List[str]] = Field(default_factory=dict)
	
	# Execution results
	workflow_results: Dict[str, Any] = Field(default_factory=dict)
	step_outputs: Dict[str, Any] = Field(default_factory=dict)
	
	# Timing
	start_time: Optional[datetime] = None
	end_time: Optional[datetime] = None
	estimated_duration: Optional[timedelta] = None
	actual_duration: Optional[timedelta] = None
	
	# Error handling
	error_handling: Dict[str, Any] = Field(default_factory=dict)
	rollback_actions: List[str] = Field(default_factory=list)
	
	created_by: str = Field(description="Workflow creator")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class HuntMetrics(BaseModel):
	"""Threat hunting metrics and KPIs"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	metric_period_start: datetime
	metric_period_end: datetime
	
	# Hunt campaign metrics
	total_hunts: int = 0
	active_hunts: int = 0
	completed_hunts: int = 0
	successful_hunts: int = 0
	
	# Hunt effectiveness
	total_findings: int = 0
	true_positives: int = 0
	false_positives: int = 0
	true_positive_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	false_positive_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Query performance
	total_queries_executed: int = 0
	average_query_execution_time: Optional[Decimal] = None
	query_success_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Data processing
	total_data_processed: Decimal = Field(default=Decimal('0.0'))
	average_data_volume_per_hunt: Decimal = Field(default=Decimal('0.0'))
	data_sources_coverage: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Hunter productivity
	total_hunters: int = 0
	active_hunters: int = 0
	average_hunts_per_hunter: Decimal = Field(default=Decimal('0.0'))
	hunter_efficiency_score: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Time metrics
	average_hunt_duration: Optional[timedelta] = None
	time_to_first_finding: Optional[timedelta] = None
	investigation_time: Optional[timedelta] = None
	
	# Quality metrics
	hunt_quality_score: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	documentation_completeness: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	evidence_preservation_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Threat coverage
	techniques_hunted: List[str] = Field(default_factory=list)
	threat_actors_tracked: List[str] = Field(default_factory=list)
	coverage_by_kill_chain: Dict[str, Decimal] = Field(default_factory=dict)
	
	# Business impact
	threats_detected: int = 0
	incidents_prevented: int = 0
	cost_avoidance: Optional[Decimal] = None
	mean_time_to_detection: Optional[timedelta] = None
	
	# Collaboration metrics
	cross_team_hunts: int = 0
	knowledge_sharing_sessions: int = 0
	hunt_templates_created: int = 0
	hunt_templates_reused: int = 0
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class HuntTemplate(BaseModel):
	"""Reusable hunt template and playbook"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	template_name: str = Field(description="Template name")
	template_description: str = Field(description="Template description")
	hunt_type: HuntType
	
	# Template definition
	hypothesis_template: str = Field(description="Hypothesis template")
	objective_templates: List[str] = Field(default_factory=list)
	query_templates: List[Dict[str, Any]] = Field(default_factory=list)
	
	# MITRE ATT&CK mapping
	attack_techniques: List[str] = Field(default_factory=list)
	attack_tactics: List[str] = Field(default_factory=list)
	
	# Data requirements
	required_data_sources: List[str] = Field(default_factory=list)
	optional_data_sources: List[str] = Field(default_factory=list)
	minimum_data_retention: timedelta = Field(default=timedelta(days=90))
	
	# Template parameters
	configurable_parameters: Dict[str, Any] = Field(default_factory=dict)
	parameter_validation: Dict[str, Any] = Field(default_factory=dict)
	
	# Quality and validation
	template_validation: Dict[str, Any] = Field(default_factory=dict)
	success_criteria: List[str] = Field(default_factory=list)
	expected_outcomes: Dict[str, Any] = Field(default_factory=dict)
	
	# Usage and effectiveness
	usage_count: int = 0
	success_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	average_execution_time: Optional[timedelta] = None
	
	# Template metadata
	difficulty_level: str = Field(default="medium")
	skill_requirements: List[str] = Field(default_factory=list)
	tools_required: List[str] = Field(default_factory=list)
	
	# Community and sharing
	is_public_template: bool = False
	template_rating: Optional[Decimal] = None
	community_feedback: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Versioning
	template_version: str = Field(description="Template version")
	changelog: List[Dict[str, Any]] = Field(default_factory=list)
	deprecated: bool = False
	
	created_by: str = Field(description="Template creator")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None