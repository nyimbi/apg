"""
APG Threat Detection & Monitoring - Pydantic Models

Enterprise-grade threat detection models with advanced validation,
behavioral analytics, and security intelligence capabilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic import ConfigDict
from uuid_extensions import uuid7str


class ThreatSeverity(str, Enum):
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	INFO = "info"


class ThreatStatus(str, Enum):
	ACTIVE = "active"
	INVESTIGATING = "investigating"
	CONTAINED = "contained"
	RESOLVED = "resolved"
	FALSE_POSITIVE = "false_positive"


class EventType(str, Enum):
	AUTHENTICATION = "authentication"
	AUTHORIZATION = "authorization"
	NETWORK_TRAFFIC = "network_traffic"
	FILE_ACCESS = "file_access"
	SYSTEM_CALL = "system_call"
	DATABASE_ACCESS = "database_access"
	APPLICATION_EVENT = "application_event"
	SECURITY_ALERT = "security_alert"


class AnalysisEngine(str, Enum):
	RULE_BASED = "rule_based"
	MACHINE_LEARNING = "machine_learning"
	BEHAVIORAL_ANALYSIS = "behavioral_analysis"
	STATISTICAL_ANALYSIS = "statistical_analysis"
	THREAT_INTELLIGENCE = "threat_intelligence"


class ResponseAction(str, Enum):
	ALERT = "alert"
	ISOLATE = "isolate"
	QUARANTINE = "quarantine"
	BLOCK = "block"
	MONITOR = "monitor"
	ESCALATE = "escalate"


class SecurityEvent(BaseModel):
	"""Security event from various sources"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	event_id: str = Field(description="Original event identifier")
	event_type: EventType
	source_system: str = Field(description="System that generated the event")
	source_ip: Optional[str] = None
	destination_ip: Optional[str] = None
	user_id: Optional[str] = None
	username: Optional[str] = None
	asset_id: Optional[str] = None
	hostname: Optional[str] = None
	
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	raw_data: Dict[str, Any] = Field(default_factory=dict)
	normalized_data: Dict[str, Any] = Field(default_factory=dict)
	
	geolocation: Optional[Dict[str, Any]] = None
	user_agent: Optional[str] = None
	process_name: Optional[str] = None
	command_line: Optional[str] = None
	file_path: Optional[str] = None
	
	risk_score: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	confidence: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ThreatIndicator(BaseModel):
	"""Indicators of Compromise (IOCs) and threat indicators"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	indicator_type: str = Field(description="Type of indicator (IP, domain, hash, etc.)")
	indicator_value: str = Field(description="Actual indicator value")
	description: str = Field(description="Description of the threat")
	
	severity: ThreatSeverity = ThreatSeverity.MEDIUM
	confidence: Decimal = Field(ge=0, le=100, description="Confidence in indicator")
	
	source: str = Field(description="Source of the indicator")
	tags: List[str] = Field(default_factory=list)
	
	first_seen: datetime = Field(default_factory=datetime.utcnow)
	last_seen: datetime = Field(default_factory=datetime.utcnow)
	expiry_date: Optional[datetime] = None
	
	malware_families: List[str] = Field(default_factory=list)
	attack_techniques: List[str] = Field(default_factory=list)
	
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class SecurityIncident(BaseModel):
	"""Security incident with investigation details"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	title: str = Field(description="Incident title")
	description: str = Field(description="Detailed incident description")
	
	severity: ThreatSeverity = ThreatSeverity.MEDIUM
	status: ThreatStatus = ThreatStatus.ACTIVE
	
	incident_type: str = Field(description="Type of security incident")
	attack_vector: Optional[str] = None
	
	affected_systems: List[str] = Field(default_factory=list)
	affected_users: List[str] = Field(default_factory=list)
	affected_assets: List[str] = Field(default_factory=list)
	
	indicators: List[str] = Field(default_factory=list, description="Associated IOC IDs")
	events: List[str] = Field(default_factory=list, description="Associated event IDs")
	
	assigned_to: Optional[str] = None
	assignee_name: Optional[str] = None
	
	first_detected: datetime = Field(default_factory=datetime.utcnow)
	last_activity: datetime = Field(default_factory=datetime.utcnow)
	resolved_at: Optional[datetime] = None
	
	timeline: List[Dict[str, Any]] = Field(default_factory=list)
	evidence: List[Dict[str, Any]] = Field(default_factory=list)
	
	containment_actions: List[Dict[str, Any]] = Field(default_factory=list)
	remediation_actions: List[Dict[str, Any]] = Field(default_factory=list)
	
	impact_assessment: Dict[str, Any] = Field(default_factory=dict)
	root_cause: Optional[str] = None
	lessons_learned: Optional[str] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class BehavioralProfile(BaseModel):
	"""User/Entity behavioral analysis profile"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	entity_id: str = Field(description="User or entity being profiled")
	entity_type: str = Field(description="Type of entity (user, system, service)")
	entity_name: str = Field(description="Name of the entity")
	
	profile_period_start: datetime = Field(description="Profile period start")
	profile_period_end: datetime = Field(description="Profile period end")
	
	baseline_established: bool = False
	baseline_confidence: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	normal_login_hours: List[int] = Field(default_factory=list)
	normal_login_locations: List[str] = Field(default_factory=list)
	normal_access_patterns: Dict[str, Any] = Field(default_factory=dict)
	
	typical_systems_accessed: List[str] = Field(default_factory=list)
	typical_data_accessed: List[str] = Field(default_factory=list)
	typical_operations: List[str] = Field(default_factory=list)
	
	peer_group: Optional[str] = None
	peer_comparison_score: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	anomaly_score: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	risk_score: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	recent_anomalies: List[Dict[str, Any]] = Field(default_factory=list)
	behavioral_changes: List[Dict[str, Any]] = Field(default_factory=list)
	
	last_analyzed: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ThreatIntelligence(BaseModel):
	"""External threat intelligence data"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	feed_source: str = Field(description="Intelligence feed source")
	feed_name: str = Field(description="Name of the feed")
	
	intelligence_type: str = Field(description="Type of intelligence")
	title: str = Field(description="Intelligence title")
	description: str = Field(description="Detailed description")
	
	severity: ThreatSeverity = ThreatSeverity.MEDIUM
	confidence: Decimal = Field(ge=0, le=100)
	
	indicators: List[str] = Field(default_factory=list, description="Associated IOC IDs")
	
	threat_actor: Optional[str] = None
	campaign: Optional[str] = None
	malware_family: Optional[str] = None
	
	attack_patterns: List[str] = Field(default_factory=list)
	kill_chain_phases: List[str] = Field(default_factory=list)
	
	industries_targeted: List[str] = Field(default_factory=list)
	countries_targeted: List[str] = Field(default_factory=list)
	
	recommendations: List[str] = Field(default_factory=list)
	mitigation_strategies: List[str] = Field(default_factory=list)
	
	published_date: datetime = Field(description="Original publication date")
	valid_until: Optional[datetime] = None
	
	raw_data: Dict[str, Any] = Field(default_factory=dict)
	processed_data: Dict[str, Any] = Field(default_factory=dict)
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class SecurityRule(BaseModel):
	"""Security detection rule"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	name: str = Field(description="Rule name")
	description: str = Field(description="Rule description")
	
	rule_type: str = Field(description="Type of rule")
	analysis_engine: AnalysisEngine = AnalysisEngine.RULE_BASED
	
	query: str = Field(description="Detection query or logic")
	query_language: str = Field(default="SQL", description="Language used for query")
	
	severity: ThreatSeverity = ThreatSeverity.MEDIUM
	confidence: Decimal = Field(ge=0, le=100, default=Decimal('80.0'))
	
	event_types: List[EventType] = Field(default_factory=list)
	data_sources: List[str] = Field(default_factory=list)
	
	false_positive_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	mitre_techniques: List[str] = Field(default_factory=list)
	tags: List[str] = Field(default_factory=list)
	
	threshold_conditions: Dict[str, Any] = Field(default_factory=dict)
	aggregation_rules: Dict[str, Any] = Field(default_factory=dict)
	
	response_actions: List[ResponseAction] = Field(default_factory=list)
	
	is_active: bool = True
	is_custom: bool = False
	
	created_by: str = Field(description="Rule creator")
	last_triggered: Optional[datetime] = None
	trigger_count: int = 0
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class IncidentResponse(BaseModel):
	"""Automated incident response execution"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	incident_id: str = Field(description="Associated incident ID")
	
	playbook_name: str = Field(description="Response playbook used")
	playbook_version: str = Field(description="Playbook version")
	
	trigger_event: str = Field(description="Event that triggered response")
	
	response_actions: List[Dict[str, Any]] = Field(default_factory=list)
	executed_actions: List[Dict[str, Any]] = Field(default_factory=list)
	failed_actions: List[Dict[str, Any]] = Field(default_factory=list)
	
	start_time: datetime = Field(default_factory=datetime.utcnow)
	end_time: Optional[datetime] = None
	
	status: str = Field(default="running")
	success_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	containment_successful: bool = False
	containment_time: Optional[timedelta] = None
	
	escalated: bool = False
	escalation_reason: Optional[str] = None
	escalated_to: Optional[str] = None
	
	human_intervention_required: bool = False
	intervention_reason: Optional[str] = None
	
	logs: List[Dict[str, Any]] = Field(default_factory=list)
	metrics: Dict[str, Any] = Field(default_factory=dict)
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ThreatAnalysis(BaseModel):
	"""Advanced threat analysis results"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	analysis_type: str = Field(description="Type of analysis performed")
	analysis_engine: AnalysisEngine
	
	input_events: List[str] = Field(default_factory=list, description="Input event IDs")
	related_incidents: List[str] = Field(default_factory=list)
	
	threat_score: Decimal = Field(ge=0, le=100)
	confidence_score: Decimal = Field(ge=0, le=100)
	
	findings: List[Dict[str, Any]] = Field(default_factory=list)
	recommendations: List[str] = Field(default_factory=list)
	
	attack_timeline: List[Dict[str, Any]] = Field(default_factory=list)
	
	attributed_threat_actor: Optional[str] = None
	attribution_confidence: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	kill_chain_mapping: Dict[str, List[str]] = Field(default_factory=dict)
	mitre_tactics: List[str] = Field(default_factory=list)
	mitre_techniques: List[str] = Field(default_factory=list)
	
	impact_assessment: Dict[str, Any] = Field(default_factory=dict)
	risk_assessment: Dict[str, Any] = Field(default_factory=dict)
	
	analysis_start: datetime = Field(default_factory=datetime.utcnow)
	analysis_end: Optional[datetime] = None
	processing_time: Optional[timedelta] = None
	
	model_versions: Dict[str, str] = Field(default_factory=dict)
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class SecurityMetrics(BaseModel):
	"""Security operations metrics and KPIs"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	metric_period_start: datetime
	metric_period_end: datetime
	
	total_events_processed: int = 0
	total_alerts_generated: int = 0
	total_incidents_created: int = 0
	
	mean_time_to_detection: Optional[timedelta] = None
	mean_time_to_response: Optional[timedelta] = None
	mean_time_to_resolution: Optional[timedelta] = None
	
	false_positive_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	alert_accuracy: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	incidents_by_severity: Dict[str, int] = Field(default_factory=dict)
	incidents_by_status: Dict[str, int] = Field(default_factory=dict)
	
	top_attack_vectors: List[Dict[str, Any]] = Field(default_factory=list)
	top_targeted_assets: List[Dict[str, Any]] = Field(default_factory=list)
	
	automated_response_success_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	escalation_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	threat_intelligence_matches: int = 0
	behavioral_anomalies_detected: int = 0
	
	system_performance_metrics: Dict[str, Any] = Field(default_factory=dict)
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ForensicEvidence(BaseModel):
	"""Digital forensic evidence collection"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	incident_id: str = Field(description="Related incident ID")
	
	evidence_type: str = Field(description="Type of evidence")
	source_system: str = Field(description="System where evidence was collected")
	
	collection_method: str = Field(description="How evidence was collected")
	collection_tool: str = Field(description="Tool used for collection")
	
	file_path: Optional[str] = None
	file_hash: Optional[str] = None
	file_size: Optional[int] = None
	
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	chain_of_custody: List[Dict[str, Any]] = Field(default_factory=list)
	
	integrity_verified: bool = False
	integrity_hash: Optional[str] = None
	
	analysis_results: Dict[str, Any] = Field(default_factory=dict)
	
	collected_by: str = Field(description="Person who collected evidence")
	collected_at: datetime = Field(default_factory=datetime.utcnow)
	
	retention_period: Optional[timedelta] = None
	destruction_date: Optional[datetime] = None
	
	legal_hold: bool = False
	legal_hold_reason: Optional[str] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None