"""
APG Advanced Threat Intelligence - Pydantic Models

Enterprise threat intelligence models with advanced correlation,
attribution analysis, and predictive threat modeling capabilities.

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


class FeedType(str, Enum):
	COMMERCIAL = "commercial"
	OPEN_SOURCE = "open_source"
	GOVERNMENT = "government"
	INDUSTRY = "industry"
	DARK_WEB = "dark_web"
	PROPRIETARY = "proprietary"


class FeedFormat(str, Enum):
	STIX_TAXII = "stix_taxii"
	JSON = "json"
	XML = "xml"
	CSV = "csv"
	RSS = "rss"
	CUSTOM_API = "custom_api"


class IntelligenceType(str, Enum):
	IOC = "ioc"
	TTP = "ttp"
	ATTRIBUTION = "attribution"
	VULNERABILITY = "vulnerability"
	CAMPAIGN = "campaign"
	MALWARE = "malware"
	INFRASTRUCTURE = "infrastructure"


class ThreatSeverity(str, Enum):
	CRITICAL = "critical"
	HIGH = "high" 
	MEDIUM = "medium"
	LOW = "low"
	INFO = "info"


class ConfidenceLevel(str, Enum):
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	UNKNOWN = "unknown"


class ThreatActorType(str, Enum):
	NATION_STATE = "nation_state"
	CYBERCRIMINAL = "cybercriminal"
	HACKTIVIST = "hacktivist"
	INSIDER = "insider"
	TERRORIST = "terrorist"
	UNKNOWN = "unknown"


class AttributionMethod(str, Enum):
	TECHNICAL = "technical"
	BEHAVIORAL = "behavioral"
	LINGUISTIC = "linguistic"
	GEOPOLITICAL = "geopolitical"
	INFRASTRUCTURE = "infrastructure"
	MALWARE = "malware"


class IntelligenceFeed(BaseModel):
	"""Intelligence feed configuration and management"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	name: str = Field(description="Feed name")
	description: str = Field(description="Feed description")
	
	feed_type: FeedType
	feed_format: FeedFormat
	
	source_url: str = Field(description="Feed source URL")
	api_key: Optional[str] = None
	authentication_method: str = Field(default="none")
	
	update_frequency: int = Field(description="Update frequency in minutes")
	last_update: Optional[datetime] = None
	next_update: Optional[datetime] = None
	
	quality_score: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	reliability_score: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	total_indicators: int = 0
	active_indicators: int = 0
	
	processing_rules: Dict[str, Any] = Field(default_factory=dict)
	enrichment_config: Dict[str, Any] = Field(default_factory=dict)
	
	is_active: bool = True
	is_premium: bool = False
	
	tags: List[str] = Field(default_factory=list)
	categories: List[str] = Field(default_factory=list)
	
	cost_per_month: Optional[Decimal] = None
	vendor_name: Optional[str] = None
	contact_info: Dict[str, Any] = Field(default_factory=dict)
	
	performance_metrics: Dict[str, Any] = Field(default_factory=dict)
	error_log: List[Dict[str, Any]] = Field(default_factory=list)
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ThreatActor(BaseModel):
	"""Comprehensive threat actor profile"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	name: str = Field(description="Threat actor name")
	aliases: List[str] = Field(default_factory=list)
	
	actor_type: ThreatActorType
	sophistication_level: str = Field(description="Sophistication assessment")
	
	description: str = Field(description="Detailed actor description")
	
	primary_motivation: str = Field(description="Primary motivation")
	secondary_motivations: List[str] = Field(default_factory=list)
	
	capabilities: List[str] = Field(default_factory=list)
	resource_level: str = Field(description="Resource assessment")
	
	origin_country: Optional[str] = None
	target_countries: List[str] = Field(default_factory=list)
	target_industries: List[str] = Field(default_factory=list)
	
	attack_patterns: List[str] = Field(default_factory=list)
	tools_used: List[str] = Field(default_factory=list)
	malware_families: List[str] = Field(default_factory=list)
	
	infrastructure: Dict[str, Any] = Field(default_factory=dict)
	communication_patterns: Dict[str, Any] = Field(default_factory=dict)
	
	first_observed: datetime = Field(default_factory=datetime.utcnow)
	last_observed: datetime = Field(default_factory=datetime.utcnow)
	
	confidence_score: Decimal = Field(ge=0, le=100)
	threat_score: Decimal = Field(ge=0, le=100)
	
	attribution_methods: List[AttributionMethod] = Field(default_factory=list)
	attribution_sources: List[str] = Field(default_factory=list)
	
	associated_campaigns: List[str] = Field(default_factory=list)
	related_actors: List[str] = Field(default_factory=list)
	
	intelligence_sources: List[str] = Field(default_factory=list)
	
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class AttackCampaign(BaseModel):
	"""Multi-stage attack campaign tracking"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	name: str = Field(description="Campaign name")
	aliases: List[str] = Field(default_factory=list)
	
	description: str = Field(description="Campaign description")
	objectives: List[str] = Field(default_factory=list)
	
	attributed_actors: List[str] = Field(default_factory=list)
	attribution_confidence: Decimal = Field(ge=0, le=100)
	
	start_date: datetime = Field(description="Campaign start date")
	end_date: Optional[datetime] = None
	
	target_countries: List[str] = Field(default_factory=list)
	target_industries: List[str] = Field(default_factory=list)
	target_technologies: List[str] = Field(default_factory=list)
	
	attack_phases: List[Dict[str, Any]] = Field(default_factory=list)
	kill_chain_mapping: Dict[str, List[str]] = Field(default_factory=dict)
	
	tactics: List[str] = Field(default_factory=list)
	techniques: List[str] = Field(default_factory=list)
	procedures: List[str] = Field(default_factory=list)
	
	indicators: List[str] = Field(default_factory=list, description="Associated IOC IDs")
	malware_families: List[str] = Field(default_factory=list)
	tools_used: List[str] = Field(default_factory=list)
	
	infrastructure: Dict[str, Any] = Field(default_factory=dict)
	
	victim_count: int = 0
	estimated_impact: Dict[str, Any] = Field(default_factory=dict)
	
	geographical_spread: Dict[str, Any] = Field(default_factory=dict)
	timeline: List[Dict[str, Any]] = Field(default_factory=list)
	
	intelligence_sources: List[str] = Field(default_factory=list)
	
	is_active: bool = True
	severity: ThreatSeverity = ThreatSeverity.MEDIUM
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ThreatIndicator(BaseModel):
	"""Enhanced threat indicator with enrichment"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	indicator_type: str = Field(description="IOC type")
	indicator_value: str = Field(description="IOC value")
	
	description: str = Field(description="Indicator description")
	
	severity: ThreatSeverity = ThreatSeverity.MEDIUM
	confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
	
	first_seen: datetime = Field(default_factory=datetime.utcnow)
	last_seen: datetime = Field(default_factory=datetime.utcnow)
	
	source_feeds: List[str] = Field(default_factory=list)
	intelligence_sources: List[str] = Field(default_factory=list)
	
	attributed_actors: List[str] = Field(default_factory=list)
	associated_campaigns: List[str] = Field(default_factory=list)
	
	malware_families: List[str] = Field(default_factory=list)
	attack_techniques: List[str] = Field(default_factory=list)
	
	kill_chain_phases: List[str] = Field(default_factory=list)
	
	geographical_context: Dict[str, Any] = Field(default_factory=dict)
	
	enrichment_data: Dict[str, Any] = Field(default_factory=dict)
	
	false_positive_probability: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	expiry_date: Optional[datetime] = None
	
	tags: List[str] = Field(default_factory=list)
	
	sightings: int = 0
	last_sighting: Optional[datetime] = None
	
	threat_context: Dict[str, Any] = Field(default_factory=dict)
	
	is_active: bool = True
	is_whitelisted: bool = False
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class IntelligenceEnrichment(BaseModel):
	"""Intelligence enrichment and contextualization"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	target_indicator_id: str = Field(description="Target indicator ID")
	
	enrichment_type: str = Field(description="Type of enrichment")
	enrichment_source: str = Field(description="Enrichment data source")
	
	enrichment_data: Dict[str, Any] = Field(default_factory=dict)
	
	confidence_score: Decimal = Field(ge=0, le=100)
	relevance_score: Decimal = Field(ge=0, le=100)
	
	contextual_information: Dict[str, Any] = Field(default_factory=dict)
	
	geolocation_data: Optional[Dict[str, Any]] = None
	reputation_data: Optional[Dict[str, Any]] = None
	
	malware_analysis: Optional[Dict[str, Any]] = None
	behavioral_analysis: Optional[Dict[str, Any]] = None
	
	attribution_hints: List[str] = Field(default_factory=list)
	
	enrichment_timestamp: datetime = Field(default_factory=datetime.utcnow)
	
	quality_metrics: Dict[str, Any] = Field(default_factory=dict)
	
	is_validated: bool = False
	validation_notes: Optional[str] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class AttributionAnalysis(BaseModel):
	"""Threat attribution analysis results"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	analysis_type: str = Field(description="Attribution analysis type")
	
	target_indicators: List[str] = Field(default_factory=list)
	target_campaigns: List[str] = Field(default_factory=list)
	
	attributed_actors: List[Dict[str, Any]] = Field(default_factory=list)
	
	attribution_confidence: Decimal = Field(ge=0, le=100)
	attribution_methods: List[AttributionMethod] = Field(default_factory=list)
	
	technical_indicators: Dict[str, Any] = Field(default_factory=dict)
	behavioral_patterns: Dict[str, Any] = Field(default_factory=dict)
	
	infrastructure_overlap: Dict[str, Any] = Field(default_factory=dict)
	malware_similarities: Dict[str, Any] = Field(default_factory=dict)
	
	tactical_overlap: List[str] = Field(default_factory=list)
	tool_overlap: List[str] = Field(default_factory=list)
	
	geopolitical_context: Dict[str, Any] = Field(default_factory=dict)
	temporal_analysis: Dict[str, Any] = Field(default_factory=dict)
	
	supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
	contradicting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
	
	alternative_attributions: List[Dict[str, Any]] = Field(default_factory=list)
	
	analyst_notes: Optional[str] = None
	
	analysis_start: datetime = Field(default_factory=datetime.utcnow)
	analysis_end: Optional[datetime] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class IntelligenceAlert(BaseModel):
	"""Intelligence-based security alert"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	alert_type: str = Field(description="Type of intelligence alert")
	title: str = Field(description="Alert title")
	description: str = Field(description="Alert description")
	
	severity: ThreatSeverity = ThreatSeverity.MEDIUM
	confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
	
	triggering_indicators: List[str] = Field(default_factory=list)
	related_campaigns: List[str] = Field(default_factory=list)
	attributed_actors: List[str] = Field(default_factory=list)
	
	threat_context: Dict[str, Any] = Field(default_factory=dict)
	
	recommended_actions: List[str] = Field(default_factory=list)
	
	affected_systems: List[str] = Field(default_factory=list)
	potential_impact: Dict[str, Any] = Field(default_factory=dict)
	
	intelligence_sources: List[str] = Field(default_factory=list)
	
	alert_timestamp: datetime = Field(default_factory=datetime.utcnow)
	
	is_acknowledged: bool = False
	acknowledged_by: Optional[str] = None
	acknowledged_at: Optional[datetime] = None
	
	is_resolved: bool = False
	resolution_notes: Optional[str] = None
	resolved_at: Optional[datetime] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class IntelligenceReport(BaseModel):
	"""Comprehensive intelligence report"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	report_type: str = Field(description="Type of intelligence report")
	title: str = Field(description="Report title")
	
	executive_summary: str = Field(description="Executive summary")
	
	report_period_start: datetime
	report_period_end: datetime
	
	key_findings: List[str] = Field(default_factory=list)
	threat_landscape: Dict[str, Any] = Field(default_factory=dict)
	
	actor_analysis: List[Dict[str, Any]] = Field(default_factory=list)
	campaign_analysis: List[Dict[str, Any]] = Field(default_factory=list)
	
	emerging_threats: List[Dict[str, Any]] = Field(default_factory=list)
	threat_predictions: List[Dict[str, Any]] = Field(default_factory=list)
	
	industry_focus: Optional[str] = None
	geographical_focus: Optional[str] = None
	
	recommendations: List[str] = Field(default_factory=list)
	
	intelligence_sources: List[str] = Field(default_factory=list)
	methodology: Dict[str, Any] = Field(default_factory=dict)
	
	distribution_list: List[str] = Field(default_factory=list)
	classification_level: str = Field(default="unclassified")
	
	generated_by: str = Field(description="Report generator")
	reviewed_by: Optional[str] = None
	approved_by: Optional[str] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class IntelligenceMetrics(BaseModel):
	"""Intelligence operations metrics"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	metric_period_start: datetime
	metric_period_end: datetime
	
	total_feeds_monitored: int = 0
	active_feeds: int = 0
	feed_uptime: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	total_indicators_processed: int = 0
	new_indicators: int = 0
	updated_indicators: int = 0
	expired_indicators: int = 0
	
	enrichment_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	attribution_accuracy: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	threat_actors_tracked: int = 0
	campaigns_identified: int = 0
	
	alerts_generated: int = 0
	false_positive_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	intelligence_quality_score: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	feed_performance: Dict[str, Any] = Field(default_factory=dict)
	
	analyst_productivity: Dict[str, Any] = Field(default_factory=dict)
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None