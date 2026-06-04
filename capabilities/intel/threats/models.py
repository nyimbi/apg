"""Pydantic v2 models for APG Threat Intelligence.

Entities: ThreatActor, ThreatIndicator (IOC), ThreatCampaign, ThreatReport,
MITRE_ATTACKTechnique, KillChainPhase, ThreatFeed, IntelRequirement,
ThreatAssessment, AttributionEvidence.

All models use UUID7 IDs, tenant isolation, soft-delete, and full audit columns.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from uuid6 import uuid7


def uuid7str() -> str:
	"""Generate a UUID7 string ID."""
	return str(uuid7())


# ── Enumerations ───────────────────────────────────────────────────────────────

class ThreatActorType(str, Enum):
	state_actor = "state_actor"
	criminal_group = "criminal_group"
	insider = "insider"
	hacktivist = "hacktivist"
	terrorist_network = "terrorist_network"
	competitor = "competitor"
	unknown = "unknown"


class ThreatActorStatus(str, Enum):
	active = "active"
	dormant = "dormant"
	retired = "retired"
	attributed = "attributed"
	suspected = "suspected"


class IndicatorType(str, Enum):
	ip_address = "ip_address"
	domain = "domain"
	url = "url"
	file_hash_md5 = "file_hash_md5"
	file_hash_sha1 = "file_hash_sha1"
	file_hash_sha256 = "file_hash_sha256"
	email_address = "email_address"
	registry_key = "registry_key"
	mutex = "mutex"
	network_signature = "network_signature"
	certificate = "certificate"
	user_agent = "user_agent"
	yara_rule = "yara_rule"
	financial_signal = "financial_signal"
	behavior = "behavior"
	ioc = "ioc"
	tactic = "tactic"
	technique = "technique"
	procedure = "procedure"
	vulnerability = "vulnerability"
	infrastructure = "infrastructure"
	narrative = "narrative"


class IndicatorStatus(str, Enum):
	active = "active"
	stale = "stale"
	revoked = "revoked"
	under_review = "under_review"
	false_positive = "false_positive"


class CampaignType(str, Enum):
	intrusion_campaign = "intrusion_campaign"
	fraud_campaign = "fraud_campaign"
	disinformation_campaign = "disinformation_campaign"
	physical_threat_campaign = "physical_threat_campaign"
	insider_campaign = "insider_campaign"
	supply_chain_campaign = "supply_chain_campaign"
	ransomware_campaign = "ransomware_campaign"
	espionage_campaign = "espionage_campaign"


class CampaignStatus(str, Enum):
	suspected = "suspected"
	active = "active"
	dormant = "dormant"
	concluded = "concluded"
	attributed = "attributed"


class RiskLevel(str, Enum):
	low = "low"
	medium = "medium"
	high = "high"
	critical = "critical"


class Classification(str, Enum):
	unclassified = "unclassified"
	confidential = "confidential"
	secret = "secret"
	top_secret = "top_secret"


class ReportType(str, Enum):
	brief = "brief"
	advisory = "advisory"
	bulletin = "bulletin"
	estimate = "estimate"
	watchlist = "watchlist"
	situation_report = "situation_report"
	flash_report = "flash_report"
	strategic_assessment = "strategic_assessment"


class ReportStatus(str, Enum):
	draft = "draft"
	under_review = "under_review"
	approved = "approved"
	published = "published"
	retracted = "retracted"


class KillChainPhaseType(str, Enum):
	reconnaissance = "reconnaissance"
	weaponization = "weaponization"
	delivery = "delivery"
	exploitation = "exploitation"
	installation = "installation"
	command_and_control = "command_and_control"
	actions_on_objectives = "actions_on_objectives"


class MitreTactic(str, Enum):
	reconnaissance = "TA0043"
	resource_development = "TA0042"
	initial_access = "TA0001"
	execution = "TA0002"
	persistence = "TA0003"
	privilege_escalation = "TA0004"
	defense_evasion = "TA0005"
	credential_access = "TA0006"
	discovery = "TA0007"
	lateral_movement = "TA0008"
	collection = "TA0009"
	command_and_control = "TA0011"
	exfiltration = "TA0010"
	impact = "TA0040"


class FeedType(str, Enum):
	stix_taxii = "stix_taxii"
	misp = "misp"
	csv = "csv"
	json_api = "json_api"
	osint_scrape = "osint_scrape"
	partner_share = "partner_share"
	internal = "internal"


class FeedStatus(str, Enum):
	active = "active"
	paused = "paused"
	error = "error"
	deprecated = "deprecated"


class AssessmentType(str, Enum):
	threat_profile = "threat_profile"
	risk_assessment = "risk_assessment"
	priority_assessment = "priority_assessment"
	attribution_assessment = "attribution_assessment"
	intent_assessment = "intent_assessment"
	capability_assessment = "capability_assessment"


class RequirementStatus(str, Enum):
	open = "open"
	in_progress = "in_progress"
	satisfied = "satisfied"
	closed = "closed"


class EvidenceType(str, Enum):
	technical_indicator = "technical_indicator"
	behavioural_pattern = "behavioural_pattern"
	infrastructure_overlap = "infrastructure_overlap"
	malware_family = "malware_family"
	ttps_match = "ttps_match"
	victim_profile = "victim_profile"
	geolocation = "geolocation"
	language_artefact = "language_artefact"
	operational_tempo = "operational_tempo"
	sigint = "sigint"
	humint = "humint"


# ── Base model ─────────────────────────────────────────────────────────────────

class TIBase(BaseModel):
	"""Base model for all Threat Intelligence entities."""
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
	)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	is_deleted: bool = False


# ── ThreatActor ────────────────────────────────────────────────────────────────

class ThreatActorCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	name: str
	actor_type: ThreatActorType
	aliases: list[str] = Field(default_factory=list)
	description: str | None = None
	motivation: str | None = None
	sophistication: str | None = None
	first_seen: datetime | None = None
	last_seen: datetime | None = None
	confidence_score: float = Field(ge=0.0, le=1.0)
	country_of_origin: str | None = None
	workspace_id: str | None = None
	evidence_reference: str


class ThreatActorUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	name: str | None = None
	aliases: list[str] | None = None
	description: str | None = None
	motivation: str | None = None
	sophistication: str | None = None
	status: ThreatActorStatus | None = None
	confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
	last_seen: datetime | None = None
	country_of_origin: str | None = None


class ThreatActorResponse(TIBase):
	name: str
	actor_type: ThreatActorType
	status: ThreatActorStatus = ThreatActorStatus.suspected
	aliases: list[str] = Field(default_factory=list)
	description: str | None = None
	motivation: str | None = None
	sophistication: str | None = None
	first_seen: datetime | None = None
	last_seen: datetime | None = None
	confidence_score: float
	country_of_origin: str | None = None
	workspace_id: str | None = None
	evidence_reference: str
	indicator_ids: list[str] = Field(default_factory=list)
	campaign_ids: list[str] = Field(default_factory=list)
	attribution_evidence_ids: list[str] = Field(default_factory=list)


# ── ThreatIndicator ────────────────────────────────────────────────────────────

class ThreatIndicatorCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	indicator_type: IndicatorType
	value: str
	description: str | None = None
	source_id: str
	feed_id: str | None = None
	confidence_score: float = Field(ge=0.0, le=1.0)
	valid_from: datetime = Field(default_factory=datetime.utcnow)
	valid_until: datetime | None = None
	kill_chain_phase_ids: list[str] = Field(default_factory=list)
	mitre_technique_ids: list[str] = Field(default_factory=list)
	actor_ids: list[str] = Field(default_factory=list)
	tags: list[str] = Field(default_factory=list)
	tlp: str = "green"
	evidence_reference: str


class ThreatIndicatorUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	description: str | None = None
	confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
	status: IndicatorStatus | None = None
	valid_until: datetime | None = None
	tags: list[str] | None = None
	tlp: str | None = None


class ThreatIndicatorResponse(TIBase):
	indicator_type: IndicatorType
	value: str
	description: str | None = None
	status: IndicatorStatus = IndicatorStatus.active
	source_id: str
	feed_id: str | None = None
	confidence_score: float
	valid_from: datetime
	valid_until: datetime | None = None
	kill_chain_phase_ids: list[str] = Field(default_factory=list)
	mitre_technique_ids: list[str] = Field(default_factory=list)
	actor_ids: list[str] = Field(default_factory=list)
	tags: list[str] = Field(default_factory=list)
	tlp: str = "green"
	evidence_reference: str
	staleness_score: float = 0.0


# ── ThreatCampaign ─────────────────────────────────────────────────────────────

class ThreatCampaignCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	name: str
	campaign_type: CampaignType
	description: str | None = None
	actor_id: str
	first_seen: datetime | None = None
	last_seen: datetime | None = None
	risk_level: RiskLevel
	classification: Classification = Classification.confidential
	target_sectors: list[str] = Field(default_factory=list)
	target_countries: list[str] = Field(default_factory=list)
	mitre_technique_ids: list[str] = Field(default_factory=list)
	kill_chain_phase_ids: list[str] = Field(default_factory=list)
	indicator_ids: list[str] = Field(default_factory=list)
	evidence_reference: str


class ThreatCampaignUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	name: str | None = None
	description: str | None = None
	status: CampaignStatus | None = None
	risk_level: RiskLevel | None = None
	last_seen: datetime | None = None
	target_sectors: list[str] | None = None
	target_countries: list[str] | None = None


class ThreatCampaignResponse(TIBase):
	name: str
	campaign_type: CampaignType
	status: CampaignStatus = CampaignStatus.suspected
	description: str | None = None
	actor_id: str
	first_seen: datetime | None = None
	last_seen: datetime | None = None
	risk_level: RiskLevel
	classification: Classification
	target_sectors: list[str] = Field(default_factory=list)
	target_countries: list[str] = Field(default_factory=list)
	mitre_technique_ids: list[str] = Field(default_factory=list)
	kill_chain_phase_ids: list[str] = Field(default_factory=list)
	indicator_ids: list[str] = Field(default_factory=list)
	evidence_reference: str


# ── ThreatReport ───────────────────────────────────────────────────────────────

class ThreatReportCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	title: str
	report_type: ReportType
	classification: Classification
	summary: str
	body: str | None = None
	assessment_id: str
	author_id: str
	analyst_ids: list[str] = Field(default_factory=list)
	related_actor_ids: list[str] = Field(default_factory=list)
	related_campaign_ids: list[str] = Field(default_factory=list)
	related_indicator_ids: list[str] = Field(default_factory=list)
	mitre_technique_ids: list[str] = Field(default_factory=list)
	tags: list[str] = Field(default_factory=list)
	tlp: str = "amber"
	approval_reference: str
	evidence_reference: str


class ThreatReportUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	title: str | None = None
	summary: str | None = None
	body: str | None = None
	status: ReportStatus | None = None
	tags: list[str] | None = None
	tlp: str | None = None


class ThreatReportResponse(TIBase):
	title: str
	report_type: ReportType
	status: ReportStatus = ReportStatus.draft
	classification: Classification
	summary: str
	body: str | None = None
	assessment_id: str
	author_id: str
	analyst_ids: list[str] = Field(default_factory=list)
	related_actor_ids: list[str] = Field(default_factory=list)
	related_campaign_ids: list[str] = Field(default_factory=list)
	related_indicator_ids: list[str] = Field(default_factory=list)
	mitre_technique_ids: list[str] = Field(default_factory=list)
	tags: list[str] = Field(default_factory=list)
	tlp: str = "amber"
	approval_reference: str
	evidence_reference: str
	published_at: datetime | None = None


# ── MITRE ATT&CK Technique ─────────────────────────────────────────────────────

class MITRETechniqueCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	technique_id: str  # e.g. T1059.001
	name: str
	tactic: MitreTactic
	description: str | None = None
	platforms: list[str] = Field(default_factory=list)
	data_sources: list[str] = Field(default_factory=list)
	detection_guidance: str | None = None
	mitigations: list[str] = Field(default_factory=list)
	sub_techniques: list[str] = Field(default_factory=list)
	url: str | None = None


class MITRETechniqueResponse(TIBase):
	technique_id: str
	name: str
	tactic: MitreTactic
	description: str | None = None
	platforms: list[str] = Field(default_factory=list)
	data_sources: list[str] = Field(default_factory=list)
	detection_guidance: str | None = None
	mitigations: list[str] = Field(default_factory=list)
	sub_techniques: list[str] = Field(default_factory=list)
	url: str | None = None


# ── KillChainPhase ─────────────────────────────────────────────────────────────

class KillChainPhaseCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	phase_name: KillChainPhaseType
	kill_chain_name: str = "lockheed-martin-cyber-kill-chain"
	description: str | None = None
	order: int = 0


class KillChainPhaseResponse(TIBase):
	phase_name: KillChainPhaseType
	kill_chain_name: str
	description: str | None = None
	order: int


# ── ThreatFeed ─────────────────────────────────────────────────────────────────

class ThreatFeedCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	name: str
	feed_type: FeedType
	url: str | None = None
	api_key: str | None = None
	collection_id: str | None = None  # TAXII collection
	poll_interval_seconds: int = 3600
	confidence_weight: float = Field(default=0.8, ge=0.0, le=1.0)
	tlp_filter: str | None = None
	indicator_types: list[str] = Field(default_factory=list)
	description: str | None = None
	custodian_id: str
	evidence_reference: str


class ThreatFeedUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	name: str | None = None
	url: str | None = None
	api_key: str | None = None
	poll_interval_seconds: int | None = None
	confidence_weight: float | None = Field(default=None, ge=0.0, le=1.0)
	status: FeedStatus | None = None


class ThreatFeedResponse(TIBase):
	name: str
	feed_type: FeedType
	status: FeedStatus = FeedStatus.active
	url: str | None = None
	collection_id: str | None = None
	poll_interval_seconds: int
	confidence_weight: float
	tlp_filter: str | None = None
	indicator_types: list[str] = Field(default_factory=list)
	description: str | None = None
	custodian_id: str
	evidence_reference: str
	last_polled_at: datetime | None = None
	indicators_ingested: int = 0


# ── IntelRequirement ───────────────────────────────────────────────────────────

class IntelRequirementCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	title: str
	description: str
	requestor_id: str
	priority: RiskLevel
	due_date: datetime | None = None
	related_actor_ids: list[str] = Field(default_factory=list)
	related_campaign_ids: list[str] = Field(default_factory=list)
	tags: list[str] = Field(default_factory=list)


class IntelRequirementUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	title: str | None = None
	description: str | None = None
	status: RequirementStatus | None = None
	priority: RiskLevel | None = None
	due_date: datetime | None = None
	assigned_analyst_id: str | None = None


class IntelRequirementResponse(TIBase):
	title: str
	description: str
	status: RequirementStatus = RequirementStatus.open
	requestor_id: str
	priority: RiskLevel
	due_date: datetime | None = None
	assigned_analyst_id: str | None = None
	related_actor_ids: list[str] = Field(default_factory=list)
	related_campaign_ids: list[str] = Field(default_factory=list)
	tags: list[str] = Field(default_factory=list)
	satisfying_report_ids: list[str] = Field(default_factory=list)


# ── ThreatAssessment ───────────────────────────────────────────────────────────

class ThreatAssessmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	assessment_type: AssessmentType
	campaign_id: str
	analyst_id: str
	risk_level: RiskLevel
	confidence_score: float = Field(ge=0.0, le=1.0)
	summary: str
	findings: list[str] = Field(default_factory=list)
	recommendations: list[str] = Field(default_factory=list)
	mitre_technique_ids: list[str] = Field(default_factory=list)
	evidence_reference: str


class ThreatAssessmentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	risk_level: RiskLevel | None = None
	confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
	summary: str | None = None
	findings: list[str] | None = None
	recommendations: list[str] | None = None


class ThreatAssessmentResponse(TIBase):
	assessment_type: AssessmentType
	campaign_id: str
	analyst_id: str
	risk_level: RiskLevel
	confidence_score: float
	summary: str
	findings: list[str] = Field(default_factory=list)
	recommendations: list[str] = Field(default_factory=list)
	mitre_technique_ids: list[str] = Field(default_factory=list)
	evidence_reference: str
	approved_by: str | None = None
	approved_at: datetime | None = None


# ── AttributionEvidence ────────────────────────────────────────────────────────

class AttributionEvidenceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	actor_id: str
	evidence_type: EvidenceType
	description: str
	confidence_score: float = Field(ge=0.0, le=1.0)
	source_id: str | None = None
	indicator_ids: list[str] = Field(default_factory=list)
	collection_date: datetime | None = None
	classification: Classification = Classification.confidential
	analyst_id: str
	raw_reference: str | None = None


class AttributionEvidenceUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	description: str | None = None
	confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
	classification: Classification | None = None


class AttributionEvidenceResponse(TIBase):
	actor_id: str
	evidence_type: EvidenceType
	description: str
	confidence_score: float
	source_id: str | None = None
	indicator_ids: list[str] = Field(default_factory=list)
	collection_date: datetime | None = None
	classification: Classification
	analyst_id: str
	raw_reference: str | None = None


# ── Aggregation / Report models ────────────────────────────────────────────────

class ThreatDashboardReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	generated_at: datetime = Field(default_factory=datetime.utcnow)
	actor_count: int = 0
	active_actor_count: int = 0
	indicator_count: int = 0
	active_indicator_count: int = 0
	stale_indicator_count: int = 0
	campaign_count: int = 0
	active_campaign_count: int = 0
	assessment_count: int = 0
	report_count: int = 0
	feed_count: int = 0
	requirement_count: int = 0
	open_requirement_count: int = 0
	mitre_technique_count: int = 0
	critical_actors: list[dict[str, Any]] = Field(default_factory=list)
	critical_campaigns: list[dict[str, Any]] = Field(default_factory=list)
	recent_indicators: list[dict[str, Any]] = Field(default_factory=list)


class CorrelationResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	generated_at: datetime = Field(default_factory=datetime.utcnow)
	indicator_clusters: list[dict[str, Any]] = Field(default_factory=list)
	actor_links: list[dict[str, Any]] = Field(default_factory=list)
	campaign_links: list[dict[str, Any]] = Field(default_factory=list)
	confidence_weighted_score: float = 0.0
	correlation_count: int = 0


class MitreHeatmap(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	generated_at: datetime = Field(default_factory=datetime.utcnow)
	tactic_coverage: dict[str, int] = Field(default_factory=dict)
	technique_hits: list[dict[str, Any]] = Field(default_factory=list)
	total_techniques: int = 0


class ConfidenceScoreBreakdown(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	entity_id: str
	entity_type: str
	raw_score: float
	source_reliability: float
	recency_decay: float
	corroboration_bonus: float
	final_score: float
	factors: dict[str, float] = Field(default_factory=dict)


class STIXBundle(BaseModel):
	"""Inbound STIX 2.1 bundle for ingestion."""
	model_config = ConfigDict(extra="allow", validate_by_name=True, validate_by_alias=True)

	type: str = "bundle"
	id: str
	spec_version: str = "2.1"
	objects: list[dict[str, Any]] = Field(default_factory=list)


class TAXIIShareRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	taxii_server_url: str
	collection_id: str
	api_root: str
	indicator_ids: list[str] = Field(default_factory=list)
	campaign_ids: list[str] = Field(default_factory=list)
	actor_ids: list[str] = Field(default_factory=list)
	tlp_max: str = "green"
	classification_max: Classification = Classification.unclassified


class MISPExportRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	indicator_ids: list[str] = Field(default_factory=list)
	campaign_ids: list[str] = Field(default_factory=list)
	include_attributes: bool = True
	include_galaxy: bool = False
	distribution: int = 0  # 0=org, 1=community, 2=connected, 3=all


class StalenessSweepResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	swept_at: datetime = Field(default_factory=datetime.utcnow)
	reviewed: int = 0
	marked_stale: int = 0
	revoked: int = 0
	still_active: int = 0
	stale_ids: list[str] = Field(default_factory=list)
	revoked_ids: list[str] = Field(default_factory=list)


class FeedIngestResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	feed_id: str
	tenant_id: str
	ingested_at: datetime = Field(default_factory=datetime.utcnow)
	total_objects: int = 0
	indicators_created: int = 0
	actors_created: int = 0
	campaigns_created: int = 0
	skipped_stale: int = 0
	skipped_duplicate: int = 0
	errors: list[str] = Field(default_factory=list)


class ThreatReportGenerationRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	created_by: str
	assessment_id: str
	report_type: ReportType
	classification: Classification
	title: str
	tlp: str = "amber"
	include_indicators: bool = True
	include_mitre_heatmap: bool = True
	include_attribution: bool = True
