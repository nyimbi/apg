"""Pydantic v2 models for APG Open Source Intelligence (OSINT).

Entity prefix: OS (e.g. OSSource, OSTask, OSRawIntel, ...)

Lifecycle:
    OSINTSource       — registered sources (web, social, darkweb, etc.)
    CollectionTask    — scheduled/ad-hoc collection job against a source
    RawIntelligence   — unprocessed artefact harvested from a task
    ProcessedIntelligence — analyst-refined intelligence item
    OSEntity          — extracted person / org / location / object
    EntityRelationship — directed relationship between two entities
    SocialMediaProfile — social profile associated with an entity
    WebContent        — scraped web page or document
    DomainRecord      — WHOIS / DNS / certificate data for a domain
    IPIntelligence    — geolocation + ASN + threat metadata for an IP
    DocumentAnalysis  — NLP analysis output for a document
    CredibilityScore  — source/item credibility assessment
    DisseminationPackage — sanitised release package
    OSINTReview        — quality/compliance review record
    OSINTAgent         — registered autonomous collection/analysis agent
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


def _now() -> datetime:
	return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
	WEB = "web"
	SOCIAL_MEDIA = "social_media"
	DARKWEB = "darkweb"
	NEWS = "news"
	FORUM = "forum"
	DOCUMENT = "document"
	REGISTRY = "registry"
	BROADCAST = "broadcast"
	DATASET = "dataset"
	API_FEED = "api_feed"
	RSS_FEED = "rss_feed"
	PASTE_SITE = "paste_site"
	CODE_REPOSITORY = "code_repository"
	IOT_SCAN = "iot_scan"


class SourceStatus(str, Enum):
	ACTIVE = "active"
	INACTIVE = "inactive"
	SUSPENDED = "suspended"
	UNDER_REVIEW = "under_review"
	DECOMMISSIONED = "decommissioned"


class RiskTier(str, Enum):
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


class TaskStatus(str, Enum):
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	RETRYING = "retrying"


class TaskType(str, Enum):
	WEB_SCRAPE = "web_scrape"
	SOCIAL_MONITOR = "social_monitor"
	DOMAIN_INTEL = "domain_intel"
	IP_GEOLOCATION = "ip_geolocation"
	ENTITY_EXTRACTION = "entity_extraction"
	DOCUMENT_ANALYSIS = "document_analysis"
	RELATIONSHIP_MAPPING = "relationship_mapping"
	DARK_WEB_CRAWL = "dark_web_crawl"
	API_COLLECTION = "api_collection"
	DEDUPLICATION = "deduplication"
	CREDIBILITY_SCORE = "credibility_score"


class CollectionMethod(str, Enum):
	CRAWLER = "crawler"
	API_FEED = "api_feed"
	RSS_FEED = "rss_feed"
	MANUAL_UPLOAD = "manual_upload"
	PARTNER_FEED = "partner_feed"
	WEBHOOK = "webhook"
	HEADLESS_BROWSER = "headless_browser"


class IntelStatus(str, Enum):
	RAW = "raw"
	TRIAGED = "triaged"
	PROCESSED = "processed"
	VERIFIED = "verified"
	REJECTED = "rejected"
	ARCHIVED = "archived"
	DISSEMINATED = "disseminated"


class EntityType(str, Enum):
	PERSON = "person"
	ORGANIZATION = "organization"
	LOCATION = "location"
	OBJECT = "object"
	EVENT = "event"
	FACILITY = "facility"
	VESSEL = "vessel"
	AIRCRAFT = "aircraft"
	VEHICLE = "vehicle"
	DOMAIN = "domain"
	IP_ADDRESS = "ip_address"
	EMAIL = "email"
	PHONE = "phone"
	CRYPTOCURRENCY_WALLET = "cryptocurrency_wallet"
	USERNAME = "username"


class RelationshipType(str, Enum):
	AFFILIATED_WITH = "affiliated_with"
	OWNS = "owns"
	OPERATES = "operates"
	LOCATED_AT = "located_at"
	COMMUNICATES_WITH = "communicates_with"
	MEMBER_OF = "member_of"
	FUNDS = "funds"
	DIRECTS = "directs"
	KNOWN_ALIAS = "known_alias"
	EMPLOYS = "employs"
	ASSOCIATED_WITH = "associated_with"
	TARGETS = "targets"
	LINKED_TO = "linked_to"


class ConfidenceLevel(str, Enum):
	UNCONFIRMED = "unconfirmed"
	POSSIBLE = "possible"
	PROBABLE = "probable"
	CONFIRMED = "confirmed"


class ClassificationLevel(str, Enum):
	UNCLASSIFIED = "unclassified"
	CONFIDENTIAL = "confidential"
	SECRET = "secret"
	TOP_SECRET = "top_secret"


class TLPLevel(str, Enum):
	CLEAR = "clear"
	GREEN = "green"
	AMBER = "amber"
	AMBER_STRICT = "amber_strict"
	RED = "red"


class Priority(str, Enum):
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


class TriageDecision(str, Enum):
	RELEVANT = "relevant"
	IRRELEVANT = "irrelevant"
	DUPLICATE = "duplicate"
	NEEDS_REVIEW = "needs_review"
	ESCALATED = "escalated"


class AssessmentType(str, Enum):
	THREAT = "threat"
	OPPORTUNITY = "opportunity"
	ENTITY_PROFILE = "entity_profile"
	EVENT_SUMMARY = "event_summary"
	TREND = "trend"
	WATCHLIST = "watchlist"
	NETWORK_MAP = "network_map"
	GEOSPATIAL = "geospatial"


class ReviewStatus(str, Enum):
	APPROVED = "approved"
	REJECTED = "rejected"
	NEEDS_CHANGES = "needs_changes"
	ESCALATED = "escalated"


class AgentRuntime(str, Enum):
	CODEX = "codex"
	CLAUDE_CODE = "claude_code"
	OPENCODE = "opencode"
	PI = "pi"


class AgentRole(str, Enum):
	SOURCE_SCOUT = "source_scout"
	COLLECTION_PLANNER = "collection_planner"
	EVIDENCE_TRIAGE = "evidence_triage"
	ENTITY_EXTRACTOR = "entity_extractor"
	RELATIONSHIP_MAPPER = "relationship_mapper"
	DEDUPLICATOR = "deduplicator"
	CREDIBILITY_ANALYST = "credibility_analyst"
	DISSEMINATION_REVIEWER = "dissemination_reviewer"
	WATCHLIST_MONITOR = "watchlist_monitor"


# ---------------------------------------------------------------------------
# Base model — shared audit fields
# ---------------------------------------------------------------------------

class OSINTBase(BaseModel):
	"""Common audit fields for every OSINT entity."""

	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
	)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: datetime = Field(default_factory=_now)
	updated_at: datetime = Field(default_factory=_now)
	created_by: str
	is_deleted: bool = False


# ---------------------------------------------------------------------------
# OSINTSource — registered intelligence sources
# ---------------------------------------------------------------------------

class OSINTSourceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	name: str
	source_type: SourceType
	url: str | None = None
	description: str | None = None
	owner_id: str
	terms_review_reference: str
	risk_tier: RiskTier
	collection_method: CollectionMethod
	requires_auth: bool = False
	auth_reference: str | None = None
	rate_limit_rps: float | None = None
	credibility_baseline: float = Field(default=0.5, ge=0.0, le=1.0)
	tags: list[str] = Field(default_factory=list)
	evidence_reference: str


class OSINTSourceUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	name: str | None = None
	description: str | None = None
	status: SourceStatus | None = None
	risk_tier: RiskTier | None = None
	credibility_baseline: float | None = Field(default=None, ge=0.0, le=1.0)
	tags: list[str] | None = None
	evidence_reference: str | None = None


class OSINTSourceResponse(OSINTBase):
	name: str
	source_type: SourceType
	url: str | None = None
	description: str | None = None
	owner_id: str
	terms_review_reference: str
	risk_tier: RiskTier
	collection_method: CollectionMethod
	status: SourceStatus = SourceStatus.ACTIVE
	requires_auth: bool = False
	auth_reference: str | None = None
	rate_limit_rps: float | None = None
	credibility_baseline: float = 0.5
	tags: list[str] = Field(default_factory=list)
	evidence_reference: str
	last_collected_at: datetime | None = None
	total_items_collected: int = 0


# ---------------------------------------------------------------------------
# CollectionTask
# ---------------------------------------------------------------------------

class CollectionTaskCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	source_id: str
	task_type: TaskType
	parameters: dict[str, Any] = Field(default_factory=dict)
	priority: Priority = Priority.MEDIUM
	scheduled_at: datetime | None = None
	max_depth: int = Field(default=2, ge=1, le=10)
	max_items: int | None = None
	keywords: list[str] = Field(default_factory=list)
	approval_reference: str | None = None
	evidence_reference: str


class CollectionTaskUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	status: TaskStatus | None = None
	parameters: dict[str, Any] | None = None
	error_message: str | None = None
	items_collected: int | None = None


class CollectionTaskResponse(OSINTBase):
	source_id: str
	task_type: TaskType
	status: TaskStatus = TaskStatus.PENDING
	parameters: dict[str, Any] = Field(default_factory=dict)
	priority: Priority = Priority.MEDIUM
	scheduled_at: datetime | None = None
	started_at: datetime | None = None
	completed_at: datetime | None = None
	max_depth: int = 2
	max_items: int | None = None
	keywords: list[str] = Field(default_factory=list)
	items_collected: int = 0
	error_message: str | None = None
	approval_reference: str | None = None
	evidence_reference: str


# ---------------------------------------------------------------------------
# RawIntelligence
# ---------------------------------------------------------------------------

class RawIntelligenceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	task_id: str
	source_id: str
	content_reference: str
	content_type: str
	raw_content: str | None = None
	url: str | None = None
	fingerprint: str
	confidence_score: float = Field(ge=0.0, le=1.0)
	language: str | None = None
	captured_at: datetime = Field(default_factory=_now)
	evidence_reference: str


class RawIntelligenceUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	status: IntelStatus | None = None
	triage_decision: TriageDecision | None = None
	analyst_id: str | None = None
	notes: str | None = None


class RawIntelligenceResponse(OSINTBase):
	task_id: str
	source_id: str
	content_reference: str
	content_type: str
	url: str | None = None
	fingerprint: str
	confidence_score: float
	language: str | None = None
	captured_at: datetime
	status: IntelStatus = IntelStatus.RAW
	triage_decision: TriageDecision | None = None
	analyst_id: str | None = None
	notes: str | None = None
	evidence_reference: str


# ---------------------------------------------------------------------------
# ProcessedIntelligence
# ---------------------------------------------------------------------------

class ProcessedIntelligenceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	raw_intel_id: str
	requirement_id: str | None = None
	assessment_type: AssessmentType
	summary: str
	key_findings: list[str] = Field(default_factory=list)
	confidence_score: float = Field(ge=0.0, le=1.0)
	confidence_level: ConfidenceLevel = ConfidenceLevel.POSSIBLE
	classification: ClassificationLevel = ClassificationLevel.UNCLASSIFIED
	tlp: TLPLevel = TLPLevel.AMBER
	analyst_id: str
	tags: list[str] = Field(default_factory=list)
	evidence_reference: str


class ProcessedIntelligenceUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	summary: str | None = None
	key_findings: list[str] | None = None
	confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
	confidence_level: ConfidenceLevel | None = None
	status: IntelStatus | None = None
	tags: list[str] | None = None
	evidence_reference: str | None = None


class ProcessedIntelligenceResponse(OSINTBase):
	raw_intel_id: str
	requirement_id: str | None = None
	assessment_type: AssessmentType
	summary: str
	key_findings: list[str] = Field(default_factory=list)
	confidence_score: float
	confidence_level: ConfidenceLevel
	classification: ClassificationLevel
	tlp: TLPLevel
	status: IntelStatus = IntelStatus.PROCESSED
	analyst_id: str
	tags: list[str] = Field(default_factory=list)
	evidence_reference: str
	entity_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# OSEntity — extracted entity (person, org, location, object, etc.)
# ---------------------------------------------------------------------------

class OSEntityCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	entity_type: EntityType
	name: str
	aliases: list[str] = Field(default_factory=list)
	description: str | None = None
	attributes: dict[str, Any] = Field(default_factory=dict)
	confidence_score: float = Field(ge=0.0, le=1.0)
	confidence_level: ConfidenceLevel = ConfidenceLevel.POSSIBLE
	classification: ClassificationLevel = ClassificationLevel.UNCLASSIFIED
	source_intel_ids: list[str] = Field(default_factory=list)
	tags: list[str] = Field(default_factory=list)
	evidence_reference: str


class OSEntityUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	name: str | None = None
	aliases: list[str] | None = None
	description: str | None = None
	attributes: dict[str, Any] | None = None
	confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
	confidence_level: ConfidenceLevel | None = None
	tags: list[str] | None = None
	evidence_reference: str | None = None


class OSEntityResponse(OSINTBase):
	entity_type: EntityType
	name: str
	aliases: list[str] = Field(default_factory=list)
	description: str | None = None
	attributes: dict[str, Any] = Field(default_factory=dict)
	confidence_score: float
	confidence_level: ConfidenceLevel
	classification: ClassificationLevel
	source_intel_ids: list[str] = Field(default_factory=list)
	tags: list[str] = Field(default_factory=list)
	evidence_reference: str
	relationship_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# EntityRelationship
# ---------------------------------------------------------------------------

class EntityRelationshipCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	source_entity_id: str
	target_entity_id: str
	relationship_type: RelationshipType
	description: str | None = None
	strength: float = Field(default=0.5, ge=0.0, le=1.0)
	confidence_score: float = Field(ge=0.0, le=1.0)
	first_seen: datetime | None = None
	last_seen: datetime | None = None
	attributes: dict[str, Any] = Field(default_factory=dict)
	evidence_reference: str


class EntityRelationshipUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	description: str | None = None
	strength: float | None = Field(default=None, ge=0.0, le=1.0)
	confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
	last_seen: datetime | None = None
	attributes: dict[str, Any] | None = None
	evidence_reference: str | None = None


class EntityRelationshipResponse(OSINTBase):
	source_entity_id: str
	target_entity_id: str
	relationship_type: RelationshipType
	description: str | None = None
	strength: float
	confidence_score: float
	first_seen: datetime | None = None
	last_seen: datetime | None = None
	attributes: dict[str, Any] = Field(default_factory=dict)
	evidence_reference: str


# ---------------------------------------------------------------------------
# SocialMediaProfile
# ---------------------------------------------------------------------------

class SocialMediaProfileCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	entity_id: str | None = None
	platform: str
	handle: str
	profile_url: str | None = None
	display_name: str | None = None
	bio: str | None = None
	followers_count: int | None = None
	following_count: int | None = None
	post_count: int | None = None
	verified: bool = False
	is_active: bool = True
	created_platform_at: datetime | None = None
	attributes: dict[str, Any] = Field(default_factory=dict)
	keywords_monitored: list[str] = Field(default_factory=list)
	evidence_reference: str


class SocialMediaProfileUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	display_name: str | None = None
	bio: str | None = None
	followers_count: int | None = None
	following_count: int | None = None
	post_count: int | None = None
	is_active: bool | None = None
	attributes: dict[str, Any] | None = None
	keywords_monitored: list[str] | None = None


class SocialMediaProfileResponse(OSINTBase):
	entity_id: str | None = None
	platform: str
	handle: str
	profile_url: str | None = None
	display_name: str | None = None
	bio: str | None = None
	followers_count: int | None = None
	following_count: int | None = None
	post_count: int | None = None
	verified: bool = False
	is_active: bool = True
	created_platform_at: datetime | None = None
	attributes: dict[str, Any] = Field(default_factory=dict)
	keywords_monitored: list[str] = Field(default_factory=list)
	evidence_reference: str
	last_scraped_at: datetime | None = None


# ---------------------------------------------------------------------------
# WebContent
# ---------------------------------------------------------------------------

class WebContentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	task_id: str
	url: str
	title: str | None = None
	content_hash: str
	content_reference: str
	mime_type: str = "text/html"
	language: str | None = None
	depth: int = Field(default=0, ge=0)
	parent_url: str | None = None
	links_extracted: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)
	scraped_at: datetime = Field(default_factory=_now)
	evidence_reference: str


class WebContentResponse(OSINTBase):
	task_id: str
	url: str
	title: str | None = None
	content_hash: str
	content_reference: str
	mime_type: str
	language: str | None = None
	depth: int
	parent_url: str | None = None
	links_extracted: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)
	scraped_at: datetime
	evidence_reference: str


# ---------------------------------------------------------------------------
# DomainRecord — WHOIS / DNS / certificate intelligence
# ---------------------------------------------------------------------------

class DomainRecordCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	domain: str
	registrar: str | None = None
	registrant_name: str | None = None
	registrant_email: str | None = None
	registrant_org: str | None = None
	registrant_country: str | None = None
	created_date: datetime | None = None
	updated_date: datetime | None = None
	expiry_date: datetime | None = None
	name_servers: list[str] = Field(default_factory=list)
	a_records: list[str] = Field(default_factory=list)
	mx_records: list[str] = Field(default_factory=list)
	txt_records: list[str] = Field(default_factory=list)
	ssl_issuer: str | None = None
	ssl_expiry: datetime | None = None
	ssl_san: list[str] = Field(default_factory=list)
	raw_whois: str | None = None
	attributes: dict[str, Any] = Field(default_factory=dict)
	queried_at: datetime = Field(default_factory=_now)
	evidence_reference: str


class DomainRecordResponse(OSINTBase):
	domain: str
	registrar: str | None = None
	registrant_name: str | None = None
	registrant_email: str | None = None
	registrant_org: str | None = None
	registrant_country: str | None = None
	created_date: datetime | None = None
	updated_date: datetime | None = None
	expiry_date: datetime | None = None
	name_servers: list[str] = Field(default_factory=list)
	a_records: list[str] = Field(default_factory=list)
	mx_records: list[str] = Field(default_factory=list)
	txt_records: list[str] = Field(default_factory=list)
	ssl_issuer: str | None = None
	ssl_expiry: datetime | None = None
	ssl_san: list[str] = Field(default_factory=list)
	raw_whois: str | None = None
	attributes: dict[str, Any] = Field(default_factory=dict)
	queried_at: datetime
	evidence_reference: str


# ---------------------------------------------------------------------------
# IPIntelligence
# ---------------------------------------------------------------------------

class IPIntelligenceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	ip_address: str
	ip_version: int = Field(default=4, ge=4, le=6)
	asn: str | None = None
	asn_org: str | None = None
	isp: str | None = None
	country_code: str | None = None
	country_name: str | None = None
	region: str | None = None
	city: str | None = None
	latitude: float | None = None
	longitude: float | None = None
	is_tor: bool = False
	is_vpn: bool = False
	is_proxy: bool = False
	is_datacenter: bool = False
	abuse_confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
	threat_types: list[str] = Field(default_factory=list)
	open_ports: list[int] = Field(default_factory=list)
	reverse_dns: str | None = None
	attributes: dict[str, Any] = Field(default_factory=dict)
	queried_at: datetime = Field(default_factory=_now)
	evidence_reference: str


class IPIntelligenceResponse(OSINTBase):
	ip_address: str
	ip_version: int
	asn: str | None = None
	asn_org: str | None = None
	isp: str | None = None
	country_code: str | None = None
	country_name: str | None = None
	region: str | None = None
	city: str | None = None
	latitude: float | None = None
	longitude: float | None = None
	is_tor: bool
	is_vpn: bool
	is_proxy: bool
	is_datacenter: bool
	abuse_confidence_score: float
	threat_types: list[str] = Field(default_factory=list)
	open_ports: list[int] = Field(default_factory=list)
	reverse_dns: str | None = None
	attributes: dict[str, Any] = Field(default_factory=dict)
	queried_at: datetime
	evidence_reference: str


# ---------------------------------------------------------------------------
# DocumentAnalysis — NLP output
# ---------------------------------------------------------------------------

class DocumentAnalysisCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	raw_intel_id: str
	language: str | None = None
	sentiment_score: float | None = Field(default=None, ge=-1.0, le=1.0)
	entities_extracted: list[dict[str, Any]] = Field(default_factory=list)
	keywords: list[str] = Field(default_factory=list)
	topics: list[str] = Field(default_factory=list)
	summary: str | None = None
	threat_indicators: list[str] = Field(default_factory=list)
	location_mentions: list[dict[str, Any]] = Field(default_factory=list)
	person_mentions: list[str] = Field(default_factory=list)
	org_mentions: list[str] = Field(default_factory=list)
	date_mentions: list[str] = Field(default_factory=list)
	model_used: str | None = None
	processing_time_ms: int | None = None
	evidence_reference: str


class DocumentAnalysisResponse(OSINTBase):
	raw_intel_id: str
	language: str | None = None
	sentiment_score: float | None = None
	entities_extracted: list[dict[str, Any]] = Field(default_factory=list)
	keywords: list[str] = Field(default_factory=list)
	topics: list[str] = Field(default_factory=list)
	summary: str | None = None
	threat_indicators: list[str] = Field(default_factory=list)
	location_mentions: list[dict[str, Any]] = Field(default_factory=list)
	person_mentions: list[str] = Field(default_factory=list)
	org_mentions: list[str] = Field(default_factory=list)
	date_mentions: list[str] = Field(default_factory=list)
	model_used: str | None = None
	processing_time_ms: int | None = None
	evidence_reference: str


# ---------------------------------------------------------------------------
# CredibilityScore
# ---------------------------------------------------------------------------

class CredibilityScoreCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	reference_id: str
	reference_type: str  # "source" | "raw_intel" | "processed_intel"
	score: float = Field(ge=0.0, le=1.0)
	factors: dict[str, float] = Field(default_factory=dict)
	analyst_id: str
	rationale: str | None = None
	evidence_reference: str


class CredibilityScoreResponse(OSINTBase):
	reference_id: str
	reference_type: str
	score: float
	factors: dict[str, float] = Field(default_factory=dict)
	analyst_id: str
	rationale: str | None = None
	evidence_reference: str


# ---------------------------------------------------------------------------
# DisseminationPackage
# ---------------------------------------------------------------------------

class DisseminationPackageCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	processed_intel_ids: list[str]
	audience: str
	release_marking: TLPLevel
	classification: ClassificationLevel = ClassificationLevel.UNCLASSIFIED
	title: str
	executive_summary: str
	approval_reference: str
	evidence_reference: str


class DisseminationPackageUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	title: str | None = None
	executive_summary: str | None = None
	approval_reference: str | None = None
	evidence_reference: str | None = None


class DisseminationPackageResponse(OSINTBase):
	processed_intel_ids: list[str]
	audience: str
	release_marking: TLPLevel
	classification: ClassificationLevel
	title: str
	executive_summary: str
	approval_reference: str
	evidence_reference: str
	disseminated_at: datetime | None = None


# ---------------------------------------------------------------------------
# OSINTReview
# ---------------------------------------------------------------------------

class OSINTReviewCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	reference_id: str
	reference_type: str
	reviewer_id: str
	status: ReviewStatus
	notes: str | None = None
	evidence_reference: str


class OSINTReviewResponse(OSINTBase):
	reference_id: str
	reference_type: str
	reviewer_id: str
	status: ReviewStatus
	notes: str | None = None
	evidence_reference: str


# ---------------------------------------------------------------------------
# OSINTAgent
# ---------------------------------------------------------------------------

class OSINTAgentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	name: str
	runtime: AgentRuntime
	role: AgentRole
	scope: str
	capabilities: list[str] = Field(default_factory=list)


class OSINTAgentResponse(OSINTBase):
	name: str
	runtime: AgentRuntime
	role: AgentRole
	scope: str
	capabilities: list[str] = Field(default_factory=list)
	is_active: bool = True


# ---------------------------------------------------------------------------
# Report / Aggregation models
# ---------------------------------------------------------------------------

class OSINTDashboard(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	source_count: int = 0
	active_source_count: int = 0
	high_risk_source_count: int = 0
	task_count: int = 0
	pending_task_count: int = 0
	running_task_count: int = 0
	raw_intel_count: int = 0
	processed_intel_count: int = 0
	entity_count: int = 0
	relationship_count: int = 0
	social_profile_count: int = 0
	domain_record_count: int = 0
	ip_intel_count: int = 0
	document_analysis_count: int = 0
	dissemination_count: int = 0
	review_count: int = 0
	agent_count: int = 0
	audit_event_count: int = 0


class EntityNetworkReport(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	entity_count: int
	relationship_count: int
	entities: list[dict[str, Any]] = Field(default_factory=list)
	relationships: list[dict[str, Any]] = Field(default_factory=list)
	clusters: list[list[str]] = Field(default_factory=list)
	high_confidence_links: int = 0


class SourceHealthReport(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	total_sources: int
	active_sources: int
	sources_by_type: dict[str, int] = Field(default_factory=dict)
	sources_by_risk: dict[str, int] = Field(default_factory=dict)
	avg_credibility: float = 0.0
	top_sources: list[dict[str, Any]] = Field(default_factory=list)


class ThreatLandscapeReport(BaseModel):
	model_config = ConfigDict(extra="forbid")
	tenant_id: str
	generated_at: datetime = Field(default_factory=_now)
	total_threats: int = 0
	critical_threats: int = 0
	high_threats: int = 0
	top_threat_actors: list[str] = Field(default_factory=list)
	top_targeted_sectors: list[str] = Field(default_factory=list)
	geographic_distribution: dict[str, int] = Field(default_factory=dict)
	trend_indicators: list[str] = Field(default_factory=list)
