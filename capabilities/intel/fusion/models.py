"""Pydantic v2 models for APG Intelligence Fusion.

Lifecycle: IntelligenceItem → FusionWorkspace → CorrelationSet →
AssessmentPicture → IntelligenceProduct → AnalyticalJudgement → Evidence →
HypothesisTest, plus aggregation/report models and domain events.

© 2025 Datacraft — Nyimbi Odero
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	from uuid import uuid4

	def uuid7str() -> str:
		return str(uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# Validators
# ─────────────────────────────────────────────────────────────────────────────

def _non_empty(v: str) -> str:
	assert v and v.strip(), "Field must not be empty"
	return v.strip()


def _confidence_range(v: float) -> float:
	assert 0.0 <= v <= 1.0, "Confidence must be in [0.0, 1.0]"
	return v


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]
ConfidenceScore = Annotated[float, AfterValidator(_confidence_range)]


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class TLPLevel(str, Enum):
	"""Traffic Light Protocol marking levels."""
	WHITE  = "TLP:WHITE"
	GREEN  = "TLP:GREEN"
	AMBER  = "TLP:AMBER"
	RED    = "TLP:RED"
	CLEAR  = "TLP:CLEAR"


class ClassificationLevel(str, Enum):
	UNCLASSIFIED = "unclassified"
	CONFIDENTIAL = "confidential"
	SECRET       = "secret"
	TOP_SECRET   = "top_secret"


class SourceType(str, Enum):
	OSINT       = "osint"
	SIGINT      = "sigint"
	HUMINT      = "humint"
	GEOINT      = "geoint"
	CYBINT      = "cybint"
	FININT      = "finint"
	SOCINT      = "socint"
	DARKWEB     = "darkweb"
	RADIO       = "radio"
	MONITORING  = "monitoring"
	PARTNER_RPT = "partner_report"


class IntelItemStatus(str, Enum):
	RAW          = "raw"
	VALIDATED    = "validated"
	FUSED        = "fused"
	ASSESSED     = "assessed"
	DISSEMINATED = "disseminated"
	ARCHIVED     = "archived"
	REJECTED     = "rejected"


class WorkspaceStatus(str, Enum):
	ACTIVE    = "active"
	SUSPENDED = "suspended"
	CLOSED    = "closed"


class WorkspaceType(str, Enum):
	CASE_FUSION       = "case_fusion"
	THREAT_FUSION     = "threat_fusion"
	FRAUD_FUSION      = "fraud_fusion"
	PUBLIC_SAFETY     = "public_safety"
	STRATEGIC         = "strategic_assessment"
	OPERATIONAL       = "operational_picture"
	INCIDENT          = "incident_fusion"


class CorrelationType(str, Enum):
	ENTITY_MATCH           = "entity_match"
	TIME_SEQUENCE          = "time_sequence"
	LOCATION_OVERLAP       = "location_overlap"
	NETWORK_LINK           = "network_link"
	PATTERN_MATCH          = "pattern_match"
	CROSS_SOURCE_CONFIRM   = "cross_source_confirmation"
	CONTRADICTION          = "contradiction"


class CorrelationSetStatus(str, Enum):
	OPEN      = "open"
	CONFIRMED = "confirmed"
	DISPUTED  = "disputed"
	CLOSED    = "closed"


class AssessmentType(str, Enum):
	THREAT        = "threat"
	FRAUD         = "fraud"
	PUBLIC_SAFETY = "public_safety"
	OPERATIONAL   = "operational"
	STRATEGIC     = "strategic"
	CONFIDENCE    = "confidence"
	IMPACT        = "impact"


class RiskLevel(str, Enum):
	LOW      = "low"
	MEDIUM   = "medium"
	HIGH     = "high"
	CRITICAL = "critical"


class ProductType(str, Enum):
	SITREP             = "sitrep"
	THREAT_ASSESSMENT  = "threat_assessment"
	INTELLIGENCE_BRIEF = "intelligence_brief"
	FINISHED_INTEL     = "finished_intelligence"
	TACTICAL_REPORT    = "tactical_report"
	STRATEGIC_ESTIMATE = "strategic_estimate"


class ProductStatus(str, Enum):
	DRAFT    = "draft"
	REVIEW   = "review"
	APPROVED = "approved"
	RELEASED = "released"
	RECALLED = "recalled"


class JudgementType(str, Enum):
	ATTRIBUTION      = "attribution"
	INTENT           = "intent"
	CAPABILITY       = "capability"
	RISK             = "risk"
	RELATIONSHIP     = "relationship"
	TIMELINE         = "timeline"
	COURSE_OF_ACTION = "course_of_action"


class ConfidenceLevel(str, Enum):
	"""Analytic confidence — IC standard words (ICD 203)."""
	ALMOST_CERTAIN  = "almost_certain"    # ≥ 0.93
	HIGHLY_LIKELY   = "highly_likely"     # 0.80–0.92
	LIKELY          = "likely"            # 0.55–0.79
	ROUGHLY_EVEN    = "roughly_even"      # 0.45–0.54
	UNLIKELY        = "unlikely"          # 0.20–0.44
	HIGHLY_UNLIKELY = "highly_unlikely"   # 0.07–0.19
	REMOTE          = "remote"            # < 0.07


class EvidenceType(str, Enum):
	DOCUMENT    = "document"
	SIGNAL      = "signal"
	IMAGE       = "image"
	VIDEO       = "video"
	GEOSPATIAL  = "geospatial"
	TRANSACTION = "transaction"
	INDICATOR   = "indicator"
	ENTITY      = "entity"
	OBSERVATION = "observation"


class EvidenceStatus(str, Enum):
	PENDING     = "pending"
	VERIFIED    = "verified"
	CHALLENGED  = "challenged"
	DISCREDITED = "discredited"


class HypothesisStatus(str, Enum):
	OPEN         = "open"
	SUPPORTED    = "supported"
	REFUTED      = "refuted"
	INCONCLUSIVE = "inconclusive"


class SATMethod(str, Enum):
	"""Structured Analytic Techniques."""
	ACH                  = "analysis_of_competing_hypotheses"
	KEY_ASSUMPTIONS      = "key_assumptions_check"
	DEVIL_ADVOCATE       = "devils_advocacy"
	RED_TEAM             = "red_team"
	CONE_OF_PLAUSIBILITY = "cone_of_plausibility"
	PREMORTEM            = "premortem"
	QUALITY_CHECK        = "quality_of_information_check"


# ─────────────────────────────────────────────────────────────────────────────
# Base model
# ─────────────────────────────────────────────────────────────────────────────

class FusionBase(BaseModel):
	"""All fusion entities share this base."""
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
	)
	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(default="system")
	is_deleted: bool = Field(default=False)


# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceItem — raw multi-source intake
# ─────────────────────────────────────────────────────────────────────────────

class IntelligenceItemCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmptyStr
	source_type: SourceType
	source_reference: NonEmptyStr
	content_summary: str = ""
	content_fingerprint: NonEmptyStr
	classification: ClassificationLevel = ClassificationLevel.UNCLASSIFIED
	tlp: TLPLevel = TLPLevel.AMBER
	confidence_score: ConfidenceScore = 0.5
	collected_at: datetime = Field(default_factory=datetime.utcnow)
	custodian_id: NonEmptyStr
	workspace_id: str = ""
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_by: str = "system"


class IntelligenceItemUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	content_summary: str | None = None
	status: IntelItemStatus | None = None
	confidence_score: ConfidenceScore | None = None
	tlp: TLPLevel | None = None
	metadata: dict[str, Any] | None = None
	workspace_id: str | None = None


class IntelligenceItem(FusionBase):
	"""A single intelligence item from any source."""
	source_type: SourceType
	source_reference: NonEmptyStr
	content_summary: str = ""
	content_fingerprint: NonEmptyStr
	classification: ClassificationLevel = ClassificationLevel.UNCLASSIFIED
	tlp: TLPLevel = TLPLevel.AMBER
	confidence_score: ConfidenceScore = 0.5
	collected_at: datetime = Field(default_factory=datetime.utcnow)
	custodian_id: NonEmptyStr
	workspace_id: str = ""
	status: IntelItemStatus = IntelItemStatus.RAW
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# FusionWorkspace — analytical container
# ─────────────────────────────────────────────────────────────────────────────

class FusionWorkspaceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmptyStr
	workspace_type: WorkspaceType
	name: NonEmptyStr
	classification: ClassificationLevel = ClassificationLevel.UNCLASSIFIED
	authority_id: NonEmptyStr
	description: str = ""
	tags: list[str] = Field(default_factory=list)
	lead_analyst_id: str = ""
	created_by: str = "system"


class FusionWorkspaceUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	name: str | None = None
	description: str | None = None
	status: WorkspaceStatus | None = None
	tags: list[str] | None = None
	lead_analyst_id: str | None = None


class FusionWorkspace(FusionBase):
	"""Analytical container linking items, correlations and products."""
	workspace_type: WorkspaceType
	name: NonEmptyStr
	classification: ClassificationLevel = ClassificationLevel.UNCLASSIFIED
	authority_id: NonEmptyStr
	description: str = ""
	tags: list[str] = Field(default_factory=list)
	status: WorkspaceStatus = WorkspaceStatus.ACTIVE
	item_count: int = 0
	lead_analyst_id: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# CorrelationSet — cross-source linkage
# ─────────────────────────────────────────────────────────────────────────────

class CorrelationSetCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmptyStr
	workspace_id: NonEmptyStr
	correlation_type: CorrelationType
	item_ids: list[str] = Field(default_factory=list)
	analyst_id: NonEmptyStr
	confidence_score: ConfidenceScore = 0.5
	rationale: str = ""
	evidence_ids: list[str] = Field(default_factory=list)
	created_by: str = "system"


class CorrelationSetUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: CorrelationSetStatus | None = None
	confidence_score: ConfidenceScore | None = None
	rationale: str | None = None
	item_ids: list[str] | None = None
	evidence_ids: list[str] | None = None


class CorrelationSet(FusionBase):
	"""A named set of correlated intelligence items."""
	workspace_id: NonEmptyStr
	correlation_type: CorrelationType
	item_ids: list[str] = Field(default_factory=list)
	analyst_id: NonEmptyStr
	confidence_score: ConfidenceScore = 0.5
	rationale: str = ""
	evidence_ids: list[str] = Field(default_factory=list)
	status: CorrelationSetStatus = CorrelationSetStatus.OPEN


# ─────────────────────────────────────────────────────────────────────────────
# AssessmentPicture — synthesised threat/risk picture
# ─────────────────────────────────────────────────────────────────────────────

class AssessmentPictureCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmptyStr
	workspace_id: NonEmptyStr
	assessment_type: AssessmentType
	risk_level: RiskLevel
	summary: str = ""
	analyst_id: NonEmptyStr
	confidence_score: ConfidenceScore = 0.5
	hypothesis_ids: list[str] = Field(default_factory=list)
	correlation_ids: list[str] = Field(default_factory=list)
	evidence_ids: list[str] = Field(default_factory=list)
	created_by: str = "system"


class AssessmentPictureUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	risk_level: RiskLevel | None = None
	summary: str | None = None
	confidence_score: ConfidenceScore | None = None
	hypothesis_ids: list[str] | None = None
	correlation_ids: list[str] | None = None
	evidence_ids: list[str] | None = None


class AssessmentPicture(FusionBase):
	"""Synthesised threat/risk/fraud picture drawn from correlations + hypotheses."""
	workspace_id: NonEmptyStr
	assessment_type: AssessmentType
	risk_level: RiskLevel
	summary: str = ""
	analyst_id: NonEmptyStr
	confidence_score: ConfidenceScore = 0.5
	hypothesis_ids: list[str] = Field(default_factory=list)
	correlation_ids: list[str] = Field(default_factory=list)
	evidence_ids: list[str] = Field(default_factory=list)
	approved_by: str = ""
	approved_at: datetime | None = None


# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceProduct — finished intelligence artefact
# ─────────────────────────────────────────────────────────────────────────────

class IntelligenceProductCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmptyStr
	workspace_id: NonEmptyStr
	product_type: ProductType
	title: NonEmptyStr
	classification: ClassificationLevel = ClassificationLevel.UNCLASSIFIED
	tlp: TLPLevel = TLPLevel.AMBER
	summary: str = ""
	body_reference: str = ""
	assessment_ids: list[str] = Field(default_factory=list)
	author_id: NonEmptyStr
	created_by: str = "system"


class IntelligenceProductUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	title: str | None = None
	summary: str | None = None
	body_reference: str | None = None
	tlp: TLPLevel | None = None
	status: ProductStatus | None = None
	reviewer_id: str | None = None


class IntelligenceProduct(FusionBase):
	"""Finished intelligence product ready for dissemination."""
	workspace_id: NonEmptyStr
	product_type: ProductType
	title: NonEmptyStr
	classification: ClassificationLevel = ClassificationLevel.UNCLASSIFIED
	tlp: TLPLevel = TLPLevel.AMBER
	summary: str = ""
	body_reference: str = ""
	assessment_ids: list[str] = Field(default_factory=list)
	author_id: NonEmptyStr
	status: ProductStatus = ProductStatus.DRAFT
	reviewer_id: str = ""
	reviewed_at: datetime | None = None
	released_at: datetime | None = None
	dissemination_ids: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# AnalyticalJudgement — calibrated estimative statement
# ─────────────────────────────────────────────────────────────────────────────

class AnalyticalJudgementCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmptyStr
	workspace_id: NonEmptyStr
	judgement_type: JudgementType
	statement: NonEmptyStr
	confidence_score: ConfidenceScore = 0.5
	confidence_level: ConfidenceLevel = ConfidenceLevel.LIKELY
	analyst_id: NonEmptyStr
	sat_method: SATMethod | None = None
	key_assumptions: list[str] = Field(default_factory=list)
	evidence_ids: list[str] = Field(default_factory=list)
	created_by: str = "system"


class AnalyticalJudgementUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	statement: str | None = None
	confidence_score: ConfidenceScore | None = None
	confidence_level: ConfidenceLevel | None = None
	key_assumptions: list[str] | None = None
	sat_method: SATMethod | None = None
	evidence_ids: list[str] | None = None


class AnalyticalJudgement(FusionBase):
	"""Calibrated estimative judgement produced via structured analytic techniques."""
	workspace_id: NonEmptyStr
	judgement_type: JudgementType
	statement: NonEmptyStr
	confidence_score: ConfidenceScore = 0.5
	confidence_level: ConfidenceLevel = ConfidenceLevel.LIKELY
	analyst_id: NonEmptyStr
	sat_method: SATMethod | None = None
	key_assumptions: list[str] = Field(default_factory=list)
	evidence_ids: list[str] = Field(default_factory=list)
	challenger_ids: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Evidence — individual evidence item
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmptyStr
	workspace_id: NonEmptyStr
	evidence_type: EvidenceType
	source_reference: NonEmptyStr
	content_fingerprint: NonEmptyStr
	classification: ClassificationLevel = ClassificationLevel.UNCLASSIFIED
	custodian_id: NonEmptyStr
	chain_of_custody: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_by: str = "system"


class EvidenceUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: EvidenceStatus | None = None
	chain_of_custody: list[str] | None = None
	metadata: dict[str, Any] | None = None


class Evidence(FusionBase):
	"""Provenance-tracked evidence item with full chain-of-custody."""
	workspace_id: NonEmptyStr
	evidence_type: EvidenceType
	source_reference: NonEmptyStr
	content_fingerprint: NonEmptyStr
	classification: ClassificationLevel = ClassificationLevel.UNCLASSIFIED
	custodian_id: NonEmptyStr
	status: EvidenceStatus = EvidenceStatus.PENDING
	chain_of_custody: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# HypothesisTest — structured ACH-style test
# ─────────────────────────────────────────────────────────────────────────────

class HypothesisTestCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: NonEmptyStr
	workspace_id: NonEmptyStr
	statement: NonEmptyStr
	sat_method: SATMethod = SATMethod.ACH
	analyst_id: NonEmptyStr
	alternative_hypotheses: list[str] = Field(default_factory=list)
	evidence_ids: list[str] = Field(default_factory=list)
	initial_confidence: ConfidenceScore = 0.5
	created_by: str = "system"


class HypothesisTestUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: HypothesisStatus | None = None
	final_confidence: ConfidenceScore | None = None
	conclusion: str | None = None
	evidence_ids: list[str] | None = None
	alternative_hypotheses: list[str] | None = None
	ach_matrix: dict[str, list[float]] | None = None


class HypothesisTest(FusionBase):
	"""Structured hypothesis test using a named SAT method."""
	workspace_id: NonEmptyStr
	statement: NonEmptyStr
	sat_method: SATMethod = SATMethod.ACH
	analyst_id: NonEmptyStr
	alternative_hypotheses: list[str] = Field(default_factory=list)
	evidence_ids: list[str] = Field(default_factory=list)
	initial_confidence: ConfidenceScore = 0.5
	final_confidence: ConfidenceScore | None = None
	status: HypothesisStatus = HypothesisStatus.OPEN
	conclusion: str = ""
	# ACH matrix: evidence_id → hypothesis_idx → consistency score [-1..1]
	ach_matrix: dict[str, list[float]] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation / Report models
# ─────────────────────────────────────────────────────────────────────────────

class FusionWorkspaceSummary(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	workspace_id: str
	workspace_name: str
	workspace_type: WorkspaceType
	item_count: int
	correlation_count: int
	hypothesis_count: int
	assessment_count: int
	product_count: int
	evidence_count: int
	lead_analyst_id: str
	status: WorkspaceStatus
	classification: ClassificationLevel


class FusionDashboardReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	total_items: int
	items_by_source: dict[str, int]
	items_by_status: dict[str, int]
	total_workspaces: int
	active_workspaces: int
	total_correlations: int
	total_assessments: int
	critical_assessments: int
	total_products: int
	released_products: int
	total_hypotheses: int
	open_hypotheses: int
	total_evidence: int
	total_judgements: int
	as_of: datetime = Field(default_factory=datetime.utcnow)


class ACHMatrix(BaseModel):
	"""Analysis of Competing Hypotheses result matrix."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	hypotheses: list[str]
	evidence_labels: list[str]
	# rows=evidence, cols=hypotheses, value in {-1, 0, 1}
	matrix: list[list[float]]
	inconsistency_scores: list[float]
	hypothesis_confidence: list[float]
	leading_hypothesis_idx: int
	leading_hypothesis: str
	confidence: ConfidenceScore


class KeyAssumptionsResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	assumptions: list[dict[str, Any]]
	robustness: float
	weakest_assumption: str
	weakest_confidence: float
	analytic_recommendation: str


class ConfidenceCalibration(BaseModel):
	"""Bayesian-updated confidence calibration result."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	prior: ConfidenceScore
	likelihood_ratio: float
	posterior: ConfidenceScore
	confidence_level: ConfidenceLevel
	word_equivalent: str


class FusionQualityResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	quality_score: float
	dimensions: dict[str, float]
	recommendation: str


class DisseminationRecord(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	product_id: NonEmptyStr
	audience: NonEmptyStr
	tlp: TLPLevel
	approval_reference: NonEmptyStr
	disseminated_at: datetime = Field(default_factory=datetime.utcnow)
	disseminated_by: NonEmptyStr
	notes: str = ""


class FusionEvent(BaseModel):
	"""Domain event emitted on every state change."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	event_id: str = Field(default_factory=uuid7str)
	event_type: str
	tenant_id: str
	actor_id: str
	resource_id: str
	resource_type: str
	payload: dict[str, Any] = Field(default_factory=dict)
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	capability_id: str = "intel_fusion"
	stream: str = "apg.intel.fusion.lifecycle"


# ─────────────────────────────────────────────────────────────────────────────
# Pagination / List wrappers
# ─────────────────────────────────────────────────────────────────────────────

class PagedResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	items: list[Any]
	total: int
	page: int
	page_size: int
	has_more: bool
