"""Pydantic v2 data models for APG Know Your Customer.

Covers all KYC lifecycle entities: applications, documents, biometrics,
risk profiles, screening results, business KYC, UBO declarations, reviews,
and onboarding journeys.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class CustomerType(str, Enum):
	individual = "individual"
	sole_proprietor = "sole_proprietor"
	business = "business"
	nonprofit = "nonprofit"
	government = "government"
	trust = "trust"
	partnership = "partnership"
	ngo = "ngo"


class ApplicationStatus(str, Enum):
	draft = "draft"
	in_progress = "in_progress"
	pending_review = "pending_review"
	pending_edd = "pending_edd"
	approved = "approved"
	rejected = "rejected"
	expired = "expired"
	suspended = "suspended"
	reactivation_required = "reactivation_required"


class DocumentType(str, Enum):
	passport = "passport"
	national_id = "national_id"
	driver_license = "driver_license"
	resident_permit = "resident_permit"
	refugee_document = "refugee_document"
	birth_certificate = "birth_certificate"
	business_registration = "business_registration"
	tax_id = "tax_id"
	utility_bill = "utility_bill"
	bank_statement = "bank_statement"
	huduma_namba = "huduma_namba"
	bvn = "bvn"
	ghana_card = "ghana_card"
	voter_id = "voter_id"
	certificate_of_incorporation = "certificate_of_incorporation"
	memorandum_of_association = "memorandum_of_association"
	board_resolution = "board_resolution"


class DocumentStatus(str, Enum):
	pending = "pending"
	verified = "verified"
	rejected = "rejected"
	expired = "expired"
	deceased_id_flagged = "deceased_id_flagged"
	synthetic_fraud_flagged = "synthetic_fraud_flagged"


class BiometricType(str, Enum):
	facial = "facial"
	fingerprint = "fingerprint"
	iris = "iris"
	voice = "voice"
	liveness_video = "liveness_video"


class BiometricStatus(str, Enum):
	pending = "pending"
	live = "live"
	spoof_detected = "spoof_detected"
	failed = "failed"
	expired = "expired"


class RiskBand(str, Enum):
	low = "low"
	medium = "medium"
	high = "high"
	very_high = "very_high"
	unacceptable = "unacceptable"


class ScreeningStatus(str, Enum):
	pending = "pending"
	clear = "clear"
	hit = "hit"
	false_positive = "false_positive"
	confirmed_hit = "confirmed_hit"
	under_review = "under_review"


class ReviewStatus(str, Enum):
	open = "open"
	in_progress = "in_progress"
	approved = "approved"
	rejected = "rejected"
	escalated = "escalated"
	closed = "closed"


class ReviewType(str, Enum):
	standard_kyc = "standard_kyc"
	enhanced_due_diligence = "enhanced_due_diligence"
	pep_review = "pep_review"
	sanctions_review = "sanctions_review"
	adverse_media_review = "adverse_media_review"
	periodic_refresh = "periodic_refresh"
	dormant_reactivation = "dormant_reactivation"


class JourneyStatus(str, Enum):
	started = "started"
	documents_pending = "documents_pending"
	biometrics_pending = "biometrics_pending"
	screening_pending = "screening_pending"
	review_pending = "review_pending"
	completed = "completed"
	abandoned = "abandoned"
	failed = "failed"


class OwnershipType(str, Enum):
	direct = "direct"
	indirect = "indirect"
	nominee = "nominee"
	beneficial = "beneficial"


# ─────────────────────────────────────────────────────────────────────────────
# Validators
# ─────────────────────────────────────────────────────────────────────────────

def _non_empty(v: str) -> str:
	assert v and v.strip(), "must not be empty"
	return v.strip()


def _confidence_range(v: float) -> float:
	assert 0.0 <= v <= 1.0, "confidence must be between 0.0 and 1.0"
	return v


def _risk_score_range(v: int) -> int:
	assert 0 <= v <= 100, "risk_score must be between 0 and 100"
	return v


def _ownership_pct(v: float) -> float:
	assert 0.0 < v <= 100.0, "ownership_percentage must be between 0 and 100"
	return v


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]
Confidence = Annotated[float, AfterValidator(_confidence_range)]
RiskScore = Annotated[int, AfterValidator(_risk_score_range)]
OwnershipPct = Annotated[float, AfterValidator(_ownership_pct)]


# ─────────────────────────────────────────────────────────────────────────────
# Base model
# ─────────────────────────────────────────────────────────────────────────────

class KYCBase(BaseModel):
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
	)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"
	is_deleted: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# KYCApplication
# ─────────────────────────────────────────────────────────────────────────────

class KYCApplicationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	customer_id: NonEmptyStr
	customer_type: CustomerType
	country_code: NonEmptyStr  # ISO-3166-1 alpha-2
	legal_name: NonEmptyStr
	consent_reference: NonEmptyStr
	kyc_tier: str = "standard"
	# Edge-case flags
	is_refugee: bool = False
	is_informal_sector: bool = False
	preferred_language: str = "en"
	metadata: dict[str, Any] = Field(default_factory=dict)


class KYCApplicationUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: ApplicationStatus | None = None
	kyc_tier: str | None = None
	metadata: dict[str, Any] | None = None


class KYCApplication(KYCBase):
	customer_id: NonEmptyStr
	customer_type: CustomerType
	country_code: NonEmptyStr
	legal_name: NonEmptyStr
	consent_reference: NonEmptyStr
	kyc_tier: str = "standard"
	status: ApplicationStatus = ApplicationStatus.draft
	risk_score: int = 0
	risk_band: RiskBand = RiskBand.low
	is_refugee: bool = False
	is_informal_sector: bool = False
	preferred_language: str = "en"
	expiry_date: date | None = None
	last_verified_at: datetime | None = None
	edd_triggered_at: datetime | None = None
	dormant_since: datetime | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# IDDocument
# ─────────────────────────────────────────────────────────────────────────────

class IDDocumentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	application_id: NonEmptyStr
	document_type: DocumentType
	token_reference: NonEmptyStr  # tokenised vault reference — never raw doc data
	document_number: str = ""
	issuing_country: str = ""
	issuing_authority: str = ""
	issue_date: date | None = None
	expiry_date: date | None = None
	extracted_name: str = ""
	extracted_dob: date | None = None
	extracted_nationality: str = ""
	name_script: str = "latin"  # arabic, chinese, cyrillic, latin, etc.
	name_transliterated: str = ""  # latin transliteration when non-latin
	confidence: Confidence = 0.0
	ocr_raw: dict[str, Any] = Field(default_factory=dict)
	metadata: dict[str, Any] = Field(default_factory=dict)


class IDDocumentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: DocumentStatus | None = None
	confidence: Confidence | None = None
	expiry_date: date | None = None
	metadata: dict[str, Any] | None = None


class IDDocument(KYCBase):
	application_id: NonEmptyStr
	document_type: DocumentType
	token_reference: NonEmptyStr
	document_number: str = ""
	issuing_country: str = ""
	issuing_authority: str = ""
	issue_date: date | None = None
	expiry_date: date | None = None
	extracted_name: str = ""
	extracted_dob: date | None = None
	extracted_nationality: str = ""
	name_script: str = "latin"
	name_transliterated: str = ""
	confidence: Confidence = 0.0
	status: DocumentStatus = DocumentStatus.pending
	deceased_check_performed: bool = False
	synthetic_fraud_score: float = 0.0
	ocr_raw: dict[str, Any] = Field(default_factory=dict)
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# BiometricData
# ─────────────────────────────────────────────────────────────────────────────

class BiometricDataCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	application_id: NonEmptyStr
	biometric_type: BiometricType
	token_reference: NonEmptyStr
	liveness_score: Confidence = 0.0
	match_score: Confidence = 0.0
	spoof_score: float = 0.0
	capture_device: str = ""
	metadata: dict[str, Any] = Field(default_factory=dict)


class BiometricData(KYCBase):
	application_id: NonEmptyStr
	biometric_type: BiometricType
	token_reference: NonEmptyStr
	liveness_score: Confidence = 0.0
	match_score: Confidence = 0.0
	spoof_score: float = 0.0
	capture_device: str = ""
	status: BiometricStatus = BiometricStatus.pending
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# RiskProfile
# ─────────────────────────────────────────────────────────────────────────────

class RiskProfileCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	application_id: NonEmptyStr
	customer_type: CustomerType
	country_code: NonEmptyStr
	is_pep: bool = False
	is_sanctioned: bool = False
	is_adverse_media: bool = False
	high_risk_country: bool = False
	high_risk_industry: bool = False
	complex_ownership_structure: bool = False
	nominee_shareholders_present: bool = False
	metadata: dict[str, Any] = Field(default_factory=dict)


class RiskProfile(KYCBase):
	application_id: NonEmptyStr
	customer_type: CustomerType
	country_code: NonEmptyStr
	risk_score: RiskScore = 0
	risk_band: RiskBand = RiskBand.low
	is_pep: bool = False
	is_sanctioned: bool = False
	is_adverse_media: bool = False
	high_risk_country: bool = False
	high_risk_industry: bool = False
	complex_ownership_structure: bool = False
	nominee_shareholders_present: bool = False
	score_breakdown: dict[str, int] = Field(default_factory=dict)
	edd_required: bool = False
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# PEPCheck
# ─────────────────────────────────────────────────────────────────────────────

class PEPCheckCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	application_id: NonEmptyStr
	full_name: NonEmptyStr
	date_of_birth: date | None = None
	nationality: str = ""
	match_threshold: Confidence = 0.85
	metadata: dict[str, Any] = Field(default_factory=dict)


class PEPCheck(KYCBase):
	application_id: NonEmptyStr
	full_name: NonEmptyStr
	date_of_birth: date | None = None
	nationality: str = ""
	match_threshold: Confidence = 0.85
	status: ScreeningStatus = ScreeningStatus.pending
	is_hit: bool = False
	match_score: float = 0.0
	matched_name: str = ""
	pep_category: str = ""  # domestic, foreign, international_org
	pep_level: str = ""  # head_of_state, minister, local_govt, etc.
	source_list: str = ""
	false_positive_reason: str = ""
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# SanctionCheck
# ─────────────────────────────────────────────────────────────────────────────

class SanctionCheckCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	application_id: NonEmptyStr
	full_name: NonEmptyStr
	date_of_birth: date | None = None
	nationality: str = ""
	lists_screened: list[str] = Field(default_factory=lambda: ["OFAC", "UN", "EU"])
	match_threshold: Confidence = 0.85
	metadata: dict[str, Any] = Field(default_factory=dict)


class SanctionCheck(KYCBase):
	application_id: NonEmptyStr
	full_name: NonEmptyStr
	date_of_birth: date | None = None
	nationality: str = ""
	lists_screened: list[str] = Field(default_factory=list)
	match_threshold: Confidence = 0.85
	status: ScreeningStatus = ScreeningStatus.pending
	is_hit: bool = False
	matched_lists: list[str] = Field(default_factory=list)
	match_score: float = 0.0
	matched_name: str = ""
	false_positive_reason: str = ""
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# AdverseMediaCheck
# ─────────────────────────────────────────────────────────────────────────────

class AdverseMediaCheckCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	application_id: NonEmptyStr
	full_name: NonEmptyStr
	search_terms: list[str] = Field(default_factory=list)
	categories: list[str] = Field(default_factory=lambda: [
		"financial_crime", "fraud", "corruption", "terrorism",
		"drug_trafficking", "human_trafficking", "money_laundering",
	])
	metadata: dict[str, Any] = Field(default_factory=dict)


class AdverseMediaCheck(KYCBase):
	application_id: NonEmptyStr
	full_name: NonEmptyStr
	search_terms: list[str] = Field(default_factory=list)
	categories: list[str] = Field(default_factory=list)
	status: ScreeningStatus = ScreeningStatus.pending
	is_hit: bool = False
	hit_categories: list[str] = Field(default_factory=list)
	article_count: int = 0
	oldest_article_date: date | None = None
	newest_article_date: date | None = None
	summary: str = ""
	false_positive_reason: str = ""
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# BusinessKYC
# ─────────────────────────────────────────────────────────────────────────────

class BusinessKYCCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	application_id: NonEmptyStr
	registered_name: NonEmptyStr
	trading_name: str = ""
	registration_number: NonEmptyStr
	registration_country: NonEmptyStr
	registration_date: date | None = None
	industry_code: str = ""  # ISIC / NAICS code
	annual_revenue_usd: float | None = None
	number_of_employees: int | None = None
	website: str = ""
	primary_business_activity: str = ""
	has_complex_structure: bool = False
	has_nominee_shareholders: bool = False
	metadata: dict[str, Any] = Field(default_factory=dict)


class BusinessKYC(KYCBase):
	application_id: NonEmptyStr
	registered_name: NonEmptyStr
	trading_name: str = ""
	registration_number: NonEmptyStr
	registration_country: NonEmptyStr
	registration_date: date | None = None
	industry_code: str = ""
	annual_revenue_usd: float | None = None
	number_of_employees: int | None = None
	website: str = ""
	primary_business_activity: str = ""
	has_complex_structure: bool = False
	has_nominee_shareholders: bool = False
	ubo_count: int = 0
	status: ApplicationStatus = ApplicationStatus.draft
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# UBODeclaration
# ─────────────────────────────────────────────────────────────────────────────

class UBODeclarationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	business_kyc_id: NonEmptyStr
	application_id: NonEmptyStr
	full_name: NonEmptyStr
	date_of_birth: date | None = None
	nationality: NonEmptyStr
	country_of_residence: str = ""
	ownership_percentage: OwnershipPct
	ownership_type: OwnershipType = OwnershipType.direct
	is_nominee: bool = False
	controlling_interest: bool = False
	metadata: dict[str, Any] = Field(default_factory=dict)


class UBODeclaration(KYCBase):
	business_kyc_id: NonEmptyStr
	application_id: NonEmptyStr
	full_name: NonEmptyStr
	date_of_birth: date | None = None
	nationality: NonEmptyStr
	country_of_residence: str = ""
	ownership_percentage: OwnershipPct
	ownership_type: OwnershipType = OwnershipType.direct
	is_nominee: bool = False
	controlling_interest: bool = False
	kyc_status: ApplicationStatus = ApplicationStatus.draft
	pep_check_id: str | None = None
	sanction_check_id: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# KYCReview
# ─────────────────────────────────────────────────────────────────────────────

class KYCReviewCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	application_id: NonEmptyStr
	review_type: ReviewType
	assigned_to: str = ""
	notes: str = ""
	metadata: dict[str, Any] = Field(default_factory=dict)


class KYCReviewUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: ReviewStatus | None = None
	decision: str | None = None
	notes: str | None = None
	assigned_to: str | None = None
	metadata: dict[str, Any] | None = None


class KYCReview(KYCBase):
	application_id: NonEmptyStr
	review_type: ReviewType
	status: ReviewStatus = ReviewStatus.open
	decision: str = ""
	assigned_to: str = ""
	notes: str = ""
	opened_at: datetime = Field(default_factory=datetime.utcnow)
	completed_at: datetime | None = None
	escalated_at: datetime | None = None
	escalation_reason: str = ""
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# OnboardingJourney
# ─────────────────────────────────────────────────────────────────────────────

class OnboardingJourneyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: NonEmptyStr
	application_id: NonEmptyStr
	channel: str = "web"  # web, mobile, agent, branch
	customer_type: CustomerType
	metadata: dict[str, Any] = Field(default_factory=dict)


class OnboardingJourney(KYCBase):
	application_id: NonEmptyStr
	channel: str = "web"
	customer_type: CustomerType
	status: JourneyStatus = JourneyStatus.started
	current_step: str = "identity"
	steps_completed: list[str] = Field(default_factory=list)
	steps_required: list[str] = Field(default_factory=list)
	started_at: datetime = Field(default_factory=datetime.utcnow)
	completed_at: datetime | None = None
	abandoned_at: datetime | None = None
	time_to_complete_seconds: int | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Report / aggregation models
# ─────────────────────────────────────────────────────────────────────────────

class KYCDashboardStats(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	total_applications: int = 0
	approved: int = 0
	rejected: int = 0
	pending_review: int = 0
	pending_edd: int = 0
	expired: int = 0
	avg_risk_score: float = 0.0
	high_risk_count: int = 0
	pep_hits: int = 0
	sanction_hits: int = 0
	adverse_media_hits: int = 0
	avg_onboarding_seconds: float = 0.0
	approval_rate: float = 0.0
	generated_at: datetime = Field(default_factory=datetime.utcnow)


class KYCExpiryReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	expiring_within_30_days: list[str] = Field(default_factory=list)
	expiring_within_90_days: list[str] = Field(default_factory=list)
	already_expired: list[str] = Field(default_factory=list)
	generated_at: datetime = Field(default_factory=datetime.utcnow)


class KYCRiskReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	by_risk_band: dict[str, int] = Field(default_factory=dict)
	by_country: dict[str, int] = Field(default_factory=dict)
	by_customer_type: dict[str, int] = Field(default_factory=dict)
	edd_pending: int = 0
	generated_at: datetime = Field(default_factory=datetime.utcnow)


class DomainEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	event_id: str = Field(default_factory=uuid7str)
	event_type: str
	tenant_id: str
	actor_id: str
	resource_id: str
	resource_type: str
	capability_id: str = "fintech_kyc"
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	payload: dict[str, Any] = Field(default_factory=dict)
