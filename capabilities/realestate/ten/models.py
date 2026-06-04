"""Pydantic v2 models for Tenant Management (ten)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


class TenantType(str, Enum):
	corporate = "corporate"
	sme = "sme"
	sole_trader = "sole_trader"
	individual = "individual"
	government = "government"
	ngo = "ngo"
	educational = "educational"
	healthcare = "healthcare"
	retail_brand = "retail_brand"
	franchise = "franchise"
	co_working_operator = "co_working_operator"


class TenantStatus(str, Enum):
	prospect = "prospect"
	applicant = "applicant"
	approved = "approved"
	active = "active"
	notice_served = "notice_served"
	vacating = "vacating"
	former = "former"
	blacklisted = "blacklisted"


class ServiceRequestType(str, Enum):
	maintenance_request = "maintenance_request"
	cleaning_request = "cleaning_request"
	access_request = "access_request"
	parking_request = "parking_request"
	delivery_coordination = "delivery_coordination"
	visitor_management = "visitor_management"
	it_support = "it_support"
	noise_complaint = "noise_complaint"
	neighbour_dispute = "neighbour_dispute"
	general_enquiry = "general_enquiry"


class RequestStatus(str, Enum):
	open = "open"
	acknowledged = "acknowledged"
	assigned = "assigned"
	in_progress = "in_progress"
	awaiting_tenant = "awaiting_tenant"
	resolved = "resolved"
	closed = "closed"
	escalated = "escalated"


class CommunicationChannel(str, Enum):
	portal = "portal"
	email = "email"
	sms = "sms"
	whatsapp = "whatsapp"
	phone = "phone"
	letter = "letter"
	in_person = "in_person"


class CreditGrade(str, Enum):
	A = "A"
	B = "B"
	C = "C"
	D = "D"
	F = "F"


class EscalationType(str, Enum):
	noise_complaint = "noise_complaint"
	rent_arrears = "rent_arrears"
	lease_breach = "lease_breach"
	property_damage = "property_damage"
	anti_social_behaviour = "anti_social_behaviour"
	subletting_unauthorised = "subletting_unauthorised"


class OnboardingStep(str, Enum):
	application_received = "application_received"
	referencing = "referencing"
	credit_check = "credit_check"
	right_to_rent = "right_to_rent"
	lease_negotiation = "lease_negotiation"
	lease_signing = "lease_signing"
	deposit_registration = "deposit_registration"
	key_handover = "key_handover"
	welcome_pack_sent = "welcome_pack_sent"
	portal_activated = "portal_activated"


# ── Tenant Entity ─────────────────────────────────────────────────────────────

class TenantEntityCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str  # platform tenant (organisation)
	name: str
	tenant_type: TenantType
	registration_number: str | None = None
	email: str
	phone: str | None = None
	address: str | None = None
	contact_person: str | None = None
	created_by: str


class TenantEntityResponse(TenantEntityCreate):
	id: str = Field(default_factory=uuid7str)
	status: TenantStatus = TenantStatus.prospect
	credit_grade: CreditGrade | None = None
	tenant_score: Decimal | None = None
	active_tenancies: int = 0
	onboarding_steps_completed: list[OnboardingStep] = Field(default_factory=list)
	mandatory_onboarding_complete: bool = False
	portal_active: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class TenantEntityUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: TenantStatus | None = None
	credit_grade: CreditGrade | None = None
	email: str | None = None
	phone: str | None = None
	contact_person: str | None = None


# ── Onboarding ────────────────────────────────────────────────────────────────

class OnboardingStepRecord(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	tenant_entity_id: str
	step: OnboardingStep
	completed_by: str
	notes: str | None = None
	document_ids: list[str] = Field(default_factory=list)


class OnboardingStepResponse(OnboardingStepRecord):
	id: str = Field(default_factory=uuid7str)
	completed_at: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Service Request ───────────────────────────────────────────────────────────

class ServiceRequestCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	tenant_entity_id: str
	property_id: str
	unit_id: str | None = None
	request_type: ServiceRequestType
	subject: str
	description: str
	preferred_channel: CommunicationChannel = CommunicationChannel.portal
	attachments: list[str] = Field(default_factory=list)
	created_by: str


class ServiceRequestResponse(ServiceRequestCreate):
	id: str = Field(default_factory=uuid7str)
	ref: str = ""
	status: RequestStatus = RequestStatus.open
	assigned_to: str | None = None
	sla_response_deadline: datetime | None = None
	sla_breached: bool = False
	resolved_at: datetime | None = None
	resolution_notes: str | None = None
	satisfaction_rating: int | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class ServiceRequestUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: RequestStatus | None = None
	assigned_to: str | None = None
	resolution_notes: str | None = None


# ── Communication ─────────────────────────────────────────────────────────────

class CommunicationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	tenant_entity_id: str
	channel: CommunicationChannel
	subject: str
	body: str
	sent_by: str
	direction: str = "outbound"  # outbound | inbound
	service_request_id: str | None = None
	created_by: str


class CommunicationResponse(CommunicationCreate):
	id: str = Field(default_factory=uuid7str)
	sent_at: datetime | None = None
	delivered: bool = False
	read_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Satisfaction Survey ───────────────────────────────────────────────────────

class SatisfactionSurveyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	tenant_entity_id: str
	property_id: str
	survey_period: str  # YYYY-MM or YYYY-Q1
	ratings: dict[str, int]  # dimension -> 1-5
	comments: str | None = None
	created_by: str

	@field_validator("ratings")
	@classmethod
	def _valid_ratings(cls, v: dict[str, int]) -> dict[str, int]:
		for dim, rating in v.items():
			if rating not in [1, 2, 3, 4, 5]:
				raise ValueError(f"rating for {dim} must be 1-5")
		return v


class SatisfactionSurveyResponse(SatisfactionSurveyCreate):
	id: str = Field(default_factory=uuid7str)
	average_score: Decimal = Decimal("0")
	score_below_threshold: bool = False
	review_triggered: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Tenant Score ──────────────────────────────────────────────────────────────

class TenantScoreCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	tenant_entity_id: str
	model: str  # from SUPPORTED_SCORING_MODELS
	score: Decimal
	components: dict[str, Decimal] = Field(default_factory=dict)
	scored_by: str  # "system" or user_id

	@field_validator("score")
	@classmethod
	def _score_range(cls, v: Decimal) -> Decimal:
		if not (Decimal("0") <= v <= Decimal("100")):
			raise ValueError("score must be 0-100")
		return v


class TenantScoreResponse(TenantScoreCreate):
	id: str = Field(default_factory=uuid7str)
	retention_risk_flagged: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Escalation ────────────────────────────────────────────────────────────────

class TenantEscalationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	tenant_entity_id: str
	escalation_type: EscalationType
	description: str
	severity: str = "medium"
	service_request_id: str | None = None
	created_by: str


class TenantEscalationResponse(TenantEscalationCreate):
	id: str = Field(default_factory=uuid7str)
	status: str = "open"  # open | in_progress | resolved | closed
	assigned_to: str | None = None
	resolved_at: datetime | None = None
	resolution_notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
