"""Pydantic v2 models for Property Contracts (con)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


class ContractType(str, Enum):
	sale_purchase = "sale_purchase"
	management_contract = "management_contract"
	construction_contract = "construction_contract"
	service_agreement = "service_agreement"
	joint_venture = "joint_venture"
	development_agreement = "development_agreement"
	agency_agreement = "agency_agreement"
	facility_management = "facility_management"


class ContractStatus(str, Enum):
	draft = "draft"
	negotiating = "negotiating"
	pending_signature = "pending_signature"
	active = "active"
	suspended = "suspended"
	expired = "expired"
	terminated = "terminated"
	disputed = "disputed"


class PartyRole(str, Enum):
	buyer = "buyer"
	seller = "seller"
	landlord = "landlord"
	tenant = "tenant"
	developer = "developer"
	contractor = "contractor"
	subcontractor = "subcontractor"
	agent = "agent"
	managing_agent = "managing_agent"
	guarantor = "guarantor"


class MilestoneType(str, Enum):
	payment = "payment"
	handover = "handover"
	inspection = "inspection"
	approval = "approval"
	completion = "completion"
	possession = "possession"
	registration = "registration"
	defect_liability = "defect_liability"


class VariationType(str, Enum):
	price_adjustment = "price_adjustment"
	scope_change = "scope_change"
	timeline_extension = "timeline_extension"
	party_substitution = "party_substitution"
	clause_amendment = "clause_amendment"
	schedule_update = "schedule_update"


class DisputeType(str, Enum):
	payment_dispute = "payment_dispute"
	quality_dispute = "quality_dispute"
	delay_dispute = "delay_dispute"
	scope_dispute = "scope_dispute"
	termination_dispute = "termination_dispute"
	title_dispute = "title_dispute"


class ContractorGrade(str, Enum):
	preferred = "preferred"
	approved = "approved"
	conditional = "conditional"
	suspended = "suspended"
	blacklisted = "blacklisted"


class RetentionMethod(str, Enum):
	percentage = "percentage"
	fixed_amount = "fixed_amount"
	milestone_linked = "milestone_linked"
	performance_bond = "performance_bond"


# ── Contract Party ────────────────────────────────────────────────────────────

class ContractParty(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	party_id: str
	party_name: str
	role: PartyRole
	signature_method: str | None = None
	signed_at: datetime | None = None
	signature_ref: str | None = None


# ── Contract ──────────────────────────────────────────────────────────────────

class ContractCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	contract_ref: str
	contract_type: ContractType
	property_id: str | None = None
	parties: list[ContractParty]
	governing_law: str
	start_date: date
	end_date: date | None = None
	contract_value: Decimal | None = None
	currency: str = "KES"
	description: str
	created_by: str

	@field_validator("parties")
	@classmethod
	def _at_least_two_parties(cls, v: list[ContractParty]) -> list[ContractParty]:
		if len(v) < 2:
			raise ValueError("contract requires at least two parties")
		return v


class ContractResponse(ContractCreate):
	id: str = Field(default_factory=uuid7str)
	status: ContractStatus = ContractStatus.draft
	legal_review_complete: bool = False
	all_signatures_present: bool = False
	executed_at: datetime | None = None
	terminated_at: datetime | None = None
	termination_reason: str | None = None
	document_ids: list[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class ContractUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: ContractStatus | None = None
	end_date: date | None = None
	legal_review_complete: bool | None = None
	contract_value: Decimal | None = None


# ── Contractor ────────────────────────────────────────────────────────────────

class ContractorCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	name: str
	contractor_type: str
	registration_number: str | None = None
	email: str
	phone: str
	insurance_expiry: date | None = None
	grade: ContractorGrade = ContractorGrade.conditional
	specialisms: list[str] = Field(default_factory=list)
	created_by: str


class ContractorResponse(ContractorCreate):
	id: str = Field(default_factory=uuid7str)
	active_contracts: int = 0
	last_grading_review: date | None = None
	performance_score: Decimal | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class ContractorUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	grade: ContractorGrade | None = None
	insurance_expiry: date | None = None
	performance_score: Decimal | None = None


# ── Milestone ─────────────────────────────────────────────────────────────────

class MilestoneCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	contract_id: str
	milestone_type: MilestoneType
	title: str
	due_date: date
	amount: Decimal | None = None
	currency: str = "KES"
	description: str | None = None
	created_by: str


class MilestoneResponse(MilestoneCreate):
	id: str = Field(default_factory=uuid7str)
	status: str = "pending"  # pending | in_progress | completed | overdue | waived
	completed_at: datetime | None = None
	evidence_ids: list[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Variation Order ───────────────────────────────────────────────────────────

class VariationOrderCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	contract_id: str
	variation_type: VariationType
	description: str
	amount_change: Decimal = Decimal("0")
	currency: str = "KES"
	timeline_change_days: int = 0
	created_by: str

	@field_validator("timeline_change_days")
	@classmethod
	def _non_negative_days(cls, v: int) -> int:
		if v < 0:
			raise ValueError("timeline_change_days cannot be negative")
		return v


class VariationOrderResponse(VariationOrderCreate):
	id: str = Field(default_factory=uuid7str)
	ref: str = ""
	status: str = "draft"  # draft | submitted | approved | rejected | board_pending
	board_approved: bool = False
	approved_by: str | None = None
	approved_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Dispute ───────────────────────────────────────────────────────────────────

class DisputeCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	contract_id: str
	dispute_type: DisputeType
	description: str
	disputed_amount: Decimal | None = None
	currency: str = "KES"
	raised_by: str
	created_by: str


class DisputeResponse(DisputeCreate):
	id: str = Field(default_factory=uuid7str)
	status: str = "open"  # open | mediation | arbitration | litigation | resolved | withdrawn
	legal_review_obtained: bool = False
	resolved_at: datetime | None = None
	resolution_summary: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Retention ─────────────────────────────────────────────────────────────────

class RetentionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	contract_id: str
	method: RetentionMethod
	retention_percentage: Decimal | None = None  # for percentage method
	fixed_amount: Decimal | None = None  # for fixed_amount method
	defect_liability_end: date
	currency: str = "KES"
	created_by: str

	@field_validator("retention_percentage")
	@classmethod
	def _valid_pct(cls, v: Decimal | None) -> Decimal | None:
		if v is not None and not (Decimal("0") < v <= Decimal("100")):
			raise ValueError("retention_percentage must be between 0 and 100")
		return v


class RetentionResponse(RetentionCreate):
	id: str = Field(default_factory=uuid7str)
	amount_held: Decimal = Decimal("0")
	amount_released: Decimal = Decimal("0")
	defect_liability_cleared: bool = False
	release_approved_by: str | None = None
	released_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Clause ────────────────────────────────────────────────────────────────────

class ClauseCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	clause_type: str
	title: str
	content: str
	is_standard: bool = True
	tags: list[str] = Field(default_factory=list)
	created_by: str


class ClauseResponse(ClauseCreate):
	id: str = Field(default_factory=uuid7str)
	usage_count: int = 0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
