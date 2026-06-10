"""ADR / Dispute Resolution — Pydantic v2 models."""
from __future__ import annotations

from typing import Any
from uuid_extensions import uuid7str
from pydantic import BaseModel, ConfigDict, Field


class AdrCaseCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	title: str
	case_type: str  # arbitration, mediation, conciliation, expert_determination
	claimant_id: str
	respondent_id: str
	counsel_ids: list[str] = Field(default_factory=list)
	claim_amount: float | None = None
	currency: str = "KES"
	seat: str  # Nairobi, London, Singapore, ICC_Paris
	governing_law: str = ""
	rules: str = ""  # UNCITRAL, ICC, LCIA, Nairobi_Centre
	description: str = ""
	filed_date: str = ""
	tags: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)


class AdrCaseUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	title: str | None = None
	status: str | None = None
	claim_amount: float | None = None
	description: str | None = None
	tags: list[str] | None = None
	metadata: dict[str, Any] | None = None


class AdrCaseResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	title: str
	case_number: str
	case_type: str
	claimant_id: str
	respondent_id: str
	counsel_ids: list[str]
	claim_amount: float | None
	currency: str
	seat: str
	governing_law: str
	rules: str
	description: str
	filed_date: str
	status: str  # filed, notice_served, panel_constituted, hearings, award_rendered, enforcement, closed
	arbitrator_ids: list[str]
	mediator_id: str | None
	proceeding_count: int
	tags: list[str]
	metadata: dict[str, Any]
	created_at: str
	updated_at: str | None = None


class AdrCaseListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	items: list[AdrCaseResponse]
	total: int
	page: int = 1
	page_size: int = 50


class AdrCaseFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	case_type: str | None = None
	status: str | None = None
	claimant_id: str | None = None
	respondent_id: str | None = None
	seat: str | None = None
	tags: list[str] | None = None


class AdrNeutralCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	case_id: str
	neutral_id: str
	role: str  # sole_arbitrator, presiding_arbitrator, co_arbitrator, mediator
	appointed_by: str  # claimant, respondent, institution, agreement
	appointment_date: str
	fee_rate: float = 0.0
	currency: str = "KES"


class AdrNeutralResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	case_id: str
	tenant_id: str
	neutral_id: str
	role: str
	appointed_by: str
	appointment_date: str
	fee_rate: float
	currency: str
	status: str
	created_at: str


class AdrProceedingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	case_id: str
	proceeding_type: str  # hearing, conference, document_submission, inspection
	scheduled_date: str
	venue: str
	description: str
	presided_by_id: str = ""
	duration_hours: float = 0.0


class AdrProceedingResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	case_id: str
	tenant_id: str
	proceeding_type: str
	scheduled_date: str
	actual_date: str | None
	venue: str
	description: str
	presided_by_id: str
	duration_hours: float
	status: str
	minutes_reference: str | None = None
	created_at: str


class AdrAwardCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	case_id: str
	award_type: str  # final, partial, interim, consent, default
	award_date: str
	awarded_to_id: str
	award_amount: float | None = None
	currency: str = "KES"
	interest_rate: float = 0.0
	costs_awarded: float = 0.0
	summary: str
	full_text_reference: str = ""


class AdrAwardResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	case_id: str
	tenant_id: str
	award_type: str
	award_date: str
	awarded_to_id: str
	award_amount: float | None
	currency: str
	interest_rate: float
	costs_awarded: float
	summary: str
	full_text_reference: str
	status: str  # rendered, challenged, upheld, set_aside, enforced
	enforcement_status: str | None = None
	created_at: str


class AdrSettlementCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	case_id: str
	settlement_date: str
	settlement_amount: float
	currency: str = "KES"
	terms_summary: str
	signed_by_claimant_id: str
	signed_by_respondent_id: str
	confidentiality_clause: bool = True


class AdrSettlementResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	case_id: str
	tenant_id: str
	settlement_date: str
	settlement_amount: float
	currency: str
	terms_summary: str
	signed_by_claimant_id: str
	signed_by_respondent_id: str
	confidentiality_clause: bool
	status: str
	created_at: str


class AdrAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	case_id: str | None
	event_type: str
	actor_id: str | None
	details: dict[str, Any]
	created_at: str
