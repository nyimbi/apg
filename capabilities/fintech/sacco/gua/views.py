"""Pydantic v2 request/response schemas for SACCO Guarantor Management API."""
from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

try:
	from uuid_extensions import uuid7str
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())


def _positive_decimal(v: Decimal) -> Decimal:
	assert v > 0, "must be positive"
	return v


def _non_negative_decimal(v: Decimal) -> Decimal:
	assert v >= 0, "must be non-negative"
	return v


PositiveDecimal = Annotated[Decimal, AfterValidator(_positive_decimal)]
NonNegativeDecimal = Annotated[Decimal, AfterValidator(_non_negative_decimal)]


# ── Request bodies ────────────────────────────────────────────────────────────

class RequestGuaranteeBody(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	loan_id: str
	guarantor_member_id: str
	amount_to_guarantee: PositiveDecimal
	loan_applicant_message: str | None = None


class AcceptGuaranteeBody(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	guarantor_member_id: str
	pin_verified: bool
	acceptance_notes: str | None = None


class DeclineGuaranteeBody(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	guarantor_member_id: str
	decline_reason: str


class CancelRequestBody(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	cancelled_by: str
	reason: str


class ReleaseGuaranteeBody(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	release_reason: str
	released_by: str = "api"


class SubstituteGuarantorBody(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	new_guarantor_id: str
	reason: str
	approved_by: str


class CallGuaranteeBody(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	amount_called: PositiveDecimal
	reason: str


class SendNoticeBody(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	notice_type: str  # warning | call_notice | release


class SetExposureLimitBody(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	limit: NonNegativeDecimal
	set_by: str


class EligibilityCheckBody(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	amount_to_guarantee: PositiveDecimal


# ── Response shapes ───────────────────────────────────────────────────────────

class GuaranteeRequestResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	loan_id: str
	guarantor_member_id: str
	amount_to_guarantee: Decimal
	status: str
	created_at: str
	updated_at: str


class ActiveGuaranteeResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	loan_id: str
	guarantor_member_id: str
	guaranteed_amount: Decimal
	frozen_amount: Decimal
	amount_called: Decimal
	status: str
	created_at: str


class ExposureResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	tenant_id: str
	total_guaranteed: Decimal
	frozen_savings: Decimal
	available_to_guarantee: Decimal
	max_exposure_limit: Decimal
	at_risk_amount: Decimal
	active_guarantees: list[dict[str, Any]] = Field(default_factory=list)
	computed_at: str


class EligibilityResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	eligible: bool
	reasons: list[str]
	free_savings: Decimal
	current_exposure: Decimal
	headroom: Decimal
	savings_cover_ratio: Decimal
	checked_at: str


class PortfolioMetricsResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tenant_id: str
	total_active_guarantees: int
	total_exposure: Decimal
	total_frozen_savings: Decimal
	total_called_amount: Decimal
	at_risk_count: int
	at_risk_exposure: Decimal
	release_rate_pct: Decimal
	call_rate_pct: Decimal
	computed_at: str


class ItemList(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	items: list[dict[str, Any]] = Field(default_factory=list)
	total: int = 0
