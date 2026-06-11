"""Pydantic v2 models for SACCO Guarantor Management."""
from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid_extensions import uuid7str
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())


class GuaranteeStatus(str, Enum):
	PENDING    = "pending"      # consent request sent, awaiting response
	ACCEPTED   = "accepted"     # guarantor consented, savings frozen
	DECLINED   = "declined"     # guarantor refused
	CANCELLED  = "cancelled"    # request withdrawn before acceptance
	ACTIVE     = "active"       # guarantee in force on live loan
	CALLED     = "called"       # savings deducted due to default
	RELEASED   = "released"     # savings unfrozen, obligation ended
	SUBSTITUTED = "substituted" # replaced by another guarantor


class ReleaseReason(str, Enum):
	LOAN_REPAID   = "loan_repaid"
	LOAN_WRITTEN_OFF = "loan_written_off"
	SUBSTITUTION  = "substitution"
	ADMIN_RELEASE = "admin_release"


class NoticeType(str, Enum):
	WARNING      = "warning"
	CALL_NOTICE  = "call_notice"
	RELEASE      = "release"


# ── Requests ──────────────────────────────────────────────────────────────────

class GuaranteeRequestCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	loan_id: str
	guarantor_member_id: str
	amount_to_guarantee: Decimal
	loan_applicant_message: str | None = None


class GuaranteeAcceptance(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	guarantee_request_id: str
	guarantor_member_id: str
	pin_verified: bool
	acceptance_notes: str | None = None


class GuaranteeDecline(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	guarantee_request_id: str
	guarantor_member_id: str
	decline_reason: str


class GuaranteeCancel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	guarantee_request_id: str
	cancelled_by: str
	reason: str


# ── Core Domain Models ────────────────────────────────────────────────────────

class GuaranteeRequest(BaseModel):
	"""Lifecycle record from consent-request through acceptance or decline."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	loan_id: str
	guarantor_member_id: str
	amount_to_guarantee: Decimal
	loan_applicant_message: str | None = None
	status: GuaranteeStatus = GuaranteeStatus.PENDING
	# acceptance fields
	pin_verified: bool = False
	acceptance_notes: str | None = None
	accepted_at: str | None = None
	# decline fields
	decline_reason: str | None = None
	declined_at: str | None = None
	# cancel fields
	cancelled_by: str | None = None
	cancel_reason: str | None = None
	cancelled_at: str | None = None
	# audit
	created_at: str = ""
	updated_at: str = ""


class ActiveGuarantee(BaseModel):
	"""A live guarantee record created after request acceptance."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	guarantee_request_id: str
	loan_id: str
	guarantor_member_id: str
	guaranteed_amount: Decimal
	frozen_amount: Decimal        # portion of savings currently frozen
	amount_called: Decimal = Decimal("0")
	status: GuaranteeStatus = GuaranteeStatus.ACTIVE
	release_reason: ReleaseReason | None = None
	released_at: str | None = None
	released_by: str | None = None
	called_at: str | None = None
	call_reason: str | None = None
	substituted_by: str | None = None  # new guarantor member_id
	notices_sent: list[str] = Field(default_factory=list)
	created_at: str = ""
	updated_at: str = ""


class GuarantorExposure(BaseModel):
	"""Snapshot of a member's total guarantor exposure."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	tenant_id: str
	total_guaranteed: Decimal         # sum of active guaranteed_amount
	frozen_savings: Decimal           # sum of currently frozen amounts
	active_guarantees: list[dict[str, Any]] = Field(default_factory=list)
	available_to_guarantee: Decimal   # free savings above minimum buffer
	max_exposure_limit: Decimal       # configured or default limit
	at_risk_amount: Decimal = Decimal("0")  # on loans with DPD > 30
	computed_at: str = ""


class EligibilityCheck(BaseModel):
	"""Result of a guarantor eligibility assessment."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	tenant_id: str
	amount_requested: Decimal
	eligible: bool
	reasons: list[str] = Field(default_factory=list)   # reasons why ineligible
	free_savings: Decimal
	current_exposure: Decimal
	max_exposure_limit: Decimal
	headroom: Decimal   # max_exposure_limit - current_exposure
	savings_cover_ratio: Decimal  # free_savings / amount_requested
	checked_at: str = ""


class ExposureLimitOverride(BaseModel):
	"""Manual override of a member's exposure cap."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	member_id: str
	limit: Decimal
	set_by: str
	created_at: str = ""


class GLEntry(BaseModel):
	"""Lightweight GL posting record for guarantee calls."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	guarantee_id: str
	loan_id: str
	guarantor_member_id: str
	amount: Decimal
	debit_account: str   # Guarantor Savings
	credit_account: str  # Loan Recovery
	narrative: str
	posted_at: str = ""


class GuaranteeNotice(BaseModel):
	"""Record of a notice dispatched to a guarantor."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	guarantee_id: str
	guarantor_member_id: str
	notice_type: NoticeType
	sent_at: str = ""
	channel: str = "sms"
	delivered: bool = False


class PortfolioMetrics(BaseModel):
	"""Aggregate guarantor portfolio statistics for a tenant."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tenant_id: str
	total_active_guarantees: int
	total_exposure: Decimal
	total_frozen_savings: Decimal
	total_called_amount: Decimal
	at_risk_count: int
	at_risk_exposure: Decimal
	avg_guarantees_per_loan: Decimal
	release_rate_pct: Decimal  # released / (released + active)
	call_rate_pct: Decimal     # called / total ever active
	computed_at: str = ""
