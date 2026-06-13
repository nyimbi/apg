"""Pydantic v2 models for APG Chama & ROSCA Engine.

All model names carry the 'Ch' prefix per APG naming convention.
IDs use uuid7str for time-ordered UUIDs.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# uuid7 shim — falls back gracefully if uuid6 not installed in this env
# ---------------------------------------------------------------------------
try:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())
except ImportError:  # pragma: no cover
	import uuid

	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ChGroupType(str, Enum):
	CHAMA = "chama"
	ROSCA = "rosca"
	TABLE_BANKING = "table_banking"


class ChFrequency(str, Enum):
	WEEKLY = "weekly"
	BIWEEKLY = "biweekly"
	MONTHLY = "monthly"
	QUARTERLY = "quarterly"


class ChContributionStatus(str, Enum):
	PENDING = "pending"
	PAID = "paid"
	OVERDUE = "overdue"
	WAIVED = "waived"
	PARTIAL = "partial"


class ChPayoutStatus(str, Enum):
	PENDING = "pending"
	PROCESSING = "processing"
	DISBURSED = "disbursed"
	FAILED = "failed"
	REVERSED = "reversed"


class ChLoanStatus(str, Enum):
	PENDING_APPROVAL = "pending_approval"
	APPROVED = "approved"
	ACTIVE = "active"
	FULLY_REPAID = "fully_repaid"
	DEFAULTED = "defaulted"
	WRITTEN_OFF = "written_off"


class ChPaymentMethod(str, Enum):
	MPESA = "mpesa"
	AIRTEL_MONEY = "airtel_money"
	BANK_TRANSFER = "bank_transfer"
	CASH = "cash"
	EQUITY_EAZZY = "equity_eazzy"
	KCB_MOBI = "kcb_mobi"


class ChCycleStatus(str, Enum):
	ACTIVE = "active"
	COMPLETED = "completed"
	SKIPPED = "skipped"
	SUSPENDED = "suspended"


class ChMeetingType(str, Enum):
	REGULAR = "regular"
	SPECIAL = "special"
	AGM = "agm"
	EMERGENCY = "emergency"


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _positive_decimal(v: Decimal) -> Decimal:
	assert v > 0, "amount must be positive"
	return v


def _non_negative_decimal(v: Decimal) -> Decimal:
	assert v >= 0, "amount must be non-negative"
	return v


def _non_empty_string(v: str) -> str:
	assert v and v.strip(), "field must not be empty"
	return v.strip()


PositiveDecimal = Annotated[Decimal, AfterValidator(_positive_decimal)]
NonNegativeDecimal = Annotated[Decimal, AfterValidator(_non_negative_decimal)]
NonEmptyStr = Annotated[str, AfterValidator(_non_empty_string)]


# ---------------------------------------------------------------------------
# Base model configuration
# ---------------------------------------------------------------------------

class ChamaBase(BaseModel):
	"""Shared Pydantic config for all Chama models."""
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
	)

	def to_dict(self) -> dict[str, Any]:
		return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class ChMemberRef(ChamaBase):
	"""Lightweight member reference embedded in other documents."""
	member_id: str
	name: str
	phone: str


class ChGroup(ChamaBase):
	"""Savings group — the primary organisational unit.

	Supports Chama (general savings), ROSCA (rotating payout),
	and Table Banking (savings + internal lending combined).
	"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	name: NonEmptyStr
	group_type: ChGroupType
	description: str = ""
	# Contribution rules
	contribution_amount: PositiveDecimal
	frequency: ChFrequency
	# Membership
	member_ids: list[str] = Field(default_factory=list)
	max_members: int = Field(default=100, ge=3)
	# ROSCA specific: ordered list of member_ids for rotation
	payout_rotation: list[str] = Field(default_factory=list)
	current_rotation_index: int = 0
	# Cycle tracking
	current_cycle_number: int = 1
	total_cycles_completed: int = 0
	# Registration / governance
	registration_number: str | None = None
	bank_account: str | None = None
	mpesa_paybill: str | None = None
	is_active: bool = True
	created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
	updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ChMember(ChamaBase):
	"""Individual member of a Chama/ROSCA group.

	KYC fields (national_id, phone) feed into fintech_kyc capability.
	"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	group_id: NonEmptyStr
	name: NonEmptyStr
	phone: NonEmptyStr
	national_id: str = ""
	email: str = ""
	# Contribution settings
	contribution_amount: PositiveDecimal  # may differ from group default
	payout_order: int = 0  # 1-based position in ROSCA rotation; 0 = not yet assigned
	# Financials (running totals for quick lookup)
	total_contributed: NonNegativeDecimal = Decimal("0")
	total_received_payouts: NonNegativeDecimal = Decimal("0")
	total_loans_outstanding: NonNegativeDecimal = Decimal("0")
	# Status
	is_active: bool = True
	joined_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
	kyc_verified: bool = False
	kyc_reference: str | None = None


class ChContribution(ChamaBase):
	"""Single contribution record from a member in a specific cycle."""
	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	group_id: NonEmptyStr
	member_id: NonEmptyStr
	cycle_number: int = Field(ge=1)
	amount: PositiveDecimal
	expected_amount: PositiveDecimal
	payment_method: ChPaymentMethod
	status: ChContributionStatus = ChContributionStatus.PAID
	# Payment reference (MPESA confirmation code, bank ref, etc.)
	payment_reference: str = ""
	mpesa_receipt: str | None = None
	notes: str = ""
	recorded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
	due_date: str | None = None
	paid_at: str | None = None


class ChPayout(ChamaBase):
	"""Payout disbursement to a member — the ROSCA core event."""
	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	group_id: NonEmptyStr
	cycle_id: NonEmptyStr
	cycle_number: int = Field(ge=1)
	recipient_member_id: NonEmptyStr
	amount: PositiveDecimal
	payment_method: ChPaymentMethod = ChPaymentMethod.MPESA
	status: ChPayoutStatus = ChPayoutStatus.PENDING
	# MPESA disbursement
	mpesa_phone: str = ""
	mpesa_receipt: str | None = None
	disbursed_at: str | None = None
	failure_reason: str | None = None
	# Approvals
	approved_by: str | None = None
	approved_at: str | None = None
	created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ChCycle(ChamaBase):
	"""A single contribution + payout cycle for the group.

	Tracks who has contributed, total collected, and payout outcome.
	"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	group_id: NonEmptyStr
	cycle_number: int = Field(ge=1)
	status: ChCycleStatus = ChCycleStatus.ACTIVE
	# Payout recipient this cycle (ROSCA rotation determines this)
	payout_member_id: str | None = None
	# Financial summary
	expected_amount: NonNegativeDecimal = Decimal("0")  # sum of all member contributions expected
	collected_amount: NonNegativeDecimal = Decimal("0")
	payout_amount: NonNegativeDecimal = Decimal("0")
	# Member contribution tracking: {member_id: status}
	contribution_status: dict[str, str] = Field(default_factory=dict)
	# Dates
	start_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
	due_date: str | None = None
	completed_at: str | None = None
	payout_id: str | None = None


class ChTreasury(ChamaBase):
	"""Real-time treasury snapshot for a group.

	Updated atomically after each contribution, payout, loan disbursement,
	and loan repayment. The source of truth for available lending capital.
	"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	group_id: NonEmptyStr
	# Core balances
	total_savings: NonNegativeDecimal = Decimal("0")       # cumulative contributions
	total_payouts_disbursed: NonNegativeDecimal = Decimal("0")
	cash_balance: NonNegativeDecimal = Decimal("0")         # savings - payouts - loans + repayments
	total_loans_outstanding: NonNegativeDecimal = Decimal("0")
	total_loans_disbursed: NonNegativeDecimal = Decimal("0")
	# Income
	total_interest_income: NonNegativeDecimal = Decimal("0")
	total_penalty_income: NonNegativeDecimal = Decimal("0")
	# Meta
	member_count: int = 0
	active_loans: int = 0
	as_of: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ChLoan(ChamaBase):
	"""Loan from group treasury to a member — Table Banking core feature.

	Interest is simple interest. Repayment schedule is monthly instalments.
	"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	group_id: NonEmptyStr
	borrower_member_id: NonEmptyStr
	# Terms
	principal: PositiveDecimal
	interest_rate_monthly_pct: PositiveDecimal  # e.g. Decimal("5") = 5% per month
	repayment_months: int = Field(ge=1, le=24)
	# Calculated at approval
	total_interest: NonNegativeDecimal = Decimal("0")
	total_repayable: NonNegativeDecimal = Decimal("0")
	monthly_instalment: NonNegativeDecimal = Decimal("0")
	# Repayment tracking
	amount_repaid: NonNegativeDecimal = Decimal("0")
	outstanding_balance: NonNegativeDecimal = Decimal("0")
	# Guarantors
	guarantor_member_ids: list[str] = Field(default_factory=list)
	# Status lifecycle
	status: ChLoanStatus = ChLoanStatus.PENDING_APPROVAL
	# Payment
	payment_method: ChPaymentMethod = ChPaymentMethod.MPESA
	disbursement_reference: str | None = None
	# Approval
	approved_by: str | None = None
	approved_at: str | None = None
	disbursed_at: str | None = None
	fully_repaid_at: str | None = None
	default_noted_at: str | None = None
	notes: str = ""
	created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ChLoanRepayment(ChamaBase):
	"""Individual repayment instalment against a loan."""
	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	group_id: NonEmptyStr
	loan_id: NonEmptyStr
	member_id: NonEmptyStr
	amount: PositiveDecimal
	principal_portion: NonNegativeDecimal = Decimal("0")
	interest_portion: NonNegativeDecimal = Decimal("0")
	penalty_portion: NonNegativeDecimal = Decimal("0")
	payment_method: ChPaymentMethod
	payment_reference: str = ""
	mpesa_receipt: str | None = None
	balance_after: NonNegativeDecimal = Decimal("0")
	recorded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ChMeetingRecord(ChamaBase):
	"""Minutes and decisions from a group meeting.

	Captures quorum, agenda, resolutions, and financial snapshot at meeting time.
	"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	group_id: NonEmptyStr
	meeting_type: ChMeetingType = ChMeetingType.REGULAR
	meeting_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
	venue: str = ""
	# Attendance
	total_members: int = 0
	members_present: int = 0
	quorum_met: bool = False
	# Content
	agenda: list[str] = Field(default_factory=list)
	resolutions: list[str] = Field(default_factory=list)
	minutes_text: str = ""
	# Financial snapshot at meeting
	treasury_balance_at_meeting: NonNegativeDecimal = Decimal("0")
	# Officiation
	chairperson_id: str = ""
	secretary_id: str = ""
	recorded_by: str = ""
	created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
