"""Pydantic v2 models for SACCO Group Lending.

Covers: group registration, membership, contributions, group loans,
merry-go-round cycles, repayments, arrears, and performance scoring.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())


# ── Enumerations ──────────────────────────────────────────────────────────────

class GroupType(str, Enum):
	CHAMA = "CHAMA"
	WELFARE = "WELFARE"
	MERRY_GO_ROUND = "MERRY_GO_ROUND"
	INVESTMENT = "INVESTMENT"


class GroupRole(str, Enum):
	CHAIRPERSON = "CHAIRPERSON"
	SECRETARY = "SECRETARY"
	TREASURER = "TREASURER"
	MEMBER = "MEMBER"


class ContributionType(str, Enum):
	MONTHLY = "MONTHLY"
	MERRY_GO_ROUND = "MERRY_GO_ROUND"
	EMERGENCY = "EMERGENCY"


class GroupStatus(str, Enum):
	ACTIVE = "ACTIVE"
	SUSPENDED = "SUSPENDED"
	DISSOLVED = "DISSOLVED"


class GroupLoanStatus(str, Enum):
	PENDING = "PENDING"
	APPROVED = "APPROVED"
	DISBURSED = "DISBURSED"
	ACTIVE = "ACTIVE"
	ARREARS = "ARREARS"
	WRITTEN_OFF = "WRITTEN_OFF"
	CLOSED = "CLOSED"


class MeetingFrequency(str, Enum):
	WEEKLY = "WEEKLY"
	BIWEEKLY = "BIWEEKLY"
	MONTHLY = "MONTHLY"


# ── Group models ──────────────────────────────────────────────────────────────

class Group(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	group_type: GroupType
	registration_number: str | None = None
	status: GroupStatus = GroupStatus.ACTIVE
	meeting_day: str | None = None          # e.g. "Monday"
	meeting_frequency: MeetingFrequency = MeetingFrequency.MONTHLY
	chairperson_member_id: str | None = None
	secretary_member_id: str | None = None
	treasurer_member_id: str | None = None
	registered_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")
	updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")
	metadata: dict[str, Any] = Field(default_factory=dict)


class GroupMember(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	group_id: str
	member_id: str
	role: GroupRole = GroupRole.MEMBER
	joining_date: date
	exit_date: date | None = None
	initial_contribution: Decimal = Decimal("0")
	total_contributions: Decimal = Decimal("0")
	total_loan_share: Decimal = Decimal("0")       # share of outstanding group loans
	total_repaid: Decimal = Decimal("0")
	active: bool = True
	exit_reason: str | None = None
	payout_amount: Decimal | None = None
	merry_go_round_position: int | None = None    # 1-indexed rotation order
	merry_go_round_received: bool = False
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")


class MemberContributionLine(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	amount: Decimal


class GroupContribution(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	group_id: str
	meeting_date: date
	contribution_type: ContributionType
	total_amount: Decimal
	lines: list[MemberContributionLine] = Field(default_factory=list)
	recorded_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")
	recorded_by: str | None = None
	notes: str | None = None


# ── Group loan models ─────────────────────────────────────────────────────────

class DisbursementInstruction(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	amount: Decimal
	account_id: str


class GroupLoan(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	loan_number: str
	tenant_id: str
	group_id: str
	borrower_member_ids: list[str] = Field(default_factory=list)   # all joint borrowers
	requested_amount: Decimal
	approved_amount: Decimal | None = None
	disbursed_amount: Decimal | None = None
	outstanding_balance: Decimal = Decimal("0")
	purpose: str
	tenure_months: int
	status: GroupLoanStatus = GroupLoanStatus.PENDING
	applied_by: str
	applied_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")
	approved_by: str | None = None
	approved_at: str | None = None
	conditions: str | None = None
	disbursed_at: str | None = None
	disbursement_instructions: list[DisbursementInstruction] = Field(default_factory=list)
	# Per-member outstanding share tracking
	member_balances: dict[str, Decimal] = Field(default_factory=dict)
	member_repaid: dict[str, Decimal] = Field(default_factory=dict)
	closed_at: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class MemberRepaymentLine(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	amount: Decimal


class GroupRepayment(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	loan_id: str
	group_id: str
	total_amount: Decimal
	payment_date: date
	payment_ref: str
	member_contributions: list[MemberRepaymentLine] = Field(default_factory=list)
	recorded_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")
	notes: str | None = None


# ── Merry-go-round models ─────────────────────────────────────────────────────

class MerryGoRoundRound(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	group_id: str
	round_number: int
	round_date: date
	beneficiary_member_id: str
	total_collected: Decimal
	contributor_lines: list[MemberContributionLine] = Field(default_factory=list)
	recorded_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")


# ── Arrears / liability models ────────────────────────────────────────────────

class MemberArrearsPosition(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	loan_share: Decimal
	total_repaid: Decimal
	arrears_amount: Decimal
	last_payment_date: date | None = None
	is_defaulting: bool = False


class GroupArrearsPosition(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	loan_id: str
	group_id: str
	as_of_date: date
	total_outstanding: Decimal
	total_arrears: Decimal
	arrears_rate_pct: Decimal
	days_in_arrears: int
	member_positions: list[MemberArrearsPosition] = Field(default_factory=list)
	defaulting_member_ids: list[str] = Field(default_factory=list)


# ── Performance / reporting models ───────────────────────────────────────────

class MerryGoRoundScheduleEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	position: int
	member_id: str
	scheduled_date: date | None = None
	has_received: bool = False


class MerryGoRoundResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	round_id: str
	group_id: str
	beneficiary_member_id: str
	total_collected: Decimal
	round_date: date
	contributor_count: int
	next_beneficiary_member_id: str | None = None


class GroupPerformanceScore(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	group_id: str
	tenant_id: str
	score: int                       # 0-100
	grade: str                       # A/B/C/D/E
	repayment_rate_pct: Decimal
	contribution_compliance_pct: Decimal
	loan_count: int
	active_loan_count: int
	total_saved: Decimal
	computed_at: str


class GroupSavingsSummary(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	group_id: str
	total_savings: Decimal
	per_member: list[dict[str, Any]]


class GroupStatementEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	entry_date: date
	entry_type: str           # CONTRIBUTION | LOAN_DISBURSEMENT | REPAYMENT | MGR_DISBURSEMENT
	reference: str
	amount: Decimal
	running_balance: Decimal
	description: str
	member_id: str | None = None
