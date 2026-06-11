"""Pydantic v2 models for SACCO Check-off Management."""
from __future__ import annotations

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


# ── Enums ─────────────────────────────────────────────────────────────────────

class DeductionFrequency(str, Enum):
	MONTHLY = "monthly"
	FORTNIGHTLY = "fortnightly"


class CheckOffStatus(str, Enum):
	PENDING = "pending"           # schedule generated, awaiting upload
	UPLOADED = "uploaded"         # employer file received
	RECONCILED = "reconciled"     # expected vs received matched
	POSTED = "posted"             # GL entries written, receipts credited
	SHORT_PAID = "short_paid"     # employer remitted less than expected
	OVER_PAID = "over_paid"       # employer remitted more than expected
	DEFAULTED = "defaulted"       # employer failed to remit
	DEMAND_ISSUED = "demand_issued"  # formal demand notice sent


class DeductionType(str, Enum):
	LOAN_PRINCIPAL = "loan_principal"
	LOAN_INTEREST = "loan_interest"
	LOAN_PENALTY = "loan_penalty"
	SAVINGS_REGULAR = "savings_regular"
	SAVINGS_SPECIAL = "savings_special"
	ARREARS = "arrears"


class RemittanceStatus(str, Enum):
	OUTSTANDING = "outstanding"
	PARTIAL = "partial"
	RECEIVED = "received"
	OVERDUE = "overdue"


# ── Employer ──────────────────────────────────────────────────────────────────

class EmployerCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	registration_number: str
	payroll_contact: str          # name or email of payroll officer
	remittance_account: str       # bank account / mpesa paybill
	check_off_agreement_date: str  # ISO date
	deduction_frequency: DeductionFrequency = DeductionFrequency.MONTHLY
	email: str | None = None
	phone: str | None = None
	address: str | None = None
	notes: str | None = None


class EmployerUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str | None = None
	payroll_contact: str | None = None
	remittance_account: str | None = None
	email: str | None = None
	phone: str | None = None
	address: str | None = None
	deduction_frequency: DeductionFrequency | None = None
	notes: str | None = None


class Employer(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	registration_number: str
	payroll_contact: str
	remittance_account: str
	check_off_agreement_date: str
	deduction_frequency: DeductionFrequency
	email: str | None = None
	phone: str | None = None
	address: str | None = None
	notes: str | None = None
	is_active: bool = True
	deactivation_reason: str | None = None
	deactivated_at: str | None = None
	member_count: int = 0
	created_at: str = ""
	updated_at: str = ""


# ── Member ↔ Employer Link ────────────────────────────────────────────────────

class MemberEmployerLink(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	member_id: str
	employer_id: str
	employee_number: str
	basic_salary: Decimal
	effective_date: str        # ISO date link became active
	end_date: str | None = None
	end_reason: str | None = None
	is_active: bool = True
	created_at: str = ""


# ── Deduction Line ────────────────────────────────────────────────────────────

class DeductionLine(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	deduction_type: DeductionType
	reference_id: str             # loan_id or savings_product_id
	description: str
	amount_due: Decimal
	amount_received: Decimal = Decimal("0")
	variance: Decimal = Decimal("0")   # received - due (negative = short)


class MemberDeductions(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	employer_id: str
	employer_name: str
	employee_number: str
	basic_salary: Decimal
	loan_deductions: list[DeductionLine] = Field(default_factory=list)
	savings_deductions: list[DeductionLine] = Field(default_factory=list)
	arrears_deductions: list[DeductionLine] = Field(default_factory=list)
	total_loan_deductions: Decimal = Decimal("0")
	total_savings_deductions: Decimal = Decimal("0")
	total_arrears: Decimal = Decimal("0")
	total_deductions: Decimal = Decimal("0")


# ── Check-off Schedule ────────────────────────────────────────────────────────

class ScheduleMemberEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	employee_number: str
	member_name: str
	basic_salary: Decimal
	deductions: list[DeductionLine] = Field(default_factory=list)
	total_loan: Decimal = Decimal("0")
	total_savings: Decimal = Decimal("0")
	total_arrears: Decimal = Decimal("0")
	total_deduction: Decimal = Decimal("0")


class CheckOffSchedule(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	employer_id: str
	employer_name: str
	payroll_month: int
	payroll_year: int
	period_label: str          # e.g. "June 2026"
	members: list[ScheduleMemberEntry] = Field(default_factory=list)
	total_members: int = 0
	grand_total_loan: Decimal = Decimal("0")
	grand_total_savings: Decimal = Decimal("0")
	grand_total_arrears: Decimal = Decimal("0")
	grand_total: Decimal = Decimal("0")
	status: CheckOffStatus = CheckOffStatus.PENDING
	generated_at: str = ""


# ── Uploaded Deduction (from employer payroll file) ───────────────────────────

class UploadedDeduction(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	amount_received: Decimal
	loan_deductions: Decimal = Decimal("0")
	savings_deductions: Decimal = Decimal("0")
	employee_number: str | None = None
	remarks: str | None = None


# ── Reconciliation ────────────────────────────────────────────────────────────

class MemberReconciliation(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	employee_number: str
	expected_total: Decimal
	received_total: Decimal
	variance: Decimal           # received - expected (negative = short)
	is_fully_paid: bool
	loan_variance: Decimal = Decimal("0")
	savings_variance: Decimal = Decimal("0")


class ReconciliationResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	employer_id: str
	employer_name: str
	payroll_month: int
	payroll_year: int
	status: CheckOffStatus
	members: list[MemberReconciliation] = Field(default_factory=list)
	total_expected: Decimal = Decimal("0")
	total_received: Decimal = Decimal("0")
	total_variance: Decimal = Decimal("0")
	short_paying_members: list[str] = Field(default_factory=list)
	over_paying_members: list[str] = Field(default_factory=list)
	demand_notice_required: bool = False
	excess_to_savings: Decimal = Decimal("0")
	reconciled_at: str = ""


# ── Remittance Record ─────────────────────────────────────────────────────────

class RemittanceRecord(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	employer_id: str
	employer_name: str
	payroll_month: int
	payroll_year: int
	period_label: str
	amount_expected: Decimal
	amount_received: Decimal = Decimal("0")
	amount_posted: Decimal = Decimal("0")
	status: RemittanceStatus = RemittanceStatus.OUTSTANDING
	check_off_status: CheckOffStatus = CheckOffStatus.PENDING
	schedule_id: str | None = None
	reconciliation_id: str | None = None
	upload_at: str | None = None
	reconciled_at: str | None = None
	posted_at: str | None = None
	reminders_sent: int = 0
	last_reminder_at: str | None = None
	defaulted: bool = False
	defaulted_at: str | None = None
	created_at: str = ""


# ── GL Entry ──────────────────────────────────────────────────────────────────

class GLEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	entry_date: str
	description: str
	debit_account: str
	credit_account: str
	amount: Decimal
	member_id: str
	employer_id: str
	reference: str             # "CKF-{employer_id}-{year}-{month}"
	deduction_type: DeductionType
	created_at: str = ""


# ── Employer Statement Line ───────────────────────────────────────────────────

class StatementLine(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	period_label: str
	payroll_month: int
	payroll_year: int
	amount_expected: Decimal
	amount_received: Decimal
	variance: Decimal
	status: CheckOffStatus
	posted_at: str | None = None


class EmployerStatement(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	employer_id: str
	employer_name: str
	tenant_id: str
	from_period: str
	to_period: str
	lines: list[StatementLine] = Field(default_factory=list)
	total_expected: Decimal = Decimal("0")
	total_received: Decimal = Decimal("0")
	total_variance: Decimal = Decimal("0")
	generated_at: str = ""


# ── Member Check-off History ──────────────────────────────────────────────────

class MemberCheckOffEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	period_label: str
	payroll_month: int
	payroll_year: int
	employer_name: str
	loan_deducted: Decimal
	savings_deducted: Decimal
	total_deducted: Decimal
	status: CheckOffStatus


class MemberCheckOffHistory(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	tenant_id: str
	employer_id: str | None = None
	employer_name: str | None = None
	entries: list[MemberCheckOffEntry] = Field(default_factory=list)
	total_loan_deducted: Decimal = Decimal("0")
	total_savings_deducted: Decimal = Decimal("0")
	months_covered: int = 0


# ── Metrics ───────────────────────────────────────────────────────────────────

class CheckOffMetrics(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tenant_id: str
	period_label: str
	total_employers: int = 0
	active_employers: int = 0
	defaulted_employers: int = 0
	total_members_on_checkoff: int = 0
	collection_rate_pct: Decimal = Decimal("0")   # % of expected collected
	compliance_rate_pct: Decimal = Decimal("0")   # % employers fully compliant
	total_expected: Decimal = Decimal("0")
	total_collected: Decimal = Decimal("0")
	total_outstanding: Decimal = Decimal("0")
	employers_short_paying: int = 0
	employers_over_paying: int = 0
	computed_at: str = ""
