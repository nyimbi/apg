"""Pydantic v2 models for Employee Self-Service capability."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _uuid() -> str:
	return str(uuid4())


# ── Leave Request models ──────────────────────────────────────────────────────

class ESSLeaveRequestCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	employee_id: str
	leave_type: str  # annual, sick, maternity, paternity, compassionate, unpaid
	start_date: str
	end_date: str
	reason: str | None = None
	handover_to: str | None = None
	attachments: list[str] = Field(default_factory=list)


class ESSLeaveRequestUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	start_date: str | None = None
	end_date: str | None = None
	reason: str | None = None
	handover_to: str | None = None
	status: str | None = None  # pending, approved, rejected, cancelled
	approved_by: str | None = None
	rejection_reason: str | None = None


class ESSLeaveRequestResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	employee_id: str
	leave_type: str
	start_date: str
	end_date: str
	days_requested: float
	reason: str | None
	handover_to: str | None
	status: str
	approved_by: str | None
	rejection_reason: str | None
	attachments: list[str]
	created_at: str
	updated_at: str | None


class ESSLeaveRequestList(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	items: list[ESSLeaveRequestResponse]
	total: int
	page: int
	page_size: int


class ESSLeaveRequestFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	employee_id: str | None = None
	leave_type: str | None = None
	status: str | None = None
	start_date_from: str | None = None
	start_date_to: str | None = None


# ── Payslip models ────────────────────────────────────────────────────────────

class ESSPayslipResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	employee_id: str
	period_month: int
	period_year: int
	gross_pay: float
	deductions: float
	net_pay: float
	currency: str
	pay_date: str
	earnings_breakdown: dict[str, float]
	deductions_breakdown: dict[str, float]
	status: str
	created_at: str


# ── Expense Claim models ──────────────────────────────────────────────────────

class ESSExpenseClaimCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	employee_id: str
	category: str  # travel, meals, accommodation, supplies, other
	amount: float
	currency: str = "KES"
	expense_date: str
	description: str
	receipts: list[str] = Field(default_factory=list)
	project_code: str | None = None


class ESSExpenseClaimUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	category: str | None = None
	amount: float | None = None
	description: str | None = None
	status: str | None = None  # draft, submitted, approved, rejected, paid
	approved_by: str | None = None
	rejection_reason: str | None = None
	paid_at: str | None = None


class ESSExpenseClaimResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	employee_id: str
	category: str
	amount: float
	currency: str
	expense_date: str
	description: str
	receipts: list[str]
	project_code: str | None
	status: str
	approved_by: str | None
	rejection_reason: str | None
	paid_at: str | None
	created_at: str


# ── Benefits Enrolment models ─────────────────────────────────────────────────

class ESSBenefitEnrolmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	employee_id: str
	benefit_plan_id: str
	benefit_type: str  # medical, dental, pension, life_insurance, gym
	coverage_tier: str = "individual"  # individual, family, spouse
	effective_date: str
	dependants: list[dict[str, Any]] = Field(default_factory=list)


class ESSBenefitEnrolmentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	coverage_tier: str | None = None
	dependants: list[dict[str, Any]] | None = None
	status: str | None = None
	effective_date: str | None = None
	end_date: str | None = None


class ESSBenefitEnrolmentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	employee_id: str
	benefit_plan_id: str
	benefit_type: str
	coverage_tier: str
	effective_date: str
	end_date: str | None
	employee_contribution: float
	employer_contribution: float
	dependants: list[dict[str, Any]]
	status: str
	created_at: str


# ── Training Registration models ──────────────────────────────────────────────

class ESSTrainingRegistrationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	employee_id: str
	course_id: str
	course_name: str
	training_type: str  # internal, external, online, conference
	start_date: str
	end_date: str
	provider: str | None = None
	cost: float = 0.0
	currency: str = "KES"
	justification: str | None = None


class ESSTrainingRegistrationUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	status: str | None = None  # pending, approved, enrolled, completed, cancelled
	approved_by: str | None = None
	completion_date: str | None = None
	certificate_url: str | None = None
	score: float | None = None


class ESSTrainingRegistrationResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	employee_id: str
	course_id: str
	course_name: str
	training_type: str
	start_date: str
	end_date: str
	provider: str | None
	cost: float
	currency: str
	justification: str | None
	status: str
	approved_by: str | None
	completion_date: str | None
	certificate_url: str | None
	score: float | None
	created_at: str


# ── Personal Data models ──────────────────────────────────────────────────────

class ESSPersonalDataUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	phone: str | None = None
	emergency_contact_name: str | None = None
	emergency_contact_phone: str | None = None
	address_line1: str | None = None
	address_line2: str | None = None
	city: str | None = None
	county: str | None = None
	country: str | None = None
	bank_account_number: str | None = None
	bank_name: str | None = None
	bank_branch: str | None = None
	nssf_number: str | None = None
	nhif_number: str | None = None
	kra_pin: str | None = None


class ESSPersonalDataResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	employee_id: str
	full_name: str
	email: str
	phone: str | None
	emergency_contact_name: str | None
	emergency_contact_phone: str | None
	address_line1: str | None
	city: str | None
	county: str | None
	country: str | None
	bank_name: str | None
	nssf_number: str | None
	nhif_number: str | None
	kra_pin: str | None
	updated_at: str | None


# ── Audit model ───────────────────────────────────────────────────────────────

class ESSAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	event_type: str
	entity_type: str
	entity_id: str
	actor_id: str | None
	payload: dict[str, Any]
	emitted_at: str
