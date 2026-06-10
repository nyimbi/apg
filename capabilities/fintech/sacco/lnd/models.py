"""Pydantic v2 models for SACCO Lending."""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid_extensions import uuid7str
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())


# ── Loan Product ──────────────────────────────────────────────────────────────

class LoanProductCreateModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	product_code: str
	product_name: str
	product_type: str  # development | emergency | school_fees | business | mortgage | asset
	interest_rate_pa: Decimal
	interest_method: str = "reducing_balance"  # reducing_balance | flat_rate
	min_amount: Decimal
	max_amount: Decimal
	min_term_months: int
	max_term_months: int
	max_multiplier: Decimal = Decimal("3")  # max loan = multiplier × savings
	grace_period_months: int = 0
	processing_fee_pct: Decimal = Decimal("0")
	insurance_fee_pct: Decimal = Decimal("0")
	min_guarantors: int = 2
	requires_collateral: bool = False
	description: str | None = None


class LoanProductUpdateModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	interest_rate_pa: Decimal | None = None
	max_amount: Decimal | None = None
	max_multiplier: Decimal | None = None
	is_active: bool | None = None
	description: str | None = None


class LoanProductResponseModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	product_code: str
	product_name: str
	product_type: str
	interest_rate_pa: Decimal
	interest_method: str
	min_amount: Decimal
	max_amount: Decimal
	min_term_months: int
	max_term_months: int
	max_multiplier: Decimal
	is_active: bool
	created_at: str


# ── Loan Application ──────────────────────────────────────────────────────────

class LoanApplicationModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	product_id: str
	amount_requested: Decimal
	term_months: int
	purpose: str
	guarantor_ids: list[str] = Field(default_factory=list)
	collateral_description: str | None = None
	collateral_value: Decimal | None = None


class LoanApprovalModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	loan_id: str
	approved_amount: Decimal
	approved_term_months: int
	approved_rate: Decimal | None = None
	approved_by: str
	approval_notes: str | None = None
	conditions: list[str] = Field(default_factory=list)


class LoanDisbursementModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	loan_id: str
	disbursement_method: str  # cash | mpesa | bank_transfer | savings_account
	disbursement_reference: str
	disbursed_by: str
	disbursement_account: str | None = None


# ── Repayment ─────────────────────────────────────────────────────────────────

class RepaymentModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	loan_id: str
	amount: Decimal
	payment_reference: str
	payment_method: str = "cash"
	payment_date: str | None = None
	recorded_by: str


class RepaymentScheduleModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	loan_id: str
	installments: list[dict[str, Any]] = Field(default_factory=list)
	total_principal: Decimal
	total_interest: Decimal
	total_payable: Decimal


# ── Credit Scoring ────────────────────────────────────────────────────────────

class CreditScoreModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	member_id: str
	tenant_id: str
	score: int  # 0-1000
	grade: str  # A | B | C | D | E
	max_loan_amount: Decimal
	factors: dict[str, Any] = Field(default_factory=dict)
	valid_until: str
	created_at: str


# ── CRB Reporting ─────────────────────────────────────────────────────────────

class CRBReportModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	report_type: str  # listing | delisting | inquiry
	reason: str
	reported_by: str
	crb_reference: str | None = None


# ── Filter & Audit ────────────────────────────────────────────────────────────

class LoanFilterModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str | None = None
	product_id: str | None = None
	status: str | None = None
	from_date: str | None = None
	to_date: str | None = None
	min_arrears_days: int | None = None


class LoanAuditModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	loan_id: str | None = None
	member_id: str | None = None
	amount: Decimal | None = None
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: str


class LoanListModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	items: list[dict[str, Any]] = Field(default_factory=list)
	total: int = 0
	page: int = 1
	page_size: int = 50
