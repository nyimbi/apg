"""Flask-AppBuilder views and Pydantic v2 request/response models for FOSA."""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid4())


# ── Request Bodies ────────────────────────────────────────────────────────────

class OpenAccountRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	account_type: str  # CURRENT | SALARY | FIXED_DEPOSIT
	opening_balance: Decimal = Decimal("0")
	currency: str = "KES"
	account_name: str | None = None
	daily_withdrawal_limit: Decimal = Decimal("100000")
	daily_transfer_limit: Decimal = Decimal("200000")

	@field_validator("account_type")
	@classmethod
	def validate_type(cls, v: str) -> str:
		if v not in {"CURRENT", "SALARY", "FIXED_DEPOSIT"}:
			raise ValueError(f"invalid account_type: {v}")
		return v


class DepositRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	amount: Decimal
	channel: str  # TELLER | MPESA | BANK_TRANSFER
	reference: str
	depositor_name: str | None = None
	narration: str | None = None
	teller_id: str | None = None

	@field_validator("amount")
	@classmethod
	def positive(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("amount must be positive")
		return v


class WithdrawRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	amount: Decimal
	channel: str  # TELLER | ATM | MPESA
	reference: str | None = None
	authorized_by: str | None = None
	narration: str | None = None
	teller_id: str | None = None

	@field_validator("amount")
	@classmethod
	def positive(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("amount must be positive")
		return v


class BosaTransferRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	amount: Decimal
	bosa_account_id: str
	reference: str
	approved_by: str | None = None


class MpesaCashInRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	mpesa_reference: str
	amount: Decimal
	phone_number: str

	@field_validator("phone_number")
	@classmethod
	def validate_phone(cls, v: str) -> str:
		v = v.strip()
		if not v.startswith(("07", "01", "+254", "254")):
			raise ValueError("phone_number must be a valid Kenyan number")
		return v


class MpesaCashOutRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	amount: Decimal
	phone_number: str
	mpesa_reference: str | None = None


class AtmCardRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	account_id: str
	card_type: str  # VISA | MASTERCARD | PREPAID
	card_name: str | None = None

	@field_validator("card_type")
	@classmethod
	def validate_card_type(cls, v: str) -> str:
		if v not in {"VISA", "MASTERCARD", "PREPAID"}:
			raise ValueError(f"invalid card_type: {v}")
		return v


class DailyLimitRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	withdrawal_limit: Decimal
	transfer_limit: Decimal

	@field_validator("withdrawal_limit", "transfer_limit")
	@classmethod
	def non_negative(cls, v: Decimal) -> Decimal:
		if v < 0:
			raise ValueError("must be >= 0")
		return v


class StandingOrderRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	beneficiary_account: str
	beneficiary_name: str | None = None
	amount: Decimal
	frequency: str
	start_date: str
	end_date: str | None = None
	narration: str | None = None

	@field_validator("frequency")
	@classmethod
	def validate_freq(cls, v: str) -> str:
		valid = {"daily", "weekly", "biweekly", "monthly", "quarterly", "annually"}
		if v not in valid:
			raise ValueError(f"frequency must be one of {valid}")
		return v


class OverdraftRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	requested_amount: Decimal
	purpose: str

	@field_validator("requested_amount")
	@classmethod
	def positive(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("must be positive")
		return v


class OverdraftApproveRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	approved_amount: Decimal
	approved_by: str
	expiry_date: str


# ── Response Envelopes ────────────────────────────────────────────────────────

class BalanceResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	account_number: str
	book_balance: Decimal
	available_balance: Decimal
	locked_balance: Decimal
	overdraft_limit: Decimal
	overdraft_used: Decimal
	currency: str
	as_at: str


class TransactionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	account_id: str
	account_number: str
	member_id: str
	txn_type: str
	amount: Decimal
	balance_before: Decimal
	balance_after: Decimal
	channel: str | None = None
	reference: str | None = None
	narration: str | None = None
	status: str
	created_at: str


class StatementResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	account_number: str
	member_id: str
	currency: str
	from_date: str
	to_date: str
	opening_balance: Decimal
	closing_balance: Decimal
	total_credits: Decimal
	total_debits: Decimal
	transaction_count: int
	transactions: list[dict[str, Any]] = Field(default_factory=list)
	generated_at: str


class PortfolioResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tenant_id: str
	total_deposits: Decimal
	active_accounts: int
	dormant_accounts: int
	frozen_accounts: int
	closed_accounts: int
	daily_deposit_volume: Decimal
	daily_withdrawal_volume: Decimal
	total_overdraft_exposure: Decimal
	total_cards_issued: int
	active_standing_orders: int
	generated_at: str


class TellerSummaryResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	teller_id: str
	date: str
	opening_float: Decimal
	total_deposits: Decimal
	total_withdrawals: Decimal
	total_transactions: int
	closing_float: Decimal
	variance: Decimal


class ListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	items: list[dict[str, Any]] = Field(default_factory=list)
	total: int = 0


class ErrorResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	error: str
	detail: str | None = None
