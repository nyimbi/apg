"""Pydantic v2 models for SACCO FOSA (Front Office Service Activity)."""
from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid4())


# ── Enums ─────────────────────────────────────────────────────────────────────

class AccountType(str, Enum):
	CURRENT = "CURRENT"
	SALARY = "SALARY"
	FIXED_DEPOSIT = "FIXED_DEPOSIT"

class AccountStatus(str, Enum):
	ACTIVE = "active"
	FROZEN = "frozen"
	DORMANT = "dormant"
	CLOSED = "closed"
	PENDING = "pending"

class DepositChannel(str, Enum):
	TELLER = "TELLER"
	MPESA = "MPESA"
	BANK_TRANSFER = "BANK_TRANSFER"

class WithdrawChannel(str, Enum):
	TELLER = "TELLER"
	ATM = "ATM"
	MPESA = "MPESA"

class CardType(str, Enum):
	VISA = "VISA"
	MASTERCARD = "MASTERCARD"
	PREPAID = "PREPAID"

class CardStatus(str, Enum):
	REQUESTED = "requested"
	ACTIVE = "active"
	BLOCKED = "blocked"
	EXPIRED = "expired"
	CANCELLED = "cancelled"

class StandingOrderFrequency(str, Enum):
	DAILY = "daily"
	WEEKLY = "weekly"
	BIWEEKLY = "biweekly"
	MONTHLY = "monthly"
	QUARTERLY = "quarterly"
	ANNUALLY = "annually"

class StandingOrderStatus(str, Enum):
	ACTIVE = "active"
	PAUSED = "paused"
	CANCELLED = "cancelled"
	COMPLETED = "completed"

class OverdraftStatus(str, Enum):
	REQUESTED = "requested"
	APPROVED = "approved"
	DECLINED = "declined"
	EXPIRED = "expired"
	CLEARED = "cleared"

class TransactionType(str, Enum):
	DEPOSIT = "fosa_deposit"
	WITHDRAWAL = "fosa_withdrawal"
	TRANSFER_IN = "fosa_transfer_in"
	TRANSFER_OUT = "fosa_transfer_out"
	MPESA_IN = "fosa_mpesa_in"
	MPESA_OUT = "fosa_mpesa_out"
	BOSA_IN = "fosa_bosa_in"
	BOSA_OUT = "fosa_bosa_out"
	STANDING_ORDER = "fosa_standing_order"
	INTEREST = "fosa_interest"
	CHARGE = "fosa_charge"
	REVERSAL = "fosa_reversal"


# ── Account ───────────────────────────────────────────────────────────────────

class FosaAccountOpenModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	account_type: AccountType
	opening_balance: Decimal = Decimal("0")
	currency: str = "KES"
	account_name: str | None = None
	daily_withdrawal_limit: Decimal = Decimal("100000")
	daily_transfer_limit: Decimal = Decimal("200000")
	overdraft_limit: Decimal = Decimal("0")

	@field_validator("opening_balance", "daily_withdrawal_limit", "daily_transfer_limit", "overdraft_limit")
	@classmethod
	def non_negative(cls, v: Decimal) -> Decimal:
		if v < Decimal("0"):
			raise ValueError("must be >= 0")
		return v


class FosaAccountCloseModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	reason: str
	closed_by: str


class FosaAccountResponseModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	account_number: str
	member_id: str
	account_type: str
	account_name: str
	currency: str
	book_balance: Decimal
	available_balance: Decimal
	locked_balance: Decimal
	overdraft_limit: Decimal
	overdraft_used: Decimal
	daily_withdrawal_limit: Decimal
	daily_transfer_limit: Decimal
	status: str
	created_at: str
	updated_at: str | None = None


# ── Transactions ──────────────────────────────────────────────────────────────

class DepositModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	amount: Decimal
	channel: DepositChannel
	reference: str
	depositor_name: str | None = None
	narration: str | None = None

	@field_validator("amount")
	@classmethod
	def positive(cls, v: Decimal) -> Decimal:
		if v <= Decimal("0"):
			raise ValueError("amount must be positive")
		return v


class WithdrawModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	amount: Decimal
	channel: WithdrawChannel
	reference: str | None = None
	authorized_by: str | None = None
	narration: str | None = None

	@field_validator("amount")
	@classmethod
	def positive(cls, v: Decimal) -> Decimal:
		if v <= Decimal("0"):
			raise ValueError("amount must be positive")
		return v


class TransferModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	amount: Decimal
	bosa_account_id: str
	reference: str
	approved_by: str | None = None

	@field_validator("amount")
	@classmethod
	def positive(cls, v: Decimal) -> Decimal:
		if v <= Decimal("0"):
			raise ValueError("amount must be positive")
		return v


class MpesaCashInModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	mpesa_reference: str
	amount: Decimal
	phone_number: str

	@field_validator("amount")
	@classmethod
	def positive(cls, v: Decimal) -> Decimal:
		if v <= Decimal("0"):
			raise ValueError("amount must be positive")
		return v


class MpesaCashOutModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	amount: Decimal
	phone_number: str
	mpesa_reference: str | None = None

	@field_validator("amount")
	@classmethod
	def positive(cls, v: Decimal) -> Decimal:
		if v <= Decimal("0"):
			raise ValueError("amount must be positive")
		return v


# ── ATM Cards ─────────────────────────────────────────────────────────────────

class AtmCardIssueModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	account_id: str
	card_type: CardType
	card_name: str | None = None  # name to emboss on card


class AtmCardResponseModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	member_id: str
	account_id: str
	card_number_masked: str  # e.g. 4111 **** **** 1234
	card_type: str
	card_name: str
	status: str
	issued_at: str | None = None
	expires_at: str | None = None
	blocked_at: str | None = None
	block_reason: str | None = None


# ── Daily Limits ──────────────────────────────────────────────────────────────

class DailyLimitModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	withdrawal_limit: Decimal
	transfer_limit: Decimal

	@field_validator("withdrawal_limit", "transfer_limit")
	@classmethod
	def non_negative(cls, v: Decimal) -> Decimal:
		if v < Decimal("0"):
			raise ValueError("must be >= 0")
		return v


# ── Standing Orders ───────────────────────────────────────────────────────────

class StandingOrderCreateModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	beneficiary_account: str
	beneficiary_name: str | None = None
	amount: Decimal
	frequency: StandingOrderFrequency
	start_date: str  # ISO date
	end_date: str | None = None
	narration: str | None = None

	@field_validator("amount")
	@classmethod
	def positive(cls, v: Decimal) -> Decimal:
		if v <= Decimal("0"):
			raise ValueError("amount must be positive")
		return v


class StandingOrderResponseModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	account_id: str
	beneficiary_account: str
	beneficiary_name: str | None = None
	amount: Decimal
	frequency: str
	start_date: str
	end_date: str | None = None
	next_execution_date: str | None = None
	last_executed_at: str | None = None
	execution_count: int
	status: str
	narration: str | None = None
	created_at: str


# ── Overdraft ─────────────────────────────────────────────────────────────────

class OverdraftRequestModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	requested_amount: Decimal
	purpose: str

	@field_validator("requested_amount")
	@classmethod
	def positive(cls, v: Decimal) -> Decimal:
		if v <= Decimal("0"):
			raise ValueError("must be positive")
		return v


class OverdraftApproveModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	approved_amount: Decimal
	approved_by: str
	expiry_date: str  # ISO date

	@field_validator("approved_amount")
	@classmethod
	def positive(cls, v: Decimal) -> Decimal:
		if v <= Decimal("0"):
			raise ValueError("must be positive")
		return v


# ── Balance ───────────────────────────────────────────────────────────────────

class BalanceResponseModel(BaseModel):
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


# ── Transaction record ────────────────────────────────────────────────────────

class TransactionResponseModel(BaseModel):
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
	created_at: str
	status: str


# ── GL Entry ──────────────────────────────────────────────────────────────────

class GLEntryModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	account_code: str
	account_name: str
	debit: Decimal = Decimal("0")
	credit: Decimal = Decimal("0")
	narration: str
	reference: str
	posting_date: str
	created_at: str


# ── Teller Summary ────────────────────────────────────────────────────────────

class TellerSummaryModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	teller_id: str
	date: str
	opening_float: Decimal
	total_deposits: Decimal
	total_withdrawals: Decimal
	total_transactions: int
	closing_float: Decimal
	variance: Decimal


# ── Portfolio ─────────────────────────────────────────────────────────────────

class FosaPortfolioModel(BaseModel):
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
