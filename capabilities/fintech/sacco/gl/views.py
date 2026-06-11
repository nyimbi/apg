"""Pydantic v2 request/response models for SACCO GL API views.

These are the API-layer contracts — thin wrappers with validation that
translate between HTTP JSON and service calls.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, AfterValidator

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())


def _positive(v: Decimal) -> Decimal:
	if v <= Decimal("0"):
		raise ValueError("must be > 0")
	return v


PositiveDecimal = Annotated[Decimal, AfterValidator(_positive)]


class DepositRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	account_type: str = "FOSA"  # FOSA | BOSA
	amount: PositiveDecimal
	channel: str = "cash"
	value_date: str | None = None
	posted_by: str = "api"


class DisbursementRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	loan_id: str
	amount: PositiveDecimal
	loan_type: str = "BOSA"
	disbursement_channel: str = "savings_account"
	value_date: str | None = None
	posted_by: str = "api"


class RepaymentRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	loan_id: str
	principal: PositiveDecimal
	interest: Decimal = Decimal("0")
	penalty: Decimal = Decimal("0")
	payment_channel: str = "cash"
	value_date: str | None = None
	posted_by: str = "api"


class InterestRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_id: str
	amount: PositiveDecimal
	period: str  # YYYY-MM
	account_type: str = "BOSA"
	value_date: str | None = None
	posted_by: str = "api"


class DividendRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	amount: PositiveDecimal
	year: int
	pay_to_deposits: bool = False
	value_date: str | None = None
	posted_by: str = "api"


class SharePurchaseRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	amount: PositiveDecimal
	channel: str = "cash"
	value_date: str | None = None
	posted_by: str = "api"


class WithdrawalRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	member_id: str
	amount: PositiveDecimal
	account_type: str = "FOSA"
	channel: str = "cash"
	value_date: str | None = None
	posted_by: str = "api"


class ProvisionRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	loan_id: str
	provision_amount: PositiveDecimal
	value_date: str | None = None
	posted_by: str = "api"


class WriteOffRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	loan_id: str
	amount: PositiveDecimal
	loan_type: str = "BOSA"
	value_date: str | None = None
	posted_by: str = "api"


class PeriodRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	year: int
	month: int


class PeriodCloseRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	year: int
	month: int
	closed_by: str = "api"


class GenericTransactionRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	transaction_type: str
	entries: list[dict]  # [{account_code, debit, credit, narrative}]
	reference: str
	value_date: str
	posted_by: str = "api"
	narration: str = ""


# ── Response models ────────────────────────────────────────────────────────────

class TransactionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	reference: str
	transaction_type: str
	total_debit: str
	posted_at: str


class BalanceResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	account_code: str
	balance: str


class ValidationResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	balanced: bool
	difference: str
	total_debit: str
	total_credit: str
	unbalanced_entries: list[dict]
	entry_count: int
