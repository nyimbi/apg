"""
Bank Account Management — Flask-AppBuilder views and Pydantic v2 request/response models.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

try:
	from .models import (
		AccountStatus, AccountType, TransactionType, SigningAuthority,
		StatementFormat, uuid7str,
	)
except ImportError:
	from models import (  # type: ignore
		AccountStatus, AccountType, TransactionType, SigningAuthority,
		StatementFormat, uuid7str,
	)

_CFG = ConfigDict(
	extra="forbid",
	validate_by_name=True,
	validate_by_alias=True,
	populate_by_name=True,
)


def _non_empty(v: str) -> str:
	if not v or not v.strip():
		raise ValueError("must be non-empty")
	return v.strip()


def _positive_decimal(v: Decimal) -> Decimal:
	if v <= Decimal("0"):
		raise ValueError("must be positive")
	return v


def _non_negative_decimal(v: Decimal) -> Decimal:
	if v < Decimal("0"):
		raise ValueError("must be non-negative")
	return v


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]
PositiveDecimal = Annotated[Decimal, AfterValidator(_positive_decimal)]
NonNegativeDecimal = Annotated[Decimal, AfterValidator(_non_negative_decimal)]


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class OpenAccountView(BaseModel):
	model_config = _CFG

	customer_id: NonEmptyStr
	product_code: NonEmptyStr
	currency: NonEmptyStr
	account_number: str | None = None
	opening_deposit: NonNegativeDecimal | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class CloseAccountView(BaseModel):
	model_config = _CFG

	reason: NonEmptyStr
	closed_by: NonEmptyStr


class FreezeAccountView(BaseModel):
	model_config = _CFG

	reason: NonEmptyStr
	frozen_by: NonEmptyStr


class UnfreezeAccountView(BaseModel):
	model_config = _CFG

	reason: NonEmptyStr
	unfrozen_by: NonEmptyStr


class CreditView(BaseModel):
	model_config = _CFG

	amount: PositiveDecimal
	currency: NonEmptyStr
	reference: NonEmptyStr
	description: NonEmptyStr
	transaction_type: TransactionType = TransactionType.DEPOSIT
	value_date: date | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class DebitView(BaseModel):
	model_config = _CFG

	amount: PositiveDecimal
	currency: NonEmptyStr
	reference: NonEmptyStr
	description: NonEmptyStr
	transaction_type: TransactionType = TransactionType.WITHDRAWAL
	value_date: date | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class TransferView(BaseModel):
	model_config = _CFG

	to_account_id: NonEmptyStr
	amount: PositiveDecimal
	reference: NonEmptyStr
	description: NonEmptyStr
	value_date: date | None = None


class LockFundsView(BaseModel):
	model_config = _CFG

	amount: PositiveDecimal
	lock_reference: NonEmptyStr
	reason: str | None = None
	expires_at: datetime | None = None


class ReleaseLockView(BaseModel):
	model_config = _CFG

	lock_reference: NonEmptyStr


class SetOverdraftView(BaseModel):
	model_config = _CFG

	limit: NonNegativeDecimal
	approved_by: NonEmptyStr


class BulkCreditItemView(BaseModel):
	model_config = _CFG

	account_id: NonEmptyStr
	amount: PositiveDecimal
	reference: NonEmptyStr
	description: str = ""
	metadata: dict[str, Any] = Field(default_factory=dict)


class BulkCreditView(BaseModel):
	model_config = _CFG

	credits: list[BulkCreditItemView] = Field(min_length=1)


class StatementView(BaseModel):
	model_config = _CFG

	from_date: date
	to_date: date
	format: StatementFormat = StatementFormat.JSON


class AddSignatoryView(BaseModel):
	model_config = _CFG

	customer_id: NonEmptyStr
	signing_authority: SigningAuthority = SigningAuthority.SINGLE


class SweepView(BaseModel):
	model_config = _CFG

	linked_account_id: NonEmptyStr
	sweep_threshold: NonNegativeDecimal = Decimal("10000")
	retain_amount: NonNegativeDecimal = Decimal("5000")


class LinkProductView(BaseModel):
	model_config = _CFG

	product_code: NonEmptyStr


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------

class ApiResponse(BaseModel):
	model_config = ConfigDict(extra="allow")

	data: Any = None
	error: dict[str, Any] | None = None
	meta: dict[str, Any] | None = None
