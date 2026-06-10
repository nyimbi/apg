"""Pydantic v2 models for Premium & Billing (ins_prm)."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
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


class PrmPremiumScheduleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	policy_id: str
	policy_number: str
	total_premium: Decimal
	frequency: str = "annual"
	currency: str = "KES"
	inception_date: date
	expiry_date: date
	instalment_count: int = 1


class PrmInstalmentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	schedule_id: str
	policy_id: str
	instalment_number: int
	due_date: date
	amount: Decimal
	currency: str
	status: str
	paid_at: datetime | None = None
	tenant_id: str
	created_at: datetime


class PrmCollectionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	instalment_id: str
	payment_method: str
	payment_reference: str
	amount: Decimal
	collected_by: str


class PrmRefundCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	policy_id: str
	refund_amount: Decimal
	reason: str
	payee_account: str
	authorised_by: str


class PrmReconciliationRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	period_start: date
	period_end: date
	reconciled_by: str


class PrmScheduleFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	policy_id: str | None = None
	frequency: str | None = None
	status: str | None = None


class PrmAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime
