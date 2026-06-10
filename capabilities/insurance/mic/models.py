"""Pydantic v2 models for Micro-Insurance Platform (ins_mic)."""
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


class MicProductCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	product_code: str
	product_name: str
	product_type: str
	sum_insured: Decimal
	premium: Decimal
	currency: str = "KES"
	coverage_days: int
	ussd_menu_code: str
	airtime_deduction: bool = False
	mobile_money_payout: bool = True
	description: str = ""


class MicEnrolmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	msisdn: str
	product_code: str
	id_number: str | None = None
	name: str
	enrolment_channel: str = "ussd"
	payment_method: str = "airtime"
	metadata: dict[str, Any] = Field(default_factory=dict)


class MicEnrolmentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	policy_number: str
	msisdn: str
	product_code: str
	name: str
	status: str
	coverage_start: date
	coverage_end: date
	tenant_id: str
	created_at: datetime


class MicAirtimeDeduction(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	msisdn: str
	amount: Decimal
	deduction_reference: str
	operator: str
	status: str


class MicMobileMoneyPayout(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	claim_id: str
	msisdn: str
	amount: Decimal
	mobile_money_reference: str
	operator: str
	status: str


class MicUSSDSession(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	session_id: str
	msisdn: str
	service_code: str
	input_text: str
	step: int


class MicClaimCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	policy_number: str
	msisdn: str
	incident_description: str
	claimed_amount: Decimal


class MicAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime
