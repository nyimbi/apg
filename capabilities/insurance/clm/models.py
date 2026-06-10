"""Pydantic v2 models for Claims Management (ins_clm)."""
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


class ClmFNOLCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	policy_id: str
	policy_number: str
	claimant_name: str
	claimant_id: str
	incident_date: date
	incident_description: str
	estimated_loss: Decimal
	currency: str = "KES"
	reported_by: str
	metadata: dict[str, Any] = Field(default_factory=dict)


class ClmClaimUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	status: str | None = None
	reserve_amount: Decimal | None = None
	assessor_id: str | None = None
	fraud_flag: bool | None = None
	metadata: dict[str, Any] | None = None


class ClmClaimResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	claim_number: str
	policy_id: str
	policy_number: str
	claimant_name: str
	claimant_id: str
	incident_date: date
	incident_description: str
	estimated_loss: Decimal
	reserve_amount: Decimal
	paid_amount: Decimal
	currency: str
	status: str
	assessor_id: str | None = None
	fraud_flag: bool
	tenant_id: str
	created_at: datetime
	metadata: dict[str, Any] = Field(default_factory=dict)


class ClmClaimList(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	items: list[ClmClaimResponse]
	total: int
	page: int = 1
	page_size: int = 50


class ClmClaimFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	status: str | None = None
	policy_id: str | None = None
	claimant_id: str | None = None
	fraud_flag: bool | None = None
	incident_date_from: date | None = None
	incident_date_to: date | None = None


class ClmReserveCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	claim_id: str
	reserve_amount: Decimal
	reserve_type: str = "outstanding"
	set_by: str
	justification: str


class ClmPaymentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	claim_id: str
	payment_amount: Decimal
	payment_type: str = "partial"
	payee_name: str
	payee_account: str
	payment_reference: str
	authorised_by: str


class ClmFraudAssessment(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	claim_id: str
	fraud_score: float
	indicators: list[str] = Field(default_factory=list)
	assessed_by: str
	recommendation: str


class ClmSubrogationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	claim_id: str
	third_party_name: str
	third_party_id: str
	recovery_amount: Decimal
	legal_reference: str | None = None


class ClmAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	actor: str | None = None
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime
