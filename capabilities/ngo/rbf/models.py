"""Results-Based Financing — Pydantic v2 models."""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid4())


_cfg = ConfigDict(extra="forbid", validate_by_name=True)


class RbfContractCreate(BaseModel):
	model_config = _cfg
	programme_id: str
	funder_reference: str
	title: str
	description: str = ""
	total_value: Decimal
	currency: str = "KES"
	start_date: str
	end_date: str
	payment_model: str = "output_based"
	contract_manager: str = ""


class RbfContractUpdate(BaseModel):
	model_config = _cfg
	title: str | None = None
	description: str | None = None
	end_date: str | None = None
	status: str | None = None
	contract_manager: str | None = None


class RbfContractResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	programme_id: str
	funder_reference: str
	title: str
	description: str
	total_value: Decimal
	paid_amount: Decimal = Decimal("0")
	currency: str
	start_date: str
	end_date: str
	payment_model: str
	contract_manager: str
	status: str
	tenant_id: str
	created_at: str
	updated_at: str | None = None


class RbfDliCreate(BaseModel):
	model_config = _cfg
	contract_id: str
	name: str
	description: str = ""
	indicator_code: str = ""
	target_value: float
	unit: str = ""
	price_per_unit: Decimal
	currency: str = "KES"
	due_date: str
	verification_method: str = "third_party"


class RbfDliResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	contract_id: str
	name: str
	description: str
	indicator_code: str
	target_value: float
	achieved_value: float = 0.0
	unit: str
	price_per_unit: Decimal
	currency: str
	due_date: str
	verification_method: str
	payment_earned: Decimal = Decimal("0")
	status: str
	tenant_id: str
	created_at: str


class RbfResultClaimCreate(BaseModel):
	model_config = _cfg
	contract_id: str
	dli_id: str
	claimed_value: float
	claim_date: str
	submitted_by: str
	evidence_references: list[str] = Field(default_factory=list)
	notes: str = ""


class RbfResultClaimResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	contract_id: str
	dli_id: str
	claimed_value: float
	verified_value: float = 0.0
	claim_date: str
	submitted_by: str
	evidence_references: list[str]
	notes: str
	payment_triggered: Decimal = Decimal("0")
	status: str
	tenant_id: str
	created_at: str


class RbfVerificationCreate(BaseModel):
	model_config = _cfg
	claim_id: str
	verifier: str
	verification_date: str
	methodology: str = ""
	verified_value: float = 0.0
	accepted: bool = True
	findings: str = ""
	adjustments: str = ""


class RbfVerificationResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	claim_id: str
	verifier: str
	verification_date: str
	methodology: str
	verified_value: float
	accepted: bool
	findings: str
	adjustments: str
	status: str
	tenant_id: str
	created_at: str


class RbfPaymentTriggerCreate(BaseModel):
	model_config = _cfg
	contract_id: str
	claim_id: str
	verification_id: str
	amount: Decimal
	currency: str = "KES"
	payment_date: str
	approved_by: str
	reference: str
	notes: str = ""


class RbfPaymentTriggerResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	contract_id: str
	claim_id: str
	verification_id: str
	amount: Decimal
	currency: str
	payment_date: str
	approved_by: str
	reference: str
	notes: str
	status: str
	tenant_id: str
	created_at: str


class RbfContractFilter(BaseModel):
	model_config = _cfg
	status: str | None = None
	payment_model: str | None = None
	programme_id: str | None = None


class RbfAuditEvent(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	details: dict[str, Any] = Field(default_factory=dict)
	emitted_at: str


__all__ = [
	"RbfContractCreate", "RbfContractUpdate", "RbfContractResponse",
	"RbfDliCreate", "RbfDliResponse",
	"RbfResultClaimCreate", "RbfResultClaimResponse",
	"RbfVerificationCreate", "RbfVerificationResponse",
	"RbfPaymentTriggerCreate", "RbfPaymentTriggerResponse",
	"RbfContractFilter", "RbfAuditEvent",
]
