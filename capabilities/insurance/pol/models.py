"""Pydantic v2 models for Policy Administration (ins_pol)."""
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


# ── Policy models ─────────────────────────────────────────────────────────────

class PolPolicyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	policy_number: str
	product_code: str
	insured_name: str
	insured_id: str
	sum_insured: Decimal
	currency: str = "KES"
	inception_date: date
	expiry_date: date
	premium: Decimal
	underwriter_id: str
	agent_id: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class PolPolicyUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	sum_insured: Decimal | None = None
	premium: Decimal | None = None
	expiry_date: date | None = None
	agent_id: str | None = None
	status: str | None = None
	metadata: dict[str, Any] | None = None


class PolPolicyResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	policy_number: str
	product_code: str
	insured_name: str
	insured_id: str
	sum_insured: Decimal
	currency: str
	inception_date: date
	expiry_date: date
	premium: Decimal
	underwriter_id: str
	agent_id: str | None = None
	status: str = "active"
	tenant_id: str
	created_at: datetime
	updated_at: datetime | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class PolPolicyList(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	items: list[PolPolicyResponse]
	total: int
	page: int = 1
	page_size: int = 50


class PolPolicyFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	product_code: str | None = None
	status: str | None = None
	insured_id: str | None = None
	agent_id: str | None = None
	inception_date_from: date | None = None
	inception_date_to: date | None = None


class PolEndorsementCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	policy_id: str
	endorsement_type: str
	effective_date: date
	description: str
	change_in_premium: Decimal = Decimal("0")
	change_in_sum_insured: Decimal = Decimal("0")
	requested_by: str
	metadata: dict[str, Any] = Field(default_factory=dict)


class PolEndorsementResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	policy_id: str
	endorsement_type: str
	effective_date: date
	description: str
	change_in_premium: Decimal
	change_in_sum_insured: Decimal
	requested_by: str
	status: str = "pending"
	tenant_id: str
	created_at: datetime
	metadata: dict[str, Any] = Field(default_factory=dict)


class PolRenewalCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	policy_id: str
	new_expiry_date: date
	new_premium: Decimal
	renewal_terms: dict[str, Any] = Field(default_factory=dict)
	initiated_by: str


class PolCancellationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	policy_id: str
	cancellation_date: date
	reason: str
	cancellation_type: str = "voluntary"
	refund_premium: bool = True
	authorised_by: str


class PolReinstatementCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	policy_id: str
	reinstatement_date: date
	outstanding_premium: Decimal
	reason: str
	authorised_by: str


class PolDocumentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	policy_id: str
	document_type: str
	generated_by: str
	metadata: dict[str, Any] = Field(default_factory=dict)


class PolAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	actor: str | None = None
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime
