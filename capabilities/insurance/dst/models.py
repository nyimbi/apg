"""Pydantic v2 models for Distribution & Agency Management (ins_dst)."""
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


class DstAgentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	agent_code: str
	agent_name: str
	agent_type: str
	id_number: str
	ira_licence_number: str
	phone: str
	email: str
	supervisor_id: str | None = None
	branch_id: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class DstAgentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	agent_name: str | None = None
	phone: str | None = None
	email: str | None = None
	status: str | None = None
	supervisor_id: str | None = None


class DstAgentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	agent_code: str
	agent_name: str
	agent_type: str
	ira_licence_number: str
	status: str
	tenant_id: str
	created_at: datetime


class DstCommissionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	agent_id: str
	policy_id: str
	policy_number: str
	product_code: str
	premium_amount: Decimal
	commission_rate: Decimal
	period: str


class DstCommissionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	agent_id: str
	policy_id: str
	commission_amount: Decimal
	commission_rate: Decimal
	status: str
	tenant_id: str
	created_at: datetime


class DstPerformanceReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	agent_id: str
	period_start: date
	period_end: date
	policies_sold: int
	premium_written: Decimal
	commission_earned: Decimal
	target_premium: Decimal | None = None


class DstComplianceRecord(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	agent_id: str
	compliance_type: str
	status: str
	expiry_date: date | None = None
	notes: str = ""


class DstBancassurancePartner(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	partner_name: str
	partner_type: str
	bank_code: str
	products: list[str] = Field(default_factory=list)
	commission_rate: Decimal
	effective_date: date


class DstAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime
