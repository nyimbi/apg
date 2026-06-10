"""Pydantic v2 models for Insurance Regulatory Reporting (ins_reg)."""
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


class RegReturnCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	return_type: str
	regulator: str
	period_start: date
	period_end: date
	prepared_by: str
	data: dict[str, Any] = Field(default_factory=dict)


class RegReturnResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	return_reference: str
	return_type: str
	regulator: str
	period_start: date
	period_end: date
	status: str
	prepared_by: str
	submitted_by: str | None = None
	submitted_at: datetime | None = None
	tenant_id: str
	created_at: datetime


class RegSolvencyReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	valuation_date: date
	total_assets: Decimal
	total_liabilities: Decimal
	eligible_own_funds: Decimal
	scr: Decimal
	mcr: Decimal
	solvency_ratio: Decimal
	prepared_by: str


class RegStatisticalReturn(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	period: str
	policies_in_force: int
	gross_premium: Decimal
	net_premium: Decimal
	gross_claims: Decimal
	net_claims: Decimal
	loss_ratio: Decimal


class RegMarketConductFiling(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	filing_type: str
	subject: str
	description: str
	attachments: list[str] = Field(default_factory=list)
	submitted_by: str


class RegComplianceCalendar(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	return_type: str
	regulator: str
	due_date: date
	frequency: str
	responsible_party: str


class RegAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime
