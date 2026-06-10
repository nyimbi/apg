"""Pydantic v2 models for Actuarial Tools (ins_act)."""
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


class ActMortalityTableCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	table_name: str
	table_type: str
	base_year: int
	ages: list[int]
	qx_values: list[float]
	lx_values: list[float]
	source: str


class ActLossRatioReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	product_code: str
	period_start: date
	period_end: date
	earned_premium: Decimal
	incurred_losses: Decimal
	loss_ratio: Decimal
	expense_ratio: Decimal | None = None
	combined_ratio: Decimal | None = None


class ActReserveCalculation(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	product_code: str
	valuation_date: date
	method: str
	gross_reserve: Decimal
	net_reserve: Decimal
	ibnr_estimate: Decimal
	assumptions: dict[str, Any] = Field(default_factory=dict)


class ActIBNREstimate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	product_code: str
	valuation_date: date
	development_method: str
	ibnr_amount: Decimal
	confidence_level: float
	triangle_periods: int
	tenant_id: str
	created_at: datetime


class ActPricingModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	model_name: str
	product_code: str
	risk_factors: list[str]
	base_rate: Decimal
	parameters: dict[str, Any] = Field(default_factory=dict)
	effective_date: date


class ActExperienceAnalysis(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	product_code: str
	analysis_period_years: int
	actual_claims: int
	expected_claims: int
	ae_ratio: float
	actual_loss_amount: Decimal
	expected_loss_amount: Decimal


class ActAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime
