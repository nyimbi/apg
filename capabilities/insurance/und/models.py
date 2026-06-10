"""Pydantic v2 models for Underwriting Engine (ins_und)."""
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


class UndRiskSubmissionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	proposer_name: str
	proposer_id: str
	product_code: str
	risk_class: str
	sum_insured: Decimal
	currency: str = "KES"
	risk_attributes: dict[str, Any] = Field(default_factory=dict)
	submitted_by: str


class UndRiskAssessmentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	submission_id: str
	risk_score: float
	risk_band: str
	recommended_premium: Decimal
	loading_factors: dict[str, float] = Field(default_factory=dict)
	exclusions: list[str] = Field(default_factory=list)
	decision: str
	assessed_by: str | None = None
	tenant_id: str
	created_at: datetime


class UndRatingRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	submission_id: str
	base_rate: Decimal
	adjustments: dict[str, Decimal] = Field(default_factory=dict)
	rated_by: str


class UndCapacityCheck(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	product_code: str
	risk_class: str
	requested_sum_insured: Decimal
	currency: str = "KES"


class UndReinsuranceTreaty(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	treaty_name: str
	treaty_type: str
	reinsurer: str
	retention: Decimal
	cession_pct: float
	treaty_limit: Decimal
	effective_date: date
	expiry_date: date


class UndUnderwritingRule(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	rule_name: str
	product_code: str
	condition: str
	action: str
	priority: int = 100
	active: bool = True


class UndRiskFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	product_code: str | None = None
	risk_band: str | None = None
	decision: str | None = None
	proposer_id: str | None = None


class UndAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime
