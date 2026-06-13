"""Pydantic v2 models for APG Product Costing."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from situ_cloudevents._uuid7 import uuid7str
except ImportError:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())


class MfPcoCostRecord(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	item_id: str
	item_code: str
	cost_type: str = "standard"  # standard | actual | average | target | simulated
	cost_version: str = "1"
	status: str = "draft"  # draft | active | frozen | archived
	currency: str = "USD"
	# Cost elements
	material_cost: float = 0.0
	labour_cost: float = 0.0
	overhead_cost: float = 0.0
	subcontract_cost: float = 0.0
	tooling_cost: float = 0.0
	total_cost: float = 0.0
	# Rollup metadata
	bom_id: str | None = None
	routing_id: str | None = None
	rolled_up_at: str | None = None
	frozen_at: str | None = None
	effective_from: str | None = None
	effective_to: str | None = None
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	created_by: str = "system"
	metadata: dict[str, Any] = Field(default_factory=dict)


class MfPcoVarianceRecord(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	work_order_id: str
	item_id: str
	item_code: str
	variance_type: str  # price | quantity | efficiency | overhead_absorption | mix
	cost_element: str  # material | labour | overhead
	standard_cost: float
	actual_cost: float
	variance_amount: float  # actual - standard
	variance_pct: float | None = None
	period: str  # YYYY-MM
	posted_to_gl: bool = False
	gl_entry_id: str | None = None
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	metadata: dict[str, Any] = Field(default_factory=dict)


class MfPcoPeriodClose(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	period: str  # YYYY-MM
	status: str = "pending"  # pending | in_progress | completed | rejected
	total_variances: float = 0.0
	variance_records_count: int = 0
	approver_id: str | None = None
	approved_at: str | None = None
	closed_at: str | None = None
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	created_by: str = "system"
	metadata: dict[str, Any] = Field(default_factory=dict)
