"""Pydantic v2 models for Supply Planning (scm_spl)."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid_extensions import uuid7str
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())


class DemandForecastCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	sku: str
	warehouse_id: str | None = None
	period: str  # e.g. "2026-07"
	forecast_quantity: float
	confidence_pct: float = 80.0
	method: str = "statistical"  # statistical | ml | manual


class DemandForecastResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	sku: str
	warehouse_id: str | None
	period: str
	forecast_quantity: float
	actual_quantity: float | None
	confidence_pct: float
	method: str
	status: str
	created_at: str


class MRPRunCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	run_name: str
	horizon_weeks: int = 12
	sku_filter: list[str] = Field(default_factory=list)
	warehouse_filter: list[str] = Field(default_factory=list)
	include_safety_stock: bool = True


class MRPRunResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	run_name: str
	horizon_weeks: int
	sku_filter: list[str]
	warehouse_filter: list[str]
	include_safety_stock: bool
	planned_orders: list[dict[str, Any]]
	status: str
	started_at: str
	completed_at: str | None


class SafetyStockCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	sku: str
	warehouse_id: str | None = None
	target_service_level_pct: float = 95.0
	lead_time_days: int
	demand_std_dev: float | None = None
	manual_override: float | None = None


class SafetyStockResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	sku: str
	warehouse_id: str | None
	target_service_level_pct: float
	lead_time_days: int
	demand_std_dev: float | None
	calculated_safety_stock: float
	manual_override: float | None
	effective_safety_stock: float
	status: str
	calculated_at: str


class ReplenishmentRuleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	sku: str
	warehouse_id: str | None = None
	rule_type: str  # min_max | reorder_point | periodic_review
	reorder_point: float | None = None
	order_quantity: float | None = None
	min_stock: float | None = None
	max_stock: float | None = None
	review_cycle_days: int | None = None


class ReplenishmentRuleResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	sku: str
	warehouse_id: str | None
	rule_type: str
	reorder_point: float | None
	order_quantity: float | None
	min_stock: float | None
	max_stock: float | None
	review_cycle_days: int | None
	status: str
	created_at: str


class CapacityPlanCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	resource_id: str
	resource_type: str  # warehouse | production_line | supplier
	period: str
	available_capacity: float
	unit: str = "units"
	notes: str | None = None


class CapacityPlanResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	resource_id: str
	resource_type: str
	period: str
	available_capacity: float
	planned_demand: float
	utilisation_pct: float
	unit: str
	notes: str | None
	status: str
	created_at: str


class SupplyDemandBalanceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	sku: str
	period: str
	supply_quantity: float
	demand_quantity: float
	opening_stock: float = 0.0


class SupplyDemandBalanceResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	sku: str
	period: str
	opening_stock: float
	supply_quantity: float
	demand_quantity: float
	closing_stock: float
	surplus_shortage: float
	status: str  # balanced | surplus | shortage
	created_at: str


class SplAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	status: str
	emitted_at: str
