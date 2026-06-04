"""In-memory models for APG Grid Operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StateEstimationRun:
	id: str
	tenant_id: str
	estimator_type: str
	grid_area: str
	network_model_ref: str
	measurement_snapshot_ref: str
	status: str
	started_at: str
	completed_at: str = ""
	iterations: int = 0
	converged: bool = False
	residual: float = 0.0
	voltage_violations: int = 0
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ContingencyCase:
	id: str
	tenant_id: str
	contingency_type: str
	contingency_name: str
	system_status: str
	base_case_ref: str
	analyzed_at: str
	violations: list[dict[str, Any]] = field(default_factory=list)
	max_overload_pct: float = 0.0
	min_voltage_pu: float = 1.0
	max_voltage_pu: float = 1.0
	remedial_actions: list[str] = field(default_factory=list)
	notes: str = ""

	def has_violations(self) -> bool:
		return len(self.violations) > 0

	def to_dict(self) -> dict[str, Any]:
		d = asdict(self)
		d["has_violations"] = self.has_violations()
		return d


@dataclass
class VoltageControlAction:
	id: str
	tenant_id: str
	control_method: str
	element_id: str
	target_voltage_pu: float
	achieved_voltage_pu: float
	approved_by: str
	executed_at: str
	status: str = "completed"
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FrequencyControlAction:
	id: str
	tenant_id: str
	control_method: str
	trigger_frequency_hz: float
	response_mw: float
	response_mvar: float = 0.0
	executed_at: str = ""
	status: str = "pending"
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class MarketSettlementInterval:
	id: str
	tenant_id: str
	market_product: str
	interval_start: str
	interval_end: str
	metered_mwh: float
	scheduled_mwh: float
	imbalance_mwh: float
	price_per_mwh: float
	settlement_amount: float
	currency: str
	status: str
	participant_id: str = ""
	bid_offer_ref: str = ""
	revised_at: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class GridAlarm:
	id: str
	tenant_id: str
	alarm_category: str
	severity: str
	element_id: str
	description: str
	raised_at: str
	acknowledged: bool = False
	acknowledged_by: str = ""
	acknowledged_at: str = ""
	cleared_at: str = ""
	status: str = "active"
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class EmsFunctionExecution:
	id: str
	tenant_id: str
	ems_function: str
	mode: str
	started_at: str
	completed_at: str = ""
	status: str = "running"
	result_summary: dict[str, Any] = field(default_factory=dict)
	triggered_by: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class GrdAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered_at: str
	active: bool = True

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AuditEvent:
	id: str
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	actor: str
	occurred_at: str
	payload: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
