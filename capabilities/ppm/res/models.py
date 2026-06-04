"""In-memory models for APG Resource Management (res)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Resource:
	id: str
	tenant_id: str
	name: str
	resource_type: str
	status: str
	department: str
	owner_id: str
	cost_rate: float
	cost_rate_type: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ResourceSkill:
	id: str
	tenant_id: str
	resource_id: str
	skill_name: str
	proficiency_level: str
	years_experience: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ResourceAllocation:
	id: str
	tenant_id: str
	resource_id: str
	project_id: str
	task_id: str
	status: str
	start_date: str
	end_date: str
	allocation_pct: float
	manager_approval_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CapacityPlan:
	id: str
	tenant_id: str
	plan_type: str
	name: str
	horizon: str
	demand_data: str  # JSON serialised
	supply_data: str  # JSON serialised
	gap_analysis: str  # JSON serialised
	created_by: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class UtilisationSnapshot:
	id: str
	tenant_id: str
	resource_id: str
	snapshot_period: str
	allocated_hours: float
	available_hours: float
	utilisation_pct: float
	utilisation_band: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DemandForecast:
	id: str
	tenant_id: str
	horizon: str
	resource_type: str
	skill_filter: str
	forecast_demand_fte: float
	current_supply_fte: float
	gap_fte: float
	generated_by: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class LeaveRecord:
	id: str
	tenant_id: str
	resource_id: str
	leave_type: str
	start_date: str
	end_date: str
	approval_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CostRate:
	id: str
	tenant_id: str
	resource_id: str
	rate_type: str
	rate_amount: float
	currency: str
	effective_date: str
	finance_approval_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ResourceAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
