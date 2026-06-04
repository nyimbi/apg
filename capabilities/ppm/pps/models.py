"""In-memory models for APG Project Planning & Scheduling (pps)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Project:
	id: str
	tenant_id: str
	name: str
	status: str
	methodology: str
	owner_id: str
	start_date: str
	end_date: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class WbsElement:
	id: str
	tenant_id: str
	project_id: str
	parent_id: str | None
	level: str
	code: str
	name: str
	description: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class Task:
	id: str
	tenant_id: str
	project_id: str
	wbs_element_id: str
	task_type: str
	status: str
	name: str
	duration_days: float
	scheduling_mode: str
	constraint_type: str
	progress_method: str
	progress_pct: float
	start_date: str
	end_date: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class TaskDependency:
	id: str
	tenant_id: str
	predecessor_id: str
	successor_id: str
	dependency_type: str
	lag_days: float

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CriticalPathResult:
	id: str
	tenant_id: str
	project_id: str
	method: str
	critical_task_ids: str  # JSON list
	total_float_days: float
	project_duration_days: float
	calculated_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ResourceLevellingResult:
	id: str
	tenant_id: str
	project_id: str
	algorithm: str
	over_allocations_resolved: int
	schedule_extension_days: float
	levelled_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ProjectCalendar:
	id: str
	tenant_id: str
	name: str
	calendar_type: str
	working_hours_per_day: float
	working_days: str  # JSON list of day names

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ScheduleAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
