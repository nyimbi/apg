"""Pydantic v2 models for APG Advanced Planning and Scheduling."""

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


class MfApsScheduledOperation(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	schedule_run_id: str
	operation_id: str
	work_order_id: str
	work_centre_id: str
	sequence_number: int
	scheduled_start: str
	scheduled_end: str
	setup_time_hrs: float = 0.0
	run_time_hrs: float = 0.0
	slack_hrs: float = 0.0
	is_critical_path: bool = False
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	metadata: dict[str, Any] = Field(default_factory=dict)


class MfApsScheduleRun(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	scheduling_method: str = "forward"
	sequencing_rule: str = "earliest_due_date"
	optimisation_objective: str | None = None
	horizon_start: str
	horizon_end: str
	status: str = "pending"  # pending | running | completed | failed
	orders_scheduled: int = 0
	operations_scheduled: int = 0
	constraint_violations: int = 0
	makespan_hrs: float | None = None
	started_at: str | None = None
	completed_at: str | None = None
	triggered_by: str = "system"
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	metadata: dict[str, Any] = Field(default_factory=dict)
