"""Async service layer for APG Advanced Planning and Scheduling."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

try:
	from .models import MfApsScheduleRun, MfApsScheduledOperation
except ImportError:
	from models import MfApsScheduleRun, MfApsScheduledOperation  # type: ignore

try:
	from situ_cloudevents._uuid7 import uuid7str
except ImportError:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


class MfgApsService:
	"""Advanced Planning and Scheduling service — async, in-memory."""

	def __init__(self, tenant_id: str = "default") -> None:
		self._tenant_id = tenant_id
		self._schedule_runs: dict[str, MfApsScheduleRun] = {}
		self._scheduled_ops: dict[str, MfApsScheduledOperation] = {}

	async def create_schedule_run(
		self,
		horizon_start: str,
		horizon_end: str,
		scheduling_method: str = "forward",
		sequencing_rule: str = "earliest_due_date",
		optimisation_objective: str | None = None,
		triggered_by: str = "system",
		metadata: dict[str, Any] | None = None,
	) -> MfApsScheduleRun:
		run = MfApsScheduleRun(
			tenant_id=self._tenant_id,
			horizon_start=horizon_start,
			horizon_end=horizon_end,
			scheduling_method=scheduling_method,
			sequencing_rule=sequencing_rule,
			optimisation_objective=optimisation_objective,
			triggered_by=triggered_by,
			metadata=metadata or {},
		)
		self._schedule_runs[run.id] = run
		return run

	async def execute_schedule_run(self, run_id: str, operations: list[dict[str, Any]]) -> MfApsScheduleRun:
		"""
		Simulate finite-capacity scheduling.
		operations: list of dicts with keys: operation_id, work_order_id, work_centre_id,
		            setup_time_hrs, run_time_hrs, due_date.
		"""
		run = self._schedule_runs.get(run_id)
		if not run:
			raise KeyError(f"Schedule run not found: {run_id}")

		run.status = "running"
		run.started_at = _now()
		await asyncio.sleep(0)  # yield

		# EDD sort (simplest finite-capacity heuristic)
		sorted_ops = sorted(operations, key=lambda o: o.get("due_date", "9999-12-31"))

		# Track last end time per work centre
		wc_time: dict[str, str] = {}

		seq = 1
		for op in sorted_ops:
			wc = op["work_centre_id"]
			start = wc_time.get(wc, run.horizon_start)
			setup = op.get("setup_time_hrs", 0.0)
			run_t = op.get("run_time_hrs", 0.0)
			total_hrs = setup + run_t

			# Naive end time string (hours offset, not real datetime arithmetic)
			scheduled_op = MfApsScheduledOperation(
				tenant_id=self._tenant_id,
				schedule_run_id=run_id,
				operation_id=op["operation_id"],
				work_order_id=op["work_order_id"],
				work_centre_id=wc,
				sequence_number=seq,
				scheduled_start=start,
				scheduled_end=start,  # simplified — real impl parses ISO + offsets
				setup_time_hrs=setup,
				run_time_hrs=run_t,
			)
			self._scheduled_ops[scheduled_op.id] = scheduled_op
			wc_time[wc] = start  # simplified
			seq += 1

		run.status = "completed"
		run.completed_at = _now()
		run.operations_scheduled = len(sorted_ops)
		run.orders_scheduled = len({op["work_order_id"] for op in sorted_ops})
		return run

	async def get_schedule_run(self, run_id: str) -> MfApsScheduleRun:
		if run_id not in self._schedule_runs:
			raise KeyError(f"Schedule run not found: {run_id}")
		return self._schedule_runs[run_id]

	async def list_schedule_runs(self, status: str | None = None) -> list[MfApsScheduleRun]:
		runs = list(self._schedule_runs.values())
		if status:
			runs = [r for r in runs if r.status == status]
		return runs

	async def get_gantt_data(self, run_id: str) -> list[dict[str, Any]]:
		"""Return Gantt-chart-ready records for a schedule run."""
		ops = [o for o in self._scheduled_ops.values() if o.schedule_run_id == run_id]
		return [
			{
				"operation_id": o.operation_id,
				"work_order_id": o.work_order_id,
				"work_centre_id": o.work_centre_id,
				"sequence": o.sequence_number,
				"start": o.scheduled_start,
				"end": o.scheduled_end,
				"setup_hrs": o.setup_time_hrs,
				"run_hrs": o.run_time_hrs,
				"is_critical": o.is_critical_path,
			}
			for o in sorted(ops, key=lambda x: x.sequence_number)
		]

	async def get_dashboard_summary(self) -> dict[str, Any]:
		runs = list(self._schedule_runs.values())
		return {
			"tenant_id": self._tenant_id,
			"schedule_runs": {"total": len(runs), "completed": sum(1 for r in runs if r.status == "completed")},
			"scheduled_operations": len(self._scheduled_ops),
		}
