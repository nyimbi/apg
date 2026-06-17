"""Async service layer for APG Repetitive Manufacturing."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

from datetime import datetime, timezone
from typing import Any

try:
	from situ_cloudevents._uuid7 import uuid7str
except ImportError:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


class MfgRfmService:
	"""Repetitive Manufacturing service — async, in-memory."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self._tenant_id = tenant_id
		self._lines = WriteThruDict('lines', tenant_id, _store)
		self._schedules = WriteThruDict('schedules', tenant_id, _store)
		self._backflush_records = WriteThruDict('backflush_records', tenant_id, _store)

	async def create_production_line(self, line_code: str, line_name: str, item_id: str, item_code: str, takt_time_sec: float | None = None, shifts_per_day: int = 1, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
		line: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"line_code": line_code,
			"line_name": line_name,
			"item_id": item_id,
			"item_code": item_code,
			"takt_time_sec": takt_time_sec,
			"shifts_per_day": shifts_per_day,
			"status": "active",
			"created_at": _now(),
			"metadata": metadata or {},
		}
		self._lines[line["id"]] = line
		return line

	async def set_line_status(self, line_id: str, status: str) -> dict[str, Any]:
		line = self._lines.get(line_id)
		if not line:
			raise KeyError(f"Production line not found: {line_id}")
		line["status"] = status
		line["updated_at"] = _now()
		return line

	async def create_rate_schedule(self, line_id: str, schedule_type: str, period_date: str, planned_rate: float, uom: str = "EA", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
		if line_id not in self._lines:
			raise KeyError(f"Production line not found: {line_id}")
		schedule: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"line_id": line_id,
			"schedule_type": schedule_type,
			"period_date": period_date,
			"planned_rate": planned_rate,
			"actual_rate": None,
			"uom": uom,
			"status": "planned",
			"created_at": _now(),
			"metadata": metadata or {},
		}
		self._schedules[schedule["id"]] = schedule
		return schedule

	async def record_backflush(self, line_id: str, schedule_id: str, actual_quantity: float, scrap_quantity: float = 0.0, operator_id: str | None = None) -> dict[str, Any]:
		rec: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"line_id": line_id,
			"schedule_id": schedule_id,
			"actual_quantity": actual_quantity,
			"scrap_quantity": scrap_quantity,
			"operator_id": operator_id,
			"recorded_at": _now(),
		}
		self._backflush_records[rec["id"]] = rec
		# Update schedule actual rate
		schedule = self._schedules.get(schedule_id)
		if schedule:
			schedule["actual_rate"] = actual_quantity
			schedule["status"] = "confirmed"
		return rec

	async def list_lines(self, status: str | None = None) -> list[dict[str, Any]]:
		lines = list(self._lines.values())
		return [l for l in lines if l["status"] == status] if status else lines

	async def list_schedules(self, line_id: str | None = None, period_date: str | None = None) -> list[dict[str, Any]]:
		schedules = list(self._schedules.values())
		if line_id:
			schedules = [s for s in schedules if s["line_id"] == line_id]
		if period_date:
			schedules = [s for s in schedules if s["period_date"] == period_date]
		return schedules

	async def get_dashboard_summary(self) -> dict[str, Any]:
		lines = list(self._lines.values())
		schedules = list(self._schedules.values())
		backflushes = list(self._backflush_records.values())
		return {
			"tenant_id": self._tenant_id,
			"lines": {"total": len(lines), "active": sum(1 for l in lines if l["status"] == "active"), "idle": sum(1 for l in lines if l["status"] == "idle")},
			"schedules": {"total": len(schedules), "planned": sum(1 for s in schedules if s["status"] == "planned"), "confirmed": sum(1 for s in schedules if s["status"] == "confirmed")},
			"backflush_records": len(backflushes),
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_lines', '_schedules', '_backflush_records']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

