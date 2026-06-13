"""Async service layer for APG Capacity Planning."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
	from .models import MfCapWorkCentreCapacity, MfCapLoadRecord
except ImportError:
	from models import MfCapWorkCentreCapacity, MfCapLoadRecord  # type: ignore

try:
	from situ_cloudevents._uuid7 import uuid7str
except ImportError:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


class MfgCapService:
	"""Capacity Planning service — async, in-memory."""

	def __init__(self, tenant_id: str = "default") -> None:
		self._tenant_id = tenant_id
		self._capacities: dict[str, MfCapWorkCentreCapacity] = {}
		self._loads: dict[str, MfCapLoadRecord] = {}

	async def define_capacity(self, work_centre_id: str, work_centre_code: str, period_start: str, period_end: str, available_hours: float, efficiency_pct: float = 100.0, capacity_type: str = "machine", metadata: dict[str, Any] | None = None) -> MfCapWorkCentreCapacity:
		effective = available_hours * efficiency_pct / 100
		cap = MfCapWorkCentreCapacity(tenant_id=self._tenant_id, work_centre_id=work_centre_id, work_centre_code=work_centre_code, capacity_type=capacity_type, period_start=period_start, period_end=period_end, available_hours=available_hours, efficiency_pct=efficiency_pct, effective_hours=effective, metadata=metadata or {})
		self._capacities[cap.id] = cap
		return cap

	async def add_load(self, work_centre_id: str, period_start: str, period_end: str, load_source: str, source_id: str, required_hours: float, metadata: dict[str, Any] | None = None) -> MfCapLoadRecord:
		# Calculate utilisation against available capacity
		cap_recs = [c for c in self._capacities.values() if c.work_centre_id == work_centre_id and c.period_start == period_start]
		effective_hours = cap_recs[0].effective_hours if cap_recs else None
		utilisation = (required_hours / effective_hours * 100) if effective_hours else None
		is_overloaded = utilisation is not None and utilisation > 100

		rec = MfCapLoadRecord(tenant_id=self._tenant_id, work_centre_id=work_centre_id, period_start=period_start, period_end=period_end, load_source=load_source, source_id=source_id, required_hours=required_hours, utilisation_pct=utilisation, is_overloaded=is_overloaded, metadata=metadata or {})
		self._loads[rec.id] = rec
		return rec

	async def get_load_summary(self, work_centre_id: str | None = None) -> list[dict[str, Any]]:
		loads = list(self._loads.values())
		if work_centre_id:
			loads = [l for l in loads if l.work_centre_id == work_centre_id]

		# Group by work_centre + period
		summary: dict[str, dict[str, Any]] = {}
		for load in loads:
			key = f"{load.work_centre_id}|{load.period_start}"
			if key not in summary:
				cap_recs = [c for c in self._capacities.values() if c.work_centre_id == load.work_centre_id and c.period_start == load.period_start]
				summary[key] = {
					"work_centre_id": load.work_centre_id,
					"period_start": load.period_start,
					"period_end": load.period_end,
					"available_hours": cap_recs[0].effective_hours if cap_recs else None,
					"required_hours": 0.0,
					"is_overloaded": False,
				}
			summary[key]["required_hours"] += load.required_hours
			avail = summary[key]["available_hours"]
			if avail:
				summary[key]["utilisation_pct"] = round(summary[key]["required_hours"] / avail * 100, 2)
				summary[key]["is_overloaded"] = summary[key]["required_hours"] > avail

		return list(summary.values())

	async def identify_constraints(self) -> list[dict[str, Any]]:
		"""Return work centres where load exceeds capacity."""
		return [s for s in await self.get_load_summary() if s.get("is_overloaded")]

	async def list_capacities(self, work_centre_id: str | None = None) -> list[MfCapWorkCentreCapacity]:
		caps = list(self._capacities.values())
		if work_centre_id:
			caps = [c for c in caps if c.work_centre_id == work_centre_id]
		return caps
