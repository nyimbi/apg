"""Async service layer for APG Manufacturing Execution System."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import evaluate_capability_rules
except ImportError:
	from capability_contract import evaluate_capability_rules  # type: ignore

try:
	from situ_cloudevents._uuid7 import uuid7str
except ImportError:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


class MfgMesService:
	"""Manufacturing Execution System service — async, in-memory store."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self._tenant_id = tenant_id
		_store = get_store(db_url)
		self._work_orders = WriteThruDict('work_orders', tenant_id, _store)
		self._production_events = WriteThruDict('production_events', tenant_id, _store)
		self._downtime_records = WriteThruDict('downtime_records', tenant_id, _store)
		self._resource_statuses = WriteThruDict('resource_statuses', tenant_id, _store)
		self._oee_records = WriteThruDict('oee_records', tenant_id, _store)

	# ------------------------------------------------------------------ #
	# Work Orders
	# ------------------------------------------------------------------ #

	async def create_work_order(
		self,
		item_id: str,
		item_code: str,
		quantity: float,
		scheduled_start: str,
		scheduled_end: str,
		work_center_id: str | None = None,
		routing_id: str | None = None,
		production_order_id: str | None = None,
		priority: int = 50,
		notes: str = "",
		created_by: str = "system",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		ctx = {
			"tenant_context_present": True,
			"operation": "create_work_order",
			"item_present": bool(item_id),
			"quantity_valid": quantity > 0,
		}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"Work order creation denied: {decision['actions']}")

		wo: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"item_id": item_id,
			"item_code": item_code,
			"quantity_scheduled": quantity,
			"quantity_started": 0.0,
			"quantity_completed": 0.0,
			"quantity_scrapped": 0.0,
			"status": "created",
			"scheduled_start": scheduled_start,
			"scheduled_end": scheduled_end,
			"actual_start": None,
			"actual_end": None,
			"work_center_id": work_center_id,
			"routing_id": routing_id,
			"production_order_id": production_order_id,
			"priority": priority,
			"notes": notes,
			"created_at": _now(),
			"updated_at": _now(),
			"created_by": created_by,
			"metadata": metadata or {},
		}
		self._work_orders[wo["id"]] = wo
		return wo

	async def start_work_order(self, wo_id: str, operator_id: str | None = None) -> dict[str, Any]:
		wo = await self.get_work_order(wo_id)
		wo["status"] = "started"
		wo["actual_start"] = _now()
		wo["updated_at"] = _now()
		if operator_id:
			wo["metadata"]["operator_id"] = operator_id
		# Record event
		await self.record_production_event(wo_id=wo_id, event_type="start", quantity=0.0, operator_id=operator_id)
		return wo

	async def pause_work_order(self, wo_id: str, reason: str = "", operator_id: str | None = None) -> dict[str, Any]:
		wo = await self.get_work_order(wo_id)
		wo["status"] = "paused"
		wo["updated_at"] = _now()
		await self.record_production_event(wo_id=wo_id, event_type="pause", quantity=0.0, operator_id=operator_id, notes=reason)
		return wo

	async def complete_work_order(
		self,
		wo_id: str,
		quantity_completed: float,
		quantity_scrapped: float = 0.0,
		operator_id: str | None = None,
	) -> dict[str, Any]:
		wo = await self.get_work_order(wo_id)
		ctx = {
			"tenant_context_present": True,
			"operation": "complete_work_order",
			"order_started": wo["status"] in ("started", "in_progress", "paused"),
		}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"Complete denied: {decision['actions']}")
		wo["status"] = "completed"
		wo["quantity_completed"] = quantity_completed
		wo["quantity_scrapped"] = quantity_scrapped
		wo["actual_end"] = _now()
		wo["updated_at"] = _now()
		await self.record_production_event(wo_id=wo_id, event_type="complete", quantity=quantity_completed, operator_id=operator_id)
		return wo

	async def get_work_order(self, wo_id: str) -> dict[str, Any]:
		if wo_id not in self._work_orders:
			raise KeyError(f"Work order not found: {wo_id}")
		return self._work_orders[wo_id]

	async def list_work_orders(
		self,
		status: str | None = None,
		work_center_id: str | None = None,
		limit: int = 100,
		offset: int = 0,
	) -> list[dict[str, Any]]:
		wos = list(self._work_orders.values())
		if status:
			wos = [w for w in wos if w["status"] == status]
		if work_center_id:
			wos = [w for w in wos if w.get("work_center_id") == work_center_id]
		return wos[offset : offset + limit]

	# ------------------------------------------------------------------ #
	# Production Events
	# ------------------------------------------------------------------ #

	async def record_production_event(
		self,
		wo_id: str,
		event_type: str,
		quantity: float = 0.0,
		operator_id: str | None = None,
		notes: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		ctx = {
			"tenant_context_present": True,
			"operation": "record_production_event",
			"work_order_present": bool(wo_id),
		}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"Event recording denied: {decision['actions']}")

		event: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"work_order_id": wo_id,
			"event_type": event_type,
			"quantity": quantity,
			"operator_id": operator_id,
			"notes": notes,
			"recorded_at": _now(),
			"metadata": metadata or {},
		}
		self._production_events[event["id"]] = event
		return event

	async def list_production_events(
		self,
		wo_id: str | None = None,
		event_type: str | None = None,
		limit: int = 200,
		offset: int = 0,
	) -> list[dict[str, Any]]:
		events = list(self._production_events.values())
		if wo_id:
			events = [e for e in events if e["work_order_id"] == wo_id]
		if event_type:
			events = [e for e in events if e["event_type"] == event_type]
		return events[offset : offset + limit]

	# ------------------------------------------------------------------ #
	# Downtime
	# ------------------------------------------------------------------ #

	async def record_downtime(
		self,
		resource_id: str,
		category: str,
		start_time: str,
		end_time: str | None = None,
		wo_id: str | None = None,
		description: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		duration_minutes: float | None = None
		if end_time:
			from datetime import datetime as dt
			try:
				s = dt.fromisoformat(start_time)
				e = dt.fromisoformat(end_time)
				duration_minutes = (e - s).total_seconds() / 60
			except Exception:
				pass

		record: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"resource_id": resource_id,
			"category": category,
			"work_order_id": wo_id,
			"description": description,
			"start_time": start_time,
			"end_time": end_time,
			"duration_minutes": duration_minutes,
			"created_at": _now(),
			"metadata": metadata or {},
		}
		self._downtime_records[record["id"]] = record
		return record

	async def list_downtime(
		self,
		resource_id: str | None = None,
		category: str | None = None,
		limit: int = 100,
		offset: int = 0,
	) -> list[dict[str, Any]]:
		records = list(self._downtime_records.values())
		if resource_id:
			records = [r for r in records if r["resource_id"] == resource_id]
		if category:
			records = [r for r in records if r["category"] == category]
		return records[offset : offset + limit]

	# ------------------------------------------------------------------ #
	# Resource Status
	# ------------------------------------------------------------------ #

	async def update_resource_status(
		self,
		resource_id: str,
		resource_name: str,
		status: str,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		rec: dict[str, Any] = {
			"id": resource_id,
			"tenant_id": self._tenant_id,
			"resource_name": resource_name,
			"status": status,
			"updated_at": _now(),
			"metadata": metadata or {},
		}
		self._resource_statuses[resource_id] = rec
		return rec

	async def list_resources(self, status: str | None = None) -> list[dict[str, Any]]:
		resources = list(self._resource_statuses.values())
		if status:
			resources = [r for r in resources if r["status"] == status]
		return resources

	# ------------------------------------------------------------------ #
	# OEE Calculation
	# ------------------------------------------------------------------ #

	async def calculate_oee(
		self,
		resource_id: str,
		period_start: str,
		period_end: str,
		planned_production_time_min: float,
		actual_run_time_min: float,
		ideal_cycle_time_sec: float,
		total_count: float,
		good_count: float,
	) -> dict[str, Any]:
		"""ISO 22400-2 OEE calculation."""
		availability = actual_run_time_min / planned_production_time_min if planned_production_time_min else 0.0
		ideal_run_time = (total_count * ideal_cycle_time_sec) / 60
		performance = ideal_run_time / actual_run_time_min if actual_run_time_min else 0.0
		quality = good_count / total_count if total_count else 0.0
		oee = availability * performance * quality

		record: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"resource_id": resource_id,
			"period_start": period_start,
			"period_end": period_end,
			"planned_production_time_min": planned_production_time_min,
			"actual_run_time_min": actual_run_time_min,
			"total_count": total_count,
			"good_count": good_count,
			"availability": round(availability, 4),
			"performance": round(performance, 4),
			"quality": round(quality, 4),
			"oee": round(oee, 4),
			"calculated_at": _now(),
		}
		self._oee_records[record["id"]] = record
		return record

	async def list_oee_records(
		self,
		resource_id: str | None = None,
		limit: int = 100,
		offset: int = 0,
	) -> list[dict[str, Any]]:
		records = list(self._oee_records.values())
		if resource_id:
			records = [r for r in records if r["resource_id"] == resource_id]
		return records[offset : offset + limit]

	# ------------------------------------------------------------------ #
	# Dashboard
	# ------------------------------------------------------------------ #

	async def get_dashboard_summary(self) -> dict[str, Any]:
		wos = list(self._work_orders.values())
		events = list(self._production_events.values())
		downtime = list(self._downtime_records.values())
		resources = list(self._resource_statuses.values())
		oee_recs = list(self._oee_records.values())

		avg_oee = (sum(r["oee"] for r in oee_recs) / len(oee_recs)) if oee_recs else None

		return {
			"tenant_id": self._tenant_id,
			"work_orders": {
				"total": len(wos),
				"created": sum(1 for w in wos if w["status"] == "created"),
				"started": sum(1 for w in wos if w["status"] in ("started", "in_progress")),
				"paused": sum(1 for w in wos if w["status"] == "paused"),
				"completed": sum(1 for w in wos if w["status"] == "completed"),
			},
			"production_events": {"total": len(events)},
			"downtime": {
				"total_events": len(downtime),
				"total_minutes": sum(r["duration_minutes"] or 0 for r in downtime),
			},
			"resources": {
				"total": len(resources),
				"available": sum(1 for r in resources if r["status"] == "available"),
				"busy": sum(1 for r in resources if r["status"] == "busy"),
				"breakdown": sum(1 for r in resources if r["status"] == "breakdown"),
			},
			"oee": {"records": len(oee_recs), "average": avg_oee},
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_work_orders', '_production_events', '_downtime_records', '_resource_statuses', '_oee_records']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

