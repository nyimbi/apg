"""Farm Management System service — agr_fms."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)
_CAPABILITY_ID = "agr_fms"


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_id(prefix: str = "") -> str:
	suffix = uuid4().hex[:12]
	return f"{prefix}-{suffix}" if prefix else suffix


class FarmManagementService:
	"""Async service for farm management: parcel registry, input recording,
	labour scheduling, cost tracking, and farm diary."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		if not tenant_id:
			raise ValueError("tenant_id required")
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self._parcels = WriteThruDict('parcels', tenant_id, _store)
		self._inputs = WriteThruDict('inputs', tenant_id, _store)
		self._labour = WriteThruDict('labour', tenant_id, _store)
		self._diary = WriteThruDict('diary', tenant_id, _store)
		self._audit = WriteThruList('audit', tenant_id, _store)

	def _emit(self, event_type: str, entity_type: str, entity_id: str, payload: dict[str, Any]) -> None:
		self._audit.append({
			"id": _new_id("evt"),
			"tenant_id": self.tenant_id,
			"event_type": event_type,
			"entity_type": entity_type,
			"entity_id": entity_id,
			"payload": payload,
			"occurred_at": _now(),
		})

	# ------------------------------------------------------------------ health

	async def health_check(self) -> dict[str, Any]:
		return {
			"status": "ok",
			"capability": _CAPABILITY_ID,
			"tenant_id": self.tenant_id,
			"counts": {
				"parcels": len(self._parcels),
				"input_records": len(self._inputs),
				"labour_schedules": len(self._labour),
				"diary_entries": len(self._diary),
			},
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": _CAPABILITY_ID,
			"name": "Farm Management System",
			"domain": "agriculture",
			"version": "1.0.0",
			"description": "Parcel registry, input recording, labour scheduling, cost tracking, farm diary.",
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		return self._audit[-limit:]

	# ------------------------------------------------------------------ parcels

	async def list_parcels(self, status: str | None = None, owner_id: str | None = None,
						limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		items = list(self._parcels.values())
		if status:
			items = [p for p in items if p.get("status") == status]
		if owner_id:
			items = [p for p in items if p.get("owner_id") == owner_id]
		return items[offset: offset + limit]

	async def get_parcel(self, parcel_id: str) -> dict[str, Any]:
		if parcel_id not in self._parcels:
			raise KeyError(f"parcel_not_found:{parcel_id}")
		return self._parcels[parcel_id]

	async def create_parcel(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			parcel_id = _new_id("par")
			ts = _now()
			record: dict[str, Any] = {
				"id": parcel_id,
				"tenant_id": self.tenant_id,
				"name": payload["name"],
				"area_ha": float(payload["area_ha"]),
				"location_lat": payload.get("location_lat"),
				"location_lng": payload.get("location_lng"),
				"soil_type": payload.get("soil_type"),
				"status": payload.get("status", "active"),
				"owner_id": payload.get("owner_id"),
				"notes": payload.get("notes"),
				"metadata": dict(payload.get("metadata", {})),
				"created_at": ts,
				"updated_at": ts,
			}
			self._parcels[parcel_id] = record
			self._emit("parcel.created", "parcel", parcel_id, record)
			return record
		except Exception as exc:
			_log.error("create_parcel failed: %s", exc)
			raise

	async def update_parcel(self, parcel_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if parcel_id not in self._parcels:
				raise KeyError(f"parcel_not_found:{parcel_id}")
			record = self._parcels[parcel_id]
			for field in ["name", "area_ha", "soil_type", "status", "notes", "metadata"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("parcel.updated", "parcel", parcel_id, payload)
			return record
		except Exception as exc:
			_log.error("update_parcel failed: %s", exc)
			raise

	async def delete_parcel(self, parcel_id: str) -> dict[str, Any]:
		try:
			if parcel_id not in self._parcels:
				raise KeyError(f"parcel_not_found:{parcel_id}")
			self._parcels.pop(parcel_id)
			self._emit("parcel.deleted", "parcel", parcel_id, {"id": parcel_id})
			return {"deleted": True, "id": parcel_id}
		except Exception as exc:
			_log.error("delete_parcel failed: %s", exc)
			raise

	async def get_parcel_summary(self, parcel_id: str) -> dict[str, Any]:
		"""Compute cost and labour summary for a parcel."""
		if parcel_id not in self._parcels:
			raise KeyError(f"parcel_not_found:{parcel_id}")
		parcel = self._parcels[parcel_id]
		input_records = [i for i in self._inputs.values() if i.get("farm_parcel_id") == parcel_id]
		labour_records = [l for l in self._labour.values() if l.get("farm_parcel_id") == parcel_id]
		total_input_cost = sum(i.get("total_cost", 0) for i in input_records)
		total_labour_cost = sum(l.get("total_labour_cost", 0) for l in labour_records)
		return {
			"parcel_id": parcel_id,
			"name": parcel["name"],
			"area_ha": parcel["area_ha"],
			"input_records": len(input_records),
			"labour_schedules": len(labour_records),
			"total_input_cost": round(total_input_cost, 2),
			"total_labour_cost": round(total_labour_cost, 2),
			"total_cost": round(total_input_cost + total_labour_cost, 2),
			"cost_per_ha": round((total_input_cost + total_labour_cost) / parcel["area_ha"], 2)
				if parcel["area_ha"] > 0 else None,
		}

	# ------------------------------------------------------------------ inputs

	async def list_inputs(self, farm_parcel_id: str | None = None, category: str | None = None,
						limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		items = list(self._inputs.values())
		if farm_parcel_id:
			items = [i for i in items if i.get("farm_parcel_id") == farm_parcel_id]
		if category:
			items = [i for i in items if i.get("category") == category]
		return items[offset: offset + limit]

	async def get_input(self, input_id: str) -> dict[str, Any]:
		if input_id not in self._inputs:
			raise KeyError(f"input_record_not_found:{input_id}")
		return self._inputs[input_id]

	async def create_input(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			input_id = _new_id("inp")
			ts = _now()
			qty = float(payload["quantity"])
			unit_cost = float(payload["unit_cost"])
			record: dict[str, Any] = {
				"id": input_id,
				"tenant_id": self.tenant_id,
				"farm_parcel_id": payload["farm_parcel_id"],
				"crop_id": payload.get("crop_id"),
				"category": payload["category"],
				"product_name": payload["product_name"],
				"quantity": qty,
				"unit": payload["unit"],
				"unit_cost": unit_cost,
				"total_cost": round(qty * unit_cost, 2),
				"supplier": payload.get("supplier"),
				"applied_date": payload["applied_date"],
				"notes": payload.get("notes"),
				"created_at": ts,
			}
			self._inputs[input_id] = record
			self._emit("input.created", "input_record", input_id, record)
			return record
		except Exception as exc:
			_log.error("create_input failed: %s", exc)
			raise

	async def delete_input(self, input_id: str) -> dict[str, Any]:
		try:
			if input_id not in self._inputs:
				raise KeyError(f"input_record_not_found:{input_id}")
			self._inputs.pop(input_id)
			self._emit("input.deleted", "input_record", input_id, {"id": input_id})
			return {"deleted": True, "id": input_id}
		except Exception as exc:
			_log.error("delete_input failed: %s", exc)
			raise

	async def get_input_cost_by_category(self, farm_parcel_id: str) -> dict[str, float]:
		"""Breakdown of input costs by category for a parcel."""
		inputs = [i for i in self._inputs.values() if i.get("farm_parcel_id") == farm_parcel_id]
		breakdown: dict[str, float] = {}
		for inp in inputs:
			cat = inp.get("category", "other")
			breakdown[cat] = breakdown.get(cat, 0.0) + inp.get("total_cost", 0.0)
		return {k: round(v, 2) for k, v in breakdown.items()}

	# ------------------------------------------------------------------ labour

	async def list_labour_schedules(self, farm_parcel_id: str | None = None,
									task_type: str | None = None, completed: bool | None = None,
									limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		items = list(self._labour.values())
		if farm_parcel_id:
			items = [l for l in items if l.get("farm_parcel_id") == farm_parcel_id]
		if task_type:
			items = [l for l in items if l.get("task_type") == task_type]
		if completed is not None:
			items = [l for l in items if l.get("completed") == completed]
		return items[offset: offset + limit]

	async def get_labour_schedule(self, schedule_id: str) -> dict[str, Any]:
		if schedule_id not in self._labour:
			raise KeyError(f"labour_schedule_not_found:{schedule_id}")
		return self._labour[schedule_id]

	async def create_labour_schedule(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			sched_id = _new_id("lab")
			ts = _now()
			worker_count = int(payload["worker_count"])
			daily_rate = float(payload["daily_rate"])
			duration_days = float(payload.get("duration_days", 1.0))
			record: dict[str, Any] = {
				"id": sched_id,
				"tenant_id": self.tenant_id,
				"farm_parcel_id": payload["farm_parcel_id"],
				"task_type": payload["task_type"],
				"scheduled_date": payload["scheduled_date"],
				"worker_count": worker_count,
				"daily_rate": daily_rate,
				"duration_days": duration_days,
				"total_labour_cost": round(worker_count * daily_rate * duration_days, 2),
				"actual_worker_count": None,
				"completed": False,
				"supervisor_id": payload.get("supervisor_id"),
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._labour[sched_id] = record
			self._emit("labour.created", "labour_schedule", sched_id, record)
			return record
		except Exception as exc:
			_log.error("create_labour_schedule failed: %s", exc)
			raise

	async def update_labour_schedule(self, schedule_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if schedule_id not in self._labour:
				raise KeyError(f"labour_schedule_not_found:{schedule_id}")
			record = self._labour[schedule_id]
			for field in ["scheduled_date", "worker_count", "actual_worker_count", "completed", "notes"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			# Recalculate if workers changed
			wc = record.get("actual_worker_count") or record.get("worker_count", 0)
			record["total_labour_cost"] = round(wc * record["daily_rate"] * record["duration_days"], 2)
			record["updated_at"] = _now()
			self._emit("labour.updated", "labour_schedule", schedule_id, payload)
			return record
		except Exception as exc:
			_log.error("update_labour_schedule failed: %s", exc)
			raise

	async def delete_labour_schedule(self, schedule_id: str) -> dict[str, Any]:
		try:
			if schedule_id not in self._labour:
				raise KeyError(f"labour_schedule_not_found:{schedule_id}")
			self._labour.pop(schedule_id)
			self._emit("labour.deleted", "labour_schedule", schedule_id, {"id": schedule_id})
			return {"deleted": True, "id": schedule_id}
		except Exception as exc:
			_log.error("delete_labour_schedule failed: %s", exc)
			raise

	async def get_labour_utilisation(self, farm_parcel_id: str) -> dict[str, Any]:
		"""Summarise scheduled vs completed labour for a parcel."""
		schedules = [l for l in self._labour.values() if l.get("farm_parcel_id") == farm_parcel_id]
		total = len(schedules)
		completed = len([l for l in schedules if l.get("completed")])
		total_cost = sum(l.get("total_labour_cost", 0) for l in schedules)
		return {
			"farm_parcel_id": farm_parcel_id,
			"total_schedules": total,
			"completed_schedules": completed,
			"completion_rate_pct": round(completed / total * 100, 1) if total > 0 else 0,
			"total_labour_cost": round(total_cost, 2),
		}

	# ------------------------------------------------------------------ diary

	async def list_diary_entries(self, farm_parcel_id: str | None = None,
								tag: str | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		items = list(self._diary.values())
		if farm_parcel_id:
			items = [d for d in items if d.get("farm_parcel_id") == farm_parcel_id]
		if tag:
			items = [d for d in items if tag in d.get("tags", [])]
		items = sorted(items, key=lambda x: x.get("entry_date", ""), reverse=True)
		return items[offset: offset + limit]

	async def get_diary_entry(self, entry_id: str) -> dict[str, Any]:
		if entry_id not in self._diary:
			raise KeyError(f"diary_entry_not_found:{entry_id}")
		return self._diary[entry_id]

	async def create_diary_entry(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			entry_id = _new_id("diy")
			ts = _now()
			record: dict[str, Any] = {
				"id": entry_id,
				"tenant_id": self.tenant_id,
				"farm_parcel_id": payload.get("farm_parcel_id"),
				"entry_date": payload["entry_date"],
				"title": payload["title"],
				"body": payload["body"],
				"tags": list(payload.get("tags", [])),
				"images": list(payload.get("images", [])),
				"author_id": payload.get("author_id"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._diary[entry_id] = record
			self._emit("diary.created", "diary_entry", entry_id, record)
			return record
		except Exception as exc:
			_log.error("create_diary_entry failed: %s", exc)
			raise

	async def update_diary_entry(self, entry_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if entry_id not in self._diary:
				raise KeyError(f"diary_entry_not_found:{entry_id}")
			record = self._diary[entry_id]
			for field in ["title", "body", "tags", "images"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("diary.updated", "diary_entry", entry_id, payload)
			return record
		except Exception as exc:
			_log.error("update_diary_entry failed: %s", exc)
			raise

	async def delete_diary_entry(self, entry_id: str) -> dict[str, Any]:
		try:
			if entry_id not in self._diary:
				raise KeyError(f"diary_entry_not_found:{entry_id}")
			self._diary.pop(entry_id)
			self._emit("diary.deleted", "diary_entry", entry_id, {"id": entry_id})
			return {"deleted": True, "id": entry_id}
		except Exception as exc:
			_log.error("delete_diary_entry failed: %s", exc)
			raise

	# ------------------------------------------------------------------ cost tracking

	async def get_farm_cost_summary(self, farm_parcel_id: str | None = None,
									from_date: str | None = None, to_date: str | None = None) -> dict[str, Any]:
		"""Aggregate cost summary across inputs and labour."""
		inputs = list(self._inputs.values())
		labour = list(self._labour.values())
		if farm_parcel_id:
			inputs = [i for i in inputs if i.get("farm_parcel_id") == farm_parcel_id]
			labour = [l for l in labour if l.get("farm_parcel_id") == farm_parcel_id]
		if from_date:
			inputs = [i for i in inputs if i.get("applied_date", "") >= from_date]
			labour = [l for l in labour if l.get("scheduled_date", "") >= from_date]
		if to_date:
			inputs = [i for i in inputs if i.get("applied_date", "") <= to_date]
			labour = [l for l in labour if l.get("scheduled_date", "") <= to_date]
		total_input = sum(i.get("total_cost", 0) for i in inputs)
		total_labour = sum(l.get("total_labour_cost", 0) for l in labour)
		breakdown: dict[str, float] = {}
		for inp in inputs:
			cat = inp.get("category", "other")
			breakdown[cat] = breakdown.get(cat, 0.0) + inp.get("total_cost", 0.0)
		breakdown["labour"] = total_labour
		parcel_area = 0.0
		if farm_parcel_id and farm_parcel_id in self._parcels:
			parcel_area = self._parcels[farm_parcel_id].get("area_ha", 0.0)
		return {
			"farm_parcel_id": farm_parcel_id,
			"total_input_cost": round(total_input, 2),
			"total_labour_cost": round(total_labour, 2),
			"total_cost": round(total_input + total_labour, 2),
			"cost_per_ha": round((total_input + total_labour) / parcel_area, 2) if parcel_area > 0 else None,
			"breakdown": {k: round(v, 2) for k, v in breakdown.items()},
		}

	async def get_input_usage_report(self, product_name: str) -> dict[str, Any]:
		"""Report total usage of a specific input product across all parcels."""
		records = [i for i in self._inputs.values() if i.get("product_name") == product_name]
		total_qty = sum(r.get("quantity", 0) for r in records)
		total_cost = sum(r.get("total_cost", 0) for r in records)
		parcels_used = list({r.get("farm_parcel_id") for r in records})
		return {
			"product_name": product_name,
			"application_count": len(records),
			"total_quantity": total_qty,
			"total_cost": round(total_cost, 2),
			"parcels_used": parcels_used,
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_parcels', '_inputs', '_labour', '_diary', '_audit']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

