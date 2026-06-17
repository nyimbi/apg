"""Irrigation Management service — agr_irg."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)
_CAPABILITY_ID = "agr_irg"


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_id(prefix: str = "") -> str:
	suffix = uuid4().hex[:12]
	return f"{prefix}-{suffix}" if prefix else suffix


class IrrigationManagementService:
	"""Async service for irrigation: sensor integration, schedule optimisation,
	water accounting, and canal management."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		if not tenant_id:
			raise ValueError("tenant_id required")
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self._sensors = WriteThruDict('sensors', tenant_id, _store)
		self._readings = WriteThruDict('readings', tenant_id, _store)
		self._schedules = WriteThruDict('schedules', tenant_id, _store)
		self._canals = WriteThruDict('canals', tenant_id, _store)
		self._water_accounts = WriteThruDict('water_accounts', tenant_id, _store)
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
				"sensors": len(self._sensors),
				"readings": len(self._readings),
				"schedules": len(self._schedules),
				"canals": len(self._canals),
				"water_accounts": len(self._water_accounts),
			},
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": _CAPABILITY_ID,
			"name": "Irrigation Management",
			"domain": "agriculture",
			"version": "1.0.0",
			"description": "Sensor integration, irrigation schedule optimisation, water accounting, canal management.",
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		return self._audit[-limit:]

	# ------------------------------------------------------------------ sensors

	async def list_sensors(self, farm_parcel_id: str | None = None, active: bool | None = None) -> list[dict[str, Any]]:
		items = list(self._sensors.values())
		if farm_parcel_id:
			items = [s for s in items if s.get("farm_parcel_id") == farm_parcel_id]
		if active is not None:
			items = [s for s in items if s.get("active") == active]
		return items

	async def get_sensor(self, sensor_id: str) -> dict[str, Any]:
		if sensor_id not in self._sensors:
			raise KeyError(f"sensor_not_found:{sensor_id}")
		return self._sensors[sensor_id]

	async def create_sensor(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			sensor_id = _new_id("sen")
			ts = _now()
			record: dict[str, Any] = {
				"id": sensor_id,
				"tenant_id": self.tenant_id,
				"name": payload["name"],
				"sensor_type": payload["sensor_type"],
				"farm_parcel_id": payload["farm_parcel_id"],
				"location_lat": payload.get("location_lat"),
				"location_lng": payload.get("location_lng"),
				"unit": payload["unit"],
				"min_threshold": payload.get("min_threshold"),
				"max_threshold": payload.get("max_threshold"),
				"last_reading": None,
				"last_reading_at": None,
				"active": True,
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._sensors[sensor_id] = record
			self._emit("sensor.created", "sensor", sensor_id, record)
			return record
		except Exception as exc:
			_log.error("create_sensor failed: %s", exc)
			raise

	async def update_sensor(self, sensor_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if sensor_id not in self._sensors:
				raise KeyError(f"sensor_not_found:{sensor_id}")
			record = self._sensors[sensor_id]
			for field in ["name", "min_threshold", "max_threshold", "notes", "active"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("sensor.updated", "sensor", sensor_id, payload)
			return record
		except Exception as exc:
			_log.error("update_sensor failed: %s", exc)
			raise

	async def delete_sensor(self, sensor_id: str) -> dict[str, Any]:
		try:
			if sensor_id not in self._sensors:
				raise KeyError(f"sensor_not_found:{sensor_id}")
			self._sensors.pop(sensor_id)
			self._emit("sensor.deleted", "sensor", sensor_id, {"id": sensor_id})
			return {"deleted": True, "id": sensor_id}
		except Exception as exc:
			_log.error("delete_sensor failed: %s", exc)
			raise

	# ------------------------------------------------------------------ readings

	async def ingest_reading(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Ingest a sensor reading, check thresholds and flag alerts."""
		try:
			sensor_id = payload["sensor_id"]
			if sensor_id not in self._sensors:
				raise KeyError(f"sensor_not_found:{sensor_id}")
			sensor = self._sensors[sensor_id]
			value = float(payload["value"])
			ts = payload.get("recorded_at") or _now()
			reading_id = _new_id("rdg")
			alert = False
			if sensor.get("min_threshold") is not None and value < sensor["min_threshold"]:
				alert = True
			if sensor.get("max_threshold") is not None and value > sensor["max_threshold"]:
				alert = True
			record: dict[str, Any] = {
				"id": reading_id,
				"tenant_id": self.tenant_id,
				"sensor_id": sensor_id,
				"value": value,
				"recorded_at": ts,
				"quality_flag": payload.get("quality_flag"),
				"alert_triggered": alert,
				"created_at": _now(),
			}
			self._readings[reading_id] = record
			sensor["last_reading"] = value
			sensor["last_reading_at"] = ts
			self._emit("reading.ingested", "sensor_reading", reading_id, {"sensor_id": sensor_id, "value": value, "alert": alert})
			return record
		except Exception as exc:
			_log.error("ingest_reading failed: %s", exc)
			raise

	async def list_readings(self, sensor_id: str, limit: int = 100) -> list[dict[str, Any]]:
		items = [r for r in self._readings.values() if r.get("sensor_id") == sensor_id]
		return sorted(items, key=lambda x: x.get("recorded_at", ""))[-limit:]

	async def get_sensor_alerts(self, farm_parcel_id: str | None = None) -> list[dict[str, Any]]:
		"""Return readings that triggered threshold alerts."""
		readings = [r for r in self._readings.values() if r.get("alert_triggered")]
		if farm_parcel_id:
			sensor_ids = {s["id"] for s in self._sensors.values() if s.get("farm_parcel_id") == farm_parcel_id}
			readings = [r for r in readings if r.get("sensor_id") in sensor_ids]
		return sorted(readings, key=lambda x: x.get("recorded_at", ""), reverse=True)

	async def get_soil_moisture_status(self, farm_parcel_id: str) -> dict[str, Any]:
		"""Return current soil moisture readings for a parcel's sensors."""
		sensors = [s for s in self._sensors.values()
				if s.get("farm_parcel_id") == farm_parcel_id and s.get("sensor_type") == "soil_moisture"]
		readings = []
		for s in sensors:
			readings.append({
				"sensor_id": s["id"],
				"sensor_name": s["name"],
				"value": s.get("last_reading"),
				"unit": s.get("unit"),
				"last_reading_at": s.get("last_reading_at"),
				"min_threshold": s.get("min_threshold"),
				"max_threshold": s.get("max_threshold"),
				"status": _classify_moisture(s),
			})
		return {"farm_parcel_id": farm_parcel_id, "sensors": readings}

	# ------------------------------------------------------------------ schedules

	async def list_schedules(self, farm_parcel_id: str | None = None, status: str | None = None,
							limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		items = list(self._schedules.values())
		if farm_parcel_id:
			items = [s for s in items if s.get("farm_parcel_id") == farm_parcel_id]
		if status:
			items = [s for s in items if s.get("status") == status]
		return items[offset: offset + limit]

	async def get_schedule(self, schedule_id: str) -> dict[str, Any]:
		if schedule_id not in self._schedules:
			raise KeyError(f"schedule_not_found:{schedule_id}")
		return self._schedules[schedule_id]

	async def create_schedule(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			sched_id = _new_id("sch")
			ts = _now()
			record: dict[str, Any] = {
				"id": sched_id,
				"tenant_id": self.tenant_id,
				"farm_parcel_id": payload["farm_parcel_id"],
				"method": payload["method"],
				"scheduled_start": payload["scheduled_start"],
				"duration_minutes": int(payload["duration_minutes"]),
				"volume_m3": payload.get("volume_m3"),
				"trigger_condition": payload.get("trigger_condition"),
				"status": "scheduled",
				"actual_start": None,
				"actual_end": None,
				"actual_volume_m3": None,
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._schedules[sched_id] = record
			self._emit("schedule.created", "irrigation_schedule", sched_id, record)
			return record
		except Exception as exc:
			_log.error("create_schedule failed: %s", exc)
			raise

	async def update_schedule(self, schedule_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if schedule_id not in self._schedules:
				raise KeyError(f"schedule_not_found:{schedule_id}")
			record = self._schedules[schedule_id]
			for field in ["scheduled_start", "duration_minutes", "volume_m3", "status",
						"actual_start", "actual_end", "actual_volume_m3", "notes"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			# Track water usage when completed
			if record.get("status") == "completed" and record.get("actual_volume_m3"):
				self._record_water_usage(record["farm_parcel_id"], float(record["actual_volume_m3"]))
			self._emit("schedule.updated", "irrigation_schedule", schedule_id, payload)
			return record
		except Exception as exc:
			_log.error("update_schedule failed: %s", exc)
			raise

	async def delete_schedule(self, schedule_id: str) -> dict[str, Any]:
		try:
			if schedule_id not in self._schedules:
				raise KeyError(f"schedule_not_found:{schedule_id}")
			self._schedules.pop(schedule_id)
			self._emit("schedule.deleted", "irrigation_schedule", schedule_id, {"id": schedule_id})
			return {"deleted": True, "id": schedule_id}
		except Exception as exc:
			_log.error("delete_schedule failed: %s", exc)
			raise

	async def optimise_schedule(self, farm_parcel_id: str, crop_type: str, soil_moisture_pct: float) -> dict[str, Any]:
		"""Suggest optimal irrigation based on crop water needs and current moisture."""
		# Simplified agronomic rules — in production these would draw on ET0 / Kc tables
		needs: dict[str, dict[str, Any]] = {
			"maize": {"min_moisture": 40, "target_moisture": 65, "duration_base_min": 60},
			"wheat": {"min_moisture": 35, "target_moisture": 55, "duration_base_min": 45},
			"sugarcane": {"min_moisture": 50, "target_moisture": 75, "duration_base_min": 90},
			"vegetables": {"min_moisture": 55, "target_moisture": 70, "duration_base_min": 30},
		}
		crop_needs = needs.get(crop_type.lower(), {"min_moisture": 40, "target_moisture": 60, "duration_base_min": 60})
		deficit = max(0, crop_needs["target_moisture"] - soil_moisture_pct)
		if deficit == 0:
			return {"farm_parcel_id": farm_parcel_id, "recommendation": "no_irrigation_needed", "deficit_pct": 0}
		duration = int(crop_needs["duration_base_min"] * deficit / 20)
		return {
			"farm_parcel_id": farm_parcel_id,
			"recommendation": "irrigate",
			"suggested_duration_minutes": duration,
			"deficit_pct": round(deficit, 1),
			"target_moisture_pct": crop_needs["target_moisture"],
		}

	# ------------------------------------------------------------------ water accounting

	def _record_water_usage(self, farm_parcel_id: str, volume_m3: float) -> None:
		period = _now()[:7]  # YYYY-MM
		key = f"{farm_parcel_id}:{period}"
		if key not in self._water_accounts:
			acct_id = _new_id("wac")
			self._water_accounts[key] = {
				"id": acct_id,
				"tenant_id": self.tenant_id,
				"farm_parcel_id": farm_parcel_id,
				"period": period,
				"allocated_m3": 1000.0,  # default allocation
				"used_m3": 0.0,
				"balance_m3": 1000.0,
				"created_at": _now(),
			}
		acct = self._water_accounts[key]
		acct["used_m3"] = round(acct["used_m3"] + volume_m3, 3)
		acct["balance_m3"] = round(acct["allocated_m3"] - acct["used_m3"], 3)

	async def list_water_accounts(self, farm_parcel_id: str | None = None, period: str | None = None) -> list[dict[str, Any]]:
		items = list(self._water_accounts.values())
		if farm_parcel_id:
			items = [a for a in items if a.get("farm_parcel_id") == farm_parcel_id]
		if period:
			items = [a for a in items if a.get("period") == period]
		return items

	async def set_water_allocation(self, farm_parcel_id: str, period: str, allocated_m3: float) -> dict[str, Any]:
		"""Set water allocation for a parcel/period."""
		try:
			key = f"{farm_parcel_id}:{period}"
			if key not in self._water_accounts:
				acct_id = _new_id("wac")
				self._water_accounts[key] = {
					"id": acct_id,
					"tenant_id": self.tenant_id,
					"farm_parcel_id": farm_parcel_id,
					"period": period,
					"allocated_m3": allocated_m3,
					"used_m3": 0.0,
					"balance_m3": allocated_m3,
					"created_at": _now(),
				}
			else:
				acct = self._water_accounts[key]
				acct["allocated_m3"] = allocated_m3
				acct["balance_m3"] = round(allocated_m3 - acct["used_m3"], 3)
			self._emit("water_account.allocated", "water_account", key, {"farm_parcel_id": farm_parcel_id, "period": period, "allocated_m3": allocated_m3})
			return self._water_accounts[key]
		except Exception as exc:
			_log.error("set_water_allocation failed: %s", exc)
			raise

	# ------------------------------------------------------------------ canals

	async def list_canals(self) -> list[dict[str, Any]]:
		return list(self._canals.values())

	async def get_canal(self, canal_id: str) -> dict[str, Any]:
		if canal_id not in self._canals:
			raise KeyError(f"canal_not_found:{canal_id}")
		return self._canals[canal_id]

	async def create_canal(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			canal_id = _new_id("can")
			ts = _now()
			record: dict[str, Any] = {
				"id": canal_id,
				"tenant_id": self.tenant_id,
				"name": payload["name"],
				"length_m": float(payload["length_m"]),
				"capacity_m3_s": float(payload["capacity_m3_s"]),
				"served_parcels": list(payload.get("served_parcels", [])),
				"maintenance_due": payload.get("maintenance_due"),
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._canals[canal_id] = record
			self._emit("canal.created", "canal", canal_id, record)
			return record
		except Exception as exc:
			_log.error("create_canal failed: %s", exc)
			raise

	async def update_canal(self, canal_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if canal_id not in self._canals:
				raise KeyError(f"canal_not_found:{canal_id}")
			record = self._canals[canal_id]
			for field in ["name", "length_m", "capacity_m3_s", "served_parcels", "maintenance_due", "notes"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("canal.updated", "canal", canal_id, payload)
			return record
		except Exception as exc:
			_log.error("update_canal failed: %s", exc)
			raise

	async def delete_canal(self, canal_id: str) -> dict[str, Any]:
		try:
			if canal_id not in self._canals:
				raise KeyError(f"canal_not_found:{canal_id}")
			self._canals.pop(canal_id)
			self._emit("canal.deleted", "canal", canal_id, {"id": canal_id})
			return {"deleted": True, "id": canal_id}
		except Exception as exc:
			_log.error("delete_canal failed: %s", exc)
			raise

	async def get_canal_utilisation(self, canal_id: str) -> dict[str, Any]:
		"""Return utilisation stats for a canal."""
		if canal_id not in self._canals:
			raise KeyError(f"canal_not_found:{canal_id}")
		canal = self._canals[canal_id]
		served = canal.get("served_parcels", [])
		schedules = [s for s in self._schedules.values() if s.get("farm_parcel_id") in served]
		total_vol = sum(s.get("actual_volume_m3") or 0 for s in schedules if s.get("status") == "completed")
		return {
			"canal_id": canal_id,
			"name": canal["name"],
			"capacity_m3_s": canal["capacity_m3_s"],
			"served_parcels": len(served),
			"completed_irrigations": len([s for s in schedules if s.get("status") == "completed"]),
			"total_volume_delivered_m3": round(total_vol, 3),
		}

	async def get_irrigation_efficiency_report(self, farm_parcel_id: str) -> dict[str, Any]:
		"""Compare scheduled vs actual irrigation volumes."""
		schedules = [s for s in self._schedules.values()
					if s.get("farm_parcel_id") == farm_parcel_id and s.get("status") == "completed"]
		planned = sum(s.get("volume_m3") or 0 for s in schedules)
		actual = sum(s.get("actual_volume_m3") or 0 for s in schedules)
		efficiency = round(actual / planned * 100, 1) if planned > 0 else None
		return {
			"farm_parcel_id": farm_parcel_id,
			"completed_irrigations": len(schedules),
			"planned_volume_m3": round(planned, 3),
			"actual_volume_m3": round(actual, 3),
			"efficiency_pct": efficiency,
		}


def _classify_moisture(sensor: dict[str, Any]) -> str:
	val = sensor.get("last_reading")
	if val is None:
		return "unknown"
	lo = sensor.get("min_threshold")
	hi = sensor.get("max_threshold")
	if lo is not None and val < lo:
		return "low"
	if hi is not None and val > hi:
		return "high"
	return "optimal"

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_sensors', '_readings', '_schedules', '_canals', '_water_accounts', '_audit']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

