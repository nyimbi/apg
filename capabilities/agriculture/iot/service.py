"""AgriIoT & Precision Farming service — agr_iot."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)
_CAPABILITY_ID = "agr_iot"


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_id(prefix: str = "") -> str:
	suffix = uuid4().hex[:12]
	return f"{prefix}-{suffix}" if prefix else suffix


class AgriIoTService:
	"""Async service for AgriIoT & precision farming: soil sensor ingestion,
	drone imagery analysis, yield mapping, and variable-rate prescriptions."""

	def __init__(self, tenant_id: str = "default") -> None:
		if not tenant_id:
			raise ValueError("tenant_id required")
		self.tenant_id = tenant_id
		self._devices: dict[str, dict[str, Any]] = {}
		self._telemetry: dict[str, dict[str, Any]] = {}
		self._imagery: dict[str, dict[str, Any]] = {}
		self._yield_maps: dict[str, dict[str, Any]] = {}
		self._prescriptions: dict[str, dict[str, Any]] = {}
		self._audit: list[dict[str, Any]] = []

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
				"devices": len(self._devices),
				"telemetry_records": len(self._telemetry),
				"drone_images": len(self._imagery),
				"yield_maps": len(self._yield_maps),
				"prescriptions": len(self._prescriptions),
			},
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": _CAPABILITY_ID,
			"name": "AgriIoT & Precision Farming",
			"domain": "agriculture",
			"version": "1.0.0",
			"description": "Soil sensor ingestion, drone imagery analysis, yield mapping, variable rate prescriptions.",
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		return self._audit[-limit:]

	# ------------------------------------------------------------------ devices

	async def list_devices(self, farm_parcel_id: str | None = None, device_type: str | None = None,
						active: bool | None = None) -> list[dict[str, Any]]:
		items = list(self._devices.values())
		if farm_parcel_id:
			items = [d for d in items if d.get("farm_parcel_id") == farm_parcel_id]
		if device_type:
			items = [d for d in items if d.get("device_type") == device_type]
		if active is not None:
			items = [d for d in items if d.get("active") == active]
		return items

	async def get_device(self, device_id: str) -> dict[str, Any]:
		if device_id not in self._devices:
			raise KeyError(f"device_not_found:{device_id}")
		return self._devices[device_id]

	async def register_device(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			did = _new_id("dev")
			ts = _now()
			record: dict[str, Any] = {
				"id": did,
				"tenant_id": self.tenant_id,
				"name": payload["name"],
				"device_type": payload["device_type"],
				"farm_parcel_id": payload["farm_parcel_id"],
				"location_lat": payload.get("location_lat"),
				"location_lng": payload.get("location_lng"),
				"serial_number": payload.get("serial_number"),
				"firmware_version": payload.get("firmware_version"),
				"calibration_date": payload.get("calibration_date"),
				"active": True,
				"last_telemetry_at": None,
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._devices[did] = record
			self._emit("device.registered", "device", did, record)
			return record
		except Exception as exc:
			_log.error("register_device failed: %s", exc)
			raise

	async def update_device(self, device_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if device_id not in self._devices:
				raise KeyError(f"device_not_found:{device_id}")
			record = self._devices[device_id]
			for field in ["name", "firmware_version", "calibration_date", "active", "notes"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("device.updated", "device", device_id, payload)
			return record
		except Exception as exc:
			_log.error("update_device failed: %s", exc)
			raise

	async def delete_device(self, device_id: str) -> dict[str, Any]:
		try:
			if device_id not in self._devices:
				raise KeyError(f"device_not_found:{device_id}")
			self._devices.pop(device_id)
			self._emit("device.deleted", "device", device_id, {"id": device_id})
			return {"deleted": True, "id": device_id}
		except Exception as exc:
			_log.error("delete_device failed: %s", exc)
			raise

	# ------------------------------------------------------------------ telemetry

	async def ingest_telemetry(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Ingest IoT telemetry from a device."""
		try:
			device_id = payload["device_id"]
			if device_id not in self._devices:
				raise KeyError(f"device_not_found:{device_id}")
			tid = _new_id("tel")
			ts = _now()
			record: dict[str, Any] = {
				"id": tid,
				"tenant_id": self.tenant_id,
				"device_id": device_id,
				"readings": dict(payload["readings"]),
				"recorded_at": payload.get("recorded_at") or ts,
				"gps_lat": payload.get("gps_lat"),
				"gps_lng": payload.get("gps_lng"),
				"created_at": ts,
			}
			self._telemetry[tid] = record
			self._devices[device_id]["last_telemetry_at"] = record["recorded_at"]
			self._emit("telemetry.ingested", "telemetry", tid, {"device_id": device_id})
			return record
		except Exception as exc:
			_log.error("ingest_telemetry failed: %s", exc)
			raise

	async def list_telemetry(self, device_id: str, limit: int = 100) -> list[dict[str, Any]]:
		items = [t for t in self._telemetry.values() if t.get("device_id") == device_id]
		return sorted(items, key=lambda x: x.get("recorded_at", ""), reverse=True)[:limit]

	async def get_field_health_snapshot(self, farm_parcel_id: str) -> dict[str, Any]:
		"""Aggregate latest sensor readings across all devices on a parcel."""
		devices = [d for d in self._devices.values()
				if d.get("farm_parcel_id") == farm_parcel_id and d.get("active")]
		snapshot = []
		for dev in devices:
			latest = [t for t in self._telemetry.values() if t.get("device_id") == dev["id"]]
			if latest:
				newest = sorted(latest, key=lambda x: x.get("recorded_at", ""), reverse=True)[0]
				snapshot.append({
					"device_id": dev["id"],
					"device_name": dev["name"],
					"device_type": dev["device_type"],
					"readings": newest["readings"],
					"recorded_at": newest["recorded_at"],
				})
		return {"farm_parcel_id": farm_parcel_id, "device_count": len(devices), "readings": snapshot}

	# ------------------------------------------------------------------ drone imagery

	async def list_imagery(self, farm_parcel_id: str | None = None, imagery_type: str | None = None,
						limit: int = 50) -> list[dict[str, Any]]:
		items = list(self._imagery.values())
		if farm_parcel_id:
			items = [i for i in items if i.get("farm_parcel_id") == farm_parcel_id]
		if imagery_type:
			items = [i for i in items if i.get("imagery_type") == imagery_type]
		return sorted(items, key=lambda x: x.get("captured_at", ""), reverse=True)[:limit]

	async def get_imagery(self, image_id: str) -> dict[str, Any]:
		if image_id not in self._imagery:
			raise KeyError(f"imagery_not_found:{image_id}")
		return self._imagery[image_id]

	async def upload_imagery(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Register drone imagery and run basic NDVI zone analysis."""
		try:
			iid = _new_id("img")
			ts = _now()
			record: dict[str, Any] = {
				"id": iid,
				"tenant_id": self.tenant_id,
				"farm_parcel_id": payload["farm_parcel_id"],
				"drone_id": payload.get("drone_id"),
				"imagery_type": payload["imagery_type"],
				"captured_at": payload["captured_at"],
				"file_url": payload["file_url"],
				"resolution_cm": payload.get("resolution_cm"),
				"coverage_ha": payload.get("coverage_ha"),
				"ndvi_mean": payload.get("ndvi_mean"),
				"ndvi_min": payload.get("ndvi_min"),
				"ndvi_max": payload.get("ndvi_max"),
				"zone_analysis": list(payload.get("zone_analysis", [])),
				"notes": payload.get("notes"),
				"created_at": ts,
			}
			# Auto-generate zone analysis from NDVI if values are present and zones are absent
			if not record["zone_analysis"] and record["ndvi_mean"] is not None:
				record["zone_analysis"] = _generate_ndvi_zones(
					record["ndvi_mean"], record["ndvi_min"], record["ndvi_max"])
			self._imagery[iid] = record
			self._emit("imagery.uploaded", "drone_imagery", iid, {"farm_parcel_id": payload["farm_parcel_id"]})
			return record
		except Exception as exc:
			_log.error("upload_imagery failed: %s", exc)
			raise

	async def delete_imagery(self, image_id: str) -> dict[str, Any]:
		try:
			if image_id not in self._imagery:
				raise KeyError(f"imagery_not_found:{image_id}")
			self._imagery.pop(image_id)
			self._emit("imagery.deleted", "drone_imagery", image_id, {"id": image_id})
			return {"deleted": True, "id": image_id}
		except Exception as exc:
			_log.error("delete_imagery failed: %s", exc)
			raise

	async def analyse_ndvi_trend(self, farm_parcel_id: str) -> dict[str, Any]:
		"""Compute NDVI trend over multiple flights for a parcel."""
		images = [i for i in self._imagery.values()
				if i.get("farm_parcel_id") == farm_parcel_id and i.get("ndvi_mean") is not None]
		if not images:
			return {"farm_parcel_id": farm_parcel_id, "data_points": 0}
		images = sorted(images, key=lambda x: x.get("captured_at", ""))
		series = [{"date": i["captured_at"], "ndvi_mean": i["ndvi_mean"]} for i in images]
		means = [i["ndvi_mean"] for i in images]
		trend = "stable"
		if len(means) >= 2:
			delta = means[-1] - means[0]
			if delta > 0.05:
				trend = "improving"
			elif delta < -0.05:
				trend = "declining"
		return {
			"farm_parcel_id": farm_parcel_id,
			"data_points": len(images),
			"latest_ndvi_mean": means[-1],
			"trend": trend,
			"series": series,
		}

	# ------------------------------------------------------------------ yield maps

	async def list_yield_maps(self, farm_parcel_id: str | None = None, season: str | None = None) -> list[dict[str, Any]]:
		items = list(self._yield_maps.values())
		if farm_parcel_id:
			items = [y for y in items if y.get("farm_parcel_id") == farm_parcel_id]
		if season:
			items = [y for y in items if y.get("season") == season]
		return items

	async def create_yield_map(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Create a spatial yield map from zone data."""
		try:
			ymid = _new_id("ymp")
			ts = _now()
			zones = list(payload["zones"])
			total_yield = sum(z.get("yield_kg", 0) for z in zones)
			total_area = sum(z.get("area_ha", 0) for z in zones)
			avg_yield_ha = round(total_yield / total_area, 2) if total_area > 0 else 0
			record: dict[str, Any] = {
				"id": ymid,
				"tenant_id": self.tenant_id,
				"farm_parcel_id": payload["farm_parcel_id"],
				"crop_id": payload["crop_id"],
				"season": payload["season"],
				"harvest_date": payload["harvest_date"],
				"zones": zones,
				"total_yield_kg": round(total_yield, 2),
				"avg_yield_kg_ha": avg_yield_ha,
				"equipment_id": payload.get("equipment_id"),
				"notes": payload.get("notes"),
				"created_at": ts,
			}
			self._yield_maps[ymid] = record
			self._emit("yield_map.created", "yield_map", ymid, record)
			return record
		except Exception as exc:
			_log.error("create_yield_map failed: %s", exc)
			raise

	async def delete_yield_map(self, map_id: str) -> dict[str, Any]:
		try:
			if map_id not in self._yield_maps:
				raise KeyError(f"yield_map_not_found:{map_id}")
			self._yield_maps.pop(map_id)
			self._emit("yield_map.deleted", "yield_map", map_id, {"id": map_id})
			return {"deleted": True, "id": map_id}
		except Exception as exc:
			_log.error("delete_yield_map failed: %s", exc)
			raise

	# ------------------------------------------------------------------ prescriptions

	async def list_prescriptions(self, farm_parcel_id: str | None = None, applied: bool | None = None) -> list[dict[str, Any]]:
		items = list(self._prescriptions.values())
		if farm_parcel_id:
			items = [p for p in items if p.get("farm_parcel_id") == farm_parcel_id]
		if applied is not None:
			items = [p for p in items if p.get("applied") == applied]
		return items

	async def get_prescription(self, prescription_id: str) -> dict[str, Any]:
		if prescription_id not in self._prescriptions:
			raise KeyError(f"prescription_not_found:{prescription_id}")
		return self._prescriptions[prescription_id]

	async def create_prescription(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Create a variable-rate application prescription map."""
		try:
			prid = _new_id("prx")
			ts = _now()
			zones = list(payload["zones"])
			total_area = sum(z.get("area_ha", 0) for z in zones)
			record: dict[str, Any] = {
				"id": prid,
				"tenant_id": self.tenant_id,
				"farm_parcel_id": payload["farm_parcel_id"],
				"crop_id": payload.get("crop_id"),
				"application_type": payload["application_type"],
				"zones": zones,
				"total_area_ha": round(total_area, 4),
				"generated_from": payload.get("generated_from"),
				"applied": False,
				"applied_at": None,
				"notes": payload.get("notes"),
				"created_at": ts,
			}
			self._prescriptions[prid] = record
			self._emit("prescription.created", "prescription", prid, record)
			return record
		except Exception as exc:
			_log.error("create_prescription failed: %s", exc)
			raise

	async def mark_prescription_applied(self, prescription_id: str) -> dict[str, Any]:
		"""Mark a prescription as applied to the field."""
		try:
			if prescription_id not in self._prescriptions:
				raise KeyError(f"prescription_not_found:{prescription_id}")
			self._prescriptions[prescription_id]["applied"] = True
			self._prescriptions[prescription_id]["applied_at"] = _now()
			self._emit("prescription.applied", "prescription", prescription_id, {"id": prescription_id})
			return self._prescriptions[prescription_id]
		except Exception as exc:
			_log.error("mark_prescription_applied failed: %s", exc)
			raise

	async def delete_prescription(self, prescription_id: str) -> dict[str, Any]:
		try:
			if prescription_id not in self._prescriptions:
				raise KeyError(f"prescription_not_found:{prescription_id}")
			self._prescriptions.pop(prescription_id)
			self._emit("prescription.deleted", "prescription", prescription_id, {"id": prescription_id})
			return {"deleted": True, "id": prescription_id}
		except Exception as exc:
			_log.error("delete_prescription failed: %s", exc)
			raise

	async def generate_prescription_from_ndvi(self, farm_parcel_id: str, application_type: str,
											base_rate: float, unit: str) -> dict[str, Any]:
		"""Generate a variable-rate prescription from the latest NDVI imagery."""
		# Get latest NDVI image
		ndvi_images = [i for i in self._imagery.values()
					if i.get("farm_parcel_id") == farm_parcel_id and i.get("imagery_type") == "ndvi"]
		if not ndvi_images:
			raise ValueError("no_ndvi_imagery_available")
		latest = sorted(ndvi_images, key=lambda x: x.get("captured_at", ""), reverse=True)[0]
		zones = latest.get("zone_analysis", [])
		if not zones:
			raise ValueError("no_zone_analysis_in_imagery")
		# Invert NDVI to prescribe: low NDVI → high input rate
		prescription_zones = []
		for z in zones:
			ndvi = z.get("ndvi_mean", 0.5)
			# Rate modifier: stressed zones get up to 1.5× base rate
			rate_multiplier = max(0.5, 1.5 - ndvi)
			prescription_zones.append({
				"zone_id": z.get("zone_id", _new_id("z")),
				"area_ha": z.get("area_ha", 0),
				"ndvi_mean": ndvi,
				"status": z.get("status", "unknown"),
				"application_rate": round(base_rate * rate_multiplier, 2),
				"unit": unit,
			})
		payload = {
			"farm_parcel_id": farm_parcel_id,
			"application_type": application_type,
			"zones": prescription_zones,
			"generated_from": latest["id"],
		}
		return await self.create_prescription(payload)

	async def get_precision_farming_summary(self, farm_parcel_id: str) -> dict[str, Any]:
		"""Precision farming status for a parcel."""
		devices = [d for d in self._devices.values() if d.get("farm_parcel_id") == farm_parcel_id]
		imagery = [i for i in self._imagery.values() if i.get("farm_parcel_id") == farm_parcel_id]
		yield_maps = [y for y in self._yield_maps.values() if y.get("farm_parcel_id") == farm_parcel_id]
		prescriptions = [p for p in self._prescriptions.values() if p.get("farm_parcel_id") == farm_parcel_id]
		latest_ndvi = None
		ndvi_images = [i for i in imagery if i.get("imagery_type") == "ndvi" and i.get("ndvi_mean")]
		if ndvi_images:
			latest_ndvi = sorted(ndvi_images, key=lambda x: x.get("captured_at", ""), reverse=True)[0].get("ndvi_mean")
		return {
			"farm_parcel_id": farm_parcel_id,
			"active_devices": len([d for d in devices if d.get("active")]),
			"drone_flights": len(imagery),
			"latest_ndvi_mean": latest_ndvi,
			"yield_maps": len(yield_maps),
			"prescriptions_created": len(prescriptions),
			"prescriptions_applied": len([p for p in prescriptions if p.get("applied")]),
		}


def _generate_ndvi_zones(mean: float, minimum: float | None, maximum: float | None) -> list[dict[str, Any]]:
	"""Synthesise simplified zone analysis from aggregate NDVI values."""
	lo = minimum if minimum is not None else mean - 0.1
	hi = maximum if maximum is not None else mean + 0.1
	zones = []
	# Three generic zones
	for i, (ndvi_val, label, area_frac) in enumerate([
		(hi, "high_vigor", 0.3),
		(mean, "medium_vigor", 0.5),
		(lo, "low_vigor", 0.2),
	]):
		status = "optimal" if ndvi_val > 0.5 else ("stressed" if ndvi_val > 0.3 else "critical")
		zones.append({
			"zone_id": f"z{i+1}",
			"ndvi_mean": round(ndvi_val, 3),
			"label": label,
			"status": status,
			"area_frac": area_frac,
			"area_ha": None,  # unknown without parcel size
		})
	return zones
