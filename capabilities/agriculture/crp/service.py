"""Crop Management service — agr_crp."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

_CAPABILITY_ID = "agr_crp"


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_id(prefix: str = "") -> str:
	suffix = uuid4().hex[:12]
	return f"{prefix}-{suffix}" if prefix else suffix


class CropManagementService:
	"""Async service for crop management: planting calendars, phenology,
	variety registry, rotation planning, and yield recording."""

	def __init__(self, tenant_id: str = "default") -> None:
		if not tenant_id:
			raise ValueError("tenant_id required")
		self.tenant_id = tenant_id
		self._varieties: dict[str, dict[str, Any]] = {}
		self._calendars: dict[str, dict[str, Any]] = {}
		self._crops: dict[str, dict[str, Any]] = {}
		self._phenology: dict[str, dict[str, Any]] = {}
		self._rotations: dict[str, dict[str, Any]] = {}
		self._yields: dict[str, dict[str, Any]] = {}
		self._audit: list[dict[str, Any]] = []

	# ------------------------------------------------------------------ helpers

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
		"""Return service health and entity counts."""
		return {
			"status": "ok",
			"capability": _CAPABILITY_ID,
			"tenant_id": self.tenant_id,
			"counts": {
				"varieties": len(self._varieties),
				"calendars": len(self._calendars),
				"crops": len(self._crops),
				"phenology_observations": len(self._phenology),
				"rotation_plans": len(self._rotations),
				"yield_records": len(self._yields),
			},
		}

	async def describe(self) -> dict[str, Any]:
		"""Return capability descriptor."""
		return {
			"capability_id": _CAPABILITY_ID,
			"name": "Crop Management",
			"domain": "agriculture",
			"version": "1.0.0",
			"description": (
				"Planting calendar, phenology tracking, variety registry, "
				"crop rotation planning, yield recording."
			),
			"endpoints": [
				"varieties", "planting_calendars", "crops",
				"phenology", "rotation_plans", "yield_records",
			],
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		"""Return most recent audit events."""
		return self._audit[-limit:]

	# ------------------------------------------------------------------ variety

	async def list_varieties(self, crop_type: str | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		"""List registered crop varieties, optionally filtered by crop type."""
		items = list(self._varieties.values())
		if crop_type:
			items = [v for v in items if v.get("crop_type") == crop_type]
		return items[offset: offset + limit]

	async def get_variety(self, variety_id: str) -> dict[str, Any]:
		"""Fetch a variety by ID."""
		if variety_id not in self._varieties:
			raise KeyError(f"variety_not_found:{variety_id}")
		return self._varieties[variety_id]

	async def create_variety(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Register a new crop variety."""
		try:
			variety_id = _new_id("var")
			ts = _now()
			record: dict[str, Any] = {
				"id": variety_id,
				"tenant_id": self.tenant_id,
				"name": payload["name"],
				"crop_type": payload["crop_type"],
				"maturity_days": int(payload["maturity_days"]),
				"yield_potential_kg_ha": float(payload.get("yield_potential_kg_ha", 0)),
				"drought_tolerance": payload.get("drought_tolerance"),
				"disease_resistance": list(payload.get("disease_resistance", [])),
				"optimal_rainfall_mm": payload.get("optimal_rainfall_mm"),
				"optimal_temp_min_c": payload.get("optimal_temp_min_c"),
				"optimal_temp_max_c": payload.get("optimal_temp_max_c"),
				"notes": payload.get("notes"),
				"metadata": dict(payload.get("metadata", {})),
				"created_at": ts,
				"updated_at": ts,
			}
			self._varieties[variety_id] = record
			self._emit("variety.created", "variety", variety_id, record)
			return record
		except Exception as exc:
			_log.error("create_variety failed: %s", exc)
			raise

	async def update_variety(self, variety_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		"""Update an existing variety."""
		try:
			if variety_id not in self._varieties:
				raise KeyError(f"variety_not_found:{variety_id}")
			record = self._varieties[variety_id]
			updatable = [
				"name", "maturity_days", "yield_potential_kg_ha",
				"drought_tolerance", "disease_resistance", "notes", "metadata",
			]
			for field in updatable:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("variety.updated", "variety", variety_id, payload)
			return record
		except Exception as exc:
			_log.error("update_variety failed: %s", exc)
			raise

	async def delete_variety(self, variety_id: str) -> dict[str, Any]:
		"""Remove a variety from the registry."""
		try:
			if variety_id not in self._varieties:
				raise KeyError(f"variety_not_found:{variety_id}")
			record = self._varieties.pop(variety_id)
			self._emit("variety.deleted", "variety", variety_id, {"id": variety_id})
			return {"deleted": True, "id": variety_id}
		except Exception as exc:
			_log.error("delete_variety failed: %s", exc)
			raise

	async def search_varieties_by_region(self, region: str, crop_type: str | None = None) -> list[dict[str, Any]]:
		"""Return varieties suitable for a region based on climate metadata."""
		items = list(self._varieties.values())
		if crop_type:
			items = [v for v in items if v.get("crop_type") == crop_type]
		# Filter varieties that have region tagged in metadata
		return [v for v in items if region.lower() in str(v.get("metadata", {})).lower()]

	# ------------------------------------------------------------------ planting calendar

	async def list_planting_calendars(self, region: str | None = None, crop_type: str | None = None) -> list[dict[str, Any]]:
		"""List planting calendars with optional filters."""
		items = list(self._calendars.values())
		if region:
			items = [c for c in items if c.get("region") == region]
		if crop_type:
			items = [c for c in items if c.get("crop_type") == crop_type]
		return items

	async def get_planting_calendar(self, calendar_id: str) -> dict[str, Any]:
		"""Fetch a planting calendar entry."""
		if calendar_id not in self._calendars:
			raise KeyError(f"calendar_not_found:{calendar_id}")
		return self._calendars[calendar_id]

	async def create_planting_calendar(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Create a planting calendar entry."""
		try:
			cal_id = _new_id("cal")
			ts = _now()
			record: dict[str, Any] = {
				"id": cal_id,
				"tenant_id": self.tenant_id,
				"crop_type": payload["crop_type"],
				"variety_id": payload.get("variety_id"),
				"region": payload["region"],
				"planting_window_start": payload["planting_window_start"],
				"planting_window_end": payload["planting_window_end"],
				"harvest_window_start": payload.get("harvest_window_start"),
				"harvest_window_end": payload.get("harvest_window_end"),
				"recommended_density_plants_ha": payload.get("recommended_density_plants_ha"),
				"input_requirements": dict(payload.get("input_requirements", {})),
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._calendars[cal_id] = record
			self._emit("calendar.created", "planting_calendar", cal_id, record)
			return record
		except Exception as exc:
			_log.error("create_planting_calendar failed: %s", exc)
			raise

	async def update_planting_calendar(self, calendar_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		"""Update a planting calendar entry."""
		try:
			if calendar_id not in self._calendars:
				raise KeyError(f"calendar_not_found:{calendar_id}")
			record = self._calendars[calendar_id]
			for field in ["planting_window_start", "planting_window_end",
						"harvest_window_start", "harvest_window_end",
						"recommended_density_plants_ha", "input_requirements", "notes"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("calendar.updated", "planting_calendar", calendar_id, payload)
			return record
		except Exception as exc:
			_log.error("update_planting_calendar failed: %s", exc)
			raise

	async def delete_planting_calendar(self, calendar_id: str) -> dict[str, Any]:
		"""Delete a planting calendar entry."""
		try:
			if calendar_id not in self._calendars:
				raise KeyError(f"calendar_not_found:{calendar_id}")
			self._calendars.pop(calendar_id)
			self._emit("calendar.deleted", "planting_calendar", calendar_id, {"id": calendar_id})
			return {"deleted": True, "id": calendar_id}
		except Exception as exc:
			_log.error("delete_planting_calendar failed: %s", exc)
			raise

	async def recommend_planting_window(self, crop_type: str, region: str) -> dict[str, Any]:
		"""Return recommended planting window for a crop/region combination."""
		matches = [
			c for c in self._calendars.values()
			if c.get("crop_type") == crop_type and c.get("region") == region
		]
		if not matches:
			return {"crop_type": crop_type, "region": region, "recommendation": "no_data"}
		best = matches[0]
		return {
			"crop_type": crop_type,
			"region": region,
			"planting_window_start": best["planting_window_start"],
			"planting_window_end": best["planting_window_end"],
			"harvest_window_start": best.get("harvest_window_start"),
			"harvest_window_end": best.get("harvest_window_end"),
			"source_calendar_id": best["id"],
		}

	# ------------------------------------------------------------------ crops

	async def list_crops(self, farm_parcel_id: str | None = None, season: str | None = None,
						status: str | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		"""List crop records with optional filters."""
		items = list(self._crops.values())
		if farm_parcel_id:
			items = [c for c in items if c.get("farm_parcel_id") == farm_parcel_id]
		if season:
			items = [c for c in items if c.get("season") == season]
		if status:
			items = [c for c in items if c.get("status") == status]
		return items[offset: offset + limit]

	async def get_crop(self, crop_id: str) -> dict[str, Any]:
		"""Fetch a crop record."""
		if crop_id not in self._crops:
			raise KeyError(f"crop_not_found:{crop_id}")
		return self._crops[crop_id]

	async def create_crop(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Create a crop record for a planting event."""
		try:
			crop_id = _new_id("crp")
			ts = _now()
			record: dict[str, Any] = {
				"id": crop_id,
				"tenant_id": self.tenant_id,
				"farm_parcel_id": payload["farm_parcel_id"],
				"crop_type": payload["crop_type"],
				"variety_id": payload.get("variety_id"),
				"season": payload["season"],
				"planting_date": payload["planting_date"],
				"expected_harvest_date": payload.get("expected_harvest_date"),
				"actual_harvest_date": None,
				"area_ha": float(payload["area_ha"]),
				"status": payload.get("status", "planned"),
				"target_yield_kg": payload.get("target_yield_kg"),
				"actual_yield_kg": None,
				"seed_lot_reference": payload.get("seed_lot_reference"),
				"notes": payload.get("notes"),
				"metadata": dict(payload.get("metadata", {})),
				"created_at": ts,
				"updated_at": ts,
			}
			self._crops[crop_id] = record
			self._emit("crop.created", "crop", crop_id, record)
			return record
		except Exception as exc:
			_log.error("create_crop failed: %s", exc)
			raise

	async def update_crop(self, crop_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		"""Update crop status, harvest dates, or yield."""
		try:
			if crop_id not in self._crops:
				raise KeyError(f"crop_not_found:{crop_id}")
			record = self._crops[crop_id]
			for field in ["status", "expected_harvest_date", "actual_harvest_date",
						"actual_yield_kg", "notes", "metadata"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("crop.updated", "crop", crop_id, payload)
			return record
		except Exception as exc:
			_log.error("update_crop failed: %s", exc)
			raise

	async def delete_crop(self, crop_id: str) -> dict[str, Any]:
		"""Delete a crop record."""
		try:
			if crop_id not in self._crops:
				raise KeyError(f"crop_not_found:{crop_id}")
			self._crops.pop(crop_id)
			self._emit("crop.deleted", "crop", crop_id, {"id": crop_id})
			return {"deleted": True, "id": crop_id}
		except Exception as exc:
			_log.error("delete_crop failed: %s", exc)
			raise

	async def get_crop_performance_summary(self, farm_parcel_id: str, season: str) -> dict[str, Any]:
		"""Summarise yield performance vs targets for a parcel/season."""
		crops = await self.list_crops(farm_parcel_id=farm_parcel_id, season=season)
		total_target = sum(c.get("target_yield_kg") or 0 for c in crops)
		total_actual = sum(c.get("actual_yield_kg") or 0 for c in crops)
		achievement_pct = (total_actual / total_target * 100) if total_target > 0 else None
		return {
			"farm_parcel_id": farm_parcel_id,
			"season": season,
			"crop_count": len(crops),
			"total_area_ha": sum(c.get("area_ha", 0) for c in crops),
			"total_target_yield_kg": total_target,
			"total_actual_yield_kg": total_actual,
			"achievement_pct": round(achievement_pct, 2) if achievement_pct is not None else None,
		}

	# ------------------------------------------------------------------ phenology

	async def list_phenology(self, crop_id: str, limit: int = 100) -> list[dict[str, Any]]:
		"""List phenology observations for a crop."""
		items = [p for p in self._phenology.values() if p.get("crop_id") == crop_id]
		return sorted(items, key=lambda x: x.get("observed_at", ""))[-limit:]

	async def record_phenology(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Record a phenology observation."""
		try:
			obs_id = _new_id("phn")
			ts = _now()
			record: dict[str, Any] = {
				"id": obs_id,
				"tenant_id": self.tenant_id,
				"crop_id": payload["crop_id"],
				"observed_at": payload.get("observed_at", ts),
				"growth_stage": payload["growth_stage"],
				"observer_id": payload.get("observer_id"),
				"notes": payload.get("notes"),
				"images": list(payload.get("images", [])),
				"measurements": dict(payload.get("measurements", {})),
				"created_at": ts,
			}
			self._phenology[obs_id] = record
			# Update crop growth stage
			if payload["crop_id"] in self._crops:
				self._crops[payload["crop_id"]]["current_growth_stage"] = payload["growth_stage"]
				self._crops[payload["crop_id"]]["updated_at"] = ts
			self._emit("phenology.recorded", "phenology", obs_id, record)
			return record
		except Exception as exc:
			_log.error("record_phenology failed: %s", exc)
			raise

	async def get_current_growth_stage(self, crop_id: str) -> dict[str, Any]:
		"""Return the latest observed growth stage for a crop."""
		observations = await self.list_phenology(crop_id)
		if not observations:
			return {"crop_id": crop_id, "growth_stage": None, "observed_at": None}
		latest = observations[-1]
		return {
			"crop_id": crop_id,
			"growth_stage": latest.get("growth_stage"),
			"observed_at": latest.get("observed_at"),
			"observation_id": latest.get("id"),
		}

	# ------------------------------------------------------------------ rotation plans

	async def list_rotation_plans(self, farm_parcel_id: str | None = None) -> list[dict[str, Any]]:
		"""List crop rotation plans."""
		items = list(self._rotations.values())
		if farm_parcel_id:
			items = [r for r in items if r.get("farm_parcel_id") == farm_parcel_id]
		return items

	async def get_rotation_plan(self, plan_id: str) -> dict[str, Any]:
		"""Fetch a rotation plan."""
		if plan_id not in self._rotations:
			raise KeyError(f"rotation_plan_not_found:{plan_id}")
		return self._rotations[plan_id]

	async def create_rotation_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Create a crop rotation plan for a parcel."""
		try:
			plan_id = _new_id("rot")
			ts = _now()
			record: dict[str, Any] = {
				"id": plan_id,
				"tenant_id": self.tenant_id,
				"farm_parcel_id": payload["farm_parcel_id"],
				"strategy": payload["strategy"],
				"start_season": payload["start_season"],
				"crop_sequence": list(payload["crop_sequence"]),
				"rationale": payload.get("rationale"),
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._rotations[plan_id] = record
			self._emit("rotation_plan.created", "rotation_plan", plan_id, record)
			return record
		except Exception as exc:
			_log.error("create_rotation_plan failed: %s", exc)
			raise

	async def update_rotation_plan(self, plan_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		"""Update a rotation plan."""
		try:
			if plan_id not in self._rotations:
				raise KeyError(f"rotation_plan_not_found:{plan_id}")
			record = self._rotations[plan_id]
			for field in ["strategy", "crop_sequence", "rationale", "notes"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("rotation_plan.updated", "rotation_plan", plan_id, payload)
			return record
		except Exception as exc:
			_log.error("update_rotation_plan failed: %s", exc)
			raise

	async def delete_rotation_plan(self, plan_id: str) -> dict[str, Any]:
		"""Delete a rotation plan."""
		try:
			if plan_id not in self._rotations:
				raise KeyError(f"rotation_plan_not_found:{plan_id}")
			self._rotations.pop(plan_id)
			self._emit("rotation_plan.deleted", "rotation_plan", plan_id, {"id": plan_id})
			return {"deleted": True, "id": plan_id}
		except Exception as exc:
			_log.error("delete_rotation_plan failed: %s", exc)
			raise

	async def suggest_next_crop(self, farm_parcel_id: str, current_crop: str) -> dict[str, Any]:
		"""Suggest next crop based on rotation plan."""
		plans = await self.list_rotation_plans(farm_parcel_id=farm_parcel_id)
		if not plans:
			# Generic agronomic rules
			legumes = {"maize", "sorghum", "wheat", "barley"}
			if current_crop.lower() in legumes:
				return {"farm_parcel_id": farm_parcel_id, "suggested_crop": "legume", "basis": "nitrogen_fix"}
			return {"farm_parcel_id": farm_parcel_id, "suggested_crop": "legume", "basis": "default_rule"}
		plan = plans[0]
		seq = plan.get("crop_sequence", [])
		if current_crop in seq:
			idx = seq.index(current_crop)
			next_crop = seq[(idx + 1) % len(seq)]
			return {"farm_parcel_id": farm_parcel_id, "suggested_crop": next_crop, "basis": "rotation_plan", "plan_id": plan["id"]}
		return {"farm_parcel_id": farm_parcel_id, "suggested_crop": seq[0] if seq else None, "basis": "rotation_plan_start"}

	# ------------------------------------------------------------------ yield records

	async def list_yield_records(self, crop_id: str | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		"""List yield records."""
		items = list(self._yields.values())
		if crop_id:
			items = [y for y in items if y.get("crop_id") == crop_id]
		return items[offset: offset + limit]

	async def get_yield_record(self, yield_id: str) -> dict[str, Any]:
		"""Fetch a yield record."""
		if yield_id not in self._yields:
			raise KeyError(f"yield_record_not_found:{yield_id}")
		return self._yields[yield_id]

	async def create_yield_record(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Record a crop yield after harvest."""
		try:
			yield_id = _new_id("yld")
			ts = _now()
			record: dict[str, Any] = {
				"id": yield_id,
				"tenant_id": self.tenant_id,
				"crop_id": payload["crop_id"],
				"harvest_date": payload["harvest_date"],
				"gross_yield_kg": float(payload["gross_yield_kg"]),
				"net_yield_kg": float(payload["net_yield_kg"]) if payload.get("net_yield_kg") else None,
				"moisture_pct": payload.get("moisture_pct"),
				"grade": payload.get("grade"),
				"storage_location": payload.get("storage_location"),
				"notes": payload.get("notes"),
				"created_at": ts,
			}
			self._yields[yield_id] = record
			# Update crop actual yield
			if payload["crop_id"] in self._crops:
				self._crops[payload["crop_id"]]["actual_yield_kg"] = record["gross_yield_kg"]
				self._crops[payload["crop_id"]]["actual_harvest_date"] = record["harvest_date"]
				self._crops[payload["crop_id"]]["status"] = "harvested"
				self._crops[payload["crop_id"]]["updated_at"] = ts
			self._emit("yield.recorded", "yield_record", yield_id, record)
			return record
		except Exception as exc:
			_log.error("create_yield_record failed: %s", exc)
			raise

	async def delete_yield_record(self, yield_id: str) -> dict[str, Any]:
		"""Delete a yield record."""
		try:
			if yield_id not in self._yields:
				raise KeyError(f"yield_record_not_found:{yield_id}")
			self._yields.pop(yield_id)
			self._emit("yield.deleted", "yield_record", yield_id, {"id": yield_id})
			return {"deleted": True, "id": yield_id}
		except Exception as exc:
			_log.error("delete_yield_record failed: %s", exc)
			raise

	async def calculate_yield_statistics(self, crop_type: str, seasons: list[str] | None = None) -> dict[str, Any]:
		"""Compute yield statistics across crops of a given type."""
		crops = [c for c in self._crops.values() if c.get("crop_type") == crop_type]
		if seasons:
			crops = [c for c in crops if c.get("season") in seasons]
		yields = []
		for c in crops:
			crop_yields = [y for y in self._yields.values() if y.get("crop_id") == c["id"]]
			for y in crop_yields:
				area = c.get("area_ha", 1)
				if area > 0:
					yields.append(y["gross_yield_kg"] / area)
		if not yields:
			return {"crop_type": crop_type, "sample_size": 0, "mean_kg_ha": None}
		mean = sum(yields) / len(yields)
		return {
			"crop_type": crop_type,
			"seasons": seasons,
			"sample_size": len(yields),
			"mean_kg_ha": round(mean, 2),
			"min_kg_ha": round(min(yields), 2),
			"max_kg_ha": round(max(yields), 2),
		}

	async def get_seasonal_summary(self, season: str) -> dict[str, Any]:
		"""Return aggregate summary for all crops in a season."""
		crops = [c for c in self._crops.values() if c.get("season") == season]
		harvested = [c for c in crops if c.get("status") == "harvested"]
		total_area = sum(c.get("area_ha", 0) for c in crops)
		total_yield = sum(c.get("actual_yield_kg") or 0 for c in harvested)
		return {
			"season": season,
			"total_crops": len(crops),
			"harvested_crops": len(harvested),
			"total_area_ha": round(total_area, 3),
			"total_yield_kg": round(total_yield, 3),
			"avg_yield_kg_ha": round(total_yield / total_area, 2) if total_area > 0 else None,
		}
