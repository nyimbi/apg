"""Weather & Climate Analytics service — agr_wth."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)
_CAPABILITY_ID = "agr_wth"


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_id(prefix: str = "") -> str:
	suffix = uuid4().hex[:12]
	return f"{prefix}-{suffix}" if prefix else suffix


_OPS = {
	"gt": lambda v, t: v > t,
	"lt": lambda v, t: v < t,
	"gte": lambda v, t: v >= t,
	"lte": lambda v, t: v <= t,
}


class WeatherClimateService:
	"""Async service for weather & climate: forecast integration, alert thresholds,
	historical patterns, and climate risk assessment."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		if not tenant_id:
			raise ValueError("tenant_id required")
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self._forecasts = WriteThruDict('forecasts', tenant_id, _store)
		self._thresholds = WriteThruDict('thresholds', tenant_id, _store)
		self._alerts = WriteThruDict('alerts', tenant_id, _store)
		self._history = WriteThruDict('history', tenant_id, _store)
		self._risk_assessments = WriteThruDict('risk_assessments', tenant_id, _store)
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
				"forecasts": len(self._forecasts),
				"thresholds": len(self._thresholds),
				"alerts": len(self._alerts),
				"historical_records": len(self._history),
				"risk_assessments": len(self._risk_assessments),
			},
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": _CAPABILITY_ID,
			"name": "Weather & Climate Analytics",
			"domain": "agriculture",
			"version": "1.0.0",
			"description": "Forecast integration, alert thresholds, historical patterns, climate risk assessment.",
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		return self._audit[-limit:]

	# ------------------------------------------------------------------ forecasts

	async def list_forecasts(self, region: str | None = None, source: str | None = None,
							limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		items = list(self._forecasts.values())
		if region:
			items = [f for f in items if f.get("region") == region]
		if source:
			items = [f for f in items if f.get("source") == source]
		items = sorted(items, key=lambda x: x.get("valid_from", ""), reverse=True)
		return items[offset: offset + limit]

	async def get_forecast(self, forecast_id: str) -> dict[str, Any]:
		if forecast_id not in self._forecasts:
			raise KeyError(f"forecast_not_found:{forecast_id}")
		return self._forecasts[forecast_id]

	async def create_forecast(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Ingest a weather forecast from an external provider."""
		try:
			fid = _new_id("fct")
			ts = _now()
			record: dict[str, Any] = {
				"id": fid,
				"tenant_id": self.tenant_id,
				"region": payload["region"],
				"source": payload["source"],
				"forecast_date": payload["forecast_date"],
				"valid_from": payload["valid_from"],
				"valid_to": payload["valid_to"],
				"temperature_min_c": payload.get("temperature_min_c"),
				"temperature_max_c": payload.get("temperature_max_c"),
				"rainfall_mm": payload.get("rainfall_mm"),
				"humidity_pct": payload.get("humidity_pct"),
				"wind_speed_kmh": payload.get("wind_speed_kmh"),
				"wind_direction": payload.get("wind_direction"),
				"conditions": payload.get("conditions"),
				"metadata": dict(payload.get("metadata", {})),
				"created_at": ts,
			}
			self._forecasts[fid] = record
			# Evaluate thresholds against the new forecast
			triggered = await self._evaluate_thresholds(record)
			self._emit("forecast.created", "forecast", fid, {"forecast_id": fid, "alerts_triggered": len(triggered)})
			return {**record, "alerts_triggered": len(triggered)}
		except Exception as exc:
			_log.error("create_forecast failed: %s", exc)
			raise

	async def delete_forecast(self, forecast_id: str) -> dict[str, Any]:
		try:
			if forecast_id not in self._forecasts:
				raise KeyError(f"forecast_not_found:{forecast_id}")
			self._forecasts.pop(forecast_id)
			self._emit("forecast.deleted", "forecast", forecast_id, {"id": forecast_id})
			return {"deleted": True, "id": forecast_id}
		except Exception as exc:
			_log.error("delete_forecast failed: %s", exc)
			raise

	async def get_latest_forecast(self, region: str) -> dict[str, Any] | None:
		"""Return the most recent forecast for a region."""
		items = [f for f in self._forecasts.values() if f.get("region") == region]
		if not items:
			return None
		return sorted(items, key=lambda x: x.get("valid_from", ""), reverse=True)[0]

	# ------------------------------------------------------------------ alert thresholds

	async def list_thresholds(self, region: str | None = None, active: bool | None = None) -> list[dict[str, Any]]:
		items = list(self._thresholds.values())
		if region:
			items = [t for t in items if t.get("region") == region]
		if active is not None:
			items = [t for t in items if t.get("active") == active]
		return items

	async def get_threshold(self, threshold_id: str) -> dict[str, Any]:
		if threshold_id not in self._thresholds:
			raise KeyError(f"threshold_not_found:{threshold_id}")
		return self._thresholds[threshold_id]

	async def create_threshold(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			tid = _new_id("thr")
			ts = _now()
			record: dict[str, Any] = {
				"id": tid,
				"tenant_id": self.tenant_id,
				"region": payload["region"],
				"parameter": payload["parameter"],
				"operator": payload["operator"],
				"threshold_value": float(payload["threshold_value"]),
				"severity": payload["severity"],
				"description": payload.get("description"),
				"active": payload.get("active", True),
				"created_at": ts,
				"updated_at": ts,
			}
			self._thresholds[tid] = record
			self._emit("threshold.created", "alert_threshold", tid, record)
			return record
		except Exception as exc:
			_log.error("create_threshold failed: %s", exc)
			raise

	async def update_threshold(self, threshold_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if threshold_id not in self._thresholds:
				raise KeyError(f"threshold_not_found:{threshold_id}")
			record = self._thresholds[threshold_id]
			for field in ["threshold_value", "severity", "description", "active"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("threshold.updated", "alert_threshold", threshold_id, payload)
			return record
		except Exception as exc:
			_log.error("update_threshold failed: %s", exc)
			raise

	async def delete_threshold(self, threshold_id: str) -> dict[str, Any]:
		try:
			if threshold_id not in self._thresholds:
				raise KeyError(f"threshold_not_found:{threshold_id}")
			self._thresholds.pop(threshold_id)
			self._emit("threshold.deleted", "alert_threshold", threshold_id, {"id": threshold_id})
			return {"deleted": True, "id": threshold_id}
		except Exception as exc:
			_log.error("delete_threshold failed: %s", exc)
			raise

	async def _evaluate_thresholds(self, forecast: dict[str, Any]) -> list[dict[str, Any]]:
		"""Evaluate all active thresholds against a forecast and create alerts."""
		region = forecast.get("region", "")
		thresholds = [t for t in self._thresholds.values()
					if t.get("active") and t.get("region") == region]
		triggered = []
		for t in thresholds:
			param = t["parameter"]
			value = forecast.get(param)
			if value is None:
				continue
			op = _OPS.get(t["operator"])
			if op and op(float(value), t["threshold_value"]):
				alert_id = _new_id("alt")
				alert: dict[str, Any] = {
					"id": alert_id,
					"tenant_id": self.tenant_id,
					"region": region,
					"threshold_id": t["id"],
					"triggered_value": float(value),
					"severity": t["severity"],
					"message": f"{param} {t['operator']} {t['threshold_value']} (actual: {value})",
					"forecast_id": forecast.get("id"),
					"issued_at": _now(),
					"acknowledged": False,
				}
				self._alerts[alert_id] = alert
				triggered.append(alert)
		return triggered

	# ------------------------------------------------------------------ alerts

	async def list_alerts(self, region: str | None = None, acknowledged: bool | None = None,
						severity: str | None = None) -> list[dict[str, Any]]:
		items = list(self._alerts.values())
		if region:
			items = [a for a in items if a.get("region") == region]
		if acknowledged is not None:
			items = [a for a in items if a.get("acknowledged") == acknowledged]
		if severity:
			items = [a for a in items if a.get("severity") == severity]
		return sorted(items, key=lambda x: x.get("issued_at", ""), reverse=True)

	async def acknowledge_alert(self, alert_id: str) -> dict[str, Any]:
		try:
			if alert_id not in self._alerts:
				raise KeyError(f"alert_not_found:{alert_id}")
			self._alerts[alert_id]["acknowledged"] = True
			self._emit("alert.acknowledged", "weather_alert", alert_id, {"id": alert_id})
			return self._alerts[alert_id]
		except Exception as exc:
			_log.error("acknowledge_alert failed: %s", exc)
			raise

	# ------------------------------------------------------------------ historical patterns

	async def list_historical_patterns(self, region: str | None = None, year: int | None = None,
									month: int | None = None) -> list[dict[str, Any]]:
		items = list(self._history.values())
		if region:
			items = [h for h in items if h.get("region") == region]
		if year:
			items = [h for h in items if h.get("year") == year]
		if month:
			items = [h for h in items if h.get("month") == month]
		return items

	async def create_historical_pattern(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			hid = _new_id("hst")
			ts = _now()
			record: dict[str, Any] = {
				"id": hid,
				"tenant_id": self.tenant_id,
				"region": payload["region"],
				"year": int(payload["year"]),
				"month": int(payload["month"]),
				"avg_rainfall_mm": payload.get("avg_rainfall_mm"),
				"avg_temp_c": payload.get("avg_temp_c"),
				"min_temp_c": payload.get("min_temp_c"),
				"max_temp_c": payload.get("max_temp_c"),
				"drought_days": payload.get("drought_days"),
				"frost_days": payload.get("frost_days"),
				"source": payload.get("source"),
				"created_at": ts,
			}
			self._history[hid] = record
			self._emit("history.created", "historical_pattern", hid, record)
			return record
		except Exception as exc:
			_log.error("create_historical_pattern failed: %s", exc)
			raise

	async def delete_historical_pattern(self, hid: str) -> dict[str, Any]:
		try:
			if hid not in self._history:
				raise KeyError(f"historical_pattern_not_found:{hid}")
			self._history.pop(hid)
			self._emit("history.deleted", "historical_pattern", hid, {"id": hid})
			return {"deleted": True, "id": hid}
		except Exception as exc:
			_log.error("delete_historical_pattern failed: %s", exc)
			raise

	async def compute_monthly_normals(self, region: str, month: int) -> dict[str, Any]:
		"""Compute 30-year normals for a region/month from historical data."""
		records = [h for h in self._history.values()
				if h.get("region") == region and h.get("month") == month]
		if not records:
			return {"region": region, "month": month, "sample_size": 0}
		def mean(vals: list[float]) -> float | None:
			v = [x for x in vals if x is not None]
			return round(sum(v) / len(v), 2) if v else None
		return {
			"region": region,
			"month": month,
			"sample_size": len(records),
			"normal_rainfall_mm": mean([r.get("avg_rainfall_mm") for r in records]),
			"normal_temp_c": mean([r.get("avg_temp_c") for r in records]),
			"normal_min_temp_c": mean([r.get("min_temp_c") for r in records]),
			"normal_max_temp_c": mean([r.get("max_temp_c") for r in records]),
		}

	# ------------------------------------------------------------------ risk assessment

	async def list_risk_assessments(self, region: str | None = None, crop_type: str | None = None) -> list[dict[str, Any]]:
		items = list(self._risk_assessments.values())
		if region:
			items = [r for r in items if r.get("region") == region]
		if crop_type:
			items = [r for r in items if r.get("crop_type") == crop_type]
		return items

	async def assess_climate_risk(self, region: str, crop_type: str, season: str) -> dict[str, Any]:
		"""Compute climate risk scores using historical data and recent forecasts."""
		try:
			history = [h for h in self._history.values() if h.get("region") == region]
			forecasts = [f for f in self._forecasts.values() if f.get("region") == region]

			# Drought risk: fraction of months with below-average rainfall
			drought_months = sum(1 for h in history if (h.get("drought_days") or 0) > 5)
			drought_score = min(1.0, drought_months / max(len(history), 1))

			# Flood risk: forecasts with high rainfall
			flood_events = sum(1 for f in forecasts if (f.get("rainfall_mm") or 0) > 100)
			flood_score = min(1.0, flood_events / max(len(forecasts), 1))

			# Frost risk: from historical min temps
			frost_months = sum(1 for h in history if (h.get("frost_days") or 0) > 0)
			frost_score = min(1.0, frost_months / max(len(history), 1))

			# Heat stress: high max temp forecasts
			heat_events = sum(1 for f in forecasts if (f.get("temperature_max_c") or 0) > 38)
			heat_score = min(1.0, heat_events / max(len(forecasts), 1))

			overall = round((drought_score + flood_score + frost_score + heat_score) / 4, 3)

			risk_level = "negligible"
			if overall > 0.7:
				risk_level = "extreme"
			elif overall > 0.5:
				risk_level = "high"
			elif overall > 0.3:
				risk_level = "moderate"
			elif overall > 0.1:
				risk_level = "low"

			recs = []
			if drought_score > 0.3:
				recs.append("Consider drought-tolerant varieties and water-conserving practices")
			if flood_score > 0.3:
				recs.append("Ensure field drainage infrastructure is adequate")
			if frost_score > 0.2:
				recs.append("Avoid frost-sensitive crops in peak frost months")
			if heat_score > 0.3:
				recs.append("Schedule critical growth stages outside peak heat periods")

			assessment_id = _new_id("rsk")
			record: dict[str, Any] = {
				"id": assessment_id,
				"tenant_id": self.tenant_id,
				"region": region,
				"crop_type": crop_type,
				"season": season,
				"risk_level": risk_level,
				"drought_risk_score": round(drought_score, 3),
				"flood_risk_score": round(flood_score, 3),
				"frost_risk_score": round(frost_score, 3),
				"heat_stress_risk_score": round(heat_score, 3),
				"overall_score": overall,
				"recommendations": recs,
				"assessed_at": _now(),
			}
			self._risk_assessments[assessment_id] = record
			self._emit("risk.assessed", "climate_risk", assessment_id, record)
			return record
		except Exception as exc:
			_log.error("assess_climate_risk failed: %s", exc)
			raise

	async def get_risk_assessment(self, assessment_id: str) -> dict[str, Any]:
		if assessment_id not in self._risk_assessments:
			raise KeyError(f"risk_assessment_not_found:{assessment_id}")
		return self._risk_assessments[assessment_id]

	async def get_seasonal_climate_summary(self, region: str, season: str) -> dict[str, Any]:
		"""Aggregate forecast data for a region/season."""
		forecasts = [f for f in self._forecasts.values() if f.get("region") == region]
		if not forecasts:
			return {"region": region, "season": season, "forecast_count": 0}
		total_rain = sum(f.get("rainfall_mm") or 0 for f in forecasts)
		temps_max = [f.get("temperature_max_c") for f in forecasts if f.get("temperature_max_c") is not None]
		temps_min = [f.get("temperature_min_c") for f in forecasts if f.get("temperature_min_c") is not None]
		return {
			"region": region,
			"season": season,
			"forecast_count": len(forecasts),
			"total_forecast_rainfall_mm": round(total_rain, 1),
			"avg_max_temp_c": round(sum(temps_max) / len(temps_max), 1) if temps_max else None,
			"avg_min_temp_c": round(sum(temps_min) / len(temps_min), 1) if temps_min else None,
			"active_alerts": len([a for a in self._alerts.values() if a.get("region") == region and not a.get("acknowledged")]),
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_forecasts', '_thresholds', '_alerts', '_history', '_risk_assessments', '_audit']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

