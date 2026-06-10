"""Revenue Management & Rates service — dynamic pricing, demand forecasting, rate parity, yield optimisation."""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)


def _uid() -> str:
	return uuid4().hex[:12]


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# Base rate multipliers by demand level
_DEMAND_MULTIPLIERS = {
	"low": 0.85,
	"medium": 1.0,
	"high": 1.20,
	"peak": 1.45,
	"super_peak": 1.70,
}


class RMSService:
	"""Revenue Management & Rates service."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.rate_plans: dict[str, dict[str, Any]] = {}
		self.demand_forecasts: dict[str, dict[str, Any]] = {}
		self.competitor_rates: dict[str, dict[str, Any]] = {}
		self.parity_alerts: dict[str, dict[str, Any]] = {}
		self.yield_reports: dict[str, dict[str, Any]] = {}
		self.price_overrides: dict[str, dict[str, Any]] = {}
		self.seasonal_rules: dict[str, dict[str, Any]] = {}
		self.revenue_targets: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _emit(self, tenant_id: str, event_type: str, record_id: str, record_type: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": _uid(),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"details": details or {},
			"created_at": _now(),
		})

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "hos_rms",
			"status": "healthy",
			"rate_plans": len(self.rate_plans),
			"forecasts": len(self.demand_forecasts),
			"parity_alerts": sum(1 for a in self.parity_alerts.values() if a.get("status") == "open"),
			"checked_at": _now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": "hos_rms",
			"name": "Revenue Management & Rates",
			"domain": "hospitality",
			"version": "1.0.0",
			"description": "Dynamic pricing, demand forecasting, rate parity, yield optimisation, competitor rate monitoring",
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Rate Plans ────────────────────────────────────────────────────────────

	async def list_rate_plans(self, tenant_id: str | None = None, room_type: str | None = None, active_only: bool = False) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		plans = [deepcopy(p) for p in self.rate_plans.values() if p["tenant_id"] == tenant]
		if room_type:
			plans = [p for p in plans if p["room_type"] == room_type]
		if active_only:
			plans = [p for p in plans if p["is_active"]]
		return plans

	async def get_rate_plan(self, plan_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		plan = self.rate_plans.get(plan_id)
		if not plan or plan["tenant_id"] != tenant:
			raise KeyError(f"rate_plan_not_found:{plan_id}")
		return deepcopy(plan)

	async def create_rate_plan(self, code: str, name: str, base_rate: float, room_type: str,
	                            min_stay: int = 1, meal_plan: str = "room_only",
	                            cancellation_policy: str = "flexible", advance_purchase_days: int = 0,
	                            is_public: bool = True, description: str | None = None,
	                            tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		# Prevent duplicate rate plan codes
		for p in self.rate_plans.values():
			if p["tenant_id"] == tenant and p["code"] == code:
				raise ValueError(f"rate_plan_code_exists:{code}")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"code": code,
			"name": name,
			"description": description,
			"base_rate": base_rate,
			"room_type": room_type,
			"min_stay": min_stay,
			"meal_plan": meal_plan,
			"cancellation_policy": cancellation_policy,
			"advance_purchase_days": advance_purchase_days,
			"is_public": is_public,
			"is_active": True,
			"status": "active",
			"created_at": _now(),
		}
		self.rate_plans[record["id"]] = record
		self._emit(tenant, "rate_plan_created", record["id"], "rate_plan", {"code": code})
		return deepcopy(record)

	async def update_rate_plan(self, plan_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		plan = self.rate_plans.get(plan_id)
		if not plan or plan["tenant_id"] != tenant:
			raise KeyError(f"rate_plan_not_found:{plan_id}")
		allowed = {"name", "base_rate", "min_stay", "is_active", "description", "meal_plan", "cancellation_policy"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				plan[k] = v
		plan["updated_at"] = _now()
		self._emit(tenant, "rate_plan_updated", plan_id, "rate_plan")
		return deepcopy(plan)

	async def delete_rate_plan(self, plan_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		plan = self.rate_plans.get(plan_id)
		if not plan or plan["tenant_id"] != tenant:
			raise KeyError(f"rate_plan_not_found:{plan_id}")
		plan["is_active"] = False
		plan["status"] = "deactivated"
		self._emit(tenant, "rate_plan_deactivated", plan_id, "rate_plan")
		return {"deactivated": True, "plan_id": plan_id}

	async def get_effective_rate(self, plan_id: str, date: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Compute the effective rate for a given rate plan and date, applying overrides and seasonal rules."""
		tenant = self._tenant(tenant_id)
		plan = self.rate_plans.get(plan_id)
		if not plan or plan["tenant_id"] != tenant:
			raise KeyError(f"rate_plan_not_found:{plan_id}")
		base = plan["base_rate"]
		multiplier = 1.0
		override = None
		# Check overrides
		for ov in self.price_overrides.values():
			if ov["tenant_id"] == tenant and ov["rate_plan_id"] == plan_id and ov["date"] == date:
				override = ov
				break
		# Apply seasonal rules
		for rule in self.seasonal_rules.values():
			if rule["tenant_id"] == tenant and rule.get("date_from", "") <= date <= rule.get("date_to", "9999"):
				multiplier *= rule.get("multiplier", 1.0)
		effective = override["rate"] if override else base * multiplier
		return {
			"plan_id": plan_id,
			"date": date,
			"base_rate": base,
			"multiplier": multiplier,
			"override_applied": override is not None,
			"effective_rate": round(effective, 2),
			"computed_at": _now(),
		}

	# ── Demand Forecasting ────────────────────────────────────────────────────

	async def list_demand_forecasts(self, tenant_id: str | None = None, room_type: str | None = None, date_from: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(f) for f in self.demand_forecasts.values() if f["tenant_id"] == tenant]
		if room_type:
			items = [f for f in items if f["room_type"] == room_type]
		if date_from:
			items = [f for f in items if f["forecast_date"] >= date_from]
		return items

	async def get_demand_forecast(self, forecast_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		fc = self.demand_forecasts.get(forecast_id)
		if not fc or fc["tenant_id"] != tenant:
			raise KeyError(f"forecast_not_found:{forecast_id}")
		return deepcopy(fc)

	async def create_demand_forecast(self, forecast_date: str, room_type: str, predicted_demand: float,
	                                  confidence: float = 0.8, events: list[str] | None = None,
	                                  notes: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		"""Store a demand forecast and compute recommended rate."""
		tenant = self._tenant(tenant_id)
		# Derive demand tier
		if predicted_demand >= 0.90:
			tier = "super_peak"
		elif predicted_demand >= 0.75:
			tier = "peak"
		elif predicted_demand >= 0.60:
			tier = "high"
		elif predicted_demand >= 0.40:
			tier = "medium"
		else:
			tier = "low"
		# Find the base rate from the most recent rate plan for this room type
		plans = [p for p in self.rate_plans.values() if p["tenant_id"] == tenant and p["room_type"] == room_type and p["is_active"]]
		base = plans[0]["base_rate"] if plans else 10000.0
		recommended_rate = round(base * _DEMAND_MULTIPLIERS[tier], 2)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"forecast_date": forecast_date,
			"room_type": room_type,
			"predicted_demand": predicted_demand,
			"demand_tier": tier,
			"confidence": confidence,
			"recommended_rate": recommended_rate,
			"events": events or [],
			"notes": notes,
			"status": "active",
			"created_at": _now(),
		}
		self.demand_forecasts[record["id"]] = record
		self._emit(tenant, "demand_forecast_created", record["id"], "demand_forecast", {"date": forecast_date, "tier": tier})
		return deepcopy(record)

	async def update_demand_forecast(self, forecast_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		fc = self.demand_forecasts.get(forecast_id)
		if not fc or fc["tenant_id"] != tenant:
			raise KeyError(f"forecast_not_found:{forecast_id}")
		for k, v in updates.items():
			if v is not None:
				fc[k] = v
		self._emit(tenant, "demand_forecast_updated", forecast_id, "demand_forecast")
		return deepcopy(fc)

	async def delete_demand_forecast(self, forecast_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		fc = self.demand_forecasts.get(forecast_id)
		if not fc or fc["tenant_id"] != tenant:
			raise KeyError(f"forecast_not_found:{forecast_id}")
		fc["status"] = "archived"
		self._emit(tenant, "demand_forecast_archived", forecast_id, "demand_forecast")
		return {"archived": True, "forecast_id": forecast_id}

	# ── Competitor Rate Monitoring ────────────────────────────────────────────

	async def list_competitor_rates(self, tenant_id: str | None = None, date: str | None = None, room_type: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.competitor_rates.values() if r["tenant_id"] == tenant]
		if date:
			items = [r for r in items if r["date"] == date]
		if room_type:
			items = [r for r in items if r["room_type"] == room_type]
		return items

	async def get_competitor_rate(self, rate_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		rate = self.competitor_rates.get(rate_id)
		if not rate or rate["tenant_id"] != tenant:
			raise KeyError(f"competitor_rate_not_found:{rate_id}")
		return deepcopy(rate)

	async def create_competitor_rate(self, competitor_name: str, room_type: str, rate: float,
	                                  date: str, source: str = "manual", channel: str | None = None,
	                                  tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"competitor_name": competitor_name,
			"room_type": room_type,
			"rate": rate,
			"date": date,
			"source": source,
			"channel": channel,
			"created_at": _now(),
		}
		self.competitor_rates[record["id"]] = record
		self._emit(tenant, "competitor_rate_recorded", record["id"], "competitor_rate", {"competitor": competitor_name, "rate": rate})
		# Auto-check rate parity
		await self._check_parity(tenant, room_type, date, rate, competitor_name, channel or "direct")
		return deepcopy(record)

	async def update_competitor_rate(self, rate_id: str, updates: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		rate = self.competitor_rates.get(rate_id)
		if not rate or rate["tenant_id"] != tenant:
			raise KeyError(f"competitor_rate_not_found:{rate_id}")
		for k, v in updates.items():
			if v is not None:
				rate[k] = v
		return deepcopy(rate)

	async def delete_competitor_rate(self, rate_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		rate = self.competitor_rates.get(rate_id)
		if not rate or rate["tenant_id"] != tenant:
			raise KeyError(f"competitor_rate_not_found:{rate_id}")
		del self.competitor_rates[rate_id]
		return {"deleted": True, "rate_id": rate_id}

	async def _check_parity(self, tenant: str, room_type: str, date: str, competitor_rate: float, competitor: str, channel: str) -> None:
		"""Auto-create a parity alert if our rate deviates significantly from competitor."""
		our_plans = [p for p in self.rate_plans.values() if p["tenant_id"] == tenant and p["room_type"] == room_type and p["is_active"]]
		if not our_plans:
			return
		our_rate = our_plans[0]["base_rate"]
		variance_pct = ((our_rate - competitor_rate) / competitor_rate * 100) if competitor_rate else 0.0
		severity = "low"
		if abs(variance_pct) >= 20:
			severity = "high"
		elif abs(variance_pct) >= 10:
			severity = "medium"
		if abs(variance_pct) >= 5:
			alert: dict[str, Any] = {
				"id": _uid(),
				"tenant_id": tenant,
				"room_type": room_type,
				"date": date,
				"our_rate": our_rate,
				"competitor_rate": competitor_rate,
				"competitor_name": competitor,
				"channel": channel,
				"variance_pct": round(variance_pct, 2),
				"severity": severity,
				"status": "open",
				"created_at": _now(),
			}
			self.parity_alerts[alert["id"]] = alert
			self._emit(tenant, "rate_parity_alert_created", alert["id"], "parity_alert", {"severity": severity})

	async def list_parity_alerts(self, tenant_id: str | None = None, severity: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		alerts = [deepcopy(a) for a in self.parity_alerts.values() if a["tenant_id"] == tenant]
		if severity:
			alerts = [a for a in alerts if a["severity"] == severity]
		if status:
			alerts = [a for a in alerts if a["status"] == status]
		return alerts

	async def resolve_parity_alert(self, alert_id: str, resolution: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		alert = self.parity_alerts.get(alert_id)
		if not alert or alert["tenant_id"] != tenant:
			raise KeyError(f"parity_alert_not_found:{alert_id}")
		alert["status"] = "resolved"
		alert["resolution"] = resolution
		alert["resolved_at"] = _now()
		self._emit(tenant, "parity_alert_resolved", alert_id, "parity_alert")
		return deepcopy(alert)

	# ── Yield Optimisation ────────────────────────────────────────────────────

	async def run_yield_optimisation(self, date_from: str, date_to: str, room_type: str,
	                                  current_occupancy: float, target_occupancy: float = 0.85,
	                                  tenant_id: str | None = None) -> dict[str, Any]:
		"""Run yield optimisation for a date range and room type."""
		tenant = self._tenant(tenant_id)
		plans = [p for p in self.rate_plans.values() if p["tenant_id"] == tenant and p["room_type"] == room_type and p["is_active"]]
		base_rate = plans[0]["base_rate"] if plans else 10000.0
		# Simple yield formula: adjust rate inversely with occupancy gap
		gap = target_occupancy - current_occupancy
		if gap > 0.20:
			strategy = "discount_heavy"
			rate_change_pct = -15.0
		elif gap > 0.10:
			strategy = "discount_moderate"
			rate_change_pct = -8.0
		elif gap < -0.10:
			strategy = "premium_moderate"
			rate_change_pct = 12.0
		elif gap < -0.20:
			strategy = "premium_aggressive"
			rate_change_pct = 20.0
		else:
			strategy = "maintain"
			rate_change_pct = 0.0
		recommended_rate = round(base_rate * (1 + rate_change_pct / 100), 2)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"date_from": date_from,
			"date_to": date_to,
			"room_type": room_type,
			"current_occupancy": current_occupancy,
			"target_occupancy": target_occupancy,
			"base_rate": base_rate,
			"recommended_rate": recommended_rate,
			"rate_change_pct": rate_change_pct,
			"strategy": strategy,
			"status": "applied",
			"generated_at": _now(),
		}
		self.yield_reports[record["id"]] = record
		self._emit(tenant, "yield_optimisation_run", record["id"], "yield_report", {"strategy": strategy})
		return deepcopy(record)

	async def list_yield_reports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.yield_reports.values() if r["tenant_id"] == tenant]

	# ── Seasonal Rules ────────────────────────────────────────────────────────

	async def create_seasonal_rule(self, name: str, date_from: str, date_to: str, multiplier: float,
	                                room_type: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"name": name,
			"date_from": date_from,
			"date_to": date_to,
			"multiplier": multiplier,
			"room_type": room_type,
			"status": "active",
			"created_at": _now(),
		}
		self.seasonal_rules[record["id"]] = record
		self._emit(tenant, "seasonal_rule_created", record["id"], "seasonal_rule")
		return deepcopy(record)

	async def list_seasonal_rules(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.seasonal_rules.values() if r["tenant_id"] == tenant]

	async def delete_seasonal_rule(self, rule_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		rule = self.seasonal_rules.get(rule_id)
		if not rule or rule["tenant_id"] != tenant:
			raise KeyError(f"seasonal_rule_not_found:{rule_id}")
		del self.seasonal_rules[rule_id]
		return {"deleted": True, "rule_id": rule_id}

	# ── Price Overrides ───────────────────────────────────────────────────────

	async def set_price_override(self, rate_plan_id: str, date: str, rate: float, reason: str,
	                              tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		plan = self.rate_plans.get(rate_plan_id)
		if not plan or plan["tenant_id"] != tenant:
			raise KeyError(f"rate_plan_not_found:{rate_plan_id}")
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"rate_plan_id": rate_plan_id,
			"date": date,
			"rate": rate,
			"reason": reason,
			"status": "active",
			"created_at": _now(),
		}
		self.price_overrides[record["id"]] = record
		self._emit(tenant, "price_override_set", record["id"], "price_override", {"date": date, "rate": rate})
		return deepcopy(record)

	async def list_price_overrides(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.price_overrides.values() if r["tenant_id"] == tenant]

	# ── Revenue Targets ───────────────────────────────────────────────────────

	async def set_revenue_target(self, period: str, room_type: str, target_revpar: float,
	                              target_adr: float, target_occupancy: float, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": tenant,
			"period": period,
			"room_type": room_type,
			"target_revpar": target_revpar,
			"target_adr": target_adr,
			"target_occupancy": target_occupancy,
			"status": "active",
			"created_at": _now(),
		}
		self.revenue_targets[record["id"]] = record
		self._emit(tenant, "revenue_target_set", record["id"], "revenue_target")
		return deepcopy(record)

	async def list_revenue_targets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(t) for t in self.revenue_targets.values() if t["tenant_id"] == tenant]

	# ── Rate Parity Report ────────────────────────────────────────────────────

	async def rate_parity_report(self, date_from: str, date_to: str, room_type: str | None = None,
	                              tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		alerts = [a for a in self.parity_alerts.values() if a["tenant_id"] == tenant]
		if date_from:
			alerts = [a for a in alerts if a["date"] >= date_from]
		if date_to:
			alerts = [a for a in alerts if a["date"] <= date_to]
		if room_type:
			alerts = [a for a in alerts if a["room_type"] == room_type]
		by_severity: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
		for a in alerts:
			by_severity[a["severity"]] = by_severity.get(a["severity"], 0) + 1
		return {
			"tenant_id": tenant,
			"date_from": date_from,
			"date_to": date_to,
			"total_alerts": len(alerts),
			"by_severity": by_severity,
			"open_alerts": sum(1 for a in alerts if a["status"] == "open"),
			"avg_variance_pct": round(sum(a["variance_pct"] for a in alerts) / len(alerts), 2) if alerts else 0.0,
			"generated_at": _now(),
		}

	async def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		plans = [p for p in self.rate_plans.values() if p["tenant_id"] == tenant and p["is_active"]]
		open_alerts = [a for a in self.parity_alerts.values() if a["tenant_id"] == tenant and a["status"] == "open"]
		return {
			"tenant_id": tenant,
			"active_rate_plans": len(plans),
			"demand_forecasts": len([f for f in self.demand_forecasts.values() if f["tenant_id"] == tenant]),
			"competitor_rates_tracked": len([r for r in self.competitor_rates.values() if r["tenant_id"] == tenant]),
			"open_parity_alerts": len(open_alerts),
			"high_severity_alerts": sum(1 for a in open_alerts if a["severity"] == "high"),
			"yield_reports": len([r for r in self.yield_reports.values() if r["tenant_id"] == tenant]),
			"generated_at": _now(),
		}
