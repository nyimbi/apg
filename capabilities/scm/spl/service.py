"""Supply Planning async service (scm_spl)."""
from __future__ import annotations

import asyncio
import logging
import math
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

_log = logging.getLogger(__name__)

CAPABILITY_ID = "scm_spl"
FORECAST_METHODS = {"statistical", "ml", "manual", "exponential_smoothing", "moving_average"}
RULE_TYPES = {"min_max", "reorder_point", "periodic_review"}
RESOURCE_TYPES = {"warehouse", "production_line", "supplier"}


class SupplyPlanningService:
	"""Async service for MRP-II, safety stock optimisation, replenishment rules,
	capacity planning and supply/demand balancing."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.demand_forecasts: dict[str, dict[str, Any]] = {}
		self.mrp_runs: dict[str, dict[str, Any]] = {}
		self.safety_stocks: dict[str, dict[str, Any]] = {}
		self.replenishment_rules: dict[str, dict[str, Any]] = {}
		self.capacity_plans: dict[str, dict[str, Any]] = {}
		self.supply_demand_balances: dict[str, dict[str, Any]] = {}
		self.planned_orders: dict[str, dict[str, Any]] = {}
		self.supply_exceptions: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _tenant(self, tenant_id: str | None = None) -> str:
		t = tenant_id or self.tenant_id
		if not t:
			raise PermissionError("tenant_context_required")
		return t

	def _emit(self, tenant_id: str, event_type: str, record_id: str, record_type: str, status: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"status": status,
			"capability_id": CAPABILITY_ID,
			"emitted_at": self._now(),
		})

	# ── Health & describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"forecast_count": len(self.demand_forecasts),
			"active_rules": sum(1 for r in self.replenishment_rules.values() if r["status"] == "active"),
			"open_mrp_runs": sum(1 for r in self.mrp_runs.values() if r["status"] in {"running", "pending"}),
			"supply_exceptions": sum(1 for e in self.supply_exceptions.values() if e["status"] == "open"),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "scm",
			"version": "1.0.0",
			"description": "MRP-II, safety stock optimisation, replenishment rules, capacity planning, supply/demand balancing",
			"forecast_methods": sorted(FORECAST_METHODS),
			"rule_types": sorted(RULE_TYPES),
			"resource_types": sorted(RESOURCE_TYPES),
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Demand forecasting ────────────────────────────────────────────────────

	async def create_demand_forecast(
		self,
		sku: str,
		period: str,
		forecast_quantity: float,
		method: str = "statistical",
		confidence_pct: float = 80.0,
		warehouse_id: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a demand forecast for a SKU/period."""
		tenant = self._tenant(tenant_id)
		if method not in FORECAST_METHODS:
			raise ValueError(f"method must be one of {FORECAST_METHODS}")
		if not 0 < confidence_pct <= 100:
			raise ValueError("confidence_pct must be in (0, 100]")
		record: dict[str, Any] = {
			"id": self._id("fcast"),
			"type": "scm_spl_demand_forecast",
			"tenant_id": tenant,
			"sku": sku,
			"warehouse_id": warehouse_id,
			"period": period,
			"forecast_quantity": forecast_quantity,
			"actual_quantity": None,
			"confidence_pct": confidence_pct,
			"method": method,
			"status": "active",
			"created_at": self._now(),
		}
		self.demand_forecasts[record["id"]] = record
		self._emit(tenant, "demand_forecast_created", record["id"], "scm_spl_demand_forecast", "active")
		return deepcopy(record)

	async def update_forecast_actual(
		self,
		forecast_id: str,
		actual_quantity: float,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record actual demand against a forecast for accuracy tracking."""
		tenant = self._tenant(tenant_id)
		forecast = self.demand_forecasts.get(forecast_id)
		if not forecast or forecast["tenant_id"] != tenant:
			raise KeyError(f"forecast '{forecast_id}' not found")
		forecast["actual_quantity"] = actual_quantity
		forecast_qty = forecast["forecast_quantity"]
		if forecast_qty:
			mape = abs(forecast_qty - actual_quantity) / forecast_qty * 100
			forecast["accuracy_pct"] = round(max(0.0, 100.0 - mape), 2)
		forecast["status"] = "reconciled"
		self._emit(tenant, "forecast_reconciled", forecast_id, "scm_spl_demand_forecast", "reconciled")
		return deepcopy(forecast)

	async def list_demand_forecasts(
		self,
		sku: str | None = None,
		period: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List demand forecasts."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(f) for f in self.demand_forecasts.values() if f["tenant_id"] == tenant]
		if sku:
			items = [f for f in items if f["sku"] == sku]
		if period:
			items = [f for f in items if f["period"] == period]
		return items

	async def get_demand_forecast(self, forecast_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Fetch a single demand forecast."""
		tenant = self._tenant(tenant_id)
		f = self.demand_forecasts.get(forecast_id)
		if not f or f["tenant_id"] != tenant:
			raise KeyError(f"forecast '{forecast_id}' not found")
		return deepcopy(f)

	async def delete_demand_forecast(self, forecast_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Deactivate a demand forecast."""
		tenant = self._tenant(tenant_id)
		f = self.demand_forecasts.get(forecast_id)
		if not f or f["tenant_id"] != tenant:
			raise KeyError(f"forecast '{forecast_id}' not found")
		f["status"] = "inactive"
		self._emit(tenant, "forecast_deactivated", forecast_id, "scm_spl_demand_forecast", "inactive")
		return deepcopy(f)

	# ── MRP-II ────────────────────────────────────────────────────────────────

	async def run_mrp(
		self,
		run_name: str,
		horizon_weeks: int = 12,
		sku_filter: list[str] | None = None,
		warehouse_filter: list[str] | None = None,
		include_safety_stock: bool = True,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Execute an MRP-II planning run and generate planned orders."""
		tenant = self._tenant(tenant_id)
		if horizon_weeks < 1 or horizon_weeks > 104:
			raise ValueError("horizon_weeks must be between 1 and 104")
		# Gather relevant forecasts
		forecasts = [
			f for f in self.demand_forecasts.values()
			if f["tenant_id"] == tenant and f["status"] == "active"
		]
		if sku_filter:
			forecasts = [f for f in forecasts if f["sku"] in sku_filter]
		if warehouse_filter:
			forecasts = [f for f in forecasts if f.get("warehouse_id") in warehouse_filter]

		planned_orders_list: list[dict[str, Any]] = []
		for fc in forecasts:
			safety = 0.0
			if include_safety_stock:
				ss_key = f"{tenant}:{fc['sku']}:{fc.get('warehouse_id', 'any')}"
				ss_record = self.safety_stocks.get(ss_key) or {}
				safety = ss_record.get("effective_safety_stock", 0.0)
			required = fc["forecast_quantity"] + safety
			if required > 0:
				po = {
					"id": self._id("mrppo"),
					"type": "scm_spl_planned_order",
					"tenant_id": tenant,
					"sku": fc["sku"],
					"warehouse_id": fc.get("warehouse_id"),
					"period": fc["period"],
					"required_quantity": required,
					"safety_stock_included": safety,
					"source": "mrp_run",
					"status": "planned",
					"created_at": self._now(),
				}
				self.planned_orders[po["id"]] = po
				planned_orders_list.append(deepcopy(po))

		record: dict[str, Any] = {
			"id": self._id("mrp"),
			"type": "scm_spl_mrp_run",
			"tenant_id": tenant,
			"run_name": run_name,
			"horizon_weeks": horizon_weeks,
			"sku_filter": sku_filter or [],
			"warehouse_filter": warehouse_filter or [],
			"include_safety_stock": include_safety_stock,
			"planned_orders": planned_orders_list,
			"planned_order_count": len(planned_orders_list),
			"status": "completed",
			"started_at": self._now(),
			"completed_at": self._now(),
		}
		self.mrp_runs[record["id"]] = record
		self._emit(tenant, "mrp_run_completed", record["id"], "scm_spl_mrp_run", "completed")
		return deepcopy(record)

	async def list_mrp_runs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List MRP runs."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.mrp_runs.values() if r["tenant_id"] == tenant]

	async def get_mrp_run(self, run_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Fetch a single MRP run."""
		tenant = self._tenant(tenant_id)
		r = self.mrp_runs.get(run_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"mrp_run '{run_id}' not found")
		return deepcopy(r)

	# ── Safety stock ──────────────────────────────────────────────────────────

	async def calculate_safety_stock(
		self,
		sku: str,
		lead_time_days: int,
		target_service_level_pct: float = 95.0,
		demand_std_dev: float | None = None,
		manual_override: float | None = None,
		warehouse_id: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Calculate optimal safety stock using service-level Z-score method."""
		tenant = self._tenant(tenant_id)
		if lead_time_days < 1:
			raise ValueError("lead_time_days must be >= 1")
		# Z-score lookup (approx for common service levels)
		z_table = {70: 0.52, 75: 0.67, 80: 0.84, 85: 1.04, 90: 1.28, 95: 1.645, 97: 1.88, 98: 2.05, 99: 2.33}
		z = z_table.get(round(target_service_level_pct), 1.645)

		std_dev = demand_std_dev or 0.0
		# safety stock = Z * σ * √(lead_time_days)
		calculated = round(z * std_dev * math.sqrt(lead_time_days), 2) if std_dev else 0.0
		effective = manual_override if manual_override is not None else calculated

		key = f"{tenant}:{sku}:{warehouse_id or 'any'}"
		record: dict[str, Any] = {
			"id": key,
			"type": "scm_spl_safety_stock",
			"tenant_id": tenant,
			"sku": sku,
			"warehouse_id": warehouse_id,
			"target_service_level_pct": target_service_level_pct,
			"lead_time_days": lead_time_days,
			"demand_std_dev": std_dev,
			"calculated_safety_stock": calculated,
			"manual_override": manual_override,
			"effective_safety_stock": effective,
			"z_score": z,
			"status": "active",
			"calculated_at": self._now(),
		}
		self.safety_stocks[key] = record
		self._emit(tenant, "safety_stock_calculated", key, "scm_spl_safety_stock", "active")
		return deepcopy(record)

	async def list_safety_stocks(
		self,
		sku: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List safety stock records."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(s) for s in self.safety_stocks.values() if s["tenant_id"] == tenant]
		if sku:
			items = [s for s in items if s["sku"] == sku]
		return items

	# ── Replenishment rules ───────────────────────────────────────────────────

	async def create_replenishment_rule(
		self,
		sku: str,
		rule_type: str,
		reorder_point: float | None = None,
		order_quantity: float | None = None,
		min_stock: float | None = None,
		max_stock: float | None = None,
		review_cycle_days: int | None = None,
		warehouse_id: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Define a replenishment rule for a SKU."""
		tenant = self._tenant(tenant_id)
		if rule_type not in RULE_TYPES:
			raise ValueError(f"rule_type must be one of {RULE_TYPES}")
		record: dict[str, Any] = {
			"id": self._id("rrule"),
			"type": "scm_spl_replenishment_rule",
			"tenant_id": tenant,
			"sku": sku,
			"warehouse_id": warehouse_id,
			"rule_type": rule_type,
			"reorder_point": reorder_point,
			"order_quantity": order_quantity,
			"min_stock": min_stock,
			"max_stock": max_stock,
			"review_cycle_days": review_cycle_days,
			"status": "active",
			"created_at": self._now(),
		}
		self.replenishment_rules[record["id"]] = record
		self._emit(tenant, "replenishment_rule_created", record["id"], "scm_spl_replenishment_rule", "active")
		return deepcopy(record)

	async def list_replenishment_rules(
		self,
		sku: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List replenishment rules."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.replenishment_rules.values() if r["tenant_id"] == tenant]
		if sku:
			items = [r for r in items if r["sku"] == sku]
		return items

	async def get_replenishment_rule(self, rule_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Fetch a single replenishment rule."""
		tenant = self._tenant(tenant_id)
		r = self.replenishment_rules.get(rule_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"replenishment_rule '{rule_id}' not found")
		return deepcopy(r)

	async def update_replenishment_rule(
		self,
		rule_id: str,
		updates: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Update a replenishment rule."""
		tenant = self._tenant(tenant_id)
		r = self.replenishment_rules.get(rule_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"replenishment_rule '{rule_id}' not found")
		allowed = {"reorder_point", "order_quantity", "min_stock", "max_stock", "review_cycle_days", "status"}
		for k, v in updates.items():
			if k in allowed:
				r[k] = v
		self._emit(tenant, "replenishment_rule_updated", rule_id, "scm_spl_replenishment_rule", r["status"])
		return deepcopy(r)

	async def delete_replenishment_rule(self, rule_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Deactivate a replenishment rule."""
		tenant = self._tenant(tenant_id)
		r = self.replenishment_rules.get(rule_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"replenishment_rule '{rule_id}' not found")
		r["status"] = "inactive"
		self._emit(tenant, "replenishment_rule_deactivated", rule_id, "scm_spl_replenishment_rule", "inactive")
		return deepcopy(r)

	async def evaluate_replenishment_triggers(
		self,
		current_stocks: dict[str, float],
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""Evaluate which SKUs need replenishment based on current stock levels."""
		tenant = self._tenant(tenant_id)
		triggered: list[dict[str, Any]] = []
		for rule in self.replenishment_rules.values():
			if rule["tenant_id"] != tenant or rule["status"] != "active":
				continue
			current = current_stocks.get(rule["sku"], 0.0)
			needs_replenishment = False
			if rule["rule_type"] == "reorder_point" and rule.get("reorder_point") is not None:
				needs_replenishment = current <= rule["reorder_point"]
			elif rule["rule_type"] == "min_max" and rule.get("min_stock") is not None:
				needs_replenishment = current <= rule["min_stock"]
			if needs_replenishment:
				suggested_qty = rule.get("order_quantity") or (
					rule.get("max_stock", 0.0) - current if rule.get("max_stock") else 0.0
				)
				triggered.append({
					"rule_id": rule["id"],
					"sku": rule["sku"],
					"current_stock": current,
					"trigger_level": rule.get("reorder_point") or rule.get("min_stock"),
					"suggested_order_quantity": suggested_qty,
					"evaluated_at": self._now(),
				})
		return triggered

	# ── Capacity planning ─────────────────────────────────────────────────────

	async def create_capacity_plan(
		self,
		resource_id: str,
		resource_type: str,
		period: str,
		available_capacity: float,
		unit: str = "units",
		notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Define available capacity for a resource in a period."""
		tenant = self._tenant(tenant_id)
		if resource_type not in RESOURCE_TYPES:
			raise ValueError(f"resource_type must be one of {RESOURCE_TYPES}")
		# Check planned demand from forecasts for this period
		planned_demand = sum(
			f["forecast_quantity"]
			for f in self.demand_forecasts.values()
			if f["tenant_id"] == tenant and f["period"] == period and f["status"] == "active"
		)
		utilisation = round(planned_demand / available_capacity * 100, 2) if available_capacity else 0.0
		record: dict[str, Any] = {
			"id": self._id("cap"),
			"type": "scm_spl_capacity_plan",
			"tenant_id": tenant,
			"resource_id": resource_id,
			"resource_type": resource_type,
			"period": period,
			"available_capacity": available_capacity,
			"planned_demand": planned_demand,
			"utilisation_pct": utilisation,
			"unit": unit,
			"notes": notes,
			"status": "overloaded" if utilisation > 100 else "available",
			"created_at": self._now(),
		}
		self.capacity_plans[record["id"]] = record
		self._emit(tenant, "capacity_plan_created", record["id"], "scm_spl_capacity_plan", record["status"])
		return deepcopy(record)

	async def list_capacity_plans(
		self,
		period: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List capacity plans."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(c) for c in self.capacity_plans.values() if c["tenant_id"] == tenant]
		if period:
			items = [c for c in items if c["period"] == period]
		return items

	# ── Supply/demand balance ─────────────────────────────────────────────────

	async def create_supply_demand_balance(
		self,
		sku: str,
		period: str,
		supply_quantity: float,
		demand_quantity: float,
		opening_stock: float = 0.0,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compute and record supply/demand balance for a SKU/period."""
		tenant = self._tenant(tenant_id)
		closing_stock = opening_stock + supply_quantity - demand_quantity
		surplus_shortage = closing_stock  # positive = surplus, negative = shortage
		status = "balanced"
		if surplus_shortage > demand_quantity * 0.1:
			status = "surplus"
		elif surplus_shortage < 0:
			status = "shortage"
		record: dict[str, Any] = {
			"id": self._id("sdb"),
			"type": "scm_spl_supply_demand_balance",
			"tenant_id": tenant,
			"sku": sku,
			"period": period,
			"opening_stock": opening_stock,
			"supply_quantity": supply_quantity,
			"demand_quantity": demand_quantity,
			"closing_stock": round(closing_stock, 4),
			"surplus_shortage": round(surplus_shortage, 4),
			"status": status,
			"created_at": self._now(),
		}
		self.supply_demand_balances[record["id"]] = record
		self._emit(tenant, f"supply_demand_{status}", record["id"], "scm_spl_supply_demand_balance", status)
		return deepcopy(record)

	async def list_supply_demand_balances(
		self,
		sku: str | None = None,
		period: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List supply/demand balance records."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(b) for b in self.supply_demand_balances.values() if b["tenant_id"] == tenant]
		if sku:
			items = [b for b in items if b["sku"] == sku]
		if period:
			items = [b for b in items if b["period"] == period]
		return items

	# ── Supply exceptions ─────────────────────────────────────────────────────

	async def raise_supply_exception(
		self,
		sku: str,
		exception_type: str,
		description: str,
		severity: str = "medium",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Raise a supply planning exception (shortage, over-supply, capacity breach)."""
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": self._id("supexc"),
			"type": "scm_spl_supply_exception",
			"tenant_id": tenant,
			"sku": sku,
			"exception_type": exception_type,
			"description": description,
			"severity": severity,
			"status": "open",
			"created_at": self._now(),
		}
		self.supply_exceptions[record["id"]] = record
		self._emit(tenant, "supply_exception_raised", record["id"], "scm_spl_supply_exception", "open")
		return deepcopy(record)

	async def resolve_supply_exception(
		self,
		exception_id: str,
		resolution: str,
		resolved_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Resolve a supply exception."""
		tenant = self._tenant(tenant_id)
		exc = self.supply_exceptions.get(exception_id)
		if not exc or exc["tenant_id"] != tenant:
			raise KeyError(f"exception '{exception_id}' not found")
		exc["status"] = "resolved"
		exc["resolution"] = resolution
		exc["resolved_by"] = resolved_by
		exc["resolved_at"] = self._now()
		self._emit(tenant, "supply_exception_resolved", exception_id, "scm_spl_supply_exception", "resolved")
		return deepcopy(exc)

	# ── Analytics ─────────────────────────────────────────────────────────────

	async def planning_dashboard(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Supply planning KPI dashboard."""
		tenant = self._tenant(tenant_id)
		forecasts = [f for f in self.demand_forecasts.values() if f["tenant_id"] == tenant]
		reconciled = [f for f in forecasts if f.get("accuracy_pct") is not None]
		avg_accuracy = round(sum(f["accuracy_pct"] for f in reconciled) / len(reconciled), 2) if reconciled else None
		balances = [b for b in self.supply_demand_balances.values() if b["tenant_id"] == tenant]
		by_balance_status: dict[str, int] = {}
		for b in balances:
			by_balance_status[b["status"]] = by_balance_status.get(b["status"], 0) + 1
		return {
			"tenant_id": tenant,
			"total_forecasts": len(forecasts),
			"forecast_accuracy_pct": avg_accuracy,
			"active_replenishment_rules": sum(1 for r in self.replenishment_rules.values() if r["tenant_id"] == tenant and r["status"] == "active"),
			"planned_orders": len([p for p in self.planned_orders.values() if p["tenant_id"] == tenant]),
			"supply_demand_balances": by_balance_status,
			"open_exceptions": sum(1 for e in self.supply_exceptions.values() if e["tenant_id"] == tenant and e["status"] == "open"),
			"capacity_overloaded": sum(1 for c in self.capacity_plans.values() if c["tenant_id"] == tenant and c["status"] == "overloaded"),
			"generated_at": self._now(),
		}

	async def bulk_create_forecasts(
		self,
		forecasts_data: list[dict[str, Any]],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Bulk-create demand forecasts."""
		tenant = self._tenant(tenant_id)
		tasks = [self.create_demand_forecast(tenant_id=tenant, **f) for f in forecasts_data]
		raw = await asyncio.gather(*tasks, return_exceptions=True)
		results, errors = [], []
		for item in raw:
			if isinstance(item, Exception):
				errors.append(str(item))
			else:
				results.append(item)
		return {"created": len(results), "failed": len(errors), "forecasts": results, "errors": errors}
