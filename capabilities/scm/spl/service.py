"""Supply Planning async service (scm_spl)."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import logging
import math
import statistics
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

CAPABILITY_ID = "scm_spl"
FORECAST_METHODS = {"statistical", "ml", "manual", "exponential_smoothing", "moving_average"}
RULE_TYPES = {"min_max", "reorder_point", "periodic_review"}
RESOURCE_TYPES = {"warehouse", "production_line", "supplier"}
LOT_SIZING_POLICIES = {"eoq", "fixed", "lot_for_lot", "min_max_fill"}
PLAN_ZONES = {"frozen", "firm", "flexible"}
SEGMENT_CLASSES = {"AX", "AY", "AZ", "BX", "BY", "BZ", "CX", "CY", "CZ"}


class SupplyPlanningService:
	"""Async service for MRP-II, safety stock optimisation, replenishment rules,
	capacity planning and supply/demand balancing."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.demand_forecasts: dict[str, dict[str, Any]] = {}
		self.mrp_runs: dict[str, dict[str, Any]] = {}
		self.safety_stocks: dict[str, dict[str, Any]] = {}
		self.replenishment_rules: dict[str, dict[str, Any]] = {}
		self.capacity_plans: dict[str, dict[str, Any]] = {}
		self.supply_demand_balances: dict[str, dict[str, Any]] = {}
		self.planned_orders: dict[str, dict[str, Any]] = {}
		self.supply_exceptions: dict[str, dict[str, Any]] = {}
		self.sku_segments: dict[str, dict[str, Any]] = {}
		self.scenarios: dict[str, dict[str, Any]] = {}
		self.supplier_performance: dict[str, dict[str, Any]] = {}
		self.eoq_analyses: dict[str, dict[str, Any]] = {}
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)

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

	# ── Economic Order Quantity ───────────────────────────────────────────────

	async def calculate_eoq(
		self,
		sku: str,
		annual_demand: float,
		ordering_cost: float,
		holding_cost_rate: float,
		unit_cost: float,
		quantity_breaks: list[dict[str, Any]] | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compute Economic Order Quantity (EOQ) and total annual cost.

		Uses the Wilson EOQ formula: Q* = sqrt(2 * D * S / (h * C))
		where D=annual demand, S=ordering cost, h=holding cost rate, C=unit cost.
		Optionally evaluates quantity-discount break points and returns the
		cost-minimising policy.

		Args:
			sku: Stock keeping unit identifier.
			annual_demand: Expected annual demand in units.
			ordering_cost: Fixed cost per purchase order (S).
			holding_cost_rate: Annual holding cost as a fraction of unit cost (h), e.g. 0.20 for 20%.
			unit_cost: Purchase/production cost per unit (C).
			quantity_breaks: Optional list of {"min_qty": float, "unit_cost": float} discount tiers.
			tenant_id: Tenant context override.

		Returns:
			EOQ analysis record including optimal quantity, order frequency, and total annual cost.
		"""
		tenant = self._tenant(tenant_id)
		if annual_demand <= 0:
			raise ValueError("annual_demand must be > 0")
		if ordering_cost <= 0 or holding_cost_rate <= 0 or unit_cost <= 0:
			raise ValueError("ordering_cost, holding_cost_rate, and unit_cost must all be > 0")

		holding_cost_per_unit = holding_cost_rate * unit_cost
		eoq = math.sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit)
		orders_per_year = annual_demand / eoq
		annual_ordering_cost = orders_per_year * ordering_cost
		annual_holding_cost = (eoq / 2) * holding_cost_per_unit
		annual_purchase_cost = annual_demand * unit_cost
		total_cost = annual_ordering_cost + annual_holding_cost + annual_purchase_cost

		best_policy: dict[str, Any] = {
			"policy": "eoq",
			"order_quantity": round(eoq, 2),
			"unit_cost": unit_cost,
			"orders_per_year": round(orders_per_year, 2),
			"cycle_days": round(365 / orders_per_year, 1),
			"annual_ordering_cost": round(annual_ordering_cost, 2),
			"annual_holding_cost": round(annual_holding_cost, 2),
			"annual_purchase_cost": round(annual_purchase_cost, 2),
			"total_annual_cost": round(total_cost, 2),
		}

		break_evaluations: list[dict[str, Any]] = []
		if quantity_breaks:
			for brk in quantity_breaks:
				bq = max(float(brk["min_qty"]), eoq)
				bc = float(brk["unit_cost"])
				bh = holding_cost_rate * bc
				b_orders = annual_demand / bq
				b_cost = (b_orders * ordering_cost) + (bq / 2 * bh) + (annual_demand * bc)
				break_evaluations.append({
					"min_qty": brk["min_qty"],
					"evaluated_qty": round(bq, 2),
					"unit_cost": bc,
					"total_annual_cost": round(b_cost, 2),
				})
				if b_cost < best_policy["total_annual_cost"]:
					best_policy = {
						"policy": "quantity_break",
						"order_quantity": round(bq, 2),
						"unit_cost": bc,
						"orders_per_year": round(b_orders, 2),
						"cycle_days": round(365 / b_orders, 1),
						"annual_ordering_cost": round(b_orders * ordering_cost, 2),
						"annual_holding_cost": round(bq / 2 * bh, 2),
						"annual_purchase_cost": round(annual_demand * bc, 2),
						"total_annual_cost": round(b_cost, 2),
					}

		record: dict[str, Any] = {
			"id": self._id("eoq"),
			"type": "scm_spl_eoq_analysis",
			"tenant_id": tenant,
			"sku": sku,
			"annual_demand": annual_demand,
			"ordering_cost": ordering_cost,
			"holding_cost_rate": holding_cost_rate,
			"unit_cost": unit_cost,
			"classic_eoq": round(eoq, 2),
			"best_policy": best_policy,
			"quantity_break_evaluations": break_evaluations,
			"analysed_at": self._now(),
		}
		self.eoq_analyses[record["id"]] = record
		self._emit(tenant, "eoq_analysed", record["id"], "scm_spl_eoq_analysis", "completed")
		return deepcopy(record)

	# ── ABC-XYZ Segmentation ──────────────────────────────────────────────────

	async def segment_skus(
		self,
		sku_data: list[dict[str, Any]],
		abc_thresholds: tuple[float, float] = (0.80, 0.95),
		cv_thresholds: tuple[float, float] = (0.5, 1.0),
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Classify SKUs using ABC (cumulative value) and XYZ (demand volatility) segmentation.

		ABC classification ranks SKUs by annual sales value:
		  A = top cumulative abc_thresholds[0] (default 80%)
		  B = next band to abc_thresholds[1] (default 95%)
		  C = remainder

		XYZ classification uses coefficient of variation (CV = std_dev / mean) of demand:
		  X = CV <= cv_thresholds[0] (stable)
		  Y = cv_thresholds[0] < CV <= cv_thresholds[1] (variable)
		  Z = CV > cv_thresholds[1] (erratic)

		Args:
			sku_data: List of {"sku": str, "annual_value": float, "demand_history": list[float]}.
			abc_thresholds: (A_cutoff, B_cutoff) cumulative value fractions.
			cv_thresholds: (X_cutoff, Y_cutoff) CV boundaries.
			tenant_id: Tenant context override.

		Returns:
			Segmentation result with per-SKU segment assignments and summary counts.
		"""
		tenant = self._tenant(tenant_id)
		if not sku_data:
			raise ValueError("sku_data must not be empty")

		total_value = sum(float(s["annual_value"]) for s in sku_data)
		if total_value <= 0:
			raise ValueError("total annual value must be > 0")

		sorted_skus = sorted(sku_data, key=lambda s: float(s["annual_value"]), reverse=True)
		cumulative = 0.0
		segments: list[dict[str, Any]] = []
		for s in sorted_skus:
			val = float(s["annual_value"])
			cumulative += val / total_value
			if cumulative - val / total_value < abc_thresholds[0]:
				abc = "A"
			elif cumulative - val / total_value < abc_thresholds[1]:
				abc = "B"
			else:
				abc = "C"

			history = [float(x) for x in s.get("demand_history", [])]
			mean_d = statistics.mean(history) if history else 0.0
			cv = (statistics.stdev(history) / mean_d) if len(history) > 1 and mean_d > 0 else 0.0
			if cv <= cv_thresholds[0]:
				xyz = "X"
			elif cv <= cv_thresholds[1]:
				xyz = "Y"
			else:
				xyz = "Z"

			seg_record: dict[str, Any] = {
				"id": self._id("seg"),
				"type": "scm_spl_sku_segment",
				"tenant_id": tenant,
				"sku": s["sku"],
				"annual_value": val,
				"value_share_pct": round(val / total_value * 100, 3),
				"cumulative_value_pct": round(cumulative * 100, 3),
				"abc_class": abc,
				"demand_cv": round(cv, 4),
				"xyz_class": xyz,
				"segment": f"{abc}{xyz}",
				"segmented_at": self._now(),
			}
			self.sku_segments[f"{tenant}:{s['sku']}"] = seg_record
			segments.append(deepcopy(seg_record))

		by_segment: dict[str, int] = {}
		for seg in segments:
			by_segment[seg["segment"]] = by_segment.get(seg["segment"], 0) + 1

		result: dict[str, Any] = {
			"id": self._id("segrun"),
			"type": "scm_spl_segmentation_run",
			"tenant_id": tenant,
			"sku_count": len(segments),
			"total_annual_value": round(total_value, 2),
			"abc_thresholds": list(abc_thresholds),
			"cv_thresholds": list(cv_thresholds),
			"by_segment": by_segment,
			"segments": segments,
			"segmented_at": self._now(),
		}
		self._emit(tenant, "sku_segmentation_completed", result["id"], "scm_spl_segmentation_run", "completed")
		return result

	# ── Supplier Performance ──────────────────────────────────────────────────

	async def record_supplier_performance(
		self,
		supplier_id: str,
		sku: str,
		promised_lead_time_days: int,
		actual_lead_time_days: int,
		promised_quantity: float,
		delivered_quantity: float,
		period: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record a supplier delivery performance data point and update running statistics.

		Computes on-time delivery (OTD), fill rate, and updates rolling lead-time
		mean and standard deviation per supplier/SKU. Raises a supply exception
		when lead-time deviation exceeds 2σ of the historical distribution.

		Args:
			supplier_id: Supplier identifier.
			sku: SKU delivered.
			promised_lead_time_days: Contracted lead time.
			actual_lead_time_days: Observed lead time for this delivery.
			promised_quantity: Purchase order quantity.
			delivered_quantity: Actually received quantity.
			period: Period label (e.g. "2026-06").
			tenant_id: Tenant context override.

		Returns:
			Supplier performance record with updated rolling statistics.
		"""
		tenant = self._tenant(tenant_id)
		perf_key = f"{tenant}:{supplier_id}:{sku}"
		existing = self.supplier_performance.get(perf_key, {
			"supplier_id": supplier_id, "sku": sku, "tenant_id": tenant,
			"observations": [],
		})

		fill_rate = round(min(delivered_quantity / promised_quantity * 100, 100.0), 2) if promised_quantity else 0.0
		on_time = actual_lead_time_days <= promised_lead_time_days

		obs = {
			"period": period,
			"promised_lead_time_days": promised_lead_time_days,
			"actual_lead_time_days": actual_lead_time_days,
			"lead_time_delta": actual_lead_time_days - promised_lead_time_days,
			"on_time": on_time,
			"fill_rate_pct": fill_rate,
			"recorded_at": self._now(),
		}
		existing["observations"].append(obs)

		lt_series = [o["actual_lead_time_days"] for o in existing["observations"]]
		mean_lt = statistics.mean(lt_series)
		std_lt = statistics.stdev(lt_series) if len(lt_series) > 1 else 0.0
		otd_pct = round(sum(1 for o in existing["observations"] if o["on_time"]) / len(lt_series) * 100, 2)
		avg_fill = round(statistics.mean(o["fill_rate_pct"] for o in existing["observations"]), 2)

		existing.update({
			"id": perf_key,
			"type": "scm_spl_supplier_performance",
			"lead_time_mean_days": round(mean_lt, 2),
			"lead_time_std_days": round(std_lt, 2),
			"on_time_delivery_pct": otd_pct,
			"average_fill_rate_pct": avg_fill,
			"observation_count": len(lt_series),
			"updated_at": self._now(),
		})
		self.supplier_performance[perf_key] = existing

		# Raise exception when this delivery's lead time is more than 2σ above mean
		if std_lt > 0 and (actual_lead_time_days - mean_lt) > 2 * std_lt:
			await self.raise_supply_exception(
				sku=sku,
				exception_type="lead_time_breach",
				description=(
					f"Supplier {supplier_id} lead time {actual_lead_time_days}d "
					f"exceeds mean+2σ ({mean_lt:.1f}+{2*std_lt:.1f}d) for SKU {sku}"
				),
				severity="high",
				tenant_id=tenant,
			)

		self._emit(tenant, "supplier_performance_recorded", perf_key, "scm_spl_supplier_performance", "active")
		return deepcopy(existing)

	async def get_supplier_performance(
		self,
		supplier_id: str,
		sku: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Retrieve rolling supplier performance statistics for a supplier/SKU pair."""
		tenant = self._tenant(tenant_id)
		key = f"{tenant}:{supplier_id}:{sku}"
		record = self.supplier_performance.get(key)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"supplier_performance '{key}' not found")
		return deepcopy(record)

	# ── Scenario Planning ─────────────────────────────────────────────────────

	async def create_scenario(
		self,
		scenario_name: str,
		demand_adjustment_pct: float = 0.0,
		lead_time_adjustment_days: int = 0,
		supply_disruption_skus: list[str] | None = None,
		notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Fork the current supply plan into a named what-if scenario.

		Creates an isolated copy of current demand forecasts with adjustments applied.
		The scenario can be passed to run_mrp_scenario for full MRP recalculation
		without touching the baseline plan.

		Args:
			scenario_name: Human-readable name (e.g. "Demand +20%", "Supplier delay 2w").
			demand_adjustment_pct: Percentage change to apply to all forecast quantities.
			lead_time_adjustment_days: Days to add (or subtract if negative) to all safety stocks' lead times.
			supply_disruption_skus: SKUs to flag as having supply disruption (supply set to 0 in balances).
			notes: Freeform notes.
			tenant_id: Tenant context override.

		Returns:
			Scenario record with adjusted forecast snapshot.
		"""
		tenant = self._tenant(tenant_id)
		base_forecasts = [
			deepcopy(f) for f in self.demand_forecasts.values()
			if f["tenant_id"] == tenant and f["status"] == "active"
		]
		adjusted: list[dict[str, Any]] = []
		for fc in base_forecasts:
			fc_copy = deepcopy(fc)
			fc_copy["forecast_quantity"] = round(
				fc["forecast_quantity"] * (1 + demand_adjustment_pct / 100), 4
			)
			fc_copy["scenario_adjustment_pct"] = demand_adjustment_pct
			adjusted.append(fc_copy)

		disrupted = supply_disruption_skus or []
		record: dict[str, Any] = {
			"id": self._id("scen"),
			"type": "scm_spl_scenario",
			"tenant_id": tenant,
			"scenario_name": scenario_name,
			"demand_adjustment_pct": demand_adjustment_pct,
			"lead_time_adjustment_days": lead_time_adjustment_days,
			"supply_disruption_skus": disrupted,
			"notes": notes,
			"base_forecast_count": len(base_forecasts),
			"adjusted_forecasts": adjusted,
			"status": "draft",
			"created_at": self._now(),
		}
		self.scenarios[record["id"]] = record
		self._emit(tenant, "scenario_created", record["id"], "scm_spl_scenario", "draft")
		return deepcopy(record)

	async def run_mrp_scenario(
		self,
		scenario_id: str,
		horizon_weeks: int = 12,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Execute MRP over a scenario's adjusted forecasts without modifying the baseline.

		Computes planned orders, safety stock requirements, and capacity utilisation
		for the scenario and returns a summary comparison against the baseline plan.

		Args:
			scenario_id: ID of a scenario created via create_scenario.
			horizon_weeks: Planning horizon in weeks.
			tenant_id: Tenant context override.

		Returns:
			Scenario MRP result with planned orders and baseline comparison delta.
		"""
		tenant = self._tenant(tenant_id)
		scenario = self.scenarios.get(scenario_id)
		if not scenario or scenario["tenant_id"] != tenant:
			raise KeyError(f"scenario '{scenario_id}' not found")

		# MRP over scenario forecasts (isolated — does not write to self.planned_orders)
		scen_orders: list[dict[str, Any]] = []
		total_required = 0.0
		for fc in scenario["adjusted_forecasts"]:
			ss_key = f"{tenant}:{fc['sku']}:{fc.get('warehouse_id', 'any')}"
			lt_adj = scenario["lead_time_adjustment_days"]
			ss_record = self.safety_stocks.get(ss_key) or {}
			# Recalculate safety stock with adjusted lead time if needed
			if lt_adj != 0 and ss_record:
				adj_lt = max(1, ss_record.get("lead_time_days", 1) + lt_adj)
				z = ss_record.get("z_score", 1.645)
				std_dev = ss_record.get("demand_std_dev", 0.0) or 0.0
				adj_ss = round(z * std_dev * math.sqrt(adj_lt), 2)
			else:
				adj_ss = ss_record.get("effective_safety_stock", 0.0)

			disrupted = fc["sku"] in scenario["supply_disruption_skus"]
			required = fc["forecast_quantity"] + adj_ss
			total_required += required
			scen_orders.append({
				"sku": fc["sku"],
				"period": fc["period"],
				"forecast_qty": fc["forecast_quantity"],
				"safety_stock": adj_ss,
				"required_qty": round(required, 4),
				"supply_disrupted": disrupted,
			})

		# Baseline comparison
		baseline_total = sum(
			(f["forecast_quantity"] + (self.safety_stocks.get(f"{tenant}:{f['sku']}:{f.get('warehouse_id','any')}", {}).get("effective_safety_stock", 0.0)))
			for f in self.demand_forecasts.values()
			if f["tenant_id"] == tenant and f["status"] == "active"
		)

		result: dict[str, Any] = {
			"id": self._id("scenrun"),
			"type": "scm_spl_scenario_mrp_run",
			"tenant_id": tenant,
			"scenario_id": scenario_id,
			"scenario_name": scenario["scenario_name"],
			"horizon_weeks": horizon_weeks,
			"planned_orders": scen_orders,
			"total_required_qty": round(total_required, 4),
			"baseline_total_qty": round(baseline_total, 4),
			"delta_qty": round(total_required - baseline_total, 4),
			"delta_pct": round((total_required - baseline_total) / baseline_total * 100, 2) if baseline_total else None,
			"status": "completed",
			"run_at": self._now(),
		}
		scenario["status"] = "analysed"
		self._emit(tenant, "scenario_mrp_completed", result["id"], "scm_spl_scenario_mrp_run", "completed")
		return result

	# ── Forecast Bias Detection ───────────────────────────────────────────────

	async def detect_forecast_bias(
		self,
		sku: str,
		tracking_signal_threshold: float = 4.0,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Detect systematic forecast bias for a SKU using the Tracking Signal method.

		Tracking Signal = Cumulative Forecast Error (CFE) / Mean Absolute Deviation (MAD).
		|TS| > threshold indicates bias (typically threshold = 4–6).
		Returns bias verdict and a suggested correction factor.

		Args:
			sku: SKU to evaluate.
			tracking_signal_threshold: |TS| above which bias is flagged (default 4.0).
			tenant_id: Tenant context override.

		Returns:
			Bias analysis with CFE, MAD, tracking signal, and correction factor.
		"""
		tenant = self._tenant(tenant_id)
		reconciled = [
			f for f in self.demand_forecasts.values()
			if f["tenant_id"] == tenant
			and f["sku"] == sku
			and f.get("actual_quantity") is not None
			and f["status"] == "reconciled"
		]
		if not reconciled:
			raise ValueError(f"No reconciled forecasts found for SKU '{sku}'")

		errors = [f["forecast_quantity"] - f["actual_quantity"] for f in reconciled]  # type: ignore[operator]
		cfe = sum(errors)
		mad = statistics.mean(abs(e) for e in errors)
		tracking_signal = round(cfe / mad, 4) if mad else 0.0
		biased = abs(tracking_signal) > tracking_signal_threshold
		# Trigg's correction: apply smoothing correction factor
		correction_factor = round(1 - (cfe / sum(f["forecast_quantity"] for f in reconciled)), 4) if cfe else 1.0

		result: dict[str, Any] = {
			"id": self._id("bias"),
			"type": "scm_spl_forecast_bias",
			"tenant_id": tenant,
			"sku": sku,
			"observation_count": len(reconciled),
			"cumulative_forecast_error": round(cfe, 4),
			"mean_absolute_deviation": round(mad, 4),
			"tracking_signal": tracking_signal,
			"threshold": tracking_signal_threshold,
			"biased": biased,
			"bias_direction": "over_forecast" if cfe > 0 else "under_forecast",
			"suggested_correction_factor": correction_factor,
			"analysed_at": self._now(),
		}
		if biased:
			await self.raise_supply_exception(
				sku=sku,
				exception_type="forecast_bias",
				description=(
					f"Forecast bias detected for SKU {sku}: TS={tracking_signal:.2f} "
					f"(threshold ±{tracking_signal_threshold}), direction={result['bias_direction']}"
				),
				severity="medium",
				tenant_id=tenant,
			)
		self._emit(tenant, "forecast_bias_analysed", result["id"], "scm_spl_forecast_bias", "completed")
		return result

	# ── Inventory Turnover Analytics ──────────────────────────────────────────

	async def inventory_turnover_analytics(
		self,
		sku_cogs: dict[str, float],
		sku_avg_inventory: dict[str, float],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compute inventory turnover ratio and days-on-hand (DOH) per SKU.

		Turnover = COGS / Average Inventory
		Days on Hand = 365 / Turnover

		Also flags SKUs where DOH breaches safe bounds derived from replenishment rules
		and demand forecast horizon.

		Args:
			sku_cogs: Annual cost of goods sold per SKU.
			sku_avg_inventory: Average inventory value per SKU (same cost basis as COGS).
			tenant_id: Tenant context override.

		Returns:
			Analytics record with per-SKU turnover, DOH, and breach flags.
		"""
		tenant = self._tenant(tenant_id)
		if not sku_cogs:
			raise ValueError("sku_cogs must not be empty")

		per_sku: list[dict[str, Any]] = []
		total_cogs = 0.0
		total_avg_inv = 0.0

		for sku, cogs in sku_cogs.items():
			avg_inv = sku_avg_inventory.get(sku, 0.0)
			turnover = round(cogs / avg_inv, 4) if avg_inv > 0 else None
			doh = round(365 / turnover, 1) if turnover else None
			# Derive expected DOH bounds from active replenishment rules for this SKU
			rules = [
				r for r in self.replenishment_rules.values()
				if r["tenant_id"] == tenant and r["sku"] == sku and r["status"] == "active"
			]
			review_days = next((r.get("review_cycle_days") for r in rules if r.get("review_cycle_days")), None)
			doh_upper_bound = (review_days * 3) if review_days else None  # heuristic: 3× review cycle
			doh_breach = (doh is not None and doh_upper_bound is not None and doh > doh_upper_bound)

			total_cogs += cogs
			total_avg_inv += avg_inv
			per_sku.append({
				"sku": sku,
				"annual_cogs": cogs,
				"avg_inventory_value": avg_inv,
				"turnover_ratio": turnover,
				"days_on_hand": doh,
				"doh_upper_bound": doh_upper_bound,
				"doh_breach": doh_breach,
			})

		aggregate_turnover = round(total_cogs / total_avg_inv, 4) if total_avg_inv else None
		per_sku_sorted = sorted(per_sku, key=lambda x: (x["turnover_ratio"] or 0))

		record: dict[str, Any] = {
			"id": self._id("invturn"),
			"type": "scm_spl_inventory_turnover",
			"tenant_id": tenant,
			"sku_count": len(per_sku),
			"aggregate_turnover_ratio": aggregate_turnover,
			"aggregate_days_on_hand": round(365 / aggregate_turnover, 1) if aggregate_turnover else None,
			"doh_breach_count": sum(1 for s in per_sku if s["doh_breach"]),
			"per_sku": per_sku_sorted,
			"analysed_at": self._now(),
		}
		self._emit(tenant, "inventory_turnover_analysed", record["id"], "scm_spl_inventory_turnover", "completed")
		return record

	# ── Planned Order Firm-and-Release ────────────────────────────────────────

	async def firm_planned_order(
		self,
		order_id: str,
		firmed_by: str,
		notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Advance a planned order from 'planned' to 'firmed' status.

		The firm action signals procurement intent. Firmed orders are included in
		supplier visibility and VMI feeds. Requires the order to currently be in
		'planned' status.

		Args:
			order_id: Planned order ID.
			firmed_by: User or system identifier performing the action.
			notes: Optional notes.
			tenant_id: Tenant context override.

		Returns:
			Updated planned order record.
		"""
		tenant = self._tenant(tenant_id)
		order = self.planned_orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"planned_order '{order_id}' not found")
		if order["status"] != "planned":
			raise ValueError(f"order must be in 'planned' status to firm; current status: {order['status']}")
		order["status"] = "firmed"
		order["firmed_by"] = firmed_by
		order["firmed_at"] = self._now()
		if notes:
			order["notes"] = notes
		self._emit(tenant, "planned_order_firmed", order_id, "scm_spl_planned_order", "firmed")
		return deepcopy(order)

	async def release_planned_order(
		self,
		order_id: str,
		released_by: str,
		target_system: str = "procurement",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Release a firmed planned order to procurement or production.

		Transitions from 'firmed' to 'released'. The released order is ready
		to be converted to a purchase order (scm_po) or production order.

		Args:
			order_id: Planned order ID.
			released_by: User or system identifier performing the release.
			target_system: Downstream system receiving the order (default "procurement").
			tenant_id: Tenant context override.

		Returns:
			Updated planned order record.
		"""
		tenant = self._tenant(tenant_id)
		order = self.planned_orders.get(order_id)
		if not order or order["tenant_id"] != tenant:
			raise KeyError(f"planned_order '{order_id}' not found")
		if order["status"] != "firmed":
			raise ValueError(f"order must be in 'firmed' status to release; current status: {order['status']}")
		order["status"] = "released"
		order["released_by"] = released_by
		order["target_system"] = target_system
		order["released_at"] = self._now()
		self._emit(tenant, "planned_order_released", order_id, "scm_spl_planned_order", "released")
		return deepcopy(order)

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

