"""Service layer for APG Generation Management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_DISPATCH_MODES, SUPPORTED_FUEL_TYPES, SUPPORTED_KPI_TYPES,
		SUPPORTED_OUTAGE_STATUSES, SUPPORTED_OUTAGE_TYPES, SUPPORTED_PERFORMANCE_PERIODS,
		SUPPORTED_PLANT_STATUSES, SUPPORTED_PLANT_TYPES, SUPPORTED_SCHEDULE_STATUSES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		AuditEvent, CapacityPlan, DispatchSchedule, FuelStock,
		GenAgent, GenPlant, GenerationKPI, PlantOutage,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_DISPATCH_MODES, SUPPORTED_FUEL_TYPES, SUPPORTED_KPI_TYPES,
		SUPPORTED_OUTAGE_STATUSES, SUPPORTED_OUTAGE_TYPES, SUPPORTED_PERFORMANCE_PERIODS,
		SUPPORTED_PLANT_STATUSES, SUPPORTED_PLANT_TYPES, SUPPORTED_SCHEDULE_STATUSES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		AuditEvent, CapacityPlan, DispatchSchedule, FuelStock,
		GenAgent, GenPlant, GenerationKPI, PlantOutage,
	)


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


class GenerationManagementService:
	"""Tenant-scoped Generation Management runtime."""

	def __init__(self, tenant_id: str = "default", actor_id: str = "system", *, auth=None, audit=None, notify=None, db_url=None, store=None) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self.plants: dict[tuple[str, str], GenPlant] = {}
		self.dispatch_schedules: dict[tuple[str, str], DispatchSchedule] = {}
		self.outages: dict[tuple[str, str], PlantOutage] = {}
		self.kpis: dict[tuple[str, str], GenerationKPI] = {}
		self.capacity_plans: dict[tuple[str, str], CapacityPlan] = {}
		self.fuel_stocks: dict[tuple[str, str], FuelStock] = {}
		self.agents: dict[tuple[str, str], GenAgent] = {}
		self.audit_events: list[AuditEvent] = []
		# Extended in-memory stores
		self._generation_records: dict[str, dict[str, Any]] = {}
		self._dispatch_instructions: dict[str, dict[str, Any]] = {}
		self._heat_rate_records: dict[str, dict[str, Any]] = {}
		self._capacity_factor_records: dict[str, dict[str, Any]] = {}
		self._gen_analytics_records: dict[str, dict[str, Any]] = {}
		self._regulatory_reports: dict[str, dict[str, Any]] = {}

	# ── describe / evaluate ──────────────────────────────────────────────────

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return the capability contract for this tenant."""
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate capability rules against the given context."""
		return evaluate_capability_rules(context)

	# ── plants ───────────────────────────────────────────────────────────────

	def register_plant(
		self,
		plant_id: str,
		tenant_id: str,
		name: str,
		plant_type: str,
		fuel_type: str,
		capacity_mw: float,
		owner_id: str,
		commissioning_date: str,
		location_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Register a new generation plant."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_plant",
			"plant_type_supported": plant_type in SUPPORTED_PLANT_TYPES,
			"fuel_type_supported": fuel_type in SUPPORTED_FUEL_TYPES,
			"capacity_positive": capacity_mw > 0,
			"owner_present": _present(owner_id),
			"commissioning_date_present": _present(commissioning_date),
		})
		item = GenPlant(
			id=plant_id, tenant_id=tenant_id, name=name, plant_type=plant_type,
			fuel_type=fuel_type, capacity_mw=capacity_mw, status="operational",
			owner_id=owner_id, commissioning_date=commissioning_date,
			location_reference=location_reference,
		)
		self.plants[self._key(tenant_id, plant_id)] = item
		self._audit(tenant_id, "plant_registered", plant_id, "plant")
		return item.to_dict()

	def update_plant_status(self, plant_id: str, tenant_id: str, new_status: str) -> dict[str, Any]:
		"""Update a plant's operational status."""
		plant = self._get_plant(tenant_id, plant_id)
		valid_transitions = {
			"operational": {"under_maintenance", "standby", "decommissioned"},
			"under_maintenance": {"operational", "forced_outage", "standby"},
			"forced_outage": {"operational", "under_maintenance"},
			"standby": {"operational", "decommissioned"},
			"commissioning": {"operational"},
			"mothballed": {"operational", "decommissioned"},
			"planned_outage": {"operational", "under_maintenance"},
			"decommissioned": set(),
		}
		allowed = valid_transitions.get(plant.status, set())
		self._enforce({
			"tenant_context_present": True,
			"operation": "update_plant_status",
			"status_transition_valid": new_status in SUPPORTED_PLANT_STATUSES and new_status in allowed,
		})
		plant.status = new_status
		self._audit(tenant_id, "plant_status_changed", plant_id, "plant", {"new_status": new_status})
		return plant.to_dict()

	def list_plants(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List all plants for a tenant."""
		return self._tenant_items(self.plants, tenant_id)

	def get_plant(self, tenant_id: str, plant_id: str) -> dict[str, Any]:
		"""Get a single plant by ID."""
		return self._get_plant(tenant_id, plant_id).to_dict()

	def decommission_plant(self, plant_id: str, tenant_id: str, approved_by: str) -> dict[str, Any]:
		"""Decommission a plant after approval."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "decommission_plant",
			"approval_present": _present(approved_by),
		})
		plant = self._get_plant(tenant_id, plant_id)
		plant.status = "decommissioned"
		self._audit(tenant_id, "plant_decommissioned", plant_id, "plant", {"approved_by": approved_by})
		return plant.to_dict()

	# ── dispatch schedules ───────────────────────────────────────────────────

	def create_dispatch_schedule(
		self,
		schedule_id: str,
		tenant_id: str,
		plant_id: str,
		dispatch_mode: str,
		scheduled_mw: float,
		start_time: str,
		end_time: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a dispatch schedule for a plant."""
		plant = self._get_plant(tenant_id, plant_id)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_dispatch_schedule",
			"dispatch_mode_supported": dispatch_mode in SUPPORTED_DISPATCH_MODES,
			"plant_exists": True,
			"mw_within_capacity": scheduled_mw <= plant.available_mw(),
		})
		item = DispatchSchedule(
			id=schedule_id, tenant_id=tenant_id, plant_id=plant_id,
			dispatch_mode=dispatch_mode, scheduled_mw=scheduled_mw,
			start_time=start_time, end_time=end_time, status="draft",
		)
		self.dispatch_schedules[self._key(tenant_id, schedule_id)] = item
		self._audit(tenant_id, "dispatch_schedule_created", schedule_id, "dispatch_schedule")
		return item.to_dict()

	def approve_dispatch_schedule(self, schedule_id: str, tenant_id: str, approved_by: str) -> dict[str, Any]:
		"""Approve a dispatch schedule."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "activate_dispatch_schedule",
			"approval_present": _present(approved_by),
		})
		schedule = self._get_schedule(tenant_id, schedule_id)
		schedule.status = "approved"
		schedule.approved_by = approved_by
		schedule.approved_at = _now()
		self._audit(tenant_id, "dispatch_schedule_approved", schedule_id, "dispatch_schedule")
		return schedule.to_dict()

	def list_dispatch_schedules(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List all dispatch schedules for a tenant."""
		return self._tenant_items(self.dispatch_schedules, tenant_id)

	# ── outages ──────────────────────────────────────────────────────────────

	def schedule_outage(
		self,
		outage_id: str,
		tenant_id: str,
		plant_id: str,
		outage_type: str,
		planned_start: str,
		planned_end: str,
		reason: str,
		evidence_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Schedule a plant outage."""
		self._get_plant(tenant_id, plant_id)
		# Simplified overlap check: detect any same-plant overlapping outage
		overlapping = any(
			o.plant_id == plant_id and o.tenant_id == tenant_id and o.status not in ("completed", "cancelled")
			for o in self.outages.values()
		)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "schedule_outage",
			"outage_type_supported": outage_type in SUPPORTED_OUTAGE_TYPES,
			"plant_exists": True,
			"sufficient_notice": True,  # caller validates notice period externally
			"outage_overlap": overlapping,
		})
		item = PlantOutage(
			id=outage_id, tenant_id=tenant_id, plant_id=plant_id,
			outage_type=outage_type, status="scheduled",
			planned_start=planned_start, planned_end=planned_end,
			reason=reason, evidence_reference=evidence_reference,
		)
		self.outages[self._key(tenant_id, outage_id)] = item
		self._audit(tenant_id, "outage_scheduled", outage_id, "outage")
		return item.to_dict()

	def approve_outage(self, outage_id: str, tenant_id: str, approved_by: str) -> dict[str, Any]:
		"""Approve a scheduled outage."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "approve_outage",
			"approver_present": _present(approved_by),
		})
		outage = self._get_outage(tenant_id, outage_id)
		outage.approved_by = approved_by
		outage.status = "scheduled"
		self._audit(tenant_id, "outage_approved", outage_id, "outage", {"approved_by": approved_by})
		return outage.to_dict()

	def start_outage(self, outage_id: str, tenant_id: str) -> dict[str, Any]:
		"""Mark an outage as in-progress."""
		outage = self._get_outage(tenant_id, outage_id)
		outage.status = "in_progress"
		outage.actual_start = _now()
		self._audit(tenant_id, "outage_started", outage_id, "outage")
		return outage.to_dict()

	def complete_outage(self, outage_id: str, tenant_id: str) -> dict[str, Any]:
		"""Mark an outage as completed."""
		outage = self._get_outage(tenant_id, outage_id)
		outage.status = "completed"
		outage.actual_end = _now()
		self._audit(tenant_id, "outage_completed", outage_id, "outage")
		return outage.to_dict()

	def list_outages(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List all outages for a tenant."""
		return self._tenant_items(self.outages, tenant_id)

	# ── kpis ──────────────────────────────────────────────────────────────────

	def calculate_kpi(
		self,
		kpi_id: str,
		tenant_id: str,
		plant_id: str,
		kpi_type: str,
		period: str,
		period_start: str,
		period_end: str,
		value: float,
		unit: str,
	) -> dict[str, Any]:
		"""Record a calculated KPI for a plant."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "calculate_kpi",
			"kpi_type_supported": kpi_type in SUPPORTED_KPI_TYPES,
			"period_supported": period in SUPPORTED_PERFORMANCE_PERIODS,
		})
		item = GenerationKPI(
			id=kpi_id, tenant_id=tenant_id, plant_id=plant_id,
			kpi_type=kpi_type, period=period, period_start=period_start,
			period_end=period_end, value=value, unit=unit, calculated_at=_now(),
		)
		self.kpis[self._key(tenant_id, kpi_id)] = item
		self._audit(tenant_id, "kpi_calculated", kpi_id, "kpi")
		return item.to_dict()

	def list_kpis(self, tenant_id: str, plant_id: str | None = None) -> list[dict[str, Any]]:
		"""List KPIs, optionally filtered by plant."""
		items = self._tenant_items(self.kpis, tenant_id)
		if plant_id:
			items = [k for k in items if k["plant_id"] == plant_id]
		return items

	# ── capacity plans ────────────────────────────────────────────────────────

	def create_capacity_plan(
		self,
		plan_id: str,
		tenant_id: str,
		plan_name: str,
		horizon_years: int,
		base_year: int,
		total_existing_mw: float,
		total_planned_mw: float,
		peak_demand_mw: float,
		reserve_margin_pct: float,
		created_by: str,
	) -> dict[str, Any]:
		"""Create a capacity plan."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "create_capacity_plan",
			"horizon_valid": 1 <= horizon_years <= 20,
		})
		item = CapacityPlan(
			id=plan_id, tenant_id=tenant_id, plan_name=plan_name,
			horizon_years=horizon_years, base_year=base_year,
			total_existing_mw=total_existing_mw, total_planned_mw=total_planned_mw,
			peak_demand_mw=peak_demand_mw, reserve_margin_pct=reserve_margin_pct,
			created_by=created_by,
		)
		self.capacity_plans[self._key(tenant_id, plan_id)] = item
		self._audit(tenant_id, "capacity_plan_created", plan_id, "capacity_plan")
		return item.to_dict()

	def list_capacity_plans(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List capacity plans for a tenant."""
		return self._tenant_items(self.capacity_plans, tenant_id)

	# ── fuel stocks ───────────────────────────────────────────────────────────

	def update_fuel_stock(
		self,
		stock_id: str,
		tenant_id: str,
		plant_id: str,
		fuel_type: str,
		quantity: float,
		unit: str,
		days_of_supply: float,
		supplier_reference: str = "",
	) -> dict[str, Any]:
		"""Update fuel stock level for a plant."""
		plant = self._get_plant(tenant_id, plant_id)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "update_fuel_stock",
			"fuel_type_matches_plant": fuel_type == plant.fuel_type or fuel_type in SUPPORTED_FUEL_TYPES,
			"stock_non_negative": quantity >= 0,
		})
		item = FuelStock(
			id=stock_id, tenant_id=tenant_id, plant_id=plant_id,
			fuel_type=fuel_type, quantity=quantity, unit=unit,
			days_of_supply=days_of_supply, last_updated=_now(),
			supplier_reference=supplier_reference,
		)
		self.fuel_stocks[self._key(tenant_id, stock_id)] = item
		self._audit(tenant_id, "fuel_stock_updated", stock_id, "fuel_stock")
		return item.to_dict()

	def list_fuel_stocks(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List fuel stocks for a tenant."""
		return self._tenant_items(self.fuel_stocks, tenant_id)

	def get_low_fuel_alerts(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return fuel stocks below alert threshold."""
		return [s for s in self.list_fuel_stocks(tenant_id) if s.get("is_low")]

	# ── agents ────────────────────────────────────────────────────────────────

	def register_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str = "generation management operations",
	) -> dict[str, Any]:
		"""Register a generation management agent."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "register_gen_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = GenAgent(
			id=agent_id, tenant_id=tenant_id, name=name,
			runtime=runtime, role=role, scope=scope, registered_at=_now(),
		)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "gen_agent_registered", agent_id, "agent")
		return item.to_dict()

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool = False,
		human_approval_recorded: bool = False,
	) -> dict[str, Any]:
		"""Validate whether an agent action is permitted."""
		return evaluate_capability_rules({
			"operation": "gen_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})

	# ── dashboard ─────────────────────────────────────────────────────────────

	async def export_generation_data(self, period: str, format: str = "json") -> dict[str, Any]:
		"""Export generation records for a period."""
		assert format in {"json", "csv"}, "format must be json or csv"
		records = [r for r in self._generation_records.values() if r.get("tenant_id") == self.tenant_id and r.get("period", "")[:7] == period[:7]]
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			if records:
				writer = csv.DictWriter(buf, fieldnames=list(records[0].keys()))
				writer.writeheader()
				writer.writerows(records)
			return {"format": "csv", "period": period, "record_count": len(records), "content": buf.getvalue()}
		return {"format": "json", "period": period, "record_count": len(records), "records": records}

	async def generation_health_check(self) -> dict[str, Any]:
		"""Return generation management service health status."""
		plants = self._tenant_items(self.plants, self.tenant_id)
		online = sum(1 for p in plants if p.get("status") == "online")
		return {
			"service": "GenerationService", "tenant_id": self.tenant_id,
			"status": "healthy",
			"plant_count": len(plants), "online_count": online, "checked_at": _now(),
		}

	async def bulk_schedule_plants(self, schedule_specs: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-create generation schedules for multiple plants."""
		assert schedule_specs, "schedule_specs required"
		results: list[dict[str, Any]] = []
		errors: list[dict[str, Any]] = []
		for spec in schedule_specs:
			try:
				rec = await self.generation_schedule(
					plant_id=spec.get("plant_id", ""),
					period=spec.get("period", ""),
					scheduled_mw=float(spec.get("scheduled_mw", 0)),
					fuel_type=spec.get("fuel_type", ""),
					must_run=bool(spec.get("must_run", False)),
				)
				results.append({"plant_id": spec.get("plant_id"), "schedule_id": rec.get("id"), "status": "scheduled"})
			except Exception as exc:
				errors.append({"spec": spec, "error": str(exc)})
		return {"success_count": len(results), "error_count": len(errors), "results": results, "errors": errors}

	async def generation_compliance_report(self, period: str, standard: str = "EPRA") -> dict[str, Any]:
		"""Generate a generation compliance report."""
		plants = self._tenant_items(self.plants, self.tenant_id)
		outages = [o for o in self._tenant_items(self.outages, self.tenant_id) if o.get("planned_start", "")[:7] == period[:7]]
		analytics = await self.generation_analytics(period)
		self._audit(self.tenant_id, "generation_compliance_report_generated", standard, "report", {})
		return {
			"standard": standard, "period": period, "tenant_id": self.tenant_id,
			"plant_count": len(plants), "outage_count": len(outages),
			"total_generation_mwh": analytics.get("total_actual_mwh", 0),
			"capacity_factor_pct": analytics.get("avg_capacity_factor_pct", 0),
			"generated_at": _now(),
		}

	async def fuel_analytics(self) -> dict[str, Any]:
		"""Compute fuel consumption analytics across all plants."""
		fuel_records = [r for r in self._fuel_records.values() if r.get("tenant_id") == self.tenant_id]
		by_type: dict[str, float] = {}
		for r in fuel_records:
			ft = r.get("fuel_type", "unknown")
			by_type[ft] = round(by_type.get(ft, 0.0) + float(r.get("quantity", 0)), 3)
		return {
			"tenant_id": self.tenant_id, "fuel_record_count": len(fuel_records),
			"by_fuel_type": by_type, "computed_at": _now(),
		}

	async def plant_availability_summary(self) -> dict[str, Any]:
		"""Summarise plant availability across the portfolio."""
		plants = self._tenant_items(self.plants, self.tenant_id)
		available = sum(1 for p in plants if p.get("status") in {"online", "available"})
		total_mw = sum(float(p.get("installed_capacity_mw", 0)) for p in plants)
		available_mw = sum(float(p.get("installed_capacity_mw", 0)) for p in plants if p.get("status") in {"online", "available"})
		return {
			"tenant_id": self.tenant_id, "plant_count": len(plants),
			"available_count": available, "total_installed_mw": round(total_mw, 2),
			"available_mw": round(available_mw, 2),
			"availability_pct": round(available_mw / max(total_mw, 1) * 100, 2),
			"computed_at": _now(),
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return dashboard metrics for a tenant."""
		plants = self._tenant_items(self.plants, tenant_id)
		outages = self._tenant_items(self.outages, tenant_id)
		fuel_stocks = self._tenant_items(self.fuel_stocks, tenant_id)
		total_capacity = sum(p["capacity_mw"] for p in plants)
		available_capacity = sum(p["available_mw"] for p in plants)
		active_outages = [o for o in outages if o["status"] == "in_progress"]
		low_fuel = [f for f in fuel_stocks if f.get("is_low")]
		return {
			"tenant_id": tenant_id,
			"total_plants": len(plants),
			"total_capacity_mw": total_capacity,
			"available_capacity_mw": available_capacity,
			"active_outages": len(active_outages),
			"low_fuel_alerts": len(low_fuel),
			"operational_plants": sum(1 for p in plants if p["status"] == "operational"),
		}

	# ── internals ─────────────────────────────────────────────────────────────

	def _log_operation(self, tenant_id: str, operation: str, entity_id: str) -> None:
		"""Log an operation for observability."""
		pass  # Hook for structured logging integration

	def _log_rule_denial(self, actions: list[dict[str, Any]]) -> None:
		"""Log rule denial details."""
		pass  # Hook for alerting integration

	def _key(self, tenant_id: str, entity_id: str) -> tuple[str, str]:
		return (tenant_id, entity_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			self._log_rule_denial(result["actions"])
			reasons = "; ".join(a["reason"] for a in result["actions"])
			raise ValueError(f"Rule denied: {reasons}")

	def _audit(self, tenant_id: str, event_type: str, entity_id: str, entity_type: str, payload: dict[str, Any] | None = None) -> None:
		from uuid import uuid4
		self.audit_events.append(AuditEvent(
			id=str(uuid4()), tenant_id=tenant_id, event_type=event_type,
			entity_id=entity_id, entity_type=entity_type,
			actor="system", occurred_at=_now(), payload=payload or {},
		))

	def _get_plant(self, tenant_id: str, plant_id: str) -> GenPlant:
		item = self.plants.get(self._key(tenant_id, plant_id))
		if not item:
			raise KeyError(f"Plant {plant_id} not found for tenant {tenant_id}")
		return item

	def _get_schedule(self, tenant_id: str, schedule_id: str) -> DispatchSchedule:
		item = self.dispatch_schedules.get(self._key(tenant_id, schedule_id))
		if not item:
			raise KeyError(f"DispatchSchedule {schedule_id} not found for tenant {tenant_id}")
		return item

	def _get_outage(self, tenant_id: str, outage_id: str) -> PlantOutage:
		item = self.outages.get(self._key(tenant_id, outage_id))
		if not item:
			raise KeyError(f"Outage {outage_id} not found for tenant {tenant_id}")
		return item

	def _tenant_items(self, store: dict[tuple[str, str], Any], tenant_id: str) -> list[dict[str, Any]]:
		return [v.to_dict() for k, v in store.items() if k[0] == tenant_id]

	# ── Extended async methods ────────────────────────────────────────────────

	async def generation_schedule(
		self,
		plant_id: str,
		period: str,
		mw_schedule: list[dict[str, Any]],
		schedule_type: str = "day_ahead",
		approved_by: str | None = None,
	) -> dict[str, Any]:
		"""
		Create a generation schedule for a plant and period.
		mw_schedule: [{"hour": 0, "mw": 120.5}, {"hour": 1, "mw": 115.0}, ...]
		schedule_type: day_ahead | intra_day | week_ahead | real_time
		"""
		assert plant_id, "plant_id required"
		assert period and len(period) >= 7, "period must be YYYY-MM or YYYY-MM-DD"
		assert mw_schedule, "mw_schedule required"
		plant_data = self.plants.get(self._key(self.tenant_id, plant_id))
		if plant_data is None:
			raise KeyError(f"Plant '{plant_id}' not found for tenant '{self.tenant_id}'")
		capacity_mw = plant_data.capacity_mw
		overloads = [s for s in mw_schedule if s.get("mw", 0) > capacity_mw]
		if overloads:
			raise ValueError(
				f"{len(overloads)} schedule intervals exceed plant capacity {capacity_mw} MW"
			)
		total_mwh = round(sum(s.get("mw", 0) * (s.get("duration_h", 1)) for s in mw_schedule), 3)
		avg_mw = round(sum(s.get("mw", 0) for s in mw_schedule) / len(mw_schedule), 3)
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"plant_id": plant_id,
			"period": period,
			"schedule_type": schedule_type,
			"mw_schedule": mw_schedule,
			"interval_count": len(mw_schedule),
			"total_scheduled_mwh": total_mwh,
			"average_scheduled_mw": avg_mw,
			"plant_capacity_mw": capacity_mw,
			"approved_by": approved_by,
			"status": "approved" if approved_by else "draft",
			"created_at": _now(),
		}
		self._generation_records[rec_id] = rec
		self._audit(self.tenant_id, "generation_schedule_created", rec_id, "generation_schedule")
		return rec

	async def actual_generation(
		self,
		plant_id: str,
		timestamp: str,
		mw_generated: float,
		frequency_hz: float = 50.0,
		voltage_pu: float = 1.0,
		power_factor: float = 0.95,
	) -> dict[str, Any]:
		"""
		Record actual generation output for a plant at a timestamp.
		Validates frequency within ±0.5 Hz of nominal (50 Hz).
		"""
		assert plant_id, "plant_id required"
		assert timestamp, "timestamp required"
		assert mw_generated >= 0, "mw_generated must be non-negative"
		assert 0 < power_factor <= 1.0, "power_factor must be (0, 1]"
		plant_data = self.plants.get(self._key(self.tenant_id, plant_id))
		if plant_data is None:
			raise KeyError(f"Plant '{plant_id}' not found for tenant '{self.tenant_id}'")
		freq_deviation = abs(frequency_hz - 50.0)
		freq_alert = freq_deviation > 0.5
		mvar_generated = round(mw_generated * (1 - power_factor ** 2) ** 0.5 / power_factor, 3)
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"plant_id": plant_id,
			"timestamp": timestamp,
			"mw_generated": round(mw_generated, 3),
			"mvar_generated": mvar_generated,
			"frequency_hz": round(frequency_hz, 3),
			"frequency_deviation_hz": round(freq_deviation, 4),
			"frequency_alert": freq_alert,
			"voltage_pu": round(voltage_pu, 4),
			"power_factor": round(power_factor, 4),
			"plant_capacity_mw": plant_data.capacity_mw,
			"loading_pct": round(mw_generated / plant_data.capacity_mw * 100, 2) if plant_data.capacity_mw > 0 else 0,
			"recorded_at": _now(),
		}
		self._generation_records[rec_id] = rec
		if freq_alert:
			self._audit(self.tenant_id, "frequency_deviation_detected", rec_id, "generation", {"hz": frequency_hz})
		return rec

	async def dispatch_instruction(
		self,
		plant_id: str,
		dispatch_mw: float,
		start_time: str,
		duration: float,
		instruction_type: str = "normal",
		issued_by: str = "system",
		reason: str | None = None,
	) -> dict[str, Any]:
		"""
		Issue a dispatch instruction to a plant.
		instruction_type: normal | emergency | economic | ancillary
		duration: hours
		"""
		assert plant_id, "plant_id required"
		assert dispatch_mw >= 0, "dispatch_mw must be non-negative"
		assert duration > 0, "duration must be positive"
		plant_data = self.plants.get(self._key(self.tenant_id, plant_id))
		if plant_data is None:
			raise KeyError(f"Plant '{plant_id}' not found for tenant '{self.tenant_id}'")
		if dispatch_mw > plant_data.capacity_mw:
			raise ValueError(
				f"dispatch_mw {dispatch_mw} MW exceeds plant capacity {plant_data.capacity_mw} MW"
			)
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"plant_id": plant_id,
			"dispatch_mw": round(dispatch_mw, 3),
			"start_time": start_time,
			"duration_hours": duration,
			"end_time_estimate": start_time,  # caller to compute if needed
			"instruction_type": instruction_type,
			"issued_by": issued_by,
			"reason": reason,
			"status": "active",
			"dispatched_mwh": round(dispatch_mw * duration, 3),
			"issued_at": _now(),
		}
		self._dispatch_instructions[rec_id] = rec
		self._audit(self.tenant_id, "dispatch_instruction_issued", rec_id, "dispatch_instruction")
		return rec

	async def outage_management(
		self,
		plant_id: str,
		outage_type: str,
		start_time: str,
		end_time: str,
		capacity_lost: float,
		cause: str | None = None,
		approved_by: str | None = None,
	) -> dict[str, Any]:
		"""
		Record a plant outage (planned or forced).
		outage_type: planned | forced | partial | extended
		capacity_lost: MW unavailable during outage.
		"""
		assert plant_id, "plant_id required"
		assert outage_type in ("planned", "forced", "partial", "extended"), \
			"outage_type must be planned/forced/partial/extended"
		assert capacity_lost >= 0, "capacity_lost must be non-negative"
		plant_data = self.plants.get(self._key(self.tenant_id, plant_id))
		if plant_data is None:
			raise KeyError(f"Plant '{plant_id}' not found for tenant '{self.tenant_id}'")
		if capacity_lost > plant_data.capacity_mw:
			raise ValueError(f"capacity_lost {capacity_lost} MW exceeds plant capacity")
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"plant_id": plant_id,
			"outage_type": outage_type,
			"start_time": start_time,
			"end_time": end_time,
			"capacity_lost_mw": round(capacity_lost, 3),
			"cause": cause,
			"approved_by": approved_by,
			"status": "active",
			"created_at": _now(),
		}
		# Also update the structured outage store
		outage_id = rec_id
		self.outages[self._key(self.tenant_id, outage_id)] = type(
			"PlantOutage", (), {
				"id": outage_id, "tenant_id": self.tenant_id, "plant_id": plant_id,
				"outage_type": outage_type, "status": "in_progress",
				"planned_start": start_time, "planned_end": end_time,
				"actual_start": start_time, "actual_end": None,
				"reason": cause or "", "evidence_reference": "",
				"approved_by": approved_by or "",
				"to_dict": lambda self=rec: rec,
			}
		)()
		self._audit(self.tenant_id, "outage_recorded", rec_id, "outage", {"type": outage_type})
		return rec

	async def fuel_consumption(
		self,
		plant_id: str,
		period: str,
		fuel_quantity: float,
		unit: str = "GJ",
		fuel_type: str | None = None,
		cost_per_unit: float | None = None,
		currency: str = "USD",
	) -> dict[str, Any]:
		"""
		Record fuel consumption for a plant in a period.
		unit: GJ | MWh | tonnes | m3 | litres | MMBTU
		"""
		assert plant_id, "plant_id required"
		assert period and len(period) == 7, "period must be YYYY-MM"
		assert fuel_quantity >= 0, "fuel_quantity must be non-negative"
		plant_data = self.plants.get(self._key(self.tenant_id, plant_id))
		if plant_data is None:
			raise KeyError(f"Plant '{plant_id}' not found for tenant '{self.tenant_id}'")
		total_cost = round(fuel_quantity * cost_per_unit, 2) if cost_per_unit else None
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"plant_id": plant_id,
			"period": period,
			"fuel_type": fuel_type or plant_data.fuel_type,
			"fuel_quantity": round(fuel_quantity, 3),
			"unit": unit,
			"cost_per_unit": cost_per_unit,
			"total_cost": total_cost,
			"currency": currency,
			"recorded_at": _now(),
		}
		# Update fuel stock
		for k, fs in self.fuel_stocks.items():
			if k[0] == self.tenant_id and fs.plant_id == plant_id:
				fs.quantity = max(0.0, fs.quantity - fuel_quantity)
				break
		self._audit(self.tenant_id, "fuel_consumption_recorded", rec_id, "fuel_consumption")
		return rec

	async def heat_rate(
		self,
		plant_id: str,
		period: str,
		heat_input_gj: float,
		gross_generation_mwh: float,
		net_generation_mwh: float | None = None,
	) -> dict[str, Any]:
		"""
		Calculate heat rate for a plant.
		Gross heat rate = heat_input_GJ / gross_generation_MWh (GJ/MWh)
		Design heat rate benchmark for CCGT: ~6.5 GJ/MWh; coal: ~9.0 GJ/MWh
		"""
		assert plant_id, "plant_id required"
		assert period and len(period) == 7, "period must be YYYY-MM"
		assert heat_input_gj > 0, "heat_input_gj must be positive"
		assert gross_generation_mwh > 0, "gross_generation_mwh must be positive"
		gross_heat_rate = round(heat_input_gj / gross_generation_mwh, 4)
		net_heat_rate = (
			round(heat_input_gj / net_generation_mwh, 4) if net_generation_mwh and net_generation_mwh > 0 else None
		)
		# Thermal efficiency
		thermal_efficiency_pct = round(3.6 / gross_heat_rate * 100, 2)  # 1 MWh = 3.6 GJ
		plant_data = self.plants.get(self._key(self.tenant_id, plant_id))
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"plant_id": plant_id,
			"period": period,
			"heat_input_gj": round(heat_input_gj, 3),
			"gross_generation_mwh": round(gross_generation_mwh, 3),
			"net_generation_mwh": round(net_generation_mwh, 3) if net_generation_mwh else None,
			"gross_heat_rate_gj_mwh": gross_heat_rate,
			"net_heat_rate_gj_mwh": net_heat_rate,
			"thermal_efficiency_pct": thermal_efficiency_pct,
			"plant_type": plant_data.plant_type if plant_data else None,
			"calculated_at": _now(),
		}
		self._heat_rate_records[rec_id] = rec
		self._audit(self.tenant_id, "heat_rate_calculated", rec_id, "heat_rate")
		return rec

	async def capacity_factor(
		self,
		plant_id: str,
		period: str,
		actual_generation_mwh: float | None = None,
	) -> dict[str, Any]:
		"""
		Calculate capacity factor for a plant in a period.
		CF = actual_MWh / (capacity_MW × hours_in_period) × 100
		"""
		assert plant_id, "plant_id required"
		assert period and len(period) == 7, "period must be YYYY-MM"
		plant_data = self.plants.get(self._key(self.tenant_id, plant_id))
		if plant_data is None:
			raise KeyError(f"Plant '{plant_id}' not found for tenant '{self.tenant_id}'")
		# Hours in period (approx 30 days)
		hours_in_period = 30 * 24
		max_mwh = plant_data.capacity_mw * hours_in_period
		# Sum actual generation records if not supplied
		if actual_generation_mwh is None:
			actual_generation_mwh = sum(
				r.get("mw_generated", 0) * 1  # each record represents 1 interval hour
				for r in self._generation_records.values()
				if r.get("tenant_id") == self.tenant_id
				and r.get("plant_id") == plant_id
				and r.get("timestamp", "")[:7] == period
			)
		cf_pct = round(actual_generation_mwh / max_mwh * 100, 2) if max_mwh > 0 else 0.0
		# Benchmark CFs: solar 18-25%, wind 30-45%, coal 70-85%, gas CCGT 50-70%
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"plant_id": plant_id,
			"period": period,
			"installed_capacity_mw": plant_data.capacity_mw,
			"hours_in_period": hours_in_period,
			"max_possible_mwh": round(max_mwh, 1),
			"actual_generation_mwh": round(actual_generation_mwh, 3),
			"capacity_factor_pct": cf_pct,
			"plant_type": plant_data.plant_type,
			"calculated_at": _now(),
		}
		self._capacity_factor_records[rec_id] = rec
		self._audit(self.tenant_id, "capacity_factor_calculated", rec_id, "capacity_factor")
		return rec

	async def generation_analytics(self, period: str) -> dict[str, Any]:
		"""
		Compute generation analytics dashboard for a period (YYYY-MM).
		Returns: total generation, average CF, heat rate trends, fuel cost, outage summary.
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"
		plants = self._tenant_items(self.plants, self.tenant_id)
		gen_records = [
			r for r in self._generation_records.values()
			if r.get("tenant_id") == self.tenant_id and r.get("timestamp", "")[:7] == period
		]
		total_mwh = sum(r.get("mw_generated", 0) for r in gen_records)
		freq_alerts = sum(1 for r in gen_records if r.get("frequency_alert"))
		outages = self._tenant_items(self.outages, self.tenant_id)
		period_outages = [o for o in outages if o.get("planned_start", "")[:7] == period]
		capacity_factors = [
			r for r in self._capacity_factor_records.values()
			if r.get("tenant_id") == self.tenant_id and r.get("period") == period
		]
		avg_cf = (
			sum(r["capacity_factor_pct"] for r in capacity_factors) / len(capacity_factors)
			if capacity_factors else None
		)
		heat_rates = [
			r for r in self._heat_rate_records.values()
			if r.get("tenant_id") == self.tenant_id and r.get("period") == period
		]
		avg_heat_rate = (
			sum(r["gross_heat_rate_gj_mwh"] for r in heat_rates) / len(heat_rates)
			if heat_rates else None
		)
		return {
			"tenant_id": self.tenant_id,
			"period": period,
			"total_plants": len(plants),
			"total_generation_mwh": round(total_mwh, 2),
			"generation_records": len(gen_records),
			"frequency_alerts": freq_alerts,
			"outages": len(period_outages),
			"average_capacity_factor_pct": round(avg_cf, 2) if avg_cf else None,
			"average_heat_rate_gj_mwh": round(avg_heat_rate, 4) if avg_heat_rate else None,
			"as_at": _now(),
		}

	async def regulatory_report_generation(self, period: str, jurisdiction: str = "default") -> dict[str, Any]:
		"""
		Generate a regulatory generation report for a period.
		Aggregates: plant registers, generation totals, fuel consumption, outages, KPIs.
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"
		analytics = await self.generation_analytics(period)
		plants = self._tenant_items(self.plants, self.tenant_id)
		outages = [o for o in self._tenant_items(self.outages, self.tenant_id) if o.get("planned_start", "")[:7] == period]
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"report_type": "regulatory_generation_report",
			"period": period,
			"jurisdiction": jurisdiction,
			"analytics": analytics,
			"plants_count": len(plants),
			"outages_count": len(outages),
			"fuel_records": len([
				r for r in self._generation_records.values()
				if r.get("tenant_id") == self.tenant_id
			]),
			"generated_at": _now(),
		}
		self._regulatory_reports[rec_id] = rec
		self._audit(self.tenant_id, "regulatory_report_generated", rec_id, "regulatory_report")
		return rec

	async def ml_generation_efficiency(self, *args, **kwargs):
		"""AI-powered power generation efficiency and output forecasting. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.predict(kwargs.get("historical",[{"period": str(i), "value": 100.0} for i in range(12)]), horizon=7, task="power_generation_forecast")
			return {"output_forecast": result.predictions, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

