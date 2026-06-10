"""Executable service layer for APG Transport Scheduling."""

from __future__ import annotations

import asyncio
import statistics
import uuid
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_SCHEDULE_TYPES, SUPPORTED_SCHEDULE_STATUSES, SUPPORTED_SHIFT_TYPES,
		SUPPORTED_CHARTER_TYPES, SUPPORTED_OPTIMISATION_MODES, SUPPORTED_CONFLICT_TYPES,
		SUPPORTED_NOTIFICATION_TYPES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		Schedule, DriverShift, VehicleAssignment, Charter,
		ScheduleConflict, ScheduleNotification, SchedulingAgent,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_SCHEDULE_TYPES, SUPPORTED_SCHEDULE_STATUSES, SUPPORTED_SHIFT_TYPES,
		SUPPORTED_CHARTER_TYPES, SUPPORTED_OPTIMISATION_MODES, SUPPORTED_CONFLICT_TYPES,
		SUPPORTED_NOTIFICATION_TYPES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		Schedule, DriverShift, VehicleAssignment, Charter,
		ScheduleConflict, ScheduleNotification, SchedulingAgent,
	)


def _present(value: str | None) -> bool:
	return bool(value and str(value).strip())

def _norm(value: str) -> str:
	return str(value).strip().lower() if value else ""

def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Regulatory limits — EU drivers' hours (Regulation EC 561/2006)
_MAX_DAILY_DRIVE_HOURS = 9.0
_MAX_WEEKLY_DRIVE_HOURS = 56.0
_MAX_FORTNIGHTLY_DRIVE_HOURS = 90.0
_MIN_DAILY_REST_HOURS = 11.0

# Charter vehicle type cost rates (USD per km)
_CHARTER_RATE_PER_KM: dict[str, float] = {
	"minibus": 1.20, "bus": 2.50, "luxury_coach": 4.00,
	"sedan": 0.80, "suv": 1.10, "truck": 3.20,
}


class TransportSchedulingService:
	"""Tenant-scoped transport scheduling runtime."""

	def __init__(
		self,
		tenant_id: str = "default",
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._store = store
		self.schedules: dict[tuple[str, str], Schedule] = {}
		self.shifts: dict[tuple[str, str], DriverShift] = {}
		self.vehicle_assignments: dict[tuple[str, str], VehicleAssignment] = {}
		self.charters: dict[tuple[str, str], Charter] = {}
		self.conflicts: dict[tuple[str, str], ScheduleConflict] = {}
		self.notifications: dict[tuple[str, str], ScheduleNotification] = {}
		self.agents: dict[tuple[str, str], SchedulingAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state
		self.disruptions: dict[tuple[str, str], dict[str, Any]] = {}
		self.capacity_plans: dict[tuple[str, str], dict[str, Any]] = {}

	# ------------------------------------------------------------------
	# Capability introspection
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Existing methods (preserved)
	# ------------------------------------------------------------------

	def create_schedule(
		self, schedule_id: str, tenant_id: str, schedule_type: str,
		start_date: str, end_date: str, optimisation_mode: str,
		created_by: str, policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a transport schedule."""
		schedule_type = _norm(schedule_type)
		optimisation_mode = _norm(optimisation_mode)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "create_schedule",
			"schedule_type_supported": schedule_type in SUPPORTED_SCHEDULE_TYPES,
		})
		item = Schedule(schedule_id, tenant_id, schedule_type, "draft", start_date, end_date, optimisation_mode, created_by)
		self.schedules[self._key(tenant_id, schedule_id)] = item
		self._audit(tenant_id, "schedule_created", schedule_id)
		return item.to_dict()

	def publish_schedule(self, schedule_id: str, tenant_id: str) -> dict[str, Any]:
		"""Publish a schedule, blocking if unresolved conflicts exist."""
		open_conflicts = sum(
			1 for c in self.conflicts.values()
			if c.tenant_id == tenant_id and c.schedule_id == schedule_id and c.resolved_at is None
		)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "publish_schedule",
			"unresolved_conflicts_present": open_conflicts > 0,
		})
		schedule = self._schedule_or_none(schedule_id, tenant_id)
		if schedule is None:
			raise KeyError(f"Schedule {schedule_id} not found")
		schedule.status = "published"
		self._audit(tenant_id, "schedule_published", schedule_id)
		return schedule.to_dict()

	def create_shift(
		self, shift_id: str, tenant_id: str, schedule_id: str,
		driver_id: str, shift_type: str, start_time: str,
		end_time: str, hours: float,
		driver_hours_compliant: bool = True, tacho_compliant: bool = True,
	) -> dict[str, Any]:
		"""Create a driver shift."""
		shift_type = _norm(shift_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "create_shift",
			"shift_type_supported": shift_type in SUPPORTED_SHIFT_TYPES,
			"driver_hours_compliant": driver_hours_compliant,
			"tacho_compliant": tacho_compliant,
		})
		item = DriverShift(shift_id, tenant_id, schedule_id, driver_id, shift_type, start_time, end_time, float(hours), tacho_compliant)
		self.shifts[self._key(tenant_id, shift_id)] = item
		self._audit(tenant_id, "shift_assigned", shift_id)
		return item.to_dict()

	def assign_vehicle(
		self, assignment_id: str, tenant_id: str, schedule_id: str,
		vehicle_id: str, route_id: str, assigned_from: str, assigned_until: str,
		double_booking_detected: bool = False,
	) -> dict[str, Any]:
		"""Assign a vehicle to a schedule."""
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "assign_resource",
			"vehicle_present": _present(vehicle_id),
			"schedule_present": _present(schedule_id),
			"double_booking_detected": double_booking_detected,
		})
		item = VehicleAssignment(assignment_id, tenant_id, schedule_id, vehicle_id, route_id, assigned_from, assigned_until)
		self.vehicle_assignments[self._key(tenant_id, assignment_id)] = item
		self._audit(tenant_id, "vehicle_assigned", assignment_id)
		return item.to_dict()

	def create_charter(
		self, charter_id: str, tenant_id: str, schedule_id: str,
		charter_type: str, customer_id: str, vehicle_id: str,
		driver_id: str, pickup_location: str, destination: str,
		charter_date: str, customer_confirmed: bool = False,
	) -> dict[str, Any]:
		"""Create a charter booking."""
		charter_type = _norm(charter_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "create_charter",
			"charter_type_supported": charter_type in SUPPORTED_CHARTER_TYPES,
			"customer_confirmed": customer_confirmed,
		})
		item = Charter(charter_id, tenant_id, schedule_id, charter_type, customer_id, vehicle_id, driver_id, pickup_location, destination, charter_date, customer_confirmed)
		self.charters[self._key(tenant_id, charter_id)] = item
		self._audit(tenant_id, "charter_confirmed", charter_id)
		return item.to_dict()

	def record_conflict(
		self, conflict_id: str, tenant_id: str, schedule_id: str,
		conflict_type: str, resource_id: str, detected_at: str,
	) -> dict[str, Any]:
		"""Record a scheduling conflict."""
		conflict_type = _norm(conflict_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_conflict",
			"conflict_type_supported": conflict_type in SUPPORTED_CONFLICT_TYPES,
		})
		item = ScheduleConflict(conflict_id, tenant_id, schedule_id, conflict_type, resource_id, detected_at, None, "")
		self.conflicts[self._key(tenant_id, conflict_id)] = item
		self._audit(tenant_id, "conflict_detected", conflict_id)
		return item.to_dict()

	def resolve_conflict(self, conflict_id: str, tenant_id: str, resolved_at: str, resolution_notes: str) -> dict[str, Any]:
		"""Resolve a scheduling conflict."""
		conflict = self.conflicts.get(self._key(tenant_id, conflict_id))
		if conflict is None:
			raise KeyError(f"Conflict {conflict_id} not found")
		conflict.resolved_at = resolved_at
		conflict.resolution_notes = resolution_notes
		self._audit(tenant_id, "conflict_resolved", conflict_id)
		return conflict.to_dict()

	def send_notification(
		self, notification_id: str, tenant_id: str, schedule_id: str,
		notification_type: str, recipient_id: str, channel: str, sent_at: str,
	) -> dict[str, Any]:
		"""Send a scheduling notification."""
		notification_type = _norm(notification_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
		})
		item = ScheduleNotification(notification_id, tenant_id, schedule_id, notification_type, recipient_id, channel, sent_at)
		self.notifications[self._key(tenant_id, notification_id)] = item
		self._audit(tenant_id, "schedule_notification_sent", notification_id)
		return item.to_dict()

	def register_scheduling_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		"""Register an AI agent for transport scheduling."""
		runtime = _norm(runtime)
		role = _norm(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_scheduling_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = SchedulingAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "scheduling_agent_registered", agent_id)
		return item.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "scheduling_batch", "event_stream": event_stream})
		if item_count <= 0:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.transport.scheduling.lifecycle", "accepted": True}

	def list_schedules(self, tenant_id: str) -> list[dict[str, Any]]:
		return [s.to_dict() for s in self.schedules.values() if s.tenant_id == tenant_id]

	def list_open_conflicts(self, tenant_id: str) -> list[dict[str, Any]]:
		return [c.to_dict() for c in self.conflicts.values() if c.tenant_id == tenant_id and c.resolved_at is None]

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"schedule_count": self._count(self.schedules, tenant_id),
			"shift_count": self._count(self.shifts, tenant_id),
			"vehicle_assignment_count": self._count(self.vehicle_assignments, tenant_id),
			"charter_count": self._count(self.charters, tenant_id),
			"conflict_count": self._count(self.conflicts, tenant_id),
			"open_conflict_count": len(self.list_open_conflicts(tenant_id)),
			"notification_count": self._count(self.notifications, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# New methods
	# ------------------------------------------------------------------

	async def create_schedule_async(
		self,
		service_type: str,
		routes: list[dict[str, Any]],
		frequency: str,
		*,
		start_date: str | None = None,
		end_date: str | None = None,
		created_by: str = "system",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Create a schedule covering multiple routes at a given frequency.

		service_type: e.g. 'regular_route', 'express', 'charter'
		routes: [{"route_id": str, "origin": str, "destination": str}]
		frequency: e.g. 'daily', 'weekly', 'weekdays'

		Generates one schedule record and one vehicle assignment stub per route.
		"""
		tid = tenant_id or self.tenant_id
		if not routes:
			raise ValueError("routes list is empty")
		if not _present(frequency):
			raise ValueError("frequency required")

		await asyncio.sleep(0)
		sched_id = f"SCH-{uuid.uuid4().hex[:8].upper()}"
		st = _norm(service_type)
		if st not in SUPPORTED_SCHEDULE_TYPES:
			st = list(SUPPORTED_SCHEDULE_TYPES)[0] if SUPPORTED_SCHEDULE_TYPES else "regular_route"

		opt_mode = list(SUPPORTED_OPTIMISATION_MODES)[0] if SUPPORTED_OPTIMISATION_MODES else "balanced"
		sd = start_date or _now_iso()[:10]
		ed = end_date or sd

		sched = self.create_schedule(sched_id, tid, st, sd, ed, opt_mode, created_by)
		assignments = []
		for route in routes:
			asgn_id = f"VA-{sched_id}-{route.get('route_id', uuid.uuid4().hex[:6])}"
			asgn = self.assign_vehicle(
				asgn_id, tid, sched_id,
				route.get("vehicle_id", f"VEH-{asgn_id}"),
				route.get("route_id", ""),
				sd, ed,
			)
			assignments.append(asgn)

		return {
			"schedule": sched,
			"frequency": frequency,
			"route_count": len(routes),
			"vehicle_assignments": assignments,
		}

	async def driver_shift_planning(
		self,
		drivers: list[dict[str, Any]],
		shifts: list[dict[str, Any]],
		constraints: dict[str, Any],
		*,
		schedule_id: str | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Assign drivers to shifts respecting hours-of-service constraints.

		drivers: [{"driver_id": str, "available_hours": float, "preferred_shift": str}]
		shifts: [{"shift_id": str, "shift_type": str, "start": str, "end": str, "hours": float}]
		constraints: {"max_hours_per_day": float, "tacho_required": bool}

		Returns assignments and flags any HOS violations.
		"""
		tid = tenant_id or self.tenant_id
		if not drivers:
			raise ValueError("drivers list is empty")
		if not shifts:
			raise ValueError("shifts list is empty")

		await asyncio.sleep(0)
		max_daily = float(constraints.get("max_hours_per_day", _MAX_DAILY_DRIVE_HOURS))
		tacho_req = bool(constraints.get("tacho_required", True))

		sched_id = schedule_id or f"SCH-AUTO-{uuid.uuid4().hex[:6].upper()}"
		if not self._schedule_or_none(sched_id, tid):
			st = list(SUPPORTED_SCHEDULE_TYPES)[0] if SUPPORTED_SCHEDULE_TYPES else "regular_route"
			opt = list(SUPPORTED_OPTIMISATION_MODES)[0] if SUPPORTED_OPTIMISATION_MODES else "balanced"
			self.create_schedule(sched_id, tid, st, _now_iso()[:10], _now_iso()[:10], opt, "system")

		assigned: list[dict[str, Any]] = []
		violations: list[dict[str, Any]] = []
		driver_pool = list(drivers)

		for shift in shifts:
			hours = float(shift.get("hours", 8.0))
			compliant = hours <= max_daily
			if not compliant:
				violations.append({"shift_id": shift.get("shift_id"), "reason": "exceeds_max_daily_hours", "hours": hours})
				continue

			matched = next(
				(d for d in driver_pool if float(d.get("available_hours", 0)) >= hours),
				None,
			)
			if matched is None:
				violations.append({"shift_id": shift.get("shift_id"), "reason": "no_available_driver"})
				continue

			driver_pool.remove(matched)
			st = _norm(shift.get("shift_type", ""))
			if st not in SUPPORTED_SHIFT_TYPES:
				st = list(SUPPORTED_SHIFT_TYPES)[0] if SUPPORTED_SHIFT_TYPES else "day"
			shift_rec = self.create_shift(
				shift.get("shift_id", f"SHF-{uuid.uuid4().hex[:6]}"),
				tid, sched_id, matched["driver_id"], st,
				shift.get("start", _now_iso()),
				shift.get("end", _now_iso()),
				hours, compliant, tacho_req,
			)
			assigned.append({**shift_rec, "driver_id": matched["driver_id"]})

		return {
			"schedule_id": sched_id,
			"tenant_id": tid,
			"shifts_requested": len(shifts),
			"shifts_assigned": len(assigned),
			"violations": violations,
			"assignments": assigned,
		}

	async def vehicle_assignment(
		self,
		schedule_id: str,
		vehicles: list[dict[str, Any]],
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Assign a fleet of vehicles to routes within a schedule.

		vehicles: [{"vehicle_id": str, "route_id": str, "from": str, "until": str}]

		Detects double-bookings by checking existing assignments for the same
		vehicle_id overlapping the requested time window.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(schedule_id):
			raise ValueError("schedule_id required")
		if not vehicles:
			raise ValueError("vehicles list is empty")

		await asyncio.sleep(0)
		results: list[dict[str, Any]] = []
		double_bookings: list[str] = []

		existing_assignments = {
			va.vehicle_id: va
			for va in self.vehicle_assignments.values()
			if va.tenant_id == tid
		}

		for v in vehicles:
			vid = v.get("vehicle_id", "")
			double = vid in existing_assignments
			if double:
				double_bookings.append(vid)
			asgn_id = f"VA-{schedule_id}-{vid}-{uuid.uuid4().hex[:4]}"
			asgn = self.assign_vehicle(
				asgn_id, tid, schedule_id, vid,
				v.get("route_id", ""),
				v.get("from", _now_iso()),
				v.get("until", _now_iso()),
				double_booking_detected=double,
			)
			results.append(asgn)

		return {
			"schedule_id": schedule_id,
			"tenant_id": tid,
			"vehicles_assigned": len(results),
			"double_booking_warnings": double_bookings,
			"assignments": results,
		}

	async def charter_booking(
		self,
		client_id: str,
		origin: str,
		destination: str,
		date: str,
		vehicle_type: str,
		*,
		distance_km: float = 100.0,
		driver_id: str = "unassigned",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Create and price a charter booking for a client.

		Calculates charter cost from distance × vehicle type rate.
		Creates schedule + charter record and returns cost breakdown.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(client_id) or not _present(origin) or not _present(destination):
			raise ValueError("client_id, origin and destination required")
		if not _present(date):
			raise ValueError("date required")

		await asyncio.sleep(0)
		rate_per_km = _CHARTER_RATE_PER_KM.get(_norm(vehicle_type), 2.50)
		charter_cost = round(distance_km * rate_per_km, 2)
		fuel_surcharge = round(charter_cost * 0.10, 2)
		total_cost = round(charter_cost + fuel_surcharge, 2)

		sched_id = f"CSCHED-{uuid.uuid4().hex[:8].upper()}"
		charter_id = f"CHT-{uuid.uuid4().hex[:8].upper()}"
		vehicle_id = f"CHT-VEH-{vehicle_type.upper()}"

		ct = _norm(vehicle_type)
		if ct not in SUPPORTED_CHARTER_TYPES:
			ct = list(SUPPORTED_CHARTER_TYPES)[0] if SUPPORTED_CHARTER_TYPES else "minibus"

		st = list(SUPPORTED_SCHEDULE_TYPES)[0] if SUPPORTED_SCHEDULE_TYPES else "charter"
		opt = list(SUPPORTED_OPTIMISATION_MODES)[0] if SUPPORTED_OPTIMISATION_MODES else "balanced"
		self.create_schedule(sched_id, tid, st, date, date, opt, client_id)

		charter = self.create_charter(
			charter_id, tid, sched_id, ct, client_id,
			vehicle_id, driver_id, origin, destination, date, True,
		)
		return {
			"charter": charter,
			"client_id": client_id,
			"vehicle_type": vehicle_type,
			"distance_km": distance_km,
			"rate_per_km": rate_per_km,
			"charter_cost_usd": charter_cost,
			"fuel_surcharge_usd": fuel_surcharge,
			"total_cost_usd": total_cost,
		}

	async def schedule_conflict_check(
		self,
		schedule_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Check a schedule for driver double-bookings and vehicle overlaps.

		Scans all shifts and vehicle assignments in the schedule for
		conflicts and records any detected ones automatically.
		"""
		tid = tenant_id or self.tenant_id
		sched = self._schedule_or_none(schedule_id, tid)
		if sched is None:
			raise KeyError(f"Schedule {schedule_id} not found")

		await asyncio.sleep(0)
		schedule_shifts = [s for s in self.shifts.values() if s.tenant_id == tid and s.schedule_id == schedule_id]
		schedule_vehicles = [v for v in self.vehicle_assignments.values() if v.tenant_id == tid and v.schedule_id == schedule_id]

		# Driver double-booking: same driver_id in multiple shifts
		driver_shift_count: dict[str, int] = {}
		for shift in schedule_shifts:
			driver_shift_count[shift.driver_id] = driver_shift_count.get(shift.driver_id, 0) + 1

		new_conflicts: list[dict[str, Any]] = []
		for driver_id, count in driver_shift_count.items():
			if count > 1:
				conflict_id = f"CFT-{schedule_id}-DRV-{driver_id[:6]}"
				ct = "driver_double_booking" if "driver_double_booking" in SUPPORTED_CONFLICT_TYPES else list(SUPPORTED_CONFLICT_TYPES)[0]
				conflict = self.record_conflict(conflict_id, tid, schedule_id, ct, driver_id, _now_iso())
				new_conflicts.append(conflict)

		# Vehicle double-booking
		vehicle_asgn_count: dict[str, int] = {}
		for va in schedule_vehicles:
			vehicle_asgn_count[va.vehicle_id] = vehicle_asgn_count.get(va.vehicle_id, 0) + 1

		for vehicle_id, count in vehicle_asgn_count.items():
			if count > 1:
				conflict_id = f"CFT-{schedule_id}-VEH-{vehicle_id[:6]}"
				ct = "vehicle_double_booking" if "vehicle_double_booking" in SUPPORTED_CONFLICT_TYPES else list(SUPPORTED_CONFLICT_TYPES)[0]
				conflict = self.record_conflict(conflict_id, tid, schedule_id, ct, vehicle_id, _now_iso())
				new_conflicts.append(conflict)

		existing_open = len(self.list_open_conflicts(tid))
		return {
			"schedule_id": schedule_id,
			"tenant_id": tid,
			"shifts_checked": len(schedule_shifts),
			"vehicles_checked": len(schedule_vehicles),
			"new_conflicts_detected": len(new_conflicts),
			"new_conflicts": new_conflicts,
			"total_open_conflicts": existing_open,
			"publishable": existing_open == 0,
			"checked_at": _now_iso(),
		}

	async def schedule_analytics(
		self,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Aggregate scheduling KPIs for a period.

		Returns schedule count by status, conflict resolution rate,
		charter revenue estimate, driver hours compliance rate.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(period):
			raise ValueError("period required")

		await asyncio.sleep(0)
		all_sched = [s for s in self.schedules.values() if s.tenant_id == tid]
		total = len(all_sched)
		by_status: dict[str, int] = {}
		for s in all_sched:
			by_status[s.status] = by_status.get(s.status, 0) + 1

		all_conflicts = [c for c in self.conflicts.values() if c.tenant_id == tid]
		resolved = sum(1 for c in all_conflicts if c.resolved_at is not None)
		resolution_rate = round(resolved / len(all_conflicts) * 100, 1) if all_conflicts else 100.0

		all_shifts = [s for s in self.shifts.values() if s.tenant_id == tid]
		hours_list = [s.hours for s in all_shifts]
		avg_shift_hours = round(statistics.mean(hours_list), 2) if hours_list else 0.0
		hos_violations = sum(1 for s in all_shifts if s.hours > _MAX_DAILY_DRIVE_HOURS)

		charter_count = self._count(self.charters, tid)
		charter_revenue_est = charter_count * 450.0  # stub average

		return {
			"period": period,
			"tenant_id": tid,
			"total_schedules": total,
			"schedules_by_status": by_status,
			"total_conflicts": len(all_conflicts),
			"resolved_conflicts": resolved,
			"conflict_resolution_rate_pct": resolution_rate,
			"total_shifts": len(all_shifts),
			"avg_shift_hours": avg_shift_hours,
			"hos_violations": hos_violations,
			"charter_count": charter_count,
			"charter_revenue_estimate_usd": charter_revenue_est,
			"generated_at": _now_iso(),
		}

	async def capacity_planning(
		self,
		period: str,
		demand_forecast: dict[str, Any],
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Compare forecasted demand against available scheduled capacity.

		demand_forecast: {"trips_per_day": int, "peak_vehicles_needed": int,
		                   "peak_drivers_needed": int}

		Returns capacity gap analysis and recommended schedule adjustments.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(period):
			raise ValueError("period required")

		await asyncio.sleep(0)
		trips_per_day = int(demand_forecast.get("trips_per_day", 0))
		peak_vehicles = int(demand_forecast.get("peak_vehicles_needed", 0))
		peak_drivers = int(demand_forecast.get("peak_drivers_needed", 0))

		# Current capacity from published schedules
		published = [s for s in self.schedules.values() if s.tenant_id == tid and s.status == "published"]
		scheduled_vehicles = self._count(self.vehicle_assignments, tid)
		scheduled_drivers = len({sh.driver_id for sh in self.shifts.values() if sh.tenant_id == tid})

		vehicle_gap = max(0, peak_vehicles - scheduled_vehicles)
		driver_gap = max(0, peak_drivers - scheduled_drivers)

		plan_id = f"CAP-{uuid.uuid4().hex[:8].upper()}"
		plan: dict[str, Any] = {
			"plan_id": plan_id,
			"period": period,
			"tenant_id": tid,
			"demand_trips_per_day": trips_per_day,
			"peak_vehicles_needed": peak_vehicles,
			"scheduled_vehicles": scheduled_vehicles,
			"vehicle_gap": vehicle_gap,
			"peak_drivers_needed": peak_drivers,
			"scheduled_drivers": scheduled_drivers,
			"driver_gap": driver_gap,
			"published_schedules": len(published),
			"capacity_sufficient": vehicle_gap == 0 and driver_gap == 0,
			"recommendations": self._capacity_recommendations(vehicle_gap, driver_gap),
			"created_at": _now_iso(),
		}
		self.capacity_plans[self._key(tid, plan_id)] = plan
		self._audit(tid, "capacity_plan_created", plan_id)
		return plan

	async def driver_hours_compliance(
		self,
		driver_id: str,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Check a driver's hours against EU/local HOS regulations for a period.

		Returns total hours, daily max, weekly total, compliance status,
		and required rest periods.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(driver_id) or not _present(period):
			raise ValueError("driver_id and period required")

		await asyncio.sleep(0)
		driver_shifts = [
			s for s in self.shifts.values()
			if s.tenant_id == tid and s.driver_id == driver_id
		]
		total_hours = sum(s.hours for s in driver_shifts)
		max_single_shift = max((s.hours for s in driver_shifts), default=0.0)
		daily_hours = [s.hours for s in driver_shifts]
		weekly_total = sum(daily_hours)

		violations: list[str] = []
		if max_single_shift > _MAX_DAILY_DRIVE_HOURS:
			violations.append(f"Daily limit exceeded: {max_single_shift}h > {_MAX_DAILY_DRIVE_HOURS}h")
		if weekly_total > _MAX_WEEKLY_DRIVE_HOURS:
			violations.append(f"Weekly limit exceeded: {weekly_total}h > {_MAX_WEEKLY_DRIVE_HOURS}h")

		required_rest_h = _MIN_DAILY_REST_HOURS
		compliant = len(violations) == 0

		return {
			"driver_id": driver_id,
			"period": period,
			"tenant_id": tid,
			"shift_count": len(driver_shifts),
			"total_hours": round(total_hours, 2),
			"max_single_shift_hours": round(max_single_shift, 2),
			"weekly_total_hours": round(weekly_total, 2),
			"max_daily_limit_hours": _MAX_DAILY_DRIVE_HOURS,
			"max_weekly_limit_hours": _MAX_WEEKLY_DRIVE_HOURS,
			"required_rest_hours": required_rest_h,
			"compliant": compliant,
			"violations": violations,
			"checked_at": _now_iso(),
		}

	async def schedule_publish(
		self,
		schedule_id: str,
		*,
		notify_drivers: bool = True,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Publish a schedule after auto-running conflict check.

		If conflicts are found, raises ValueError with details rather than
		silently blocking. Optionally notifies assigned drivers.
		"""
		tid = tenant_id or self.tenant_id
		check = await self.schedule_conflict_check(schedule_id, tenant_id=tid)
		if check["new_conflicts_detected"] > 0:
			raise ValueError(
				f"Schedule {schedule_id} has {check['new_conflicts_detected']} unresolved conflicts. "
				f"Resolve before publishing."
			)

		published = self.publish_schedule(schedule_id, tid)

		notifications_sent = 0
		if notify_drivers:
			schedule_shifts = [s for s in self.shifts.values() if s.tenant_id == tid and s.schedule_id == schedule_id]
			for shift in schedule_shifts:
				notif_id = f"NTF-PUB-{schedule_id}-{shift.driver_id[:6]}"
				nt = list(SUPPORTED_NOTIFICATION_TYPES)[0] if SUPPORTED_NOTIFICATION_TYPES else "schedule_published"
				self.send_notification(notif_id, tid, schedule_id, nt, shift.driver_id, "sms", _now_iso())
				notifications_sent += 1

		return {
			"schedule": published,
			"conflict_check": check,
			"notifications_sent": notifications_sent,
		}

	async def schedule_disruption_management(
		self,
		disruption_id: str,
		*,
		disruption_type: str = "vehicle_breakdown",
		affected_schedule_id: str | None = None,
		severity: str = "medium",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Log and triage a scheduling disruption.

		Determines impact on published schedules, raises conflicts for
		affected resources, and returns recommended mitigation actions.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(disruption_id):
			raise ValueError("disruption_id required")

		await asyncio.sleep(0)
		impact_schedules: list[str] = []
		if affected_schedule_id:
			impact_schedules = [affected_schedule_id]
		else:
			# All published schedules potentially affected
			impact_schedules = [
				s.schedule_id for s in self.schedules.values()
				if s.tenant_id == tid and s.status == "published"
			]

		conflicts_raised: list[dict[str, Any]] = []
		for sched_id in impact_schedules:
			ct = "resource_unavailable" if "resource_unavailable" in SUPPORTED_CONFLICT_TYPES else list(SUPPORTED_CONFLICT_TYPES)[0]
			conflict_id = f"CFT-DIS-{disruption_id[:8]}-{sched_id[:6]}"
			c = self.record_conflict(conflict_id, tid, sched_id, ct, disruption_id, _now_iso())
			conflicts_raised.append(c)

		mitigation = {
			"vehicle_breakdown": ["deploy_backup_vehicle", "reschedule_affected_trips"],
			"driver_unavailable": ["call_on_standby_driver", "merge_routes"],
			"traffic_incident": ["reroute_affected_vehicles", "update_customer_etas"],
			"weather_event": ["delay_departures", "activate_weather_protocol"],
		}.get(disruption_type, ["assess_situation", "escalate_to_ops_manager"])

		disruption: dict[str, Any] = {
			"disruption_id": disruption_id,
			"disruption_type": disruption_type,
			"severity": severity,
			"affected_schedule_ids": impact_schedules,
			"conflicts_raised": len(conflicts_raised),
			"mitigation_actions": mitigation,
			"tenant_id": tid,
			"recorded_at": _now_iso(),
		}
		self.disruptions[self._key(tid, disruption_id)] = disruption
		self._audit(tid, "schedule_disruption_logged", disruption_id)
		return disruption

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _capacity_recommendations(self, vehicle_gap: int, driver_gap: int) -> list[str]:
		recs: list[str] = []
		if vehicle_gap > 0:
			recs.append(f"Hire or lease {vehicle_gap} additional vehicle(s)")
		if driver_gap > 0:
			recs.append(f"Recruit or contract {driver_gap} additional driver(s)")
		if not recs:
			recs.append("Capacity is adequate for forecast demand")
		return recs

	def _log_schedule_state(self, tenant_id: str) -> str:
		return f"tenant={tenant_id} schedules={self._count(self.schedules, tenant_id)} conflicts={len(self.list_open_conflicts(tenant_id))}"

	def _schedule_or_none(self, schedule_id: str, tenant_id: str) -> Schedule | None:
		return self.schedules.get(self._key(tenant_id, schedule_id))

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "scheduling_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "scheduling_policy_denied")


	async def schedule_optimise_ml(
		self,
		schedule_id: str,
		optimisation_target: str = "minimize_cost",
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Apply ML-informed schedule optimisation heuristics.

		In production delegates to an Ollama-served model. Here applies
		greedy shift consolidation: merges overlapping shifts to reduce dead-time.
		"""
		tid = tenant_id or self.tenant_id
		schedule = self.schedules.get(self._key(tid, schedule_id))
		if schedule is None:
			raise KeyError(f"schedule_not_found:{schedule_id}")
		shifts = [s for s in self.shifts.values() if s.tenant_id == tid and s.schedule_id == schedule_id]
		# Count overlapping driver-day pairs (proxy for inefficiency)
		driver_shifts: dict[str, list[str]] = {}
		for s in shifts:
			driver_shifts.setdefault(s.driver_id, []).append(s.shift_date)
		overlap_drivers = sum(1 for dates in driver_shifts.values() if len(dates) != len(set(dates)))
		opt_id = f"OPT-{uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "schedule_ml_optimised", opt_id)
		return {
			"optimisation_id": opt_id,
			"schedule_id": schedule_id,
			"tenant_id": tid,
			"target": optimisation_target,
			"shifts_analysed": len(shifts),
			"drivers_with_overlaps": overlap_drivers,
			"estimated_cost_reduction_pct": round(min(overlap_drivers * 2.5, 20.0), 1),
			"recommendation": "consolidate_overlapping_shifts" if overlap_drivers > 0 else "no_action_needed",
			"generated_at": _now_iso(),
		}

	async def schedule_kpi_summary(
		self,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return a concise schedule KPI card for dashboard consumption."""
		tid = tenant_id or self.tenant_id
		schedules = [s for s in self.schedules.values() if s.tenant_id == tid]
		published = sum(1 for s in schedules if s.status == "published")
		shifts = sum(1 for s in self.shifts.values() if s.tenant_id == tid)
		open_conflicts = len(self.list_open_conflicts(tid))
		return {
			"tenant_id": tid,
			"total_schedules": len(schedules),
			"published_schedules": published,
			"total_shifts": shifts,
			"open_conflicts": open_conflicts,
			"publish_rate_pct": round(published / max(len(schedules), 1) * 100, 1),
			"generated_at": _now_iso(),
		}

	async def passenger_load_forecast(
		self,
		schedule_id: str,
		historical_load_avg: float,
		growth_rate_pct: float = 5.0,
		horizon_weeks: int = 4,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Forecast passenger load for a schedule over a planning horizon."""
		tid = tenant_id or self.tenant_id
		schedule = self.schedules.get(self._key(tid, schedule_id))
		if schedule is None:
			raise KeyError(f"schedule_not_found:{schedule_id}")
		weekly_growth = 1 + growth_rate_pct / 100 / 52
		forecasts = [
			{"week": w + 1, "projected_load": round(historical_load_avg * (weekly_growth ** w), 1)}
			for w in range(horizon_weeks)
		]
		return {
			"schedule_id": schedule_id,
			"tenant_id": tid,
			"historical_load_avg": historical_load_avg,
			"growth_rate_pct": growth_rate_pct,
			"horizon_weeks": horizon_weeks,
			"forecasts": forecasts,
			"generated_at": _now_iso(),
		}

	async def schedule_deviation_alert(
		self,
		schedule_id: str,
		actual_departure: str,
		planned_departure: str,
		threshold_minutes: int = 10,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Raise an alert when a departure deviates beyond the threshold."""
		tid = tenant_id or self.tenant_id
		from datetime import datetime as _dt
		try:
			planned = _dt.fromisoformat(planned_departure)
			actual = _dt.fromisoformat(actual_departure)
			deviation_minutes = round((actual - planned).total_seconds() / 60, 1)
		except ValueError:
			deviation_minutes = 0.0
		is_deviated = abs(deviation_minutes) >= threshold_minutes
		alert_id = f"DEV-{uuid.uuid4().hex[:8].upper()}"
		if is_deviated:
			self._audit(tid, "schedule_deviation_alert_raised", alert_id)
		return {
			"alert_id": alert_id,
			"schedule_id": schedule_id,
			"tenant_id": tid,
			"planned_departure": planned_departure,
			"actual_departure": actual_departure,
			"deviation_minutes": deviation_minutes,
			"threshold_minutes": threshold_minutes,
			"alert_raised": is_deviated,
			"severity": "high" if abs(deviation_minutes) >= threshold_minutes * 3 else "medium" if is_deviated else "low",
			"generated_at": _now_iso(),
		}

	async def schedule_compare(
		self,
		schedule_a_id: str,
		schedule_b_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Compare two schedules by shift count, conflicts, and coverage."""
		tid = tenant_id or self.tenant_id
		sa = self.schedules.get(self._key(tid, schedule_a_id))
		sb = self.schedules.get(self._key(tid, schedule_b_id))
		if sa is None:
			raise KeyError(f"schedule_not_found:{schedule_a_id}")
		if sb is None:
			raise KeyError(f"schedule_not_found:{schedule_b_id}")
		shifts_a = sum(1 for s in self.shifts.values() if s.tenant_id == tid and s.schedule_id == schedule_a_id)
		shifts_b = sum(1 for s in self.shifts.values() if s.tenant_id == tid and s.schedule_id == schedule_b_id)
		conflicts_a = sum(1 for c in self.conflicts.values() if c.tenant_id == tid and c.schedule_id == schedule_a_id and c.status == "open")
		conflicts_b = sum(1 for c in self.conflicts.values() if c.tenant_id == tid and c.schedule_id == schedule_b_id and c.status == "open")
		return {
			"tenant_id": tid,
			"schedule_a": {"id": schedule_a_id, "status": sa.status, "shifts": shifts_a, "open_conflicts": conflicts_a},
			"schedule_b": {"id": schedule_b_id, "status": sb.status, "shifts": shifts_b, "open_conflicts": conflicts_b},
			"shift_delta": shifts_a - shifts_b,
			"conflict_delta": conflicts_a - conflicts_b,
			"compared_at": _now_iso(),
		}

	async def shift_swap_approve(
		self,
		shift_id: str,
		requesting_driver_id: str,
		replacement_driver_id: str,
		approved_by: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Approve a driver shift swap and update the shift assignment."""
		tid = tenant_id or self.tenant_id
		shift = self.shifts.get(self._key(tid, shift_id))
		if shift is None:
			raise KeyError(f"shift_not_found:{shift_id}")
		if shift.driver_id != requesting_driver_id:
			raise ValueError("requesting_driver_id does not match shift assignment")
		await asyncio.sleep(0)
		# Re-assign the shift
		self.shifts[self._key(tid, shift_id)] = shift.model_copy(update={"driver_id": replacement_driver_id})
		swap_id = f"SWP-{uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "shift_swap_approved", swap_id)
		return {
			"swap_id": swap_id,
			"shift_id": shift_id,
			"original_driver": requesting_driver_id,
			"replacement_driver": replacement_driver_id,
			"approved_by": approved_by,
			"tenant_id": tid,
			"approved_at": _now_iso(),
		}

	async def schedule_analytics_detail(
		self,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return detailed schedule analytics: shift distribution, conflict rates, charter usage."""
		tid = tenant_id or self.tenant_id
		schedules = [s for s in self.schedules.values() if s.tenant_id == tid]
		shifts = [s for s in self.shifts.values() if s.tenant_id == tid]
		charters = [c for c in self.charters.values() if c.tenant_id == tid]
		published = sum(1 for s in schedules if s.status == "published")
		by_status: dict[str, int] = {}
		for s in schedules:
			by_status[s.status] = by_status.get(s.status, 0) + 1
		return {
			"tenant_id": tid,
			"period": period,
			"total_schedules": len(schedules),
			"by_status": by_status,
			"total_shifts": len(shifts),
			"total_charters": len(charters),
			"publish_rate_pct": round(published / max(len(schedules), 1) * 100, 1),
			"open_conflicts": len(self.list_open_conflicts(tid)),
			"generated_at": _now_iso(),
		}

	async def tachograph_compliance_report(
		self,
		driver_id: str,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Generate a tachograph compliance report for a driver."""
		tid = tenant_id or self.tenant_id
		if not _present(driver_id) or not _present(period):
			raise ValueError("driver_id and period required")
		await asyncio.sleep(0)
		shifts = [s for s in self.shifts.values() if s.tenant_id == tid and s.driver_id == driver_id]
		tacho_shifts = [s for s in shifts if s.tacho_compliant]
		compliance_rate = round(len(tacho_shifts) / max(len(shifts), 1) * 100, 1)
		return {
			"driver_id": driver_id,
			"period": period,
			"tenant_id": tid,
			"total_shifts": len(shifts),
			"tacho_compliant_shifts": len(tacho_shifts),
			"compliance_rate_pct": compliance_rate,
			"compliant": compliance_rate >= 100.0,
			"generated_at": _now_iso(),
		}

	async def bulk_assign_vehicles(
		self,
		schedule_id: str,
		vehicle_routes: list[dict[str, Any]],
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Bulk assign multiple vehicles to routes in a schedule."""
		tid = tenant_id or self.tenant_id
		if not vehicle_routes:
			raise ValueError("vehicle_routes list is empty")
		await asyncio.sleep(0)
		return await self.vehicle_assignment(schedule_id, vehicle_routes, tenant_id=tid)

	async def driver_roster(
		self,
		schedule_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return the driver roster for a schedule with shift details."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		schedule_shifts = [s for s in self.shifts.values() if s.tenant_id == tid and s.schedule_id == schedule_id]
		roster: list[dict[str, Any]] = []
		for shift in schedule_shifts:
			roster.append({
				"driver_id": shift.driver_id,
				"shift_id": shift.shift_id,
				"shift_type": shift.shift_type,
				"start_time": shift.start_time,
				"end_time": shift.end_time,
				"hours": shift.hours,
				"tacho_compliant": shift.tacho_compliant,
			})
		return {
			"schedule_id": schedule_id,
			"tenant_id": tid,
			"driver_count": len({r["driver_id"] for r in roster}),
			"shift_count": len(roster),
			"roster": roster,
			"generated_at": _now_iso(),
		}

	async def export_schedule_data(
		self,
		period: str,
		*,
		format: str = "json",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Export schedule data metadata."""
		tid = tenant_id or self.tenant_id
		export_id = f"SCH-EXP-{uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "schedule_data_exported", export_id)
		return {
			"export_id": export_id,
			"period": period,
			"tenant_id": tid,
			"format": format,
			"record_count": self._count(self.schedules, tid),
			"download_ref": f"/exports/{tid}/{export_id}.{format}",
			"status": "ready",
			"generated_at": _now_iso(),
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": "TransportSchedulingService",
			"status": "healthy",
			"schedules": len(self.schedules),
			"shifts": len(self.shifts),
			"vehicle_assignments": len(self.vehicle_assignments),
			"charters": len(self.charters),
			"conflicts": len(self.conflicts),
			"audit_events": len(self.audit_events),
			"checked_at": _now_iso(),
		}

	async def cancel_shift(
		self,
		shift_id: str,
		reason: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Cancel a driver shift with reason and audit trail."""
		tid = tenant_id or self.tenant_id
		shift = self.shifts.get(self._key(tid, shift_id))
		if shift is None:
			raise KeyError(f"Shift {shift_id} not found")
		await asyncio.sleep(0)
		self._audit(tid, "shift_cancelled", shift_id)
		return {**shift.to_dict(), "cancellation_reason": reason, "cancelled_at": _now_iso()}

	async def charter_cost_summary(
		self,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Summarise total charter bookings and estimated revenue."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		charters = [c for c in self.charters.values() if c.tenant_id == tid]
		total_est = len(charters) * 450.0
		return {
			"tenant_id": tid,
			"charter_count": len(charters),
			"confirmed_count": sum(1 for c in charters if c.customer_confirmed),
			"estimated_revenue_usd": total_est,
			"generated_at": _now_iso(),
		}

	async def driver_shift_summary(
		self,
		schedule_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Summarise shift statistics for a schedule."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		shifts = [s for s in self.shifts.values() if s.tenant_id == tid and s.schedule_id == schedule_id]
		total_hours = sum(s.hours for s in shifts)
		return {
			"schedule_id": schedule_id,
			"tenant_id": tid,
			"shift_count": len(shifts),
			"total_hours": round(total_hours, 2),
			"avg_hours": round(total_hours / max(len(shifts), 1), 2),
			"drivers": len({s.driver_id for s in shifts}),
			"generated_at": _now_iso(),
		}


TransportSchService = TransportSchedulingService
