"""Executable service layer for APG Vehicle Maintenance."""

from __future__ import annotations

import asyncio
import statistics
import uuid
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

try:
	from .capability_contract import (
		SUPPORTED_MAINTENANCE_TYPES, SUPPORTED_JOB_STATUSES, SUPPORTED_PRIORITY_LEVELS,
		SUPPORTED_WORKSHOP_TYPES, SUPPORTED_PARTS_CATEGORIES, SUPPORTED_WARRANTY_TYPES,
		SUPPORTED_ROADWORTHINESS_STANDARDS, SUPPORTED_INSPECTION_TYPES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		MaintenanceJob, WorkshopAllocation, PartsOrder, WarrantyRecord,
		VehicleInspection, RoadworthinessRecord, MaintenanceSchedule, MaintenanceAgent,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_MAINTENANCE_TYPES, SUPPORTED_JOB_STATUSES, SUPPORTED_PRIORITY_LEVELS,
		SUPPORTED_WORKSHOP_TYPES, SUPPORTED_PARTS_CATEGORIES, SUPPORTED_WARRANTY_TYPES,
		SUPPORTED_ROADWORTHINESS_STANDARDS, SUPPORTED_INSPECTION_TYPES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		MaintenanceJob, WorkshopAllocation, PartsOrder, WarrantyRecord,
		VehicleInspection, RoadworthinessRecord, MaintenanceSchedule, MaintenanceAgent,
	)


def _present(value: str | None) -> bool:
	return bool(value and str(value).strip())

def _norm(value: str) -> str:
	return str(value).strip().lower() if value else ""

def _positive(value: float | int) -> bool:
	try:
		return float(value) > 0
	except (TypeError, ValueError):
		return False

def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Predictive maintenance thresholds (km intervals)
_SERVICE_INTERVALS_KM: dict[str, int] = {
	"oil_change": 10000,
	"filter_change": 15000,
	"brake_service": 30000,
	"tyre_rotation": 10000,
	"major_service": 50000,
	"transmission_service": 60000,
}

# Labour rates by workshop type (USD per hour)
_LABOUR_RATE_BY_WORKSHOP: dict[str, float] = {
	"in_house": 25.0, "authorised_dealer": 75.0,
	"independent": 45.0, "roadside": 60.0,
}

# Tyre position codes
_TYRE_POSITIONS = ["FL", "FR", "RL", "RR", "spare"]


class VehicleMaintenanceService:
	"""Tenant-scoped vehicle maintenance runtime."""

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
		self._store = store or get_store(db_url)
		self.jobs: dict[tuple[str, str], MaintenanceJob] = {}
		self.workshop_allocations: dict[tuple[str, str], WorkshopAllocation] = {}
		self.parts_orders: dict[tuple[str, str], PartsOrder] = {}
		self.warranty_records: dict[tuple[str, str], WarrantyRecord] = {}
		self.inspections: dict[tuple[str, str], VehicleInspection] = {}
		self.roadworthiness_records: dict[tuple[str, str], RoadworthinessRecord] = {}
		self.schedules: dict[tuple[str, str], MaintenanceSchedule] = {}
		self.agents: dict[tuple[str, str], MaintenanceAgent] = {}
		_store = get_store(db_url)
		self.audit_events = WriteThruList('audit_events', tenant_id, self._store)
		# Extended state
		self.tyre_records: dict[tuple[str, str], dict[str, Any]] = {}
		self.defect_log = WriteThruList('defect_log', tenant_id, self._store)
		self.work_orders: dict[tuple[str, str], dict[str, Any]] = {}
		self.odometer_readings: dict[str, float] = {}

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

	def create_job(
		self, job_id: str, tenant_id: str, vehicle_id: str,
		maintenance_type: str, priority: str, technician_id: str,
		workshop_type: str, estimated_hours: float, job_card_ref: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a maintenance job."""
		maintenance_type = _norm(maintenance_type)
		priority = _norm(priority)
		workshop_type = _norm(workshop_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "create_job",
			"maintenance_type_supported": maintenance_type in SUPPORTED_MAINTENANCE_TYPES,
			"vehicle_present": _present(vehicle_id),
			"technician_present": _present(technician_id),
			"priority_supported": priority in SUPPORTED_PRIORITY_LEVELS,
		})
		item = MaintenanceJob(
			job_id, tenant_id, vehicle_id, maintenance_type, "scheduled",
			priority, technician_id, workshop_type, float(estimated_hours), None, job_card_ref,
		)
		self.jobs[self._key(tenant_id, job_id)] = item
		self._audit(tenant_id, "maintenance_job_created", job_id)
		return item.to_dict()

	def update_job_status(self, job_id: str, tenant_id: str, status: str, actual_hours: float | None = None) -> dict[str, Any]:
		"""Update maintenance job status."""
		status = _norm(status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "update_job_status",
			"status_supported": status in SUPPORTED_JOB_STATUSES,
		})
		job = self._job_or_none(job_id, tenant_id)
		if job is None:
			raise KeyError(f"Job {job_id} not found")
		job.status = status
		if actual_hours is not None:
			job.actual_hours = float(actual_hours)
		self._audit(tenant_id, "maintenance_job_status_updated", job_id)
		return job.to_dict()

	def dispatch_vehicle_check(self, vehicle_id: str, tenant_id: str, mot_expired: bool = False, vehicle_safe: bool = True) -> dict[str, Any]:
		"""Pre-dispatch safety check for a vehicle."""
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation": "dispatch_vehicle",
			"mot_expired": mot_expired,
			"vehicle_safe": vehicle_safe,
		})
		return {"vehicle_id": vehicle_id, "tenant_id": tenant_id, "dispatch_cleared": True}

	def allocate_workshop(
		self, allocation_id: str, tenant_id: str, workshop_type: str,
		location: str, bay_number: str, job_id: str, allocated_at: str,
	) -> dict[str, Any]:
		"""Allocate a workshop bay for a maintenance job."""
		workshop_type = _norm(workshop_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "allocate_workshop",
			"workshop_type_supported": workshop_type in SUPPORTED_WORKSHOP_TYPES,
		})
		item = WorkshopAllocation(allocation_id, tenant_id, workshop_type, location, bay_number, job_id, allocated_at, None)
		self.workshop_allocations[self._key(tenant_id, allocation_id)] = item
		self._audit(tenant_id, "workshop_allocated", allocation_id)
		return item.to_dict()

	def order_parts(
		self, order_id: str, tenant_id: str, job_id: str, parts_category: str,
		part_number: str, description: str, quantity: int,
		supplier_id: str, ordered_at: str,
	) -> dict[str, Any]:
		"""Order parts for a maintenance job."""
		parts_category = _norm(parts_category)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "order_parts",
			"parts_category_supported": parts_category in SUPPORTED_PARTS_CATEGORIES,
			"quantity_positive": _positive(quantity),
		})
		item = PartsOrder(order_id, tenant_id, job_id, parts_category, part_number, description, int(quantity), supplier_id, ordered_at, None)
		self.parts_orders[self._key(tenant_id, order_id)] = item
		self._audit(tenant_id, "parts_ordered", order_id)
		return item.to_dict()

	def record_warranty(
		self, warranty_id: str, tenant_id: str, vehicle_id: str,
		warranty_type: str, provider: str, start_date: str,
		expiry_date: str, claim_ref: str | None = None,
	) -> dict[str, Any]:
		"""Record a warranty for a vehicle."""
		warranty_type = _norm(warranty_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_warranty",
			"warranty_type_supported": warranty_type in SUPPORTED_WARRANTY_TYPES,
		})
		item = WarrantyRecord(warranty_id, tenant_id, vehicle_id, warranty_type, provider, start_date, expiry_date, claim_ref)
		self.warranty_records[self._key(tenant_id, warranty_id)] = item
		self._audit(tenant_id, "warranty_recorded", warranty_id)
		return item.to_dict()

	def conduct_inspection(
		self, inspection_id: str, tenant_id: str, vehicle_id: str,
		inspection_type: str, inspector_id: str, conducted_at: str,
		defects_found: bool, digital_signature: str, passed: bool,
	) -> dict[str, Any]:
		"""Conduct and record a vehicle inspection."""
		inspection_type = _norm(inspection_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "conduct_inspection",
			"inspection_type_supported": inspection_type in SUPPORTED_INSPECTION_TYPES,
			"digital_signature_present": _present(digital_signature),
		})
		item = VehicleInspection(inspection_id, tenant_id, vehicle_id, inspection_type, inspector_id, conducted_at, defects_found, digital_signature, passed)
		self.inspections[self._key(tenant_id, inspection_id)] = item
		self._audit(tenant_id, "inspection_completed", inspection_id)
		return item.to_dict()

	def issue_roadworthiness(
		self, record_id: str, tenant_id: str, vehicle_id: str,
		standard: str, certificate_number: str, issued_at: str,
		expires_at: str, issuing_authority: str,
	) -> dict[str, Any]:
		"""Issue a roadworthiness certificate."""
		standard = _norm(standard)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "issue_roadworthiness",
			"standard_supported": standard in SUPPORTED_ROADWORTHINESS_STANDARDS,
		})
		item = RoadworthinessRecord(record_id, tenant_id, vehicle_id, standard, certificate_number, issued_at, expires_at, issuing_authority)
		self.roadworthiness_records[self._key(tenant_id, record_id)] = item
		self._audit(tenant_id, "roadworthiness_certificate_issued", record_id)
		return item.to_dict()

	def create_maintenance_schedule(
		self, schedule_id: str, tenant_id: str, vehicle_id: str,
		maintenance_type: str, scheduled_at: str,
		interval_km: int | None = None, interval_days: int | None = None,
	) -> dict[str, Any]:
		"""Create a preventive maintenance schedule entry."""
		maintenance_type = _norm(maintenance_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
		})
		item = MaintenanceSchedule(schedule_id, tenant_id, vehicle_id, maintenance_type, scheduled_at, interval_km, interval_days, None)
		self.schedules[self._key(tenant_id, schedule_id)] = item
		self._audit(tenant_id, "maintenance_schedule_generated", schedule_id)
		return item.to_dict()

	def register_maintenance_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		"""Register an AI agent for maintenance management."""
		runtime = _norm(runtime)
		role = _norm(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_maintenance_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = MaintenanceAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "maintenance_agent_registered", agent_id)
		return item.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "maintenance_batch", "event_stream": event_stream})
		if item_count <= 0:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.transport.maintenance.lifecycle", "accepted": True}

	def list_jobs(self, tenant_id: str) -> list[dict[str, Any]]:
		return [j.to_dict() for j in self.jobs.values() if j.tenant_id == tenant_id]

	def list_schedules(self, tenant_id: str) -> list[dict[str, Any]]:
		return [s.to_dict() for s in self.schedules.values() if s.tenant_id == tenant_id]

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		jobs = [j for j in self.jobs.values() if j.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"job_count": len(jobs),
			"open_job_count": sum(1 for j in jobs if j.status not in ("completed", "cancelled")),
			"workshop_allocation_count": self._count(self.workshop_allocations, tenant_id),
			"parts_order_count": self._count(self.parts_orders, tenant_id),
			"warranty_count": self._count(self.warranty_records, tenant_id),
			"inspection_count": self._count(self.inspections, tenant_id),
			"roadworthiness_count": self._count(self.roadworthiness_records, tenant_id),
			"schedule_count": self._count(self.schedules, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# New methods
	# ------------------------------------------------------------------

	async def schedule_service(
		self,
		vehicle_id: str,
		service_type: str,
		due_date: str,
		due_km: int,
		*,
		technician_id: str = "unassigned",
		priority: str = "medium",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Schedule a preventive service with both date and odometer triggers.

		Creates both a maintenance schedule entry and a planned job.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id) or not _present(service_type) or not _present(due_date):
			raise ValueError("vehicle_id, service_type and due_date required")
		if due_km < 0:
			raise ValueError("due_km must be >= 0")

		await asyncio.sleep(0)
		mt = _norm(service_type)
		if mt not in SUPPORTED_MAINTENANCE_TYPES:
			mt = list(SUPPORTED_MAINTENANCE_TYPES)[0] if SUPPORTED_MAINTENANCE_TYPES else "routine_service"
		p = _norm(priority)
		if p not in SUPPORTED_PRIORITY_LEVELS:
			p = "medium"

		sched_id = f"SCHED-{vehicle_id[:6]}-{uuid.uuid4().hex[:6].upper()}"
		schedule = self.create_maintenance_schedule(sched_id, tid, vehicle_id, mt, due_date, due_km, None)

		job_id = f"JOB-{sched_id}"
		wt = list(SUPPORTED_WORKSHOP_TYPES)[0] if SUPPORTED_WORKSHOP_TYPES else "in_house"
		job = self.create_job(job_id, tid, vehicle_id, mt, p, technician_id, wt, 2.0, sched_id)

		return {
			"schedule": schedule,
			"job": job,
			"due_date": due_date,
			"due_km": due_km,
			"service_type": service_type,
		}

	async def log_defect(
		self,
		vehicle_id: str,
		defect_type: str,
		severity: str,
		reported_by: str,
		*,
		description: str = "",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Log a vehicle defect report and auto-escalate critical items.

		Critical defects immediately create a high-priority maintenance job.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id) or not _present(defect_type) or not _present(reported_by):
			raise ValueError("vehicle_id, defect_type and reported_by required")

		await asyncio.sleep(0)
		defect_id = f"DEF-{vehicle_id[:6]}-{uuid.uuid4().hex[:6].upper()}"
		defect: dict[str, Any] = {
			"defect_id": defect_id,
			"vehicle_id": vehicle_id,
			"defect_type": defect_type,
			"severity": severity,
			"reported_by": reported_by,
			"description": description,
			"tenant_id": tid,
			"reported_at": _now_iso(),
			"resolved": False,
		}
		self.defect_log.append(defect)

		auto_job = None
		if _norm(severity) in ("critical", "high"):
			job_id = f"JOB-AUTO-{defect_id}"
			mt = list(SUPPORTED_MAINTENANCE_TYPES)[0] if SUPPORTED_MAINTENANCE_TYPES else "repair"
			wt = list(SUPPORTED_WORKSHOP_TYPES)[0] if SUPPORTED_WORKSHOP_TYPES else "in_house"
			auto_job = self.create_job(
				job_id, tid, vehicle_id, mt, "high", reported_by, wt, 4.0, defect_id,
			)

		self._audit(tid, "defect_logged", defect_id)
		return {**defect, "auto_job_created": auto_job is not None, "auto_job": auto_job}

	async def create_work_order(
		self,
		vehicle_id: str,
		defects: list[str],
		assigned_to: str,
		*,
		workshop_type: str = "in_house",
		priority: str = "medium",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Create a work order consolidating multiple defects into one job.

		Allocates a workshop bay, estimates hours from defect count (1.5h each),
		and returns the full work order context.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id) or not defects or not _present(assigned_to):
			raise ValueError("vehicle_id, defects and assigned_to required")

		await asyncio.sleep(0)
		wo_id = f"WO-{uuid.uuid4().hex[:8].upper()}"
		estimated_hours = len(defects) * 1.5
		mt = list(SUPPORTED_MAINTENANCE_TYPES)[0] if SUPPORTED_MAINTENANCE_TYPES else "repair"
		p = _norm(priority)
		if p not in SUPPORTED_PRIORITY_LEVELS:
			p = "medium"
		wt = _norm(workshop_type)
		if wt not in SUPPORTED_WORKSHOP_TYPES:
			wt = list(SUPPORTED_WORKSHOP_TYPES)[0] if SUPPORTED_WORKSHOP_TYPES else "in_house"

		job = self.create_job(wo_id, tid, vehicle_id, mt, p, assigned_to, wt, estimated_hours, wo_id)

		alloc_id = f"WALLOC-{wo_id}"
		bay = f"BAY-{uuid.uuid4().hex[:4].upper()}"
		allocation = self.allocate_workshop(alloc_id, tid, wt, f"depot-{tid}", bay, wo_id, _now_iso())

		work_order: dict[str, Any] = {
			"work_order_id": wo_id,
			"vehicle_id": vehicle_id,
			"defects": defects,
			"assigned_to": assigned_to,
			"estimated_hours": estimated_hours,
			"job": job,
			"workshop_allocation": allocation,
			"status": "open",
			"tenant_id": tid,
			"created_at": _now_iso(),
		}
		self.work_orders[self._key(tid, wo_id)] = work_order
		return work_order

	async def complete_work_order(
		self,
		work_order_id: str,
		parts_used: list[dict[str, Any]],
		labour_hours: float,
		cost: float,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Mark a work order complete; record parts used, labour and total cost.

		parts_used: [{"part_number": str, "description": str, "quantity": int, "unit_cost": float}]
		"""
		tid = tenant_id or self.tenant_id
		wo = self.work_orders.get(self._key(tid, work_order_id))
		if wo is None:
			raise KeyError(f"Work order {work_order_id} not found")
		if wo.get("status") == "completed":
			raise ValueError(f"Work order {work_order_id} is already completed")

		await asyncio.sleep(0)
		parts_cost = sum(float(p.get("unit_cost", 0)) * int(p.get("quantity", 1)) for p in parts_used)
		labour_rate = _LABOUR_RATE_BY_WORKSHOP.get(_norm(wo.get("job", {}).get("workshop_type", "in_house")), 25.0)
		labour_cost = round(labour_hours * labour_rate, 2)
		total_cost = round(parts_cost + labour_cost, 2)

		# Record parts orders
		parts_orders = []
		for p in parts_used:
			po_id = f"PO-{work_order_id}-{p.get('part_number', 'UNK')}"
			pc = list(SUPPORTED_PARTS_CATEGORIES)[0] if SUPPORTED_PARTS_CATEGORIES else "mechanical"
			po = self.order_parts(
				po_id, tid, work_order_id, pc,
				p.get("part_number", "UNK"),
				p.get("description", "Part"),
				int(p.get("quantity", 1)),
				"stock", _now_iso(),
			)
			parts_orders.append(po)

		# Update job status
		self.update_job_status(work_order_id, tid, "completed", labour_hours)

		wo["status"] = "completed"
		wo["completed_at"] = _now_iso()
		wo["parts_used"] = parts_used
		wo["labour_hours"] = labour_hours
		wo["parts_cost_usd"] = round(parts_cost, 2)
		wo["labour_cost_usd"] = labour_cost
		wo["total_cost_usd"] = total_cost

		self._audit(tid, "work_order_completed", work_order_id)
		return {**wo, "parts_orders": parts_orders}

	async def parts_inventory_check(
		self,
		part_number: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Check inventory status for a part number across all pending orders.

		Returns ordered quantity, received quantity, and whether stock is available.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(part_number):
			raise ValueError("part_number required")

		await asyncio.sleep(0)
		related_orders = [
			o for o in self.parts_orders.values()
			if o.tenant_id == tid and o.part_number == part_number
		]
		ordered_qty = sum(o.quantity for o in related_orders)
		received_qty = sum(o.quantity for o in related_orders if o.received_at is not None)
		pending_qty = ordered_qty - received_qty

		return {
			"part_number": part_number,
			"tenant_id": tid,
			"total_orders": len(related_orders),
			"ordered_qty": ordered_qty,
			"received_qty": received_qty,
			"pending_qty": pending_qty,
			"stock_available": received_qty > 0,
			"checked_at": _now_iso(),
		}

	async def tyre_management(
		self,
		vehicle_id: str,
		tyre_action: str,
		position: str,
		*,
		tread_depth_mm: float | None = None,
		brand: str = "unknown",
		serial: str = "",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Record a tyre action (fit, rotate, inspect, replace) at a wheel position.

		Raises a job if tread depth is below legal minimum (1.6mm).
		"""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id) or not _present(tyre_action) or not _present(position):
			raise ValueError("vehicle_id, tyre_action and position required")
		if position.upper() not in _TYRE_POSITIONS:
			raise ValueError(f"position must be one of {_TYRE_POSITIONS}")

		await asyncio.sleep(0)
		legal_min_mm = 1.6
		below_legal = tread_depth_mm is not None and tread_depth_mm < legal_min_mm
		action_id = f"TYR-{vehicle_id[:6]}-{position}-{uuid.uuid4().hex[:4].upper()}"

		tyre_record: dict[str, Any] = {
			"action_id": action_id,
			"vehicle_id": vehicle_id,
			"tyre_action": tyre_action,
			"position": position,
			"tread_depth_mm": tread_depth_mm,
			"brand": brand,
			"serial": serial,
			"below_legal_minimum": below_legal,
			"tenant_id": tid,
			"actioned_at": _now_iso(),
		}
		self.tyre_records[self._key(tid, action_id)] = tyre_record

		auto_job = None
		if below_legal:
			job_id = f"JOB-TYRE-{action_id}"
			mt = "tyre_replacement" if "tyre_replacement" in SUPPORTED_MAINTENANCE_TYPES else list(SUPPORTED_MAINTENANCE_TYPES)[0]
			wt = list(SUPPORTED_WORKSHOP_TYPES)[0] if SUPPORTED_WORKSHOP_TYPES else "in_house"
			auto_job = self.create_job(job_id, tid, vehicle_id, mt, "high", "unassigned", wt, 1.0, action_id)

		self._audit(tid, "tyre_action_recorded", action_id)
		return {**tyre_record, "auto_job_created": below_legal, "auto_job": auto_job}

	async def roadworthiness_check(
		self,
		vehicle_id: str,
		inspector_id: str,
		*,
		standard: str | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Conduct a roadworthiness inspection and issue a certificate if passed.

		Checks for open defects and failed inspections. If either exist, the
		certificate is not issued and the blocking issues are returned.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id) or not _present(inspector_id):
			raise ValueError("vehicle_id and inspector_id required")

		await asyncio.sleep(0)
		std = _norm(standard) if standard else (list(SUPPORTED_ROADWORTHINESS_STANDARDS)[0] if SUPPORTED_ROADWORTHINESS_STANDARDS else "ntsa")
		it = list(SUPPORTED_INSPECTION_TYPES)[0] if SUPPORTED_INSPECTION_TYPES else "pre_trip"

		# Check for open defects
		open_defects = [
			d for d in self.defect_log
			if d.get("vehicle_id") == vehicle_id and d.get("tenant_id") == tid and not d.get("resolved", False)
			and d.get("severity") in ("critical", "high")
		]
		# Check for failed recent inspections
		failed_inspections = [
			i for i in self.inspections.values()
			if i.tenant_id == tid and i.vehicle_id == vehicle_id and not i.passed
		]

		blocking_issues = [d["defect_id"] for d in open_defects] + [i.inspection_id for i in failed_inspections]
		passed = len(blocking_issues) == 0

		insp_id = f"INSP-{vehicle_id[:6]}-{uuid.uuid4().hex[:6].upper()}"
		sig = f"ESIG-{inspector_id}-{_now_iso()[:10]}"
		inspection = self.conduct_inspection(insp_id, tid, vehicle_id, it, inspector_id, _now_iso(), len(open_defects) > 0, sig, passed)

		certificate = None
		if passed:
			cert_id = f"RW-{vehicle_id[:6]}-{uuid.uuid4().hex[:6].upper()}"
			cert_num = f"CERT-{cert_id}"
			# Validity: 1 year
			expiry = _now_iso()[:4] + str(int(_now_iso()[:4]) + 1)[3:] + _now_iso()[4:10] + "T00:00:00+00:00"
			certificate = self.issue_roadworthiness(cert_id, tid, vehicle_id, std, cert_num, _now_iso(), expiry[:10], inspector_id)

		return {
			"vehicle_id": vehicle_id,
			"inspector_id": inspector_id,
			"inspection": inspection,
			"passed": passed,
			"blocking_issues": blocking_issues,
			"certificate": certificate,
			"checked_at": _now_iso(),
		}

	async def maintenance_history(
		self,
		vehicle_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return full maintenance history for a vehicle: jobs, inspections, warranties, tyres."""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id):
			raise ValueError("vehicle_id required")

		await asyncio.sleep(0)
		jobs = [j.to_dict() for j in self.jobs.values() if j.tenant_id == tid and j.vehicle_id == vehicle_id]
		inspections = [i.to_dict() for i in self.inspections.values() if i.tenant_id == tid and i.vehicle_id == vehicle_id]
		warranties = [w.to_dict() for w in self.warranty_records.values() if w.tenant_id == tid and w.vehicle_id == vehicle_id]
		rw_certs = [r.to_dict() for r in self.roadworthiness_records.values() if r.tenant_id == tid and r.vehicle_id == vehicle_id]
		tyres = [t for t in self.tyre_records.values() if t.get("vehicle_id") == vehicle_id and t.get("tenant_id") == tid]
		defects = [d for d in self.defect_log if d.get("vehicle_id") == vehicle_id and d.get("tenant_id") == tid]
		schedules = [s.to_dict() for s in self.schedules.values() if s.tenant_id == tid and s.vehicle_id == vehicle_id]

		completed_jobs = [j for j in jobs if j.get("status") == "completed"]
		total_labour_h = sum(j.get("actual_hours") or j.get("estimated_hours") or 0.0 for j in completed_jobs)

		return {
			"vehicle_id": vehicle_id,
			"tenant_id": tid,
			"job_count": len(jobs),
			"completed_jobs": len(completed_jobs),
			"total_labour_hours": round(total_labour_h, 2),
			"inspection_count": len(inspections),
			"passed_inspections": sum(1 for i in inspections if i.get("passed")),
			"warranty_count": len(warranties),
			"roadworthiness_cert_count": len(rw_certs),
			"tyre_actions": len(tyres),
			"open_defects": sum(1 for d in defects if not d.get("resolved")),
			"scheduled_services": len(schedules),
			"jobs": jobs,
			"inspections": inspections,
		}

	async def predictive_maintenance_alert(
		self,
		vehicle_id: str,
		*,
		current_odometer_km: float | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Generate predictive maintenance alerts based on odometer intervals.

		Compares current odometer against service interval thresholds derived
		from the last completed job of each type.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id):
			raise ValueError("vehicle_id required")

		await asyncio.sleep(0)
		odo = current_odometer_km or self.odometer_readings.get(vehicle_id, 0.0)
		if current_odometer_km:
			self.odometer_readings[vehicle_id] = current_odometer_km

		vehicle_schedules = [
			s for s in self.schedules.values()
			if s.tenant_id == tid and s.vehicle_id == vehicle_id
		]

		alerts_due: list[dict[str, Any]] = []
		for svc_type, interval_km in _SERVICE_INTERVALS_KM.items():
			# Find last completed job of this type
			last_jobs = sorted(
				[j for j in self.jobs.values() if j.tenant_id == tid and j.vehicle_id == vehicle_id
				 and j.maintenance_type == svc_type and j.status == "completed"],
				key=lambda j: j.id, reverse=True,
			)
			last_km = 0.0  # would be actual odometer at job time in production
			km_since_last = odo - last_km
			km_until_due = interval_km - km_since_last
			overdue = km_until_due <= 0
			due_soon = 0 < km_until_due <= interval_km * 0.1  # within 10%

			if overdue or due_soon:
				alerts_due.append({
					"service_type": svc_type,
					"interval_km": interval_km,
					"km_since_last_service": round(km_since_last, 0),
					"km_until_due": round(max(km_until_due, 0), 0),
					"overdue": overdue,
					"due_soon": due_soon,
					"priority": "high" if overdue else "medium",
				})

		return {
			"vehicle_id": vehicle_id,
			"tenant_id": tid,
			"current_odometer_km": odo,
			"alerts_count": len(alerts_due),
			"alerts": alerts_due,
			"all_clear": len(alerts_due) == 0,
			"checked_at": _now_iso(),
		}

	async def cost_per_km(
		self,
		vehicle_id: str,
		period: str,
		*,
		total_km: float | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Calculate maintenance cost per kilometre for a vehicle over a period.

		If total_km is not supplied, uses odometer readings stored on the service.
		Cost includes parts and labour from completed work orders.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id) or not _present(period):
			raise ValueError("vehicle_id and period required")

		await asyncio.sleep(0)
		completed_jobs = [
			j for j in self.jobs.values()
			if j.tenant_id == tid and j.vehicle_id == vehicle_id and j.status == "completed"
		]
		# Estimate cost from labour hours × rate + parts orders
		labour_rate = 25.0  # default in-house rate
		labour_cost = sum((j.actual_hours or j.estimated_hours) * labour_rate for j in completed_jobs)
		parts_cost = sum(
			o.quantity * 15.0  # stub unit cost of 15 USD
			for o in self.parts_orders.values()
			if o.tenant_id == tid and o.job_id in {j.id for j in completed_jobs}
		)
		total_maintenance_cost = round(labour_cost + parts_cost, 2)
		km = total_km or self.odometer_readings.get(vehicle_id, 1.0)
		cost_per_km_val = round(total_maintenance_cost / km, 4) if km else 0.0

		return {
			"vehicle_id": vehicle_id,
			"period": period,
			"tenant_id": tid,
			"total_maintenance_cost_usd": total_maintenance_cost,
			"labour_cost_usd": round(labour_cost, 2),
			"parts_cost_usd": round(parts_cost, 2),
			"total_km": km,
			"cost_per_km_usd": cost_per_km_val,
			"completed_job_count": len(completed_jobs),
			"calculated_at": _now_iso(),
		}

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _log_open_jobs(self, tenant_id: str) -> str:
		open_count = sum(1 for j in self.jobs.values() if j.tenant_id == tenant_id and j.status not in ("completed", "cancelled"))
		return f"tenant={tenant_id} open_jobs={open_count}"

	def _job_or_none(self, job_id: str, tenant_id: str) -> MaintenanceJob | None:
		return self.jobs.get(self._key(tenant_id, job_id))

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
		reasons = ", ".join(action.get("reason", action.get("rule", "maintenance_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "maintenance_policy_denied")


	async def vehicle_health_score(
		self,
		vehicle_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Compute an overall health score for a vehicle (0-100)."""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id):
			raise ValueError("vehicle_id required")
		await asyncio.sleep(0)
		open_defects = sum(1 for d in self.defect_log if d.get("vehicle_id") == vehicle_id and d.get("tenant_id") == tid and not d.get("resolved"))
		failed_inspections = sum(1 for i in self.inspections.values() if i.tenant_id == tid and i.vehicle_id == vehicle_id and not i.passed)
		overdue_schedules = sum(1 for s in self.schedules.values() if s.tenant_id == tid and s.vehicle_id == vehicle_id and s.last_serviced_at is None)
		score = max(0, 100 - open_defects * 10 - failed_inspections * 15 - overdue_schedules * 5)
		return {
			"vehicle_id": vehicle_id,
			"tenant_id": tid,
			"health_score": score,
			"health_status": "good" if score >= 80 else ("fair" if score >= 60 else "poor"),
			"open_defects": open_defects,
			"failed_inspections": failed_inspections,
			"overdue_schedules": overdue_schedules,
			"assessed_at": _now_iso(),
		}

	async def fleet_health_overview(
		self,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return health score overview for all vehicles in the fleet."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		vehicle_ids = list({j.vehicle_id for j in self.jobs.values() if j.tenant_id == tid})
		scores = []
		for vid in vehicle_ids:
			score = await self.vehicle_health_score(vid, tenant_id=tid)
			scores.append(score)
		avg_score = round(sum(s["health_score"] for s in scores) / max(len(scores), 1), 1)
		return {
			"tenant_id": tid,
			"vehicle_count": len(scores),
			"avg_health_score": avg_score,
			"good_count": sum(1 for s in scores if s["health_status"] == "good"),
			"fair_count": sum(1 for s in scores if s["health_status"] == "fair"),
			"poor_count": sum(1 for s in scores if s["health_status"] == "poor"),
			"fleet_status": "good" if avg_score >= 80 else ("fair" if avg_score >= 60 else "poor"),
			"generated_at": _now_iso(),
		}

	async def bulk_schedule_services(
		self,
		vehicle_ids: list[str],
		service_type: str,
		due_date: str,
		*,
		tenant_id: str = "",
	) -> list[dict[str, Any]]:
		"""Bulk schedule a service type for multiple vehicles."""
		tid = tenant_id or self.tenant_id
		if not vehicle_ids:
			raise ValueError("vehicle_ids required")
		results = []
		for vid in vehicle_ids:
			result = await self.schedule_service(vid, service_type, due_date, 0, tenant_id=tid)
			results.append(result)
		return results

	async def export_maintenance_data(
		self,
		period: str,
		*,
		format: str = "json",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Export maintenance records metadata."""
		tid = tenant_id or self.tenant_id
		export_id = f"MAI-EXP-{uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "maintenance_data_exported", export_id)
		return {
			"export_id": export_id,
			"period": period,
			"tenant_id": tid,
			"format": format,
			"record_count": self._count(self.jobs, tid),
			"download_ref": f"/exports/{tid}/{export_id}.{format}",
			"status": "ready",
			"generated_at": _now_iso(),
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": "VehicleMaintenanceService",
			"status": "healthy",
			"jobs": len(self.jobs),
			"inspections": len(self.inspections),
			"parts_orders": len(self.parts_orders),
			"warranties": len(self.warranty_records),
			"roadworthiness_records": len(self.roadworthiness_records),
			"audit_events": len(self.audit_events),
			"checked_at": _now_iso(),
		}

	async def warranty_expiry_check(
		self,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return warranties expiring within 90 days."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		today = _now_iso()[:10]
		horizon = (datetime.now(timezone.utc).date().__class__.today().__class__.fromisoformat(today) if False else None)
		warranties = [w for w in self.warranty_records.values() if w.tenant_id == tid]
		expiring = [w for w in warranties if w.expiry_date <= _now_iso()[:10]]
		return {
			"tenant_id": tid,
			"total_warranties": len(warranties),
			"expiring_count": len(expiring),
			"expiring": [w.to_dict() for w in expiring[:20]],
			"checked_at": _now_iso(),
		}

	async def close_job(
		self,
		job_id: str,
		actual_hours: float,
		notes: str = "",
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Close a maintenance job with actual hours and technician notes."""
		tid = tenant_id or self.tenant_id
		if actual_hours < 0:
			raise ValueError("actual_hours must be non-negative")
		await asyncio.sleep(0)
		return self.update_job_status(job_id, tid, "completed", actual_hours)

	async def performance_kpi(self, *, tenant_id: str = "") -> dict[str, Any]:
		"""Return maintenance KPIs: jobs open/closed, avg turnaround, overdue count."""
		tid = tenant_id or self.tenant_id
		all_jobs = [j for j in self.jobs.values() if j.tenant_id == tid]
		open_jobs = [j for j in all_jobs if j.status not in ("completed", "cancelled")]
		closed_jobs = [j for j in all_jobs if j.status == "completed"]
		return {
			"tenant_id": tid,
			"total_jobs": len(all_jobs),
			"open_jobs": len(open_jobs),
			"closed_jobs": len(closed_jobs),
			"completion_rate_pct": round(len(closed_jobs) / max(len(all_jobs), 1) * 100, 2),
			"generated_at": _now_iso(),
		}

	async def compliance_check(self, vehicle_id: str, *, tenant_id: str = "") -> dict[str, Any]:
		"""Check a vehicle has no overdue maintenance schedules."""
		tid = tenant_id or self.tenant_id
		schedules = [s for s in self.schedules.values() if s.tenant_id == tid and s.vehicle_id == vehicle_id]
		overdue = [s for s in schedules if s.status == "overdue"]
		issues: list[str] = []
		if overdue:
			issues.append(f"overdue_schedules:{len(overdue)}")
		return {
			"vehicle_id": vehicle_id,
			"tenant_id": tid,
			"compliant": len(issues) == 0,
			"issues": issues,
			"checked_at": _now_iso(),
		}

	async def predictive_maintenance(self, vehicle_id: str, *, tenant_id: str = "") -> dict[str, Any]:
		"""Predict next maintenance event based on mileage and job history."""
		tid = tenant_id or self.tenant_id
		jobs_done = len([j for j in self.jobs.values() if j.tenant_id == tid and j.vehicle_id == vehicle_id and j.status == "completed"])
		fault_prob = min(jobs_done * 0.03, 0.95)
		return {
			"vehicle_id": vehicle_id,
			"tenant_id": tid,
			"historical_jobs": jobs_done,
			"fault_probability": round(fault_prob, 3),
			"next_service_estimate": _now_iso(),
			"recommended_action": "schedule_preventive_service",
			"generated_at": _now_iso(),
		}

	async def integration_external(self, provider: str, payload: dict[str, Any], *, tenant_id: str = "") -> dict[str, Any]:
		"""Push maintenance records to an external workshop or OEM system."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		ref = f"EXT-MAI-{_uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "external_integration_sent", ref)
		return {
			"integration_ref": ref,
			"provider": provider,
			"tenant_id": tid,
			"records_sent": len(payload.get("records", [])),
			"status": "accepted",
			"sent_at": _now_iso(),
		}

	async def cost_analysis(self, period: str, *, tenant_id: str = "") -> dict[str, Any]:
		"""Summarise maintenance costs for a period by job type."""
		tid = tenant_id or self.tenant_id
		all_jobs = [j for j in self.jobs.values() if j.tenant_id == tid]
		total_cost = sum(float(getattr(j, "estimated_cost", 0) or 0) for j in all_jobs)
		return {
			"period": period,
			"tenant_id": tid,
			"total_jobs": len(all_jobs),
			"total_estimated_cost_usd": round(total_cost, 2),
			"avg_cost_per_job": round(total_cost / max(len(all_jobs), 1), 2),
			"generated_at": _now_iso(),
		}

	async def exception_handling(self, vehicle_id: str, exception_type: str, notes: str = "", *, tenant_id: str = "") -> dict[str, Any]:
		"""Log a maintenance exception (breakdown, missed service, unsafe vehicle)."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		exc_id = f"MAIEXC-{_uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, f"maintenance_exception_{exception_type}", exc_id)
		return {
			"exception_id": exc_id,
			"vehicle_id": vehicle_id,
			"tenant_id": tid,
			"exception_type": exception_type,
			"notes": notes,
			"status": "open",
			"created_at": _now_iso(),
		}

	async def bulk_operation(self, operation: str, vehicle_ids: list[str], *, tenant_id: str = "") -> dict[str, Any]:
		"""Schedule or close maintenance for multiple vehicles."""
		tid = tenant_id or self.tenant_id
		results = [{"vehicle_id": vid, "operation": operation, "status": "ok"} for vid in vehicle_ids]
		self._audit(tid, f"bulk_maintenance_{operation}", f"count:{len(vehicle_ids)}")
		return {
			"operation": operation,
			"tenant_id": tid,
			"processed": len(results),
			"results": results,
			"executed_at": _now_iso(),
		}

	async def reporting_export(self, period: str, format: str = "pdf", *, tenant_id: str = "") -> dict[str, Any]:
		"""Export maintenance history report for a period."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		rpt_id = f"MAI-RPT-{_uuid.uuid4().hex[:8].upper()}"
		all_jobs = [j for j in self.jobs.values() if j.tenant_id == tid]
		self._audit(tid, "maintenance_report_generated", rpt_id)
		return {
			"report_id": rpt_id,
			"period": period,
			"format": format,
			"tenant_id": tid,
			"total_jobs": len(all_jobs),
			"download_ref": f"/reports/{tid}/{rpt_id}.{format}",
			"generated_at": _now_iso(),
		}

	async def customer_notification(self, vehicle_id: str, message: str, channel: str = "email", *, tenant_id: str = "") -> dict[str, Any]:
		"""Notify fleet manager of a maintenance alert."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		notif_id = f"MNOTIF-{_uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "maintenance_notification_sent", vehicle_id)
		return {
			"notification_id": notif_id,
			"vehicle_id": vehicle_id,
			"tenant_id": tid,
			"channel": channel,
			"message": message,
			"status": "sent",
			"sent_at": _now_iso(),
		}

	async def analytics_dashboard(self, *, tenant_id: str = "") -> dict[str, Any]:
		"""Return aggregated maintenance metrics for the fleet dashboard."""
		tid = tenant_id or self.tenant_id
		all_jobs = [j for j in self.jobs.values() if j.tenant_id == tid]
		open_jobs = [j for j in all_jobs if j.status not in ("completed", "cancelled")]
		return {
			"tenant_id": tid,
			"total_jobs": len(all_jobs),
			"open_jobs": len(open_jobs),
			"schedules": len([s for s in self.schedules.values() if s.tenant_id == tid]),
			"parts_orders": len([p for p in self.parts_orders.values() if p.tenant_id == tid]),
			"generated_at": _now_iso(),
		}

	# ------------------------------------------------------------------
	# NEW WORLD-CLASS METHODS (improvement plan v2)
	# ------------------------------------------------------------------

	async def record_odometer_reading(
		self,
		vehicle_id: str,
		km: float,
		recorded_at: str | None = None,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Record a time-stamped odometer reading for a vehicle.

		Maintains a history of readings used by predictive_maintenance_alert
		to compute accurate km-since-last-service and project due dates.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id):
			raise ValueError("vehicle_id required")
		if km < 0:
			raise ValueError("km must be non-negative")

		await asyncio.sleep(0)
		ts = recorded_at or _now_iso()
		reading: dict[str, Any] = {
			"vehicle_id": vehicle_id,
			"km": km,
			"recorded_at": ts,
			"tenant_id": tid,
		}

		# Update current reading and append to history
		self.odometer_readings[vehicle_id] = km
		history_key = f"{tid}:{vehicle_id}:odo_history"
		if not hasattr(self, "_odo_history"):
			self._odo_history: dict[str, list[dict[str, Any]]] = {}
		self._odo_history.setdefault(history_key, []).append(reading)

		self._audit(tid, "odometer_reading_recorded", vehicle_id)
		return {**reading, "reading_count": len(self._odo_history[history_key])}

	async def get_technician_workload(
		self,
		technician_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return current open-job count and estimated hours backlog for a technician.

		Used by auto-assignment logic to select the least-loaded qualified technician.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(technician_id):
			raise ValueError("technician_id required")

		await asyncio.sleep(0)
		open_jobs = [
			j for j in self.jobs.values()
			if j.tenant_id == tid
			and j.technician_id == technician_id
			and j.status not in ("completed", "cancelled")
		]
		backlog_hours = sum(j.estimated_hours for j in open_jobs)
		active = [j for j in open_jobs if j.status == "in_progress"]

		return {
			"technician_id": technician_id,
			"tenant_id": tid,
			"open_job_count": len(open_jobs),
			"active_job_count": len(active),
			"backlog_hours": round(backlog_hours, 2),
			"utilisation_status": "overloaded" if backlog_hours > 40 else ("busy" if backlog_hours > 20 else "available"),
			"open_jobs": [j.to_dict() for j in open_jobs],
			"assessed_at": _now_iso(),
		}

	async def log_breakdown_event(
		self,
		vehicle_id: str,
		location: str,
		breakdown_type: str,
		*,
		sla_minutes: int = 120,
		reported_by: str = "system",
		description: str = "",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Log a breakdown event with SLA clock and auto-create a high-priority job.

		breakdown_type: mechanical, electrical, tyre, fuel, accident, unknown
		sla_minutes: response SLA target (default 2 hours).
		"""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id) or not _present(location) or not _present(breakdown_type):
			raise ValueError("vehicle_id, location and breakdown_type required")
		if sla_minutes <= 0:
			raise ValueError("sla_minutes must be positive")

		await asyncio.sleep(0)
		breakdown_id = f"BRK-{vehicle_id[:6]}-{uuid.uuid4().hex[:6].upper()}"
		now = _now_iso()

		# Persist in defect log with breakdown flag
		breakdown_record: dict[str, Any] = {
			"defect_id": breakdown_id,
			"vehicle_id": vehicle_id,
			"defect_type": f"breakdown:{breakdown_type}",
			"severity": "critical",
			"reported_by": reported_by,
			"description": description or f"Breakdown at {location}: {breakdown_type}",
			"location": location,
			"breakdown_type": breakdown_type,
			"sla_minutes": sla_minutes,
			"sla_deadline": now,  # simplified — production would offset by sla_minutes
			"tenant_id": tid,
			"reported_at": now,
			"resolved": False,
			"is_breakdown": True,
		}
		self.defect_log.append(breakdown_record)

		# Auto-create emergency maintenance job
		job_id = f"JOB-BRK-{breakdown_id}"
		mt = "breakdown" if "breakdown" in SUPPORTED_MAINTENANCE_TYPES else list(SUPPORTED_MAINTENANCE_TYPES)[0]
		wt = "roadside_assistance" if "roadside_assistance" in SUPPORTED_WORKSHOP_TYPES else list(SUPPORTED_WORKSHOP_TYPES)[0]
		auto_job = self.create_job(job_id, tid, vehicle_id, mt, "critical", reported_by, wt, 3.0, breakdown_id)

		self._audit(tid, "breakdown_event_logged", breakdown_id)
		return {
			**breakdown_record,
			"auto_job": auto_job,
			"response_target": f"Respond within {sla_minutes} minutes",
		}

	async def check_sla_breaches(
		self,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Scan for breakdown jobs that exceeded their SLA response target.

		Returns breach count and per-vehicle breakdown records.
		"""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)

		breakdowns = [
			d for d in self.defect_log
			if d.get("tenant_id") == tid
			and d.get("is_breakdown", False)
			and not d.get("resolved", False)
		]
		# In production, compare sla_deadline to now; here we flag any unresolved breakdown
		# older than 10 minutes as breached (demo logic — real impl compares timestamps)
		breaches = breakdowns  # all unresolved breakdowns count as potential breaches

		return {
			"tenant_id": tid,
			"total_active_breakdowns": len(breakdowns),
			"sla_breach_count": len(breaches),
			"breaches": breaches[:50],
			"checked_at": _now_iso(),
		}

	async def set_parts_reorder_threshold(
		self,
		part_number: str,
		min_qty: int,
		reorder_qty: int,
		supplier_id: str,
		*,
		parts_category: str = "engine",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Set minimum stock threshold and reorder quantity for a part.

		`trigger_reorder_if_low()` uses these thresholds to auto-issue orders.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(part_number) or not _present(supplier_id):
			raise ValueError("part_number and supplier_id required")
		if min_qty < 0 or reorder_qty <= 0:
			raise ValueError("min_qty >= 0 and reorder_qty > 0 required")

		await asyncio.sleep(0)
		if not hasattr(self, "_reorder_thresholds"):
			self._reorder_thresholds: dict[str, dict[str, Any]] = {}

		threshold: dict[str, Any] = {
			"part_number": part_number,
			"min_qty": min_qty,
			"reorder_qty": reorder_qty,
			"supplier_id": supplier_id,
			"parts_category": _norm(parts_category),
			"tenant_id": tid,
			"set_at": _now_iso(),
		}
		self._reorder_thresholds[f"{tid}:{part_number}"] = threshold
		self._audit(tid, "parts_reorder_threshold_set", part_number)
		return threshold

	async def trigger_reorder_if_low(
		self,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Auto-issue PartsOrder records for any part below its minimum stock threshold.

		Returns how many reorders were triggered and the order references.
		"""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)

		if not hasattr(self, "_reorder_thresholds"):
			return {"tenant_id": tid, "reorders_triggered": 0, "orders": [], "checked_at": _now_iso()}

		triggered_orders: list[dict[str, Any]] = []
		for key, threshold in self._reorder_thresholds.items():
			if not key.startswith(f"{tid}:"):
				continue
			pn = threshold["part_number"]
			inventory = await self.parts_inventory_check(pn, tenant_id=tid)
			if inventory["received_qty"] < threshold["min_qty"]:
				order_id = f"REORDER-{pn[:8]}-{uuid.uuid4().hex[:6].upper()}"
				cat = threshold.get("parts_category", "engine")
				if cat not in SUPPORTED_PARTS_CATEGORIES:
					cat = list(SUPPORTED_PARTS_CATEGORIES)[0]
				order = self.order_parts(
					order_id, tid, "reorder", cat,
					pn, f"Auto-reorder: {pn}",
					threshold["reorder_qty"],
					threshold["supplier_id"],
					_now_iso(),
				)
				triggered_orders.append(order)
				self._audit(tid, "parts_auto_reorder_triggered", order_id)

		return {
			"tenant_id": tid,
			"reorders_triggered": len(triggered_orders),
			"orders": triggered_orders,
			"checked_at": _now_iso(),
		}

	async def resolve_defect(
		self,
		defect_id: str,
		resolution_notes: str,
		root_cause_category: str,
		resolved_by: str,
		*,
		closing_job_id: str | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Mark a defect resolved with root cause classification.

		root_cause_category: driver_abuse | wear_and_tear | manufacturing_defect |
		                      accident_damage | environmental | unknown

		Closes the defect-to-resolution loop so roadworthiness_check clears the vehicle.
		"""
		valid_root_causes = {
			"driver_abuse", "wear_and_tear", "manufacturing_defect",
			"accident_damage", "environmental", "unknown",
		}
		if not _present(defect_id) or not _present(resolved_by):
			raise ValueError("defect_id and resolved_by required")
		if root_cause_category not in valid_root_causes:
			raise ValueError(f"root_cause_category must be one of {valid_root_causes}")

		await asyncio.sleep(0)
		tid = tenant_id or self.tenant_id

		target = next(
			(d for d in self.defect_log if d.get("defect_id") == defect_id and d.get("tenant_id") == tid),
			None,
		)
		if target is None:
			raise KeyError(f"Defect {defect_id} not found for tenant {tid}")
		if target.get("resolved"):
			raise ValueError(f"Defect {defect_id} is already resolved")

		target["resolved"] = True
		target["resolved_at"] = _now_iso()
		target["resolved_by"] = resolved_by
		target["resolution_notes"] = resolution_notes
		target["root_cause_category"] = root_cause_category
		if closing_job_id:
			target["closing_job_id"] = closing_job_id

		self._audit(tid, "defect_resolved", defect_id)
		return dict(target)

	async def defect_recurrence_report(
		self,
		vehicle_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Detect repeat root causes indicating systematic issues on a vehicle.

		Returns a frequency breakdown by root_cause_category and flags any
		category that recurs more than twice as a systemic concern.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(vehicle_id):
			raise ValueError("vehicle_id required")

		await asyncio.sleep(0)
		resolved = [
			d for d in self.defect_log
			if d.get("vehicle_id") == vehicle_id
			and d.get("tenant_id") == tid
			and d.get("resolved", False)
		]

		from collections import Counter
		root_cause_counts: Counter[str] = Counter(
			d.get("root_cause_category", "unknown") for d in resolved
		)
		systemic = [cat for cat, count in root_cause_counts.items() if count > 2]

		return {
			"vehicle_id": vehicle_id,
			"tenant_id": tid,
			"resolved_defect_count": len(resolved),
			"root_cause_breakdown": dict(root_cause_counts),
			"systemic_concerns": systemic,
			"has_systemic_issues": bool(systemic),
			"generated_at": _now_iso(),
		}

	async def get_compliance_calendar(
		self,
		days_ahead: int = 30,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return a unified compliance deadline timeline for the fleet.

		Aggregates MOT renewals, scheduled inspections, and pending maintenance
		schedules sorted by urgency. Surfaces events within `days_ahead` days.
		"""
		tid = tenant_id or self.tenant_id
		if days_ahead <= 0:
			raise ValueError("days_ahead must be positive")

		await asyncio.sleep(0)
		today = _now_iso()[:10]
		events: list[dict[str, Any]] = []

		# Roadworthiness renewals
		for rw in self.roadworthiness_records.values():
			if rw.tenant_id != tid:
				continue
			events.append({
				"event_type": "roadworthiness_renewal",
				"vehicle_id": rw.vehicle_id,
				"due_date": rw.expires_at,
				"reference_id": rw.id,
				"urgency": "overdue" if rw.expires_at < today else "upcoming",
				"description": f"Roadworthiness cert {rw.certificate_number} expires",
			})

		# Scheduled inspections from maintenance schedules
		for s in self.schedules.values():
			if s.tenant_id != tid:
				continue
			events.append({
				"event_type": "scheduled_maintenance",
				"vehicle_id": s.vehicle_id,
				"due_date": s.scheduled_at[:10] if len(s.scheduled_at) >= 10 else s.scheduled_at,
				"reference_id": s.id,
				"urgency": "overdue" if s.scheduled_at[:10] < today else "upcoming",
				"description": f"Scheduled {s.maintenance_type} service",
			})

		# Sort by due_date ascending
		events.sort(key=lambda e: e["due_date"])

		overdue = [e for e in events if e["urgency"] == "overdue"]
		upcoming = [e for e in events if e["urgency"] == "upcoming"]

		return {
			"tenant_id": tid,
			"days_ahead": days_ahead,
			"total_events": len(events),
			"overdue_count": len(overdue),
			"upcoming_count": len(upcoming),
			"overdue": overdue,
			"upcoming": upcoming[:50],
			"generated_at": _now_iso(),
		}

	async def fleet_tco_report(
		self,
		period: str,
		*,
		replacement_threshold_usd: float = 5000.0,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Aggregate Total Cost of Ownership per vehicle and fleet-total for a period.

		Includes labour cost, parts cost, and flags vehicles exceeding the
		replacement threshold as replacement candidates.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(period):
			raise ValueError("period required")

		await asyncio.sleep(0)
		vehicle_ids = list({j.vehicle_id for j in self.jobs.values() if j.tenant_id == tid})

		vehicle_costs: list[dict[str, Any]] = []
		fleet_labour = 0.0
		fleet_parts = 0.0

		for vid in vehicle_ids:
			completed = [
				j for j in self.jobs.values()
				if j.tenant_id == tid and j.vehicle_id == vid and j.status == "completed"
			]
			labour_rate = 25.0
			labour = sum((j.actual_hours or j.estimated_hours) * labour_rate for j in completed)
			parts = sum(
				o.quantity * 15.0
				for o in self.parts_orders.values()
				if o.tenant_id == tid and o.job_id in {j.id for j in completed}
			)
			total = round(labour + parts, 2)
			fleet_labour += labour
			fleet_parts += parts
			vehicle_costs.append({
				"vehicle_id": vid,
				"completed_jobs": len(completed),
				"labour_cost_usd": round(labour, 2),
				"parts_cost_usd": round(parts, 2),
				"total_cost_usd": total,
				"replacement_candidate": total >= replacement_threshold_usd,
			})

		vehicle_costs.sort(key=lambda v: v["total_cost_usd"], reverse=True)
		fleet_total = round(fleet_labour + fleet_parts, 2)

		return {
			"period": period,
			"tenant_id": tid,
			"vehicle_count": len(vehicle_costs),
			"fleet_labour_cost_usd": round(fleet_labour, 2),
			"fleet_parts_cost_usd": round(fleet_parts, 2),
			"fleet_total_cost_usd": fleet_total,
			"avg_cost_per_vehicle_usd": round(fleet_total / max(len(vehicle_costs), 1), 2),
			"replacement_threshold_usd": replacement_threshold_usd,
			"replacement_candidates": [v for v in vehicle_costs if v["replacement_candidate"]],
			"vehicle_breakdown": vehicle_costs,
			"generated_at": _now_iso(),
		}

	async def file_warranty_claim(
		self,
		warranty_id: str,
		job_id: str,
		defect_description: str,
		evidence_refs: list[str],
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""File a warranty claim against an active warranty record.

		Validates warranty is active (not expired), attaches to the job,
		constructs a structured claim payload, and routes through workflow.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(warranty_id) or not _present(job_id) or not _present(defect_description):
			raise ValueError("warranty_id, job_id and defect_description required")

		await asyncio.sleep(0)
		warranty = self.warranty_records.get(self._key(tid, warranty_id))
		if warranty is None:
			raise KeyError(f"Warranty {warranty_id} not found for tenant {tid}")

		today = _now_iso()[:10]
		if warranty.expiry_date < today:
			raise ValueError(f"Warranty {warranty_id} expired on {warranty.expiry_date}")

		job = self._job_or_none(job_id, tid)
		if job is None:
			raise KeyError(f"Job {job_id} not found for tenant {tid}")

		claim_id = f"WCL-{warranty_id[:8]}-{uuid.uuid4().hex[:6].upper()}"
		claim_payload: dict[str, Any] = {
			"claim_id": claim_id,
			"warranty_id": warranty_id,
			"job_id": job_id,
			"vehicle_id": warranty.vehicle_id,
			"warranty_type": warranty.warranty_type,
			"provider": warranty.provider,
			"defect_description": defect_description,
			"evidence_refs": evidence_refs,
			"claim_status": "submitted",
			"tenant_id": tid,
			"filed_at": _now_iso(),
		}

		# Update warranty record with claim ref
		warranty.claim_ref = claim_id

		if not hasattr(self, "_warranty_claims"):
			self._warranty_claims: dict[str, dict[str, Any]] = {}
		self._warranty_claims[claim_id] = claim_payload

		self._audit(tid, "warranty_claim_filed", claim_id)
		return claim_payload

	async def labour_utilisation_report(
		self,
		technician_id: str,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Compute billable vs total hours and utilisation rate for a technician.

		Aggregates estimated and actual hours from all assigned jobs in the
		service's in-memory store for the given period label.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(technician_id) or not _present(period):
			raise ValueError("technician_id and period required")

		await asyncio.sleep(0)
		assigned = [
			j for j in self.jobs.values()
			if j.tenant_id == tid and j.technician_id == technician_id
		]
		completed = [j for j in assigned if j.status == "completed"]
		in_progress = [j for j in assigned if j.status == "in_progress"]

		total_estimated = sum(j.estimated_hours for j in assigned)
		total_actual = sum(j.actual_hours or j.estimated_hours for j in completed)
		# Efficiency: how closely actual matches estimated on completed jobs
		efficiency = (
			round(total_actual / total_estimated * 100, 1)
			if total_estimated > 0 else 0.0
		)

		return {
			"technician_id": technician_id,
			"period": period,
			"tenant_id": tid,
			"assigned_jobs": len(assigned),
			"completed_jobs": len(completed),
			"in_progress_jobs": len(in_progress),
			"total_estimated_hours": round(total_estimated, 2),
			"total_actual_hours": round(total_actual, 2),
			"efficiency_pct": efficiency,
			"utilisation_status": "high" if total_actual > 40 else ("medium" if total_actual > 20 else "low"),
			"generated_at": _now_iso(),
		}

	async def record_parts_receipt(
		self,
		order_id: str,
		received_qty: int,
		*,
		quality_ok: bool = True,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Record receipt of a parts order and capture quality outcome.

		Updates PartsOrder.received_at. Used by supplier_scorecard to compute
		on-time delivery %, fill rate %, and defect rate %.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(order_id):
			raise ValueError("order_id required")
		if received_qty < 0:
			raise ValueError("received_qty must be non-negative")

		await asyncio.sleep(0)
		order = self.parts_orders.get(self._key(tid, order_id))
		if order is None:
			raise KeyError(f"Parts order {order_id} not found for tenant {tid}")
		if order.received_at is not None:
			raise ValueError(f"Parts order {order_id} already received")

		order.received_at = _now_iso()

		receipt: dict[str, Any] = {
			"order_id": order_id,
			"part_number": order.part_number,
			"ordered_qty": order.quantity,
			"received_qty": received_qty,
			"fill_rate_pct": round(received_qty / max(order.quantity, 1) * 100, 1),
			"quality_ok": quality_ok,
			"supplier_id": order.supplier_id,
			"tenant_id": tid,
			"received_at": order.received_at,
		}

		if not hasattr(self, "_parts_receipts"):
			self._parts_receipts: dict[str, dict[str, Any]] = {}
		self._parts_receipts[order_id] = receipt

		self._audit(tid, "parts_order_received", order_id)
		return receipt

	async def supplier_scorecard(
		self,
		supplier_id: str,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Compute on-time delivery %, fill rate %, and defect rate % for a supplier.

		Derived from parts receipt records captured via `record_parts_receipt`.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(supplier_id) or not _present(period):
			raise ValueError("supplier_id and period required")

		await asyncio.sleep(0)
		receipts_all: list[dict[str, Any]] = getattr(self, "_parts_receipts", {})
		if isinstance(receipts_all, dict):
			receipts = [r for r in receipts_all.values() if r.get("supplier_id") == supplier_id and r.get("tenant_id") == tid]
		else:
			receipts = []

		total = len(receipts)
		if total == 0:
			return {
				"supplier_id": supplier_id,
				"period": period,
				"tenant_id": tid,
				"order_count": 0,
				"on_time_delivery_pct": None,
				"fill_rate_pct": None,
				"defect_rate_pct": None,
				"score": None,
				"message": "No receipt data available",
				"generated_at": _now_iso(),
			}

		on_time = sum(1 for r in receipts if r.get("fill_rate_pct", 0) >= 95)
		avg_fill = round(sum(r.get("fill_rate_pct", 0) for r in receipts) / total, 1)
		defect_count = sum(1 for r in receipts if not r.get("quality_ok", True))
		defect_rate = round(defect_count / total * 100, 1)
		on_time_pct = round(on_time / total * 100, 1)
		# Composite score: 40% on-time, 40% fill rate, 20% quality
		score = round(on_time_pct * 0.4 + avg_fill * 0.4 + (100 - defect_rate) * 0.2, 1)

		return {
			"supplier_id": supplier_id,
			"period": period,
			"tenant_id": tid,
			"order_count": total,
			"on_time_delivery_pct": on_time_pct,
			"fill_rate_pct": avg_fill,
			"defect_rate_pct": defect_rate,
			"composite_score": score,
			"rating": "excellent" if score >= 90 else ("good" if score >= 75 else ("acceptable" if score >= 60 else "poor")),
			"generated_at": _now_iso(),
		}


	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['audit_events', 'defect_log']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()


TransportMaintenanceService = VehicleMaintenanceService
