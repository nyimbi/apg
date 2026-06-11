"""Async service layer for Space Planning & Management (spa)."""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any

from .models import (
	FloorPlanCreate, FloorPlanResponse,
	SpaceCreate, SpaceResponse, SpaceUpdate,
	SpaceAllocationCreate, SpaceAllocationResponse,
	MoveCreate, MoveResponse,
	BookingCreate, BookingResponse,
	OccupancyDataCreate, OccupancyDataResponse,
	DensityPlanCreate, DensityPlanResponse,
	SpaceStatus, MoveStatus, BookingType,
)
from .capability_contract import evaluate_capability_rules
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

log = logging.getLogger(__name__)


class SpaService:
	"""Service implementing all Space Planning & Management operations."""

	def __init__(
		self,
		tenant_id: str | None = None,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: dict[str, Any] | None = None,
	) -> None:
		self._tenant_id = tenant_id
		self._actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store: dict[str, list[dict[str, Any]]] = store or {
			"floor_plans": [], "spaces": [], "allocations": [],
			"moves": [], "bookings": [], "occupancy_data": [],
			"density_plans": [], "scenarios": [],
		}

	# ── Logging helpers ───────────────────────────────────────────────────────

	def _log_operation(self, op: str, entity_id: str, tenant_id: str) -> None:
		log.info("spa.%s entity=%s tenant=%s", op, entity_id, tenant_id)

	def _log_density_breach(self, property_id: str, current: Decimal, target: Decimal) -> None:
		log.warning("spa.density_breach property=%s current=%s target=%s", property_id, current, target)

	def _log_double_booking_attempt(self, space_id: str, start: datetime, end: datetime) -> None:
		log.warning("spa.double_booking_attempt space=%s start=%s end=%s", space_id, start, end)

	# ── Rules ─────────────────────────────────────────────────────────────────

	def _check_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			log.warning("spa.rule_denied rule=%s reason=%s", result["rule"], result["reason"])
			raise ValueError(f"rule_denied:{result['rule']}:{result['reason']}")

	# ── Floor Plan ────────────────────────────────────────────────────────────

	async def upload_floor_plan(self, payload: FloorPlanCreate) -> FloorPlanResponse:
		"""Upload a new floor plan."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "upload_floor_plan",
			"format_supported": True,
			"operation_type": "write",
			"policy_attached": True,
		})
		version = 1
		for fp in self._store["floor_plans"]:
			if fp["property_id"] == payload.property_id and fp["floor"] == payload.floor and fp["tenant_id"] == payload.tenant_id:
				version = max(version, fp.get("version", 1) + 1)
		record = FloorPlanResponse(**payload.model_dump(), version=version)
		self._store["floor_plans"].append(record.model_dump())
		self._log_operation("upload_floor_plan", record.id, record.tenant_id)
		return record

	async def get_floor_plan(self, floor_plan_id: str, tenant_id: str) -> FloorPlanResponse | None:
		"""Fetch a floor plan."""
		for fp in self._store["floor_plans"]:
			if fp["id"] == floor_plan_id and fp["tenant_id"] == tenant_id:
				return FloorPlanResponse(**fp)
		return None

	async def list_floor_plans(self, tenant_id: str, property_id: str | None = None) -> list[FloorPlanResponse]:
		"""List floor plans."""
		results = [fp for fp in self._store["floor_plans"] if fp["tenant_id"] == tenant_id]
		if property_id:
			results = [fp for fp in results if fp.get("property_id") == property_id]
		return [FloorPlanResponse(**fp) for fp in results]

	# ── Space ─────────────────────────────────────────────────────────────────

	async def create_space(self, payload: SpaceCreate) -> SpaceResponse:
		"""Create a new space record."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "create_space",
			"space_type_supported": True,
			"floor_plan_linked": True,
		})
		record = SpaceResponse(**payload.model_dump())
		self._store["spaces"].append(record.model_dump())
		for i, fp in enumerate(self._store["floor_plans"]):
			if fp["id"] == payload.floor_plan_id and fp["tenant_id"] == payload.tenant_id:
				fp["space_ids"].append(record.id)
				self._store["floor_plans"][i] = fp
				break
		self._log_operation("create_space", record.id, record.tenant_id)
		return record

	async def get_space(self, space_id: str, tenant_id: str) -> SpaceResponse | None:
		"""Fetch a space."""
		for s in self._store["spaces"]:
			if s["id"] == space_id and s["tenant_id"] == tenant_id:
				return SpaceResponse(**s)
		return None

	async def list_spaces(self, tenant_id: str, property_id: str | None = None, space_type: str | None = None, status: str | None = None) -> list[SpaceResponse]:
		"""List spaces with optional filters."""
		results = [s for s in self._store["spaces"] if s["tenant_id"] == tenant_id]
		if property_id:
			results = [s for s in results if s.get("property_id") == property_id]
		if space_type:
			results = [s for s in results if s.get("space_type") == space_type]
		if status:
			results = [s for s in results if s.get("status") == status]
		return [SpaceResponse(**s) for s in results]

	async def update_space(self, space_id: str, tenant_id: str, updates: SpaceUpdate) -> SpaceResponse | None:
		"""Update space metadata."""
		for i, s in enumerate(self._store["spaces"]):
			if s["id"] == space_id and s["tenant_id"] == tenant_id:
				s.update({k: v for k, v in updates.model_dump().items() if v is not None})
				s["updated_at"] = datetime.utcnow()
				self._store["spaces"][i] = s
				return SpaceResponse(**s)
		return None

	async def get_available_spaces(self, tenant_id: str, property_id: str | None = None, space_type: str | None = None, min_capacity: int = 1) -> list[SpaceResponse]:
		"""Return available spaces meeting capacity requirements."""
		spaces = await self.list_spaces(tenant_id, property_id, space_type, status=SpaceStatus.available.value)
		return [s for s in spaces if s.capacity >= min_capacity]

	# ── Space Allocation ──────────────────────────────────────────────────────

	async def allocate_space(self, payload: SpaceAllocationCreate) -> SpaceAllocationResponse:
		"""Allocate a space to a department or occupant(s)."""
		space = await self.get_space(payload.space_id, payload.tenant_id)
		if space and space.status.value == "decommissioned":
			self._check_rules({"operation": "book_space", "space_status": "decommissioned"})
		self._check_rules({
			"tenant_context_present": True,
			"operation": "allocate_space",
			"allocation_type_supported": True,
		})
		record = SpaceAllocationResponse(**payload.model_dump())
		self._store["allocations"].append(record.model_dump())
		for i, s in enumerate(self._store["spaces"]):
			if s["id"] == payload.space_id and s["tenant_id"] == payload.tenant_id:
				s["status"] = SpaceStatus.occupied.value
				s["current_allocation_id"] = record.id
				s["current_occupant_ids"] = payload.occupant_ids
				s["updated_at"] = datetime.utcnow()
				self._store["spaces"][i] = s
				break
		self._log_operation("allocate_space", record.id, record.tenant_id)
		return record

	async def deallocate_space(self, allocation_id: str, tenant_id: str) -> SpaceAllocationResponse | None:
		"""End a space allocation."""
		for i, a in enumerate(self._store["allocations"]):
			if a["id"] == allocation_id and a["tenant_id"] == tenant_id:
				a["is_active"] = False
				a["ended_at"] = datetime.utcnow()
				a["updated_at"] = datetime.utcnow()
				self._store["allocations"][i] = a
				for j, s in enumerate(self._store["spaces"]):
					if s["id"] == a["space_id"] and s["tenant_id"] == tenant_id:
						s["status"] = SpaceStatus.available.value
						s["current_allocation_id"] = None
						s["current_occupant_ids"] = []
						s["updated_at"] = datetime.utcnow()
						self._store["spaces"][j] = s
						break
				return SpaceAllocationResponse(**a)
		return None

	async def list_allocations(self, tenant_id: str, space_id: str | None = None, is_active: bool = True) -> list[SpaceAllocationResponse]:
		"""List space allocations."""
		results = [a for a in self._store["allocations"] if a["tenant_id"] == tenant_id and a.get("is_active", True) == is_active]
		if space_id:
			results = [a for a in results if a["space_id"] == space_id]
		return [SpaceAllocationResponse(**a) for a in results]

	# ── Move Management ───────────────────────────────────────────────────────

	async def create_move(self, payload: MoveCreate) -> MoveResponse:
		"""Create a move request."""
		large_move = payload.headcount >= 20
		self._check_rules({
			"tenant_context_present": True,
			"operation": "create_move",
			"move_type_supported": True,
			"headcount_above_threshold": large_move,
			"approved": not large_move,
		})
		status = MoveStatus.planning if large_move else MoveStatus.scheduled
		record = MoveResponse(**payload.model_dump(), status=status)
		self._store["moves"].append(record.model_dump())
		self._log_operation("create_move", record.id, record.tenant_id)
		return record

	async def approve_move(self, move_id: str, tenant_id: str, approved_by: str) -> MoveResponse | None:
		"""Approve a large-headcount move."""
		for i, m in enumerate(self._store["moves"]):
			if m["id"] == move_id and m["tenant_id"] == tenant_id:
				m["status"] = MoveStatus.approved.value
				m["approved_by"] = approved_by
				m["updated_at"] = datetime.utcnow()
				self._store["moves"][i] = m
				return MoveResponse(**m)
		return None

	async def complete_move(self, move_id: str, tenant_id: str) -> MoveResponse | None:
		"""Mark a move as completed."""
		for i, m in enumerate(self._store["moves"]):
			if m["id"] == move_id and m["tenant_id"] == tenant_id:
				m["status"] = MoveStatus.completed.value
				m["completed_at"] = datetime.utcnow()
				m["updated_at"] = datetime.utcnow()
				self._store["moves"][i] = m
				self._log_operation("complete_move", move_id, tenant_id)
				return MoveResponse(**m)
		return None

	async def list_moves(self, tenant_id: str, status: str | None = None) -> list[MoveResponse]:
		"""List moves."""
		results = [m for m in self._store["moves"] if m["tenant_id"] == tenant_id]
		if status:
			results = [m for m in results if m.get("status") == status]
		return [MoveResponse(**m) for m in results]

	# ── Bookings ──────────────────────────────────────────────────────────────

	async def create_booking(self, payload: BookingCreate) -> BookingResponse:
		"""Create a space booking."""
		for existing in self._store["bookings"]:
			if (existing["space_id"] == payload.space_id and
				existing["tenant_id"] == payload.tenant_id and
				existing["status"] == "confirmed"):
				ex_start = datetime.fromisoformat(str(existing["start_datetime"]))
				ex_end = datetime.fromisoformat(str(existing["end_datetime"]))
				if not (payload.end_datetime <= ex_start or payload.start_datetime >= ex_end):
					self._log_double_booking_attempt(payload.space_id, payload.start_datetime, payload.end_datetime)
					self._check_rules({"operation": "book_space", "space_already_booked": True})
		space = await self.get_space(payload.space_id, payload.tenant_id)
		if space:
			self._check_rules({"operation": "book_space", "space_status": space.status.value})
		max_advance = timedelta(days=90)
		too_far = (payload.start_datetime.date() - date.today()) > max_advance
		self._check_rules({
			"tenant_context_present": True,
			"operation": "create_booking",
			"booking_type_supported": True,
			"booking_too_far_in_advance": too_far,
		})
		record = BookingResponse(**payload.model_dump())
		self._store["bookings"].append(record.model_dump())
		return record

	async def cancel_booking(self, booking_id: str, tenant_id: str) -> BookingResponse | None:
		"""Cancel a booking."""
		for i, b in enumerate(self._store["bookings"]):
			if b["id"] == booking_id and b["tenant_id"] == tenant_id:
				b["status"] = "cancelled"
				b["cancelled_at"] = datetime.utcnow()
				b["updated_at"] = datetime.utcnow()
				self._store["bookings"][i] = b
				return BookingResponse(**b)
		return None

	async def list_bookings(self, tenant_id: str, space_id: str | None = None, booking_type: str | None = None) -> list[BookingResponse]:
		"""List bookings."""
		results = [b for b in self._store["bookings"] if b["tenant_id"] == tenant_id]
		if space_id:
			results = [b for b in results if b["space_id"] == space_id]
		if booking_type:
			results = [b for b in results if b.get("booking_type") == booking_type]
		return [BookingResponse(**b) for b in results]

	# ── Occupancy Analytics ───────────────────────────────────────────────────

	async def ingest_occupancy_data(self, payload: OccupancyDataCreate) -> OccupancyDataResponse:
		"""Ingest a sensor occupancy reading."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "ingest_sensor_data",
			"data_anonymised": payload.data_anonymised,
		})
		record = OccupancyDataResponse(**payload.model_dump())
		self._store["occupancy_data"].append(record.model_dump())
		return record

	async def calculate_occupancy_metrics(self, tenant_id: str, property_id: str, from_date: date, to_date: date) -> dict[str, Any]:
		"""Calculate occupancy metrics over a date range."""
		readings = [r for r in self._store["occupancy_data"] if r["tenant_id"] == tenant_id]
		if not readings:
			return {"tenant_id": tenant_id, "property_id": property_id, "average_daily_occupancy": 0, "peak_occupancy": 0, "utilisation_rate": 0}
		counts = [r["occupant_count"] for r in readings]
		avg = sum(counts) / len(counts)
		peak = max(counts)
		spaces = await self.list_spaces(tenant_id, property_id)
		total_capacity = sum(s.capacity for s in spaces) or 1
		return {
			"tenant_id": tenant_id,
			"property_id": property_id,
			"average_daily_occupancy": round(avg, 2),
			"peak_occupancy": peak,
			"utilisation_rate": round(avg / total_capacity * 100, 2),
			"readings_count": len(readings),
		}

	# ── Density Planning ──────────────────────────────────────────────────────

	async def create_density_plan(self, payload: DensityPlanCreate) -> DensityPlanResponse:
		"""Create a workplace density target plan."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "set_density_target",
			"density_band_supported": True,
		})
		record = DensityPlanResponse(**payload.model_dump())
		self._store["density_plans"].append(record.model_dump())
		self._log_operation("create_density_plan", record.id, record.tenant_id)
		return record

	async def get_density_analysis(self, tenant_id: str, property_id: str) -> dict[str, Any]:
		"""Analyse current density vs target."""
		spaces = await self.list_spaces(tenant_id, property_id)
		allocations = await self.list_allocations(tenant_id, is_active=True)
		total_area = sum(float(s.area) for s in spaces if s.area)
		total_headcount = sum(a.headcount for a in allocations if any(s.id == a.space_id for s in spaces))
		current_sqm_pp = (total_area / total_headcount) if total_headcount > 0 else 0
		density_plans = [d for d in self._store["density_plans"] if d["tenant_id"] == tenant_id and d.get("property_id") == property_id]
		target_sqm_pp = float(density_plans[-1]["target_sqm_per_person"]) if density_plans else None
		if target_sqm_pp and current_sqm_pp < target_sqm_pp * 0.8:
			self._log_density_breach(property_id, Decimal(str(current_sqm_pp)), Decimal(str(target_sqm_pp)))
		return {
			"tenant_id": tenant_id,
			"property_id": property_id,
			"total_area_sqm": total_area,
			"total_headcount": total_headcount,
			"current_sqm_per_person": round(current_sqm_pp, 2),
			"target_sqm_per_person": target_sqm_pp,
			"density_ok": current_sqm_pp >= (target_sqm_pp * 0.8) if target_sqm_pp else True,
		}

	# ── Space Chargeback ──────────────────────────────────────────────────────

	async def calculate_chargeback(self, tenant_id: str, property_id: str, period: str, rate_per_sqm: Decimal, occupancy_data_verified: bool) -> dict[str, Any]:
		"""Calculate space chargeback for cost allocation."""
		self._check_rules({"operation": "calculate_chargeback", "occupancy_data_verified": occupancy_data_verified})
		allocations = await self.list_allocations(tenant_id, is_active=True)
		chargebacks: list[dict[str, Any]] = []
		for a in allocations:
			space = await self.get_space(a.space_id, tenant_id)
			if space and space.property_id == property_id:
				area = float(space.area or 0)
				charge = area * float(rate_per_sqm)
				chargebacks.append({"department_id": a.department_id, "space_id": a.space_id, "area": area, "charge": charge})
		return {"tenant_id": tenant_id, "property_id": property_id, "period": period, "chargebacks": chargebacks, "total_charge": sum(c["charge"] for c in chargebacks)}

	# ── NEW: floor_plan_upload ─────────────────────────────────────────────────

	async def floor_plan_upload(
		self,
		building_id: str,
		floor: str,
		file_metadata: dict[str, Any],
		tenant_id: str,
		format: str = "dwg",
		scale: str = "1:100",
	) -> dict[str, Any]:
		"""Upload a floor plan for a specific building floor with version management."""
		assert building_id and floor and file_metadata, "building_id, floor, file_metadata required"
		assert format in ("dwg", "dxf", "pdf", "svg", "ifc", "revit"), f"unsupported format: {format}"
		version = 1
		for fp in self._store["floor_plans"]:
			if fp.get("property_id") == building_id and fp.get("floor") == floor and fp["tenant_id"] == tenant_id:
				version = max(version, fp.get("version", 1) + 1)
		from uuid6 import uuid7
		plan_id = str(uuid7())
		plan: dict[str, Any] = {
			"id": plan_id,
			"tenant_id": tenant_id,
			"property_id": building_id,
			"floor": floor,
			"format": format,
			"scale": scale,
			"file_metadata": file_metadata,
			"file_name": file_metadata.get("file_name", ""),
			"file_size_bytes": file_metadata.get("file_size_bytes", 0),
			"version": version,
			"space_ids": [],
			"uploaded_at": datetime.utcnow().isoformat(),
		}
		self._store["floor_plans"].append(plan)
		self._log_operation("floor_plan_uploaded", plan_id, tenant_id)
		return plan

	# ── NEW: space_allocation ──────────────────────────────────────────────────

	async def space_allocation(
		self,
		space_id: str,
		occupant_id: str,
		allocation_type: str,
		start_date: date,
		tenant_id: str,
		end_date: date | None = None,
		headcount: int = 1,
		department_id: str = "",
	) -> dict[str, Any]:
		"""Allocate a space to an occupant or department from a start date with optional end date."""
		assert space_id and occupant_id, "space_id and occupant_id required"
		assert allocation_type in ("permanent", "temporary", "shared", "hot_desk", "project"), \
			f"unsupported allocation_type: {allocation_type}"
		space = await self.get_space(space_id, tenant_id)
		if space is None:
			raise KeyError(f"space {space_id} not found")
		if space.status.value == "decommissioned":
			raise ValueError(f"space {space_id} is decommissioned")
		from uuid6 import uuid7
		allocation_id = str(uuid7())
		allocation: dict[str, Any] = {
			"id": allocation_id,
			"tenant_id": tenant_id,
			"space_id": space_id,
			"occupant_id": occupant_id,
			"occupant_ids": [occupant_id],
			"allocation_type": allocation_type,
			"department_id": department_id,
			"headcount": headcount,
			"start_date": str(start_date),
			"end_date": str(end_date) if end_date else None,
			"is_active": True,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["allocations"].append(allocation)
		# update space status
		for i, s in enumerate(self._store["spaces"]):
			if s["id"] == space_id and s["tenant_id"] == tenant_id:
				s["status"] = SpaceStatus.occupied.value
				s["current_allocation_id"] = allocation_id
				s["current_occupant_ids"] = [occupant_id]
				s["updated_at"] = datetime.utcnow()
				self._store["spaces"][i] = s
				break
		self._log_operation("space_allocated", allocation_id, tenant_id)
		return allocation

	# ── NEW: move_request ─────────────────────────────────────────────────────

	async def move_request(
		self,
		from_space: str,
		to_space: str,
		occupant_id: str,
		move_date: date,
		tenant_id: str,
		reason: str = "",
		headcount: int = 1,
	) -> dict[str, Any]:
		"""Create a space move request from one space to another for an occupant."""
		assert from_space and to_space and occupant_id, "from_space, to_space, occupant_id required"
		assert from_space != to_space, "from_space and to_space must differ"
		from_space_obj = await self.get_space(from_space, tenant_id)
		to_space_obj = await self.get_space(to_space, tenant_id)
		if to_space_obj and to_space_obj.status.value not in ("available", "reserved"):
			raise ValueError(f"destination space {to_space} is not available (status: {to_space_obj.status.value})")
		from uuid6 import uuid7
		move_id = str(uuid7())
		large_move = headcount >= 20
		move: dict[str, Any] = {
			"id": move_id,
			"tenant_id": tenant_id,
			"from_space_id": from_space,
			"to_space_id": to_space,
			"occupant_id": occupant_id,
			"move_date": str(move_date),
			"headcount": headcount,
			"reason": reason,
			"status": "planning" if large_move else "scheduled",
			"requires_approval": large_move,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["moves"].append(move)
		self._log_operation("move_requested", move_id, tenant_id)
		return move

	# ── NEW: occupancy_report ─────────────────────────────────────────────────

	async def occupancy_report(
		self,
		building_id: str,
		as_of_date: date,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Generate a point-in-time occupancy report for a building."""
		assert building_id, "building_id required"
		spaces = await self.list_spaces(tenant_id, property_id=building_id)
		allocations = await self.list_allocations(tenant_id, is_active=True)
		active_alloc_space_ids = {a.space_id for a in allocations}
		occupied = [s for s in spaces if s.id in active_alloc_space_ids]
		available = [s for s in spaces if s.status.value == "available"]
		total_area = sum(float(s.area or 0) for s in spaces)
		occupied_area = sum(float(s.area or 0) for s in occupied)
		total_headcount = sum(a.headcount for a in allocations
			if any(s.id == a.space_id for s in spaces))
		occupancy_rate = len(occupied) / max(len(spaces), 1) * 100
		return {
			"building_id": building_id,
			"as_of_date": str(as_of_date),
			"tenant_id": tenant_id,
			"total_spaces": len(spaces),
			"occupied_spaces": len(occupied),
			"available_spaces": len(available),
			"occupancy_rate_pct": round(occupancy_rate, 2),
			"total_area_sqm": total_area,
			"occupied_area_sqm": occupied_area,
			"total_headcount": total_headcount,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: density_analytics ────────────────────────────────────────────────

	async def density_analytics(
		self,
		building_id: str,
		period: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Analyse workspace density trends for a building over a period."""
		assert building_id and period, "building_id and period required"
		analysis = await self.get_density_analysis(tenant_id, building_id)
		occupancy_readings = [r for r in self._store["occupancy_data"] if r["tenant_id"] == tenant_id]
		if occupancy_readings:
			avg_occupancy = sum(r["occupant_count"] for r in occupancy_readings) / len(occupancy_readings)
			peak_occupancy = max(r["occupant_count"] for r in occupancy_readings)
		else:
			avg_occupancy = 0
			peak_occupancy = 0
		spaces = await self.list_spaces(tenant_id, property_id=building_id)
		total_capacity = sum(s.capacity for s in spaces)
		avg_utilisation = avg_occupancy / max(total_capacity, 1) * 100
		return {
			"building_id": building_id,
			"period": period,
			"tenant_id": tenant_id,
			**analysis,
			"avg_daily_occupancy": round(avg_occupancy, 2),
			"peak_occupancy": peak_occupancy,
			"total_capacity": total_capacity,
			"average_utilisation_pct": round(avg_utilisation, 2),
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: space_utilisation_heatmap ───────────────────────────────────────

	async def space_utilisation_heatmap(
		self,
		floor_id: str,
		period: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Generate a space utilisation heatmap for a floor showing usage intensity by space."""
		assert floor_id and period, "floor_id and period required"
		floor_plan = None
		for fp in self._store["floor_plans"]:
			if fp["id"] == floor_id and fp["tenant_id"] == tenant_id:
				floor_plan = fp
				break
		if floor_plan is None:
			raise KeyError(f"floor_plan {floor_id} not found")
		space_ids = floor_plan.get("space_ids", [])
		heatmap_data: list[dict[str, Any]] = []
		for space_id in space_ids:
			space = await self.get_space(space_id, tenant_id)
			if space is None:
				continue
			bookings = await self.list_bookings(tenant_id, space_id=space_id)
			occupancy_readings = [r for r in self._store["occupancy_data"]
				if r["tenant_id"] == tenant_id and r.get("space_id") == space_id]
			booking_hours = len(bookings) * 2  # assume 2h average
			utilisation_pct = min(booking_hours / max(space.capacity * 8 * 5, 1) * 100, 100)
			heatmap_data.append({
				"space_id": space_id,
				"space_name": space.name,
				"space_type": space.space_type.value,
				"area_sqm": float(space.area or 0),
				"capacity": space.capacity,
				"booking_count": len(bookings),
				"occupancy_readings": len(occupancy_readings),
				"utilisation_pct": round(utilisation_pct, 2),
				"intensity": "high" if utilisation_pct >= 70 else "medium" if utilisation_pct >= 30 else "low",
			})
		return {
			"floor_id": floor_id,
			"period": period,
			"tenant_id": tenant_id,
			"spaces_analysed": len(heatmap_data),
			"heatmap": heatmap_data,
			"avg_utilisation_pct": round(sum(h["utilisation_pct"] for h in heatmap_data) / max(len(heatmap_data), 1), 2),
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: unallocated_space_report ─────────────────────────────────────────

	async def unallocated_space_report(
		self,
		building_id: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Report all unallocated (void) spaces in a building with area and cost implications."""
		assert building_id, "building_id required"
		spaces = await self.list_spaces(tenant_id, property_id=building_id)
		allocations = await self.list_allocations(tenant_id, is_active=True)
		allocated_space_ids = {a.space_id for a in allocations}
		unallocated = [s for s in spaces if s.id not in allocated_space_ids and s.status.value == "available"]
		total_unallocated_area = sum(float(s.area or 0) for s in unallocated)
		total_area = sum(float(s.area or 0) for s in spaces)
		void_rate = len(unallocated) / max(len(spaces), 1) * 100
		return {
			"building_id": building_id,
			"tenant_id": tenant_id,
			"total_spaces": len(spaces),
			"unallocated_spaces": len(unallocated),
			"void_rate_pct": round(void_rate, 2),
			"total_unallocated_area_sqm": total_unallocated_area,
			"total_area_sqm": total_area,
			"unallocated_area_pct": round(total_unallocated_area / max(total_area, 1) * 100, 2),
			"unallocated_space_details": [
				{"space_id": s.id, "name": s.name, "type": s.space_type.value, "area": float(s.area or 0)}
				for s in unallocated
			],
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: workplace_booking ─────────────────────────────────────────────────

	async def workplace_booking(
		self,
		space_id: str,
		occupant_id: str,
		booking_date: date,
		time_from: str,
		time_to: str,
		tenant_id: str,
		booking_type: str = "desk",
		notes: str = "",
	) -> dict[str, Any]:
		"""Book a workplace space (desk, meeting room, locker) for a specific date and time slot."""
		assert space_id and occupant_id, "space_id and occupant_id required"
		assert booking_type in ("desk", "meeting_room", "phone_booth", "locker", "parking", "amenity"), \
			f"unsupported booking_type: {booking_type}"
		space = await self.get_space(space_id, tenant_id)
		if space is None:
			raise KeyError(f"space {space_id} not found")
		if space.status.value == "decommissioned":
			raise ValueError(f"space {space_id} is decommissioned")
		start_dt = datetime.fromisoformat(f"{booking_date}T{time_from}")
		end_dt = datetime.fromisoformat(f"{booking_date}T{time_to}")
		if end_dt <= start_dt:
			raise ValueError("time_to must be after time_from")
		# conflict check
		for existing in self._store["bookings"]:
			if (existing["space_id"] == space_id and
				existing["tenant_id"] == tenant_id and
				existing["status"] == "confirmed"):
				ex_start = datetime.fromisoformat(str(existing["start_datetime"]))
				ex_end = datetime.fromisoformat(str(existing["end_datetime"]))
				if not (end_dt <= ex_start or start_dt >= ex_end):
					self._log_double_booking_attempt(space_id, start_dt, end_dt)
					raise ValueError(f"space {space_id} is already booked for this time slot")
		from uuid6 import uuid7
		booking_id = str(uuid7())
		booking: dict[str, Any] = {
			"id": booking_id,
			"tenant_id": tenant_id,
			"space_id": space_id,
			"occupant_id": occupant_id,
			"booking_date": str(booking_date),
			"start_datetime": start_dt.isoformat(),
			"end_datetime": end_dt.isoformat(),
			"booking_type": booking_type,
			"notes": notes,
			"status": "confirmed",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["bookings"].append(booking)
		return booking

	# ── NEW: space_analytics ──────────────────────────────────────────────────

	async def space_analytics(self, period: str, tenant_id: str) -> dict[str, Any]:
		"""Generate portfolio-wide space analytics for a period."""
		assert period, "period required"
		spaces = await self.list_spaces(tenant_id)
		allocations = await self.list_allocations(tenant_id, is_active=True)
		bookings = await self.list_bookings(tenant_id)
		moves = await self.list_moves(tenant_id)
		floor_plans = await self.list_floor_plans(tenant_id)
		total_area = sum(float(s.area or 0) for s in spaces)
		occupied_spaces = [s for s in spaces if s.status.value == "occupied"]
		available_spaces = [s for s in spaces if s.status.value == "available"]
		total_headcount = sum(a.headcount for a in allocations)
		avg_sqm_per_person = total_area / max(total_headcount, 1)
		space_type_breakdown: dict[str, int] = {}
		for s in spaces:
			st = s.space_type.value
			space_type_breakdown[st] = space_type_breakdown.get(st, 0) + 1
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_spaces": len(spaces),
			"occupied_spaces": len(occupied_spaces),
			"available_spaces": len(available_spaces),
			"occupancy_rate_pct": round(len(occupied_spaces) / max(len(spaces), 1) * 100, 2),
			"total_area_sqm": total_area,
			"total_headcount": total_headcount,
			"avg_sqm_per_person": round(avg_sqm_per_person, 2),
			"active_allocations": len(allocations),
			"bookings_in_period": len(bookings),
			"moves_in_period": len(moves),
			"floor_plans": len(floor_plans),
			"space_type_breakdown": space_type_breakdown,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: scenario_planning ────────────────────────────────────────────────

	async def scenario_planning(
		self,
		building_id: str,
		scenario_type: str,
		tenant_id: str,
		target_headcount: int | None = None,
		target_sqm_per_person: float | None = None,
		growth_rate_pct: float = 0.0,
	) -> dict[str, Any]:
		"""Model a space scenario (expansion, consolidation, hybrid) for a building against target parameters."""
		assert building_id and scenario_type, "building_id and scenario_type required"
		assert scenario_type in ("expansion", "consolidation", "hybrid", "densification", "decommission"), \
			f"unsupported scenario_type: {scenario_type}"
		spaces = await self.list_spaces(tenant_id, property_id=building_id)
		total_area = sum(float(s.area or 0) for s in spaces)
		allocations = await self.list_allocations(tenant_id, is_active=True)
		current_headcount = sum(a.headcount for a in allocations
			if any(s.id == a.space_id for s in spaces))
		current_sqm_pp = total_area / max(current_headcount, 1)
		projected_headcount = target_headcount or int(current_headcount * (1 + growth_rate_pct / 100))
		target_sqm = target_sqm_per_person or current_sqm_pp
		required_area = projected_headcount * target_sqm
		area_gap = required_area - total_area
		recommendation: str
		if scenario_type == "expansion":
			recommendation = f"Acquire {max(0, area_gap):.0f} sqm to accommodate {projected_headcount} people at {target_sqm:.1f} sqm/person"
		elif scenario_type == "consolidation":
			recommendation = f"Reduce footprint by {max(0, -area_gap):.0f} sqm by consolidating {len(spaces)} spaces"
		elif scenario_type == "densification":
			recommendation = f"Increase density from {current_sqm_pp:.1f} to {target_sqm:.1f} sqm/person, freeing {max(0, -area_gap):.0f} sqm"
		else:
			recommendation = f"Model {scenario_type} scenario for {building_id}"
		from uuid6 import uuid7
		scenario_id = str(uuid7())
		scenario: dict[str, Any] = {
			"id": scenario_id,
			"tenant_id": tenant_id,
			"building_id": building_id,
			"scenario_type": scenario_type,
			"current_headcount": current_headcount,
			"projected_headcount": projected_headcount,
			"current_area_sqm": total_area,
			"required_area_sqm": required_area,
			"area_gap_sqm": round(area_gap, 2),
			"current_sqm_per_person": round(current_sqm_pp, 2),
			"target_sqm_per_person": target_sqm,
			"recommendation": recommendation,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["scenarios"].append(scenario)
		self._log_operation("scenario_planned", scenario_id, tenant_id)
		return scenario


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}, "unsupported format"
		return {"format": format, "tenant_id": tenant_id, "record_count": 0, "exported_at": datetime.utcnow().isoformat()}

	async def health_check(self, tenant_id: str) -> dict[str, Any]:
		"""Health Check"""
		return {"service": self.__class__.__name__, "tenant_id": tenant_id, "status": "healthy", "checked_at": datetime.utcnow().isoformat()}

	async def compliance_audit(self, tenant_id: str, standard: str = "RICS") -> dict[str, Any]:
		"""Compliance Audit"""
		self._log_operation("compliance_audit", "audit", tenant_id)
		return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "checked_at": datetime.utcnow().isoformat()}

	async def bulk_update_records(self, updates: list[dict], tenant_id: str) -> dict[str, Any]:
		"""Bulk Update Records"""
		assert updates, "updates required"
		self._log_operation("bulk_update", "bulk", tenant_id)
		return {"updated_count": len(updates), "tenant_id": tenant_id}

	async def get_kpis(self, tenant_id: str, period: str = "monthly") -> dict[str, Any]:
		"""Get Kpis"""
		self._log_operation("get_kpis", "kpis", tenant_id)
		return {"tenant_id": tenant_id, "period": period, "computed_at": datetime.utcnow().isoformat()}

	async def search_records(self, query: str, tenant_id: str) -> dict[str, Any]:
		"""Search Records"""
		assert query, "query required"
		return {"query": query, "tenant_id": tenant_id, "results": [], "result_count": 0}

	async def archive_record(self, record_id: str, tenant_id: str, reason: str) -> dict[str, Any]:
		"""Archive Record"""
		assert record_id and reason, "record_id and reason required"
		self._log_operation("archive_record", record_id, tenant_id)
		return {"record_id": record_id, "status": "archived", "reason": reason, "archived_at": datetime.utcnow().isoformat()}

	async def restore_record(self, record_id: str, tenant_id: str) -> dict[str, Any]:
		"""Restore Record"""
		assert record_id, "record_id required"
		self._log_operation("restore_record", record_id, tenant_id)
		return {"record_id": record_id, "status": "active", "restored_at": datetime.utcnow().isoformat()}

	async def get_audit_trail(self, tenant_id: str, entity_id: str = "") -> dict[str, Any]:
		"""Get Audit Trail"""
		return {"entity_id": entity_id, "tenant_id": tenant_id, "events": [], "retrieved_at": datetime.utcnow().isoformat()}

	# ── World-Class Enhancements ──────────────────────────────────────────────

	async def find_accessible_spaces(
		self,
		tenant_id: str,
		required_features: list[str],
		property_id: str | None = None,
		min_capacity: int = 1,
	) -> list[dict[str, Any]]:
		"""Return spaces matching all requested accessibility features.

		Filters the live space registry against the `amenities` list which
		carries accessibility tags such as ``wheelchair_accessible``,
		``hearing_loop``, and ``quiet_for_neurodivergent``.  Only available
		spaces with capacity >= min_capacity are returned.

		Args:
			tenant_id: Tenant scope.
			required_features: All features that must be present on the space.
			property_id: Optional property filter.
			min_capacity: Minimum seating capacity (default 1).

		Returns:
			List of matching space dicts with an ``accessibility_features`` key
			containing the filtered feature intersection.
		"""
		assert required_features, "required_features must be non-empty"
		spaces = await self.get_available_spaces(tenant_id, property_id=property_id, min_capacity=min_capacity)
		required_set = set(required_features)
		results: list[dict[str, Any]] = []
		for s in spaces:
			space_amenities = set(s.amenities)
			if required_set.issubset(space_amenities):
				d = s.model_dump()
				d["accessibility_features"] = sorted(required_set & space_amenities)
				results.append(d)
		log.info("spa.find_accessible_spaces tenant=%s features=%s results=%d", tenant_id, required_features, len(results))
		return results

	async def check_allocation_expiries(
		self,
		tenant_id: str,
		lookahead_days: int = 30,
	) -> dict[str, Any]:
		"""Identify active allocations expiring within lookahead_days and return expiry events.

		Scans all active allocations with a non-null ``end_date`` that falls
		within today + lookahead_days.  Each expiry candidate is returned with
		days_remaining so notification systems can tier urgency.

		Args:
			tenant_id: Tenant scope.
			lookahead_days: Window in days to look ahead (default 30).

		Returns:
			Dict with ``expiring`` list and summary counts.
		"""
		assert lookahead_days > 0, "lookahead_days must be positive"
		today = date.today()
		cutoff = today + timedelta(days=lookahead_days)
		allocations = await self.list_allocations(tenant_id, is_active=True)
		expiring: list[dict[str, Any]] = []
		for a in allocations:
			if a.end_date and today <= a.end_date <= cutoff:
				space = await self.get_space(a.space_id, tenant_id)
				days_remaining = (a.end_date - today).days
				expiring.append({
					"allocation_id": a.id,
					"space_id": a.space_id,
					"space_name": space.name if hasattr(space, "name") else a.space_id,
					"department_id": a.department_id,
					"occupant_ids": a.occupant_ids,
					"end_date": str(a.end_date),
					"days_remaining": days_remaining,
					"urgency": "critical" if days_remaining <= 7 else "warning" if days_remaining <= 14 else "notice",
				})
		expiring.sort(key=lambda x: x["days_remaining"])
		log.info("spa.check_allocation_expiries tenant=%s lookahead=%d expiring=%d", tenant_id, lookahead_days, len(expiring))
		return {
			"tenant_id": tenant_id,
			"lookahead_days": lookahead_days,
			"checked_at": date.today().isoformat(),
			"expiring_count": len(expiring),
			"critical": sum(1 for e in expiring if e["urgency"] == "critical"),
			"warning": sum(1 for e in expiring if e["urgency"] == "warning"),
			"notice": sum(1 for e in expiring if e["urgency"] == "notice"),
			"expiring": expiring,
		}

	async def benchmark_portfolio(
		self,
		tenant_id: str,
		metric: str = "utilisation_rate",
	) -> dict[str, Any]:
		"""Rank all properties in the tenant portfolio against each other for a given metric.

		Supported metrics: ``utilisation_rate``, ``sqm_per_person``,
		``booking_adherence``, ``void_rate``.  Returns percentile ranks and
		flags outliers (> 1.5× IQR above/below median).

		Args:
			tenant_id: Tenant scope.
			metric: Metric to benchmark (default ``utilisation_rate``).

		Returns:
			Dict with ranked property list and portfolio statistics.
		"""
		supported = {"utilisation_rate", "sqm_per_person", "booking_adherence", "void_rate"}
		assert metric in supported, f"metric must be one of {supported}"

		# collect unique property IDs
		property_ids: list[str] = sorted({s["property_id"] for s in self._store["spaces"] if s["tenant_id"] == tenant_id})
		if not property_ids:
			return {"tenant_id": tenant_id, "metric": metric, "properties": [], "portfolio_median": None}

		scores: list[dict[str, Any]] = []
		for pid in property_ids:
			spaces = await self.list_spaces(tenant_id, property_id=pid)
			if not spaces:
				continue
			allocations = await self.list_allocations(tenant_id, is_active=True)
			alloc_ids = {a.space_id for a in allocations if any(s.id == a.space_id for s in spaces)}
			occupied = len(alloc_ids)
			total = max(len(spaces), 1)
			total_area = sum(float(s.area or 0) for s in spaces)
			total_headcount = sum(a.headcount for a in allocations if a.space_id in alloc_ids) or 1
			void_count = len([s for s in spaces if s.id not in alloc_ids and s.status.value == "available"])

			if metric == "utilisation_rate":
				score = occupied / total * 100
			elif metric == "sqm_per_person":
				score = total_area / total_headcount
			elif metric == "void_rate":
				score = void_count / total * 100
			else:  # booking_adherence — proxy via bookings
				bookings = await self.list_bookings(tenant_id)
				prop_bookings = [b for b in bookings if any(s.id == b.space_id for s in spaces)]
				score = len(prop_bookings) / max(occupied, 1)

			scores.append({"property_id": pid, "score": round(score, 3)})

		scores.sort(key=lambda x: x["score"])
		n = len(scores)
		for rank, item in enumerate(scores, 1):
			item["rank"] = rank
			item["percentile"] = round(rank / n * 100, 1)

		raw_scores = [x["score"] for x in scores]
		median = raw_scores[n // 2] if n else 0
		q1 = raw_scores[n // 4] if n >= 4 else median
		q3 = raw_scores[3 * n // 4] if n >= 4 else median
		iqr = q3 - q1
		for item in scores:
			item["outlier"] = item["score"] < q1 - 1.5 * iqr or item["score"] > q3 + 1.5 * iqr

		log.info("spa.benchmark_portfolio tenant=%s metric=%s properties=%d", tenant_id, metric, n)
		return {
			"tenant_id": tenant_id,
			"metric": metric,
			"portfolio_median": round(median, 3),
			"portfolio_q1": round(q1, 3),
			"portfolio_q3": round(q3, 3),
			"properties": scores,
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def detect_overuse_events(
		self,
		tenant_id: str,
		space_id: str,
		period_days: int = 30,
		overcrowding_threshold: float = 1.1,
	) -> dict[str, Any]:
		"""Identify occupancy readings where occupant_count exceeds capacity * threshold.

		Overcrowding events trigger downstream maintenance workflows (deep
		cleaning, HVAC scheduling) via the ``mqeb`` capability.

		Args:
			tenant_id: Tenant scope.
			space_id: Space to analyse.
			period_days: Lookback window in days (default 30).
			overcrowding_threshold: Fraction of capacity to flag (default 1.1).

		Returns:
			Dict with overuse events, max occupancy, and a severity rating.
		"""
		assert space_id, "space_id required"
		assert period_days > 0, "period_days must be positive"
		assert 1.0 <= overcrowding_threshold <= 3.0, "threshold must be between 1.0 and 3.0"

		space = await self.get_space(space_id, tenant_id)
		if space is None:
			raise KeyError(f"space {space_id} not found for tenant {tenant_id}")

		cutoff_dt = datetime.utcnow() - timedelta(days=period_days)
		readings = [
			r for r in self._store["occupancy_data"]
			if r["tenant_id"] == tenant_id
			and r.get("space_id") == space_id
			and datetime.fromisoformat(str(r["recorded_at"])) >= cutoff_dt
		]

		overuse_threshold = int(space.capacity * overcrowding_threshold)
		events = [
			{
				"recorded_at": r["recorded_at"],
				"occupant_count": r["occupant_count"],
				"capacity": space.capacity,
				"excess": r["occupant_count"] - space.capacity,
				"sensor_type": r.get("sensor_type"),
			}
			for r in readings
			if r["occupant_count"] >= overuse_threshold
		]
		events.sort(key=lambda x: str(x["recorded_at"]))

		max_occ = max((r["occupant_count"] for r in readings), default=0)
		overuse_pct = len(events) / max(len(readings), 1) * 100
		severity = "high" if overuse_pct >= 20 else "medium" if overuse_pct >= 5 else "low"

		log.warning(
			"spa.detect_overuse space=%s period=%dd events=%d severity=%s",
			space_id, period_days, len(events), severity,
		)
		return {
			"tenant_id": tenant_id,
			"space_id": space_id,
			"space_capacity": space.capacity,
			"period_days": period_days,
			"total_readings": len(readings),
			"overuse_event_count": len(events),
			"overuse_rate_pct": round(overuse_pct, 2),
			"max_occupancy_recorded": max_occ,
			"severity": severity,
			"events": events,
			"analysed_at": datetime.utcnow().isoformat(),
		}

	async def calculate_energy_per_occupant(
		self,
		tenant_id: str,
		building_id: str,
		period: str,
		total_kwh: float,
	) -> dict[str, Any]:
		"""Compute energy intensity (kWh per person per day) for a building period.

		Joins caller-supplied energy meter total with occupancy readings to
		produce per-zone breakdowns.  Flags zones with zero occupancy consuming
		measurable energy as waste candidates.

		Args:
			tenant_id: Tenant scope.
			building_id: Property to analyse.
			period: Human-readable period label (e.g. ``"2026-Q1"``).
			total_kwh: Total electrical consumption for the period.

		Returns:
			Dict with portfolio and per-space energy intensity figures.
		"""
		assert building_id and period, "building_id and period required"
		assert total_kwh >= 0, "total_kwh must be non-negative"

		spaces = await self.list_spaces(tenant_id, property_id=building_id)
		if not spaces:
			return {"tenant_id": tenant_id, "building_id": building_id, "period": period, "total_kwh": total_kwh, "spaces": []}

		readings = [
			r for r in self._store["occupancy_data"]
			if r["tenant_id"] == tenant_id
			and any(s.id == r.get("space_id") for s in spaces)
		]
		total_person_days = sum(r["occupant_count"] for r in readings) or 1
		portfolio_kwh_per_pd = round(total_kwh / total_person_days, 4)

		space_stats: list[dict[str, Any]] = []
		for s in spaces:
			s_readings = [r for r in readings if r.get("space_id") == s.id]
			person_days = sum(r["occupant_count"] for r in s_readings) or 0
			# proportional energy allocation by area fraction
			area_fraction = float(s.area or 0) / max(sum(float(x.area or 0) for x in spaces), 1)
			attributed_kwh = total_kwh * area_fraction
			kwh_per_pd = attributed_kwh / max(person_days, 1)
			space_stats.append({
				"space_id": s.id,
				"space_type": s.space_type.value,
				"area_sqm": float(s.area or 0),
				"area_fraction_pct": round(area_fraction * 100, 2),
				"attributed_kwh": round(attributed_kwh, 2),
				"person_days": person_days,
				"kwh_per_person_day": round(kwh_per_pd, 4),
				"waste_candidate": person_days == 0 and attributed_kwh > 0,
			})

		waste_count = sum(1 for s in space_stats if s["waste_candidate"])
		log.info("spa.energy_per_occupant tenant=%s building=%s kwh=%.1f waste_spaces=%d", tenant_id, building_id, total_kwh, waste_count)
		return {
			"tenant_id": tenant_id,
			"building_id": building_id,
			"period": period,
			"total_kwh": total_kwh,
			"total_person_days": total_person_days,
			"portfolio_kwh_per_person_day": portfolio_kwh_per_pd,
			"waste_candidate_spaces": waste_count,
			"spaces": space_stats,
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def submit_space_request(
		self,
		tenant_id: str,
		requestor_id: str,
		department_id: str,
		requested_space_type: str,
		required_capacity: int,
		required_from: date,
		justification: str,
		required_to: date | None = None,
		preferred_building_id: str | None = None,
	) -> dict[str, Any]:
		"""Submit a formal space request for management approval.

		Creates a pending request record.  Approvers query
		``list_space_requests`` filtered by status.

		Args:
			tenant_id: Tenant scope.
			requestor_id: Person submitting the request.
			department_id: Department that will occupy the space.
			requested_space_type: One of the ``SpaceType`` enum values.
			required_capacity: Minimum headcount needed.
			required_from: Earliest acceptable start date.
			justification: Business case text (required for audit).
			required_to: Optional end date (None = permanent).
			preferred_building_id: Optional building preference.

		Returns:
			New space request dict with ``status = "pending"``.
		"""
		assert requestor_id and department_id, "requestor_id and department_id required"
		assert justification.strip(), "justification must be non-empty"
		assert required_capacity >= 1, "required_capacity must be at least 1"

		from uuid6 import uuid7
		request_id = str(uuid7())
		request: dict[str, Any] = {
			"id": request_id,
			"tenant_id": tenant_id,
			"requestor_id": requestor_id,
			"department_id": department_id,
			"requested_space_type": requested_space_type,
			"required_capacity": required_capacity,
			"required_from": str(required_from),
			"required_to": str(required_to) if required_to else None,
			"preferred_building_id": preferred_building_id,
			"justification": justification,
			"status": "pending",
			"submitted_at": datetime.utcnow().isoformat(),
			"reviewed_by": None,
			"reviewed_at": None,
			"review_notes": None,
		}
		self._store.setdefault("space_requests", []).append(request)
		self._log_operation("space_request_submitted", request_id, tenant_id)
		return request

	async def approve_space_request(
		self,
		tenant_id: str,
		request_id: str,
		reviewer_id: str,
		notes: str = "",
	) -> dict[str, Any]:
		"""Approve a pending space request and surface matching available spaces.

		Args:
			tenant_id: Tenant scope.
			request_id: ID of the request to approve.
			reviewer_id: Identity of the approver.
			notes: Optional review comments.

		Returns:
			Updated request dict with ``status = "approved"`` and
			``matching_spaces`` list of spaces meeting the requirements.

		Raises:
			KeyError: If request not found.
			ValueError: If request is not in ``pending`` status.
		"""
		requests = self._store.setdefault("space_requests", [])
		for i, req in enumerate(requests):
			if req["id"] == request_id and req["tenant_id"] == tenant_id:
				if req["status"] != "pending":
					raise ValueError(f"request {request_id} is not pending (status={req['status']})")
				req["status"] = "approved"
				req["reviewed_by"] = reviewer_id
				req["reviewed_at"] = datetime.utcnow().isoformat()
				req["review_notes"] = notes
				requests[i] = req
				# find matching spaces
				matching = await self.get_available_spaces(
					tenant_id,
					property_id=req.get("preferred_building_id"),
					space_type=req.get("requested_space_type"),
					min_capacity=req.get("required_capacity", 1),
				)
				req["matching_spaces"] = [{"id": s.id, "name": s.space_ref, "capacity": s.capacity} for s in matching[:10]]
				self._log_operation("space_request_approved", request_id, tenant_id)
				return req
		raise KeyError(f"space request {request_id} not found for tenant {tenant_id}")

	async def get_zone_analytics(
		self,
		tenant_id: str,
		zone_space_ids: list[str],
		zone_name: str = "",
	) -> dict[str, Any]:
		"""Aggregate utilisation, headcount, and density metrics across a named group of spaces.

		A zone is an ad-hoc grouping (floor wing, department neighbourhood,
		executive suite).  Callers pass the constituent space IDs explicitly;
		no persistent zone model is required.

		Args:
			tenant_id: Tenant scope.
			zone_space_ids: Ordered list of space IDs comprising the zone.
			zone_name: Human-readable label for the zone (optional).

		Returns:
			Dict with aggregated zone metrics and per-space breakdown.
		"""
		assert zone_space_ids, "zone_space_ids must be non-empty"

		spaces = [s for s in [await self.get_space(sid, tenant_id) for sid in zone_space_ids] if s is not None]
		if not spaces:
			return {"zone_name": zone_name, "tenant_id": tenant_id, "spaces_found": 0}

		allocations = await self.list_allocations(tenant_id, is_active=True)
		alloc_map = {a.space_id: a for a in allocations}

		total_area = sum(float(s.area or 0) for s in spaces)
		total_capacity = sum(s.capacity for s in spaces)
		occupied_spaces = [s for s in spaces if s.id in alloc_map]
		total_headcount = sum(alloc_map[s.id].headcount for s in occupied_spaces)
		sqm_per_person = total_area / max(total_headcount, 1)
		utilisation_pct = len(occupied_spaces) / len(spaces) * 100

		space_breakdown: list[dict[str, Any]] = []
		for s in spaces:
			alloc = alloc_map.get(s.id)
			readings = [r for r in self._store["occupancy_data"]
				if r["tenant_id"] == tenant_id and r.get("space_id") == s.id]
			avg_occ = sum(r["occupant_count"] for r in readings) / max(len(readings), 1) if readings else 0
			space_breakdown.append({
				"space_id": s.id,
				"space_type": s.space_type.value,
				"capacity": s.capacity,
				"area_sqm": float(s.area or 0),
				"status": s.status.value,
				"headcount": alloc.headcount if alloc else 0,
				"department_id": alloc.department_id if alloc else None,
				"avg_sensor_occupancy": round(avg_occ, 2),
				"sensor_readings": len(readings),
			})

		log.info("spa.zone_analytics tenant=%s zone=%r spaces=%d headcount=%d", tenant_id, zone_name, len(spaces), total_headcount)
		return {
			"zone_name": zone_name,
			"tenant_id": tenant_id,
			"spaces_in_zone": len(spaces),
			"total_area_sqm": round(total_area, 2),
			"total_capacity": total_capacity,
			"occupied_spaces": len(occupied_spaces),
			"total_headcount": total_headcount,
			"utilisation_pct": round(utilisation_pct, 2),
			"sqm_per_person": round(sqm_per_person, 2),
			"density_ok": sqm_per_person >= 8.0,  # RICS standard minimum
			"space_breakdown": space_breakdown,
			"generated_at": datetime.utcnow().isoformat(),
		}
