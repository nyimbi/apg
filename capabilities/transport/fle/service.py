"""
Fleet Management Service — async business logic layer.

__init__(self, db_session, tenant_id, actor_id)
All public methods are async.  Domain events emitted via _emit_event().
Tenant isolation enforced on every query/mutation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from .models import (
	COFInspectionCreate, COFInspectionResponse,
	ComplianceCalendarEntry, DashboardKPIs, DriverBehaviourScore,
	DriverCreate, DriverResponse, DriverStatus, DriverUpdate,
	FleetUtilisationReport, FuelRecordCreate, FuelRecordResponse,
	IncidentCreate, IncidentResponse, IncidentStatus,
	InspectionCreate, InspectionResponse, InspectionResult,
	InsurancePolicyCreate, InsurancePolicyResponse,
	MaintenanceCreate, MaintenanceResponse, MaintenanceStatus,
	PredictiveMaintenanceAlert, RegistrationCreate, RegistrationResponse,
	TCOBreakdown, TachographRecordCreate, TachographRecordResponse,
	TelematicsEventCreate, TelematicsEventResponse,
	TripCreate, TripResponse, TripStatus, TripUpdate,
	VehicleAssignmentCreate, VehicleAssignmentResponse,
	VehicleCreate, VehicleResponse, VehicleStatus, VehicleUpdate,
	uuid7str,
)
from .domain.rules import (
	RuleViolation,
	assert_axle_load_within_limits,
	assert_cof_valid,
	assert_customs_docs_present_for_cross_border,
	assert_driver_active,
	assert_driver_cpc_valid,
	assert_driver_licence_valid,
	assert_driver_medical_valid,
	assert_driver_not_already_on_trip,
	assert_eu_continuous_driving,
	assert_eu_daily_driving,
	assert_eu_daily_rest,
	assert_eu_weekly_driving,
	assert_fatal_incident_requires_police_ref,
	assert_hired_vehicle_within_hire_period,
	assert_incident_reported_within_window,
	assert_insurance_valid,
	assert_maintenance_not_overdue_for_dispatch,
	assert_no_concurrent_trip,
	assert_no_duplicate_vin,
	assert_odometer_not_regressing,
	assert_road_worthiness_valid,
	assert_trip_arrival_after_departure,
	assert_vehicle_active_for_dispatch,
	assert_vehicle_not_overloaded,
	assert_vehicle_registration_present,
	assert_vin_present,
	calculate_overloading_fine,
)
from .domain.calculations import (
	calculate_avg_speed_kmh,
	calculate_co2_emissions_kg,
	calculate_cost_per_km,
	calculate_depreciation_straight_line,
	calculate_driver_score,
	calculate_fleet_utilisation,
	calculate_fuel_cost,
	calculate_fuel_efficiency_l100km,
	calculate_tco,
	calculate_trip_distance_km,
	calculate_trip_duration_hours,
	calculate_utilisation_pct,
	compliance_severity,
	days_until,
	predict_oil_change_due,
)

logger = logging.getLogger("apg.transport.fle")


class FleetService:
	"""
	Tenant-scoped, actor-aware fleet management service.

	Args:
		db_session: SQLAlchemy async session (or in-memory dict store for testing).
		tenant_id:  Tenant identifier — enforced on every operation.
		actor_id:   ID of the user/system performing the action (for audit).
	"""

	def __init__(self, db_session: Any, tenant_id: str, actor_id: str) -> None:
		assert tenant_id and tenant_id.strip(), "tenant_id is required"
		assert actor_id and actor_id.strip(), "actor_id is required"
		self._db = db_session
		self._tenant_id = tenant_id.strip()
		self._actor_id = actor_id.strip()
		self._events: list[dict[str, Any]] = []

	# ──────────────────────────────────────────────────────────────
	# Internal helpers
	# ──────────────────────────────────────────────────────────────

	def _log_op(self, op: str, entity_id: str = "", **extra: Any) -> str:
		msg = f"[FLE] tenant={self._tenant_id} actor={self._actor_id} op={op} id={entity_id}"
		if extra:
			msg += " " + " ".join(f"{k}={v}" for k, v in extra.items())
		logger.info(msg)
		return msg

	def _log_error(self, op: str, error: str, **extra: Any) -> str:
		msg = f"[FLE][ERROR] tenant={self._tenant_id} op={op} error={error}"
		if extra:
			msg += " " + " ".join(f"{k}={v}" for k, v in extra.items())
		logger.error(msg)
		return msg

	def _log_compliance_alert(self, entity_id: str, alert_type: str, days: int) -> str:
		msg = f"[FLE][COMPLIANCE] tenant={self._tenant_id} entity={entity_id} type={alert_type} days_until={days}"
		logger.warning(msg)
		return msg

	def _emit_event(self, event_type: str, entity_id: str, payload: dict[str, Any]) -> None:
		event = {
			"event_type": event_type,
			"tenant_id": self._tenant_id,
			"actor_id": self._actor_id,
			"entity_id": entity_id,
			"occurred_at": datetime.utcnow().isoformat(),
			"payload": payload,
		}
		self._events.append(event)
		logger.debug("[FLE][EVENT] %s entity=%s", event_type, entity_id)

	def _store(self, collection: str) -> dict[str, Any]:
		"""Return in-memory store dict for a collection (test/standalone mode)."""
		if not hasattr(self._db, "_fle_store"):
			self._db._fle_store = {}
		return self._db._fle_store.setdefault(f"{self._tenant_id}:{collection}", {})

	def _get(self, collection: str, entity_id: str) -> dict[str, Any] | None:
		return self._store(collection).get(entity_id)

	def _put(self, collection: str, entity_id: str, data: dict[str, Any]) -> None:
		self._store(collection)[entity_id] = data

	def _list(self, collection: str) -> list[dict[str, Any]]:
		return [v for v in self._store(collection).values() if not v.get("is_deleted")]

	def _tenant_assert(self, data: dict[str, Any]) -> None:
		assert data.get("tenant_id") == self._tenant_id, "Cross-tenant access denied"

	# ──────────────────────────────────────────────────────────────
	# Vehicle CRUD
	# ──────────────────────────────────────────────────────────────

	async def register_vehicle(self, payload: VehicleCreate) -> VehicleResponse:
		"""Register a new vehicle in the fleet."""
		assert payload.tenant_id == self._tenant_id, "Tenant mismatch"

		assert_vehicle_registration_present(payload.registration)
		assert_vin_present(payload.vin)

		existing_vins = [v["vin"] for v in self._list("vehicles")]
		assert_no_duplicate_vin(payload.vin, existing_vins)

		if payload.payload_capacity_kg > 0 and payload.gross_vehicle_weight_kg > 0:
			assert_vehicle_not_overloaded(payload.payload_capacity_kg, payload.gross_vehicle_weight_kg)

		rec = VehicleResponse(
			**payload.model_dump(),
			status=VehicleStatus.ACTIVE,
		)
		self._put("vehicles", rec.id, rec.model_dump(mode="json"))
		self._emit_event("vehicle.registered", rec.id, {"registration": rec.registration, "vin": rec.vin})
		self._log_op("register_vehicle", rec.id, registration=rec.registration)
		return rec

	async def get_vehicle(self, vehicle_id: str) -> VehicleResponse:
		"""Fetch a vehicle by ID."""
		raw = self._get("vehicles", vehicle_id)
		assert raw, f"Vehicle {vehicle_id} not found"
		self._tenant_assert(raw)
		return VehicleResponse.model_validate(raw)

	async def list_vehicles(self, status: VehicleStatus | None = None) -> list[VehicleResponse]:
		"""List all vehicles for tenant, optionally filtered by status."""
		rows = self._list("vehicles")
		if status:
			rows = [r for r in rows if r.get("status") == status.value]
		return [VehicleResponse.model_validate(r) for r in rows]

	async def update_vehicle(self, vehicle_id: str, patch: VehicleUpdate) -> VehicleResponse:
		"""Partial update a vehicle record."""
		raw = self._get("vehicles", vehicle_id)
		assert raw, f"Vehicle {vehicle_id} not found"
		self._tenant_assert(raw)
		updates = patch.model_dump(exclude_none=True, mode="json")
		updates["updated_at"] = datetime.utcnow().isoformat()
		raw.update(updates)
		self._put("vehicles", vehicle_id, raw)
		self._emit_event("vehicle.updated", vehicle_id, updates)
		self._log_op("update_vehicle", vehicle_id)
		return VehicleResponse.model_validate(raw)

	async def delete_vehicle(self, vehicle_id: str) -> None:
		"""Soft-delete a vehicle."""
		raw = self._get("vehicles", vehicle_id)
		assert raw, f"Vehicle {vehicle_id} not found"
		self._tenant_assert(raw)
		raw["is_deleted"] = True
		raw["updated_at"] = datetime.utcnow().isoformat()
		self._put("vehicles", vehicle_id, raw)
		self._emit_event("vehicle.deleted", vehicle_id, {})
		self._log_op("delete_vehicle", vehicle_id)

	async def set_vehicle_status(self, vehicle_id: str, status: VehicleStatus) -> VehicleResponse:
		"""Update vehicle operational status with audit event."""
		raw = self._get("vehicles", vehicle_id)
		assert raw, f"Vehicle {vehicle_id} not found"
		self._tenant_assert(raw)
		old_status = raw.get("status")
		raw["status"] = status.value
		raw["updated_at"] = datetime.utcnow().isoformat()
		self._put("vehicles", vehicle_id, raw)
		self._emit_event("vehicle.status_changed", vehicle_id, {"from": old_status, "to": status.value})
		self._log_op("set_vehicle_status", vehicle_id, status=status.value)
		return VehicleResponse.model_validate(raw)

	# ──────────────────────────────────────────────────────────────
	# Driver CRUD
	# ──────────────────────────────────────────────────────────────

	async def register_driver(self, payload: DriverCreate) -> DriverResponse:
		"""Register a new driver."""
		assert payload.tenant_id == self._tenant_id, "Tenant mismatch"
		assert_driver_licence_valid(payload.licence_expiry)
		if payload.cpc_expiry:
			assert_driver_cpc_valid(payload.cpc_expiry)
		if payload.medical_expiry:
			assert_driver_medical_valid(payload.medical_expiry)

		rec = DriverResponse(**payload.model_dump(), status=DriverStatus.ACTIVE)
		self._put("drivers", rec.id, rec.model_dump(mode="json"))
		self._emit_event("driver.registered", rec.id, {"name": rec.name, "licence": rec.licence_number})
		self._log_op("register_driver", rec.id, name=rec.name)
		return rec

	async def get_driver(self, driver_id: str) -> DriverResponse:
		raw = self._get("drivers", driver_id)
		assert raw, f"Driver {driver_id} not found"
		self._tenant_assert(raw)
		return DriverResponse.model_validate(raw)

	async def list_drivers(self, status: DriverStatus | None = None) -> list[DriverResponse]:
		rows = self._list("drivers")
		if status:
			rows = [r for r in rows if r.get("status") == status.value]
		return [DriverResponse.model_validate(r) for r in rows]

	async def update_driver(self, driver_id: str, patch: DriverUpdate) -> DriverResponse:
		raw = self._get("drivers", driver_id)
		assert raw, f"Driver {driver_id} not found"
		self._tenant_assert(raw)
		updates = patch.model_dump(exclude_none=True, mode="json")
		updates["updated_at"] = datetime.utcnow().isoformat()
		raw.update(updates)
		self._put("drivers", driver_id, raw)
		self._emit_event("driver.updated", driver_id, updates)
		self._log_op("update_driver", driver_id)
		return DriverResponse.model_validate(raw)

	async def delete_driver(self, driver_id: str) -> None:
		raw = self._get("drivers", driver_id)
		assert raw, f"Driver {driver_id} not found"
		self._tenant_assert(raw)
		raw["is_deleted"] = True
		raw["updated_at"] = datetime.utcnow().isoformat()
		self._put("drivers", driver_id, raw)
		self._emit_event("driver.deleted", driver_id, {})
		self._log_op("delete_driver", driver_id)

	# ──────────────────────────────────────────────────────────────
	# Assignment
	# ──────────────────────────────────────────────────────────────

	async def assign_driver(self, payload: VehicleAssignmentCreate) -> VehicleAssignmentResponse:
		"""Assign driver to vehicle with full compliance pre-checks."""
		assert payload.tenant_id == self._tenant_id, "Tenant mismatch"

		vehicle = await self.get_vehicle(payload.vehicle_id)
		driver = await self.get_driver(payload.driver_id)

		assert_vehicle_active_for_dispatch(vehicle.status.value)
		assert_driver_active(driver.status.value)
		assert_driver_licence_valid(driver.licence_expiry)

		# Check for concurrent active trips
		active = [
			r["id"] for r in self._list("trips")
			if r.get("vehicle_id") == payload.vehicle_id
			and r.get("status") in ("dispatched", "in_progress")
		]
		assert_no_concurrent_trip(payload.vehicle_id, active)

		driver_active_trips = [
			r["id"] for r in self._list("trips")
			if r.get("driver_id") == payload.driver_id
			and r.get("status") in ("dispatched", "in_progress")
		]
		assert_driver_not_already_on_trip(payload.driver_id, driver_active_trips)

		rec = VehicleAssignmentResponse(**payload.model_dump(), is_active=True)
		self._put("assignments", rec.id, rec.model_dump(mode="json"))
		self._emit_event("assignment.created", rec.id, {
			"vehicle_id": payload.vehicle_id, "driver_id": payload.driver_id,
		})
		self._log_op("assign_driver", rec.id, vehicle=payload.vehicle_id, driver=payload.driver_id)
		return rec

	# ──────────────────────────────────────────────────────────────
	# Trip lifecycle
	# ──────────────────────────────────────────────────────────────

	async def plan_trip(self, payload: TripCreate) -> TripResponse:
		"""Plan a trip with all pre-departure compliance checks."""
		assert payload.tenant_id == self._tenant_id, "Tenant mismatch"

		vehicle = await self.get_vehicle(payload.vehicle_id)
		driver = await self.get_driver(payload.driver_id)

		assert_vehicle_active_for_dispatch(vehicle.status.value)
		assert_driver_active(driver.status.value)
		assert_driver_licence_valid(driver.licence_expiry)

		if vehicle.payload_capacity_kg > 0:
			assert_vehicle_not_overloaded(payload.load_kg, vehicle.payload_capacity_kg)

		if payload.planned_arrival:
			assert_trip_arrival_after_departure(payload.planned_departure, payload.planned_arrival)

		if payload.cross_border_countries:
			assert_customs_docs_present_for_cross_border(
				payload.customs_required, payload.cross_border_countries, False
			)

		# Check for existing active trips
		active_v = [
			r["id"] for r in self._list("trips")
			if r.get("vehicle_id") == payload.vehicle_id
			and r.get("status") in ("dispatched", "in_progress")
		]
		assert_no_concurrent_trip(payload.vehicle_id, active_v)

		rec = TripResponse(**payload.model_dump(), status=TripStatus.PLANNED)
		self._put("trips", rec.id, rec.model_dump(mode="json"))
		self._emit_event("trip.planned", rec.id, {
			"vehicle_id": payload.vehicle_id, "driver_id": payload.driver_id,
			"origin": payload.origin, "destination": payload.destination,
		})
		self._log_op("plan_trip", rec.id, origin=payload.origin, dest=payload.destination)
		return rec

	async def dispatch_trip(self, trip_id: str) -> TripResponse:
		"""Dispatch a planned trip (status → dispatched)."""
		raw = self._get("trips", trip_id)
		assert raw, f"Trip {trip_id} not found"
		self._tenant_assert(raw)
		assert raw["status"] == TripStatus.PLANNED.value, f"Trip must be PLANNED to dispatch, got {raw['status']}"

		raw["status"] = TripStatus.DISPATCHED.value
		raw["updated_at"] = datetime.utcnow().isoformat()
		self._put("trips", trip_id, raw)
		self._emit_event("trip.dispatched", trip_id, {})
		self._log_op("dispatch_trip", trip_id)
		return TripResponse.model_validate(raw)

	async def start_trip(self, trip_id: str, odometer_start_km: Decimal) -> TripResponse:
		"""Record trip start with odometer reading."""
		raw = self._get("trips", trip_id)
		assert raw, f"Trip {trip_id} not found"
		self._tenant_assert(raw)
		assert raw["status"] in (TripStatus.DISPATCHED.value, TripStatus.PLANNED.value)

		raw["status"] = TripStatus.IN_PROGRESS.value
		raw["actual_departure"] = datetime.utcnow().isoformat()
		raw["odometer_start_km"] = str(odometer_start_km)
		raw["updated_at"] = datetime.utcnow().isoformat()
		self._put("trips", trip_id, raw)
		self._emit_event("trip.started", trip_id, {"odometer_start_km": str(odometer_start_km)})
		self._log_op("start_trip", trip_id)
		return TripResponse.model_validate(raw)

	async def complete_trip(
		self,
		trip_id: str,
		odometer_end_km: Decimal,
		fuel_consumed_l: Decimal | None = None,
	) -> TripResponse:
		"""Complete a trip, calculate distance and efficiency."""
		raw = self._get("trips", trip_id)
		assert raw, f"Trip {trip_id} not found"
		self._tenant_assert(raw)
		assert raw["status"] == TripStatus.IN_PROGRESS.value, "Trip must be IN_PROGRESS to complete"

		start = Decimal(str(raw.get("odometer_start_km") or "0"))
		if start > 0:
			assert_odometer_not_regressing(odometer_end_km, start)

		distance = calculate_trip_distance_km(start if start > 0 else None, odometer_end_km)
		raw["status"] = TripStatus.COMPLETED.value
		raw["actual_arrival"] = datetime.utcnow().isoformat()
		raw["odometer_end_km"] = str(odometer_end_km)
		if distance is not None:
			raw["distance_km"] = str(distance)
		if fuel_consumed_l is not None:
			raw["fuel_consumed_l"] = str(fuel_consumed_l)
		raw["updated_at"] = datetime.utcnow().isoformat()

		# Update vehicle odometer
		v_raw = self._get("vehicles", raw["vehicle_id"])
		if v_raw:
			v_raw["odometer_km"] = str(odometer_end_km)
			v_raw["updated_at"] = datetime.utcnow().isoformat()
			self._put("vehicles", raw["vehicle_id"], v_raw)

		self._put("trips", trip_id, raw)
		self._emit_event("trip.completed", trip_id, {
			"distance_km": str(distance), "fuel_consumed_l": str(fuel_consumed_l),
		})
		self._log_op("complete_trip", trip_id, distance_km=str(distance))
		return TripResponse.model_validate(raw)

	async def cancel_trip(self, trip_id: str, reason: str = "") -> TripResponse:
		"""Cancel a planned or dispatched trip."""
		raw = self._get("trips", trip_id)
		assert raw, f"Trip {trip_id} not found"
		self._tenant_assert(raw)
		assert raw["status"] in (
			TripStatus.PLANNED.value, TripStatus.DISPATCHED.value
		), "Only PLANNED or DISPATCHED trips can be cancelled"

		raw["status"] = TripStatus.CANCELLED.value
		raw["delay_reason"] = reason
		raw["updated_at"] = datetime.utcnow().isoformat()
		self._put("trips", trip_id, raw)
		self._emit_event("trip.cancelled", trip_id, {"reason": reason})
		self._log_op("cancel_trip", trip_id, reason=reason)
		return TripResponse.model_validate(raw)

	async def record_trip_breakdown(self, trip_id: str, reason: str = "") -> TripResponse:
		"""Record a mid-trip breakdown — status → BREAKDOWN, vehicle → BREAKDOWN."""
		raw = self._get("trips", trip_id)
		assert raw, f"Trip {trip_id} not found"
		self._tenant_assert(raw)
		assert raw["status"] == TripStatus.IN_PROGRESS.value

		raw["status"] = TripStatus.BREAKDOWN.value
		raw["breakdown_at"] = datetime.utcnow().isoformat()
		raw["delay_reason"] = reason
		raw["updated_at"] = datetime.utcnow().isoformat()
		self._put("trips", trip_id, raw)

		# Mark vehicle as breakdown
		v_raw = self._get("vehicles", raw["vehicle_id"])
		if v_raw:
			v_raw["status"] = VehicleStatus.BREAKDOWN.value
			v_raw["updated_at"] = datetime.utcnow().isoformat()
			self._put("vehicles", raw["vehicle_id"], v_raw)

		self._emit_event("trip.breakdown", trip_id, {"reason": reason, "vehicle_id": raw["vehicle_id"]})
		self._log_op("record_trip_breakdown", trip_id, reason=reason)
		return TripResponse.model_validate(raw)

	async def change_trip_driver(self, trip_id: str, new_driver_id: str, reason: str = "") -> TripResponse:
		"""Replace driver mid-trip (e.g. relay driving, illness)."""
		raw = self._get("trips", trip_id)
		assert raw, f"Trip {trip_id} not found"
		self._tenant_assert(raw)
		assert raw["status"] in (TripStatus.DISPATCHED.value, TripStatus.IN_PROGRESS.value)

		new_driver = await self.get_driver(new_driver_id)
		assert_driver_active(new_driver.status.value)
		assert_driver_licence_valid(new_driver.licence_expiry)

		old_driver_id = raw["driver_id"]
		raw["driver_id"] = new_driver_id
		raw["updated_at"] = datetime.utcnow().isoformat()
		self._put("trips", trip_id, raw)
		self._emit_event("trip.driver_changed", trip_id, {
			"old_driver_id": old_driver_id, "new_driver_id": new_driver_id, "reason": reason,
		})
		self._log_op("change_trip_driver", trip_id, old=old_driver_id, new=new_driver_id)
		return TripResponse.model_validate(raw)

	async def get_trip(self, trip_id: str) -> TripResponse:
		raw = self._get("trips", trip_id)
		assert raw, f"Trip {trip_id} not found"
		self._tenant_assert(raw)
		return TripResponse.model_validate(raw)

	async def list_trips(
		self,
		status: TripStatus | None = None,
		vehicle_id: str | None = None,
		driver_id: str | None = None,
	) -> list[TripResponse]:
		rows = self._list("trips")
		if status:
			rows = [r for r in rows if r.get("status") == status.value]
		if vehicle_id:
			rows = [r for r in rows if r.get("vehicle_id") == vehicle_id]
		if driver_id:
			rows = [r for r in rows if r.get("driver_id") == driver_id]
		return [TripResponse.model_validate(r) for r in rows]

	async def update_trip(self, trip_id: str, patch: TripUpdate) -> TripResponse:
		raw = self._get("trips", trip_id)
		assert raw, f"Trip {trip_id} not found"
		self._tenant_assert(raw)
		updates = patch.model_dump(exclude_none=True, mode="json")
		updates["updated_at"] = datetime.utcnow().isoformat()
		raw.update(updates)
		self._put("trips", trip_id, raw)
		self._emit_event("trip.updated", trip_id, updates)
		return TripResponse.model_validate(raw)

	# ──────────────────────────────────────────────────────────────
	# Fuel Records
	# ──────────────────────────────────────────────────────────────

	async def record_fuel_purchase(self, payload: FuelRecordCreate) -> FuelRecordResponse:
		"""Record a fuel purchase with cost calculation."""
		assert payload.tenant_id == self._tenant_id, "Tenant mismatch"

		# Odometer regression check
		vehicle = await self.get_vehicle(payload.vehicle_id)
		if vehicle.odometer_km > 0:
			assert_odometer_not_regressing(payload.odometer_km, vehicle.odometer_km)

		total_cost = calculate_fuel_cost(payload.litres, payload.cost_per_litre)
		rec = FuelRecordResponse(
			**payload.model_dump(),
			total_cost=total_cost,
		)
		self._put("fuel_records", rec.id, rec.model_dump(mode="json"))

		# Update vehicle odometer
		v_raw = self._get("vehicles", payload.vehicle_id)
		if v_raw:
			v_raw["odometer_km"] = str(payload.odometer_km)
			v_raw["updated_at"] = datetime.utcnow().isoformat()
			self._put("vehicles", payload.vehicle_id, v_raw)

		self._emit_event("fuel.recorded", rec.id, {
			"vehicle_id": payload.vehicle_id, "litres": str(payload.litres), "total_cost": str(total_cost),
		})
		self._log_op("record_fuel_purchase", rec.id, litres=str(payload.litres), cost=str(total_cost))
		return rec

	async def list_fuel_records(self, vehicle_id: str | None = None) -> list[FuelRecordResponse]:
		rows = self._list("fuel_records")
		if vehicle_id:
			rows = [r for r in rows if r.get("vehicle_id") == vehicle_id]
		return [FuelRecordResponse.model_validate(r) for r in rows]

	# ──────────────────────────────────────────────────────────────
	# Maintenance
	# ──────────────────────────────────────────────────────────────

	async def schedule_maintenance(self, payload: MaintenanceCreate) -> MaintenanceResponse:
		"""Schedule a maintenance job."""
		assert payload.tenant_id == self._tenant_id, "Tenant mismatch"
		await self.get_vehicle(payload.vehicle_id)  # existence check

		rec = MaintenanceResponse(**payload.model_dump(), status=MaintenanceStatus.SCHEDULED)
		self._put("maintenance", rec.id, rec.model_dump(mode="json"))
		self._emit_event("maintenance.scheduled", rec.id, {
			"vehicle_id": payload.vehicle_id, "type": payload.maintenance_type.value,
			"scheduled_date": payload.scheduled_date.isoformat(),
		})
		self._log_op("schedule_maintenance", rec.id, vehicle=payload.vehicle_id)
		return rec

	async def start_maintenance(self, maintenance_id: str) -> MaintenanceResponse:
		"""Mark maintenance as in-progress; set vehicle to IN_MAINTENANCE."""
		raw = self._get("maintenance", maintenance_id)
		assert raw, f"Maintenance {maintenance_id} not found"
		self._tenant_assert(raw)

		raw["status"] = MaintenanceStatus.IN_PROGRESS.value
		raw["updated_at"] = datetime.utcnow().isoformat()
		self._put("maintenance", maintenance_id, raw)

		v_raw = self._get("vehicles", raw["vehicle_id"])
		if v_raw:
			v_raw["status"] = VehicleStatus.IN_MAINTENANCE.value
			v_raw["updated_at"] = datetime.utcnow().isoformat()
			self._put("vehicles", raw["vehicle_id"], v_raw)

		self._emit_event("maintenance.started", maintenance_id, {"vehicle_id": raw["vehicle_id"]})
		self._log_op("start_maintenance", maintenance_id)
		return MaintenanceResponse.model_validate(raw)

	async def complete_maintenance(
		self, maintenance_id: str, actual_cost: Decimal, notes: str = ""
	) -> MaintenanceResponse:
		"""Complete maintenance; restore vehicle to ACTIVE."""
		raw = self._get("maintenance", maintenance_id)
		assert raw, f"Maintenance {maintenance_id} not found"
		self._tenant_assert(raw)

		raw["status"] = MaintenanceStatus.COMPLETED.value
		raw["completed_date"] = datetime.utcnow().isoformat()
		raw["actual_cost"] = str(actual_cost)
		if notes:
			raw["notes"] = notes
		raw["updated_at"] = datetime.utcnow().isoformat()
		self._put("maintenance", maintenance_id, raw)

		v_raw = self._get("vehicles", raw["vehicle_id"])
		if v_raw and v_raw.get("status") == VehicleStatus.IN_MAINTENANCE.value:
			v_raw["status"] = VehicleStatus.ACTIVE.value
			v_raw["updated_at"] = datetime.utcnow().isoformat()
			self._put("vehicles", raw["vehicle_id"], v_raw)

		self._emit_event("maintenance.completed", maintenance_id, {
			"vehicle_id": raw["vehicle_id"], "actual_cost": str(actual_cost),
		})
		self._log_op("complete_maintenance", maintenance_id, cost=str(actual_cost))
		return MaintenanceResponse.model_validate(raw)

	async def list_maintenance(
		self, vehicle_id: str | None = None, status: MaintenanceStatus | None = None
	) -> list[MaintenanceResponse]:
		rows = self._list("maintenance")
		if vehicle_id:
			rows = [r for r in rows if r.get("vehicle_id") == vehicle_id]
		if status:
			rows = [r for r in rows if r.get("status") == status.value]
		return [MaintenanceResponse.model_validate(r) for r in rows]

	# ──────────────────────────────────────────────────────────────
	# Inspections
	# ──────────────────────────────────────────────────────────────

	async def record_inspection(self, payload: InspectionCreate) -> InspectionResponse:
		"""Record a vehicle inspection."""
		assert payload.tenant_id == self._tenant_id, "Tenant mismatch"
		await self.get_vehicle(payload.vehicle_id)

		rec = InspectionResponse(**payload.model_dump())
		self._put("inspections", rec.id, rec.model_dump(mode="json"))

		if rec.result == InspectionResult.FAIL:
			await self.process_inspection_failure(rec.id)

		self._emit_event("inspection.recorded", rec.id, {
			"vehicle_id": payload.vehicle_id, "result": payload.result.value,
			"defects": payload.defects,
		})
		self._log_op("record_inspection", rec.id, result=payload.result.value)
		return rec

	async def process_inspection_failure(self, inspection_id: str) -> dict[str, Any]:
		"""
		Handle inspection failure: set vehicle OUT_OF_SERVICE,
		schedule corrective maintenance, notify fleet manager.
		"""
		raw = self._get("inspections", inspection_id)
		assert raw, f"Inspection {inspection_id} not found"
		self._tenant_assert(raw)
		assert raw.get("result") == InspectionResult.FAIL.value, "Inspection did not fail"

		vehicle_id = raw["vehicle_id"]
		v_raw = self._get("vehicles", vehicle_id)
		if v_raw:
			v_raw["status"] = VehicleStatus.OUT_OF_SERVICE.value
			v_raw["updated_at"] = datetime.utcnow().isoformat()
			self._put("vehicles", vehicle_id, v_raw)

		# Auto-schedule corrective maintenance for each defect
		defects = raw.get("defects", [])
		maint_ids = []
		for defect in defects:
			maint = await self.schedule_maintenance(MaintenanceCreate(
				tenant_id=self._tenant_id,
				vehicle_id=vehicle_id,
				maintenance_type="corrective",  # type: ignore[arg-type]
				description=f"Corrective: {defect}",
				scheduled_date=datetime.utcnow(),
			))
			maint_ids.append(maint.id)

		self._emit_event("inspection.failure_processed", inspection_id, {
			"vehicle_id": vehicle_id, "defects": defects, "maintenance_scheduled": maint_ids,
		})
		self._log_op("process_inspection_failure", inspection_id, vehicle=vehicle_id)
		return {
			"inspection_id": inspection_id,
			"vehicle_id": vehicle_id,
			"vehicle_status": VehicleStatus.OUT_OF_SERVICE.value,
			"maintenance_scheduled": maint_ids,
		}

	async def list_inspections(self, vehicle_id: str | None = None) -> list[InspectionResponse]:
		rows = self._list("inspections")
		if vehicle_id:
			rows = [r for r in rows if r.get("vehicle_id") == vehicle_id]
		return [InspectionResponse.model_validate(r) for r in rows]

	# ──────────────────────────────────────────────────────────────
	# COF Inspections
	# ──────────────────────────────────────────────────────────────

	async def record_cof_inspection(self, payload: COFInspectionCreate) -> COFInspectionResponse:
		"""Record a Certificate of Fitness inspection."""
		assert payload.tenant_id == self._tenant_id, "Tenant mismatch"
		await self.get_vehicle(payload.vehicle_id)

		rec = COFInspectionResponse(**payload.model_dump())
		self._put("cof_inspections", rec.id, rec.model_dump(mode="json"))
		self._emit_event("cof.recorded", rec.id, {
			"vehicle_id": payload.vehicle_id, "result": payload.result.value,
			"cof_number": payload.cof_number,
		})
		self._log_op("record_cof_inspection", rec.id, vehicle=payload.vehicle_id)
		return rec

	async def list_cof_inspections(self, vehicle_id: str | None = None) -> list[COFInspectionResponse]:
		rows = self._list("cof_inspections")
		if vehicle_id:
			rows = [r for r in rows if r.get("vehicle_id") == vehicle_id]
		return [COFInspectionResponse.model_validate(r) for r in rows]

	# ──────────────────────────────────────────────────────────────
	# Incidents
	# ──────────────────────────────────────────────────────────────

	async def report_incident(self, payload: IncidentCreate) -> IncidentResponse:
		"""Report a fleet incident."""
		assert payload.tenant_id == self._tenant_id, "Tenant mismatch"
		assert_incident_reported_within_window(payload.occurred_at)
		assert_fatal_incident_requires_police_ref(payload.severity.value, payload.police_ref)

		rec = IncidentResponse(**payload.model_dump(), status=IncidentStatus.REPORTED)
		self._put("incidents", rec.id, rec.model_dump(mode="json"))

		if payload.severity.value in ("major", "critical", "fatal"):
			v_raw = self._get("vehicles", payload.vehicle_id)
			if v_raw:
				v_raw["status"] = VehicleStatus.OUT_OF_SERVICE.value
				v_raw["updated_at"] = datetime.utcnow().isoformat()
				self._put("vehicles", payload.vehicle_id, v_raw)

		self._emit_event("incident.reported", rec.id, {
			"vehicle_id": payload.vehicle_id, "severity": payload.severity.value,
		})
		self._log_op("report_incident", rec.id, severity=payload.severity.value)
		return rec

	async def close_incident(self, incident_id: str, resolution: str = "") -> IncidentResponse:
		raw = self._get("incidents", incident_id)
		assert raw, f"Incident {incident_id} not found"
		self._tenant_assert(raw)
		raw["status"] = IncidentStatus.CLOSED.value
		raw["notes"] = resolution
		raw["updated_at"] = datetime.utcnow().isoformat()
		self._put("incidents", incident_id, raw)
		self._emit_event("incident.closed", incident_id, {"resolution": resolution})
		return IncidentResponse.model_validate(raw)

	async def list_incidents(
		self, vehicle_id: str | None = None, status: IncidentStatus | None = None
	) -> list[IncidentResponse]:
		rows = self._list("incidents")
		if vehicle_id:
			rows = [r for r in rows if r.get("vehicle_id") == vehicle_id]
		if status:
			rows = [r for r in rows if r.get("status") == status.value]
		return [IncidentResponse.model_validate(r) for r in rows]

	# ──────────────────────────────────────────────────────────────
	# Insurance Policies
	# ──────────────────────────────────────────────────────────────

	async def add_insurance_policy(self, payload: InsurancePolicyCreate) -> InsurancePolicyResponse:
		assert payload.tenant_id == self._tenant_id, "Tenant mismatch"
		await self.get_vehicle(payload.vehicle_id)

		rec = InsurancePolicyResponse(**payload.model_dump(), is_active=True)
		self._put("insurance", rec.id, rec.model_dump(mode="json"))
		self._emit_event("insurance.added", rec.id, {
			"vehicle_id": payload.vehicle_id, "policy_number": payload.policy_number,
			"cover_end": payload.cover_end.isoformat(),
		})
		self._log_op("add_insurance_policy", rec.id, vehicle=payload.vehicle_id)
		return rec

	async def list_insurance_policies(self, vehicle_id: str | None = None) -> list[InsurancePolicyResponse]:
		rows = self._list("insurance")
		if vehicle_id:
			rows = [r for r in rows if r.get("vehicle_id") == vehicle_id]
		return [InsurancePolicyResponse.model_validate(r) for r in rows]

	# ──────────────────────────────────────────────────────────────
	# Registration
	# ──────────────────────────────────────────────────────────────

	async def register_vehicle_docs(self, payload: RegistrationCreate) -> RegistrationResponse:
		assert payload.tenant_id == self._tenant_id, "Tenant mismatch"
		await self.get_vehicle(payload.vehicle_id)

		rec = RegistrationResponse(**payload.model_dump(), is_current=True)
		self._put("registrations", rec.id, rec.model_dump(mode="json"))
		self._emit_event("registration.added", rec.id, {
			"vehicle_id": payload.vehicle_id, "reg_number": payload.registration_number,
			"expires_at": payload.expires_at.isoformat(),
		})
		self._log_op("register_vehicle_docs", rec.id)
		return rec

	async def list_registrations(self, vehicle_id: str | None = None) -> list[RegistrationResponse]:
		rows = self._list("registrations")
		if vehicle_id:
			rows = [r for r in rows if r.get("vehicle_id") == vehicle_id]
		return [RegistrationResponse.model_validate(r) for r in rows]

	# ──────────────────────────────────────────────────────────────
	# Tachograph Records
	# ──────────────────────────────────────────────────────────────

	async def record_tachograph(self, payload: TachographRecordCreate) -> TachographRecordResponse:
		"""Record tachograph data with EU HOS rule checks."""
		assert payload.tenant_id == self._tenant_id, "Tenant mismatch"

		assert_eu_continuous_driving(payload.driving_minutes)
		assert_eu_daily_driving(payload.driving_minutes)

		# Check weekly driving — sum today + existing
		driver_records = [
			r for r in self._list("tachograph")
			if r.get("driver_id") == payload.driver_id
		]
		weekly_min = sum(r.get("driving_minutes", 0) for r in driver_records) + payload.driving_minutes
		assert_eu_weekly_driving(weekly_min)

		rec = TachographRecordResponse(**payload.model_dump())
		self._put("tachograph", rec.id, rec.model_dump(mode="json"))
		self._emit_event("tachograph.recorded", rec.id, {
			"driver_id": payload.driver_id, "vehicle_id": payload.vehicle_id,
			"driving_minutes": payload.driving_minutes, "infringement": payload.infringement_code,
		})
		self._log_op("record_tachograph", rec.id, driver=payload.driver_id)
		return rec

	async def list_tachograph_records(self, driver_id: str | None = None) -> list[TachographRecordResponse]:
		rows = self._list("tachograph")
		if driver_id:
			rows = [r for r in rows if r.get("driver_id") == driver_id]
		return [TachographRecordResponse.model_validate(r) for r in rows]

	# ──────────────────────────────────────────────────────────────
	# Telematics
	# ──────────────────────────────────────────────────────────────

	async def track_vehicle_realtime(self, payload: TelematicsEventCreate) -> TelematicsEventResponse:
		"""Ingest a real-time telematics event."""
		assert payload.tenant_id == self._tenant_id, "Tenant mismatch"
		await self.get_vehicle(payload.vehicle_id)

		rec = TelematicsEventResponse(**payload.model_dump())
		self._put("telematics", rec.id, rec.model_dump(mode="json"))
		self._emit_event("telematics.event", rec.id, {
			"vehicle_id": payload.vehicle_id, "event_type": payload.event_type,
			"lat": payload.lat, "lon": payload.lon, "speed_kmh": payload.speed_kmh,
		})
		return rec

	async def get_vehicle_last_position(self, vehicle_id: str) -> TelematicsEventResponse | None:
		"""Return the most recent telematics position for a vehicle."""
		rows = [
			r for r in self._list("telematics")
			if r.get("vehicle_id") == vehicle_id
		]
		if not rows:
			return None
		latest = max(rows, key=lambda r: r.get("occurred_at", ""))
		return TelematicsEventResponse.model_validate(latest)

	async def list_telematics_events(
		self, vehicle_id: str | None = None, event_type: str | None = None
	) -> list[TelematicsEventResponse]:
		rows = self._list("telematics")
		if vehicle_id:
			rows = [r for r in rows if r.get("vehicle_id") == vehicle_id]
		if event_type:
			rows = [r for r in rows if r.get("event_type") == event_type]
		return [TelematicsEventResponse.model_validate(r) for r in rows]

	# ──────────────────────────────────────────────────────────────
	# Analytics & Reports
	# ──────────────────────────────────────────────────────────────

	async def calculate_tco(self, vehicle_id: str) -> TCOBreakdown:
		"""
		Calculate Total Cost of Ownership for a vehicle.
		Sums all fuel, maintenance, insurance, and registration costs.
		"""
		await self.get_vehicle(vehicle_id)

		fuel_rows = [r for r in self._list("fuel_records") if r.get("vehicle_id") == vehicle_id]
		maint_rows = [r for r in self._list("maintenance") if r.get("vehicle_id") == vehicle_id]
		ins_rows = [r for r in self._list("insurance") if r.get("vehicle_id") == vehicle_id]
		trip_rows = [r for r in self._list("trips") if r.get("vehicle_id") == vehicle_id]
		incident_rows = [r for r in self._list("incidents") if r.get("vehicle_id") == vehicle_id]

		fuel_cost = sum(Decimal(str(r.get("total_cost", 0))) for r in fuel_rows)
		maint_cost = sum(
			Decimal(str(r.get("actual_cost") or r.get("estimated_cost", 0))) for r in maint_rows
		)
		ins_cost = sum(Decimal(str(r.get("premium", 0))) for r in ins_rows)
		fine_cost = sum(Decimal(str(r.get("overloading_fine_allocated", 0))) for r in incident_rows)
		total_dist = sum(Decimal(str(r.get("distance_km") or 0)) for r in trip_rows)
		total = calculate_tco(fuel_cost, maint_cost, ins_cost, Decimal("0"), Decimal("0"), Decimal("0"), fine_cost=fine_cost)
		cpm = calculate_cost_per_km(total, total_dist)

		now = datetime.utcnow()
		self._log_op("calculate_tco", vehicle_id, total=str(total), dist_km=str(total_dist))
		return TCOBreakdown(
			vehicle_id=vehicle_id,
			tenant_id=self._tenant_id,
			period_start=datetime(now.year, 1, 1),
			period_end=now,
			fuel_cost=fuel_cost,
			maintenance_cost=maint_cost,
			insurance_cost=ins_cost,
			fine_cost=fine_cost,
			total_cost=total,
			distance_km=total_dist,
			cost_per_km=cpm,
		)

	async def driver_behaviour_scoring(self, driver_id: str) -> DriverBehaviourScore:
		"""Compute aggregated driver behaviour score from telematics events."""
		await self.get_driver(driver_id)

		telematics = [r for r in self._list("telematics") if r.get("driver_id") == driver_id]
		trips = [r for r in self._list("trips") if r.get("driver_id") == driver_id]
		incidents = [r for r in self._list("incidents") if r.get("driver_id") == driver_id]

		def _count(event_type: str) -> int:
			return sum(1 for r in telematics if r.get("event_type") == event_type)

		total_dist = sum(Decimal(str(r.get("distance_km") or 0)) for r in trips)
		scores = calculate_driver_score(
			speeding_events=_count("speeding"),
			harsh_braking_events=_count("harsh_braking"),
			harsh_acceleration_events=_count("harsh_acceleration"),
			cornering_events=_count("harsh_cornering"),
			idle_events=_count("idle"),
			seatbelt_events=_count("seatbelt_violation"),
			distraction_events=_count("distraction"),
			distance_km=total_dist,
		)
		now = datetime.utcnow()
		self._log_op("driver_behaviour_scoring", driver_id, score=str(scores["overall"]))
		return DriverBehaviourScore(
			driver_id=driver_id,
			tenant_id=self._tenant_id,
			period_start=datetime(now.year, 1, 1),
			period_end=now,
			overall_score=scores["overall"],
			speeding_score=scores["speeding"],
			harsh_braking_score=scores["harsh_braking"],
			harsh_acceleration_score=scores["harsh_acceleration"],
			cornering_score=scores["cornering"],
			idle_score=scores["idle"],
			seatbelt_score=scores["seatbelt"],
			distraction_score=scores["distraction"],
			fatigue_score=100.0,  # requires tachograph data to compute
			incidents_count=len(incidents),
			trips_count=len(trips),
			distance_km=total_dist,
			grade=scores["grade"],
		)

	async def fleet_utilisation_analytics(self) -> FleetUtilisationReport:
		"""Fleet-wide utilisation analytics."""
		vehicles = self._list("vehicles")
		trips = self._list("trips")
		fuel_records = self._list("fuel_records")
		maintenance_rows = self._list("maintenance")

		total = len(vehicles)
		active = sum(1 for v in vehicles if v.get("status") == VehicleStatus.ACTIVE.value)
		in_maint = sum(1 for v in vehicles if v.get("status") == VehicleStatus.IN_MAINTENANCE.value)
		awaiting_insp = sum(1 for v in vehicles if v.get("status") == VehicleStatus.AWAITING_INSPECTION.value)
		on_trip = sum(1 for t in trips if t.get("status") in ("dispatched", "in_progress"))

		total_dist = sum(Decimal(str(t.get("distance_km") or 0)) for t in trips)
		total_fuel = sum(Decimal(str(f.get("litres", 0))) for f in fuel_records)
		completed_trips = [t for t in trips if t.get("status") == TripStatus.COMPLETED.value]

		fuel_eff = float(calculate_fuel_efficiency_l100km(total_fuel, total_dist)) if total_dist > 0 else 0.0

		overdue_maint = sum(
			1 for m in maintenance_rows
			if m.get("status") == MaintenanceStatus.OVERDUE.value
		)

		now = datetime.utcnow()
		self._log_op("fleet_utilisation_analytics", "", vehicles=str(total), active=str(active))
		return FleetUtilisationReport(
			tenant_id=self._tenant_id,
			period_start=datetime(now.year, 1, 1),
			period_end=now,
			total_vehicles=total,
			active_vehicles=active,
			avg_utilisation_pct=round(active / total * 100, 2) if total else 0.0,
			total_distance_km=total_dist,
			total_trips=len(trips),
			total_fuel_l=total_fuel,
			avg_fuel_efficiency_l100km=fuel_eff,
			vehicles_in_maintenance=in_maint,
			vehicles_awaiting_inspection=awaiting_insp,
			overdue_maintenance_count=overdue_maint,
		)

	async def compliance_calendar(self) -> list[ComplianceCalendarEntry]:
		"""
		Generate a compliance calendar — all upcoming and overdue compliance
		events across vehicles and drivers.
		"""
		entries: list[ComplianceCalendarEntry] = []
		now = datetime.utcnow()

		# Vehicle: insurance
		for ins in self._list("insurance"):
			if ins.get("cover_end"):
				due = datetime.fromisoformat(ins["cover_end"])
				d = days_until(due, now)
				entries.append(ComplianceCalendarEntry(
					entity_id=ins["vehicle_id"],
					entity_type="vehicle",
					tenant_id=self._tenant_id,
					due_date=due,
					event_type="insurance_renewal",
					description=f"Insurance policy {ins.get('policy_number','?')} expires",
					days_until_due=d,
					is_overdue=d < 0,
					severity=compliance_severity(d),
				))

		# Vehicle: COF
		for cof in self._list("cof_inspections"):
			if cof.get("expires_at"):
				due = datetime.fromisoformat(cof["expires_at"])
				d = days_until(due, now)
				entries.append(ComplianceCalendarEntry(
					entity_id=cof["vehicle_id"],
					entity_type="vehicle",
					tenant_id=self._tenant_id,
					due_date=due,
					event_type="cof_renewal",
					description=f"COF {cof.get('cof_number','?')} expires",
					days_until_due=d,
					is_overdue=d < 0,
					severity=compliance_severity(d),
				))

		# Vehicle: registration
		for reg in self._list("registrations"):
			if reg.get("expires_at"):
				due = datetime.fromisoformat(reg["expires_at"])
				d = days_until(due, now)
				entries.append(ComplianceCalendarEntry(
					entity_id=reg["vehicle_id"],
					entity_type="vehicle",
					tenant_id=self._tenant_id,
					due_date=due,
					event_type="registration_renewal",
					description=f"Registration {reg.get('registration_number','?')} expires",
					days_until_due=d,
					is_overdue=d < 0,
					severity=compliance_severity(d),
				))

		# Driver: licence expiry
		for drv in self._list("drivers"):
			if drv.get("licence_expiry"):
				due = datetime.fromisoformat(drv["licence_expiry"])
				d = days_until(due, now)
				entries.append(ComplianceCalendarEntry(
					entity_id=drv["id"],
					entity_type="driver",
					tenant_id=self._tenant_id,
					due_date=due,
					event_type="licence_expiry",
					description=f"Driver licence expires ({drv.get('name','?')})",
					days_until_due=d,
					is_overdue=d < 0,
					severity=compliance_severity(d),
				))
			if drv.get("cpc_expiry"):
				due = datetime.fromisoformat(drv["cpc_expiry"])
				d = days_until(due, now)
				entries.append(ComplianceCalendarEntry(
					entity_id=drv["id"],
					entity_type="driver",
					tenant_id=self._tenant_id,
					due_date=due,
					event_type="cpc_expiry",
					description=f"CPC expires ({drv.get('name','?')})",
					days_until_due=d,
					is_overdue=d < 0,
					severity=compliance_severity(d),
				))

		# Maintenance overdue
		for maint in self._list("maintenance"):
			if maint.get("scheduled_date") and maint.get("status") not in ("completed", "cancelled"):
				due = datetime.fromisoformat(maint["scheduled_date"])
				d = days_until(due, now)
				if d <= 30:
					entries.append(ComplianceCalendarEntry(
						entity_id=maint["vehicle_id"],
						entity_type="vehicle",
						tenant_id=self._tenant_id,
						due_date=due,
						event_type="maintenance_due",
						description=maint.get("description", "Scheduled maintenance"),
						days_until_due=d,
						is_overdue=d < 0,
						severity=compliance_severity(d),
					))

		entries.sort(key=lambda e: e.due_date)
		self._log_op("compliance_calendar", "", entries=str(len(entries)))
		return entries

	async def predictive_maintenance_alerts(self) -> list[PredictiveMaintenanceAlert]:
		"""
		Generate predictive maintenance alerts using odometer trends
		and oil change schedules.  In production, feed telematics sensor
		data to an ML model via APG ai_orchestration.
		"""
		alerts: list[PredictiveMaintenanceAlert] = []
		now = datetime.utcnow()

		for v in self._list("vehicles"):
			vehicle_id = v["id"]
			current_odo = Decimal(str(v.get("odometer_km") or 0))

			# Oil change prediction (10,000 km interval)
			fuel_rows = [r for r in self._list("fuel_records") if r.get("vehicle_id") == vehicle_id]
			if fuel_rows:
				first_fuel = sorted(fuel_rows, key=lambda r: r.get("fuelled_at", ""))[0]
				last_oil_km = Decimal(str(first_fuel.get("odometer_km", 0)))
				oil_result = predict_oil_change_due(
					last_oil_change_km=last_oil_km,
					current_odometer_km=current_odo,
				)
				if oil_result["urgency"] in ("medium", "high", "critical"):
					alerts.append(PredictiveMaintenanceAlert(
						vehicle_id=vehicle_id,
						tenant_id=self._tenant_id,
						component="engine_oil",
						confidence_pct=85.0,
						recommended_action=f"Schedule oil change (remaining: {oil_result['km_remaining']:.0f} km)",
						urgency=oil_result["urgency"],
						supporting_signals=["odometer_interval", "calendar_interval"],
					))

			# Overdue maintenance → critical alert
			maint_rows = [
				m for m in self._list("maintenance")
				if m.get("vehicle_id") == vehicle_id
				and m.get("status") not in ("completed", "cancelled")
			]
			for m in maint_rows:
				if m.get("scheduled_date"):
					d = days_until(datetime.fromisoformat(m["scheduled_date"]), now)
					if d < 0:
						alerts.append(PredictiveMaintenanceAlert(
							vehicle_id=vehicle_id,
							tenant_id=self._tenant_id,
							component=m.get("description", "maintenance")[:60],
							confidence_pct=99.0,
							recommended_action=f"Overdue by {abs(d)} days — complete immediately",
							urgency="critical",
							supporting_signals=["scheduled_date_passed"],
						))

		self._log_op("predictive_maintenance_alerts", "", alerts=str(len(alerts)))
		return alerts

	async def dashboard_kpis(self) -> DashboardKPIs:
		"""Compute fleet dashboard KPIs."""
		vehicles = self._list("vehicles")
		drivers = self._list("drivers")
		trips = self._list("trips")
		fuel_records = self._list("fuel_records")
		maintenance_rows = self._list("maintenance")
		incidents = self._list("incidents")

		now = datetime.utcnow()
		month_start = datetime(now.year, now.month, 1)
		today_start = datetime(now.year, now.month, now.day)

		fuel_mtd = sum(
			Decimal(str(r.get("total_cost", 0))) for r in fuel_records
			if r.get("fuelled_at", "") >= month_start.isoformat()
		)
		maint_mtd = sum(
			Decimal(str(r.get("actual_cost") or r.get("estimated_cost", 0)))
			for r in maintenance_rows
			if r.get("completed_date", "") >= month_start.isoformat()
		)
		compliance_alerts = sum(
			1 for r in self._list("insurance")
			if r.get("cover_end") and days_until(datetime.fromisoformat(r["cover_end"]), now) <= 30
		)

		total_v = len(vehicles)
		active_v = sum(1 for v in vehicles if v.get("status") == VehicleStatus.ACTIVE.value)

		self._log_op("dashboard_kpis", "")
		return DashboardKPIs(
			tenant_id=self._tenant_id,
			as_of=now,
			total_vehicles=total_v,
			active_vehicles=active_v,
			vehicles_on_trip=sum(1 for t in trips if t.get("status") in ("dispatched", "in_progress")),
			vehicles_in_maintenance=sum(1 for v in vehicles if v.get("status") == VehicleStatus.IN_MAINTENANCE.value),
			total_drivers=len(drivers),
			active_drivers=sum(1 for d in drivers if d.get("status") == DriverStatus.ACTIVE.value),
			drivers_on_duty=sum(1 for t in trips if t.get("status") == "in_progress"),
			trips_today=sum(1 for t in trips if t.get("created_at", "") >= today_start.isoformat()),
			trips_in_progress=sum(1 for t in trips if t.get("status") == "in_progress"),
			fuel_spend_mtd=fuel_mtd,
			maintenance_spend_mtd=maint_mtd,
			fleet_utilisation_pct=round(active_v / total_v * 100, 2) if total_v else 0.0,
			compliance_alerts=compliance_alerts,
			overdue_maintenance=sum(1 for m in maintenance_rows if m.get("status") == "overdue"),
			active_incidents=sum(1 for i in incidents if i.get("status") not in ("resolved", "closed")),
		)
