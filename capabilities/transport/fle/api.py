"""
Fleet Management REST API — Flask Blueprint.

url_prefix: /api/fle/v1

All routes return JSON.  Errors follow:
  {"error": "<message>", "code": "<code>"}

Auth: bearer token via X-API-Key header (stub — delegates to APG auth adapter).
Tenant: X-Tenant-ID header (required on every request).
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from functools import wraps
from typing import Any

from flask import Blueprint, jsonify, request

from .models import (
	COFInspectionCreate,
	DriverCreate, DriverStatus, DriverUpdate,
	FuelRecordCreate,
	IncidentCreate, IncidentStatus,
	InspectionCreate,
	InsurancePolicyCreate,
	MaintenanceCreate, MaintenanceStatus,
	RegistrationCreate,
	TachographRecordCreate,
	TelematicsEventCreate,
	TripCreate, TripStatus, TripUpdate,
	VehicleAssignmentCreate,
	VehicleCreate, VehicleStatus, VehicleUpdate,
)
from .service import FleetService

fle_bp = Blueprint("fle", __name__, url_prefix="/api/fle/v1")


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

class _InMemoryDB:
	"""Minimal in-process store used when no SQLAlchemy session is provided."""
	pass


def _get_service() -> FleetService:
	tenant_id = request.headers.get("X-Tenant-ID", "default")
	actor_id = request.headers.get("X-Actor-ID", "api")
	db = _InMemoryDB()
	return FleetService(db, tenant_id, actor_id)


def _run(coro: Any) -> Any:
	"""Run an async coroutine from a sync Flask route."""
	loop = asyncio.new_event_loop()
	try:
		return asyncio.run(coro)
	finally:
		loop.close()


def _ok(data: Any, status: int = 200):
	return jsonify(data), status


def _err(message: str, code: str = "error", status: int = 400):
	return jsonify({"error": message, "code": code}), status


def _paginate(rows: list[Any], page: int, per_page: int) -> dict[str, Any]:
	total = len(rows)
	start = (page - 1) * per_page
	end = start + per_page
	return {
		"items": [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in rows[start:end]],
		"total": total,
		"page": page,
		"per_page": per_page,
		"pages": max(1, (total + per_page - 1) // per_page),
	}


def _catch(fn):
	"""Decorator: convert exceptions to structured JSON errors."""
	@wraps(fn)
	def wrapper(*args, **kwargs):
		try:
			return fn(*args, **kwargs)
		except AssertionError as e:
			return _err(str(e), "not_found", 404)
		except (ValueError, TypeError) as e:
			return _err(str(e), "validation_error", 422)
		except PermissionError as e:
			return _err(str(e), "forbidden", 403)
		except Exception as e:
			return _err(str(e), "internal_error", 500)
	return wrapper


def _page_args() -> tuple[int, int]:
	page = max(1, int(request.args.get("page", 1)))
	per_page = min(200, max(1, int(request.args.get("per_page", 50))))
	return page, per_page


# ──────────────────────────────────────────────────────────────────
# Vehicles
# ──────────────────────────────────────────────────────────────────

@fle_bp.get("/vehicles")
@_catch
def list_vehicles():
	"""GET /vehicles?status=active&page=1&per_page=50"""
	svc = _get_service()
	status_raw = request.args.get("status")
	status = VehicleStatus(status_raw) if status_raw else None
	rows = _run(svc.list_vehicles(status=status))
	page, per_page = _page_args()
	return _ok(_paginate(rows, page, per_page))


@fle_bp.post("/vehicles")
@_catch
def create_vehicle():
	"""POST /vehicles"""
	svc = _get_service()
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
	payload = VehicleCreate(**body)
	result = _run(svc.register_vehicle(payload))
	return _ok(result.model_dump(mode="json"), 201)


@fle_bp.get("/vehicles/<vehicle_id>")
@_catch
def get_vehicle(vehicle_id: str):
	svc = _get_service()
	result = _run(svc.get_vehicle(vehicle_id))
	return _ok(result.model_dump(mode="json"))


@fle_bp.put("/vehicles/<vehicle_id>")
@_catch
def update_vehicle(vehicle_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	patch = VehicleUpdate(**body)
	result = _run(svc.update_vehicle(vehicle_id, patch))
	return _ok(result.model_dump(mode="json"))


@fle_bp.delete("/vehicles/<vehicle_id>")
@_catch
def delete_vehicle(vehicle_id: str):
	svc = _get_service()
	_run(svc.delete_vehicle(vehicle_id))
	return _ok({"deleted": True})


@fle_bp.post("/vehicles/<vehicle_id>/status")
@_catch
def set_vehicle_status(vehicle_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	status = VehicleStatus(body["status"])
	result = _run(svc.set_vehicle_status(vehicle_id, status))
	return _ok(result.model_dump(mode="json"))


# ──────────────────────────────────────────────────────────────────
# Drivers
# ──────────────────────────────────────────────────────────────────

@fle_bp.get("/drivers")
@_catch
def list_drivers():
	svc = _get_service()
	status_raw = request.args.get("status")
	status = DriverStatus(status_raw) if status_raw else None
	rows = _run(svc.list_drivers(status=status))
	page, per_page = _page_args()
	return _ok(_paginate(rows, page, per_page))


@fle_bp.post("/drivers")
@_catch
def create_driver():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
	payload = DriverCreate(**body)
	result = _run(svc.register_driver(payload))
	return _ok(result.model_dump(mode="json"), 201)


@fle_bp.get("/drivers/<driver_id>")
@_catch
def get_driver(driver_id: str):
	svc = _get_service()
	result = _run(svc.get_driver(driver_id))
	return _ok(result.model_dump(mode="json"))


@fle_bp.put("/drivers/<driver_id>")
@_catch
def update_driver(driver_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	patch = DriverUpdate(**body)
	result = _run(svc.update_driver(driver_id, patch))
	return _ok(result.model_dump(mode="json"))


@fle_bp.delete("/drivers/<driver_id>")
@_catch
def delete_driver(driver_id: str):
	svc = _get_service()
	_run(svc.delete_driver(driver_id))
	return _ok({"deleted": True})


@fle_bp.get("/drivers/<driver_id>/score")
@_catch
def driver_score(driver_id: str):
	svc = _get_service()
	result = _run(svc.driver_behaviour_scoring(driver_id))
	return _ok(result.model_dump(mode="json"))


# ──────────────────────────────────────────────────────────────────
# Assignments
# ──────────────────────────────────────────────────────────────────

@fle_bp.post("/assignments")
@_catch
def create_assignment():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
	payload = VehicleAssignmentCreate(**body)
	result = _run(svc.assign_driver(payload))
	return _ok(result.model_dump(mode="json"), 201)


# ──────────────────────────────────────────────────────────────────
# Trips
# ──────────────────────────────────────────────────────────────────

@fle_bp.get("/trips")
@_catch
def list_trips():
	svc = _get_service()
	status_raw = request.args.get("status")
	status = TripStatus(status_raw) if status_raw else None
	vehicle_id = request.args.get("vehicle_id")
	driver_id = request.args.get("driver_id")
	rows = _run(svc.list_trips(status=status, vehicle_id=vehicle_id, driver_id=driver_id))
	page, per_page = _page_args()
	return _ok(_paginate(rows, page, per_page))


@fle_bp.post("/trips")
@_catch
def create_trip():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
	payload = TripCreate(**body)
	result = _run(svc.plan_trip(payload))
	return _ok(result.model_dump(mode="json"), 201)


@fle_bp.get("/trips/<trip_id>")
@_catch
def get_trip(trip_id: str):
	svc = _get_service()
	result = _run(svc.get_trip(trip_id))
	return _ok(result.model_dump(mode="json"))


@fle_bp.put("/trips/<trip_id>")
@_catch
def update_trip(trip_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	patch = TripUpdate(**body)
	result = _run(svc.update_trip(trip_id, patch))
	return _ok(result.model_dump(mode="json"))


@fle_bp.post("/trips/<trip_id>/dispatch")
@_catch
def dispatch_trip(trip_id: str):
	svc = _get_service()
	result = _run(svc.dispatch_trip(trip_id))
	return _ok(result.model_dump(mode="json"))


@fle_bp.post("/trips/<trip_id>/start")
@_catch
def start_trip(trip_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	result = _run(svc.start_trip(trip_id, Decimal(str(body.get("odometer_start_km", 0)))))
	return _ok(result.model_dump(mode="json"))


@fle_bp.post("/trips/<trip_id>/complete")
@_catch
def complete_trip(trip_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	fuel = Decimal(str(body["fuel_consumed_l"])) if body.get("fuel_consumed_l") else None
	result = _run(svc.complete_trip(trip_id, Decimal(str(body["odometer_end_km"])), fuel))
	return _ok(result.model_dump(mode="json"))


@fle_bp.post("/trips/<trip_id>/cancel")
@_catch
def cancel_trip(trip_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	result = _run(svc.cancel_trip(trip_id, body.get("reason", "")))
	return _ok(result.model_dump(mode="json"))


@fle_bp.post("/trips/<trip_id>/breakdown")
@_catch
def trip_breakdown(trip_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	result = _run(svc.record_trip_breakdown(trip_id, body.get("reason", "")))
	return _ok(result.model_dump(mode="json"))


@fle_bp.post("/trips/<trip_id>/change-driver")
@_catch
def change_trip_driver(trip_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	result = _run(svc.change_trip_driver(trip_id, body["new_driver_id"], body.get("reason", "")))
	return _ok(result.model_dump(mode="json"))


# ──────────────────────────────────────────────────────────────────
# Fuel Records
# ──────────────────────────────────────────────────────────────────

@fle_bp.get("/fuel")
@_catch
def list_fuel():
	svc = _get_service()
	vehicle_id = request.args.get("vehicle_id")
	rows = _run(svc.list_fuel_records(vehicle_id=vehicle_id))
	page, per_page = _page_args()
	return _ok(_paginate(rows, page, per_page))


@fle_bp.post("/fuel")
@_catch
def create_fuel():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
	payload = FuelRecordCreate(**body)
	result = _run(svc.record_fuel_purchase(payload))
	return _ok(result.model_dump(mode="json"), 201)


# ──────────────────────────────────────────────────────────────────
# Maintenance
# ──────────────────────────────────────────────────────────────────

@fle_bp.get("/maintenance")
@_catch
def list_maintenance():
	svc = _get_service()
	vehicle_id = request.args.get("vehicle_id")
	status_raw = request.args.get("status")
	status = MaintenanceStatus(status_raw) if status_raw else None
	rows = _run(svc.list_maintenance(vehicle_id=vehicle_id, status=status))
	page, per_page = _page_args()
	return _ok(_paginate(rows, page, per_page))


@fle_bp.post("/maintenance")
@_catch
def create_maintenance():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
	payload = MaintenanceCreate(**body)
	result = _run(svc.schedule_maintenance(payload))
	return _ok(result.model_dump(mode="json"), 201)


@fle_bp.post("/maintenance/<maintenance_id>/start")
@_catch
def start_maintenance(maintenance_id: str):
	svc = _get_service()
	result = _run(svc.start_maintenance(maintenance_id))
	return _ok(result.model_dump(mode="json"))


@fle_bp.post("/maintenance/<maintenance_id>/complete")
@_catch
def complete_maintenance(maintenance_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	result = _run(svc.complete_maintenance(
		maintenance_id, Decimal(str(body.get("actual_cost", 0))), body.get("notes", "")
	))
	return _ok(result.model_dump(mode="json"))


# ──────────────────────────────────────────────────────────────────
# Inspections
# ──────────────────────────────────────────────────────────────────

@fle_bp.get("/inspections")
@_catch
def list_inspections():
	svc = _get_service()
	vehicle_id = request.args.get("vehicle_id")
	rows = _run(svc.list_inspections(vehicle_id=vehicle_id))
	page, per_page = _page_args()
	return _ok(_paginate(rows, page, per_page))


@fle_bp.post("/inspections")
@_catch
def create_inspection():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
	payload = InspectionCreate(**body)
	result = _run(svc.record_inspection(payload))
	return _ok(result.model_dump(mode="json"), 201)


@fle_bp.post("/inspections/<inspection_id>/process-failure")
@_catch
def process_inspection_failure(inspection_id: str):
	svc = _get_service()
	result = _run(svc.process_inspection_failure(inspection_id))
	return _ok(result)


# ──────────────────────────────────────────────────────────────────
# COF Inspections
# ──────────────────────────────────────────────────────────────────

@fle_bp.get("/cof")
@_catch
def list_cof():
	svc = _get_service()
	vehicle_id = request.args.get("vehicle_id")
	rows = _run(svc.list_cof_inspections(vehicle_id=vehicle_id))
	page, per_page = _page_args()
	return _ok(_paginate(rows, page, per_page))


@fle_bp.post("/cof")
@_catch
def create_cof():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
	payload = COFInspectionCreate(**body)
	result = _run(svc.record_cof_inspection(payload))
	return _ok(result.model_dump(mode="json"), 201)


# ──────────────────────────────────────────────────────────────────
# Incidents
# ──────────────────────────────────────────────────────────────────

@fle_bp.get("/incidents")
@_catch
def list_incidents():
	svc = _get_service()
	vehicle_id = request.args.get("vehicle_id")
	status_raw = request.args.get("status")
	status = IncidentStatus(status_raw) if status_raw else None
	rows = _run(svc.list_incidents(vehicle_id=vehicle_id, status=status))
	page, per_page = _page_args()
	return _ok(_paginate(rows, page, per_page))


@fle_bp.post("/incidents")
@_catch
def create_incident():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
	payload = IncidentCreate(**body)
	result = _run(svc.report_incident(payload))
	return _ok(result.model_dump(mode="json"), 201)


@fle_bp.post("/incidents/<incident_id>/close")
@_catch
def close_incident(incident_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	result = _run(svc.close_incident(incident_id, body.get("resolution", "")))
	return _ok(result.model_dump(mode="json"))


# ──────────────────────────────────────────────────────────────────
# Insurance
# ──────────────────────────────────────────────────────────────────

@fle_bp.get("/insurance")
@_catch
def list_insurance():
	svc = _get_service()
	vehicle_id = request.args.get("vehicle_id")
	rows = _run(svc.list_insurance_policies(vehicle_id=vehicle_id))
	page, per_page = _page_args()
	return _ok(_paginate(rows, page, per_page))


@fle_bp.post("/insurance")
@_catch
def create_insurance():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
	payload = InsurancePolicyCreate(**body)
	result = _run(svc.add_insurance_policy(payload))
	return _ok(result.model_dump(mode="json"), 201)


# ──────────────────────────────────────────────────────────────────
# Registrations
# ──────────────────────────────────────────────────────────────────

@fle_bp.get("/registrations")
@_catch
def list_registrations():
	svc = _get_service()
	vehicle_id = request.args.get("vehicle_id")
	rows = _run(svc.list_registrations(vehicle_id=vehicle_id))
	page, per_page = _page_args()
	return _ok(_paginate(rows, page, per_page))


@fle_bp.post("/registrations")
@_catch
def create_registration():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
	payload = RegistrationCreate(**body)
	result = _run(svc.register_vehicle_docs(payload))
	return _ok(result.model_dump(mode="json"), 201)


# ──────────────────────────────────────────────────────────────────
# Tachograph
# ──────────────────────────────────────────────────────────────────

@fle_bp.get("/tachograph")
@_catch
def list_tachograph():
	svc = _get_service()
	driver_id = request.args.get("driver_id")
	rows = _run(svc.list_tachograph_records(driver_id=driver_id))
	page, per_page = _page_args()
	return _ok(_paginate(rows, page, per_page))


@fle_bp.post("/tachograph")
@_catch
def create_tachograph():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
	payload = TachographRecordCreate(**body)
	result = _run(svc.record_tachograph(payload))
	return _ok(result.model_dump(mode="json"), 201)


# ──────────────────────────────────────────────────────────────────
# Telematics
# ──────────────────────────────────────────────────────────────────

@fle_bp.get("/telematics")
@_catch
def list_telematics():
	svc = _get_service()
	vehicle_id = request.args.get("vehicle_id")
	event_type = request.args.get("event_type")
	rows = _run(svc.list_telematics_events(vehicle_id=vehicle_id, event_type=event_type))
	page, per_page = _page_args()
	return _ok(_paginate(rows, page, per_page))


@fle_bp.post("/telematics")
@_catch
def create_telematics():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", request.headers.get("X-Tenant-ID", "default"))
	payload = TelematicsEventCreate(**body)
	result = _run(svc.track_vehicle_realtime(payload))
	return _ok(result.model_dump(mode="json"), 201)


@fle_bp.get("/telematics/position/<vehicle_id>")
@_catch
def vehicle_last_position(vehicle_id: str):
	svc = _get_service()
	result = _run(svc.get_vehicle_last_position(vehicle_id))
	if result is None:
		return _err(f"No position data for vehicle {vehicle_id}", "not_found", 404)
	return _ok(result.model_dump(mode="json"))


# ──────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────

@fle_bp.get("/reports/tco/<vehicle_id>")
@_catch
def report_tco(vehicle_id: str):
	svc = _get_service()
	result = _run(svc.calculate_tco(vehicle_id))
	return _ok(result.model_dump(mode="json"))


@fle_bp.get("/reports/utilisation")
@_catch
def report_utilisation():
	svc = _get_service()
	result = _run(svc.fleet_utilisation_analytics())
	return _ok(result.model_dump(mode="json"))


@fle_bp.get("/reports/compliance-calendar")
@_catch
def report_compliance_calendar():
	svc = _get_service()
	entries = _run(svc.compliance_calendar())
	return _ok([e.model_dump(mode="json") for e in entries])


@fle_bp.get("/reports/predictive-maintenance")
@_catch
def report_predictive_maintenance():
	svc = _get_service()
	alerts = _run(svc.predictive_maintenance_alerts())
	return _ok([a.model_dump(mode="json") for a in alerts])


@fle_bp.get("/reports/driver-score/<driver_id>")
@_catch
def report_driver_score(driver_id: str):
	svc = _get_service()
	result = _run(svc.driver_behaviour_scoring(driver_id))
	return _ok(result.model_dump(mode="json"))


# ──────────────────────────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────────────────────────

@fle_bp.get("/dashboard")
@_catch
def dashboard():
	svc = _get_service()
	result = _run(svc.dashboard_kpis())
	return _ok(result.model_dump(mode="json"))


# ──────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────

@fle_bp.get("/health")
def health():
	return _ok({"status": "ok", "capability": "transport_fle", "version": "2.0.0"})
