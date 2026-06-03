"""
Fleet Management UI views — Flask Blueprint.

url_prefix: /fle

All view functions are sync wrappers around async service calls.
"""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, abort, redirect, render_template, request, url_for

from .models import (
	DriverCreate, DriverUpdate,
	FuelRecordCreate,
	MaintenanceCreate,
	TripCreate, VehicleCreate, VehicleUpdate,
)
from .service import FleetService

fle_views_bp = Blueprint(
	"fle_views",
	__name__,
	url_prefix="/fle",
	template_folder="templates",
	static_folder="static",
)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

class _InMemoryDB:
	pass


def _svc() -> FleetService:
	tenant_id = request.cookies.get("tenant_id", "default")
	actor_id = request.cookies.get("user_id", "ui")
	return FleetService(_InMemoryDB(), tenant_id, actor_id)


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _ctx(**extra: Any) -> dict[str, Any]:
	return {
		"tenant_id": request.cookies.get("tenant_id", "default"),
		"user_id": request.cookies.get("user_id", "ui"),
		"capability": "Fleet Management",
		**extra,
	}


# ──────────────────────────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────────────────────────

def dashboard():
	"""Fleet KPI dashboard — active vehicles, trips, compliance alerts, predictive maintenance."""
	svc = _svc()
	kpis = _run(svc.dashboard_kpis())
	alerts = _run(svc.predictive_maintenance_alerts())
	calendar = _run(svc.compliance_calendar())
	critical = [e for e in calendar if e.severity == "critical"][:10]
	return render_template(
		"dashboards/fleet_dashboard.html",
		**_ctx(
			kpis=kpis.model_dump(mode="json"),
			predictive_alerts=alerts[:5],
			critical_compliance=critical,
			page_title="Fleet Dashboard",
		),
	)


# ──────────────────────────────────────────────────────────────────
# Vehicles
# ──────────────────────────────────────────────────────────────────

def list_vehicles():
	"""Vehicle fleet list with status filter."""
	from .models import VehicleStatus
	svc = _svc()
	status_raw = request.args.get("status")
	status = VehicleStatus(status_raw) if status_raw else None
	vehicles = _run(svc.list_vehicles(status=status))
	return render_template(
		"base/vehicle_list.html",
		**_ctx(vehicles=vehicles, status_filter=status_raw, page_title="Vehicles"),
	)


def detail_vehicle(vehicle_id: str):
	"""Vehicle detail — TCO, maintenance history, compliance, last position."""
	svc = _svc()
	try:
		vehicle = _run(svc.get_vehicle(vehicle_id))
		tco = _run(svc.calculate_tco(vehicle_id))
		maintenance = _run(svc.list_maintenance(vehicle_id=vehicle_id))
		inspections = _run(svc.list_inspections(vehicle_id=vehicle_id))
		fuel = _run(svc.list_fuel_records(vehicle_id=vehicle_id))
		trips = _run(svc.list_trips(vehicle_id=vehicle_id))
		incidents = _run(svc.list_incidents(vehicle_id=vehicle_id))
		position = _run(svc.get_vehicle_last_position(vehicle_id))
	except AssertionError:
		abort(404)
	return render_template(
		"base/vehicle_detail.html",
		**_ctx(
			vehicle=vehicle.model_dump(mode="json"),
			tco=tco.model_dump(mode="json"),
			maintenance=maintenance,
			inspections=inspections,
			fuel=fuel[-5:],
			trips=trips[-10:],
			incidents=incidents,
			position=position.model_dump(mode="json") if position else None,
			page_title=f"Vehicle: {vehicle.registration}",
		),
	)


def create_vehicle():
	"""Vehicle registration form."""
	from .models import FuelType, OwnershipType, VehicleType
	if request.method == "POST":
		svc = _svc()
		body = request.form.to_dict()
		body.setdefault("tenant_id", request.cookies.get("tenant_id", "default"))
		body.setdefault("created_by", request.cookies.get("user_id", "ui"))
		try:
			payload = VehicleCreate(**body)
			vehicle = _run(svc.register_vehicle(payload))
			return redirect(url_for("fle_views.detail_vehicle", vehicle_id=vehicle.id))
		except Exception as e:
			return render_template(
				"forms/vehicle_form.html",
				**_ctx(
					error=str(e), page_title="Register Vehicle",
					vehicle_types=[t.value for t in VehicleType],
					fuel_types=[f.value for f in FuelType],
					ownership_types=[o.value for o in OwnershipType],
				),
			)
	from .models import FuelType, OwnershipType, VehicleType
	return render_template(
		"forms/vehicle_form.html",
		**_ctx(
			page_title="Register Vehicle",
			vehicle_types=[t.value for t in VehicleType],
			fuel_types=[f.value for f in FuelType],
			ownership_types=[o.value for o in OwnershipType],
		),
	)


def edit_vehicle(vehicle_id: str):
	"""Edit vehicle form."""
	svc = _svc()
	try:
		vehicle = _run(svc.get_vehicle(vehicle_id))
	except AssertionError:
		abort(404)
	if request.method == "POST":
		body = request.form.to_dict()
		try:
			patch = VehicleUpdate(**{k: v for k, v in body.items() if v})
			_run(svc.update_vehicle(vehicle_id, patch))
			return redirect(url_for("fle_views.detail_vehicle", vehicle_id=vehicle_id))
		except Exception as e:
			return render_template(
				"forms/vehicle_edit.html",
				**_ctx(vehicle=vehicle.model_dump(mode="json"), error=str(e), page_title="Edit Vehicle"),
			)
	return render_template(
		"forms/vehicle_edit.html",
		**_ctx(vehicle=vehicle.model_dump(mode="json"), page_title="Edit Vehicle"),
	)


# ──────────────────────────────────────────────────────────────────
# Drivers
# ──────────────────────────────────────────────────────────────────

def list_drivers():
	"""Driver roster with licence expiry warnings."""
	from .models import DriverStatus
	svc = _svc()
	status_raw = request.args.get("status")
	status = DriverStatus(status_raw) if status_raw else None
	drivers = _run(svc.list_drivers(status=status))
	return render_template(
		"base/driver_list.html",
		**_ctx(drivers=drivers, status_filter=status_raw, page_title="Drivers"),
	)


def detail_driver(driver_id: str):
	"""Driver detail — trips, tachograph, behaviour score."""
	svc = _svc()
	try:
		driver = _run(svc.get_driver(driver_id))
		score = _run(svc.driver_behaviour_scoring(driver_id))
		trips = _run(svc.list_trips(driver_id=driver_id))
		tacho = _run(svc.list_tachograph_records(driver_id=driver_id))
	except AssertionError:
		abort(404)
	return render_template(
		"base/driver_detail.html",
		**_ctx(
			driver=driver.model_dump(mode="json"),
			score=score.model_dump(mode="json"),
			trips=trips[-10:],
			tachograph=tacho[-20:],
			page_title=f"Driver: {driver.name}",
		),
	)


def create_driver():
	"""Driver registration form."""
	from .models import LicenceClass
	if request.method == "POST":
		svc = _svc()
		body = request.form.to_dict()
		body.setdefault("tenant_id", request.cookies.get("tenant_id", "default"))
		body.setdefault("created_by", request.cookies.get("user_id", "ui"))
		try:
			payload = DriverCreate(**body)
			driver = _run(svc.register_driver(payload))
			return redirect(url_for("fle_views.detail_driver", driver_id=driver.id))
		except Exception as e:
			return render_template(
				"forms/driver_form.html",
				**_ctx(
					error=str(e), page_title="Register Driver",
					licence_classes=[c.value for c in LicenceClass],
				),
			)
	return render_template(
		"forms/driver_form.html",
		**_ctx(page_title="Register Driver", licence_classes=[c.value for c in LicenceClass]),
	)


def edit_driver(driver_id: str):
	"""Edit driver form."""
	svc = _svc()
	try:
		driver = _run(svc.get_driver(driver_id))
	except AssertionError:
		abort(404)
	if request.method == "POST":
		body = request.form.to_dict()
		try:
			patch = DriverUpdate(**{k: v for k, v in body.items() if v})
			_run(svc.update_driver(driver_id, patch))
			return redirect(url_for("fle_views.detail_driver", driver_id=driver_id))
		except Exception as e:
			return render_template(
				"forms/driver_edit.html",
				**_ctx(driver=driver.model_dump(mode="json"), error=str(e), page_title="Edit Driver"),
			)
	return render_template(
		"forms/driver_edit.html",
		**_ctx(driver=driver.model_dump(mode="json"), page_title="Edit Driver"),
	)


# ──────────────────────────────────────────────────────────────────
# Trips
# ──────────────────────────────────────────────────────────────────

def list_trips():
	"""Trip list with status filter."""
	from .models import TripStatus
	svc = _svc()
	status_raw = request.args.get("status")
	status = TripStatus(status_raw) if status_raw else None
	trips = _run(svc.list_trips(status=status))
	return render_template(
		"base/trip_list.html",
		**_ctx(trips=trips, status_filter=status_raw, page_title="Trips"),
	)


def detail_trip(trip_id: str):
	"""Trip detail — status, route, fuel, timeline."""
	svc = _svc()
	try:
		trip = _run(svc.get_trip(trip_id))
	except AssertionError:
		abort(404)
	return render_template(
		"base/trip_detail.html",
		**_ctx(
			trip=trip.model_dump(mode="json"),
			page_title=f"Trip: {trip.origin} → {trip.destination}",
		),
	)


def create_trip():
	"""Trip planning form."""
	svc = _svc()
	if request.method == "POST":
		body = request.form.to_dict()
		body.setdefault("tenant_id", request.cookies.get("tenant_id", "default"))
		body.setdefault("created_by", request.cookies.get("user_id", "ui"))
		try:
			payload = TripCreate(**body)
			trip = _run(svc.plan_trip(payload))
			return redirect(url_for("fle_views.detail_trip", trip_id=trip.id))
		except Exception as e:
			vehicles = _run(svc.list_vehicles())
			drivers = _run(svc.list_drivers())
			return render_template(
				"forms/trip_form.html",
				**_ctx(error=str(e), vehicles=vehicles, drivers=drivers, page_title="Plan Trip"),
			)
	vehicles = _run(svc.list_vehicles())
	drivers = _run(svc.list_drivers())
	return render_template(
		"forms/trip_form.html",
		**_ctx(vehicles=vehicles, drivers=drivers, page_title="Plan Trip"),
	)


# ──────────────────────────────────────────────────────────────────
# Maintenance
# ──────────────────────────────────────────────────────────────────

def list_maintenance():
	"""Maintenance schedule — upcoming and overdue."""
	from .models import MaintenanceStatus
	svc = _svc()
	status_raw = request.args.get("status")
	status = MaintenanceStatus(status_raw) if status_raw else None
	maintenance = _run(svc.list_maintenance(status=status))
	return render_template(
		"base/maintenance_list.html",
		**_ctx(maintenance=maintenance, status_filter=status_raw, page_title="Maintenance"),
	)


def create_maintenance():
	"""Schedule maintenance form."""
	from .models import MaintenanceType
	svc = _svc()
	if request.method == "POST":
		body = request.form.to_dict()
		body.setdefault("tenant_id", request.cookies.get("tenant_id", "default"))
		body.setdefault("created_by", request.cookies.get("user_id", "ui"))
		try:
			payload = MaintenanceCreate(**body)
			_run(svc.schedule_maintenance(payload))
			return redirect(url_for("fle_views.list_maintenance"))
		except Exception as e:
			vehicles = _run(svc.list_vehicles())
			return render_template(
				"forms/maintenance_form.html",
				**_ctx(
					error=str(e), vehicles=vehicles,
					maintenance_types=[t.value for t in MaintenanceType],
					page_title="Schedule Maintenance",
				),
			)
	vehicles = _run(svc.list_vehicles())
	return render_template(
		"forms/maintenance_form.html",
		**_ctx(
			vehicles=vehicles,
			maintenance_types=[t.value for t in MaintenanceType],
			page_title="Schedule Maintenance",
		),
	)


# ──────────────────────────────────────────────────────────────────
# Compliance calendar
# ──────────────────────────────────────────────────────────────────

def compliance_calendar():
	"""Compliance calendar — all upcoming/overdue events."""
	svc = _svc()
	entries = _run(svc.compliance_calendar())
	critical = [e for e in entries if e.severity == "critical"]
	warning = [e for e in entries if e.severity == "warning"]
	info = [e for e in entries if e.severity == "info"]
	return render_template(
		"base/compliance_calendar.html",
		**_ctx(
			critical=critical, warning=warning, info=info,
			total=len(entries),
			page_title="Compliance Calendar",
		),
	)


# ──────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────

def fleet_utilisation_report():
	"""Fleet utilisation analytics report."""
	svc = _svc()
	report = _run(svc.fleet_utilisation_analytics())
	return render_template(
		"base/utilisation_report.html",
		**_ctx(report=report.model_dump(mode="json"), page_title="Fleet Utilisation"),
	)


def tco_report(vehicle_id: str):
	"""Total Cost of Ownership report for a specific vehicle."""
	svc = _svc()
	try:
		vehicle = _run(svc.get_vehicle(vehicle_id))
		tco = _run(svc.calculate_tco(vehicle_id))
	except AssertionError:
		abort(404)
	return render_template(
		"base/tco_report.html",
		**_ctx(
			vehicle=vehicle.model_dump(mode="json"),
			tco=tco.model_dump(mode="json"),
			page_title=f"TCO: {vehicle.registration}",
		),
	)


def predictive_maintenance_report():
	"""Predictive maintenance alerts report."""
	svc = _svc()
	alerts = _run(svc.predictive_maintenance_alerts())
	critical = [a for a in alerts if a.urgency == "critical"]
	high = [a for a in alerts if a.urgency == "high"]
	medium = [a for a in alerts if a.urgency == "medium"]
	return render_template(
		"base/predictive_maintenance.html",
		**_ctx(
			critical=critical, high=high, medium=medium,
			total=len(alerts),
			page_title="Predictive Maintenance",
		),
	)


# ──────────────────────────────────────────────────────────────────
# Route registration
# ──────────────────────────────────────────────────────────────────

fle_views_bp.add_url_rule("/", "dashboard", dashboard)
fle_views_bp.add_url_rule("/vehicles", "list_vehicles", list_vehicles)
fle_views_bp.add_url_rule("/vehicles/new", "create_vehicle", create_vehicle, methods=["GET", "POST"])
fle_views_bp.add_url_rule("/vehicles/<vehicle_id>", "detail_vehicle", detail_vehicle)
fle_views_bp.add_url_rule("/vehicles/<vehicle_id>/edit", "edit_vehicle", edit_vehicle, methods=["GET", "POST"])
fle_views_bp.add_url_rule("/drivers", "list_drivers", list_drivers)
fle_views_bp.add_url_rule("/drivers/new", "create_driver", create_driver, methods=["GET", "POST"])
fle_views_bp.add_url_rule("/drivers/<driver_id>", "detail_driver", detail_driver)
fle_views_bp.add_url_rule("/drivers/<driver_id>/edit", "edit_driver", edit_driver, methods=["GET", "POST"])
fle_views_bp.add_url_rule("/trips", "list_trips", list_trips)
fle_views_bp.add_url_rule("/trips/new", "create_trip", create_trip, methods=["GET", "POST"])
fle_views_bp.add_url_rule("/trips/<trip_id>", "detail_trip", detail_trip)
fle_views_bp.add_url_rule("/maintenance", "list_maintenance", list_maintenance)
fle_views_bp.add_url_rule("/maintenance/new", "create_maintenance", create_maintenance, methods=["GET", "POST"])
fle_views_bp.add_url_rule("/compliance", "compliance_calendar", compliance_calendar)
fle_views_bp.add_url_rule("/reports/utilisation", "fleet_utilisation_report", fleet_utilisation_report)
fle_views_bp.add_url_rule("/reports/tco/<vehicle_id>", "tco_report", tco_report)
fle_views_bp.add_url_rule("/reports/predictive-maintenance", "predictive_maintenance_report", predictive_maintenance_report)
