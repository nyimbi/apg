"""Flask Blueprint — REST API for Property Management System."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import PMSService

_log = logging.getLogger(__name__)

pms_bp = Blueprint("hos_pms", __name__, url_prefix="/api/hospitality/pms")
_svc = PMSService()


def _run(coro: Any) -> Any:
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@pms_bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@pms_bp.get("/rooms")
def list_rooms():
	status = request.args.get("status")
	room_type = request.args.get("room_type")
	return jsonify(_run(_svc.list_rooms(_tenant(), status=status, room_type=room_type)))


@pms_bp.get("/rooms/<room_id>")
def get_room(room_id: str):
	try:
		return jsonify(_run(_svc.get_room(room_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@pms_bp.post("/rooms")
def create_room():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_room(
			room_number=data["room_number"],
			room_type=data.get("room_type", "standard"),
			floor=data.get("floor", 1),
			capacity=data.get("capacity", 2),
			rate_per_night=float(data.get("rate_per_night", 0)),
			amenities=data.get("amenities", []),
			notes=data.get("notes"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@pms_bp.put("/rooms/<room_id>")
def update_room(room_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.update_room(room_id, data, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@pms_bp.delete("/rooms/<room_id>")
def delete_room(room_id: str):
	try:
		return jsonify(_run(_svc.delete_room(room_id, _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@pms_bp.get("/rooms/availability")
def room_availability():
	check_in = request.args.get("check_in", "")
	check_out = request.args.get("check_out", "")
	room_type = request.args.get("room_type")
	return jsonify(_run(_svc.get_room_availability(check_in, check_out, room_type, _tenant())))


@pms_bp.get("/guests")
def list_guests():
	vip_level = request.args.get("vip_level")
	return jsonify(_run(_svc.list_guests(_tenant(), vip_level=vip_level)))


@pms_bp.get("/guests/<guest_id>")
def get_guest(guest_id: str):
	try:
		return jsonify(_run(_svc.get_guest(guest_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@pms_bp.post("/guests")
def create_guest():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_guest(
			first_name=data["first_name"],
			last_name=data["last_name"],
			email=data.get("email", ""),
			phone=data.get("phone"),
			nationality=data.get("nationality"),
			id_type=data.get("id_type"),
			id_number=data.get("id_number"),
			vip_level=data.get("vip_level", "standard"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@pms_bp.put("/guests/<guest_id>")
def update_guest(guest_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.update_guest(guest_id, data, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@pms_bp.get("/reservations")
def list_reservations():
	status = request.args.get("status")
	return jsonify(_run(_svc.list_reservations(_tenant(), status=status)))


@pms_bp.get("/reservations/<reservation_id>")
def get_reservation(reservation_id: str):
	try:
		return jsonify(_run(_svc.get_reservation(reservation_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@pms_bp.post("/reservations")
def create_reservation():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_reservation(
			guest_id=data["guest_id"],
			room_id=data["room_id"],
			check_in_date=data["check_in_date"],
			check_out_date=data["check_out_date"],
			adults=data.get("adults", 1),
			children=data.get("children", 0),
			rate_plan=data.get("rate_plan", "standard"),
			special_requests=data.get("special_requests"),
			source=data.get("source", "direct"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@pms_bp.put("/reservations/<reservation_id>")
def update_reservation(reservation_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.update_reservation(reservation_id, data, _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@pms_bp.delete("/reservations/<reservation_id>")
def cancel_reservation(reservation_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.delete_reservation(reservation_id, data.get("reason", "guest_request"), _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@pms_bp.post("/reservations/<reservation_id>/check-in")
def check_in(reservation_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.check_in(reservation_id, data.get("checked_in_by", "front_desk"), _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@pms_bp.post("/reservations/<reservation_id>/check-out")
def check_out(reservation_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.check_out(reservation_id, data.get("checked_out_by", "front_desk"), data.get("final_payment", 0.0), _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@pms_bp.get("/reservations/<reservation_id>/folio")
def get_folio(reservation_id: str):
	try:
		return jsonify(_run(_svc.get_folio_summary(reservation_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@pms_bp.post("/reservations/<reservation_id>/folio/charges")
def add_charge(reservation_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.add_folio_charge(
			reservation_id,
			data.get("charge_type", "other"),
			data["description"],
			float(data["amount"]),
			int(data.get("quantity", 1)),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@pms_bp.get("/housekeeping")
def list_housekeeping():
	status = request.args.get("status")
	return jsonify(_run(_svc.list_housekeeping_tasks(_tenant(), status=status)))


@pms_bp.post("/housekeeping")
def create_housekeeping_task():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_housekeeping_task(
			room_id=data["room_id"],
			task_type=data.get("task_type", "clean"),
			priority=data.get("priority", "normal"),
			assigned_to=data.get("assigned_to"),
			notes=data.get("notes"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@pms_bp.post("/housekeeping/<task_id>/complete")
def complete_housekeeping(task_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.complete_housekeeping_task(task_id, data.get("completed_by", "staff"), _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@pms_bp.post("/night-audit")
def run_night_audit():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.run_night_audit(data["audit_date"], data.get("run_by", "system"), _tenant()))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@pms_bp.get("/dashboard")
def dashboard():
	return jsonify(_run(_svc.dashboard_summary(_tenant())))


@pms_bp.get("/audit-events")
def audit_events():
	return jsonify(_run(_svc.get_audit_events(_tenant())))
