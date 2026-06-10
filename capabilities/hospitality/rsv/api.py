"""Flask Blueprint — REST API for Reservations & Channel Manager."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import RSVService

_log = logging.getLogger(__name__)

rsv_bp = Blueprint("hos_rsv", __name__, url_prefix="/api/hospitality/rsv")
_svc = RSVService()


def _run(coro: Any) -> Any:
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@rsv_bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@rsv_bp.get("/channels")
def list_channels():
	channel_type = request.args.get("channel_type")
	return jsonify(_run(_svc.list_channels(_tenant(), channel_type=channel_type)))


@rsv_bp.get("/channels/<channel_id>")
def get_channel(channel_id: str):
	try:
		return jsonify(_run(_svc.get_channel(channel_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@rsv_bp.post("/channels")
def create_channel():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_channel(
			code=data["code"],
			name=data["name"],
			channel_type=data.get("channel_type", "ota"),
			commission_pct=float(data.get("commission_pct", 0)),
			api_endpoint=data.get("api_endpoint"),
			credentials_ref=data.get("credentials_ref"),
			is_active=data.get("is_active", True),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@rsv_bp.put("/channels/<channel_id>")
def update_channel(channel_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.update_channel(channel_id, data, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@rsv_bp.delete("/channels/<channel_id>")
def delete_channel(channel_id: str):
	try:
		return jsonify(_run(_svc.delete_channel(channel_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@rsv_bp.get("/bookings")
def list_bookings():
	channel_id = request.args.get("channel_id")
	status = request.args.get("status")
	date_from = request.args.get("date_from")
	return jsonify(_run(_svc.list_bookings(_tenant(), channel_id=channel_id, status=status, date_from=date_from)))


@rsv_bp.get("/bookings/<booking_id>")
def get_booking(booking_id: str):
	try:
		return jsonify(_run(_svc.get_booking(booking_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@rsv_bp.post("/bookings")
def create_booking():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_booking(
			channel_id=data["channel_id"],
			guest_name=data["guest_name"],
			guest_email=data.get("guest_email", ""),
			room_type=data["room_type"],
			check_in_date=data["check_in_date"],
			check_out_date=data["check_out_date"],
			rate=float(data["rate"]),
			adults=data.get("adults", 1),
			children=data.get("children", 0),
			guest_phone=data.get("guest_phone"),
			external_booking_ref=data.get("external_booking_ref"),
			special_requests=data.get("special_requests"),
			currency=data.get("currency", "KES"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@rsv_bp.put("/bookings/<booking_id>")
def update_booking(booking_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.update_booking(booking_id, data, _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@rsv_bp.delete("/bookings/<booking_id>")
def cancel_booking(booking_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.cancel_booking(booking_id, data.get("reason", "cancelled"), data.get("cancelled_by", "system"), _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@rsv_bp.get("/availability")
def get_availability():
	room_type = request.args.get("room_type", "")
	date = request.args.get("date", "")
	return jsonify(_run(_svc.get_availability(room_type, date, _tenant())))


@rsv_bp.put("/availability")
def set_availability():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.set_availability(
			room_type=data["room_type"],
			date=data["date"],
			available_count=int(data["available_count"]),
			stop_sell=data.get("stop_sell", False),
			tenant_id=_tenant(),
		)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@rsv_bp.put("/availability/bulk")
def bulk_availability():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.bulk_set_availability(
			room_type=data["room_type"],
			date_from=data["date_from"],
			date_to=data["date_to"],
			available_count=int(data["available_count"]),
			stop_sell=data.get("stop_sell", False),
			tenant_id=_tenant(),
		)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@rsv_bp.post("/gds-connections")
def create_gds_connection():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_gds_connection(
			gds_provider=data["gds_provider"],
			property_code=data["property_code"],
			credentials_ref=data["credentials_ref"],
			chain_code=data.get("chain_code"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@rsv_bp.get("/gds-connections")
def list_gds_connections():
	return jsonify(_run(_svc.list_gds_connections(_tenant())))


@rsv_bp.post("/gds-connections/<connection_id>/sync")
def sync_gds(connection_id: str):
	try:
		return jsonify(_run(_svc.sync_gds_availability(connection_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@rsv_bp.post("/waitlist")
def add_to_waitlist():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.add_to_waitlist(
			guest_name=data["guest_name"],
			guest_email=data["guest_email"],
			room_type=data["room_type"],
			check_in_date=data["check_in_date"],
			check_out_date=data["check_out_date"],
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@rsv_bp.get("/waitlist")
def list_waitlist():
	room_type = request.args.get("room_type")
	return jsonify(_run(_svc.list_waitlist(_tenant(), room_type=room_type)))


@rsv_bp.get("/channel-performance")
def channel_performance():
	date_from = request.args.get("date_from")
	return jsonify(_run(_svc.channel_performance(_tenant(), date_from=date_from)))


@rsv_bp.get("/dashboard")
def dashboard():
	return jsonify(_run(_svc.dashboard_summary(_tenant())))


@rsv_bp.get("/audit-events")
def audit_events():
	return jsonify(_run(_svc.get_audit_events(_tenant())))
