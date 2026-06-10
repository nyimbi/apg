"""Flask Blueprint — REST API for Events & Venue Management."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import EVNService

_log = logging.getLogger(__name__)

evn_bp = Blueprint("hos_evn", __name__, url_prefix="/api/hospitality/evn")
_svc = EVNService()


def _run(coro: Any) -> Any:
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@evn_bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@evn_bp.get("/venues")
def list_venues():
	venue_type = request.args.get("venue_type")
	available_date = request.args.get("available_date")
	return jsonify(_run(_svc.list_venues(_tenant(), venue_type=venue_type, available_date=available_date)))


@evn_bp.get("/venues/<venue_id>")
def get_venue(venue_id: str):
	try:
		return jsonify(_run(_svc.get_venue(venue_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@evn_bp.post("/venues")
def create_venue():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_venue(
			name=data["name"],
			venue_type=data.get("venue_type", "conference_room"),
			capacity_seated=int(data.get("capacity_seated", 50)),
			capacity_standing=int(data.get("capacity_standing", 0)),
			area_sqm=float(data.get("area_sqm", 0)),
			rental_rate_per_day=float(data.get("rental_rate_per_day", 0)),
			av_included=data.get("av_included", False),
			catering_allowed=data.get("catering_allowed", True),
			notes=data.get("notes"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@evn_bp.put("/venues/<venue_id>")
def update_venue(venue_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.update_venue(venue_id, data, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@evn_bp.delete("/venues/<venue_id>")
def delete_venue(venue_id: str):
	try:
		return jsonify(_run(_svc.delete_venue(venue_id, _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@evn_bp.get("/event-bookings")
def list_event_bookings():
	venue_id = request.args.get("venue_id")
	event_type = request.args.get("event_type")
	date_from = request.args.get("date_from")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_event_bookings(_tenant(), venue_id=venue_id, event_type=event_type, date_from=date_from, status=status)))


@evn_bp.get("/event-bookings/<booking_id>")
def get_event_booking(booking_id: str):
	try:
		return jsonify(_run(_svc.get_event_booking(booking_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@evn_bp.post("/event-bookings")
def create_event_booking():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_event_booking(
			venue_id=data["venue_id"],
			event_name=data["event_name"],
			client_name=data["client_name"],
			client_email=data.get("client_email", ""),
			event_type=data.get("event_type", "other"),
			event_date=data["event_date"],
			start_time=data.get("start_time", "09:00"),
			end_time=data.get("end_time", "17:00"),
			expected_attendance=int(data.get("expected_attendance", 1)),
			catering_required=data.get("catering_required", False),
			av_required=data.get("av_required", False),
			decoration_required=data.get("decoration_required", False),
			client_phone=data.get("client_phone"),
			notes=data.get("notes"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@evn_bp.put("/event-bookings/<booking_id>")
def update_event_booking(booking_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.update_event_booking(booking_id, data, _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@evn_bp.post("/event-bookings/<booking_id>/confirm")
def confirm_event_booking(booking_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.confirm_event_booking(booking_id, float(data.get("deposit_amount", 0)), _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@evn_bp.delete("/event-bookings/<booking_id>")
def cancel_event_booking(booking_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.delete_event_booking(booking_id, data.get("reason", "client_cancellation"), _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@evn_bp.post("/beos")
def generate_beo():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.generate_beo(
			event_booking_id=data["event_booking_id"],
			menu_selections=data.get("menu_selections", []),
			av_requirements=data.get("av_requirements", []),
			setup_style=data.get("setup_style", "theatre"),
			special_requirements=data.get("special_requirements"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@evn_bp.get("/beos")
def list_beos():
	return jsonify(_run(_svc.list_beos(_tenant())))


@evn_bp.get("/beos/<beo_id>")
def get_beo(beo_id: str):
	try:
		return jsonify(_run(_svc.get_beo(beo_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@evn_bp.post("/beos/<beo_id>/finalise")
def finalise_beo(beo_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.finalise_beo(beo_id, data.get("approved_by", "events_manager"), _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@evn_bp.post("/contracts")
def issue_contract():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.issue_contract(
			event_booking_id=data["event_booking_id"],
			deposit_pct=float(data.get("deposit_pct", 30)),
			payment_terms=data.get("payment_terms", "50% 30 days before, balance on day"),
			cancellation_policy=data.get("cancellation_policy", "standard"),
			special_clauses=data.get("special_clauses"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@evn_bp.get("/contracts")
def list_contracts():
	return jsonify(_run(_svc.list_contracts(_tenant())))


@evn_bp.post("/contracts/<contract_id>/sign")
def sign_contract(contract_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.sign_contract(contract_id, data.get("signed_by", "client"), _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@evn_bp.post("/event-bookings/<booking_id>/payments")
def record_payment(booking_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.record_event_payment(
			event_booking_id=booking_id,
			amount=float(data["amount"]),
			payment_type=data.get("payment_type", "deposit"),
			payment_method=data.get("payment_method", "bank_transfer"),
			reference=data.get("reference"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@evn_bp.get("/utilisation-report")
def utilisation_report():
	date_from = request.args.get("date_from", "")
	date_to = request.args.get("date_to", "9999-12-31")
	return jsonify(_run(_svc.venue_utilisation_report(date_from, date_to, _tenant())))


@evn_bp.get("/dashboard")
def dashboard():
	return jsonify(_run(_svc.dashboard_summary(_tenant())))


@evn_bp.get("/audit-events")
def audit_events():
	return jsonify(_run(_svc.get_audit_events(_tenant())))
