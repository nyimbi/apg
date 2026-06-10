"""Flask Blueprint — REST API for Spa & Activities Management."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import SPAService

_log = logging.getLogger(__name__)

spa_bp = Blueprint("hos_spa", __name__, url_prefix="/api/hospitality/spa")
_svc = SPAService()


def _run(coro: Any) -> Any:
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@spa_bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@spa_bp.get("/treatments")
def list_treatments():
	category = request.args.get("category")
	active_only = request.args.get("active_only", "false").lower() == "true"
	return jsonify(_run(_svc.list_treatments(_tenant(), category=category, active_only=active_only)))


@spa_bp.get("/treatments/<treatment_id>")
def get_treatment(treatment_id: str):
	try:
		return jsonify(_run(_svc.get_treatment(treatment_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@spa_bp.post("/treatments")
def create_treatment():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_treatment(
			name=data["name"],
			category=data.get("category", "massage"),
			duration_mins=int(data.get("duration_mins", 60)),
			price=float(data["price"]),
			therapist_required=int(data.get("therapist_required", 1)),
			description=data.get("description"),
			is_active=data.get("is_active", True),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@spa_bp.put("/treatments/<treatment_id>")
def update_treatment(treatment_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.update_treatment(treatment_id, data, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@spa_bp.delete("/treatments/<treatment_id>")
def delete_treatment(treatment_id: str):
	try:
		return jsonify(_run(_svc.delete_treatment(treatment_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@spa_bp.get("/therapists")
def list_therapists():
	specialisation = request.args.get("specialisation")
	return jsonify(_run(_svc.list_therapists(_tenant(), specialisation=specialisation)))


@spa_bp.post("/therapists")
def create_therapist():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_therapist(
			first_name=data["first_name"],
			last_name=data["last_name"],
			specialisations=data.get("specialisations", []),
			employment_type=data.get("employment_type", "full_time"),
			phone=data.get("phone"),
			email=data.get("email"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@spa_bp.get("/therapists/<therapist_id>")
def get_therapist(therapist_id: str):
	try:
		return jsonify(_run(_svc.get_therapist(therapist_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@spa_bp.get("/therapists/<therapist_id>/schedule")
def get_therapist_schedule(therapist_id: str):
	date = request.args.get("date", "")
	return jsonify(_run(_svc.get_therapist_schedule(therapist_id, date, _tenant())))


@spa_bp.get("/appointments")
def list_appointments():
	date = request.args.get("date")
	therapist_id = request.args.get("therapist_id")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_appointments(_tenant(), date=date, therapist_id=therapist_id, status=status)))


@spa_bp.get("/appointments/<appointment_id>")
def get_appointment(appointment_id: str):
	try:
		return jsonify(_run(_svc.get_appointment(appointment_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@spa_bp.post("/appointments")
def create_appointment():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_appointment(
			guest_name=data["guest_name"],
			guest_email=data.get("guest_email", ""),
			treatment_id=data["treatment_id"],
			appointment_date=data["appointment_date"],
			start_time=data["start_time"],
			therapist_id=data.get("therapist_id"),
			reservation_id=data.get("reservation_id"),
			special_notes=data.get("special_notes"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@spa_bp.put("/appointments/<appointment_id>")
def update_appointment(appointment_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.update_appointment(appointment_id, data, _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@spa_bp.delete("/appointments/<appointment_id>")
def cancel_appointment(appointment_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.delete_appointment(appointment_id, data.get("reason", "cancelled"), _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@spa_bp.post("/appointments/<appointment_id>/complete")
def complete_appointment(appointment_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.complete_appointment(appointment_id, data.get("payment_method", "cash"), _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@spa_bp.get("/memberships")
def list_memberships():
	status = request.args.get("status")
	return jsonify(_run(_svc.list_memberships(_tenant(), status=status)))


@spa_bp.post("/memberships")
def create_membership():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_membership(
			guest_name=data["guest_name"],
			guest_email=data["guest_email"],
			membership_type=data.get("membership_type", "basic"),
			price=float(data["price"]),
			valid_months=int(data.get("valid_months", 12)),
			included_treatments=int(data.get("included_treatments", 0)),
			discount_pct=float(data.get("discount_pct", 0)),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@spa_bp.post("/memberships/<membership_id>/renew")
def renew_membership(membership_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.renew_membership(membership_id, int(data.get("months", 12)), _tenant())))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@spa_bp.get("/retail")
def list_retail():
	category = request.args.get("category")
	return jsonify(_run(_svc.list_retail_items(_tenant(), category=category)))


@spa_bp.post("/retail")
def create_retail_item():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_retail_item(
			name=data["name"],
			category=data.get("category", "skincare"),
			price=float(data["price"]),
			cost=float(data.get("cost", 0)),
			stock_quantity=int(data.get("stock_quantity", 0)),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@spa_bp.post("/retail/<item_id>/sell")
def sell_retail(item_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.sell_retail_item(
			item_id=item_id,
			quantity=int(data.get("quantity", 1)),
			guest_name=data.get("guest_name", ""),
			payment_method=data.get("payment_method", "cash"),
			tenant_id=_tenant(),
		)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@spa_bp.get("/revenue-report")
def revenue_report():
	from datetime import date
	date_str = request.args.get("date", date.today().isoformat())
	return jsonify(_run(_svc.revenue_report(date_str, _tenant())))


@spa_bp.get("/therapist-utilisation")
def therapist_utilisation():
	from datetime import date
	date_str = request.args.get("date", date.today().isoformat())
	return jsonify(_run(_svc.therapist_utilisation(date_str, _tenant())))


@spa_bp.get("/dashboard")
def dashboard():
	return jsonify(_run(_svc.dashboard_summary(_tenant())))


@spa_bp.get("/audit-events")
def audit_events():
	return jsonify(_run(_svc.get_audit_events(_tenant())))
