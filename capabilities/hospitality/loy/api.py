"""Flask Blueprint — REST API for Guest Loyalty Programme."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import LOYService

_log = logging.getLogger(__name__)

loy_bp = Blueprint("hos_loy", __name__, url_prefix="/api/hospitality/loy")
_svc = LOYService()


def _run(coro: Any) -> Any:
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@loy_bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@loy_bp.get("/members")
def list_members():
	tier = request.args.get("tier")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_members(_tenant(), tier=tier, status=status)))


@loy_bp.get("/members/<member_id>")
def get_member(member_id: str):
	try:
		return jsonify(_run(_svc.get_member(member_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@loy_bp.post("/members/enroll")
def enroll_member():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.enroll_member(
			guest_id=data.get("guest_id", ""),
			first_name=data["first_name"],
			last_name=data["last_name"],
			email=data["email"],
			phone=data.get("phone"),
			enrollment_source=data.get("enrollment_source", "front_desk"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@loy_bp.put("/members/<member_id>")
def update_member(member_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.update_member(member_id, data, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@loy_bp.delete("/members/<member_id>")
def deactivate_member(member_id: str):
	try:
		return jsonify(_run(_svc.delete_member(member_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@loy_bp.get("/members/<member_id>/transactions")
def list_transactions(member_id: str):
	txn_type = request.args.get("transaction_type")
	return jsonify(_run(_svc.list_transactions(member_id, _tenant(), transaction_type=txn_type)))


@loy_bp.post("/members/<member_id>/earn")
def earn_points(member_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.earn_points(
			member_id=member_id,
			spend_amount=float(data["spend_amount"]),
			description=data.get("description", "Stay points"),
			reference_id=data.get("reference_id"),
			nights=int(data.get("nights", 0)),
			tenant_id=_tenant(),
		)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@loy_bp.post("/members/<member_id>/redeem")
def redeem_points(member_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.redeem_points(
			member_id=member_id,
			points=int(data["points"]),
			description=data.get("description", "Redemption"),
			reference_id=data.get("reference_id"),
			tenant_id=_tenant(),
		)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@loy_bp.post("/members/<member_id>/adjust")
def adjust_points(member_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.adjust_points(
			member_id=member_id,
			points_delta=int(data["points_delta"]),
			reason=data.get("reason", "manual_adjustment"),
			adjusted_by=data.get("adjusted_by", "admin"),
			tenant_id=_tenant(),
		)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@loy_bp.post("/members/<member_id>/tier-upgrade")
def force_tier_upgrade(member_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.force_tier_upgrade(
			member_id=member_id,
			new_tier=data["new_tier"],
			reason=data.get("reason", "manual_upgrade"),
			upgraded_by=data.get("upgraded_by", "admin"),
			tenant_id=_tenant(),
		)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@loy_bp.get("/members/<member_id>/preferences")
def get_preferences(member_id: str):
	return jsonify(_run(_svc.get_recognition_preferences(member_id, _tenant())))


@loy_bp.put("/members/<member_id>/preferences")
def set_preferences(member_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.set_recognition_preferences(member_id, data, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@loy_bp.get("/partners")
def list_partners():
	return jsonify(_run(_svc.list_partners(_tenant())))


@loy_bp.post("/partners")
def create_partner():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_partner(
			partner_name=data["partner_name"],
			partner_type=data.get("partner_type", "airline"),
			earn_rate=float(data.get("earn_rate", 1.0)),
			redeem_rate=float(data.get("redeem_rate", 1.0)),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@loy_bp.post("/members/<member_id>/partner-earn")
def earn_partner_points(member_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.earn_partner_points(
			member_id=member_id,
			partner_id=data["partner_id"],
			partner_spend=float(data["partner_spend"]),
			description=data.get("description", "Partner transaction"),
			tenant_id=_tenant(),
		)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@loy_bp.post("/bonus-campaigns")
def create_bonus_campaign():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_bonus_campaign(
			name=data["name"],
			date_from=data["date_from"],
			date_to=data["date_to"],
			multiplier=float(data["multiplier"]),
			description=data.get("description"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@loy_bp.get("/bonus-campaigns")
def list_bonus_campaigns():
	return jsonify(_run(_svc.list_bonus_campaigns(_tenant())))


@loy_bp.get("/tier-distribution")
def tier_distribution():
	return jsonify(_run(_svc.tier_distribution(_tenant())))


@loy_bp.get("/dashboard")
def dashboard():
	return jsonify(_run(_svc.dashboard_summary(_tenant())))


@loy_bp.get("/audit-events")
def audit_events():
	return jsonify(_run(_svc.get_audit_events(_tenant())))
