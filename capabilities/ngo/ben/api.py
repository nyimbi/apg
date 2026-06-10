"""Beneficiary Registry — Flask Blueprint with async REST endpoints."""
from __future__ import annotations

import logging
from decimal import Decimal

from flask import Blueprint, jsonify, request

from .service import BeneficiaryRegistryService

_log = logging.getLogger(__name__)

bp = Blueprint("ngo_ben", __name__, url_prefix="/api/ngo/ben")

_svc: BeneficiaryRegistryService | None = None


def _get_service() -> BeneficiaryRegistryService:
	global _svc
	if _svc is None:
		_svc = BeneficiaryRegistryService()
	return _svc


def _run(coro):
	import asyncio
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor() as pool:
				return pool.submit(asyncio.run, coro).result()
		return loop.run_until_complete(coro)
	except Exception as exc:
		_log.error("async execution error: %s", exc)
		raise


@bp.get("/health")
def health():
	return jsonify(_run(_get_service().health_check())), 200


@bp.get("/")
def list_beneficiaries():
	svc = _get_service()
	result = _run(svc.list_beneficiaries(
		status=request.args.get("status"),
		county=request.args.get("county"),
		vulnerability_category=request.args.get("vulnerability_category"),
	))
	return jsonify({"beneficiaries": result, "count": len(result)}), 200


@bp.get("/<beneficiary_id>")
def get_beneficiary(beneficiary_id: str):
	try:
		return jsonify(_run(_get_service().get_beneficiary(beneficiary_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/")
def create_beneficiary():
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_beneficiary(
			first_name=data["first_name"],
			last_name=data["last_name"],
			national_id=data.get("national_id", ""),
			date_of_birth=data.get("date_of_birth", ""),
			gender=data.get("gender", "unknown"),
			phone=data.get("phone", ""),
			location=data.get("location", ""),
			county=data.get("county", ""),
			household_size=int(data.get("household_size", 1)),
			vulnerability_category=data.get("vulnerability_category", ""),
			notes=data.get("notes", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/<beneficiary_id>")
def update_beneficiary(beneficiary_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_get_service().update_beneficiary(beneficiary_id, **data))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/<beneficiary_id>")
def delete_beneficiary(beneficiary_id: str):
	try:
		return jsonify(_run(_get_service().delete_beneficiary(beneficiary_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/<beneficiary_id>/enrolments")
def list_enrolments(beneficiary_id: str):
	result = _run(_get_service().list_enrolments(beneficiary_id=beneficiary_id))
	return jsonify({"enrolments": result, "count": len(result)}), 200


@bp.post("/<beneficiary_id>/enrolments")
def enrol_beneficiary(beneficiary_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().enrol_beneficiary(
			beneficiary_id=beneficiary_id,
			programme_id=data["programme_id"],
			enrolment_date=data["enrolment_date"],
			enrolled_by=data["enrolled_by"],
			notes=data.get("notes", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/<beneficiary_id>/assessments")
def create_assessment(beneficiary_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_vulnerability_assessment(
			beneficiary_id=beneficiary_id,
			assessor=data["assessor"],
			assessment_date=data["assessment_date"],
			food_security_score=float(data.get("food_security_score", 0)),
			shelter_score=float(data.get("shelter_score", 0)),
			health_score=float(data.get("health_score", 0)),
			income_score=float(data.get("income_score", 0)),
			protection_score=float(data.get("protection_score", 0)),
			notes=data.get("notes", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/<beneficiary_id>/assessments")
def list_assessments(beneficiary_id: str):
	result = _run(_get_service().list_assessments(beneficiary_id=beneficiary_id))
	return jsonify({"assessments": result, "count": len(result)}), 200


@bp.get("/<beneficiary_id>/transfers")
def list_transfers(beneficiary_id: str):
	result = _run(_get_service().list_transfers(beneficiary_id=beneficiary_id))
	return jsonify({"transfers": result, "count": len(result)}), 200


@bp.post("/<beneficiary_id>/transfers")
def create_transfer(beneficiary_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_transfer(
			beneficiary_id=beneficiary_id,
			programme_id=data["programme_id"],
			amount=Decimal(str(data["amount"])),
			transfer_date=data["transfer_date"],
			reference=data["reference"],
			approved_by=data["approved_by"],
			currency=data.get("currency", "KES"),
			payment_method=data.get("payment_method", "mpesa"),
			notes=data.get("notes", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/<beneficiary_id>/dedup")
def check_duplicate(beneficiary_id: str):
	try:
		return jsonify(_run(_get_service().check_duplicate(beneficiary_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/analytics/vulnerability")
def vulnerability_distribution():
	return jsonify(_run(_get_service().vulnerability_distribution())), 200


@bp.post("/analytics/dedup-scan")
def bulk_deduplication_scan():
	return jsonify(_run(_get_service().bulk_deduplication_scan())), 200


@bp.get("/audit-events")
def get_audit_events():
	limit = int(request.args.get("limit", 100))
	result = _run(_get_service().get_audit_events(limit=limit))
	return jsonify({"events": result, "count": len(result)}), 200
