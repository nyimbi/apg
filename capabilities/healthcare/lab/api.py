"""Flask Blueprint REST API for APG Laboratory Information System.

Endpoints
---------
GET  /api/healthcare/lab/contract
GET  /api/healthcare/lab/dashboard

GET  /api/healthcare/lab/tests
POST /api/healthcare/lab/tests
GET  /api/healthcare/lab/tests/<id>
PUT  /api/healthcare/lab/tests/<id>
DELETE /api/healthcare/lab/tests/<id>

GET  /api/healthcare/lab/orders
POST /api/healthcare/lab/orders
GET  /api/healthcare/lab/orders/<id>
PUT  /api/healthcare/lab/orders/<id>
DELETE /api/healthcare/lab/orders/<id>
POST /api/healthcare/lab/orders/<id>/cancel
POST /api/healthcare/lab/orders/<id>/receive
POST /api/healthcare/lab/orders/<id>/hold
POST /api/healthcare/lab/orders/<id>/unhold

GET  /api/healthcare/lab/specimens
POST /api/healthcare/lab/specimens
GET  /api/healthcare/lab/specimens/<id>
PUT  /api/healthcare/lab/specimens/<id>
POST /api/healthcare/lab/specimens/<id>/reject
POST /api/healthcare/lab/specimens/<id>/receive
POST /api/healthcare/lab/specimens/<id>/track
GET  /api/healthcare/lab/specimens/<id>/custody

GET  /api/healthcare/lab/reference-ranges
POST /api/healthcare/lab/reference-ranges
GET  /api/healthcare/lab/reference-ranges/<id>
PUT  /api/healthcare/lab/reference-ranges/<id>
DELETE /api/healthcare/lab/reference-ranges/<id>

GET  /api/healthcare/lab/results
POST /api/healthcare/lab/results
GET  /api/healthcare/lab/results/<id>
PUT  /api/healthcare/lab/results/<id>
POST /api/healthcare/lab/results/<id>/verify
POST /api/healthcare/lab/results/<id>/release
POST /api/healthcare/lab/results/<id>/amend
POST /api/healthcare/lab/results/process-test

GET  /api/healthcare/lab/critical-values
POST /api/healthcare/lab/critical-values
GET  /api/healthcare/lab/critical-values/<id>
POST /api/healthcare/lab/critical-values/<id>/acknowledge

GET  /api/healthcare/lab/qc
POST /api/healthcare/lab/qc
GET  /api/healthcare/lab/qc/<id>
PUT  /api/healthcare/lab/qc/<id>
POST /api/healthcare/lab/qc/<id>/failure-action
POST /api/healthcare/lab/qc/proficiency-test

GET  /api/healthcare/lab/instruments
POST /api/healthcare/lab/instruments
GET  /api/healthcare/lab/instruments/<id>
PUT  /api/healthcare/lab/instruments/<id>/status
POST /api/healthcare/lab/instruments/<id>/calibrate
POST /api/healthcare/lab/instruments/<id>/message

GET  /api/healthcare/lab/referrals
POST /api/healthcare/lab/referrals
GET  /api/healthcare/lab/referrals/<id>
PUT  /api/healthcare/lab/referrals/<id>
POST /api/healthcare/lab/referrals/<id>/receive-result

GET  /api/healthcare/lab/reports/tat
GET  /api/healthcare/lab/reports/workload
GET  /api/healthcare/lab/reports/qc-summary
GET  /api/healthcare/lab/reports/critical-values
GET  /api/healthcare/lab/reports/rejection-rate

GET  /api/healthcare/lab/delta-check

© 2025 Datacraft — nyimbi@gmail.com
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from flask import Blueprint, jsonify, request

from .models import (
	AnalyserInterfaceCreate,
	AnalyserInterfaceUpdate,
	CriticalValueCreate,
	ExternalReferralCreate,
	ExternalReferralUpdate,
	LabOrderCreate,
	LabOrderUpdate,
	LabResultCreate,
	LabResultUpdate,
	LabTestCreate,
	LabTestUpdate,
	QCRunCreate,
	QCRunUpdate,
	ReferenceRangeCreate,
	ReferenceRangeUpdate,
	SpecimenCreate,
	SpecimenUpdate,
	SpecimenTrackRequest,
)
from .service import LaboratoryInformationService, PolicyViolationError

bp = Blueprint("healthcare_lab", __name__, url_prefix="/api/healthcare/lab")
_svc = LaboratoryInformationService()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(coro: Any) -> Any:
	"""Run an async coroutine from a synchronous Flask route."""
	return asyncio.run(coro)


def _ok(data: Any, status: int = 200) -> Any:
	return jsonify(data), status


def _err(msg: str, status: int = 400) -> Any:
	return jsonify({"error": msg, "status": status}), status


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", request.args.get("tenant_id", "default"))


def _actor() -> str:
	return request.headers.get("X-Actor-ID", request.args.get("actor_id", "unknown"))


def _body() -> dict[str, Any]:
	return request.get_json(silent=True) or {}


def _list_response(items: list[Any], serialise: bool = True) -> Any:
	if serialise:
		data = [i.model_dump(mode="json") if hasattr(i, "model_dump") else i for i in items]
	else:
		data = items
	return jsonify({"items": data, "count": len(data)})


def _handle(coro: Any, created: bool = False) -> Any:
	"""Run coro, return 201 on create, 200 otherwise.  Handle known exceptions."""
	try:
		result = _run(coro)
		code = 201 if created else 200
		if result is None:
			return _err("not_found", 404)
		if hasattr(result, "model_dump"):
			return jsonify(result.model_dump(mode="json")), code
		return jsonify(result), code
	except PolicyViolationError as e:
		return _err(str(e), 403)
	except KeyError as e:
		return _err(f"not_found: {e}", 404)
	except ValueError as e:
		return _err(str(e), 400)


# ── Contract / Dashboard ───────────────────────────────────────────────────────

@bp.get("/contract")
def get_contract():
	"""Return the capability contract for this tenant."""
	return _handle(_svc.describe(_tenant()))


@bp.get("/dashboard")
def dashboard():
	"""Return dashboard KPIs for the LIS home screen."""
	return _handle(_svc.dashboard_summary(_tenant()))


# ── Lab Test Catalogue ────────────────────────────────────────────────────────

@bp.get("/tests")
def list_tests():
	"""List all lab tests in the tenant catalogue."""
	category = request.args.get("category")
	active = request.args.get("active")
	active_bool: bool | None = None
	if active is not None:
		active_bool = active.lower() == "true"
	items = _run(_svc.list_tests(_tenant(), category=category, active=active_bool))
	return _list_response(items)


@bp.post("/tests")
def create_test():
	"""Add a new test to the catalogue."""
	data = _body()
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	return _handle(_svc.create_test(LabTestCreate(**data)), created=True)


@bp.get("/tests/<test_id>")
def get_test(test_id: str):
	"""Retrieve a single test catalogue entry."""
	return _handle(_svc.get_test(_tenant(), test_id))


@bp.put("/tests/<test_id>")
def update_test(test_id: str):
	"""Update a test catalogue entry."""
	return _handle(_svc.update_test(_tenant(), test_id, LabTestUpdate(**_body())))


@bp.delete("/tests/<test_id>")
def delete_test(test_id: str):
	"""Soft-delete a test catalogue entry."""
	return _handle(_svc.delete_test(_tenant(), test_id, _actor()))


# ── Lab Orders ────────────────────────────────────────────────────────────────

@bp.get("/orders")
def list_orders():
	"""List lab orders with optional filtering."""
	items = _run(_svc.list_orders(
		_tenant(),
		patient_id=request.args.get("patient_id"),
		status=request.args.get("status"),
		priority=request.args.get("priority"),
	))
	return _list_response(items)


@bp.post("/orders")
def create_order():
	"""Place a new lab order."""
	data = _body()
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	return _handle(_svc.create_order(LabOrderCreate(**data)), created=True)


@bp.get("/orders/<order_id>")
def get_order(order_id: str):
	"""Retrieve a single lab order."""
	return _handle(_svc.get_order(_tenant(), order_id))


@bp.put("/orders/<order_id>")
def update_order(order_id: str):
	"""Partially update a lab order."""
	return _handle(_svc.update_order(_tenant(), order_id, LabOrderUpdate(**_body())))


@bp.delete("/orders/<order_id>")
def delete_order(order_id: str):
	"""Soft-delete a lab order."""
	return _handle(_svc.cancel_order(_tenant(), order_id, _body().get("reason", "deleted")))


@bp.post("/orders/<order_id>/cancel")
def cancel_order(order_id: str):
	"""Cancel a lab order with a documented reason."""
	data = _body()
	return _handle(_svc.cancel_order(_tenant(), order_id, data.get("reason", "")))


@bp.post("/orders/<order_id>/receive")
def receive_order(order_id: str):
	"""Receive and acknowledge a lab order at the laboratory."""
	data = _body()
	return _handle(_svc.receive_lab_order(
		_tenant(), order_id,
		data.get("specimen_requirements", {}),
		data.get("received_by", _actor()),
	))


@bp.post("/orders/<order_id>/hold")
def hold_order(order_id: str):
	"""Place a lab order on hold."""
	data = _body()
	return _handle(_svc.hold_order(_tenant(), order_id, data.get("reason", "")))


@bp.post("/orders/<order_id>/unhold")
def unhold_order(order_id: str):
	"""Release a lab order from hold."""
	return _handle(_svc.unhold_order(_tenant(), order_id))


# ── Specimens ─────────────────────────────────────────────────────────────────

@bp.get("/specimens")
def list_specimens():
	"""List specimens with optional filtering."""
	items = _run(_svc.list_specimens(
		_tenant(),
		order_id=request.args.get("order_id"),
		status=request.args.get("status"),
	))
	return _list_response(items)


@bp.post("/specimens")
def collect_specimen():
	"""Record specimen collection."""
	data = _body()
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	return _handle(_svc.collect_specimen(SpecimenCreate(**data)), created=True)


@bp.get("/specimens/<specimen_id>")
def get_specimen(specimen_id: str):
	"""Retrieve a single specimen."""
	return _handle(_svc.get_specimen(_tenant(), specimen_id))


@bp.put("/specimens/<specimen_id>")
def update_specimen(specimen_id: str):
	"""Update specimen details."""
	return _handle(_svc.update_specimen(_tenant(), specimen_id, SpecimenUpdate(**_body())))


@bp.post("/specimens/<specimen_id>/reject")
def reject_specimen(specimen_id: str):
	"""Reject a specimen with a documented reason."""
	data = _body()
	return _handle(_svc.reject_specimen(_tenant(), specimen_id, data.get("rejection_reason", "")))


@bp.post("/specimens/<specimen_id>/receive")
def receive_specimen(specimen_id: str):
	"""Mark a specimen as received at the laboratory."""
	return _handle(_svc.receive_specimen(_tenant(), specimen_id))


@bp.post("/specimens/<specimen_id>/track")
def track_specimen(specimen_id: str):
	"""Append a custody event to the specimen's chain-of-custody."""
	data = _body()
	return _handle(_svc.track_specimen(
		_tenant(), specimen_id, SpecimenTrackRequest(**data)
	))


@bp.get("/specimens/<specimen_id>/custody")
def get_custody_chain(specimen_id: str):
	"""Retrieve the full chain-of-custody log for a specimen."""
	return _handle(_svc.get_custody_chain(_tenant(), specimen_id))


# ── Reference Ranges ──────────────────────────────────────────────────────────

@bp.get("/reference-ranges")
def list_reference_ranges():
	"""List reference ranges with optional test_code filter."""
	items = _run(_svc.list_reference_ranges(
		_tenant(),
		test_code=request.args.get("test_code"),
	))
	return _list_response(items)


@bp.post("/reference-ranges")
def create_reference_range():
	"""Create a new reference range."""
	data = _body()
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	return _handle(_svc.create_reference_range(ReferenceRangeCreate(**data)), created=True)


@bp.get("/reference-ranges/<rr_id>")
def get_reference_range(rr_id: str):
	"""Retrieve a single reference range."""
	return _handle(_svc.get_reference_range(_tenant(), rr_id))


@bp.put("/reference-ranges/<rr_id>")
def update_reference_range(rr_id: str):
	"""Update a reference range."""
	return _handle(_svc.update_reference_range(_tenant(), rr_id, ReferenceRangeUpdate(**_body())))


@bp.delete("/reference-ranges/<rr_id>")
def delete_reference_range(rr_id: str):
	"""Soft-delete a reference range."""
	return _handle(_svc.delete_reference_range(_tenant(), rr_id, _actor()))


# ── Results ───────────────────────────────────────────────────────────────────

@bp.get("/results")
def list_results():
	"""List results with optional filtering."""
	critical_only = request.args.get("critical_only", "false").lower() == "true"
	items = _run(_svc.list_results(
		_tenant(),
		order_id=request.args.get("order_id"),
		critical_only=critical_only,
	))
	return _list_response(items)


@bp.post("/results")
def enter_result():
	"""Enter a lab result."""
	data = _body()
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	return _handle(_svc.enter_result(LabResultCreate(**data)), created=True)


@bp.get("/results/<result_id>")
def get_result(result_id: str):
	"""Retrieve a single result."""
	return _handle(_svc.get_result(_tenant(), result_id))


@bp.put("/results/<result_id>")
def update_result(result_id: str):
	"""Update a result (pre-verification only)."""
	return _handle(_svc.update_result(_tenant(), result_id, LabResultUpdate(**_body())))


@bp.post("/results/<result_id>/verify")
def verify_result(result_id: str):
	"""Verify a result for release."""
	data = _body()
	return _handle(_svc.verify_result(
		_tenant(), result_id,
		data.get("verifier_id", _actor()),
		data.get("notification_sent", False),
	))


@bp.post("/results/<result_id>/release")
def release_result(result_id: str):
	"""Release a verified result to the ordering clinician."""
	data = _body()
	return _handle(_svc.release_result(
		_tenant(), result_id,
		data.get("released_by", _actor()),
		data.get("release_method", "portal"),
	))


@bp.post("/results/<result_id>/amend")
def amend_result(result_id: str):
	"""Amend a released result with a documented reason."""
	data = _body()
	return _handle(_svc.result_amend(
		_tenant(), result_id,
		data.get("amended_value"),
		data.get("amendment_reason", ""),
		data.get("amended_by", _actor()),
	))


@bp.post("/results/process-test")
def process_test():
	"""Process a test on an analyser and return a raw result with flags."""
	data = _body()
	return _handle(_svc.process_test(
		_tenant(),
		data.get("specimen_id", ""),
		data.get("analyser_id", ""),
		data.get("result_value"),
		data.get("result_unit", ""),
		data.get("reference_range", {}),
		data.get("performed_by", _actor()),
		data.get("test_code", ""),
	), created=True)


@bp.get("/delta-check")
def delta_check():
	"""Perform a delta check for a patient result."""
	return _handle(_svc.delta_check(
		_tenant(),
		request.args.get("patient_id", ""),
		request.args.get("test_code", ""),
		request.args.get("new_result"),
		float(request.args.get("threshold_pct", "25.0")),
	))


# ── Critical Values ───────────────────────────────────────────────────────────

@bp.get("/critical-values")
def list_critical_values():
	"""List critical value notifications."""
	unack_only = request.args.get("unacknowledged_only", "false").lower() == "true"
	items = _run(_svc.list_critical_values(_tenant(), unacknowledged_only=unack_only))
	return _list_response(items)


@bp.post("/critical-values")
def notify_critical_value():
	"""Record a critical value notification."""
	data = _body()
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	return _handle(_svc.create_critical_value(CriticalValueCreate(**data)), created=True)


@bp.get("/critical-values/<notif_id>")
def get_critical_value(notif_id: str):
	"""Retrieve a single critical value notification."""
	return _handle(_svc.get_critical_value(_tenant(), notif_id))


@bp.post("/critical-values/<notif_id>/acknowledge")
def acknowledge_critical_value(notif_id: str):
	"""Acknowledge a critical value notification."""
	data = _body()
	return _handle(_svc.acknowledge_critical_value(
		_tenant(), notif_id,
		data.get("acknowledged_by", _actor()),
	))


# ── QC ────────────────────────────────────────────────────────────────────────

@bp.get("/qc")
def list_qc_runs():
	"""List QC runs with optional instrument filter."""
	items = _run(_svc.list_qc_runs(
		_tenant(),
		instrument_id=request.args.get("instrument_id"),
	))
	return _list_response(items)


@bp.post("/qc")
def run_qc():
	"""Record a QC run and evaluate Westgard rules."""
	data = _body()
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	return _handle(_svc.run_qc(QCRunCreate(**data)), created=True)


@bp.get("/qc/<qc_id>")
def get_qc_run(qc_id: str):
	"""Retrieve a single QC run."""
	return _handle(_svc.get_qc_run(_tenant(), qc_id))


@bp.put("/qc/<qc_id>")
def update_qc_run(qc_id: str):
	"""Update QC run review status."""
	return _handle(_svc.update_qc_run(_tenant(), qc_id, QCRunUpdate(**_body())))


@bp.post("/qc/<qc_id>/failure-action")
def qc_failure_action(qc_id: str):
	"""Record corrective action following a QC failure."""
	data = _body()
	return _handle(_svc.qc_failure_action(
		_tenant(), qc_id,
		data.get("corrective_action", ""),
		data.get("performed_by", _actor()),
	))


@bp.post("/qc/proficiency-test")
def proficiency_test():
	"""Record external proficiency testing / EQA participation."""
	data = _body()
	return _handle(_svc.external_proficiency_testing(
		_tenant(),
		data.get("scheme", ""),
		data.get("result_submission", {}),
		data.get("score"),
		data.get("submitted_by", _actor()),
	), created=True)


@bp.post("/qc/material-run")
def qc_material_run():
	"""Run QC material against Westgard multi-rules via detailed analyser context."""
	data = _body()
	return _handle(_svc.qc_material_run(
		_tenant(),
		data.get("analyser_id", ""),
		data.get("qc_level", ""),
		float(data.get("measured_value", 0)),
		data.get("expected_range", {}),
		data.get("performed_by", _actor()),
		data.get("test_code", ""),
		data.get("lot_number", ""),
	), created=True)


# ── Instruments ───────────────────────────────────────────────────────────────

@bp.get("/instruments")
def list_instruments():
	"""List registered analyser interfaces."""
	items = _run(_svc.list_instruments(_tenant()))
	return _list_response(items)


@bp.post("/instruments")
def register_instrument():
	"""Register a new analyser interface."""
	data = _body()
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	return _handle(_svc.register_instrument(AnalyserInterfaceCreate(**data)), created=True)


@bp.get("/instruments/<instrument_id>")
def get_instrument(instrument_id: str):
	"""Retrieve a single analyser interface."""
	return _handle(_svc.get_instrument(_tenant(), instrument_id))


@bp.put("/instruments/<instrument_id>/status")
def update_instrument_status(instrument_id: str):
	"""Update the operational status of an analyser."""
	data = _body()
	return _handle(_svc.update_instrument_status(_tenant(), instrument_id, data.get("status", "")))


@bp.put("/instruments/<instrument_id>")
def update_instrument(instrument_id: str):
	"""Update analyser interface properties."""
	return _handle(_svc.update_instrument(_tenant(), instrument_id, AnalyserInterfaceUpdate(**_body())))


@bp.post("/instruments/<instrument_id>/calibrate")
def calibrate_instrument(instrument_id: str):
	"""Record an instrument calibration event."""
	data = _body()
	return _handle(_svc.record_calibration(
		_tenant(), instrument_id,
		data.get("calibrated_by", _actor()),
		data.get("notes"),
		data.get("pass_fail", True),
	), created=True)


@bp.post("/instruments/<instrument_id>/message")
def ingest_instrument_message(instrument_id: str):
	"""Ingest a raw analyser interface message (HL7/ASTM)."""
	data = _body()
	return _handle(_svc.interface_analyser(
		_tenant(), instrument_id,
		data.get("protocol", "hl7_v2"),
		data.get("message_type", ""),
		data.get("raw_payload", ""),
	), created=True)


# ── External Referrals ────────────────────────────────────────────────────────

@bp.get("/referrals")
def list_referrals():
	"""List external referrals."""
	items = _run(_svc.list_referrals(
		_tenant(),
		status=request.args.get("status"),
	))
	return _list_response(items)


@bp.post("/referrals")
def create_referral():
	"""Create an external referral."""
	data = _body()
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	return _handle(_svc.create_referral(ExternalReferralCreate(**data)), created=True)


@bp.get("/referrals/<referral_id>")
def get_referral(referral_id: str):
	"""Retrieve a single external referral."""
	return _handle(_svc.get_referral(_tenant(), referral_id))


@bp.put("/referrals/<referral_id>")
def update_referral(referral_id: str):
	"""Update external referral details."""
	return _handle(_svc.update_referral(_tenant(), referral_id, ExternalReferralUpdate(**_body())))


@bp.post("/referrals/<referral_id>/receive-result")
def receive_external_result(referral_id: str):
	"""Record a result received from an external laboratory."""
	data = _body()
	return _handle(_svc.receive_external_result(
		_tenant(), referral_id,
		data.get("result_data", {}),
		data.get("verified_by", _actor()),
	))


# ── Reports ───────────────────────────────────────────────────────────────────

@bp.get("/reports/tat")
def report_tat():
	"""Turnaround time analysis report."""
	return _handle(_svc.tat_monitoring(
		_tenant(),
		request.args.get("period", "today"),
		by_analyser=request.args.get("by_analyser", "true").lower() == "true",
	))


@bp.get("/reports/workload")
def report_workload():
	"""Laboratory workload summary report."""
	return _handle(_svc.lab_workload_report(
		_tenant(),
		request.args.get("period", "today"),
		by_analyser=request.args.get("by_analyser", "true").lower() == "true",
	))


@bp.get("/reports/qc-summary")
def report_qc_summary():
	"""QC pass/fail summary per instrument."""
	return _handle(_svc.generate_qc_summary(_tenant()))


@bp.get("/reports/critical-values")
def report_critical_values():
	"""Critical value notification compliance report."""
	return _handle(_svc.generate_critical_value_report(
		_tenant(),
		date_from=request.args.get("date_from"),
		date_to=request.args.get("date_to"),
	))


@bp.get("/reports/rejection-rate")
def report_rejection_rate():
	"""Specimen rejection rate report by reason."""
	return _handle(_svc.generate_rejection_report(_tenant()))


@bp.get("/reports/lab-report/<order_id>")
def generate_lab_report(order_id: str):
	"""Generate a full patient lab report for an order."""
	return _handle(_svc.generate_lab_report(
		_tenant(), order_id,
		fmt=request.args.get("format", "json"),
	))
