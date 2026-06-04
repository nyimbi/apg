"""Flask Blueprint UI views for APG Electronic Medical Records.

Each view function returns a dict suitable for Jinja2 template rendering.
The Blueprint registers HTML routes under /healthcare-emr/.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any

from flask import Blueprint, render_template, request, redirect, url_for, flash

from .capability_contract import get_capability_contract
from .service import EMRService, PolicyViolationError, DrugSafetyError

ui = Blueprint(
	"healthcare_emr_ui",
	__name__,
	url_prefix="/healthcare-emr",
	template_folder="templates",
	static_folder="static",
)


# ── async bridge ──────────────────────────────────────────────────────────────

def _run(coro: Any) -> Any:
	return asyncio.run(coro)


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _actor() -> str:
	return request.headers.get("X-Actor-ID", "ui_user")


def _svc() -> EMRService:
	return EMRService(tenant_id=_tenant(), actor_id=_actor())


# ── view-model builders (pure functions, usable outside Flask context) ────────

def dashboard_view_model(
	service: EMRService,
	tenant_id: str = "default",
) -> dict[str, Any]:
	"""KPI dashboard: encounter counts, open notes, critical labs, medication alerts."""
	contract = get_capability_contract(tenant_id)
	summary = _run(service.dashboard_summary(tenant_id))
	try:
		critical_labs = _run(service.list_unnotified_critical_labs(tenant_id))
	except Exception:
		critical_labs = []
	return {
		"title": "Electronic Medical Records — Dashboard",
		"tenant_id": tenant_id,
		"summary": summary,
		"critical_labs_count": len(critical_labs),
		"critical_labs": [r.model_dump(mode="json") for r in critical_labs[:5]],
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
		"generated_at": datetime.utcnow().isoformat(),
	}


def patient_list_view_model(
	service: EMRService,
	tenant_id: str,
	search: str | None = None,
	status: str | None = None,
	page: int = 1,
	page_size: int = 25,
) -> dict[str, Any]:
	"""Paginated patient list with optional search and status filter."""
	try:
		patients = _run(service.list_patients(tenant_id, status=status, search=search))
	except Exception:
		patients = []
	start = (page - 1) * page_size
	slice_ = patients[start : start + page_size]
	total = len(patients)
	return {
		"title": "Patients",
		"tenant_id": tenant_id,
		"patients": [p.model_dump(mode="json") for p in slice_],
		"total": total,
		"page": page,
		"page_size": page_size,
		"pages": max(1, (total + page_size - 1) // page_size),
		"search": search,
		"status_filter": status,
	}


def patient_chart_view_model(
	service: EMRService,
	tenant_id: str,
	patient_id: str,
) -> dict[str, Any]:
	"""Full patient chart: demographics + all clinical data aggregated."""
	patient = _run(service.get_patient(tenant_id, patient_id))
	notes = _run(service.list_notes(tenant_id, patient_id=patient_id))
	problems = _run(service.list_problems(tenant_id, patient_id))
	medications = _run(service.list_medications(tenant_id, patient_id))
	allergies = _run(service.list_allergies(tenant_id, patient_id))
	vitals = _run(service.list_vitals(tenant_id, patient_id))
	encounters = _run(service.list_encounters(tenant_id, patient_id=patient_id))
	try:
		cds_alerts = _run(service.clinical_decision_support(patient_id))
	except Exception:
		cds_alerts = []

	# most recent vital per type
	latest_vitals: dict[str, Any] = {}
	for v in vitals:
		vtype = str(v.vital_type)
		if vtype not in latest_vitals:
			latest_vitals[vtype] = {
				"value": v.value,
				"value2": v.value2,
				"unit": v.unit,
				"recorded_at": v.recorded_at.isoformat(),
			}

	critical_allergies = [a for a in allergies if a.severity in ("severe", "life_threatening")]

	return {
		"title": f"Patient Chart — {patient.name.family if patient else patient_id}",
		"tenant_id": tenant_id,
		"patient_id": patient_id,
		"patient": patient.model_dump(mode="json") if patient else None,
		"notes": [n.model_dump(mode="json") for n in notes[:10]],
		"problems": [p.model_dump(mode="json") for p in problems],
		"active_problems": [p.model_dump(mode="json") for p in problems if p.status == "active"],
		"medications": [m.model_dump(mode="json") for m in medications],
		"active_medications": [m.model_dump(mode="json") for m in medications if m.status == "active"],
		"allergies": [a.model_dump(mode="json") for a in allergies if a.status == "active"],
		"critical_allergies": [a.model_dump(mode="json") for a in critical_allergies],
		"latest_vitals": latest_vitals,
		"encounters": [e.model_dump(mode="json") for e in encounters[:5]],
		"open_encounters": [e.model_dump(mode="json") for e in encounters if e.status == "in_progress"],
		"cds_alerts": [a.model_dump() for a in cds_alerts],
		"alert_count": len(cds_alerts),
	}


def note_list_view_model(
	service: EMRService,
	tenant_id: str,
	patient_id: str | None = None,
	note_type: str | None = None,
	page: int = 1,
	page_size: int = 20,
) -> dict[str, Any]:
	notes = _run(service.list_notes(tenant_id, patient_id=patient_id, note_type=note_type))
	start = (page - 1) * page_size
	slice_ = notes[start : start + page_size]
	return {
		"title": "Clinical Notes",
		"tenant_id": tenant_id,
		"notes": [n.model_dump(mode="json") for n in slice_],
		"total": len(notes),
		"page": page,
		"page_size": page_size,
		"filter": {"patient_id": patient_id, "note_type": note_type},
	}


def note_detail_view_model(
	service: EMRService,
	tenant_id: str,
	note_id: str,
) -> dict[str, Any]:
	note = _run(service.get_note(tenant_id, note_id))
	if note is None:
		return {"error": "note_not_found", "note_id": note_id}
	return {
		"title": f"Clinical Note — {note.note_type}",
		"tenant_id": tenant_id,
		"note": note.model_dump(mode="json"),
	}


def medication_list_view_model(
	service: EMRService,
	tenant_id: str,
	patient_id: str,
) -> dict[str, Any]:
	meds = _run(service.list_medications(tenant_id, patient_id))
	active = [m for m in meds if m.status == "active"]
	return {
		"title": "Medications",
		"tenant_id": tenant_id,
		"patient_id": patient_id,
		"medications": [m.model_dump(mode="json") for m in meds],
		"active_count": len(active),
		"total_count": len(meds),
	}


def allergy_list_view_model(
	service: EMRService,
	tenant_id: str,
	patient_id: str,
) -> dict[str, Any]:
	allergies = _run(service.list_allergies(tenant_id, patient_id))
	critical = [a for a in allergies if a.severity in ("severe", "life_threatening")]
	return {
		"title": "Allergies & Intolerances",
		"tenant_id": tenant_id,
		"patient_id": patient_id,
		"allergies": [a.model_dump(mode="json") for a in allergies],
		"critical_count": len(critical),
		"total_count": len(allergies),
	}


def vitals_view_model(
	service: EMRService,
	tenant_id: str,
	patient_id: str,
) -> dict[str, Any]:
	vitals = _run(service.list_vitals(tenant_id, patient_id))
	by_type: dict[str, list[dict[str, Any]]] = {}
	for v in vitals:
		vtype = str(v.vital_type)
		by_type.setdefault(vtype, []).append({
			"value": v.value,
			"value2": v.value2,
			"unit": v.unit,
			"recorded_at": v.recorded_at.isoformat(),
		})
	return {
		"title": "Vital Signs",
		"tenant_id": tenant_id,
		"patient_id": patient_id,
		"vitals": [v.model_dump(mode="json") for v in vitals],
		"vitals_by_type": by_type,
		"total_count": len(vitals),
	}


def encounter_list_view_model(
	service: EMRService,
	tenant_id: str,
	patient_id: str,
) -> dict[str, Any]:
	encs = _run(service.list_encounters(tenant_id, patient_id=patient_id))
	open_ = [e for e in encs if e.status == "in_progress"]
	return {
		"title": "Encounters",
		"tenant_id": tenant_id,
		"patient_id": patient_id,
		"encounters": [e.model_dump(mode="json") for e in encs],
		"open_count": len(open_),
		"total_count": len(encs),
	}


def lab_orders_view_model(
	service: EMRService,
	tenant_id: str,
	patient_id: str,
) -> dict[str, Any]:
	orders = _run(service.list_lab_orders(tenant_id, patient_id))
	results = _run(service.list_lab_results(tenant_id, patient_id))
	critical = [r for r in results if r.flag in ("critical_low", "critical_high")]
	return {
		"title": "Lab Orders & Results",
		"tenant_id": tenant_id,
		"patient_id": patient_id,
		"orders": [o.model_dump(mode="json") for o in orders],
		"results": [r.model_dump(mode="json") for r in results],
		"critical_results": [r.model_dump(mode="json") for r in critical],
		"critical_count": len(critical),
	}


def prescription_list_view_model(
	service: EMRService,
	patient_id: str,
) -> dict[str, Any]:
	rxs = _run(service.generate_prescription_list(patient_id))
	active = [r for r in rxs if r.get("status") == "active"]
	controlled = [r for r in rxs if r.get("is_controlled") or r.get("drug_class") in (
		"opioid", "benzodiazepine", "stimulant"
	)]
	return {
		"title": "Prescriptions",
		"patient_id": patient_id,
		"prescriptions": rxs,
		"active_count": len(active),
		"controlled_count": len(controlled),
		"total_count": len(rxs),
	}


def care_plan_view_model(
	service: EMRService,
	tenant_id: str,
	patient_id: str,
) -> dict[str, Any]:
	plans = _run(service.list_care_plans(tenant_id, patient_id))
	active = [p for p in plans if p.status == "active"]
	return {
		"title": "Care Plans",
		"tenant_id": tenant_id,
		"patient_id": patient_id,
		"plans": [p.model_dump(mode="json") for p in plans],
		"active_count": len(active),
	}


def referral_list_view_model(
	service: EMRService,
	tenant_id: str,
	patient_id: str,
) -> dict[str, Any]:
	refs = _run(service.list_referrals(tenant_id, patient_id))
	pending = [r for r in refs if r.get("status") == "active"]
	return {
		"title": "Referrals",
		"tenant_id": tenant_id,
		"patient_id": patient_id,
		"referrals": refs,
		"pending_count": len(pending),
	}


def fhir_export_view_model(
	service: EMRService,
	tenant_id: str,
) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"title": "FHIR R4 Export",
		"tenant_id": tenant_id,
		"supported_resources": contract["configuration"]["fhir"]["supported_resources"],
		"fhir_version": contract["configuration"]["fhir"]["version"],
	}


def cds_view_model(
	service: EMRService,
	patient_id: str,
) -> dict[str, Any]:
	"""Clinical decision support summary for a patient."""
	try:
		alerts = _run(service.clinical_decision_support(patient_id))
		reminders = _run(service.clinical_reminder_check(patient_id))
		chads2 = _run(service.CHADS2_VASc_score(patient_id))
	except Exception:
		alerts, reminders, chads2 = [], [], {}
	critical_alerts = [a for a in alerts if a.severity == "critical"]
	return {
		"title": "Clinical Decision Support",
		"patient_id": patient_id,
		"alerts": [a.model_dump() for a in alerts],
		"critical_count": len(critical_alerts),
		"reminders": reminders,
		"reminder_count": len(reminders),
		"chads2_vasc": chads2,
	}


# ── Flask route handlers ──────────────────────────────────────────────────────

@ui.get("/dashboard")
def dashboard():
	svc = _svc()
	vm = dashboard_view_model(svc, _tenant())
	try:
		return render_template("healthcare/emr/dashboard.html", **vm)
	except Exception:
		return vm  # dev fallback: return dict


@ui.get("/patients")
def patient_list():
	svc = _svc()
	search = request.args.get("search")
	status = request.args.get("status")
	page = int(request.args.get("page", 1))
	vm = patient_list_view_model(svc, _tenant(), search=search, status=status, page=page)
	try:
		return render_template("healthcare/emr/patient_list.html", **vm)
	except Exception:
		return vm


@ui.get("/patients/<patient_id>")
def patient_chart(patient_id: str):
	svc = _svc()
	vm = patient_chart_view_model(svc, _tenant(), patient_id)
	try:
		return render_template("healthcare/emr/patient_chart.html", **vm)
	except Exception:
		return vm


@ui.get("/notes")
def note_list():
	svc = _svc()
	patient_id = request.args.get("patient_id")
	note_type = request.args.get("note_type")
	page = int(request.args.get("page", 1))
	vm = note_list_view_model(svc, _tenant(), patient_id=patient_id, note_type=note_type, page=page)
	try:
		return render_template("healthcare/emr/note_list.html", **vm)
	except Exception:
		return vm


@ui.get("/notes/<note_id>")
def note_detail(note_id: str):
	svc = _svc()
	vm = note_detail_view_model(svc, _tenant(), note_id)
	if "error" in vm:
		try:
			return render_template("healthcare/emr/404.html", **vm), 404
		except Exception:
			return vm, 404
	try:
		return render_template("healthcare/emr/note_detail.html", **vm)
	except Exception:
		return vm


@ui.get("/patients/<patient_id>/medications")
def medications(patient_id: str):
	svc = _svc()
	vm = medication_list_view_model(svc, _tenant(), patient_id)
	try:
		return render_template("healthcare/emr/medications.html", **vm)
	except Exception:
		return vm


@ui.get("/patients/<patient_id>/allergies")
def allergies(patient_id: str):
	svc = _svc()
	vm = allergy_list_view_model(svc, _tenant(), patient_id)
	try:
		return render_template("healthcare/emr/allergies.html", **vm)
	except Exception:
		return vm


@ui.get("/patients/<patient_id>/vitals")
def vitals(patient_id: str):
	svc = _svc()
	vm = vitals_view_model(svc, _tenant(), patient_id)
	try:
		return render_template("healthcare/emr/vitals.html", **vm)
	except Exception:
		return vm


@ui.get("/patients/<patient_id>/encounters")
def encounters(patient_id: str):
	svc = _svc()
	vm = encounter_list_view_model(svc, _tenant(), patient_id)
	try:
		return render_template("healthcare/emr/encounters.html", **vm)
	except Exception:
		return vm


@ui.get("/patients/<patient_id>/labs")
def labs(patient_id: str):
	svc = _svc()
	vm = lab_orders_view_model(svc, _tenant(), patient_id)
	try:
		return render_template("healthcare/emr/labs.html", **vm)
	except Exception:
		return vm


@ui.get("/patients/<patient_id>/care-plans")
def care_plans(patient_id: str):
	svc = _svc()
	vm = care_plan_view_model(svc, _tenant(), patient_id)
	try:
		return render_template("healthcare/emr/care_plans.html", **vm)
	except Exception:
		return vm


@ui.get("/patients/<patient_id>/referrals")
def referrals(patient_id: str):
	svc = _svc()
	vm = referral_list_view_model(svc, _tenant(), patient_id)
	try:
		return render_template("healthcare/emr/referrals.html", **vm)
	except Exception:
		return vm


@ui.get("/fhir-export")
def fhir_export():
	svc = _svc()
	vm = fhir_export_view_model(svc, _tenant())
	try:
		return render_template("healthcare/emr/fhir_export.html", **vm)
	except Exception:
		return vm


@ui.get("/patients/<patient_id>/cds")
def cds(patient_id: str):
	svc = _svc()
	vm = cds_view_model(svc, patient_id)
	try:
		return render_template("healthcare/emr/cds.html", **vm)
	except Exception:
		return vm
