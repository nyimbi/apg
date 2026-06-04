"""View model builders for APG Electronic Medical Records screens."""

from __future__ import annotations

import asyncio
from typing import Any

from .capability_contract import get_capability_contract
from .service import ElectronicMedicalRecordsService


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def dashboard_view_model(service: ElectronicMedicalRecordsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = _run(service.dashboard_summary(tenant_id))
	return {"title": "Electronic Medical Records", "tenant_id": tenant_id, "summary": summary, "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def patient_chart_view_model(service: ElectronicMedicalRecordsService, tenant_id: str, patient_id: str) -> dict[str, Any]:
	notes = _run(service.list_notes(tenant_id, patient_id=patient_id))
	problems = _run(service.list_problems(tenant_id, patient_id))
	medications = _run(service.list_medications(tenant_id, patient_id))
	allergies = _run(service.list_allergies(tenant_id, patient_id))
	vitals = _run(service.list_vitals(tenant_id, patient_id))
	encounters = _run(service.list_encounters(tenant_id, patient_id=patient_id))
	return {
		"title": "Patient Chart",
		"tenant_id": tenant_id,
		"patient_id": patient_id,
		"notes": [n.model_dump() for n in notes],
		"problems": [p.model_dump() for p in problems],
		"medications": [m.model_dump() for m in medications],
		"allergies": [a.model_dump() for a in allergies],
		"vitals": [v.model_dump() for v in vitals],
		"encounters": [e.model_dump() for e in encounters],
	}


def note_list_view_model(service: ElectronicMedicalRecordsService, tenant_id: str, patient_id: str | None = None, note_type: str | None = None) -> dict[str, Any]:
	notes = _run(service.list_notes(tenant_id, patient_id=patient_id, note_type=note_type))
	return {"title": "Clinical Notes", "tenant_id": tenant_id, "notes": [n.model_dump() for n in notes], "filter": {"patient_id": patient_id, "note_type": note_type}}


def note_detail_view_model(service: ElectronicMedicalRecordsService, tenant_id: str, note_id: str) -> dict[str, Any]:
	note = _run(service.get_note(tenant_id, note_id))
	if note is None:
		return {"error": "note_not_found", "note_id": note_id}
	return {"title": "Clinical Note", "tenant_id": tenant_id, "note": note.model_dump()}


def medication_list_view_model(service: ElectronicMedicalRecordsService, tenant_id: str, patient_id: str) -> dict[str, Any]:
	meds = _run(service.list_medications(tenant_id, patient_id))
	active = [m for m in meds if m.status == "active"]
	return {"title": "Medications", "tenant_id": tenant_id, "patient_id": patient_id, "medications": [m.model_dump() for m in meds], "active_count": len(active)}


def allergy_list_view_model(service: ElectronicMedicalRecordsService, tenant_id: str, patient_id: str) -> dict[str, Any]:
	allergies = _run(service.list_allergies(tenant_id, patient_id))
	critical = [a for a in allergies if a.severity in ("severe", "life_threatening")]
	return {"title": "Allergies", "tenant_id": tenant_id, "patient_id": patient_id, "allergies": [a.model_dump() for a in allergies], "critical_count": len(critical)}
