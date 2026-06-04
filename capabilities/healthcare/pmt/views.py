"""View model builders for APG Patient Management screens."""

from __future__ import annotations
import asyncio
from typing import Any
from .capability_contract import get_capability_contract
from .service import PatientManagementService


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def dashboard_view_model(service: PatientManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Patient Management", "tenant_id": tenant_id, "summary": _run(service.dashboard_summary(tenant_id)), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def patient_list_view_model(service: PatientManagementService, tenant_id: str, last_name: str | None = None, mrn: str | None = None) -> dict[str, Any]:
	patients = _run(service.search_patients(tenant_id, last_name=last_name, mrn=mrn))
	return {"title": "Patients", "tenant_id": tenant_id, "patients": [p.model_dump() for p in patients], "count": len(patients), "filter": {"last_name": last_name, "mrn": mrn}}


def bed_board_view_model(service: PatientManagementService, tenant_id: str, unit_id: str | None = None) -> dict[str, Any]:
	beds = _run(service.list_beds(tenant_id, unit_id=unit_id))
	available = sum(1 for b in beds if b.status == "available")
	occupied = sum(1 for b in beds if b.status == "occupied")
	return {"title": "Bed Board", "tenant_id": tenant_id, "beds": [b.model_dump() for b in beds], "available_count": available, "occupied_count": occupied, "occupancy_rate": round(occupied / len(beds) * 100, 1) if beds else 0}


def appointment_calendar_view_model(service: PatientManagementService, tenant_id: str, provider_id: str | None = None) -> dict[str, Any]:
	appts = _run(service.list_appointments(tenant_id, provider_id=provider_id, status="scheduled"))
	return {"title": "Appointment Calendar", "tenant_id": tenant_id, "appointments": [a.model_dump() for a in appts], "count": len(appts)}
