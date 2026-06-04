"""View model builders for APG Telemedicine screens."""

from __future__ import annotations
import asyncio
from typing import Any
from .capability_contract import get_capability_contract
from .service import TelemedicineService


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def dashboard_view_model(service: TelemedicineService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Telemedicine", "tenant_id": tenant_id, "summary": _run(service.dashboard_summary(tenant_id)), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def consultation_schedule_view_model(service: TelemedicineService, tenant_id: str, patient_id: str | None = None) -> dict[str, Any]:
	consults = _run(service.list_consultations(tenant_id, patient_id=patient_id))
	upcoming = [c for c in consults if c.status == "scheduled"]
	return {"title": "Consultation Schedule", "tenant_id": tenant_id, "consultations": [c.model_dump() for c in consults], "upcoming_count": len(upcoming)}


def session_list_view_model(service: TelemedicineService, tenant_id: str, patient_id: str | None = None) -> dict[str, Any]:
	sessions = _run(service.list_sessions(tenant_id, patient_id=patient_id))
	active = [s for s in sessions if s.status == "in_progress"]
	return {"title": "Sessions", "tenant_id": tenant_id, "sessions": [s.model_dump() for s in sessions], "active_count": len(active)}


def monitoring_view_model(service: TelemedicineService, tenant_id: str, patient_id: str | None = None) -> dict[str, Any]:
	enrollments = _run(service.list_monitoring(tenant_id, patient_id=patient_id))
	return {"title": "Remote Monitoring", "tenant_id": tenant_id, "enrollments": [e.model_dump() for e in enrollments], "active_count": sum(1 for e in enrollments if e.status == "active")}


def billing_view_model(service: TelemedicineService, tenant_id: str) -> dict[str, Any]:
	bills = _run(service.list_billing(tenant_id))
	pending = [b for b in bills if b.status == "pending"]
	return {"title": "Telehealth Billing", "tenant_id": tenant_id, "records": [b.model_dump() for b in bills], "pending_count": len(pending)}
