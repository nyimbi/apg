"""View model builders for APG Healthcare Regulatory screens."""

from __future__ import annotations
import asyncio
from typing import Any
from .capability_contract import get_capability_contract
from .service import HealthcareRegulatoryService


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def dashboard_view_model(service: HealthcareRegulatoryService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Healthcare Regulatory", "tenant_id": tenant_id, "summary": _run(service.dashboard_summary(tenant_id)), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def license_view_model(service: HealthcareRegulatoryService, tenant_id: str) -> dict[str, Any]:
	licenses = _run(service.list_licenses(tenant_id))
	expiring = [l for l in licenses if l.days_to_expiry <= 90]
	return {"title": "Licenses", "tenant_id": tenant_id, "licenses": [l.model_dump() for l in licenses], "expiring_count": len(expiring)}


def incident_view_model(service: HealthcareRegulatoryService, tenant_id: str, status: str | None = None) -> dict[str, Any]:
	incidents = _run(service.list_incidents(tenant_id, status=status))
	sentinels = [i for i in incidents if i.incident_type == "sentinel_event"]
	return {"title": "Incidents", "tenant_id": tenant_id, "incidents": [i.model_dump() for i in incidents], "sentinel_count": len(sentinels)}


def submission_view_model(service: HealthcareRegulatoryService, tenant_id: str) -> dict[str, Any]:
	submissions = _run(service.list_submissions(tenant_id))
	pending = [s for s in submissions if s.status == "submitted"]
	return {"title": "Regulatory Submissions", "tenant_id": tenant_id, "submissions": [s.model_dump() for s in submissions], "pending_count": len(pending)}


def corrective_action_view_model(service: HealthcareRegulatoryService, tenant_id: str) -> dict[str, Any]:
	cas = _run(service.list_corrective_actions(tenant_id))
	open_cas = [ca for ca in cas if ca.status == "open"]
	return {"title": "Corrective Actions", "tenant_id": tenant_id, "corrective_actions": [ca.model_dump() for ca in cas], "open_count": len(open_cas)}
